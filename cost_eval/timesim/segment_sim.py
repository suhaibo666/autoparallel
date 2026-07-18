# cost_eval/timesim/segment_sim.py
"""段内多流离散事件仿真（M9-L1，spec §5）。

流推进语义（§5.1）：
  H（host 发射）：全部 op 按段内序串行 `h_end_i = h_start_i + t_host_i`——PyNative 异步下发，
    **默认不等 device**（同步点白名单 v1 = pass 边界，段内无）。
  D（device 计算）：FIFO；`d_start = max(该 op 的 h_end, D 前 op 结束, 跨流 deps 完成)`。
  C_*（通信按 group 轴分道）：同道 FIFO + 跨流 deps；不同轴通信域并发（多流带宽争抢不建，
    §7.2-2）。
host_only op 的"完成时刻"= 其 h_end（视图只有发射成本）。
未知 dep（跨段/外部，pass_builder 契约）→ 段首已满足=0（run-ahead 跨 pass 截断，§7.2-4）。

三态归因（§5.3，涌现非拍定；判据次序先查发射、再查依赖）：对 D 流 [0, makespan] 逐段积分：
  D 忙 → 按 OpCost.bound 劈 t_compute / t_membound（comm/host 类不上 D 流）；
  D 闲且下一 op 未被 H 发射 → t_host_gap；
  D 闲且已发射但依赖未完 → t_exposed_comm[阻塞 dep 的通信轴]（取完成最晚的 comm dep；
    无 comm dep 可归（纯 host_only 依赖）→ 归 host_gap，保守恒）。
  设备之后的尾段 [d_last, makespan]：逐通信道busy区间归 exposed（重叠取完成更晚者），
    其余（host 尾巴）归 host_gap——Σ三态 == makespan 恒等（L0①，测试守恒）。"""
from __future__ import annotations

from dataclasses import dataclass

from .ir import TimedSegment, STREAM_DEVICE, STREAM_HOST_ONLY
from .pass_builder import layer_of


@dataclass(frozen=True)
class SegmentTime:
    seg_id: str
    duration_us: float
    t_compute: float
    t_membound: float
    t_host_gap: float
    t_exposed_comm: dict
    host_len_us: float
    per_layer: dict
    top_contributors: list


def simulate_segment(seg: TimedSegment, costs: dict) -> SegmentTime:
    h_clock = 0.0
    end: dict[str, float] = {}
    lane_clock: dict[str, float] = {}
    dev_rows = []                       # (start, fin, op, h_end, blocking_comm_axis|None)
    comm_rows = []                      # (start, fin, axis)
    op_by_id = {o.op_id: o for o in seg.ops}

    for op in seg.ops:
        c = costs[op.op_id]
        h_clock += c.t_host_us
        if op.stream == STREAM_HOST_ONLY:
            end[op.op_id] = h_clock
            continue
        dep_end = 0.0
        blocking = None
        for d in op.deps:
            e = end.get(d, 0.0)          # 未知 dep=段首已满足（模块 docstring）
            if e > dep_end:
                dep_end = e
                dep_op = op_by_id.get(d)
                blocking = (dep_op.comm.group_axis
                            if dep_op is not None and dep_op.comm is not None else None)
        dur = c.t_comm_us if op.op_type == "CommOp" else c.t_dev_us
        start = max(lane_clock.get(op.stream, 0.0), h_clock, dep_end)
        fin = start + dur
        lane_clock[op.stream] = fin
        end[op.op_id] = fin
        if op.stream == STREAM_DEVICE:
            dev_rows.append((start, fin, op, h_clock, blocking))
        else:
            comm_rows.append((start, fin, op.comm.group_axis, op.op_id))

    makespan = max(list(lane_clock.values()) + [h_clock] + [0.0])

    t_compute = t_membound = host_gap = 0.0
    exposed: dict[str, float] = {}
    per_layer: dict[str, float] = {}
    cursor = 0.0
    for start, fin, op, h_end, blocking in dev_rows:
        gap = start - cursor
        if gap > 0:
            hg = min(max(h_end - cursor, 0.0), gap)         # 先查发射
            host_gap += hg
            rest = gap - hg
            if rest > 0:                                     # 再查依赖
                if blocking is not None:
                    exposed[blocking] = exposed.get(blocking, 0.0) + rest
                else:
                    host_gap += rest                         # 无 comm 可归（保守恒）
        c = costs[op.op_id]
        if c.bound == "memory":
            t_membound += fin - start
        else:
            t_compute += fin - start
        per_layer[layer_of(op.op_id)] = per_layer.get(layer_of(op.op_id), 0.0) + (fin - start)
        cursor = fin

    # 设备后尾段：通信道覆盖归 exposed（重叠取 fin 更晚者），未覆盖归 host_gap
    t = cursor
    tail = sorted((r for r in comm_rows if r[1] > cursor), key=lambda r: r[1])
    for start, fin, axis, _ in tail:
        lo = max(start, t)
        if fin > lo:
            exposed[axis] = exposed.get(axis, 0.0) + (fin - lo)
            t = fin
    if makespan > t:
        host_gap += makespan - t

    dur_of = {op.op_id: fin - start for start, fin, op, _, _ in dev_rows}
    dur_of.update({op_id: fin - start for start, fin, _, op_id in comm_rows})
    top = sorted(dur_of, key=dur_of.get, reverse=True)[:10]

    return SegmentTime(seg.seg_id, makespan, t_compute, t_membound, host_gap,
                       exposed, h_clock, per_layer, top)
