# cost_eval/timesim/frame_comm.py
"""frame_comm：框架层通信注入（spec §3.3c 第二类——不在 layer construct 源码里的通信，
与 producer.py 的 TP/SP 通信注入互补：那边管 layer construct 源码内显式通信，这边管 FSDP/dp、
EP dispatch/combine、CP 三条并行轴的注入，均为 fwd）。

FSDP：per-segment 权重 all-gather 注入段头（fwd 预取语义——按 spec §5.5，第 i+1 段的权重 gather
挂在第 i 段序列头部，与上段计算的重叠是 pipeline_sim（T1）里"位置"的自然结果，本模块不建模
重叠本身）；载荷 = 段内 GEMM 族权重（in_shapes[1]）字节和——len(in_shapes)<2 的 GEMM 跳过
（module=="" 的透传 matmul 可能仅 1 入，arity 现实见 ir.py in_shapes 字段注释）。dp_shard<=1
（未切片）→ 恒等，不注入。

EP：GroupedMatMul 区两侧包一对 dispatch/combine all-to-all（balanced 路由口径——继承内存侧同一
假设，v1 不建 unbalanced 容量溢出）；dispatch 载荷 = 该区首个 GroupedMatMul 的输入激活字节，
combine 载荷 = 该区末个 GroupedMatMul 的输出字节。ep<=1 或段内无 GroupedMatMul → 恒等。

CP：colossal（环状 p2p）→ 每条 FlashAttention 前插一条 kv p2p（载荷=k+v 字节；环的跳数系数归
op_cost/T1，本模块只管"有一跳 p2p 通信、且在 FA 前"这一结构事实）；ulysses → FlashAttention
前后各插一条 all_to_all（seq-parallel↔head-parallel 切换）。method 不在
{"colossal","ulysses"} → ValueError（fail-loud，不猜语义）。cp<=1 → 恒等。

三者共同点：注入 op 的 deps=()——本模块只管"通信该出现在段内什么位置"（段头 / 区两侧 / FA 两侧）
这一结构性事实，位置本身就是 spec §5.5 讲的 overlap 语义来源；具体的跨流依赖排程留给
pipeline_sim（T1）按段序展开，段首/边界语义下不在此处杜撰。bwd 侧的对偶通信（gather↔
reduce-scatter 等）归 bwd_rules（Task 11），本模块只做 fwd。
"""
from __future__ import annotations

from .ir import TimedOp, TimedSegment, CommSpec, tensor_bytes, COMM_STREAM

_GEMM_FAMILY = ("MatMul", "GroupedMatMul")


def inject_fsdp(seg: TimedSegment, dp_shard: int) -> TimedSegment:
    """dp_shard<=1 → 恒等；否则在段头插入本段权重的 all_gather（fwd 预取语义）。"""
    if dp_shard <= 1:
        return seg
    volume = sum(
        tensor_bytes(op.in_shapes[1], op.dtype)
        for op in seg.ops
        if op.op_type in _GEMM_FAMILY and len(op.in_shapes) >= 2
    )
    gather_op = TimedOp(
        op_id=f"{seg.seg_id}.fsdp_ag", op_type="CommOp", phase=seg.ops[0].phase,
        in_shapes=(), out_shape=(), dtype="bf16",   # 段级聚合载荷，无单一张量 shape/dtype 可挂
        stream=COMM_STREAM["dp"], deps=(),
        comm=CommSpec("all_gather", volume, "dp", dp_shard))
    return TimedSegment(seg.seg_id, (gather_op,) + seg.ops)


def inject_ep(seg: TimedSegment, ep: int) -> TimedSegment:
    """ep<=1 或段内无 GroupedMatMul → 恒等；否则在 grouped 区（首..末个 GroupedMatMul）两侧
    各插一条 dispatch/combine all_to_all。"""
    grouped_idx = [i for i, op in enumerate(seg.ops) if op.op_type == "GroupedMatMul"]
    if ep <= 1 or not grouped_idx:
        return seg
    first, last = grouped_idx[0], grouped_idx[-1]
    first_op, last_op = seg.ops[first], seg.ops[last]
    disp = TimedOp(
        op_id=f"{seg.seg_id}.ep_disp", op_type="CommOp", phase=first_op.phase,
        in_shapes=(first_op.in_shapes[0],), out_shape=first_op.in_shapes[0],
        dtype=first_op.dtype, stream=COMM_STREAM["ep"], deps=(),
        comm=CommSpec("all_to_all", tensor_bytes(first_op.in_shapes[0], first_op.dtype),
                      "ep", ep))
    comb = TimedOp(
        op_id=f"{seg.seg_id}.ep_comb", op_type="CommOp", phase=last_op.phase,
        in_shapes=(last_op.out_shape,), out_shape=last_op.out_shape,
        dtype=last_op.dtype, stream=COMM_STREAM["ep"], deps=(),
        comm=CommSpec("all_to_all", tensor_bytes(last_op.out_shape, last_op.dtype),
                      "ep", ep))
    ops = seg.ops[:first] + (disp,) + seg.ops[first:last + 1] + (comb,) + seg.ops[last + 1:]
    return TimedSegment(seg.seg_id, ops)


def inject_cp(seg: TimedSegment, cp: int, method: str = "colossal") -> TimedSegment:
    """cp<=1 → 恒等。method="colossal" → 每条 FlashAttention 前插一条 kv p2p；
    method="ulysses" → 每条 FlashAttention 前后各插一条 all_to_all；
    其余 method → ValueError（fail-loud）。"""
    if cp <= 1:
        return seg
    if method not in ("colossal", "ulysses"):
        raise ValueError(f"frame_comm: 未知 CP method {method!r}（fail-loud，不猜语义）")

    ops: list[TimedOp] = []
    for op in seg.ops:
        if op.op_type != "FlashAttention":
            ops.append(op)
            continue
        if method == "colossal":
            volume = (tensor_bytes(op.in_shapes[1], op.dtype)
                       + tensor_bytes(op.in_shapes[2], op.dtype))
            p2p = TimedOp(
                op_id=f"{op.op_id}.cp", op_type="CommOp", phase=op.phase,
                in_shapes=(op.in_shapes[1], op.in_shapes[2]), out_shape=op.in_shapes[1],
                dtype=op.dtype, stream=COMM_STREAM["cp"], deps=(),
                comm=CommSpec("p2p", volume, "cp", cp))
            ops.append(p2p)
            ops.append(op)
        else:   # ulysses：FA 前后各一条 all_to_all（seq-parallel↔head-parallel 切换）
            in_volume = sum(tensor_bytes(s, op.dtype) for s in op.in_shapes)
            pre = TimedOp(
                op_id=f"{op.op_id}.cp_pre", op_type="CommOp", phase=op.phase,
                in_shapes=op.in_shapes, out_shape=op.in_shapes[0],
                dtype=op.dtype, stream=COMM_STREAM["cp"], deps=(),
                comm=CommSpec("all_to_all", in_volume, "cp", cp))
            post = TimedOp(
                op_id=f"{op.op_id}.cp_post", op_type="CommOp", phase=op.phase,
                in_shapes=(op.out_shape,), out_shape=op.out_shape,
                dtype=op.dtype, stream=COMM_STREAM["cp"], deps=(),
                comm=CommSpec("all_to_all", tensor_bytes(op.out_shape, op.dtype), "cp", cp))
            ops.append(pre)
            ops.append(op)
            ops.append(post)
    return TimedSegment(seg.seg_id, tuple(ops))
