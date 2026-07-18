# cost_eval/timesim/report.py
"""StepTimeReport（spec §6.4）+ evaluate_step_time 门面：段装配 → 定价 → L1 → L2 → 步收尾。

装配口径（v1）：
  - 输入 = per-layer **fwd** TimedSegment 列表（producer 产物，未注入框架通信——本门面统一做
    cp/ep/fsdp 注入，组合序 cp→ep→fsdp，位置语义见各注入器 docstring）；层数须被 pp 整除
    （v1 均匀切层，layers_per_stage 非均匀=v1.5）；stage 内 chunk 均衡切分复用
    schedule.chunk_layer_ids（契约2 中立模块）。
  - pass = 该 (stage, chunk) 全部层段 concat（spec §5.2 连续 pass）；bwd pass = 各层
    expand_bwd 逆序 concat（+ reshard!="never" 时 fsdp_regather）。
  - 稳态口径：每 (stage, phase, chunk) 仿真一次，pipeline 内 m 个微批复用同一时长（§6.1）。
步收尾（§6.3）：
  1. grad sync：per-layer grad RS 已在 bwd 段内（expand_bwd 对偶自动涌现，可遮盖部分在段内
     DES 自动遮盖）；dp_replicate>1 的 DDP grad AR 无处可挂 → 作 barrier 尾**全暴露**串行加
     （v1 保守，诚实边界）。
  2. optimizer step：带宽类粗口径 = 本 stage 权重字节/2 · (state_bytes_per_param +
     grad_dtype_bytes) ÷ dp_shard ÷ (HBM·η_opt)（分片优化器；opt=None → 0——评估"纯前反向"）。
  3. per-step 固定开销 fixed_step_us：标定常数（T2 锚点反解），默认 0、单列不混 η。
MFU/HFU（§6.4，Megatron 惯例分开报）：分子 = m·Σ_(stage,chunk,phase) OpCost.flops（MFU 不含
recomp、HFU 含）；分母 = t_step · peak(dtype) · pp（每 stage world/pp 个 rank 算各自分片，
约分后剩 pp——推导见测试）。provenance_mix 按 (t_dev+t_comm) 加权聚合（§4.3）。"""
from __future__ import annotations

from dataclasses import dataclass

from ..schedule import chunk_layer_ids
from .ir import CommSpec, tensor_bytes
from .machine import TimeHardware
from .op_cost import CostModel, price_segment
from .pass_builder import concat_segments
from .segment_sim import simulate_segment
from .pipeline_sim import simulate_pipeline
from .bwd_rules import expand_bwd
from .frame_comm import inject_cp, inject_ep, inject_fsdp, fsdp_regather
from .shard_rules import Degrees

_GEMM = ("MatMul", "GroupedMatMul")


@dataclass(frozen=True)
class StepTimeReport:
    t_step_us: float
    t_pipeline_us: float
    t_opt_us: float
    t_grad_sync_tail_us: float
    fixed_step_us: float
    per_stage: tuple              # dict/stage：busy/bubble/host_gap/exposed_comm（×m 聚合）
    bubble_fraction: float
    bubble_fraction_closed_form: float
    critical_path: tuple
    mfu: float
    hfu: float
    provenance_mix: dict
    bottleneck_ranking: tuple     # ((名, us), ...) 降序
    uncalibrated: bool


def _weight_bytes(seg) -> int:
    return sum(tensor_bytes(o.in_shapes[1], o.dtype) for o in seg.ops
               if o.op_type in _GEMM and o.phase == "fwd" and len(o.in_shapes) >= 2)


def evaluate_step_time(layer_segments: list, deg: Degrees, hw: TimeHardware, *,
                        pp: int, m: int, v: int = 1, group_size: int | None = None,
                        recompute: str | None = None, recomp_comm: bool = False,
                        cp_method: str = "colossal", dp_replicate: int = 1,
                        reshard_after_forward: str = "default",
                        opt=None, p2p_bytes: int = 0, fixed_step_us: float = 0.0,
                        causal: bool = True) -> StepTimeReport:
    L = len(layer_segments)
    if pp <= 0 or L % pp:
        raise ValueError(f"report: 层数 {L} 不被 pp={pp} 整除（v1 均匀切层——fail-loud）")
    cm = CostModel(hw, causal=causal)
    per_stage_layers = [layer_segments[s * (L // pp):(s + 1) * (L // pp)]
                        for s in range(pp)]

    durations: dict = {}
    seg_times: dict = {}
    flops_eff = flops_recomp = 0
    prov_us: dict = {"hit": 0.0, "model": 0.0, "theory": 0.0}
    stage_weight_bytes = [0] * pp
    for s in range(pp):
        chunks = chunk_layer_ids(list(range(len(per_stage_layers[s]))), v)
        for c, idxs in enumerate(chunks):
            fwd_layers = []
            for k in idxs:
                seg = per_stage_layers[s][k]
                seg = inject_cp(seg, deg.cp, method=cp_method)
                seg = inject_ep(seg, deg.ep)
                seg = inject_fsdp(seg, deg.dp)
                fwd_layers.append(seg)
            bwd_layers = [expand_bwd(fs, recompute=recompute, recomp_comm=recomp_comm)
                          for fs in reversed(fwd_layers)]
            if deg.dp > 1 and reshard_after_forward != "never":
                bwd_layers = [fsdp_regather(bs, fs, deg.dp)
                              for bs, fs in zip(bwd_layers, reversed(fwd_layers))]
            for kind, layers in (("FWD", fwd_layers), ("BWD", bwd_layers)):
                p = concat_segments(f"s{s}.c{c}.{kind.lower()}", layers)
                costs = price_segment(p, cm)
                st = simulate_segment(p, costs)
                durations[(s, kind, c)] = st.duration_us
                seg_times[(s, kind, c)] = st
                for op in p.ops:
                    oc = costs[op.op_id]
                    if op.phase == "recomp":
                        flops_recomp += oc.flops
                    else:
                        flops_eff += oc.flops
                    prov_us[oc.provenance] = prov_us.get(oc.provenance, 0.0) \
                        + oc.t_dev_us + oc.t_comm_us
            stage_weight_bytes[s] += sum(_weight_bytes(f) for f in fwd_layers)

    p2p_us = 0.0
    if p2p_bytes and pp > 1:
        alpha, bw = hw.link("pp")
        p2p_us = alpha + p2p_bytes / bw * 1e6
    pipe = simulate_pipeline(durations, pp, m, v=v, group_size=group_size, p2p_us=p2p_us)

    # —— 步收尾（§6.3）——
    grad_tail = 0.0
    if dp_replicate > 1:
        gb = max(stage_weight_bytes) // 2 * (opt.grad_dtype_bytes if opt else 4)
        grad_tail = cm.comm_time_us(CommSpec("all_reduce", gb, "dp", dp_replicate))
    t_opt = 0.0
    if opt is not None:
        params = max(stage_weight_bytes) // 2                    # bf16 权重 → 参数量
        traffic = params * (opt.state_bytes_per_param + opt.grad_dtype_bytes)
        t_opt = traffic / max(deg.dp, 1) / (hw.hbm_bw * hw.eta["opt"]) * 1e6 \
            + hw.host_us("MatMul", "fwd")                        # host 发射一笔（粗口径）
    t_step = pipe.t_total_us + grad_tail + t_opt + fixed_step_us

    # —— MFU/HFU（模块 docstring 推导）——
    denom = t_step * 1e-6 * hw.peak("bf16") * pp
    mfu = m * flops_eff / denom if denom else 0.0
    hfu = m * (flops_eff + flops_recomp) / denom if denom else 0.0

    total_prov = sum(prov_us.values()) or 1.0
    prov_mix = {k: prov_us.get(k, 0.0) / total_prov for k in ("hit", "model", "theory")}

    per_stage = []
    agg = {"compute": 0.0, "membound": 0.0, "host": 0.0, "bubble": sum(pipe.per_stage_bubble)}
    for s in range(pp):
        hostg = m * sum(seg_times[k].t_host_gap for k in seg_times if k[0] == s)
        exp: dict = {}
        for k, st in seg_times.items():
            if k[0] != s:
                continue
            for ax, us in st.t_exposed_comm.items():
                exp[ax] = exp.get(ax, 0.0) + m * us
            agg["compute"] += m * st.t_compute
            agg["membound"] += m * st.t_membound
        agg["host"] += hostg
        for ax, us in exp.items():
            agg[f"comm_{ax}"] = agg.get(f"comm_{ax}", 0.0) + us
        per_stage.append({"stage": s, "busy_us": pipe.per_stage_busy[s],
                          "bubble_us": pipe.per_stage_bubble[s],
                          "host_gap_us": hostg, "exposed_comm": exp})
    if grad_tail:
        agg["grad_sync_tail"] = grad_tail
    ranking = tuple(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))

    v_ = max(v, 1)
    bf_closed = (pp - 1) / (m * v_ + pp - 1) if (m * v_ + pp - 1) else 0.0
    return StepTimeReport(
        t_step_us=t_step, t_pipeline_us=pipe.t_total_us, t_opt_us=t_opt,
        t_grad_sync_tail_us=grad_tail, fixed_step_us=fixed_step_us,
        per_stage=tuple(per_stage), bubble_fraction=pipe.bubble_fraction,
        bubble_fraction_closed_form=bf_closed, critical_path=pipe.critical_path,
        mfu=mfu, hfu=hfu, provenance_mix=prov_mix, bottleneck_ranking=ranking,
        uncalibrated=not hw.calibrated)
