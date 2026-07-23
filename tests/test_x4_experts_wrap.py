"""P1-14 闭环：FSDP experts 子模块独立 wrap 的 gather/reshard 时间线（Task B）。

源忠实（mindformers pynative parallelize.py:1108-1161）：`layer.mlp.experts` 是**独立 FSDP 单元**
（efsdp mesh，:1108-1112，"small module first"），与整层 wrap（:1155-1161，"large module"）分开
——experts 的 all-gather/reshard 生命周期 ≠ 整层：experts 在层中段（dispatch 前）才 gather、
combine 后 reshard，**不在 attn 段驻留**。router/norm/shared-expert 权重仍在整层残余组（层入口 gather）。

建模：对**有真实 expert 分片**（efsdp>1）的 MoE 层，把前向 gather 拆成两段生命周期事件：
  - `fwd:{lid}`         —— attn+router 段（层入口）gather = 非专家权重 + 预取；
  - `fwd:{lid}#experts` —— experts 段（dispatch 前）gather = 非专家（驻留）+ 专家权重 + 预取。
两事件之差 == 专家权重字节 → experts gather 只在 expert 段可见、不在 attn 段。

**峰值口径保护**：experts 段事件（gather 含全部权重 + 该层 max workspace = dispatch/combine 56MiB）
逐字节复现旧单事件 `fwd:{lid}`；attn 段是**新增的更小事件**。故前向 gather 峰不变、且 DSv3/DSv4
锚点峰在 loss/optstep（backward，非 MoE 前向 gather 峰）→ 12 锚点逐字节不动（闭合到「时间线粒度」）。
efsdp<=1（无真实专家 all-gather，如 ep 全覆盖 / pp dp1 未分片）与 dense 层不拆（byte-identical）。
"""
from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.static_mem import StaticMem
from cost_eval.mem_timeline import MemTimeline
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from validate_dsv3 import build_dsv3_spec

MiB, GiB = 2 ** 20, 2 ** 30
BLK = 512


def _align(n, blk=BLK):
    return ((n + blk - 1) // blk) * blk if blk > 1 else n


def _param_split(layer):
    """该层去重后 (非专家, 专家) 权重字节（compute dtype，逐权重块对齐——与 param_full 同口径）。"""
    params = {}
    for op in layer.ops:
        for w in op.params:
            params[w.name] = w
    exp = sum(_align(w.local_numel * w.dtype_bytes) for w in params.values() if w.is_expert)
    non = sum(_align(w.local_numel * w.dtype_bytes) for w in params.values() if not w.is_expert)
    return non, exp


def _sim_dsv3(N=4, *, dp=2, ep=1, cp=1, recompute=None, record=True, depth=1):
    spec, d, fl = build_dsv3_spec(N)
    if cp > 1:
        d.B = 2
    pc = ParallelConfig(dp_shard=dp, cp=cp, tp=1, ep=ep, pp=1, sequence_parallel=True,
                        prefetch_depth=depth)
    pm = ParallelModel(pc, spec.dims.n_layers, world_size=dp * cp * ep)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(params_fp32=True), pm, False,
                                     alloc_block_bytes=BLK)
    r = MemTimeline().simulate(
        g, recompute or RecomputeSpec("None"), SwapSpec(), pm, persistent,
        framework_reserve=0, max_device_memory=64 * GiB,
        grad_dtype_bytes=4, record_timeline=record, alloc_block_bytes=BLK)
    return r, g


def _moe_layer_ids(g):
    return [l.layer_id for l in g.stages[0]
            if any(w.is_expert for op in l.ops for w in op.params)]


def _gather_at(sp, event):
    return next(s for s in sp.timeline if s.event == event).breakdown.gather_buf


# ---------------------------------------------------------------------------
# efsdp>1（DSv3 dp2）：MoE 层前向拆成两段 gather 事件
# ---------------------------------------------------------------------------

def test_moe_forward_emits_experts_gather_event():
    # P0-2(2026-07-23):expert 独立 wrap 仅 ep>1 存在(runtime parallelize.py:1496-1520)→
    # 拆分用例改 dp=4/ep=2(efsdp=2>1);ep=1 不再拆(见 test_ep1_no_experts_split)。
    r, g = _sim_dsv3(dp=4, ep=2)
    events = {s.event for s in r[0].timeline}
    moe = _moe_layer_ids(g)
    assert moe
    for lid in moe:
        assert f"fwd:{lid}" in events
        assert f"fwd:{lid}#experts" in events, lid


def test_experts_gather_delta_equals_expert_params():
    r, g = _sim_dsv3(dp=4, ep=2)   # P0-2:ep>1 才有 expert wrap
    by_id = {l.layer_id: l for l in g.stages[0]}
    for lid in _moe_layer_ids(g):
        non, exp = _param_split(by_id[lid])
        assert exp > 0
        attn_g = _gather_at(r[0], f"fwd:{lid}")
        exp_g = _gather_at(r[0], f"fwd:{lid}#experts")
        # experts 段比 attn 段多出的正是专家权重（预取/其余驻留项在两事件相同，抵消）。
        assert exp_g - attn_g == exp, (lid, exp_g - attn_g, exp)


def test_attn_segment_excludes_expert_weights():
    """attn 段 gather 不含专家权重：== 非专家 + 预取（预取=下一层整层，depth=1）。"""
    r, g = _sim_dsv3(dp=4, ep=2, depth=0)   # depth=0 关预取 → attn gather 恰 = 非专家(P0-2:ep>1)
    by_id = {l.layer_id: l for l in g.stages[0]}
    for lid in _moe_layer_ids(g):
        non, exp = _param_split(by_id[lid])
        assert _gather_at(r[0], f"fwd:{lid}") == non, (lid, _gather_at(r[0], f"fwd:{lid}"), non)


def test_experts_segment_is_full_layer_gather():
    """experts 段 gather = 非专家 + 专家 = 整层 param_full（depth=0）→ 复现旧单事件口径。"""
    from cost_eval.structure_mem import estimate_structure_memory
    r, g = _sim_dsv3(dp=4, ep=2, depth=0)   # P0-2:ep>1 才有 expert wrap
    by_id = {l.layer_id: l for l in g.stages[0]}
    for lid in _moe_layer_ids(g):
        full = estimate_structure_memory(by_id[lid].ops, alloc_block_bytes=BLK).param_full_bytes
        assert _gather_at(r[0], f"fwd:{lid}#experts") == full, lid


def test_ep1_no_experts_split():
    """P0-2(2026-07-23,runtime 377c9c344):ep=1 无 expert mesh、不单独 fully_shard experts
    (parallelize.py:1030-1037/:1106-1113/:1496-1520)→ 专家随父层 dense wrap,**无** #experts
    gather 段(修前 efsdp=dp>1 时误拆)。"""
    r, g = _sim_dsv3(dp=2, ep=1)
    events = {s.event for s in r[0].timeline}
    assert not any(e.endswith("#experts") for e in events)


# ---------------------------------------------------------------------------
# 锚点保护：split 不移峰（峰仍在 backward）
# ---------------------------------------------------------------------------

def test_split_does_not_move_peak_off_backward():
    r, g = _sim_dsv3(dp=2, recompute=RecomputeSpec("full", full_layers={1, 2, 3, 4}),
                     record=False)
    assert r[0].peak_event == "bwd@5", r[0].peak_event   # lm_head（非 MoE）反向


def test_dsv3_dp2_full_anchor_byte_stable():
    """DSv3 4L full dp2 锚点（真机 12473.1，本基线 12437.9 MiB）：experts 拆分后峰值不变。

    走完整 Evaluator（含 norm fp32 / 对齐 / kept_frag 口径，与 sim_vs_real_report 同路径），
    否则裸 MemTimeline 缺 norm_compute_dtype_bytes 等参数会得另一份数。"""
    from cost_eval.report import Evaluator
    spec, d, fl = build_dsv3_spec(4)
    pc = ParallelConfig(dp_shard=2, cp=1, tp=1, ep=1, pp=1, sequence_parallel=True)
    r = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True),
                  HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0),
                  RecomputeSpec("full", full_layers={1, 2, 3, 4}), SwapSpec()).evaluate()
    p = r.per_stage[0]
    assert p.peak_event == "bwd@5", p.peak_event
    assert abs(p.peak_bytes / MiB - 12437.9) < 0.1


# ---------------------------------------------------------------------------
# 不拆分：efsdp<=1（ep 全覆盖）/ dense 层 → 无 #experts 事件（byte-identical）
# ---------------------------------------------------------------------------

def test_no_experts_split_when_experts_unsharded_ep():
    # DSv3 4L dp2 ep2 → efsdp = (dp_shard*cp*tp)/ep = 2/2 = 1（experts 无真实 all-gather）。
    r, g = _sim_dsv3(dp=2, ep=2)
    assert not any(s.event.endswith("#experts") for s in r[0].timeline)


def test_dense_toy_no_experts_split():
    D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=4)
    spec = ModelSpec("toy", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(dp_shard=2, pp=1), n_layers=4, world_size=2)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(), pm, False, alloc_block_bytes=BLK)
    r = MemTimeline().simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                               framework_reserve=0, max_device_memory=GiB,
                               record_timeline=True, alloc_block_bytes=BLK)
    assert not any(s.event.endswith("#experts") for s in r[0].timeline)
