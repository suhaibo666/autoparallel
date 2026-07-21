"""persistent（持久态）组成分解（用户报告 2026-07-21）。

`StaticMem.persistent_breakdown` 与 `compute` **同口径**（同 fsdp/efsdp/offload/按名去重），把每卡
持久字节拆成显式分量：**参数副本 + 优化器 master + momentum(m) + v(二阶动量)**（Muon 2D 矩阵权重
无 v）。各分量（含 512B 块对齐残差）之和恒 == `compute()` 的 persistent。
"""
from cost_eval.presets import deepseek_v3
from cost_eval.build_llm import build_llm_spec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import ParallelConfig, OptimizerSpec
from cost_eval.static_mem import StaticMem


def _setup(pc):
    spec = build_llm_spec(deepseek_v3(8))
    world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
    pm = ParallelModel(pc, spec.dims.n_layers, world)
    return ShapeEval().resolve(spec, pm), pm


def _comp_sum(bd_stage):
    return sum(byt for _n, _p, byt in bd_stage["components"])


def _byt(bd_stage, key):
    return sum(byt for n, _p, byt in bd_stage["components"] if key in n)


def _per(bd_stage, key):
    return [p for n, p, _b in bd_stage["components"] if key in n][0]


# ── 守恒:分量之和 == compute 的 persistent（全口径） ────────────────────────────
def test_breakdown_sums_to_persistent_adamw_fp32():
    pc = ParallelConfig(dp_shard=2)
    g, pm = _setup(pc)
    opt = OptimizerSpec.adamw(params_fp32=True)
    sm = StaticMem()
    persistent = sm.compute(g, opt, pm)
    bd = sm.persistent_breakdown(g, opt, pm)
    for stage in persistent:
        assert _comp_sum(bd[stage]) == persistent[stage] == bd[stage]["total_bytes"], (
            stage, _comp_sum(bd[stage]), persistent[stage])


def test_breakdown_sums_to_persistent_bf16_and_sharded():
    pc = ParallelConfig(dp_shard=4, tp=2, ep=2)
    g, pm = _setup(pc)
    opt = OptimizerSpec.adamw(params_fp32=False)
    sm = StaticMem()
    persistent = sm.compute(g, opt, pm)
    bd = sm.persistent_breakdown(g, opt, pm)
    for stage in persistent:
        assert _comp_sum(bd[stage]) == persistent[stage]


# ── 分量语义 ──────────────────────────────────────────────────────────────────
def test_adamw_fp32_components():
    # AdamW fp32:参数副本=0(无独立 compute 副本);master/m/v 各 4B/param(3 项优化器状态);总 12B/param。
    g, pm = _setup(ParallelConfig(dp_shard=1))
    bd = StaticMem().persistent_breakdown(g, OptimizerSpec.adamw(params_fp32=True), pm)[0]
    assert _per(bd, "参数副本") == 0
    assert _per(bd, "master") == 4
    state4 = [byt for n, p, byt in bd["components"] if p == 4]     # master + m + v
    assert len(state4) == 3 and all(b == bd["param_count"] * 4 for b in state4)
    assert bd["param_count"] * 12 == bd["total_bytes"]
    assert bd["optimizer"] == "AdamW"


def test_adamw_bf16_has_param_copy():
    # bf16 params → 多一份 compute 副本 2B/param → 14B/param。
    g, pm = _setup(ParallelConfig(dp_shard=1))
    bd = StaticMem().persistent_breakdown(g, OptimizerSpec.adamw(params_fp32=False), pm)[0]
    assert _per(bd, "参数副本") == 2
    assert bd["param_count"] * 14 == bd["total_bytes"]


def test_muon_matrix_has_no_variance():
    # Muon:2D 矩阵权重 momentum-only(无 v)→ v 分量 < AdamW,总持久 < AdamW。
    g, pm = _setup(ParallelConfig(dp_shard=1))
    sm = StaticMem()
    bd_a = sm.persistent_breakdown(g, OptimizerSpec.adamw(params_fp32=True), pm)[0]
    bd_m = sm.persistent_breakdown(g, OptimizerSpec.muon(params_fp32=True), pm)[0]
    assert bd_m["optimizer"] == "Muon" and bd_m["matrix_count"] > 0
    assert _byt(bd_m, "v(") < _byt(bd_a, "v(")        # Muon v 更少(矩阵无 v)
    assert bd_m["total_bytes"] < bd_a["total_bytes"]  # 省一份矩阵 v


def test_offload_optimizer_zeros_state_components():
    # offload_optimizer → master/m/v 归零,仅剩参数副本(bf16)。
    g, pm = _setup(ParallelConfig(dp_shard=1))
    bd = StaticMem().persistent_breakdown(
        g, OptimizerSpec.adamw(params_fp32=False), pm, offload_optimizer=True)[0]
    assert _byt(bd, "master") == 0 and _byt(bd, "momentum") == 0 and _byt(bd, "v(") == 0
    assert _byt(bd, "参数副本") > 0
