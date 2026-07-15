"""P1-16 闭环：完整 runtime feasibility constraint matrix（test_y2_*，2026-07-15）。

`feasibility_errors(pc, optimizer, swap)`（report.py）此前只有 3 条（PP+swap、tp>1+SP、非
Adam）——P1-16 审计判「部分（非完整约束矩阵）」。本文件把它扩成覆盖 mindformers pynative 运行时
可行性的**约束矩阵**（搜索器/评估器接受一个组合前先判真机可跑），逐条：非法组合 fail-loud（错误串
含字段/值/合法范围）+ 合法组合通过 + 12 锚点并行组合全通过（守卫不破合法配置）。

每条新约束的 mindformers 源码依据见 report.feasibility_errors 内联注释。分工：结构合法性（维度
正值/整除、pp≤层数、S%cp、tp%heads）留在 build_llm._validate_structure / ParallelModel，
feasibility 只管**并行/调度组合**的可跑性——**不重复**结构校验。
"""
import pytest

from cost_eval.specs import (ParallelConfig, OptimizerSpec, SwapSpec,
                             HardwareSpec, RecomputeSpec)
from cost_eval.report import feasibility_errors, Evaluator


def _f(pc=None, *, opt=None, swap=None):
    """feasibility_errors 直调（返回错误串列表，空=可行）。"""
    return feasibility_errors(pc if pc is not None else ParallelConfig(),
                              opt or OptimizerSpec.adamw(), swap or SwapSpec())


# ── 约束 4（新）：ep 须整除 dp_shard·cp·tp（parallel_dims.py:140-148）──────────────
def test_ep_not_dividing_region_flagged():
    # region = dp_shard·cp·tp = 1；ep=4 不整除 → 真机 efsdp 静默 floor 到 0/错值、mesh 尺寸不匹配。
    errs = _f(ParallelConfig(ep=4, dp_shard=1, cp=1, tp=1))
    assert any("整除" in e for e in errs), errs
    assert any("ep" in e and "4" in e for e in errs), errs


def test_ep_divides_region_ok():
    # region=2，ep=2 → 整除，可行。
    assert _f(ParallelConfig(ep=2, dp_shard=2, cp=1, tp=1)) == []


def test_ep_region_via_cp_tp_ok():
    # region = 1·2·2 = 4，ep=4 → 整除（tp>1 配 SP）。
    assert _f(ParallelConfig(ep=4, dp_shard=1, cp=2, tp=2, sequence_parallel=True)) == []


def test_ep1_always_ok():
    # ep=1 恒整除任意 region。
    assert _f(ParallelConfig(ep=1, dp_shard=3, cp=1, tp=1)) == []


# ── 约束 5（新）：interleave(VPP)>1 须 pp>1（单 stage 交错无意义、会被静默丢弃）────────
def test_vpp_interleave_requires_pp_gt1():
    errs = _f(ParallelConfig(interleave=2, pp=1))
    assert any("interleave" in e for e in errs), errs


def test_vpp_interleave_with_pp_ok():
    assert _f(ParallelConfig(interleave=2, pp=2, num_microbatches=2)) == []


def test_interleave1_pp1_ok():
    # 无 VPP（默认 interleave=1）→ pp=1 完全合法。
    assert _f(ParallelConfig(interleave=1, pp=1)) == []


# ── 约束 6（新）：VPP 下 num_microbatches ≥ pp（交错 warmup 深度）───────────────────
def test_vpp_microbatches_below_pp_flagged():
    errs = _f(ParallelConfig(interleave=2, pp=4, num_microbatches=2))
    assert any("num_microbatches" in e for e in errs), errs


def test_vpp_microbatches_equal_pp_ok():
    assert _f(ParallelConfig(interleave=2, pp=2, num_microbatches=2)) == []


def test_plain_1f1b_microbatches_below_pp_ok():
    """关键守卫：plain 1F1B（interleave=1）下 m<pp **合法**——build_1f1b warmup=min(pp-1-stage,m)
    会 clamp，真机可跑（只是流水气泡）。约束 6 仅对 VPP（interleave>1）生效，不误伤 plain。"""
    assert _f(ParallelConfig(pp=4, num_microbatches=1, interleave=1)) == []


# ── 约束 7（新）：swap 开启时 default_prefetch ≥ 1（config.py:300-307）──────────────
def test_swap_prefetch_below_1_flagged():
    errs = _f(ParallelConfig(pp=1),
              swap=SwapSpec(enable=True, default_prefetch=0, swap_layers={1}))
    assert any("prefetch" in e for e in errs), errs


def test_swap_prefetch_ok():
    assert _f(ParallelConfig(pp=1),
              swap=SwapSpec(enable=True, default_prefetch=1, swap_layers={1})) == []


def test_swap_disabled_prefetch_irrelevant():
    # swap 关 → default_prefetch 无关紧要，不校验。
    assert _f(ParallelConfig(pp=1), swap=SwapSpec(enable=False, default_prefetch=0)) == []


# ── 既有 3 条仍生效（回归守卫，不因扩展而丢）───────────────────────────────────────
def test_existing_pp_swap_still_flagged():
    errs = _f(ParallelConfig(pp=2, num_microbatches=2),
              swap=SwapSpec(enable=True, swap_layers={1}))
    assert any("swap" in e for e in errs), errs


def test_existing_tp_requires_sp():
    errs = _f(ParallelConfig(tp=2, sequence_parallel=False))
    assert any("sequence_parallel" in e for e in errs), errs


def test_existing_non_adam():
    errs = _f(ParallelConfig(), opt=OptimizerSpec(type="SGD"))
    assert any("optimizer" in e for e in errs), errs


# ── 12 锚点并行组合守卫：全部可行（feasibility 返回空）────────────────────────────
# 源自 sim_vs_real_report.py 的 dsv3()/dsv4 锚点并行元组（去重后的 5 个不同组合，覆盖
# dp2/ep2/cp2(colossal|ulysses)/pp2/tp1；DSv4 与锚点 1 同组合）。
_ANCHOR_PCS = [
    dict(dp_shard=2, cp=1, tp=1, ep=1, pp=1, num_microbatches=1),                                 # 1,2,9-11,DSv4
    dict(dp_shard=2, cp=1, tp=1, ep=2, pp=1, num_microbatches=1),                                 # 3 ep=2
    dict(dp_shard=1, cp=2, tp=1, ep=1, pp=1, num_microbatches=1, context_parallel_method="colossal"),  # 4,8
    dict(dp_shard=1, cp=2, tp=1, ep=1, pp=1, num_microbatches=1, context_parallel_method="ulysses"),   # 5
    dict(dp_shard=1, cp=1, tp=1, ep=1, pp=2, num_microbatches=2),                                 # 6,7 pp2
]


@pytest.mark.parametrize("kw", _ANCHOR_PCS)
def test_all_anchor_parallel_combos_feasible(kw):
    kw = dict(kw)
    kw.setdefault("sequence_parallel", True)   # dsv3 锚点恒 SP=True
    errs = feasibility_errors(ParallelConfig(**kw),
                              OptimizerSpec.adamw(params_fp32=True), SwapSpec())
    assert errs == [], (kw, errs)


# 其它现有测试里出现过的合法并行组合（防误拦）。
def test_representative_existing_combos_feasible():
    assert _f(ParallelConfig(tp=8, dp_shard=8, cp=1, ep=4, pp=1, sequence_parallel=True)) == []
    assert _f(ParallelConfig(ep=4, tp=2, dp_shard=2, num_microbatches=1,
                             sequence_parallel=True)) == []
    assert _f(ParallelConfig(dp_shard=1, pp=2, interleave=2, num_microbatches=4)) == []
    assert _f(ParallelConfig()) == []


# ── Evaluator 集成 + check_feasibility=False 绕过通道不变 ──────────────────────────
from cost_eval.build_llm import build_llm_spec
from cost_eval.presets import deepseek_v3

G = 2 ** 30


def _spec():
    return build_llm_spec(deepseek_v3(4))


def test_evaluator_rejects_infeasible_ep_at_construction():
    # 新约束在**核心评估入口**生效（构造即 fail-loud，早于 evaluate/ParallelModel）。
    with pytest.raises(ValueError, match="整除"):
        Evaluator(_spec(), ParallelConfig(ep=4, dp_shard=1, cp=1, tp=1),
                  OptimizerSpec.adamw(), HardwareSpec(64 * G), RecomputeSpec(), SwapSpec())


def test_evaluator_rejects_vpp_without_pp():
    with pytest.raises(ValueError, match="interleave"):
        Evaluator(_spec(), ParallelConfig(interleave=2, pp=1),
                  OptimizerSpec.adamw(), HardwareSpec(64 * G), RecomputeSpec(), SwapSpec())


def test_evaluator_bypass_channel_unchanged():
    # check_feasibility=False 仍可构造不可跑组合（纯内存口径）——绕过通道不变。
    Evaluator(_spec(), ParallelConfig(interleave=2, pp=1),
              OptimizerSpec.adamw(), HardwareSpec(64 * G), RecomputeSpec(), SwapSpec(),
              check_feasibility=False)
