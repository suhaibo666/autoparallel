"""round3 A(F3 / D1-R)：OOM-安全咨询告警（不改数值,只提示欠预测风险）。

F3：n_layers 远超已验证锚点尺度(≤16L) → 全尺寸外推,累计每层残差(欠方向)无验证点 → 警告。
D1-R：MoE + pp==1 + 无重算 但 nr_moe_frag_factor=0（直连 LLMConfig 绕过 preset/adapter）→
      无重算-MoE loss 峰静默欠 ~0.93x → 警告。preset/adapter(factor=0.6)与 pp>1 不触发。
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator
from cost_eval.advisories import OOMSafetyWarning

GiB = 2 ** 30
_OPT = OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4)
_HW = HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0)


def _warns(spec, rc, *, pp=1, dp=2, needle=""):
    pc = ParallelConfig(dp_shard=dp, cp=1, tp=1, pp=pp, sequence_parallel=True, num_microbatches=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Evaluator(spec, pc, _OPT, _HW, rc, SwapSpec()).evaluate()
    return [str(x.message) for x in w
            if issubclass(x.category, OOMSafetyWarning) and needle in str(x.message)]


def test_f3_large_layer_count_warns_extrapolation():
    spec, _, _ = build_dsv3_spec(20)      # 20 > 16 已验证尺度
    assert _warns(spec, RecomputeSpec("None"), needle="n_layers")


def test_f3_anchor_scale_does_not_warn():
    spec, _, _ = build_dsv3_spec(8)       # 8 ≤ 16 → 不误报
    assert not _warns(spec, RecomputeSpec("None"), needle="n_layers")


def test_d1r_direct_api_margin_off_warns():
    spec, _, _ = build_dsv3_spec(8)
    spec.dims.nr_moe_frag_factor = 0.0    # 模拟直连 LLMConfig（margin 未开）
    assert _warns(spec, RecomputeSpec("None"), needle="nr_moe_frag_factor")


def test_d1r_preset_injected_margin_silent():
    spec, _, _ = build_dsv3_spec(8)       # preset 注入 0.6
    assert spec.dims.nr_moe_frag_factor == 0.6
    assert not _warns(spec, RecomputeSpec("None"), needle="nr_moe_frag_factor")


def test_d1r_pp_gt_1_silent_kce_covers():
    spec, _, _ = build_dsv3_spec(8)
    spec.dims.nr_moe_frag_factor = 0.0
    # pp>1 无重算 loss stage 走 `K_CE_PP` 分支（不吃 D1 margin）→ D1-R 不触发（不误报）。
    # ⚠ 2026-07-30 `K_CE_PP` 8→7 后该 stage 已不再「被平衡到 ~1.0」（0.9565）；本门守的是
    #   **gate 语义**（pp>1 不进 margin），与 K_CE 取值无关，故逐字节仍绿。
    assert not _warns(spec, RecomputeSpec("None"), pp=2, dp=1, needle="nr_moe_frag_factor")


def test_d1r_full_recompute_silent():
    spec, _, _ = build_dsv3_spec(8)
    spec.dims.nr_moe_frag_factor = 0.0
    # full 重算 → 非无重算域 → 不触发。
    assert not _warns(spec, RecomputeSpec("full", full_layers=set(range(1, 9))),
                      needle="nr_moe_frag_factor")
