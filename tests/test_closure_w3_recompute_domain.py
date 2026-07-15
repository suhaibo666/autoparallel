"""闭环审计第四轮 §F4（analysis/closure_audit_v2_verification_2026-07-15.md）回归：
core 重算输入域校验仍可静默空转/误选。

`_validate_recompute_against_graph`（report.py）此前只处理 mode in {full, select}，其它 mode
直接 return，且 select 分支未防空串选择器。探针（closure_audit_v2_verification_probe_2026-07-15.py）
实证两类此前静默接受的坏配置：
  - mode typo：`RecomputeSpec("ful", {1})` 被接受，峰值与 `RecomputeSpec("None")` 逐字节相同 →
    静默空转（配置看似启用重算、实际什么都没算）；
  - 空/纯空白 selector：`RecomputeSpec("select", select_ops={1:{""}})` 被接受——空串是所有 op
    名/类型的子串、`op_matches` 意外命中整层（若真想重算整层应显式 mode='full'）。

守卫写在核心评估入口 `_validate_recompute_against_graph`，统一 fail-loud（不只 UI 封口）。
合法的 None/'None'/'none'（不重算）与 full/select（有效层号 / 命中选择器）必须继续通过。
"""
import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.presets import deepseek_v3
from cost_eval.report import Evaluator
from cost_eval.specs import (HardwareSpec, OptimizerSpec, ParallelConfig,
                             RecomputeSpec, SwapSpec)

G = 2 ** 30


def _ev(rc):
    """deepseek_v3(4)：图层 id = 0(embedding)/1(mla_dense)/2..4(mla_moe)/5(lm_head)。"""
    return Evaluator(build_llm_spec(deepseek_v3(4)),
                     ParallelConfig(dp_shard=2, sequence_parallel=True),
                     OptimizerSpec.adamw(), HardwareSpec(64 * G), rc, SwapSpec())


# ── ① mode 枚举：非法 mode（typo）必须 fail-loud（此前 mode not in (full,select) 静默 return）──
def test_mode_typo_ful_rejected():
    # "ful" 是 "full" 的 typo → 此前静默等效不重算、与 RecomputeSpec("None") 逐字节相同
    with pytest.raises(ValueError, match="mode"):
        _ev(RecomputeSpec("ful", {1})).evaluate()


def test_mode_typo_selct_rejected():
    with pytest.raises(ValueError, match="mode"):
        _ev(RecomputeSpec("selct", select_ops={1: {"flash"}})).evaluate()


def test_mode_typo_capitalized_full_rejected():
    # "Full" 大小写 typo（合法 mode 是小写 full）→ 也须拒，避免静默空转
    with pytest.raises(ValueError, match="mode"):
        _ev(RecomputeSpec("Full", {1})).evaluate()


# ── mode 合法的不重算取值（None / 'None' / 'none'）必须继续放行 ─────────────────────
def test_mode_none_string_accepted():
    assert _ev(RecomputeSpec("None")).evaluate().per_stage[0].peak_bytes > 0


def test_mode_none_lower_accepted():
    assert _ev(RecomputeSpec("none")).evaluate().per_stage[0].peak_bytes > 0


def test_mode_none_object_accepted():
    assert _ev(RecomputeSpec(None)).evaluate().per_stage[0].peak_bytes > 0


def test_mode_default_accepted():
    assert _ev(RecomputeSpec()).evaluate().per_stage[0].peak_bytes > 0


# ── ② select 空 / 纯空白 selector：空串命中整层，是配置错误、须 fail-loud ─────────────
def test_select_empty_string_selector_rejected():
    # {1:{""}}：空串是所有 op 名/类型子串 → op_matches 命中整层（非合法「重算整层」，应用 mode='full'）
    with pytest.raises(ValueError, match="空"):
        _ev(RecomputeSpec("select", select_ops={1: {""}})).evaluate()


def test_select_whitespace_selector_rejected():
    # 纯空白 "   ".strip()=="" 同样命中整层
    with pytest.raises(ValueError, match="空"):
        _ev(RecomputeSpec("select", select_ops={1: {"   "}})).evaluate()


def test_select_mixed_valid_and_empty_selector_rejected():
    # 有效 selector "flash" 混入空串 "" 也须拒（空串仍会命中整层）
    with pytest.raises(ValueError, match="空"):
        _ev(RecomputeSpec("select", select_ops={1: {"flash", ""}})).evaluate()


# ── 合法 select（非空、命中）继续通过 ────────────────────────────────────────────
def test_select_valid_selector_still_accepted():
    # layer 1(mla_dense) 含 flash op → 命中，合法。
    r = _ev(RecomputeSpec("select", select_ops={1: {"flash"}})).evaluate()
    assert r.per_stage[0].peak_bytes > 0
