"""闭环审计 v3（closure_audit_verification_2026-07-15.md §4.3，P1-02）反例转正式回归。

复核实证：`_validate_recompute_against_graph` 能拒 full 空集 + 存在层的 selector 零命中，但仍
静默接受下述「配置看起来启用、实际空转」三类反例（与无重算逐字节相同）：
  - full 越界层号（`full_layers={999}`，不在图层 id 范围）；
  - select 全空（`select_ops={}` 或全 `set()`，无任何层配非空选择器 → 等效 None）；
  - select 越界层号（`select_ops={999:{...}}`，此前对 `by_id.get(lid) is None` 直接 continue）。

守卫写在核心评估入口 `Evaluator.evaluate`（经 `_validate_recompute_against_graph`），层号范围按
**实际 resolved 图的 layer_id 并集**（`{l.layer_id for lys in g.stages.values() for l in lys}`）判断——
g 是全图（`ShapeEval.resolve` 遍历整个 layer_pattern、按 stage_of 分组，g.stages 含所有 stage/层）。
合法用法（full/select 层号在 0..n_layers-1 范围内、选择器有命中）必须继续通过。
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


# ── 反例：坏配置必须 fail-loud（此前静默接受、等效不重算）─────────────────────────
def test_full_out_of_range_layer_rejected():
    with pytest.raises(ValueError, match="越界"):
        _ev(RecomputeSpec("full", {999})).evaluate()


def test_full_out_of_range_mixed_with_valid_rejected():
    # 合法层 {1,2} + 越界层 {999} 混配也必须拒（越界层等效不重算、与意图相反）。
    with pytest.raises(ValueError, match="越界"):
        _ev(RecomputeSpec("full", {1, 2, 999})).evaluate()


def test_select_empty_map_rejected():
    with pytest.raises(ValueError, match="select"):
        _ev(RecomputeSpec("select", select_ops={})).evaluate()


def test_select_all_empty_selectors_rejected():
    # 层存在但选择器全空集 → 无任何非空选择器 → 等效 None、配置空转。
    with pytest.raises(ValueError, match="select"):
        _ev(RecomputeSpec("select", select_ops={1: set(), 2: set()})).evaluate()


def test_select_out_of_range_layer_rejected():
    # 此前 by_id.get(999) is None → 直接 continue 静默接受；现须 fail-loud（不是 continue）。
    with pytest.raises(ValueError, match="越界"):
        _ev(RecomputeSpec("select", select_ops={999: {"flash"}})).evaluate()


def test_select_zero_hit_still_rejected():
    # 层存在但选择器零命中 → 保持既有 fail-loud（不被新逻辑吞掉）。
    with pytest.raises(ValueError, match="零命中"):
        _ev(RecomputeSpec("select", select_ops={1: {"definitely_missing"}})).evaluate()


# ── 合法用法守卫：层号在范围内、选择器有命中 → 必须继续通过 ───────────────────────
def test_full_valid_layers_accepted():
    r = _ev(RecomputeSpec("full", {1, 2, 3, 4})).evaluate()
    assert r.per_stage[0].peak_bytes > 0


def test_select_valid_hitting_selector_accepted():
    # layer 1(mla_dense) 含 flash op → 命中，合法。
    r = _ev(RecomputeSpec("select", select_ops={1: {"flash"}})).evaluate()
    assert r.per_stage[0].peak_bytes > 0


def test_none_mode_unaffected():
    r = _ev(RecomputeSpec()).evaluate()   # mode="None"：不进重算校验
    assert r.per_stage[0].peak_bytes > 0
