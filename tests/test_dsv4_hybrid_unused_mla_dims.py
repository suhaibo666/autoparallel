"""dsv4_hybrid 的 op 图不含 kv_lora_rank / qk_nope_head_dim —— 守卫豁免的回归。

背景(2026-07-25):`build_llm._validate_structure` 原先对 attn_type ∈ {mla, dsv4_hybrid, dsa}
一律要求五个 MLA 维 > 0。但逐符号核实:`layers/dsv4_hybrid.py` 中 kv_lora_rank / qk_nope_head_dim
出现 **0 次**(而 q_lora_rank/qk_rope_head_dim/v_head_dim 分别 6/2/31 次)。机制原因见
dsv4_hybrid.py:49-51 —— 该变体 `q_head_dim = config.v_head_dim`
(deepseek_v4_hybrid_attention.py:66),Q/K 头维整体取 v_head_dim、不按 nope+rope 拆;MLA 的
KV-LoRA 通路被 CSA compressor 取代(CMP_PROJ_OUT = coff*v_head_dim)。现场 DSv4-Flash yaml
因此根本不写 kv_lora_rank,旧守卫逼使用者编一个值 —— 正是"静默替代"失效模式。

本测试是**双向**守着的:
  · 若这两维哪天真进了 dsv4_hybrid 的图(峰值随其变化)→ 不变性断言失败 → 回去恢复守卫;
  · mla / dsa 仍须 fail-loud(它们真在用:attention.py 7/3 次、dsa.py 13/2 次)。
"""
import dataclasses

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.report import Evaluator
from cost_eval.specs import (HardwareSpec, OptimizerSpec, ParallelConfig,
                             RecomputeSpec, SwapSpec)


def _peak(cfg):
    spec = build_llm_spec(cfg)
    rep = Evaluator(spec, ParallelConfig(), OptimizerSpec(),
                    HardwareSpec(max_device_memory=64 * 2 ** 30, framework_reserve=0),
                    RecomputeSpec("None"), SwapSpec(),
                    check_feasibility=False).evaluate()
    return max(p.peak_bytes for p in rep.per_stage)


@pytest.mark.parametrize("field", ["kv_lora_rank", "qk_nope_head_dim"])
def test_dsv4_hybrid_peak_invariant_to_unused_mla_dims(field):
    """dsv4_hybrid 峰值对这两维**逐字节不变** → 它们确实不进 op 图。"""
    base = deepseek_v4(num_layers=4)
    peaks = {v: _peak(dataclasses.replace(base, **{field: v})) for v in (0, 1, 512, 4096)}
    assert len(set(peaks.values())) == 1, (
        f"{field} 改变了 dsv4_hybrid 峰值 {peaks} —— 它已进入 op 图,"
        "请回 build_llm._validate_structure 恢复对本变体的 >0 守卫")


@pytest.mark.parametrize("field", ["kv_lora_rank", "qk_nope_head_dim"])
def test_dsv4_hybrid_accepts_zero_unused_mla_dims(field):
    """=0 可直接建图(现场 yaml 不写 kv_lora_rank 也应能评估,不必编值)。"""
    cfg = dataclasses.replace(deepseek_v4(num_layers=4), **{field: 0})
    assert _peak(cfg) > 0


@pytest.mark.parametrize("field", ["kv_lora_rank", "qk_nope_head_dim"])
def test_mla_still_fail_loud_on_zero(field):
    """mla 仍严查 —— attention.py 真用这两维,≤0 会静默产零/负 numel。"""
    cfg = dataclasses.replace(deepseek_v3(num_layers=4), **{field: 0})
    with pytest.raises(ValueError, match=field):
        build_llm_spec(cfg)


def test_dsv4_hybrid_still_fail_loud_on_genuinely_used_dims():
    """豁免只针对那两维:真在用的 q_lora_rank / v_head_dim / qk_rope_head_dim 仍须 >0。"""
    for field in ("q_lora_rank", "qk_rope_head_dim", "v_head_dim"):
        cfg = dataclasses.replace(deepseek_v4(num_layers=4), **{field: 0})
        with pytest.raises(ValueError, match=field):
            build_llm_spec(cfg)
