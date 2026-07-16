"""Review-2 F9：build_llm 结构校验拒绝「OOM-under-safe」的 MoE dispatch 配置。

探针（`analysis/closure_report_verification_probe_2026-07-16.py::moe_skew_factor_below_one`）实证：
`moe_dispatch_mode="skew", moe_skew_factor=0.5` 把 dispatched token 从均衡 16 压到 **8**（低于均衡值）
却无任何报错——skew 建模的是**最忙 rank** 相对均值的放大，因子 <1 反而低估 OOM 边界，语义颠倒。

修复：`_validate_structure` 对 `moe_skew_factor`（须有限且 ≥1）、`moe_capacity_factor`（须 ≥1，
capacity=drop-and-pad 上界）、`moe_dispatch_mode`（仅 balanced|capacity|skew）fail-loud。
默认 balanced/1.0/1.0 全部通过 → 惰性、锚点逐字节不变。
"""
import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.llm_config import LLMConfig
from cost_eval.presets import deepseek_v3


def _moe_cfg(**over):
    base = dict(
        num_layers=2, hidden_size=8, num_attention_heads=2, num_query_groups=2,
        vocab_size=16, seq_length=8, batch_size=1, head_dim=4, attn_type="gqa",
        ffn_hidden_size=16, num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=16,
    )
    base.update(over)
    return LLMConfig(**base)


def test_skew_factor_below_one_rejected():
    with pytest.raises(ValueError, match="moe_skew_factor"):
        build_llm_spec(_moe_cfg(moe_dispatch_mode="skew", moe_skew_factor=0.5))


def test_skew_factor_non_finite_rejected():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="moe_skew_factor"):
            build_llm_spec(_moe_cfg(moe_dispatch_mode="skew", moe_skew_factor=bad))


def test_capacity_factor_below_one_rejected():
    with pytest.raises(ValueError, match="moe_capacity_factor"):
        build_llm_spec(_moe_cfg(moe_capacity_factor=0.5))


def test_unknown_dispatch_mode_rejected():
    with pytest.raises(ValueError, match="moe_dispatch_mode"):
        build_llm_spec(_moe_cfg(moe_dispatch_mode="p99"))


def test_skew_factor_ge_one_accepted():
    assert build_llm_spec(_moe_cfg(moe_dispatch_mode="skew", moe_skew_factor=1.5)) is not None


def test_capacity_mode_accepted():
    assert build_llm_spec(_moe_cfg(moe_dispatch_mode="capacity")) is not None


def test_defaults_still_build_and_are_off():
    """默认 balanced/1.0/1.0：新校验放行 → DSv3 锚点仍正常建（byte-identical 由既有锚点测试守）。"""
    cfg = _moe_cfg()   # 默认 dispatch 口径
    assert cfg.moe_dispatch_mode == "balanced"
    assert cfg.moe_skew_factor == 1.0
    assert cfg.moe_capacity_factor == 1.0
    assert build_llm_spec(cfg) is not None
    assert build_llm_spec(deepseek_v3(4)) is not None
