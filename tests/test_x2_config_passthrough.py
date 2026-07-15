"""P1-12 + 协调补充：LLMConfig → DimTable 新字段直通与惰性默认。

- MoE 多口径字段 `moe_dispatch_mode` / `moe_skew_factor`（任务 B）经 to_dimtable 直通到 DimTable。
- `qk_layernorm`（协调补充，供 X3 在 attention.py 建 GQA q/k norm 时从 DimTable 读取）直通。
- 默认惰性：不设时 balanced/1.0/False，DSv3 现有 to_dimtable 路径逐字段不变。
"""
import dataclasses

from cost_eval.llm_config import LLMConfig, to_dimtable
from cost_eval.presets import deepseek_v3


def _base_cfg(**over):
    cfg = LLMConfig(num_layers=2, hidden_size=16, num_attention_heads=4, vocab_size=32,
                    seq_length=8, num_moe_experts=6, moe_router_topk=1,
                    moe_ffn_hidden_size=32)
    return dataclasses.replace(cfg, **over) if over else cfg


# ── MoE 多口径字段直通 ─────────────────────────────────────────────────────────────
def test_dispatch_mode_and_skew_default_lazy():
    d = to_dimtable(_base_cfg())
    assert d.moe_dispatch_mode == "balanced"
    assert d.moe_skew_factor == 1.0


def test_dispatch_mode_passthrough():
    d = to_dimtable(_base_cfg(moe_dispatch_mode="capacity"))
    assert d.moe_dispatch_mode == "capacity"


def test_skew_factor_passthrough():
    d = to_dimtable(_base_cfg(moe_dispatch_mode="skew", moe_skew_factor=1.75))
    assert d.moe_dispatch_mode == "skew"
    assert d.moe_skew_factor == 1.75


# ── qk_layernorm 直通（协调补充）────────────────────────────────────────────────────
def test_qk_layernorm_default_false_lazy():
    d = to_dimtable(_base_cfg())
    assert d.qk_layernorm is False


def test_qk_layernorm_passthrough_true():
    d = to_dimtable(_base_cfg(qk_layernorm=True))
    assert d.qk_layernorm is True


# ── 现有 DSv3 路径不受影响（惰性默认）──────────────────────────────────────────────
def test_dsv3_dimtable_defaults_unchanged():
    d = to_dimtable(deepseek_v3(4))
    assert d.moe_dispatch_mode == "balanced"
    assert d.moe_skew_factor == 1.0
    assert d.qk_layernorm is False
