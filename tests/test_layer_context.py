"""Task 1 — structured ``LayerContext`` replaces per-layer string keys.

`gen_layer_pattern(cfg)` now returns ``list[LayerContext]`` (frozen/hashable) instead
of ``list[str]``; `build_llm_spec` dedups by the LayerContext itself and derives the
ModelSpec's string layer-type names via ``LayerContext.name`` (an OUTPUT label — never
re-parsed for logic). The string encode/parse (`_DSV4_KEY_PREFIX`, ``rsplit``,
``int(prefix...)``) is removed; dispatch is structural on ctx fields.
"""
import inspect

import pytest

from cost_eval.layer_context import LayerContext
from cost_eval.build_llm import gen_layer_pattern, build_llm_spec
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.llm_config import LLMConfig


# ── LayerContext is a frozen, hashable dataclass ───────────────────────────────

def test_layer_context_frozen_and_hashable():
    ctx = LayerContext(kind="decoder", attn_type="mla", ffn_type="moe")
    # frozen → assignment raises
    with pytest.raises(Exception):
        ctx.kind = "embedding"
    # hashable → usable as a dict key / set member (dedup of identical layers)
    d = {ctx: 1}
    assert d[LayerContext(kind="decoder", attn_type="mla", ffn_type="moe")] == 1
    assert len({ctx, LayerContext(kind="decoder", attn_type="mla", ffn_type="moe")}) == 1


def test_layer_context_defaults():
    ctx = LayerContext(kind="embedding")
    assert ctx.attn_type is None
    assert ctx.compress_ratio is None
    assert ctx.ffn_type is None
    assert ctx.residual_variant == "plain"


# ── .name reproduces the deterministic string layer-type label ─────────────────

def test_name_property_matches_legacy_format():
    assert LayerContext(kind="embedding").name == "embedding"
    assert LayerContext(kind="lm_head").name == "lm_head"
    assert LayerContext(kind="mtp").name == "mtp"
    assert LayerContext(kind="decoder", attn_type="mla", ffn_type="dense").name == "mla_dense"
    assert LayerContext(kind="decoder", attn_type="mla", ffn_type="moe").name == "mla_moe"
    assert LayerContext(kind="decoder", attn_type="gqa", ffn_type="dense").name == "gqa_dense"
    # dsv4_hybrid encodes per-layer compress_ratio in the label
    assert LayerContext(kind="decoder", attn_type="dsv4_hybrid",
                        compress_ratio=4, ffn_type="moe").name == "dsv4hyb_r4_moe"
    assert LayerContext(kind="decoder", attn_type="dsv4_hybrid",
                        compress_ratio=0, ffn_type="dense").name == "dsv4hyb_r0_dense"


# ── gen_layer_pattern returns structured LayerContext objects ──────────────────

def test_gen_layer_pattern_returns_layer_contexts():
    pattern = gen_layer_pattern(deepseek_v3(4))
    assert all(isinstance(c, LayerContext) for c in pattern)
    assert pattern[0].kind == "embedding"
    assert pattern[-1].kind == "lm_head"
    # 1 dense MLA layer (first_k_dense=1) + 3 MoE MLA layers
    decoders = [c for c in pattern if c.kind == "decoder"]
    assert [c.attn_type for c in decoders] == ["mla"] * 4
    assert [c.ffn_type for c in decoders] == ["dense", "moe", "moe", "moe"]


def test_gen_layer_pattern_dsv4_compress_ratios_structured():
    """deepseek_v4(4): decoder ctxs carry compress_ratios [0,4,128,0] + one mtp ctx."""
    pattern = gen_layer_pattern(deepseek_v4(4))
    decoders = [c for c in pattern if c.kind == "decoder"]
    assert [c.compress_ratio for c in decoders] == [0, 4, 128, 0]
    assert all(c.attn_type == "dsv4_hybrid" for c in decoders)
    assert [c.kind for c in pattern if c.kind == "mtp"] == ["mtp"]


# ── the string key is an OUTPUT label, never re-parsed for logic ───────────────

def test_no_string_key_parsing_in_build_llm():
    src = inspect.getsource(__import__("cost_eval.build_llm", fromlist=["x"]))
    assert "rsplit" not in src, "layer key must not be re-parsed via rsplit"
    assert "_DSV4_KEY_PREFIX" not in src, "string-prefix key encoding must be removed"
    assert "int(prefix" not in src, "compress_ratio must come from ctx, not string parse"


def test_modelspec_layer_pattern_is_name_strings():
    """ModelSpec keeps str layer-type names (dict[str, LayerSpec]) via ctx.name."""
    cfg = deepseek_v4(4)
    spec = build_llm_spec(cfg)
    assert spec.layer_pattern == [c.name for c in gen_layer_pattern(cfg)]
    assert all(isinstance(k, str) for k in spec.layer_specs)


def test_layer_context_dedups_identical_layers():
    """Two MoE MLA decoder layers dedup to a single LayerSpec (hashable ctx key)."""
    cfg = deepseek_v3(4)
    spec = build_llm_spec(cfg)
    # 4 transformer layers but only 2 unique decoder keys (mla_dense, mla_moe)
    decoder_keys = [k for k in spec.layer_specs if k in ("mla_dense", "mla_moe")]
    assert set(decoder_keys) == {"mla_dense", "mla_moe"}
