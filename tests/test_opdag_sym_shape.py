# tests/test_opdag_sym_shape.py
"""符号 shape 代数(T9 底座):把 "S·B·H" 这类 shape 串解析成可乘/除/加/整除的因子多重集,
支持 reshape 的 -1 因子消元。axes 以 `·` 连接,乘积轴用 `(...)` 包裹以消歧(如 "S·B·(n_heads·qk_head_dim)")。"""
import pytest

from cost_eval.opdag.sym_shape import (
    parse_shape, render_shape, parse_axis, render_term,
    mul, divide, add, floordiv, product_of, resolve_reshape, NEG1,
)


def test_parse_and_render_roundtrip_simple():
    assert render_shape(parse_shape("S·B·H")) == "S·B·H"
    assert render_shape(parse_shape("E·cap·H")) == "E·cap·H"


def test_parse_product_axis_is_one_axis():
    axes = parse_shape("S·B·(n_heads·qk_head_dim)")
    assert len(axes) == 3
    assert render_term(axes[2]) == "n_heads·qk_head_dim"
    assert render_shape(axes) == "S·B·(n_heads·qk_head_dim)"


def test_parse_sum_axis_kept_atomic():
    axes = parse_shape("S·B·(kv_lora_rank+qk_pos_emb_head_dim)")
    assert len(axes) == 3
    assert render_shape(axes) == "S·B·(kv_lora_rank+qk_pos_emb_head_dim)"


def test_coeff_renders_first():
    ax = parse_axis("2·ffn_hidden")
    assert render_term(ax) == "2·ffn_hidden"
    # product with a nested sum wraps the sum in parens
    ax2 = mul(parse_axis("n_heads"), parse_axis("(qk_head_dim+v_head_dim)"))
    assert render_term(ax2) == "n_heads·(qk_head_dim+v_head_dim)"


def test_floordiv_halves_coeff():
    assert render_term(floordiv(parse_axis("2·ffn_hidden"), 2)) == "ffn_hidden"
    assert floordiv(parse_axis("H"), 2) is None    # H not divisible by 2 → None (not fabricated)


def test_mul_and_divide_cancel():
    total = product_of(parse_shape("S·B·(n_heads·v_head_dim)"))
    denom = product_of(parse_shape("S·B·n_heads"))
    assert render_term(divide(total, denom)) == "v_head_dim"


def test_add_makes_sum_unit_canonical():
    s = add(parse_axis("qk_head_dim"), parse_axis("qk_pos_emb_head_dim"))
    # canonical: terms sorted
    assert render_term(s) == "qk_head_dim+qk_pos_emb_head_dim"
    s2 = add(parse_axis("qk_pos_emb_head_dim"), parse_axis("qk_head_dim"))
    assert render_term(s2) == render_term(s)


def test_resolve_reshape_fills_single_neg1_by_cancellation():
    # kv: [S,B,(n_heads·(qk_head_dim+v_head_dim))] reshape to (S,B,n_heads,-1)
    inp = parse_shape("S·B·(n_heads·(qk_head_dim+v_head_dim))")
    target = [parse_axis("S"), parse_axis("B"), parse_axis("n_heads"), NEG1]
    out = resolve_reshape(inp, target)
    assert render_shape(out) == "S·B·n_heads·(qk_head_dim+v_head_dim)"


def test_resolve_reshape_no_neg1_uses_target():
    inp = parse_shape("S·B·(2·ffn_hidden)")
    target = [parse_axis("S"), parse_axis("B"), parse_axis("ffn_hidden"), parse_axis("2")]
    out = resolve_reshape(inp, target)
    assert render_shape(out) == "S·B·ffn_hidden·2"


def test_resolve_reshape_unresolvable_neg1_returns_none():
    # two -1 → cannot resolve
    inp = parse_shape("S·B·H")
    target = [NEG1, NEG1]
    assert resolve_reshape(inp, target) is None
