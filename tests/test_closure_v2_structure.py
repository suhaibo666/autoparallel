"""闭环审计 v2（closure_audit_verification_2026-07-15 §4.5/§4.6）反例转正式回归。

三轮审计探针实证：上一轮只补了 `o_groups=0`、`topk<=0`、`capacity<=0` 三个边界，
但**负 o_groups**（§4.5）与**核心维度负/零**（§4.6）仍能建图/解析、静默产负参数 numel /
零激活 / 退化层图。本文件把这些反例逐条转成 `_validate_structure` 守卫的回归：

- ① P1-10：`dsv4_hybrid` 的 `o_groups` 须 **>0**（不只 !=0）——`o_groups=-1` 此前两条件
  都不拒、最终 `o_group_out.local_numel=-4194304` / `o_w=-1835008`（§4.5 探针）。
- ② P1-11：核心维度统一正值 validator——`num_layers/hidden_size/num_attention_heads/
  vocab_size/seq_length/batch_size` ≥1；`ffn_hidden_size`（若非 None）≥1；MoE 相关
  `num_moe_experts`（若用）≥1、`moe_ffn_hidden_size`（若非 None）≥1；MLA 家族维度
  （q_lora_rank/kv_lora_rank/qk_rope/qk_nope/v_head_dim）在 attn_type ∈ {mla,dsv4_hybrid,dsa}
  下须 >0（gqa/mha 的 MLA 维惰性为 0，**不查**）。

**正向守卫**：所有真实预设（deepseek_v3(4)/deepseek_v4(4)/llama/qwen2/mixtral）+ 最小
gqa/mha/dsa 配置都必须继续 build 成功——新校验绝不误伤合法配置。
"""
import dataclasses

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.llm_config import LLMConfig
from cost_eval.presets import deepseek_v3, deepseek_v4, llama, mixtral, qwen2


# ── helpers ──────────────────────────────────────────────────────────────────
def _dsv3(**over):
    return dataclasses.replace(deepseek_v3(4), **over)


def _dsv4(**over):
    return dataclasses.replace(deepseek_v4(4), **over)


def _min_gqa(**over):
    """最小合法 gqa dense 配置（MLA 维全 0 惰性，不该被 MLA-维校验误伤）。"""
    base = dict(num_layers=2, hidden_size=8, num_attention_heads=2, vocab_size=16,
                seq_length=16, batch_size=1, head_dim=4, attn_type="gqa", ffn_hidden_size=16)
    base.update(over)
    return LLMConfig(**base)


def _min_mha(**over):
    return _min_gqa(attn_type="mha", **over)


def _full_dsa(**over):
    """完整合法 dsa 配置（MLA 维 + 三个 indexer 维均 >0）。"""
    return _dsv3(attn_type="dsa", dsa_indexer_n_heads=64,
                 dsa_indexer_head_dim=128, dsa_indexer_topk=2048, **over)


# ══════════════════════════════════════════════════════════════════════════════
# 正向守卫：所有预设 + 最小配置必须继续 build 成功（新校验不得误伤）
# ══════════════════════════════════════════════════════════════════════════════
def test_all_presets_still_build():
    for cfg in (deepseek_v3(4), deepseek_v4(4), llama(num_layers=2),
                qwen2(num_layers=2), mixtral(num_layers=2)):
        assert build_llm_spec(cfg), f"预设 build 失败：{cfg.attn_type}"


def test_deepseek_v3_and_v4_various_layers_build():
    for n in (1, 2, 3, 4, 8):
        assert build_llm_spec(deepseek_v3(n))
        assert build_llm_spec(deepseek_v4(n))


def test_min_gqa_and_mha_build():
    assert build_llm_spec(_min_gqa())
    assert build_llm_spec(_min_mha())


def test_full_dsa_builds():
    assert build_llm_spec(_full_dsa())


def test_gqa_mha_lazy_mla_dims_not_rejected():
    """gqa/mha 的 MLA 维默认全 0（惰性），**绝不**被 MLA-维 >0 校验拦下。"""
    assert build_llm_spec(_min_gqa(q_lora_rank=0, kv_lora_rank=0, v_head_dim=0))
    assert build_llm_spec(_min_mha(q_lora_rank=0, kv_lora_rank=0, v_head_dim=0))


# ══════════════════════════════════════════════════════════════════════════════
# ① P1-10（§4.5）：dsv4_hybrid o_groups 须 >0——负数也 fail-loud
# ══════════════════════════════════════════════════════════════════════════════
def test_dsv4_negative_o_groups_rejected():
    with pytest.raises(ValueError, match="o_groups"):
        build_llm_spec(_dsv4(o_groups=-1))


def test_dsv4_zero_o_groups_still_rejected():
    with pytest.raises(ValueError, match="o_groups"):
        build_llm_spec(_dsv4(o_groups=0))


def test_dsv4_positive_o_groups_builds():
    # 16 是 deepseek_v4 默认；显式再验一次正向路径不被误伤
    assert build_llm_spec(_dsv4(o_groups=16))


def test_o_groups_error_message_has_value():
    with pytest.raises(ValueError, match="-1"):
        build_llm_spec(_dsv4(o_groups=-1))


# ══════════════════════════════════════════════════════════════════════════════
# ② P1-11（§4.6）：核心维度统一正值 validator
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("field,val", [
    ("num_layers", -1),
    ("num_layers", 0),
    ("hidden_size", -1792),
    ("hidden_size", 0),
    ("num_attention_heads", 0),
    ("num_attention_heads", -8),
    ("vocab_size", 0),
    ("vocab_size", -1),
    ("seq_length", 0),
    ("seq_length", -4096),
    ("batch_size", 0),
    ("batch_size", -1),
])
def test_core_dim_nonpositive_rejected(field, val):
    """核心维度 <1 → fail-loud（报错含字段名）。head_dim=192 避开 head_dim floor 分支。"""
    with pytest.raises(ValueError, match=field):
        build_llm_spec(_dsv3(**{field: val}, head_dim=192))


def test_core_dim_error_message_has_value():
    with pytest.raises(ValueError, match="-1792"):
        build_llm_spec(_dsv3(hidden_size=-1792, head_dim=192))


def test_num_attention_heads_zero_no_divzero():
    """n_heads=0 此前在 head_dim floor 的 H % n_heads 处会裸除零/退化——正值校验须先拦。"""
    with pytest.raises(ValueError, match="num_attention_heads"):
        build_llm_spec(_dsv3(num_attention_heads=0))   # head_dim=None → 会触到取模


def test_ffn_hidden_size_nonpositive_rejected():
    with pytest.raises(ValueError, match="ffn_hidden_size"):
        build_llm_spec(_dsv3(ffn_hidden_size=0))
    with pytest.raises(ValueError, match="ffn_hidden_size"):
        build_llm_spec(_dsv3(ffn_hidden_size=-3072))


def test_ffn_hidden_size_none_allowed():
    """ffn_hidden_size=None（默认 4H）合法，不该被 ≥1 校验误伤。"""
    assert build_llm_spec(_min_gqa(ffn_hidden_size=None))


# ── MoE 相关 ──────────────────────────────────────────────────────────────────
def test_negative_num_moe_experts_rejected():
    with pytest.raises(ValueError, match="num_moe_experts"):
        build_llm_spec(_dsv3(num_moe_experts=-8))


def test_moe_ffn_hidden_size_nonpositive_rejected():
    with pytest.raises(ValueError, match="moe_ffn_hidden_size"):
        build_llm_spec(_dsv3(moe_ffn_hidden_size=0))
    with pytest.raises(ValueError, match="moe_ffn_hidden_size"):
        build_llm_spec(_dsv3(moe_ffn_hidden_size=-1024))


def test_existing_topk_capacity_guards_preserved():
    """既有 topk<=0 / capacity<=0 guard 不得被删。"""
    with pytest.raises(ValueError, match="moe_router_topk"):
        build_llm_spec(_dsv3(moe_router_topk=-1))
    with pytest.raises(ValueError, match="moe_capacity_factor"):
        build_llm_spec(_dsv3(moe_capacity_factor=0))


# ── MLA 家族维度 >0（仅 mla/dsv4_hybrid/dsa）─────────────────────────────────
@pytest.mark.parametrize("field", [
    "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
])
def test_mla_dims_zero_rejected_for_mla(field):
    with pytest.raises(ValueError, match=field):
        build_llm_spec(_dsv3(**{field: 0}))


@pytest.mark.parametrize("field", [
    "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
])
def test_mla_dims_negative_rejected_for_mla(field):
    with pytest.raises(ValueError, match=field):
        build_llm_spec(_dsv3(**{field: -64}))


def test_mla_dims_zero_rejected_for_dsv4():
    with pytest.raises(ValueError, match="v_head_dim"):
        build_llm_spec(_dsv4(v_head_dim=0))
    with pytest.raises(ValueError, match="q_lora_rank"):
        build_llm_spec(_dsv4(q_lora_rank=0))


def test_mla_dims_zero_rejected_for_dsa():
    """dsa 也用 MLA 维（layers/dsa.py 符号表达式）→ 须 >0。indexer 三维保持 >0。"""
    with pytest.raises(ValueError, match="kv_lora_rank"):
        build_llm_spec(_full_dsa(kv_lora_rank=0))


def test_mla_dim_error_message_has_attn_type_and_value():
    with pytest.raises(ValueError, match="mla"):
        build_llm_spec(_dsv3(q_lora_rank=0))
