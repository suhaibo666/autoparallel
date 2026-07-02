"""Task 1.1：gen_layer_pattern —— LLMConfig → 层序列 pattern 展开。

`gen_layer_pattern` 现返回 `list[LayerContext]`（结构化逐层身份）；这里对其派生的
确定性字符串标签（`ctx.name`）断言层序列形状（标签格式沿用历史命名，供 ModelSpec 用）。
"""
from cost_eval.build_llm import gen_layer_pattern
from cost_eval.llm_config import LLMConfig


def _names(cfg):
    return [c.name for c in gen_layer_pattern(cfg)]


def test_dsv3_pattern():
    c = LLMConfig(num_layers=4, hidden_size=1, num_attention_heads=1, vocab_size=1, seq_length=1,
                  attn_type="mla", num_moe_experts=8, moe_router_topk=4, first_k_dense_replace=1)
    assert _names(c) == ["embedding", "mla_dense", "mla_moe", "mla_moe", "mla_moe", "lm_head"]


def test_gqa_dense_pattern():
    c = LLMConfig(num_layers=3, hidden_size=1, num_attention_heads=1, vocab_size=1, seq_length=1,
                  attn_type="gqa")   # no MoE → all dense
    assert _names(c) == ["embedding", "gqa_dense", "gqa_dense", "gqa_dense", "lm_head"]


def test_mtp_appended():
    c = LLMConfig(num_layers=2, hidden_size=1, num_attention_heads=1, vocab_size=1, seq_length=1,
                  attn_type="mla", num_moe_experts=8, moe_router_topk=4, first_k_dense_replace=1,
                  mtp_num_layers=1)
    assert _names(c)[-2:] == ["mtp", "lm_head"]
