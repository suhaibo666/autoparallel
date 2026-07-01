"""Task 0.2：LLMConfig 数据结构 + to_dimtable 映射测试。"""
from cost_eval.llm_config import LLMConfig, to_dimtable


def test_defaults_and_dimtable():
    c = LLMConfig(num_layers=4, hidden_size=1792, num_attention_heads=8,
                  vocab_size=129280, seq_length=4096)
    assert c.attn_type == "gqa" and c.residual_variant == "plain"
    d = to_dimtable(c)
    assert d.H == 1792 and d.vocab == 129280 and d.S == 4096
    # head_dim 默认 hidden_size // num_attention_heads
    assert d.head_dim == 1792 // 8
    # n_kv 默认 = n_heads（num_query_groups=None → MHA/对称 GQA）
    assert d.n_heads == 8 and d.n_kv == 8
    # n_layers = num_layers + 2（含 embedding + head），与 validate_dsv3 一致
    assert d.n_layers == 6


def test_mla_and_moe_dims_map_through():
    c = LLMConfig(num_layers=4, hidden_size=1792, num_attention_heads=8,
                  vocab_size=129280, seq_length=4096,
                  attn_type="mla", q_lora_rank=1536,
                  num_moe_experts=8, moe_router_topk=4)
    d = to_dimtable(c)
    assert d.q_lora_rank == 1536
    assert d.n_experts == 8 and d.topk == 4


def test_frozen_dataclass():
    import dataclasses
    c = LLMConfig(num_layers=1, hidden_size=8, num_attention_heads=2,
                  vocab_size=10, seq_length=4)
    assert dataclasses.is_dataclass(c)
    try:
        c.hidden_size = 16  # frozen → 应抛 FrozenInstanceError
        assert False, "LLMConfig 应为 frozen"
    except dataclasses.FrozenInstanceError:
        pass
