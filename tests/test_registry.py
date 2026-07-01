"""Task 0.3：attn/ffn op-builder 注册表骨架测试。"""
from cost_eval.model_spec import DimTable, OpSpec
from cost_eval.layers.registry import ATTN_REGISTRY, FFN_REGISTRY
from cost_eval.layers.attention import build_gqa_attn_ops, build_mla_attn_ops
from cost_eval.layers.ffn import build_dense_ffn_ops, build_moe_ffn_ops

# MLA + MoE 维度，覆盖两条注册表
D = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=192, S=4096, B=1,
             vocab=129280, n_layers=6, n_experts=8, topk=4, n_shared=1, moe_F=1024,
             q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64,
             qk_nope_head_dim=128, v_head_dim=192, moe_shared_F=1024, dtype_bytes=2)


def _is_op_list(ops):
    return isinstance(ops, list) and len(ops) > 0 and all(isinstance(o, OpSpec) for o in ops)


def test_attn_registry_callables_return_op_lists():
    for key in ("gqa", "mla"):
        builder = ATTN_REGISTRY[key]
        assert callable(builder)
        assert _is_op_list(builder(D))


def test_ffn_registry_callables_return_op_lists():
    for key in ("dense", "moe"):
        builder = FFN_REGISTRY[key]
        assert callable(builder)
        assert _is_op_list(builder(D))


def test_registry_binds_expected_builders():
    # mha 复用 gqa builder（对称 GQA）
    assert ATTN_REGISTRY["mha"] is build_gqa_attn_ops
    assert ATTN_REGISTRY["gqa"] is build_gqa_attn_ops
    assert ATTN_REGISTRY["mla"] is build_mla_attn_ops
    assert FFN_REGISTRY["dense"] is build_dense_ffn_ops
    assert FFN_REGISTRY["moe"] is build_moe_ffn_ops


def test_unimplemented_variants_absent_or_placeholder():
    # dsv4_hybrid（Phase 2 填）：要么不在表里，要么占位 builder 明确报 NotImplementedError
    for key in ("dsv4_hybrid",):
        if key in ATTN_REGISTRY:
            try:
                ATTN_REGISTRY[key](D)
                assert False, f"{key} 占位应 raise NotImplementedError"
            except NotImplementedError:
                pass
