"""Task 0.3 / Task 2：attn/ffn op-builder 注册表——统一 (dims, ctx) API。

Task 2 起，所有注册入口签名统一为 ``REGISTRY[type](dims, ctx)``：
  - `gqa`/`mla`/`dense`/`moe` 是把原 `(DimTable)`-only builder 适配的薄 wrapper，忽略 ctx；
  - `dsv4_hybrid` 从 ``ctx.compress_ratio`` 读 per-layer 压缩比（去掉旧的双签名特例 wrapper）。
底层 `(DimTable)`-only builder 仍可直接调用（`__wrapped__` 暴露），供 validate_dsv3/其它测试用。
"""
from cost_eval.model_spec import DimTable, OpSpec
from cost_eval.layer_context import LayerContext
from cost_eval.layers.registry import ATTN_REGISTRY, FFN_REGISTRY
from cost_eval.layers.attention import build_gqa_attn_ops, build_mla_attn_ops
from cost_eval.layers.ffn import build_dense_ffn_ops, build_moe_ffn_ops
from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops

# MLA + MoE 维度，覆盖两条注册表
D = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=192, S=4096, B=1,
             vocab=129280, n_layers=6, n_experts=8, topk=4, n_shared=1, moe_F=1024,
             q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64,
             qk_nope_head_dim=128, v_head_dim=192, moe_shared_F=1024, dtype_bytes=2)


def _is_op_list(ops):
    return isinstance(ops, list) and len(ops) > 0 and all(isinstance(o, OpSpec) for o in ops)


def _names(ops):
    return [o.name for o in ops]


def _dec(attn, ffn="dense", ratio=None):
    return LayerContext(kind="decoder", attn_type=attn, compress_ratio=ratio, ffn_type=ffn)


# ── 统一 (dims, ctx) API ────────────────────────────────────────────────────────

def test_attn_registry_callables_return_op_lists():
    for key in ("gqa", "mla"):
        builder = ATTN_REGISTRY[key]
        assert callable(builder)
        assert _is_op_list(builder(D, _dec(key)))


def test_ffn_registry_callables_return_op_lists():
    for key in ("dense", "moe"):
        builder = FFN_REGISTRY[key]
        assert callable(builder)
        assert _is_op_list(builder(D, _dec("mla", key)))


def test_uniform_builders_ignore_ctx_and_match_underlying():
    # gqa/mla/dense/moe 忽略 ctx，等价于直接调底层 (DimTable)-only builder
    assert _names(ATTN_REGISTRY["gqa"](D, _dec("gqa"))) == _names(build_gqa_attn_ops(D))
    assert _names(ATTN_REGISTRY["mla"](D, _dec("mla"))) == _names(build_mla_attn_ops(D))
    assert _names(FFN_REGISTRY["dense"](D, _dec("mla", "dense"))) == _names(build_dense_ffn_ops(D))
    assert _names(FFN_REGISTRY["moe"](D, _dec("mla", "moe"))) == _names(build_moe_ffn_ops(D))


def test_registry_binds_expected_builders():
    # mha == 对称 GQA：复用同一 (dims, ctx) adapter 实例
    assert ATTN_REGISTRY["mha"] is ATTN_REGISTRY["gqa"]
    # adapter 底层仍是原 (DimTable)-only builder（保持可直接调用）
    assert ATTN_REGISTRY["gqa"].__wrapped__ is build_gqa_attn_ops
    assert ATTN_REGISTRY["mla"].__wrapped__ is build_mla_attn_ops
    assert FFN_REGISTRY["dense"].__wrapped__ is build_dense_ffn_ops
    assert FFN_REGISTRY["moe"].__wrapped__ is build_moe_ffn_ops


# ── dsv4_hybrid：从 ctx.compress_ratio 读比、按比分支（不再是双签名特例）────────────

def test_dsv4_registry_reads_ratio_from_ctx():
    # ratio 128 → HCA：有 compressor、无 indexer；ratio 4 → CSA：有 indexer
    n128 = _names(ATTN_REGISTRY["dsv4_hybrid"](D, _dec("dsv4_hybrid", "moe", 128)))
    n4 = _names(ATTN_REGISTRY["dsv4_hybrid"](D, _dec("dsv4_hybrid", "moe", 4)))
    assert "compressor" in n128 and "indexer" not in n128
    assert "indexer" in n4 and "compressor" in n4 and "sparse_attn" in n4
    # 等价于直接调底层 (dims, ratio) builder
    assert n128 == _names(build_dsv4_hybrid_attn_ops(D, 128))
    assert n4 == _names(build_dsv4_hybrid_attn_ops(D, 4))


def test_dsv4_registry_ratio_zero_is_mla_base():
    # ratio 0/1 → 滑窗 == MLA base（§7.4 内存中性）
    n0 = _names(ATTN_REGISTRY["dsv4_hybrid"](D, _dec("dsv4_hybrid", "moe", 0)))
    assert n0 == _names(build_mla_attn_ops(D))


def test_dsv4_registry_without_ratio_raises_clearly():
    # 缺 compress_ratio（单参 / 非 LayerContext）→ 清晰 NotImplementedError，不静默产错图
    import pytest
    assert "dsv4_hybrid" in ATTN_REGISTRY
    with pytest.raises(NotImplementedError):
        ATTN_REGISTRY["dsv4_hybrid"](D)
    with pytest.raises(NotImplementedError):
        ATTN_REGISTRY["dsv4_hybrid"](D, LayerContext(kind="decoder", attn_type="dsv4_hybrid"))
