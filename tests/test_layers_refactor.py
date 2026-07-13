"""Task 0.1：命名 attn/ffn op-builder 抽取的行为不变性回归测试。

验证从 dense/moe/mla 里抽出的 build_gqa_attn_ops / build_dense_ffn_ops /
build_moe_ffn_ops / build_mla_attn_ops 与旧的切片写法逐 op 逐字段一致（byte-identical）。
"""
from cost_eval.model_spec import DimTable
from cost_eval.layers.attention import build_gqa_attn_ops, build_mla_attn_ops
from cost_eval.layers.ffn import build_dense_ffn_ops, build_moe_ffn_ops
from cost_eval.layers import dense, moe, mla   # old paths still work

D = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=192, S=4096, B=1,
             vocab=129280, n_layers=6, dtype_bytes=2)

# MoE + MLA 组合层需要的额外维度（供 moe/mla FFN builder 求值）
DM = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=192, S=4096, B=1,
              vocab=129280, n_layers=6, dtype_bytes=2,
              n_experts=8, topk=2, moe_F=1536, moe_shared_F=1536,
              q_lora_rank=1536, kv_lora_rank=512,
              qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128)


def _names(ops):
    return [o.name for o in ops]


def _tref_key(t):
    """TensorRef → 可比较的全字段快照。"""
    if t is None:
        return None
    return (t.name, tuple(t.shape), tuple(sorted(t.shard.items())),
            t.is_weight, t.partial, t.dtype_bytes)


def _op_key(op):
    """OpSpec → 全字段快照（type/inputs/output/params/saves/workspace/bwd_scratch/attrs）。"""
    return (
        op.name,
        op.type,
        tuple(_tref_key(i) for i in op.inputs),
        _tref_key(op.output),
        tuple(_tref_key(p) for p in op.params),
        tuple(_tref_key(s) for s in op.saves),
        op.workspace,
        op.bwd_scratch,
        tuple(sorted(op.attrs.items())),
    )


def _assert_ops_identical(new_ops, old_ops):
    assert len(new_ops) == len(old_ops)
    for n, o in zip(new_ops, old_ops):
        assert _op_key(n) == _op_key(o), f"op {n.name} 字段不一致"


# ── 名字级别一致性（task 指定的三条）─────────────────────────────────────────
def test_gqa_attn_ops_equal_old_dense_prefix():
    assert _names(build_gqa_attn_ops(D)) == _names(dense.build_dense_decoder(D).ops[:6])


def test_dense_ffn_ops_equal_old_suffix():
    assert _names(build_dense_ffn_ops(D)) == _names(dense.build_dense_decoder(D).ops[6:])


def test_mla_decoder_recomposed_equal():
    from cost_eval.layers.mla import build_mla_dense_decoder
    recomposed = _names(build_mla_attn_ops(D)) + _names(build_dense_ffn_ops(D))
    assert recomposed == _names(build_mla_dense_decoder(D).ops)


# ── 全字段 byte-identical（更强的等价性）────────────────────────────────────
def test_gqa_attn_ops_fields_identical():
    _assert_ops_identical(build_gqa_attn_ops(D), dense.build_dense_decoder(D).ops[:6])


def test_dense_ffn_ops_fields_identical():
    _assert_ops_identical(build_dense_ffn_ops(D), dense.build_dense_decoder(D).ops[6:])


def test_moe_ffn_ops_fields_identical():
    _assert_ops_identical(build_moe_ffn_ops(DM), moe.build_moe_decoder(DM).ops[6:])


def test_moe_decoder_recomposed_fields_identical():
    recomposed = build_gqa_attn_ops(DM) + build_moe_ffn_ops(DM)
    _assert_ops_identical(recomposed, moe.build_moe_decoder(DM).ops)


def test_mla_dense_decoder_fields_identical():
    recomposed = build_mla_attn_ops(DM) + build_dense_ffn_ops(DM)
    _assert_ops_identical(recomposed, mla.build_mla_dense_decoder(DM).ops)


def test_mla_moe_decoder_fields_identical():
    from cost_eval.layers.ffn import build_shared_expert_ops, build_moe_merge_op
    recomposed = (build_mla_attn_ops(DM) + build_moe_ffn_ops(DM) + build_shared_expert_ops(DM)
                  + [build_moe_merge_op(DM)])
    _assert_ops_identical(recomposed, mla.build_mla_moe_decoder(DM).ops)


def test_old_import_paths_still_work():
    # 旧公共名仍从旧模块可导入
    from cost_eval.layers.dense import build_dense_decoder
    from cost_eval.layers.moe import build_moe_decoder
    from cost_eval.layers.mla import (
        build_mla_attn_ops as mla_attn,
        build_mla_dense_decoder,
        build_mla_moe_decoder,
    )
    assert build_dense_decoder is dense.build_dense_decoder
    assert build_moe_decoder is moe.build_moe_decoder
    assert callable(build_mla_dense_decoder)
    assert callable(build_mla_moe_decoder)
    # mla.build_mla_attn_ops 与 attention.build_mla_attn_ops 是同一对象
    assert mla_attn is build_mla_attn_ops
