"""M1 层级 op 图构建测试（Task 8: dense_decoder，Task 9: moe_decoder）。"""
from cost_eval.model_spec import DimTable, OpType
from cost_eval.shape_eval import eval_expr

D = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)


# ── Task 8: dense_decoder ────────────────────────────────────────────────────

from cost_eval.layers.dense import build_dense_decoder


def test_dense_param_numel_matches_formula():
    layer = build_dense_decoder(D)
    total = sum(
        eval_expr(w.shape[0], D) * eval_expr(w.shape[1], D)
        for op in layer.ops
        for w in op.params
    )
    qkv = D.H * (D.n_heads + 2 * D.n_kv) * D.head_dim
    o = D.n_heads * D.head_dim * D.H
    fc1 = D.H * 2 * D.F
    fc2 = D.F * D.H
    assert total == qkv + o + fc1 + fc2


def test_dense_has_flash_attn_and_saves():
    layer = build_dense_decoder(D)
    types = [op.type for op in layer.ops]
    assert OpType.FLASH_ATTN in types
    fa = next(op for op in layer.ops if op.type == OpType.FLASH_ATTN)
    assert any(s.name.startswith("qkv") or s.name in ("q", "k", "v") for s in fa.saves)
