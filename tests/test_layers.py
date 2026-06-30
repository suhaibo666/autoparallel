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


# ── Task 9: moe_decoder ──────────────────────────────────────────────────────

from cost_eval.layers.moe import build_moe_decoder

DM = DimTable(
    H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10,
    n_layers=2, n_experts=8, topk=2, moe_F=16,
)


def test_moe_expert_weight_is_ep_only():
    """专家权重只含 ep 切分，不含 tp（纯 EP 设计）。"""
    layer = build_moe_decoder(DM)
    gemms = [op for op in layer.ops if op.type == OpType.MOE_GEMM]
    assert gemms, "should have MOE_GEMM ops"
    for op in gemms:
        for w in op.params:
            assert "ep" in w.shard.values(), f"{w.name} 缺少 ep shard"
            assert "tp" not in w.shard.values(), f"{w.name} 不应含 tp shard"


def test_moe_reuses_dense_attn_segment():
    """moe_decoder 前 6 个 op 与 dense_decoder 前 6 个 op 名称相同（复用 attn 段）。"""
    dense = build_dense_decoder(DM)
    moe   = build_moe_decoder(DM)
    dense_names = [op.name for op in dense.ops[:6]]
    moe_names   = [op.name for op in moe.ops[:6]]
    assert dense_names == moe_names, f"attn 段不同: {dense_names} vs {moe_names}"


def test_moe_has_dispatch_and_combine():
    """FFN 段必须包含 dispatch（all-to-all）和 combine。"""
    layer = build_moe_decoder(DM)
    types = {op.type for op in layer.ops}
    assert OpType.DISPATCH in types
    assert OpType.COMBINE  in types
