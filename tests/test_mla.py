"""Tests for MLA op-graph builders (Task C — TDD red phase first).

Covers:
- Task A: DimTable MLA fields (defaults + as_dict)
- Task B: build_mla_attn_ops / build_mla_dense_decoder / build_mla_moe_decoder
- Integration: ShapeEval.resolve over a ModelSpec with mla_dense + mla_moe layers
"""
import pytest
from cost_eval.model_spec import DimTable, OpType, ModelSpec
from cost_eval.shape_eval import eval_expr, ShapeEval
from cost_eval.specs import ParallelConfig
from cost_eval.parallel_model import ParallelModel

# ── Small dims for MLA (all chosen to be divisible by tp=1, sp=1) ──────────────
DM = DimTable(
    H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=2, vocab=10, n_layers=2,
    n_experts=8, topk=2, moe_F=16, moe_shared_F=8,
    q_lora_rank=4, kv_lora_rank=4, qk_rope_head_dim=2, qk_nope_head_dim=2, v_head_dim=3,
)


# ---------------------------------------------------------------------------
# Task A — DimTable MLA fields
# ---------------------------------------------------------------------------

def test_dimtable_mla_fields_default_zero():
    """New MLA fields must exist and default to 0."""
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)
    assert d.q_lora_rank == 0
    assert d.kv_lora_rank == 0
    assert d.qk_rope_head_dim == 0
    assert d.qk_nope_head_dim == 0
    assert d.v_head_dim == 0
    assert d.moe_shared_F == 0


def test_dimtable_mla_fields_in_as_dict():
    """as_dict() must include the new MLA fields."""
    d = DimTable(
        H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2,
        q_lora_rank=4, kv_lora_rank=4, qk_rope_head_dim=2, qk_nope_head_dim=2,
        v_head_dim=3, moe_shared_F=8,
    )
    d_dict = d.as_dict()
    assert "q_lora_rank" in d_dict
    assert "kv_lora_rank" in d_dict
    assert "qk_rope_head_dim" in d_dict
    assert "qk_nope_head_dim" in d_dict
    assert "v_head_dim" in d_dict
    assert "moe_shared_F" in d_dict
    assert d_dict["moe_shared_F"] == 8


def test_dimtable_existing_fields_unchanged():
    """Existing fields still present and eval_expr resolves them correctly."""
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)
    assert eval_expr("H", d) == 8
    assert eval_expr("n_heads*head_dim", d) == 8


# ---------------------------------------------------------------------------
# Task B — build_mla_attn_ops
# ---------------------------------------------------------------------------

def test_build_mla_attn_ops_last_output_is_h1():
    """The final op in the MLA attn segment must output a tensor named 'h1'."""
    from cost_eval.layers.mla import build_mla_attn_ops
    ops = build_mla_attn_ops(DM)
    assert ops[-1].output.name == "h1"
    assert ops[-1].output.shard == {0: "sp"}


def test_build_mla_attn_ops_length():
    """MLA attn segment: ln1 + linear_qkv + q_a_norm + kv_a_norm + linear_qb
    + linear_kvb + rope + flash + o_proj + add1 = 10 ops."""
    from cost_eval.layers.mla import build_mla_attn_ops
    ops = build_mla_attn_ops(DM)
    assert len(ops) == 10


def test_build_mla_attn_ops_has_flash_with_saves():
    """flash op must be present and have q/kv/attn/lse in saves."""
    from cost_eval.layers.mla import build_mla_attn_ops
    ops = build_mla_attn_ops(DM)
    types = [op.type for op in ops]
    assert OpType.FLASH_ATTN in types
    flash = next(op for op in ops if op.type == OpType.FLASH_ATTN)
    assert len(flash.saves) == 4    # qb_out, kvb_out, attn_out, lse


def test_mla_param_numel_matches_formula():
    """Σ param numel must equal the closed-form expression for MLA attn weights."""
    from cost_eval.layers.mla import build_mla_attn_ops
    ops = build_mla_attn_ops(DM)
    from math import prod
    total = sum(
        prod(eval_expr(e, DM) for e in w.shape)
        for op in ops
        for w in op.params
    )
    expected = (
        DM.H * (DM.q_lora_rank + DM.kv_lora_rank + DM.qk_rope_head_dim)           # linear_qkv
        + DM.q_lora_rank * DM.n_heads * (DM.qk_nope_head_dim + DM.qk_rope_head_dim)  # linear_qb
        + DM.kv_lora_rank * DM.n_heads * (DM.qk_nope_head_dim + DM.v_head_dim)       # linear_kvb
        + DM.n_heads * DM.v_head_dim * DM.H                                            # o_proj
        + DM.H + DM.q_lora_rank + DM.kv_lora_rank    # P1-01: ln1/q_a_norm/kv_a_norm gamma
    )
    assert total == expected


def test_mla_attn_o_proj_is_partial_tp():
    """o_proj output must be partial='tp' (row-parallel, pre-allreduce/RS)."""
    from cost_eval.layers.mla import build_mla_attn_ops
    ops = build_mla_attn_ops(DM)
    o_proj = next(op for op in ops if op.name == "o_proj")
    assert o_proj.output.partial == "tp"


# ---------------------------------------------------------------------------
# Task B — build_mla_dense_decoder
# ---------------------------------------------------------------------------

def test_build_mla_dense_decoder_is_layer_spec():
    from cost_eval.layers.mla import build_mla_dense_decoder
    from cost_eval.model_spec import LayerSpec
    layer = build_mla_dense_decoder(DM)
    assert isinstance(layer, LayerSpec)


def test_build_mla_dense_decoder_ends_with_dense_ffn():
    """dense decoder: 10 attn ops + 5 dense FFN ops = 15 total."""
    from cost_eval.layers.mla import build_mla_dense_decoder
    layer = build_mla_dense_decoder(DM)
    assert len(layer.ops) == 15
    # last op is add2 from dense FFN
    assert layer.ops[-1].name == "add2"


# ---------------------------------------------------------------------------
# Task B — build_mla_moe_decoder
# ---------------------------------------------------------------------------

def test_mla_moe_decoder_has_moe_gemm():
    """MoE FFN must include MOE_GEMM ops."""
    from cost_eval.layers.mla import build_mla_moe_decoder
    layer = build_mla_moe_decoder(DM)
    types = [op.type for op in layer.ops]
    assert OpType.MOE_GEMM in types


def test_mla_moe_decoder_has_shared_expert():
    """Shared expert (3 non-ep MATMUL ops) must appear after combine."""
    from cost_eval.layers.mla import build_mla_moe_decoder
    layer = build_mla_moe_decoder(DM)
    combine_idx = next(
        i for i, op in enumerate(layer.ops) if op.type == OpType.COMBINE
    )
    post_combine_ops = layer.ops[combine_idx + 1:]
    # 3 shared expert ops + moe_add(2026-07-11 补边:routed+shared 合流,零字节)
    assert len(post_combine_ops) == 4
    # The two matmul ones must not use ep sharding
    shared_matmuls = [op for op in post_combine_ops if op.type == OpType.MATMUL]
    assert len(shared_matmuls) == 2
    for op in shared_matmuls:
        for w in op.params:
            assert "ep" not in w.shard.values(), f"{w.name} should not be ep-sharded"


def test_mla_moe_decoder_total_op_count():
    """10 attn + 6 moe FFN + 3 shared expert + 1 moe_add(合流,2026-07-11 补边) = 20 ops total."""
    from cost_eval.layers.mla import build_mla_moe_decoder
    layer = build_mla_moe_decoder(DM)
    assert len(layer.ops) == 20


# ---------------------------------------------------------------------------
# Integration — ShapeEval.resolve over MLA layers
# ---------------------------------------------------------------------------

def test_mla_resolve_no_error():
    """ShapeEval.resolve must not raise for a ModelSpec with mla_dense + mla_moe layers."""
    from cost_eval.layers.mla import build_mla_dense_decoder, build_mla_moe_decoder
    spec = ModelSpec(
        "mla_test", DM,
        layer_pattern=["mla_dense", "mla_moe"],
        layer_specs={
            "mla_dense": build_mla_dense_decoder(DM),
            "mla_moe": build_mla_moe_decoder(DM),
        },
    )
    pm = ParallelModel(ParallelConfig(dp_shard=2), n_layers=2, world_size=2)
    g = ShapeEval().resolve(spec, pm)
    # Both layers land in stage 0 (single stage, pp=1)
    assert 0 in g.stages
    assert len(g.stages[0]) == 2


def test_mla_resolve_flash_workspace():
    """flash op workspace = softmax LSE 机理公式 64·B·n_heads·S bytes（∝ S·n_heads）。

    Ascend FlashAttentionScore 返回 softmax_max+softmax_sum，各 [B,n_heads,S,8] fp32
    （flash_attention.py:136-196）→ 2×8×4B×B·n_heads·S。取代旧 S·B·n_heads·v_head_dim 近似。
    """
    from cost_eval.layers.mla import build_mla_attn_ops
    from cost_eval.shape_eval import eval_expr
    ops = build_mla_attn_ops(DM)
    flash = next(op for op in ops if op.type == OpType.FLASH_ATTN)
    ws = eval_expr(flash.workspace, DM)
    assert ws == 64 * DM.B * DM.n_heads * DM.S
