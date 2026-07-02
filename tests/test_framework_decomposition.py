"""Decompose the empirical `framework_reserve` blob into per-op, mechanism-grounded
FORMULAS placed IN the op graph, + model the missing backward transients.

Directive: eliminate ALL empirical hardcoded constants; make everything
formula-computable. The last blob was `framework_reserve` (DSv3 residual 63 MiB).
Each component becomes a per-op formula counted only at the event where that op is
live:

  T1  mHC / MTP backward transients  (residual.py / head.py)
  T2  flash-attention workspace       (attention.py / mla.py / dsv4_hybrid.py)
  T3  MoE all-to-all staging buffers   (ffn.py dispatch/combine)
  T4  framework_reserve -> allocator block-rounding formula (framework.py)

Every term is a FORMULA in dims with a source citation (mindformers file:line).
NO fitted MiB constants.
"""
from cost_eval.model_spec import DimTable
from cost_eval.llm_config import LLMConfig, to_dimtable
from cost_eval.shape_eval import eval_expr
from cost_eval.layers.attention import build_gqa_attn_ops, build_mla_attn_ops
from cost_eval.layers.ffn import build_dense_ffn_ops
from cost_eval.layers.residual import mhc_wrap

N = 4  # num_residual_streams

DS = DimTable(
    H=16, F=32, n_heads=2, n_kv=2, head_dim=8, S=8, B=1, vocab=32, n_layers=3,
    num_residual_streams=N,
)


def _body(d):
    return build_gqa_attn_ops(d) + build_dense_ffn_ops(d)


# ===========================================================================
# T1 — mHC backward transients (×n residual-stream gradient)
# ===========================================================================
# Source: hyper_connection.py:87-112 HyperConnectionOutputCell.construct —
#   new_streams = h_res @ x_streams + h_post * sublayer_out  ([s,b,n,H]).
# Its backward materialises the ×n packed-stream gradient grad_x_streams
# [s,b,n*H] plus the reconstructed res_part [s,b,n*H] (compute dtype). This
# transient is NOT a saved forward activation; it is a per-layer backward
# scratch counted only at the mHC layer's backward event.

def test_mhc_sinkhorn_op_carries_bwd_scratch_formula():
    wrapped = mhc_wrap(_body(DS), N, DS)
    sink = [op for op in wrapped if op.name.endswith("_hc_sinkhorn")]
    assert len(sink) == 2                       # attn_hc + ffn_hc
    for op in sink:
        # ×n packed residual-stream backward gradient (2 bf16 tensors: grad_x_streams
        # + reconstructed res_part), scaling with num_residual_streams (hyper_connection.py:102-112).
        assert op.bwd_scratch == "4*S*B*num_residual_streams*H"


def test_mhc_bwd_scratch_scales_with_n_streams():
    wrapped = mhc_wrap(_body(DS), N, DS)
    sink = next(op for op in wrapped if op.name.endswith("_hc_sinkhorn"))
    got = eval_expr(sink.bwd_scratch, DS)
    # 2 packed-stream tensors, bf16 (2B): 2 * 2 * S*B*(n*H)
    assert got == 4 * DS.S * DS.B * N * DS.H
    # n=1 (plain) would be a quarter of this — confirms the ×n dependence.
    d1 = DimTable(H=16, F=32, n_heads=2, n_kv=2, head_dim=8, S=8, B=1, vocab=32,
                  n_layers=3, num_residual_streams=1)
    assert eval_expr(sink.bwd_scratch, d1) == 4 * DS.S * DS.B * 1 * DS.H


# ===========================================================================
# T1 — MTP inner transformer uses mHC (mtp:381-399) → decoder must be wrapped
# ===========================================================================

def _mtp_cfg(**kw):
    base = dict(num_layers=2, hidden_size=16, num_attention_heads=2, vocab_size=64,
                seq_length=8, batch_size=1, attn_type="gqa", mtp_num_layers=1)
    base.update(kw)
    return LLMConfig(**base)


def test_mtp_decoder_is_mhc_wrapped_when_residual_mhc():
    from cost_eval.layers.head import build_mtp_ops
    cfg = _mtp_cfg(residual_variant="mhc", num_residual_streams=4)
    ops = build_mtp_ops(cfg)
    names = [op.name for op in ops]
    # MTP's inner transformer_layer runs on packed streams (mtp:381-399):
    # expand before, HC modules inside, collapse after.
    assert any(n.endswith("_hc_sinkhorn") for n in names)
    assert "mtp_hc_expand" in names and "mtp_hc_collapse" in names


def test_mtp_plain_residual_has_no_hc_ops():
    from cost_eval.layers.head import build_mtp_ops
    cfg = _mtp_cfg(residual_variant="plain")
    names = [op.name for op in build_mtp_ops(cfg)]
    assert not any("_hc_" in n for n in names)


# ===========================================================================
# T2 — flash-attention workspace (mechanism-grounded softmax LSE, ∝ S·n_heads)
# ===========================================================================
# Source: flash_attention.py:136-196 — MindSpore FlashAttentionScore returns
#   softmax_val (softmax_max) + softmax_sum, each [B, n_heads, S, 8] fp32 (the
#   flash inner reduce block = 8), saved for FlashAttentionScoreGrad. The
#   workspace is 2 tensors × 8 × 4B = 64·B·n_heads·S bytes — scales with the
#   sequence (the whole point) and REPLACES the ad-hoc numel-as-bytes fa_ws.

from cost_eval.layers.attention import FLASH_LSE_WS   # noqa: E402

DM = DimTable(
    H=16, F=32, n_heads=4, n_kv=2, head_dim=8, S=8, B=1, vocab=32, n_layers=3,
    q_lora_rank=8, kv_lora_rank=4, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=8,
    dsa_indexer_n_heads=2, dsa_indexer_head_dim=4, dsa_indexer_topk=4,
    o_groups=2, o_lora_rank=4, csa_window_size=2,
)


def test_flash_workspace_is_softmax_lse_formula():
    # GQA + MLA flash ops carry the mechanism LSE workspace (not the old fa_ws).
    for build in (build_gqa_attn_ops, build_mla_attn_ops):
        from cost_eval.model_spec import OpType
        flash = next(op for op in build(DM) if op.type == OpType.FLASH_ATTN)
        assert flash.workspace == FLASH_LSE_WS == "64*B*n_heads*S"


def test_flash_workspace_scales_with_seq():
    # doubling the sequence doubles the workspace — mechanism-grounded ∝ S.
    ws = eval_expr(FLASH_LSE_WS, DM)
    assert ws == 64 * DM.B * DM.n_heads * DM.S
    d2 = DimTable(H=16, F=32, n_heads=4, n_kv=2, head_dim=8, S=16, B=1, vocab=32, n_layers=3)
    assert eval_expr(FLASH_LSE_WS, d2) == 2 * ws


def test_dsv4_sparse_attn_carries_flash_workspace():
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    for ratio in (4, 128):
        ops = build_dsv4_hybrid_attn_ops(DM, ratio)
        sp = next(op for op in ops if op.name == "sparse_attn")
        assert sp.workspace == FLASH_LSE_WS


# ===========================================================================
# T3 — MoE all-to-all staging buffers (∝ dispatched_tokens · H)
# ===========================================================================
# Source: experts.py:103-146 GroupedMLP.permute — tokens are sorted by expert into
#   `routed_input` [S·B·topk, H] (the send/permute staging buffer BEFORE the ep
#   all-to-all); experts.py:149-173 unpermute scatters back (combine staging).
# The received post-a2a tokens (`disp`, [TLOCAL,H]{ep}) are already a save; the
# missing piece is the permute/scatter STAGING (compute dtype), transient during
# dispatch/combine → workspace on those ops. dispatched_tokens = S·B·topk·capacity.

MOE_STAGING_WS = "2*S*B*topk*capacity_factor*H"

DMOE = DimTable(
    H=16, F=32, n_heads=4, n_kv=2, head_dim=8, S=8, B=1, vocab=32, n_layers=3,
    n_experts=4, topk=2, moe_F=32, moe_shared_F=32,
)


def test_moe_dispatch_and_combine_carry_a2a_staging_workspace():
    from cost_eval.layers.ffn import build_moe_ffn_ops
    ops = build_moe_ffn_ops(DMOE)
    disp = next(op for op in ops if op.name == "dispatch")
    comb = next(op for op in ops if op.name == "combine")
    assert disp.workspace == MOE_STAGING_WS
    assert comb.workspace == MOE_STAGING_WS


def test_moe_staging_scales_with_dispatched_tokens():
    # send+recv permute buffer (bf16 2B): 2 · S·B·topk·capacity · H.
    ws = eval_expr(MOE_STAGING_WS, DMOE)
    assert ws == 2 * DMOE.S * DMOE.B * DMOE.topk * DMOE.H     # capacity_factor=1.0
    # doubling topk doubles the staging (∝ dispatched tokens).
    d2 = DimTable(H=16, F=32, n_heads=4, n_kv=2, head_dim=8, S=8, B=1, vocab=32,
                  n_layers=3, n_experts=4, topk=4, moe_F=32)
    assert eval_expr(MOE_STAGING_WS, d2) == 2 * ws
