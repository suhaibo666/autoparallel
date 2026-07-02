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
