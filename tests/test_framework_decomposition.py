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
    # C3（2026-07-15）：flash workspace 改 TensorRef 型 workspace_ref（按 TP 切）——检查它是
    # [2,B,n_heads,S,8] fp32 = 64·B·n_heads·S 字节（TP=1/CP=1 时与旧 FLASH_LSE_WS 同值），
    # 且 head 维标 tp shard（此前字符串 workspace 不按 TP 切，P1-09 audit）。
    from cost_eval.shape_eval import resolve_tensor
    from cost_eval.parallel_model import ParallelModel
    from cost_eval.specs import ParallelConfig
    pm = ParallelModel(ParallelConfig(), n_layers=DM.n_layers, world_size=1)
    for build in (build_gqa_attn_ops, build_mla_attn_ops):
        from cost_eval.model_spec import OpType
        flash = next(op for op in build(DM) if op.type == OpType.FLASH_ATTN)
        assert flash.workspace is None and flash.workspace_ref is not None
        assert flash.workspace_ref.shard == {2: "tp"}          # head 维按 TP 切
        rt = resolve_tensor(flash.workspace_ref, DM, pm)
        assert rt.local_numel * rt.dtype_bytes == 64 * DM.B * DM.n_heads * DM.S


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
        assert sp.workspace_ref is not None and sp.workspace_ref.shard == {2: "tp"}   # C3: TP 切


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


# ===========================================================================
# T4 — framework_reserve → documented allocator block-rounding FORMULA (no blob)
# ===========================================================================
# The old `framework_reserve` (calibrated 2197→177→63 MiB) is decomposed: FSDP
# prefetch → gather_buf; flash-ws → flash workspace; MoE-staging → dispatch/combine
# workspace. What physically remains in the ALLOCATED peak is per-allocation pool
# alignment — a documented HW property, NOT a fit. MindSpore DynamicMemPool aligns
# each allocation to `kDynamicMemAlignSize` (512B). framework_reserve is now 0 by
# default (the mechanism pieces live in the op graph); the calibrated arg is kept
# only as an AUDIT/regression knob (reproduces the old lumped behaviour).

def test_hardware_alloc_block_bytes_documented_default():
    from cost_eval.specs import HardwareSpec
    hw = HardwareSpec(max_device_memory=1)
    # MindSpore DynamicMemPoolBestFit kDynamicMemAlignSize (platform property, not a fit).
    assert hw.alloc_block_bytes == 512


def test_framework_reserve_default_zero_no_fitted_blob():
    from cost_eval.framework import framework_reserve
    from cost_eval.specs import ParallelConfig
    # No hand-set MiB residual: default framework_reserve is exactly 0.
    assert framework_reserve(ParallelConfig()) == 0


def test_framework_reserve_calibrated_arg_is_audit_regression_path():
    """Explicit calibrated reserve is still returned verbatim (auditable old behaviour)."""
    from cost_eval.framework import framework_reserve
    from cost_eval.specs import ParallelConfig
    assert framework_reserve(ParallelConfig(), 177 * 2 ** 20) == 177 * 2 ** 20


def test_structure_mem_rounds_each_tensor_up_to_alloc_block():
    """Allocator FORMULA: each live tensor's bytes rounded up to the pool block."""
    from cost_eval.model_spec import OpSpec, OpType, TensorRef
    from cost_eval.shape_eval import ShapeEval
    from cost_eval.structure_mem import estimate_structure_memory
    from cost_eval.parallel_model import ParallelModel
    from cost_eval.specs import ParallelConfig
    from cost_eval.model_spec import ModelSpec, LayerSpec, DimTable as DT
    # one save of 100 bytes (odd) → rounds up to 512 with a 512B block.
    d = DT(H=100, F=1, n_heads=1, n_kv=1, head_dim=1, S=1, B=1, vocab=1, n_layers=1, dtype_bytes=1)
    x = TensorRef("x", ("S", "B", "H"))
    op = OpSpec("op", OpType.NORM, [x], TensorRef("y", ("S", "B", "H")), saves=[x])
    spec = ModelSpec("m", d, ["l"], {"l": LayerSpec([op])})
    g = ShapeEval().resolve(spec, ParallelModel(ParallelConfig(), 1, 1))
    ops = g.stages[0][0].ops
    assert estimate_structure_memory(ops).activation_saves == 100                 # block=1 → no rounding
    assert estimate_structure_memory(ops, alloc_block_bytes=512).activation_saves == 512


def test_dsv3_tensors_already_block_aligned_so_reserve_is_zero():
    """DSv3's modeled tensors are 512-aligned → the allocator formula adds 0; the
    old 63 MiB was over-attribution, not real allocated block fragmentation."""
    from cost_eval.presets import deepseek_v3
    from cost_eval.build_llm import build_llm_spec
    from cost_eval.structure_mem import estimate_structure_memory
    from cost_eval.shape_eval import ShapeEval
    from cost_eval.parallel_model import ParallelModel
    from cost_eval.specs import ParallelConfig
    spec = build_llm_spec(deepseek_v3(4))
    pm = ParallelModel(ParallelConfig(dp_shard=2, sequence_parallel=True), spec.dims.n_layers, 2)
    g = ShapeEval().resolve(spec, pm)
    for layer in g.stages[0]:
        base = estimate_structure_memory(layer.ops, grad_dtype_bytes=4)
        rounded = estimate_structure_memory(layer.ops, grad_dtype_bytes=4, alloc_block_bytes=512)
        assert base.activation_saves == rounded.activation_saves
        assert base.param_full_bytes == rounded.param_full_bytes
