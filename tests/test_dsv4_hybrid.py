"""Task 2.1 — dsv4_hybrid attention op-builder (TDD).

DeepSeek-V4 hybrid compressed attention = DSA indexer + compressor + CSA/HCA
sparse attention.  Builder branches per-layer by ``compress_ratio``:

  * ratio 0/1   → sliding-window only == MLA base (memory-neutral, §7.4).
  * ratio 4     → CSA: MLA base + indexer + compressor(overlap, coff=2) + sparse.
  * ratio 128   → HCA: MLA base + compressor(non-overlap, coff=1), no top-k indexer.

Source (mindformers, faithful):
  experimental_attention_variant/{deepseek_v4_hybrid_attention,indexer,compressor,csa}.py
Design: specs/2026-07-01-unified-llm-modelspec-design.md §7.3 / §7.2 / §4.
"""
import pytest

from cost_eval.model_spec import DimTable, OpType, ModelSpec, LayerSpec
from cost_eval.llm_config import LLMConfig, to_dimtable
from cost_eval.shape_eval import eval_expr, ShapeEval
from cost_eval.specs import ParallelConfig
from cost_eval.parallel_model import ParallelModel

# ── Small dims for structural assertions (all divisible at tp=1) ───────────────
DS = DimTable(
    H=16, F=32, n_heads=2, n_kv=1, head_dim=8, S=8, B=1, vocab=32, n_layers=3,
    q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=2, qk_nope_head_dim=2, v_head_dim=4,
    dsa_indexer_n_heads=2, dsa_indexer_head_dim=4, dsa_indexer_topk=4,
    o_groups=2, o_lora_rank=4, csa_window_size=4,
)

# ── Big dims for the numeric O(S^2) / O(S*topk) scaling checks ─────────────────
DBIG = DimTable(
    H=2048, F=4096, n_heads=8, n_kv=1, head_dim=256, S=4096, B=1, vocab=1000, n_layers=4,
    q_lora_rank=512, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128,
    v_head_dim=192,
    dsa_indexer_n_heads=4, dsa_indexer_head_dim=128, dsa_indexer_topk=2048,
    o_groups=8, o_lora_rank=256, csa_window_size=128,
)


def _names(ops):
    return [o.name for o in ops]


# ---------------------------------------------------------------------------
# DimTable / to_dimtable new dsv4 symbolic dims
# ---------------------------------------------------------------------------

def test_dimtable_dsv4_fields_default_zero():
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)
    assert d.dsa_indexer_n_heads == 0
    assert d.dsa_indexer_head_dim == 0
    assert d.dsa_indexer_topk == 0
    assert d.o_groups == 0
    assert d.o_lora_rank == 0
    assert d.csa_window_size == 0


def test_dimtable_dsv4_fields_eval_expr():
    """New fields must be resolvable by eval_expr (they are DimTable symbols)."""
    assert eval_expr("dsa_indexer_topk", DS) == 4
    assert eval_expr("o_groups*o_lora_rank", DS) == 8
    assert eval_expr("B*S*dsa_indexer_topk*v_head_dim", DBIG) == 4096 * 2048 * 192


def test_to_dimtable_maps_dsv4_fields():
    cfg = LLMConfig(
        num_layers=2, hidden_size=16, num_attention_heads=2, vocab_size=32, seq_length=8,
        attn_type="dsv4_hybrid",
        v_head_dim=4, q_lora_rank=8,
        dsa_indexer_n_heads=2, dsa_indexer_head_dim=4, dsa_indexer_topk=4,
        o_groups=2, o_lora_rank=4, csa_window_size=64,
    )
    d = to_dimtable(cfg)
    assert d.dsa_indexer_n_heads == 2
    assert d.dsa_indexer_head_dim == 4
    assert d.dsa_indexer_topk == 4
    assert d.o_groups == 2
    assert d.o_lora_rank == 4
    assert d.csa_window_size == 64


# ---------------------------------------------------------------------------
# ratio 4 (CSA): indexer + compressor + sparse attention
# ---------------------------------------------------------------------------

def test_ratio4_has_indexer_compressor_sparse():
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, 4)
    names = _names(ops)
    assert "indexer" in names
    assert "compressor" in names
    assert "sparse_attn" in names
    # composes with FFN: last op outputs h1
    assert ops[-1].output.name == "h1"
    assert ops[-1].output.shard == {0: "sp"}


def test_ratio4_index_scores_bwd_scratch_is_S2():
    """Indexer materializes index_scores [B,S,S] fp32 -> O(S^2) as bwd_scratch."""
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, 4)
    idx = next(op for op in ops if op.name == "indexer")
    assert idx.bwd_scratch == "4*B*S*S"
    # numeric: S=4096, B=1 -> index_scores numel = 4096**2 (bytes = 4 * numel, fp32)
    bytes_big = eval_expr(idx.bwd_scratch, DBIG)
    assert bytes_big == 4 * 4096 * 4096
    assert bytes_big // 4 == 4096 ** 2          # numel


def test_ratio4_kv_gathered_saved_is_S_times_topk():
    """Sparse attention saves kv_gathered [B,S,topk,v_head_dim] -> O(S*topk)."""
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, 4)
    kvg = next(s for op in ops for s in op.saves if s.name == "kv_gathered")
    numel = 1
    for e in kvg.shape:
        numel *= eval_expr(e, DBIG)
    assert numel == 4096 * 2048 * 192


def test_ratio4_attn_weights_saved_is_S_times_topk():
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, 4)
    aw = next(s for op in ops for s in op.saves if s.name == "attn_weights")
    numel = 1
    for e in aw.shape:
        numel *= eval_expr(e, DBIG)
    assert numel == 1 * 8 * 4096 * 2048        # B * n_heads * S * topk


def test_ratio4_compressed_kv_first_dim_is_S_div_4():
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, 4)
    cmp = next(op for op in ops if op.name == "compressor")
    assert cmp.output.name == "compressed_kv"
    assert eval_expr(cmp.output.shape[0], DBIG) == 4096 // 4


def test_ratio4_compressor_coff2():
    """CSA overlap compressor: proj out dim = coff * v_head_dim with coff=2."""
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, 4)
    cmp = next(op for op in ops if op.name == "compressor")
    wkv = next(w for w in cmp.params if w.name == "cmp_wkv")
    assert eval_expr(wkv.shape[1], DS) == 2 * DS.v_head_dim


def test_ratio4_grouped_output_weights():
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, 4)
    wnames = [w.name for op in ops for w in op.params]
    assert "wo_group" in wnames          # linear_o_group_proj
    assert "o_w" in wnames               # linear_proj -> H


# ---------------------------------------------------------------------------
# ratio 128 (HCA): compressor(non-overlap, coff=1), NO top-k indexer
# ---------------------------------------------------------------------------

def test_ratio128_has_compressor_no_indexer():
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, 128)
    names = _names(ops)
    assert "compressor" in names
    assert "indexer" not in names
    # no O(S^2) index_scores op anywhere
    assert all(op.bwd_scratch != "4*B*S*S" for op in ops)


def test_ratio128_compressor_coff1_and_S_div_128():
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, 128)
    cmp = next(op for op in ops if op.name == "compressor")
    wkv = next(w for w in cmp.params if w.name == "cmp_wkv")
    assert eval_expr(wkv.shape[1], DS) == 1 * DS.v_head_dim
    assert eval_expr(cmp.output.shape[0], DBIG) == 4096 // 128


# ---------------------------------------------------------------------------
# ratio 0/1: MLA-base-like, no sparse additions
# ---------------------------------------------------------------------------

def test_ratio0_is_mla_base_no_sparse():
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    from cost_eval.layers.attention import build_mla_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, 0)
    names = _names(ops)
    assert "indexer" not in names
    assert "compressor" not in names
    assert "sparse_attn" not in names
    assert all(op.bwd_scratch != "4*B*S*S" for op in ops)
    # reuse MLA base structure verbatim
    assert names == _names(build_mla_attn_ops(DS))


def test_ratio1_same_as_ratio0():
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    assert _names(build_dsv4_hybrid_attn_ops(DS, 1)) == _names(build_dsv4_hybrid_attn_ops(DS, 0))


# ---------------------------------------------------------------------------
# op counts per branch
# ---------------------------------------------------------------------------

def test_op_counts_per_branch():
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    assert len(build_dsv4_hybrid_attn_ops(DS, 0)) == 10     # MLA base
    assert len(build_dsv4_hybrid_attn_ops(DS, 128)) == 12   # base7 + compressor + sparse + o_group + o_proj + add1
    assert len(build_dsv4_hybrid_attn_ops(DS, 4)) == 13     # + indexer


# ---------------------------------------------------------------------------
# Integration — ShapeEval.resolve must not raise
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_resolve_no_error(ratio):
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    ops = build_dsv4_hybrid_attn_ops(DS, ratio)
    spec = ModelSpec("dsv4", DS, layer_pattern=["dsv4"], layer_specs={"dsv4": LayerSpec(ops)})
    pm = ParallelModel(ParallelConfig(), n_layers=1, world_size=1)
    g = ShapeEval().resolve(spec, pm)
    assert 0 in g.stages
    assert len(g.stages[0]) == 1


# ---------------------------------------------------------------------------
# Registry — placeholder raises without ratio, builds with ratio
# ---------------------------------------------------------------------------

def test_registry_dsv4_requires_ratio():
    from cost_eval.layers.registry import ATTN_REGISTRY
    assert "dsv4_hybrid" in ATTN_REGISTRY
    # called without ratio -> clear NotImplementedError (assembler wiring is Task 2.4)
    with pytest.raises(NotImplementedError):
        ATTN_REGISTRY["dsv4_hybrid"](DS)
    # called with ratio -> real op list
    ops = ATTN_REGISTRY["dsv4_hybrid"](DS, 4)
    assert isinstance(ops, list) and "indexer" in _names(ops)
