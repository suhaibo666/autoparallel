"""Tests for M4 shape_eval: eval_expr / resolve_tensor / detect_reshard / ShapeEval."""
from cost_eval.model_spec import DimTable, TensorRef, OpSpec, OpType, LayerSpec, ModelSpec

DIMS = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)


# ---------------------------------------------------------------------------
# Task 4 — eval_expr
# ---------------------------------------------------------------------------

def test_eval_plain_symbol():
    from cost_eval.shape_eval import eval_expr
    assert eval_expr("H", DIMS) == 8


def test_eval_arithmetic():
    from cost_eval.shape_eval import eval_expr
    assert eval_expr("(n_heads+2*n_kv)*head_dim", DIMS) == (2 + 2 * 2) * 4   # 24


def test_eval_integer_literal():
    from cost_eval.shape_eval import eval_expr
    assert eval_expr("2*F", DIMS) == 32


def test_eval_rejects_unknown_name():
    import pytest
    from cost_eval.shape_eval import eval_expr
    with pytest.raises(ValueError):
        eval_expr("import os", DIMS)


# ---------------------------------------------------------------------------
# Task 5 — resolve_tensor
# ---------------------------------------------------------------------------

from cost_eval.specs import ParallelConfig
from cost_eval.parallel_model import ParallelModel


def _pm(**kw):
    return ParallelModel(ParallelConfig(**kw), n_layers=2, world_size=64)


def test_resolve_shards_tp_dim():
    from cost_eval.shape_eval import resolve_tensor
    pm = _pm(tp=8, dp_shard=8)
    t = TensorRef("y", ("S", "B", "2*F"), shard={2: "tp"})
    rt = resolve_tensor(t, DIMS, pm)               # S*B*(2*16/8) = 4*1*4 = 16
    assert rt.local_numel == 4 * 1 * (2 * 16 // 8)


def test_resolve_marks_expert():
    from cost_eval.shape_eval import resolve_tensor
    import cost_eval.model_spec as ms
    pm = _pm(ep=4, tp=2, dp_shard=2)
    d = ms.DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10,
                    n_layers=2, n_experts=8)
    w = TensorRef("w", ("n_experts", "H"), shard={0: "ep"}, is_weight=True)
    rt = resolve_tensor(w, d, pm)
    assert rt.is_expert and rt.local_numel == (8 // 4) * 8


def test_resolve_indivisible_raises():
    import pytest
    from cost_eval.shape_eval import resolve_tensor
    pm = _pm(tp=3)
    with pytest.raises(ValueError):
        resolve_tensor(TensorRef("y", ("H",), shard={0: "tp"}), DIMS, pm)  # 8%3 != 0


# ---------------------------------------------------------------------------
# Task 6 — Placement / CommSpec / detect_reshard
# ---------------------------------------------------------------------------

def test_partial_to_replicate_allreduce():
    from cost_eval.shape_eval import detect_reshard, Placement
    src = Placement(shard={}, partial="tp")
    dst = Placement(shard={}, partial=None)
    c = detect_reshard(src, dst, numel=128, dtype_bytes=2)
    assert c.ctype == "all_reduce" and c.group_axis == "tp" and c.volume_bytes == 256


def test_shard_to_shard_alltoall():
    from cost_eval.shape_eval import detect_reshard, Placement
    src = Placement(shard={0: "ep"}, partial=None)
    dst = Placement(shard={1: "ep"}, partial=None)
    assert detect_reshard(src, dst, 64, 2).ctype == "all_to_all"


def test_no_reshard_when_equal():
    from cost_eval.shape_eval import detect_reshard, Placement
    p = Placement(shard={2: "tp"}, partial=None)
    assert detect_reshard(p, p, 64, 2) is None


# ---------------------------------------------------------------------------
# Task 7 — ResolvedGraph + ShapeEval.resolve
# ---------------------------------------------------------------------------

def _toy_dense_layer():
    x = TensorRef("x", ("S", "B", "H"))
    w1 = TensorRef("w1", ("H", "2*F"), shard={1: "tp"}, is_weight=True)
    h = TensorRef("h", ("S", "B", "2*F"), shard={2: "tp"})
    op = OpSpec("fc1", OpType.MATMUL, inputs=[x, w1], output=h, params=[w1], saves=[x])
    return LayerSpec(ops=[op])


def test_resolve_graph_groups_by_stage():
    from cost_eval.shape_eval import ShapeEval
    spec = ModelSpec("toy", DIMS, ["dense", "dense"], {"dense": _toy_dense_layer()})
    pm = _pm(tp=8, dp_shard=8, pp=2)
    g = ShapeEval().resolve(spec, pm)
    assert set(g.stages.keys()) == {0, 1}
    op0 = g.stages[0][0].ops[0]
    assert op0.params[0].local_numel == 8 * (2 * 16 // 8)   # H * (2F/tp)
    assert op0.saves[0].name == "x"
