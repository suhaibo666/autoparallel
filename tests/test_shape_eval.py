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
