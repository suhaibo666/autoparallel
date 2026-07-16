# tests/test_opdag_comm_probe.py
"""comm_probe：从真 mindformers 源静态提取 TP 集合通信调用点（spec §3.3c 惯用法A/B，R1）。"""
import os
import pytest

from cost_eval.opdag.comm_probe import probe_cell_comm

MF_ROOT = os.environ.get("MINDFORMERS_ROOT",
                         r"E:\97-codes\torch_parallel\mindformers\mindformers")
TP_REL = "parallel_core/training_graph/tensor_parallel/layers.py"


def _require_mf():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")


def test_row_parallel_linear_sites():
    _require_mf()
    sites = probe_cell_comm(MF_ROOT, TP_REL, "RowParallelLinear")
    kinds = {(s.ctype, s.guard) for s in sites}
    assert ("reduce_scatter", "sequence_parallel") in kinds       # :619/:646
    assert ("all_reduce", "!sequence_parallel") in kinds          # :621/:648
    assert all(s.src.startswith("layers.py:") for s in sites)


def test_vocab_parallel_embedding_sites():
    _require_mf()
    sites = probe_cell_comm(MF_ROOT, TP_REL, "VocabParallelEmbedding")
    ctypes = {s.ctype for s in sites}
    assert "reduce_scatter" in ctypes                             # embedding_func sp 支（内联惯用法B）
    assert "all_reduce" in ctypes                                 # 非 sp 支（docstring 声明）
    rs = [s for s in sites if s.ctype == "reduce_scatter"]
    assert any("sequence_parallel" in s.guard for s in rs)


def test_column_parallel_linear_has_no_explicit_comm():
    _require_mf()
    assert probe_cell_comm(MF_ROOT, TP_REL, "ColumnParallelLinear") == []
