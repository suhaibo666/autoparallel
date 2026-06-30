"""Tests for M3 parallel_model: degree/fsdp/efsdp/stage helpers."""
import pytest
from cost_eval.specs import ParallelConfig
from cost_eval.parallel_model import ParallelModel


def test_degrees_and_groups():
    pc = ParallelConfig(tp=8, dp_shard=8, cp=1, ep=4, pp=1, sequence_parallel=True)
    pm = ParallelModel(pc, n_layers=4, world_size=8 * 8)
    assert pm.degree("tp") == 8 and pm.degree("sp") == 8     # sp follows tp when SP on
    assert pm.fsdp_degree() == 8                              # dp_shard*cp
    assert pm.efsdp_degree() == 8 * 1 * 8 // 4               # dp_shard*cp*tp//ep = 16


def test_sp_off_degree_is_one():
    pm = ParallelModel(ParallelConfig(tp=8, sequence_parallel=False), 4, 8)
    assert pm.degree("sp") == 1


def test_even_stage_assignment():
    pm = ParallelModel(ParallelConfig(pp=2), n_layers=4, world_size=2)
    assert pm.stage_of(0) == 0 and pm.stage_of(3) == 1
    assert pm.stage_layers(1) == [2, 3]


def test_indivisible_efsdp_raises():
    with pytest.raises(ValueError):
        ParallelModel(ParallelConfig(tp=2, dp_shard=1, cp=1, ep=4), 4, 8)  # ep>dp_shard*cp*tp
