"""D-8：PP 每 stage 层数**显式配置**（忠实 mindformers `offset`/`num_layer_list`）+ 校验。

评估器首选 `ParallelConfig.layers_per_stage`（每 stage 层数列表，含 embedding+head 两伪层、
和==n_layers），不自行推测；缺省才退化均匀切。本测试锁 __init__ 校验 + 映射正确。
"""
import pytest

from cost_eval.specs import ParallelConfig
from cost_eval.parallel_model import ParallelModel


def test_explicit_layers_per_stage_uneven_mapping():
    # n_layers=8（含 emb+head），pp=3，显式非均匀切 [3,3,2]
    pc = ParallelConfig(pp=3, layers_per_stage=[3, 3, 2])
    pm = ParallelModel(pc, n_layers=8, world_size=3)
    assert pm.stage_layers(0) == [0, 1, 2]
    assert pm.stage_layers(1) == [3, 4, 5]
    assert pm.stage_layers(2) == [6, 7]
    assert [pm.stage_of(l) for l in range(8)] == [0, 0, 0, 1, 1, 1, 2, 2]


def test_explicit_len_must_equal_pp():
    with pytest.raises(ValueError, match="长度"):
        ParallelModel(ParallelConfig(pp=4, layers_per_stage=[3, 3, 2]), n_layers=8, world_size=4)


def test_explicit_sum_must_equal_n_layers():
    with pytest.raises(ValueError, match="之和"):
        ParallelModel(ParallelConfig(pp=3, layers_per_stage=[3, 3, 3]), n_layers=8, world_size=3)


def test_explicit_each_stage_positive():
    with pytest.raises(ValueError, match=">0"):
        ParallelModel(ParallelConfig(pp=3, layers_per_stage=[4, 4, 0]), n_layers=8, world_size=3)


def test_none_falls_back_to_uniform_split():
    # None → 均匀切：8 层 pp=2 → per=4 → [0,0,0,0,1,1,1,1]
    pm = ParallelModel(ParallelConfig(pp=2, layers_per_stage=None), n_layers=8, world_size=2)
    assert [pm.stage_of(l) for l in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_uniform_split_standard():
    # 标准均匀切（2026-07-14 修,用户指正）:只切中间层(mid=n_layers-2),余数**前置**(前 rem 个
    # stage 各多 1);embedding→stage0、head→末 stage。n_layers=7(mid=5) pp=2 → 3+2 →
    # 映射 [emb|3 层 | 2 层|head] = [0,0,0,0,1,1,1]。旧行为(余数堆末 stage)已废。
    pm = ParallelModel(ParallelConfig(pp=2, layers_per_stage=None), n_layers=7, world_size=2)
    assert [pm.stage_of(l) for l in range(7)] == [0, 0, 0, 0, 1, 1, 1]
