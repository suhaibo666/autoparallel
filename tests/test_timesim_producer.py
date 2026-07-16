# tests/test_timesim_producer.py
"""shard_rules + producer（spec §3.3 b/c）。"""
import pytest

from cost_eval.timesim.shard_rules import Degrees, axis_values, localize
from cost_eval.model_spec import DimTable

DIMS = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                S=4096, B=1, vocab=129280, n_layers=4)


def test_axis_values_resolves_symbols():
    assert axis_values("S·B·H", DIMS) == [4096, 1, 1792]


def test_axis_values_fail_loud_on_unknown():
    with pytest.raises(ValueError):
        axis_values("S·B·unknown_dim", DIMS)


def test_localize_seq_axis_cp_and_sp():
    deg = Degrees(tp=2, cp=2, sequence_parallel=True)
    # sp 驻留区：S 轴 ÷(cp·tp)；feature 不动
    assert localize([4096, 1, 1792], "S·B·H", deg, sp_active=True) == [1024, 1, 1792]
    # 非 sp 驻留区：只 ÷cp
    assert localize([4096, 1, 1792], "S·B·H", deg, sp_active=False) == [2048, 1, 1792]


def test_localize_feature_and_expert_axes():
    deg = Degrees(tp=2, ep=4)
    # Column 出激活末轴 ÷tp。gated fc1 出维是带系数单轴 "(2·ffn_hidden)"——sym_shape 语法：
    # 顶层 `·` 分轴，系数轴须括号（sym_shape.py 模块头）。
    assert localize([4096, 1, 6144], "S·B·(2·ffn_hidden)", deg, feat_div_last=deg.tp) \
        == [4096, 1, 3072]
    # expert 轴（E）÷ep。cap 走 consumer._cap_value 的标准容量公式，需 moe 字段齐全的 DimTable
    # （n_experts/topk/capacity_factor），现场求值钉死实际数字（不用臆造占位值）。
    moe_dims = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                         S=4096, B=1, vocab=129280, n_layers=4,
                         n_experts=8, topk=4, moe_F=1024)
    vals = axis_values("E·cap·moe_ffn", moe_dims)
    assert vals == [8, 2048, 1024]  # cap = ceil(1.0 · 4096·1 · 4 / 8) = 2048
    assert localize(vals, "E·cap·moe_ffn", deg) == [2, 2048, 1024]


def test_localize_fail_loud_on_indivisible():
    with pytest.raises(ValueError):
        localize([4096, 1, 1793], "S·B·H", Degrees(tp=3), feat_div_last=3)


def test_weight_local_divides_correct_axis():
    from cost_eval.timesim.shard_rules import weight_local
    # Column 权重 [H, 2F]：out 维=末轴 ÷tp；Row 权重 [F, H]：in 维=轴0 ÷tp（Megatron 语义）
    assert weight_local("H·(2·ffn_hidden)", DIMS, "ColumnParallelLinear", 2) == (1792, 3072)
    assert weight_local("ffn_hidden·H", DIMS, "RowParallelLinear", 2) == (1536, 1792)
