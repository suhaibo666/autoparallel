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


def test_localize_fail_loud_on_indivisible_cp():
    """Task 8 review 委托:indivisible 序轴的 fail-loud 也要覆盖 cp（非仅 tp/feat_div_last）。"""
    with pytest.raises(ValueError):
        localize([4096, 1, 1792], "S·B·H", Degrees(cp=3))


# ── producer（spec §3.3 b/c：并行代入装配 + TP 通信注入 + SP 状态机）─────────────
import os

MF_ROOT = os.environ.get("MINDFORMERS_ROOT",
                         r"E:\97-codes\torch_parallel\mindformers\mindformers")


def _mlp_dag():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import ResolvedSpec
    from cost_eval.opdag.shape_infer import infer_shapes
    dag = extract_cell(
        MF_ROOT, "parallel_core/training_graph/transformer/mlp.py", "MLPInterleaved",
        ResolvedSpec(cell="MLPInterleaved",
                     submodules={"linear_fc1": "ColumnParallelLinear",
                                 "linear_fc2": "RowParallelLinear"}),
        {"gated_linear_unit": True, "activation_type": "silu",
         "add_bias_linear": False, "compute_dtype": "bf16"})
    # extract_cell 本身只产符号骨架（ins/out 段是 `?`占位，实证见现场 dump）；shape 落实是
    # opdag 自己的 PART B（shape_infer.infer_shapes），producer 消费的是**已落实符号 shape**的
    # DAG——与 tests/test_opdag_shape_infer.py::mlp_dag 同一约定（"hidden_states" 是该 Cell
    # construct 的形参名，Inputs 文档注明 shape=(S,B,H)）。
    return infer_shapes(dag, {"hidden_states": "S·B·H"})


def test_build_segment_tp2_sp_injects_comm():
    from cost_eval.timesim.producer import build_segment
    deg = Degrees(tp=2, cp=1, sequence_parallel=True)
    seg = build_segment("layer_0.mlp.fwd", _mlp_dag(), DIMS, deg)
    comms = [o for o in seg.ops if o.op_type == "CommOp"]
    # Column 前 sp all-gather（模块语义注入）+ Row 后 reduce_scatter（comm_probe 源惯用法）
    assert [c.comm.ctype for c in comms] == ["all_gather", "reduce_scatter"]
    assert all(c.stream == "comm_tp" for c in comms)
    # fc1 出 feature ÷tp：gated 2F=6144 → 3072；且 all_gather 后 sp 退出 → seq 全长
    fc1 = next(o for o in seg.ops if o.op_type == "MatMul")
    assert fc1.out_shape == (4096, 1, 3072)
    # View 全部 host_only
    assert all(o.stream == "host_only" for o in seg.ops if o.op_type == "View")
    # reduce_scatter 载荷 = 全 seq 输出字节（S·B·H·2B）
    assert comms[-1].comm.volume_bytes == 4096 * 1 * 1792 * 2
    # 中间态 feature 分片状态机:fc1~fc2 之间的激活轴（含 ffn_hidden 的轴,非末轴)也要 ÷tp
    # (swiglu/silu 支路的 x0 reshape 输出:S·B·ffn_hidden → 3072/tp=1536)
    act = next(o for o in seg.ops if o.op_type == "Activation")
    assert act.out_shape == (4096, 1, 1536)


def test_build_segment_tp1_has_no_comm():
    from cost_eval.timesim.producer import build_segment
    seg = build_segment("layer_0.mlp.fwd", _mlp_dag(), DIMS, Degrees())
    assert all(o.op_type != "CommOp" for o in seg.ops)


def test_build_segment_unknown_module_fail_loud():
    """tp>1 时 matmul 的未知非空 module 名 → fail-loud（防 typo 静默不切分）。"""
    from cost_eval.timesim.producer import build_segment
    from cost_eval.opdag.schema import OpDAG, OpNode
    dag = OpDAG(cell="X", nodes=[OpNode(id=1, op="MatMul", src="x.py:1",
                                        module="ColumnParalleLinear",  # typo
                                        ins=["x:S·B·H:bf16", "w:H·H:bf16"],
                                        out="y:S·B·H:bf16")], edges=[])
    with pytest.raises(ValueError):
        build_segment("x.fwd", dag, DIMS, Degrees(tp=2))


def test_build_segment_feature_axis_ambiguity_fail_loud():
    """F1（reviewer 验证的漏洞）：feature token 集不再硬编码 {"ffn_hidden","moe_ffn"}，而是从
    触发 feat_sharded 的 Column 节点 out_dim 自推导；且强制恰一轴命中。合成一个
    attention-family Column（out_dim="n_heads·v_head_dim"，MLP 之外的 family——旧硬编码集合下
    这条链会静默不切分：weight ÷tp 但激活轴始终不命中 _FEATURE_SYMS，矩乘内部不一致却不报错），
    下游 View 把这两个 sym 拆成两条独立轴（歧义，无 per-tensor 分片跟踪）→ 必须 fail-loud。"""
    from cost_eval.timesim.producer import build_segment
    from cost_eval.opdag.schema import OpDAG, OpNode
    dims = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                     S=4096, B=1, vocab=129280, n_layers=4, v_head_dim=128)
    dag = OpDAG(cell="Attn", nodes=[
        OpNode(id=1, op="MatMul", src="attn.py:1", module="ColumnParallelLinear",
               attrs={"in_dim": "H", "out_dim": "n_heads·v_head_dim"},
               ins=["x:S·B·H:bf16"], out="q:S·B·(n_heads·v_head_dim):bf16"),
        OpNode(id=2, op="View", src="attn.py:2",
               ins=["q:S·B·(n_heads·v_head_dim):bf16"],
               out="q4:S·B·n_heads·v_head_dim:bf16"),
    ], edges=[[1, 2]])
    with pytest.raises(ValueError):
        build_segment("attn.fwd", dag, dims, Degrees(tp=2, sequence_parallel=True))
