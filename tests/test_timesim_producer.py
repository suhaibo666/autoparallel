# tests/test_timesim_producer.py
"""shard_rules + producer（spec §3.3 b/c）。"""
from types import SimpleNamespace

import pytest

from cost_eval.timesim.producer import build_segment
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
# `_mlp_dag` 夹具已提升为 tests/conftest.py::mlp_dag（真源 MLP 段构建三处复制收敛——Task 11 review）。


def test_build_segment_tp2_sp_injects_comm(mlp_dag):
    from cost_eval.timesim.producer import build_segment
    deg = Degrees(tp=2, cp=1, sequence_parallel=True)
    seg = build_segment("layer_0.mlp.fwd", mlp_dag, DIMS, deg)
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


def test_build_segment_tp1_has_no_comm(mlp_dag):
    from cost_eval.timesim.producer import build_segment
    seg = build_segment("layer_0.mlp.fwd", mlp_dag, DIMS, Degrees())
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


def test_injected_ag_keeps_data_producer_dep():
    """F3 Bug A 回归：注入 AG 挂 comm_tp 流——它的 deps 必须含矩乘的**全部**入边生产者
    （device 流的 Norm 从 AG 视角是跨流，不能被"与矩乘同 device 流"的过滤误丢、挂空）。"""
    from cost_eval.timesim.producer import build_segment
    from cost_eval.opdag.schema import OpDAG, OpNode
    dag = OpDAG(cell="X", nodes=[
        OpNode(id=1, op="Norm", src="norm.py:1",
               ins=["x0:S·B·H:bf16"], out="x:S·B·H:bf16"),
        OpNode(id=2, op="MatMul", src="col.py:1", module="ColumnParallelLinear",
               attrs={"in_dim": "H", "out_dim": "H"},
               ins=["x:S·B·H:bf16"], out="y:S·B·H:bf16"),
    ], edges=[[1, 2]])
    seg = build_segment("x.fwd", dag, DIMS, Degrees(tp=2, sequence_parallel=True))
    ag = next(o for o in seg.ops if o.op_id.endswith(".ag"))
    assert ag.deps == ("X#1",)
    mm = next(o for o in seg.ops if o.op_type == "MatMul")
    assert mm.deps == ("X#2.ag",)


def test_row_downstream_dep_redirects_to_rs():
    """F3 Bug B 回归：Row 注入 .rs 后，下游消费者的依赖须重定向解析到 `<matmul>.rs`
    （comm_tp 流），而非矩乘本身（device 流会被同流过滤掉、下游永远看不到 .rs）。"""
    from cost_eval.timesim.producer import build_segment
    from cost_eval.opdag.schema import OpDAG, OpNode
    dag = OpDAG(cell="X", nodes=[
        OpNode(id=1, op="MatMul", src="row.py:1", module="RowParallelLinear",
               attrs={"in_dim": "H", "out_dim": "H"},
               ins=["x:S·B·H:bf16"], out="y:S·B·H:bf16"),
        OpNode(id=2, op="Norm", src="norm.py:1",
               ins=["y:S·B·H:bf16"], out="z:S·B·H:bf16"),
    ], edges=[[1, 2]])
    seg = build_segment("x.fwd", dag, DIMS, Degrees(tp=2, sequence_parallel=True))
    norm = next(o for o in seg.ops if o.op_type == "Norm")
    assert norm.deps == ("X#1.rs",)


def test_column_inside_feature_shard_zone_fail_loud():
    """Column∘Column 直连守卫（spec re-review）：前一 Column 的 feature 分片区未被 Row 关闭时
    再遇 Column → fail-loud（否则激活收缩维已 ÷tp 而 weight-in 仍全量，矩乘静默不一致——
    复审探针实证 contraction 3072 vs weight-in 6144 无报错）。"""
    from cost_eval.timesim.producer import build_segment
    from cost_eval.opdag.schema import OpDAG, OpNode
    dag = OpDAG(cell="X", nodes=[
        OpNode(id=1, op="MatMul", src="col1.py:1", module="ColumnParallelLinear",
               attrs={"in_dim": "H", "out_dim": "ffn_hidden"},
               ins=["x:S·B·H:bf16"], out="h:S·B·ffn_hidden:bf16"),
        OpNode(id=2, op="MatMul", src="col2.py:1", module="ColumnParallelLinear",
               attrs={"in_dim": "ffn_hidden", "out_dim": "H"},
               ins=["h:S·B·ffn_hidden:bf16"], out="y:S·B·H:bf16"),
    ], edges=[[1, 2]])
    with pytest.raises(ValueError):
        build_segment("x.fwd", dag, DIMS, Degrees(tp=2))


def test_build_segment_feature_axis_resolved_by_carrier():
    """F1（reviewer 验证的漏洞）：feature token 集不再硬编码 {"ffn_hidden","moe_ffn"}，而是从
    触发 feat_sharded 的 Column 节点 out_dim 自推导；且强制恰一轴命中。合成一个
    attention-family Column（out_dim="n_heads·v_head_dim"，MLP 之外的 family——旧硬编码集合下
    这条链会静默不切分：weight ÷tp 但激活轴始终不命中 _FEATURE_SYMS，矩乘内部不一致却不报错），
    下游 View 把这两个 sym 拆成两条独立轴。

    **T1 语义变更（per-tensor carrier 规则）**：T0 把 out_dim 全 token 集 {"n_heads","v_head_dim"}
    当 feature 轴集合——两 sym 拆两轴即"≥2 轴命中"歧义 fail-loud；T1 的 carrier=out_dim 顶层乘积
    首个符号因子（"n_heads"），拆轴后只命中 n_heads 一轴——同一 DAG 现在**成功构段**且分片正确
    落在 carrier 轴上（÷tp）。真歧义守卫由 test_carrier_duplicate_axes_fail_loud 接棒。"""
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
    seg = build_segment("attn.fwd", dag, dims, Degrees(tp=2, sequence_parallel=True))
    mm = next(o for o in seg.ops if o.op_type == "MatMul")
    assert mm.out_shape == (4096, 1, 8 * 128 // 2)         # flat 轴 ÷tp
    view = next(o for o in seg.ops if o.op_type == "View")
    assert view.out_shape == (4096, 1, 4, 128)             # carrier=n_heads 恰一轴 ÷tp


def test_build_segment_opaque_comm_call_fail_loud_without_flag():
    """Task 9 quality review 搭车项 1：dag.opaque_calls 里若混进 AllReduce/ReduceScatter/AllGather/
    AlltoAll 字样的调用点（embedding 段等 opaque 段的通信本该由 comm_probe+装配层注入，不是
    walker fallthrough 静默漏记）——不传 opaque_comm_ok 时须 fail-loud；调用方确认后传
    opaque_comm_ok=True 放行（schema.py opaque_calls docstring 消费契约）。"""
    from cost_eval.timesim.producer import build_segment
    from cost_eval.opdag.schema import OpDAG
    dag = OpDAG(cell="X", nodes=[], edges=[],
                opaque_calls=[{"src": "layers.py:182",
                               "expr": "ops.AllReduce(group=self.group)(x)"}])
    with pytest.raises(ValueError):
        build_segment("x.fwd", dag, DIMS, Degrees())
    seg = build_segment("x.fwd", dag, DIMS, Degrees(), opaque_comm_ok=True)
    assert seg.ops == ()


# ── per-tensor 分片状态（T1 交接要点5 端态设计）——纯合成 DAG，不依赖 mindformers ──────


def _synth_dag(nodes, edges, opaque=()):
    return SimpleNamespace(cell="synth", nodes=nodes, edges=edges,
                           opaque_calls=list(opaque))


def _node(id, op, src, ins, out, module="", attrs=None):
    return SimpleNamespace(id=id, op=op, src=src, ins=ins, out=out,
                           module=module, attrs=attrs or {})


# 合成测试用 dims：qk_head_dim token 映射 DimTable.qk_nope_head_dim（consumer._SYM2FIELD），
# 既有模块级 DIMS 未设该字段（=0 → 未解析 fail-loud），此处单独给值。
DIMS_ATTN = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                     S=4096, B=1, vocab=129280, n_layers=4,
                     qk_nope_head_dim=224, v_head_dim=224)


def test_per_tensor_state_branch_isolation():
    """per-tensor 核心性质：不过 Column 的旁支不携 feature 分片（T0 全局位在此必错）。
    合成结构 = MLA pe_concat 惯用法缩影：主支过 Column（heads 分片），旁支直连 View。"""
    nodes = [
        _node(1, "MatMul", "f.py:1", ["x:S·B·H:bf16"],
              "u:S·B·(n_heads·qk_head_dim):bf16", module="ColumnParallelLinear",
              attrs={"in_dim": "H", "out_dim": "n_heads·qk_head_dim"}),
        _node(2, "View", "f.py:2", ["u:S·B·(n_heads·qk_head_dim):bf16"],
              "u4:S·B·n_heads·qk_head_dim:bf16", attrs={"view": "reshape"}),
        _node(3, "View", "f.py:3", ["x:S·B·H:bf16"],
              "side:S·B·H:bf16", attrs={"view": "reshape"}),
    ]
    seg = build_segment("s.fwd", _synth_dag(nodes, [[1, 2]]), DIMS_ATTN, Degrees(tp=2))
    by_src = {o.src: o for o in seg.ops if o.op_type != "CommOp"}
    # 主支：flat 轴÷tp → reshape 后 n_heads 轴÷tp（carrier=n_heads，恰一轴）
    assert by_src["f.py:1"].out_shape == (4096, 1, 8 * 224 // 2)
    assert by_src["f.py:2"].out_shape == (4096, 1, 4, 224)
    # 旁支：从未过 Column → 不分片（T0 全局 feat_sharded 位在此会误除或 fail-loud）
    assert by_src["f.py:3"].out_shape == (4096, 1, 1792)


def test_per_tensor_sp_reconciliation_ag():
    """S 分歧汇合（源事实4：MLA pe_concat 惯用法）：SP 驻留支与已聚合支在多输入 op 汇合 →
    驻留支注入 layout-redistribution AG（volume=分片字节，AG 口径）。"""
    nodes = [
        _node(1, "MatMul", "f.py:1", ["x:S·B·H:bf16"],
              "u:S·B·(n_heads·qk_head_dim):bf16", module="ColumnParallelLinear",
              attrs={"in_dim": "H", "out_dim": "n_heads·qk_head_dim"}),
        _node(2, "View", "f.py:2", ["x:S·B·H:bf16"],
              "side:S·B·H:bf16", attrs={"view": "reshape"}),
        _node(3, "Elementwise", "f.py:3",
              ["u:S·B·(n_heads·qk_head_dim):bf16", "side:S·B·H:bf16"],
              "z:S·B·H:bf16"),
    ]
    deg = Degrees(tp=2, sequence_parallel=True)
    seg = build_segment("s.fwd", _synth_dag(nodes, [[1, 3], [2, 3]]), DIMS_ATTN, deg)
    comms = [o for o in seg.ops if o.op_type == "CommOp"]
    # Column 前模块语义 AG（.ag）+ side 支在节点3 汇合前的重分布 AG（.ag1）
    assert [c.module for c in comms] == ["injected:module-semantics",
                                         "injected:layout-redistribution"]
    redis = comms[1]
    assert redis.comm.ctype == "all_gather"
    assert redis.in_shapes == ((2048, 1, 1792),)          # S/(tp) 驻留分片
    assert redis.out_shape == (4096, 1, 1792)
    assert redis.comm.volume_bytes == 2048 * 1792 * 2      # AG=分片入参字节
    # 汇合节点吃聚合后的 side → 两输入 S 一致（注入 AG 与汇合节点同 src，须滤 CommOp）
    z = next(o for o in seg.ops if o.src == "f.py:3" and o.op_type != "CommOp")
    assert z.in_shapes[1] == (4096, 1, 1792)
    assert redis.op_id in z.deps


def test_carrier_duplicate_axes_fail_loud():
    """同一 carrier 命中 ≥2 轴 = 真歧义 → fail-loud（per-tensor 后守卫收窄到此形态）。"""
    nodes = [
        _node(1, "MatMul", "f.py:1", ["x:S·B·H:bf16"],
              "u:S·B·(n_heads·qk_head_dim):bf16", module="ColumnParallelLinear",
              attrs={"in_dim": "H", "out_dim": "n_heads·qk_head_dim"}),
        _node(2, "View", "f.py:2", ["u:S·B·(n_heads·qk_head_dim):bf16"],
              "bad:S·B·n_heads·(n_heads·qk_head_dim):bf16", attrs={"view": "reshape"}),
    ]
    with pytest.raises(ValueError, match="命中 2 轴"):
        build_segment("s.fwd", _synth_dag(nodes, [[1, 2]]), DIMS_ATTN, Degrees(tp=2))


def test_row_without_sharded_input_fail_loud():
    """tp>1 的 Row 输入无 feature carrier（上游没有 Column）→ fail-loud（Megatron Row 恒
    消费分片激活；静默不切会双倍计算量）。"""
    nodes = [
        _node(1, "MatMul", "f.py:1", ["x:S·B·H:bf16"],
              "y:S·B·H:bf16", module="RowParallelLinear",
              attrs={"in_dim": "H", "out_dim": "H"}),
    ]
    with pytest.raises(ValueError, match="Row"):
        build_segment("s.fwd", _synth_dag(nodes, []), DIMS_ATTN, Degrees(tp=2))
