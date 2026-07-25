# tests/test_opdag_gpt_segments.py
"""GPTModel 级段提取（spec §3.3a）：loss / embedding / lm_head + construct 段序核对。"""
import os
import pytest

from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.module_resolver import ResolvedSpec

MF_ROOT = os.environ.get("MINDFORMERS_ROOT",
                         r"E:\97-codes\torch_parallel\mindformers\mindformers")
LOSS_REL = "parallel_core/training_graph/loss_func.py"


def _require_mf():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")


LOSS_FLAGS = {
    "compute_dtype": "bf16", "add_bias_linear": False,
    # 以下均为 CrossEntropyLoss.construct 里旁路 if 分支的裁决（config 会剪掉的旁支，非主链）：
    "enable_force_redistribute": False,  # 非 semi/auto-parallel 强制重分布（主链无关，:309）
    "need_monitor": False,               # local/device loss 监控关闭（默认，:318）
    "calculate_per_token_loss": False,   # TransformerConfig 默认（:281→:338）
    "seq_pipe": False,                   # seq_split_num 默认 1（:262→:338）
}


def test_loss_dag_nested_call_args_materialized():
    """T0-6.5 Fix1(Hole1 回归):`numerator = self.sum2(self.mul2(loss_reduce, input_mask))`
    （loss_func.py:334）与 `denominator = self.add2(self.sum2(input_mask), self.cast(...))`
    （:335-337）里的嵌套 Call 实参此前被 _emit 静默丢弃（`if not isinstance(a, ast.Name): continue`）——
    不仅丢边，mul2/内层 sum2/内层 cast 三个真实算子直接不产节点，DAG 因此裂成 3 个不连通岛
    （{1..11} softmax/nllloss、{12,13} mask 处理、{14,15,16} numerator/denominator/div2 自成一簇，
    彼此只靠 14/15→16 单向喂，11/13 的真实产出从未接进 14/15）。
    物化后新增 3 个真实节点（mul2 :334、inner sum2 :336、inner cast :337——F.tuple_to_array
    实参 opaque，见 Fix2），全图应合一为单连通分量。"""
    _require_mf()
    from cost_eval.opdag.gpt_segments import extract_loss
    dag = extract_loss(MF_ROOT, LOSS_FLAGS)

    # (a) census 长度:16(修复前基线)+ 3 个新增真实节点 = 19。
    assert len(dag.nodes) == 19

    # (b) mul2(:334)、inner sum2(:336) 已产出为独立节点,且不再是空 ins。
    by_src = {}
    for n in dag.nodes:
        by_src.setdefault(n.src, []).append(n)
    assert any(n.op == "Elementwise" and n.ins for n in by_src.get("loss_func.py:334", [])), (
        "mul2(:334) 应已物化为独立节点且有真实 ins")
    assert any(n.op == "Elementwise" and n.ins for n in by_src.get("loss_func.py:336", [])), (
        "inner sum2(input_mask)(:336) 应已物化为独立节点且有真实 ins")

    # (c) 除 loss_func.py:337 的字面量 Cast(F.tuple_to_array((1e-8,)) 是 opaque 自由函数调用,
    #     其实参本就非可追踪张量,ins=[] 是真实语义而非丢边——见 Fix2 opaque_calls)外,
    #     不应再有节点 ins==[](修复前 numerator/denominator 因嵌套实参被丢而误成 ins==[]）。
    empty_ins = [n for n in dag.nodes if not n.ins]
    assert [n.src for n in empty_ins] == ["loss_func.py:337"]

    # (d) 单连通分量:以 dag.edges 建无向邻接,所有节点(含 id=1)必须彼此可达。
    node_ids = {n.id for n in dag.nodes}
    assert {i for e in dag.edges for i in e} <= node_ids, "边端点不在节点集"
    adj: dict[int, set] = {i: set() for i in node_ids}
    for s, d in dag.edges:
        adj[s].add(d)
        adj[d].add(s)
    start = next(iter(node_ids))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    assert seen == node_ids, f"DAG 不是单连通分量:未触达 {node_ids - seen}"


def test_extract_cross_entropy_loss_inlines_subcells():
    _require_mf()
    dag = extract_cell(
        MF_ROOT, LOSS_REL, "CrossEntropyLoss",
        ResolvedSpec(cell="CrossEntropyLoss", submodules={}),
        LOSS_FLAGS, recurse=True,
        subcell_specs={
            "_LogSoftmax": ResolvedSpec(cell="_LogSoftmax", submodules={}),
            "_NLLLoss": ResolvedSpec(cell="_NLLLoss", submodules={}),
        },
    )
    assert len(dag.nodes) >= 5                              # 两子 Cell 已内联（非 2 个 SubCell 占位）
    assert all(n.op != "SubCell" for n in dag.nodes)
    assert all(n.src.split(":")[0] == "loss_func.py" for n in dag.nodes)


def test_extract_embedding_walks_morph_func():
    _require_mf()
    from cost_eval.opdag.gpt_segments import extract_embedding
    dag = extract_embedding(MF_ROOT, {"compute_dtype": "bf16"})
    # 精确 census 钉死(仓库惯例;是 ReLU/Minimum/Equal _CLS2OP 与 FREE_CALL_MAP ops.mul 映射的唯一
    # tripwire):reshape 铺平 → relu→minimum→equal(TP mask 三连,layers.py:153-155)→ mint embedding
    # 查表(Gather,:160)→ ops.mul mask(:165)→ reshape 回 (bs,-1,hidden)(:167)。行号为 2026-07-16 基线。
    # 2026-07-25(P0#5):census 7 → 9 —— 两条**真算子**此前被 `_handle_assign` 静默丢:
    #   layers.py:152 `input_ = input_ - self.vocab_start_index`(BinOp,张量−标量 → 线性)
    #   layers.py:164 `input_mask = input_mask.expand_dims(-1)`(张量方法形态的视图)
    # 二者 PIN 都不存激活 → `derive_saves` **逐字节不变**(实测同为 4 项 saves,同 dtype/shape)。
    assert [n.op for n in dag.nodes] == [
        "View", "Elementwise", "Activation", "Elementwise", "Elementwise",
        "Gather", "View", "Elementwise", "View"]
    assert [int(n.src.split(":")[1]) for n in dag.nodes] == [
        149, 152, 153, 154, 155, 160, 164, 165, 167]
    assert all(n.src.split(":")[0] == "layers.py" for n in dag.nodes)


def test_extract_embedding_records_opaque_allreduce():
    """T0-6.5 Fix2(Hole2 回归):`ops.AllReduce(group=self.group)(output_parallel)`（layers.py:182,
    guard !sequence_parallel && enable_embedding_tp,由 extract_embedding 默认注入命中该支）是
    walker 四种调用形态外的双层调用(`OpCls(kwargs)(x)`,func 本身是 Call 而非 Attribute/Name)——
    此前在 _handle_call 终端 fallthrough 静默丢,现应显式记入 dag.opaque_calls,且不产 DAG 节点
    (通信节点由 comm_probe 另侧覆盖,walker 若也发射会双重计数,见 gpt_segments.extract_embedding
    docstring)——精确 7-op census(test_extract_embedding_walks_morph_func)不变。"""
    _require_mf()
    from cost_eval.opdag.gpt_segments import extract_embedding
    dag = extract_embedding(MF_ROOT, {"compute_dtype": "bf16"})
    assert any("AllReduce" in c["expr"] for c in dag.opaque_calls), dag.opaque_calls
    hit = next(c for c in dag.opaque_calls if "AllReduce" in c["expr"])
    assert hit["src"] == "layers.py:182"
    # census 未变:walker 未为这条 opaque 调用产节点(9-op census 见上一测试的 2026-07-25 说明)。
    assert [n.op for n in dag.nodes] == [
        "View", "Elementwise", "Activation", "Elementwise", "Elementwise",
        "Gather", "View", "Elementwise", "View"]


def test_lm_head_segment_source_pinned():
    _require_mf()
    from cost_eval.opdag.gpt_segments import head_segment_dag
    dag = head_segment_dag()
    assert [n.op for n in dag.nodes] == ["MatMul", "View", "View", "Cast"]
    assert dag.nodes[0].module == "ColumnParallelLinear"
    assert all(n.src.startswith("gpt_model.py:") for n in dag.nodes)
    assert dag.nodes[-1].attrs.get("to_dtype") == "fp32"  # logits cast fp32（gpt_model.py:507，2026-07-16 基线；计划稿注 :509 已按实际源行订正）


def test_verify_gpt_order():
    _require_mf()
    from cost_eval.opdag.gpt_segments import verify_gpt_order
    order = verify_gpt_order(MF_ROOT)
    lm = order.index("language_model")
    head = order.index("output_layer")
    loss = order.index("compute_language_model_loss")
    assert lm < head < loss


def test_head_segment_composes_with_producer():
    """I1(终审 2026-07-17 producer 契约回归,不需要 mindformers 源——head_segment_dag() 是纯合成 DAG）:
    producer.build_segment 对 Col/Row MatMul 强制要求 in_dim/out_dim attrs 以重建权重符号 shape
    （producer.py:184-188:`if not in_dim or not out_dim: raise ValueError(...)`）——head_segment_dag()
    的 MatMul(module=ColumnParallelLinear) 此前未标注这两个属性,即便 tp=1 也会在此 fail-loud
    （cross-module 不一致,build_segment(head_segment_dag()) 无法组装）。"""
    from cost_eval.opdag.gpt_segments import head_segment_dag
    from cost_eval.timesim.producer import build_segment
    from cost_eval.timesim.shard_rules import Degrees
    from cost_eval.model_spec import DimTable

    dims = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                     S=4096, B=1, vocab=129280, n_layers=4)
    seg = build_segment("head.fwd", head_segment_dag(), dims, Degrees())
    mm = next(op for op in seg.ops if op.op_type == "MatMul")
    assert mm.in_shapes[1] == (1792, 129280)


def test_head_segment_producer_tp_sp_injects_module_semantics_ag():
    """回归（T1-2 质量复审附加发现）：head_segment_dag() 的 Column MatMul 带**显式权重 ref**
    （ins=["h:...", "W_head:H·vocab:..."]，真实 lm_head 生产代码非合成 fixture），tp>1+sp 下
    须走 Column 模块语义 all_gather（op_id 尾 ".ag"、module="injected:module-semantics"），
    而**不是**被 S 分歧重分布误触发成 layout-redistribution（".ag0"）。producer 的 is_linear
    守卫（reconciliation 对线性族 MatMul 跳过）锁死此正确行为——lm_head 的 TP+SP 是主流配置，
    不是边角。tp=1 的 test_head_segment_composes_with_producer 走不到 reconciliation,故此处专测。"""
    from cost_eval.opdag.gpt_segments import head_segment_dag
    from cost_eval.timesim.producer import build_segment
    from cost_eval.timesim.shard_rules import Degrees
    from cost_eval.model_spec import DimTable

    dims = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                     S=4096, B=1, vocab=129280, n_layers=4)
    seg = build_segment("head.fwd", head_segment_dag(), dims,
                        Degrees(tp=2, sequence_parallel=True))
    comms = [o for o in seg.ops if o.op_type == "CommOp"]
    assert len(comms) == 1                                    # 单个 AG，非重复注入
    ag = comms[0]
    assert ag.module == "injected:module-semantics"          # Column 语义，非 layout-redistribution
    assert ag.op_id.endswith(".ag") and not ag.op_id.endswith(".ag0")
    assert ag.comm.ctype == "all_gather"
    # 权重全量按 Column 语义 out(末轴 vocab)÷tp、Column 输出 carrier=vocab ÷tp
    mm = next(o for o in seg.ops if o.op_type == "MatMul")
    assert mm.in_shapes[1] == (1792, 129280 // 2)
    assert mm.out_shape == (4096, 1, 129280 // 2)
