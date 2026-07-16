# cost_eval/opdag/gpt_segments.py
"""GPTModel 级非重复段（spec §3.3a）：embedding / lm_head / loss 提取 + construct 段序核对。
transformer 层段沿用既有 per-cell 提取。所有节点 src 回指真源（源忠实链完整）。

事实源（mindformers @ parallel_core/training_graph，行号为 2026-07-16 基线，漂移时按惯用法重定位）：
  embedding: tensor_parallel/layers.py VocabParallelEmbedding——construct 调 self.embedding_morph
             （P.Morph(self.embedding_func, ...)，__init__:108-109 绑定；注意此处用 `P.Morph`
             属性形式，非裸 `Morph(...)`——见 extractor._unwrap_to_call 的属性形式扩展）
             → 走既有 _morph_aliases 内联机制，展开 embedding_func（:140-186）。
             该方法内 `if self.sequence_parallel:`（:168）分支下还嵌 `if bs > 1:`（:170/:175，
             bs 为运行期形状标量，AST 无法判定）——为避免剪枝走查在此 fail-loud，抽取默认取
             sequence_parallel=False（走 :181-186 的 enable_embedding_tp 分支，同为标准 TP 训练
             路径的一支，:110 `enable_embedding_tp = tp>1 and vocab%tp==0`）。
  lm_head:   gpt_model.py:364 `self.output_layer = ColumnParallelLinear(hidden→vocab)`；
             construct :503-507 logits = output_layer(h) → transpose(:505) → morphed_reshape(:506)
             → cast(fp32,:507)。GPTModel.construct 含 mtp/eod/zbv 等大量 config 分支，T0 不整体
             走查——head 段由本模块按上述已核实调用序**合成**（每节点 src 钉到真源行，顺序由
             verify_gpt_order 的 AST 断言守护；整体走查待 v1.5 MTP 时一并做）。
  loss:      loss_func.py CrossEntropyLoss（_LogSoftmax + _NLLLoss 直接实例化组合，Task 5 惯用法B）。
"""
from __future__ import annotations

import ast
import os

from .extractor import extract_cell, _find_class, _method_of
from .module_resolver import ResolvedSpec
from .schema import OpDAG, OpNode

TP_REL = "parallel_core/training_graph/tensor_parallel/layers.py"
GPT_REL = "parallel_core/training_graph/base_models/gpt/gpt_model.py"
LOSS_REL = "parallel_core/training_graph/loss_func.py"


def extract_loss(mf_root: str, config_flags: dict) -> OpDAG:
    return extract_cell(
        mf_root, LOSS_REL, "CrossEntropyLoss",
        ResolvedSpec(cell="CrossEntropyLoss", submodules={}),
        config_flags, recurse=True,
        subcell_specs={
            "_LogSoftmax": ResolvedSpec(cell="_LogSoftmax", submodules={}),
            "_NLLLoss": ResolvedSpec(cell="_NLLLoss", submodules={}),
        },
    )


def extract_embedding(mf_root: str, config_flags: dict) -> OpDAG:
    """抽取 VocabParallelEmbedding 的 op-DAG（真机 AST 走查，非合成）。

    默认注入 enable_embedding_tp=True（标准 TP 训练路径的 mask 分支，layers.py:150-165）、
    sequence_parallel=False（避开 :168-179 sp 分支里 `if bs > 1:` 的运行期不可判定形状分支，
    layers.py:170/175——AST 剪枝走查下该 if 无法由 config 判定，会 fail-loud；sequence_parallel=False
    时直接走 :181-186，同为真实存在的标准路径，非回避覆盖）。调用方可在 config_flags 里覆盖这两键。
    """
    flags = {"enable_embedding_tp": True, "sequence_parallel": False, **config_flags}
    return extract_cell(
        mf_root, TP_REL, "VocabParallelEmbedding",
        ResolvedSpec(cell="VocabParallelEmbedding", submodules={}),
        flags,
    )


def head_segment_dag() -> OpDAG:
    """lm_head 段（合成，逐节点钉真源行；结构由 verify_gpt_order 守护——见模块 docstring）。"""
    nodes = [
        OpNode(id=0, op="MatMul", src="gpt_model.py:503", module="ColumnParallelLinear",
               ins=["h:S·B·H:bf16", "W_head:H·vocab:bf16"], out="logits:S·B·vocab:bf16"),
        OpNode(id=1, op="View", src="gpt_model.py:505",
               ins=["logits:S·B·vocab:bf16"], out="logits_t:B·S·vocab:bf16",
               attrs={"view": "transpose"}),
        OpNode(id=2, op="View", src="gpt_model.py:506",
               ins=["logits_t:B·S·vocab:bf16"], out="logits_2d:S·B·vocab:bf16",
               attrs={"view": "reshape"}),
        OpNode(id=3, op="Cast", src="gpt_model.py:507",
               ins=["logits_2d:S·B·vocab:bf16"], out="logits32:S·B·vocab:fp32",
               attrs={"to_dtype": "fp32"}),
    ]
    return OpDAG(cell="GPTModel.head", nodes=nodes, edges=[[0, 1], [1, 2], [2, 3]])


def verify_gpt_order(mf_root: str) -> list[str]:
    """AST 读 GPTModel.construct，按**源序**（非 ast.walk 的 BFS 序——见下）返回 `self.X(...)`
    调用的 attr 名列表；language_model → output_layer → compute_language_model_loss 缺一或
    错序 → fail-loud。

    ast.walk 是 BFS，不保证按源码行序产出节点，故收集后按 lineno 排序还原真实调用序。
    """
    path = os.path.join(mf_root, *GPT_REL.split("/"))
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    cls = _find_class(tree, "GPTModel")
    construct = _method_of(cls, "construct") if cls else None
    if construct is None:
        raise ValueError("gpt_segments: GPTModel.construct 定位失败（fail-loud）")
    calls: list[tuple[int, str]] = []
    for node in ast.walk(construct):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            calls.append((node.lineno, node.func.attr))
    calls.sort(key=lambda t: t[0])
    order = [name for _, name in calls]
    for need in ("language_model", "output_layer", "compute_language_model_loss"):
        if need not in order:
            raise ValueError(f"gpt_segments: GPTModel.construct 里找不到 self.{need}(...) —— "
                              f"源结构变了，段合成失效（fail-loud）")
    if not (order.index("language_model") < order.index("output_layer")
            < order.index("compute_language_model_loss")):
        raise ValueError("gpt_segments: GPTModel 段序变化（fail-loud）")
    return order
