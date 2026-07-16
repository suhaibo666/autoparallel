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
    "calculate_per_token_loss": False,   # TransformerConfig 默认（:338）
    "seq_pipe": False,                   # seq_split_num 默认 1（:262→:338）
}


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
    assert [n.op for n in dag.nodes] == [
        "View", "Activation", "Elementwise", "Elementwise", "Gather", "Elementwise", "View"]
    assert [int(n.src.split(":")[1]) for n in dag.nodes] == [149, 153, 154, 155, 160, 165, 167]
    assert all(n.src.split(":")[0] == "layers.py" for n in dag.nodes)


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
