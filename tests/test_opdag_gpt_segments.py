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
