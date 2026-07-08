# tests/test_opdag_extractor.py
"""PIECE 2:extractor 端到端 —— 读真 mindformers `MLPInterleaved` Cell,按 DSv3 config 剪枝,
产出 op-DAG。夹具 = 真源(仅 ast 静态读,绝不 import/执行 mindspore/mindformers)。

DSv3 dense FFN 读值(见 yaml `pretrain_deepseek3_671b.yaml`):
  * use_interleaved_weight_layout_mlp=True  → dense MLP cell = MLPInterleaved
  * gated_linear_unit=True                  → 走门控(GLU)主支
  * add_bias_linear=False                   → linear 无 bias,bias_parallel is None → 剪掉 add
  * bias_swiglu_fusion=False                → activation_type ≠ 'fusedswiglu'
  * hidden_act 未在 yaml 设置 → DeepseekV3Config 默认 'silu'(configuration_deepseek_v3.py:186);
    models/utils.py:249 只把它记进独立的 `swiglu` 标志,并不改写 hidden_act。
    ⇒ activation_type = 'silu' → 既非 'fusedswiglu' 也非 'swiglu' → 走 GLU 的 **else**(手工门控)支:
       split → reshape(x0) → reshape(x1) → silu 激活 → mul。
"""
import os

import pytest

from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.module_resolver import ResolvedSpec


MF_ROOT = os.environ.get(
    "MINDFORMERS_ROOT",
    r"E:\97-codes\torch_parallel\mindformers\mindformers",
)
MLP_REL = "parallel_core/training_graph/transformer/mlp.py"

# DSv3 dense-FFN 派生 flags(见文件头读值说明)。
DSV3_MLP_FLAGS = {
    "gated_linear_unit": True,
    "activation_type": "silu",   # hidden_act 默认 'silu' + bias_swiglu_fusion=False
    "add_bias_linear": False,
    "compute_dtype": "bf16",
}


@pytest.fixture(scope="module")
def mlp_spec():
    # dense MLP 的 get_mlp_module_spec:fc1=ColumnParallelLinear、fc2=RowParallelLinear(均叶子 MatMul)。
    return ResolvedSpec(
        cell="MLPInterleaved",
        submodules={
            "linear_fc1": "ColumnParallelLinear",
            "linear_fc2": "RowParallelLinear",
        },
    )


def _require_mf():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")


def test_extract_mlp_interleaved_pruned_op_sequence(mlp_spec):
    _require_mf()
    dag = extract_cell(MF_ROOT, MLP_REL, "MLPInterleaved", mlp_spec, DSV3_MLP_FLAGS)
    ops = [n.op for n in dag.nodes]
    # GLU else 支(手工门控):fc1 → reshape → split → reshape → reshape → silu → mul → fc2
    assert ops == ["MatMul", "View", "View", "View", "View", "Activation", "Elementwise", "MatMul"]


def test_extract_mlp_starts_and_ends_with_the_linears(mlp_spec):
    _require_mf()
    dag = extract_cell(MF_ROOT, MLP_REL, "MLPInterleaved", mlp_spec, DSV3_MLP_FLAGS)
    assert dag.nodes[0].op == "MatMul" and dag.nodes[0].module == "ColumnParallelLinear"   # linear_fc1
    assert dag.nodes[-1].op == "MatMul" and dag.nodes[-1].module == "RowParallelLinear"    # linear_fc2
    # 主线含且仅含一个激活
    assert [n.op for n in dag.nodes].count("Activation") == 1


def test_extract_mlp_src_lines_point_into_mlp_py(mlp_spec):
    _require_mf()
    dag = extract_cell(MF_ROOT, MLP_REL, "MLPInterleaved", mlp_spec, DSV3_MLP_FLAGS)
    assert all(n.src.startswith("mlp.py:") for n in dag.nodes)
    lines = [int(n.src.split(":")[1]) for n in dag.nodes]
    # 与真源逐行对齐(MLPInterleaved.construct):
    assert lines == [206, 212, 221, 222, 223, 224, 225, 230]


def test_extract_mlp_excludes_non_taken_branches(mlp_spec):
    _require_mf()
    dag = extract_cell(MF_ROOT, MLP_REL, "MLPInterleaved", mlp_spec, DSV3_MLP_FLAGS)
    lines = [int(n.src.split(":")[1]) for n in dag.nodes]
    # fusedswiglu 专属 transpose(214)不得出现;bias add(209)被 add_bias_linear=False 剪掉。
    assert 214 not in lines
    assert 209 not in lines
    # 没有从 fusedswiglu/swiglu 支泄漏进来的多余 op(节点数恰为 8)。
    assert len(dag.nodes) == 8
