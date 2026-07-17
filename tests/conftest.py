# tests/conftest.py
"""共享夹具：timesim 测试的真源 MLP 段构建（三处复制收敛于此——Task 11 review）。"""
import os
import pytest

MF_ROOT = os.environ.get("MINDFORMERS_ROOT",
                         r"E:\97-codes\torch_parallel\mindformers\mindformers")


@pytest.fixture(scope="session")
def mlp_dag():
    """真 mindformers MLPInterleaved 提取 + shape 推断（producer 契约要求 infer_shapes）。"""
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
    dag = infer_shapes(dag, {"hidden_states": "S·B·H"})
    return dag
