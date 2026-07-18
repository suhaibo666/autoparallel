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


@pytest.fixture(scope="session")
def mla_dag():
    """真 mindformers MLASelfAttention（DSv3, mla_qkv_concat=False）+ shape 推断。
    种子名是 `x`（MLA construct 入参，≠ MLP 的 hidden_states——2026-07-17 探针核实）。"""
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import resolve_layer_spec
    from cost_eval.opdag.shape_infer import infer_shapes
    spec_flags = {
        "multi_latent_attention": True, "mla_qkv_concat": False, "num_experts": 256,
        "qk_layernorm": True, "sparse_attention": False, "fused_norm": True,
        "moe_grouped_gemm": True, "use_contiguous_weight_layout_attention": False,
        "use_interleaved_weight_layout_mlp": True,
    }
    mla_flags = {
        "use_dsa": False, "use_flash_attention": True,
        "use_eod_attn_mask_compression": False, "cp": 1, "cp_ds": 1,
        "input_layout": "BNSD", "q_lora_rank": 1536,
        "compute_dtype": "bf16", "layernorm_compute_dtype": "fp32",
    }
    top = resolve_layer_spec(MF_ROOT, spec_flags)
    dag = extract_cell(
        MF_ROOT, "parallel_core/training_graph/transformer/multi_latent_attention.py",
        "MLASelfAttention", top.submodules["self_attention"], mla_flags,
        present_params={"rotary_pos_emb"})
    return infer_shapes(dag, {"x": "S·B·H"})
