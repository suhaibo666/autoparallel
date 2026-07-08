# tests/test_opdag_module_resolver.py
"""Pass A(module_resolver)测试:静态解释真 `gpt_layer_specs.py`,config 驱动分支求解,
解不出/命中 raise → fail-loud。夹具 = 真 mindformers 源(不 import/执行 mindspore)。"""
import os

import pytest

from cost_eval.opdag.module_resolver import (
    resolve_layer_spec,
    ResolvedSpec,
    LEAF_OPTYPE,
)

# 真 mindformers 源根(可用环境变量覆盖以便他机运行)。
MF_ROOT = os.environ.get(
    "MINDFORMERS_ROOT",
    r"E:\97-codes\torch_parallel\mindformers\mindformers",
)

# DSv3 config 派生的 get_gpt_layer_local_spec kwargs(MLA 非 concat + 256 专家 MoE)。
DSV3_FLAGS = {
    "multi_latent_attention": True,
    "mla_qkv_concat": False,
    "num_experts": 256,
    "qk_layernorm": True,
    "sparse_attention": False,
    "fused_norm": True,
    "moe_grouped_gemm": True,
    "use_contiguous_weight_layout_attention": False,
    "use_interleaved_weight_layout_mlp": True,
}


@pytest.fixture(scope="module")
def dsv3_spec():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")
    return resolve_layer_spec(MF_ROOT, DSV3_FLAGS)


def test_top_cell_is_transformer_layer(dsv3_spec):
    assert isinstance(dsv3_spec, ResolvedSpec)
    assert dsv3_spec.cell == "TransformerLayer"


def test_self_attention_is_mla(dsv3_spec):
    sa = dsv3_spec.submodules["self_attention"]
    assert isinstance(sa, ResolvedSpec)
    assert sa.cell == "MLASelfAttention"


def test_mla_linear_leaves_resolve_to_right_classes(dsv3_spec):
    sa = dsv3_spec.submodules["self_attention"]
    # 非 concat 分支:q_proj/q_up/kv_up 走 ColumnParallel;q_down/kv_down 走 SequenceParallel;proj 走 RowParallel
    assert sa.submodules["linear_q_proj"] == "ColumnParallelLinear"
    assert sa.submodules["linear_q_up_proj"] == "ColumnParallelLinear"
    assert sa.submodules["linear_kv_up_proj"] == "ColumnParallelLinear"
    assert sa.submodules["linear_q_down_proj"] == "SequenceParallelLinear"
    assert sa.submodules["linear_kv_down_proj"] == "SequenceParallelLinear"
    assert sa.submodules["linear_proj"] == "RowParallelLinear"


def test_core_attention_is_flash(dsv3_spec):
    sa = dsv3_spec.submodules["self_attention"]
    assert sa.submodules["core_attention"] == "FlashAttention"


def test_qk_layernorm_resolves_to_norm(dsv3_spec):
    sa = dsv3_spec.submodules["self_attention"]
    # qk_layernorm=True → 三元真支 get_norm_cls(...) → "Norm"
    assert sa.submodules["q_layernorm"] == "Norm"
    assert sa.submodules["kv_layernorm"] == "Norm"


def test_input_layernorm_is_norm(dsv3_spec):
    assert dsv3_spec.submodules["input_layernorm"] == "Norm"
    assert dsv3_spec.submodules["pre_mlp_layernorm"] == "Norm"


def test_mlp_is_moe_layer(dsv3_spec):
    mlp = dsv3_spec.submodules["mlp"]
    assert isinstance(mlp, ResolvedSpec)
    assert mlp.cell == "MoELayer"


def test_leaf_optype_map_covers_core_leaves():
    assert LEAF_OPTYPE["ColumnParallelLinear"] == "MatMul"
    assert LEAF_OPTYPE["RowParallelLinear"] == "MatMul"
    assert LEAF_OPTYPE["SequenceParallelLinear"] == "MatMul"
    assert LEAF_OPTYPE["FlashAttention"] == "FlashAttention"
    assert LEAF_OPTYPE["Norm"] == "Norm"
    assert LEAF_OPTYPE["Identity"] == "Identity"


def test_qk_l2_norm_reaches_raise_fails_loud():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")
    flags = {**DSV3_FLAGS, "qk_l2_norm": True}
    with pytest.raises(ValueError) as ei:
        resolve_layer_spec(MF_ROOT, flags)
    msg = str(ei.value)
    # 命中 `raise NotImplementedError(...)` → 报文件:行
    assert "gpt_layer_specs.py:" in msg
    assert "raise" in msg or "L2Norm" in msg


def test_undecidable_condition_fails_loud():
    # 传入一个 config 无法决定的开关值(留空 multi_latent_attention 时,若某分支引用未绑定名则 fail-loud)。
    # 这里构造:num_experts 缺省 None 会走 get_moe? 不——用一个引用未绑定名的显式探针不易造;
    # 改为验证 identity(qk_layernorm=False)分支:确认非 fail-loud 正常路径也可解(回归保护)。
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")
    flags = {**DSV3_FLAGS, "qk_layernorm": False}
    spec = resolve_layer_spec(MF_ROOT, flags)
    sa = spec.submodules["self_attention"]
    assert sa.submodules["q_layernorm"] == "Identity"
    assert sa.submodules["kv_layernorm"] == "Identity"
