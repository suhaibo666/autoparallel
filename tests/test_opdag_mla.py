# tests/test_opdag_mla.py
"""PIECE:extractor 端到端抽取**真** mindformers `MLASelfAttention` Cell(DSv3, mla_qkv_concat=False)。

MLA 比 MLP 复杂:construct 定义在基类 `MultiLatentAttention`,并调用两个**内部方法**
`get_query_key_value_tensors` / `qkv_up_proj_and_rope_apply`(定义在派生类 `MLASelfAttention`),
walker 必须**内联**它们。夹具 = 真源(仅 ast 静态读,绝不 import/执行 mindspore/mindformers)。

DSv3 MLA 读值(q_lora_rank!=None、qk_layernorm=True、use_dsa=False、FA=True、cp=1、非 TND):
  下投影 q_down/kv_down(SequenceParallelLinear=MatMul)→ q/kv layernorm(Norm, fp32)
  → 上投影 q_up/kv_up(ColumnParallelLinear=MatMul)→ split/reshape/expand → rope(Elementwise)
  → concat → cast(bf16)→ core_attention(FlashAttention)→ reshape → linear_proj(RowParallelLinear)。
"""
import os

import pytest

from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.module_resolver import resolve_layer_spec
from cost_eval.opdag.bprop_rules import derive_saves


MF_ROOT = os.environ.get(
    "MINDFORMERS_ROOT",
    r"E:\97-codes\torch_parallel\mindformers\mindformers",
)
MLA_REL = "parallel_core/training_graph/transformer/multi_latent_attention.py"

# DSv3 → get_gpt_layer_local_spec kwargs(与 test_opdag_module_resolver 一致)。
DSV3_SPEC_FLAGS = {
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

# DSv3 → MLASelfAttention.construct 分支剪枝派生 flags(hoisted self.<flag> 与 self.config.<flag>)。
DSV3_MLA_FLAGS = {
    "use_dsa": False,
    "use_flash_attention": True,
    "use_eod_attn_mask_compression": False,
    "cp": 1,
    "cp_ds": 1,
    "input_layout": "BNSD",
    "q_lora_rank": 1536,
    "compute_dtype": "bf16",
    "layernorm_compute_dtype": "fp32",
}
# MLA 恒用 RoPE:rotary_pos_emb 虽默认 None,但真机总被传入(present)→ 走 rope apply 支。
PRESENT = {"rotary_pos_emb"}


def _require_mf():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")


@pytest.fixture(scope="module")
def mla_spec():
    _require_mf()
    top = resolve_layer_spec(MF_ROOT, DSV3_SPEC_FLAGS)
    return top.submodules["self_attention"]   # ResolvedSpec(cell="MLASelfAttention", ...)


@pytest.fixture(scope="module")
def mla_dag(mla_spec):
    _require_mf()
    return extract_cell(MF_ROOT, MLA_REL, "MLASelfAttention", mla_spec,
                        DSV3_MLA_FLAGS, present_params=PRESENT)


def test_mla_matmuls_are_the_five_projections(mla_dag):
    mm = [n for n in mla_dag.nodes if n.op == "MatMul"]
    # q_down, kv_down (Sequence) → q_up, kv_up (Column) → proj (Row)
    assert [n.module for n in mm] == [
        "SequenceParallelLinear", "SequenceParallelLinear",
        "ColumnParallelLinear", "ColumnParallelLinear",
        "RowParallelLinear",
    ]
    # q_lora_rank!=None ⇒ 不走 linear_q_proj(该支被剪)
    lines = [int(n.src.split(":")[1]) for n in mm]
    assert lines == [797, 802, 734, 767, 306]


def test_mla_has_two_fp32_layernorms(mla_dag):
    norms = [n for n in mla_dag.nodes if n.op == "Norm"]
    assert len(norms) == 2
    assert [int(n.src.split(":")[1]) for n in norms] == [798, 808]
    assert all(n.attrs.get("ln_compute_dtype") == "fp32" for n in norms)


def test_mla_has_one_flash_attention(mla_dag):
    fa = [n for n in mla_dag.nodes if n.op == "FlashAttention"]
    assert len(fa) == 1
    assert int(fa[0].src.split(":")[1]) == 293


def test_mla_rope_is_two_elementwise(mla_dag):
    rope = [n for n in mla_dag.nodes if n.op == "Elementwise"]
    assert [int(n.src.split(":")[1]) for n in rope] == [748, 754]


def test_mla_full_op_sequence(mla_dag):
    ops = [n.op for n in mla_dag.nodes]
    assert ops == [
        "View",                      # 233 shape
        "MatMul", "Norm",            # 797 q_down, 798 q_ln
        "MatMul", "View", "Norm",    # 802 kv_down, 804 split3d, 808 kv_ln
        "MatMul",                    # 734 q_up
        "View", "View", "View", "View",   # 739 shape,741 reshape,742 split,745 expand
        "Elementwise", "Elementwise",     # 748/754 rope q,k
        "MatMul",                    # 767 kv_up
        "View", "View",              # 768 reshape,769 split
        "View", "View", "View",      # 772 concat-q,773 tile,774 concat-k
        "Cast", "Cast", "Cast",      # 277/278/279 q,k,v -> bf16
        "FlashAttention",            # 293 core_attention
        "View",                      # 298 reshape
        "MatMul",                    # 306 linear_proj
        "Cast",                      # 307 output -> ori_dtype
    ]


def test_mla_all_src_point_into_mla_file(mla_dag):
    assert all(n.src.startswith("multi_latent_attention.py:") for n in mla_dag.nodes)
    for n in mla_dag.nodes:
        assert int(n.src.split(":")[1]) > 0


def test_mla_key_dataflow_edges(mla_dag):
    e = mla_dag.edges
    ids = {n.id: n for n in mla_dag.nodes}
    # q_down(2)->q_ln(3);kv split(5)->kv_ln(6);q_ln(3)->q_up(7);kv_ln(6)->kv_up(14)
    assert [2, 3] in e and [5, 6] in e and [3, 7] in e and [6, 14] in e
    # 三个 cast(20/21/22)-> FA(23)
    fa = next(n.id for n in mla_dag.nodes if n.op == "FlashAttention")
    casts = [n.id for n in mla_dag.nodes if n.op == "Cast"][:3]
    for c in casts:
        assert [c, fa] in e


def test_mla_saveset_norm_inputs_are_fp32(mla_dag):
    saves = derive_saves(mla_dag)
    norm_ids = {n.id for n in mla_dag.nodes if n.op == "Norm"}
    norm_saves = [s for s in saves if s.op_id in norm_ids]
    # 两个 layernorm 的输入各 pin 一份 fp32(fp32-残差机制)
    assert len(norm_saves) == 2
    assert all(s.dtype == "fp32" for s in norm_saves)


def test_mla_saveset_flash_pins_qkv_and_matmuls_pin_operands(mla_dag):
    saves = derive_saves(mla_dag)
    fa_id = next(n.id for n in mla_dag.nodes if n.op == "FlashAttention")
    fa_saves = [s for s in saves if s.op_id == fa_id]
    assert len(fa_saves) == 3            # q,k,v
    assert all(s.dtype == "bf16" for s in fa_saves)
    # 每个 MatMul 至少 pin 一个操作数(其激活输入)
    mm_ids = {n.id for n in mla_dag.nodes if n.op == "MatMul"}
    assert any(s.op_id in mm_ids for s in saves)
