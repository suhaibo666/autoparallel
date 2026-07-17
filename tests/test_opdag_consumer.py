# tests/test_opdag_consumer.py
"""T11:consumer 桥（符号 shape → 字节 via DimTable）。合成单测各解析规则 + 真 MLA/MoE 集成
（用 build_dsv3_spec(4) 的真 DimTable，字节数真实）。未解析入 margin 列表、不杜撰。"""
import math
import os

import pytest

from cost_eval.opdag.bprop_rules import Save
from cost_eval.opdag.consumer import (
    resolve_shape_elems, save_bytes, dag_saved_bytes, _cap_value,
)
from validate_dsv3 import build_dsv3_spec


@pytest.fixture(scope="module")
def dims():
    _, d, _ = build_dsv3_spec(4)
    return d


# ── 合成:解析规则 ────────────────────────────────────────────────────────────
def test_resolve_plain_shape(dims):
    assert resolve_shape_elems("S·B·H", dims) == dims.S * dims.B * dims.H   # 4096·1·1792


def test_resolve_sum_unit_axis(dims):
    # 一个轴 = n_heads·(qk_head_dim+qk_pos_emb_head_dim) = 8·(128+64) = 1536
    got = resolve_shape_elems("S·B·(n_heads·(qk_head_dim+qk_pos_emb_head_dim))", dims)
    assert got == dims.S * dims.B * (dims.n_heads * (dims.qk_nope_head_dim + dims.qk_rope_head_dim))


def test_resolve_coeff_axis(dims):
    # 2·moe_ffn 系数轴
    assert resolve_shape_elems("E·cap·(2·moe_ffn)", dims) == \
        dims.n_experts * _cap_value(dims) * (2 * dims.moe_F)


def test_save_bytes_fp32_vs_bf16(dims):
    s_fp32 = Save(name="q", dtype="fp32", op_id=1, sym_shape="S·B·q_lora_rank")
    s_bf16 = Save(name="q", dtype="bf16", op_id=1, sym_shape="S·B·q_lora_rank")
    elems = dims.S * dims.B * dims.q_lora_rank
    assert save_bytes(s_fp32, dims) == elems * 4        # fp32 = ×4（norm 残差机制）
    assert save_bytes(s_bf16, dims) == elems * 2


def test_unresolvable_returns_none_no_fabrication(dims):
    assert resolve_shape_elems("?", dims) is None
    assert resolve_shape_elems("S·B·frobnicate", dims) is None      # 未知 token
    assert save_bytes(Save("g", "bf16", 1, "?"), dims) is None


def test_cap_uses_routing_formula(dims):
    # cap = ceil(capacity_factor × S·B·topk / E) = ceil(1.0×4096×1×4/8) = 2048
    assert _cap_value(dims) == math.ceil(dims.capacity_factor * dims.S * dims.B * dims.topk / dims.n_experts)
    assert _cap_value(dims) == 2048


def test_dag_saved_bytes_buckets_unresolved(dims):
    # 合成 DAG:一个可解析 MatMul 存操作数 + 一个 ? 操作数 → 一进 per_save 一进 unresolved
    from cost_eval.opdag.schema import OpNode, OpDAG
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="MatMul", src="x:1", ins=["a:S·B·H:bf16", "ghost:?:bf16"], out="y:S·B·F:bf16"),
    ])
    r = dag_saved_bytes(dag, dims)
    names_ok = {p[0] for p in r["per_save"]}
    names_bad = {u[0] for u in r["unresolved"]}
    assert "a" in names_ok and "ghost" in names_bad
    assert r["total_bytes"] == dims.S * dims.B * dims.H * 2


# ── 真 mindformers 集成 ──────────────────────────────────────────────────────
MF_ROOT = os.environ.get("MINDFORMERS_ROOT", r"E:\97-codes\torch_parallel\mindformers\mindformers")


def _require_mf():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源不存在: {MF_ROOT}")


from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.module_resolver import ResolvedSpec, resolve_layer_spec
from cost_eval.opdag.shape_infer import infer_shapes

MLA_REL = "parallel_core/training_graph/transformer/multi_latent_attention.py"
DSV3_SPEC_FLAGS = {"multi_latent_attention": True, "mla_qkv_concat": False, "num_experts": 256,
                   "qk_layernorm": True, "sparse_attention": False, "fused_norm": True,
                   "moe_grouped_gemm": True, "use_contiguous_weight_layout_attention": False,
                   "use_interleaved_weight_layout_mlp": True}
DSV3_MLA_FLAGS = {"use_dsa": False, "use_flash_attention": True,
                  "use_eod_attn_mask_compression": False, "cp": 1, "cp_ds": 1,
                  "input_layout": "BNSD", "q_lora_rank": 1536, "compute_dtype": "bf16",
                  "layernorm_compute_dtype": "fp32"}


@pytest.fixture(scope="module")
def mla_bytes(dims):
    _require_mf()
    top = resolve_layer_spec(MF_ROOT, DSV3_SPEC_FLAGS)
    mla = top.submodules["self_attention"]
    dag = extract_cell(MF_ROOT, MLA_REL, "MLASelfAttention", mla, DSV3_MLA_FLAGS,
                       present_params={"rotary_pos_emb"})
    dag = infer_shapes(dag, {"x": "S·B·H"})
    return dag_saved_bytes(dag, dims)


def test_mla_total_positive_and_norms_fp32(mla_bytes, dims):
    assert mla_bytes["total_bytes"] > 0
    # 两个 norm 存 fp32:q_layernorm 输入=S·B·q_lora_rank、kv=S·B·kv_lora_rank,各按 ×4
    per = {p[0]: p for p in mla_bytes["per_save"]}
    fp32 = [p for p in mla_bytes["per_save"] if p[2] == "fp32"]
    assert len(fp32) >= 2, mla_bytes["per_save"]
    for name, shp, dt, b in fp32:
        assert b == resolve_shape_elems(shp, dims) * 4


def test_mla_all_saves_resolved(mla_bytes):
    # MLA 全 save 应可解析(无 ? / 未知 token)——否则打印哪些没解析
    assert mla_bytes["unresolved"] == [], mla_bytes["unresolved"]


# --- MoE FFNGroupedGEMM ---
FFN_REL = "parallel_core/training_graph/transformer/moe/ffn.py"
DSV3_FFN_FLAGS = {"moe_token_dispatcher_type": "alltoall", "compute_dtype": "bf16",
                  "add_bias_linear": False}


@pytest.fixture(scope="module")
def ffn_bytes(dims):
    _require_mf()
    spec = ResolvedSpec(cell="FFNGroupedGEMM", submodules={})
    dag = extract_cell(MF_ROOT, FFN_REL, "FFNGroupedGEMM", spec, DSV3_FFN_FLAGS)
    dag = infer_shapes(dag, {
        "tokens": "S·B·H", "dispatched_input": "E·cap·H",
        "w1": "(E·H)·(2·moe_ffn)", "w2": "(E·moe_ffn)·H"})
    return dag_saved_bytes(dag, dims)


def test_moe_dispatched_input_sized_e_cap_h(ffn_bytes, dims):
    # 18% 根因缓冲拿到具体字节:E·cap·H × 2(bf16)
    disp = [p for p in ffn_bytes["per_save"] if p[0].startswith("dispatched_input")]
    assert disp, ffn_bytes["per_save"]
    expect = dims.n_experts * _cap_value(dims) * dims.H * 2
    assert disp[0][3] == expect


def test_axis_value_public_name():
    """T0 交接要点6:跨包消费(timesim.shard_rules)用公名,私名保留兼容。"""
    from cost_eval.opdag import consumer
    assert consumer.axis_value is consumer._axis_value
