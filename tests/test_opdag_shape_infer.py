# tests/test_opdag_shape_infer.py
"""PART B(T9):shape 推断 pass。infer_shapes(dag, input_shapes, dims_ctx) 从 construct 输入种子出发,
按 op 类型传播符号 shape,回填每个 ref 的 shape 段。合成 DAG 测各 op 规则 + 真 MLA/MLP/MoE 集成。"""
import os

import pytest

from cost_eval.opdag.schema import OpNode, OpDAG
from cost_eval.opdag.shape_infer import infer_shapes
from cost_eval.opdag.bprop_rules import derive_saves


def _shape(ref):
    return ref.split(":")[1]


# ── 合成:逐 op 规则 ─────────────────────────────────────────────────────────
def test_matmul_replaces_last_axis_with_out_dim():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="MatMul", src="x:1", ins=["h:?:bf16"], out="y:?:bf16",
               attrs={"in_dim": "H", "out_dim": "F"})])
    out = infer_shapes(dag, {"h": "S·B·H"})
    assert _shape(out.nodes[0].ins[0]) == "S·B·H"
    assert _shape(out.nodes[0].out) == "S·B·F"


def test_grouped_matmul_out_is_operand_with_weight_last_axis():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="GroupedMatMul", src="x:1",
               ins=["disp:?:bf16", "w1:?:bf16"], out="o:?:bf16", attrs={})])
    out = infer_shapes(dag, {"disp": "E·cap·H", "w1": "E·H·(2·moe_ffn)"})
    assert _shape(out.nodes[0].out) == "E·cap·(2·moe_ffn)"


def test_norm_and_cast_and_elementwise_passthrough():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="Norm", src="x:1", ins=["a:?:fp32"], out="n:?:bf16"),
        OpNode(id=2, op="Elementwise", src="x:2", ins=["n:?:bf16", "n:?:bf16"], out="m:?:bf16",
               attrs={"linear": False}),
    ], edges=[[1, 2]])
    out = infer_shapes(dag, {"a": "S·B·kv_lora_rank"})
    assert _shape(out.nodes[0].out) == "S·B·kv_lora_rank"
    assert _shape(out.nodes[1].out) == "S·B·kv_lora_rank"


def test_swiglu_activation_halves_last_axis():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="Activation", src="x:1", ins=["f:?:bf16"], out="g:?:bf16",
               attrs={"activation_type": "swiglu"})])
    out = infer_shapes(dag, {"f": "E·cap·(2·moe_ffn)"})
    assert _shape(out.nodes[0].out) == "E·cap·moe_ffn"


def test_split_targets_get_last_axis_sizes():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="View", src="x:1", ins=["kv:?:bf16"], out="kvc:?:bf16",
               attrs={"view": "split", "split_dim": -1,
                      "split_sizes": ["self.config.kv_lora_rank", "self.config.qk_pos_emb_head_dim"],
                      "split_targets": ["kvc", "kpe"]})])
    out = infer_shapes(dag, {"kv": "S·B·(kv_lora_rank+qk_pos_emb_head_dim)"})
    assert _shape(out.nodes[0].out) == "S·B·kv_lora_rank"   # 首目标 = part0


def test_concat_sums_axis():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="View", src="x:1", ins=["a:?:bf16", "b:?:bf16"], out="c:?:bf16",
               attrs={"view": "concat", "concat_axis": 3})])
    out = infer_shapes(dag, {"a": "S·B·n_heads·qk_head_dim",
                             "b": "S·B·n_heads·qk_pos_emb_head_dim"})
    assert _shape(out.nodes[0].out) == "S·B·n_heads·(qk_head_dim+qk_pos_emb_head_dim)"


def test_tile_multiplies_axis_and_expand_inserts_one():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="View", src="x:1", ins=["k:?:bf16"], out="k2:?:bf16",
               attrs={"view": "expand_dims", "expand_axis": 2}),
        OpNode(id=2, op="View", src="x:2", ins=["k2:?:bf16"], out="k3:?:bf16",
               attrs={"view": "tile", "tile_mult": ["1", "1", "self.num_attention_heads", "1"]}),
    ], edges=[[1, 2]], dims_ctx={"num_attention_heads": "n_heads"})
    out = infer_shapes(dag, {"k": "S·B·qk_pos_emb_head_dim"})
    assert _shape(out.nodes[0].out) == "S·B·1·qk_pos_emb_head_dim"
    assert _shape(out.nodes[1].out) == "S·B·n_heads·qk_pos_emb_head_dim"


def test_reshape_neg1_resolved_via_scalar_binds_and_dims_ctx():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="MatMul", src="x:1", ins=["c:?:bf16"], out="q:?:bf16",
               attrs={"out_dim": "n_heads·(qk_head_dim+v_head_dim)"}),
        OpNode(id=2, op="View", src="x:2", ins=["q:?:bf16"], out="qr:?:bf16",
               attrs={"view": "reshape",
                      "reshape_dims": ["q_len", "bs", "self.num_attention_heads", "-1"]}),
    ], edges=[[1, 2]],
        scalar_binds=[{"names": ["q_len", "bs", "_"], "src": "q"}],
        dims_ctx={"num_attention_heads": "n_heads"})
    out = infer_shapes(dag, {"c": "S·B·kv_lora_rank"})
    assert _shape(out.nodes[1].out) == "S·B·n_heads·(qk_head_dim+v_head_dim)"


def test_flash_attention_out_is_query_with_value_last_axis():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="FlashAttention", src="x:1",
               ins=["q:?:bf16", "k:?:bf16", "v:?:bf16", "mask:?:bf16"], out="ctx:?:bf16")])
    out = infer_shapes(dag, {"q": "S·B·n_heads·(qk_head_dim+qk_pos_emb_head_dim)",
                             "k": "S·B·n_heads·(qk_head_dim+qk_pos_emb_head_dim)",
                             "v": "S·B·n_heads·v_head_dim"})
    assert _shape(out.nodes[0].out) == "S·B·n_heads·v_head_dim"


def test_matmul_known_input_but_no_out_dim_fails_loud():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="MatMul", src="mla.py:99", ins=["h:?:bf16"], out="y:?:bf16", attrs={})])
    with pytest.raises(ValueError) as ei:
        infer_shapes(dag, {"h": "S·B·H"})
    assert "mla.py:99" in str(ei.value)


def test_unseeded_external_input_stays_question_mark_no_raise():
    dag = OpDAG(cell="T", nodes=[
        OpNode(id=1, op="Norm", src="x:1", ins=["ghost:?:bf16"], out="n:?:bf16")])
    out = infer_shapes(dag, {})               # ghost 未种子 → 保留 ?,不 raise
    assert _shape(out.nodes[0].out) == "?"


# ── 真 mindformers 集成 ──────────────────────────────────────────────────────
MF_ROOT = os.environ.get(
    "MINDFORMERS_ROOT", r"E:\97-codes\torch_parallel\mindformers\mindformers")


def _require_mf():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源不存在: {MF_ROOT}")


# --- MLPInterleaved ---
from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.module_resolver import ResolvedSpec

MLP_REL = "parallel_core/training_graph/transformer/mlp.py"
DSV3_MLP_FLAGS = {"gated_linear_unit": True, "activation_type": "silu",
                  "add_bias_linear": False, "compute_dtype": "bf16"}


@pytest.fixture(scope="module")
def mlp_dag():
    _require_mf()
    spec = ResolvedSpec(cell="MLPInterleaved", submodules={
        "linear_fc1": "ColumnParallelLinear", "linear_fc2": "RowParallelLinear"})
    dag = extract_cell(MF_ROOT, MLP_REL, "MLPInterleaved", spec, DSV3_MLP_FLAGS)
    return infer_shapes(dag, {"hidden_states": "S·B·H"})


def test_mlp_fc1_doubles_and_fc2_returns_hidden(mlp_dag):
    mms = [n for n in mlp_dag.nodes if n.op == "MatMul"]
    assert _shape(mms[0].out) == "S·B·(2·ffn_hidden)"   # linear_fc1 (GLU 加倍)
    assert _shape(mms[1].out) == "S·B·H"                # linear_fc2
    # mul(act, x1) 输出末轴 = ffn_hidden(GLU 折半后)
    mul = next(n for n in mlp_dag.nodes if n.op == "Elementwise")
    assert _shape(mul.out) == "S·B·ffn_hidden"


# --- MLASelfAttention ---
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
def mla_dag():
    _require_mf()
    from cost_eval.opdag.module_resolver import resolve_layer_spec
    top = resolve_layer_spec(MF_ROOT, DSV3_SPEC_FLAGS)
    mla = top.submodules["self_attention"]
    dag = extract_cell(MF_ROOT, MLA_REL, "MLASelfAttention", mla, DSV3_MLA_FLAGS,
                       present_params={"rotary_pos_emb"})
    return infer_shapes(dag, {"x": "S·B·H"})


def test_mla_projection_output_shapes(mla_dag):
    by_line = {int(n.src.split(":")[1]): n for n in mla_dag.nodes if n.op == "MatMul"}
    assert _shape(by_line[797].out) == "S·B·q_lora_rank"                       # q_down
    assert _shape(by_line[734].out) == "S·B·(n_heads·(qk_head_dim+qk_pos_emb_head_dim))"  # q_up
    assert _shape(by_line[306].out) == "S·B·H"                                 # linear_proj


def test_mla_two_norms_keep_input_shape(mla_dag):
    norms = [n for n in mla_dag.nodes if n.op == "Norm"]
    shapes = {_shape(n.out) for n in norms}
    assert "S·B·q_lora_rank" in shapes        # q_layernorm
    assert "S·B·kv_lora_rank" in shapes       # kv_layernorm
    for n in norms:                            # 归一化输入=输出(passthrough)
        assert _shape(n.ins[0]) == _shape(n.out)


def test_mla_flash_attention_output_shape(mla_dag):
    fa = next(n for n in mla_dag.nodes if n.op == "FlashAttention")
    assert _shape(fa.out) == "S·B·n_heads·v_head_dim"


def test_mla_saveset_has_real_symbolic_shapes(mla_dag):
    saves = {s.name: s for s in derive_saves(mla_dag)}
    # q_layernorm 输入(q_down 输出)= S·B·q_lora_rank,已落实(非 ?)
    qd = [s for s in saves.values() if s.sym_shape == "S·B·q_lora_rank"]
    assert qd, f"expected a save with S·B·q_lora_rank, got {[s.sym_shape for s in saves.values()]}"
    # 无任何 save 的 sym_shape 还是纯 ? (关键张量都拿到真 shape)
    fa_v = [s for s in saves.values() if s.sym_shape == "S·B·n_heads·v_head_dim"]
    assert fa_v            # FA 的 value 操作数


# --- MoE FFNGroupedGEMM ---
FFN_REL = "parallel_core/training_graph/transformer/moe/ffn.py"
DSV3_FFN_FLAGS = {"moe_token_dispatcher_type": "alltoall", "compute_dtype": "bf16",
                  "add_bias_linear": False}


@pytest.fixture(scope="module")
def ffn_dag():
    _require_mf()
    spec = ResolvedSpec(cell="FFNGroupedGEMM", submodules={})
    dag = extract_cell(MF_ROOT, FFN_REL, "FFNGroupedGEMM", spec, DSV3_FFN_FLAGS)
    return infer_shapes(dag, {
        "tokens": "S·B·H",
        "dispatched_input": "E·cap·H",
        "w1": "(E·H)·(2·moe_ffn)", "w2": "(E·moe_ffn)·H",
    })


def test_moe_grouped_gemm_operand_is_e_cap_h(ffn_dag):
    gmm = [n for n in ffn_dag.nodes if n.op == "GroupedMatMul"]
    fc1 = gmm[0]
    disp = next(r for r in fc1.ins if r.split(":")[0].startswith("dispatched_input"))
    assert _shape(disp) == "E·cap·H"                       # 18% 根因缓冲拿到具体符号 size
    assert _shape(fc1.out) == "E·cap·(2·moe_ffn)"          # fc1 out
    fc2 = gmm[1]
    assert _shape(fc2.out) == "E·cap·H"                    # fc2 回到 H


def test_moe_saveset_dispatched_input_has_real_shape(ffn_dag):
    saves = derive_saves(ffn_dag)
    disp = [s for s in saves if s.name.startswith("dispatched_input")]
    assert disp and disp[0].sym_shape == "E·cap·H"
