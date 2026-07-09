# tests/test_opdag_init_dims.py
"""PART A(T9):从 Cell.__init__ 静态抠出每个 `build_module(submodules.Y, <in>, <out>, ...)` 的
(in,out) 符号维度,并建 dims_ctx(self.<attr> → 符号 token,供 reshape/split 表达式解析)。
轻量符号求值:config.X→符号、self.X/局部名→环境、`*2`/`num_heads*head_dim`/和式 → 组合;
无法判定的分支/表达式 → 跳过或保留 None(不杜撰)。"""
import ast
import os

import pytest

from cost_eval.opdag.init_dims import eval_init_dims


# ── 合成 MLP:GLU 加倍 fc1、fc2 收回 H(受控,不依赖 mindformers)────────────────
SRC_MLP = '''
class MLP:
    def __init__(self, config, submodules, is_expert=False, input_size=None):
        self.input_size = input_size if input_size is not None else config.hidden_size
        if is_expert and config.moe_ffn_hidden_size is not None:
            map_ffn_hidden_size = config.moe_ffn_hidden_size
        else:
            map_ffn_hidden_size = config.ffn_hidden_size
        self.gated_linear_unit = config.gated_linear_unit
        if self.gated_linear_unit:
            map_ffn_hidden_size *= 2
        self.linear_fc1 = build_module(submodules.linear_fc1, self.input_size, map_ffn_hidden_size, config=config)
        self.linear_fc2 = build_module(submodules.linear_fc2, config.ffn_hidden_size, config.hidden_size, config=config)
'''


def test_mlp_glu_doubles_fc1_and_fc2_returns_hidden():
    d = eval_init_dims(ast.parse(SRC_MLP), "MLP", {"gated_linear_unit": True})
    assert d.linear_dims["linear_fc1"] == ("H", "2·ffn_hidden")
    assert d.linear_dims["linear_fc2"] == ("ffn_hidden", "H")


def test_mlp_non_glu_keeps_fc1_single():
    d = eval_init_dims(ast.parse(SRC_MLP), "MLP", {"gated_linear_unit": False})
    assert d.linear_dims["linear_fc1"] == ("H", "ffn_hidden")


# ── 真 mindformers MLA / MLP(仅 ast 静态读)──────────────────────────────────
MF_ROOT = os.environ.get(
    "MINDFORMERS_ROOT", r"E:\97-codes\torch_parallel\mindformers\mindformers")
MLA_REL = "parallel_core/training_graph/transformer/multi_latent_attention.py"
MLP_REL = "parallel_core/training_graph/transformer/mlp.py"


def _read(rel):
    p = os.path.join(MF_ROOT, *rel.split("/"))
    if not os.path.isfile(p):
        pytest.skip(f"mindformers 源不存在: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return fh.read()


MLA_FLAGS = {"q_lora_rank": 1536}


def test_real_mla_linear_dims():
    d = eval_init_dims(ast.parse(_read(MLA_REL)), "MLASelfAttention", MLA_FLAGS)
    ld = d.linear_dims
    assert ld["linear_q_down_proj"] == ("H", "q_lora_rank")
    assert ld["linear_q_up_proj"] == ("q_lora_rank", "n_heads·(qk_head_dim+qk_pos_emb_head_dim)")
    assert ld["linear_kv_down_proj"] == ("H", "kv_lora_rank+qk_pos_emb_head_dim")
    assert ld["linear_kv_up_proj"] == ("kv_lora_rank", "n_heads·(qk_head_dim+v_head_dim)")
    assert ld["linear_proj"] == ("n_heads·v_head_dim", "H")


def test_real_mla_dims_ctx_has_derived_head_dims():
    d = eval_init_dims(ast.parse(_read(MLA_REL)), "MLASelfAttention", MLA_FLAGS)
    ctx = d.dims_ctx
    assert ctx["num_attention_heads"] == "n_heads"
    assert ctx["q_head_dim"] == "qk_head_dim+qk_pos_emb_head_dim"
    assert ctx["query_projection_size"] == "n_heads·v_head_dim"


def test_real_mlp_interleaved_dims():
    d = eval_init_dims(ast.parse(_read(MLP_REL)), "MLPInterleaved",
                       {"gated_linear_unit": True})
    assert d.linear_dims["linear_fc1"] == ("H", "2·ffn_hidden")
    assert d.linear_dims["linear_fc2"] == ("ffn_hidden", "H")
