"""D-6：ungated MLP（gated_linear_unit=False）op 图 —— fc1 输出 F（非 2F）。

gated（默认 SwiGLU）fc1 输出 2·F（gate+up）；ungated（GPT-2 式 plain MLP）fc1 输出 F、
纯 gelu。此前 build_llm fail-loud 拒 ungated；D-6 按 `d.gated_linear_unit` 分派建图。
"""
from dataclasses import replace

from cost_eval.model_spec import DimTable
from cost_eval.shape_eval import eval_expr
from cost_eval.layers.ffn import build_dense_ffn_ops, build_moe_ffn_ops, build_shared_expert_ops
from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec

D = DimTable(H=16, F=32, n_heads=2, n_kv=2, head_dim=8, S=8, B=1, vocab=32, n_layers=3,
             n_experts=4, topk=2, moe_F=16, moe_shared_F=16)
D_UNGATED = replace(D, gated_linear_unit=False)


def _out_dim(ops, op_name, d):
    op = next(o for o in ops if o.name == op_name)
    # 末维符号求值
    return eval_expr(op.output.shape[-1], d)


def test_dense_gated_fc1_outputs_2F():
    assert _out_dim(build_dense_ffn_ops(D), "fc1", D) == 2 * D.F
    assert any(o.name == "swiglu" for o in build_dense_ffn_ops(D))


def test_dense_ungated_fc1_outputs_F():
    ops = build_dense_ffn_ops(D_UNGATED)
    assert _out_dim(ops, "fc1", D_UNGATED) == D_UNGATED.F      # 不是 2F
    assert any(o.name == "gelu" for o in ops)
    assert not any(o.name == "swiglu" for o in ops)


def test_dense_ungated_fc1_weight_half():
    """ungated fc1_w = [H,F]，gated = [H,2F]：param numel 减半。"""
    g = next(o for o in build_dense_ffn_ops(D) if o.name == "fc1").params[0]
    u = next(o for o in build_dense_ffn_ops(D_UNGATED) if o.name == "fc1").params[0]
    gn = eval_expr(g.shape[0], D) * eval_expr(g.shape[1], D)
    un = eval_expr(u.shape[0], D_UNGATED) * eval_expr(u.shape[1], D_UNGATED)
    assert un * 2 == gn


def test_moe_ungated_expert_fc1_outputs_moeF():
    ops = build_moe_ffn_ops(D_UNGATED)
    assert _out_dim(ops, "e_fc1", D_UNGATED) == D_UNGATED.moe_F   # 不是 2*moe_F
    assert any(o.name == "e_gelu" for o in ops)


def test_shared_expert_ungated_outputs_moe_shared_F():
    ops = build_shared_expert_ops(D_UNGATED)
    assert _out_dim(ops, "shared_fc1", D_UNGATED) == D_UNGATED.moe_shared_F
    assert any(o.name == "shared_gelu" for o in ops)


def test_build_llm_spec_ungated_no_longer_raises():
    cfg = LLMConfig(num_layers=2, hidden_size=16, num_attention_heads=2, vocab_size=32,
                    seq_length=8, attn_type="gqa", num_query_groups=2,
                    ffn_hidden_size=32, gated_linear_unit=False)
    spec = build_llm_spec(cfg)          # 此前 raise NotImplementedError；D-6 后应成功
    assert spec.dims.gated_linear_unit is False
