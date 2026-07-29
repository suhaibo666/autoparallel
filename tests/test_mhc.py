"""Task 2.2 — mHC residual wrapper (TDD).

``residual_variant="mhc"`` packs the residual-carried hidden as n streams
``[S,B,H] -> [S,B,n*H]`` across the whole stack (act_live residual ×n), and each
decoder layer gains 2 HyperConnection modules (attn_hc before attention, ffn_hc
before ffn), each doing RMSNorm(n·H) + mapping_proj + sinkhorn -> h_res [S,B,n,n].

Source (mindformers, faithful):
  pynative/transformers/hyper_connection.py       — HyperConnectionModule.construct
  pynative/transformers/transformer_block.py       — expand/collapse streams
  pynative/transformers/transformer_layer.py       — HyperConnectionTransformerLayer
Design: specs/2026-07-01-unified-llm-modelspec-design.md §9.
"""
from cost_eval.model_spec import DimTable, OpType
from cost_eval.llm_config import LLMConfig, to_dimtable
from cost_eval.shape_eval import eval_expr
from cost_eval.layers.attention import build_gqa_attn_ops
from cost_eval.layers.ffn import build_dense_ffn_ops

N = 4  # num_residual_streams for the wrapped body

# tp=1-divisible small dims; num_residual_streams=N drives the ×n symbol.
DS = DimTable(
    H=16, F=32, n_heads=2, n_kv=2, head_dim=8, S=8, B=1, vocab=32, n_layers=3,
    num_residual_streams=N,
)


def _body(d):
    return build_gqa_attn_ops(d) + build_dense_ffn_ops(d)


def _names(ops):
    return [o.name for o in ops]


def _numel(t, d):
    n = 1
    for e in t.shape:
        n *= eval_expr(e, d)
    return n


# ---------------------------------------------------------------------------
# DimTable / to_dimtable — new inert field
# ---------------------------------------------------------------------------

def test_dimtable_num_residual_streams_default_one():
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)
    assert d.num_residual_streams == 1
    assert eval_expr("num_residual_streams", d) == 1


def test_to_dimtable_maps_num_residual_streams():
    cfg = LLMConfig(num_layers=2, hidden_size=16, num_attention_heads=2, vocab_size=32,
                    seq_length=8, residual_variant="mhc", num_residual_streams=4)
    assert to_dimtable(cfg).num_residual_streams == 4


# ---------------------------------------------------------------------------
# mhc_wrap — HC ops inserted, residual hidden ×n
# ---------------------------------------------------------------------------

def test_wrap_adds_two_hc_modules():
    from cost_eval.layers.residual import mhc_wrap
    body = _body(DS)
    wrapped = mhc_wrap(body, N, DS)
    names = _names(wrapped)
    # 2 HC modules (attn + ffn), each contributing norm + mapping_proj + sinkhorn.
    assert names.count("attn_hc_norm") == 1
    assert names.count("ffn_hc_norm") == 1
    assert sum(1 for n in names if n.endswith("_hc_mapping_proj")) == 2
    assert sum(1 for n in names if n.endswith("_hc_sinkhorn")) == 2
    # wrapped is strictly longer than the body by the HC ops.
    assert len(wrapped) == len(body) + 6


def test_wrap_h_res_shape_is_SBnn():
    from cost_eval.layers.residual import mhc_wrap
    wrapped = mhc_wrap(_body(DS), N, DS)
    h_res = next(op.output for op in wrapped if op.name.endswith("_hc_sinkhorn"))
    assert h_res.name.endswith("_h_res")
    # symbolic [S, B, n, n]
    assert h_res.shape == ("S", "B", "num_residual_streams", "num_residual_streams")
    assert _numel(h_res, DS) == DS.S * DS.B * N * N


def test_wrap_mapping_proj_weight_present():
    from cost_eval.layers.residual import mhc_wrap
    wrapped = mhc_wrap(_body(DS), N, DS)
    proj_ws = [w for op in wrapped for w in op.params if w.name.endswith("_hc_proj_w")]
    assert len(proj_ws) == 2                      # attn + ffn mapping_proj
    w = proj_ws[0]
    assert w.is_weight
    # input dim = n*H (packed residual streams)
    assert eval_expr(w.shape[0], DS) == N * DS.H


def test_wrap_scales_residual_hidden_by_n():
    """The layer-entry packed residual stream becomes [S,B,n*H]; ×n confirmed numerically.

    2026-07-29（普查收口 ②，docs/census_fix_residual_carrier_2026-07-29.md）——**不变量未变**
    （层入口的残差承载张量 `x` 被打包成 `x_xn [S,B,n·H]`，数值恰为 n×），**只换举例位置**：
    源 `transformer_layer.py:290-311` 逐字，`input_layernorm` 吃的是 `aggregated_attn [s,b,H]`，
    打包流 `streams_before_attn` 是**送进 attn_hc** 的那一个。故举例改用 attn_hc 的入流。
    """
    from cost_eval.layers.residual import mhc_wrap
    body = _body(DS)
    wrapped = mhc_wrap(body, N, DS)

    # attn_hc 的入流就是层入口打包流；P1-03(2026-07-14): 放大后重命名 x→x_xn(同名异形会被
    # 按名去重静默错算)。
    hc0 = wrapped[0]
    x_stream = next(t for t in hc0.inputs if t.name == "x_xn")
    assert "num_residual_streams*H" in x_stream.shape
    # numeric: exactly n× the unscaled [S,B,H] residual save.
    base = next(s for op in body if op.name == "ln1" for s in op.saves if s.name == "x")
    assert _numel(x_stream, DS) == N * _numel(base, DS)


def test_prenorm_saves_aggregated_not_packed_stream():
    """段首 layernorm 保留的是 HC 的 aggregated `[S,B,H]`，**不是**打包流（收口 ②）。

    源 `transformer_layer.py:308-311`：
        aggregated_attn, h_res_attn, h_post_attn = self.attn_hc(hidden_states)
        input_layernorm_output = self.input_layernorm(aggregated_attn)
    → ln1 的输入/saves 必须是 `attn_hc_agg [S,B,H]`；打包流 `x_xn` 不得再出现在 ln1 的 saves 里
    （那会把 32 MiB 的真值记成 128 MiB，并逼得 HC 模块不敢声明自己的 ctx `x`）。
    """
    from cost_eval.layers.residual import mhc_wrap
    wrapped = mhc_wrap(_body(DS), N, DS)
    ln1 = next(op for op in wrapped if op.name == "ln1")
    saved = {s.name for s in ln1.saves}
    assert "attn_hc_agg" in saved and "x_xn" not in saved
    agg = next(s for s in ln1.saves if s.name == "attn_hc_agg")
    assert agg.shape == ("S", "B", "H")
    # aggregated 由 attn_hc 的第 2 个 op 产出（两条融合分支同名同形，_link 按下标接边）。
    assert wrapped[1].output.name == "attn_hc_agg"


def test_wrap_scales_every_sp_residual_carrier():
    """All [S,B,H] shard={0:'sp'} residual carriers (x/h1/h2) are scaled ×n; the
    tp-partial sublayer outputs (o/o2) and norm outputs (ln1/ln2) are NOT."""
    from cost_eval.layers.residual import mhc_wrap
    wrapped = mhc_wrap(_body(DS), N, DS)
    # P1-03(2026-07-14): 放大后的承载张量统一重命名 {name}_xn，原名不得再出现（防同名异形）。
    carriers = {"x_xn", "h1_xn", "h2_xn"}
    seen = set()
    for op in wrapped:
        for t in (list(op.inputs) + [op.output] + list(op.saves)):
            if t.name in carriers:
                seen.add(t.name)
                assert t.shape == ("S", "B", "num_residual_streams*H"), t.name
            assert t.name not in ("x", "h1", "h2"), f"未重命名的承载张量 {t.name}"
            if t.name in ("o", "o2"):              # sublayer outputs stay [S,B,H]
                assert t.shape == ("S", "B", "H"), t.name
    assert carriers <= seen


def test_n1_is_noop():
    from cost_eval.layers.residual import mhc_wrap
    body = _body(DS)
    assert mhc_wrap(body, 1, DS) == body


# ---------------------------------------------------------------------------
# Integration — ShapeEval.resolve of a wrapped layer must not raise
# ---------------------------------------------------------------------------

def test_wrapped_layer_resolves():
    from cost_eval.layers.residual import mhc_wrap
    from cost_eval.model_spec import ModelSpec, LayerSpec
    from cost_eval.shape_eval import ShapeEval
    from cost_eval.specs import ParallelConfig
    from cost_eval.parallel_model import ParallelModel

    wrapped = mhc_wrap(_body(DS), N, DS)
    spec = ModelSpec("mhc", DS, layer_pattern=["mhc"], layer_specs={"mhc": LayerSpec(wrapped)})
    pm = ParallelModel(ParallelConfig(), n_layers=1, world_size=1)
    g = ShapeEval().resolve(spec, pm)
    assert len(g.stages[0]) == 1
