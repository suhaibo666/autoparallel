"""Task 2.3 — MTP head + loss 变体（chunked / vocab_parallel_ce）（TDD）。

设计 §10：
  - `chunked`（chunk_loss_num=k）：分块 CE 一次只物化 1/k 满 vocab 梯度 → loss 区
    `bwd_scratch ÷ k`（`8*S*B*vocab // k`）。
  - `vocab_parallel_ce`：logits 按 vocab 切（tp）→ loss 区大张量 ∝1/tp。
  - MTP 头 ≈ embedding + 1 个 decoder 层 + head。

源码忠实映射（mindformers pynative）：
  - `loss/loss.py` `_LogSoftmax`/`_NLLLoss`（默认路径）、`_ChunkCrossEntropyLoss`
    （分块，:376-465）、`_VocabParallelCrossEntropy`（vocab 切，:95-134，ctx.exp_vals 保存）。
  - `transformers/multi_token_prediction.py` `MultiTokenPredictionLayer`（enorm/hnorm/
    eh_proj + 1 transformer_layer + final_norm + 共享 head，:245-404）。

铁律：默认 `logsoftmax_nll` 路径**逐字节不变**（DSv3 硬门）。
"""
import pytest

from cost_eval.llm_config import LLMConfig, to_dimtable
from cost_eval.shape_eval import eval_expr, resolve_tensor
from cost_eval.layers.head import build_head_and_loss_ops, build_mtp_ops
from cost_eval.specs import ParallelConfig
from cost_eval.parallel_model import ParallelModel


def _cfg(**kw):
    base = dict(num_layers=2, hidden_size=16, num_attention_heads=2, vocab_size=64,
                seq_length=8, batch_size=1, attn_type="gqa")
    base.update(kw)
    return LLMConfig(**base)


def _names(ops):
    return [o.name for o in ops]


# ---------------------------------------------------------------------------
# Default logsoftmax_nll — MUST stay byte-identical (DSv3 hard gate)
# ---------------------------------------------------------------------------

def test_default_logsoftmax_nll_unchanged():
    ops = build_head_and_loss_ops(_cfg())
    # P1-01(2026-07-14): 补主干 final_norm op(真机 output_layer 前 final_layernorm,此前整体缺失)
    assert _names(ops) == ["final_norm", "lm_head", "logsoftmax", "nll"]
    nll = next(op for op in ops if op.name == "nll")
    assert nll.bwd_scratch == "8*S*B*vocab"
    logits = next(op.output for op in ops if op.name == "lm_head")
    logsm = next(op.output for op in ops if op.name == "logsoftmax")
    assert logits.shard == {}                 # no vocab shard on default path
    assert logsm.shard == {}
    # no extra 'probs' tensor on the default path
    assert not any(s.name == "probs" for op in ops for s in op.saves)


# ---------------------------------------------------------------------------
# (a) chunked — bwd_scratch ÷ chunk_loss_num
# ---------------------------------------------------------------------------

def test_chunked_bwd_scratch_divided_by_k():
    cfg = _cfg(loss_type="chunked", chunk_loss_num=4)
    ops = build_head_and_loss_ops(cfg)
    nll = next(op for op in ops if op.name == "nll")
    assert nll.bwd_scratch == "8*S*B*vocab//4"
    d = to_dimtable(cfg)
    # numeric ÷4 vs the default 8*S*B*vocab
    assert eval_expr(nll.bwd_scratch, d) == eval_expr("8*S*B*vocab", d) // 4
    assert eval_expr(nll.bwd_scratch, d) == eval_expr("2*S*B*vocab", d)


def test_chunked_k1_equals_default_scratch():
    """Guard chunk_loss_num>=1: k=1 (or unset) degrades to the full 8*S*B*vocab."""
    cfg = _cfg(loss_type="chunked", chunk_loss_num=1)
    nll = next(op for op in build_head_and_loss_ops(cfg) if op.name == "nll")
    d = to_dimtable(cfg)
    assert eval_expr(nll.bwd_scratch, d) == eval_expr("8*S*B*vocab", d)


def test_chunked_does_not_shard_vocab():
    cfg = _cfg(loss_type="chunked", chunk_loss_num=4)
    ops = build_head_and_loss_ops(cfg)
    logits = next(op.output for op in ops if op.name == "lm_head")
    assert logits.shard == {}                 # chunked stays vocab-replicated


# ---------------------------------------------------------------------------
# (b) vocab_parallel_ce — loss-region tensors sharded on vocab (∝1/tp)
# ---------------------------------------------------------------------------

def test_vocab_parallel_ce_shards_vocab_axis():
    cfg = _cfg(loss_type="vocab_parallel_ce")
    ops = build_head_and_loss_ops(cfg)
    logits = next(op.output for op in ops if op.name == "lm_head")
    logsm = next(op.output for op in ops if op.name == "logsoftmax")
    assert logits.shard == {2: "tp"}          # vocab is dim 2 of [S,B,vocab]
    assert logsm.shard == {2: "tp"}
    # backward softmax/grad materialization (probs, faithful to ctx.exp_vals) also sharded
    probs = next(s for op in ops for s in op.saves if s.name == "probs")
    assert probs.shard == {2: "tp"}


def test_vocab_parallel_ce_numeric_1_over_tp():
    cfg = _cfg(loss_type="vocab_parallel_ce")
    d = to_dimtable(cfg)
    logits = next(op.output for op in build_head_and_loss_ops(cfg) if op.name == "lm_head")
    pm = ParallelModel(ParallelConfig(tp=4), n_layers=1, world_size=4)
    rt = resolve_tensor(logits, d, pm)
    assert rt.local_numel == cfg.seq_length * cfg.batch_size * (cfg.vocab_size // 4)


def test_vocab_parallel_ce_bad_type_raises():
    cfg = _cfg(loss_type="not_a_loss")
    with pytest.raises(NotImplementedError):
        build_head_and_loss_ops(cfg)


# ---------------------------------------------------------------------------
# (c) build_mtp_ops — embedding + one decoder layer + head
# ---------------------------------------------------------------------------

def test_mtp_structure_embedding_decoder_head():
    cfg = _cfg(mtp_num_layers=1)
    ops = build_mtp_ops(cfg)
    names = _names(ops)
    # embedding-like front
    assert "embedding" in names
    # MTP-specific projection (enorm/hnorm + eh_proj over cat[decoder,hidden] 2H->H)
    assert {"enorm", "hnorm", "eh_proj"} <= set(names)
    # one decoder layer (gqa attn + dense ffn present)
    assert "flash" in names and "swiglu" in names
    # shared head projection + loss
    assert "lm_head" in names
    logits = next(op.output for op in ops if op.name == "lm_head")
    assert logits.shape == ("S", "B", "vocab")
    loss = next(op.output for op in ops if op.name == "nll")
    assert loss.shape == ("B",)


def test_mtp_eh_proj_2h_to_h():
    cfg = _cfg(mtp_num_layers=1)
    ops = build_mtp_ops(cfg)
    d = to_dimtable(cfg)
    eh = next(op for op in ops if op.name == "eh_proj")
    w = next(w for w in eh.params if w.is_weight)
    # eh_proj input is the concat of (decoder_input, hidden) = 2H (mtp:306-307/:379-380)
    assert eval_expr(w.shape[0], d) == 2 * d.H


def test_mtp_op_count():
    cfg = _cfg(mtp_num_layers=1)
    ops = build_mtp_ops(cfg)
    # embedding(1) + enorm/hnorm/eh_cat/eh_proj(4) + gqa(6) + dense(5) + head(4 含 final_norm) = 20
    # P1-01(2026-07-14): head 段补 final_norm → 19→20
    assert len(ops) == 20


def test_mtp_resolves_without_error():
    from cost_eval.model_spec import ModelSpec, LayerSpec
    from cost_eval.shape_eval import ShapeEval
    cfg = _cfg(mtp_num_layers=1)
    d = to_dimtable(cfg)
    spec = ModelSpec("mtp", d, layer_pattern=["mtp"],
                     layer_specs={"mtp": LayerSpec(build_mtp_ops(cfg))})
    pm = ParallelModel(ParallelConfig(), n_layers=1, world_size=1)
    g = ShapeEval().resolve(spec, pm)
    assert len(g.stages[0]) == 1
