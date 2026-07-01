"""M1：装配件 op-builder —— embedding / lm_head+loss（设计 §6/§10）。

`build_embedding_ops` / `build_head_and_loss_ops` 从 `validate_dsv3.build_embedding` /
`build_lm_head` **逐字段移植**（同 TensorRef / dtype / saves / nll 的
`bwd_scratch="8*S*B*vocab"`），使 `build_llm_spec(deepseek_v3(N))` 复现现有
`build_dsv3_spec(N)` 的 embedding/head 段（Task 1.3 硬门）。返回 op 列表（list[OpSpec]），
由装配器包成 `LayerSpec`，与 `build_*_attn_ops`/`build_*_ffn_ops` 约定一致。
"""
from __future__ import annotations

from ..llm_config import LLMConfig
from ..model_spec import OpSpec, OpType, TensorRef


def build_embedding_ops(cfg: LLMConfig) -> list:
    """word embedding 段（1 op），逐字段同 `validate_dsv3.build_embedding`。

    ``emb_w (vocab,H)`` 为 vocab embedding 权重（tp=1 不切）；输出 ``emb_out (S,B,H)``
    在 SP 轴分布。ELEMENTWISE（gather 语义），无 saves。
    """
    w = TensorRef("emb_w", ("vocab", "H"), is_weight=True)        # vocab_emb_dp：tp=1 不切
    out = TensorRef("emb_out", ("S", "B", "H"), shard={0: "sp"})
    return [OpSpec("embedding", OpType.ELEMENTWISE, [], out, params=[w], saves=[])]


def build_head_and_loss_ops(cfg: LLMConfig) -> list:
    """lm_head + loss 段（3 op），逐字段同 `validate_dsv3.build_lm_head`。

    对照 loss.py：logits(bf16) → cast fp32 → log_softmax(fp32,saved) → NLL；
    反向物化 probs(fp32)。NLL 反向同时物化 probs=exp(-log_softmax) 与 scatter_add 出的
    grad_log_softmax，二者皆 fp32 满 vocab、与 saved log_softmax 共存（loss.py:80-82）
    → ``bwd_scratch="8*S*B*vocab"`` = 2×(4·S·B·vocab)。

    ``tie_word_embeddings=True`` 时 lm_head 复用 embedding 权重（无独立 head_w 参数，
    ``params=[]``，不重复计入 vocab×H 持久量）；DeepSeek-V3 tie=False，走默认路径，
    逐字段等于 `validate_dsv3.build_lm_head`。
    """
    if cfg.loss_type != "logsoftmax_nll":
        raise NotImplementedError(
            f"loss_type={cfg.loss_type!r} 暂未建 op 图（Phase 2：chunked / vocab_parallel_ce）")

    # 对照 loss.py：logits(bf16) → cast fp32 → log_softmax(fp32,saved) → NLL；反向物化 probs(fp32)
    x = TensorRef("h_final", ("S", "B", "H"), shard={0: "sp"})
    logits = TensorRef("logits_lm", ("S", "B", "vocab"))                      # bf16, saved(ctx.logits)
    logsm = TensorRef("logsm", ("S", "B", "vocab"), dtype_bytes=4)            # fp32, saved
    loss = TensorRef("loss", ("B",))

    if cfg.tie_word_embeddings:
        # 复用 embedding 权重：不新增 head_w 参数（vocab×H 只在 embedding 计一次）
        w = TensorRef("emb_w", ("vocab", "H"), is_weight=True)
        head_op = OpSpec("lm_head", OpType.MATMUL, [x, w], logits, params=[], saves=[x])
    else:
        w = TensorRef("head_w", ("H", "vocab"), is_weight=True)
        head_op = OpSpec("lm_head", OpType.MATMUL, [x, w], logits, params=[w], saves=[x])

    return [
        head_op,
        OpSpec("logsoftmax", OpType.NORM, [logits], logsm, saves=[logits]),
        # NLL 反向同时物化 probs 与 grad_log_softmax（fp32 满 vocab，与 saved log_softmax 共存）
        OpSpec("nll", OpType.ELEMENTWISE, [logsm], loss, saves=[logsm], bwd_scratch="8*S*B*vocab"),
    ]
