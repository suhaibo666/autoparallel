"""预设模型工厂：把常见 LLM 的真实超参固化为 `LLMConfig` 工厂函数。

每个工厂返回一份 `LLMConfig`（内存结构子集），交给 `build_llm_spec` 装配成
`ModelSpec`。默认值取自各模型公开的架构超参；调用方可覆盖 `num_layers` 等维度做
缩层/变体研究。

- `deepseek_v3(N)`：DeepSeek-V3 缩层结构（MLA + first_k_dense=1 + 8 专家 top4 +
  1 shared expert），**逐字段等于** `validate_dsv3.build_dsv3_spec(N)[0]` 的 DimTable，
  用于复现真机峰值锚点（Task 1.3 硬门：N=4→12472.5 MiB，N=8→13896.1 MiB）。
"""
from __future__ import annotations

from .llm_config import LLMConfig


def deepseek_v3(num_layers: int = 4) -> LLMConfig:
    """DeepSeek-V3 缩层预设（MLA + MoE），复现真机峰值锚点。

    维度逐字段对齐 `validate_dsv3.build_dsv3_spec` 的 `DimTable`
    （H=1792, MLA lora/rope/nope/v dims, 8 专家 top4, moe_F=1024, 1 shared expert,
    first_k_dense_replace=1, compute=bf16）。层序列展开为
    `["embedding", "mla_dense", "mla_moe"*(N-1), "lm_head"]`，即 1 个 dense MLA 层 +
    (N-1) 个 MoE MLA 层，与 `build_dsv3_spec` 的 `layer_pattern` 一致 → 峰值 byte-identical。

    参数
    ----
    num_layers : int
        transformer 层数 N（不含 embedding / lm_head）。
    """
    return LLMConfig(
        num_layers=num_layers,
        hidden_size=1792,
        num_attention_heads=8,
        num_query_groups=8,
        vocab_size=129280,
        seq_length=4096,
        batch_size=1,
        head_dim=192,
        attn_type="mla",
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=192,
        ffn_hidden_size=3072,
        num_moe_experts=8,
        moe_router_topk=4,
        moe_ffn_hidden_size=1024,
        moe_shared_expert_num=1,
        moe_shared_ffn_hidden_size=1024,
        moe_capacity_factor=1.0,
        first_k_dense_replace=1,
        loss_type="logsoftmax_nll",
        compute_dtype_bytes=2,
    )


def llama(
    num_layers: int = 32,
    hidden_size: int = 4096,
    num_attention_heads: int = 32,
    num_query_groups: int = 32,
    vocab_size: int = 32000,
    seq_length: int = 4096,
    ffn_hidden_size: int = 11008,
    batch_size: int = 1,
    head_dim: int | None = None,
    tie_word_embeddings: bool = False,
) -> LLMConfig:
    """Llama 系（GQA + dense SwiGLU）预设，默认 = **Llama-2-7B**（≈6.7e9 params）。

    默认超参：H=4096, 32 层, 32 头（num_query_groups=32 → 对称 GQA/MHA），
    vocab=32000, ffn=11008, head_dim=128（H//n_heads），tie=False（独立 lm_head）。
    覆盖 `num_query_groups < num_attention_heads` 即得真正 GQA（如 Llama-3 8B/70B）。

    param 守恒（全 1 配置）：per-layer attn=(H+2·n_kv·d)·H + n_heads·d·H，
    dense ffn=(2F+F)·H，×层数 + emb(vocab×H) + head(H×vocab) ≈ 6.738e9。
    """
    return LLMConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        vocab_size=vocab_size,
        seq_length=seq_length,
        batch_size=batch_size,
        head_dim=head_dim,
        attn_type="gqa",
        ffn_hidden_size=ffn_hidden_size,
        gated_linear_unit=True,
        tie_word_embeddings=tie_word_embeddings,
        loss_type="logsoftmax_nll",
        compute_dtype_bytes=2,
    )


def qwen2(
    num_layers: int = 28,
    hidden_size: int = 1536,
    num_attention_heads: int = 12,
    num_query_groups: int = 2,
    vocab_size: int = 151936,
    seq_length: int = 4096,
    ffn_hidden_size: int = 8960,
    batch_size: int = 1,
    head_dim: int | None = None,
    tie_word_embeddings: bool = True,
    qk_layernorm: bool = False,
    add_qkv_bias: bool = True,
) -> LLMConfig:
    """Qwen2 系（GQA + dense SwiGLU）预设，默认 = **Qwen2-1.5B**。

    默认超参：H=1536, 28 层, 12 头 / 2 KV 组（真 GQA），vocab=151936, ffn=8960,
    head_dim=128, tie_word_embeddings=True（小模型 tie）。`add_qkv_bias=True`（Qwen QKV 带
    偏置）；`qk_layernorm=True` 为 Qwen3 变体标记。

    注：Tier-1 的 `build_gqa_attn_ops` op 图不随 `qk_layernorm`/`add_qkv_bias` 变化
    （bias / q-k norm 的显存量级可忽略，未建独立 op）——这两个字段作为架构标记透传，
    不改变内存 op 图。param ref 可选：本预设主要验证"能建能评估"。
    """
    return LLMConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        vocab_size=vocab_size,
        seq_length=seq_length,
        batch_size=batch_size,
        head_dim=head_dim,
        attn_type="gqa",
        ffn_hidden_size=ffn_hidden_size,
        gated_linear_unit=True,
        qk_layernorm=qk_layernorm,
        add_qkv_bias=add_qkv_bias,
        tie_word_embeddings=tie_word_embeddings,
        loss_type="logsoftmax_nll",
        compute_dtype_bytes=2,
    )


def mixtral(
    num_layers: int = 32,
    hidden_size: int = 4096,
    num_attention_heads: int = 32,
    num_query_groups: int = 8,
    vocab_size: int = 32000,
    seq_length: int = 4096,
    num_moe_experts: int = 8,
    moe_router_topk: int = 2,
    moe_ffn_hidden_size: int = 14336,
    batch_size: int = 1,
    head_dim: int | None = None,
    tie_word_embeddings: bool = False,
) -> LLMConfig:
    """Mixtral 系（GQA + 每层 MoE，无 shared expert）预设，默认 = **Mixtral-8x7B**（≈46.7e9 params）。

    默认超参：H=4096, 32 层, 32 头 / 8 KV 组（GQA），vocab=32000, 8 专家 top2,
    moe_ffn=14336。`moe_shared_expert_num=0`（无共享专家）、`first_k_dense_replace=0`
    （全层 MoE，无前置 dense）——区别于 DeepSeek-V3 的 shared+first_k_dense 结构。

    param 守恒（全 1 配置）：per-layer attn=(H+2·n_kv·d)·H + n_heads·d·H，
    MoE ffn = n_experts·(2·moe_F + moe_F)·H，×层数 + emb + head ≈ 46.70e9
    （router 权重 H·n_experts 量级可忽略，未建 param）。
    """
    return LLMConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        vocab_size=vocab_size,
        seq_length=seq_length,
        batch_size=batch_size,
        head_dim=head_dim,
        attn_type="gqa",
        num_moe_experts=num_moe_experts,
        moe_router_topk=moe_router_topk,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        moe_shared_expert_num=0,
        first_k_dense_replace=0,
        tie_word_embeddings=tie_word_embeddings,
        loss_type="logsoftmax_nll",
        compute_dtype_bytes=2,
    )
