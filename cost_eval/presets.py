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
        # B 标定 margin（2026-07-09）：select 重算下保留-MoE 层 loss 峰的 fp32-cast+碎片长尾
        #   （op 图粒度之下，源码级 op-DAG 提取证实，见 opdag_validation.md）。标定自真机
        #   select_attn（DSv3 8L）：18828 vs 无 margin 15488 → 缺口 = 1.9× 当前 kept-MoE 激活 → ~1.00（OOM-安全）。
        #   仅 select-kept-MoE 生效；full/no-recompute/select-keep-attn 不触发（锚点不破，见 mem_timeline._is_kept）。
        kept_frag_factor=1.9,
    )


def _v4_compress_ratios(num_layers: int) -> tuple:
    """每层 compress_ratio（`configuration_deepseek_v4.py:250` `compress_ratios=[128]*61+[0]`）。

    真机为「几乎全 HCA(128) + 末层滑窗(0)」；缩层预设改混 `0`(滑窗/MLA base)、`4`(CSA+DSA 索引器)、
    `128`(HCA) 三档循环，以在小 N 下同时覆盖三条注意力分支（§7.3）。
    """
    cycle = (0, 4, 128)
    return tuple(cycle[i % len(cycle)] for i in range(num_layers))


def deepseek_v4(num_layers: int = 4) -> LLMConfig:
    """DeepSeek-V4 缩层预设（dsv4_hybrid 注意力 + MoE + mHC 残差 + MTP）。

    结构三派发轴（设计 §3/§11）：
      ① 注意力 = `dsv4_hybrid`（DSA/CSA/HCA，每层按 compress_ratio 分支，§7.3）；
      ② FFN = MoE + shared expert（first_k_dense=1，与 V3 同）；
      ③ 残差 = `mhc`（HyperConnection，num_residual_streams=hc_mult=4，§9）；外加 MTP 头。

    **核心维度取自 DeepSeek-V3 缩层锚点**（H=1792 等，复用 V3 已真机验证的 MLA/MoE base，
    使峰值落在与 DSv3 锚点可比的量级）；**dsv4 前沿字段取自** mindformers
    `models/deepseek4/configuration_deepseek_v4.py`（下方逐字段标注 file:line）：

      - `attn_type="dsv4_hybrid"`      ← `experimental_attention_variant="dsv4_hybrid"`（:211）
      - `q_lora_rank=1536`             ← `q_lora_rank=1536`（:182）
      - `qk_rope_head_dim=64`          ← `qk_rope_head_dim=64`（:183）
      - `o_lora_rank=1024`             ← `o_lora_rank=1024`（:184）
      - `o_groups=16`                  ← `o_groups=16`（:185）
      - `dsa_indexer_n_heads=64`       ← `index_n_heads=64`（:186）
      - `dsa_indexer_head_dim=128`     ← `index_head_dim=128`（:187）
      - `dsa_indexer_topk=1024`        ← `index_topk=1024`（:188）
      - `csa_window_size=128`/`window_size=128` ← `sliding_window=128`（:192，内存中性 §7.4）
      - `moe_router_topk=6`            ← `num_experts_per_tok=6`（:172）
      - `moe_ffn_hidden_size=3072`     ← `moe_intermediate_size=3072`（:168）
      - `moe_shared_expert_num=1`      ← `n_shared_experts=1`（:173）
      - `num_residual_streams=4`       ← `hc_mult=4`（enable_hyper_connections=True，:195/:212）
      - `mtp_num_layers=1`             ← `num_nextn_predict_layers=1`（:171）
      - `vocab_size=129280`            ← `vocab_size=129280`（:167）

    缩放注记：真机 `hidden_size=7168`（:168）、`num_attention_heads=128`（:174）、`head_dim=512`
    （:181）、`n_routed_experts=384`（:169）、`num_hidden_layers=61`（:170）在此缩到 V3 锚点量级
    （H=1792、8 头、head_dim=192、8 专家、num_layers=N），`num_key_value_heads=1`（:175，MLA 下惰性）。
    """
    return LLMConfig(
        num_layers=num_layers,
        # 核心维度（DeepSeek-V3 缩层锚点，复用已验证 MLA/MoE base）
        hidden_size=1792,
        num_attention_heads=8,
        num_query_groups=1,                     # num_key_value_heads=1（:175，MLA 下惰性）
        vocab_size=129280,                      # :167
        seq_length=4096,
        batch_size=1,
        head_dim=192,
        # ① dsv4_hybrid 注意力
        attn_type="dsv4_hybrid",                # experimental_attention_variant（:211）
        q_lora_rank=1536,                       # :182
        kv_lora_rank=512,
        qk_rope_head_dim=64,                    # :183
        qk_nope_head_dim=128,
        v_head_dim=192,
        csa_compress_ratios=_v4_compress_ratios(num_layers),  # compress_ratios（:250）
        csa_window_size=128,                    # sliding_window（:192）
        dsa_indexer_n_heads=64,                 # index_n_heads（:186）
        dsa_indexer_head_dim=128,               # index_head_dim（:187）
        dsa_indexer_topk=1024,                  # index_topk（:188）
        o_groups=16,                            # o_groups（:185）
        o_lora_rank=1024,                       # o_lora_rank（:184）
        window_size=128,                        # sliding_window（:192，内存中性 §7.4）
        # ② FFN / MoE（first_k_dense=1 + shared expert，与 V3 同）
        ffn_hidden_size=3072,
        num_moe_experts=8,                      # 缩层（真机 n_routed_experts=384，:169）
        moe_router_topk=6,                      # num_experts_per_tok（:172）
        moe_ffn_hidden_size=3072,               # moe_intermediate_size（:168）
        moe_shared_expert_num=1,                # n_shared_experts（:173）
        moe_shared_ffn_hidden_size=3072,
        moe_capacity_factor=1.0,
        first_k_dense_replace=1,
        # ③ mHC 残差
        residual_variant="mhc",                 # enable_hyper_connections=True（:212）
        num_residual_streams=4,                 # hc_mult（:195）
        # MTP
        mtp_num_layers=1,                       # num_nextn_predict_layers（:171）
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
    add_qkv_bias: bool = False,
) -> LLMConfig:
    """Qwen2 系（GQA + dense SwiGLU）预设，默认 = **Qwen2-1.5B**。

    默认超参：H=1536, 28 层, 12 头 / 2 KV 组（真 GQA），vocab=151936, ffn=8960,
    head_dim=128, tie_word_embeddings=True（小模型 tie）。

    **`add_qkv_bias` / `qk_layernorm` 默认 False（Task 3 决策 b）**：Qwen 架构上 QKV
    投影确带偏置、Qwen3 在 Q/K 上加 RMSNorm，但二者的**显存量级可忽略且当前未建为
    op/param**。为避免「设了却被静默忽略、产貌似合理实则偏差的图」，这两个字段现由
    `build_llm.py:_check_implemented_dispatch` **fail-loud 守卫**：置 True 会抛
    NotImplementedError（而非默默出一个略偏的数）。故本预设把它们归零（内存中性），
    仅在此文档标注 Qwen 的真实架构；如日后要忠实建模，请在 attn builder 补 bias/q-k norm
    op 后再放开。param ref 可选：本预设主要验证"能建能评估"。
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
