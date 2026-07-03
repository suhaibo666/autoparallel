"""统一 LLM ModelSpec 构建器：`LLMConfig`（内存结构子集）+ `to_dimtable`。

`LLMConfig` = Megatron/mindformers `TransformerConfig` 的「内存结构」子集
（设计 `specs/2026-07-01-unified-llm-modelspec-design.md` §4）：只收**改变 op 图**
（哪些 op/张量存在、其 shape/saves/params）的字段。并行/recompute/swap/dtype/optimizer
由 `ParallelConfig`/`RecomputeSpec`/`SwapSpec`/`OptimizerSpec`/`HardwareSpec` 各自承担。

`to_dimtable(cfg)` 把 `LLMConfig` 映射为现有 `DimTable`（具体架构超参），供下游
`build_llm_spec` / `ShapeEval` / `StaticMem` 使用。
"""
from __future__ import annotations

from dataclasses import dataclass

from .model_spec import DimTable


@dataclass(frozen=True)
class LLMConfig:
    """LLM 结构配置（内存结构子集，设计 §4）。frozen：一处构造、全程只读。"""

    # ---- 核心维度 ----
    num_layers: int
    hidden_size: int                        # H
    num_attention_heads: int                # n_heads
    vocab_size: int
    seq_length: int                         # S
    batch_size: int = 1                     # B（local）
    head_dim: int | None = None             # 默认 H // n_heads

    # ---- ① 注意力 ----
    attn_type: str = "gqa"                  # mha | gqa | mla | dsv4_hybrid
    num_query_groups: int | None = None     # GQA 的 KV 组数（None→=n_heads）
    window_size: int | None = None          # None=全注意力；int=滑窗宽度（SWA）
    window_pattern: tuple | None = None      # 每层 0=全/1=SWA；None=全层同 window_size
    # MLA / dsv4 专用
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0               # = qk_pos_emb_head_dim
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    # dsv4_hybrid（DSA 索引器 + 压缩器 + 稀疏注意力）专用
    csa_compress_ratios: tuple | None = None    # 每层 ∈ {0/1, 4=CSA, 128=HCA}
    csa_window_size: int = 128
    dsa_indexer_n_heads: int = 0
    dsa_indexer_head_dim: int = 0
    dsa_indexer_topk: int = 0
    o_groups: int = 0                       # 分组输出投影
    o_lora_rank: int = 0
    dsa_fused: bool = True                   # 融合 DSA kernel（生产默认）：稀疏中间量不物化（§真机）

    # ---- ② FFN / MoE ----
    ffn_hidden_size: int | None = None      # 默认 4*H
    gated_linear_unit: bool = True          # SwiGLU（vs ungated）
    num_moe_experts: int | None = None      # None→纯 dense
    moe_router_topk: int = 0
    moe_ffn_hidden_size: int | None = None  # 专家 FFN 隐藏维
    moe_shared_expert_num: int = 0
    moe_shared_ffn_hidden_size: int = 0
    first_k_dense_replace: int | None = None    # 前 K 层 dense；或用下面的 freq
    moe_layer_freq: tuple | int | None = None   # 每层 0=dense/1=MoE（覆盖 first_k）
    moe_capacity_factor: float = 1.0        # 影响 dispatched token 数（内存相关）

    # ---- 归一化 / 位置编码（结构相关部分）----
    normalization: str = "RMSNorm"          # RMSNorm | LayerNorm
    norm_placement: str = "pre"             # pre | post | sandwich
    qk_layernorm: bool = False              # Q/K 上加 norm
    position_embedding_type: str = "rope"   # rope | learned_absolute | none

    # ---- 装配：embedding / head / loss ----
    tie_word_embeddings: bool = False       # tie→无独立 lm_head 权重
    loss_type: str = "logsoftmax_nll"       # logsoftmax_nll | chunked | vocab_parallel_ce
    chunk_loss_num: int = 0                 # >1：分块 CE，降 loss 区峰值
    embedding_params_dtype_bytes: int = 4   # embedding/输出 fp32

    # ---- ③ 残差变体（横切）----
    residual_variant: str = "plain"         # plain | mhc
    num_residual_streams: int = 1           # mhc：hidden ×n

    # ---- MTP ----
    mtp_num_layers: int = 0                 # DeepSeek-V3/V4 多 token 预测头

    # ---- bias ----
    add_bias_linear: bool = False
    add_qkv_bias: bool = False

    compute_dtype_bytes: int = 2            # bf16


def to_dimtable(cfg: LLMConfig) -> DimTable:
    """把 `LLMConfig` 映射为现有 `DimTable`（具体架构超参）。

    映射（设计 §5 步骤 1）：
      hidden_size→H, num_attention_heads→n_heads, (num_query_groups or n_heads)→n_kv,
      seq_length→S, batch_size→B, vocab_size→vocab, (ffn_hidden_size or 4*H)→F,
      num_moe_experts→n_experts, moe_router_topk→topk, moe_shared_expert_num→n_shared,
      moe_ffn_hidden_size→moe_F, moe_shared_ffn_hidden_size→moe_shared_F,
      moe_capacity_factor→capacity_factor, compute_dtype_bytes→dtype_bytes；
      MLA dims（q_lora_rank/kv_lora_rank/qk_rope_head_dim/qk_nope_head_dim/v_head_dim）直通。

    - `head_dim` 默认 `hidden_size // num_attention_heads`（当 cfg.head_dim is None）。
    - `n_layers = num_layers + 2`（含 embedding + head），与
      `validate_dsv3.build_dsv3_spec` 的 `n_layers=N+2` 一致（= len(layer_pattern)）。
    """
    head_dim = cfg.head_dim if cfg.head_dim is not None else cfg.hidden_size // cfg.num_attention_heads
    n_kv = cfg.num_query_groups if cfg.num_query_groups is not None else cfg.num_attention_heads
    F = cfg.ffn_hidden_size if cfg.ffn_hidden_size is not None else 4 * cfg.hidden_size
    n_experts = cfg.num_moe_experts if cfg.num_moe_experts is not None else 0
    moe_F = cfg.moe_ffn_hidden_size if cfg.moe_ffn_hidden_size is not None else 0

    return DimTable(
        H=cfg.hidden_size,
        F=F,
        n_heads=cfg.num_attention_heads,
        n_kv=n_kv,
        head_dim=head_dim,
        S=cfg.seq_length,
        B=cfg.batch_size,
        vocab=cfg.vocab_size,
        n_layers=cfg.num_layers + 2,        # + embedding + head（= len(layer_pattern)）
        n_experts=n_experts,
        topk=cfg.moe_router_topk,
        n_shared=cfg.moe_shared_expert_num,
        moe_F=moe_F,
        # MLA dims 直通
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        v_head_dim=cfg.v_head_dim,
        # dsv4_hybrid dims 直通（DSA 索引器 / 分组输出 / 滑窗）
        dsa_indexer_n_heads=cfg.dsa_indexer_n_heads,
        dsa_indexer_head_dim=cfg.dsa_indexer_head_dim,
        dsa_indexer_topk=cfg.dsa_indexer_topk,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora_rank,
        csa_window_size=cfg.csa_window_size,
        dsa_fused=cfg.dsa_fused,
        moe_shared_F=cfg.moe_shared_ffn_hidden_size,
        capacity_factor=cfg.moe_capacity_factor,
        # ③ 残差变体（mHC）：hidden ×n 的符号维（设计 §9）；plain 时 =1 惰性。
        num_residual_streams=cfg.num_residual_streams,
        dtype_bytes=cfg.compute_dtype_bytes,
    )
