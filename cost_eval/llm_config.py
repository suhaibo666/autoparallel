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
    attn_type: str = "gqa"                  # mha | gqa | mla | dsa | dsv4_hybrid
    num_query_groups: int | None = None     # GQA 的 KV 组数（None→=n_heads）
    window_size: int | None = None          # None=全注意力；int=滑窗宽度（SWA）
    window_pattern: tuple | None = None      # 每层 0=全/1=SWA；None=全层同 window_size
    # MLA / dsv4 专用
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0               # = qk_pos_emb_head_dim
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    # dsv4_hybrid（DSA 索引器 + 压缩器 + 稀疏注意力）专用；dsa_indexer_* 三维同时被
    # attn_type="dsa"（DSv3.2/GLM-5 lightning indexer，layers/dsa.py 预估计）使用（须 >0）。
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
    moe_shared_expert_gating: bool = False   # C3：shared-expert [H,1] 门（默认关，DSv3 不用）
    first_k_dense_replace: int | None = None    # 前 K 层 dense；或用下面的 freq
    moe_layer_freq: tuple | int | None = None   # 每层 0=dense/1=MoE（覆盖 first_k）
    moe_capacity_factor: float = 1.0        # 影响 dispatched token 数（内存相关）
    # MoE dispatched-token 口径（P1-12）：balanced（默认，均值）/ capacity（ceil 最忙口径）/
    #   skew（均值 × moe_skew_factor）。均值适合吞吐、capacity/skew 适合 OOM 边界（审计 P1-12）。
    #   默认 balanced + 1.0 → 惰性，不影响现有 to_dimtable 路径（DSv3/DSv4 锚点不动）。
    moe_dispatch_mode: str = "balanced"     # balanced | capacity | skew
    moe_skew_factor: float = 1.0            # skew 口径的倾斜因子（percentile 倾斜，≥1）

    # ---- 归一化 / 位置编码（结构相关部分）----
    normalization: str = "RMSNorm"          # RMSNorm | LayerNorm
    layernorm_compute_dtype_bytes: int = 4  # norm 计算 dtype（真机 layernorm_compute_dtype，一般 fp32=4）→ norm 激活 fp32
    norm_placement: str = "pre"             # pre | post | sandwich
    qk_layernorm: bool = False              # Q/K 上加 norm
    position_embedding_type: str = "rope"   # rope | learned_absolute | none

    # ---- 装配：embedding / head / loss ----
    tie_word_embeddings: bool = False       # tie→无独立 lm_head 权重
    loss_type: str = "logsoftmax_nll"       # logsoftmax_nll | chunked | vocab_parallel_ce
    cross_entropy_fused: bool = False        # ①：融合 CE kernel（DSv4=True lean）/ unfused pynative（False，无重算下 fat）
    # **标定 margin 因子**（B 方案，2026-07-09；非 op 图导出，明示为标定常数）：select 重算下**保留-MoE**层
    #   在 loss 峰的 fp32-cast 横切 + 小张量长尾（占当前 kept-MoE 激活的比例）。源码级 op-DAG 提取证实此
    #   残差**在 op 图粒度之下**（profiler live-set 313 个 <100MiB 碎片，`analysis/realmachine/opdag_validation.md`）
    #   → 无法从显式 op 导出，退标定。默认 0=关（回归安全）。仅 select-kept-MoE 生效（mem_timeline kept_frag）。
    kept_frag_factor: float = 0.0
    chunk_loss_num: int = 0                 # >1：分块 CE，降 loss 区峰值
    # embedding/head 权重的驻留/gather 副本 dtype（P1-04 接线，2026-07-14）：真机 FSDP 下
    # compute 副本为 bf16=2（全部锚点在此口径验证；fp32 master 在 persistent 的 opt 倍数里另计）。
    # 旧默认 4（"fp32"）从未生效（builder 未传，恒 resolve 成 2）——属 dead config + 文档漂移，
    # 现改默认 2 = 已验证行为、字段真实接线；fp32 直存场景显式置 4。
    embedding_params_dtype_bytes: int = 2

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
        moe_shared_gate=cfg.moe_shared_expert_gating,
        capacity_factor=cfg.moe_capacity_factor,
        # MoE dispatched-token 多口径（P1-12）：默认 balanced/1.0 惰性直通。
        moe_dispatch_mode=cfg.moe_dispatch_mode,
        moe_skew_factor=cfg.moe_skew_factor,
        # Q/K layernorm 直通（2026-07-15 协调补充，供 attention.py 建 GQA q/k norm 读取）。
        qk_layernorm=cfg.qk_layernorm,
        # ③ 残差变体（mHC）：hidden ×n 的符号维（设计 §9）；plain 时 =1 惰性。
        num_residual_streams=cfg.num_residual_streams,
        gated_linear_unit=cfg.gated_linear_unit,   # D-6：ungated MLP（fc1 不 2×）
        cross_entropy_fused=cfg.cross_entropy_fused,   # ①：fused CE（DSv4）lean / unfused fat
        norm_compute_dtype_bytes=cfg.layernorm_compute_dtype_bytes,   # norm 激活 fp32（真机 layernorm_compute_dtype）
        kept_frag_factor=cfg.kept_frag_factor,   # B 标定 margin（select-kept-MoE loss 峰碎片长尾）
        dtype_bytes=cfg.compute_dtype_bytes,
    )
