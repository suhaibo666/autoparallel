"""统一 LLM ModelSpec 构建器：`LLMConfig`（内存结构子集）+ `to_dimtable`。

`LLMConfig` = Megatron/mindformers `TransformerConfig` 的「内存结构」子集
（设计 `specs/2026-07-01-unified-llm-modelspec-design.md` §4）：只收**改变 op 图**
（哪些 op/张量存在、其 shape/saves/params）的字段。并行/recompute/swap/dtype/optimizer
由 `ParallelConfig`/`RecomputeSpec`/`SwapSpec`/`OptimizerSpec`/`HardwareSpec` 各自承担。

`to_dimtable(cfg)` 把 `LLMConfig` 映射为现有 `DimTable`（具体架构超参），供下游
`build_llm_spec` / `ShapeEval` / `StaticMem` 使用。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .model_spec import DimTable, NORM_KIND_CASTING, NORM_KIND_NONCASTING


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
    # GQA colossal CP 下 KV all-gather 到 full-S 的额外 buffer（modeling opt-in，P1-13/Y1）。默认
    # False → shape_eval 三重门（method==colossal & cp>1 & 本字段）恒不触发、workspace 逐字节不变
    # （消费点 shape_eval.py:265）。此前仅靠测试给 DimTable 动态挂属性启用、经公开 LLMConfig 不可达
    # （F8 订正 2026-07-16）——现为真字段并经 to_dimtable 直通到 DimTable.cp_kv_allgather_buffer。
    cp_kv_allgather_buffer: bool = False

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
    # **无重算-MoE OOM-安全标定 margin 因子**（D1，2026-07-16；非物理，2 点标定）：与 kept_frag 同族碎片
    #   （dispatch/permute/grouped-GEMM fp32-cast + <100MiB 长尾），但作用域是**无重算-MoE**而非 select-kept。
    #   仅 pp==1 单 stage 无重算 loss-BWD 生效（pp>1 走 `K_CE_PP` 分支，不进本 margin；⚠ 2026-07-30
    #   `K_CE` 重标定 8→7 后「已由 K_CE 平衡」这条旧理由不再成立，gate 未动、如实记）。factor=0.6 由 8L-none+cp2-none 两锚点
    #   联合标定使二者 OOM-安全（预测≥真机）；两点理想 factor 0.53/0.45 差 ~15%，故明示为标定常数、可单值调/关。
    #   默认 0=关（回归安全）。fused-CE（DSv4）不触发。见 mem_timeline nr_moe_frag_factor / kept_frag 桶。
    nr_moe_frag_factor: float = 0.0
    chunk_loss_num: int = 0                 # >1：分块 CE，降 loss 区峰值
    # embedding/head 权重的驻留/gather 副本 dtype（P1-04 接线，2026-07-14）：真机 FSDP 下
    # compute 副本为 bf16=2（全部锚点在此口径验证；fp32 master 在 persistent 的 opt 倍数里另计）。
    # 旧默认 4（"fp32"）从未生效（builder 未传，恒 resolve 成 2）——属 dead config + 文档漂移，
    # 现改默认 2 = 已验证行为、字段真实接线；fp32 直存场景显式置 4。
    embedding_params_dtype_bytes: int = 2
    # unfused CE 链 lean 口径（2026-07-23，116 std MHA/GQA 锚点定标）：True = 无重算 loss stage 的
    # CE fat 取 `K_CE_LEAN=4`（116 std pp1 与 pp2-s1 峰值差分反解「~3.3-4 份」,与 pp 无关；
    # **无逐块台账**,2026-07-30 重标定刻意未动它——见 mem_timeline 顶部 `K_CE_LEAN` 注释）；
    # False = 按 pp 分档的 `K_CE_PP / K_CE_PP1`（2026-07-30 由 8/4 重标定为 **7/3**,
    # 依据是 9 份仓内 profiler CSV 逐块清点,`docs/k_ce_recalibration_2026-07-30.md`）。
    # 仅 cross_entropy_fused=False 且无重算 loss stage 有差异。
    ce_pynative_lean: bool = False

    # ---- ③ 残差变体（横切）----
    residual_variant: str = "plain"         # plain | mhc
    num_residual_streams: int = 1           # mhc：hidden ×n
    # 融合 mHC（yaml `use_fused_mhc`，站点 dsv4h_*_pp4_recomp.yaml:109 = true）：走
    #   `npu_mhc_pre_sinkhorn`/`npu_mhc_post`（hyper_connection.py:368,413-419）。默认 False =
    #   非融合小算子路径（该路径**确实**物化 [S,B,n·H] fp32 归一化流）→ 现有 spec 惰性。
    use_fused_mhc: bool = False
    # sinkhorn 迭代次数（yaml `hc_sinkhorn_iters`；megatron 名 `mhc_sinkhorn_iterations`，
    #   transformer_config.py:2084 默认 20）。仅 use_fused_mhc=True 时进内存（sum_out/norm_out ∝ 它）。
    mhc_sinkhorn_iterations: int = 20

    # ---- MTP ----
    mtp_num_layers: int = 0                 # DeepSeek-V3/V4 多 token 预测头

    # ---- bias ----
    add_bias_linear: bool = False
    add_qkv_bias: bool = False

    compute_dtype_bytes: int = 2            # bf16


# 声明类型为 `tuple` 的序列字段：JSON 没有 tuple，解码时必须还原，否则 round-trip 得到 `list`
# 与原 `tuple` **字段不相等**（`(0,4,128) != [0,4,128]`）→ 保真判据会误报/漏报。
_TUPLE_FIELDS = ("window_pattern", "csa_compress_ratios", "moe_layer_freq")


def to_jsonable(cfg: LLMConfig) -> dict:
    """`LLMConfig` → 纯 JSON 可序列化 dict（**全字段**，无省略）。

    用途（`serve_explorer` 的 yaml 导入闭环，2026-07-25）：把 yaml 解析出的**权威**
    `LLMConfig` 原样带过「UI query dict」这道扁平表示，使页面评估不再从预设基座重建结构
    （重建会静默沿用预设值 —— 逐层 `csa_compress_ratios` / `o_groups` / `chunk_loss_num` 等
    在 UI 里无字段可承载，现场实测造成 35% 峰值分歧）。

    **全字段无省略**是刻意的：任何"只带一部分"的编码都会在新增字段时静默退化成预设值。
    """
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


def from_jsonable(d: dict) -> LLMConfig:
    """`to_jsonable` 的逆：JSON dict → `LLMConfig`。**字段集必须逐字对齐**，否则 fail-loud。

    - 多键（未知字段）→ 报错：编码方与本类定义漂移，静默丢弃会评估另一份模型；
    - 缺键 → 报错：缺的字段会**静默取 dataclass 默认值**，正是本次修复要消灭的静默替换；
    - `_TUPLE_FIELDS` 的 list → tuple 还原（JSON 无 tuple，见上方注释）。
    """
    names = [f.name for f in dataclasses.fields(LLMConfig)]
    got = set(d)
    unknown, missing = sorted(got - set(names)), sorted(set(names) - got)
    if unknown or missing:
        raise ValueError(
            f"LLMConfig JSON 字段集不匹配（未知 {unknown} / 缺失 {missing}）——缺字段会静默取"
            "dataclass 默认值（= 静默替换结构），未知字段说明编码方与 LLMConfig 定义漂移；"
            "两者都拒绝，请用同版本的 `to_jsonable` 重新编码。")
    kw = dict(d)
    for k in _TUPLE_FIELDS:
        if isinstance(kw.get(k), list):
            kw[k] = tuple(kw[k])
    return LLMConfig(**kw)


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
        # GQA colossal CP KV all-gather full-S buffer opt-in 直通（F8 订正 2026-07-16）：默认 False
        # → shape_eval 门恒不触发、workspace 逐字节不变。
        cp_kv_allgather_buffer=cfg.cp_kv_allgather_buffer,
        # ③ 残差变体（mHC）：hidden ×n 的符号维（设计 §9）；plain 时 =1 惰性。
        num_residual_streams=cfg.num_residual_streams,
        # 融合 mHC 门 + sinkhorn 迭代数（residual.build_hyper_connection_ops 读）。
        use_fused_mhc=cfg.use_fused_mhc,
        mhc_sinkhorn_iterations=cfg.mhc_sinkhorn_iterations,
        gated_linear_unit=cfg.gated_linear_unit,   # D-6：ungated MLP（fc1 不 2×）
        cross_entropy_fused=cfg.cross_entropy_fused,   # ①：fused CE（DSv4）lean / unfused fat
        ce_pynative_lean=cfg.ce_pynative_lean,         # unfused CE lean K=4（116 std 实测口径）
        norm_compute_dtype_bytes=cfg.layernorm_compute_dtype_bytes,   # norm 激活 fp32（真机 layernorm_compute_dtype）
        # norm 种类（2026-07-29）：`get_norm_cls`（layer_norm.py:187-191）按 `normalization` 返回
        # FusedRMSNorm（:151-155 输入直通，**不** cast → 不抬 fp32）或 FusedLayerNorm（:93-101 真 cast
        # → 抬）。抬升本身仍由 `norm_compute_dtype_bytes` 控幅度，本字段只决定它对哪类 norm 成立。
        norm_kind=(NORM_KIND_NONCASTING if cfg.normalization == "RMSNorm" else NORM_KIND_CASTING),
        kept_frag_factor=cfg.kept_frag_factor,   # B 标定 margin（select-kept-MoE loss 峰碎片长尾）
        nr_moe_frag_factor=cfg.nr_moe_frag_factor,   # D1 标定 margin（无重算-MoE loss 峰碎片长尾，pp==1）
        dtype_bytes=cfg.compute_dtype_bytes,
    )
