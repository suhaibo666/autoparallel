"""M1：声明式 op 图数据结构 + 内存契约。"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OpType(Enum):
    MATMUL = "matmul"
    FLASH_ATTN = "flash_attn"
    ELEMENTWISE = "elementwise"
    NORM = "norm"
    ROPE = "rope"
    MOE_ROUTER = "moe_router"
    MOE_GEMM = "moe_gemm"
    DISPATCH = "dispatch"
    COMBINE = "combine"


@dataclass
class DimTable:
    """架构超参（具体值）。"""
    H: int; F: int; n_heads: int; n_kv: int; head_dim: int
    S: int; B: int; vocab: int; n_layers: int
    n_experts: int = 0; topk: int = 0; n_shared: int = 0; moe_F: int = 0
    # MLA (Multi-Latent Attention) dimensions
    q_lora_rank: int = 0; kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0; qk_nope_head_dim: int = 0; v_head_dim: int = 0
    # DSv4 hybrid (DSA indexer + compressor + CSA/HCA sparse attention) dimensions.
    # Defaults 0 → inert for every existing spec (no op图/shape 变化).
    dsa_indexer_n_heads: int = 0; dsa_indexer_head_dim: int = 0; dsa_indexer_topk: int = 0
    o_groups: int = 0; o_lora_rank: int = 0; csa_window_size: int = 0
    # 融合 DSA kernel（生产默认 True）：稀疏中间量 kv_gathered/attn_weights 走 kernel scratch、
    # 不物化成张量（真机 15415 已证）；unfused 小算子路径才逐个物化（False）。仅 dsv4 用。
    dsa_fused: bool = True
    # Q/K layernorm（GQA q/k norm，2026-07-15 协调补充）：True 时 attention 段在 Q/K 上加 norm
    # （builder 侧 attention.py 用 `getattr(d, "qk_layernorm")` 读取）。默认 False（DSv3/DSv4
    # qk_layernorm=False）→ 惰性，锚点不动。此前仅 LLMConfig 有此字段、DimTable 缺 → 直通补齐。
    qk_layernorm: bool = False
    # GQA colossal CP KV all-gather full-S buffer opt-in（P1-13/Y1；F8 订正 2026-07-16 提为真字段）。
    # shape_eval.py:265 靠 `getattr(spec.dims, "cp_kv_allgather_buffer", False)` 读取；默认 False →
    # 三重门恒不触发、workspace 逐字节不变（12 golden 锚点走 mla/dsv4 不经 GQA builder，惰性）。
    # 由 llm_config.to_dimtable 从 LLMConfig 同名字段直通（此前仅测试动态挂属性、经公开 API 不可达）。
    cp_kv_allgather_buffer: bool = False
    # Shared expert intermediate size (MoE + MLA combined layers)
    moe_shared_F: int = 0
    # shared-expert 门控（closure-audit C3，2026-07-15）：use_shared_expert_gating=True 时有
    # [H,1] Dense 门（shared_experts.py:56-64）。字节可忽略但为参数守恒完整性建其权重。默认 False
    # （DSv3 不用 → golden 不变、惰性）。
    moe_shared_gate: bool = False
    capacity_factor: float = 1.0
    # MoE dispatched-token 口径（P1-12，2026-07-15）：控制 disp/e_* 张量 dim0（每卡 token 数）与
    # all-to-all staging 的**符号表达式**。均值（balanced）适合吞吐估计，但真实 MoE 受路由倾斜/
    # capacity ceil/padding/最忙 rank 影响，均值不足以做 OOM 安全边界 → 提供 3 口径（见 ffn.py
    #   `_moe_dispatch_token_expr`）：
    #   - "balanced"（默认）：S·B·topk·C —— 逐字节等旧 TLOCAL（理想均分；DSv3/DSv4 锚点口径）。
    #   - "capacity"：ceil(S·B·topk·C/n_experts)·n_experts —— capacity 上取整的**最忙口径**。
    #   - "skew"：S·B·topk·C·moe_skew_factor —— 均值 × 倾斜因子（percentile 倾斜）。
    # 默认 balanced + moe_skew_factor=1.0 → 完全惰性，现有 spec 逐字节不变。
    moe_dispatch_mode: str = "balanced"
    moe_skew_factor: float = 1.0
    # mHC residual streams (设计 §9)：hidden 打包为 n 条残差流 [S,B,n*H]。
    # 默认 1 → 完全惰性（plain 残差，n*H==H），不影响任何现有 spec。
    num_residual_streams: int = 1
    # gated_linear_unit（SwiGLU）：True→fc1 输出 2·F（gate+up）；False→ungated MLP，fc1 输出 F
    # （plain gelu/relu，D-6）。默认 True（现有全部 spec 走 SwiGLU，故 DSv3/preset 不变）。
    gated_linear_unit: bool = True
    # layernorm/RMSNorm 计算 dtype 的字节数（真机 `layernorm_compute_dtype`，一般 fp32=4）：norm 在
    # fp32 下算,会**保留输入的 fp32 cast 副本**供反向（真机 profiler 的 Cast 大头）。默认 4=fp32
    # （主流配置）。=2 时退回 bf16（compute dtype）。仅影响 norm op 的 saved 激活字节（fp32 = 2× bf16）。
    norm_compute_dtype_bytes: int = 4
    # 交叉熵是否融合 kernel（①，真机 profiler）：False=unfused pynative log_softmax+NLL op 链
    #   （无重算下 ~8 满 vocab fp32 中间量共存，fat）；True=fused kernel（精简 ~3，如 DSv4 生产）。
    #   默认 False（pynative 常态）。仅在**无重算 + unfused** 下 loss bwd_scratch fat（mem_timeline ①）。
    cross_entropy_fused: bool = False
    # **标定 margin 因子**（B 方案，2026-07-09；非 op 图导出）：保留(非重算)模块在 loss 峰的
    #   fp32-cast 横切 + 小张量长尾占「当前 kept 激活」的比例。源码级 op-DAG 提取证实此残差**在 op 图
    #   粒度之下**（profiler live-set 313 个 <100MiB 碎片，`analysis/realmachine/opdag_validation.md`），
    #   非 op 图可导出 → 明示为标定常数。默认 0=不加（回归安全，full/cp-full 锚点不破）。仅 loss-BWD 事件、
    #   对 **select-kept-MoE** 层生效（mem_timeline kept_frag 桶）。
    kept_frag_factor: float = 0.0
    # **无重算-MoE OOM-安全标定 margin 因子**（D1，2026-07-16；同族碎片、独立作用域）：仅 pp==1 单 stage
    #   无重算 loss-BWD 生效（pp>1 由 K_CE 平衡）。2 点标定（8L-none+cp2-none），明示为标定常数、非物理。
    #   默认 0=关。fused-CE 不触发。见 mem_timeline nr_moe_frag_factor。
    nr_moe_frag_factor: float = 0.0
    dtype_bytes: int = 2

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TensorRef:
    """符号 shape + 切分标注。shard: {dim_index -> axis('tp'|'cp'|'ep'|'sp')}。"""
    name: str
    shape: tuple
    shard: dict = field(default_factory=dict)
    is_weight: bool = False
    partial: Optional[str] = None      # 该张量在此轴上是未规约部分和
    dtype_bytes: Optional[int] = None  # 覆盖 DimTable.dtype_bytes（如 fp32 loss 张量=4）
    # ── context-parallel（cp）切分标注（D-1 修正 2026-07-07；P2-08 对齐 2026-07-14）──────────
    # cp_shard=False：该激活为**全序列 full-S**，cp 下**不** ÷cp。注意：loss/head 区**不再**是
    #   此例——早期"cp=2 满 vocab full-S 实测"已被 Bug A 复核推翻（B=2·S/cp 误读为 B=1·full-S），
    #   loss/head 区现随 cp ÷cp（权威口径 head.py:59-64），全库当前**无任何张量**设 cp_shard=False，
    #   字段保留供未来真 full-S 语义使用。默认 True = 随序列切分（S/cp）。
    # cp_kv=True：该张量为 attention **KV 侧**激活；`colossal`（ulysses_degree=1）下 KV all-gather
    #   到 **full-S**（不 ÷cp），其余 cp 算法（ulysses/ring/hybrid）仍随 body ÷cp。默认 False。
    cp_shard: bool = True
    cp_kv: bool = False

    def has_ep(self) -> bool:
        return "ep" in self.shard.values()


@dataclass
class OpSpec:
    """算子 + 内存契约。params/saves 都是 TensorRef 列表。"""
    name: str
    type: OpType
    inputs: list
    output: TensorRef
    params: list = field(default_factory=list)   # is_weight 张量 → param/grad/opt
    saves: list = field(default_factory=list)     # save_for_backward → 激活
    workspace: Optional[str] = None               # fwd 期 kernel scratch（符号字节）
    bwd_scratch: Optional[str] = None             # bwd 期临时物化（符号字节），如 loss probs(fp32)
    # TensorRef 型 workspace / bwd_scratch（P0-05/P0-04，2026-07-14）：字符串表达式只 ÷cp、
    # 无法表达 tp/ep 切分（shape_eval:224-229）；须按并行域缩放的瞬态（如 DSA indexer KL 的
    # [B,n_heads,S,S] fp32 head 维 ÷tp、vocab-parallel CE 的 [N,V/tp] fp32 梯度）走 TensorRef
    # ——经 resolve_tensor 的 shard/cp 机制求 local 字节后并入对应 *_bytes。与字符串表达式可叠加。
    workspace_ref: Optional[TensorRef] = None
    bwd_scratch_ref: Optional[TensorRef] = None
    attrs: dict = field(default_factory=dict)


@dataclass
class LayerSpec:
    ops: list


@dataclass
class ModelSpec:
    name: str
    dims: DimTable
    layer_pattern: list           # list[str]
    layer_specs: dict             # str -> LayerSpec

    def get_layer(self, layer_type: str) -> LayerSpec:
        return self.layer_specs[layer_type]
