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
    # Shared expert intermediate size (MoE + MLA combined layers)
    moe_shared_F: int = 0
    capacity_factor: float = 1.0
    # mHC residual streams (设计 §9)：hidden 打包为 n 条残差流 [S,B,n*H]。
    # 默认 1 → 完全惰性（plain 残差，n*H==H），不影响任何现有 spec。
    num_residual_streams: int = 1
    # gated_linear_unit（SwiGLU）：True→fc1 输出 2·F（gate+up）；False→ungated MLP，fc1 输出 F
    # （plain gelu/relu，D-6）。默认 True（现有全部 spec 走 SwiGLU，故 DSv3/preset 不变）。
    gated_linear_unit: bool = True
    # 交叉熵是否融合 kernel（①，真机 profiler）：False=unfused pynative log_softmax+NLL op 链
    #   （无重算下 ~8 满 vocab fp32 中间量共存，fat）；True=fused kernel（精简 ~3，如 DSv4 生产）。
    #   默认 False（pynative 常态）。仅在**无重算 + unfused** 下 loss bwd_scratch fat（mem_timeline ①）。
    cross_entropy_fused: bool = False
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
    # ── context-parallel（cp）切分标注（D-1 修正 2026-07-07，真机确认）───────────────────
    # cp_shard=False：该激活为**全序列 full-S**，cp 下**不** ÷cp（loss/head 区——head 前 hidden
    #   all-gather 回 full-S，对**所有** cp 算法一致；真机 cp=2 峰实测满 vocab logsm/probs/grad
    #   各 2020 MiB full-S）。默认 True = 随序列切分（emb_out + 所有 decoder 层激活 = S/cp）。
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
