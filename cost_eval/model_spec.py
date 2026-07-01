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
    # Shared expert intermediate size (MoE + MLA combined layers)
    moe_shared_F: int = 0
    capacity_factor: float = 1.0
    # mHC residual streams (设计 §9)：hidden 打包为 n 条残差流 [S,B,n*H]。
    # 默认 1 → 完全惰性（plain 残差，n*H==H），不影响任何现有 spec。
    num_residual_streams: int = 1
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
