"""并行/优化器/硬件/重算/swap 配置。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParallelConfig:
    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    sequence_parallel: bool = False
    reshard_after_forward: str = "default"   # always|never|default
    cpu_offload: bool = False
    microbatch: int = 1
    interleave: int = 1
    layers_per_stage: Optional[list] = None  # None=均匀切
    prefetch_depth: int = 1
    num_microbatches: int = 1                # m = global_batch/(dp*microbatch)，由 adapter 算好


@dataclass
class OptimizerSpec:
    type: str = "AdamW"
    # 持久 = param + optimizer state（**剔除 grad**，真机修正 §8.4）
    state_bytes_per_param: int = 14          # bf16 params: bf16 2 + fp32 master4 + m4 + v4
    grad_dtype_bytes: int = 4                # 反向瞬态 grad(grad_buf) 的 dtype：fp32=4 / bf16=2

    @classmethod
    def adamw(cls, params_fp32: bool = False, grad_dtype_bytes: int = 4) -> "OptimizerSpec":
        # 持久(剔grad): fp32 params=master4+m4+v4=12; bf16 params=bf16 2+master4+m4+v4=14
        return cls("AdamW", 12 if params_fp32 else 14, grad_dtype_bytes)


@dataclass
class HardwareSpec:
    max_device_memory: int                   # bytes（来自 ContextConfig.max_device_memory）
    # framework_reserve：**审计/回归旋钮**（默认 0，非生产项）。生产路径下框架瞬态已按机理
    # 拆进 op 图（FSDP 预取→gather_buf、flash-ws→flash workspace、MoE staging→dispatch/combine
    # workspace，见 framework.py）；此项仅在显式给值时复现旧「经验兜底常数」以供审计（如 golden
    # 的 177 MiB）。**不是拟合的物理量**。
    framework_reserve: int = 0
    # alloc_block_bytes：**平台属性**（不是拟合值）——MindSpore 设备内存池 `DynamicMemPoolBestFit`
    # 对每次分配按 `kMemAlignSize`(=512B, `kDynamicMemAlignSize`) 对齐。分配峰值(max_memory_allocated)
    # 里每个张量按此块粒度上取整 → 少量对齐碎片。这是 framework_reserve「分配器块对齐」项的**公式化**
    # 落地（structure_mem 逐张量 roundup），取代经验常数。默认 512（可按硬件覆盖）。
    alloc_block_bytes: int = 512


@dataclass
class RecomputeSpec:
    mode: str = "None"                       # None|full
    full_layers: set = field(default_factory=set)

    def is_full(self, layer_id: int) -> bool:
        return self.mode == "full" and layer_id in self.full_layers


@dataclass
class SwapSpec:
    enable: bool = False
    default_prefetch: int = 1
    swap_layers: set = field(default_factory=set)

    def swaps(self, layer_id: int) -> bool:
        return self.enable and layer_id in self.swap_layers
