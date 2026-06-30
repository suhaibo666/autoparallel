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
    framework_reserve: int = 0               # O_framework 标定常数


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
