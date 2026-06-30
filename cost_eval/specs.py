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
    state_bytes_per_param: int = 16          # bf16 param2+grad2+master4+m4+v4

    @classmethod
    def adamw(cls, fp32_grad: bool = False) -> "OptimizerSpec":
        return cls("AdamW", 18 if fp32_grad else 16)


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
