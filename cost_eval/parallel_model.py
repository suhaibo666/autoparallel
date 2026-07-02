"""M3：并行度数 + mesh 关系 + stage→层 分配（镜像 parallel_dims.py）。"""
from __future__ import annotations
from .specs import ParallelConfig


class ParallelModel:
    def __init__(self, pc: ParallelConfig, n_layers: int, world_size: int):
        self.pc = pc
        self.n_layers = n_layers
        self.world_size = world_size
        region = pc.dp_shard * pc.cp * pc.tp
        if region % pc.ep != 0:
            raise ValueError(
                f"ep({pc.ep}) 必须整除 dp_shard*cp*tp={region}（专家在该区内分片）")
        self._efsdp = region // pc.ep
        # 均匀切层要求每 stage 至少一层；pp>n_layers 会令 per=0 → ZeroDivisionError（Task 8）。
        if pc.layers_per_stage is None and pc.pp > n_layers:
            raise ValueError(
                f"pp({pc.pp}) > n_layers({n_layers})：每 stage 至少需一层，无法均匀切层"
                "（或显式给 layers_per_stage）")

    def degree(self, axis: str) -> int:
        if axis == "sp":
            return self.pc.tp if self.pc.sequence_parallel else 1
        return {
            "tp": self.pc.tp, "cp": self.pc.cp, "ep": self.pc.ep,
            "dp_shard": self.pc.dp_shard, "dp_replicate": self.pc.dp_replicate,
            "pp": self.pc.pp,
        }[axis]

    def fsdp_degree(self) -> int:
        return self.pc.dp_shard * self.pc.cp

    def efsdp_degree(self) -> int:
        return self._efsdp

    def stage_of(self, layer_id: int) -> int:
        return self._layer_to_stage()[layer_id]

    def stage_layers(self, stage: int) -> list:
        return [l for l, s in enumerate(self._layer_to_stage()) if s == stage]

    def _layer_to_stage(self) -> list:
        pp = self.pc.pp
        if self.pc.layers_per_stage:
            mapping = []
            for stage, count in enumerate(self.pc.layers_per_stage):
                mapping += [stage] * count
            return mapping
        per = self.n_layers // pp
        return [min(l // per, pp - 1) for l in range(self.n_layers)]
