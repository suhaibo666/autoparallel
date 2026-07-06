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
        # stage→层分配（D-8，忠实 mindformers 流水线阶段配置）：**首选显式** `layers_per_stage`
        # ——每 stage 层数列表（含 embedding + head 两个伪层，和须 == n_layers），对应 mindformers
        # pipeline 的 `offset`/`num_layer_list`（用户显式配、不自行推测）。缺省(None)才退化为均匀切
        # （remainder 归末 stage，仅作便利；pp>1 非整除时建议显式配 —— 见 D-8 审计）。
        if pc.layers_per_stage is not None:
            if len(pc.layers_per_stage) != pc.pp:
                raise ValueError(
                    f"layers_per_stage 长度({len(pc.layers_per_stage)}) 必须 == pp({pc.pp})")
            if sum(pc.layers_per_stage) != n_layers:
                raise ValueError(
                    f"layers_per_stage 之和({sum(pc.layers_per_stage)}) 必须 == n_layers({n_layers})"
                    "（n_layers 含 embedding + head 两个伪层）")
            if any(c <= 0 for c in pc.layers_per_stage):
                raise ValueError(
                    f"layers_per_stage 每项须 >0（每 stage 至少一层）：{pc.layers_per_stage}")
        elif pc.pp > n_layers:
            # 均匀切层要求每 stage 至少一层；pp>n_layers 会令 per=0 → ZeroDivisionError（Task 8）。
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
        """层 id → stage 映射。**首选** `layers_per_stage`（显式，已在 __init__ 校验和==n_layers）；
        否则退化为均匀切（floor-division，remainder 归末 stage）。"""
        pp = self.pc.pp
        if self.pc.layers_per_stage:
            mapping = []
            for stage, count in enumerate(self.pc.layers_per_stage):
                mapping += [stage] * count
            return mapping
        per = self.n_layers // pp
        return [min(l // per, pp - 1) for l in range(self.n_layers)]
