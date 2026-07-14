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
        """层 id → stage 映射。**首选** `layers_per_stage`（显式，已在 __init__ 校验和==n_layers）。

        默认均匀切（2026-07-11 修正,对齐 mindformers 语义）:**只均匀切中间层**（transformer/mtp,
        floor-division、remainder 归末 stage）;**embedding（层 0）固定归 stage0、head（末层）固定
        归末 stage**——伪层不占中间层配额。旧实现把 n_layers（含 2 伪层）整体均匀切,非整除时
        伪层挤占 transformer 名额（如 N=8 pp=4 → transformer 实分 1,2,2,3 而非 2,2,2,2）。
        pp 整除 N 时（如 8L pp2 → emb+L1-4 | L5-8+head）与旧实现逐层一致 → pp2 锚点不动。"""
        pp = self.pc.pp
        if self.pc.layers_per_stage:
            mapping = []
            for stage, count in enumerate(self.pc.layers_per_stage):
                mapping += [stage] * count
            return mapping
        if pp <= 1:
            return [0] * self.n_layers
        mid = self.n_layers - 2                      # 中间层数（transformer + mtp）
        # 标准均匀切（2026-07-14 修:此前 remainder 全堆末 stage,N=8 pp=3 切成 2,2,4 不均匀）:
        # 前 rem 个 stage 各多 1 层 → N=8 pp=3 = 3,3,2。整除时逐层不变（锚点安全）。
        base, rem = divmod(mid, pp)
        mid_map = []
        for s in range(pp):
            mid_map += [s] * (base + (1 if s < rem else 0))
        return [0] + mid_map + [pp - 1]              # embedding→stage0,head→末 stage
