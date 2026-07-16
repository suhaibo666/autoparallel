"""M3：并行度数 + mesh 关系 + stage→层 分配（镜像 parallel_dims.py）。"""
from __future__ import annotations
from .advisories import warn_modeling_approx
from .specs import ParallelConfig


class ParallelModel:
    def __init__(self, pc: ParallelConfig, n_layers: int, world_size: int,
                 edge_pseudo: tuple = (1, 1)):
        self.pc = pc
        self.n_layers = n_layers
        self.world_size = world_size
        # (首伪层数, 末伪层数):生产 pattern 恒 [embedding]+decoder*+[lm_head] → 默认 (1,1),
        # 伪层不占均匀切/round-robin 配额。无伪层的裸 ModelSpec(toy/单测)显式传 (0,0)。
        self.edge_pseudo = edge_pseudo
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

    def dense_fsdp_degree(self) -> int:
        """dense（非专家）权重的**有效 FSDP 分母**（Z3，grouped-FSDP 子域）。

        忠实 mindformers `pynative/distributed/parallel_dims.py:443-470`（`get_fsdp_shard_mesh`）：
        `dense_fsdp_shard_size` 是 dense 权重分片的**子域**大小（须整除完整 fsdp=dp_shard·cp）。
        配了子域（>0）→ 用该子域（<完整 fsdp）→ 每卡 dense 持久 = dense_global/子域（÷更小域 → 更大）；
        未配（0/None）→ 完整 `fsdp_degree()`（缺省逐字节不变）。**专家权重不受影响**，仍用
        `efsdp_degree()`；`fsdp_degree()` 本身（gather 全域口径）也不变。"""
        s = getattr(self.pc, "dense_fsdp_shard_size", 0)
        return s if s else self.fsdp_degree()

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
        hp, tp_ = self.edge_pseudo
        mid = self.n_layers - hp - tp_               # 中间层数（transformer + mtp）
        v = max(1, getattr(self.pc, "interleave", 1))
        if v > 1:
            # VPP round-robin 放置（2026-07-14 修,源:mindformers pynative
            # `pynative/distributed/pipeline_parallel.py:258` `chunk_id*pp_size+pp_rank`）:
            # 中间层均匀切成 pp·v 个**虚拟 stage**（余数前置）,虚拟段 s_virt 归物理 rank
            # `s_virt % pp`——rank r 持虚拟 stage {r, r+pp, ...} 的**非连续**层段。层结构
            # 不均匀（dense/moe 混布/逐层 compress_ratio）时与旧「连续块再切 chunk」近似不同。
            nvirt = pp * v
            base, rem = divmod(mid, nvirt)
            mid_map = []
            for sv in range(nvirt):
                mid_map += [sv % pp] * (base + (1 if sv < rem else 0))
            return [0] * hp + mid_map + [pp - 1] * tp_
        # 标准均匀切（2026-07-14 修:此前 remainder 全堆末 stage,N=8 pp=3 切成 2,2,4 不均匀）:
        # 前 rem 个 stage 各多 1 层 → N=8 pp=3 = 3,3,2。整除时逐字节不变（锚点安全）。
        base, rem = divmod(mid, pp)
        mid_map = []
        for s in range(pp):
            mid_map += [s] * (base + (1 if s < rem else 0))
        return [0] * hp + mid_map + [pp - 1] * tp_   # embedding→stage0,head→末 stage

    def stage_chunks(self, stage: int) -> list:
        """该物理 stage 的 **per-chunk 层组**（VPP,v 个 chunk;v<=1 → 单 chunk=全部层）。

        round-robin 语义（`pipeline_parallel.py:258`）:chunk c = 虚拟 stage `c*pp+stage` 的层。
        embedding 伪层附加到 stage0 的 chunk0、head 伪层附加到末 stage 的末 chunk（与
        1F1B 事件序一致:embedding 最先前向、loss 最后反向）。显式 layers_per_stage 下退化为
        「stage 内连续均衡切 v 段」（mindformers 显式 per-chunk ranges 暂不支持,文档化近似）。"""
        v = max(1, getattr(self.pc, "interleave", 1))
        lids = self.stage_layers(stage)
        if v <= 1:
            return [lids]
        pp = self.pc.pp
        hp, tp_ = self.edge_pseudo
        pseudo_first = [l for l in lids if l < hp]                            # embedding
        pseudo_last = [l for l in lids if l >= self.n_layers - tp_]           # head
        mids = [l for l in lids if hp <= l < self.n_layers - tp_]
        if self.pc.layers_per_stage:
            # 显式物理配额:stage 内连续均衡切 v 段（近似,见 docstring）。
            # round3 A(N9):此路径是**静默文档化近似**——mindformers 显式 per-chunk ranges
            #   (layers_per_stage + interleave 组合)本库暂用连续均衡切代替,层不均匀时 chunk 归属
            #   可能与真实 round-robin 放置有偏 → 补一条 ModelingApproxWarning(不改数值,只提示)。
            warn_modeling_approx(
                f"layers_per_stage={self.pc.layers_per_stage} 与 interleave={v}>1 同时给定:"
                "本库对该组合用「stage 内连续均衡切 v 段」近似(mindformers 显式 per-chunk ranges "
                "暂不支持)——层不均匀时 chunk 归属可能与真实放置有偏,VPP 激活峰估计为近似值。")
            base, rem = divmod(len(mids), v)
            chunks, i = [], 0
            for c in range(v):
                n = base + (1 if c < rem else 0)
                chunks.append(mids[i:i + n]); i += n
        else:
            # round-robin:按虚拟 stage 归组（_layer_to_stage 的 v>1 分支已按此放置,
            # 此处按层 id 段重建各 chunk——虚拟段连续、chunk 间非连续）。
            mid_total = self.n_layers - hp - tp_
            nvirt = pp * v
            base, rem = divmod(mid_total, nvirt)
            bounds, acc = [], hp                      # 中间层 id 从 hp 开始
            for sv in range(nvirt):
                n = base + (1 if sv < rem else 0)
                bounds.append((sv, acc, acc + n)); acc += n
            chunks = []
            for c in range(v):
                sv = c * pp + stage
                lo, hi = next((a, b) for s, a, b in bounds if s == sv)
                chunks.append([l for l in mids if lo <= l < hi])
        if pseudo_first:
            chunks[0] = pseudo_first + chunks[0]
        if pseudo_last:
            chunks[-1] = chunks[-1] + pseudo_last
        return chunks
