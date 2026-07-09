"""M7：组装峰值显存报告 + Evaluator 门面。

Evaluator 是整个评估器的对外入口：接收 ModelSpec + 并行/优化器/硬件/重算/swap
配置，依次调用 M3→M4→M5→M6，返回 PeakMemoryReport。
"""
from __future__ import annotations
from dataclasses import dataclass

from .parallel_model import ParallelModel
from .shape_eval import ShapeEval
from .static_mem import StaticMem
from .mem_timeline import MemTimeline, StagePeak
from .framework import framework_reserve, hccl_reserved_buffer


@dataclass(frozen=True)
class PeakMemoryReport:
    """各 PP stage 峰值显存报告。

    `peak_bytes`/`oom` 是 **allocated 峰值**（max_memory_allocated，OOM 主判据，真机验证）。
    `hccl_reserved_bytes`（D-2）是 **reserved 池**的 HCCL 通信缓冲估计（按通信域数，`framework.
    hccl_reserved_buffer`）——**不进 allocated 峰值**（ep=2 真机证实），但计入 `reserved 估计`：
    `reserved ≈ allocated_peak + hccl_reserved (+ 池碎片)`。设备 HBM 的真实约束是 reserved，
    故给出 `reserved_estimate_bytes(stage)` 供 reserved 口径的 OOM 余量核查。
    """
    per_stage: list        # list[StagePeak]，按 stage 升序
    tightest_stage: int    # peak_bytes 最大的 stage
    oom: bool              # 任意 stage OOM（allocated 口径）
    hccl_reserved_bytes: int = 0   # D-2：HCCL 通信缓冲（reserved 池，按通信域数；world-level 同值）

    def reserved_estimate_bytes(self, stage: int) -> int:
        """该 stage 的 reserved 池估计 = allocated 峰值 + HCCL 通信缓冲（reserved 口径上界）。"""
        return self.per_stage[stage].peak_bytes + self.hccl_reserved_bytes


class Evaluator:
    """离线并行策略代价评估器门面（P0：内存）。"""

    def __init__(self, model_spec, parallel_config, optimizer, hardware,
                 recompute, swap):
        self.spec = model_spec
        self.pc = parallel_config
        self.opt = optimizer
        self.hw = hardware
        self.recompute = recompute
        self.swap = swap

    def evaluate(self, record_timeline: bool = False) -> PeakMemoryReport:
        """执行全链路评估，返回 PeakMemoryReport。

        record_timeline=True 时，每个 StagePeak.timeline 记录全事件内存序列（内存曲线）。
        """
        world = (self.pc.dp_replicate * self.pc.dp_shard * self.pc.cp
                 * self.pc.tp * self.pc.pp)
        pm = ParallelModel(self.pc, self.spec.dims.n_layers, world)
        g = ShapeEval().resolve(self.spec, pm)
        # 分配器块对齐（平台属性 HardwareSpec.alloc_block_bytes，默认 512）：逐张量 roundup —
        # 「分配器碎片」项的公式化落地（framework_reserve「块对齐取整」分量，取代经验常数）。
        block = getattr(self.hw, "alloc_block_bytes", 1)
        persistent = StaticMem().compute(g, self.opt, pm, self.pc.cpu_offload, alloc_block_bytes=block)
        # framework_reserve 现默认 0（生产）：框架瞬态已按机理拆进 op 图（FSDP 预取→gather_buf、
        # flash-ws→flash workspace、MoE staging→dispatch/combine workspace）+ 分配器对齐→上面的
        # 逐张量 roundup。hw.framework_reserve 仅审计/回归旋钮（显式给值复现旧经验常数，如 golden 177）。
        fr = framework_reserve(self.pc, self.hw.framework_reserve)
        peaks = MemTimeline().simulate(
            g, self.recompute, self.swap, pm, persistent,
            fr, self.hw.max_device_memory,
            grad_dtype_bytes=getattr(self.opt, "grad_dtype_bytes", 4),
            record_timeline=record_timeline, alloc_block_bytes=block,
            cross_entropy_fused=getattr(self.spec.dims, "cross_entropy_fused", False),
            norm_compute_dtype_bytes=getattr(self.spec.dims, "norm_compute_dtype_bytes", 0),
            kept_frag_factor=getattr(self.spec.dims, "kept_frag_factor", 0.0))
        per_stage = [peaks[s] for s in sorted(peaks)]
        tightest = max(per_stage, key=lambda p: p.peak_bytes).stage
        # D-2：HCCL 通信缓冲（reserved 池，按启用的通信域数估计；不进 allocated 峰值）→ 接入报告。
        hccl = hccl_reserved_buffer(self.pc)
        return PeakMemoryReport(per_stage, tightest,
                                any(p.oom for p in per_stage), hccl_reserved_bytes=hccl)
