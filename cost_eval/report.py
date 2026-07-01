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
from .framework import framework_reserve


@dataclass(frozen=True)
class PeakMemoryReport:
    """各 PP stage 峰值显存报告。"""
    per_stage: list        # list[StagePeak]，按 stage 升序
    tightest_stage: int    # peak_bytes 最大的 stage
    oom: bool              # 任意 stage OOM


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
        persistent = StaticMem().compute(g, self.opt, pm, self.pc.cpu_offload)
        # framework_reserve 按配置分解：HCCL(200MB×组数) + hw.framework_reserve(残余标定项)
        fr = framework_reserve(self.pc, self.hw.framework_reserve)
        peaks = MemTimeline().simulate(
            g, self.recompute, self.swap, pm, persistent,
            fr, self.hw.max_device_memory,
            grad_dtype_bytes=getattr(self.opt, "grad_dtype_bytes", 4),
            record_timeline=record_timeline)
        per_stage = [peaks[s] for s in sorted(peaks)]
        tightest = max(per_stage, key=lambda p: p.peak_bytes).stage
        return PeakMemoryReport(per_stage, tightest, any(p.oom for p in per_stage))
