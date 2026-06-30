"""M6：事件驱动内存时间线仿真（1F1B 调度 + 桶式峰值追踪）。"""
from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Task 11: Event + build_1f1b
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    kind: str      # "FWD" | "BWD"
    mb: int
    layer: int = -1


def build_1f1b(stage: int, pp: int, m: int):
    """返回 (FWD/BWD, microbatch) 事件序列（层粒度在 simulate 内展开）。

    warmup=min(pp-1-stage, m) 个前向先行，然后 1F1B 交替，最后 cooldown BWD。
    """
    warmup = min(pp - 1 - stage, m)
    evs = [Event("FWD", i) for i in range(warmup)]
    fwd_i, bwd_i = warmup, 0
    while bwd_i < m:
        if fwd_i < m:
            evs.append(Event("FWD", fwd_i))
            fwd_i += 1
        evs.append(Event("BWD", bwd_i))
        bwd_i += 1
    return evs
