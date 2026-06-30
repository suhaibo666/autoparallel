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


# ---------------------------------------------------------------------------
# Task 12: Buckets / MemBreakdown / StagePeak + helpers + MemTimeline
# ---------------------------------------------------------------------------

@dataclass
class Buckets:
    """7 桶内存状态（每桶均为字节数，瞬时值）。"""
    persistent: int = 0       # param + grad + optimizer state（持久态）
    act_live: int = 0         # 当前存活的 saved activations
    gather_buf: int = 0       # FSDP all-gather 缓冲（P0 简化：暂置 0）
    grad_buf: int = 0         # 参数梯度缓冲（BWD 一层的瞬时峰值）
    recomp_scratch: int = 0   # full 重算时临时重建的 saves
    bwd_scratch: int = 0      # 反向临时物化（如 loss probs fp32）
    swap_buf: int = 0         # swap prefetch 缓冲（P0 简化：暂置 0）
    workspace: int = 0        # 算子 workspace（FWD 逐层临时）

    def total(self) -> int:
        return (self.persistent + self.act_live + self.gather_buf + self.grad_buf
                + self.recomp_scratch + self.bwd_scratch + self.swap_buf + self.workspace)


@dataclass(frozen=True)
class MemBreakdown:
    """峰值时刻各桶的快照。"""
    persistent: int
    act_live: int
    gather_buf: int
    grad_buf: int
    recomp_scratch: int
    bwd_scratch: int
    swap_buf: int
    workspace: int
    framework: int


@dataclass(frozen=True)
class StagePeak:
    """单 stage 仿真结果。"""
    stage: int
    peak_bytes: int
    breakdown: MemBreakdown
    peak_event: str
    oom: bool


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _layer_saves_bytes(layer) -> int:
    """该层所有 op 的 saves 张量总字节（全量保存时 pin 进 act_live）。"""
    return sum(s.local_numel * s.dtype_bytes for op in layer.ops for s in op.saves)


def _checkpoint_input_bytes(layer) -> int:
    """full 重算时仅保留层输入：取第一个有 saves 的 op 的首个 save。"""
    for op in layer.ops:
        if op.saves:
            s = op.saves[0]
            return s.local_numel * s.dtype_bytes
    return 0


def _layer_param_bytes(layer) -> int:
    """该层 full-unsharded 参数（compute dtype）——FSDP all-gather 缓冲；
    权重 local_numel 已含 tp/ep 切但未含 fsdp，故即 full-unsharded。"""
    return sum(w.local_numel * w.dtype_bytes for op in layer.ops for w in op.params)


def _layer_grad_bytes(layer, grad_dtype_bytes: int) -> int:
    """反向 reduce-scatter 前的 full-unsharded 梯度（按 grad dtype）。"""
    return sum(w.local_numel * grad_dtype_bytes for op in layer.ops for w in op.params)


def _layer_bwd_scratch(layer) -> int:
    """该层各 op 反向临时物化之和（如 loss probs fp32）。"""
    return sum(getattr(op, "bwd_scratch_bytes", 0) for op in layer.ops)


def _layer_workspace(layer) -> int:
    """该层各 op workspace_bytes 的最大值（FWD 逐层临时占用）。"""
    return max((op.workspace_bytes for op in layer.ops), default=0)


# ---------------------------------------------------------------------------
# MemTimeline
# ---------------------------------------------------------------------------

class MemTimeline:
    """事件驱动峰值仿真器（M6）。

    P0 简化：
    - gather_buf 暂置 0（FSDP 双缓冲预取在后续增量补）。
    - swap 仅将被 swap 层的 saved 置 0（离开 act_live），swap_buf 暂置 0。
    """

    def simulate(self, g, recompute, swap, pm, static_persistent: dict,
                 framework_reserve: int, max_device_memory: int,
                 grad_dtype_bytes: int = 4) -> dict:
        """仿真各 stage 峰值。

        参数
        ----
        g                  : ResolvedGraph
        recompute          : RecomputeSpec
        swap               : SwapSpec
        pm                 : ParallelModel
        static_persistent  : dict[stage, int]  — M5 StaticMem.compute() 输出
        framework_reserve  : int  — 框架常驻开销（字节）
        max_device_memory  : int  — 设备容量上限（字节）

        返回
        ----
        dict[stage, StagePeak]
        """
        res: dict = {}
        pp = pm.degree("pp")
        m = pm.pc.num_microbatches

        for stage, layers in g.stages.items():
            layer_ids = [l.layer_id for l in layers]
            by_id = {l.layer_id: l for l in layers}

            B = Buckets(persistent=static_persistent.get(stage, 0))
            peak: int = -1
            peak_ev: str = ""
            peak_bd: MemBreakdown = None  # type: ignore[assignment]

            def rec(tag: str) -> None:
                nonlocal peak, peak_ev, peak_bd
                t = B.total() + framework_reserve
                if t > peak:
                    peak = t
                    peak_ev = tag
                    peak_bd = MemBreakdown(
                        B.persistent, B.act_live, B.gather_buf, B.grad_buf,
                        B.recomp_scratch, B.bwd_scratch, B.swap_buf, B.workspace,
                        framework_reserve,
                    )

            # (mb, layer_id) -> saved bytes currently pinned in act_live
            pinned: dict = {}

            for ev in build_1f1b(stage, pp, m):
                if ev.kind == "FWD":
                    for lid in layer_ids:
                        layer = by_id[lid]
                        # 1. FSDP all-gather 整层参数(compute dtype) + workspace → 采样
                        B.gather_buf = _layer_param_bytes(layer)
                        B.workspace = _layer_workspace(layer)
                        rec(f"fwd:{lid}")
                        B.workspace = 0
                        B.gather_buf = 0   # reshard_after_forward(default)：用完即释
                        # 2. 决定该层 pin 多少 activation
                        if recompute.is_full(lid):
                            saved = _checkpoint_input_bytes(layer)   # 仅保留层入口
                        elif swap.swaps(lid):
                            saved = 0                                 # 全部卸载到 CPU
                        else:
                            saved = _layer_saves_bytes(layer)        # 全量 saves
                        pinned[(ev.mb, lid)] = saved
                        B.act_live += saved
                    # 所有层 pin 完毕 → 该 microbatch FWD 峰
                    rec("fwd_end")

                else:  # BWD（逆序层）—— FSDP gather + full grad + recompute + bwd_scratch 共存
                    for lid in reversed(layer_ids):
                        layer = by_id[lid]
                        # 反向某层峰值 = 该层 FSDP 重新 gather 的整层参数(compute)
                        #   + reduce-scatter 前 full 梯度(grad dtype)
                        #   + (full 重算)重物化激活 + (op)反向临时物化(如 loss probs)
                        #   共存，叠在 persistent + 其余 act_live 之上
                        B.gather_buf = _layer_param_bytes(layer)
                        B.grad_buf = _layer_grad_bytes(layer, grad_dtype_bytes)
                        if recompute.is_full(lid):
                            # 去双算：checkpoint 输入已在 act_live（fwd 时 pin），不再计入重物化
                            # TODO(§8.5②): 严格应为该层 forward 的 max-live(mini-fwd 时间线)，非 saves 之和
                            B.recomp_scratch = max(
                                0, _layer_saves_bytes(layer) - _checkpoint_input_bytes(layer))
                        B.bwd_scratch = _layer_bwd_scratch(layer)
                        rec(f"bwd@{lid}")
                        B.gather_buf = B.grad_buf = B.recomp_scratch = B.bwd_scratch = 0
                        # 该层反向结束，释放其 pinned 激活
                        B.act_live -= pinned.pop((ev.mb, lid))

            res[stage] = StagePeak(
                stage=stage,
                peak_bytes=peak,
                breakdown=peak_bd,
                peak_event=peak_ev,
                oom=(peak > max_device_memory),
            )

        return res
