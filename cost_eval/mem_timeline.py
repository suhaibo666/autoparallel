"""M6：事件驱动内存时间线仿真（1F1B 调度 + 桶式峰值追踪）。"""
from __future__ import annotations
from dataclasses import dataclass, field

from .structure_mem import estimate_structure_memory


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
    persistent: int = 0       # param + optimizer state（持久态，剔 grad §8.4）
    act_live: int = 0         # 当前存活的 saved activations
    gather_buf: int = 0       # FSDP all-gather 缓冲（当前层整层权重 + 预取下 depth 层双缓冲、reshard 后释）
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
class TimelineSample:
    """内存时间线上一个采样点（一次 rec() = 一个事件）。"""
    idx: int              # 事件序号（0 起）
    event: str            # 事件标签（如 fwd:3 / fwd_end / bwd@5）
    total_bytes: int      # 该事件时刻总占用（Σ桶 + framework_reserve）
    breakdown: MemBreakdown


@dataclass(frozen=True)
class StagePeak:
    """单 stage 仿真结果。"""
    stage: int
    peak_bytes: int
    breakdown: MemBreakdown
    peak_event: str
    oom: bool
    timeline: tuple = ()   # record_timeline=True 时为 tuple[TimelineSample]（全事件序列），否则空


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

# 以下 `_layer_*` 均**组装** `estimate_structure_memory`（structure_mem.py）取对应桶，
# 不再各自裸 walk raw ops——去重（saves/params 按名）与 rollup 逻辑集中在那一处。

def _layer_saves_bytes(layer) -> int:
    """该层 saves 去重后总字节（全量保存时 pin 进 act_live）。

    去重语义（设计 §2.1/§7.1「Σ去重」）见 `estimate_structure_memory`：`attn` 被
    `flash`（saves=[qkv,attn,lse]）与 `o_proj`（saves=[attn]）双 save（attention.py GQA
    :69/:73、MLA :149/:153）只算一次（C1）。"""
    return estimate_structure_memory(layer.ops).activation_saves


def _checkpoint_input_bytes(layer) -> int:
    """full 重算时仅保留层输入：第一个有 saves 的 op 的首个 save。"""
    return estimate_structure_memory(layer.ops).checkpoint_input


def _layer_param_bytes(layer) -> int:
    """该层 full-unsharded 参数（compute dtype）——FSDP all-gather 缓冲（按名去重）。"""
    return estimate_structure_memory(layer.ops).param_full_bytes


def _layer_grad_bytes(layer, grad_dtype_bytes: int) -> int:
    """反向 reduce-scatter 前的 full-unsharded 梯度（按 grad dtype，按名去重）。"""
    return estimate_structure_memory(
        layer.ops, grad_dtype_bytes=grad_dtype_bytes).grad_full_bytes


def _layer_bwd_scratch(layer) -> int:
    """该层各 op 反向临时物化之和（如 loss probs fp32）。"""
    return estimate_structure_memory(layer.ops).bwd_scratch


def _layer_workspace(layer) -> int:
    """该层各 op workspace_bytes 的最大值（FWD 逐层临时占用）。"""
    return estimate_structure_memory(layer.ops).workspace


def _prefetch_param_bytes(order: list, idx: int, depth: int, sm_by_id: dict) -> int:
    """FSDP2 参数预取双缓冲：执行序 `order` 中位置 idx 之后 `depth` 个单元的
    full-unsharded 参数字节之和（compute dtype）——与当前层共存的预取缓冲。

    忠实 PyTorch FSDP2 默认 **depth-1**（`_fsdp_param_group.py:854-856`
    `_backward_prefetch` → `target = post_forward_order[curr_index - 1]`，恰回退 1
    个单元；`:854` 的 `elif curr_index > 0` → 反向最后一个单元不预取。前向隐式
    depth-1：`wait_for_unshard:486-495` + Note:61-70 当前 all-gather 输出保留至下一
    copy-in）。mindformers pynative 显式同深度（`parallelize.py:245-273`
    set_modules_to_{forward,backward}_prefetch，每层→相邻一层）。

    - depth=1：恰取执行序下一个单元的 param_full；边界（末单元 idx+1 越界）→ 求和为
      0（不预取）。
    - depth=0：范围空 → 0 → 复现旧单缓冲（回归路径）。
    - reshard_after_forward=default（非 root 用完即 reshard，`_fsdp_state.py:203-207`）
      → 与当前层共存的仅这 depth 个预取单元，**不累积**。
    """
    total = 0
    for j in range(idx + 1, min(idx + 1 + depth, len(order))):
        total += sm_by_id[order[j]].param_full_bytes
    return total


# ---------------------------------------------------------------------------
# MemTimeline
# ---------------------------------------------------------------------------

class MemTimeline:
    """事件驱动峰值仿真器（M6）。

    建模要点：
    - gather_buf 在 FWD/BWD 逐层 = 当前层 full-unsharded 权重 + **预取下 depth 层双缓冲**
      （FSDP2 参数预取：`_prefetch_param_bytes`，depth 来自 ParallelConfig.prefetch_depth，
      默认 1；depth=0 复现旧单缓冲），reshard 后即释（reshard_after_forward=default）。
    - swap 仅将被 swap 层的 saved 置 0（离开 act_live），swap_buf 暂置 0。
    """

    def simulate(self, g, recompute, swap, pm, static_persistent: dict,
                 framework_reserve: int, max_device_memory: int,
                 grad_dtype_bytes: int = 4, record_timeline: bool = False,
                 alloc_block_bytes: int = 1) -> dict:
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
        record_timeline    : bool — True 则每个事件都记进 StagePeak.timeline（内存曲线）

        返回
        ----
        dict[stage, StagePeak]
        """
        res: dict = {}
        pp = pm.degree("pp")
        m = pm.pc.num_microbatches
        # FSDP2 参数预取深度（config 驱动，非魔法常数）：默认 1 = FSDP2 默认双缓冲
        # （survey：PyTorch _fsdp_param_group.py:854-856 / mindformers parallelize.py:245-273）。
        # depth=0 → 复现旧单缓冲（回归路径）。
        depth = getattr(pm.pc, "prefetch_depth", 1)

        for stage, layers in g.stages.items():
            layer_ids = [l.layer_id for l in layers]
            by_id = {l.layer_id: l for l in layers}
            # 每层的 StructureMemory rollup（模块化组装，单点去重）——预算一次，事件循环直取各桶。
            sm_by_id = {
                l.layer_id: estimate_structure_memory(
                    l.ops, grad_dtype_bytes=grad_dtype_bytes,
                    alloc_block_bytes=alloc_block_bytes)
                for l in layers
            }

            B = Buckets(persistent=static_persistent.get(stage, 0))
            peak: int = -1
            peak_ev: str = ""
            peak_bd: MemBreakdown = None  # type: ignore[assignment]
            series: list = []

            def rec(tag: str) -> None:
                nonlocal peak, peak_ev, peak_bd
                t = B.total() + framework_reserve
                is_peak = t > peak
                bd = None
                if record_timeline or is_peak:
                    bd = MemBreakdown(
                        B.persistent, B.act_live, B.gather_buf, B.grad_buf,
                        B.recomp_scratch, B.bwd_scratch, B.swap_buf, B.workspace,
                        framework_reserve,
                    )
                if record_timeline:
                    series.append(TimelineSample(len(series), tag, t, bd))
                if is_peak:
                    peak = t
                    peak_ev = tag
                    peak_bd = bd

            # (mb, layer_id) -> saved bytes currently pinned in act_live
            pinned: dict = {}

            for ev in build_1f1b(stage, pp, m):
                if ev.kind == "FWD":
                    for idx, lid in enumerate(layer_ids):
                        sm = sm_by_id[lid]
                        # 1. FSDP all-gather 整层参数(compute dtype) + 预取下 depth 层双缓冲
                        #    （FSDP2 前向隐式 depth-1 overlap）+ workspace → 采样
                        B.gather_buf = sm.param_full_bytes + _prefetch_param_bytes(
                            layer_ids, idx, depth, sm_by_id)
                        B.workspace = sm.workspace
                        rec(f"fwd:{lid}")
                        B.workspace = 0
                        B.gather_buf = 0   # reshard_after_forward(default)：用完即释
                        # 2. 决定该层 pin 多少 activation
                        if recompute.is_full(lid):
                            saved = sm.checkpoint_input               # 仅保留层入口
                        elif swap.swaps(lid):
                            saved = 0                                 # 全部卸载到 CPU
                        else:
                            saved = sm.activation_saves               # 全量 saves（去重）
                        pinned[(ev.mb, lid)] = saved
                        B.act_live += saved
                    # 所有层 pin 完毕 → 该 microbatch FWD 峰
                    rec("fwd_end")

                else:  # BWD（逆序层）—— FSDP gather + full grad + recompute + bwd_scratch 共存
                    bwd_order = list(reversed(layer_ids))
                    for idx, lid in enumerate(bwd_order):
                        sm = sm_by_id[lid]
                        # 反向某层峰值 = 该层 FSDP 重新 gather 的整层参数(compute)
                        #   + 预取反向下 depth 层的双缓冲（FSDP2 默认 depth-1 反向预取，
                        #     逆前向序：_fsdp_param_group.py:854-856；output_layer 首个反向
                        #     单元预取末 transformer 层）
                        #   + reduce-scatter 前 full 梯度(grad dtype)
                        #   + (full 重算)重物化激活 + (op)反向临时物化(如 loss probs)
                        #   共存，叠在 persistent + 其余 act_live 之上
                        B.gather_buf = sm.param_full_bytes + _prefetch_param_bytes(
                            bwd_order, idx, depth, sm_by_id)
                        B.grad_buf = sm.grad_full_bytes
                        if recompute.is_full(lid):
                            # 去双算：checkpoint 输入已在 act_live（fwd 时 pin），不再计入重物化
                            # TODO(§8.5②): 严格应为该层 forward 的 max-live(mini-fwd 时间线)，非 saves 之和
                            B.recomp_scratch = max(
                                0, sm.activation_saves - sm.checkpoint_input)
                        B.bwd_scratch = sm.bwd_scratch
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
                timeline=tuple(series),
            )

        return res
