"""M6：事件驱动内存时间线仿真（1F1B 调度 + 桶式峰值追踪）。"""
from __future__ import annotations
from dataclasses import dataclass, field

from .structure_mem import estimate_structure_memory, estimate_select_memory


# ---------------------------------------------------------------------------
# Task 11: Event + build_1f1b（+ Task[VPP]: 交错式 1F1B / 虚拟流水 warmup）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    kind: str      # "FWD" | "BWD"
    mb: int
    layer: int = -1


def _1f1b_from_warmup(warmup: int, m: int):
    """给定 warmup（先行前向数）构造 1F1B 事件序列：warmup 个 FWD 先行，然后
    steady 段 F/B 交替直到 m 个前向发完，最后 cooldown 段把剩余 BWD 收尾。

    plain 与 interleaved 仅 **warmup 深度** 不同 → 共用此段（DRY，且保证 V=1
    与旧 build_1f1b 逐事件一致）。"""
    evs = [Event("FWD", i) for i in range(warmup)]
    fwd_i, bwd_i = warmup, 0
    while bwd_i < m:
        if fwd_i < m:
            evs.append(Event("FWD", fwd_i))
            fwd_i += 1
        evs.append(Event("BWD", bwd_i))
        bwd_i += 1
    return evs


def build_1f1b(stage: int, pp: int, m: int):
    """plain 1F1B（forward_backward_pipelining_without_interleaving）事件序列。

    warmup = min(pp-1-stage, m) 个前向先行（Megatron `schedules.py:870`
    `num_warmup = pp - rank - 1`，clamp 到 total=m，`:889-890`），然后 1F1B
    交替，最后 cooldown BWD。层粒度在 simulate 内展开。"""
    return _1f1b_from_warmup(min(pp - 1 - stage, m), m)


def interleaved_warmup(stage: int, pp: int, m: int, v: int) -> int:
    """交错式 1F1B（VPP / 虚拟流水）某 stage 的 warmup（先行前向）微批数。

    锚定 **Megatron** `forward_backward_pipelining_with_interleaving` 的
    `num_warmup_microbatches`（`schedules.py:877-878`）：

        num_warmup = (pp - rank - 1) * 2 + (num_model_chunks - 1)
                     * microbatch_group_size_per_vp_stage

    默认 `microbatch_group_size_per_vp_stage = pipeline_model_parallel_size = pp`
    （深度优先调度，`model_parallel_config.py:519-520`），故本库取

        warmup = (pp - stage - 1) * 2 + (v - 1) * pp

    其中 v = num_model_chunks（虚拟流水级数 / interleave 数），stage = PP rank。
    再 clamp 到本模型的可用微批数 m（Megatron 在虚拟粒度 clamp 到 m*v，
    `schedules.py:862/889-890`；本库事件为**物理微批粒度**——一个 FWD 事件 pin 整
    个物理 stage 的全部层，见 build_interleaved_1f1b 文档——故 clamp 到 m）。

    **V=1 不走此式**：交错式在 v=1 处给出 2*(pp-1-stage)，是 plain 的 2 倍；Megatron
    在 v=1 时根本走 `virtual_pipeline_parallel_size is None` 的 plain 分支
    （`schedules.py:868-870`），mindformers pynative 亦要求 v>1 才 interleaved
    （`pipeline_parallel.py:275-276`/`:360-371`）。故 v<=1 由 build_interleaved_1f1b
    直接委托 plain build_1f1b。此函数按定义对 v<=1 返回 plain warmup（min(pp-1-stage, m)）。
    """
    if v <= 1:
        return min(pp - 1 - stage, m)
    raw = (pp - stage - 1) * 2 + (v - 1) * pp
    return min(raw, m)


def build_interleaved_1f1b(stage: int, pp: int, m: int, v: int):
    """交错式 1F1B（interleaved-1F1B / VPP）事件序列 —— **更深的 warmup**。

    与 plain build_1f1b 唯一区别是 warmup 深度（interleaved_warmup）：V 个虚拟模型
    块交错各自的微批 → 首个反向前有更多在飞微批 → warmup 更深 → simulate 的 FWD 循环
    里 `pinned` 同时驻留更多微批激活 → 每 stage 激活峰更高（本次建模的内存效应）。

    **V=1 → 逐事件复现 build_1f1b**（byte-identical；本库现有配置全部 v=1，anchors/
    golden 不动）。

    ── 层→(stage,chunk) 与内存幅度的忠实性说明（faithful-but-minimal，见 CLAUDE.md）──
    真实 VPP 下每个物理 stage 拥有 **V 个非连续 chunk**（round-robin：物理 rank 拥有虚拟
    stage {chunk*pp + rank}，Megatron 及 mindformers `pipeline_parallel.py:257-258`；层按
    虚拟 stage 均分，`:188-195`），每个在飞虚拟微批只 pin **一个 chunk（L/V 层）**，净激活
    膨胀比 ≈ 1 + (pp-1)/(pp*V)（在 V=2 处最大，随 V 增大回落）。本模型走**最小侵入**路径：
    沿用既有「一个 FWD 事件 pin 整个物理 stage 的全部层」的 pinned 循环，仅把 warmup 加深到
    Megatron 交错式公式 → 内存效应表现为「pinned 同时驻留 warmup+1 个 **整 stage** 微批激活」。
    这是对真实 VPP 峰值的**保守上界**（每微批按整 stage 而非 L/V 计，偏高约 V 倍），换取零侵入
    与 V=1 逐字节回归；把 per-chunk（×1/V）细化列为文档化后续项。warmup 深度本身严格取自 Megatron
    源码，V 越大 warmup 越深、峰值单调不减，方向与真实一致。
    """
    if v <= 1:
        return build_1f1b(stage, pp, m)
    return _1f1b_from_warmup(interleaved_warmup(stage, pp, m, v), m)


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
    recomp_scratch: int = 0   # full 重算层反向重跑 forward 的 max-live（§8.5②，扣 checkpoint 输入）
    bwd_scratch: int = 0      # 反向临时物化（如 loss probs fp32 / mHC sinkhorn grad）
    bwd_working_set: int = 0  # 无重算层反向工作集（激活梯度 dL/dact，= forward_max_live − bwd_scratch，§8.5②）
    swap_buf: int = 0         # 激活 swap H2D 预取缓冲（BWD：复原当前卸载层 saves + 反向序后 depth 层在飞预取窗，双缓冲）
    workspace: int = 0        # 算子 workspace（FWD 逐层临时）

    def total(self) -> int:
        return (self.persistent + self.act_live + self.gather_buf + self.grad_buf
                + self.recomp_scratch + self.bwd_scratch + self.bwd_working_set
                + self.swap_buf + self.workspace)


@dataclass(frozen=True)
class MemBreakdown:
    """峰值时刻各桶的快照。"""
    persistent: int
    act_live: int
    gather_buf: int
    grad_buf: int
    recomp_scratch: int
    bwd_scratch: int
    bwd_working_set: int
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


def _prefetch_swap_bytes(order: list, idx: int, depth: int, offloaded: set, sm_by_id: dict) -> int:
    """激活 swap 反向 H2D 预取窗（双缓冲）：反向执行序 `order` 中位置 idx 之后 `depth` 个单元里
    **被卸载的层**（`offloaded` 集）的 activation_saves 之和——反向到当前层时，其后 depth 层的
    激活正被预取回 HBM（在飞/已驻留），与当前层激活共存。

    忠实 mindformers pynative `apply_swap`：预取对 `(layer_id, layer_id+prefetch)`
    （`activation_checkpoint.py:807/:812`）经 `SwapManager().set_forward_prefetch_layer`
    （`:814-815`）——处理"更后一层"(反向更早、id 更大)时触发"更前一层"的 H2D swap-in；反向到某层
    时，其 swap-in 已在 `prefetch` 步前触发 → 驻留。故反向序里位置 idx 之后 `depth` 个单元 = 在飞
    预取窗。depth 来自 `SwapSpec.default_prefetch`（`config.py:826` 默认 1，校验 ≥1；本库额外允许
    depth=0 = 仅复原当前层、无预取窗）。

    - `offloaded` = 实际卸载的层集（`swap.swaps(lid) and not recompute.is_full(lid)`，与 FWD
      `saved=0` 条件一致）；非卸载层激活走 act_live/recomp_scratch，不进此窗（三态互斥，去双算）。
    - 边界（idx+1.. 越界）→ 求和为 0（反向末端无更多可预取）。
    - swap 关（`offloaded` 空）→ 求和 0（回归路径，逐字节复现旧行为）。
    """
    total = 0
    for j in range(idx + 1, min(idx + 1 + depth, len(order))):
        lid = order[j]
        if lid in offloaded:
            total += sm_by_id[lid].activation_saves
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
    - 激活 swap（`SwapSpec`，忠实 mindformers pynative `apply_swap` / PyTorch `save_on_cpu`）：
      FWD 被 swap 层 saves 卸载到 CPU（D2H）→ `saved=0`，前向→反向间隙不驻留 act_live；
      BWD 反向前 H2D 取回 → `swap_buf` = 复原当前层 saves +（反向序后 depth 层的）在飞预取窗
      （`_prefetch_swap_bytes`，depth 来自 `SwapSpec.default_prefetch`，与 FSDP 参数预取同构的
      双缓冲）。swap 关（enable=False）→ swap_buf 恒 0，逐字节复现旧行为。
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
        # 虚拟流水（VPP）级数 V：来自 ParallelConfig.interleave（默认 1 = plain 1F1B，
        # build_interleaved_1f1b 对 v<=1 逐事件复现 build_1f1b → anchors/golden 不动）。
        # V>1 → 交错式更深 warmup（Megatron `schedules.py:877-878`），更多在飞微批激活。
        v = getattr(pm.pc, "interleave", 1)
        # FSDP2 参数预取深度（config 驱动，非魔法常数）：默认 1 = FSDP2 默认双缓冲
        # （survey：PyTorch _fsdp_param_group.py:854-856 / mindformers parallelize.py:245-273）。
        # depth=0 → 复现旧单缓冲（回归路径）。
        depth = getattr(pm.pc, "prefetch_depth", 1)
        # 激活 swap 反向 H2D 预取深度（config 驱动，非魔法常数）：来自 SwapSpec.default_prefetch
        # （survey：mindformers config.py:826 默认 1 = 双缓冲；activation_checkpoint.py:807/:814）。
        # swap 关时 swaps() 恒 False → 下方 swap_buf 恒 0（与预取深度无关）。
        swap_depth = getattr(swap, "default_prefetch", 1)

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
            # 选择性重算：每层按选择器（op 名/类型子串）把 op 划分为选中/非选中，预算三桶
            # （act_live_pinned / recomp_scratch / bwd_working_set，复用 forward_max_live 机理，
            # 单点去重）。仅对 is_select 的层预算；full/None 层走既有路径（字节级不变）。
            select_mem_by_id = {
                l.layer_id: estimate_select_memory(
                    l.ops,
                    (lambda op, _lid=l.layer_id: recompute.op_matches(
                        _lid, op.name, getattr(op.type, "value", op.type))),
                    alloc_block_bytes=alloc_block_bytes)
                for l in layers if recompute.is_select(l.layer_id)
            }
            # 实际卸载到 CPU 的层集（三态互斥：recompute 优先于 swap，与 FWD `saved=0` 分支同条件）。
            # 这些层 saves 前向不驻留 act_live、反向经 swap_buf 复原（H2D 取回）。full 与 select
            # 均属「重算态」，不进卸载集。
            offloaded = {
                lid for lid in layer_ids
                if swap.swaps(lid)
                and not recompute.is_full(lid) and not recompute.is_select(lid)
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
                        B.recomp_scratch, B.bwd_scratch, B.bwd_working_set,
                        B.swap_buf, B.workspace,
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

            for ev in build_interleaved_1f1b(stage, pp, m, v):
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
                        elif recompute.is_select(lid):
                            # 选择性重算：非选中 op 的 saves（去重）+ 层入口边界常驻；
                            # 选中 op 的 saves 丢弃（反向重物化）。介于 full 与全量之间。
                            saved = select_mem_by_id[lid].act_live_pinned
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
                        #   + (full 重算)重物化激活 recomp_scratch / (无重算)反向工作集 bwd_working_set
                        #   + (op)反向临时物化 bwd_scratch(如 loss probs)
                        #   共存，叠在 persistent + 其余 act_live 之上
                        B.gather_buf = sm.param_full_bytes + _prefetch_param_bytes(
                            bwd_order, idx, depth, sm_by_id)
                        B.grad_buf = sm.grad_full_bytes
                        B.bwd_scratch = sm.bwd_scratch
                        # 激活 swap（§8.1 swap_buf="从 CPU 预取回的激活"）：被卸载层反向前 H2D 取回，
                        #   swap_buf = 复原当前层 saves（不在 act_live）+ 反向序后 swap_depth 层在飞预取窗
                        #   （双缓冲，`_prefetch_swap_bytes`）。当前层复原量 = 前向从 act_live 扣掉的同一
                        #   `activation_saves`（对称：卸载多少、取回多少），故被 swap 层反向激活仍驻留、不欠算。
                        #   非卸载层此项 0 → swap 关逐字节复现旧行为。
                        B.swap_buf = (
                            (sm.activation_saves if lid in offloaded else 0)
                            + _prefetch_swap_bytes(bwd_order, idx, swap_depth, offloaded, sm_by_id))
                        if recompute.is_full(lid):
                            # 重算层：反向重跑 forward，其重物化 = **该层 forward 的 max-live**
                            # （mini-fwd 时间线峰值，§8.5②①②）扣掉已 pin 进 act_live 的 checkpoint
                            # 输入。取代旧 `saves 之和 − checkpoint` 近似（saves 累加全层、非峰值；
                            # 多中间量层 max-live 常 < Σsaves，旧式高估）。
                            B.recomp_scratch = max(
                                0, sm.forward_max_live - sm.checkpoint_input)
                        elif recompute.is_select(lid):
                            # 选择性重算：选中 op 反向重物化（recomp_scratch，= 选中段 forward_max_live
                            # 扣段边界）与非选中 op 反向工作集（bwd_working_set，= 非选中段
                            # forward_max_live 扣 bwd_scratch）共存。全选→复现 full（recomp=fml−ci、
                            # bwd_ws=0）；全不选→复现无重算（recomp=0、bwd_ws=fml−bwd_scratch）。
                            smem = select_mem_by_id[lid]
                            B.recomp_scratch = smem.recomp_scratch
                            B.bwd_working_set = smem.bwd_working_set
                        else:
                            # 无重算层：反向仍需再遍历 forward 求梯度，激活梯度 dL/dact 与激活同形、
                            # 同样共存 → 反向工作集 ≈ 该层 forward_max_live（§8.5②）。其中已被显式建模
                            # 的 op 级 fp32 反向物化（loss probs+grad_log_softmax / mHC sinkhorn grad）
                            # 已在 bwd_scratch 计入 → 扣除避免双算。loss 层 bwd_scratch(满 vocab fp32
                            # ×2) ≥ forward_max_live → 该项 = 0（不动已验证的 DSv3/loss 峰，无双算）；
                            # transformer 层 bwd_scratch=0 → = forward_max_live（此前欠建的反向工作集）。
                            B.bwd_working_set = max(
                                0, sm.forward_max_live - sm.bwd_scratch)
                        rec(f"bwd@{lid}")
                        B.gather_buf = B.grad_buf = B.recomp_scratch = 0
                        B.bwd_scratch = B.bwd_working_set = B.swap_buf = 0
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
