"""纯 1F1B/VPP 调度代数（自 mem_timeline 搬家，spec 2026-07-16 §2.2 契约2）。

**中立模块**：无任何内存/时间语义，mem_timeline（内存仿真）与 timesim（时间仿真）各自消费，
两者互不 import（tests/test_timesim_decoupling.py 强制）。逐行 port Megatron schedules.py
的注释与 file:line 引用随函数原样保留。"""
from __future__ import annotations
from dataclasses import dataclass


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


# ── m==pp 特例（2026-07-14 检视 P2.9 裁决）────────────────────────────────────────────
# Megatron 在 num_microbatches==pp 时特判 all-warmup（全前向先行,schedules.py）;mindformers
# pynative 的调度器在 hyper_parallel.core.pipeline_parallel（外部库,本地无源码）,经与
# mindformers 侧确认其行为**非全前向先行** → 本库**不移植** Megatron 该特例,统一走通式
# warmup（下函数）。待 hyper_parallel 源码可得/真机 VPP 验证（pp_select_vpp_validation.md
# §4 清单）后再确证。
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

    ── 物理粒度 vs chunk 粒度（D-4）──
    本函数是**物理微批粒度**事件（一个 FWD 事件对应一整个物理 stage）。它现在**只用于 v<=1**
    （simulate 对 v<=1 走此路径、每事件 pin 整个 layer_ids，逐字节复现旧行为）以及作为 warmup 深度
    的物理粒度视图（`interleaved_warmup` 系列 schedule-level 测试）。**v>1 的激活峰值已改由 chunk
    粒度调度**（`interleaved_virtual_order` + `chunk_layer_ids`，见其 docstring 与 §D-4）估计——每
    虚拟步只驻留一个 chunk（L/V 层），峰 = Σ_chunk n_c·(L/V)，去除旧「每在飞微批整 stage L 层」的
    ~V× 过估。真实 VPP 下每物理 stage 拥 **V 个非连续 chunk**（round-robin：物理 rank 拥虚拟 stage
    {chunk*pp+rank}，mindformers `pipeline_parallel.py:257-258`，层均分 `:188-195`），净激活膨胀比
    ≈ 1 + (pp-1)/(pp*V)（V=2 处最大、随 V 回落 → 峰值**非**单调增）。
    """
    if v <= 1:
        return build_1f1b(stage, pp, m)
    return _1f1b_from_warmup(interleaved_warmup(stage, pp, m, v), m)


# ---------------------------------------------------------------------------
# Task [D-4]: 交错式 1F1B 的 **chunk 粒度** 调度 + 切层（VPP 激活忠实累加，去 ~V× 过估）
#
# 旧 build_interleaved_1f1b 是**物理微批粒度**：每个 FWD 事件 pin 整个物理 stage 的全部 L 层，
# 峰 ≈ (warmup+1)·L —— 每在飞微批高估 V 倍（真实 VPP 每虚拟步只驻留一个 chunk=L/V 层）。下面三
# 个函数把调度细化到 **(microbatch, chunk) 虚拟步**（逐行 port Megatron），simulate 对 v>1 改用它
# 们 → 峰 = Σ_chunk n_c·(L/V)，方向(>plain)与幅度(<V×)同时修正。v=1 仍走上面的物理路径（byte 一致）。
# ---------------------------------------------------------------------------

def get_schedule_table(m: int, v: int, group_size: int) -> list:
    """交错式 1F1B 的 `(microbatch_id, chunk_id)` 调度表 —— 逐行 port Megatron
    `get_schedule_table`（`schedules.py:902-929`）。

    deep-first：每组连跑 `group_size` 个微批的**同一 chunk** 再切下一 chunk；末组（`lo+gs>=m`）
    把剩余微批一次排完。`group_size` 默认 = pp（`microbatch_group_size_per_vp_stage`，
    `model_parallel_config.py:519-520`）。返回长度 `m*v` 的 `(mb, chunk)` 列表（每对恰一次）。"""
    table: list = []
    for lo in range(0, m, group_size):
        if lo + group_size >= m:                      # 末组（Megatron :908）
            table.extend([(mb, c) for c in range(v) for mb in range(lo, m)])
        else:
            table.extend([(mb, c) for c in range(v)
                          for mb in range(lo, lo + group_size)])
    return table


def interleaved_virtual_order(stage: int, pp: int, m: int, v: int, group_size: int) -> list:
    """交错式 1F1B 的**虚拟步（chunk 粒度）**事件序列：每步 = 一个 `(mb, chunk)` 的 FWD 或 BWD。

    逐行 port Megatron `convert_schedule_table_to_order`（`schedules.py:932-955`）；warmup 取
    `get_pp_rank_microbatches`（`:877-878`，**clamp 到 total=m*v**，`:889-890`）：

        warmup(虚拟步) = (pp-stage-1)*2 + (v-1)*group_size  →  min(·, m*v)

    前 warmup 步纯 FWD（`:949`）；随后 steady 逐 `(FWD_i, BWD_{i-warmup})` 交替（`:950-952`）；末尾
    warmup 个 BWD 收尾（`:953-954`）。forward/backward 均按同一 schedule table FIFO → 每 `(mb,chunk)`
    前向一次、反向一次（pin/pop 平衡，无泄漏）。返回 `list[(kind, mb, chunk)]`，长度 `2*m*v`。"""
    table = get_schedule_table(m, v, group_size)
    total = len(table)                                # = m*v
    warmup = min((pp - stage - 1) * 2 + (v - 1) * group_size, total)
    steps: list = []
    for i in range(warmup):                           # warmup 纯前向
        mb, c = table[i]
        steps.append(("FWD", mb, c))
    for i in range(warmup, total):                    # steady 1F1B（同一虚拟步序）
        mbf, cf = table[i]
        steps.append(("FWD", mbf, cf))
        mbb, cb = table[i - warmup]
        steps.append(("BWD", mbb, cb))
    for j in range(total - warmup, total):            # cooldown 收尾反向
        mbb, cb = table[j]
        steps.append(("BWD", mbb, cb))
    return steps


def chunk_layer_ids(layer_ids: list, v: int) -> list:
    """把一个物理 stage 的层 id 列表**均衡切成 v 个 chunk**（VPP：每 device 持 V 个 model chunk，
    各 L/V 层）。前 `L%v` 个 chunk 取 ⌈L/v⌉、其余取 ⌊L/v⌋。

    **关键不变式**：`Σ_c len(chunk_c) == L`（所有 chunk 层数之和 = 实际 device 层数）→ 按 chunk pin
    的总激活最多 = L 层 × 在飞倍数，**绝不 V·L**（D-4 修的 ~V× 过估）。真实 VPP 是 round-robin 非连续
    放置（chunk c = 虚拟 stage `c*pp+rank`）；本库沿用既有「连续切层→stage」抽象故此处连续切 —— 层均匀
    时二者 Σ 相同，非均匀时（embedding 归 chunk0 / loss 归末 chunk）连续切亦把伪层落在正确首/末 chunk。
    L<v（非物理 VPP 配置）时尾部 chunk 为空 → 该步 pin 0，graceful 不崩。"""
    L = len(layer_ids)
    base, extra = divmod(L, v)
    chunks: list = []
    i = 0
    for c in range(v):
        n = base + (1 if c < extra else 0)
        chunks.append(layer_ids[i:i + n])
        i += n
    return chunks
