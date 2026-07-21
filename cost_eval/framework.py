"""框架瞬态缓冲 framework_reserve —— **已按机理完全拆解为 op 图内的逐 op 公式（消除经验常数）**。

设计见 specs §8.6/§14。评估器预测 `max_memory_allocated`（张量占用，OOM 相关）。

**演进：从经验兜底常数 → 逐 op 机理公式（2026-07-02）**
原 `framework_reserve` 是个**经验兜底位**（曾 2197→177→63 MiB），把一切没显式建模的东西一锅端。
本次把它**逐项按机理拆进 op 图**（每项只在其算子活跃的事件计入，绝非全局常数——全局常数会在
loss 峰值也错误地叠加 flash-ws 等，而那里根本没有 attention 在跑）：

  1. **FSDP2 参数预取双缓冲**（~114 MiB @DSv3）→ `gather_buf`（`mem_timeline._prefetch_param_bytes`，
     忠实 FSDP2 默认 depth-1，2026-07-02 已拆）。
  2. **flash-attention workspace**（softmax LSE，∝S·n_heads）→ flash op 的 `workspace`
     （`attention.FLASH_LSE_WS`，T2；仅 attention 前向事件）。
  3. **MoE all-to-all staging**（置换发送/散射缓冲，∝dispatched_tokens·H）→ dispatch/combine op 的
     `workspace`（`ffn.MOE_STAGING_WS`，T3；仅 MoE 前向事件）。
  4. **mHC / MTP 反向瞬态**（×n 残差流梯度、MTP 内层 mHC）→ mHC op `bwd_scratch` + MTP mHC 包装
     （T1；仅对应层反向事件）。
  5. **分配器块对齐碎片** → **公式**：`structure_mem` 逐张量按 `HardwareSpec.alloc_block_bytes`
     （=512B，MindSpore `DynamicMemPoolBestFit.kDynamicMemAlignSize`，**平台属性非拟合**）上取整
     （T4）。DSv3 各建模张量本已 512 对齐 → 该项为 0。

→ **`framework_reserve` 生产默认 = 0**（无拟合 MiB 常数）。剩余极小残差（DSv3 ~0.5%，属未建的
sub-block 临时量：rope cos/sin 表、norm rstd、cast 临时张量等）是**已文档化的已知小残差**，不再兜进常数。
`residual_calibrated` 仅作**审计/回归旋钮**（显式给值复现旧经验常数，如 golden 的 177 MiB）。

HCCL 通信缓冲在 **reserved** 池、不进 allocated 峰值（ep=2 真机证实），归 `hccl_reserved_buffer`
（仅 reserved 预测用）。
"""
from __future__ import annotations

MIB = 2 ** 20
HCCL_BYTES_PER_GROUP = 200 * MIB   # 真机日志 hcclBufferSize=200MB（CANN 9.0）；属 reserved 池


def communicator_breakdown(pc) -> list:
    """真机会建的 HCCL 通信域枚举（**去重后**，源忠实 mindformers/TorchTitan
    `pynative/distributed/parallel_dims.py` `build_mesh`）。返回 `[(name, size), ...]`，**仅** size>1
    的通信器（size==1 无跨卡通信、不占 `hcclBufferSize`；单卡 → 空）。

    `build_mesh` 从 1D world mesh unflatten 出三张网格 + 一个 flatten，各命名轴建独立进程组：
      - dataloading `["pp","batch","cp","tp"]`；dense `["pp","dp_replicate","fsdp","tp"]`；
        sparse `["pp","dp_replicate","efsdp","ep"]`；`loss`=flatten(batch,cp)。
    真正建 HCCL 通信器的轴（`_mesh_exist`）→ 本函数逐个列出：

      | 通信域 | size | 说明 |
      |---|---|---|
      | world | Πdegrees | 全局组，恒建 |
      | fsdp | dp_shard·cp | dense，**恒 real**（MixedPrecision）；size>1 才占 buffer |
      | loss | dp_replicate·dp_shard·cp | flatten(batch,cp)，**恒 real**（每步 all-reduce loss/aux）|
      | tp / cp / pp / dp_replicate | 各自 | size>1 才建 |
      | ep / efsdp | ep / dp_shard·cp·tp//ep | ep>1 才建 |

    **去重规则**（你要求的 groups 去重）：
      1. `pp`/`tp`/`dp_replicate` 在三张网格里**都出现，但是同一个进程组** → 各只列一次（此处天然不重列）。
      2. **只计做「大张量集合」的域**（真占 ~200MB `hcclBufferSize`）：`fsdp`/`dp_replicate`（param/grad
         all-gather·reduce-scatter）、`tp`（激活 all-reduce）、`cp`（KV all-gather/ring）、`ep`（token
         all-to-all）、`efsdp`（专家 FSDP）、`pp`（激活 P2P）+ `world`（全局屏障/global 归约，沿用 DSv4
         标定）。**`loss`（仅标量 loss/aux all-reduce，缓冲可忽略）与 `batch`（纯数据加载、无集合）不计。**
      3. `fsdp`/`cp`/`ep`/`efsdp` 各自独立域 → 分别计（**此前旧模型把 cp 误折进 fsdp、漏 efsdp**）。
      4. **rank 集重叠不去重**：框架按轴各建一个进程组，即便两组覆盖同一批 rank（如 pure-FSDP 下 world
         与 fsdp 同为全体 rank）也**各占一份** buffer——DSv4 真机 reserved 佐证（world+fsdp 各 200MB）。
      5. 仅 **size>1** 才有跨卡通信/占 buffer（单卡 → 空、0 HCCL）。
    """
    dp_r, dp_s = pc.dp_replicate, pc.dp_shard
    cp, tp, pp, ep = pc.cp, pc.tp, pc.pp, pc.ep
    world = dp_r * dp_s * cp * tp * pp
    fsdp = dp_s * cp
    comms = [
        ("world", world),                      # 全局组（沿用 DSv4 world+fsdp=2 标定）
        ("fsdp", fsdp),                        # dense：param/grad all-gather·RS（大集合）
    ]
    if pp > 1:
        comms.append(("pp", pp))
    if tp > 1:
        comms.append(("tp", tp))
    if cp > 1:
        comms.append(("cp", cp))               # dataloading 轴，**独立于 fsdp**（旧模型误折）
    if dp_r > 1:
        comms.append(("dp_replicate", dp_r))
    if ep > 1:
        comms.append(("ep", ep))
        comms.append(("efsdp", fsdp * tp // ep))
    return [(n, s) for n, s in comms if s > 1]   # 仅 size>1 才真占 hcclBufferSize


def num_distinct_communicators(pc) -> int:
    """去重后 size>1 的**不同** HCCL 通信器数（见 `communicator_breakdown` 的枚举与去重规则）。"""
    return len(communicator_breakdown(pc))


def hccl_reserved_buffer(pc) -> int:
    """HCCL 缓冲估计（**reserved** 池，不进 allocated）。= 200MB × 去重后 size>1 通信器数。

    诚实边界：`HCCL_BYTES_PER_GROUP` 是 CANN 默认 hcclBufferSize（可按 config 调）；每域是否满 200MB
    随集合类型/消息量波动，故为口径估计。**只影响 reserved 预测、不进 allocated 峰值**（ep=2 真机证实）。
    reserved 的 HCCL/pool 分账未被独立真机锚点约束（见 §pool 注释），本次把通信器**枚举**做忠实，绝对
    分账仍是标定近似。"""
    return HCCL_BYTES_PER_GROUP * num_distinct_communicators(pc)


# ── Allocator pool 碎片（P2-01 §F7 闭环，2026-07-15）──────────────────────────────
# MindSpore 设备内存池 `DynamicMemPoolBestFit`（best-fit 分配 + 块管理）里 **reserved > allocated**
# 的差额由三部分组成：
#   ① best-fit 分配在已占块内留下的**空洞**（碎片）——张量释放/重分配后留下无法立即复用的间隙；
#   ② `mempool_block_size` 预分配的整块**尾部**（池按大块向设备申请，最后一块通常用不满）；
#   ③ 512B 对齐（`kDynamicMemAlignSize`）——这一分量**已在** structure_mem 逐张量 roundup 建模
#      （见本模块 docstring §5 / framework_reserve 「块对齐取整」项），故此处**不重复计**。
# 剩余 ①②是**聚合碎片**，尺度上随驻留峰值放大，最简可辩护的物理模型是**碎片率 × allocated 峰值**。
#
# 单点标定（真机 DSv4 hybrid 4L，FSDP-2）：allocated 15415.5 MiB、reserved 16092-16096 MiB，
# 差 676.5-680.5 MiB；扣除 HCCL 通信缓冲 ~400 MiB（world + FSDP 组 = 2×200MB）后，pool 分量
# ≈ 277-281 MiB → 碎片率 ≈ 277/15415 ≈ 1.8%。取 1.8%：pool(15415.5)≈277.5 MiB，
# reserved 估计 ≈ 15415.5+400+277.5 ≈ 16093 MiB，落在真机 16092-16096 区间内。
#
# ⚠ **单点标定、跨模型稳定性待验证**：realmachine profiler（memory_record_rank0.csv，另一 DSv3 run）
# 在峰值 allocated 处 reserved−allocated ≈ 996 MiB（扣 HCCL 后 pool 分量 ~4-5%），与本 1.8% 不一致
# ——碎片率随 batch/重算/分配序列显著波动。本模型仅取 DSv4 单点，是**近似**而非严格上界。
POOL_FRAGMENTATION_RATE = 0.018   # 自 DSv4 单点标定的聚合 best-fit 碎片率（277 MiB / 15415.5 MiB）


def allocator_pool_fragmentation(pc, allocated_bytes: int) -> int:
    """DynamicMemPoolBestFit 聚合块级碎片估计（**reserved** 池分量，**不进** allocated 峰值）。

    模型：`碎片率 × allocated 峰值`（碎片率 `POOL_FRAGMENTATION_RATE`=1.8%，自真机 DSv4 单点标定）。
    覆盖 best-fit 空洞 + mempool_block_size 预留块尾部；512B 对齐分量已在 structure_mem 逐张量
    roundup 单独建模，此处不重复计。

    参数
    ----
    pc : ParallelConfig | None
        并行配置。当前**纯碎片率模型不依赖** pc（保留形参供未来「按块数 / mempool_block_size
        建块级模型」时细化——不同并行度下的分配块数不同）。允许传 None。
    allocated_bytes : int
        allocated 峰值字节（结构/激活/优化器等真实张量占用）。≤0 时返回 0（无分配即无碎片）。

    返回
    ----
    int : pool 碎片字节数（**近似**，非严格上界；自 DSv4 单点标定，跨模型稳定性待验证）。
    """
    if allocated_bytes <= 0:
        return 0
    return int(POOL_FRAGMENTATION_RATE * allocated_bytes)


def framework_reserve(pc, residual_calibrated: int = 0) -> int:
    """framework_reserve(**allocated 峰值**) —— **生产默认 0**（机理项已拆进 op 图，见模块 docstring）。

    参数
    ----
    residual_calibrated : int
        **审计/回归旋钮**，默认 0。生产路径不传（返回 0，无拟合常数）。仅在需要复现旧「经验兜底
        常数」以审计锚点漂移时显式给值（如 `test_dsv3_golden` 传 177 MiB 复现拆解前的逐桶 breakdown）。
        **不是拟合的物理量**：真正的机理项（预取/flash-ws/MoE-staging/mHC-MTP/分配器对齐）都已在别处
        按公式建模。
    """
    return residual_calibrated
