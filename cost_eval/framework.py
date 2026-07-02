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


def num_distinct_communicators(pc) -> int:
    """启用的**不同** HCCL 子通信器数（world + 各启用并行域的子组）。

    ⚠ 这些子域**复用同一 rank 网格**（`parallel_dims.py`: EP/CP/TP 从 dp_shard·cp·tp 区
    carve 出，非新增 rank）——即都是 world group 的**子通信器**，**不是独立叠加的**。
    每个不同子通信器在 **reserved** 池里预留一份缓冲，但因域重叠、buffer 可部分共享，
    **不是干净的 ×200MB**；且这一切只影响 reserved、**不进 allocated 峰值**（ep=2 真机证实）。
    """
    n = 1  # hccl_world_group
    if pc.dp_shard * pc.cp > 1:   # FSDP 组（dp_shard·cp）——已含 cp，勿再单列 cp（Task 8 去重）
        n += 1
    if pc.tp > 1:
        n += 1
    if pc.ep > 1:
        n += 1
    if pc.pp > 1:
        n += 1
    if pc.dp_replicate > 1:
        n += 1
    return n


def hccl_reserved_buffer(pc) -> int:
    """HCCL 缓冲粗估（**reserved** 池，不进 allocated）。仅 reserved 预测用，且因域复用为上界估计。"""
    return HCCL_BYTES_PER_GROUP * num_distinct_communicators(pc)


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
