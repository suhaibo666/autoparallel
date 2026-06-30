"""框架瞬态缓冲 framework_reserve —— 按配置建模（真机 ep=2 修正）。

设计见 specs §8.6。**关键修正（2026-06-30，ep=2 真机点）**：
评估器预测的是 `max_memory_allocated`（张量占用，OOM 相关）。真机 ep=1→2 对比：
allocated 峰值不变（12473→12474），但 reserved 涨（13446→13750）——
**HCCL 通信缓冲在 reserved 池、不在 allocated 峰值**。故 HCCL **不计入** framework_reserve(allocated)。

`framework_reserve(allocated)` = MoE all-to-all staging + flash workspace + bf16 cast 副本 + 池碎片，
实测在 ep=1/2、层数 4/8 下**近恒定**（≈2197 MiB @ seq4096）。其随 seq（flash_ws）等的缩放
仍待 seq-varying 真机点验证。HCCL 归 `hccl_reserved_buffer`（仅 reserved 预测用）。
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
    if pc.dp_shard * pc.cp > 1:
        n += 1
    if pc.tp > 1:
        n += 1
    if pc.ep > 1:
        n += 1
    if pc.cp > 1:
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
    """framework_reserve(**allocated 峰值**) = residual_calibrated（**不含 HCCL**，§8.6 ep=2 修正）。

    `residual_calibrated`：MoE all-to-all staging + flash workspace + bf16 cast + 池碎片。
    真机 ep=1/2、层 4/8 下近恒定；随 seq（flash_ws）等的缩放待 seq-varying 真机点验证（§8.7 开放项）。
    """
    return residual_calibrated
