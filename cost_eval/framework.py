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


def num_comm_groups(pc) -> int:
    """启用的 HCCL 通信组数 = world + 各启用并行域（随 ep/tp/... 缩放）。"""
    n = 1  # hccl_world_group 恒有
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
    """HCCL 通信缓冲（在 **reserved** 池，不进 allocated 峰值）。仅当预测 reserved 时用。"""
    return HCCL_BYTES_PER_GROUP * num_comm_groups(pc)


def framework_reserve(pc, residual_calibrated: int = 0) -> int:
    """framework_reserve(**allocated 峰值**) = residual_calibrated（**不含 HCCL**，§8.6 ep=2 修正）。

    `residual_calibrated`：MoE all-to-all staging + flash workspace + bf16 cast + 池碎片。
    真机 ep=1/2、层 4/8 下近恒定；随 seq（flash_ws）等的缩放待 seq-varying 真机点验证（§8.7 开放项）。
    """
    return residual_calibrated
