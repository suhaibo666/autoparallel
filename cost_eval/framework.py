"""框架/通信瞬态缓冲 framework_reserve —— 按并行配置分解，非固定常数。

设计见 specs §8.6。固定常数只在"变层数"下成立（HCCL/comm/flash 与层数无关），
换并行配置即失效。这里把可解析缩放项（HCCL 组数）单列；moe_comm/flash 系数与池碎片
当前合并为 `residual_calibrated`，须用**变并行配置真机点（ep=2/4卡/变seq）**拆分并验证。
"""
from __future__ import annotations

MIB = 2 ** 20
HCCL_BYTES_PER_GROUP = 200 * MIB   # 真机日志 hcclBufferSize=200MB（CANN 9.0）


def num_comm_groups(pc) -> int:
    """启用的 HCCL 通信组数 = world + 各启用并行域。

    随并行配置缩放（ep=2 多一组 EP all-to-all 组、tp=2 多一组 TP 组…）。
    """
    n = 1  # hccl_world_group 恒有
    if pc.dp_shard * pc.cp > 1:
        n += 1   # FSDP 组
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


def framework_reserve(pc, residual_calibrated: int = 0) -> int:
    """framework_reserve(config) = HCCL(200MB × 组数) + residual_calibrated。

    `residual_calibrated`：MoE all-to-all staging + flash workspace + bf16 cast 副本 + 池碎片，
    目前合并为 1 个标定值。⚠ specs §8.6/§8.7 开放项：须跑变并行配置真机点把它再拆成
    `moe_comm(op图 a2a 量) + flash_ws(seq×heads) + frag(平台小常数)` 并各自验证。
    """
    return HCCL_BYTES_PER_GROUP * num_comm_groups(pc) + residual_calibrated
