"""把 per-layer TimedSegment 拼成一个连续 pass 段（spec §5.2：L1 仿真单元=完整 pass，
host 流一条贯到底；"层"只是报告标记）。

op_id 前缀 = f"{layer_seg_id}/"（同 cell 多层的 opdag 节点 id 相同，不前缀必碰撞）；
deps 改写规则：dep 指向本层段内某 op → 加同层前缀；查无此 id（跨段/外部）→ 原样保留，
segment_sim 对未知 dep 按段首已满足处理（pass 边界 run-ahead 截断，spec §7.2-4 有界近似）。"""
from __future__ import annotations

from dataclasses import replace

from .ir import TimedOp, TimedSegment


def layer_of(op_id: str) -> str:
    """pass 内 op_id → 层标记（无前缀=拼接前的裸段，归 ""）。"""
    return op_id.rsplit("/", 1)[0] if "/" in op_id else ""


def concat_segments(seg_id: str, segments: list) -> TimedSegment:
    seg_ids = [s.seg_id for s in segments]
    if len(set(seg_ids)) != len(seg_ids):
        raise ValueError(f"pass_builder: concat_segments 入参段 seg_id 有重复 {seg_ids}——"
                         f"op_id 前缀去碰撞依赖 seg_id 互异(fail-loud)")
    ops: list[TimedOp] = []
    for seg in segments:
        local_ids = {o.op_id for o in seg.ops}
        prefix = seg.seg_id + "/"
        for o in seg.ops:
            ops.append(replace(
                o, op_id=prefix + o.op_id,
                deps=tuple((prefix + d) if d in local_ids else d for d in o.deps)))
    return TimedSegment(seg_id, tuple(ops))
