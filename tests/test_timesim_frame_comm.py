"""frame_comm：框架层通信注入 FSDP/EP/CP（spec §3.3c 第二类——不在 layer construct 源码里的通信）。"""
import pytest

from cost_eval.timesim.ir import TimedOp, TimedSegment
from cost_eval.timesim.frame_comm import inject_fsdp, inject_ep, inject_cp


def _mm(op_id="a#1", w=(1792, 6144)):
    return TimedOp(op_id=op_id, op_type="MatMul", phase="fwd",
                   in_shapes=((4096, 1, 1792), w), out_shape=(4096, 1, 6144),
                   dtype="bf16", stream="device", src="mlp.py:1")


def test_fsdp_gather_injected_at_segment_head():
    seg = TimedSegment("layer_0.fwd", (_mm(), _mm("a#2", w=(3072, 1792))))
    seg2 = inject_fsdp(seg, dp_shard=4)
    first = seg2.ops[0]
    assert first.op_type == "CommOp" and first.comm.ctype == "all_gather"
    assert first.stream == "comm_dp" and first.comm.group_axis == "dp"
    # 载荷 = 段内权重字节（两 matmul 的 in_shapes[1]，bf16）
    assert first.comm.volume_bytes == (1792 * 6144 + 3072 * 1792) * 2
    assert seg2.ops[1:] == seg.ops


def test_fsdp_noop_when_unsharded():
    seg = TimedSegment("l.fwd", (_mm(),))
    assert inject_fsdp(seg, dp_shard=1) is seg


def test_ep_alltoall_wraps_grouped_region():
    g = TimedOp(op_id="m#5", op_type="GroupedMatMul", phase="fwd",
                in_shapes=((2, 2048, 1792), (2, 1792, 1024)), out_shape=(2, 2048, 1024),
                dtype="bf16", stream="device", src="moe.py:1")
    seg = TimedSegment("moe.fwd", (_mm(), g, _mm("a#9", w=(1024, 1792))))
    seg2 = inject_ep(seg, ep=4)
    kinds = [(o.op_type, getattr(o.comm, "ctype", None)) for o in seg2.ops]
    assert kinds[1] == ("CommOp", "all_to_all")   # dispatch 在 grouped 区前
    assert kinds[3] == ("CommOp", "all_to_all")   # combine 在 grouped 区后
    disp = seg2.ops[1]
    assert disp.stream == "comm_ep" and disp.comm.group_size == 4
    assert disp.comm.volume_bytes == 2 * 2048 * 1792 * 2   # grouped 输入激活字节（balanced 口径）


def test_ep_noop_without_grouped():
    seg = TimedSegment("l.fwd", (_mm(),))
    assert inject_ep(seg, ep=4) is seg


def test_cp_ring_p2p_precedes_flash_attention():
    fa = TimedOp(op_id="a#3", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224),) * 3, out_shape=(2048, 1, 8, 224),
                 dtype="bf16", stream="device", src="attention.py:1")
    seg = TimedSegment("l.fwd", (fa,))
    seg2 = inject_cp(seg, cp=2, method="colossal")
    assert seg2.ops[0].op_type == "CommOp" and seg2.ops[0].comm.ctype == "p2p"
    assert seg2.ops[0].stream == "comm_cp" and seg2.ops[0].comm.group_axis == "cp"
    # 载荷 = k+v 字节（in_shapes[1]/[2]；跳数系数归 op_cost/T1）
    assert seg2.ops[0].comm.volume_bytes == 2 * (2048 * 1 * 8 * 224 * 2)


def test_cp_ulysses_alltoall_both_sides():
    fa = TimedOp(op_id="a#3", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224),) * 3, out_shape=(2048, 1, 8, 224),
                 dtype="bf16", stream="device", src="attention.py:1")
    seg2 = inject_cp(TimedSegment("l.fwd", (fa,)), cp=2, method="ulysses")
    ctypes = [o.comm.ctype for o in seg2.ops if o.op_type == "CommOp"]
    assert ctypes == ["all_to_all", "all_to_all"]


def test_cp_unknown_method_fail_loud():
    with pytest.raises(ValueError):
        inject_cp(TimedSegment("l.fwd", (_mm(),)), cp=2, method="ring2")
