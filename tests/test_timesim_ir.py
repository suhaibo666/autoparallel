# tests/test_timesim_ir.py
"""TimedOp/TimedSegment/CommSpec（spec §3.2）+ op_flops 纯函数。"""
import dataclasses
import pytest

from cost_eval.timesim.ir import TimedOp, TimedSegment, CommSpec, op_flops


def _op(**kw):
    base = dict(op_id="n0", op_type="MatMul", phase="fwd",
                in_shapes=((4096, 1, 1792), (1792, 3072)), out_shape=(4096, 1, 3072),
                dtype="bf16", stream="device", src="mlp.py:1")
    base.update(kw)
    return TimedOp(**base)


def test_timedop_frozen():
    top = _op()
    with pytest.raises(dataclasses.FrozenInstanceError):
        top.phase = "bwd"


def test_op_flops_matmul():
    # 2·M·K·N：M=4096·1（batch 摊平）、K=1792、N=3072 —— 与 2·numel(A)·N_out 等值
    assert op_flops(_op()) == 2 * 4096 * 1 * 1792 * 3072


def test_op_flops_dx_dw_shapes():
    """普适式 2·numel(A)·N_out 对 bwd 两支同样成立（dW 用朴素 k=in[0][-1] 会算错——回归点）。"""
    fwd = _op()
    dx = _op(in_shapes=(fwd.out_shape, fwd.in_shapes[1]), out_shape=fwd.in_shapes[0])
    dw = _op(in_shapes=(fwd.in_shapes[0], fwd.out_shape), out_shape=fwd.in_shapes[1])
    assert op_flops(dx) == op_flops(fwd)
    assert op_flops(dw) == op_flops(fwd)


def test_op_flops_grouped_matmul_sums_groups():
    top = _op(op_type="GroupedMatMul",
              in_shapes=((8, 512, 1792), (8, 1792, 1024)), out_shape=(8, 512, 1024))
    assert op_flops(top) == 8 * (2 * 512 * 1792 * 1024)


def test_op_flops_nonmatmul_zero():
    assert op_flops(_op(op_type="Norm")) == 0        # FA/带宽类 flops 归 op_cost（T1）


def test_commspec_on_comm_op():
    c = CommSpec(ctype="reduce_scatter", volume_bytes=4096 * 1792 * 2,
                 group_axis="tp", group_size=2)
    top = _op(op_type="CommOp", stream="comm_tp", comm=c, in_shapes=(), out_shape=())
    assert top.comm.ctype == "reduce_scatter"


def test_segment_holds_ops():
    seg = TimedSegment(seg_id="layer_0.fwd", ops=(_op(),))
    assert seg.ops[0].op_type == "MatMul"
