"""bwd 展开规则库（spec §3.3d/e）：bprop_rules 的姊妹件，逆拓扑序 + 通信对偶(volume 换算) + recompute 前缀。"""
import pytest

from cost_eval.timesim.ir import TimedOp, TimedSegment, CommSpec, op_flops
from cost_eval.timesim.bwd_rules import expand_bwd


def _mm():
    return TimedOp(op_id="c#0", op_type="MatMul", phase="fwd",
                   in_shapes=((4096, 1, 1792), (1792, 3072)), out_shape=(4096, 1, 3072),
                   dtype="bf16", stream="device", src="mlp.py:1")


def _rs():
    return TimedOp(op_id="c#1.rs", op_type="CommOp", phase="fwd", in_shapes=((4096, 1, 1792),),
                   out_shape=(2048, 1, 1792), dtype="bf16", stream="comm_tp", src="layers.py:619",
                   comm=CommSpec("reduce_scatter", 4096 * 1792 * 2, "tp", 2))


def _ag():
    return TimedOp(op_id="c#2.ag", op_type="CommOp", phase="fwd", in_shapes=((2048, 1, 1792),),
                   out_shape=(4096, 1, 1792), dtype="bf16", stream="comm_tp", src="layers.py:1",
                   comm=CommSpec("all_gather", 2048 * 1792 * 2, "tp", 2))


def _view():
    return TimedOp(op_id="c#3", op_type="View", phase="fwd", in_shapes=((4096, 1, 1792),),
                   out_shape=(4096, 1792), dtype="bf16", stream="host_only", src="mlp.py:2")


def test_matmul_expands_to_dx_dw_with_2x_flops():
    seg = expand_bwd(TimedSegment("l.fwd", (_mm(),)))
    bw = [o for o in seg.ops if o.phase == "bwd"]
    assert [o.op_type for o in bw] == ["MatMul", "MatMul"]        # dX + dW
    assert sum(op_flops(o) for o in bw) == 2 * op_flops(_mm())
    assert bw[0].out_shape == _mm().in_shapes[0]                  # dX shape=输入
    assert bw[1].out_shape == _mm().in_shapes[1]                  # dW shape=权重


def test_single_input_matmul_fail_loud():
    """module=="" 透传 matmul 可能仅 1 入（ir.py 元数警示）——bwd 无权重可回传 → fail-loud 而非 IndexError。"""
    one_in = TimedOp(op_id="c#9", op_type="MatMul", phase="fwd",
                     in_shapes=((4096, 1, 1792),), out_shape=(4096, 1, 3072),
                     dtype="bf16", stream="device", src="x.py:1")
    with pytest.raises(ValueError):
        expand_bwd(TimedSegment("l.fwd", (one_in,)))


def test_comm_dual_with_volume_rescale():
    """RS→AG volume ÷group_size；AG→RS volume ×group_size（ir.py CommSpec 对偶换算规则）。"""
    seg = expand_bwd(TimedSegment("l.fwd", (_mm(), _rs())))
    bw = [o for o in seg.ops if o.phase == "bwd"]
    assert bw[0].op_type == "CommOp" and bw[0].comm.ctype == "all_gather"   # RS↔AG 对偶，先反向
    assert bw[0].comm.volume_bytes == (4096 * 1792 * 2) // 2               # RS 记全量 → AG 记分片 ÷2
    assert bw[0].stream == "comm_tp"
    seg2 = expand_bwd(TimedSegment("l.fwd", (_ag(),)))
    dual2 = [o for o in seg2.ops if o.phase == "bwd"][0]
    assert dual2.comm.ctype == "reduce_scatter"
    assert dual2.comm.volume_bytes == (2048 * 1792 * 2) * 2                # AG 记分片 → RS 记全量 ×2


def test_view_stays_host_only():
    seg = expand_bwd(TimedSegment("l.fwd", (_view(),)))
    assert all(o.stream == "host_only" for o in seg.ops if o.phase == "bwd")


def test_flash_attention_single_grad_kernel():
    fa = TimedOp(op_id="c#4", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224),) * 3, out_shape=(2048, 1, 8, 224),
                 dtype="bf16", stream="device", src="attention.py:1")
    bw = [o for o in expand_bwd(TimedSegment("l.fwd", (fa,))).ops if o.phase == "bwd"]
    assert [o.op_type for o in bw] == ["FlashAttentionGrad"]


def test_bandwidth_ops_single_grad():
    norm = TimedOp(op_id="c#5", op_type="Norm", phase="fwd", in_shapes=((4096, 1, 1792),),
                   out_shape=(4096, 1, 1792), dtype="fp32", stream="device", src="n.py:1")
    bw = [o for o in expand_bwd(TimedSegment("l.fwd", (norm,))).ops if o.phase == "bwd"]
    assert [o.op_type for o in bw] == ["NormGrad"] and bw[0].stream == "device"


def test_reverse_order():
    seg = expand_bwd(TimedSegment("l.fwd", (_mm(), _view())))
    bw = [o for o in seg.ops if o.phase == "bwd"]
    assert bw[0].op_type == "View" and bw[1].op_type == "MatMul"   # 逆序展开


def test_recompute_prefix_full_and_comm_drop():
    fwd = (_mm(), _rs(), _view())
    seg = expand_bwd(TimedSegment("l.fwd", fwd), recompute="full", recomp_comm=False)
    rc = [o for o in seg.ops if o.phase == "recomp"]
    assert [o.op_type for o in rc] == ["MatMul", "View"]          # 重放 fwd 序，剔 CommOp
    assert seg.ops[:len(rc)] == tuple(rc)                          # 前缀在 bwd 之前
    seg2 = expand_bwd(TimedSegment("l.fwd", fwd), recompute="full", recomp_comm=True)
    assert [o.op_type for o in seg2.ops if o.phase == "recomp"] == ["MatMul", "CommOp", "View"]


def test_seg_id_suffix():
    assert expand_bwd(TimedSegment("layer_0.fwd", (_mm(),))).seg_id == "layer_0.bwd"
