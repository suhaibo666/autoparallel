"""concat_segments：op_id 层前缀 + deps 改写 + 层标记。"""
import pytest

from cost_eval.timesim.ir import TimedOp, TimedSegment
from cost_eval.timesim.pass_builder import concat_segments, layer_of


def _seg(seg_id):
    a = TimedOp(op_id="c#0", op_type="MatMul", phase="fwd",
                in_shapes=((4, 4), (4, 4)), out_shape=(4, 4), dtype="bf16",
                stream="device", src="f.py:1")
    b = TimedOp(op_id="c#1", op_type="CommOp", phase="fwd", in_shapes=((4, 4),),
                out_shape=(4, 4), dtype="bf16", stream="comm_tp", src="f.py:2",
                deps=("c#0",))
    return TimedSegment(seg_id, (a, b))


def test_concat_prefixes_and_remaps():
    p = concat_segments("s0.mb0.fwd", [_seg("layer_0.fwd"), _seg("layer_1.fwd")])
    assert p.seg_id == "s0.mb0.fwd"
    ids = [o.op_id for o in p.ops]
    assert ids == ["layer_0.fwd/c#0", "layer_0.fwd/c#1",
                   "layer_1.fwd/c#0", "layer_1.fwd/c#1"]
    assert p.ops[1].deps == ("layer_0.fwd/c#0",)
    assert p.ops[3].deps == ("layer_1.fwd/c#0",)          # 层内 deps 只指向本层前缀
    assert layer_of(p.ops[2].op_id) == "layer_1.fwd"


def test_concat_drops_cross_layer_unknown_deps():
    """跨段/外部 dep（段内查无此 id）保留原样——segment_sim 视作段首已满足
    （pass 边界截断，spec §7.2-4）；本函数不静默删除信息。"""
    a = TimedOp(op_id="c#0", op_type="MatMul", phase="bwd",
                in_shapes=((4, 4), (4, 4)), out_shape=(4, 4), dtype="bf16",
                stream="device", src="f.py:1", deps=("external#9",))
    p = concat_segments("s0.mb0.bwd", [TimedSegment("layer_0.bwd", (a,))])
    assert p.ops[0].deps == ("external#9",)


def test_concat_rejects_duplicate_seg_ids():
    """review [15]：op_id 前缀去碰撞方案（`seg.seg_id + "/"`）整个成立的前提是入参段
    seg_id 互异。若重复，op_id 会跨段碰撞，price_segment/simulate_segment 的 op_by_id
    dict 只留最后一个、层间 deps 解析到错的 op——静默出错，须 fail-loud 拒绝。"""
    with pytest.raises(ValueError, match="seg_id"):
        concat_segments("p", [_seg("layer_0.fwd"), _seg("layer_0.fwd")])
