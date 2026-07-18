# tests/test_timesim_segment_sim.py
"""段内多流 DES（M9-L1，spec §5）：H 串行不等 device、D/C_* FIFO+跨流 deps、三态归因守恒。
全部用手搓 TimedOp + 手写 OpCost（不依赖 CostModel——L1 逻辑与定价解耦）。"""
import pytest

from cost_eval.timesim.ir import TimedOp, CommSpec, TimedSegment
from cost_eval.timesim.op_cost import OpCost
from cost_eval.timesim.segment_sim import simulate_segment


def _cost(host=1.0, dev=0.0, comm=0.0, bound="compute"):
    return OpCost(host, dev, comm, 0, 0, 0.0, 100.0, bound, host > dev, "t", "theory")


def _dev(op_id, deps=(), src="f.py:1"):
    return TimedOp(op_id=op_id, op_type="MatMul", phase="fwd",
                   in_shapes=((4, 4), (4, 4)), out_shape=(4, 4), dtype="bf16",
                   stream="device", src=src, deps=deps)


def _comm(op_id, deps=(), axis="tp"):
    return TimedOp(op_id=op_id, op_type="CommOp", phase="fwd", in_shapes=((4, 4),),
                   out_shape=(4, 4), dtype="bf16", stream=f"comm_{axis}", src="f.py:2",
                   deps=deps, comm=CommSpec("all_reduce", 32, axis, 2))


def _view(op_id, deps=()):
    return TimedOp(op_id=op_id, op_type="View", phase="fwd", in_shapes=((4, 4),),
                   out_shape=(4, 4), dtype="bf16", stream="host_only", src="f.py:3",
                   deps=deps)


def test_device_pipelining_no_gap():
    """host 发射快于 device 执行 → device 背靠背，无空洞。"""
    seg = TimedSegment("s", (_dev("a"), _dev("b"), _dev("c")))
    costs = {i: _cost(host=1.0, dev=10.0) for i in ("a", "b", "c")}
    st = simulate_segment(seg, costs)
    assert st.duration_us == pytest.approx(1.0 + 30.0)      # 首 op 等发射，后续流水
    assert st.t_host_gap == pytest.approx(1.0)
    assert st.t_compute == pytest.approx(30.0)
    assert st.host_len_us == pytest.approx(3.0)


def test_host_bound_emerges():
    """host 单价 > device 时长 → 每 op 都等发射 → host_gap 主导（D2：涌现非拍定）。"""
    seg = TimedSegment("s", (_dev("a"), _dev("b"), _dev("c")))
    costs = {i: _cost(host=10.0, dev=1.0) for i in ("a", "b", "c")}
    st = simulate_segment(seg, costs)
    assert st.duration_us == pytest.approx(31.0)            # 3×10 发射 + 尾 op 1
    assert st.t_host_gap == pytest.approx(28.0)             # 10-1=9 ×2 + 首 10
    assert st.t_compute == pytest.approx(3.0)


def test_exposed_comm_attribution():
    """b 依赖慢通信 → device 空洞记 exposed_comm[tp]（发射早已完成——先查发射再查依赖）。"""
    seg = TimedSegment("s", (_dev("a"), _comm("c1", deps=("a",)), _dev("b", deps=("c1",))))
    costs = {"a": _cost(host=1.0, dev=5.0), "c1": _cost(host=1.0, comm=50.0),
             "b": _cost(host=1.0, dev=5.0)}
    st = simulate_segment(seg, costs)
    # 手算：a 发射@1→D[1,6]；c1 发射@2、dep a@6 → C_tp[6,56]；b 发射@3、dep c1@56 → D[56,61]
    assert st.duration_us == pytest.approx(61.0)
    assert st.t_exposed_comm["tp"] == pytest.approx(50.0)
    assert st.t_host_gap == pytest.approx(1.0)
    assert st.t_compute == pytest.approx(10.0)


def test_overlapped_comm_not_exposed():
    """通信与后续无依赖计算并行 → 不暴露（overlap 是位置+DES 的涌现结果）。"""
    seg = TimedSegment("s", (_dev("a"), _comm("c1", deps=("a",)), _dev("b")))
    costs = {"a": _cost(host=1.0, dev=5.0), "c1": _cost(host=1.0, comm=3.0),
             "b": _cost(host=1.0, dev=20.0)}
    st = simulate_segment(seg, costs)
    assert st.t_exposed_comm.get("tp", 0.0) == 0.0
    assert st.duration_us == pytest.approx(26.0)


def test_three_state_conservation():
    """L0①：Σ(t_compute+t_membound+t_host_gap+Σexposed) == makespan（逐段守恒可测）。"""
    seg = TimedSegment("s", (_view("v0"), _dev("a", deps=("v0",)),
                             _comm("c1", deps=("a",)), _dev("b", deps=("c1",)),
                             _view("v1", deps=("b",))))
    costs = {"v0": _cost(host=2.0), "a": _cost(host=1.0, dev=7.0, bound="memory"),
             "c1": _cost(host=1.0, comm=9.0), "b": _cost(host=3.0, dev=4.0),
             "v1": _cost(host=2.0)}
    st = simulate_segment(seg, costs)
    total = st.t_compute + st.t_membound + st.t_host_gap + sum(st.t_exposed_comm.values())
    assert total == pytest.approx(st.duration_us)
    assert st.t_membound == pytest.approx(7.0)


def test_per_layer_and_top_contributors():
    from cost_eval.timesim.pass_builder import concat_segments
    seg = concat_segments("p", [TimedSegment("layer_0.fwd", (_dev("a"),)),
                                TimedSegment("layer_1.fwd", (_dev("a"),))])
    costs = {o.op_id: _cost(host=1.0, dev=5.0) for o in seg.ops}
    st = simulate_segment(seg, costs)
    assert set(st.per_layer) == {"layer_0.fwd", "layer_1.fwd"}
    assert st.per_layer["layer_1.fwd"] == pytest.approx(5.0)
    assert st.top_contributors[0] in ("layer_0.fwd/a", "layer_1.fwd/a")
