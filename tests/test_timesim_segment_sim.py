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


# ── 设备后尾段扫描线（Task 8 对抗性 review 修：跨轴嵌套通信守恒）──────────────────────


def test_tail_single_axis_comm_exceeds_device():
    """单轴通信尾超过 device 收尾：device 尾段 [6,56] 全归该通信轴 exposed + 守恒（S3 形态）。"""
    seg = TimedSegment("s", (_dev("a"), _comm("c1", deps=("a",), axis="tp")))
    costs = {"a": _cost(host=1.0, dev=5.0), "c1": _cost(host=1.0, comm=50.0)}
    st = simulate_segment(seg, costs)
    # a 发射@1→D[1,6]；c1 发射@2、dep a@6 → C_tp[6,56]；device 尾段 [6,56] 全 c1 暴露
    assert st.duration_us == pytest.approx(56.0)
    assert st.t_exposed_comm["tp"] == pytest.approx(50.0)
    assert st.t_host_gap == pytest.approx(1.0)
    total = st.t_compute + st.t_membound + st.t_host_gap + sum(st.t_exposed_comm.values())
    assert total == pytest.approx(st.duration_us)


def test_tail_crossing_axes_conserves():
    """尾段跨轴嵌套通信（dp 预取 deps=() 长且早 + 段内 tp 通信短且晚、严格嵌套在 dp 内——
    frame_comm 的 FSDP/EP/CP 预取真实形态）：Σ三态==makespan 守恒，归因给完成最晚的 dp 轴
    （嵌套 tp 得 0）。旧 fin 排序单前沿版在此**既漏计 1.0（破守恒，Σ=101≠102）又错配轴**
    （给成 {tp:5, dp:94}）——Task 8 对抗性 review S4b。
    关键：c_tp 起点（3）须**严格 >** device 收尾 cursor（=2，由 dev=1 决定）；若 c_tp 恰在
    cursor 起（如 dev=5→cursor=6），旧版在该输入下反而守恒、仅错配轴，conservation 断言便
    区分不出新旧（review 指出的测试精度点，故此处 dev=1 让守恒断言本身也能判别）。"""
    seg = TimedSegment("s", (_dev("a"),
                             _comm("c_dp", deps=(), axis="dp"),
                             _comm("c_tp", deps=("a",), axis="tp")))
    costs = {"a": _cost(host=1.0, dev=1.0),
             "c_dp": _cost(host=1.0, comm=100.0),
             "c_tp": _cost(host=1.0, comm=5.0)}
    st = simulate_segment(seg, costs)
    assert st.duration_us == pytest.approx(102.0)
    total = st.t_compute + st.t_membound + st.t_host_gap + sum(st.t_exposed_comm.values())
    assert total == pytest.approx(st.duration_us)             # L0① 守恒（旧版此输入 Σ=101，缺 1.0）
    assert st.t_exposed_comm["dp"] == pytest.approx(100.0)    # 完成最晚的 dp 轴全担尾段
    assert st.t_exposed_comm.get("tp", 0.0) == pytest.approx(0.0)  # 嵌套 tp 轴 0
    assert st.t_host_gap == pytest.approx(1.0)
    assert st.t_compute == pytest.approx(1.0)


def test_tail_pure_host_no_comm():
    """末尾 host_only op 决定 makespan（无通信尾段）：尾段 [6,11] 纯 host_gap，守恒（elif 分支）。"""
    seg = TimedSegment("s", (_dev("a"), _view("v", deps=("a",))))
    costs = {"a": _cost(host=1.0, dev=5.0), "v": _cost(host=10.0)}
    st = simulate_segment(seg, costs)
    # a: h@1→D[1,6]；v host_only h_clock=1+10=11 → makespan=11；尾段 [6,11] 纯 host 尾巴
    assert st.duration_us == pytest.approx(11.0)
    assert st.t_host_gap == pytest.approx(6.0)                # 首发射 1 + 尾段 5
    assert sum(st.t_exposed_comm.values()) == pytest.approx(0.0)
    total = st.t_compute + st.t_membound + st.t_host_gap + sum(st.t_exposed_comm.values())
    assert total == pytest.approx(st.duration_us)
