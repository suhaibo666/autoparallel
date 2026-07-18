# tests/test_timesim_l0.py
"""验证阶梯 L0（spec §7.1）：①三态守恒（segment_sim 测试已盖，此处盖 pipeline 级）
③退化还原（单卡串行 roofline 和；1F1B 闭式在 test_timesim_pipeline_sim 已盖）
④性质（tp↑→GEMM t↓通信↑；m↑/recompute 剪刀差在既有测试已盖）
+ L1 轻量互证（MFU 数量级 sanity；Calculon 对标=T2 人工步，见计划收尾说明）。"""
import pytest

from cost_eval.model_spec import DimTable
from cost_eval.timesim.shard_rules import Degrees
from cost_eval.timesim.producer import build_segment
from cost_eval.timesim.machine import synth_hw
from cost_eval.timesim.op_cost import CostModel, price_segment
from cost_eval.timesim.report import evaluate_step_time

DIMS = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                S=4096, B=1, vocab=129280, n_layers=4)


def _layers(mlp_dag, deg, n=2):
    return [build_segment(f"layer_{i}.fwd", mlp_dag, DIMS, deg) for i in range(n)]


def test_l0_serial_roofline_sum_exact(mlp_dag):
    """L0③：全度=1 + host=0 + 无重算 + pp=1,m=1 → t_pipeline == Σ op 时长（串行 roofline 和，
    零公差）。"""
    hw = synth_hw(host_unit_us={("*", "fwd"): 0.0, ("*", "bwd"): 0.0, ("*", "recomp"): 0.0})
    layers = _layers(mlp_dag, Degrees())
    r = evaluate_step_time(layers, Degrees(), hw, pp=1, m=1)
    from cost_eval.timesim.bwd_rules import expand_bwd
    cm = CostModel(hw)
    total = 0.0
    for seg in layers:
        for costs in (price_segment(seg, cm), price_segment(expand_bwd(seg), cm)):
            total += sum(c.t_dev_us + c.t_comm_us for c in costs.values())
    assert r.t_pipeline_us == pytest.approx(total)


def test_l0_pipeline_conservation(mlp_dag):
    """L0①（pipeline 级）：每 stage busy+bubble == T_pipeline。"""
    layers = _layers(mlp_dag, Degrees(), n=4)
    r = evaluate_step_time(layers, Degrees(pp=2), synth_hw(), pp=2, m=4)
    for st in r.per_stage:
        assert st["busy_us"] + st["bubble_us"] == pytest.approx(r.t_pipeline_us)


def test_l0_tp_scissors_gemm_down_comm_up(mlp_dag):
    """L0④：tp↑ → 单卡 GEMM 时间↓、通信时间↑（spec §7.1-L0④）。"""
    hw = synth_hw()
    cm = CostModel(hw)
    seg1 = build_segment("l.fwd", mlp_dag, DIMS, Degrees())
    seg2 = build_segment("l.fwd", mlp_dag, DIMS, Degrees(tp=2, sequence_parallel=True))
    def split(seg):
        costs = price_segment(seg, cm)
        gemm = sum(costs[o.op_id].t_dev_us for o in seg.ops if o.op_type == "MatMul")
        comm = sum(costs[o.op_id].t_comm_us for o in seg.ops)
        return gemm, comm
    g1, c1 = split(seg1)
    g2, c2 = split(seg2)
    assert g2 == pytest.approx(g1 / 2)
    assert c1 == 0.0 and c2 > 0.0


def test_l1_mfu_sanity(mlp_dag):
    """L1 轻量：合成硬件（η=1、host 缺省单价）下 dense MLP 栈的 MFU 落公开数量级
    （0.1~1.0）；未标定绝对值无意义，只做量级 sanity（spec §7.1-L1）。"""
    r = evaluate_step_time(_layers(mlp_dag, Degrees(), n=4), Degrees(), synth_hw(),
                           pp=1, m=2)
    assert 0.1 < r.mfu <= 1.0


def test_l1_report_flags_uncalibrated():
    """诚实边界：DEFAULT_910B（厂商规格）驱动的报告必须带 uncalibrated 标记（§7.2-1）。"""
    from cost_eval.timesim.machine import DEFAULT_910B
    from cost_eval.timesim.ir import TimedOp, TimedSegment
    mm = TimedOp(op_id="c#0", op_type="MatMul", phase="fwd",
                 in_shapes=((64, 1, 64), (64, 64)), out_shape=(64, 1, 64),
                 dtype="bf16", stream="device", src="x.py:1")
    r = evaluate_step_time([TimedSegment("layer_0.fwd", (mm,))], Degrees(),
                           DEFAULT_910B, pp=1, m=1)
    assert r.uncalibrated is True
