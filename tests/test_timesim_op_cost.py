# tests/test_timesim_op_cost.py
"""machine.TimeHardware + op_cost.CostModel（M8，spec §4.1）。
测试恒用 SYNTH 合成硬件（整数好算，断言零公差）；DEFAULT_910B 只测元属性不测数值。"""
import pytest

from cost_eval.timesim.machine import TimeHardware, DEFAULT_910B, synth_hw


def test_synth_hw_roundtrip():
    hw = synth_hw()
    assert hw.peak("bf16") == 100e12
    assert hw.link("tp") == (1.0, 1e11)          # (alpha_us, bytes_per_s)
    assert hw.host_us("View", "fwd") == 1.0
    assert hw.host_us("MatMul", "bwd") == 2.0    # 相位缺省价
    assert hw.calibrated is False


def test_default_910b_marked_uncalibrated():
    """厂商规格代入（spec §7.2 诚实边界1）：任何消费方都能看到未标定标记。"""
    assert DEFAULT_910B.calibrated is False
    assert DEFAULT_910B.name == "910B"
    for axis in ("tp", "cp", "ep", "dp", "pp"):
        alpha, bw = DEFAULT_910B.link(axis)
        assert alpha > 0 and bw > 0


def test_unknown_axis_fail_loud():
    with pytest.raises(KeyError):
        synth_hw().link("nvlink")
