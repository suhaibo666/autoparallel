# tests/test_timesim_op_cost.py
"""machine.TimeHardware + op_cost.CostModel（M8，spec §4.1）。
测试恒用 SYNTH 合成硬件（整数好算，断言零公差）；DEFAULT_910B 只测元属性不测数值。"""
import pytest

from cost_eval.timesim.machine import TimeHardware, DEFAULT_910B, synth_hw
from cost_eval.timesim.ir import TimedOp, TimedSegment, CommSpec
from cost_eval.timesim.op_cost import CostModel, OpCost, price_segment


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


def test_peak_unknown_dtype_fail_loud():
    """peak() 未知 dtype 应 KeyError fail-loud，与姊妹方法 link() 同口径（review Tier C），
    不得静默回退 bf16。"""
    with pytest.raises(KeyError):
        synth_hw().peak("fp8")
    assert synth_hw().peak("bf16") == 100e12
    assert synth_hw().peak("fp32") == 25e12


HW = synth_hw()
CM = CostModel(HW)


def _op(**kw):
    base = dict(op_id="n0", op_type="MatMul", phase="fwd",
                in_shapes=((4096, 1, 1792), (1792, 3072)), out_shape=(4096, 1, 3072),
                dtype="bf16", stream="device", src="mlp.py:1")
    base.update(kw)
    return TimedOp(**base)


def test_gemm_cost_exact():
    c = CM.cost(_op())
    flops = 2 * 4096 * 1 * 1792 * 3072
    assert c.flops == flops
    assert c.t_dev_us == pytest.approx(flops / 100e12 * 1e6)   # η_gemm=1
    assert c.t_comm_us == 0.0
    assert c.bound == "compute"                                 # AI 远超 ridge=100
    assert c.provenance == "theory"
    assert c.eta_key == "gemm:bf16"


def test_gemm_memory_bound_takes_memory_branch():
    """瘦 GEMM（flops>0 但 ai<ridge）：roofline t=max(compute,memory) 应取内存分支，
    而非恒取 compute 分支（review Tier A——被自己标成 memory-bound 的 op 却按 compute 定价）。"""
    thin = _op(in_shapes=((1, 16), (16, 16)), out_shape=(1, 16))
    c = CM.cost(thin)
    flops = 2 * (1 * 16) * 16
    bytes_rw = (16 + 256 + 16) * 2                  # in0+in1+out，bf16=2B/elem
    assert c.flops == flops
    assert c.bytes_rw == bytes_rw
    assert flops / bytes_rw < c.ridge                # 确认真落在 memory 区（ai<ridge=100）
    assert c.bound == "memory"
    assert c.t_dev_us == pytest.approx(bytes_rw / (1e12 * 1.0) * 1e6)      # 内存分支
    assert c.t_dev_us != pytest.approx(flops / (100e12 * 1.0) * 1e6)      # 非 compute 分支


def test_fa_cost_formula():
    """FA fwd = 2·B·N·Sq·Skv·(Dq+Dv)·causal系数（spec §4.1；causal 默认 0.5）；Grad=2.5×。"""
    fa = _op(op_type="FlashAttention",
             in_shapes=((4096, 1, 8, 192), (4096, 1, 8, 192), (4096, 1, 8, 128), ()),
             out_shape=(4096, 1, 8, 128))
    c = CM.cost(fa)
    want = int(2 * 1 * 8 * 4096 * 4096 * (192 + 128) * 0.5)
    assert c.flops == want
    g = CM.cost(fa := _op(op_type="FlashAttentionGrad", phase="bwd",
                          in_shapes=fa.in_shapes, out_shape=fa.out_shape))
    assert g.flops == int(want * 2.5)
    full = CostModel(HW, causal=False).cost(_op(
        op_type="FlashAttention",
        in_shapes=((4096, 1, 8, 192), (4096, 1, 8, 192), (4096, 1, 8, 128), ()),
        out_shape=(4096, 1, 8, 128)))
    assert full.flops == want * 2


def test_fa_requires_4d_fail_loud():
    fa = _op(op_type="FlashAttention", in_shapes=((4096, 1792), (4096, 1792), (4096, 1792)))
    with pytest.raises(ValueError, match="4D"):
        CM.cost(fa)


def test_bandwidth_op_cost():
    """带宽类（flops=0）：t_dev = bytes_rw/(HBM·η)；空 shape（`?` 容忍位）计 0 字节。"""
    norm = _op(op_type="Norm", in_shapes=((4096, 1, 1792), ()), out_shape=(4096, 1, 1792))
    c = CM.cost(norm)
    b = 4096 * 1792 * 2 * 2                       # in + out，bf16
    assert c.flops == 0 and c.bytes_rw == b
    assert c.t_dev_us == pytest.approx(b / 1e12 * 1e6)
    assert c.bound == "memory"


def test_view_host_only_zero_device():
    v = _op(op_type="View", stream="host_only")
    c = CM.cost(v)
    assert c.t_dev_us == 0.0 and c.bound == "host"
    assert c.t_host_us == 1.0


def test_comm_costs_per_ctype():
    """t_comm 按 ir.py volume 口径（AG=分片、RS/AR=全量、p2p 单跳）+ ring 系数（spec §4.1）。"""
    n, bw, alpha = 4, 1e11, 1.0
    mk = lambda ct, vol: _op(op_type="CommOp", stream="comm_tp", in_shapes=(), out_shape=(),
                             comm=CommSpec(ct, vol, "tp", n))
    v = 1_000_000
    assert CM.cost(mk("all_gather", v)).t_comm_us == pytest.approx(alpha + v * (n - 1) / bw * 1e6)
    assert CM.cost(mk("reduce_scatter", v)).t_comm_us == pytest.approx(alpha + v * (n - 1) / n / bw * 1e6)
    assert CM.cost(mk("all_reduce", v)).t_comm_us == pytest.approx(alpha + 2 * v * (n - 1) / n / bw * 1e6)
    assert CM.cost(mk("all_to_all", v)).t_comm_us == pytest.approx(alpha + v * (n - 1) / n / bw * 1e6)
    assert CM.cost(mk("p2p", v)).t_comm_us == pytest.approx(alpha + v / bw * 1e6)
    assert CM.cost(mk("all_gather", v)).bound == "comm"


def test_generic_grad_bytes_convention():
    """通用 <op>Grad（bwd_rules 约定 in=(dy,*fwd_ins)、out=dy）：bytes 直接按 shapes 求和。"""
    g = _op(op_type="NormGrad", phase="bwd",
            in_shapes=((4096, 1, 1792), (4096, 1, 1792)), out_shape=(4096, 1, 1792))
    c = CM.cost(g)
    assert c.bytes_rw == 3 * 4096 * 1792 * 2
    assert c.t_host_us == 2.0                     # bwd 相位单价


def test_host_dominated_flag():
    tiny = _op(op_type="Cast", in_shapes=((8, 8),), out_shape=(8, 8))
    assert CM.cost(tiny).host_dominated is True


def test_lib_param_not_implemented():
    """T1/T2 边界守卫（防静默假装标定过）：传 OpTimeLibrary → NotImplementedError（review Minor-1）。"""
    with pytest.raises(NotImplementedError):
        CostModel(HW, lib=object())


def test_price_segment_returns_cost_per_op():
    """price_segment：整段 {op_id → OpCost}（segment_sim/report 的输入，review Minor-3）。"""
    seg = TimedSegment("s.fwd", (_op(op_id="a"),
                                 _op(op_id="b", op_type="Norm",
                                     in_shapes=((4096, 1, 1792),), out_shape=(4096, 1, 1792))))
    costs = price_segment(seg, CM)
    assert set(costs) == {"a", "b"}
    assert all(isinstance(c, OpCost) for c in costs.values())
    assert costs["a"].bound == "compute" and costs["b"].bound == "memory"
