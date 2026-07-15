"""P1-19 闭环：param/grad/optimizer **分离**卸载。

审计 P1-19：`cpu_offload` 是一个布尔量，无法表达 param/grad/optimizer 分别卸载。真机 mindformers
可分项卸载（CPUOffloadPolicy 的 offload_params/offload_grads/offload_optimizer 语义）。本测试守卫
四件事：
  1. ParallelConfig 分离标志 + 向后兼容派生（cpu_offload=True 等价三者全 True；缺省全 False）。
  2. OptimizerSpec.state_bytes_per_param 的 param 副本 / opt 状态拆分（bf16=2+12；fp32=0+12）。
  3. static_mem 持久态按 offload_params/offload_optimizer 分项归零（数值例子 + 兼容性逐字节证明）。
  4. mem_timeline：offload_optimizer→optstep=0；offload_grads→grad_accum=0；各自独立。
"""
import pytest

from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.static_mem import StaticMem
from cost_eval.mem_timeline import MemTimeline
from cost_eval.report import Evaluator

GiB = 2 ** 30


# ---------------------------------------------------------------------------
# 1. ParallelConfig 分离标志 + 向后兼容派生
# ---------------------------------------------------------------------------

def test_cpu_offload_true_derives_all_three():
    """cpu_offload=True 等价三者全 True（向后兼容：全或无）。"""
    pc = ParallelConfig(cpu_offload=True)
    assert pc.offload_params is True
    assert pc.offload_grads is True
    assert pc.offload_optimizer is True


def test_default_offload_flags_all_false():
    """缺省全 False（cpu_offload 缺省 False，三分离标志缺省 False）。"""
    pc = ParallelConfig()
    assert pc.cpu_offload is False
    assert pc.offload_params is False
    assert pc.offload_grads is False
    assert pc.offload_optimizer is False


def test_individual_flag_independent_of_cpu_offload():
    """单独设某个分离标志不牵连其它标志、也不隐含 cpu_offload。"""
    pc = ParallelConfig(offload_optimizer=True)
    assert pc.offload_optimizer is True
    assert pc.offload_params is False
    assert pc.offload_grads is False
    assert pc.cpu_offload is False


def test_offload_flag_bool_guard():
    """分离标志须为 bool（现有整数/bool 守卫风格：int 1 不算 bool）。"""
    with pytest.raises(ValueError):
        ParallelConfig(offload_params=1)


# ---------------------------------------------------------------------------
# 2. OptimizerSpec：param 副本 / opt 状态拆分
# ---------------------------------------------------------------------------

def test_optimizer_split_bf16():
    """bf16 params：state=14 = 2 param 副本 + 12 opt(master4+m4+v4)。"""
    o = OptimizerSpec.adamw()
    assert o.state_bytes_per_param == 14
    assert o.optimizer_state_bytes() == 12
    assert o.param_persist_bytes() == 2
    assert o.param_persist_bytes() + o.optimizer_state_bytes() == o.state_bytes_per_param


def test_optimizer_split_fp32():
    """fp32 params：state=12 = 0 param 副本 + 12 opt（master fp32 即 param，无独立副本）。"""
    o = OptimizerSpec.adamw(params_fp32=True)
    assert o.state_bytes_per_param == 12
    assert o.optimizer_state_bytes() == 12
    assert o.param_persist_bytes() == 0
    assert o.param_persist_bytes() + o.optimizer_state_bytes() == o.state_bytes_per_param


# ---------------------------------------------------------------------------
# 3. static_mem 持久态分项归零（数值例子 + 兼容性逐字节）
# ---------------------------------------------------------------------------

_D = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)


def _spec():
    return ModelSpec("toy", _D, ["dense", "dense"], {"dense": build_dense_decoder(_D)})


def _persist(opt=None, **compute_kw):
    """单卡（fsdp=1）持久态[stage0]。默认 alloc_block_bytes=1 → 逐张量对齐是恒等 →
    持久态在每元素倍数上**线性**，便于给数值例子。"""
    pm = ParallelModel(ParallelConfig(), n_layers=2, world_size=1)
    g = ShapeEval().resolve(_spec(), pm)
    return StaticMem().compute(g, opt or OptimizerSpec.adamw(), pm, **compute_kw)[0]


def test_static_default_equals_old_full_byte_identical():
    """缺省（都不卸）逐字节 == 旧 cpu_offload=False。"""
    assert (_persist(offload_params=False, offload_optimizer=False)
            == _persist(cpu_offload=False))


def test_static_bf16_split_numeric():
    """bf16 数值例子（K = Σ 每卡 param numel；block=1 线性 → 持久 = mult·K）。"""
    full = _persist(cpu_offload=False)          # = 14·K
    assert full % 14 == 0
    K = full // 14
    # 只 offload_optimizer → opt(12)=0、param 副本(2) 留
    assert _persist(offload_optimizer=True) == 2 * K
    # 只 offload_params → param 副本(2)=0、opt(12) 留
    assert _persist(offload_params=True) == 12 * K
    # 两者都卸 → 0（== 旧全卸）
    assert _persist(offload_params=True, offload_optimizer=True) == 0
    # cpu_offload=True（全或无）→ 0（== 旧全卸）
    assert _persist(cpu_offload=True) == 0


def test_static_fp32_split_numeric():
    """fp32 数值例子：param 副本分量=0 → offload_params 单独卸不改持久；opt 单独卸即全卸。"""
    fp32 = OptimizerSpec.adamw(params_fp32=True)
    full = _persist(fp32, cpu_offload=False)    # = 12·K
    assert full % 12 == 0
    # fp32 无 param 副本 → offload_optimizer 单独卸即把持久全部（=opt 12）归零
    assert _persist(fp32, offload_optimizer=True) == 0
    # offload_params 单独卸 → 持久不变（param 分量本就 0）
    assert _persist(fp32, offload_params=True) == full


# ---------------------------------------------------------------------------
# 4. mem_timeline：optstep / grad_accum 分项
# ---------------------------------------------------------------------------

_DT = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=4)


def _sim(**pc_kw):
    spec = ModelSpec("toy", _DT, ["dense"] * 4, {"dense": build_dense_decoder(_DT)})
    pc = ParallelConfig(pp=1, **pc_kw)
    pm = ParallelModel(pc, n_layers=4, world_size=1)
    g = ShapeEval().resolve(spec, pm)
    opt = OptimizerSpec.adamw()
    persistent = StaticMem().compute(g, opt, pm, offload_params=pc.offload_params,
                                     offload_optimizer=pc.offload_optimizer)
    return MemTimeline().simulate(
        g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
        framework_reserve=0, max_device_memory=10 ** 12,
        grad_dtype_bytes=opt.grad_dtype_bytes, record_timeline=True)[0]


def _has_optstep(sp):
    xs = [s for s in sp.timeline if s.event == "optstep"]
    return bool(xs) and xs[0].breakdown.optstep > 0


def _max_grad_accum(sp):
    return max(s.breakdown.grad_accum for s in sp.timeline)


def test_timeline_default_has_optstep_and_grad_accum():
    sp = _sim()
    assert _has_optstep(sp)
    assert _max_grad_accum(sp) > 0


def test_offload_optimizer_zeros_optstep_keeps_grad_accum():
    """只 offload_optimizer → optstep=0（优化器 step 在 CPU）、grad_accum 留（梯度仍驻设备）。"""
    sp = _sim(offload_optimizer=True)
    assert not _has_optstep(sp)
    assert _max_grad_accum(sp) > 0


def test_offload_grads_zeros_grad_accum_keeps_optstep():
    """只 offload_grads → grad_accum=0（梯度在 CPU）、optstep 留（优化器 step 仍在设备）。"""
    sp = _sim(offload_grads=True)
    assert _max_grad_accum(sp) == 0
    assert _has_optstep(sp)


def test_cpu_offload_true_zeros_persistent_optstep_grad_accum():
    """cpu_offload=True → 持久=0、optstep=0、grad_accum=0（== 旧全卸）。"""
    sp = _sim(cpu_offload=True)
    assert not _has_optstep(sp)
    assert _max_grad_accum(sp) == 0
    assert all(s.breakdown.persistent == 0 for s in sp.timeline)


def test_full_offload_equiv_all_three_flags_byte_identical():
    """cpu_offload=True 与「三分离标志全 True」逐字节一致。"""
    a = _sim(cpu_offload=True)
    b = _sim(offload_params=True, offload_grads=True, offload_optimizer=True)
    assert a.peak_bytes == b.peak_bytes
    assert a.breakdown == b.breakdown


def test_default_equiv_all_false_byte_identical():
    """缺省与「三分离标志全 False」逐字节一致（回归门）。"""
    a = _sim()
    b = _sim(offload_params=False, offload_grads=False, offload_optimizer=False)
    assert a.peak_bytes == b.peak_bytes
    assert a.breakdown == b.breakdown


# ---------------------------------------------------------------------------
# 5. report 端到端：cpu_offload=True == 三标志全 True（Evaluator 传参闭环）
# ---------------------------------------------------------------------------

def _peak(**pc_kw):
    spec = ModelSpec("toy", _DT, ["dense"] * 4, {"dense": build_dense_decoder(_DT)})
    hw = HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0)
    pc = ParallelConfig(pp=1, **pc_kw)
    return Evaluator(spec, pc, OptimizerSpec.adamw(), hw,
                     RecomputeSpec("None"), SwapSpec()).evaluate().per_stage[0].peak_bytes


def test_report_cpu_offload_equiv_separated():
    assert _peak(cpu_offload=True) == _peak(
        offload_params=True, offload_grads=True, offload_optimizer=True)
    assert _peak() == _peak(
        offload_params=False, offload_grads=False, offload_optimizer=False)
