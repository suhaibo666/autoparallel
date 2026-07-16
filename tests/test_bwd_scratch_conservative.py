"""round3 A(F10)：bwd_scratch conservative/estimated 双模式。

estimated（默认 window=2）对**非相邻的 >2 个大 scratch 共存**会欠估（OOM-不安全方向）；
conservative（Σ 上界）构造即安全。现实模型全退化为单 scratch → 默认逐字节不变、锚点不破。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_eval.structure_mem import _backward_max_live
from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

MiB, GiB = 2 ** 20, 2 ** 30


class _Op:
    def __init__(self, b):
        self.bwd_scratch_bytes = b


def test_dual_mode_multi_scratch_upper_bound():
    # 非相邻 3 大 scratch：window-2 欠估 4000，conservative 取严格上界 7000。
    ops = [_Op(4000), _Op(0), _Op(3000)]
    assert _backward_max_live(ops) == 4000
    assert _backward_max_live(ops, conservative=True) == 7000


def test_single_scratch_identical():
    # 单大 scratch（现实模型：loss nll / DSA indexer）→ 两模式相等（默认不改锚点的机理根因）。
    ops = [_Op(0), _Op(5000), _Op(0)]
    assert _backward_max_live(ops) == _backward_max_live(ops, conservative=True) == 5000


def test_monotonic_cons_ge_est_random():
    # 任意非负向量：conservative(Σ) ≥ estimated(window-2) 恒成立（OOM 安全、不塌单峰以下）。
    for vec in ([1, 2, 3, 4], [0, 0, 9], [5, 5, 5, 5, 5], [100, 0, 0, 100]):
        ops = [_Op(v) for v in vec]
        assert _backward_max_live(ops, conservative=True) >= _backward_max_live(ops)


def _peak(conservative):
    spec, d, fl = build_dsv3_spec(8)
    d.B = 1
    pc = ParallelConfig(dp_shard=2, cp=1, tp=1, pp=1, sequence_parallel=True, num_microbatches=1)
    hw = HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0,
                      bwd_scratch_conservative=conservative)
    ev = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                   hw, RecomputeSpec("None"), SwapSpec())
    return ev.evaluate().per_stage[0].peak_bytes


def test_dsv3_real_model_single_scratch_default_unchanged():
    # 真实 DSv3 是单大 scratch（loss）→ conservative == estimated（默认锚点逐字节不破的证据）。
    assert _peak(conservative=True) == _peak(conservative=False)


def test_hardware_spec_default_is_estimated():
    assert HardwareSpec(max_device_memory=1).bwd_scratch_conservative is False
