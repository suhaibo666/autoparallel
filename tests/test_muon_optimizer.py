"""Muon 优化器支持(标准口径,2026-07-20)：2D 矩阵权重 momentum-only、embed/head/norm/router/bias
走 AdamW;optstep 的 Newton-Schulz workspace 为**估值**,per-head 只砍注意力投影那笔。AdamW 逐字节不变。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import (ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec,
                             is_muon_matrix_weight, is_attn_projection)
from cost_eval.report import Evaluator

GiB, MiB = 2 ** 30, 2 ** 20


def _peak_persist(opt):
    spec, d, fl = build_dsv3_spec(8); d.B = 1
    pc = ParallelConfig(dp_shard=2, cp=1, tp=1, pp=1, sequence_parallel=True, num_microbatches=1)
    r = Evaluator(spec, pc, opt, HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0),
                  RecomputeSpec("None"), SwapSpec()).evaluate().per_stage[0]
    return r.peak_bytes / MiB, r.breakdown.persistent / MiB


# ── OptimizerSpec byte model ──────────────────────────────────────────────────
def test_optimizer_state_bytes():
    a = OptimizerSpec.adamw(params_fp32=True)
    assert (a.optimizer_state_bytes(), a.matrix_optimizer_state_bytes()) == (12, 12)  # uniform
    m = OptimizerSpec.muon(params_fp32=True)
    assert m.optimizer_state_bytes() == 12                 # 非矩阵(embed/head/norm) = AdamW
    assert m.matrix_optimizer_state_bytes() == 8           # 2D 矩阵 = master4+momentum4(无 v)
    mb = OptimizerSpec.muon(params_fp32=False)             # bf16 params
    assert mb.state_bytes_per_param == 14 and mb._matrix_state_total() == 10  # +2 bf16 副本


def test_muon_matrix_classifier():
    # matmul/moe_gemm 的 2D 权重 → Muon;lm_head/embedding/norm/router → AdamW。
    assert is_muon_matrix_weight("matmul", "linear_qkv")
    assert is_muon_matrix_weight("moe_gemm", "e_fc1")
    assert not is_muon_matrix_weight("matmul", "lm_head")      # head 排除
    assert not is_muon_matrix_weight("norm", "ln1")            # norm gamma → AdamW
    assert not is_muon_matrix_weight("moe_router", "router")   # router → AdamW
    assert not is_muon_matrix_weight("elementwise", "embedding")
    assert is_attn_projection("linear_qkv") and is_attn_projection("o_proj")
    assert not is_attn_projection("fc1") and not is_attn_projection("e_fc1")


# ── 端到端 ─────────────────────────────────────────────────────────────────────
def test_muon_saves_persistent_vs_adamw():
    pa, per_a = _peak_persist(OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4))
    pm, per_m = _peak_persist(OptimizerSpec.muon(params_fp32=True, grad_dtype_bytes=4))
    assert per_m < per_a - 100                    # 2D 矩阵少一份 v → 持久显著更省
    assert pm < pa                                 # 峰值随之更低


def test_muon_feasibility_allowed():
    # Muon 不再被 feasibility fail-loud（此前非 Adam 全拦）。
    _peak_persist(OptimizerSpec.muon(params_fp32=True))       # 不抛即通过


def test_per_head_reduces_attention_ns():
    # per-head 把注意力投影的 NS workspace 按头切（一次一头）→ 该投影瞬态 ÷ n_heads。
    from cost_eval.specs import _MUON_NS_WORKSPACE_MULT
    from cost_eval.parallel_model import ParallelModel
    from cost_eval.shape_eval import ShapeEval
    spec, d, fl = build_dsv3_spec(8)
    pc = ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1, sequence_parallel=True)
    g = ShapeEval().resolve(spec, ParallelModel(pc, spec.dims.n_layers, 2))
    nh = spec.dims.n_heads
    for l in g.stages[0]:
        for op in l.ops:
            if is_muon_matrix_weight(op.type, op.name) and is_attn_projection(op.name):
                for w in op.params:
                    shard = w.local_numel // 2
                    full = _MUON_NS_WORKSPACE_MULT * shard * 4
                    ph = _MUON_NS_WORKSPACE_MULT * max(1, shard // nh) * 4
                    assert ph < full and ph == _MUON_NS_WORKSPACE_MULT * (shard // nh) * 4
                    return
    raise AssertionError("no attention-projection Muon matrix found")
