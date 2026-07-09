"""B 标定 margin（2026-07-09）：select 重算下**保留-MoE**层 loss 峰的 fp32-cast + 小张量长尾
（源码级 op-DAG 提取证实其在 op 图粒度之下，`analysis/realmachine/opdag_validation.md`）→ 明示为标定常数
`kept_frag_factor`（DSv3 preset=1.9，自真机 select_attn 标定）。仅 select-kept-MoE 生效；full /
no-recompute / select-keep-attn 不触发（锚点不破）。真机：select_attn(keepFFN)=18828。
"""
from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

GiB = 2 ** 30
MiB = 2 ** 20
ATTN = {lid: {"linear_q", "linear_kv", "q_a", "kv_a", "rope", "flash", "o_proj"} for lid in range(1, 9)}
MLP = {lid: {"fc", "swiglu", "gelu", "router", "dispatch", "e_", "combine", "shared"} for lid in range(1, 9)}


def _peak(rc, *, N=8, factor=None):
    spec, d, fl = build_dsv3_spec(N)   # preset 已带 kept_frag_factor=1.9
    d.B = 1
    if factor is not None:
        d.kept_frag_factor = factor
    pc = ParallelConfig(dp_shard=2, cp=1, tp=1, pp=1, sequence_parallel=True, num_microbatches=1)
    ev = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                   HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0), rc, SwapSpec())
    return ev.evaluate().per_stage[0].peak_bytes / MiB


def test_select_attn_keepFFN_margin_closes_residual():
    # 靶心：真机 18828。无 margin 15488（0.823 欠预测）；1.9 标定 margin → ~18844（1.001，OOM-安全）。
    with_m = _peak(RecomputeSpec("select", select_ops=ATTN))
    assert 18000 <= with_m <= 19500, with_m          # ≥0.95 且 OOM-安全（≥真机）
    assert with_m / 18828 >= 0.95


def test_margin_off_reproduces_pre_fix_underprediction():
    # factor=0（关）→ 复现修前 15488（证明 margin 是唯一变量、可关）。
    off = _peak(RecomputeSpec("select", select_ops=ATTN), factor=0.0)
    assert abs(off - 15487.5) < 1.0, off


def test_select_mlp_keepattn_unchanged_moe_recomputed():
    # keep-attn（重算 FFN → MoE 被重算）：margin gate 关 → 与 factor=0 逐字节相同（不被推过头）。
    on = _peak(RecomputeSpec("select", select_ops=MLP))
    off = _peak(RecomputeSpec("select", select_ops=MLP), factor=0.0)
    assert abs(on - off) < 1e-6, (on, off)
    assert abs(off - 14821.6) < 1.0, off             # 保持 0.940 准确


def test_full_recompute_hard_gate_unbroken():
    # full 重算：kept-MoE=0 → margin 0 → DSv3 4L 硬门 12409.5 逐字节不破。
    p = _peak(RecomputeSpec("full", full_layers={1, 2, 3, 4}), N=4)
    assert abs(p - 12409.5) < 0.05, p
