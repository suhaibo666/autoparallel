from cost_eval.presets import deepseek_v3
from cost_eval.build_llm import build_llm_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator
from validate_dsv3 import build_dsv3_spec, RESIDUAL_MiB
MiB, GiB = 2**20, 2**30
def _peak(spec, N):
    full = set(range(1, N+1))
    # 默认 depth=1（FSDP2 参数预取双缓冲）+ 拆解后的 RESIDUAL_MiB(=177−ΔP)；总峰值不变
    ev = Evaluator(spec, ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1, sequence_parallel=True),
                   OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                   HardwareSpec(max_device_memory=59*GiB, framework_reserve=RESIDUAL_MiB*MiB),
                   RecomputeSpec(mode="full", full_layers=full), SwapSpec())
    return ev.evaluate().per_stage[0]
# 真机实测锚点（max_memory_allocated MiB）。framework_reserve 经验常数已消除（RESIDUAL_MiB=0），
# 预测由「逐桶结构 + 分配器对齐公式」给出（无拟合 blob）→ 从逐字节命中改为**真机 ±1% 容差**。
# 预测现略低于真机（DSv3 4L 12409.5 vs 12473.1 = 0.995），差额是未建的 sub-block 临时量（文档化小残差）。
REAL_4L, REAL_8L = 12473.1, 13953.3
#: 2026-07-30 起的理论钉（`REAL_*` 一个字节未动；这是**新增**的理论值，不替换真机常数）
THEO_4L, THEO_8L = 13481.9, 14906.0


def test_preset_equals_oracle_and_anchor_4L():
    new = _peak(build_llm_spec(deepseek_v3(4)), 4)
    old = _peak(build_dsv3_spec(4)[0], 4)
    assert abs(new.peak_bytes - old.peak_bytes) < 1               # preset≡oracle（逐字节）
    # 2026-07-30：`lm_head` 反向 kernel workspace 入账（+1058.0 MiB @4L / +952.7 @8L）→
    #   理论 12423.9→13481.9（4L）、13848.0→14906.0（8L），对真机比值 0.996→1.081 /
    #   0.992→1.068 = **过读侧**（OOM 安全）。原「真机 ±1%」门因此不成立：
    #   **不放宽成 ±9% 掩盖**，改为钉住新理论值（±0.1 MiB）并把比值单列一条断言，
    #   任何进一步漂移仍必红。见 docs/head_loss_bwd_workspace_2026-07-30.md 6.4。
    assert abs(new.peak_bytes / MiB - THEO_4L) < 0.1, new.peak_bytes / MiB
    assert abs(new.peak_bytes / MiB / REAL_4L - 1.0809) < 0.001
def test_anchor_8L():
    new = _peak(build_llm_spec(deepseek_v3(8)), 8)
    assert abs(new.peak_bytes / MiB - THEO_8L) < 0.1, new.peak_bytes / MiB
    assert abs(new.peak_bytes / MiB / REAL_8L - 1.0683) < 0.001
