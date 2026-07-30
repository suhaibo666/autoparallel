from cost_eval.presets import deepseek_v3
from cost_eval.build_llm import build_llm_spec
from validate_dsv3 import build_dsv3_spec, RESIDUAL_MiB
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator
MiB, GiB = 2**20, 2**30
def _peak(spec, N):
    full=set(range(1,N+1))
    # 默认 depth=1 预取双缓冲 + 拆解后的 RESIDUAL_MiB(=177−ΔP)
    ev=Evaluator(spec, ParallelConfig(dp_shard=2,tp=1,ep=1,pp=1,cp=1,sequence_parallel=True),
        OptimizerSpec.adamw(params_fp32=True,grad_dtype_bytes=4),
        HardwareSpec(max_device_memory=59*GiB,framework_reserve=RESIDUAL_MiB*MiB),
        RecomputeSpec(mode="full",full_layers=full),SwapSpec())
    return ev.evaluate().per_stage[0].peak_bytes
def test_build_dsv3_spec_now_uses_preset():
    # build_dsv3_spec keeps its signature but internally routes through build_llm_spec(deepseek_v3(N))
    for N in (4,8):
        assert _peak(build_dsv3_spec(N)[0], N) == _peak(build_llm_spec(deepseek_v3(N)), N)
def test_anchors_unchanged():
    # 真机 12473.1 / 13953.3。2026-07-30：`lm_head` 反向 kernel workspace 入账 →
    #   理论 13481.9 / 14906.0（比值 1.081 / 1.068，过读侧 = OOM 安全）。**不放宽 ±1%
    #   成 ±9% 掩盖**，改钉理论值；见 docs/head_loss_bwd_workspace_2026-07-30.md 6.4。
    assert abs(_peak(build_dsv3_spec(4)[0], 4) / MiB - 13481.9) < 0.1
    assert abs(_peak(build_dsv3_spec(8)[0], 8) / MiB - 14906.0) < 0.1
