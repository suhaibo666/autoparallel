from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)

def test_optimizer_adamw_bytes():
    # 持久 = param+opt（剔 grad）：bf16 params=14, fp32 params=12
    assert OptimizerSpec.adamw().state_bytes_per_param == 14
    assert OptimizerSpec.adamw(params_fp32=True).state_bytes_per_param == 12

def test_parallelconfig_defaults_single():
    pc = ParallelConfig()
    assert pc.tp == 1 and pc.dp_shard == 1 and pc.reshard_after_forward == "default"

def test_recompute_full_layers():
    rc = RecomputeSpec(mode="full", full_layers={0, 1})
    assert rc.is_full(0) and not rc.is_full(2)
