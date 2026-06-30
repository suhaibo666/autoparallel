from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)

def test_optimizer_adamw_bytes():
    assert OptimizerSpec.adamw().state_bytes_per_param == 16

def test_parallelconfig_defaults_single():
    pc = ParallelConfig()
    assert pc.tp == 1 and pc.dp_shard == 1 and pc.reshard_after_forward == "default"

def test_recompute_full_layers():
    rc = RecomputeSpec(mode="full", full_layers={0, 1})
    assert rc.is_full(0) and not rc.is_full(2)
