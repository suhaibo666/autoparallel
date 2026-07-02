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

def test_recompute_none_full_back_compat():
    # 位置参数 + is_full 语义保持不变（None/full 字节级不变的前提）
    assert RecomputeSpec("None").mode == "None"
    rc = RecomputeSpec("full", {0, 1})
    assert rc.is_full(0) and not rc.is_full(2)
    assert not rc.is_select(0)                    # full 不是 select
    assert rc.selectors(0) == set()

def test_recompute_select_mode_flags():
    # mode=select + 每层选择器集（忠实 mindformers select_module 反转后 {layer_id: [module]}）
    rc = RecomputeSpec(mode="select", select_ops={0: {"flash"}, 2: {"flash", "swiglu"}})
    assert rc.is_select(0) and rc.is_select(2)
    assert not rc.is_select(1)                    # 1 不在 select_ops
    assert not rc.is_full(0)                      # select 不是 full
    assert rc.selectors(0) == {"flash"}
    assert rc.selectors(1) == set()               # 缺省空集

def test_recompute_select_empty_selectors_is_not_select():
    # 空选择器集视作「该层不选」（退化为无重算）
    rc = RecomputeSpec(mode="select", select_ops={0: set()})
    assert not rc.is_select(0)

def test_recompute_op_matches_substring_name_or_type():
    # 忠实 mindformers exclude_op：算子名/类型子串（大小写不敏感）命中
    rc = RecomputeSpec(mode="select", select_ops={0: {"flash"}, 1: {"attn"}, 2: {"swiglu"}})
    assert rc.op_matches(0, "flash", "flash_attn")        # 名字命中
    assert not rc.op_matches(0, "o_proj", "matmul")       # 不命中
    assert rc.op_matches(1, "flash", "flash_attn")        # 'attn' 是 type 'flash_attn' 子串（Megatron core_attn）
    assert rc.op_matches(2, "swiglu", "elementwise")
    assert not rc.op_matches(3, "flash", "flash_attn")    # 3 无选择器
