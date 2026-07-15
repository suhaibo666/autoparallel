"""闭环审计 v3（closure_audit_verification_2026-07-15.md §4.7，P1-16）反例转正式回归。

复核实证：`ParallelConfig(interleave=0)` / `ParallelConfig(prefetch_depth=-1)` /
`ParallelConfig(num_microbatches=0)` 均被接受并返回峰值（调度中退化/空循环/无意义峰）。
在 `ParallelConfig.__post_init__`（已有 cp_method/reshard 校验）补标量正值校验，报错含字段名与值。
关键守卫：默认构造 `ParallelConfig()`（全 1）与现有测试里的合法值必须继续通过。
"""
import pytest

from cost_eval.specs import ParallelConfig


# ── 反例：非法标量必须 fail-loud（此前静默接受）─────────────────────────────────
def test_interleave_zero_rejected():
    with pytest.raises(ValueError, match="interleave"):
        ParallelConfig(interleave=0)


def test_interleave_negative_rejected():
    with pytest.raises(ValueError, match="interleave"):
        ParallelConfig(interleave=-2)


def test_prefetch_depth_negative_rejected():
    with pytest.raises(ValueError, match="prefetch_depth"):
        ParallelConfig(prefetch_depth=-1)


def test_num_microbatches_zero_rejected():
    with pytest.raises(ValueError, match="num_microbatches"):
        ParallelConfig(num_microbatches=0)


def test_microbatch_zero_rejected():
    with pytest.raises(ValueError, match="microbatch"):
        ParallelConfig(microbatch=0)


@pytest.mark.parametrize("field", ["dp_replicate", "dp_shard", "cp", "tp", "pp", "ep"])
def test_parallel_degree_zero_rejected(field):
    with pytest.raises(ValueError, match=field):
        ParallelConfig(**{field: 0})


def test_error_message_carries_field_and_value():
    with pytest.raises(ValueError) as ei:
        ParallelConfig(tp=-3)
    msg = str(ei.value)
    assert "tp" in msg and "-3" in msg


# ── 合法用法守卫：默认构造与现有合法值必须继续通过 ─────────────────────────────
def test_default_construction_passes():
    ParallelConfig()   # 全 1，必须通过


def test_prefetch_depth_zero_allowed():
    ParallelConfig(prefetch_depth=0)   # 0 = 无预取，合法


def test_typical_vpp_config_passes():
    ParallelConfig(dp_shard=2, tp=1, ep=1, pp=2, cp=1, sequence_parallel=True,
                   interleave=2, num_microbatches=4)


def test_representative_existing_configs_pass():
    # 覆盖既有回归里出现过的合法组合（防误拦）。
    ParallelConfig(tp=8, dp_shard=8, cp=1, ep=4, pp=1, sequence_parallel=True)
    ParallelConfig(dp_shard=1, cp=2, tp=1, pp=1, sequence_parallel=False,
                   num_microbatches=1, context_parallel_method="colossal")
    ParallelConfig(pp=3, layers_per_stage=[3, 3, 2])
    ParallelConfig(prefetch_depth=1)
