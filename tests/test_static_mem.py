"""M5 static_mem 测试：守恒、退化（单卡=全局）、cpu_offload 归零、切分不整除报错。"""
import pytest
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.specs import ParallelConfig, OptimizerSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval, eval_expr
from cost_eval.static_mem import StaticMem

D = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)


def _spec():
    return ModelSpec("toy", D, ["dense", "dense"], {"dense": build_dense_decoder(D)})


def _global_param_numel():
    """2 层 dense 参数量（无切分的符号积，视为全局 numel）。"""
    from math import prod
    layer = build_dense_decoder(D)
    return 2 * sum(
        prod(eval_expr(e, D) for e in w.shape)
        for op in layer.ops
        for w in op.params
    )


def test_single_device_equals_global_times_bytes():
    """单卡（无任何并行）：持久态 = 全局 param numel × state_bytes_per_param。"""
    pm = ParallelModel(ParallelConfig(), n_layers=2, world_size=1)
    g = ShapeEval().resolve(_spec(), pm)
    out = StaticMem().compute(g, OptimizerSpec.adamw(), pm, cpu_offload=False)
    assert out[0] == _global_param_numel() * 14   # 持久 b_state=14（param+opt，剔 grad）


def test_conservation_under_sharding():
    """tp=2, dp_shard=2：per_dev_numel × (tp × fsdp) = 全局 numel（守恒）。"""
    pm = ParallelModel(ParallelConfig(tp=2, dp_shard=2), n_layers=2, world_size=4)
    g = ShapeEval().resolve(_spec(), pm)
    out = StaticMem().compute(g, OptimizerSpec.adamw(), pm, cpu_offload=False)
    per_dev_numel = out[0] // 14       # undo state_bytes_per_param（持久 14）
    # P1-01 后守恒式分两族：matmul 权重按 tp·fsdp 全切；norm gamma(H,) 在 tp 内**复制**
    # （真机语义：norm 独立 wrap 只随 fsdp 切），只 ÷fsdp → 重构全局时 gamma 族不乘 tp。
    gammas = 2 * 2 * D.H                                # 2 层 × (ln1_g + ln2_g)
    sharded = _global_param_numel() - gammas
    assert per_dev_numel == sharded // (2 * 2) + gammas // 2   # tp·fsdp=4；gamma 仅 ÷fsdp=2


def test_cpu_offload_zeroes_persistent():
    """cpu_offload=True：每 stage 持久态 = 0（参数/优化器态卸载到 CPU）。"""
    pm = ParallelModel(ParallelConfig(), n_layers=2, world_size=1)
    g = ShapeEval().resolve(_spec(), pm)
    out = StaticMem().compute(g, OptimizerSpec.adamw(), pm, cpu_offload=True)
    assert out[0] == 0


def test_indivisible_fsdp_ceil_oom_safe():
    """切分不整除 → **FSDP2 flat-param 补齐**：每卡 `ceil(numel/divisor)`（OOM 安全，不再 fail-loud）。

    2026-07-21 现场 DSv4-Flash 修：`attn_sink=n_heads`、fsdp 大（如 256）等 tiny-param/不整除场景，
    真机 FSDP2 把展平参数 pad 到 world 倍数再切、每卡 ceil；ceil≥floor 不低估 → OOM 安全。整除时
    ceil==floor 逐字节不变（所有锚点均整除）。此前该场景 fail-loud（订正）。
    """
    Dsmall = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=1)
    spec = ModelSpec("t", Dsmall, ["dense"], {"dense": build_dense_decoder(Dsmall)})
    pm3 = ParallelModel(ParallelConfig(dp_shard=3), n_layers=1, world_size=3)
    out = StaticMem().compute(ShapeEval().resolve(spec, pm3), OptimizerSpec.adamw(), pm3)  # 不再 raise
    pm1 = ParallelModel(ParallelConfig(dp_shard=1), n_layers=1, world_size=1)
    full = StaticMem().compute(ShapeEval().resolve(spec, pm1), OptimizerSpec.adamw(), pm1)
    # 分片(ceil)后严格小于不分片，且 ≥ 精确 1/3（ceil 不低估 → OOM 安全）。
    assert 0 < out[0] < full[0]
    assert out[0] >= full[0] // 3
