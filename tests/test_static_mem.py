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


def test_indivisible_fsdp_replicates_whole_param():
    """切分不整除 → **整参 replicate_params（全量驻留）**——runtime 口径（P0-3 重写,2026-07-23）。

    mindformers 377c9c344 `parallelize.py:331-350`：FSDP 按 TP/EP 后参数**首维**判定
    `shape[0] % shard_size != 0` → 整参不做 FSDP（replicate_params,每卡全量;:353-378 特殊小参数
    亦显式复制）。旧「FSDP2 flat-param 补齐 ceil」假设（2026-07-21 引入）被 runtime 源码推翻——
    ceil 对整参复制可低估 divisor 倍,并非 OOM 安全（audit 报告 §6 数值例:P=64,K=256:ceil→1 vs
    runtime→64）。整除参数仍 numel//divisor（与旧 ceil 相等 → 锚点不动）。
    """
    Dsmall = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=1)
    spec = ModelSpec("t", Dsmall, ["dense"], {"dense": build_dense_decoder(Dsmall)})
    pm3 = ParallelModel(ParallelConfig(dp_shard=3), n_layers=1, world_size=3)
    out = StaticMem().compute(ShapeEval().resolve(spec, pm3), OptimizerSpec.adamw(), pm3)  # 不 raise
    pm1 = ParallelModel(ParallelConfig(dp_shard=1), n_layers=1, world_size=1)
    full = StaticMem().compute(ShapeEval().resolve(spec, pm1), OptimizerSpec.adamw(), pm1)
    # 该 toy 全部权重首维(H=8/F=16/2F=32/vocab=10/gamma H=8)均不被 3 整除 → 全部整参复制
    # → 分片配置下持久 == 不分片(每卡全量)。这正是 runtime replicate_params 的可观测语义。
    assert out[0] == full[0]

    # 对照:可整除 divisor(2)→ 正常 shard,严格小于全量(且 == 全量的一半,块对齐默认 1)。
    pm2 = ParallelModel(ParallelConfig(dp_shard=2), n_layers=1, world_size=2)
    half = StaticMem().compute(ShapeEval().resolve(spec, pm2), OptimizerSpec.adamw(), pm2)
    assert 0 < half[0] < full[0]
