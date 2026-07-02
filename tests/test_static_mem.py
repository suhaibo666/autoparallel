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
    layer = build_dense_decoder(D)
    return 2 * sum(
        eval_expr(w.shape[0], D) * eval_expr(w.shape[1], D)
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
    assert per_dev_numel * (2 * 2) == _global_param_numel()   # tp * fsdp = 4


def test_cpu_offload_zeroes_persistent():
    """cpu_offload=True：每 stage 持久态 = 0（参数/优化器态卸载到 CPU）。"""
    pm = ParallelModel(ParallelConfig(), n_layers=2, world_size=1)
    g = ShapeEval().resolve(_spec(), pm)
    out = StaticMem().compute(g, OptimizerSpec.adamw(), pm, cpu_offload=True)
    assert out[0] == 0


def test_indivisible_fsdp_raises():
    """切分不整除（权重 numel 不被 fsdp 整除）→ ValueError，而非静默 floor 低估（I1，OOM 安全）。

    D 下 o_w=n_heads·head_dim·H=8×8=64、fc1_w=256 等不被 fsdp=3 整除。与 resolve_tensor
    对 sharded 维不整除即 raise 的口径一致（shape_eval.py:59-63）。
    """
    Dsmall = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=1)
    spec = ModelSpec("t", Dsmall, ["dense"], {"dense": build_dense_decoder(Dsmall)})
    pm = ParallelModel(ParallelConfig(dp_shard=3), n_layers=1, world_size=3)
    g = ShapeEval().resolve(spec, pm)     # resolve 不会 raise（权重仅 tp=1 切）
    with pytest.raises(ValueError):
        StaticMem().compute(g, OptimizerSpec.adamw(), pm, cpu_offload=False)
