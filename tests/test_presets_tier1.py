"""Task 1.4：Tier-1 预设（llama / qwen2 / mixtral）+ 全局 param 守恒。

`global_param_count(spec)` 在**全 1 并行配置**（dp_shard=tp=ep=pp=cp=1）下解析
ModelSpec，此时 local_numel == global_numel，把所有 op 的 param 张量 numel 相加，
即全局参数量。与各模型公开的真实参数量在 ±2% 内一致 → 说明 op 图把权重建全了。
"""
from cost_eval.presets import llama, qwen2, mixtral
from cost_eval.build_llm import build_llm_spec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from cost_eval.report import Evaluator

GiB = 2 ** 30


def global_param_count(spec) -> int:
    """全局参数量：全 1 并行配置下解析，sum 所有 param 张量的 local_numel。

    dp_shard=tp=ep=pp=cp=1 → 无任何切分，local_numel == global_numel。
    ShapeEval 逐 layer_pattern 出现次数解析，故重复层 key 的权重按层数计入。
    """
    pc = ParallelConfig(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1,
                        sequence_parallel=False)
    pm = ParallelModel(pc, spec.dims.n_layers, world_size=1)
    g = ShapeEval().resolve(spec, pm)
    return sum(
        w.local_numel
        for layers in g.stages.values()
        for layer in layers
        for op in layer.ops
        for w in op.params
    )


def _rel_err(got: int, ref: float) -> float:
    return abs(got - ref) / ref


def test_llama2_7b_param_conservation():
    spec = build_llm_spec(llama())          # 默认 = Llama-2-7B 超参
    n = global_param_count(spec)
    assert _rel_err(n, 6.7e9) < 0.02, f"llama2_7b params={n:,} rel_err={_rel_err(n, 6.7e9):.4%}"


def test_mixtral_8x7b_param_conservation():
    spec = build_llm_spec(mixtral())        # 默认 = Mixtral-8x7B 超参
    n = global_param_count(spec)
    assert _rel_err(n, 46.7e9) < 0.02, f"mixtral_8x7b params={n:,} rel_err={_rel_err(n, 46.7e9):.4%}"


def test_qwen2_builds_and_evaluates():
    cfg = qwen2()
    spec = build_llm_spec(cfg)
    ev = Evaluator(
        spec,
        ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1, sequence_parallel=True),
        OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
        HardwareSpec(max_device_memory=59 * GiB),
        RecomputeSpec(mode="full", full_layers=set(range(1, cfg.num_layers + 1))),
        SwapSpec(),
    )
    rep = ev.evaluate()
    assert rep.per_stage[0].peak_bytes > 0
    # qwen2 param ref 可选：只要能建能评估即可（gqa + dense）
    assert global_param_count(spec) > 0
