"""Executable counterexamples for the 2026-07-14 memory-model review.

These are strict expected failures: the normal suite records known gaps as
XFAIL, while ``pytest --runxfail`` exposes the concrete current values.
"""
from __future__ import annotations

from collections import defaultdict

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.configs.from_mindformers import _build_parallel
from cost_eval.parallel_model import ParallelModel
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.report import Evaluator
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import HardwareSpec, OptimizerSpec, ParallelConfig, RecomputeSpec, SwapSpec


GIB = 2**30


def _evaluate(spec, pc: ParallelConfig, *, timeline: bool = False):
    evaluator = Evaluator(
        spec,
        pc,
        OptimizerSpec.adamw(params_fp32=True),
        HardwareSpec(max_device_memory=64 * GIB),
        RecomputeSpec(),
        SwapSpec(),
    )
    return evaluator.evaluate(record_timeline=timeline)


@pytest.mark.xfail(strict=True, reason="MindFormers parallel fields are silently replaced by defaults")
def test_parallel_adapter_preserves_memory_semantic_fields():
    mf = {
        "training": {"local_batch_size": 1, "global_batch_size": 2},
        "parallelism": {
            "tensor_parallel": 2,
            "context_parallel": 2,
            "pipeline_parallel": 2,
            "pipeline_parallel_microbatch_size": 2,
            "context_parallel_method": "ulysses",
            "pipeline_parallel_interleave_num": 4,
            "reshard_after_forward_policy": "never",
        },
    }

    pc = _build_parallel(mf, mtp=0, num_layers=4)

    assert (pc.context_parallel_method, pc.interleave, pc.reshard_after_forward) == (
        "ulysses",
        4,
        "never",
    )


def test_reshard_policy_changes_the_gather_lifetime():
    # P0-03 已修（2026-07-14）：always/never 的 gather 生命周期不同——never 下 unsharded 权重
    # 从 fwd 驻留到本模块 post_backward（hyper_parallel hsdp_scheduler.py:225-250），xfail 移除。
    spec = build_llm_spec(deepseek_v3(4))
    always = _evaluate(spec, ParallelConfig(dp_shard=2, reshard_after_forward="always"), timeline=True)
    never = _evaluate(spec, ParallelConfig(dp_shard=2, reshard_after_forward="never"), timeline=True)

    always_signature = [
        (point.event, point.total_bytes, point.breakdown.gather_buf)
        for point in always.per_stage[0].timeline
    ]
    never_signature = [
        (point.event, point.total_bytes, point.breakdown.gather_buf)
        for point in never.per_stage[0].timeline
    ]
    assert always_signature != never_signature


def test_every_moe_router_owns_its_weight_parameter():
    # P1-01 已修（2026-07-14）：router 权重 [n_experts,H] fp32 入图（真机 router 因 fp32 精度
    # 单独 FSDP wrap，parallelize.py:1116-1128；optimizer 参数表含 router.weight）。xfail 移除。
    spec = build_llm_spec(deepseek_v3(4))
    routers = [
        op
        for layer in spec.layer_specs.values()
        for op in layer.ops
        if op.name == "router"
    ]

    assert routers
    assert all(op.params for op in routers), [len(op.params) for op in routers]


def test_tp_halves_embedding_and_lm_head_local_parameters():
    # P0-04 已修（2026-07-14）：pynative TP>1 无条件 vocab-parallel——embedding RowwiseParallel
    # Shard(0)（parallelize.py:751-759）、lm_head ColwiseParallel Shard(0) gather_output=False
    # （:765-767），每卡 local 参数 = 全量/tp。xfail 移除。
    spec = build_llm_spec(deepseek_v3(4))

    def local_numel(tp: int) -> dict[str, int]:
        pc = ParallelConfig(tp=tp)
        graph = ShapeEval().resolve(spec, ParallelModel(pc, spec.dims.n_layers, world_size=tp))
        return {
            op.name: sum(weight.local_numel for weight in op.params)
            for layer in graph.stages[0]
            for op in layer.ops
            if op.name in {"embedding", "lm_head"}
        }

    tp1 = local_numel(1)
    tp2 = local_numel(2)
    assert tp2 == {name: numel // 2 for name, numel in tp1.items()}


def test_tensor_name_has_one_shape_within_each_resolved_layer():
    # P1-03 已修（2026-07-14）：mHC 放大承载张量重命名 {name}_xn + ShapeEval.resolve 同名异
    # numel fail-loud 不变量。xfail 移除。
    spec = build_llm_spec(deepseek_v4(4))
    graph = ShapeEval().resolve(
        spec,
        ParallelModel(ParallelConfig(), spec.dims.n_layers, world_size=1),
    )
    collisions = {}

    for layers in graph.stages.values():
        for layer in layers:
            sizes = defaultdict(set)
            for op in layer.ops:
                for tensor in (*op.inputs, op.output, *op.params, *op.saves):
                    sizes[tensor.name].add(tensor.local_numel)
            bad = {name: sorted(values) for name, values in sizes.items() if len(values) > 1}
            if bad:
                collisions[(layer.layer_id, layer.layer_type)] = bad

    assert not collisions, collisions


def test_optimizer_event_retains_accumulated_gradient_buffers():
    # P0-01 已修（2026-07-14）：累计梯度桶 grad_accum（step-scoped cumulative，NPU 探针
    # 1889.5 MiB 证实）——optimizer 事件时刻全部 reduced grad shard 与 optstep 瞬态共存，
    # xfail 移除。断言从 grad_buf（每层 full 瞬态，另一语义）改为 grad_accum。
    spec = build_llm_spec(deepseek_v3(4))
    report = _evaluate(spec, ParallelConfig(dp_shard=2), timeline=True)
    optstep = [point for point in report.per_stage[0].timeline if point.event == "optstep"]

    assert len(optstep) == 1
    assert optstep[0].breakdown.grad_accum > 0
