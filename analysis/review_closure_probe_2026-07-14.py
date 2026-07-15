"""Executable semantic probes for the 2026-07-14 review-response closure audit.

Unlike the regression suite, this script also prints currently accepted counterexamples.  It is
diagnostic evidence, not a passing unit-test contract.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import runpy
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cost_eval.build_llm import build_llm_spec
from cost_eval.configs.from_mindformers import from_mindformers_dict
from cost_eval.parallel_model import ParallelModel
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.report import Evaluator
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import HardwareSpec, OptimizerSpec, ParallelConfig, RecomputeSpec, SwapSpec
from cost_eval.structure_mem import estimate_structure_memory
from serve_explorer import _bundle_to_fields


GIB = 2**30
MIB = 2**20


def evaluate(spec, pc, *, max_memory=64 * GIB, timeline=False):
    return Evaluator(
        spec,
        pc,
        OptimizerSpec.adamw(params_fp32=True),
        HardwareSpec(max_device_memory=max_memory),
        RecomputeSpec(),
        SwapSpec(),
    ).evaluate(record_timeline=timeline)


def resolve(spec, pc):
    world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
    return ShapeEval().resolve(spec, ParallelModel(pc, spec.dims.n_layers, world_size=world))


def adapter_probes():
    fixtures = runpy.run_path("tests/test_from_mindformers.py")
    dsv3_mf = fixtures["_dsv3_mf"]
    dsv4_mf = fixtures["_dsv4align_mf"]

    base = dsv3_mf()
    invalid_reshard = copy.deepcopy(base)
    invalid_reshard["parallelism"]["reshard_after_forward_policy"] = "nevver"

    training_typo = copy.deepcopy(base)
    training_typo["training"].pop("local_batch_size", None)
    training_typo["training"]["local_batch_szie"] = 4

    context_typo = copy.deepcopy(base)
    context_typo["context"].pop("max_device_memory", None)
    context_typo["context"]["max_device_memry"] = "1GB"

    swap_typo = copy.deepcopy(base)
    swap_typo["swap"] = {"enablee": True}

    qk = copy.deepcopy(base)
    qk["model"]["qk_layernorm"] = True

    shared_gate = copy.deepcopy(base)
    shared_gate["model"]["use_shared_expert_gating"] = True

    dropout = copy.deepcopy(base)
    dropout["model"]["attention_dropout"] = 0.1

    fused = dsv4_mf()
    unfused = copy.deepcopy(fused)
    unfused["model"]["apply_dsa_kernel_fusion"] = False
    unfused["model"]["force_unfused_dsa"] = True

    tp1_bundle = from_mindformers_dict(base)
    tp2_mf = copy.deepcopy(base)
    tp2_mf["parallelism"]["tensor_parallel"] = 2
    tp2_mf["parallelism"]["sequence_parallel"] = True
    tp2_bundle = from_mindformers_dict(tp2_mf)
    tp1_graph = resolve(build_llm_spec(tp1_bundle.llm), tp1_bundle.parallel)
    tp2_graph = resolve(build_llm_spec(tp2_bundle.llm), tp2_bundle.parallel)

    def loss_bytes(graph):
        lm_head = next(op for layer in graph.stages[0] for op in layer.ops if op.name == "lm_head")
        nll = next(op for layer in graph.stages[0] for op in layer.ops if op.name == "nll")
        return lm_head.output.local_numel * lm_head.output.dtype_bytes, nll.bwd_scratch_bytes

    logits1, scratch1 = loss_bytes(tp1_graph)
    logits2, scratch2 = loss_bytes(tp2_graph)

    ui_mf = copy.deepcopy(base)
    ui_mf["parallelism"]["reshard_after_forward_policy"] = "never"
    ui_mf["parallelism"]["cpu_offload"] = True
    ui_fields = _bundle_to_fields(from_mindformers_dict(ui_mf))

    independent_dsa = copy.deepcopy(base)
    independent_dsa["model"].update(
        experimental_attention_variant="dsa",
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
    )

    return {
        "invalid_reshard_accepted": from_mindformers_dict(invalid_reshard).parallel.reshard_after_forward,
        "training_typo_batch_defaulted": from_mindformers_dict(training_typo).llm.batch_size,
        "context_typo_capacity_GiB": from_mindformers_dict(context_typo).hardware.max_device_memory / GIB,
        "swap_typo_enable_defaulted": from_mindformers_dict(swap_typo).swap.enable,
        "qk_layernorm_true_became": from_mindformers_dict(qk).llm.qk_layernorm,
        "shared_expert_gate": (
            "rejected_not_modeled"
            if _raises_not_implemented(shared_gate)
            else "silently_accepted"
        ),
        "dropout_0.1_same_llm_as_0.0": (
            from_mindformers_dict(dropout).llm == from_mindformers_dict(base).llm
        ),
        "dsa_fusion_still_controls_ce_fusion": {
            "dsa_fused": from_mindformers_dict(fused).llm.cross_entropy_fused,
            "dsa_unfused": from_mindformers_dict(unfused).llm.cross_entropy_fused,
        },
        "tp_vocab_loss": {
            "loss_type_tp2": tp2_bundle.llm.loss_type,
            "logits_tp2_is_half": logits2 * 2 == logits1,
            "bwd_scratch_tp2_matches_local_vp_grad": (
                scratch2
                == 4
                * tp2_bundle.llm.seq_length
                * tp2_bundle.llm.batch_size
                * tp2_bundle.llm.vocab_size
                // tp2_bundle.parallel.tp
            ),
            "bwd_scratch_MiB": {"tp1_unfused_nll": scratch1 / MIB, "tp2_vocab_ce": scratch2 / MIB},
        },
        "independent_dsa_adapter_and_build": (
            build_llm_spec(from_mindformers_dict(independent_dsa).llm) is not None
            and from_mindformers_dict(independent_dsa).llm.attn_type
        ),
        "ui_roundtrip_missing_fields": sorted(
            {"reshard_after_forward", "cpu_offload", "dp_replicate", "prefetch_depth"}
            - set(ui_fields)
        ),
    }


def _raises_not_implemented(mf):
    try:
        from_mindformers_dict(mf)
    except NotImplementedError:
        return True
    return False


def timeline_and_validation_probes():
    spec = build_llm_spec(deepseek_v3(4))

    default = evaluate(spec, ParallelConfig(dp_shard=2), timeline=True)
    always = evaluate(
        spec, ParallelConfig(dp_shard=2, reshard_after_forward="always"), timeline=True
    )
    never = evaluate(
        spec, ParallelConfig(dp_shard=2, reshard_after_forward="never"), timeline=True
    )
    typo = evaluate(
        spec, ParallelConfig(dp_shard=2, reshard_after_forward="nevver"), timeline=True
    )

    def gather_signature(report):
        return [s.breakdown.gather_buf for s in report.per_stage[0].timeline]

    opt = next(s for s in default.per_stage[0].timeline if s.event == "optstep")
    graph = resolve(spec, ParallelConfig(dp_shard=2))
    expected_grad = sum(
        estimate_structure_memory(layer.ops, fsdp=2, efsdp=2).grad_shard_bytes
        for layer in graph.stages[0]
    )

    vpp = evaluate(
        spec,
        ParallelConfig(pp=2, interleave=2, num_microbatches=4),
        timeline=True,
    )
    samples = vpp.per_stage[0].timeline
    duplicate_event_mb = [
        [event, mb, count]
        for (event, mb), count in Counter((s.event, s.mb) for s in samples).items()
        if count > 1
    ]

    direct_tp_without_sp = "accepted"
    try:
        evaluate(spec, ParallelConfig(tp=2, sequence_parallel=False))
    except Exception as exc:  # pragma: no cover - diagnostic branch
        direct_tp_without_sp = type(exc).__name__

    none_report = evaluate(spec, ParallelConfig(), timeline=True)
    select_miss = Evaluator(
        spec,
        ParallelConfig(),
        OptimizerSpec.adamw(params_fp32=True),
        HardwareSpec(max_device_memory=64 * GIB),
        RecomputeSpec(mode="select", select_ops={1: {"definitely_missing"}}),
        SwapSpec(),
    ).evaluate(record_timeline=True)
    full_empty = Evaluator(
        spec,
        ParallelConfig(),
        OptimizerSpec.adamw(params_fp32=True),
        HardwareSpec(max_device_memory=64 * GIB),
        RecomputeSpec(mode="full", full_layers=set()),
        SwapSpec(),
    ).evaluate(record_timeline=True)

    non_adam = "accepted_with_adam_optstep"
    try:
        report = Evaluator(
            spec,
            ParallelConfig(),
            OptimizerSpec(type="SGD", state_bytes_per_param=4, grad_dtype_bytes=4),
            HardwareSpec(max_device_memory=64 * GIB),
            RecomputeSpec(),
            SwapSpec(),
        ).evaluate(record_timeline=True)
        if not any(s.event == "optstep" for s in report.per_stage[0].timeline):
            non_adam = "accepted_without_optstep"
    except Exception as exc:  # pragma: no cover - diagnostic branch
        non_adam = type(exc).__name__

    bad_topk = dataclasses.replace(deepseek_v3(4), moe_router_topk=-1)
    bad_capacity = dataclasses.replace(deepseek_v3(4), moe_capacity_factor=0.0)
    topk_graph = resolve(build_llm_spec(bad_topk), ParallelConfig())
    capacity_graph = resolve(build_llm_spec(bad_capacity), ParallelConfig())
    min_topk_numel = min(
        t.local_numel
        for layers in topk_graph.stages.values()
        for layer in layers
        for op in layer.ops
        for t in (*op.inputs, op.output, *op.saves)
    )
    zero_capacity_numel = any(
        t.local_numel == 0
        for layers in capacity_graph.stages.values()
        for layer in layers
        for op in layer.ops
        for t in (*op.inputs, op.output, *op.saves)
    )

    o_groups_zero = "accepted"
    try:
        resolve(build_llm_spec(dataclasses.replace(deepseek_v4(4), o_groups=0)), ParallelConfig())
    except Exception as exc:
        o_groups_zero = type(exc).__name__

    return {
        "grad_accum_MiB": opt.breakdown.grad_accum / MIB,
        "grad_accum_exact_formula": opt.breakdown.grad_accum == expected_grad,
        "reshard_signatures": {
            "always_never_differ": gather_signature(always) != gather_signature(never),
            "invalid_value_equals_default": gather_signature(typo) == gather_signature(default),
        },
        "vpp_duplicate_event_mb_without_chunk": duplicate_event_mb[:8],
        "timeline_sample_has_chunk_field": hasattr(samples[0], "chunk"),
        "direct_tp2_sp_false": direct_tp_without_sp,
        "direct_select_zero_hit_equals_none": (
            [s.total_bytes for s in select_miss.per_stage[0].timeline]
            == [s.total_bytes for s in none_report.per_stage[0].timeline]
        ),
        "direct_full_empty_equals_none": (
            [s.total_bytes for s in full_empty.per_stage[0].timeline]
            == [s.total_bytes for s in none_report.per_stage[0].timeline]
        ),
        "direct_non_adam_optimizer": non_adam,
        "negative_topk_min_resolved_numel": min_topk_numel,
        "zero_capacity_produces_zero_tensor": zero_capacity_numel,
        "dsv4_o_groups_zero": o_groups_zero,
    }


def attention_and_report_probes():
    spec = build_llm_spec(deepseek_v3(4))

    def flash_values(tp):
        graph = resolve(spec, ParallelConfig(tp=tp, sequence_parallel=(tp > 1)))
        flash = next(
            op
            for layer in graph.stages[0]
            for op in layer.ops
            if op.name == "flash"
        )
        stats = next(s for s in flash.saves if s.name == "fa_stats")
        return {"workspace": flash.workspace_bytes, "fa_stats": stats.local_numel * stats.dtype_bytes}

    tp1 = flash_values(1)
    tp2 = flash_values(2)

    dsa_cfg = dataclasses.replace(
        deepseek_v3(1),
        attn_type="dsa",
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
    )
    dsa_graph = resolve(build_llm_spec(dsa_cfg), ParallelConfig(tp=2, cp=2, sequence_parallel=True))
    kl = next(
        op
        for layer in dsa_graph.stages[0]
        for op in layer.ops
        if op.name == "idx_kl_loss"
    )
    expected_kl = (
        4
        * dsa_cfg.batch_size
        * dsa_cfg.num_attention_heads
        * dsa_cfg.seq_length
        * dsa_cfg.seq_length
        // (2 * 2)
    )

    high = evaluate(spec, ParallelConfig(tp=2, sequence_parallel=True))
    peak = high.per_stage[0].peak_bytes
    threshold = peak + high.hccl_reserved_bytes // 2
    split = evaluate(
        spec,
        ParallelConfig(tp=2, sequence_parallel=True),
        max_memory=threshold,
    )

    return {
        "fa_stats_tp2_is_half": tp2["fa_stats"] * 2 == tp1["fa_stats"],
        "fa_workspace_tp2_is_half": tp2["workspace"] * 2 == tp1["workspace"],
        "fa_bytes": {"tp1": tp1, "tp2": tp2},
        "dsa_kl_workspace_exact_tp2_cp2": kl.workspace_bytes == expected_kl,
        "dsa_kl_workspace_MiB": kl.workspace_bytes / MIB,
        "allocated_oom_false_while_reserved_exceeds": (
            not split.oom and split.reserved_estimate_bytes(0) > threshold
        ),
    }


def main():
    result = {
        "adapter": adapter_probes(),
        "timeline_validation": timeline_and_validation_probes(),
        "attention_report": attention_and_report_probes(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
