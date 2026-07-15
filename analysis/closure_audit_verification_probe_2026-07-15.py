"""Independent semantic probes for closure_audit_response_2026-07-15.md.

This is deliberately not a production regression suite: it exercises boundaries that the
response's new tests do not cover and prints machine-readable evidence for the audit report.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cost_eval.build_llm import build_llm_spec
from cost_eval.configs.from_mindformers import from_mindformers_dict
from cost_eval.parallel_model import ParallelModel
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.report import Evaluator
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import HardwareSpec, OptimizerSpec, ParallelConfig, RecomputeSpec, SwapSpec


GIB = 2**30


def capture(fn):
    try:
        return {"accepted": True, "value": fn()}
    except Exception as exc:  # the exception type/message are part of the evidence
        return {"accepted": False, "error": type(exc).__name__, "message": str(exc)}


def minimal_mf() -> dict:
    return {
        "model": {
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "vocab_size": 1000,
            "seq_length": 512,
            "compute_dtype": "bfloat16",
        },
        "training": {"local_batch_size": 1},
    }


def evaluator(recompute=None, pc=None):
    return Evaluator(
        build_llm_spec(deepseek_v3(4)),
        pc or ParallelConfig(dp_shard=2, sequence_parallel=True),
        OptimizerSpec.adamw(),
        HardwareSpec(64 * GIB),
        recompute or RecomputeSpec(),
        SwapSpec(),
    )


def peak_for(recompute=None, pc=None):
    return evaluator(recompute, pc).evaluate().per_stage[0].peak_bytes


def recompute_boundaries():
    baseline = peak_for()
    cases = {
        "full_nonexistent_layer": RecomputeSpec("full", {999}),
        "select_empty_map": RecomputeSpec("select", select_ops={}),
        "select_nonexistent_layer": RecomputeSpec(
            "select", select_ops={999: {"definitely_missing"}}
        ),
    }
    out = {"baseline_peak": baseline}
    for name, rc in cases.items():
        out[name] = capture(lambda rc=rc: peak_for(rc))
        if out[name]["accepted"]:
            out[name]["same_as_no_recompute"] = out[name]["value"] == baseline
    return out


def adapter_boundaries():
    negative_dropout = minimal_mf()
    negative_dropout["model"]["attention_dropout"] = -0.1

    optimizer_key_typo = minimal_mf()
    optimizer_key_typo["optimizer"] = {"type": "AdamW", "tyep": "SGD"}

    optimizer_segment_typo = minimal_mf()
    optimizer_segment_typo["optimzier"] = {"type": "SGD"}

    unsupported_overlap = minimal_mf()
    unsupported_overlap["parallelism"] = {"pipeline_parallel_overlap_p2p": True}

    qk = minimal_mf()
    qk["model"]["qk_layernorm"] = True

    return {
        "negative_nonzero_dropout": capture(
            lambda: from_mindformers_dict(negative_dropout).llm.qk_layernorm
        ),
        "optimizer_key_typo": capture(
            lambda: from_mindformers_dict(optimizer_key_typo).optimizer.type
        ),
        "optimizer_segment_typo": capture(
            lambda: from_mindformers_dict(optimizer_segment_typo).optimizer.type
        ),
        "unsupported_overlap_true": capture(
            lambda: from_mindformers_dict(unsupported_overlap).parallel.pp
        ),
        "qk_layernorm_true_became": capture(
            lambda: from_mindformers_dict(qk).llm.qk_layernorm
        ),
    }


def structure_boundaries():
    neg_og = dataclasses.replace(deepseek_v4(4), o_groups=-1)

    def resolved_tensor_summary(config):
        spec = build_llm_spec(config)
        graph = ShapeEval().resolve(spec, ParallelModel(ParallelConfig(), spec.dims.n_layers, 1))
        tensors = []
        for layers in graph.stages.values():
            for layer in layers:
                for op in layer.ops:
                    tensors.extend((*op.inputs, op.output, *op.params, *op.saves))
        negative = sorted({(t.name, t.local_numel) for t in tensors if t.local_numel < 0})
        zero_count = sum(t.local_numel == 0 for t in tensors)
        return {
            "layer_pattern_size": len(spec.layer_pattern),
            "negative_tensor_count": len(negative),
            "negative_examples": negative[:8],
            "zero_tensor_count": zero_count,
        }

    invalid_parallel = {}
    for name, pc in {
        "interleave_zero": ParallelConfig(interleave=0),
        "prefetch_depth_negative": ParallelConfig(prefetch_depth=-1),
        "num_microbatches_zero": ParallelConfig(num_microbatches=0),
    }.items():
        invalid_parallel[name] = capture(lambda pc=pc: peak_for(pc=pc))

    return {
        "negative_o_groups": capture(lambda: resolved_tensor_summary(neg_og)),
        "invalid_basic_dimensions": {
            name: capture(lambda config=config: resolved_tensor_summary(config))
            for name, config in {
                "negative_hidden_size": dataclasses.replace(deepseek_v3(4), hidden_size=-1792),
                "zero_seq_length": dataclasses.replace(deepseek_v3(4), seq_length=0),
                "zero_vocab_size": dataclasses.replace(deepseek_v3(4), vocab_size=0),
                "negative_num_layers": dataclasses.replace(deepseek_v3(4), num_layers=-1),
            }.items()
        },
        "invalid_parallel_configs": invalid_parallel,
    }


def param_rosters(spec):
    graph = ShapeEval().resolve(spec, ParallelModel(ParallelConfig(), spec.dims.n_layers, 1))
    numel = {}
    byte_count = {}
    for layers in graph.stages.values():
        for layer in layers:
            for op in layer.ops:
                for weight in op.params:
                    key = (layer.layer_id, layer.layer_type, weight.name)
                    numel[key] = numel.get(key, 0) + weight.local_numel
                    byte_count[key] = byte_count.get(key, 0) + weight.local_numel * weight.dtype_bytes
    return numel, byte_count


def parameter_evidence():
    original = build_llm_spec(deepseek_v3(4))
    altered = copy.deepcopy(original)
    changed = []
    for layer_spec in altered.layer_specs.values():
        for op in layer_spec.ops:
            for weight in op.params:
                if weight.name == "router_w":
                    changed.append((weight.dtype_bytes, 2))
                    weight.dtype_bytes = 2
    original_numel, original_bytes = param_rosters(original)
    altered_numel, altered_bytes = param_rosters(altered)

    gated = build_llm_spec(dataclasses.replace(deepseek_v4(4), moe_shared_expert_gating=True))
    gate_ops = []
    for layer_type, layer_spec in gated.layer_specs.items():
        for op in layer_spec.ops:
            if op.name == "shared_gate":
                output_name = op.output.name
                consumers = [other.name for other in layer_spec.ops
                             if any(inp.name == output_name for inp in other.inputs)]
                gate_ops.append({
                    "layer_type": layer_type,
                    "param_names": [w.name for w in op.params],
                    "consumers": consumers,
                })

    return {
        "router_dtype_mutations": changed,
        "numel_roster_unchanged_after_dtype_corruption": original_numel == altered_numel,
        "byte_roster_unchanged_after_dtype_corruption": original_bytes == altered_bytes,
        "shared_gate_ops": gate_ops,
    }


def flash_workspace_evidence():
    spec = build_llm_spec(deepseek_v3(4))
    out = {}
    for tp in (1, 2, 8):
        pc = ParallelConfig(tp=tp, sequence_parallel=(tp > 1))
        graph = ShapeEval().resolve(spec, ParallelModel(pc, spec.dims.n_layers, tp))
        values = [op.workspace_bytes for layers in graph.stages.values() for layer in layers
                  for op in layer.ops if op.name == "flash"]
        out[str(tp)] = sorted(set(values))
    return out


def timeline_identity_evidence():
    results = {}
    for pp, v, m, layers in ((2, 2, 4, 8), (2, 3, 7, 12), (3, 2, 8, 12)):
        spec = build_llm_spec(deepseek_v3(layers))
        pc = ParallelConfig(pp=pp, interleave=v, num_microbatches=m)
        report = Evaluator(
            spec, pc, OptimizerSpec.adamw(), HardwareSpec(64 * GIB),
            RecomputeSpec(), SwapSpec()
        ).evaluate(record_timeline=True)
        stages = {}
        for stage in report.per_stage:
            identities = [(s.event, s.mb, s.chunk) for s in stage.timeline]
            stages[str(stage.stage)] = {
                "samples": len(identities),
                "unique": len(set(identities)),
                "has_chunk": any(s.chunk >= 0 for s in stage.timeline),
            }
        results[f"pp{pp}_v{v}_m{m}_l{layers}"] = stages
    return results


def main():
    evidence = {
        "adapter_boundaries": adapter_boundaries(),
        "recompute_boundaries": recompute_boundaries(),
        "structure_boundaries": structure_boundaries(),
        "parameter_evidence": parameter_evidence(),
        "flash_workspace_bytes": flash_workspace_evidence(),
        "timeline_identity": timeline_identity_evidence(),
    }
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
