"""Independent counterexamples for closure_audit_v2_response_2026-07-15.md.

This is an audit probe, not a production test.  It intentionally exercises
boundaries that are absent from the closure-v* regression files.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_eval.build_llm import build_llm_spec
from cost_eval.configs.from_mindformers import from_mindformers_dict
from cost_eval.llm_config import LLMConfig
from cost_eval.parallel_model import ParallelModel
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.report import Evaluator
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import (
    HardwareSpec,
    OptimizerSpec,
    ParallelConfig,
    RecomputeSpec,
    SwapSpec,
)

GIB = 2**30


def _capture(fn):
    try:
        value = fn()
        return {"accepted": True, "value": value}
    except Exception as exc:  # audit output must retain the exact rejection class/message
        return {
            "accepted": False,
            "exception": type(exc).__name__,
            "message": str(exc),
        }


def _mf(parallelism=None, *, qk_layernorm=False):
    return {
        "model": {
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "hidden_size": 1024,
            "vocab_size": 1000,
            "seq_length": 512,
            "compute_dtype": "bfloat16",
            "qk_layernorm": qk_layernorm,
        },
        "training": {"local_batch_size": 1},
        "parallelism": parallelism or {},
    }


def adapter_numeric_boundaries():
    def adapt(par):
        bundle = from_mindformers_dict(_mf(par))
        return dataclasses.asdict(bundle.parallel)

    plain = from_mindformers_dict(_mf({"data_parallel_shard": 8}))
    dense_one = from_mindformers_dict(_mf({
        "data_parallel_shard": 8,
        "dense_fsdp_shard_size": 1,
    }))
    return {
        # Runtime source says shard_size=1 replicates dense weights when the full
        # FSDP domain is 8, but the adapter discards the key and returns the same pc.
        "dense_fsdp_1_with_full_fsdp_8": _capture(lambda: adapt({
            "data_parallel_shard": 8,
            "dense_fsdp_shard_size": 1,
        })),
        "dense_fsdp_1_semantics_lost": plain.parallel == dense_one.parallel,
        # Runtime source says shard_size==full fsdp reuses the normal FSDP mesh.
        "dense_fsdp_8_with_full_fsdp_8": _capture(lambda: adapt({
            "data_parallel_shard": 8,
            "dense_fsdp_shard_size": 8,
        })),
        # Runtime requires a positive int.
        "dense_fsdp_zero": _capture(lambda: adapt({
            "data_parallel_shard": 8,
            "dense_fsdp_shard_size": 0,
        })),
        # degree=1 is neutral under colossal CP, but is classified as a truthy flag.
        "colossal_ulysses_degree_one": _capture(lambda: adapt({
            "context_parallel_method": "colossal",
            "ulysses_degree_in_cp": 1,
        })),
    }


def _peak(recompute):
    spec = build_llm_spec(deepseek_v3(4))
    report = Evaluator(
        spec,
        ParallelConfig(dp_shard=2, sequence_parallel=True),
        OptimizerSpec.adamw(),
        HardwareSpec(64 * GIB),
        recompute,
        SwapSpec(),
    ).evaluate()
    return report.per_stage[0].peak_bytes


def recompute_boundaries():
    none_peak = _peak(RecomputeSpec("None"))
    typo = _capture(lambda: _peak(RecomputeSpec("ful", {1})))
    empty_selector = _capture(
        lambda: _peak(RecomputeSpec("select", select_ops={1: {""}}))
    )
    if typo["accepted"]:
        typo["same_as_none"] = typo["value"] == none_peak
    return {
        "none_peak": none_peak,
        "unknown_mode_ful": typo,
        # The empty string is a substring of every op name/type and therefore
        # silently selects the whole layer rather than being rejected.
        "empty_string_selector": empty_selector,
    }


def scalar_boundaries():
    return {
        # bool is an int subclass in Python; the validator does not exclude it.
        "parallel_tp_true": _capture(
            lambda: dataclasses.asdict(ParallelConfig(tp=True))
        ),
        "parallel_num_microbatches_true": _capture(
            lambda: dataclasses.asdict(ParallelConfig(num_microbatches=True))
        ),
        "llm_boolean_dimensions": _capture(
            lambda: build_llm_spec(LLMConfig(
                num_layers=2,
                hidden_size=True,
                num_attention_heads=True,
                vocab_size=16,
                seq_length=16,
                head_dim=1,
                attn_type="gqa",
                ffn_hidden_size=4,
            )).dims.as_dict()
        ),
        # int(4.9)==4 in both validation and dispatch: invalid input is truncated.
        "csa_ratio_4_9": _capture(
            lambda: list(build_llm_spec(dataclasses.replace(
                deepseek_v4(4),
                csa_compress_ratios=(4.9, 1, 1, 4),
            )).layer_specs)
        ),
    }


def shared_gate_dtype_and_dataflow():
    spec = build_llm_spec(dataclasses.replace(
        deepseek_v3(4), moe_shared_expert_gating=True
    ))
    graph = ShapeEval().resolve(
        spec,
        ParallelModel(ParallelConfig(), spec.dims.n_layers, world_size=1),
    )
    gates = []
    for layers in graph.stages.values():
        for layer in layers:
            for op in layer.ops:
                if op.name == "shared_gate":
                    gates.append({
                        "layer_id": layer.layer_id,
                        "layer_type": layer.layer_type,
                        "input_dtypes": {t.name: t.dtype_bytes for t in op.inputs},
                        "param_dtypes": {t.name: t.dtype_bytes for t in op.params},
                        "output_dtype": op.output.dtype_bytes,
                        "saves": [t.name for t in op.saves],
                    })
    return {
        "resolved_gate_ops": gates,
        "has_explicit_fp32_hidden_cast_op": any(
            op.name == "shared_gate_input_cast_fp32"
            for layer_spec in spec.layer_specs.values()
            for op in layer_spec.ops
        ),
    }


def qk_layernorm_boundary():
    off = from_mindformers_dict(_mf(qk_layernorm=False)).llm
    on = from_mindformers_dict(_mf(qk_layernorm=True)).llm
    # Local MindFormers Qwen3-32B config: S=4096, 64 Q heads, 8 KV
    # heads, head_dim=128, 64 layers.  This is the tensor extent entering
    # q_layernorm/k_layernorm; it is not an assertion about allocator overhead.
    qk_elements_per_layer = 4096 * (64 + 8) * 128
    return {
        "adapter_output_flag_when_false": off.qk_layernorm,
        "adapter_output_flag_when_true": on.qk_layernorm,
        "llm_configs_equal": off == on,
        "built_specs_equal": build_llm_spec(off) == build_llm_spec(on),
        "qwen3_32b_qk_extent_mib_per_layer_bf16": qk_elements_per_layer * 2 / 2**20,
        "qwen3_32b_qk_extent_mib_per_layer_fp32": qk_elements_per_layer * 4 / 2**20,
        "qwen3_32b_qk_extent_gib_64_layers_bf16": qk_elements_per_layer * 2 * 64 / 2**30,
        "qwen3_32b_qk_extent_gib_64_layers_fp32": qk_elements_per_layer * 4 * 64 / 2**30,
    }


def main():
    evidence = {
        "adapter_numeric_boundaries": adapter_numeric_boundaries(),
        "recompute_boundaries": recompute_boundaries(),
        "scalar_boundaries": scalar_boundaries(),
        "shared_gate_dtype_and_dataflow": shared_gate_dtype_and_dataflow(),
        "qk_layernorm_boundary": qk_layernorm_boundary(),
    }
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
