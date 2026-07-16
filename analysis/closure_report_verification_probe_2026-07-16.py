"""Independent adversarial probes for ``closure_report_2026-07-15.md``.

This file is audit evidence only.  It does not modify production code and it
deliberately exercises integration boundaries not covered by the closure-wave
tests.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_eval.build_llm import build_llm_spec
from cost_eval.configs.from_mindformers import from_mindformers_dict
from cost_eval.layers.ffn import _moe_dispatch_token_expr
from cost_eval.llm_config import LLMConfig, to_dimtable
from cost_eval.model_spec import DimTable
from cost_eval.opdag import crosscheck as opdag_crosscheck
from cost_eval.presets import deepseek_v3
from cost_eval.report import Evaluator
from cost_eval.shape_eval import eval_expr
from cost_eval.specs import (
    HardwareSpec,
    OptimizerSpec,
    ParallelConfig,
    RecomputeSpec,
    SwapSpec,
)
from cost_eval.structure_mem import _backward_max_live
from validate_dsv3 import build_dsv3_spec

MIB = 2**20
GIB = 2**30


def capture(fn):
    try:
        return {"accepted": True, "value": fn()}
    except Exception as exc:  # retain exact fail-loud behavior as evidence
        return {
            "accepted": False,
            "exception": type(exc).__name__,
            "message": str(exc),
        }


def minimal_mf(*, model_extra=None, parallel_extra=None):
    model = {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "hidden_size": 1024,
        "vocab_size": 1000,
        "seq_length": 512,
        "compute_dtype": "bfloat16",
    }
    model.update(model_extra or {})
    return {
        "model": model,
        "training": {"local_batch_size": 1},
        "parallelism": dict(parallel_extra or {}),
    }


def qk_norm_adapter_split_brain():
    direct_cfg = LLMConfig(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_query_groups=2,
        vocab_size=16,
        seq_length=8,
        head_dim=4,
        attn_type="gqa",
        ffn_hidden_size=16,
        qk_layernorm=True,
    )
    direct_spec = build_llm_spec(direct_cfg)
    direct_names = sorted({op.name for ls in direct_spec.layer_specs.values() for op in ls.ops})
    adapter = capture(
        lambda: dataclasses.asdict(
            from_mindformers_dict(
                minimal_mf(model_extra={"qk_layernorm": True})
            ).llm
        )
    )
    return {
        "direct_api_has_q_norm": "q_norm" in direct_names,
        "direct_api_has_k_norm": "k_norm" in direct_names,
        "yaml_adapter": adapter,
    }


def router_dtype_is_still_ignored():
    common = {
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 512,
        "moe_shared_expert_intermediate_size": 512,
        "use_shared_expert_gating": True,
        "first_k_dense_replace": 1,
    }

    def load(dtype):
        cfg = dict(common, router_dense_type=dtype)
        bundle = from_mindformers_dict(minimal_mf(model_extra=cfg))
        spec = build_llm_spec(bundle.llm)
        gates = [
            op
            for ls in spec.layer_specs.values()
            for op in ls.ops
            if op.name == "shared_gate"
        ]
        return bundle.llm, [
            {
                "weight_dtype": op.params[0].dtype_bytes,
                "saved_dtypes": [s.dtype_bytes for s in op.saves],
            }
            for op in gates
        ]

    fp32_cfg, fp32_gate = load("float32")
    bf16_cfg, bf16_gate = load("bfloat16")
    return {
        "adapter_llm_configs_equal": fp32_cfg == bf16_cfg,
        "fp32_gate": fp32_gate,
        "bf16_gate": bf16_gate,
        "has_router_dtype_field": hasattr(fp32_cfg, "moe_router_dtype_bytes"),
    }


def modeled_features_that_are_not_publicly_reachable():
    base = dict(
        num_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        num_query_groups=2,
        vocab_size=16,
        seq_length=16,
        batch_size=1,
        head_dim=4,
        attn_type="gqa",
        ffn_hidden_size=16,
    )
    return {
        "gqa_cp_buffer_llmconfig": capture(
            lambda: dataclasses.asdict(LLMConfig(**base, cp_kv_allgather_buffer=True))
        ),
        "pp_overlap_parallelconfig": capture(
            lambda: dataclasses.asdict(
                ParallelConfig(pp=2, pipeline_parallel_overlap_p2p=True)
            )
        ),
        "pp_overlap_yaml_adapter": capture(
            lambda: dataclasses.asdict(
                from_mindformers_dict(
                    minimal_mf(
                        parallel_extra={
                            "pipeline_parallel": 2,
                            "pipeline_parallel_overlap_p2p": True,
                        }
                    )
                ).parallel
            )
        ),
    }


def moe_skew_factor_below_one_is_accepted():
    base = dict(
        H=16,
        F=32,
        n_heads=4,
        n_kv=2,
        head_dim=4,
        S=8,
        B=1,
        vocab=32,
        n_layers=3,
        n_experts=4,
        topk=2,
        moe_F=32,
        moe_shared_F=32,
    )
    balanced = DimTable(**base)
    unsafe = DimTable(moe_dispatch_mode="skew", moe_skew_factor=0.5, **base)
    balanced_tokens = eval_expr(_moe_dispatch_token_expr(balanced), balanced)
    unsafe_tokens = eval_expr(_moe_dispatch_token_expr(unsafe), unsafe)
    return {
        "balanced_tokens": balanced_tokens,
        "skew_0p5_tokens": unsafe_tokens,
        "underestimation_accepted": unsafe_tokens < balanced_tokens,
    }


def backward_scratch_window_is_not_an_upper_bound_proof():
    class ScratchOp:
        def __init__(self, value):
            self.bwd_scratch_bytes = value

    values = [4000, 0, 3000]
    modeled = _backward_max_live([ScratchOp(value) for value in values])
    return {
        "scratch_bytes_by_op": values,
        "previous_conservative_sum": sum(values),
        "window2_result": modeled,
        "bytes_removed_without_runtime_lifetime_evidence": sum(values) - modeled,
        "less_than_sum_only_proves_tighter_not_oom_safe": modeled < sum(values),
    }


def opdag_false_green_paths():
    root = opdag_crosscheck.default_mf_root()
    if not os.path.isdir(root):
        return {"source_available": False, "root": root}

    # A memory-affecting saves drift preserves the coarse operator-category
    # census and therefore remains green.
    drifted = build_llm_spec(deepseek_v3(4))
    target = next(
        op
        for ls in drifted.layer_specs.values()
        for op in ls.ops
        if op.name == "linear_qb"
    )
    old_saves = [s.name for s in target.saves]
    target.saves = []
    drift_report = opdag_crosscheck.validate_against_opdag(
        drifted, mf_root=root, strict=True, warn=False
    )

    # Extraction failures are recorded as ``uncovered`` but ``ok`` only looks
    # at findings, so even strict=True does not fail.
    original = opdag_crosscheck._MLA_CORR.census_fn

    def fail_extract(_root):
        raise RuntimeError("injected extractor failure")

    try:
        opdag_crosscheck._MLA_CORR.census_fn = fail_extract
        extraction = capture(
            lambda: _opdag_summary(
                opdag_crosscheck.validate_against_opdag(
                    build_llm_spec(deepseek_v3(4)),
                    mf_root=root,
                    strict=True,
                    warn=False,
                )
            )
        )
    finally:
        opdag_crosscheck._MLA_CORR.census_fn = original

    return {
        "source_available": True,
        "root": root,
        "memory_drift": {
            "removed_saves": old_saves,
            "strict_report_ok": drift_report.ok,
            "findings": len(drift_report.findings),
        },
        "extractor_failure": extraction,
    }


def _opdag_summary(report):
    return {
        "available": report.available,
        "ok": report.ok,
        "findings": len(report.findings),
        "uncovered": report.uncovered,
    }


def kopt_is_not_observed_by_loss_peak_runs():
    spec, _dims, full_layers = build_dsv3_spec(4)
    report = Evaluator(
        spec,
        ParallelConfig(dp_shard=2, sequence_parallel=True),
        OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
        HardwareSpec(max_device_memory=59 * GIB),
        RecomputeSpec(mode="full", full_layers=full_layers),
        SwapSpec(),
    ).evaluate(record_timeline=True)
    stage = report.per_stage[0]
    opt = next(s for s in stage.timeline if s.event == "optstep")
    return {
        "global_peak_event": stage.peak_event,
        "global_peak_mib": round(stage.peak_bytes / MIB, 3),
        "optstep_total_mib": round(opt.total_bytes / MIB, 3),
        "optstep_bucket_mib": round(opt.breakdown.optstep / MIB, 3),
        "optstep_below_global_peak": opt.total_bytes < stage.peak_bytes,
    }


def reserved_model_against_new_npu_rows():
    # Values are copied verbatim from analysis/realmachine/npu_closure_2026-07-15.md.
    rows = {
        "B": (11733.0, 475.0),
        "A_historical": (12473.1, 972.9),
        "C": (13213.2, 1056.8),
        "E": (13953.3, 1000.7),
        "D": (8810.5, 921.5),
        "S": (14716.4, 541.6),
    }
    out = {}
    for name, (allocated, real_delta) in rows.items():
        modeled_delta = 400.0 + 0.018 * allocated  # world+FSDP HCCL plus 1.8% pool
        out[name] = {
            "real_reserved_minus_alloc_mib": real_delta,
            "modeled_reserved_minus_alloc_mib": round(modeled_delta, 3),
            "reserved_error_mib": round(modeled_delta - real_delta, 3),
        }
    return out


def npu_claim_arithmetic():
    observed = {
        (2, 4096, "full"),
        (4, 4096, "full"),
        (6, 4096, "full"),
        (8, 4096, "full"),
        (4, 2048, "full"),
        (4, 4096, "select_attn"),
    }
    full_grid = {
        (n, seq, mode)
        for n in (2, 4, 6, 8)
        for seq in (2048, 4096)
        for mode in ("full", "select_attn")
    }
    historical = {(4, 4096, "full")}
    return {
        "claimed_cartesian_grid_size": len(full_grid),
        "observed_combinations": len(observed),
        "missing_combinations": sorted(full_grid - observed),
        "new_points": len(observed - historical),
        "historical_replays": len(observed & historical),
    }


def main():
    evidence = {
        "qk_norm_adapter_split_brain": qk_norm_adapter_split_brain(),
        "router_dtype_is_still_ignored": router_dtype_is_still_ignored(),
        "modeled_features_not_publicly_reachable": modeled_features_that_are_not_publicly_reachable(),
        "moe_skew_factor_below_one": moe_skew_factor_below_one_is_accepted(),
        "backward_scratch_upper_bound_gap": backward_scratch_window_is_not_an_upper_bound_proof(),
        "opdag_false_green_paths": opdag_false_green_paths(),
        "kopt_non_identifiability": kopt_is_not_observed_by_loss_peak_runs(),
        "reserved_model_vs_npu": reserved_model_against_new_npu_rows(),
        "npu_claim_arithmetic": npu_claim_arithmetic(),
    }
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
