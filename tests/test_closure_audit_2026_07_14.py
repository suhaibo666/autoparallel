"""闭环审计（closure-audit 2026-07-14/15）反例转正式回归。

把 `analysis/review_closure_probe_2026-07-14.py` 揭示的**核心 API 未闭环**反例逐条转成守卫：
C1 核心校验（reshard 枚举/tp-SP/非 Adam/full 空集/select 零命中/topk≤0/capacity≤0/o_groups=0）、
C2 adapter 全段 schema（typo/未知键/dropout）、C3 P1-06 解耦，C4 P2-06 chunk 身份 / P2-01 reserved_oom。
"""
import dataclasses

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.configs.from_mindformers import from_mindformers_dict
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.report import Evaluator
from cost_eval.specs import (HardwareSpec, OptimizerSpec, ParallelConfig,
                             RecomputeSpec, SwapSpec)

G = 2**30


def _ev(pc, rc=None, opt=None, swap=None, **kw):
    return Evaluator(build_llm_spec(deepseek_v3(4)), pc,
                     opt or OptimizerSpec.adamw(), HardwareSpec(64 * G),
                     rc or RecomputeSpec(), swap or SwapSpec(), **kw)


# ── C1：核心 API 统一校验 ────────────────────────────────────────────────────────
def test_reshard_enum_typo_rejected_at_config():
    with pytest.raises(ValueError, match="reshard_after_forward"):
        ParallelConfig(reshard_after_forward="nevver")


def test_tp_gt1_requires_sequence_parallel_at_evaluator():
    with pytest.raises(ValueError, match="sequence_parallel"):
        _ev(ParallelConfig(tp=2))
    # 显式绕过通道仍可评（纯内存口径）
    _ev(ParallelConfig(tp=2), check_feasibility=False)


def test_non_adam_optimizer_rejected():
    with pytest.raises(ValueError, match="optimizer"):
        _ev(ParallelConfig(), opt=OptimizerSpec(type="SGD"))


def test_pp_plus_swap_rejected():
    with pytest.raises(ValueError, match="swap"):
        _ev(ParallelConfig(pp=2, num_microbatches=2),
            swap=SwapSpec(enable=True, swap_layers={1}))


def test_full_empty_layers_rejected_at_evaluate():
    with pytest.raises(ValueError, match="full_layers"):
        _ev(ParallelConfig(dp_shard=2), rc=RecomputeSpec("full", set())).evaluate()


def test_select_zero_hit_rejected_at_evaluate():
    with pytest.raises(ValueError, match="零命中"):
        _ev(ParallelConfig(dp_shard=2),
            rc=RecomputeSpec("select", select_ops={1: {"definitely_missing"}})).evaluate()


def test_negative_topk_rejected():
    with pytest.raises(ValueError, match="moe_router_topk"):
        build_llm_spec(dataclasses.replace(deepseek_v3(4), moe_router_topk=-1))


def test_zero_capacity_rejected():
    with pytest.raises(ValueError, match="moe_capacity_factor"):
        build_llm_spec(dataclasses.replace(deepseek_v3(4), moe_capacity_factor=0))


def test_dsv4_zero_o_groups_rejected_before_divzero():
    with pytest.raises(ValueError, match="o_groups"):
        build_llm_spec(dataclasses.replace(deepseek_v4(4), o_groups=0))


# ── C2：adapter 全段 schema ──────────────────────────────────────────────────────
def _mf(**over):
    m = {"model": {"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 1024,
                   "vocab_size": 1000, "seq_length": 512, "compute_dtype": "bfloat16"},
         "training": {"local_batch_size": 1}}
    for k, v in over.items():
        m[k] = v
    return m


def test_training_typo_rejected():
    with pytest.raises(NotImplementedError, match="training"):
        from_mindformers_dict(_mf(training={"local_batch_szie": 2}))


def test_context_typo_rejected():
    with pytest.raises(NotImplementedError, match="context"):
        from_mindformers_dict(_mf(context={"max_device_memry": "50GB"}))


def test_swap_typo_rejected():
    with pytest.raises(NotImplementedError, match="swap"):
        from_mindformers_dict(_mf(swap={"enablee": True}))


def test_nonzero_dropout_rejected():
    m = _mf()
    m["model"]["attention_dropout"] = 0.1
    with pytest.raises(NotImplementedError, match="dropout"):
        from_mindformers_dict(m)
    # =0 通过
    m["model"]["attention_dropout"] = 0.0
    assert from_mindformers_dict(m)


# ── C3：P1-06 CE 与 DSA fusion 解耦 ──────────────────────────────────────────────
def _dsv4_mf(dsa_fused):
    return {"model": {"num_hidden_layers": 4, "num_attention_heads": 128, "hidden_size": 2048,
                      "vocab_size": 1000, "seq_length": 2048, "multi_latent_attention": True,
                      "experimental_attention_variant": "dsv4_hybrid",
                      "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                      "qk_nope_head_dim": 128, "v_head_dim": 128, "o_groups": 16, "o_lora_rank": 256,
                      "dsa_indexer_n_heads": 64, "dsa_indexer_head_dim": 128, "dsa_indexer_topk": 2048,
                      "csa_compress_ratios": [1, 1, 1, 4], "apply_dsa_kernel_fusion": dsa_fused,
                      "force_unfused_dsa": not dsa_fused, "compute_dtype": "bfloat16",
                      "n_routed_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 2048},
            "training": {"local_batch_size": 1}}


def test_ce_fused_decoupled_from_dsa_fusion():
    ce_on = from_mindformers_dict(_dsv4_mf(True)).llm.cross_entropy_fused
    ce_off = from_mindformers_dict(_dsv4_mf(False)).llm.cross_entropy_fused
    assert ce_on == ce_off is True    # 切 DSA 融合不改 CE（DSv4 恒 lean，独立于 attention kernel）


def test_ce_fused_explicit_key_overrides():
    mf = _dsv4_mf(True)
    mf["model"]["cross_entropy_fused"] = False
    assert from_mindformers_dict(mf).llm.cross_entropy_fused is False   # 显式键优先


# ── C4：P2-06 timeline chunk 身份 / P2-01 reserved_oom ───────────────────────────
def test_vpp_timeline_event_mb_chunk_unique():
    sp = build_llm_spec(deepseek_v3(8))
    rep = Evaluator(sp, ParallelConfig(dp_shard=1, pp=2, interleave=2, num_microbatches=4),
                    OptimizerSpec.adamw(), HardwareSpec(64 * G), RecomputeSpec(), SwapSpec()
                    ).evaluate(record_timeline=True)
    triples = [(s.event, s.mb, s.chunk) for s in rep.per_stage[0].timeline]
    assert len(triples) == len(set(triples))                 # (event,mb,chunk) 唯一
    assert any(s.chunk >= 0 for s in rep.per_stage[0].timeline)   # VPP 下有 chunk 身份


def test_reserved_oom_split_from_allocated():
    sp = build_llm_spec(deepseek_v3(4))
    pc = ParallelConfig(dp_shard=2, ep=2, num_microbatches=1)
    r0 = Evaluator(sp, pc, OptimizerSpec.adamw(), HardwareSpec(999 * G),
                   RecomputeSpec(), SwapSpec()).evaluate()
    peak = r0.per_stage[r0.tightest_stage].peak_bytes
    resv = r0.reserved_estimate_bytes(r0.tightest_stage)
    assert resv > peak    # HCCL 缓冲使 reserved > allocated
    cap = (peak + resv) // 2
    r = Evaluator(sp, pc, OptimizerSpec.adamw(), HardwareSpec(cap),
                  RecomputeSpec(), SwapSpec()).evaluate()
    assert r.allocated_oom is False and r.reserved_oom is True   # 两口径分离
