"""DSv4 real-machine-aligned anchor: evaluator prediction vs real peak.

Reconstructs the **real-machine dsv4 align config** used to capture the anchors
(`.claude/skills/real-machine-memory-sim/prep_dsv4align.py`): seq=2048, heads=64,
v_head_dim=512, FSDP-2 (dp_shard=2), tp=ep=pp=cp=1, SP=on, **no recompute**,
compute=bf16 / params=fp32.

Real-machine FUSED (production) peaks (max_memory_allocated MiB):
  - base   (no mHC, no MTP): 15415.5   peak op = ScatterAddExt (loss)
  - +mHC(4)+MTP(1)         : 21153.1

Usage: MHC=0 MTP=0 python validate_dsv4align.py   (base)
       MHC=1 MTP=1 python validate_dsv4align.py   (mHC+MTP)
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

MiB = 2 ** 20
GiB = 2 ** 30

# real-machine FUSED anchors (max_memory_allocated MiB) by (MHC, MTP)
MEASURED = {(0, 0): 15415.5, (1, 1): 21153.1}


def _cycle_ratios(n):
    cyc = (0, 4, 128)
    return tuple(cyc[i % 3] for i in range(n))


def dsv4_align_config(num_layers=4, mhc=0, mtp=0, seq=2048):
    """LLMConfig matching prep_dsv4align.py real-machine dsv4 config (memory subset).

    qk_layernorm/add_bias omitted (memory-negligible; would fail-loud in build_llm).
    """
    return LLMConfig(
        num_layers=num_layers,
        hidden_size=1792,
        num_attention_heads=64,          # prep_dsv4align.py:77
        num_query_groups=1,
        vocab_size=129280,               # :72
        seq_length=seq,                  # :73 (SEQ=2048)
        batch_size=1,                    # local_batch_size=1
        # MLA dims (prep_dsv4align.py:83-87)
        attn_type="dsv4_hybrid",
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=448,
        v_head_dim=512,
        # dsv4_hybrid frontier (prep_dsv4align.py:101-111)
        csa_compress_ratios=_cycle_ratios(num_layers),
        csa_window_size=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,            # :107
        o_groups=8,                      # :110
        o_lora_rank=1024,                # :111
        window_size=128,
        # FFN / MoE (prep_dsv4align.py:129-131)
        ffn_hidden_size=3072,
        num_moe_experts=4,               # n_routed_experts=4
        moe_router_topk=2,               # num_experts_per_tok=2
        moe_ffn_hidden_size=1024,        # moe_intermediate_size=1024
        moe_shared_expert_num=1,
        moe_shared_ffn_hidden_size=1024,
        moe_capacity_factor=1.0,
        first_k_dense_replace=1,
        # mHC residual
        residual_variant="mhc" if mhc else "plain",
        num_residual_streams=4 if mhc else 1,   # hc_mult=4
        # MTP
        mtp_num_layers=mtp,
        loss_type="logsoftmax_nll",
        cross_entropy_fused=True,       # DSv4 生产用融合 CE kernel（精简 ~3 buffer）→ ① 不 fat（0.930 不动）
        compute_dtype_bytes=2,
    )


def evaluate(num_layers=4, mhc=0, mtp=0, seq=2048, reserve_mib=None):
    from validate_dsv3 import RESIDUAL_MiB
    if reserve_mib is None:
        reserve_mib = RESIDUAL_MiB
    cfg = dsv4_align_config(num_layers, mhc, mtp, seq)
    spec = build_llm_spec(cfg)
    pc = ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1,
                        sequence_parallel=True, num_microbatches=1)
    opt = OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4)
    # NO recompute (real-machine dsv4 runs no-recompute; see prep_dsv4align.py:142)
    ev = Evaluator(spec, pc, opt,
                   HardwareSpec(max_device_memory=54 * GiB,
                                framework_reserve=reserve_mib * MiB),
                   RecomputeSpec(mode="None", full_layers=set()), SwapSpec())
    return ev.evaluate()


def main():
    N = int(os.environ.get("SIM_LAYERS", "4"))
    MHC = int(os.environ.get("MHC", "0"))
    MTP = int(os.environ.get("MTP", "0"))
    SEQ = int(os.environ.get("SIM_SEQ", "2048"))
    rep = evaluate(N, MHC, MTP, SEQ)
    p = rep.per_stage[0]
    b = p.breakdown
    pk = p.peak_bytes / MiB
    real = MEASURED.get((MHC, MTP))
    print(f"=== DSv4-align {N}L seq={SEQ} mHC={MHC} MTP={MTP} (no-recompute, FSDP-2) ===")
    print(f"[peak] pred = {pk:8.1f} MiB @ {p.peak_event}"
          + (f" ; real = {real} ; ratio = {pk/real:.4f}" if real else ""))
    # 已知残差（源码级已定位，2026-07-06 用户定夺维持纯公式、不加常数；设计 §14.2 / DIAGNOSIS.md）：
    # dsv4-fused 结构性 UNDER ~7%——融合 kernel 内部 fp32 [S,12288]×层 + ½·vocab·H lm_head 反向瞬态 +
    # allocator 尾，无 config 公式可算。**OOM 评估请对本预测自留 ≥8% 裕度**（否则会把会 OOM 的配置判为可放下）。
    if real and pk < real:
        print(f"[caveat] dsv4-fused 已知 UNDER {(1 - pk/real) * 100:.1f}%（融合 kernel 内部量，源码级已定位）"
              f" → OOM 评估请自留 ≥8% 裕度；详见 specs §14.2 / analysis/realmachine/dsv4_fused/DIAGNOSIS.md")
    print(f"--- breakdown (MiB) ---")
    for k in ("persistent", "act_live", "gather_buf", "grad_buf", "recomp_scratch",
              "bwd_scratch", "bwd_working_set", "swap_buf", "workspace", "framework"):
        print(f"  {k:16s} = {getattr(b, k)/MiB:9.1f}")


if __name__ == "__main__":
    main()
