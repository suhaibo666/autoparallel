"""可复跑的验收 oracle（closure-audit v2 §F8，2026-07-15）——每个反例独立 capture，
修复前后都能完整输出（不会像旧探针那样在第一个 `ParallelConfig(interleave=0)` 就崩）。

用法：`python analysis/closure_audit_v2_oracle_2026-07-15.py`
期望：F1-F6 反例全部 REJECTED（或语义正确的 ACCEPTED）、F3 gate 权重 fp32。
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cost_eval.build_llm import build_llm_spec
from cost_eval.configs.from_mindformers import from_mindformers_dict, _build_parallel
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.report import Evaluator
from cost_eval.specs import (HardwareSpec, OptimizerSpec, ParallelConfig,
                             RecomputeSpec, SwapSpec)

GIB = 2**30


def _rejected(fn) -> bool:
    try:
        fn(); return False
    except Exception:
        return True


def _accepted(fn) -> bool:
    try:
        fn(); return True
    except Exception:
        return False


def _mf(**over):
    m = {"model": {"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 1024,
                   "vocab_size": 1000, "seq_length": 512, "compute_dtype": "bfloat16"},
         "training": {"local_batch_size": 1}}
    m.update(over)
    return m


def _ev(rc):
    sp = build_llm_spec(deepseek_v3(4))
    return Evaluator(sp, ParallelConfig(dp_shard=2, sequence_parallel=True), OptimizerSpec.adamw(),
                     HardwareSpec(64 * GIB), rc, SwapSpec()).evaluate()


CASES = {
    # F1 dense_fsdp_shard_size（按完整 FSDP 域判定；closure-wave3a Z3 后：1≤v<fsdp 整除**已建模**、不再拒）
    "F1 dense=1 with fsdp=8 (subdomain NOW MODELED, Z3) → ACCEPT":
        lambda: _accepted(lambda: _build_parallel({"parallelism": {"data_parallel_shard": 8, "dense_fsdp_shard_size": 1}}, 0, 4)),
    "F1 dense=8 with fsdp=8 (==fsdp neutral) → ACCEPT":
        lambda: _accepted(lambda: _build_parallel({"parallelism": {"data_parallel_shard": 8, "dense_fsdp_shard_size": 8}}, 0, 4)),
    "F1 dense=0 (non-positive) → REJECT":
        lambda: _rejected(lambda: _build_parallel({"parallelism": {"data_parallel_shard": 8, "dense_fsdp_shard_size": 0}}, 0, 4)),
    "F1 dense=3 with fsdp=8 (non-divisor) → REJECT":
        lambda: _rejected(lambda: _build_parallel({"parallelism": {"data_parallel_shard": 8, "dense_fsdp_shard_size": 3}}, 0, 4)),
    # F6 ulysses_degree_in_cp
    "F6 colossal degree=1 (was false-reject) → ACCEPT":
        lambda: _accepted(lambda: from_mindformers_dict({**_mf(), "parallelism": {"context_parallel_method": "colossal", "ulysses_degree_in_cp": 1}})),
    # F2/F7 qk_layernorm：gqa/mha 现由 build_llm 建 q_norm/k_norm（X3），adapter 亦透传（F7 订正
    # 2026-07-16 消除 split-brain）→ 从「REJECT」翻正为「ACCEPT」。
    "F2/F7 gqa qk_layernorm=True (X3-modeled, adapter reachable) → ACCEPT":
        lambda: _accepted(lambda: from_mindformers_dict({**_mf(), "model": {**_mf()["model"], "num_key_value_heads": 4, "qk_layernorm": True}})),
    "F2 mla qk_layernorm=True (subsumed) → ACCEPT":
        lambda: _accepted(lambda: from_mindformers_dict({**_mf(), "model": {**_mf()["model"], "multi_latent_attention": True, "q_lora_rank": 512, "kv_lora_rank": 256, "qk_rope_head_dim": 64, "qk_nope_head_dim": 64, "v_head_dim": 128, "qk_layernorm": True}})),
    # F4 recompute mode/selector
    "F4 mode 'ful' typo → REJECT": lambda: _rejected(lambda: _ev(RecomputeSpec("ful", {1}))),
    "F4 empty-string selector → REJECT": lambda: _rejected(lambda: _ev(RecomputeSpec("select", select_ops={1: {""}}))),
    # F5 bool/float domain
    "F5 ParallelConfig(tp=True) → REJECT": lambda: _rejected(lambda: ParallelConfig(tp=True)),
    "F5 hidden_size=True → REJECT": lambda: _rejected(lambda: build_llm_spec(dataclasses.replace(deepseek_v3(4), hidden_size=True))),
    "F5 csa ratio 4.9 (non-integer) → REJECT": lambda: _rejected(lambda: build_llm_spec(dataclasses.replace(deepseek_v4(4), csa_compress_ratios=(4.9, 1, 1, 4)))),
}


def main() -> int:
    ok = True
    for name, check in CASES.items():
        passed = check()
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name}")
    # F3 sh_gate_w dtype
    gated = build_llm_spec(dataclasses.replace(deepseek_v4(4), moe_shared_expert_gating=True))
    dts = {w.dtype_bytes for ls in gated.layer_specs.values() for op in ls.ops
           if op.name == "shared_gate" for w in op.params}
    f3 = dts == {4}
    ok = ok and f3
    print(f"  [{'PASS' if f3 else 'FAIL'}]  F3 sh_gate_w dtype fp32(4B): {dts}")
    print(f"\n=== all closure-audit-v2 counterexamples handled: {ok} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
