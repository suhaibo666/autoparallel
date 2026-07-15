"""D-2：HCCL 通信缓冲（reserved 池）接入报告 —— 按通信域数估计，surfaced 到 PeakMemoryReport。

HCCL 缓冲不进 allocated 峰值（ep=2 真机证实），但计入 reserved 估计。此前 `framework.
hccl_reserved_buffer` 有模型却从不被 `Evaluator.evaluate` 调用；D-2 接入 + 给 reserved 估计。
"""
from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator
from cost_eval.framework import HCCL_BYTES_PER_GROUP

GiB = 2 ** 30


def _report(**pcargs):
    cfg = LLMConfig(num_layers=2, hidden_size=16, num_attention_heads=2, num_query_groups=2,
                    vocab_size=32, seq_length=8, attn_type="gqa", ffn_hidden_size=32)
    spec = build_llm_spec(cfg)
    pcargs.setdefault("sequence_parallel", pcargs.get("tp", 1) > 1)   # C1: tp>1 强制 SP（真机约束）
    pc = ParallelConfig(**pcargs)
    ev = Evaluator(spec, pc, OptimizerSpec.adamw(), HardwareSpec(max_device_memory=59 * GiB),
                   RecomputeSpec(), SwapSpec())
    return ev.evaluate()


def test_hccl_world_only_single_config():
    # 全 1：只有 world 通信器 → 1×200MB
    assert _report().hccl_reserved_bytes == HCCL_BYTES_PER_GROUP


def test_hccl_scales_with_tp():
    # tp>1 多一个 tp 子通信器 → 2×200MB
    assert _report(tp=2).hccl_reserved_bytes == 2 * HCCL_BYTES_PER_GROUP


def test_hccl_scales_with_fsdp():
    # dp_shard*cp>1 多一个 FSDP 子通信器 → 2×200MB
    assert _report(dp_shard=2).hccl_reserved_bytes == 2 * HCCL_BYTES_PER_GROUP


def test_hccl_not_in_allocated_peak():
    # HCCL 不进 allocated 峰值：tp=2 与 tp=1 的 per_stage peak 不因 HCCL 而变（结构项各自算）
    r1 = _report()
    r2 = _report(tp=2)
    # peak_bytes 是 allocated（不含 hccl）；hccl 只在独立字段
    assert r1.hccl_reserved_bytes != r2.hccl_reserved_bytes
    assert r2.hccl_reserved_bytes == 2 * HCCL_BYTES_PER_GROUP


def test_reserved_estimate_adds_hccl():
    rep = _report(tp=2)
    assert rep.reserved_estimate_bytes(0) == rep.per_stage[0].peak_bytes + rep.hccl_reserved_bytes
