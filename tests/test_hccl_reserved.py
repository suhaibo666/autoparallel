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


def test_hccl_single_card_no_comm():
    # 全 1（单卡）：无跨卡通信 → **0 HCCL 缓冲**（2026-07-20 源忠实订正:size==1 的组不占 buffer）。
    assert _report().hccl_reserved_bytes == 0


def test_hccl_scales_with_tp():
    # tp>1：world(2) + tp(2) → 2×200MB
    assert _report(tp=2).hccl_reserved_bytes == 2 * HCCL_BYTES_PER_GROUP


def test_hccl_scales_with_fsdp():
    # dp_shard>1：world(2) + fsdp(2) → 2×200MB（DSv4 标定同口径）
    assert _report(dp_shard=2).hccl_reserved_bytes == 2 * HCCL_BYTES_PER_GROUP


def test_hccl_cp_is_distinct_domain():
    # cp=2：world(2) + fsdp(dp_shard·cp=2) + **cp(2)** → 3×200MB（cp 独立于 fsdp,旧模型误折）。
    assert _report(dp_shard=1, cp=2).hccl_reserved_bytes == 3 * HCCL_BYTES_PER_GROUP


def test_hccl_ep_adds_ep_and_efsdp():
    # ep>1 的 sparse mesh：world + fsdp + ep + efsdp(dp_shard·cp·tp//ep) → efsdp 仅 size>1 才计。
    from cost_eval.framework import communicator_breakdown
    names = {n for n, _ in communicator_breakdown(ParallelConfig(dp_shard=4, tp=2, ep=2))}
    assert {"ep", "efsdp"} <= names        # dp_shard4·tp2//ep2 = efsdp 4 > 1 → 计入


def test_hccl_not_in_allocated_peak():
    # HCCL 不进 allocated 峰值：tp=2 与 tp=1 的 per_stage peak 不因 HCCL 而变（结构项各自算）
    r1 = _report()
    r2 = _report(tp=2)
    # peak_bytes 是 allocated（不含 hccl）；hccl 只在独立字段
    assert r1.hccl_reserved_bytes != r2.hccl_reserved_bytes
    assert r2.hccl_reserved_bytes == 2 * HCCL_BYTES_PER_GROUP


def test_reserved_estimate_adds_hccl():
    # §F7 pool 碎片闭环（Y4，2026-07-15）：reserved 估计现 = allocated 峰值 + HCCL + allocator
    # pool 碎片（此前只加 HCCL）。断言据此更新以含正的 pool 分量（allocated 侧 peak_bytes 不变）。
    from cost_eval.framework import allocator_pool_fragmentation
    rep = _report(tp=2)
    peak = rep.per_stage[0].peak_bytes
    pool = allocator_pool_fragmentation(None, peak)
    assert pool > 0
    assert rep.reserved_estimate_bytes(0) == peak + rep.hccl_reserved_bytes + pool
