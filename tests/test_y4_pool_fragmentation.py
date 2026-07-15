"""Y4 / P2-01 §F7 闭环：allocator pool 碎片建模 —— reserved 从「纯下界」升级为含 pool 近似分量。

审计缺口（closure-audit §F7）：`reserved_estimate_bytes` 只加 HCCL、**不含** allocator pool
碎片。真机 DSv4：allocated 15415.5 MiB、reserved 16092-16096 MiB，扣除 HCCL ~400 MiB
（FSDP-2 world + FSDP 组 = 2×200MB）后仍缺 ~277-281 MiB pool 分量 →
存在 `reserved_oom=False` 而真实 reserved 已超的临界区。

物理依据：MindSpore 设备内存池 `DynamicMemPoolBestFit`（best-fit 分配 + 块管理）的
reserved > allocated 来自 ①best-fit 留下的块级空洞 ②mempool_block_size 预分配块尾部
③512B 对齐（对齐分量已在 structure_mem 逐张量 roundup 建模）。聚合碎片率 ≈ 277/15415 ≈ 1.8%
（自 DSv4 单点标定；跨模型稳定性待验证——见 realmachine profiler 旁证）。

本组钉住：① pool 碎片函数按碎片率 × allocated 出值 ② reserved_estimate = allocated + HCCL
+ pool（新项）③ DSv4 单点标定：reserved 估计 ≈ 真机 16092 ④ allocated 侧 property 不变
（reserved 只在原基础上加正的 pool 分量）。
"""
from cost_eval.framework import (allocator_pool_fragmentation, POOL_FRAGMENTATION_RATE,
                                 HCCL_BYTES_PER_GROUP)
from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec
from cost_eval.specs import (ParallelConfig, OptimizerSpec, HardwareSpec,
                             RecomputeSpec, SwapSpec)
from cost_eval.report import Evaluator

MiB = 2 ** 20
GiB = 2 ** 30


def _report(**pcargs):
    cfg = LLMConfig(num_layers=2, hidden_size=16, num_attention_heads=2, num_query_groups=2,
                    vocab_size=32, seq_length=8, attn_type="gqa", ffn_hidden_size=32)
    spec = build_llm_spec(cfg)
    pcargs.setdefault("sequence_parallel", pcargs.get("tp", 1) > 1)   # tp>1 强制 SP（真机约束）
    pc = ParallelConfig(**pcargs)
    return Evaluator(spec, pc, OptimizerSpec.adamw(),
                     HardwareSpec(max_device_memory=59 * GiB),
                     RecomputeSpec(), SwapSpec()).evaluate()


# ── ① pool 碎片函数：碎片率 × allocated ──────────────────────────────────────────
def test_pool_fragmentation_is_rate_times_allocated():
    alloc = 10000 * MiB
    assert allocator_pool_fragmentation(None, alloc) == int(POOL_FRAGMENTATION_RATE * alloc)


def test_pool_fragmentation_rate_calibrated_near_1p8pct():
    # DSv4 单点标定：277/15415 ≈ 1.8%（允许 1.5%-2.1% 的标定带）
    assert 0.015 <= POOL_FRAGMENTATION_RATE <= 0.021


def test_pool_fragmentation_scales_with_allocated():
    a = 10000 * MiB
    # ∝ allocated：allocated 翻倍，pool 分量翻倍（int 截断误差 ≤ 1B）
    assert abs(allocator_pool_fragmentation(None, 2 * a)
               - 2 * allocator_pool_fragmentation(None, a)) <= 1


def test_pool_fragmentation_zero_or_negative_allocated():
    # 无分配 → 无碎片；负值防御性归零
    assert allocator_pool_fragmentation(None, 0) == 0
    assert allocator_pool_fragmentation(None, -5) == 0


# ── ③ DSv4 单点标定验证：pool ≈ 277 MiB, reserved 估计 ≈ 真机 16092 ────────────────
def test_dsv4_single_point_calibration_reserved_near_16092():
    real_alloc = int(15415.5 * MiB)          # 真机 DSv4 max_allocated
    pool = allocator_pool_fragmentation(None, real_alloc)
    assert 275 <= pool / MiB <= 282          # 真机缺口 277-281 MiB pool 分量
    hccl = 2 * HCCL_BYTES_PER_GROUP          # FSDP-2：world + FSDP 组 = 2×200MB ≈ 400 MiB
    reserved_est = real_alloc + hccl + pool
    assert 16089 <= reserved_est / MiB <= 16097   # 真机 reserved 16092-16096


# ── ② reserved_estimate = allocated 峰值 + HCCL + pool（新项）────────────────────
def test_reserved_estimate_now_includes_pool():
    rep = _report(tp=2)
    peak = rep.per_stage[0].peak_bytes
    hccl = rep.hccl_reserved_bytes
    pool = allocator_pool_fragmentation(None, peak)
    assert rep.reserved_estimate_bytes(0) == peak + hccl + pool
    assert pool > 0                          # 非零 pool 分量已计入（不再纯下界）


def test_reserved_estimate_strictly_above_hccl_only_lower_bound():
    # 与旧口径（allocated + HCCL）相比，新估计因 pool 项而严格更高（更接近真实上界）
    rep = _report(tp=2)
    peak = rep.per_stage[0].peak_bytes
    hccl_only = peak + rep.hccl_reserved_bytes            # 旧「纯下界」
    assert rep.reserved_estimate_bytes(0) > hccl_only


# ── ④ allocated 口径不变：pool 只加在 reserved 侧，peak_bytes/oom 不受影响 ─────────
def test_allocated_side_untouched_by_pool():
    rep = _report(tp=2)
    peak = rep.per_stage[0].peak_bytes
    # reserved 估计 − pool − HCCL 精确还原 allocated 峰值（peak 侧零改动）
    pool = allocator_pool_fragmentation(None, peak)
    assert rep.reserved_estimate_bytes(0) - pool - rep.hccl_reserved_bytes == peak
    # allocated 口径 OOM 判据仍只看 peak（未混入 pool）
    assert rep.allocated_oom is (peak > rep.max_device_memory)
