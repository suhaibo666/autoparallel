"""P2-01 双 OOM 口径对外输出（closure-audit v5, 2026-07-15 §4.8）。

`PeakMemoryReport` 已有 `allocated_oom`/`reserved_oom` 核心属性，但主要消费者（web
explorer 的 `eval_config`）此前仅输出 allocated 单口径。本组回归钉住 `eval_config` 返回里
**顶层 + 每 stage** 都带 reserved 口径，且两口径聚合一致；并在 `eval_config` 依赖的 report
层验证 P2-01「reserved 超但 allocated 未超」的分离语义（eval_config 硬编 64GiB 容量，无法用
参数把 allocated 精确落进 reserved-only 窄带，故分离语义在 report 层直接验证）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve_explorer import eval_config
from cost_eval.build_llm import build_llm_spec
from cost_eval.presets import deepseek_v3
from cost_eval.report import Evaluator
from cost_eval.specs import (HardwareSpec, OptimizerSpec, ParallelConfig,
                             RecomputeSpec, SwapSpec)

G = 2 ** 30
MiB = 2 ** 20
CAP_MIB = 64 * G / MiB   # eval_config 硬编码容量（HardwareSpec max_device_memory=64*2**30）

BASE = {"attn": "mla", "layers": "4", "dp": "1", "tp": "1", "pp": "1", "batch": "1"}


def _cfg(**over):
    d = dict(BASE)
    d.update({k: str(v) for k, v in over.items()})
    return d


def test_eval_config_toplevel_dual_oom_fields():
    r = eval_config(_cfg())
    assert r["ok"], r
    for k in ("allocated_oom", "reserved_oom", "reserved_margin_mib"):
        assert k in r, f"顶层缺字段 {k}"
    # 小配置远未超 64GiB → 两口径均 False、余量为正
    assert r["allocated_oom"] is False
    assert r["reserved_oom"] is False
    assert r["reserved_margin_mib"] > 0
    # 旧字段保留兼容
    assert "device_peak" in r and "hccl_mib" in r


def test_eval_config_stage_dual_oom_fields():
    r = eval_config(_cfg())
    assert r["ok"], r
    for s in r["stages"]:
        assert "oom" in s and "reserved_oom" in s and "reserved_mib" in s
        assert isinstance(s["reserved_oom"], bool)
        # reserved 估计 = allocated 峰值 + HCCL(>=0) ⇒ 不小于 allocated 峰值
        assert s["reserved_mib"] >= s["peak"]


def test_eval_config_reserved_strictly_above_allocated_with_ep_hccl():
    # ep=2 引入 EP 通信域 → HCCL 缓冲>0 → reserved 估计严格高于 allocated（audit ep=2 手法）
    r = eval_config(_cfg(dp=2, ep=2))
    assert r["ok"], r
    assert r["hccl_mib"] > 0
    tight = r["tightest"]
    st = next(s for s in r["stages"] if s["stage"] == tight)
    assert st["reserved_mib"] > st["peak"]
    # 顶层 reserved 余量 = 容量 − 最紧 stage reserved 估计
    worst_reserved = max(s["reserved_mib"] for s in r["stages"])
    assert abs(r["reserved_margin_mib"] - (CAP_MIB - worst_reserved)) < 1.0


def test_eval_config_oom_aggregation_consistency():
    r = eval_config(_cfg(dp=2, ep=2))
    assert r["ok"], r
    assert r["allocated_oom"] == any(s["oom"] for s in r["stages"])
    assert r["reserved_oom"] == any(s["reserved_oom"] for s in r["stages"])
    # reserved_oom 与余量符号一致（余量<0 即超容）
    assert r["reserved_oom"] == (r["reserved_margin_mib"] < 0)


def test_reserved_only_oom_split_via_report_api():
    """P2-01 两口径分离：reserved 超但 allocated 未超（eval_config 依赖的 report 层直证）。"""
    sp = build_llm_spec(deepseek_v3(4))
    pc = ParallelConfig(dp_shard=2, ep=2, num_microbatches=1)
    r0 = Evaluator(sp, pc, OptimizerSpec.adamw(), HardwareSpec(999 * G),
                   RecomputeSpec(), SwapSpec()).evaluate()
    peak = r0.per_stage[r0.tightest_stage].peak_bytes
    resv = r0.reserved_estimate_bytes(r0.tightest_stage)
    assert resv > peak   # HCCL 缓冲使 reserved 估计 > allocated 峰值
    cap = (peak + resv) // 2   # peak < cap < resv
    r = Evaluator(sp, pc, OptimizerSpec.adamw(), HardwareSpec(cap),
                  RecomputeSpec(), SwapSpec()).evaluate()
    assert r.allocated_oom is False and r.reserved_oom is True
