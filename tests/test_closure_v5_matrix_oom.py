"""P2-01 双 OOM 口径 —— 策略矩阵 analyze_matrix.evaluate() 输出（closure-audit v5）。

矩阵每配置此前仅返回 `oom`（allocated 单口径）。本组钉住 evaluate() 返回额外带
`allocated_oom` / `reserved_oom` 两字段，且 `oom` 与 `allocated_oom` 同义（保留兼容）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze_matrix


def test_analyze_matrix_evaluate_has_dual_oom_fields():
    r = analyze_matrix.evaluate(N=4, dp_shard=2, ep=2)
    assert "oom" in r and "allocated_oom" in r and "reserved_oom" in r
    assert isinstance(r["allocated_oom"], bool)
    assert isinstance(r["reserved_oom"], bool)
    # allocated_oom 与旧 oom 同义（兼容）
    assert r["oom"] == r["allocated_oom"]


def test_analyze_matrix_reserved_not_below_allocated():
    # reserved 是 allocated + HCCL，故 reserved OOM ⊇ allocated OOM：
    # allocated 超容时 reserved 必然也超（reserved_oom 至少与 allocated_oom 一样紧）。
    r = analyze_matrix.evaluate(N=4, dp_shard=2, ep=2)
    if r["allocated_oom"]:
        assert r["reserved_oom"]
