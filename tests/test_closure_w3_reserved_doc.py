"""闭环审计第四轮 §F7（analysis/closure_audit_v2_verification_2026-07-15.md）回归：
reserved_oom docstring 自相矛盾。

`reserved_estimate_bytes` docstring 正确称「只加 HCCL、**不含** allocator pool 碎片、是下界」；
但紧接着 `reserved_oom` property docstring 曾称它包含「allocated + HCCL + **池碎片近似**」——矛盾。
真机 DSv4 allocated 15415.5 / reserved 16092-16096（差 676.5-680.5 MiB），HCCL 估 ~400 MiB，
仍缺 ~277-281 MiB pool 分量 → 存在 `reserved_oom=False` 而真实 reserved 已超的临界区。

本组钉住两 docstring 一致：`reserved_oom` 是**基于下界估计（allocated + HCCL，不含 allocator pool
碎片）**的判定——`reserved_oom=True` 是确定超容，`reserved_oom=False` **不保证**真实 reserved 不超；
删「池碎片近似」的错误措辞。property 名 `allocated_oom`/`reserved_oom` 保持不变（下游依赖）。
"""
from cost_eval.report import PeakMemoryReport


def _resv_doc():
    return PeakMemoryReport.reserved_oom.fget.__doc__ or ""


def _est_doc():
    return PeakMemoryReport.reserved_estimate_bytes.__doc__ or ""


def test_reserved_oom_doc_drops_pool_fragment_inclusion_claim():
    # 错误措辞「池碎片近似」必须删除——reserved_oom 判定不含 allocator pool 碎片
    assert "池碎片近似" not in _resv_doc()


def test_reserved_oom_doc_states_lower_bound():
    # 明确它是基于**下界**估计的判定（allocated + HCCL，不含 pool 碎片）
    assert "下界" in _resv_doc()


def test_reserved_oom_doc_false_not_guaranteed():
    # reserved_oom=False **不保证**真实 reserved 不超（pool 碎片未建模）
    assert "不保证" in _resv_doc()


def test_reserved_estimate_doc_consistent_lower_bound():
    # 与 reserved_estimate_bytes 一致：不含 pool 碎片、是下界（防两 docstring 再次分叉）
    d = _est_doc()
    assert "下界" in d and "碎片" in d


def test_property_names_unchanged():
    # 下游 serve_explorer / test_closure_v5_dual_oom 依赖 allocated_oom/reserved_oom 命名
    assert isinstance(PeakMemoryReport.allocated_oom, property)
    assert isinstance(PeakMemoryReport.reserved_oom, property)
