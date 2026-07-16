"""Z2（2026-07-16）：把 13+1 真机记分卡锚点包成**参数化 pytest 门**。

背景（Z2 缺陷）：`sim_vs_real_report.py` 是手动脚本，从不入 pytest 门 → 任一锚点比值漂移/翻转
都不会让测试变红。实证：mHC+MTP（D2）在 MTP tie 修复后由 1.088 过预测**翻转**为 0.920 欠预测，
因无门守卫而无人察觉。本门消灭该盲区。

口径：每锚点重算 sim（走与真机同一 Evaluator 路径，`scorecard_anchors.anchors()` 单一事实源），
断言 `ratio = sim/real ∈ band`。**欠预测侧比过预测侧更紧**——预测<真机会误报"放得下"却 OOM，是
记分卡最不能容忍的漂移方向。band 均足够紧，能让 D2 的 1.088↔0.920（0.168 摆幅）触红。

DSv4 依赖缺失 → 该锚点 sim_fn 返回 None → skip（不硬挂整门；镜像脚本 try/except 行为）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from scorecard_anchors import anchors

_ANCHORS = anchors()


@pytest.mark.parametrize("a", _ANCHORS, ids=[a.label for a in _ANCHORS])
def test_anchor_ratio_within_band(a):
    sim = a.sim_fn()
    if sim is None:
        pytest.skip(f"{a.label}: sim 依赖缺失（DSv4 align 未装）→ 跳过（镜像脚本 try/except）")
    ratio = sim / a.real
    lo, hi = a.band
    assert lo <= ratio <= hi, (
        f"锚点 {a.label!r}（{a.regime}）比值漂移：ratio={ratio:.4f} 越界 band=({lo}, {hi})；"
        f"sim={sim:.1f} MiB / real={a.real:.1f} MiB。{a.note or ''} "
        f"—— 欠侧漂移（<{lo}）是 OOM-不安全方向，过侧（>{hi}）是保守膨胀，二者都应查明再改带。")


def test_scorecard_has_no_oom_unsafe_dsv3_no_recompute():
    """D1 专项守卫：DSv3 无重算-MoE 两锚点（8L-none / cp2-none）必须 OOM-安全（ratio ≥ 1.0）。

    这是 D1 `nr_moe_frag_factor=0.6` margin 的直接回归门——若 margin 被误删/误关，二者回落到
    0.931/0.937（OOM-不安全），本测试立刻触红（区别于上面按 band 的通用漂移门）。
    """
    byname = {a.label: a for a in _ANCHORS}
    for label in ("DSv3 8L none (dp2)", "cp2-none (loss,k_ce=4)"):
        a = byname[label]
        sim = a.sim_fn()
        ratio = sim / a.real
        assert ratio >= 1.0, (
            f"{label}: ratio={ratio:.4f} < 1.0 → 无重算-MoE 回到 OOM-不安全欠预测；"
            f"D1 nr_moe_frag margin 是否被删/关？（sim={sim:.1f} / real={a.real:.1f}）")


def test_d2_flip_would_trip_mhc_mtp_band():
    """元测试：证明 mHC+MTP 的 band 能抓住 D2 那次 1.088→0.920 翻转（防"哑门"）。

    若 mHC+MTP 回到历史 1.088 过预测，比值必越出其 band 上界 → 触红。纯断言 band 语义，不跑模型。
    """
    a = next(x for x in _ANCHORS if x.label == "DSv4 mHC(x4)+MTP")
    lo, hi = a.band
    assert not (lo <= 1.088 <= hi), (
        f"mHC+MTP band=({lo},{hi}) 竟容纳历史翻转值 1.088 → 哑门，D2 翻转不会触红，请收紧上界。")
