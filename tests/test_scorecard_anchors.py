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


#: DSv3 无重算-MoE 两锚点**关掉 D1 margin**（`DimTable.nr_moe_frag_factor=0`）后的比值。
#: **实测**（`scratchpad/probe_d1_margin_off.py`，2026-07-30 `K_CE` 重标定后重测）：
#:   DSv3 8L none (dp2)     ON 18629.5/0.9330  OFF 17268.5/0.8648（real 19967.3）
#:   cp2-none (loss,k_ce=3) ON 18971.1/0.9429  OFF 17492.5/0.8694（real 20119.4）
#: （2026-07-29 那次的对照点是 OFF 0.9130 / 0.9172，随 `K_CE` 4→3 各下移一张满 vocab
#:  fp32 平面 = 2020.0 MiB —— 平面在 ON/OFF 两侧同幅下移，故 `on − off` 的**幅度不变**，
#:  margin 的存在性判据因此**不受本次重标定影响**。）
#: 这是 `test_d1_margin_is_present_and_effective` 的对照点：margin 的**存在与有效性**由
#: 「开 > 关」证明，而不再由「开 ≥ 1.0」证明（后者在 2026-07-29 已不成立，见该测试 docstring）。
_D1_OFF_RATIO = {"DSv3 8L none (dp2)": 0.8648, "cp2-none (loss,k_ce=3)": 0.8694}


def test_d1_margin_is_present_and_effective():
    """D1 专项守卫（**2026-07-29 二次重钉：断言从「≥1.0」改为「margin 开 > 关，且幅度钉住」**）。

    **同一条不变量、换了举例方式**。原断言：DSv3 无重算-MoE 两锚点（8L-none / cp2-none）
    ratio ≥ 1.0，用来证明 D1 `nr_moe_frag_factor=0.6` margin 没被误删/误关。
    2026-07-29 起该**举例**失效：`FusedRMSNorm` 不 cast（`layer_norm.py:151-155`，`:149` 的
    self.cast 是死属性）被修正后，两锚点落到 **0.9812 / 0.9907**——**OOM-不安全**。
    （margin 关掉时是 0.9130 / 0.9172，实测；margin 本身仍是 0.6，一个字节没动。）

    **不调参掩盖**：margin 一个字节没动（仍 0.6），是普查去掉了一处**真实的过读**，而那处过读
    此前正好在掩盖别处的欠读（本项目反复出现的「相消误差」）。真正的补法是 kernel workspace 项，
    需真机 profiler 明细，不是把 margin 往上标。

    故本门保留**机制**断言（margin 必须在、必须有效、幅度必须钉住），并把两个锚点的
    **OOM-不安全**状态如实写进断言里——它们再欠一分就会红。
    """
    byname = {a.label: a for a in _ANCHORS}
    for label, off_ratio in _D1_OFF_RATIO.items():
        a = byname[label]
        on = a.sim_fn() / a.real
        # ① margin 仍**有效**：开着比关掉高（若被误删/误关，二者相等 → 立刻触红）。
        #    off_ratio 是实测记录值（见上方常量注释），不是拟合。
        assert on > off_ratio + 0.02, (
            f"{label}: margin 开 ratio={on:.4f} 未显著高于关 ratio={off_ratio:.4f} → "
            f"D1 nr_moe_frag margin 疑似被删/关。")
        # ② 幅度钉住（防漂移，也防有人把 margin 悄悄调大去凑 ≥1.0）。
        lo, hi = a.band
        assert lo <= on <= hi, f"{label}: ratio={on:.4f} 越出 band=({lo},{hi})"
        # ③ **如实记账**（2026-07-30 举例第二次更新，不变量原样）：2026-07-29 起二者都在
        #    OOM-**不安全**侧（0.981 / 0.991）；同日 `lm_head` 反向 kernel workspace 实测入账
        #    把它们翻到 OOM-**安全**侧（1.034 / 1.043）；紧接着的 `K_CE` 重标定（4→3，
        #    `docs/k_ce_recalibration_2026-07-30.md`）又把它们**翻回 OOM-不安全侧**
        #    （**0.9330 / 0.9429**）。不变量仍是「不得靠调大标定 margin 凑」——margin 一个
        #    字节没动（仍 0.6，由 ① 的 on>off 断言守住）；两次翻转都来自**台账级证据**
        #    （前者是 kernel workspace 逐字节实测，后者是 9 份仓内 profiler 明细逐块清点）。
        #    举例因此第二次搬家：从「过读带 (1.02,1.05)」换成「欠读带 (0.92,0.95)」，两侧都守。
        #    **这一侧不放宽方向**：下界防欠读继续恶化，上界防有人把 K_CE 或 margin 调回去凑。
        assert 0.92 <= on <= 0.95, (
            f"{label}: ratio={on:.4f} 越出已记录的欠读带 (0.92, 0.95)。<0.92 说明欠读继续恶化，"
            f"须查明（缺口方向见 docs/k_ce_recalibration_2026-07-30.md §9）；>0.95 说明有人"
            f"把 K_CE 调回 4 或把 margin 调大去凑 OOM-安全——请核对是不是又加了拟合常数。"
            f"任何一侧都须查明再改带，不得靠调 margin 凑。")


def test_d2_flip_would_trip_mhc_mtp_band():
    """元测试：证明 mHC+MTP 的 band 能抓住 D2 那次 1.088→0.920 翻转（防"哑门"）。

    若 mHC+MTP 回到历史 1.088 过预测，比值必越出其 band 上界 → 触红。纯断言 band 语义，不跑模型。
    """
    a = next(x for x in _ANCHORS if x.label == "DSv4 mHC(x4)+MTP")
    lo, hi = a.band
    assert not (lo <= 1.088 <= hi), (
        f"mHC+MTP band=({lo},{hi}) 竟容纳历史翻转值 1.088 → 哑门，D2 翻转不会触红，请收紧上界。")
