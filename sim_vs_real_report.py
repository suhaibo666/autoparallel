"""新架构下 仿真器 vs 真机 综合差异报告（2026-07-09；2026-07-16 单一事实源重构）。

本地跑当前仿真器（含全部标定 margin + fp32-norm + 全部修复）对**全部已捕获真机锚点**，算
sim/real 比值 + 汇总（OOM-安全性、±5% 命中、最差欠/过预测）。无 NPU / 无网络。

**2026-07-16（Z2）**：锚点定义已抽到 `scorecard_anchors.py`（单一事实源），本脚本与 pytest 门
`tests/test_scorecard_anchors.py` 共同消费之，杜绝脚本/门漂移。真机值出处见各 DIAGNOSIS.md / §14。

**2026-07-16（D1）**：cp2-none / DSv3-8L-none 的无重算-MoE 欠预测已由 `nr_moe_frag_factor=0.6`
（mem_timeline 无重算-MoE OOM-安全标定 margin，pp==1 单 stage）补齐 → 1.021 / 1.009（修前
0.937 / 0.931，其中仅 ~196 MiB 是 F1 的 ln2 结构、余为 op 图粒度之下的碎片长尾，退 2 点标定）。
**2026-07-16（D2）**：mHC+MTP 锚点入卡（真机 21153.1，当前 0.920 欠预测）——历史上 MTP tie 修复
后由 1.088 翻转为欠预测、因未入卡而无人察觉；D1 margin 不覆盖它（DSv4 fused-CE），留待单独诊断。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import statistics

from scorecard_anchors import anchors

# ── 计算各锚点 sim 并输出 ──
print("\n## 新架构下 仿真器 vs 真机 —— 综合差异报告\n")
print("| 锚点 | regime | 仿真 MiB | 真机 MiB | ratio(sim/real) | OOM |")
print("|---|---|---|---|---|---|")
ratios = []
for a in anchors():
    sim = a.sim_fn()
    if sim is None:
        print(f"| {a.label} | {a.regime} | ERR | {a.real:.0f} | — | — |")
        continue
    ratio = sim / a.real
    ratios.append((a.label, ratio, sim, a.real))
    oom = "✅安全" if ratio >= 0.995 else ("⚠️欠" if ratio < 0.95 else "≈")
    print(f"| {a.label} | {a.regime} | {sim:.1f} | {a.real:.0f} | **{ratio:.3f}** | {oom} |")

# 汇总
rs = [r for _, r, _, _ in ratios]
absdev = [abs(r - 1) for r in rs]
within5 = sum(1 for r in rs if 0.95 <= r <= 1.05)
under = [(l, r) for l, r, _, _ in ratios if r < 0.95]
over = [(l, r) for l, r, _, _ in ratios if r > 1.05]
print(f"\n### 汇总（{len(rs)} 锚点）")
print(f"- 平均 |ratio−1| = **{statistics.mean(absdev)*100:.1f}%**；中位比值 {statistics.median(rs):.3f}")
print(f"- ratio ∈ [0.95,1.05]：**{within5}/{len(rs)}**")
print(f"- 最大过预测：{max(rs):.3f}（{[l for l,r,_,_ in ratios if r==max(rs)][0]}）")
print(f"- 最大欠预测：{min(rs):.3f}（{[l for l,r,_,_ in ratios if r==min(rs)][0]}）")
print(f"- OOM-不安全（欠预测 <0.95）：{under if under else '无'}")
print(f"- 过预测 >1.05：{over if over else '无'}")
