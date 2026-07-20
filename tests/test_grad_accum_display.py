"""非-PP 梯度累积（2026-07-20）：serve_explorer 展示 grad_accum 桶 + 梯度累积/有效batch 读数。

核心早已正确建模——num_microbatches≥2 时,每微批 F/B 后 reduced 梯度分片常驻(grad_accum 桶)直到
optimizer step。缺口全在 UI：`mbs` 对 pp=1 auto=1(无累积) 且 grad_accum/p2p_buf 此前不在 HTML 桶集
(BK)与标签(BKD)里 → 非-PP 梯度累积在网页上既配不了也看不见 → 欠估无从察觉。本组钉住修复。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve_explorer import eval_config, BK


def _peak_ga(r):
    """峰值断面(任一 stage 任一时刻)的最大 grad_accum 桶值（MiB）。"""
    return max((s["buckets"].get("grad_accum", 0)
                for st in r["stages"] for s in st["timeline"]), default=0)


def test_grad_accum_and_p2p_in_bucket_list():
    # 回归根因：BK 此前漏 grad_accum/p2p_buf → HTML 永不展示。
    assert "grad_accum" in BK and "p2p_buf" in BK


def test_pp1_gradient_accumulation_modeled_and_displayed():
    # pp=1 + 梯度累积步数 4：grad_accum 桶出现在 timeline，读数字段齐全。
    r = eval_config({"layers": "8", "dp": "2", "pp": "1", "batch": "1", "mbs": "4"})
    assert r["ok"]
    assert r["num_microbatches"] == 4 and r["grad_accum_steps"] == 4
    assert r["eff_batch"] == 1 * 2 * 4            # micro·dp·num_microbatches
    assert r["grad_accum_mib"] > 0               # 非-PP 梯度累积驻留被计入
    assert _peak_ga(r) == r["grad_accum_mib"]    # timeline 桶与读数一致（HTML 可见）


def test_pp1_no_accumulation_default_is_zero():
    # pp=1 默认(mbs 空 → auto=1)：无多步累积 → **峰值断面**不含梯度累积驻留（m=1 峰在 loss-BWD、
    #   此刻梯度尚未累计；反向末虽有 reduced 梯度驻留但不在峰上）→ grad_accum_mib(峰贡献)=0。
    r = eval_config({"layers": "8", "dp": "2", "pp": "1", "batch": "1"})
    assert r["ok"] and r["num_microbatches"] == 1
    assert r["grad_accum_mib"] == 0              # 峰值不含梯度累积（不叠加设备峰值 → 不致 OOM）


def test_accumulation_raises_peak_vs_no_accumulation():
    # 有累积 vs 无累积：设备峰值应升高 ≈ grad_accum（这正是此前欠估的量）。
    r1 = eval_config({"layers": "8", "dp": "2", "pp": "1", "batch": "1"})            # m=1
    r4 = eval_config({"layers": "8", "dp": "2", "pp": "1", "batch": "1", "mbs": "4"})  # m=4
    assert r4["device_peak"] > r1["device_peak"]
    assert abs((r4["device_peak"] - r1["device_peak"]) - r4["grad_accum_mib"]) < 1.0


def test_accumulation_m_independent_beyond_2():
    # 就地累积(AssignAdd)：m=4 与 m=8 驻留相同（不随步数线性增长）。
    r4 = eval_config({"layers": "8", "dp": "2", "pp": "1", "batch": "1", "mbs": "4"})
    r8 = eval_config({"layers": "8", "dp": "2", "pp": "1", "batch": "1", "mbs": "8"})
    assert r4["grad_accum_mib"] == r8["grad_accum_mib"]
    assert r8["eff_batch"] == 1 * 2 * 8          # 有效 batch 仍随步数增长
