# -*- coding: utf-8 -*-
"""185 真机 pp4+全重算 锚点门（2026-07-24 **口径切换：去经验补偿转纯理论 + 框架缺口暴露**）。

真机（185, 8×910B2, agent accbe10d6cc704d16 实测,交接文档 §11.1）：8 层 dsv4_hybrid **FUSED**
（DSA kernel + fused mHC + fused CE）、pp4（2层/stage）/dp2/ep2/tp1/cp1 八卡、seq4096、hidden4096、
8 专家（ep2→4/卡）、Muon、bf16、global_batch 8 = dp2×mbs1×4微批、mHC×4。

**口径切换（2026-07-24）**：评估器不再做经验 pin 补偿——全重算按 MindSpore checkpoint 理论语义
只留**每微批层入口 checkpoint_input**（bf16 128MiB/层，去 fused ctx 免疫 pin）。理论峰值因此
**显著低于**真机实测：差距 = mindformers+hyper_parallel+MindSpore 的**框架释放缺口**（真机每微批
层 ~1.9G 驻留、全重算实际只释 ~30% 激活；证据 pp4 ON−OFF 仅省 6.2/21.6GB；判决实验 E5/E1b 证
机制上应释放）——不吸收进数字，显式暴露。

本门为**双断言**：
  (a) 理论值 pin 住（防建模漂移，数值由 eval_config 现算写死）；
  (b) 框架缺口文档化（`assert 理论 < 真机`，缺口=真机−理论，属框架缺口非模型误差）。
无重算 OFF 锚点**不受口径切换影响**（全量 saves、无 pin/ci-dtype 影响峰值），band 原样保留。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serve_explorer as S

# ── 真机 alloc 锚点（MiB;绝不改动——实测值,交接文档 §11.1）────────────────────────────
REAL_ON = {0: 24153.3, 1: 14641.7, 2: 14097.7, 3: 23508.0}
REAL_OFF = {0: 30395.0, 1: 21019.4, 2: 17799.9, 3: 27822.0}

# ── 全重算 ON **纯理论值**（2026-07-24 口径切换后,pin 住防漂移）───────────────────────
# 全重算只留每微批层入口 checkpoint_input（bf16 128MiB/层）——无经验 ctx 免疫 pin。理论 << 真机,
# 差距=框架释放缺口（见 test_full_recompute_framework_gap）。改动建模而此值漂移即回归。
# 2026-07-24 §8.5 三结构桶(FSDP re-gather / Muon NS 反向重叠 / 全重算反向工作集)入账后上移
# (仅 pp>1 全重算反向;pp1 安全网不动)。理论仍 << 真机,差距=csa/indexer fp32 物化框架缺口。
THEO_ON = {0: 14551.0, 1: 9555.7, 2: 9478.5, 3: 21037.9}

# ── no-recompute OFF band（口径切换不影响 OFF:全量 saves,无 pin/ci-dtype 影响峰值）──────
# OFF 过估根因仍为 mHC ×n 记账（残差②,未修,与 scorecard mHC+MTP 锚 band 联动，不单独动）。
BAND_OFF = {0: (0.95, 1.34), 1: (0.95, 1.39), 2: (0.95, 1.25), 3: (0.90, 1.05)}

# 185 pp4 锚点配置（= run_sim_pp4.py BASE,§11.2 仿真基线同源）
_BASE = {
    "preset": "dsv4_flash", "attn": "dsv4_hybrid",
    "layers": "8", "seq": "4096", "batch": "1", "mtp": "0",
    "experts": "8", "topk": "2", "dense_k": "1",
    "heads": "64", "kv_groups": "1",
    "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
    "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
    "v_head": "512", "vocab": "129280",
    "hc": "4", "dsa_fused": "1", "ce_fused": "1",
    "dp": "2", "tp": "1", "ep": "2", "pp": "4", "cp": "1", "method": "colossal",
    "optimizer": "muon", "opt_dtype": "fp32", "grad_bytes": "4",
    "maxdev_gib": "58",
    "select": "attn", "sel_layers": "", "sel_ops": "", "sel_cfg": "", "vpp": "1", "mbs": "",
    "dp_replicate": "1", "reshard": "default", "cpu_offload": "0", "prefetch": "1", "sp": "",
}


def _stage_peaks(recompute: str) -> dict:
    p = dict(_BASE)
    p["recompute"] = recompute
    r = S.eval_config(p)
    assert r.get("ok"), r.get("errors")
    return {st["stage"]: st["peak"] for st in r["stages"]}


@pytest.fixture(scope="module")
def peaks_on():
    return _stage_peaks("full")


@pytest.fixture(scope="module")
def peaks_off():
    return _stage_peaks("None")


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_full_recompute_theoretical_value(peaks_on, stage):
    """(a) 纯理论值 pin 住防漂移：全重算 = 每微批层入口 checkpoint_input（bf16 128MiB）。"""
    sim = peaks_on[stage]
    assert abs(sim - THEO_ON[stage]) < 0.5, (
        f"全重算 ON stage{stage} 理论值漂移: sim={sim:.1f} vs pinned={THEO_ON[stage]:.1f}"
        f"——建模改动致纯理论口径漂移(非真机误差,重算此值并核查改动)。")


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_full_recompute_framework_gap(peaks_on, stage):
    """(b) 框架释放缺口文档化：理论 << 真机,差距为**框架缺口**非模型误差。

    缺口(真机−理论,MiB)：s0≈11985 / s1≈7301 / s2≈6867 / s3≈4165。
    = MS2.10 全重算实际只释 ~30% 激活、每微批层 ~1.9G 驻留（重算边界×在途深度 + mHC ×n 放大）。
    归属待 MS per-tensor 设备内存 API 终裁；判决实验 E5/E1b 证机制上应释放（报告§7.9）。"""
    sim, real = peaks_on[stage], REAL_ON[stage]
    assert sim < real, (
        f"全重算 ON stage{stage}: 理论 sim={sim:.1f} 应 < 真机 {real:.1f}"
        f"（纯理论只留重算边界；真机每微批层 ~1.9G 驻留=框架释放缺口，缺口={real-sim:.0f}MiB）。")


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_no_recompute_stage_anchor(peaks_off, stage):
    """OFF 锚点不受口径切换影响（全量 saves）——band 原样保留（残差②见模块 docstring）。"""
    sim, real = peaks_off[stage], REAL_OFF[stage]
    ratio = sim / real
    lo, hi = BAND_OFF[stage]
    assert lo <= ratio <= hi, (
        f"no-recompute OFF stage{stage}: sim={sim:.1f} vs 真机 alloc={real:.1f} MiB,"
        f" ratio={ratio:.3f} 越出 band=({lo},{hi})（mHC ×n 记账残差②）。")


def test_recompute_net_saving_direction(peaks_on, peaks_off):
    """方向性守卫：全重算 ON 仍低于 OFF（重算省内存）。纯理论口径下 OFF 过估（mHC ×n 记账残差,
    未修）而 ON 只留重算边界 → sim 净省远大于真机 6242（真机全重算仅释 ~30% 是框架缺口）。此处
    只守方向、不约束幅度（幅度差 = 框架缺口 + OFF 过估两来源，非模型误差）。"""
    saving = peaks_off[0] - peaks_on[0]
    assert saving > 0, "全重算竟比无重算更贵——纯理论口径建模错误"


def test_device_peak_framework_gap(peaks_on):
    """设备峰值 stage 是 OOM 判定的锚。纯理论口径：理论峰值显著低于真机峰值——差距=框架缺口，
    **OOM 判断勿直接采用理论值**（务必留框架缺口余量）。"""
    sim_peak = max(peaks_on.values())
    real_peak = max(REAL_ON.values())
    assert sim_peak < real_peak, (
        f"理论设备峰 {sim_peak:.1f} 应 < 真机峰 {real_peak:.1f}（框架释放缺口，"
        f"缺口={real_peak-sim_peak:.0f}MiB）——OOM 勿采用此理论值。")


# ═══════════════════════════════════════════════════════════════════════════════════════
# MTP 尾 stage（185 pp4+mtp=1 全重算）与 pp8 饱和探针（185 pp8）
# ═══════════════════════════════════════════════════════════════════════════════════════

# ── MTP 锚（185,pp4/dp2/ep2,fused dsv4,seq4096,全重算,num_nextn_predict_layers=1；
#    log_dsv4h_pp4_mtp）:真机 mtp=1 仅尾 stage 净增 +16390,s0-s2 与无 MTP 锚逐 MiB 重合 ──────
REAL_MTP = {0: 24153.0, 1: 14641.0, 2: 14100.0, 3: 39898.0}
# 纯理论值：mtp_resident=0（代码核查 116 multi_token_prediction.py：`_MTPLossAutoScaler` 逐微批
#   attach、`save_to_mtp_losses_tracker` 只累加 `.detach()` 标量 → 无步内跨微批驻留，见 mem_timeline
#   注释）。s0-s2 与无 MTP 理论一致；s3(尾)仅多 MTP decoder 层自身（非 loss 链步内驻留）。
THEO_MTP = {0: 14551.0, 1: 9555.7, 2: 9478.5, 3: 24360.3}  # 2026-07-24 §8.5 三桶入账


@pytest.fixture(scope="module")
def peaks_mtp():
    p = dict(_BASE)
    p.update({"mtp": "1", "recompute": "full", "pp_split": "2,2,2,3"})   # MTP 归尾 stage(真机口径)
    r = S.eval_config(p)
    assert r.get("ok"), r.get("errors")
    return {st["stage"]: st["peak"] for st in r["stages"]}


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_mtp_tail_stage_theoretical(peaks_mtp, stage):
    """(a) MTP 纯理论值 pin 住（mtp_resident=0，只多 MTP decoder 层自身）。"""
    sim = peaks_mtp[stage]
    assert abs(sim - THEO_MTP[stage]) < 0.5, (
        f"pp4+MTP ON stage{stage} 理论漂移: sim={sim:.1f} vs {THEO_MTP[stage]:.1f}")


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_mtp_tail_stage_framework_gap(peaks_mtp, stage):
    """(b) 框架缺口：真机尾 stage +16.4GB = MS 未及时释放逐微批 MTP loss 反向图的 m 份累积，
    属**框架释放缺口**（非模型误差；代码语义上逐微批 attach、随该微批反向释放）。"""
    sim, real = peaks_mtp[stage], REAL_MTP[stage]
    assert sim < real, (
        f"pp4+MTP stage{stage}: 理论 {sim:.1f} 应 < 真机 {real:.1f}。尾 stage 缺口 "
        f"{REAL_MTP[3]-THEO_MTP[3]:.0f}MiB = MS 未及时释放逐微批 MTP 反向图(框架缺口)。")


def test_mtp_front_stages_unchanged(peaks_mtp, peaks_on):
    """真机:加 MTP 后 s0-s2 逐 MiB 不变——仿真同款守卫（mtp_resident=0，尾 stage 才多 MTP 层）。"""
    for s in (0, 1, 2):
        assert abs(peaks_mtp[s] - peaks_on[s]) < 1.0, (
            f"stage{s}: mtp=1({peaks_mtp[s]:.1f}) ≠ mtp=0({peaks_on[s]:.1f}) —— "
            f"MTP 驻留泄漏到前部 stage")


# ── pp8 饱和探针锚（185,fused dsv4 8L,pp8/dp1/ep1,1层/stage,seq4096,全重算,m=8）────────
REAL_PP8 = {0: 24759.0, 1: 12324.0, 2: 11678.0, 3: 11919.0,
            4: 10867.0, 5: 11113.0, 6: 10074.0, 7: 26449.0}
# 纯理论值（每微批层入口 ci 线性 pin 到 warmup 深度；无经验饱和 cap——cap 是死 ctx 的经验回收模型,
#   纯理论口径无免疫 ctx 可回收）。理论 << 真机，差距=框架释放缺口。
# 2026-07-24 §8.5 三结构桶入账后上移(s0 无 BWD-SEND 不叠 re-gather 窗故仅 muon+bwd_ws;中部叠满)。
THEO_PP8 = {0: 19697.7, 1: 10625.4, 2: 10237.8, 3: 10028.3,
            4: 10241.4, 5: 9853.8, 6: 9644.3, 7: 25304.9}


@pytest.fixture(scope="module")
def peaks_pp8():
    p = dict(_BASE)
    p.update({"dp": "1", "ep": "1", "pp": "8", "mbs": "8", "recompute": "full"})
    r = S.eval_config(p)
    assert r.get("ok"), r.get("errors")
    return {st["stage"]: st["peak"] for st in r["stages"]}


@pytest.mark.parametrize("stage", list(range(8)))
def test_pp8_theoretical_value(peaks_pp8, stage):
    """(a) pp8 纯理论值 pin 住防漂移。"""
    sim = peaks_pp8[stage]
    assert abs(sim - THEO_PP8[stage]) < 0.5, (
        f"pp8 ON stage{stage} 理论漂移: sim={sim:.1f} vs {THEO_PP8[stage]:.1f}")


@pytest.mark.parametrize("stage", list(range(8)))
def test_pp8_framework_gap(peaks_pp8, stage):
    """(b) 框架缺口：理论 << 真机（真机中部 steady 段死 ctx 未及时回收 + warmup 驻留=框架缺口）。"""
    sim, real = peaks_pp8[stage], REAL_PP8[stage]
    assert sim < real, (
        f"pp8 ON stage{stage}: 理论 {sim:.1f} 应 < 真机 {real:.1f}（框架释放缺口，缺口={real-sim:.0f}MiB）。")
