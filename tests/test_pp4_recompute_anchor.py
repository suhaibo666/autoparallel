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

**2026-07-25 `remat_saves`（重算再物化的 saved 集）入账**：pp4 四 stage 与 pp8 s0/s7 的理论值
上移、缺口**收窄**且方向不变（仍欠读真机）；**pp8 中部 s1-s6 方向翻转为过读 1.14~1.31×**——
1 层/stage 极端配置下单层 `A−ci`≈3.5GB 占断面 26%，叠上 `recomp_scratch`/`bwd_working_set` 的
`forward_max_live` 重叠。**如实记录、未反向调参**，见 THEO_PP8 与 `test_pp8_framework_gap`。
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
# ── 2026-07-25 `remat_saves`(重算再物化的 saved 集)入账后再上移 ──────────────────────────
#   s0 14551.0→18172.2 / s1 9555.7→13049.0 / s2 9478.5→12975.9 / s3 21037.9→**不变**。
#   机理:重算区域反向重跑 forward 时 backward 需要的整个 `activation_saves` 集必须同时在世
#   (`max(0,A−ci)`,fused dsv4 层 A−ci≈3.5GB),此前记 0。逐层赋在该层 bwd@lid 事件 → ×1。
#   s3(尾 stage)峰在 **loss/head 层**(非重算层)的 bwd 事件 → 该事件此桶 0 → 逐 MiB 不变(真实,
#   不是被门挡掉:重算层 bwd 事件确实涨了,只是仍不及 loss 层断面)。
#   方向:与真机比 s0 0.602→0.752 / s1 0.653→0.891 / s2 0.672→0.920 / s3 0.895 不变——**收窄**,
#   四个 stage 仍全部 sim < real（框架缺口方向不变）。

# **2026-07-29 重钉**（`docs/census_arbitration_2026-07-29.md`）：手写普查按权威快照逐条订正
# （q_hnorm/cg fp32→bf16、去伪 norm 抬升、去 inv_rope_out、补前向 rope 保留对、cmp_residual
# 改 int32 标量、补 sinks/sparse_indices、idx_weights→fp32）→ 每层 saves 降 ~901 MiB@seq4096。
# 不变量一条没动（理论 << 真机的方向、×1 结构、真机常数），只移动样例值。
# 2026-07-29 二次重钉（`docs/census_fix_mhc_rmsnorm_2026-07-29.md`）：融合 mHC ctx 按源建
# （−421.8 MiB/层）+ FusedRMSNorm 不 cast（−268.0 MiB/层）→ `remat_saves` 同源再降。
# s0 16766.2→16498.2 / s1 11643.0→11375.0 / s2 11565.9→11297.9 / s3 21037.9→21005.9
# （s3 峰在 head/loss 段，只吃到 lm_head 的 final_norm −32.0）。不变量（理论 << 真机）不动。
# 2026-07-29 三次重钉（`docs/census_fix_residual_carrier_2026-07-29.md`）：
# ⚠ **本锚点走的是**非融合** mHC 分支** —— `serve_explorer.parse_and_validate` 没有
#   `use_fused_mhc` 旋钮（对照 `dsa_fused`/`ce_fused` 都有），故恒取 `LLMConfig` 默认 False；
#   而本文件 docstring 逐字写的真机是「FUSED（DSA kernel + **fused mHC** + fused CE）」，
#   站点 yaml 亦 `use_fused_mhc: true`（`ab_fusion_2026-07-25/dsv4h_*_pp4_recomp.yaml:109`）。
#   **配置错配**（上一轮加融合门时留下、此前不可见），非本轮引入 —— 见该文档 §5，含 what-if 数。
#   本轮 ④（非融合分支自己的三份 fp32 副本，`hyper_connection.py:298`/`:109`/`:120`）因此**只**
#   打在这条错配路径上：+480 MiB/模块 → +960 MiB/层。
# s0 16498.2→17458.2 / s1 11375.0→12239.0 / s2 11297.9→12161.9 / s3 21005.9→**不变**
#   （s3 峰在 head/loss 段，不含解码层 saves）。四 stage 仍全部 sim < real（不变量方向不动）。
THEO_ON = {0: 17458.2, 1: 12239.0, 2: 12161.9, 3: 21005.9}

# ── no-recompute OFF band（口径切换不影响 OFF:全量 saves,无 pin/ci-dtype 影响峰值）──────
# OFF 过估根因仍为 mHC ×n 记账（残差②,未修,与 scorecard mHC+MTP 锚 band 联动，不单独动）。
# 2026-07-29 重钉：普查订正后 ratio = 1.045 / 1.073 / 0.977 / **0.885**。前三个 stage
#   由 1.3x 过读**收敛到 ~1.0**（这正是本次修正的目标）；s3（head/loss 段主导）由 0.97
#   落到 0.885 —— **OOM-不安全方向，如实记带、不调参掩盖**（残差见仲裁 §3）。
# 2026-07-29 二次重钉：ratio = 0.975 / 0.996 / 0.917 / 0.864（前次 1.045/1.073/0.977/0.885）。
#   s0/s1 由过读收到 ~1.0（**目标达成**）；s0/s2/s3 已转**欠读 = OOM-不安全**——如实记带，
#   不调参掩盖。残差 = kernel workspace（快照读不出，需真机 profiler 明细）。
# 2026-07-29 三次重钉：ratio = **1.215 / 1.243 / 1.111 / 0.927**（前次 0.975/0.996/0.917/0.864）。
#   全部来自 ④ 打在**非融合**分支上（见 THEO_ON 上方的配置错配说明）：+960 MiB/层。
#   s0-s2 由欠读**翻回过读**（OOM 安全侧，但**不准** —— 它准不准取决于错配是否被修）；
#   s3 仍欠读 0.927。**不调参掩盖**：把错配后的真实读数如实钉住，修错配时本门会红并被重钉。
#   对照 what-if（探针 `scratchpad/probe_pp4_fused_mhc_whatif.py`，把 `use_fused_mhc` 打开）：
#   0.868 / 0.867 / 0.815 / 0.832 —— 与逐层直测（0.89–0.99×）同侧同量级。
BAND_OFF = {0: (1.16, 1.27), 1: (1.19, 1.29), 2: (1.06, 1.16), 3: (0.88, 0.97)}

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
# 2026-07-24 §8.5 三桶入账;**2026-07-25 `remat_saves` 入账**再上移:s0-s2 同 THEO_ON
#   (14551.0/9555.7/9478.5 → 18172.2/13049.0/12975.9)、s3 24360.3 → 29276.5(尾 stage 的 MTP
#   decoder 层是重算层,其 bwd 事件此刻成为该 stage 峰 → 带上 A−ci)。四 stage 仍全部 sim < real
#   (s3 29276.5 vs 39898 → 0.734,缺口收窄自 0.611)。
# 2026-07-29 重钉（同 THEO_ON 的普查订正）：s0-s2 同 THEO_ON；s3 29276.5→28375.0。
# 2026-07-29 二次重钉：s0-s2 同 THEO_ON；s3 28375.0→28011.0。
# 2026-07-29 三次重钉：s0-s2 同 THEO_ON；s3 28011.0→28875.0（同一 ④，见 THEO_ON 上方说明）。
THEO_MTP = {0: 17458.2, 1: 12239.0, 2: 12161.9, 3: 28875.0}


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
#   纯理论口径无免疫 ctx 可回收）。
# 2026-07-24 §8.5 三结构桶入账后上移(s0 无 BWD-SEND 不叠 re-gather 窗故仅 muon+bwd_ws;中部叠满)。
# ── 2026-07-25 `remat_saves` 入账后再上移(每 stage +2817~3656 = 该 stage 单个 dsv4 层的 A−ci)──
#   s0 19697.7→22514.6 / s1 10625.4→14281.2 / s2 10237.8→13815.1 / s3 10028.3→13601.6 /
#   s4 10241.4→13897.2 / s5 9853.8→13431.1 / s6 9644.3→13217.6 / s7 25304.9→**不变**(峰在
#   loss/head 层 bwd 事件,非重算层)。
#   ⚠ **方向翻转(如实记录,未做任何反向调参)**:pp8 是 **1 层/stage** 的极端配置——中部 stage
#   的 persistent/act_live 都很小(s3 persistent 仅 2551),该层 A−ci≈3.5GB 一入账就占了断面的
#   26%,于是 s1-s6 从「欠读真机」翻成「**过读**真机」1.14~1.31×(见 test_pp8_framework_gap 的
#   逐 stage 记录)。s0/s7 仍欠读。成因两条:①本项与 `recomp_scratch`(fml−ci)/`bwd_working_set`
#   (fml−bwd_scratch)存在**部分重叠**(fml 里含一部分 saved 张量;s3 两项合计仅 1216MiB,不足以
#   解释全部 1683MiB 过读);②pp8 中部 stage 的 Σ re-gather + Muon NS 三桶在 1 层/stage 下也偏
#   保守。**按纪律不调参掩盖**,如实钉住并在 framework_gap 门里逐 stage 记录方向。
# 2026-07-29 重钉（同上普查订正）：s0 22514.6→21108.6 / s1 14281.2→12867.7 /
#   s2 13815.1→12405.1 / s3 13601.6→12195.6 / s4 13897.2→12483.7 / s5 13431.1→12021.1 /
#   s6 13217.6→11811.6 / s7 25304.9→**不变**（峰在 head/loss 段，不含解码层 saves）。
# 2026-07-29 二次重钉：s0 21108.6→20840.6 / s1 12867.7→12599.7 / s2 12405.1→12137.1 /
#   s3 12195.6→11927.6 / s4 12483.7→12215.7 / s5 12021.1→11753.1 / s6 11811.6→11543.6 /
#   s7 25304.9→25272.9（峰在 head/loss 段，只吃 final_norm −32.0）。
# 2026-07-29 三次重钉：s0 20840.6→21800.6 / s1 12599.7→13463.7 / s2 12137.1→13001.1 /
#   s3 11927.6→12791.6 / s4 12215.7→13079.7 / s5 11753.1→12617.1 / s6 11543.6→12407.6 /
#   s7 25272.9→**不变**（峰在 head/loss 段）。同一 ④ + 同一配置错配（见 THEO_ON 上方说明）。
THEO_PP8 = {0: 21800.6, 1: 13463.7, 2: 13001.1, 3: 12791.6,
            4: 13079.7, 5: 12617.1, 6: 12407.6, 7: 25272.9}


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


# pp8 逐 stage 方向（2026-07-25 `remat_saves` 入账后重测；**如实记录，未反向调参**）：
#   s0/s7 仍 **欠读**真机（框架释放缺口方向不变，0.909 / 0.957）；
#   s1-s6 **翻成过读** 1.14~1.31×——1 层/stage 极端配置下该层 A−ci≈3.5GB 一入账即占断面 26%，
#   叠上 recomp_scratch/bwd_working_set 的 fml 重叠（s3 合计 1216MiB）与 Σre-gather/MuonNS 的
#   保守性。OOM 方向上过读是安全侧，但**不准**——留待真机逐桶 micro-anchor 细化重叠扣减。
_PP8_UNDER = {0, 7}          # 仍 sim < real 的 stage
# 2026-07-29 重钉：1.10–1.35 → **1.00–1.25**。普查订正把过读幅度压掉大半（s1 1.16→1.044、
#   s2 1.18→1.062、s3 1.14→1.023、s5 1.21→1.082），方向仍是过读（OOM 安全侧）。
# 2026-07-29 二次重钉：1.00–1.25 → **1.00–1.20**。实测 s1 1.022 / s2 1.039 / **s3 1.0007** /
#   s4 1.124 / s5 1.058 / s6 1.146。⚠ **s3 只高出真机 8.6 MiB（1.0007），已在翻转边缘**——
#   下界刻意保持 1.00：它真翻成欠读时本门必须变红（那是 OOM-不安全方向，须显式记账而非静默通过）。
# 2026-07-29 三次重钉：1.00–1.20 → **1.00–1.25**。实测 s1 1.093 / s2 1.113 / s3 1.073 /
#   s4 1.204 / s5 1.135 / s6 1.232。⚠ 上一轮记的「s3 只剩 1.0007、在翻转边缘」**已远离边缘**
#   （1.0007→1.073），但那是 ④ 打在错配分支上的结果，不是精度变好；错配一修就会退回边缘。
#   下界仍刻意保持 1.00（真翻成欠读必须变红）。
_PP8_OVER_BAND = (1.00, 1.25)  # 翻成过读的 stage 的 ratio 带（钉住幅度，防再漂）


@pytest.mark.parametrize("stage", list(range(8)))
def test_pp8_framework_gap(peaks_pp8, stage):
    """(b) 理论 vs 真机方向门（2026-07-25 起**分两档**，如实记录 remat 入账后的方向翻转）。

    s0/s7：仍 `sim < real`——框架释放缺口（真机中部 steady 段死 ctx 未及时回收 + warmup 驻留）。
    s1-s6：`sim > real` 1.14~1.31×——**方向已翻转**（成因见 THEO_PP8 注释；OOM 安全侧但不准，
      不做反向调参掩盖）。此门把翻转**钉死**：既守它没继续恶化，也守它没被偷偷"调回去"。"""
    sim, real = peaks_pp8[stage], REAL_PP8[stage]
    if stage in _PP8_UNDER:
        assert sim < real, (
            f"pp8 ON stage{stage}: 理论 {sim:.1f} 应 < 真机 {real:.1f}"
            f"（框架释放缺口，缺口={real-sim:.0f}MiB）。")
    else:
        lo, hi = _PP8_OVER_BAND
        assert lo <= sim / real <= hi, (
            f"pp8 ON stage{stage}: 理论 {sim:.1f} vs 真机 {real:.1f}, ratio={sim/real:.3f} 越出"
            f" 已记录的过读带 ({lo},{hi})——2026-07-25 remat 入账使 1 层/stage 中部 stage 由欠读"
            f"翻为过读（fml 重叠 + Σre-gather/MuonNS 保守）。此为如实记录,勿调参掩盖。")
