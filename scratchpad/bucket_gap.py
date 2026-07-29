"""逐桶差值表：模型各桶 vs 真机可核验量。

真机只给标量峰值，所以「逐桶对账」只能这样做：
  真机峰值 − Σ(模型其余桶) = 反解出的「该桶真机值」
只有当**某一个桶**被独立测量过时，这个反解才有意义。2026-07-29 的逐事件轨迹
正好独立测出了 run c 的 `act_live`（在世微批数 × 逐层增量），所以 run c 可以真正逐桶对账。
"""
import os
import sys
import json
import warnings

warnings.simplefilter("ignore")
REPO = r"E:\97-codes\torch_parallel\pynative-cost-evaluator"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import liveness_ab_validate as G  # noqa: E402
from cost_eval.report import Evaluator  # noqa: E402

MiB = 2 ** 20
FIELDS = ("persistent", "act_live", "gather_buf", "grad_buf", "recomp_scratch", "remat_saves",
          "bwd_scratch", "bwd_working_set", "swap_buf", "workspace", "optstep",
          "framework_reserve", "kept_frag", "grad_accum", "p2p_buf", "mtp_resident")

# ── 真机实测（167, 2026-07-29 逐事件轨迹, run c = fused / 无重算 / L8 / m4）──────────
# 峰值时刻「在世微批数」与「逐层前向增量(MiB)」——直接数出来的，不是反解
REAL_C_DEPTH = {0: 4, 1: 3, 2: 2, 3: 1}
REAL_C_PER_LAYER_SUM = {0: 4590.4, 1: 4478.2, 2: 4475.0, 3: 4404.7}   # 每 stage 两层之和
REAL_C_PEAK = {0: 30391.6, 1: 21019.4, 2: 17759.1, 3: 27720.1}


def model_run(letter):
    v = {x.letter: x for x in G.VARIANTS}[letter]
    mf = G.derive_mf_config(G.DEFAULT_BASE_DIR, v)
    b, spec = G.build_bundle(mf)
    rep = Evaluator(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                    check_feasibility=False).evaluate(record_timeline=True)
    return {i: {"peak": sp.peak_bytes / MiB, "ev": sp.peak_event,
                "bd": {f: getattr(sp.breakdown, f, 0) / MiB for f in FIELDS
                       if getattr(sp.breakdown, f, 0)}}
            for i, sp in enumerate(rep.per_stage)}


print("=" * 104)
print("run c（fused / 无重算 / L8 / m4）—— 唯一能真正逐桶对账的一跑")
print("  因为 2026-07-29 逐事件轨迹**独立测出了 act_live**（在世微批数 × 逐层增量）")
print("=" * 104)
c = model_run("c")
for st in range(4):
    bd = c[st]["bd"]
    real_act = REAL_C_DEPTH[st] * REAL_C_PER_LAYER_SUM[st]
    other = sum(v for k, v in bd.items() if k != "act_live")
    m_act = bd.get("act_live", 0.0)
    explained = other + real_act
    resid = REAL_C_PEAK[st] - explained
    print("\n── stage%d ──  模型峰值 %.1f   真机峰值 %.1f   事件 %s"
          % (st, c[st]["peak"], REAL_C_PEAK[st], c[st]["ev"]))
    print("   %-18s %12s %12s %12s   %s" % ("桶", "模型", "真机", "差(真机-模型)", "核验方式"))
    print("   %-18s %12.1f %12.1f %12.1f   实测(深度%d × 逐层%.1f)"
          % ("act_live", m_act, real_act, real_act - m_act, REAL_C_DEPTH[st], REAL_C_PER_LAYER_SUM[st]))
    for k, v in sorted(bd.items(), key=lambda kv: -kv[1]):
        if k == "act_live":
            continue
        print("   %-18s %12.1f %12s %12s   未独立测（下方按合计核）" % (k, v, "-", "-"))
    print("   %-18s %12.1f %12.1f %12.1f   ← 其余桶合计"
          % ("[其余桶合计]", other, REAL_C_PEAK[st] - real_act, (REAL_C_PEAK[st] - real_act) - other))
    print("   %-18s %12.1f %12.1f %12.1f   残差 = %.1f%%"
          % ("合计", c[st]["peak"], REAL_C_PEAK[st], resid, 100.0 * resid / REAL_C_PEAK[st]))
    if m_act:
        print("   → act_live 模型/真机 = %.3f×" % (m_act / real_act))

print()
print("=" * 104)
print("有重算的两跑：真机未逐桶测，只能看模型侧哪些桶在动（差值住在这里）")
print("=" * 104)
a, b = model_run("a"), model_run("b")
real_a = {0: 24153.3, 1: 14641.7, 2: 14097.7, 3: 23508.0}
real_b = {0: 50187.9, 1: 43940.0, 2: 43407.1, 3: 47888.1}
for st in range(4):
    ka, kb = a[st]["bd"], b[st]["bd"]
    keys = sorted(set(ka) | set(kb), key=lambda k: -(kb.get(k, 0) - ka.get(k, 0)))
    print("\n── stage%d ──  a(fused) 模型 %.1f/真机 %.1f    b(unfused) 模型 %.1f/真机 %.1f"
          % (st, a[st]["peak"], real_a[st], b[st]["peak"], real_b[st]))
    print("   %-18s %11s %11s %11s" % ("桶", "a模型", "b模型", "b−a"))
    for k in keys:
        d = kb.get(k, 0) - ka.get(k, 0)
        if abs(d) < 0.05 and not ka.get(k) and not kb.get(k):
            continue
        print("   %-18s %11.1f %11.1f %11.1f" % (k, ka.get(k, 0), kb.get(k, 0), d))
    print("   %-18s %11.1f %11.1f %11.1f   真机 b−a = %.1f"
          % ("合计", a[st]["peak"], b[st]["peak"], b[st]["peak"] - a[st]["peak"],
             real_b[st] - real_a[st]))
