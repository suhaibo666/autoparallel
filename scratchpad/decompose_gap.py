"""把「仿真器 vs 真机」的差值拆开：绝对残差 → 物理对比量 → 桶截面归因。

方法：8 组真机跑每次只改一个因子，故任意两跑之差 **隔离出一个物理量**。
比较「模型的差」与「真机的差」，就把总残差归因到具体机制上，而不是只报一个总比值。
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
GATE_JSON = (r"C:\Users\suhaibo\AppData\Local\Temp\claude\E--97-codes-torch-parallel"
             r"\2e0255a0-6a8f-4ca7-9d03-0a9d1bc6d246\scratchpad\gate.json")

BUCKET_FIELDS = ("persistent", "act_live", "gather_buf", "grad_buf", "recomp_scratch",
                 "remat_saves", "bwd_scratch", "bwd_working_set", "swap_buf", "workspace",
                 "optstep", "framework_reserve", "kept_frag", "grad_accum", "p2p_buf",
                 "mtp_resident")

LET = {v.letter: v for v in G.VARIANTS}


def bucket_run(letter):
    """跑 bucket 路径，取每 stage 峰值 + 峰值事件的桶截面。"""
    v = LET[letter]
    mf = G.derive_mf_config(G.DEFAULT_BASE_DIR, v)
    b, spec = G.build_bundle(mf)
    rep = Evaluator(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                    check_feasibility=False).evaluate(record_timeline=True)
    out = {}
    for i, sp in enumerate(rep.per_stage):
        bd = {f: round(getattr(sp.breakdown, f, 0) / MiB, 1) for f in BUCKET_FIELDS
              if getattr(sp.breakdown, f, 0)}
        out[i] = {"peak": round(sp.peak_bytes / MiB, 1), "event": sp.peak_event, "bd": bd}
    return out


gate = json.load(open(GATE_JSON, encoding="utf-8"))
runs = gate["runs"]
tag_of = {t.split()[0]: t for t in runs}


def series(letter, key):
    r = runs[tag_of[letter]][key]
    return {int(k): v for k, v in r.items()}


print("=" * 108)
print("A. 绝对残差  real − sim   (MiB;  正=模型欠读, 负=模型过读)")
print("=" * 108)
print("%-22s %-6s %10s %10s %10s %12s %12s" %
      ("run", "stage", "real", "bucket", "hand_spec", "real-bucket", "real-hand"))
resid = {}
for letter in "abcefgh":                       # d 真机 OOM，不可评分
    tag = tag_of[letter]
    real, buc, han = series(letter, "real_MiB"), series(letter, "bucket_MiB"), series(letter, "hand_spec_MiB")
    for st in range(4):
        rv, bv, hv = real.get(st), buc.get(st), han.get(st)
        if rv is None:
            continue
        resid[(letter, st)] = (rv - bv, rv - hv)
        print("%-22s %-6d %10.1f %10.1f %10.1f %12.1f %12.1f"
              % (tag, st, rv, bv, hv, rv - bv, rv - hv))

# ── B. 物理对比量：每个差隔离一个机制 ────────────────────────────────────────
CONTRASTS = (
    ("重算节省   (fused: OFF−ON)", "c", "a"),
    ("unfused 物化 @L8 (b−a)",     "b", "a"),
    ("unfused 物化 @L4 (h−g)",     "h", "g"),
    ("微批 4→8   (e−a)",           "e", "a"),
    ("层数 4→8   (a−g)",           "a", "g"),
)
print()
print("=" * 108)
print("B. 物理对比量：模型 vs 真机（误差 = 模型差 − 真机差；正=模型高估该机制）")
print("=" * 108)
print("%-28s %-6s %11s %11s %11s %11s %11s" %
      ("对比量", "stage", "real_d", "bucket_d", "hand_d", "err_buc", "err_hand"))
contrast_err = {}
for name, hi, lo in CONTRASTS:
    for st in range(4):
        try:
            rd = series(hi, "real_MiB")[st] - series(lo, "real_MiB")[st]
            bd = series(hi, "bucket_MiB")[st] - series(lo, "bucket_MiB")[st]
            hd = series(hi, "hand_spec_MiB")[st] - series(lo, "hand_spec_MiB")[st]
        except (KeyError, TypeError):
            continue
        contrast_err[(name, st)] = (bd - rd, hd - rd)
        print("%-28s %-6d %11.1f %11.1f %11.1f %11.1f %11.1f"
              % (name, st, rd, bd, hd, bd - rd, hd - rd))
    print("-" * 108)

# ── C. 残差传播：某跑的残差 = 基线残差 + 该对比量的误差(取负) ────────────────
print()
print("=" * 108)
print("C. 残差归因：run b(unfused ON) 的欠读，多少继承自 fused 基线、多少来自 unfused 机制建模错")
print("=" * 108)
print("%-6s %14s %14s %14s   %s" % ("stage", "resid_b", "resid_a(基线)", "Δ=机制引入", "校验 resid_b−resid_a == −err"))
for st in range(4):
    if ("b", st) not in resid or ("a", st) not in resid:
        continue
    rb, hb = resid[("b", st)]
    ra, ha = resid[("a", st)]
    eb, eh = contrast_err[("unfused 物化 @L8 (b−a)", st)]
    print("%-6d %14.1f %14.1f %14.1f   bucket: %.1f vs %.1f  |  hand: %.1f vs %.1f"
          % (st, rb, ra, rb - ra, rb - ra, -eb, hb - ha, -eh))

# ── D. 峰值桶截面 ───────────────────────────────────────────────────────────
print()
print("=" * 108)
print("D. bucket 路径峰值桶截面（残差得住在这些桶的某一个里）")
print("=" * 108)
for letter in ("a", "b", "c", "g", "h"):
    bd = bucket_run(letter)
    print("\n--- run %s : %s ---" % (letter, LET[letter].tag))
    for st in range(4):
        e = bd[st]
        rv = series(letter, "real_MiB").get(st)
        tail = ("   real=%.1f  resid=%+.1f" % (rv, rv - e["peak"])) if rv is not None else "   (real OOM)"
        print("  stage%d peak=%9.1f  ev=%-28s%s" % (st, e["peak"], e["event"], tail))
        print("         %s" % json.dumps(e["bd"], ensure_ascii=False))
