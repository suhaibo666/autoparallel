"""ADVERSARIAL PROBE (read-only): (a) how sharp is the model peak (headroom to runner-up),
(b) the residual noise scale, and whether the r0 delta error is inside it.

The diagnosis (sec 2.3) argues: "g's per-stage baseline residuals are ~+-2300, so r128's
-134.1 is inside noise and r0's -3477.2 / r4's +8849.2 are outside".  A DIFFERENCE of two
cells carries sqrt(2) x the single-cell noise.  This probe does that arithmetic.
"""
import os
import sys
import math
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
warnings.simplefilter("ignore")
MiB = 2 ** 20

import liveness_ab_validate as V  # noqa: E402
from cost_eval.report import Evaluator  # noqa: E402


def rep(tag):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    return Evaluator(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                     check_feasibility=False).evaluate(record_timeline=True)


print("=" * 96)
print("PART 1 -- peak sharpness (bucket model, L4): top-5 events per stage")
print("=" * 96)
rg, rh = rep("g fused   ON  L4 m4"), rep("h unfused ON  L4 m4")
for name, r in (("g fused L4", rg), ("h unfused L4", rh)):
    for st in range(4):
        ts = sorted(r.per_stage[st].timeline, key=lambda s: -s.total_bytes)[:5]
        print("  %-14s s%d  peak=%9.1f  runners-up: %s"
              % (name, st, ts[0].total_bytes / MiB,
                 "  ".join("%s/mb%d=%.1f" % (s.event, s.mb, s.total_bytes / MiB)
                           for s in ts[1:])))
        print("       headroom to #2 = %.1f MiB"
              % ((ts[0].total_bytes - ts[1].total_bytes) / MiB))

print()
print("=" * 96)
print("PART 2 -- residual noise scale and significance of the three L4 delta errors")
print("=" * 96)
# per-cell residuals real - bucket, all 24 scorable cells (run d excluded)
res = []
for v in V.VARIANTS:
    r = rep(v.tag)
    for st in range(4):
        real = V.REAL[v.tag][st]
        if real is None:
            continue
        res.append((v.letter, st, real - r.per_stage[st].peak_bytes / MiB))
print("  per-cell residual (real - bucket):")
for lt, st, d in res:
    print("     %s s%d  %+9.1f" % (lt, st, d))
vals = [d for _, _, d in res]
n = len(vals)
mean = sum(vals) / n
sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1))
rms = math.sqrt(sum(x * x for x in vals) / n)
print("  n=%d  mean=%+.1f  sd=%.1f  rms=%.1f" % (n, mean, sd, rms))

gvals = [d for lt, _, d in res if lt == "g"]
grms = math.sqrt(sum(x * x for x in gvals) / len(gvals))
print("  run g alone: %s  rms=%.1f" % (["%+.1f" % x for x in gvals], grms))

print("\n  If a single cell has residual sigma ~= S, then a DIFFERENCE of two cells")
print("  (unfused_cell - fused_cell) has sigma ~= S*sqrt(2) (independent errors).")
for S, lbl in ((grms, "sigma from run g only"), (rms, "sigma from all 28 cells")):
    Sd = S * math.sqrt(2)
    print("\n  --- %s: S=%.1f -> S_diff=%.1f ---" % (lbl, S, Sd))
    for layer, err in (("r0  (s0)", -3477.2), ("r4  (s1)", +8849.2),
                       ("r128(s2)", -134.1), ("r4  (s3)", +9690.3)):
        print("     %-9s model-real delta err = %+9.1f  -> %.2f sigma_diff  %s"
              % (layer, err, abs(err) / Sd,
                 "SIGNIFICANT" if abs(err) / Sd >= 2 else "NOT significant"))
