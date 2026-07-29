"""ADVERSARIAL PROBE (read-only): is h(e) >= g(e) pointwise?

If unfused dominates fused at EVERY event, then
    peak_h - peak_g  <=  max_e (h(e) - g(e))
i.e. the observed REAL delta is a LOWER BOUND on the true max single-layer unfused
increment. That makes "model over-reads" unprovable by differencing and "model
under-reads" robust. This probe checks the premise in both model sources.
"""
import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
warnings.simplefilter("ignore")
MiB = 2 ** 20

import liveness_ab_validate as V  # noqa: E402
from cost_eval.report import Evaluator  # noqa: E402
from cost_eval.liveness import simulate_liveness  # noqa: E402


def both(tag):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    rep = Evaluator(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                    check_feasibility=False).evaluate(record_timeline=True)
    lv = simulate_liveness(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                           record_timeline=True, grad_mode="chain2", graph_source="hand_spec")
    return rep, lv


def compare(gt, ht, name, label):
    tg = {(s.event, s.mb, s.chunk): s for s in gt}
    th = {(s.event, s.mb, s.chunk): s for s in ht}
    common = sorted(set(tg) & set(th), key=lambda k: tg[k].idx)
    only_g, only_h = set(tg) - set(th), set(th) - set(tg)
    ds = [(th[k].total_bytes - tg[k].total_bytes, k) for k in common]
    neg = [(d, k) for d, k in ds if d < 0]
    mx = max(ds)
    pk_g = max(gt, key=lambda s: s.total_bytes)
    pk_h = max(ht, key=lambda s: s.total_bytes)
    print("  %-10s %-4s events g=%d h=%d common=%d only_g=%d only_h=%d"
          % (label, name, len(gt), len(ht), len(common), len(only_g), len(only_h)))
    print("        peak_h-peak_g = %9.1f | max_e(h-g) = %9.1f @%s | #events where h<g = %d"
          % ((pk_h.total_bytes - pk_g.total_bytes) / MiB, mx[0] / MiB, mx[1], len(neg)))
    if neg:
        worst = min(neg)
        print("        WORST h<g: %.1f MiB @%s  -> pointwise dominance VIOLATED"
              % (worst[0] / MiB, worst[1]))
    att = mx[0] - (pk_h.total_bytes - pk_g.total_bytes)
    print("        ATTENUATION (max_e delta - peak delta) = %9.1f MiB  (%.1f%%)"
          % (att / MiB, 100.0 * att / mx[0] if mx[0] else 0.0))
    print("        g peak @%s  h peak @%s   same=%s"
          % (pk_g.event, pk_h.event, pk_g.event == pk_h.event))


if __name__ == "__main__":
    for lbl, ftag, utag in (("L4 m4", "g fused   ON  L4 m4", "h unfused ON  L4 m4"),
                            ("L8 m4", "a fused   ON  L8 m4", "b unfused ON  L8 m4")):
        rg, lg = both(ftag)
        rh, lh = both(utag)
        print("\n### %s  (%s  vs  %s)" % (lbl, ftag.strip(), utag.strip()))
        for st in range(4):
            print("  -- stage %d --" % st)
            compare(rg.per_stage[st].timeline, rh.per_stage[st].timeline, "s%d" % st, "bucket")
            compare(lg.per_stage[st].timeline, lh.per_stage[st].timeline, "s%d" % st, "hand_spec")
            print("        hand_spec peak substeps: g=%s/%s   h=%s/%s"
                  % (lg.per_stage[st].peak_event, lg.per_stage[st].peak_substep,
                     lh.per_stage[st].peak_event, lh.per_stage[st].peak_substep))
