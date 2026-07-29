"""PROBE (read-only): dump pp4/pp8/MTP theory anchors + ratios after a census change."""
import os, sys, warnings
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "tests"))
warnings.simplefilter("ignore")
import tests.test_pp4_recompute_anchor as T

def run(**kw):
    p = dict(T._BASE); p.update(kw)
    r = T.S.eval_config(p)
    assert r.get("ok"), r.get("errors")
    return {st["stage"]: st["peak"] for st in r["stages"]}

for name, kw, real in (
    ("ON  pp4", dict(recompute="full"), T.REAL_ON),
    ("OFF pp4", dict(recompute="None"), T.REAL_OFF),
    ("MTP pp4", dict(recompute="full", mtp="1"), getattr(T, "REAL_MTP", {})),
    ("PP8",     dict(dp="1", ep="1", pp="8", mbs="8", recompute="full"), T.REAL_PP8),
):
    ps = run(**kw)
    print(f"--- {name} ---")
    for i in sorted(ps):
        rr = real.get(i)
        line = f"  s{i}: sim={ps[i]:.1f}"
        if rr:
            line += f"  real={rr:.1f}  ratio={ps[i]/rr:.4f}"
        print(line)
