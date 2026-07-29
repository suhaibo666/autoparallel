"""PROBE (read-only): dump pp4/pp8/MTP theory anchors **using the test's own fixture params**."""
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
    ("THEO_ON ", dict(recompute="full"), T.REAL_ON),
    ("BAND_OFF", dict(recompute="None"), T.REAL_OFF),
    ("THEO_MTP", dict(mtp="1", recompute="full", pp_split="2,2,2,3"), T.REAL_MTP),
    ("THEO_PP8", dict(dp="1", ep="1", pp="8", mbs="8", recompute="full"), T.REAL_PP8),
):
    ps = run(**kw)
    print(f"{name} = {{" + ", ".join(f"{i}: {ps[i]:.1f}" for i in sorted(ps)) + "}")
    print("      ratios: " + "  ".join(
        f"s{i}={ps[i]/real[i]:.4f}" for i in sorted(ps) if i in real))
