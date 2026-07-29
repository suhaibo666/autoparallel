"""PROBE (read-only): pp4/pp8 anchors under the **fused** mHC branch (what-if).

The pp4/pp8/scorecard anchors go through `serve_explorer.parse_and_validate`, which has no
`use_fused_mhc` knob -> always the LLMConfig default False (= UNFUSED mHC), while the real
185/167 runs those anchors are compared against ran `use_fused_mhc: true`
(`analysis/realmachine/ab_fusion_2026-07-25/dsv4h_*_pp4_recomp.yaml:109`, and this test's own
docstring says "FUSED (DSA kernel + fused mHC + fused CE)").

This probe monkeypatches the flag ON (no source change) and prints both.

    PYTHONIOENCODING=utf-8 python scratchpad/probe_pp4_fused_mhc_whatif.py
"""
import os
import sys
import warnings

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, "tests"))
warnings.simplefilter("ignore")

import serve_explorer as S  # noqa: E402
import tests.test_pp4_recompute_anchor as T  # noqa: E402

_orig = S.parse_and_validate


def patched(p):
    errs, cfg, pa = _orig(p)
    if not errs and cfg.residual_variant == "mhc":
        import dataclasses
        cfg = dataclasses.replace(cfg, use_fused_mhc=True)
    return errs, cfg, pa


def run(**kw):
    q = dict(T._BASE)
    q.update(kw)
    r = S.eval_config(q)
    assert r.get("ok"), r.get("errors")
    return {st["stage"]: st["peak"] for st in r["stages"]}


CASES = (
    ("ON  pp4", dict(recompute="full"), T.REAL_ON),
    ("OFF pp4", dict(recompute="None"), T.REAL_OFF),
    ("MTP pp4", dict(recompute="full", mtp="1"), getattr(T, "REAL_MTP", {})),
    ("PP8", dict(dp="1", ep="1", pp="8", mbs="8", recompute="full"), T.REAL_PP8),
)

for label, patch in (("UNFUSED mHC (today)", None), ("FUSED mHC (what-if)", patched)):
    if patch:
        S.parse_and_validate = patch
    print("=" * 70)
    print(label)
    print("=" * 70)
    for name, kw, real in CASES:
        ps = run(**kw)
        row = "  %-8s " % name
        for i in sorted(ps):
            rr = real.get(i)
            row += " s%d=%.1f%s" % (i, ps[i], ("(%.3f)" % (ps[i] / rr)) if rr else "")
        print(row)
S.parse_and_validate = _orig
