"""PROBE (read-only): std-attn 116 anchors + scorecard ratios after the norm-kind change."""
import os, sys, warnings
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "tests"))
warnings.simplefilter("ignore")
import tests.test_std_attn_anchor as S
pk = {(a, pp): S._peaks(32 if a == "mha" else 8, pp) for a in ("mha", "gqa") for pp in (2, 1)}
print("--- std 116 ---")
for k, lohi in S.BAND.items():
    a, pp, st = k
    sim = pk[(a, pp)][st]; real = S.REAL_116[(a, pp)][st]
    print(f"  {a} pp{pp} s{st}: sim={sim:.1f} real={real:.1f} ratio={sim/real:.4f} band={lohi}")
from scorecard_anchors import anchors
print("--- scorecard ---")
for an in anchors():
    sim = an.sim_fn()
    if sim is None:
        print(f"  {an.label}: SKIP"); continue
    print(f"  {an.label!r}: sim={sim:.1f} real={an.real:.1f} ratio={sim/an.real:.4f} band={an.band}")
