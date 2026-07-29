"""PROBE (read-only): regenerate the acceptance-gate golden tables after a census change."""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")
from tools.liveness_ab_validate import (DEFAULT_BASE_DIR, VARIANTS, aggregate,
                                        run_matrix, magnitude_report, PAIRS)

M = {gm: run_matrix(DEFAULT_BASE_DIR, ["hand_spec"], gm) for gm in ("dataflow", "chain2")}

def tup(xs):
    return "(" + ", ".join(f"{x:.1f}" for x in xs) + ")"

print("GOLDEN_BUCKET = {")
for v in VARIANTS:
    print(f'    "{v.tag}": {tup(M["chain2"][v.tag].bucket_mib)},')
print("}")
for gm, name in (("dataflow", "GOLDEN_LIVENESS_DATAFLOW"), ("chain2", "GOLDEN_LIVENESS_CHAIN2")):
    print(f"{name} = {{")
    for v in VARIANTS:
        print(f'    "{v.tag}": {tup(M[gm][v.tag].liveness_mib["hand_spec"])},')
    print("}")
print("GOLDEN_AGG = {")
for gm in ("dataflow", "chain2"):
    for key in ("bucket", "hand_spec"):
        a = aggregate(M[gm], key)
        print(f'    ("{gm}", "{key}"): ({a["n"]}, {round(a["mean"],3)}, '
              f'{round(a["min"],3)}, {round(a["max"],3)}),')
print("}")

# delta magnitude @ stage3 (L8 m4)
rep = magnitude_report(M["chain2"], PAIRS)
print("\n-- delta magnitude (stage3) --")
for row in rep:
    print(row)

r = M["chain2"]["b unfused ON  L8 m4"]
lv = [round(r.liveness_mib["hand_spec"][i] / r.real(i), 3) for i in range(4)]
bk = [r.bucket_mib[i] / r.real(i) for i in range(4)]
print("\nunfused-ON chain2 liveness ratios:", lv)
print("unfused-ON bucket ratios: min=%.4f max=%.4f  all=%s"
      % (min(bk), max(bk), [round(x, 4) for x in bk]))

# run c per-stage bucket ratio
c = M["chain2"]["c fused   OFF L8 m4"]
print("run c bucket sim/real:", [round(c.bucket_mib[i] / c.real(i), 3) for i in range(4)])
