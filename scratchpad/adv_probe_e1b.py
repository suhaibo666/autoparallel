"""ADVERSARIAL PROBE (read-only): enumerate every skipped op of the r0 layers, and
trace where kv_flat loses its axis structure (the real cause of the csa.py:485 gate).
"""
import os
import sys
import warnings
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
warnings.simplefilter("ignore")
os.environ["COST_EVAL_EXTRACTED_ALLOW_PARTIAL"] = "1"

MiB = 2 ** 20

import liveness_ab_validate as V  # noqa: E402
from cost_eval.parallel_model import ParallelModel  # noqa: E402
from cost_eval.opdag import to_resolved as TR  # noqa: E402


def build(tag):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    pm = ParallelModel(p, spec.dims.n_layers, world)
    return spec, pm


def dump(tag, want):
    spec, pm = build(tag)
    g = TR.resolve_graph(spec, pm, allow_partial=True)
    top = next(c for c in g.coverage.children if c.tag == want)
    fam = [top] + top._all_children()
    print("\n" + "=" * 96)
    print("%s  /  %s     (%d coverage nodes in family)" % (tag, want, len(fam)))
    print("=" * 96)
    tot = top.totals()
    print("  ops=%d skipped=%d" % (tot["n_ops"], tot["skipped_ops"]))
    grp = Counter()
    allsk = []
    for c in fam:
        for nid, op, src, reason in c.skipped_ops:
            grp[(src, op, reason.split(" @")[0])] += 1
            allsk.append((c.tag, nid, op, src, reason.split(" @")[0]))
        if c.first_skipped:
            print("  segment %-24s first_skipped = %-12s %-38s %s"
                  % (c.tag or "<top>", c.first_skipped[1], c.first_skipped[2],
                     c.first_skipped[3].split(" @")[0]))
    print("  --- skipped ops grouped by (src, op, reason) ---")
    for (src, op, reason), k in sorted(grp.items(), key=lambda kv: (-kv[1], kv[0])):
        print("    x%-3d %-14s %-40s %s" % (k, op, src, reason))
    print("  total grouped = %d" % sum(grp.values()))
    print("  --- in execution order ---")
    for i, (ctag, nid, op, src, reason) in enumerate(allsk):
        print("    %3d  %-10s %-14s %-40s %s" % (i, nid, op, src, reason))
    return grp


if __name__ == "__main__":
    g1 = dump("b unfused ON  L8 m4", "L1:dsv4hyb_r0_dense")
    g2 = dump("a fused   ON  L8 m4", "L1:dsv4hyb_r0_dense")
    print("\n" + "#" * 96)
    print("# E1 ANALYSIS: which of the unfused r0 skipped ops are NOT in csa.py")
    print("#" * 96)
    non_csa = {k: v for k, v in g1.items() if "csa.py" not in k[0]}
    csa = {k: v for k, v in g1.items() if "csa.py" in k[0]}
    print("  in csa.py      : %d ops  %s" % (sum(csa.values()), dict(csa)))
    print("  NOT in csa.py  : %d ops" % sum(non_csa.values()))
    for k, v in sorted(non_csa.items(), key=lambda kv: -kv[1]):
        print("     x%-3d %-14s %-40s %s" % (v, k[1], k[0], k[2]))
    print("\n  fused r0 skipped (independent of the unfused chain):")
    for k, v in sorted(g2.items(), key=lambda kv: -kv[1]):
        print("     x%-3d %-14s %-40s %s" % (v, k[1], k[0], k[2]))
