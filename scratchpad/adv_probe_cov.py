"""ADVERSARIAL PROBE (read-only): per-layer coverage + the csa.py:485 node's actual attrs.

Answers:
  A. reproduce the diagnosis' per-layer coverage table (run b, unfused L8)
  B. list ALL cascade roots for BOTH branches (diagnosis says 3, coverage doc says 4)
  C. dump the csa.py:485 node: attrs, ins, and which of the 4 guards in
     shape_infer._advanced_index actually fails (diagnosis §9-1 says "I did not determine this")
"""
import os
import sys
import warnings

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


def per_layer(tag):
    spec, pm = build(tag)
    g = TR.resolve_graph(spec, pm, allow_partial=True)
    cov = g.coverage
    print("\n### %s" % tag)
    t = cov.totals()
    print("  TOTALS n_nodes=%d n_ops=%d skipped=%d unresolved_saves=%d "
          "unresolved_params=%d" % (t["n_nodes"], t["n_ops"], t["skipped_ops"],
                                    t["unresolved_saves"], t["unresolved_params"]))
    print("  %-28s %6s %6s %8s   %s" % ("child tag", "ops", "skip", "saves_MiB", "first_skipped"))
    layer_saves = {}
    for st, layers in sorted(g.stages.items()):
        for L in layers:
            layer_saves[L.layer_id] = (L.layer_type, sum(
                r.local_numel*r.dtype_bytes for op in L.ops for r in op.saves))
    for c in cov.children:
        fs = c.first_skipped
        fstr = ("%-12s %-34s %s" % (fs[1], fs[2], fs[3].split(" @")[0])) if fs else "-"
        ct = c.totals()
        print("  %-28s %6d %6d %8s   %s"
              % (c.tag, ct["n_ops"], ct["skipped_ops"], "", fstr))
        if c.node_gap_reasons:
            print("      reasons: %s" % dict(sorted(c.node_gap_reasons.items(),
                                                    key=lambda kv: -kv[1])))
    print("  --- ALL cascade roots (blockers, n=20) ---")
    for (src, op, reason), k in cov.blockers(20):
        print("    x%-3d %-14s %-42s %s" % (k, op, src, reason))
    print("  --- per-layer activation_saves (extracted) ---")
    for st, layers in sorted(g.stages.items()):
        for L in layers:
            tot = sum(r.local_numel*r.dtype_bytes for op in L.ops for r in op.saves)
            print("    s%d L%-2d %-22s saves=%10.1f MiB  (%d ops)"
                  % (st, L.layer_id, L.layer_type, tot / MiB, len(L.ops)))
    return g


def dump_csa485():
    """Extract the r0-unfused decoder DAG and dump the csa.py:485 node + guard analysis."""
    from cost_eval.opdag import shape_infer as SI
    spec, pm = build("b unfused ON  L8 m4")
    dims = spec.dims
    site = TR._Site(fused=False, ratio=0, moe=False)
    root = TR.mf_root()
    dag = TR._decoder_dag(root, site, dims)
    hits = [n for n in dag.nodes if "csa.py:485" in (n.src or "")]
    print("\n### csa.py:485 nodes in the r0-unfused decoder DAG: %d" % len(hits))
    for n in hits:
        print("  id=%s op=%s src=%s" % (n.id, n.op, n.src))
        print("     ins   = %s" % (n.ins,))
        print("     outs  = %s" % (n.outs,))
        print("     attrs = %s" % (n.attrs,))
    # which guard fails: replicate _advanced_index guards with the real inferred axes
    orig = SI._advanced_index
    verdicts = []

    def spy(n, in_axes_list, in_numel_only):
        if "csa.py:485" in (n.src or ""):
            why = []
            if not n.attrs.get("advanced_index"):
                why.append("GUARD1 attrs['advanced_index'] missing/false")
            if len(n.ins) != 2 or len(in_axes_list) != 2:
                why.append("GUARD2 not exactly 2 tensor operands (len(ins)=%d, len(axes)=%d)"
                           % (len(n.ins), len(in_axes_list)))
            else:
                if in_axes_list[0] is None:
                    why.append("GUARD3a in_axes_list[0] is None (x has no axes)")
                if in_axes_list[1] is None:
                    why.append("GUARD3b in_axes_list[1] is None (idx has no axes)")
                if any(in_numel_only[:2]):
                    why.append("GUARD3c numel_only=%s (axis structure is '~')"
                               % (list(in_numel_only[:2]),))
                dt = n.ins[1].split(":")[2] if n.ins[1].count(":") == 2 else ""
                if dt not in SI._INT_DTYPES:
                    why.append("GUARD4 index dtype=%r not integer" % dt)
                if in_axes_list[0] is not None and len(in_axes_list[0]) < 1:
                    why.append("GUARD5 x has 0 axes")
            verdicts.append((n.id, n.src, n.ins, list(in_axes_list), list(in_numel_only),
                             why or ["ALL GUARDS PASS"]))
        return orig(n, in_axes_list, in_numel_only)

    SI._advanced_index = spy
    try:
        # go through the normal folder path so seeds/dims_ctx are identical to production
        TR.resolve_graph(spec, pm, allow_partial=True)
    finally:
        SI._advanced_index = orig
    print("\n### GUARD ANALYSIS at csa.py:485 (%d invocations)" % len(verdicts))
    seen = set()
    for nid, src, ins, axes, numel, why in verdicts:
        key = (src, tuple(ins), str(axes), str(numel), tuple(why))
        if key in seen:
            continue
        seen.add(key)
        print("  node=%s src=%s" % (nid, src))
        print("     ins        = %s" % (ins,))
        print("     in_axes    = %s" % (axes,))
        print("     numel_only = %s" % (numel,))
        for w in why:
            print("     -> %s" % w)


if __name__ == "__main__":
    per_layer("b unfused ON  L8 m4")
    per_layer("a fused   ON  L8 m4")
    dump_csa485()
