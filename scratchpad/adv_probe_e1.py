"""ADVERSARIAL PROBE (read-only): direct test of refutation criterion E1.

The diagnosis predicts: fixing csa.py:485 drops L1(unfused r0) skipped_ops 42 -> <=5.
E1 (its own refutation criterion): if still > 22, "one node gates the whole function body"
is FALSE.

This probe does NOT need to implement the fix. It enumerates the 42 skipped ops of the
unfused r0 layer with (src, reason) and asks which of them are (a) the csa.py:485 cascade
and (b) independent gates that would survive the fix.

It also (1) settles which of the 4 guards in shape_infer._advanced_index actually fails
(the diagnosis §9-1 explicitly declined to determine this), and (2) runs a counterfactual:
monkeypatch _advanced_index to succeed at csa.py:485 and re-measure skipped_ops.
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
from cost_eval.opdag import shape_infer as SI  # noqa: E402


def build(tag):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    pm = ParallelModel(p, spec.dims.n_layers, world)
    return spec, pm


def layer_cov(tag, want):
    spec, pm = build(tag)
    TR.resolve_graph.cache_clear() if hasattr(TR.resolve_graph, "cache_clear") else None
    g = TR.resolve_graph(spec, pm, allow_partial=True)
    for c in g.coverage.children:
        if c.tag == want:
            return c, g
    raise KeyError(want)


def show(c, title):
    print("\n=== %s ===" % title)
    t = c.totals()
    print("  ops=%d skipped=%d  reasons=%s"
          % (t["n_ops"], t["skipped_ops"],
             dict(sorted(c.node_gap_reasons.items(), key=lambda kv: -kv[1]))))
    print("  first_skipped = %s" % (c.first_skipped,))
    print("  --- all skipped ops, grouped by (src, reason) ---")
    grp = Counter()
    for item in c.skipped_ops:
        nid, op, src, reason = item
        grp[(src, op, reason.split(" @")[0])] += 1
    for (src, op, reason), k in sorted(grp.items(), key=lambda kv: (-kv[1], kv[0])):
        print("    x%-3d %-14s %-42s %s" % (k, op, src, reason))
    print("  --- execution order of skipped ops (first 50) ---")
    for item in c.skipped_ops[:50]:
        nid, op, src, reason = item
        print("    %-6s %-14s %-42s %s" % (nid, op, src, reason.split(" @")[0]))
    return grp


def main():
    print("#" * 100)
    print("# PART A -- baseline: the 42 skipped ops of L1 (unfused r0)")
    print("#" * 100)
    c, _ = layer_cov("b unfused ON  L8 m4", "L1:dsv4hyb_r0_dense")
    show(c, "BASELINE L1 unfused r0")

    print()
    print("#" * 100)
    print("# PART B -- the FUSED r0 layer's skipped ops (13). Does it share a gate?")
    print("#" * 100)
    cf, _ = layer_cov("a fused   ON  L8 m4", "L1:dsv4hyb_r0_dense")
    show(cf, "BASELINE L1 fused r0")

    print()
    print("#" * 100)
    print("# PART C -- GUARD ANALYSIS at csa.py:485 (diagnosis sec.9-1: 'I did not determine')")
    print("#" * 100)
    orig = SI._advanced_index
    seen = {}

    def spy(n, in_axes_list, in_numel_only):
        if "csa.py:485" in (n.src or ""):
            why = []
            if not n.attrs.get("advanced_index"):
                why.append("GUARD1 FAIL: attrs['advanced_index'] missing/false")
            if len(n.ins) != 2 or len(in_axes_list) != 2:
                why.append("GUARD2 FAIL: not exactly 2 tensor operands "
                           "(len(ins)=%d, len(in_axes)=%d)" % (len(n.ins), len(in_axes_list)))
            else:
                if in_axes_list[0] is None:
                    why.append("GUARD3a FAIL: x axes = None")
                if in_axes_list[1] is None:
                    why.append("GUARD3b FAIL: idx axes = None")
                if any(in_numel_only[:2]):
                    why.append("GUARD3c FAIL: numel_only=%s ('~' axis structure)"
                               % (list(in_numel_only[:2]),))
                dt = n.ins[1].split(":")[2] if n.ins[1].count(":") == 2 else ""
                if dt not in SI._INT_DTYPES:
                    why.append("GUARD4 FAIL: index dtype=%r not in %s" % (dt, SI._INT_DTYPES))
            k = (n.id, n.src)
            seen.setdefault(k, (n.ins, list(in_axes_list), list(in_numel_only),
                                dict(n.attrs), why or ["ALL GUARDS PASS"]))
        return orig(n, in_axes_list, in_numel_only)

    SI._advanced_index = spy
    try:
        layer_cov("b unfused ON  L8 m4", "L1:dsv4hyb_r0_dense")
    finally:
        SI._advanced_index = orig
    if not seen:
        print("  !! _advanced_index was NEVER CALLED for a csa.py:485 node.")
        print("     => the node never reaches the IndexSelect dispatch arm, or _gather "
              "already returned non-None, or the node id/src differs.")
    for (nid, src), (ins, axes, numel, attrs, why) in seen.items():
        print("  node=%s src=%s" % (nid, src))
        print("     ins        = %s" % (ins,))
        print("     in_axes    = %s" % (axes,))
        print("     numel_only = %s" % (numel,))
        print("     attrs      = %s" % (attrs,))
        for w in why:
            print("     -> %s" % w)


if __name__ == "__main__":
    main()
