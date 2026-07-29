"""BASELINE PROBE (read-only) for the symbolic-axis-structure work (2026-07-29).

Committed per the process note in the task brief: every probe whose numbers are cited
must be in the repo so the numbers can be re-run.

What it prints:
  PART 1  per-layer coverage + cascade roots, both fused (run a) and unfused (run b)
  PART 2  View sub-type census: which axis attrs the walker actually recorded
          (permute_dims / perm / swap_axes / squeeze_axis / ...) and how many of each
          sub-type currently degrade to `~` (numel_only)
  PART 3  the four cascade-root sites, verbatim: node attrs + input syms + out sym
  PART 4  per-layer extracted activation_saves

Usage:  PYTHONIOENCODING=utf-8 python scratchpad/axis_probe_baseline.py
NOTHING in cost_eval/ is modified.
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

TAG_UNFUSED = "b unfused ON  L8 m4"
TAG_FUSED = "a fused   ON  L8 m4"

_ROOT_SRC = ("compressor.py:216", "compressor.py:233",
             "deepseek_v4_hybrid_attention.py:205", "csa.py:485", "csa.py:496")


def build(tag):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    return spec, ParallelModel(p, spec.dims.n_layers, world)


def coverage(tag, label):
    spec, pm = build(tag)
    g = TR.resolve_graph(spec, pm, allow_partial=True)
    cov = g.coverage
    t = cov.totals()
    print("\n=== %s  (%s) ===" % (label, tag))
    print("  GLOBAL n_nodes=%d n_ops=%d skipped=%d"
          % (t.get("n_nodes", -1), t["n_ops"], t["skipped_ops"]))
    for c in cov.children:
        ct = c.totals()
        print("    %-28s ops=%4d skipped=%4d" % (c.tag, ct["n_ops"], ct["skipped_ops"]))
    print("  cascade roots:")
    for (src, op, reason), k in cov.blockers(20):
        print("    x%-3d %-14s %-42s %s" % (k, op, src, reason))
    print("  per-layer extracted activation_saves:")
    for st, layers in sorted(g.stages.items()):
        for L in layers:
            b = sum(r.local_numel * r.dtype_bytes for op in L.ops for r in op.saves)
            print("    s%d L%-2d %-24s %10.1f MiB" % (st, L.layer_id, L.layer_type, b / MiB))
    return g


def view_census(tag, label):
    """Walk the raw DAGs (pre-`to_resolved`) counting View sub-types + recorded axis attrs."""
    from cost_eval.opdag import extractor as EX
    spec, pm = build(tag)
    dags = TR._layer_dags(spec, pm) if hasattr(TR, "_layer_dags") else None
    if dags is None:
        print("\n(view_census: no _layer_dags hook; skipped)")
        return
    print("\n=== VIEW CENSUS %s ===" % label)
    sub = Counter()
    has_axis = Counter()
    for dag in dags:
        for n in dag.nodes:
            if n.op != "View":
                continue
            v = n.attrs.get("view")
            sub[v] += 1
            for k in ("permute_dims", "perm", "swap_axes", "squeeze_axis", "expand_axis",
                      "reshape_dims", "concat_axis", "chunk_dim", "chunks", "split_dim",
                      "tile_mult", "index", "broadcast_shape", "stack_axis"):
                if n.attrs.get(k) is not None:
                    has_axis[(v, k)] += 1
    for v, k in sub.most_common():
        keys = [kk for (vv, kk), c in has_axis.items() if vv == v]
        print("  %-14s x%-4d  attrs seen: %s" % (v, k, sorted(keys)))


def root_sites(tag, label):
    """Dump the node record at each of the four cascade-root sites, verbatim."""
    import cost_eval.opdag.shape_infer as SI
    spec, pm = build(tag)
    seen = []

    orig_note = SI._note

    def spy_note(ctx, n, reason, detail=""):
        if any(s in (n.src or "") for s in _ROOT_SRC):
            seen.append((n.src, n.op, reason, detail, list(n.ins), n.out,
                         {k: v for k, v in n.attrs.items()
                          if k in ("view", "prim", "permute_dims", "perm", "swap_axes",
                                   "reshape_dims", "split_dim", "split_sizes",
                                   "split_targets", "index", "advanced_index",
                                   "reduce_dim", "keepdim", "reduce", "concat_axis")}))
        return orig_note(ctx, n, reason, detail)

    SI._note = spy_note
    try:
        TR.resolve_graph(spec, pm, allow_partial=True)
    finally:
        SI._note = orig_note

    print("\n=== ROOT SITES %s ===" % label)
    dedup = {}
    for rec in seen:
        key = (rec[0], rec[2])
        if key not in dedup:
            dedup[key] = rec
    for (src, reason), rec in sorted(dedup.items()):
        _, op, _, detail, ins, out, attrs = rec
        print("\n  %s  op=%s  reason=%s" % (src, op, reason))
        print("    detail : %s" % detail)
        for r in ins:
            print("    in     : %s" % r)
        print("    out    : %s" % out)
        print("    attrs  : %s" % attrs)


if __name__ == "__main__":
    coverage(TAG_UNFUSED, "UNFUSED (run b)")
    coverage(TAG_FUSED, "FUSED (run a)")
    view_census(TAG_UNFUSED, "UNFUSED")
    root_sites(TAG_UNFUSED, "UNFUSED")
    root_sites(TAG_FUSED, "FUSED")
