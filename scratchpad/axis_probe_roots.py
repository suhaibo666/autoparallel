"""PROBE (read-only): dump the node record at whatever the current cascade roots are.

Committed per the process note in the task brief (every probe whose numbers are cited must
be re-runnable).  Pass source substrings on the command line to inspect specific sites:

    PYTHONIOENCODING=utf-8 python scratchpad/axis_probe_roots.py csa.py:439 dsv4:291

With no arguments it inspects whatever `Coverage.blockers()` currently reports.
NOTHING in cost_eval/ is modified.
"""
import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
warnings.simplefilter("ignore")
os.environ["COST_EVAL_EXTRACTED_ALLOW_PARTIAL"] = "1"

import liveness_ab_validate as V  # noqa: E402
from cost_eval.parallel_model import ParallelModel  # noqa: E402
from cost_eval.opdag import to_resolved as TR  # noqa: E402
import cost_eval.opdag.shape_infer as SI  # noqa: E402

_ATTR_KEYS = ("view", "prim", "permute_dims", "perm", "swap_axes", "squeeze_axis",
              "reshape_dims", "split_dim", "split_sizes", "split_targets", "index",
              "advanced_index", "reduce_dim", "keepdim", "reduce", "concat_axis",
              "const_shape", "const_shape_src", "const_argc", "param_operands",
              "ins_slots", "expand_axis", "tile_mult", "out_dim")


def build(tag):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    return spec, ParallelModel(p, spec.dims.n_layers, world)


def run(tag, label, wanted):
    spec, pm = build(tag)
    seen = []
    orig = SI._note

    def spy(ctx, n, reason, detail=""):
        if any(s in (n.src or "") for s in wanted):
            seen.append((n.src, n.op, reason, detail, list(n.ins), n.out,
                         {k: v for k, v in n.attrs.items() if k in _ATTR_KEYS}))
        return orig(ctx, n, reason, detail)

    SI._note = spy
    try:
        g = TR.resolve_graph(spec, pm, allow_partial=True)
    finally:
        SI._note = orig

    print("\n=== %s (%s) ===" % (label, tag))
    print("  cascade roots:")
    for (src, op, reason), k in g.coverage.blockers(20):
        print("    x%-3d %-14s %-42s %s" % (k, op, src, reason))
    dedup = {}
    for rec in seen:
        dedup.setdefault((rec[0], rec[2]), rec)
    for (src, reason), rec in sorted(dedup.items()):
        _, op, _, detail, ins, out, attrs = rec
        print("\n  %s  op=%s  reason=%s" % (src, op, reason))
        print("    detail : %s" % detail)
        for r in ins:
            print("    in     : %s" % r)
        print("    out    : %s" % out)
        print("    attrs  : %s" % attrs)
    return g


if __name__ == "__main__":
    args = sys.argv[1:]
    for tag, label in (("b unfused ON  L8 m4", "UNFUSED (run b)"),
                       ("a fused   ON  L8 m4", "FUSED (run a)")):
        g = None
        wanted = args
        if not wanted:
            spec, pm = build(tag)
            g = TR.resolve_graph(spec, pm, allow_partial=True)
            wanted = [src for (src, _o, _r), _k in g.coverage.blockers(20)]
        run(tag, label, wanted)
