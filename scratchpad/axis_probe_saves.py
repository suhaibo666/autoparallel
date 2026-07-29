"""PROBE (read-only): per-layer `activation_saves` broken down per tensor, largest first.

Used to CHECK the recovered axis structure against the source rather than to trust it:
every large entry must be reconcilable with a `file:line` in the mindformers snapshot.

    PYTHONIOENCODING=utf-8 python scratchpad/axis_probe_saves.py [layer_id ...]

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
    return spec, ParallelModel(p, spec.dims.n_layers, world)


def dump(tag, label, want_layers, top=25):
    spec, pm = build(tag)
    g = TR.resolve_graph(spec, pm, allow_partial=True)
    print("\n=== %s (%s) ===" % (label, tag))
    for st, layers in sorted(g.stages.items()):
        for L in layers:
            if want_layers and L.layer_id not in want_layers:
                continue
            rows, seen = [], set()
            for op in L.ops:
                for r in op.saves:
                    if r.name in seen:
                        continue
                    seen.add(r.name)
                    rows.append((r.local_numel * r.dtype_bytes, r.name, r.sym_shape,
                                 r.dtype_bytes, r.src))
            rows.sort(reverse=True)
            print("\n  L%d %s   total=%.1f MiB over %d tensors"
                  % (L.layer_id, L.layer_type, sum(x[0] for x in rows) / MiB, len(rows)))
            for b, nm, sym, db, src in rows[:top]:
                print("    %9.1f MiB  %-26s %-42s %db  %s" % (b / MiB, nm, sym, db, src))


if __name__ == "__main__":
    want = {int(x) for x in sys.argv[1:]} or {1, 2, 3}
    dump("b unfused ON  L8 m4", "UNFUSED (run b)", want)
