"""ADVERSARIAL PROBE (read-only): verify the hand-written r0 census numbers the
diagnosis quotes (8468.2 MiB, W=128, 5248 MiB of _ucopies) straight from ModelSpec.
"""
import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
warnings.simplefilter("ignore")
MiB = 2 ** 20

import liveness_ab_validate as V  # noqa: E402
from cost_eval.parallel_model import ParallelModel  # noqa: E402
from cost_eval.liveness import resolve_graph  # noqa: E402


def dump(tag, want_types=("dsv4hyb_r0_dense", "dsv4hyb_r4_moe", "dsv4hyb_r128_moe")):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    pm = ParallelModel(p, spec.dims.n_layers, world)
    g = resolve_graph(spec, pm, "hand_spec")
    d = spec.dims
    got = {k: getattr(d, k, None) for k in
           ("S", "B", "H", "n_heads", "v_head_dim", "csa_window_size",
            "dsa_indexer_topk", "seq", "micro_batch")}
    print("\n### %s   dims: %s" % (tag, {k: v for k, v in got.items() if v is not None}))
    seen = set()
    for st, layers in sorted(g.stages.items()):
        for L in layers:
            if L.layer_type not in want_types or L.layer_type in seen:
                continue
            seen.add(L.layer_type)
            saves = [(r.name, r.local_numel * r.dtype_bytes / MiB)
                     for op in L.ops for r in op.saves]
            tot = sum(x[1] for x in saves)
            print("  %-22s activation_saves = %10.1f MiB  (%d entries)"
                  % (L.layer_type, tot, len(saves)))
            for nm, mb in sorted(saves, key=lambda x: -x[1]):
                print("      %-26s %9.2f MiB" % (nm, mb))


if __name__ == "__main__":
    dump("h unfused ON  L4 m4")
    dump("g fused   ON  L4 m4")
