"""PROBE (read-only): dump the **extracted** op list of one layer (fused run c), with
每个 op 的 src / inputs / output / saves —— 用来判断「抽取侧某张量不在场」到底是
「VJP 说不保留」还是「这个节点根本没被走到」（后者是跳过，不能当证据）。

    PYTHONIOENCODING=utf-8 python scratchpad/census_ops_dump.py [layer_id]

NOTHING in cost_eval/ is modified.
"""
import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
warnings.simplefilter("ignore")
os.environ.setdefault("COST_EVAL_EXTRACTED_ALLOW_PARTIAL", "1")

MiB = 2 ** 20
TAG = "c fused   OFF L8 m4"

import liveness_ab_validate as V  # noqa: E402
from cost_eval.parallel_model import ParallelModel  # noqa: E402
from cost_eval.opdag import to_resolved as TR  # noqa: E402


def main():
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    v = V.BY_TAG[TAG]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    pm = ParallelModel(p, spec.dims.n_layers, world)
    g = TR.resolve_graph(spec, pm, allow_partial=True)
    L = next(x for layers in g.stages.values() for x in layers if x.layer_id == want)
    print("layer_id=%d type=%s  ops=%d" % (L.layer_id, L.layer_type, len(L.ops)))
    for i, op in enumerate(L.ops):
        sv = ", ".join("%s(%.1fMiB,%db)" % (r.name, r.local_numel * r.dtype_bytes / MiB, r.dtype_bytes)
                       for r in op.saves) or "-"
        outn = getattr(op.output, "name", None)
        print("  %3d %-16s %-46s out=%-24s saves=[%s]"
              % (i, getattr(op.type, "value", op.type), getattr(op, "src", ""), outn, sv))
    cov = getattr(g, "coverage", None)
    if cov is not None:
        print("\ncoverage.is_partial=%s" % getattr(cov, "is_partial", "?"))
        sk = getattr(cov, "skipped", None)
        if sk:
            print("skipped: %d" % len(sk))
            for s in list(sk)[:40]:
                print("   ", s)


if __name__ == "__main__":
    main()
