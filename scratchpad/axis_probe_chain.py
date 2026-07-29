"""PROBE (read-only): dump every node's resolved out-shape for a source-file range.

    PYTHONIOENCODING=utf-8 python scratchpad/axis_probe_chain.py csa.py 430 530 [--fused]

Prints, in walk order, `src  op  view/prim  ins -> out` for every node whose `src` basename
matches and whose line falls in [lo, hi].  Used to find WHERE a wrong axis first appears
(the discipline being: a recovered axis must be checkable against the source, not trusted).

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
from cost_eval.opdag import shape_infer as SI  # noqa: E402


def build(tag):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    return spec, ParallelModel(p, spec.dims.n_layers, world)


def _line(src):
    tail = (src or "").rsplit("/", 1)[-1]
    if ":" not in tail:
        return None, tail
    base, _, ln = tail.partition(":")
    try:
        return int(ln), base
    except ValueError:
        return None, base


def main(fname, lo, hi, tag):
    spec, pm = build(tag)
    rows = []
    # `to_resolved` 里是 `from .shape_infer import infer_shapes`（已绑定名）→ 必须改 TR 上那份。
    orig = TR.infer_shapes

    def spy(dag, *a, **kw):
        r = orig(dag, *a, **kw)
        for n in dag.nodes:
            ln, base = _line(n.src)
            if base == fname and ln is not None and lo <= ln <= hi:
                rows.append((n.id, n.src, n.op, n.attrs.get("view") or n.attrs.get("prim") or "",
                             list(n.ins), n.out))
        return r

    TR.infer_shapes = spy
    try:
        TR.resolve_graph(spec, pm, allow_partial=True)
    finally:
        TR.infer_shapes = orig

    seen = set()
    for nid, src, op, kind, ins, out in rows:
        key = (src, op, kind, tuple(ins), out)
        if key in seen:
            continue
        seen.add(key)
        print("%-34s %-13s %-16s" % (src, op, kind))
        for r in ins:
            print("      in  %s" % r)
        print("      OUT %s" % out)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    fused = "--fused" in sys.argv
    main(a[0], int(a[1]), int(a[2]),
         "a fused   ON  L8 m4" if fused else "b unfused ON  L8 m4")
