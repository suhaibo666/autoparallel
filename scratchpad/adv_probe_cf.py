"""ADVERSARIAL PROBE (read-only): COUNTERFACTUAL test of prediction P1 / criterion E1.

Monkeypatch shape_infer._advanced_index so that csa.py:485 RESOLVES with the shape the
operator definition mandates (idx.shape ++ x.shape[1:], with x.shape[1:] = [v_head_dim]
read literally from csa.py:482 `kv_flat = mint.reshape(kv_t, (b*sk, d))`).

This simulates the BEST possible outcome of the recommended fix, then re-measures:
  - L1(unfused r0) skipped_ops   (diagnosis P1: 42 -> <=5 ; E1 fires if >22)
  - global skipped_ops           (diagnosis P1: 1010 -> 968 +- 8)
  - cascade roots                (diagnosis P1: csa.py:485 disappears)
  - extracted L1 activation_saves(diagnosis P2: 1067.6 -> [3000,5000], < 8468.2)

NOTHING in cost_eval/ is modified on disk.
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
from cost_eval.opdag.sym_shape import Factors  # noqa: E402


def build(tag):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    pm = ParallelModel(p, spec.dims.n_layers, world)
    return spec, pm


def measure(tag, label):
    spec, pm = build(tag)
    g = TR.resolve_graph(spec, pm, allow_partial=True)
    cov = g.coverage
    t = cov.totals()
    out = {"label": label, "n_ops": t["n_ops"], "skipped": t["skipped_ops"]}
    per = {}
    for c in cov.children:
        ct = c.totals()
        per[c.tag] = (ct["n_ops"], ct["skipped_ops"])
    out["per_layer"] = per
    out["roots"] = cov.blockers(20)
    saves = {}
    for st, layers in sorted(g.stages.items()):
        for L in layers:
            saves[(st, L.layer_id, L.layer_type)] = sum(
                r.local_numel * r.dtype_bytes for op in L.ops for r in op.saves)
    out["saves"] = saves
    out["names"] = {}
    for st, layers in sorted(g.stages.items()):
        for L in layers:
            if L.layer_id in (1,):
                out["names"][L.layer_id] = sorted(
                    ((r.name, r.local_numel * r.dtype_bytes / MiB, r.sym_shape, r.src)
                     for op in L.ops for r in op.saves),
                    key=lambda x: -x[1])
    return out


def report(o):
    print("\n=== %s ===" % o["label"])
    print("  GLOBAL  n_ops=%d  skipped=%d" % (o["n_ops"], o["skipped"]))
    for k, (n, s) in o["per_layer"].items():
        print("    %-28s ops=%4d skipped=%4d" % (k, n, s))
    print("  cascade roots:")
    for (src, op, reason), k in o["roots"]:
        print("    x%-3d %-14s %-40s %s" % (k, op, src, reason))
    print("  per-layer extracted activation_saves:")
    for (st, lid, lt), b in sorted(o["saves"].items()):
        print("    s%d L%-2d %-22s %10.1f MiB" % (st, lid, lt, b / MiB))


_orig = SI._advanced_index
_D_AXIS = Factors(1, {"v_head_dim": 1})   # csa.py:482: kv_flat = reshape(kv_t, (b*sk, d))


def patched(n, in_axes_list, in_numel_only):
    r = _orig(n, in_axes_list, in_numel_only)
    if r is not None:
        return r
    if "csa.py:485" not in (n.src or ""):
        return None
    # replicate the operator definition with the literal trailing axis from csa.py:482
    if len(in_axes_list) != 2 or in_axes_list[1] is None or in_numel_only[1]:
        return None
    idx = in_axes_list[1]
    return [f.copy() for f in idx] + [_D_AXIS.copy()], False


if __name__ == "__main__":
    base = measure("b unfused ON  L8 m4", "BASELINE (HEAD 8849f97)")
    report(base)
    SI._advanced_index = patched
    try:
        cf = measure("b unfused ON  L8 m4", "COUNTERFACTUAL (csa.py:485 resolves)")
    finally:
        SI._advanced_index = _orig
    report(cf)

    print("\n" + "#" * 92)
    print("# VERDICT ON THE DIAGNOSIS' OWN PREDICTIONS")
    print("#" * 92)
    b1 = base["per_layer"]["L1:dsv4hyb_r0_dense"][1]
    c1 = cf["per_layer"]["L1:dsv4hyb_r0_dense"][1]
    print("  P1  L1 skipped        : %d -> %d      (predicted <=5 ; E1 fires if >22)  => %s"
          % (b1, c1, "P1 FAILS" if c1 > 5 else "P1 holds"))
    print("  P1  global skipped    : %d -> %d      (predicted 968 +- 8)              => %s"
          % (base["skipped"], cf["skipped"],
             "P1 holds" if abs(cf["skipped"] - 968) <= 8 else "P1 FAILS"))
    still = [x for x in cf["roots"] if "csa.py:485" in x[0][0]]
    print("  P1  csa.py:485 root   : %s" % ("STILL PRESENT -> not fixed" if still
                                            else "gone (as predicted)"))
    k = [x for x in cf["saves"] if x[1] == 1][0]
    print("  P2  L1 activation_saves: %.1f -> %.1f MiB   (predicted [3000,5000], "
          "hand_spec=8468.2)  => %s"
          % (base["saves"][k] / MiB, cf["saves"][k] / MiB,
             "P2 holds" if 3000 <= cf["saves"][k] / MiB <= 5000 else "P2 FAILS"))
    print("\n  --- P2 per-tensor list: extracted L1 saves after the counterfactual ---")
    for nm, mb, sym, src in cf["names"].get(1, []):
        print("    %-26s %9.1f MiB  %-34s %s" % (nm, mb, sym, src))
