"""PROBE (read-only): dump the mHC-wrapped decoder body op graph — which tensors got ×n renamed.

    PYTHONIOENCODING=utf-8 python scratchpad/probe_mhc_carrier.py
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
from cost_eval.shape_eval import resolve_tensor  # noqa: E402


def build(tag=TAG):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    return spec, ParallelModel(p, spec.dims.n_layers, world)


spec, pm = build()
want = sys.argv[1:] or ["dsv4hyb_r0_dense", "dsv4hyb_r4_moe"]
for key, layer in spec.layer_specs.items():
    if key not in want:
        continue
    print("=" * 100)
    print(key)
    print("=" * 100)
    for i, op in enumerate(layer.ops):
        ins = ",".join(t.name for t in op.inputs)
        sv = ",".join(t.name for t in op.saves)
        print(f"  {i:3d} {op.name:22s} {op.type.value:12s} in=[{ins}] out={op.output.name} "
              f"saves=[{sv}] nk={op.norm_kind}")
    print("  --- tensors with _xn suffix (bytes) ---")
    seen = {}
    for op in layer.ops:
        for t in list(op.inputs) + [op.output] + list(op.saves):
            if t.name.endswith("_xn") and t.name not in seen:
                r = resolve_tensor(t, spec.dims, pm)
                seen[t.name] = r.local_numel * r.dtype_bytes / MiB
    for n, mb in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"      {n:24s} {mb:9.3f} MiB")
