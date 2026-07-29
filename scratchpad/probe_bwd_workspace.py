"""PROBE (read-only): the measurement-derived bwd kernel workspace, per layer type.

Reproduces the 167 memory-tracker law
    bwd_ws(fused sparse flash-MLA + indexer) = 209715200 B + B*S*135680 B
through the full spec -> ShapeEval -> StructureMemory path, at the three sequence
lengths that were actually measured.

    PYTHONIOENCODING=utf-8 python scratchpad/probe_bwd_workspace.py

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

import liveness_ab_validate as V  # noqa: E402
from cost_eval.parallel_model import ParallelModel  # noqa: E402
from cost_eval.shape_eval import ShapeEval  # noqa: E402
from cost_eval.structure_mem import estimate_structure_memory  # noqa: E402

TAG = "c fused   OFF L8 m4"


def build(seq):
    v = V.BY_TAG[TAG]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    spec.dims.S = seq
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    return spec, ParallelModel(p, spec.dims.n_layers, world)

#: 167 真机 memory-tracker 实测（`docs/kernel_workspace_2026-07-29.md`）——**测量值，不得编辑**
REAL_BWD_WS_MiB = {1024: 332.500, 2048: 465.000, 4096: 730.000}


def main():
    for seq in (1024, 2048, 4096):
        spec, pm = build(seq)
        g = ShapeEval().resolve(spec, pm)
        print("\n=== seq_length = %d ===" % seq)
        seen = set()
        for layers in g.stages.values():
            for lay in layers:
                if lay.layer_type in seen:
                    continue
                seen.add(lay.layer_type)
                sm = estimate_structure_memory(lay.ops)
                per_op = [(op.name, getattr(op, "bwd_workspace_bytes", 0) / MiB)
                          for op in lay.ops if getattr(op, "bwd_workspace_bytes", 0)]
                print("  %-22s fwd_workspace=%8.3f  bwd_workspace=%9.3f MiB   ops=%s"
                      % (lay.layer_type, sm.workspace / MiB, sm.bwd_workspace / MiB,
                         [(n, round(v, 3)) for n, v in per_op] or "-"))
        r4 = [lay for layers in g.stages.values() for lay in layers
              if lay.layer_type == "dsv4hyb_r4_moe"]
        got = estimate_structure_memory(r4[0].ops).bwd_workspace / MiB
        exp = REAL_BWD_WS_MiB[seq]
        print("  r4 model=%.3f  REAL(measured)=%.3f  delta=%+.4f MiB  -> %s"
              % (got, exp, got - exp, "MATCH" if abs(got - exp) < 1e-6 else "MISMATCH"))


if __name__ == "__main__":
    main()
