"""PROBE (read-only): fused vs unfused mHC branch — op structure + per-module saved bytes."""
import os, sys, warnings
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
warnings.simplefilter("ignore")
from cost_eval.model_spec import DimTable
from cost_eval.layers.residual import build_hyper_connection_ops
from cost_eval.shape_eval import resolve_tensor
from cost_eval.parallel_model import ParallelModel
from cost_eval.specs import ParallelConfig
MiB = 2 ** 20

d = DimTable(H=4096, F=16384, n_heads=64, n_kv=1, head_dim=576, S=4096, B=1,
             vocab=129280, n_layers=8, num_residual_streams=4, mhc_sinkhorn_iterations=20)
pm = ParallelModel(ParallelConfig(dp_shard=1, tp=1, ep=1, pp=1, cp=1), 8, 1)

for fused in (False, True):
    d.use_fused_mhc = fused
    ops = build_hyper_connection_ops("attn", d)
    print(f"--- use_fused_mhc={fused} ({len(ops)} ops) ---")
    tot = 0.0
    for op in ops:
        pn = [p.name for p in op.params]
        print(f"  {op.name:28s} type={op.type.value:12s} params={pn} bwd={op.bwd_scratch}")
        for sv in op.saves:
            r = resolve_tensor(sv, d, pm)
            b = r.local_numel * r.dtype_bytes
            tot += b / MiB
            print(f"      save {sv.name:26s} {b/MiB:9.4f} MiB  dtype={r.dtype_bytes}b")
    print(f"  == per-module saved total: {tot:.4f} MiB")
