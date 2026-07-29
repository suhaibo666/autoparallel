"""PROBE (read-only): D1 nr_moe_frag margin ON vs OFF for the two DSv3 no-recompute anchors."""
import os, sys, warnings
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
warnings.simplefilter("ignore")
import scorecard_anchors as SA
from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import ParallelConfig, SwapSpec
from cost_eval.report import Evaluator
MiB = 2 ** 20

def run(N, rc, factor, *, B=1, dp=2, cp=1, pp=1, ep=1, mbs=1, method="colossal", stage=0):
    spec, d, fl = build_dsv3_spec(N)
    d.B = B
    d.nr_moe_frag_factor = factor
    pc = ParallelConfig(dp_shard=dp, cp=cp, tp=1, ep=ep, pp=pp, sequence_parallel=True,
                        num_microbatches=mbs, context_parallel_method=method)
    r = Evaluator(spec, pc, SA._opt, SA._hw, rc, SwapSpec()).evaluate()
    return r.per_stage[stage].peak_bytes / MiB

cases = {
    "DSv3 8L none (dp2)": (dict(N=8, rc=SA.NONE), 19967.3),
    "cp2-none (loss,k_ce=4)": (dict(N=8, rc=SA.NONE, B=2, dp=1, cp=2, method="colossal"), 20119.4),
}
_dflt = build_dsv3_spec(4)[1].nr_moe_frag_factor
print("default nr_moe_frag_factor =", _dflt)
for label, (kw, real) in cases.items():
    on = run(factor=_dflt, **kw)
    off = run(factor=0.0, **kw)
    print(f"{label}: ON={on:.1f} ({on/real:.4f})  OFF={off:.1f} ({off/real:.4f})  real={real}")
