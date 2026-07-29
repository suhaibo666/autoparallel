"""PROBE (read-only): dump DSv3-side theory anchors after the norm-kind change."""
import os, sys, warnings
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "tests"))
warnings.simplefilter("ignore")
MiB = 2 ** 20

import tests.test_dsv3_golden as G
p = G._eval(G._spec())
print("--- dsv3_golden ---")
print("breakdown:", {k: getattr(p.breakdown, k) for k in G.GOLDEN_BREAKDOWN})
print("peak_event:", p.peak_event, "peak_bytes:", p.peak_bytes, "MiB:", round(p.peak_bytes / MiB, 4))

import tests.test_x4_p2p_pp as P
from cost_eval.report import Evaluator
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from validate_dsv3 import build_dsv3_spec
GiB = 1024 * MiB
spec, d, fl = build_dsv3_spec(8)
d.B = 2
pc = ParallelConfig(dp_shard=1, cp=1, tp=1, ep=1, pp=2, sequence_parallel=True, num_microbatches=2)
r = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True),
              HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0),
              RecomputeSpec("None"), SwapSpec()).evaluate()
print("--- x4_p2p_pp2 ---")
for i in (0, 1):
    print(f"  s{i}: {r.per_stage[i].peak_bytes / MiB:.1f}  event={r.per_stage[i].peak_event}")

import tests.test_std_attn_anchor as S
print("--- std_attn_anchor ---")
for k in dir(S):
    if k.isupper():
        print(" ", k, "=", getattr(S, k))
