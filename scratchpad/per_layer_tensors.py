"""评估器侧：逐层型列出「它认为该层要保留哪些张量、各多大」。

对照真机实测（167, 2026-07-29, run c = fused/无重算/L8/m4, seq4096）的**逐层前向增量**：
  r4  ≈ 2355 MiB   (L1 2355.3 / L3 2354.0 / L5 2370.0 / L7 2285.5)
  r128≈ 2116 MiB   (L2 2124.2 / L4 2105.0 / L6 2119.2)
  r0  =  2235.1 MiB (L0)
评估器隐含的每(层×微批) = 3664.6 MiB。
"""
import os
import sys
import warnings
from collections import OrderedDict

warnings.simplefilter("ignore")
REPO = r"E:\97-codes\torch_parallel\pynative-cost-evaluator"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import liveness_ab_validate as G  # noqa: E402
from cost_eval.shape_eval import ShapeEval  # noqa: E402
from cost_eval.parallel_model import ParallelModel  # noqa: E402

MiB = 2 ** 20
REAL = {"r4": 2355.0, "r128": 2116.0, "r0": 2235.1}

v = {x.letter: x for x in G.VARIANTS}["c"]        # fused / 无重算 / L8 / m4
mf = G.derive_mf_config(G.DEFAULT_BASE_DIR, v)
b, spec = G.build_bundle(mf)
pc = b.parallel
world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
pm = ParallelModel(pc, spec.dims.n_layers, world)
rg = ShapeEval().resolve(spec, pm)

# layer_pattern 里认层型
seen = OrderedDict()
for st, layers in sorted(rg.stages.items()):
    for L in layers:
        seen.setdefault(L.layer_type, (st, L))

print("compress_ratios =", list(b.llm.csa_compress_ratios))
print("层型 →", list(seen))
print()

for ltype, (st, L) in seen.items():
    if "dsv4hyb" not in ltype:
        continue
    key = ("r4" if "_r4" in ltype else "r128" if "_r128" in ltype
           else "r0" if "_r0" in ltype else None)
    # 按名去重（structure_mem.py:261-262 就是这么做的）
    dedup = OrderedDict()
    owner = {}
    for op in L.ops:
        for s in op.saves:
            if s.name not in dedup:
                dedup[s.name] = s
                owner[s.name] = op.name
    total = sum(s.local_numel * s.dtype_bytes for s in dedup.values()) / MiB
    print("=" * 96)
    print("层型 %-24s (stage%d, layer_id=%d)   评估器 saves 合计 = %.1f MiB"
          % (ltype, st, L.layer_id, total))
    if key and key in REAL:
        print("   真机实测该层型驻留 = %.1f MiB   →  评估器/真机 = %.2f×"
              % (REAL[key], total / REAL[key]))
    print("-" * 96)
    print("   %-26s %10s %6s %12s  %s" % ("张量", "MiB", "dtype", "numel", "由哪个 op 声明"))
    for nm, s in sorted(dedup.items(), key=lambda kv: -(kv[1].local_numel * kv[1].dtype_bytes)):
        mib = s.local_numel * s.dtype_bytes / MiB
        if mib < 0.05:
            continue
        print("   %-26s %10.1f %6d %12d  %s" % (nm, mib, s.dtype_bytes, s.local_numel, owner[nm]))
    small = sum(s.local_numel * s.dtype_bytes for s in dedup.values()
                if s.local_numel * s.dtype_bytes / MiB < 0.05) / MiB
    if small:
        print("   %-26s %10.3f" % ("(其余 <0.05MiB 合计)", small))
    print()
