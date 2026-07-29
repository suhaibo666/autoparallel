"""PROBE (read-only): which mHC branch (fused / unfused) does each anchor path build?

    PYTHONIOENCODING=utf-8 python scratchpad/probe_mhc_branch_of_anchors.py
"""
import os
import sys
import warnings

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, "tests"))
warnings.simplefilter("ignore")

import serve_explorer as S  # noqa: E402
import tests.test_pp4_recompute_anchor as T  # noqa: E402
import validate_dsv4align as V  # noqa: E402

p = dict(T._BASE)
p["recompute"] = "None"
errs, cfg, pa = S.parse_and_validate(p)
assert not errs, errs
print("pp4/pp8 anchor (serve_explorer, preset=dsv4_flash):")
print("   residual_variant=%r n_streams=%r use_fused_mhc=%r dsa_fused=%r"
      % (cfg.residual_variant, cfg.num_residual_streams, cfg.use_fused_mhc, cfg.dsa_fused))

for mhc, mtp in ((0, 0), (4, 1)):
    c = V.make_cfg(num_layers=4, mhc=mhc, mtp=mtp) if hasattr(V, "make_cfg") else None
    if c is None:
        break
    print("scorecard DSv4 (mhc=%d,mtp=%d): residual=%r n=%r use_fused_mhc=%r"
          % (mhc, mtp, c.residual_variant, c.num_residual_streams, c.use_fused_mhc))
