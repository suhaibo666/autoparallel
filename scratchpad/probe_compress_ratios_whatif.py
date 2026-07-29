# -*- coding: utf-8 -*-
"""what-if 探针（**只观测，不改源**）：扁平 query 路的 `csa_compress_ratios` 也没有 UI 键。

发现（2026-07-30，`docs/fused_mhc_branch_mismatch_2026-07-30.md` §9）：pp4/pp8/185 锚点用
`preset=dsv4_flash` 的**循环**压缩比 `(0,4,128,0,4,128,0,4)`，而它们比对的站点 yaml 是
**逐层表** `[0,4,128,4,128,4,128,4]`
（`analysis/realmachine/ab_fusion_2026-07-25/dsv4h_fused_pp4_recomp.yaml`）。
层型分布 3×r0/3×r4/2×r128 vs 1×r0/4×r4/3×r128 —— 与 mHC 分支错配**同一个 bug class**
（扁平 query 表达不了的字段静默取预设值）。

本探针 monkeypatch `parse_and_validate` 覆盖该字段，量化影响；**不改任何源文件**。
跑法：PYTHONIOENCODING=utf-8 python scratchpad/probe_compress_ratios_whatif.py
"""
import dataclasses
import os
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serve_explorer as S

SITE = (0, 4, 128, 4, 128, 4, 128, 4)   # dsv4h_fused_pp4_recomp.yaml compress_ratios

REAL_ON = {0: 24153.3, 1: 14641.7, 2: 14097.7, 3: 23508.0}
REAL_OFF = {0: 30395.0, 1: 21019.4, 2: 17799.9, 3: 27822.0}
REAL_PP8 = {0: 24759.0, 1: 12324.0, 2: 11678.0, 3: 11919.0,
            4: 10867.0, 5: 11113.0, 6: 10074.0, 7: 26449.0}

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tests"))
from test_pp4_recompute_anchor import _BASE   # noqa: E402

_orig = S.parse_and_validate


def _patched(p):
    errs, cfg, pa = _orig(p)
    if cfg is not None and len(cfg.csa_compress_ratios) == len(SITE):
        cfg = dataclasses.replace(cfg, csa_compress_ratios=SITE)
    return errs, cfg, pa


def peaks(q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = S.eval_config(q)
    assert r.get("ok"), r.get("errors")
    return {st["stage"]: st["peak"] for st in r["stages"]}


def run(label, q, real):
    a = peaks(q)
    S.parse_and_validate = _patched
    try:
        b = peaks(q)
    finally:
        S.parse_and_validate = _orig
    for s in sorted(a):
        r = real.get(s)
        print(f"{label} s{s}: preset-cycle={a[s]:9.1f} ({a[s]/r:5.3f})  "
              f"site-table={b[s]:9.1f} ({b[s]/r:5.3f})  d={b[s]-a[s]:+8.1f}")


errs, cfg, pa = _orig(dict(_BASE, recompute="None"))
print("今天扁平 query 路解析出的 csa_compress_ratios =", cfg.csa_compress_ratios)
print("站点 yaml 的 compress_ratios                  =", SITE)
print()
run("pp4-ON ", dict(_BASE, recompute="full"), REAL_ON)
run("pp4-OFF", dict(_BASE, recompute="None"), REAL_OFF)
run("pp8    ", dict(_BASE, dp="1", ep="1", pp="8", mbs="8", recompute="full"), REAL_PP8)
