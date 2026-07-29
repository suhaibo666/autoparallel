# -*- coding: utf-8 -*-
"""`csa_compress_ratios` 错配：① 锚点扁平 query vs 站点 yaml 的**逐字段**差异；
② 逐锚点 before(预设循环) → after(站点逐层表) 读数（含对真机比值）。

跑法：PYTHONIOENCODING=utf-8 python scratchpad/probe_compress_ratios_repin.py
（**只观测**：after 用 monkeypatch 覆盖字段，不改任何源文件。）
"""
import dataclasses
import os
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

import serve_explorer as S                                       # noqa: E402
from cost_eval.configs.from_mindformers import load_mindformers_yaml   # noqa: E402

SITE_YAML = os.path.join(_REPO, "analysis", "realmachine", "ab_fusion_2026-07-25",
                         "dsv4h_fused_pp4_recomp.yaml")

# ── ① 站点 yaml 的权威 LLMConfig vs pp4 锚点扁平 query ────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    b = load_mindformers_yaml(SITE_YAML)
print("站点 yaml LLMConfig.csa_compress_ratios =", b.llm.csa_compress_ratios)

from test_pp4_recompute_anchor import _BASE                      # noqa: E402
from test_probe185_recon import _dsv4_q                          # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    errs, cfg, _ = S.parse_and_validate(dict(_BASE, recompute="None"))
assert not errs, errs
print("pp4 锚点扁平 query 解析出的     =", cfg.csa_compress_ratios)
print()
print("--- pp4 锚点 query 与站点 yaml 的**全部** LLMConfig 字段差异 ---")
for k, va, vc in S._llm_field_diffs(b.llm, cfg):
    print(f"  {k:36s} yaml={va!r:30s} anchor={vc!r}")
print()

# ── ② 逐锚点 before → after ──────────────────────────────────────────────────────────
SITE8 = (0, 4, 128, 4, 128, 4, 128, 4)   # dsv4h_fused_pp4_recomp.yaml:121（8 层站点表）
SITE4 = (0, 4, 128, 4)                   # analysis/dsv4_flash_calibration_handoff_2026-07-22.md:121
_TABLE = {4: SITE4, 8: SITE8}

_orig = S.parse_and_validate


def _patched(p):
    errs, cfg, pa = _orig(p)
    if cfg is not None and cfg.csa_compress_ratios is not None:
        t = _TABLE.get(len(cfg.csa_compress_ratios))
        if t is not None:
            cfg = dataclasses.replace(cfg, csa_compress_ratios=t)
    return errs, cfg, pa


def peaks(q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = S.eval_config(q)
    assert r.get("ok"), r.get("errors")
    return {st["stage"]: st["peak"] for st in r["stages"]}


def show(label, q, real):
    a = peaks(q)
    S.parse_and_validate = _patched
    try:
        c = peaks(q)
    finally:
        S.parse_and_validate = _orig
    for s in sorted(a):
        r = real.get(s)
        rr = (f"{a[s]/r:6.3f} → {c[s]/r:6.3f}" if r else "        —      ")
        print(f"{label} s{s}: {a[s]:9.1f} → {c[s]:9.1f}  (d={c[s]-a[s]:+8.1f})   real={r}  {rr}")
    return a, c


REAL_ON = {0: 24153.3, 1: 14641.7, 2: 14097.7, 3: 23508.0}
REAL_OFF = {0: 30395.0, 1: 21019.4, 2: 17799.9, 3: 27822.0}
REAL_MTP = {0: 24153.0, 1: 14641.0, 2: 14100.0, 3: 39898.0}
REAL_PP8 = {0: 24759.0, 1: 12324.0, 2: 11678.0, 3: 11919.0,
            4: 10867.0, 5: 11113.0, 6: 10074.0, 7: 26449.0}

print("--- pp4 ON（全重算）---")
show("pp4-ON ", dict(_BASE, recompute="full"), REAL_ON)
print("--- pp4 OFF（无重算）---")
show("pp4-OFF", dict(_BASE, recompute="None"), REAL_OFF)
print("--- pp4 + MTP（尾 stage）---")
show("MTP    ", dict(_BASE, mtp="1", recompute="full", pp_split="2,2,2,3"), REAL_MTP)
print("--- pp8 ---")
show("pp8    ", dict(_BASE, dp="1", ep="1", pp="8", mbs="8", recompute="full"), REAL_PP8)

print("--- 185 U1 (4L unfused-DSA seq2048; real 40194) ---")
show("U1     ", _dsv4_q(4, fused=False), {0: 40194.0})
print("--- 185 U2 (8L unfused-DSA seq2048; real OOM @56010) ---")
show("U2     ", _dsv4_q(8, fused=False), {0: 56010.0})
print("--- 185 F0 (4L fused-DSA seq2048; real 26499) / F1 (8L; real 38936) ---")
f4a, f4c = show("F0     ", _dsv4_q(4, fused=True), {0: 26499.0})
f8a, f8c = show("F1     ", _dsv4_q(8, fused=True), {0: 38936.0})
print(f"       每层差分 (F1-F0)/4: {(f8a[0]-f4a[0])/4:8.1f} → {(f8c[0]-f4c[0])/4:8.1f}"
      f"   (/3109 = {(f8a[0]-f4a[0])/4/3109:.3f} → {(f8c[0]-f4c[0])/4/3109:.3f})")
print("--- 185 P3-P m8 (8L fused seq4096 pp4 全重算; real s0 25343.5) ---")
show("P3P-m8 ", _dsv4_q(8, fused=True, seq="4096", pp="4", recompute="full",
                       mbs="8", split="2,2,2,2"), {0: 25343.5})
print("--- 185 P3-P m4（用于 m 无关不变量）---")
show("P3P-m4 ", _dsv4_q(8, fused=True, seq="4096", pp="4", recompute="full",
                       mbs="4", split="2,2,2,2"), {})
