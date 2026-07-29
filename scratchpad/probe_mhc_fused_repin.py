# -*- coding: utf-8 -*-
"""重钉用探针：pp4/pp8/MTP/185 各锚点在 `mhc_fused` 旋钮 OFF(=今天) / ON(=站点 yaml) 下的读数。

跑法：PYTHONIOENCODING=utf-8 python scratchpad/probe_mhc_fused_repin.py
"""
import os
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serve_explorer as S

REAL_ON = {0: 24153.3, 1: 14641.7, 2: 14097.7, 3: 23508.0}
REAL_OFF = {0: 30395.0, 1: 21019.4, 2: 17799.9, 3: 27822.0}
REAL_MTP = {0: 24153.0, 1: 14641.0, 2: 14100.0, 3: 39898.0}
REAL_PP8 = {0: 24759.0, 1: 12324.0, 2: 11678.0, 3: 11919.0,
            4: 10867.0, 5: 11113.0, 6: 10074.0, 7: 26449.0}

_BASE = {
    "preset": "dsv4_flash", "attn": "dsv4_hybrid",
    "layers": "8", "seq": "4096", "batch": "1", "mtp": "0",
    "experts": "8", "topk": "2", "dense_k": "1",
    "heads": "64", "kv_groups": "1",
    "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
    "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
    "v_head": "512", "vocab": "129280",
    "hc": "4", "dsa_fused": "1", "ce_fused": "1",
    "dp": "2", "tp": "1", "ep": "2", "pp": "4", "cp": "1", "method": "colossal",
    "optimizer": "muon", "opt_dtype": "fp32", "grad_bytes": "4",
    "maxdev_gib": "58",
    "select": "attn", "sel_layers": "", "sel_ops": "", "sel_cfg": "", "vpp": "1", "mbs": "",
    "dp_replicate": "1", "reshard": "default", "cpu_offload": "0", "prefetch": "1", "sp": "",
}

_DEF185 = {"preset": "custom", "ffn": "3072", "select": "attn", "sel_layers": "",
           "sel_ops": "", "sel_cfg": "", "vpp": "1", "mbs": "", "grad_bytes": "4",
           "dp_replicate": "1", "reshard": "default", "cpu_offload": "0", "prefetch": "1",
           "sp": ""}


def _dsv4_q(layers, fused, seq="2048", pp="1", recompute="None", mbs="1", split=None):
    q = dict(_DEF185)
    q.update({
        "preset": "dsv4_flash", "attn": "dsv4_hybrid",
        "layers": str(layers), "seq": seq, "batch": "1", "mtp": "0",
        "experts": "8", "topk": "2", "dense_k": "1",
        "heads": "64", "kv_groups": "1",
        "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
        "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
        "v_head": "512", "vocab": "129280",
        "hc": "4", "dsa_fused": ("1" if fused else "0"), "ce_fused": "1",
        "dp": "2", "tp": "1", "ep": "2", "pp": pp, "cp": "1", "method": "colossal",
        "optimizer": "muon", "opt_dtype": "fp32", "grad_bytes": "4",
        "maxdev_gib": "58", "recompute": recompute, "mbs": mbs,
    })
    if split:
        q["pp_split"] = split
    return q


def peaks(q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = S.eval_config(q)
    assert r.get("ok"), r.get("errors")
    return {st["stage"]: st["peak"] for st in r["stages"]}


def with_mhc(q, on):
    q = dict(q)
    q["mhc_fused"] = "1" if on else ""
    return q


def show(label, q, real=None):
    off = peaks(with_mhc(q, False))
    on = peaks(with_mhc(q, True))
    for s in sorted(off):
        r = (real or {}).get(s)
        rs = (f"  real={r:9.1f}  ratio {off[s]/r:6.3f} -> {on[s]/r:6.3f}" if r else "")
        print(f"{label} s{s}: OFF={off[s]:9.1f}  ON={on[s]:9.1f}  d={on[s]-off[s]:+9.1f}{rs}")
    return off, on


print("--- pp4 ON (full recompute) ---")
show("pp4-ON ", {**_BASE, "recompute": "full"}, REAL_ON)
print("--- pp4 OFF (no recompute) ---")
show("pp4-OFF", {**_BASE, "recompute": "None"}, REAL_OFF)
print("--- pp4 + MTP ---")
show("MTP    ", {**_BASE, "mtp": "1", "recompute": "full", "pp_split": "2,2,2,3"}, REAL_MTP)
print("--- pp8 ---")
show("pp8    ", {**_BASE, "dp": "1", "ep": "1", "pp": "8", "mbs": "8", "recompute": "full"},
     REAL_PP8)

print("--- 185 U1 (4L unfused DSA, seq2048) ---")
show("U1     ", _dsv4_q(4, fused=False), {0: 40194.0})
print("--- 185 U2 (8L unfused DSA, seq2048; real OOM @56010) ---")
show("U2     ", _dsv4_q(8, fused=False), {0: 56010.0})
print("--- 185 F0/F1 (fused DSA, seq2048) ---")
f4o, f4n = show("F0     ", _dsv4_q(4, fused=True), {0: 26499.0})
f8o, f8n = show("F1     ", _dsv4_q(8, fused=True))
print(f"F per-layer: OFF={(f8o[0]-f4o[0])/4:.1f}  ON={(f8n[0]-f4n[0])/4:.1f}"
      f"  (real diff anchor 3109 -> {(f8o[0]-f4o[0])/4/3109:.3f} / {(f8n[0]-f4n[0])/4/3109:.3f})")
print("--- 185 P3-P (8L fused, seq4096, pp4, m8, full) ---")
show("P3P-m8 ", _dsv4_q(8, fused=True, seq="4096", pp="4", recompute="full", mbs="8",
                        split="2,2,2,2"), {0: 25343.5})
print("--- 185 P3-P m4 (中部 stage m-无关对照) ---")
show("P3P-m4 ", _dsv4_q(8, fused=True, seq="4096", pp="4", recompute="full", mbs="4",
                        split="2,2,2,2"))
