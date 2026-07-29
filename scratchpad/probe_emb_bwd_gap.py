"""探针：各 explorer 锚点里 `bwd@0`（embedding 伪层反向）距该 stage 峰值有多远。

用法： PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_emb_bwd_gap.py
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import serve_explorer as S  # noqa: E402

_BASE = {
    "preset": "dsv4_flash", "attn": "dsv4_hybrid",
    "layers": "8", "seq": "4096", "batch": "1", "mtp": "0",
    "experts": "8", "topk": "2", "dense_k": "1",
    "heads": "64", "kv_groups": "1",
    "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
    "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
    "v_head": "512", "vocab": "129280",
    "hc": "4", "mhc_fused": "1", "dsa_fused": "1", "ce_fused": "1",
    "dp": "2", "tp": "1", "ep": "2", "pp": "4", "cp": "1", "method": "colossal",
    "optimizer": "muon", "opt_dtype": "fp32", "grad_bytes": "4",
    "maxdev_gib": "58",
    "select": "attn", "sel_layers": "", "sel_ops": "", "sel_cfg": "", "vpp": "1",
    "mbs": "", "dp_replicate": "1", "reshard": "default", "cpu_offload": "0",
    "prefetch": "1", "sp": "",
}

CASES = [
    ("pp4 ON  (185 F)", {"recompute": "full"}),
    ("pp4 OFF", {"recompute": "None"}),
    ("pp4 ON +MTP", {"recompute": "full", "mtp": "1"}),
    ("pp8 ON", {"dp": "1", "ep": "1", "pp": "8", "mbs": "8", "recompute": "full"}),
    ("185 P3-P (m8)", {"recompute": "full", "mbs": "8"}),
]


def main():
    for label, over in CASES:
        p = dict(_BASE)
        p.update(over)
        r = S.eval_config(p)
        if not r.get("ok"):
            print(f"{label}: ERR {r.get('errors')}")
            continue
        print(f"\n── {label} ──")
        for st in r["stages"]:
            tl = st["timeline"]
            peak = st["peak"]
            bwd = [e for e in tl if e["event"].startswith("bwd@")]
            if not bwd:
                continue
            lids = sorted({int(e["event"].split("@")[1].split("#")[0]) for e in bwd})
            first = max((e for e in bwd
                         if int(e["event"].split("@")[1].split("#")[0]) == lids[0]),
                        key=lambda e: e["total"])
            print(f"   s{st['stage']}: peak={peak:9.1f}@{st['peak_event']:<10} "
                  f"bwd@{lids[0]}={first['total']:9.1f}  差={peak - first['total']:8.1f}")


if __name__ == "__main__":
    main()
