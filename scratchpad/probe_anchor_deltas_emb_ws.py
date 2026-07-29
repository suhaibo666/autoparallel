"""逐锚点 before/after 台账（本轮 embedding 反向 kernel workspace 入账）。

在**两棵树**里各跑一遍（before = 裸 HEAD 的 worktree），逐行对比。
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_anchor_deltas_emb_ws.py
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import serve_explorer as S            # noqa: E402
from scorecard_anchors import anchors  # noqa: E402

MiB = 2 ** 20
OUT = []


def p(label, val, real=None, ev=None):
    r = f"  ratio={val / real:.4f}" if real else ""
    e = f"  @{ev}" if ev else ""
    OUT.append(f"{label:<44} {val:10.1f}{r}{e}")


def scorecard():
    OUT.append("### scorecard_anchors ###")
    for a in anchors():
        sim = a.sim_fn()
        if sim is None:
            OUT.append(f"{a.label:<44}       ERR")
            continue
        p(a.label, sim, a.real)


_DSV4 = {
    "preset": "dsv4_flash", "attn": "dsv4_hybrid",
    "layers": "8", "seq": "4096", "batch": "1", "mtp": "0",
    "experts": "8", "topk": "2", "dense_k": "1", "heads": "64", "kv_groups": "1",
    "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
    "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
    "v_head": "512", "vocab": "129280",
    "hc": "4", "mhc_fused": "1", "dsa_fused": "1", "ce_fused": "1",
    "dp": "2", "tp": "1", "ep": "2", "pp": "4", "cp": "1", "method": "colossal",
    "optimizer": "muon", "opt_dtype": "fp32", "grad_bytes": "4", "maxdev_gib": "58",
    "select": "attn", "sel_layers": "", "sel_ops": "", "sel_cfg": "", "vpp": "1",
    "mbs": "", "dp_replicate": "1", "reshard": "default", "cpu_offload": "0",
    "prefetch": "1", "sp": "",
}


def _std_q(kv):
    return {
        "preset": "custom", "attn": "gqa", "layers": "8", "seq": "4096", "batch": "1",
        "mtp": "0", "experts": "0", "topk": "1", "dense_k": "8",
        "heads": "32", "kv_groups": str(kv), "hidden": "2048",
        "ffn": "8192", "moe_ffn": "8192", "vocab": "129280", "hc": "1",
        "dp": "2", "tp": "1", "ep": "1", "pp": "2", "cp": "1", "method": "colossal",
        "optimizer": "adamw", "opt_dtype": "bf16", "grad_bytes": "4",
        "maxdev_gib": "58", "recompute": "full", "mbs": "4", "pp_split": "4,4",
        "emb_bytes": "4",
        "q_lora": "", "kv_lora": "", "qk_nope": "", "qk_rope": "", "v_head": "",
        "select": "attn", "sel_layers": "", "sel_ops": "", "sel_cfg": "", "vpp": "1",
        "dp_replicate": "1", "reshard": "default", "cpu_offload": "0",
        "prefetch": "1", "sp": "", "mhc_fused": "0", "dsa_fused": "0", "ce_fused": "0",
    }


def explorer():
    OUT.append("### explorer 锚点（185 / pp4 / pp8 / std）###")
    cases = [
        ("pp4 ON", dict(recompute="full")),
        ("pp4 OFF", dict(recompute="None")),
        ("pp4 ON MTP", dict(recompute="full", mtp="1")),
        ("pp8 ON", dict(dp="1", ep="1", pp="8", mbs="8", recompute="full")),
        ("185 P3-P m8", dict(recompute="full", mbs="8")),
    ]
    for label, over in cases:
        q = dict(_DSV4)
        q.update(over)
        r = S.eval_config(q)
        if not r.get("ok"):
            OUT.append(f"{label}: ERR {r.get('errors')}")
            continue
        for st in r["stages"]:
            p(f"{label} s{st['stage']}", st["peak"], ev=st["peak_event"])
    for kv in (32, 8):
        r = S.eval_config(_std_q(kv))
        if not r.get("ok"):
            OUT.append(f"185 std kv={kv}: ERR {r.get('errors')}")
            continue
        for st in r["stages"]:
            p(f"185 std ON kv={kv} s{st['stage']}", st["peak"], ev=st["peak_event"])


def std116():
    """走 `tests/test_std_attn_anchor.py` 自己的 `_peaks`（含 _BUILD_FACTS 等隐藏字段）。"""
    OUT.append("### 116 std（test_std_attn_anchor._peaks 路径）###")
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "tests"))
    import test_std_attn_anchor as T
    for attn, kv in (("mha", 32), ("gqa", 8)):
        for pp in (2, 1):
            pk = T._peaks(kv, pp)
            for st, v in sorted(pk.items()):
                p(f"116 std {attn} pp{pp} s{st}", v, T.REAL_116.get((attn, pp), {}).get(st))


def main():
    scorecard()
    explorer()
    try:
        std116()
    except Exception as e:
        OUT.append(f"[116 std] 跳过：{type(e).__name__} {e}")
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
