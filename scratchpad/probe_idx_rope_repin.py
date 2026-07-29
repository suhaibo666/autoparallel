"""逐锚点 before/after 台账（本轮 = r4 indexer 内部 RoPE 保留对入账，+128.0 MiB/r4 层）。

只读探针：打每个锚点配置的逐 stage `peak` + `peak_event`，用来**解释**每条重钉的幅度
（尤其「为什么 s0 只 +100.0 而不是 +128.0」这类不整数差）。

    PYTHONIOENCODING=utf-8 python scratchpad/probe_idx_rope_repin.py

NOTHING in cost_eval/ is modified.
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
warnings.filterwarnings("ignore")

import serve_explorer as S            # noqa: E402
from scorecard_anchors import anchors  # noqa: E402

# 与 tests/test_pp4_recompute_anchor.py::_BASE 逐键相同（含 2026-07-30 的
# `mhc_fused` / `compress_ratios` 两个旋钮）——锚点走的就是这条 query。
_BASE = {
    "preset": "dsv4_flash", "attn": "dsv4_hybrid",
    "layers": "8", "seq": "4096", "batch": "1", "mtp": "0",
    "experts": "8", "topk": "2", "dense_k": "1", "heads": "64", "kv_groups": "1",
    "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
    "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
    "v_head": "512", "vocab": "129280",
    "hc": "4", "mhc_fused": "1", "dsa_fused": "1", "ce_fused": "1",
    "compress_ratios": "0,4,128,4,128,4,128,4",
    "dp": "2", "tp": "1", "ep": "2", "pp": "4", "cp": "1", "method": "colossal",
    "optimizer": "muon", "opt_dtype": "fp32", "grad_bytes": "4", "maxdev_gib": "58",
    "select": "attn", "sel_layers": "", "sel_ops": "", "sel_cfg": "", "vpp": "1",
    "mbs": "", "dp_replicate": "1", "reshard": "default", "cpu_offload": "0",
    "prefetch": "1", "sp": "",
}

CASES = [
    ("pp4 ON", dict(recompute="full")),
    ("pp4 OFF", dict(recompute="None")),
    ("pp4 ON MTP", dict(mtp="1", recompute="full", pp_split="2,2,2,3")),
    ("pp8 ON", dict(dp="1", ep="1", pp="8", mbs="8", recompute="full")),
    ("185 P3-P m8", dict(recompute="full", mbs="8")),
]


def explorer():
    print("### explorer 锚点（pp4 / MTP / pp8 / 185 P3-P）###")
    for label, over in CASES:
        q = dict(_BASE)
        q.update(over)
        r = S.eval_config(q)
        if not r.get("ok"):
            print(f"{label}: ERR {r.get('errors')}")
            continue
        for st in r["stages"]:
            print(f"{label} s{st['stage']:<2} {st['peak']:10.1f}  @{st.get('peak_event')}")


def scorecard():
    print("\n### scorecard_anchors ###")
    for a in anchors():
        sim = a.sim_fn()
        if sim is None:
            print(f"{a.label:<44}       ERR")
            continue
        print(f"{a.label:<44} {sim:10.1f}  ratio={sim / a.real:.4f}")


def per_layer_increment():
    """185 fused 逐层差分（tests/test_probe185_recon.py::test_fused_per_layer_increment）。"""
    print("\n### 185 fused 逐层差分 ###")
    import test_probe185_recon as T
    for name in ("_per_layer_increment", "_increment", "per_layer_increment"):
        fn = getattr(T, name, None)
        if fn is not None:
            print(f"  {name}() = {fn():.1f}")
            return
    print("  （未找到差分辅助函数，见测试自身）")


if __name__ == "__main__":
    explorer()
    scorecard()
    try:
        per_layer_increment()
    except Exception as e:                                     # pragma: no cover
        print(f"  [逐层差分] 跳过：{type(e).__name__} {e}")
