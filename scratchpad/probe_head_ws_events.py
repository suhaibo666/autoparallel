"""item-2 判据探针：`GatherDGradV2`（embedding 反向）与 `bwd@head`（lm_head 反向）
在**模型自己的事件时间线**上是不是同一个事件；以及 `bwd@<embedding>` 距各锚点峰值多远。

只读；不改任何模型行为。用法：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_head_ws_events.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MiB = 2 ** 20


def _lid(ev: str) -> int:
    return int(ev.split("@")[1].split("#")[0])


def _row(label, sp, n_layers_hint=None):
    tl = sp.timeline
    bwd = [s for s in tl if s.event.startswith("bwd@")]
    if not bwd:
        print(f"{label:<34} peak_event={sp.peak_event:<12} peak={sp.peak_bytes / MiB:9.1f}  (无 bwd 事件)")
        return
    lo_id = min(_lid(s.event) for s in bwd)
    hi_id = max(_lid(s.event) for s in bwd)
    lo_best = max((s for s in bwd if _lid(s.event) == lo_id), key=lambda s: s.total_bytes)
    hi_best = max((s for s in bwd if _lid(s.event) == hi_id), key=lambda s: s.total_bytes)
    peak = sp.peak_bytes / MiB
    emb = lo_best.total_bytes / MiB
    print(f"{label:<34} peak_ev={sp.peak_event:<10} peak={peak:9.1f} | "
          f"bwd@{lo_id}(first)={emb:9.1f} 差={peak - emb:9.1f} | "
          f"bwd@{hi_id}(last)={hi_best.total_bytes / MiB:9.1f}")


def dsv3_anchors():
    from cost_eval.specs import (ParallelConfig, OptimizerSpec, HardwareSpec,
                                 RecomputeSpec, SwapSpec)
    from cost_eval.report import Evaluator
    from validate_dsv3 import build_dsv3_spec

    GiB = 2 ** 30
    _opt = OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4)
    _hw = HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0)
    ATTN = {lid: {"linear_q", "linear_kv", "q_a", "kv_a", "rope", "flash", "o_proj"}
            for lid in range(1, 9)}
    MLP = {lid: {"fc", "swiglu", "gelu", "router", "dispatch", "e_", "combine", "shared"}
           for lid in range(1, 9)}
    NONE = RecomputeSpec("None")

    print("\n──────── DSv3 锚点族（peak vs bwd@0=embedding 伪层）────────")
    cases = [
        ("DSv3 8L none (dp2)", 8, NONE, dict(B=1, dp=2, cp=1, pp=1, mbs=1), 0),
        ("select self_attn", 8, RecomputeSpec("select", select_ops=ATTN),
         dict(B=1, dp=2, cp=1, pp=1, mbs=1), 0),
        ("select mlp", 8, RecomputeSpec("select", select_ops=MLP),
         dict(B=1, dp=2, cp=1, pp=1, mbs=1), 0),
        ("pp2-stage0", 8, NONE, dict(B=2, dp=1, cp=1, pp=2, mbs=2), 0),
        ("pp2-stage1", 8, NONE, dict(B=2, dp=1, cp=1, pp=2, mbs=2), 1),
        ("cp2-none", 8, NONE, dict(B=2, dp=1, cp=2, pp=1, mbs=1), 0),
    ]
    for label, N, rc, kw, stage in cases:
        spec, d, fl = build_dsv3_spec(N)
        d.B = kw["B"]
        pc = ParallelConfig(dp_shard=kw["dp"], cp=kw["cp"], tp=1, ep=1, pp=kw["pp"],
                            sequence_parallel=True, num_microbatches=kw["mbs"],
                            context_parallel_method="colossal")
        rep = Evaluator(spec, pc, _opt, _hw, rc, SwapSpec()).evaluate(record_timeline=True)
        _row(f"{label} s{stage}", rep.per_stage[stage])


def explorer_anchors():
    import serve_explorer as S
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
    print("\n──────── 185/pp4/pp8 锚点族（explorer 路径，逐 stage 峰值事件）────────")
    for label, over in [("pp4 ON  ", {"recompute": "full"}),
                        ("pp4 OFF ", {"recompute": "None"}),
                        ("pp8 ON  ", {"dp": "1", "ep": "1", "pp": "8", "mbs": "8",
                                      "recompute": "full"})]:
        p = dict(_BASE)
        p.update(over)
        r = S.eval_config(p)
        assert r.get("ok"), r.get("errors")
        evs = [(st["stage"], st["peak"], st.get("peak_event")) for st in r["stages"]]
        print(f"  {label}: " + "  ".join(f"s{s}={pk:.1f}@{ev}" for s, pk, ev in evs))


def main():
    dsv3_anchors()
    try:
        explorer_anchors()
    except Exception as e:
        print("[explorer] 跳过：", type(e).__name__, e)


if __name__ == "__main__":
    main()
