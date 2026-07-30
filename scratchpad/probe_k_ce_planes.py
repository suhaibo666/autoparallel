"""`K_CE` 重标定诊断：**模型侧**的 loss 区满 vocab 平面清点（逐锚点）。

对每个 DSv3 族锚点（`scorecard_anchors.dsv3` 同参）报：
  · 峰值事件 / 峰值 MiB / ratio；
  · **loss 层 bwd 事件**（lid 最大的那个 `bwd@`）的 `bwd_scratch` 与 `act_live`；
  · `bwd_scratch ÷ (4·S/cp·B·vocab)` = 模型记的**瞬态 fp32 满 vocab 平面份数**
    （= `K_CE − 1` 若 fat 门开；= 2 若门关）。

用途：与 `analysis/realmachine/**` 既有 profiler 台账里逐块数出来的实测份数对照。
不动任何真机常数；纯读数。

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_k_ce_planes.py
"""
from __future__ import annotations

import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "tests")):
    sys.path.insert(0, _p)
os.environ.setdefault("COST_EVAL_EXTRACTED_ALLOW_PARTIAL", "1")
warnings.filterwarnings("ignore")

MiB = 2 ** 20

from validate_dsv3 import build_dsv3_spec                      # noqa: E402
from cost_eval.specs import (                                   # noqa: E402
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from cost_eval.report import Evaluator                          # noqa: E402
from scorecard_anchors import ATTN, MLP, BOTH, FULL4, FULL8, NONE, _opt, _hw  # noqa: E402


CASES = [
    # (label, N_layers, rc, kwargs, real)
    ("DSv3 4L full (dp2,sp)", 4, FULL4, {}, 12473.1),
    ("DSv3 8L full (dp2)", 8, FULL8, {}, 13953.3),
    ("DSv3 4L full ep=2", 4, FULL4, dict(ep=2), 12474.1),
    ("cp2 colossal full 4L (B2)", 4, FULL4,
     dict(B=2, dp=1, cp=2, method="colossal"), 12433.0),
    ("cp2 ulysses full 4L (B2)", 4, FULL4,
     dict(B=2, dp=1, cp=2, method="ulysses"), 12441.0),
    ("pp2-stage0 (optstep)", 8, NONE, dict(B=2, dp=1, pp=2, mbs=2, stage=0), 10246.0),
    ("pp2-stage1 (loss,k_ce=7)", 8, NONE, dict(B=2, dp=1, pp=2, mbs=2, stage=1), 45655.0),
    ("cp2-none (loss,k_ce=3)", 8, NONE,
     dict(B=2, dp=1, cp=2, method="colossal"), 20119.4),
    ("DSv3 8L none (dp2)", 8, NONE, {}, 19967.3),
    ("select self_attn (keep-FFN)", 8, RecomputeSpec("select", select_ops=ATTN), {}, 18828.2),
    ("select mlp (keep-attn)", 8, RecomputeSpec("select", select_ops=MLP), {}, 15764.7),
    ("select both (=full,退化端)", 8, RecomputeSpec("select", select_ops=BOTH), {}, 13953.3),
]


def run(nl, rc, *, B=1, dp=2, cp=1, pp=1, ep=1, mbs=1, method="colossal", stage=0):
    spec, d, fl = build_dsv3_spec(nl)
    d.B = B
    pc = ParallelConfig(dp_shard=dp, cp=cp, tp=1, ep=ep, pp=pp, sequence_parallel=True,
                        num_microbatches=mbs, context_parallel_method=method)
    rep = Evaluator(spec, pc, _opt, _hw, rc, SwapSpec()).evaluate(record_timeline=True)
    return rep.per_stage[stage], d, cp, B


ROWS = []
for label, nl, rc, kw, real in CASES:
    sp, d, cp, B = run(nl, rc, **kw)
    plane = 4 * (d.S // cp) * B * d.vocab          # 一张满 vocab fp32 平面（bytes）
    bl = [s for s in sp.timeline if s.event.startswith("bwd@")]
    lids = [int(s.event.split("@")[1]) for s in bl]
    top = max(lids)
    loss_ev = max((s for s in bl if int(s.event.split("@")[1]) == top),
                  key=lambda s: s.total_bytes)
    bs = loss_ev.breakdown.bwd_scratch
    ROWS.append((
        label, f"{sp.peak_bytes / MiB:.1f}", f"{real:.1f}",
        f"{sp.peak_bytes / real / MiB:.4f}", sp.peak_event,
        f"bwd@{top}", f"{loss_ev.total_bytes / MiB:.1f}",
        f"{bs / MiB:.1f}", f"{bs / plane:.2f}",
        f"{loss_ev.breakdown.act_live / MiB:.1f}", f"{plane / MiB:.1f}",
    ))

HDR = ("锚点", "peak", "real", "ratio", "峰值事件", "loss层", "loss断面",
       "bwd_scratch", "÷平面", "act_live", "平面 MiB")
w = [max(len(str(r[i])) for r in ROWS + [HDR]) for i in range(len(HDR))]
print(" | ".join(h.ljust(w[i]) for i, h in enumerate(HDR)))
print("-+-".join("-" * x for x in w))
for r in ROWS:
    print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
