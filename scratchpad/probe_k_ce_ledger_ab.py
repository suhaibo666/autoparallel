"""`K_CE` 重标定的**逐锚点 before/after** 台账（同一进程内两跑，逐行 join，不手数）。

`before` = 把 `mem_timeline.K_CE_PP1/K_CE_PP` 临时改回 `4/8`（旧值）再跑；
`after`  = 当前树的值。两跑都走 `probe_head_bwd_ws_ledger` 完全同一条取数路径
（同一 `Evaluator`、同一 anchor 集合），故差异只可能来自这两个常数。

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_k_ce_ledger_ab.py
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "tests")):
    sys.path.insert(0, _p)
os.environ.setdefault("COST_EVAL_EXTRACTED_ALLOW_PARTIAL", "1")
warnings.filterwarnings("ignore")

# 先把会 `sys.stdout.reconfigure(...)` 的模块导入完（它们在 import 期动 stdout，
# 而下面 `redirect_stdout` 到 StringIO 时那个调用会炸）。
import serve_explorer as _S             # noqa: E402,F401
import validate_dsv3 as _V             # noqa: E402,F401
import validate_dsv4align as _V4        # noqa: E402,F401  少了它，两条 DSv4 锚点会静默缺行
import scorecard_anchors as _A          # noqa: E402,F401
import cost_eval.mem_timeline as MT     # noqa: E402

OLD = {"K_CE_PP1": 4, "K_CE_PP": 8, "K_CE_LEAN": 4}
NEW = {k: getattr(MT, k) for k in OLD}


def run(vals):
    for k, v in vals.items():
        setattr(MT, k, v)
    import scratchpad.probe_head_bwd_ws_ledger as L
    importlib.reload(L)
    L.ROWS.clear()
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        L.main()
    # ROWS: (label, sim, real, ratio, peak_event, head_gap, note)
    return {r[0]: (float(r[1]), r[2], r[3], r[4]) for r in L.ROWS}


before = run(OLD)
after = run(NEW)
assert set(before) == set(after)

print(f"K_CE before = {OLD}")
print(f"K_CE after  = {NEW}\n")
hdr = ("锚点/stage", "real", "sim before", "sim after", "Δ MiB",
       "ratio before", "ratio after", "峰值事件")
rows = []
for label in before:
    sb, real, rb, ev = before[label]
    sa, _, ra, ev2 = after[label]
    rows.append((label, real, f"{sb:.1f}", f"{sa:.1f}", f"{sa - sb:+.1f}", rb, ra, ev2))
w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
print(" | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
print("-+-".join("-" * x for x in w))
for r in rows:
    print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def _f(x):
    try:
        return float(x)
    except ValueError:
        return None


moved = [r for r in rows if abs(float(r[4])) > 1e-6]
scored = [r for r in rows if _f(r[5]) is not None]
ub = [r for r in scored if _f(r[5]) < 1.0]
ua = [r for r in scored if _f(r[6]) < 1.0]
flip_dn = [r for r in scored if _f(r[5]) >= 1.0 > _f(r[6])]
flip_up = [r for r in scored if _f(r[5]) < 1.0 <= _f(r[6])]
print(f"\n条目总数 {len(rows)}；可评分（有 real）{len(scored)}；**位移** {len(moved)}、"
      f"逐位不动 {len(rows) - len(moved)}")
print(f"OOM-不安全（ratio<1）: before {len(ub)} → after {len(ua)}")
print(f"由 ≥1.0 跌到 <1.0（转 OOM-不安全）{len(flip_dn)} 条: "
      f"{[(r[0], r[5], r[6]) for r in flip_dn]}")
print(f"由 <1.0 升到 ≥1.0 {len(flip_up)} 条: {[(r[0], r[5], r[6]) for r in flip_up]}")
print("\n位移明细：")
for r in moved:
    print(f"  {r[0]:<28} real={r[1]:>8}  {r[2]:>8} → {r[3]:>8}  ({r[4]:>9})  "
          f"ratio {r[5]} → {r[6]}")
print("\n上一轮翻过 1.0 的那 15 条，本轮落在哪里（ratio_before ∈ [1.02, 1.09]）：")
band = [r for r in scored if 1.02 <= _f(r[5]) <= 1.09]
for r in sorted(band, key=lambda r: -_f(r[5])):
    tag = "**本轮下移**" if abs(float(r[4])) > 1e-6 else "逐位不动"
    print(f"  {r[0]:<28} {r[5]} → {r[6]}   {tag}")
print(f"  （该带内 {len(band)} 条，其中本轮位移 {sum(1 for r in band if abs(float(r[4])) > 1e-6)} 条）")
