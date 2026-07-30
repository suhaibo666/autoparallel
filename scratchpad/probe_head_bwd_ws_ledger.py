"""lm_head 反向 workspace 入账：**逐锚点台账**（ratio + 峰值事件 + head 段头寸）。

（由 `probe_mhc_fwd_ws_ledger.py` 复制改造：把「前向头寸」换成「bwd@<head> 距峰」，
其余取数路径逐字未动，故两轮的锚点集合与 ratio 定义可直接对照。）

一张表回答任务书的三问：
  · 每个锚点对真机的比值（本项前后）；
  · 峰值事件是 FWD 还是 BWD（= 前向侧的项**结构上**够不够得着）；
  · 若够不着，还差多少（`peak − max(fwd:<lid>)`；只有 `fwd:<lid>` 那一枚带
    `Buckets.workspace`，`fwd_end` 在清零之后，见 `cost_eval/mem_timeline.py:583-585`）。

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_mhc_fwd_ws_ledger.py
"""
from __future__ import annotations

import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "tools"), os.path.join(_REPO, "tests")):
    sys.path.insert(0, _p)
os.environ.setdefault("COST_EVAL_EXTRACTED_ALLOW_PARTIAL", "1")
warnings.filterwarnings("ignore")

MiB = 2 ** 20
ROWS = []


def _add(label, sim, real, sp=None, mhc=None):
    ratio = f"{sim / real:.4f}" if real else "—"
    if sp is None:
        ROWS.append((label, f"{sim:.1f}", f"{real:.1f}" if real else "—", ratio, "—", "—", mhc or "—"))
        return
    # head 段头寸 = peak − (lid 最大的那个 bwd@ 事件的断面)。lid 最大的层就是 lm_head 段
    # （`build_llm.py:313,325`：layer_pattern 恒为 [embedding] + decoder* + [lm_head]），
    # 故本列只在**含 head 的那个 stage**（末 stage）上有意义，其余 stage 记 "n/a"。
    bl = [s for s in sp.timeline if s.event.startswith("bwd@")]
    lids = [int(s.event.split("@")[1]) for s in bl] if bl else []
    if lids:
        top = max(lids)
        cut = max(s.total_bytes for s in bl if int(s.event.split("@")[1]) == top)
        head = f"{sp.peak_bytes / MiB - cut / MiB:.1f}"
    else:
        head = "—"
    ROWS.append((label, f"{sim:.1f}", f"{real:.1f}" if real else "—", ratio,
                 sp.peak_event, head, mhc or "—"))


def _eval(p):
    import serve_explorer as S
    from cost_eval.report import Evaluator
    errs, cfg, pa = S.parse_and_validate(p)
    assert not errs, errs
    spec = S.build_llm_spec(cfg)
    pc, opt, hw, swap = S._build_eval_specs(p, pa)
    rep = Evaluator(spec, pc, opt, hw, S._rc_from_pa(pa), swap).evaluate(record_timeline=True)
    return rep, cfg


def main():
    import test_pp4_recompute_anchor as A
    import test_probe185_recon as P
    import test_std_attn_anchor as T
    from scorecard_anchors import anchors

    # ── pp4 ON / OFF / MTP / pp8 / P3-P ────────────────────────────────────────
    for tag, over, real in (
            ("pp4 ON", {"recompute": "full"}, A.REAL_ON),
            ("pp4 OFF", {"recompute": "None"}, A.REAL_OFF),
            ("pp4 MTP", {"recompute": "full", "mtp": "1", "pp_split": "2,2,2,3"}, A.REAL_MTP),
            ("pp8 ON", {"dp": "1", "ep": "1", "pp": "8", "mbs": "8", "recompute": "full"},
             A.REAL_PP8)):
        q = dict(A._BASE)
        q.update(over)
        rep, cfg = _eval(q)
        for sp in rep.per_stage:
            _add(f"{tag} s{sp.stage}", sp.peak_bytes / MiB, real.get(sp.stage),
                 sp, f"fused={cfg.use_fused_mhc}")

    rep, cfg = _eval(P._dsv4_q(8, fused=True, seq="4096", pp="4", recompute="full", mbs="8"))
    _add("185 P3-P m8 s0", rep.per_stage[0].peak_bytes / MiB, 25343.5,
         rep.per_stage[0], f"fused={cfg.use_fused_mhc}")

    for nm, q, real in (("185 F0 4L", P._dsv4_q(4, fused=True), 26499.0),
                        ("185 U1 4L", P._dsv4_q(4, fused=False), 40194.0),
                        ("185 U2 8L", P._dsv4_q(8, fused=False), 56010.0)):
        rep, cfg = _eval(q)
        _add(nm, rep.per_stage[0].peak_bytes / MiB, real, rep.per_stage[0],
             f"fused={cfg.use_fused_mhc}")

    for kv in (32, 8):
        rep, cfg = _eval(P._std_on_q(kv))
        for sp in rep.per_stage:
            _add(f"185 std ON kv{kv} s{sp.stage}", sp.peak_bytes / MiB,
                 P._STD_ON_REAL[kv].get(sp.stage), sp, "无 mHC")

    for attn, kv in (("mha", 32), ("gqa", 8)):
        for pp in (2, 1):
            for st, v in sorted(T._peaks(kv, pp).items()):
                _add(f"116 std {attn} pp{pp} s{st}", v,
                     T.REAL_116.get((attn, pp), {}).get(st), None, "无 mHC")

    for a in anchors():
        sim = a.sim_fn()
        if sim is not None:
            _add(a.label, sim, a.real, None, "无 mHC / 非融合")

    w = [max(len(str(r[i])) for r in ROWS) for i in range(7)]
    hdr = ("锚点", "sim", "real", "ratio", "峰值事件", "head距峰", "备注")
    w = [max(w[i], len(hdr[i])) for i in range(7)]
    print(" | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("-+-".join("-" * x for x in w))
    for r in ROWS:
        print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    bwd = [r for r in ROWS if r[4].startswith("bwd@")]
    fwd = [r for r in ROWS if r[4].startswith("fwd")]
    print(f"\n有事件线的锚点：{len(bwd) + len(fwd)}  其中峰在 BWD={len(bwd)}  峰在 FWD={len(fwd)}")
    under = [r for r in ROWS if r[3] != "—" and float(r[3]) < 1.0]
    print(f"ratio < 1.0（OOM-不安全）的锚点/stage 数：{len(under)}")


if __name__ == "__main__":
    main()
