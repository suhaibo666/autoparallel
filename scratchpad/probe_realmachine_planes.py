"""从**仓内既有** profiler 台账逐块清点 loss 区「满 vocab 平面」的共存份数。

**不重跑真机**：只读 `analysis/realmachine/**` 已有的 CSV（每份的 pool high-water 都逐 MiB
命中它自己那条记分卡锚点的 `real`，本脚本一并打出以自证对齐）。

判据（与 `docs/k_ce_recalibration_2026-07-30.md` §2.2 同一把尺）：
  · 一张 fp32 满 vocab 平面 = `4·N·vocab ± 8 KiB`，N = B·S/cp（loss/head 区随 cp 切，
    `cost_eval/layers/head.py:201-206` 的 D-1 权威口径）；
  · 一张 bf16 满 vocab 平面 = `2·N·vocab ± 8 KiB`；
  · 「在世」= `Allocation Time(us) <= t_hw < Release Time(us)`（无 Release → 视为一直在世）；
  · `t_hw` = `Allocation Total Allocated(MB)` 取最大值那一行的分配时刻。

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_realmachine_planes.py
"""
from __future__ import annotations

import csv
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MiB = 2 ** 20

# (label, path, N=B*S/cp, vocab, H, 记分卡锚点 real MiB)
CAMPAIGNS = [
    ("pp2-stage1 (loss,k_ce=7)", "pp2_norecomp/op_816365.csv", 8192, 129280, 1792, 45655.0),
    ("pp2-stage0 (optstep)", "pp2_norecomp/op_816362.csv", 8192, 129280, 1792, 10246.0),
    ("cp2-none (loss,k_ce=3)", "cp2_none/operator_memory.csv", 4096, 129280, 1792, 20119.4),
    ("DSv3 8L none (dp2)", "select_ffn/operator_memory.csv", 4096, 129280, 1792, 19967.3),
    ("select self_attn (keep-FFN)", "select_attn/operator_memory.csv", 4096, 129280, 1792, 18828.2),
    ("select mlp (keep-attn)", "select_mlp/operator_memory.csv", 4096, 129280, 1792, 15764.7),
    ("DSv3 4L full (dp2,sp)", "operator_memory_rank0.csv", 4096, 129280, 1792, 12473.1),
    ("cp2 colossal full 4L (B2)", "cp2_colossal/operator_memory.csv", 4096, 129280, 1792, 12433.0),
    # DSv4-align fused：**seq2048**（`dsv4_fused/DIAGNOSIS.md:3` 逐字「4L seq2048」）→ N = B·S = 2048。
    ("DSv4-fused (base)", "dsv4_fused/operator_memory.csv", 2048, 129280, 7168, 15415.5),
]


def load(path: str):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                sz = int(round(float(r["Size(KB)"]) * 1024))
                t0 = float(r["Allocation Time(us)"])
            except (TypeError, ValueError):
                continue
            try:
                t1 = float(r["Release Time(us)"])
            except (TypeError, ValueError):
                t1 = float("inf")
            try:
                tot = float(r["Allocation Total Allocated(MB)"])
            except (TypeError, ValueError):
                tot = None
            rows.append((r["Name"], sz, t0, t1, tot))
    return rows


def main() -> None:
    base = os.path.join(_REPO, "analysis", "realmachine")
    out = []
    for label, rel, N, V, H, real in CAMPAIGNS:
        rows = load(os.path.join(base, rel))
        fp32, bf16 = 4 * N * V, 2 * N * V
        hw = max((r for r in rows if r[4] is not None), key=lambda r: r[4])
        t = hw[2]
        live = [r for r in rows if r[2] <= t < r[3]]

        def n_of(base_sz: int) -> int:
            return sum(1 for r in live if abs(r[1] - base_sz) <= 8192)

        amb = (N == H)      # 2*N*V == 2*H*V → bf16 logits 平面与 [H,vocab] wgrad 输出同尺寸
        nf, nb = n_of(fp32), n_of(bf16)
        out.append((label, rel, N, real, hw[4], hw[0], nf, nb, amb,
                    dict(Counter(r[0] for r in live if abs(r[1] - fp32) <= 8192)),
                    dict(Counter(r[0] for r in live if abs(r[1] - bf16) <= 8192))))

    print(f"{'锚点':<28} {'N':>5} {'锚点real':>9} {'CSV高水位':>10} {'差':>5} "
          f"{'fp32':>5} {'bf16':>5} {'fp32等效':>8}  high-water 设定者")
    print("-" * 118)
    for (label, rel, N, real, pool, setter, nf, nb, amb, pf, pb) in out:
        eq = "不可判" if amb else f"{nf + nb / 2:.1f}"
        print(f"{label:<28} {N:>5} {real:>9.1f} {pool:>10.2f} {pool - real:>5.2f} "
              f"{nf:>5} {nb:>5} {eq:>8}  {setter}")
    print("\n产出者名字（佐证归属）：")
    for (label, rel, N, real, pool, setter, nf, nb, amb, pf, pb) in out:
        print(f"  {label:<28} fp32={pf}  bf16={pb}"
              + ("   ⚠ N==H → bf16 一列与 [H,vocab] wgrad 输出同尺寸，不可分离" if amb else ""))


if __name__ == "__main__":
    main()
