"""167 A/B 八跑验证表：**真机 vs 桶模型 vs liveness 仿真**（2026-07-25）。

真机数据来自 `192.168.9.167:/home/suhaibo/workspace/log_ab_fusion_2026-07-25/<run>/worker_<N>.log`
的 `[MEMPROBE] rank=N peak_alloc_MiB=...` 行（rank→stage 映射 = `rank // (world/pp)` = `rank//2`）。
八跑的 launcher yaml 之间**只**差 5 个字段（已在 167 上 grep 核对）：
``training.global_batch_size`` / ``recompute.mode`` + ``full_recompute_layer`` /
``model.num_hidden_layers`` / ``model.compress_ratios`` / ``model.apply_dsa_kernel_fusion``，
故此处从两份 base yaml（fused / unfused，pp4 全重算 L8 m4）派生其余六份，而不是各存一份。

用法：
    python tools/liveness_ab_validate.py --base-dir <含两份 base yaml 的目录>
    python tools/liveness_ab_validate.py --base-dir <dir> --grad-mode chain2 --deltas
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import warnings

MiB = 2 ** 20

# 167, 2026-07-25, peak_alloc_MiB per stage（None = 该跑 OOM，无可比值）。
REAL = {
    "a fused   ON  L8 m4": {0: 24153.3, 1: 14641.7, 2: 14097.7, 3: 23508.0},
    "b unfused ON  L8 m4": {0: 50187.9, 1: 43940.0, 2: 43407.1, 3: 47888.1},
    "c fused   OFF L8 m4": {0: 30391.6, 1: 21019.4, 2: 17759.1, 3: 27720.1},
    "d unfused OFF L8 m4": {0: None, 1: None, 2: None, 3: None},   # OOM @s0（曾达 57712.7）
    "e fused   ON  L8 m8": {0: 25343.5, 1: 14631.5, 2: 14097.8, 3: 23508.6},
    "f unfused ON  L8 m8": {0: 51378.2, 1: 43940.5, 2: 43407.6, 3: 47888.6},
    "g fused   ON  L4 m4": {0: 18096.8, 1: 8867.9, 2: 7845.3, 3: 19623.0},
    "h unfused ON  L4 m4": {0: 19862.6, 1: 36582.6, 2: 13750.2, 3: 44003.1},
}
# tag -> (fused?, global_batch_size, recompute_on, num_hidden_layers)
VARIANTS = {
    "a fused   ON  L8 m4": (True, 8, True, 8),
    "b unfused ON  L8 m4": (False, 8, True, 8),
    "c fused   OFF L8 m4": (True, 8, False, 8),
    "d unfused OFF L8 m4": (False, 8, False, 8),
    "e fused   ON  L8 m8": (True, 16, True, 8),
    "f unfused ON  L8 m8": (False, 16, True, 8),
    "g fused   ON  L4 m4": (True, 8, True, 4),
    "h unfused ON  L4 m4": (False, 8, True, 4),
}
RATIOS = {8: [0, 4, 128, 4, 128, 4, 128, 4], 4: [0, 4, 128, 4]}
BASE = {True: "dsv4h_fused_pp4_recomp.yaml", False: "dsv4h_unfused_pp4_recomp.yaml"}
PAIRS = (("L8 m4", "a fused   ON  L8 m4", "b unfused ON  L8 m4"),
         ("L8 m8", "e fused   ON  L8 m8", "f unfused ON  L8 m8"),
         ("L4 m4", "g fused   ON  L4 m4", "h unfused ON  L4 m4"))


def _build(base_dir, tag):
    import yaml

    import serve_explorer as S
    from cost_eval.build_llm import build_llm_spec
    from cost_eval.configs.from_mindformers import from_mindformers_dict
    fused, gbs, rc_on, nl = VARIANTS[tag]
    with open(os.path.join(base_dir, BASE[fused]), encoding="utf-8") as fh:
        mf = copy.deepcopy(yaml.safe_load(fh))
    mf["training"]["global_batch_size"] = gbs
    mf["model"]["num_hidden_layers"] = nl
    mf["model"]["compress_ratios"] = list(RATIOS[nl])
    mf["model"]["apply_dsa_kernel_fusion"] = fused
    mf["recompute"] = ({"mode": "full", "full_recompute_layer": ["0-%d" % (nl - 1)]}
                       if rc_on else {"mode": "None"})
    m = mf["model"]
    if not m.get("qk_nope_head_dim"):          # launcher 省略；mindformers 由 head_dim − rope 导出
        m["qk_nope_head_dim"] = int(m["head_dim"]) - int(m["qk_rope_head_dim"])
    mf2, _ = S._mf_adapt(mf)
    S._materialize_nested_offset(mf2, [])
    b = from_mindformers_dict(mf2)
    return b, build_llm_spec(b.llm)


def _run(base_dir, tag, grad_mode):
    from cost_eval.liveness import simulate_liveness
    from cost_eval.report import Evaluator
    b, spec = _build(base_dir, tag)
    rep = Evaluator(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                    check_feasibility=False).evaluate()
    lv = simulate_liveness(spec, b.parallel, b.optimizer, b.hardware, b.recompute,
                           b.swap, record_timeline=True, grad_mode=grad_mode)
    return b, rep, lv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True,
                    help="含 dsv4h_{fused,unfused}_pp4_recomp.yaml 的目录")
    ap.add_argument("--grad-mode", default="dataflow", choices=("dataflow", "chain2"))
    ap.add_argument("--deltas", action="store_true", help="另打 unfused−fused 差值表")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    cache, agg = {}, {"bucket": [], "liveness": []}
    print("grad_mode=%s" % args.grad_mode)
    print("%-20s %-6s %10s %10s %10s %9s %9s  %s" % (
        "run", "stage", "real", "bucket", "liveness", "buck/re", "live/re", "liveness peak"))
    for tag in VARIANTS:
        cache[tag] = _run(args.base_dir, tag, args.grad_mode)
        _, rep, lv = cache[tag]
        for i, st in enumerate(lv.per_stage):
            real = REAL[tag][i]
            bk, li = rep.per_stage[i].peak_bytes / MiB, st.peak_bytes / MiB
            if real:
                agg["bucket"].append(bk / real)
                agg["liveness"].append(li / real)
            print("%-20s %-6d %10s %10.1f %10.1f %9s %9s  %s/%s" % (
                tag if i == 0 else "", i, ("%.1f" % real) if real else "OOM", bk, li,
                ("%.3f" % (bk / real)) if real else "-",
                ("%.3f" % (li / real)) if real else "-", st.peak_event, st.peak_substep))
    for k, v in agg.items():
        print("  %-9s n=%d  mean=%.3f  min=%.3f  max=%.3f"
              % (k, len(v), sum(v) / len(v), min(v), max(v)))

    if args.deltas:
        print("\nunfused − fused delta（重算工作集的 ×1 量）:")
        print("%-8s %-6s %10s %10s %10s %9s %9s"
              % ("cfg", "stage", "real_d", "buck_d", "live_d", "buck/re", "live/re"))
        for lbl, ft, ut in PAIRS:
            for i in range(len(cache[ut][1].per_stage)):
                rd = REAL[ut][i] - REAL[ft][i]
                bd = (cache[ut][1].per_stage[i].peak_bytes
                      - cache[ft][1].per_stage[i].peak_bytes) / MiB
                ld = (cache[ut][2].per_stage[i].peak_bytes
                      - cache[ft][2].per_stage[i].peak_bytes) / MiB
                print("%-8s %-6d %10.1f %10.1f %10.1f %9.3f %9.3f"
                      % (lbl if i == 0 else "", i, rd, bd, ld, bd / rd, ld / rd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
