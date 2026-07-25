"""`python -m cost_eval.liveness <mindformers.yaml>` —— per-stage 峰值 + 峰值时刻逐张量 live-set。

只读交叉校验工具：同时打印桶模型（`Evaluator`）与 liveness 仿真的 per-stage 峰值、二者差额，
以及 liveness 峰值时刻的 live-set dump（按字节降序，含类别标签）与折算到既有桶名的 breakdown。

    python -m cost_eval.liveness cfg.yaml
    python -m cost_eval.liveness cfg.yaml --grad-mode chain2 --top 30 --stage 1
    python -m cost_eval.liveness cfg.yaml --recompute-working-set   # 换成重算工作集时刻的 dump
"""
from __future__ import annotations

import argparse
import sys
import warnings

MiB = 2 ** 20


def _load(path):
    """mindformers yaml → (bundle, ModelSpec)，走与 `serve_explorer` 相同的 adapter 链。"""
    import yaml

    import serve_explorer as S
    from cost_eval.build_llm import build_llm_spec
    from cost_eval.configs.from_mindformers import from_mindformers_dict
    mf = yaml.safe_load(open(path, encoding="utf-8"))
    m = mf.get("model", {})
    # A/B launcher 省略 qk_nope_head_dim；mindformers 由 head_dim − qk_rope_head_dim 导出。
    if not m.get("qk_nope_head_dim") and m.get("head_dim") and m.get("qk_rope_head_dim"):
        m["qk_nope_head_dim"] = int(m["head_dim"]) - int(m["qk_rope_head_dim"])
    mf2, _ = S._mf_adapt(mf)
    S._materialize_nested_offset(mf2, [])
    b = from_mindformers_dict(mf2)
    return b, build_llm_spec(b.llm)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cost_eval.liveness")
    ap.add_argument("config", help="mindformers 训练 yaml")
    ap.add_argument("--grad-mode", default="dataflow", choices=("dataflow", "chain2"))
    ap.add_argument("--top", type=int, default=25, help="live-set dump 条数")
    ap.add_argument("--stage", type=int, default=None, help="只 dump 该 stage（默认最紧 stage）")
    ap.add_argument("--recompute-working-set", action="store_true",
                    help="dump「重算工作集最大」时刻而非总峰值时刻的 live-set")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")

    from cost_eval.liveness import simulate_liveness
    from cost_eval.report import Evaluator
    b, spec = _load(args.config)
    rep = Evaluator(b.llm and spec, b.parallel, b.optimizer, b.hardware,
                    b.recompute, b.swap, check_feasibility=False).evaluate()
    res = simulate_liveness(spec, b.parallel, b.optimizer, b.hardware,
                            b.recompute, b.swap, record_timeline=True,
                            grad_mode=args.grad_mode)

    print(f"config={args.config}  grad_mode={args.grad_mode}")
    print("%-7s %12s %12s %10s  %-14s %-22s" % (
        "stage", "bucket_MiB", "liveness_MiB", "live/buck", "bucket_event", "liveness_event"))
    for i, st in enumerate(res.per_stage):
        bk = rep.per_stage[i].peak_bytes / MiB
        lv = st.peak_bytes / MiB
        print("%-7d %12.1f %12.1f %10.3f  %-14s %-22s" % (
            i, bk, lv, lv / bk if bk else 0, rep.per_stage[i].peak_event,
            f"{st.peak_event}/{st.peak_substep}"))
    stage = args.stage if args.stage is not None else res.tightest_stage
    st = res.per_stage[stage]
    items = (sorted(st.max_recompute_live_set, key=lambda it: -it.nbytes)
             if args.recompute_working_set else st.top_live(10 ** 9))
    label = ("recompute-working-set 峰" if args.recompute_working_set else "总峰值")
    print(f"\n── stage{stage} {label}时刻 live-set（{st.peak_event}/{st.peak_substep}）"
          f"  重算工作集峰={st.max_recompute_working_set / MiB:.1f} MiB"
          f" @{st.max_recompute_substep} ──")
    print("%-26s %-6s %-4s %10s  %s" % ("tensor", "layer", "mb", "MiB", "category"))
    for it in items[:args.top]:
        print("%-26s %-6d %-4d %10.1f  %s" % (it.name, it.layer_id, it.mb,
                                              it.nbytes / MiB, it.category))
    if len(items) > args.top:
        rest = sum(i.nbytes for i in items[args.top:]) / MiB
        print("%-26s %-6s %-4s %10.1f  (%d 个尾部张量)"
              % ("…", "", "", rest, len(items) - args.top))
    print("\n── 峰值 breakdown（liveness 类别 → 既有桶名）──")
    for k, v in sorted(st.bucket_view().items(), key=lambda kv: -kv[1]):
        print("  %-18s %10.1f MiB" % (k, v / MiB))
    print("  %-18s %10.1f MiB" % ("TOTAL", st.peak_bytes / MiB))
    return 0


if __name__ == "__main__":
    sys.exit(main())
