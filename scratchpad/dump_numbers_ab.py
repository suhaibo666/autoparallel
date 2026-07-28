# -*- coding: utf-8 -*-
"""**字节中性证明（167 A/B 站点版）**：手写 `ModelSpec` 全字段 dump + 桶模型 `Evaluator` 数字
+ `liveness(hand_spec)` 逐 stage 峰值 —— 供 before/after `diff`。

为什么另起一份（不用 `scratchpad/dump_numbers.py`）：那份 import `validate_dsv3`，而
`validate_dsv3.py` 从未进版本库（前一轮的临时脚本），在任何 clean checkout 上都跑不起来。
本份只依赖仓内既有的 A/B 配置派生链（`tools/liveness_ab_validate`），**站点就是验收门那八跑**，
因此它证的正是验收要求的那件事：`bucket` / `hand_spec` 的数字逐字节不变。

    python scratchpad/dump_numbers_ab.py > after.txt
    （在 base worktree 里同样跑一遍 → before.txt）；diff 必须为空。
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.simplefilter("ignore")

MiB = 1 << 20


def _tags():
    from tools.liveness_ab_validate import VARIANTS
    return list(VARIANTS)


def spec_dump():
    """手写 spec 的**全字段** dump（op 类型 / workspace / 每个张量的每个字段）。"""
    from tools.liveness_ab_validate import DEFAULT_BASE_DIR, build_bundle, derive_mf_config
    out = []
    for v in _tags():
        b, spec = build_bundle(derive_mf_config(DEFAULT_BASE_DIR, v))
        out.append(f"=== {v.tag} dims={sorted(spec.dims.as_dict().items())}")
        out.append(f"    layer_pattern={list(spec.layer_pattern)}")
        for ltype in sorted(spec.layer_specs):
            for op in spec.get_layer(ltype).ops:
                out.append(f"  {ltype}/{op.name} type={op.type.value} "
                           f"ws={op.workspace} bws={op.bwd_scratch} "
                           f"attrs={sorted(op.attrs.items())}")
                for slot, seq in (("in", op.inputs), ("out", [op.output]),
                                  ("par", op.params), ("sav", op.saves)):
                    for t in seq:
                        out.append(
                            f"    {slot} {t.name} shape={t.shape} "
                            f"shard={sorted(t.shard.items())} w={t.is_weight} "
                            f"partial={t.partial} dt={t.dtype_bytes} cp={t.cp_shard} "
                            f"cpkv={t.cp_kv} pin={t.pin_under_recompute} det={t.detached}")
                for nm, r in (("wsref", op.workspace_ref), ("bwsref", op.bwd_scratch_ref)):
                    if r is not None:
                        out.append(f"    {nm} {r.name} shape={r.shape} "
                                   f"shard={sorted(r.shard.items())}")
    return out


def eval_dump():
    """桶模型（`bucket`）与 `liveness(hand_spec)` 的逐 stage 峰值 + 全 breakdown。"""
    from cost_eval.liveness import simulate_liveness
    from cost_eval.report import Evaluator
    from tools.liveness_ab_validate import DEFAULT_BASE_DIR, build_bundle, derive_mf_config
    out = []
    for v in _tags():
        b, spec = build_bundle(derive_mf_config(DEFAULT_BASE_DIR, v))
        rep = Evaluator(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                        check_feasibility=False).evaluate(record_timeline=True)
        for si, p in enumerate(rep.per_stage):
            bd = p.breakdown
            fields = sorted(k for k in vars(bd) if not k.startswith("_"))
            out.append(f"bucket {v.tag} stage={si} peak={p.peak_bytes} "
                       f"ev={getattr(p, 'peak_event', None)} "
                       + " ".join(f"{k}={getattr(bd, k)}" for k in fields))
        for gm in ("chain2", "dataflow"):
            res = simulate_liveness(spec, b.parallel, b.optimizer, b.hardware, b.recompute,
                                    b.swap, record_timeline=True, grad_mode=gm,
                                    graph_source="hand_spec")
            for si, st in enumerate(res.per_stage):
                out.append(f"hand_spec[{gm}] {v.tag} stage={si} peak={st.peak_bytes} "
                           f"ev={getattr(st, 'peak_event', None)}")
    return out


if __name__ == "__main__":
    for line in spec_dump():
        print(line)
    print("#### EVAL ####")
    for line in eval_dump():
        print(line)
