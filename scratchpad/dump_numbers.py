# -*- coding: utf-8 -*-
"""字节中性证明:手写 spec 全字段 dump + Evaluator 数字 dump(供 before/after diff)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def spec_dump():
    from validate_dsv3 import build_dsv3_spec
    out = []
    for L in (4, 6, 8):
        spec, dims, _pm = build_dsv3_spec(L)
        out.append(f"=== dsv3 L={L} dims={sorted(dims.as_dict().items())}")
        for ltype in sorted(spec.layer_specs):
            for op in spec.get_layer(ltype).ops:
                out.append(f"  {ltype}/{op.name} type={op.type.value} "
                           f"ws={op.workspace} bws={op.bwd_scratch} attrs={sorted(op.attrs.items())}")
                for slot, seq in (("in", op.inputs), ("out", [op.output]),
                                  ("par", op.params), ("sav", op.saves)):
                    for t in seq:
                        out.append(
                            f"    {slot} {t.name} shape={t.shape} shard={sorted(t.shard.items())} "
                            f"w={t.is_weight} partial={t.partial} dt={t.dtype_bytes} "
                            f"cp={t.cp_shard} cpkv={t.cp_kv} pin={t.pin_under_recompute} "
                            f"det={t.detached}")
                for nm, r in (("wsref", op.workspace_ref), ("bwsref", op.bwd_scratch_ref)):
                    if r is not None:
                        out.append(f"    {nm} {r.name} shape={r.shape} shard={sorted(r.shard.items())}")
    return out


def eval_dump():
    from validate_dsv3 import build_dsv3_spec
    from cost_eval.report import Evaluator
    from cost_eval.specs import (ParallelConfig, OptimizerSpec, HardwareSpec,
                                 RecomputeSpec, SwapSpec)
    GiB, MiB = 1 << 30, 1 << 20
    out = []
    for L in (4, 8):
        for rc in (None, "full"):
            spec, d, full_layers = build_dsv3_spec(L)
            pc = ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1,
                                sequence_parallel=True, num_microbatches=1)
            ev = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                           HardwareSpec(max_device_memory=59 * GiB, framework_reserve=0),
                           RecomputeSpec(mode=rc, full_layers=full_layers if rc else None),
                           SwapSpec())
            rep = ev.evaluate()
            for si, p in enumerate(rep.per_stage):
                b = p.breakdown
                fields = sorted(k for k in vars(b) if not k.startswith("_"))
                out.append(f"dsv3 L={L} rc={rc} stage={si} peak={p.peak_bytes} "
                           f"ev={getattr(p, 'peak_event', None)} "
                           + " ".join(f"{k}={getattr(b, k)}" for k in fields))
    return out


if __name__ == "__main__":
    for line in spec_dump():
        print(line)
    print("#### EVAL ####")
    for line in eval_dump():
        print(line)
