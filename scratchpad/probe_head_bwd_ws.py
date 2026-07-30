"""`lm_head` 反向 kernel workspace 入账：结构可达性 + 逐 config 落位。

三问：
  ① 实测律逐字节穿过 `build_llm_spec → ShapeEval → StructureMemory` 了吗（含 mHC / MTP 路）？
  ② 它挂在**哪个 op** 上（必须只有 `lm_head`）？
  ③ 各 config 下 head 层的 `bwd_workspace` 与逐 stage 峰值事件是什么？

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_head_bwd_ws.py
"""
from __future__ import annotations

import dataclasses
import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "tools"), os.path.join(_REPO, "tests")):
    sys.path.insert(0, _p)
os.environ.setdefault("COST_EVAL_EXTRACTED_ALLOW_PARTIAL", "1")
warnings.filterwarnings("ignore")

MiB = 2 ** 20


def law(N, H, V):
    """167 实测律：(2*vocab + 4*H)*(B*S) + 20 MiB + 1024 B。"""
    return (2 * V + 4 * H) * N + 20971520 + 1024


def main():
    from cost_eval.build_llm import build_llm_spec
    from cost_eval.presets import deepseek_v3
    from cost_eval.parallel_model import ParallelModel
    from cost_eval.shape_eval import ShapeEval
    from cost_eval.specs import ParallelConfig
    from cost_eval.structure_mem import estimate_structure_memory

    print("=" * 104)
    print("(1) law through the chain: DSv3 preset (H=1792 / vocab=129280 / B=1 / tp=cp=1), 3 S points")
    print("=" * 104)
    for S in (1024, 2048, 4096):
        cfg = dataclasses.replace(deepseek_v3(4), seq_length=S)
        spec = build_llm_spec(cfg)
        pm = ParallelModel(ParallelConfig(), spec.dims.n_layers, 1)
        g = ShapeEval().resolve(spec, pm)
        for layers in g.stages.values():
            for lay in layers:
                names = [o.name for o in lay.ops]
                if "lm_head" not in names:
                    continue
                sm = estimate_structure_memory(lay.ops)
                per = [(o.name, o.bwd_workspace_bytes) for o in lay.ops
                       if getattr(o, "bwd_workspace_bytes", 0)]
                exp = law(S * spec.dims.B, cfg.hidden_size, cfg.vocab_size)
                print("  S=%-5d %-12s sm.bwd_workspace=%12d  law=%12d  delta=%+d  carriers=%s"
                      % (S, lay.layer_type, sm.bwd_workspace, exp,
                         sm.bwd_workspace - exp, per))

    print()
    print("=" * 104)
    print("(2) attribution: bwd_workspace_bytes of every op of every layer type")
    print("=" * 104)
    cfg = deepseek_v3(4)
    spec = build_llm_spec(cfg)
    pm = ParallelModel(ParallelConfig(), spec.dims.n_layers, 1)
    g = ShapeEval().resolve(spec, pm)
    seen = set()
    for layers in g.stages.values():
        for lay in layers:
            if lay.layer_type in seen:
                continue
            seen.add(lay.layer_type)
            print("  %-14s %s" % (lay.layer_type,
                                  [(o.name, o.bwd_workspace_bytes) for o in lay.ops]))

    print()
    print("=" * 104)
    print("(3) DSv4-hybrid (fused mHC / MTP): per-layer bwd_workspace + per-stage peak event")
    print("=" * 104)
    import serve_explorer as S_
    import test_pp4_recompute_anchor as A
    from cost_eval.report import Evaluator

    for tag, over in (("pp4 OFF", {"recompute": "None"}),
                      ("pp4 ON", {"recompute": "full"}),
                      ("pp4 MTP", {"recompute": "full", "mtp": "1", "pp_split": "2,2,2,3"})):
        q = dict(A._BASE)
        q.update(over)
        errs, cfg2, pa = S_.parse_and_validate(q)
        assert not errs, errs
        spec2 = S_.build_llm_spec(cfg2)
        pc2, opt, hw, swap = S_._build_eval_specs(q, pa)
        rep = Evaluator(spec2, pc2, opt, hw, S_._rc_from_pa(pa), swap).evaluate(
            record_timeline=True)
        pm2 = ParallelModel(pc2, spec2.dims.n_layers,
                            pc2.dp_replicate * pc2.dp_shard * pc2.cp * pc2.tp * pc2.pp)
        g2 = ShapeEval().resolve(spec2, pm2)
        print("  %-9s H=%d vocab=%d S=%d B=%d  law(head)=%d"
              % (tag, cfg2.hidden_size, cfg2.vocab_size, cfg2.seq_length, spec2.dims.B,
                 law(cfg2.seq_length * spec2.dims.B, cfg2.hidden_size, cfg2.vocab_size)))
        seen2 = set()
        for st, layers in sorted(g2.stages.items()):
            for lay in layers:
                sm = estimate_structure_memory(lay.ops)
                if not sm.bwd_workspace or lay.layer_type in seen2:
                    continue
                seen2.add(lay.layer_type)
                car = [o.name for o in lay.ops if getattr(o, "bwd_workspace_bytes", 0)]
                print("      %-24s bwd_workspace=%12d = %10.4f MiB  carriers=%s"
                      % (lay.layer_type, sm.bwd_workspace, sm.bwd_workspace / MiB, car))
        for sp in rep.per_stage:
            print("      stage %d peak=%10.1f MiB @ %s"
                  % (sp.stage, sp.peak_bytes / MiB, sp.peak_event))


if __name__ == "__main__":
    main()
