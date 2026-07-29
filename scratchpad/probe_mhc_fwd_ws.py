"""前向 mHC pre-sinkhorn kernel workspace 入账：**结构可达性**探针 + 逐锚点台账。

回答两个问题（只读，不改任何模型行为）：

  ① 每个锚点的**峰值事件**是前向事件（`fwd:*` / `fwd_end`）还是反向事件（`bwd@*`）？
     —— 前向侧的项**结构上**只能抬 `fwd:*`（`Buckets.workspace` 在 FWD 事件置、`rec()` 后清零，
     `cost_eval/mem_timeline.py:583-585`），故峰在 `bwd@*` 的锚点**够不着**（除非该项经
     `forward_max_live` 间接进 `bwd_working_set`/`recomp_scratch`，本探针把这条通道也量出来）。
  ② 每个锚点用的是**融合**还是**非融合** mHC（本项只挂在融合分支）。

用法：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_mhc_fwd_ws.py
在裸 HEAD 的 worktree 与本树各跑一遍逐行 diff = before/after 台账。
"""
from __future__ import annotations

import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
os.environ.setdefault("COST_EVAL_EXTRACTED_ALLOW_PARTIAL", "1")
warnings.filterwarnings("ignore")

MiB = 2 ** 20
OUT = []


def _emit(s=""):
    OUT.append(s)


# ═══════════════════════════════════════════════════════════════════════════════
# 0. 层内直读：`StructureMemory.workspace`（fwd 期层内 max）逐层型
# ═══════════════════════════════════════════════════════════════════════════════

def layer_workspace():
    """逐层型的 fwd `workspace` 与 `forward_max_live`（后者用来量 §① 的那条间接通道）。"""
    import liveness_ab_validate as V
    from cost_eval.parallel_model import ParallelModel
    from cost_eval.shape_eval import ShapeEval
    from cost_eval.structure_mem import estimate_structure_memory

    _emit("### 逐层型 fwd workspace / forward_max_live（八跑 builder 路）###")
    for tag in ("c fused   OFF L8 m4", "d unfused OFF L8 m4"):
        v = V.BY_TAG[tag]
        mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
        bundle, spec = V.build_bundle(mf)
        p = bundle.parallel
        world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
        g = ShapeEval().resolve(spec, ParallelModel(p, spec.dims.n_layers, world))
        seen = set()
        _emit(f"  [{tag}]  use_fused_mhc={getattr(spec.dims, 'use_fused_mhc', None)}")
        for layers in g.stages.values():
            for lay in layers:
                if lay.layer_type in seen:
                    continue
                seen.add(lay.layer_type)
                sm = estimate_structure_memory(lay.ops)
                carriers = {op.name: op.workspace_bytes / MiB
                            for op in lay.ops if op.workspace_bytes}
                _emit(f"    {lay.layer_type:<22} ws={sm.workspace / MiB:9.3f} "
                      f"fml={sm.forward_max_live / MiB:9.1f} bww={sm.bwd_workspace / MiB:8.3f} "
                      f"carriers={ {k: round(v, 3) for k, v in carriers.items()} }")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. explorer 路锚点：峰值事件 + 最高前向事件 + 差额
# ═══════════════════════════════════════════════════════════════════════════════

_DSV4 = {
    "preset": "dsv4_flash", "attn": "dsv4_hybrid",
    "layers": "8", "seq": "4096", "batch": "1", "mtp": "0",
    "experts": "8", "topk": "2", "dense_k": "1", "heads": "64", "kv_groups": "1",
    "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
    "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
    "v_head": "512", "vocab": "129280",
    "hc": "4", "mhc_fused": "1", "dsa_fused": "1", "ce_fused": "1",
    "compress_ratios": "0,4,128,4,128,4,128,4",
    "dp": "2", "tp": "1", "ep": "2", "pp": "4", "cp": "1", "method": "colossal",
    "optimizer": "muon", "opt_dtype": "fp32", "grad_bytes": "4", "maxdev_gib": "58",
    "select": "attn", "sel_layers": "", "sel_ops": "", "sel_cfg": "", "vpp": "1",
    "mbs": "", "dp_replicate": "1", "reshard": "default", "cpu_offload": "0",
    "prefetch": "1", "sp": "",
}


def _std_q(kv):
    """185 std（无 mHC：`hc="1"`）—— 用来证明本项对它恒 0。"""
    return {
        "preset": "custom", "attn": "gqa", "layers": "8", "seq": "4096", "batch": "1",
        "mtp": "0", "experts": "0", "topk": "1", "dense_k": "8",
        "heads": "32", "kv_groups": str(kv), "hidden": "2048",
        "ffn": "8192", "moe_ffn": "8192", "vocab": "129280", "hc": "1",
        "dp": "2", "tp": "1", "ep": "1", "pp": "2", "cp": "1", "method": "colossal",
        "optimizer": "adamw", "opt_dtype": "bf16", "grad_bytes": "4",
        "maxdev_gib": "58", "recompute": "full", "mbs": "4", "pp_split": "4,4",
        "emb_bytes": "4",
        "q_lora": "", "kv_lora": "", "qk_nope": "", "qk_rope": "", "v_head": "",
        "select": "attn", "sel_layers": "", "sel_ops": "", "sel_cfg": "", "vpp": "1",
        "dp_replicate": "1", "reshard": "default", "cpu_offload": "0",
        "prefetch": "1", "sp": "", "mhc_fused": "0", "dsa_fused": "0", "ce_fused": "0",
    }


def _eval_with_timeline(p):
    """复刻 `serve_explorer.eval_config` 的评估路径，但把 `rep` 整个交出来（要 timeline）。"""
    import serve_explorer as S
    from cost_eval.report import Evaluator
    errs, cfg, pa = S.parse_and_validate(p)
    assert not errs, errs
    spec = S.build_llm_spec(cfg)
    rc = S._rc_from_pa(pa)
    pc, opt, hw, swap = S._build_eval_specs(p, pa)
    return Evaluator(spec, pc, opt, hw, rc, swap).evaluate(record_timeline=True), cfg


def _fwd_best(sp):
    """该 stage 时间线上最高的**前向**事件（`fwd:*` / `fwd_end`）。"""
    fw = [s for s in sp.timeline if s.event.startswith("fwd")]
    if not fw:
        return None
    return max(fw, key=lambda s: s.total_bytes)


def _fwd_layer_best(sp):
    """最高的**逐层前向**事件 `fwd:<lid>` —— **只有它带 `Buckets.workspace`**
    （`mem_timeline.py:583-585`：置 `sm.workspace` → `rec(f"fwd:{lid}")` → 立即清零；
    `fwd_end` 那一枚在清零之后，恒不带 workspace）。故「本项还差多少才能顶翻峰值」
    = `peak − max(fwd:<lid>)`，而**不是** `peak − max(全部前向事件)`。"""
    fw = [s for s in sp.timeline if s.event.startswith("fwd:")]
    if not fw:
        return None
    return max(fw, key=lambda s: s.total_bytes)


def _row(label, sp):
    fb = _fwd_best(sp)
    fl = _fwd_layer_best(sp)
    peak = sp.peak_bytes / MiB
    if fb is None:
        _emit(f"  {label:<26} peak={peak:9.1f} @{sp.peak_event:<14} (无前向事件)")
        return
    kind = "FWD" if sp.peak_event.startswith("fwd") else "BWD"
    lid_txt = (f" | best_fwd:lid={fl.total_bytes / MiB:9.1f} @{fl.event:<10} "
               f"ws头寸={peak - fl.total_bytes / MiB:9.1f}") if fl else " | (无 fwd:lid 事件)"
    _emit(f"  {label:<26} peak={peak:9.1f} @{sp.peak_event:<14} [{kind}] | "
          f"best_fwd={fb.total_bytes / MiB:9.1f} @{fb.event:<12} 差={peak - fb.total_bytes / MiB:9.1f}"
          f"{lid_txt}")


def explorer_anchors():
    _emit()
    _emit("### explorer 路锚点：峰值事件属相（FWD 可达 / BWD 不可达）###")
    cases = [
        ("pp4 ON", dict(recompute="full")),
        ("pp4 OFF", dict(recompute="None")),
        ("pp4 ON MTP", dict(recompute="full", mtp="1", pp_split="2,2,2,3")),
        ("pp8 ON", dict(dp="1", ep="1", pp="8", mbs="8", recompute="full")),
        ("185 P3-P m8", dict(recompute="full", mbs="8")),
    ]
    for label, over in cases:
        q = dict(_DSV4)
        q.update(over)
        rep, cfg = _eval_with_timeline(q)
        _emit(f"  --- {label}  use_fused_mhc={cfg.use_fused_mhc} ---")
        for sp in rep.per_stage:
            _row(f"{label} s{sp.stage}", sp)
    sys.path.insert(0, os.path.join(_REPO, "tests"))
    import test_probe185_recon as P
    for kv in (32, 8):
        rep, cfg = _eval_with_timeline(P._std_on_q(kv))
        _emit(f"  --- 185 std ON kv={kv}  use_fused_mhc={cfg.use_fused_mhc} "
              f"(hc=1 → 无 mHC 模块) ---")
        for sp in rep.per_stage:
            _row(f"185 std ON kv{kv} s{sp.stage}", sp)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 185 U/F 相位（test_probe185_recon 的口径）与 116 std
# ═══════════════════════════════════════════════════════════════════════════════

def probe185_and_std116():
    _emit()
    _emit("### 185 U/F 相位 + 116 std（各自测试文件自己的 query）###")
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "tests"))
    import test_probe185_recon as P
    for name, q in (("185 F0 4L", P._dsv4_q(4, fused=True)),
                    ("185 F1 8L", P._dsv4_q(8, fused=True)),
                    ("185 U1 4L", P._dsv4_q(4, fused=False)),
                    ("185 U2 8L", P._dsv4_q(8, fused=False))):
        rep, cfg = _eval_with_timeline(q)
        _emit(f"  --- {name}  use_fused_mhc={cfg.use_fused_mhc} ---")
        for sp in rep.per_stage:
            _row(f"{name} s{sp.stage}", sp)
    import test_std_attn_anchor as T
    _emit("  --- 116 std（test_std_attn_anchor 路径，无 mHC）---")
    for attn, kv in (("mha", 32), ("gqa", 8)):
        for pp in (2, 1):
            pk = T._peaks(kv, pp)
            for st, v in sorted(pk.items()):
                _emit(f"  116 std {attn} pp{pp} s{st}: {v:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 记分卡
# ═══════════════════════════════════════════════════════════════════════════════

def scorecard():
    from scorecard_anchors import anchors
    _emit()
    _emit("### scorecard_anchors ###")
    for a in anchors():
        sim = a.sim_fn()
        if sim is None:
            _emit(f"  {a.label:<34}       ERR")
            continue
        _emit(f"  {a.label:<34} {sim:10.1f}  ratio={sim / a.real:.4f}")


def scorecard_events():
    """记分卡两个 DSv4 锚 + DSv3 族的峰值事件属相。"""
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
    _emit()
    _emit("### 记分卡锚点峰值事件属相（DSv3 族无 mHC；DSv4 族 use_fused_mhc=False）###")
    cases = [
        ("DSv3 8L none", 8, NONE, dict(B=1, dp=2, cp=1, pp=1, mbs=1), 0),
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
    try:
        # `validate_dsv4align.evaluate` 不带 record_timeline → 就地复刻它（逐参照抄，
        # `validate_dsv4align.py:100-116`），只多一个 record_timeline=True。
        from validate_dsv3 import RESIDUAL_MiB
        from validate_dsv4align import dsv4_align_config
        from cost_eval.build_llm import build_llm_spec
        MiB_ = 2 ** 20
        for lbl, mhc, mtp, ufm in (("DSv4-fused base", 0, 0, False),
                                   ("DSv4 mHC(x4)+MTP", 4, 1, False),
                                   ("[what-if] mHC+MTP fused", 4, 1, True)):
            cfg = dsv4_align_config(4, mhc, mtp, 2048, use_fused_mhc=ufm)
            spec = build_llm_spec(cfg)
            pc = ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1,
                                sequence_parallel=True, num_microbatches=1)
            hw = HardwareSpec(max_device_memory=54 * GiB,
                              framework_reserve=RESIDUAL_MiB * MiB_)
            rep = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                            hw, RecomputeSpec(mode="None", full_layers=set()),
                            SwapSpec()).evaluate(record_timeline=True)
            _row(f"{lbl} s0", rep.per_stage[0])
    except Exception as e:
        _emit(f"  [dsv4align] 跳过：{type(e).__name__} {e}")


def main():
    layer_workspace()
    explorer_anchors()
    probe185_and_std116()
    scorecard()
    scorecard_events()
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
