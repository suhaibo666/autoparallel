"""**验收门**：167 A/B 八跑 × 任意 graph source × 桶模型，同表对照（2026-07-25）。

这个脚本是本项目的 acceptance gate：任何"换图来源"的改动（手写 census → 源抽取图）都必须先过它。
它回答四个问题，全部在同一张表里：

  1. **真机 vs 模型**：per-stage 峰值、逐格 `sim/real`、28 个可评分格的聚合（run d 真机 OOM →
     不可评分，如实标 `OOM` 并**不**入统计）；
  2. **来源之间**：`real / bucket / liveness(hand_spec) / liveness(extracted，可用时)` 并排 —— 两条
     spec 来源喂**同一个** liveness 仿真器，故差异只可能来自图本身（`cost_eval/liveness/sources.py`）；
  3. **两条真机不变量**（×1 于层数、×1 于微批数）：既校验 `REAL` 表自身没被改坏，也作为**任何图
     来源都必须满足的结构断言**（正确的模型是**导出**它们，而不是被公式告知）；
  4. **峰值时刻逐张量 live-set**：`--dump-top N` 打峰值瞬间在世张量清单（哪一个字节是谁）。

真机地面真值
------------
`REAL` 表来自 `192.168.9.167:/home/suhaibo/workspace/log_ab_fusion_2026-07-25/<run>/worker_<N>.log`
的 `[MEMPROBE] rank=N peak_alloc_MiB=...` 行（rank→stage = `rank // (world/pp)` = `rank//2`），
2026-07-25 实测。**这是测量值：不得编辑、不得编造、不得"改好看"。**
`REAL_SHA256` 是它的规范化指纹，`--gate` 会核对（`tests/test_acceptance_gate.py` 亦然）。

八跑的派生
----------
八跑的 launcher yaml 之间**只**差 5 个字段（已在 167 上 grep 核对）：
``training.global_batch_size`` / ``recompute.mode`` + ``full_recompute_layer`` /
``model.num_hidden_layers`` / ``model.compress_ratios`` / ``model.apply_dsa_kernel_fusion``。
故本脚本从**两份 base yaml**（fused / unfused，pp4 全重算 L8 m4）**程序化派生**其余六份，并
`check_derivation()` **回头断言派生结果与标签一致**（微批数、层数、fused 位、重算域、
compress_ratios 长度、每 stage 隐藏层数）—— 手抄八份 yaml 无法自证一致，派生 + 反查可以。

评估路径 = **bundle-direct**（UI 字段路径曾有损，`5be5ccd` 修了，但 bundle-direct 仍是基准）：

    b = from_mindformers_dict(mf2); spec = build_llm_spec(b.llm)
    rep = Evaluator(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                    check_feasibility=False).evaluate(record_timeline=True)

用法
----
    python tools/liveness_ab_validate.py                       # 默认 base-dir = 仓内 A/B 配置
    python tools/liveness_ab_validate.py --grad-mode chain2 --deltas --gate
    python tools/liveness_ab_validate.py --sources hand_spec,extracted --dump-top 20
    python tools/liveness_ab_validate.py --config my.yaml      # 单配置 × 各来源（非八跑门）
    python tools/liveness_ab_validate.py --json out.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass, field

MiB = 2 ** 20
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

#: 仓内固化的两份 base yaml（= 167 上真跑的 launcher 配置，只差 apply_dsa_kernel_fusion）。
DEFAULT_BASE_DIR = os.path.join(_REPO, "analysis", "realmachine", "ab_fusion_2026-07-25")

REAL_PROVENANCE = (
    "192.168.9.167:/home/suhaibo/workspace/log_ab_fusion_2026-07-25/<run>/worker_<N>.log"
    " 的 [MEMPROBE] rank=N peak_alloc_MiB=...；rank→stage = rank//2；2026-07-25 实测")

# ═══════════════════════════════════════════════════════════════════════════
# 真机地面真值 —— **测量值，不得编辑/编造/美化**（None = 该跑 OOM，无可比值）
# ═══════════════════════════════════════════════════════════════════════════
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
#: run d 的 s0 在 OOM 前观测到的区间（**不是**峰值，仅供参考，不参与任何评分）。
REAL_D_S0_OBSERVED_RANGE = (57712.7, 57718.4)
#: `REAL` 的规范化指纹 —— 防"顺手改数据让模型好看"。改了真机数据必须显式改这里并说明来源。
REAL_SHA256 = "41e279e591ae4ae962642813f87d86a34cce545dfb11374b6818bbdc4fc621f9"


def real_fingerprint() -> str:
    """`REAL` 表的规范化 sha256（键排序、浮点按 repr）。"""
    payload = json.dumps({k: {str(s): v for s, v in d.items()} for k, d in REAL.items()},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# 八跑矩阵（从两份 base yaml 派生）
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Variant:
    """一跑：标签 + 5 个被改动字段的目标值。`m` 是**期望**微批数，由 check_derivation 反查。"""
    tag: str
    fused: bool
    global_batch_size: int
    m: int
    recompute_on: bool
    n_layers: int

    @property
    def letter(self) -> str:
        return self.tag.split()[0]


VARIANTS = (
    Variant("a fused   ON  L8 m4", True, 8, 4, True, 8),
    Variant("b unfused ON  L8 m4", False, 8, 4, True, 8),
    Variant("c fused   OFF L8 m4", True, 8, 4, False, 8),
    Variant("d unfused OFF L8 m4", False, 8, 4, False, 8),
    Variant("e fused   ON  L8 m8", True, 16, 8, True, 8),
    Variant("f unfused ON  L8 m8", False, 16, 8, True, 8),
    Variant("g fused   ON  L4 m4", True, 8, 4, True, 4),
    Variant("h unfused ON  L4 m4", False, 8, 4, True, 4),
)
BY_TAG = {v.tag: v for v in VARIANTS}
#: 真机 launcher 的 compress_ratios（长度必须 == num_hidden_layers）。
RATIOS = {8: [0, 4, 128, 4, 128, 4, 128, 4], 4: [0, 4, 128, 4]}
BASE_YAML = {True: "dsv4h_fused_pp4_recomp.yaml", False: "dsv4h_unfused_pp4_recomp.yaml"}
#: (标签, fused 跑, unfused 跑) —— delta 表与 ×1 不变量都建在这三对上。
PAIRS = (("L8 m4", "a fused   ON  L8 m4", "b unfused ON  L8 m4"),
         ("L8 m8", "e fused   ON  L8 m8", "f unfused ON  L8 m8"),
         ("L4 m4", "g fused   ON  L4 m4", "h unfused ON  L4 m4"))

#: ×1 不变量的判据 stage 与容差（**来自真机**：见 §真机不变量）。
INVARIANT_STAGE = 3
TOL_LAYERS_MIB = 0.1       # ×1 于层数：L8(2 层/stage) 与 L4(1 层/stage) 的 delta 逐 0.1 MiB 相同
TOL_MICROBATCH_MIB = 0.4   # ×1 于微批数：a↔e / b↔f 把 delta 移动 ≤0.4 MiB


# ═══════════════════════════════════════════════════════════════════════════
# 派生 + 反查
# ═══════════════════════════════════════════════════════════════════════════

def derive_mf_config(base_dir: str, variant: Variant) -> dict:
    """两份 base yaml → 该跑的 mindformers 配置 dict（只改那 5 个字段）。"""
    import yaml
    path = os.path.join(base_dir, BASE_YAML[variant.fused])
    with open(path, encoding="utf-8") as fh:
        mf = copy.deepcopy(yaml.safe_load(fh))
    nl = variant.n_layers
    mf["training"]["global_batch_size"] = variant.global_batch_size
    mf["model"]["num_hidden_layers"] = nl
    mf["model"]["compress_ratios"] = list(RATIOS[nl])
    mf["model"]["apply_dsa_kernel_fusion"] = variant.fused
    mf["recompute"] = ({"mode": "full", "full_recompute_layer": ["0-%d" % (nl - 1)]}
                       if variant.recompute_on else {"mode": "None"})
    m = mf["model"]
    if not m.get("qk_nope_head_dim"):      # launcher 省略；mindformers 由 head_dim − rope 导出
        m["qk_nope_head_dim"] = int(m["head_dim"]) - int(m["qk_rope_head_dim"])
    return mf


def build_bundle(mf: dict):
    """mindformers dict → **bundle-direct** `(bundle, ModelSpec)`（UI 字段路径不参与）。"""
    import serve_explorer as S
    from cost_eval.build_llm import build_llm_spec
    from cost_eval.configs.from_mindformers import from_mindformers_dict
    mf2, _ = S._mf_adapt(copy.deepcopy(mf))
    S._materialize_nested_offset(mf2, [])
    b = from_mindformers_dict(mf2)
    return b, build_llm_spec(b.llm)


def check_derivation(mf: dict, bundle, spec, variant: Variant) -> list:
    """**反查派生结果与标签一致**（派生代替手抄，反查代替信任）。返回违规字符串列表。"""
    bad: list = []
    nl, tag = variant.n_layers, variant.tag
    m = mf["model"]
    if m["apply_dsa_kernel_fusion"] is not variant.fused:
        bad.append(f"{tag}: yaml apply_dsa_kernel_fusion={m['apply_dsa_kernel_fusion']}"
                   f" != 标签 fused={variant.fused}")
    if bool(getattr(bundle.llm, "dsa_fused", None)) is not variant.fused:
        bad.append(f"{tag}: bundle.llm.dsa_fused={bundle.llm.dsa_fused} != 标签 {variant.fused}")
    if m["num_hidden_layers"] != nl or getattr(bundle.llm, "num_layers", None) != nl:
        bad.append(f"{tag}: 层数 yaml={m['num_hidden_layers']} bundle="
                   f"{getattr(bundle.llm, 'num_layers', None)} != 标签 L{nl}")
    if list(m["compress_ratios"]) != list(RATIOS[nl]):
        bad.append(f"{tag}: compress_ratios={m['compress_ratios']} != {RATIOS[nl]}")
    if len(m["compress_ratios"]) != nl:
        bad.append(f"{tag}: len(compress_ratios)={len(m['compress_ratios'])} != L{nl}")
    if list(getattr(bundle.llm, "csa_compress_ratios", ()) or ()) != list(RATIOS[nl]):
        bad.append(f"{tag}: bundle.llm.csa_compress_ratios="
                   f"{getattr(bundle.llm, 'csa_compress_ratios', None)} != {RATIOS[nl]}")
    if bundle.parallel.num_microbatches != variant.m:
        bad.append(f"{tag}: 派生微批数 {bundle.parallel.num_microbatches} != 标签 m={variant.m}"
                   f"（gbs={variant.global_batch_size}）")
    rc = bundle.recompute
    if variant.recompute_on:
        # layer_pattern 首项是 embedding 伪层 → 隐藏层 id 是 1..nl。
        missing = [lid for lid in range(1, nl + 1) if not rc.is_full(lid)]
        if rc.mode != "full" or missing:
            bad.append(f"{tag}: 期望全部 {nl} 层全重算，实得 mode={rc.mode} 未覆盖 {missing}")
    else:
        on = [lid for lid in range(1, nl + 1) if rc.is_full(lid) or rc.is_select(lid)]
        if on:
            bad.append(f"{tag}: 期望无重算，实得 {on} 仍在重算")
    # pp/dp/ep 三度数与 seq 长度：八跑共用，不随标签变。
    p = bundle.parallel
    for name, got, want in (("pp", p.pp, 4), ("dp_shard", p.dp_shard, 2), ("ep", p.ep, 2),
                            ("tp", p.tp, 1), ("cp", p.cp, 1)):
        if got != want:
            bad.append(f"{tag}: {name}={got} != {want}（八跑共用 pp4/dp2/ep2）")
    if m["seq_length"] != 4096:
        bad.append(f"{tag}: seq_length={m['seq_length']} != 4096")
    # 每 stage 隐藏层数（标签里的 "2 层/stage" vs "1 层/stage" 语义）。
    hidden_per_stage = nl // 4
    got = _hidden_layers_per_stage(spec, bundle)
    if sorted(set(got.values())) != [hidden_per_stage]:
        bad.append(f"{tag}: 每 stage 隐藏层数 {got} != 恒 {hidden_per_stage}")
    return bad


def _hidden_layers_per_stage(spec, bundle) -> dict:
    from cost_eval.liveness import resolve_graph
    from cost_eval.parallel_model import ParallelModel
    p = bundle.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    pm = ParallelModel(p, spec.dims.n_layers, world)
    g = resolve_graph(spec, pm, "hand_spec")
    return {st: sum(1 for l in layers if l.layer_type.startswith("dsv4hyb"))
            for st, layers in sorted(g.stages.items())}


# ═══════════════════════════════════════════════════════════════════════════
# 一跑的评估结果（桶模型 + 每个 graph source 一份 liveness）
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    variant: Variant
    bucket_mib: tuple                     # per-stage 桶模型峰值（MiB）
    liveness_mib: dict = field(default_factory=dict)   # source -> per-stage MiB
    liveness_res: dict = field(default_factory=dict)   # source -> LivenessResult
    derivation_violations: list = field(default_factory=list)
    n_stages: int = 0

    def real(self, stage: int):
        return REAL[self.variant.tag][stage]

    def series(self, key: str) -> tuple:
        return self.bucket_mib if key == "bucket" else self.liveness_mib[key]


def evaluate_run(base_dir: str, variant: Variant, sources, grad_mode: str) -> RunResult:
    """跑一份配置：桶模型 + 每个 graph source 的 liveness（bundle-direct 路径）。"""
    from cost_eval.liveness import simulate_liveness
    from cost_eval.report import Evaluator
    mf = derive_mf_config(base_dir, variant)
    b, spec = build_bundle(mf)
    viol = check_derivation(mf, b, spec, variant)
    rep = Evaluator(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                    check_feasibility=False).evaluate(record_timeline=True)
    bucket = tuple(p.peak_bytes / MiB for p in rep.per_stage)
    lv_mib, lv_res = {}, {}
    for src in sources:
        res = simulate_liveness(spec, b.parallel, b.optimizer, b.hardware, b.recompute,
                                b.swap, record_timeline=True, grad_mode=grad_mode,
                                graph_source=src)
        lv_res[src] = res
        lv_mib[src] = tuple(st.peak_bytes / MiB for st in res.per_stage)
    return RunResult(variant=variant, bucket_mib=bucket, liveness_mib=lv_mib,
                     liveness_res=lv_res, derivation_violations=viol,
                     n_stages=len(bucket))


def run_matrix(base_dir: str, sources, grad_mode: str) -> dict:
    return {v.tag: evaluate_run(base_dir, v, sources, grad_mode) for v in VARIANTS}


# ═══════════════════════════════════════════════════════════════════════════
# 不变量
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'}  {self.name}: {self.detail}"


def _delta(series_u, series_f, stage: int):
    if series_u[stage] is None or series_f[stage] is None:
        return None
    return series_u[stage] - series_f[stage]


def real_invariants() -> list:
    """**真机自身**的两条不变量 + `REAL` 表指纹守卫（测量值没被改坏）。"""
    out: list = []
    fp = real_fingerprint()
    out.append(Check("REAL 指纹", fp == REAL_SHA256,
                     f"sha256={fp[:16]}…{' == 预期' if fp == REAL_SHA256 else ' != ' + REAL_SHA256[:16]}"
                     "（真机数据是测量值，改动必须显式改指纹并说明来源）"))
    d = {}
    for lbl, ft, ut in PAIRS:
        d[lbl] = _delta(REAL[ut], REAL[ft], INVARIANT_STAGE)
    ok = abs(d["L8 m4"] - d["L4 m4"]) <= TOL_LAYERS_MIB
    out.append(Check(
        "I1 真机 ×1 于层数",
        ok, f"stage{INVARIANT_STAGE} unfused−fused: L8(2 层/stage)={d['L8 m4']:.1f} "
            f"L4(1 层/stage)={d['L4 m4']:.1f} |diff|={abs(d['L8 m4'] - d['L4 m4']):.1f}"
            f" <= {TOL_LAYERS_MIB}"))
    ok = abs(d["L8 m4"] - d["L8 m8"]) <= TOL_MICROBATCH_MIB
    out.append(Check(
        "I2 真机 ×1 于微批数",
        ok, f"stage{INVARIANT_STAGE} unfused−fused: m4={d['L8 m4']:.1f} m8={d['L8 m8']:.1f}"
            f" |diff|={abs(d['L8 m4'] - d['L8 m8']):.1f} <= {TOL_MICROBATCH_MIB}"))
    return out


def model_invariants(results: dict, key: str) -> list:
    """**任何图来源都必须满足**的同两条 ×1 结构（正确的模型是导出它们，不是被公式告知）。

    断言的是**结构**（delta 在层数/微批数上不动），**不**断言幅值 —— 幅值另行如实报告
    （见 `magnitude_report`），因为今天两条来源都欠读这个 delta，不该被断言掩盖。"""
    out: list = []
    d = {lbl: _delta(results[ut].series(key), results[ft].series(key), INVARIANT_STAGE)
         for lbl, ft, ut in PAIRS}
    ok = abs(d["L8 m4"] - d["L4 m4"]) <= TOL_LAYERS_MIB
    out.append(Check(f"I1 {key} ×1 于层数", ok,
                     f"stage{INVARIANT_STAGE} delta L8={d['L8 m4']:.1f} L4={d['L4 m4']:.1f}"
                     f" |diff|={abs(d['L8 m4'] - d['L4 m4']):.1f} <= {TOL_LAYERS_MIB}"))
    ok = abs(d["L8 m4"] - d["L8 m8"]) <= TOL_MICROBATCH_MIB
    out.append(Check(f"I2 {key} ×1 于微批数", ok,
                     f"stage{INVARIANT_STAGE} delta m4={d['L8 m4']:.1f} m8={d['L8 m8']:.1f}"
                     f" |diff|={abs(d['L8 m4'] - d['L8 m8']):.1f} <= {TOL_MICROBATCH_MIB}"))
    return out


def magnitude_report(results: dict, keys) -> list:
    """delta 的**幅值**对账（只报告，不断言）：`model_delta / real_delta` per 配置对。"""
    rows: list = []
    for lbl, ft, ut in PAIRS:
        rd = _delta(REAL[ut], REAL[ft], INVARIANT_STAGE)
        for key in keys:
            md = _delta(results[ut].series(key), results[ft].series(key), INVARIANT_STAGE)
            rows.append((lbl, key, rd, md, (md / rd) if rd else None))
    return rows


def aggregate(results: dict, key: str) -> dict:
    """可评分格（真机非 OOM）的 `sim/real` 聚合。run d 全 OOM → 不入统计。"""
    vals = []
    for tag, r in results.items():
        for i in range(r.n_stages):
            real = r.real(i)
            if real:
                vals.append(r.series(key)[i] / real)
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": sum(vals) / len(vals),
            "min": min(vals), "max": max(vals)}


def unscorable_cells(results: dict) -> list:
    return [(tag, i) for tag, r in results.items()
            for i in range(r.n_stages) if r.real(i) is None]


# ═══════════════════════════════════════════════════════════════════════════
# 打表
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(x, w=10, p=1):
    return ("%*.*f" % (w, p, x)) if x is not None else ("%*s" % (w, "OOM"))


def print_matrix_table(results: dict, keys, grad_mode: str) -> None:
    """主表：run × stage × {real, bucket, liveness(每来源)} + 逐格 sim/real + 峰值事件。"""
    hdr = "%-20s %-3s %10s" % ("run", "st", "real")
    for k in keys:
        hdr += " %10s" % (k[:10] if k == "bucket" else ("lv:" + k)[:10])
    for k in keys:
        hdr += " %9s" % (k[:7] + "/re")
    hdr += "  peak@%s" % keys[-1]
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        r = results[v.tag]
        for i in range(r.n_stages):
            real = r.real(i)
            line = "%-20s %-3d %10s" % (v.tag if i == 0 else "", i,
                                        ("%.1f" % real) if real else "OOM")
            for k in keys:
                line += " %10.1f" % r.series(k)[i]
            for k in keys:
                line += " %9s" % (("%.3f" % (r.series(k)[i] / real)) if real else "-")
            last = keys[-1]
            if last == "bucket":
                line += "  -"
            else:
                st = r.liveness_res[last].per_stage[i]
                line += "  %s/%s" % (st.peak_event, st.peak_substep)
            print(line)


def print_aggregate(results: dict, keys) -> None:
    uns = unscorable_cells(results)
    print("\n-- 聚合 sim/real（可评分格；不可评分 %d 格：%s）--"
          % (len(uns), ", ".join("%s@s%d" % (t.split()[0], i) for t, i in uns) or "无"))
    for k in keys:
        a = aggregate(results, k)
        print("  %-12s n=%d  mean=%.3f  min=%.3f  max=%.3f"
              % (k, a["n"], a["mean"], a["min"], a["max"]))
    if uns:
        lo, hi = REAL_D_S0_OBSERVED_RANGE
        print("  ⚠ run d 真机 OOM（s0 在崩前观测到 %.1f–%.1f MiB，**不是**峰值）→ 该跑"
              "**不可评分**，模型值仅列出备查，不入任何统计。" % (lo, hi))


def print_deltas(results: dict, keys) -> None:
    print("\n-- unfused − fused delta（重算工作集的 ×1 量）--")
    hdr = "%-8s %-3s %10s" % ("cfg", "st", "real_d")
    for k in keys:
        hdr += " %10s" % (k[:8] + "_d")
    for k in keys:
        hdr += " %9s" % (k[:7] + "/re")
    print(hdr)
    for lbl, ft, ut in PAIRS:
        for i in range(results[ut].n_stages):
            rd = _delta(REAL[ut], REAL[ft], i)
            line = "%-8s %-3d %10s" % (lbl if i == 0 else "", i,
                                       ("%.1f" % rd) if rd is not None else "OOM")
            ds = []
            for k in keys:
                d = _delta(results[ut].series(k), results[ft].series(k), i)
                ds.append(d)
                line += " %10.1f" % d
            for d in ds:
                line += " %9s" % (("%.3f" % (d / rd)) if rd else "-")
            print(line)


def print_invariants(results: dict, keys) -> list:
    checks = real_invariants()
    print("\n-- 真机不变量（REAL 表自身；这是本项目的两条核心物理发现）--")
    for c in checks:
        print("  " + str(c))
    print("-- 模型不变量（**任何 graph source 都必须满足同样的 ×1 结构**）--")
    for k in keys:
        for c in model_invariants(results, k):
            checks.append(c)
            print("  " + str(c))
    print("-- delta 幅值（**不做断言**，如实报告：两条来源今天都欠读这个 delta）--")
    for lbl, key, rd, md, ratio in magnitude_report(results, keys):
        print("  %-8s %-12s stage%d  model=%9.1f  real=%9.1f  ratio=%s"
              % (lbl, key, INVARIANT_STAGE, md, rd,
                 ("%.3f" % ratio) if ratio else "-"))
    return checks


def print_live_set(results: dict, source: str, top: int, stages) -> None:
    """峰值时刻**逐张量** live-set dump（哪一个字节是谁）。"""
    print("\n-- 峰值时刻逐张量 live-set（source=%s，top %d）--" % (source, top))
    for v in VARIANTS:
        r = results[v.tag]
        res = r.liveness_res[source]
        want = range(r.n_stages) if stages == "all" else (res.tightest_stage,)
        for si in want:
            st = res.per_stage[si]
            print("\n  [%s] stage%d  peak=%.1f MiB @%s/%s  重算工作集峰=%.1f MiB @%s"
                  % (v.tag, si, st.peak_bytes / MiB, st.peak_event, st.peak_substep,
                     st.max_recompute_working_set / MiB, st.max_recompute_substep))
            items = st.top_live(10 ** 9)
            print("    %-28s %-5s %-3s %10s  %s"
                  % ("tensor", "layer", "mb", "MiB", "category"))
            for it in items[:top]:
                print("    %-28s %-5d %-3d %10.1f  %s"
                      % (it.name, it.layer_id, it.mb, it.nbytes / MiB, it.category))
            if len(items) > top:
                rest = sum(i.nbytes for i in items[top:]) / MiB
                print("    %-28s %-5s %-3s %10.1f  (%d 个尾部张量)"
                      % ("…", "", "", rest, len(items) - top))
            print("    breakdown（liveness 类别 → 既有桶名）: %s"
                  % ", ".join("%s=%.1f" % (k, x / MiB)
                              for k, x in sorted(st.bucket_view().items(),
                                                 key=lambda kv: -kv[1])))


def to_json(results: dict, keys, grad_mode: str) -> dict:
    return {
        "grad_mode": grad_mode, "sources": list(keys),
        "real_provenance": REAL_PROVENANCE, "real_sha256": real_fingerprint(),
        "runs": {v.tag: {
            "variant": {"fused": v.fused, "m": v.m, "recompute_on": v.recompute_on,
                        "n_layers": v.n_layers},
            "real_MiB": {str(i): REAL[v.tag][i] for i in range(results[v.tag].n_stages)},
            **{k + "_MiB": {str(i): results[v.tag].series(k)[i]
                            for i in range(results[v.tag].n_stages)} for k in keys},
        } for v in VARIANTS},
        "aggregate": {k: aggregate(results, k) for k in keys},
        "invariants": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                       for c in real_invariants()
                       + [c for k in keys for c in model_invariants(results, k)]],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 单配置模式（source × 一份 yaml；非八跑门）
# ═══════════════════════════════════════════════════════════════════════════

def run_single(config: str, sources, grad_mode: str, top: int) -> int:
    import yaml

    from cost_eval.liveness import simulate_liveness, validate_resolved_graph
    from cost_eval.report import Evaluator
    with open(config, encoding="utf-8") as fh:
        mf = yaml.safe_load(fh)
    m = mf.get("model", {})
    if not m.get("qk_nope_head_dim") and m.get("head_dim") and m.get("qk_rope_head_dim"):
        m["qk_nope_head_dim"] = int(m["head_dim"]) - int(m["qk_rope_head_dim"])
    b, spec = build_bundle(mf)
    rep = Evaluator(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                    check_feasibility=False).evaluate(record_timeline=True)
    print("config=%s  grad_mode=%s  sources=%s" % (config, grad_mode, ",".join(sources)))
    res = {}
    for src in sources:
        res[src] = simulate_liveness(spec, b.parallel, b.optimizer, b.hardware, b.recompute,
                                     b.swap, record_timeline=True, grad_mode=grad_mode,
                                     graph_source=src)
    hdr = "%-6s %12s" % ("stage", "bucket_MiB")
    for s in sources:
        hdr += " %14s" % ("lv:%s" % s)[:14]
    print(hdr)
    for i in range(len(rep.per_stage)):
        line = "%-6d %12.1f" % (i, rep.per_stage[i].peak_bytes / MiB)
        for s in sources:
            line += " %14.1f" % (res[s].per_stage[i].peak_bytes / MiB)
        print(line)
    # 契约体检：每个来源的图逐层过 ResolvedLayer 契约。
    from cost_eval.parallel_model import ParallelModel

    from cost_eval.liveness import resolve_graph
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    pm = ParallelModel(p, spec.dims.n_layers, world)
    rc = 0
    for src in sources:
        v = validate_resolved_graph(resolve_graph(spec, pm, src))
        print("\n契约体检 source=%s: %d 条违约" % (src, len(v)))
        for x in v[:20]:
            print("  " + str(x))
        rc |= (1 if v else 0)
    if top:
        for src in sources:
            st = res[src].per_stage[res[src].tightest_stage]
            print("\n-- stage%d 峰值 live-set（source=%s）--" % (st.stage, src))
            for it in st.top_live(top):
                print("  %-28s %-5d %-3d %10.1f  %s"
                      % (it.name, it.layer_id, it.mb, it.nbytes / MiB, it.category))
    return rc


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    from cost_eval.liveness import available_graph_sources, has_graph_source
    ap = argparse.ArgumentParser(description="167 A/B 八跑验收门（source × config）")
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                    help="含 dsv4h_{fused,unfused}_pp4_recomp.yaml 的目录（默认仓内 A/B 配置）")
    ap.add_argument("--config", default=None,
                    help="单配置模式：只跑这一份 yaml × 各来源（不跑八跑门）")
    ap.add_argument("--sources", default=None,
                    help="逗号分隔的 graph source（默认：全部可用，今天 = hand_spec）")
    ap.add_argument("--grad-mode", default="dataflow", choices=("dataflow", "chain2"))
    ap.add_argument("--deltas", action="store_true", help="另打 unfused−fused 差值表")
    ap.add_argument("--dump-top", type=int, default=0,
                    help="峰值时刻逐张量 live-set dump 条数（0=不打）")
    ap.add_argument("--dump-stages", default="tightest", choices=("tightest", "all"))
    ap.add_argument("--dump-source", default=None, help="dump 用哪个来源（默认最后一个）")
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    ap.add_argument("--gate", action="store_true",
                    help="验收门模式：派生反查 + 真机/模型不变量全过才 exit 0")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")

    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        missing = [s for s in sources if not has_graph_source(s)]
        if missing:
            print("未知 graph source %s；当前可用：%s" % (missing, available_graph_sources()))
            return 2
    else:
        sources = list(available_graph_sources())

    if args.config:
        return run_single(args.config, sources, args.grad_mode, args.dump_top)

    for fused, fn in BASE_YAML.items():
        p = os.path.join(args.base_dir, fn)
        if not os.path.isfile(p):
            print("缺 base yaml: %s" % p)
            return 2

    print("== 167 A/B 八跑验收表 ==  grad_mode=%s  sources=%s" % (args.grad_mode,
                                                                ",".join(sources)))
    print("真机来源：%s" % REAL_PROVENANCE)
    print("base yaml：%s（两份，只差 apply_dsa_kernel_fusion；其余六跑程序化派生 + 反查）\n"
          % args.base_dir)
    results = run_matrix(args.base_dir, sources, args.grad_mode)

    dv = [x for r in results.values() for x in r.derivation_violations]
    if dv:
        print("!! 派生反查失败 %d 条：" % len(dv))
        for x in dv:
            print("   " + x)
        print()
    else:
        print("派生反查：8 跑 × 12 项标签一致性全过（fused 位 / 层数 / m / 重算域 /"
              " compress_ratios / pp·dp·ep·tp·cp / seq / 每 stage 隐藏层数）\n")

    keys = ["bucket"] + sources
    print_matrix_table(results, keys, args.grad_mode)
    print_aggregate(results, keys)
    if args.deltas:
        print_deltas(results, keys)
    checks = print_invariants(results, keys)
    if args.dump_top:
        src = args.dump_source or sources[-1]
        print_live_set(results, src, args.dump_top, args.dump_stages)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(to_json(results, keys, args.grad_mode), fh,
                      ensure_ascii=False, indent=2)
        print("\nJSON → %s" % args.json)

    failed = [c for c in checks if not c.passed]
    if args.gate:
        print("\n== 验收门 ==")
        print("  派生反查违规: %d" % len(dv))
        print("  不变量失败  : %d / %d" % (len(failed), len(checks)))
        for c in failed:
            print("   " + str(c))
        ok = not dv and not failed
        print("  结论: %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
