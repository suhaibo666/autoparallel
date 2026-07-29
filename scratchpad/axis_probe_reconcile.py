"""PROBE (read-only): reconcile the extracted r0 `activation_saves` against
(a) the per-tensor PIN prediction in `docs/next_fix_diagnosis_2026-07-28.md` §5 P2, and
(b) the hand census in `cost_eval/layers/dsv4_hybrid.py:246-268` (the sliding-window
    `else` branch — the diagnosis mis-cited `:188-229`, which is the sparse branch;
    correction from `docs/next_fix_adversarial_review_2026-07-28.md` §7).

The point is NOT to make the number bigger.  It is to show, tensor by tensor, that every
byte the extractor now reports is traceable to a `file:line` + a `bprop_rules.PIN` entry,
and that every byte the hand census has and the extractor does not is one the VJP rules
say is NOT retained.

    PYTHONIOENCODING=utf-8 python scratchpad/axis_probe_reconcile.py

NOTHING in cost_eval/ is modified.
"""
import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
warnings.simplefilter("ignore")
os.environ["COST_EVAL_EXTRACTED_ALLOW_PARTIAL"] = "1"

MiB = 2 ** 20

import liveness_ab_validate as V  # noqa: E402
from cost_eval.parallel_model import ParallelModel  # noqa: E402
from cost_eval.opdag import to_resolved as TR  # noqa: E402

TAG = "b unfused ON  L8 m4"

#: `docs/next_fix_diagnosis_2026-07-28.md` §5 P2 的逐张量预言（bf16 口径的 MiB；
#: 抽取侧这些量实际是 fp32 → 字节 ×2。这里比的是**在场/不在场**与相对量级）。
PREDICTED_PRESENT = {"q_bm": 512, "kv_bm": 1024, "kvo_bm": 1024,
                     "scores": 128, "exp_scores": 128, "aw_bm": 128}
PREDICTED_ABSENT = {"uq_f32": 512, "kv_gathered": 512, "attn_weights": 128,
                    "uscore1": 128, "uout_f32": 512, "uout_pm": 512}
#: 手写普查里名字不同、指同一物的对照（`dsv4_hybrid.py:251-263` ↔ `csa.py` 局部名）。
CENSUS_ALIAS = {"uq_bm": "q_bm", "ukv_bm": "kv_bm", "kv_g_fp32": "kvo_bm",
                "uexp": "exp_scores", "uaw_bm": "aw_bm", "uscore2": "scores"}


def build(tag):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    return spec, ParallelModel(p, spec.dims.n_layers, world)


def main():
    spec, pm = build(TAG)
    g = TR.resolve_graph(spec, pm, allow_partial=True)
    layer = next(L for layers in g.stages.values() for L in layers if L.layer_id == 1)

    rows, seen = [], set()
    for op in layer.ops:
        for r in op.saves:
            if r.name in seen:
                continue
            seen.add(r.name)
            rows.append((r.local_numel * r.dtype_bytes, r.name, r.sym_shape, r.src))

    total = sum(x[0] for x in rows)
    csa = [x for x in rows if "csa.py" in (x[3] or "")]
    csa_b = sum(x[0] for x in csa)

    print("=== extracted L1 (dsv4hyb_r0_dense, unfused) ===")
    print("  layer total          %10.1f MiB over %d tensors" % (total / MiB, len(rows)))
    print("  of which csa.py:*    %10.1f MiB over %d tensors" % (csa_b / MiB, len(csa)))
    print("  （诊断书 §5 P2 对 `unfused_compressed_sparse_attn` 函数体的净预言 = 3072 MiB）")
    print("  逐条 csa.py：")
    for b, nm, sym, src in sorted(csa, reverse=True):
        print("    %9.1f MiB  %-24s %-38s %s" % (b / MiB, nm, sym, src))

    base = {}
    for b, nm, _s, _src in rows:
        base[nm.split("__i")[0].split("#")[0]] = b

    print("\n  --- 诊断书 P2 预言「应当出现」的 6 项 ---")
    for nm, mb in sorted(PREDICTED_PRESENT.items(), key=lambda kv: -kv[1]):
        got = base.get(nm)
        print("    %-14s 预言 %6d MiB(bf16 口径) → 实测 %s"
              % (nm, mb, ("%.1f MiB" % (got / MiB)) if got else "**不在场**"))
    print("\n  --- 诊断书 P2 预言「应当**不**出现」的 6 项 ---")
    for nm, mb in sorted(PREDICTED_ABSENT.items(), key=lambda kv: -kv[1]):
        got = base.get(nm)
        print("    %-14s 手写普查 %6d MiB → 实测 %s"
              % (nm, mb, "**在场**（预言被证伪）" if got else "不在场（如预言）"))
    print("\n  （手写普查的别名对照：%s）"
          % ", ".join("%s≡%s" % (a, b) for a, b in sorted(CENSUS_ALIAS.items())))


if __name__ == "__main__":
    main()
