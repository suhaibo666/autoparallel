"""PROBE (read-only): three-way per-layer-type census — real vs hand vs extracted.

Run `c` = fused / NO recompute / L8 / m4 / seq4096 (pp4·dp2·ep2).

`act_live` is exactly `structure_mem.activation_saves`, so this probe reproduces
`activation_saves` **per tensor** for the hand census (including the norm-fp32 lift that
`structure_mem._dt` applies) and puts it next to the extracted graph's per-tensor saves.

    PYTHONIOENCODING=utf-8 python scratchpad/census_threeway.py [--extracted]

NOTHING in cost_eval/ is modified.
"""
import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
warnings.simplefilter("ignore")
os.environ.setdefault("COST_EVAL_EXTRACTED_ALLOW_PARTIAL", "1")

MiB = 2 ** 20
TAG = "c fused   OFF L8 m4"

import liveness_ab_validate as V  # noqa: E402
from cost_eval.parallel_model import ParallelModel  # noqa: E402
from cost_eval.shape_eval import ShapeEval  # noqa: E402
from cost_eval.structure_mem import _align_up, _dt, _norm_save_names  # noqa: E402

#: 真机实测（167 / 2026-07-29 逐 (微批,层,阶段) 轨迹, run c）——**测量值，不得编辑**
#: 键是**真机 decoder 层号 L0..L7**；评估器 layer_id 里 0 是 embedding → decoder k = layer_id-1。
REAL_PER_LAYER = {0: 2235.1, 1: 2355.3, 2: 2124.2, 3: 2354.0,
                  4: 2105.0, 5: 2370.0, 6: 2119.2, 7: 2285.5}
RATIOS = [0, 4, 128, 4, 128, 4, 128, 4]


def _dec(layer_id):
    """评估器 layer_id → 真机 decoder 层号（embedding 占 0，lm_head 在末尾）。"""
    k = layer_id - 1
    return k if 0 <= k < len(RATIOS) else None


def build(tag=TAG):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    return spec, ParallelModel(p, spec.dims.n_layers, world), b


def hand_rows(layer, blk, norm_dtype):
    """复现 structure_mem.activation_saves 的逐张量分解（同 dedup / 同 _dt / 同 blk 对齐）。"""
    norm_names = _norm_save_names(layer.ops) if norm_dtype else frozenset()
    saves = {}
    owner = {}
    for op in layer.ops:
        for s in op.saves:
            if s.name not in saves:
                owner[s.name] = op.name
            saves[s.name] = s
    rows = []
    for name, s in saves.items():
        db = _dt(s, norm_names, norm_dtype)
        rows.append((_align_up(s.local_numel * db, blk), name, db, s.dtype_bytes,
                     owner[name], name in norm_names))
    rows.sort(reverse=True)
    return rows


def extracted_rows(layer):
    rows, seen = [], set()
    for op in layer.ops:
        for r in op.saves:
            if r.name in seen:
                continue
            seen.add(r.name)
            rows.append((r.local_numel * r.dtype_bytes, r.name, r.dtype_bytes,
                         getattr(r, "sym_shape", ""), getattr(r, "src", "")))
    rows.sort(reverse=True)
    return rows


def main():
    spec, pm, bundle = build()
    blk = getattr(bundle.hardware, "alloc_block_bytes", 1)
    norm_dtype = getattr(spec.dims, "norm_compute_dtype_bytes", 0)
    print("blk=%r  norm_compute_dtype_bytes=%r  layer_pattern=%r"
          % (blk, norm_dtype, list(spec.layer_pattern)))

    g = ShapeEval().resolve(spec, pm)
    print("\n" + "=" * 108)
    print("HAND census — activation_saves per layer (fused / no-recompute)")
    print("=" * 108)
    per_type = {}
    for st, layers in sorted(g.stages.items()):
        for L in layers:
            rows = hand_rows(L, blk, norm_dtype)
            tot = sum(r[0] for r in rows)
            k = _dec(L.layer_id)
            real = REAL_PER_LAYER.get(k)
            r = RATIOS[k] if k is not None else None
            per_type.setdefault((r, L.layer_type), []).append((L.layer_id, tot / MiB, real))
            print("\n  L%d %-28s r=%-4s  hand=%9.1f MiB (%d tensors)  real=%s  ratio=%s"
                  % (L.layer_id, L.layer_type, r, tot / MiB, len(rows),
                     ("%.1f" % real) if real else "-",
                     ("%.3f×" % (tot / MiB / real)) if real else "-"))
            for b, nm, db, own_db, own, is_norm in rows:
                if b < 1 * MiB:
                    continue
                print("      %9.1f MiB  %-22s dtype=%db%s  by=%s"
                      % (b / MiB, nm, db, ("(lifted from %db)" % own_db) if is_norm and db != own_db else "",
                         own))
            small = [r for r in rows if r[0] < 1 * MiB]
            if small:
                print("      %9.1f MiB  [%d tensors < 1 MiB]" % (sum(x[0] for x in small) / MiB, len(small)))
    print("\n  --- per layer-type summary (hand) ---")
    for k, v in sorted(per_type.items(), key=lambda kv: str(kv[0])):
        print("   r=%-4s %-28s  layers=%s  hand=%s"
              % (k[0], k[1], [x[0] for x in v], ["%.1f" % x[1] for x in v]))

    if "--extracted" in sys.argv:
        from cost_eval.opdag import to_resolved as TR
        eg = TR.resolve_graph(spec, pm, allow_partial=True)
        print("\n" + "=" * 108)
        print("EXTRACTED census — saves per layer (fused / no-recompute)")
        print("=" * 108)
        for st, layers in sorted(eg.stages.items()):
            for L in layers:
                rows = extracted_rows(L)
                tot = sum(r[0] for r in rows)
                real = REAL_PER_LAYER.get(_dec(L.layer_id))
                print("\n  L%d %-28s  extracted=%9.1f MiB (%d tensors)  real=%s  ratio=%s"
                      % (L.layer_id, L.layer_type, tot / MiB, len(rows),
                         ("%.1f" % real) if real else "-",
                         ("%.3f×" % (tot / MiB / real)) if real else "-"))
                for b, nm, db, sym, src in rows:
                    if b < 1 * MiB:
                        continue
                    print("      %9.1f MiB  %-24s %-34s %db  %s" % (b / MiB, nm, sym, db, src))
                small = [r for r in rows if r[0] < 1 * MiB]
                if small:
                    print("      %9.1f MiB  [%d tensors < 1 MiB]" % (sum(x[0] for x in small) / MiB, len(small)))
        print("\n  coverage: %s" % getattr(eg, "coverage", None))


if __name__ == "__main__":
    main()
