"""ADVERSARIAL PROBE (read-only): re-derive the L4 g/h per-stage bucket cross-section
from first principles, independent of docs/next_fix_diagnosis_2026-07-28.md.

Checks:
  1. stage -> layer_type mapping for L4 (compress_ratios [0,4,128,4], pp4)
  2. bucket-model peak_event + full breakdown for g and h, all 4 stages
  3. whether g and h peak at the SAME event (and what happens if forced to the same event)
  4. the "every bucket byte-identical except remat_saves" claim
"""
import os
import sys
import warnings

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
warnings.simplefilter("ignore")

MiB = 2 ** 20

import liveness_ab_validate as V  # noqa: E402
from cost_eval.report import Evaluator  # noqa: E402
from cost_eval.liveness import resolve_graph  # noqa: E402
from cost_eval.parallel_model import ParallelModel  # noqa: E402

BUCKETS = ("persistent", "act_live", "gather_buf", "grad_buf", "recomp_scratch",
           "bwd_scratch", "bwd_working_set", "swap_buf", "workspace", "optstep",
           "framework", "kept_frag", "grad_accum", "p2p_buf", "mtp_resident",
           "remat_saves")


def run(tag):
    v = V.BY_TAG[tag]
    mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
    b, spec = V.build_bundle(mf)
    rep = Evaluator(spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap,
                    check_feasibility=False).evaluate(record_timeline=True)
    p = b.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    pm = ParallelModel(p, spec.dims.n_layers, world)
    g = resolve_graph(spec, pm, "hand_spec")
    return rep, g


def stage_layers(g):
    return {st: [(l.layer_id, l.layer_type) for l in layers]
            for st, layers in sorted(g.stages.items())}


def main():
    print("=" * 100)
    print("PART 1 -- stage -> layer mapping (hand_spec resolved graph)")
    print("=" * 100)
    for tag in ("g fused   ON  L4 m4", "h unfused ON  L4 m4",
                "a fused   ON  L8 m4", "b unfused ON  L8 m4"):
        rep, gr = run(tag)
        print("\n%s" % tag)
        for st, ls in stage_layers(gr).items():
            print("   s%d: %s" % (st, ls))

    print()
    print("=" * 100)
    print("PART 2 -- bucket model peak event + breakdown, g vs h (L4)")
    print("=" * 100)
    rg, _ = run("g fused   ON  L4 m4")
    rh, _ = run("h unfused ON  L4 m4")
    for st in range(4):
        pg, ph = rg.per_stage[st], rh.per_stage[st]
        print("\n--- stage %d ---" % st)
        print("  g peak = %10.1f MiB @ %-24s" % (pg.peak_bytes / MiB, pg.peak_event))
        print("  h peak = %10.1f MiB @ %-24s" % (ph.peak_bytes / MiB, ph.peak_event))
        print("  SAME PEAK EVENT? %s" % (pg.peak_event == ph.peak_event))
        print("  delta = %.1f MiB" % ((ph.peak_bytes - pg.peak_bytes) / MiB))
        print("  %-18s %12s %12s %12s" % ("bucket", "g", "h", "h-g"))
        for nm in BUCKETS:
            a = getattr(pg.breakdown, nm) / MiB
            c = getattr(ph.breakdown, nm) / MiB
            flag = "" if abs(c - a) < 1e-6 else "   <<< DIFFERS"
            print("  %-18s %12.1f %12.1f %12.1f%s" % (nm, a, c, c - a, flag))

    print()
    print("=" * 100)
    print("PART 3 -- EVENT-MIGRATION DECOMPOSITION (L4, bucket model)")
    print("  same-event delta  = h(at g's peak event) - g(at g's peak event)")
    print("  migration bonus   = h(peak) - h(at g's peak event)")
    print("=" * 100)
    for st in range(4):
        pg, ph = rg.per_stage[st], rh.per_stage[st]
        tg = {(s.event, s.mb, s.chunk): s for s in pg.timeline}
        th = {(s.event, s.mb, s.chunk): s for s in ph.timeline}
        # locate g's peak sample
        gpk = max(pg.timeline, key=lambda s: s.total_bytes)
        key = (gpk.event, gpk.mb, gpk.chunk)
        hs = th.get(key)
        print("\n--- stage %d ---" % st)
        print("  g peak event key      : %s" % (key,))
        print("  g peak total          : %10.1f MiB" % (gpk.total_bytes / MiB))
        if hs is None:
            print("  h has NO sample at that key -> event set differs between runs!")
        else:
            print("  h total at SAME event : %10.1f MiB" % (hs.total_bytes / MiB))
            print("  h peak total          : %10.1f MiB @ %s"
                  % (ph.peak_bytes / MiB, ph.peak_event))
            same = (hs.total_bytes - gpk.total_bytes) / MiB
            migr = (ph.peak_bytes - hs.total_bytes) / MiB
            print("  => same-event delta   : %10.1f MiB" % same)
            print("  => migration bonus    : %10.1f MiB" % migr)
            print("  => total delta        : %10.1f MiB" % (same + migr))
        # also the reverse: g at h's peak event
        hpk = max(ph.timeline, key=lambda s: s.total_bytes)
        gk = tg.get((hpk.event, hpk.mb, hpk.chunk))
        if gk is not None:
            print("  g total at h's peak ev: %10.1f MiB (g peak %.1f -> headroom %.1f)"
                  % (gk.total_bytes / MiB, pg.peak_bytes / MiB,
                     (pg.peak_bytes - gk.total_bytes) / MiB))

    print()
    print("=" * 100)
    print("PART 4 -- max over ALL events of (h(e) - g(e)) vs peak-difference")
    print("  If model delta==max_e delta but real delta is peak_h-peak_g, the two are")
    print("  NOT the same functional and the comparison is biased.")
    print("=" * 100)
    for st in range(4):
        pg, ph = rg.per_stage[st], rh.per_stage[st]
        tg = {(s.event, s.mb, s.chunk): s.total_bytes for s in pg.timeline}
        best, bkey = None, None
        for s in ph.timeline:
            k = (s.event, s.mb, s.chunk)
            if k in tg:
                d = s.total_bytes - tg[k]
                if best is None or d > best:
                    best, bkey = d, k
        pk_delta = (ph.peak_bytes - pg.peak_bytes) / MiB
        print("  s%d: peak_h-peak_g = %9.1f | max_e (h(e)-g(e)) = %9.1f @ %s"
              % (st, pk_delta, (best or 0) / MiB, bkey))


if __name__ == "__main__":
    main()
