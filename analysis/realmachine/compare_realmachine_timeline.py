"""真机内存 timeline（MindSpore Profiler）vs 仿真内存 timeline 比对。

输入：本目录 memory_record_rank0.csv（真机 allocated/reserved 随时间）+ operator_memory_rank0.csv（逐算子）。
输出：控制台统计 + 峰值算子 + 与仿真曲线的关键台阶对齐；comparison.png（上=真机一步曲线，下=仿真逐事件）。
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.stdout.reconfigure(encoding="utf-8")

MiB = 2 ** 20


# ---------- 1) 真机 memory_record 曲线 ----------
def load_real_curve():
    ts, alloc, reserved = [], [], []
    path = os.path.join(HERE, "memory_record_rank0.csv")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Component"] != "MindSpore":     # 单一组件，避免 +GE 重复
                continue
            ts.append(float(row["Timestamp(us)"]))
            alloc.append(float(row["Total Allocated(MB)"]))
            reserved.append(float(row["Total Reserved(MB)"]))
    t0 = ts[0]
    t = [(x - t0) / 1e6 for x in ts]                # 相对秒
    return t, alloc, reserved


# ---------- 2) 峰值算子（operator_memory） ----------
def peak_operator():
    path = os.path.join(HERE, "operator_memory_rank0.csv")
    best = (-1.0, "", "")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                a = float(row["Allocation Total Allocated(MB)"])
            except (ValueError, KeyError):
                continue
            if a > best[0]:
                best = (a, row["Name"], row.get("Allocation Time(us)", ""))
    return best


# ---------- 3) 仿真曲线（Evaluator record_timeline） ----------
def sim_curve():
    from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
    from cost_eval.report import Evaluator
    from validate_dsv3 import build_dsv3_spec, RESIDUAL_MiB, MEASURED
    GiB = 2 ** 30
    spec, d, full = build_dsv3_spec(4)
    pc = ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1, sequence_parallel=True, num_microbatches=1)
    ev = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                   HardwareSpec(max_device_memory=59 * GiB, framework_reserve=RESIDUAL_MiB * MiB),
                   RecomputeSpec(mode="full", full_layers=full), SwapSpec())
    rep = ev.evaluate(record_timeline=True)
    tl = rep.per_stage[0].timeline
    return ([s.event for s in tl], [s.total_bytes / MiB for s in tl],
            rep.per_stage[0].peak_bytes / MiB, MEASURED[(4, 1)][0])


def main():
    t, alloc, reserved = load_real_curve()
    base = min(alloc)
    peak = max(alloc)
    ipeak = alloc.index(peak)
    pa, pname, ptime = peak_operator()
    events, sim_tot, sim_peak, measured = sim_curve()

    print(f"=== 真机 memory_record (rank0, Component=MindSpore, {len(t)} 采样点) ===")
    print(f"  时长            = {t[-1]:.2f} s（含 warmup+编译+3 步）")
    print(f"  allocated 基线  = {base:8.1f} MB（≈ 常驻 persistent）")
    print(f"  allocated 峰值  = {peak:8.1f} MB @ t={t[ipeak]:.2f}s")
    print(f"  reserved  峰值  = {max(reserved):8.1f} MB")
    print(f"  峰值算子        = {pname!r}  (allocated_total={pa:.1f} MB)")
    print()
    print(f"=== 仿真 timeline (4L FSDP-2, stage0, {len(events)} 事件) ===")
    print(f"  仿真基线(fwd)   = {min(sim_tot):8.1f} MB")
    print(f"  仿真峰值        = {sim_peak:8.1f} MB @ {events[sim_tot.index(max(sim_tot))]}")
    print()
    print("=== 对齐关键台阶（MB） ===")
    print(f"  {'':16s}{'真机':>10}{'仿真':>10}{'真机实测锚点':>14}")
    print(f"  {'基线/persistent':16s}{base:>10.1f}{min(sim_tot):>10.1f}{'3862':>14}")
    print(f"  {'峰值 peak':16s}{peak:>10.1f}{sim_peak:>10.1f}{measured:>14.1f}")
    print(f"  峰值比 仿真/真机 = {sim_peak / peak:.4f}   仿真/实测锚点 = {sim_peak / measured:.4f}")

    _plot(t, alloc, reserved, events, sim_tot, peak, ipeak, measured)


def _plot(t, alloc, reserved, events, sim_tot, peak, ipeak, measured):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n[跳过 PNG] 未装 matplotlib")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(t, alloc, lw=0.7, label="real allocated")
    ax1.plot(t, reserved, lw=0.5, alpha=0.5, label="real reserved")
    ax1.axhline(measured, color="r", ls="--", lw=0.8, label=f"MEMPROBE peak {measured:.0f}")
    ax1.scatter([t[ipeak]], [peak], color="r", zorder=5, s=20)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("MB")
    ax1.set_title("Real machine — MindSpore Profiler allocated-vs-time (3 steps, DSv3 4L FSDP-2)")
    ax1.legend(fontsize=8)
    xs = list(range(len(events)))
    ax2.plot(xs, sim_tot, "-o", ms=4, color="tab:orange", label="sim total")
    ax2.axhline(max(sim_tot), color="r", ls="--", lw=0.8, label=f"sim peak {max(sim_tot):.0f}")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(events, rotation=60, ha="right", fontsize=7)
    ax2.set_ylabel("MB")
    ax2.set_title("Simulator — per-event allocated (cost_eval, same config)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "comparison.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n[已写] {out}")


if __name__ == "__main__":
    main()
