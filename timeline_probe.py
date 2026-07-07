"""内存时间线仿真曲线：逐事件打印 DSv3 缩层某 stage 的显存变化（FWD→BWD）。

评估器 `MemTimeline.simulate(record_timeline=True)` 会把**每个事件**（逐层 FWD / fwd_end / 逐层 BWD）
的总占用 + 8 桶快照记进 `StagePeak.timeline`。本脚本取某 stage 的曲线，打出文本表 + ASCII 折线，
并在有 matplotlib 时输出堆叠面积 PNG（analysis/timeline_<cfg>.png）。可与真机内存 timeline 对齐比对。

用法（env 驱动，同 validate_dsv3）：
  python timeline_probe.py                 # 默认 4L FSDP-2 anchor，stage 0
  SIM_LAYERS=8 SIM_PP=2 SIM_STAGE=1 python timeline_probe.py
"""
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator
from cost_eval.parallel_model import ParallelModel
from validate_dsv3 import build_dsv3_spec, MEASURED, RESIDUAL_MiB

MiB = 2 ** 20
GiB = 2 ** 30


def build_report(N, dp_shard, tp, ep, pp):
    spec, d, full_layers = build_dsv3_spec(N)
    mbs = pp if pp > 1 else 1
    pc = ParallelConfig(dp_shard=dp_shard, tp=tp, ep=ep, pp=pp, cp=1,
                        sequence_parallel=True, num_microbatches=mbs)
    opt = OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4)
    ev = Evaluator(spec, pc, opt,
                   HardwareSpec(max_device_memory=59 * GiB, framework_reserve=RESIDUAL_MiB * MiB),
                   RecomputeSpec(mode="full", full_layers=full_layers), SwapSpec())
    rep = ev.evaluate(record_timeline=True)
    pm = ParallelModel(pc, spec.dims.n_layers, dp_shard * tp * pp)
    return rep, pm, mbs


def ascii_curve(samples, width=48):
    """每个事件一行：event 标签 + 归一化条 + total MiB。峰值行标 ★。"""
    peak = max(s.total_bytes for s in samples)
    lines = []
    for s in samples:
        filled = int(round(s.total_bytes / peak * width)) if peak else 0
        bar = "█" * filled + "·" * (width - filled)
        star = " ★peak" if s.total_bytes == peak else ""
        lines.append(f"  {s.idx:>2} {s.event:<10} |{bar}| {s.total_bytes / MiB:8.1f}{star}")
    return "\n".join(lines)


def maybe_png(samples, title, path):
    """有 matplotlib 就画 8 桶堆叠面积图（内存 vs 事件），返回是否成功。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    buckets = ["persistent", "act_live", "gather_buf", "grad_buf",
               "recomp_scratch", "bwd_scratch", "bwd_working_set", "swap_buf",
               "workspace", "optstep", "framework"]
    xs = [s.idx for s in samples]
    stacks = [[getattr(s.breakdown, b) / MiB for s in samples] for b in buckets]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(xs, *stacks, labels=buckets)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.event for s in samples], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("MiB")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main():
    N = int(os.environ.get("SIM_LAYERS", "4"))
    EP = int(os.environ.get("SIM_EP", "1"))
    TP = int(os.environ.get("SIM_TP", "1"))
    PP = int(os.environ.get("SIM_PP", "1"))
    DPSHARD = int(os.environ.get("SIM_DPSHARD", "2"))
    STAGE = int(os.environ.get("SIM_STAGE", "0"))

    rep, pm, mbs = build_report(N, DPSHARD, TP, EP, PP)
    sp = rep.per_stage[STAGE]
    samples = sp.timeline
    cfg = f"N{N}_dp{DPSHARD}_tp{TP}_ep{EP}_pp{PP}_stage{STAGE}"

    print(f"=== 内存时间线仿真  {cfg}  ({len(samples)} 事件) ===")
    print(f"stage{STAGE} 层={pm.stage_layers(STAGE)}  峰值={sp.peak_bytes / MiB:.1f} MiB @ {sp.peak_event}  "
          f"tightest=stage{rep.tightest_stage}  oom={sp.oom}")
    meas = MEASURED.get((N, EP), (None, None))[0]
    if meas and STAGE == 0 and PP == 1:
        print(f"真机 max_memory_allocated = {meas:.1f} MiB  → 仿真峰值/真机 = {sp.peak_bytes / MiB / meas:.4f}")
    print()
    print(f"{'idx':>3} {'event':<10} {'total':>9} {'persist':>8} {'act':>8} {'gather':>7} "
          f"{'grad':>7} {'recomp':>7} {'bwd_scr':>8} {'workspc':>8} {'framewk':>7}")
    for s in samples:
        b = s.breakdown
        print(f"{s.idx:>3} {s.event:<10} {s.total_bytes / MiB:>9.1f} {b.persistent / MiB:>8.1f} "
              f"{b.act_live / MiB:>8.1f} {b.gather_buf / MiB:>7.1f} {b.grad_buf / MiB:>7.1f} "
              f"{b.recomp_scratch / MiB:>7.1f} {b.bwd_scratch / MiB:>8.1f} {b.workspace / MiB:>8.1f} "
              f"{b.framework / MiB:>7.1f}")
    print("\nASCII 内存曲线（每事件总占用，归一化到峰值）:")
    print(ascii_curve(samples))

    png = os.path.join("analysis", f"timeline_{cfg}.png")
    if maybe_png(samples, f"DSv3 mem timeline  {cfg}", png):
        print(f"\n[已写] 堆叠面积图 → {png}")
    else:
        print("\n[跳过 PNG] 未装 matplotlib（pip install matplotlib 后可出图）")


if __name__ == "__main__":
    main()
