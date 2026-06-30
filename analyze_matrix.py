"""多维并行策略峰值显存扫描（A–E）：复用 validate_dsv3 的 DSv3 缩层模型。

纯解析侧（无 MindSpore / 无真机）：对每个 ParallelConfig 跑一次 Evaluator.evaluate()，
收集 预测峰值(含reserve) / 结构峰值(剔framework) / persistent / 8 桶明细 / HCCL 子通信器数。
模型只构建一次（按 N 缓存），跨配置复用，仅 ParallelConfig/Evaluator 变化。

扫描（全程 dp_replicate=cp=1，sequence_parallel=True，PP 时 num_microbatches=PP）：
  A. FSDP：dp_shard ∈ {1,2,4,8}，tp=ep=pp=1
  B. EP@FSDP8：dp_shard=8，ep ∈ {1,2,4,8}，tp=pp=1
  C. TP：dp_shard=2，tp ∈ {1,2,4}，ep=pp=1
  D. PP：dp_shard=2，pp ∈ {1,2,4}（num_microbatches=pp），tp=ep=1，N=8（per-stage[0]）
  E. 组合

用法：python analyze_matrix.py   （从 repo 根目录，PYTHONPATH=repo 根）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator
from cost_eval.parallel_model import ParallelModel
from cost_eval.framework import num_distinct_communicators
from validate_dsv3 import build_dsv3_spec, MEASURED, RESIDUAL_MiB

MiB = 2 ** 20
GiB = 2 ** 30

# 模型按 N 缓存（构建一次复用）
_SPEC_CACHE = {}


def get_spec(N):
    if N not in _SPEC_CACHE:
        _SPEC_CACHE[N] = build_dsv3_spec(N)
    return _SPEC_CACHE[N]


def evaluate(N=4, dp_shard=1, tp=1, ep=1, pp=1, stage=0):
    """跑一个配置，返回该 stage 的指标 dict。"""
    spec, d, full_layers = get_spec(N)
    mbs = pp if pp > 1 else 1
    pc = ParallelConfig(dp_shard=dp_shard, tp=tp, ep=ep, pp=pp, cp=1,
                        sequence_parallel=True, num_microbatches=mbs)
    opt = OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4)
    ev = Evaluator(spec, pc, opt,
                   HardwareSpec(max_device_memory=59 * GiB,
                                framework_reserve=RESIDUAL_MiB * MiB),
                   RecomputeSpec(mode="full", full_layers=full_layers), SwapSpec())
    rep = ev.evaluate()
    p = rep.per_stage[stage]
    b = p.breakdown
    # stage 拓扑（仅供 PP 行解读用）
    world = dp_shard * tp * pp  # cp=dp_replicate=1
    pm = ParallelModel(pc, spec.dims.n_layers, world)
    return {
        "N": N, "dp_shard": dp_shard, "tp": tp, "ep": ep, "pp": pp,
        "pred_peak": p.peak_bytes / MiB,
        "struct_peak": (p.peak_bytes - b.framework) / MiB,
        "persistent": b.persistent / MiB,
        "act_live": b.act_live / MiB,
        "gather_buf": b.gather_buf / MiB,
        "grad_buf": b.grad_buf / MiB,
        "recomp_scratch": b.recomp_scratch / MiB,
        "bwd_scratch": b.bwd_scratch / MiB,
        "swap_buf": b.swap_buf / MiB,
        "workspace": b.workspace / MiB,
        "hccl": num_distinct_communicators(pc),
        "oom": rep.oom,
        "peak_event": p.peak_event,
        "n_stages": len(rep.per_stage),
        "stage0_layers": pm.stage_layers(0),
    }


_COLS = ["pred_peak", "struct_peak", "persistent", "act_live", "gather_buf",
         "grad_buf", "recomp_scratch", "bwd_scratch", "workspace", "hccl"]
_HDR = ["config"] + ["pred_peak", "struct_peak", "persistent", "act_live", "gather_buf",
                     "grad_buf", "recomp", "bwd_scr", "workspc", "hccl"]


def _fmt(label, r):
    cells = [label]
    for c in _COLS:
        v = r[c]
        cells.append(str(int(v)) if c == "hccl" else f"{v:.1f}")
    return "| " + " | ".join(cells) + " |"


def print_table(title, rows):
    print(f"\n### {title}\n")
    print("| " + " | ".join(_HDR) + " |")
    print("|" + "|".join(["---"] * len(_HDR)) + "|")
    for label, r in rows:
        print(_fmt(label, r))


def main():
    # ---- Sweep A: FSDP ----
    A = [(f"dp_shard={n}", evaluate(N=4, dp_shard=n)) for n in (1, 2, 4, 8)]
    print_table("Sweep A — FSDP (N=4, tp=ep=pp=1)", A)

    # ---- Sweep B: EP on FSDP=8 ----
    B = [(f"ep={e}", evaluate(N=4, dp_shard=8, ep=e)) for e in (1, 2, 4, 8)]
    print_table("Sweep B — EP on dp_shard=8 (N=4, tp=pp=1)", B)

    # ---- Sweep C: TP ----
    C = [(f"tp={t}", evaluate(N=4, dp_shard=2, tp=t)) for t in (1, 2, 4)]
    print_table("Sweep C — TP (N=4, dp_shard=2, ep=pp=1)", C)

    # ---- Sweep D: PP (N=8, per-stage[0]) ----
    D = [(f"pp={p}", evaluate(N=8, dp_shard=2, pp=p)) for p in (1, 2, 4)]
    print_table("Sweep D — PP (N=8, dp_shard=2, tp=ep=1, per-stage[0])", D)
    print("\nPP stage 拓扑（per-stage[0] = stage 0）:")
    for label, r in D:
        print(f"  {label}: n_stages={r['n_stages']}, stage0_layers={r['stage0_layers']}, "
              f"in-flight(warmup)mb={max(r['pp']-1, 0)}, peak_event={r['peak_event']}")

    # ---- Sweep E: Combinations ----
    combos = [
        ("dp_shard=2,ep=2",            dict(N=4, dp_shard=2, ep=2)),
        ("dp_shard=4,tp=2",            dict(N=4, dp_shard=4, tp=2)),
        ("dp_shard=2,tp=2,ep=2",       dict(N=4, dp_shard=2, tp=2, ep=2)),
        ("dp_shard=2,pp=2",            dict(N=8, dp_shard=2, pp=2)),
        ("dp_shard=2,tp=2,ep=2,pp=2",  dict(N=8, dp_shard=2, tp=2, ep=2, pp=2)),
    ]
    E = [(label, evaluate(**kw)) for label, kw in combos]
    print_table("Sweep E — Combinations", E)

    # ---- Validation against real machine ----
    print("\n### Validation against real machine\n")
    print("| anchor config | pred_peak | measured | ratio (pred/meas) |")
    print("|---|---|---|---|")
    anchors = [
        ("N=4, dp_shard=2 (A)",        evaluate(N=4, dp_shard=2),        MEASURED[(4, 1)][0]),
        ("N=4, dp_shard=2, ep=2 (E)",  evaluate(N=4, dp_shard=2, ep=2),  MEASURED[(4, 2)][0]),
        ("N=8, dp_shard=2",            evaluate(N=8, dp_shard=2),        MEASURED[(8, 1)][0]),
    ]
    for label, r, meas in anchors:
        pred = r["pred_peak"]
        print(f"| {label} | {pred:.1f} | {meas:.1f} | {pred / meas:.4f} |")


if __name__ == "__main__":
    main()
