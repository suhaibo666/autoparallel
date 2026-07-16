"""新架构下 仿真器 vs 真机 综合差异报告（2026-07-09）。

本地跑当前仿真器（含 B margin + fp32-norm + 全部修复）对**全部已捕获真机锚点**，算 sim/real 比值 +
汇总（OOM-安全性、±5% 命中、最差欠/过预测）。无 NPU / 无网络。真机值出处见各 DIAGNOSIS.md / §14。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

GiB = 2 ** 30
MiB = 2 ** 20
ATTN = {lid: {"linear_q", "linear_kv", "q_a", "kv_a", "rope", "flash", "o_proj"} for lid in range(1, 9)}
MLP = {lid: {"fc", "swiglu", "gelu", "router", "dispatch", "e_", "combine", "shared"} for lid in range(1, 9)}
BOTH = {lid: ATTN[lid] | MLP[lid] for lid in range(1, 9)}
_opt = OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4)
_hw = HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0)


def dsv3(N, rc, *, B=1, dp=2, cp=1, pp=1, ep=1, mbs=1, method="colossal", stage=0):
    spec, d, fl = build_dsv3_spec(N)
    d.B = B
    pc = ParallelConfig(dp_shard=dp, cp=cp, tp=1, ep=ep, pp=pp, sequence_parallel=True,
                        num_microbatches=mbs, context_parallel_method=method)
    r = Evaluator(spec, pc, _opt, _hw, rc, SwapSpec()).evaluate()
    return r.per_stage[stage].peak_bytes / MiB


FULL8 = RecomputeSpec("full", full_layers=set(range(1, 9)))
FULL4 = RecomputeSpec("full", full_layers={1, 2, 3, 4})
NONE = RecomputeSpec("None")

# (label, regime, sim_value_fn, real_MiB)
rows = []
rows.append(("DSv3 4L full (dp2,sp)",           "full",       dsv3(4, FULL4),                                   12473.1))
rows.append(("DSv3 8L full (dp2)",              "full",       dsv3(8, FULL8),                                   13953.3))
rows.append(("DSv3 4L full ep=2",               "full+ep",    dsv3(4, FULL4, ep=2),                             12474.1))
rows.append(("cp2 colossal full 4L (B2)",       "cp+full",    dsv3(4, FULL4, B=2, dp=1, cp=2, method="colossal"), 12433.0))
rows.append(("cp2 ulysses full 4L (B2)",        "cp+full",    dsv3(4, FULL4, B=2, dp=1, cp=2, method="ulysses"),  12441.0))
rows.append(("pp2-stage0 (optstep)",            "pp+norecomp", dsv3(8, NONE, B=2, dp=1, pp=2, mbs=2, stage=0),    10246.0))
rows.append(("pp2-stage1 (loss,k_ce=8)",        "pp+norecomp", dsv3(8, NONE, B=2, dp=1, pp=2, mbs=2, stage=1),    45655.0))
rows.append(("cp2-none (loss,k_ce=4)",          "cp+norecomp", dsv3(8, NONE, B=2, dp=1, cp=2, method="colossal"),  20119.4))
# 2026-07-16 测试工程轮新采（run_axis.sh TAG=memval_none 卡6/7）：plain dp2 无重算锚点。
# 复现并证实旧"feed_forward 19967 错名数据"实为真无重算测量（两次独立采样一致）。
# 欠预测 ~1567 MiB 与 cp2-none 同族（无重算 MoE 保留态）；其中 ~196 MiB 已定性为
# MoE decoder 缺 pre_mlp_layernorm op（F1，gpt_layer_specs.py:110/:129 真机源码确证）。
rows.append(("DSv3 8L none (dp2)",              "norecomp",   dsv3(8, NONE),                                     19967.3))
rows.append(("select self_attn (keep-FFN)",     "select",     dsv3(8, RecomputeSpec("select", select_ops=ATTN)), 18828.2))
rows.append(("select mlp (keep-attn)",          "select",     dsv3(8, RecomputeSpec("select", select_ops=MLP)),  15764.7))
rows.append(("select both (=full,退化端)",       "select",     dsv3(8, RecomputeSpec("select", select_ops=BOTH)), 13953.3))

# DSv4-fused（独立模型）
try:
    from validate_dsv4align import evaluate as dsv4_eval
    dsv4_sim = dsv4_eval(num_layers=4, mhc=0, mtp=0).per_stage[0].peak_bytes / MiB
    rows.append(("DSv4-fused (base)", "dsv4+norecomp", dsv4_sim, 15415.5))
except Exception as e:
    rows.append(("DSv4-fused (base)", "dsv4+norecomp", None, 15415.5))
    print("DSv4 eval failed:", e)

# ── 输出 ──
print("\n## 新架构下 仿真器 vs 真机 —— 综合差异报告\n")
print("| 锚点 | regime | 仿真 MiB | 真机 MiB | ratio(sim/real) | OOM |")
print("|---|---|---|---|---|---|")
ratios = []
for label, regime, sim, real in rows:
    if sim is None:
        print(f"| {label} | {regime} | ERR | {real:.0f} | — | — |")
        continue
    ratio = sim / real
    ratios.append((label, ratio, sim, real))
    oom = "✅安全" if ratio >= 0.995 else ("⚠️欠" if ratio < 0.95 else "≈")
    print(f"| {label} | {regime} | {sim:.1f} | {real:.0f} | **{ratio:.3f}** | {oom} |")

# 汇总
import statistics
rs = [r for _, r, _, _ in ratios]
absdev = [abs(r - 1) for r in rs]
within5 = sum(1 for r in rs if 0.95 <= r <= 1.05)
under = [(l, r) for l, r, _, _ in ratios if r < 0.95]
over = [(l, r) for l, r, _, _ in ratios if r > 1.05]
print(f"\n### 汇总（{len(rs)} 锚点）")
print(f"- 平均 |ratio−1| = **{statistics.mean(absdev)*100:.1f}%**；中位比值 {statistics.median(rs):.3f}")
print(f"- ratio ∈ [0.95,1.05]：**{within5}/{len(rs)}**")
print(f"- 最大过预测：{max(rs):.3f}（{[l for l,r,_,_ in ratios if r==max(rs)][0]}）")
print(f"- 最大欠预测：{min(rs):.3f}（{[l for l,r,_,_ in ratios if r==min(rs)][0]}）")
print(f"- OOM-不安全（欠预测 <0.95）：{under if under else '无'}")
print(f"- 过预测 >1.05：{over if over else '无'}")
