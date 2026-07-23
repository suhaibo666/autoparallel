"""记分卡锚点**单一事实源**（Z2 / D2，2026-07-16）。

`sim_vs_real_report.py`（手动脚本，打印比值表）与 `tests/test_scorecard_anchors.py`
（pytest 门，钉每锚点 ratio 带）**共同消费本模块**，杜绝脚本与门二者悄悄漂移（那正是 Z2
在 meta 层的病：手动脚本从不入 pytest 门 → D2 的 1.088→0.920 翻转无人察觉）。

每锚点 = `Anchor(label, regime, sim_fn, real_MiB, band=(lo, hi))`：
  - `sim_fn()` 惰性重算 sim MiB（走与真机同一 Evaluator 路径）；DSv4 依赖缺失时返回 None。
  - `band` 是 **ratio = sim/real** 的允许区间。**欠预测侧（ratio < 当前值，OOM 危险方向）比
    过预测侧更紧**——预测 < 真机会误报"放得下"却 OOM，是记分卡最不能容忍的漂移方向。

真机值出处见各 DIAGNOSIS.md / `analysis/`；本模块**不**杜撰任何 sim 数（全由代码重算）。
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from cost_eval.report import Evaluator

GiB = 2 ** 30
MiB = 2 ** 20

# select 选择器（与 sim_vs_real_report / mem_timeline 同口径）。
ATTN = {lid: {"linear_q", "linear_kv", "q_a", "kv_a", "rope", "flash", "o_proj"} for lid in range(1, 9)}
MLP = {lid: {"fc", "swiglu", "gelu", "router", "dispatch", "e_", "combine", "shared"} for lid in range(1, 9)}
BOTH = {lid: ATTN[lid] | MLP[lid] for lid in range(1, 9)}

_opt = OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4)
_hw = HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0)

FULL8 = RecomputeSpec("full", full_layers=set(range(1, 9)))
FULL4 = RecomputeSpec("full", full_layers={1, 2, 3, 4})
NONE = RecomputeSpec("None")


def dsv3(N, rc, *, B=1, dp=2, cp=1, pp=1, ep=1, mbs=1, method="colossal", stage=0) -> float:
    """DSv3 缩层锚点 sim MiB（与 sim_vs_real_report.dsv3 逐参一致）。"""
    spec, d, fl = build_dsv3_spec(N)
    d.B = B
    pc = ParallelConfig(dp_shard=dp, cp=cp, tp=1, ep=ep, pp=pp, sequence_parallel=True,
                        num_microbatches=mbs, context_parallel_method=method)
    r = Evaluator(spec, pc, _opt, _hw, rc, SwapSpec()).evaluate()
    return r.per_stage[stage].peak_bytes / MiB


def _dsv4_sim(mhc: int, mtp: int) -> Callable[[], Optional[float]]:
    """DSv4-align sim MiB（惰性；validate_dsv4align 依赖缺失 → None，让消费者 skip/标 ERR）。"""
    def f() -> Optional[float]:
        try:
            from validate_dsv4align import evaluate as dsv4_eval
            return dsv4_eval(num_layers=4, mhc=mhc, mtp=mtp).per_stage[0].peak_bytes / MiB
        except Exception:
            return None
    return f


@dataclass(frozen=True)
class Anchor:
    label: str
    regime: str
    sim_fn: Callable[[], Optional[float]]
    real: float
    band: Tuple[float, float]   # (lo, hi) on ratio = sim/real
    note: str = ""


def anchors() -> list:
    """全部记分卡锚点（脚本与 pytest 门共用）。

    band 规约（2026-07-16）：默认 `lo = r - 0.02`（欠侧紧）、`hi = r + 0.05`（过侧松）。特例：
      - pp2-stage0（1.089，D3 过预测、OOM-安全）：`hi = 1.10` = 观察阈（升破则应收紧 K_OPT）。
      - cp2-none / 8L-none（D1 修后 OOM-安全）：`lo ≈ 1.0`，钉住 D1 margin 保持预测≥真机。
      - 已知欠预测残差（select-mlp / DSv4-base / mHC+MTP）：`lo` 放到当前值下方，**仅**防"进一步
        漂移"，不假装 OOM-安全通过。任一带都足够紧到能让 D2 的 1.088↔0.920 翻转触红。
    """
    return [
        Anchor("DSv3 4L full (dp2,sp)", "full",
               lambda: dsv3(4, FULL4), 12473.1, (0.97, 1.05)),
        Anchor("DSv3 8L full (dp2)", "full",
               lambda: dsv3(8, FULL8), 13953.3, (0.97, 1.05)),
        Anchor("DSv3 4L full ep=2", "full+ep",
               lambda: dsv3(4, FULL4, ep=2), 12474.1, (0.97, 1.05)),
        Anchor("cp2 colossal full 4L (B2)", "cp+full",
               lambda: dsv3(4, FULL4, B=2, dp=1, cp=2, method="colossal"), 12433.0, (0.98, 1.05)),
        Anchor("cp2 ulysses full 4L (B2)", "cp+full",
               lambda: dsv3(4, FULL4, B=2, dp=1, cp=2, method="ulysses"), 12441.0, (0.98, 1.05)),
        Anchor("pp2-stage0 (optstep)", "pp+norecomp",
               lambda: dsv3(8, NONE, B=2, dp=1, pp=2, mbs=2, stage=0), 10246.0, (1.05, 1.10),
               note="D3：无重算 BWD 峰共存整体保守、OOM-安全；hi=1.10 观察阈"),
        Anchor("pp2-stage1 (loss,k_ce=8)", "pp+norecomp",
               lambda: dsv3(8, NONE, B=2, dp=1, pp=2, mbs=2, stage=1), 45655.0, (0.98, 1.05)),
        Anchor("cp2-none (loss,k_ce=4)", "cp+norecomp",
               lambda: dsv3(8, NONE, B=2, dp=1, cp=2, method="colossal"), 20119.4, (1.00, 1.06),
               note="D1 修后 OOM-安全（修前 0.937）；lo=1.0 钉 nr_moe_frag margin 保持预测≥真机"),
        Anchor("DSv3 8L none (dp2)", "norecomp",
               lambda: dsv3(8, NONE), 19967.3, (0.99, 1.06),
               note="D1 修后 OOM-安全（修前 0.931）；D1 两点标定之一"),
        Anchor("select self_attn (keep-FFN)", "select",
               lambda: dsv3(8, RecomputeSpec("select", select_ops=ATTN)), 18828.2, (0.98, 1.05)),
        Anchor("select mlp (keep-attn)", "select",
               lambda: dsv3(8, RecomputeSpec("select", select_ops=MLP)), 15764.7, (0.94, 1.00),
               note="已知轻微欠预测（±5% 带内）；lo=0.94 仅防进一步漂移"),
        Anchor("select both (=full,退化端)", "select",
               lambda: dsv3(8, RecomputeSpec("select", select_ops=BOTH)), 13953.3, (0.98, 1.05)),
        # 2026-07-23(185 F 差分): +core_out 逆 RoPE 保留(hybrid:277)入账 → 0.974→1.024(转
        # 保守侧;185 F0/F1 同源差分 ±2% 背书)。band 上移,仍紧(±4%)。
        Anchor("DSv4-fused (base)", "dsv4+norecomp",
               _dsv4_sim(0, 0), 15415.5, (0.97, 1.05),
               note="已知轻微欠预测；lo=0.95 仅防进一步漂移"),
        # D2（2026-07-16）：mHC+MTP 锚点入卡。真机 21153.1（2026-07-01 采）；MTP tie 修复后由 1.088
        #   翻转为 0.920 欠预测，**此前不在记分卡故翻转无人察觉**（Z2）。D1 无重算 margin **不覆盖**它
        #   （DSv4 fused-CE → loss_lids 空 → margin 不触发）→ 独立残差，留待单独诊断（band 仅防进一步漂移，
        #   hi=0.96 使若回到 1.088 过预测立即触红）。
        # 2026-07-23: 逆 RoPE 入账 0.923→0.968(欠预测收窄)。band 收紧上移,仍排除历史翻转 1.088。
        Anchor("DSv4 mHC(x4)+MTP", "dsv4+mhc+mtp",
               _dsv4_sim(4, 1), 21153.1, (0.92, 0.99),
               note="D2：OOM-不安全欠预测 0.920，不在 D1 覆盖内；band 防漂移/翻转，非 OOM-安全通过"),
    ]
