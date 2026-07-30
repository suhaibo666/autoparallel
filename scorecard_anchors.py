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


def _dsv4_sim(mhc: int, mtp: int, use_fused_mhc: bool = False) -> Callable[[], Optional[float]]:
    """DSv4-align sim MiB（惰性；validate_dsv4align 依赖缺失 → None，让消费者 skip/标 ERR）。

    `use_fused_mhc` 显式传（2026-07-30，`docs/fused_mhc_branch_mismatch_2026-07-30.md` §2.2）：
    本族锚点的真机跑（2026-07-01，`prep_dsv4align.py`）容器 vendor OPP **没有**
    `aclnnMhcPreSinkhorn` 融合 kernel → mHC 走**非融合**，故 `False` 就是忠实口径。
    写成显式参数而非依赖默认值，是为了让「这一跑是哪条 mHC 分支」在记分卡这一层可见可审。
    """
    def f() -> Optional[float]:
        try:
            from validate_dsv4align import evaluate as dsv4_eval
            return dsv4_eval(num_layers=4, mhc=mhc, mtp=mtp,
                             use_fused_mhc=use_fused_mhc).per_stage[0].peak_bytes / MiB
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
        # ── 2026-07-30 重钉（`lm_head` 反向 kernel workspace 实测入账）────────────────
        #   167 memory-tracker 实测 dgrad `(2*vocab + 4*H)*(B*S) + 20 MiB + 1024`，16 个
        #   vocab=129280 的点逐字节吻合（docs/head_loss_bwd_workspace_2026-07-30.md 2.1/2.2）。
        #   DSv3 族（H=1792）每条 +1058.0 MiB（B*S=4096）或 +2096.0（B*S=8192）。
        #   ⚠ 因此**十条锚点由欠读翻成过读**（1.02–1.08）= OOM-**安全**侧。band 只上移到
        #   刚好封住当前实测落点（±0.01），**没有为了凑回 ~1.00 而削本项**：本项是逐字节
        #   实测，缺口在 DSv3-era 冻结常数群（实测证据：真机 loss 层反向只共存 **3** 张
        #   瞬态满 vocab fp32 平面，而 `mem_timeline` 的 `K_CE=8` 记 **7** 张 —— 该文 5.1/7.3。
        #   修 `K_CE` 属另一条任务线，**本轮一个字节没动**）。
        Anchor("DSv3 4L full (dp2,sp)", "full",
               lambda: dsv3(4, FULL4), 12473.1, (1.07, 1.09),
               note="2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.996 → **1.081**（过读 = OOM 安全）"),
        Anchor("DSv3 8L full (dp2)", "full",
               lambda: dsv3(8, FULL8), 13953.3, (1.06, 1.08),
               note="2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.992 → **1.068**"),
        Anchor("DSv3 4L full ep=2", "full+ep",
               lambda: dsv3(4, FULL4, ep=2), 12474.1, (1.07, 1.09),
               note="2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.993 → **1.077**"),
        Anchor("cp2 colossal full 4L (B2)", "cp+full",
               lambda: dsv3(4, FULL4, B=2, dp=1, cp=2, method="colossal"), 12433.0, (1.08, 1.09),
               note="2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.999 → **1.084**；每-token 项按 cp 切、常数项不切"),
        Anchor("cp2 ulysses full 4L (B2)", "cp+full",
               lambda: dsv3(4, FULL4, B=2, dp=1, cp=2, method="ulysses"), 12441.0, (1.08, 1.09),
               note="2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.999 → **1.084**"),
        # 2026-07-29 二次重钉（融合 mHC ctx + FusedRMSNorm 不 cast，docs/census_fix_mhc_rmsnorm_2026-07-29.md）：1.089 → **1.021**。仍 OOM-安全，但保守余量被压掉大半。
        # 2026-07-30 三次重钉（**word-embedding 反向 kernel workspace 实测入账**，
        #   docs/head_workspace_2026-07-30.md）：1.021 → **1.032**，且**峰值事件由 `bwd@4`
        #   易主为 `bwd@0`（embedding 反向）**。这是全库唯一被该项抬动的记分卡锚点——只有它的
        #   `bwd@0` 本来就贴着峰（差 952.9 MiB < 该项 1067.753 MiB）。
        #   ⚠ 那 1067.753 MiB **正是这条锚点自己那次采集的 profiler 量到的**：
        #   `analysis/realmachine/pp2_norecomp/op_816362.csv` 里 `GatherDGradV2` 的瞬态块
        #   `Size(KB)=1093379.0` = 1119620096 B，与模型逐字节相同（守卫门见
        #   `tests/test_emb_bwd_kernel_workspace.py`）。**band 刻意不放宽**：1.032 仍在
        #   (1.00, 1.06) 内，继续漂移必须触红。
        Anchor("pp2-stage0 (optstep)", "pp+norecomp",
               lambda: dsv3(8, NONE, B=2, dp=1, pp=2, mbs=2, stage=0), 10246.0, (1.00, 1.06),
               note="D3：无重算 BWD 峰共存整体保守、OOM-安全（2026-07-29 由 1.089 收到 1.021；"
                    "2026-07-30 因 embedding 反向 workspace 实测入账回到 1.032）；"
                    "lo=1.00 守住「仍在安全侧」这一条不变量"),
        # 2026-07-29 二次重钉（融合 mHC ctx + FusedRMSNorm 不 cast，docs/census_fix_mhc_rmsnorm_2026-07-29.md）：1.007 → **0.999**（欠 43.6 MiB / 0.10%）——**刚翻到 OOM-不安全侧**，如实记。
        Anchor("pp2-stage1 (loss,k_ce=8)", "pp+norecomp",
               lambda: dsv3(8, NONE, B=2, dp=1, pp=2, mbs=2, stage=1), 45655.0, (1.04, 1.05),
               note="2026-07-29 起 0.999（OOM-不安全，欠 43.6 MiB）；2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.999 → **1.045**，翻回 OOM-安全侧。这一笔正是这条锚点自己那次采集的 profiler 量到的（op_816365.csv 的 MatMulExt wgrad 2168457216 B 与本律逐字节吻合）"),
        # 2026-07-29 二次重钉（融合 mHC ctx + FusedRMSNorm 不 cast，docs/census_fix_mhc_rmsnorm_2026-07-29.md）：1.007 → **0.991**。D1 margin 未动（仍开、仍是 0.6），是普查去掉了一处真实过读。
        Anchor("cp2-none (loss,k_ce=4)", "cp+norecomp",
               lambda: dsv3(8, NONE, B=2, dp=1, cp=2, method="colossal"), 20119.4, (1.04, 1.05),
               note="2026-07-29 起 0.991（OOM-不安全）；2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.991 → **1.043**，翻回 OOM-安全侧；D1 margin 一个字节没动（仍 0.6）"),
        # 2026-07-29 二次重钉（融合 mHC ctx + FusedRMSNorm 不 cast，docs/census_fix_mhc_rmsnorm_2026-07-29.md）：1.003 → **0.981**。同上，D1 两点标定之一，margin 未重标。
        Anchor("DSv3 8L none (dp2)", "norecomp",
               lambda: dsv3(8, NONE), 19967.3, (1.03, 1.04),
               note="2026-07-29 起 0.981（OOM-不安全）；2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.981 → **1.034**，翻回 OOM-安全侧；D1 margin 一个字节没动（仍 0.6）"),
        # 2026-07-29 二次重钉（融合 mHC ctx + FusedRMSNorm 不 cast，docs/census_fix_mhc_rmsnorm_2026-07-29.md）：1.001 → **0.970**（保留层的 ln1/ln2/q_a_norm/kv_a_norm 不再抬 fp32）。
        Anchor("select self_attn (keep-FFN)", "select",
               lambda: dsv3(8, RecomputeSpec("select", select_ops=ATTN)), 18828.2, (1.02, 1.03),
               note="2026-07-29 起 0.970（OOM-不安全）；2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.970 → **1.026**，翻回 OOM-安全侧；kept_frag margin 一个字节没动"),
        # 2026-07-29 二次重钉（融合 mHC ctx + FusedRMSNorm 不 cast，docs/census_fix_mhc_rmsnorm_2026-07-29.md）：0.955 → **0.932**（已跌出 ±5% 安全带）。
        Anchor("select mlp (keep-attn)", "select",
               lambda: dsv3(8, RecomputeSpec("select", select_ops=MLP)), 15764.7, (0.99, 1.005),
               note="2026-07-29 起 0.932（OOM-不安全，跌出 ±5%）；2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.932 → **0.9993** —— 本项恰好补上了这一格的缺口（仍在欠侧 0.7 MiB，最贴近的一条）"),
        Anchor("select both (=full,退化端)", "select",
               lambda: dsv3(8, RecomputeSpec("select", select_ops=BOTH)), 13953.3, (1.07, 1.08),
               note="2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 1.0005 → **1.076**（select-both 退化到 full，与 8L full 同因）"),
        # 2026-07-23(185 F 差分): +core_out 逆 RoPE 保留(hybrid:277)入账 → 0.974→1.024(转
        # 保守侧;185 F0/F1 同源差分 ±2% 背书)。band 上移,仍紧(±4%)。
        # **2026-07-29**（`docs/census_arbitration_2026-07-29.md`）：手写普查按权威快照逐条订正
        #   （q_hnorm/cg fp32→bf16、去伪 norm 抬升、去 inv_rope_out、补前向 rope 保留对、
        #   cmp_residual→int32 标量、补 sinks/sparse_indices）→ 1.024→**0.908**。
        #   ⚠ 如实记账：这是 OOM-**不安全**方向。之所以仍改：167/2026-07-29 的逐(微批,层,阶段)
        #   **直测**给出 fused 每层驻留 2239.1 MiB@seq4096，而本锚 seq2048 的 3109/层是**全过程
        #   峰值差分**——其"两侧峰在可比事件"的前提已被同项目实测证伪（拆解报告 §5.5）。
        #   两个真机数彼此不自洽（seq2048 的 3109 > seq4096 的 2239），故以直测为准。
        # 2026-07-29 二次重钉（融合 mHC ctx + FusedRMSNorm 不 cast，docs/census_fix_mhc_rmsnorm_2026-07-29.md）：0.908 → **0.902**（融合 mHC ctx −421.8/层、RMSNorm −268.0/层）。
        Anchor("DSv4-fused (base)", "dsv4+norecomp",
               _dsv4_sim(0, 0), 15415.5, (0.93, 0.95),
               note="2026-07-29 起为已知欠预测 0.902；2026-07-30 因 `lm_head` 反向 kernel workspace 实测入账（docs/head_loss_bwd_workspace_2026-07-30.md） 由 0.906 → **0.941**：仍 OOM-不安全，缺口收窄 912.7 MiB"),
        # D2（2026-07-16）：mHC+MTP 锚点入卡。真机 21153.1（2026-07-01 采）；MTP tie 修复后由 1.088
        #   翻转为 0.920 欠预测，**此前不在记分卡故翻转无人察觉**（Z2）。D1 无重算 margin **不覆盖**它
        #   （DSv4 fused-CE → loss_lids 空 → margin 不触发）→ 独立残差，留待单独诊断（band 仅防进一步漂移，
        #   hi=0.96 使若回到 1.088 过预测立即触红）。
        # 2026-07-23: 逆 RoPE 入账 0.923→0.968(欠预测收窄)。band 收紧上移,仍排除历史翻转 1.088。
        # 2026-07-29: 同上普查订正 → 0.968→**0.862**。band 下移，仍排除历史翻转 1.088。
        # 2026-07-29 二次重钉（融合 mHC ctx + FusedRMSNorm 不 cast，docs/census_fix_mhc_rmsnorm_2026-07-29.md）：0.862 → **0.846**。band 下移，仍排除历史翻转 1.088。
        # 2026-07-29 三次重钉（mHC 残差承载归位，docs/census_fix_residual_carrier_2026-07-29.md）：
        #   0.846 → **0.891**。上移几乎全部来自 ④（非融合分支自己的三份 fp32 副本，
        #   `hyper_connection.py:298`/`:109`/`:120`）。
        #   ⚠ **2026-07-30 订正上一行紧邻的一条事实错误**（`docs/fused_mhc_branch_mismatch_2026-07-30.md`
        #   §2.2）：上一轮在此写「真机 21153.1 的站点 yaml 为 `use_fused_mhc: true` → 配置错配」。
        #   **这条是错的** —— 它把 167/185 的 `dsv4h_*_pp4_recomp.yaml` 与本锚点 2026-07-01 的
        #   **dsv4-align** 跑混为一谈了。本锚点真机跑的 mHC **本来就是非融合**，三处独立源一致：
        #     · `.claude/skills/real-machine-memory-sim/prep_dsv4align.py:120-122`（生成该跑 yaml 的
        #       脚本）逐字「容器 vendor OPP 无 aclnnMhcPreSinkhorn 融合 kernel → mHC 走 unfused」，
        #       键值 `os.environ.get("FUSED_MHC") == "1"`（该跑未设 → False）；
        #     · `.claude/skills/real-machine-memory-sim/SKILL.md` §7 第 6 条同结论；
        #     · `specs/2026-07-01-unified-llm-modelspec-design.md:297` 锚点标题逐字
        #       「fused DSA + **unfused mHC** + MTP」。
        #   → 模型侧 `use_fused_mhc=False` 就是**对**的分支，本锚点 **ratio 0.891 一个字节不动**；
        #   仅把该位显式传参（见 `_dsv4_sim`）。仍 OOM-不安全（<1.0），band 仍排除历史翻转 1.088。
        Anchor("DSv4 mHC(x4)+MTP", "dsv4+mhc+mtp",
               _dsv4_sim(4, 1, use_fused_mhc=False), 21153.1, (0.86, 0.93),
               note="D2：OOM-不安全欠预测 0.891，不在 D1 覆盖内；band 防漂移/翻转，非 OOM-安全通过"),
    ]
