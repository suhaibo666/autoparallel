"""① unfused CE 链 fat（无重算）+ ② optimizer-step 事件 —— 真机 profiler 标定（pp=2 8L 无重算 DSv3）。

真机（`analysis/realmachine/pp2_norecomp/`）：stage0 峰 = AdamW 更新 embedding 的瞬态（②），
stage1 峰 = unfused CE 链多张满 vocab 平面共存（①）。

**2026-07-30 订正上一行原先的「~8 满 vocab fp32 共存」**（`docs/k_ce_recalibration_2026-07-30.md`）：
从 `op_816365.csv` 逐块数出来，high-water 那一刻在世的是 **5 张 fp32 + 5 张 bf16**
（= 7.5 张 fp32-等效），产出者 `{Log×3, Neg, ScatterAddExt}` / `{Copy×4, ZerosLikeExt}`
—— 不是 8 张 fp32。`K_CE` 因此由 8 改 **7**（模型侧总量 = `K_CE + 0.5`）。

P2-08 文档订正（2026-07-15）：stage0 的 optstep 瞬态**当前建模为 K_OPT=4 份 fp32 + 累计梯度桶
grad_accum**（P0-01，2026-07-14：旧「7× / 6×883.8」是含累计梯度的混合常数，已拆分——K_OPT
6→4，梯度独立进 grad_accum）。stage0→0.997、stage1→1.001（grad_accum 修后），锚点见各测试断言。
"""
from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

GiB = 2 ** 30
MiB = 2 ** 20


def _rep(N, *, B=1, pp=1, mode="None", ce_fused=False, mbs=1, dp=1):
    spec, d, fl = build_dsv3_spec(N)
    d.B = B
    d.cross_entropy_fused = ce_fused
    pc = ParallelConfig(dp_shard=dp, cp=1, tp=1, pp=pp, sequence_parallel=(dp > 1),
                        num_microbatches=mbs)
    rc = RecomputeSpec(mode="full", full_layers=fl) if mode == "full" else RecomputeSpec(mode="None")
    ev = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                   HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0), rc, SwapSpec())
    return ev.evaluate()


def test_pp2_8L_norecompute_matches_real_machine():
    # 真机：stage0=10246.2, stage1=45655.5（dp=1 pp=2 global_batch=2 → B=2）
    # 2026-07-16：pre-FFN norm(ln2) 补建 → 无重算下各 MoE 层 ln2 fp32-cast 常驻 → stage0 11162.1
    #   (真机 10246 → 1.089 过预测、**OOM-安全**；bwd@4 无重算逐层反向峰的整体保守，非 ln2 缺陷)、
    #   stage1 45991.4（1.007 精确）。
    r = _rep(8, B=2, pp=2, mode="None", mbs=2)
    s0, s1 = r.per_stage[0].peak_bytes / MiB, r.per_stage[1].peak_bytes / MiB
    assert 10200 <= s0 <= 11300, s0            # 无重算逐层反向（含 ln2/fp32 norm 常驻，OOM-安全过预测）
    # 2026-07-30：`lm_head` 反向 kernel workspace 入账 → s1 45611.4 → 47707.4
    #   （B*S=8192 → 律给 2096.0 MiB）。真机 45655.5 → 0.999 → **1.045**（过读 = OOM 安全）。
    # 2026-07-30 同日 `K_CE` 重标定（8 → 7，`docs/k_ce_recalibration_2026-07-30.md`）：
    #   s1 47707.4 → **43667.4**（−4040.0 = 一张满 vocab fp32 平面 4·S·B·vocab @ B*S=8192）。
    #   真机 45655.0 → 1.045 → **0.9565**（**翻回 OOM-不安全，欠 1987.6 MiB —— 如实记**）。
    #   7 不是拟合值：本 config 自己那次采集（`analysis/realmachine/pp2_norecomp/op_816365.csv`，
    #   其 pool high-water = 45655.46 MiB 逐 MiB 命中本锚点 real）在 high-water 那一刻在世
    #   **5 张 fp32 + 5 张 bf16** 满 vocab 平面 = 7.5 fp32-等效；模型侧总量是 `K_CE + 0.5`
    #   （K_CE−1 张瞬态 + 1 张 saved logsm fp32 + 0.5 张 saved logits bf16）→ 解出 7。
    #   **不放宽方向**：带两侧都守，>hi 说明有人把 K_CE 调回 8 去凑安全侧。
    assert 43400 <= s1 <= 44000, s1            # fat CE(K_CE=7) + fp32 norm + ln2 + head-ws


def test_optstep_event_present_stage0():
    # ② opt-step 事件仍存在（在 timeline 里）——fp32 norm 后无重算逐层反向略超它、峰值事件变 bwd@N,
    #   但 optstep 事件本身仍被记录（timeline）。校验 optstep 事件存在且非零。
    from validate_dsv3 import build_dsv3_spec
    from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
    from cost_eval.report import Evaluator
    spec, d, fl = build_dsv3_spec(8); d.B = 2
    pc = ParallelConfig(dp_shard=1, cp=1, tp=1, pp=2, sequence_parallel=False, num_microbatches=2)
    r = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                  HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0),
                  RecomputeSpec(mode="None"), SwapSpec()).evaluate(record_timeline=True)
    opt_samples = [s for s in r.per_stage[0].timeline if s.event == "optstep"]
    assert opt_samples and opt_samples[0].breakdown.optstep > 0


def test_ce_fat_only_no_recompute_and_unfused():
    # ① fat 仅在 无重算 + unfused CE。full 重算 或 fused CE → lean（stage1 远小于 fat）。
    fat = _rep(8, B=2, pp=2, mode="None", ce_fused=False, mbs=2).per_stage[1].peak_bytes / MiB
    fused = _rep(8, B=2, pp=2, mode="None", ce_fused=True, mbs=2).per_stage[1].peak_bytes / MiB
    assert fat > fused + 15000, (fat, fused)   # fat 比 lean 多 ~7 份满 vocab fp32


def test_dsv3_4L_full_recompute_anchor_unchanged():
    # DSv3 4L full 重算：① 不触发（非无重算）、② 不上峰（opt-step < loss 反向）→ 12437.9 逐字节
    r = _rep(4, B=1, pp=1, mode="full", dp=2, mbs=1)
    # 2026-07-29 二次重钉：12437.9 → 12423.9（RMSNorm 不 cast，见 test_dsv3_golden 同注）。
    # 2026-07-30 三次重钉：12423.9 → 13481.9（`lm_head` 反向 kernel workspace 实测入账 +1058.0；
    #   docs/head_loss_bwd_workspace_2026-07-30.md）。① / ② 两条结论**不变**。
    assert abs(r.per_stage[0].peak_bytes / MiB - 13481.9) < 0.05, r.per_stage[0].peak_bytes / MiB
