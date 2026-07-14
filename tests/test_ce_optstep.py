"""① unfused CE 链 fat（无重算）+ ② optimizer-step 事件 —— 真机 profiler 标定（pp=2 8L 无重算 DSv3）。

真机（`analysis/realmachine/pp2_norecomp/`）：stage0 峰 = AdamW 更新 embedding 的瞬态（②，
7×883.8 MiB fp32），stage1 峰 = unfused CE 链 ~8 满 vocab fp32 共存（①）。估计器原欠计
stage0 23%、stage1 3×。两 fix 后 stage0→1.006、stage1→0.956，锚点不动。
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
    # fp32 layernorm 后:stage0 峰移到 bwd@4（无重算逐层反向 + fp32 norm 略超 optstep,1.05 过预测、OOM 安全）；
    # stage1 43899（0.962，较修前 0.956 提升）。
    r = _rep(8, B=2, pp=2, mode="None", mbs=2)
    s0, s1 = r.per_stage[0].peak_bytes / MiB, r.per_stage[1].peak_bytes / MiB
    assert 10200 <= s0 <= 11000, s0            # 无重算逐层反向（含 fp32 norm）
    assert 43000 <= s1 <= 47000, s1            # ① fat CE + ② + fp32 norm


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
    assert abs(r.per_stage[0].peak_bytes / MiB - 12437.9) < 0.05, r.per_stage[0].peak_bytes / MiB
