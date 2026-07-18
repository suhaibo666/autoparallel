"""StepTimeReport 门面（spec §6.3/6.4）：合成小段端到端（不依赖 mindformers），
验证组装管线、步收尾、MFU/HFU、fsdp_regather。"""
import pytest

from cost_eval.specs import OptimizerSpec
from cost_eval.timesim.ir import TimedOp, TimedSegment
from cost_eval.timesim.shard_rules import Degrees
from cost_eval.timesim.machine import synth_hw
from cost_eval.timesim.report import evaluate_step_time


def _layer(i):
    """一层 = 一个带权重的 GEMM + 一个 Norm（module="" 无 TP 语义，合成层）。"""
    mm = TimedOp(op_id="c#0", op_type="MatMul", phase="fwd",
                 in_shapes=((1024, 1, 512), (512, 512)), out_shape=(1024, 1, 512),
                 dtype="bf16", stream="device", src=f"l{i}.py:1")
    nm = TimedOp(op_id="c#1", op_type="Norm", phase="fwd",
                 in_shapes=((1024, 1, 512),), out_shape=(1024, 1, 512),
                 dtype="bf16", stream="device", src=f"l{i}.py:2", deps=("c#0",))
    return TimedSegment(f"layer_{i}.fwd", (mm, nm))


HW = synth_hw()


def test_report_basic_composition():
    r = evaluate_step_time([_layer(0), _layer(1)], Degrees(), HW, pp=1, m=2)
    assert r.t_step_us > 0
    assert r.t_step_us == pytest.approx(
        r.t_pipeline_us + r.t_opt_us + r.t_grad_sync_tail_us + r.fixed_step_us)
    assert r.uncalibrated is True
    assert r.provenance_mix == {"hit": 0.0, "model": 0.0, "theory": 1.0}
    assert 0.0 < r.mfu <= 1.0 and r.hfu == pytest.approx(r.mfu)   # 无重算 HFU==MFU
    assert r.bottleneck_ranking[0][1] >= r.bottleneck_ranking[-1][1]


def test_recompute_scissors():
    """L0④剪刀差：recompute=full → t_step↑、MFU↓、HFU>MFU（spec §6.4）。"""
    base = evaluate_step_time([_layer(0), _layer(1)], Degrees(), HW, pp=1, m=2)
    rc = evaluate_step_time([_layer(0), _layer(1)], Degrees(), HW, pp=1, m=2,
                            recompute="full")
    assert rc.t_step_us > base.t_step_us
    assert rc.mfu < base.mfu
    assert rc.hfu > rc.mfu


def test_pp2_uses_pipeline_and_p2p():
    r1 = evaluate_step_time([_layer(0), _layer(1)], Degrees(pp=2), HW, pp=2, m=4,
                            p2p_bytes=1024 * 512 * 2)
    r0 = evaluate_step_time([_layer(0), _layer(1)], Degrees(pp=2), HW, pp=2, m=4)
    assert r1.t_pipeline_us > r0.t_pipeline_us          # p2p 时延拉长
    assert r1.bubble_fraction > 0
    assert r1.bubble_fraction_closed_form == pytest.approx(1 / 5)


def test_opt_step_and_ddp_tail():
    opt = OptimizerSpec()
    r = evaluate_step_time([_layer(0), _layer(1)], Degrees(), HW, pp=1, m=1,
                           opt=opt, dp_replicate=2)
    assert r.t_opt_us > 0
    assert r.t_grad_sync_tail_us > 0                    # dp_replicate AR 尾（v1 保守暴露）


def test_fsdp_regather_and_grad_rs():
    """dp_shard>1：fwd 段头 AG（inject_fsdp）→ bwd 尾对偶 RS（grad sync 自动涌现）+
    reshard!=never 时 bwd 段头重 gather（交接要点3 裁决）。"""
    from cost_eval.timesim.frame_comm import inject_fsdp, fsdp_regather
    from cost_eval.timesim.bwd_rules import expand_bwd
    fwd = inject_fsdp(_layer(0), dp_shard=4)
    bwd = expand_bwd(fwd)
    bwd2 = fsdp_regather(bwd, fwd, dp_shard=4)
    comms = [o for o in bwd2.ops if o.op_type == "CommOp"]
    assert comms[0].comm.ctype == "all_gather" and comms[0].phase == "bwd"   # 重 gather 段头
    assert comms[-1].comm.ctype == "reduce_scatter"                          # grad RS 段尾
    assert comms[-1].comm.volume_bytes == comms[0].comm.volume_bytes * 4     # AG分片→RS全量
    # reshard="never" 语义由门面控制：不调 fsdp_regather 即无重 gather
    assert all(o.comm.ctype != "all_gather" for o in bwd.ops if o.op_type == "CommOp")


def test_recompute_comm_fsdp_no_double_gather():
    """Task 10 review Important：recompute=full + recomp_comm=True 时，expand_bwd 的重算前缀
    已重放 fwd 的 .fsdp_ag（即重 gather）；门面据此守卫 `not(recompute=='full' and recomp_comm)`
    **跳过** fsdp_regather——否则 comm_dp 上双 all_gather 重复计（方向保守但错）。"""
    from cost_eval.timesim.frame_comm import inject_fsdp, fsdp_regather
    from cost_eval.timesim.bwd_rules import expand_bwd

    def _dp_ag(seg):
        return [o for o in seg.ops if o.op_type == "CommOp"
                and o.stream == "comm_dp" and o.comm.ctype == "all_gather"]

    fwd = inject_fsdp(_layer(0), dp_shard=4)
    bwd = expand_bwd(fwd, recompute="full", recomp_comm=True)   # 前缀重放 .fsdp_ag
    assert len(_dp_ag(bwd)) == 1                                # 重算前缀已含 1 条重 gather
    assert len(_dp_ag(fsdp_regather(bwd, fwd, dp_shard=4))) == 2  # 反证：再 regather 则双 AG
    # 门面在该组合下跳过 fsdp_regather、跑通不 crash（守卫路径）
    r = evaluate_step_time([_layer(0), _layer(1)], Degrees(dp=4), HW, pp=1, m=1,
                           recompute="full", recomp_comm=True,
                           reshard_after_forward="default")
    assert r.t_step_us > 0
    # recomp_comm=False（默认）时前缀不重放 AG，门面仍需 fsdp_regather（守卫不触发）
    bwd_nc = expand_bwd(fwd, recompute="full", recomp_comm=False)
    assert len(_dp_ag(bwd_nc)) == 0                             # 前缀跳过 CommOp → 无重 gather
