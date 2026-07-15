"""P1-15 闭环：PP stage 间 P2P send/recv 激活 buffer（Task A）。

真机 PP（1F1B）：每个 stage 边界 send 本 stage 输出激活 [S,B,H] 给下 stage、recv 上 stage
输出。建模约定（源忠实 + OOM 安全）：
  - **recv 侧**（非首 stage）= 该 stage 首层输入 [S,B,H]，**已隐含在 act_live 首层 pin**
    （首层 saves 的 checkpoint_input 就是 recv 回来的那块）→ 不双算。
  - **send 侧**（非末 stage）= 本 stage 输出激活 [S,B,H]，send 通信期间驻留（1 份；
    `pipeline_parallel_overlap_p2p` 时 2 份双缓冲）→ **新桶 p2p_buf**，仅 pp>1 生效。
  - pp=1 全惰性：p2p_buf 恒 0，DSv3/DSv4 单 stage 锚点逐字节不动。

关键不变式（OOM 安全 + 锚点保护）：p2p_buf 仅在 FWD 事件驻留、BWD/optstep 清零。pp2 两 stage
的峰都在 BWD（stage0 bwd@4 / stage1 loss bwd@9），send buffer 只叠在 FWD 峰（远低于 BWD 峰）
→ 锚点峰值/峰事件逐字节不变（闭合到「时间线粒度」而非改峰）。
"""
from cost_eval.mem_timeline import MemTimeline
from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.static_mem import StaticMem
from cost_eval.structure_mem import estimate_structure_memory
from validate_dsv3 import build_dsv3_spec

MiB, GiB = 2 ** 20, 2 ** 30
BLK = 512


def _sim(pp, *, mbs=2, B=2, dp=1, record=True, overlap=None):
    spec, d, fl = build_dsv3_spec(8)
    d.B = B
    kw = dict(dp_shard=dp, cp=1, tp=1, ep=1, pp=pp, sequence_parallel=True,
              num_microbatches=mbs)
    pc = ParallelConfig(**kw)
    if overlap is not None:
        # 运行时开关（源 pipeline_parallel.py:396 pipeline_parallel_overlap_p2p，默认 False）——
        # 评估器按 getattr 读取，测试直接注入以验证双缓冲。
        pc.pipeline_parallel_overlap_p2p = overlap
    pm = ParallelModel(pc, spec.dims.n_layers, world_size=dp * pp)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(params_fp32=True), pm, False,
                                     alloc_block_bytes=BLK)
    r = MemTimeline().simulate(
        g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
        framework_reserve=0, max_device_memory=64 * GiB,
        grad_dtype_bytes=4, record_timeline=record, alloc_block_bytes=BLK)
    return r, g


def _last_layer_boundary_bytes(g, stage):
    """该 stage 最后一层（最大 layer_id）的层入口 [S,B,H] 字节 = 其输出激活的 send 量级。"""
    layers = g.stages[stage]
    last = max(layers, key=lambda l: l.layer_id)
    return estimate_structure_memory(last.ops, alloc_block_bytes=BLK).checkpoint_input


# ---------------------------------------------------------------------------
# pp=1：完全惰性，无 P2P buffer（单 stage 锚点保护）
# ---------------------------------------------------------------------------

def test_pp1_has_no_p2p_buffer_anywhere():
    r, g = _sim(pp=1, mbs=1)
    for s in r[0].timeline:
        assert s.breakdown.p2p_buf == 0, (s.event, s.breakdown.p2p_buf)


# ---------------------------------------------------------------------------
# pp>1：非末 stage 有 send buffer；末 stage 无（recv 已在 act_live）
# ---------------------------------------------------------------------------

def test_pp2_nonlast_stage_send_buffer_equals_hidden_boundary():
    r, g = _sim(pp=2)
    send = _last_layer_boundary_bytes(g, stage=0)
    assert send > 0
    fwd_ends = [s for s in r[0].timeline if s.event.startswith("fwd_end")]
    assert fwd_ends
    # 每个 fwd_end 都驻留 1 份 send buffer（默认 overlap=False）。
    for s in fwd_ends:
        assert s.breakdown.p2p_buf == send, (s.event, s.breakdown.p2p_buf, send)


def test_pp2_last_stage_has_no_send_buffer():
    r, g = _sim(pp=2)
    # 末 stage（stage1）：无 send；recv 已隐含在 act_live 首层 pin → p2p_buf 恒 0。
    for s in r[1].timeline:
        assert s.breakdown.p2p_buf == 0, (s.event, s.breakdown.p2p_buf)


def test_p2p_buffer_cleared_during_backward_and_optstep():
    r, g = _sim(pp=2)
    for s in r[0].timeline:
        if s.event.startswith("bwd") or s.event == "optstep":
            assert s.breakdown.p2p_buf == 0, (s.event, s.breakdown.p2p_buf)


# ---------------------------------------------------------------------------
# overlap → 双缓冲（2 份）
# ---------------------------------------------------------------------------

def test_overlap_p2p_doubles_the_send_buffer():
    r1, g1 = _sim(pp=2, overlap=False)
    r2, g2 = _sim(pp=2, overlap=True)
    send = _last_layer_boundary_bytes(g1, stage=0)
    f1 = next(s for s in r1[0].timeline if s.event.startswith("fwd_end"))
    f2 = next(s for s in r2[0].timeline if s.event.startswith("fwd_end"))
    assert f1.breakdown.p2p_buf == send
    assert f2.breakdown.p2p_buf == 2 * send


# ---------------------------------------------------------------------------
# 锚点保护：pp2 两 stage 峰仍在 BWD，send buffer 不移峰、峰值 p2p_buf==0
# ---------------------------------------------------------------------------

def test_pp2_anchors_peak_still_in_backward_and_p2p_zero_at_peak():
    r, g = _sim(pp=2)
    s0, s1 = r[0], r[1]
    assert s0.peak_event.startswith("bwd"), s0.peak_event
    assert s1.peak_event.startswith("bwd"), s1.peak_event
    # 峰值时刻 p2p_buf 必须为 0（否则 send buffer 推离了已很接近的 pp2 锚点）。
    assert s0.breakdown.p2p_buf == 0
    assert s1.breakdown.p2p_buf == 0


def test_pp2_stage_peaks_byte_identical_to_recorded_anchor():
    """pp2 锚点（真机 10246/45655）逐字节对照本次基线（10826.0/45767.3 MiB）——
    加 P2P send buffer 后**峰值不变**（send 只叠 FWD 峰、远低于 BWD 峰）。

    走完整 Evaluator（与 sim_vs_real_report 同路径）取真锚点值。"""
    from cost_eval.report import Evaluator
    spec, d, fl = build_dsv3_spec(8)
    d.B = 2
    pc = ParallelConfig(dp_shard=1, cp=1, tp=1, ep=1, pp=2, sequence_parallel=True,
                        num_microbatches=2)
    r = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True),
                  HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0),
                  RecomputeSpec("None"), SwapSpec()).evaluate()
    s0, s1 = r.per_stage[0], r.per_stage[1]
    assert abs(s0.peak_bytes / MiB - 10826.0) < 0.1
    assert abs(s1.peak_bytes / MiB - 45767.3) < 0.1
    assert s0.peak_event.startswith("bwd") and s1.peak_event.startswith("bwd")
    # 仍 OOM 安全（预测 ≥ 真机）
    assert s0.peak_bytes / MiB >= 10246.0
    assert s1.peak_bytes / MiB >= 45655.0
