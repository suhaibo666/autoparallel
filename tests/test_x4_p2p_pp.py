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
    if overlap is not None:
        # 真字段（源 pipeline_parallel.py:396 pipeline_parallel_overlap_p2p，默认 False）——F8 订正后
        # 经公开构造字段可达（此前只能构造后动态挂属性）；用于验证 send buffer 双缓冲。
        kw["pipeline_parallel_overlap_p2p"] = overlap
    pc = ParallelConfig(**kw)
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
    """pp2 锚点（真机 10246/45655）逐字节对照本次基线（**11162.1/45991.4 MiB**）——
    加 P2P send buffer 后峰值不变（send 只叠 FWD 峰、远低于 BWD 峰）。

    2026-07-16：pre-FFN norm(ln2) 补建 → 无重算下各 MoE 层 ln2 fp32-cast 常驻 → 基线
    10826.0/45767.3 → 11162.1/45991.4。stage0 峰在 bwd@4（无重算逐层反向、forward 激活常驻），
    对真机 10246 过预测 1.089（**OOM-安全**）——此过预测属「无重算全层 act_live 于 BWD 峰共存」的
    整体保守（非 ln2 缺陷，ln2 是正确结构；与 P2-01/无重算聚合过预测同族），stage1 1.007 精确。
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
    # 2026-07-29 二次重钉：11162.1/45991.4 → 10458.1/45611.4（RMSNorm 不 cast）。
    #   ⚠ s0 对真机 10246 的比值 1.089 → **1.021**（仍 OOM-安全，但余量大幅收窄）；
    #   s1 1.007 → 0.999（**刚跌破 1.0，转 OOM-不安全侧** —— 如实记，见下方守卫的注释）。
    # 2026-07-30 三次重钉（**word-embedding 反向 kernel workspace 实测入账**，
    #   `docs/head_workspace_2026-07-30.md`）：s0 10458.1 → **10573.0**，且**峰值事件由
    #   `bwd@4` 易主为 `bwd@0`（embedding 反向）**。比值 1.021 → **1.032**。
    #   ⚠ 这是全库**唯一**被本项抬动的记分卡锚点，因为只有它的 `bwd@0` 本来就贴着峰
    #   （差 952.9 MiB < 本项 1067.753 MiB）。**这一笔正是本 config 自己的真机 profiler
    #   量到的**：`analysis/realmachine/pp2_norecomp/op_816362.csv` 里 `GatherDGradV2` 的
    #   瞬态块 `Size(KB)=1093379.0` = 1119620096 B = 1067.753 MiB，逐字节对上（该文件正是
    #   这条锚点 real=10246.0 的同一次采集）。s1 逐 MiB 不变（末 stage 无 embedding 层）。
    assert abs(s0.peak_bytes / MiB - 10573.0) < 0.1
    # 2026-07-30 重钉：s1 45611.4 → **47707.4**（`lm_head` 反向 kernel workspace 实测入账；
    #   该 config B*S=8192、H=1792 → 律给 (2*129280+4*1792)*8192 + 20 MiB + 1024 = 2096.0 MiB）。
    #   ⚠ 这一笔**正是这条锚点自己那次采集的 profiler 量到的**：
    #   `analysis/realmachine/pp2_norecomp/op_816365.csv`（持有 lm_head 的那个 rank）里
    #   `MatMulExt` 的瞬态块 `2168457216 B` == wgrad 律 `(2*vocab+2*H)*8192 + 20 MiB + 2048`
    #   **逐字节吻合**；同文件里 dgrad 那一档是 `4315940352 B`（比本律多一份 `2*vocab*B*S`
    #   的 operand 拷贝 —— 那是**另一个 build 的 kernel 选择**，今天单卡同 shape 量到的是
    #   本律的值，见 docs/head_loss_bwd_workspace_2026-07-30.md 2.3/7.2）。s0 逐 MiB 不变。
    assert abs(s1.peak_bytes / MiB - 47707.4) < 0.1
    assert s0.peak_event.startswith("bwd") and s1.peak_event.startswith("bwd")
    # ── 方向门（2026-07-29 二次重钉后**分两档**，如实记录 s1 的翻转）────────────────────
    # s0 仍 OOM-安全（预测 ≥ 真机 10246.0，比值 1.032；原 1.021，本轮因实测 workspace 入账回升）。
    assert s0.peak_bytes / MiB >= 10246.0
    # s1 **已翻成 OOM-不安全**：45611.4 vs 真机 45655.0 = 0.9990，欠 43.6 MiB（0.10%）。
    #   成因：`FusedRMSNorm` 不 cast（`layer_norm.py:151-155`）修掉了一处**真实的过读**，
    #   而它此前正好在掩盖别处的欠读 —— 这是本项目反复出现的「相消误差」故事的又一例。
    #   **按纪律不调参掩盖**：不重标 margin、不放宽 census。真正的补法是 kernel workspace 项，
    #   需真机 profiler 明细（见 docs/census_fix_mhc_rmsnorm_2026-07-29.md §残差）。
    #   本门把翻转**钉死**：既守它没继续恶化（下界），也守它没被偷偷"调回去"（上界）。
    # 2026-07-30：s1 由 0.9990（OOM-不安全）**翻回 OOM-安全侧 1.0450** —— 翻转来自
    #   真机实测项入账，**不是**调参（margin / census 一个字节没动）。带随之上移，
    #   两侧仍都守：<lo 说明本项被削或又出现新欠读；>hi 说明过读继续膨胀。
    _s1_ratio = s1.peak_bytes / MiB / 45655.0
    assert 1.040 <= _s1_ratio <= 1.050, (
        f"pp2 stage1 sim/real={_s1_ratio:.4f} 越出已记录的过读带 [1.040, 1.050]；"
        f"sim={s1.peak_bytes / MiB:.1f} real=45655.0。>1.0 说明有人把它调回 OOM-安全侧——"
        f"请核对是不是又加了拟合常数；<0.995 说明欠读继续恶化，须查明。")
