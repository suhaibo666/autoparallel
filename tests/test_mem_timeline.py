"""Task 11 + 12 tests for M6 mem_timeline (build_1f1b + simulate)."""
from cost_eval.mem_timeline import (
    build_1f1b, build_interleaved_1f1b, interleaved_warmup,
    Event, MemTimeline, _layer_saves_bytes)
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.layers.mla import build_mla_dense_decoder
from cost_eval.specs import ParallelConfig, RecomputeSpec, SwapSpec, OptimizerSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.static_mem import StaticMem
from cost_eval.structure_mem import estimate_structure_memory


# ---------------------------------------------------------------------------
# Task 11: 1F1B 调度构建
# ---------------------------------------------------------------------------

def test_1f1b_warmup_depth():
    # PP=4, stage 0: warmup = PP-1-0 = 3 个前向先行
    evs = build_1f1b(stage=0, pp=4, m=8)
    fwd_prefix = []
    for e in evs:
        if e.kind == "FWD":
            fwd_prefix.append(e)
        else:
            break
    assert len(fwd_prefix) == 4    # warmup(3) + steady 第一个 F = 4 个 F 才出现 B


def test_event_counts_balanced():
    evs = build_1f1b(stage=1, pp=4, m=8)
    assert sum(e.kind == "FWD" for e in evs) == 8
    assert sum(e.kind == "BWD" for e in evs) == 8


# ---------------------------------------------------------------------------
# Task 12: simulate — 峰值落点迁移 + OOM 标志
# ---------------------------------------------------------------------------

D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=4)


def _setup(pp=1):
    spec = ModelSpec("toy", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(pp=pp), n_layers=4, world_size=pp)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(), pm, False)
    return g, pm, persistent


def test_full_recompute_lowers_peak_and_moves_event():
    g, pm, persistent = _setup(pp=1)
    mt = MemTimeline()
    none = mt.simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                       framework_reserve=0, max_device_memory=10**12)
    full = mt.simulate(g, RecomputeSpec("full", {0, 1, 2, 3}), SwapSpec(), pm, persistent,
                       framework_reserve=0, max_device_memory=10**12)
    # full 重算降低峰值（act_live 大降 > 反向重物化一层）；none 峰仍在反向（FSDP gather+grad 共存）。
    # P0-01（2026-07-14）：grad_accum 常驻至 optstep → full（激活峰被压低）的全局峰移到 optstep
    # （persistent + K_OPT 瞬态 + 累计梯度共存）——真机语义（optimizer 前全部 reduced grad 驻留）。
    assert full[0].peak_bytes < none[0].peak_bytes
    assert none[0].peak_event.startswith("bwd")
    assert full[0].peak_event == "optstep"
    assert full[0].breakdown.grad_accum > 0            # 累计梯度与 optstep 瞬态共存


def test_oom_flag():
    g, pm, persistent = _setup(pp=1)
    r = MemTimeline().simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                               framework_reserve=0, max_device_memory=1)
    assert r[0].oom is True


def _peak_layer_sm(g, peak_event):
    lid = int(peak_event.split("@")[1])
    ops = next(l for l in g.stages[0] if l.layer_id == lid).ops
    return estimate_structure_memory(ops)


def test_no_recompute_adds_bwd_working_set():
    """§8.5②：无重算层反向叠加 `bwd_working_set`（= 该层 forward 峰值工作集的反向镜像，
    激活梯度 dL/dact），扣掉已显式建模的 `bwd_scratch` 部分。transformer 为主的配置（toy 全
    dense、无 loss 层）→ 峰值落在某 transformer 反向，bwd_working_set > 0 且入峰。"""
    g, pm, persistent = _setup(pp=1)
    r = MemTimeline().simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                               framework_reserve=0, max_device_memory=10**12)
    b = r[0].breakdown
    assert r[0].peak_event.startswith("bwd")
    assert b.bwd_working_set > 0                       # 无重算反向工作集入峰
    sm = _peak_layer_sm(g, r[0].peak_event)
    assert b.bwd_working_set == max(0, sm.forward_max_live - sm.bwd_scratch)
    # 逐桶之和 == 峰值（新桶已并入 total，无遗漏/重复；P0-01 后含 grad_accum/kept_frag）
    assert (b.persistent + b.act_live + b.gather_buf + b.grad_buf + b.recomp_scratch
            + b.bwd_scratch + b.bwd_working_set + b.swap_buf + b.workspace
            + b.framework + b.kept_frag + b.grad_accum) == r[0].peak_bytes


def test_full_recompute_scratch_is_forward_max_live():
    """§8.5②①②：重算层反向重物化 = 重跑 forward 的 **max-live**（非 saves 之和）。
    `recomp_scratch = max(0, forward_max_live − checkpoint_input)`（层入口已 pin 进 act_live）。"""
    g, pm, persistent = _setup(pp=1)
    full = MemTimeline().simulate(g, RecomputeSpec("full", {0, 1, 2, 3}), SwapSpec(), pm,
                                  persistent, framework_reserve=0, max_device_memory=10**12,
                                  record_timeline=True)
    # P0-01 后 full 的全局峰移到 optstep（grad_accum 常驻）——重算机理断言改从 timeline 取
    # 首个反向事件（bwd@3，重算层）验证 recomp_scratch 语义。
    bwd3 = next(s for s in full[0].timeline if s.event == "bwd@3")
    b = bwd3.breakdown
    assert b.recomp_scratch > 0                        # 重算 transformer 层反向重物化
    sm = _peak_layer_sm(g, "bwd@3")
    assert b.recomp_scratch == max(0, sm.forward_max_live - sm.checkpoint_input)
    # forward_max_live ≠ activation_saves（守「取峰值工作集而非 saves 之和」的实质改动）
    assert sm.forward_max_live != sm.activation_saves


# ---------------------------------------------------------------------------
# Task [SELECTIVE]: 选择性重算（mode=select，按 op/模块）
# ---------------------------------------------------------------------------

def _sim(recompute):
    g, pm, persistent = _setup(pp=1)
    return MemTimeline().simulate(
        g, recompute, SwapSpec(), pm, persistent,
        framework_reserve=0, max_device_memory=10**12)[0]


def test_none_full_byte_identical_anchor():
    """None/full 峰值字节锚点（守卫两态；toy dense 4 层）。

    P0-01（2026-07-14）：full 锚 2768896 → 3211264——峰移到 optstep
    （= persistent 2293760 + K_OPT·max_w·4 262144 + grad_accum 655360，
    累计梯度常驻至 optimizer 的真机语义）。
    P1-09（2026-07-14）：none 锚 3641344 → 3768320——fa_stats（softmax max/sum
    [2,B,N,S,8] fp32）驻留 fwd→bwd 取代 lse：+62·B·n_heads·S = +31744 B/层 × 4 层
    = +126976 B（无重算层 act_live 净增；full 层 saves 被丢弃重物化 → full 锚不受此项影响）。
    P1-01（2026-07-14）：+norm gamma（ln1_g/ln2_g (H,) fp32 ×4 层）→ persistent/grad_accum
    小幅增：none 3768320→3777024、full（optstep 峰）3211264→3220480。"""
    none = _sim(RecomputeSpec("None"))
    full = _sim(RecomputeSpec("full", {0, 1, 2, 3}))
    # 2026-07-23（std census）：none 3777024 → 5284352——标准路径 pynative 全保留 census 补齐
    # （split/rope-fp32/TND/mask/ctx/残差保留,116 stdL1/2/4/8 逐层差分 728.0(MHA) 实测背书,
    # attention.py build_gqa_attn_ops ①-⑥）→ 无重算 act_live 每层净增;full 锚不动（saves 被
    # 丢弃重物化,census 追加成员默认无 pin_under_recompute → 全重算行为不变）。
    assert none.peak_bytes == 5284352
    assert full.peak_bytes == 3220480


def test_select_core_attn_peak_between_none_and_full():
    """选中 core-attn（flash，Megatron 默认）：峰值严格介于无重算与全重算之间，
    act_live 较无重算下降，recomp_scratch 入峰 > 0，峰仍落反向。"""
    none = _sim(RecomputeSpec("None"))
    full = _sim(RecomputeSpec("full", {0, 1, 2, 3}))
    sel = _sim(RecomputeSpec("select", select_ops={lid: {"flash"} for lid in range(4)}))
    assert full.peak_bytes < sel.peak_bytes < none.peak_bytes
    assert sel.breakdown.act_live < none.breakdown.act_live
    assert sel.breakdown.recomp_scratch > 0
    assert sel.peak_event.startswith("bwd")


def test_select_monotonic_more_ops_lower_peak():
    """重算的 op 越多，激活峰越低（内存↔重算旋钮）：
    None ≥ select(flash) ≥ select(整个 attention 模块) ≥ full，严格单调。"""
    none = _sim(RecomputeSpec("None"))
    full = _sim(RecomputeSpec("full", {0, 1, 2, 3}))
    flash_only = _sim(RecomputeSpec(
        "select", select_ops={lid: {"flash"} for lid in range(4)}))
    attn_mod = _sim(RecomputeSpec(
        "select",
        select_ops={lid: {"ln1", "qkv", "rope", "flash", "o_proj", "add1"}
                    for lid in range(4)}))
    assert (none.peak_bytes > flash_only.peak_bytes
            > attn_mod.peak_bytes > full.peak_bytes)


def test_select_all_ops_equals_full():
    """select 全部 op == full 重算（两条路径一致）：峰值与逐桶 breakdown 皆相等。"""
    g, pm, persistent = _setup(pp=1)
    all_names = {op.name for op in g.stages[0][0].ops}
    mt = MemTimeline()
    full = mt.simulate(g, RecomputeSpec("full", {0, 1, 2, 3}), SwapSpec(), pm,
                       persistent, framework_reserve=0, max_device_memory=10**12)[0]
    sel_all = mt.simulate(
        g, RecomputeSpec("select", select_ops={lid: set(all_names) for lid in range(4)}),
        SwapSpec(), pm, persistent, framework_reserve=0, max_device_memory=10**12)[0]
    assert sel_all.peak_bytes == full.peak_bytes
    assert sel_all.breakdown.act_live == full.breakdown.act_live
    assert sel_all.breakdown.recomp_scratch == full.breakdown.recomp_scratch
    assert sel_all.breakdown.bwd_working_set == full.breakdown.bwd_working_set


def test_select_none_selectors_equals_no_recompute():
    """select 但选择器命中空（或未配该层）→ 逐字节复现无重算峰值。"""
    none = _sim(RecomputeSpec("None"))
    sel_empty = _sim(RecomputeSpec("select", select_ops={0: set()}))
    assert sel_empty.peak_bytes == none.peak_bytes


# ---------------------------------------------------------------------------
# Task 1 [C1]: 激活 saves 按张量名去重（attn 被 flash + o_proj 双 save）
# ---------------------------------------------------------------------------

def _resolve_single_layer(decoder):
    """把一个 LayerSpec 解析成 ResolvedLayer（单卡、单层）。"""
    spec = ModelSpec("toy", D, ["x"], {"x": decoder})
    pm = ParallelModel(ParallelConfig(), n_layers=1, world_size=1)
    return ShapeEval().resolve(spec, pm).stages[0][0]


def _naive_saves_bytes(layer):
    """去重前的裸求和（旧 _layer_saves_bytes 行为，用作对照）。"""
    return sum(s.local_numel * s.dtype_bytes for op in layer.ops for s in op.saves)


def test_gqa_layer_dedups_attn_save_once():
    """GQA：attn 被 flash(saves=[qkv,attn,lse]) 与 o_proj(saves=[attn]) 双 save，
    去重后应恰好比裸求和少一个 attn 张量的字节。"""
    layer = _resolve_single_layer(build_dense_decoder(D))
    naive = _naive_saves_bytes(layer)
    dedup = _layer_saves_bytes(layer)
    attn = next(s for op in layer.ops for s in op.saves if s.name == "attn")
    attn_bytes = attn.local_numel * attn.dtype_bytes
    assert attn_bytes > 0
    assert dedup == naive - attn_bytes


def test_mla_layer_dedups_attn_save_once():
    """MLA：attn 被 flash 与 o_proj 双 save（attention.py:149/153），去重后少一个 attn。"""
    Dmla = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100,
                    n_layers=1, q_lora_rank=32, kv_lora_rank=16,
                    qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16)
    spec = ModelSpec("toy", Dmla, ["x"], {"x": build_mla_dense_decoder(Dmla)})
    pm = ParallelModel(ParallelConfig(), n_layers=1, world_size=1)
    layer = ShapeEval().resolve(spec, pm).stages[0][0]
    naive = _naive_saves_bytes(layer)
    dedup = _layer_saves_bytes(layer)
    attn = next(s for op in layer.ops for s in op.saves if s.name == "attn")
    attn_bytes = attn.local_numel * attn.dtype_bytes
    assert attn_bytes > 0
    assert dedup == naive - attn_bytes


# ---------------------------------------------------------------------------
# Task [VPP]: 交错式 1F1B（interleaved-1F1B / 虚拟流水 VPP）—— 更深 warmup
#
# 锚定公式（Megatron `schedules.py:877-878`，默认
# `microbatch_group_size_per_vp_stage = pp`，`model_parallel_config.py:519-520`）：
#     num_warmup = (pp - rank - 1)*2 + (V - 1)*pp        （clamp 到本模型的 m）
# 对照 plain-1F1B（`schedules.py:870`）：num_warmup = pp - rank - 1。
# mindformers pynative 同构（round-robin chunk：`pipeline_parallel.py:257-258`；
# V=1→plain、V>1→interleaved：`:275-276`/`:360-371`）。
# ---------------------------------------------------------------------------

def _leading_fwd(evs):
    """事件序列开头连续 FWD 的个数（第一个 BWD 之前）。plain/interleaved 里
    = warmup + 1（未被 m 截断时：warmup 个热身 FWD + steady 第一个 FWD）。"""
    n = 0
    for e in evs:
        if e.kind == "FWD":
            n += 1
        else:
            break
    return n


def test_interleaved_v1_byte_identical_to_1f1b():
    """V=1 → 逐事件复现 plain build_1f1b（byte-identical 门；守卫 anchors/golden）。"""
    for pp in (1, 2, 4):
        for stage in range(pp):
            for m in (1, 4, 8):
                assert build_interleaved_1f1b(stage, pp, m, 1) == build_1f1b(stage, pp, m)


def test_interleaved_warmup_formula_matches_megatron():
    """warmup 计数命中 Megatron 交错式表达式（手算字面值，非自指）。"""
    assert interleaved_warmup(0, 4, 16, 2) == 2 * 3 + 1 * 4      # 10
    assert interleaved_warmup(1, 4, 16, 2) == 2 * 2 + 1 * 4      # 8
    assert interleaved_warmup(3, 4, 16, 2) == 2 * 0 + 1 * 4      # 4（末 stage 仍热身）
    assert interleaved_warmup(0, 4, 30, 3) == 2 * 3 + 2 * 4      # 14
    assert interleaved_warmup(0, 2, 8, 2) == 2 * 1 + 1 * 2       # 4
    # V=1 退回 plain-1F1B 的 warmup（**不是**交错式在 V=1 处的 2×(pp-1-stage)）
    assert interleaved_warmup(0, 4, 16, 1) == min(4 - 1 - 0, 16)  # 3


def test_interleaved_warmup_deeper_than_plain():
    """任意 stage：交错式 warmup 严格深于 plain-1F1B（更多在飞 microbatch）。"""
    for stage in range(4):
        plain = min(4 - 1 - stage, 16)
        assert interleaved_warmup(stage, 4, 16, 2) > plain


def test_interleaved_warmup_monotonic_in_v():
    """warmup 计数随 V 单调增（V 越多 → warmup 越深）。m 足够大避免截断。"""
    prev = interleaved_warmup(0, 4, 100, 1)
    for v in (2, 3, 4, 5):
        w = interleaved_warmup(0, 4, 100, v)
        assert w > prev
        prev = w


def test_interleaved_warmup_clamped_to_m():
    """raw 2*3+3*4=18 > m=8 → 截断到 m=8；事件计数仍平衡（m F / m B）。"""
    assert interleaved_warmup(0, 4, 8, 4) == 8
    evs = build_interleaved_1f1b(0, 4, 8, 4)
    assert sum(e.kind == "FWD" for e in evs) == 8
    assert sum(e.kind == "BWD" for e in evs) == 8


def test_interleaved_event_counts_balanced():
    """V>1：总 FWD == m、总 BWD == m（pinned 全 pop，无泄漏）。"""
    evs = build_interleaved_1f1b(0, 4, 16, 2)
    assert sum(e.kind == "FWD" for e in evs) == 16
    assert sum(e.kind == "BWD" for e in evs) == 16


def test_interleaved_leading_fwd_is_warmup_plus_one():
    """把 warmup 公式与事件序列扣起来：未截断时开头 FWD 连跑 = warmup + 1。"""
    w = interleaved_warmup(0, 4, 16, 2)          # 10 < 16
    assert _leading_fwd(build_interleaved_1f1b(0, 4, 16, 2)) == w + 1


# ── simulate 接线：V 来自 ParallelConfig.interleave ─────────────────────────

def _sim_interleave(pp, m, v, stage=0, recompute=None, n_layers=4, record_timeline=False):
    spec = ModelSpec("toy", D, ["dense"] * n_layers, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(pp=pp, num_microbatches=m, interleave=v),
                       n_layers=n_layers, world_size=pp)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(), pm, False)
    rc = recompute or RecomputeSpec("None")
    return MemTimeline().simulate(g, rc, SwapSpec(), pm, persistent,
                                  framework_reserve=0, max_device_memory=10 ** 12,
                                  record_timeline=record_timeline)[stage]


def _max_act_live(sp):
    """时间线上 act_live 的最大值（record_timeline=True 时可用）。"""
    return max(t.breakdown.act_live for t in sp.timeline)


def test_interleave1_simulate_reproduces_default():
    """simulate 接线 interleave=1 与「不设该字段（默认 1）」逐字节一致（回归门）。"""
    a = _sim_interleave(4, 8, 1, stage=0)
    spec = ModelSpec("toy", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(pp=4, num_microbatches=8), n_layers=4, world_size=4)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(), pm, False)
    b = MemTimeline().simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                               framework_reserve=0, max_device_memory=10 ** 12)[0]
    assert a.peak_bytes == b.peak_bytes
    assert a.breakdown == b.breakdown


def test_interleave_deeper_warmup_raises_peak_stage0():
    """warmup-受限的 stage 0：V=2 比 plain 更深 warmup → act_live/peak 更高，但（D-4 后）**远低于
    ~V× 过估**。物理配置 pp=2、8 层（L=4/stage ≥ v，可真切 chunk；旧 4 层/pp=4 是 L=1 退化配置）。"""
    p1 = _sim_interleave(2, 20, 1, n_layers=8, record_timeline=True)
    p2 = _sim_interleave(2, 20, 2, n_layers=8, record_timeline=True)
    assert _max_act_live(p2) > _max_act_live(p1)          # 交错更深 warmup → 更吃激活（10·s > 8·s）
    assert p2.peak_bytes > p1.peak_bytes
    assert _max_act_live(p2) < 2 * _max_act_live(p1)      # D-4：< V×plain（~V× 过估已消）


def test_interleave_peak_bulges_at_v2_stage0():
    """D-4 修正：VPP 峰值随 V **非单调**——膨胀比 ≈ 1+(pp-1)/(pp·V) 在 **V=2 处最大、随 V 回落**
    （旧 simulate「单调增」是 ~V× 过估的副产物）。物理 pp=2、8 层（L=4/stage）：max act_live（层单位）
    v1=(1+1)·4=8 → v2=(4+1)·2=10 → v4=(8+1)·1=9 → v2 > v4 > v1；peak_bytes 同签名。"""
    pk = {v: _sim_interleave(2, 20, v, n_layers=8).peak_bytes for v in (1, 2, 4)}
    assert pk[2] > pk[4] > pk[1]                          # bulge at v2（非单调增）
    ac = {v: _max_act_live(_sim_interleave(2, 20, v, n_layers=8, record_timeline=True))
          for v in (1, 2, 4)}
    assert ac[2] > ac[4] > ac[1]                          # act_live 同签名（10 > 9 > 8 层单位）
    assert ac[2] < 2 * ac[1] and ac[4] < 4 * ac[1]        # 每个 v>1 均 < V×plain
