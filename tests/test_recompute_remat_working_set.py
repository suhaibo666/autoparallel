# -*- coding: utf-8 -*-
"""重算再物化 saved 集(`remat_saves` 桶)——167/MS2.10 单卡微基准实测驱动(2026-07-25)。

**实测事实**(`log_release_probe/VERDICT_2026-07-25.txt`,spec §1):
  ① rc=ON `fwd_end` ≡ 0(NBLK=2/4/8,三锚法同值)→ 重算**确实**释放激活("不释放"被证伪);
  ② rc=ON `fwd_peak` 与 NBLK **无关**(复刻 552 / 真模块 403 恒定)→ 残余量是**单区域瞬态 ×1**,
     非 ×微批深度、非 ×层数;
  ③ rc=ON `total` > `fwd_peak`(1064 vs 552)→ 反向期「重算再物化的激活 + 瞬态 + 梯度缓冲」同驻。

**缺口**:`forward_max_live`(structure_mem.py:142-175)的语义是「前向中间量在最后一次被用完就死」;
但**重算再执行**的目的恰是重建 backward 需要的那批 saved 张量——它们**不能**在前向末尾死,必须活到
该区域 backward 消费完。故重算再执行期真正同驻的是**整个 `activation_saves` 集**,此前记 0。

**公式**:`remat_saves(R) = max(0, A(R) − ci(R))`,A=区域 `activation_saves`、ci=`checkpoint_input`
(已计在 `act_live`,减掉防重复)。逐层赋在**该层自己的 `bwd@lid` 事件**上 → 同一时刻只有一个 bwd
事件在世 → **×1 自动成立**(对应实测 ②),且逐层精确。**不沿用 `_pp_full_recomp`(pp>1) 门**——重算
再物化与 pp 无关。
"""
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_eval.layers.dense import build_dense_decoder
from cost_eval.mem_timeline import MemTimeline
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import (OptimizerSpec, ParallelConfig, RecomputeSpec,
                             SwapSpec)
from cost_eval.static_mem import StaticMem
from cost_eval.structure_mem import (estimate_select_memory,
                                     estimate_structure_memory)

D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=4)


def _setup(pp=1, m=1):
    spec = ModelSpec("toy", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(pp=pp, num_microbatches=m), n_layers=4, world_size=pp)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(), pm, False)
    return g, pm, persistent


def _sim(recompute, pp=1, m=1):
    g, pm, persistent = _setup(pp, m)
    r = MemTimeline().simulate(g, recompute, SwapSpec(), pm, persistent,
                               framework_reserve=0, max_device_memory=10 ** 12,
                               record_timeline=True)
    return g, r


def _layer_sm(g, lid, stage=0):
    ops = next(l for l in g.stages[stage] if l.layer_id == lid).ops
    return estimate_structure_memory(ops)


def _max_remat(sp) -> int:
    return max(s.breakdown.remat_saves for s in sp.timeline)


# ---------------------------------------------------------------------------
# ① mode=None → 该项恒 0，且峰值逐字节不变（严格回归保护）
# ---------------------------------------------------------------------------

def test_none_mode_remat_zero_and_peak_byte_identical():
    """无重算 → 无重算区域 → `remat_saves` 全时间线恒 0，峰值逐字节复现旧锚
    （与 test_mem_timeline.test_none_full_byte_identical_anchor 同锚 5284352）。"""
    _, r = _sim(RecomputeSpec("None"))
    sp = r[0]
    assert all(s.breakdown.remat_saves == 0 for s in sp.timeline)
    assert sp.breakdown.remat_saves == 0
    assert sp.peak_bytes == 5284352


# ---------------------------------------------------------------------------
# ② mode=full → 该项 = max(0, activation_saves − checkpoint_input)，落在该层 bwd@lid
# ---------------------------------------------------------------------------

def test_full_mode_remat_equals_saves_minus_checkpoint_input():
    g, r = _sim(RecomputeSpec("full", {0, 1, 2, 3}))
    sp = r[0]
    for lid in range(4):
        sm = _layer_sm(g, lid)
        ev = next(s for s in sp.timeline if s.event == f"bwd@{lid}")
        expect = max(0, sm.activation_saves - sm.checkpoint_input)
        assert expect > 0, "toy dense 层 saves 应 > 层入口（否则本测试无判别力）"
        assert ev.breakdown.remat_saves == expect, (lid, ev.breakdown.remat_saves, expect)
    # 非反向事件（FWD / optstep）不带该项——它是反向瞬时项，不是驻留项
    for s in sp.timeline:
        if not s.event.startswith("bwd@"):
            assert s.breakdown.remat_saves == 0, s.event


def test_full_mode_remat_is_larger_than_recomp_scratch_for_dense_toy():
    """机理断言：`activation_saves`（整层 saved 集）与 `forward_max_live`（单时刻峰）无大小序
    （structure_mem.py:43-48）。toy dense 层上 saves 显著大于 fml → 新项确实补了此前记 0 的量，
    不是 recomp_scratch 的重命名。"""
    g, r = _sim(RecomputeSpec("full", {0, 1, 2, 3}))
    ev = next(s for s in r[0].timeline if s.event == "bwd@3")
    sm = _layer_sm(g, 3)
    assert sm.activation_saves > sm.forward_max_live
    assert ev.breakdown.remat_saves > ev.breakdown.recomp_scratch > 0


# ---------------------------------------------------------------------------
# ③ 微批数 m 不改变该项（实测 NBLK 无关性的直接类比）
# ---------------------------------------------------------------------------

def test_full_mode_remat_independent_of_microbatch_count():
    vals = []
    for m in (1, 2, 4, 8):
        _, r = _sim(RecomputeSpec("full", {0, 1, 2, 3}), m=m)
        vals.append(_max_remat(r[0]))
    assert vals[0] > 0
    assert len(set(vals)) == 1, f"remat 随微批数变化 {vals}——违反实测 NBLK 无关性（×1）"


# ---------------------------------------------------------------------------
# ④ 增加重算层数不使该项翻倍（×1；只可能改变哪层是 max）
# ---------------------------------------------------------------------------

def test_full_mode_remat_not_multiplied_by_recomputed_layer_count():
    vals = []
    for lset in ({3}, {2, 3}, {1, 2, 3}, {0, 1, 2, 3}):
        _, r = _sim(RecomputeSpec("full", lset))
        vals.append(_max_remat(r[0]))
    assert vals[0] > 0
    assert len(set(vals)) == 1, f"remat 随重算层数增长 {vals}——同一时刻只有一个 bwd 事件在世，应 ×1"
    g, _ = _sim(RecomputeSpec("full", {3}))
    sm = _layer_sm(g, 3)
    assert vals[0] == max(0, sm.activation_saves - sm.checkpoint_input)


# ---------------------------------------------------------------------------
# ⑤ mode=select → 该项 = max over islands，< 同层 full，且未选中激活仍驻留
# ---------------------------------------------------------------------------

def test_select_mode_remat_smaller_than_full_and_nonselected_still_resident():
    _, full = _sim(RecomputeSpec("full", {0, 1, 2, 3}))
    _, sel = _sim(RecomputeSpec("select", select_ops={lid: {"flash"} for lid in range(4)}))
    f, s = _max_remat(full[0]), _max_remat(sel[0])
    assert 0 < s < f, (s, f)
    # 未选中 op 的激活仍计驻留（select act_live 高于 full）
    assert (max(x.breakdown.act_live for x in sel[0].timeline)
            > max(x.breakdown.act_live for x in full[0].timeline))


def test_select_all_ops_remat_equals_full():
    """两端退化（守 select-all == full 的桶级等价）。"""
    g, full = _sim(RecomputeSpec("full", {0, 1, 2, 3}))
    all_names = {op.name for op in g.stages[0][0].ops}
    _, sel = _sim(RecomputeSpec("select", select_ops={lid: set(all_names) for lid in range(4)}))
    assert _max_remat(sel[0]) == _max_remat(full[0])


def test_select_none_remat_equals_no_recompute():
    """两端退化（守 select-none == None）。"""
    _, none = _sim(RecomputeSpec("None"))
    _, sel0 = _sim(RecomputeSpec("select", select_ops={0: set()}))
    assert _max_remat(sel0[0]) == _max_remat(none[0]) == 0


def test_select_memory_island_remat_degenerate_ends():
    """`SelectMemory` 层：全选 → remat == A−ci（== full 式）；全不选 → remat == 0（== None）。"""
    spec = ModelSpec("toy", D, ["dense"], {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(), n_layers=1, world_size=1)
    ops = ShapeEval().resolve(spec, pm).stages[0][0].ops
    sm = estimate_structure_memory(ops)
    all_sel = estimate_select_memory(ops, lambda op: True)
    assert all_sel.remat_saves == max(0, sm.activation_saves - sm.checkpoint_input)
    assert all_sel.island_remat == (all_sel.remat_saves,)
    none_sel = estimate_select_memory(ops, lambda op: False)
    assert none_sel.remat_saves == 0
    assert none_sel.island_remat == ()


def test_select_memory_multi_island_remat_is_max_not_sum():
    """非连续选中 → 多 island 各自独立重算单元（反向逐 island 重物化、算完释放）→ 取 max、非 Σ。"""
    spec = ModelSpec("toy", D, ["dense"], {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(), n_layers=1, world_size=1)
    ops = ShapeEval().resolve(spec, pm).stages[0][0].ops
    names = [op.name for op in ops]
    # 选首、末两个有 saves 的 op → 两个不相邻 island
    picks = [n for n in names if n in ("ln1", "fc2")] or [names[0], names[-1]]
    sel = estimate_select_memory(ops, lambda op: op.name in set(picks))
    assert sel.n_islands >= 2, (sel.n_islands, picks)
    assert sel.remat_saves == max(sel.island_remat)
    if len([x for x in sel.island_remat if x > 0]) >= 2:
        assert sel.remat_saves < sum(sel.island_remat)


# ---------------------------------------------------------------------------
# ⑦ pp=1 全重算 → 该项非零（**不**沿用 bwd_working_set 的 pp>1 门）
# ---------------------------------------------------------------------------

def test_pp1_full_recompute_remat_not_pp_gated():
    """spec §3：重算再物化与 pp 无关，pp=1 同样发生。对照组 `bwd_working_set` 在 pp=1
    全重算下**是** pp-gated（mem_timeline.py `_pp_full_recomp`）→ 恒 0，凸显新项没被同门挡掉。"""
    _, r = _sim(RecomputeSpec("full", {0, 1, 2, 3}), pp=1)
    ev = next(s for s in r[0].timeline if s.event == "bwd@3")
    assert ev.breakdown.bwd_working_set == 0      # 旧项：pp>1 门
    assert ev.breakdown.remat_saves > 0          # 新项：无 pp 门


def test_pp2_full_recompute_remat_present_on_both_stages():
    """pp=2 下每 stage 各自的重算层 bwd 事件都带该项（逐层赋值，非 stage 级 max）。"""
    g, r = _sim(RecomputeSpec("full", {0, 1, 2, 3}), pp=2, m=4)
    for stage, sp in r.items():
        lids = [l.layer_id for l in g.stages[stage]]
        for lid in lids:
            ev = next(s for s in sp.timeline if s.event == f"bwd@{lid}")
            sm = _layer_sm(g, lid, stage)
            assert ev.breakdown.remat_saves == max(
                0, sm.activation_saves - sm.checkpoint_input)


# ---------------------------------------------------------------------------
# ⑥ fused vs unfused DSA → unfused 该项更大（unfused 显式物化 fp32 复本群）
# ---------------------------------------------------------------------------

_DSV4 = {
    "preset": "dsv4_flash", "attn": "dsv4_hybrid",
    "layers": "4", "seq": "2048", "batch": "1", "mtp": "0",
    "experts": "8", "topk": "2", "dense_k": "1",
    "heads": "64", "kv_groups": "1",
    "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
    "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
    "v_head": "512", "vocab": "129280",
    "hc": "4", "ce_fused": "1",
    "dp": "2", "tp": "1", "ep": "2", "pp": "1", "cp": "1", "method": "colossal",
    "optimizer": "muon", "opt_dtype": "fp32", "grad_bytes": "4",
    "maxdev_gib": "58", "recompute": "full",
    "select": "attn", "sel_layers": "", "sel_ops": "", "sel_cfg": "", "vpp": "1", "mbs": "",
    "dp_replicate": "1", "reshard": "default", "cpu_offload": "0", "prefetch": "1", "sp": "",
}


def _dsv4_remat(fused: int):
    import serve_explorer as S
    p = dict(_DSV4)
    p["dsa_fused"] = str(fused)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = S.eval_config(p)
    assert r.get("ok"), r.get("errors")
    return max(s["buckets"].get("remat_saves", 0)
               for st in r["stages"] for s in st["timeline"])


def test_unfused_dsa_remat_larger_than_fused():
    """unfused `unfused_compressed_sparse_attn` 逐 op bprop 显式物化 fp32 复本群
    （`layers/dsv4_hybrid.py:187-211` uq_f32/uq_bm/ukv_bm/uscore1/uscore2/uexp/uaw_bm/
    uout_f32/uout_pm(+KL 链)）→ 被重算区域的 saved 集更大 → 该项更大。fused 那批走 kernel
    scratch、不进 saves。"""
    fused, unfused = _dsv4_remat(1), _dsv4_remat(0)
    assert fused > 0 and unfused > fused, (fused, unfused)
