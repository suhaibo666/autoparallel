"""逐张量 liveness 仿真器（`cost_eval/liveness/`）的**不变量测试**。

这些不变量全部来自 **167/MS2.10 真机实测**（2026-07-25），先写测试再写实现（TDD）：

  A. **重算驻留 → 0**（单卡微基准，`ms.runtime.memory_allocated()` 在 N 个独立 checkpoint 块
     前向后、反向前采样）：`rc=ON` 下 `fwd_end ≡ 0`（NBLK=2/4/8 三锚同值，bare_ctx/saved/pyref
     三种锚法同值）→ 重算**确实**释放 saved 激活，前向末只剩区域边界。
  B. **重算工作集 ×1 于微批数**：`rc=ON` 的 `fwd_peak` 与 N **无关**（复刻 552 MiB / 真生产模块
     403 MiB 恒定）→ 残余是**单区域**工作集，不随在飞微批数增长。
  C. **重算工作集 ×1 于层数**：167 多卡 A/B（table B）stage3 的 unfused−fused 差在 **L8（2 层/
     stage）= 24380.1** 与 **L4（1 层/stage）= 24380.1** 逐 0.1 MiB 相同。
  D. **select 区域 < full 整层**：选中岛只重算自身，重算工作集必小于整层。
  E. **unfused > fused**：unfused 小算子链物化 fp32 复本群，fused kernel 走 scratch。
  F. **grad 可达性**：`stop_gradient` 后的子计算不建 autograd 节点 → 其张量**不是 saved 张量**
     （真机 csa.py:794-795 `ops.stop_gradient(query)/(compressed_kv)` → indexer.py:350 的
     `matmul` + :380 的 fp32 softmax 全在 detached 侧）。

不变量 A/B/C/D/G 用**合成 dense 模型**（快、与 DSv4 census 解耦）；E/F 用 DSv4 真 census。
"""
from __future__ import annotations

import pytest

from cost_eval.layers.dense import build_dense_decoder
from cost_eval.liveness import simulate_liveness
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.specs import (HardwareSpec, OptimizerSpec, ParallelConfig,
                             RecomputeSpec, SwapSpec)


# ---------------------------------------------------------------------------
# 合成模型夹具
# ---------------------------------------------------------------------------

def _toy_spec(n_layers: int) -> ModelSpec:
    """合成 dense GQA decoder 模型（无 embedding/head 伪层 → layer_id = 0..n-1）。"""
    d = DimTable(H=256, F=512, n_heads=8, n_kv=8, head_dim=32,
                 S=512, B=1, vocab=1024, n_layers=n_layers)
    return ModelSpec("toy", d, ["dense"] * n_layers,
                     {"dense": build_dense_decoder(d)})


_HW = HardwareSpec(max_device_memory=64 * 2 ** 30)


def _run(spec, *, pp=1, m=2, recompute=None, hw=None, **pckw):
    pc = ParallelConfig(pp=pp, num_microbatches=m, **pckw)
    return simulate_liveness(
        spec, pc, OptimizerSpec.adamw(), hw or _HW,
        recompute or RecomputeSpec("None"), SwapSpec(), record_timeline=True)


# ---------------------------------------------------------------------------
# A. 重算驻留 → 0（前向末只剩区域边界）
# ---------------------------------------------------------------------------

def test_full_recompute_forward_residency_is_boundary_only():
    """全重算下前向末（`fwd_end`）的 act_live 只含**区域边界**，saved 内部量驻留 ≡ 0。

    真机锚点：167/MS2.10 微基准 `rc=ON fwd_end ≡ 0`（NBLK=2/4/8）——即区域内部 saved 全部释放。
    本库口径下前向末仍留 checkpoint 边界（层入口 hidden），故断言 = m_inflight × 边界字节，
    且 `act_saved`（内部保留量）类别恒 0。"""
    spec = _toy_spec(2)
    res = _run(spec, m=3, recompute=RecomputeSpec("full", full_layers={0, 1}))
    st = res.per_stage[0]
    fwd_ends = [s for s in st.timeline if s.event == "fwd_end"]
    assert fwd_ends, "应有 fwd_end 采样点"
    for s in fwd_ends:
        assert s.buckets.get("act_saved", 0) == 0, (
            f"全重算前向末不应驻留任何区域内部 saved 张量，实得 {s.buckets}")
        assert s.buckets.get("act_boundary", 0) > 0, "应只剩区域边界"


def test_no_recompute_forward_residency_grows_with_microbatch():
    """无重算下前向驻留**随在飞微批线性增长**（对照组：微基准 rc=OFF 320 MiB/块 线性）。"""
    spec = _toy_spec(2)
    res = _run(spec, pp=2, m=4)          # stage0 warmup=1 → 多个在飞微批
    st = res.per_stage[0]
    fwd_ends = [s.buckets.get("act_saved", 0) for s in st.timeline if s.event == "fwd_end"]
    assert len(fwd_ends) >= 2
    assert fwd_ends[1] > fwd_ends[0], f"无重算驻留应随微批累加，实得 {fwd_ends}"


# ---------------------------------------------------------------------------
# B. 重算工作集 ×1 于微批数
# ---------------------------------------------------------------------------

def test_recompute_working_set_invariant_to_microbatch_count():
    """重算工作集（`recomp_*` 类别）与微批数 m **无关**。

    真机锚点：`rc=ON fwd_peak` 与 NBLK∈{2,4,8} 无关（552/403 MiB 恒定）；多卡 (a)vs(e)/(b)vs(f)
    在 m=4→8 上 delta 移动 ≤0.4 MiB。"""
    spec = _toy_spec(2)
    rc = RecomputeSpec("full", full_layers={0, 1})
    ws = [_run(spec, pp=2, m=m, recompute=rc).per_stage[0].max_recompute_working_set
          for m in (2, 4, 8)]
    assert len(set(ws)) == 1, f"重算工作集应 ×1 于 m，实得 {ws}"
    assert ws[0] > 0


# ---------------------------------------------------------------------------
# C. 重算工作集 ×1 于层数
# ---------------------------------------------------------------------------

def test_recompute_working_set_invariant_to_layer_count():
    """重算工作集与**每 stage 层数**无关（同时只有一个区域在重算）。

    真机锚点：table B stage3 unfused−fused delta 在 L8(2 层/stage) 与 L4(1 层/stage)
    均为 24380.1 MiB（逐 0.1 MiB 相同）。"""
    ws = []
    for n_layers in (1, 2, 4):
        spec = _toy_spec(n_layers)
        rc = RecomputeSpec("full", full_layers=set(range(n_layers)))
        st = _run(spec, pp=1, m=2, recompute=rc).per_stage[0]
        ws.append(st.max_recompute_working_set)
    assert len(set(ws)) == 1, f"重算工作集应 ×1 于层数，实得 {ws}"
    assert ws[0] > 0


# ---------------------------------------------------------------------------
# D. select 区域 < full 整层
# ---------------------------------------------------------------------------

def test_select_region_working_set_below_full_layer():
    """选中岛（单 op 粒度）的重算工作集 < 整层全重算的重算工作集。"""
    spec = _toy_spec(2)
    full = _run(spec, m=2, recompute=RecomputeSpec("full", full_layers={0, 1})).per_stage[0]
    sel = _run(spec, m=2, recompute=RecomputeSpec(
        "select", select_ops={0: {"flash"}, 1: {"flash"}})).per_stage[0]

    a, b = sel.max_recompute_working_set, full.max_recompute_working_set
    assert 0 < a < b, f"select={a} 应 ∈ (0, full={b})"


def test_select_all_ops_matches_full_layer_working_set():
    """选中**全部 op** 时 select 的重算工作集退化为 full（区域 == 整层）。"""
    spec = _toy_spec(1)
    ops = [o.name for o in spec.get_layer("dense").ops]
    full = _run(spec, m=2, recompute=RecomputeSpec("full", full_layers={0})).per_stage[0]
    sel = _run(spec, m=2, recompute=RecomputeSpec("select",
                                                  select_ops={0: set(ops)})).per_stage[0]
    assert sel.peak_bytes == full.peak_bytes


# ---------------------------------------------------------------------------
# G. 每张量只算一次（liveness 的核心正确性）
# ---------------------------------------------------------------------------

def test_peak_live_set_counts_each_tensor_once():
    """峰值 live-set 的 key 唯一，且 Σ 逐项字节 == 该采样点的 liveness 分量。"""
    spec = _toy_spec(2)
    st = _run(spec, pp=2, m=4,
              recompute=RecomputeSpec("full", full_layers={0, 1})).per_stage[0]
    keys = [it.key for it in st.live_set]
    assert len(keys) == len(set(keys)), "live-set 出现重复 key（双算）"
    tot = sum(it.nbytes for it in st.live_set)
    lv = sum(v for k, v in st.buckets.items() if k in st.liveness_categories)
    assert tot == lv, f"Σ live-set {tot} != liveness 桶合计 {lv}"


def test_recompute_peak_below_no_recompute_peak():
    """全重算峰 < 无重算峰（重算就是拿时间换内存；liveness 不应把它算反）。"""
    spec = _toy_spec(4)
    none = _run(spec, pp=1, m=4).per_stage[0]
    full = _run(spec, pp=1, m=4,
                recompute=RecomputeSpec("full", full_layers={0, 1, 2, 3})).per_stage[0]
    assert full.peak_bytes < none.peak_bytes


# ---------------------------------------------------------------------------
# E/F. DSv4 真 census：unfused > fused、grad 可达性
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dsv4_specs():
    """DSv4-hybrid preset 的 fused / unfused 两份 spec（其余逐字段相同）。"""
    import dataclasses

    from cost_eval.build_llm import build_llm_spec
    from cost_eval.presets import deepseek_v4
    base = deepseek_v4(num_layers=2)
    return {tag: build_llm_spec(dataclasses.replace(base, dsa_fused=fused))
            for tag, fused in (("fused", True), ("unfused", False))}


def test_unfused_exceeds_fused(dsv4_specs):
    """unfused 小算子链的重算工作集 >> fused kernel（fp32 复本群物化 vs 走 kernel scratch）。"""
    ws = {}
    for tag, spec in dsv4_specs.items():
        lids = [i for i, t in enumerate(spec.layer_pattern) if "dsv4" in t]
        st = _run(spec, pp=1, m=2,
                  recompute=RecomputeSpec("full", full_layers=set(lids))).per_stage[0]
        ws[tag] = st.max_recompute_working_set
    assert ws["unfused"] > ws["fused"] * 2, ws


def test_grad_reachability_drops_detached_subgraph(dsv4_specs):
    """**grad 可达性**：detached 子计算的张量不被保留为 saved（真机 csa.py:794-795）。

    unfused CSA(ratio=4) 层的 indexer-KL 目标分布张量 `ukl1/ukl2` 由
    `ops.stop_gradient(query)` / `ops.stop_gradient(compressed_kv)` 喂出（csa.py:794-795 →
    indexer.py:350 `matmul` → :380 fp32 `softmax`）→ 该子计算不建 autograd 节点 → 反向无节点
    读它 → **不是 saved 张量**（微基准实测 ~2 MiB/blk 纯瞬态，而 saved 会显 ~128 MiB/blk）。

    断言的是**机制**：`kept_for_backward` 由 grad 可达性导出，凡 detached 张量一律不入；
    并同时确认**同层非 detached 的大 saved 张量仍在**（不是把整层误杀）。"""
    from cost_eval.liveness import build_stage_graphs
    spec = dsv4_specs["unfused"]
    graphs = build_stage_graphs(spec, pp=1, m=1)
    r4 = [lg for lg in graphs[0] if lg.layer_type.startswith("dsv4hyb_r4")]
    assert r4, f"应有 ratio=4 层，实得 {[lg.layer_type for lg in graphs[0]]}"
    lg = r4[0]
    detached = {n for n, t in lg.tensors.items() if t.detached}
    assert detached, "unfused r4 层应有 detached 张量（indexer KL 目标分布）"
    assert not (detached & lg.kept_for_backward), (
        f"detached 张量不应被保留为 saved：{sorted(detached & lg.kept_for_backward)}")
    # 未被误杀：unfused 的其它 fp32 复本群仍是 saved。
    assert "kv_g_fp32" in lg.kept_for_backward
    assert "uq_f32" in lg.kept_for_backward


def test_params_are_gradient_roots():
    """params 是梯度根：只有权重输入、激活输入不需梯度的 op 仍建反向节点。"""
    from cost_eval.liveness import build_stage_graphs
    spec = _toy_spec(1)
    lg = build_stage_graphs(spec, pp=1, m=1)[0][0]
    assert any(n.has_bwd for n in lg.fwd)
    # 权重恒 requires_grad
    assert all(t.requires_grad for t in lg.tensors.values() if t.is_weight)
