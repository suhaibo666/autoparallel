"""liveness 仿真器与桶模型的**等价性交叉校验**（口径对齐 + 非 liveness 层原样保留）。

liveness 是**只读交叉校验**：它必须与桶路径**同字节口径**地解释同一份 op 清单，差异只允许出现
在「激活如何随时间存活」这一件事上。本文件把这条约束固化成断言：

  1. **无重算前向末驻留 ≡ `Σ activation_saves`** —— liveness 的 saved 记账（含 norm-fp32 cast
     口径、逐张量块对齐）与 `structure_mem.activation_saves` 逐字节一致。
  2. **全重算前向末驻留 ≡ `Σ checkpoint_input`** —— liveness 的区域边界记账（bf16 层入口，
     **不**被 norm-fp32 污染）与 `structure_mem.checkpoint_input` 逐字节一致。
  3. **非 liveness 桶原样保留**（设计要求 5）——`persistent` / `gather_buf` / `grad_buf` /
     `grad_accum` / `optstep` 在同一事件上与桶模型逐字节相同。
  4. **grad 可达性的判别性**（真机 csa.py:764-765 vs :794-795）：同一层里
     * `index_scores`（`CSAIndexer` 的 bmm 输出）**保留** —— 尽管它的激活路径被
       `x_detach = ops.stop_gradient(x)` / `qr_detach = ops.stop_gradient(qr)` 截断
       （csa.py:764-765），但 `forward_before_topk` 里 `q = self.linear_wq_b(qr)`
       （indexer.py:177）带**自己的权重** → **params 是梯度根** → 该 op 有反向节点
       （其 bprop 产 `d_query_index/d_key_index/d_weights`，csa.py:300-302）→ 须保留；
     * `ukl1/ukl2`（KL **目标**分布）**丢弃** —— `unfused_indexer_loss(...,
       ops.stop_gradient(query), ops.stop_gradient(compressed_kv))`（csa.py:794-795）的目标侧
       链路（indexer.py:350 `matmul` + :380 fp32 `softmax`）**一个 param 都没有** → 无反向节点。
     二者由**同一条规则**分开（`has_bwd = 任一输入 requires_grad 或 带 params`；`detached` 切链），
     不含任何针对这一例的特判。
"""
from __future__ import annotations

import dataclasses

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.liveness import (LIVENESS_CATEGORIES, build_stage_graphs,
                                simulate_liveness)
from cost_eval.mem_timeline import MemTimeline
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.presets import deepseek_v4
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import (HardwareSpec, OptimizerSpec, ParallelConfig,
                             RecomputeSpec, SwapSpec)
from cost_eval.static_mem import StaticMem
from cost_eval.structure_mem import estimate_structure_memory

_HW = HardwareSpec(max_device_memory=64 * 2 ** 30)
_OPT = OptimizerSpec.adamw()


def _toy(n_layers=3):
    d = DimTable(H=256, F=512, n_heads=8, n_kv=8, head_dim=32,
                 S=512, B=1, vocab=1024, n_layers=n_layers)
    return ModelSpec("toy", d, ["dense"] * n_layers,
                     {"dense": build_dense_decoder(d)})


def _resolved(spec, pc):
    world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
    pm = ParallelModel(pc, spec.dims.n_layers, world)
    return ShapeEval().resolve(spec, pm), pm


@pytest.mark.parametrize("norm_dt", [0, 4])
def test_no_recompute_forward_residency_equals_activation_saves(norm_dt):
    """无重算前向末的 liveness 驻留 ≡ `Σ_layers activation_saves`（逐字节，含 norm-fp32 口径）。"""
    spec = _toy(3)
    spec.dims.norm_compute_dtype_bytes = norm_dt
    pc = ParallelConfig(pp=1, num_microbatches=1)
    res = simulate_liveness(spec, pc, _OPT, _HW, RecomputeSpec("None"), SwapSpec(),
                            record_timeline=True)
    g, _ = _resolved(spec, pc)
    want = sum(estimate_structure_memory(
        l.ops, alloc_block_bytes=_HW.alloc_block_bytes,
        norm_compute_dtype_bytes=norm_dt).activation_saves
        for l in g.stages[0])
    fe = [s for s in res.per_stage[0].timeline if s.event == "fwd_end"][-1]
    got = sum(v for k, v in fe.buckets.items() if k in LIVENESS_CATEGORIES)
    assert got == want, f"liveness 驻留 {got} != Σactivation_saves {want}"


def test_full_recompute_forward_residency_equals_checkpoint_inputs():
    """全重算前向末的 liveness 驻留 ≡ `Σ_layers checkpoint_input`（bf16 层入口，非 fp32）。"""
    spec = _toy(3)
    spec.dims.norm_compute_dtype_bytes = 4          # norm-fp32 开启也不应污染边界口径
    pc = ParallelConfig(pp=1, num_microbatches=1)
    res = simulate_liveness(spec, pc, _OPT, _HW,
                            RecomputeSpec("full", full_layers={0, 1, 2}), SwapSpec(),
                            record_timeline=True)
    g, _ = _resolved(spec, pc)
    want = sum(estimate_structure_memory(
        l.ops, alloc_block_bytes=_HW.alloc_block_bytes,
        norm_compute_dtype_bytes=4).checkpoint_input for l in g.stages[0])
    fe = [s for s in res.per_stage[0].timeline if s.event == "fwd_end"][-1]
    got = sum(v for k, v in fe.buckets.items() if k in LIVENESS_CATEGORIES)
    assert got == want, f"liveness 边界驻留 {got} != Σcheckpoint_input {want}"


def test_non_liveness_buckets_preserved_from_bucket_model():
    """非 liveness 桶（设计要求 5）在**同一事件**上与桶模型逐字节相同。

    取两模型各自的 `bwd@lid` 事件序列（同一事件网格），逐事件比对 persistent / gather_buf /
    grad_buf / grad_accum —— 这些是既有标定/公式，liveness 只沿用、不重算。"""
    spec = _toy(3)
    pc = ParallelConfig(pp=1, num_microbatches=2)
    rc = RecomputeSpec("full", full_layers={0, 1, 2})
    g, pm = _resolved(spec, pc)
    persistent = StaticMem().compute(g, _OPT, pm,
                                     alloc_block_bytes=_HW.alloc_block_bytes)
    bucket = MemTimeline().simulate(
        g, rc, SwapSpec(), pm, persistent, 0, _HW.max_device_memory,
        record_timeline=True, alloc_block_bytes=_HW.alloc_block_bytes)[0]
    live = simulate_liveness(spec, pc, _OPT, _HW, rc, SwapSpec(),
                            record_timeline=True).per_stage[0]
    # 事件身份用 (event, mb)——`bwd@lid` 在每个微批各出现一次，只用标签会互相覆盖。
    lb = {(s.event, s.mb): s.buckets for s in live.timeline}
    n = 0
    for s in bucket.timeline:
        k = (s.event, s.mb)
        if not s.event.startswith("bwd@") or k not in lb:
            continue
        n += 1
        for field in ("persistent", "gather_buf", "grad_buf", "grad_accum"):
            assert lb[k].get(field, 0) == getattr(s.breakdown, field), (
                f"{k}.{field}: liveness={lb[k].get(field, 0)} "
                f"!= bucket={getattr(s.breakdown, field)}")
    assert n >= 3, f"应比对到至少 3 个 bwd 事件，实得 {n}"


# ---------------------------------------------------------------------------
# grad 可达性的判别性（同一层里保留 index_scores、丢弃 detached 的 KL 目标分布）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def unfused_r4_graph():
    spec = build_llm_spec(dataclasses.replace(deepseek_v4(num_layers=2), dsa_fused=False))
    graphs = build_stage_graphs(spec, pp=1, m=1)
    r4 = [lg for lg in graphs[0] if lg.layer_type.startswith("dsv4hyb_r4")]
    assert r4, [lg.layer_type for lg in graphs[0]]
    return r4[0]


def test_params_as_grad_root_keeps_indexer_index_scores(unfused_r4_graph):
    """`index_scores` **保留**：激活路径被 stop_gradient 截断，但 indexer 自带权重 → params 是梯度根。"""
    lg = unfused_r4_graph
    assert "index_scores" in lg.tensors, sorted(lg.tensors)
    assert lg.tensors["index_scores"].requires_grad
    assert "index_scores" in lg.kept_for_backward
    idx = [n for n in lg.fwd if n.op_name == "indexer"]
    assert idx and idx[0].has_bwd, "indexer op 带 params → 必须有反向节点"


def test_detached_kl_target_branch_dropped(unfused_r4_graph):
    """`ukl1/ukl2`（KL 目标分布，无 param、双输入 detached）**不**保留。"""
    lg = unfused_r4_graph
    for n in ("ukl1", "ukl2"):
        assert n in lg.tensors, sorted(lg.tensors)
        assert lg.tensors[n].detached
        assert not lg.tensors[n].requires_grad
        assert n not in lg.kept_for_backward


def test_grad_reachability_is_discriminating_not_blanket(unfused_r4_graph):
    """判别性：同一层同一 op 的 saves 里，detached 的被丢、其余大 fp32 复本群全部保留。"""
    lg = unfused_r4_graph
    dropped = {n for n, t in lg.tensors.items()
               if not t.is_weight and not t.requires_grad}
    assert dropped == {"ukl1", "ukl2"}, f"只应丢 KL 目标分布，实得 {sorted(dropped)}"
    for n in ("kv_gathered", "kv_g_fp32", "attn_weights", "uq_f32", "uq_bm",
              "ukv_bm", "uscore1", "uscore2", "uexp", "uaw_bm", "uout_f32", "uout_pm"):
        assert n in lg.kept_for_backward, n


def test_bucket_path_still_declares_ukl_as_saves(unfused_r4_graph):
    """`detached` 是**旁路标注**：桶路径的 `saves` 清单不变（liveness 只在自己这侧过滤）。

    守住「只读交叉校验、不改桶模型一个字节」这条边界——若哪天有人把 ukl 从 census 的 `saves`
    里删掉，桶模型的锚点会变，此测试提醒那是另一个决定。"""
    spec = build_llm_spec(dataclasses.replace(deepseek_v4(num_layers=2), dsa_fused=False))
    lids = [i for i, t in enumerate(spec.layer_pattern) if t.startswith("dsv4hyb_r4")]
    layer = spec.get_layer(spec.layer_pattern[lids[0]])
    names = {s.name for op in layer.ops for s in op.saves}
    assert {"ukl1", "ukl2"} <= names
