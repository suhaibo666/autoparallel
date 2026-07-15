"""Task Z1 [P1-07] — checkpoint islands 一等对象。

`estimate_select_memory` 此前把「选中」op 压成**一整块**（`estimate_structure_memory(selected)`
的 `forward_max_live` + 全局 `_pinned_input_boundary`），未表达**多个不相邻区段（island）**——非连续
选中区段各是独立的重算单元，反向逆序逐 island 重物化、算完释放 → 各 island 的 recomp scratch **不
同时存活** → 峰应取 **max over islands**，而非把它们当一块混算。

本文件把 island 建成显式结构（`_checkpoint_islands`）并锁定：
  - 单 island（连续选中）**逐字节** == 旧「一整块」公式（→ 12 锚点连续 island 不动的机理根因）；
  - 多 island（非连续，如选 a+c 跳 b）recomp = max(各 island)，修旧「一整块」把**某 island 的进入
    边界从另一 island 的峰值里错减**导致的**低估（OOM 不安全）**；
  - 全选 == full、全不选 == no-recompute 两端退化仍逐字节复现；
  - 真实 DSv3 层 ATTN/MLP/BOTH 选择器（select_both 实为 [attn][fc] 两 island）recomp 不变 → 锚点不动。
"""
from cost_eval.model_spec import DimTable, ModelSpec, LayerSpec, OpSpec, OpType, TensorRef
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.specs import ParallelConfig
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.structure_mem import (
    estimate_structure_memory, estimate_select_memory, SelectMemory,
    _checkpoint_islands, _forward_max_live, _pinned_input_boundary, _align_up)

D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100,
             n_layers=1, n_experts=8, topk=2, moe_F=128)


def _resolve(op_list, pc=None):
    pc = pc or ParallelConfig()
    spec = ModelSpec("t", D, ["x"], {"x": LayerSpec(list(op_list))})
    pm = ParallelModel(pc, n_layers=1, world_size=1)
    return ShapeEval().resolve(spec, pm).stages[0][0].ops, pm


def _island_ops():
    """4-op 层，令选 {a,c} 跳 {b,mid} → 两不相邻 island [[a]] 与 [[c]]。

      a:   [X,W]->P  saves=[X]     （X=层入口=ci，全 [S,B,H] 大张量）
      b:   [P,W]->Q  saves=[P]     （非选中：P 常驻）
      mid: [Q]->R    saves=[Q,R]   （非选中：Q,R 常驻）
      c:   [R,W]->Y  saves=[R]     （R 由非选中 mid save → 已 pin）
    全部 [S,B,H] = 128·1·64·4 = 32768 B（无对齐取整），W 是权重不计。
    """
    X = TensorRef("X", ("S", "B", "H"), dtype_bytes=4)                 # 32768（放大以使低估显著）
    W = TensorRef("W", ("H", "H"),      is_weight=True, dtype_bytes=4)
    P = TensorRef("P", ("S", "B", "H"), dtype_bytes=4)                 # 32768
    Q = TensorRef("Q", ("S", "B", "H"), dtype_bytes=4)                 # 32768
    R = TensorRef("R", ("S", "B", "H"), dtype_bytes=4)                 # 32768
    Y = TensorRef("Y", ("S", "B", "H"), dtype_bytes=4)                 # 32768
    ops, _ = _resolve([
        OpSpec("a",   OpType.MATMUL,      [X, W], P, params=[W], saves=[X]),
        OpSpec("b",   OpType.MATMUL,      [P, W], Q, params=[W], saves=[P]),
        OpSpec("mid", OpType.ELEMENTWISE, [Q],    R, saves=[Q, R]),
        OpSpec("c",   OpType.MATMUL,      [R, W], Y, params=[W], saves=[R]),
    ])
    return ops


def _old_block_recomp(ops, is_sel, blk=1):
    """旧「把选中当一整块」的 recomp 公式（合并 forward_max_live − 全局 pinned 边界），
    作为独立参照守卫「新式在单 island 上 byte-identical、在多 island 上分歧」。"""
    selected = [op for op in ops if is_sel(op)]
    nonsel = [op for op in ops if not is_sel(op)]
    ci_name = None
    for op in ops:
        if op.saves:
            ci_name = op.saves[0].name
            break
    pinned = {s.name for op in nonsel for s in op.saves} | ({ci_name} if ci_name else set())
    return max(0, _forward_max_live(selected, blk) - _pinned_input_boundary(selected, pinned, blk))


# ---------------------------------------------------------------------------
# 1. checkpoint island 分段结构（一等对象）
# ---------------------------------------------------------------------------

def test_checkpoint_islands_partitions_by_contiguity():
    ops = _island_ops()
    names = lambda isls: [[op.name for op in isl] for isl in isls]
    # 非连续 {a,c} → 两 island
    assert names(_checkpoint_islands(ops, lambda op: op.name in {"a", "c"})) == [["a"], ["c"]]
    # 连续 {a,b} → 一 island
    assert names(_checkpoint_islands(ops, lambda op: op.name in {"a", "b"})) == [["a", "b"]]
    # 中段连续 {b,mid} → 一 island
    assert names(_checkpoint_islands(ops, lambda op: op.name in {"b", "mid"})) == [["b", "mid"]]
    # 三段跳选 {a,mid} → 两 island（a 与 mid 之间隔 b）
    assert names(_checkpoint_islands(ops, lambda op: op.name in {"a", "mid"})) == [["a"], ["mid"]]
    # 全选 → 一 island；全不选 → 零 island
    assert names(_checkpoint_islands(ops, lambda op: True)) == [["a", "b", "mid", "c"]]
    assert _checkpoint_islands(ops, lambda op: False) == []


# ---------------------------------------------------------------------------
# 2. 多 island：recomp = max(各 island)，修旧「一整块」的低估（OOM 不安全）
# ---------------------------------------------------------------------------

def test_multi_island_recomp_is_max_over_islands_not_merged_block():
    """选 {a,c}（非连续）→ 两独立重算单元。反向逐 island 重物化、算完释放 → recomp 峰 =
    max(island_a, island_c)，各 island 各扣**自己**的进入边界。

    旧「一整块」式把 **a 的边界(X) 与 c 的边界(R) 一并从合并峰值里减掉** → 65536−65536 = 0，
    严重低估（OOM 不安全）；新式各 island 只从自身峰减自身边界 → 32768。"""
    ops = _island_ops()
    is_sel = lambda op: op.name in {"a", "c"}
    sel = estimate_select_memory(ops, is_sel)
    assert isinstance(sel, SelectMemory)
    # island_a: fml([a])=X+P=65536, 边界 X 已 pin(=ci) → 65536−32768 = 32768
    # island_c: fml([c])=R+Y=65536, 边界 R 已 pin(非选中 mid save) → 65536−32768 = 32768
    assert sel.n_islands == 2
    assert sel.island_recomp == (32768, 32768)
    assert sel.recomp_scratch == 32768                       # max(各 island)
    # 旧「一整块」式在此配置低估到 0（把两 island 边界从单峰全减）——新式修正之
    assert _old_block_recomp(ops, is_sel) == 0
    assert sel.recomp_scratch > _old_block_recomp(ops, is_sel)   # 方向：不再低估（OOM 安全）


def test_multi_island_recomp_below_naive_sum():
    """各 island recomp **不相加**（反向不同时存活）：recomp = max ≤ Σ islands（守 max 语义）。"""
    ops = _island_ops()
    sel = estimate_select_memory(ops, lambda op: op.name in {"a", "c"})
    assert sel.recomp_scratch == max(sel.island_recomp)
    assert sel.recomp_scratch < sum(sel.island_recomp)       # 32768 < 65536


# ---------------------------------------------------------------------------
# 3. 单 island（连续选中）逐字节 == 旧「一整块」公式（锚点机理根因）
# ---------------------------------------------------------------------------

def test_single_contiguous_island_byte_identical_to_old_block():
    ops = _island_ops()
    for sel_names in ({"a", "b"}, {"b", "mid"}, {"mid", "c"}, {"a", "b", "mid"}):
        is_sel = lambda op, s=sel_names: op.name in s
        sel = estimate_select_memory(ops, is_sel)
        assert sel.n_islands == 1
        assert sel.recomp_scratch == _old_block_recomp(ops, is_sel)   # 逐字节复现旧值


def test_single_op_island_byte_identical_to_old_block():
    """细粒度单 op 选择（D-3）：单 island → 逐字节复现旧值（守 D-3 语义不被 island 化破坏）。"""
    ops = _island_ops()
    for name in ("a", "b", "mid", "c"):
        is_sel = lambda op, n=name: op.name == n
        sel = estimate_select_memory(ops, is_sel)
        assert sel.n_islands == 1
        assert sel.recomp_scratch == _old_block_recomp(ops, is_sel)


# ---------------------------------------------------------------------------
# 4. 两端退化：全选 == full、全不选 == no-recompute（逐字节）
# ---------------------------------------------------------------------------

def test_select_all_reproduces_full_on_island_layer():
    ops = _island_ops()
    sm = estimate_structure_memory(ops)
    sel = estimate_select_memory(ops, lambda op: True)
    assert sel.n_islands == 1
    assert sel.act_live_pinned == sm.checkpoint_input
    assert sel.recomp_scratch == max(0, sm.forward_max_live - sm.checkpoint_input)
    assert sel.bwd_working_set == 0


def test_select_none_reproduces_no_recompute_on_island_layer():
    ops = _island_ops()
    sm = estimate_structure_memory(ops)
    sel = estimate_select_memory(ops, lambda op: False)
    assert sel.n_islands == 0
    assert sel.island_recomp == ()
    assert sel.act_live_pinned == sm.activation_saves
    assert sel.recomp_scratch == 0
    assert sel.bwd_working_set == max(0, sm.forward_max_live - sm.bwd_scratch)


# ---------------------------------------------------------------------------
# 5. 真实 DSv3 层：ATTN/MLP/BOTH 选择器的 recomp 不变（12 锚点连续/多-island 都不动）
#    —— select_both 实为 [attn][fc] 两 island，但 max 落在 fc island == 旧合并值 → 锚点不动。
# ---------------------------------------------------------------------------

def _dsv3_layer_ops():
    """构造 DSv3 8L 真实 transformer 层（MLA + dense FFN）的 ResolvedOp（dp2/sp，与报告同口径）。"""
    from validate_dsv3 import build_dsv3_spec
    spec, d, fl = build_dsv3_spec(8)
    pc = ParallelConfig(dp_shard=2, cp=1, tp=1, ep=1, pp=1,
                        sequence_parallel=True, num_microbatches=1)
    world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
    pm = ParallelModel(pc, spec.dims.n_layers, world)
    g = ShapeEval().resolve(spec, pm)
    for _st, layers in g.stages.items():
        for l in layers:
            if l.layer_id == 1:
                return l.ops
    raise AssertionError("layer 1 not found")


def _matcher(selset):
    def is_sel(op):
        name = op.name.lower()
        typ = str(getattr(op.type, "value", op.type)).lower()
        return any(s.lower() in name or s.lower() in typ for s in selset)
    return is_sel


def test_dsv3_select_anchors_recomp_byte_identical():
    """真机锚点保护：DSv3 层在 ATTN/MLP/BOTH 选择器下，island 式 recomp 逐字节 == 旧合并式。
    block=512（报告口径）。select_both 是两 island 但 max 命中 fc island == 旧值 → 锚点不动。"""
    ops = _dsv3_layer_ops()
    ATTN = {"linear_q", "linear_kv", "q_a", "kv_a", "rope", "flash", "o_proj"}
    MLP = {"fc", "swiglu", "gelu", "router", "dispatch", "e_", "combine", "shared"}
    BOTH = ATTN | MLP
    for selset, exp_islands in ((ATTN, 1), (MLP, 1), (BOTH, 2)):
        is_sel = _matcher(selset)
        sel = estimate_select_memory(ops, is_sel, alloc_block_bytes=512)
        assert sel.n_islands == exp_islands
        assert sel.recomp_scratch == _old_block_recomp(ops, is_sel, blk=512)
