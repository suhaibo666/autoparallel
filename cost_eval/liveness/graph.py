"""显式**前向图 + 反向图**（设计要求 1/2）：从既有 op 清单导出逐张量的生死区间与 grad 可达性。

桶模型的天花板在于三个量（`activation_saves` / `forward_max_live` / `checkpoint_input`）都是
**闭式聚合**：`forward_max_live` 的活性区间止于「最后一次前向使用」，而重算下反向要求每个
backward 需要的张量活到 **backward 消费完**；补出来的 `remat_saves` 又与 `recomp_scratch` /
`bwd_working_set` 部分重叠、无可解析扣减 → 同一组公式在 1 层/stage 的 pp8 上**过读 1.1~1.3×**、
在 unfused pp4 上**欠读 ~25%**。本模块把它变成一条规则：

    **前向产出的张量，直到它的最后一个消费者（前向 *或* 反向）跑完才释放。**

这条规则**导出**重算工作集，而不是被公式告知。

── 结构 ────────────────────────────────────────────────────────────────────────────
前向节点直接来自 `LayerSpec.ops`（每个 `OpSpec` 的 `inputs`/`output`/`params`/`saves`/
`workspace`/`bwd_scratch`）。对每个前向 op 造一个反向节点，它消费该 op 的 saved 张量 + 入梯度，
产出各输入的梯度。

── grad 可达性（设计要求 2，最高价值项）────────────────────────────────────────────
张量**只有**在「某个从 loss 可达的反向节点确实会读它」时才需要保留：

  * **params 是梯度根**（`is_weight` → `requires_grad=True`）；
  * `stop_gradient` **切断**梯度路径 —— 由 `TensorRef.detached` 标注（新增、默认 False、
    对既有桶路径逐字节无影响）；detached 张量**不 requires_grad**、其下游亦不（除非另有
    requires_grad 的输入或权重重新引入）；
  * 一个 op **有反向节点** ⟺ 其任一非权重输入 requires_grad **或** 它带 params
    （params 是梯度根 → 即使输出是整型索引，也要跑 bprop 出 param 梯度）；
  * `saves` 是**手写候选清单**（`cost_eval/layers/*.py` 声明），liveness 按 grad 可达性
    **过滤**它：`kept_for_backward = {s ∈ saves(op) : has_bwd(op) 且 requires_grad(s)}`。

真机实例（**规则通用，不硬编码这一例**）：`csa.py:794-795` 的 unfused indexer KL loss 调用
``self.unfused_indexer_loss(index_scores, topk_idx, ops.stop_gradient(query),
ops.stop_gradient(compressed_kv))`` —— 其内部（`indexer.py:350` `matmul(query,key)*scale` 得
O(S·S/r) 张量、`:380` 对它做 fp32 `softmax`）两个输入都被 detach → 整条链**不建 autograd
节点** → 无反向节点读它 → 纯瞬态。census（`layers/dsv4_hybrid.py:207-210`）把这对张量
（`ukl1`/`ukl2`）声明成 `saves`；标上 `detached=True` 后本模块**导出**「不保留」。而
`CSAIndexer` 自己的 `index_scores`（`indexer.py:245`）经 indexer 自身 params 携带梯度 →
requires_grad → 照常保留。二者的区别由同一条规则分开，不靠特例。

⚠ 口径说明：这里的 `requires_grad` 语义是「**grad 可达 ⇒ 该 op 的 bprop 节点存在且会读它**」。
整型索引张量（如 `topk_indices`）数学上不可微，但 gather 的 bprop **确实**要读它 → 本规则
（其生产 op 在梯度路径上 ⇒ 保留）给出的结论与真机一致，故不额外区分 dtype 可微性。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..structure_mem import _align_up, _dt, _norm_save_names


# ---------------------------------------------------------------------------
# 张量 / 节点
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LTensor:
    """liveness 图里的一个张量（字节已按分配器块对齐，与桶路径同口径）。"""
    name: str
    nbytes_raw: int        # 自身 dtype 字节（数据流上的真实副本 / 区域边界 / 梯度同形）
    nbytes_saved: int      # saved 副本字节：norm op 保留的输入按 norm_compute_dtype（fp32 cast）
    is_weight: bool
    detached: bool         # 由 stop_gradient 后的输入喂出的子计算产物 → 无 autograd 节点
    internal: bool         # 只出现在 `saves`（op 内部中间量，非数据流边）
    requires_grad: bool = False

    @property
    def dtype_note(self) -> str:
        return "fp32-cast" if self.nbytes_saved != self.nbytes_raw else ""


@dataclass(frozen=True)
class FwdNode:
    """前向节点（= 一个 `OpSpec`）。"""
    idx: int
    op_name: str
    op_type: str
    reads: tuple           # 非权重输入名（去重、保序）
    produces: tuple        # output 名 + internal saves 名（本 op 新物化的张量）
    saves: tuple           # 经 grad 可达性过滤后、该 op 供反向保留的张量名
    workspace: int
    has_bwd: bool


@dataclass(frozen=True)
class BwdNode:
    """反向节点（对应前向 op `idx`）。"""
    idx: int
    op_name: str
    reads: tuple                 # 它读的 saved 张量名
    grad_out: tuple              # 它产出梯度的（非权重）输入张量名
    internal_grads: tuple        # grad_mode="chain2" 下的内部中间量梯度候选 [(name, nbytes), ...]
    scratch: int                 # op 自身反向临时物化（bwd_scratch_bytes）


@dataclass(frozen=True)
class LayerGraph:
    """一层（或一个伪层：embedding / lm_head / mtp）的前向 + 反向图与生死区间索引。"""
    layer_id: int
    layer_type: str
    tensors: dict                # name -> LTensor
    fwd: tuple                   # tuple[FwdNode]，前向序
    bwd: tuple                   # tuple[BwdNode]，**反向执行序**（op idx 递减）
    boundary: str = None         # 区域边界张量名（= structure_mem.checkpoint_input 所指）
    fwd_last_use: dict = field(default_factory=dict)   # name -> 前向最后一次出现的 op idx
    bwd_last_read: dict = field(default_factory=dict)  # name -> **最小**的读它的反向 op idx
    grad_born_at: dict = field(default_factory=dict)   # name -> grad(name) 诞生的反向 op idx（最大消费者）
    grad_free_after: dict = field(default_factory=dict)  # name -> grad(name) 释放于该反向 op idx 之后
    kept_for_backward: frozenset = frozenset()
    saved_by: dict = field(default_factory=dict)       # name -> frozenset(保留它的 op idx)
    entries: frozenset = frozenset()   # 图入口激活（层 construct 入参 / 上一 stage 送来的激活）

    def region_last_use(self, region: tuple) -> dict:
        """区域内最后一次使用位置：`name -> max{k ∈ region : name 在 op k 出现}`。

        重算再执行只跑 `region` 里的 op，故瞬态的死亡点是**区域内**的最后一次使用（而非整层）。"""
        rset = set(region)
        out: dict = {}
        for n in self.fwd:
            if n.idx not in rset:
                continue
            for name in (*n.reads, *n.produces):
                out[name] = n.idx
        return out


# ---------------------------------------------------------------------------
# 构图
# ---------------------------------------------------------------------------

def _dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return tuple(out)


def build_layer_graph(resolved_layer, *, alloc_block_bytes: int = 1,
                      norm_compute_dtype_bytes: int = 0,
                      grad_mode: str = "dataflow") -> LayerGraph:
    """把一个 `ResolvedLayer` 编译成 `LayerGraph`（前向图 + 反向图 + grad 可达性 + 生死索引）。

    参数
    ----
    alloc_block_bytes        : 分配器块对齐（与 `structure_mem` 逐张量同口径，可比）。
    norm_compute_dtype_bytes : norm op 保留输入的 fp32 cast 字节（与 `activation_saves` 同口径）。
    grad_mode                : ``"dataflow"``（默认，严格：只给**真实数据流边**建激活梯度）
                               / ``"chain2"``（另给 op **内部**中间量建梯度，按 window-2 相邻
                               共存模型取该反向节点所读内部张量的**两个最大者**——沿用
                               `structure_mem._backward_max_live` 已论证的相邻窗模型，
                               structure_mem.py:178-219，非新增拟合常数）。
    """
    ops = list(resolved_layer.ops)
    blk = alloc_block_bytes
    norm_names = (_norm_save_names(ops) if norm_compute_dtype_bytes else frozenset())

    # ── ① 张量清册：数据流边（inputs/output/params）+ 只在 saves 里出现的 op 内部中间量 ──────
    flow_names: set = set()
    for op in ops:
        for t in op.inputs:
            flow_names.add(t.name)
        flow_names.add(op.output.name)

    raw: dict = {}          # name -> ResolvedTensor（首见定型；P1-03 已保证同层同名同 numel）
    for op in ops:
        for t in (*op.inputs, op.output, *op.params, *op.saves):
            raw.setdefault(t.name, t)

    tensors: dict = {}
    for name, t in raw.items():
        nb_raw = _align_up(t.local_numel * t.dtype_bytes, blk)
        nb_saved = _align_up(
            t.local_numel * _dt(t, norm_names, norm_compute_dtype_bytes), blk)
        tensors[name] = LTensor(
            name=name, nbytes_raw=nb_raw, nbytes_saved=nb_saved,
            is_weight=t.is_weight, detached=getattr(t, "detached", False),
            internal=(name not in flow_names and not t.is_weight))

    # ── ② 生产者 / 消费者索引 ───────────────────────────────────────────────────────
    producers: dict = {}     # name -> [op idx...]（in-place op 会让同名出现多次）
    for i, op in enumerate(ops):
        producers.setdefault(op.output.name, []).append(i)
        for s in op.saves:
            if tensors[s.name].internal:
                producers.setdefault(s.name, []).append(i)
    # 未被任何 op 产出的非权重输入 = 图入口（层入口 hidden / 上一 stage 送来的激活）。
    entries = {t.name for op in ops for t in op.inputs
               if not t.is_weight and t.name not in producers}

    # ── ③ grad 可达性（不动点迭代；in-place op 的自环由不动点天然消化）────────────────────
    rg: dict = {}
    for name, lt in tensors.items():
        if lt.detached:
            rg[name] = False                      # stop_gradient 切断
        elif lt.is_weight:
            rg[name] = True                       # params 是梯度根
        elif name in entries:
            rg[name] = True                       # 图入口激活（pp 反向经 P2P 回传梯度）
        else:
            rg[name] = False
    changed = True
    while changed:
        changed = False
        for i, op in enumerate(ops):
            src = any(rg[t.name] for t in op.inputs) or bool(op.params)
            if not src:
                continue
            outs = [op.output.name] + [s.name for s in op.saves
                                       if tensors[s.name].internal]
            for name in outs:
                if not tensors[name].detached and not rg[name]:
                    rg[name] = True
                    changed = True
    tensors = {n: LTensor(**{**lt.__dict__, "requires_grad": rg[n]})
               for n, lt in tensors.items()}

    # ── ④ 反向节点存在性 + saves 过滤 ────────────────────────────────────────────────
    has_bwd = []
    for op in ops:
        has_bwd.append(any(rg[t.name] for t in op.inputs) or bool(op.params))

    kept: set = set()
    saved_by: dict = {}
    for i, op in enumerate(ops):
        if not has_bwd[i]:
            continue
        for s in op.saves:
            if s.is_weight or not rg[s.name]:
                continue                          # detached / 不在梯度路径上 → 反向无节点读它
            kept.add(s.name)
            saved_by.setdefault(s.name, set()).add(i)

    # ── ⑤ 节点 ────────────────────────────────────────────────────────────────────
    fwd = []
    for i, op in enumerate(ops):
        reads = _dedup(t.name for t in op.inputs if not t.is_weight)
        prod = [op.output.name] + [s.name for s in op.saves
                                   if tensors[s.name].internal]
        fwd.append(FwdNode(
            idx=i, op_name=op.name, op_type=str(getattr(op, "type", "")),
            reads=reads, produces=_dedup(prod),
            saves=_dedup(s.name for s in op.saves
                         if not s.is_weight and s.name in kept and i in saved_by.get(s.name, ())),
            workspace=getattr(op, "workspace_bytes", 0), has_bwd=has_bwd[i]))

    # 梯度的生死：grad(t) 由**最后执行**的消费者反向节点诞生（反向序 = op idx 递减 →
    # 最大 idx 的消费者最先跑），由 t 的**首个**生产者的反向节点消费后释放。
    grad_consumers: dict = {}        # name -> [消费它的 op idx...]
    for i, op in enumerate(ops):
        if not has_bwd[i]:
            continue
        for t in op.inputs:
            if t.is_weight or not rg[t.name]:
                continue
            grad_consumers.setdefault(t.name, []).append(i)
    grad_born_at = {n: max(v) for n, v in grad_consumers.items()}
    grad_free_after: dict = {}
    for n in grad_consumers:
        prods = [p for p in producers.get(n, []) if has_bwd[p]]
        # 生产者的反向节点消费它；无生产者（图入口）→ 活到整层反向结束（idx 0 之后）。
        grad_free_after[n] = min(prods) if prods else 0

    bwd = []
    for i in range(len(ops) - 1, -1, -1):
        if not has_bwd[i]:
            continue
        op = ops[i]
        reads = fwd[i].saves
        internal_grads: tuple = ()
        if grad_mode == "chain2":
            cand = sorted((tensors[n].nbytes_raw, n) for n in reads
                          if tensors[n].internal and tensors[n].requires_grad)
            internal_grads = tuple((n, nb) for nb, n in cand[-2:])
        bwd.append(BwdNode(
            idx=i, op_name=op.name, reads=reads,
            grad_out=_dedup(t.name for t in op.inputs
                            if not t.is_weight and rg[t.name]),
            internal_grads=internal_grads,
            scratch=getattr(op, "bwd_scratch_bytes", 0)))

    # ── ⑥ 生死索引 ───────────────────────────────────────────────────────────────
    fwd_last_use: dict = {}
    for n in fwd:
        for name in (*n.reads, *n.produces):
            fwd_last_use[name] = n.idx
    bwd_last_read: dict = {}         # 反向序递减 → 最后跑的消费者 = 最小 idx
    for b in bwd:
        for name in b.reads:
            bwd_last_read[name] = min(bwd_last_read.get(name, b.idx), b.idx)

    # ── ⑦ 区域边界（= structure_mem.checkpoint_input 的所指：层 construct 入参）────────
    boundary = None
    for op in ops:
        if op.saves:
            b = next((t for t in op.inputs if not getattr(t, "is_weight", False)), None)
            boundary = (b or op.saves[0]).name
            break

    return LayerGraph(
        layer_id=resolved_layer.layer_id, layer_type=resolved_layer.layer_type,
        tensors=tensors, fwd=tuple(fwd), bwd=tuple(bwd), boundary=boundary,
        fwd_last_use=fwd_last_use, bwd_last_read=bwd_last_read,
        grad_born_at=grad_born_at, grad_free_after=grad_free_after,
        kept_for_backward=frozenset(kept),
        saved_by={n: frozenset(v) for n, v in saved_by.items()},
        entries=frozenset(entries))


def build_stage_graphs(spec, *, pp: int = 1, m: int = 1, tp: int = 1, cp: int = 1,
                       ep: int = 1, dp_shard: int = 1, alloc_block_bytes: int = 1,
                       norm_compute_dtype_bytes: int = 0,
                       grad_mode: str = "dataflow",
                       graph_source: str = "hand_spec") -> dict:
    """便捷入口：`ModelSpec` → `{stage: [LayerGraph, ...]}`（供图层面的单测/诊断直用）。

    `graph_source` 见 `liveness/sources.py`；默认 `"hand_spec"` 与改造前逐字节相同。"""
    from ..parallel_model import ParallelModel
    from ..specs import ParallelConfig
    from .sources import resolve_graph
    pc = ParallelConfig(pp=pp, num_microbatches=m, tp=tp, cp=cp, ep=ep,
                        dp_shard=dp_shard, sequence_parallel=(tp > 1))
    world = dp_shard * cp * tp * pp
    pm = ParallelModel(pc, spec.dims.n_layers, world)
    g = resolve_graph(spec, pm, graph_source)
    return {st: [build_layer_graph(l, alloc_block_bytes=alloc_block_bytes,
                                  norm_compute_dtype_bytes=norm_compute_dtype_bytes,
                                  grad_mode=grad_mode)
                 for l in layers]
            for st, layers in g.stages.items()}
