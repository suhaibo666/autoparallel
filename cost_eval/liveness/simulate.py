"""逐张量 liveness 仿真器 —— **只读交叉校验**，非权威口径（设计要求 6）。

与 `mem_timeline.MemTimeline`（桶模型）的关系：
  * **完全旁路**：不 import 桶路径的任何桶计算、不改它一个字节；`Evaluator` 的数值不变。
  * **同事件网格**：走同一份 `cost_eval/schedule.py` 事件代数（`build_1f1b` / VPP），事件标签
    与桶模型一致（`fwd:{lid}` / `fwd_end` / `bwd@{lid}` / `optstep`）→ 可逐事件对账。
  * **同非 liveness 层**（设计要求 5）：`persistent` / `gather_buf`（FSDP 预取+resident+全重算
    re-gather）/ `grad_buf` / `grad_accum` / `bwd_scratch` / `p2p_buf` / `swap_buf` / `optstep`
    / `kept_frag` / `framework_reserve` 与块对齐**沿用桶模型的既有公式与标定**，原样保留。
  * **只替换激活记账**：桶模型的 `act_live` / `recomp_scratch` / `bwd_working_set` /
    `remat_saves` 四桶（彼此部分重叠、无可解析扣减）由**一个 live-set**取代：
    ``peak = max over schedule of Σ 在世张量字节``，每张量恰算一次。

── 重算 = 图变换（设计要求 3）────────────────────────────────────────────────────
对一个 checkpoint 区域：前向末**丢掉**它的 saved 张量（只留区域边界 `checkpoint_input`）；在该
区域 backward **之前**插入区域前向子图的**再执行**。之后 liveness 规则自动给出「再物化集」与
「瞬态重叠」，无双算。`mode: full`（区域 = 整层）与 `mode: select`（区域 = 连续选中 op 岛，
`structure_mem._checkpoint_islands`）同构处理。

── 真机不变量（167/MS2.10，2026-07-25）──────────────────────────────────────────
  * `rc=ON` 前向末残留 ≡ 0（区域内部 saved 全释放）→ 本模型前向末只留 `act_boundary`；
  * `rc=ON` 前向峰与微批数 N **无关**（552/403 MiB 恒定）→ 再执行发生在**该区域自己的反向步**，
    同一时刻只有一个区域在重算 → ×1 自动成立（不做任何 stage 级求和/取 max 的公式选择）；
  * 多卡 A/B：stage3 的 unfused−fused 差在 L8(2 层/stage) 与 L4(1 层/stage) 同为 24380.1 MiB
    → ×1 于层数亦自动成立。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..framework import framework_reserve as _framework_reserve
from ..parallel_model import ParallelModel
from ..schedule import (_1f1b_from_warmup, build_interleaved_1f1b,
                        interleaved_virtual_order)
from ..shape_eval import ShapeEval
from ..specs import _MUON_NS_WORKSPACE_MULT, is_attn_projection, is_muon_matrix_weight
from ..static_mem import StaticMem
from ..structure_mem import (_checkpoint_islands, _fsdp_local_count,
                             estimate_structure_memory)
from .categories import LIVENESS_CATEGORIES, NON_LIVENESS_BUCKETS, to_buckets
from .graph import build_layer_graph

__all__ = ["simulate_liveness", "LivenessResult", "StageLiveness", "LiveItem",
           "LiveSample"]


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiveItem:
    """峰值时刻 live-set 里的一项（逐张量，附类别标签）。"""
    key: tuple             # (mb, layer_id, name) 或 ("grad", mb, layer_id, name)
    name: str
    layer_id: int
    mb: int
    nbytes: int
    category: str          # categories.LIVENESS_CATEGORIES 之一

    @property
    def is_grad(self) -> bool:
        return self.key[0] == "grad"


@dataclass(frozen=True)
class LiveSample:
    """时间线采样点（每个层级事件取其内部微步的**最大**总量，事件网格与桶模型一致）。"""
    idx: int
    event: str
    mb: int
    chunk: int
    total_bytes: int
    buckets: dict          # 类别/桶名 -> 字节
    n_live: int
    substep: str = ""      # 峰值所在微步（如 "op7:flash" / "rerun3:qkv" / "bwd5:fc1"）


@dataclass(frozen=True)
class StageLiveness:
    """单 stage 的 liveness 结果。"""
    stage: int
    peak_bytes: int
    peak_event: str
    peak_substep: str
    peak_mb: int
    buckets: dict                  # 峰值时刻 {类别/桶名: 字节}
    live_set: tuple                # tuple[LiveItem]，峰值时刻逐张量
    timeline: tuple = ()
    liveness_categories: tuple = LIVENESS_CATEGORIES
    # 全时间线上「重算工作集」（`recomp_saved + recomp_transient`）的最大值 —— 三条真机不变量
    # （驻留→0 / ×1 于 m / ×1 于层数）直接作用在这个量上，故单独暴露（峰值时刻可能不在重算微步）。
    max_recompute_working_set: int = 0
    max_recompute_substep: str = ""
    max_recompute_live_set: tuple = ()

    def bucket_view(self) -> dict:
        """峰值 breakdown 折算到 `mem_timeline.Buckets` 的桶名（供与桶模型/真机逐桶对账）。"""
        return to_buckets(self.buckets)

    def top_live(self, n: int = 20) -> tuple:
        return tuple(sorted(self.live_set, key=lambda it: -it.nbytes)[:n])


@dataclass(frozen=True)
class LivenessResult:
    per_stage: tuple
    tightest_stage: int
    grad_mode: str
    hccl_reserved_bytes: int = 0
    max_device_memory: int = 0

    def peak_bytes(self, stage: int) -> int:
        return self.per_stage[stage].peak_bytes

    def device_peak(self) -> int:
        return max(p.peak_bytes for p in self.per_stage)

    def reserved_estimate_bytes(self, stage: int) -> int:
        """与 `report.PeakMemoryReport.reserved_estimate_bytes` 同口径（设计要求 5：
        非 liveness 层原样保留）：allocated 峰 + HCCL 通信缓冲 + allocator pool 碎片近似。"""
        from ..framework import allocator_pool_fragmentation
        peak = self.per_stage[stage].peak_bytes
        return peak + self.hccl_reserved_bytes + allocator_pool_fragmentation(None, peak)


# ---------------------------------------------------------------------------
# live-set 容器
# ---------------------------------------------------------------------------

class _LiveSet:
    """当前在世张量集合：key -> (bytes, category, name, layer_id, mb)。"""

    __slots__ = ("_it", "_total", "_by_cat")

    def __init__(self):
        self._it: dict = {}
        self._total = 0
        self._by_cat: dict = {}

    def add(self, key, nbytes, category, name, lid, mb) -> None:
        if key in self._it:                       # 已在世（in-place / 多消费者）→ 不重复分配
            return
        self._it[key] = (nbytes, category, name, lid, mb)
        self._total += nbytes
        self._by_cat[category] = self._by_cat.get(category, 0) + nbytes

    def retag(self, key, category) -> None:
        """就地改类别（如前向瞬态在层末变成驻留 saved / 边界）。"""
        cur = self._it.get(key)
        if cur is None or cur[1] == category:
            return
        nbytes, _old, name, lid, mb = cur
        self._by_cat[_old] -= nbytes
        self._it[key] = (nbytes, category, name, lid, mb)
        self._by_cat[category] = self._by_cat.get(category, 0) + nbytes

    def drop(self, key) -> None:
        cur = self._it.pop(key, None)
        if cur is None:
            return
        self._total -= cur[0]
        self._by_cat[cur[1]] -= cur[0]

    def has(self, key) -> bool:
        return key in self._it

    @property
    def total(self) -> int:
        return self._total

    def by_cat(self) -> dict:
        return {k: v for k, v in self._by_cat.items() if v}

    def keys_of(self, mb, lid):
        return [k for k, v in self._it.items() if v[4] == mb and v[3] == lid]

    def bytes_of_layers(self, lids) -> int:
        return sum(v[0] for v in self._it.values() if v[3] in lids)

    def snapshot(self) -> tuple:
        return tuple(LiveItem(key=k, name=v[2], layer_id=v[3], mb=v[4],
                              nbytes=v[0], category=v[1])
                     for k, v in self._it.items())


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def simulate_liveness(model_spec, parallel_config, optimizer, hardware,
                      recompute, swap, *, record_timeline: bool = False,
                      grad_mode: str = "dataflow") -> LivenessResult:
    """给定与 `Evaluator` 完全相同的一组配置，跑逐张量 liveness 仿真。

    参数
    ----
    grad_mode : ``"dataflow"``（默认）只对**真实数据流边**建激活梯度（严格可导出、不假设
        op 内部结构）；``"chain2"`` 额外给 op **内部**中间量建梯度，按 window-2 相邻共存取每
        反向节点所读内部张量的两个最大者（沿用 `structure_mem._backward_max_live` 的相邻窗
        模型，structure_mem.py:178-219）——用于量化「census 把一条 11-op 小算子链塌成单个 op
        的 `saves` 平表后，反向梯度链**不可导出**」这个缺口的规模。
    """
    if grad_mode not in ("dataflow", "chain2"):
        raise ValueError(f"grad_mode={grad_mode!r} 非法：只支持 'dataflow' / 'chain2'。")
    pc = parallel_config
    world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
    pm = ParallelModel(pc, model_spec.dims.n_layers, world)
    g = ShapeEval().resolve(model_spec, pm)
    blk = getattr(hardware, "alloc_block_bytes", 1)
    persistent = StaticMem().compute(
        g, optimizer, pm, alloc_block_bytes=blk,
        offload_params=pc.offload_params, offload_optimizer=pc.offload_optimizer)
    fr = _framework_reserve(pc, hardware.framework_reserve)
    dims = model_spec.dims
    sim = _StageSim(
        g=g, pm=pm, recompute=recompute, swap=swap, optimizer=optimizer,
        hardware=hardware, dims=dims, persistent=persistent,
        framework_reserve=fr, alloc_block_bytes=blk, grad_mode=grad_mode,
        record_timeline=record_timeline)
    per_stage = tuple(sim.run(st) for st in sorted(g.stages))
    tightest = max(per_stage, key=lambda p: p.peak_bytes).stage
    from ..framework import hccl_reserved_buffer
    return LivenessResult(per_stage=per_stage, tightest_stage=tightest,
                          grad_mode=grad_mode,
                          hccl_reserved_bytes=hccl_reserved_buffer(pc),
                          max_device_memory=hardware.max_device_memory)


# ---------------------------------------------------------------------------
# per-stage 仿真
# ---------------------------------------------------------------------------

@dataclass
class _StageSim:
    g: object
    pm: object
    recompute: object
    swap: object
    optimizer: object
    hardware: object
    dims: object
    persistent: dict
    framework_reserve: int
    alloc_block_bytes: int
    grad_mode: str
    record_timeline: bool
    _K_OPT: int = 4                    # 与 mem_timeline 同（P0-01 重标 6→4）

    # -- 事件循环 ------------------------------------------------------------
    def run(self, stage: int) -> StageLiveness:
        pm, rc, swap = self.pm, self.recompute, self.swap
        blk = self.alloc_block_bytes
        pp, m = pm.degree("pp"), pm.pc.num_microbatches
        v = getattr(pm.pc, "interleave", 1)
        depth = getattr(pm.pc, "prefetch_depth", 1)
        swap_depth = getattr(swap, "default_prefetch", 1)
        offload_optimizer = getattr(pm.pc, "offload_optimizer", pm.pc.cpu_offload)
        offload_grads = getattr(pm.pc, "offload_grads", pm.pc.cpu_offload)
        fsdp_d, efsdp_d = pm.dense_fsdp_degree(), pm.efsdp_degree()
        muon = str(getattr(self.optimizer, "type", "")).lower() == "muon"
        ns_mult = getattr(self.optimizer, "ns_workspace_mult", _MUON_NS_WORKSPACE_MULT)
        norm_dt = getattr(self.dims, "norm_compute_dtype_bytes", 0)

        layers = self.g.stages[stage]
        layer_ids = [l.layer_id for l in layers]
        by_id = {l.layer_id: l for l in layers}
        sm_by_id = {
            l.layer_id: estimate_structure_memory(
                l.ops, grad_dtype_bytes=getattr(self.optimizer, "grad_dtype_bytes", 4),
                fsdp=fsdp_d, efsdp=efsdp_d, alloc_block_bytes=blk,
                norm_compute_dtype_bytes=norm_dt,
                bwd_scratch_conservative=getattr(
                    self.hardware, "bwd_scratch_conservative", False),
                ep_degree=pm.degree("ep"))
            for l in layers}
        lg_by_id = {l.layer_id: build_layer_graph(
            l, alloc_block_bytes=blk, norm_compute_dtype_bytes=norm_dt,
            grad_mode=self.grad_mode) for l in layers}
        # 重算区域（islands）：full → 整层一个岛；select → 连续选中段；None → 无岛。
        regions_by_id = {lid: self._regions(by_id[lid]) for lid in layer_ids}
        offloaded = {lid for lid in layer_ids
                     if swap.swaps(lid) and not rc.is_full(lid) and not rc.is_select(lid)}
        loss_lids = (set() if getattr(self.dims, "cross_entropy_fused", False) else
                     {l.layer_id for l in layers
                      if any(getattr(op, "name", "") == "nll" for op in l.ops)})
        stage_no_recompute = not any(rc.is_full(l.layer_id) or rc.is_select(l.layer_id)
                                     for l in layers)
        moe_lids = {l.layer_id for l in layers
                    if any(getattr(op.type, "value", op.type) == "moe_gemm" for op in l.ops)}
        pp_full_recomp = pp > 1
        # Fix①（mem_timeline §8.5）：全重算 backward 先重跑 forward → 须 re-gather 参数，
        #   PP 下 checkpoint 区整段重算、区内各全重算层 param 在反向峰同驻。原样沿用。
        recomp_regather = (sum(sm_by_id[l.layer_id].param_full_bytes
                               for l in layers if rc.is_full(l.layer_id))
                           if pp_full_recomp else 0)
        muon_ns_overlap = 0
        if muon and pp_full_recomp and recomp_regather > 0:
            muon_ns_overlap = max(
                (round(ns_mult * _fsdp_local_count(
                    w, efsdp_d if getattr(w, "is_expert", False) else fsdp_d,
                    pm.degree("ep")) * 4)
                 for l in layers for op in l.ops for w in op.params
                 if is_muon_matrix_weight(op.type, getattr(op, "name", ""))), default=0)

        _pol = getattr(pm.pc, "reshard_after_forward", "default")
        if fsdp_d > 1 or efsdp_d > 1:
            if _pol == "never":
                no_reshard = set(layer_ids)
            elif _pol == "always":
                no_reshard = set()
            else:
                no_reshard = set(layer_ids) if pp > 1 else set(loss_lids)
        else:
            no_reshard = set()
        resident_gather: dict = {}
        p2p_send = 0
        if pp > 1 and stage < pp - 1 and layer_ids:
            _ov = 2 if getattr(pm.pc, "pipeline_parallel_overlap_p2p", False) else 1
            p2p_send = sm_by_id[max(layer_ids)].checkpoint_input * _ov

        # ── 状态 ───────────────────────────────────────────────────────────────
        live = _LiveSet()
        nb: dict = {k: 0 for k in NON_LIVENESS_BUCKETS}
        nb["persistent"] = self.persistent.get(stage, 0)
        nb["framework"] = self.framework_reserve
        kept_lids = {lid for lid in layer_ids
                     if rc.is_select(lid) and lid not in loss_lids
                     and not (set(rc.selectors(lid)) & _FFN_MARKERS)}
        kept_frag_factor = getattr(self.dims, "kept_frag_factor", 0.0)
        nr_moe_factor = getattr(self.dims, "nr_moe_frag_factor", 0.0)
        nr_moe_lids = {lid for lid in moe_lids
                       if lid not in loss_lids and not rc.is_full(lid)
                       and not rc.is_select(lid) and not swap.swaps(lid)}

        peak = {"bytes": -1, "event": "", "substep": "", "mb": -1,
                "buckets": {}, "live": ()}
        rcw = {"bytes": 0, "substep": "", "live": ()}
        series: list = []
        cur_event = {"label": "", "mb": -1, "chunk": -1, "best": -1,
                     "substep": "", "buckets": {}, "n": 0}

        def _res() -> int:
            return sum(resident_gather.values())

        def _prefetch(order, idx, d) -> int:
            tot = 0
            for j in range(idx + 1, min(idx + 1 + d, len(order))):
                if order[j] not in resident_gather:
                    tot += sm_by_id[order[j]].param_full_bytes
            return tot

        def _prefetch_swap(order, idx, d) -> int:
            tot = 0
            for j in range(idx + 1, min(idx + 1 + d, len(order))):
                if order[j] in offloaded:
                    tot += sm_by_id[order[j]].activation_saves
            return tot

        def sample(substep: str, extra: int = 0, extra_cat: str = "") -> None:
            """记一个微步采样点（`extra` = 该微步专属的瞬态，如 workspace / 内部梯度）。"""
            cats = live.by_cat()
            if extra:
                cats = dict(cats)
                cats[extra_cat] = cats.get(extra_cat, 0) + extra
            total = live.total + extra + sum(nb.values())
            if total > cur_event["best"]:
                cur_event["best"] = total
                cur_event["substep"] = substep
                cur_event["buckets"] = {**{k: v for k, v in nb.items() if v}, **cats}
                cur_event["n"] = len(live.snapshot())
            if total > peak["bytes"]:
                peak.update(bytes=total, event=cur_event["label"], substep=substep,
                            mb=cur_event["mb"],
                            buckets={**{k: v for k, v in nb.items() if v}, **cats},
                            live=live.snapshot())
            _rw = cats.get("recomp_saved", 0) + cats.get("recomp_transient", 0)
            if _rw > rcw["bytes"]:
                rcw.update(bytes=_rw, substep=f'{cur_event["label"]}/{substep}',
                           live=live.snapshot())

        def open_event(label: str, mb: int = -1, chunk: int = -1) -> None:
            cur_event.update(label=(f"{label}#c{chunk}" if chunk >= 0 else label),
                             mb=mb, chunk=chunk, best=-1, substep="", buckets={}, n=0)

        def close_event() -> None:
            if cur_event["best"] < 0:
                return
            if self.record_timeline:
                series.append(LiveSample(
                    idx=len(series), event=cur_event["label"], mb=cur_event["mb"],
                    chunk=cur_event["chunk"], total_bytes=cur_event["best"],
                    buckets=dict(cur_event["buckets"]), n_live=cur_event["n"],
                    substep=cur_event["substep"]))

        # ── 调度（与桶模型同一份 schedule.py）────────────────────────────────────
        if v > 1:
            chunks = pm.stage_chunks(stage)
            steps = [(k, mbi, chunks[c], c)
                     for k, mbi, c in interleaved_virtual_order(stage, pp, m, v, pp)]
        elif getattr(pm.pc, "sched_warmup_plus_one", False) and pp > 1 and stage < pp - 1:
            steps = [(e.kind, e.mb, layer_ids, -1)
                     for e in _1f1b_from_warmup(min(pp - stage, m), m)]
        else:
            steps = [(e.kind, e.mb, layer_ids, -1)
                     for e in build_interleaved_1f1b(stage, pp, m, v)]

        grad_done: set = set()

        for ev_kind, ev_mb, ev_layers, ev_chunk in steps:
            if ev_kind == "FWD":
                for idx, lid in enumerate(ev_layers):
                    sm, lg = sm_by_id[lid], lg_by_id[lid]
                    if lid in no_reshard:
                        resident_gather[lid] = sm.param_full_bytes
                        nb["gather_buf"] = _res() + _prefetch(ev_layers, idx, depth)
                    else:
                        nb["gather_buf"] = (_res() + sm.param_full_bytes
                                            + _prefetch(ev_layers, idx, depth))
                    _resd, _bonly = self._resident_after_fwd(lg, lid)
                    open_event(f"fwd:{lid}", ev_mb, ev_chunk)
                    self._run_forward(live, lg, ev_mb, lid, sample,
                                      resident=_resd, boundary_only=_bonly)
                    close_event()
                    nb["gather_buf"] = _res()
                nb["p2p_buf"] = p2p_send
                open_event("fwd_end", ev_mb, ev_chunk)
                sample("fwd_end")
                close_event()
            else:                                   # BWD（逆序层）
                nb["p2p_buf"] = 0
                order = list(reversed(ev_layers))
                for idx, lid in enumerate(order):
                    sm, lg = sm_by_id[lid], lg_by_id[lid]
                    regather = 0 if lid in resident_gather else sm.param_full_bytes
                    nb["gather_buf"] = (_res() + regather
                                        + _prefetch(order, idx, depth) + recomp_regather)
                    nb["grad_buf"] = sm.grad_full_bytes
                    nb["swap_buf"] = ((sm.activation_saves if lid in offloaded else 0)
                                      + _prefetch_swap(order, idx, swap_depth))
                    nb["optstep"] = muon_ns_overlap
                    # unfused CE 链 K_CE 份满 vocab fp32（与 mem_timeline 同门同式）。
                    ce_lean = getattr(self.dims, "ce_pynative_lean", False)
                    k_ce = (4 if ce_lean else (8 if pp > 1 else 4))
                    if kept_frag_factor and lid in loss_lids:
                        ka = live.bytes_of_layers(kept_lids)
                        nb["kept_frag"] = round(kept_frag_factor * ka) if ka else 0
                    if (nr_moe_factor and pp == 1 and stage_no_recompute
                            and lid in loss_lids):
                        na = live.bytes_of_layers(nr_moe_lids)
                        nb["kept_frag"] += round(nr_moe_factor * na) if na else 0
                    open_event(f"bwd@{lid}", ev_mb, ev_chunk)
                    self._run_backward(live, lg, ev_mb, lid, sample, nb,
                                       regions=regions_by_id[lid],
                                       resident=self._resident_after_fwd(lg, lid)[0],
                                       loss_layer=(lid in loss_lids
                                                   and stage_no_recompute),
                                       k_ce=k_ce, layer_bwd_scratch=sm.bwd_scratch)
                    close_event()
                    nb["grad_buf"] = nb["swap_buf"] = nb["optstep"] = 0
                    nb["bwd_scratch"] = nb["kept_frag"] = 0
                    resident_gather.pop(lid, None)
                    nb["gather_buf"] = _res()
                    for k in live.keys_of(ev_mb, lid):
                        live.drop(k)
                    if lid not in grad_done:
                        grad_done.add(lid)
                        if not offload_grads:
                            nb["grad_accum"] += sm.grad_shard_bytes

        # ── optimizer-step 事件（与桶模型同式：K_OPT×最大权重分片 fp32 / Muon NS）────────
        def _shard(w):
            return _fsdp_local_count(
                w, efsdp_d if getattr(w, "is_expert", False) else fsdp_d, pm.degree("ep"))

        if offload_optimizer:
            optstep = 0
        elif muon:
            def _mt(op, w):
                sh = _shard(w)
                if is_muon_matrix_weight(op.type, getattr(op, "name", "")):
                    unit = sh
                    if (getattr(self.optimizer, "per_head", False)
                            and getattr(self.dims, "n_heads", 0) > 1
                            and is_attn_projection(getattr(op, "name", ""))):
                        unit = max(1, sh // getattr(self.dims, "n_heads", 1))
                    return round(ns_mult * unit * 4)
                return self._K_OPT * sh * 4
            optstep = max((_mt(op, w) for l in layers for op in l.ops for w in op.params),
                          default=0)
        else:
            optstep = self._K_OPT * 4 * max(
                (_shard(w) for l in layers for op in l.ops for w in op.params), default=0)
        if optstep > 0:
            for k in ("gather_buf", "grad_buf", "bwd_scratch", "swap_buf",
                      "workspace", "kept_frag", "p2p_buf"):
                nb[k] = 0
            nb["optstep"] = optstep
            open_event("optstep")
            sample("optstep")
            close_event()
            nb["optstep"] = 0

        return StageLiveness(
            stage=stage, peak_bytes=peak["bytes"], peak_event=peak["event"],
            peak_substep=peak["substep"], peak_mb=peak["mb"],
            buckets=peak["buckets"], live_set=peak["live"], timeline=tuple(series),
            max_recompute_working_set=rcw["bytes"],
            max_recompute_substep=rcw["substep"],
            max_recompute_live_set=rcw["live"])

    # -- 重算区域 -----------------------------------------------------------
    def _regions(self, layer) -> tuple:
        """该层的 checkpoint 区域（op idx 元组的元组）。full → 整层；select → 连续选中岛。"""
        rc, lid = self.recompute, layer.layer_id
        ops = list(layer.ops)
        if rc.is_full(lid):
            return (tuple(range(len(ops))),)
        if rc.is_select(lid):
            idx_of = {id(op): i for i, op in enumerate(ops)}
            islands = _checkpoint_islands(
                ops, lambda op: rc.op_matches(lid, op.name,
                                              getattr(op.type, "value", op.type)))
            return tuple(tuple(idx_of[id(op)] for op in isl) for isl in islands)
        return ()

    def _resident_after_fwd(self, lg, lid) -> tuple:
        """该层前向末**驻留**的张量名 + 其中「仅作为区域边界驻留」的子集（重算 = 图变换的前向侧）。

        - 无重算：全部 `kept_for_backward` ∪ {区域边界}；
        - full 重算：**只有区域边界**（真机 rc=ON `fwd_end ≡ 0`：区域内部 saved 全释放）；
        - select：被**非选中** op 保留的那些 ∪ {边界}（选中 op 的独占 saves 丢弃）；
        - swap：全部卸载到 CPU → 空集（反向经 swap_buf 复原，与桶路径同口径）。

        返回 `(resident, boundary_only)`。`boundary_only` 里的张量按**自身 dtype**（bf16 层入口）
        计字节——与 `structure_mem.checkpoint_input` 同口径；其余驻留量按 saved 口径（norm 保留
        输入的 fp32 cast），与 `activation_saves` 同口径。
        """
        rc, swap = self.recompute, self.swap
        b = frozenset([lg.boundary]) if lg.boundary else frozenset()
        if rc.is_full(lid):
            return b, b
        if rc.is_select(lid):
            sel = set()
            for reg in self._regions_cache(lg, lid):
                sel |= set(reg)
            nonsel_kept = {n for n, owners in lg.saved_by.items()
                           if n in lg.kept_for_backward and (owners - sel)}
            return frozenset(nonsel_kept) | b, b - frozenset(nonsel_kept)
        if swap.swaps(lid):
            return frozenset(), frozenset()
        kept = frozenset(lg.kept_for_backward)
        return kept | b, b - kept

    def _regions_cache(self, lg, lid) -> tuple:
        rc = self.recompute
        if not rc.is_select(lid):
            return ()
        # lg 不带 raw ops，故从 fwd 节点的 op 名/类型重算选中集（与 _regions 同判据）。
        sel_flags = [rc.op_matches(lid, n.op_name, n.op_type) for n in lg.fwd]
        out, cur = [], []
        for i, f in enumerate(sel_flags):
            if f:
                cur.append(i)
            elif cur:
                out.append(tuple(cur))
                cur = []
        if cur:
            out.append(tuple(cur))
        return tuple(out)

    # -- 前向 ---------------------------------------------------------------
    def _run_forward(self, live, lg, mb, lid, sample, resident, boundary_only) -> None:
        """跑该层前向微步：逐 op 分配产出、在**最后一次使用**后释放非保留量，层末只留 resident。"""
        def key(name):
            return (mb, lid, name)

        def alloc(name):
            t = lg.tensors[name]
            if t.is_weight:
                return
            nbytes = (t.nbytes_saved if name in resident and name not in boundary_only
                      else t.nbytes_raw)
            live.add(key(name), nbytes, self._fwd_category(lg, name, resident), name, lid, mb)

        # 图入口激活（层 construct 入参 / 上一 stage 送来的 hidden）——重算区域的边界就在这里。
        for name in lg.entries:
            alloc(name)
        for n in lg.fwd:
            for name in n.produces:
                alloc(name)
            sample(f"op{n.idx}:{n.op_name}", n.workspace, "workspace")
            # 该 op 之后死亡的张量：前向最后一次使用已过、且不被任何反向节点读、且不驻留。
            for name in (*n.reads, *n.produces):
                if (lg.fwd_last_use.get(name) == n.idx
                        and name not in resident
                        and name not in lg.bwd_last_read):
                    live.drop(key(name))
        # 层末：丢掉非 resident 的一切（重算 = 图变换的前向侧）。
        for k in live.keys_of(mb, lid):
            if k[2] not in resident:
                live.drop(k)
            else:
                live.retag(k, self._fwd_category(lg, k[2], resident))

    def _fwd_category(self, lg, name, resident) -> str:
        if name == lg.boundary and name in resident:
            return "act_boundary"
        if name in resident:
            return "act_saved"
        return "fwd_transient"

    # -- 反向 ---------------------------------------------------------------
    def _run_backward(self, live, lg, mb, lid, sample, nb, regions, resident,
                      loss_layer: bool, k_ce: int, layer_bwd_scratch: int) -> None:
        """跑该层反向微步：区域重算（图变换）+ 反向节点 + 激活梯度，逐张量按最后消费者释放。"""
        def key(name):
            return (mb, lid, name)

        def gkey(name):
            return ("grad", mb, lid, name)

        rerun_at: dict = {}
        for reg in regions:
            if reg:
                rerun_at[max(reg)] = reg
        bwd_by_idx = {b.idx: b for b in lg.bwd}

        for i in range(len(lg.fwd) - 1, -1, -1):
            if i in rerun_at:
                self._rerun_region(live, lg, mb, lid, sample, rerun_at[i], resident)
            b = bwd_by_idx.get(i)
            if b is None:
                continue
            # 该反向节点产出的激活梯度（同形、同 dtype；由最后执行的消费者诞生）。
            for name in b.grad_out:
                if lg.grad_born_at.get(name) != i:
                    continue
                live.add(gkey(name), lg.tensors[name].nbytes_raw, "grad_act", name, lid, mb)
            scratch = b.scratch
            if loss_layer and layer_bwd_scratch > 0 and scratch > 0:
                scratch = layer_bwd_scratch // 2 * (k_ce - 1)
            nb["bwd_scratch"] = scratch
            extra = sum(v for _n, v in b.internal_grads)
            sample(f"bwd{i}:{b.op_name}", extra, "grad_internal")
            nb["bwd_scratch"] = 0
            # 释放：该反向节点是某张量的最后读者（bwd_last_read == i）→ 释放它；
            #       某梯度的消费者（grad_free_after == i）→ 释放该梯度。
            for name in b.reads:
                if lg.bwd_last_read.get(name) == i:
                    live.drop(key(name))
            for name, after in lg.grad_free_after.items():
                if after == i:
                    live.drop(gkey(name))

    def _rerun_region(self, live, lg, mb, lid, sample, region, resident) -> None:
        """区域前向子图的**再执行**（插在该区域 backward 之前）。

        liveness 规则在此自动分出两类：会被反向读的 → `recomp_saved`（活到 backward 消费完）；
        不被任何反向节点读的 → `recomp_transient`（区域内最后一次使用即释）。二者**无双算**。"""
        def key(name):
            return (mb, lid, name)

        rlast = lg.region_last_use(region)
        rset = set(region)
        for n in lg.fwd:
            if n.idx not in rset:
                continue
            for name in n.produces:
                t = lg.tensors[name]
                if t.is_weight or live.has(key(name)):
                    continue
                if name in lg.bwd_last_read:
                    live.add(key(name), t.nbytes_saved, "recomp_saved", name, lid, mb)
                else:
                    live.add(key(name), t.nbytes_raw, "recomp_transient", name, lid, mb)
            sample(f"rerun{n.idx}:{n.op_name}", n.workspace, "workspace")
            for name in (*n.reads, *n.produces):
                if (rlast.get(name) == n.idx and name not in lg.bwd_last_read
                        and name not in resident):
                    live.drop(key(name))


_FFN_MARKERS = {"fc", "swiglu", "gelu", "router", "dispatch", "e_", "combine", "shared"}
