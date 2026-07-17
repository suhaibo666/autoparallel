# cost_eval/timesim/producer.py
"""TimedOpSeq producer（spec §3.3：b 并行代入 + c 通信注入装配）——per-tensor 分片状态版（T1）。

输入契约与 T0 版一致：只收 extract_cell 后经 opdag.shape_infer.infer_shapes 落实符号 shape 的
DAG；Col/Row/SPL MatMul 须携 in_dim/out_dim attrs；含通信样 opaque_calls 的 DAG 须显式
opaque_comm_ok=True。`?` 容忍规则不变：ins 一律容忍为空 tuple；out 仅 device 流 fail-loud。

**per-tensor 分片状态**（T0 交接要点5 端态设计）：T0 的 sp_active/feat_sharded 是两个全局
bool——对 MLP 单链成立，MLA 多支路（rope 支不过 Column、与过 Column 的 k_no_pe 在 pe_concat
汇合）立即失效。本版把状态挂**生产者节点**：node_state[node_id] = {carrier_sym: divisor}：
  - "S" 键 = SP 驻留（S 轴在 cp 之外再 ÷tp）。段入口种子：外部输入（无生产者）sym 含 S 轴且
    deg.sequence_parallel 且 tp>1 时 = {"S": tp}；input_states 参数可按名覆盖。
  - 其余键 = feature carrier（Column 输出起）。carrier = out_dim 顶层乘积**首个符号因子**
    （_carrier_sym）：MLP fc1 "2·ffn_hidden"→"ffn_hidden"（interleaved 布局切 ffn_hidden，
    T0 核实）；MLA q_up "n_heads·(qk_head_dim+qk_pos_emb_head_dim)"→"n_heads"（Megatron
    按 head 切）。切分轴唯一化：reshape 把 heads/head_dim 拆两轴后只有 n_heads 轴命中——
    T0「≥2 轴命中歧义」自然消失；真歧义（同 carrier 现身 ≥2 轴）仍 fail-loud。
状态传播规则（通用节点的 merged 并集 / 各模块分支）：
  - 单输入 op：透传；out sym 缺 carrier 轴 → fail-loud（防静默丢跟踪）。
  - 多输出 split：共享一个节点 state（切的是 D 轴，各半分片语义相同）；消费端经
    attrs["split_targets"] 登记的 name→node 映射找到生产者。
  - 多输入汇合：feature carrier 取并集（无 carrier 侧=复制量，真机按目的 op shard
    in_strategy 本地切片、零通信——multi_latent_attention.py shard_self_attn :715-722
    pe_concat/tile_kv 布局实证；in_shapes 因此允许 heads 轴不齐，host_only 无成本）；
    "S" 分歧（部分 SP 驻留、部分已聚合）→ 给驻留侧注入 all_gather
    （module="injected:layout-redistribution"）——对应 semi-auto 布局重分布（真机声明点
    expand_dims :722，本模块注入点=首个多输入汇合 op，位置差几个小 op 的 S 局部度，
    量级 S·B·rope_dim 字节，诚实边界）。
  - ColumnParallelLinear：输入含 feature carrier → fail-loud（Column∘Column 不在支持族）；
    输入含 "S" → 矩乘前注入 all_gather（模块语义注入，comm_probe 实证 Column 源无显式通信）
    并清 "S"；输出 = {carrier(out_dim): tp}（tp>1）。
  - RowParallelLinear：tp>1 时输入携 feature carrier 的须恰一个且命中 in_dim syms、且无 "S"
    残留（Megatron Row 恒消费 feature 分片/全 seq 激活）；无 carrier 且无 "S" 的输入（上游
    没有 Column）fail-loud——静默不切会双倍计算量；**无 carrier 但 "S" 驻留**的输入按状态
    本地化放行（T0 行为兼容形态：.rs redirect 契约测试的合成 DAG 即此形，真实 Megatron 段
    不出现——Row 前恒有 Column）；矩乘后注入 RS（sp，出 {"S":tp}）
    / AR（非 sp，出 {}）——源惯用法 layers.py:619/:621；下游经 redirect 指向 .rs（F3 Bug B）。
  - SequenceParallelLinear（T1 新支持，layers.py:819「A is not parallelized. X is
    parallelized with data_parallel and sequence_parallel」，shard 布局 :845-866 入/出均
    ("cp","tp") 切 S、权重不切）：权重全量、无通信、状态透传。
deps 只记跨流依赖（同流 FIFO 隐含，ir.py 字段注释）；依赖发现按输入 ref 名→生产者节点
（与 dag.edges 等价——walker 对 tuple 全目标登记 producer，_emit :898-908）。
"""
from __future__ import annotations

from .ir import (TimedOp, TimedSegment, CommSpec, tensor_bytes,
                 STREAM_DEVICE, STREAM_HOST_ONLY, COMM_STREAM)
from .shard_rules import Degrees, axis_values, localize, weight_local
from ..opdag.comm_probe import COMM_CLS
from ..opdag.sym_shape import parse_shape, _split_top

_COL = "ColumnParallelLinear"
_ROW = "RowParallelLinear"
_SPL = "SequenceParallelLinear"
_KNOWN_MATMUL_MODULES = {_COL, _ROW, _SPL, ""}

# 从 comm_probe 的通信原语类名表派生（单一事实源）——新原语进 COMM_CLS 时本守卫自动跟进。
_OPAQUE_COMM_MARKERS = tuple(COMM_CLS)


def _parse_ref(ref: str):
    """opdag 'name:符号shape:dtype' → (name, sym_shape, dtype)。"""
    parts = ref.split(":")
    if len(parts) != 3:
        raise ValueError(f"producer: 非法 TensorRef {ref!r}（fail-loud）")
    return parts[0], parts[1], parts[2]


def _wrap_dim(tok: str) -> str:
    """PART A 的 in_dim/out_dim 扁平 token 拼复合 weight sym 前补括号消歧（T0 注释）。"""
    return f"({tok})" if "·" in tok else tok


def _carrier_sym(out_dim: str) -> str:
    """out_dim 顶层乘积的首个**符号**因子 = TP 切分 carrier（模块 docstring 源事实5）。"""
    for part in _split_top(out_dim, "·"):
        p = part.strip()
        if p.startswith("(") and p.endswith(")"):
            p = p[1:-1].strip()
        if p and not p.isdigit():
            return p
    raise ValueError(f"producer: out_dim {out_dim!r} 无符号因子，carrier 不可定（fail-loud）")


def _has_s_axis(sym: str) -> bool:
    if not sym or sym == "?":
        return False
    return any("S" in f.syms for f in parse_shape(sym))


def _localize_with_state(sym: str, dims, deg: Degrees, state: dict, src: str) -> tuple[int, ...]:
    """符号 shape → local：先 localize 代入全局均匀轴（S÷cp、E÷ep），再按 per-tensor 状态
    对 carrier 命中轴（恰一轴强制）÷divisor。"""
    axes = parse_shape(sym)
    out = localize(axis_values(sym, dims), sym, deg, feat_div_last=1, sp_active=False)
    for carrier, div in state.items():
        hits = [i for i, f in enumerate(axes) if carrier in f.syms]
        if not hits:
            raise ValueError(
                f"producer: {sym!r} @ {src} 不含分片 carrier {carrier!r} 轴——跟踪断裂（fail-loud）")
        if len(hits) > 1:
            raise ValueError(
                f"producer: {sym!r} @ {src} 的 carrier {carrier!r} 命中 {len(hits)} 轴——歧义（fail-loud）")
        i = hits[0]
        if out[i] % div:
            raise ValueError(
                f"producer: {sym!r} 第{i}轴={out[i]} 不被 {div} 整除 @ {src}（fail-loud）")
        out[i] //= div
    return tuple(out)


def build_segment(seg_id: str, dag, dims, deg: Degrees, *, phase: str = "fwd",
                   opaque_comm_ok: bool = False,
                   input_states: dict | None = None) -> TimedSegment:
    if not opaque_comm_ok:
        for call in dag.opaque_calls:
            expr = call.get("expr", "")
            if any(marker in expr for marker in _OPAQUE_COMM_MARKERS):
                raise ValueError(
                    f"producer: dag.opaque_calls 命中疑似通信调用 @{call.get('src')}: "
                    f"{expr!r}——walker fallthrough 不应被静默吞掉（schema.py opaque_calls "
                    f"消费契约）。调用方现场核实语义后传 opaque_comm_ok=True 放行。")
    cell = dag.cell
    id2opid = {n.id: f"{cell}#{n.id}" for n in dag.nodes}
    in_edges: dict[int, list[int]] = {}
    for s, d in dag.edges:
        in_edges.setdefault(d, []).append(s)

    ops: list[TimedOp] = []
    stream_of: dict[int, str] = {}
    redirect: dict[int, tuple[str, str]] = {}
    node_state: dict[int, dict] = {}
    name2node: dict[str, int] = {}
    seed = {"S": deg.tp} if (deg.sequence_parallel and deg.tp > 1) else {}
    input_states = input_states or {}

    def _producer_ref(p: int) -> tuple[str, str]:
        if p in redirect:
            return redirect[p]
        stream = stream_of.get(p)
        if stream is None:
            raise ValueError(
                f"producer: 节点 id={p} 的生产者尚未发射（前向依赖，walker 序理论不可达）"
                f"（fail-loud）")
        return id2opid[p], stream

    for n in dag.nodes:
        module = n.module or ""
        out_name, out_sym, out_dt = _parse_ref(n.out) if n.out else ("", "", "bf16")
        this_stream = STREAM_HOST_ONLY if n.op == "View" else STREAM_DEVICE

        if n.op == "MatMul" and deg.tp > 1 and module not in _KNOWN_MATMUL_MODULES:
            raise ValueError(
                f"producer: MatMul@{n.src} 的 module {module!r} 不在已知集"
                f"{sorted(_KNOWN_MATMUL_MODULES - {''})}（tp>1 下静默不切分会错切——fail-loud）")

        # —— 输入侧：ref 名 → [名, sym, dtype, 状态, 生产者节点id|None, 重分布AG|None] ——
        # 名字优先匹配（node.out 名 + split_targets）；未匹配名与未匹配入边"恰一对一"时按
        # 排除法配对——walker 的链式视图别名把目标名指到内层节点而**不新增 out 名**
        # （construct_walker.py:461-469：ssa/producer 有记录、DAG 节点无此名），名字查不到
        # 但边在。sym=="?" 的 ref（rotary_pos_emb/attention_mask 类外部量）不参与配对。
        # 配对歧义（候选名>1 或 剩余边>1 且有候选名）→ fail-loud 不猜。
        producers = list(dict.fromkeys(in_edges.get(n.id, ())))
        in_infos: list[list] = []
        for ref in n.ins:
            nm, sym, dt = _parse_ref(ref)
            in_infos.append([nm, sym, dt, None, name2node.get(nm), None])
        matched = {info[4] for info in in_infos if info[4] is not None}
        free_prods = [p for p in producers if p not in matched]
        candidates = [info for info in in_infos
                      if info[4] is None and info[1] and info[1] != "?"]
        if free_prods and candidates:
            if len(free_prods) == 1 and len(candidates) == 1:
                candidates[0][4] = free_prods[0]
            else:
                raise ValueError(
                    f"producer: 节点 {n.src} 有 {len(free_prods)} 条未匹配入边与 "
                    f"{len(candidates)} 个未匹配输入名，无法唯一配对（fail-loud）")
        leftover_prods = [p for p in free_prods
                          if all(info[4] != p for info in in_infos)]
        for info in in_infos:
            if info[4] is not None:
                info[3] = dict(node_state.get(info[4], {}))
            elif info[0] in input_states:
                info[3] = dict(input_states[info[0]])
            else:
                info[3] = dict(seed) if _has_s_axis(info[1]) else {}

        # —— "S" 分歧重分布（多输入汇合，模块 docstring 规则；源事实4）——
        real = [info for info in in_infos if info[1] and info[1] != "?"]
        if len(real) > 1:
            s_flags = [bool(info[3].get("S")) for info in real]
            if any(s_flags) and not all(s_flags):
                for k, info in enumerate(in_infos):
                    nm, sym, dt, st, pnode, _ = info
                    if not (sym and sym != "?" and st.get("S")):
                        continue
                    shard_shape = _localize_with_state(sym, dims, deg, st, n.src)
                    new_st = {c: d for c, d in st.items() if c != "S"}
                    full_shape = _localize_with_state(sym, dims, deg, new_st, n.src)
                    ag_deps = ()
                    if pnode is not None:
                        ag_deps = (_producer_ref(pnode)[0],)
                    ag = TimedOp(
                        op_id=f"{id2opid[n.id]}.ag{k}", op_type="CommOp", phase=phase,
                        in_shapes=(shard_shape,), out_shape=full_shape, dtype=dt,
                        stream=COMM_STREAM["tp"], src=n.src, deps=ag_deps,
                        module="injected:layout-redistribution",
                        comm=CommSpec("all_gather", tensor_bytes(shard_shape, dt),
                                      "tp", deg.tp))
                    ops.append(ag)
                    info[3] = new_st
                    info[5] = ag

        def _base_deps() -> tuple[str, ...]:
            deps, seen = [], set()
            refs = [((ag.op_id, ag.stream) if ag is not None else _producer_ref(pnode))
                    for nm, sym, dt, st, pnode, ag in in_infos if ag is not None or pnode is not None]
            refs += [_producer_ref(p) for p in leftover_prods]   # 无名可配的入边不丢依赖
            for ref in refs:
                if ref[1] != this_stream and ref[0] not in seen:
                    deps.append(ref[0])
                    seen.add(ref[0])
            return tuple(deps)

        # ================= 线性层族 =================
        if n.op == "MatMul" and module in (_COL, _ROW, _SPL):
            nm, x_sym, x_dt, x_st, x_pnode, x_ag = in_infos[0]
            in_dim, out_dim = n.attrs.get("in_dim"), n.attrs.get("out_dim")
            if not in_dim or not out_dim:
                raise ValueError(
                    f"producer: MatMul@{n.src}（{module}）缺 in_dim/out_dim attrs"
                    f"（PART A 未标注线性维度，fail-loud）")
            weight_sym = f"{_wrap_dim(in_dim)}·{_wrap_dim(out_dim)}"
            feat_carriers = {c: d for c, d in x_st.items() if c != "S"}

            if module in (_COL, _SPL) and feat_carriers:
                raise ValueError(
                    f"producer: {module}@{n.src} 输入落在 feature 分片区内"
                    f"（Column∘Column 不在支持族——fail-loud）")

            deps = _base_deps()
            if module == _COL and deg.tp > 1 and x_st.get("S"):
                gather_in = _localize_with_state(x_sym, dims, deg, x_st, n.src)
                x_st = {c: d for c, d in x_st.items() if c != "S"}
                gather_out = _localize_with_state(x_sym, dims, deg, x_st, n.src)
                # AG 挂 comm_tp 流：其 deps 取该输入的生产者（跨流规则从 AG 自己的视角，F3 Bug A）
                ag_deps = ()
                if x_ag is not None:
                    ag_deps = (x_ag.op_id,)
                elif x_pnode is not None:
                    ag_deps = (_producer_ref(x_pnode)[0],)
                gather_op = TimedOp(
                    op_id=id2opid[n.id] + ".ag", op_type="CommOp", phase=phase,
                    in_shapes=(gather_in,), out_shape=gather_out, dtype=x_dt,
                    stream=COMM_STREAM["tp"], src=n.src, deps=ag_deps,
                    module="injected:module-semantics",
                    comm=CommSpec("all_gather", tensor_bytes(gather_in, x_dt), "tp", deg.tp))
                ops.append(gather_op)
                deps = (gather_op.op_id,)

            if module == _ROW and deg.tp > 1:
                if feat_carriers:
                    if x_st.get("S"):
                        raise ValueError(
                            f"producer: Row@{n.src} 输入残留 SP 驻留（Megatron Row 恒消费全 "
                            f"seq feature 分片激活）——fail-loud")
                    in_syms = set()
                    for f in parse_shape(_wrap_dim(in_dim)):
                        in_syms |= set(f.syms)
                    if len(feat_carriers) != 1 or not (set(feat_carriers) & in_syms):
                        raise ValueError(
                            f"producer: Row@{n.src} 输入 feature 分片状态 {feat_carriers} "
                            f"与 in_dim {in_dim!r} 不匹配（carrier 不命中——fail-loud）")
                elif not x_st.get("S"):
                    raise ValueError(
                        f"producer: Row@{n.src} 输入 feature 分片状态 {{}} 与 in_dim "
                        f"{in_dim!r} 不匹配（无 Column 上游——fail-loud）")
                # else：无 carrier 但 "S" 驻留——T0 行为兼容形态（模块 docstring Row 条目；
                # .rs redirect 契约测试的合成 DAG），按状态本地化放行，不 fail-loud。

            x_local = _localize_with_state(x_sym, dims, deg, x_st, n.src)
            weight_shape = weight_local(weight_sym, dims, module, deg.tp)

            if module == _COL and deg.tp > 1:
                out_st = {_carrier_sym(out_dim): deg.tp}
            elif module == _SPL:
                out_st = dict(x_st)                       # SP 驻留透传（源事实1）
            else:                                          # Row（通信前全量）/ tp==1
                out_st = {}
            out_shape = _localize_with_state(out_sym, dims, deg, out_st, n.src) \
                if out_sym and out_sym != "?" else ()
            if not out_shape:
                raise ValueError(f"producer: device op {n.src} 输出 shape 未解析（fail-loud）")
            mm_op = TimedOp(op_id=id2opid[n.id], op_type="MatMul", phase=phase,
                            in_shapes=(x_local, weight_shape), out_shape=out_shape,
                            dtype=out_dt, stream=STREAM_DEVICE, src=n.src, deps=deps,
                            module=module)
            ops.append(mm_op)
            stream_of[n.id] = STREAM_DEVICE
            node_state[n.id] = out_st
            name2node[out_name] = n.id

            if module == _ROW and deg.tp > 1:
                full = out_shape
                ctype = "reduce_scatter" if deg.sequence_parallel else "all_reduce"
                post_st = {"S": deg.tp} if deg.sequence_parallel else {}
                out_after = _localize_with_state(out_sym, dims, deg, post_st, n.src)
                # src 双落点说明见 T0 注释：带 bias 分支 layers.py:619/:621（无 bias :646/:648）
                rs_op = TimedOp(
                    op_id=id2opid[n.id] + ".rs", op_type="CommOp", phase=phase,
                    in_shapes=(full,), out_shape=out_after, dtype=out_dt,
                    stream=COMM_STREAM["tp"],
                    src="layers.py:619" if ctype == "reduce_scatter" else "layers.py:621",
                    deps=(mm_op.op_id,), module=module,
                    comm=CommSpec(ctype, tensor_bytes(full, out_dt), "tp", deg.tp))
                ops.append(rs_op)
                redirect[n.id] = (rs_op.op_id, COMM_STREAM["tp"])
                node_state[n.id] = post_st
            continue

        # ================= 通用节点 =================
        merged: dict[str, int] = {}
        for nm, sym, dt, st, pnode, ag in in_infos:
            for c, d in st.items():
                if c in merged and merged[c] != d:
                    raise ValueError(
                        f"producer: 汇合节点 {n.src} 的 carrier {c!r} 度数冲突"
                        f"（{merged[c]} vs {d}）——fail-loud")
                merged[c] = d
        in_shapes = tuple(
            _localize_with_state(sym, dims, deg, st, n.src) if sym and sym != "?" else ()
            for nm, sym, dt, st, pnode, ag in in_infos
        )
        if out_sym and out_sym != "?":
            # 汇合出边状态 = 并集中 out sym 实际含有的 carrier（_localize_with_state 对缺轴
            # fail-loud，此处先过滤——多输入并集里允许某 carrier 只属于部分输入（切片口径），
            # 但**单输入透传**缺轴仍要炸：单输入时不过滤，保留 T0 防丢跟踪守卫。
            if len(real) > 1:
                axes_syms: set = set()
                for f in parse_shape(out_sym):
                    axes_syms |= set(f.syms)
                out_st = {c: d for c, d in merged.items() if c in axes_syms}
            else:
                out_st = merged
            out_shape = _localize_with_state(out_sym, dims, deg, out_st, n.src)
        else:
            if this_stream == STREAM_DEVICE:
                raise ValueError(f"producer: device op {n.src} 输出 shape 未解析（fail-loud）")
            out_st = merged
            out_shape = ()
        ops.append(TimedOp(op_id=id2opid[n.id], op_type=n.op, phase=phase,
                           in_shapes=in_shapes, out_shape=out_shape, dtype=out_dt,
                           stream=this_stream, src=n.src, deps=_base_deps(), module=module))
        stream_of[n.id] = this_stream
        node_state[n.id] = out_st
        if out_name:
            name2node[out_name] = n.id
        for t in n.attrs.get("split_targets", ()):           # 源事实3：split 多目标共享本节点
            name2node[t] = n.id

    return TimedSegment(seg_id=seg_id, ops=tuple(ops))
