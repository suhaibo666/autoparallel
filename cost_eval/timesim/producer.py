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
    expand_dims :722，本模块注入点=**每个**满足分歧条件的多输入汇合 op（不去重）：驻留
    张量喂多个汇合点时相对真机"重分布一次+复用"会重复计通信量，当前 MLA 惯用法只有一处
    汇合故成立；位置差几个小 op 的 S 局部度，量级 S·B·rope_dim 字节，诚实边界）。线性族
    MatMul 不参与此重分布（权重符号 shape 由 attrs 重建；个别手写 DAG——gpt_segments
    head_segment_dag 的 lm_head Column——在 ins 带显式权重 ref，参与会误触发，Task 2 质量复审）。
  - ColumnParallelLinear：输入含 feature carrier → fail-loud（Column∘Column 不在支持族）；
    输入含 "S" → 矩乘前注入 all_gather（模块语义注入，comm_probe 实证 Column 源无显式通信）
    并清 "S"；输出 = {carrier(out_dim): tp}（tp>1）。
  - RowParallelLinear：tp>1 时输入携 feature carrier 的须恰一个且命中 in_dim syms、且无 "S"
    残留（Megatron Row 恒消费 feature 分片/全 seq 激活）；无 carrier 且无 "S" 的输入（上游
    没有 Column）fail-loud——静默不切会双倍计算量；**无 carrier 但 "S" 驻留**的输入仅当来自
    段入口裸输入（x_pnode is None）时按状态本地化放行（T0 行为兼容形态：.rs redirect 契约
    测试的合成 DAG 即此形，真实 Megatron 段不出现——Row 前恒有 Column），来自图内生产链的
    （如 SPL→Row 直连）则 fail-loud（收缩维不一致，Task 2 质量复审收紧）；矩乘后注入 RS（sp，
    出 {"S":tp}）/ AR（非 sp，出 {}）——源惯用法 layers.py:619/:621；下游经 redirect 指向
    .rs（F3 Bug B）。
  - SequenceParallelLinear（T1 新支持，layers.py:819「A is not parallelized. X is
    parallelized with data_parallel and sequence_parallel」，shard 布局 :845-866 入/出均
    ("cp","tp") 切 S、权重不切）：权重全量、无通信、状态透传。
deps 只记跨流依赖（同流 FIFO 隐含，ir.py 字段注释）；依赖发现按输入 ref 名→生产者节点
（与 dag.edges 等价——walker 对 tuple 全目标登记 producer，_emit :898-908）。

comm_probe 消费现状（code-review T1a [5] 补记，避免"探针建了不用"的死代码误读）：spec §3.3c
设计的是**探针驱动**注入——识别出源码里的显式通信惯用法就发 CommOp，识别不出才退回模块语义
注入并 fail-loud。本模块（T1 范围）反过来：TP/SP 通信是**硬编码**语义（上文 Row `.rs` 的
reduce_scatter/all_reduce 二选一、Column 的 sp all_gather），从不调用
`opdag.comm_probe.probe_cell_comm`——下面 import 的 `COMM_CLS` 只用于 opaque_calls 疑似通信
调用的字符串守卫（_OPAQUE_COMM_MARKERS），不是消费探针的解析结果。为不让 comm_probe 沦为
死代码，`tests/test_producer_comm_matches_probe.py` 做**交叉校验**：用 probe 从真 layers.py
提取的 CommSite（源真相）断言与本模块硬编码的 ctype/guard（消费假设）一致，并断言 Row/
Embedding 站点不出现 guard=="?"——源码 idiom 漂移、或本模块硬编码被误改，测试即变红，不是
静默分道扬镳。**运行时探针驱动注入**（producer 直接消费 probe_cell_comm 结果决定发哪种
CommOp，取代现有 if/else 硬编码）是 v1.5 工作，当前 T1 只做静态交叉校验锁一致性。
"""
from __future__ import annotations

from dataclasses import dataclass

from .ir import (TimedOp, TimedSegment, CommSpec, tensor_bytes,
                 STREAM_DEVICE, STREAM_HOST_ONLY, COMM_STREAM)
from .shard_rules import Degrees, axis_values, localize, weight_local
from ..opdag.comm_probe import COMM_CLS
from ..opdag.sym_shape import parse_shape, _split_top

_COL = "ColumnParallelLinear"
_ROW = "RowParallelLinear"
_SPL = "SequenceParallelLinear"
_KNOWN_MATMUL_MODULES = {_COL, _ROW, _SPL, ""}


@dataclass
class _InInfo:
    """build_segment 主循环里一条输入 ref 的可变工作记录（替代 6 元素裸 list——扩展主循环时
    字段名比下标更抗滑错）。`state`=该输入的 per-tensor 分片状态；`pnode`=生产者 DAG 节点 id
    （None=段入口外部量）；`ag`=为消解 S 分歧给本输入注入的重分布 AG（None=未注入）。"""
    name: str
    sym: str
    dt: str
    state: dict | None
    pnode: int | None
    ag: object = None

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

        # —— 输入侧：每个 ref 建一条 _InInfo(name/sym/dt/state/pnode/ag) 工作记录 ——
        # 名字优先匹配（node.out 名 + split_targets）；未匹配名与未匹配入边"恰一对一"时按
        # 排除法配对——walker 的链式视图别名把目标名指到内层节点而**不新增 out 名**
        # （construct_walker.py:461-469：ssa/producer 有记录、DAG 节点无此名），名字查不到
        # 但边在。sym=="?" 的 ref（rotary_pos_emb/attention_mask 类外部量）不参与配对。
        # 配对歧义（候选名>1 或 剩余边>1 且有候选名）→ fail-loud 不猜。
        producers = list(dict.fromkeys(in_edges.get(n.id, ())))
        in_infos: list[_InInfo] = [
            _InInfo(nm, sym, dt, None, name2node.get(nm))
            for nm, sym, dt in (_parse_ref(ref) for ref in n.ins)
        ]
        matched = {info.pnode for info in in_infos if info.pnode is not None}
        free_prods = [p for p in producers if p not in matched]
        candidates = [info for info in in_infos
                      if info.pnode is None and info.sym and info.sym != "?"]
        if free_prods and candidates:
            if len(free_prods) == 1 and len(candidates) == 1:
                candidates[0].pnode = free_prods[0]
            else:
                raise ValueError(
                    f"producer: 节点 {n.src} 有 {len(free_prods)} 条未匹配入边与 "
                    f"{len(candidates)} 个未匹配输入名，无法唯一配对（fail-loud）")
        leftover_prods = [p for p in free_prods
                          if all(info.pnode != p for info in in_infos)]
        for info in in_infos:
            if info.pnode is not None:
                info.state = dict(node_state.get(info.pnode, {}))
            elif info.name in input_states:
                info.state = dict(input_states[info.name])
            else:
                info.state = dict(seed) if _has_s_axis(info.sym) else {}

        # —— "S" 分歧重分布（多输入汇合，模块 docstring 规则；源事实4）——
        # 线性族 MatMul 跳过：其权重符号 shape 由 in_dim/out_dim attrs 重建（非 ins[1]），但
        # **个别 DAG 仍在 ins 里带显式权重 ref**——gpt_segments.head_segment_dag() 的 lm_head
        # Column 即 ins=["h:S·B·H", "W_head:H·vocab"]（真实生产代码，非合成 fixture）。若不跳过，
        # 权重 ref（无 S）与激活（S 驻留）在此被判 S 分歧 → 误注入 layout-redistribution AG 并
        # 提前清激活的 S，把 Column 自己的模块语义 AG 顶替掉（Task 2 质量复审在 lm_head tp+sp
        # 路径实证）。extract_cell 产的 MatMul 确是激活单入，但本守卫不能依赖它。
        is_linear = n.op == "MatMul" and module in (_COL, _ROW, _SPL)
        real = [info for info in in_infos if info.sym and info.sym != "?"]
        if len(real) > 1 and not is_linear:
            s_flags = [bool(info.state.get("S")) for info in real]
            if any(s_flags) and not all(s_flags):
                for k, info in enumerate(in_infos):
                    if not (info.sym and info.sym != "?" and info.state.get("S")):
                        continue
                    shard_shape = _localize_with_state(info.sym, dims, deg, info.state, n.src)
                    new_st = {c: d for c, d in info.state.items() if c != "S"}
                    full_shape = _localize_with_state(info.sym, dims, deg, new_st, n.src)
                    ag_deps = ()
                    if info.pnode is not None:
                        ag_deps = (_producer_ref(info.pnode)[0],)
                    ag = TimedOp(
                        op_id=f"{id2opid[n.id]}.ag{k}", op_type="CommOp", phase=phase,
                        in_shapes=(shard_shape,), out_shape=full_shape, dtype=info.dt,
                        stream=COMM_STREAM["tp"], src=n.src, deps=ag_deps,
                        module="injected:layout-redistribution",
                        comm=CommSpec("all_gather", tensor_bytes(shard_shape, info.dt),
                                      "tp", deg.tp))
                    ops.append(ag)
                    info.state = new_st
                    info.ag = ag

        def _base_deps() -> tuple[str, ...]:
            deps, seen = [], set()
            refs = [((info.ag.op_id, info.ag.stream) if info.ag is not None
                     else _producer_ref(info.pnode))
                    for info in in_infos if info.ag is not None or info.pnode is not None]
            refs += [_producer_ref(p) for p in leftover_prods]   # 无名可配的入边不丢依赖
            for ref in refs:
                if ref[1] != this_stream and ref[0] not in seen:
                    deps.append(ref[0])
                    seen.add(ref[0])
            return tuple(deps)

        # ================= 线性层族 =================
        if is_linear:
            x_info = in_infos[0]
            x_sym, x_dt, x_st, x_pnode, x_ag = (x_info.sym, x_info.dt, x_info.state,
                                                x_info.pnode, x_info.ag)
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
                elif x_pnode is not None:
                    # 无 carrier 但 "S" 驻留，且来自**图内生产链**（如 SPL→Row 直连）：收缩维
                    # 与全量 weight-in 不一致却无人报——fail-loud（Task 2 质量复审收紧）。
                    raise ValueError(
                        f"producer: Row@{n.src} 输入无 feature 分片且非段入口（图内生产者 "
                        f"id={x_pnode} 的 S 驻留激活喂 Row，收缩维不一致——fail-loud）")
                # else（x_pnode is None）：段入口裸输入的 S 驻留——T0 行为兼容形态（.rs
                # redirect 契约测试的合成 DAG），按状态本地化放行，真实 Megatron 段不出现。

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
        for info in in_infos:
            for c, d in info.state.items():
                if c in merged and merged[c] != d:
                    raise ValueError(
                        f"producer: 汇合节点 {n.src} 的 carrier {c!r} 度数冲突"
                        f"（{merged[c]} vs {d}）——fail-loud")
                merged[c] = d
        in_shapes = tuple(
            _localize_with_state(info.sym, dims, deg, info.state, n.src)
            if info.sym and info.sym != "?" else ()
            for info in in_infos
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
