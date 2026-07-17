# cost_eval/timesim/producer.py
"""TimedOpSeq producer（spec §3.3：b 并行代入 + c 通信注入装配）。

单元 = 一个**已落实符号 shape 的** cell DAG（extract_cell 后再经 opdag.shape_infer.infer_shapes
的 PART B 输出——本模块不做符号推断,只做并行代入+通信注入,契约边界见 test_timesim_producer.py
的 `_mlp_dag()`：extract_cell 单独产出的 ins/out 仍是 `?` 占位,须先 infer_shapes 落实符号）
→ 一个 TimedSegment（fwd）。

MatMul 节点的权重符号 shape 现场核实**不是**第二个 ins 条目（真 DAG 的 MatMul 节点只携一个
激活侧 ins；PART A 把线性层的 (in,out) 符号维度记在 `node.attrs["in_dim"]`/`["out_dim"]`——
见 extractor._bind_build_module 的 PART A 注释、shape_infer._matmul 同样读 attrs 而非 ins[1]）。
本模块据此把 `in_dim·out_dim` 拼成 weight 的符号 shape 喂给 `shard_rules.weight_local`
（复合 token 需按 sym_shape 语法補括号消歧,见 `_wrap_dim`）。

SP 状态机（spec §5.5 结构性 overlap 的"位置"基础）：sp_active 从段边界起为 deg.sequence_parallel；
  - ColumnParallelLinear 且 sp_active 且 tp>1：矩乘**前**注入 all_gather（模块语义注入——
    Column 源无显式通信，comm_probe 实证为空（test_opdag_comm_probe），spec §3.3c 允许并标
    module="injected:module-semantics"），sp 退出（激活恢复全 seq、feature 分片起）；
  - RowParallelLinear 且 tp>1：矩乘**后**注入 reduce_scatter（sp，进入 sp_active）或
    all_reduce（非 sp）——源惯用法 layers.py:619/:621（comm_probe 实证，guard
    sequence_parallel/!sequence_parallel）。

feature 分片状态机（现场核实新增,原方案未预见的必要机制）：gated-FFN 在 Column 输出~Row 输入
之间的中间量（reshape/split/activation/elementwise）符号 shape 含 feature 轴的位置
**不一定是末轴**（如 MLPInterleaved 的 reshape 把 "(2·ffn_hidden)" 拆成 "ffn_hidden·2" 四轴，
feature 轴落在倒数第二位）——因此不能复用 `shard_rules.localize` 的 `feat_div_last`（只除末轴）。
`feat_sharded` 状态位（True 从 Column 输出起、False 到 Row 输入止）驱动 `_local_shape` 按**轴内容**
除以 tp，与 sp_active 同构的小状态机；feature token 集**不再硬编码**，而是从触发该状态机的
那个 Column 节点的 `attrs["out_dim"]` 现场推导（`parse_axis(out_dim).syms` 键集——MLP fc1 →
{"ffn_hidden"}；MLA q_up 等 attention-family Column 会给出别的 token 集，同一状态机通用无需
改代码），且强制**恰一轴**命中（0 轴/≥2 轴均 fail-loud，见 `_local_shape`）——0 轴防"某中间量
其实没被这套 token 覆盖却被静默放过切错"（reviewer 用 attention-family Column 反例证实的漏洞），
≥2 轴防同一轴集合在两条独立轴上都命中时的分片指派歧义。本状态机把 feature 分片建模成"一个
全局 bool + 一个 token 集"，端态设计应是 **per-tensor** 的分片状态（值键控，非全局位）——T1，
见 spec。

deps 只记跨流依赖（同流 FIFO 隐含，ir.py 字段注释）：本模块按处理顺序维护每个已发节点的
stream，某节点的入边生产者与它同流则不进 deps（注入 comm op 例外，见 ir.py deps 注释）。

未知非空 module 的 MatMul 在 tp>1 时 fail-loud（防 typo 静默不切分——Task 8 review 裁决:
fail-loud 边界在 producer 调用侧）；module=="" 视为复制式 matmul（router gate 等，无 TP 语义），
不做 weight_local/comm 注入，走通用节点路径。该守卫不只是防 typo："SequenceParallelLinear"
是已知未实现模块（MLA q/kv down-proj，见 module_resolver.py:40 的映射登记），命中它同样应
fail-loud 而非静默放过。

View → host_only；符号 shape 若为 `?`（未落实）：ins 侧一律容忍为空 tuple；out 侧仅 device 流
op fail-loud（View 等 host_only op 容忍，与既有 shape_infer"解不出就保 ?"哲学一致）。
"""
from __future__ import annotations

from .ir import (TimedOp, TimedSegment, CommSpec, tensor_bytes,
                 STREAM_DEVICE, STREAM_HOST_ONLY, COMM_STREAM)
from .shard_rules import Degrees, axis_values, localize, weight_local
from ..opdag.sym_shape import parse_shape, parse_axis

_COL = "ColumnParallelLinear"
_ROW = "RowParallelLinear"
_KNOWN_MATMUL_MODULES = {_COL, _ROW, ""}


def _parse_ref(ref: str):
    """opdag 'name:符号shape:dtype' → (name, sym_shape, dtype)。"""
    parts = ref.split(":")
    if len(parts) != 3:
        raise ValueError(f"producer: 非法 TensorRef {ref!r}（fail-loud）")
    return parts[0], parts[1], parts[2]


def _wrap_dim(tok: str) -> str:
    """PART A 记的 in_dim/out_dim 是扁平 token（如 "2·ffn_hidden"）；拼进复合 weight sym 前，
    多因子 token 须按 sym_shape 语法补括号消歧（"H"+"2·ffn_hidden" → "H·(2·ffn_hidden)"，
    否则顶层 "·" 切分会把 "2" 和 "ffn_hidden" 误拆成两条独立轴）。"""
    return f"({tok})" if "·" in tok else tok


def _local_shape(sym: str, dims, deg: Degrees, sp_active: bool, feat_tp: int,
                  feat_syms: frozenset = frozenset(), src: str = "") -> tuple[int, ...]:
    """符号 shape → local 具体 shape：先复用 shard_rules.localize 代入 S(cp/sp)/E(ep) 轴
    （feat_div_last=1——feature 轴不一定是末轴，此处不用该机制），再按 feat_sharded 状态机对
    syms 与调用方给的 `feat_syms`（触发该状态机的 Column 节点 out_dim 自推导，见 build_segment）
    相交的轴（不论位置）÷feat_tp——恰一轴强制：0 轴/≥2 轴均 fail-loud（fail-loud 整除）。"""
    axes = parse_shape(sym)
    vals = axis_values(sym, dims)
    out = localize(vals, sym, deg, feat_div_last=1, sp_active=sp_active)
    if feat_tp > 1:
        hits = [i for i, f in enumerate(axes) if set(f.syms) & feat_syms]
        if not hits:
            raise ValueError(
                f"producer: feature-sharded 区内未识别的中间张量: {sym!r} @ {src}"
                f"（fail-loud，防静默不切分）")
        if len(hits) > 1:
            raise ValueError(
                f"producer: feature 轴歧义（{sym!r} 有 {len(hits)} 轴命中），"
                f"需 per-tensor 分片跟踪（T1）——fail-loud")
        i = hits[0]
        v = out[i]
        if v % feat_tp:
            raise ValueError(
                f"producer: feature 轴 {sym!r} 第{i}轴={v} 不被 tp={feat_tp} 整除（fail-loud）")
        out[i] = v // feat_tp
    return tuple(out)


def _in_shape_or_empty(sym: str, dims, deg: Degrees, sp_active: bool, feat_tp: int,
                        feat_syms: frozenset = frozenset(), src: str = "") -> tuple[int, ...]:
    if not sym or sym == "?":
        return ()
    return _local_shape(sym, dims, deg, sp_active, feat_tp, feat_syms, src)


def _out_shape_or_fail(sym: str, stream: str, src: str, dims, deg: Degrees,
                        sp_active: bool, feat_tp: int,
                        feat_syms: frozenset = frozenset()) -> tuple[int, ...]:
    if not sym or sym == "?":
        if stream == STREAM_DEVICE:
            raise ValueError(f"producer: device op {src} 输出 shape 未解析（fail-loud）")
        return ()
    return _local_shape(sym, dims, deg, sp_active, feat_tp, feat_syms, src)


_OPAQUE_COMM_MARKERS = ("AllReduce", "ReduceScatter", "AllGather", "AlltoAll")


def build_segment(seg_id: str, dag, dims, deg: Degrees, *, phase: str = "fwd",
                   opaque_comm_ok: bool = False) -> TimedSegment:
    if not opaque_comm_ok:
        for call in dag.opaque_calls:
            expr = call.get("expr", "")
            if any(marker in expr for marker in _OPAQUE_COMM_MARKERS):
                raise ValueError(
                    f"producer: dag.opaque_calls 命中疑似通信调用 @{call.get('src')}: "
                    f"{expr!r}——embedding 类段的通信由 comm_probe+装配层注入，walker "
                    f"fallthrough 记录的这条不应被静默吞掉（schema.py opaque_calls docstring "
                    f"消费契约）。调用方现场核实其语义后传 opaque_comm_ok=True 放行。")
    cell = dag.cell
    id2opid = {n.id: f"{cell}#{n.id}" for n in dag.nodes}
    in_edges: dict[int, list[int]] = {}
    for s, d in dag.edges:
        in_edges.setdefault(d, []).append(s)

    ops: list[TimedOp] = []
    stream_of: dict[int, str] = {}     # 原 DAG 节点 id → 已发 TimedOp 的 stream（同流 dep 过滤）
    # F3 Bug B：Row 注入 .rs 后，该矩乘节点的"当前有效"产出从其自身 device-op_id 改指向
    # `<matmul_op_id>.rs`（comm_tp 流）——下游同 DAG 消费者须解析到 .rs，不能被"与矩乘同 device
    # 流"误判为同流而丢依赖。{原 DAG 节点 id → (重定向 op_id, 重定向 stream)}。
    redirect: dict[int, tuple[str, str]] = {}
    sp_active = deg.sequence_parallel
    feat_sharded = False
    feat_syms: frozenset = frozenset()   # feat_sharded 区间当前生效的 feature token 集（F1 自推导）

    def _producer_ref(p: int) -> tuple[str, str]:
        """原 DAG 节点 id → 该生产者当前有效的 (op_id, stream)——查 redirect 优先于原始记录。
        stream_of 查不到（p 尚未发射）→ fail-loud：walker 序=程序序，前向依赖理论不可达，
        真出现说明上游不变量被打破，不应静默回退成 None 再让下游同流过滤逻辑误判。"""
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

        if n.op == "MatMul" and deg.tp > 1 and module not in _KNOWN_MATMUL_MODULES:
            raise ValueError(
                f"producer: MatMul@{n.src} 的 module {module!r} 不在已知集"
                f"{sorted(_KNOWN_MATMUL_MODULES - {''})}（tp>1 下静默不切分会错切——fail-loud）")

        this_stream = STREAM_HOST_ONLY if n.op == "View" else STREAM_DEVICE
        producers = in_edges.get(n.id, ())
        base_deps = tuple(
            ref[0] for ref in (_producer_ref(p) for p in producers) if ref[1] != this_stream
        )

        if n.op == "MatMul" and module in (_COL, _ROW):
            _, x_sym, x_dt = _parse_ref(n.ins[0])
            in_dim, out_dim = n.attrs.get("in_dim"), n.attrs.get("out_dim")
            if not in_dim or not out_dim:
                raise ValueError(
                    f"producer: MatMul@{n.src}（{module}）缺 in_dim/out_dim attrs"
                    f"（PART A 未标注线性维度，fail-loud）")
            weight_sym = f"{_wrap_dim(in_dim)}·{_wrap_dim(out_dim)}"

            if module == _COL and deg.tp > 1 and feat_sharded:
                # 复审探针实证：Column∘Column 直连（上一个 Column 的 feature 分片区尚未被 Row
                # 关闭）会静默不一致——激活收缩维已 ÷tp 而本节点 weight-in 仍是全量（如 3072 vs
                # 6144），矩乘内部矛盾却不报错。
                raise ValueError(
                    f"producer: Column@{n.src} 落在前一个 Column 的 feature 分片区内"
                    f"（背靠背 Column 不在支持族，需 per-tensor 分片跟踪（T1）——fail-loud）")

            deps = base_deps
            if module == _COL and deg.tp > 1 and sp_active:
                gather_in = _local_shape(x_sym, dims, deg, sp_active=True, feat_tp=1)
                gather_out = _local_shape(x_sym, dims, deg, sp_active=False, feat_tp=1)
                # F3 Bug A：AG 挂在 comm_tp 流——从它自己的视角，矩乘的所有入边生产者都是
                # "跨流"，不能沿用按矩乘 device 流过滤过的 base_deps（会把某个 device 流生产者
                # 过滤掉、结果挂空——矩乘随后只 deps 这个 AG，那个生产者就无人依赖了）。
                ag_deps = tuple(_producer_ref(p)[0] for p in producers)
                gather_op = TimedOp(
                    op_id=id2opid[n.id] + ".ag", op_type="CommOp", phase=phase,
                    in_shapes=(gather_in,), out_shape=gather_out, dtype=x_dt,
                    stream=COMM_STREAM["tp"], src=n.src, deps=ag_deps,
                    module="injected:module-semantics",
                    comm=CommSpec("all_gather", tensor_bytes(gather_in, x_dt), "tp", deg.tp))
                ops.append(gather_op)
                deps = (gather_op.op_id,)
                sp_active = False

            # x_local 用的是**进入本节点前**的 feat_sharded/feat_syms（Column 自己的输入还不在
            # feature 分片区——分片区从它的输出才开始；Row 的输入则沿用上一个 Column 留下的状态）。
            x_local = _local_shape(x_sym, dims, deg, sp_active=sp_active,
                                   feat_tp=(deg.tp if feat_sharded else 1),
                                   feat_syms=feat_syms, src=n.src)

            if module == _COL and deg.tp > 1:
                feat_sharded = True   # Column 输出起，feature 轴进入 tp 分片区
                # F1：feature token 集自推导——不再猜全局硬编码集合，从**本节点** out_dim 现场取
                # （MLP fc1 out_dim="2·ffn_hidden" → {"ffn_hidden"}；attention-family Column
                # out_dim="n_heads·v_head_dim" → {"n_heads","v_head_dim"}，同一状态机通用）。
                feat_syms = frozenset(parse_axis(out_dim).syms.keys())
            elif module == _ROW and deg.tp > 1:
                feat_sharded = False   # Row 输入（x_local 已按旧 feat_syms 除过）消费完毕，
                                       # 本节点输出退出 feature 分片区——需在算 out_shape 前复位，
                                       # 否则 out_sym（H 轴，不含 feature token）会被误判 0 轴命中。
                feat_syms = frozenset()

            weight_shape = weight_local(weight_sym, dims, module, deg.tp)
            # Row 矩乘的（通信前）输出恒为全 seq、非 feature 轴——上面已把 feat_sharded 提前复位
            # 为 False，故这里 feat_tp=1，不会误闯"恰一轴"检查；
            # Column 矩乘的输出此时 sp_active 已由上面的 gather 逻辑正确落定（未 sp 则从未 True
            # 过），feat_sharded/feat_syms 也已刷新成本节点的新值。
            out_sp = False if module == _ROW else sp_active
            out_shape = _out_shape_or_fail(out_sym, STREAM_DEVICE, n.src, dims, deg, out_sp,
                                           feat_tp=(deg.tp if feat_sharded else 1),
                                           feat_syms=feat_syms)
            mm_op = TimedOp(op_id=id2opid[n.id], op_type="MatMul", phase=phase,
                            in_shapes=(x_local, weight_shape), out_shape=out_shape,
                            dtype=out_dt, stream=STREAM_DEVICE, src=n.src, deps=deps,
                            module=module)
            ops.append(mm_op)
            stream_of[n.id] = STREAM_DEVICE

            if module == _ROW and deg.tp > 1:
                full = _local_shape(out_sym, dims, deg, sp_active=False, feat_tp=1)
                ctype = "reduce_scatter" if deg.sequence_parallel else "all_reduce"
                out_after = _local_shape(out_sym, dims, deg, sp_active=deg.sequence_parallel,
                                         feat_tp=1)
                # src 双落点（comm_probe 源惯用法实证）：sequence_parallel/!sequence_parallel 两条
                # guard 分叉出的 RS/AR 分别落在带 bias 分支 layers.py:619/:621 与无 bias 分支
                # layers.py:646/:648——两处语义等价，此处固定取带 bias 行号（T1 若需精确源行，
                # 可加 add_bias_linear 标志切换，本模块当前不建模 bias 张量本身）。
                rs_op = TimedOp(
                    op_id=id2opid[n.id] + ".rs", op_type="CommOp", phase=phase,
                    in_shapes=(full,), out_shape=out_after, dtype=out_dt,
                    stream=COMM_STREAM["tp"],
                    src="layers.py:619" if ctype == "reduce_scatter" else "layers.py:621",
                    deps=(mm_op.op_id,), module=module,
                    comm=CommSpec(ctype, tensor_bytes(full, out_dt), "tp", deg.tp))
                ops.append(rs_op)
                # F3 Bug B：本节点 (Row 矩乘) 的"当前有效"产出改指向 .rs（comm_tp 流）——下游
                # 同 DAG 消费者解析依赖时经 _producer_ref 查到这个重定向，而非误判成与矩乘同
                # device 流（从而被同流过滤、永远看不到 .rs）。
                redirect[n.id] = (rs_op.op_id, COMM_STREAM["tp"])
                sp_active = deg.sequence_parallel
            continue

        # —— 通用节点（View/Norm/Cast/Activation/Elementwise/module=="" 的复制式 MatMul 等）——
        feat_tp = deg.tp if feat_sharded else 1
        in_shapes = tuple(
            _in_shape_or_empty(_parse_ref(ref)[1], dims, deg, sp_active, feat_tp,
                               feat_syms, n.src)
            for ref in n.ins
        )
        out_shape = _out_shape_or_fail(out_sym, this_stream, n.src, dims, deg, sp_active, feat_tp,
                                       feat_syms)
        ops.append(TimedOp(op_id=id2opid[n.id], op_type=n.op, phase=phase,
                           in_shapes=in_shapes, out_shape=out_shape, dtype=out_dt,
                           stream=this_stream, src=n.src, deps=base_deps, module=module))
        stream_of[n.id] = this_stream

    return TimedSegment(seg_id=seg_id, ops=tuple(ops))
