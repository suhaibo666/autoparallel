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
之间的中间量（reshape/split/activation/elementwise）符号 shape 含 "ffn_hidden"/"moe_ffn" 的轴
**不一定是末轴**（如 MLPInterleaved 的 reshape 把 "(2·ffn_hidden)" 拆成 "ffn_hidden·2" 四轴，
feature 轴落在倒数第二位）——因此不能复用 `shard_rules.localize` 的 `feat_div_last`（只除末轴）。
`feat_sharded` 状态位（True 从 Column 输出起、False 到 Row 输入止）驱动 `_local_shape` 按**轴内容**
（`syms ∩ _FEATURE_SYMS` 非空即除，不问位置）除以 tp，与 sp_active 同构的小状态机。

deps 只记跨流依赖（同流 FIFO 隐含，ir.py 字段注释）：本模块按处理顺序维护每个已发节点的
stream，某节点的入边生产者与它同流则不进 deps。

未知非空 module 的 MatMul 在 tp>1 时 fail-loud（防 typo 静默不切分——Task 8 review 裁决:
fail-loud 边界在 producer 调用侧）；module=="" 视为复制式 matmul（router gate 等，无 TP 语义），
不做 weight_local/comm 注入，走通用节点路径。

View → host_only；符号 shape 若为 `?`（未落实）：ins 侧一律容忍为空 tuple；out 侧仅 device 流
op fail-loud（View 等 host_only op 容忍，与既有 shape_infer"解不出就保 ?"哲学一致）。
"""
from __future__ import annotations

from .ir import (TimedOp, TimedSegment, CommSpec,
                 STREAM_DEVICE, STREAM_HOST_ONLY, COMM_STREAM)
from .shard_rules import Degrees, axis_values, localize, weight_local
from ..opdag.sym_shape import parse_shape

_COL = "ColumnParallelLinear"
_ROW = "RowParallelLinear"
_KNOWN_MATMUL_MODULES = {_COL, _ROW, ""}

# gated-FFN 中间量的 tp 分片 family（Column 出~Row 入之间，feat_sharded 状态机专用；
# 对照 sym_shape.CONFIG2SYM：ffn_hidden_size→ffn_hidden、moe_ffn_hidden_size→moe_ffn）。
_FEATURE_SYMS = {"ffn_hidden", "moe_ffn"}


def _parse_ref(ref: str):
    """opdag 'name:符号shape:dtype' → (name, sym_shape, dtype)。"""
    parts = ref.split(":")
    if len(parts) != 3:
        raise ValueError(f"producer: 非法 TensorRef {ref!r}（fail-loud）")
    return parts[0], parts[1], parts[2]


def _bytes_of(shape, dtype: str) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * (4 if dtype == "fp32" else 2)


def _wrap_dim(tok: str) -> str:
    """PART A 记的 in_dim/out_dim 是扁平 token（如 "2·ffn_hidden"）；拼进复合 weight sym 前，
    多因子 token 须按 sym_shape 语法补括号消歧（"H"+"2·ffn_hidden" → "H·(2·ffn_hidden)"，
    否则顶层 "·" 切分会把 "2" 和 "ffn_hidden" 误拆成两条独立轴）。"""
    return f"({tok})" if "·" in tok else tok


def _local_shape(sym: str, dims, deg: Degrees, sp_active: bool, feat_tp: int) -> tuple[int, ...]:
    """符号 shape → local 具体 shape：先复用 shard_rules.localize 代入 S(cp/sp)/E(ep) 轴
    （feat_div_last=1——feature 轴不一定是末轴，此处不用该机制），再按 feat_sharded 状态机对
    syms 含 _FEATURE_SYMS 的轴（不论位置）÷feat_tp（fail-loud 整除）。"""
    axes = parse_shape(sym)
    vals = axis_values(sym, dims)
    out = localize(vals, sym, deg, feat_div_last=1, sp_active=sp_active)
    if feat_tp > 1:
        for i, f in enumerate(axes):
            if set(f.syms) & _FEATURE_SYMS:
                v = out[i]
                if v % feat_tp:
                    raise ValueError(
                        f"producer: feature 轴 {sym!r} 第{i}轴={v} 不被 tp={feat_tp} 整除（fail-loud）")
                out[i] = v // feat_tp
    return tuple(out)


def _in_shape_or_empty(sym: str, dims, deg: Degrees, sp_active: bool, feat_tp: int) -> tuple[int, ...]:
    if not sym or sym == "?":
        return ()
    return _local_shape(sym, dims, deg, sp_active, feat_tp)


def _out_shape_or_fail(sym: str, stream: str, src: str, dims, deg: Degrees,
                        sp_active: bool, feat_tp: int) -> tuple[int, ...]:
    if not sym or sym == "?":
        if stream == STREAM_DEVICE:
            raise ValueError(f"producer: device op {src} 输出 shape 未解析（fail-loud）")
        return ()
    return _local_shape(sym, dims, deg, sp_active, feat_tp)


def build_segment(seg_id: str, dag, dims, deg: Degrees, *, phase: str = "fwd") -> TimedSegment:
    cell = dag.cell
    id2opid = {n.id: f"{cell}#{n.id}" for n in dag.nodes}
    in_edges: dict[int, list[int]] = {}
    for s, d in dag.edges:
        in_edges.setdefault(d, []).append(s)

    ops: list[TimedOp] = []
    stream_of: dict[int, str] = {}     # 原 DAG 节点 id → 已发 TimedOp 的 stream（同流 dep 过滤）
    sp_active = deg.sequence_parallel
    feat_sharded = False

    for n in dag.nodes:
        module = n.module or ""
        out_name, out_sym, out_dt = _parse_ref(n.out) if n.out else ("", "", "bf16")

        if n.op == "MatMul" and deg.tp > 1 and module not in _KNOWN_MATMUL_MODULES:
            raise ValueError(
                f"producer: MatMul@{n.src} 的 module {module!r} 不在已知集"
                f"{sorted(_KNOWN_MATMUL_MODULES - {''})}（tp>1 下静默不切分会错切——fail-loud）")

        this_stream = STREAM_HOST_ONLY if n.op == "View" else STREAM_DEVICE
        base_deps = tuple(
            id2opid[p] for p in in_edges.get(n.id, ())
            if stream_of.get(p) != this_stream
        )

        if n.op == "MatMul" and module in (_COL, _ROW):
            _, x_sym, x_dt = _parse_ref(n.ins[0])
            in_dim, out_dim = n.attrs.get("in_dim"), n.attrs.get("out_dim")
            if not in_dim or not out_dim:
                raise ValueError(
                    f"producer: MatMul@{n.src}（{module}）缺 in_dim/out_dim attrs"
                    f"（PART A 未标注线性维度，fail-loud）")
            weight_sym = f"{_wrap_dim(in_dim)}·{_wrap_dim(out_dim)}"

            deps = base_deps
            if module == _COL and deg.tp > 1 and sp_active:
                gather_in = _local_shape(x_sym, dims, deg, sp_active=True, feat_tp=1)
                gather_out = _local_shape(x_sym, dims, deg, sp_active=False, feat_tp=1)
                gather_op = TimedOp(
                    op_id=id2opid[n.id] + ".ag", op_type="CommOp", phase=phase,
                    in_shapes=(gather_in,), out_shape=gather_out, dtype=x_dt,
                    stream=COMM_STREAM["tp"], src=n.src, deps=deps,
                    module="injected:module-semantics",
                    comm=CommSpec("all_gather", _bytes_of(gather_in, x_dt), "tp", deg.tp))
                ops.append(gather_op)
                deps = (gather_op.op_id,)
                sp_active = False

            if module == _COL and deg.tp > 1:
                feat_sharded = True   # Column 输出起，feature 轴进入 tp 分片区

            x_local = _local_shape(x_sym, dims, deg, sp_active=sp_active,
                                   feat_tp=(deg.tp if feat_sharded else 1))
            weight_shape = weight_local(weight_sym, dims, module, deg.tp)
            # Row 矩乘的（通信前）输出恒为全 seq、非 feature 轴（H 轴本不在 _FEATURE_SYMS）；
            # Column 矩乘的输出此时 sp_active 已由上面的 gather 逻辑正确落定（未 sp 则从未 True 过）。
            out_sp = False if module == _ROW else sp_active
            out_shape = _out_shape_or_fail(out_sym, STREAM_DEVICE, n.src, dims, deg, out_sp,
                                           feat_tp=(deg.tp if feat_sharded else 1))
            mm_op = TimedOp(op_id=id2opid[n.id], op_type="MatMul", phase=phase,
                            in_shapes=(x_local, weight_shape), out_shape=out_shape,
                            dtype=out_dt, stream=STREAM_DEVICE, src=n.src, deps=deps,
                            module=module)
            ops.append(mm_op)
            stream_of[n.id] = STREAM_DEVICE

            if module == _ROW and deg.tp > 1:
                feat_sharded = False   # Row 输入消费完毕，退出 feature 分片区
                full = _local_shape(out_sym, dims, deg, sp_active=False, feat_tp=1)
                ctype = "reduce_scatter" if deg.sequence_parallel else "all_reduce"
                out_after = _local_shape(out_sym, dims, deg, sp_active=deg.sequence_parallel,
                                         feat_tp=1)
                rs_op = TimedOp(
                    op_id=id2opid[n.id] + ".rs", op_type="CommOp", phase=phase,
                    in_shapes=(full,), out_shape=out_after, dtype=out_dt,
                    stream=COMM_STREAM["tp"],
                    src="layers.py:619" if ctype == "reduce_scatter" else "layers.py:621",
                    deps=(mm_op.op_id,), module=module,
                    comm=CommSpec(ctype, _bytes_of(full, out_dt), "tp", deg.tp))
                ops.append(rs_op)
                sp_active = deg.sequence_parallel
            continue

        # —— 通用节点（View/Norm/Cast/Activation/Elementwise/module=="" 的复制式 MatMul 等）——
        feat_tp = deg.tp if feat_sharded else 1
        in_shapes = tuple(
            _in_shape_or_empty(_parse_ref(ref)[1], dims, deg, sp_active, feat_tp)
            for ref in n.ins
        )
        out_shape = _out_shape_or_fail(out_sym, this_stream, n.src, dims, deg, sp_active, feat_tp)
        ops.append(TimedOp(op_id=id2opid[n.id], op_type=n.op, phase=phase,
                           in_shapes=in_shapes, out_shape=out_shape, dtype=out_dt,
                           stream=this_stream, src=n.src, deps=base_deps, module=module))
        stream_of[n.id] = this_stream

    return TimedSegment(seg_id=seg_id, ops=tuple(ops))
