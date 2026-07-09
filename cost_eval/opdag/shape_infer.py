# cost_eval/opdag/shape_infer.py
"""PART B(T9):shape 推断 pass。

`infer_shapes(dag, input_shapes, dims_ctx=None)`:从 construct 的输入种子出发,按**源序**逐节点
传播符号 shape,把每个 ref 的 shape 段(`name:?:dtype` 里的 `?`)回填为真符号 shape,并返回同一个
`OpDAG`(原地改写 nodes 的 ins/out)。

设计取向(与 sym_shape 的"决不杜撰"一致):
  * 每个算子按其**类型**施一条 shape 规则(MatMul 换末轴 / GroupedMatMul 换末轴为权重末轴 /
    Norm·Cast·Elementwise 透传 / Activation-swiglu 末轴折半 / View 各子类型 / FlashAttention 取
    query·value)。规则里凡是 `sym_shape` 代数返回 None(消元/整除不干净)→ 该张量保留 `?`,继续。
  * **fail-loud 仅一处**:MatMul 输入 shape 已知却无 `out_dim` attr —— 说明 PART A 漏了该 linear 的
    输出维度,静默留 `?` 会掩盖 bug,故带节点 `src` 抛 ValueError。别处一律"解不出就保 `?`"。
  * **未种子的外部输入**(既不在种子 env、也无产出节点)→ 保 `?`,不 raise。

env 键:张量的**基名**(ref 的第一段)。种子来自 `input_shapes`;之后每个节点把产出写回 env。
**内联后缀容忍**:walker 内联会把变量改名 `x__i<frame>`,故查 env 既按精确名、也按 `key+"__i"` 前缀
匹配种子(如 ref `dispatched_input__i0` 命中种子 `dispatched_input`)。

标量轴名(reshape/split 表达式里的 `seq`/`q_len`/`bs`):两条来源——
  * `dag.scalar_binds`(`seq,bs,h = x.shape` 这类属性解包)→ 惰性按 src 当时的 env shape 逐轴解,缓存;
  * View `shape` 节点(`self.shape(x)` 原语)→ 处理到该节点时按 shape_src 逐轴登记到同一 scalar_map。

`self.<attr>` / `self.config.<attr>` / `-1` 表达式经 `_resolve_token` 解成 `Factors`/`NEG1`:
`self.config.X`→CONFIG2SYM;`self.X`→dims_ctx(退 CONFIG2SYM);标量名→scalar_map;支持 `*`/`+`/`//` 组合。
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .schema import OpDAG, OpNode
from .sym_shape import (
    Factors, NEG1, CONFIG2SYM,
    parse_shape, render_shape, parse_axis,
    mul, add, floordiv, resolve_reshape,
)

_QMARK = "?"
# walker 内联时用的合成别名前缀(链式视图 / return 物化):这些名字**不会**被下游按名消费,
# 下游改用别名名 + 一条数据流边引用它们 → 唯一需要靠 edge 桥接的场景。
_SYNTH_PREFIXES = ("__chain__", "__ret__")


def _base(ref: str) -> str:
    return ref.split(":", 1)[0]


def _with_shape(ref: str, shape: str) -> str:
    """把 ref 的 shape 段(中段)替换为 shape。ref 恒为 `name:shape:dtype` 三段。"""
    parts = ref.split(":")
    if len(parts) != 3:
        return ref
    parts[1] = shape
    return ":".join(parts)


def _is_synth(name: str) -> bool:
    return any(name.startswith(p) for p in _SYNTH_PREFIXES)


@dataclass
class _Ctx:
    env: dict = field(default_factory=dict)          # 张量基名 -> shape 串(已解出的才入)
    dims_ctx: dict = field(default_factory=dict)     # self.<attr> -> 符号 token 串
    scalar_binds: list = field(default_factory=list) # [{"names":[...], "src": var}]
    scalar_map: dict = field(default_factory=dict)   # 标量轴名 -> Factors | None


# ── env 查询(内联后缀容忍)────────────────────────────────────────────────────
def _lookup(env: dict, name: str):
    if name in env:
        return env[name]
    for key, shp in env.items():
        if name.startswith(key + "__i"):
            return shp
    return None


# ── 标量轴名解析(scalar_binds 惰性 + View shape 节点即时)────────────────────────
def _scalar(ctx: _Ctx, name: str):
    if name in ctx.scalar_map:
        return ctx.scalar_map[name]
    for sb in ctx.scalar_binds:
        if name in sb.get("names", []):
            _fill_scalar_bind(ctx, sb)
            return ctx.scalar_map.get(name)
    return None


def _fill_scalar_bind(ctx: _Ctx, sb: dict) -> None:
    """`names = src.shape`:按 src 当前 env shape 逐轴填 names(仅在 src 已解出时填,setdefault
    避免覆盖 shape 节点的权威登记;src 未解出则暂不缓存,留待后续重试)。"""
    shp = _lookup(ctx.env, sb.get("src", ""))
    if not shp:
        return
    axes = parse_shape(shp)
    for i, nm in enumerate(sb.get("names", [])):
        ctx.scalar_map.setdefault(nm, axes[i] if i < len(axes) else None)


# ── 维度表达式解析:token 串 -> Factors | NEG1 | None ──────────────────────────
def _resolve_token(tok: str, ctx: _Ctx):
    tok = tok.strip()
    if tok == "-1":
        return NEG1
    try:
        node = ast.parse(tok, mode="eval").body
    except SyntaxError:
        return None
    return _eval_expr(node, ctx)


def _eval_expr(node, ctx: _Ctx):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return Factors(coeff=node.value)
        return None
    if isinstance(node, ast.Name):
        return _scalar(ctx, node.id)
    if isinstance(node, ast.Attribute):
        return _attr_dim(node, ctx)
    if isinstance(node, ast.BinOp):
        a = _eval_expr(node.left, ctx)
        if a is None or a is NEG1:
            return None
        if isinstance(node.op, ast.Mult):
            b = _eval_expr(node.right, ctx)
            return mul(a, b) if (b is not None and b is not NEG1) else None
        if isinstance(node.op, ast.Add):
            b = _eval_expr(node.right, ctx)
            return add(a, b) if (b is not None and b is not NEG1) else None
        if isinstance(node.op, ast.FloorDiv):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                return floordiv(a, node.right.value)
            return None
    return None


def _attr_dim(node: ast.Attribute, ctx: _Ctx):
    v = node.value
    # self.config.<X> / config.<X> → CONFIG2SYM(未映射则保留原名,与 init_dims 一致,决不映数字)
    if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
            and v.value.id == "self" and v.attr == "config"):
        return parse_axis(CONFIG2SYM.get(node.attr, node.attr))
    if isinstance(v, ast.Name) and v.id == "config":
        return parse_axis(CONFIG2SYM.get(node.attr, node.attr))
    # self.<X> → dims_ctx(__init__ 求得的符号维度),退 CONFIG2SYM,再无 → None(保 ?)
    if isinstance(v, ast.Name) and v.id == "self":
        attr = node.attr
        if attr in ctx.dims_ctx:
            return parse_axis(ctx.dims_ctx[attr])
        if attr in CONFIG2SYM:
            return parse_axis(CONFIG2SYM[attr])
    return None


# ── 逐 op shape 规则(输入是各操作数的 Factors 列表,None=未解出)────────────────
def _passthrough(in_axes_list):
    a = in_axes_list[0] if in_axes_list else None
    return [f.copy() for f in a] if a else None


def _matmul(node: OpNode, in_axes_list):
    a = in_axes_list[0] if in_axes_list else None
    if not a:                          # 输入未知 → 保 ?,不 raise
        return None
    out_dim = node.attrs.get("out_dim")
    if not out_dim:                    # 输入已知却无 out_dim → fail-loud(带 src)
        raise ValueError(
            f"shape_infer: MatMul 输入 shape 已知却缺 out_dim attr（{node.src}）—— "
            f"PART A 漏该 linear 输出维度,fail-loud"
        )
    out = [f.copy() for f in a]
    out[-1] = parse_axis(out_dim)
    return out


def _grouped_matmul(in_axes_list):
    a = in_axes_list[0] if len(in_axes_list) > 0 else None
    w = in_axes_list[1] if len(in_axes_list) > 1 else None
    if not a or not w:
        return None
    out = [f.copy() for f in a]
    out[-1] = w[-1].copy()
    return out


def _flash_attention(in_axes_list):
    q = in_axes_list[0] if len(in_axes_list) > 0 else None
    v = in_axes_list[2] if len(in_axes_list) > 2 else None
    if not q or not v:
        return None
    out = [f.copy() for f in q]
    out[-1] = v[-1].copy()
    return out


def _activation(node: OpNode, in_axes_list):
    a = in_axes_list[0] if in_axes_list else None
    if not a:
        return None
    if node.attrs.get("activation_type") == "swiglu":
        last = floordiv(a[-1], 2)
        if last is None:
            return None
        out = [f.copy() for f in a]
        out[-1] = last
        return out
    return [f.copy() for f in a]       # 非 swiglu 激活:透传


def _concat(in_axes_list, axis: int):
    if any(a is None for a in in_axes_list) or not in_axes_list:
        return None
    base = [f.copy() for f in in_axes_list[0]]
    ax = axis if axis >= 0 else len(base) + axis
    if not (0 <= ax < len(base)):
        return None
    acc = base[ax]
    for other in in_axes_list[1:]:
        if ax >= len(other):
            return None
        acc = add(acc, other[ax])
    base[ax] = acc
    return base


def _tile(in_axes, tile_mult, ctx: _Ctx):
    out = [f.copy() for f in in_axes]
    for i, m in enumerate(tile_mult):
        if i >= len(out) or str(m).strip() == "1":
            continue
        fac = _resolve_token(str(m), ctx)
        if fac is None or fac is NEG1:
            continue                   # 解不出该轴倍数 → 保持该轴不变
        out[i] = mul(out[i], fac)
    return out


def _expand_dims(in_axes, axis: int):
    out = [f.copy() for f in in_axes]
    ax = axis if axis >= 0 else len(out) + axis + 1
    ax = max(0, min(ax, len(out)))
    out.insert(ax, Factors(coeff=1))
    return out


def _reshape(in_axes, reshape_dims, ctx: _Ctx):
    target = []
    for tok in reshape_dims:
        r = _resolve_token(str(tok), ctx)
        if r is None:
            return None
        target.append(r)
    return resolve_reshape(in_axes, target)


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def infer_shapes(dag: OpDAG, input_shapes: dict, dims_ctx: dict | None = None) -> OpDAG:
    """从 input_shapes 种子出发,逐节点符号传播 shape,回填 ins/out 的 shape 段,返回同一 dag。"""
    ctx = _Ctx(
        env=dict(input_shapes or {}),
        dims_ctx=dict(dims_ctx if dims_ctx is not None else getattr(dag, "dims_ctx", {}) or {}),
        scalar_binds=list(getattr(dag, "scalar_binds", []) or []),
    )

    # 入边:consumer id -> [producer id...](按 edge 顺序,供合成别名桥接)
    incoming: dict[int, list[int]] = {}
    for e in dag.edges:
        if len(e) == 2:
            incoming.setdefault(e[1], []).append(e[0])

    node_out_shape: dict[int, str] = {}   # 节点 id -> 已解出 out shape 串
    node_out_name: dict[int, str] = {}    # 节点 id -> out 基名

    for n in dag.nodes:
        node_out_name[n.id] = _base(n.out) if n.out else ""

        # 1) 解各输入 shape(先按名/种子,内联后缀容忍)
        shapes = [_lookup(ctx.env, _base(r)) for r in n.ins]

        # 2) 合成别名桥接:未按名解出的输入 ← 该节点的合成产出(__chain__/__ret__)按序补
        synth_prods = [p for p in incoming.get(n.id, [])
                       if _is_synth(node_out_name.get(p, "")) and p in node_out_shape]
        li = 0
        for i in range(len(shapes)):
            if shapes[i] is None and li < len(synth_prods):
                shapes[i] = node_out_shape[synth_prods[li]]
                li += 1

        # 3) 回填输入 ref 的 shape 段(供 derive_saves 读到真 shape)
        for i, sh in enumerate(shapes):
            if sh is not None:
                n.ins[i] = _with_shape(n.ins[i], sh)

        in_axes_list = [parse_shape(sh) if sh is not None else None for sh in shapes]

        # 4) View "shape" 原语:即时登记标量轴名(权威,覆盖),不产张量输出
        if n.op == "View" and n.attrs.get("view") == "shape":
            src_axes = in_axes_list[0] if in_axes_list else None
            for i, nm in enumerate(n.attrs.get("shape_unpack", [])):
                ctx.scalar_map[nm] = (src_axes[i] if (src_axes and i < len(src_axes)) else None)
            node_out_name[n.id] = _base(n.out) if n.out else ""
            continue

        # 5) View "split":每个目标各得其 size 轴,全部写回 env;node.out = 首目标
        if n.op == "View" and n.attrs.get("view") == "split":
            out_shape = _apply_split(n, in_axes_list, ctx)
            if out_shape is not None:
                node_out_shape[n.id] = out_shape
                if n.out:
                    n.out = _with_shape(n.out, out_shape)
            continue

        # 6) 其余 op:算出 out 轴列表
        out_axes = _dispatch(n, in_axes_list, ctx)
        if out_axes is not None:
            out_shape = render_shape(out_axes)
            node_out_shape[n.id] = out_shape
            if n.out:
                n.out = _with_shape(n.out, out_shape)
                ctx.env[_base(n.out)] = out_shape

    return dag


def _apply_split(n: OpNode, in_axes_list, ctx: _Ctx):
    """split:输入沿 split_dim 切成若干 target,各 target 该轴换成对应 size。返回首目标 shape 串。"""
    in_axes = in_axes_list[0] if in_axes_list else None
    if not in_axes:
        return None
    targets = n.attrs.get("split_targets", [])
    sizes = n.attrs.get("split_sizes", [])
    split_dim = n.attrs.get("split_dim", -1)
    ax = split_dim if split_dim >= 0 else len(in_axes) + split_dim
    if not (0 <= ax < len(in_axes)):
        return None
    first_shape = None
    for i, tgt in enumerate(targets):
        sz = _resolve_token(str(sizes[i]), ctx) if i < len(sizes) else None
        if sz is None or sz is NEG1:
            continue
        new_axes = [f.copy() for f in in_axes]
        new_axes[ax] = sz
        shp = render_shape(new_axes)
        ctx.env[tgt] = shp
        if i == 0:
            first_shape = shp
    return first_shape


def _dispatch(n: OpNode, in_axes_list, ctx: _Ctx):
    op = n.op
    if op == "MatMul":
        return _matmul(n, in_axes_list)
    if op == "GroupedMatMul":
        return _grouped_matmul(in_axes_list)
    if op == "FlashAttention":
        return _flash_attention(in_axes_list)
    if op == "Activation":
        return _activation(n, in_axes_list)
    if op in ("Norm", "Cast", "Elementwise", "Softmax", "Dropout", "Gather"):
        return _passthrough(in_axes_list)
    if op == "View":
        view = n.attrs.get("view")
        if view == "concat":
            return _concat(in_axes_list, int(n.attrs.get("concat_axis", 0)))
        if view == "tile":
            a = in_axes_list[0] if in_axes_list else None
            return _tile(a, n.attrs.get("tile_mult", []), ctx) if a else None
        if view == "expand_dims":
            a = in_axes_list[0] if in_axes_list else None
            ax = n.attrs.get("expand_axis")
            return _expand_dims(a, int(ax)) if (a and ax is not None) else None
        if view == "reshape":
            a = in_axes_list[0] if in_axes_list else None
            return _reshape(a, n.attrs.get("reshape_dims", []), ctx) if a else None
        # transpose / 其它未列举视图:形状(轴的多重集)不变 → 透传(best-effort)
        return _passthrough(in_axes_list)
    # 未知 op:保守透传首输入(不杜撰)
    return _passthrough(in_axes_list)
