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
    parse_shape, render_shape, parse_axis, product_of, render_term,
    mul, add, floordiv, resolve_reshape,
    mark_numel_only, strip_numel_only,
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
    scalar_map: dict = field(default_factory=dict)   # (帧, 标量轴名) -> Factors | None
    # ── 内联帧作用域(G3/G5,2026-07-25)────────────────────────────────────────────────
    # `frame` = 当前正在推断的节点所属的内联帧(`OpNode.attrs["frame"]`,根帧 = "");
    # `node_dims` = 该节点所属类的 `dims_ctx`(`OpNode.attrs["dims_ctx"]`),优先于全局那份。
    # 为什么必须按帧:同一个类的两个构造点会让 `self.head_dim` 解出不同符号
    # (`v_head_dim` @ csa.py:604 vs `index_head_dim` @ indexer.py:128),局部标量名
    # (`seqlen`)也会在父子帧里指不同张量。扁平一张表只能 fail-loud 或静默取一个。
    frame: str = ""
    node_dims: dict | None = None
    # 逐节点"解不出"台账(调用方传 `report=[]` 才收):每条 {node,op,src,name,reason}。
    # 纪律(任务书第 2 点):**任何解不出的东西都必须显式带 `file:line` 出现在这里**,
    # 绝不用一个"看起来合理"的数替代。
    report: list | None = None
    # dtype 订正表(张量基名 -> 真 dtype):目前只装 `Compare` 的 bool 产出。
    # walker 的 `_emit` 按 ins 推产出 dtype,故比较类算子的产出被记成 compute dtype(bf16)= 2×;
    # 订正必须**连带下游 ins ref** 一起改,否则 `derive_saves` 读的是下游那份旧串。
    dtype_fix: dict = field(default_factory=dict)


#: 未解析原因码 → 一句话(**带定位符**)。报表直接引用,避免每处现编措辞。
UNRESOLVED_REASONS = {
    "no_out_dim": "MatMul 输入 shape 已知却缺 `out_dim` attr —— PART A 未求出该 linear 的输出维度",
    "no_input_shape": "输入 shape 未解出(上游未解析 / 未种子的外部输入)",
    "reduce_axis_unknown": "归约算子(sum/mean/max)的 `dim` **未被 walker 记进 attrs** →"
                           " 被约掉的轴长未知 → 元素数不可知(passthrough 会**多算**该轴倍)",
    "chunk_axis_unknown": "`chunk` 的切分份数/轴 **未被 walker 记进 attrs** →"
                          " 元素数不可知(passthrough 会多算 n 倍)",
    "concat_axis_unknown": "`cat` 的轴 **未被 walker 记进 attrs**,且各输入 shape 不同 →"
                           " 猜轴会算错(实测 `cat([kv_nope, kv_pe], -1)` 两输入末轴不同)",
    "slice_bounds_unknown": "`slice` 的起止/步长解不出(`attrs['index']` 里的标量名无来源)",
    "constant_shape_unknown": "常量构造(arange/full/zeros/ones)的 shape 实参 **未被 walker 记进"
                              " attrs** → 无从得知形状",
    "reshape_unresolved": "reshape 目标维里有解不出的 token(construct 局部标量未导出 /"
                          " `self.<attr>` 不在 dims_ctx)",
    "split_size_unresolved": "split 的 size 表达式解不出",
    "needs_axis_structure": "上游只解出**元素数**、无轴结构(`~` 标记),而本算子要按轴改形 →"
                            " 拒绝按假轴序往下算",
    "tile_mult_unresolved": "tile 的倍数表达式解不出",
}


def merge_dims_ctx(*ctxs) -> dict:
    """把多个 Cell 的 `dims_ctx` 并成一张表;**同名不同值即 fail-loud**。

    为什么需要:`extract_cell(recurse=True)` 把子 Cell **内联**进父图,但 `dag.dims_ctx` 只装
    **顶层** Cell 的 `__init__` 求值结果(`extractor.py:725`)→ 内联进来的节点里
    `self.<attr>` 引用的是**它自己那个类**的属性,顶层表里没有。实测后果:
    `indexer.py:178` `reshape(q, (seqlen, bsz, self.index_n_heads, self.index_head_dim))`
    在抽 `CompressedSparseAttention` 时解不出(那两个符号只在 `CSAIndexer.__init__` 里)。

    合并是**安全**的做法(不是猜):每个 `dims_ctx` 都由各自类的 `__init__` 静态求值得到,
    值是符号串;冲突(同名不同符号)会**抛**,不会静默取一个。调用方把链上各 Cell 的
    `dag.dims_ctx` 都传进来即可。
    """
    out: dict = {}
    for c in ctxs:
        for k, v in (c or {}).items():
            prev = out.setdefault(k, v)
            if prev != v:
                raise ValueError(
                    f"merge_dims_ctx: `self.{k}` 在不同 Cell 的 __init__ 里解出不同符号维度 "
                    f"({prev!r} vs {v!r}) —— 合并会静默取一个,fail-loud。"
                    f"请按 Cell 分别做 shape 推断,或让 extractor 传播构造实参。")
    return out


def _note(ctx: _Ctx, n: OpNode, reason: str, extra: str = "") -> None:
    """记一条"解不出"。`report is None` 时静默(既有调用方逐字不变)。"""
    if ctx.report is None:
        return
    ctx.report.append({
        "node": n.id, "op": n.op, "src": n.src,
        "name": _base(n.out) if n.out else "",
        "reason": reason,
        "detail": (UNRESOLVED_REASONS.get(reason, reason)
                   + (f"；{extra}" if extra else "")),
        "prim": n.attrs.get("prim") or n.attrs.get("view") or "",
    })


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
    """标量轴名 → Factors。**按帧**查:先当前帧,再根帧(父自己的名字对子也可见 —— 子的
    `reshape` 表达式里不会引用父的局部名,故这一档只是既有单帧行为的自然延续)。"""
    for fr in _frame_chain(ctx.frame):
        if (fr, name) in ctx.scalar_map:
            return ctx.scalar_map[(fr, name)]
        for sb in ctx.scalar_binds:
            if sb.get("frame", "") == fr and name in sb.get("names", []):
                _fill_scalar_bind(ctx, sb, fr)
                return ctx.scalar_map.get((fr, name))
    return None


def _frame_chain(frame: str) -> list[str]:
    """帧的**可见链**:当前帧 → 逐级外层 → 根帧("")。`a@1/b@7` → `["a@1/b@7", "a@1", ""]`。"""
    out, cur = [], frame
    while cur:
        out.append(cur)
        cur = cur.rsplit("/", 1)[0] if "/" in cur else ""
    out.append("")
    return out


def _fill_scalar_bind(ctx: _Ctx, sb: dict, frame: str = "") -> None:
    """`names = src.shape`:按 src 当前 env shape 逐轴填 names(仅在 src 已解出时填,setdefault
    避免覆盖 shape 节点的权威登记;src 未解出则暂不缓存,留待后续重试)。

    ⚠ src 是 `numel_only`(`~`)时**逐轴绑定不成立** —— 那种 shape 只有一条"总积"轴,按位取轴
    会把 `b, s, n, d = x.shape` 绑成完全错的值(实测泄漏样例:`q_bm` 被解成
    `(B·~S)·n_heads·v_head_dim` —— `~` 都跑进轴里了)。故一律绑 None(保 `?`)。
    """
    raw = _lookup(ctx.env, sb.get("src", ""))
    if not raw:
        return
    shp, numel_only = strip_numel_only(raw)
    if numel_only:
        for nm in sb.get("names", []):
            ctx.scalar_map.setdefault((frame, nm), None)
        return
    axes = parse_shape(shp)
    for i, nm in enumerate(sb.get("names", [])):
        ctx.scalar_map.setdefault((frame, nm), axes[i] if i < len(axes) else None)


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
            # 除数是个**解出来只有系数、没有符号**的表达式(如 `self.compress_ratio` → `4`):
            # 按该整数整除。真源 `deepseek_v4_hybrid_attention.py:276`
            # `d = self.query_projection_size // o_groups`、`csa.py:779` `positions // ratio`。
            b = _eval_expr(node.right, ctx)
            if b is not None and b is not NEG1 and not b.syms and b.coeff:
                return floordiv(a, b.coeff)
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
    # G3:**先查本节点所属类那份**(`attrs["dims_ctx"]`,由 `_inline_subcell` 按帧挂上),
    # 再退顶层那份。次序不能反 —— `self.head_dim` 在两个 compressor 构造点解出不同符号。
    if isinstance(v, ast.Name) and v.id == "self":
        attr = node.attr
        if ctx.node_dims and attr in ctx.node_dims:
            return parse_axis(ctx.node_dims[attr])
        if attr in ctx.dims_ctx:
            return parse_axis(ctx.dims_ctx[attr])
        if attr in CONFIG2SYM:
            return parse_axis(CONFIG2SYM[attr])
    return None


# ── 逐 op shape 规则 ──────────────────────────────────────────────────────────
# 约定:每条规则收 `(node, in_axes_list, in_numel_only, ctx)`,返回 `(out_axes, numel_only)`
# 或 `None`(解不出 —— **必须**先 `_note(...)` 记账)。`in_axes_list[i] is None` = 该输入未解出。
#
# 三条纪律(任务书第 2/5 点):
#   ① 元素数不可知就返回 None + 记账,**绝不** passthrough 顶替(那会**多算**被约掉的轴);
#   ② 元素数可知但**轴结构不可知**(walker 未记轴)→ 走 `numel_only`(`~` 前缀),它对字节记账
#      够用(契约只要 `local_numel`),但需要轴结构的下游算子必须拒绝它;
#   ③ 广播/整除等**语义已知**的推断照做(那是算子定义,不是猜)。

def _passthrough(in_axes_list, in_numel_only):
    """形状与元素数都不变的算子(cast / norm / softmax / dropout / permute / roll / squeeze)。

    注意 permute/transpose:轴的**多重集**不变、元素数不变;walker 未记轴序,故轴**顺序**
    可能与真实不符 → 标 `numel_only`(字节正确、结构不可信),不让下游按假轴序改形。
    """
    a = in_axes_list[0] if in_axes_list else None
    if not a:
        return None
    return [f.copy() for f in a], (in_numel_only[0] if in_numel_only else False)


def _numel_axes(axes):
    """把一列轴压成"单轴 = 总积"(numel_only 的载体)。"""
    return [product_of(axes)]


def _broadcast(in_axes_list, in_numel_only, ctx, n):
    """逐元素算子的**广播**结果(算子定义,不是猜):右对齐,每位取非 1 的那个轴。

    仅当**全部**输入已解出且都有轴结构时才用;两个都非 1 且符号不同 → 判不出 → None。
    """
    if not in_axes_list or any(a is None for a in in_axes_list) or any(in_numel_only):
        return None
    rank = max(len(a) for a in in_axes_list)
    out = []
    for k in range(rank):
        cand = []
        for a in in_axes_list:
            j = k - (rank - len(a))
            if j >= 0:
                cand.append(a[j])
        non1 = [f for f in cand if not (f.coeff == 1 and not f.syms)]
        if not non1:
            out.append(Factors(coeff=1))
            continue
        terms = {render_term(f) for f in non1}
        if len(terms) != 1:
            return None                 # 两个都非 1 且不同 → 广播结果判不出(不猜)
        out.append(non1[0].copy())
    return out, False


def _elementwise(n: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    """逐元素算子。**归约**(sum/mean/max 带 `reduce`)另走:轴未记录 → 元素数不可知 → 记账。"""
    if n.attrs.get("reduce"):
        _note(ctx, n, "reduce_axis_unknown")
        return None
    b = _broadcast(in_axes_list, in_numel_only, ctx, n)
    if b is not None:
        return b
    return _passthrough(in_axes_list, in_numel_only)     # 退回首输入(既有行为)


def _matmul(node: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    a = in_axes_list[0] if in_axes_list else None
    if not a:                          # 输入未知 → 保 ?,不 raise
        _note(ctx, node, "no_input_shape")
        return None
    if in_numel_only and in_numel_only[0]:
        _note(ctx, node, "needs_axis_structure", "MatMul 要换**末轴**")
        return None
    out_dim = node.attrs.get("out_dim")
    if not out_dim:
        # 输入已知却无 out_dim。**默认 fail-loud**(逐字保留:静默留 `?` 会掩盖 PART A 的 bug);
        # 调用方显式传 `report=` 时改为记账 —— 因为存在**已知的合法缺口**:构造点传入的维度
        # (`Compressor(head_dim=...)` @ csa.py:604/128)没有被 extractor 传播到子 Cell 的
        # `__init__` 求值里(见 docs/opdag_bytes §4 的 G2)。记账比崩掉整张图更有信息量。
        if ctx.report is None:
            raise ValueError(
                f"shape_infer: MatMul 输入 shape 已知却缺 out_dim attr（{node.src}）—— "
                f"PART A 漏该 linear 输出维度,fail-loud"
            )
        _note(ctx, node, "no_out_dim", f"module={node.attrs.get('module')}")
        return None
    out = [f.copy() for f in a]
    out[-1] = parse_axis(out_dim)
    return out, False


def _grouped_matmul(n: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    a = in_axes_list[0] if len(in_axes_list) > 0 else None
    w = in_axes_list[1] if len(in_axes_list) > 1 else None
    if not a or not w:
        _note(ctx, n, "no_input_shape")
        return None
    if any(in_numel_only[:2]):
        _note(ctx, n, "needs_axis_structure", "GroupedMatMul 要取权重末轴")
        return None
    out = [f.copy() for f in a]
    out[-1] = w[-1].copy()
    return out, False


def _flash_attention(n: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    q = in_axes_list[0] if len(in_axes_list) > 0 else None
    v = in_axes_list[2] if len(in_axes_list) > 2 else None
    if not q or not v:
        _note(ctx, n, "no_input_shape")
        return None
    if in_numel_only and (in_numel_only[0] or (len(in_numel_only) > 2 and in_numel_only[2])):
        _note(ctx, n, "needs_axis_structure", "FlashAttention 要取 value 末轴")
        return None
    out = [f.copy() for f in q]
    out[-1] = v[-1].copy()
    return out, False


def _bmm(n: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    """`mint.bmm(a, b)`:`[..., m, k] @ [..., k, n]` → `[..., m, n]`(算子定义)。"""
    a = in_axes_list[0] if len(in_axes_list) > 0 else None
    b = in_axes_list[1] if len(in_axes_list) > 1 else None
    if not a or not b:
        _note(ctx, n, "no_input_shape")
        return None
    if any(in_numel_only[:2]) or len(a) < 2 or len(b) < 2:
        _note(ctx, n, "needs_axis_structure", "BMM 要取两操作数的末两轴")
        return None
    out = [f.copy() for f in a]
    out[-1] = b[-1].copy()
    return out, False


def _activation(node: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    a = in_axes_list[0] if in_axes_list else None
    if not a:
        _note(ctx, node, "no_input_shape")
        return None
    nm = in_numel_only[0] if in_numel_only else False
    if node.attrs.get("activation_type") == "swiglu":
        # swiglu 末轴折半 = 元素数折半;numel_only 下"末轴"就是总积,折半同样精确。
        last = floordiv(a[-1], 2)
        if last is None:
            return None
        out = [f.copy() for f in a]
        out[-1] = last
        return out, nm
    return [f.copy() for f in a], nm    # 非 swiglu 激活:透传


def _concat(n: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    """`cat`:**元素数 = 各输入元素数之和**(与轴无关,故精确);轴结构未记 → `numel_only`。

    此前实现按 `attrs["concat_axis"]`(自由调用点**没有**这个 attr,默认 0)在轴 0 上相加 ——
    实测会算错:`compressor.py:243` `cat([kv_nope, kv_pe], dim=-1)` 两输入末轴不同
    (`d-qk_pos_emb_head_dim` vs `qk_pos_emb_head_dim`),按轴 0 合并给出 `2n·b·(d-64)`
    而真值是 `n·b·d`。轴已记(`concat_axis` in attrs,由 `init_binder._concat_axis` 从
    `__init__` 绑定点求得)时仍走精确的按轴相加。
    """
    if not in_axes_list or any(a is None for a in in_axes_list):
        _note(ctx, n, "no_input_shape")
        return None
    axis = n.attrs.get("concat_axis")
    if axis is not None and not any(in_numel_only):
        base = [f.copy() for f in in_axes_list[0]]
        ax = int(axis) if int(axis) >= 0 else len(base) + int(axis)
        if 0 <= ax < len(base):
            acc = base[ax]
            ok = True
            for other in in_axes_list[1:]:
                if ax >= len(other):
                    ok = False
                    break
                acc = add(acc, other[ax])
            if ok:
                base[ax] = acc
                return base, False
    total = product_of(in_axes_list[0])
    for a in in_axes_list[1:]:
        total = add(total, product_of(a))
    return [total], True


def _chunk(n: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    """`chunk(x, k, dim=?)`:轴未记,但**份数 k 可由源侧的元组解包元数证得**
    (`tensor_prev, tensor_next = self.chunk(tensor, 2, dim=-1)` @ `compressor.py:169`
    —— Python 的元组解包只有在 chunk 恰返回 k 份时才成立,故 `len(attrs['outs'])` 就是 k,
    这是**源侧事实**,不是猜)。元素数 = 总积 / k,且只在**符号上能精确整除**时才认
    (不能精确整除 = 各份可能不等长 → 记账,不给一个 floor 值)。
    """
    a = in_axes_list[0] if in_axes_list else None
    if not a:
        _note(ctx, n, "no_input_shape")
        return None
    k = len(n.attrs.get("outs") or ())
    if k < 2:
        _note(ctx, n, "chunk_axis_unknown", "`outs` 元数 < 2,无法据解包元数证得份数")
        return None
    total = product_of(a)
    if total.coeff % k != 0:
        _note(ctx, n, "chunk_axis_unknown",
              f"总积 {render_term(total)} 的系数不被 {k} 精确整除 → 各份可能不等长")
        return None
    return [floordiv(total, k)], True


def _slice(n: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    """`x[<index>]`:仅支持 `":<stop>"` 这一种能从 attrs 解出的形态(切轴 0)。其余记账。"""
    a = in_axes_list[0] if in_axes_list else None
    if not a:
        _note(ctx, n, "no_input_shape")
        return None
    idx = str(n.attrs.get("index") or "")
    if "," in idx or idx.count(":") != 1 or not idx.startswith(":"):
        _note(ctx, n, "slice_bounds_unknown", f"index={idx!r}")
        return None
    stop = _resolve_token(idx[1:], ctx)
    if stop is None or stop is NEG1:
        _note(ctx, n, "slice_bounds_unknown", f"index={idx!r} 的 stop 解不出")
        return None
    if in_numel_only and in_numel_only[0]:
        _note(ctx, n, "needs_axis_structure", "slice 要换轴 0")
        return None
    out = [f.copy() for f in a]
    out[0] = stop
    return out, False


def _tile(n: OpNode, in_axes, tile_mult, ctx: _Ctx):
    out = [f.copy() for f in in_axes]
    for i, m in enumerate(tile_mult):
        if i >= len(out) or str(m).strip() == "1":
            continue
        fac = _resolve_token(str(m), ctx)
        if fac is None or fac is NEG1:
            # 倍数解不出 → **元素数不可知**(此前"保持该轴不变"会**少算**该倍数)。
            _note(ctx, n, "tile_mult_unresolved", f"轴 {i} 倍数 {m!r}")
            return None
        out[i] = mul(out[i], fac)
    return out, False


def _expand_dims(in_axes, axis: int, numel_only: bool):
    """`unsqueeze`:插一条长度 1 的轴 → **元素数不变**。numel_only 下直接透传(numel 不变)。"""
    if numel_only:
        return [f.copy() for f in in_axes], True
    out = [f.copy() for f in in_axes]
    ax = axis if axis >= 0 else len(out) + axis + 1
    ax = max(0, min(ax, len(out)))
    out.insert(ax, Factors(coeff=1))
    return out, False


def _reshape(n: OpNode, in_axes, reshape_dims, ctx: _Ctx):
    """`reshape`:目标各维由源表达式给出;单个 `-1` 靠总积消元 —— 故 **numel_only 的输入也能用**
    (总积已知),且 reshape 在成功时**恢复轴结构**。

    **目标维解不出时退 `numel_only`**(而不是整个放弃):reshape **恒不改变元素数**——这是算子
    定义,不是估计。故即便一个目标维解不出(如 `compressor.py:203`
    `reshape(kv, (n_compressed, ratio, b, -1))` 里的 construct 局部标量 `n_compressed`/`ratio`
    **未被 walker 导出**,见 docs §4 的 G5),**元素数仍然精确等于输入的元素数**。
    退化档仍记一条账(轴结构丢了 → 下游要轴的算子照样拒绝)。
    """
    target = []
    lost = None
    for tok in reshape_dims:
        r = _resolve_token(str(tok), ctx)
        if r is None:
            lost = tok
            break
        target.append(r)
    if lost is None:
        out = resolve_reshape(in_axes, target)
        if out is not None:
            return out, False
        lost = f"`-1` 消元不干净（目标 {list(reshape_dims)}）"
    _note(ctx, n, "reshape_unresolved",
          f"目标维 {lost!r} → 退 numel_only（reshape 恒不改元素数,故元素数仍精确）")
    return [product_of(in_axes)], True


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def infer_shapes(dag: OpDAG, input_shapes: dict, dims_ctx: dict | None = None,
                 *, report: list | None = None) -> OpDAG:
    """从 input_shapes 种子出发,逐节点符号传播 shape,回填 ins/out 的 shape 段,返回同一 dag。

    `report`(可选,2026-07-25):给一个 list → 每个**解不出**的节点追加一条
    `{node, op, src, name, reason, detail, prim}`(带 `file:line`)。不给 → 行为逐字不变
    (含 MatMul 缺 `out_dim` 的 fail-loud)。纪律:**解不出的东西必须显式可见**,
    绝不用一个"看起来合理"的数替代(任务书第 2 点)。
    """
    ctx = _Ctx(
        env=dict(input_shapes or {}),
        dims_ctx=dict(dims_ctx if dims_ctx is not None else getattr(dag, "dims_ctx", {}) or {}),
        scalar_binds=list(getattr(dag, "scalar_binds", []) or []),
        report=report,
    )

    # 入边:consumer id -> [producer id...](按 edge 顺序,供合成别名桥接)
    incoming: dict[int, list[int]] = {}
    for e in dag.edges:
        if len(e) == 2:
            incoming.setdefault(e[1], []).append(e[0])

    node_out_shape: dict[int, str] = {}   # 节点 id -> 已解出 out shape 串
    node_out_name: dict[int, str] = {}    # 节点 id -> out 基名
    produced_names: set = set()           # 已被**本图某节点**产出过的基名(区别于调用方种子)

    for n in dag.nodes:
        node_out_name[n.id] = _base(n.out) if n.out else ""
        # G3/G5:切到本节点所属的内联帧(根帧 = "")+ 它那个类的 dims_ctx。
        ctx.frame = n.attrs.get("frame", "") or ""
        ctx.node_dims = n.attrs.get("dims_ctx") or None

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

        # 3) 回填输入 ref 的 shape 段(供 derive_saves 读到真 shape)+ dtype 订正
        for i, sh in enumerate(shapes):
            if sh is not None:
                n.ins[i] = _with_shape(n.ins[i], sh)
            dt = ctx.dtype_fix.get(_base(n.ins[i]))
            if dt and n.ins[i].count(":") == 2:
                p = n.ins[i].split(":")
                p[2] = dt
                n.ins[i] = ":".join(p)

        bare = [strip_numel_only(sh) if sh is not None else (None, False) for sh in shapes]
        in_numel_only = [b[1] for b in bare]
        in_axes_list = [parse_shape(b[0]) if b[0] is not None else None for b in bare]

        # 4) View "shape" 原语:即时登记标量轴名(权威,覆盖),不产张量输出
        if n.op == "View" and n.attrs.get("view") == "shape":
            # numel_only 的输入没有可信轴结构 → 逐轴登记一律 None(同 `_fill_scalar_bind`)。
            src_axes = (None if (in_numel_only and in_numel_only[0])
                        else (in_axes_list[0] if in_axes_list else None))
            for i, nm in enumerate(n.attrs.get("shape_unpack", [])):
                ctx.scalar_map[(ctx.frame, nm)] = (
                    src_axes[i] if (src_axes and i < len(src_axes)) else None)
            node_out_name[n.id] = _base(n.out) if n.out else ""
            continue

        # 5) View "split":每个目标各得其 size 轴,全部写回 env;node.out = 首目标
        if n.op == "View" and n.attrs.get("view") == "split":
            if in_numel_only and in_numel_only[0]:
                _note(ctx, n, "needs_axis_structure", "split 要按 split_dim 换轴")
                continue
            out_shape = _apply_split(n, in_axes_list, ctx)
            if out_shape is not None:
                node_out_shape[n.id] = out_shape
                if n.out:
                    n.out = _with_shape(n.out, out_shape)
            continue

        # 5b) **dtype 订正**:比较 / 逻辑 / isfinite 产 **bool**(1 字节),不是继承来的
        #     compute dtype。walker 的 `_emit` 按 ins 推产出 dtype → 这批节点被记成 bf16 = 2×。
        #     真源两个大 mask:`csa.py:779` `future = cm >= positions // ratio`(O(S·S/r))、
        #     `csa.py:810` `valid = topk_… < unsqueeze(…)`(O(S·topk))。它们会被
        #     `mint.where` 的 `PIN{"inputs":[0]}` 当 cond **保留**(bprop_rules.py:55)→ dtype 要对。
        if n.op == "Compare" and n.out and n.out.count(":") == 2:
            parts = n.out.split(":")
            parts[2] = "bool"
            n.out = ":".join(parts)
            ctx.dtype_fix[_base(n.out)] = "bool"
            for k, o in enumerate(n.attrs.get("outs") or ()):
                if o.count(":") == 2:
                    p = o.split(":")
                    p[2] = "bool"
                    n.attrs["outs"][k] = ":".join(p)

        # 6) 其余 op:算出 out 轴列表(+ 是否只知元素数)
        got = _dispatch(n, in_axes_list, in_numel_only, ctx)
        if got is None and n.out and _base(n.out) in produced_names:
            # **失效 env 里的同名旧 shape**。源里大量 `kv = self.reshape(kv, ...)` 这类**同名重绑**
            # (compressor.py:203/204、csa.py 多处):若本节点解不出而 env 仍留着**重绑前**那个
            # 张量的 shape,下游会拿着一个已经不成立的形状继续算 —— 实测这正是把
            # `Compressor` 的 concat 产物算成 8·10⁶ MiB 的第二个原因(第一个见 `sym_shape._sum_term`)。
            # 名字现在指向另一个张量 → 旧 shape 必须作废(宁 `?` 勿错)。
            #
            # ⚠ 只对**本图里先前已被某节点产出过**的名字生效(`produced_names`)。调用方给的
            # **种子**(`input_shapes`,如 MoE 的 `w1`/`w2`)是外部权威事实,不能被一个"产出它但
            # 自己解不出"的节点抹掉(实测:`ffn.py:146` 的 `w1 = cast(self.weight1)` 因权重不进
            # `ins` 而 ins 为空 → 若一并作废种子,`GroupedMatMul` 的权重末轴就丢了)。
            ctx.env.pop(_base(n.out), None)
        if got is not None:
            out_axes, numel_only = got
            out_shape = render_shape(out_axes)
            if numel_only:
                out_shape = mark_numel_only(out_shape)
            node_out_shape[n.id] = out_shape
            if n.out:
                n.out = _with_shape(n.out, out_shape)
                ctx.env[_base(n.out)] = out_shape
            # 多输出算子(chunk/split/topk/…):`outs` 里其余项同形(chunk 各份等长已在
            # `_chunk` 里证过;`max` 的 values/indices 同形)。逐条回填,否则下游拿 `?`。
            for k, o in enumerate(n.attrs.get("outs") or ()):
                if o.count(":") != 2:
                    continue
                n.attrs["outs"][k] = _with_shape(o, out_shape)
                ctx.env[_base(o)] = out_shape
                produced_names.add(_base(o))
        if n.out:
            produced_names.add(_base(n.out))

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
            _note(ctx, n, "split_size_unresolved",
                  f"目标 {tgt!r} 的 size {sizes[i] if i < len(sizes) else '?'!r}")
            continue
        new_axes = [f.copy() for f in in_axes]
        new_axes[ax] = sz
        shp = render_shape(new_axes)
        ctx.env[tgt] = shp
        if i == 0:
            first_shape = shp
    return first_shape


#: **元素数不变**的视图子类型(轴序可能变 → `_passthrough` 会标 numel_only)。
_NUMEL_PRESERVING_VIEWS = ("permute", "transpose", "roll", "squeeze", "contiguous", "view")


def _dispatch(n: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    op = n.op
    if op == "MatMul":
        return _matmul(n, in_axes_list, in_numel_only, ctx)
    if op == "BMM":
        return _bmm(n, in_axes_list, in_numel_only, ctx)
    if op == "GroupedMatMul":
        return _grouped_matmul(n, in_axes_list, in_numel_only, ctx)
    if op == "FlashAttention":
        return _flash_attention(n, in_axes_list, in_numel_only, ctx)
    if op == "Activation":
        return _activation(n, in_axes_list, in_numel_only, ctx)
    if op == "Elementwise":
        return _elementwise(n, in_axes_list, in_numel_only, ctx)
    if op == "Where":
        # `mint.where(cond, a, b)`:广播(算子定义)。任一输入未解出 → 记账,不拿 cond 顶。
        r = _broadcast(in_axes_list, in_numel_only, ctx, n)
        if r is None:
            _note(ctx, n, "no_input_shape", "where 的三个操作数须全部解出方可定广播形")
        return r
    if op in ("Norm", "Cast", "Softmax", "Dropout", "Gather", "Compare", "Detach",
              "Identity", "Scatter"):
        # 全部**形状不变**:Norm/Cast/Softmax/Dropout 逐元素;`Compare` 产同形 bool;
        # `Detach` 是**同一块存储的别名**(见 consumer 的 detach 别名去重);
        # `Scatter(input, dim, index, src)` 产出与 `input` 同形。
        r = _passthrough(in_axes_list, in_numel_only)
        if r is None:
            _note(ctx, n, "no_input_shape")
        return r
    if op == "TopK":
        # `mint.topk(x, k=<expr>, dim=-1)`:末轴换成 k。**k 未被 walker 记进 attrs**
        # (`indexer.py:262` 的 `k=effective_topk` 是 construct 局部标量)→ 记账。
        _note(ctx, n, "reduce_axis_unknown",
              "topk 的 k 未记进 attrs(`indexer.py:262` `k=effective_topk`)→ 末轴长未知")
        return None
    if op == "IndexSelect":
        # `x[idx]` / `mint.gather(x, dim, idx)`:产出形状由 **index 的形状**决定,
        # 而 index 的形状本身常来自未解出的 topk → 无从推。记账。
        _note(ctx, n, "constant_shape_unknown",
              "gather/advanced-index 的产出形由 index 形状决定,attrs 未记轴")
        return None
    if op == "Constant":
        _note(ctx, n, "constant_shape_unknown", f"ctor={n.attrs.get('prim') or n.attrs.get('ctor')}")
        return None
    if op in ("FusedFunction", "Kernel"):
        # 融合算子:产出形状是 kernel 内部契约,attrs 只带 saved 名单(源真值)→ 产出记账。
        # **它保存的输入**照常按 ins 的已解析 shape 计字节(这才是字节大头)。
        _note(ctx, n, "constant_shape_unknown",
              f"融合算子产出形非源可读(function={n.attrs.get('function') or n.attrs.get('kernel')})")
        return None
    if op == "View":
        view = n.attrs.get("view")
        if view == "concat":
            return _concat(n, in_axes_list, in_numel_only, ctx)
        if view == "chunk":
            return _chunk(n, in_axes_list, in_numel_only, ctx)
        if view == "slice":
            return _slice(n, in_axes_list, in_numel_only, ctx)
        if view == "tile":
            a = in_axes_list[0] if in_axes_list else None
            if not a:
                _note(ctx, n, "no_input_shape")
                return None
            if in_numel_only and in_numel_only[0]:
                _note(ctx, n, "needs_axis_structure", "tile 按轴放大")
                return None
            return _tile(n, a, n.attrs.get("tile_mult", []), ctx)
        if view == "expand_dims":
            a = in_axes_list[0] if in_axes_list else None
            ax = n.attrs.get("expand_axis")
            if not a or ax is None:
                _note(ctx, n, "no_input_shape")
                return None
            return _expand_dims(a, int(ax), bool(in_numel_only and in_numel_only[0]))
        if view == "reshape":
            a = in_axes_list[0] if in_axes_list else None
            if not a:
                _note(ctx, n, "no_input_shape")
                return None
            return _reshape(n, a, n.attrs.get("reshape_dims", []), ctx)
        if view in _NUMEL_PRESERVING_VIEWS or view is None:
            # permute/transpose:轴的多重集与元素数不变;轴**序**未记 → `_passthrough` 标
            # numel_only,防下游按假轴序改形。
            r = _passthrough(in_axes_list, in_numel_only)
            if r is None:
                _note(ctx, n, "no_input_shape")
                return None
            axes, nm = r
            return axes, (nm or view in ("permute", "transpose"))
        _note(ctx, n, "constant_shape_unknown", f"未列举的视图子类型 view={view!r}")
        return None
    # 未知 op:保守透传首输入(不杜撰形状,但记一条账让它可见)
    r = _passthrough(in_axes_list, in_numel_only)
    if r is None:
        _note(ctx, n, "no_input_shape", f"未列举的 op 类型 {op!r}")
    return r
