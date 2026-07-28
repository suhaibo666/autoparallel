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
    mul, add, sub, floordiv, resolve_reshape,
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
    # construct 局部标量的 host 整数值(`dag.const_scalars`)——`_scalar` 的**最后一档**。
    const_scalars: dict = field(default_factory=dict)
    # construct 局部标量的赋值表达式原文(`dag.scalar_exprs`)——`_scalar` 的**表达式档**,
    # 排在符号档之后、数值档之前(见 `_scalar` 的论证)。`_resolving` 是递归护栏。
    scalar_exprs: dict = field(default_factory=dict)
    _resolving: set = field(default_factory=set)
    # ── 内联帧作用域(G3/G5,2026-07-25)────────────────────────────────────────────────
    # `frame` = 当前正在推断的节点所属的内联帧(`OpNode.attrs["frame"]`,根帧 = "");
    # `node_dims` = 该节点所属类的 `dims_ctx`(`OpNode.attrs["dims_ctx"]`),优先于全局那份。
    # 为什么必须按帧:同一个类的两个构造点会让 `self.head_dim` 解出不同符号
    # (`v_head_dim` @ csa.py:604 vs `index_head_dim` @ indexer.py:128),局部标量名
    # (`seqlen`)也会在父子帧里指不同张量。扁平一张表只能 fail-loud 或静默取一个。
    frame: str = ""
    node_dims: dict | None = None
    # 权重形状表(`(源文件基名, 属性名) | 属性名 -> 符号 shape 串`),见 `_param_axes`。
    param_shapes: dict = field(default_factory=dict)
    # 逐节点"解不出"台账(调用方传 `report=[]` 才收):每条 {node,op,src,name,reason}。
    # 纪律(任务书第 2 点):**任何解不出的东西都必须显式带 `file:line` 出现在这里**,
    # 绝不用一个"看起来合理"的数替代。
    report: list | None = None
    # **调用方声明 / 源侧 docstring 派生**的形状台账(调用方传 `declared=[]` 才收):
    # 每条 `(名, 符号 shape, 出处, 理由)`。纪律:声明永远与"推断出来的"分开记。
    declared: list | None = None
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
    "weight_shape_unresolved": "**权重派生节点**(`ins` 为空、操作数全是 `Parameter`)的权重形状不在"
                               " `param_shapes` 里 → 无从推形状(权重形状由 `init_dims` 从"
                               " `__init__` 读出,缺就是那个 `Parameter` 的声明没被求值到)",
}

# ── (a) 权重派生节点的形状通路(2026-07-28)───────────────────────────────────────
# 背景:`Parameter` 操作数按契约 W2/W4 **不进 `ins`**(进了就会被 `derive_saves` 当激活 save 计),
# 单列 `attrs["param_operands"]`。于是"操作数全是权重"的节点 `ins` 恒空 → 无从推形状。
# 实测级联根:`hyper_connection.py:408` 的 `alpha = concat(alpha_pre, alpha_post, alpha_res)`
# 与 `linear.py:132` 的权重转置 —— 前者掐断整条 mHC 主数据路。
# **权重形状源侧其实是知道的**:`init_dims.param_shapes` 从 `__init__` 的
# `Parameter(mint.empty(...))` 逐字读出。缺的只是"喂回来"这一步。
#
# 纪律:① 只在 `ins` **为空**时启用(有真激活操作数就照旧按 `ins` 推,否则混合节点会被权重形状顶掉);
#       ② 查不到形状 → 记 `weight_shape_unresolved` + 保 `?`,**绝不**顶一个数;
#       ③ 权重仍然**不进 `ins`**(契约 W2/W4 逐字不变),这里只借形状做推断。


def _param_axes(n: OpNode, param_shapes: dict):
    """权重派生节点的"伪输入轴列表"。查不到任何一个 → `None`(调用方记账保 `?`)。

    `param_shapes` 键按 `(源文件基名, 属性名)` 消歧(与 `to_resolved._InitBook.param` 同口径:
    `Linear.weight` 是 `(vocab,H)`、`TopKRouter.weight` 是 `(E,H)` —— 全局按名合并必然撞车);
    退一档接受裸属性名(合成源单测用)。
    """
    names = n.attrs.get("param_operands") or ()
    if not names:
        return None
    # **按构造点**的那份优先(`attrs["param_decl"]`,由 `_inline_subcell` 按帧挂上):
    # `Compressor.ape` 的形状 `(compress_ratio, coff·head_dim)` 在两个构造点不同
    # (`csa.py:604` v_head_dim vs `indexer.py:128` index_head_dim)—— 全局表只能取一个。
    decl = n.attrs.get("param_decl") or {}
    base = (n.src or "").rsplit("/", 1)[-1].split(":", 1)[0]
    out = []
    for pn in names:
        sym = None
        rec = decl.get(pn)
        if isinstance(rec, dict) and rec.get("axes"):
            sym = "·".join(str(a) for a in rec["axes"])
        if sym is None and param_shapes:
            sym = param_shapes.get((base, pn)) or param_shapes.get(pn)
        if not sym:
            return None
        out.append(str(sym))
    return out


# ── (b) opaque 产出的**声明式**形状(2026-07-28)──────────────────────────────────
# `Kernel` / `FusedFunction` 的产出形是 kernel 内部契约,`shape_infer` 推不出来 —— 但**源侧
# docstring 往往逐字给了**(`hyper_connection.py:396-399`)。于是开一条与 `extract_cell(
# kernel_saves=...)` **同一条纪律**的通路:调用方**声明**,声明必须带 `file:line` 出处,
# 每条逐项进 `declared` 台账(报告里可见),**永不**与"推断出来的"混为一谈。
#
# 声明值的三种形态(`declared_outs[name]["outs"]` 按**源侧元组解包次序**逐位给):
#   `None`        —— 该位不声明(照旧 unresolved,不给数);
#   `"S·B·H"`     —— 逐轴形状(docstring 逐字);
#   `"~S·B·n"`    —— 只知**元素数**(`~` 前缀):轴结构未逐字给出,但元素数由源可证
#                    (如"紧接着被 reshape 成 (s,b,n,1)"⇒ reshape 恒不改元素数 ⇒ numel = s·b·n);
#   `"=in<k>"`    —— 与第 k 个**张量输入**同形(算子语义由源侧 forward 逐字可证)。
#                    镜像源自己没解出 → 照旧 `?`(声明不是凭空造数的许可证)。

def _declared_out_shape(spec, in_axes_list, in_numel_only):
    """一条声明 → `shape 串`(可带 `~`);解不出 → None。"""
    if not spec:
        return None
    s = str(spec).strip()
    if s.startswith("=in"):
        try:
            k = int(s[3:])
        except ValueError:
            return None
        if k >= len(in_axes_list) or in_axes_list[k] is None:
            return None                       # 镜像源未解出 → 不给数
        shp = render_shape(in_axes_list[k])
        return mark_numel_only(shp) if (k < len(in_numel_only) and in_numel_only[k]) else shp
    return s


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


def _declare(ctx: _Ctx, n: OpNode, sym: str, src: str, why: str) -> None:
    """记一条**源侧文档派生**的形状(不是逐节点推断出来的)。`declared is None` 时静默。"""
    if ctx.declared is None:
        return
    ctx.declared.append((_base(n.out) if n.out else "", sym, src, why))


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
    `reshape` 表达式里不会引用父的局部名,故这一档只是既有单帧行为的自然延续)。

    **最后一档**(2026-07-28):`dag.const_scalars` —— construct 里 host 值已知的整数局部标量
    (`pos_dim = self.config.qk_pos_emb_head_dim` @ `deepseek_v4_hybrid_attention.py:203`)。
    排在符号档**之后**是有意的:`x.shape` 解包出的轴名要优先拿到**符号**(`S`/`B`),
    符号形状对报表与分片都更有信息量;`const_scalars` 只接住那些符号档根本没有的名字。
    """
    for fr in _frame_chain(ctx.frame):
        if (fr, name) in ctx.scalar_map:
            return ctx.scalar_map[(fr, name)]
        for sb in ctx.scalar_binds:
            if sb.get("frame", "") == fr and name in sb.get("names", []):
                _fill_scalar_bind(ctx, sb, fr)
                return ctx.scalar_map.get((fr, name))
    # **表达式档**:局部标量的赋值原文,递归解 —— 优先于数值档,因为它能保住**符号**
    # (`n_compressed` → `cutoff // ratio` → `S//4`),而数值档会把符号压成数字、让
    # reshape 的 `-1` 消元约不干净(实测 `compressor.py:196-203` 整条压缩链因此断掉)。
    expr = ctx.scalar_exprs.get(name)
    if expr and name not in ctx._resolving:
        ctx._resolving.add(name)
        try:
            f = _resolve_token(str(expr), ctx)
        finally:
            ctx._resolving.discard(name)
        if f is not None and f is not NEG1:
            ctx.scalar_map[(ctx.frame, name)] = f      # 缓存(同帧后续直接命中)
            return f
    v = ctx.const_scalars.get(name)
    if isinstance(v, int) and not isinstance(v, bool) and v > 0:
        return Factors(coeff=v)
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
        if isinstance(node.op, ast.Sub):
            # `self.index_head_dim - self.qk_pos_emb_head_dim`(`indexer.py:179-180` 的
            # split size)、`self.config.v_head_dim - pos_dim`(`deepseek_v4:204`)。
            b = _eval_expr(node.right, ctx)
            return sub(a, b) if (b is not None and b is not NEG1) else None
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


def _reduce(n: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    """归约算子(`sum`/`mean`/`max`/`min`)。**轴信息 walker 已经记了**(G4 的 `_axis_capture`),
    此前 `shape_infer` 一律记 `reduce_axis_unknown` 不用它 —— 实测这是抽取图第二大的级联根
    (`deepseek_v4_hybrid_attention.py:245` 的 Q-head RMS 统计 ×8、`loss.py:344/346`)。

    三档,全部是**算子定义**、不是猜:
      * `attrs["reduce_all"]`(源侧**没传** dim/axis)→ 全轴归约 ⇒ **标量**(1 个元素);
      * `attrs["reduce_dim"]`(int 或 int 列表)+ `attrs["keepdim"]` → 被约的轴压成 1(keepdim)
        或整条去掉;
      * 两者都没有(walker 抠不出该轴)→ 照旧 `reduce_axis_unknown`,保 `?`(绝不 passthrough
        顶替 —— 那会**多算**被约掉的轴倍)。

    `mint.cumsum` 虽在 `_REDUCE_LIN` 里,但它**不归约**(前缀和,输出与输入同形)→ 透传。
    """
    prim = str(n.attrs.get("prim") or "")
    if prim.rsplit(".", 1)[-1] == "cumsum":
        r = _passthrough(in_axes_list, in_numel_only)
        if r is None:
            _note(ctx, n, "no_input_shape")
        return r
    a = in_axes_list[0] if in_axes_list else None
    if not a:
        _note(ctx, n, "no_input_shape")
        return None
    if n.attrs.get("reduce_all"):
        return [Factors(coeff=1)], False
    dims = n.attrs.get("reduce_dim")
    if dims is None:
        _note(ctx, n, "reduce_axis_unknown")
        return None
    if in_numel_only and in_numel_only[0]:
        _note(ctx, n, "needs_axis_structure", "归约要按轴去掉 / 压成 1")
        return None
    rank = len(a)
    victims: set = set()
    for d in (dims if isinstance(dims, (list, tuple)) else [dims]):
        k = int(d)
        k = k if k >= 0 else rank + k
        if not (0 <= k < rank):
            _note(ctx, n, "reduce_axis_unknown", f"轴 {d!r} 越界（rank={rank}）")
            return None
        victims.add(k)
    keep = bool(n.attrs.get("keepdim"))
    out = []
    for i, f in enumerate(a):
        if i in victims:
            if keep:
                out.append(Factors(coeff=1))
            continue
        out.append(f.copy())
    return (out or [Factors(coeff=1)]), False


def _elementwise(n: OpNode, in_axes_list, in_numel_only, ctx: _Ctx):
    """逐元素算子。**归约**(sum/mean/max 带 `reduce`)另走 `_reduce`。"""
    if n.attrs.get("reduce"):
        return _reduce(n, in_axes_list, in_numel_only, ctx)
    b = _broadcast(in_axes_list, in_numel_only, ctx, n)
    if b is not None:
        return b
    return _passthrough(in_axes_list, in_numel_only)     # 退回首输入(既有行为)


def _weight_operand_axes(n: OpNode, ctx: _Ctx):
    """该节点**唯一**权重操作数的当前轴列表(exact,非 numel_only);判不出 → None。

    两档:① `ctx.env[名]` —— 该权重在本图里被某个权重派生节点改过形(转置/cast),env 里是
    **当前**那份;② `ctx.param_shapes[(文件, 名)]` —— `__init__` 里的声明形状。
    多于一个权重操作数 → 判不出"哪个是矩阵",返回 None(不猜)。
    """
    names = n.attrs.get("param_operands") or ()
    if len(names) != 1:
        return None
    shp = _lookup(ctx.env, names[0])
    if shp is None:
        got = _param_axes(n, ctx.param_shapes or {})
        shp = got[0] if got else None
    if not shp:
        return None
    bare, numel_only = strip_numel_only(shp)
    if numel_only:
        return None
    axes = parse_shape(bare)
    return axes or None


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
        # **权重操作数的末轴**(算子定义:`[..., k] @ [k, n] → [..., n]`,同 `_bmm`/`_grouped_matmul`)。
        # 真源 `pynative/layers/linear.py:132-135`:`weight = transpose(weight, 1, 0)` 之后
        # `output = matmul(input_, weight)`。`weight` 是 `Parameter` → 走 `param_operands`、
        # 不进 `ins`(契约 W2/W4),而 `Linear` 作为**顶层** Cell 抽取时也没有
        # `build_module(..., output_size=…)` 那个调用点 ⇒ `attrs["out_dim"]` 缺席。
        # 取法:先查 env(转置后的那份**当前**形状,由本图节点产出),再退 `param_shapes`
        # (未经变换的声明形状)。任一档解不出 → 照旧记账保 `?`。
        w = _weight_operand_axes(node, ctx)
        if w is not None and len(w) >= 2:
            out = [f.copy() for f in a]
            out[-1] = w[-1].copy()
            return out, False
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


#: `Constant` 里**元素数可由源证得**的两类构造(2026-07-28)。其余照旧 `constant_shape_unknown`。
#:
#: ① **RoPE 频率表**(`attrs["rope_freqs"]`,由调用方 `injected_binds` 注入):
#:    `RotaryEmbedding.construct(max_seq_len)` 逐字([SRC] `rotary_pos_embedding.py:109-148`)
#:      `inv_freq = 1/(base ** (np.arange(0, dim, 2)/dim))`  → `dim//2` 个元素(`:61`)
#:      `freqs = outer(seq, inv_freq)`                       → `[max_seq_len, dim//2]`(`:132`)
#:      `emb = cat((freqs, freqs), -1)`                      → `[max_seq_len, dim]`(`:141`)
#:      `out = reshape(emb, (-1, bs, 1, emb.shape[1]))`,`bs = 1`(`:123`,非 position_ids 支)
#:                                                           → `[max_seq_len, 1, 1, dim]`(`:144`)
#:    而 `dim = config.qk_pos_emb_head_dim`([SRC] `deepseek_v4_hybrid_attention.py:175-177`
#:    `RotaryEmbedding(config.qk_pos_emb_head_dim, rotary_percent=…)`;`rotary_percent` 缺省
#:    1.0 ⇒ `:54-55` 的 `dim = int(dim*rotary_percent)` 不改值)。三个调用点
#:    (`deepseek_v4:257` / `indexer.py:174` / `compressor.py:231`)共用**同一个实例**
#:    (`:88` `rotary_pos_emb=self.rotary_pos_emb` 层层下传),故末轴同为 `qk_pos_emb_head_dim`;
#:    首轴 = 该调用点的 `max_seq_len` 实参(`attrs["const_shape_src"]`,逐点不同)。
#:
#: ② `ops.tuple_to_array((1e-8,))`(`loss.py:347`):实参是**值元组**,产出是 1 维张量,
#:    元素数 = 元组长度(`attrs["const_shape"]` 记的就是那些值的字面量)。
_ROPE_LAST_AXIS = "qk_pos_emb_head_dim"


def _constant(n: OpNode, ctx: _Ctx):
    if n.attrs.get("rope_freqs"):
        src = n.attrs.get("const_shape_src")
        seq = _resolve_token(str(src), ctx) if src else None
        if seq is None or seq is NEG1:
            return None
        axes = [seq, Factors(coeff=1), Factors(coeff=1), parse_axis(_ROPE_LAST_AXIS)]
        _declare(ctx, n, render_shape(axes), "rotary_pos_embedding.py:109-148",
                 "RotaryEmbedding.construct 逐字：freqs=[max_seq_len,dim//2] → "
                 "cat 成 [max_seq_len,dim] → reshape(-1,1,1,dim)；"
                 "dim=config.qk_pos_emb_head_dim（deepseek_v4_hybrid_attention.py:175-177）")
        return axes, False
    prim = str(n.attrs.get("prim") or "").rsplit(".", 1)[-1]
    if prim == "tuple_to_array":
        elts = n.attrs.get("const_shape")
        if not elts:
            return None
        return [Factors(coeff=len(elts))], False
    # ③ `mint.arange(stop)`:产出是一维 `(stop,)`(算子定义)。**只**在恰好 1 个位置实参时
    #    才用 —— `arange(start, stop[, step])` 的第 0 位是 start,当成长度会算错
    #    (`attrs["const_argc"]` 就是为这个判据记的)。真源 `csa.py:452-453`
    #    `mint.arange(seqlen)` / `mint.arange(window_size)`(滑窗索引矩阵的两条轴)。
    if prim == "arange" and int(n.attrs.get("const_argc") or 0) == 1:
        elts = n.attrs.get("const_shape") or ()
        stop = _resolve_token(str(elts[0]), ctx) if elts else None
        if stop is None or stop is NEG1:
            return None
        return [stop], False
    # ④ `mint.zeros/ones/full/empty(<shape 元组>, …)`:目标各维由源表达式给出,全部解出才用。
    if prim in ("zeros", "ones", "full", "empty") and n.attrs.get("const_shape"):
        axes = []
        for tok in n.attrs["const_shape"]:
            f = _resolve_token(str(tok), ctx)
            if f is None or f is NEG1:
                return None
            axes.append(f)
        return (axes or None) and (axes, False)
    # ⑤ `mint.full(<x>.shape, v, …)` / `zeros(<x>.shape)`:**与 `<x>` 同形**(算子定义)。
    #    真源 `csa.py:442/457` `mint.full(matrix.shape, -1, dtype=int32)`。
    src = str(n.attrs.get("const_shape_src") or "")
    if prim in ("zeros", "ones", "full", "empty") and src.endswith(".shape"):
        shp = _lookup(ctx.env, src[: -len(".shape")])
        if not shp:
            return None
        bare, numel_only = strip_numel_only(shp)
        axes = parse_shape(bare)
        return (axes, numel_only) if axes else None
    return None


def _gather(n: OpNode, in_axes_list, in_numel_only):
    """`mint.gather(input, dim, index)` → 产出与 **index**(张量操作数位序 1)同形。

    位序来自 `attrs["ins_slots"]`(walker 记的「每个存活 `ins` 项的张量操作数位序」——
    `dim` 是 int 不占位、`Parameter` 占位但不进 `ins`)。没有 `ins_slots` 时 `ins` 位序即
    张量位序 → index 是 `ins[1]`。认不出就返回 None(调用方记账保 `?`)。
    """
    if n.attrs.get("prim") != "mint.gather":
        return None
    slots = n.attrs.get("ins_slots")
    idx = None
    if slots:
        for i, s in enumerate(slots):
            if s == 1:
                idx = i
                break
    elif len(in_axes_list) >= 2:
        idx = 1
    if idx is None or idx >= len(in_axes_list) or in_axes_list[idx] is None:
        return None
    return ([f.copy() for f in in_axes_list[idx]],
            bool(idx < len(in_numel_only) and in_numel_only[idx]))


_INT_DTYPES = ("int8", "int16", "int32", "int64", "uint8", "uint32", "uint64")


def _advanced_index(n: OpNode, in_axes_list, in_numel_only):
    """`x[idx]`(`attrs["advanced_index"]`,**整数索引数组**)→ `idx.shape ++ x.shape[1:]`。

    这是算子定义(NumPy/torch 的 advanced indexing:轴 0 被 index 数组替换成 index 的整个形状)。
    真源 `csa.py:485` `kv_flat[flat_indices]`:`kv_flat` 是 `[b·sk, d]`、`flat_indices` 是 1 维
    → 产出 `[len(idx), d]`,紧接着 `:485` reshape 成 `(b, sq, topk, d)`(与本规则一致)。

    守卫(缺一条就返回 None,宁 `?` 勿错):恰两个张量操作数;index **dtype 是整数**
    (布尔掩码的语义完全不同 —— 那是 `x[mask]`,产出长度取决于**值**而非形状);
    两侧都有可信轴结构(不是 `~`);被索引侧至少 1 轴。
    """
    if not n.attrs.get("advanced_index"):
        return None
    if len(n.ins) != 2 or len(in_axes_list) != 2:
        return None
    if in_axes_list[0] is None or in_axes_list[1] is None or any(in_numel_only[:2]):
        return None
    dt = n.ins[1].split(":")[2] if n.ins[1].count(":") == 2 else ""
    if dt not in _INT_DTYPES:
        return None
    x, idx = in_axes_list[0], in_axes_list[1]
    if len(x) < 1:
        return None
    return [f.copy() for f in idx] + [f.copy() for f in x[1:]], False


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
        # `allow_expr=True`:`-1` 消元约不干净时形成整除原子而不是整条退 numel_only
        # (见 `sym_shape.divide_expr` —— 压缩链 `S` vs `S//4` 就卡在这一步)。
        out = resolve_reshape(in_axes, target, allow_expr=True)
        if out is not None:
            return out, False
        lost = f"`-1` 消元不干净（目标 {list(reshape_dims)}）"
    _note(ctx, n, "reshape_unresolved",
          f"目标维 {lost!r} → 退 numel_only（reshape 恒不改元素数,故元素数仍精确）")
    return [product_of(in_axes)], True


# ── 主流程 ─────────────────────────────────────────────────────────────────────
def infer_shapes(dag: OpDAG, input_shapes: dict, dims_ctx: dict | None = None,
                 *, report: list | None = None,
                 param_shapes: dict | None = None,
                 declared_outs: dict | None = None,
                 declared: list | None = None,
                 bridge_by_edge: bool = False) -> OpDAG:
    """从 input_shapes 种子出发,逐节点符号传播 shape,回填 ins/out 的 shape 段,返回同一 dag。

    `report`(可选,2026-07-25):给一个 list → 每个**解不出**的节点追加一条
    `{node, op, src, name, reason, detail, prim}`(带 `file:line`)。不给 → 行为逐字不变
    (含 MatMul 缺 `out_dim` 的 fail-loud)。纪律:**解不出的东西必须显式可见**,
    绝不用一个"看起来合理"的数替代(任务书第 2 点)。

    `param_shapes`(可选,2026-07-28):`{(源文件基名, 属性名) | 属性名: 符号 shape 串}` ——
    **权重派生节点**(`ins` 为空、操作数全是 `Parameter`)的形状通路,见 `_param_axes`。

    `declared_outs` / `declared`(可选,2026-07-28):opaque 产出(`Kernel` / `FusedFunction`)的
    **调用方声明**表与其台账,见 `_declared_out_shape`。声明必须带出处;逐条进 `declared`,
    **永不**与推断结果混为一谈。

    `bridge_by_edge`(可选,2026-07-28,**缺省关**):把"按名解不出的输入 ← 入边 producer"
    的桥接从合成别名放宽到全部入边(见主循环第 2 步的论证)。**缺省关**是有意的:
    `timesim/producer.py` 的通信注入按「有几个输入的 shape 已解出」判 S 分歧
    (`producer.py:227` `real = [info for info in in_infos if info.sym and info.sym != "?"]`),
    多解出一个输入就会多注入一条 layout-redistribution AG —— 那是另一个子系统的口径,
    不该被本轮的覆盖度改造顺带改掉。故只有 `to_resolved`(显存记账)打开它。
    """
    ctx = _Ctx(
        env=dict(input_shapes or {}),
        dims_ctx=dict(dims_ctx if dims_ctx is not None else getattr(dag, "dims_ctx", {}) or {}),
        scalar_binds=list(getattr(dag, "scalar_binds", []) or []),
        const_scalars=dict(getattr(dag, "const_scalars", {}) or {}),
        scalar_exprs=dict(getattr(dag, "scalar_exprs", {}) or {}),
        param_shapes=dict(param_shapes or {}),
        report=report,
        declared=declared,
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

        # 2) **数据流边桥接**:未按名解出的输入 ← 本节点的入边 producer,按边序(= 实参序)补。
        #
        # 缺省只对合成别名(`__chain__` / `__ret__`)生效(既有调用方逐字不变)。
        # `bridge_by_edge=True` 时放宽到"**产出名没被本节点按名消费**的任何入边 producer"
        # (2026-07-28),因为**内联子 Cell 的返回值丢的是名字、不是数据流边**:
        #   `aggregated_attn, h_res, h_post = self.attn_hc(hidden_states)`
        #       → 子里叫 `h_in`(`hyper_connection.py:421`),父里叫 `aggregated_attn`
        #         (`transformer_layer.py:311` 的 `ins`),**边 4→7 在**,只是名字对不上;
        #   `log_softmax = self.log_softmax(logits)`(`loss.py:335`)同理(边 1→2 在);
        #   `words_embeddings`(`language_model_embedding.py:134`)同理。
        # 判据是保守的:只补**按名没解出**的位,且只用**产出名不在本节点 ins 名单里**的 producer
        # (名字对得上的那些本来就该按名解;它们没解出说明 producer 自己也没解出)。
        # 位序按边序 = `_emit` 里 `pending_prods` 的追加序 = 实参序,故一一对应是源侧次序,不是猜。
        in_names = {_base(r) for r in n.ins}
        bridge_prods = [p for p in incoming.get(n.id, [])
                        if p in node_out_shape
                        and (_is_synth(node_out_name.get(p, ""))
                             or (bridge_by_edge and node_out_name.get(p, "") not in in_names))]
        li = 0
        for i in range(len(shapes)):
            if shapes[i] is None and li < len(bridge_prods):
                shapes[i] = node_out_shape[bridge_prods[li]]
                li += 1
        # **过期种子**:某些内联把子 Cell 的形参名**永久**映射成调用方那个 ref
        # (`VocabEmbedding.construct(input_)` 内联进 `LanguageModelEmbedding` 后,
        # `input_ = self.reshape(input_, (-1,1))`、`self.tile(...)` 这两步的 `ins`
        # 仍写作调用方的 `input_ids`)。于是 `_lookup` 拿到的是**入口种子**那个形状,
        # 而真正的上游是那条边 —— 实测 `vocab_embedding.py:85` 的 gather 因此把产出
        # 算成 `B·S`(应为 `B·S·H`,源 docstring `:76` `output: (B, S, H)`),整段 embedding
        # 差 `H` 倍。判据保守到只在**全部 ins 都是"从未被本图节点产出过"的名字**、
        # 且入边 producer 数与 ins 数**恰好相等**时才改用边(此时配对唯一)。
        if (bridge_by_edge and n.ins and all(s is not None for s in shapes)
                and all(_base(r) not in produced_names for r in n.ins)):
            prods = [p for p in incoming.get(n.id, []) if p in node_out_shape]
            if len(prods) == len(n.ins):
                shapes = [node_out_shape[p] for p in prods]

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

        # 3b) **权重派生节点**(`ins` 为空、操作数全是 `Parameter`)的伪输入轴(2026-07-28)。
        #     只在 `ins` 为空时启用 —— 有真激活操作数的节点照旧按 `ins` 推(否则
        #     `self.embedding(weight, 0, input_)` 这类混合节点会被权重形状顶掉)。
        #     权重**仍然不进 `ins`**(契约 W2/W4 逐字不变),这里只借形状做推断。
        weight_derived = False
        if not n.ins and n.attrs.get("param_operands") and n.op != "Constant":
            psyms = _param_axes(n, param_shapes or {})
            if psyms is None:
                _note(ctx, n, "weight_shape_unresolved",
                      f"权重 {list(n.attrs.get('param_operands') or ())}")
            else:
                weight_derived = True
                pb = [strip_numel_only(s) for s in psyms]
                in_numel_only = [b[1] for b in pb]
                in_axes_list = [parse_shape(b[0]) for b in pb]

        # 3c) **opaque 产出的调用方声明**(2026-07-28):`Kernel` / `FusedFunction` 的产出形
        #     源侧 docstring 逐字给了 → 按声明回填,逐条进 `declared` 台账(带出处)。
        if n.op in ("FusedFunction", "Kernel") and declared_outs:
            key = n.attrs.get("kernel") or n.attrs.get("function")
            rec = (declared_outs or {}).get(key)
            if rec:
                specs = list(rec.get("outs") or ())
                outs = list(n.attrs.get("outs") or ())
                got_any = False
                for k, spec in enumerate(specs):
                    shp = _declared_out_shape(spec, in_axes_list, in_numel_only)
                    if shp is None:
                        continue
                    nm = ""
                    if k < len(outs) and outs[k].count(":") == 2:
                        outs[k] = _with_shape(outs[k], shp)
                        nm = _base(outs[k])
                    if k == 0 and n.out:
                        n.out = _with_shape(n.out, shp)
                        nm = nm or _base(n.out)
                        node_out_shape[n.id] = shp
                    if not nm:
                        continue
                    ctx.env[nm] = shp
                    produced_names.add(nm)
                    got_any = True
                    if declared is not None:
                        declared.append((nm, shp, rec.get("src", ""), rec.get("why", "")))
                if outs:
                    n.attrs["outs"] = outs
                if got_any:
                    if n.out:
                        produced_names.add(_base(n.out))
                    continue

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
        # `mint.gather(input, dim, index)`:产出与 **index 同形** —— 这是**算子定义**
        # (torch/mindspore 的 `gather` 语义),不是猜。真源 `vocab_embedding.py:85`
        # `self.embedding(weight, 0, input_)`:`weight` 是 `Parameter` 走 `param_operands`,
        # 存活的那个 `ins` 的张量位序就是 1 = index(`attrs["ins_slots"]`,walker 记的)。
        # 整个 embedding 段(以及那 1010 MiB 词表)此前就卡在这一条上。
        #
        # ⚠ **advanced indexing** (`kv_flat[flat_indices]` @ `csa.py:485`)不是这个语义:
        # 它的产出形 = index 形 **+ input 的尾轴** → 套 gather 规则会**少算**尾轴。
        # 故只对 `prim == "mint.gather"` 生效,其余照旧记账保 `?`(宁 `?` 勿错)。
        r = _gather(n, in_axes_list, in_numel_only)
        if r is None:
            r = _advanced_index(n, in_axes_list, in_numel_only)
        if r is not None:
            return r
        _note(ctx, n, "constant_shape_unknown",
              "gather/advanced-index 的产出形由 index 形状决定,attrs 未记轴")
        return None
    if op == "Constant":
        r = _constant(n, ctx)
        if r is not None:
            return r
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
        if view == "transpose" and n.attrs.get("swap_axes"):
            # `mint.transpose(x, d0, d1)`:**两轴互换**(算子定义)。轴是源里逐字写着的常数
            # (`linear.py:132` `transpose(weight, 1, 0)`)→ 可以精确换,不必退 numel_only。
            a = in_axes_list[0] if in_axes_list else None
            if not a:
                _note(ctx, n, "no_input_shape")
                return None
            if in_numel_only and in_numel_only[0]:
                _note(ctx, n, "needs_axis_structure", "transpose 要按轴互换")
                return None
            d0, d1 = (int(x) for x in n.attrs["swap_axes"][:2])
            rank = len(a)
            d0 = d0 if d0 >= 0 else rank + d0
            d1 = d1 if d1 >= 0 else rank + d1
            if not (0 <= d0 < rank and 0 <= d1 < rank):
                _note(ctx, n, "needs_axis_structure",
                      f"transpose 轴 {n.attrs['swap_axes']} 越界（rank={rank}）")
                return None
            out = [f.copy() for f in a]
            out[d0], out[d1] = out[d1], out[d0]
            return out, False
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
