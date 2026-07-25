# cost_eval/opdag/init_dims.py
"""PART A(T9):静态求值 Cell.__init__,抠出每个 `build_module(submodules.Y, <in>, <out>, ...)`
的 (in,out) 符号维度,并建 dims_ctx(self.<attr> → 符号 token 串)。

原理:__init__ 是**普通静态 Python**,维度以 config 符号写出(如 `self.config.hidden_size`、
`num_heads * head_dim`、GLU 的 `*= 2`)。我们按 MRO(base→derived)顺序**逐语句**求值,维护:
  * self_env / local_env —— 名字 → 求得的**维度**(sym_shape.Factors)或**具体值**(int/bool/None/str);
  * 遇 `if` 用 config_flags/param 默认剪枝(可判定才进分支;不可判定则跳过,best-effort);
  * 遇 `self.X = build_module(sub.Y, a1, a2, ...)`(≥2 位维度实参)→ 记 linear_dims[X]=(a1,a2)。

**源忠实、不杜撰**:任一维度表达式解不出 config 符号 → 该维度记 None(上层保留 `?`),绝不编造。
绝不 import/执行 mindspore/mindformers——纯 ast。
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .sym_shape import (Factors, mul, add, floordiv, render_term, parse_axis,
                        CONFIG2SYM, _split_top)

_UNDECIDED = object()
# 真·维度符号集合(dims_ctx 只保留由这些符号构成的项,避免把 compute_dtype/input_layout 等
# 被当作维度符号透传;它们从不出现在 reshape/split 表达式里)。
_KNOWN_DIM_SYMS = set(CONFIG2SYM.values())

#: `config_flags` 的**保留键**:`__init__` **形参**的维度/值注入(2026-07-25,字节解析)。
#:
#: 为什么必须由调用方显式给、不能自动推:
#:   * `Compressor(head_dim=config.v_head_dim)`(`csa.py:604`)—— 单抽 `Compressor` 时父不在场,
#:     `head_dim` 是个无缺省的位置形参;而它**同名于** `DimTable.head_dim`(GQA 的 128),真值却是
#:     `v_head_dim`(512)。靠名字猜 = 4× 错。
#:   * `CompressedSparseAttention(compress_ratio=...)`(`csa.py:556`)缺省 `0`,真机是 4/128 ——
#:     拿缺省去凑正是 walker 文档 §2.1 拒绝的那个反例(会静默算出与真机相反的 `enable_compress`)。
#:
#: 形态:`{形参名: "符号 token 串"}`(按**维度符号**处理,如 `{"head_dim": "v_head_dim"}`)
#: 或 `{形参名: int|bool}`(具体值,如 `{"compress_ratio": 4}` —— 既进维度也进条件判定)。
#: 不给 → 该形参判不出 → 相关维度记 None(上层保留 `?`),**决不**编造。
INIT_PARAM_SEEDS = "__init_param_seeds__"

#: mindspore dtype 名 → 本库规范串(与 `consumer._DTYPE_BYTES` 的键域一致)。
_MSTYPE2DTYPE = {
    "float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
    "float64": "fp64", "int32": "int32", "int64": "int64",
    "int8": "int8", "uint8": "uint8", "bool_": "bool",
}


def _all_known_dim(f: Factors) -> bool:
    for unit in f.syms:
        for term in _split_top(unit, "+"):
            if term.strip() not in _KNOWN_DIM_SYMS:
                return False
    return True


@dataclass
class InitDims:
    # self.<module_attr> -> (in_dim_str|None, out_dim_str|None)
    linear_dims: dict = field(default_factory=dict)
    # self.<attr> -> 符号 token 串(dim,供 reshape/split 表达式解析)
    dims_ctx: dict = field(default_factory=dict)
    # self.<attr> -> "param" / "scalar" / "module"(P0#5 判据,2026-07-25)。
    #   param  —— `self.attn_sink = Parameter(...)`(csa.py:589)、`self.ape = Parameter(...)`
    #             (compressor.py:117)、`self.linear_o_group_proj`/`self.q_rms_gamma`
    #             (deepseek_v4_hybrid_attention.py:139/159)→ **权重,不是激活**;
    #   scalar —— `self.softmax_scale = softmax_scale or config.v_head_dim ** -0.5`(csa.py:568)
    #             这类由 config/算术求出的 host 值 → 参与 `if`/BinOp 时按标量处理;
    #   module —— `build_module(...)` / 类实例化 → 是算子,不是值。
    # 这让 walker 判「`self.X` 是张量还是标量」有**源侧判据**,而不是靠猜。
    self_kinds: dict = field(default_factory=dict)
    # self.<param_attr> -> 符号 shape 轴串元组(2026-07-25,契约 B4 的 param census 要它)。
    #   `self.ape = Parameter(mint.empty((compress_ratio, proj_out_dim)))`(compressor.py:117)
    #     → ("4", "2·v_head_dim");
    #   `self.attn_sink = Parameter(mint.zeros(config.num_attention_heads))`(csa.py:589)
    #     → ("n_heads",)。
    # 任一轴解不出 → **整条不记**(宁缺勿编;调用方据此报 unresolved)。
    param_shapes: dict = field(default_factory=dict)
    # self.<param_attr> -> dtype 串(逐字来自源的 `dtype=` 实参;缺省用 config.params_dtype)。
    #   `csa.py:589` 明写 `dtype=mstype.float32` → "fp32"(**不是** params_dtype 的 bf16)。
    param_dtypes: dict = field(default_factory=dict)


@dataclass
class _Cell:
    dim: Factors | None = None          # 维度(可乘除加)
    val_known: bool = False             # 是否求得具体值(用于条件判定)
    val: object = None                  # 具体值(int/bool/None/str)
    kind: str = "scalar"                # "param" / "scalar" / "module"(见 InitDims.self_kinds)


def _find_class(tree, name):
    return next((n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == name), None)


def _init_of(cls):
    return next((n for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)


def _init_classes(tree, cls_name) -> list[str]:
    """MRO 里定义了 __init__ 的类,derived→base 顺序(与 extractor 一致)。"""
    order, seen, queue = [], set(), [cls_name]
    while queue:
        c = queue.pop(0)
        if c in seen:
            continue
        seen.add(c)
        cls = _find_class(tree, c)
        if cls is None:
            continue
        if _init_of(cls) is not None:
            order.append(c)
        for b in cls.bases:
            if isinstance(b, ast.Name):
                queue.append(b.id)
    return order


class _Eval:
    def __init__(self, config_flags: dict):
        self.flags = dict(config_flags or {})
        self.param_seeds: dict = dict(self.flags.get(INIT_PARAM_SEEDS) or {})
        self.self_env: dict[str, _Cell] = {}
        self.linear_dims: dict = {}
        self.param_shapes: dict = {}
        self.param_dtypes: dict = {}

    # ── 维度求值(→ Factors 或 None)──────────────────────────────────────────
    def eval_dim(self, node, local: dict) -> Factors | None:
        try:
            f = self._eval_dim(node, local)
        except Exception:
            f = None
        if f is not None:
            return f
        # 结构化路径解不出时的**最后一档**:若该表达式的**具体值**已知且是整数,就按整数系数用。
        # 判据:结构化路径(config.X / self.X / BinOp)已先行,故这一档只接住
        # `1 + int(self.overlap)`(compressor.py:89-90 `self.coff`)、`x or y`、三元这类
        # **纯 host 算术**——它们的值完全由 config_flags 决定,不是编造。
        try:
            k, v = self._eval_val(node, local)
        except Exception:
            return None
        if k and isinstance(v, int) and not isinstance(v, bool):
            return Factors(coeff=v)
        return None

    def _eval_dim(self, node, local) -> Factors | None:
        if isinstance(node, ast.Constant):
            return Factors(coeff=int(node.value)) if isinstance(node.value, int) and not isinstance(node.value, bool) else None
        if isinstance(node, ast.Name):
            cell = local.get(node.id)
            return cell.dim if cell else None
        if isinstance(node, ast.Attribute):
            cfg = self._config_attr(node)
            if cfg is not None:
                return Factors(1, {CONFIG2SYM.get(cfg, cfg): 1})
            sattr = self._self_attr(node)
            if sattr is not None:
                cell = self.self_env.get(sattr)
                return cell.dim if cell else None
            return None
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Mult):
                a, b = self._eval_dim(node.left, local), self._eval_dim(node.right, local)
                return mul(a, b) if (a and b) else None
            if isinstance(node.op, ast.Add):
                a, b = self._eval_dim(node.left, local), self._eval_dim(node.right, local)
                return add(a, b) if (a and b) else None
            if isinstance(node.op, ast.FloorDiv):
                a = self._eval_dim(node.left, local)
                if a and isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                    return floordiv(a, node.right.value)
                return None
            return None
        if isinstance(node, ast.IfExp):
            t = self._truthy(node.test, local)
            if t is _UNDECIDED:
                return None
            return self._eval_dim(node.body if t else node.orelse, local)
        return None

    # ── 具体值求值(用于条件)→ (known, value)──────────────────────────────────
    def eval_val(self, node, local: dict):
        try:
            return self._eval_val(node, local)
        except Exception:
            return (False, None)

    def _eval_val(self, node, local):
        if isinstance(node, ast.Constant):
            return (True, node.value)
        if isinstance(node, ast.Name):
            cell = local.get(node.id)
            if cell and cell.val_known:
                return (True, cell.val)
            return (False, None)
        if isinstance(node, ast.Attribute):
            cfg = self._config_attr(node)
            if cfg is not None:
                if cfg in self.flags:
                    return (True, self.flags[cfg])
                return (False, None)
            sattr = self._self_attr(node)
            if sattr is not None:
                cell = self.self_env.get(sattr)
                if cell and cell.val_known:
                    return (True, cell.val)
            return (False, None)
        if isinstance(node, ast.Compare):
            return self._eval_compare(node, local)
        if isinstance(node, ast.BoolOp):
            return self._eval_boolop(node, local)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            k, v = self._eval_val(node.operand, local)
            return (True, not v) if k else (False, None)
        if isinstance(node, ast.IfExp):                  # `a if cond else b`
            t = self._truthy(node.test, local)
            if t is _UNDECIDED:
                return (False, None)
            return self._eval_val(node.body if t else node.orelse, local)
        if isinstance(node, ast.BinOp):                  # host 算术(`1 + int(self.overlap)`)
            lk, lv = self._eval_val(node.left, local)
            rk, rv = self._eval_val(node.right, local)
            if not (lk and rk) or isinstance(lv, bool) or isinstance(rv, bool):
                return (False, None)
            if not (isinstance(lv, int) and isinstance(rv, int)):
                return (False, None)
            if isinstance(node.op, ast.Add):
                return (True, lv + rv)
            if isinstance(node.op, ast.Sub):
                return (True, lv - rv)
            if isinstance(node.op, ast.Mult):
                return (True, lv * rv)
            if isinstance(node.op, ast.FloorDiv) and rv:
                return (True, lv // rv)
            return (False, None)
        if isinstance(node, ast.Call):
            # `int(x)` / `bool(x)`:真源 `self.coff = 1 + int(self.overlap)`(compressor.py:90)。
            # 只认这两个内建、且实参值已知 —— 其它调用一律判不出(不猜)。
            f = node.func
            if (isinstance(f, ast.Name) and f.id in ("int", "bool")
                    and len(node.args) == 1 and not node.keywords):
                k, v = self._eval_val(node.args[0], local)
                if k and isinstance(v, (int, bool)):
                    return (True, int(v) if f.id == "int" else bool(v))
            return (False, None)
        return (False, None)

    def _eval_compare(self, node, local):
        if len(node.ops) != 1:
            return (False, None)
        lk, lv = self._eval_val(node.left, local)
        rk, rv = self._eval_val(node.comparators[0], local)
        if not (lk and rk):
            return (False, None)
        op = node.ops[0]
        if isinstance(op, ast.Is):
            return (True, lv is rv)
        if isinstance(op, ast.IsNot):
            return (True, lv is not rv)
        if isinstance(op, ast.Eq):
            return (True, lv == rv)
        if isinstance(op, ast.NotEq):
            return (True, lv != rv)
        return (False, None)

    def _eval_boolop(self, node, local):
        vals = [self._eval_val(v, local) for v in node.values]
        if isinstance(node.op, ast.And):
            if any(k and not v for k, v in vals):
                return (True, False)
            if all(k for k, _ in vals):
                return (True, all(v for _, v in vals))
            return (False, None)
        # Or
        if any(k and v for k, v in vals):
            return (True, True)
        if all(k for k, _ in vals):
            return (True, any(v for _, v in vals))
        return (False, None)

    def _truthy(self, node, local):
        k, v = self._eval_val(node, local)
        return bool(v) if k else _UNDECIDED

    # ── 名字识别 ────────────────────────────────────────────────────────────
    @staticmethod
    def _self_attr(node) -> str | None:
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            return node.attr
        return None

    @staticmethod
    def _config_attr(node) -> str | None:
        """`config.X`(参数名 config)或 `self.config.X` → X;否则 None。"""
        if not isinstance(node, ast.Attribute):
            return None
        v = node.value
        if isinstance(v, ast.Name) and v.id == "config":
            return node.attr
        if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                and v.value.id == "self" and v.attr == "config"):
            return node.attr
        return None

    # ── 语句执行 ────────────────────────────────────────────────────────────
    def exec_body(self, body, local: dict):
        for stmt in body:
            self.exec_stmt(stmt, local)

    def exec_stmt(self, stmt, local: dict):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            self._do_assign(stmt.targets[0], stmt.value, local)
        elif isinstance(stmt, ast.AugAssign):
            self._do_aug(stmt, local)
        elif isinstance(stmt, ast.If):
            t = self._truthy(stmt.test, local)
            if t is _UNDECIDED:
                return                      # best-effort:不可判定分支跳过(不乱设维度)
            self.exec_body(stmt.body if t else stmt.orelse, local)
        # 其它语句(raise/expr/for...)对维度捕获无关,忽略

    def _cell_for(self, value, local) -> _Cell:
        dim = self.eval_dim(value, local)
        vk, vv = self.eval_val(value, local)
        return _Cell(dim=dim, val_known=vk, val=vv)

    @staticmethod
    def _ctor_name(value):
        """`Parameter(...)` / `mint.zeros(...)` 这类调用的"构造名"(取 func 末段)。"""
        if not isinstance(value, ast.Call):
            return None
        f = value.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
        return None

    @staticmethod
    def _kwarg(call: ast.Call, name: str):
        return next((kw.value for kw in call.keywords if kw.arg == name), None)

    def _build_module_dims(self, call: ast.Call, local):
        """`build_module(sub.X, <in>, <out>, ...)` 的 (in,out) 维度表达式节点。

        两种形态**都要认**(此前只认位序 → pynative 侧全部 linear 的 out_dim 缺失):
          * **位序**:`build_module(sub.x, self.config.hidden_size, self.config.q_lora_rank, ...)`
            —— `parallel_core/training_graph/transformer/multi_latent_attention.py:634-643`;
          * **关键字**:`build_module(sub.x, input_size=..., output_size=..., ...)`
            —— `pynative/.../compressor.py:97-104`、`deepseek_v4_hybrid_attention.py:93-96`、
            `indexer.py:117-119`、`dsa_indexer.py:169-171`。
        位序优先(既有行为逐字不变),缺位再看关键字。
        """
        in_n = call.args[1] if len(call.args) >= 2 else self._kwarg(call, "input_size")
        out_n = call.args[2] if len(call.args) >= 3 else self._kwarg(call, "output_size")
        if in_n is None and out_n is None:
            return None
        return (self.eval_dim(in_n, local) if in_n is not None else None,
                self.eval_dim(out_n, local) if out_n is not None else None)

    def _param_shape_of(self, call: ast.Call, local):
        """`Parameter(mint.empty((a, b), dtype=...), name=...)` → (轴串元组, dtype 串)。

        真源两形态:`compressor.py:117` `mint.empty((compress_ratio, proj_out_dim), dtype=…)`
        (元组 shape)、`csa.py:589` `mint.zeros(config.num_attention_heads, dtype=…)`(裸标量)。
        任一轴解不出 → (None, dtype):**整条不记**(param census 宁缺勿编)。
        """
        inner = call.args[0] if call.args else None
        if not isinstance(inner, ast.Call):
            return None, None
        dt = self._dtype_str(self._kwarg(inner, "dtype"))
        shp_node = inner.args[0] if inner.args else None
        if shp_node is None:
            return None, dt
        elts = shp_node.elts if isinstance(shp_node, (ast.Tuple, ast.List)) else [shp_node]
        axes = []
        for e in elts:
            f = self.eval_dim(e, local)
            if f is None:
                return None, dt
            axes.append(render_term(f))
        return tuple(axes), dt

    def _dtype_str(self, node):
        """dtype 实参 → 规范串。`mstype.float32` → "fp32";`config.params_dtype` → flags 值;
        解不出 → None(调用方退回 params_dtype)。"""
        if node is None:
            return None
        if isinstance(node, ast.Attribute):
            cfg = self._config_attr(node)
            if cfg is not None:
                v = self.flags.get(cfg)
                return v if isinstance(v, str) else None
            return _MSTYPE2DTYPE.get(node.attr, node.attr)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return _MSTYPE2DTYPE.get(node.value, node.value)
        return None

    def _do_assign(self, tgt, value, local):
        # build_module(...) 赋值:捕捉 (in,out) 维度,不把该 self 名当标量
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "build_module":
            name = self._self_attr(tgt)
            if name is not None:
                self.self_env[name] = _Cell(kind="module")
            dims = self._build_module_dims(value, local) if name is not None else None
            if dims is not None:
                in_f, out_f = dims
                self.linear_dims[name] = (
                    render_term(in_f) if in_f else None,
                    render_term(out_f) if out_f else None,
                )
            return
        cell = self._cell_for(value, local)
        ctor = self._ctor_name(value)
        if ctor == "Parameter":
            cell.kind = "param"          # `self.attn_sink = Parameter(mint.zeros(...))`
            name = self._self_attr(tgt)
            if name is not None:
                shp, dt = self._param_shape_of(value, local)
                if shp is not None:
                    self.param_shapes[name] = shp
                self.param_dtypes[name] = dt or self.flags.get("params_dtype") or "bf16"
        elif isinstance(value, ast.Call) and ctor and ctor[:1].isupper():
            cell.kind = "module"         # 类实例化(Cell / 原语类)
        sattr = self._self_attr(tgt)
        if sattr is not None:
            self.self_env[sattr] = cell
        elif isinstance(tgt, ast.Name):
            local[tgt.id] = cell

    def _do_aug(self, stmt: ast.AugAssign, local):
        tgt = stmt.target
        if isinstance(tgt, ast.Name):
            base = local.get(tgt.id)
        elif self._self_attr(tgt) is not None:
            base = self.self_env.get(self._self_attr(tgt))
        else:
            return
        if base is None or base.dim is None:
            return
        rhs = self.eval_dim(stmt.value, local)
        if rhs is None:
            return
        new = mul(base.dim, rhs) if isinstance(stmt.op, ast.Mult) else None
        if new is None:
            return
        cell = _Cell(dim=new)
        if isinstance(tgt, ast.Name):
            local[tgt.id] = cell
        else:
            self.self_env[self._self_attr(tgt)] = cell


def _param_defaults(init_fn: ast.FunctionDef) -> dict:
    args = init_fn.args.args
    defaults = init_fn.args.defaults
    n, nd = len(args), len(defaults)
    out = {}
    for i, a in enumerate(args):
        j = i - (n - nd)
        if j >= 0:
            d = defaults[j]
            if isinstance(d, ast.Constant):
                out[a.arg] = d.value
    return out


def eval_init_dims(tree: ast.AST, cls_name: str, config_flags: dict,
                   mro_units: list | None = None) -> InitDims:
    """求值 cls_name 的 __init__(沿 MRO base→derived),返回 linear_dims + dims_ctx + self_kinds。

    `mro_units`(可选,P0#3):`[(tree, 类名), ...]`,**derived→base** 顺序,用于**跨文件** MRO
    (`DSv4HybridSelfAttention` 的基类 `MultiLatentAttention` 在另一个文件里,单 tree 找不到)。
    不给时退化为在 `tree` 内做同文件 BFS(既有调用逐字不变)。
    """
    ev = _Eval(config_flags)
    if mro_units is None:
        classes = _init_classes(tree, cls_name)     # derived→base
        units = [(tree, c) for c in classes]
    else:
        units = list(mro_units)
    for utree, cname in reversed(units):        # base 先,derived 覆盖
        cls = _find_class(utree, cname)
        init_fn = _init_of(cls) if cls is not None else None
        if init_fn is None:
            continue
        local: dict[str, _Cell] = {}
        # 形参缺省(input_size=None / is_expert=False 等)注入 local
        for pname, pval in _param_defaults(init_fn).items():
            local[pname] = _Cell(val_known=True, val=pval)
        # 调用方**显式**给的形参种子(`INIT_PARAM_SEEDS`)覆盖缺省 —— 见该常量的 docstring:
        # `compress_ratio` 缺省 0 而真机 4/128、`head_dim` 同名于另一个维度,都不能靠推。
        _names = {a.arg for a in init_fn.args.args} | {a.arg for a in init_fn.args.kwonlyargs}
        for pname, seed in ev.param_seeds.items():
            if pname not in _names:
                continue
            if isinstance(seed, str):
                local[pname] = _Cell(dim=parse_axis(seed))
            elif isinstance(seed, bool):
                local[pname] = _Cell(val_known=True, val=seed)
            elif isinstance(seed, int):
                local[pname] = _Cell(dim=Factors(coeff=seed), val_known=True, val=seed)
        ev.exec_body(init_fn.body, local)
    dims_ctx = {k: render_term(c.dim) for k, c in ev.self_env.items()
                if c.dim is not None and _all_known_dim(c.dim)}
    self_kinds = {k: c.kind for k, c in ev.self_env.items()}
    return InitDims(linear_dims=ev.linear_dims, dims_ctx=dims_ctx, self_kinds=self_kinds,
                    param_shapes=ev.param_shapes, param_dtypes=ev.param_dtypes)
