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

from .sym_shape import Factors, mul, add, floordiv, render_term, CONFIG2SYM, _split_top

_UNDECIDED = object()
# 真·维度符号集合(dims_ctx 只保留由这些符号构成的项,避免把 compute_dtype/input_layout 等
# 被当作维度符号透传;它们从不出现在 reshape/split 表达式里)。
_KNOWN_DIM_SYMS = set(CONFIG2SYM.values())


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


@dataclass
class _Cell:
    dim: Factors | None = None          # 维度(可乘除加)
    val_known: bool = False             # 是否求得具体值(用于条件判定)
    val: object = None                  # 具体值(int/bool/None/str)


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
        self.self_env: dict[str, _Cell] = {}
        self.linear_dims: dict = {}

    # ── 维度求值(→ Factors 或 None)──────────────────────────────────────────
    def eval_dim(self, node, local: dict) -> Factors | None:
        try:
            return self._eval_dim(node, local)
        except Exception:
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

    def _do_assign(self, tgt, value, local):
        # build_module(...) 赋值:捕捉 (in,out) 维度,不把该 self 名当标量
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "build_module":
            name = self._self_attr(tgt)
            if name is not None and len(value.args) >= 3:
                in_f = self.eval_dim(value.args[1], local)
                out_f = self.eval_dim(value.args[2], local)
                self.linear_dims[name] = (
                    render_term(in_f) if in_f else None,
                    render_term(out_f) if out_f else None,
                )
            return
        cell = self._cell_for(value, local)
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


def eval_init_dims(tree: ast.AST, cls_name: str, config_flags: dict) -> InitDims:
    """求值 cls_name 的 __init__(沿 MRO base→derived),返回 linear_dims + dims_ctx。"""
    ev = _Eval(config_flags)
    classes = _init_classes(tree, cls_name)     # derived→base
    for cname in reversed(classes):             # base 先,derived 覆盖
        cls = _find_class(tree, cname)
        init_fn = _init_of(cls)
        if init_fn is None:
            continue
        local: dict[str, _Cell] = {}
        # 形参缺省(input_size=None / is_expert=False 等)注入 local
        for pname, pval in _param_defaults(init_fn).items():
            local[pname] = _Cell(val_known=True, val=pval)
        ev.exec_body(init_fn.body, local)
    dims_ctx = {k: render_term(c.dim) for k, c in ev.self_env.items()
                if c.dim is not None and _all_known_dim(c.dim)}
    return InitDims(linear_dims=ev.linear_dims, dims_ctx=dims_ctx)
