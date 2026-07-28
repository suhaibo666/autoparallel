# cost_eval/opdag/fn_saves.py
"""自定义 autograd `_Function` 的**源真值**抽取（纯 AST，不 import mindspore）。

动机（`docs/opdag_coverage_assessment_2026-07-25.md` §4/§5，路线 A）：融合注意力走的是
`_Function` 子类，它们**没有 `construct`**，`construct_walker` 的入口写死在 `construct`
（`construct_walker.py:990`）→ 走不进去。但**它们的 saved 集在源里是逐字写着的**，~10 行 AST
就能取到（实测 `csa.py:113` 8 项、`:224` 11 项）。手写 `saves=[...]` 名册漂移这一类错误，
`crosscheck.py` 结构上发现不了（它只比 op 类别计数，§1）——本模块 + `tests/
test_source_truth_saves_audit.py` 把它永久钉住。

抽三样东西，**都不猜、不建图**：

1. **`ctx.save_for_backward(...)` 实参名单** —— 真源两种形态：
   - `plain_args`：`ctx.save_for_backward(gathered, w)`（`pynative/layers/mc2.py:83`）;
   - `starred_comprehension`：`ctx.save_for_backward(*[tensor for tensor in (<元组>)
     if tensor is not None])`（`csa.py:113` / `:224`，两处一模一样）。
   还从 `backward` 侧读 `next(ctx.saved_tensors)` 的**消费序**（`csa.py:134-142` / `:251-262`）
   作**同序自校验** —— forward 名单与 backward 消费序逐名相等是源自身给的冗余校验。

2. **裸 `ctx.<attr> = <rhs>` 赋值** —— 这类**绕过** MindSpore `saved_tensors_hooks`
   （hooks 只挂在 `save_for_backward` 通路上），所以它们既不会被 offload、也不会被重算释放
   逻辑看见。真源实例：
   - `indexer.py:276` `_IndexerLossAutoScaler.forward`：`ctx.indexer_loss = indexer_loss`;
   - `dsa_indexer.py:54-56` `_DSAIndexerFunction`：`ctx.q / ctx.k / ctx.weights`（**三个大张量**）;
   - `dsa_indexer_loss.py:70-72` `_DSAIndexerGradFunction`：`ctx.d_query_index` 等三份**梯度**。
   与之对照，`csa.py:111-112`（bool 谓词）与 `:123-128`（kernel 标量）**不是**张量。本模块
   **不猜**，给两个各自定义明确的判据（都报，互为上/下界）：
   - `tensor_candidate`（**上界**，纯句法）：`kind` 不是 `predicate`/`const`。
   - `bwd_tensor_evidence`（**下界**，源证据）：`backward` 里该 attr（或绑定它的局部名）**被
     张量内在函数消费** —— 出现在 `mint.*` / `ops.*` 自由调用的实参里、或 `isinstance(x, DTensor)`
     里、或对它调张量方法（`.to_local()` / `.mul_()` / `.contiguous()` …）。实测判别力：
     `indexer.py:284-285`（`isinstance(indexer_loss, DTensor)` + `mint.ones_like`）、
     `dsa_indexer.py:64-66`（`mint.zeros_like(ctx.q)`）、`dsa_indexer_loss.py:87-89`
     （`d_query_index.mul_(...)`）→ 全部命中；而 `csa.py:123-128` 的 6 个 kernel 标量只以
     `softmax_scale=ctx.softmax_scale` 形式做 kwarg、`ctx.loss_coeff` 只参与算术 → 全部不命中。
     **纯算术不算证据**（标量也做算术），这是刻意保守。

3. **detach 站点与 `_no_grad` 区域** —— `ops.stop_gradient(...)`（真源 6 处全在 `csa.py`：
   `:665,666`、`:764,765`、`:794,795`）、`.detach()` 方法调用、以及 `with _no_grad():` 块
   （`indexer.py:214`；**注意**评估文档 §3.1/§6.2 记的 `:220` 是块内的
   `npu_lightning_indexer(` 调用行，此处以实测 `:214` 为准）。`:794-795` 那两条**内联实参**形态
   是 `ukl1`/`ukl2` 判决（KL 目标分布链无参数 → 从不 saved）的源依据，故必须同时支持
   「赋值形」与「内联实参形」。

**纪律**：任何看不懂的 `save_for_backward` 形态一律进 `unresolved`（`strict=True` 直接抛），
**绝不**当成「无 saved 集」静默丢 —— 这是与 Task 2（P0#1 fail-loud 门）同一条不变量。
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "FUNCTION_BASES", "DETACH_CALLS",
    "SavedSet", "CtxAttr", "DetachSite", "NoGradRegion",
    "UnresolvedSaveForm", "FunctionRecord", "FnSourceTruth",
    "scan_source", "scan_tree",
    "mhc_kernel_saves", "KERNEL_SAVE_SOURCE_KEYS",
]

# `_Function` 是 MindSpore 自定义 autograd 基类（`from mindspore.ops import ...` 各版本导入
# 路径不同，故按**基类名**匹配，含 `a.b._Function` 这类点号形式取末段）。
FUNCTION_BASES = frozenset({"_Function", "Function"})
# detach 自由函数（点号全名）。真源只出现 `ops.stop_gradient`；`mint.detach` / `F.stop_gradient`
# 预置以覆盖同族写法。`.detach()` 方法调用另走 `form="method"`。
DETACH_CALLS = frozenset({"ops.stop_gradient", "mint.detach", "F.stop_gradient",
                          "stop_gradient"})
_NO_GRAD_NAMES = frozenset({"_no_grad", "no_grad"})
# `bwd_tensor_evidence` 的判据（见模块 docstring）：张量内在函数命名空间 + 张量方法。
_TENSOR_NAMESPACES = ("mint.", "ops.", "mindspore.mint.", "mindspore.ops.")
_TENSOR_METHODS = frozenset({
    "to_local", "contiguous", "detach", "astype", "view", "reshape", "permute",
    "transpose", "float", "half", "bfloat16", "to", "copy_",
    "mul_", "add_", "sub_", "div_", "zero_", "clamp_",
})


# ---------------------------------------------------------------------------
# 数据类（纯数据，JSON 可往返）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SavedSet:
    """一处 `ctx.save_for_backward(...)` 的逐字名单。"""
    file: str
    lineno: int
    names: tuple                  # tuple[str, ...] —— 逐字 unparse（多为裸名）
    form: str                     # "plain_args" | "starred_comprehension" | "starred_seq"
    conditional: bool = False     # 名单被 `if <x> is not None` 过滤（→ 实际项数可少于 names）

    @property
    def src(self) -> str:
        return f"{self.file}:{self.lineno}"


@dataclass(frozen=True)
class CtxAttr:
    """一处裸 `ctx.<attr> = <rhs>`（**绕过** saved_tensors_hooks）。"""
    cls: str
    func: str
    file: str
    lineno: int
    attr: str
    rhs: str                      # ast.unparse(rhs)
    kind: str                     # predicate|const|name|attribute|call|expr
    tensor_candidate: bool        # 上界（纯句法）：kind 非 predicate/const
    is_forward_param: bool = False
    bwd_tensor_evidence: bool = False   # 下界（源证据）：backward 里被张量内在函数消费
    bwd_uses: tuple = ()          # backward 里引用该 attr 的表达式（unparse，纯事实）

    @property
    def src(self) -> str:
        return f"{self.file}:{self.lineno}"


@dataclass(frozen=True)
class DetachSite:
    """一处 detach。`assigned` 为被赋名（内联实参形态为 None）。"""
    file: str
    lineno: int
    fn: str                       # "ops.stop_gradient" | ".detach"
    arg: str                      # 被 detach 的表达式（unparse）
    form: str                     # assign|assign_ifexp|inline_arg|method|expr
    assigned: Optional[str] = None
    cls: Optional[str] = None
    func: Optional[str] = None
    enclosing_call: Optional[str] = None   # 内联实参形态：外层被调者（unparse 的 func）

    @property
    def src(self) -> str:
        return f"{self.file}:{self.lineno}"


@dataclass(frozen=True)
class NoGradRegion:
    """一处 `with _no_grad():` —— 块内产物**整块 detach**。"""
    file: str
    lineno: int
    end_lineno: int
    assigned: tuple               # tuple[str, ...] 块内被绑定的名（按出现序）
    cls: Optional[str] = None
    func: Optional[str] = None
    items: tuple = ()             # with 的 context 表达式（unparse），可多项

    @property
    def src(self) -> str:
        return f"{self.file}:{self.lineno}"


@dataclass(frozen=True)
class UnresolvedSaveForm:
    """看不懂的 `save_for_backward` 实参形态 —— 显式记录，绝不静默丢。"""
    cls: str
    file: str
    lineno: int
    expr: str
    reason: str

    @property
    def src(self) -> str:
        return f"{self.file}:{self.lineno}"


@dataclass(frozen=True)
class FunctionRecord:
    """一个 `_Function` 子类的源真值。"""
    cls: str
    file: str
    lineno: int
    bases: tuple
    saved: Optional[SavedSet] = None
    ctx_attrs: tuple = ()
    backward_read_order: tuple = ()
    backward_conditional: tuple = ()
    has_unresolved_save: bool = False

    @property
    def src(self) -> str:
        return f"{self.file}:{self.lineno}"


@dataclass(frozen=True)
class FnSourceTruth:
    """一个文件或一棵源树的抽取结果。"""
    functions: tuple = ()
    detach_sites: tuple = ()
    no_grad_regions: tuple = ()
    unresolved: tuple = ()
    files: tuple = ()

    def by_class(self, name: str) -> Optional[FunctionRecord]:
        for rec in self.functions:
            if rec.cls == name:
                return rec
        return None

    def saved_names(self, name: str) -> tuple:
        rec = self.by_class(name)
        if rec is None or rec.saved is None:
            raise KeyError(f"{name}: 无 _Function 记录或无 save_for_backward")
        return rec.saved.names

    def tensor_candidate_ctx_attrs(self) -> tuple:
        """裸 ctx 张量的**上界**（纯句法：kind 非 predicate/const）。"""
        return tuple(c for c in _all_ctx_attrs(self) if c.tensor_candidate)

    def confirmed_tensor_ctx_attrs(self) -> tuple:
        """裸 ctx 张量的**下界**（backward 侧有张量内在函数消费证据）——绕过 saved_tensors_hooks。"""
        return tuple(c for c in _all_ctx_attrs(self)
                     if c.tensor_candidate and c.bwd_tensor_evidence)

    def merge(self, other: "FnSourceTruth") -> "FnSourceTruth":
        return FnSourceTruth(
            functions=self.functions + other.functions,
            detach_sites=self.detach_sites + other.detach_sites,
            no_grad_regions=self.no_grad_regions + other.no_grad_regions,
            unresolved=self.unresolved + other.unresolved,
            files=self.files + other.files,
        )


def _all_ctx_attrs(truth: FnSourceTruth) -> list:
    out = []
    for rec in truth.functions:
        out.extend(rec.ctx_attrs)
    return out


# ---------------------------------------------------------------------------
# AST 辅助
# ---------------------------------------------------------------------------
def _dotted(node) -> str:
    """`ops.stop_gradient` / `self.unfused_indexer_loss` → 点号全名；取不出返回 ""。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _base_names(cls: ast.ClassDef) -> tuple:
    return tuple(ast.unparse(b) for b in cls.bases)


def _is_function_subclass(cls: ast.ClassDef) -> bool:
    for b in cls.bases:
        name = _dotted(b) or ast.unparse(b)
        if name.rsplit(".", 1)[-1] in FUNCTION_BASES:
            return True
    return False


def _assigned_names(stmts) -> tuple:
    """块内被绑定的名（Assign/AnnAssign/AugAssign 目标，含元组解包），按出现序去重。"""
    out, seen = [], set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for tgt in targets:
                for elt in (tgt.elts if isinstance(tgt, (ast.Tuple, ast.List)) else [tgt]):
                    if isinstance(elt, ast.Name) and elt.id not in seen:
                        seen.add(elt.id)
                        out.append(elt.id)
    return tuple(out)


def _seq_elts(node):
    """Tuple/List → elts；否则 None。"""
    return node.elts if isinstance(node, (ast.Tuple, ast.List)) else None


# ---------------------------------------------------------------------------
# save_for_backward 名单
# ---------------------------------------------------------------------------
def _parse_save_call(call: ast.Call, file: str, cls: str):
    """→ (SavedSet | None, UnresolvedSaveForm | None)。看不懂就产 Unresolved，不静默丢。"""
    args = call.args
    starred = [a for a in args if isinstance(a, ast.Starred)]
    if not starred:
        if not args:
            return (SavedSet(file=file, lineno=call.lineno, names=(),
                             form="plain_args"), None)
        return (SavedSet(file=file, lineno=call.lineno,
                         names=tuple(ast.unparse(a) for a in args),
                         form="plain_args"), None)
    if len(starred) != 1 or len(args) != 1:
        return (None, UnresolvedSaveForm(
            cls=cls, file=file, lineno=call.lineno, expr=ast.unparse(call),
            reason="混合 *starred 与裸实参（或多个 *starred），实参名单不可逐字确定"))
    value = starred[0].value
    # `*[t for t in (<seq>) if ...]` / `*(t for t in (<seq>) if ...)`
    if isinstance(value, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
        gens = value.generators
        if len(gens) != 1 or _seq_elts(gens[0].iter) is None:
            return (None, UnresolvedSaveForm(
                cls=cls, file=file, lineno=call.lineno, expr=ast.unparse(call),
                reason="推导式的 iter 不是字面元组/列表，名单不可逐字确定"))
        elts = _seq_elts(gens[0].iter)
        return (SavedSet(file=file, lineno=call.lineno,
                         names=tuple(ast.unparse(e) for e in elts),
                         form="starred_comprehension",
                         conditional=bool(gens[0].ifs)), None)
    # `*(a, b)` / `*[a, b]`
    elts = _seq_elts(value)
    if elts is not None:
        return (SavedSet(file=file, lineno=call.lineno,
                         names=tuple(ast.unparse(e) for e in elts),
                         form="starred_seq"), None)
    return (None, UnresolvedSaveForm(
        cls=cls, file=file, lineno=call.lineno, expr=ast.unparse(call),
        reason=f"`*{ast.unparse(value)}` 非字面序列/推导式，实参名单不可逐字确定"))


# ---------------------------------------------------------------------------
# 裸 ctx.<attr> 分类
# ---------------------------------------------------------------------------
def _rhs_kind(rhs) -> str:
    if isinstance(rhs, (ast.Compare, ast.BoolOp)):
        return "predicate"
    if isinstance(rhs, ast.UnaryOp) and isinstance(rhs.op, ast.Not):
        return "predicate"
    if isinstance(rhs, ast.Constant):
        return "const"
    if isinstance(rhs, ast.Name):
        return "name"
    if isinstance(rhs, ast.Attribute):
        return "attribute"
    if isinstance(rhs, ast.Call):
        return "call"
    return "expr"


def _ctx_attr_records(cls_name, fn: ast.FunctionDef, ctx_name, file, bwd_ev=None):
    """收集 `ctx.<attr> = <rhs>`。判据见模块 docstring（上界纯句法 / 下界源证据）。"""
    params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    if fn.args.vararg:
        params.add(fn.args.vararg.arg)
    bwd_ev = bwd_ev or {}
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == ctx_name):
                continue
            kind = _rhs_kind(node.value)
            is_param = isinstance(node.value, ast.Name) and node.value.id in params
            ev, uses = bwd_ev.get(tgt.attr, (False, ()))
            out.append(CtxAttr(
                cls=cls_name, func=fn.name, file=file, lineno=node.lineno,
                attr=tgt.attr, rhs=ast.unparse(node.value), kind=kind,
                tensor_candidate=kind not in ("predicate", "const"),
                is_forward_param=is_param,
                bwd_tensor_evidence=ev, bwd_uses=uses))
    return out


def _bwd_tensor_evidence(fn: ast.FunctionDef, ctx_name):
    """扫 `backward`，判定每个 `ctx.<attr>` 是否被**张量内在函数**消费。

    → {attr: (evidence: bool, uses: tuple[str, ...])}。`uses` 是引用该 attr 的最近外层调用
    （或语句）的 unparse —— 纯事实记录，供人工复核，不参与判定。
    """
    # 1) 别名：`x = ctx.attr` → x 代表 attr（真源 indexer.py:282 / dsa_indexer_loss.py:84-86）
    alias = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == ctx_name):
            alias[node.targets[0].id] = node.value.attr

    def _refers(node):
        """该表达式是否直指某个 ctx attr（含别名）→ attr 名，否则 None。"""
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == ctx_name):
            return node.attr
        if isinstance(node, ast.Name) and node.id in alias:
            return alias[node.id]
        return None

    ev, uses = {}, {}
    # 2) 收集全部引用点的最近外层调用（用于 `uses`）
    for parent in ast.walk(fn):
        for child in ast.iter_child_nodes(parent):
            attr = _refers(child)
            if attr is None:
                continue
            uses.setdefault(attr, [])
            text = ast.unparse(parent)
            if text not in uses[attr]:
                uses[attr].append(text)
    # 3) 判据：mint./ops. 自由调用实参、isinstance(x, DTensor)、张量方法调用
    for call in ast.walk(fn):
        if not isinstance(call, ast.Call):
            continue
        dotted = _dotted(call.func)
        args = list(call.args) + [k.value for k in call.keywords]
        if dotted.startswith(_TENSOR_NAMESPACES) or dotted == "isinstance":
            targets = call.args if dotted == "isinstance" else args
            for a in targets:
                attr = _refers(a)
                if attr is not None:
                    ev[attr] = True
        if isinstance(call.func, ast.Attribute) and call.func.attr in _TENSOR_METHODS:
            attr = _refers(call.func.value)
            if attr is not None:
                ev[attr] = True
    return {a: (ev.get(a, False), tuple(uses.get(a, ()))) for a in set(ev) | set(uses)}


# ---------------------------------------------------------------------------
# backward 侧消费序
# ---------------------------------------------------------------------------
def _is_next_saved(value, iter_names) -> bool:
    return (isinstance(value, ast.Call) and _dotted(value.func) == "next"
            and len(value.args) == 1 and isinstance(value.args[0], ast.Name)
            and value.args[0].id in iter_names)


def _backward_read_order(fn: ast.FunctionDef, ctx_name):
    """`saved_tensors = iter(ctx.saved_tensors)` + `x = next(saved_tensors)` 的消费序；
    也支持 `a, b = ctx.saved_tensors` 元组解包。返回 (order, conditional_names)。"""
    iter_names = set()
    order, conditional = [], []
    for stmt in ast.walk(fn):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        tgt, val = stmt.targets[0], stmt.value
        # saved_tensors = iter(ctx.saved_tensors)
        if (isinstance(tgt, ast.Name) and isinstance(val, ast.Call)
                and _dotted(val.func) == "iter" and len(val.args) == 1
                and _dotted(val.args[0]) == f"{ctx_name}.saved_tensors"):
            iter_names.add(tgt.id)
            continue
        # a, b = ctx.saved_tensors
        if _dotted(val) == f"{ctx_name}.saved_tensors":
            elts = _seq_elts(tgt)
            if elts is not None:
                order.extend(ast.unparse(e) for e in elts)
            continue
        if not isinstance(tgt, ast.Name):
            continue
        if _is_next_saved(val, iter_names):
            order.append(tgt.id)
        elif isinstance(val, ast.IfExp) and _is_next_saved(val.body, iter_names):
            order.append(tgt.id)
            conditional.append(tgt.id)
    return tuple(order), tuple(conditional)


# ---------------------------------------------------------------------------
# detach / _no_grad
# ---------------------------------------------------------------------------
def _detach_call_name(call: ast.Call):
    """→ "ops.stop_gradient" / ".detach" / None。"""
    dotted = _dotted(call.func)
    if dotted in DETACH_CALLS:
        return dotted
    if (isinstance(call.func, ast.Attribute) and call.func.attr == "detach"
            and not call.args and not call.keywords):
        return ".detach"
    return None


def _detach_arg(call: ast.Call, fn_name) -> str:
    if fn_name == ".detach":
        return ast.unparse(call.func.value)
    return ast.unparse(call.args[0]) if call.args else ""


class _DetachVisitor(ast.NodeVisitor):
    """遍历一棵 tree，收 detach 站点 + `_no_grad` 区域，携带 cls/func 上下文。"""

    def __init__(self, file):
        self.file = file
        self.sites = []
        self.regions = []
        self._cls = None
        self._func = None
        # call node -> (form, assigned) 预登记（由 Assign 层填），detach 访问时取用
        self._assign_hint = {}
        self._call_stack = []

    # -- 上下文 -------------------------------------------------------------
    def visit_ClassDef(self, node):
        prev, self._cls = self._cls, node.name
        self.generic_visit(node)
        self._cls = prev

    def _visit_func(self, node):
        prev, self._func = self._func, node.name
        self.generic_visit(node)
        self._func = prev

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    # -- 赋值：预登记「这个 Call 是某个名的 RHS」---------------------------
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            val = node.value
            if isinstance(val, ast.Call) and _detach_call_name(val):
                self._assign_hint[id(val)] = ("assign", name)
            elif isinstance(val, ast.IfExp) and isinstance(val.body, ast.Call) \
                    and _detach_call_name(val.body):
                self._assign_hint[id(val.body)] = ("assign_ifexp", name)
        self.generic_visit(node)

    # -- 调用 ---------------------------------------------------------------
    def visit_Call(self, node):
        fn_name = _detach_call_name(node)
        if fn_name is not None:
            hint = self._assign_hint.get(id(node))
            if hint is not None:
                form, assigned, encl = hint[0], hint[1], None
            elif self._call_stack:
                form, assigned = "inline_arg", None
                encl = _dotted(self._call_stack[-1].func) or \
                    ast.unparse(self._call_stack[-1].func)
            else:
                form, assigned, encl = ("method" if fn_name == ".detach" else "expr"), None, None
            if hint is not None and fn_name == ".detach":
                form = "method"
            self.sites.append(DetachSite(
                file=self.file, lineno=node.lineno, fn=fn_name,
                arg=_detach_arg(node, fn_name), form=form, assigned=assigned,
                cls=self._cls, func=self._func, enclosing_call=encl))
        self._call_stack.append(node)
        self.generic_visit(node)
        self._call_stack.pop()

    # -- with _no_grad(): ---------------------------------------------------
    def visit_With(self, node):
        items = tuple(ast.unparse(it.context_expr) for it in node.items)
        hit = False
        for it in node.items:
            expr = it.context_expr
            target = expr.func if isinstance(expr, ast.Call) else expr
            name = (_dotted(target) or "").rsplit(".", 1)[-1]
            if name in _NO_GRAD_NAMES:
                hit = True
        if hit:
            self.regions.append(NoGradRegion(
                file=self.file, lineno=node.lineno,
                end_lineno=node.body[-1].end_lineno if node.body else node.lineno,
                assigned=_assigned_names(node.body), cls=self._cls,
                func=self._func, items=items))
        self.generic_visit(node)

    visit_AsyncWith = visit_With


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------
def scan_source(file: str, text: str, strict: bool = False) -> FnSourceTruth:
    """抽一个文件。`file` 只作定位符（写进 `src`），不读盘。

    `strict=True`：遇到看不懂的 `save_for_backward` 形态直接 `ValueError`
    （与 Task 2 的 fail-loud 门同一条纪律）。
    """
    tree = ast.parse(text, filename=file)
    functions, unresolved = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not _is_function_subclass(node):
            continue
        saved = None
        ctx_attrs = []
        order, conditional = (), ()
        cls_unresolved = False
        # 先过 backward，拿张量消费证据（forward 的 ctx_attrs 判定要用）。
        bwd_ev = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and item.name == "backward":
                pos = item.args.args
                bwd_ev = _bwd_tensor_evidence(item, pos[0].arg if pos else "ctx")
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            pos = item.args.args
            ctx_name = pos[0].arg if pos else "ctx"
            if item.name == "forward":
                for call in ast.walk(item):
                    if not (isinstance(call, ast.Call)
                            and isinstance(call.func, ast.Attribute)
                            and call.func.attr == "save_for_backward"):
                        continue
                    got, bad = _parse_save_call(call, file, node.name)
                    if bad is not None:
                        unresolved.append(bad)
                        cls_unresolved = True
                    elif saved is None:
                        saved = got
                    else:   # 同一 forward 里多处 save_for_backward → 不可逐字合并
                        unresolved.append(UnresolvedSaveForm(
                            cls=node.name, file=file, lineno=call.lineno,
                            expr=ast.unparse(call),
                            reason=f"同一 forward 内多处 save_for_backward（已见 {saved.src}）"))
                        cls_unresolved = True
                ctx_attrs.extend(_ctx_attr_records(node.name, item, ctx_name, file, bwd_ev))
            elif item.name == "backward":
                order, conditional = _backward_read_order(item, ctx_name)
                ctx_attrs.extend(_ctx_attr_records(node.name, item, ctx_name, file, bwd_ev))
        functions.append(FunctionRecord(
            cls=node.name, file=file, lineno=node.lineno, bases=_base_names(node),
            saved=saved, ctx_attrs=tuple(ctx_attrs),
            backward_read_order=order, backward_conditional=conditional,
            has_unresolved_save=cls_unresolved))

    visitor = _DetachVisitor(file)
    visitor.visit(tree)
    truth = FnSourceTruth(
        functions=tuple(functions),
        detach_sites=tuple(sorted(visitor.sites, key=lambda s: s.lineno)),
        no_grad_regions=tuple(sorted(visitor.regions, key=lambda r: r.lineno)),
        unresolved=tuple(unresolved), files=(file,))
    if strict and truth.unresolved:
        raise ValueError(
            "fn_saves: 看不懂的 save_for_backward 形态（strict=True 拒绝静默丢）：\n  "
            + "\n  ".join(f"{u.src} {u.cls}: {u.reason} -> {u.expr}"
                          for u in truth.unresolved))
    return truth


def scan_tree(root: str, strict: bool = False, recurse: bool = True) -> FnSourceTruth:
    """抽一棵源树（或单个目录）。`src` 里的路径相对 `root`，用 posix 分隔便于断言。"""
    root = os.path.abspath(root)
    truth = FnSourceTruth()
    paths = []
    if recurse:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            paths.extend(os.path.join(dirpath, f)
                         for f in filenames if f.endswith(".py"))
    else:
        paths = [os.path.join(root, f) for f in sorted(os.listdir(root))
                 if f.endswith(".py")]
    for path in sorted(paths):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        truth = truth.merge(scan_source(rel, text, strict=strict))
    if strict and truth.unresolved:
        raise ValueError(
            "fn_saves: 看不懂的 save_for_backward 形态（strict=True 拒绝静默丢）：\n  "
            + "\n  ".join(f"{u.src} {u.cls}: {u.reason}" for u in truth.unresolved))
    return truth


# ═══════════════════════════════════════════════════════════════════════════════
# 融合 NPU 自定义算子（`hyper_parallel` 侧 `DFunction`）的 saved 集 —— **从源逐字读**
# ═══════════════════════════════════════════════════════════════════════════════
#
# 为什么单开一段而不复用上面的 `_Function` 通路：`hyper_parallel` 的包装类基类叫
# `DFunction`，且它的 `forward` 里 saved 名单**混着两类东西**——
#   ① `forward` 的形参（= 调用点的**输入**操作数）；
#   ② 本次调用**自己的输出**（先 `_, _, _, h_pre, ... = result` 解包再存）。
# 上面的 `SavedSet` 只给名字串，分不出这两类；而字节记账里它们的去向完全不同
# （输入 → `saved_ins_idx`，输出 → `saved_outs_idx`，后者的形状还得另找源侧依据）。
#
# 处置纪律（与 `docs/opdag_component_coverage_2026-07-25.md` §1.4 的「绝不猜」一脉）：
#   * 名单、位序、输出名 —— **全部** AST 读出，形态不认即 fail-loud；
#   * 输出**形状** —— 源侧 Python 只给得出**阶数**（tensor_map 元组长度），末轴长度在
#     `.cc` kernel 里（`custom_op_impl.py:38-41` 列了 4 个 `.cc`，**未纳入快照**）。
#     故只对「注释里逐字写出且阶数与 tensor_map 对得上」的两项给形状，其余**不给数**。

#: 声明来源在节点 attrs 上的两个键 —— 「从源读出来的」与「调用方声明的上界」必须可区分，
#: 绝不把上界静默升格为事实（任务书铁律）。
KERNEL_SAVE_SOURCE_KEYS = ("saved_from_source", "saved_declared_by_caller")

_DFUNCTION_BASES = frozenset({"DFunction"})
_MHC_IMPL_REL = "platform/mindspore/custom_ops/custom_op_impl.py"
_MHC_LAYOUT_REL = "core/shard/ops/parallel_mhc_pre_sinkhorn.py"

#: `hyper_parallel` 里 mHC 两个内核的 `DFunction` 包装类（`_op_name` 由 AST 逐字读回校验）。
_MHC_CLASSES = {
    "NpuMhcPostDFunction": "npu_mhc_post",
    "NpuMhcPreSinkhornDFunction": "npu_mhc_pre_sinkhorn",
}

#: **源侧确定**的输出形状（逐字来自 `parallel_mhc_pre_sinkhorn.py` 的行内注释），
#: 每条带 `(形状串, 定位符, 期望阶数)`。阶数会与 `infer_output_layouts` 里那个 tensor_map
#: 元组的长度**逐条对账**（对不上即 fail-loud）——所以这不是裸常量，是有源侧交叉校验的读数。
#: 其余输出（`h_pre` / `hc_before_norm` / `inv_rms`）共用 `tm_3d = (b_map, s_map, -1)`，
#: 末轴写 `-1`（复制，与分片无关）**不给长度** → 一律不进本表，消费方落 `unresolved`。
_MHC_PRE_OUT_SHAPES = {
    "sum_out": ("2·num_iters·B·S·N", _MHC_LAYOUT_REL + ":262-263", 4),
    "norm_out": ("2·num_iters·B·S·N·N", _MHC_LAYOUT_REL + ":264-265", 5),
}


def _dfunction_classes(tree: ast.AST) -> dict:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and (
                set(_base_names(node)) & _DFUNCTION_BASES):
            out[node.name] = node
    return out


def _method_named(cls: ast.ClassDef, name: str):
    return next((n for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _class_str_attr(cls: ast.ClassDef, attr: str):
    for stmt in cls.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == attr
                and isinstance(stmt.value, ast.Constant)):
            return stmt.value.value
    return None


def _result_unpack_names(fwd: ast.FunctionDef, result_name: str) -> list:
    """`_, _, _, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = result` → 逐位名字。

    `_` 位记为 `None`（源里被丢弃的输出，位序仍占）。找不到这条解包 → 空表。
    """
    for stmt in fwd.body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        tgt, val = stmt.targets[0], stmt.value
        if not (isinstance(tgt, ast.Tuple) and isinstance(val, ast.Name)
                and val.id == result_name):
            continue
        return [None if (isinstance(e, ast.Name) and e.id == "_")
                else (e.id if isinstance(e, ast.Name) else None)
                for e in tgt.elts]
    return []


def _output_ranks(layout_tree: ast.AST, cls_name: str, method: str) -> dict:
    """`infer_output_layouts` 的 `return (_create_output_layout(mesh, tm_X), ...)` →
    `{输出位序: 该 tm_X 元组的长度}`（= 该输出的**阶数**，源侧确定的那一半）。

    只取 4-D 输入那一支（`if x_tm_len == 4:`，即 BSND；本模型 `input_layout=BSND`）。
    """
    cls = next((n for n in ast.walk(layout_tree)
                if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    fn = _method_named(cls, method) if cls is not None else None
    if fn is None:
        return {}
    branch = None
    for stmt in fn.body:
        if not isinstance(stmt, ast.If):
            continue
        cmp_ = stmt.test
        if (isinstance(cmp_, ast.Compare) and len(cmp_.comparators) == 1
                and isinstance(cmp_.comparators[0], ast.Constant)
                and cmp_.comparators[0].value == 4):
            branch = stmt.body
            break
    if branch is None:
        return {}
    tm_len: dict = {}
    for stmt in branch:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Tuple)):
            tm_len[stmt.targets[0].id] = len(stmt.value.elts)
    ranks: dict = {}
    for stmt in branch:
        if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Tuple):
            continue
        for i, e in enumerate(stmt.value.elts):
            if (isinstance(e, ast.Call) and len(e.args) == 2
                    and isinstance(e.args[1], ast.Name)):
                ranks[i] = tm_len.get(e.args[1].id)
        break
    return ranks


def mhc_kernel_saves(hp_root: str) -> dict:
    """融合 mHC 两个内核的 saved 集 —— **逐字从 `hyper_parallel` 源读**，不是上界。

    背景：`hyper_parallel` 于 2026-07-25 补入权威快照之前，这两个内核的 bprop 读不出来，
    只能由调用方声明一个「张量实参全存」的保守上界
    （`docs/opdag_component_coverage_2026-07-25.md` §1.4）。现在可以读了，而且读出来的结果
    **与那个上界实质不同**：

      * `npu_mhc_post`（`custom_op_impl.py:331`）
        `ctx.save_for_backward(x, h_res, h_out, h_post)` → 4 个输入全存，**恰等于**旧上界；
      * `npu_mhc_pre_sinkhorn`（`custom_op_impl.py:390-391`）
        `ctx.save_for_backward(x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms,
                               sum_out, norm_out)`
        → 4 个输入 **+ 自己的 5 个输出**。旧上界漏掉了后 5 项 ⇒ 它**不是上界，是欠读**，
        其中 `sum_out`（`2·num_iters·B·S·N`）/ `norm_out`（`2·num_iters·B·S·N·N`）
        还随 `num_iters`（缺省 20，`parallel_mhc_pre_sinkhorn.py:26`）线性放大。

    返回 `{op_name: {saved_ins_idx, saved_outs_idx, saved_out_names, saved_out_shapes,
                     saved_out_ranks, source, reason, saved_from_source}}`。
    形态不认一律 `ValueError`（fail-loud），**绝不**退回一个「看起来合理」的名单。
    """
    impl_path = os.path.join(hp_root, *_MHC_IMPL_REL.split("/"))
    if not os.path.isfile(impl_path):
        raise ValueError(
            "fn_saves: 找不到 " + _MHC_IMPL_REL
            + "（hyper_parallel 未纳入快照？）—— fail-loud")
    with open(impl_path, "r", encoding="utf-8") as fh:
        impl_tree = ast.parse(fh.read(), filename=impl_path)
    classes = _dfunction_classes(impl_tree)

    layout_path = os.path.join(hp_root, *_MHC_LAYOUT_REL.split("/"))
    layout_tree = None
    if os.path.isfile(layout_path):
        with open(layout_path, "r", encoding="utf-8") as fh:
            layout_tree = ast.parse(fh.read(), filename=layout_path)

    out: dict = {}
    for cls_name, want_op in _MHC_CLASSES.items():
        cls = classes.get(cls_name)
        if cls is None:
            raise ValueError(
                "fn_saves: " + _MHC_IMPL_REL + " 里没有 DFunction 子类 " + cls_name)
        op_name = _class_str_attr(cls, "_op_name")
        if op_name != want_op:
            raise ValueError(
                "fn_saves: " + cls_name + "._op_name = " + repr(op_name)
                + " != " + repr(want_op) + " —— 快照与本表不一致,拒绝按旧假设继续(fail-loud)")
        fwd = _method_named(cls, "forward")
        if fwd is None:
            raise ValueError("fn_saves: " + cls_name + " 无 forward —— fail-loud")
        params = [a.arg for a in fwd.args.args][1:]        # 去掉 ctx
        save_call = next(
            (n for n in ast.walk(fwd)
             if isinstance(n, ast.Call) and _dotted(n.func).endswith("save_for_backward")),
            None)
        if save_call is None:
            raise ValueError(
                "fn_saves: " + cls_name + ".forward 里没有 `ctx.save_for_backward(...)` —— "
                "拒绝当成「无 saved 集」静默放行(fail-loud)")
        saved, unresolved = _parse_save_call(save_call, _MHC_IMPL_REL, cls_name)
        if saved is None:
            raise ValueError("fn_saves: " + cls_name + " 的 save_for_backward 形态看不懂："
                             + unresolved.reason)

        # 本次调用**自己输出**的逐位名字（`_, _, _, h_pre, ... = result`）。
        result_name = next(
            (s.targets[0].id for s in fwd.body
             if isinstance(s, ast.Assign) and len(s.targets) == 1
             and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Call)
             and _dotted(s.value.func).endswith(op_name)), None)
        out_names = _result_unpack_names(fwd, result_name) if result_name else []

        ins_idx, outs_idx, out_saved_names = [], [], []
        for nm in saved.names:
            if nm in params:
                ins_idx.append(params.index(nm))
            elif nm in out_names:
                outs_idx.append(out_names.index(nm))
                out_saved_names.append(nm)
            else:
                raise ValueError(
                    "fn_saves: " + cls_name + " 存了 " + repr(nm) + ",它既不是 forward 形参 "
                    + repr(params) + " 也不是输出解包名 " + repr(out_names)
                    + " —— 不猜,fail-loud")

        ranks: dict = {}
        if layout_tree is not None and op_name == "npu_mhc_pre_sinkhorn":
            ranks = _output_ranks(layout_tree, "NpuMhcPreSinkhornDistributedOp",
                                  "infer_output_layouts")
        shapes: dict = {}
        for nm in out_saved_names:
            spec = _MHC_PRE_OUT_SHAPES.get(nm)
            if spec is None:
                continue                    # 源侧不确定 → **不给数**，消费方落 unresolved
            shp, loc, want_rank = spec
            got = ranks.get(out_names.index(nm))
            if got is not None and got != want_rank:
                raise ValueError(
                    "fn_saves: " + nm + " 的 tensor_map 阶数 " + str(got)
                    + " != 本表记的 " + str(want_rank) + "（" + loc
                    + "）—— 快照变了,拒绝按旧读数继续(fail-loud)")
            shapes[nm] = shp

        out[op_name] = {
            "saved_ins_idx": sorted(ins_idx),
            "saved_outs_idx": sorted(outs_idx),
            # **按输出位序**建索引（不是一串平表）：`derive_saves` 要按 idx 取名字，
            # 而源里那些输出被 `*_` 丢弃、图上没有 ref，只能靠这张表登记。
            "saved_out_names": {out_names.index(n): n for n in out_saved_names},
            "saved_out_shapes": shapes,
            "saved_out_ranks": {out_names.index(n): ranks.get(out_names.index(n))
                                for n in out_saved_names if ranks},
            "source": _MHC_IMPL_REL + ":" + str(saved.lineno),
            "reason": ("`ctx.save_for_backward(" + ", ".join(saved.names) + ")` "
                       "逐字读自 hyper_parallel 快照(commit 41495aa2)"),
            "saved_from_source": True,
        }
    return out
