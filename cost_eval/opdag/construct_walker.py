# cost_eval/opdag/construct_walker.py
"""Pass C(设计 §3.3):静态走查 Cell.construct() 的 AST,产出 op-DAG。

原理:construct 里每条 `self.<name>(...)` 调用 = 一个算子。我们**从不执行**源码,
只按**源序**遍历语句,用一张 SSA 表 `varname -> ref` 记录"哪个中间变量当前由哪个
节点产出",从而在后续调用消费该变量时补一条数据流边 [producer_id, consumer_id]。

设计取向(供 MLA / MoE 抽取任务扩展,勿加 MLP 专属 hack):
  * 语句处理器(_handle_assign / _handle_call)与算子发射(_emit)解耦;
  * dtype 随 SSA 变量传播,Cast(含 `x.astype(dtype)`)会改写产出变量的 dtype;
  * 未在 Pass B 绑定的 self.<name> 一律 **fail-loud**(静默丢算子 = DAG 少算子 = 错)。

ref 串格式与 schema/bprop 一致:`"name:符号shape:dtype"`,三段、各段不含冒号;
shape 此阶段一律占位 `?`(由后续 shape 解析任务回填)。
"""
from __future__ import annotations
import ast

from .schema import OpNode, OpDAG

# 三态哨兵:一个 `if`/三元条件在剪枝上下文下无法由已知 config 判定。
_UNDECIDED = object()

# mindspore dtype 名 → 本库短标签(best-effort;未知名原样透传)。
_DTYPE_ALIAS = {
    "float32": "fp32", "float": "fp32",
    "float16": "fp16", "half": "fp16",
    "bfloat16": "bf16", "bfloat": "bf16",
    "float64": "fp64", "double": "fp64",
    "int32": "int32", "int64": "int64", "int16": "int16", "int8": "int8",
    "uint8": "uint8", "bool_": "bool", "bool": "bool",
}


def _dtype_name(expr) -> str | None:
    """从一个"命名 dtype 的表达式"里抽出短标签:ms.float32 / mstype.bfloat16 / "float32"。
    解析不出返回 None(交给上层兜底)。"""
    tok = None
    if isinstance(expr, ast.Attribute):        # ms.float32 / mstype.bfloat16
        tok = expr.attr
    elif isinstance(expr, ast.Name):           # 裸名 float32
        tok = expr.id
    elif isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        tok = expr.value
    if tok is None:
        return None
    return _DTYPE_ALIAS.get(tok, tok)


def _target_names(targets) -> list[str]:
    """把赋值左侧目标摊平成变量名列表:x / (a, b) / [a, b] 都支持;非 Name 目标忽略。"""
    names: list[str] = []
    for tgt in targets:
        if isinstance(tgt, ast.Name):
            names.append(tgt.id)
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            for e in tgt.elts:
                if isinstance(e, ast.Name):
                    names.append(e.id)
    return names


def _self_attr(node) -> str | None:
    """若 node 形如 `self.<name>` 返回 <name>,否则 None。"""
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self"):
        return node.attr
    return None


def _none_compare_var(left, right) -> str | None:
    """识别 `<Name> is/is not None` 里的变量名(None 常量可在任一侧);不匹配返回 None。"""
    for a, b in ((left, right), (right, left)):
        if isinstance(a, ast.Name) and isinstance(b, ast.Constant) and b.value is None:
            return a.id
    return None


class _Walker:
    """一次 construct 走查的可变状态容器(便于子类/后续任务复用与扩展)。"""

    def __init__(
        self,
        binds: dict,
        src_file: str,
        config_flags: dict | None = None,
        none_vars: set | None = None,
        param_defaults: dict | None = None,
    ):
        self.binds = binds                 # self.<name> -> Binding(op, attrs)(Pass B 产)
        self.src_file = src_file
        self.nodes: list[OpNode] = []
        self.edges: list[list[int]] = []
        self.ssa: dict[str, str] = {}      # varname -> 当前 ref "name:?:dtype"
        self.producer: dict[str, int] = {} # varname -> 产出它的节点 id
        self._next_id = 1                  # 节点 id 从 1 单调递增
        # ---- 剪枝上下文(§Task5):任一非 None 即"开启剪枝",此后不可判定的 if → fail-loud ----
        self.config_flags: dict = dict(config_flags or {})   # self.<flag> / self.<attr>==字面量
        self.param_defaults: dict = dict(param_defaults or {})  # construct 形参的缺省值
        # "已知为 None 的变量名":显式 none_vars ∪ 缺省即 None 的 construct 形参。
        self.known_none: set[str] = set(none_vars or ())
        for k, v in self.param_defaults.items():
            if v is None:
                self.known_none.add(k)
        self._pruning = any(x is not None for x in (config_flags, none_vars, param_defaults))

    # ---- 语句层:按源序遍历,分派到具体处理器 ----
    def walk_body(self, body) -> None:
        for stmt in body:
            self.walk_stmt(stmt)

    def walk_stmt(self, stmt) -> None:
        if isinstance(stmt, ast.Assign):
            self._handle_assign(stmt)
        elif isinstance(stmt, ast.Expr):
            # 裸表达式语句(无赋值目标),只关心其中的调用
            if isinstance(stmt.value, ast.Call):
                self._handle_call(stmt.value, target_names=[])
        elif isinstance(stmt, ast.If):
            if self._pruning:
                # 配置门控分支:按 config 求值,只走命中支;不可判定 → fail-loud(拒绝双走)。
                self._handle_if_pruned(stmt)
            else:
                # 无剪枝上下文 → 保持旧的"两支都按源序线性走查"行为(Task 4 回归)。
                self.walk_body(stmt.body)
                self.walk_body(stmt.orelse)
        elif isinstance(stmt, ast.Raise):
            # 剪枝模式下:能"走到"一条 raise 说明它在被选中的分支里 → 选到了不支持的路径,fail-loud。
            # (未命中的 raise 守卫根本不会被 walk 到,天然跳过。)非剪枝模式忽略 raise。
            if self._pruning:
                raise ValueError(
                    f"construct 剪枝命中被选中的 raise 分支（{self.src_file}:{stmt.lineno}）:"
                    f"`{self._describe(stmt)}` —— config 实际选到了不支持的路径,fail-loud"
                )
        # return / pass / For / While / 增强赋值等:当前不产 op(后续任务按需扩展)

    # ---- if 剪枝:求值条件,只走命中支 ----
    def _handle_if_pruned(self, stmt: ast.If) -> None:
        r = self._eval_test(stmt.test)
        if r is _UNDECIDED:
            raise ValueError(
                f"construct 的 if 条件无法由 config 判定（{self.src_file}:{stmt.lineno}）:"
                f"`{self._describe(stmt.test)}` —— 剪枝上下文下拒绝双走(会重复计激活/内存),fail-loud"
            )
        self.walk_body(stmt.body if r else stmt.orelse)

    def _handle_assign(self, stmt: ast.Assign) -> None:
        val = stmt.value
        targets = _target_names(stmt.targets)
        if isinstance(val, ast.Call):
            self._handle_call(val, targets)
        elif isinstance(val, ast.IfExp):
            # 三元:`self.act(x) if <cond> else x`。按 config 求值,只落命中侧那个表达式。
            self._handle_ifexp(val, targets)
        # 其它(算术/常量/切片/属性)当前不产 op

    def _handle_ifexp(self, ifexp: ast.IfExp, targets: list[str]) -> None:
        r = self._eval_test(ifexp.test)
        if r is _UNDECIDED:
            if self._pruning:
                raise ValueError(
                    f"construct 的三元条件无法由 config 判定（{self.src_file}:{ifexp.lineno}）:"
                    f"`{self._describe(ifexp.test)}` —— 剪枝上下文下拒绝双走,fail-loud"
                )
            r = True  # 非剪枝:保守取 body(优先保留算子,别静默丢)
        chosen = ifexp.body if r else ifexp.orelse
        if isinstance(chosen, ast.Call):
            self._handle_call(chosen, targets)
        elif isinstance(chosen, ast.Name):
            # `... else x`:目标别名 x —— 复用 x 的 ref 与 producer(下游消费能连回真源)。
            self._alias(targets, chosen.id)
        # 其它表达式(字面量等):不产 op,目标不登记 SSA

    def _alias(self, targets: list[str], src_name: str) -> None:
        ref = self.ssa.get(src_name, f"{src_name}:?:bf16")
        prod = self.producer.get(src_name)
        for t in targets:
            self.ssa[t] = ref
            if prod is not None:
                self.producer[t] = prod

    # ---- 调用层:识别 self.<name>(...) 与 <expr>.astype(dtype) 两种形态 ----
    def _handle_call(self, call: ast.Call, target_names: list[str]) -> None:
        func = call.func
        # 形态一:self.<name>(...)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            name = func.attr
            binding = self.binds.get(name)
            if binding is None:
                # fail-loud:未绑定算子决不能静默跳过(否则 DAG 缺算子 → 代价算错)
                raise ValueError(
                    f"construct 调用了未绑定的 self.{name}(...)（{self.src_file}:{call.lineno}）:"
                    f"Pass B(init_binder)未覆盖此名,fail-loud"
                )
            if binding.op == "Cast":
                # cast 目标 dtype 取第 2 个位置实参(idx=1):self.cast(x, ms.float32)
                out_dtype = self._resolve_cast_dtype(call.args, 1, binding.attrs)
            else:
                out_dtype = binding.attrs.get("compute_dtype") or "bf16"
            self._emit(binding.op, binding.attrs, call.lineno, list(call.args), target_names, out_dtype)
            return
        # 形态二:<expr>.astype(<dtype>) —— 视作 Cast(dtype 传播关键路径)
        if isinstance(func, ast.Attribute) and func.attr == "astype":
            out_dtype = self._resolve_cast_dtype(call.args, 0, {})  # astype 目标 dtype 在 idx=0
            self._emit("Cast", {}, call.lineno, [func.value], target_names, out_dtype)
            return
        # 其它调用(非 self.、非 astype):当前不产 op(后续任务按需扩展)

    # ---- 条件求值层(剪枝):返回 True / False / _UNDECIDED,决不"猜" ----
    def _eval_test(self, test):
        if isinstance(test, ast.Constant):
            return bool(test.value)
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            r = self._eval_test(test.operand)
            return _UNDECIDED if r is _UNDECIDED else (not r)
        if isinstance(test, ast.BoolOp):
            return self._eval_boolop(test)
        if isinstance(test, ast.Compare):
            return self._eval_compare(test)
        if isinstance(test, ast.Attribute):
            # `if self.<flag>:` → bool(config_flags[flag])
            nm = _self_attr(test)
            if nm is not None and nm in self.config_flags:
                return bool(self.config_flags[nm])
            return _UNDECIDED
        if isinstance(test, ast.Name):
            # `if <var>:` —— 已知 None → False;有非 None 缺省 → 取其真值
            if test.id in self.known_none:
                return False
            if test.id in self.param_defaults:
                return bool(self.param_defaults[test.id])
            return _UNDECIDED
        return _UNDECIDED

    def _eval_boolop(self, node: ast.BoolOp):
        results = [self._eval_test(v) for v in node.values]
        if isinstance(node.op, ast.And):
            if any(r is False for r in results):
                return False
            if all(r is True for r in results):
                return True
            return _UNDECIDED
        # Or
        if any(r is True for r in results):
            return True
        if all(r is False for r in results):
            return False
        return _UNDECIDED

    def _eval_compare(self, node: ast.Compare):
        if len(node.ops) != 1:
            return _UNDECIDED
        op = node.ops[0]
        left, right = node.left, node.comparators[0]
        # `<x> is/is not None`
        if isinstance(op, (ast.Is, ast.IsNot)):
            var = _none_compare_var(left, right)
            if var is None:
                return _UNDECIDED
            if var in self.known_none:
                val_is_none = True
            elif var in self.param_defaults:
                val_is_none = self.param_defaults[var] is None
            else:
                return _UNDECIDED
            return val_is_none if isinstance(op, ast.Is) else (not val_is_none)
        # `<x> ==/!= 字面量`
        if isinstance(op, (ast.Eq, ast.NotEq)):
            lk, lv = self._value(left)
            rk, rv = self._value(right)
            if not (lk and rk):
                return _UNDECIDED
            return (lv == rv) if isinstance(op, ast.Eq) else (lv != rv)
        # `<x> in (字面量...)` / `not in`
        if isinstance(op, (ast.In, ast.NotIn)):
            lk, lv = self._value(left)
            if not lk or not isinstance(right, (ast.Tuple, ast.List)):
                return _UNDECIDED
            elts = []
            for e in right.elts:
                ek, ev = self._value(e)
                if not ek:
                    return _UNDECIDED
                elts.append(ev)
            inside = lv in elts
            return inside if isinstance(op, ast.In) else (not inside)
        return _UNDECIDED

    def _value(self, node):
        """把一个表达式求成"已知 config 值":返回 (known: bool, value)。"""
        if isinstance(node, ast.Constant):
            return True, node.value
        nm = _self_attr(node)  # self.<attr> → config_flags
        if nm is not None and nm in self.config_flags:
            return True, self.config_flags[nm]
        if isinstance(node, ast.Name):
            if node.id in self.known_none:
                return True, None
            if node.id in self.param_defaults:
                return True, self.param_defaults[node.id]
        return False, None

    @staticmethod
    def _describe(node) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return ast.dump(node)

    def _resolve_cast_dtype(self, args, idx: int, attrs: dict) -> str:
        """解析一次 cast 的目标 dtype:优先 idx 位置实参,其次 attrs['to_dtype'],
        最后保守 fp32(cast 多用于升精度)。"""
        if len(args) > idx:
            d = _dtype_name(args[idx])
            if d:
                return d
        if attrs.get("to_dtype"):
            return attrs["to_dtype"]
        return "fp32"

    # ---- 发射层:建 OpNode + 连数据流边 + 更新 SSA ----
    def _emit(self, op: str, attrs: dict, lineno: int, arg_exprs, target_names, out_dtype: str) -> OpNode:
        node_id = self._next_id
        self._next_id += 1

        ins: list[str] = []
        seen_prod: set[int] = set()  # 同一节点内对同一 producer 只连一条边(去重)
        for a in arg_exprs:
            if not isinstance(a, ast.Name):
                continue  # 只有张量形参(Name)算数据流操作数;字面量/属性/嵌套调用暂略
            if a.id in self.ssa:  # 已知 SSA 中间变量 → 用其当前 ref 并向其 producer 连边
                ins.append(self.ssa[a.id])
                prod = self.producer.get(a.id)
                if prod is not None and prod not in seen_prod:
                    self.edges.append([prod, node_id])
                    seen_prod.add(prod)
            else:  # 方法形参 / 未知名 → 占位 ref(shape 未知,dtype 缺省 bf16)
                ins.append(f"{a.id}:?:bf16")

        # 元组多目标(a, b = self.f(...)):首目标作主 out;所有目标都登记为本节点产出的 SSA,
        # 以便后续语句消费任意一个都能连回本节点。
        out_ref = f"{target_names[0]}:?:{out_dtype}" if target_names else ""
        node = OpNode(
            id=node_id, op=op, src=f"{self.src_file}:{lineno}",
            module=attrs.get("module", ""), ins=ins, out=out_ref, attrs=dict(attrs),
        )
        self.nodes.append(node)
        for t in target_names:
            self.ssa[t] = f"{t}:?:{out_dtype}"  # dtype 随 SSA 传播(Cast 会改写)
            self.producer[t] = node_id
        return node


def walk_construct(
    src: str,
    cls_name: str,
    binds: dict,
    src_file: str,
    config_flags: dict | None = None,
    none_vars: set | None = None,
    param_defaults: dict | None = None,
) -> OpDAG:
    """走查 `cls_name` 的 construct(),把每个 self.<name>(...) 调用落成 OpNode,返回 op-DAG。

    参数:
      src            — Cell 源码字符串(不执行,仅 ast 解析)
      cls_name       — 目标类名
      binds          — Pass B 产的 {self.<name> -> Binding(op, attrs)}
      src_file       — 源文件名,用于 OpNode.src=file:line 回指
      config_flags   — 可选。{"gated_linear_unit": True, "activation_type": "swiglu", ...}:
                       求值 `if self.<flag>:`(→ bool(flag))与 `if self.<attr> ==/!=/in 字面量:`。
      none_vars      — 可选。已知为 None 的变量名集合(如 bias 关闭时的 bias_parallel);
                       用于 `if <var> is [not] None:`。
      param_defaults — 可选。construct 形参在"不传实参"时的缺省值(如 {"per_token_scale": None});
                       缺省即 None 的形参并入 none_vars 语义,让顶部守卫 `if x is not None: raise` 剪掉。

    剪枝语义:config_flags/none_vars/param_defaults **任一非 None 即开启剪枝**——此后每个 `if`
    只走 config 命中支;**不可判定的 if → fail-loud**(拒绝双走:静默重复计激活比响亮停下更糟)。
    三者全为 None 时保持旧的"两支都线性走查"行为(Task 4 回归不受影响)。

    找不到类或其 construct 方法时 fail-loud(ValueError)。
    """
    tree = ast.parse(src)
    cls = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name),
        None,
    )
    if cls is None:
        raise ValueError(f"源码里找不到 class {cls_name}(fail-loud)")
    construct = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "construct"),
        None,
    )
    if construct is None:
        raise ValueError(f"class {cls_name} 缺少 construct 方法(fail-loud)")

    walker = _Walker(binds, src_file, config_flags, none_vars, param_defaults)
    walker.walk_body(construct.body)
    return OpDAG(cell=cls_name, nodes=walker.nodes, edges=walker.edges)
