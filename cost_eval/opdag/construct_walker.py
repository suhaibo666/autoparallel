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


class _Walker:
    """一次 construct 走查的可变状态容器(便于子类/后续任务复用与扩展)。"""

    def __init__(self, binds: dict, src_file: str):
        self.binds = binds                 # self.<name> -> Binding(op, attrs)(Pass B 产)
        self.src_file = src_file
        self.nodes: list[OpNode] = []
        self.edges: list[list[int]] = []
        self.ssa: dict[str, str] = {}      # varname -> 当前 ref "name:?:dtype"
        self.producer: dict[str, int] = {} # varname -> 产出它的节点 id
        self._next_id = 1                  # 节点 id 从 1 单调递增

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
            # 配置门控分支:两支都按源序线性走查(后续任务再按 config 剪枝,
            # 此处保持简单线性,不做条件求值)。
            self.walk_body(stmt.body)
            self.walk_body(stmt.orelse)
        # return / pass / For / While / 增强赋值等:当前不产 op(后续任务按需扩展)

    def _handle_assign(self, stmt: ast.Assign) -> None:
        if not isinstance(stmt.value, ast.Call):
            return  # 非调用赋值(算术/常量/切片)当前不产 op
        self._handle_call(stmt.value, _target_names(stmt.targets))

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


def walk_construct(src: str, cls_name: str, binds: dict, src_file: str) -> OpDAG:
    """走查 `cls_name` 的 construct(),把每个 self.<name>(...) 调用落成 OpNode,返回 op-DAG。

    参数:
      src       — Cell 源码字符串(不执行,仅 ast 解析)
      cls_name  — 目标类名
      binds     — Pass B 产的 {self.<name> -> Binding(op, attrs)}
      src_file  — 源文件名,用于 OpNode.src=file:line 回指
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

    walker = _Walker(binds, src_file)
    walker.walk_body(construct.body)
    return OpDAG(cell=cls_name, nodes=walker.nodes, edges=walker.edges)
