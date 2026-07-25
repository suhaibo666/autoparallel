# cost_eval/opdag/construct_walker.py
"""Pass C(设计 §3.3):静态走查 Cell.construct() 的 AST,产出 op-DAG。

原理:construct 里每条 `self.<name>(...)` 调用 = 一个算子。我们**从不执行**源码,
只按**源序**遍历语句,用一张 SSA 表 `varname -> ref` 记录"哪个中间变量当前由哪个
节点产出",从而在后续调用消费该变量时补一条数据流边 [producer_id, consumer_id]。

设计取向(供 MLA / MoE 抽取任务扩展,勿加模型专属 hack):
  * 语句处理器(_handle_assign / _handle_call)与算子发射(_emit)解耦;
  * dtype 随 SSA 变量传播,Cast(含 `x.astype(dtype)`)会改写产出变量的 dtype;
  * 未在 Pass B 绑定的 self.<name>:若是**本类(含基类 MRO)里的一个内部方法**(`def`)→
    **内联展开**;否则 **fail-loud**(静默丢算子 = DAG 少算子 = 错)。

内部方法内联(MLA 需要——construct 在基类、helper 在派生类):
  * `_lookup_method(name)` 按 MRO(cls_name 起,沿同文件基类 BFS)找最贴近的 `def name`;
  * 内联时把方法形参绑定到调用点实参:**Name 实参 → 直接复用调用方变量名**(共享 SSA / present /
    known_none,数据流边跨内联边界连续);非 Name 实参 / 未传实参 → 帧内合成局部名(占位);
  * 方法内的**局部变量**改写为帧唯一名 `<name>__i<frame>`(避免与调用方/兄弟帧同名互扰);
  * 方法 `return <expr>` 值绑定回调用点赋值目标(元组按位对齐,别名到 producer);
  * 递归内联(方法直接/间接自调用,或深度超上限)→ fail-loud。

ref 串格式与 schema/bprop 一致:`"name:符号shape:dtype"`,三段、各段不含冒号;
shape 此阶段一律占位 `?`(由后续 shape 解析任务回填)。
"""
from __future__ import annotations
import ast
import copy
from dataclasses import dataclass, field

from .schema import OpNode, OpDAG

# ── 抽取诊断(Task 2 / 评估文档 P0#1,2026-07-25)──────────────────────────────────────────
# 语义与逐键含义见 `schema.OpDAG.diagnostics` 的 docstring。此处只声明键序(报告/summary 用)。
DIAG_KINDS = (
    "dropped_stmts",         # walk_stmt 不处理的语句类
    "dropped_assigns",       # _handle_assign 不支持的 RHS 形态
    "unregistered_targets",  # 赋值目标从未进 SSA(下游会拿占位 ref、丢边)
    "unresolved_operands",   # _emit 里落占位 ref 的操作数
    "unbound_aliases",       # __init__ 里 _CLS2OP 绑不上的裸函数别名(由 extractor 并入)
)


class ExtractionDroppedError(ValueError):
    """抽取过程**丢了东西**却本会静默通过 —— 显式拒绝。

    继承 `ValueError`:①既有 `pytest.raises(ValueError)` 惯用法照旧;②`crosscheck._check_segment`
    的 `except Exception` 会把它漏斗进 `extraction_failures` → `ok=False`、strict 下 raise
    （即「已声明覆盖族抽取失败 ≠ 合法无对应」那条既有纪律自动生效）。
    """


def _empty_diagnostics() -> dict:
    return {k: [] for k in DIAG_KINDS}


def diagnostics_summary(dag) -> dict:
    """`{kind: 条数}` + `total`(仅 DIAG_KINDS 之和) + `opaque_calls`。缺字段按 0。"""
    diag = getattr(dag, "diagnostics", None) or {}
    out = {k: len(diag.get(k) or ()) for k in DIAG_KINDS}
    out["total"] = sum(out[k] for k in DIAG_KINDS)
    out["opaque_calls"] = len(getattr(dag, "opaque_calls", None) or ())
    return out


def _format_diagnostics(dag, kinds) -> str:
    diag = getattr(dag, "diagnostics", None) or {}
    lines = []
    for kind in kinds:
        items = diag.get(kind) or ()
        if not items:
            continue
        lines.append(f"  [{kind}] {len(items)} 条:")
        for it in items:
            src = it.get("src", "?")
            what = it.get("code") or it.get("alias") or it.get("target") or it.get("operand") or ""
            tag = (it.get("node") or it.get("rhs") or it.get("cause")
                   or it.get("node_op") or it.get("attr") or "")
            note = f"  ({it['note']})" if it.get("note") else ""
            lines.append(f"    - {src} {tag}: {what}{note}")
    return "\n".join(lines)


def assert_extraction_clean(dag, *, allow=(), check_opaque: bool = False) -> None:
    """消费方的一行门:抽取诊断非空即 `ExtractionDroppedError`(可用 `allow` 显式豁免某些类)。

    评估文档 §11 的纪律「这三个列表非空时消费方默认拒绝出数」的落地。`check_opaque=True`
    时把 `opaque_calls` 也算进来(默认不算——它按设计是「消费方自带白名单」的语义)。
    """
    kinds = tuple(k for k in DIAG_KINDS if k not in allow)
    summary = diagnostics_summary(dag)
    bad = [k for k in kinds if summary[k]]
    opaque_bad = check_opaque and summary["opaque_calls"] and "opaque_calls" not in allow
    if not bad and not opaque_bad:
        return
    parts = [f"抽取诊断非空,拒绝当成「抽好了」(cell={getattr(dag, 'cell', '?')}):"]
    parts.append(_format_diagnostics(dag, bad))
    if opaque_bad:
        parts.append(f"  [opaque_calls] {summary['opaque_calls']} 条:")
        for it in dag.opaque_calls:
            parts.append(f"    - {it.get('src', '?')}: {it.get('expr', '')}")
    parts.append("  —— 静默丢算子 = DAG 少算子 = 字节少算。请补处理器,或用 allow=(...) 显式豁免。")
    raise ExtractionDroppedError("\n".join(p for p in parts if p))


@dataclass
class SubExtract:
    """一次子 Cell 递归抽取的结果(供父 walker 在调用点内联):
      * nodes/edges —— 子 DAG(id 从 1 起,src 指向子文件),edges 为子内部边(子编号);
      * param_names —— 子 construct 形参(去 self,按序),用于把父调用实参按位重映射到子操作数;
      * returns     —— 子 construct 返回值逐项分类:("node", 子内 producer id)/("param", 形参名)/("none", None),
                       用于把"子输出 → 下游消费者"的边接回父 SSA。
      * opaque_calls —— 子 walker 记录的 fallthrough 调用点(T0-6.5 Fix2),原样并入父 opaque_calls
                       (src 已指向子文件,不需重映射;镜像 nodes/edges 的子→父传播,防止子 Cell
                       边界二次静默丢)。
      * diagnostics —— 子 walker 的抽取诊断(Task 2),同理逐类并入父 diagnostics(src 已指子文件),
                       否则子 Cell 边界会把「子里丢了一大块」洗白。"""
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    param_names: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    opaque_calls: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

# 直接实例化即调用的算子 `OpClass(...)(...)`(mindspore 无状态原语的常见写法):类名 → (op 类型, attrs)。
# flatten=True:算子接受"张量列表"操作数(如 GroupedMatmul([x],[w],...)),把 List/Tuple 字面量摊平为多操作数。
DIRECT_OP_MAP = {
    "GroupedMatmul": ("GroupedMatMul", {"flatten": True}),  # MoE 分组 GEMM(experts_forward)
    "Reshape": ("View", {"view": "reshape"}),               # Morph 里的 Reshape()(x, shp)
}

# 具名"自由函数调用"(非 self.<x>、非直接实例化,形如 `mod.sub.func(...)`)按完整点号路径匹配:
# 点号路径 → (op 类型, attrs)。表的纪律:**只收实测触发点**(勿预置未在真源命中的别名写法)。
# T0-5 增补(spec §3.3a embedding 段实测触发)—— VocabParallelEmbedding.embedding_func(layers.py):
#   `mint.nn.functional.embedding(masked_input, weight)`(:160)—— mint 查表 lookup,归 Gather;
#   `ops.mul(output_parallel, input_mask)`(:165)—— TP mask 逐元素乘,归 Elementwise(linear=False)。
FREE_CALL_MAP = {
    "mint.nn.functional.embedding": ("Gather", {"embedding": True}),
    "ops.mul": ("Elementwise", {"linear": False}),
}


def _dotted_path(node) -> str | None:
    """把 `a.b.c.d` 形态的 Attribute 链摊平成点号路径串;链首非 Name(如以 Call/Subscript 起)则 None。"""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return ".".join(reversed(parts))

# 张量方法(链式 `<expr>.method(...)`)里视作纯视图/元数据(反向不新增激活)的方法名。
_VIEW_METHODS = {"reshape", "view", "transpose", "swapaxes", "flatten", "expand_dims",
                 "tile", "permute", "squeeze", "unsqueeze"}

# 三态哨兵:一个 `if`/三元条件在剪枝上下文下无法由已知 config 判定。
_UNDECIDED = object()
# "已知存在(非 None)"的取值哨兵(用于 present_vars 走 `if v is not None:` 真支)。
_PRESENT = object()

# mindspore dtype 名 → 本库短标签(best-effort;未知名原样透传)。
_DTYPE_ALIAS = {
    "float32": "fp32", "float": "fp32",
    "float16": "fp16", "half": "fp16",
    "bfloat16": "bf16", "bfloat": "bf16",
    "float64": "fp64", "double": "fp64",
    "int32": "int32", "int64": "int64", "int16": "int16", "int8": "int8",
    "uint8": "uint8", "bool_": "bool", "bool": "bool",
}


class _ReturnSignal(Exception):
    """内联方法体命中 `return` 的冒泡信号(携返回值表达式),供 _inline_method 捕获。"""
    __slots__ = ("value",)

    def __init__(self, value):
        super().__init__()
        self.value = value


class _Renamer(ast.NodeTransformer):
    """按 mapping 改写方法体里的 ast.Name.id(内联作用域重命名)。"""

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node


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


def _config_flag_name(node) -> str | None:
    """识别 config-flag 引用,返回其扁平 flag 名:
       `self.<name>` → <name>(如 hoist 的 self.use_dsa);
       `self.config.<name>` → <name>(如 self.config.q_lora_rank)。
    两者都在扁平 config_flags 字典里按叶子名查(不区分是否 hoist)。其它 → None。"""
    if not isinstance(node, ast.Attribute):
        return None
    v = node.value
    if isinstance(v, ast.Name) and v.id == "self":
        return node.attr
    if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
            and v.value.id == "self" and v.attr == "config"):
        return node.attr
    return None


def _assigned_names(body) -> set:
    """收集方法体(含嵌套 if 等)里所有被赋值的变量名——即"局部变量"候选。"""
    names: set = set()
    for top in body:
        for n in ast.walk(top):
            if isinstance(n, ast.Assign):
                names |= set(_target_names(n.targets))
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(n.target, ast.Name):
                    names.add(n.target.id)
    return names


class _Walker:
    """一次 construct 走查的可变状态容器(便于子类/后续任务复用与扩展)。"""

    _INLINE_CAP = 8   # 内联深度上限(防失控递归);MLA 实测仅 2 层。

    def __init__(
        self,
        binds: dict,
        src_file: str,
        config_flags: dict | None = None,
        none_vars: set | None = None,
        param_defaults: dict | None = None,
        present_vars: set | None = None,
        tree: ast.AST | None = None,
        cls_name: str | None = None,
        subcell_resolver=None,
        method_aliases: dict | None = None,
        strict: bool = False,
    ):
        self.binds = binds                 # self.<name> -> Binding(op, attrs)(Pass B 产)
        self.src_file = src_file
        self.nodes: list[OpNode] = []
        self.edges: list[list[int]] = []
        self.ssa: dict[str, str] = {}      # varname -> 当前 ref "name:?:dtype"
        self.producer: dict[str, int] = {} # varname -> 产出它的节点 id
        self._next_id = 1                  # 节点 id 从 1 单调递增
        # ---- 子 Cell 递归内联:resolver(cell_name, field, bare) -> SubExtract(None=不递归) ----
        self._subcell_resolver = subcell_resolver
        # ---- Morph(self.method) 别名:self.<attr> -> 被 Morph 包裹的方法名(调用点内联该方法)----
        self._method_aliases: dict = dict(method_aliases or {})
        self.construct_params: list[str] = []   # construct 形参(去 self),供上层做子内联时按位重映射
        self.returns: list = []                 # construct 返回值分类(见 SubExtract.returns)
        self._param_set: set[str] = set()
        # `<a,b,c> = x.shape`(不产 op 的标量解包)记录,供 shape 推断按 x 已知 shape 逐轴填标量名。
        self.scalar_binds: list = []
        # T0-6.5 Fix2:_handle_call 终端 fallthrough 命中的调用点(既非四种已知形态、也非内部方法/
        # Morph 别名)——显式记录,不静默丢(schema.OpDAG.opaque_calls docstring 详述语义)。
        self.opaque_calls: list = []
        # Task 2(P0#1):一切「看不懂而没建节点」的东西逐条记账(语义见 schema.OpDAG.diagnostics)。
        # 默认只记录(既有路径逐字节不变);strict=True 时走查结束统一抛。
        self.diagnostics: dict = _empty_diagnostics()
        self.strict = bool(strict)
        # ---- 内联支持:类层级 AST(找内部方法定义)+ 递归/帧状态 ----
        self._tree = tree
        self._cls_name = cls_name
        self._inline_stack: list[str] = [] # 当前内联链(检测递归)
        # 帧号:五族合成名共用的 uniquifier(前缀注册表——新增前缀勿与既有撞形):
        #   __ret__i<n>        _materialize_return_call  内联方法 `return <Call>` 的返回值物化
        #   __chain__i<n>      _handle_chained_call      链式调用 `<call>.method(...)` 的内层中值
        #   __arg__i<n>        _materialize_call_arg     嵌套 Call 实参物化(T0-6.5 Fix1)
        #   <local>__i<frame>  _inline_method            内联方法局部变量按帧改写(防跨帧同名互扰)
        #   <param>__i<frame>  _bind_params              非 Name 实参/未传形参的帧内占位名
        self._frame_seq = 0
        # ---- 剪枝上下文(§Task5):任一非 None 即"开启剪枝",此后不可判定的 if → fail-loud ----
        self.config_flags: dict = dict(config_flags or {})   # self.<flag> / self.config.<flag>==字面量
        self.param_defaults: dict = dict(param_defaults or {})  # construct 形参的缺省值
        self.present_vars: set = set(present_vars or ())     # 已知"存在(非 None)"的变量名
        # "已知为 None 的变量名":显式 none_vars ∪ 缺省即 None 的 construct 形参。
        self.known_none: set[str] = set(none_vars or ())
        for k, v in self.param_defaults.items():
            if v is None:
                self.known_none.add(k)
        self._pruning = any(
            x is not None for x in (config_flags, none_vars, param_defaults, present_vars)
        )

    # ---- 诊断记账(Task 2 / P0#1):看不懂就记,绝不静默 ----
    def _diag(self, kind: str, **rec) -> None:
        self.diagnostics[kind].append(rec)

    @staticmethod
    def _short_code(text: str, cap: int = 120) -> str:
        """复合语句(with/for/try)的 unparse 含整个块体 —— 报告里只留首行 + 尾部标记。"""
        first = text.split("\n", 1)[0]
        more = "  …" if "\n" in text else ""
        if len(first) > cap:
            first = first[:cap] + "…"
        return first + more

    def _diag_drop_stmt(self, stmt) -> None:
        """未处理的语句类。`with _no_grad():` 额外带「整块 detach」note(路线 B P0#4 的挂钩)。"""
        node = type(stmt).__name__
        note = ""
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            items = " / ".join(self._describe(it.context_expr) for it in stmt.items)
            if "_no_grad" in items or "no_grad" in items:
                note = ("`_no_grad()` = 整块 detach:块内产物应标 detached,而非当普通节点发射"
                        "——故此处刻意**不**内联走查(会造出一批无 detach 语义的假节点)")
            else:
                note = "with 块体未走查(walk_stmt 无处理器)"
        elif isinstance(stmt, ast.Delete):
            # `del x`(如 indexer.py:211 `del actual_seq_qlen, actual_seq_klen`)不产 op,但对
            # **liveness** 有语义(显式释放点)。记账、不当 op 丢弃处理。
            note = "del 无 op 语义,但是 liveness 的显式释放点(未建模)"
        self._diag("dropped_stmts", src=f"{self.src_file}:{stmt.lineno}", node=node,
                   code=self._short_code(self._describe(stmt)), note=note)

    def _diag_unregistered(self, targets, lineno: int, cause: str) -> None:
        for t in targets:
            self._diag("unregistered_targets", src=f"{self.src_file}:{lineno}",
                       target=t, cause=cause)

    def _raise_diagnostics(self, cell_name: str, head: str = "") -> None:
        kinds = [k for k in DIAG_KINDS if self.diagnostics.get(k)]
        if not kinds:
            return
        raise ExtractionDroppedError(
            (head or f"construct 走查有静默丢弃(strict=True 拒绝放行,cell={cell_name}):")
            + "\n" + _format_diagnostics(
                type("_D", (), {"cell": cell_name, "diagnostics": self.diagnostics,
                                "opaque_calls": self.opaque_calls})(), kinds))

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
            elif not isinstance(stmt.value, ast.Constant):
                # docstring(Expr(Constant))合法无 op;其余裸表达式(`x.foo`/await/…)是真丢弃。
                self._diag_drop_stmt(stmt)
        elif isinstance(stmt, ast.If):
            if self._pruning:
                # 配置门控分支:按 config 求值,只走命中支;不可判定 → fail-loud(拒绝双走)。
                self._handle_if_pruned(stmt)
            else:
                # 无剪枝上下文 → 保持旧的"两支都按源序线性走查"行为(Task 4 回归)。
                self.walk_body(stmt.body)
                self.walk_body(stmt.orelse)
        elif isinstance(stmt, ast.Return):
            self._handle_return(stmt)
        elif isinstance(stmt, ast.Raise):
            # 剪枝模式下:能"走到"一条 raise 说明它在被选中的分支里 → 选到了不支持的路径,fail-loud。
            # (未命中的 raise 守卫根本不会被 walk 到,天然跳过。)非剪枝模式忽略 raise。
            if self._pruning:
                raise ValueError(
                    f"construct 剪枝命中被选中的 raise 分支（{self.src_file}:{stmt.lineno}）:"
                    f"`{self._describe(stmt)}` —— config 实际选到了不支持的路径,fail-loud"
                )
        elif isinstance(stmt, (ast.Pass, ast.Break, ast.Continue, ast.Global,
                               ast.Nonlocal, ast.Import, ast.ImportFrom,
                               ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            pass    # 真·无算子语义(控制流/声明/嵌套定义):不是丢弃,不记账
        else:
            # Task 2(P0#1):**其余一切语句类**(With/For/While/Try/AugAssign/AnnAssign/Assert/
            # Delete/Match/…)此前是**静默 return**(实测 0 nodes / 0 opaque)——现逐条记账。
            # 实测最危险的一条:`indexer.py:214` 的 `with _no_grad():` 包住整个 fused indexer 支
            # → 整块丢 → `extract_cell` 返回 `ok, 0 nodes` 假成功(评估文档 §3.1)。
            self._diag_drop_stmt(stmt)

    # ---- if 剪枝:求值条件,只走命中支 ----
    def _handle_if_pruned(self, stmt: ast.If) -> None:
        r = self._eval_test(stmt.test)
        if r is _UNDECIDED:
            if self._is_pure_raise_guard(stmt):
                # 纯断言守卫:条件不可判定、整支仅 raise、无 else —— 视作对合法输入恒成立的
                # 校验断言(如 `if x.ndim != 3: raise`),跳过(不取 raise 支)而非 fail-loud。
                # config 门控的 raise 守卫仍可判定(有对应 flag),故只有真·数据/形状断言落到这里。
                return
            raise ValueError(
                f"construct 的 if 条件无法由 config 判定（{self.src_file}:{stmt.lineno}）:"
                f"`{self._describe(stmt.test)}` —— 剪枝上下文下拒绝双走(会重复计激活/内存),fail-loud"
            )
        self.walk_body(stmt.body if r else stmt.orelse)

    @staticmethod
    def _is_pure_raise_guard(stmt: ast.If) -> bool:
        return (not stmt.orelse) and bool(stmt.body) and all(
            isinstance(s, ast.Raise) for s in stmt.body
        )

    def _handle_return(self, stmt: ast.Return) -> None:
        """return 语句:
          * 内联方法体内:若返回值直接是 Call / Subscript(Call)(如 helper 的 `return self.op(x)`),
            先把该算子发射到一个合成目标,再以该合成名冒泡,供 _bind_return 别名到调用点目标;
            否则(Name/Tuple)原样冒泡表达式。
          * 顶层 construct:无赋值目标,但 `return self.<child>(x)` 这类直接返回的调用/子 Cell 仍需
            发射 / 递归内联(否则漏算子、且子 Cell 递归环检测不到)。"""
        val = stmt.value
        if self._inline_stack:
            raise _ReturnSignal(self._materialize_return_call(val))
        if isinstance(val, ast.Call):
            self._handle_call(val, target_names=[])
        elif isinstance(val, ast.Subscript) and isinstance(val.value, ast.Call):
            self._handle_call(val.value, target_names=[])
        # 返回 Name/Tuple/其它:顶层无需产 op(下游没有消费者)

    def _materialize_return_call(self, val):
        """内联方法 `return <Call>`:把该调用发射到合成临时名并返回该 Name(供别名到调用点目标)。"""
        call = None
        if isinstance(val, ast.Call):
            call = val
        elif isinstance(val, ast.Subscript) and isinstance(val.value, ast.Call):
            call = val.value
        if call is None:
            return val
        tmp = f"__ret__i{self._frame_seq}"
        self._frame_seq += 1
        self._handle_call(call, target_names=[tmp])
        return ast.Name(id=tmp, ctx=ast.Load())

    def _handle_assign(self, stmt: ast.Assign) -> None:
        val = stmt.value
        targets = _target_names(stmt.targets)
        if isinstance(val, ast.Call):
            self._handle_call(val, targets)
        elif isinstance(val, ast.Subscript) and isinstance(val.value, ast.Call):
            # `out = self.linear(x)[0]`:下标只是选调用的某个输出张量 → 按内层 Call 发射算子。
            self._handle_call(val.value, targets)
        elif isinstance(val, ast.IfExp):
            # 三元:`self.act(x) if <cond> else x`。按 config 求值,只落命中侧那个表达式。
            self._handle_ifexp(val, targets)
        elif (isinstance(val, ast.Attribute) and val.attr == "shape"
              and isinstance(val.value, ast.Name) and len(targets) >= 2):
            # `seq, bs, h = x.shape`:不产 op,但记标量→轴解包(shape 推断阶段按 x 已知 shape 填)。
            self.scalar_binds.append({"names": list(targets), "src": val.value.id})
        else:
            # Task 2(P0#1):此前静默丢。评估文档 §6.2 实测 **34 处**,且**恰好包含这次要修的张量**:
            #   `q_hnorm_fp32`(deepseek_v4_hybrid_attention.py:245 BinOp)、O(S·S/r) fp32
            #   `attention_scores`(indexer.py:350 BinOp)、`score_f32`(compressor.py:209 BinOp)、
            #   两个 bool mask(csa.py:779/810 Compare)、切片(compressor.py:198/199/233 Subscript)。
            # 记两笔:①RHS 形态本身;②目标名从未进 SSA(下游消费它会拿占位 ref、**丢边**)。
            self._diag("dropped_assigns", src=f"{self.src_file}:{stmt.lineno}",
                       targets=list(targets), rhs=type(val).__name__,
                       code=self._describe(stmt))
            self._diag_unregistered(targets, stmt.lineno, f"dropped_assign_rhs_{type(val).__name__}")

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

    # ---- 调用层:识别 self.<name>(...) / 内部方法内联 / <expr>.astype(dtype) ----
    def _handle_call(self, call: ast.Call, target_names: list[str]) -> None:
        func = call.func
        # 形态一:self.<name>(...)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            self._handle_self_call(call, func.attr, target_names)
            return
        # 形态二:直接实例化即调用的算子 `OpClass(...)(...)`(如 GroupedMatmul(split_item=3)(...)、Reshape()(x,shp))。
        if isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and func.func.id in DIRECT_OP_MAP:
            op, attrs = DIRECT_OP_MAP[func.func.id]
            arg_exprs = self._expand_args(call.args, {"variadic": True}) if attrs.get("flatten") else list(call.args)
            if op == "View" and attrs.get("view"):
                attrs = {**attrs, **self._view_capture(call, attrs["view"], target_names)}
            self._emit(op, attrs, call.lineno, arg_exprs, target_names, attrs.get("compute_dtype") or "bf16")
            return
        # 形态二.五:具名自由函数调用 `mod.sub.func(...)`(非 self.<x>、非直接实例化)——按完整点号路径
        # 精确匹配 FREE_CALL_MAP(如 `mint.nn.functional.embedding(...)` / `ops.mul(...)`)。
        # 与形态三互斥(形态三要求 func.value 是 Call/Subscript,而点号路径链首必为 Name——
        # _dotted_path 对 Call/Subscript 链头返回 None):若此处不命中,这类调用不会被形态三吞,
        # 而是落到本方法末尾的终端兜底记入 opaque_calls(T0-6.5 Fix2,不产节点)——故本分支
        # 仍是它们成为算子的唯一入口。
        if isinstance(func, ast.Attribute):
            path = _dotted_path(func)
            if path in FREE_CALL_MAP:
                op, attrs = FREE_CALL_MAP[path]
                self._emit(op, attrs, call.lineno, list(call.args), target_names,
                           attrs.get("compute_dtype") or "bf16")
                return
        # 形态三:链式方法 `<innercall>.method(...)`(如 self.swiglu(x).reshape(...)):先发射内层算子,再处理外层方法。
        if isinstance(func, ast.Attribute) and isinstance(func.value, (ast.Call, ast.Subscript)):
            self._handle_chained_call(call, func, target_names)
            return
        # 形态四:`<expr>.astype(<dtype>)`(expr 为 Name/Attribute)—— 视作 Cast(dtype 传播关键路径)。
        if isinstance(func, ast.Attribute) and func.attr == "astype":
            out_dtype = self._resolve_cast_dtype(call.args, 0, {})  # astype 目标 dtype 在 idx=0
            self._emit("Cast", {}, call.lineno, [func.value], target_names, out_dtype)
            return
        # 其它调用(非上述形态):当前不产 op(如 self.token_dispatcher.token_permutation —— AllToAll 派发,
        # opaque;或 layers.py:182 `ops.AllReduce(group=...)(x)` 双层调用形态)—— T0-6.5 Fix2:
        # 显式记录而非静默丢(walker 模块 docstring 纪律:静默丢算子 = DAG 少算子 = 错)。
        self.opaque_calls.append(
            {"src": f"{self.src_file}:{call.lineno}", "expr": self._describe(call)}
        )
        # Task 2(P0#1):终端 fallthrough 只记了「调用文本」,**赋值目标从未进 SSA/producer** →
        # 下游消费它会拿占位 ref `t:?:bf16` 且**无边**,与合法的 construct 形参操作数长得一模一样
        # （评估文档 §5 第 3 条:`ops.stop_gradient` 就是这样把数据流静默切断的）。单列记账。
        if target_names:
            self._diag_unregistered(target_names, call.lineno, "opaque_call")

    def _handle_self_call(self, call: ast.Call, name: str, target_names: list[str]) -> None:
        binding = self.binds.get(name)
        if binding is None:
            # 未绑定:①Morph(self.method) 别名 → 内联被包裹的方法;②本类(含基类)内部方法 → 内联;③否则 fail-loud。
            if name in self._method_aliases:
                m = self._lookup_method(self._method_aliases[name])
                if m is not None:
                    self._inline_method(m, call, target_names)
                    return
            method = self._lookup_method(name)
            if method is not None:
                self._inline_method(method, call, target_names)
                return
            raise ValueError(
                f"construct 调用了未绑定的 self.{name}(...)（{self.src_file}:{call.lineno}）:"
                f"Pass B(init_binder)未覆盖此名、且非本类内部方法/Morph 别名,fail-loud"
            )
        if binding.op == "SubCell" and self._subcell_resolver is not None:
            # 子 Cell:递归抽取其 DAG,并在调用点内联(镜像内部方法内联的 SSA/边/id 处理)。
            self._inline_subcell(binding.attrs, call, target_names)
            return
        if binding.op == "Cast":
            # cast 目标 dtype 取第 2 个位置实参(idx=1):self.cast(x, ms.float32 / self.compute_dtype)
            out_dtype = self._resolve_cast_dtype(call.args, 1, binding.attrs)
        else:
            out_dtype = binding.attrs.get("compute_dtype") or "bf16"
        attrs = binding.attrs
        if binding.op == "View" and binding.attrs.get("view"):
            # View 子类型:捕获变换元信息(reshape 目标 / split 尺寸 / perm / ...)供 shape 推断。
            attrs = {**binding.attrs, **self._view_capture(call, binding.attrs["view"], target_names)}
        arg_exprs = self._expand_args(call.args, binding.attrs)
        self._emit(binding.op, attrs, call.lineno, arg_exprs, target_names, out_dtype)

    def _handle_chained_call(self, call: ast.Call, func: ast.Attribute, target_names: list[str]) -> None:
        """链式方法 `<innercall>.method(...)`:先把内层调用发射到合成临时名,再按外层方法处理:
           .astype → Cast(消费临时名);视图类方法(reshape/view/…)→ 目标别名到临时名(不新增激活节点)。"""
        inner = func.value
        if isinstance(inner, ast.Subscript) and isinstance(inner.value, ast.Call):
            inner = inner.value
        tmp = f"__chain__i{self._frame_seq}"
        self._frame_seq += 1
        self._handle_call(inner, [tmp])          # 发射内层算子(如 swiglu)到 tmp
        method = func.attr
        if method == "astype":
            out_dtype = self._resolve_cast_dtype(call.args, 0, {})
            self._emit("Cast", {}, call.lineno, [ast.Name(id=tmp, ctx=ast.Load())], target_names, out_dtype)
        else:
            # reshape/view/transpose 等:纯视图/元数据,反向不新增激活 → 目标承接内层产物(保 producer 边);
            # ref 用目标名(而非合成临时名),使 save-set 可读。
            prod = self.producer.get(tmp)
            dt = self.ssa.get(tmp, f"{tmp}:?:bf16").split(":")[2]
            for t in target_names:
                self.ssa[t] = f"{t}:?:{dt}"
                if prod is not None:
                    self.producer[t] = prod

    # ---- View 变换元信息捕获(shape 推断的输入;walker 本身不求值)----
    @staticmethod
    def _tuple_elts(node):
        return list(node.elts) if isinstance(node, (ast.Tuple, ast.List)) else None

    @staticmethod
    def _const_int(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)):
            return -int(node.operand.value)
        return None

    def _kw_or_arg_int(self, call, name, argidx):
        for kw in call.keywords:
            if kw.arg == name:
                return self._const_int(kw.value)
        if len(call.args) > argidx:
            return self._const_int(call.args[argidx])
        return None

    def _view_capture(self, call: ast.Call, kind: str, target_names) -> dict:
        """把一次 View 调用的变换参数抠成"未求值符号表达式串"(reshape/split 目标)或整数(perm/axis)。"""
        U = self._describe
        args = call.args
        if kind == "reshape":
            elts = self._tuple_elts(args[1]) if len(args) >= 2 else None
            return {"reshape_dims": [U(e) for e in elts]} if elts is not None else {}
        if kind == "split":
            out = {"split_targets": list(target_names)}
            elts = self._tuple_elts(args[1]) if len(args) >= 2 else None
            if elts is not None:
                out["split_sizes"] = [U(e) for e in elts]
            dim = self._kw_or_arg_int(call, "dim", 2)
            if dim is not None:
                out["split_dim"] = dim
            return out
        if kind == "transpose":
            elts = self._tuple_elts(args[1]) if len(args) >= 2 else None
            return {"perm": [self._const_int(e) for e in elts]} if elts is not None else {}
        if kind == "expand_dims":
            ax = self._const_int(args[1]) if len(args) >= 2 else None
            return {"expand_axis": ax} if ax is not None else {}
        if kind == "tile":
            elts = self._tuple_elts(args[1]) if len(args) >= 2 else None
            return {"tile_mult": [U(e) for e in elts]} if elts is not None else {}
        if kind == "shape":
            return {"shape_src": (U(args[0]) if args else None), "shape_unpack": list(target_names)}
        return {}

    @staticmethod
    def _expand_args(args, attrs: dict):
        """variadic 算子(concat/stack 等)接受一个"张量列表"作单实参 → 摊平其元素为多操作数。
        非 variadic 算子原样返回(reshape 的形状元组等靠 _emit 只取 Name 实参天然忽略)。"""
        if not attrs.get("variadic"):
            return list(args)
        out = []
        for a in args:
            if isinstance(a, (ast.List, ast.Tuple)):
                out.extend(a.elts)
            else:
                out.append(a)
        return out

    # ---- 内部方法内联 ----
    def _find_class(self, name: str):
        if self._tree is None:
            return None
        return next(
            (n for n in ast.walk(self._tree)
             if isinstance(n, ast.ClassDef) and n.name == name),
            None,
        )

    def _lookup_method(self, name: str):
        """按 MRO(cls_name 起沿同文件基类 BFS)找最贴近的 `def name`,找不到返回 None。"""
        if self._tree is None or self._cls_name is None:
            return None
        seen: set = set()
        queue = [self._cls_name]
        while queue:
            c = queue.pop(0)
            if c in seen:
                continue
            seen.add(c)
            cls = self._find_class(c)
            if cls is None:
                continue
            m = next(
                (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name),
                None,
            )
            if m is not None:
                return m
            for b in cls.bases:
                if isinstance(b, ast.Name):
                    queue.append(b.id)
        return None

    def _inline_method(self, method: ast.FunctionDef, call: ast.Call, target_names: list[str]) -> None:
        name = method.name
        if name in self._inline_stack:
            raise ValueError(
                f"construct 内联检测到递归调用 self.{name}(...)（{self.src_file}:{call.lineno}）:"
                f"内联链 {' -> '.join(self._inline_stack + [name])},fail-loud"
            )
        if len(self._inline_stack) >= self._INLINE_CAP:
            raise ValueError(
                f"construct 内联深度超过上限 {self._INLINE_CAP}"
                f"(self.{name} @ {self.src_file}:{call.lineno}),fail-loud"
            )
        frame = self._frame_seq
        self._frame_seq += 1
        param_rename = self._bind_params(method, call, frame)
        locals_ = _assigned_names(method.body) - set(param_rename.keys())
        mapping = {loc: f"{loc}__i{frame}" for loc in locals_}
        mapping.update(param_rename)   # 形参绑定优先(复用调用方变量名)

        body_copy = [copy.deepcopy(s) for s in method.body]
        renamer = _Renamer(mapping)
        body_copy = [renamer.visit(s) for s in body_copy]

        self._inline_stack.append(name)
        ret = None
        try:
            self.walk_body(body_copy)
        except _ReturnSignal as sig:
            ret = sig.value
        finally:
            self._inline_stack.pop()
        self._bind_return(ret, target_names)

    def _bind_params(self, method: ast.FunctionDef, call: ast.Call, frame: int) -> dict:
        params = [a.arg for a in method.args.args if a.arg != "self"]
        pos = list(call.args)
        kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
        rename: dict = {}
        for i, p in enumerate(params):
            if i < len(pos):
                arg = pos[i]
            elif p in kw:
                arg = kw[p]
            else:
                arg = None   # 未传实参 → 用方法自带默认(此处按帧内合成局部处理)
            if isinstance(arg, ast.Name):
                rename[p] = arg.id            # 复用调用方变量名(共享 SSA / present / known_none)
            else:
                rename[p] = f"{p}__i{frame}"  # 非 Name 实参 / 缺省 → 帧内合成局部名(占位)
        return rename

    def _bind_return(self, ret_expr, target_names: list[str]) -> None:
        if ret_expr is None or not target_names:
            return
        elts = ret_expr.elts if isinstance(ret_expr, (ast.Tuple, ast.List)) else [ret_expr]
        for tgt, e in zip(target_names, elts):
            if isinstance(e, ast.Name):
                self._alias([tgt], e.id)
            # 返回项非 Name(字面量/调用等):该目标不登记 producer(下游若消费则占位 ref)

    # ---- 子 Cell 递归内联 ----
    def _inline_subcell(self, attrs: dict, call: ast.Call, target_names: list[str]) -> None:
        """把 resolver 递归抽出的子 DAG 内联到调用点:
          1) 子 construct 形参按位重映射到调用方实参(Name 实参→复用其 SSA ref 与 producer);
          2) 子节点 id 统一加偏移(接父 _next_id,不从 1 重启),子内部边同偏移平移;
          3) 子叶子消费的"形参操作数"→ 改写成调用方 ref,并补父 producer→子叶子的跨界边;
          4) 子返回值 → 绑回调用点赋值目标(下游消费即连"子输出→消费者"边)。"""
        sub = self._subcell_resolver(attrs.get("cell"), attrs.get("field"), attrs.get("bare", False))

        # 1) 形参 -> (调用方 ref, 调用方 producer 或 None)
        pos = list(call.args)
        kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
        param_map: dict[str, tuple[str, int | None]] = {}
        for i, p in enumerate(sub.param_names):
            if i < len(pos):
                arg = pos[i]
            elif p in kw:
                arg = kw[p]
            else:
                arg = None
            if isinstance(arg, ast.Name):
                ref = self.ssa.get(arg.id, f"{arg.id}:?:bf16")
                param_map[p] = (ref, self.producer.get(arg.id))
            else:
                param_map[p] = (f"{p}:?:bf16", None)  # 非 Name 实参 / 未传 → 占位,不连边

        # 2)+3) 偏移平移 + 形参操作数重映射 + 跨界边
        offset = self._next_id - 1
        for n in sub.nodes:
            new_id = n.id + offset
            new_ins: list[str] = []
            seen_prod: set[int] = set()
            for ref in n.ins:
                nm = ref.split(":")[0]
                if nm in param_map:
                    cref, cprod = param_map[nm]
                    new_ins.append(cref)
                    if cprod is not None and cprod not in seen_prod:
                        self.edges.append([cprod, new_id])
                        seen_prod.add(cprod)
                else:
                    new_ins.append(ref)   # 子内部 SSA 操作数:原样保留(边由下面的子内部边补)
            self.nodes.append(OpNode(
                id=new_id, op=n.op, src=n.src, module=n.module,
                ins=new_ins, out=n.out, attrs=dict(n.attrs),
            ))
        for s, d in sub.edges:
            self.edges.append([s + offset, d + offset])
        self._next_id = offset + len(sub.nodes) + 1
        # T0-6.5 Fix2:子 walker 的 opaque_calls 原样并入(src 已指子文件,无需重映射)——不这样做
        # 的话子 Cell 边界内的 fallthrough 调用会在父 DAG 视角下二次静默丢(违反 Fix2 初衷)。
        self.opaque_calls.extend(sub.opaque_calls)
        # Task 2:抽取诊断同理逐类并入(否则子 Cell 边界会把「子里丢了一大块」洗白)。
        for kind, items in (sub.diagnostics or {}).items():
            self.diagnostics.setdefault(kind, []).extend(items)

        # 4) 子返回值绑回调用点目标
        for tgt, r in zip(target_names, sub.returns):
            kind, val = r
            if kind == "node":
                pid = val + offset
                self.producer[tgt] = pid
                node = next((n for n in self.nodes if n.id == pid), None)
                dt = node.out.split(":")[2] if (node and node.out.count(":") == 2) else "bf16"
                self.ssa[tgt] = f"{tgt}:?:{dt}"
            elif kind == "param":
                cref, cprod = param_map.get(val, (f"{val}:?:bf16", None))
                self.ssa[tgt] = cref
                if cprod is not None:
                    self.producer[tgt] = cprod
            # kind == "none":该目标不登记(下游消费则占位 ref)

    def _resolve_returns(self, body) -> list:
        """定位 construct 顶层(含嵌套)最后一条 `return`,把返回值逐项分类:
           Name 且已被某节点产出 → ("node", producer id);Name 且是形参 → ("param", 名);其它 → ("none", None)。"""
        rets: list[ast.Return] = []
        for top in body:
            for n in ast.walk(top):
                if isinstance(n, ast.Return) and n.value is not None:
                    rets.append(n)
        if not rets:
            return []
        ret = max(rets, key=lambda r: getattr(r, "lineno", 0))
        val = ret.value
        elts = val.elts if isinstance(val, (ast.Tuple, ast.List)) else [val]
        out: list = []
        for e in elts:
            if isinstance(e, ast.Name) and e.id in self.producer:
                out.append(("node", self.producer[e.id]))
            elif isinstance(e, ast.Name) and e.id in self._param_set:
                out.append(("param", e.id))
            else:
                out.append(("none", None))
        return out

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
            # `if self.<flag>:` / `if self.config.<flag>:` → bool(config_flags[flag])
            nm = _config_flag_name(test)
            if nm is not None and nm in self.config_flags:
                return bool(self.config_flags[nm])
            return _UNDECIDED
        if isinstance(test, ast.Name):
            # `if <var>:` —— present → True;已知 None → False;有非 None 缺省 → 取其真值
            if test.id in self.present_vars:
                return True
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
        lk, lv = self._value(left)
        rk, rv = self._value(right)
        # `<x> is/is not <y>`(两侧都要能求成已知值——None 或 present 哨兵或 config 字面量)
        if isinstance(op, (ast.Is, ast.IsNot)):
            if lk and rk:
                res = (lv is rv)
                return res if isinstance(op, ast.Is) else (not res)
            return _UNDECIDED
        # `<x> ==/!= 字面量`
        if isinstance(op, (ast.Eq, ast.NotEq)):
            if not (lk and rk):
                return _UNDECIDED
            res = (lv == rv)
            return res if isinstance(op, ast.Eq) else (not res)
        # 数值序比较 `<x> >/>=/</<= <y>`
        if isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
            if not (lk and rk):
                return _UNDECIDED
            try:
                if isinstance(op, ast.Gt):
                    return lv > rv
                if isinstance(op, ast.GtE):
                    return lv >= rv
                if isinstance(op, ast.Lt):
                    return lv < rv
                return lv <= rv
            except TypeError:
                return _UNDECIDED
        # `<x> in (字面量...)` / `not in`
        if isinstance(op, (ast.In, ast.NotIn)):
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
        """把一个表达式求成"已知值":返回 (known: bool, value)。
        value 可为 config 字面量 / None / _PRESENT 哨兵。"""
        if isinstance(node, ast.Constant):
            return True, node.value
        nm = _config_flag_name(node)  # self.<flag> / self.config.<flag> → config_flags
        if nm is not None and nm in self.config_flags:
            return True, self.config_flags[nm]
        if isinstance(node, ast.Name):
            if node.id in self.present_vars:
                return True, _PRESENT
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
        """解析一次 cast 的目标 dtype:优先 idx 位置实参
        (`self.<dtype_flag>` → config_flags;或 ms.float32 字面量),其次 attrs['to_dtype'],
        最后保守 fp32(cast 多用于升精度)。"""
        if len(args) > idx:
            a = args[idx]
            nm = _config_flag_name(a)                 # self.compute_dtype → config_flags['compute_dtype']
            if nm is not None and nm in self.config_flags:
                v = self.config_flags[nm]
                if isinstance(v, str):
                    return _DTYPE_ALIAS.get(v, v)
            d = _dtype_name(a)
            if d:
                return d
        if attrs.get("to_dtype"):
            return attrs["to_dtype"]
        return "fp32"

    def _materialize_call_arg(self, call: ast.Call) -> ast.AST:
        """Hole1 修复:嵌套 Call 实参(如 `self.sum2(self.mul2(x, y))` 里的 `self.mul2(x, y)`)
        先递归发射(镜像 `_handle_chained_call` 的 `__chain__` 合成临时名模式),绑定到合成临时名,
        再把该临时名作为外层调用的 Name 操作数消费——取代此前 `_emit` 参数环里的
        `if not isinstance(a, ast.Name): continue`(静默丢整条嵌套算子,非仅丢一条边)。
        若嵌套调用本身落进 `_handle_call` 终端 fallthrough(opaque,如 `F.tuple_to_array(...)`)—
        不产节点、临时名未登记 SSA——按修复前语义原样返回该 Call(上层判 isinstance Name 会跳过,
        该实参不计入 ins;fallthrough 已由 Fix2 记入 opaque_calls,不再静默)。"""
        tmp = f"__arg__i{self._frame_seq}"
        self._frame_seq += 1
        before = len(self.nodes)
        self._handle_call(call, [tmp])
        if len(self.nodes) == before and tmp not in self.ssa:
            return call
        return ast.Name(id=tmp, ctx=ast.Load())

    # ---- 发射层:建 OpNode + 连数据流边 + 更新 SSA ----
    def _emit(self, op: str, attrs: dict, lineno: int, arg_exprs, target_names, out_dtype: str) -> OpNode:
        ins: list[str] = []
        seen_prod: set[int] = set()  # 同一节点内对同一 producer 只连一条边(去重)
        pending_prods: list[int] = []  # producer id(节点 id 尚未分配,先收集,分配后统一补边)
        for a in arg_exprs:
            if isinstance(a, ast.Call):
                # 嵌套 Call 实参:必须先于本节点分配 id(保证 id/nodes 列表顺序与真实数据流一致——
                # 下游 infer_shapes 等按 dag.nodes 顺序做正向传播,依赖 producer 先于 consumer 出现)。
                a = self._materialize_call_arg(a)
            if not isinstance(a, ast.Name):
                continue  # 字面量/属性/未产节点的 opaque 嵌套调用:非可追踪张量操作数,略过
            if a.id in self.ssa:  # 已知 SSA 中间变量 → 用其当前 ref 并向其 producer 连边
                ins.append(self.ssa[a.id])
                prod = self.producer.get(a.id)
                if prod is not None and prod not in seen_prod:
                    pending_prods.append(prod)
                    seen_prod.add(prod)
            else:  # 方法形参 / 未知名 → 占位 ref(shape 未知,dtype 缺省 bf16)
                ins.append(f"{a.id}:?:bf16")
                # Task 2(P0#1):形参占位是**合法**的(种子由 infer_shapes 喂);但「既非形参、也无
                # producer」的名意味着上游有东西被丢了(被丢的赋值 / opaque 调用的目标 / 内联未接上)。
                # 这是让所有上游静默丢弃**变得不可见**的那个汇聚点(评估文档 §「其它静默丢弃站点」),
                # 故单列记账。形参/帧内合成名(`__i<n>` 后缀)不记。
                if a.id not in self._param_set and "__i" not in a.id:
                    self._diag("unresolved_operands",
                               src=f"{self.src_file}:{lineno}", operand=a.id, node_op=op,
                               code=f"{op}(… {a.id} …)")

        # id 在实参(含嵌套 Call 已递归发射的子节点)处理完毕后才分配,保证 id 单调 = 数据流序。
        node_id = self._next_id
        self._next_id += 1
        for prod in pending_prods:
            self.edges.append([prod, node_id])

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
    present_vars: set | None = None,
    subcell_resolver=None,
    method_aliases: dict | None = None,
    strict: bool = False,
) -> OpDAG:
    """走查 `cls_name` 的 construct(),把每个 self.<name>(...) 调用落成 OpNode,返回 op-DAG。

    参数:
      src            — Cell 源码字符串(不执行,仅 ast 解析)
      cls_name       — 目标类名(construct 可继承自基类;内部方法按 MRO 内联)
      binds          — Pass B 产的 {self.<name> -> Binding(op, attrs)}
      src_file       — 源文件名,用于 OpNode.src=file:line 回指
      config_flags   — 可选。{"gated_linear_unit": True, "q_lora_rank": 1536, ...}:
                       求值 `if self.<flag>:` / `if self.config.<flag>:` 与各类字面量比较。
      none_vars      — 可选。已知为 None 的变量名集合(如 bias 关闭时的 bias_parallel)。
      param_defaults — 可选。construct 形参在"不传实参"时的缺省值(如 {"rotary_pos_cos": None})。
      present_vars   — 可选。已知"存在(非 None)"的变量名(如 MLA 恒传的 rotary_pos_emb):
                       让 `if v is not None:` 判 True、走 rope apply 等"输入存在"支。

    剪枝语义:config_flags/none_vars/param_defaults/present_vars **任一非 None 即开启剪枝**——
    此后每个 `if` 只走 config 命中支;**不可判定的 if → fail-loud**,唯一例外是"纯断言守卫"
    (条件不可判定、整支仅 raise、无 else)按对合法输入恒成立跳过。四者全为 None 时保持旧的
    "两支都线性走查"行为(Task 4 回归不受影响)。

    未绑定的 self.<name>:若是本类(含基类)内部方法则内联展开,否则 fail-loud。
    子 Cell(binding.op=="SubCell")且传入 subcell_resolver 时递归内联;否则退化为发射 SubCell 节点。
    找不到类或其 construct 方法时 fail-loud(ValueError)。

    **静默丢弃门(Task 2 / 评估文档 P0#1)**:
      * `strict=False`(默认)—— 一切看不懂的东西逐条记进 `dag.diagnostics`(见 schema),
        既有路径行为/字节逐字不变;
      * `strict=True` —— 任一诊断非空即 `ExtractionDroppedError`;
      * **0 节点硬门恒开(与 strict 无关)**:construct 有非平凡 body 却抽出 0 节点 →
        `ExtractionDroppedError`。`return x`/`pass` 这类恒等/空 Cell 的合法 0 节点不误伤。
    """
    return _run_walker(
        src, cls_name, binds, src_file,
        config_flags=config_flags, none_vars=none_vars,
        param_defaults=param_defaults, present_vars=present_vars,
        subcell_resolver=subcell_resolver, method_aliases=method_aliases,
        strict=strict,
    )[0]


def walk_construct_meta(
    src: str,
    cls_name: str,
    binds: dict,
    src_file: str,
    config_flags: dict | None = None,
    none_vars: set | None = None,
    param_defaults: dict | None = None,
    present_vars: set | None = None,
    subcell_resolver=None,
    method_aliases: dict | None = None,
    strict: bool = False,
):
    """同 walk_construct,但额外返回 (OpDAG, construct 形参名列表, 返回值分类)——供上层递归内联子 Cell。"""
    return _run_walker(
        src, cls_name, binds, src_file,
        config_flags=config_flags, none_vars=none_vars,
        param_defaults=param_defaults, present_vars=present_vars,
        subcell_resolver=subcell_resolver, method_aliases=method_aliases,
        strict=strict,
    )


def _body_is_trivial(body) -> bool:
    """construct body 是否**本就没有算子语义**(只 docstring / pass / 裸 return)。

    恒等 Cell(`Identity.construct: return x`)与空壳 Cell 的 0 节点是**合法**的,不得被 0 节点
    硬门误伤。除此之外的 body 若抽出 0 节点,一律视为「全丢了」。
    """
    for s in body:
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            continue                                  # docstring
        if isinstance(s, (ast.Pass, ast.Return)):
            continue                                  # `pass` / `return x`
        return False
    return True


def _run_walker(
    src, cls_name, binds, src_file, *,
    config_flags=None, none_vars=None, param_defaults=None,
    present_vars=None, subcell_resolver=None, method_aliases=None, strict=False,
):
    tree = ast.parse(src)
    if next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name), None) is None:
        raise ValueError(f"源码里找不到 class {cls_name}(fail-loud)")

    walker = _Walker(
        binds, src_file, config_flags, none_vars, param_defaults, present_vars,
        tree=tree, cls_name=cls_name, subcell_resolver=subcell_resolver,
        method_aliases=method_aliases, strict=strict,
    )
    construct = walker._lookup_method("construct")  # 支持 construct 定义在基类
    if construct is None:
        raise ValueError(f"class {cls_name}(及其基类)缺少 construct 方法(fail-loud)")

    walker.construct_params = [a.arg for a in construct.args.args if a.arg != "self"]
    walker._param_set = set(walker.construct_params)
    walker.walk_body(construct.body)
    walker.returns = walker._resolve_returns(construct.body)
    dag = OpDAG(cell=cls_name, nodes=walker.nodes, edges=walker.edges,
                scalar_binds=walker.scalar_binds, opaque_calls=walker.opaque_calls,
                diagnostics=walker.diagnostics)

    # ── 0 节点硬门(Task 2 / P0#1;**恒开**,与 strict 无关)────────────────────────────────
    # 实测反例:`CSAIndexer` fused 支整块在 `with _no_grad():`(indexer.py:214)里 → 整块丢 →
    # 此前返回 `ok, 0 nodes / 0 opaque`,若接进数字链路会**贡献 0 字节而不报错**(评估文档 §3.1
    # 「最危险的一条」)。有非平凡 body 却 0 节点 = 全丢了,绝不许看起来像成功。
    if not walker.nodes and not _body_is_trivial(construct.body):
        summary = diagnostics_summary(dag)
        detail = _format_diagnostics(dag, [k for k in DIAG_KINDS if summary[k]])
        opaque = "\n".join(f"    - {c.get('src','?')}: {c.get('expr','')}"
                           for c in walker.opaque_calls)
        raise ExtractionDroppedError(
            f"construct 走查产出 **0 个节点**,但 {cls_name}.construct 的 body 非平凡"
            f"（{src_file}）—— 整段被丢弃,这**不是**成功。\n"
            f"  诊断计数: " + ", ".join(f"{k}={summary[k]}" for k in DIAG_KINDS)
            + f", opaque_calls={summary['opaque_calls']}\n"
            + (detail + "\n" if detail else "")
            + (f"  [opaque_calls] {summary['opaque_calls']} 条:\n{opaque}\n" if opaque else "")
            + "  —— 请为上列语句/RHS 形态补处理器(评估文档 §10 P0#1/#4/#5)。"
        )
    if strict:
        walker._raise_diagnostics(cls_name)
    return dag, walker.construct_params, walker.returns
