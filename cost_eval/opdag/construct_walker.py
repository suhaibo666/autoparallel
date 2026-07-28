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
from . import primitives as prims
from .primitives import SHAPE_OF, UnknownPrimitiveError

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
    # detached —— 子里被 detach 的产物名(no-grad 块 / stop_gradient),逐名上浮到父
    #             (grad 可达性不能在子 Cell 边界丢)。
    # param_operands —— 子里出现的权重(Parameter)操作数,同理上浮。
    detached: list = field(default_factory=list)
    param_operands: list = field(default_factory=list)
    # ── G3 / G5(2026-07-25):子 Cell 的**符号环境**也必须过边界 ────────────────────────
    # dims_ctx —— 子类 `__init__` 求出的 `self.<attr>` → 符号维度。为什么不能在调用方"并表":
    #   `Compressor` 有两个构造点,`self.head_dim` 一处是 `v_head_dim`(csa.py:604)、
    #   一处是 `index_head_dim`(indexer.py:128)——**同名不同值**,扁平并表只能 fail-loud
    #   或静默取一个(`shape_infer.merge_dims_ctx` 的两难)。故按**内联帧**挂到节点上。
    # scalar_binds —— 子 construct 里的 `seqlen, bsz, _ = x.shape`(indexer.py:173)这类解包。
    #   walker 内部已把它们求成具体值,但此前不过边界 → 内联后子里的 `reshape(q, (seqlen, …))`
    #   整个解不出(实测 `CompressedSparseAttention` 的 `scalar_binds` 只剩 1 条)。
    #   同样按帧携带:子里的 `seqlen` 与父里可能同名而指不同张量。
    dims_ctx: dict = field(default_factory=dict)
    scalar_binds: list = field(default_factory=list)

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
# **布局重包的自由调用**:点号路径 → 承载张量的实参下标。语义与 `PASSTHRU_METHODS` 完全同一条
# (数学恒等、不复制、不新增激活),只是写成了自由函数形式。
#   `DTensor.from_local(experts_output, mesh, placements)` @ `moe/experts.py:211`
#   —— 是 `.to_local()`(`primitives.PASSTHRU_METHODS` 已收)的逆操作。
_PASSTHRU_FREE_CALLS = {"DTensor.from_local": 0}

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
# 表已合并到 `primitives.VIEW_METHODS`(单点维护),此处保留旧名做别名以免动既有引用。
_VIEW_METHODS = prims.VIEW_METHODS

# ── `with` 上下文管理器的语义分类(路线 B P0#4,2026-07-25)────────────────────────────────
# `walk_stmt` 此前对 `ast.With` 只记账不走查(`7b9aa86` 的理由是对的:走进去发普通节点会造出
# 一批**看起来梯度可达**的假节点,比丢更危险)。正确做法是**走进去 + 带上块级语义**:
#   * NO_GRAD    —— `with _no_grad():` 整块不建 autograd 图 → 块内产物**全部标 detached**
#                    (`indexer.py:214`,区域 `:214-232`,块内绑定
#                     q/k/weights/key_length/cmp_residual_k/topk_indices/index_scores);
#   * TRANSPARENT —— 只切换**派发/布局模式**,数学恒等、不改梯度可达性 → 照常走查
#                    (`SkipDTensorDispatch`:hyper_parallel 的 DTensor 派发旁路,
#                      `multi_latent_attention.py:244/288`、`flash_attention.py:311`);
# 其余一切 `with` —— **语义未知即 fail-loud**(任务纪律:不许猜)。
_WITH_NO_GRAD = frozenset({"_no_grad", "no_grad", "NoGrad", "_NoGrad", "stop_gradient_region"})
_WITH_TRANSPARENT = frozenset({"SkipDTensorDispatch"})

# **host 侧元数据属性**:读它们得到的是布局/分片描述对象或 python 数,**不是张量**。
#   layout / placements / device_mesh / mesh / alias_placements —— hyper_parallel DTensor 的
#   分片元数据(`moe/experts.py:186,211-212`);ndim / size —— 形状元数据。
# 不登记的话 `tokens_layout = tokens.layout` 会落 `dropped_assigns`,而下游
# `if tokens_layout is not None:`(`experts.py:210`)随即不可判定 → 整个 GroupedMLP fail-loud。
_HOST_META_ATTRS = frozenset({
    "layout", "placements", "alias_placements", "device_mesh", "mesh", "mesh_dim_names",
})

# host 侧内建:产出是 python 数,不是张量。
# host 侧内建:产出是 python 数 / **host 侧容器或索引描述子**,一律不是张量。
#   `slice(0, dim_size - shifts)` / `tuple(slices)` / `list(shape)` ——
#   `roll_tensor`(`pynative/transformers/multi_token_prediction.py:181-193`)用它们拼切片下标。
_HOST_BUILTINS = frozenset({"int", "float", "bool", "str", "len", "min", "max", "sum",
                            "abs", "round", "range", "slice", "tuple", "list"})

# host 侧字面量构造的小常量张量:`Tensor([...], dtype=...)`(indexer.py:219 / csa.py:118)。
_CONST_CTORS = frozenset({"Tensor"})

# 三态哨兵:一个 `if`/三元条件在剪枝上下文下无法由已知 config 判定。
_UNDECIDED = object()
# "已知存在(非 None)"的取值哨兵(用于 present_vars 走 `if v is not None:` 真支)。
_PRESENT = object()
# 标量环境里的"是标量但值未知"哨兵(如 `sk = kv_full.shape[0]`,轴长符号未种子化)。
# **不可**参与数值比较(`_eval_test` 遇它返回 _UNDECIDED),只用来判"这不是张量"。
_UNKNOWN_SCALAR = object()


class _PositiveDim:
    """**公理**哨兵:一个「张量某轴的长度」——值未知,但**必然 >= 1**。

    这不是"猜一个尺寸",而是张量语义本身给的事实:一个参与计算的张量的任一轴长至少是 1。
    用途只有一处:让 `if n_compressed > 0:`(csa.py:762,`n_compressed = int(compressed_kv.shape[0])`)
    这类**只问轴长是否非空**的条件可判定,而 `if sq < ratio:`(compressor.py:190,要真值)
    仍然 `_UNDECIDED` → fail-loud。判据实现见 `_Walker._cmp_positive_dim`。
    """

    def __repr__(self):
        return "<dim>=1>"


_POSITIVE_DIM = _PositiveDim()


class _Kind:
    """一个表达式在 walker 眼里的种类。**张量与标量必须分开** —— 这是"不造假节点"的关键:
    `ori_dtype = x.dtype`(dtype 记号)、`head_dim = query.shape[-1]`(轴长标量)、
    `d = self.query_projection_size // o_groups`(config 算术)都**字节中性**,建节点就是造假;
    而 `q_hnorm_fp32 = q * rsqrt(...)`、`score_f32 = score.astype(fp32) + ape`
    是**真张量**,不建节点就是漏字节。"""
    TENSOR = "tensor"
    SCALAR = "scalar"      # python 数 / 轴长 / config 值(含 _UNKNOWN_SCALAR)
    DTYPE = "dtype"        # `x.dtype` 这类 dtype 记号
    NONE = "none"          # 已知 None
    PARAM = "param"        # `self.<attr>`,且 __init__ 里是 `Parameter(...)` —— 权重,不是激活
    UNKNOWN = "unknown"    # 判不出来 —— 调用方按"记账 + 不建节点"处理

# mindspore dtype 名 → 本库短标签(best-effort;未知名原样透传)。
_DTYPE_ALIAS = {
    "float32": "fp32", "float": "fp32",
    "float16": "fp16", "half": "fp16",
    "bfloat16": "bf16", "bfloat": "bf16",
    "float64": "fp64", "double": "fp64",
    "int32": "int32", "int64": "int64", "int16": "int16", "int8": "int8",
    "uint8": "uint8", "bool_": "bool", "bool": "bool",
}

# 合法 dtype 短标签的全集(`_resolve_cast_dtype` 的"假 dtype"门用它判)。
_KNOWN_DTYPES = frozenset(_DTYPE_ALIAS.values())


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
        class_index=None,
        cls_rel: str | None = None,
        module_funcs: dict | None = None,
        runtime_predicates: dict | None = None,
        fn_classes: dict | None = None,
        self_kinds: dict | None = None,
        param_literals: dict | None = None,
        alias_unknown: dict | None = None,
        host_call_allow: tuple = (),
        kernel_call_allow: tuple = (),
        kernel_saves: dict | None = None,
        param_cells: dict | None = None,
        input_axes: dict | None = None,
        module_consts: dict | None = None,
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
        # ── 跨文件 MRO(P0#3):class_index 能顺 import 解析基类;cls_rel = 定义 cls_name 的文件 ──
        self._class_index = class_index
        self._cls_rel = cls_rel
        # 模块级自由函数(**同文件**)→ 可内联:`unfused_compressed_sparse_attn`(csa.py:464)、
        # `parse_cu_seqlens`(:322)、`get_window_topk_idxs`(:449)、`get_compress_topk_idxs`(:430)…
        # 值可为 `FunctionDef` 或 `(FunctionDef, 定义它的文件 rel)`——后者用于**跨文件**的
        # 模块级自由函数(`compute_routing_scores_for_aux_loss` 定义在
        # `moe/moe_utils.py:343`,被 `moe/router.py:466` 调用)。带 rel 才能让内联出的节点
        # `file:line` 指向真正定义它的文件。
        self._module_funcs: dict = {
            k: (v if isinstance(v, tuple) else (v, None))
            for k, v in (module_funcs or {}).items()}
        # `hasattr(x,"to_local")` / `isinstance(x, DTensor)` 这类**部署形态**谓词:必须由调用方
        # 显式给值(键形如 `hasattr:to_local` / `isinstance:DTensor`),缺键 → fail-loud,不猜。
        self.runtime_predicates: dict = dict(runtime_predicates or {})
        # `_Function` 子类信息:{类名: {"saves": [名单], "forward_params": [...], "bare_ctx": [...]}}
        # 供 `<Cls>.apply(...)` 发射 FusedFunction 节点时把**源真值 saved 集**逐字挂上(复用 fn_saves)。
        self._fn_classes: dict = dict(fn_classes or {})
        # `self.<attr>` 的种类(由 init_dims 静态求值 __init__ 得):"param"/"scalar"/"module"。
        # 这是"`self.attn_sink` 是权重 / `self.softmax_scale` 是标量"的**源侧判据**,不是猜。
        self._self_kinds: dict = dict(self_kinds or {})
        # construct 形参的**字面量缺省**(如 `rope_pos_offset: int = 0`)→ 标量环境种子。
        self._param_literals: dict = dict(param_literals or {})
        # __init__ 里裸别名**查表失败**的条目(attr -> {alias, src}):调用到它才 fail-loud,
        # 且报错要指名道姓说"加哪条表项"(未被调用的未知别名只记诊断)。
        self._alias_unknown: dict = dict(alias_unknown or {})
        # 显式白名单:纯宿主副作用调用(如 `save_to_indexer_losses_tracker(...)` 记 loss 到
        # 模块级 dict,utils.py:41)—— 记 opaque、不记 unregistered(它没有被消费的返回值)。
        self._host_call_allow: tuple = tuple(host_call_allow)
        # 融合 NPU 内核的自由函数名(如 `npu_lightning_indexer`,indexer.py:220):发射 `Kernel`
        # 节点(**保边、登记目标**),但它的 saved 集**不由 PIN 猜** —— 只有能证明"该调用在
        # no-grad 区里因而没有反向"时才写 saves=[];否则留空让 `derive_saves` fail-loud。
        self._kernel_call_allow: tuple = tuple(kernel_call_allow)
        # 融合内核的 saved 集**调用方显式声明**表(2026-07-25):
        #   `{内核名: {"saved_ins_idx": [...] | "all", "source": 定位符, "reason": 理由}}`
        # 存在的唯一理由:有些内核的 bprop **不在权威快照里**——`npu_mhc_pre_sinkhorn` /
        # `npu_mhc_post` 来自 `hyper_parallel.custom_ops.experimental`
        # (`hyper_connection.py:20-23`,try/except ImportError 的外部包)。此时
        #   ①  walker **绝不猜**(未声明 → 不写 `saved_ins_idx` → `derive_saves` fail-loud);
        #   ②  调用方要出数就必须把「据什么、为什么」写进声明里,并**留在节点 attrs 上可评审**
        #       (`saved_source` / `saved_reason` / `saved_declared_by_caller`)。
        # 这与 `FusedFunction`(saved 集能从 `ctx.save_for_backward` 逐字读)是**两种**来源,
        # 故在节点上分开标记,不许把「调用方声明」洗成「源真值」。
        self._kernel_saves: dict = dict(kernel_saves or {})
        # **construct 形参持有的子 Cell**:`{形参名: 类名}`。真源必需:
        #   `MultiTokenPredictionLayer.construct(..., embedding=None, ...)` 里
        #   `decoder_input = embedding(input_ids=..., position_ids=...)`
        #   (`pynative/transformers/multi_token_prediction.py:447`)—— 这个 `embedding`
        #   是 `gpt_model.py:340` 传进来的 `self.embedding`(= `LanguageModelEmbedding`,
        #   `gpt_model.py:183`)。MTP 层**真的会**再跑一遍 embedding(输入是 roll 过的 input_ids),
        #   故它的激活必须在图里。调用方显式声明类名(顺源里的实参链读出),walker 才递归;
        #   不声明就照旧落 `opaque_calls` + `unregistered_targets`(不猜)。
        self._param_cells: dict = dict(param_cells or {})
        # 入口方法**形参的轴符号种子**:{形参名: (轴0符号, 轴1符号, ...)},符号可是 int 或
        # config_flags 里的键名。与 `infer_shapes(dag, input_shapes)` 同一套契约 —— 调用方本来
        # 就得给这份种子,这里只是让 `sq, b, _ = x.shape` 之后的 `if sq < ratio` 可判定。
        # **不给就是 `_UNKNOWN_SCALAR` → 条件不可判定 → fail-loud(绝不杜撰尺寸)**。
        self._input_axes: dict = dict(input_axes or {})
        # 模块级常量(`_BF16_MIN = -3.3895...e38`,indexer.py:47;`eps` 之类)—— host 侧标量,
        # 不是张量。不登记的话它们会以"未知名"进 ins 变成假操作数。
        self._module_consts: dict = dict(module_consts or {})

        # ── 标量 / dtype 环境(P0#5 的另一半:把"标量记账"与"张量建节点"分开)────────────
        self.scalars: dict[str, object] = {**dict(module_consts or {}),
                                          **dict(self._param_literals)}
        self.dtypes: dict[str, str] = {}     # 变量名 -> dtype 短标签(`ori_dtype = x.dtype`)
        # ── 块级 detach(P0#4)/ 逐点 detach(P1#11)────────────────────────────────────
        self._nograd_depth = 0
        self.detached: list[str] = []        # 被 detach 的张量名(源序、去重)
        # `del x` 的显式释放点(无 op 语义,但对 liveness 有意义)——记录而非当"丢弃"。
        self.deletes: list = []
        # `self.<attr>` 权重操作数(Parameter):不进 `ins`(否则权重被当激活 save 计),
        # 单列 attrs["param_operands"] + dag.param_operands,使其**可见**而非静默丢。
        self.param_operands: list = []
        # **权重别名**:`attn_sink = self.attn_sink`(csa.py:683)这类局部名其实指向一个
        # `Parameter`。它们**不进 SSA**(否则该权重会以张量操作数身份进 `ins`,被
        # `derive_saves` 当激活 save 计——实测 FFNGroupedGEMM「236 MiB」里 88 MiB 就是这个病),
        # 而是记在这里,消费点统一路由到 `param_operands`。
        # 契约(并行 agent 的 W2/W3):**权重永不出现在 saves、永不是任何节点的 out**。
        self.param_aliases: set[str] = set()
        self._param_of: dict[str, str] = {}     # 权重别名局部名 -> `self.<attr>` 名
        # **权重派生量**:某个节点的操作数**全是权重**(`ins` 为空 + 本行有 param_operands),
        # 则它的产出仍是"权重派生"而不是激活。实测两处:
        #   `w1 = self.cast_op(self.weight1, self.compute_dtype)`
        #   (`parallel_core/training_graph/transformer/moe/ffn.py:146`)→ `w1` 随后进
        #   `GroupedMatmul` 的 ins,被 `PIN["GroupedMatMul"]={"inputs":"all"}` **当激活 save 计**
        #   —— 这正是评估文档 §7.2 实测的「FFNGroupedGEMM 236 MiB 里 88 MiB 是权重」;
        #   `weight = self.transpose(weight, 1, 0)`(`pynative/layers/linear.py:132`,lm_head 的
        #   vocab 投影)同理。
        # 处置(W2/W3/W4 的"永不进 saves"那一半):**保留在 `ins` 里**(下游 `shape_infer`
        # 要靠权重末轴推 matmul 输出维),但在节点上标 `attrs["weight_ins_idx"]`,
        # 由 `derive_saves` 排除 —— 权重不是激活。
        self._weight_derived: set[str] = set()
        # `_eval_predicate_call` 最近一次缺键的谓词(用于把 fail-loud 报错写成"加哪个键")。
        self._pending_predicate = None
        # **python 级方法派发表**(2026-07-25):`{局部名: {键: 方法名}}` 与
        # `{局部名: 方法名}`。真源必需:`router.py:479-483` 建
        #   `aux_loss_func_map = {"aux_loss": self._apply_aux_loss,
        #                         "seq_aux_loss": self._apply_seq_aux_loss, ...}`
        # 再 `aux_loss_func = aux_loss_func_map.get(self.aux_loss_type)`(:485)、
        # `top_scores = aux_loss_func(...)`(:489)。`aux_loss_type` 是 config
        # (yaml `moe_router_load_balancing_type: seq_aux_loss`)→ **静态可解**;
        # 不解就是把整段 aux-loss 计算(真张量)丢掉。
        self._method_tables: dict = {}
        self._method_refs: dict = {}
        # 顶层 `return` 命中标志(剪枝上下文下用于中止后续语句走查)+ 实际命中的返回表达式。
        self._stop = False
        self._taken_return = None

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
            if self._stop:
                # 顶层 `return` 已命中 → 后续语句在运行期**不可达**,继续走就是把互斥支
                # 一起算进来(见本类 `_handle_return` 的注释)。仅剪枝上下文下生效。
                break
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
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            self._handle_with(stmt)
        elif isinstance(stmt, ast.Delete):
            self._handle_delete(stmt)
        elif isinstance(stmt, ast.AugAssign):
            self._handle_augassign(stmt)
        elif isinstance(stmt, ast.For):
            self._handle_for(stmt)
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

    # ---- with 块:走查块体 + 带上块级语义(P0#4)----
    @staticmethod
    def _ctx_head(expr) -> str:
        """取上下文管理器的"名字":`_no_grad()` → `_no_grad`;`ms._no_grad()` → `_no_grad`;
        `SkipDTensorDispatch()` → `SkipDTensorDispatch`;裸名同理。"""
        node = expr.func if isinstance(expr, ast.Call) else expr
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        return ""

    def _handle_with(self, stmt) -> None:
        """`with <ctx>:` —— 按 `_WITH_NO_GRAD` / `_WITH_TRANSPARENT` 分类走查;未知语义 fail-loud。

        为什么不能"走进去就当普通语句":`_no_grad()` 块内的产物**没有 autograd 节点**,当普通节点
        发射会造出一批"看起来梯度可达"的张量 → 反向会去 save 它们 → **字节多算**。所以走查的同时
        必须把块内产物标 `detached`(见 `_emit`)。这正是 `7b9aa86` 刻意不走查的那个顾虑的正解。
        """
        heads = [self._ctx_head(it.context_expr) for it in stmt.items]
        if any(h in _WITH_NO_GRAD for h in heads):
            self._nograd_depth += 1
            try:
                self.walk_body(stmt.body)
            finally:
                self._nograd_depth -= 1
            return
        if heads and all(h in _WITH_TRANSPARENT for h in heads):
            # 只切换派发/布局模式,数学恒等、梯度可达性不变 → 照常走查,不打任何标。
            self.walk_body(stmt.body)
            return
        raise ValueError(
            f"construct 里的 `with` 上下文语义未知（{self.src_file}:{stmt.lineno}）:"
            f"`{self._short_code(self._describe(stmt))}` —— 上下文管理器 {heads!r} 既不在"
            f"`_WITH_NO_GRAD`(整块 detach)也不在 `_WITH_TRANSPARENT`(数学恒等)里。"
            f"**拒绝猜它改不改梯度可达性**:猜错就是一整块张量的 saved 语义错。"
            f"请在 construct_walker 的两张表里显式登记它。"
        )

    # ---- for 循环:**只**展开静态可数的 `range(...)`,其余照旧记诊断 ----
    _FOR_UNROLL_CAP = 256      # 展开上限(防一个笔误的巨大 range 把图炸掉)

    def _handle_for(self, stmt: ast.For) -> None:
        """`for <t> in range(<静态可求值>):` —— **按真实迭代次数逐轮展开**;其余形态记诊断。

        为什么必须展开而不是"走一遍循环体":
          `SinkhornKnopp.construct` 的 `for _ in range(self.iterations - 1)`
          (`hyper_connection.py:63`,`iterations = config.mhc_sinkhorn_iterations`,
           yaml `hc_sinkhorn_iters: 20`)每轮产 4 个 `[s,b,n,n]` fp32 中间量;走一遍
          = 少算 18 轮。走 0 遍(此前:整条 `For` 静默丢)= 少算全部 19 轮。
        为什么只认 `range(<静态>)`:
          迭代次数必须是**已知事实**(来自 config/字面量),否则展开几轮就是编造。
          `for x in <张量>` / `enumerate(...)` / 次数不可判定的 `range` → 一律记诊断
          (strict 下抛),不猜。
        """
        vals = self._for_iter_values(stmt)
        n = len(vals) if vals is not None else None
        if n is None:
            self._diag_drop_stmt(stmt)
            return
        if n > self._FOR_UNROLL_CAP:
            raise ValueError(
                f"`for ... in range({n})` 展开数超上限 {self._FOR_UNROLL_CAP}"
                f"（{self.src_file}:{stmt.lineno}）—— 拒绝展开(可能是种子/配置注错),fail-loud"
            )
        if stmt.orelse:
            # `for ... else:` —— 语义是"未 break 时执行";本库源里零出现,不猜。
            self._diag_drop_stmt(stmt)
            return
        tgt = stmt.target
        for k in range(n):
            if isinstance(tgt, ast.Name) and tgt.id != "_":
                # 循环变量是**host 侧值**(chunk 下标 / 通信组句柄),按标量登记 —— 绝不当张量。
                self._forget(tgt.id)
                self.scalars[tgt.id] = vals[k]
            self.walk_body(stmt.body)

    def _for_iter_values(self, stmt: ast.For):
        """`for <t> in <可静态确定的可迭代>` 的元素表;确定不了 → None。

        两种形态(都**由调用方给的值**决定,不是猜):
          ① `range(<静态整数>)` —— 见 `_handle_for`;
          ② `self.<flag>` / `self.config.<flag>` 且该 flag 的值是 tuple/list ——
             真源必需:`for group in self._cp_groups:`(`moe/router.py:497`)。
             yaml `context_parallel: 1` ⇒ `_cp_groups = ()`(`router.py:170` 的初值,
             `enable_sequence_parallel` 未被调用/传空)⇒ **0 轮**,循环体不存在。
             调用方显式给 `_cp_groups: ()` 才判定;不给 → 记诊断(strict 下抛)。
        """
        it = stmt.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
            return self._for_range_values(stmt)
        nm = _config_flag_name(it)
        if nm is not None and nm in self.config_flags:
            v = self.config_flags[nm]
            if isinstance(v, (tuple, list)):
                return list(v)
        return None

    def _for_range_values(self, stmt: ast.For):
        """`range(...)` 的元素表;实参有一个求不出整数 → None(**不是** 0 轮)。"""
        call = stmt.iter
        if not (1 <= len(call.args) <= 3):
            return None
        args = [self._scalar_of(a) for a in call.args]
        if any(v is _UNKNOWN_SCALAR or v is _POSITIVE_DIM
               or not isinstance(v, int) or isinstance(v, bool) for v in args):
            return None
        if len(args) == 1:
            return list(range(args[0]))
        if len(args) == 2:
            return list(range(args[0], args[1]))
        return list(range(args[0], args[1], args[2]))

    def _handle_delete(self, stmt: ast.Delete) -> None:
        """`del a, b`(如 `indexer.py:211`):无 op 语义,但是 **liveness 的显式释放点**。
        从 SSA/标量环境里摘掉这些名字并记进 `dag.deletes`(不记诊断——它不是"看不懂")。"""
        for t in stmt.targets:
            if not isinstance(t, ast.Name):
                continue
            self.deletes.append({"src": f"{self.src_file}:{stmt.lineno}", "name": t.id})
            self.ssa.pop(t.id, None)
            self.producer.pop(t.id, None)
            self.scalars.pop(t.id, None)

    def _handle_augassign(self, stmt: ast.AugAssign) -> None:
        """`x += <expr>` / `x *= 2`:等价于 `x = x <op> <expr>`,复用 BinOp 通路
        (张量 → 建节点;标量 → 更新标量环境)。非 Name 目标(下标/属性)仍记账。"""
        if not isinstance(stmt.target, ast.Name):
            self._diag_drop_stmt(stmt)
            return
        binop = ast.BinOp(left=ast.Name(id=stmt.target.id, ctx=ast.Load()),
                          op=stmt.op, right=stmt.value)
        ast.copy_location(binop, stmt)
        ast.fix_missing_locations(binop)
        self._handle_binop(binop, [stmt.target.id], stmt.lineno)

    # ---- if 剪枝:求值条件,只走命中支 ----
    def _handle_if_pruned(self, stmt: ast.If) -> None:
        self._pending_predicate = None
        r = self._eval_test(stmt.test)
        if r is _UNDECIDED:
            if self._pending_predicate:
                raise ValueError(
                    f"construct 的 if 条件是**部署形态谓词**且未给值"
                    f"（{self.src_file}:{stmt.lineno}）:`{self._describe(stmt.test)}` —— "
                    f"请在 `runtime_predicates` 里显式给 `{self._pending_predicate}`(True/False)。"
                    f"绝不默认取某一支(两支常常字节等价,但「常常」不是「总是」)。"
                )
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
        if isinstance(val, (ast.Call, ast.Subscript)) or isinstance(
                val, (ast.BinOp, ast.Compare, ast.BoolOp)):
            mat = self._materialize_expr(val, stmt.lineno)
            if mat is not None:
                val = mat
        # 返回 Name/Tuple/其它:顶层无需产 op(下游没有消费者)
        if self._pruning:
            # 剪枝上下文:命中的 return 就是**真正的出口** —— 中止后续语句(否则互斥的
            # 分派支会被同时走查)。同时记下它,`_resolve_returns` 优先用它而不是"最后一条 return"。
            self._stop = True
            self._taken_return = val

    def _materialize_return_call(self, val):
        """内联方法 `return <表达式>`:把它发射到合成临时名并返回该 Name(供别名到调用点目标)。

        2026-07-25:除 `Call` / `Subscript(Call)` 外,**`BinOp` / `UnaryOp` / `Compare` /
        `BoolOp` / `Subscript` 也必须物化**。真源反例:
          `return mint.sum(agg_probs * tokens_per_expert) * (...)`(`moe/router.py:549-553`,
          `_aux_loss_from_stats`)—— 返回一个 BinOp。此前 `_bind_return` 只认 Name/Constant,
          于是这条 return 的值**整个丢掉**,调用点 `aux_loss = self._aux_loss_from_stats(...) / bsz`
          (`:613-615`)的外层节点变成 `ins=[]`(实测第一版 dump 里就是),即断链。
        """
        call = None
        if isinstance(val, ast.Call):
            call = val
        elif isinstance(val, ast.Subscript) and isinstance(val.value, ast.Call):
            call = val.value
        if call is None:
            if isinstance(val, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
                                ast.Subscript)):
                mat = self._materialize_expr(val, getattr(val, "lineno", 0))
                return mat if mat is not None else val
            return val
        tmp = f"__ret__i{self._frame_seq}"
        self._frame_seq += 1
        self._handle_call(call, target_names=[tmp])
        return ast.Name(id=tmp, ctx=ast.Load())

    def _handle_assign(self, stmt: ast.Assign) -> None:
        val = stmt.value
        targets = _target_names(stmt.targets)
        if targets and all(t == "_" for t in targets):
            # `_ = <...>` —— 显式丢弃(源里用来消掉 lint 告警,如
            # `_ = rotary_pos_emb, attention_mask, mscale, rotary_cos_sin`,
            # deepseek_v4_hybrid_attention.py:226)。`_` 永不被读 → 无 op 语义,**不记诊断**。
            return
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
            self._bind_shape_unpack(list(targets), val.value.id)
        elif isinstance(val, (ast.BinOp, ast.UnaryOp)):
            self._handle_binop(val, targets, stmt.lineno)
        elif isinstance(val, (ast.Compare, ast.BoolOp)):
            self._handle_compare(val, targets, stmt.lineno)
        elif isinstance(val, ast.Subscript):
            self._handle_subscript_assign(val, targets, stmt.lineno)
        elif isinstance(val, ast.Name):
            self._bind_name_rhs(val.id, targets)
        elif isinstance(val, ast.Attribute):
            self._handle_attribute_assign(val, targets, stmt)
        elif isinstance(val, ast.Constant):
            for t in targets:
                self._forget(t)
                if val.value is None:
                    self.known_none.add(t)
                else:
                    self.scalars[t] = val.value
        elif isinstance(val, (ast.Tuple, ast.List)) and len(targets) == len(val.elts):
            for t, e in zip(targets, val.elts):
                self._bind_expr_to_target(e, t, stmt.lineno)
        elif isinstance(val, ast.Dict) and self._method_table_of(val) is not None:
            # `{"seq_aux_loss": self._apply_seq_aux_loss, ...}`(router.py:479)——
            # **python 级方法派发表**,不是张量。登记表,不建节点、不记诊断。
            table = self._method_table_of(val)
            for t in targets:
                self._forget(t)
                self._method_tables[t] = table
        else:
            # 仍不支持的 RHS 形态(Dict/Set/Lambda/ListComp/Starred/JoinedStr/…):
            # 记两笔——①RHS 形态本身;②目标名从未进 SSA(下游消费它会拿占位 ref、**丢边**)。
            self._diag("dropped_assigns", src=f"{self.src_file}:{stmt.lineno}",
                       targets=list(targets), rhs=type(val).__name__,
                       code=self._describe(stmt))
            self._diag_unregistered(targets, stmt.lineno, f"dropped_assign_rhs_{type(val).__name__}")

    def _method_table_of(self, d: ast.Dict):
        """`{字面量键: self.<method>, ...}` → `{键: 方法名}`;不是这个形态返回 None。

        只认**全部**值都是本类可解析方法(`_lookup_method` 找得到)的字典 —— 有一个不是就
        返回 None(退回既有的 `dropped_assigns` 记账),绝不半解。
        """
        out = {}
        for k, v in zip(d.keys, d.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, (str, int))):
                return None
            m = _self_attr(v)
            if m is None or self._lookup_method(m)[0] is None:
                return None
            out[k.value] = m
        return out or None

    # ---- 标量 / dtype / 张量 三分(P0#5 的判据层)------------------------------------
    def _forget(self, name: str) -> None:
        """重新绑定一个名字前,清掉它在各环境里的旧身份(SSA/标量/dtype/None/权重别名)。"""
        self.param_aliases.discard(name)
        self._weight_derived.discard(name)
        self._method_tables.pop(name, None)
        self._method_refs.pop(name, None)
        self.ssa.pop(name, None)
        self.producer.pop(name, None)
        self.scalars.pop(name, None)
        self.dtypes.pop(name, None)
        self.known_none.discard(name)
        self.present_vars.discard(name)

    def _bind_shape_unpack(self, names: list[str], src_name: str) -> None:
        """`sq, b, _ = x.shape` —— **不产 op**(产出的是 python int 元组)。
        既记进 `scalar_binds`(shape 推断按 x 的已知 shape 逐轴回填),又把每个名字登记成
        **标量**(值未知),使下游 `if sq < ratio` 这类判定至少知道"它不是张量"。"""
        self.scalar_binds.append({"names": list(names), "src": src_name})
        for i, n in enumerate(names):
            if n == "_":
                continue
            self._forget(n)
            self.scalars[n] = self._axis_scalar(src_name, i)

    def _axes_tuple(self, src_name: str):
        """`<x>.shape` 的**逐轴值元组**(有 `input_axes` 种子时解成数,无种子的轴给公理哨兵);
        完全没有该形参的种子 → `_UNKNOWN_SCALAR`。

        真源必需:`shape = tensor.shape` 后 `dim_size = shape[dims]`
        (`multi_token_prediction.py:173-174` 的 `roll_tensor`)—— 只把 `shape` 记成
        「某个不知道的标量」的话,`if abs(shifts) >= dim_size:`(`:178`)永远判不出。
        """
        axes = self._input_axes.get(src_name)
        if not axes:
            return _UNKNOWN_SCALAR
        return tuple(self._axis_scalar(src_name, i) for i in range(len(axes)))

    def _axis_scalar(self, src_name: str, axis: int):
        """一个张量轴长的标量值:**只在调用方给了该形参的轴种子时**才求出,否则 `_UNKNOWN_SCALAR`。

        种子形态 `input_axes={"x": ("seq_length", "b", "hidden_size")}`(与 `infer_shapes` 同契约)。
        符号名再经 `config_flags` 解成数(`seq_length` → 4096)。**绝不杜撰尺寸**:没种子就是
        未知 → 依赖它的 `if` 不可判定 → fail-loud。
        """
        axes = self._input_axes.get(src_name)
        tok = axes[axis] if (axes and axis < len(axes)) else None
        if isinstance(tok, int):
            return tok
        if isinstance(tok, str):
            v = self.config_flags.get(tok)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return v
        return _POSITIVE_DIM        # 无种子:值未知,但**必然 >= 1**(见 _PositiveDim)

    def _classify(self, node) -> str:
        """判一个表达式是 张量 / 标量 / dtype 记号 / None / 权重 / 判不出。**只按已知事实**。"""
        if isinstance(node, ast.Constant):
            return _Kind.NONE if node.value is None else _Kind.SCALAR
        if isinstance(node, ast.Name):
            if node.id in self.param_aliases:
                return _Kind.PARAM          # 权重别名(见 param_aliases 的契约)
            if node.id in self.ssa or node.id in self.producer:
                return _Kind.TENSOR
            if node.id in self.dtypes:
                return _Kind.DTYPE
            if node.id in self.scalars:
                return _Kind.SCALAR
            if node.id in self.known_none:
                return _Kind.NONE
            if node.id in self._param_set:
                return _Kind.TENSOR          # construct 形参:默认是张量(种子由 infer_shapes 喂)
            if "__i" in node.id:
                return _Kind.UNKNOWN         # 帧内合成占位名(实参非 Name / 未传)
            return _Kind.UNKNOWN
        if isinstance(node, ast.Attribute):
            if node.attr == "dtype":
                return _Kind.DTYPE
            if node.attr == "shape":
                return _Kind.SCALAR          # shape 元组(host 侧)
            if node.attr == "ndim":
                # 轴数:host 侧整数(定义如此)。`[slice(None)] * tensor.ndim`
                # (`multi_token_prediction.py:184`)判不出它是标量,整条 RHS 会落 dropped_assigns。
                return _Kind.SCALAR
            if _config_flag_name(node) is not None and isinstance(node.value, ast.Attribute):
                return _Kind.SCALAR          # `self.config.<x>` —— config 对象里没有张量
            sattr = _self_attr(node)
            if sattr is not None:
                kind = self._self_kinds.get(sattr)
                if kind == "param":
                    return _Kind.PARAM
                if kind == "scalar":
                    return _Kind.SCALAR
                if sattr in self.config_flags:
                    return _Kind.SCALAR
                return _Kind.UNKNOWN
            # `ms.float32` / `mstype.int32` 这类 dtype 字面量
            return _Kind.DTYPE if _dtype_name(node) in _DTYPE_ALIAS.values() else _Kind.UNKNOWN
        if isinstance(node, ast.Subscript):
            base = self._classify(node.value)
            if base == _Kind.SCALAR:
                return _Kind.SCALAR          # `x.shape[0]` / 标量元组下标
            return base
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp)):
            kinds = [self._classify(o) for o in self._operands(node)]
            if _Kind.TENSOR in kinds or _Kind.PARAM in kinds:
                return _Kind.TENSOR
            if kinds and all(k == _Kind.SCALAR for k in kinds):
                return _Kind.SCALAR
            return _Kind.UNKNOWN
        if isinstance(node, ast.Compare):
            kinds = [self._classify(o) for o in self._operands(node)]
            if _Kind.TENSOR in kinds or _Kind.PARAM in kinds:
                return _Kind.TENSOR          # 张量比较 → bool **张量**
            if kinds and all(k in (_Kind.SCALAR, _Kind.NONE) for k in kinds):
                return _Kind.SCALAR
            return _Kind.UNKNOWN
        if isinstance(node, ast.Call):
            return self._classify_call(node)
        if isinstance(node, (ast.Tuple, ast.List)):
            return _Kind.SCALAR
        return _Kind.UNKNOWN

    def _classify_call(self, call: ast.Call) -> str:
        f = call.func
        if isinstance(f, ast.Name):
            if f.id in _HOST_BUILTINS:
                # host 侧内建:`int(k.shape[1])` / `len(...)` / `abs(shifts)`
                # (`multi_token_prediction.py:178`)/ `slice(...)`、`tuple(...)`。
                # 表与 `_handle_call` 的形态二.七五共用一份(`_HOST_BUILTINS`),避免两处漂移
                # —— 此前这里少了 `abs`/`round`/`slice`/`tuple`/`list`,于是
                # `if abs(shifts) >= dim_size:` 的左侧判成 UNKNOWN → 整个 MTP fail-loud。
                return _Kind.SCALAR
            if not call.args and not call.keywords                     and f"call:{f.id}" in self.runtime_predicates:
                return _Kind.SCALAR          # 调用方声明的**零参 host 事实**(见 _eval_predicate_call)
            if f.id in self._module_funcs or f.id in self._fn_classes:
                return _Kind.TENSOR
            return _Kind.UNKNOWN
        sattr = _self_attr(f)
        if sattr is not None:
            b = self.binds.get(sattr)
            if b is not None:
                return _Kind.SCALAR if b.op == SHAPE_OF else _Kind.TENSOR
            return _Kind.TENSOR if self._lookup_method(sattr)[0] is not None else _Kind.UNKNOWN
        if isinstance(f, ast.Attribute):
            # **先**按完整点号路径查表(`mint.permute` / `ops.cast` / …)——注意 `permute` 之类
            # 方法名与命名空间函数名同名,若先走"张量方法"分支会把 `mint` 当成 base 判成 UNKNOWN。
            path = _dotted_path(f)
            if path is not None:
                hit = prims.lookup(path)
                if hit is not None:
                    return _Kind.SCALAR if hit[0] == SHAPE_OF else _Kind.TENSOR
                if path in FREE_CALL_MAP:
                    return _Kind.TENSOR
                if prims.is_alias_namespace(path):
                    return _Kind.UNKNOWN     # 命名空间调用但表里没有 → 未知,别当张量方法
            if f.attr in ("astype", "to"):
                return _Kind.TENSOR
            if (f.attr in prims.PASSTHRU_METHODS or f.attr in _VIEW_METHODS
                    or prims.lookup_method(f.attr) is not None):
                return self._classify(f.value)
        return _Kind.UNKNOWN

    @staticmethod
    def _operands(node) -> list:
        if isinstance(node, ast.BinOp):
            return [node.left, node.right]
        if isinstance(node, ast.UnaryOp):
            return [node.operand]
        if isinstance(node, ast.BoolOp):
            return list(node.values)
        if isinstance(node, ast.Compare):
            return [node.left] + list(node.comparators)
        return []

    def _scalar_of(self, node):
        """把一个**标量**表达式求成 python 值;求不出返回 `_UNKNOWN_SCALAR`。
        支持:字面量、标量名、config 属性、`x.shape[i]`、`int()/len()` 包裹、+ - * // / % ** 与一元负。"""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.scalars:
                return self.scalars[node.id]
            if node.id in self.known_none:
                return None
            return _UNKNOWN_SCALAR
        if isinstance(node, ast.Attribute):
            nm = _config_flag_name(node)
            if nm is not None and nm in self.config_flags:
                return self.config_flags[nm]
            if node.attr == "ndim" and isinstance(node.value, ast.Name):
                # `<入口形参>.ndim` —— **轴数**由 `input_axes` 的种子长度给出(与
                # `infer_shapes(dag, input_shapes)` 同一套契约)。真源必需:
                # `if logits.ndim == 3:`(`pynative/loss/loss.py:363`,ChunkCrossEntropyLoss)
                # 与 `if logits.ndim != 3: raise`(`:373` 的守卫)。没种子 → 未知(不猜)。
                axes = self._input_axes.get(node.value.id)
                if axes:
                    return len(axes)
            return _UNKNOWN_SCALAR
        if isinstance(node, ast.Subscript):
            # `<x>.shape[i]` —— 优先用 `input_axes` 的轴种子解成**真值**(与 `_axis_scalar` /
            # `infer_shapes` 同一套契约);没种子才退回「值未知但 >= 1」的公理哨兵。
            # 真源必需:`dim_size = shape[dims]` 后 `if abs(shifts) >= dim_size:`
            # (`multi_token_prediction.py:174/178` 的 `roll_tensor`)—— 只有 `>= 1` 判不出真值。
            base = node.value
            if isinstance(base, ast.Attribute) and base.attr == "shape":
                idx = self._const_int(node.slice)
                if idx is not None and isinstance(base.value, ast.Name):
                    v = self._axis_scalar(base.value.id, idx)
                    if v is not _POSITIVE_DIM:
                        return v
                return _POSITIVE_DIM
            if isinstance(base, ast.Name) and isinstance(
                    self.scalars.get(base.id), (tuple, list)):
                # `shape[dims]`,其中 `shape` 是上面记下的轴长元组(见 `_axes_tuple`)。
                seq = self.scalars[base.id]
                idx = self._const_int(node.slice)
                if idx is None:
                    iv = self._scalar_of(node.slice)
                    idx = iv if isinstance(iv, int) and not isinstance(iv, bool) else None
                if idx is not None and -len(seq) <= idx < len(seq):
                    return seq[idx]
                return _UNKNOWN_SCALAR
            return _UNKNOWN_SCALAR
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and not node.args and not node.keywords
                    and f"call:{f.id}" in self.runtime_predicates):
                # 零参 host 事实的**声明值**。真源必需:`tp_cp_size =
                # get_moe_aux_loss_group_size()`(`moe/router.py:687`;该函数返回模块级全局
                # `_AUX_LOSS_GROUP_SIZE`,`moe/moe_utils.py:261-263`,由运行时的 tp×cp 组设置)。
                # 它随后参与 `aux_loss * grad_aux_loss_group_size` 的**标量**乘 → 必须是数,不是张量。
                return self.runtime_predicates[f"call:{f.id}"]
            if isinstance(f, ast.Name) and f.id in ("abs", "len", "min", "max") and node.args:
                vs = [self._scalar_of(a) for a in node.args]
                if any(v is _UNKNOWN_SCALAR or v is _POSITIVE_DIM or v is None for v in vs):
                    return _UNKNOWN_SCALAR
                try:
                    if f.id == "abs":
                        return abs(vs[0])
                    if f.id == "len":
                        return len(vs[0])
                    return (min if f.id == "min" else max)(*vs) if len(vs) > 1 else (
                        min(vs[0]) if f.id == "min" else max(vs[0]))
                except (TypeError, ValueError):
                    return _UNKNOWN_SCALAR
            if isinstance(f, ast.Name) and f.id in ("int", "float") and node.args:
                v = self._scalar_of(node.args[0])
                if v is _POSITIVE_DIM:
                    return _POSITIVE_DIM        # `int(x.shape[0])` 仍是"某轴长" (>=1)
                if v is _UNKNOWN_SCALAR or v is None:
                    return _UNKNOWN_SCALAR
                try:
                    return int(v) if f.id == "int" else float(v)
                except (TypeError, ValueError):
                    return _UNKNOWN_SCALAR
            return _UNKNOWN_SCALAR
        if isinstance(node, ast.UnaryOp):
            v = self._scalar_of(node.operand)
            if v is _UNKNOWN_SCALAR:
                return _UNKNOWN_SCALAR
            try:
                if isinstance(node.op, ast.USub):
                    return -v
                if isinstance(node.op, ast.UAdd):
                    return +v
                if isinstance(node.op, ast.Not):
                    return not v
            except TypeError:
                return _UNKNOWN_SCALAR
            return _UNKNOWN_SCALAR
        if isinstance(node, ast.BinOp):
            a, b = self._scalar_of(node.left), self._scalar_of(node.right)
            if a is _UNKNOWN_SCALAR or b is _UNKNOWN_SCALAR or a is None or b is None:
                return _UNKNOWN_SCALAR
            try:
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if isinstance(node.op, ast.FloorDiv):
                    return a // b
                if isinstance(node.op, ast.Div):
                    return a / b
                if isinstance(node.op, ast.Mod):
                    return a % b
                if isinstance(node.op, ast.Pow):
                    return a ** b
            except (TypeError, ZeroDivisionError):
                return _UNKNOWN_SCALAR
        return _UNKNOWN_SCALAR

    # ---- 具体 RHS 形态 --------------------------------------------------------------
    def _bind_expr_to_target(self, expr, target: str, lineno: int) -> None:
        """把任意 RHS 子表达式绑到单个目标(元组赋值的逐项通路)。"""
        if isinstance(expr, ast.Name):
            self._bind_name_rhs(expr.id, [target])
        elif isinstance(expr, ast.Constant):
            self._forget(target)
            if expr.value is None:
                self.known_none.add(target)
            else:
                self.scalars[target] = expr.value
        elif isinstance(expr, (ast.BinOp, ast.UnaryOp)):
            self._handle_binop(expr, [target], lineno)
        elif isinstance(expr, (ast.Compare, ast.BoolOp)):
            self._handle_compare(expr, [target], lineno)
        elif isinstance(expr, ast.Call):
            self._handle_call(expr, [target])
        elif isinstance(expr, ast.Attribute):
            self._handle_attribute_assign(
                expr, [target], ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())],
                                           value=expr, lineno=lineno))
        elif isinstance(expr, ast.Subscript):
            self._handle_subscript_assign(expr, [target], lineno)
        else:
            self._forget(target)
            self.scalars[target] = _UNKNOWN_SCALAR

    def _bind_name_rhs(self, src_name: str, targets: list[str]) -> None:
        """`a = b` —— 按 b 的身份别名过去(张量保 producer 边;标量/dtype/None 传身份)。"""
        if src_name in targets:
            # **自赋值**(`x = <passthru>(x, ...)`):`_forget(t)` 会先把源的身份清掉,
            # 于是下面所有分支都判不出来 → 目标掉出 SSA → 下游拿占位 ref 且**丢边**。
            # 真源触发点:`experts_output = DTensor.from_local(experts_output, ...)`
            # (`moe/experts.py:211`)—— 丢掉它,`MoELayer` 的 `self.reshape(routed_output, ...)`
            # (`moe_layer.py:137`)就落 `unresolved_operands`。恒等赋值:什么都不用做。
            for t in targets:
                if t != src_name:
                    self._bind_name_rhs(src_name, [t])
            return
        for t in targets:
            self._forget(t)
        if src_name in self.param_aliases:
            for t in targets:
                self.param_aliases.add(t)
                self._param_of[t] = self._param_of.get(src_name, src_name)
            return
        if src_name in self.ssa or src_name in self.producer:
            self._alias(targets, src_name)
            for t in targets:
                if src_name in self.detached and t not in self.detached:
                    self.detached.append(t)
            return
        for t in targets:
            if src_name in self.dtypes:
                self.dtypes[t] = self.dtypes[src_name]
            elif src_name in self.scalars:
                self.scalars[t] = self.scalars[src_name]
            elif src_name in self.known_none:
                self.known_none.add(t)
            elif src_name in self._param_set:
                self._alias([t], src_name)   # 形参:占位 ref 别名(合法,种子由 infer_shapes 喂)
            else:
                self.scalars[t] = _UNKNOWN_SCALAR

    def _handle_attribute_assign(self, val: ast.Attribute, targets: list[str], stmt) -> None:
        """`t = <expr>.<attr>` 的三类:dtype 记号 / shape 元组 / config 标量 / **权重 Parameter**。"""
        if (val.attr == "shape" and isinstance(val.value, ast.Name)
                and self._axes_tuple(val.value.id) is not _UNKNOWN_SCALAR):
            # `shape = tensor.shape` —— host 侧**轴长元组**(有种子时逐轴带真值)。
            for t in targets:
                self._forget(t)
                self.scalars[t] = self._axes_tuple(val.value.id)
            return
        if val.attr in _HOST_META_ATTRS and self._classify(val.value) in (
                _Kind.TENSOR, _Kind.PARAM):
            # `tokens_layout = tokens.layout`(`moe/experts.py:186`):DTensor 的**分片布局
            # 元数据对象**(host 侧),不是张量、不占激活。它**确实存在**(刚从一个张量上读到,
            # 且该分支是 `isinstance(tokens, DTensor)` 判 True 才进来的)→ 记 `_PRESENT`,
            # 使下游 `if tokens_layout is not None:`(`:210`)可判定。
            for t in targets:
                self._forget(t)
                self.scalars[t] = _PRESENT
            return
        kind = self._classify(val)
        if kind == _Kind.DTYPE and val.attr == "dtype":
            # `ori_dtype = x.dtype`(multi_latent_attention.py:232):dtype **记号**,字节中性。
            # 记进 dtype 环境 → 下游 `self.cast(y, ori_dtype)` 能解出真 dtype(此前解成字面串
            # "ori_dtype",即一个**假 dtype**)。
            base = val.value
            dt = None
            if isinstance(base, ast.Name):
                ref = self.ssa.get(base.id)
                dt = ref.split(":")[2] if ref and ref.count(":") == 2 else None
            for t in targets:
                self._forget(t)
                self.dtypes[t] = dt or self.config_flags.get("compute_dtype", "bf16")
            return
        if kind == _Kind.PARAM:
            # `attn_sink = self.attn_sink`(csa.py:683):**权重**,不是激活 →
            # 记成权重别名(不进 SSA),消费点路由到 param_operands(W2/W3)。
            sattr = _self_attr(val)
            for t in targets:
                self._forget(t)
                self.param_aliases.add(t)
                self._param_of[t] = sattr
                self._note_param_operand(sattr, stmt.lineno)
            return
        if kind == _Kind.SCALAR:
            v = self._scalar_of(val)
            for t in targets:
                self._forget(t)
                if v is None:
                    self.known_none.add(t)
                else:
                    self.scalars[t] = v
            return
        # 判不出来的 `self.<attr>` / `<x>.<attr>` —— 保持记账(它确实是"看不懂")。
        self._diag("dropped_assigns", src=f"{self.src_file}:{stmt.lineno}",
                   targets=list(targets), rhs="Attribute", code=self._describe(stmt))
        self._diag_unregistered(targets, stmt.lineno, "dropped_assign_rhs_Attribute")

    def _param_dtype(self, sattr: str) -> str:
        _ = sattr
        return self.config_flags.get("params_dtype", "bf16")

    def _note_param_operand(self, name: str, lineno: int) -> None:
        rec = {"src": f"{self.src_file}:{lineno}", "param": name}
        if rec not in self.param_operands:
            self.param_operands.append(rec)

    def _binop_kind(self, node) -> tuple[str, dict, str]:
        """二元/一元算术 → (op 类型, attrs, 记号)。**关键判据(直接决定字节)**:
        `tensor <op> scalar` 的反向是 `dy·const` —— **与输入值无关 → 线性 → 不存激活**;
        只有 `tensor <op> tensor` 的 mul/div 才需要存两个操作数。
        实例:`attention_scores = self.matmul(q,k) * self.softmax_scale`(indexer.py:350)
        —— 若误判非线性,就会把那个 O(S·S/r) fp32 张量错声明成 saved。"""
        if isinstance(node, ast.UnaryOp):
            return "Elementwise", {"linear": True, "arith": "neg"}, "neg"
        op = node.op
        both_tensor = sum(
            1 for o in self._operands(node)
            if self._classify(o) in (_Kind.TENSOR, _Kind.PARAM)
        ) >= 2
        if isinstance(op, (ast.Add, ast.Sub)):
            return "Elementwise", {"linear": True, "arith": "add"}, "add"
        if isinstance(op, (ast.BitOr, ast.BitAnd, ast.BitXor)):
            # bool 掩码的逻辑组合(csa.py:376 `(causal_mask < 0) | (...)`):不可微。
            return "Compare", {"logical": True}, "logical"
        if isinstance(op, ast.MatMult):
            return "MatMul", {}, "matmul"
        if isinstance(op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            return "Elementwise", {"linear": not both_tensor, "arith": "mul"}, "mul"
        return "Elementwise", {"linear": False, "arith": "other"}, "other"

    def _handle_binop(self, val, targets: list[str], lineno: int) -> None:
        """`BinOp`/`UnaryOp` RHS。张量 → 建节点(**这次要修的张量全在这里**);标量 → 只记标量。"""
        kind = self._classify(val)
        if kind == _Kind.SCALAR:
            v = self._scalar_of(val)
            for t in targets:
                self._forget(t)
                self.scalars[t] = v
            return
        if kind != _Kind.TENSOR:
            self._diag("dropped_assigns", src=f"{self.src_file}:{lineno}",
                       targets=list(targets), rhs=type(val).__name__,
                       code=self._describe(val))
            self._diag_unregistered(targets, lineno, f"dropped_assign_rhs_{type(val).__name__}")
            return
        op, attrs, _tag = self._binop_kind(val)
        operands = self._tensor_operands(self._operands(val), lineno)
        self._emit(op, attrs, lineno, operands, targets,
                   "bool" if op == "Compare" else None)

    def _handle_compare(self, val, targets: list[str], lineno: int) -> None:
        """`Compare`/`BoolOp` RHS。张量比较 → bool **张量**节点(`Compare`,不可微、反向不读)。
        实例:`future = cm >= positions // ratio`(csa.py:779)、
        `valid = topk_indices_compressed < self.unsqueeze(n_valid_per_pos, 0)`(csa.py:810)
        —— 两个 O(S·S/r) bool mask,此前完全不可见。"""
        kind = self._classify(val)
        if kind != _Kind.TENSOR:
            # **host 侧谓词**:整条 RHS 能由 config / 部署形态谓词判成一个 bool → 它是一个
            # python 布尔,不是张量;登记标量身份、**不建节点**。真源必需(2026-07-25):
            #   `need_dispatch = not isinstance(self.weight1, DTensor) or "ep" not in
            #    self.weight1.device_mesh.mesh_dim_names`(`moe/experts.py:181`)
            # 之后 `if need_dispatch:`(`:191`)决定 permute/unpermute 那一整段是否存在。
            # 判不出来仍走下面的记账(strict 下抛)—— 不猜。
            # 守卫 `kind != TENSOR`:张量比较绝不能走这条(`_value` 对有 producer 的名给
            # `_PRESENT`,`t1 == t2` 会假判成 True)。
            r = self._eval_test(val)
            if r is not _UNDECIDED:
                for t in targets:
                    self._forget(t)
                    self.scalars[t] = bool(r)
                return
        if kind == _Kind.SCALAR:
            for t in targets:
                self._forget(t)
                self.scalars[t] = _UNKNOWN_SCALAR
            return
        if kind != _Kind.TENSOR:
            self._diag("dropped_assigns", src=f"{self.src_file}:{lineno}",
                       targets=list(targets), rhs=type(val).__name__,
                       code=self._describe(val))
            self._diag_unregistered(targets, lineno, f"dropped_assign_rhs_{type(val).__name__}")
            return
        attrs = {"logical": True} if isinstance(val, ast.BoolOp) else {"compare": True}
        operands = self._tensor_operands(self._operands(val), lineno)
        self._emit("Compare", attrs, lineno, operands, targets, "bool")

    def _handle_subscript_assign(self, val: ast.Subscript, targets: list[str], lineno: int) -> None:
        """`t = <x>[<idx>]` 的三类:
          * base 是标量元组(`x.shape[0]`)→ 标量,**不建节点**;
          * 下标含 `Slice` → `View{view:"slice"}`(反向零填,不存激活)。
            实例 `freqs = freqs[:total:ratio][:n]`(compressor.py:233)、`kv = kv[:cutoff]`(:198);
          * 下标是**张量**(advanced indexing)→ `IndexSelect`(反向 scatter_add → 存 index)。
            实例 `kv_flat[flat_indices]`(csa.py:485)—— 这是 unfused 链里真正的 gather。"""
        base_kind = self._classify(val.value)
        if base_kind == _Kind.SCALAR:
            v = self._scalar_of(val)
            for t in targets:
                self._forget(t)
                self.scalars[t] = v
            return
        if base_kind not in (_Kind.TENSOR, _Kind.PARAM):
            self._diag("dropped_assigns", src=f"{self.src_file}:{lineno}",
                       targets=list(targets), rhs="Subscript", code=self._describe(val))
            self._diag_unregistered(targets, lineno, "dropped_assign_rhs_Subscript")
            return
        idx = val.slice
        idx_tensors = [n for n in ast.walk(idx)
                       if isinstance(n, ast.Name) and self._classify(n) == _Kind.TENSOR]
        if idx_tensors:
            operands = self._tensor_operands([val.value] + idx_tensors, lineno)
            self._emit("IndexSelect", {"advanced_index": True}, lineno, operands, targets, None)
            return
        operands = self._tensor_operands([val.value], lineno)
        attrs = {"view": "slice", "index": self._describe(idx)}
        bounds = self._slice_bounds(idx)
        if bounds is not None:
            attrs["slice_bounds"] = bounds
        self._emit("View", attrs, lineno, operands, targets, None)

    def _slice_bounds(self, idx):
        """逐轴 `(lower, upper, step)`(未求值符号串或 None);抠不出返回 None(**不猜**)。

        真源:`kv = kv[:cutoff]` / `score = score[:cutoff]`(`compressor.py:198/199`)、
        `freqs = freqs[:total:ratio][:n]`(`:233`)—— 没有 bounds,切片在下游只能按整轴算。
        """
        items = idx.elts if isinstance(idx, ast.Tuple) else [idx]
        out = []
        for it in items:
            if isinstance(it, ast.Slice):
                out.append(tuple(None if p is None else self._describe(p)
                                 for p in (it.lower, it.upper, it.step)))
            elif isinstance(it, ast.Constant) or self._const_int(it) is not None:
                out.append(("index", self._describe(it), None))
            else:
                return None                  # 有一轴抠不出 → 整条不写(缺键 = 未知,不是错)
        return out

    def _tensor_operands(self, exprs, lineno: int) -> list:
        """从混合实参里挑出**可追踪张量操作数**(Name / 嵌套 Call 先物化 / `self.<Parameter>` 记权重),
        标量与 dtype 记号一律略过(它们不是激活,进 `ins` 就会被 bprop 当张量算字节)。"""
        out = []
        for e in exprs:
            k = self._classify(e)
            if k == _Kind.PARAM:
                sattr = _self_attr(e) or (self._param_of.get(e.id)
                                          if isinstance(e, ast.Name) else None)
                if sattr:
                    self._note_param_operand(sattr, lineno)
                continue                       # 权重不进 ins(见 param_operands 的理由)
            if k != _Kind.TENSOR:
                continue
            if isinstance(e, (ast.Name,)):
                out.append(e)
            elif isinstance(e, (ast.Call, ast.Subscript, ast.BinOp, ast.UnaryOp,
                               ast.Compare, ast.BoolOp, ast.Attribute)):
                out.append(self._materialize_expr(e, lineno))
        return [o for o in out if o is not None]

    def _materialize_expr(self, expr, lineno: int):
        """把一个**张量子表达式**先发射成节点,返回指向它的 `ast.Name`(保 id 单调 = 数据流序)。
        镜像既有 `_materialize_call_arg` 的 `__arg__i<n>` 合成名模式。"""
        if isinstance(expr, ast.Name):
            return expr
        tmp = f"__arg__i{self._frame_seq}"
        self._frame_seq += 1
        before = len(self.nodes)
        if isinstance(expr, ast.Call):
            self._handle_call(expr, [tmp])
        elif isinstance(expr, (ast.BinOp, ast.UnaryOp)):
            self._handle_binop(expr, [tmp], lineno)
        elif isinstance(expr, (ast.Compare, ast.BoolOp)):
            self._handle_compare(expr, [tmp], lineno)
        elif isinstance(expr, ast.Subscript):
            self._handle_subscript_assign(expr, [tmp], lineno)
        elif isinstance(expr, ast.Attribute):
            sattr = _self_attr(expr)
            if sattr:
                self._note_param_operand(sattr, lineno)
            return None
        if len(self.nodes) == before and tmp not in self.ssa:
            return None
        return ast.Name(id=tmp, ctx=ast.Load())

    def _promote_dtype(self, exprs) -> str:
        """产出 dtype = 参与张量操作数里"最宽"的那个(fp32 > bf16/fp16 > 其它);无从判断则 bf16。
        这条让 `score_f32 = score.astype(fp32) + ape`(compressor.py:209)的产出正确落 fp32。"""
        rank = {"fp64": 4, "fp32": 3, "bf16": 2, "fp16": 2}
        best, best_r = None, -1
        for e in exprs:
            dt = None
            if isinstance(e, ast.Name):
                ref = self.ssa.get(e.id)
                if ref and ref.count(":") == 2:
                    dt = ref.split(":")[2]
            if dt and rank.get(dt, 1) > best_r:
                best, best_r = dt, rank.get(dt, 1)
        return best or self.config_flags.get("compute_dtype", "bf16")

    def _handle_ifexp(self, ifexp: ast.IfExp, targets: list[str]) -> None:
        self._pending_predicate = None
        r = self._eval_test(ifexp.test)
        if r is _UNDECIDED:
            if self._pruning:
                extra = (f" 该条件是**部署形态谓词**:请在 `runtime_predicates` 里给 "
                         f"`{self._pending_predicate}`。" if self._pending_predicate else "")
                raise ValueError(
                    f"construct 的三元条件无法由 config 判定（{self.src_file}:{ifexp.lineno}）:"
                    f"`{self._describe(ifexp.test)}` —— 剪枝上下文下拒绝双走,fail-loud{extra}"
                )
            r = True  # 非剪枝:保守取 body(优先保留算子,别静默丢)
        chosen = ifexp.body if r else ifexp.orelse
        if isinstance(chosen, ast.Call):
            self._handle_call(chosen, targets)
        elif isinstance(chosen, ast.Name):
            # `... else x`:目标别名 x —— 复用 x 的 ref 与 producer(下游消费能连回真源)。
            self._bind_name_rhs(chosen.id, targets)
        else:
            # 其余表达式(`self.<Parameter>` / config 标量 / 字面量 / BinOp / …):走统一 RHS 通路,
            # 别再"不登记 SSA"——那正是下游拿占位 ref、丢边的来源。
            #   `ape = self.ape.to_local() if hasattr(...) else self.ape`(compressor.py:208)
            #   `effective_topk = self.index_topk if ... else min(...)`(indexer.py:212)
            for t in targets:
                self._bind_expr_to_target(chosen, t, ifexp.lineno)

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
        # 形态零.五:**python 级方法派发表**的两步(见 `_method_tables` 的注释)。
        #   ① `<local> = <table>.get(<config flag>)`(router.py:485)→ 解成一个方法名;
        #   ② `<local>(...)`(router.py:489)→ 内联那个方法。
        if (isinstance(func, ast.Attribute) and func.attr == "get"
                and isinstance(func.value, ast.Name)
                and func.value.id in self._method_tables):
            if self._resolve_method_table_get(call, func.value.id, target_names):
                return
        if isinstance(func, ast.Name) and func.id in self._method_refs:
            m, mrel = self._lookup_method(self._method_refs[func.id])
            if m is not None:
                self._inline_method(m, call, target_names, method_rel=mrel)
                return
        # 形态一.二:`super().<method>(...)` —— 内联**基类**的同名方法。
        # 真源必需:`shared_experts_output = super().construct(hidden_states)`
        # (`moe/shared_experts.py:68`)—— `SharedExpertMLP(MLP)` 的整个 fc1→act→fc2 主体
        # 都在基类 `MLP.construct` 里;不解这条,共享专家段抽出 **0 节点**。
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "super"):
            m, mrel = self._lookup_super_method(func.attr)
            if m is None:
                raise ValueError(
                    f"`super().{func.attr}(...)`（{self.src_file}:{call.lineno}）在 "
                    f"{self._cls_name} 的基类里找不到 —— fail-loud(不猜)"
                )
            self._inline_method(m, call, target_names, method_rel=mrel)
            return
        # 形态一.五:`self.<subcell>.<method>(...)` —— 子 Cell 的**指定方法**(非 construct)。
        if isinstance(func, ast.Attribute):
            outer = _self_attr(func.value)
            if outer is not None and func.attr != "apply":
                if self._handle_subcell_method(call, outer, func.attr, target_names):
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
            # 形态二.六(P0#2):裸 `mint.*` / `ops.*` **自由调用**按同一张 primitives 表发射。
            # `unfused_compressed_sparse_attn`(csa.py:464-533)整条链就是这么写的(24 处),
            # `CSAIndexer` 的 `mint.squeeze`(indexer.py:231-232)亦然。
            if path is not None and prims.lookup(path) is not None:
                self._emit_primitive(path, call, target_names)
                return
        # 形态二.七:`<Cls>.apply(...)`(mindspore `_Function` 自定义反向)/ `self.<attr>.apply(...)`。
        if isinstance(func, ast.Attribute) and func.attr == "apply":
            if self._handle_function_apply(call, func, target_names):
                return
        # 形态二.七二:`Tensor([...], dtype=...)` —— host 侧字面量建的小常量张量
        # (`cmp_residual_k = Tensor([int(key_length) % self.compress_ratio], ...)`,indexer.py:219)。
        if isinstance(func, ast.Name) and func.id in _CONST_CTORS:
            self._emit("Constant", {"ctor": func.id}, call.lineno, [], target_names,
                       self._const_dtype(call))
            return
        # 形态二.七三:融合 NPU 内核的自由函数(调用方白名单)。**建节点 + 登记目标 + 保边**;
        # saved 集只在"可证明无反向"(no-grad 区内)时写 [],否则不写 → derive_saves fail-loud。
        if isinstance(func, ast.Name) and func.id in self._kernel_call_allow:
            attrs = {"kernel": func.id}
            operands = self._tensor_operands(list(call.args) +
                                            [k.value for k in call.keywords], call.lineno)
            if self._nograd_depth:
                attrs["saved_ins_idx"] = []
                attrs["no_backward_reason"] = "in _no_grad region"
            else:
                decl = self._kernel_saves.get(func.id)
                if decl is not None:
                    # saved 集的两个来源,在图上**分开记**(绝不把上界静默升格为事实):
                    #   * `saved_from_source=True` —— 逐字读自源(`fn_saves.mhc_kernel_saves`
                    #     从 `hyper_parallel` 的 `ctx.save_for_backward(...)` 读出);
                    #   * `saved_declared_by_caller=True` —— 调用方声明的**上界**
                    #     (内核 bprop 在快照外时唯一诚实的通路)。
                    # 两者都必须带 source/reason,否则与「猜」不可区分。
                    idx = decl.get("saved_ins_idx")
                    n_tensor = len([a for a in operands if isinstance(a, ast.Name)])
                    attrs["saved_ins_idx"] = (list(range(n_tensor)) if idx == "all"
                                              else sorted(set(idx or ())))
                    # 融合内核**保存自己输出**的情形(`npu_mhc_pre_sinkhorn` 存 5 个自身输出,
                    # custom_op_impl.py:390-391)——源里 `h_in, h_post, h_res_flat, *_ = ...`
                    # 把它们丢弃了,但 autograd ctx 仍持有 ⇒ 显存**真实占用**,不能因为
                    # Python 侧没名字就漏掉。
                    if decl.get("saved_outs_idx"):
                        attrs["saved_outs_idx"] = sorted(set(decl["saved_outs_idx"]))
                        attrs["saved_out_names"] = dict(decl.get("saved_out_names") or {})
                        attrs["saved_out_shapes"] = dict(decl.get("saved_out_shapes") or {})
                    if decl.get("saved_from_source"):
                        attrs["saved_from_source"] = True
                    else:
                        attrs["saved_declared_by_caller"] = True
                    attrs["saved_source"] = decl.get("source", "")
                    attrs["saved_reason"] = decl.get("reason", "")
                    if not attrs["saved_source"] or not attrs["saved_reason"]:
                        raise ValueError(
                            f"内核 `{func.id}` 的 saved 集声明缺 source/reason"
                            f"（{self.src_file}:{call.lineno}）—— 声明必须带「据什么、为什么」,"
                            f"否则它与「猜」不可区分,fail-loud"
                        )
            self._emit("Kernel", attrs, call.lineno, operands, target_names,
                       self._promote_dtype(operands))
            return
        # 形态二.七五:host 侧内建 `int()/float()/min()/max()/len()/...`(**全标量实参**)——
        # 产出是 python 数,不是张量:登记标量身份,**不建节点、不记诊断**。
        # `effective_topk = ... min(self.index_topk, int(k.shape[1]))`(indexer.py:212)。
        if isinstance(func, ast.Name) and func.id in _HOST_BUILTINS:
            if all(self._classify(a) != _Kind.TENSOR for a in call.args):
                for t in target_names:
                    self._forget(t)
                    self.scalars[t] = self._scalar_of(call)
                return
        # 形态二.七六:**调用方声明的零参 host 事实** `<name>()`(键 `call:<name>`)——
        # 产出是 python 数/bool,不是张量。`tp_cp_size = get_moe_aux_loss_group_size()`
        # (`moe/router.py:687`;该函数返回模块级全局 `_AUX_LOSS_GROUP_SIZE`,`moe_utils.py:261`)。
        if (isinstance(func, ast.Name) and not call.args and not call.keywords
                and f"call:{func.id}" in self.runtime_predicates):
            for t in target_names:
                self._forget(t)
                self.scalars[t] = self.runtime_predicates[f"call:{func.id}"]
            return
        # 形态二.八:模块级自由函数 `foo(...)`(同文件 `def`)→ **内联展开**。
        # `unfused_compressed_sparse_attn`(csa.py:823)、`parse_cu_seqlens`(:743)、
        # `get_window_topk_idxs`(:758)…此前只进 opaque_calls → 整条 11-op 注意力链不可见。
        # 形态二.七九:construct 形参持有的子 Cell(见 `_param_cells`)。
        if (isinstance(func, ast.Name) and func.id in self._param_cells
                and self._subcell_resolver is not None):
            self._inline_subcell({"cell": self._param_cells[func.id],
                                  "field": func.id, "bare": True}, call, target_names)
            return
        if isinstance(func, ast.Name) and func.id in self._module_funcs                 and func.id not in self._host_call_allow:
            fn, frel = self._module_funcs[func.id]
            self._inline_method(fn, call, target_names, is_module_func=True, method_rel=frel)
            return
        # 形态二.九:`<expr>.<张量方法>(...)` —— base 可以是 Call / BinOp / Name / Subscript。
        # 实测必需:`pooled = (kv.astype(fp32) * weights).sum(dim=1)`(compressor.py:216)
        # —— base 是**带括号的 BinOp**,既有的"链式调用"分支只认 Call/Subscript base,
        # 于是此前整句落 opaque、`pooled` 从未进 SSA、下游 `self.norm(pooled...)` 丢边。
        if isinstance(func, ast.Attribute) and prims.lookup_method(func.attr) is not None:
            if self._classify(func.value) in (_Kind.TENSOR, _Kind.PARAM):
                op, attrs = prims.lookup_method(func.attr)
                base = self._materialize_expr(func.value, call.lineno)
                if base is not None:
                    args = [base] + list(call.args)
                    attrs = {**attrs, "prim": f"<tensor>.{func.attr}"}
                    if op == "View" and attrs.get("view"):
                        attrs = {**attrs, **self._view_capture(call, attrs["view"], target_names)}
                    # `(kv.astype(fp32) * weights).sum(dim=1)`(`compressor.py:216`)——
                    # 缺 `reduce_dim` 时下游按直通算,实测 **8× 过读**(G4)。**注意**:张量方法的
                    # 实参不含 receiver,故轴的位置比 `mint.sum(x, dim)` 少一位 → 用一个
                    # 合成的"补上 receiver"的 call 去抠,位序才与命名空间形式一致。
                    shim = ast.Call(func=call.func,
                                    args=[base] + list(call.args), keywords=list(call.keywords))
                    ast.copy_location(shim, call)
                    attrs = {**attrs, **self._axis_capture(op, attrs, shim, target_names)}
                    out_dtype = "bool" if op == "Compare" else None
                    self._emit(op, attrs, call.lineno,
                               self._expand_args(args, attrs), target_names, out_dtype)
                    return
        # 形态三:链式方法 `<innercall>.method(...)`(如 self.swiglu(x).reshape(...)):先发射内层算子,再处理外层方法。
        if isinstance(func, ast.Attribute) and isinstance(func.value, (ast.Call, ast.Subscript)):
            self._handle_chained_call(call, func, target_names)
            return
        # 形态四:`<expr>.astype(<dtype>)` / `<expr>.to(<dtype>)`(expr 为 Name/Attribute)—— Cast。
        # `.to(mstype.float32)`:indexer.py:359 `mask.to(...)`、:365 `.to(mstype.int64)`。
        if isinstance(func, ast.Attribute) and func.attr in ("astype", "to"):
            out_dtype = self._resolve_cast_dtype(call.args, 0, {})  # 目标 dtype 在 idx=0
            self._emit("Cast", {}, call.lineno, [func.value], target_names, out_dtype)
            return
        # 形态四.五:`<expr>.to_local()` / `.full_tensor()` / `.contiguous()` —— **直通别名**,
        # 不新增激活、不建节点(DTensor 局部分片视图 / 布局重排,数学恒等)。
        if isinstance(func, ast.Attribute) and func.attr in prims.PASSTHRU_METHODS:
            self._bind_passthru(func.value, target_names, call.lineno)
            return
        # 形态四.五二:`DTensor.from_local(<local tensor>, mesh, placements)`
        # (`moe/experts.py:211`)—— `.to_local()` 的**逆**(已在 `PASSTHRU_METHODS` 里):
        # 把一个已存在的 local 张量**包**成 DTensor 视图,不复制、不新增激活、数学恒等。
        # 故与 `.to_local()` 同一条判据:目标承接首个实参的身份(保 producer 边),不建节点。
        if _dotted_path(func) in _PASSTHRU_FREE_CALLS and call.args:
            self._bind_passthru(call.args[_PASSTHRU_FREE_CALLS[_dotted_path(func)]],
                                target_names, call.lineno)
            return
        # 形态四.六:张量方法形态的视图 `<Name>.unsqueeze(1)` / `.broadcast_to(...)`(链首是 Name/Attribute)。
        if (isinstance(func, ast.Attribute) and func.attr in _VIEW_METHODS
                and self._classify(func.value) in (_Kind.TENSOR, _Kind.PARAM)):
            self._emit("View", {"view": func.attr}, call.lineno, [func.value], target_names,
                       self._promote_dtype([func.value]))
            return
        # 形态四.六五:**非梯度 buffer 的原地更新** `self.<buf>.<method>_(...)`,无赋值目标。
        # 真源:`self.tokens_per_expert.add_(num_tokens_per_expert)`(`moe/moe_layer.py:127`)。
        # 为什么字节中性 **可证明**(不是断言):
        #   ① 接收者是 `Parameter(..., requires_grad=False)`(`moe_layer.py:84-88` 逐字)
        #      → 它不在 autograd 图上,不产生 saved 激活;
        #   ② 方法名以 `_` 结尾 = mindspore/pytorch 的**原地**约定 → 写进已存在的持久 buffer,
        #      **不新分配**;③ 无赋值目标 → 没有下游张量消费它。
        # 仍记一条 `opaque_calls`(`kind="inplace_param_buffer_update"`)+ 记 param_operands,
        # 使它**可见可评审**,而不是静默消失。缺 `requires_grad=False` 的源侧证据 → 不走这条。
        if (isinstance(func, ast.Attribute) and func.attr.endswith("_")
                and not func.attr.startswith("_") and not target_names):
            sattr = _self_attr(func.value)
            if (sattr is not None and self._self_kinds.get(sattr) == "param"
                    and self._self_kinds.get(f"__buffer__{sattr}")):
                self._note_param_operand(sattr, call.lineno)
                self.opaque_calls.append(
                    {"src": f"{self.src_file}:{call.lineno}", "expr": self._describe(call),
                     "kind": "inplace_param_buffer_update", "param": sattr})
                return
        # 形态四.七:显式白名单的**纯宿主副作用调用**(无被消费的返回值),记 opaque、不记 unregistered。
        if not target_names and self._host_allow_key(func) is not None:
            self.opaque_calls.append(
                {"src": f"{self.src_file}:{call.lineno}", "expr": self._describe(call),
                 "kind": "host_side_effect", "allow": self._host_allow_key(func)})
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

    def _resolve_method_table_get(self, call: ast.Call, table_name: str,
                                  target_names: list[str]) -> bool:
        """`<table>.get(<key>)` —— 键必须能由 config 判定,否则**不解**(返回 False 走既有兜底)。"""
        if not call.args:
            return False
        known, key = self._value(call.args[0])
        if not known or not isinstance(key, (str, int)):
            return False
        method = self._method_tables[table_name].get(key)
        if method is None:
            raise ValueError(
                f"方法派发表 `{table_name}` 里没有 config 给定的键 {key!r}"
                f"（{self.src_file}:{call.lineno}）—— 源里紧随其后就是 `raise ValueError`"
                f"(router.py:486-487),即这个 config 组合不被支持,fail-loud"
            )
        for t in target_names:
            self._forget(t)
            self._method_refs[t] = method
            self.present_vars.add(t)     # `if aux_loss_func is None: raise`(:486)得以剪掉
        return True

    def _host_allow_key(self, func):
        """调用方白名单里命中的键:裸名 / 完整点号路径 / 属性末段;都没命中 → None。

        点号路径与属性末段两种形态是 2026-07-25 补的:真源里有
        `Validator.check_type_name("input_ids", input_.dtype, [...], self.cls_name)`
        (`pynative/base_models/common/embeddings/vocab_embedding.py:78`)—— 一条**纯 host 侧
        类型断言**(无返回值被消费、不产张量),此前只匹配 `ast.Name` 形态而漏掉。
        """
        if isinstance(func, ast.Name):
            return func.id if func.id in self._host_call_allow else None
        if isinstance(func, ast.Attribute):
            path = _dotted_path(func)
            if path and path in self._host_call_allow:
                return path
            if func.attr in self._host_call_allow:
                return func.attr
        return None

    def _handle_self_call(self, call: ast.Call, name: str, target_names: list[str]) -> None:
        binding = self.binds.get(name)
        if binding is None:
            # 未绑定:①Morph(self.method) 别名 → 内联被包裹的方法;②本类(含基类)内部方法 → 内联;③否则 fail-loud。
            if name in self._method_aliases:
                m, mrel = self._lookup_method(self._method_aliases[name])
                if m is not None:
                    self._inline_method(m, call, target_names, method_rel=mrel)
                    return
            method, mrel = self._lookup_method(name)
            if method is not None:
                self._inline_method(method, call, target_names, method_rel=mrel)
                return
            if name in self._alias_unknown:
                # P0#2 的 fail-loud 面:这个名字**是**裸函数别名,但 primitives 表里没有它。
                # 报错必须指名道姓——归错类会静默产出错的 saved 集(这件事要杀的正是那类 bug)。
                info = self._alias_unknown[name]
                raise UnknownPrimitiveError(
                    f"construct 调用了裸别名 self.{name}(...)（{self.src_file}:{call.lineno}）,"
                    f"其别名目标 `{info.get('alias')}`(定义于 {info.get('src')})"
                    f"**不在 primitives.PRIMITIVES 表里** —— 拒绝猜它的 bprop 类别。"
                    f"请加表项(并写清「反向要读什么」+ 定位符)。"
                )
            raise ValueError(
                f"construct 调用了未绑定的 self.{name}(...)（{self.src_file}:{call.lineno}）:"
                f"Pass B(init_binder)未覆盖此名、且非本类内部方法/Morph 别名,fail-loud"
            )
        if binding.op == SHAPE_OF:
            # `sq, bsz, _ = self.shape(x)`(deepseek_v4_hybrid_attention.py:233):产出是 python
            # int 元组,**不是张量** → 绝不发射节点(发了就是造假节点),只记轴解包 + 标量身份。
            src = call.args[0].id if (call.args and isinstance(call.args[0], ast.Name)) else None
            if src is not None:
                self._bind_shape_unpack(list(target_names), src)
            else:
                for t in target_names:
                    self._forget(t)
                    self.scalars[t] = _UNKNOWN_SCALAR
            return
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
        # 轴/形状元信息(归约 dim / topk k / 常量 shape)—— 缺轴时下游按直通算 = 静默错(G4)。
        attrs = {**attrs, **self._axis_capture(binding.op, attrs, call, target_names)}
        arg_exprs = self._expand_args(call.args, binding.attrs)
        self._emit(binding.op, attrs, call.lineno, arg_exprs, target_names, out_dtype)

    # ---- primitives 表驱动的发射(裸别名与自由调用共用)----
    def _emit_primitive(self, path: str, call: ast.Call, target_names: list[str],
                        op_attrs=None) -> None:
        op, attrs = op_attrs if op_attrs is not None else prims.lookup(path)
        if op == SHAPE_OF:
            src = call.args[0].id if (call.args and isinstance(call.args[0], ast.Name)) else None
            if src is not None:
                self._bind_shape_unpack(list(target_names), src)
            else:
                for t in target_names:
                    self._forget(t)
                    self.scalars[t] = _UNKNOWN_SCALAR
            return
        if op == "Cast":
            out_dtype = self._resolve_cast_dtype(call.args, 1, attrs)
        elif op == "Compare":
            out_dtype = "bool"
        elif op == "Constant":
            out_dtype = self._const_dtype(call)
        else:
            out_dtype = None
        arg_exprs = self._expand_args(call.args, attrs)
        if op == "View" and attrs.get("view"):
            attrs = {**attrs, **self._view_capture(call, attrs["view"], target_names)}
        attrs = {**attrs, "prim": path}
        attrs = {**attrs, **self._axis_capture(op, attrs, call, target_names)}
        # out_dtype 仍为 None → 交给 `_emit` 按**已解析的 ins** 推(实参可能还没物化)。
        self._emit(op, attrs, call.lineno, arg_exprs, target_names, out_dtype)

    def _const_dtype(self, call: ast.Call) -> str:
        """`mint.full(shape, v, dtype=mstype.float32)` / `mint.arange(n, dtype=...)` 的产出 dtype。
        没写 dtype 就按 compute_dtype(不杜撰 fp32)。"""
        for kw in call.keywords:
            if kw.arg == "dtype":
                d = _dtype_name(kw.value)
                if d:
                    return d
        return self.config_flags.get("compute_dtype", "bf16")

    def _handle_function_apply(self, call: ast.Call, func: ast.Attribute,
                               target_names: list[str]) -> bool:
        """`<Cls>.apply(...)` / `self.<attr>.apply(...)` —— mindspore `_Function` 自定义反向。

        这类节点的 saved 集**不该由 PIN 猜**:源里 `ctx.save_for_backward(...)` 逐字写着。
        故此处发射 `FusedFunction` 节点并把**源真值名单**(由已落地的 `fn_saves` 抽取器给)
        挂到 `attrs["save_for_backward"]`,同时把能按位映射到调用点实参的项记成
        `attrs["saved_operand_idx"]`(其余是 forward 内部张量,记 `saved_internal`)。
        返回 True 表示已处理。"""
        cls_name = None
        if isinstance(func.value, ast.Name):
            cls_name = func.value.id
        else:
            sattr = _self_attr(func.value)
            if sattr is not None:
                cls_name = (self.binds.get(sattr).attrs.get("cell")
                            if self.binds.get(sattr) is not None else None) \
                           or self._self_kinds.get(f"__cls__{sattr}")
        info = self._fn_classes.get(cls_name) if cls_name else None
        if info is None:
            return False
        fparams = list(info.get("forward_params") or ())
        # 源真值的**两条来源合并**:`ctx.save_for_backward(...)` 的名单,与裸
        # `ctx.<attr> = <forward 形参>`(绕过 hooks 但同样保留到反向 —— `pynative/loss/loss.py:136`
        # 的 `ctx.logits = logits`)。两者的内存事实相同,故同等对待;来源分别记在 attrs 上可评审。
        saved = list(info.get("saves") or ())
        bare_params = [n for n in (info.get("bare_ctx_params") or ()) if n not in saved]
        saved = saved + bare_params
        pos_args = list(call.args)
        saved_idx, saved_internal = [], []
        # forward(ctx, p0, p1, ...) 的形参按位对应 apply(a0, a1, ...) 的实参。
        arg_of_param = {p: i for i, p in enumerate(fparams)}
        for nm in saved:
            i = arg_of_param.get(nm)
            if i is None or i >= len(pos_args):
                saved_internal.append(nm)
            else:
                saved_idx.append((nm, i))
        attrs = {
            "function": cls_name,
            "save_for_backward": list(info.get("saves") or ()),
            "saved_internal": saved_internal + list(info.get("bare_ctx_internal") or ()),
            "bare_ctx_tensors": list(info.get("bare_ctx") or ()),
            "bare_ctx_retained": bare_params,
        }
        # 只有**张量**实参进 ins(标量 kernel 参数如 softmax_scale/cmp_ratio 不是激活)。
        operands = self._tensor_operands(pos_args, call.lineno)
        _ = operands
        # saved 名单里能定位到 ins 位置的,记成 ins 下标(供 bprop_rules 逐字取)。
        name_to_ins = {}
        for j, e in enumerate(operands):
            if isinstance(e, ast.Name):
                name_to_ins[e.id] = j
        ins_idx = []
        for nm, i in saved_idx:
            a = pos_args[i]
            key = a.id if isinstance(a, ast.Name) else None
            if key is not None and key in name_to_ins:
                ins_idx.append(name_to_ins[key])
        attrs["saved_ins_idx"] = sorted(set(ins_idx))
        self._emit("FusedFunction", attrs, call.lineno, operands, target_names,
                   self._promote_dtype(operands))
        return True

    def _bind_passthru(self, base, target_names: list[str], lineno: int) -> None:
        """`.to_local()` / `.contiguous()`:目标承接 base 的身份(保 producer 边),不建节点。"""
        if isinstance(base, ast.Name):
            self._bind_name_rhs(base.id, target_names)
            return
        sattr = _self_attr(base)
        if sattr is not None and self._self_kinds.get(sattr) == "param":
            self._note_param_operand(sattr, lineno)
            for t in target_names:
                self._forget(t)
                self.param_aliases.add(t)       # 权重别名,不进 SSA(W2/W3)
                self._param_of[t] = sattr
            return
        for t in target_names:
            self._forget(t)
            self.scalars[t] = _UNKNOWN_SCALAR

    def _handle_subcell_method(self, call: ast.Call, sattr: str, method: str,
                               target_names: list[str]) -> bool:
        """`self.<subcell>.<method>(...)` —— 递归抽取子 Cell 的**指定方法**(不是 construct)。
        实测必需:`self.indexer.forward_before_topk(x_detach, qr_detach)`(csa.py:667/766)
        是 indexer 的**主要算力**(linear_wq_b / compressor / weights proj / RoPE / Hadamard),
        此前整块落 opaque_calls。"""
        binding = self.binds.get(sattr)
        if (binding is None or binding.op != "SubCell"
                or self._subcell_resolver is None):
            return False
        self._inline_subcell(binding.attrs, call, target_names, method=method)
        return True

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

    def _axis_capture(self, op: str, attrs: dict, call: ast.Call, target_names) -> dict:
        """把一次调用的**轴/形状元信息**抠成 int / int 列表 / 未求值符号串。

        为什么必须在发射点记(G4,2026-07-25 并行 agent 实测):没有轴,下游字节解析对这些算子
        只能退化成"直通",于是**不是"未知"而是"错"**:
          * `.sum(dim=1)`(`compressor.py:216`)按直通算 → **8× 过读**;
          * `chunk`(`experts.py:229` / `compressor.py:169`)→ **n× 过读**;
          * `cat([kv_nope, kv_pe], -1)`(`compressor.py:243`)→ 解成 `2n·b·(d−64)` 而非 `n·b·d`。
        纪律不变:抠不出的**一律不写**该键(消费方按缺键落 `unresolved`),**绝不填一个默认轴**
        —— 填错轴比"不知道"危险得多,这正是上面三条的成因。
        """
        out: dict = {}
        prim = (attrs.get("prim") or "").rsplit(".", 1)[-1]
        # ── 归约算子的轴 + keepdim(`mint.sum/mean/max/min/cumsum` 与同名张量方法)────────
        if op == "Elementwise" and attrs.get("reduce"):
            d = self._kw_or_arg_int(call, "dim", 1)
            if d is None:
                d = self._kw_or_arg_int(call, "axis", 1)
            if d is not None:
                out["reduce_dim"] = d
            else:
                elts = None
                for kw in call.keywords:
                    if kw.arg in ("dim", "axis"):
                        elts = self._tuple_elts(kw.value)
                if elts is None and len(call.args) > 1:
                    elts = self._tuple_elts(call.args[1])
                if elts is not None:
                    dims = [self._const_int(e) for e in elts]
                    if all(x is not None for x in dims):
                        out["reduce_dim"] = dims
            for kw in call.keywords:
                if kw.arg == "keepdim" and isinstance(kw.value, ast.Constant):
                    out["keepdim"] = bool(kw.value.value)
        # ── topk 的 k(反向要 scatter 回 k 个位置;shape 也靠它)`indexer.py:262`──────────
        if op == "TopK":
            k = self._kw_or_arg_int(call, "k", 1)
            if k is None and len(call.args) > 1:
                v = self._scalar_of(call.args[1])
                k = v if isinstance(v, int) and not isinstance(v, bool) else None
            if k is not None:
                out["topk_k"] = k
            d = self._kw_or_arg_int(call, "dim", 2)
            if d is not None:
                out["topk_dim"] = d
        # ── 常量产出的 shape(`mint.zeros(shape, dtype)` / `full` / `arange`)────────────
        if op == "Constant" and call.args:
            if prim in ("arange",):
                out["const_shape"] = [self._describe(call.args[0])]
            else:
                elts = self._tuple_elts(call.args[0])
                if elts is not None:
                    out["const_shape"] = [self._describe(e) for e in elts]
                elif not isinstance(call.args[0], (ast.Constant,)):
                    out["const_shape_src"] = self._describe(call.args[0])
        return out

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
        # ── G4(2026-07-25):此前**没有**捕获轴的几种 View —— 缺轴时下游按直通算 = 静默错。
        if kind == "concat":
            # `mint.cat(tensors, dim=-1)`(`compressor.py:243` 的 `cat([kv_nope, kv_pe], -1)`)
            ax = self._kw_or_arg_int(call, "dim", 1)
            if ax is None:
                ax = self._kw_or_arg_int(call, "axis", 1)
            return {"concat_axis": ax} if ax is not None else {}
        if kind == "stack":
            ax = self._kw_or_arg_int(call, "dim", 1)
            return {"stack_axis": ax} if ax is not None else {}
        if kind == "chunk":
            # `mint.chunk(input, chunks, dim)`(`experts.py:229` 的 `self.chunk(fc1_output, 2, -1)`)
            out = {"chunk_targets": list(target_names)}
            n = self._kw_or_arg_int(call, "chunks", 1)
            if n is not None:
                out["chunks"] = n
            d = self._kw_or_arg_int(call, "dim", 2)
            if d is not None:
                out["chunk_dim"] = d
            return out
        if kind == "permute":
            # `mint.permute(input, dims)`:dims 既可是元组实参也可是散开的位置实参
            elts = self._tuple_elts(args[1]) if len(args) >= 2 else None
            if elts is None and len(args) > 2:
                elts = list(args[1:])
            if elts is not None:
                dims = [self._const_int(e) for e in elts]
                if all(x is not None for x in dims):
                    return {"permute_dims": dims}
            return {}
        if kind == "squeeze":
            ax = self._kw_or_arg_int(call, "dim", 1)
            return {"squeeze_axis": ax} if ax is not None else {}
        if kind == "roll":
            out = {}
            sh = self._kw_or_arg_int(call, "shifts", 1)
            if sh is not None:
                out["roll_shifts"] = sh
            d = self._kw_or_arg_int(call, "dims", 2)
            if d is not None:
                out["roll_dims"] = d
            return out
        if kind == "broadcast":
            elts = self._tuple_elts(args[1]) if len(args) >= 2 else None
            return {"broadcast_shape": [U(e) for e in elts]} if elts is not None else {}
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
        """按 MRO 找最贴近的 `def name`,返回 `(FunctionDef|None, 定义它的文件相对路径|None)`。

        **跨文件(P0#3)**:有 `class_index` 时用它顺 import 解析基类
        (`DSv4HybridSelfAttention` → `MultiLatentAttention` 在 `multi_latent_attention.py`);
        没有时退化为旧的"同文件基类 BFS"(既有 DSv3 路径行为逐字不变)。
        """
        if self._tree is None or self._cls_name is None:
            return None, None
        if self._class_index is not None and self._cls_rel:
            for rc in self._class_index.mro(self._cls_name, self._cls_rel):
                m = next((n for n in rc.node.body
                          if isinstance(n, ast.FunctionDef) and n.name == name), None)
                if m is not None:
                    return m, rc.rel
            return None, None
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
                return m, None
            for b in cls.bases:
                if isinstance(b, ast.Name):
                    queue.append(b.id)
        return None, None

    def _lookup_super_method(self, name: str):
        """`super().<name>` —— 按 MRO 跳过**已经在走的那一层**,取更靠基类的那个 `def name`。

        跳过数 = `1 + 当前内联链里 <name> 出现的次数`:顶层走 `SharedExpertMLP.construct` 时
        跳 1 层拿到 `MLP.construct`;若 `MLP.construct` 里再 `super().construct(...)`,
        此时 `construct` 已在 `_inline_stack` 里 → 跳 2 层。**绝不**回到同一层
        (那会是无穷内联,而 `_inline_method` 的递归门只会把它变成一个 fail-loud)。
        """
        skip = 1 + self._inline_stack.count(name)
        if self._class_index is not None and self._cls_rel and self._cls_name:
            for rc in self._class_index.mro(self._cls_name, self._cls_rel):
                m = next((n for n in rc.node.body
                          if isinstance(n, ast.FunctionDef) and n.name == name), None)
                if m is None:
                    continue
                skip -= 1
                if skip < 0:
                    return m, rc.rel
            return None, None
        # 无 class_index:退化为同文件基类 BFS(既有单文件路径行为不变)
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
            m = next((n for n in cls.body
                      if isinstance(n, ast.FunctionDef) and n.name == name), None)
            if m is not None:
                skip -= 1
                if skip < 0:
                    return m, None
            for b in cls.bases:
                if isinstance(b, ast.Name):
                    queue.append(b.id)
        return None, None

    def _inline_method(self, method: ast.FunctionDef, call: ast.Call, target_names: list[str],
                       method_rel: str | None = None, is_module_func: bool = False) -> None:
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
        # 跨文件内联:节点 src 必须指向**真正定义该方法的文件**(否则 file:line 是错的)。
        prev_file, prev_rel = self.src_file, self._cls_rel
        if method_rel:
            self.src_file = method_rel.rsplit("/", 1)[-1]
        ret = None
        try:
            self.walk_body(body_copy)
        except _ReturnSignal as sig:
            ret = sig.value
        finally:
            self._inline_stack.pop()
            self.src_file, self._cls_rel = prev_file, prev_rel
        self._bind_return(ret, target_names)

    def _bind_params(self, method: ast.FunctionDef, call: ast.Call, frame: int) -> dict:
        params = [a.arg for a in method.args.args if a.arg != "self"]
        pos = list(call.args)
        kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
        defaults = self._fn_defaults(method)
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
                continue
            local = f"{p}__i{frame}"          # 非 Name 实参 / 缺省 → 帧内合成局部名
            rename[p] = local
            # 帧内占位名也要有**身份**:标量实参(`self.softmax_scale` / 字面量 / config)记标量,
            # 未传且缺省 None 的记 known_none —— 否则下游 `if x is None` / `y * scale` 判不出来。
            if arg is None:
                if p in defaults:
                    if defaults[p] is None:
                        self.known_none.add(local)
                    else:
                        self.scalars[local] = defaults[p]
                continue
            k = self._classify(arg)
            if k == _Kind.SCALAR:
                self.scalars[local] = self._scalar_of(arg)
            elif k == _Kind.NONE:
                self.known_none.add(local)
            elif k == _Kind.DTYPE:
                self.dtypes[local] = _dtype_name(arg) or "fp32"
            elif k in (_Kind.TENSOR, _Kind.PARAM):
                mat = self._materialize_expr(arg, getattr(call, "lineno", 0))
                if isinstance(mat, ast.Name):
                    rename[p] = mat.id        # 张量表达式实参:先物化,再按 Name 共享 SSA
        return rename

    @staticmethod
    def _fn_defaults(method: ast.FunctionDef) -> dict:
        args = [a.arg for a in method.args.args if a.arg != "self"]
        defaults = method.args.defaults
        out: dict = {}
        n, nd = len(args), len(defaults)
        for i, a in enumerate(args):
            j = i - (n - nd)
            if j >= 0 and isinstance(defaults[j], ast.Constant):
                out[a] = defaults[j].value
        return out

    def _bind_return(self, ret_expr, target_names: list[str]) -> None:
        if ret_expr is None or not target_names:
            return
        elts = ret_expr.elts if isinstance(ret_expr, (ast.Tuple, ast.List)) else [ret_expr]
        for tgt, e in zip(target_names, elts):
            if isinstance(e, ast.Name):
                self._bind_name_rhs(e.id, [tgt])
            elif isinstance(e, ast.Constant):
                # 内联函数 `return None`(如 `parse_cu_seqlens` 的 `actual_seq_len is None` 支,
                # csa.py:325)—— 目标必须进 `known_none`,否则下游 `if cu_seqlens is not None:`
                # 判不出来 → fail-loud(此前正是这样卡住 naive 支的)。
                self._forget(tgt)
                if e.value is None:
                    self.known_none.add(tgt)
                else:
                    self.scalars[tgt] = e.value

    # ---- 子 Cell 递归内联 ----
    def _inline_subcell(self, attrs: dict, call: ast.Call, target_names: list[str],
                        method: str = "construct") -> None:
        """把 resolver 递归抽出的子 DAG 内联到调用点:
          1) 子 construct 形参按位重映射到调用方实参(Name 实参→复用其 SSA ref 与 producer);
          2) 子节点 id 统一加偏移(接父 _next_id,不从 1 重启),子内部边同偏移平移;
          3) 子叶子消费的"形参操作数"→ 改写成调用方 ref,并补父 producer→子叶子的跨界边;
          4) 子返回值 → 绑回调用点赋值目标(下游消费即连"子输出→消费者"边)。"""
        sub = self._call_subcell_resolver(attrs, method)

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
            if not isinstance(arg, ast.Name) and arg is not None:
                # 非 Name 的**张量**实参先物化成节点(否则边被静默切断)。实测关键点:
                # `self.unfused_indexer_loss(..., ops.stop_gradient(query),
                #  ops.stop_gradient(compressed_kv), ...)`(csa.py:794-795)—— 内联实参形的
                # detach,此前既不建 Detach 节点、也不连边,子里拿到的是凭空的占位 ref。
                if self._classify(arg) in (_Kind.TENSOR, _Kind.PARAM):
                    mat = self._materialize_expr(arg, getattr(call, "lineno", 0))
                    if isinstance(mat, ast.Name):
                        arg = mat
            if isinstance(arg, ast.Name):
                ref = self.ssa.get(arg.id, f"{arg.id}:?:bf16")
                param_map[p] = (ref, self.producer.get(arg.id))
            else:
                param_map[p] = (f"{p}:?:bf16", None)  # 非 Name 实参 / 未传 → 占位,不连边

        # 2)+3) 偏移平移 + 形参操作数重映射 + 跨界边
        offset = self._next_id - 1
        # ── G3/G5:本次内联的**帧**标签 ────────────────────────────────────────────────
        # 帧 = 「哪个构造点的哪一次内联」。根帧是空串(父自己的节点)。嵌套时把子里已有的帧
        # 追加在后面(`indexer@36/compressor@42`),于是每个节点的 `self.<attr>` / 局部标量名
        # 都在**它自己那个类**的环境里解 —— 这正是扁平 `merge_dims_ctx` 做不到的事。
        frame = f"{attrs.get('field') or attrs.get('cell') or 'sub'}@{offset}"
        sub_dims = dict(getattr(sub, "dims_ctx", {}) or {})
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
            new_attrs = dict(n.attrs)
            inner = new_attrs.get("frame")
            new_attrs["frame"] = f"{frame}/{inner}" if inner else frame
            # 更深的帧已经带着自己那份 dims_ctx(setdefault 语义):只给还没有的补本类那份。
            if sub_dims and "dims_ctx" not in new_attrs:
                new_attrs["dims_ctx"] = sub_dims
            self.nodes.append(OpNode(
                id=new_id, op=n.op, src=n.src, module=n.module,
                ins=new_ins, out=n.out, attrs=new_attrs,
            ))
        # G5:子的 construct 局部标量绑定按帧上浮。`src` 若是子的 construct 形参 → 换成调用方
        # 那一侧的**基名**(不然 `_fill_scalar_bind` 在父的 env 里查不到子形参名)。
        for sb in (getattr(sub, "scalar_binds", ()) or ()):
            src = sb.get("src", "")
            if src in param_map:
                src = param_map[src][0].split(":")[0]
            inner = sb.get("frame")
            self.scalar_binds.append({
                "names": list(sb.get("names", [])), "src": src,
                "frame": f"{frame}/{inner}" if inner else frame,
            })
        for s, d in sub.edges:
            self.edges.append([s + offset, d + offset])
        self._next_id = offset + len(sub.nodes) + 1
        # T0-6.5 Fix2:子 walker 的 opaque_calls 原样并入(src 已指子文件,无需重映射)——不这样做
        # 的话子 Cell 边界内的 fallthrough 调用会在父 DAG 视角下二次静默丢(违反 Fix2 初衷)。
        self.opaque_calls.extend(sub.opaque_calls)
        # Task 2:抽取诊断同理逐类并入(否则子 Cell 边界会把「子里丢了一大块」洗白)。
        for kind, items in (sub.diagnostics or {}).items():
            self.diagnostics.setdefault(kind, []).extend(items)

        # 4') 子里被 detach 的产物名(no-grad 块 / stop_gradient)上浮(源已是子文件内的名字)。
        for nm in getattr(sub, "detached", ()) or ():
            if nm not in self.detached:
                self.detached.append(nm)
        for rec in getattr(sub, "param_operands", ()) or ():
            if rec not in self.param_operands:
                self.param_operands.append(rec)

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

    def _call_subcell_resolver(self, attrs: dict, method: str):
        """兼容两种 resolver 签名:新的带 `method=`/`injected_binds=`,旧的三位置参数
        (既有测试自带的 resolver)。

        `injected_binds`:把**父这一侧已绑好的** `self.<attr>` 顺着构造点的关键字实参
        (`rotary_pos_emb=self.rotary_pos_emb`)传给子,让子的
        `self.rotary_pos_emb = rotary_pos_emb` 绑得上(见 init_binder 形态 5)。"""
        cell, field, bare = attrs.get("cell"), attrs.get("field"), attrs.get("bare", False)
        injected = {kw: self.binds[a] for kw, a in (attrs.get("kw_self") or {}).items()
                    if a in self.binds}
        spec = attrs.get("spec")        # 调用点手搭的 submodules(见 _inline_submodules_spec)
        # G2:`build_module(...)` / 直接实例化的**构造点维度关键字**(extractor._ctor_seeds 求出)。
        # 它决定子 `__init__` 里 `proj_out_dim = self.coff * head_dim` 这类维度的真值 ——
        # 同一个类的两个构造点 head_dim 差 4× 就靠这条区分。
        seeds = attrs.get("ctor_seeds") or None
        try:
            return self._subcell_resolver(cell, field, bare, method=method,
                                          injected_binds=injected, spec=spec,
                                          ctor_seeds=seeds)
        except TypeError:
            pass
        try:
            return self._subcell_resolver(cell, field, bare, method=method,
                                          injected_binds=injected, spec=spec)
        except TypeError:
            pass
        try:
            return self._subcell_resolver(cell, field, bare, method=method,
                                          injected_binds=injected)
        except TypeError:
            if method and method != "construct":
                raise
            return self._subcell_resolver(cell, field, bare)

    def _resolve_returns(self, body) -> list:
        """定位 construct 的返回值并逐项分类:
           Name 且已被某节点产出 → ("node", producer id);Name 且是形参 → ("param", 名);其它 → ("none", None)。

        剪枝上下文下**优先用实际命中的那条 return**(`_taken_return`);否则退化为"最后一条
        return"(既有非剪枝行为)。"""
        if self._taken_return is not None:
            return self._classify_return_value(self._taken_return)
        rets: list[ast.Return] = []
        for top in body:
            for n in ast.walk(top):
                if isinstance(n, ast.Return) and n.value is not None:
                    rets.append(n)
        if not rets:
            return []
        ret = max(rets, key=lambda r: getattr(r, "lineno", 0))
        return self._classify_return_value(ret.value)

    def _classify_return_value(self, val) -> list:
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
            if test.id in self.scalars:
                v = self.scalars[test.id]
                if v is _POSITIVE_DIM:
                    return True                 # 轴长 >= 1 → 真(公理)
                return _UNDECIDED if v is _UNKNOWN_SCALAR else bool(v)
            return _UNDECIDED
        if isinstance(test, ast.Call):
            return self._eval_predicate_call(test)
        return _UNDECIDED

    def _eval_predicate_call(self, call: ast.Call):
        """`hasattr(x, "attr")` / `isinstance(x, Cls)` —— **部署形态**谓词(DTensor 分支)。

        真源到处用它们做「参数是不是 DTensor / 张量有没有 to_local」的分支
        (`csa.py:467/684`、`compressor.py:208`、`deepseek_v4_hybrid_attention.py:281`、
        `indexer.py:284`)。这些**不是** config,而是运行时部署形态 → 必须由调用方显式给值
        (键 `hasattr:<attr>` / `isinstance:<Cls>`),缺键 → `_UNDECIDED` → fail-loud。
        **绝不默认取某一支**:两支虽常常字节等价,但"常常"不是"总是"。
        """
        f = call.func
        if not isinstance(f, ast.Name) or f.id not in ("hasattr", "isinstance"):
            # **零参具名谓词** `<name>()`:同一档的"运行时相位"事实,键 `call:<name>`。
            # 真源必需:`if not is_in_recompute():`(`moe/moe_layer.py:126`,
            # 定义在 `pynative/distributed/activation_checkpoint.py`)—— 它区分「首次前向」
            # 与「重算中的前向」。前向图建模的是**首次前向**,故调用方给
            # `{"call:is_in_recompute": False}`;不给 → fail-loud 指名要哪个键(不猜相位)。
            if isinstance(f, ast.Name) and not call.args and not call.keywords:
                k = f"call:{f.id}"
                if k in self.runtime_predicates:
                    return bool(self.runtime_predicates[k])
                self._pending_predicate = k
            return _UNDECIDED
        if len(call.args) < 2:
            return _UNDECIDED
        second = call.args[1]
        if f.id == "hasattr":
            key = second.value if isinstance(second, ast.Constant) else None
            if not isinstance(key, str):
                return _UNDECIDED
            k = f"hasattr:{key}"
        else:
            cls = None
            if isinstance(second, ast.Name):
                cls = second.id
            elif isinstance(second, ast.Attribute):
                cls = second.attr
            if cls is None:
                return _UNDECIDED
            k = f"isinstance:{cls}"
        if k in self.runtime_predicates:
            return bool(self.runtime_predicates[k])
        self._pending_predicate = k
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
            pd = self._cmp_positive_dim(op, lk, lv, rk, rv)
            if pd is not None:
                return pd
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
            # `"ep" not in <x>.device_mesh.mesh_dim_names`(`moe/experts.py:181`):右侧是
            # **部署形态**(device mesh 的维名单),源里读不出来 —— 由调用方在
            # `runtime_predicates["mesh_dim_names"]` 里显式给(它就是 yaml parallelism 段的
            # 事实,和 `isinstance:DTensor` 同一档)。缺键 → fail-loud 并指名要哪个键。
            if isinstance(right, ast.Attribute) and right.attr == "mesh_dim_names":
                dims = self.runtime_predicates.get("mesh_dim_names")
                if dims is None:
                    self._pending_predicate = "mesh_dim_names"
                    return _UNDECIDED
                if not lk:
                    return _UNDECIDED
                inside = lv in tuple(dims)
                return inside if isinstance(op, ast.In) else (not inside)
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

    @staticmethod
    def _cmp_positive_dim(op, lk, lv, rk, rv):
        """`<轴长> >/>= <字面量>` 的判定(仅当结论对**任何** >=1 的取值都成立时才给结论)。

        实测用途:`if ratio > 1 and n_compressed > 0:`(csa.py:762)—— `n_compressed` 是
        `int(compressed_kv.shape[0])`,值未知但 >=1 → `> 0` 恒真。反过来 `sq < ratio`
        (compressor.py:190)对 >=1 的 sq 既可能真也可能假 → 仍 `_UNDECIDED`(不猜)。
        """
        def num(k, v):
            return isinstance(v, (int, float)) and not isinstance(v, bool) and k

        if lv is _POSITIVE_DIM and num(rk, rv):
            if isinstance(op, ast.Gt):
                return True if rv < 1 else None
            if isinstance(op, ast.GtE):
                return True if rv <= 1 else None
            if isinstance(op, (ast.Lt, ast.LtE)):
                return False if rv <= 1 and isinstance(op, ast.Lt) else None
            return None
        if rv is _POSITIVE_DIM and num(lk, lv):
            if isinstance(op, ast.Lt):
                return True if lv < 1 else None
            if isinstance(op, ast.LtE):
                return True if lv <= 1 else None
            return None
        return None

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
            if node.id in self.dtypes:
                # **dtype 记号**(`ori_dtype = input_.dtype`,`pynative/layers/linear.py:121`)
                # 参与比较:`if output.dtype != ori_dtype:`(`:143`)门控最后那个 cast。
                return True, self.dtypes[node.id]
            if self.scalars.get(node.id) is _POSITIVE_DIM:
                return True, _POSITIVE_DIM
            # 已由某个节点产出的名字 = 一个**真张量** → 对 `is not None` 判定必为"存在"。
            # 这不是猜:它有 producer,说明源里刚刚算出了它。
            #   `compressed_kv = self.compressor(x) if self.enable_compress else None`
            #   → `if compressed_kv is not None:`(csa.py:676/746)
            if node.id in self.producer or node.id in self.ssa:
                return True, _PRESENT
            if node.id in self.scalars:
                v = self.scalars[node.id]
                return (False, None) if v is _UNKNOWN_SCALAR else (True, v)
            if node.id in self.param_defaults:
                return True, self.param_defaults[node.id]
        if isinstance(node, (ast.BinOp, ast.UnaryOp)) and self._classify(node) == _Kind.SCALAR:
            v = self._scalar_of(node)
            return (False, None) if v is _UNKNOWN_SCALAR else (True, v)
        if isinstance(node, ast.Call) and self._classify(node) == _Kind.SCALAR:
            v = self._scalar_of(node)
            return (False, None) if v is _UNKNOWN_SCALAR else (True, v)
        if isinstance(node, ast.Attribute):
            if node.attr == "ndim":
                v = self._scalar_of(node)
                if v is not _UNKNOWN_SCALAR:
                    return True, v
            if node.attr == "dtype":
                d = self._dtype_value(node.value)
                if d is not None:
                    return True, d
            sattr = _self_attr(node)
            if sattr is not None:
                if self._self_kinds.get(sattr) == "param":
                    return True, _PRESENT       # Parameter 恒存在
                if sattr in self.binds:
                    # 已绑成算子的子模块/原语(`self.rotary_pos_emb` 绑成 Constant{rope_freqs},
                    # deepseek_v4_hybrid_attention.py:80)→ 它**存在**。
                    # `if self.rotary_pos_emb is not None:`(:256 / compressor.py:220)
                    return True, _PRESENT
        return False, None

    def _dtype_value(self, base):
        """`<base>.dtype` 的**短标签**;判不出返回 None。

        真源必需(2026-07-25):`pynative/layers/linear.py:126/128/143` 用
        `weight.dtype != self.compute_dtype` / `input_.dtype != self.compute_dtype` /
        `output.dtype != ori_dtype` 把 cast **门控**掉。判不出这三条,lm_head 的 vocab 投影
        (全模型最大的 GEMM)一个节点都抽不出来。
        判据全来自已知事实:
          * 权重(`self.<Parameter>` / 权重别名)→ `config_flags["params_dtype"]`;
          * 已产出的 SSA 变量 → 它 ref 里的 dtype 段;
          * dtype 环境里的名字(`ori_dtype = input_.dtype`)→ 该记号;
          * construct 形参 → `config_flags["compute_dtype"]`(与 `_resolve_cast_dtype` 同口径)。
        """
        if isinstance(base, ast.Name):
            if base.id in self.param_aliases:
                return _DTYPE_ALIAS.get(self._param_dtype(base.id), self._param_dtype(base.id))
            ref = self.ssa.get(base.id)
            if ref and ref.count(":") == 2:
                return ref.split(":")[2]
            if base.id in self.dtypes:
                return self.dtypes[base.id]
            if base.id in self._param_set:
                d = self.config_flags.get("compute_dtype", "bf16")
                return _DTYPE_ALIAS.get(d, d)
            return None
        sattr = _self_attr(base)
        if sattr is not None and self._self_kinds.get(sattr) == "param":
            return _DTYPE_ALIAS.get(self._param_dtype(sattr), self._param_dtype(sattr))
        return None

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
            # `self.cast(y, ori_dtype)`(multi_latent_attention.py:307):dtype 环境里查真 dtype。
            # 此前 `_dtype_name` 对裸 Name 直接返回该**变量名**("ori_dtype")= 一个假 dtype。
            if isinstance(a, ast.Name) and a.id in self.dtypes:
                return self.dtypes[a.id]
            # `<张量>.dtype`(`ops.cast(invalid_mask, scores.dtype)`,csa.py:502)——
            # 同理:此前 `_dtype_name` 返回字面串 "dtype",也是个假 dtype。
            if (isinstance(a, ast.Attribute) and a.attr == "dtype"
                    and isinstance(a.value, ast.Name)):
                ref = self.ssa.get(a.value.id)
                if ref and ref.count(":") == 2:
                    return ref.split(":")[2]
                if a.value.id in self.dtypes:
                    return self.dtypes[a.value.id]
                if a.value.id in self._param_set:
                    # construct 形参的 dtype:按 compute_dtype(`ops.cast(output, query.dtype)`,
                    # csa.py:521)。形参的真 dtype 要靠 infer_shapes 的种子(P2),此处不杜撰别的。
                    return self.config_flags.get("compute_dtype", "bf16")
            d = _dtype_name(a)
            if d in _KNOWN_DTYPES:
                return d
            if d is not None:
                # **假 dtype 的最后一道门**(2026-07-25):`_dtype_name` 对判不出的表达式会把
                # 表达式里的**名字**当 dtype 返回。实测:`x = self.cast(x, self.dtype)`
                # (`hyper_connection.py:406`,`self.dtype = config.compute_dtype` @ `:205`)
                # —— `self.dtype` 不在 config_flags 里时,此前得到字面串 `"dtype"`,一个
                # **不存在的 dtype**,它会顺着 SSA 传播到下游每个节点的 ref 上、进而进 saves
                # (字节按未知 dtype 计)。不许静默。
                raise ValueError(
                    f"cast 的目标 dtype 解析出一个**不是 dtype 的记号** {d!r}"
                    f"（{self.src_file}:{getattr(a, 'lineno', '?')}，表达式 "
                    f"`{self._describe(a)}`）—— 请在 config_flags 里显式给它的值"
                    f"（如 `self.dtype = config.compute_dtype` → 注入 `dtype`），"
                    f"绝不让一个假 dtype 顺着 SSA 传下去，fail-loud"
                )
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
        ins_slots: list[int] = []    # 每个存活 `ins` 项的**张量操作数位序**(见 attrs["ins_slots"])
        tslot = -1                   # 已见的"张量类操作数"计数(权重也计,非张量字面量不计)
        seen_prod: set[int] = set()  # 同一节点内对同一 producer 只连一条边(去重)
        pending_prods: list[int] = []  # producer id(节点 id 尚未分配,先收集,分配后统一补边)
        for a in arg_exprs:
            if isinstance(a, ast.Call):
                # 嵌套 Call 实参:必须先于本节点分配 id(保证 id/nodes 列表顺序与真实数据流一致——
                # 下游 infer_shapes 等按 dag.nodes 顺序做正向传播,依赖 producer 先于 consumer 出现)。
                a = self._materialize_call_arg(a)
            elif isinstance(a, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.Subscript)):
                # 张量子表达式实参:同样先物化(否则整条子算子链被静默丢,见本方法上方注释)。
                if self._classify(a) in (_Kind.TENSOR, _Kind.PARAM):
                    a = self._materialize_expr(a, lineno) or a
            if isinstance(a, ast.Attribute):
                # `self.<Parameter>` 操作数(`unfused_compressed_sparse_attn(..., self.attn_sink, ...)`
                # @ csa.py:824):此前**静默丢出 ins**。现单列 param_operands 使其可见 —— 仍不进
                # `ins`,因为把权重塞进 ins 会让 `derive_saves` 把它当激活 save 计
                # (实测 FFNGroupedGEMM「236 MiB」里 88 MiB 就是这个病,评估文档 §7.2)。
                # 完整的 `is_weight`/`op.params` 建模是 P1#14,不在本轮。
                sattr = _self_attr(a)
                if sattr is not None and self._self_kinds.get(sattr) == "param":
                    self._note_param_operand(sattr, lineno)
                    tslot += 1              # 权重**占一个张量操作数位**(只是不进 ins)
                continue
            if not isinstance(a, ast.Name):
                continue  # 字面量/属性/未产节点的 opaque 嵌套调用:非可追踪张量操作数,略过
            if a.id in self.param_aliases:
                # 权重(Parameter)别名:**不进 ins**(W2/W3 契约),单列 param_operands。
                self._note_param_operand(self._param_of.get(a.id, a.id), lineno)
                tslot += 1                  # 同上:占位,使后续操作数的张量位序不左移
                continue
            if (a.id not in self.ssa
                    and (a.id in self.scalars or a.id in self.dtypes
                         or a.id in self.known_none)):
                continue  # 已知是标量 / dtype 记号 / None —— **不是激活**,绝不当张量操作数
            tslot += 1
            if a.id in self.ssa:  # 已知 SSA 中间变量 → 用其当前 ref 并向其 producer 连边
                ins.append(self.ssa[a.id])
                ins_slots.append(tslot)
                prod = self.producer.get(a.id)
                if prod is not None and prod not in seen_prod:
                    pending_prods.append(prod)
                    seen_prod.add(prod)
            else:  # 方法形参 / 未知名 → 占位 ref(shape 未知,dtype 缺省 bf16)
                ins.append(f"{a.id}:?:bf16")
                ins_slots.append(tslot)
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
        if out_dtype is None:
            # 由已解析的 ins 推(dtype 保持类算子:View/Elementwise/Where/IndexSelect/BMM/...)。
            # dtype 提升序(float > int > bool,与 numpy/mindspore 的提升规则同向)。
            rank = {"bool": 0, "uint8": 1, "int8": 1, "int16": 2, "int32": 3, "int64": 4,
                    "fp16": 5, "bf16": 5, "fp32": 6, "fp64": 7}
            best, best_r = None, -1
            for ref in ins:
                if ref.count(":") != 2:
                    continue
                dt = ref.split(":")[2]
                if rank.get(dt, 1) > best_r:
                    best, best_r = dt, rank.get(dt, 1)
            out_dtype = best or self.config_flags.get("compute_dtype", "bf16")
        out_ref = f"{target_names[0]}:?:{out_dtype}" if target_names else ""
        attrs = dict(attrs)
        # ── `ins` 位置 → **张量操作数位序**的映射(2026-07-25)──────────────────────────
        # `bprop_rules.PIN` 的 `inputs:[k]` 一直是按「张量操作数位序」写的:
        # `mint.gather(input, dim, index)` 的 `dim` 是 int(非张量,不计位),故 index 记作 `[1]`;
        # advanced indexing `kv_flat[flat_indices]`(csa.py:485)的 index 同样是 `[1]`。
        # **权重被路由去 `param_operands` 后 `ins` 会左移**,这个位序就对不上了 —— 实测
        # `self.embedding(weight, 0, input_)`(`vocab_embedding.py:85`)的 `ins` 只剩
        # `[input_]`,PIN 的 `[1]` 越界 → **saved 集变成空**,而那条 gather 的 index
        # (tile 后的 `[B·S, embedding_dim]` int 张量)是真 saved 张量。
        # 故记下每个存活 `ins` 项的张量操作数位序,让 `derive_saves` 按它取。
        # 只在**确有张量位被略过**时才写(位序恒等时不写 = 既有节点 attrs 逐字不变)。
        if ins_slots and ins_slots != list(range(len(ins_slots))):
            attrs["ins_slots"] = list(ins_slots)
        if len(target_names) > 1:
            # 多输出算子(`topk_scores, topk_indices = self.topk(...)`):`out` 只放首目标,
            # 其余目标此前完全不可见 —— 而 topk 的**反向恰恰要存第 2 个输出(indices)**。
            attrs["outs"] = [f"{t}:?:{'int32' if op == 'TopK' and i else out_dtype}"
                             for i, t in enumerate(target_names)]
        # ── 块级 / 逐点 detach(P0#4 + P1#11)────────────────────────────────────────
        # `with _no_grad():` 块内产物 与 `ops.stop_gradient(...)` 的产物:**保边、打标**。
        # 打标而不是"不发节点":下游消费方要能看见"这张张量的上游被切断了"(grad 可达性),
        # 而不是看见一个凭空出现的、看起来梯度可达的张量。
        detached = bool(self._nograd_depth) or op == "Detach"
        if detached:
            attrs["detached"] = True
            if self._nograd_depth:
                attrs["no_grad_region"] = True
            for t in target_names:
                if t not in self.detached:
                    self.detached.append(t)
        line_params = []
        if self.param_operands and op != "Detach":
            line_params = [r["param"] for r in self.param_operands
                           if r["src"] == f"{self.src_file}:{lineno}"]
            if line_params:
                attrs["param_operands"] = line_params
        # 权重派生的 `ins` 项(见 `_weight_derived`)→ 标出来,`derive_saves` 排除它们。
        wix = [i for i, ref in enumerate(ins)
               if ref.split(":")[0] in self._weight_derived]
        if wix:
            attrs["weight_ins_idx"] = wix
        node = OpNode(
            id=node_id, op=op, src=f"{self.src_file}:{lineno}",
            module=attrs.get("module", ""), ins=ins, out=out_ref, attrs=attrs,
        )
        self.nodes.append(node)
        # 产出是"权重派生量"(不是激活)的两种情形,**都要求没有任何激活参与**:
        #   ① `ins` 为空且本行有权重操作数:`w1 = cast(self.weight1, ...)`(ffn.py:146);
        #   ② `ins` 非空但**全部**是已标记的权重派生量:`w1 = reshape(w1, ...)`(ffn.py:157)
        #      —— 不传播这一步,链就断了,`w1` 会在下一跳的 GroupedMatmul 里重新被当成激活
        #      (实测:那正是 88/236 MiB 的来源)。
        derived = ((not ins) and bool(line_params)) or (
            bool(ins) and all(r.split(":")[0] in self._weight_derived for r in ins))
        # 多输出算子:**每个输出用它自己的 dtype**(T3,2026-07-25)。
        # 此前所有目标都按同一个 `out_dtype` 进 SSA,于是
        #   `topk_scores, topk_indices = self.topk(...)`(`indexer.py:262`)的 `topk_indices`
        # 在 SSA 里是 bf16(而 `attrs["outs"]` 里正确写着 int32),下一行
        #   `topk_indices = self.cast(topk_indices, int32)`(`:263`)的 `ins` 就带着**错的 dtype**;
        # `derive_saves` 按名去重只留一个 → 到底留下 int32 还是 bf16 取决于谁先被 pin,
        # 也就是**静默取错分支**(与 op 类型拼错同一类)。现按 `outs` 逐个登记。
        outs_refs = attrs.get("outs")
        for i, t in enumerate(target_names):
            dt = out_dtype
            if outs_refs and i < len(outs_refs) and outs_refs[i].count(":") == 2:
                dt = outs_refs[i].split(":")[2]
            self.ssa[t] = f"{t}:?:{dt}"       # dtype 随 SSA 传播(Cast 会改写)
            self.producer[t] = node_id
            if derived:
                self._weight_derived.add(t)
            else:
                self._weight_derived.discard(t)
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
    **kw,
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
        strict=strict, **kw,
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
    **kw,
):
    """同 walk_construct,但额外返回 (OpDAG, construct 形参名列表, 返回值分类)——供上层递归内联子 Cell。"""
    return _run_walker(
        src, cls_name, binds, src_file,
        config_flags=config_flags, none_vars=none_vars,
        param_defaults=param_defaults, present_vars=present_vars,
        subcell_resolver=subcell_resolver, method_aliases=method_aliases,
        strict=strict, **kw,
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


def _took_identity_return(walker, dag) -> bool:
    """剪枝后**实际命中**的 `return` 就是「原样返回入参」→ 本 config 下该 Cell 可证明是恒等。

    0 节点硬门要抓的是「整段被丢了」,不是「config 选中了一条恒等路径」。真源必需
    (2026-07-25):`Dropout.construct` 的 `if not self.training or not self.use_dropout:
    return x`(`pynative/layers/dropout.py:74-75`)—— yaml 未给 `hidden_dropout`(默认 0.0)
    ⇒ `use_dropout = drop_prob != 0` 为 False ⇒ 真机上这个 Cell **就是**恒等,一个 kernel
    都不发。此前会被 0 节点门当「全丢了」抛掉。

    三重收紧,免得它变成"丢了也放行"的后门:
      ① 必须在**剪枝上下文**里且确实**走到**了一条 `return`(`_taken_return` 非 None);
      ② 返回值必须是 construct 的**形参名**(或 None 常量)—— 即真的原样传回,不是某个
         没被建出来的中间量;
      ③ 诊断与 `opaque_calls` 必须**全空** —— 有任何"看不懂"就不许走这条豁免。
    """
    if not walker._pruning or walker._taken_return is None:
        return False
    val = walker._taken_return
    ok_val = (isinstance(val, ast.Name) and val.id in walker._param_set) or (
        isinstance(val, ast.Constant) and val.value is None)
    if not ok_val:
        return False
    s = diagnostics_summary(dag)
    return s["total"] == 0 and s["opaque_calls"] == 0


def _run_walker(
    src, cls_name, binds, src_file, *,
    config_flags=None, none_vars=None, param_defaults=None,
    present_vars=None, subcell_resolver=None, method_aliases=None, strict=False,
    class_index=None, cls_rel=None, module_funcs=None, runtime_predicates=None,
    fn_classes=None, self_kinds=None, param_literals=None, alias_unknown=None,
    host_call_allow=(), kernel_call_allow=(), kernel_saves=None, param_cells=None,
    input_axes=None, module_consts=None,
    entry_method="construct",
):
    tree = ast.parse(src)
    if next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name), None) is None:
        raise ValueError(f"源码里找不到 class {cls_name}(fail-loud)")

    walker = _Walker(
        binds, src_file, config_flags, none_vars, param_defaults, present_vars,
        tree=tree, cls_name=cls_name, subcell_resolver=subcell_resolver,
        method_aliases=method_aliases, strict=strict,
        class_index=class_index, cls_rel=cls_rel, module_funcs=module_funcs,
        runtime_predicates=runtime_predicates, fn_classes=fn_classes,
        self_kinds=self_kinds, param_literals=param_literals,
        alias_unknown=alias_unknown, host_call_allow=host_call_allow,
        kernel_call_allow=kernel_call_allow, kernel_saves=kernel_saves,
        param_cells=param_cells,
        input_axes=input_axes, module_consts=module_consts,
    )
    construct, _crel = walker._lookup_method(entry_method)  # 支持定义在基类(含跨文件)
    if construct is None:
        raise ValueError(
            f"class {cls_name}(及其基类)缺少 {entry_method} 方法(fail-loud)")

    walker.construct_params = [a.arg for a in construct.args.args if a.arg != "self"]
    walker._param_set = set(walker.construct_params)
    walker.walk_body(construct.body)
    walker.returns = walker._resolve_returns(construct.body)
    dag = OpDAG(cell=cls_name, nodes=walker.nodes, edges=walker.edges,
                scalar_binds=walker.scalar_binds, opaque_calls=walker.opaque_calls,
                diagnostics=walker.diagnostics,
                detached=list(walker.detached), deletes=list(walker.deletes),
                param_operands=list(walker.param_operands))

    # ── 0 节点硬门(Task 2 / P0#1;**恒开**,与 strict 无关)────────────────────────────────
    # 实测反例:`CSAIndexer` fused 支整块在 `with _no_grad():`(indexer.py:214)里 → 整块丢 →
    # 此前返回 `ok, 0 nodes / 0 opaque`,若接进数字链路会**贡献 0 字节而不报错**(评估文档 §3.1
    # 「最危险的一条」)。有非平凡 body 却 0 节点 = 全丢了,绝不许看起来像成功。
    if not walker.nodes and not _body_is_trivial(construct.body) \
            and not _took_identity_return(walker, dag):
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
