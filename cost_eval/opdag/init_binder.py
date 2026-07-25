# cost_eval/opdag/init_binder.py
"""Pass B(设计 §3.2):解析 Cell.__init__ 的 AST,建 self.<name> → Binding(op 类型,attrs)。
按 import 的类名解析规范 op 类型;链式(.recompute/.shard)剥到基 Call。未知类名 fail-loud。"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field

# 规范 op 类型映射(mindspore 类名 → 我们的 op 类型)。linear=反向是否线性。
_CLS2OP = {
    "AddExt": ("Elementwise", {"linear": True}),  "Add": ("Elementwise", {"linear": True}),
    "Sub":   ("Elementwise", {"linear": True}), "SubExt": ("Elementwise", {"linear": True}),
    "Mul":   ("Elementwise", {"linear": False}),
    "Cast":  ("Cast", {}),
    # View 子类型用 attrs["view"] 标注(供 shape 推断按类型施变换:reshape/transpose/split/...)。
    "Reshape": ("View", {"view": "reshape"}), "Transpose": ("View", {"view": "transpose"}),
    "SplitWithSize": ("View", {"view": "split"}), "Shape": ("View", {"view": "shape"}),
    "ExpandDims": ("View", {"view": "expand_dims"}), "Tile": ("View", {"view": "tile"}),
    # concat/stack:接受"张量列表"作单实参 → variadic(摊平为多操作数);反向仅切片,不存激活 → View。
    "Concat": ("View", {"variadic": True, "view": "concat"}),
    # RoPE 应用(ApplyRotaryPosEmb):对 q/k 的位置分量做旋转 = **线性正交变换**(cos/sin 为位置常量),
    # 反向不需存激活 → Elementwise(linear=True)。构造点 `ApplyRotaryPosEmb(config)` 直接实例化。
    "ApplyRotaryPosEmb": ("Elementwise", {"linear": True, "rope": True}),
    # Swiglu(门控 SiLU):MoE experts 的融合激活(FFNGroupedGEMM.swiglu)。反向需存输入 → 归 Activation。
    "Swiglu": ("Activation", {"activation_type": "swiglu"}),
    # 具名 linear/attention/norm/activation 由 construct 调用点解析(build_module/get_activation),此处不绑。

    # loss 段原语(T0-4 增补;语义=教科书 VJP 分类,与 bprop_rules 口径一致;类名取自
    # loss_func.py 实际 import 的 mindspore.ops.auto_generate 算子,与旧版/其它模块的同名类
    # 不同——ArgMaxWithValue(非 ReduceMax):max 归约,反向只回传 argmax 位置 → 非线性归约)。
    "ArgMaxWithValue": ("Elementwise", {"linear": False, "reduce": True}),
    "Exp": ("Elementwise", {"linear": False}),
    "SumExt": ("Elementwise", {"linear": True, "reduce": True}),
    "Log": ("Elementwise", {"linear": False}),
    "OneHotExt": ("Elementwise", {"linear": True}),
    "Neg": ("Elementwise", {"linear": True}),
    "Div": ("Elementwise", {"linear": False}),

    # loss/embedding 段原语(T0-5 增补;类名取自 layers.py 实际 import 的
    # mindspore.ops.auto_generate 算子:VocabParallelEmbedding.embedding_func 的 TP mask 分支
    # `self.relu`/`self.minimum`/`self.equal`,layers.py:99-101 __init__ 绑定,:153-155 调用)。
    "ReLU": ("Activation", {"activation_type": "relu"}),
    "Minimum": ("Elementwise", {"linear": False}),
    "Equal": ("Elementwise", {"linear": True}),  # 不可微→反向不存(linear 语义按消费方 bprop_rules.py:47 口径)
}

@dataclass
class Binding:
    op: str
    attrs: dict = field(default_factory=dict)

def _innermost_call(node):
    """剥链式 .recompute()/.shard()/.add_prim_attr() 等,取最内层"func 为 Name"的 Call 节点。"""
    while isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return node
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
            node = f.value
            continue
        return None
    return None


def _base_call_name(node):
    """剥链式 .recompute()/.shard() 等,取最内层 Call 的类名。"""
    call = _innermost_call(node)
    return call.func.id if call is not None else None


def _concat_axis(node) -> int:
    """`Concat(axis=3)` / `Concat(3)` → 3;缺省 0。"""
    call = _innermost_call(node)
    if call is None:
        return 0
    for kw in call.keywords:
        if kw.arg == "axis" and isinstance(kw.value, ast.Constant):
            return int(kw.value.value)
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, int):
        return int(call.args[0].value)
    return 0

def bind_init(src: str, cls_name: str) -> dict[str, Binding]:
    tree = ast.parse(src)
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    if cls is None:
        raise ValueError(f"源码里找不到 class {cls_name}(fail-loud)")
    init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    out: dict[str, Binding] = {}
    if init is None:
        # 该类自身不定义 __init__(继承自基类)。此前直接 `ast.walk(None)` → AttributeError;
        # 调用方(extractor 用 _init_classes 逐类调本函数)本就按「该类无绑定」处理,故返回空。
        return out
    for stmt in ast.walk(init):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self"):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        clsname = _base_call_name(stmt.value)
        if clsname in _CLS2OP:
            op, attrs = _CLS2OP[clsname]
            attrs = dict(attrs)
            if clsname == "Concat":
                attrs["concat_axis"] = _concat_axis(stmt.value)
            out[tgt.attr] = Binding(op=op, attrs=attrs)
    return out


# ── 裸函数别名(pynative 惯用法)—— Task 2 / 评估文档 §2.4 + P0#2 ────────────────────────────
# `bind_init` 只认「类实例化」(`isinstance(stmt.value, ast.Call)` + 类名查 `_CLS2OP`),这是
# `parallel_core/training_graph/` 的惯用法(`self.reshape = Reshape()`)。pynative 侧是**成文约定**
# 的另一套:`self.reshape = mint.reshape` / `self.cast = ops.cast`(裸别名,不是 Call)
# —— 见 `csa.py:629-630` 注释「Alias the non-trivial mint ops used in construct/forward per the
# fine-grained-recompute convention (RFC §3.1 #9)」。实测 dsv4 链上 **39 个不同原语 / 196 个调用点**
# 一个都绑不上(评估文档 §2.4 表)。
#
# 真正的绑定通路(裸别名 → op 类型的新表)属**路线 B P0#2**(M 级)。本函数只做 Task 2 要求的那半:
# **把它们枚举出来、计数、surface**,绝不静默跳过 —— 否则「196 个调用点不可见」这件事在报告里
# 一个字都不会出现。
_ALIAS_NAMESPACES = ("mint", "ops", "F", "mindspore", "P", "nn")


def unbound_aliases(src: str, cls_name: str, file: str = "") -> list[dict]:
    """列出 `__init__` 里 `self.<attr> = <ns>.<fn>` 形态的**裸函数别名**(`_CLS2OP` 绑不上的)。

    只收「点号路径且链首在已知张量算子命名空间」的形态,避免把 `self.n = config.num_heads`
    这类配置读取误当算子别名。返回逐条 dict:`{attr, alias, lineno, src, note}`。
    """
    tree = ast.parse(src)
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    if cls is None:
        raise ValueError(f"源码里找不到 class {cls_name}(fail-loud)")
    init = next((n for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    out: list[dict] = []
    if init is None:
        return out                      # 该类自身不定义 __init__(调用方按 MRO 逐类调本函数)
    bound = set(bind_init(src, cls_name))
    for stmt in ast.walk(init):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"):
            continue
        val = stmt.value
        if not isinstance(val, ast.Attribute):      # 裸别名恒是 Attribute(不是 Call)
            continue
        # 链首必须是已知张量算子命名空间(`mint.reshape` / `ops.cast` / `mint.nn.functional.x`)
        head = val
        while isinstance(head, ast.Attribute):
            head = head.value
        if not (isinstance(head, ast.Name) and head.id in _ALIAS_NAMESPACES):
            continue
        if tgt.attr in bound:                       # 已被 _CLS2OP 绑住 → 不算未绑
            continue
        out.append({
            "attr": tgt.attr, "alias": ast.unparse(val), "lineno": stmt.lineno,
            "src": f"{file}:{stmt.lineno}" if file else str(stmt.lineno),
            "note": "pynative 裸函数别名:init_binder._CLS2OP 只认类实例化 → 未绑定"
                    "(路线 B P0#2 的绑定表尚未建)",
        })
    return out
