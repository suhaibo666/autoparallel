# cost_eval/opdag/init_binder.py
"""Pass B(设计 §3.2):解析 Cell.__init__ 的 AST,建 self.<name> → Binding(op 类型,attrs)。
按 import 的类名解析规范 op 类型;链式(.recompute/.shard)剥到基 Call。未知类名 fail-loud。"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field

# 规范 op 类型映射(mindspore 类名 → 我们的 op 类型)。linear=反向是否线性。
_CLS2OP = {
    "AddExt": ("Elementwise", {"linear": True}),  "Add": ("Elementwise", {"linear": True}),
    "Sub":   ("Elementwise", {"linear": True}),
    "Mul":   ("Elementwise", {"linear": False}),
    "Cast":  ("Cast", {}),
    "Reshape": ("View", {}), "Transpose": ("View", {}), "SplitWithSize": ("View", {}),
    "Shape": ("View", {}), "ExpandDims": ("View", {}), "Tile": ("View", {}),
    # concat/stack:接受"张量列表"作单实参 → variadic(摊平为多操作数);反向仅切片,不存激活 → View。
    "Concat": ("View", {"variadic": True}),
    # RoPE 应用(ApplyRotaryPosEmb):对 q/k 的位置分量做旋转 = **线性正交变换**(cos/sin 为位置常量),
    # 反向不需存激活 → Elementwise(linear=True)。构造点 `ApplyRotaryPosEmb(config)` 直接实例化。
    "ApplyRotaryPosEmb": ("Elementwise", {"linear": True, "rope": True}),
    # Swiglu(门控 SiLU):MoE experts 的融合激活(FFNGroupedGEMM.swiglu)。反向需存输入 → 归 Activation。
    "Swiglu": ("Activation", {"activation_type": "swiglu"}),
    # 具名 linear/attention/norm/activation 由 construct 调用点解析(build_module/get_activation),此处不绑。
}

@dataclass
class Binding:
    op: str
    attrs: dict = field(default_factory=dict)

def _base_call_name(node):
    """剥链式 .recompute()/.shard() 等,取最内层 Call 的类名。"""
    while isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            # X().method() → 递归到 X()
            if isinstance(f.value, ast.Call):
                node = f.value
                continue
            return None
        return None
    return None

def bind_init(src: str, cls_name: str) -> dict[str, Binding]:
    tree = ast.parse(src)
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    if cls is None:
        raise ValueError(f"源码里找不到 class {cls_name}(fail-loud)")
    init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    out: dict[str, Binding] = {}
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
            out[tgt.attr] = Binding(op=op, attrs=dict(attrs))
    return out
