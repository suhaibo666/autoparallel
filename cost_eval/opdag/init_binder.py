# cost_eval/opdag/init_binder.py
"""Pass B(设计 §3.2):解析 Cell.__init__ 的 AST,建 self.<name> → Binding(op 类型,attrs)。
按 import 的类名解析规范 op 类型;链式(.recompute/.shard)剥到基 Call。未知类名 fail-loud。"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field

from . import primitives as prims

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

    # ── pynative 侧直接实例化的小 Cell(P0#2 增补,2026-07-25;定位符见每条)───────────────
    # IdentityOp:恒等映射(`pynative/layers/identity_op.py`),反向 dx=dy 直通。
    #   `self.hadamard = ... else IdentityOp()`(compressor.py:129)、
    #   `self.post_head_sum = IdentityOp()`(indexer.py:303)。
    "IdentityOp": ("Identity", {}),
    # Hadamard(`utils.py:101`):`y = linear(x, H) * scale`,H 是**常量正交矩阵**
    #   (`utils.py:139-141`,`hadamard(head_dim)` 生成后缓存)。故 dx = dy·Hᵀ·scale
    #   —— **与 x 的值无关 → 反向不存激活**。这与既有 `ApplyRotaryPosEmb` 的归类理由
    #   (线性正交变换,见本表上方注释)是同一条,故同归 Elementwise(linear=True)。
    #   `self.hadamard = Hadamard(self.index_head_dim)`(indexer.py:143 / compressor.py:129)。
    "Hadamard": ("Elementwise", {"linear": True, "hadamard": True}),
    # RoPE 频率表(`RotaryEmbedding` / `YarnRotaryEmbedding`):产出 (freqs, mscale) 只依赖
    #   序列长度与常量基频,**无参数、无梯度** → 常量产出。
    #   `freqs, _ = self.rotary_pos_emb(sq)`(deepseek_v4_hybrid_attention.py:257、
    #   indexer.py:174、compressor.py:231)。
    "RotaryEmbedding": ("Constant", {"rope_freqs": True}),
    "YarnRotaryEmbedding": ("Constant", {"rope_freqs": True}),
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

def _self_assigns(init: ast.FunctionDef):
    """`__init__` 里所有 `self.<attr> = <value>` 的 (attr, value 节点, stmt)。"""
    for stmt in ast.walk(init):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"):
            continue
        yield tgt.attr, stmt.value, stmt


def _binding_for_ctor(value) -> Binding | None:
    """`<Cls>(...)`(含链式 .recompute()/.shard())→ Binding;类名不在 `_CLS2OP` 则 None。"""
    clsname = _base_call_name(value)
    if clsname not in _CLS2OP:
        return None
    op, attrs = _CLS2OP[clsname]
    attrs = dict(attrs)
    if clsname == "Concat":
        attrs["concat_axis"] = _concat_axis(value)
    return Binding(op=op, attrs=attrs)


def _class_method(cls: ast.ClassDef, name: str):
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _binding_from_factory_method(cls: ast.ClassDef, method_name: str) -> Binding | None:
    """`self.X = self.<method>(...)` —— 取该方法**所有 `return <Cls>(...)`** 的类名。

    全部映射到**同一** op 类型时才绑(否则该走 config 判定,不许猜)。实测必需:
    `self.rotary_pos_emb = self._build_rotary_pos_emb(config, base, use_yarn)`
    (`deepseek_v4_hybrid_attention.py:80`)—— 该 staticmethod 的两条 return 分别是
    `RotaryEmbedding(...)`(`:177`)与 `YarnRotaryEmbedding(...)`(`:185`),两者都是
    「RoPE 频率表 = 常量产出」→ 同一 op 类型 → 可绑。
    """
    m = _class_method(cls, method_name)
    if m is None:
        return None
    binds = []
    for n in ast.walk(m):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Call):
            b = _binding_for_ctor(n.value)
            if b is None:
                return None
            binds.append(b)
    if not binds:
        return None
    ops = {b.op for b in binds}
    if len(ops) != 1:
        return None
    return binds[0]


def bind_init(src: str, cls_name: str, config_flags: dict | None = None,
              file: str = "", init_param_binds: dict | None = None) -> dict[str, Binding]:
    """`__init__` → `{self.<name>: Binding}`。四种形态(前两种是既有,后两种是 P0#2 增补):

      1. **类实例化** `self.mul = Mul()` —— training_graph 惯用法,按类名查 `_CLS2OP`;
      2. 链式 `.recompute()/.shard()` 剥到基 Call(同上);
      3. **裸函数别名** `self.reshape = mint.reshape` —— **pynative 惯用法**,按
         `primitives.PRIMITIVES` 查(未知原语不在此 raise,只是不绑 → 调用到它时 fail-loud);
      4. **三元 / 工厂方法** `self.hadamard = Hadamard(d) if rotate else IdentityOp()`
         (compressor.py:129)、`self.rotary_pos_emb = self._build_rotary_pos_emb(...)`
         (deepseek_v4_hybrid_attention.py:80)。三元按 `config_flags` 判定;不可判定但两支
         **同 op 类型**时按该类型绑(两支等价,无需判定);否则不绑。
    """
    tree = ast.parse(src)
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    if cls is None:
        raise ValueError(f"源码里找不到 class {cls_name}(fail-loud)")
    init = _class_method(cls, "__init__")
    out: dict[str, Binding] = {}
    if init is None:
        # 该类自身不定义 __init__(继承自基类)。此前直接 `ast.walk(None)` → AttributeError;
        # 调用方(extractor 用 _init_classes 逐类调本函数)本就按「该类无绑定」处理,故返回空。
        return out
    flags = dict(config_flags or {})
    for attr, value, _stmt in _self_assigns(init):
        if isinstance(value, ast.Call):
            b = _binding_for_ctor(value)
            if b is None:
                f = value.func
                if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                        and f.value.id == "self"):
                    b = _binding_from_factory_method(cls, f.attr)     # 形态 4b
            if b is not None:
                out[attr] = b
            continue
        if isinstance(value, ast.Attribute):                          # 形态 3
            b = _alias_binding(value)
            if b is not None:
                out[attr] = b
            continue
        if isinstance(value, ast.IfExp):                              # 形态 4a
            b = _ifexp_binding(value, cls, flags)
            if b is not None:
                out[attr] = b
            continue
        if isinstance(value, ast.Name) and value.id in (init_param_binds or {}):
            # 形态 5:**构造函数注入的子模块** `self.rotary_pos_emb = rotary_pos_emb`
            # (indexer.py:87 / compressor.py:92)。绑定来自**父的构造点**:父在
            # `build_module(..., rotary_pos_emb=self.rotary_pos_emb)`(indexer.py:131、
            # csa.py:602/614、deepseek_v4_hybrid_attention.py:88)里把自己那个已绑好的
            # `self.rotary_pos_emb` 传进来 —— 所以这不是猜,是顺着源的实参传递读出来的。
            out[attr] = init_param_binds[value.id]
    return out


def _alias_binding(value: ast.Attribute) -> Binding | None:
    """裸别名 `self.x = <ns>.<fn>` → Binding;链首不在张量算子命名空间或表里没有 → None。"""
    dotted = ast.unparse(value)
    if not prims.is_alias_namespace(dotted):
        return None
    hit = prims.lookup(dotted)
    if hit is None:
        return None
    op, attrs = hit
    attrs = dict(attrs)
    attrs["prim"] = dotted
    return Binding(op=op, attrs=attrs)


def _ifexp_binding(value: ast.IfExp, cls: ast.ClassDef, flags: dict) -> Binding | None:
    """三元构造 `A(...) if <cond> else B(...)`:按 config 判定;判不出但两支同 op 类型即可绑。"""
    def side(node):
        if isinstance(node, ast.Call):
            b = _binding_for_ctor(node)
            if b is None:
                f = node.func
                if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                        and f.value.id == "self"):
                    b = _binding_from_factory_method(cls, f.attr)
            return b
        if isinstance(node, ast.Attribute):
            return _alias_binding(node)
        return None

    b_true, b_false = side(value.body), side(value.orelse)
    test = value.test
    known = None
    if isinstance(test, ast.Name) and test.id in flags:
        known = bool(flags[test.id])
    elif (isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name)
          and test.value.id == "self" and test.attr in flags):
        known = bool(flags[test.attr])
    elif isinstance(test, ast.Constant):
        known = bool(test.value)
    if known is not None:
        return b_true if known else b_false
    if b_true is not None and b_false is not None and b_true.op == b_false.op:
        # 两支同 op 类型(如 `Hadamard(...) if rotate else IdentityOp()`——都是「反向不存激活的
        # 单输入变换」)→ 判定与否**不影响 saved 集**,取任一支即可,无需猜。
        return b_true
    return None


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
    """列出 `__init__` 里**仍绑不上**的裸函数别名 —— 即 `primitives.PRIMITIVES` 表里**没有**的。

    P0#2 落地后语义收窄:能查表绑上的(绝大多数)已由 `bind_init` 形态 3 绑定,不再算「未绑」;
    留在这里的是**真·未知原语**——它们是「该加表项」的清单,也是调用到它们时 fail-loud 的依据。
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
        if tgt.attr in bound:                       # 已绑住(类实例化 / 表内裸别名)→ 不算未绑
            continue
        out.append({
            "attr": tgt.attr, "alias": ast.unparse(val), "lineno": stmt.lineno,
            "src": f"{file}:{stmt.lineno}" if file else str(stmt.lineno),
            "note": "裸函数别名,但 primitives.PRIMITIVES 无此原语表项 —— 调用到它会 fail-loud;"
                    "请加表项并写清「反向要读什么」+ 定位符(绝不猜类别)",
        })
    return out
