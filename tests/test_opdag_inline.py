# tests/test_opdag_inline.py
"""PIECE:construct_walker 的**内部方法内联**(STEP 2)。合成类,受控。

契约:
  * `self.<name>(...)` 当 <name> 不是绑定算子、但**是本类(含基类 MRO)里的一个 `def <name>`**
    → 把该方法体在调用点**内联展开**:形参绑定到实参(Name 实参共享调用方 SSA),
      方法 `return` 值绑定回调用点的赋值目标,数据流边跨内联边界连续。
  * 递归内联(方法直接/间接自调用)→ **fail-loud**(带方法名)。
  * 既非绑定算子、又非内部方法 → 仍旧 fail-loud(带名+行号)。
  * `y = self.op(x)[0]`(Subscript 包裹的调用)照常发射算子(下标只是选输出张量)。
  * variadic 算子(如 concat)的单个 List/Tuple 实参**摊平**为多操作数(连各自 producer)。
  * present_vars 里的变量在 `if v is not None:` 里判 True(走 rope 等"输入存在"支)。
"""
import pytest

from cost_eval.opdag.init_binder import Binding
from cost_eval.opdag.construct_walker import walk_construct


def _mm(mod="X"):
    return Binding(op="MatMul", attrs={"module": mod, "compute_dtype": "bf16"})


INLINE_BASIC = '''
class Blk:
    def construct(self, x):
        y = self.helper(x)
        z = self.mm2(y)
        return z
    def helper(self, a):
        b = self.mm1(a)
        return b
'''


def test_internal_method_is_inlined_to_its_ops():
    binds = {"mm1": _mm("A"), "mm2": _mm("B")}
    dag = walk_construct(INLINE_BASIC, "Blk", binds, "blk.py")
    assert [n.op for n in dag.nodes] == ["MatMul", "MatMul"]
    assert dag.nodes[0].module == "A" and dag.nodes[1].module == "B"


def test_inline_preserves_ssa_edges_across_boundary():
    binds = {"mm1": _mm("A"), "mm2": _mm("B")}
    dag = walk_construct(INLINE_BASIC, "Blk", binds, "blk.py")
    # y = helper(x) 返回 mm1 输出;mm2 消费 y → 边 mm1->mm2
    assert dag.edges == [[1, 2]]
    # mm1 消费的是 helper 形参 a(绑定到调用方 x)→ 占位 ref,不连边
    assert dag.nodes[0].ins == ["x:?:bf16"]


def test_inline_src_points_into_helper_body():
    binds = {"mm1": _mm("A"), "mm2": _mm("B")}
    dag = walk_construct(INLINE_BASIC, "Blk", binds, "blk.py")
    assert dag.nodes[0].src == "blk.py:8"   # helper 体内 mm1
    assert dag.nodes[1].src == "blk.py:5"   # construct 里 mm2


INLINE_TUPLE = '''
class Blk:
    def construct(self, x):
        q, k = self.qk(x)
        o = self.attn(q, k)
        return o
    def qk(self, h):
        a = self.pa(h)
        b = self.pb(h)
        return a, b
'''


def test_inline_tuple_return_binds_each_target():
    binds = {"pa": _mm("A"), "pb": _mm("B"), "attn": Binding("FlashAttention", {})}
    dag = walk_construct(INLINE_TUPLE, "Blk", binds, "blk.py")
    assert [n.op for n in dag.nodes] == ["MatMul", "MatMul", "FlashAttention"]
    # attn 消费 q(=pa 输出) 与 k(=pb 输出) → 边 [1,3] 与 [2,3]
    assert [1, 3] in dag.edges and [2, 3] in dag.edges


NESTED = '''
class Blk:
    def construct(self, x):
        y = self.outer(x)
        return y
    def outer(self, a):
        b = self.inner(a)
        c = self.mm(b)
        return c
    def inner(self, p):
        q = self.down(p)
        return q
'''


def test_nested_inline_two_levels():
    binds = {"down": _mm("D"), "mm": _mm("M")}
    dag = walk_construct(NESTED, "Blk", binds, "blk.py")
    assert [n.op for n in dag.nodes] == ["MatMul", "MatMul"]
    assert dag.nodes[0].module == "D" and dag.nodes[1].module == "M"
    assert dag.edges == [[1, 2]]


REC = '''
class Blk:
    def construct(self, x):
        y = self.rec(x)
        return y
    def rec(self, a):
        b = self.rec(a)
        return b
'''


def test_inline_recursion_fails_loud():
    with pytest.raises(ValueError) as ei:
        walk_construct(REC, "Blk", {}, "blk.py")
    assert "rec" in str(ei.value)


UNK = '''
class Blk:
    def construct(self, x):
        y = self.ghost(x)
        return y
'''


def test_unknown_non_method_still_fails_loud():
    with pytest.raises(ValueError) as ei:
        walk_construct(UNK, "Blk", {}, "blk.py")
    msg = str(ei.value)
    assert "ghost" in msg and "blk.py:4" in msg


SUBSCR = '''
class Blk:
    def construct(self, x):
        y = self.proj(x)[0]
        z = self.mm(y)
        return z
'''


def test_subscript_wrapped_call_emits_op_and_wires_edge():
    binds = {"proj": _mm("R"), "mm": _mm("M")}
    dag = walk_construct(SUBSCR, "Blk", binds, "blk.py")
    assert [n.op for n in dag.nodes] == ["MatMul", "MatMul"]
    assert dag.nodes[0].out == "y:?:bf16"
    assert dag.edges == [[1, 2]]          # proj -> mm(y)


CONCAT = '''
class Blk:
    def construct(self, x):
        a = self.pa(x)
        b = self.pb(x)
        c = self.cat([a, b])
        return c
'''


def test_variadic_concat_flattens_list_operands():
    binds = {"pa": _mm("A"), "pb": _mm("B"),
             "cat": Binding("View", {"variadic": True})}
    dag = walk_construct(CONCAT, "Blk", binds, "blk.py")
    assert [n.op for n in dag.nodes] == ["MatMul", "MatMul", "View"]
    assert [1, 3] in dag.edges and [2, 3] in dag.edges


PRESENT = '''
class Blk:
    def construct(self, x, rope=None):
        h = self.fc(x)
        if rope is not None:
            h = self.rope(h)
        return h
'''


def test_present_var_takes_is_not_none_branch():
    binds = {"fc": _mm(), "rope": Binding("Elementwise", {"linear": True})}
    dag = walk_construct(PRESENT, "Blk", binds, "blk.py",
                         present_vars={"rope"})
    assert [n.op for n in dag.nodes] == ["MatMul", "Elementwise"]


PURE_RAISE = '''
class Blk:
    def construct(self, x):
        if x.ndim != 3:
            raise ValueError("bad shape")
        y = self.fc(x)
        return y
'''


def test_undecidable_pure_raise_guard_is_skipped():
    # 纯断言守卫(条件不可判定、整支仅 raise、无 else)→ 视作校验断言,跳过而非 fail-loud
    binds = {"fc": _mm()}
    dag = walk_construct(PURE_RAISE, "Blk", binds, "blk.py",
                         config_flags={"anything": True})
    assert [n.op for n in dag.nodes] == ["MatMul"]
