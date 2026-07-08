# tests/test_opdag_walk_pruning.py
"""PIECE 1:construct_walker 的 config 驱动分支剪枝(合成 construct,受控)。

契约:
  * 传入 config_flags / none_vars / param_defaults 后,`if` 只走 config 命中的那一支;
  * 不可判定的 `if` 在剪枝上下文下 **fail-loud**(拒绝双走 —— 双走会重复计激活/内存);
  * 命中(被选中)的裸 `raise` 分支 → fail-loud(选到了不支持的路径);未命中的 raise 守卫跳过;
  * 不给任何剪枝上下文时保持旧的"两支都线性走查"行为(Task 4 回归)。
"""
import pytest

from cost_eval.opdag.init_binder import Binding
from cost_eval.opdag.construct_walker import walk_construct


# (a) if self.gated_linear_unit: → True 支
GLU = '''
class C:
    def construct(self, x):
        if self.gated_linear_unit:
            y = self.act(x)
        else:
            y = self.other(x)
        return y
'''

def test_self_flag_prunes_to_true_branch():
    binds = {"act": Binding("Activation", {}), "other": Binding("MatMul", {})}
    dag = walk_construct(GLU, "C", binds, "c.py", config_flags={"gated_linear_unit": True})
    assert [n.op for n in dag.nodes] == ["Activation"]

def test_self_flag_prunes_to_false_branch():
    binds = {"act": Binding("Activation", {}), "other": Binding("MatMul", {})}
    dag = walk_construct(GLU, "C", binds, "c.py", config_flags={"gated_linear_unit": False})
    assert [n.op for n in dag.nodes] == ["MatMul"]


# (b) if bias_parallel is not None: with none_vars → False 支(跳过 add)
BIAS = '''
class C:
    def construct(self, x, bias_parallel):
        h = self.fc(x)
        if bias_parallel is not None:
            h = self.add(h, bias_parallel)
        return h
'''

def test_is_not_none_prunes_via_none_vars():
    binds = {"fc": Binding("MatMul", {}), "add": Binding("Elementwise", {})}
    dag = walk_construct(BIAS, "C", binds, "c.py", none_vars={"bias_parallel"})
    assert [n.op for n in dag.nodes] == ["MatMul"]  # add 被剪掉


# (c) if self.activation_type == 'swiglu': 字面量比较
ACT = '''
class C:
    def construct(self, x):
        if self.activation_type == 'swiglu':
            y = self.swi(x)
        else:
            y = self.gen(x)
        return y
'''

def test_eq_literal_compare_selects_branch():
    binds = {"swi": Binding("Activation", {"kind": "swiglu"}),
             "gen": Binding("Activation", {"kind": "gen"})}
    dag = walk_construct(ACT, "C", binds, "c.py", config_flags={"activation_type": "swiglu"})
    assert [n.attrs.get("kind") for n in dag.nodes] == ["swiglu"]
    dag2 = walk_construct(ACT, "C", binds, "c.py", config_flags={"activation_type": "silu"})
    assert [n.attrs.get("kind") for n in dag2.nodes] == ["gen"]


# (d) 顶部 `if per_token_scale is not None: raise` —— param_defaults 让它判 False → 守卫跳过
GUARD = '''
class C:
    def construct(self, x, per_token_scale=None):
        if per_token_scale is not None:
            raise NotImplementedError("per_token_scale not supported")
        y = self.fc(x)
        return y
'''

def test_not_taken_raise_guard_is_skipped():
    binds = {"fc": Binding("MatMul", {})}
    dag = walk_construct(GUARD, "C", binds, "c.py", param_defaults={"per_token_scale": None})
    assert [n.op for n in dag.nodes] == ["MatMul"]  # raise 守卫被跳过,fc 正常发射


# (e) 不可判定的 if → fail-loud(点名 file:line + 条件)
UNK = '''
class C:
    def construct(self, x):
        if self.something_unknown:
            y = self.a(x)
        else:
            y = self.b(x)
        return y
'''

def test_undecidable_if_fails_loud():
    binds = {"a": Binding("MatMul", {}), "b": Binding("MatMul", {})}
    with pytest.raises(ValueError) as ei:
        walk_construct(UNK, "C", binds, "c.py", config_flags={"gated_linear_unit": True})
    msg = str(ei.value)
    assert "something_unknown" in msg and "c.py:" in msg


# (f) 被选中的裸 raise 分支 → fail-loud(选到了不支持的路径)
BADPATH = '''
class C:
    def construct(self, x):
        if self.use_bad_path:
            raise NotImplementedError("unsupported")
        y = self.fc(x)
        return y
'''

def test_selected_raise_branch_fails_loud():
    binds = {"fc": Binding("MatMul", {})}
    with pytest.raises(ValueError) as ei:
        walk_construct(BADPATH, "C", binds, "c.py", config_flags={"use_bad_path": True})
    assert "raise" in str(ei.value) or "unsupported" in str(ei.value)


# (g) 无剪枝上下文 → 保持旧行为(两支都走)——Task 4 回归保护
BOTH = '''
class C:
    def construct(self, x):
        if self.flag:
            a = self.p(x)
        else:
            b = self.q(x)
        return a
'''

def test_no_context_walks_both_branches_legacy():
    binds = {"p": Binding("MatMul", {}), "q": Binding("Activation", {})}
    dag = walk_construct(BOTH, "C", binds, "c.py")  # 无 config → 两支都走
    assert [n.op for n in dag.nodes] == ["MatMul", "Activation"]


# 三元 IfExp:`self.act(h) if self.activation_func else h` —— 关键路径(真 MLP else 支)
TERNARY = '''
class C:
    def construct(self, x):
        h = self.fc(x)
        y = self.act(h) if self.activation_func else h
        z = self.fc2(y)
        return z
'''

def test_ternary_true_emits_activation():
    binds = {"fc": Binding("MatMul", {}), "act": Binding("Activation", {}), "fc2": Binding("MatMul", {})}
    dag = walk_construct(TERNARY, "C", binds, "c.py", config_flags={"activation_func": True})
    assert [n.op for n in dag.nodes] == ["MatMul", "Activation", "MatMul"]
    assert dag.edges == [[1, 2], [2, 3]]

def test_ternary_false_aliases_operand_no_activation():
    binds = {"fc": Binding("MatMul", {}), "act": Binding("Activation", {}), "fc2": Binding("MatMul", {})}
    dag = walk_construct(TERNARY, "C", binds, "c.py", config_flags={"activation_func": False})
    assert [n.op for n in dag.nodes] == ["MatMul", "MatMul"]  # act 被剪掉
    # y 别名 h(fc 产出)→ fc2 消费 y 应连回 fc
    assert dag.edges == [[1, 2]]
