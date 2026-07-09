# tests/test_opdag_subcell.py
"""STEP 2:extractor 的**子 Cell 递归**(设计 Task-acceptance)。合成 parent+child Cell,受控。

契约:
  * `self.<X>(...)` 当其 Pass-B 绑定为 `op="SubCell"`(build_module 的 submodules.<Y> 解成子
    ResolvedSpec、或裸 Cell 类名且不在 LEAF_OPTYPE)→ **递归抽取该子 Cell 的 DAG**,并在调用点
    **内联**其节点:
      - 节点 id 重新编号,保持父 DAG 内唯一(不从 1 重启);
      - 子 DAG 的输入操作数 ref(= 子 construct 形参)按位重映射到调用方实参的 SSA ref;
      - 跨内联边界补数据流边(父 producer -> 子叶子;子输出 -> 下游消费者)。
  * 子 Cell 的源文件由在 mf_root 里搜 `class <CellName>` 定位。
  * 递归深度/环 → fail-loud。
"""
import os
import textwrap

import pytest

from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.module_resolver import ResolvedSpec


PARENT_SRC = textwrap.dedent('''
    from mindspore.ops.auto_generate import Mul
    from mindformers.parallel_core.utils.spec_utils import build_module

    class Parent(nn.Cell):
        def __init__(self, config, submodules):
            super().__init__()
            self.pre = Mul()
            self.child = build_module(submodules.child, config=config)
            self.post = Mul()

        def construct(self, x):
            a = self.pre(x, x)
            b = self.child(a)
            c = self.post(b, b)
            return c
''')

CHILD_SRC = textwrap.dedent('''
    from mindformers.parallel_core.utils.spec_utils import build_module

    class Child(nn.Cell):
        def __init__(self, config, submodules):
            super().__init__()
            self.fc1 = build_module(submodules.fc1, config=config)
            self.fc2 = build_module(submodules.fc2, config=config)

        def construct(self, t):
            u = self.fc1(t)
            v = self.fc2(u)
            return v
''')


@pytest.fixture
def mf_tmp(tmp_path):
    (tmp_path / "parent.py").write_text(PARENT_SRC, encoding="utf-8")
    (tmp_path / "child.py").write_text(CHILD_SRC, encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def parent_spec():
    return ResolvedSpec(
        cell="Parent",
        submodules={
            "child": ResolvedSpec(
                cell="Child",
                submodules={"fc1": "ColumnParallelLinear", "fc2": "RowParallelLinear"},
            ),
        },
    )


FLAGS = {"compute_dtype": "bf16", "add_bias_linear": False}


def test_subcell_nodes_inlined_with_ops(mf_tmp, parent_spec):
    dag = extract_cell(mf_tmp, "parent.py", "Parent", parent_spec, FLAGS, recurse=True)
    ops = [n.op for n in dag.nodes]
    # pre(Mul) -> [child: fc1 MatMul, fc2 MatMul] -> post(Mul)
    assert ops == ["Elementwise", "MatMul", "MatMul", "Elementwise"]
    assert dag.nodes[1].module == "ColumnParallelLinear"
    assert dag.nodes[2].module == "RowParallelLinear"


def test_subcell_ids_are_unique_and_monotonic(mf_tmp, parent_spec):
    dag = extract_cell(mf_tmp, "parent.py", "Parent", parent_spec, FLAGS, recurse=True)
    ids = [n.id for n in dag.nodes]
    assert ids == [1, 2, 3, 4]                 # 子节点没有从 1 重启
    assert len(set(ids)) == len(ids)


def test_subcell_edges_wired_across_boundary(mf_tmp, parent_spec):
    dag = extract_cell(mf_tmp, "parent.py", "Parent", parent_spec, FLAGS, recurse=True)
    e = dag.edges
    # pre(1) -> child.fc1(2)(形参 t=a 重映射到 pre 输出)
    assert [1, 2] in e
    # child.fc1(2) -> child.fc2(3)(子 DAG 内部边,已平移)
    assert [2, 3] in e
    # child 输出 fc2(3) -> post(4)(子输出被下游消费)
    assert [3, 4] in e


def test_subcell_src_points_into_child_file(mf_tmp, parent_spec):
    dag = extract_cell(mf_tmp, "parent.py", "Parent", parent_spec, FLAGS, recurse=True)
    # 内联的子节点 src 指回 child.py
    assert dag.nodes[1].src.startswith("child.py:")
    assert dag.nodes[2].src.startswith("child.py:")
    # 父节点 src 指回 parent.py
    assert dag.nodes[0].src.startswith("parent.py:")
    assert dag.nodes[3].src.startswith("parent.py:")


def test_subcell_input_ref_remapped_to_caller_arg(mf_tmp, parent_spec):
    dag = extract_cell(mf_tmp, "parent.py", "Parent", parent_spec, FLAGS, recurse=True)
    # child.fc1 的操作数原是形参 t;内联后应重映射成调用方变量 a
    fc1 = dag.nodes[1]
    assert any(ref.split(":")[0] == "a" for ref in fc1.ins)


BARE_PARENT_SRC = textwrap.dedent('''
    from mindformers.parallel_core.utils.spec_utils import build_module

    class BareParent(nn.Cell):
        def __init__(self, config, submodules):
            super().__init__()
            self.child = build_module(submodules.child, config=config)

        def construct(self, x):
            y = self.child(x)
            return y
''')


def test_bare_class_submodule_is_recursed(tmp_path):
    # experts=FFNGroupedGEMM 这类:build_module 的 submodule 是**裸 Cell 类名**(str)且不在 LEAF_OPTYPE
    # → 也应递归(而不是 fail-loud "不在 LEAF_OPTYPE")。
    (tmp_path / "bareparent.py").write_text(BARE_PARENT_SRC, encoding="utf-8")
    (tmp_path / "child.py").write_text(CHILD_SRC, encoding="utf-8")
    spec = ResolvedSpec(cell="BareParent", submodules={"child": "Child"})
    child_specs = {
        "Child": ResolvedSpec(cell="Child",
                              submodules={"fc1": "ColumnParallelLinear", "fc2": "RowParallelLinear"}),
    }
    dag = extract_cell(str(tmp_path), "bareparent.py", "BareParent", spec, FLAGS,
                       recurse=True, subcell_specs=child_specs)
    assert [n.op for n in dag.nodes] == ["MatMul", "MatMul"]


CYCLE_A = textwrap.dedent('''
    from mindformers.parallel_core.utils.spec_utils import build_module
    class CycA(nn.Cell):
        def __init__(self, config, submodules):
            super().__init__()
            self.child = build_module(submodules.child, config=config)
        def construct(self, x):
            return self.child(x)
''')


def test_subcell_cycle_fails_loud(tmp_path):
    (tmp_path / "cyca.py").write_text(CYCLE_A, encoding="utf-8")
    # 自指:CycA.child 解成 CycA 自己 → 递归环,必须 fail-loud。
    spec = ResolvedSpec(cell="CycA", submodules={"child": ResolvedSpec(cell="CycA", submodules={})})
    with pytest.raises(ValueError) as ei:
        extract_cell(str(tmp_path), "cyca.py", "CycA", spec, FLAGS, recurse=True)
    assert "CycA" in str(ei.value)
