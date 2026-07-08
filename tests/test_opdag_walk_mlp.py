# tests/test_opdag_walk_mlp.py
import pytest

from cost_eval.opdag.init_binder import Binding
from cost_eval.opdag.construct_walker import walk_construct

MLP_CONSTRUCT = '''
class MLP:
    def construct(self, hidden_states):
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias
'''

def test_walk_mlp_emits_matmul_activation_matmul():
    binds = {
        "linear_fc1": Binding(op="MatMul", attrs={"module": "ColumnParallelLinear", "compute_dtype": "bf16"}),
        "activation_func": Binding(op="Activation", attrs={"kind": "swiglu"}),
        "linear_fc2": Binding(op="MatMul", attrs={"module": "RowParallelLinear", "compute_dtype": "bf16"}),
    }
    dag = walk_construct(MLP_CONSTRUCT, "MLP", binds, src_file="mlp.py")
    ops = [n.op for n in dag.nodes]
    assert ops == ["MatMul", "Activation", "MatMul"]
    assert dag.nodes[0].src.startswith("mlp.py:")


def test_walk_mlp_wires_ssa_edges_along_the_chain():
    # 契约:中间变量被下游消费时,补一条数据流边 producer->consumer。
    binds = {
        "linear_fc1": Binding(op="MatMul", attrs={"module": "ColumnParallelLinear", "compute_dtype": "bf16"}),
        "activation_func": Binding(op="Activation", attrs={"kind": "swiglu"}),
        "linear_fc2": Binding(op="MatMul", attrs={"module": "RowParallelLinear", "compute_dtype": "bf16"}),
    }
    dag = walk_construct(MLP_CONSTRUCT, "MLP", binds, src_file="mlp.py")
    assert [n.id for n in dag.nodes] == [1, 2, 3]        # id 从 1 单调递增
    assert dag.edges == [[1, 2], [2, 3]]                 # fc1->act->fc2 数据流
    # fc1 的输入是方法形参 hidden_states → 占位 ref,不连边
    assert dag.nodes[0].ins == ["hidden_states:?:bf16"]


UNKNOWN_CONSTRUCT = '''
class Blk:
    def construct(self, x):
        y = self.mystery_op(x)
        return y
'''

def test_unknown_self_call_fails_loud():
    # 契约:Pass B 未绑定的 self.<name> 决不能静默跳过 —— 必须 fail-loud 且点名+行号。
    with pytest.raises(ValueError) as ei:
        walk_construct(UNKNOWN_CONSTRUCT, "Blk", {}, src_file="blk.py")
    msg = str(ei.value)
    assert "mystery_op" in msg and "blk.py:4" in msg


CAST_CONSTRUCT = '''
class Blk:
    def construct(self, x):
        x32 = self.cast(x, ms.float32)
        n = self.norm(x32)
        return n
'''

def test_self_cast_propagates_dtype_to_consumer():
    # 契约:op=="Cast" 时,产出变量的 dtype 变为 cast 目标 dtype,并随 SSA 传给下游消费者。
    binds = {
        "cast": Binding(op="Cast", attrs={}),
        "norm": Binding(op="Norm", attrs={}),
    }
    dag = walk_construct(CAST_CONSTRUCT, "Blk", binds, src_file="blk.py")
    assert [n.op for n in dag.nodes] == ["Cast", "Norm"]
    # cast 产出 x32:?:fp32(第 2 个位置实参 ms.float32 → fp32)
    assert dag.nodes[0].out == "x32:?:fp32"
    # norm 消费 x32,拿到传播后的 fp32 dtype,并连回 cast 节点
    assert dag.nodes[1].ins == ["x32:?:fp32"]
    assert dag.edges == [[1, 2]]


ASTYPE_CONSTRUCT = '''
class Blk:
    def construct(self, x):
        h = self.dense(x)
        h32 = h.astype(ms.float32)
        return h32
'''

def test_astype_becomes_cast_op_and_propagates_dtype():
    # 契约:<var>.astype(dtype) 也视作 Cast,发射节点、连边、传播 dtype。
    binds = {"dense": Binding(op="MatMul", attrs={"compute_dtype": "bf16"})}
    dag = walk_construct(ASTYPE_CONSTRUCT, "Blk", binds, src_file="blk.py")
    assert [n.op for n in dag.nodes] == ["MatMul", "Cast"]
    assert dag.nodes[1].out == "h32:?:fp32"           # astype 目标 dtype 传播
    assert dag.nodes[1].ins == ["h:?:bf16"]           # 接收者 h 作为输入操作数
    assert dag.edges == [[1, 2]]                       # dense -> astype-cast
