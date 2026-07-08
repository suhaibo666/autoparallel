# tests/test_opdag_schema.py
from cost_eval.opdag.schema import OpNode, OpDAG

def test_opnode_roundtrip_json():
    n = OpNode(id=13, op="MatMul", src="mlp.py:141", module="ColumnParallelLinear",
               ins=["h:S·B·H:bf16", "Wfc1:H·F:bf16"], out="i:S·B·F:bf16", attrs={"bias": False})
    dag = OpDAG(cell="MLP", nodes=[n], edges=[])
    js = dag.to_json()
    back = OpDAG.from_json(js)
    assert back.nodes[0].op == "MatMul"
    assert back.nodes[0].ins == ["h:S·B·H:bf16", "Wfc1:H·F:bf16"]
    assert back.nodes[0].src == "mlp.py:141"
