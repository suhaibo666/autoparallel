# tests/test_opdag_bprop.py
from cost_eval.opdag.schema import OpNode, OpDAG
from cost_eval.opdag.bprop_rules import derive_saves

def _dag(*nodes):
    return OpDAG(cell="T", nodes=list(nodes),
                 edges=[[nodes[i].id, nodes[i+1].id] for i in range(len(nodes)-1)])

def test_matmul_pins_both_operands():
    d = _dag(OpNode(id=1, op="MatMul", src="x:1", ins=["a:S·B·H:bf16", "W:H·F:bf16"], out="y:S·B·F:bf16"))
    saves = derive_saves(d)
    names = {s.name for s in saves}
    assert "a" in names and "W" in names

def test_cast_not_self_saved_but_output_pinned_by_norm_consumer_at_fp32():
    cast = OpNode(id=1, op="Cast", src="x:1", ins=["h:S·B·H:bf16"], out="h32:S·B·H:fp32", attrs={"to_dtype": "fp32"})
    norm = OpNode(id=2, op="Norm", src="x:2", ins=["h32:S·B·H:fp32"], out="n:S·B·H:bf16")
    d = OpDAG(cell="T", nodes=[cast, norm], edges=[[1, 2]])
    saves = derive_saves(d)
    h32 = [s for s in saves if s.name == "h32"]
    assert h32 and h32[0].dtype == "fp32"
    assert not any(s.op_id == 1 for s in saves)

def test_add_residual_pins_nothing():
    d = _dag(OpNode(id=1, op="Elementwise", src="x:1", module="AddExt",
                    ins=["a:S·B·H:bf16", "b:S·B·H:bf16"], out="y:S·B·H:bf16", attrs={"linear": True}))
    assert derive_saves(d) == []
