from cost_eval.model_spec import (
    DimTable, TensorRef, OpSpec, OpType, LayerSpec, ModelSpec)

def test_dimtable_as_dict_exposes_symbols():
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)
    assert d.as_dict()["H"] == 8 and d.as_dict()["n_kv"] == 2

def test_tensorref_defaults():
    t = TensorRef("x", ("S", "B", "H"))
    assert t.shard == {} and t.is_weight is False and t.partial is None

def test_opspec_holds_memory_contract():
    w = TensorRef("w", ("H", "F"), shard={1: "tp"}, is_weight=True)
    o = OpSpec("fc", OpType.MATMUL, inputs=[TensorRef("x", ("S", "B", "H"))],
               output=TensorRef("y", ("S", "B", "F"), shard={2: "tp"}),
               params=[w], saves=[TensorRef("x", ("S", "B", "H"))])
    assert o.params[0].is_weight and o.saves[0].name == "x"

def test_modelspec_layer_lookup():
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)
    ls = LayerSpec(ops=[])
    m = ModelSpec("toy", d, layer_pattern=["dense", "dense"], layer_specs={"dense": ls})
    assert m.get_layer("dense") is ls
