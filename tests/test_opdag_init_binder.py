# tests/test_opdag_init_binder.py
from cost_eval.opdag.init_binder import bind_init

SRC = '''
from mindspore.ops.auto_generate import Mul, AddExt, SplitWithSize, Reshape, Transpose
class MLP:
    def __init__(self, config, submodules):
        self.mul = Mul()
        self.add = AddExt()
        self.split = SplitWithSize()
        self.reshape = Reshape().recompute(True)
        self.compute_dtype = config.compute_dtype
'''

def test_bind_self_names_to_canonical_optype():
    b = bind_init(SRC, "MLP")
    assert b["mul"].op == "Elementwise" and b["mul"].attrs.get("linear") is False
    assert b["add"].op == "Elementwise" and b["add"].attrs.get("linear") is True
    assert b["split"].op == "View"
    assert b["reshape"].op == "View"        # .recompute(True) 链式调用不影响类型解析
    assert "compute_dtype" not in b          # 非 op 绑定不入表
