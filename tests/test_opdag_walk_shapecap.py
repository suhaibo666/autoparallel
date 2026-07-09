# tests/test_opdag_walk_shapecap.py
"""walker 的 **View 变换元信息捕获**(T9 shape 推断的输入):reshape 目标 / split 尺寸+轴+目标 /
transpose perm / expand 轴 / tile 倍数 / shape 解包源,以及 `x.shape` 解包(不产 op)记入 dag.scalar_binds。
捕获的是**未求值的符号表达式串**(shape 推断阶段才代入),walker 本身仍不跟 shape(ref 仍 `?`)。"""
from cost_eval.opdag.init_binder import Binding
from cost_eval.opdag.construct_walker import walk_construct


CAP = '''
class C:
    def construct(self, x):
        seq_len, bs, _ = self.shape(x)
        q = self.up(x)
        q = self.reshape(q, (seq_len, bs, self.num_attention_heads, self.q_head_dim))
        q_no_pe, q_pe = self.split(q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1)
        k = self.expand(q_pe, 2)
        k = self.tile(k, (1, 1, self.num_attention_heads, 1))
        key = self.cat([q_no_pe, k])
        t = self.tr(key, (0, 1, 3, 2))
        seq2, bs2, feat = key.shape
        return t
'''

BINDS = {
    "shape": Binding("View", {"view": "shape"}),
    "up": Binding("MatMul", {}),
    "reshape": Binding("View", {"view": "reshape"}),
    "split": Binding("View", {"view": "split"}),
    "expand": Binding("View", {"view": "expand_dims"}),
    "tile": Binding("View", {"view": "tile"}),
    "cat": Binding("View", {"variadic": True, "view": "concat", "concat_axis": 3}),
    "tr": Binding("View", {"view": "transpose"}),
}


def _by(dag, view):
    return next(n for n in dag.nodes if n.attrs.get("view") == view)


def test_shape_node_captures_src_and_unpack():
    dag = walk_construct(CAP, "C", BINDS, "c.py")
    n = _by(dag, "shape")
    assert n.attrs["shape_src"] == "x"
    assert n.attrs["shape_unpack"] == ["seq_len", "bs", "_"]


def test_reshape_captures_target_dim_exprs():
    dag = walk_construct(CAP, "C", BINDS, "c.py")
    n = _by(dag, "reshape")
    assert n.attrs["reshape_dims"] == [
        "seq_len", "bs", "self.num_attention_heads", "self.q_head_dim"]


def test_split_captures_sizes_dim_and_targets():
    dag = walk_construct(CAP, "C", BINDS, "c.py")
    n = _by(dag, "split")
    assert n.attrs["split_sizes"] == ["self.config.qk_head_dim", "self.config.qk_pos_emb_head_dim"]
    assert n.attrs["split_dim"] == -1
    assert n.attrs["split_targets"] == ["q_no_pe", "q_pe"]


def test_expand_and_tile_and_transpose_capture():
    dag = walk_construct(CAP, "C", BINDS, "c.py")
    assert _by(dag, "expand_dims").attrs["expand_axis"] == 2
    assert _by(dag, "tile").attrs["tile_mult"] == ["1", "1", "self.num_attention_heads", "1"]
    assert _by(dag, "transpose").attrs["perm"] == [0, 1, 3, 2]


def test_dot_shape_unpack_recorded_in_scalar_binds():
    dag = walk_construct(CAP, "C", BINDS, "c.py")
    assert {"names": ["seq2", "bs2", "feat"], "src": "key"} in dag.scalar_binds


def test_walker_still_emits_placeholder_shape():
    # 契约不变:walker 本身 shape 仍是 `?`(推断在 shape_infer 独立 pass)。
    dag = walk_construct(CAP, "C", BINDS, "c.py")
    up = next(n for n in dag.nodes if n.op == "MatMul")
    assert up.out == "q:?:bf16"
