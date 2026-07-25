"""Task 2（评估文档 P0#1 门）—— 把**静默丢弃**变成大声失败 / 可断言的结构化计数。

背景（`docs/opdag_coverage_assessment_2026-07-25.md` §6.2、§11「一条必须写进设计的纪律」）：
这条流水线在**四个地方静默给错**而不是 fail-loud。最危险的实例：`CSAIndexer` 的 fused 支整块在
`with _no_grad():`（`indexer.py:214`）里 → `walk_stmt` 无 `ast.With` 处理器 → **整块静默丢，
连 `opaque_calls` 都不记** → `extract_cell` 返回 `ok, 0 nodes`。若照现状把 opdag 接进数字链路，
这一段会贡献 **0 字节而不报错** —— 于是后面每一句「它抽出来了」都是**假绿**。

本轮建立的不变量：**不可能把「全丢了」误认成「抽好了」**。
  1. **0 节点硬门（不受 `strict` 影响，恒开）**：construct 有非平凡 body 却抽出 0 节点 →
     `ExtractionDroppedError`。这一条单独就杀死了 `CSAIndexer` 的假成功。
  2. **结构化计数 `dag.diagnostics`**：未处理语句类 / 未处理赋值 RHS / 未登记的赋值目标 /
     未解析操作数 / 未绑定裸别名，各自逐条带 `src=file:line` + 原文，消费方可断言。
  3. **`strict=True`**：任一诊断非空即抛（默认 False → 既有路径逐字节不变，只多了记录）。
  4. **`assert_extraction_clean(dag)`**：消费方（`to_resolved` 适配器等）的一行门。
  5. **`crosscheck` 的空判决**：`covered=0` 却有 decoder 层 → `ok=False`（此前 `ok=True`）。
"""
import os

import pytest

from cost_eval.opdag.construct_walker import (
    DIAG_KINDS,
    ExtractionDroppedError,
    assert_extraction_clean,
    diagnostics_summary,
    walk_construct,
)
from cost_eval.opdag.init_binder import Binding, unbound_aliases

BINDS = {"act": Binding("Activation", {}), "mm": Binding("MatMul", {})}

# 单抽子 Cell 时,父本会通过构造点传入的注入项(csa.py:614 / indexer.py:131 的
# `rotary_pos_emb=...`)。RoPE 频率表 = 常量产出(无参数、无梯度)。
_INJECTED = {"rotary_pos_emb": Binding("Constant", {"rope_freqs": True})}


def _walk(src, **kw):
    return walk_construct(src, "C", BINDS, "c.py", **kw)


# ---------------------------------------------------------------------------
# 1. 未处理语句类 —— `ast.With` 是实测那一条
# ---------------------------------------------------------------------------
WITH_ONLY = '''
class C:
    def construct(self, x):
        with _no_grad():
            y = self.mm(x, x)
        return y
'''

WITH_PLUS = '''
class C:
    def construct(self, x):
        h = self.mm(x, x)
        with _no_grad():
            y = self.act(h)
        return y
'''

LOOPS = '''
class C:
    def construct(self, x):
        h = self.mm(x, x)
        for i in range(3):
            h = self.act(h)
        while True:
            h = self.act(h)
        try:
            h = self.act(h)
        except Exception:
            pass
        h += self.act(h)
        return h
'''


UNKNOWN_WITH = '''
class C:
    def construct(self, x):
        h = self.mm(x, x)
        with some_unknown_ctx():
            y = self.act(h)
        return y
'''


def test_with_no_grad_body_is_walked_and_marked_detached():
    """P0#4:`with _no_grad():` 的块体**要走查**，块内产物标 `detached`（不是丢、也不是当普通节点发）。

    这条替代了原先的「整块记 dropped_stmts」断言：同一条不变量（`_no_grad` 是整块 detach 的信号）
    现在由**节点上的 detached 标记**承载，比只记一行诊断强。
    """
    dag = _walk(WITH_ONLY, config_flags={})
    assert [n.op for n in dag.nodes] == ["MatMul"]                 # 块体真的走进去了
    assert dag.diagnostics["dropped_stmts"] == []                  # 不再是"被丢弃"
    assert dag.nodes[0].attrs.get("detached") is True              # 块内产物 = detached
    assert dag.nodes[0].attrs.get("no_grad_region") is True
    assert dag.detached == ["y"]                                   # 名册上浮到 DAG


def test_with_no_grad_marks_only_the_block_body():
    """块外的节点**不得**被误标 detached（边界要准）。"""
    dag = _walk(WITH_PLUS, config_flags={})
    assert [n.op for n in dag.nodes] == ["MatMul", "Activation"]
    assert dag.nodes[0].attrs.get("detached") is None              # 块外
    assert dag.nodes[1].attrs.get("detached") is True              # 块内
    assert dag.detached == ["y"]


def test_unknown_with_context_is_fail_loud():
    """语义判不出来的 `with` → **fail-loud**，绝不猜它改不改梯度可达性（任务纪律）。"""
    with pytest.raises(ValueError) as ei:
        _walk(UNKNOWN_WITH, config_flags={})
    msg = str(ei.value)
    assert "c.py:5" in msg and "some_unknown_ctx" in msg


def test_for_while_try_all_recorded():
    """`For`/`While`/`Try` 仍未支持 → 仍逐条记账（`AugAssign` 已被 P0#5 支持，见下条）。"""
    dag = _walk(LOOPS, config_flags={})
    kinds = [d["node"] for d in dag.diagnostics["dropped_stmts"]]
    assert kinds == ["For", "While", "Try"]
    assert all(d["src"].startswith("c.py:") for d in dag.diagnostics["dropped_stmts"])


def test_augassign_on_a_tensor_becomes_a_node():
    """`h += self.act(h)`(LOOPS 末行)等价于 `h = h + self.act(h)` → 建节点，不再记丢弃。"""
    dag = _walk(LOOPS, config_flags={})
    adds = [n for n in dag.nodes if n.op == "Elementwise" and n.attrs.get("arith") == "add"]
    assert adds and adds[-1].src == "c.py:13"     # LOOPS 里 `h += self.act(h)` 那一行
    assert adds[-1].attrs["linear"] is True       # `+` 反向直通 → 不存激活
    assert "AugAssign" not in [d["node"] for d in dag.diagnostics["dropped_stmts"]]


def test_docstring_is_not_counted_as_a_drop():
    src = '''
class C:
    def construct(self, x):
        """docstring 不是被丢的语句。"""
        return self.mm(x, x)
'''
    dag = _walk(src, config_flags={})
    assert dag.diagnostics["dropped_stmts"] == []


# ---------------------------------------------------------------------------
# 2. 未处理赋值 RHS（BinOp / Compare / Subscript）
# ---------------------------------------------------------------------------
RHS = '''
class C:
    def construct(self, x, y):
        h = self.mm(x, y)
        q = h * 0.5
        m = h > 0
        s = h[:2]
        a = h
        w = self.weight
        return self.act(q)
'''


UNSUPPORTED_RHS = '''
class C:
    def construct(self, x, y):
        h = self.mm(x, y)
        d = {"k": h}
        f = lambda t: t
        g = f"{h}"
        return self.act(h)
'''


def test_tensor_rhs_forms_now_become_nodes():
    """P0#5:`BinOp`/`Compare`/`Subscript` 的**张量** RHS 建节点（此前全进 dropped_assigns）。

    原断言「这五条 RHS 全被丢」是在钉病症；同一份合成源现在必须产出对应节点。
    """
    dag = _walk(RHS, config_flags={})
    by_src = {n.src: n for n in dag.nodes}
    assert by_src["c.py:5"].op == "Elementwise"          # q = h * 0.5   (tensor × 标量 → 线性)
    assert by_src["c.py:5"].attrs["linear"] is True
    assert by_src["c.py:6"].op == "Compare"              # m = h > 0     (bool、不可微)
    assert by_src["c.py:6"].out.endswith(":bool")
    assert by_src["c.py:7"].op == "View"                 # s = h[:2]     (切片 = 视图)
    assert by_src["c.py:7"].attrs["view"] == "slice"
    # 只剩 `w = self.weight` 一条:合成源**没有 `__init__`**,故 walker 无从知道 `self.weight`
    # 是 Parameter 还是别的东西 → 保持 fail-loud 记账(这是对的:不猜)。给了 `self_kinds`
    # 的真路径上它会变成 param 操作数,见 `test_weight_attr_becomes_a_param_operand`。
    assert [d["src"] for d in dag.diagnostics["dropped_assigns"]] == ["c.py:9"]


def test_alias_and_weight_rhs_do_not_fabricate_nodes():
    """`a = h`（别名）与 `w = self.weight`（权重）**不建节点**，但也不再"丢边"。"""
    dag = _walk(RHS, config_flags={})
    assert not [n for n in dag.nodes if n.src in ("c.py:8", "c.py:9")]
    # `a = h` → 别名到 h 的 producer(边接得回去),不再进 unregistered_targets
    unreg = {d["target"] for d in dag.diagnostics["unregistered_targets"]}
    assert "a" not in unreg
    assert unreg == {"w"}          # 只剩没有 __init__ 信息的 self.weight(见上条注释)


def test_weight_attr_becomes_a_param_operand():
    """给了 `self_kinds`(由 `init_dims` 静态求值 `__init__` 得)后,`self.weight` 是**权重**:

    契约(W2/W3):权重**不进 `ins`**、**不是任何节点的 out** —— 否则 `derive_saves` 会把它
    当激活 save 计(实测 FFNGroupedGEMM「236 MiB」里 88 MiB 就是这个病)。它单列 param_operands。
    """
    src = '''
class C:
    def construct(self, x):
        w = self.weight
        return self.mm(x, w)
'''
    dag = walk_construct(src, "C", BINDS, "c.py", config_flags={},
                         self_kinds={"weight": "param"})
    mm = [n for n in dag.nodes if n.op == "MatMul"][0]
    assert mm.ins == ["x:?:bf16"]                       # 权重不在 ins 里
    assert mm.attrs["param_operands"] == ["weight"]     # 但**可见**
    # 两处出现点都被记(:4 的别名赋值 + :5 的消费点)——出现点越全,消费方越好对账。
    assert [r["param"] for r in dag.param_operands] == ["weight", "weight"]
    assert {r["src"] for r in dag.param_operands} == {"c.py:4", "c.py:5"}
    assert all(n.out.split(":")[0] != "weight" for n in dag.nodes)   # 权重不是任何 out
    assert diagnostics_summary(dag)["total"] == 0


def test_downstream_consumer_now_gets_a_real_ref_not_a_placeholder():
    """原「丢边」的可见证据反过来用:下游 `self.act(q)` 现在拿到真 ref + 真边。"""
    dag = _walk(RHS, config_flags={})
    act = [n for n in dag.nodes if n.op == "Activation"][0]
    q_node = [n for n in dag.nodes if n.src == "c.py:5"][0]
    assert act.ins == ["q:?:bf16"]
    assert [q_node.id, act.id] in dag.edges           # **边在**（此前没有）
    assert dag.diagnostics["unresolved_operands"] == []


def test_still_unsupported_rhs_forms_are_recorded_and_loud():
    """机制不变:真的还不支持的 RHS 形态(Dict/Lambda/JoinedStr)仍逐条记账 + strict 下抛。"""
    dag = _walk(UNSUPPORTED_RHS, config_flags={})
    got = [(d["src"], tuple(d["targets"]), d["rhs"]) for d in dag.diagnostics["dropped_assigns"]]
    assert got == [("c.py:5", ("d",), "Dict"), ("c.py:6", ("f",), "Lambda"),
                   ("c.py:7", ("g",), "JoinedStr")]
    assert all("code" in d for d in dag.diagnostics["dropped_assigns"])
    unreg = {d["target"] for d in dag.diagnostics["unregistered_targets"]}
    assert {"d", "f", "g"} <= unreg
    with pytest.raises(ExtractionDroppedError) as ei:
        _walk(UNSUPPORTED_RHS, config_flags={}, strict=True)
    msg = str(ei.value)
    for line in ("c.py:5", "c.py:6", "c.py:7"):
        assert line in msg


# ---------------------------------------------------------------------------
# 3. opaque 调用的赋值目标（终端 fallthrough）
# ---------------------------------------------------------------------------
OPAQUE = '''
class C:
    def construct(self, x):
        h = self.mm(x, x)
        t = self.dispatcher.permute(h)
        return self.act(t)
'''


def test_opaque_call_target_is_recorded_as_unregistered():
    dag = _walk(OPAQUE, config_flags={})
    assert len(dag.opaque_calls) == 1                     # 既有机制不变
    unreg = dag.diagnostics["unregistered_targets"]
    assert [(d["src"], d["target"], d["cause"]) for d in unreg] == [
        ("c.py:5", "t", "opaque_call")]


# ---------------------------------------------------------------------------
# 4. 0 节点硬门（恒开，不受 strict 影响）
# ---------------------------------------------------------------------------
ALL_OPAQUE = '''
class C:
    def construct(self, x):
        a = self.dispatcher.permute(x)
        b = self.dispatcher.unpermute(a)
        return b
'''

TRIVIAL = '''
class C:
    def construct(self, x):
        """恒等 Cell（Identity）。"""
        return x
'''


def test_zero_nodes_with_nontrivial_body_raises_even_without_strict():
    with pytest.raises(ExtractionDroppedError) as ei:
        _walk(ALL_OPAQUE, config_flags={})
    msg = str(ei.value)
    assert "0" in msg and "opaque" in msg.lower()


def test_zero_nodes_is_allowed_for_a_trivial_body():
    """`return x` / `pass` 这类恒等/空 Cell 合法 0 节点 —— 不得误伤。"""
    dag = _walk(TRIVIAL, config_flags={})
    assert dag.nodes == []
    assert diagnostics_summary(dag)["total"] == 0


# ---------------------------------------------------------------------------
# 5. diagnostics_summary / assert_extraction_clean
# ---------------------------------------------------------------------------
def test_diagnostics_summary_counts_every_kind():
    dag = _walk(UNSUPPORTED_RHS, config_flags={})
    s = diagnostics_summary(dag)
    assert set(s) == set(DIAG_KINDS) | {"total", "opaque_calls"}
    assert s["dropped_assigns"] == 3
    assert s["total"] == sum(s[k] for k in DIAG_KINDS)


def test_assert_extraction_clean_raises_with_readable_listing():
    dag = _walk(UNSUPPORTED_RHS, config_flags={})
    with pytest.raises(ExtractionDroppedError) as ei:
        assert_extraction_clean(dag)
    msg = str(ei.value)
    assert "dropped_assigns" in msg and "c.py:5" in msg and "C" in msg


def test_assert_extraction_clean_allows_explicit_waivers():
    dag = _walk(UNSUPPORTED_RHS, config_flags={})
    assert_extraction_clean(dag, allow=("dropped_assigns", "unregistered_targets",
                                        "unresolved_operands"))


def test_assert_extraction_clean_passes_on_a_clean_dag():
    src = '''
class C:
    def construct(self, x):
        h = self.mm(x, x)
        return self.act(h)
'''
    dag = _walk(src, config_flags={})
    assert_extraction_clean(dag, check_opaque=True)
    assert diagnostics_summary(dag)["total"] == 0


# ---------------------------------------------------------------------------
# 6. init_binder：pynative 的「裸函数别名」惯用法必须被计数（评估文档 §2.4：39 原语 / 196 调用点）
# ---------------------------------------------------------------------------
ALIAS_INIT = '''
class C:
    def __init__(self, config):
        self.reshape = mint.reshape
        self.cast = ops.cast
        self.permute = mint.permute
        self.topk = mint.topk
        self.weird = mint.some_unreviewed_primitive
        self.mul = Mul()
        self.n_heads = config.num_attention_heads
        self.flag = True
'''


def test_bare_function_aliases_are_bound_and_unknown_ones_enumerated():
    """P0#2:能查 `primitives.PRIMITIVES` 的裸别名**绑上**;查不到的仍逐条列出(该加表项的清单)。

    原断言「这四条裸别名全未绑」是在钉病症;现在它们必须绑成正确的 op 类型,
    而 `unbound_aliases` 的语义收窄为「真·未知原语」——不变量(不得静默跳过)保持。
    """
    from cost_eval.opdag.init_binder import bind_init
    binds = bind_init(ALIAS_INIT, "C")
    assert binds["reshape"].op == "View" and binds["reshape"].attrs["view"] == "reshape"
    assert binds["cast"].op == "Cast"
    assert binds["permute"].op == "View"
    assert binds["topk"].op == "TopK"
    assert binds["mul"].op == "Elementwise"                       # `Mul()` 走既有 _CLS2OP
    assert "n_heads" not in binds and "flag" not in binds         # 配置读取不是算子别名

    got = unbound_aliases(ALIAS_INIT, "C")
    assert [(a["attr"], a["alias"]) for a in got] == [
        ("weird", "mint.some_unreviewed_primitive")]
    assert all(a["src"].startswith("c") is False for a in got)   # src 由调用方给文件名
    assert all("lineno" in a for a in got)


def test_calling_an_unknown_bare_alias_is_fail_loud():
    """未知原语被**调用** → `UnknownPrimitiveError`,报错指名道姓说"加哪条表项"(绝不猜类别)。"""
    from cost_eval.opdag.primitives import UnknownPrimitiveError
    from cost_eval.opdag.init_binder import bind_init
    src = ALIAS_INIT + '''
    def construct(self, x):
        return self.weird(x)
'''
    with pytest.raises(UnknownPrimitiveError) as ei:
        walk_construct(src, "C", bind_init(src, "C"), "c.py", config_flags={},
                       alias_unknown={a["attr"]: a for a in unbound_aliases(src, "C", "c.py")})
    msg = str(ei.value)
    assert "mint.some_unreviewed_primitive" in msg and "primitives" in msg


# ---------------------------------------------------------------------------
# 7. crosscheck：mHC 名匹配 + 「覆盖 0 不得为绿」
# ---------------------------------------------------------------------------
def test_split_decoder_matches_mhc_renamed_residual():
    """mHC 开启后残差输出名带 `_xn` 后缀（`residual.py:62` `{name}_xn`）→ 探针须命中。"""
    from cost_eval.model_spec import OpSpec, OpType, TensorRef
    from cost_eval.opdag.crosscheck import _split_decoder

    def _op(name, out):
        return OpSpec(name, OpType.MATMUL, [], TensorRef(out, ("S", "B", "H")))

    plain = [_op("a", "x"), _op("add1", "h1"), _op("b", "y")]
    assert _split_decoder(plain) == (plain[:2], plain[2:])
    mhc = [_op("a", "x"), _op("add1", "h1_xn"), _op("b", "y")]
    attn, ffn = _split_decoder(mhc)
    assert attn is not None and [o.name for o in attn] == ["a", "add1"]
    assert [o.name for o in ffn] == ["b"]
    # 非 decoder body（无 h1*）→ 仍 (None, None)
    assert _split_decoder([_op("emb", "z")]) == (None, None)
    # `h2*` 不得被误当 attn/ffn 边界
    assert _split_decoder([_op("a", "x"), _op("moe_add", "h2_xn")]) == (None, None)


def test_zero_coverage_is_not_green():
    """含注意力的层段一个都没被校验到 → `ok=False`（此前 `covered=0` 也返回 True = 假绿）。"""
    from cost_eval.opdag.crosscheck import CrossCheckReport

    rep = CrossCheckReport(available=True, mf_root="x")
    rep.attn_layers.append("layer0")
    assert rep.ok is False
    assert rep.coverage_findings and "空判决" in rep.summary()


def test_zero_coverage_reproduces_the_reported_dsv4_shape():
    """评估文档 §1 的**原始病症形状**：covered=0、layer_norm_checked 只有 `lm_head`
    （来自 A5 final_norm 支，不是 attn 层）、decoder_bodies=0（h1 探针失配）→ 必须非绿。"""
    from cost_eval.opdag.crosscheck import CrossCheckReport

    rep = CrossCheckReport(available=True, mf_root="x")
    rep.attn_layers.extend(["dsv4hyb_r0_dense", "dsv4hyb_r4_moe"])
    rep.layer_norm_checked.append("lm_head")
    rep.uncovered.extend([(lt, "无 opdag 提取源") for lt in rep.attn_layers])
    assert rep.validated_attn_layers == []
    assert rep.ok is False


def test_zero_coverage_rule_does_not_fire_when_layer_norms_validated():
    """全 GQA/dense **合法**无 MLA/MoE 对应，但其 decoder body 被 layer_norms 族校验 → 仍绿。"""
    from cost_eval.opdag.crosscheck import CrossCheckReport

    rep = CrossCheckReport(available=True, mf_root="x")
    rep.attn_layers.append("gqa0")
    rep.layer_norm_checked.append("gqa0")          # layer_norms 族真校验过它
    assert rep.validated_attn_layers == ["gqa0"]
    assert rep.coverage_findings == [] and rep.ok is True


def test_zero_coverage_rule_does_not_fire_without_source_or_attn_layer():
    from cost_eval.opdag.crosscheck import CrossCheckReport
    # 缺源 → 无从校验 → 仍 True（既有契约 test_missing_source_skips_gracefully）
    rep = CrossCheckReport(available=False, mf_root="x")
    rep.attn_layers.append("layer0")
    assert rep.ok is True
    # 有源但 spec 无含注意力层段（纯 embedding/head toy spec）→ 不误伤
    assert CrossCheckReport(available=True, mf_root="x").ok is True


@pytest.mark.skipif(
    not os.path.isdir(__import__("cost_eval.opdag.crosscheck", fromlist=["x"]).default_mf_root()),
    reason="mindformers 源不可用（CI）")
def test_dsv4_crosscheck_now_really_covers_something():
    """`h1_*` 探针修复的**具体收益**：DSv4 从「covered=0 / layer_norm_checked=[lm_head]」
    （评估文档 §1 实测）变成 MoE 专家核逐层 census + 全 decoder body 的 pre-norm 名册校验。"""
    from cost_eval.build_llm import build_llm_spec
    from cost_eval.opdag.crosscheck import validate_against_opdag
    from cost_eval.presets import deepseek_v4

    spec = build_llm_spec(deepseek_v4(6))
    rep = validate_against_opdag(spec, warn=False)
    dsv4_layers = [lt for lt in spec.layer_specs if lt.startswith("dsv4hyb")]
    assert rep.available
    # mtp 层内含一个 decoder body → 也带注意力 op；它被路由出 delta 家族、只校验 enorm/hnorm。
    assert rep.attn_layers == dsv4_layers + ["mtp"]
    assert rep.decoder_bodies == dsv4_layers              # h1_xn 探针现在全部命中
    assert {c.family for c in rep.covered} == {"moe_experts"}
    assert set(dsv4_layers) <= set(rep.layer_norm_checked)   # 此前只有 ['lm_head']
    assert rep.validated_attn_layers == rep.attn_layers
    assert rep.coverage_findings == []                    # 非空判决 → 这是**真**绿
    assert rep.ok and not rep.findings


# ---------------------------------------------------------------------------
# 8. 真源验收：`CSAIndexer` 不再 `OK -> 0 nodes`
# ---------------------------------------------------------------------------
from tests.test_opdag_fn_saves import authoritative_variant_dir  # noqa: E402

_SPEC_FLAGS = {
    "num_experts": 8, "moe_grouped_gemm": True, "qk_layernorm": True,
    "multi_latent_attention": True, "enable_hyper_connections": True,
    "fused_norm": True, "normalization": "RMSNorm", "is_dsv4_hybrid": True,
}
_CELL_FLAGS = {
    "compute_dtype": "bf16", "layernorm_compute_dtype": "fp32",
    "v_head_dim": 512, "qk_pos_emb_head_dim": 64, "q_lora_rank": 1024,
    "num_attention_heads": 64, "hidden_size": 4096, "o_groups": 8, "o_lora_rank": 1024,
    "add_bias_linear": False, "input_layout": "BSND", "num_layers": 8, "mtp_num_layers": 0,
    "index_topk": 512, "index_n_heads": 64, "index_head_dim": 128,
    "csa_window_size": 128, "csa_dense_mode": False,
    # `__init__` 派生量（walker 今天算不出，须手喂 —— 评估文档 P0#6）
    "compress_ratio": 4, "enable_compress": True, "enable_indexer": True,
    "is_tnd": False, "window_size": 128, "training": True,
    "apply_dsa_kernel_fusion": True,
}


@pytest.fixture(scope="module")
def mf_pkg():
    d = authoritative_variant_dir()
    if d is None:
        pytest.skip("权威 mindformers 快照缺失（见 tests/test_opdag_fn_saves.py 的 md5 门控）")
    # <pkg>/pynative/transformers/experimental_attention_variant → 上溯 3 级到包目录 <pkg>
    return os.path.abspath(os.path.join(d, "..", "..", ".."))


def test_csaindexer_fused_branch_is_walked_and_fully_detached(mf_pkg):
    """评估文档 §3.1「最危险的一条」的**最终**验收(P0#4):

    `CSAIndexer` fused 支整块在 `with _no_grad():`(`indexer.py:214`,区域 `:214-232`)里。
    `7b9aa86` 把它从「假成功 0 节点」改成了「抛」;本轮把它改成**真抽出来**,且块内产物
    **全部标 detached**(这才是 `_no_grad` 的语义)。三段演进的不变量始终是同一条:
    **不可能把「全丢了」误认成「抽好了」**。
    """
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import PYNATIVE_SPEC_FILES, resolve_layer_spec

    top = resolve_layer_spec(mf_pkg, _SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    sa = top.submodules["self_attention"]
    assert sa.cell == "DSv4HybridSelfAttention"        # 断言解的是**目标模型**(不是 DSv3 MLA)
    csa = sa.submodules["core_attention"]
    rel = "pynative/transformers/experimental_attention_variant/indexer.py"
    dag = extract_cell(mf_pkg, rel, "CSAIndexer", csa.submodules["indexer"], _CELL_FLAGS,
                       present_params={"rotary_pos_emb"}, recurse=True, subcell_specs={},
                       cross_file=True, injected_binds=_INJECTED,
                       kernel_call_allow=("npu_lightning_indexer",))
    assert dag.nodes, "fused 支必须抽出非空图"
    assert all(n.src.startswith("indexer.py:") for n in dag.nodes)
    # 块内(`:214-232`)产出的每个节点都必须带 detached + no_grad_region
    for n in dag.nodes:
        line = int(n.src.split(":")[1])
        assert 214 <= line <= 232, n.src
        assert n.attrs.get("detached") is True, n.src
        assert n.attrs.get("no_grad_region") is True, n.src
    # 源真值(fn_saves 已核):块内绑定 q/k/weights/key_length/cmp_residual_k/topk_indices/index_scores
    assert {"q", "k", "weights", "topk_indices", "index_scores"} <= set(dag.detached)
    assert diagnostics_summary(dag)["total"] == 0      # 零诊断
    # 融合内核在 no-grad 区里 → **可证明**无反向 → saved 集为空(而不是"猜它没有")
    kern = [n for n in dag.nodes if n.op == "Kernel"]
    assert len(kern) == 1 and kern[0].attrs["kernel"] == "npu_lightning_indexer"
    assert kern[0].attrs["saved_ins_idx"] == []
    assert "no_grad" in kern[0].attrs["no_backward_reason"]


def test_working_dsv3_mla_path_scalar_forms_are_handled_not_dropped():
    """反向验收(P0#5):DSv3 MLA 路径上那两条**标量/dtype** RHS 现在被**正确处理**,而不是记丢弃。

    `7b9aa86` 实测的两条 `dropped_assigns` 是 `ori_dtype = x.dtype`(Attribute)与
    `head_dim = query.shape[-1]`(Subscript)。它们**是标量/dtype 记号,不是张量** ——
    正确行为是「进标量/dtype 环境、**不建节点**」,而不是「记一笔丢弃」。判据:
      * 该路径 `assert_extraction_clean` 通过(零诊断);
      * 这两行**没有**产生节点(没有造假张量);
      * `derive_saves` 的名册与 `7b9aa86` 逐字节相同(字节中性)。
    """
    from cost_eval.opdag.crosscheck import default_mf_root
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import resolve_layer_spec

    mf = default_mf_root()
    if not os.path.isdir(mf):
        pytest.skip(f"mindformers 源根不存在: {mf}")
    spec_flags = {
        "multi_latent_attention": True, "mla_qkv_concat": False, "num_experts": 256,
        "qk_layernorm": True, "sparse_attention": False, "fused_norm": True,
        "moe_grouped_gemm": True, "use_contiguous_weight_layout_attention": False,
        "use_interleaved_weight_layout_mlp": True,
    }
    mla_flags = {
        "use_dsa": False, "use_flash_attention": True,
        "use_eod_attn_mask_compression": False, "cp": 1, "cp_ds": 1, "input_layout": "BNSD",
        "q_lora_rank": 1536, "compute_dtype": "bf16", "layernorm_compute_dtype": "fp32",
    }
    top = resolve_layer_spec(mf, spec_flags)
    dag = extract_cell(
        mf, "parallel_core/training_graph/transformer/multi_latent_attention.py",
        "MLASelfAttention", top.submodules["self_attention"], mla_flags,
        present_params={"rotary_pos_emb"})
    assert len(dag.nodes) > 0                       # 这条路径本来就通(不受新门影响)
    s = diagnostics_summary(dag)
    assert s["total"] == 0, dag.diagnostics         # 零诊断 —— 消费方门现在**放行**
    assert_extraction_clean(dag)
    # `ori_dtype = x.dtype`(:232)与 `head_dim = query.shape[-1]`(:258)不得产生任何节点
    lines = {int(n.src.split(":")[1]) for n in dag.nodes}
    assert 232 not in lines and 258 not in lines
    # 下游 `self.cast(..., ori_dtype)`(:307)的 dtype 现在解成**真 dtype**,
    # 而不是字面串 "ori_dtype"(那是个假 dtype)
    cast307 = [n for n in dag.nodes if n.src.endswith(":307")]
    assert cast307 and cast307[0].out.split(":")[2] in ("bf16", "fp32", "fp16")
    # 字节中性:saves 名册与 `7b9aa86` 逐字相同(该 commit 实跑值)
    from cost_eval.opdag.bprop_rules import derive_saves
    names = sorted(sv.name for sv in derive_saves(dag))
    assert names == ["attn_out", "key", "kv_compressed__i0", "q_compressed__i0",
                     "query", "value", "x"], names
