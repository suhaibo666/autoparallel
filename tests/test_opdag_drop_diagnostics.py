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


def test_with_only_body_raises_instead_of_zero_nodes():
    """整个 construct 都在 `with` 里（= `CSAIndexer` fused 支的形状）→ 抛，不再 `ok, 0 nodes`。"""
    with pytest.raises(ExtractionDroppedError) as ei:
        _walk(WITH_ONLY, config_flags={})
    msg = str(ei.value)
    assert "0 " in msg and "c.py:4" in msg          # 指到 `with` 那一行
    assert "With" in msg


def test_with_is_recorded_as_dropped_stmt_when_other_nodes_exist():
    """有别的节点 → 0 节点门不触发，但 `With` 必须逐条记账（非静默）。"""
    dag = _walk(WITH_PLUS, config_flags={})
    assert [n.op for n in dag.nodes] == ["MatMul"]          # act 在 with 里，被丢
    drops = dag.diagnostics["dropped_stmts"]
    assert len(drops) == 1
    assert drops[0]["node"] == "With"
    assert drops[0]["src"] == "c.py:5"
    assert "_no_grad" in drops[0]["code"]
    # `_no_grad` 是**整块 detach** 的信号 —— 记进 note，供路线 B P0#4 接 detach 语义
    assert "detach" in drops[0]["note"]


def test_with_dropped_is_loud_under_strict():
    with pytest.raises(ExtractionDroppedError) as ei:
        _walk(WITH_PLUS, config_flags={}, strict=True)
    assert "c.py:5" in str(ei.value)


def test_for_while_try_augassign_all_recorded():
    dag = _walk(LOOPS, config_flags={})
    kinds = [d["node"] for d in dag.diagnostics["dropped_stmts"]]
    assert kinds == ["For", "While", "Try", "AugAssign"]
    assert all(d["src"].startswith("c.py:") for d in dag.diagnostics["dropped_stmts"])


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


def test_unhandled_assign_rhs_recorded_with_targets_and_kind():
    dag = _walk(RHS, config_flags={})
    got = [(d["src"], tuple(d["targets"]), d["rhs"]) for d in dag.diagnostics["dropped_assigns"]]
    assert got == [
        ("c.py:5", ("q",), "BinOp"),
        ("c.py:6", ("m",), "Compare"),
        ("c.py:7", ("s",), "Subscript"),
        ("c.py:8", ("a",), "Name"),
        ("c.py:9", ("w",), "Attribute"),
    ]
    assert all("code" in d for d in dag.diagnostics["dropped_assigns"])


def test_dropped_assign_target_is_flagged_unregistered():
    """被丢的赋值目标从未进 SSA → 下游消费它会拿占位 ref、丢边。必须显式记账。"""
    dag = _walk(RHS, config_flags={})
    unreg = {d["target"] for d in dag.diagnostics["unregistered_targets"]}
    assert {"q", "m", "s", "a", "w"} <= unreg
    # 下游 `self.act(q)` 确实拿到了占位 ref（这是「丢边」的可见证据）
    act = [n for n in dag.nodes if n.op == "Activation"][0]
    assert act.ins == ["q:?:bf16"]
    assert any(d["operand"] == "q" for d in dag.diagnostics["unresolved_operands"])


def test_strict_reports_every_kind_with_src():
    with pytest.raises(ExtractionDroppedError) as ei:
        _walk(RHS, config_flags={}, strict=True)
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
    dag = _walk(RHS, config_flags={})
    s = diagnostics_summary(dag)
    assert set(s) == set(DIAG_KINDS) | {"total", "opaque_calls"}
    assert s["dropped_assigns"] == 5
    assert s["total"] == sum(s[k] for k in DIAG_KINDS)


def test_assert_extraction_clean_raises_with_readable_listing():
    dag = _walk(RHS, config_flags={})
    with pytest.raises(ExtractionDroppedError) as ei:
        assert_extraction_clean(dag)
    msg = str(ei.value)
    assert "dropped_assigns" in msg and "c.py:5" in msg and "C" in msg


def test_assert_extraction_clean_allows_explicit_waivers():
    dag = _walk(RHS, config_flags={})
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
        self.mul = Mul()
        self.n_heads = config.num_attention_heads
        self.flag = True
'''


def test_bare_function_aliases_are_enumerated():
    """`self.reshape = mint.reshape` 这类裸别名 —— `_CLS2OP` 绑不上，但必须被列出、不得静默跳过。"""
    got = unbound_aliases(ALIAS_INIT, "C")
    assert [(a["attr"], a["alias"]) for a in got] == [
        ("reshape", "mint.reshape"), ("cast", "ops.cast"),
        ("permute", "mint.permute"), ("topk", "mint.topk"),
    ]
    assert all(a["src"].startswith("c") is False for a in got)   # src 由调用方给文件名
    assert all("lineno" in a for a in got)
    # `Mul()` 已被 _CLS2OP 绑住、`config.num_attention_heads`/`True` 不是算子别名 → 都不入列
    assert {a["attr"] for a in got} == {"reshape", "cast", "permute", "topk"}


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


def test_csaindexer_no_longer_reports_ok_zero_nodes(mf_pkg):
    """评估文档 §3.1「最危险的一条」的验收：fused `CSAIndexer` 现在**抛**，不再假成功。"""
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import PYNATIVE_SPEC_FILES, resolve_layer_spec

    top = resolve_layer_spec(mf_pkg, _SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    csa = top.submodules["self_attention"].submodules["core_attention"]
    rel = "pynative/transformers/experimental_attention_variant/indexer.py"
    with pytest.raises(ExtractionDroppedError) as ei:
        extract_cell(mf_pkg, rel, "CSAIndexer", csa.submodules["indexer"], _CELL_FLAGS,
                     present_params={"rotary_pos_emb"}, recurse=True, subcell_specs={})
    msg = str(ei.value)
    assert "CSAIndexer" in msg
    assert "indexer.py:214" in msg            # 就是那个 `with _no_grad():`
    assert "With" in msg
    assert "0" in msg and "detach" in msg     # 计数 + `_no_grad` 整块 detach 提示


def test_working_dsv3_mla_path_also_surfaces_its_drops():
    """反向验收：诊断在**抽得通**的 DSv3 MLA 路径上也真的记到了东西（不是只对失败路径生效）。

    实测两条 `dropped_assigns`（`ori_dtype = x.dtype` 的 Attribute、`head_dim = query.shape[-1]`
    的 Subscript）+ 由此派生的 `unresolved_operands`（下游 `Cast` 的 dtype 操作数）。二者都是
    **标量/dtype、不是张量** → 对字节无影响，但它们此前完全不可见。此处只断言机制在场，不钉行号。
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
    assert len(dag.nodes) > 0                       # 这条路径本来就通（不受新门影响）
    s = diagnostics_summary(dag)
    assert s["dropped_assigns"] >= 2 and s["unresolved_operands"] >= 1
    for it in dag.diagnostics["dropped_assigns"]:
        assert it["src"].endswith(tuple(f":{n}" for n in range(1, 1000))) or ":" in it["src"]
        assert it["rhs"] and it["code"] and it["targets"]
    # 消费方门在真路径上确实会拦（这是路线 B 的适配器要用的那道门）
    with pytest.raises(ExtractionDroppedError):
        assert_extraction_clean(dag)
