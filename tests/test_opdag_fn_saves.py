"""Task 1 — `cost_eval/opdag/fn_saves.py`：自定义 autograd `_Function` 的**源真值**抽取器。

背景（`docs/opdag_coverage_assessment_2026-07-25.md` §4/§5，路线 A）：`_Function` 子类没有
`construct`，`construct_walker` 走不进去；**但它们的 saved 集在源里是逐字写着的**。本抽取器只做
三件纯 AST 的事，不建图、不猜：

  1. `ctx.save_for_backward(...)` 的**实参名单**（真源两种形态：裸实参、以及
     `*[tensor for tensor in (<元组>) if tensor is not None]` 过滤式）；
  2. 每一个裸 `ctx.<attr> = <rhs>` 赋值 —— 这类**绕过** MindSpore 的 `saved_tensors_hooks`
     （真源实例：`indexer.py:276` `_IndexerLossAutoScaler`、`dsa_indexer.py:54-56`
     `_DSAIndexerFunction` 的 `ctx.q/ctx.k/ctx.weights`）；
  3. 每一个 `ops.stop_gradient(...)` / `.detach()` 站点与 `with _no_grad():` 区域，带被赋名。

本文件两段：**合成源单测**（不依赖 mindformers 快照，永远跑）+ **快照源真值断言**
（md5 门控；非权威树一律 skip，绝不对着另一个 commit 断言）。
"""
import hashlib
import os

import pytest

from cost_eval.opdag.fn_saves import (
    DETACH_CALLS,
    FUNCTION_BASES,
    UnresolvedSaveForm,
    scan_source,
    scan_tree,
)

# ── 合成源：逐形态复刻真源出现过的每一种写法（只 parse，不 exec）───────────────
SYNTH = '''\
from mindspore import ops, mint, _no_grad


class Plain(_Function):
    """裸实参形态（真源 mc2.py:83 / :129 同形）。"""

    @staticmethod
    def forward(ctx, x, w):
        y = mm(x, w)
        ctx.save_for_backward(x, w)
        ctx.alpha = 3
        return y

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        return g, g


class Filtered(_Function):
    """过滤式（真源 csa.py:113 / :224 同形）。"""

    @staticmethod
    def forward(ctx, query, ori_kv, cmp_kv, sinks):
        out = kern(query)
        ctx.has_cmp_kv = cmp_kv is not None
        ctx.save_for_backward(*[tensor for tensor in (
            query,
            ori_kv,
            cmp_kv,
            sinks,
            out,
        ) if tensor is not None])
        ctx.softmax_scale = 1.0
        return out

    @staticmethod
    def backward(ctx, g):
        saved_tensors = iter(ctx.saved_tensors)
        query = next(saved_tensors)
        ori_kv = next(saved_tensors)
        cmp_kv = next(saved_tensors) if ctx.has_cmp_kv else None
        sinks = next(saved_tensors)
        out = next(saved_tensors)
        return g


class Scaler(_Function):
    """裸 ctx 属性存张量（真源 indexer.py:276 同形）——绕过 saved_tensors_hooks。"""

    @staticmethod
    def forward(ctx, output, indexer_loss):
        ctx.indexer_loss = indexer_loss
        ctx.coeff = coeff_arg
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indexer_loss = ctx.indexer_loss
        indexer_loss = indexer_loss.to_local() if isinstance(indexer_loss, DTensor) else indexer_loss
        scaled_grad = mint.ones_like(indexer_loss) * ctx.coeff
        return grad_output, scaled_grad


class Weird(_Function):
    """未知 save 形态 —— 必须落 unresolved，不得静默丢。"""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(*pack_them(x))
        return x


class NotAFunction(nn.Cell):
    def construct(self, x):
        return x


def helper(x, qr, query, halo_x):
    x_detach = ops.stop_gradient(x)
    qr_detach = ops.stop_gradient(qr)
    halo_detach = ops.stop_gradient(halo_x) if halo_x is not None else None
    z = loss_fn(index_scores, ops.stop_gradient(query), mask=None)
    m = x.detach()
    with _no_grad():
        a = f(x_detach)
        b, c = g(a)
    return z, m, a, b, c
'''


@pytest.fixture(scope="module")
def synth():
    return scan_source("synth.py", SYNTH)


def _synth_line(text):
    """SYNTH 里某行的 1-based 行号（不硬编码，改合成源不必改断言）。"""
    return SYNTH.splitlines().index(text) + 1


# ---------------------------------------------------------------------------
# save_for_backward 名单
# ---------------------------------------------------------------------------

def test_plain_args_form(synth):
    rec = synth.by_class("Plain")
    assert rec.saved is not None
    assert rec.saved.names == ("x", "w")
    assert rec.saved.form == "plain_args"
    assert rec.saved.src == f"synth.py:{_synth_line('        ctx.save_for_backward(x, w)')}"


def test_listcomp_filter_none_form(synth):
    rec = synth.by_class("Filtered")
    assert rec.saved.names == ("query", "ori_kv", "cmp_kv", "sinks", "out")
    assert rec.saved.form == "starred_comprehension"
    assert rec.saved.conditional is True          # `if tensor is not None`


def test_class_without_save_for_backward_has_none(synth):
    assert synth.by_class("Scaler").saved is None


def test_unknown_save_form_is_recorded_not_dropped(synth):
    assert any(u.cls == "Weird" for u in synth.unresolved)
    weird = [u for u in synth.unresolved if u.cls == "Weird"][0]
    assert isinstance(weird, UnresolvedSaveForm)
    assert "pack_them" in weird.expr
    # 未知形态的类不得伪装成「无 saved 集」
    assert synth.by_class("Weird").saved is None
    assert synth.by_class("Weird").has_unresolved_save is True


def test_strict_mode_raises_on_unknown_save_form():
    with pytest.raises(ValueError, match="save_for_backward"):
        scan_source("synth.py", SYNTH, strict=True)


def test_non_function_base_is_ignored(synth):
    assert synth.by_class("NotAFunction") is None
    assert "_Function" in FUNCTION_BASES


# ---------------------------------------------------------------------------
# 裸 ctx.<attr> 赋值
# ---------------------------------------------------------------------------

def test_ctx_attr_kinds_and_tensor_candidacy(synth):
    by_attr = {c.attr: c for c in synth.by_class("Filtered").ctx_attrs}
    assert set(by_attr) == {"has_cmp_kv", "softmax_scale"}
    assert by_attr["has_cmp_kv"].kind == "predicate"
    assert by_attr["has_cmp_kv"].tensor_candidate is False
    assert by_attr["softmax_scale"].kind == "const"
    assert by_attr["softmax_scale"].tensor_candidate is False

    scaler = {c.attr: c for c in synth.by_class("Scaler").ctx_attrs}
    assert set(scaler) == {"indexer_loss", "coeff"}
    assert scaler["indexer_loss"].kind == "name"
    assert scaler["indexer_loss"].is_forward_param is True

    plain = {c.attr: c for c in synth.by_class("Plain").ctx_attrs}
    assert plain["alpha"].kind == "const" and plain["alpha"].tensor_candidate is False


def test_tensor_candidate_is_syntactic_upper_bound(synth):
    """上界：kind 非 predicate/const 即入（`coeff` 也在，虽然它只参与算术）。"""
    got = {(c.cls, c.attr) for c in synth.tensor_candidate_ctx_attrs()}
    assert got == {("Scaler", "indexer_loss"), ("Scaler", "coeff")}


def test_confirmed_tensor_ctx_attrs_is_source_evidence_lower_bound(synth):
    """下界：只有 backward 里被 `mint.*`/`isinstance(...,DTensor)`/张量方法消费的才算实。"""
    got = {(c.cls, c.attr) for c in synth.confirmed_tensor_ctx_attrs()}
    assert got == {("Scaler", "indexer_loss")}
    rec = {c.attr: c for c in synth.by_class("Scaler").ctx_attrs}
    assert rec["indexer_loss"].bwd_tensor_evidence is True
    assert rec["coeff"].bwd_tensor_evidence is False      # 纯算术不算证据
    assert any("ones_like" in u for u in rec["indexer_loss"].bwd_uses)


# ---------------------------------------------------------------------------
# backward 侧消费序（与 forward 名单自校验）
# ---------------------------------------------------------------------------

def test_backward_read_order_matches_forward_names(synth):
    rec = synth.by_class("Filtered")
    assert rec.backward_read_order == ("query", "ori_kv", "cmp_kv", "sinks", "out")
    assert rec.saved.names == rec.backward_read_order       # 同序自校验
    assert rec.backward_conditional == ("cmp_kv",)


def test_backward_tuple_unpack_form(synth):
    assert synth.by_class("Plain").backward_read_order == ("x", "w")


# ---------------------------------------------------------------------------
# detach 站点 / _no_grad 区域
# ---------------------------------------------------------------------------

def test_detach_sites_cover_assign_inline_and_conditional(synth):
    sites = {(s.lineno, s.assigned, s.arg, s.form) for s in synth.detach_sites}
    assert ("x_detach", "x", "assign") in {(s.assigned, s.arg, s.form)
                                          for s in synth.detach_sites}
    forms = {s.form for s in synth.detach_sites}
    assert {"assign", "inline_arg", "method"} <= forms
    inline = [s for s in synth.detach_sites if s.form == "inline_arg"]
    assert [s.arg for s in inline] == ["query"]
    assert inline[0].assigned is None
    assert inline[0].enclosing_call == "loss_fn"
    # 条件式 detach（真源 csa_context_parallel.py:604 同形）仍要拿到被赋名
    cond = [s for s in synth.detach_sites if s.assigned == "halo_detach"]
    assert len(cond) == 1 and cond[0].form == "assign_ifexp"
    assert sites  # 非空
    assert "ops.stop_gradient" in DETACH_CALLS


def test_method_detach_site(synth):
    m = [s for s in synth.detach_sites if s.form == "method"]
    assert len(m) == 1 and m[0].assigned == "m" and m[0].fn == ".detach"


def test_no_grad_region_records_bound_names(synth):
    assert len(synth.no_grad_regions) == 1
    reg = synth.no_grad_regions[0]
    assert reg.assigned == ("a", "b", "c")
    assert reg.func == "helper"
    assert reg.lineno < reg.end_lineno


def test_detach_sites_carry_enclosing_function(synth):
    assert {s.func for s in synth.detach_sites} == {"helper"}


# ---------------------------------------------------------------------------
# 快照源真值（md5 门控 —— 缺权威树则 skip；绝不对着另一 commit 断言）
# ---------------------------------------------------------------------------
# `docs/opdag_coverage_assessment_2026-07-25.md` §0 与 `mf-src-167/SNAPSHOT.md` 的钉定值。
SNAPSHOT_MD5 = {
    "csa.py": "81673be3ad3cdd2191e28dd000a13f0e",
    "indexer.py": "8e18fee21c33507dd629c89f401986e6",
}
_VAR_REL = "pynative/transformers/experimental_attention_variant"
_CANDIDATE_ROOTS = (
    os.environ.get("MINDFORMERS_ROOT"),
    r"E:\97-codes\torch_parallel\mf-src-167",
    r"E:\97-codes\torch_parallel\mf-src-167\mindformers",
)


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def authoritative_variant_dir():
    """返回 md5 与钉定快照**逐字节一致**的 `experimental_attention_variant` 目录，否则 None。

    刻意不用 `default_mf_root()`：本机默认解到 `…\\mindformers\\mindformers`（master
    `c1f5e11f5`，`csa.py` md5=`0d3422b4…`）——**内容不同**。审计只允许对着权威快照跑。
    """
    for root in _CANDIDATE_ROOTS:
        if not root:
            continue
        for pkg in (os.path.join(root, "mindformers"), root):
            d = os.path.join(pkg, *_VAR_REL.split("/"))
            if not os.path.isdir(d):
                continue
            try:
                if all(_md5(os.path.join(d, f)) == m for f, m in SNAPSHOT_MD5.items()):
                    return d
            except OSError:
                continue
    return None


@pytest.fixture(scope="module")
def variant_truth():
    d = authoritative_variant_dir()
    if d is None:
        pytest.skip(
            "权威 mindformers 快照缺失（需 csa.py md5="
            f"{SNAPSHOT_MD5['csa.py']} / indexer.py md5={SNAPSHOT_MD5['indexer.py']}）；"
            "设 MINDFORMERS_ROOT=E:\\97-codes\\torch_parallel\\mf-src-167 后重跑")
    return scan_tree(d)


def test_snapshot_function_classes_enumerated(variant_truth):
    """快照 `experimental_attention_variant/` 下全部 `_Function` 子类（5 个）。"""
    assert {r.cls for r in variant_truth.functions} == {
        "FusedSparseFlashMla", "FusedSparseFlashMlaWithIndexerLoss",
        "_IndexerLossAutoScaler", "_DSAIndexerFunction", "_DSAIndexerGradFunction",
    }


def test_snapshot_fused_sparse_flash_mla_saved_8(variant_truth):
    rec = variant_truth.by_class("FusedSparseFlashMla")
    assert rec.src.endswith("csa.py:73")
    assert rec.saved.src.endswith("csa.py:113")
    assert rec.saved.names == (
        "query", "ori_kv", "cmp_kv", "sparse_indices",
        "cmp_residual", "sinks", "output", "softmax_lse",
    )
    assert rec.saved.names == rec.backward_read_order      # csa.py:134-142 同序
    assert rec.backward_conditional == ("cmp_kv", "sparse_indices")


def test_snapshot_fused_with_indexer_loss_saved_11(variant_truth):
    rec = variant_truth.by_class("FusedSparseFlashMlaWithIndexerLoss")
    assert rec.src.endswith("csa.py:178")
    assert rec.saved.src.endswith("csa.py:224")
    assert rec.saved.names == (
        "query", "ori_kv", "cmp_kv", "sparse_indices",
        "query_index", "key_index", "weights",
        "cmp_residual", "sinks", "output", "softmax_lse",
    )
    assert len(rec.saved.names) == 11
    assert rec.saved.names == rec.backward_read_order      # csa.py:251-262 同序


def test_snapshot_bare_ctx_tensor_attrs(variant_truth):
    """绕过 `saved_tensors_hooks` 的裸 ctx **张量**（下界，源证据）—— 快照里 7 个 / 3 个类。"""
    got = {(c.cls, c.attr, c.src) for c in variant_truth.confirmed_tensor_ctx_attrs()}
    assert got == {
        ("_IndexerLossAutoScaler", "indexer_loss", "indexer.py:276"),
        ("_DSAIndexerFunction", "q", "dsa_indexer.py:55"),
        ("_DSAIndexerFunction", "k", "dsa_indexer.py:56"),
        ("_DSAIndexerFunction", "weights", "dsa_indexer.py:57"),
        ("_DSAIndexerGradFunction", "d_query_index", "dsa_indexer_loss.py:70"),
        ("_DSAIndexerGradFunction", "d_key_index", "dsa_indexer_loss.py:71"),
        ("_DSAIndexerGradFunction", "d_weights", "dsa_indexer_loss.py:72"),
    }


def test_snapshot_fused_ctx_scalars_are_not_confirmed_tensors(variant_truth):
    """`csa.py:111-112` 是 bool 谓词、`:123-128` 是 kernel 标量 —— 不得判为实张量。"""
    for cls in ("FusedSparseFlashMla", "FusedSparseFlashMlaWithIndexerLoss"):
        rec = variant_truth.by_class(cls)
        kinds = {c.attr: c.kind for c in rec.ctx_attrs}
        assert kinds["has_cmp_kv"] == "predicate"
        assert kinds["has_sparse_indices"] == "predicate"
        assert {c.attr for c in rec.ctx_attrs
                if c.tensor_candidate and c.bwd_tensor_evidence} == set()


def test_snapshot_stop_gradient_sites(variant_truth):
    """`ops.stop_gradient` 快照内 6 处，全在 csa.py（评估文档 §5）。"""
    sites = [s for s in variant_truth.detach_sites if s.fn == "ops.stop_gradient"]
    got = [(s.src.rsplit(os.sep, 1)[-1], s.arg, s.assigned, s.form) for s in sites]
    assert got == [
        ("csa.py:665", "x", "x_detach", "assign"),
        ("csa.py:666", "qr", "qr_detach", "assign"),
        ("csa.py:764", "x", "x_detach", "assign"),
        ("csa.py:765", "qr", "qr_detach", "assign"),
        ("csa.py:794", "query", None, "inline_arg"),
        ("csa.py:795", "compressed_kv", None, "inline_arg"),
    ]
    # :794/:795 的两条内联 detach 就是 ukl1/ukl2 判决的源依据
    kl = [s for s in sites if s.form == "inline_arg"]
    assert {s.enclosing_call for s in kl} == {"self.unfused_indexer_loss"}
    assert {s.func for s in kl} == {"_construct_naive"}


def test_snapshot_no_grad_region_is_indexer_214(variant_truth):
    """**更正评估文档 §3.1/§6.2 的 `indexer.py:220`**：`with _no_grad():` 实际在 `:214`。"""
    regs = [r for r in variant_truth.no_grad_regions if r.file.endswith("indexer.py")]
    assert len(regs) == 1
    reg = regs[0]
    assert reg.lineno == 214 and reg.end_lineno == 232
    assert reg.func == "construct" and reg.cls == "CSAIndexer"
    assert reg.assigned == (
        "q", "k", "weights", "key_length", "cmp_residual_k",
        "topk_indices", "index_scores",
    )
