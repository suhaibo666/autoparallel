# -*- coding: utf-8 -*-
"""**覆盖度收口**（2026-07-28）：`shape_infer` 的三条新通路 + walker 的权重归属修复。

上游判决单（`docs/to_resolved_adapter_2026-07-25.md` §4.1）定位到抽取图的两类**结构性盲区**：

  (a) **权重派生节点**：`ins` 为空（全部操作数是 `Parameter`，按契约 W2/W4 路由去
      `param_operands`）→ 无从推形状，**尽管 `init_dims.param_shapes` 已经把形状读出来了**；
  (b) **opaque 产出**（`Kernel` / `FusedFunction`）没有 shape 规则，**尽管源侧 docstring
      逐字给了**（`hyper_connection.py:396-399`）。

再加一条纯算子定义的缺口：

  (c) `mint.gather(input, dim, index)` 的产出形 **= index 的形状**（算子定义，不是猜）——
      `vocab_embedding.py:85` 整个 embedding 段卡在这一条上。

本文件的纪律（每条都是"少了它就会静默编数"的那种）：
  * 权重形状**解不出**就照旧记账 + 保 `?`，绝不用"看起来合理"的数顶；
  * opaque 产出的形状是**调用方声明**（必须带 `file:line` 出处），逐条进 `declared` 台账，
    **永不**与"推断出来的"混为一谈；未声明的照旧 unresolved；
  * `mint.gather` 之外的 advanced-index 不许套用 gather 语义（两者产出形不同）。
"""
from __future__ import annotations

import hashlib
import os

import pytest

from cost_eval.opdag import shape_infer as SI


def _dag(*nodes, **kw):
    from cost_eval.opdag.schema import OpDAG
    return OpDAG(cell="T", nodes=list(nodes), **kw)


def _node(nid, op, src, ins=(), out="", **attrs):
    from cost_eval.opdag.schema import OpNode
    return OpNode(id=nid, op=op, src=src, ins=list(ins), out=out, attrs=dict(attrs))


# ═══════════════════════════════════════════════════════════════════════════
# (a) 权重派生节点：形状从 `param_shapes` 喂回来
# ═══════════════════════════════════════════════════════════════════════════

def test_weight_derived_concat_gets_its_shape_from_param_shapes():
    """`alpha = concat((alpha_pre, alpha_post, alpha_res), -1)`（`hyper_connection.py:408`）。

    三个操作数都是 `Parameter` → `ins` 为空（契约 W2/W4）。它们的形状 `__init__` 里有
    （`init_dims.param_shapes`），喂回来即可 —— **这是源读，不是猜**。
    """
    n = _node(1, "View", "hyper_connection.py:408", [], "alpha:?:bf16",
              view="concat", prim="mint.concat", concat_axis=-1,
              param_operands=["alpha_pre", "alpha_post", "alpha_res"])
    SI.infer_shapes(_dag(n), {}, param_shapes={
        ("hyper_connection.py", "alpha_pre"): "1",
        ("hyper_connection.py", "alpha_post"): "1",
        ("hyper_connection.py", "alpha_res"): "1"})
    sym = n.out.split(":")[1]
    # `sym_shape.add` 保持和式原子(不折叠常数),故串是 `((1+1)+1)` —— 值必须是 3。
    assert eval(sym.replace("·", "*")) == 3, sym       # noqa: S307 —— 纯常数表达式


def test_weight_derived_cast_passthrough_gets_the_weight_shape():
    """`attn_sink_f32 = ops.cast(attn_sink, float32)`（`csa.py:687`）—— 一份**独立的 fp32 复本**。

    台账第 3 项「漏 `sinks`」卡的就是这个节点：`ins` 空 → 形状解不出 → 整个节点被跳过。
    """
    n = _node(1, "Cast", "csa.py:687", [], "sinks:?:fp32",
              prim="ops.cast", param_operands=["attn_sink"])
    SI.infer_shapes(_dag(n), {}, param_shapes={("csa.py", "attn_sink"): "n_heads"})
    assert n.out.split(":")[1] == "n_heads", n.out


def test_missing_weight_shape_is_recorded_never_invented():
    """权重形状**不在** `param_shapes` 里 → 保 `?` + 记一条带 `file:line` 的账。"""
    n = _node(1, "Cast", "csa.py:687", [], "sinks:?:fp32",
              prim="ops.cast", param_operands=["mystery"])
    rep = []
    SI.infer_shapes(_dag(n), {}, param_shapes={}, report=rep)
    assert n.out.split(":")[1] == "?"
    assert rep and rep[0]["src"] == "csa.py:687"
    assert rep[0]["reason"] == "weight_shape_unresolved", rep


def test_param_shapes_never_override_a_node_that_has_real_ins():
    """只有 `ins` **为空**的节点才走权重通路 —— 有真激活操作数时一律照旧按 `ins` 推。

    否则 `self.embedding(weight, 0, input_)` 这类「权重 + 激活」混合节点会被权重形状顶掉。
    """
    n = _node(1, "Cast", "csa.py:687", ["x:S·B·H:bf16"], "y:?:bf16",
              prim="ops.cast", param_operands=["attn_sink"])
    SI.infer_shapes(_dag(n), {"x": "S·B·H"},
                    param_shapes={("csa.py", "attn_sink"): "n_heads"})
    assert n.out.split(":")[1] == "S·B·H", n.out


# ═══════════════════════════════════════════════════════════════════════════
# (b) opaque 产出：**调用方声明**（带出处），与推断分开记
# ═══════════════════════════════════════════════════════════════════════════

_MHC_DECL = {
    "npu_mhc_pre_sinkhorn": {
        "outs": ("S·B·H", "~S·B·hc_mult", "~S·B·hc_mult·hc_mult"),
        "src": "hyper_connection.py:396-399,421-423",
        "why": "docstring 逐字 + reshape 恒不改元素数",
    },
}


def test_declared_kernel_output_shape_is_applied_and_labelled_as_declared():
    n = _node(1, "Kernel", "hyper_connection.py:413", ["x:S·B·hc_mult·H:bf16"], "h_in:?:bf16",
              kernel="npu_mhc_pre_sinkhorn",
              outs=["h_in:?:bf16", "h_post:?:bf16", "h_res_flat:?:bf16"])
    decl = []
    SI.infer_shapes(_dag(n), {"x": "S·B·hc_mult·H"},
                    declared_outs=_MHC_DECL, declared=decl)
    assert n.out.split(":")[1] == "S·B·H", n.out
    assert n.attrs["outs"][1].split(":")[1] == "~S·B·hc_mult"
    assert n.attrs["outs"][2].split(":")[1] == "~S·B·hc_mult·hc_mult"
    # 每一条都带出处，且**标成声明**（不与推断混为一谈）
    names = {d[0] for d in decl}
    assert names == {"h_in", "h_post", "h_res_flat"}, decl
    assert all(d[2] == "hyper_connection.py:396-399,421-423" for d in decl), decl


def test_undeclared_opaque_output_stays_unresolved():
    """没声明就**不给数** —— 既有 fail-soft 行为逐字不变。"""
    n = _node(1, "Kernel", "x.py:1", ["x:S·B·H:bf16"], "y:?:bf16",
              kernel="npu_something_else")
    rep = []
    SI.infer_shapes(_dag(n), {"x": "S·B·H"}, declared_outs=_MHC_DECL, report=rep)
    assert n.out.split(":")[1] == "?"
    assert rep[0]["reason"] == "constant_shape_unknown"


def test_declared_output_can_mirror_an_input_shape():
    """`_LogSoftmax.apply(logits)`（`loss.py:197`）的产出与 `logits` **同形**
    （`loss.py:136-143` 逐字：`sub(log_sum, shifted)`，`shifted = logits - max` 广播）。"""
    decl_t = {"_LogSoftmax": {"outs": ("=in0",), "src": "loss.py:136-143",
                              "why": "forward 返回 sub(log_sum, shifted)，与 logits 同形"}}
    n = _node(1, "FusedFunction", "loss.py:197", ["logits:B·S·vocab:bf16"], "ls:?:bf16",
              function="_LogSoftmax")
    decl = []
    SI.infer_shapes(_dag(n), {"logits": "B·S·vocab"}, declared_outs=decl_t, declared=decl)
    assert n.out.split(":")[1] == "B·S·vocab", n.out
    assert decl and decl[0][2] == "loss.py:136-143"


def test_declared_mirror_of_an_unresolved_input_stays_unresolved():
    """镜像源本身没解出 → 照旧 `?`（声明不是凭空造数的许可证）。"""
    decl_t = {"_LogSoftmax": {"outs": ("=in0",), "src": "loss.py:136-143", "why": "同形"}}
    n = _node(1, "FusedFunction", "loss.py:197", ["logits:?:bf16"], "ls:?:bf16",
              function="_LogSoftmax")
    decl = []
    SI.infer_shapes(_dag(n), {}, declared_outs=decl_t, declared=decl)
    assert n.out.split(":")[1] == "?"
    assert not decl


# ═══════════════════════════════════════════════════════════════════════════
# (c) `mint.gather`：产出形 = index 形（算子定义）
# ═══════════════════════════════════════════════════════════════════════════

def test_mint_gather_output_takes_the_index_shape():
    """`self.embedding(weight, 0, input_)`（`vocab_embedding.py:85`）= `mint.gather`。

    torch/mindspore `gather(input, dim, index)` 的产出与 **index 同形**。
    权重 `weight` 走 `param_operands`（不进 `ins`）→ 存活的那个 `ins` 的张量位序是 1 = index。
    """
    n = _node(1, "IndexSelect", "vocab_embedding.py:85", ["idx:B·S·H:int32"], "output:?:bf16",
              prim="mint.gather", ins_slots=[1], param_operands=["weight"])
    SI.infer_shapes(_dag(n), {"idx": "B·S·H"})
    assert n.out.split(":")[1] == "B·S·H", n.out


def test_advanced_indexing_is_not_given_gather_semantics():
    """`kv_flat[flat_indices]`（`csa.py:485`）的产出形 = index 形 **+ input 尾轴** —— 不同语义。
    没有 `prim=mint.gather` 就不许套 gather 规则（宁 `?` 勿错）。"""
    n = _node(1, "IndexSelect", "csa.py:485", ["idx:B·S:int32"], "sel:?:bf16",
              prim="<subscript>", ins_slots=[1])
    rep = []
    SI.infer_shapes(_dag(n), {"idx": "B·S"}, report=rep)
    assert n.out.split(":")[1] == "?"
    assert rep[0]["reason"] == "constant_shape_unknown"


def test_mint_gather_without_an_identifiable_index_stays_unresolved():
    """认不出哪个操作数是 index（没有 `ins_slots`、也不是两操作数形态）→ 保 `?`。"""
    n = _node(1, "IndexSelect", "x.py:1", [], "o:?:bf16", prim="mint.gather")
    rep = []
    SI.infer_shapes(_dag(n), {}, report=rep)
    assert n.out.split(":")[1] == "?"
    assert rep


# ═══════════════════════════════════════════════════════════════════════════
# (d) walker：权重归属按**发射作用域**，不按行号相等
# ═══════════════════════════════════════════════════════════════════════════

_MD5 = {
    "pynative/transformers/hyper_connection.py": "14cf0de51c41df31f5e4439549934660",
}


def _authoritative_pkg():
    for root in (os.environ.get("MINDFORMERS_ROOT"),
                 r"E:\97-codes\torch_parallel\mf-src-167\mindformers",
                 r"E:\97-codes\torch_parallel\mf-src-167"):
        if not root:
            continue
        for pkg in (root, os.path.join(root, "mindformers")):
            ok = True
            for rel, want in _MD5.items():
                p = os.path.join(pkg, *rel.split("/"))
                if not os.path.isfile(p):
                    ok = False
                    break
                with open(p, "rb") as fh:
                    if hashlib.md5(fh.read()).hexdigest() != want:
                        ok = False
                        break
            if ok:
                return pkg
    return None


@pytest.fixture(scope="module")
def mf_pkg():
    pkg = _authoritative_pkg()
    if pkg is None:
        pytest.skip("权威 mindformers 快照缺失/md5 不匹配（绝不对着另一个 commit 断言）")
    os.environ["MINDFORMERS_ROOT"] = pkg
    return pkg


def test_multiline_call_attributes_its_params_to_the_consuming_node(mf_pkg):
    """`hyper_connection.py:408-411` 的 concat 跨四行，三个 `Parameter` 记在 **:409**。

    此前 `_emit` 按「param 记录的行号 == 节点行号」匹配 → 该节点 `attrs` 里**没有**
    `param_operands`，于是既拿不到形状、也拿不到那三个权重的字节。改成**按发射作用域**
    归属（本节点处理实参期间新记的权重就是本节点的）后必须能对上。
    """
    from cost_eval.opdag import to_resolved as TR
    root = TR.mf_root()
    site = TR._Site(fused=True, ratio=4, moe=True, dims_key=())
    from cost_eval.model_spec import DimTable
    dims = DimTable(H=4096, F=4096, n_heads=64, n_kv=1, head_dim=128, S=4096, B=1,
                    vocab=129280, n_layers=8, q_lora_rank=1024, kv_lora_rank=512,
                    qk_rope_head_dim=64, qk_nope_head_dim=0, v_head_dim=512,
                    dsa_indexer_n_heads=64, dsa_indexer_head_dim=128,
                    dsa_indexer_topk=512, o_groups=8, o_lora_rank=1024,
                    csa_window_size=128, num_residual_streams=4)
    dag = TR._decoder_dag(root, site, dims)
    hits = [n for n in dag.nodes if n.src == "hyper_connection.py:408"]
    assert hits, "抽不到 hyper_connection.py:408 的 concat 节点"
    for n in hits:
        assert n.attrs.get("param_operands") == ["alpha_pre", "alpha_post", "alpha_res"], \
            n.attrs
