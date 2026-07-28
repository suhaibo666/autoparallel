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


# ═══════════════════════════════════════════════════════════════════════════
# (e) 归约 / 转置 / MatMul 末轴 / advanced-index / 整除原子（都是**算子定义**）
# ═══════════════════════════════════════════════════════════════════════════

_SITE = dict(H=4096, F=4096, n_heads=64, n_kv=1, head_dim=128, S=4096, B=1,
             vocab=129280, n_layers=8, q_lora_rank=1024, kv_lora_rank=512,
             qk_rope_head_dim=64, qk_nope_head_dim=0, v_head_dim=512,
             dsa_indexer_n_heads=64, dsa_indexer_head_dim=128,
             dsa_indexer_topk=512, o_groups=8, o_lora_rank=1024,
             csa_window_size=128, num_residual_streams=4)


def test_reduce_uses_the_axis_the_walker_already_recorded():
    """`mint.mean(q*q, dim=-1, keepdim=True)`（`deepseek_v4_hybrid_attention.py:245`）：
    轴 walker 早就记进 `attrs`（G4），`shape_infer` 此前一律记 `reduce_axis_unknown` 不用它。"""
    n = _node(1, "Elementwise", "deepseek_v4_hybrid_attention.py:245",
              ["q:S·B·n_heads·v_head_dim:bf16"], "m:?:bf16",
              reduce=True, linear=True, prim="mint.mean", reduce_dim=-1, keepdim=True)
    SI.infer_shapes(_dag(n), {"q": "S·B·n_heads·v_head_dim"})
    assert n.out.split(":")[1] == "S·B·n_heads·1", n.out


def test_reduce_without_keepdim_drops_the_axis():
    n = _node(1, "Elementwise", "x.py:1", ["a:S·B·H:bf16"], "m:?:bf16",
              reduce=True, prim="mint.sum", reduce_dim=1)
    SI.infer_shapes(_dag(n), {"a": "S·B·H"})
    assert n.out.split(":")[1] == "S·H", n.out


def test_reduce_all_is_a_source_fact_and_gives_a_scalar():
    """源侧**根本没传** `dim` ⇒ 全轴归约 ⇒ 标量。与「抠不出轴」必须分开记
    （真源 `loss.py:344/346` 的 `self.sum(x)`）。"""
    n = _node(1, "Elementwise", "loss.py:344", ["a:B·S:fp32"], "num:?:fp32",
              reduce=True, prim="mint.sum", reduce_all=True)
    SI.infer_shapes(_dag(n), {"a": "B·S"})
    assert n.out.split(":")[1] == "1", n.out


def test_reduce_without_any_axis_info_still_records_and_keeps_qmark():
    """既没记轴、也不是全轴归约 → 照旧 `?` + 记账（**绝不** passthrough 顶替）。"""
    n = _node(1, "Elementwise", "compressor.py:216", ["kv:S·B·H:fp32"], "p:?:fp32",
              reduce=True, prim="<tensor>.sum")
    rep = []
    SI.infer_shapes(_dag(n), {"kv": "S·B·H"}, report=rep)
    assert n.out.split(":")[1] == "?"
    assert [r["reason"] for r in rep] == ["reduce_axis_unknown"]


def test_cumsum_is_not_a_reduction():
    """`mint.cumsum` 在 `_REDUCE_LIN` 里，但它**不归约**（前缀和，输出同形）。"""
    n = _node(1, "Elementwise", "x.py:1", ["a:S·B:fp32"], "c:?:fp32",
              reduce=True, prim="mint.cumsum", reduce_dim=0)
    SI.infer_shapes(_dag(n), {"a": "S·B"})
    assert n.out.split(":")[1] == "S·B", n.out


def test_two_axis_transpose_is_exact_when_the_axes_are_recorded():
    """`mint.transpose(weight, 1, 0)`（`linear.py:132`）—— 两个轴是源里逐字写着的常数，
    可以精确互换，不必退 `numel_only`。"""
    n = _node(1, "View", "linear.py:132", [], "weight:?:bf16",
              view="transpose", prim="mint.transpose", swap_axes=[1, 0],
              param_operands=["weight"])
    SI.infer_shapes(_dag(n), {}, param_shapes={("linear.py", "weight"): "vocab·H"})
    assert n.out.split(":")[1] == "H·vocab", n.out


def test_matmul_takes_the_weight_last_axis_when_out_dim_is_absent():
    """`output = matmul(input_, weight)`（`linear.py:135`）：`Linear` 作为**顶层** Cell 抽取时
    没有 `build_module(..., output_size=…)` 那个调用点 ⇒ `out_dim` 缺席。算子定义补上。"""
    t = _node(1, "View", "linear.py:132", [], "weight:?:bf16",
              view="transpose", prim="mint.transpose", swap_axes=[1, 0],
              param_operands=["weight"])
    mm = _node(2, "MatMul", "linear.py:135", ["input_:S·B·H:bf16"], "output:?:bf16",
               prim="mint.matmul", param_operands=["weight"])
    SI.infer_shapes(_dag(t, mm), {"input_": "S·B·H"},
                    param_shapes={("linear.py", "weight"): "vocab·H"})
    assert mm.out.split(":")[1] == "S·B·vocab", mm.out


def test_advanced_index_output_is_index_shape_plus_trailing_axes():
    """`kv_flat[flat_indices]`（`csa.py:485`）：`[b·sk, d][idx] → idx.shape ++ (d,)`。"""
    n = _node(1, "IndexSelect", "csa.py:485",
              ["kv_flat:(B·S)·v_head_dim:bf16", "idx:B·S·index_topk:int64"],
              "g:?:bf16", advanced_index=True)
    SI.infer_shapes(_dag(n), {"kv_flat": "(B·S)·v_head_dim", "idx": "B·S·index_topk"})
    assert n.out.split(":")[1] == "B·S·index_topk·v_head_dim", n.out


def test_advanced_index_refuses_a_boolean_mask():
    """`x[mask]` 的产出长度取决于**值**而不是形状 —— 套整数索引规则会算错，必须保 `?`。"""
    n = _node(1, "IndexSelect", "x.py:1",
              ["a:(B·S)·H:bf16", "m:B·S:bool"], "g:?:bf16", advanced_index=True)
    rep = []
    SI.infer_shapes(_dag(n), {"a": "(B·S)·H", "m": "B·S"}, report=rep)
    assert n.out.split(":")[1] == "?"
    assert rep and rep[0]["reason"] == "constant_shape_unknown"


def test_reshape_minus_one_forms_a_divide_atom_instead_of_giving_up():
    """`reshape(kv, (n_compressed, ratio, b, -1))`（`compressor.py:203`）：已知积里是 `S//4`、
    总积里是 `S`，符号约不干净 —— 但 `-1` 位按定义就是两者之商，保成表达式即可。"""
    from cost_eval.model_spec import DimTable
    from cost_eval.opdag.consumer import resolve_shape_elems
    n = _node(1, "View", "compressor.py:203", ["kv:S·B·(2·v_head_dim):bf16"], "kv:?:bf16",
              view="reshape", prim="mint.reshape",
              reshape_dims=["n_compressed", "ratio", "b", "-1"])
    dag = _dag(n, const_scalars={"ratio": 4},
               scalar_exprs={"n_compressed": "cutoff // ratio",
                             "cutoff": "(sq // ratio) * ratio"},
               scalar_binds=[{"names": ["sq", "b", "_"], "src": "kv", "frame": ""}])
    SI.infer_shapes(dag, {"kv": "S·B·(2·v_head_dim)"})
    sym = n.out.split(":")[1]
    assert not sym.startswith("~"), f"应当解出**轴结构**，而不是退 numel_only：{sym}"
    # 元素数**必须守恒**：reshape 恒不改元素数。
    assert resolve_shape_elems(sym, DimTable(**_SITE)) == 4096 * 1 * 2 * 512


def test_divide_atom_refuses_when_it_is_not_exact():
    """整除原子除不尽 → `None`（**不取整**）：reshape 的 `-1` 按定义整除，除不尽说明上游解错了。"""
    from cost_eval.model_spec import DimTable
    from cost_eval.opdag.consumer import resolve_shape_elems
    dims = DimTable(**dict(_SITE, S=4097))
    assert resolve_shape_elems("(S)//(H)", dims) is None       # 4097 % 4096 != 0


def test_paired_paren_stripping_does_not_break_a_product_term():
    """`consumer._sym_value` 里若用 `str.strip` 剥括号，`2·((a)//(b))·c` 的**配对**尾括号会被
    剥掉、整串弄坏。回归钉（2026-07-28 实测：整除原子进乘积项后就撞上这条）。"""
    from cost_eval.model_spec import DimTable
    from cost_eval.opdag.consumer import resolve_shape_elems
    dims = DimTable(**_SITE)
    # 2·((S·H)//(H))·B = 2·4096·1 = 8192
    assert resolve_shape_elems("(2·((S·H)//(H))·B)", dims) == 8192


def test_stale_seed_is_overridden_by_the_dataflow_edge():
    """内联把子 Cell 形参名永久映射成调用方 ref（`vocab_embedding.py:83-85` 三步的 `ins` 都写作
    调用方的 `input_ids`）→ 按名查会拿到**入口种子**，而真正的上游是那条边。
    源 docstring `vocab_embedding.py:76` 逐字：`output: (B, S, H)`。"""
    n1 = _node(1, "View", "vocab_embedding.py:83", ["input_ids:?:int32"], "input_:?:int32",
               view="reshape", prim="mint.reshape", reshape_dims=["-1", "1"])
    n2 = _node(2, "View", "vocab_embedding.py:84", ["input_ids:?:int32"], "input_:?:int32",
               view="tile", prim="mint.tile", tile_mult=["1", "self.embedding_dim"])
    n3 = _node(3, "IndexSelect", "vocab_embedding.py:85", ["input_ids:?:int32"],
               "output:?:bf16", prim="mint.gather", ins_slots=[1], param_operands=["weight"])
    dag = _dag(n1, n2, n3, edges=[[1, 2], [2, 3]], dims_ctx={"embedding_dim": "H"})
    SI.infer_shapes(dag, {"input_ids": "B·S"}, bridge_by_edge=True)
    assert n3.out.split(":")[1] == "(B·S)·H", n3.out


def test_stale_seed_override_is_off_by_default():
    """缺省关（既有调用方逐字不变）——`timesim/producer.py` 的通信注入按「几个输入已解出」判
    S 分歧，多解出一个就多注入一条 AG，那是另一个子系统的口径。"""
    n1 = _node(1, "View", "vocab_embedding.py:83", ["input_ids:?:int32"], "input_:?:int32",
               view="reshape", prim="mint.reshape", reshape_dims=["-1", "1"])
    n3 = _node(3, "IndexSelect", "vocab_embedding.py:85", ["input_ids:?:int32"],
               "output:?:bf16", prim="mint.gather", ins_slots=[1])
    dag = _dag(n1, n3, edges=[[1, 3]])
    SI.infer_shapes(dag, {"input_ids": "B·S"})
    assert n3.out.split(":")[1] == "B·S", n3.out       # 拿的是种子，不是边
