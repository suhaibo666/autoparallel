# -*- coding: utf-8 -*-
"""**符号轴结构恢复**（2026-07-29）：让符号 shape 层能**携带并恢复轴结构**，而不只是元素数。

立项依据：`docs/next_fix_adversarial_review_2026-07-28.md` §3.5/§6 —— 四个级联根里**三个**
归约到同一项能力缺口（`compressor.py:216` ×4、`deepseek_v4_hybrid_attention.py:205` ×1、
`csa.py:485`→`:496` ×1），第四个（`compressor.py:233`）是切片边界。该复核用
`scratchpad/adv_probe_cf.py` 做了**反事实实测**：单点修 `csa.py:485` 只把 L1 跳过 42→36、
`activation_saves` 1067.6→1071.6 MiB —— 证明"一条规则"不够，缺的是能力。

本文件的四条纪律（每条都是"少了它就会静默给出一个错的张量尺寸"的那种）：

  * **恢复出来的轴永远不能是猜的**。轴结构不能由「源 + config」证得 → 照旧 `unresolved`
    并带 `file:line`。测试用**否定断言**钉住：缺 `permute_dims` / 非法置换 / 不整除的步长
    / 没有整除事实时，一律拒绝，**不**给一个"看起来合理"的轴。
  * **整除事实必须先核验再使用**。`ratio·(S//ratio) == S` 只在 `ratio | S` 时成立；
    事实由 `DimTable` 的**实际值**核验后才生成，核验不过就**没有**这条事实（于是表达式
    保持保守形态），显式断言不成立时 **fail-loud**。
  * **规范化只做恒等变形**：任何被折叠的表达式，代入 `DimTable` 求值必须**逐字节不变**。
  * **不引入新 OpType**：全部改动落在既有 `OpType` 枚举与既有 View 子类型上。
"""
from __future__ import annotations

import pytest

from cost_eval.opdag import shape_infer as SI
from cost_eval.opdag import sym_shape as SS
from cost_eval.opdag.sym_shape import (
    DivFacts, Factors, NO_DIV_FACTS, assert_divisible, div_facts_from_values,
    normalize_axis, normalize_axes, parse_axis, parse_shape, product_of,
    render_shape, render_term,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 表达式规范化：和/差项的抵消
# ══════════════════════════════════════════════════════════════════════════════

def test_sum_diff_atom_cancels_to_the_bare_symbol():
    """`64+v_head_dim-64` → `v_head_dim`（实测串，见 §攻击点 3.2 的守卫探针输出）。

    真源：`deepseek_v4_hybrid_attention.py:203-205/215`
        pos_dim  = self.config.qk_pos_emb_head_dim        # host 值 64
        nope_dim = self.config.v_head_dim - pos_dim       # 差式原子 `v_head_dim-64`
        ... cat([t_nope, t_pe], dim=-1)                   # 和式 `64+v_head_dim-64`
    concat 把切开的两段**原样拼回**，末轴当然还是 `v_head_dim` —— 这是算子定义下的恒等式，
    不是近似。此前它以原子串形态存活，于是 `csa.py:482` 的两轴 reshape 拿不到干净的末轴。
    """
    f = parse_axis("64+v_head_dim-64")
    assert render_term(normalize_axis(f)) == "v_head_dim"


def test_sum_diff_cancellation_is_order_insensitive():
    assert render_term(normalize_axis(parse_axis("v_head_dim-64+64"))) == "v_head_dim"
    assert render_term(normalize_axis(parse_axis("(v_head_dim-64)+64"))) == "v_head_dim"


def test_normalization_folds_pure_integer_arithmetic():
    assert render_term(normalize_axis(parse_axis("64+64"))) == "128"
    assert render_term(normalize_axis(parse_axis("128-64"))) == "64"


def test_normalization_keeps_a_genuinely_irreducible_sum():
    """约不掉的和式**原样保留**（不许"简化"成任何一项）。"""
    got = render_term(normalize_axis(parse_axis("qk_head_dim+v_head_dim")))
    assert got == "qk_head_dim+v_head_dim"


def test_normalization_never_drops_a_term_it_cannot_cancel():
    """`a+b-c`：三项互不相消 → 三项都必须还在。"""
    got = render_term(normalize_axis(parse_axis("H+v_head_dim-qk_head_dim")))
    assert "H" in got and "v_head_dim" in got and "qk_head_dim" in got


def test_normalization_is_idempotent():
    """规范化两次 == 规范化一次（串**逐字**相同）—— 否则报表/台账会抖。"""
    for s in ("64+v_head_dim-64", "H+v_head_dim-qk_head_dim", "2·(S//4)",
              "(2·B·S·v_head_dim)//(4·B·(S//4))"):
        once = render_term(normalize_axis(parse_axis(s)))
        twice = render_term(normalize_axis(parse_axis(once)))
        assert once == twice, f"{s!r}: {once!r} != {twice!r}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. 整除事实：`ratio·(S//ratio) == S`
# ══════════════════════════════════════════════════════════════════════════════

def test_divisibility_identity_needs_the_fact_and_is_refused_without_it():
    """**没有**已核验的整除事实时，`4·(S//4)` 必须**原样保留** —— 它确实不等于 `S`
    （源侧 `compressor.py:196-199` 在 `cutoff < sq` 时**真的会截断**）。"""
    f = parse_axis("4·(S//4)")
    assert render_term(normalize_axis(f, NO_DIV_FACTS)) == "4·(S//4)"


def test_divisibility_identity_folds_when_the_fact_is_verified():
    facts = DivFacts(frozenset({(4, "S")}))
    assert render_term(normalize_axis(parse_axis("4·(S//4)"), facts)) == "S"


def test_divisibility_identity_unlocks_the_compressor_reduce_axis():
    """`compressor.py:216` 的级联根，逐字：`(2·B·S·v_head_dim)//(4·B·(S//4))` → `2·v_head_dim`。

    这是 `compressor.py:203` `reshape(kv, (n_compressed, ratio, b, -1))` 的 `-1` 位
    （= `numel // (n_compressed·ratio·b)`）。`4 | S` 成立时分母 `4·B·(S//4)` 恒等于 `B·S`，
    于是消元变成**精确**的多重集差，而不是一个解不开的整除原子。
    """
    facts = DivFacts(frozenset({(4, "S")}))
    f = parse_axis("(2·B·S·v_head_dim)//(4·B·(S//4))")
    assert render_term(normalize_axis(f, facts)) == "2·v_head_dim"


def test_div_facts_are_built_only_from_verified_values():
    """事实由**实际 DimTable 值**核验后才生成；除不尽的 pair 进 `rejected`，**不**成为事实。"""
    ok = div_facts_from_values([(4, "S")], {"S": 4096})
    assert ok.has(4, "S") and ok.rejected == ()

    bad = div_facts_from_values([(7, "S")], {"S": 4096})
    assert not bad.has(7, "S")
    assert bad.rejected == ((7, "S", 4096),)

    unknown = div_facts_from_values([(4, "S")], {})
    assert not unknown.has(4, "S") and unknown.rejected == ()


def test_assert_divisible_fails_loud_when_it_does_not_hold():
    assert assert_divisible(4, "S", {"S": 4096}) is True
    with pytest.raises(ValueError) as e:
        assert_divisible(7, "S", {"S": 4096})
    assert "S" in str(e.value) and "7" in str(e.value)


def test_normalization_preserves_the_value_under_a_dimtable():
    """**恒等变形**：折叠前后代入同一份 DimTable，元素数必须逐字节相同。"""
    from cost_eval.opdag.consumer import _axis_value

    class D:
        S, B, H, v_head_dim = 4096, 1, 7168, 512
        qk_rope_head_dim, qk_nope_head_dim = 64, 128
        n_heads = 128

    facts = DivFacts(frozenset({(4, "S")}))
    for s in ("64+v_head_dim-64", "4·(S//4)", "(2·B·S·v_head_dim)//(4·B·(S//4))",
              "qk_head_dim+v_head_dim", "2·(S//4)"):
        raw = parse_axis(s)
        got_raw = _axis_value(raw, D)
        got_norm = _axis_value(normalize_axis(raw, facts), D)
        assert got_raw is not None, f"基线就解不出：{s!r}"
        assert got_raw == got_norm, f"{s!r}: {got_raw} != {got_norm}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. shape_infer：permute / squeeze 的**精确**轴恢复
# ══════════════════════════════════════════════════════════════════════════════

def _dag(*nodes, **kw):
    from cost_eval.opdag.schema import OpDAG
    return OpDAG(cell="T", nodes=list(nodes), **kw)


def _node(nid, op, src, ins=(), out="", **attrs):
    from cost_eval.opdag.schema import OpNode
    return OpNode(id=nid, op=op, src=src, ins=list(ins), out=out, attrs=dict(attrs))


def _run(dag, seeds, **kw):
    rep: list = []
    SI.infer_shapes(dag, seeds, report=rep, **kw)
    return rep


def test_permute_recovers_the_exact_axis_order_from_walker_attrs():
    """`mint.permute(kv_full, (1, 2, 0, 3))` @ `csa.py:472` —— walker **早就记了**
    `permute_dims`（`construct_walker.py:2295-2303`），`shape_infer` 此前没用它，
    于是整条 CSA 主链退成 `~`。置换是算子定义：`out.shape[i] = x.shape[dims[i]]`。"""
    n = _node(1, "View", "csa.py:472", ins=["kv_full:?:bf16"], out="kv_t:?:bf16",
              view="permute", permute_dims=[1, 2, 0, 3])
    rep = _run(_dag(n), {"kv_full": "S·B·1·v_head_dim"})
    assert n.out.split(":")[1] == "B·1·S·v_head_dim"
    assert not n.out.split(":")[1].startswith("~"), "应恢复轴结构，而不是退 numel_only"
    assert rep == []


def test_permute_without_recorded_dims_stays_numel_only():
    """轴序**没记** → 不许猜一个（哪怕"看起来"就是转置）。"""
    n = _node(1, "View", "x.py:1", ins=["a:?:bf16"], out="b:?:bf16", view="permute")
    _run(_dag(n), {"a": "S·B·H"})
    assert n.out.split(":")[1].startswith(SS.NUMEL_ONLY)


def test_permute_with_a_non_permutation_is_refused_loudly():
    """`permute_dims` 不是 `range(rank)` 的一个置换 → 记账 + 退 numel_only，不许硬套。"""
    n = _node(1, "View", "x.py:1", ins=["a:?:bf16"], out="b:?:bf16",
              view="permute", permute_dims=[0, 0, 1])
    rep = _run(_dag(n), {"a": "S·B·H"})
    assert n.out.split(":")[1].startswith(SS.NUMEL_ONLY)
    assert any(r["reason"] == "needs_axis_structure" for r in rep)


def test_permute_of_a_numel_only_input_stays_numel_only():
    n = _node(1, "View", "x.py:1", ins=["a:?:bf16"], out="b:?:bf16",
              view="permute", permute_dims=[1, 0, 2])
    _run(_dag(n), {"a": "~S·B·H"})
    assert n.out.split(":")[1].startswith(SS.NUMEL_ONLY)


def test_transpose_with_a_full_perm_tuple_is_exact():
    """`ops.transpose(x, (0, 2, 1))` 走 `perm`（`construct_walker.py:2255-2257`）。"""
    n = _node(1, "View", "x.py:1", ins=["a:?:bf16"], out="b:?:bf16",
              view="transpose", perm=[0, 2, 1])
    _run(_dag(n), {"a": "S·B·H"})
    assert n.out.split(":")[1] == "S·H·B"


def test_squeeze_drops_only_a_provably_unit_axis():
    """`self.squeeze(out, -2)` @ `compressor.py:244`：该轴**必须**已证为 1 才允许去掉。"""
    n = _node(1, "View", "compressor.py:244", ins=["a:?:bf16"], out="b:?:bf16",
              view="squeeze", squeeze_axis=-2)
    _run(_dag(n), {"a": "S·B·1·v_head_dim"})
    assert n.out.split(":")[1] == "S·B·v_head_dim"


def test_squeeze_refuses_when_the_axis_is_not_provably_one():
    n = _node(1, "View", "x.py:1", ins=["a:?:bf16"], out="b:?:bf16",
              view="squeeze", squeeze_axis=0)
    _run(_dag(n), {"a": "S·B·H"})
    assert n.out.split(":")[1].startswith(SS.NUMEL_ONLY)


def test_chunk_splits_the_recorded_axis_exactly():
    """`self.chunk(tensor, 2, dim=-1)` @ `compressor.py:169`（`_overlap_transform`）。

    份数与轴 walker 都记了（`construct_walker.py:2286-2295` 的 `chunks`/`chunk_dim`），
    `shape_infer._chunk` 此前**只用份数、丢掉轴**，一律压成"单轴 = 总积/k" ⇒ `~`。
    于是 `compressor.py:216 ×4` 那条级联根一直挡着：`_overlap_transform` 的产物没有轴结构，
    `:215` 的 softmax(dim=1) 与 `:216` 的 sum(dim=1) 都按轴走，只能拒绝。
    """
    n = _node(1, "View", "compressor.py:169", ins=["t:?:bf16"], out="a:?:bf16",
              view="chunk", chunks=2, chunk_dim=-1, outs=["a:?:bf16", "b:?:bf16"])
    _run(_dag(n), {"t": "S·B·(2·v_head_dim)"})
    assert n.out.split(":")[1] == "S·B·v_head_dim"


def test_chunk_refuses_the_axis_when_it_is_not_provably_divisible():
    """轴长除不尽份数 ⇒ 各份不等长 ⇒ 不许给一个 floor 值当"每份"。

    这里**元素数**仍可精确二分（总积 `2·S·B·H`），故退回既有的"只知元素数"档；
    轴结构不给（`~`），下游按轴改形的算子照旧拒绝。
    """
    n = _node(1, "View", "x.py:1", ins=["t:?:bf16"], out="a:?:bf16",
              view="chunk", chunks=2, chunk_dim=0, outs=["a:?:bf16", "b:?:bf16"])
    _run(_dag(n), {"t": "S·B·(2·H)"})
    assert n.out.split(":")[1].startswith(SS.NUMEL_ONLY)
    assert "S//2" not in n.out


def test_chunk_gives_nothing_when_even_the_element_count_is_unprovable():
    """连元素数都除不尽 → `?`（既有行为，回归钉）。"""
    n = _node(1, "View", "x.py:1", ins=["t:?:bf16"], out="a:?:bf16",
              view="chunk", chunks=2, chunk_dim=0, outs=["a:?:bf16", "b:?:bf16"])
    rep = _run(_dag(n), {"t": "S·B·H"})
    assert n.out.split(":")[1] == "?"
    assert any(r["reason"] == "chunk_axis_unknown" for r in rep)


def test_chunk_fails_loud_when_the_two_source_facts_disagree():
    """份数有两条源侧事实：字面量 `chunks` 与元组解包元数。**不一致就是解错了**。"""
    n = _node(1, "View", "x.py:1", ins=["t:?:bf16"], out="a:?:bf16",
              view="chunk", chunks=3, chunk_dim=-1,
              outs=["a:?:bf16", "b:?:bf16"])
    rep = _run(_dag(n), {"t": "S·B·(6·H)"})
    assert n.out.split(":")[1] == "?"
    assert any(r["reason"] == "chunk_axis_unknown" for r in rep)


# ══════════════════════════════════════════════════════════════════════════════
# 4. shape_infer：切片边界（`compressor.py:233`）
# ══════════════════════════════════════════════════════════════════════════════

def test_strided_slice_length_is_exact_when_the_step_divides_the_stop():
    """`freqs[:total_seq_len:self.compress_ratio]` @ `compressor.py:233`。

    `total_seq_len = n_compressed * self.compress_ratio`（`compressor.py:230`）⇒ 步长
    **整除**终点 ⇒ 长度 = `total_seq_len // ratio` = `n_compressed`，**精确**，不用取整。
    """
    n = _node(1, "View", "compressor.py:233", ins=["freqs:?:bf16"], out="f2:?:bf16",
              view="slice", index=":total_seq_len:self.compress_ratio")
    dag = _dag(n, const_scalars={}, scalar_exprs={"total_seq_len": "n_compressed * 4",
                                                  "n_compressed": "128"})
    rep = _run(dag, {"freqs": "S·1·1·qk_pos_emb_head_dim"},
               dims_ctx={"compress_ratio": "4"})
    assert n.out.split(":")[1] == "128·1·1·qk_pos_emb_head_dim", rep


def test_strided_slice_is_refused_when_the_step_does_not_divide_the_stop():
    """除不尽 → 长度要向上取整，取整就是**猜** → 记账保 `?`。"""
    n = _node(1, "View", "x.py:1", ins=["freqs:?:bf16"], out="f2:?:bf16",
              view="slice", index=":S:3")
    rep = _run(_dag(n), {"freqs": "S·H"})
    assert n.out.split(":")[1] == "?"
    assert any(r["reason"] == "slice_bounds_unknown" for r in rep)


def test_plain_stop_slice_is_unchanged():
    """既有 `[:stop]` 形态**逐字不变**（回归钉）。"""
    n = _node(1, "View", "x.py:1", ins=["a:?:bf16"], out="b:?:bf16",
              view="slice", index=":cutoff")
    _run(_dag(n, const_scalars={"cutoff": 128}), {"a": "S·B"})
    assert n.out.split(":")[1] == "128·B"


# ══════════════════════════════════════════════════════════════════════════════
# 5. 纪律：恢复出来的轴不许是猜的
# ══════════════════════════════════════════════════════════════════════════════

def test_recovered_axes_never_come_from_a_declared_bound(monkeypatch):
    """与 `test_mhc_declared_bound_stays_labelled_as_a_bound_not_a_fact` 同一条纪律：
    本轮**没有**引入任何"声明即事实"的通路 —— 轴恢复只用 walker 记下的**源侧常量**
    （`permute_dims` / `squeeze_axis` / 切片 index）与**已核验**的整除事实。"""
    facts = div_facts_from_values([(4, "S")], {"S": 4097})
    assert not facts.has(4, "S")
    # 事实不成立 ⇒ 表达式保持保守形态 ⇒ 下游按轴改形的算子照旧拒绝。
    assert render_term(normalize_axis(parse_axis("4·(S//4)"), facts)) == "4·(S//4)"


def test_normalize_axes_maps_over_a_whole_shape():
    axes = parse_shape("S·B·(64+v_head_dim-64)")
    assert render_shape(normalize_axes(axes)) == "S·B·v_head_dim"
