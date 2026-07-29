"""Task 1 — 手写 `saves=[...]` 名册 **vs** 源真值（`ctx.save_for_backward` / detach）审计。

`crosscheck.py` 结构上发现不了这一类错误 —— 它只比 op **类别计数**，不比每张量 saves/shape/dtype
（`crosscheck.py:81-86` 自己声明）。本文件补上那一刀：把 `cost_eval/layers/dsv4_hybrid.py` 的
`sparse_attn.saves` 与 `csa.py:224` 逐字名单对齐，**已知差异登记在 `KNOWN_GAPS` 台账里并带字节影响**；
任何**台账之外**的新差异 → 大声失败并打可读 diff。

## 为什么是台账（ratchet）而不是「直接改 spec」

改 `saves` 会移动已标定锚点（unfused pp4 全重算 `50187.9/43940.0/43407.1/47888.1` MiB、
fused `24153.3/14641.7/14097.7/23508.0`，评估文档 §「最短路径」8），属于**要单独决策**的口径变更。
故本轮只**钉住**差异 + 报字节影响，不动数字。台账是双向棘轮：
  - 新增未登记差异 → `test_sparse_attn_saves_vs_source_truth` 失败；
  - 差异被修掉（台账项不再成立）→ `test_known_gaps_ledger_is_still_accurate` 失败，
    迫使删台账项，不会留下过期注释。

## 名字映射

源侧（kernel 形参名）与手写侧（评估器张量名）不同名，映射逐条有源依据（`ALIAS`）。不是「猜同义词」：
每条都能在 `csa.py` 的 `.apply(...)` 实参位置上对上（`csa.py:689-707` / `:709-716`）。
"""
import os

import pytest

from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
from cost_eval.model_spec import DimTable, TensorRef
from cost_eval.opdag.fn_saves import scan_tree
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import resolve_tensor
from cost_eval.specs import ParallelConfig

from tests.test_opdag_fn_saves import authoritative_variant_dir

MiB = 1024 * 1024

# ── 站点配置（seq4096 / topk 512 / 64 heads / v_head_dim 512 / b=1，tp=cp=ep=1）──────
SITE_DIMS = DimTable(
    H=4096, F=4096, n_heads=64, n_kv=1, head_dim=128, S=4096, B=1, vocab=129280,
    n_layers=8, q_lora_rank=1024, kv_lora_rank=512,
    qk_rope_head_dim=64, qk_nope_head_dim=0, v_head_dim=512,
    dsa_indexer_n_heads=64, dsa_indexer_head_dim=128, dsa_indexer_topk=512,
    o_groups=8, o_lora_rank=1024, csa_window_size=128,
)

# ── 源名 → 手写名。逐条依据 `csa.py` 的 `.apply(...)` 实参位序 ──────────────────────
# FusedSparseFlashMlaWithIndexerLoss.apply(query, key, compressed_kv, topk_indices,
#     cast(query_index), cast(key_index), cast(weights), attn_sink, ...)  @ csa.py:689-707
# forward 形参序 (csa.py:184-191): query, ori_kv, cmp_kv, cmp_sparse_indices,
#     query_index, key_index, weights, sinks
ALIAS = {
    "query": "q_hnorm",             # csa.py:690 实参 = permute(query)，即顶层 per-head Q-norm(**bf16**)
    "ori_kv": "kv_a_out",           # csa.py:691 实参 = permute(key)，即 kv_a_norm 输出
    "cmp_kv": "compressed_kv",      # csa.py:692 实参 = compressor 输出
    "sparse_indices": "topk_indices",   # csa.py:693 实参 = indexer 的 topk_indices
    "query_index": "idx_query",     # csa.py:694 cast(query_index, bf16)
    "key_index": "idx_key",         # csa.py:695 cast(key_index, bf16)
    "weights": "idx_weights",       # csa.py:696 cast(weights, fp32)
    "sinks": "sinks",               # csa.py:687 attn_sink 的 **fp32 cast 副本**（≠ params 里的 Parameter）
    "cmp_residual": "cmp_residual",     # _prepare_sparse_flash_mla 第 2 返回（csa.py:203）
    "output": "core_out",           # kernel 输出
    "softmax_lse": "softmax_lse",   # kernel 第 2 输出（csa.py:206）
}


def _pm():
    pc = ParallelConfig(dp_shard=1, tp=1, cp=1, ep=1, pp=1)
    return ParallelModel(pc, n_layers=SITE_DIMS.n_layers, world_size=1, edge_pseudo=(0, 0))


def _bytes(t: TensorRef, dims=None) -> int:
    rt = resolve_tensor(t, dims or SITE_DIMS, _pm())
    return rt.local_numel * rt.dtype_bytes


def _dims(fused: bool) -> DimTable:
    return DimTable(**{**SITE_DIMS.as_dict(), "dsa_fused": fused})


def _op(name, ratio, fused):
    d = _dims(fused)
    for op in build_dsv4_hybrid_attn_ops(d, ratio):
        if op.name == name:
            return op, d
    raise AssertionError(f"{name} 不在 ratio={ratio} fused={fused} 的 op 图里")


# ── 已知差异台账（每条带字节影响；数字由 `resolve_tensor` 在站点配置下算出，**非**真机测量）──
# **2026-07-29：台账清空** —— 两条都已按源修掉（`docs/census_arbitration_2026-07-29.md` §1.7/§1.9），
# 故按本文件开头声明的「双向棘轮」规则删除条目，改为**正向**断言（源 11 项逐名在场）。
#   - `sparse_indices`（`csa.py:61-62` `mint.unsqueeze(topk_indices, dim=2)`，视图）：补进
#     `sparse_attn.saves`。`structure_mem` 按名去重（`structure_mem.py:261-262`）且上游 `indexer`
#     已声明同名 saved → `activation_saves` **净 0**（下方 `test_byte_impact_*` 仍钉这一点）。
#   - `sinks`（`csa.py:687` `ops.cast(attn_sink, float32)`）：补进 saves，**+256 B/层**。
KNOWN_GAPS = {}
FIXED_2026_07_29 = {
    "sparse_indices": dict(hand_name="topk_indices", raw_bytes=8 * MiB,
                           net_activation_saves_delta=0),
    "sinks": dict(hand_name="sinks", raw_bytes=256, net_activation_saves_delta=256),
}

# ── ukl1/ukl2：源侧**从不 saved**（KL 目标分布链全在 detach 侧）——单列，因为方向相反 ──
UKL_GAP = dict(
    names=("ukl1", "ukl2"),
    why=("`csa.py:794-795` 两个入参都过 `ops.stop_gradient` → `indexer.py:350` "
         "`matmul(query,key)*softmax_scale` → `:380` fp32 `softmax` 这条链上**没有任何参数**"
         " → MS 不建 autograd 节点 → 反向无节点读它 → 不是 saved 张量。手写侧仍把它们放在"
         "`saves` 里（只额外打了 `detached=True`，而该 flag 只被 `liveness/` 读）"),
    contrast=("同一份 `CSAIndexer` 自己的 `index_scores`（`indexer.py:245` bmm → relu → "
              "×weights → sum）经 `linear_wq_b` 参数携带梯度（`indexer.py:177`）→ **是**真 saved"),
)


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def source_saved_names():
    d = authoritative_variant_dir()
    if d is None:
        pytest.skip("权威 mindformers 快照缺失（见 tests/test_opdag_fn_saves.py 的 md5 门控）")
    return scan_tree(d).saved_names("FusedSparseFlashMlaWithIndexerLoss")


def _diff_message(source_names, hand_names, missing, extra):
    lines = [
        "手写 `sparse_attn.saves` 与源真值 `csa.py:224` 不一致（台账 KNOWN_GAPS 之外的差异）：",
        f"  源说（{len(source_names)} 项，csa.py:224）: {list(source_names)}",
        f"  spec 说（{len(hand_names)} 项，dsv4_hybrid.py sparse_saves）: {sorted(hand_names)}",
    ]
    if missing:
        lines.append("  源里有、spec 里缺（未登记）:")
        for s in sorted(missing):
            lines.append(f"    - {s} (→ 期望手写名 {ALIAS.get(s, '?')})")
    if extra:
        lines.append(f"  spec 里有、源 saved 名单里没有（未登记）: {sorted(extra)}")
    lines.append("  改 saves 会移动已标定锚点 —— 请**显式决策**后更新 KNOWN_GAPS 或 spec，"
                 "勿静默改数字。")
    return "\n".join(lines)


def test_alias_table_covers_every_source_name(source_saved_names):
    """映射表必须覆盖源名单的每一项 —— 否则 diff 会把「没映射」误报成「spec 缺」。"""
    assert set(source_saved_names) <= set(ALIAS), \
        f"ALIAS 未覆盖: {sorted(set(source_saved_names) - set(ALIAS))}"


def test_sparse_attn_saves_vs_source_truth(source_saved_names):
    """fused r4 的 `sparse_attn.saves` ⊇ 源 11 项（2026-07-29 起台账为空 → 逐名全在场）。"""
    op, _ = _op("sparse_attn", 4, fused=True)
    hand = {t.name for t in op.saves}
    missing = {s for s in source_saved_names if ALIAS[s] not in hand}
    unlogged_missing = missing - set(KNOWN_GAPS)
    assert not unlogged_missing, _diff_message(
        source_saved_names, hand, unlogged_missing, set())
    assert missing == set(KNOWN_GAPS) == set()
    for s in source_saved_names:
        assert ALIAS[s] in hand, f"{s} → {ALIAS[s]} 不在 saves 里"


def test_known_gaps_ledger_is_still_accurate(source_saved_names):
    """棘轮反向：台账项若已被修掉（spec 补上了），本测试失败 → 必须删台账项。"""
    op, _ = _op("sparse_attn", 4, fused=True)
    hand = {t.name for t in op.saves}
    for src_name, entry in KNOWN_GAPS.items():
        assert src_name in source_saved_names, \
            f"KNOWN_GAPS 登记了 {src_name!r}，但源 csa.py:224 名单里没有它 —— 台账过期"
        assert entry["hand_name"] not in hand, (
            f"KNOWN_GAPS[{src_name!r}] 已被修复（{entry['hand_name']!r} 现在在 saves 里）"
            f" —— 请删除该台账项，并复核锚点是否随之移动")


def test_hand_saves_count_matches_source_eleven(source_saved_names):
    """源 11 项、手写 **11 项**（2026-07-29 补齐 sparse_indices / sinks；此前 9 项）。"""
    op, _ = _op("sparse_attn", 4, fused=True)
    assert len(source_saved_names) == 11
    assert len(op.saves) == 11
    assert set(KNOWN_GAPS) == set()
    assert set(FIXED_2026_07_29) == {"sparse_indices", "sinks"}


# ── 字节影响（数字由 resolve_tensor 现算，非真机测量）────────────────────────────
def test_byte_impact_sparse_indices_is_activation_saves_neutral():
    """`sparse_indices`(=topk_indices) 8.000 MiB，但按名去重后对 activation_saves **净 0**。"""
    indexer, d = _op("indexer", 4, fused=True)
    names = {t.name for t in indexer.saves}
    assert "topk_indices" in names          # 上游 indexer 已声明 saved → 字典去重
    t = [x for x in indexer.saves if x.name == "topk_indices"][0]
    assert _bytes(t, d) == FIXED_2026_07_29["sparse_indices"]["raw_bytes"] == 8 * MiB
    assert FIXED_2026_07_29["sparse_indices"]["net_activation_saves_delta"] == 0
    # fwd 序：indexer 在 sparse_attn 之前 → 反向序里 indexer.bwd 更晚 → 活跃区间不延长
    order = [o.name for o in build_dsv4_hybrid_attn_ops(_dims(True), 4)]
    assert order.index("indexer") < order.index("sparse_attn")


def test_byte_impact_sinks_is_256_bytes():
    """`sinks` = `ops.cast(attn_sink, fp32)`（csa.py:687）→ [n_heads] fp32 = 256 B/层。"""
    sinks = TensorRef("sinks", ("n_heads",), dtype_bytes=4)
    assert _bytes(sinks) == FIXED_2026_07_29["sinks"]["raw_bytes"] == 256
    op, d = _op("sparse_attn", 4, fused=True)
    assert [p.name for p in op.params] == ["attn_sink"]   # 权重那份仍在 params
    saved = [t for t in op.saves if t.name == "sinks"]    # cast 副本在 saves（2026-07-29 补）
    assert len(saved) == 1 and _bytes(saved[0], d) == 256


def test_byte_impact_ukl_pair_if_removed():
    """`ukl1`/`ukl2`（unfused r4）各 1024.000 MiB —— 这是「误声明为 saved」的字节代价。"""
    op, d = _op("sparse_attn", 4, fused=False)
    ukl = [t for t in op.saves if t.name in UKL_GAP["names"]]
    assert len(ukl) == 2
    for t in ukl:
        assert t.detached is True                      # 已按源侧 detach 标注
        assert t.dtype_bytes == 4                      # indexer.py:380 fp32 softmax
        assert _bytes(t, d) == 1024 * MiB
    total = sum(_bytes(t, d) for t in ukl)
    assert total == 2048 * MiB
    # 全 saves 里的占比（口径变更的量级参考）
    all_bytes = sum(_bytes(t, d) for t in op.saves)
    # 2026-07-29：`q_hnorm` 由 fp32(512 MiB) 改回 bf16(256 MiB) → 17664 → 17408 MiB。
    assert all_bytes == 17408 * MiB
    assert 0.11 < total / all_bytes < 0.12


def test_index_scores_is_legitimately_saved_contrast():
    """对照项：`CSAIndexer` 自己的 `index_scores`（indexer.py:245）**不** detached、照常 saved。"""
    op, d = _op("indexer", 4, fused=False)
    idx = [t for t in op.saves if t.name == "index_scores"]
    assert len(idx) == 1
    assert idx[0].detached is False
    assert _bytes(idx[0], d) == 2048 * MiB
    assert [p.name for p in op.params] == ["idx_wq_b", "idx_wproj"]   # 梯度经参数可达


# ── detach 站点 ↔ `TensorRef.detached` 的对账（源侧证据 → 手写侧 flag）───────────────
@pytest.fixture(scope="module")
def detach_truth():
    d = authoritative_variant_dir()
    if d is None:
        pytest.skip("权威 mindformers 快照缺失（见 tests/test_opdag_fn_saves.py 的 md5 门控）")
    return scan_tree(d)


def test_detached_flags_are_backed_by_source_detach_sites(detach_truth):
    """手写侧每个 `detached=True` 都要有源侧 detach 站点背书（否则就是凭空标）。"""
    inline = [s for s in detach_truth.detach_sites
              if s.form == "inline_arg" and s.enclosing_call == "self.unfused_indexer_loss"]
    assert {(s.src, s.arg) for s in inline} == {
        ("csa.py:794", "query"), ("csa.py:795", "compressed_kv")}
    op, _ = _op("sparse_attn", 4, fused=False)
    flagged = {t.name for t in op.saves if t.detached}
    assert flagged == set(UKL_GAP["names"])


def test_no_other_saves_are_silently_detached():
    """除 ukl1/ukl2，全库 dsv4 attn saves 不得有 `detached` —— 防止悄悄扩散该 flag。"""
    for ratio in (0, 4, 128):
        for fused in (True, False):
            for op in build_dsv4_hybrid_attn_ops(_dims(fused), ratio):
                for t in op.saves:
                    if t.detached:
                        assert (ratio, fused, t.name) in {
                            (4, False, "ukl1"), (4, False, "ukl2")}, \
                            f"未预期的 detached: ratio={ratio} fused={fused} {t.name}"


def test_stop_gradient_site_count_is_six(detach_truth):
    """评估文档 §5：`ops.stop_gradient` 快照内 6 处，全在 csa.py。"""
    sg = [s for s in detach_truth.detach_sites if s.fn == "ops.stop_gradient"]
    assert len(sg) == 6
    assert {os.path.basename(s.file) for s in sg} == {"csa.py"}
    assert [s.lineno for s in sg] == [665, 666, 764, 765, 794, 795]
