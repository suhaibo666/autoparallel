"""守卫门：r4 `CSAIndexer` **内部** RoPE 的反向保留对（`idx_rope_f32` / `idx_rope_rot`）。

依据：`docs/r4_indexer_census_2026-07-30.md`。守的是**认领**（源结构 + 实测块数），
不是模型自洽 —— 每条断言背后都有一个能变红的具体错法：

1. 站点尺寸下各**恰好 64.000 MiB**、`dtype_bytes == 4` —— 抓「lane 写错 / fp32 写回 bf16」。
2. lane 表达式恒等于 `dsa_indexer_n_heads*qk_rope_head_dim`（**≠** 顶层 q/inv rope 的
   `n_heads*qk_rope_head_dim`）—— 两者在本站点碰巧同值（都是 64×64），用**表达式**钉，
   防「站点巧合」被固化成建模事实。
3. fused **与** unfused 两侧都在场 —— `forward_before_topk` 在两条路径上被逐字调用
   （`csa.py:667` `_construct_fused` / `csa.py:766` `_construct_naive`），rope 调用在它内部，
   与 `apply_dsa_kernel_fusion` 无关（该开关只切 `CSAIndexer.construct` 里的打分实现）。
4. **r0 / r128 恒无**（fused × unfused 四种组合），且它们的注意力段 saves 与本轮之前逐字节
   相等 —— 挡「拿 r4 的东西去顶别的层型」。源判据：`csa.py:608`
   `if compress_ratio == 4 and not config.csa_dense_mode and submodules.indexer is not None:`
   → `self.enable_indexer = True`，否则 `:623-626` 全置 `None`（**没有** `CSAIndexer` 对象）。
5. 名字不与顶层 rope 的六张相撞 —— `structure_mem` 按名去重（`structure_mem.py:261-262`），
   撞名会把新增**静默吞成 0**（这是本仓踩过的一类 bug）。
6. **实测块数门**：fused 侧模型里 r4 比 r128 多出的「恰好 64.000 MiB」张量**数**必须 == **3**
   （真机块台账，`docs/kernel_workspace_2026-07-29.md` §5.2 run c：r4 层 7 块 / r128 层 4 块）。
   这道门守的是**测量**：再往 r4 的 64 MiB 档里塞一张、或把本对删掉，都会变红。
   ⚠ 只对 **fused** 成立：`idx_query` 是融合 kernel 的 `ctx.save_for_backward` 项
   （`csa.py:224,229`），unfused 路径没有那个 ctx，故 unfused 侧的差是 2 —— 而块台账本身
   就采自 run c（fused），口径一致。
7. 注意力段 saves 净增**恰好 128.000 MiB / r4 层**（去重后），r0/r128 净增 0。

复用 `tests/test_source_truth_saves_audit.py` 的站点 `DimTable` 与 `_bytes`（同一把尺子）。
"""
import pytest

from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
from cost_eval.shape_eval import resolve_tensor

from tests.test_source_truth_saves_audit import _bytes, _dims, _pm

MiB = 1024 * 1024

#: 本轮新建的两张（`indexer.py:182-187` → `rope_utils.py:186-187` 的 `t` / `t_rot`）。
IDX_ROPE = ("idx_rope_f32", "idx_rope_rot")
#: indexer 内部那次 rope 的 lane —— **索引器自己的头数**，不是顶层注意力的头数。
IDX_ROPE_LANE = "dsa_indexer_n_heads*qk_rope_head_dim"
#: 顶层 DSv4HybridSelfAttention 的三次 rope 调用留下的六张（对照组，名字不得相撞）。
TOP_ROPE = ("q_rope_f32", "q_rope_rot", "k_rope_f32", "k_rope_rot",
            "inv_rope_f32", "inv_rope_rot")

#: 站点尺寸下、**注意力段** op 图去重后的 saves 总字节（MiB）。r4 两侧含本轮的 +128.000。
#: 这三行是**漂移哨兵**：数字由 `resolve_tensor` 现算，不是真机测量。
SITE_ATTN_SAVES_MIB = {
    (True, 0): 1435.000, (True, 4): 1638.000, (True, 128): 1435.031,
    (False, 0): 6678.000, (False, 4): 20511.000, (False, 128): 7478.031,
}
#: 本轮之前的同一口径（r4 = 上表 − 128.000；r0/r128 逐字节相等）。
SITE_ATTN_SAVES_MIB_BEFORE = {
    (True, 0): 1435.000, (True, 4): 1510.000, (True, 128): 1435.031,
    (False, 0): 6678.000, (False, 4): 20383.000, (False, 128): 7478.031,
}
#: 本项的净增（MiB/层）—— 只有 r4 非零。
NET_DELTA_MIB = {0: 0.0, 4: 128.0, 128: 0.0}


def _saves(ratio, fused):
    """注意力段 op 图里**按名去重**后的 saved 张量（与 `structure_mem` 同一去重规则）。"""
    d = _dims(fused)
    seen = {}
    for op in build_dsv4_hybrid_attn_ops(d, ratio):
        for t in op.saves:
            seen.setdefault(t.name, t)
    return seen, d


# ---------------------------------------------------------------------------
# 1 / 2 —— 字节与 lane 表达式
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("name", IDX_ROPE)
def test_idx_rope_is_exactly_64_mib_fp32(name, fused):
    """站点 S=4096/B=1/index_n_heads=64/qk_pos_emb_head_dim=64 → 4096·(64·64)·4B = 64.000 MiB。

    fp32 的源依据：`rope_utils.py:169` `t = self.cast(t, self.rotary_dtype)`，站点
    `rotary_dtype: "float32"`（`prep_dsv4align.py` 与 `transformer_config.py:166-167`）。"""
    seen, d = _saves(4, fused)
    assert name in seen, f"{name} 不在 ratio=4 fused={fused} 的 op 图里"
    t = seen[name]
    assert t.dtype_bytes == 4, f"{name} 应为 fp32（rope_utils.py:169 的 rotary_dtype）"
    assert _bytes(t, d) == 64 * MiB, f"{name} = {_bytes(t, d) / MiB:.3f} MiB ≠ 64.000"


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("name", IDX_ROPE)
def test_idx_rope_lane_is_the_indexer_head_count_not_the_attention_one(name, fused):
    """用**表达式**钉 lane：本站点 `n_heads` 与 `dsa_indexer_n_heads` 碰巧都是 64，
    只钉字节数会把这个巧合固化。顶层 q/inv rope 用的是 `n_heads*qk_rope_head_dim`。"""
    seen, _ = _saves(4, fused)
    assert seen[name].shape == ("S", "B", IDX_ROPE_LANE), seen[name].shape
    top = _saves(4, fused)[0]["q_rope_f32"]
    assert top.shape == ("S", "B", "n_heads*qk_rope_head_dim")
    assert seen[name].shape != top.shape, "lane 表达式与顶层撞了 —— 站点巧合被固化"


# ---------------------------------------------------------------------------
# 3 —— fused 与 unfused 两侧都在场
# ---------------------------------------------------------------------------
def test_present_on_both_fused_and_unfused_paths():
    """`forward_before_topk` 在 `csa.py:667`（融合）与 `csa.py:766`（小算子）被**逐字**调用，
    rope 调用在它内部 → 与 `apply_dsa_kernel_fusion` 无关，两侧都必须声明。"""
    for fused in (True, False):
        seen, _ = _saves(4, fused)
        for name in IDX_ROPE:
            assert name in seen, f"fused={fused} 侧缺 {name}"
        owners = {op.name for op in build_dsv4_hybrid_attn_ops(_dims(fused), 4)
                  for t in op.saves if t.name in IDX_ROPE}
        assert owners == {"indexer"}, f"归属 op 应是 indexer，实为 {owners}"


# ---------------------------------------------------------------------------
# 4 —— r0 / r128 恒无，且逐字节不动
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("ratio", [0, 128])
def test_absent_from_r0_and_r128(ratio, fused):
    """`csa.py:608`：只有 `compress_ratio == 4` 才建 `CSAIndexer`；否则 `:623-626` 全 None
    → 没有 `forward_before_topk`、没有那次 rope 调用。"""
    seen, _ = _saves(ratio, fused)
    leaked = [n for n in seen if n.startswith("idx_rope")]
    assert not leaked, f"r{ratio} fused={fused} 里出现了 {leaked} —— 拿 r4 的东西顶别的层型"


@pytest.mark.parametrize("fused,ratio", sorted(SITE_ATTN_SAVES_MIB))
def test_attention_saves_totals_and_net_delta(fused, ratio):
    """注意力段去重 saves 的绝对值（漂移哨兵）+ 相对本轮之前的净增。

    净增必须**恰好** 128.000 MiB/r4 层、r0/r128 **恰好 0** —— 这同时挡住「r4 少建一张」
    与「顺手把别的层型也改了」两个方向。"""
    seen, d = _saves(ratio, fused)
    total = sum(_bytes(t, d) for t in seen.values()) / MiB
    assert abs(total - SITE_ATTN_SAVES_MIB[(fused, ratio)]) < 1e-3, total
    delta = total - SITE_ATTN_SAVES_MIB_BEFORE[(fused, ratio)]
    assert abs(delta - NET_DELTA_MIB[ratio]) < 1e-3, (
        f"r{ratio} fused={fused} 净增 {delta:.3f} MiB ≠ 期望 {NET_DELTA_MIB[ratio]:.3f}")


# ---------------------------------------------------------------------------
# 5 —— 名字不与顶层 rope 的六张相撞
# ---------------------------------------------------------------------------
def test_names_do_not_collide_with_top_level_rope():
    """`structure_mem` 按名去重（`structure_mem.py:261-262`）→ 撞名会把新增**静默吞成 0**。"""
    seen, _ = _saves(4, True)
    assert set(IDX_ROPE).isdisjoint(TOP_ROPE)
    for n in TOP_ROPE:
        assert n in seen, f"顶层 rope 的 {n} 不见了 —— 对照组本身漂了"
    assert len(set(IDX_ROPE) | set(TOP_ROPE)) == 8


# ---------------------------------------------------------------------------
# 6 —— 实测块数门（守的是**测量**，不是模型自洽）
# ---------------------------------------------------------------------------
def test_block_count_delta_matches_the_real_machine_ledger():
    """真机块台账（`docs/kernel_workspace_2026-07-29.md` §5.2，run **c = fused**，同 rank 比较）：
    r4 层 **7** 块 64.0005 MiB、r128 层 **4** 块 → **差 3**。模型 fused 侧必须给出同样的差。

    ⚠ 只断言**差**，不断言绝对块数：模型 fused 侧是 8 / 5，实测是 7 / 4 —— 绝对值各多 1 块，
    那一块在**三个层型共有**的部分（模型对所有 DSv4 层型都声明 5 张 64 MiB，实测 r128 只见 4），
    与本轮的 r4 专属项无关，**未闭合**、如实登记在
    `docs/r4_indexer_census_2026-07-30.md` §7 ③。不得为了凑绝对数去删/加张量。"""
    def n64(ratio):
        seen, d = _saves(ratio, True)
        return sum(1 for t in seen.values() if _bytes(t, d) == 64 * MiB)

    assert n64(4) - n64(128) == 3, (n64(4), n64(128))
    assert n64(4) == 8 and n64(128) == 5 and n64(0) == 5   # 钉住绝对数，好让 ③ 一旦闭合就变红


def test_the_third_block_is_idx_query_and_predates_this_round():
    """三块里的第 ① 块是 `query_index`（`csa.py:229` 在 `ctx.save_for_backward` 名单内，
    `csa.py:224`），模型侧名为 `idx_query`，**本轮之前就已建**。本测试把这条认领钉住，
    免得后人以为三块全是本轮加的。"""
    seen, d = _saves(4, True)
    assert "idx_query" in seen and _bytes(seen["idx_query"], d) == 64 * MiB
    # `csa.py:694` `ops.cast(query_index, mstype.bfloat16)` → 走 compute dtype（`dtype_bytes=None`
    # 表示「随 compute dtype」，站点是 bf16）→ resolve 后 2 B/元素。与本轮两张显式 fp32 相对照。
    assert seen["idx_query"].dtype_bytes is None
    assert resolve_tensor(seen["idx_query"], d, _pm()).dtype_bytes == 2
    owners = {op.name for op in build_dsv4_hybrid_attn_ops(_dims(True), 4)
              for t in op.saves if t.name == "idx_query"}
    assert owners == {"sparse_attn"}, owners       # 融合 kernel 的 ctx，不是 indexer 自己
    # unfused 侧没有那个 ctx（`FusedSparseFlashMlaWithIndexerLoss` 只在融合路径上）
    assert "idx_query" not in _saves(4, False)[0]
