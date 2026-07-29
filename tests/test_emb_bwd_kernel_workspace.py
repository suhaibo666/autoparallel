# -*- coding: utf-8 -*-
"""word-embedding 反向 kernel workspace（`GatherDGradV2`）—— **实测项**的守卫门（2026-07-30）。

来源：`docs/head_workspace_2026-07-30.md`。**两轮独立真机采集，四个点，逐字节吻合**：

  1. **167 / MS2.10 memory-tracker**（`docs/kernel_workspace_2026-07-29.md` §4.4/§8④）：
     DSv4-hybrid 站点 vocab=129280 / **H=4096** / **B=1** / tp=cp=1，BWD 相位 stage0 单 kernel
     极大值 —— S=1024/2048/4096 → 2084.003 / 2132.003 / 2228.003 MiB。
  2. **`analysis/realmachine/pp2_norecomp/op_816362.csv`**（**仓内既有** profiler，DSv3 pp2 无重算）：
     vocab=129280 / **H=1792** / **B=2** / S=4096 / tp=cp=1 —— 瞬态块 `Size(KB)=1093379.0`
     = **1119620096 B**。本门**直接从该 CSV 读**，不转抄常数。

守四件事：

  A. 两个站点的实测值都逐字节复现（跨 **H**、跨 **B**、跨 **S**）。
  B. **归属**：整个 embedding 段的 bwd_workspace 全部来自 `embedding` 这一个 op
     —— 挡「把它摊到层上」以及「挪到 lm_head 上好让锚点动」。
  C. **它不是 loss 侧的**：`4·vocab·H` 主项与 S 无关（S 减半，值只掉 48 MiB 而非折半）。
     这是把 `GatherDGradV2` 判给 embedding 而非 CE-gather 的**决定性判据**（§2.1），
     在 167 站点上 S·B==H 使两种假设在单点上数值相同，只有扫 S 能分开。
  D. cp 结构：常数项不缩放、每-token 项 ÷cp（与 `docs` §6 的适用边界一致）。
"""
import csv
import dataclasses
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cost_eval.build_llm import build_llm_spec              # noqa: E402
from cost_eval.parallel_model import ParallelModel          # noqa: E402
from cost_eval.presets import deepseek_v3                   # noqa: E402
from cost_eval.shape_eval import ShapeEval                  # noqa: E402
from cost_eval.specs import ParallelConfig                  # noqa: E402
from cost_eval.structure_mem import estimate_structure_memory  # noqa: E402

MiB = 2 ** 20

#: 采集 ①：167 真机 memory-tracker 实测（MS2.10/CANN9.1/910B2，run c，H=4096/B=1/tp=cp=1）。
#: **测量值，不得为了让模型好看而编辑**；三位小数即 tracker 报出的精度。
REAL_EMB_GATHER_DGRAD_WS_MiB = {1024: 2084.003, 2048: 2132.003, 4096: 2228.003}

#: 采集 ② 的 profiler 文件（仓内既有真机数据）。
_PP2_STAGE0_CSV = os.path.join(_REPO, "analysis", "realmachine",
                               "pp2_norecomp", "op_816362.csv")
_PP2_STAGE1_CSV = os.path.join(_REPO, "analysis", "realmachine",
                               "pp2_norecomp", "op_816365.csv")


def _gather_dgrad_blocks(path):
    """(bytes, duration_us) —— 该 profiler 文件里全部 `GatherDGradV2` 分配。"""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return [(int(float(r["Size(KB)"]) * 1024), float(r["Duration(us)"]))
                for r in csv.DictReader(fh) if r["Name"] == "GatherDGradV2"]


def _emb_bwd_workspace(seq, B, H, vocab=129280, tp=1, cp=1, dp=2):
    """走生产 builder（`build_llm_spec` → ShapeEval → StructureMemory）取 embedding 段的值。"""
    cfg = dataclasses.replace(deepseek_v3(4), seq_length=seq, hidden_size=H, vocab_size=vocab)
    spec = build_llm_spec(cfg)
    spec.dims.B = B
    pc = ParallelConfig(dp_shard=dp, tp=tp, ep=1, pp=1, cp=cp,
                        sequence_parallel=True, num_microbatches=1)
    pm = ParallelModel(pc, len(spec.layer_pattern), dp * tp * cp)
    graph = ShapeEval().resolve(spec, pm)
    lay = [l for l in graph.stages[0] if l.layer_type == "embedding"][0]
    return estimate_structure_memory(lay.ops).bwd_workspace, lay


# ── A. 两个站点的实测值逐字节复现 ────────────────────────────────────────────────
@pytest.mark.parametrize("seq", sorted(REAL_EMB_GATHER_DGRAD_WS_MiB))
def test_167_measured_law_reproduces_end_to_end(seq):
    """采集 ①（H=4096/B=1）：三个 seq 各一格，逐字节（到 tracker 的三位小数精度）。"""
    got, _ = _emb_bwd_workspace(seq, B=1, H=4096)
    want = REAL_EMB_GATHER_DGRAD_WS_MiB[seq]
    assert round(got / MiB, 3) == want, (
        f"seq={seq}: 模型 {got / MiB:.5f} MiB != 167 真机实测 {want} MiB。"
        f"这条实测值只能由新的真机测量改动（docs/head_workspace_2026-07-30.md §1.2）。")


def test_pp2_profiler_transient_block_reproduces_byte_exactly():
    """采集 ②（H=1792/B=2/S=4096）：**从仓内 profiler CSV 直接读**，不转抄常数。

    该文件里 `GatherDGradV2` 的**最大瞬态块**就是这笔 workspace（另有一档长寿块 =
    `4·vocab·H` 的 fp32 梯度表本体，寿命≈全程，由 `grad_buf` 桶承担，不在本项内）。"""
    blocks = _gather_dgrad_blocks(_PP2_STAGE0_CSV)
    transient = [b for b, dur in blocks if dur < 1000]      # 瞬态判据：远短于长寿块的 5.6e6 us
    observed = max(transient)
    got, _ = _emb_bwd_workspace(4096, B=2, H=1792)
    assert got == observed, (
        f"模型 {got} B != profiler 实测 {observed} B（{_PP2_STAGE0_CSV}）——"
        f"跨 H(1792 vs 4096) / 跨 B(2 vs 1) 的缩放被改坏了。")


def test_pp2_profiler_long_lived_block_is_the_grad_table_not_this_workspace():
    """同一 CSV 里那档**长寿**块 = `4·vocab·H` 的 fp32 梯度表本体（+512 B 分配器尾）。

    这条钉住「workspace 与梯度表是两块、不双计」这个前提：模型把梯度表算在
    `grad_buf`（`StructureMemory.grad_full_bytes`）里，把本项算在 `workspace` 桶里。"""
    blocks = _gather_dgrad_blocks(_PP2_STAGE0_CSV)
    long_lived = [b for b, dur in blocks if dur > 1e6]
    assert long_lived, "该 profiler 里应有长寿的梯度表块"
    table = 4 * 129280 * 1792
    assert 0 <= max(long_lived) - table <= 4096, (
        f"长寿块 {max(long_lived)} B 与 4·vocab·H = {table} B 不再吻合")


def test_it_only_appears_on_the_rank_that_owns_the_embedding():
    """判据 ②（§2.2）：同跑的另一个 rank（stage 1，**无 embedding**）只剩小件。

    这是把它判给 embedding 而非 lm_head/loss 的证据之一 —— pp2 的 **lm_head 在 stage 1**，
    若这笔 workspace 属 head/loss，它就该出现在 `op_816365.csv` 而不是 `op_816362.csv`。
    实见：stage1 最大 16.126 MiB（与 167 采集里 rank2 的 16.03 MiB 同构），
    只有 stage0 那笔的 1/66。"""
    s1 = [b for b, _ in _gather_dgrad_blocks(_PP2_STAGE1_CSV)]
    s0 = [b for b, dur in _gather_dgrad_blocks(_PP2_STAGE0_CSV) if dur < 1000]
    assert s1, "stage1 也应有小件 GatherDGradV2"
    assert max(s1) * 50 < max(s0), (
        f"stage1（无 embedding）最大块 {max(s1)} B 不该与 stage0 的 {max(s0)} B 同量级"
        f"——那会推翻「这笔 workspace 属 embedding 反向」的归属判据。")


# ── B. 归属 ─────────────────────────────────────────────────────────────────────
def test_it_is_carried_by_the_embedding_op_only():
    """整个 embedding 段的 bwd_workspace 全部来自 `embedding` 这一个 op。

    挡两种失败模式：①把层级常数摊在层上；②为了让 `bwd@head` 峰值锚点动而把它挪到
    lm_head 段——测量把它定位到 word-embedding 的 gather 反向，模型就必须挂在那里
    （即便这意味着 13 个 OOM-不安全锚点一个字节不动，见 docs §2.4）。"""
    _, lay = _emb_bwd_workspace(4096, B=1, H=4096)
    carriers = {op.name: getattr(op, "bwd_workspace_bytes", 0)
                for op in lay.ops if getattr(op, "bwd_workspace_bytes", 0)}
    assert list(carriers) == ["embedding"], carriers


# ── C. 它不是 loss 侧的（决定性判据）────────────────────────────────────────────
def test_dominant_term_is_S_independent_which_rules_out_the_loss_side_gather():
    """S 折半时本项**几乎不变**（−48 MiB / −2.2 %），而 loss 侧 `[S·B,vocab]` fp32 会折半。

    167 站点上 S·B == H == 4096 → 两种假设在 S=4096 单点上数值恰好相同；只有扫 S 能分开。
    本门把这条判据固化：主项必须是 `4·vocab·H`（S 无关），不是 `4·vocab·B·S`。"""
    hi, _ = _emb_bwd_workspace(4096, B=1, H=4096)
    lo, _ = _emb_bwd_workspace(2048, B=1, H=4096)
    assert hi - lo == 12 * 4096 * 1 * 2048, "S 差分必须恰是每-token 项 12·H·ΔS"
    loss_side = 4 * 129280 * 4096         # [S·B, vocab] fp32 @ S=4096,B=1
    assert lo > 0.9 * hi, (
        f"S 折半后本项应几乎不变（实测 {lo / MiB:.1f} vs {hi / MiB:.1f} MiB）；"
        f"若它按 loss 侧假设折半就该跌到 ~{(hi - loss_side / 2) / MiB:.1f} MiB。")


# ── D. cp 结构 ──────────────────────────────────────────────────────────────────
def test_cp_shards_only_the_per_token_term():
    """cp>1：常数项（`4·vocab·H + 16 MiB + 3072 B`）不缩放、每-token 项 ÷cp。

    与 `docs/head_workspace_2026-07-30.md` §6 的适用边界一致：结构上正确，但 cp>1 **未实测**。
    （整体 ÷cp 会把常数项也除掉 → cp>1 欠读 = OOM-不安全，本门挡的就是那个。）"""
    base, _ = _emb_bwd_workspace(4096, B=2, H=1792, dp=1)
    cp2, _ = _emb_bwd_workspace(4096, B=2, H=1792, dp=1, cp=2)
    per_token = 12 * 1792 * 2 * 4096
    assert base - cp2 == per_token // 2
    assert cp2 > base // 2, "常数项不得随 cp 缩放（那会造成 cp>1 欠读）"
