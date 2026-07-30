# -*- coding: utf-8 -*-
"""`lm_head` **反向** kernel workspace 的守卫门（2026-07-30）。

守的是**测量**，不是模型自洽。测量出处：`docs/head_loss_bwd_workspace_2026-07-30.md`
（167 / MindSpore 2.10 memory-tracker，8 卡真机 run `c` 家族，tracker 装在末 stage 的
rank 6；kernel 由**输出签名**认定；同层型无 head 的 rank 4 作对照）。

三组不变量：
  A  实测律逐字节穿过 `build_llm_spec → ShapeEval → StructureMemory`（7 个实测点）；
  B  归属只在 `lm_head` 这一个 op 上，且**已知未建模的那几笔**必须留 0（如实欠读）；
  C  轴向语义（cp / tp / B）与重建口不吞字段。
"""
import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("COST_EVAL_EXTRACTED_ALLOW_PARTIAL", "1")

from cost_eval.build_llm import build_llm_spec                      # noqa: E402
from cost_eval.parallel_model import ParallelModel                  # noqa: E402
from cost_eval.presets import deepseek_v3                           # noqa: E402
from cost_eval.shape_eval import ShapeEval                          # noqa: E402
from cost_eval.specs import ParallelConfig                          # noqa: E402
from cost_eval.structure_mem import estimate_structure_memory       # noqa: E402

MiB = 2 ** 20

#: 167 真机 memory-tracker 实测（rank 6 = 末 stage，每格 12 个窗口的极大值），**测量值，不得编辑**。
#: key = (S, vocab, H)，B=1 / tp=1 / cp=1。出处见本文件头的报告 §2。
REAL_HEAD_DGRAD_WS_B = {
    (4096, 129280, 4096): 1147143168,
    (2048, 129280, 4096): 584057856,
    (1024, 129280, 4096): 302515200,
    (4096, 32320, 4096): 352846848,
    (4096, 129280, 2048): 1113588736,
    (4096, 129280, 1792): 1109394432,
}
#: 同 kernel 的 wgrad 那一笔（**恒小于 dgrad**，层内取 max 即 dgrad）——同一次采集。
REAL_HEAD_WGRAD_WS_B = {
    (4096, 129280, 4096): 1113589760,
    (2048, 129280, 4096): 567281664,
    (1024, 129280, 4096): 294127616,
    (4096, 129280, 2048): 1096812544,
    (4096, 129280, 1792): 1094715392,
}
#: ⚠ 实测**不成立**的那一点：vocab=64640 处 `2*vocab` 整项消失（单卡密集扫描逐字节复现）。
#: 本律在该 shape 上**过读**——OOM 安全侧，但必须钉住这条已知偏差，防止被当成「律普适」。
REAL_HEAD_DGRAD_WS_B_ANOMALY = {(4096, 64640, 4096): 88081408}


def _law(S, B, H, V):
    return (2 * V + 4 * H) * (B * S) + 20971520 + 1024


def _head_layer(S=4096, V=129280, H=4096, pc=None):
    cfg = dataclasses.replace(deepseek_v3(4), seq_length=S, vocab_size=V, hidden_size=H)
    spec = build_llm_spec(cfg)
    pc = pc or ParallelConfig()
    world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
    pm = ParallelModel(pc, spec.dims.n_layers, world or 1)
    g = ShapeEval().resolve(spec, pm)
    for layers in g.stages.values():
        for lay in layers:
            if any(o.name == "lm_head" for o in lay.ops):
                return lay, spec
    raise AssertionError("no lm_head layer")


# ═══════════════════════════════════════════════════════════════════════════════════
# A. 实测律逐字节穿链
# ═══════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("key", sorted(REAL_HEAD_DGRAD_WS_B))
def test_measured_dgrad_workspace_is_byte_exact(key):
    """7 个实测点里的 6 个（另一个是反例，见下）逐字节穿过整条链。"""
    S, V, H = key
    lay, _ = _head_layer(S, V, H)
    got = estimate_structure_memory(lay.ops).bwd_workspace
    assert got == REAL_HEAD_DGRAD_WS_B[key], (key, got, REAL_HEAD_DGRAD_WS_B[key])
    # 律与实测同值（若有人改律但顺手改了常数表，这一条会分道扬镳）
    assert got == _law(S, 1, H, V)


@pytest.mark.parametrize("key", sorted(REAL_HEAD_WGRAD_WS_B))
def test_layer_max_is_the_dgrad_not_the_wgrad(key):
    """同 op 的 wgrad 实测更小 → 层内 max 必须是 dgrad。挡「顺手换成 2*H 那条式子」。"""
    S, V, H = key
    assert REAL_HEAD_WGRAD_WS_B[key] < REAL_HEAD_DGRAD_WS_B[key]
    lay, _ = _head_layer(S, V, H)
    assert estimate_structure_memory(lay.ops).bwd_workspace == REAL_HEAD_DGRAD_WS_B[key]


def test_vocab_64640_is_a_recorded_over_read_not_a_law():
    """⚠ vocab 扫描证伪了「光滑律」：64640 那一点实测只有 `4*H*(B*S) + 20 MiB + 1024`。

    本门把这条**已知过读**钉住（OOM 安全侧），并防止有人为了让它「对上」而把
    `2*vocab` 项删掉 —— 删掉会让其余 10 个 vocab 值上全部欠读 = OOM-**不安全**。"""
    (key, real), = REAL_HEAD_DGRAD_WS_B_ANOMALY.items()
    S, V, H = key
    lay, _ = _head_layer(S, V, H)
    got = estimate_structure_memory(lay.ops).bwd_workspace
    assert got == _law(S, 1, H, V)
    assert got > real, "本律在 vocab=64640 上必须是过读（安全侧）"
    assert got - real == 2 * V * S, "过读量必须恰是那一份消失的 operand 拷贝 2*vocab*(B*S)"


def test_scales_with_B_through_token_count():
    """B 轴：per-token 项经 `B*S`。B=2 的实测点（单卡 + 仓内 profiler 两处）为 N=8192。"""
    lay1, spec1 = _head_layer(4096, 129280, 1792)
    v1 = estimate_structure_memory(lay1.ops).bwd_workspace
    assert spec1.dims.B == 1
    spec2 = build_llm_spec(dataclasses.replace(deepseek_v3(4), hidden_size=1792))
    spec2.dims.B = 2
    pm = ParallelModel(ParallelConfig(), spec2.dims.n_layers, 1)
    g = ShapeEval().resolve(spec2, pm)
    lay2 = [l for ls in g.stages.values() for l in ls
            if any(o.name == "lm_head" for o in l.ops)][0]
    v2 = estimate_structure_memory(lay2.ops).bwd_workspace
    # per-token 项翻倍、常数项不翻倍
    assert v2 - v1 == (2 * 129280 + 4 * 1792) * 4096
    # 且 B=2 那一点就是单卡 + 仓内 profiler 都量到的 2197816320 B
    assert v2 == 2197816320


# ═══════════════════════════════════════════════════════════════════════════════════
# B. 归属 + 已知未建模项必须留 0
# ═══════════════════════════════════════════════════════════════════════════════════

def test_carrier_is_only_lm_head():
    """挡两件事：把层级常数摊在别的 op 上；为了让某个锚点动而挪到 `nll` / `final_norm` 上。"""
    lay, _ = _head_layer()
    per = {o.name: getattr(o, "bwd_workspace_bytes", 0) for o in lay.ops}
    assert per["lm_head"] == REAL_HEAD_DGRAD_WS_B[(4096, 129280, 4096)]
    for nm in ("final_norm", "logsoftmax", "nll"):
        assert per[nm] == 0, nm


def test_grad_accumulate_workspace_is_deliberately_not_modelled():
    """`InplaceAddExt`（head 权重梯度累加）的 workspace 实测 `2*vocab*H + 2560`，**刻意不建**。

    理由（报告 §7 ①）：它**不是 head 专有**——解码层的每个参数也各有一笔（同跑实测
    `67111424 B` ×24 / `33556992 B` ×12）→ 只挂在 head 上是误归属。**影响面**：只在
    `2*vocab*H > 本律` 时它才是层内 max，即本站点 `B*S ≲ 3775`；今天全部锚点 head 段
    `B*S ≥ 4096`。本门把这条边界钉住：S=4096 时本律必须已经盖过它。"""
    S, V, H = 4096, 129280, 4096
    accum = 2 * V * H + 2560            # 实测（3 个 H × 3 个 vocab 共 5 点逐字节）
    assert accum == 1059064320
    lay, _ = _head_layer(S, V, H)
    got = estimate_structure_memory(lay.ops).bwd_workspace
    assert got > accum, "S=4096 站点上本律必须盖过梯度累加那一笔（否则层内 max 选错）"
    # 而 S=1024 时它反过来 —— 这就是那条**已知欠读**，如实钉住
    lay_lo, _ = _head_layer(1024, V, H)
    assert estimate_structure_memory(lay_lo.ops).bwd_workspace < accum


def test_wgrad_law_matches_the_archived_profiler_byte_exactly():
    """跨站点：仓内既有 profiler（H=1792 / B=2 / 另一次 campaign）里的 wgrad 那一笔。

    **直接从 CSV 读**，不转抄常数 —— 跨 H、跨 B、跨 build 的缩放被改坏必红。"""
    import csv
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "analysis", "realmachine", "pp2_norecomp", "op_816365.csv")
    if not os.path.exists(p):
        pytest.skip("archived profiler CSV not present")
    sizes = set()
    with open(p) as fp:
        for r in csv.DictReader(fp):
            if r["Name"] == "MatMulExt":
                sizes.add(int(float(r["Size(KB)"]) * 1024))
    V, H, N = 129280, 1792, 8192
    wgrad_law = (2 * V + 2 * H) * N + 20971520 + 2048
    assert wgrad_law == 2168457216
    assert wgrad_law in sizes, "wgrad 律必须逐字节命中 op_816365.csv 的一个 MatMulExt 块"
    # 同文件里 dgrad 那一档比本律多整整一份 `2*vocab*N`（另一个 build 的 kernel 选择，报告 §2.3）
    assert 4315940352 in sizes
    assert 4315940352 - _law(4096, 2, H, V) == 2 * V * N + 512


# ═══════════════════════════════════════════════════════════════════════════════════
# C. 轴向语义 + 重建口不吞字段
# ═══════════════════════════════════════════════════════════════════════════════════

def test_cp_divides_only_the_per_token_part():
    """常数项吃 ÷cp 会让 cp>1 **欠读 = OOM-不安全**；per-token 项必须 ÷cp（loss/head 区随 cp 切）。"""
    lay1, _ = _head_layer(pc=ParallelConfig(dp_shard=1, cp=1))
    lay2, _ = _head_layer(pc=ParallelConfig(dp_shard=1, cp=2))
    v1 = estimate_structure_memory(lay1.ops).bwd_workspace
    v2 = estimate_structure_memory(lay2.ops).bwd_workspace
    per_token = (2 * 129280 + 4 * 4096) * 4096
    assert v1 - v2 == per_token // 2, (v1, v2)
    assert v2 - per_token // 2 == 20971520 + 1024, "常数项不得被 cp 整除"
    # 非空转自检：cp=2 必须真切到别的量上（否则这道门可能在测一个没生效的旋钮）
    assert (estimate_structure_memory(lay2.ops).activation_saves
            < estimate_structure_memory(lay1.ops).activation_saves)


def test_tp_is_not_divided_which_is_the_safe_direction():
    """真机 `head_w` 是 `Shard(1)`（vocab ÷tp）→ tp>1 真值更小；本式不 ÷tp = **过读 = 安全侧**。

    未实测，故钉住「不 ÷tp」这个方向（若哪天有人加了 ÷tp 而没有实测背书，必红）。"""
    lay1, _ = _head_layer(pc=ParallelConfig(dp_shard=1, tp=1))
    lay2, _ = _head_layer(pc=ParallelConfig(dp_shard=1, tp=2))
    assert (estimate_structure_memory(lay1.ops).bwd_workspace
            == estimate_structure_memory(lay2.ops).bwd_workspace)


def test_survives_the_mtp_rebuild_that_swallowed_fields_twice_before():
    """`head.py` 的 MTP 路对 `mhc_wrap` 的 `wrapped[0]` 做 `dataclasses.replace`；
    `residual._rebuild` 同款。这条 bug class（重建口静默吞字段）真的发生过三次
    （`norm_kind` / `workspace_ref` / 见 census_fix_residual_carrier §5b）。

    本门取**MTP 层里的共享 head op**，确认本字段活着穿过了那些重建口。"""
    from cost_eval.layers.head import build_mtp_ops
    cfg = dataclasses.replace(deepseek_v3(4), mtp_num_layers=1)
    ops = build_mtp_ops(cfg)
    heads = [o for o in ops if o.name == "lm_head"]
    assert len(heads) == 1
    assert heads[0].bwd_workspace is not None and heads[0].bwd_workspace_ref is not None


def test_phase_channels_do_not_bleed():
    """本项只占 BWD 通道；head 段的 **fwd** `workspace` 必须一个字节不变。"""
    lay, _ = _head_layer()
    sm = estimate_structure_memory(lay.ops)
    assert sm.bwd_workspace == REAL_HEAD_DGRAD_WS_B[(4096, 129280, 4096)]
    assert sm.workspace == 0, "head 段前向 workspace 未测 → 必须仍是 0"
    for o in lay.ops:
        if o.name == "lm_head":
            assert o.workspace_bytes == 0
