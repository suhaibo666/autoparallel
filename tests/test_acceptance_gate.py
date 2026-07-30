"""**验收门**（`tools/liveness_ab_validate.py`）的门测试 —— 167 A/B 八跑（2026-07-25）。

这个文件把"验收台"本身钉住，包含四类断言：

  A. **真机地面真值不可漂移**：`REAL` 表的规范化指纹必须等于记录值。真机数据是**测量值**
     （167，2026-07-25，`peak_alloc_MiB` per stage），任何"顺手改数据让模型好看"都会在这里炸。
  B. **派生 = 标签**：八跑从两份 base yaml 程序化派生（只改 5 个字段），派生结果逐项反查
     标签语义（fused 位 / 层数 / 微批数 / 重算域 / compress_ratios / pp·dp·ep·tp·cp / seq /
     每 stage 隐藏层数）。手抄八份 yaml 无法自证一致，派生 + 反查可以。
  C. **两条真机不变量**（本项目的核心物理发现），既校验真机表自身、也作为**任何 graph source
     都必须满足的结构断言**：
       ×1 于层数    stage3 `unfused − fused` 在 L8(2 层/stage) 与 L4(1 层/stage) 逐 0.1 MiB 相同
                    （真机 24380.1 / 24380.1）；
       ×1 于微批数  a↔e 与 b↔f 把该 delta 移动 ≤0.4 MiB（真机 24380.1 → 24380.0）。
     ⚠ 断言的是**结构**（delta 在层数/微批数上不动），**不**断言幅值 —— 今天两条来源都欠读这个
     delta（bucket 0.603×、liveness·chain2 0.670×），如实记在 `test_delta_magnitude_gap_is_recorded`
     里而不是藏进容差。
  D. **逐字节中立**：桶模型与 liveness 的 8×4 峰值全部锁成 golden（0.05 MiB 容差 = 打印精度），
     聚合均值锁到 3 位小数。加验收台不许动模型一个字节。
"""
from __future__ import annotations

import os
import warnings

import pytest

from tools.liveness_ab_validate import (BY_TAG, DEFAULT_BASE_DIR,
                                        INVARIANT_STAGE, PAIRS, RATIOS, REAL,
                                        REAL_SHA256, TOL_LAYERS_MIB,
                                        TOL_MICROBATCH_MIB, VARIANTS,
                                        aggregate, build_bundle, check_derivation,
                                        derive_mf_config, magnitude_report, main,
                                        model_invariants, real_fingerprint,
                                        real_invariants, run_matrix,
                                        unscorable_cells)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DEFAULT_BASE_DIR),
    reason="缺 167 A/B base yaml 目录：%s" % DEFAULT_BASE_DIR)


@pytest.fixture(scope="module")
def matrix():
    """八跑 × {桶模型, liveness(hand_spec)} × 两个 grad_mode，跑一次全模块复用。"""
    warnings.simplefilter("ignore")
    return {gm: run_matrix(DEFAULT_BASE_DIR, ["hand_spec"], gm)
            for gm in ("dataflow", "chain2")}


# ---------------------------------------------------------------------------
# A. 真机地面真值不可漂移
# ---------------------------------------------------------------------------

def test_real_table_fingerprint_is_pinned():
    """`REAL` 表是**测量值**：改它必须显式改指纹并说明来源。"""
    assert real_fingerprint() == REAL_SHA256, (
        "真机地面真值被改动了。167/2026-07-25 的 peak_alloc_MiB 是测量值，"
        "不得为了让模型好看而编辑；确有新测量则更新 REAL_SHA256 并注明日志路径。")


def test_real_table_shape_and_oom_run():
    """8 跑 × 4 stage；run d 全 OOM（真机崩了）→ 不可评分，必须显式为 None。"""
    assert len(REAL) == 8 and {len(v) for v in REAL.values()} == {4}
    assert all(v is None for v in REAL["d unfused OFF L8 m4"].values())
    assert all(all(v is not None for v in REAL[t].values())
               for t in REAL if not t.startswith("d "))


# ---------------------------------------------------------------------------
# B. 派生 = 标签
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", [v.tag for v in VARIANTS])
def test_derivation_matches_label(tag):
    """程序化派生的配置逐项等于标签语义（12 项反查，见 `check_derivation`）。"""
    warnings.simplefilter("ignore")
    v = BY_TAG[tag]
    mf = derive_mf_config(DEFAULT_BASE_DIR, v)
    b, spec = build_bundle(mf)
    bad = check_derivation(mf, b, spec, v)
    assert not bad, "派生反查失败：\n  " + "\n  ".join(bad)


def test_only_five_fields_differ_across_the_eight_runs():
    """八跑之间**只**差 5 个字段（在 167 上 grep 核对过的那 5 个）。"""
    import json
    base = derive_mf_config(DEFAULT_BASE_DIR, BY_TAG["a fused   ON  L8 m4"])
    allowed = {("training", "global_batch_size"), ("model", "num_hidden_layers"),
               ("model", "compress_ratios"), ("model", "apply_dsa_kernel_fusion"),
               ("recompute",)}
    for v in VARIANTS:
        mf = derive_mf_config(DEFAULT_BASE_DIR, v)
        assert set(mf) == set(base), f"{v.tag}: 顶层键集变了"
        for k in base:
            if json.dumps(mf[k], sort_keys=True, default=str) == \
                    json.dumps(base[k], sort_keys=True, default=str):
                continue
            if (k,) in allowed:
                continue
            diff = {sk for sk in set(mf[k]) | set(base[k])
                    if mf[k].get(sk) != base[k].get(sk)}
            extra = {sk for sk in diff if (k, sk) not in allowed}
            assert not extra, f"{v.tag}: {k} 里出现了未授权的差异字段 {extra}"


def test_compress_ratios_length_equals_layer_count():
    """`compress_ratios` 长度必须等于 `num_hidden_layers`（真机 launcher 的硬约束）。"""
    for nl, ratios in RATIOS.items():
        assert len(ratios) == nl
    for v in VARIANTS:
        assert len(RATIOS[v.n_layers]) == v.n_layers


# ---------------------------------------------------------------------------
# C. 两条真机不变量
# ---------------------------------------------------------------------------

def test_real_machine_invariants_hold():
    """真机自身的 ×1 于层数 / ×1 于微批数（附指纹守卫）。"""
    checks = real_invariants()
    failed = [c for c in checks if not c.passed]
    assert not failed, "\n".join(str(c) for c in failed)


def test_real_layer_count_invariant_exact_numbers():
    """把真机数字明写出来：stage3 delta 在 L8 与 L4 均为 24380.1 MiB（逐 0.1 相同）。"""
    d_l8 = REAL["b unfused ON  L8 m4"][3] - REAL["a fused   ON  L8 m4"][3]
    d_l4 = REAL["h unfused ON  L4 m4"][3] - REAL["g fused   ON  L4 m4"][3]
    assert round(d_l8, 1) == 24380.1 and round(d_l4, 1) == 24380.1
    assert abs(d_l8 - d_l4) <= TOL_LAYERS_MIB


def test_real_microbatch_invariant_exact_numbers():
    """a↔e / b↔f：stage3 delta 24380.1 → 24380.0，移动 ≤0.4 MiB。"""
    d_m4 = REAL["b unfused ON  L8 m4"][3] - REAL["a fused   ON  L8 m4"][3]
    d_m8 = REAL["f unfused ON  L8 m8"][3] - REAL["e fused   ON  L8 m8"][3]
    assert round(d_m8, 1) == 24380.0
    assert abs(d_m4 - d_m8) <= TOL_MICROBATCH_MIB


@pytest.mark.parametrize("grad_mode", ["dataflow", "chain2"])
@pytest.mark.parametrize("key", ["bucket", "hand_spec"])
def test_model_satisfies_the_same_x1_invariants(matrix, grad_mode, key):
    """**任何 graph source 都必须满足**同样的 ×1 结构 —— 正确的模型是**导出**它，不是被公式告知。

    抽取来源（`extracted`）落地后走同一条参数化，不需要新写断言。"""
    failed = [c for c in model_invariants(matrix[grad_mode], key) if not c.passed]
    assert not failed, "\n".join(str(c) for c in failed)


def test_delta_magnitude_gap_is_recorded(matrix):
    """**如实记账**：×1 结构成立，但 delta 的**幅值**两条来源今天都欠读，不许被容差掩盖。

    这不是"允许误差"，是把已知缺口写在测试里，改好了这个测试会红（提醒更新记录）。"""
    rows = magnitude_report(matrix["chain2"], ["bucket", "hand_spec"])
    ratios = {(lbl, key): ratio for lbl, key, _rd, _md, ratio in rows}
    for lbl in ("L8 m4", "L8 m8", "L4 m4"):
        # 2026-07-29 二次重钉：0.545→0.518（bucket）/ 0.644→0.636（hand_spec）。
        # **缺口继续变大仍是如实记账**：融合 mHC ctx + RMSNorm 不 cast 两条修正**只减 fused 侧**
        # （unfused 跑同样吃 RMSNorm 一条，但其欠读大头 —— gathered-KV fp32 三份梯度，拆解报告
        # §4.2 —— 一分未修）→ `unfused − fused` 的 delta 相应更欠。
        # 2026-07-29 三次重钉：0.518→0.517（bucket）/ 0.636→0.633（hand_spec）。同一原因：
        # mHC 残差承载归位的净额（MoE 层 −32/层）落在 fused/unfused **两侧同幅**，stage3 峰
        # 位于 head 段、差值几乎不动；欠读大头依旧未修。
        # 2026-07-30 四次重钉（**r4 indexer 内部 RoPE 保留对入账**，
        # `docs/r4_indexer_census_2026-07-30.md`）：0.517→**0.522**（bucket）/
        # 0.633→**0.638**（hand_spec）。**方向与前三次相反、缺口收窄**：本项在
        # `forward_before_topk` 内部，fused 与 unfused **两条路径逐字共用**（`csa.py:667` /
        # `csa.py:766`）→ 两侧同幅 +128.0 MiB/r4 层；stage3 的 fused 侧峰在 head 段（`bwd@9`
        # 不含解码层 saves，逐字节不动），unfused 侧峰在 `bwd@8` 的 sparse_attn（吃满 +128.0）
        # → delta 净 **+128.0**（12594.4→12722.4 / 15421.8→15549.8）。**未调参**：分子变大是因为
        # unfused 侧真实多了一块，不是把容差放宽。
        # 2026-07-30 五次重钉（`lm_head` 反向 kernel workspace 实测入账，docs/head_loss_bwd_workspace_2026-07-30.md）：
        #   bucket 0.522 → **0.477**（hand_spec **0.638 逐字节不动**）。
        #   **方向与上一次相反、缺口变大，如实记**：本项只抬 fused 侧 stage3（其峰在 head 段
        #   `bwd@9`，吃满 +1094.0），unfused 侧 stage3 的峰在 `bwd@8:sparse_attn`（解码层）
        #   → 一分未动 ⇒ `unfused − fused` 净 **−1094.0**（12722.4→11628.4）。
        #   分母（真机 delta 24380.1）一个字节没动；**未调参**。
        assert round(ratios[(lbl, "bucket")], 3) == 0.477, ratios
        assert round(ratios[(lbl, "hand_spec")], 3) == 0.638, ratios


# ---------------------------------------------------------------------------
# D. 逐字节中立（golden 锁）
# ---------------------------------------------------------------------------

#: 桶模型 per-stage 峰值（MiB）——与 grad_mode 无关。
#: **2026-07-29 重钉**（`docs/census_arbitration_2026-07-29.md` §2）：手写普查的注意力主干按
#: 权威快照逐条订正（q_hnorm/cg 由 fp32 改回 bf16、去伪 norm 抬升、去 inv_rope_out、补前向 rope
#: 保留对、cmp_residual 改标量、补 sinks/sparse_indices、idx_weights 改 fp32）。**不变量一条没动**
#: （×1 于层数/微批数、run d 不可评分、真机指纹），只移动了样例值。
#: **2026-07-29 二次重钉**（`docs/census_fix_mhc_rmsnorm_2026-07-29.md`）：①融合 mHC 按
#: `custom_op_impl.py:390-391` / `mhc_pre_sinkhorn.cc:24-50` 建 ctx（−421.8 MiB/层）；
#: ②`FusedRMSNorm` 不 cast（`layer_norm.py:151-155`）→ norm-fp32 抬升按种类分辨（−268.0 MiB/层）。
#: 同样**一条不变量没动**，只移动样例值。
#: **2026-07-29 三次重钉**（`docs/census_fix_residual_carrier_2026-07-29.md`）：mHC 残差承载判定
#: 归位 —— ①段首 layernorm 保留 aggregated `[S,B,H]` 而非打包流（`transformer_layer.py:308-311,
#: 326-329`），打包流改由融合 ctx 的 `x` 声明（`custom_op_impl.py:390`）→ **+64.0 MiB/层**；
#: ②MoE 的 `comb` 不再被误当打包流放大（`_stream_names` 按数据流位置判定）→ **−96.0 MiB/MoE 层**。
#: 净 −32.0 MiB/MoE 层、+64.0 MiB/dense 层。八跑全部 `use_fused_mhc: true` → ④（非融合 fp32
#: 副本）**不触及本表**。同样一条不变量没动，只移动样例值。
#: **2026-07-30 四次重钉**（`docs/r4_indexer_census_2026-07-30.md`）：`CSAIndexer` 内部那次
#: `ApplyRotaryPosEmb`（`indexer.py:182-187`）的反向保留对 `t`/`t_rot` 入账 → **每个 r4 层
#: `activation_saves` +128.000 MiB**（seq4096 下各 64.000 MiB fp32）。r0/r128 层**逐字节不动**
#: （`enable_indexer` 只在 `compress_ratio==4` 为真，`csa.py:608`；已按模型输出核实）。
#: 与前几轮不同，本项在 `forward_before_topk` 内部，**fused / unfused 两条路径逐字共用**
#: （`csa.py:667` / `csa.py:766`）→ 八跑**全部**上移，不是只动 fused 四跑。
#: 一条不变量没动（×1 于层数/微批数、run d 不可评分、真机指纹），只移动样例值。
GOLDEN_BUCKET = {
    # 2026-07-29 重钉：**fused 跑**的每个 r4 层 bwd 事件加上真机实测的融合稀疏 flash-MLA
    # 反向 kernel workspace（seq4096 → **730.0 MiB**，三点验证的实测律，
    # docs/kernel_workspace_2026-07-29.md）。unfused 四跑（b/d/f/h）**逐字节不变**——
    # 该 kernel 只在 fused 分支存在，实测律不外推到小算子链。
    # 2026-07-30 四次重钉的逐跑幅度 = 该 stage 峰值事件在世的 r4 层数 ×128.0：
    #   a/e s1,s2 +128.0（各 1 层）、s3 +0.0（峰在 head 段 `bwd@9`）；a s0 +100.0 / e s0 +128.0
    #   （a s0 的**峰值事件由 `bwd@1` 换成 `bwd@2`** —— r4 层那一格实涨 128.0 后越过了原本更高
    #   的 r0 层那一格，故差额只有 100.0，不是本项只算了一部分）；
    #   b/f 四格各 +128.0；c s0-s2 +512.0/+384.0/+256.0（无重算 = 在途微批深度 4/3/2 各 ×128）、
    #   s3 +128.0；d 同 c 的深度律；g s1 +128.0、其余 0.0；h s1/s3 +128.0、s0/s2 0.0。
    # ── 2026-07-30 五次重钉（`lm_head` 反向 kernel workspace 实测入账，docs/head_loss_bwd_workspace_2026-07-30.md）──
    #   只有**峰值事件落在 head 段**（`bwd@9` / `bwd@5`，`peak@hand_spec` 列写作
    #   `bwd@N/bwd4:nll`）的那些格会动，且动的幅度**恰好是实测律在本站点的值 +1094.0 MiB**
    #   （H=4096 / S=4096 / B=1 / vocab=129280 → `(2*129280+4*4096)*4096 + 20 MiB + 1024`）：
    #     a s3 22212.8→23306.8 ／ c s3 24467.6→25561.7 ／ d s3 49383.6→50477.6
    #     ／ e s3 22212.8→23306.8 ／ g s3 18911.9→20005.9
    #   **b/f/h 的 s3 逐字节不变**——它们的峰在 `bwd@8`/`bwd@4` 的 `sparse_attn`（解码层），
    #   不在 head 段；**全部 s0/s1/s2 逐字节不变**（同理）。
    #   注意 c s3 由 24467.6 → 25561.7 = +1094.1（0.1 是显示舍入，非第二个数字介入）。
    "a fused   ON  L8 m4": (17378.6, 12759.1, 12503.1, 23306.8),
    "b unfused ON  L8 m4": (35521.6, 30902.1, 30646.1, 34935.1),
    "c fused   OFF L8 m4": (28759.6, 20425.6, 16211.4, 25561.7),
    "d unfused OFF L8 m4": (124493.6, 94443.6, 65313.4, 50477.6),
    "e fused   ON  L8 m8": (18644.6, 12759.1, 12503.1, 23306.8),
    "f unfused ON  L8 m8": (36787.6, 30902.1, 30646.1, 34935.1),
    "g fused   ON  L4 m4": (14195.6, 9202.2, 7942.6, 20005.9),
    "h unfused ON  L4 m4": (19438.6, 27345.2, 13985.6, 31634.2),
}
#: liveness(hand_spec) per-stage 峰值（MiB），grad_mode=dataflow。（同上，2026-07-29 重钉）
#: **2026-07-29 三次重钉·补丁**：`mhc_wrap` 的 OpSpec 重建改用 `dataclasses.replace`，修好一处
#: **静默丢字段** —— 此前它手写字段清单，把被包装层的 `workspace_ref`（flash-attn softmax-LSE
#: 工作区，`attention.py:225,313` / `dsv4_hybrid.py:331,358,381` 的 `_fa_workspace()`）整个丢掉。
#: 只有 DSA-unfused 的四跑（b/d/f/h）的 dataflow 峰落在带该 workspace 的事件上 → **+16.0 MiB**；
#: chain2 与 bucket 逐字节不变。同一通道上一轮丢过 `norm_kind`，现已在结构上不可能再丢。
#: **2026-07-30 四次重钉**：同 `GOLDEN_BUCKET` 的 +128.0/r4 层。⚠ 与上一轮（bwd kernel
#: workspace）**不同**：那一项只走桶模型的 `bwd_workspace_bytes` 通道，故 hand_spec 两表当时
#: 逐字节不动；本项改的是 spec 自己的 `saves`，`cost_eval/liveness/` 按 saves 建图 → 两条来源
#: **同时**移动。逐跑幅度与 bucket 表一致，唯一例外是 **a/g/h s0 各 +0.0**（其 hand_spec 峰值
#: 事件是 `bwd@1`/`bwd21:swiglu`、不落在 r4 层上；bucket 侧 a s0 则因换事件得 +100.0）。
GOLDEN_LIVENESS_DATAFLOW = {
    "a fused   ON  L8 m4": (16574.4, 11212.6, 10956.6, 24232.8),
    "b unfused ON  L8 m4": (33719.8, 29100.3, 28844.3, 33133.3),
    "c fused   OFF L8 m4": (27557.6, 19223.6, 15009.4, 26487.6),
    "d unfused OFF L8 m4": (116143.7, 88013.7, 60931.5, 49355.6),
    "e fused   ON  L8 m8": (17098.1, 11212.6, 10956.6, 24232.8),
    "f unfused ON  L8 m8": (34985.8, 29100.3, 28844.3, 33133.3),
    "g fused   ON  L4 m4": (13491.4, 7655.7, 7142.1, 20931.9),
    "h unfused ON  L4 m4": (18734.4, 25543.4, 13185.1, 29832.4),
}
#: 同上，grad_mode=chain2。（同上，2026-07-29 重钉 / 2026-07-30 四次重钉）
GOLDEN_LIVENESS_CHAIN2 = {
    "a fused   ON  L8 m4": (16574.4, 11212.6, 10956.6, 24232.8),
    "b unfused ON  L8 m4": (40369.1, 35749.5, 35493.5, 39782.6),
    "c fused   OFF L8 m4": (27592.3, 19258.3, 15044.0, 26487.6),
    "d unfused OFF L8 m4": (123621.3, 95619.2, 68537.0, 49355.6),
    "e fused   ON  L8 m8": (17098.1, 11212.6, 10956.6, 24232.8),
    "f unfused ON  L8 m8": (41635.1, 35749.5, 35493.5, 39782.6),
    "g fused   ON  L4 m4": (13491.4, 7655.7, 7142.1, 20931.9),
    "h unfused ON  L4 m4": (19973.1, 32192.6, 15256.0, 36481.6),
}
#: 28 个可评分格的 `sim/real` 聚合（run d 真机 OOM → 不入统计）。
#: **2026-07-29**：mean 0.946→0.867（bucket）/ 0.955→0.899（hand_spec·chain2）。max 从 1.394/1.400
#: 降到 1.125/1.143 —— 那正是 run c 的**过读**被修掉；均值下降是因为过读此前在掩盖别处的欠读
#: （unfused 反向工作集、fused 单层成本），两者本是相反方向的误差（拆解报告 §0）。
#: **二次重钉**：0.867→0.822（bucket）/ 0.899→0.862（hand_spec·chain2）；max 1.125→1.019 /
#: 1.143→1.116。run c 逐 stage 由 1.083/1.125/1.020/0.931 收到 0.901/0.928/0.865/0.880 ——
#: **过读已被完全消掉、转入欠侧**。均值继续下降同上：抵消消失、欠读露出，不为均值好看而留错字节。
#: **三次重钉**：0.822→0.821（bucket）/ 0.862→0.855（hand_spec·chain2）；max 1.019→1.017 /
#: 1.116→1.110。run c 逐 stage 0.901/0.928/0.865/0.880 → **0.905/0.919/0.857/0.878**
#: （s0 含 dense r0 层 → +64/层使其**回升**；其余 stage 全是 MoE 层 → −32/层继续下探）。
GOLDEN_AGG = {
    # 2026-07-29 重钉（bucket 两行）：加入真机实测的 bwd kernel workspace 后，桶模型均值
    # 0.821 → **0.836**（欠读被真实地补上一块），max 1.017 → **1.023**（`g` stage1 由
    # 0.941 转为 1.023 = 轻度过读，OOM 安全侧，如实记）。min 0.700 未动。
    # **hand_spec 两行逐字节不变**：`cost_eval/liveness/` 不读 `bwd_workspace_bytes`
    # （它按 saves + grad 可达性自建图），故该来源与本次改动正交 —— 这也是本次改动
    # 只动 bucket 一条口径的证据。
    # 2026-07-30 四次重钉（r4 indexer 内部 RoPE 保留对入账，`docs/r4_indexer_census_2026-07-30.md`）：
    #   bucket    0.836 → **0.841**（min 0.700→0.703，max 1.023→**1.038** = `g` s1 由 1.023 再上移）；
    #   hand_spec 0.855 → **0.860**（chain2；min 0.670→0.675，max **1.110 逐字节不动** = `h` s2，
    #             该格峰在 `bwd@12:sparse_attn` 且其所在 stage 无 r4 层 → 本项够不着）。
    #   dataflow·hand_spec 0.795 → **0.799**（min 0.659→0.662，max 1.067 不动）。
    #   28 个可评分格里 **0 格下移**：bucket 21 上移 / 7 不动，hand_spec(chain2) 20 上移 / 8 不动。
    #   欠读被真实补上一块，没有一格由欠读翻成过读（`g` s1 / `g` s2 / `h` s2 本来就在过读侧）。
    #   **未调参**。
    # 2026-07-30 五次重钉（`lm_head` 反向 kernel workspace 实测入账）：
    #   bucket 0.841 → **0.847**（5 格上移各 +1094.0；min 0.703 / max 1.038 **逐字节不动**
    #   —— 上移的都不是极值格）。hand_spec 三行**逐字节不变**：`cost_eval/liveness/`
    #   按 saves + grad 可达性自建图、不读 `bwd_workspace_bytes`，这也是本次只动一条口径的证据。
    ("dataflow", "bucket"): (28, 0.847, 0.703, 1.038),
    ("dataflow", "hand_spec"): (28, 0.799, 0.662, 1.067),
    ("chain2", "bucket"): (28, 0.847, 0.703, 1.038),
    ("chain2", "hand_spec"): (28, 0.860, 0.675, 1.110),
}


@pytest.mark.parametrize("grad_mode", ["dataflow", "chain2"])
def test_bucket_peaks_byte_neutral(matrix, grad_mode):
    """桶模型（`Evaluator`）的 8×4 峰值 == golden：验收台没有动权威口径一个字节。"""
    for v in VARIANTS:
        got = matrix[grad_mode][v.tag].bucket_mib
        want = GOLDEN_BUCKET[v.tag]
        assert len(got) == len(want)
        for i, (a, b) in enumerate(zip(got, want)):
            assert abs(a - b) <= 0.05, f"{v.tag} stage{i}: bucket {a:.1f} != golden {b:.1f}"


@pytest.mark.parametrize("grad_mode,golden", [("dataflow", GOLDEN_LIVENESS_DATAFLOW),
                                              ("chain2", GOLDEN_LIVENESS_CHAIN2)])
def test_liveness_peaks_byte_neutral(matrix, grad_mode, golden):
    """liveness(hand_spec) 的 8×4 峰值 == golden（两个 grad_mode 各一套）。"""
    for v in VARIANTS:
        got = matrix[grad_mode][v.tag].liveness_mib["hand_spec"]
        want = golden[v.tag]
        for i, (a, b) in enumerate(zip(got, want)):
            assert abs(a - b) <= 0.05, f"{v.tag} stage{i}: liveness {a:.1f} != golden {b:.1f}"


@pytest.mark.parametrize("grad_mode", ["dataflow", "chain2"])
@pytest.mark.parametrize("key", ["bucket", "hand_spec"])
def test_aggregate_regression(matrix, grad_mode, key):
    """聚合 `sim/real`（n / mean / min / max）锁到 3 位小数。"""
    a = aggregate(matrix[grad_mode], key)
    n, mean, lo, hi = GOLDEN_AGG[(grad_mode, key)]
    assert a["n"] == n
    assert (round(a["mean"], 3), round(a["min"], 3), round(a["max"], 3)) == (mean, lo, hi)


def test_run_d_is_unscorable_and_excluded(matrix):
    """run d 真机 OOM → 4 格不可评分，**不入**任何统计（模型值仅备查）。"""
    uns = unscorable_cells(matrix["chain2"])
    assert sorted(uns) == [("d unfused OFF L8 m4", i) for i in range(4)]
    total_cells = 8 * 4
    assert aggregate(matrix["chain2"], "hand_spec")["n"] == total_cells - len(uns) == 28


def test_unfused_on_cell_ratios_match_recorded_reference(matrix):
    """回归参照：unfused ON 的 liveness·chain2 sim/real ≈ 0.804 / 0.814 / 0.818 / 0.831；
    bucket ≈ 0.703–0.730。（2026-07-29 二次重钉，原 0.808/0.818/0.822/0.835 与 0.717–0.742；
    RMSNorm 不 cast 一条也作用在 unfused 跑上 → 该跑比值再下移一档。
    2026-07-29 三次重钉，原 0.804/0.813/0.817/0.830 与 0.701–0.728：mHC 残差承载归位在
    MoE 层净 −32/层；该跑八条 yaml 同样 `use_fused_mhc: true`，故只吃 ②③、不吃 ④。
    2026-07-30 四次重钉，原 0.802/0.811/0.815/0.828 与 0.700–0.727：r4 indexer 内部 RoPE
    保留对入账（`docs/r4_indexer_census_2026-07-30.md`）。**这一轮 unfused 跑也吃**——本项在
    `CSAIndexer.forward_before_topk` 内部，融合/小算子两条路径逐字共用（`csa.py:667` /
    `csa.py:766`），与 `apply_dsa_kernel_fusion` 无关；四个 stage 各 +128.0 MiB（每 stage
    恰好 1 个 r4 层在峰值事件上）。**未调参**：比值上移是 unfused 侧真实多了一块。）"""
    r = matrix["chain2"]["b unfused ON  L8 m4"]
    lv = [r.liveness_mib["hand_spec"][i] / r.real(i) for i in range(4)]
    bk = [r.bucket_mib[i] / r.real(i) for i in range(4)]
    assert [round(x, 3) for x in lv] == [0.804, 0.814, 0.818, 0.831]
    assert min(bk) >= 0.703 - 5e-4 and max(bk) <= 0.730 + 5e-4


# ---------------------------------------------------------------------------
# 门本身
# ---------------------------------------------------------------------------

def test_gate_cli_returns_zero(capsys):
    """`--gate` 端到端过：派生反查 0 违规、真机+模型不变量全过 → exit 0。"""
    warnings.simplefilter("ignore")
    rc = main(["--grad-mode", "chain2", "--deltas", "--gate"])
    out = capsys.readouterr().out
    assert rc == 0, out[-3000:]
    assert "结论: PASS" in out
    assert "OOM" in out, "run d 必须以 OOM 显示，不得填一个编造的数"


def test_gate_reports_live_set_dump(capsys):
    """`--dump-top` 必须给出峰值时刻**逐张量** live-set（哪一个字节是谁）。"""
    warnings.simplefilter("ignore")
    rc = main(["--grad-mode", "chain2", "--dump-top", "5"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "峰值时刻逐张量 live-set" in out and "category" in out
    assert "recomp_saved" in out or "act_saved" in out or "fwd_transient" in out


def test_gate_lists_sources_and_bucket_side_by_side(capsys):
    """每一跑都并排给出 real / bucket / liveness(每个可用来源)。"""
    warnings.simplefilter("ignore")
    main(["--grad-mode", "dataflow"])
    out = capsys.readouterr().out
    assert "real" in out and "bucket" in out and "lv:hand_sp" in out
    for v in VARIANTS:
        assert v.tag in out


def test_gate_rejects_unknown_source(capsys):
    assert main(["--sources", "no_such_source"]) == 2
    assert "未知 graph source" in capsys.readouterr().out
