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
        assert round(ratios[(lbl, "bucket")], 3) == 0.603, ratios
        assert round(ratios[(lbl, "hand_spec")], 3) == 0.670, ratios


# ---------------------------------------------------------------------------
# D. 逐字节中立（golden 锁）
# ---------------------------------------------------------------------------

#: 桶模型 per-stage 峰值（MiB）——与 grad_mode 无关。改造前后逐字节相同。
GOLDEN_BUCKET = {
    "a fused   ON  L8 m4": (19307.1, 14033.1, 13777.1, 22241.8),
    "b unfused ON  L8 m4": (37518.1, 32898.6, 32642.6, 36931.6),
    "c fused   OFF L8 m4": (40342.2, 29293.3, 21964.1, 27611.6),
    "d unfused OFF L8 m4": (136776.2, 104006.8, 71773.1, 52516.1),
    "e fused   ON  L8 m8": (19918.6, 14033.1, 13777.1, 22241.8),
    "f unfused ON  L8 m8": (38784.1, 32898.6, 32642.6, 36931.6),
    "g fused   ON  L4 m4": (16225.6, 10477.7, 10072.6, 18942.4),
    "h unfused ON  L4 m4": (21468.6, 29343.2, 16111.6, 33632.2),
}
#: liveness(hand_spec) per-stage 峰值（MiB），grad_mode=dataflow。
GOLDEN_LIVENESS_DATAFLOW = {
    "a fused   ON  L8 m4": (17963.0, 12592.9, 12336.9, 24261.8),
    "b unfused ON  L8 m4": (34145.7, 29526.1, 29270.1, 33559.2),
    "c fused   OFF L8 m4": (40734.2, 29429.3, 21844.1, 29887.6),
    "d unfused OFF L8 m4": (128976.2, 97998.8, 67557.1, 52744.1),
    "e fused   ON  L8 m8": (18478.5, 12592.9, 12336.9, 24261.8),
    "f unfused ON  L8 m8": (35411.7, 29526.1, 29270.1, 33559.2),
    "g fused   ON  L4 m4": (14881.5, 9037.5, 8648.4, 20962.4),
    "h unfused ON  L4 m4": (20124.5, 25970.7, 14687.4, 30259.8),
}
#: 同上，grad_mode=chain2。
GOLDEN_LIVENESS_CHAIN2 = {
    "a fused   ON  L8 m4": (17963.0, 12592.9, 12336.9, 24261.8),
    "b unfused ON  L8 m4": (41194.8, 36575.3, 36319.3, 40608.3),
    "c fused   OFF L8 m4": (40734.2, 29429.3, 21844.1, 29887.6),
    "d unfused OFF L8 m4": (136141.0, 105163.6, 74721.9, 52744.1),
    "e fused   ON  L8 m8": (18478.5, 12592.9, 12336.9, 24261.8),
    "f unfused ON  L8 m8": (42460.8, 36575.3, 36319.3, 40608.3),
    "g fused   ON  L4 m4": (14881.5, 9037.5, 8648.4, 20962.4),
    "h unfused ON  L4 m4": (20928.4, 33019.8, 16211.3, 37308.9),
}
#: 28 个可评分格的 `sim/real` 聚合（run d 真机 OOM → 不入统计）。
GOLDEN_AGG = {
    ("dataflow", "bucket"): (28, 0.946, 0.748, 1.394),
    ("dataflow", "hand_spec"): (28, 0.893, 0.672, 1.400),
    ("chain2", "bucket"): (28, 0.946, 0.748, 1.394),
    ("chain2", "hand_spec"): (28, 0.955, 0.729, 1.400),
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
    """回归参照（任务交接记录的已知结果）：unfused ON 的 liveness·chain2 sim/real
    ≈ 0.821 / 0.832 / 0.837 / 0.848；bucket ≈ 0.748–0.771。"""
    r = matrix["chain2"]["b unfused ON  L8 m4"]
    lv = [r.liveness_mib["hand_spec"][i] / r.real(i) for i in range(4)]
    bk = [r.bucket_mib[i] / r.real(i) for i in range(4)]
    assert [round(x, 3) for x in lv] == [0.821, 0.832, 0.837, 0.848]
    assert min(bk) >= 0.748 - 5e-4 and max(bk) <= 0.771 + 5e-4


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
