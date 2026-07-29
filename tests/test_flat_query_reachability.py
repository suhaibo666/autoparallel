# -*- coding: utf-8 -*-
"""**扁平 query 可达性门**（2026-07-30，`docs/fused_mhc_branch_mismatch_2026-07-30.md` §8）。

## 为什么既有的 round-trip 不变量拦不住这类 bug

`_assert_bundle_roundtrip`（2026-07-25 那轮，`test_yaml_roundtrip_fidelity.py`）问的是：
**yaml 导入路**上，fields 经 `parse_and_validate` 能不能复现 bundle 的 `LLMConfig`？
它用 `_llm_field_diffs` 逐字段比，**确实覆盖了包括 `use_fused_mhc` 在内的每一个字段**。

但那一轮的修法是把**权威 `LLMConfig` 整体**塞进隐藏字段 `llm_json`（`_llm_to_fields`），
`parse_and_validate` 见到 `llm_json` 就以它为基座。于是：

> 任何新增字段都**自动**随 `llm_json` 过桥 → 该判据对「这个字段在扁平 dict 里有没有承载」
> **结构上永远绿**。判据不是漏判，是**问错了问题**。

而**另一条路**没人守：锚点 / 探针 / 手配 query（`tests/test_pp4_recompute_anchor.py`、
`tests/test_probe185_recon.py`、web 页面手配）**没有 `llm_json`**，基座是**预设**，凡不在
`_LLM_FIELD_GATE` 里的字段一律**静默取预设值**。`use_fused_mhc` 就是这么让 ~20 个锚点
拿**非融合** mHC 去对**融合** mHC 的真机（896 MiB/层），持续两轮无人察觉。

## 本门守什么

① **分类完备**：每个 `LLMConfig` 字段必须**要么**有 UI 门控键（`_LLM_FIELD_GATE`）、
   **要么**显式声明为 `llm_json` 专属（`_LLM_JSON_ONLY_FIELDS`）。新增字段两边都不登记 → 红。
   这正是当初该逼出决定、却没有任何东西逼的那一步。
② **两表互斥**：一个字段不能同时在两表里（否则"改没改"的判定含义不明）。
③ **接线为真**：`_LLM_FIELD_GATE` 里的每个字段，在**无 `llm_json` 的扁平 query** 上翻动它的
   UI 键，必须真的改变解析出的 `LLMConfig` 字段 —— 光"登记了"不算，得真的接上。
④ **本次 bug 的定点回归**：`mhc_fused` 1/0 必须切到 `_fused_hc_ops` / `_unfused_hc_ops`
   两条**不同**的 op 链；缺省/空则保留基座（手配路径逐字节不变）。
"""
import dataclasses
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serve_explorer as S
from cost_eval.build_llm import build_llm_spec
from cost_eval.llm_config import LLMConfig


# ── ① 分类完备 / ② 两表互斥 ──────────────────────────────────────────────────────────
def test_every_llmconfig_field_is_classified():
    """新增 `LLMConfig` 字段必须当场决定它在**扁平 query 路**上怎么办。

    两个选项：给它一个 UI 键（进 `_LLM_FIELD_GATE`，锚点/手配可显式指定），或者承认
    「扁平路上它恒取预设值」（进 `_LLM_JSON_ONLY_FIELDS`，并在那里写清代价）。
    不做决定就静默取预设值 —— 这正是 `use_fused_mhc` 错配 ~20 个锚点的成因。
    """
    unclassified = S.unclassified_llm_fields()
    assert not unclassified, (
        f"这些 LLMConfig 字段既没有 UI 门控键、也没声明为 llm_json 专属：{unclassified}。\n"
        f"扁平 query（锚点/探针/页面手配，无 llm_json）会让它们**静默取预设值** —— "
        f"若它影响内存且真机侧另有取值，锚点就会对着错的模型打分。\n"
        f"请二选一：① 加 UI/隐藏字段 + `_LLM_FIELD_GATE` 登记 + `_llm_to_fields` 回填"
        f"（照 `mhc_fused`/`dsa_fused`/`ce_fused` 的样子）；"
        f"② 加进 `_LLM_JSON_ONLY_FIELDS` 并写明「扁平路上恒取预设值」的代价。")


def test_gate_and_json_only_registries_are_disjoint():
    both = sorted(set(S._LLM_FIELD_GATE) & S._LLM_JSON_ONLY_FIELDS)
    assert not both, f"字段同时登记在两表里，'改没改'的判定含义不明：{both}"


def test_classification_gate_actually_fires(monkeypatch):
    """**证明本门会响**：把 `use_fused_mhc` 从登记表里拿掉（= 复现 2026-07-29 的真实状态），
    `unclassified_llm_fields()` 必须点名它。不会响的门等于没有门。"""
    gate = {k: v for k, v in S._LLM_FIELD_GATE.items() if k != "use_fused_mhc"}
    monkeypatch.setattr(S, "_LLM_FIELD_GATE", gate)
    monkeypatch.setattr(S, "_LLM_JSON_ONLY_FIELDS",
                        S._LLM_JSON_ONLY_FIELDS - {"use_fused_mhc"})
    assert S.unclassified_llm_fields() == ["use_fused_mhc"]


def test_registries_reference_real_llmconfig_fields():
    """两表里不许有 `LLMConfig` 上不存在的字段（改名/删字段后残留 → 判据悄悄失效）。"""
    names = {f.name for f in dataclasses.fields(LLMConfig)}
    stale = sorted((set(S._LLM_FIELD_GATE) | set(S._LLM_JSON_ONLY_FIELDS)) - names)
    assert not stale, f"登记表引用了 LLMConfig 上不存在的字段（改名/删除后残留）：{stale}"


# ── ③ 接线为真：无 llm_json 的扁平 query 上逐字段翻动 ─────────────────────────────────
#: 扁平基座（**刻意不带 `llm_json`** —— 这就是锚点/探针/页面手配走的那条路）。
_FLAT_BASE = {
    "preset": "custom", "attn": "gqa", "layers": "8", "seq": "4096", "batch": "1",
    "heads": "8", "kv_groups": "8", "dense_k": "1", "experts": "8", "topk": "4",
    "hidden": "1792", "ffn": "3072", "moe_ffn": "1024", "q_lora": "1536",
    "kv_lora": "512", "qk_nope": "128", "qk_rope": "64", "v_head": "192",
    "vocab": "129280", "hc": "1", "mtp": "0",
    "dp": "2", "tp": "1", "ep": "1", "pp": "1", "cp": "1",
}

#: (LLMConfig 字段, UI 键, 取值 A, 取值 B, 期望 A, 期望 B)。两个取值**必须不同**，
#: 否则"接线为真"无从证明（下方 `test_wiring_cases_are_discriminating` 守这条）。
_WIRING_CASES = [
    ("num_layers", "layers", "8", "12", 8, 12),
    ("batch_size", "batch", "1", "3", 1, 3),
    ("seq_length", "seq", "4096", "2048", 4096, 2048),
    ("attn_type", "attn", "gqa", "mla", "gqa", "mla"),
    ("num_attention_heads", "heads", "8", "16", 8, 16),
    ("num_query_groups", "kv_groups", "8", "4", 8, 4),
    ("first_k_dense_replace", "dense_k", "1", "2", 1, 2),
    ("num_moe_experts", "experts", "8", "16", 8, 16),
    ("moe_router_topk", "topk", "4", "2", 4, 2),
    ("mtp_num_layers", "mtp", "0", "1", 0, 1),
    ("dsa_fused", "dsa_fused", "1", "0", True, False),
    ("hidden_size", "hidden", "1792", "2048", 1792, 2048),
    ("ffn_hidden_size", "ffn", "3072", "4096", 3072, 4096),
    ("moe_ffn_hidden_size", "moe_ffn", "1024", "2048", 1024, 2048),
    ("q_lora_rank", "q_lora", "1536", "1024", 1536, 1024),
    ("kv_lora_rank", "kv_lora", "512", "256", 512, 256),
    ("qk_nope_head_dim", "qk_nope", "128", "64", 128, 64),
    ("qk_rope_head_dim", "qk_rope", "64", "32", 64, 32),
    ("v_head_dim", "v_head", "192", "128", 192, 128),
    ("vocab_size", "vocab", "129280", "65536", 129280, 65536),
    ("head_dim", "head_dim", "224", "192", 224, 192),
    # `hc` 一键带两个字段：1 = 无 mHC(plain)、≥2 = mHC(hidden×n)。
    ("residual_variant", "hc", "1", "4", "plain", "mhc"),
    ("num_residual_streams", "hc", "1", "4", 1, 4),
    ("cross_entropy_fused", "ce_fused", "0", "1", False, True),
    ("ce_pynative_lean", "ce_lean", "0", "1", False, True),
    # ★ 2026-07-30 第一轮补的那一条（此前**没有**这行 → ~20 个锚点走错 mHC 分支）。
    ("use_fused_mhc", "mhc_fused", "0", "1", False, True),
    # ★ 2026-07-30 第二轮补的那一条（`docs/compress_ratios_mismatch_2026-07-30.md`）：
    #   逐层压缩比曾停在 `_LLM_JSON_ONLY_FIELDS` 里（声明「扁平路上恒取预设值」），
    #   于是同一批锚点用预设的 0/4/128 **循环** 去对站点**逐层表**的真机跑。
    ("csa_compress_ratios", "compress_ratios",
     "0,4,128,4,128,4,128,4", "0,4,128,0,4,128,0,4",
     (0, 4, 128, 4, 128, 4, 128, 4), (0, 4, 128, 0, 4, 128, 0, 4)),
    ("embedding_params_dtype_bytes", "emb_bytes", "2", "4", 2, 4),
]


def _parse_flat(**over):
    q = dict(_FLAT_BASE)
    q.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errs, cfg, pa = S.parse_and_validate(q)
    assert not errs, errs
    assert "llm_json" not in q, "本门必须走**无 llm_json** 的扁平路"
    return cfg


def test_wiring_cases_cover_every_gated_field():
    """`_LLM_FIELD_GATE` 新增一项就必须在此加一行接线用例，否则"登记了但没接线"仍可能发生。"""
    missing = sorted(set(S._LLM_FIELD_GATE) - {c[0] for c in _WIRING_CASES})
    assert not missing, f"这些已登记 UI 键的字段缺接线用例：{missing}"


def test_wiring_cases_are_discriminating():
    for field, key, va, vb, ea, eb in _WIRING_CASES:
        assert va != vb and ea != eb, f"{field}: 两个取值必须不同，否则证明不了接线"


@pytest.mark.parametrize("field,key,va,vb,ea,eb", _WIRING_CASES,
                         ids=[c[0] for c in _WIRING_CASES])
def test_gated_field_wires_through_flat_query(field, key, va, vb, ea, eb):
    """无 `llm_json` 的扁平 query 上翻动 UI 键 → `LLMConfig` 字段必须跟着变。"""
    for v, expect in ((va, ea), (vb, eb)):
        got = getattr(_parse_flat(**{key: v}), field)
        assert got == expect, (
            f"扁平 query {key}={v!r} → LLMConfig.{field} = {got!r}，期望 {expect!r}。"
            f"该字段登记在 `_LLM_FIELD_GATE` 却没真正接上 `parse_and_validate` —— "
            f"扁平路（锚点/探针/页面手配）会静默沿用预设值。")


# ── ④ 本次 bug 的定点回归：mHC 融合分支 ───────────────────────────────────────────────
_DSV4_FLAT = {
    "preset": "dsv4_flash", "attn": "dsv4_hybrid", "layers": "4", "seq": "2048",
    "batch": "1", "mtp": "0", "experts": "8", "topk": "2", "dense_k": "1",
    "heads": "64", "kv_groups": "1", "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
    "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
    "v_head": "512", "vocab": "129280", "hc": "4", "dsa_fused": "1", "ce_fused": "1",
    "dp": "2", "tp": "1", "ep": "2", "pp": "1", "cp": "1",
}


def _hc_op_names(mhc_fused):
    q = dict(_DSV4_FLAT)
    if mhc_fused is not None:
        q["mhc_fused"] = mhc_fused
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errs, cfg, pa = S.parse_and_validate(q)
    assert not errs, errs
    spec = build_llm_spec(cfg)
    ls = spec.layer_specs[spec.layer_pattern[1]]
    return cfg.use_fused_mhc, [o.name for o in ls.ops if "_hc_" in o.name]


def test_mhc_fused_flat_query_selects_fused_branch():
    """`mhc_fused=1` → `_fused_hc_ops`（`residual.py:192`）：`npu_mhc_pre_sinkhorn` 那条链。"""
    fused, names = _hc_op_names("1")
    assert fused is True
    assert "attn_hc_pre_sinkhorn" in names and "attn_hc_aggregate" in names, names
    assert "attn_hc_norm" not in names, (
        f"mhc_fused=1 却仍出现非融合的 `attn_hc_norm`（那张 [S,B,n·H] fp32 是非融合独有）：{names}")


def test_mhc_unfused_flat_query_selects_unfused_branch():
    """`mhc_fused=0` → `_unfused_hc_ops`：`hc_norm` + 三份 fp32 打包副本那条链。"""
    fused, names = _hc_op_names("0")
    assert fused is False
    assert "attn_hc_norm" in names and "attn_hc_mapping_proj" in names, names
    assert "attn_hc_pre_sinkhorn" not in names, names


def test_mhc_fused_absent_or_blank_keeps_base_default():
    """缺省/空串 → 保留基座（与 `ce_fused` 同款语义）→ 手配路径逐字节不变。"""
    base_fused, base_names = _hc_op_names(None)
    blank_fused, blank_names = _hc_op_names("")
    assert base_fused is False and blank_fused is False   # dsv4_flash 基座默认非融合
    assert base_names == blank_names


def test_two_hc_branches_have_same_op_count_and_prefixes():
    """两条分支 op 数/前缀一致（`mhc_wrap._link` 按下标接边）—— 切分支不改图拓扑。"""
    _, fused = _hc_op_names("1")
    _, unfused = _hc_op_names("0")
    assert len(fused) == len(unfused) == 6, (fused, unfused)
    assert [n.split("_hc_")[0] for n in fused] == [n.split("_hc_")[0] for n in unfused]


# ── ⑤ 第二个 bug 的定点回归：逐层压缩比（2026-07-30，本 bug class 的第三例）───────────
#    `docs/compress_ratios_mismatch_2026-07-30.md`
def _layer_keys(compress_ratios, layers="8"):
    q = dict(_DSV4_FLAT)
    q["layers"] = layers
    q["mhc_fused"] = "1"
    if compress_ratios is not None:
        q["compress_ratios"] = compress_ratios
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errs, cfg, pa = S.parse_and_validate(q)
    assert not errs, errs
    spec = build_llm_spec(cfg)
    return cfg.csa_compress_ratios, [k for k in spec.layer_pattern if k.startswith("dsv4hyb_")]


def test_compress_ratios_flat_query_selects_per_layer_table():
    """扁平 query 的逐层表必须**逐层**落到 `dsv4hyb_r{ratio}_*` 层 key 上。

    站点表 `[0,4,128,4,128,4,128,4]`（`ab_fusion_2026-07-25/dsv4h_fused_pp4_recomp.yaml:121`）
    的层型分布是 1×r0 / 4×r4 / 3×r128 —— 与预设**循环**的 3/3/2 不同，而三种层型驻留实测
    各不相同（167 逐层直测 r0 2235.1 / r4 2341.2 / r128 2116.1 MiB）。"""
    ratios, keys = _layer_keys("0,4,128,4,128,4,128,4")
    assert ratios == (0, 4, 128, 4, 128, 4, 128, 4)
    assert [k.split("_")[1] for k in keys] == ["r0", "r4", "r128", "r4", "r128", "r4", "r128", "r4"]
    mix = {r: [k.split("_")[1] for k in keys].count(f"r{r}") for r in (0, 4, 128)}
    assert mix == {0: 1, 4: 4, 128: 3}, mix


def test_compress_ratios_absent_or_blank_keeps_preset_cycle():
    """缺省/空串 → 保留基座（预设 `dsv4_flash` 的 0/4/128 循环近似）→ 手配路径逐字节不变。

    这条**同时**把缺陷本体钉成回归：基座那张表是 3×r0/3×r4/2×r128，与站点表不同；
    若哪天有人把旋钮摘掉，锚点会静默退回这张近似表（= 2026-07-30 之前的真实状态）。"""
    for v in (None, ""):
        ratios, keys = _layer_keys(v)
        assert ratios == (0, 4, 128, 0, 4, 128, 0, 4), (v, ratios)
        mix = {r: [k.split("_")[1] for k in keys].count(f"r{r}") for r in (0, 4, 128)}
        assert mix == {0: 3, 4: 3, 128: 2}, (v, mix)


def test_compress_ratios_rejects_non_integer_items():
    """fail-loud 而非静默截断（`4.9 → 4` 会错走 `dsv4hyb_r4_*` 图，同 build_llm §F5c 的判据）。"""
    for bad in ("0,4,128,4.9,128,4,128,4", "0,4,128,,128,4,128,4", "0,4,128,x,128,4,128,4"):
        q = dict(_DSV4_FLAT, layers="8", compress_ratios=bad)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            errs, cfg, pa = S.parse_and_validate(q)
        assert errs and cfg is None, (bad, errs)
        assert "compress_ratios" in errs[0], (bad, errs)
