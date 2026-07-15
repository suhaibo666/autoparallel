"""重算「具体层数」输入须支持多段不连续范围 `1-8;12-13;23-25`（用户报告 2026-07-15）。

此前 `重算层范围`（sel_layers）只解析单个 `a-b`；`细粒度重算` 层号形态仅 pattern 内逗号多段。
修复：统一 `_parse_layer_ranges`（`,`/`;`/全角均可，闭区间并集）作用于**每个配置具体层数的地方**：
  - `重算层范围`（full/select/custom 层集）
  - `细粒度重算` 层号形态的 pattern 层范围（0-indexed）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve_explorer import eval_config, _parse_layer_ranges

BASE = {"attn": "mla", "layers": "30", "dp": "1", "tp": "1", "pp": "1", "batch": "1"}


def _cfg(**over):
    d = dict(BASE)
    d.update({k: str(v) for k, v in over.items()})
    return d


def _decoder(r):
    return [L for s in r["stages"] for L in s["graph"] if L["type"] not in ("embedding", "lm_head")]


# ── 单元：_parse_layer_ranges ────────────────────────────────────────────────
def test_parse_layer_ranges_semicolon():
    e, s = _parse_layer_ranges("1-8;12-13;23-25", 1, 30)
    assert e == [] and s == set(range(1, 9)) | {12, 13, 23, 24, 25}


def test_parse_layer_ranges_comma_and_fullwidth():
    _, s1 = _parse_layer_ranges("1-8,12-13,23-25", 1, 30)
    _, s2 = _parse_layer_ranges("1-8；12-13，23-25", 1, 30)   # 全角 ；，
    assert s1 == s2 == set(range(1, 9)) | {12, 13, 23, 24, 25}


def test_parse_layer_ranges_single_and_empty():
    assert _parse_layer_ranges("5", 1, 30)[1] == {5}
    assert _parse_layer_ranges("", 1, 30) == ([], None)
    assert _parse_layer_ranges("  ", 1, 30) == ([], None)


def test_parse_layer_ranges_errors():
    assert _parse_layer_ranges("1-8;40", 1, 30)[0]        # 越界
    assert _parse_layer_ranges("8-1", 1, 30)[0]           # 倒序
    assert _parse_layer_ranges("1-x", 1, 30)[0]           # 非整数
    assert _parse_layer_ranges("0-3", 1, 30)[0]           # 下界 0 < 1


# ── 集成：重算层范围多段 ─────────────────────────────────────────────────────
def test_sel_layers_multirange_full():
    r = eval_config(_cfg(recompute="full", sel_layers="1-8;12-13;23-25"))
    assert r["ok"], r
    want = set(range(1, 9)) | {12, 13, 23, 24, 25}
    for L in _decoder(r):
        assert L["recomp"] == ("full" if L["id"] in want else "none"), (L["id"], L["recomp"])


def test_sel_layers_semicolon_comma_equivalent():
    a = {L["id"] for L in _decoder(eval_config(_cfg(recompute="full", sel_layers="1-8;12-13;23-25"))) if L["recomp"] == "full"}
    b = {L["id"] for L in _decoder(eval_config(_cfg(recompute="full", sel_layers="1-8,12-13,23-25"))) if L["recomp"] == "full"}
    assert a == b == set(range(1, 9)) | {12, 13, 23, 24, 25}


def test_sel_layers_multirange_select():
    r = eval_config(_cfg(recompute="select", select="attn", sel_layers="1-2;29-30"))
    assert r["ok"], r
    sel = {L["id"] for L in _decoder(r) if L["recomp"] == "select"}
    assert sel == {1, 2, 29, 30}


def test_sel_layers_multirange_out_of_range_rejected():
    r = eval_config(_cfg(recompute="full", sel_layers="1-8;40-41"))
    assert not r["ok"] and any("层范围" in e for e in r["errors"]), r


def test_sel_layers_empty_means_all():
    r = eval_config(_cfg(recompute="full", sel_layers=""))
    assert r["ok"], r
    assert all(L["recomp"] == "full" for L in _decoder(r))


# ── 集成：细粒度重算 pattern 内多段（0-indexed） ─────────────────────────────
def test_sel_cfg_pattern_multirange_disjoint():
    # 0-indexed 0-7,11-12,22-24 → 评估器层 id +1 → 1-8,12-13,23-25
    r = eval_config(_cfg(recompute="custom", sel_cfg="self_attention:0-7,11-12,22-24"))
    assert r["ok"], r
    sel = {L["id"] for L in _decoder(r) if L["recomp"] == "select"}
    assert sel == set(range(1, 9)) | {12, 13, 23, 24, 25}


# ── 统一入口 dispatcher：多段 stage 号识别（`s0,2-3`），与层号形态区分 ─────────
def test_stage_key_detection_multiseg():
    from serve_explorer import _is_stage_seg_key
    assert _is_stage_seg_key("s0") and _is_stage_seg_key("s1-2")
    assert _is_stage_seg_key("s0,2-3") and _is_stage_seg_key("s0,1")
    assert not _is_stage_seg_key("self_attention")   # 字母 → 层号 pattern
    assert not _is_stage_seg_key("mlp") and not _is_stage_seg_key("flash")
    assert not _is_stage_seg_key("s")                # 空 body


def test_sel_cfg_stage_multiseg_eval():
    # pp=2、30 层 → stage0/1 各 15 层；`s0,1:mlp` 经 dispatcher 识别为 stage 形态 → 全 30 层 mlp 重算
    r = eval_config(_cfg(pp=2, mbs=2, recompute="custom", sel_cfg="s0,1:mlp"))
    assert r["ok"], r
    sel = {L["id"] for L in _decoder(r) if L["recomp"] == "select"}
    assert len(sel) == 30, sorted(sel)
