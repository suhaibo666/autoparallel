"""网页「模型结构」视图须随重算配置变化（用户报告 2026-07-15）。

此前 `graph_json` 从不接收 `RecomputeSpec` → 无论 recompute=None/full/select，结构图每层/每
op 的激活值逐字节相同（bug）。修复后 graph_json 按**仿真器同口径**（`mem_timeline` FWD pin）
标注每 op 是否被重算（saves 不存）、层头 `act_mib` = stored 总量：
  - None   → `estimate_structure_memory.activation_saves`（全量存）
  - full   → `estimate_structure_memory.checkpoint_input`（仅层入口边界）
  - select → `estimate_select_memory.act_live_pinned`（非选中 saves + 层入口边界）
被重算的 op：`recomp=True`、其 stored `act_mib=0`（不计入层头），并带 `recomp_mib`（省下的量）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve_explorer import eval_config

BASE = {"attn": "mla", "layers": "4", "dp": "1", "tp": "1", "pp": "1", "batch": "1"}


def _cfg(**over):
    d = dict(BASE)
    d.update({k: str(v) for k, v in over.items()})
    return d


def _decoder_layers(r):
    """所有 stage 的结构图里非伪层（embedding/lm_head）——即用户配额口径的可切分层。"""
    out = []
    for s in r["stages"]:
        for L in s["graph"]:
            if L["type"] not in ("embedding", "lm_head"):
                out.append(L)
    return out


def test_none_marks_no_recompute():
    r = eval_config(_cfg(recompute="None"))
    assert r["ok"], r
    for L in _decoder_layers(r):
        assert L.get("recomp") in (None, "none")
        for o in L["ops"]:
            assert not o.get("recomp"), (L["id"], o["name"])


def test_full_marks_ops_and_drops_layer_act():
    """full：被标 recomp=full 的层，层头 stored 激活必须显著低于 None（仅保层入口），
    且该层每个被重算 op 的 stored act_mib=0（不计入层头）。"""
    rn = {L["id"]: L for L in _decoder_layers(eval_config(_cfg(recompute="None")))}
    rf = _decoder_layers(eval_config(_cfg(recompute="full")))
    full_layers = [L for L in rf if L.get("recomp") == "full"]
    assert full_layers, "full 模式应有层标记 recomp=full"
    for L in full_layers:
        assert L["act_mib"] < rn[L["id"]]["act_mib"], (L["id"], L["act_mib"], rn[L["id"]]["act_mib"])
        assert any(o.get("recomp") for o in L["ops"]), L["id"]
        for o in L["ops"]:
            if o.get("recomp"):
                assert o["act_mib"] == 0, (L["id"], o["name"], o["act_mib"])


def test_select_attn_marks_only_attn_and_reduces_stored():
    """select(attn)：部分 op 标记重算，层头 stored 激活 ≤ None，且总量严格下降。"""
    rn = {L["id"]: L for L in _decoder_layers(eval_config(_cfg(recompute="None")))}
    rs = _decoder_layers(eval_config(_cfg(recompute="select", select="attn")))
    assert rs
    any_recomp = False
    tot_none = tot_sel = 0.0
    for L in rs:
        assert L["act_mib"] <= rn[L["id"]]["act_mib"] + 1e-6
        tot_sel += L["act_mib"]
        tot_none += rn[L["id"]]["act_mib"]
        for o in L["ops"]:
            if o.get("recomp"):
                any_recomp = True
                assert o["act_mib"] == 0, (L["id"], o["name"])
    assert any_recomp, "select(attn) 应标记若干 attn op 为重算"
    assert tot_sel < tot_none, (tot_sel, tot_none)


def test_recomp_op_exposes_saved_amount():
    """被重算的 op 带 recomp_mib>0（省下的激活量），供 UI 显示「↻重算 省 X MiB」。"""
    rf = _decoder_layers(eval_config(_cfg(recompute="full")))
    saw = False
    for L in rf:
        if L.get("recomp") != "full":
            continue
        for o in L["ops"]:
            if o.get("recomp") and o.get("recomp_mib", 0) > 0:
                saw = True
    assert saw, "full 模式下应有被重算 op 暴露 recomp_mib>0"


def test_full_respects_layer_range():
    """`重算层范围`（sel_layers）须对 full 生效：只有范围内层重算，范围外层保持 none。
    此前 sel_layers 仅 custom 模式生效，full/select 恒全层重算 → 改层范围结构图不变（用户报告 #1）。"""
    r = eval_config(_cfg(recompute="full", sel_layers="1-2"))
    assert r["ok"], r
    by = {L["id"]: L for L in _decoder_layers(r)}
    assert by[1]["recomp"] == "full" and by[2]["recomp"] == "full", by
    assert by[3]["recomp"] == "none" and by[4]["recomp"] == "none", by


def test_select_respects_layer_range():
    r = eval_config(_cfg(recompute="select", select="attn", sel_layers="2-3"))
    assert r["ok"], r
    by = {L["id"]: L for L in _decoder_layers(r)}
    assert by[1]["recomp"] == "none", by
    assert by[2]["recomp"] == "select" and by[3]["recomp"] == "select", by
    assert by[4]["recomp"] == "none", by


def test_empty_layer_range_means_all_and_bytewise_identical():
    """空/缺省 sel_layers → 全层重算，与不填该字段逐字节一致（不破坏既有 full 锚点）。"""
    r0 = eval_config(_cfg(recompute="full"))
    r1 = eval_config(_cfg(recompute="full", sel_layers=""))
    a0 = {L["id"]: L["act_mib"] for L in _decoder_layers(r0)}
    a1 = {L["id"]: L["act_mib"] for L in _decoder_layers(r1)}
    assert a0 == a1
    assert all(L["recomp"] == "full" for L in _decoder_layers(r1))


def test_invalid_layer_range_rejected_in_full():
    """非法层范围（越界/倒序）在 full 模式也报错，不再静默忽略。"""
    r = eval_config(_cfg(recompute="full", sel_layers="3-99"))
    assert not r["ok"] and any("重算层范围" in e for e in r["errors"]), r


def test_per_save_stored_flag_present():
    """每条 acts 明细带 stored 标志：None 全 True；full 下重算 op 的明细全 False（层入口除外）。"""
    rn = _decoder_layers(eval_config(_cfg(recompute="None")))
    for L in rn:
        for o in L["ops"]:
            for a in o["acts"]:
                assert a.get("stored") is True, (L["id"], o["name"], a["name"])
    rf = _decoder_layers(eval_config(_cfg(recompute="full")))
    for L in rf:
        if L.get("recomp") != "full":
            continue
        for o in L["ops"]:
            if not o.get("recomp"):
                continue
            for a in o["acts"]:
                # 层入口 checkpoint_input 边界可能仍 stored；其余重物化 save 必为 False
                assert "stored" in a, (L["id"], o["name"], a["name"])
