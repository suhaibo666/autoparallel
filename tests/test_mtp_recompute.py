"""MTP 层须能像普通 transformer 层一样配置重算（用户报告 2026-07-21）。

根因（parse_and_validate）：重算层域被限制在 **1..N**（仅 transformer），而 MTP 层占
`layer_id N+1..N+mtp`（`gen_layer_pattern`：`[embedding] + N decoder + mtp×MTP + [lm_head]`，
enumerate → embedding=0、transformer=1..N、mtp=N+1..N+mtp、head 末位）。故此前：
  - 默认「空=全部层」= `range(1, N+1)` **不含 MTP** → MTP 恒不重算；
  - 显式范围 `hi=N` → 用户填 MTP 层号（N+1..）被判越界拒绝。
修复：重算层域改为 **1..T**（`T = N + mtp`），与 PP 切分 / pp_split 同口径（MTP 一等层）。
`mtp=0` 时 `T==N`，逐字节不变（golden 锚点不破）。

layers=8, mtp=1 → MTP 层 id = 9；transformer = 1..8。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve_explorer import eval_config

BASE = {"attn": "mla", "layers": "8", "mtp": "1", "dp": "1", "tp": "1", "pp": "1", "batch": "1"}


def _cfg(**over):
    d = dict(BASE)
    d.update({k: str(v) for k, v in over.items()})
    return d


def _layers(r):
    """{layer_id -> recomp}，排除 embedding/lm_head 伪层（保留 MTP）。"""
    return {L["id"]: L["recomp"] for s in r["stages"] for L in s["graph"]
            if L["type"] not in ("embedding", "lm_head")}


def test_mtp_recomputed_by_default_full():
    # 默认「空 sel_layers = 全部层」现含 MTP（id 9）。修复前 MTP=none。
    r = eval_config(_cfg(recompute="full", sel_layers=""))
    assert r["ok"], r
    lys = _layers(r)
    assert 9 in lys, ("MTP 层缺失", sorted(lys))
    assert all(v == "full" for v in lys.values()), lys      # 含 MTP 全 full


def test_mtp_recompute_explicit_layer_number():
    # 显式只重算 MTP 层（层号 9 = N+1）。修复前该层号被判越界（r not ok）。
    r = eval_config(_cfg(recompute="full", sel_layers="9"))
    assert r["ok"], r
    lys = _layers(r)
    assert lys[9] == "full"
    assert all(lys[i] == "none" for i in range(1, 9)), lys   # transformer 层不重算


def test_mtp_recompute_select_attn():
    # select 模块重算对 MTP 同样生效（MTP 内层 decoder 与主干同 op 名，selector 命中）。
    r = eval_config(_cfg(recompute="select", select="attn", sel_layers="9"))
    assert r["ok"], r
    assert _layers(r)[9] == "select"


def test_layer_number_above_T_still_rejected():
    # 域上界现为 T=N+mtp=9；层号 10 仍越界（不是无界）。
    r = eval_config(_cfg(recompute="full", sel_layers="10"))
    assert not r["ok"] and any("层范围" in e for e in r["errors"]), r


def test_mtp_zero_domain_unchanged():
    # mtp=0 → T==N，默认全部仍是 1..N（无 id 9 的 MTP 层），逐字节不变。
    r = eval_config(_cfg(mtp="0", recompute="full", sel_layers=""))
    assert r["ok"], r
    lys = _layers(r)
    assert set(lys) == set(range(1, 9)) and all(v == "full" for v in lys.values()), lys
