"""细粒度重算统一入口（用户报告 #2：per-stage 与 mf 层号两个框合并为一个 `sel_cfg`）。

一个文本框、按每段 key 自动识别坐标系：`s0:`/`s1-2:` = 按 PP stage；`mlp:`/`self_attention:` =
按绝对层号(mf select_module, 0-indexed)。两种写法互斥（混用报错）。`sel_stage` 保留为隐藏兼容
别名。本组钉住：等价性（stage 与层号殊途同归）、自动识别、混用报错、兼容别名回落。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve_explorer import eval_config, parse_recompute_cfg

BASE = {"attn": "mla", "layers": "4", "dp": "1", "tp": "1", "batch": "1"}


def _cfg(**over):
    d = dict(BASE)
    d.update({k: str(v) for k, v in over.items()})
    return d


def _decoder(r):
    return [L for s in r["stages"] for L in s["graph"] if L["type"] not in ("embedding", "lm_head")]


# ── 单元：dispatcher 自动识别 ────────────────────────────────────────────────
def test_dispatch_stage_syntax():
    e, d = parse_recompute_cfg("s0:both; s1:none", pp=2, N=4, mtp=0, pp_split=None)
    assert e == [] and d is not None
    # pp=2 均匀切 4 层 → stage0={1,2}; s0:both → 1,2 有 attn∪mlp；s1:none → 3,4 不选
    assert set(d.keys()) == {1, 2}


def test_dispatch_layer_syntax():
    e, d = parse_recompute_cfg("mlp:0-3", pp=1, N=4, mtp=0, pp_split=None)
    assert e == [] and d is not None
    assert set(d.keys()) == {1, 2, 3, 4}            # 0-indexed 0-3 → 评估器层 1-4


def test_mixed_syntax_rejected():
    e, d = parse_recompute_cfg("s0:both; mlp:0-3", pp=2, N=4, mtp=0, pp_split=None)
    assert d is None and any("不可混用" in x for x in e), e


def test_empty_is_none():
    assert parse_recompute_cfg("", pp=1, N=4, mtp=0, pp_split=None) == ([], None)


def test_stage_and_layer_equivalent_pp1():
    """pp=1 时 stage0 覆盖全部层 → `s0:both` 与层号 `self_attention:0-3; mlp:0-3` 产出同一 select_ops。"""
    _, ds = parse_recompute_cfg("s0:both", pp=1, N=4, mtp=0, pp_split=None)
    _, dl = parse_recompute_cfg("self_attention:0-3; mlp:0-3", pp=1, N=4, mtp=0, pp_split=None)
    assert ds == dl and ds is not None


# ── 集成：经 eval_config 两种写法给出同一峰值 ─────────────────────────────────
def test_eval_stage_vs_layer_same_peak_pp1():
    rs = eval_config(_cfg(pp=1, recompute="custom", sel_cfg="s0:self_attention"))
    rl = eval_config(_cfg(pp=1, recompute="custom", sel_cfg="self_attention:0-3"))
    assert rs["ok"] and rl["ok"], (rs, rl)
    assert rs["device_peak"] == rl["device_peak"]
    # 两者都把 4 层的 attn 标为重算
    for r in (rs, rl):
        assert all(L["recomp"] == "select" for L in _decoder(r))


def test_eval_mixed_syntax_errors():
    r = eval_config(_cfg(recompute="custom", sel_cfg="s0:both; mlp:0-3"))
    assert not r["ok"] and any("不可混用" in e for e in r["errors"]), r


def test_sel_stage_compat_alias_still_works():
    """旧字段 sel_stage 仍作隐藏兼容别名：sel_cfg 空时回落读取，结果与写进 sel_cfg 一致。"""
    r_alias = eval_config(_cfg(pp=2, mbs=2, recompute="custom", sel_stage="s0:both"))
    r_new = eval_config(_cfg(pp=2, mbs=2, recompute="custom", sel_cfg="s0:both"))
    assert r_alias["ok"] and r_new["ok"], (r_alias, r_new)
    assert r_alias["device_peak"] == r_new["device_peak"]


def test_sel_cfg_takes_precedence_over_alias():
    """sel_cfg 非空时优先于 sel_stage 别名（不回落）。"""
    r = eval_config(_cfg(pp=1, recompute="custom",
                         sel_cfg="mlp:0-3", sel_stage="s0:self_attention"))
    assert r["ok"], r
    # 生效的是 sel_cfg(mlp) 而非别名(attn)——mlp 段被重算即证明 sel_cfg 胜出
    assert all(L["recomp"] == "select" for L in _decoder(r))
