"""mHC 配置开关（2026-07-20）：网页可配 num_residual_streams(HyperConnection 残差流)。

此前 mHC 只由 deepseek_v4 preset 基座隐式带上(residual_variant="mhc"/num_residual_streams=4),
网页无开关——既看不到、也无法对非 DSv4 模型开、或对 DSv4 关。本组钉住 hc 字段的语义:
空=保留基座(不误关预设 mHC)、1=plain、≥2=mhc(hidden×n)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve_explorer import parse_and_validate, eval_config


def _cfg(**q):
    e, cfg, pa = parse_and_validate(q)
    assert not e, e
    return cfg


def test_hc_toggles_residual_variant():
    # v3 基座(默认无预设):hc 显式控制 residual。
    assert (_cfg(layers="8").residual_variant,
            _cfg(layers="8").num_residual_streams) == ("plain", 1)          # 空 → 基座 plain
    assert _cfg(layers="8", hc="1").residual_variant == "plain"             # 1 → plain
    c4 = _cfg(layers="8", hc="4")
    assert c4.residual_variant == "mhc" and c4.num_residual_streams == 4    # 4 → mhc×4


def test_hc_empty_preserves_v4_preset_mhc():
    # DSv4 预设基座自带 mHC(4);hc 留空**不得**误关它。
    c = _cfg(preset="dsv4_flash", attn="dsv4_hybrid", layers="4", experts="256",
             topk="6", dense_k="1", heads="64", kv_groups="1")
    assert c.residual_variant == "mhc" and c.num_residual_streams == 4


def test_hc_can_turn_off_mhc_on_v4():
    # 用户显式 hc=1 → 对 DSv4 关掉 mHC(plain)。
    c = _cfg(preset="dsv4_flash", attn="dsv4_hybrid", layers="4", experts="256",
             topk="6", dense_k="1", heads="64", kv_groups="1", hc="1")
    assert c.residual_variant == "plain" and c.num_residual_streams == 1


def test_hc_invalid_rejected():
    e, cfg, pa = parse_and_validate({"layers": "8", "hc": "0"})
    assert cfg is None and any("mHC" in x for x in e)


def test_mhc_raises_memory():
    # mHC ×4 抬持久(×n 残差流参数)+激活 → 峰值显著高于 plain。
    base = eval_config({"layers": "8", "dp": "2", "pp": "1", "batch": "1", "hc": "1"})
    mhc = eval_config({"layers": "8", "dp": "2", "pp": "1", "batch": "1", "hc": "4"})
    assert base["ok"] and mhc["ok"]
    assert mhc["device_peak"] > base["device_peak"] + 1000    # ×4 残差流明显更贵
