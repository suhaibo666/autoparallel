"""P1-10/P1-11（2026-07-14 review）：结构合法性统一校验——非法/不自洽配置 fail-loud。

修前反例（review §4 探针）：`csa_compress_ratios=(2,3,2,3)` 被当稀疏路径接受并产 r2/r3 错图；
`H=10, n_heads=3` 静默 floor 出 head_dim=3。
"""
import dataclasses

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.presets import deepseek_v3, deepseek_v4


def _dsv3(**over):
    return dataclasses.replace(deepseek_v3(4), **over)


def test_good_configs_still_build():
    assert build_llm_spec(deepseek_v3(4))
    assert build_llm_spec(deepseek_v4(4))


def test_head_dim_floor_rejected():
    bad = _dsv3(hidden_size=1793, head_dim=None)     # 1793 % 8 ≠ 0
    with pytest.raises(ValueError, match="head_dim"):
        build_llm_spec(bad)


def test_explicit_head_dim_allows_indivisible_hidden():
    ok = _dsv3(hidden_size=1793, head_dim=192)       # 显式 head_dim → 放行
    assert build_llm_spec(ok)


def test_illegal_compress_ratio_rejected():
    v4 = deepseek_v4(4)
    bad = dataclasses.replace(v4, csa_compress_ratios=(2, 3, 2, 3))
    with pytest.raises(NotImplementedError, match="压缩比"):
        build_llm_spec(bad)


def test_ratio_length_mismatch_rejected():
    v4 = deepseek_v4(4)
    bad = dataclasses.replace(v4, csa_compress_ratios=v4.csa_compress_ratios[:2])
    with pytest.raises(ValueError, match="csa_compress_ratios 长度"):
        build_llm_spec(bad)


def test_seq_not_divisible_by_ratio_rejected():
    v4 = deepseek_v4(4)
    bad = dataclasses.replace(v4, seq_length=4098)   # 4098 % 4 ≠ 0
    with pytest.raises(ValueError, match="整除"):
        build_llm_spec(bad)


def test_topk_exceeds_experts_rejected():
    bad = _dsv3(moe_router_topk=10**6)
    with pytest.raises(ValueError, match="moe_router_topk"):
        build_llm_spec(bad)


def test_moe_layer_freq_length_rejected():
    bad = _dsv3(moe_layer_freq=(1, 0))               # 长度 2 ≠ num_layers 4
    with pytest.raises(ValueError, match="moe_layer_freq"):
        build_llm_spec(bad)


def test_query_groups_must_divide_heads():
    bad = _dsv3(attn_type="gqa", num_query_groups=3,
                q_lora_rank=0, kv_lora_rank=0, qk_rope_head_dim=0,
                qk_nope_head_dim=0, v_head_dim=0, head_dim=224)
    with pytest.raises(ValueError, match="num_query_groups"):
        build_llm_spec(bad)
