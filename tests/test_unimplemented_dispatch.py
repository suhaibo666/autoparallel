"""Task 5 [I2] — 未实现的结构分派项显式报错（不静默产错图）。

以下 LLMConfig 字段会改变 op 图，但当前实现只建了其中一个取值；其它取值若被 preset 设置
会**静默忽略**、产出「貌似合理实则错误」的图。仿 head.py:50（loss_type 未知即 NotImplementedError）
在装配点显式报错：
  - gated_linear_unit=False（ffn 硬编码 2*F gated）
  - norm_placement != "pre"
  - normalization != "RMSNorm"
  - position_embedding_type != "rope"
"""
import dataclasses

import pytest

from cost_eval.presets import deepseek_v3, deepseek_v4, llama, qwen2, mixtral
from cost_eval.build_llm import build_llm_spec


@pytest.mark.parametrize("field,value", [
    ("gated_linear_unit", False),
    ("norm_placement", "post"),
    ("norm_placement", "sandwich"),
    ("normalization", "LayerNorm"),
    ("position_embedding_type", "learned_absolute"),
    ("position_embedding_type", "none"),
])
def test_unimplemented_dispatch_raises(field, value):
    cfg = dataclasses.replace(llama(4), **{field: value})
    with pytest.raises(NotImplementedError):
        build_llm_spec(cfg)


def test_all_presets_still_build():
    """所有现有 preset 用的都是已实现取值 → 正常建 spec。"""
    for cfg in (deepseek_v3(4), deepseek_v4(4), llama(4), qwen2(4), mixtral(4)):
        assert build_llm_spec(cfg) is not None
