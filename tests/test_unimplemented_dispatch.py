"""Task 5 [I2] — 未实现的结构分派项显式报错（不静默产错图）。

以下 LLMConfig 字段会改变 op 图，但当前实现只建了其中一个取值；其它取值若被 preset 设置
会**静默忽略**、产出「貌似合理实则错误」的图。仿 head.py:50（loss_type 未知即 NotImplementedError）
在装配点显式报错：
  - norm_placement != "pre"
  - normalization != "RMSNorm"
  - position_embedding_type != "rope"
"""
import dataclasses

import pytest

from cost_eval.presets import deepseek_v3, deepseek_v4, llama, qwen2, mixtral
from cost_eval.build_llm import build_llm_spec


@pytest.mark.parametrize("field,value", [
    # gated_linear_unit=False 现已建 op 图（D-6，ffn 按 d.gated_linear_unit 分派）→ 不再 raise，
    # 覆盖迁至 test_ungated_ffn.py。
    ("norm_placement", "post"),
    ("norm_placement", "sandwich"),
    ("normalization", "LayerNorm"),
    ("position_embedding_type", "learned_absolute"),
    ("position_embedding_type", "none"),
    # Task 3：补齐此前 set-but-ignored 的 op-图相关字段（不再静默忽略）
    ("add_bias_linear", True),      # linear bias 未建为 param
    ("add_qkv_bias", True),         # QKV bias 未建为 param（Qwen）
    ("qk_layernorm", True),         # Q/K RMSNorm 未建为 op（Qwen3 变体）
])
def test_unimplemented_dispatch_raises(field, value):
    cfg = dataclasses.replace(llama(4), **{field: value})
    with pytest.raises(NotImplementedError):
        build_llm_spec(cfg)


def _op_graph(spec):
    """层名 → op 名序列（比较 op 图是否被某字段改变）。"""
    return {k: [o.name for o in v.ops] for k, v in spec.layer_specs.items()}


@pytest.mark.parametrize("field,value", [
    ("window_size", 128),
    ("window_pattern", (0, 1, 1, 1)),
])
def test_swa_fields_are_memory_neutral_noops(field, value):
    """window_size / window_pattern：flash-attn 下 SWA 对训练激活内存**中性**（设计 §7.4：
    saves 仍是 Q/K/V/O+lse，[S,S] 分数从不物化）。故意**不 raise**，且**不改 op 图**——
    与全注意力层逐 op 一致（区别于上面 fail-loud 的字段）。"""
    base = llama(4)
    swa = dataclasses.replace(base, **{field: value})
    assert _op_graph(build_llm_spec(swa)) == _op_graph(build_llm_spec(base))


def test_all_presets_still_build():
    """所有现有 preset 用的都是已实现取值 → 正常建 spec（含 qwen2 的 bias 已按 §7.4/Task3 归零）。"""
    for cfg in (deepseek_v3(4), deepseek_v4(4), llama(4), qwen2(4), mixtral(4)):
        assert build_llm_spec(cfg) is not None
