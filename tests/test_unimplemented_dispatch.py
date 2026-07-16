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
    # 注：add_bias_linear / add_qkv_bias 已由 round3 A(N4，2026-07-16)从 fail-loud **降级为
    #   ModelingApproxWarning**（bias 内存中性）→ 移至下方 test_bias_fields_warn_and_build。
    # 注：qk_layernorm=True 已由 X3（closure-v4 P1-13/F2，2026-07-15）为 gqa/mha **建 q/k norm op**
    #   → 不再 fail-loud（llama 是 gqa）。移至下方正向测试 test_qk_layernorm_gqa_builds_norm_ops。
])
def test_unimplemented_dispatch_raises(field, value):
    cfg = dataclasses.replace(llama(4), **{field: value})
    with pytest.raises(NotImplementedError):
        build_llm_spec(cfg)


@pytest.mark.parametrize("field", ["add_bias_linear", "add_qkv_bias"])
def test_bias_fields_warn_and_build(field):
    """round3 A(N4)：bias 类字段是**内存中性**（bias = [out] 一维,相对权重可忽略）→ 从 fail-loud
    降级为 `ModelingApproxWarning`：发警告但**继续按无 bias 建 spec**（不再硬拒、不产错数）。
    并核对 op 图与 base 逐 op 一致（bias 未建为 param → 不改图,与 SWA 内存中性字段同性质）。"""
    from cost_eval.advisories import ModelingApproxWarning
    base = llama(4)
    cfg = dataclasses.replace(base, **{field: True})
    with pytest.warns(ModelingApproxWarning, match=field):
        spec = build_llm_spec(cfg)
    assert spec is not None
    assert _op_graph(spec) == _op_graph(build_llm_spec(base))   # bias 未建 param → op 图不变


def test_qk_layernorm_gqa_builds_norm_ops():
    """qk_layernorm=True（gqa/mha）：X3 已建 q_norm/k_norm op（Qwen3 变体，per-head head_dim
    RMSNorm）→ 正常 build 且 op 图含两个 norm；此前 fail-loud 是缺陷（静默丢 4.5-9 GiB）。"""
    base = llama(4)
    qk = dataclasses.replace(base, qk_layernorm=True)
    spec = build_llm_spec(qk)
    names = {n for ops in _op_graph(spec).values() for n in ops}
    assert "q_norm" in names and "k_norm" in names
    # mla 仍 fail-loud（其 qk norm 由 latent/per-head norm 已覆盖，不重复建）
    import pytest as _pt
    with _pt.raises(NotImplementedError):
        build_llm_spec(dataclasses.replace(deepseek_v3(4), qk_layernorm=True))


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
