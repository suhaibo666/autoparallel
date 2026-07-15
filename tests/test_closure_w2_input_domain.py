"""第四轮闭环审计（`analysis/closure_audit_v2_verification_2026-07-15.md` §F5）反例转正式回归。

上轮「正值 validator」用 `isinstance(v, int)` + `v < 1`，但 **Python `bool` 是 `int` 子类**
（`True==1`/`False==0`），且 CSA ratio 校验用 `int(r)` 会**静默截断浮点**。独立探针实证以下
本应被拒的输入全部被接受：

- **F5a**（`specs.py` ParallelConfig）：`ParallelConfig(tp=True)`、`ParallelConfig(num_microbatches=True)`
  等——bool 当整数并行度（`isinstance(True,int)` 为真、`True>=1` 通过）。
- **F5b**（`build_llm.py` _validate_structure）：`hidden_size=True, num_attention_heads=True`
  → `DimTable(H=True, n_heads=True)`——bool 当维度。
- **F5c**（`build_llm.py` _validate_structure CSA 循环）：`csa_compress_ratios=(4.9,1,1,4)`
  通过，`int(4.9)=4` 静默变 `dsv4hyb_r4_*` 图。

三档修复：所有整数标量/维度判据**先排除 bool**（严格整数），CSA ratio **拒绝非整数值**
（4.9 拒；4 与 4.0 接受）。**正向守卫**：默认构造 / 所有预设 / 合法 int 值必须继续通过——
类型守卫绝不误伤合法配置。
"""
import dataclasses

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.llm_config import LLMConfig
from cost_eval.presets import deepseek_v3, deepseek_v4, llama, mixtral, qwen2
from cost_eval.specs import ParallelConfig


# ── helpers ──────────────────────────────────────────────────────────────────
def _dsv3(**over):
    return dataclasses.replace(deepseek_v3(4), **over)


def _dsv4(**over):
    return dataclasses.replace(deepseek_v4(4), **over)


def _min_gqa(**over):
    base = dict(num_layers=2, hidden_size=8, num_attention_heads=2, vocab_size=16,
                seq_length=16, batch_size=1, head_dim=4, attn_type="gqa", ffn_hidden_size=16)
    base.update(over)
    return LLMConfig(**base)


# ══════════════════════════════════════════════════════════════════════════════
# F5a — ParallelConfig：bool 当整数标量/并行度必须被拒（specs.py）
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("field", [
    "dp_replicate", "dp_shard", "cp", "tp", "pp", "ep",
    "interleave", "microbatch", "num_microbatches",
])
def test_parallel_scalar_bool_true_rejected(field):
    """bool True 是 int 子类且 `True>=1` → 旧校验放行；须 fail-loud（报错含字段名）。"""
    with pytest.raises(ValueError, match=field):
        ParallelConfig(**{field: True})


@pytest.mark.parametrize("field", [
    "dp_replicate", "dp_shard", "cp", "tp", "pp", "ep",
    "interleave", "microbatch", "num_microbatches",
])
def test_parallel_scalar_bool_false_rejected(field):
    """bool False（==0）此前被 `<1` 拒，但应以「非整数类型」而非「<1」语义拒；仍须 fail-loud。"""
    with pytest.raises(ValueError, match=field):
        ParallelConfig(**{field: False})


def test_prefetch_depth_bool_rejected():
    with pytest.raises(ValueError, match="prefetch_depth"):
        ParallelConfig(prefetch_depth=True)
    with pytest.raises(ValueError, match="prefetch_depth"):
        ParallelConfig(prefetch_depth=False)


def test_parallel_bool_error_carries_type_and_field():
    with pytest.raises(ValueError) as ei:
        ParallelConfig(tp=True)
    msg = str(ei.value)
    assert "tp" in msg and "bool" in msg


# ── 正向守卫：默认构造与合法 int 必须继续通过 ─────────────────────────────────
def test_default_parallel_construction_passes():
    ParallelConfig()                         # 全 int 1


def test_legal_int_parallel_configs_pass():
    ParallelConfig(tp=8, dp_shard=8, cp=1, ep=4, pp=1, sequence_parallel=True)
    ParallelConfig(dp_shard=2, pp=2, interleave=2, num_microbatches=4)
    ParallelConfig(prefetch_depth=0)         # 0=无预取合法
    ParallelConfig(prefetch_depth=3)


# ══════════════════════════════════════════════════════════════════════════════
# F5b — 核心/整数维度：bool 当维度必须被拒（build_llm._validate_structure）
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("field", [
    "num_layers", "hidden_size", "num_attention_heads",
    "vocab_size", "seq_length", "batch_size",
])
def test_core_dim_bool_rejected(field):
    """bool 维度 → 旧校验只比 `<1`，`True` 放行产 DimTable(H=True) 错图；须 fail-loud。"""
    with pytest.raises(ValueError, match=field):
        build_llm_spec(_dsv3(**{field: True}, head_dim=192))


def test_core_dim_bool_error_carries_type():
    with pytest.raises(ValueError) as ei:
        build_llm_spec(_dsv3(hidden_size=True, head_dim=192))
    assert "bool" in str(ei.value)


def test_ffn_hidden_size_bool_rejected():
    with pytest.raises(ValueError, match="ffn_hidden_size"):
        build_llm_spec(_dsv3(ffn_hidden_size=True, head_dim=192))


def test_moe_ffn_hidden_size_bool_rejected():
    with pytest.raises(ValueError, match="moe_ffn_hidden_size"):
        build_llm_spec(_dsv3(moe_ffn_hidden_size=True))


def test_num_moe_experts_bool_rejected():
    """num_moe_experts=True 是 truthy → 进 MoE 路径当 True 个专家；须以类型守卫拒。

    topk=1 时不触发 `topk>experts`(1>1 False) / `topk<=0`(False) → 只有类型守卫能拦，
    保证测的是 bool 守卫本身而非相邻数值检查。"""
    with pytest.raises(ValueError) as ei:
        build_llm_spec(_dsv3(num_moe_experts=True, moe_router_topk=1))
    assert "num_moe_experts" in str(ei.value) and "bool" in str(ei.value)


@pytest.mark.parametrize("field", [
    "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
])
def test_mla_dim_bool_rejected(field):
    """MLA 家族维（mla/dsv4_hybrid/dsa）用 `<=0` 校验，bool True 放行；须拒。"""
    with pytest.raises(ValueError, match=field):
        build_llm_spec(_dsv3(**{field: True}))


# ── None 惰性维度守卫：非 None 才查类型，None 保持合法 ───────────────────────
def test_ffn_hidden_size_none_still_allowed():
    assert build_llm_spec(_min_gqa(ffn_hidden_size=None))


def test_head_dim_none_still_allowed():
    """head_dim=None（惰性 H//n_heads）合法，不该被类型守卫误伤。"""
    assert build_llm_spec(_dsv3())     # deepseek_v3 head_dim 默认 None


# ── 正向守卫：合法 int 维度 + 所有预设仍 build ────────────────────────────────
def test_all_presets_still_build_under_type_guard():
    for cfg in (deepseek_v3(4), deepseek_v4(4), llama(num_layers=2),
                qwen2(num_layers=2), mixtral(num_layers=2)):
        assert build_llm_spec(cfg), f"预设 build 失败：{cfg.attn_type}"


def test_legal_int_core_dims_pass():
    assert build_llm_spec(_min_gqa(hidden_size=16, num_attention_heads=4, head_dim=4))


# ══════════════════════════════════════════════════════════════════════════════
# F5c — CSA 压缩比：非整数值必须被拒（拒绝静默 int() 截断）
# ══════════════════════════════════════════════════════════════════════════════
def test_csa_ratio_float_49_rejected():
    """4.9 此前 int(4.9)=4 静默变 dsv4hyb_r4_* 图；须以「非整数压缩比」fail-loud。"""
    with pytest.raises(ValueError, match="csa_compress_ratios|压缩比"):
        build_llm_spec(_dsv4(csa_compress_ratios=(4.9, 1, 1, 4)))


def test_csa_ratio_float_non_integer_various_rejected():
    for ratios in ((3.5, 1, 1, 4), (0, 1, 1, 127.5), (1.1, 1, 1, 4)):
        with pytest.raises(ValueError):
            build_llm_spec(_dsv4(csa_compress_ratios=ratios))


def test_csa_ratio_bool_rejected():
    """bool 混进 ratios（int(True)=1）也应以非整数类型拒。"""
    with pytest.raises(ValueError):
        build_llm_spec(_dsv4(csa_compress_ratios=(True, 1, 1, 4)))


def test_csa_ratio_float_carries_value():
    with pytest.raises(ValueError, match="4.9"):
        build_llm_spec(_dsv4(csa_compress_ratios=(4.9, 1, 1, 4)))


# ── 正向守卫：整数值（含 4.0 这种整数浮点）与合法 int 档必须通过 ────────────────
def test_csa_ratio_int_values_pass():
    assert build_llm_spec(_dsv4(csa_compress_ratios=(0, 4, 128, 0)))
    assert build_llm_spec(_dsv4(csa_compress_ratios=(0, 1, 4, 128)))


def test_csa_ratio_integer_valued_float_accepted():
    """4.0 是整数值浮点 → float.is_integer() 为真 → 接受（≠ 4.9）。"""
    assert build_llm_spec(_dsv4(csa_compress_ratios=(0.0, 4.0, 128.0, 0.0)))


def test_dsv4_default_preset_csa_still_builds():
    """deepseek_v4 预设自带 csa_compress_ratios（全 int）必须继续 build。"""
    assert build_llm_spec(deepseek_v4(4))
    assert build_llm_spec(deepseek_v4(8))


# ── 既有非法整数档守卫不得被删（回归护栏）─────────────────────────────────────
def test_existing_illegal_int_ratio_still_rejected():
    """2/3 等非实现整数压缩比仍须拒（不因新类型守卫回退）。"""
    with pytest.raises((ValueError, NotImplementedError)):
        build_llm_spec(_dsv4(csa_compress_ratios=(2, 1, 1, 4)))
