"""X3 任务 A — Qwen3 qk-layernorm 建模（把 build_llm fail-loud 变真评估）。

真机（mindformers/pynative/transformers/attention.py:311-357）：Qwen3 的 SelfAttention 在
linear_qkv 投影后、rope 前，对**带 head 维的 Q/K** 各作一个 RMSNorm（`dim=head_dim`，
per-head、作用在 head_dim 上，compute=layernorm_compute_dtype=fp32）。此前 GQA/MHA op-builder
（attention.py `build_gqa_attn_ops`）无 q/k norm，且 build_llm 对 `qk_layernorm=True` fail-loud。

本组测试驱动：
  - `build_gqa_attn_ops` 在 `getattr(d,"qk_layernorm",False)` 为真时插入 q_norm/k_norm 两个
    NORM op（qkv 投影后、rope 前），各带 [head_dim] fp32 gamma、saves 各自 per-head 输入切片；
  - qk off（默认）逐字节不变（6 op，无 q/k norm）——保 DSv3(mla)/DSv4 锚点；
  - build_llm dispatch 守卫对 gqa/mha + qk_layernorm=True **放行**，非 gqa/mha 仍 fail-loud；
  - qk-norm 的 per-head 切片按 norm_compute fp32 计（与 ln1 同）、随 cp ÷cp（不破 CP halving）。
"""
import dataclasses

import pytest

from cost_eval.model_spec import DimTable, OpType
from cost_eval.layers.attention import build_gqa_attn_ops
from cost_eval.shape_eval import ShapeEval, resolve_tensor, eval_expr
from cost_eval.structure_mem import estimate_structure_memory
from cost_eval.parallel_model import ParallelModel
from cost_eval.specs import ParallelConfig
from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec


def _D(qk=False, **over):
    """小 GQA DimTable；qk=True 时打开 qk_layernorm（用 setattr 走 builder 的 getattr 读取路径）。"""
    base = dict(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=8, B=1, vocab=16, n_layers=3)
    base.update(over)
    d = DimTable(**base)
    d.qk_layernorm = qk
    return d


def _names(ops):
    return [op.name for op in ops]


# ── qk OFF：逐字节不变（保 DSv3/DSv4 锚点）─────────────────────────────────────────────
def test_qk_off_is_unchanged_six_ops():
    ops = build_gqa_attn_ops(_D(qk=False))
    assert _names(ops) == ["ln1", "qkv", "rope", "flash", "o_proj", "add1"]
    assert not any("q_norm" in n or "k_norm" in n for n in _names(ops))


def test_qk_off_default_when_field_absent():
    """DimTable 无 qk_layernorm（不显式设）时惰性——getattr 兜底 False，不建 q/k norm。"""
    base = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=8, B=1, vocab=16, n_layers=3)
    # 不设 base.qk_layernorm；若字段存在则默认 False，若不存在则 getattr 兜底 False
    if hasattr(base, "qk_layernorm"):
        object.__setattr__(base, "qk_layernorm", False)
    ops = build_gqa_attn_ops(base)
    assert len(ops) == 6


# ── qk ON：插入 q_norm / k_norm 两个 NORM op，位于 qkv 后、rope 前 ─────────────────────────
def test_qk_on_inserts_q_and_k_norm_ops():
    ops = build_gqa_attn_ops(_D(qk=True))
    names = _names(ops)
    assert "q_norm" in names and "k_norm" in names
    # 位置：qkv 之后、rope 之前
    assert names.index("qkv") < names.index("q_norm") < names.index("rope")
    assert names.index("qkv") < names.index("k_norm") < names.index("rope")
    # 两者均为 NORM type
    q = next(op for op in ops if op.name == "q_norm")
    k = next(op for op in ops if op.name == "k_norm")
    assert q.type == OpType.NORM and k.type == OpType.NORM
    # 尾 op 仍是 add1、输出 h1（可拼 FFN）
    assert ops[-1].name == "add1" and ops[-1].output.name == "h1"


def test_qk_on_gamma_is_per_head_fp32():
    """q/k norm 的 gamma 权重 = [head_dim]、fp32(4B)、is_weight（真机 dim=head_dim，per-head 共享）。"""
    ops = build_gqa_attn_ops(_D(qk=True))
    for name in ("q_norm", "k_norm"):
        op = next(o for o in ops if o.name == name)
        assert len(op.params) == 1
        g = op.params[0]
        assert g.is_weight is True
        assert g.shape == ("head_dim",)
        assert g.dtype_bytes == 4                 # fp32 norm gamma（真机独立 FSDP wrap）


def test_qk_on_saves_per_head_slices():
    """q_norm 保留 Q 切片 [S,B,n_heads·head_dim]、k_norm 保留 K 切片 [S,B,n_kv·head_dim]（反向所需的 fp32 cast）。"""
    d = _D(qk=True)
    ops = build_gqa_attn_ops(d)
    q = next(o for o in ops if o.name == "q_norm")
    k = next(o for o in ops if o.name == "k_norm")
    assert len(q.saves) == 1 and len(k.saves) == 1
    assert eval_expr("*".join(q.saves[0].shape), d) == d.S * d.B * d.n_heads * d.head_dim
    assert eval_expr("*".join(k.saves[0].shape), d) == d.S * d.B * d.n_kv * d.head_dim


def test_qk_on_saves_counted_fp32_under_norm_compute():
    """norm_compute_dtype=fp32 下，q/k norm 的保留切片按 fp32 计（同 ln1）；关掉则不额外放大。
    fp32 增量 = 4·S·B·(n_heads+n_kv)·head_dim（与真机 Qwen3 每层 BF16→FP32 2× 比一致）。"""
    d = _D(qk=True)
    spec_ops = build_gqa_attn_ops(d)
    pm = ParallelModel(ParallelConfig(), n_layers=d.n_layers, world_size=1)
    from cost_eval.model_spec import ModelSpec, LayerSpec
    spec = ModelSpec("t", d, ["l"], {"l": LayerSpec(spec_ops)})
    ops = ShapeEval().resolve(spec, pm).stages[0][0].ops
    on = build_gqa_attn_ops(_D(qk=True))
    off = build_gqa_attn_ops(_D(qk=False))
    spec_off = ModelSpec("t2", _D(qk=False), ["l"], {"l": LayerSpec(off)})
    ops_off = ShapeEval().resolve(spec_off, pm).stages[0][0].ops
    saves_fp32_on = estimate_structure_memory(ops, norm_compute_dtype_bytes=4).activation_saves
    saves_fp32_off = estimate_structure_memory(ops_off, norm_compute_dtype_bytes=4).activation_saves
    delta = saves_fp32_on - saves_fp32_off
    assert delta == 4 * d.S * d.B * (d.n_heads + d.n_kv) * d.head_dim


# ── build_llm dispatch：gqa/mha 放行、非 gqa/mha 仍 fail-loud ─────────────────────────────
def _gqa_cfg(**over):
    base = dict(num_layers=2, hidden_size=8, num_attention_heads=2, num_query_groups=2,
                vocab_size=16, seq_length=8, batch_size=1, head_dim=4, attn_type="gqa",
                ffn_hidden_size=16)
    base.update(over)
    return LLMConfig(**base)


def test_build_llm_allows_gqa_qk_layernorm_and_builds_norm_ops():
    spec = build_llm_spec(_gqa_cfg(qk_layernorm=True))
    # decoder 层含 q_norm / k_norm（端到端：cfg → to_dimtable(qk 直通) → builder getattr 读取）
    dec = next(v for k, v in spec.layer_specs.items() if k not in ("embedding", "lm_head")
               and any(o.name == "flash" for o in v.ops))
    names = [o.name for o in dec.ops]
    assert "q_norm" in names and "k_norm" in names


def test_build_llm_mha_qk_layernorm_also_builds():
    """mha 复用 gqa builder（对称 GQA）→ 同样放行建 q/k norm。"""
    spec = build_llm_spec(_gqa_cfg(attn_type="mha", qk_layernorm=True))
    dec = next(v for k, v in spec.layer_specs.items()
               if any(o.name == "flash" for o in v.ops))
    assert "q_norm" in [o.name for o in dec.ops]


def test_build_llm_non_gqa_qk_layernorm_still_fail_loud():
    """mla/dsv4/dsa 的 attn builder 未建 q/k norm → qk_layernorm=True 仍 fail-loud（不静默产错图）。"""
    mla = _gqa_cfg(attn_type="mla", q_lora_rank=4, kv_lora_rank=4, qk_rope_head_dim=2,
                   qk_nope_head_dim=2, v_head_dim=4, qk_layernorm=True)
    with pytest.raises(NotImplementedError, match="qk_layernorm"):
        build_llm_spec(mla)


def test_build_llm_gqa_qk_off_unchanged():
    """gqa + qk_layernorm=False → 无 q/k norm（默认路径逐字节不变）。"""
    spec = build_llm_spec(_gqa_cfg(qk_layernorm=False))
    dec = next(v for k, v in spec.layer_specs.items()
               if any(o.name == "flash" for o in v.ops))
    assert "q_norm" not in [o.name for o in dec.ops]


# ── CP 一致性：q/k norm 切片随 cp ÷cp（Task A 不破 CP halving）─────────────────────────────
def test_qk_norm_slices_shard_under_cp():
    d = _D(qk=True)
    ops = build_gqa_attn_ops(d)
    q = next(o for o in ops if o.name == "q_norm")
    pm1 = ParallelModel(ParallelConfig(cp=1), n_layers=d.n_layers, world_size=1)
    pm2 = ParallelModel(ParallelConfig(cp=2), n_layers=d.n_layers, world_size=2)
    n1 = resolve_tensor(q.saves[0], d, pm1).local_numel
    n2 = resolve_tensor(q.saves[0], d, pm2).local_numel
    assert n1 == d.S * d.B * d.n_heads * d.head_dim
    assert n2 * 2 == n1                          # 序列维 ÷cp，与 body 一致
