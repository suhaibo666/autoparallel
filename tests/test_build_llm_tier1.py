"""Task 1.2：build_llm_spec 装配器（Tier-1）+ head/loss builders 回归。

核心断言（预示 1.3 硬门）：装配出的 `mla_dense`/`mla_moe` LayerSpec 的 op 序列
**逐字段**等于现有 `build_mla_dense_decoder`/`build_mla_moe_decoder`；embedding/lm_head
逐字段等于 `validate_dsv3.build_embedding`/`build_lm_head`。
"""
from cost_eval.build_llm import build_llm_spec, gen_layer_pattern
from cost_eval.llm_config import LLMConfig, to_dimtable
from cost_eval.model_spec import ModelSpec
from cost_eval.layers.mla import build_mla_dense_decoder, build_mla_moe_decoder
from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from cost_eval.report import Evaluator
import validate_dsv3


def _dsv3_like_cfg(num_layers=3):
    """DSv3 缩层结构：mla + moe(first_k_dense=1) + shared expert（复现锚点结构）。"""
    return LLMConfig(
        num_layers=num_layers, hidden_size=1792, num_attention_heads=8,
        vocab_size=129280, seq_length=4096, batch_size=1, head_dim=192,
        attn_type="mla",
        q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64,
        qk_nope_head_dim=128, v_head_dim=192,
        ffn_hidden_size=3072,
        num_moe_experts=8, moe_router_topk=4, moe_ffn_hidden_size=1024,
        moe_shared_expert_num=1, moe_shared_ffn_hidden_size=1024,
        first_k_dense_replace=1,
        compute_dtype_bytes=2,
    )


def _names(ops):
    return [o.name for o in ops]


def test_returns_modelspec_with_matching_pattern():
    cfg = _dsv3_like_cfg(3)
    spec = build_llm_spec(cfg)
    assert isinstance(spec, ModelSpec)
    assert spec.layer_pattern == gen_layer_pattern(cfg)
    assert spec.layer_pattern == ["embedding", "mla_dense", "mla_moe", "mla_moe", "lm_head"]
    # DimTable 与 validate_dsv3.build_dsv3_spec(3) 结构一致
    assert spec.dims.n_layers == 5 and spec.dims.n_experts == 8 and spec.dims.n_shared == 1


def test_mla_moe_layer_matches_decoder_full_field():
    cfg = _dsv3_like_cfg(3)
    dims = to_dimtable(cfg)
    spec = build_llm_spec(cfg)
    ref = build_mla_moe_decoder(dims)
    got = spec.layer_specs["mla_moe"]
    assert _names(got.ops) == _names(ref.ops)      # 名字一致
    assert got.ops == ref.ops                       # 全字段一致（1.3 硬门前提）


def test_mla_dense_layer_matches_decoder_full_field():
    cfg = _dsv3_like_cfg(3)
    dims = to_dimtable(cfg)
    spec = build_llm_spec(cfg)
    ref = build_mla_dense_decoder(dims)
    got = spec.layer_specs["mla_dense"]
    assert _names(got.ops) == _names(ref.ops)
    assert got.ops == ref.ops


def test_embedding_and_head_match_validate_dsv3():
    cfg = _dsv3_like_cfg(3)
    dims = to_dimtable(cfg)
    spec = build_llm_spec(cfg)
    emb_ref = validate_dsv3.build_embedding(dims)
    head_ref = validate_dsv3.build_lm_head(dims)
    assert _names(spec.layer_specs["embedding"].ops) == _names(emb_ref.ops)
    assert _names(spec.layer_specs["lm_head"].ops) == _names(head_ref.ops)
    # verbatim 移植：全字段一致
    assert spec.layer_specs["embedding"].ops == emb_ref.ops
    assert spec.layer_specs["lm_head"].ops == head_ref.ops


def test_spec_evaluates_without_error():
    cfg = _dsv3_like_cfg(3)
    spec = build_llm_spec(cfg)
    ev = Evaluator(
        spec,
        ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1,
                       sequence_parallel=True, num_microbatches=1),
        OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
        HardwareSpec(max_device_memory=59 * 2 ** 30),
        RecomputeSpec(mode="full", full_layers=set(range(1, cfg.num_layers + 1))),
        SwapSpec(),
    )
    rep = ev.evaluate()
    assert rep.per_stage[0].peak_bytes > 0
