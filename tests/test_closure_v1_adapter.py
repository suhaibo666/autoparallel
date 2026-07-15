"""三轮闭环审计（closure_audit_verification_2026-07-15.md §4.1/§4.4/§4.10）反例转正式回归。

只针对 adapter `cost_eval/configs/from_mindformers.py` 的三组缺陷（探针实证）：

① P0-02 adapter 静默丢语义（§4.1）：
   1. bool True 泄漏（§4.1.1）——`_PAR_UNSUPPORTED_TRUTHY` 允许集含 `1`,而 Python `True==1`,
      致 `pipeline_parallel_overlap_p2p=True` 静默放行；数值键 `dense_fsdp_shard_size` 需与布尔键分开
      （=1 合法、>1 才拒）。
   2. optimizer 段无 schema（§4.1.2）——`tyep` 键typo / `optimzier` 顶层段typo 静默回落 AdamW。
   3. dropout 只拒 >0（§4.1.4）——`attention_dropout=-0.1` 非法非零被接受。

② P1-06 CE provenance（§4.4）：无显式键时 dsv4_hybrid→True 的架构默认须**可追溯**（发 warning）、
   可被显式键覆盖；DSv4 锚点 lean=True 不破。

③ P2-08 qk_layernorm（§4.10.2）：qk_layernorm 在忽略集（内存中性、刻意不建），adapter 不映射它
   → yaml qk_layernorm=true 不传到 LLMConfig（保持默认 False）。
"""
import warnings

import pytest

from cost_eval.configs.from_mindformers import (
    _IGNORED_MODEL_KEYS,
    _MAPPED_MODEL_KEYS,
    from_mindformers_dict,
)


# ── 最小合法 mindformers dict（mha、无 MoE；已知可成功构建，见 test_closure_audit C2）──────
def _mf(**over):
    m = {"model": {"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 1024,
                   "vocab_size": 1000, "seq_length": 512, "compute_dtype": "bfloat16"},
         "training": {"local_batch_size": 1}}
    for k, v in over.items():
        m[k] = v
    return m


def _dsv4_mf(dsa_fused=True):
    """dsv4_hybrid 最小 dict（走 CE 架构默认路径）。"""
    return {"model": {"num_hidden_layers": 4, "num_attention_heads": 128, "hidden_size": 2048,
                      "vocab_size": 1000, "seq_length": 2048, "multi_latent_attention": True,
                      "experimental_attention_variant": "dsv4_hybrid",
                      "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                      "qk_nope_head_dim": 128, "v_head_dim": 128, "o_groups": 16, "o_lora_rank": 256,
                      "dsa_indexer_n_heads": 64, "dsa_indexer_head_dim": 128,
                      "dsa_indexer_topk": 2048, "csa_compress_ratios": [1, 1, 1, 4],
                      "apply_dsa_kernel_fusion": dsa_fused, "force_unfused_dsa": not dsa_fused,
                      "compute_dtype": "bfloat16", "n_routed_experts": 8,
                      "num_experts_per_tok": 2, "moe_intermediate_size": 2048},
            "training": {"local_batch_size": 1}}


# ── ① P0-02.1 bool True 泄漏（§4.1.1）────────────────────────────────────────────────
def test_par_overlap_p2p_true_rejected():
    """`pipeline_parallel_overlap_p2p=True`（未建模布尔键）必须 fail-loud——此前 True==1 静默放行。"""
    with pytest.raises(NotImplementedError, match="pipeline_parallel_overlap_p2p"):
        from_mindformers_dict(_mf(parallelism={"pipeline_parallel_overlap_p2p": True}))


def test_par_overlap_p2p_int_one_rejected():
    """整数 1 也是真值（True==1）——必须与 True 同拒,不得漏过。"""
    with pytest.raises(NotImplementedError, match="pipeline_parallel_overlap_p2p"):
        from_mindformers_dict(_mf(parallelism={"pipeline_parallel_overlap_p2p": 1}))


def test_par_context_parallel_async_true_rejected():
    with pytest.raises(NotImplementedError, match="context_parallel_async"):
        from_mindformers_dict(_mf(parallelism={"context_parallel_async": True}))


def test_par_unsupported_bool_false_accepted():
    """False/0（禁用）放行——只拒真值,不误伤显式关闭。"""
    assert from_mindformers_dict(_mf(parallelism={"pipeline_parallel_overlap_p2p": False}))
    assert from_mindformers_dict(_mf(parallelism={"context_parallel_async": 0}))


# ── ① P0-02.1 数值键 dense_fsdp_shard_size（=1 合法、>1 拒）────────────────────────────
def test_dense_fsdp_shard_size_gt1_rejected():
    with pytest.raises(NotImplementedError, match="dense_fsdp_shard_size"):
        from_mindformers_dict(_mf(parallelism={"dense_fsdp_shard_size": 2}))


def test_dense_fsdp_shard_size_one_accepted():
    """=1 = 不分组切分,合法（数值键,区别于布尔真值泄漏）——此前被误拒。"""
    assert from_mindformers_dict(_mf(parallelism={"dense_fsdp_shard_size": 1}))


# ── ① P0-02.2 optimizer 段 schema（§4.1.2）────────────────────────────────────────────
def test_optimizer_key_typo_rejected():
    """optimizer 段内 `tyep` typo → fail-loud,不静默回落 AdamW。"""
    with pytest.raises(NotImplementedError, match="optimizer"):
        from_mindformers_dict(_mf(optimizer={"tyep": "SGD", "learning_rate": 1e-4}))


def test_optimizer_known_keys_accepted():
    assert from_mindformers_dict(_mf(optimizer={
        "type": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8,
        "weight_decay": 0.01, "learning_rate": 1e-4}))


# ── ① P0-02.2 顶层段 schema（catches `optimzier` 整段拼错）─────────────────────────────
def test_top_level_segment_typo_rejected():
    """顶层段名 `optimzier` 拼错 → 整段静默丢失、回落 AdamW 默认；顶层 schema fail-loud。"""
    m = _mf()
    m["optimzier"] = {"type": "AdamW"}
    with pytest.raises(NotImplementedError, match="optimzier"):
        from_mindformers_dict(m)


def test_legacy_top_level_segments_not_misfired():
    """老式 mindformers 顶层段（经 _mf_adapt 保留在 dict 里）不得被顶层 schema 误伤（防回归）。"""
    m = _mf()
    m.update(parallel_config={}, parallel={}, runner_config={}, recompute_config={},
             callbacks=[], lr_schedule={}, seed=42, output_dir="x", use_parallel=True,
             context={"max_device_memory": "54GB"})
    assert from_mindformers_dict(m)


# ── ① P0-02.3 dropout 拒一切非零（§4.1.4）─────────────────────────────────────────────
def test_negative_attention_dropout_rejected():
    """`attention_dropout=-0.1` 非法非零 → fail-loud（此前只拒 >0,负值漏过）。"""
    m = _mf()
    m["model"]["attention_dropout"] = -0.1
    with pytest.raises(NotImplementedError, match="dropout"):
        from_mindformers_dict(m)


def test_negative_hidden_dropout_rejected():
    m = _mf()
    m["model"]["hidden_dropout"] = -0.5
    with pytest.raises(NotImplementedError, match="dropout"):
        from_mindformers_dict(m)


def test_zero_dropout_accepted():
    m = _mf()
    m["model"]["attention_dropout"] = 0.0
    m["model"]["hidden_dropout"] = 0
    assert from_mindformers_dict(m)


# ── ② P1-06 CE provenance（§4.4）──────────────────────────────────────────────────────
def test_dsv4_arch_default_ce_true_with_provenance_warning():
    """无显式键 → dsv4_hybrid 架构兼容默认 True(lean CE,锚点不破),且发 provenance warning。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = from_mindformers_dict(_dsv4_mf())
    assert bundle.llm.cross_entropy_fused is True
    assert any("cross_entropy_fused" in str(w.message) for w in caught), \
        "架构默认必须发 provenance 警告（可追溯）"


def test_dsv4_ce_default_independent_of_dsa_fusion():
    """切 DSA 融合不改 CE（DSv4 恒 lean，独立于 attention kernel）——锚点 lean 双向不破。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = from_mindformers_dict(_dsv4_mf(True)).llm.cross_entropy_fused
        off = from_mindformers_dict(_dsv4_mf(False)).llm.cross_entropy_fused
    assert on is True and off is True


def test_explicit_ce_key_overrides_without_warning():
    """显式 cross_entropy_fused 覆盖架构默认,且不发 provenance 警告（已可追溯）。"""
    mf = _dsv4_mf()
    mf["model"]["cross_entropy_fused"] = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = from_mindformers_dict(mf)
    assert bundle.llm.cross_entropy_fused is False
    assert not any("cross_entropy_fused" in str(w.message) for w in caught)


def test_non_dsv4_ce_false_no_warning():
    """非 dsv4 模型 → CE False（pynative unfused-fat 常态），无 provenance 警告。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = from_mindformers_dict(_mf())
    assert bundle.llm.cross_entropy_fused is False
    assert not any("cross_entropy_fused" in str(w.message) for w in caught)


# ── ③ P2-08 qk_layernorm 分类订正（§4.10.2）──────────────────────────────────────────
def test_qk_layernorm_in_ignored_not_mapped():
    """qk_layernorm 归**忽略集**（内存中性,刻意不建）,不在已映射集——校正 P2-08 文档漂移。"""
    assert "qk_layernorm" in _IGNORED_MODEL_KEYS
    assert "qk_layernorm" not in _MAPPED_MODEL_KEYS


def test_qk_layernorm_true_not_propagated_to_llmconfig():
    """yaml qk_layernorm=True 不映射 → LLMConfig.qk_layernorm 保持默认 False（adapter 刻意丢弃）。
    这正是订正后注释描述的真实不变量（旧注释误称由 build_llm 守卫）。"""
    m = _mf()
    m["model"]["qk_layernorm"] = True
    bundle = from_mindformers_dict(m)
    assert bundle.llm.qk_layernorm is False
