"""D-7：mindformers 配置文件 → 评估器配置对象 转换器测试。

保真判据（核心）：把**规范 mindformers dict**（重构 `prep_dsv4align.py` / `prep_ds3_sim.py`）
喂进 `from_mindformers_dict`，得到的 `LLMConfig` 与 `dsv4_align_config()` / `deepseek_v3()`
**逐字段相等** → 同一评估峰值（DSv4-align 14336.5 / DSv3 12409.5）。这证明映射忠实、非杜撰。

覆盖：(a) DSv4-align round-trip + 峰值；(b) DSv3 round-trip + 峰值；(c) 并行/重算/layers_per_stage
映射；(d) fail-loud（未映射结构字段 / flash=False / 非法 attn 变体 / fusion 自相矛盾）；(e) 惰性
yaml path-loader。
"""
import dataclasses

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.configs import (
    EvaluatorConfigBundle,
    from_mindformers_dict,
    load_mindformers_yaml,
)
from cost_eval.presets import deepseek_v3
from cost_eval.report import Evaluator
from validate_dsv4align import dsv4_align_config

MiB = 2 ** 20
GiB = 2 ** 30


# ── 规范 mindformers dict 重构（逐字段镜像 prep_*.py）─────────────────────────────────
def _dsv4align_mf(n=4, seq=2048, mhc=0, mtp=0, fused=True):
    """重构 `prep_dsv4align.py` 生成的 dict（FUSED=生产 → dsa_fused=True，对齐真机 FUSED 锚点）。"""
    cyc = [0, 4, 128]
    compress = [cyc[i % 3] for i in range(n)] + [0] * mtp
    model = {
        "model_type": "deepseek_v3", "architectures": "DeepseekV3ForCausalLM",
        "vocab_size": 129280, "seq_length": seq, "hidden_size": 1792, "intermediate_size": 3072,
        "num_hidden_layers": n, "max_position_embeddings": 163840, "hidden_act": "silu",
        "num_attention_heads": 64, "rms_norm_eps": 1.0e-6, "add_bias_linear": False,
        "use_flash_attention": True, "multi_latent_attention": True, "mla_qkv_concat": False,
        "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 64, "qk_nope_head_dim": 448,
        "v_head_dim": 512, "qk_layernorm": True, "attention_dropout": 0.0, "hidden_dropout": 0.0,
        "params_dtype": "float32", "compute_dtype": "bfloat16", "layernorm_compute_dtype": "float32",
        "softmax_compute_dtype": "float32", "rotary_dtype": "float32", "initializer_range": 0.01,
        "experimental_attention_variant": "dsv4_hybrid",
        "apply_dsa_kernel_fusion": fused, "force_unfused_dsa": not fused,
        "csa_compress_ratios": compress, "csa_window_size": 128,
        "csa_compress_rotary_base": 40000.0, "csa_dense_mode": False,
        "dsa_indexer_n_heads": 64, "dsa_indexer_head_dim": 128, "dsa_indexer_topk": 512,
        "dsa_indexer_loss_coeff": 0.001, "dsa_indexer_use_sparse_loss": True,
        "o_groups": 8, "o_lora_rank": 1024,
        "enable_hyper_connections": mhc == 1, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1.0e-6,
        "use_fused_mhc": False, "num_nextn_predict_layers": mtp, "mtp_loss_scaling_factor": 0.3,
        "position_embedding_type": "yarn", "scaling_factor": 40, "beta_fast": 32, "beta_slow": 1,
        "mscale": 1, "mscale_all_dim": 1, "rope_theta": 10000,
        "router_dense_type": "float32", "gated_linear_unit": True, "moe_intermediate_size": 1024,
        "routed_scaling_factor": 2.5, "first_k_dense_replace": 1, "n_routed_experts": 4,
        "num_experts_per_tok": 2, "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 1024,
        "moe_token_dispatcher_type": "alltoall", "moe_grouped_gemm": True,
        "moe_router_load_balancing_type": "seq_aux_loss", "moe_aux_loss_coeff": 0.001,
        "scoring_func": "sigmoid", "norm_topk_prob": True, "moe_token_drop_policy": "probs",
        "moe_router_enable_expert_bias": True, "moe_router_bias_update_rate": 0.001,
        "use_pad_tokens": True, "topk_group": 2, "n_group": 2,
    }
    return {
        "context": {"device_target": "Ascend", "max_device_memory": "54GB", "mode": 1},
        "training": {"steps": 3, "local_batch_size": 1, "global_batch_size": 2},
        "optimizer": {"type": "AdamW", "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.01},
        "parallelism": {
            "data_parallel_shard": -1, "data_parallel_shard_strategy": "optim_grads_params",
            "disable_gradient_division": True, "expert_parallel": 1, "tensor_parallel": 1,
            "context_parallel": 1, "pipeline_parallel": 1, "pipeline_parallel_microbatch_size": 1,
            "sequence_parallel": True,
        },
        "model": model,
    }


def _dsv3_mf(n=4, seq=4096):
    """重构 DSv3 缩层 mindformers dict（映射到 `deepseek_v3(n)`）——`multi_latent_attention` 无 dsv4 变体
    → attn_type=mla；`num_key_value_heads=8`（→ num_query_groups=8）；MLA dims 128/64/192。"""
    model = {
        "model_type": "deepseek_v3", "architectures": "DeepseekV3ForCausalLM",
        "vocab_size": 129280, "seq_length": seq, "hidden_size": 1792, "intermediate_size": 3072,
        "num_hidden_layers": n, "num_attention_heads": 8, "num_key_value_heads": 8,
        "hidden_act": "silu", "rms_norm_eps": 1.0e-6, "add_bias_linear": False,
        "use_flash_attention": True, "multi_latent_attention": True,
        "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 64, "qk_nope_head_dim": 128,
        "v_head_dim": 192, "params_dtype": "float32", "compute_dtype": "bfloat16",
        "position_embedding_type": "rope", "gated_linear_unit": True, "moe_intermediate_size": 1024,
        "first_k_dense_replace": 1, "n_routed_experts": 8, "num_experts_per_tok": 4,
        "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 1024,
    }
    return {
        "context": {"device_target": "Ascend", "max_device_memory": "59GB", "mode": 1},
        "training": {"steps": 3, "local_batch_size": 1, "global_batch_size": 2},
        "optimizer": {"type": "AdamW", "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.01},
        "parallelism": {
            "data_parallel_shard": -1, "expert_parallel": 1, "tensor_parallel": 1,
            "context_parallel": 1, "pipeline_parallel": 1, "pipeline_parallel_microbatch_size": 1,
            "sequence_parallel": True,
        },
        "recompute": {"mode": "full", "full_recompute_layer": [f"0-{n - 1}"]},
        "model": model,
    }


def _peak(bundle: EvaluatorConfigBundle) -> float:
    spec = build_llm_spec(bundle.llm)
    ev = Evaluator(spec, bundle.parallel, bundle.optimizer, bundle.hardware,
                   bundle.recompute, bundle.swap)
    return ev.evaluate().per_stage[0].peak_bytes / MiB


def _assert_llm_field_equal(got, ref):
    """逐字段相等（失败时打印所有分叉字段，便于定位）。"""
    diffs = [(f.name, getattr(got, f.name), getattr(ref, f.name))
             for f in dataclasses.fields(ref)
             if getattr(got, f.name) != getattr(ref, f.name)]
    assert not diffs, f"LLMConfig 字段分叉：{diffs}"


# ── (a) DSv4-align round-trip ─────────────────────────────────────────────────────────
def test_dsv4align_roundtrip_llmconfig_field_for_field():
    bundle = from_mindformers_dict(_dsv4align_mf())
    _assert_llm_field_equal(bundle.llm, dsv4_align_config(4))
    assert bundle.llm == dsv4_align_config(4)


def test_dsv4align_roundtrip_same_peak_14336():
    bundle = from_mindformers_dict(_dsv4align_mf())
    # 与 validate_dsv4align.evaluate 同 bundle（dp_shard=2 / no-recompute / SP）→ 逐字节同峰。
    assert bundle.parallel.dp_shard == 2 and bundle.recompute.mode == "None"
    assert abs(_peak(bundle) - 14336.5) < 1.0


# ── (b) DSv3 round-trip ───────────────────────────────────────────────────────────────
def test_dsv3_roundtrip_llmconfig_field_for_field():
    bundle = from_mindformers_dict(_dsv3_mf())
    _assert_llm_field_equal(bundle.llm, deepseek_v3(4))
    assert bundle.llm == deepseek_v3(4)
    # 关键推断：mla + head_dim=nope+rope=192 + num_query_groups=8。
    assert bundle.llm.attn_type == "mla"
    assert bundle.llm.head_dim == 192 and bundle.llm.num_query_groups == 8


def test_dsv3_roundtrip_same_peak_12409():
    bundle = from_mindformers_dict(_dsv3_mf())
    assert bundle.recompute.mode == "full" and bundle.recompute.full_layers == {1, 2, 3, 4}
    assert abs(_peak(bundle) - 12409.5) < 1.0


def test_dsv3_8L_roundtrip_peak_matches_preset():
    bundle = from_mindformers_dict(_dsv3_mf(n=8))
    assert bundle.llm == deepseek_v3(8)


# ── (c) 并行 / 重算 / layers_per_stage 映射 ────────────────────────────────────────────
def test_parallelism_axes_map_through():
    mf = _dsv4align_mf()
    mf["parallelism"].update(tensor_parallel=2, expert_parallel=2, context_parallel=2,
                             data_parallel_shard=4, sequence_parallel=False)
    mf["model"]["num_hidden_layers"] = 4
    pc = from_mindformers_dict(mf).parallel
    assert (pc.tp, pc.ep, pc.cp, pc.dp_shard) == (2, 2, 2, 4)
    assert pc.sequence_parallel is False


def test_auto_dp_shard_from_batch():
    # data_parallel_shard=-1（auto）→ global(2)//(local(1)·num_microbatches(1)) = 2。
    assert from_mindformers_dict(_dsv3_mf()).parallel.dp_shard == 2


def test_recompute_none_when_absent():
    mf = _dsv4align_mf()  # prep_dsv4align FUSED 生产已 pop recompute 段
    assert "recompute" not in mf
    rc = from_mindformers_dict(mf).recompute
    assert rc.mode == "None" and rc.full_layers == set()


def test_recompute_full_layer_offset_plus_one():
    # mindformers full_recompute_layer 0-indexed decoder ["1-2"] → 评估器层 {2,3}（+1 偏移，layer0=embedding）。
    mf = _dsv3_mf()
    mf["recompute"] = {"mode": "full", "full_recompute_layer": ["1-2"]}
    assert from_mindformers_dict(mf).recompute.full_layers == {2, 3}


def test_layers_per_stage_from_pipeline_offset():
    # pp=2, num_hidden_layers=4, offset=[1,-1] → base=2 → decoder[3,1] → +emb@0 +head@末 → [4,2]（和=6=n_layers）。
    mf = _dsv3_mf()
    mf["model"]["num_hidden_layers"] = 4
    mf["parallelism"].update(pipeline_parallel=2, pipeline_parallel_microbatch_size=2,
                             offset=[1, -1])
    mf["recompute"]["full_recompute_layer"] = ["0-3"]
    pc = from_mindformers_dict(mf).parallel
    assert pc.pp == 2 and pc.layers_per_stage == [4, 2]
    assert pc.num_microbatches == 2


def test_layers_per_stage_from_num_layer_list():
    mf = _dsv3_mf()
    mf["model"]["num_hidden_layers"] = 6
    mf["parallelism"].update(pipeline_parallel=2, pipeline_parallel_microbatch_size=2,
                             num_layer_list=[4, 2])
    mf["recompute"]["full_recompute_layer"] = ["0-5"]
    pc = from_mindformers_dict(mf).parallel
    # decoder[4,2] → +emb@0 +head@末 → [5,3]（和=8=6+2）。
    assert pc.layers_per_stage == [5, 3]


def test_pp1_layers_per_stage_none():
    assert from_mindformers_dict(_dsv3_mf()).parallel.layers_per_stage is None


def test_optimizer_params_fp32_from_dtype():
    opt = from_mindformers_dict(_dsv3_mf()).optimizer
    assert opt.type == "AdamW" and opt.state_bytes_per_param == 12  # fp32 master+m+v


def test_hardware_max_device_memory_parsed():
    hw = from_mindformers_dict(_dsv3_mf()).hardware
    assert hw.max_device_memory == 59 * GiB
    assert hw.framework_reserve == 0 and hw.alloc_block_bytes == 512


# ── (d) fail-loud ─────────────────────────────────────────────────────────────────────
def test_fail_loud_on_unmapped_structural_field():
    mf = _dsv3_mf()
    mf["model"]["some_new_attention_hack"] = True   # 既未映射也不在忽略集
    with pytest.raises(NotImplementedError, match="未识别"):
        from_mindformers_dict(mf)


def test_fail_loud_on_non_flash_attention():
    mf = _dsv3_mf()
    mf["model"]["use_flash_attention"] = False
    with pytest.raises(NotImplementedError, match="use_flash_attention"):
        from_mindformers_dict(mf)


def test_fail_loud_on_unknown_attention_variant():
    mf = _dsv4align_mf()
    mf["model"]["experimental_attention_variant"] = "some_future_variant"
    with pytest.raises(NotImplementedError, match="experimental_attention_variant"):
        from_mindformers_dict(mf)


def test_fail_loud_on_contradictory_fusion_flags():
    mf = _dsv4align_mf()
    mf["model"]["apply_dsa_kernel_fusion"] = True
    mf["model"]["force_unfused_dsa"] = True          # 二者都 True → 自相矛盾
    with pytest.raises(NotImplementedError, match="互反"):
        from_mindformers_dict(mf)


def test_fail_loud_on_add_bias_linear_via_build():
    # add_bias_linear=True 忠实映射到 LLMConfig → build_llm 原生 fail-loud（DRY，不在转换器重复守卫）。
    mf = _dsv3_mf()
    mf["model"]["add_bias_linear"] = True
    bundle = from_mindformers_dict(mf)
    assert bundle.llm.add_bias_linear is True
    with pytest.raises(NotImplementedError, match="add_bias_linear"):
        build_llm_spec(bundle.llm)


# ── (e) 惰性 yaml path-loader ─────────────────────────────────────────────────────────
def test_lazy_yaml_path_loader(tmp_path):
    import yaml
    mf = _dsv3_mf()
    p = tmp_path / "ds3_sim.yaml"
    p.write_text(yaml.safe_dump(mf), encoding="utf-8")
    from_path = load_mindformers_yaml(str(p))
    from_dict = from_mindformers_dict(mf)
    assert from_path.llm == from_dict.llm == deepseek_v3(4)
    assert from_path.parallel == from_dict.parallel


def test_core_has_no_hard_yaml_import():
    """纯核 from_mindformers 模块不在顶层 import yaml（仅 loader 内惰性导入）。"""
    import cost_eval.configs.from_mindformers as m
    import inspect
    src = inspect.getsource(m)
    # 顶层无 `^import yaml`；yaml 只出现在 load_mindformers_yaml 函数体内。
    top_level = [ln for ln in src.splitlines()
                 if ln.startswith("import yaml") or ln.startswith("from yaml")]
    assert not top_level, f"纯核不应顶层 import yaml：{top_level}"
