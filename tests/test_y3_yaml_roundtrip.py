"""P1-17 / §4.8 闭环:yaml 导入的**完整 round-trip**——页面评估用解析出的完整
`EvaluatorConfigBundle`(dp_replicate / reshard / cpu_offload / prefetch / 设备容量 /
优化器 dtype),而非固定假设(64GiB / AdamW fp32 / swap 关)。

设计:
- `_build_eval_specs(p, pa)` 从 query dict 读非 UI 可表达的 extra 键(缺省=历史手配假设),
  构造 (ParallelConfig, OptimizerSpec, HardwareSpec, SwapSpec)。eval_config 调它。
- `_bundle_to_fields(bundle)` 把 bundle 的完整解析值(含 extra)回填成 UI/隐藏字段 →
  随 qs() 回传 → eval_config 按解析值评估(真 round-trip)。

不破手配路径:extra 键缺省时 eval_config 输出与修前逐字节相同(test_g_*)。
"""
from serve_explorer import eval_config, parse_and_validate, _build_eval_specs, _bundle_to_fields
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, SwapSpec
from cost_eval.configs.from_mindformers import from_mindformers_dict

GiB = 2 ** 30


# ── (A) _build_eval_specs 缺 extra → 复现历史手配固定假设(逐字段) ──────────────────────
def test_a_defaults_reproduce_manual_baseline():
    q = {"layers": "8", "dp": "2", "batch": "1"}
    errs, cfg, pa = parse_and_validate(q)
    assert not errs
    pc, opt, hw, swap = _build_eval_specs(q, pa)
    # 并行:extra 全缺 → ParallelConfig 默认(dp_replicate=1 / reshard=default / offload 关 / prefetch=1)
    assert pc.dp_replicate == 1
    assert pc.reshard_after_forward == "default"
    assert pc.cpu_offload is False
    assert pc.prefetch_depth == 1
    assert pc.dp_shard == 2 and pc.sequence_parallel is True   # dp>1 推导 SP(与修前一致)
    # 优化器:AdamW fp32(state=12)、grad 4B
    assert opt.state_bytes_per_param == 12 and opt.grad_dtype_bytes == 4
    # 硬件:64GiB、framework_reserve=0
    assert hw.max_device_memory == 64 * GiB and hw.framework_reserve == 0
    # swap:关
    assert swap.enable is False


# ── (B) _build_eval_specs 读取 extra 键 → 进 pc/opt/hw ───────────────────────────────
def test_b_reads_extra_keys():
    q = {"layers": "4", "heads": "8", "kv_groups": "8", "hidden": "512", "ffn": "2048",
         "vocab": "1024", "seq": "1024", "attn": "mha", "dp": "2", "tp": "1", "ep": "1",
         "pp": "1", "cp": "1", "dense_k": "4", "experts": "0",
         "dp_replicate": "2", "reshard": "never", "cpu_offload": "1", "prefetch": "2",
         "maxdev_gib": "54", "opt_dtype": "bf16"}
    errs, cfg, pa = parse_and_validate(q)
    assert not errs, errs
    pc, opt, hw, swap = _build_eval_specs(q, pa)
    assert pc.dp_replicate == 2
    assert pc.reshard_after_forward == "never"
    assert pc.cpu_offload is True
    assert pc.prefetch_depth == 2
    assert opt.state_bytes_per_param == 14         # bf16 params → +2B compute copy
    assert hw.max_device_memory == 54 * GiB


# ── (C) dp_replicate extra 进 eval_config 的 world ────────────────────────────────────
def test_c_dp_replicate_scales_world():
    base = {"layers": "4", "dp": "2", "batch": "1"}
    r1 = eval_config({**base, "dp_replicate": "1"})
    r3 = eval_config({**base, "dp_replicate": "3"})
    assert r1["ok"] and r3["ok"]
    assert r1["world"] == 2          # dp_replicate 1 × dp_shard 2
    assert r3["world"] == 6          # dp_replicate 3 × dp_shard 2
    # 单卡峰值不随 dp_replicate 变(持久态不 ÷dp_replicate,复制语义)
    assert abs(r1["device_peak"] - r3["device_peak"]) < 1e-6


# ── (D) reshard=never 改 gather 生命周期 → device_peak ≠ default ───────────────────────
def test_d_reshard_never_changes_peak():
    base = {"layers": "4", "dp": "2", "batch": "1"}
    default = eval_config({**base, "reshard": "default"})
    never = eval_config({**base, "reshard": "never"})
    assert default["ok"] and never["ok"]
    # never:unsharded 权重驻留至反向 → gather_buf 累积 → 峰值更高
    assert never["device_peak"] > default["device_peak"]


# ── (E) 设备容量 extra 进 eval_config 的 OOM 判定 ─────────────────────────────────────
def test_e_maxdev_drives_reserved_oom():
    base = {"layers": "4", "dp": "2", "batch": "1"}
    big = eval_config({**base, "maxdev_gib": "64"})
    small = eval_config({**base, "maxdev_gib": "1"})
    assert big["ok"] and small["ok"]
    assert small["reserved_oom"] is True
    assert big["reserved_oom"] is False


# ── (F) 优化器 dtype extra 进 eval_config 的持久态峰值 ────────────────────────────────
def test_f_optimizer_dtype_changes_peak():
    base = {"layers": "4", "dp": "2", "batch": "1"}
    fp32 = eval_config({**base, "opt_dtype": "fp32"})
    bf16 = eval_config({**base, "opt_dtype": "bf16"})
    assert fp32["ok"] and bf16["ok"]
    # bf16 params 多存一份 compute 副本(state 14 vs 12) → 持久态更高
    assert bf16["device_peak"] > fp32["device_peak"]


# ── (G) 手配路径不变:缺 extra 与显式默认 extra 逐字节相同 ─────────────────────────────
def test_g_manual_path_unchanged_vs_explicit_defaults():
    manual = {"layers": "8", "dp": "1", "pp": "2", "batch": "2"}
    a = eval_config(manual)
    b = eval_config({**manual, "dp_replicate": "1", "reshard": "default", "cpu_offload": "0",
                     "prefetch": "1", "maxdev_gib": "64", "opt_dtype": "fp32"})
    assert a["ok"] and b["ok"]
    assert a["world"] == b["world"]
    assert a["device_peak"] == b["device_peak"]
    assert a["reserved_oom"] == b["reserved_oom"]
    assert a["reserved_margin_mib"] == b["reserved_margin_mib"]
    assert [s["peak"] for s in a["stages"]] == [s["peak"] for s in b["stages"]]


# ── (H) 端到端:mindformers dict → bundle → _bundle_to_fields → eval_config ────────────
def _mf_roundtrip(reshard="never"):
    """MLA MoE dict（镜像 _dsv3_mf 的可 build 结构）+ dp_replicate=2 + reshard。"""
    return {
        "context": {"device_target": "Ascend", "max_device_memory": "54GB", "mode": 1},
        "training": {"steps": 3, "local_batch_size": 1},
        "optimizer": {"type": "AdamW", "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.01},
        "parallelism": {
            "data_parallel": 4, "data_parallel_shard": 2, "expert_parallel": 1,
            "tensor_parallel": 1, "context_parallel": 1, "pipeline_parallel": 1,
            "sequence_parallel": True, "reshard_after_forward_policy": reshard,
        },
        "model": {
            "model_type": "deepseek_v3", "architectures": "DeepseekV3ForCausalLM",
            "vocab_size": 129280, "seq_length": 4096, "hidden_size": 1792,
            "intermediate_size": 3072, "num_hidden_layers": 4, "num_attention_heads": 8,
            "num_key_value_heads": 8, "hidden_act": "silu", "rms_norm_eps": 1.0e-6,
            "add_bias_linear": False, "use_flash_attention": True, "multi_latent_attention": True,
            "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128, "v_head_dim": 192, "params_dtype": "float32",
            "compute_dtype": "bfloat16", "position_embedding_type": "rope",
            "gated_linear_unit": True, "moe_intermediate_size": 1024, "first_k_dense_replace": 1,
            "n_routed_experts": 8, "num_experts_per_tok": 4, "n_shared_experts": 1,
            "moe_shared_expert_intermediate_size": 1024,
        },
    }


def test_h_bundle_fields_carry_full_parsed_values():
    bundle = from_mindformers_dict(_mf_roundtrip())
    f = _bundle_to_fields(bundle)
    # bundle 解析:dp_replicate=2(4//2)、dp_shard=2、reshard=never、54GiB、fp32
    assert bundle.parallel.dp_replicate == 2 and bundle.parallel.dp_shard == 2
    assert f["dp_replicate"] == 2
    assert f["reshard"] == "never"
    assert int(f["cpu_offload"]) == 0
    assert abs(float(f["maxdev_gib"]) - 54.0) < 1e-6
    assert f["opt_dtype"] == "fp32"


def test_h_page_eval_uses_full_bundle_roundtrip():
    bundle = from_mindformers_dict(_mf_roundtrip("never"))
    q = {k: str(v) for k, v in _bundle_to_fields(bundle).items()}
    res = eval_config(q)
    assert res["ok"], res
    # world 反映完整解析:dp_replicate(2) × dp_shard(2) × cp × tp × pp = 4
    assert res["world"] == 4
    # reshard=never ≠ default:改 fields 里的 reshard 复评 → 生命周期不同
    q_def = {**q, "reshard": "default"}
    res_def = eval_config(q_def)
    assert res_def["ok"]
    assert res["device_peak"] > res_def["device_peak"]
