"""缩层 DeepSeek-V4 (dsv4_hybrid, unfused) 仿真 config —— **自包含**（不依赖已被删的 align 基座）。

v4 用 `model_type: deepseek_v3` + `experimental_attention_variant: dsv4_hybrid` + `force_unfused_dsa: true`
（deepseek_v4 config 类路径不稳；v3 类 + dsv4 变体是 test_deepseekv4/dsv4_align_naive_fsdp.yaml 的写法）。

在 v4 checkout 任意目录下运行，PYTHONPATH 含 v4 仓库根（为 tests.utils）：
    SIM_LAYERS=4 SIM_STEPS=3 python prep_dsv4align.py
产出：同目录 dsv4_sim.yaml + train_dataset_v4/ 合成数据（seq=2048）。
"""
import os
import yaml

CUR = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(CUR, "train_dataset_v4")
DS_FILE = os.path.join(DS_DIR, "dataset.mindrecord")

N = int(os.environ.get("SIM_LAYERS", "4"))
STEPS = int(os.environ.get("SIM_STEPS", "3"))
SEQ = int(os.environ.get("SIM_SEQ", "2048"))
MTP = int(os.environ.get("SIM_MTP", "0"))    # num_nextn_predict_layers（MTP 头数）
MHC = int(os.environ.get("MHC", "0"))        # 1 → 开 mHC（hc_mult=HC 残差流）
HC = int(os.environ.get("SIM_HC", "4"))
FUSED = os.environ.get("FUSED") == "1"
_CYCLE = [0, 4, 128]
# compress_ratios 长度 = num_layers + mtp（config __post_init__ 约束）；mtp 层用 0（滑窗）
compress_ratios = [_CYCLE[i % 3] for i in range(N)] + [0] * MTP

# 1) 合成数据集（seq=2048）
if not os.path.exists(DS_FILE):
    from tests.utils.generate_dataset import generate_mindrecord_file
    os.makedirs(DS_DIR, exist_ok=True)
    generate_mindrecord_file(
        seq_length=SEQ, batch_size=2, train_steps=20, dataset_path=DS_FILE,
        data_schema={
            "input_ids": {"type": "int32", "shape": [-1]},
            "labels": {"type": "int32", "shape": [-1]},
            "loss_mask": {"type": "int32", "shape": [-1]},
            "position_ids": {"type": "int32", "shape": [-1]},
        },
    )
    print("DATASET_GENERATED", DS_DIR)
else:
    print("DATASET_EXISTS", DS_DIR)

# 2) 自包含 config（base 段仿 ds3；model 段仿 dsv4_align_naive_fsdp.yaml）
cfg = {
    "checkpoint": {"enable_save": False, "load_path": "", "no_load_optim": True,
                   "save_max": 1, "prefix": "custom", "remove_redundancy": False},
    "context": {"device_target": "Ascend", "max_device_memory": "54GB", "mode": 1},
    "training": {"steps": STEPS, "local_batch_size": 1, "global_batch_size": 2,
                 "max_norm": 1.0, "seed": 42, "deterministic": True},
    "optimizer": {"type": "AdamW", "betas": [0.9, 0.95], "eps": 1.0e-8, "weight_decay": 0.01},
    "lr_scheduler": {"type": "ConstantWarmUpLR", "learning_rate": 1.0e-5, "warmup_ratio": 0},
    "parallelism": {
        "data_parallel_shard": -1, "data_parallel_shard_strategy": "optim_grads_params",
        "disable_gradient_division": True, "expert_parallel": 1, "tensor_parallel": 1,
        "context_parallel": 1, "pipeline_parallel": 1, "pipeline_parallel_microbatch_size": 1,
        "sequence_parallel": True,
    },
    "recompute": {"mode": "full", "full_recompute_layer": [f"0-{N - 1}"]},
    "train_dataset": {
        "dataloader": {"type": "MindDataset", "dataset_files": [DS_DIR],
                       "column_names": ["input_ids", "labels", "loss_mask", "position_ids"],
                       "shuffle": False},
        "drop_remainder": True, "num_parallel_workers": 8, "prefetch_size": 1, "numa_enable": False,
    },
    "model": {
        "model_type": "deepseek_v3",
        "architectures": "DeepseekV3ForCausalLM",
        "vocab_size": 129280,
        "seq_length": SEQ,
        "hidden_size": 1792,
        "intermediate_size": 3072,
        "num_hidden_layers": N,
        "max_position_embeddings": 163840,
        "hidden_act": "silu",
        "num_attention_heads": 64,
        "rms_norm_eps": 1.0e-6,
        "add_bias_linear": False,
        "use_flash_attention": True,
        "multi_latent_attention": True,
        "mla_qkv_concat": False,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 448,
        "v_head_dim": 512,
        "qk_layernorm": True,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "params_dtype": "float32",
        "compute_dtype": "bfloat16",
        "layernorm_compute_dtype": "float32",
        "softmax_compute_dtype": "float32",
        "rotary_dtype": "float32",
        "initializer_range": 0.01,
        # ---- dsv4_hybrid（FUSED=1 走融合 npu_* 算子，需 hyper_parallel；默认 unfused 小算子）----
        "experimental_attention_variant": "dsv4_hybrid",
        "apply_dsa_kernel_fusion": os.environ.get("FUSED") == "1",
        "force_unfused_dsa": os.environ.get("FUSED") != "1",
        "csa_compress_ratios": compress_ratios,
        "csa_window_size": 128,
        "csa_compress_rotary_base": 40000.0,
        "csa_dense_mode": False,
        "dsa_indexer_n_heads": 64,
        "dsa_indexer_head_dim": 128,
        "dsa_indexer_topk": 512,
        "dsa_indexer_loss_coeff": 0.001,
        "dsa_indexer_use_sparse_loss": True,
        "o_groups": 8,
        "o_lora_rank": 1024,
        # ---- mHC（MHC=1 开；FUSED=1 用融合 npu_mhc_* 算子）----
        "enable_hyper_connections": MHC == 1,
        "hc_mult": HC,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1.0e-6,
        # 注：容器 vendor OPP 无 aclnnMhcPreSinkhorn 融合 kernel → mHC 走 unfused（纯 MS 算子，
        # 内存与 fused 等价：主导项是 ×n 残差流，sinkhorn n×n 中间量可忽略）。FUSED_MHC=1 强开融合。
        "use_fused_mhc": os.environ.get("FUSED_MHC") == "1",
        # ---- MTP（SIM_MTP 头数）----
        "num_nextn_predict_layers": MTP,
        "mtp_loss_scaling_factor": 0.3,
        # ---- 位置编码 ----
        "position_embedding_type": "yarn",
        "scaling_factor": 40, "beta_fast": 32, "beta_slow": 1,
        "mscale": 1, "mscale_all_dim": 1, "rope_theta": 10000,
        # ---- MoE ----
        "router_dense_type": "float32", "gated_linear_unit": True,
        "moe_intermediate_size": 1024, "routed_scaling_factor": 2.5,
        "first_k_dense_replace": 1, "n_routed_experts": 4, "num_experts_per_tok": 2,
        "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 1024,
        "moe_token_dispatcher_type": "alltoall", "moe_grouped_gemm": True,
        "moe_router_load_balancing_type": "seq_aux_loss", "moe_aux_loss_coeff": 0.001,
        "scoring_func": "sigmoid", "norm_topk_prob": True, "moe_token_drop_policy": "probs",
        "moe_router_enable_expert_bias": True, "moe_router_bias_update_rate": 0.001,
        "use_pad_tokens": True, "topk_group": 2, "n_group": 2,
    },
}

# 全重算在此 MS 版本对 dsv4 触发 recompute() context_fn 冲突 → 默认关重算（激活全存，
# 反而让 dsv4 内存大头 index_scores/kv_gathered 落在峰值，便于验证）。RECOMPUTE=1 可开回。
if os.environ.get("RECOMPUTE") != "1":
    cfg.pop("recompute", None)

out = os.path.join(CUR, "dsv4_sim.yaml")
with open(out, "w") as f:
    yaml.dump(cfg, f, indent=2, sort_keys=False)
print("WROTE_CONFIG", out, "layers", N, "seq", SEQ, "compress_ratios", compress_ratios,
      "mtp", MTP, "mhc", MHC, "fused", FUSED, "use_fused_mhc", cfg["model"]["use_fused_mhc"])
