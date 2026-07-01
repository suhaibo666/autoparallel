"""构造缩层 DeepSeek-V4 pynative 仿真 config（dsv4_hybrid + mHC + MTP，非融合可跑）+ 合成数据。

在参考测试目录下运行（含 pynarive_ds3.yaml 作 base，取其 model 外的 checkpoint/training/optimizer/
parallelism/recompute/dataset 段），且 PYTHONPATH 含 mindformers 仓库根（为 tests.utils）：
    SIM_LAYERS=4 SIM_STEPS=3 python prep_ds4_sim.py
产出：同目录 ds4_sim.yaml + train_dataset_sim/ 合成数据。

关键（无 hyper_parallel 也能跑）：apply_dsa_kernel_fusion=False + use_fused_mhc=False → 走 unfused 小算子。
约束（config __post_init__）：len(compress_ratios)=num_layers+mtp；值∈{0,4,128}；
(num_heads·v_head_dim)%o_groups==0；qk_nope+qk_rope==v_head_dim。
"""
import os
import yaml

CUR = os.path.dirname(os.path.abspath(__file__))
# 服务器上 base 文件名是 pynarive_ds3.yaml（历史拼写）；本地是 pynative_ds3.yaml。两个都试。
BASE = None
for cand in ("pynarive_ds3.yaml", "pynative_ds3.yaml"):
    if os.path.exists(os.path.join(CUR, cand)):
        BASE = os.path.join(CUR, cand)
        break
if BASE is None:
    raise FileNotFoundError("需要同目录的 pynarive_ds3.yaml / pynative_ds3.yaml 作 base")

DS_DIR = os.path.join(CUR, "train_dataset_sim")
DS_FILE = os.path.join(DS_DIR, "dataset.mindrecord")

N = int(os.environ.get("SIM_LAYERS", "4"))
STEPS = int(os.environ.get("SIM_STEPS", "3"))
SEQ = int(os.environ.get("SIM_SEQ", "4096"))
MTP = int(os.environ.get("SIM_MTP", "1"))          # num_nextn_predict_layers
HC = int(os.environ.get("SIM_HC", "4"))            # hc_mult（残差流数）
EP = int(os.environ.get("SIM_EP", "1"))
TP = int(os.environ.get("SIM_TP", "1"))
PP = int(os.environ.get("SIM_PP", "1"))
DPSHARD = int(os.environ.get("SIM_DPSHARD", "-1"))

# compress_ratios：长度 = N + MTP，值∈{0,4,128}。混合以覆盖 HCA(128)/CSA(4)/滑窗(0) 三支。
_CYCLE = [0, 4, 128]
compress_ratios = [_CYCLE[i % 3] for i in range(N)] + [0] * MTP

# 1) 合成数据集（与 ds3 同 schema）
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

# 2) 取 base 的非 model 段
with open(BASE) as f:
    cfg = yaml.safe_load(f)
cfg["train_dataset"]["dataloader"]["dataset_files"] = [DS_DIR]
cfg["training"]["local_batch_size"] = 1
cfg["training"]["global_batch_size"] = 2
cfg["training"]["steps"] = STEPS
cfg["recompute"]["full_recompute_layer"] = [f"0-{N - 1}"]
cfg["checkpoint"]["enable_save"] = False
cfg["parallelism"]["expert_parallel"] = EP
cfg["parallelism"]["tensor_parallel"] = TP
cfg["parallelism"]["pipeline_parallel"] = PP
cfg["parallelism"]["data_parallel_shard"] = DPSHARD
if PP > 1:
    cfg["parallelism"]["pipeline_parallel_microbatch_size"] = PP

# 3) 替换为 DeepSeek-V4 model 段（保留 ds3 的 hidden/heads/MoE，换 v4 MLA + 前沿 + 关融合）
cfg["model"] = {
    "model_type": "deepseek_v4",
    "architectures": "DeepseekV4ForCausalLM",
    "vocab_size": 129280,
    "seq_length": SEQ,
    "hidden_size": 1792,
    "intermediate_size": 3072,
    "num_hidden_layers": N,
    "max_position_embeddings": 163840,
    "hidden_act": "silu",
    "num_attention_heads": 8,
    "rms_norm_eps": 1.0e-6,
    "add_bias_linear": False,
    "use_flash_attention": True,
    "multi_latent_attention": True,
    "mla_qkv_concat": True,
    # v4 MLA 头维（head_dim=512 → qk_nope=448, v_head=512）
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_rope_head_dim": 64,
    "head_dim": 512,
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
    # ---- dsv4_hybrid 稀疏压缩注意力 ----
    "experimental_attention_variant": "dsv4_hybrid",
    "compress_ratios": compress_ratios,
    "sliding_window": 128,
    "compress_rope_theta": 160000,
    "o_lora_rank": 1024,
    "o_groups": 16,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 1024,
    "apply_dsa_kernel_fusion": False,   # ★ 无 hyper_parallel 也能跑（unfused）
    # ---- mHC 残差 ----
    "enable_hyper_connections": True,
    "hc_mult": HC,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1.0e-6,
    "use_fused_mhc": False,             # ★ unfused
    # ---- MTP ----
    "num_nextn_predict_layers": MTP,
    "mtp_loss_scaling_factor": 0.3,
    # ---- 位置编码 ----
    "position_embedding_type": "yarn",
    "scaling_factor": 40,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1,
    "mscale_all_dim": 1,
    "rope_theta": 10000,
    # ---- MoE（同 ds3）----
    "router_dense_type": "float32",
    "gated_linear_unit": True,
    "moe_intermediate_size": 1024,
    "routed_scaling_factor": 2.5,
    "first_k_dense_replace": 1,
    "n_routed_experts": 8,
    "num_experts_per_tok": 4,
    "n_shared_experts": 1,
    "moe_shared_expert_intermediate_size": 1024,
    "moe_token_dispatcher_type": "alltoall",
    "moe_grouped_gemm": True,
    "moe_router_load_balancing_type": "seq_aux_loss",
    "moe_aux_loss_coeff": 0.001,
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "moe_token_drop_policy": "probs",
    "moe_router_enable_expert_bias": True,
    "moe_router_bias_update_rate": 0.001,
    "use_pad_tokens": True,
    "topk_group": 2,
    "n_group": 4,
}

out = os.path.join(CUR, "ds4_sim.yaml")
with open(out, "w") as f:
    yaml.dump(cfg, f, indent=2, sort_keys=False)
print("WROTE_CONFIG", out, "layers", N, "mtp", MTP, "hc", HC,
      "compress_ratios", compress_ratios)
