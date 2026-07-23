"""dsv4_hybrid **unfused** 激活标定(现场 DSv4-Flash 真机对照,2026-07-22)。

真机实测(非融合 dsv4_hybrid+mHC4+DSA+MoE,4L/seq2048/H4096/pp1/dp2/ep2/无重算/Muon,2 卡完整步):
  allocated peak ≈ **45557 MiB**（rank0）。此前仿真按 fused 路径估 ~28172 → 欠 62%。

两处源忠实修复(`cost_eval/layers/dsv4_hybrid.py`，**仅 unfused 分支**，fused 锚点 15415.5 不动）：
  ① indexer `index_scores` 前向物化 **[B,S,dsa_indexer_n_heads,S]**（indexer.py:246；此前 [B,S,S] 漏 ×n_idx）；
  ② sparse_attn 补 `kv_g` **fp32 副本**（csa.py:490）+ `attn_weights` 升 fp32（csa.py:531）。
另修 serve_explorer：`_bundle_to_fields`/`parse_and_validate` 透传 `dsa_fused`（此前 UI round-trip 丢 →
  yaml unfused 恒按 fused 估）。

修复后仿真 unfused ≈ **36276 MiB**（闭掉约一半差距）。**残差 ~20%**（eager 下 fp32 softmax 中间量/mHC
x_streams fp32 副本，需真机逐桶 micro-anchor 才能精确归因，见 analysis 报告）——故本测试锁**区间**（改进
方向 + fused 安全），不锁死 45557。
"""
import warnings
warnings.simplefilter("ignore")

import serve_explorer as S
from cost_eval.configs.from_mindformers import from_mindformers_dict


def _mf(fused):
    return {"context": {"max_device_memory": "58GB"}, "training": {"local_batch_size": 1, "global_batch_size": 2},
            "optimizer": {"type": "Muon"}, "lr_scheduler": {"type": "ConstantWarmUpLR", "learning_rate": 1e-5},
            "parallelism": {"pipeline_parallel": 1, "data_parallel_shard": 2, "expert_parallel": 2,
                            "tensor_parallel": 1, "context_parallel": 1, "sequence_parallel": False},
            "recompute": {"mode": "None"}, "train_dataset": {"dataloader": {"type": "MindDataset"}},
            "model": {"model_type": "deepseek_v4", "vocab_size": 129280, "seq_length": 2048, "hidden_size": 4096,
                      "num_hidden_layers": 4, "num_attention_heads": 64, "multi_latent_attention": True,
                      "experimental_attention_variant": "dsv4_hybrid", "q_lora_rank": 1024, "kv_lora_rank": 512,
                      "qk_rope_head_dim": 64, "qk_nope_head_dim": 448, "v_head_dim": 512, "o_groups": 8,
                      "o_lora_rank": 1024, "qk_layernorm": True, "params_dtype": "bfloat16", "compute_dtype": "bfloat16",
                      "num_nextn_predict_layers": 0, "moe_intermediate_size": 2048, "n_routed_experts": 8,
                      "num_experts_per_tok": 2, "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 2048,
                      "first_k_dense_replace": 1, "gated_linear_unit": True, "num_residual_streams": 4,
                      "enable_hyper_connections": True, "use_fused_mhc": False, "index_n_heads": 64,
                      "index_head_dim": 128, "index_topk": 512, "compress_ratios": [0, 4, 128, 4],
                      "apply_dsa_kernel_fusion": fused, "use_flash_attention": True}}


def _peak(fused):
    b = from_mindformers_dict(_mf(fused))
    f = S._bundle_to_fields(b)
    DEF = {"preset": "custom", "ffn": "3072", "select": "attn", "sel_layers": "", "sel_ops": "",
           "vpp": "1", "mbs": "", "grad_bytes": "4"}
    q = dict(DEF)
    q.update({k: str(v) for k, v in f.items() if v is not None})
    return S.eval_config(q)["device_peak"], f["dsa_fused"]


def test_dsa_fused_flag_round_trips():
    # apply_dsa_kernel_fusion → dsa_fused 字段透传（此前遗漏,UI 恒按 fused）。
    assert _peak(False)[1] == 0        # unfused
    assert _peak(True)[1] == 1         # fused


def test_unfused_much_larger_than_fused():
    # unfused 物化 kv_g fp32/index_scores[B,S,n_idx,S]/attn_weights fp32 → 显著大于 fused。
    pu, _ = _peak(False)
    pf, _ = _peak(True)
    assert pu > pf * 1.25, (pu, pf)


def test_unfused_calibrated_toward_real():
    # 真机 45557;修复后 ~36276(闭掉约一半)。锁区间[33000,50000](含改进值与真机,残差待 micro-anchor)。
    pu, _ = _peak(False)
    # 2026-07-23(185 U1 相位合账): unfused 反向图 fp32 复本群/naive-r0 链/KL 链入账后
    # 本配置(116 真机 45557)从 36276 收敛到 ~50.8k(+11%,185 U1 同源配置已 ±5%)。
    assert 40000 <= pu <= 54000, pu


def test_fused_path_unchanged_small_moe_still_ok():
    # 保护:fused dsv4_hybrid 仍在合理量级(不因 unfused 改动而漂)。
    pf, _ = _peak(True)
    assert 24000 <= pf <= 32000, pf
