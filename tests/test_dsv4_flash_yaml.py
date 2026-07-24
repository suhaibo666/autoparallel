"""现场 DSv4-Flash yaml 导入回归（用户报告 2026-07-21：test.yaml 导入失败）。

覆盖修复的每一环（trimmed 但保真的 mindformers dict）：
  ① 顶层 `checkpoint`/`lr_scheduler` 段（TorchTitan 命名，内存无关）不再 fail-loud。
  ② optimizer `type: Muon` + Muon/AdamW 数值旋钮键（ns_coefficients/qk_clip/... 内存无关）。
  ③ model 异名映射：`num_residual_streams`→hc_mult（mHC×n）、`index_*`→dsa_indexer_*、
     `compress_ratios`→csa_compress_ratios；数值旋钮（activation_func_clamp_value/... ）忽略。
  ④ parallelism：`moe_token_dispatcher_type` 中性；`pipeline_parallel_overlap_p2p/b_f` 近似 warn 放行；
     `data_parallel_shard: -1` 自动解析；`pipeline_parallel_layers_per_stage` 为 **list** 写法。
  ⑤ tiny param（dsv4_hybrid `attn_sink`=n_heads）在大 fsdp 下按 FSDP2 ceil 补齐、不 fail-loud。
  ⑥ round-trip 到 UI 字段：Muon 与 mHC 不被静默降级。
"""
import warnings

from cost_eval.configs.from_mindformers import from_mindformers_dict


def _dsv4_flash(**par_over):
    par = {
        "data_parallel_shard": -1, "reshard_after_forward_policy": "default",
        "tensor_parallel": 1, "sequence_parallel": False,
        "context_parallel": 1, "context_parallel_method": "colossal",
        "pipeline_parallel": 8, "pipeline_parallel_interleave_num": 1,
        "pipeline_parallel_layers_per_stage": ["0-3", "4-8", "9-13", "14-19",
                                               "20-25", "26-31", "32-37", "38-42"],
        "pipeline_parallel_overlap_p2p": True, "pipeline_parallel_overlap_b_f": True,
        "expert_parallel": 32, "moe_token_dispatcher_type": "alltoall",
    }
    par.update(par_over)
    return {
        "checkpoint": {"enable_save": False, "load_balanced": False},   # ① 顶层内存无关段
        "lr_scheduler": {"type": "ConstantWarmUpLR", "learning_rate": 1.e-5},
        "training": {"local_batch_size": 1, "global_batch_size": 256},
        "optimizer": {                                                  # ② Muon + 数值旋钮
            "type": "Muon", "weight_decay": 0.1, "matched_adamw_rms": 0.2, "momentum": 0.95,
            "nesterov": True, "adamw_betas": [0.9, 0.95], "adamw_eps": 1e-8,
            "qk_clip_enabled": True, "qk_clip_threshold": 100,
            "ns_coefficients": [[[3.4445, -4.775, 2.0315], 8]],
            "comm_strategy": "allgather_deredundency", "use_fused_adamw": True,
        },
        "parallelism": par,
        "recompute": {"mode": "full", "full_recompute_layer": ["0-43"]},
        "model": {
            "model_type": "deepseek_v4", "vocab_size": 129280, "seq_length": 4096,
            "hidden_size": 4096, "num_hidden_layers": 43, "num_attention_heads": 64,
            "multi_latent_attention": True, "experimental_attention_variant": "dsv4_hybrid",
            "q_lora_rank": 1024, "qk_rope_head_dim": 64, "qk_nope_head_dim": 448, "v_head_dim": 512,
            "o_groups": 8, "o_lora_rank": 1024, "qk_layernorm": True,
            "params_dtype": "bfloat16", "compute_dtype": "bfloat16",
            "num_nextn_predict_layers": 1,
            "moe_intermediate_size": 2048, "n_routed_experts": 256, "num_experts_per_tok": 6,
            "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 2048,
            "gated_linear_unit": True,
            # ③ 异名映射（内存相关）
            "num_residual_streams": 4, "enable_hyper_connections": True,
            "index_n_heads": 64, "index_head_dim": 128, "index_topk": 512,
            "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0],   # 44 = 43 层 + MTP
            # ③ 数值旋钮（内存无关）
            "activation_func_clamp_value": 10.0, "compress_rotary_base": 160000,
            "mhc_init_gating_factor": 0.01, "mhc_sinkhorn_iterations": 20,
            "moe_router_score_function": "sqrtsoftplus", "num_hash_layers": 1,
            "apply_dsa_kernel_fusion": True,
        },
    }


def test_dsv4_flash_yaml_imports():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        b = from_mindformers_dict(_dsv4_flash())
    assert b is not None
    # ② Muon 生效（2D 矩阵 momentum-only，比 AdamW 每参省 v）
    assert str(b.optimizer.type).lower() == "muon"
    # ③ mHC×4 生效
    assert b.llm.residual_variant == "mhc" and b.llm.num_residual_streams == 4
    # ③ DSA indexer 异名映射
    assert b.llm.dsa_indexer_n_heads == 64 and b.llm.dsa_indexer_topk == 512
    # ③ dsv4_hybrid + compress_ratios（44 = 43 + MTP）
    assert b.llm.attn_type == "dsv4_hybrid" and len(b.llm.csa_compress_ratios) == 44
    # ④ 结构：43 层 + 1 MTP、pp=8、ep=32
    assert b.llm.num_layers == 43 and b.llm.mtp_num_layers == 1
    assert b.parallel.pp == 8 and b.parallel.ep == 32
    # ④ data_parallel_shard=-1 auto：num_microbatches 缺省=pp（非 1）→ dp_shard=global/(local·pp)=
    #   256/(1·8)=32（真机 256卡/pp8 值）；此前误用 ppm 默认 1 → dp_shard=256、持久欠估 ~8×。
    assert b.parallel.dp_shard == 32
    # ④ PP overlap → 近似 warn（放行）
    assert any("overlap" in str(w.message) for w in rec)


def test_dsv4_flash_layers_per_stage_list():
    # list 写法每 stage 区间 → 每 stage decoder 层数（sum=43 transformer）。
    b = from_mindformers_dict(_dsv4_flash())
    lps = b.parallel.layers_per_stage
    assert lps is not None and len(lps) == 8
    # 含伪层：embedding 归 stage0、head+mtp 归末 stage → 和 = 43 + mtp(1) + 2 伪层 = 46
    assert sum(lps) == 43 + 1 + 2


def test_dsv4_flash_evaluates_via_ui_roundtrip():
    # 用户真实路径:yaml → _bundle_to_fields → UI 字段 → eval_config（非直连 bundle）。
    # 验证 tiny attn_sink（n_heads=64）在大 fsdp 下不再 fail-loud → 完整评估不崩、出正峰值,且
    # Muon/mHC 经 round-trip 不被静默降级。
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import serve_explorer as S
    fields = S._bundle_to_fields(from_mindformers_dict(_dsv4_flash()))
    assert fields["optimizer"] == "muon" and fields["hc"] == 4     # Muon + mHC 不丢
    # 浏览器 skip-null（输入保留默认）：None 字段用页面默认兜底（ffn 纯 MoE 无关）。
    DEF = {"preset": "custom", "ffn": "3072", "select": "attn", "sel_layers": "",
           "sel_ops": "", "vpp": "1", "mbs": "", "grad_bytes": "4"}
    q = dict(DEF)
    q.update({k: str(v) for k, v in fields.items() if v is not None})
    r = S.eval_config(q)
    assert r["ok"], r.get("errors")
    assert r["device_peak"] > 0 and len(r["stages"]) == 8         # pp=8
    st = max(r["stages"], key=lambda s: s["peak"])
    assert st["persist_breakdown"]["optimizer"] == "Muon"         # 持久分解按 Muon


def test_dsv4_flash_structure_graph_matches_timeline_pure_theory():
    """结构页层卡与 timeline 全重算**纯理论口径**一致（2026-07-24 口径切换后重写）。

    去经验 ctx 免疫 pin 后：full 态层卡只存**层入口 checkpoint_input**（bf16，无 pinned 免疫量）。
      - `pinned_mib == 0`（免疫量已删）；
      - `act_mib == entry_mib`（= checkpoint_input）；
      - `act_mib < full_act_mib`（全重算仍省激活）；
      - 与 timeline 逐微批 act_live 增量同口径（每层 pin checkpoint_input）。
    真机每微批层 ~1.9G 驻留是框架释放缺口，显式暴露、不进层卡/timeline 数字。
    """
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import serve_explorer as S
    fields = S._bundle_to_fields(from_mindformers_dict(_dsv4_flash()))
    DEF = {"preset": "custom", "ffn": "3072", "select": "attn", "sel_layers": "",
           "sel_ops": "", "vpp": "1", "mbs": "", "grad_bytes": "4"}
    q = dict(DEF)
    q.update({k: str(v) for k, v in fields.items() if v is not None})
    r = S.eval_config(q)
    assert r["ok"], r.get("errors")
    g0 = [L for L in r["stages"][0]["graph"] if L["recomp"] == "full"]
    assert g0, "stage0 应有全重算层"
    for L in g0:
        assert L["pinned_mib"] == 0, "纯理论口径:full 态无 ctx 免疫量"
        # 层卡 stored = 层入口 checkpoint_input（与 mem_timeline full 分支同式；容差=独立 round 0.1M）
        assert abs(L["act_mib"] - L["entry_mib"]) < 0.2, L
        assert L["act_mib"] < L["full_act_mib"], "全重算存量仍应小于无重算全量"
    # 与 timeline 对账:首微批各层 fwd 的 act_live 逐层增量 == 层卡 act_mib（= checkpoint_input）
    tl = r["stages"][0]["timeline"]
    incs, prev = [], 0.0
    for e in tl:
        if e["event"].startswith("fwd:"):
            cur = e["buckets"].get("act_live", 0.0)
            incs.append(cur - prev); prev = cur
        elif not e["event"].startswith("fwd"):
            break
    # 跳过伪层(embedding,增量≈0),取与全重算层数相同的非零增量段比对
    nonzero = [i for i in incs if i > 1.0][:len(g0)]
    for inc, L in zip(nonzero, g0):
        assert abs(inc - L["act_mib"]) < 1.0, (inc, L["act_mib"])
