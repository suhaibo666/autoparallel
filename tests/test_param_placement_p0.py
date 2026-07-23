# -*- coding: utf-8 -*-
"""P0 参数 placement 修正矩阵（2026-07-23,audit `analysis/fsdp_moe_hsdp_sharding_report.md` §11）。

Runtime 基线:mindformers commit 377c9c344（本地 E:/97-codes/torch_parallel/mindformers）。
四项 P0（报告 §4/§6/§7,两侧核实属实）:
  P0-1 shared expert 权重 TP **复制**（parallelize.py:718-724）—— 修前按 tp 切,欠估 T 倍。
  P0-2 routed expert **ep=1 退化**:不建 expert wrap,随父层走 dense FSDP（:700-716/
       :1030-1037/:1106-1113/:1496-1520）—— 修前恒走 efsdp=SCT,欠估 T 倍。
  P0-3 FSDP 首维不可整除 → **整参 replicate_params**（:331-350;特殊小参数 :353-378）——
       修前 total-numel ceil（已被 runtime 源码推翻,非 OOM 安全）。
  P0-4 adapter `data_parallel`+`shard<=0` → 纯 FSDP（trainer.py:449-454:R=1,S=D）——
       修前映射成纯复制 DP（方向性全错）。

本文件按报告 §11 的 7 行测试矩阵逐条落地。
"""
import warnings

import pytest

from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.ffn import build_shared_expert_ops
from cost_eval.layers.moe import build_moe_decoder
from cost_eval.specs import ParallelConfig, OptimizerSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval, resolve_tensor
from cost_eval.static_mem import StaticMem
from cost_eval.structure_mem import _fsdp_local_count
from cost_eval.configs.from_mindformers import from_mindformers_dict

# AdamW fp32 参数持久倍数（param4+m4+v4）
_STATE = 12

# toy MoE（维度全部被 tp=4/dp=8/子域 2 整除;含 shared expert）
_MD = DimTable(H=16, F=32, n_heads=4, n_kv=4, head_dim=4, S=8, B=1, vocab=16, n_layers=1,
               n_experts=8, topk=2, moe_F=32, moe_shared_F=32, n_shared=1)


def _moe_spec():
    return ModelSpec("moe", _MD, ["moe"], {"moe": build_moe_decoder(_MD)})


def _expert_weight_count(pm):
    """resolved 图里 e_fc1 专家权重的**持久驻留元素数**（经 _fsdp_local_count,与 StaticMem 同口径）。"""
    g = ShapeEval().resolve(_moe_spec(), pm)
    for layers in g.stages.values():
        for layer in layers:
            for op in layer.ops:
                for w in op.params:
                    if getattr(w, "is_expert", False) and w.name == "e_w1":
                        div = pm.efsdp_degree() if w.is_expert else pm.dense_fsdp_degree()
                        return _fsdp_local_count(w, div, pm.degree("ep")), w.local_numel
    raise AssertionError("未找到专家权重")


# ── 矩阵行 1:ep=1, tp=4, K=F —— routed expert 应为 P/K,而非 P/(F·T) ─────────────────
def test_row1_ep1_tp4_routed_expert_uses_dense_divisor():
    pm = ParallelModel(ParallelConfig(dp_shard=8, tp=4, ep=1), n_layers=1, world_size=32)
    assert pm.efsdp_degree() == pm.dense_fsdp_degree() == 8      # K=F=8,非 SCT=32
    cnt, p_layout = _expert_weight_count(pm)
    assert p_layout % 8 == 0 and cnt == p_layout // 8            # N = P/K（修前 P/32,欠 4 倍）


# ── 矩阵行 2:ep=1, tp=4, K<F —— 应随 dense 子域 K 变化 ────────────────────────────────
def test_row2_ep1_tp4_subdomain_scales_expert():
    pm = ParallelModel(ParallelConfig(dp_shard=8, tp=4, ep=1, dense_fsdp_shard_size=2),
                       n_layers=1, world_size=32)
    assert pm.efsdp_degree() == 2                                # 随 K_cfg=2
    cnt, p_layout = _expert_weight_count(pm)
    assert cnt == p_layout // 2


# ── 矩阵行 3:ep>1, tp>1 —— 应为 P/(SCT),且不受 dense K 影响 ─────────────────────────
def test_row3_ep2_tp4_expert_sct_and_k_independent():
    pm = ParallelModel(ParallelConfig(dp_shard=8, tp=4, ep=2), n_layers=1, world_size=32)
    assert pm.efsdp_degree() == 8 * 4 // 2                       # SCT/E=16
    cnt, p_layout = _expert_weight_count(pm)                     # p_layout 已 ÷ep
    assert cnt == p_layout // 16                                 # N = P/(E·Fe) = P/SCT
    pm_sub = ParallelModel(ParallelConfig(dp_shard=8, tp=4, ep=2, dense_fsdp_shard_size=2),
                           n_layers=1, world_size=32)
    assert pm_sub.efsdp_degree() == 16                           # 不随 dense K


# ── 矩阵行 4:shared expert + tp=4 —— 权重 TP 复制:P/K,非 P/(4K) ────────────────────
def test_row4_shared_expert_weights_tp_replicated():
    pm4 = ParallelModel(ParallelConfig(dp_shard=2, tp=4, sequence_parallel=True),
                        n_layers=1, world_size=8)
    ops = build_shared_expert_ops(_MD)
    ws = {w.name: w for op in ops for w in op.params}
    # 权重无 tp shard（parallelize.py:718-724 TP replicate）→ resolve 后 numel 为全量。
    r1 = resolve_tensor(ws["sh_w1"], _MD, pm4)
    assert r1.local_numel == _MD.H * 2 * _MD.moe_shared_F        # 不 ÷tp（修前 ÷4）
    # 激活按序列 SP 切（SequenceParallel(sequence_dim=0)）:sh_g 首维 S/sp。
    sh_g = next(op.output for op in ops if op.output.name == "sh_g")
    rg = resolve_tensor(sh_g, _MD, pm4)
    assert rg.local_numel == (_MD.S // 4) * _MD.B * 2 * _MD.moe_shared_F
    # tp=1 时两种口径恒等（真机锚点全 tp=1 → 逐字节不变的机理）。
    pm1 = ParallelModel(ParallelConfig(dp_shard=2), n_layers=1, world_size=2)
    assert resolve_tensor(ws["sh_w1"], _MD, pm1).local_numel == _MD.H * 2 * _MD.moe_shared_F


# ── 矩阵行 5:dense 首维不可整除 —— 整参复制,非 total-numel ceil ─────────────────────
def test_row5_indivisible_dim0_replicates_whole_param():
    class W:
        name = "toy_w"
        local_numel = 64 * 7
        is_expert = False
        dim0 = 7                     # 7 % 4 != 0 → replicate_params
    assert _fsdp_local_count(W(), 4, 1) == 64 * 7               # 整参驻留（旧 ceil→112,低估 4 倍）

    class W2(W):
        dim0 = 8                     # 8 % 4 == 0 → 正常 shard
        local_numel = 64 * 8
    assert _fsdp_local_count(W2(), 4, 1) == 64 * 2

    class WE(W):
        is_expert = True
        local_numel = 7              # 扁平 numel 不被 4 整除（expert 按扁平 grouped 存储判定）
    with pytest.raises(ValueError):
        _fsdp_local_count(WE(), 4, ep_degree=2)                  # expert 独立 wrap:fail-loud
    assert _fsdp_local_count(WE(), 4, ep_degree=1) == 7          # ep=1 随 dense → replicate


# ── 矩阵行 6:R>1 HSDP —— 单卡驻留不除 R ────────────────────────────────────────────────
def test_row6_hsdp_replicate_axis_not_in_divisor():
    pm_r1 = ParallelModel(ParallelConfig(dp_shard=4, dp_replicate=1), n_layers=1, world_size=4)
    pm_r2 = ParallelModel(ParallelConfig(dp_shard=4, dp_replicate=2), n_layers=1, world_size=8)
    g1 = ShapeEval().resolve(_moe_spec(), pm_r1)
    g2 = ShapeEval().resolve(_moe_spec(), pm_r2)
    p1 = StaticMem().compute(g1, OptimizerSpec.adamw(), pm_r1)[0]
    p2 = StaticMem().compute(g2, OptimizerSpec.adamw(), pm_r2)[0]
    assert p1 == p2                                              # R 只进 world,不进持久分母


# ── 矩阵行 7:raw `data_parallel` + shard<=0 —— 纯 FSDP(R=1,S=D),非纯复制 ─────────────
def test_row7_raw_dp_with_negative_shard_is_pure_fsdp():
    mf = {
        "training": {"local_batch_size": 1, "global_batch_size": 8},
        "optimizer": {"type": "AdamW", "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.01},
        "parallelism": {
            "data_parallel": 4, "data_parallel_shard": -1,
            "tensor_parallel": 1, "context_parallel": 1,
            "pipeline_parallel": 1, "expert_parallel": 1, "sequence_parallel": False,
        },
        "recompute": {"mode": "None"},
        "model": {
            "model_type": "deepseek_v3", "vocab_size": 128, "seq_length": 16,
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 4,
            "use_flash_attention": True, "multi_latent_attention": False,
            "params_dtype": "bfloat16", "compute_dtype": "bfloat16",
            "gated_linear_unit": True, "first_k_dense_replace": 1,
            "position_embedding_type": "rope", "num_nextn_predict_layers": 0,
        },
    }
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        b = from_mindformers_dict(mf)
    # trainer.py:449-454:shard<0 → dp_replicate=1, dp_shard=data_parallel（纯 FSDP）。
    assert b.parallel.dp_shard == 4 and b.parallel.dp_replicate == 1
    assert any("纯 FSDP" in str(w.message) for w in rec)          # 归一化语义 warning
