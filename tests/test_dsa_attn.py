"""P2.11 — `dsa` 独立 attention 变体（**预估计**，非 MLA 别名）。

op 图基于 mindformers **training_graph 静态图** DSA 代码推导（无真机锚点，待 pynative
DSA 落地重校准）：
  - `training_graph/transformer/dsa/dsa_attention.py`（fused sfa，全量 KV + topk_indices）
  - `training_graph/transformer/dsa/dsa_indexer.py`（wq_b/wk/k_norm/weights_proj +
    dense-warmup index_scores O(S²) fp32）
  - `transformer/multi_latent_attention.py` use_dsa 分支（无 linear_kvb 展开、MQA absorb）

要点：
  1. registry 有 "dsa"，且 != mla（结构上多 indexer + absorb，saves 更大）。
  2. indexer 的 index_scores bwd_scratch = "4*B*S*S"（O(S²) fp32，与 dsv4 同公式族）。
  3. 全链路（build_llm_spec / serve_explorer.eval_config）：dsa 峰值 ≥ mla 同维峰值。
  4. fail-loud：indexer 三维缺失即报错，不静默产错图。
"""
from dataclasses import replace

import pytest

from cost_eval.model_spec import DimTable, OpSpec
from cost_eval.layer_context import LayerContext
from cost_eval.layers.registry import ATTN_REGISTRY
from cost_eval.layers.attention import build_mla_attn_ops
from cost_eval.layers.dsa import build_dsa_attn_ops
from cost_eval.presets import deepseek_v3
from cost_eval.build_llm import build_llm_spec
from cost_eval.shape_eval import eval_expr

# DSv3 缩层维度 + DSv3.2-Exp 的 indexer 三维（64/128/2048）
D = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=192, S=4096, B=1,
             vocab=129280, n_layers=6, n_experts=8, topk=4, n_shared=1, moe_F=1024,
             q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64,
             qk_nope_head_dim=128, v_head_dim=192, moe_shared_F=1024,
             dsa_indexer_n_heads=64, dsa_indexer_head_dim=128, dsa_indexer_topk=2048,
             dtype_bytes=2)


def _names(ops):
    return [o.name for o in ops]


def _dec(attn):
    return LayerContext(kind="decoder", attn_type=attn, ffn_type="dense")


def _saved_bytes(ops, dims):
    """symbolic saves 去重求和（按名去重，同 structure_mem 口径；不含 workspace/scratch）。"""
    seen, total = set(), 0
    for op in ops:
        for t in op.saves:
            if t.name in seen:
                continue
            seen.add(t.name)
            numel = 1
            for e in t.shape:
                numel *= eval_expr(str(e), dims)
            total += numel * (t.dtype_bytes if t.dtype_bytes is not None else dims.dtype_bytes)
    return total


# ── ① registry：dsa 是真实变体，不是 mla 别名 ────────────────────────────────────

def test_registry_has_dsa_bound_to_builder():
    assert "dsa" in ATTN_REGISTRY
    assert ATTN_REGISTRY["dsa"].__wrapped__ is build_dsa_attn_ops
    ops = ATTN_REGISTRY["dsa"](D, _dec("dsa"))
    assert isinstance(ops, list) and all(isinstance(o, OpSpec) for o in ops)


def test_dsa_structure_matches_static_graph():
    ops = build_dsa_attn_ops(D)
    names = _names(ops)
    # indexer 四投影 + top-k（dsa_indexer.py:121-161 / :257-265）
    for n in ("idx_q", "idx_k", "idx_k_norm", "idx_weights", "indexer"):
        assert n in names
    # MQA absorb（multi_latent_attention.py:243-254/:301-305）：无 linear_kvb 前向展开（:575-579）
    assert "q_absorb" in names and "v_absorb" in names
    assert "linear_kvb" not in names
    assert "sparse_flash" in names
    # 与 FFN 尾可拼接
    assert ops[-1].output.name == "h1"
    assert ops[-1].output.shard == {0: "sp"}


def test_dsa_is_not_mla_and_saves_strictly_more():
    dsa_ops = build_dsa_attn_ops(D)
    mla_ops = build_mla_attn_ops(D)
    assert _names(dsa_ops) != _names(mla_ops)
    # 同维下 dsa saves > mla saves（indexer 双输出 + 更大的 absorb q/attn latent）
    assert _saved_bytes(dsa_ops, D) > _saved_bytes(mla_ops, D)


def test_sparse_flash_saves_full_kv_not_topk_subset():
    """fused sfa kernel 输入为**全量** latent KV + topk_indices（dsa_attention.py:140-149），
    不物化 selected-KV gather → saves 含 kv_a_out/key_cat（∝S），topk_indices int32。"""
    ops = build_dsa_attn_ops(D)
    sf = next(op for op in ops if op.name == "sparse_flash")
    saved = {t.name: t for t in sf.saves}
    assert "kv_a_out" in saved and "key_cat" in saved       # latent KV 全量
    assert "topk_indices" in saved
    assert saved["topk_indices"].dtype_bytes == 4           # int32
    assert "q_cat" in saved                                  # Q 全量（absorb 后）


# ── ② index_scores O(S²) scratch（与 dsv4 indexer 同公式族）─────────────────────

def test_indexer_scratch_is_S2_fp32_and_scales_quadratically():
    ops = build_dsa_attn_ops(D)
    idx = next(op for op in ops if op.name == "indexer")
    assert idx.bwd_scratch == "4*B*S*S"
    b4k = eval_expr(idx.bwd_scratch, D)
    b8k = eval_expr(idx.bwd_scratch, replace(D, S=8192))
    assert b4k == 4 * 4096 * 4096
    assert b8k == 4 * b4k                                   # S 翻倍 → scratch ×4（~S²）
    # 对照：topk 侧输出只随 S 线性（[B,S,topk]）
    tk4 = eval_expr("B*S*dsa_indexer_topk", D)
    tk8 = eval_expr("B*S*dsa_indexer_topk", replace(D, S=8192))
    assert tk8 == 2 * tk4


# ── ③ fail-loud：indexer 三维缺失即报错 ─────────────────────────────────────────

def test_builder_requires_indexer_dims():
    with pytest.raises(ValueError, match="dsa_indexer"):
        build_dsa_attn_ops(replace(D, dsa_indexer_n_heads=0))
    with pytest.raises(ValueError, match="dsa_indexer"):
        build_dsa_attn_ops(replace(D, dsa_indexer_topk=0))


def test_build_llm_spec_requires_indexer_dims():
    cfg = replace(deepseek_v3(2), attn_type="dsa")          # 预设无 indexer 三维（默认 0）
    with pytest.raises(NotImplementedError, match="dsa_indexer"):
        build_llm_spec(cfg)


# ── ④ 全链路：build_llm_spec 派发 + dsa 峰值 ≥ mla 同维峰值 ─────────────────────

def _dsa_cfg(n=4):
    return replace(deepseek_v3(n), attn_type="dsa",
                   dsa_indexer_n_heads=64, dsa_indexer_head_dim=128, dsa_indexer_topk=2048)


def test_build_llm_spec_dispatches_dsa():
    spec = build_llm_spec(_dsa_cfg(2))
    dec = next(k for k in spec.layer_specs if k.startswith("dsa"))   # 层标签为 dsa_dense/dsa_moe
    names = _names(spec.layer_specs[dec].ops)
    assert "indexer" in names and "sparse_flash" in names


def test_eval_config_dsa_peak_ge_mla():
    """serve_explorer：attn=dsa 真实流经 dsa builder（别名已移除），峰值 ≥ 同维 mla。"""
    from serve_explorer import eval_config
    base = {"layers": "4", "dp": "1", "seq": "4096"}
    r_dsa = eval_config({**base, "attn": "dsa"})
    r_mla = eval_config({**base, "attn": "mla"})
    assert r_dsa["ok"] and r_mla["ok"]
    assert r_dsa["device_peak"] >= r_mla["device_peak"]
    # 严格大于（indexer saves/scratch + absorb 大 q 非零）
    assert r_dsa["device_peak"] > r_mla["device_peak"]


def test_presets_carry_indexer_dims_and_preestimate_label():
    from serve_explorer import PRESETS
    d32 = PRESETS["dsv32_exp"]
    g5 = PRESETS["glm5"]
    assert d32["dims"]["dsa_indexer_n_heads"] == 64
    assert d32["dims"]["dsa_indexer_head_dim"] == 128
    assert d32["dims"]["dsa_indexer_topk"] == 2048
    assert g5["dims"]["dsa_indexer_n_heads"] == 32
    assert g5["dims"]["dsa_indexer_head_dim"] == 128
    assert g5["dims"]["dsa_indexer_topk"] == 2048
    assert "预估计" in d32["source"] and "预估计" in g5["source"]
    assert "未建模" not in d32["source"] and "未建模" not in g5["source"]
