"""第四轮闭环审计（closure_audit_v2_verification_2026-07-15.md §F1/§F6/§F2）反例转正式回归。

针对上一轮（b4b8579）引入的**新回归**与固化的**错误 oracle**，逐条守卫真实 mindformers 语义：

① F1（P0 新回归）`dense_fsdp_shard_size` 上下文无关误判 —— 真实约束（parallel_dims.py:443-470）：
   须正整数、整除 `fsdp=dp_shard·cp`、∈[1,fsdp]；**只有 ==fsdp 才复用完整 FSDP mesh（中性）**，
   `1<=value<fsdp` 是分组 FSDP 子域（显存语义改变，未建模）→ fail-loud；非正/非整除/类型错 → ValueError。

② F6（新回归）`ulysses_degree_in_cp=1` 假拒 —— 真实逻辑（context_parallel.py:248-265）：
   colossal 有效度恒 1；ulysses 要求 degree==cp；只有 hybrid（1<degree<cp 且整除）才是二维 CP（未建模）。

③ F2（旧缺陷 + 固化错误 oracle）`qk_layernorm` 静默改写 —— mindformers 真构造 q/k_layernorm
   （attention.py:311-357），非内存中性。按 attn_type 条件：gqa/mha 真值 → fail-loud；
   mla/dsv4_hybrid/dsa → subsumed（q_a_norm/kv_a_norm/q_hnorm 已建覆盖）。
"""
import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.configs.from_mindformers import (
    _PAR_UNSUPPORTED_TRUTHY,
    from_mindformers_dict,
)


# ── 最小合法 mindformers dict（mha、无 MoE）───────────────────────────────────────────
def _mf(parallelism=None, **model_over):
    m = {"model": {"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 1024,
                   "vocab_size": 1000, "seq_length": 512, "compute_dtype": "bfloat16"},
         "training": {"local_batch_size": 1}}
    if parallelism is not None:
        m["parallelism"] = parallelism
    m["model"].update(model_over)
    return m


def _mla_mf(qk_layernorm=True):
    """最小 plain-MLA dict（multi_latent_attention=True,无变体 → attn_type=mla）。"""
    return {"model": {"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 1024,
                      "vocab_size": 1000, "seq_length": 512, "compute_dtype": "bfloat16",
                      "multi_latent_attention": True, "num_key_value_heads": 8,
                      "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 64,
                      "qk_nope_head_dim": 128, "v_head_dim": 192, "qk_layernorm": qk_layernorm},
            "training": {"local_batch_size": 1}}


def _dsv4_mf(qk_layernorm=True):
    """最小 dsv4_hybrid dict（experimental_attention_variant=dsv4_hybrid → subsumed 分支）。"""
    return {"model": {"num_hidden_layers": 4, "num_attention_heads": 128, "hidden_size": 2048,
                      "vocab_size": 1000, "seq_length": 2048, "multi_latent_attention": True,
                      "experimental_attention_variant": "dsv4_hybrid",
                      "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                      "qk_nope_head_dim": 128, "v_head_dim": 128, "o_groups": 16, "o_lora_rank": 256,
                      "dsa_indexer_n_heads": 64, "dsa_indexer_head_dim": 128,
                      "dsa_indexer_topk": 2048, "csa_compress_ratios": [1, 1, 1, 4],
                      "apply_dsa_kernel_fusion": True, "force_unfused_dsa": False,
                      "compute_dtype": "bfloat16", "n_routed_experts": 8,
                      "num_experts_per_tok": 2, "moe_intermediate_size": 2048,
                      "qk_layernorm": qk_layernorm, "cross_entropy_fused": True},
            "training": {"local_batch_size": 1}}


# ══ ① F1 dense_fsdp_shard_size 按完整 FSDP 域（fsdp=dp_shard·cp）判定 ═══════════════════
def test_dense_fsdp_equals_full_fsdp_accepted():
    """value == fsdp（dp_shard=8·cp=1）→ 复用完整 FSDP mesh、与默认语义一致（中性）→ 接受。
    探针的 `dense_fsdp_8_with_full_fsdp_8` 假阳性案例（此前被误拒）。"""
    bundle = from_mindformers_dict(_mf(parallelism={
        "data_parallel_shard": 8, "dense_fsdp_shard_size": 8}))
    assert bundle.parallel.dp_shard == 8


def test_dense_fsdp_below_full_fsdp_modeled():
    """1<=value<fsdp 且整除（dp_shard=8,shard=1）= 分组 FSDP 子域 → **真建模**（Z3,2026-07-15）：
    映射到 ParallelConfig.dense_fsdp_shard_size=1（此前 F1 fail-loud——「审计要么 fail-loud 要么建模」,
    本轮选真建模：static_mem 用子域分母切 dense 持久,每卡 dense 驻留 ÷1=global 更大）。"""
    bundle = from_mindformers_dict(_mf(parallelism={
        "data_parallel_shard": 8, "dense_fsdp_shard_size": 1}))
    assert bundle.parallel.dense_fsdp_shard_size == 1


def test_dense_fsdp_below_full_fsdp_divisor_modeled():
    """shard=2（整除 fsdp=8 但 <fsdp）同属分组子域 → 建模映射为 2（dense 持久 ÷2,是完整域 ÷8 的 ×4）。"""
    bundle = from_mindformers_dict(_mf(parallelism={
        "data_parallel_shard": 8, "dense_fsdp_shard_size": 2}))
    assert bundle.parallel.dense_fsdp_shard_size == 2


def test_dense_fsdp_zero_rejected():
    """value=0 非正 → runtime 明确拒（parallel_dims.py:458-462）,ValueError。
    探针的 `dense_fsdp_zero` 假阴性案例（此前被误接受）。"""
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        from_mindformers_dict(_mf(parallelism={
            "data_parallel_shard": 8, "dense_fsdp_shard_size": 0}))


def test_dense_fsdp_non_divisor_rejected():
    """shard=3 不整除 fsdp=8 → runtime 拒（parallel_dims.py:464-468）,ValueError。"""
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        from_mindformers_dict(_mf(parallelism={
            "data_parallel_shard": 8, "dense_fsdp_shard_size": 3}))


def test_dense_fsdp_gt_fsdp_rejected():
    """shard=16 > fsdp=8 → 超域,runtime 拒,ValueError。"""
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        from_mindformers_dict(_mf(parallelism={
            "data_parallel_shard": 8, "dense_fsdp_shard_size": 16}))


def test_dense_fsdp_bool_rejected():
    """bool 是 int 子类,会绕过整除检查（parallel_dims.py:455-462 显式排除）→ ValueError。"""
    with pytest.raises(ValueError, match="dense_fsdp_shard_size"):
        from_mindformers_dict(_mf(parallelism={
            "data_parallel_shard": 8, "dense_fsdp_shard_size": True}))


def test_dense_fsdp_absent_accepted():
    """未配 → 默认全域 FSDP,跳过（不 fail-loud）。"""
    assert from_mindformers_dict(_mf(parallelism={"data_parallel_shard": 8}))


def test_dense_fsdp_domain_includes_cp():
    """fsdp=dp_shard·cp 含 cp：dp_shard=2·cp=2 → fsdp=4；shard=4(==fsdp) 中性(0)、shard=2(<fsdp) 建模(2)。
    证明判定用的是**完整 FSDP 域**而非仅 dp_shard。"""
    b_full = from_mindformers_dict(_mf(parallelism={
        "data_parallel_shard": 2, "context_parallel": 2, "dense_fsdp_shard_size": 4}))
    assert b_full.parallel.dense_fsdp_shard_size == 0        # ==fsdp → 复用完整 mesh,中性
    b_sub = from_mindformers_dict(_mf(parallelism={
        "data_parallel_shard": 2, "context_parallel": 2, "dense_fsdp_shard_size": 2}))
    assert b_sub.parallel.dense_fsdp_shard_size == 2         # <fsdp → 子域,建模


def test_dense_fsdp_not_in_truthy_set():
    """dense_fsdp_shard_size 不得再落在布尔 truthy 拒绝集（回归守卫：曾在 _PAR_UNSUPPORTED_NUMERIC_GT1）。"""
    assert "dense_fsdp_shard_size" not in _PAR_UNSUPPORTED_TRUTHY


# ══ ② F6 ulysses_degree_in_cp 按 method + cp + value 联合判定 ═══════════════════════════
def test_colossal_ulysses_degree_one_accepted():
    """colossal + ulysses_degree_in_cp=1 → 有效度恒 1（context_parallel.py:253-254）,全域 CP、已建 → 接受。
    探针的 `colossal_ulysses_degree_one` 假拒案例（b4b8579 把 1 当布尔真值拒）。"""
    assert from_mindformers_dict(_mf(parallelism={
        "context_parallel_method": "colossal", "ulysses_degree_in_cp": 1}))


def test_ulysses_degree_absent_accepted():
    """未配 ulysses_degree_in_cp → colossal:1 / ulysses:cp（context_parallel.py:253-259）,均全域 → 接受。"""
    assert from_mindformers_dict(_mf(parallelism={"context_parallel_method": "colossal"}))


def test_ulysses_full_domain_accepted():
    """ulysses + degree==cp（=2）→ 全域 all-to-all（context_parallel.py:256-259）,评估器已建 → 接受。"""
    assert from_mindformers_dict(_mf(parallelism={
        "context_parallel": 2, "context_parallel_method": "ulysses", "ulysses_degree_in_cp": 2}))


def test_ulysses_degree_not_equal_cp_rejected():
    """ulysses + degree(1) != cp(2) → runtime 拒（context_parallel.py:257-258）,ValueError。"""
    with pytest.raises(ValueError, match="[Uu]lysses"):
        from_mindformers_dict(_mf(parallelism={
            "context_parallel": 2, "context_parallel_method": "ulysses", "ulysses_degree_in_cp": 1}))


def test_hybrid_2d_cp_failloud():
    """hybrid + 1<degree(2)<cp(4) 且整除 → 二维 CP（ulysses×ring）未建模（context_parallel.py:264）→ fail-loud。"""
    with pytest.raises(NotImplementedError, match="hybrid|ulysses_degree_in_cp"):
        from_mindformers_dict(_mf(parallelism={
            "context_parallel": 4, "context_parallel_method": "hybrid", "ulysses_degree_in_cp": 2}))


def test_hybrid_non_divisor_rejected():
    """hybrid + degree(3) 不整除 cp(4) → runtime 拒（context_parallel.py:264-265）,ValueError。"""
    with pytest.raises(ValueError):
        from_mindformers_dict(_mf(parallelism={
            "context_parallel": 4, "context_parallel_method": "hybrid", "ulysses_degree_in_cp": 3}))


def test_hybrid_missing_degree_rejected():
    """hybrid 未给 degree → runtime 拒（context_parallel.py:261-262）,ValueError。"""
    with pytest.raises(ValueError):
        from_mindformers_dict(_mf(parallelism={
            "context_parallel": 4, "context_parallel_method": "hybrid"}))


def test_ulysses_degree_not_in_truthy_set():
    """ulysses_degree_in_cp 不得再落在布尔 truthy 拒绝集（回归守卫：曾在 _PAR_UNSUPPORTED_TRUTHY）。"""
    assert "ulysses_degree_in_cp" not in _PAR_UNSUPPORTED_TRUTHY


# ══ ③ F2 qk_layernorm 按 attn_type 条件（gqa/mha fail-loud;mla 家族 subsumed）═══════════
def test_qk_layernorm_mha_truthy_failloud():
    """mha + qk_layernorm=True → Q/K 上 2 个 RMSNorm 未建 op → fail-loud（绕过 build_llm 静默评错模型）。"""
    with pytest.raises(NotImplementedError, match="qk_layernorm"):
        from_mindformers_dict(_mf(qk_layernorm=True))


def test_qk_layernorm_gqa_truthy_failloud():
    """gqa + qk_layernorm=True（Qwen3 系）→ fail-loud（Qwen3-32B 每层 BF16 72 MiB、64 层 4.5-9 GiB）。"""
    with pytest.raises(NotImplementedError, match="qk_layernorm"):
        from_mindformers_dict(_mf(parallelism=None, num_key_value_heads=2, qk_layernorm=True))


def test_qk_layernorm_mha_false_accepted():
    """mha + qk_layernorm=False → 无 norm,正常接受,llm.qk_layernorm 保持默认 False。"""
    bundle = from_mindformers_dict(_mf(qk_layernorm=False))
    assert bundle.llm.qk_layernorm is False


def test_qk_layernorm_mla_subsumed_builds():
    """mla + qk_layernorm=True → subsumed（q_a_norm/kv_a_norm 已建覆盖）→ 不 fail-loud、可 build。
    llm.qk_layernorm 保持 False（语义由已建 MLA norm 覆盖,不再走 build_llm 的 qk fail-loud）。"""
    bundle = from_mindformers_dict(_mla_mf(qk_layernorm=True))
    assert bundle.llm.qk_layernorm is False
    assert build_llm_spec(bundle.llm) is not None


def test_qk_layernorm_dsv4_subsumed_builds():
    """dsv4_hybrid + qk_layernorm=True → subsumed（q_a_norm/kv_a_norm/q_hnorm 已建）→ 可 build（round-trip 不破）。"""
    bundle = from_mindformers_dict(_dsv4_mf(qk_layernorm=True))
    assert bundle.llm.qk_layernorm is False
    assert build_llm_spec(bundle.llm) is not None
