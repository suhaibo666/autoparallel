"""D-1：context-parallel（cp）切激活。

守卫忠实模型（specs/2026-07-06-audit-remediation.md §4·D-1）：cp 是全局序列并行域，每个**非
权重激活**的 token/query 维 ÷cp（含 flash workspace / loss·index bwd_scratch），而**参数**
（gather_buf/grad_buf/持久 param 路径）不被本次改动触碰。硬不变量：cp=1 逐字节 no-op。
"""
from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval, resolve_tensor
from cost_eval.structure_mem import estimate_structure_memory
from cost_eval.model_spec import DimTable, TensorRef, OpSpec, OpType, LayerSpec, ModelSpec
from cost_eval.layers.ffn import build_moe_ffn_ops


def _small_cfg():
    """小 gqa dense 模型：embedding + 2 decoder(gqa+dense) + lm_head。
    覆盖 act_live(saves) / workspace(FLASH_LSE_WS∝S) / bwd_scratch(loss 8·S·B·vocab)。"""
    return LLMConfig(
        num_layers=2, hidden_size=8, num_attention_heads=2, vocab_size=16,
        seq_length=16, batch_size=1, head_dim=4, attn_type="gqa", ffn_hidden_size=16,
    )


def _resolve_layers(spec, **pc_kw):
    pm = ParallelModel(ParallelConfig(**pc_kw), n_layers=len(spec.layer_pattern), world_size=64)
    g = ShapeEval().resolve(spec, pm)
    return [l for st in sorted(g.stages) for l in g.stages[st]], pm


def _D():
    return DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=16, B=1, vocab=16, n_layers=4)


# ─────────────────────────────────────────────────────────────────────────────
# 1. cp=2 把「激活派生」桶减半，参数桶（gather/grad）不变
# ─────────────────────────────────────────────────────────────────────────────

def test_cp2_halves_activation_buckets_params_unchanged():
    spec = build_llm_spec(_small_cfg())
    l1, _ = _resolve_layers(spec, cp=1)
    l2, _ = _resolve_layers(spec, cp=2)
    tot_saves1 = tot_saves2 = 0
    saw_ws = saw_bws = False
    for a, b in zip(l1, l2):
        sa = estimate_structure_memory(a.ops)
        sb = estimate_structure_memory(b.ops)
        # 激活 saves ÷cp（含去重后）
        assert sb.activation_saves * 2 == sa.activation_saves
        # 参数 full-gather / grad 缓冲（无 S、非本次路径）不变
        assert sb.param_full_bytes == sa.param_full_bytes
        assert sb.grad_full_bytes == sa.grad_full_bytes
        # workspace（flash∝S）÷cp
        if sa.workspace:
            assert sb.workspace * 2 == sa.workspace
            saw_ws = True
        # bwd_scratch（loss 8·S·B·vocab∝S）÷cp
        if sa.bwd_scratch:
            assert sb.bwd_scratch * 2 == sa.bwd_scratch
            saw_bws = True
        tot_saves1 += sa.activation_saves
        tot_saves2 += sb.activation_saves
    assert tot_saves1 > 0 and tot_saves2 * 2 == tot_saves1
    assert saw_ws and saw_bws       # workspace + bwd_scratch 两条 ÷cp 路径都被真实触发


# ─────────────────────────────────────────────────────────────────────────────
# 2. 持久 param 只随 fsdp=dp_shard*cp 变（既有路径），本次激活改动不泄漏进参数
# ─────────────────────────────────────────────────────────────────────────────

def test_cp_persistent_follows_fsdp_not_activation():
    spec = build_llm_spec(_small_cfg())
    opt = OptimizerSpec.adamw(params_fp32=True)

    def persistent(dp_shard, cp):
        layers, pm = _resolve_layers(spec, dp_shard=dp_shard, cp=cp)
        return sum(estimate_structure_memory(
            l.ops, fsdp=pm.fsdp_degree(), efsdp=pm.efsdp_degree(),
            opt_state_bytes=opt.state_bytes_per_param).persistent for l in layers)

    # fsdp=dp_shard*cp 恒定(=2) → 持久 param 完全相同（cp 只经 fsdp 影响参数，激活改动不碰参数）
    assert persistent(2, 1) == persistent(1, 2) > 0
    # dp_shard 固定纯增 cp → 持久减半：这是既有 fsdp=dp_shard*cp 的正确行为（非本次新增）
    assert persistent(1, 1) == 2 * persistent(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. cp=1 是逐字节 no-op（张量级）
# ─────────────────────────────────────────────────────────────────────────────

def test_cp1_is_noop():
    d = _D()
    pm = ParallelModel(ParallelConfig(cp=1), n_layers=4, world_size=1)
    x = TensorRef("x", ("S", "B", "H"), shard={0: "sp"})
    assert resolve_tensor(x, d, pm).local_numel == 16 * 1 * 8       # 无任何切分


# ─────────────────────────────────────────────────────────────────────────────
# 4. SP × CP 组合：SP 张量 → S/(tp·cp)；非 SP 全序列激活 → S/cp
# ─────────────────────────────────────────────────────────────────────────────

def test_sp_cp_composition():
    d = _D()
    pm = ParallelModel(
        ParallelConfig(tp=2, cp=2, dp_shard=1, sequence_parallel=True),
        n_layers=4, world_size=8)
    # SP 承载张量 shard={0:'sp'}：先 ÷sp(=tp) 再 ÷cp → S/(tp·cp)=16/4=4
    x = TensorRef("x", ("S", "B", "H"), shard={0: "sp"})
    assert resolve_tensor(x, d, pm).local_numel == (16 // (2 * 2)) * 1 * 8
    # 非 SP 全序列激活（tp 切在特征维、S 无 sp 标注）：seq→S/cp=8，特征→(n_heads·head_dim)/tp
    qkv = TensorRef("qkv", ("S", "B", "n_heads*head_dim"), shard={2: "tp"})
    assert resolve_tensor(qkv, d, pm).local_numel == (16 // 2) * 1 * ((2 * 4) // 2)


def test_non_divisible_seq_raises():
    import pytest
    d = _D()                                    # S=16
    pm = ParallelModel(ParallelConfig(cp=3), n_layers=4, world_size=1)  # 16 % 3 != 0
    with pytest.raises(ValueError):
        resolve_tensor(TensorRef("x", ("S", "B", "H"), shard={0: "sp"}), d, pm)


# ─────────────────────────────────────────────────────────────────────────────
# 5. index_scores(S²) 在 CP 下只去掉一个 S 因子 → S²/cp（ring：query 切、key 全量），非 S²/cp²
# ─────────────────────────────────────────────────────────────────────────────

def test_index_scores_s2_sheds_one_cp_factor():
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=16, B=2, vocab=16, n_layers=4)
    a = TensorRef("a", ("S", "B", "H"), shard={0: "sp"})
    op = OpSpec("indexer", OpType.MATMUL, [a], a, saves=[], bwd_scratch="4*B*S*S")
    spec = ModelSpec("t", d, ["x"], {"x": LayerSpec([op])})

    def bws(cp):
        pm = ParallelModel(ParallelConfig(cp=cp), n_layers=4, world_size=1)
        return ShapeEval().resolve(spec, pm).stages[0][0].ops[0].bwd_scratch_bytes

    base = 4 * 2 * 16 * 16
    assert bws(1) == base
    assert bws(2) == base // 2      # 一个 cp 因子（query 维切分）
    assert bws(4) == base // 4      # //cp（≠ //cp²=base//16）——守卫「切 query 维一次」


# ─────────────────────────────────────────────────────────────────────────────
# 6. MoE：dispatched-token 维（TLOCAL=S·B·topk·C∝S）+ staging workspace(∝S) 随 cp 减半
# ─────────────────────────────────────────────────────────────────────────────

def test_cp_shards_moe_dispatched_tokens():
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=16, B=1, vocab=16,
                 n_layers=4, n_experts=4, topk=2, moe_F=16)

    def sm(cp):
        spec = ModelSpec("t", d, ["x"], {"x": LayerSpec(build_moe_ffn_ops(d))})
        pm = ParallelModel(ParallelConfig(cp=cp, ep=1), n_layers=4, world_size=8)
        ops = ShapeEval().resolve(spec, pm).stages[0][0].ops
        return estimate_structure_memory(ops)

    s1, s2 = sm(1), sm(2)
    assert s2.activation_saves * 2 == s1.activation_saves     # disp/e_*/logits/comb 皆 ∝S
    assert s2.workspace * 2 == s1.workspace                   # MOE_STAGING_WS ∝S
    assert s2.param_full_bytes == s1.param_full_bytes         # 专家权重无 S，不变
