"""D-1：context-parallel（cp）切激活（含 2026-07-07 真机修正）。

守卫忠实模型（specs/2026-07-06-audit-remediation.md §4·D-1 + 「D-1 修正」）：cp 是全局序列并行域，
**decoder body** 的每个非权重激活 token/query 维 ÷cp（含 flash workspace / index bwd_scratch）；
**参数**（gather_buf/grad_buf/持久 param）不被触碰。

**D-1 修正（2026-07-07，真机 cp=2 + 算子级 Profiler 确认）**：
  - **loss/head 区对所有 cp 算法都是 full-S**（`cp_shard=False`）——head 前 hidden all-gather 回 full-S，
    真机峰实测 logsm/probs/grad_log_softmax 各 2020 MiB 满 vocab full-S（**非** S/cp）。故 loss 层
    activation_saves 与 nll `bwd_scratch=8·S·B·vocab` **不 ÷cp**（旧「整体 ÷cp」欠估 ~29%，8839 vs 12433）。
  - **body ÷cp 依 `context_parallel_method`**：ulysses/ring/hybrid KV 随 body ÷cp；`colossal`
    （ulysses_degree=1）把 attention KV（`cp_kv=True`）all-gather 到 full-S。
硬不变量：cp=1 逐字节 no-op（两处 cp 分支 `if cp>1` 不进入）。
"""
from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
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
# 1. cp=2：**所有激活**（decoder body + loss/head）÷cp；参数桶不变
#    （2026-07-07 Bug A 再修正——`analysis/realmachine/cp2_none/` profiler 证 loss buffer=[S/cp,B,V]
#     → loss/head 区在 cp 下**是 ÷cp（序列并行）**。此前 D-1 误判 full-S 是把 [B=2,S/cp] 误读成
#     [B=1,full-S]（数值都 2020），cp=2 full 的「0.996」是 B 与 cp 数值抵消蒙对。现全激活 ÷cp。）
# ─────────────────────────────────────────────────────────────────────────────

def test_cp2_all_activations_halve_including_loss():
    spec = build_llm_spec(_small_cfg())   # embedding + 2×(gqa+dense) + lm_head
    l1, _ = _resolve_layers(spec, cp=1)
    l2, _ = _resolve_layers(spec, cp=2)
    saw_body_saves = saw_ws = saw_head = False
    for a, b in zip(l1, l2):
        sa = estimate_structure_memory(a.ops)
        sb = estimate_structure_memory(b.ops)
        # 参数 full-gather / grad 缓冲（无 S、非本次路径）——所有层恒不变
        assert sb.param_full_bytes == sa.param_full_bytes
        assert sb.grad_full_bytes == sa.grad_full_bytes
        # loss/head 层 = 唯一带 bwd_scratch（loss 8·S·B·vocab）的层 → **也 ÷cp**（Bug A 修正）：
        #   activation_saves（h_final+logits+logsm）与 bwd_scratch 均随 cp ÷cp（真机 [S/cp,B,V]）。
        if sa.bwd_scratch:
            assert sb.activation_saves * 2 == sa.activation_saves   # loss/head saves ÷cp
            assert sb.bwd_scratch * 2 == sa.bwd_scratch             # nll bwd_scratch ÷cp
            saw_head = True
        else:
            # decoder body / embedding：激活 saves ÷cp、workspace（flash∝S）÷cp
            assert sb.activation_saves * 2 == sa.activation_saves
            if sa.activation_saves:
                saw_body_saves = True
            if sa.workspace:
                assert sb.workspace * 2 == sa.workspace
                saw_ws = True
    # 三条路径都被真实触发：body saves ÷cp、body flash-ws ÷cp、loss/head 亦 ÷cp
    assert saw_body_saves and saw_ws and saw_head


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
    # cp=1 下 cp_shard / cp_kv / context_parallel_method 全部无效（分支不进入）→ 逐字节 no-op：
    full = TensorRef("logsm", ("S", "B", "vocab"), cp_shard=False)         # loss 区标记也 no-op
    kv = TensorRef("kvb_out", ("S", "B", "H"), cp_kv=True)                 # KV 标记也 no-op
    for method in ("colossal", "ulysses", "ring", "hybrid"):
        pmm = ParallelModel(ParallelConfig(cp=1, context_parallel_method=method), n_layers=4, world_size=1)
        assert resolve_tensor(x, d, pmm).local_numel == 16 * 1 * 8
        assert resolve_tensor(full, d, pmm).local_numel == 16 * 1 * 16
        assert resolve_tensor(kv, d, pmm).local_numel == 16 * 1 * 8


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


# ─────────────────────────────────────────────────────────────────────────────
# 7. Bug A 修正（2026-07-07 cp2-none profiler）：loss/head 区**是 ÷cp（序列并行）**——非 full-S
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_head_region_is_cp_sharded():
    """loss/head 区随 cp ÷cp（真机 cp2-none profiler：loss buffer=[S/cp,B,V]）。head.py 现用默认
    cp_shard=True。此前 D-1 误判 full-S（把 [B=2,S/cp] 误读成 [B=1,full-S]）已推翻。"""
    d = _D()                                    # S=16, B=1, H=8, vocab=16
    pm2 = ParallelModel(ParallelConfig(cp=2), n_layers=4, world_size=2)
    # head.py 的 loss/head 张量现为默认 cp_shard=True → 序列维 ÷cp
    logsm = TensorRef("logsm", ("S", "B", "vocab"), dtype_bytes=4)
    h_final = TensorRef("h_final", ("S", "B", "H"), shard={0: "sp"})
    assert resolve_tensor(logsm, d, pm2).local_numel == (16 // 2) * 1 * 16   # S/cp=8（非 full 16）
    assert resolve_tensor(h_final, d, pm2).local_numel == (16 // 2) * 1 * 8  # S/cp=8


def test_loss_bwd_scratch_cp_sharded_via_output_marker():
    """nll op 输出 `loss`(默认 cp_shard=True) → 其 bwd_scratch=8·S·B·vocab 在 cp 下 ÷cp（真机 [S/cp,B,V]）。"""
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=16, B=1, vocab=16, n_layers=4)
    loss = TensorRef("loss", ("B",))            # 默认 cp_shard=True
    nll = OpSpec("nll", OpType.ELEMENTWISE, [], loss, saves=[], bwd_scratch="8*S*B*vocab")
    spec = ModelSpec("t", d, ["x"], {"x": LayerSpec([nll])})

    def bws(cp):
        pm = ParallelModel(ParallelConfig(cp=cp), n_layers=4, world_size=1)
        return ShapeEval().resolve(spec, pm).stages[0][0].ops[0].bwd_scratch_bytes

    base = 8 * 16 * 1 * 16
    assert bws(1) == base
    assert bws(2) == base // 2       # loss bwd_scratch ÷cp（cp=2 减半，Bug A 修正）


# ─────────────────────────────────────────────────────────────────────────────
# 8. D-1 修正 B：cp_kv 张量在 colossal 下 full-S（KV all-gather）；其余算法随 body ÷cp
# ─────────────────────────────────────────────────────────────────────────────

def test_colossal_kv_full_s_others_shard():
    d = _D()                                    # S=16, B=1, H=8
    kv = TensorRef("kvb_out", ("S", "B", "H"), cp_kv=True)   # attention KV 侧
    q = TensorRef("qb_out", ("S", "B", "H"))                 # 非 KV body（默认 cp_shard=True, cp_kv=False）
    # colossal：KV all-gather → full-S；body 仍 ÷cp
    pm_col = ParallelModel(ParallelConfig(cp=2, context_parallel_method="colossal"), n_layers=4, world_size=2)
    assert resolve_tensor(kv, d, pm_col).local_numel == 16 * 1 * 8         # KV full-S=16
    assert resolve_tensor(q, d, pm_col).local_numel == (16 // 2) * 1 * 8   # body S/cp=8
    # ulysses/ring/hybrid：KV 随 body ÷cp（无 all-gather）
    for m in ("ulysses", "ring", "hybrid"):
        pm = ParallelModel(ParallelConfig(cp=2, context_parallel_method=m), n_layers=4, world_size=2)
        assert resolve_tensor(kv, d, pm).local_numel == (16 // 2) * 1 * 8  # KV S/cp=8


def test_cp_method_fail_loud_and_accepts_valid():
    import pytest
    with pytest.raises(ValueError):
        ParallelConfig(cp=2, context_parallel_method="megatron_cp")   # 未实现 → fail-loud
    for m in ("colossal", "ulysses", "ring", "hybrid"):
        assert ParallelConfig(context_parallel_method=m).context_parallel_method == m
    assert ParallelConfig().context_parallel_method == "colossal"     # 默认对齐 mindformers DSv3 yaml


# ─────────────────────────────────────────────────────────────────────────────
# 9. 验证靶（真机硬门）：cp=2 4L full-recompute 预测 ≈ 真机（colossal 12433 / ulysses 12441）
#    修正前 buggy=8839（loss 区错半，欠估 ~29%）；修正后 loss/head full-S → ~12381（band 12000-12600）
# ─────────────────────────────────────────────────────────────────────────────

def test_cp2_4L_full_recompute_matches_real_machine():
    from validate_dsv3 import build_dsv3_spec
    from cost_eval.report import Evaluator
    GiB = 2 ** 30
    MiB = 2 ** 20
    spec, d, fl = build_dsv3_spec(4)
    d.B = 2                        # per-device batch = global_batch(2)/dp(1) = 2（真机口径；Bug A 后 loss ÷cp，B 须对齐）
    peaks = {}
    for method in ("colossal", "ulysses"):
        pc = ParallelConfig(dp_shard=1, cp=2, tp=1, pp=1, sequence_parallel=False,
                            num_microbatches=1, context_parallel_method=method)
        ev = Evaluator(spec, pc,
                       OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                       HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0),
                       RecomputeSpec(mode="full", full_layers=fl), SwapSpec())
        peaks[method] = ev.evaluate().per_stage[0].peak_bytes / MiB
        # 真机 colossal 12433 / ulysses 12441；Bug A(loss÷cp) + 正确 B=2 → ~12409.5（band 12000-12600）。
        assert 12000 <= peaks[method] <= 12600, f"{method}: {peaks[method]:.1f} MiB 越界"
    # 全重算下 colossal==ulysses（KV all-gather 在重算层、off loss 峰）——修正模型的预期恒等
    assert peaks["colossal"] == peaks["ulysses"]
