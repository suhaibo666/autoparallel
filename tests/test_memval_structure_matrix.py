"""内存仿真验证（测试工程视角，2026-07-16）：模型结构 × 并行 × 重计算 三维矩阵不变量。

覆盖「各类模型结构」维度：9 种结构（GQA/MHA/ungated/GQA+MoE 共享专家门控/MLA+MoE/
tied+MTP/mHC/DSA/DSv4-hybrid）× 6 种并行（dp2/tp2+sp/pp2/cp2/vpp2/hybrid）× 3 种重计算
（None/full/select），每组合跑全链路 Evaluator 并断言**结构无关的通用不变量**：

  I1  会计闭合：每个时间线采样点 total == Σ 14 桶（含 framework）——桶漏记/多记即破；
  I2  非负：任意采样点任意桶 ≥ 0；
  I3  峰值一致：StagePeak.peak_bytes == max(时间线 total)；
  I4  重算偏序：max act_live(full) < max act_live(None) 且 full ≤ select ≤ None
      ——对任意结构/并行/调度成立（ci ⊆ select-pinned ⊆ 全 saves）；
  I5  参数守恒：对固定结构，Σ_stage(每卡持久 el × 该 stage 卡数) 在所有并行配置下
      == tp 切分权重全量 + tp 复制权重全量 × tp（基线自举、跨配置互证——多切/漏切即破）。

绝对字节值不在本文件断言（那是 test_memval_hybrid_recompute.py 的解析用例与
DSv3/DSv4 golden 的职责）；本矩阵的价值是把不变量铺满结构×并行×重算的组合空间。
"""
import pytest

from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec
from cost_eval.specs import ParallelConfig, RecomputeSpec, SwapSpec, OptimizerSpec, HardwareSpec
from cost_eval.report import Evaluator

# ---------------------------------------------------------------------------
# 结构矩阵：全部维度经挑选，对 tp2 / cp2 / fsdp4 / ep2 整除
# ---------------------------------------------------------------------------

_BASE = dict(num_layers=4, hidden_size=64, num_attention_heads=4, head_dim=16,
             vocab_size=256, seq_length=64, batch_size=1, ffn_hidden_size=128)
_MLA = dict(q_lora_rank=32, kv_lora_rank=32, qk_rope_head_dim=16,
            qk_nope_head_dim=16, v_head_dim=16)
_MOE = dict(num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=64)

STRUCTS = {
    "gqa_dense": LLMConfig(**_BASE, num_query_groups=2),
    "mha_dense": LLMConfig(**_BASE, attn_type="mha"),
    "gqa_ungated": LLMConfig(**_BASE, num_query_groups=2, gated_linear_unit=False),
    "gqa_moe_shared_gate": LLMConfig(**_BASE, num_query_groups=2, **_MOE,
                                     moe_shared_expert_num=1,
                                     moe_shared_ffn_hidden_size=64,
                                     moe_shared_expert_gating=True,
                                     first_k_dense_replace=1),
    "mla_moe": LLMConfig(**_BASE, attn_type="mla", **_MLA, **_MOE,
                         first_k_dense_replace=1),
    "gqa_tied_mtp": LLMConfig(**_BASE, num_query_groups=2,
                              tie_word_embeddings=True, mtp_num_layers=1),
    "gqa_mhc": LLMConfig(**_BASE, num_query_groups=2,
                         residual_variant="mhc", num_residual_streams=2),
    "dsa_dense": LLMConfig(**_BASE, attn_type="dsa", **_MLA,
                           dsa_indexer_n_heads=2, dsa_indexer_head_dim=16,
                           dsa_indexer_topk=16),
    "dsv4_moe": LLMConfig(**_BASE, attn_type="dsv4_hybrid", **_MLA,
                          o_groups=2, o_lora_rank=16,
                          csa_compress_ratios=(1, 1, 4, 4), csa_window_size=32,
                          dsa_indexer_n_heads=2, dsa_indexer_head_dim=16,
                          dsa_indexer_topk=16, **_MOE, first_k_dense_replace=1,
                          cross_entropy_fused=True),
}


def _is_moe(name: str) -> bool:
    return "moe" in name


def _parallels(moe: bool) -> dict:
    ep = 2 if moe else 1
    return {
        "dp2": ParallelConfig(dp_shard=2, num_microbatches=1),
        "tp2sp": ParallelConfig(tp=2, sequence_parallel=True, num_microbatches=1),
        "pp2": ParallelConfig(pp=2, num_microbatches=4),
        "cp2": ParallelConfig(cp=2, num_microbatches=1),
        "vpp2": ParallelConfig(pp=2, interleave=2, num_microbatches=4),
        "hybrid": ParallelConfig(dp_shard=2, tp=2, pp=2, ep=ep,
                                 sequence_parallel=True, num_microbatches=4),
    }


def _recomputes() -> dict:
    dec = range(1, 5)                       # decoder 层 id（0=embedding）
    return {
        "none": RecomputeSpec("None"),
        "full": RecomputeSpec("full", set(dec)),
        "sel_fc": RecomputeSpec("select", select_ops={l: {"fc"} for l in dec}),
    }


def _evaluate(cfg: LLMConfig, pc: ParallelConfig, rec: RecomputeSpec):
    return Evaluator(build_llm_spec(cfg), pc, OptimizerSpec.adamw(),
                     HardwareSpec(max_device_memory=1 << 50, alloc_block_bytes=1),
                     rec, SwapSpec()).evaluate(record_timeline=True)


def _bd_fields(bd) -> list:
    return [bd.persistent, bd.act_live, bd.gather_buf, bd.grad_buf,
            bd.recomp_scratch, bd.bwd_scratch, bd.bwd_working_set, bd.swap_buf,
            bd.workspace, bd.optstep, bd.framework, bd.kept_frag,
            bd.grad_accum, bd.p2p_buf]


def _max_act(rep) -> int:
    return max(t.breakdown.act_live for p in rep.per_stage for t in p.timeline)


_MATRIX = [(s, p, r) for s in STRUCTS
           for p in _parallels(_is_moe(s))
           for r in _recomputes()]


@pytest.mark.parametrize("sname,pname,rname", _MATRIX,
                         ids=[f"{s}-{p}-{r}" for s, p, r in _MATRIX])
def test_invariants_accounting_nonneg_peak(sname, pname, rname):
    """I1 会计闭合 + I2 非负 + I3 峰值一致：对 9 结构 × 6 并行 × 3 重算全组合成立。"""
    rep = _evaluate(STRUCTS[sname], _parallels(_is_moe(sname))[pname],
                    _recomputes()[rname])
    assert rep.per_stage, "至少一个 stage"
    for p in rep.per_stage:
        assert p.peak_bytes > 0
        assert p.timeline, "record_timeline 应产出事件序列"
        for t in p.timeline:
            fields = _bd_fields(t.breakdown)
            assert all(v >= 0 for v in fields), (p.stage, t.event, fields)
            assert sum(fields) == t.total_bytes, (p.stage, t.event)
        assert p.peak_bytes == max(t.total_bytes for t in p.timeline)


@pytest.mark.parametrize("sname,pname", [(s, p) for s in STRUCTS
                                         for p in _parallels(True)],
                         ids=[f"{s}-{p}" for s in STRUCTS for p in _parallels(True)])
def test_invariant_recompute_ordering(sname, pname):
    """I4 重算偏序：max act_live 满足 full < none 且 full ≤ select ≤ none，
    对每个结构×并行组合成立（含 VPP/hybrid 调度）。"""
    cfg, pc = STRUCTS[sname], _parallels(_is_moe(sname))[pname]
    recs = _recomputes()
    a_none = _max_act(_evaluate(cfg, pc, recs["none"]))
    a_full = _max_act(_evaluate(cfg, pc, recs["full"]))
    a_sel = _max_act(_evaluate(cfg, pc, recs["sel_fc"]))
    assert a_full < a_none, "full 重算必须严格降低驻留激活"
    assert a_full <= a_sel <= a_none, "select 介于 full 与 none 之间"


@pytest.mark.parametrize("sname", list(STRUCTS), ids=list(STRUCTS))
def test_invariant_param_conservation_per_structure(sname):
    """I5 参数守恒：对固定结构，全集群持久 numel 在 6 种并行配置下互等
    （= 基线全量中 tp 切分部分 + tp 复制部分 × tp）。tp 复制量（norm gamma/router 等
    不随 tp 切的权重）由 tp2sp 与基线的差自举求出，再用 hybrid 组合交叉验证。"""
    cfg = STRUCTS[sname]
    rec = RecomputeSpec("None")

    def cluster_el(pc: ParallelConfig) -> int:
        rep = _evaluate(cfg, pc, rec)
        ranks = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp
        return sum((p.breakdown.persistent // 14) * ranks for p in rep.per_stage)

    base = cluster_el(ParallelConfig())
    pcs = _parallels(_is_moe(sname))
    # tp=1 的配置必须逐一守恒到基线
    for name in ("dp2", "pp2", "cp2", "vpp2"):
        assert cluster_el(pcs[name]) == base, name
    # tp=2：复制权重多出一份 → 差值 = 复制量 rep_el ≥ 0。
    rep_el = cluster_el(pcs["tp2sp"]) - base
    assert rep_el >= 0
    # P0-2(2026-07-23,runtime 377c9c344):tp2sp 是 **ep=1** → expert 权重 TP 复制(随父层
    # dense wrap,parallelize.py:700-716),其复制量含在 rep_el;hybrid 是 **ep=2** → experts
    # 走 EP 切、无 TP 复制 → 期望 = base + rep_el − expert_global×(tp−1)。非 MoE 结构
    # expert_global=0,退化回原式「hybrid 与 tp2sp 复制量一致」。
    exp_g = _expert_global_el(cfg)
    assert cluster_el(pcs["hybrid"]) == base + rep_el - exp_g * (2 - 1),         "hybrid(ep=2) 与 tp2sp(ep=1) 的 tp 复制量差应恰为 expert 全量(P0-2 ep 退化语义)"


def _expert_global_el(cfg) -> int:
    """结构的 expert 权重全局 numel（并行全 1 时 local==global;按名去重/层）。"""
    from cost_eval.parallel_model import ParallelModel
    from cost_eval.shape_eval import ShapeEval
    spec = build_llm_spec(cfg)
    pm = ParallelModel(ParallelConfig(), spec.dims.n_layers, world_size=1)
    g = ShapeEval().resolve(spec, pm)
    total = 0
    for layers in g.stages.values():
        for layer in layers:
            seen = {}
            for op in layer.ops:
                for w in op.params:
                    if getattr(w, "is_expert", False):
                        seen[w.name] = w.local_numel
            total += sum(seen.values())
    return total
