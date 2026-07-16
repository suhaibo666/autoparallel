"""内存仿真验证（测试工程视角，2026-07-16）：多维混合并行 × 重计算的解析级用例。

针对覆盖盘点确认的缺口（现有 905 用例之外）：
  ①  TP 作为独立激活维度无逐字节断言（CP 有整套、TP 没有）；
  ②  多维混合并行（tp×pp×ep×dp 同开）无任何解析/黄金峰值断言（仅 smoke/可行性）；
  ③  重计算 × VPP（interleave>1）零覆盖——所有 VPP 用例都是 RecomputeSpec("None")；
  ④  重计算(select) × CP 无解析断言。

方法论：**期望值全部独立手工推导**（由 layers/*.py 的声明式 op 图逐张量列式，
不从实现取数）。dense GQA 层 op 序列（attention.py + ffn.py，去重后 saves 共 9 项）：
  ln1(saves x) → qkv(saves ln1) → rope → flash(saves qkv,attn,fa_stats)
  → o_proj(saves attn，去重) → add1 → ln2(saves h1) → fc1(saves ln2)
  → swiglu(saves g) → fc2(saves act) → add2
切分口径（shape_eval.resolve_tensor）：
  x/h1 [S,B,H]{0:sp} → S/(sp·cp)；ln1/ln2 [S,B,H] → S/cp；
  qkv [S,B,(n_heads+2n_kv)·hd]{2:tp} → ÷tp、S/cp；attn 同理；
  fa_stats [2,B,n_heads,S,8] fp32 {2:tp} → n_heads/tp、S/cp；
  g [S,B,2F]{2:tp}、act [S,B,F]{2:tp}。
直调 simulate 路径 norm_compute_dtype_bytes 缺省 0（bf16 口径）、alloc_block=1，
与既有 toy 测试（test_vpp_activation.py）同基准。
"""
import pytest

from cost_eval.mem_timeline import MemTimeline
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.specs import ParallelConfig, RecomputeSpec, SwapSpec, OptimizerSpec, HardwareSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec
from cost_eval.report import Evaluator


# ---------------------------------------------------------------------------
# 手工推导公式（独立于实现：逐张量按 op 图列式）
# ---------------------------------------------------------------------------

def dense_saves(d: DimTable, tp: int = 1, cp: int = 1, sp_on: bool = True) -> int:
    """dense GQA 层去重后 saves 总字节（无重算时整层 pin 进 act_live 的量）——手工列式。"""
    sp = tp if sp_on else 1
    qkv_cols = (d.n_heads + 2 * d.n_kv) * d.head_dim
    return (
        (d.S // (sp * cp)) * d.B * d.H * 2                     # x   （ln1 保留）
        + (d.S // cp) * d.B * d.H * 2                          # ln1 （qkv 保留）
        + (d.S // cp) * d.B * (qkv_cols // tp) * 2             # qkv （flash 保留）
        + (d.S // cp) * d.B * (d.n_heads * d.head_dim // tp) * 2   # attn（flash+o_proj，去重一次）
        + _fa_bytes(d, tp, cp)                                 # fa_stats（flash 保留，fp32）
        + (d.S // (sp * cp)) * d.B * d.H * 2                   # h1  （ln2 保留）
        + (d.S // cp) * d.B * d.H * 2                          # ln2 （fc1 保留）
        + (d.S // cp) * d.B * (2 * d.F // tp) * 2              # g   （swiglu 保留）
        + (d.S // cp) * d.B * (d.F // tp) * 2                  # act （fc2 保留）
    )


def _fa_bytes(d: DimTable, tp: int, cp: int) -> int:
    """fa_stats / fa_ws 字节：[2,B,n_heads/tp,S/cp,8] fp32。"""
    return 2 * d.B * (d.n_heads // tp) * (d.S // cp) * 8 * 4


def _qkv_bytes(d: DimTable, tp: int, cp: int) -> int:
    return (d.S // cp) * d.B * ((d.n_heads + 2 * d.n_kv) * d.head_dim // tp) * 2


def _attn_bytes(d: DimTable, tp: int, cp: int) -> int:
    return (d.S // cp) * d.B * (d.n_heads * d.head_dim // tp) * 2


def checkpoint_input(d: DimTable, tp: int = 1, cp: int = 1, sp_on: bool = True) -> int:
    """full 重算时保留的层入口 = 首个有 saves 的 op（ln1）的首个 save（x）。"""
    sp = tp if sp_on else 1
    return (d.S // (sp * cp)) * d.B * d.H * 2


def select_flash_pinned(d: DimTable, tp: int = 1, cp: int = 1, sp_on: bool = True) -> int:
    """select {"flash"} 时每层常驻激活 = 全 saves − 仅被 flash 保留的 {qkv, fa_stats}
    （attn 由非选中 o_proj 保留仍常驻；层入口 x 本就在非选中 ln1 的 saves 里）。"""
    return (dense_saves(d, tp, cp, sp_on)
            - _qkv_bytes(d, tp, cp) - _fa_bytes(d, tp, cp))


def select_flash_recomp(d: DimTable, tp: int = 1, cp: int = 1) -> int:
    """select {"flash"} 的反向重物化 = forward_max_live([flash])：
    活跃集 = 输入 qkv + 输出 attn + workspace_ref fa_ws；
    进入边界 qkv 未被 pin（唯一 saver 是 flash 自己，已随选中丢弃）→ 不扣。"""
    return _qkv_bytes(d, tp, cp) + _attn_bytes(d, tp, cp) + _fa_bytes(d, tp, cp)


def dense_forward_max_live(d: DimTable) -> int:
    """dense 层 forward max-live（tp=cp=1，B/S 由 d 定）——手工时间线：
    峰在 flash 步（qkv+attn+fa_ws）与 swiglu 步（g+act），取 max。"""
    flash_peak = _qkv_bytes(d, 1, 1) + _attn_bytes(d, 1, 1) + _fa_bytes(d, 1, 1)
    swiglu_peak = (d.S * d.B * 2 * d.F * 2) + (d.S * d.B * d.F * 2)
    return max(flash_peak, swiglu_peak)


# ---------------------------------------------------------------------------
# 直调 simulate 的 toy 装置（沿用 test_vpp_activation 风格：无 embedding/head 伪层）
# ---------------------------------------------------------------------------

def _sim(d: DimTable, pc: ParallelConfig, rec: RecomputeSpec, n_layers: int):
    spec = ModelSpec("toy", d, ["dense"] * n_layers, {"dense": build_dense_decoder(d)})
    world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
    pm = ParallelModel(pc, n_layers=n_layers, world_size=world, edge_pseudo=(0, 0))
    g = ShapeEval().resolve(spec, pm)
    zeros = {s: 0 for s in g.stages}
    return MemTimeline().simulate(g, rec, SwapSpec(), pm, zeros,
                                  framework_reserve=0, max_device_memory=1 << 60,
                                  record_timeline=True)


def _max_bucket(sp, field: str) -> int:
    return max(getattr(t.breakdown, field) for t in sp.timeline)


# toy 维度：全部维度对 tp=2/cp=2/sp=2 整除
D_HY = DimTable(H=64, F=128, n_heads=4, n_kv=2, head_dim=16, S=128, B=2, vocab=128, n_layers=4)
# 与 test_vpp_activation 同款（n_kv=4, B=1），便于与既有 VPP 锚点交叉对照
D_VPP = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=4)


# ---------------------------------------------------------------------------
# ① TP（含 SP）作为激活维度：逐字节解析断言 + TP×CP 乘法合成
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp,cp", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_tp_cp_activation_saves_analytic(tp, cp):
    """无重算、pp=1、m=1：整段 FWD 结束时 act_live = Σ两层 saves —— 与手工列式逐字节相等。
    覆盖 TP 单独（缺口④' TP 无字节断言）与 TP×CP 同开（乘法合成不重切/不漏切）。"""
    pc = ParallelConfig(tp=tp, cp=cp, sequence_parallel=True, num_microbatches=1)
    sp = _sim(D_HY, pc, RecomputeSpec("None"), n_layers=2)[0]
    assert _max_bucket(sp, "act_live") == 2 * dense_saves(D_HY, tp, cp)


def test_tp2_shards_exactly_the_sharded_tensors_only():
    """tp=2（sp 同开）相对 tp=1 的削减恰好 = 被 tp 切的张量之半（qkv/attn/fa/g/act）
    加 sp 切的 x/h1 之半——非均匀 ÷2（ln1/ln2 不随 tp 切），防「全体一刀切」类错误建模。"""
    s1, s2 = dense_saves(D_HY, 1, 1), dense_saves(D_HY, 2, 1)
    unsharded = 2 * (D_HY.S * D_HY.B * D_HY.H * 2)          # ln1 + ln2 不切
    assert s2 == unsharded + (s1 - unsharded) // 2
    # 仿真侧同款（间接经 test_tp_cp_activation_saves_analytic 的字节相等保证，此处锁关系式）
    pc1 = ParallelConfig(tp=1, sequence_parallel=True, num_microbatches=1)
    pc2 = ParallelConfig(tp=2, sequence_parallel=True, num_microbatches=1)
    a1 = _max_bucket(_sim(D_HY, pc1, RecomputeSpec("None"), 2)[0], "act_live")
    a2 = _max_bucket(_sim(D_HY, pc2, RecomputeSpec("None"), 2)[0], "act_live")
    assert a2 == 2 * unsharded + (a1 - 2 * unsharded) // 2


# ---------------------------------------------------------------------------
# ② 混合并行 1F1B：tp2×cp2×pp2 的在飞微批激活（缺口②的激活分量）
# ---------------------------------------------------------------------------

def test_hybrid_tp_cp_pp_1f1b_inflight_activation():
    """tp=2×cp=2×pp=2（sp 开）、m=4、每 stage 2 层：
    stage0 warmup=1 → 峰 2 在飞微批 × 2 层 = 4·s(2,2)；stage1 warmup=0 → 2·s(2,2)。
    同时验证 1F1B 在飞倍数与 tp/cp 切分在同一次仿真里正确复合。"""
    pc = ParallelConfig(tp=2, cp=2, pp=2, sequence_parallel=True, num_microbatches=4)
    res = _sim(D_HY, pc, RecomputeSpec("None"), n_layers=4)
    s = dense_saves(D_HY, tp=2, cp=2)
    assert _max_bucket(res[0], "act_live") == 4 * s
    assert _max_bucket(res[1], "act_live") == 2 * s


# ---------------------------------------------------------------------------
# ③ select 重算 × CP：三桶解析值 + 精确减半（缺口③）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cp", [1, 2])
def test_select_flash_cp_pinned_and_recomp_analytic(cp):
    """select {"flash"}（丢 qkv+fa_stats、重物化 flash）在 cp∈{1,2} 下：
    act_live = 2 层 × pinned 手工值；recomp_scratch = fml([flash]) 手工值（边界 qkv 未 pin 不扣）。"""
    pc = ParallelConfig(cp=cp, num_microbatches=1)
    rec = RecomputeSpec("select", select_ops={0: {"flash"}, 1: {"flash"}})
    sp = _sim(D_HY, pc, rec, n_layers=2)[0]
    assert _max_bucket(sp, "act_live") == 2 * select_flash_pinned(D_HY, 1, cp, sp_on=False)
    assert _max_bucket(sp, "recomp_scratch") == select_flash_recomp(D_HY, 1, cp)


def test_select_flash_cp2_exactly_halves_cp1():
    """cp=2 相对 cp=1：select 三桶（act_live/recomp/bwd_working_set）与 bwd_scratch 均精确减半
    （body 全部张量含 S 因子 → ÷cp 无一例外），且非零防空转。"""
    rec = RecomputeSpec("select", select_ops={0: {"flash"}, 1: {"flash"}})
    sp1 = _sim(D_HY, ParallelConfig(cp=1, num_microbatches=1), rec, 2)[0]
    sp2 = _sim(D_HY, ParallelConfig(cp=2, num_microbatches=1), rec, 2)[0]
    for f in ("act_live", "recomp_scratch", "bwd_working_set"):
        v1, v2 = _max_bucket(sp1, f), _max_bucket(sp2, f)
        assert v1 > 0, f
        assert v2 * 2 == v1, f


# ---------------------------------------------------------------------------
# ④ VPP × 重计算（缺口③'：现有 VPP 用例全部不重算）
# ---------------------------------------------------------------------------

def test_vpp_full_recompute_act_is_per_chunk_checkpoint_sum():
    """pp=2, v=2, m=8, L=4（每 chunk 1 层）、全层 full 重算：
    峰 5 个在飞虚拟步 × 每步 1 层 × ci —— act_live = 5·ci（chunk 粒度 × checkpoint 输入的复合）。
    v=1 对照：2 在飞微批 × 2 层 × ci = 4·ci。"""
    ci = checkpoint_input(D_VPP, sp_on=False)
    rec = RecomputeSpec("full", {0, 1, 2, 3})
    v2 = _sim(D_VPP, ParallelConfig(pp=2, num_microbatches=8, interleave=2), rec, 4)[0]
    v1 = _sim(D_VPP, ParallelConfig(pp=2, num_microbatches=8), rec, 4)[0]
    assert _max_bucket(v2, "act_live") == 5 * ci
    assert _max_bucket(v1, "act_live") == 4 * ci


def test_vpp_full_recompute_scratch_is_schedule_independent():
    """recomp_scratch = fml(层) − ci 是**层属性**，不随调度（v=1/v=2）变：
    手工时间线 fml = max(flash 步 qkv+attn+fa_ws, swiglu 步 g+act) = 98304（D_VPP 维度）。"""
    expected = dense_forward_max_live(D_VPP) - checkpoint_input(D_VPP, sp_on=False)
    rec = RecomputeSpec("full", {0, 1, 2, 3})
    for v in (1, 2):
        sp = _sim(D_VPP, ParallelConfig(pp=2, num_microbatches=8, interleave=v), rec, 4)[0]
        assert _max_bucket(sp, "recomp_scratch") == expected, f"v={v}"


def test_vpp_select_recompute_act_analytic_and_ordering():
    """pp=2, v=2, m=8、全层 select {"flash"}：act_live = 5 × pinned 手工值；
    并锁 full < select < none 的 VPP 序（重算越多驻留越少，调度不破坏偏序）。"""
    pc = lambda: ParallelConfig(pp=2, num_microbatches=8, interleave=2)
    sel = RecomputeSpec("select", select_ops={l: {"flash"} for l in range(4)})
    sp_sel = _sim(D_VPP, pc(), sel, 4)[0]
    assert _max_bucket(sp_sel, "act_live") == 5 * select_flash_pinned(D_VPP, sp_on=False)
    a_full = _max_bucket(_sim(D_VPP, pc(), RecomputeSpec("full", {0, 1, 2, 3}), 4)[0], "act_live")
    a_none = _max_bucket(_sim(D_VPP, pc(), RecomputeSpec("None"), 4)[0], "act_live")
    a_sel = _max_bucket(sp_sel, "act_live")
    assert a_full < a_sel < a_none


# ---------------------------------------------------------------------------
# ⑤ 端到端混合并行（tp2×ep2×dp2×pp2）持久态/梯度累计/optstep：逐字节解析（缺口②核心）
# ---------------------------------------------------------------------------

# GQA+MoE 小模型：所有维度对 tp2/ep4/fsdp4 整除
_MOE_CFG = LLMConfig(
    num_layers=4, hidden_size=64, num_attention_heads=4, num_query_groups=2,
    head_dim=16, vocab_size=256, seq_length=64, batch_size=1, ffn_hidden_size=128,
    num_moe_experts=4, moe_router_topk=2, moe_ffn_hidden_size=64, moe_layer_freq=1,
)

# ── 每权重全局 numel（手工，对照 layers/*.py 的 TensorRef 形状）──────────────────
_H, _NKV_COLS, _NHD = 64, (4 + 2 * 2) * 16, 4 * 16
_W_DEC_DENSE = {                      # 随 tp 切、÷fsdp 的 dense 权重（GQA attn 段）
    "qkv_w": _H * _NKV_COLS,          # 8192
    "o_w": _NHD * _H,                 # 4096
}
# 2026-07-16: MoE 层补 ln2_g（post_attention_layernorm gamma [H] fp32），与 ln1_g 同为
# norm gamma —— TP 组内复制、仅 ÷fsdp，故入 NOSHARD 桶（每 MoE 层 +_H el）。
_W_DEC_NOSHARD = {"ln1_g": _H, "ln2_g": _H, "router_w": 4 * _H}  # 不随 tp 切（router fp32 [E,H]）
_W_DEC_EXPERT = {"e_w1": 4 * _H * (2 * 64), "e_w2": 4 * 64 * _H}   # ÷ep 再 ÷efsdp
_EMB = 256 * _H                        # emb_w [vocab,H] ÷tp
_HEAD = _H * 256                       # head_w [H,vocab] ÷tp
_FN_G = _H                             # final_norm_g

TOTAL_EL = (_EMB + _HEAD + _FN_G
            + 4 * (sum(_W_DEC_DENSE.values()) + sum(_W_DEC_NOSHARD.values())
                   + sum(_W_DEC_EXPERT.values())))          # = 280128 (含 4×ln2_g)
# TP 组内**复制**（不随 tp 切、仅 ÷fsdp）的权重：norm gamma(ln1_g+ln2_g) + router fp32（真机独立
# FSDP wrap、TP 各 rank 各持一份）。集群守恒须计 ×tp 的复制份数。
TP_REPLICATED_EL = 4 * sum(_W_DEC_NOSHARD.values()) + _FN_G   # = 1600 (含 4×ln2_g)
TP_SHARDED_EL = TOTAL_EL - TP_REPLICATED_EL


def _dec_layer_el(tp, fsdp, ep, efsdp):
    """单 MoE decoder 层每卡持久 numel（手工分母：dense ÷tp÷fsdp、专家 ÷ep÷efsdp、无切 ÷fsdp）。"""
    return (sum(v // tp // fsdp for v in _W_DEC_DENSE.values())
            + sum(v // fsdp for v in _W_DEC_NOSHARD.values())
            + sum(v // ep // efsdp for v in _W_DEC_EXPERT.values()))


def _evaluate(pc, rec=None):
    spec = build_llm_spec(_MOE_CFG)
    return Evaluator(spec, pc, OptimizerSpec.adamw(),           # bf16 params：mult=14
                     HardwareSpec(max_device_memory=1 << 50, alloc_block_bytes=1),
                     rec or RecomputeSpec("None"), SwapSpec()).evaluate(record_timeline=True)


def test_hybrid_tp_ep_dp_pp_persistent_exact():
    """tp2×ep2×dp2×pp2（sp 开，m=2）：每 stage 持久字节逐位相等于手工推导。
    fsdp=dp_shard·cp=2、efsdp=dp_shard·cp·tp/ep=2；mult=14（bf16 AdamW）。"""
    pc = ParallelConfig(dp_shard=2, tp=2, ep=2, pp=2, sequence_parallel=True,
                        num_microbatches=2)
    rep = _evaluate(pc)
    dec = _dec_layer_el(tp=2, fsdp=2, ep=2, efsdp=2)             # = 15552 (含 ln2_g //fsdp=32)
    s0 = (_EMB // 2 // 2 + 2 * dec) * 14                          # emb + 2 decoder
    s1 = (2 * dec + _HEAD // 2 // 2 + _FN_G // 2) * 14            # 2 decoder + head
    assert rep.per_stage[0].breakdown.persistent == s0 == 492800
    assert rep.per_stage[1].breakdown.persistent == s1 == 493248


def test_hybrid_grad_accum_and_optstep_exact():
    """同配置：optstep 事件的 grad_accum = 全 stage 已规约梯度分片 Σ(numel_local//分母)·4B；
    optstep 瞬态 = K_OPT(4)·max_w·4，max_w = 每卡最大单权重分片（e_w1 → 32768/ep/efsdp = 8192）。
    验证 dense/expert 分母（fsdp vs efsdp）在 timeline 侧与持久态同口径。"""
    pc = ParallelConfig(dp_shard=2, tp=2, ep=2, pp=2, sequence_parallel=True,
                        num_microbatches=2)
    rep = _evaluate(pc)
    dec = _dec_layer_el(tp=2, fsdp=2, ep=2, efsdp=2)
    expect_g = {0: (_EMB // 2 // 2 + 2 * dec) * 4,
                1: (2 * dec + _HEAD // 2 // 2 + _FN_G // 2) * 4}
    for p in rep.per_stage:
        opt = [t for t in p.timeline if t.event == "optstep"]
        assert len(opt) == 1
        bd = opt[0].breakdown
        assert bd.grad_accum == expect_g[p.stage]
        assert bd.optstep == 4 * (_W_DEC_EXPERT["e_w1"] // 2 // 2) * 4    # 4·8192·4
        assert bd.act_live == 0                                           # step 时激活已释


# ---------------------------------------------------------------------------
# ⑥ 全集群参数守恒：9 组并行配置的不变量（缺口②的守恒面）
# ---------------------------------------------------------------------------

_SWEEP = [
    dict(),                                                                # 基线全 1
    dict(dp_shard=2),
    dict(tp=2, sequence_parallel=True),
    dict(pp=2, num_microbatches=2),
    dict(dp_shard=2, ep=2),
    dict(cp=2),
    dict(dp_replicate=2),
    dict(dp_shard=2, tp=2, ep=2, pp=2, sequence_parallel=True, num_microbatches=2),
    dict(dp_shard=2, cp=2, tp=2, ep=4, pp=3, sequence_parallel=True, num_microbatches=3),
    dict(pp=2, interleave=2, num_microbatches=4),                          # VPP
]


def test_baseline_total_params_closed_form():
    """基线（全 1）持久 = 全模型 numel × 14 —— 手工闭式 279872 el。"""
    rep = _evaluate(ParallelConfig())
    assert sum(p.breakdown.persistent for p in rep.per_stage) == TOTAL_EL * 14


@pytest.mark.parametrize("kw", _SWEEP, ids=lambda kw: "+".join(f"{k}{v}" for k, v in kw.items() if k != "sequence_parallel" and k != "num_microbatches") or "base")
def test_cluster_param_conservation_across_hybrid_configs(kw):
    """Σ_stage（每卡持久 el × 承载该 stage 的卡数）==
       dp_replicate × (tp 切分权重全量 + tp 复制权重全量 × tp)，
    对任意 (dp_replicate,dp_shard,cp,tp,ep,pp,vpp) 组合成立——多切一次/漏切一次都会破约。
    （norm gamma / router fp32 在 TP 组内复制、仅 ÷fsdp——首轮跑此用例即揪出该物理事实：
    误设「全权重都被 tp 切」时 tp=2 组合恰差 1344 el = 复制权重量。）"""
    pc = ParallelConfig(**kw)
    rep = _evaluate(pc)
    ranks_per_stage = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp
    cluster_el = sum((p.breakdown.persistent // 14) * ranks_per_stage
                     for p in rep.per_stage)
    assert cluster_el == pc.dp_replicate * (TP_SHARDED_EL + TP_REPLICATED_EL * pc.tp)
