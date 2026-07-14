"""B 标定 margin（2026-07-09）：select 重算下**保留-MoE**层 loss 峰的 fp32-cast + 小张量长尾
（源码级 op-DAG 提取证实其在 op 图粒度之下，`analysis/realmachine/opdag_validation.md`）→ 明示为标定常数
`kept_frag_factor`（DSv3 preset=1.9，自真机 select_attn 标定）。仅 select-kept-MoE 生效；full /
no-recompute / select-keep-attn 不触发（锚点不破）。真机：select_attn(keepFFN)=18828。
"""
from validate_dsv3 import build_dsv3_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

GiB = 2 ** 30
MiB = 2 ** 20
ATTN = {lid: {"linear_q", "linear_kv", "q_a", "kv_a", "rope", "flash", "o_proj"} for lid in range(1, 9)}
MLP = {lid: {"fc", "swiglu", "gelu", "router", "dispatch", "e_", "combine", "shared"} for lid in range(1, 9)}


def _peak(rc, *, N=8, factor=None):
    spec, d, fl = build_dsv3_spec(N)   # preset 已带 kept_frag_factor=1.9
    d.B = 1
    if factor is not None:
        d.kept_frag_factor = factor
    pc = ParallelConfig(dp_shard=2, cp=1, tp=1, pp=1, sequence_parallel=True, num_microbatches=1)
    ev = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                   HardwareSpec(max_device_memory=64 * GiB, framework_reserve=0), rc, SwapSpec())
    return ev.evaluate().per_stage[0].peak_bytes / MiB


def test_select_attn_keepFFN_margin_closes_residual():
    # 靶心：真机 18828。无 margin 15488（0.823 欠预测）；1.9 标定 margin → ~18844（1.001，OOM-安全）。
    with_m = _peak(RecomputeSpec("select", select_ops=ATTN))
    assert 18000 <= with_m <= 19500, with_m          # ≥0.95 且 OOM-安全（≥真机）
    assert with_m / 18828 >= 0.95


def test_margin_off_reproduces_pre_fix_underprediction():
    # factor=0（关）→ 复现修前 15488（证明 margin 是唯一变量、可关）。
    off = _peak(RecomputeSpec("select", select_ops=ATTN), factor=0.0)
    assert abs(off - 15516.4) < 1.0, off


def test_select_mlp_keepattn_unchanged_moe_recomputed():
    # keep-attn（重算 FFN → MoE 被重算）：margin gate 关 → 与 factor=0 逐字节相同（不被推过头）。
    # P1-09（2026-07-14）：14821.6 → 14865.9——attn 保留层的 fa_stats（softmax max/sum）
    # 驻留至反向（+15.5 MiB，8 层），ratio 0.940→0.941（真机 15765，欠预测方向收窄）。
    on = _peak(RecomputeSpec("select", select_ops=MLP))
    off = _peak(RecomputeSpec("select", select_ops=MLP), factor=0.0)
    assert abs(on - off) < 1e-6, (on, off)
    assert abs(off - 14865.9) < 1.0, off


def test_full_recompute_hard_gate_unbroken():
    # full 重算：kept-MoE=0 → margin 0 → DSv3 4L 硬门 12437.9 逐字节不破。
    p = _peak(RecomputeSpec("full", full_layers={1, 2, 3, 4}), N=4)
    assert abs(p - 12437.9) < 0.05, p


def test_per_stage_none_loss_stage_keeps_kce_fat():
    """review P0.1（2026-07-14）:K_CE fat 判据按 stage——per-stage select 下未重算的 loss stage
    须与全局 None 等值(修前被全局 mode=='select' 误关 k_ce,低估 71%:43899→12569)。"""
    from serve_explorer import eval_config
    mixed = eval_config({"layers": "8", "dp": "1", "pp": "2", "batch": "2",
                         "sel_stage": "s0:both; s1:none"})
    glob_none = eval_config({"layers": "8", "dp": "1", "pp": "2", "batch": "2"})
    assert mixed["ok"] and glob_none["ok"]
    # stage1(无重算+loss)与全局 None 的 stage1 等值(fat 生效)
    assert abs(mixed["stages"][1]["peak"] - glob_none["stages"][1]["peak"]) < 1.0
    # stage0(both 重算)低于全局 None 的 stage0,且峰回落到 optstep 锚点(重算后逐层反向不再超它)。
    # P0-01(2026-07-14):optstep 锚 10311→10213——K_OPT 6→4 重标(旧 6 含 ≈1.9 份累计梯度)
    # + grad_accum 桶显式化,二者净效应 −98 MiB;对照真机 10246.2 → 0.997(修前 1.006)。
    assert mixed["stages"][0]["peak"] < glob_none["stages"][0]["peak"]
    assert abs(mixed["stages"][0]["peak"] - 10213.0) < 50


def test_select_module_ops_single_source_and_qkv():
    """review P1.5（2026-07-14）:serve 与转换器 select 选择器**单一来源**,且 self_attention 集
    含 "qkv"(GQA/MHA 融合投影 op 名——修前转换器缺失,GQA yaml select 静默漏选)。"""
    import serve_explorer
    from cost_eval.configs.from_mindformers import _SELECT_MODULE_OPS, _build_recompute
    assert serve_explorer._SEL_ATTN == set(_SELECT_MODULE_OPS["self_attention"])
    assert serve_explorer._SEL_MLP == set(_SELECT_MODULE_OPS["mlp"])
    assert "qkv" in _SELECT_MODULE_OPS["self_attention"]
    rc = _build_recompute({"recompute": {"mode": "select",
                                         "select_module": {"self_attention": ["0-3"]}}})
    assert rc.op_matches(1, "qkv", "matmul")     # GQA 融合投影被选中
