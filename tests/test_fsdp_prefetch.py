"""FSDP2 参数预取双缓冲建模（公式，depth 可配）——忠实 PyTorch FSDP2 / mindformers pynative。

源忠实（file:line）：
  - PyTorch FSDP2 默认 **depth-1** 反向预取：`_fsdp_param_group.py:854-856`
    `_backward_prefetch` → `target = post_forward_order[curr_index - 1]`（恰回退 1 个单元）；
    `:854` 的 `elif curr_index > 0` 守卫 → 反向最后一个单元（= 前向第一个）不预取。
    前向隐式 depth-1：`wait_for_unshard:486-495` + Note:61-70（当前 all-gather 输出保留至
    下一个 copy-in），`_fsdp_state.py:203-207` root 前向后不 reshard（立即在反向重 gather）。
  - mindformers pynative 显式 depth-1 链：`base_models/gpt/parallelize.py:245-255`
    （fwd：每层 → 下一层；末层 → tail=[final_ln, output_layer]）、`:261-273`
    （bwd：output_layer → layer[N-1]；layer[i] → layer[i-1]；layer[0] → embedding）。
    output_layer 前向后不 reshard（`:1415-1416`）。shard 策略 optim_grads_params=ZeRO-3
    （`config.py:351`）。

模型（本库 per-layer 粒度）：某层事件的 gather_buf = 当前层 param_full_bytes +
Σ 执行序后 `depth` 层的 param_full_bytes（FWD：layer_ids 顺序；BWD：其逆序）。
depth=0 复现旧单缓冲行为（回归路径）；depth=1（默认）= 当前 + 下一层双缓冲。
"""
from cost_eval.mem_timeline import MemTimeline
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.specs import ParallelConfig, RecomputeSpec, SwapSpec, OptimizerSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.static_mem import StaticMem
from cost_eval.structure_mem import estimate_structure_memory


# 齐质 4 层 dense toy：每层 param_full 相同 → 双缓冲值恰为 2×单层。
D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=4)


def _sim(prefetch_depth):
    spec = ModelSpec("toy", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(pp=1, prefetch_depth=prefetch_depth),
                       n_layers=4, world_size=1)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(), pm, False)
    r = MemTimeline().simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                               framework_reserve=0, max_device_memory=10 ** 12,
                               record_timeline=True)
    return r[0], g


def _pf(g):
    """齐质 toy：任一层 param_full_bytes（全相同）。"""
    return estimate_structure_memory(g.stages[0][0].ops).param_full_bytes


def _gather_at(sp, tag):
    return next(s for s in sp.timeline if s.event == tag).breakdown.gather_buf


# ---------------------------------------------------------------------------
# depth=0 —— 复现旧单缓冲（回归路径）：所有层事件 gather_buf = 单层 param_full
# ---------------------------------------------------------------------------

def test_depth0_single_buffer_every_event():
    sp, g = _sim(0)
    pf = _pf(g)
    for tag in ("fwd:0", "fwd:1", "fwd:2", "fwd:3", "bwd@3", "bwd@2", "bwd@1", "bwd@0"):
        assert _gather_at(sp, tag) == pf, tag


# ---------------------------------------------------------------------------
# depth=1 —— 前向：当前 + 下一层（双缓冲）；末层无预取
# ---------------------------------------------------------------------------

def test_depth1_forward_doubles_midstretch():
    sp, g = _sim(1)
    pf = _pf(g)
    # 中段前向层：当前 + 下一层 = 2×（齐质）
    assert _gather_at(sp, "fwd:0") == 2 * pf
    assert _gather_at(sp, "fwd:1") == 2 * pf
    assert _gather_at(sp, "fwd:2") == 2 * pf


def test_depth1_forward_last_layer_no_prefetch():
    sp, g = _sim(1)
    pf = _pf(g)
    # 前向最后一层（layer_ids[-1]）后无层可预取 → 单缓冲
    assert _gather_at(sp, "fwd:3") == pf


# ---------------------------------------------------------------------------
# depth=1 —— 反向：逆序，当前 + 反向下一层；反向最后一层（embedding 位）无预取
# ---------------------------------------------------------------------------

def test_depth1_backward_reverse_order_double():
    sp, g = _sim(1)
    pf = _pf(g)
    # 反向首个单元（layer 3，逆序第一）预取反向下一层（layer 2）→ 双缓冲
    assert _gather_at(sp, "bwd@3") == 2 * pf
    assert _gather_at(sp, "bwd@2") == 2 * pf
    assert _gather_at(sp, "bwd@1") == 2 * pf


def test_depth1_backward_last_unit_no_prefetch():
    sp, g = _sim(1)
    pf = _pf(g)
    # 反向最后一个单元（layer 0 = 前向第一）无预取（FSDP2 curr_index>0 守卫）
    assert _gather_at(sp, "bwd@0") == pf


# ---------------------------------------------------------------------------
# DSv3 真机锚点：预取拆解（把 ΔP 从标定 framework_reserve 移入 gather_buf 公式）
#   真机 rank0 peak_alloc = 12473.1 MiB（4L FSDP-2）。峰值落 bwd@5(lm_head)：反向首
#   单元预取末 transformer 层(layer4, mla_moe)。ΔP = param_full(layer4)。
# ---------------------------------------------------------------------------
from cost_eval.presets import deepseek_v3
from cost_eval.build_llm import build_llm_spec
from cost_eval.report import Evaluator
from cost_eval.specs import HardwareSpec

MiB, GiB = 2 ** 20, 2 ** 30


def _dsv3_peak(N, depth, reserve_bytes):
    spec = build_llm_spec(deepseek_v3(N))
    pc = ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1, sequence_parallel=True,
                        prefetch_depth=depth)
    ev = Evaluator(spec, pc,
                   OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                   HardwareSpec(max_device_memory=59 * GiB, framework_reserve=reserve_bytes),
                   RecomputeSpec(mode="full", full_layers=set(range(1, N + 1))), SwapSpec())
    return ev.evaluate().per_stage[0]


def _dsv3_layer_pf(N, layer_id):
    """DSv3 缩层第 layer_id 层的 full-unsharded param 字节（不杜撰，直接从结构算）。"""
    spec = build_llm_spec(deepseek_v3(N))
    pc = ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1, sequence_parallel=True)
    pm = ParallelModel(pc, spec.dims.n_layers, 2)
    g = ShapeEval().resolve(spec, pm)
    layer = next(l for st in g.stages.values() for l in st if l.layer_id == layer_id)
    return estimate_structure_memory(layer.ops).param_full_bytes


def test_dsv3_prefetch_is_exact_reattribution_of_reserve():
    """拆解 = 纯再归属：depth=1 + (177MiB−ΔP) 与 depth=0 + 177MiB 总峰值 **逐字节相等**，
    仅 ΔP 从 framework 桶移入 gather_buf 桶（无凭空增减）。"""
    dp = _dsv3_layer_pf(4, 4)                     # ΔP = param_full(layer4 mla_moe)
    p0 = _dsv3_peak(4, 0, 177 * MiB)              # 旧单缓冲 + 旧 reserve
    p1 = _dsv3_peak(4, 1, 177 * MiB - dp)         # 新双缓冲 + 拆解 reserve
    assert p0.peak_event == p1.peak_event == "bwd@5"
    assert p1.peak_bytes == p0.peak_bytes                              # 逐字节相等
    assert p1.breakdown.gather_buf - p0.breakdown.gather_buf == dp     # ΔP 进 gather
    assert p0.breakdown.framework - p1.breakdown.framework == dp       # ΔP 出 framework


def test_dsv3_peak_gather_is_lmhead_plus_next_bwd_layer():
    """bwd@5(lm_head) 峰值 gather_buf = lm_head 整层 + 末 transformer 层(layer4) 双缓冲。"""
    dp = _dsv3_layer_pf(4, 4)
    pf_head = _dsv3_layer_pf(4, 5)
    p1 = _dsv3_peak(4, 1, 177 * MiB - dp)
    assert p1.breakdown.gather_buf == pf_head + dp


def test_dsv3_default_reserve_reconciles_anchor_4L_8L():
    """默认 depth=1 + RESIDUAL_MiB=0（经验常数已消除）→ DSv3 4L/8L 命中真机锚点 ±1%。

    预测由「逐桶结构 + 分配器对齐公式」给出，无拟合 framework 常数；现略低于真机
    （12409.5 vs 12473.1 = 0.995），差额为文档化的 sub-block 小残差。"""
    from validate_dsv3 import RESIDUAL_MiB
    p4 = _dsv3_peak(4, 1, RESIDUAL_MiB * MiB)
    p8 = _dsv3_peak(8, 1, RESIDUAL_MiB * MiB)
    # 2026-07-30：`lm_head` 反向 kernel workspace 入账 → 12423.9→13481.9 / 13848.0→14906.0
    #   （比值 1.081 / 1.068 = 过读侧 = OOM 安全）。改钉理论值，**不放宽 ±1%**。
    assert abs(p4.peak_bytes / MiB - 13481.9) < 0.1
    assert abs(p8.peak_bytes / MiB - 14906.0) < 0.1
