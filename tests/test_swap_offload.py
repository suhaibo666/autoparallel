"""激活 swap / CPU-offload 建模（公式，depth 可配）——忠实 mindformers pynative / PyTorch。

源忠实（file:line）：
  - mindformers pynative 是本库对标对象，**有真的激活 offload 实现**：
    `mindformers/pynative/distributed/activation_checkpoint.py:782-815` `apply_swap`——
    **逐层** `swap_wrapper(model.layers[layer_id])`（:806），预取深度 = `default_prefetch`
    （`config.py:826` `SwapConfig.default_prefetch: int = 1`，:300-307 校验 ≥1 且 <num_layers）；
    预取对 `(layer_id, layer_id + prefetch)`（:807/:812）经 `SwapManager().set_forward_prefetch_layer`
    （:814-815）——处理"更后一层"(反向更早)时触发"更前一层"的 H2D swap-in，反向到该层时激活已驻留。
    policy_fn（:678-690）除注意力 mask（uint8 seq×seq）保留在设备外，其余 saved 张量 MUST_SWAP。
    `pp>1` 不支持 swap（:898-900）。
  - PyTorch `save_on_cpu`（`torch/autograd/graph.py:350-420`）：**逐张量**——前向 `pack_to_cpu`
    (:403-414) `tensor.cpu()` D2H 卸载后释放；反向 `unpack_from_cpu`(:416-418)
    `tensor.to(device)` H2D 取回。`offload_wrapper`（`checkpoint_wrapper.py:174-193`）逐模块封装。
    torch 无显式预取深度（unpack 处同步逐张量）；mindformers 显式 default_prefetch 双缓冲。

模型（本库 per-layer 粒度，§8.1 `swap_buf`="从 CPU 预取回的激活"）：
  - FWD：被 swap 层 saves 卸载到 CPU → 不驻留 act_live（`saved=0`，前向→反向间隙不占）。
  - BWD：被 swap 层反向前 H2D 取回 → 必须驻留。`swap_buf(bwd@L)` =（L 自身被 swap 则其 saves 复原）
    + Σ 反向执行序后 depth 层里被 swap 层的 saves（在飞预取窗）——与 FSDP2 参数预取
    `_prefetch_param_bytes` 同构的双缓冲。depth=swap.default_prefetch。
  - 净效应：swap 从 act_live 扣掉被 swap 层的 saves（整栈不驻留）、但加回 swap_buf 预取窗（仅当前
    microbatch）。swap 关（enable=False）→ swap_buf 恒 0、逐字节复现旧行为。
"""
from cost_eval.mem_timeline import MemTimeline
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.specs import ParallelConfig, RecomputeSpec, SwapSpec, OptimizerSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.static_mem import StaticMem
from cost_eval.structure_mem import estimate_structure_memory


# 齐质 4 层 dense toy（同 test_fsdp_prefetch）：每层 saves 相同 → 预取窗口值可精确预测。
D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=4)


def _sim(swap):
    spec = ModelSpec("toy", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(pp=1), n_layers=4, world_size=1)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(), pm, False)
    r = MemTimeline().simulate(g, RecomputeSpec("None"), swap, pm, persistent,
                               framework_reserve=0, max_device_memory=10 ** 12,
                               record_timeline=True)
    return r[0], g


def _saves(g, lid):
    layer = next(l for l in g.stages[0] if l.layer_id == lid)
    return estimate_structure_memory(layer.ops).activation_saves


def _bd(sp, tag):
    return next(s for s in sp.timeline if s.event == tag).breakdown


def _swap_buf(sp, tag):
    return _bd(sp, tag).swap_buf


def _act_live(sp, tag):
    return _bd(sp, tag).act_live


def _swap(layers, depth=1):
    return SwapSpec(enable=True, default_prefetch=depth, swap_layers=set(layers))


# ---------------------------------------------------------------------------
# 回归：swap 关（SwapSpec 默认 enable=False）→ swap_buf 恒 0（逐字节复现旧行为守卫）
# ---------------------------------------------------------------------------

def test_swap_disabled_swap_buf_zero_every_event():
    sp, _ = _sim(SwapSpec())
    assert all(s.breakdown.swap_buf == 0 for s in sp.timeline)


def test_enable_but_empty_swap_layers_is_noop():
    """enable=True 但 swap_layers 为空 → 无层被 swap → 与关闭逐事件相同（swap_buf 恒 0）。"""
    off, _ = _sim(SwapSpec())
    on, _ = _sim(SwapSpec(enable=True, swap_layers=set()))
    assert [s.total_bytes for s in on.timeline] == [s.total_bytes for s in off.timeline]
    assert all(s.breakdown.swap_buf == 0 for s in on.timeline)


# ---------------------------------------------------------------------------
# FWD：被 swap 层 saves 离开 act_live（前向→反向间隙不驻留）
# ---------------------------------------------------------------------------

def test_swap_removes_saved_from_act_live_forward():
    off, g = _sim(SwapSpec())
    on, _ = _sim(_swap({1, 2}))
    drop = _saves(g, 1) + _saves(g, 2)
    assert drop > 0
    assert _act_live(off, "fwd_end") - _act_live(on, "fwd_end") == drop


# ---------------------------------------------------------------------------
# BWD：swap_buf = 复原当前层 + 预取窗（双缓冲，depth=1）
# ---------------------------------------------------------------------------

def test_swap_buf_double_buffer_depth1():
    on, g = _sim(_swap({1, 2}, depth=1))
    # 反向序 [3,2,1,0]
    # bwd@3(idx0)：3 未 swap；预取 pos1=layer2(swap) → swap_buf = saves(2)
    assert _swap_buf(on, "bwd@3") == _saves(g, 2)
    # bwd@2(idx1)：2 复原 saves(2)；预取 pos2=layer1(swap) → + saves(1)
    assert _swap_buf(on, "bwd@2") == _saves(g, 2) + _saves(g, 1)
    # bwd@1(idx2)：1 复原 saves(1)；预取 pos3=layer0(未 swap) → +0
    assert _swap_buf(on, "bwd@1") == _saves(g, 1)
    # bwd@0(idx3)：0 未 swap；预取越界 → 0
    assert _swap_buf(on, "bwd@0") == 0


def test_swapped_layer_backward_activation_resident_not_zero():
    """被 swap 层自身反向：其激活经 swap_buf 复原、必须驻留（≥ 自身 saves），反向不欠算。"""
    on, g = _sim(_swap({2}, depth=1))
    # layer2 反向：自身 saves 复原（预取的下一层 layer1 未 swap → 恰等自身 saves）
    assert _swap_buf(on, "bwd@2") >= _saves(g, 2)
    assert _swap_buf(on, "bwd@2") == _saves(g, 2)
    # 其激活在前一反向单元 bwd@3 已被预取（双缓冲）
    assert _swap_buf(on, "bwd@3") == _saves(g, 2)


# ---------------------------------------------------------------------------
# 预取深度来自 SwapSpec.default_prefetch（更深 → 预取窗更宽）
# ---------------------------------------------------------------------------

def test_prefetch_depth_from_spec_widens_window():
    d1, g = _sim(_swap({0, 1, 2, 3}, depth=1))
    d2, _ = _sim(_swap({0, 1, 2, 3}, depth=2))
    # bwd@3(idx0)：depth1 → 复原 saves(3)+预取{layer2}；depth2 → +预取{layer2,layer1}
    assert _swap_buf(d1, "bwd@3") == _saves(g, 3) + _saves(g, 2)
    assert _swap_buf(d2, "bwd@3") == _saves(g, 3) + _saves(g, 2) + _saves(g, 1)
    assert _swap_buf(d2, "bwd@3") > _swap_buf(d1, "bwd@3")


# ---------------------------------------------------------------------------
# 净效应：swap 降峰值（offload 省显存），但非零成本（swap_buf 是代价）
# ---------------------------------------------------------------------------

def test_swap_lowers_peak_but_not_free():
    off, _ = _sim(SwapSpec())
    on, _ = _sim(_swap({1, 2}, depth=1))
    assert on.peak_bytes < off.peak_bytes                       # offload 省显存
    assert any(s.breakdown.swap_buf > 0 for s in on.timeline)   # 代价：预取窗非零


def test_monotonic_more_offload_lowers_or_floors_peak():
    """offload 更多层 → 峰值非增；至少从"不 swap"到"swap"严格下降（有 swap_buf 下限）。
    前向 act_live（fwd_end，无 swap_buf）随 offload 层数严格单调下降。"""
    p_none, _ = _sim(SwapSpec())
    p1, _ = _sim(_swap({1}, depth=1))
    p12, _ = _sim(_swap({1, 2}, depth=1))
    # 全局峰值：非增 + 相对不 swap 严格下降
    assert p12.peak_bytes <= p1.peak_bytes <= p_none.peak_bytes
    assert p12.peak_bytes < p_none.peak_bytes
    # 前向 act_live（无预取窗）严格单调：offload 越多、驻留越少
    a_none = _act_live(p_none, "fwd_end")
    a1 = _act_live(p1, "fwd_end")
    a12 = _act_live(p12, "fwd_end")
    assert a12 < a1 < a_none


# ---------------------------------------------------------------------------
# DSv3 4L 真结构 sanity（无重算，swap transformer 层）——机理/不变量（§8.7 无真机点）
# ---------------------------------------------------------------------------
from cost_eval.presets import deepseek_v3
from cost_eval.build_llm import build_llm_spec

GiB = 2 ** 30


def _dsv3_sim(N, swap):
    spec = build_llm_spec(deepseek_v3(N))
    pc = ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1, sequence_parallel=True)
    pm = ParallelModel(pc, spec.dims.n_layers, 2)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(
        g, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4), pm, False)
    r = MemTimeline().simulate(
        g, RecomputeSpec("None"), swap, pm, persistent,
        framework_reserve=0, max_device_memory=59 * GiB, grad_dtype_bytes=4,
        record_timeline=True)
    return r[0], g


def test_dsv3_swap_transformer_layers_invariants():
    # DSv3 4L 层：0=embedding,1..4=transformer,5=lm_head。swap transformer {1,2,3,4}。
    off, g = _dsv3_sim(4, SwapSpec())
    on, _ = _dsv3_sim(4, _swap({1, 2, 3, 4}, depth=1))

    # (a) 被 swap 层 saves 离开 act_live：bwd@5（首反向，尚无层释放）act_live 下降 = Σsaves(1..4)
    drop = sum(_saves(g, i) for i in range(1, 5))
    assert drop > 0
    assert _act_live(off, "bwd@5") - _act_live(on, "bwd@5") == drop

    # (b) swap_buf 非零 = 预取窗
    assert any(s.breakdown.swap_buf > 0 for s in on.timeline)

    # (c) 被 swap 层反向激活驻留（不欠算）：bwd@4 = 自身 saves(4) + 预取 layer3
    assert _swap_buf(on, "bwd@4") == _saves(g, 4) + _saves(g, 3)
    assert _swap_buf(on, "bwd@4") >= _saves(g, 4)

    # (d) 峰值更低（offload 省显存）
    assert on.peak_bytes < off.peak_bytes

    # (e) swap 关：swap_buf 恒 0（byte-identical 守卫）
    assert all(s.breakdown.swap_buf == 0 for s in off.timeline)
