"""Task 11 + 12 tests for M6 mem_timeline (build_1f1b + simulate)."""
from cost_eval.mem_timeline import build_1f1b, Event, MemTimeline, _layer_saves_bytes
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.layers.mla import build_mla_dense_decoder
from cost_eval.specs import ParallelConfig, RecomputeSpec, SwapSpec, OptimizerSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.static_mem import StaticMem
from cost_eval.structure_mem import estimate_structure_memory


# ---------------------------------------------------------------------------
# Task 11: 1F1B 调度构建
# ---------------------------------------------------------------------------

def test_1f1b_warmup_depth():
    # PP=4, stage 0: warmup = PP-1-0 = 3 个前向先行
    evs = build_1f1b(stage=0, pp=4, m=8)
    fwd_prefix = []
    for e in evs:
        if e.kind == "FWD":
            fwd_prefix.append(e)
        else:
            break
    assert len(fwd_prefix) == 4    # warmup(3) + steady 第一个 F = 4 个 F 才出现 B


def test_event_counts_balanced():
    evs = build_1f1b(stage=1, pp=4, m=8)
    assert sum(e.kind == "FWD" for e in evs) == 8
    assert sum(e.kind == "BWD" for e in evs) == 8


# ---------------------------------------------------------------------------
# Task 12: simulate — 峰值落点迁移 + OOM 标志
# ---------------------------------------------------------------------------

D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=4)


def _setup(pp=1):
    spec = ModelSpec("toy", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(pp=pp), n_layers=4, world_size=pp)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(), pm, False)
    return g, pm, persistent


def test_full_recompute_lowers_peak_and_moves_event():
    g, pm, persistent = _setup(pp=1)
    mt = MemTimeline()
    none = mt.simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                       framework_reserve=0, max_device_memory=10**12)
    full = mt.simulate(g, RecomputeSpec("full", {0, 1, 2, 3}), SwapSpec(), pm, persistent,
                       framework_reserve=0, max_device_memory=10**12)
    # full 重算降低峰值（act_live 大降 > 反向重物化一层），峰值落在反向（FSDP gather+grad 共存）
    assert full[0].peak_bytes < none[0].peak_bytes
    assert none[0].peak_event.startswith("bwd")
    assert full[0].peak_event.startswith("bwd")


def test_oom_flag():
    g, pm, persistent = _setup(pp=1)
    r = MemTimeline().simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                               framework_reserve=0, max_device_memory=1)
    assert r[0].oom is True


def _peak_layer_sm(g, peak_event):
    lid = int(peak_event.split("@")[1])
    ops = next(l for l in g.stages[0] if l.layer_id == lid).ops
    return estimate_structure_memory(ops)


def test_no_recompute_adds_bwd_working_set():
    """§8.5②：无重算层反向叠加 `bwd_working_set`（= 该层 forward 峰值工作集的反向镜像，
    激活梯度 dL/dact），扣掉已显式建模的 `bwd_scratch` 部分。transformer 为主的配置（toy 全
    dense、无 loss 层）→ 峰值落在某 transformer 反向，bwd_working_set > 0 且入峰。"""
    g, pm, persistent = _setup(pp=1)
    r = MemTimeline().simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                               framework_reserve=0, max_device_memory=10**12)
    b = r[0].breakdown
    assert r[0].peak_event.startswith("bwd")
    assert b.bwd_working_set > 0                       # 无重算反向工作集入峰
    sm = _peak_layer_sm(g, r[0].peak_event)
    assert b.bwd_working_set == max(0, sm.forward_max_live - sm.bwd_scratch)
    # 逐桶之和 == 峰值（新桶已并入 total，无遗漏/重复）
    assert (b.persistent + b.act_live + b.gather_buf + b.grad_buf + b.recomp_scratch
            + b.bwd_scratch + b.bwd_working_set + b.swap_buf + b.workspace
            + b.framework) == r[0].peak_bytes


def test_full_recompute_scratch_is_forward_max_live():
    """§8.5②①②：重算层反向重物化 = 重跑 forward 的 **max-live**（非 saves 之和）。
    `recomp_scratch = max(0, forward_max_live − checkpoint_input)`（层入口已 pin 进 act_live）。"""
    g, pm, persistent = _setup(pp=1)
    full = MemTimeline().simulate(g, RecomputeSpec("full", {0, 1, 2, 3}), SwapSpec(), pm,
                                  persistent, framework_reserve=0, max_device_memory=10**12)
    b = full[0].breakdown
    assert full[0].peak_event.startswith("bwd")
    assert b.recomp_scratch > 0                        # 峰值落在重算 transformer 层反向
    sm = _peak_layer_sm(g, full[0].peak_event)
    assert b.recomp_scratch == max(0, sm.forward_max_live - sm.checkpoint_input)
    # forward_max_live ≠ activation_saves（守「取峰值工作集而非 saves 之和」的实质改动）
    assert sm.forward_max_live != sm.activation_saves


# ---------------------------------------------------------------------------
# Task 1 [C1]: 激活 saves 按张量名去重（attn 被 flash + o_proj 双 save）
# ---------------------------------------------------------------------------

def _resolve_single_layer(decoder):
    """把一个 LayerSpec 解析成 ResolvedLayer（单卡、单层）。"""
    spec = ModelSpec("toy", D, ["x"], {"x": decoder})
    pm = ParallelModel(ParallelConfig(), n_layers=1, world_size=1)
    return ShapeEval().resolve(spec, pm).stages[0][0]


def _naive_saves_bytes(layer):
    """去重前的裸求和（旧 _layer_saves_bytes 行为，用作对照）。"""
    return sum(s.local_numel * s.dtype_bytes for op in layer.ops for s in op.saves)


def test_gqa_layer_dedups_attn_save_once():
    """GQA：attn 被 flash(saves=[qkv,attn,lse]) 与 o_proj(saves=[attn]) 双 save，
    去重后应恰好比裸求和少一个 attn 张量的字节。"""
    layer = _resolve_single_layer(build_dense_decoder(D))
    naive = _naive_saves_bytes(layer)
    dedup = _layer_saves_bytes(layer)
    attn = next(s for op in layer.ops for s in op.saves if s.name == "attn")
    attn_bytes = attn.local_numel * attn.dtype_bytes
    assert attn_bytes > 0
    assert dedup == naive - attn_bytes


def test_mla_layer_dedups_attn_save_once():
    """MLA：attn 被 flash 与 o_proj 双 save（attention.py:149/153），去重后少一个 attn。"""
    Dmla = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100,
                    n_layers=1, q_lora_rank=32, kv_lora_rank=16,
                    qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16)
    spec = ModelSpec("toy", Dmla, ["x"], {"x": build_mla_dense_decoder(Dmla)})
    pm = ParallelModel(ParallelConfig(), n_layers=1, world_size=1)
    layer = ShapeEval().resolve(spec, pm).stages[0][0]
    naive = _naive_saves_bytes(layer)
    dedup = _layer_saves_bytes(layer)
    attn = next(s for op in layer.ops for s in op.saves if s.name == "attn")
    attn_bytes = attn.local_numel * attn.dtype_bytes
    assert attn_bytes > 0
    assert dedup == naive - attn_bytes
