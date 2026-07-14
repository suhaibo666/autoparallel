"""D-4：VPP（交错式 1F1B, v>1）激活按 **chunk** 忠实累加 —— 去 ~V× 过估。

忠实模型（逐行锚定 Megatron `pipeline_parallel/schedules.py`）：每物理 device 持 V 个 model chunk，
各 L/V 层；micro 数取 VPP 理论 warmup 公式（`:877-878`，clamp 到 m·V `:889-890`），(mb,chunk) 下发
顺序 port 自 `get_schedule_table`（`:902-929`）+ `convert_schedule_table_to_order`（`:932-955`）。
每虚拟 FWD 步只驻留一个 chunk → 峰 act_live = Σ_chunk n_c·(L/V)，Σ_c n_c = warmup+1。对照旧**物理**
粒度（每在飞微批 pin 整 stage L 层）= ~V× 过估。设计见 `specs/2026-07-06-audit-remediation.md` §D-4。
"""
from cost_eval.mem_timeline import (
    MemTimeline, build_1f1b, build_interleaved_1f1b,
    get_schedule_table, interleaved_virtual_order, chunk_layer_ids, _layer_saves_bytes)
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.specs import ParallelConfig, RecomputeSpec, SwapSpec, OptimizerSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.static_mem import StaticMem

# 纯 dense 栈（无 embedding/head 伪层）→ 各层 saves 均匀 = s，便于手算 per-chunk 和。
D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=4)


def _sim(pp, m, v, n_layers=4, stage=0):
    """toy dense 栈的 stage 峰值仿真（无重算/无 swap，record_timeline 以便取 max act_live）。"""
    spec = ModelSpec("toy", D, ["dense"] * n_layers, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(pp=pp, num_microbatches=m, interleave=v),
                       n_layers=n_layers, world_size=pp, edge_pseudo=(0, 0))  # toy 栈无 embedding/head 伪层
    g = ShapeEval().resolve(spec, pm)
    per = StaticMem().compute(g, OptimizerSpec.adamw(), pm, False)
    return MemTimeline().simulate(g, RecomputeSpec("None"), SwapSpec(), pm, per,
                                  framework_reserve=0, max_device_memory=10 ** 12,
                                  record_timeline=True)[stage]


def _max_act(sp):
    """时间线上 act_live 的最大值（= 真实激活峰，独立于峰值落在 fwd/bwd 哪个事件）。"""
    return max(t.breakdown.act_live for t in sp.timeline)


def _dense_saves():
    """单个 dense 层去重后 saves 字节 s（各 dense 层同形 → 均匀）。"""
    spec = ModelSpec("toy", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(pp=2, num_microbatches=8), n_layers=4, world_size=2)
    g = ShapeEval().resolve(spec, pm)
    return _layer_saves_bytes(g.stages[0][0])


# ---------------------------------------------------------------------------
# Megatron 调度 port 校验（get_schedule_table + convert_schedule_table_to_order）
# ---------------------------------------------------------------------------

def test_schedule_table_matches_megatron_port():
    """`get_schedule_table` 命中 Megatron 手推表（pp=2,v=2,m=4,gs=pp=2；schedules.py:902-929）。"""
    assert get_schedule_table(4, 2, 2) == [
        (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1)]


def test_virtual_order_warmup_and_peak_distribution():
    """`interleaved_virtual_order`（port convert_schedule_table_to_order，schedules.py:932-955）：
    pp=2,v=2,m=4,stage0 → warmup=(2-0-1)*2+(2-1)*2=4；总 2*m*v=16 步，FWD/BWD 各 8；峰时在飞
    5 个虚拟步分布 chunk0:3 / chunk1:2（= 前 5 个前向的 chunk）。"""
    order = interleaved_virtual_order(0, 2, 4, 2, 2)
    assert len(order) == 16
    assert sum(k == "FWD" for k, _, _ in order) == 8
    assert sum(k == "BWD" for k, _, _ in order) == 8
    fwd_chunks = [c for k, _, c in order if k == "FWD"][:5]      # warmup(4)+steady 首个 FWD
    assert fwd_chunks.count(0) == 3 and fwd_chunks.count(1) == 2


# ---------------------------------------------------------------------------
# (a) v=1 == plain-1F1B（byte-identical）
# ---------------------------------------------------------------------------

def test_v1_is_plain_1f1b_byte_identical():
    """(a) v=1：simulate 走 build_interleaved_1f1b→build_1f1b、每步 pin 整 layer_ids → 与 plain
    逐字节一致（chunk 化只改 v>1 路径）。① 调度序列逐事件等于 plain build_1f1b；② max act_live 命中
    plain 手算基线：pp=2,stage0,L=2 → warmup=1、峰 2 微批 × 2 层 = 4·s。"""
    assert build_interleaved_1f1b(0, 2, 8, 1) == build_1f1b(0, 2, 8)
    s = _dense_saves()
    assert _max_act(_sim(pp=2, m=8, v=1)) == 4 * s


# ---------------------------------------------------------------------------
# (b) v=2 峰值 < 2×plain 且 == 手算 per-chunk 和（过估消去）
# ---------------------------------------------------------------------------

def test_v2_peak_below_2x_and_equals_per_chunk_sum():
    """(b) pp=2,stage0,L=2,v=2（每 chunk=1 层）：warmup=(2-0-1)*2+(2-1)*2=4 → 峰 warmup+1=5 个
    在飞虚拟步，每步一个 chunk=1 层 → **max act_live = 5·s**（手算，s 取自模型；分布 chunk0=3/chunk1=2）。

    - 方向：5·s > plain 4·s（交错确实更吃激活）。
    - 幅度：5·s < 2×plain=8·s（~V× 过估已消）。旧**物理**粒度会给 (4+1)微批×2层 = **10·s**（=2.5×plain，
      = 忠实值的 V=2 倍）→ 本断言即证过估消去。
    """
    s = _dense_saves()
    v1, v2 = _sim(pp=2, m=8, v=1), _sim(pp=2, m=8, v=2)
    assert _max_act(v2) == 5 * s                        # 精确 per-chunk 和（手算 5）
    assert _max_act(v2) > _max_act(v1)                  # 5·s > 4·s
    assert _max_act(v2) < 2 * _max_act(v1)              # 5·s < 8·s（< V×）
    assert v2.peak_bytes < 2 * v1.peak_bytes            # peak_bytes 头条同向（persistent 共有）


def test_v2_equals_old_physical_over_count_divided_by_V():
    """层均匀时忠实峰 == 旧物理过估 ÷V —— 定量钉死「V× 过估」的幅度。旧物理：build_interleaved_1f1b
    warmup=min(4,m=8)=4、leading fwd=5 个整-stage 微批 × L=2 层 = 10·s；忠实 = 5·s = 10·s / V(=2)。"""
    s = _dense_saves()
    old_physical = 5 * 2 * s                            # 5 微批 × L=2 层（旧「整 stage」口径）
    assert _max_act(_sim(pp=2, m=8, v=2)) == old_physical // 2


# ---------------------------------------------------------------------------
# (c) 常驻层数不超过「实际 device 层数 × 在飞倍数」，绝不 V·L
# ---------------------------------------------------------------------------

def test_chunk_partition_sums_to_L_never_vL():
    """(c) 切分不变式：Σ chunk 层数 == L（**实际** device 层数，非 V·L），互斥无遗漏、均衡。
    → 任一微批走完全 V chunk 最多 pin L 层。覆盖整除 / 非整除 / L<v（尾部空 chunk graceful）。"""
    for L, v in [(4, 2), (4, 4), (6, 3), (5, 2), (7, 4), (1, 2)]:
        chunks = chunk_layer_ids(list(range(L)), v)
        assert len(chunks) == v
        assert sum(len(c) for c in chunks) == L                     # Σ = L，不是 v·L
        flat = [x for c in chunks for x in c]
        assert flat == list(range(L))                               # 互斥、无遗漏、有序
        assert max(len(c) for c in chunks) - min(len(c) for c in chunks) <= 1   # 均衡


def test_resident_bounded_by_actual_layers_times_multiplicity():
    """(c) 时间线上任一时刻 act_live ≤ (warmup+1)·(单 chunk 层数)·s，且**严格 <** (warmup+1)·L·s
    （= 旧「每步整 stage L 层」口径）→ 证明按 chunk(L/V) 而非整 stage 累加。pp=2,L=2,v=2：warmup=4，
    单 chunk=1 层 → 上界 5·s（实测达界，非超界）；旧口径 5·2·s=10·s。"""
    s = _dense_saves()
    v2 = _sim(pp=2, m=8, v=2)
    warmup = 4                                          # (2-0-1)*2 + (2-1)*2
    per_chunk_layers = 1                                # L=2 // v=2
    for t in v2.timeline:
        assert t.breakdown.act_live <= (warmup + 1) * per_chunk_layers * s
    assert _max_act(v2) < (warmup + 1) * 2 * s          # 严格小于「整 stage L=2 层」旧口径


# ---------------------------------------------------------------------------
# 非单调（bulge at V=2）—— 修正后的真实物理签名
# ---------------------------------------------------------------------------

def test_peak_bulges_at_v2_then_falls():
    """膨胀比 ≈ 1+(pp-1)/(pp·V)，**V=2 处最大、随 V 回落**（旧 simulate 单调增是 ~V× 过估副产物）。
    pp=2,L=4,stage0：max act_live（层单位）v1=(1+1)·4=8 → v2=(4+1)·2=10 → v4=(8+1)·1=9 → v2>v4>v1。"""
    s = _dense_saves()
    a = {v: _max_act(_sim(pp=2, m=20, v=v, n_layers=8)) for v in (1, 2, 4)}
    assert a[1] == 8 * s and a[2] == 10 * s and a[4] == 9 * s     # 手算三点
    assert a[2] > a[4] > a[1]                                     # bulge：非单调
    assert a[2] < 2 * a[1] and a[4] < 4 * a[1]                    # 每个 v>1 均 < V×plain


def test_vpp_round_robin_chunk_placement_real_pattern():
    """VPP round-robin 放置锁定（2026-07-14,源:mindformers pynative pipeline_parallel.py:258
    `chunk_id*pp+rank`）:真实 pattern(含 embedding/head 伪层)下 rank 持**非连续**层段。
    DSv3 8L pp2 v2:8 中间层切 4 虚拟段(各 2 层),rank0 持 sv0{1,2}+sv2{5,6},rank1 持 sv1{3,4}+sv3{7,8};
    embedding→stage0 chunk0、head→stage1 末 chunk。"""
    from validate_dsv3 import build_dsv3_spec
    spec, d, _ = build_dsv3_spec(8)
    pm = ParallelModel(ParallelConfig(pp=2, num_microbatches=2, interleave=2),
                       n_layers=d.n_layers, world_size=2)
    c0, c1 = pm.stage_chunks(0), pm.stage_chunks(1)
    assert c0 == [[0, 1, 2], [5, 6]], c0          # emb+sv0 | sv2（非连续!）
    assert c1 == [[3, 4], [7, 8, 9]], c1          # sv1 | sv3+head
    # 全层恰好覆盖一次
    all_l = sorted(sum(c0 + c1, []))
    assert all_l == list(range(10))
