"""**graph source 插座**（`cost_eval/liveness/sources.py`）的缝测试（2026-07-25）。

设计背景：显存今天算自**手写** op 名册（`cost_eval/layers/*.py` 的 `saves=[...]`），要换成从真
MindFormers 源抽出的图（`cost_eval/opdag/`）。仿真器（`liveness/simulate.py`）本该与来源无关。
本文件证明三件事：

  1. **逐字节中立**：加了 `graph_source` 参数后，默认路径与显式 `"hand_spec"` 的结果**逐字段
     相同**（含 timeline 与峰值 live-set），即既有 liveness 数值一字节没动。
  2. **缝是真的**：换一个来源，结果**确实**跟着变（不是参数被忽略）；未知来源 fail-loud，
     返回值缺 `.stages` fail-loud，重复注册 fail-loud。
  3. **来源无关**：一个**完全外部**的生产者（自己的 dataclass，只满足契约、不是
     `shape_eval.ResolvedLayer`）喂进去，仿真结果与手写路径**完全一致** —— 这正是
     `opdag/to_resolved.py` 落地后要走的路。
"""
from __future__ import annotations

import pytest

from cost_eval.layers.dense import build_dense_decoder
from cost_eval.liveness import (available_graph_sources, build_stage_graphs,
                                graph_source, has_graph_source,
                                register_graph_source, resolve_graph,
                                simulate_liveness, unregister_graph_source)
from cost_eval.liveness.sources import EXTRACTED, EXTRACTED_ENTRY_POINT, HAND_SPEC
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.specs import (HardwareSpec, OptimizerSpec, ParallelConfig,
                             RecomputeSpec, SwapSpec)

_HW = HardwareSpec(max_device_memory=64 * 2 ** 30)


def _toy_spec(n_layers: int = 2) -> ModelSpec:
    d = DimTable(H=256, F=512, n_heads=8, n_kv=8, head_dim=32,
                 S=512, B=1, vocab=1024, n_layers=n_layers)
    return ModelSpec("toy", d, ["dense"] * n_layers, {"dense": build_dense_decoder(d)})


def _run(spec, *, source=None, m=3, pp=2, recompute=None):
    pc = ParallelConfig(pp=pp, num_microbatches=m)
    kw = {} if source is None else {"graph_source": source}
    return simulate_liveness(
        spec, pc, OptimizerSpec.adamw(), _HW,
        recompute or RecomputeSpec("full", full_layers={0, 1}), SwapSpec(),
        record_timeline=True, grad_mode="chain2", **kw)


# ---------------------------------------------------------------------------
# 1. 逐字节中立
# ---------------------------------------------------------------------------

def test_default_source_is_byte_identical_to_explicit_hand_spec():
    """不传 `graph_source` 与显式传 `"hand_spec"` **逐字段完全相同**（含 timeline / live-set）。

    这是"加验收台不改模型"的机械证明：`hand_spec` 就是改造前 simulate.py 里那一行
    `ShapeEval().resolve(spec, pm)`。"""
    spec = _toy_spec(2)
    a, b = _run(spec), _run(spec, source=HAND_SPEC)
    assert a == b, "默认路径与显式 hand_spec 结果不同 —— 插座改动不是逐字节中立的"
    # 逐 stage 再显式比一遍关键字段（== 万一被 dataclass 语义放过）。
    for x, y in zip(a.per_stage, b.per_stage):
        assert (x.peak_bytes, x.peak_event, x.peak_substep, x.peak_mb) == \
               (y.peak_bytes, y.peak_event, y.peak_substep, y.peak_mb)
        assert x.buckets == y.buckets
        assert sorted(x.live_set, key=lambda i: str(i.key)) == \
               sorted(y.live_set, key=lambda i: str(i.key))
        assert x.timeline == y.timeline
        assert x.max_recompute_working_set == y.max_recompute_working_set


def test_build_stage_graphs_default_source_byte_identical():
    """`build_stage_graphs` 的插座同样中立。"""
    spec = _toy_spec(2)
    a = build_stage_graphs(spec, pp=1, m=1, grad_mode="chain2")
    b = build_stage_graphs(spec, pp=1, m=1, grad_mode="chain2", graph_source=HAND_SPEC)
    assert a.keys() == b.keys()
    for st in a:
        assert a[st] == b[st]


def test_hand_spec_matches_shape_eval_directly():
    """`resolve_graph(..., "hand_spec")` == `ShapeEval().resolve(...)`（同一个调用，无包装差异）。"""
    from cost_eval.shape_eval import ShapeEval
    spec = _toy_spec(2)
    pc = ParallelConfig(pp=1, num_microbatches=2)
    pm = ParallelModel(pc, spec.dims.n_layers, 1)
    g1 = resolve_graph(spec, pm, HAND_SPEC)
    g2 = ShapeEval().resolve(spec, pm)
    assert g1 == g2


# ---------------------------------------------------------------------------
# 2. 缝是真的 / fail-loud
# ---------------------------------------------------------------------------

def test_hand_spec_always_available_and_undeletable():
    assert HAND_SPEC in available_graph_sources()
    assert has_graph_source(HAND_SPEC)
    with pytest.raises(ValueError):
        unregister_graph_source(HAND_SPEC)


def test_unknown_source_fails_loud_with_available_list():
    spec = _toy_spec(1)
    with pytest.raises(KeyError) as ei:
        _run(spec, source="no_such_source", pp=1, recompute=RecomputeSpec("None"))
    assert "no_such_source" in str(ei.value) and HAND_SPEC in str(ei.value)


def test_duplicate_registration_fails_unless_override():
    def _dummy(spec, pm):
        from cost_eval.shape_eval import ShapeEval
        return ShapeEval().resolve(spec, pm)

    register_graph_source("_seam_dup", _dummy)
    try:
        with pytest.raises(ValueError):
            register_graph_source("_seam_dup", _dummy)
        register_graph_source("_seam_dup", _dummy, override=True)   # 显式覆盖放行
    finally:
        unregister_graph_source("_seam_dup")


def test_non_callable_and_stageless_return_fail_loud():
    with pytest.raises(TypeError):
        register_graph_source("_seam_bad", 42)

    register_graph_source("_seam_nostages", lambda spec, pm: object())
    try:
        spec = _toy_spec(1)
        pm = ParallelModel(ParallelConfig(pp=1, num_microbatches=1), spec.dims.n_layers, 1)
        with pytest.raises(TypeError) as ei:
            resolve_graph(spec, pm, "_seam_nostages")
        assert "stages" in str(ei.value)
    finally:
        unregister_graph_source("_seam_nostages")


def test_source_is_actually_consulted():
    """换来源结果**确实**变 —— 否则参数是装饰性的（最坏的假绿）。

    这里注册一个「砍掉最后一层」的来源：峰值必然更低。"""
    spec = _toy_spec(2)

    def _drop_last_layer(model_spec, pm):
        from cost_eval.shape_eval import ShapeEval
        g = ShapeEval().resolve(model_spec, pm)
        stages = {st: list(layers) for st, layers in g.stages.items()}
        last = max(stages)
        stages[last] = stages[last][:-1] or stages[last]
        return type(g)(stages=stages)

    register_graph_source("_seam_drop", _drop_last_layer)
    try:
        base = _run(spec, pp=1, recompute=RecomputeSpec("full", full_layers={0, 1}))
        cut = _run(spec, source="_seam_drop", pp=1,
                   recompute=RecomputeSpec("full", full_layers={0, 1}))
        assert cut.device_peak() < base.device_peak(), (
            "换来源后峰值未变 —— graph_source 参数没有真正接到解析上")
    finally:
        unregister_graph_source("_seam_drop")


def test_callable_source_accepted_without_registration():
    """`graph_source` 也可直接给 `fn(spec, pm)`（一次性实验不必注册）。"""
    from cost_eval.shape_eval import ShapeEval
    spec = _toy_spec(1)
    fn = graph_source(lambda s, pm: ShapeEval().resolve(s, pm))
    pm = ParallelModel(ParallelConfig(pp=1, num_microbatches=1), spec.dims.n_layers, 1)
    assert hasattr(fn(spec, pm), "stages")


def test_extracted_entry_point_is_documented_and_probed():
    """`extracted` 的入口点是**约定死**的；opdag 侧未落地时它就该不可用（而非报错）。"""
    assert EXTRACTED_ENTRY_POINT == ("cost_eval.opdag.to_resolved", "resolve_graph")
    # 探测不得抛（文件不存在是预期状态）；落地后 available 里会自动出现。
    avail = available_graph_sources()
    assert HAND_SPEC in avail
    if has_graph_source(EXTRACTED):
        # 已落地 → 必须真的可调用（不允许注册一个空壳）
        assert callable(graph_source(EXTRACTED))


# ---------------------------------------------------------------------------
# 3. 来源无关：外部生产者喂出同样的结果
# ---------------------------------------------------------------------------

def test_foreign_contract_producer_gives_identical_peaks():
    """一个**完全外部**的生产者（自家 dataclass，非 `shape_eval.ResolvedLayer`）结果完全一致。

    这就是 `opdag/to_resolved.py` 要走的路：仿真器只依赖
    `cost_eval/liveness/contract.py` 的契约面，不依赖具体类。"""
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class FT:                     # 外部 tensor：**恰好**契约要求的字段，一个不多
        name: str
        local_numel: int
        dtype_bytes: int
        is_weight: bool
        detached: bool
        is_expert: bool
        dim0: int
        pin_under_recompute: bool

    @dataclass(frozen=True)
    class FO:                     # 外部 op
        name: str
        type: str
        inputs: tuple
        output: FT
        params: tuple
        saves: tuple
        workspace_bytes: int = 0
        bwd_scratch_bytes: int = 0
        collectives: tuple = ()

    @dataclass(frozen=True)
    class FL:                     # 外部 layer
        layer_id: int
        layer_type: str
        ops: tuple

    @dataclass(frozen=True)
    class FG:
        stages: dict = field(default_factory=dict)

    def _mirror(t):
        return FT(name=t.name, local_numel=t.local_numel, dtype_bytes=t.dtype_bytes,
                  is_weight=t.is_weight, detached=t.detached, is_expert=t.is_expert,
                  dim0=t.dim0, pin_under_recompute=t.pin_under_recompute)

    def _foreign(model_spec, pm):
        from cost_eval.shape_eval import ShapeEval
        g = ShapeEval().resolve(model_spec, pm)
        stages = {}
        for st, layers in g.stages.items():
            stages[st] = [FL(layer_id=l.layer_id, layer_type=l.layer_type,
                             ops=tuple(FO(name=op.name,
                                          type=str(getattr(op.type, "value", op.type)),
                                          inputs=tuple(_mirror(t) for t in op.inputs),
                                          output=_mirror(op.output),
                                          params=tuple(_mirror(t) for t in op.params),
                                          saves=tuple(_mirror(t) for t in op.saves),
                                          workspace_bytes=op.workspace_bytes,
                                          bwd_scratch_bytes=op.bwd_scratch_bytes)
                                       for op in l.ops))
                          for l in layers]
        return FG(stages=stages)

    register_graph_source("_seam_foreign", _foreign)
    try:
        spec = _toy_spec(2)
        a = _run(spec)
        b = _run(spec, source="_seam_foreign")
        assert [p.peak_bytes for p in a.per_stage] == [p.peak_bytes for p in b.per_stage], (
            "外部契约生产者的峰值与手写路径不同 —— 仿真器仍在依赖具体类而非契约")
        assert [p.peak_substep for p in a.per_stage] == [p.peak_substep for p in b.per_stage]
        assert [p.max_recompute_working_set for p in a.per_stage] == \
               [p.max_recompute_working_set for p in b.per_stage]
    finally:
        unregister_graph_source("_seam_foreign")
