"""逐张量 **liveness 仿真器** —— 解析式内存估计器的只读交叉校验（2026-07-25）。

桶模型（`cost_eval/mem_timeline.py`）用闭式聚合把峰值**算**成一组桶之和；本包反过来：
显式建前向图 + 反向图，按

    **前向产出的张量，直到它的最后一个消费者（前向 *或* 反向）跑完才释放**

一条规则求 ``peak = max over schedule of Σ 在世字节``，每张量恰算一次；重算表达为**图变换**
（前向丢 saved、反向前插区域前向再执行）而不是桶公式。详见 `graph.py` / `simulate.py` 头注。

**非权威**：不改桶路径一个字节（`Evaluator` 数值不变），只作对账与缺口定位工具。

用法
----
    from cost_eval.liveness import simulate_liveness
    res = simulate_liveness(spec, pc, opt, hw, recompute, swap, record_timeline=True)
    st = res.per_stage[0]
    st.peak_bytes, st.peak_event, st.peak_substep
    st.bucket_view()          # 折算到 mem_timeline 桶名
    st.top_live(20)           # 峰值时刻最大的 20 个张量（逐张量 dump）

CLI（per-stage 峰值 + 峰值时刻 live-set dump）：
    python -m cost_eval.liveness <mindformers.yaml> [--grad-mode chain2] [--top 25]
"""
from .categories import (BUCKET_OF_CATEGORY, LIVENESS_CATEGORIES,
                         NON_LIVENESS_BUCKETS, bucket_of, to_buckets)
from .graph import (BwdNode, FwdNode, LayerGraph, LTensor, build_layer_graph,
                    build_stage_graphs)
from .simulate import (LiveItem, LiveSample, LivenessResult, StageLiveness,
                       simulate_liveness)

__all__ = [
    "simulate_liveness", "LivenessResult", "StageLiveness", "LiveItem", "LiveSample",
    "build_layer_graph", "build_stage_graphs", "LayerGraph", "LTensor", "FwdNode", "BwdNode",
    "LIVENESS_CATEGORIES", "NON_LIVENESS_BUCKETS", "BUCKET_OF_CATEGORY",
    "bucket_of", "to_buckets",
]
