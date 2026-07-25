"""liveness 张量类别 → 既有桶名的映射（设计要求 7：桶分类变成**标签**，不再是计算本身）。

桶模型（`mem_timeline.Buckets`）把内存**算**成一组闭式桶之和；liveness 反过来：先逐张量算出
「峰值时刻同时在世的集合」，再给每个张量**贴一个类别标签**，聚合后即可与既有桶名对账。
这样每个字节**只被算一次**（桶模型里 `remat_saves`/`recomp_scratch`/`bwd_working_set` 三者
都由 `forward_max_live`/`saves` 派生、彼此部分重叠且无可解析扣减的重叠项，是两向偏差的根因）。

类别（liveness 自己产出的、逐张量的）：
  - ``act_boundary``     区域边界（checkpoint_input）——重算下前向末唯一驻留量。
  - ``act_saved``        save_for_backward 张量，从前向驻留到其反向消费者（无重算 / select 未选段）。
  - ``fwd_transient``    前向中间量，最后一次被用完即释（不被任何反向节点读）。
  - ``recomp_saved``     **重算再执行**期重物化的 saved 张量（必须活到该区域 backward 消费完）。
  - ``recomp_transient`` 重算再执行期的纯瞬态（区域内用完即释）。
  - ``grad_act``         激活梯度 dL/dact（反向节点产出、被其生产者的反向节点消费）。
  - ``grad_internal``    op **内部**中间量的梯度（仅 ``grad_mode="chain2"``；见 graph.py）。

非 liveness 桶（原样沿用 `mem_timeline` 的既有标定/公式，设计要求 5）：
  ``persistent`` / ``gather_buf`` / ``grad_buf`` / ``grad_accum`` / ``bwd_scratch`` /
  ``workspace`` / ``optstep`` / ``kept_frag`` / ``p2p_buf`` / ``swap_buf`` / ``framework``。
"""
from __future__ import annotations

# liveness 自产的逐张量类别（Σ 这些 == 峰值 live-set 字节和）。
LIVENESS_CATEGORIES = (
    "act_boundary", "act_saved", "fwd_transient",
    "recomp_saved", "recomp_transient",
    "grad_act", "grad_internal",
)

# 非 liveness 桶（沿用 mem_timeline 既有公式/标定，不由 liveness 重算）。
NON_LIVENESS_BUCKETS = (
    "persistent", "gather_buf", "grad_buf", "grad_accum", "bwd_scratch",
    "workspace", "optstep", "kept_frag", "p2p_buf", "swap_buf", "framework",
)

# liveness 类别 → `mem_timeline.Buckets` 字段名（供与桶模型/真机逐桶对账）。
BUCKET_OF_CATEGORY = {
    "act_boundary": "act_live",
    "act_saved": "act_live",
    "fwd_transient": "act_live",
    "recomp_saved": "remat_saves",
    "recomp_transient": "recomp_scratch",
    "grad_act": "bwd_working_set",
    "grad_internal": "bwd_working_set",
}


def bucket_of(category: str) -> str:
    """类别 → 既有桶名；非 liveness 桶名原样返回（便于统一聚合）。"""
    return BUCKET_OF_CATEGORY.get(category, category)


def to_buckets(counts: dict) -> dict:
    """把 {类别: 字节} 聚合成 {既有桶名: 字节}（liveness 与非 liveness 混合字典亦可）。"""
    out: dict = {}
    for cat, nbytes in counts.items():
        out[bucket_of(cat)] = out.get(bucket_of(cat), 0) + nbytes
    return out
