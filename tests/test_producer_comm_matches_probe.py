# tests/test_producer_comm_matches_probe.py
"""comm_probe × producer 通信语义交叉校验（code-review T1a [5]：comm_probe 子系统建了但从不
被 producer 消费——死架构）。

spec §3.3c 设计的是**探针驱动**注入：识别出源码里的显式通信惯用法就发 CommOp，识别不出才退回
模块语义注入并 fail-loud。producer.py（T1 范围）反过来——TP/SP 通信是**硬编码**语义（Row 的
`.rs` reduce_scatter/all_reduce 二选一、Column 的 sp all_gather），从不调用
`opdag.comm_probe.probe_cell_comm`。不重设计成运行时探针驱动（v1.5 工作），而是把 comm_probe
钉成**交叉校验守卫**：用 probe 从真 layers.py 提取的 CommSite（源真相）断言与 producer 硬编码
的 ctype/guard（消费假设）一致——源码 idiom 漂移或硬编码被误改，本文件测试即变红。

无 mindformers 源时全部 skip（沿用 test_opdag_comm_probe.py 的 _require_mf 模式）。
"""
import os

import pytest

from cost_eval.opdag.comm_probe import probe_cell_comm
from cost_eval.timesim.producer import build_segment
from cost_eval.timesim.shard_rules import Degrees
from cost_eval.model_spec import DimTable

MF_ROOT = os.environ.get("MINDFORMERS_ROOT",
                         r"E:\97-codes\torch_parallel\mindformers\mindformers")
TP_REL = "parallel_core/training_graph/tensor_parallel/layers.py"

# mlp_dag 夹具（tests/conftest.py）用真 MLPInterleaved 抽取；linear_fc2=RowParallelLinear，
# 与 test_timesim_producer.py 的 DIMS 保持一致，两处独立求值应得同一现实数字。
DIMS = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                S=4096, B=1, vocab=129280, n_layers=4)


def _require_mf():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")


def _row_sites():
    return probe_cell_comm(MF_ROOT, TP_REL, "RowParallelLinear")


def test_row_probe_ground_truth_has_both_ctype_guards():
    """基线（与 test_opdag_comm_probe.py 独立重复）：probe 从真源提取的 Row 通信二选一站点，
    是本文件后续交叉校验的源真相——不依赖另一测试文件的执行顺序或结果缓存。"""
    _require_mf()
    kinds = {(s.ctype, s.guard) for s in _row_sites()}
    assert ("reduce_scatter", "sequence_parallel") in kinds       # :619/:646
    assert ("all_reduce", "!sequence_parallel") in kinds          # :621/:648


def test_producer_row_sp_reduce_scatter_matches_probe_site(mlp_dag):
    """producer 硬编码：Row + tp>1 + sequence_parallel=True → `.rs` 的 ctype=reduce_scatter。
    交叉校验：probe 在真源里确实存在 guard=="sequence_parallel" 的 reduce_scatter 站点——
    producer 的 SP 选择不是凭空编的语义，与源站点一致。"""
    _require_mf()
    seg = build_segment("layer_0.mlp.fwd", mlp_dag, DIMS,
                         Degrees(tp=2, sequence_parallel=True))
    rs = next(o for o in seg.ops if o.op_id.endswith(".rs"))
    assert rs.comm.ctype == "reduce_scatter"
    kinds = {(s.ctype, s.guard) for s in _row_sites()}
    assert ("reduce_scatter", "sequence_parallel") in kinds


def test_producer_row_non_sp_all_reduce_matches_probe_site(mlp_dag):
    """producer 硬编码：Row + tp>1 + sequence_parallel=False → `.rs` 的 ctype=all_reduce。
    交叉校验：probe 在真源里确实存在 guard=="!sequence_parallel" 的 all_reduce 站点。"""
    _require_mf()
    seg = build_segment("layer_0.mlp.fwd", mlp_dag, DIMS,
                         Degrees(tp=2, sequence_parallel=False))
    rs = next(o for o in seg.ops if o.op_id.endswith(".rs"))
    assert rs.comm.ctype == "all_reduce"
    kinds = {(s.ctype, s.guard) for s in _row_sites()}
    assert ("all_reduce", "!sequence_parallel") in kinds


def test_row_and_embedding_probe_sites_have_no_opaque_guard():
    """guard 漂移守卫：probe 提取的 Row/Embedding 通信站点里不该出现 guard=="?"（全程不可识别）
    ——comm_probe.py docstring 承诺"消费方对 '?' fail-loud，不猜"。此测试钉住"当前源没有触发
    该形态"；源 idiom 变化到 probe 认不出的形态时本测试变红，是漂移预警，不是改测试去迁就。"""
    _require_mf()
    row_sites = _row_sites()
    emb_sites = probe_cell_comm(MF_ROOT, TP_REL, "VocabParallelEmbedding")
    assert all(s.guard != "?" for s in row_sites)
    assert all(s.guard != "?" for s in emb_sites)
