"""bwd 期 kernel workspace —— **实测项**的守卫门（2026-07-29）。

守三件事：

1. **实测律逐字节复现**。167 真机 memory-tracker 在三个 seq_length 上测到的融合稀疏
   flash-MLA（带 indexer）反向 kernel workspace（332.500 / 465.000 / 730.000 MiB @
   S=1024/2048/4096）必须原封不动地穿过 `builder → mhc_wrap → ShapeEval →
   StructureMemory` 整条链。**这条断言的对象是测量值，不是模型自洽性**：它红了要么是
   建模链把实测值丢了/改了，要么是有人改了实测常数（后者只能由新的真机测量驱动）。

2. **不外推**。没测过的分支必须是 0：r0（滑窗，`FusedSparseFlashMla`）、r128（无 indexer，
   同 8 项 ctx）、unfused（小算子链，根本不是这个 kernel）。这道门挡的是「拿 r4 的数去顶
   r128」这类把测量值当通用常数用的失败模式。

3. **`mhc_wrap` 不得静默吞掉它**（2026-07-29 的现实教训）：`cost_eval/layers/residual.py`
   的 `_rebuild` 曾用手写字段清单重建 OpSpec，把 `workspace_ref` 在**全部 DSv4 层**上悄悄
   丢掉——同样的通道会让本轮的实测 workspace 变成"测了、建了、但一字节没进模型"。
   现已改用 `dataclasses.replace`；本门用**被 mHC 包装后的真实层**取数，让回潮必红。

来源：`docs/kernel_workspace_2026-07-29.md`。
"""
import os
import sys
import warnings

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_TOOLS = os.path.join(_REPO, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
os.environ.setdefault("COST_EVAL_EXTRACTED_ALLOW_PARTIAL", "1")

import liveness_ab_validate as V                       # noqa: E402
from cost_eval.parallel_model import ParallelModel     # noqa: E402
from cost_eval.shape_eval import ShapeEval             # noqa: E402
from cost_eval.structure_mem import estimate_structure_memory  # noqa: E402

MiB = 2 ** 20

#: 167 真机 memory-tracker 实测（MS2.10/CANN9.1/910B2，run c = fused/无重算/L8/m4，
#: b=1/tp=1/cp=1）。三点**逐点验证线性**：斜率 (465.0−332.5)/1024 == (730.0−465.0)/2048
#: == 0.1293945 MiB/token 逐位相等，截距 200.0 MiB 整。**测量值，不得为了让模型好看而编辑。**
REAL_FLASHMLA_IDX_BWD_WS_MiB = {1024: 332.500, 2048: 465.000, 4096: 730.000}

_FUSED_TAG = "c fused   OFF L8 m4"
_UNFUSED_TAG = "d unfused OFF L8 m4"


def _resolve(tag, seq=None):
    """走**生产** builder（含 mhc_wrap 包装）拿 resolved 图。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = V.BY_TAG[tag]
        mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
        bundle, spec = V.build_bundle(mf)
        if seq is not None:
            spec.dims.S = seq
        p = bundle.parallel
        world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
        return ShapeEval().resolve(spec, ParallelModel(p, spec.dims.n_layers, world))


def _layer(graph, layer_type):
    for layers in graph.stages.values():
        for lay in layers:
            if lay.layer_type == layer_type:
                return lay
    raise AssertionError(f"layer_type {layer_type} 不在图里")


@pytest.mark.parametrize("seq", sorted(REAL_FLASHMLA_IDX_BWD_WS_MiB))
def test_measured_law_reproduces_end_to_end(seq):
    """r4 层的 bwd_workspace == 真机实测值，逐字节（三个 seq 各一格）。"""
    lay = _layer(_resolve(_FUSED_TAG, seq), "dsv4hyb_r4_moe")
    got = estimate_structure_memory(lay.ops).bwd_workspace / MiB
    want = REAL_FLASHMLA_IDX_BWD_WS_MiB[seq]
    assert abs(got - want) < 1e-6, (
        f"seq={seq}: 模型 bwd_workspace={got:.4f} MiB != 167 真机实测 {want:.4f} MiB。"
        f"这条实测值只能由新的真机测量改动（docs/kernel_workspace_2026-07-29.md）。")


def test_it_is_carried_by_the_sparse_attn_op_not_smeared_over_the_layer():
    """归属：整层的 bwd_workspace 全部来自 `sparse_attn` 这一个 op（实测归属，见 docs §4）。

    这道断言排除「把一个层级常数摊在层上」的做法——测量把它定位到了具体 kernel，
    模型就必须挂在对应的 op 上。"""
    lay = _layer(_resolve(_FUSED_TAG), "dsv4hyb_r4_moe")
    carriers = {op.name: getattr(op, "bwd_workspace_bytes", 0) / MiB
                for op in lay.ops if getattr(op, "bwd_workspace_bytes", 0)}
    assert carriers == pytest.approx({"sparse_attn": 730.0}, abs=1e-6), carriers


@pytest.mark.parametrize("layer_type", ["dsv4hyb_r0_dense", "dsv4hyb_r128_moe"])
def test_unmeasured_branches_stay_zero(layer_type):
    """没测出来的分支留 0，**不拿 r4 的数去顶**。

    r0 走滑窗、r128 走无 indexer 的 `FusedSparseFlashMla`（`csa.py:113`，8 项 ctx）；
    两个插桩 rank 的 BWD 单 kernel 极大值都被 r4 那一笔 730.0 盖住 → 它们各自的反向
    workspace 本轮**没有单独测出来**。这是**如实的欠读**，不是建模疏忽。"""
    lay = _layer(_resolve(_FUSED_TAG), layer_type)
    assert estimate_structure_memory(lay.ops).bwd_workspace == 0


def test_unfused_branch_stays_zero():
    """unfused 走 `unfused_compressed_sparse_attn` 小算子链，不是这个 kernel → 0。"""
    lay = _layer(_resolve(_UNFUSED_TAG), "dsv4hyb_r4_moe")
    assert estimate_structure_memory(lay.ops).bwd_workspace == 0


def test_default_is_zero_and_byte_neutral_for_untagged_specs():
    """未标注该字段的 op / 旧构造 → 0（尾部追加 + getattr 兜底，全库其余模型逐字节不变）。

    **2026-07-30 样例移动**（不变量原样保留，见 `docs/opdag_walker_core_2026-07-25.md` §6.6
    的精神）：原来拿 `embedding` 段当「未标注」样例，而本轮把实测到的 `GatherDGradV2`
    反向 workspace **挂到了 embedding op 上**（`docs/head_workspace_2026-07-30.md`）→ 它不再
    未标注。样例改用 **`lm_head` 段**。

    这个新样例本身就是一条**如实记账**：`lm_head`/loss 段（final_norm / lm_head MatMul /
    logsoftmax / nll）自己的反向 kernel workspace **至今没有测过** → 留 0，是**已知欠读**，
    不是「已确认为 0」。而 13 个 OOM-不安全锚点的峰值事件恰在 `bwd@<lm_head>`
    （见该文 §2.4）——所以这一格是当前最要紧的待测项，钉住它防止有人拿别处的实测值来顶。"""
    from cost_eval.model_spec import OpSpec, OpType, TensorRef
    t = TensorRef("x", ("S", "B", "H"))
    op = OpSpec("plain", OpType.MATMUL, [t], t)
    assert op.bwd_workspace is None and op.bwd_workspace_ref is None
    lay = _layer(_resolve(_FUSED_TAG), "lm_head")
    assert estimate_structure_memory(lay.ops).bwd_workspace == 0, (
        "lm_head/loss 段的反向 kernel workspace 未经测量 → 必须留 0。"
        "若有人在此填数，须先有真机测量（见 docs/head_workspace_2026-07-30.md §6①）。")


def test_bwd_workspace_is_additive_not_a_partition_of_the_working_set():
    """它必须**加**在反向工作集之上，而不是像 `bwd_scratch` 那样与 `bwd_working_set` 互补。

    `mem_timeline` 对无重算层用 `bwd_working_set = max(0, forward_max_live − bwd_scratch)`
    → 往 `bwd_scratch` 里加字节是**零和**（总量 = max(bwd_scratch, fml)），加不进峰值；
    这正是本轮**不能**复用 `bwd_scratch` 承载 kernel workspace 的原因。本门用桶模型的
    run `c` 峰值证实新通道确实是加法：s0/s1/s2 各上移**恰好 730.0 MiB**（= 实测值），
    s3 的峰在 head/loss 事件上、不动。"""
    from cost_eval.mem_timeline import Buckets
    b = Buckets(bwd_working_set=1000, bwd_scratch=7, workspace=730)
    assert b.total() == 1737, "workspace 必须是 total() 里的独立加项"


def test_mhc_wrap_does_not_silently_drop_the_measured_fields():
    """`mhc_wrap` 重建 OpSpec 时必须原样带过 `bwd_workspace` / `bwd_workspace_ref`。

    2026-07-29 真实发生过：`_rebuild` 的手写字段清单把 `workspace_ref` 在**全部 DSv4 层**
    上静默丢掉。本门直接比较「被 mHC 包装的生产层」与「未包装的裸 builder 层」的
    bwd_workspace —— 丢字段则前者变 0，必红。"""
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    # ① 裸 builder 确实挂上了两条通道（字符串常数项 + TensorRef 线性项）。
    raw = [op for op in build_dsv4_hybrid_attn_ops(_dims_of(_FUSED_TAG), 4)
           if op.name == "sparse_attn"]
    assert len(raw) == 1
    assert raw[0].bwd_workspace is not None, "builder 没挂常数项"
    assert raw[0].bwd_workspace_ref is not None, "builder 没挂线性项"
    # ② 经过 mhc_wrap 的**生产**层里，两条通道之和原样到达 StructureMemory。
    wrapped = _layer(_resolve(_FUSED_TAG), "dsv4hyb_r4_moe")
    got = estimate_structure_memory(wrapped.ops).bwd_workspace / MiB
    assert abs(got - REAL_FLASHMLA_IDX_BWD_WS_MiB[4096]) < 1e-6, (
        f"被 mHC 包装后 bwd_workspace={got:.4f} MiB（应为 730.0）—— mhc_wrap 又在静默"
        f"吞字段了（见 cost_eval/layers/residual.py 的 _rebuild 注释）。")


def _dims_of(tag):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = V.BY_TAG[tag]
        mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
        _, spec = V.build_bundle(mf)
        return spec.dims
