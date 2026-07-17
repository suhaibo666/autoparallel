"""bwd 展开规则库（spec §3.3d/e）：bprop_rules 的姊妹件，逆拓扑序 + 通信对偶(volume 换算) + recompute 前缀。"""
import os

import pytest

from cost_eval.timesim.ir import TimedOp, TimedSegment, CommSpec, op_flops
from cost_eval.timesim.bwd_rules import expand_bwd


def _mm():
    return TimedOp(op_id="c#0", op_type="MatMul", phase="fwd",
                   in_shapes=((4096, 1, 1792), (1792, 3072)), out_shape=(4096, 1, 3072),
                   dtype="bf16", stream="device", src="mlp.py:1")


def _rs():
    return TimedOp(op_id="c#1.rs", op_type="CommOp", phase="fwd", in_shapes=((4096, 1, 1792),),
                   out_shape=(2048, 1, 1792), dtype="bf16", stream="comm_tp", src="layers.py:619",
                   comm=CommSpec("reduce_scatter", 4096 * 1792 * 2, "tp", 2))


def _ag():
    return TimedOp(op_id="c#2.ag", op_type="CommOp", phase="fwd", in_shapes=((2048, 1, 1792),),
                   out_shape=(4096, 1, 1792), dtype="bf16", stream="comm_tp", src="layers.py:1",
                   comm=CommSpec("all_gather", 2048 * 1792 * 2, "tp", 2))


def _view():
    return TimedOp(op_id="c#3", op_type="View", phase="fwd", in_shapes=((4096, 1, 1792),),
                   out_shape=(4096, 1792), dtype="bf16", stream="host_only", src="mlp.py:2")


def test_matmul_expands_to_dx_dw_with_2x_flops():
    seg = expand_bwd(TimedSegment("l.fwd", (_mm(),)))
    bw = [o for o in seg.ops if o.phase == "bwd"]
    assert [o.op_type for o in bw] == ["MatMul", "MatMul"]        # dX + dW
    assert sum(op_flops(o) for o in bw) == 2 * op_flops(_mm())
    assert bw[0].out_shape == _mm().in_shapes[0]                  # dX shape=输入
    assert bw[1].out_shape == _mm().in_shapes[1]                  # dW shape=权重


def test_single_input_matmul_fail_loud():
    """module=="" 透传 matmul 可能仅 1 入（ir.py 元数警示）——bwd 无权重可回传 → fail-loud 而非 IndexError。"""
    one_in = TimedOp(op_id="c#9", op_type="MatMul", phase="fwd",
                     in_shapes=((4096, 1, 1792),), out_shape=(4096, 1, 3072),
                     dtype="bf16", stream="device", src="x.py:1")
    with pytest.raises(ValueError):
        expand_bwd(TimedSegment("l.fwd", (one_in,)))


def test_comm_dual_with_volume_rescale():
    """RS→AG volume ÷group_size；AG→RS volume ×group_size（ir.py CommSpec 对偶换算规则）。"""
    seg = expand_bwd(TimedSegment("l.fwd", (_mm(), _rs())))
    bw = [o for o in seg.ops if o.phase == "bwd"]
    assert bw[0].op_type == "CommOp" and bw[0].comm.ctype == "all_gather"   # RS↔AG 对偶，先反向
    assert bw[0].comm.volume_bytes == (4096 * 1792 * 2) // 2               # RS 记全量 → AG 记分片 ÷2
    assert bw[0].stream == "comm_tp"
    seg2 = expand_bwd(TimedSegment("l.fwd", (_ag(),)))
    dual2 = [o for o in seg2.ops if o.phase == "bwd"][0]
    assert dual2.comm.ctype == "reduce_scatter"
    assert dual2.comm.volume_bytes == (2048 * 1792 * 2) * 2                # AG 记分片 → RS 记全量 ×2


def test_view_stays_host_only():
    seg = expand_bwd(TimedSegment("l.fwd", (_view(),)))
    assert all(o.stream == "host_only" for o in seg.ops if o.phase == "bwd")


def test_flash_attention_single_grad_kernel():
    fa = TimedOp(op_id="c#4", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224),) * 3, out_shape=(2048, 1, 8, 224),
                 dtype="bf16", stream="device", src="attention.py:1")
    bw = [o for o in expand_bwd(TimedSegment("l.fwd", (fa,))).ops if o.phase == "bwd"]
    assert [o.op_type for o in bw] == ["FlashAttentionGrad"]


def test_bandwidth_ops_single_grad():
    norm = TimedOp(op_id="c#5", op_type="Norm", phase="fwd", in_shapes=((4096, 1, 1792),),
                   out_shape=(4096, 1, 1792), dtype="fp32", stream="device", src="n.py:1")
    bw = [o for o in expand_bwd(TimedSegment("l.fwd", (norm,))).ops if o.phase == "bwd"]
    assert [o.op_type for o in bw] == ["NormGrad"] and bw[0].stream == "device"


def test_reverse_order():
    seg = expand_bwd(TimedSegment("l.fwd", (_mm(), _view())))
    bw = [o for o in seg.ops if o.phase == "bwd"]
    assert bw[0].op_type == "View" and bw[1].op_type == "MatMul"   # 逆序展开


def test_recompute_prefix_full_and_comm_drop():
    fwd = (_mm(), _rs(), _view())
    seg = expand_bwd(TimedSegment("l.fwd", fwd), recompute="full", recomp_comm=False)
    rc = [o for o in seg.ops if o.phase == "recomp"]
    assert [o.op_type for o in rc] == ["MatMul", "View"]          # 重放 fwd 序，剔 CommOp
    assert seg.ops[:len(rc)] == tuple(rc)                          # 前缀在 bwd 之前
    seg2 = expand_bwd(TimedSegment("l.fwd", fwd), recompute="full", recomp_comm=True)
    assert [o.op_type for o in seg2.ops if o.phase == "recomp"] == ["MatMul", "CommOp", "View"]


def test_seg_id_suffix():
    assert expand_bwd(TimedSegment("layer_0.fwd", (_mm(),))).seg_id == "layer_0.bwd"


def test_dual_table_ar_a2a_p2p_volume_unchanged():
    """_DUAL 表全覆盖：AR/A2A/p2p 自对偶且 volume 不变（ir.py 对偶换算规则的另半边）；
    未知 ctype fail-loud（ValueError 而非裸 KeyError）。"""
    from cost_eval.timesim.bwd_rules import _dual_comm
    for ct in ("all_reduce", "all_to_all", "p2p"):
        d = _dual_comm(CommSpec(ct, 1024, "tp", 4))
        assert d.ctype == ct and d.volume_bytes == 1024
    with pytest.raises(ValueError):
        _dual_comm(CommSpec("broadcast", 1024, "tp", 4))


# ── spec review F1/F2：bwd deps=fwd 依赖边反转 + recompute 前缀 deps 段内 remap ────
# （真 MLP tp2/sp 段验证——spec §5.1:230 跨流依赖只来自 TimedOp.deps，无"等前序列表项"规则）
MF_ROOT = os.environ.get("MINDFORMERS_ROOT",
                         r"E:\97-codes\torch_parallel\mindformers\mindformers")


def _mlp_seg():
    """producer 真 MLP tp2/sp fwd 段（与 test_timesim_producer._mlp_dag 同一装配约定）。"""
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import ResolvedSpec
    from cost_eval.opdag.shape_infer import infer_shapes
    from cost_eval.timesim.producer import build_segment
    from cost_eval.timesim.shard_rules import Degrees
    from cost_eval.model_spec import DimTable
    dims = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                    S=4096, B=1, vocab=129280, n_layers=4)
    dag = extract_cell(
        MF_ROOT, "parallel_core/training_graph/transformer/mlp.py", "MLPInterleaved",
        ResolvedSpec(cell="MLPInterleaved",
                     submodules={"linear_fc1": "ColumnParallelLinear",
                                 "linear_fc2": "RowParallelLinear"}),
        {"gated_linear_unit": True, "activation_type": "silu",
         "add_bias_linear": False, "compute_dtype": "bf16"})
    dag = infer_shapes(dag, {"hidden_states": "S·B·H"})
    return build_segment("layer_0.mlp.fwd", dag, dims,
                         Degrees(tp=2, sequence_parallel=True))


def test_bwd_deps_reverse_fwd_edges_real_mlp():
    """F1：bwd deps=fwd 依赖边反转——否则 bwd exposed comm 结构性归零。两条关键反转边：
    ① fwd Row矩乘→.rs（deps=(row,)）反转 ⇒ Row 的 dX(.b0) 和 dW(.b1) 都等 .rs 的对偶 AG
      （dW 也要 dy，与真实 autodiff 一致）；
    ② fwd .ag→Column矩乘（deps=(ag,)）反转 ⇒ 对偶 grad-RS(.ag.b0) 等 Column 的 dX(.b0)。"""
    fwd = _mlp_seg()
    ag = next(o for o in fwd.ops if o.op_id.endswith(".ag"))
    rs = next(o for o in fwd.ops if o.op_id.endswith(".rs"))
    col_id = ag.op_id[:-len(".ag")]
    row_id = rs.op_id[:-len(".rs")]
    by_id = {o.op_id: o for o in expand_bwd(fwd).ops}
    assert rs.op_id + ".b0" in by_id[row_id + ".b0"].deps   # ① dX ← 对偶AG
    assert rs.op_id + ".b0" in by_id[row_id + ".b1"].deps   # ① dW ← 对偶AG（dW 也要 dy）
    assert col_id + ".b0" in by_id[ag.op_id + ".b0"].deps   # ② grad-RS ← dX


def test_recompute_prefix_deps_remap_real_mlp():
    """F2：recompute 前缀 deps 段内 remap——被重放的生产者 +".r"，未重放（被剔 CommOp）drop
    （其输出 fwd 已保存故段首即可用，这正是它不重算的原因）；否则前缀 deps 悬空指 fwd op_id，
    重算的 AG→matmul 串行化丢失。"""
    fwd = _mlp_seg()
    ag = next(o for o in fwd.ops if o.op_id.endswith(".ag"))
    col_id = ag.op_id[:-len(".ag")]
    # recomp_comm=False：被剔通信的 dep 消失；前缀内全部 deps 段内可解析
    seg_f = expand_bwd(fwd, recompute="full", recomp_comm=False)
    rc_f = [o for o in seg_f.ops if o.phase == "recomp"]
    ids_f = {o.op_id for o in rc_f}
    assert all(d in ids_f for o in rc_f for d in o.deps)     # 无悬空
    col_r = next(o for o in rc_f if o.op_id == col_id + ".r")
    assert ag.op_id + ".r" not in col_r.deps and ag.op_id not in col_r.deps
    # recomp_comm=True：remap 到 .ag.r（重算的 AG→matmul 串行化保留）；同样无悬空
    seg_t = expand_bwd(fwd, recompute="full", recomp_comm=True)
    rc_t = [o for o in seg_t.ops if o.phase == "recomp"]
    ids_t = {o.op_id for o in rc_t}
    assert all(d in ids_t for o in rc_t for d in o.deps)     # 无悬空
    col_rt = next(o for o in rc_t if o.op_id == col_id + ".r")
    assert ag.op_id + ".r" in col_rt.deps
