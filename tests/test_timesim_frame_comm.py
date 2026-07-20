"""frame_comm：框架层通信注入 FSDP/EP/CP（spec §3.3c 第二类——不在 layer construct 源码里的通信）。"""
import pytest

from cost_eval.timesim.ir import TimedOp, TimedSegment, STREAM_DEVICE
from cost_eval.timesim.frame_comm import inject_fsdp, inject_ep, inject_cp, fsdp_regather


def _mm(op_id="a#1", w=(1792, 6144)):
    return TimedOp(op_id=op_id, op_type="MatMul", phase="fwd",
                   in_shapes=((4096, 1, 1792), w), out_shape=(4096, 1, 6144),
                   dtype="bf16", stream="device", src="mlp.py:1")


def test_fsdp_gather_injected_at_segment_head():
    seg = TimedSegment("layer_0.fwd", (_mm(), _mm("a#2", w=(3072, 1792))))
    seg2 = inject_fsdp(seg, dp_shard=4)
    first = seg2.ops[0]
    assert first.op_type == "CommOp" and first.comm.ctype == "all_gather"
    assert first.stream == "comm_dp" and first.comm.group_axis == "dp"
    # AG volume=gather 前本 rank 分片（ir.py 约定）；记全量会被 T1 的 (n-1)·系数二次放大 ~n×
    assert first.comm.volume_bytes == (1792 * 6144 + 3072 * 1792) * 2 // 4
    assert seg2.ops[1:] == seg.ops


def test_fsdp_noop_when_unsharded():
    seg = TimedSegment("l.fwd", (_mm(),))
    assert inject_fsdp(seg, dp_shard=1) is seg


def test_fsdp_noop_on_empty_segment():
    seg = TimedSegment("l.fwd", ())
    assert inject_fsdp(seg, dp_shard=4) is seg


def test_fsdp_noop_when_no_gemm_weights():
    """无 GEMM 权重（full==0）→ 无可 gather → 恒等（与既有恒等语义一致）。"""
    norm = TimedOp(op_id="a#7", op_type="Norm", phase="fwd",
                   in_shapes=((4096, 1, 1792),), out_shape=(4096, 1, 1792),
                   dtype="bf16", stream="device", src="norm.py:1")
    seg = TimedSegment("l.fwd", (norm,))
    assert inject_fsdp(seg, dp_shard=4) is seg


def test_fsdp_gather_volume_ceils_on_indivisible_shard():
    # full=3·5·2=30 字节，dp_shard=4 → ceil(30/4)=8——钉 ceil 分支（FSDP flat-param pad 口径）
    seg = TimedSegment("l.fwd", (_mm(w=(3, 5)),))
    seg2 = inject_fsdp(seg, dp_shard=4)
    assert seg2.ops[0].comm.volume_bytes == 8


def _fsdp_layer():
    """mm（权重）+ norm，与 test_timesim_report.py 的 _layer(0) 同构——用于验证 bwd 段
    "第一个 device op"落在 bwd 逆序首位（NormGrad），而非碰巧的单 op 段。"""
    mm = TimedOp(op_id="c#0", op_type="MatMul", phase="fwd",
                 in_shapes=((1024, 1, 512), (512, 512)), out_shape=(1024, 1, 512),
                 dtype="bf16", stream="device", src="l.py:1")
    nm = TimedOp(op_id="c#1", op_type="Norm", phase="fwd",
                 in_shapes=((1024, 1, 512),), out_shape=(1024, 1, 512),
                 dtype="bf16", stream="device", src="l.py:2", deps=("c#0",))
    return TimedSegment("layer_0.fwd", (mm, nm))


def test_fsdp_bwd_regather_wired_into_first_device_op():
    """review [12]：重 gather AG 插 bwd 段头，但 deps=() 且没有任何 bwd op 依赖它——结构上
    恒可重叠（segment_sim 只认 TimedOp.deps，看不到它其实挡在 bwd 计算前面）。
    修复：AG 本身仍 deps=()（可预取），但 bwd 段第一个 device（计算）op 追加依赖该 AG
    （bwd 计算等权重重 gather 完成；segment_sim 的 device 流 FIFO 会让后续 device op
    也自然排在其后，无需逐个都挂 dep）。"""
    from cost_eval.timesim.bwd_rules import expand_bwd

    fwd = inject_fsdp(_fsdp_layer(), dp_shard=4)
    bwd = expand_bwd(fwd)
    bwd2 = fsdp_regather(bwd, fwd, dp_shard=4)

    ag = bwd2.ops[0]
    assert ag.op_type == "CommOp" and ag.comm.ctype == "all_gather"
    assert ag.deps == ()                          # AG 本身仍可预取，不被卡
    first_device = next(o for o in bwd2.ops if o.stream == STREAM_DEVICE)
    assert first_device.op_id == "c#1.b0"         # bwd 逆序首位＝NormGrad（非碰巧的头一个）
    assert ag.op_id in first_device.deps          # bwd 首个 device op 须等重 gather 完成


def test_ep_alltoall_wraps_grouped_region():
    g = TimedOp(op_id="m#5", op_type="GroupedMatMul", phase="fwd",
                in_shapes=((2, 2048, 1792), (2, 1792, 1024)), out_shape=(2, 2048, 1024),
                dtype="bf16", stream="device", src="moe.py:1")
    seg = TimedSegment("moe.fwd", (_mm(), g, _mm("a#9", w=(1024, 1792))))
    seg2 = inject_ep(seg, ep=4)
    kinds = [(o.op_type, getattr(o.comm, "ctype", None)) for o in seg2.ops]
    assert kinds[1] == ("CommOp", "all_to_all")   # dispatch 在 grouped 区前
    assert kinds[3] == ("CommOp", "all_to_all")   # combine 在 grouped 区后
    disp = seg2.ops[1]
    assert disp.stream == "comm_ep" and disp.comm.group_size == 4
    assert disp.comm.volume_bytes == 2 * 2048 * 1792 * 2   # grouped 输入激活字节（balanced 口径）
    comb = seg2.ops[3]
    assert comb.comm.volume_bytes == 2 * 2048 * 1024 * 2   # grouped 输出字节（combine 侧）


def test_ep_noop_without_grouped():
    seg = TimedSegment("l.fwd", (_mm(),))
    assert inject_ep(seg, ep=4) is seg


def test_ep_dispatch_combine_wired_into_deps():
    """review [0]：旧码 disp/comb 的 deps=()、grouped 区首 op 保持原 deps、下游仍指向 grouped
    op——segment_sim 的跨流排程只认显式 deps、不看列表位置，于是这条强制串行的通信（token
    dispatch/combine）被当成完全可重叠。修复：串成 router→disp→grouped→comb→下游
    （mirror producer.py Row.rs 的下游 redirect 手法）。"""
    before = TimedOp(op_id="a#1", op_type="MatMul", phase="fwd",
                      in_shapes=((4096, 1, 1792), (1792, 6144)), out_shape=(4096, 1, 6144),
                      dtype="bf16", stream="device", src="mlp.py:1")
    g = TimedOp(op_id="m#5", op_type="GroupedMatMul", phase="fwd",
                in_shapes=((2, 2048, 1792), (2, 1792, 1024)), out_shape=(2, 2048, 1024),
                dtype="bf16", stream="device", src="moe.py:1", deps=("router#0",))
    after = TimedOp(op_id="a#9", op_type="MatMul", phase="fwd",
                    in_shapes=((2, 2048, 1024), (1024, 1792)), out_shape=(2, 2048, 1792),
                    dtype="bf16", stream="device", src="mlp.py:2", deps=("m#5",))
    seg = TimedSegment("moe.fwd", (before, g, after))
    seg2 = inject_ep(seg, ep=4)

    ops = {o.op_id: o for o in seg2.ops}
    disp, comb = ops["moe.fwd.ep_disp"], ops["moe.fwd.ep_comb"]
    grouped, downstream = ops["m#5"], ops["a#9"]
    assert disp.deps == ("router#0",)                  # ① disp 等 grouped 区首 op 的原生产者
    assert grouped.deps == ("router#0", disp.op_id)     # ② grouped 首 op 追加等 dispatch
    assert comb.deps == ("m#5",)                        # ③ combine 等 grouped 区末 op
    assert downstream.deps == (comb.op_id,)             # ④ 下游改指 combine（不再直连 grouped）


def test_ep_dispatch_combine_exposed_in_segment_sim():
    """集成（review [0]）：串上 deps 后，simulate_segment 应把 dispatch/combine 的传输时间
    算进 exposed_comm['ep']——修复前两者 deps=() 且无人依赖，t_exposed_comm 里恒无 'ep' 键
    （或恒为 0），通信被系统性当成免费。"""
    from cost_eval.timesim.machine import synth_hw
    from cost_eval.timesim.op_cost import CostModel, price_segment
    from cost_eval.timesim.segment_sim import simulate_segment

    before = TimedOp(op_id="a#1", op_type="MatMul", phase="fwd",
                      in_shapes=((4096, 1, 1792), (1792, 6144)), out_shape=(4096, 1, 6144),
                      dtype="bf16", stream="device", src="mlp.py:1")
    g = TimedOp(op_id="m#5", op_type="GroupedMatMul", phase="fwd",
                in_shapes=((2, 2048, 1792), (2, 1792, 1024)), out_shape=(2, 2048, 1024),
                dtype="bf16", stream="device", src="moe.py:1")
    after = TimedOp(op_id="a#9", op_type="MatMul", phase="fwd",
                    in_shapes=((2, 2048, 1024), (1024, 1792)), out_shape=(2, 2048, 1792),
                    dtype="bf16", stream="device", src="mlp.py:2", deps=("m#5",))
    seg2 = inject_ep(TimedSegment("moe.fwd", (before, g, after)), ep=4)

    cm = CostModel(synth_hw())
    costs = price_segment(seg2, cm)
    st = simulate_segment(seg2, costs)
    assert st.t_exposed_comm.get("ep", 0.0) > 0.0


def test_cp_ring_structural_blocks():
    """colossal ring 结构化（T1-4）：cp 个 FA 块、块间 cp-1 条单跳 kv p2p；
    p2p_k deps=()（段首即可发）、FA_k deps 含 p2p_k → p2p 与上一块 FA 的重叠交给 DES 涌现。
    下游依赖不换绑：最末块保留原 op_id。"""
    from cost_eval.timesim.ir import TimedOp, TimedSegment
    from cost_eval.timesim.frame_comm import inject_cp
    fa = TimedOp(op_id="a#0", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224),) * 3 + ((),), out_shape=(2048, 1, 8, 224),
                 dtype="bf16", stream="device", src="attention.py:1", deps=("a#9",))
    seg = inject_cp(TimedSegment("l.fwd", (fa,)), cp=4, method="colossal")
    fas = [o for o in seg.ops if o.op_type == "FlashAttention"]
    p2ps = [o for o in seg.ops if o.op_type == "CommOp"]
    assert len(fas) == 4 and len(p2ps) == 3
    assert [o.comm.ctype for o in p2ps] == ["p2p"] * 3
    assert all(o.stream == "comm_cp" and o.deps == () for o in p2ps)
    # 交替序：FA0, p2p1, FA1, p2p2, FA2, p2p3, FA3
    assert [o.op_type for o in seg.ops] == ["FlashAttention", "CommOp"] * 3 + ["FlashAttention"]
    # 末块保留原 id（下游 deps 不换绑）；前块带 .cpK 后缀
    assert fas[-1].op_id == "a#0"
    assert fas[0].op_id == "a#0.cp0"
    # 每块 FA 保留原跨流 deps；k≥1 块追加对应 p2p dep
    assert fas[0].deps == ("a#9",)
    assert p2ps[0].op_id in fas[1].deps and "a#9" in fas[1].deps
    # 末块（保留原 op_id、下游 deps 实际挂钩的块）也须换绑其对应 p2p（review 覆盖缺口补齐）
    assert p2ps[-1].op_id in fas[-1].deps and "a#9" in fas[-1].deps
    # p2p 载荷 = k+v 单块字节（单跳口径）
    kv = 2048 * 1 * 8 * 224 * 2
    assert all(o.comm.volume_bytes == 2 * kv for o in p2ps)


def test_cp_ulysses_alltoall_both_sides():
    # 真机 FA 调用含 attn_mask（第 4 入参，bf16 S×S 可比肩 q）——ulysses a2a 载荷只切
    # q,k,v（in_shapes[:3]，bprop_rules.py:31 口径），mask 不重排、全量求和会静默膨胀载荷。
    fa = TimedOp(op_id="a#3", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224),) * 3 + ((2048, 2048),),
                 out_shape=(2048, 1, 8, 224),
                 dtype="bf16", stream="device", src="attention.py:1")
    seg2 = inject_cp(TimedSegment("l.fwd", (fa,)), cp=2, method="ulysses")
    assert [o.op_type for o in seg2.ops] == ["CommOp", "FlashAttention", "CommOp"]
    ctypes = [o.comm.ctype for o in seg2.ops if o.op_type == "CommOp"]
    assert ctypes == ["all_to_all", "all_to_all"]
    pre, post = seg2.ops[0], seg2.ops[2]
    assert pre.comm.volume_bytes == 3 * (2048 * 1 * 8 * 224 * 2)
    assert post.comm.volume_bytes == 2048 * 1 * 8 * 224 * 2   # post a2a 载荷 = FA 输出字节


def test_cp_unknown_method_fail_loud():
    with pytest.raises(ValueError):
        inject_cp(TimedSegment("l.fwd", (_mm(),)), cp=2, method="ring2")
    # cp 不活跃（cp<=1）时 typo 也要报——method 校验先于早退
    with pytest.raises(ValueError):
        inject_cp(TimedSegment("l.fwd", (_mm(),)), cp=1, method="ring2")


def test_cp_colossal_fa_arity_guard():
    """review [14]：colossal 支直接读 op.in_shapes[1]/[2]（k/v）取 kv_bytes，FA op 若 <3 个
    in_shapes → 裸 IndexError（op_cost._fa_flops 与 ulysses 支的 in_shapes[:3] 切片都是
    fail-loud/安全切片，colossal 独漏）。修复：读之前加与 op_cost 同口径的 fail-loud 守卫。"""
    fa = TimedOp(op_id="a#0", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224), (2048, 1, 8, 224)),   # 仅 2 个（缺 v）
                 out_shape=(2048, 1, 8, 224), dtype="bf16", stream="device",
                 src="attention.py:1")
    seg = TimedSegment("l.fwd", (fa,))
    with pytest.raises(ValueError, match="in_shapes"):
        inject_cp(seg, cp=2, method="colossal")


def test_injector_order_independence_on_mixed_segment():
    """三注入器组合顺序无关（冻结该性质）：各注入器锚定的 op 类型（GEMM 权重/FlashAttention/
    GroupedMatMul）都不是对方的产物（CommOp），故应用先后不影响产出 op 序列。"""
    fa = TimedOp(op_id="a#3", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224),) * 3, out_shape=(2048, 1, 8, 224),
                 dtype="bf16", stream="device", src="attention.py:1")
    g = TimedOp(op_id="m#5", op_type="GroupedMatMul", phase="fwd",
                in_shapes=((2, 2048, 1792), (2, 1792, 1024)), out_shape=(2, 2048, 1024),
                dtype="bf16", stream="device", src="moe.py:1")
    seg = TimedSegment("mix.fwd", (_mm(), fa, g))
    a = inject_cp(inject_ep(inject_fsdp(seg, dp_shard=4), ep=4), cp=2, method="colossal")
    b = inject_fsdp(inject_ep(inject_cp(seg, cp=2, method="colossal"), ep=4), dp_shard=4)
    assert a.ops == b.ops
