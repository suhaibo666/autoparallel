# cost_eval/timesim/frame_comm.py
"""frame_comm：框架层通信注入（spec §3.3c 第二类——不在 layer construct 源码里的通信，
与 producer.py 的 TP/SP 通信注入互补：那边管 layer construct 源码内显式通信，这边管 FSDP/dp、
EP dispatch/combine、CP 三条并行轴的注入，均为 fwd）。

FSDP：per-segment 权重 all-gather 注入段头（fwd 预取语义——按 spec §5.5，第 i+1 段的权重 gather
挂在第 i 段序列头部，与上段计算的重叠是 pipeline_sim（T1）里"位置"的自然结果，本模块不建模
重叠本身）；载荷 = 段内 GEMM 族权重（in_shapes[1]）字节和 ÷ dp_shard（ir.py CommSpec AG 约定
=分片入参字节）——len(in_shapes)<2 的 GEMM 跳过（module=="" 的透传 matmul 可能仅 1 入，arity
现实见 ir.py in_shapes 字段注释）。dp_shard<=1（未切片）→ 恒等，不注入。T1 契约：FSDP gather
挂各段自身头部+deps=()——等效于 §5.5 预取语义的前提是 T1 DES 允许 comm_dp 跨段 run-ahead；
若 T1 加段边界同步，此 AG 将全暴露（届时需改挂上一段）。

EP：GroupedMatMul 区两侧包一对 dispatch/combine all-to-all（balanced 路由口径——继承内存侧同一
假设，v1 不建 unbalanced 容量溢出）；dispatch 载荷 = 该区首个 GroupedMatMul 的输入激活字节，
combine 载荷 = 该区末个 GroupedMatMul 的输出字节。ep<=1 或段内无 GroupedMatMul → 恒等。

CP：colossal（环状 attention）→ 结构化为 cp 个 FlashAttention 块 × 块间 cp−1 条单跳 kv p2p
交替（T1-4；载荷=k+v 字节，p2p 恒单跳——跳数已结构化，不再留给 op_cost 系数；块间 p2p 与
上一块 FA 的重叠交给 pipeline_sim/segment_sim 的 DES 涌现，本模块只管"块间该出现一跳 p2p
通信"这一结构事实）；FA op 的 in_shapes 须 ≥3（q/k/v）——不足则 ValueError（fail-loud，
与 op_cost._fa_flops 同口径，code-review [14]）；ulysses → FlashAttention
前后各插一条 all_to_all（seq-parallel↔head-parallel 切换）。method 不在
{"colossal","ulysses"} → ValueError（fail-loud，不猜语义）。cp<=1 → 恒等。

三者共同点：注入 op 优先摆在段内自然位置（段头 / 区两侧 / FA 两侧），位置本身是 spec §5.5
overlap 语义的来源；但 segment_sim（T1）的跨流排程只认 TimedOp.deps、不看列表位置，故凡是
结构上必须等通信完成才能继续的下游 op，本模块必须显式把 deps 串上——否则强制串行的通信会被
误判为全可重叠（exposed_comm 系统性归零）。CP colossal 的 FA 块↔块间 p2p 一直如此；EP 的
router→dispatch→grouped→combine→下游、FSDP bwd 重 gather→bwd 首个 device op 现在也是
（code-review [0][12] 补齐；此前遗漏）。唯 FSDP fwd 段头 AG 本身仍 deps=()（可预取，等效
§5.5 预取语义的前提是 T1 DES 允许 comm_dp 跨段 run-ahead）。bwd 侧的对偶通信（gather↔
reduce-scatter 等）归 bwd_rules（Task 11），本模块只做 fwd（fsdp_regather 是例外，见其自身
docstring）。
"""
from __future__ import annotations

from dataclasses import replace

from .ir import TimedOp, TimedSegment, CommSpec, tensor_bytes, COMM_STREAM, STREAM_DEVICE

_GEMM_FAMILY = ("MatMul", "GroupedMatMul")


def inject_fsdp(seg: TimedSegment, dp_shard: int) -> TimedSegment:
    """dp_shard<=1 / 空段 / 段内无 GEMM 权重（full==0，无可 gather）→ 恒等；
    否则在段头插入本段权重的 all_gather（fwd 预取语义）。"""
    if dp_shard <= 1 or not seg.ops:
        return seg
    full = sum(
        tensor_bytes(op.in_shapes[1], op.dtype)
        for op in seg.ops
        if op.op_type in _GEMM_FAMILY and len(op.in_shapes) >= 2
    )
    if full == 0:
        return seg
    # ir.py CommSpec 约定：AG volume=分片入参字节（gather 前本 rank 持有的 1/dp_shard；
    # (n-1)·环系数归 op_cost/T1）。ceil≈FSDP flat-param pad 到整除
    volume = -(-full // dp_shard)
    gather_op = TimedOp(
        op_id=f"{seg.seg_id}.fsdp_ag", op_type="CommOp", phase=seg.ops[0].phase,
        in_shapes=(), out_shape=(), dtype="bf16",   # 段级聚合载荷，无单一张量 shape/dtype 可挂
        stream=COMM_STREAM["dp"], deps=(),
        comm=CommSpec("all_gather", volume, "dp", dp_shard))
    return TimedSegment(seg.seg_id, (gather_op,) + seg.ops)


def inject_ep(seg: TimedSegment, ep: int) -> TimedSegment:
    """ep<=1 或段内无 GroupedMatMul → 恒等；否则在 grouped 区（首..末个 GroupedMatMul）两侧
    各插一条 dispatch/combine all_to_all，并串成 router→disp→grouped→comb→下游（disp 继承
    grouped 区首 op 的原生产者 deps，grouped 首 op 追加等 disp，comb 等 grouped 区末 op，
    grouped 区之后引用末 op 的下游 deps 改指 comb——mirror producer.py Row.rs 的下游 redirect
    手法；code-review [0]：deps=() 会让 segment_sim 把这条强制串行的通信当成全可重叠）。
    同段多个不相邻 grouped 区会被单对 a2a 包裹（v1 简化；现源无此模式）。"""
    grouped_idx = [i for i, op in enumerate(seg.ops) if op.op_type == "GroupedMatMul"]
    if ep <= 1 or not grouped_idx:
        return seg
    first, last = grouped_idx[0], grouped_idx[-1]
    first_op, last_op = seg.ops[first], seg.ops[last]
    disp = TimedOp(
        op_id=f"{seg.seg_id}.ep_disp", op_type="CommOp", phase=first_op.phase,
        in_shapes=(first_op.in_shapes[0],), out_shape=first_op.in_shapes[0],
        dtype=first_op.dtype, stream=COMM_STREAM["ep"], deps=first_op.deps,
        comm=CommSpec("all_to_all", tensor_bytes(first_op.in_shapes[0], first_op.dtype),
                      "ep", ep))
    grouped = list(seg.ops[first:last + 1])
    grouped[0] = replace(grouped[0], deps=grouped[0].deps + (disp.op_id,))
    comb = TimedOp(
        op_id=f"{seg.seg_id}.ep_comb", op_type="CommOp", phase=last_op.phase,
        in_shapes=(last_op.out_shape,), out_shape=last_op.out_shape,
        dtype=last_op.dtype, stream=COMM_STREAM["ep"], deps=(last_op.op_id,),
        comm=CommSpec("all_to_all", tensor_bytes(last_op.out_shape, last_op.dtype),
                      "ep", ep))
    tail = tuple(replace(o, deps=tuple(comb.op_id if d == last_op.op_id else d for d in o.deps))
                 for o in seg.ops[last + 1:])
    ops = seg.ops[:first] + (disp,) + tuple(grouped) + (comb,) + tail
    return TimedSegment(seg.seg_id, ops)


def inject_cp(seg: TimedSegment, cp: int, method: str = "colossal") -> TimedSegment:
    """cp<=1 → 恒等。method="colossal" → 结构化为 cp 个 FlashAttention 块、块间 cp−1 条单跳
    kv p2p 交替（T1-4，overlap 交 DES 涌现）；method="ulysses" → 每条 FlashAttention 前后各插
    一条 all_to_all；其余 method → ValueError（fail-loud，且先于 cp<=1 早退——cp 不活跃时 typo
    也要报）。"""
    if method not in ("colossal", "ulysses"):
        raise ValueError(f"frame_comm: 未知 CP method {method!r}（fail-loud，不猜语义）")
    if cp <= 1:
        return seg

    ops: list[TimedOp] = []
    for op in seg.ops:
        if op.op_type != "FlashAttention":
            ops.append(op)
            continue
        if method == "colossal":
            if len(op.in_shapes) < 3:
                raise ValueError(
                    f"frame_comm: FlashAttention 期待 ≥3 个 in_shapes(q/k/v),"
                    f"got {len(op.in_shapes)} @ {op.src}——fail-loud(与 op_cost 同口径)")
            # ring 结构化（T1-4）：cp 个 FA 块 × 块间 cp-1 条单跳 kv p2p。每块 shape 与
            # localize 后的原 FA 相同（q=S/cp × kv 环转块=S/cp）；p2p deps=()（kv 段首可发，
            # 与上一块 FA 的重叠由 DES 涌现——spec §5.5 位置即语义）；末块保留原 op_id，
            # 下游 deps 不换绑。causal zigzag 负载均衡差异归 op_cost 的 causal 系数（v1 均匀）。
            kv_bytes = (tensor_bytes(op.in_shapes[1], op.dtype)
                        + tensor_bytes(op.in_shapes[2], op.dtype))
            for k in range(cp):
                if k > 0:
                    p2p = TimedOp(
                        op_id=f"{op.op_id}.p2p{k}", op_type="CommOp", phase=op.phase,
                        in_shapes=(op.in_shapes[1], op.in_shapes[2]),
                        out_shape=op.in_shapes[1], dtype=op.dtype,
                        stream=COMM_STREAM["cp"], deps=(),
                        src="cp[colossal]:module-semantics",
                        comm=CommSpec("p2p", kv_bytes, "cp", cp))
                    ops.append(p2p)
                    blk_deps = op.deps + (p2p.op_id,)
                else:
                    blk_deps = op.deps
                blk_id = op.op_id if k == cp - 1 else f"{op.op_id}.cp{k}"
                ops.append(TimedOp(
                    op_id=blk_id, op_type="FlashAttention", phase=op.phase,
                    in_shapes=op.in_shapes, out_shape=op.out_shape, dtype=op.dtype,
                    stream=op.stream, src=op.src, deps=blk_deps, module=op.module))
        else:   # ulysses：FA 前后各一条 all_to_all（seq-parallel↔head-parallel 切换）
            # 载荷只切 q,k,v（in_shapes[:3]，bprop_rules.py:31 口径）——真机 FA 调用含
            # attn_mask（bf16 S×S 可比肩 q），mask 不参与重排，全量求和会静默膨胀载荷。
            qkv = op.in_shapes[:3]
            in_volume = sum(tensor_bytes(s, op.dtype) for s in qkv)
            pre = TimedOp(
                op_id=f"{op.op_id}.cp_pre", op_type="CommOp", phase=op.phase,
                in_shapes=qkv, out_shape=op.in_shapes[0],
                dtype=op.dtype, stream=COMM_STREAM["cp"], deps=(),
                comm=CommSpec("all_to_all", in_volume, "cp", cp))
            post = TimedOp(
                op_id=f"{op.op_id}.cp_post", op_type="CommOp", phase=op.phase,
                in_shapes=(op.out_shape,), out_shape=op.out_shape,
                dtype=op.dtype, stream=COMM_STREAM["cp"], deps=(),
                comm=CommSpec("all_to_all", tensor_bytes(op.out_shape, op.dtype), "cp", cp))
            ops.append(pre)
            ops.append(op)
            ops.append(post)
    return TimedSegment(seg.seg_id, tuple(ops))


def fsdp_regather(bwd_seg: TimedSegment, fwd_seg: TimedSegment, dp_shard: int) -> TimedSegment:
    """ZeRO-3 bwd 权重重 gather（spec §3.3c「fwd 预取 + bwd 重 gather，随
    reshard_after_forward」；T0 交接要点3 裁决=注入，门面按 reshard!="never" 调用）。
    克隆 fwd 段的 .fsdp_ag 到 bwd 段头（fwd 无 gather / dp_shard<=1 → 恒等）。
    注：fwd AG 的 bwd 对偶（grad reduce-scatter）由 expand_bwd 自动产出且落 bwd 段尾
    （fwd 段头反转），本函数只补"重新拿回权重"这一条。
    AG 本身仍 deps=()（可预取），但 bwd 段第一个 device（计算）op 追加依赖该 AG——bwd 计算须
    等权重重 gather 完成；segment_sim 的 device 流 FIFO 会让该段后续 device op 自然排在其后，
    无需逐个显式挂 dep（code-review [12]：此前无任何 bwd op 依赖它，结构上恒可重叠）。"""
    if dp_shard <= 1:
        return bwd_seg
    src_ag = next((o for o in fwd_seg.ops
                   if o.op_type == "CommOp" and o.op_id.endswith(".fsdp_ag")), None)
    if src_ag is None:
        return bwd_seg
    ag = replace(src_ag, op_id=f"{bwd_seg.seg_id}.fsdp_ag", phase="bwd")
    new_ops, wired = [], False
    for o in bwd_seg.ops:
        if not wired and o.stream == STREAM_DEVICE:
            o = replace(o, deps=o.deps + (ag.op_id,))
            wired = True
        new_ops.append(o)
    return TimedSegment(bwd_seg.seg_id, (ag,) + tuple(new_ops))
