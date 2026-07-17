"""bwd 展开规则库（spec §3.3d/e）：opdag bprop_rules 的姊妹件——per-op-type 的教科书事实，
把一段 fwd TimedSegment 逆拓扑序展开为对应的 bwd TimedOp 序列（可选 recompute 前缀）。

规则表（按 op_type 分派，见 `_bwd_of`）：
  MatMul/GroupedMatMul → dX + dW 两条 GEMM（各自 flops ≈ fwd 一次，故 bwd 总 flops = 2× fwd；
                          op_flops 的 in_shapes[0]=激活侧收缩维 惯例保证该式对 dX/dW 都成立，
                          见 ir.py op_flops docstring）；无权重操作数（len(in_shapes)<2，
                          module=="" 透传 matmul 的已知形态，ir.py in_shapes 字段注释）时
                          fail-loud（ValueError），不猜权重形状。
  CommOp            → 对偶通信（ctype 互换 + volume 按 ir.py CommSpec 类 docstring 的
                          **对偶换算规则**缩放：RS(全量)→AG(分片) 需 ÷group_size，
                          AG(分片)→RS(全量) 需 ×group_size；AR/A2A/p2p 对偶 volume 不变）。
  View/host_only    → 单条 host_only View（形状转置，无设备算力/带宽成本）。
  FlashAttention    → 单条 FlashAttentionGrad（内核不拆 dQ/dK/dV，T1 经验库整体标定）。
  其余（Norm/Activation/Elementwise/Cast/Gather…) → 单条 "<op_type>Grad"（带宽类，op_flops
                          对其返回 0，成本走 T1 经验库，不在此杜撰系数）。

bwd deps = **fwd 依赖边反转**（spec §5.1 跨流依赖只来自 TimedOp.deps，无"等前序列表项"规则——
deps=() 会使 bwd exposed comm 结构性归零）：fwd edge P→C 反转后，C 的输入梯度生产者恒为
`C.op_id+".b0"`（dX 惯例居 .b0），故 P 的**全部** bwd 产物（含 dW .b1——dW 也要 dy，与真实
autodiff 一致）deps 都取其 fwd 消费者的 .b0 集；同流冗余项无害（ir.py deps 字段注释）。

recompute（spec §2.2-3 契约）：recompute="full" 时，在 bwd 序之前插入 phase="recomp" 前缀——
按 fwd 顺序重放原 ops（非逆序，因为 recompute 只是"重新跑一遍 fwd"）；recomp_comm=False（默认，
对应 mindformers recompute_comm=False 的常见配置）时前缀跳过 CommOp，不重放通信——这是从"内存
侧"独立实现的语义，不依赖 mindformers 侧 recompute 具体代码路径（解耦契约，测试用例覆盖两种
取值）。前缀 deps **段内 remap**：被重放的生产者 dep 追加 ".r"（重算 AG→matmul 的串行化保留）；
未重放的（被剔 CommOp）drop——其输出 fwd 已保存（这正是它不重算的原因），段首即可用无需等待；
producer 只发段内 deps，drop 规则无外部误伤。

bwd host 单价（host_only 流上 bwd op 的时间成本）：本模块只产出 IR 节点，不做计时；T1 op_cost
经验库再对 bwd/recomp 的 host_only 单价分相标定（fwd/bwd 可能不同，此处不预设）。"""
from __future__ import annotations

from dataclasses import replace

from .ir import TimedOp, TimedSegment, CommSpec, STREAM_HOST_ONLY

_DUAL = {"all_reduce": "all_reduce", "reduce_scatter": "all_gather",
         "all_gather": "reduce_scatter", "all_to_all": "all_to_all", "p2p": "p2p"}


def _dual_comm(c: CommSpec) -> CommSpec:
    """对偶通信（volume 按 ir.py CommSpec 换算规则缩放，勿直接复制）。"""
    ctype = _DUAL[c.ctype]
    vol = c.volume_bytes
    if c.ctype == "reduce_scatter":      # RS(全量) → AG(分片)
        vol = vol // c.group_size
    elif c.ctype == "all_gather":        # AG(分片) → RS(全量)
        vol = vol * c.group_size
    return CommSpec(ctype, vol, c.group_axis, c.group_size)


def _bwd_of(o: TimedOp, deps: tuple[str, ...] = ()) -> list[TimedOp]:
    b = dict(phase="bwd", deps=deps, src=o.src, dtype=o.dtype, module=o.module)
    if o.op_type == "CommOp":
        return [TimedOp(op_id=o.op_id + ".b0", op_type="CommOp",
                        in_shapes=(o.out_shape,), out_shape=o.in_shapes[0] if o.in_shapes else (),
                        stream=o.stream, comm=_dual_comm(o.comm), **b)]
    if o.stream == STREAM_HOST_ONLY or o.op_type == "View":
        return [TimedOp(op_id=o.op_id + ".b0", op_type="View",
                        in_shapes=(o.out_shape,), out_shape=o.in_shapes[0] if o.in_shapes else (),
                        stream=STREAM_HOST_ONLY, **b)]
    if o.op_type in ("MatMul", "GroupedMatMul"):
        if len(o.in_shapes) < 2:
            raise ValueError(
                f"bwd_rules: {o.op_type}@{o.src} 仅 {len(o.in_shapes)} 入（无权重操作数），"
                f"dX/dW 展开无据——fail-loud（ir.py in_shapes 元数警示）")
        x, w = o.in_shapes[0], o.in_shapes[1]
        dx = TimedOp(op_id=o.op_id + ".b0", op_type=o.op_type,
                     in_shapes=(o.out_shape, w), out_shape=x, stream=o.stream, **b)
        dw = TimedOp(op_id=o.op_id + ".b1", op_type=o.op_type,
                     in_shapes=(x, o.out_shape), out_shape=w, stream=o.stream, **b)
        return [dx, dw]
    if o.op_type == "FlashAttention":
        return [TimedOp(op_id=o.op_id + ".b0", op_type="FlashAttentionGrad",
                        in_shapes=o.in_shapes, out_shape=o.out_shape, stream=o.stream, **b)]
    return [TimedOp(op_id=o.op_id + ".b0", op_type=o.op_type + "Grad",
                    in_shapes=(o.out_shape,) + o.in_shapes, out_shape=o.out_shape,
                    stream=o.stream, **b)]


def expand_bwd(seg: TimedSegment, *, recompute: str | None = None,
               recomp_comm: bool = False) -> TimedSegment:
    """fwd 段 → bwd 段（可选 recompute 前缀）。seg_id 的 .fwd 后缀替换为 .bwd。"""
    prefix: list[TimedOp] = []
    if recompute == "full":
        replayed: set[str] = set()   # F2：dep 段内 remap——重放者 +".r"，被剔 CommOp drop
        for o in seg.ops:
            if o.op_type == "CommOp" and not recomp_comm:
                continue
            prefix.append(replace(o, op_id=o.op_id + ".r", phase="recomp",
                                  deps=tuple(d + ".r" for d in o.deps if d in replayed)))
            replayed.add(o.op_id)
    # F1：fwd 依赖边反转——P 的 bwd deps = 其 fwd 消费者 C 的输入梯度生产者 C.op_id+".b0" 集
    rev: dict[str, tuple[str, ...]] = {}
    for o in seg.ops:
        for d in o.deps:
            rev[d] = rev.get(d, ()) + (o.op_id + ".b0",)
    bwd: list[TimedOp] = []
    for o in reversed(seg.ops):
        bwd.extend(_bwd_of(o, deps=rev.get(o.op_id, ())))
    return TimedSegment(seg.seg_id.replace(".fwd", "") + ".bwd", tuple(prefix + bwd))
