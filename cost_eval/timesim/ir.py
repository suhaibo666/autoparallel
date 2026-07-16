# cost_eval/timesim/ir.py
"""TimedOpSeq IR（spec §3.2）：timesim 的唯一输入数据结构 + op_flops 纯函数。

流标注（spec §5.1，cp 按"通信按 group 轴分道"补为独立道）：
  device | host_only | comm_tp | comm_cp | comm_ep | comm_dp | comm_pp
op_flops 只算 GEMM 族（6ND 不变量的主体）；FA/带宽类的时间成本由 op_cost（T1）经经验库给出，
不在 IR 层杜撰系数。"""
from __future__ import annotations

from dataclasses import dataclass

STREAM_DEVICE = "device"
STREAM_HOST_ONLY = "host_only"
COMM_STREAM = {"tp": "comm_tp", "cp": "comm_cp", "ep": "comm_ep",
               "dp": "comm_dp", "pp": "comm_pp"}


@dataclass(frozen=True)
class CommSpec:
    ctype: str            # all_reduce | reduce_scatter | all_gather | all_to_all | p2p
    volume_bytes: int     # 本 rank 载荷字节（(n-1)/n 等算法系数归 op_cost，T1）
    group_axis: str       # tp | cp | ep | dp | pp
    group_size: int


@dataclass(frozen=True)
class TimedOp:
    op_id: str                       # 回指 opdag 节点（"<cell>#<id>" / 展开后缀 ".bK"/".rK"）
    op_type: str                     # opdag 词表 + CommOp/…Grad
    phase: str                       # fwd | bwd | recomp
    in_shapes: tuple                 # tuple[tuple[int,...],...]，已代入 local
    out_shape: tuple
    dtype: str
    stream: str
    src: str = ""                    # mindformers file:line（源忠实）
    deps: tuple = ()                 # 跨流依赖的 op_id（同流 FIFO 隐含）
    comm: CommSpec | None = None
    module: str = ""                 # ColumnParallelLinear 等（shard/通信语义键）


@dataclass(frozen=True)
class TimedSegment:
    seg_id: str                      # "layer_3.fwd" / "embedding.fwd" / "loss.bwd" …
    ops: tuple


def op_flops(top: TimedOp) -> int:
    """GEMM 族 FLOPs = **2 · numel(A) · N_out**（A=in_shapes[0]，N_out=out_shape 末轴）。

    该式对 fwd（C=A·B：numel(A)=M·K）、bwd 的 dX=dy·Bᵀ（numel(dy)=M·N，N_out=K）、
    dW=Aᵀ·dy（numel(A)=M·K，N_out=N）**一致成立**——收缩维总在 numel(A) 里，无需分情况；
    GroupedMatMul 同式（expert 维在 numel(A) 里）。朴素 `k=in[0][-1]` 启发式对 dW 会取错
    收缩维（k 取成 K 而非 M），故弃用。其余 op 返回 0（见模块 docstring）。"""
    if top.op_type in ("MatMul", "GroupedMatMul") \
            and top.in_shapes and top.in_shapes[0] and top.out_shape:
        a = 1
        for d in top.in_shapes[0]:
            a *= d
        return 2 * a * top.out_shape[-1]
    return 0
