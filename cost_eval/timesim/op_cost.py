# cost_eval/timesim/op_cost.py
"""M8 op_cost（spec §4.1）：TimedOp → OpCost。

T1 = 三级退化的第 3 级（roofline × 默认 η，provenance="theory"，spec §4.2）；OpTimeLibrary
命中/内插（第 1/2 级）T2 接入——CostModel 预留 lib 参数位，传入即 NotImplementedError
（防静默假装标定过）。

FLOPs/bytes 公式按 op_type 内置（spec §4.1）：
  GEMM 族   = ir.op_flops（2·numel(A)·N_out，对 fwd/dX/dW 一致成立）；
  FA        = 2·B·N·Sq·Skv·(Dq+Dv)·causal系数（QKᵀ+PV 两个 batched GEMM；causal=0.5，
              ring zigzag 均衡差异并入该系数）——q/k/v 取 in_shapes[:3]，**约定提取序 4D
              (S,B,N,D)**（MLA 探针实证；非 4D fail-loud 不猜 layout）；Grad=2.5×（spec）；
  带宽类    = bytes_rw = Σ in/out tensor_bytes（空 shape=`?` 容忍位计 0）；
  View/host_only = 全 0（发射成本走 host 单价）；
  CommOp    = t_comm = α(axis) + moved/BW(axis)，moved 按 ir.py CommSpec volume 口径 ×
              ring 系数：AG(分片)×(n−1)、RS/AR(全量)×(n−1)/n（AR 再 ×2）、A2A×(n−1)/n
              （对分带宽口径）、p2p 恒单跳（T1-4 结构化后无跳数系数）。
bound/host_dominated 是 op 内禀静态属性；真 host-bound（device 空洞）只能由 segment_sim
涌现（spec §4.1「分类口径的关键区分」）。"""
from __future__ import annotations

from dataclasses import dataclass

from .ir import TimedOp, tensor_bytes, op_flops, STREAM_HOST_ONLY
from .machine import TimeHardware

_GEMM = ("MatMul", "GroupedMatMul")


@dataclass(frozen=True)
class OpCost:
    t_host_us: float
    t_dev_us: float
    t_comm_us: float
    flops: int
    bytes_rw: int
    arith_intensity: float
    ridge: float
    bound: str                # compute | memory | comm | host —— "host" 仅表示"该 op 无 device
                              # 分量、roofline 分类无意义"（View/host_only），**≠** spec §4.1-D2
                              # 讲的 segment_sim 涌现态 host-bound（那是"有 device 工作却被延迟"，
                              # 由段仿真判定，不是 op 静态属性）。勿把二者混为一谈（Task 6 review）。
    host_dominated: bool
    eta_key: str
    provenance: str           # T1 恒 "theory"（spec §4.2 第3级）


def _fa_flops(top: TimedOp, causal: bool) -> int:
    if len(top.in_shapes) < 3 or any(len(s) != 4 for s in top.in_shapes[:3]):
        raise ValueError(
            f"op_cost: FlashAttention 期待 4D q/k/v（提取序 (S,B,N,D)），"
            f"got {top.in_shapes[:3]} @ {top.src}——fail-loud 不猜 layout")
    q, k, v = top.in_shapes[:3]
    sq, b, n, dq = q
    skv, dv = k[0], v[3]
    coeff = 0.5 if causal else 1.0
    return int(2 * b * n * sq * skv * (dq + dv) * coeff)


class CostModel:
    def __init__(self, hw: TimeHardware, *, causal: bool = True, lib=None):
        if lib is not None:
            raise NotImplementedError("OpTimeLibrary=T2（spec §9）；T1 只有 theory 档")
        self.hw = hw
        self.causal = causal

    def _bytes_rw(self, top: TimedOp) -> int:
        total = sum(tensor_bytes(s, top.dtype) for s in top.in_shapes if s)
        if top.out_shape:
            total += tensor_bytes(top.out_shape, top.dtype)
        return total

    def comm_time_us(self, c) -> float:
        """CommSpec → us（公有：report 的步收尾 dp_replicate AR 等复用同一口径）。"""
        alpha, bw = self.hw.link(c.group_axis)
        n, v = c.group_size, c.volume_bytes
        if c.ctype == "all_gather":
            moved = v * (n - 1)
        elif c.ctype == "reduce_scatter":
            moved = v * (n - 1) / n
        elif c.ctype == "all_reduce":
            moved = 2 * v * (n - 1) / n
        elif c.ctype == "all_to_all":
            moved = v * (n - 1) / n
        elif c.ctype == "p2p":
            moved = v
        else:
            raise ValueError(f"op_cost: 未知 ctype {c.ctype!r}（fail-loud）")
        return alpha + moved / bw * 1e6

    def cost(self, top: TimedOp) -> OpCost:
        hw = self.hw
        t_host = hw.host_us(top.op_type, top.phase)
        bytes_rw = self._bytes_rw(top)
        ridge = hw.peak(top.dtype) / hw.hbm_bw

        if top.op_type == "CommOp":
            t_comm = self.comm_time_us(top.comm)
            return OpCost(t_host, 0.0, t_comm, 0, bytes_rw, 0.0, ridge, "comm",
                          t_host > t_comm, f"comm:{top.comm.ctype}", "theory")

        if top.stream == STREAM_HOST_ONLY or top.op_type == "View":
            return OpCost(t_host, 0.0, 0.0, 0, bytes_rw, 0.0, ridge, "host",
                          True, "host_only", "theory")

        if top.op_type in _GEMM:
            flops, eta_key = op_flops(top), "gemm"
        elif top.op_type == "FlashAttention":
            flops, eta_key = _fa_flops(top, self.causal), "fa"
        elif top.op_type == "FlashAttentionGrad":
            flops, eta_key = int(_fa_flops(top, self.causal) * 2.5), "fa"
        else:
            flops, eta_key = 0, "bw"

        if flops > 0:
            t_dev = flops / (hw.peak(top.dtype) * hw.eta[eta_key]) * 1e6
        else:
            t_dev = bytes_rw / (hw.hbm_bw * hw.eta["bw"]) * 1e6
        ai = flops / bytes_rw if bytes_rw else 0.0
        bound = "compute" if ai >= ridge else "memory"
        return OpCost(t_host, t_dev, 0.0, flops, bytes_rw, ai, ridge, bound,
                      t_host > t_dev, f"{eta_key}:{top.dtype}", "theory")


def price_segment(seg, cm: CostModel) -> dict:
    """整段定价：{op_id → OpCost}（segment_sim/report 的输入）。"""
    return {op.op_id: cm.cost(op) for op in seg.ops}
