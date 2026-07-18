# cost_eval/timesim/machine.py
"""时间侧硬件常数（spec §2.1 的 HardwareSpec 输入位）。

**为什么不进 specs.HardwareSpec**：那是内存侧命名空间（max_device_memory/framework_reserve），
解耦契约4「标定命名空间分离」——时间标定（峰值/带宽/η/host 单价）绝不回流内存侧，反之亦然。

单位约定：peak_flops = FLOP/s；hbm_bw / link bw = bytes/s；alpha = us；host 单价 = us。
DEFAULT_910B 全部是**厂商规格/公开口径占位**（calibrated=False，spec §7.2-1：跨机 α-β 未标定、
report 须标记）——T2 经验库标定后换库即换数，代码零改动（结构与常数分离铁律）。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeHardware:
    peak_flops: dict                 # dtype → FLOP/s（cube 峰值）
    hbm_bw: float                    # bytes/s
    link_alpha_us: dict              # 通信轴 → 启动时延 us（axis: tp/cp/ep/dp/pp）
    link_bw: dict                    # 通信轴 → bytes/s（v1 按轴给定；topo 推断=T2）
    eta: dict                        # {"gemm","fa","bw","opt"} → 默认效率（三级退化第3级）
    host_unit_us: dict               # {("*",phase): us} 相位缺省 + {(op_type,phase): us} 覆盖
    name: str = ""
    calibrated: bool = False

    def peak(self, dtype: str) -> float:
        return self.peak_flops.get(dtype) or self.peak_flops["bf16"]

    def link(self, axis: str) -> tuple[float, float]:
        return (self.link_alpha_us[axis], self.link_bw[axis])   # 未知轴 KeyError=fail-loud

    def host_us(self, op_type: str, phase: str) -> float:
        v = self.host_unit_us.get((op_type, phase))
        return v if v is not None else self.host_unit_us[("*", phase)]


def synth_hw(**over) -> TimeHardware:
    """测试用合成硬件：整数好算（peak=100 TFLOPS、HBM=1 TB/s、链路=0.1 TB/s、η 全 1），
    断言可零公差。生产勿用。"""
    base = dict(
        peak_flops={"bf16": 100e12, "fp16": 100e12, "fp32": 25e12},
        hbm_bw=1e12,
        link_alpha_us={a: 1.0 for a in ("tp", "cp", "ep", "dp", "pp")},
        link_bw={a: 1e11 for a in ("tp", "cp", "ep", "dp", "pp")},
        eta={"gemm": 1.0, "fa": 1.0, "bw": 1.0, "opt": 1.0},
        host_unit_us={("*", "fwd"): 1.0, ("*", "bwd"): 2.0, ("*", "recomp"): 1.0,
                      ("View", "fwd"): 1.0, ("MatMul", "bwd"): 2.0},
        name="synth", calibrated=False,
    )
    base.update(over)
    return TimeHardware(**base)


# 910B 占位（**未标定**，厂商规格/公开口径；T2 标定覆盖）。数值仅用于「未标定相对排序」模式
# （spec D1 退化档），不进任何测试断言。
DEFAULT_910B = TimeHardware(
    peak_flops={"bf16": 376e12, "fp16": 376e12, "fp32": 94e12},
    hbm_bw=1.6e12,
    link_alpha_us={"tp": 8.0, "cp": 8.0, "ep": 10.0, "dp": 10.0, "pp": 15.0},
    link_bw={"tp": 2.8e11, "cp": 2.8e11, "ep": 2.5e10, "dp": 2.5e10, "pp": 2.5e10},
    eta={"gemm": 0.7, "fa": 0.45, "bw": 0.8, "opt": 0.8},   # spec §4.2 三级退化默认档
    host_unit_us={("*", "fwd"): 3.0, ("*", "bwd"): 1.8, ("*", "recomp"): 3.0},
    name="910B", calibrated=False,
)
