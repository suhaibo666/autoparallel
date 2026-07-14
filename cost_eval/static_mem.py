"""M5：持久 param/opt 内存估算（**剔 grad**——P2-08 文档对齐 2026-07-14：梯度不是持久态，
是 step-scoped cumulative，由 mem_timeline 的 grad_accum 桶建模，P0-01）。

切分逻辑（严格遵循计划 Task 10）：
- M4（ShapeEval）已对图内参数施加 tp 或 ep 切分：
    dense 权重   local_numel = global_numel // tp
    expert 权重  local_numel = global_numel // ep
- M5 在此之上再除 FSDP 组：
    dense 权重   再 // fsdp_degree()      （= dp_shard * cp）
    expert 权重  再 // efsdp_degree()     （= dp_shard * cp * tp // ep，tp 已含其中）
- cpu_offload=True 时该 stage 持久态 = 0。
"""
from __future__ import annotations

from .structure_mem import estimate_structure_memory


class StaticMem:
    """M5：计算每 stage 每卡的持久内存字节（param + optimizer state，**剔 grad**——
    specs.py OptimizerSpec 的 state_bytes_per_param 同口径：AdamW fp32=12/bf16=14）。"""

    def compute(self, g, opt, pm, cpu_offload: bool, alloc_block_bytes: int = 1) -> dict:
        """返回 {stage: bytes} 字典。

        参数
        ----
        g            : ResolvedGraph  — M4 输出，params.local_numel 已含图内切分。
        opt          : OptimizerSpec  — 含 state_bytes_per_param（AdamW fp32=12/bf16=14，剔 grad）。
        pm           : ParallelModel  — 提供 fsdp_degree() / efsdp_degree()。
        cpu_offload  : bool           — True 则该 stage 持久态全部卸载 CPU，返回 0。
        alloc_block_bytes : int       — 设备内存池分配对齐块（平台属性，默认 1=不取整）。
        """
        fsdp = pm.fsdp_degree()
        efsdp = pm.efsdp_degree()
        out: dict = {}

        for stage, layers in g.stages.items():
            if cpu_offload:
                out[stage] = 0
                continue

            # 逐结构（层）组装 StructureMemory.persistent（含 fsdp/efsdp 切 + opt 倍数、按名
            # 去重），跨结构相加即该 stage 持久态。去重集中在 estimate_structure_memory 一处。
            out[stage] = sum(
                estimate_structure_memory(
                    layer.ops, fsdp=fsdp, efsdp=efsdp,
                    opt_state_bytes=opt.state_bytes_per_param,
                    alloc_block_bytes=alloc_block_bytes,
                ).persistent
                for layer in layers
            )

        return out
