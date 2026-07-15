"""M5：持久 param/opt 内存估算（**剔 grad**——P2-08 文档对齐 2026-07-14：梯度不是持久态，
是 step-scoped cumulative，由 mem_timeline 的 grad_accum 桶建模，P0-01）。

切分逻辑（严格遵循计划 Task 10）：
- M4（ShapeEval）已对图内参数施加 tp 或 ep 切分：
    dense 权重   local_numel = global_numel // tp
    expert 权重  local_numel = global_numel // ep
- M5 在此之上再除 FSDP 组：
    dense 权重   再 // dense_fsdp_degree()（缺省 = fsdp_degree() = dp_shard * cp；配了 grouped-FSDP
                 子域 dense_fsdp_shard_size 时 = 该子域 < 完整 fsdp → 每卡 dense 持久更大，Z3）
    expert 权重  再 // efsdp_degree()     （= dp_shard * cp * tp // ep，tp 已含其中；**不受子域影响**）
- **CPU offload（P1-19，2026-07-15）**：持久 = param 副本 + optimizer state 两分量。旧 `cpu_offload=True`
  一次卸掉整个持久态（=0）；现按 `offload_params` / `offload_optimizer` **分项**卸：
    - offload_params    → param 副本分量归零（bf16 每元素 2；fp32 无副本→本就 0）。
    - offload_optimizer → optimizer state 分量归零（master+m+v，每元素 12）。
  实现上取**有效每元素倍数** = 未卸分量之和，喂给同一 `estimate_structure_memory`（单一倍数 → 逐张量
  对齐口径与旧路径完全一致）：都不卸 → 倍数=state_bytes_per_param → **逐字节复现旧值**；都卸（含
  `cpu_offload=True` 派生）→ 倍数=0 → 0（== 旧全卸，且短路跳过 estimate 不触发整除校验，与旧
  `out[stage]=0; continue` 同行为）。
"""
from __future__ import annotations

from .structure_mem import estimate_structure_memory


class StaticMem:
    """M5：计算每 stage 每卡的持久内存字节（param + optimizer state，**剔 grad**——
    specs.py OptimizerSpec 的 state_bytes_per_param 同口径：AdamW fp32=12/bf16=14）。"""

    def compute(self, g, opt, pm, cpu_offload: bool = False, alloc_block_bytes: int = 1,
                *, offload_params: bool = None, offload_optimizer: bool = None) -> dict:
        """返回 {stage: bytes} 字典。

        参数
        ----
        g            : ResolvedGraph  — M4 输出，params.local_numel 已含图内切分。
        opt          : OptimizerSpec  — 含 state_bytes_per_param（AdamW fp32=12/bf16=14，剔 grad），
                       及 param_persist_bytes()/optimizer_state_bytes() 分量拆分（P1-19）。
        pm           : ParallelModel  — 提供 fsdp_degree() / efsdp_degree()。
        cpu_offload  : bool           — **向后兼容**入口：True 等价 offload_params 与 offload_optimizer
                       皆 True（该 stage 持久态全卸 → 0）。分离标志缺省（None）时由它推导。
        offload_params    : bool|None — 只卸 param 副本分量（None → 用 cpu_offload）。
        offload_optimizer : bool|None — 只卸 optimizer state 分量（None → 用 cpu_offload）。
        alloc_block_bytes : int       — 设备内存池分配对齐块（平台属性，默认 1=不取整）。
        """
        # 分离标志缺省（None）→ 从 cpu_offload 推导（全或无向后兼容）：旧调用方只传 cpu_offload，
        # 逐字节复现旧「全卸/全留」；report.py 显式传分离标志。
        if offload_params is None:
            offload_params = cpu_offload
        if offload_optimizer is None:
            offload_optimizer = cpu_offload
        # 有效每元素持久倍数 = 未卸分量之和（param 副本 + optimizer state，P1-19）。喂给同一
        # estimate_structure_memory 的单一 opt_state_bytes → 逐张量对齐口径与旧路径完全一致：
        #   - 都不卸：mult = param_persist + opt_state = state_bytes_per_param → **逐字节复现旧值**。
        #   - 都卸（mult==0）：短路 out[stage]=0（== 旧 cpu_offload 分支，不调 estimate、不触发整除校验）。
        mult = 0
        if not offload_params:
            mult += opt.param_persist_bytes()
        if not offload_optimizer:
            mult += opt.optimizer_state_bytes()

        # dense（非专家）权重分母：grouped-FSDP 子域（Z3）——配了 dense_fsdp_shard_size 时用子域
        # （< 完整 fsdp → 每卡 dense 持久更大），否则完整 fsdp。`estimate_structure_memory` 对**非专家**
        # 权重统一用 `fsdp` 分母、对专家用 `efsdp`；故只需把 dense 分母作为 fsdp 传入即改到 dense、
        # experts 仍走原 efsdp（structure_mem 不改）。缺省 dense_fsdp_degree()==fsdp_degree() → 逐字节不变。
        fsdp = pm.dense_fsdp_degree()
        efsdp = pm.efsdp_degree()
        out: dict = {}

        for stage, layers in g.stages.items():
            if mult == 0:
                out[stage] = 0                       # 全卸（含 fp32+offload_optimizer 的 param=0 情形）
                continue

            # 逐结构（层）组装 StructureMemory.persistent（含 fsdp/efsdp 切 + 有效倍数、按名
            # 去重），跨结构相加即该 stage 持久态。去重集中在 estimate_structure_memory 一处。
            out[stage] = sum(
                estimate_structure_memory(
                    layer.ops, fsdp=fsdp, efsdp=efsdp,
                    opt_state_bytes=mult,
                    alloc_block_bytes=alloc_block_bytes,
                ).persistent
                for layer in layers
            )

        return out
