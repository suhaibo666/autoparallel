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
        # Muon:2D 矩阵权重（momentum-only）单独的有效倍数（同 offload 口径）。AdamW 时 matrix_* 分量
        #   == 非矩阵 → matrix_mult == mult → estimate_structure_memory 全 uniform、逐字节复现旧值。
        matrix_mult = 0
        if not offload_params:
            matrix_mult += opt.matrix_param_persist_bytes()
        if not offload_optimizer:
            matrix_mult += opt.matrix_optimizer_state_bytes()

        # dense（非专家）权重分母：grouped-FSDP 子域（Z3）——配了 dense_fsdp_shard_size 时用子域
        # （< 完整 fsdp → 每卡 dense 持久更大），否则完整 fsdp。`estimate_structure_memory` 对**非专家**
        # 权重统一用 `fsdp` 分母、对专家用 `efsdp`；故只需把 dense 分母作为 fsdp 传入即改到 dense、
        # experts 仍走原 efsdp（structure_mem 不改）。缺省 dense_fsdp_degree()==fsdp_degree() → 逐字节不变。
        fsdp = pm.dense_fsdp_degree()
        efsdp = pm.efsdp_degree()
        out: dict = {}

        for stage, layers in g.stages.items():
            if mult == 0 and matrix_mult == 0:
                out[stage] = 0                       # 全卸（含 fp32+offload_optimizer 的 param=0 情形）
                continue

            # 逐结构（层）组装 StructureMemory.persistent（含 fsdp/efsdp 切 + 有效倍数、按名
            # 去重），跨结构相加即该 stage 持久态。去重集中在 estimate_structure_memory 一处。
            out[stage] = sum(
                estimate_structure_memory(
                    layer.ops, fsdp=fsdp, efsdp=efsdp,
                    opt_state_bytes=mult,
                    matrix_opt_state_bytes=matrix_mult,   # Muon 2D 矩阵倍数（AdamW==mult → uniform）
                    alloc_block_bytes=alloc_block_bytes,
                    ep_degree=pm.degree("ep"),            # P0-3:expert wrap 不可整除 fail-loud 门
                ).persistent
                for layer in layers
            )

        return out

    def persistent_breakdown(self, g, opt, pm, cpu_offload: bool = False,
                             alloc_block_bytes: int = 1, *,
                             offload_params: bool = None,
                             offload_optimizer: bool = None) -> dict:
        """每 stage 持久态（persistent）**组成分解** —— 与 `compute` **完全同口径**（同 fsdp/efsdp
        分母、同 offload 语义、同 `estimate_structure_memory` 按名去重），把每卡持久字节拆成显式分量：

          persistent = 本卡驻留参数 P × 每参数持久字节
                     = P × [ 参数副本(compute-dtype) + master(fp32) + momentum + v(二阶动量) ]

        每分量每元素字节（优化器状态恒 fp32：master/m/v 各 4B）：
          - **参数副本** = `opt.param_persist_bytes()`（bf16 params=2；fp32=0，无独立 compute 副本）；
          - **master(fp32)** = 4；**momentum(m)** = 4；**v(二阶动量)** = 4。
          - **Muon**：2D 矩阵权重（`is_muon_matrix_weight`）momentum-only → **无 v**（v 仅计非矩阵参数）；
            AdamW：矩阵/非矩阵同口径（都含 v），矩阵计数折回非矩阵一并计 v。
          - **offload**：`offload_params`→参数副本归零；`offload_optimizer`→master/m/v 全零（与 `compute`
            的 `mult` 派生一致）。
          - **512B 块对齐残差**（DSv3=0，各张量本已对齐）单列为「块对齐」分量，保证
            **Σ 分量 == `compute()[stage]`**（逐字节相等）。

        返回 `{stage: {param_count, matrix_count, optimizer, offloaded, components, total_bytes}}`，
        `components=[[名称, 每元素字节, 该分量总字节], ...]`。
        """
        if offload_params is None:
            offload_params = cpu_offload
        if offload_optimizer is None:
            offload_optimizer = cpu_offload
        # 与 compute 同：喂给 estimate 的每元素倍数（含 offload 派生）——用它跑同一去重/切分/块对齐,
        # 既拿到 aligned persistent，又拿到驻留参数计数（matrix/other）。
        mult = matrix_mult = 0
        if not offload_params:
            mult += opt.param_persist_bytes()
            matrix_mult += opt.matrix_param_persist_bytes()
        if not offload_optimizer:
            mult += opt.optimizer_state_bytes()
            matrix_mult += opt.matrix_optimizer_state_bytes()
        fsdp = pm.dense_fsdp_degree()
        efsdp = pm.efsdp_degree()
        is_muon = str(getattr(opt, "type", "")).lower() == "muon"
        # 每分量每元素字节（offload 归零对应项）。优化器状态恒 fp32=4B。
        pc_b = 0 if offload_params else opt.param_persist_bytes()   # 参数副本(bf16=2/fp32=0)
        st_b = 0 if offload_optimizer else 4                        # master/m/v 各 4B(fp32)
        out: dict = {}
        for stage, layers in g.stages.items():
            n_mat = n_oth = aligned = 0
            for layer in layers:
                sm = estimate_structure_memory(
                    layer.ops, fsdp=fsdp, efsdp=efsdp,
                    opt_state_bytes=mult, matrix_opt_state_bytes=matrix_mult,
                    alloc_block_bytes=alloc_block_bytes,
                    ep_degree=pm.degree("ep"))            # P0-3 同口径
                n_mat += sm.persist_numel_matrix
                n_oth += sm.persist_numel_other
                aligned += sm.persistent
            n_total = n_mat + n_oth
            n_no_v = n_mat if is_muon else 0        # 无 v 的参数（Muon 2D 矩阵）；AdamW=0（矩阵折回）
            n_with_v = n_total - n_no_v
            pc_lbl = "参数副本(bf16)" if pc_b else "参数副本(fp32,无独立副本)"
            mom_lbl = "优化器 momentum" if is_muon else "优化器 m(一阶动量)"
            v_lbl = "优化器 v(二阶动量,仅非矩阵)" if is_muon else "优化器 v(二阶动量)"
            comps = [
                [pc_lbl, pc_b, n_total * pc_b],
                ["优化器 master(fp32)", st_b, n_total * st_b],
                [mom_lbl, st_b, n_total * st_b],
                [v_lbl, st_b, n_with_v * st_b],
            ]
            resid = aligned - sum(c[2] for c in comps)   # 512B 块对齐残差（DSv3=0）
            if resid > 0:
                comps.append(["块对齐(512B)", 0, resid])
            out[stage] = {
                "param_count": n_total, "matrix_count": (n_mat if is_muon else 0),
                "optimizer": ("Muon" if is_muon else "AdamW"),
                "offloaded": bool(offload_params or offload_optimizer),
                "components": comps, "total_bytes": aligned,
            }
        return out
