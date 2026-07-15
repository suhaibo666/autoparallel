"""M7：组装峰值显存报告 + Evaluator 门面。

Evaluator 是整个评估器的对外入口：接收 ModelSpec + 并行/优化器/硬件/重算/swap
配置，依次调用 M3→M4→M5→M6，返回 PeakMemoryReport。
"""
from __future__ import annotations
from dataclasses import dataclass

from .parallel_model import ParallelModel
from .shape_eval import ShapeEval
from .static_mem import StaticMem
from .mem_timeline import MemTimeline, StagePeak
from .framework import framework_reserve, hccl_reserved_buffer


def feasibility_errors(pc, optimizer, swap) -> list:
    """运行时可行性检查（closure-audit C1，2026-07-15）：返回**真机跑不起来**的组合的错误串列表
    （空=可行）。集中一处，供 Evaluator/adapter/搜索器统一调用，不各处零散封口。

    依据均为 mindformers pynative 运行时硬约束：
    - PP>1 + activation swap 不支持（tests/test_swap_offload.py 源码核对）。
    - tp>1 强制 sequence_parallel=True（config.py:471-477）——SP=false 时真机不物化序列切分，
      当前评估器算的是另一份（无 SP）激活，属"评了个跑不了的配置"。
    - 优化器仅建模 Adam/AdamW（K_OPT/state_bytes 均自 AdamW op 链导出）；非 Adam 的持久态
      与 optstep 瞬态都不同 → fail-loud，不静默按 AdamW 近似（P1-19）。
    """
    errs = []
    if pc.pp > 1 and getattr(swap, "enable", False):
        errs.append("PP>1 + activation swap：mindformers 不支持该组合"
                    "（tests/test_swap_offload.py）——真机跑不起来，拒绝评估。")
    if pc.tp > 1 and not pc.sequence_parallel:
        errs.append("tensor_parallel>1 强制 sequence_parallel=True（config.py:471-477）——"
                    "SP=false 时评的是跑不起来的无 SP 配置，拒绝评估（如确需绕过用 "
                    "Evaluator(..., check_feasibility=False)）。")
    otype = str(getattr(optimizer, "type", "AdamW")).lower()
    if otype not in ("adamw", "adam"):
        errs.append(f"optimizer.type={getattr(optimizer, 'type', None)!r} 未建模："
                    "K_OPT/optstep 瞬态与 persistent 均自 AdamW 导出，非 Adam 会全错——"
                    "请用 OptimizerSpec.adamw() 或补对应优化器建模。")
    return errs


def _validate_recompute_against_graph(recompute, g) -> None:
    """针对已解析 op 图校验重算配置（closure-audit C1，2026-07-15；越界/空转封口 P1-02，2026-07-15）
    ——需要图才能判定，故在 evaluate 时做（非 RecomputeSpec 构造时）。核心入口统一 fail-loud，不只
    UI 封口。**合法层号范围 = 实际 resolved 图的 layer_id 并集**：这里的 `g` 是**全图**（evaluate 里
    `ShapeEval().resolve` 遍历整个 layer_pattern、按 `stage_of` 分组，`g.stages` 含所有 stage/层），
    故并集 = 全模型 layer id（0=embedding、1..N decoder、head、mtp）——即便 pp>1 也不会误判某 stage
    图不含的层为越界。评估器 select/full 口径用 1..N decoder（+1 偏移，见 from_mindformers `_build_recompute`）。
    捕获的空转反例（此前静默接受、与无重算逐字节相同）：
    - mode=full 但 full_layers 空 → 等效不重算、与配置意图相反。
    - mode=full 但 full_layers 含**越界层号**（不在图 layer id 集）→ 等效不重算。
    - mode=select 但**无任何层配非空选择器**（select_ops 全空 / 全 set()）→ 等效 None、配置空转。
    - mode=select 某层号**越界**（不在图 layer id 集）→ 等效不重算（此前对 by_id 缺失直接 continue）。
    - mode=select 某层选择器在该层 op 图**零命中** → 静默空转（层仍标 select、kept_frag 生效，「越错越贵」）。
    """
    mode = getattr(recompute, "mode", "None")
    if mode not in ("full", "select"):
        return
    layers = [l for lys in g.stages.values() for l in lys]
    valid_ids = {l.layer_id for l in layers}

    def _range_hint() -> str:
        if not valid_ids:
            return "（当前图无层）"
        return f"（合法层号 {min(valid_ids)}..{max(valid_ids)}，含 embedding/head/mtp 伪层）"

    if mode == "full":
        if not recompute.full_layers:
            raise ValueError(
                "RecomputeSpec(mode='full') 但 full_layers 为空——等效不重算、与配置意图相反，"
                "请显式给层号（不重算请用 mode='None'）。")
        oob = sorted(l for l in recompute.full_layers if l not in valid_ids)
        if oob:
            raise ValueError(
                f"RecomputeSpec(mode='full') full_layers 含越界层号 {oob}——不在图 layer id 集内、"
                f"等效不重算 {_range_hint()}。")
        return

    # mode == "select"
    by_id = {l.layer_id: l for l in layers}
    if not any(sels for sels in recompute.select_ops.values()):
        raise ValueError(
            "RecomputeSpec(mode='select') 但没有任何层配置非空选择器（select_ops 全空 / 全 set()）"
            "——等效不重算、配置空转，请显式给 {layer_id: {op 选择器}}（不重算请用 mode='None'）。")
    for lid, sels in sorted(recompute.select_ops.items()):
        if not sels:
            continue
        layer = by_id.get(lid)
        if layer is None:
            raise ValueError(
                f"select 选择器 {sorted(sels)} 配在越界层号 {lid}——不在图 layer id 集内、"
                f"等效不重算 {_range_hint()}。")
        hit = any(recompute.op_matches(lid, op.name, getattr(op.type, "value", op.type))
                  for op in layer.ops)
        if not hit:
            raise ValueError(
                f"select 选择器 {sorted(sels)} 在层 {lid}({layer.layer_type}) 的 op 图零命中——"
                f"静默空转会错算（真机同样不生效）。该层可用 op: "
                f"{', '.join(op.name for op in layer.ops)}")


@dataclass(frozen=True)
class PeakMemoryReport:
    """各 PP stage 峰值显存报告。

    `peak_bytes`/`oom` 是 **allocated 峰值**（max_memory_allocated，OOM 主判据，真机验证）。
    **P2-01 口径声明（2026-07-14）**：`.oom` 只回答 allocated 口径；设备真实容量约束是 reserved
    （真机实测 reserved − allocated ≈ 658-680 MiB，review_evidence_2026-07-14.md）——调用方做
    容量临界判定时应同时核查 `reserved_estimate_bytes(stage)`，勿把单一布尔当最终 OOM 结论。
    `hccl_reserved_bytes`（D-2）是 **reserved 池**的 HCCL 通信缓冲估计（按通信域数，`framework.
    hccl_reserved_buffer`）——**不进 allocated 峰值**（ep=2 真机证实），但计入 `reserved 估计`：
    `reserved ≈ allocated_peak + hccl_reserved (+ 池碎片)`。设备 HBM 的真实约束是 reserved，
    故给出 `reserved_estimate_bytes(stage)` 供 reserved 口径的 OOM 余量核查。
    """
    per_stage: list        # list[StagePeak]，按 stage 升序
    tightest_stage: int    # peak_bytes 最大的 stage
    oom: bool              # 任意 stage OOM（allocated 口径）——= allocated_oom，保留旧名兼容
    hccl_reserved_bytes: int = 0   # D-2：HCCL 通信缓冲（reserved 池，按通信域数；world-level 同值）
    max_device_memory: int = 0     # P2-01（C4）：设备容量，供 reserved 口径 OOM 判定

    def reserved_estimate_bytes(self, stage: int) -> int:
        """该 stage 的 reserved 池估计 = allocated 峰值 + HCCL 通信缓冲。

        **口径边界（P2-01 文档订正，closure-audit 2026-07-15）**：当前**只加 HCCL 通信缓冲**，
        **未含** allocator pool 碎片。真机 reserved − allocated ≈ 658-680 MiB 里，除 HCCL 外还有
        分配器池碎片（当前未建模）——故本值是 reserved 的**下界近似**，不是精确上界。
        """
        return self.per_stage[stage].peak_bytes + self.hccl_reserved_bytes

    @property
    def allocated_oom(self) -> bool:
        """allocated 口径 OOM（= `.oom`，max_memory_allocated > 容量）。"""
        return self.oom

    @property
    def reserved_oom(self) -> bool:
        """reserved 口径超容（P2-01，closure-audit C4，2026-07-15）：任一 stage 的 reserved 估计
        （allocated + HCCL 缓冲 + 池碎片近似）> 设备容量。设备 HBM 真实约束是 reserved，故
        `.oom=False`（allocated 未超）时 reserved 仍可能已超阈值——两口径**分开报告**，调用方
        不应把单一 allocated 布尔当最终 OOM 结论。max_device_memory=0（未提供）时恒 False。"""
        if not self.max_device_memory:
            return False
        return any(self.reserved_estimate_bytes(i) > self.max_device_memory
                   for i in range(len(self.per_stage)))


class Evaluator:
    """离线并行策略代价评估器门面（P0：内存）。"""

    def __init__(self, model_spec, parallel_config, optimizer, hardware,
                 recompute, swap, *, check_feasibility: bool = True):
        # closure-audit C1（2026-07-15）：运行时可行性守卫集中在**核心评估入口** Evaluator，
        # 不只在 adapter/UI 封口（此前直接核心 API 仍接受 tp>1+SP=false、非 Adam）。
        # check_feasibility=False 供纯内存建模场景显式绕过（如只想要某不可跑组合的字节数）。
        if check_feasibility:
            for msg in feasibility_errors(parallel_config, optimizer, swap):
                raise ValueError(msg)
        self.spec = model_spec
        self.pc = parallel_config
        self.opt = optimizer
        self.hw = hardware
        self.recompute = recompute
        self.swap = swap

    def evaluate(self, record_timeline: bool = False) -> PeakMemoryReport:
        """执行全链路评估，返回 PeakMemoryReport。

        record_timeline=True 时，每个 StagePeak.timeline 记录全事件内存序列（内存曲线）。
        """
        world = (self.pc.dp_replicate * self.pc.dp_shard * self.pc.cp
                 * self.pc.tp * self.pc.pp)
        pm = ParallelModel(self.pc, self.spec.dims.n_layers, world)
        g = ShapeEval().resolve(self.spec, pm)
        _validate_recompute_against_graph(self.recompute, g)     # C1：full 空集/select 零命中 fail-loud
        # 分配器块对齐（平台属性 HardwareSpec.alloc_block_bytes，默认 512）：逐张量 roundup —
        # 「分配器碎片」项的公式化落地（framework_reserve「块对齐取整」分量，取代经验常数）。
        block = getattr(self.hw, "alloc_block_bytes", 1)
        persistent = StaticMem().compute(g, self.opt, pm, self.pc.cpu_offload, alloc_block_bytes=block)
        # framework_reserve 现默认 0（生产）：框架瞬态已按机理拆进 op 图（FSDP 预取→gather_buf、
        # flash-ws→flash workspace、MoE staging→dispatch/combine workspace）+ 分配器对齐→上面的
        # 逐张量 roundup。hw.framework_reserve 仅审计/回归旋钮（显式给值复现旧经验常数，如 golden 177）。
        fr = framework_reserve(self.pc, self.hw.framework_reserve)
        peaks = MemTimeline().simulate(
            g, self.recompute, self.swap, pm, persistent,
            fr, self.hw.max_device_memory,
            grad_dtype_bytes=getattr(self.opt, "grad_dtype_bytes", 4),
            record_timeline=record_timeline, alloc_block_bytes=block,
            cross_entropy_fused=getattr(self.spec.dims, "cross_entropy_fused", False),
            norm_compute_dtype_bytes=getattr(self.spec.dims, "norm_compute_dtype_bytes", 0),
            kept_frag_factor=getattr(self.spec.dims, "kept_frag_factor", 0.0))
        per_stage = [peaks[s] for s in sorted(peaks)]
        tightest = max(per_stage, key=lambda p: p.peak_bytes).stage
        # D-2：HCCL 通信缓冲（reserved 池，按启用的通信域数估计；不进 allocated 峰值）→ 接入报告。
        hccl = hccl_reserved_buffer(self.pc)
        return PeakMemoryReport(per_stage, tightest,
                                any(p.oom for p in per_stage), hccl_reserved_bytes=hccl,
                                max_device_memory=self.hw.max_device_memory)   # P2-01：reserved 口径判定
