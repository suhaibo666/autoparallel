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
    """运行时可行性**约束矩阵**（closure-audit C1，2026-07-15；P1-16 完整化 2026-07-15）：返回
    **真机跑不起来 / 会静默退化成另一份配置**的组合的错误串列表（空=可行）。集中一处，供
    Evaluator/adapter/搜索器统一调用（搜索器接受一个组合前先判它真机能不能跑，无需先建模型）。

    **分工边界**：本函数只吃 `(pc, optimizer, swap)`——即只判**并行/调度组合**的可跑性；结构合法性
    （维度正值/整除、pp≤可切分层数、S%cp、tp%heads、layers_per_stage 段数/和）需要模型维度，留在
    `build_llm._validate_structure` / `ParallelModel.__init__`，**此处不重复**。ParallelConfig 标量
    正值/枚举（degree≥1、interleave≥1、reshard 枚举等）已在 `ParallelConfig.__post_init__` 拦，故此处
    入参已是「各字段自洽」的 pc，只需判**字段间**的并行组合约束。

    约束矩阵（逐条 mindformers pynative 源码依据）：
    - **[并行区一致性] ep 须整除 dp_shard·cp·tp**（`parallel_dims.py:140-148`）：EP 在单个
      dp_replicate 组内的 `(dp_shard·cp·tp)` 区里切专家，`efsdp = dp_shard·cp·tp // ep`；ep 不整除该区
      → efsdp 静默 floor 到错值（ep 超区时为 0），真机 sparse mesh 尺寸不匹配、跑不起来。
      注：world = dp_replicate·dp_shard·cp·tp·pp 一致性在评估器里**由构造保证**（report:evaluate 用
      各度之积当 world，`parallel_dims._validate:127` 的等式恒成立），无需另判。
    - **[VPP] interleave>1 须 pp>1**：VPP（交错式 1F1B / 虚拟流水）是**流水线**上的技术——mindformers
      仅在 `pp_enabled`（pp>1）时走 `apply_pp`（`parallelize.py:1805`），且 interleaved 调度要求
      `interleave_num>1`（`pipeline_parallel.py:367-372`）；pp=1 时 interleave 被静默丢弃
      （评估器 `parallel_model.py:76` `pp<=1 → 单 stage`），即"配了 VPP 却评了个非交错单 stage"。
    - **[VPP] num_microbatches ≥ pp（交错 warmup 深度）**：交错式 1F1B 的 warmup =
      `(pp-rank-1)*2 + (v-1)*group_size`（Megatron `schedules.py:877-878`，group_size 默认=pp；见
      `mem_timeline.interleaved_warmup`），要填满 v·pp 个虚拟 stage 需要足够微批；m<pp 时交错流水无法
      建立稳态（Megatron 交错调度要求 num_microbatches 是 pp 的正倍数）。**仅 VPP 施加**：plain 1F1B
      （interleave=1）的 warmup=`min(pp-1-rank, m)`（`mem_timeline.build_1f1b`，clamp 到 m）——m<pp 真机
      可跑（只是流水气泡），故不拦，避免误伤合法 plain 配置。
    - **[swap] PP>1 + activation swap 不支持**（`activation_checkpoint.py:898-900` 直接 raise）。
    - **[swap] swap 开启时 default_prefetch ≥ 1**（`config.py:300-307` `prefetch<1` → raise）：预取深度
      <1 无双缓冲窗、真机报错。（上界 `<num_layers` 需模型层数，留结构侧/ParallelModel 校验。）
    - **[tp] tp>1 强制 sequence_parallel=True**（`config.py:471-477` 直接 raise）——SP=false 时真机不
      物化序列切分，评估器算的是另一份（无 SP）激活，属"评了个跑不了的配置"。
    - **[优化器] 仅建模 Adam/AdamW**（K_OPT/state_bytes 均自 AdamW op 链导出）；非 Adam 的持久态与
      optstep 瞬态都不同 → fail-loud，不静默按 AdamW 近似（P1-19）。
    """
    errs = []
    # [并行区一致性] ep | dp_shard·cp·tp（parallel_dims.py:140-148）。
    region = pc.dp_shard * pc.cp * pc.tp
    if region % pc.ep != 0:
        errs.append(
            f"expert_parallel(ep={pc.ep}) 必须整除 dp_shard·cp·tp={region}"
            f"（dp_shard={pc.dp_shard}·cp={pc.cp}·tp={pc.tp}）——EP 在该并行区内切专家"
            f"（efsdp=region//ep，parallel_dims.py:140-148）；{region} 不被 {pc.ep} 整除会令 efsdp "
            f"静默 floor 到错值（ep 超区时为 0），真机 sparse mesh 尺寸不匹配、跑不起来。合法 ep：能整除 "
            f"{region} 的正整数（≤{region}）。")
    # [VPP] interleave>1 须 pp>1（单 stage 交错无意义、会被静默丢弃）。
    if pc.interleave > 1 and pc.pp <= 1:
        errs.append(
            f"interleave(VPP)={pc.interleave}>1 但 pp={pc.pp}≤1：VPP 是流水线上的交错技术，pp=1 无流水线"
            "——mindformers 仅在 pp>1 走 apply_pp（parallelize.py:1805），pp=1 时 interleave 被静默丢弃"
            "（评估器 parallel_model.py:76 pp≤1→单 stage），等于评了个非交错单 stage 配置。合法：VPP 需 "
            "pp≥2，或设 interleave=1。")
    # [VPP] num_microbatches ≥ pp（交错 warmup 深度；仅 VPP 施加，plain 1F1B 的 m<pp 由 warmup clamp 兜底可跑）。
    if pc.interleave > 1 and pc.pp > 1 and pc.num_microbatches < pc.pp:
        errs.append(
            f"VPP(interleave={pc.interleave}) 下 num_microbatches={pc.num_microbatches} < pp={pc.pp}："
            "交错式 1F1B 需足够微批填满 v·pp 个虚拟 stage 的 warmup（Megatron schedules.py:877-878；"
            "num_microbatches 应为 pp 的正倍数），m<pp 无法建立交错稳态。合法：num_microbatches≥pp"
            f"（≥{pc.pp}）。（plain 1F1B interleave=1 时 m<pp 合法、warmup 会 clamp，故本约束仅对 VPP。）")
    # [swap] PP>1 + activation swap 不支持（activation_checkpoint.py:898-900）。
    if pc.pp > 1 and getattr(swap, "enable", False):
        errs.append("PP>1 + activation swap：mindformers 不支持该组合"
                    "（activation_checkpoint.py:898-900 直接 raise；tests/test_swap_offload.py）"
                    "——真机跑不起来，拒绝评估。")
    # [swap] swap 开启时 default_prefetch ≥ 1（config.py:300-307）。
    if getattr(swap, "enable", False) and getattr(swap, "default_prefetch", 1) < 1:
        errs.append(
            f"swap.enable=True 但 default_prefetch={getattr(swap, 'default_prefetch', None)} < 1："
            "预取深度须 ≥1 才有双缓冲预取窗（config.py:300-307 `prefetch<1` 直接 raise）——<1 真机报错。"
            "合法 default_prefetch：≥1 的整数（且 <num_layers，上界由结构侧校验）。")
    # [tp] tp>1 强制 sequence_parallel=True（config.py:471-477）。
    if pc.tp > 1 and not pc.sequence_parallel:
        errs.append("tensor_parallel>1 强制 sequence_parallel=True（config.py:471-477）——"
                    "SP=false 时评的是跑不起来的无 SP 配置，拒绝评估（如确需绕过用 "
                    "Evaluator(..., check_feasibility=False)）。")
    # [优化器] 仅 Adam/AdamW 已建模（P1-19）。
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
        # mode 枚举 fail-loud（F4，closure-audit v2 §F4，2026-07-15）：合法「不重算」取值只有
        # {None, "None", "none"}（RecomputeSpec 默认 mode="None"，specs.py:116；is_full/is_select
        # 仅在 mode=="full"/"select" 触发）。其它任何串（typo "ful"/"selct"/"Full"）此前直接 return →
        # 静默等效不重算、峰值与 RecomputeSpec("None") 逐字节相同（探针实证）。现拒；合法 None 放行。
        if mode in (None, "None", "none"):
            return
        raise ValueError(
            f"RecomputeSpec.mode={mode!r} 非法 mode——合法取值 None / 'None' / 'none'（不重算）、"
            "'full'、'select'；疑似 typo（如 'ful'/'selct'/'Full'），此前会静默等效不重算"
            "（与配置意图相反）。")
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
        # 空/纯空白 selector fail-loud（F4，closure-audit v2 §F4，2026-07-15）：空串 "" 是所有 op
        # 名/类型的子串，op_matches 会**意外命中整层**（真机同样不生效，是配置错误而非「重算整层」——
        # 重算整层应显式 mode='full'）。此前静默接受、看似只选几个 op 实则整层重算。
        if any(not s.strip() for s in sels):
            raise ValueError(
                f"select 层 {lid} 的选择器集含空串/纯空白 {sorted(sels)}——空串是所有 op 名/类型的"
                "子串、会命中整层（配置错误；重算整层请显式 mode='full'）。")
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
        """该 stage 的 reserved 池估计 = allocated 峰值 + HCCL 通信缓冲 + allocator pool 碎片。

        **口径演进（P2-01 §F7 闭环，2026-07-15）**：此前只加 HCCL、是 reserved 的纯**下界**；现补
        `framework.allocator_pool_fragmentation`（DynamicMemPoolBestFit best-fit 空洞 + mempool 预留块
        尾部，碎片率 1.8%×allocated），令估计**更接近真实 reserved 上界**、**不再是纯下界**。真机 DSv4：
        allocated 15415.5 + HCCL ~400 + pool ~277 ≈ 16093 MiB，落在真机 reserved 16092-16096 区间内。

        ⚠ pool 碎片模型**自 DSv4 单点标定、是近似**（非严格上界；跨模型碎片率稳定性待验证），故本值仍是
        近似而非可证上界——见 `framework.POOL_FRAGMENTATION_RATE` 标定注记。
        """
        from .framework import allocator_pool_fragmentation
        peak = self.per_stage[stage].peak_bytes
        pool = allocator_pool_fragmentation(None, peak)   # DynamicMemPoolBestFit 块级碎片近似
        return peak + self.hccl_reserved_bytes + pool

    @property
    def allocated_oom(self) -> bool:
        """allocated 口径 OOM（= `.oom`，max_memory_allocated > 容量）。"""
        return self.oom

    @property
    def reserved_oom(self) -> bool:
        """reserved 口径超容判定（P2-01，§F7 pool 碎片闭环，2026-07-15）：任一 stage 的
        `reserved_estimate_bytes`（= allocated 峰值 + HCCL 缓冲 + allocator pool 碎片近似）> 设备容量。

        **语义演进**：此前 reserved 估计只含 allocated + HCCL、是纯**下界**，`reserved_oom=True` 曾是
        「确定超容」；现估计已补 allocator pool 碎片近似分量（对应真机 DSv4 实测的 ~277-281 MiB pool
        分量），**不再是纯下界**，而是更接近真实 reserved 的近似判定。代价：pool 碎片是**单点标定近似**
        （非严格上界、跨模型稳定性待验证），故 `reserved_oom` 两侧都是近似——`reserved_oom=False` 仍
        **不保证**真实 reserved 不超容，`reserved_oom=True` 也可能因高估而偏保守。设备 HBM 真实约束是
        reserved，故 `.oom=False`（allocated 未超）时 reserved 仍可能已超——两口径**分开报告**，调用方
        不应把单一 allocated 布尔当最终 OOM 结论。max_device_memory=0（未提供）时恒 False。字段名
        allocated_oom/reserved_oom 保持不变（下游依赖）。"""
        if not self.max_device_memory:
            return False
        return any(self.reserved_estimate_bytes(i) > self.max_device_memory
                   for i in range(len(self.per_stage)))


class Evaluator:
    """离线并行策略代价评估器门面（P0：内存）。"""

    def __init__(self, model_spec, parallel_config, optimizer, hardware,
                 recompute, swap, *, check_feasibility: bool = True,
                 validate_opdag: bool = False, opdag_strict: bool = False):
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
        # P2-02（2026-07-15）：opdag 一致性交叉校验钩子——把 `cost_eval/opdag/` 从**离线工具**升级为
        # **生产链路的可选约束**。开启后 evaluate() 会用 opdag 从真 mindformers 源抽出的重算子名册，
        # 交叉校验手写 LayerSpec 的对应层段（MLA 注意力 / MoE 专家 grouped-GEMM），漂移即 warn
        # （opdag_strict=True 则 fail）。**默认关（validate_opdag=False）→ 评估行为逐字节不变**、不接触
        # opdag（连 import 都惰性），不破 12 锚点 / 全部测试。缺 mindformers 源时静默跳过（不 fail）。
        self.validate_opdag = validate_opdag
        self.opdag_strict = opdag_strict

    def evaluate(self, record_timeline: bool = False) -> PeakMemoryReport:
        """执行全链路评估，返回 PeakMemoryReport。

        record_timeline=True 时，每个 StagePeak.timeline 记录全事件内存序列（内存曲线）。
        """
        # P2-02：opdag 一致性交叉校验（默认关，旁路——不改变下方任何评估计算）。开启时先跑校验，
        # 让 opdag 从源码抽出的 op 名册真正**约束**手写 LayerSpec（漂移 warn / strict fail）。lazy import
        # 保证默认路径不接触 opdag 模块。
        if self.validate_opdag:
            from .opdag.crosscheck import validate_against_opdag
            validate_against_opdag(self.spec, strict=self.opdag_strict)
        world = (self.pc.dp_replicate * self.pc.dp_shard * self.pc.cp
                 * self.pc.tp * self.pc.pp)
        pm = ParallelModel(self.pc, self.spec.dims.n_layers, world)
        g = ShapeEval().resolve(self.spec, pm)
        _validate_recompute_against_graph(self.recompute, g)     # C1：full 空集/select 零命中 fail-loud
        # 分配器块对齐（平台属性 HardwareSpec.alloc_block_bytes，默认 512）：逐张量 roundup —
        # 「分配器碎片」项的公式化落地（framework_reserve「块对齐取整」分量，取代经验常数）。
        block = getattr(self.hw, "alloc_block_bytes", 1)
        # P1-19：持久态按 param/optimizer **分离**卸载（ParallelConfig 已把 cpu_offload=True 派生为
        # 三标志全 True → 逐字节复现旧全卸；缺省全 False → 逐字节复现旧全留）。grad 卸载不影响持久态
        # （梯度非持久，见 mem_timeline 的 grad_accum 桶随 offload_grads 归零）。
        persistent = StaticMem().compute(
            g, self.opt, pm, alloc_block_bytes=block,
            offload_params=self.pc.offload_params,
            offload_optimizer=self.pc.offload_optimizer)
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
            kept_frag_factor=getattr(self.spec.dims, "kept_frag_factor", 0.0),
            nr_moe_frag_factor=getattr(self.spec.dims, "nr_moe_frag_factor", 0.0))
        per_stage = [peaks[s] for s in sorted(peaks)]
        tightest = max(per_stage, key=lambda p: p.peak_bytes).stage
        # D-2：HCCL 通信缓冲（reserved 池，按启用的通信域数估计；不进 allocated 峰值）→ 接入报告。
        hccl = hccl_reserved_buffer(self.pc)
        return PeakMemoryReport(per_stage, tightest,
                                any(p.oom for p in per_stage), hccl_reserved_bytes=hccl,
                                max_device_memory=self.hw.max_device_memory)   # P2-01：reserved 口径判定
