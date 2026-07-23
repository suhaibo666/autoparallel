"""M6：事件驱动内存时间线仿真（1F1B 调度 + 桶式峰值追踪）。"""
from __future__ import annotations
from dataclasses import dataclass, field

from .structure_mem import estimate_structure_memory, estimate_select_memory
from .specs import is_muon_matrix_weight, is_attn_projection, _MUON_NS_WORKSPACE_MULT


# ---------------------------------------------------------------------------
# 1F1B/VPP 调度代数已搬家至 cost_eval/schedule.py（spec 2026-07-16 §2.2 契约2：
# 时间/内存两仿真器共享的中立调度模块）。此处 re-export 保持全部旧 import 路径兼容。
# ---------------------------------------------------------------------------
from .schedule import (                                    # noqa: F401
    Event, _1f1b_from_warmup, build_1f1b, interleaved_warmup,
    build_interleaved_1f1b, get_schedule_table, interleaved_virtual_order,
    chunk_layer_ids,
)


# ---------------------------------------------------------------------------
# Task 12: Buckets / MemBreakdown / StagePeak + helpers + MemTimeline
# ---------------------------------------------------------------------------

@dataclass
class Buckets:
    """7 桶内存状态（每桶均为字节数，瞬时值）。"""
    persistent: int = 0       # param + optimizer state（持久态，剔 grad §8.4）
    act_live: int = 0         # 当前存活的 saved activations
    gather_buf: int = 0       # FSDP all-gather 缓冲（当前层整层权重 + 预取下 depth 层双缓冲、reshard 后释）
    grad_buf: int = 0         # 参数梯度缓冲（BWD 一层的瞬时峰值）
    recomp_scratch: int = 0   # full 重算层反向重跑 forward 的 max-live（§8.5②，扣 checkpoint 输入）
    bwd_scratch: int = 0      # 反向临时物化（如 loss probs fp32 / mHC sinkhorn grad）
    bwd_working_set: int = 0  # 无重算层反向工作集（激活梯度 dL/dact，= forward_max_live − bwd_scratch，§8.5②）
    swap_buf: int = 0         # 激活 swap H2D 预取缓冲（BWD：复原当前卸载层 saves + 反向序后 depth 层在飞预取窗，双缓冲）
    workspace: int = 0        # 算子 workspace（FWD 逐层临时）
    optstep: int = 0          # 优化器-step 瞬态（②，真机 profiler）：AdamW 更新最大权重时物化的 k_opt 个 fp32 [weight] 临时（Square/sqrt/m̂/update；grad 已拆去 grad_accum 桶，P0-01 重标 K_OPT 6→4）；step 在反向后、激活已释，故与激活互斥
    kept_frag: int = 0        # **标定 margin**（非 op 图导出）：保留(非重算)模块 loss 峰的 fp32-cast 横切 + 小张量长尾（313 个 <100MiB 碎片，源码级 op-DAG 提取证实其在 op 图粒度之下，见 opdag_validation.md）。仅 loss-BWD 事件、按 kept 激活比例计；full 重算 kept=0→此项 0（锚点不破）。**两作用域共用此桶**：① select-kept-MoE（kept_frag_factor×kept_act）；② 无重算-MoE（D1，nr_moe_frag_factor×_nr_moe_act，仅 pp==1 单 stage 无重算 loss-BWD）——同族碎片、不同 gate，互斥不双算
    grad_accum: int = 0       # **已规约梯度累计驻留**（P0-01，2026-07-14）：真机证实 step-scoped cumulative——每层首次反向后其 reduced 本地分片常驻，至 optimizer 后 zero_grad 释放（两卡探针 1889.5 MiB 吻合）。与 grad_buf（当前层 reduce-scatter 前 full 瞬态）正交
    p2p_buf: int = 0          # **PP stage 间 P2P send 激活缓冲**（P1-15，Task A）：非末 stage 前向把本 stage 输出激活 [S,B,H] send 给下 stage，send 通信期间驻留（1 份；overlap_p2p 时 2 份双缓冲）。recv 侧（非首 stage 首层输入）已隐含在 act_live 首层 pin → 不双算。pp=1 恒 0。仅 FWD 事件驻留、BWD/optstep 清零（pp2 峰在 BWD，不移锚点）
    mtp_resident: int = 0     # **MTP loss 链 per-微批步内驻留**（2026-07-23，185 pp4+MTP 锚点）：mtp 层的 loss 段激活（h_last/h_final/logits/logsm）每微批前向后**不随该微批反向释放**、驻留至 step 末（真机 mtp_1_loss 逐步聚合;185 实测 mtp=1 仅尾 stage 净增 +16.4GB ≈ m×3.1GB,前 stage 逐 MiB 不变）。仅 **full-recompute 的 mtp 层** 计入（该层 saved 已塌缩到 ci+ctx,无双算;无重算 mtp 的同款驻留未有锚点,见 gate 注释）。optstep 前清零

    def total(self) -> int:
        return (self.persistent + self.act_live + self.gather_buf + self.grad_buf
                + self.recomp_scratch + self.bwd_scratch + self.bwd_working_set
                + self.swap_buf + self.workspace + self.optstep + self.kept_frag
                + self.grad_accum + self.p2p_buf + self.mtp_resident)


@dataclass(frozen=True)
class MemBreakdown:
    """峰值时刻各桶的快照。"""
    persistent: int
    act_live: int
    gather_buf: int
    grad_buf: int
    recomp_scratch: int
    bwd_scratch: int
    bwd_working_set: int
    swap_buf: int
    workspace: int
    optstep: int
    framework: int
    kept_frag: int = 0
    grad_accum: int = 0       # P0-01 尾部追加（default=0，序不破，照 kept_frag 先例）
    p2p_buf: int = 0          # P1-15 尾部追加（default=0，序不破，照 grad_accum 先例）
    mtp_resident: int = 0     # 2026-07-23 尾部追加（default=0,序不破）:MTP loss 链步内驻留


@dataclass(frozen=True)
class TimelineSample:
    """内存时间线上一个采样点（一次 rec() = 一个事件）。

    P2-06 结构化身份（closure-audit C4，2026-07-15）：`(event, mb, chunk)` 三元组唯一标识
    VPP 下的重复事件——此前 `(event, mb)` 在 v>1 时会重复（同 mb 不同 chunk），无法与 profiler
    的 (microbatch, model-chunk) 序列对齐。`chunk` = VPP 虚拟模型块 id（v=1 时恒 -1）。
    """
    idx: int              # 事件序号（0 起）
    event: str            # 事件标签（如 fwd:3 / fwd_end / bwd@5；VPP 下带 #c<chunk>）
    total_bytes: int      # 该事件时刻总占用（Σ桶 + framework_reserve）
    breakdown: MemBreakdown
    mb: int = -1          # microbatch 序号（optstep 等非微批事件 = -1）
    chunk: int = -1       # VPP 虚拟模型块 id（v=1 或非微批事件 = -1）


@dataclass(frozen=True)
class StagePeak:
    """单 stage 仿真结果。"""
    stage: int
    peak_bytes: int
    breakdown: MemBreakdown
    peak_event: str
    oom: bool
    timeline: tuple = ()   # record_timeline=True 时为 tuple[TimelineSample]（全事件序列），否则空
    peak_mb: int = -1      # 峰值事件的 microbatch 序号（P2-06；非微批事件 = -1）


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

# 以下 `_layer_*` 均**组装** `estimate_structure_memory`（structure_mem.py）取对应桶，
# 不再各自裸 walk raw ops——去重（saves/params 按名）与 rollup 逻辑集中在那一处。

def _layer_saves_bytes(layer) -> int:
    """该层 saves 去重后总字节（全量保存时 pin 进 act_live）。

    去重语义（设计 §2.1/§7.1「Σ去重」）见 `estimate_structure_memory`：`attn` 被
    `flash`（saves=[qkv,attn,lse]）与 `o_proj`（saves=[attn]）双 save（attention.py GQA
    :69/:73、MLA :149/:153）只算一次（C1）。"""
    return estimate_structure_memory(layer.ops).activation_saves


def _checkpoint_input_bytes(layer) -> int:
    """full 重算时仅保留层输入：第一个有 saves 的 op 的首个 save。"""
    return estimate_structure_memory(layer.ops).checkpoint_input


def _layer_param_bytes(layer) -> int:
    """该层 full-unsharded 参数（compute dtype）——FSDP all-gather 缓冲（按名去重）。"""
    return estimate_structure_memory(layer.ops).param_full_bytes


def _layer_expert_param_bytes(layer, blk: int = 1) -> int:
    """该层去重后 **专家权重** full-unsharded 字节（compute dtype，逐权重块对齐）。

    P1-14（Task B）：`layer.mlp.experts` 是**独立 FSDP 单元**（efsdp mesh，忠实 mindformers
    pynative `base_models/gpt/parallelize.py:1108-1112` `fully_shard(layer.mlp.experts, **efsdp_config)`
    ——"small module first"，与整层 wrap `:1155-1161`「large module」分开），其 all-gather/reshard
    生命周期 ≠ 整层：experts 在层中段（dispatch 前）才 gather、combine 后 reshard，**不在 attn 段
    驻留**。此函数取该层专家权重字节（= `param_full_bytes` 的专家分量），供前向 gather 两段拆分。
    非 MoE 层返回 0。与 `estimate_structure_memory.param_full_bytes` 逐权重块对齐口径一致（按名去重）。"""
    def _align(n: int) -> int:                # 与 structure_mem 逐张量块对齐同口径（自含，不依赖其私有符号）
        return ((n + blk - 1) // blk) * blk if blk > 1 else n
    params: dict = {}
    for op in layer.ops:
        for w in op.params:
            params[w.name] = w
    return sum(_align(w.local_numel * w.dtype_bytes)
               for w in params.values() if getattr(w, "is_expert", False))


def _layer_grad_bytes(layer, grad_dtype_bytes: int) -> int:
    """反向 reduce-scatter 前的 full-unsharded 梯度（按 grad dtype，按名去重）。"""
    return estimate_structure_memory(
        layer.ops, grad_dtype_bytes=grad_dtype_bytes).grad_full_bytes


def _layer_bwd_scratch(layer) -> int:
    """该层各 op 反向临时物化之和（如 loss probs fp32）。"""
    return estimate_structure_memory(layer.ops).bwd_scratch


def _layer_workspace(layer) -> int:
    """该层各 op workspace_bytes 的最大值（FWD 逐层临时占用）。"""
    return estimate_structure_memory(layer.ops).workspace


def _prefetch_param_bytes(order: list, idx: int, depth: int, sm_by_id: dict) -> int:
    """FSDP2 参数预取双缓冲：执行序 `order` 中位置 idx 之后 `depth` 个单元的
    full-unsharded 参数字节之和（compute dtype）——与当前层共存的预取缓冲。

    忠实 PyTorch FSDP2 默认 **depth-1**（`_fsdp_param_group.py:854-856`
    `_backward_prefetch` → `target = post_forward_order[curr_index - 1]`，恰回退 1
    个单元；`:854` 的 `elif curr_index > 0` → 反向最后一个单元不预取。前向隐式
    depth-1：`wait_for_unshard:486-495` + Note:61-70 当前 all-gather 输出保留至下一
    copy-in）。mindformers pynative 显式同深度（`parallelize.py:245-273`
    set_modules_to_{forward,backward}_prefetch，每层→相邻一层）。

    - depth=1：恰取执行序下一个单元的 param_full；边界（末单元 idx+1 越界）→ 求和为
      0（不预取）。
    - depth=0：范围空 → 0 → 复现旧单缓冲（回归路径）。
    - reshard_after_forward=default（非 root 用完即 reshard，`_fsdp_state.py:203-207`）
      → 与当前层共存的仅这 depth 个预取单元，**不累积**。
    """
    total = 0
    for j in range(idx + 1, min(idx + 1 + depth, len(order))):
        total += sm_by_id[order[j]].param_full_bytes
    return total


def _prefetch_swap_bytes(order: list, idx: int, depth: int, offloaded: set, sm_by_id: dict) -> int:
    """激活 swap 反向 H2D 预取窗（双缓冲）：反向执行序 `order` 中位置 idx 之后 `depth` 个单元里
    **被卸载的层**（`offloaded` 集）的 activation_saves 之和——反向到当前层时，其后 depth 层的
    激活正被预取回 HBM（在飞/已驻留），与当前层激活共存。

    忠实 mindformers pynative `apply_swap`：预取对 `(layer_id, layer_id+prefetch)`
    （`activation_checkpoint.py:807/:812`）经 `SwapManager().set_forward_prefetch_layer`
    （`:814-815`）——处理"更后一层"(反向更早、id 更大)时触发"更前一层"的 H2D swap-in；反向到某层
    时，其 swap-in 已在 `prefetch` 步前触发 → 驻留。故反向序里位置 idx 之后 `depth` 个单元 = 在飞
    预取窗。depth 来自 `SwapSpec.default_prefetch`（`config.py:826` 默认 1，校验 ≥1；本库额外允许
    depth=0 = 仅复原当前层、无预取窗）。

    - `offloaded` = 实际卸载的层集（`swap.swaps(lid) and not recompute.is_full(lid)`，与 FWD
      `saved=0` 条件一致）；非卸载层激活走 act_live/recomp_scratch，不进此窗（三态互斥，去双算）。
    - 边界（idx+1.. 越界）→ 求和为 0（反向末端无更多可预取）。
    - swap 关（`offloaded` 空）→ 求和 0（回归路径，逐字节复现旧行为）。
    """
    total = 0
    for j in range(idx + 1, min(idx + 1 + depth, len(order))):
        lid = order[j]
        if lid in offloaded:
            total += sm_by_id[lid].activation_saves
    return total


# ---------------------------------------------------------------------------
# MemTimeline
# ---------------------------------------------------------------------------

class MemTimeline:
    """事件驱动峰值仿真器（M6）。

    建模要点：
    - gather_buf 在 FWD/BWD 逐层 = 当前层 full-unsharded 权重 + **预取下 depth 层双缓冲**
      （FSDP2 参数预取：`_prefetch_param_bytes`，depth 来自 ParallelConfig.prefetch_depth，
      默认 1；depth=0 复现旧单缓冲），reshard 后即释（reshard_after_forward=default）。
    - 激活 swap（`SwapSpec`，忠实 mindformers pynative `apply_swap` / PyTorch `save_on_cpu`）：
      FWD 被 swap 层 saves 卸载到 CPU（D2H）→ `saved=0`，前向→反向间隙不驻留 act_live；
      BWD 反向前 H2D 取回 → `swap_buf` = 复原当前层 saves +（反向序后 depth 层的）在飞预取窗
      （`_prefetch_swap_bytes`，depth 来自 `SwapSpec.default_prefetch`，与 FSDP 参数预取同构的
      双缓冲）。swap 关（enable=False）→ swap_buf 恒 0，逐字节复现旧行为。
    """

    def simulate(self, g, recompute, swap, pm, static_persistent: dict,
                 framework_reserve: int, max_device_memory: int,
                 grad_dtype_bytes: int = 4, record_timeline: bool = False,
                 alloc_block_bytes: int = 1, cross_entropy_fused: bool = False,
                 ce_pynative_lean: bool = False,
                 norm_compute_dtype_bytes: int = 0,
                 kept_frag_factor: float = 0.0,
                 nr_moe_frag_factor: float = 0.0,
                 bwd_scratch_conservative: bool = False,
                 muon: bool = False, muon_per_head: bool = False,
                 muon_n_heads: int = 0) -> dict:
        """仿真各 stage 峰值。

        参数
        ----
        g                  : ResolvedGraph
        recompute          : RecomputeSpec
        swap               : SwapSpec
        pm                 : ParallelModel
        static_persistent  : dict[stage, int]  — M5 StaticMem.compute() 输出
        framework_reserve  : int  — 框架常驻开销（字节）
        max_device_memory  : int  — 设备容量上限（字节）
        record_timeline    : bool — True 则每个事件都记进 StagePeak.timeline（内存曲线）

        返回
        ----
        dict[stage, StagePeak]
        """
        res: dict = {}
        pp = pm.degree("pp")
        m = pm.pc.num_microbatches
        # 虚拟流水（VPP）级数 V：来自 ParallelConfig.interleave（默认 1 = plain 1F1B，
        # build_interleaved_1f1b 对 v<=1 逐事件复现 build_1f1b → anchors/golden 不动）。
        # V>1 → 交错式更深 warmup（Megatron `schedules.py:877-878`），更多在飞微批激活。
        v = getattr(pm.pc, "interleave", 1)
        # FSDP2 参数预取深度（config 驱动，非魔法常数）：默认 1 = FSDP2 默认双缓冲
        # （survey：PyTorch _fsdp_param_group.py:854-856 / mindformers parallelize.py:245-273）。
        # depth=0 → 复现旧单缓冲（回归路径）。
        depth = getattr(pm.pc, "prefetch_depth", 1)
        # 激活 swap 反向 H2D 预取深度（config 驱动，非魔法常数）：来自 SwapSpec.default_prefetch
        # （survey：mindformers config.py:826 默认 1 = 双缓冲；activation_checkpoint.py:807/:814）。
        # swap 关时 swaps() 恒 False → 下方 swap_buf 恒 0（与预取深度无关）。
        swap_depth = getattr(swap, "default_prefetch", 1)
        # CPU offload 分离标志（P1-19，2026-07-15）：optstep 分量随 offload_optimizer、grad_accum 桶随
        # offload_grads 分别归零（旧单布尔 cpu_offload 一次卸全部）。ParallelConfig.__post_init__ 已保证
        # cpu_offload=True → 三标志全 True，故 fallback 到 cpu_offload 令旧构造的 pc 也逐字节复现旧行为。
        offload_optimizer = getattr(pm.pc, "offload_optimizer", pm.pc.cpu_offload)
        offload_grads = getattr(pm.pc, "offload_grads", pm.pc.cpu_offload)

        # dense 权重分母用 dense_fsdp_degree()（grouped-FSDP 子域，Z3；缺省==fsdp_degree() 逐字节不变）
        # —— grad_accum 累计梯度分片与 optstep 的 max_w//fsdp_d 都随子域变（配子域时每卡 dense 梯度/
        # 优化器瞬态更大，与 static_mem 持久态同口径，修 Z3 flag 的 timeline OOM 欠估）。experts 走 efsdp。
        fsdp_d, efsdp_d = pm.dense_fsdp_degree(), pm.efsdp_degree()
        for stage, layers in g.stages.items():
            layer_ids = [l.layer_id for l in layers]
            by_id = {l.layer_id: l for l in layers}
            # 每层的 StructureMemory rollup（模块化组装，单点去重）——预算一次，事件循环直取各桶。
            # fsdp/efsdp 传入供 grad_shard_bytes（P0-01 累计梯度分片，divisor 与 persistent 同口径）。
            sm_by_id = {
                l.layer_id: estimate_structure_memory(
                    l.ops, grad_dtype_bytes=grad_dtype_bytes,
                    fsdp=fsdp_d, efsdp=efsdp_d,
                    alloc_block_bytes=alloc_block_bytes,
                    norm_compute_dtype_bytes=norm_compute_dtype_bytes,
                    bwd_scratch_conservative=bwd_scratch_conservative)   # F10 双模式
                for l in layers
            }
            # 选择性重算：每层按选择器（op 名/类型子串）把 op 划分为选中/非选中，预算三桶
            # （act_live_pinned / recomp_scratch / bwd_working_set，复用 forward_max_live 机理，
            # 单点去重）。仅对 is_select 的层预算；full/None 层走既有路径（字节级不变）。
            select_mem_by_id = {
                l.layer_id: estimate_select_memory(
                    l.ops,
                    (lambda op, _lid=l.layer_id: recompute.op_matches(
                        _lid, op.name, getattr(op.type, "value", op.type))),
                    alloc_block_bytes=alloc_block_bytes,
                    norm_compute_dtype_bytes=norm_compute_dtype_bytes)
                for l in layers if recompute.is_select(l.layer_id)
            }
            # 实际卸载到 CPU 的层集（三态互斥：recompute 优先于 swap，与 FWD `saved=0` 分支同条件）。
            # 这些层 saves 前向不驻留 act_live、反向经 swap_buf 复原（H2D 取回）。full 与 select
            # 均属「重算态」，不进卸载集。
            offloaded = {
                lid for lid in layer_ids
                if swap.swaps(lid)
                and not recompute.is_full(lid) and not recompute.is_select(lid)
            }

            # ① unfused 交叉熵链（真机 profiler，pp=2 8L 无重算 stage1）：**无重算**下 loss 层反向峰
            #   同时挂着 ~8 份满 vocab fp32 中间量（log_softmax+NLL op 链：cast/sub/exp/log/Neg/
            #   ScatterAdd/ZerosLike/probs），而估计器只建 3 份（logsm saved + probs+grad）。**全/选择
            #   重算**下真机只 ~3 份共存（峰在链尾、早期中间量已释，cp2 profiler 4L 见证）→ 仅无重算 fat。
            #   loss 层由 `nll` op 认定（唯一满 vocab bwd_scratch）。**fused CE（DSv4 生产）精简、不 fat**
            #   → `cross_entropy_fused` 时 loss_lids 空（DSv4-align 0.930 不动）。
            loss_lids = (set() if cross_entropy_fused else
                         {l.layer_id for l in layers
                          if any(getattr(op, "name", "") == "nll" for op in l.ops)})
            # 本 stage 是否完全无重算（K_CE fat 判据,review P0.1）:该 stage 任一层被 full/select
            # 重算则 lean（真机:重算模式下 CE 链早期中间量已释,cp2 profiler 见证）。
            _stage_no_recompute = not any(
                recompute.is_full(l.layer_id) or recompute.is_select(l.layer_id)
                for l in layers)
            # D1（2026-07-16）：本 stage 含 MoE 专家核（grouped-GEMM `moe_gemm` op）的层集——
            #   无重算-MoE margin 的作用层（识别方式与 crosscheck `_moe_expert_window` 一致，取
            #   op.type 归一化字符串 == "moe_gemm"；resolved 图里 op.type 为纯字符串，故用 getattr 兜底）。
            _moe_lids = {l.layer_id for l in layers
                         if any(getattr(op.type, "value", op.type) == "moe_gemm"
                                for op in l.ops)}
            # ── MTP loss 链步内驻留（2026-07-23，185 pp4+MTP 锚点:尾 stage +16.4GB≈m×3.1GB）──
            # mtp 层 loss 段（final_norm/lm_head/logsoftmax/nll 的 saves = h_last/h_final/
            # logits_lm/logsm）每微批前向后**驻留至 step 末**（真机 mtp_k_loss 逐步聚合,其反向图
            # 跨微批持有——185 实测 mtp=1 只尾 stage 净增,s0-s2 逐 MiB 不变）。仅 **full-recompute
            # 的 mtp 层**计入（其 saved 已塌缩到 ci+ctx,loss 段不在 act_live → 无双算;无重算 mtp
            # 的同款驻留无锚点,不外推——OFF 下 loss 段本就在 act_live 逐微批计 1 份,保持既有口径）。
            _MTP_LOSS_OPS = ("final_norm", "lm_head", "logsoftmax", "nll")
            _mtp_loss_bytes = {
                l.layer_id: estimate_structure_memory(
                    [op for op in l.ops if getattr(op, "name", "") in _MTP_LOSS_OPS],
                    alloc_block_bytes=alloc_block_bytes,
                    norm_compute_dtype_bytes=norm_compute_dtype_bytes).activation_saves
                for l in layers
                if getattr(l, "layer_type", "") == "mtp" and recompute.is_full(l.layer_id)}

            # ── P0-03（2026-07-14 review）：reshard_after_forward 接线 ─────────────────
            # 语义（hyper_parallel fsdp.py:42-74 / hsdp_scheduler.py:225-250 / parallelize.py:1172-1182）:
            #   always → 前向后即 reshard,反向 re-gather（旧行为,transient）;
            #   never  → unsharded 权重从 fwd 驻留到**本模块 post_backward**（reshard_after_backward
            #            默认 True,state.py:505-537——非整 step 驻留）,反向不 re-gather;
            #   default→ PP 整体不 reshard（not pp_enabled,fsdp.py:74）= never 语义;
            #            非 PP 同 always,**除 output_layer**（parallelize.py:1176 显式 False = never 语义）。
            # 粒度简化（文档化）：按层残余组建（norm 组恒 always、字节 ~H 级忽略;experts/router 独立
            # wrap 的时间线差异并入层组）。fsdp=1（无真实切分）时 gather_buf 是逐层 bf16 cast 缓冲
            # （用完即释）,reshard 语义不适用 → 恒走 transient 路径（pp2 dp1 锚点不动）。
            _pol = getattr(pm.pc, "reshard_after_forward", "default")
            if fsdp_d > 1 or efsdp_d > 1:
                if _pol == "never":
                    no_reshard = set(layer_ids)
                elif _pol == "always":
                    no_reshard = set()
                else:  # default
                    no_reshard = set(layer_ids) if pp > 1 else set(loss_lids)
            else:
                no_reshard = set()
            resident_gather: dict = {}     # lid -> param_full_bytes（当前 unsharded 驻留）

            def _res() -> int:
                return sum(resident_gather.values())

            def _prefetch_nonres(order: list, idx: int, d: int) -> int:
                """预取双缓冲（_prefetch_param_bytes 语义）,跳过已 resident（unsharded）的层
                （对其 prefetch 是 no-op）。resident 空 ≡ 原函数（回归路径零漂移）。"""
                total = 0
                for j in range(idx + 1, min(idx + 1 + d, len(order))):
                    if order[j] not in resident_gather:
                        total += sm_by_id[order[j]].param_full_bytes
                return total

            # ── P1-15（Task A）：PP stage 间 P2P send 激活缓冲 ─────────────────────────────
            # 非末 stage（stage<pp-1）前向把本 stage 输出激活 [S,B,H] send 给下 stage。send 量级 =
            # 本 stage 最后一层的层入口 [S,B,H]（残差流跨层同形 → = 该层 checkpoint_input，已按
            # sp/tp/cp 切分，与激活口径一致）。overlap_p2p（`pipeline_parallel.py:396`
            # pipeline_parallel_overlap_p2p，默认 False）时双缓冲 2 份。**recv 侧**（非首 stage 首层
            # 输入）已隐含在 act_live 首层 pin（首层 saves 的 checkpoint_input 即 recv 回来那块）→ 不
            # 双算。pp=1 → p2p_send_bytes=0（全惰性，单 stage 锚点不动）。仅 FWD 事件驻留、BWD/optstep
            # 清零——pp2 两 stage 峰均在 BWD（stage0 bwd@4 / stage1 loss），send 只叠 FWD 峰（远低于
            # BWD 峰）→ 锚点峰值/峰事件逐字节不变；backward 方向的 grad-P2P 未建（量级同 [S,B,H]，已
            # 在 pp2-stage0 现有 580MiB 过预测余量内，OOM 安全，文档化为保守残差）。
            p2p_send_bytes = 0
            if pp > 1 and stage < pp - 1 and layer_ids:
                _overlap = 2 if getattr(pm.pc, "pipeline_parallel_overlap_p2p", False) else 1
                p2p_send_bytes = sm_by_id[max(layer_ids)].checkpoint_input * _overlap

            # ── P1-14（Task B）：experts 子模块独立 wrap 的 gather 两段生命周期 ────────────────
            # 有真实专家分片（efsdp>1）且非 resident（reshard 态）的 MoE 层，前向 gather 拆成
            # 「attn+router 段（层入口）」与「experts 段（dispatch 前 gather、combine 后 reshard）」。
            # experts 段事件逐字节复现旧单事件（gather 含全部权重 + 层 max workspace），attn 段是**新增
            # 更小事件**（不含专家权重）→ 峰值口径不变、时间线粒度闭合。expert_pf=0（dense）/efsdp<=1
            # （ep 全覆盖或未分片）/resident → 不拆（走原单事件路径，byte-identical）。
            expert_pf_by_id = {
                lid: _layer_expert_param_bytes(by_id[lid], alloc_block_bytes)
                for lid in layer_ids
            }

            B = Buckets(persistent=static_persistent.get(stage, 0))
            peak: int = -1
            peak_ev: str = ""
            peak_mb: int = -1
            peak_bd: MemBreakdown = None  # type: ignore[assignment]
            series: list = []

            def rec(tag: str, mb: int = -1, chunk: int = -1) -> None:
                nonlocal peak, peak_ev, peak_bd, peak_mb
                t = B.total() + framework_reserve
                is_peak = t > peak
                bd = None
                # P2-06（C4）：VPP 下事件标签带 #c<chunk> → (event,mb,chunk) 唯一，可与 profiler 对齐。
                lbl = f"{tag}#c{chunk}" if chunk >= 0 else tag
                if record_timeline or is_peak:
                    bd = MemBreakdown(
                        B.persistent, B.act_live, B.gather_buf, B.grad_buf,
                        B.recomp_scratch, B.bwd_scratch, B.bwd_working_set,
                        B.swap_buf, B.workspace, B.optstep,
                        framework_reserve, B.kept_frag, B.grad_accum, B.p2p_buf,
                        B.mtp_resident,
                    )
                if record_timeline:
                    series.append(TimelineSample(len(series), lbl, t, bd, mb, chunk))
                if is_peak:
                    peak = t
                    peak_ev = lbl
                    peak_mb = mb
                    peak_bd = bd

            # (mb, layer_id) -> saved bytes currently pinned in act_live
            pinned: dict = {}
            # 保留-MoE 层当前驻留激活之和（kept_frag margin 用）。碎片长尾主要来自 **MoE 的
            # dispatch/permute/router + grouped-GEMM 的 fp32 cast**（select_attn 真机 live-set 主导）。
            # margin **仅 gate 到「select 重算下 MoE/FFN 被保留」的层**——这是唯一**严重且未被补偿**的
            # 残差族（真机 select_attn 0.823）：
            #   - full 层 / loss 层：不计（→ margin 0，12409.5/cp-full 锚点不破）；
            #   - select-attn（重算 attn、留 FFN）：MoE 保留 → 计（sa 靶心，0.823→≥0.95）；
            #   - select-mlp（重算 FFN）：MoE 已重算 → 不计（sm 0.940 保持准确、不推过头）；
            #   - **no-recompute（None）此 kept_act 桶不计**：无重算-MoE 的同族残差改由**独立的
            #     `_nr_moe_act` + `nr_moe_frag_factor` 桶**处理（D1，2026-07-16，见下），gate 到 pp==1
            #     单 stage 无重算 loss-BWD；pp>1 无重算 loss stage 仍由 k_ce=8 制度化平衡、不进任一 margin。
            # 随 FWD pin / BWD pop 同步。
            kept_act = 0
            # D1（2026-07-16）：无重算-MoE 保留态碎片长尾的**标定基**——无重算下各 MoE 层（非 loss）
            #   全量驻留激活之和。margin = nr_moe_frag_factor × 此值，只在 pp==1 无重算 loss-BWD 生效。
            #   随 FWD pin / BWD pop 同步（与 kept_act 同机理，但作用域是无重算-MoE 而非 select-kept）。
            _nr_moe_act = 0
            _FFN_MARKERS = {"fc", "swiglu", "gelu", "router", "dispatch", "e_", "combine", "shared"}

            def _is_kept(lid):
                # 仅 select 重算、且 FFN 未被选中重算（选择器不含 FFN 标记）→ MoE 保留。
                if not recompute.is_select(lid) or lid in loss_lids:
                    return False
                return not (set(recompute.selectors(lid)) & _FFN_MARKERS)

            # D-4：把调度统一成 steps=[(kind, mb, ev_layers)]。
            #   v>1（交错式 VPP）：**chunk 粒度** —— 每虚拟步只处理一个 chunk(L/V 层)，忠实 Megatron
            #     (get_schedule_table + convert_schedule_table_to_order) → 峰 = Σ_chunk n_c·(L/V)，
            #     不再是「每在飞微批整 stage L 层」的 ~V× 过估（旧 build_interleaved_1f1b 物理粒度）。
            #   v<=1：ev_layers 恒 = 整个 layer_ids，逐字节复现旧 build_interleaved_1f1b→build_1f1b。
            # steps=[(kind, mb, ev_layers, chunk)]；chunk=-1 表示非 VPP（v<=1）。
            if v > 1:
                # round-robin chunk 放置（2026-07-14 修,源:mindformers pynative
                # pipeline_parallel.py:258 chunk_id*pp+rank）——由 ParallelModel.stage_chunks
                # 提供 per-chunk 层组（此前 chunk_layer_ids 连续切为文档化近似,层不均匀时有偏）。
                chunks = pm.stage_chunks(stage)
                steps = [(kind, mb, chunks[c], c)                    # C4：保留 chunk id c
                         for kind, mb, c in interleaved_virtual_order(stage, pp, m, v, pp)]
            elif getattr(pm.pc, "sched_warmup_plus_one", False) and pp > 1 and stage < pp - 1:
                # hyper_parallel Schedule1F1B 深 warmup（2026-07-23，specs.ParallelConfig
                # `sched_warmup_plus_one` docstring）：fork 调度器 warmup = min(pp−stage, m)
                # （scheduler.py:957,比 Megatron 深 1）,116 std m∈{2,4,8} 差分实测非末 stage
                # 峰值在途 = min(m, pp−stage+1) 组（m=warmup 时无 steady → 恰 warmup 组;
                # m>warmup 时 steady 首 F 与上一 B 的释放跨流共存 → warmup+1 组,饱和）。
                # 末 stage warmup=1、在途 1,与 Megatron 口径同 → 不改（走下方分支）。
                _wu = min(pp - stage, m)
                steps = [(ev.kind, ev.mb, layer_ids, -1)
                         for ev in _1f1b_from_warmup(_wu, m)]
            else:
                steps = [(ev.kind, ev.mb, layer_ids, -1)
                         for ev in build_interleaved_1f1b(stage, pp, m, v)]

            # P0-01：已完成首次反向的层集（其 reduced grad shard 已常驻 grad_accum）。
            grad_done: set = set()

            for ev_kind, ev_mb, ev_layers, ev_chunk in steps:
                if ev_kind == "FWD":
                    for idx, lid in enumerate(ev_layers):
                        sm = sm_by_id[lid]
                        # 1. FSDP all-gather 整层参数(compute dtype) + 预取下 depth 层双缓冲
                        #    （FSDP2 前向隐式 depth-1 overlap）+ workspace → 采样。
                        #    no-reshard 层（P0-03）：本层 gather 进 resident（驻留到其 post_backward）。
                        _exp_pf = expert_pf_by_id[lid]
                        _split_experts = (efsdp_d > 1 and _exp_pf > 0
                                          and lid not in no_reshard)
                        if _split_experts:
                            # P1-14（Task B）：experts 独立 wrap → 前向 gather 两段生命周期。
                            #   attn+router 段（层入口）：gather = 非专家权重 + 预取（experts 未 gather，
                            #     不在 attn 段驻留——parallelize.py:1155-1161 整层 wrap 只含非专家残余）。
                            _non_exp = sm.param_full_bytes - _exp_pf
                            B.gather_buf = _res() + _non_exp + _prefetch_nonres(
                                ev_layers, idx, depth)
                            B.workspace = 0                # attn 段（flash-ws 小），层 max-ws 归 experts 段
                            rec(f"fwd:{lid}", ev_mb, ev_chunk)
                            #   experts 段（dispatch 前 gather、combine 后 reshard）：gather = 非专家
                            #     （驻留）+ 专家权重 + 预取 = 整层 param_full + 预取；workspace = 层 max
                            #     （dispatch/combine MoE-staging，属 expert 段）→ **逐字节复现旧单事件**。
                            B.gather_buf = _res() + sm.param_full_bytes + _prefetch_nonres(
                                ev_layers, idx, depth)
                            B.workspace = sm.workspace
                            rec(f"fwd:{lid}#experts", ev_mb, ev_chunk)
                            B.workspace = 0
                            B.gather_buf = _res()   # experts + 非专家 combine/层末 reshard
                        else:
                            if lid in no_reshard:
                                resident_gather[lid] = sm.param_full_bytes
                                B.gather_buf = _res() + _prefetch_nonres(ev_layers, idx, depth)
                            else:
                                B.gather_buf = _res() + sm.param_full_bytes + _prefetch_nonres(
                                    ev_layers, idx, depth)
                            B.workspace = sm.workspace
                            rec(f"fwd:{lid}", ev_mb, ev_chunk)
                            B.workspace = 0
                            B.gather_buf = _res()   # reshard_after_forward：非 resident 部分用完即释
                        # 2. 决定该层 pin 多少 activation
                        if recompute.is_full(lid):
                            # 全重算保留 = 层入口 checkpoint_input + **重算免疫 saves**
                            # （2026-07-22，185 pp4+全重算锚点）：fused 自定义算子的
                            # `ctx.save_for_backward` 状态（SparseFlashMla 11 张量集，
                            # csa.py:224-235）在 MindSpore use_reentrant=False checkpoint
                            # 下**不被释放**（真机 ON−OFF 净省仅 6.2GB vs 修前模型 16-23GB），
                            # 随 1F1B warmup 在途微批累积、至该微批该层反向才释。
                            # 无免疫标记的 spec `recompute_pinned_saves=0` → 逐字节复现旧行为。
                            saved = sm.checkpoint_input + sm.recompute_pinned_saves
                        elif recompute.is_select(lid):
                            # 选择性重算：非选中 op 的 saves（去重）+ 层入口边界常驻；
                            # 选中 op 的 saves 丢弃（反向重物化）。介于 full 与全量之间。
                            saved = select_mem_by_id[lid].act_live_pinned
                        elif swap.swaps(lid):
                            saved = 0                                 # 全部卸载到 CPU
                        else:
                            saved = sm.activation_saves               # 全量 saves（去重）
                        pinned[(ev_mb, lid)] = saved
                        B.act_live += saved
                        # MTP loss 链步内驻留（见 _mtp_loss_bytes 注释）：每微批前向累加,
                        # 不随该微批反向释放,至 optstep 前清零。
                        B.mtp_resident += _mtp_loss_bytes.get(lid, 0)
                        if _is_kept(lid):
                            kept_act += saved
                        # D1：无重算-MoE 非 loss 层的全量驻留激活累计（margin 标定基）。仅 nr margin
                        #   开启时用；no-recompute 判据 = 非 full/非 select/非 swap（与该层 saved 口径一致）。
                        if (nr_moe_frag_factor and lid in _moe_lids and lid not in loss_lids
                                and not recompute.is_full(lid) and not recompute.is_select(lid)
                                and not swap.swaps(lid)):
                            _nr_moe_act += saved
                    # 该虚拟步(v>1: 一个 chunk / v<=1: 整 stage)所有层 pin 完毕 → FWD 峰。
                    # P1-15（Task A）：本 stage 输出激活已产出 → send buffer 驻留（非末 stage）。
                    B.p2p_buf = p2p_send_bytes
                    rec("fwd_end", ev_mb, ev_chunk)

                else:  # BWD（逆序层）—— FSDP gather + full grad + recompute + bwd_scratch 共存
                    # P1-15（Task A）：反向进入 → 前向 send buffer 已释（本 stage 峰在 BWD，
                    # 清零使 send buffer 不叠加 BWD 峰 → pp2 锚点逐字节不动）。
                    B.p2p_buf = 0
                    bwd_order = list(reversed(ev_layers))
                    for idx, lid in enumerate(bwd_order):
                        sm = sm_by_id[lid]
                        # 反向某层峰值 = 该层 FSDP 重新 gather 的整层参数(compute)
                        #   + 预取反向下 depth 层的双缓冲（FSDP2 默认 depth-1 反向预取，
                        #     逆前向序：_fsdp_param_group.py:854-856；output_layer 首个反向
                        #     单元预取末 transformer 层）
                        #   + reduce-scatter 前 full 梯度(grad dtype)
                        #   + (full 重算)重物化激活 recomp_scratch / (无重算)反向工作集 bwd_working_set
                        #   + (op)反向临时物化 bwd_scratch(如 loss probs)
                        #   共存，叠在 persistent + 其余 act_live 之上。
                        #   resident（no-reshard）层反向不 re-gather（hsdp_scheduler.py:241-250：
                        #   仅 reshard_after_forward=True 才 re-gather 自身）——其 param_full 已在 _res()。
                        _regather = 0 if lid in resident_gather else sm.param_full_bytes
                        B.gather_buf = _res() + _regather + _prefetch_nonres(
                            bwd_order, idx, depth)
                        B.grad_buf = sm.grad_full_bytes
                        B.bwd_scratch = sm.bwd_scratch
                        # ① 无重算下 loss 层：unfused CE 链共存 k_ce 份满 vocab fp32。现 bwd_scratch
                        #   =8·S·B·vocab=2 份（probs+grad）→ 改到 k_ce-1 份（logsm 1 份在 act_live）。
                        #   **k_ce 与制度相关（真机 profiler）**：流水线末 stage（pp>1，有 loss）CE 链保留更多
                        #   中间量 → k_ce≈8（pp2-stage1 实测）；单 stage（pp=1）CE 链释放快 → 实测 3 份
                        #   共存（cp2-none/select），代码取 **4 = 3 观测 + 1 保守**（OOM-安全侧，
                        #   P2-08 注释对齐 2026-07-14）。
                        #   **2026-07-14 修（review P0.1）**：判据从「全局 mode=='None'」改为
                        #   「**本 stage 无任何层被重算**」——全局 None/full/select 下行为逐字节不变
                        #   （None→全 stage 无重算→fat ✓;full/select→loss stage 含被重算 transformer→lean ✓）;
                        #   per-stage select（如 s0:both;s1:none）时未重算的 loss stage 恢复 fat
                        #   （修前被全局 mode=='select' 误关,低估 45%）。
                        if _stage_no_recompute and lid in loss_lids and sm.bwd_scratch > 0:
                            #   **ce_pynative_lean（2026-07-23，116 std MHA/GQA 锚点定标）**：该 build
                            #   的 unfused CE 链实测 ~3.3-4 份 co-live 且 **与 pp 无关**（std pp1 与
                            #   pp2-s1 差分一致）→ lean K=4。制度常数 8/4 保留为默认（DSv3-era 冻结
                            #   口径——那批探针的 K=8 是含当时未建模效应的混合常数,勿动其锚点）。
                            K_CE = 4 if ce_pynative_lean else (8 if pp > 1 else 4)
                            B.bwd_scratch = sm.bwd_scratch // 2 * (K_CE - 1)
                        # 激活 swap（§8.1 swap_buf="从 CPU 预取回的激活"）：被卸载层反向前 H2D 取回，
                        #   swap_buf = 复原当前层 saves（不在 act_live）+ 反向序后 swap_depth 层在飞预取窗
                        #   （双缓冲，`_prefetch_swap_bytes`）。当前层复原量 = 前向从 act_live 扣掉的同一
                        #   `activation_saves`（对称：卸载多少、取回多少），故被 swap 层反向激活仍驻留、不欠算。
                        #   非卸载层此项 0 → swap 关逐字节复现旧行为。
                        B.swap_buf = (
                            (sm.activation_saves if lid in offloaded else 0)
                            + _prefetch_swap_bytes(bwd_order, idx, swap_depth, offloaded, sm_by_id))
                        if recompute.is_full(lid):
                            # 重算层：反向重跑 forward，其重物化 = **该层 forward 的 max-live**
                            # （mini-fwd 时间线峰值，§8.5②①②）扣掉已 pin 进 act_live 的 checkpoint
                            # 输入。取代旧 `saves 之和 − checkpoint` 近似（saves 累加全层、非峰值；
                            # 多中间量层 max-live 常 < Σsaves，旧式高估）。
                            B.recomp_scratch = max(
                                0, sm.forward_max_live - sm.checkpoint_input)
                        elif recompute.is_select(lid):
                            # 选择性重算：选中 op 反向重物化（recomp_scratch，= 选中段 forward_max_live
                            # 扣段边界）与非选中 op 反向工作集（bwd_working_set，= 非选中段
                            # forward_max_live 扣 bwd_scratch）共存。全选→复现 full（recomp=fml−ci、
                            # bwd_ws=0）；全不选→复现无重算（recomp=0、bwd_ws=fml−bwd_scratch）。
                            smem = select_mem_by_id[lid]
                            B.recomp_scratch = smem.recomp_scratch
                            B.bwd_working_set = smem.bwd_working_set
                        else:
                            # 无重算层：反向仍需再遍历 forward 求梯度，激活梯度 dL/dact 与激活同形、
                            # 同样共存 → 反向工作集 ≈ 该层 forward_max_live（§8.5②）。其中已被显式建模
                            # 的 op 级 fp32 反向物化（loss probs+grad_log_softmax / mHC sinkhorn grad）
                            # 已在 bwd_scratch 计入 → 扣除避免双算。loss 层 bwd_scratch(满 vocab fp32
                            # ×2) ≥ forward_max_live → 该项 = 0（不动已验证的 DSv3/loss 峰，无双算）；
                            # transformer 层 bwd_scratch=0 → = forward_max_live（此前欠建的反向工作集）。
                            B.bwd_working_set = max(
                                0, sm.forward_max_live - sm.bwd_scratch)
                        # **标定 margin**（B 方案，2026-07-09）：保留(非重算)模块在 loss 峰的
                        #   fp32-cast 横切 + 小张量长尾——源码级 op-DAG 提取证实此残差**在 op 图粒度之下**
                        #   （profiler live-set 313 个 <100MiB 碎片，opdag_validation.md），非 op 图可导出 →
                        #   明示为**标定常数**（factor × 当前 kept 激活），仅 loss-BWD 事件生效、与 loss 区共存那一刻。
                        #   full 重算 kept_act=0 → 0（12409.5/cp-full 锚点不破）；无-loss stage 无 loss_lids → 不触发。
                        if kept_frag_factor and lid in loss_lids and kept_act > 0:
                            B.kept_frag = round(kept_frag_factor * kept_act)
                        # D1（2026-07-16）：**无重算-MoE OOM-安全标定 margin**（非物理，2 点标定）。
                        #   无重算下 MoE 保留态的 dispatch/permute/grouped-GEMM fp32-cast 横切 + 小张量
                        #   长尾（profiler live-set 313 个 <100MiB 碎片，opdag_validation.md：**在 op 图
                        #   粒度之下**，不可显式建模——与 select-kept 的 kept_frag 同族残差，另一作用域）。
                        #   **gate = pp==1 单 stage 无重算 loss-BWD**：pp>1 的无重算 loss stage 已由
                        #   K_CE=8（line 上文）制度化平衡到 ~1.007，故显式排除（探针实证：pp2-stage0/1、
                        #   select、full 锚点在任意 factor 下逐字节不动）。factor=0.6 由 8L-none(→1.009)+
                        #   cp2-none(→1.021) 两锚点联合标定使二者 OOM-安全（预测≥真机）；两点理想 factor
                        #   0.53/0.45 差 ~15% → 明示为**标定常数**、非精确物理，可由 preset 单值调/关。
                        #   fused-CE（loss_lids 空，DSv4）不触发 → 不覆盖 mHC+MTP，其欠预测另档（D2）。
                        if (nr_moe_frag_factor and pp == 1 and _stage_no_recompute
                                and lid in loss_lids and _nr_moe_act > 0):
                            B.kept_frag += round(nr_moe_frag_factor * _nr_moe_act)
                        rec(f"bwd@{lid}", ev_mb, ev_chunk)
                        B.grad_buf = B.recomp_scratch = 0
                        B.bwd_scratch = B.bwd_working_set = B.swap_buf = B.kept_frag = 0
                        # post_backward：resident 层此刻 reshard（reshard_after_backward 默认 True，
                        # state.py:505-537）→ 从 resident 集移除；gather_buf 回落到其余 resident。
                        resident_gather.pop(lid, None)
                        B.gather_buf = _res()
                        # 该层反向结束，释放其 pinned 激活（(mb,lid) 唯一键，chunk 互斥→无碰撞）
                        _popped = pinned.pop((ev_mb, lid))
                        B.act_live -= _popped
                        if _is_kept(lid):
                            kept_act -= _popped
                        # D1：无重算-MoE 层反向结束 → 从 margin 标定基移除（与 FWD 累计对称；loss 层
                        #   先反向、其时 _nr_moe_act 仍满，故 margin 取全量；此处保持后续层平衡、无泄漏）。
                        if (nr_moe_frag_factor and lid in _moe_lids and lid not in loss_lids
                                and not recompute.is_full(lid) and not recompute.is_select(lid)
                                and not swap.swaps(lid)):
                            _nr_moe_act -= _popped
                        # P0-01：该层**首次**反向完成 → reduced grad shard 常驻（后续 microbatch
                        # 就地 AssignAdd/RS-accumulate 累加,不新增内存 → 只加一次）。当前层的 shard
                        # 在事件结束后才计入——fsdp=1 时 grad_buf 与驻留 grad 是同一缓冲,事件内不双计;
                        # mb≥2 时该层已在 done 集,full grad(新物化)与旧 shard 共存,如实计。
                        # offload_grads（P1-19）：梯度在 host 侧、不驻设备 → grad_accum 桶不累计
                        #   （旧 cpu_offload 一次卸全部；现独立于 param/optimizer 卸载）。
                        if lid not in grad_done:
                            grad_done.add(lid)
                            if not offload_grads:
                                B.grad_accum += sm.grad_shard_bytes

            # ② 优化器-step 事件（真机 profiler：pp=2 stage0 峰 = AdamW 更新 embedding 的瞬态，
            #   非层反向）。step 在**所有反向之后**、激活已释 → 与激活桶互斥（**累计梯度 grad_accum
            #   除外**——真机探针证实全部 reduced grad 在 optimizer 前仍驻留，P0-01）。AdamW 逐参数
            #   更新，峰在**最大单权重**：其 fp32 [weight] 临时（Square(g²)/sqrt(v̂)/m̂/update）。
            #   ★K_OPT 重标 6→4（P0-01，2026-07-14）：旧 6 是在**含累计梯度**的真机 optstep 峰
            #   （DSv3 8L pp2 stage0 = 10246.2）上标定的混合常数——其中 ≈1.9 份（=G_s0 1669.5 MiB）
            #   实为累计梯度、非 AdamW 瞬态。拆出 grad_accum 桶后重标：
            #   (10246.2 − persistent 5008.5 − G 1669.5) / max_w 883.8 = 4.04 ≈ 4（重标后 0.997）。
            #   ★权重须按 FSDP 切（optim_grads_params：AdamW step 只跑本 rank 的 1/fsdp 分片）——
            #   dense÷fsdp、expert÷efsdp（与 static_mem.persistent 同口径，resolve 只切了 tp/ep）。
            #   offload_optimizer（P1-19）时优化器 step 在 CPU、无设备瞬态 → 该项 0（旧 cpu_offload
            #   一次卸全部；现独立于 param/grad 卸载）。
            K_OPT = 4

            def _shard(w):
                return w.local_numel // (efsdp_d if getattr(w, "is_expert", False) else fsdp_d)

            if offload_optimizer:
                optstep_bytes = 0
            elif muon:
                # Muon optstep = 逐参瞬态取最大：**2D 矩阵权重走 Newton-Schulz workspace**
                #   (≈ _MUON_NS_WORKSPACE_MULT × 分片 numel × 4，**估值,无真机锚点**；per-head 时注意力
                #   投影按头切、一次一头 → 该投影 NS 单元 = 整块/n_heads);**embed/head/norm 走 AdamW**
                #   (K_OPT×分片×4)。取二者最大——大 vocab head/embed 的 AdamW 瞬态常与最大专家 NS 争峰。
                def _muon_transient(op, w):
                    shard = _shard(w)
                    if is_muon_matrix_weight(op.type, getattr(op, "name", "")):
                        unit = shard
                        if (muon_per_head and muon_n_heads > 1
                                and is_attn_projection(getattr(op, "name", ""))):
                            unit = max(1, shard // muon_n_heads)
                        return round(_MUON_NS_WORKSPACE_MULT * unit * 4)   # NS workspace(fp32 temp)
                    return K_OPT * shard * 4                                # 非矩阵 → AdamW
                optstep_bytes = max(
                    (_muon_transient(op, w) for l in layers for op in l.ops for w in op.params),
                    default=0)
            else:
                optstep_bytes = K_OPT * max(
                    (_shard(w) for l in layers for op in l.ops for w in op.params), default=0) * 4
            if optstep_bytes > 0:
                B.act_live = B.gather_buf = B.grad_buf = B.recomp_scratch = 0
                B.bwd_scratch = B.bwd_working_set = B.swap_buf = B.workspace = 0
                B.p2p_buf = 0                          # P1-15：step 在所有反向后、P2P 已收尾
                B.mtp_resident = 0                     # MTP loss 图随全部反向完成释放（step 末）
                B.optstep = optstep_bytes              # fp32 瞬态（grad_accum 保持驻留，与之共存）
                rec("optstep")
                B.optstep = 0
            B.mtp_resident = 0                         # 无 optstep 事件路径同样清零（跨 stage 复用 B）
            B.grad_accum = 0                           # zero_grad 语义：optimizer 后释放（真机探针）

            res[stage] = StagePeak(
                stage=stage,
                peak_bytes=peak,
                breakdown=peak_bd,
                peak_event=peak_ev,
                oom=(peak > max_device_memory),
                timeline=tuple(series),
                peak_mb=peak_mb,
            )

        return res
