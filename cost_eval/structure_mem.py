"""每结构内存 rollup（模块化组装 + 单点去重）。

用户架构诉求：内存估计应**按基本结构模块化**——每个基本结构（attention / ffn / moe /
embedding / head / …）都有自己一份可组合的内存估计，装配起来即整层/整模型。OpSpec 契约与
op-builder 已经是模块化的；本模块补上**每结构 rollup**：把一段 ResolvedOp（属于同一个结构）
按内存契约（§2.1）汇总成 `StructureMemory`，且**这是全库唯一的按名去重点**——
`static_mem`（持久 param+opt）与 `mem_timeline`（gather/grad/act_live/recomp/bwd_scratch）
都组装它，不再各自裸 walk raw ops、各写一份求和。

去重规则（设计 §2.1）：
  - **params 按名去重**：同一物理权重被多 op 引用只算一次（结构内）。跨结构（跨层）不去重
    ——不同层的同名权重是不同物理张量，由调用方逐结构相加。
  - **saves 按名去重**：同一 `save_for_backward` 张量被多 op 保留只占一份显存（修 C1 attn 双算）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureMemory:
    """单个基本结构的内存分解（各字段均为字节，已按名去重）。

    - `persistent`：持久 param+opt 字节（已按 fsdp/efsdp 切 + opt 倍数）。需 fsdp/efsdp/
      opt_state_bytes 才非零，否则 0（mem_timeline 只取瞬态桶时不传，持久由 static_mem 供）。
    - `activation_saves`：saves 去重后总字节（全量保存时 pin 进 act_live）。
    - `param_full_bytes`：full-unsharded 参数字节（compute dtype）——FSDP all-gather 缓冲。
    - `grad_full_bytes`：full-unsharded 梯度字节（grad dtype）——反向 reduce-scatter 前 grad_buf。
    - `bwd_scratch`：该结构各 op 反向临时物化之和（如 loss probs/grad_log_softmax fp32）。
    - `workspace`：该结构各 op workspace 的最大值（FWD 逐层瞬时）。
    - `checkpoint_input`：full 重算时保留的层入口激活（首个有 saves 的 op 的首个 save）字节。
    - `forward_max_live`：**该结构 forward 的峰值工作集**——mini-forward 时间线上同时存活激活
      张量字节 + 该 op workspace 的最大值（设计 §8.5②）。反向再遍历 forward 求梯度，其工作集
      ≈ 此峰值（激活梯度 dL/dact 与激活同形、同样共存）。**≠ `activation_saves`**：后者是整层
      saved 张量去重之和（一个子集、跨全层累加），前者是单时刻峰值——二者无大小关系（多中间量
      层常 `forward_max_live < activation_saves`，故不能用 `fml − saves` 当反向工作集，见
      `mem_timeline`）。
    """
    persistent: int = 0
    activation_saves: int = 0
    param_full_bytes: int = 0
    grad_full_bytes: int = 0
    bwd_scratch: int = 0
    workspace: int = 0
    checkpoint_input: int = 0
    forward_max_live: int = 0


def _align_up(nbytes: int, block: int) -> int:
    """分配器块对齐上取整：把一次分配的字节数按内存池块粒度上取整（平台属性，非拟合）。

    MindSpore 设备内存池 `DynamicMemPoolBestFit` 对每次分配按 `kDynamicMemAlignSize`(=512B)
    对齐；`max_memory_allocated`（分配峰值）因此 = Σ 各张量按块上取整的字节。block<=1 → 不取整
    （回归/直调路径逐字节复现旧行为）。"""
    if block <= 1:
        return nbytes
    return ((nbytes + block - 1) // block) * block


def _forward_max_live(resolved_ops, blk: int) -> int:
    """mini-forward 时间线求**峰值工作集**（设计 §8.5②）。

    逐 op 顺序走一遍，维护"当前存活激活张量"集合：某 op 的 `output` 变为存活；某激活作为
    input 被消费——它存活到**最后一次**被引用（作 input 或 output）的 op。峰值 =
    `max_op(Σ 存活激活字节 + 该 op.workspace)`。

    规则（与库内其它桶一致）：
      - **仅激活**：`params`（is_weight 权重）不计——权重在 gather_buf/persistent 另算，反向
        工作集是激活及其梯度。MATMUL 的权重同时出现在 `inputs` 与 `params`，按 `is_weight` 剔除。
      - **按名去重**：同名张量（含 in-place 复用同一引用，如 rope 读写 qkv）只占一份。
      - **逐张量块对齐**：与 `activation_saves` 同口径 `_align_up(numel·dtype, blk)`，可比。
    活性区间取 `[首次出现 op 序, 末次出现 op 序]`：output 在其产出 op 诞生、input 在其消费 op
    仍活；产而不被本结构消费的张量（如层输出 h2）在其产出 op 即计入（下一结构再接手）。
    """
    ops = list(resolved_ops)
    if not ops:
        return 0
    byt: dict = {}          # name -> 对齐后字节（首次出现定，去重）
    first: dict = {}        # name -> 首次出现 op 序
    last: dict = {}         # name -> 末次出现 op 序
    for i, op in enumerate(ops):
        acts = [t for t in op.inputs if not t.is_weight]
        acts.append(op.output)                      # output 恒为激活
        for t in acts:
            if t.name not in byt:
                byt[t.name] = _align_up(t.local_numel * t.dtype_bytes, blk)
                first[t.name] = i
            last[t.name] = i
    peak = 0
    for i, op in enumerate(ops):
        live = sum(b for n, b in byt.items() if first[n] <= i <= last[n])
        peak = max(peak, live + op.workspace_bytes)
    return peak


def estimate_structure_memory(
    resolved_ops,
    *,
    fsdp: int = 1,
    efsdp: int = 1,
    opt_state_bytes: int = 0,
    grad_dtype_bytes: int = 4,
    alloc_block_bytes: int = 1,
) -> StructureMemory:
    """把一段属于同一结构的 ResolvedOp 汇总成 `StructureMemory`（按名去重）。

    参数
    ----
    resolved_ops    : Iterable[ResolvedOp]  — 一个结构（attn/ffn/整层/embedding/head …）的 op。
    fsdp / efsdp    : int  — 持久 param 的 FSDP 分母（专家用 efsdp，其余用 fsdp）。
    opt_state_bytes : int  — 持久 = param+opt 的每元素字节倍数（AdamW fp32=12/bf16=14）；0=不算持久。
    grad_dtype_bytes: int  — 反向瞬态 grad 的 dtype 字节。
    alloc_block_bytes: int — 设备内存池分配对齐块（平台属性，`HardwareSpec.alloc_block_bytes`，
        默认 512）。**逐张量**按此上取整——分配峰值(max_memory_allocated)的分配器碎片公式，
        取代经验 framework_reserve 常数。默认 1（无取整）供直调/回归逐字节复现旧值。
    """
    blk = alloc_block_bytes
    # ── 按名去重：params / saves（结构内同名 = 同一物理张量，只算一次）──────────────
    params: dict = {}
    saves: dict = {}
    for op in resolved_ops:
        for w in op.params:
            params[w.name] = w
        for s in op.saves:
            saves[s.name] = s

    # 持久 = param+opt（按 fsdp/efsdp 切）。**切分不整除即报错，不静默 floor**（I1，OOM 安全）：
    # 静默截断会低估每卡显存 → OOM 风险，且与 resolve_tensor 对 sharded 维不整除即 raise 的
    # 口径不一致（shape_eval.py:59-63）。opt_state_bytes==0（mem_timeline 只取瞬态桶）时不查。
    persistent = 0
    if opt_state_bytes:
        for w in params.values():
            divisor = efsdp if w.is_expert else fsdp
            if w.local_numel % divisor != 0:
                raise ValueError(
                    f"{w.name} local_numel={w.local_numel} 不被 "
                    f"{'efsdp' if w.is_expert else 'fsdp'}={divisor} 整除"
                    f"（切分不整除，静默截断会低估显存→OOM 不安全，改为报错）")
            persistent += _align_up((w.local_numel // divisor) * opt_state_bytes, blk)
    activation_saves = sum(_align_up(s.local_numel * s.dtype_bytes, blk) for s in saves.values())
    param_full_bytes = sum(_align_up(w.local_numel * w.dtype_bytes, blk) for w in params.values())
    grad_full_bytes = sum(_align_up(w.local_numel * grad_dtype_bytes, blk) for w in params.values())
    bwd_scratch = sum(getattr(op, "bwd_scratch_bytes", 0) for op in resolved_ops)
    workspace = max((op.workspace_bytes for op in resolved_ops), default=0)

    checkpoint_input = 0
    for op in resolved_ops:
        if op.saves:
            s = op.saves[0]
            checkpoint_input = _align_up(s.local_numel * s.dtype_bytes, blk)
            break

    forward_max_live = _forward_max_live(resolved_ops, blk)

    return StructureMemory(
        persistent=persistent,
        activation_saves=activation_saves,
        param_full_bytes=param_full_bytes,
        grad_full_bytes=grad_full_bytes,
        bwd_scratch=bwd_scratch,
        workspace=workspace,
        checkpoint_input=checkpoint_input,
        forward_max_live=forward_max_live,
    )


# ---------------------------------------------------------------------------
# 选择性重算（selective recompute）——按 op 划分的内存分解
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectMemory:
    """一层在**选择性重算**下的三个内存桶（字节，已对齐/去重）。

    选择性重算把一层的 op 分成「选中（重算）」与「非选中（保存）」两部分：选中 op 的 saves
    前向不常驻、反向重物化；非选中 op 的 saves 照常常驻。据此：

    - ``act_live_pinned``：前向常驻激活 = **非选中 op 的 saves**（去重）∪ **层入口
      checkpoint_input**（重算边界，始终保留）。丢掉的正是「只被选中 op 保存」的那些张量。
    - ``recomp_scratch``：反向重物化选中 op = `forward_max_live(选中 op) −
      _pinned_input_boundary(选中 op)`——**每个选中 op 自估自身**足迹（D-3）。选择性重算把每个选中
      op / 模块**各自**包成一个 checkpoint region（反向逐个、独立重跑），故层的重算工作集峰 =
      `forward_max_live(选中集)`（连续选中岛内共存取峰、跨岛取 max），**再扣其中确实已 pin 的进入
      边界输入**（已在 act_live 计过，避免双算）——**而非**无条件扣「选中段首个 save」。旧式无条件
      扣 `checkpoint_input(选中段)`，隐含「段边界已由前驱 pin」，对**模块/cell 粒度**精确、但对
      **细粒度单 op**（如仅 `flash`，其输入边界未必 pin）会把并未常驻的输入当「已提供」减掉 →
      **低估**（OOM 不安全）。D-3 改为**按实际 pin 的进入边界扣**，修正低估、两端仍退化不变。
    - ``bwd_working_set``：非选中段反向工作集 = `forward_max_live(非选中 op) −
      bwd_scratch(非选中)`——与无重算路径同构，只是**范围收窄到非选中 op**。

    两端退化**逐字节复现**既有公式（这是 None/full 不变、且 select-all==full /
    select-none==None 的机理根因）：
      - 全选（选中=全部 op）：非选中=∅ → act_live_pinned=checkpoint_input、
        recomp=forward_max_live(全)−checkpoint_input、bwd_working_set=0 == **full**。
      - 全不选（选中=∅）：act_live_pinned=activation_saves、recomp=0、
        bwd_working_set=forward_max_live(全)−bwd_scratch == **None**。

    > D-3（已修正）：mindformers 选择粒度可细到**单个 op**（`config.py:759-796` `select_module`/
    > `exclude_op`；`activation_checkpoint.py` 对每个命中 op/模块**各自**包 checkpoint）。旧式
    > `− checkpoint_input(选中段)` 只在**模块/cell 粒度**（边界=层/模块入口，已 pin）精确；对细粒度
    > 单 op（如仅 `flash`，输入边界未必 pin）会**低估** recomp（OOM 不安全）。现按用户定夺「细粒度选
    > 重根据单个 op 自己去估计自己」——`recomp = forward_max_live(选中集) − _pinned_input_boundary`，
    > 只扣**实际已 pin** 的进入边界输入。模块/全选粒度边界=层入口已 pin → 逐字节复现旧值；单 op 边界
    > 未 pin → 不扣 → 按该 op 自身完整重算足迹计，不再低估。
    """
    act_live_pinned: int = 0
    recomp_scratch: int = 0
    bwd_working_set: int = 0


def _pinned_input_boundary(selected, pinned_names, blk: int) -> int:
    """选中集**进入边界里已 pin 的输入**字节之和（按名去重、逐张量块对齐）。D-3 的核心。

    进入边界 = 被选中 op 消费、但**不由任何选中 op 产出**的非权重输入（段外产出或层入口）——须
    常驻才能起算选中 op 的重算。其中**已 pin**（名字 ∈ ``pinned_names`` = 非选中 saves ∪ {层
    checkpoint_input}）的部分，其字节已计入 act_live，故从 recomp 扣掉避免双算；**未 pin** 的
    （细粒度单 op 的输入边界——唯一 saver 是它自己且已随选中丢弃）**保留**在 recomp（该 op 须自付
    其输入的重物化）。这取代旧式无条件扣「选中段首个 save」的乐观假设（对模块粒度精确、对单 op
    低估 → OOM 不安全，见 §SelectMemory D-3）。

    两端退化保证 byte-identical：
      - 全选 / 模块粒度：进入边界 = 层入口 `x`（= `ci`，已 pin）→ 扣 `x` 字节 = 旧
        `checkpoint_input`，故 `recomp = forward_max_live − x` 逐字节复现 full。
      - 全不选：selected=∅ → 无输入可迭代 → 返回 0（recomp 另经 `forward_max_live(∅)=0` 归零）。
    """
    produced = {op.output.name for op in selected}   # 段内产出 → 现算，已在 forward_max_live 计
    seen: set = set()
    total = 0
    for op in selected:
        for t in op.inputs:
            if t.is_weight or t.name in produced or t.name in seen:
                continue
            seen.add(t.name)
            if t.name in pinned_names:               # 进入边界且已 pin → 扣（已在 act_live）
                total += _align_up(t.local_numel * t.dtype_bytes, blk)
    return total


def estimate_select_memory(resolved_ops, is_selected, *, alloc_block_bytes: int = 1) -> SelectMemory:
    """按 `is_selected(op) -> bool` 把一层 op 划分为选中/非选中，算选择性重算三桶。

    复用 `estimate_structure_memory` 的 rollup / 去重 / `forward_max_live` 机理（不另写一份）：
    对选中子集、非选中子集各调一次，再按 §SelectMemory 组装。`alloc_block_bytes` 与主路径同口径
    传导（逐张量块对齐）。

    参数
    ----
    resolved_ops     : Iterable[ResolvedOp]  — 整层（或整结构）的 op。
    is_selected      : callable(op) -> bool  — 该 op 是否选中重算（选中→saves 丢弃、反向重物化）。
    alloc_block_bytes: int                    — 设备内存池分配对齐块（平台属性），默认 1（不取整）。
    """
    ops = list(resolved_ops)
    blk = alloc_block_bytes
    selected = [op for op in ops if is_selected(op)]
    nonselected = [op for op in ops if not is_selected(op)]

    sm_sel = estimate_structure_memory(selected, alloc_block_bytes=blk)
    sm_non = estimate_structure_memory(nonselected, alloc_block_bytes=blk)

    # 层入口 checkpoint_input（重算边界，始终保留）：整层第一个有 saves 的 op 的首个 save。
    ci_name = None
    ci_bytes = 0
    for op in ops:
        if op.saves:
            ci_name = op.saves[0].name
            ci_bytes = _align_up(op.saves[0].local_numel * op.saves[0].dtype_bytes, blk)
            break

    # act_live_pinned = 非选中 saves（去重）∪ {ci}。ci 若已在非选中 saves 里则不重复加（按名去重）。
    nonsel_save_names = {s.name for op in nonselected for s in op.saves}
    act_live_pinned = sm_non.activation_saves + (ci_bytes if ci_name not in nonsel_save_names else 0)

    # D-3：recomp = 选中集 forward_max_live − **进入边界里实际已 pin 的输入**（每 op 自估自身足迹），
    # 不再无条件扣「选中段首个 save」（旧式对细粒度单 op 低估→OOM 不安全；见 §SelectMemory D-3）。
    # pinned = 非选中 saves ∪ {层 ci}（ci 恒常驻）。模块/全选粒度边界=层入口(=ci)已 pin → 逐字节复现
    # 旧公式；单 op 边界未 pin → 不扣 → 按 forward_max_live([op]) 全额计。
    pinned_names = nonsel_save_names | ({ci_name} if ci_name is not None else set())
    recomp_scratch = max(
        0, sm_sel.forward_max_live - _pinned_input_boundary(selected, pinned_names, blk))
    # bwd_working_set = 非选中段 forward_max_live − 非选中 bwd_scratch（与无重算同构，范围收窄）。
    bwd_working_set = max(0, sm_non.forward_max_live - sm_non.bwd_scratch)

    return SelectMemory(
        act_live_pinned=act_live_pinned,
        recomp_scratch=recomp_scratch,
        bwd_working_set=bwd_working_set,
    )
