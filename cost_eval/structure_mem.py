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
