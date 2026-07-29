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

from .specs import is_muon_matrix_weight
from .model_spec import NORM_KIND_CASTING, NORM_KIND_NONCASTING


@dataclass(frozen=True)
class StructureMemory:
    """单个基本结构的内存分解（各字段均为字节，已按名去重）。

    - `persistent`：持久 param+opt 字节（已按 fsdp/efsdp 切 + opt 倍数）。需 fsdp/efsdp/
      opt_state_bytes 才非零，否则 0（mem_timeline 只取瞬态桶时不传，持久由 static_mem 供）。
    - `activation_saves`：saves 去重后总字节（全量保存时 pin 进 act_live）。
    - `param_full_bytes`：full-unsharded 参数字节（compute dtype）——FSDP all-gather 缓冲。
    - `grad_full_bytes`：full-unsharded 梯度字节（grad dtype）——反向 reduce-scatter 前 grad_buf。
    - `grad_shard_bytes`：**已规约本地梯度分片**字节（P0-01，2026-07-14）——真机证实梯度是
      step-scoped cumulative（两卡探针 7566.1−5676.6=1889.5 MiB ≡ 逻辑梯度/2）：该结构首次反向后
      其 reduced shard 常驻至 optimizer、zero_grad 释放。divisor 与 persistent 完全同口径
      （dense÷fsdp、expert÷efsdp）+ 逐权重块对齐；公式经 DSv4 FSDP-2 真机 0.999 交叉验证
      （1887.6 vs 1889.5，review_evidence_2026-07-14.md）。需传 fsdp/efsdp，默认 1=全量。
    - `bwd_scratch`：该结构反向临时物化的 **max-live 峰值**（P1-08，如 loss probs/grad_log_softmax
      fp32）。旧口径是各 op 求和（隐含同时存活、保守高估）；现按 op 时序取 backward max-live
      （逆序滑窗 window=2，见 `_backward_max_live`）——**单 scratch op 层**（loss `nll` / DSA·dsv4
      `indexer`，唯一大 scratch）逐字节 == 旧 sum（K_CE/golden 不破），**多 scratch op 层**取更紧
      的相邻对峰（分离的 mHC sinkhorn 退化为纯 max）。恒有 max-live ≤ sum（OOM 安全）。
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
    grad_shard_bytes: int = 0
    # ── 重算免疫 saves（**2026-07-24 口径切换后恒 0**）──────────────────────────────
    # `recompute_pinned_saves`：saves 中标 `pin_under_recompute=True` 的张量字节和。经验 pin 补偿
    #   已删除（见 model_spec.TensorRef.pin_under_recompute 注释）→ 当前无任一 spec 设此标志 →
    #   **恒 0**，全重算 saved 恒 = checkpoint_input（纯理论重算边界）。字段/聚合保留作通用机制占位。
    recompute_pinned_saves: int = 0
    # ── bwd 期 kernel workspace（2026-07-29，167 真机 memory-tracker 实测）────────────────
    # `bwd_workspace`：该结构各 op `bwd_workspace_bytes` 的 **max**（不是 sum）。
    #   取 max 是**实测判据**，不是保守假设：tracker 逐块记录显示这类块的寿命**恒为 1 个
    #   tracker tick**（rank0 4554/4554、rank2 5565/5565 全部如此），且在真机峰值那一刻
    #   **只有一块 workspace 在世**（rank0 30307.8 MiB 峰、rank2 21019.5 MiB 峰，均只有一块
    #   730.00 MiB）→ 同一时刻至多一个 kernel 的 workspace 存活，层内叠加取 max 即精确。
    # 与 `workspace`（fwd 期，同样是 max）对称；与 `bwd_scratch` **正交**：后者是反向工作集
    #   的一部分（`mem_timeline` 用 `bwd_working_set = fml − bwd_scratch` 与之互补），前者是
    #   叠在工作集之上的池 scratch。见 `model_spec.OpSpec.bwd_workspace` 的对比说明。
    # 尾部追加、默认 0 → 未标注的 spec/抽取图逐字节不变。
    bwd_workspace: int = 0
    # persistent 组成分解用：本结构内、按 fsdp/efsdp 切后的**驻留参数量**（去重、未乘倍数、未块对齐）。
    #   matrix = Muon 分类的 2D 矩阵权重（is_muon_matrix_weight）；other = 其余。persistent 分量拆解
    #   （参数副本/master/momentum/v）= 这两个计数 × 每分量每元素字节（static_mem.persistent_breakdown）。
    persist_numel_matrix: int = 0
    persist_numel_other: int = 0


def _align_up(nbytes: int, block: int) -> int:
    """分配器块对齐上取整：把一次分配的字节数按内存池块粒度上取整（平台属性，非拟合）。

    MindSpore 设备内存池 `DynamicMemPoolBestFit` 对每次分配按 `kDynamicMemAlignSize`(=512B)
    对齐；`max_memory_allocated`（分配峰值）因此 = Σ 各张量按块上取整的字节。block<=1 → 不取整
    （回归/直调路径逐字节复现旧行为）。"""
    if block <= 1:
        return nbytes
    return ((nbytes + block - 1) // block) * block


def _fsdp_local_count(w, divisor: int, ep_degree: int = 1) -> int:
    """单参数的 FSDP 后本地元素数——runtime 切分口径（P0-3，2026-07-23，mindformers
    commit 377c9c344）。

    runtime 按 TP/EP placement 后参数**首维**切 FSDP（parallelize.py:331-350）：
      - `shape[0] % divisor == 0` → 正常 shard：`numel // divisor`（整除时与旧 ceil 相等，
        全部真机锚点逐字节不变）。
      - 不可整除（dense 路径,或 ep=1 时 expert 随父层走 dense wrap）→ **整参
        replicate_params**（不做 FSDP,全量驻留;特殊小参数 :353-378 亦显式复制）——
        旧 total-numel ceil 假设（FSDP2 flat-param pad）被 runtime 源码推翻,ceil 对整参
        复制可低估 divisor 倍,并非 OOM 安全（audit §6）。
      - expert 独立 wrap（`is_expert and ep_degree>1`,:1109-1113 未传 replicate_params）
        不可整除 → **fail-loud**（runtime 无 replicate/pad 语义,不静默猜）。
    `dim0==0`（旧构造/测试 stub 无首维信息）→ 退回 total-numel 整除判定（可整除 shard/
    不可整除 replicate,与首维判定在无 TP 的常见形状下同值）。divisor<=1 → 全量。
    """
    if divisor <= 1:
        return w.local_numel
    d0 = getattr(w, "dim0", 0)
    if getattr(w, "is_expert", False):
        # expert 权重 runtime 按**扁平 grouped 存储**切（GroupedMLP weight1 [e·H, F]，
        # expert_parallel.py Shard(0) 作用在 e·H 维）——评估器 3D shape (E,H,F) 的首维 e 不是
        # runtime shard 维,故以 local_numel 整除为判定粒度（e·H·F ∝ e·H,常见形状同判）。
        key = w.local_numel
    else:
        key = d0 if d0 else w.local_numel
    if key % divisor == 0:
        return w.local_numel // divisor
    if getattr(w, "is_expert", False) and ep_degree > 1:
        raise ValueError(
            f"expert 权重 {w.name}（dim0={d0}, numel={w.local_numel}）不被 efsdp={divisor} 整除："
            f"runtime expert 独立 wrap（parallelize.py:1109-1113）无 replicate_params/pad 语义,"
            f"该配置真机会拒绝——fail-loud,不静默估算。")
    return w.local_numel                       # replicate_params：整参驻留


def _norm_save_names(resolved_ops) -> set:
    """norm-type op（layernorm/RMSNorm）**保留的输入**激活名集合——真机 fp32 compute 下保留输入的
    fp32 cast 供反向（profiler 的 Cast 大头）。**只标 op.saves**（保留的输入 fp32）：norm 输出会被
    下游 cast 回 bf16、不 fp32；且该 fp32 cast 是**反向保留量**，只进 activation_saves（no-recompute
    act_live），**不进 forward_max_live**（重算瞬态里 fp32 cast 转瞬即释、不共存,故 full 重算不受影响,
    DSv3 锚点不动）。

    **2026-07-29 按 norm 种类分辨（普查收口 Fix 2）**：抬升只对**真的 cast 输入**的 norm 成立。
    权威快照 `mindformers/pynative/layers/layer_norm.py`：
      · `FusedLayerNorm.construct:93-101` —— `x = self.cast(x, compute_type)`（:97）**真 cast** → 抬。
      · `FusedRMSNorm.construct:151-155` —— `output = self.norm(x, _to_local(self.weight), self.eps)[0]`，
        `x` **直通**；`self.cast`（:149）是**死属性**，`construct` 从不调用 → **不抬**。
        `layernorm_compute_dtype` 在 RMSNorm 里只决定 gamma 参数 dtype（:146）。
    故按 `ResolvedOp.norm_kind`（由 `OpSpec.norm_kind` ← `DimTable.norm_kind` ← yaml `normalization`
    经 `get_norm_cls`（:187-191）派生）逐 op 过滤，而**不是**整体翻 `norm_compute_dtype_bytes`
    ——后者是**跨模型**口径，翻它会把真的 cast 的 norm 一并关掉。默认 CASTING → 未标注的 op（含
    全部手搭测试 spec、mHC 非融合的 hc_norm）行为逐字节不变。"""
    out = set()
    for op in resolved_ops:
        tv = getattr(getattr(op, "type", ""), "value", getattr(op, "type", ""))
        # 仅 **layernorm/RMSNorm**（layernorm_compute_dtype 驱动）；**排除 softmax/logsoftmax**
        # ——它是 softmax_compute_dtype 的事、且 loss 区 logsm/probs 已显式建 fp32,勿重复放大 logits。
        if tv != "norm" or "softmax" in op.name.lower():
            continue
        if getattr(op, "norm_kind", NORM_KIND_CASTING) == NORM_KIND_NONCASTING:
            continue                      # FusedRMSNorm：输入直通，无 fp32 副本
        for s in op.saves:
            out.add(s.name)
    return out


def _dt(tensor, norm_names: set, norm_dtype: int) -> int:
    """张量字节 dtype:norm 保留的激活按 norm_compute_dtype（fp32）,其余按自身 dtype。"""
    if norm_dtype and tensor.name in norm_names:
        return max(tensor.dtype_bytes, norm_dtype)
    return tensor.dtype_bytes


def _forward_max_live(resolved_ops, blk: int, norm_names: set = frozenset(), norm_dtype: int = 0) -> int:
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
                byt[t.name] = _align_up(t.local_numel * _dt(t, norm_names, norm_dtype), blk)
                first[t.name] = i
            last[t.name] = i
    peak = 0
    for i, op in enumerate(ops):
        live = sum(b for n, b in byt.items() if first[n] <= i <= last[n])
        peak = max(peak, live + op.workspace_bytes)
    return peak


def _backward_max_live(resolved_ops, *, conservative: bool = False) -> int:
    """反向瞬态 `bwd_scratch` 的 **max-live** 峰值（P1-08，`_forward_max_live` 的反向镜像）。

    旧口径 `Σ op.bwd_scratch_bytes` 把一层所有 op 的反向临时物化**直接求和**，隐含它们同时存活
    → 保守高估（OOM 安全但不准，审计判「开放（接受的保守上界）」）。真机上 `bwd_scratch` 是
    **op 内瞬态**：在各自 op 的反向步物化、步末即释（loss 链 probs+grad 的真实共存已编码进**单个**
    `nll` op 的 `8·S·B·vocab`=2 份里，跨 op 之间并不共存）。故按 op 时序取 max-live 更准。

    **模型：逆序滑窗（window=2）——OOM 安全侧。** 反向按 forward 逆序执行（op n-1, …, 0）。
    forward-index 坐标下，op i 的反向 scratch 活性区间取 ``[i-1, i]``：在它自身反向步 i 物化，并
    **保守地**延续一步到其反向消费者（前一 forward op i-1 = 后一 backward 步）——bound 住「本 op 的
    反向 scratch 未在其消费者反向开始前释放」的握手情形。于是 backward 时间点 t 的存活和 =
    ``bwd_scratch[t] + bwd_scratch[t+1]``，峰 = **相邻两 op 之和的最大值**。

    为何选这个而非纯 max（各 op 完全独立、只取单 op 最大）：严格说 op 内 scratch 在其反向步末即释、
    跨 op 不叠加（纯 max 更紧、且理论上精确），但「宁可略保守也不欠估破 OOM 门」——window=2 对**相邻**
    scratch 保留一步握手余量（比纯 max 保守、比求和紧），且物理可辩护。三档退化：
      - **单 scratch op**（loss 的 `nll` / DSA·dsv4 的 `indexer`，唯一大 scratch）→ 无相邻非零 →
        = 该 op 值 = **sum**（**逐字节不变**：K_CE fat 锚点、DSv3/DSv4 golden 不破的机理根因）。
      - **分离的多 scratch op**（真实 mHC：`attn_hc`/`ffn_hc` sinkhorn 隔着整段 body，非相邻）→
        各自与零邻居配对 → 退化为**纯 max**（= max 单 op），< 求和。
      - **相邻的多 scratch op**（合成/极端）→ 相邻对和峰 < 全体求和。
    单调性：任意非负向量下 ``max_i(v_i+v_{i+1}) ≤ Σ v_i`` 且 ``≥ max_i v_i`` 恒成立 → OOM 安全、
    不塌到单峰以下。（未来可从 op 数据依赖判真实存活区间做区间并进一步收紧；当前 positional
    window=2 已足够安全且对全部真实模型退化为纯 max。）

    **conservative 模式（round3 A / F10，2026-07-16）**：window=2 只 bound「**相邻**两 scratch 共存」，
    对**非相邻的 >2 个大 scratch 同时存活**会欠估（反例 ``[4000,0,3000]``：window-2=4000，真实若三者
    共存则 7000）。现实模型全退化为单 scratch（loss/DSA 唯一大 scratch）故当前锚点不受影响；但作为
    OOM-安全**双模式**提供 ``conservative=True`` → 取 **Σ**（全 scratch 共存的严格上界，构造即安全）。
    默认 ``estimated``（window=2）——对全部真实模型退化为纯 max、逐字节复现旧行为；conservative 仅在
    用户显式要求"最保守上界"时启用（`HardwareSpec.bwd_scratch_conservative`）。二者恒 est ≤ cons。
    """
    vals = [getattr(op, "bwd_scratch_bytes", 0) for op in resolved_ops]
    n = len(vals)
    if n == 0:
        return 0
    if conservative:
        return sum(vals)                          # 全 scratch 共存严格上界（OOM 最保守侧）
    if n == 1:
        return vals[0]
    return max(vals[i] + vals[i + 1] for i in range(n - 1))


def estimate_structure_memory(
    resolved_ops,
    *,
    fsdp: int = 1,
    efsdp: int = 1,
    opt_state_bytes: int = 0,
    grad_dtype_bytes: int = 4,
    alloc_block_bytes: int = 1,
    norm_compute_dtype_bytes: int = 0,
    bwd_scratch_conservative: bool = False,
    matrix_opt_state_bytes: int = 0,   # Muon:2D 矩阵权重每元素持久字节(0=同 opt_state_bytes → uniform)
    ep_degree: int = 1,                # P0-3(2026-07-23):expert 独立 wrap 是否激活(>1)——不可整除时 fail-loud vs replicate
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
    # Muon:2D 矩阵权重名集（matmul/moe_gemm 的权重，排除 lm_head/embedding）——分类需 op 上下文，
    #   故在 walk 时按 op 类型/名判定（resolved 权重 shape 为 None，不能靠 shape）。matrix_opt_state_bytes
    #   ==0（AdamW/uniform）时不分类、集恒空 → 逐字节复现旧值。
    muon_matrix_names: set = set()
    for op in resolved_ops:
        _is_muon_op = matrix_opt_state_bytes and is_muon_matrix_weight(op.type, getattr(op, "name", ""))
        for w in op.params:
            params[w.name] = w
            if _is_muon_op:
                muon_matrix_names.add(w.name)
        for s in op.saves:
            saves[s.name] = s

    # 持久 = param+opt（按 fsdp/efsdp 切）。**不可整除口径（P0-3 修，2026-07-23,runtime
    # 377c9c344）**：runtime 按 TP/EP 后参数**首维**判定 `shape[0] % shard_size`
    # （parallelize.py:331-350），不可整除 → **整参不做 FSDP、全量驻留**（replicate_params;
    # 特殊小参数 :353-378 亦显式复制）。旧「total-numel ceil（FSDP2 flat-param 补齐）」假设被
    # runtime 源码推翻（audit 报告 §6：ceil 对 replicate 整参可低估 K 倍,并非 OOM 安全）。
    # expert 独立 wrap（ep>1）runtime 未传 replicate_params（:1109-1113）→ 不可整除 fail-loud。
    # 整除时 numel//divisor 与旧 ceil 相等 → 全部真机锚点逐字节不变。见 `_fsdp_local_count`。
    persistent = 0
    persist_numel_matrix = persist_numel_other = 0   # 驻留参数计数(分量拆解用,见 static_mem.persistent_breakdown)
    if opt_state_bytes or matrix_opt_state_bytes:
        for w in params.values():
            divisor = efsdp if w.is_expert else fsdp
            # Muon:2D 矩阵权重用 matrix_opt_state_bytes(momentum-only,较小)，其余走 opt_state_bytes。
            #   AdamW/uniform 时 muon_matrix_names 空 → 全走 opt_state_bytes（逐字节不变）。
            _cnt = _fsdp_local_count(w, divisor, ep_degree)
            _osb = matrix_opt_state_bytes if w.name in muon_matrix_names else opt_state_bytes
            persistent += _align_up(_cnt * _osb, blk)
            if w.name in muon_matrix_names:      # 计数按 Muon-矩阵分类(AdamW 也分类,但同倍数;
                persist_numel_matrix += _cnt     #   分量拆解时由 static_mem 按 optimizer 类型决定是否折回)
            else:
                persist_numel_other += _cnt
    # norm 激活 fp32（真机 layernorm_compute_dtype=fp32 → 保留输入 fp32 cast，profiler 的 Cast 大头）
    norm_names = _norm_save_names(resolved_ops) if norm_compute_dtype_bytes else frozenset()
    activation_saves = sum(_align_up(s.local_numel * _dt(s, norm_names, norm_compute_dtype_bytes), blk)
                           for s in saves.values())
    # 重算免疫 saves（fused 自定义算子 ctx 集，见 StructureMemory docstring）：同一 dedup 字典、
    # 同块对齐;按张量自身 dtype（ctx 持有的是 kernel 实际张量，无 norm-fp32 cast 语义）。
    recompute_pinned_saves = sum(
        _align_up(s.local_numel * s.dtype_bytes, blk)
        for s in saves.values() if getattr(s, "pin_under_recompute", False))
    param_full_bytes = sum(_align_up(w.local_numel * w.dtype_bytes, blk) for w in params.values())
    grad_full_bytes = sum(_align_up(w.local_numel * grad_dtype_bytes, blk) for w in params.values())
    # P0-01（2026-07-14 review）：已规约梯度分片（step-scoped cumulative）。divisor 与上方 persistent
    # 完全同口径（dense÷fsdp、expert÷efsdp）；不可整除同按 P0-3 replicate_params 口径
    # （整参驻留,`_fsdp_local_count`——replicate 参数梯度不 reduce-scatter,全量 DDP all-reduce 后驻留）。
    grad_shard_bytes = 0
    for w in params.values():
        divisor = efsdp if w.is_expert else fsdp
        grad_shard_bytes += _align_up(
            _fsdp_local_count(w, divisor, ep_degree) * grad_dtype_bytes, blk)
    # P1-08：bwd_scratch 由「求和上界」精化为 backward **max-live**（逆序滑窗 window=2）。
    # 单 scratch op 层（loss 的 nll / DSA·dsv4 的 indexer）逐字节不变 == 旧 sum；分离/相邻的多
    # scratch op 层取更紧的 max-live（OOM 安全，≤ sum 恒成立）。见 `_backward_max_live` docstring。
    bwd_scratch = _backward_max_live(resolved_ops, conservative=bwd_scratch_conservative)
    workspace = max((op.workspace_bytes for op in resolved_ops), default=0)
    # bwd 期 kernel workspace：同一时刻至多一块在世（实测，见 StructureMemory.bwd_workspace）
    # → max。`getattr` 兜 0 让抽取图（`opdag.to_resolved.ROp` 无此字段）与旧构造逐字节不变。
    bwd_workspace = max((getattr(op, "bwd_workspace_bytes", 0) for op in resolved_ops), default=0)

    # 重算边界 checkpoint_input = 层「construct 入参」hidden_states（MS `_InputSaver`,
    # recompute.py:158，以 compute/bf16 dtype 持有），**不是**首个 op 恰好 save 的那张激活。
    # mHC 下首 op 是 RMSNorm：它 **save 的是 fp32 [S,B,n·H] 输出（256MiB）**，而吃进的 construct
    # 入参是 **bf16 打包流 [S,B,n·H]（128MiB）** —— 旧口径取 save 被 norm-fp32 污染成 256。取首个有
    # saves 的 op 的**首个非权重输入**（层入口张量、真实 dtype），无激活输入(embedding 类)时回退首个
    # save。非 mHC(std/MLA/DSv3) 首 op ln1 save 的即其输入 x → 逐字节不变；mHC 修正回 bf16 128。
    checkpoint_input = 0
    for op in resolved_ops:
        if op.saves:
            boundary = next((t for t in op.inputs if not getattr(t, "is_weight", False)), None)
            if boundary is None:
                boundary = op.saves[0]
            checkpoint_input = _align_up(boundary.local_numel * boundary.dtype_bytes, blk)
            break

    # forward_max_live（重算瞬态）**不用 norm fp32**：重算时 fp32 cast 转瞬即释、不与峰值共存,
    # full 重算 DSv3 锚点 12409.5 逐字节不动（只 no-recompute act_live 的 saved fp32 cast 长驻）。
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
        grad_shard_bytes=grad_shard_bytes,
        persist_numel_matrix=persist_numel_matrix,
        persist_numel_other=persist_numel_other,
        recompute_pinned_saves=recompute_pinned_saves,
        bwd_workspace=bwd_workspace,
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
    - ``remat_saves``：**重算再物化的 saved 集**（2026-07-25，167/MS2.10 微基准实测驱动）=
      `max over islands( max(0, activation_saves(island) − pinned_input_boundary(island)) )`。
      与 ``recomp_scratch`` 的区别是**语义**而非范围：``recomp_scratch`` 用 `forward_max_live`
      ——「前向中间量在最后一次被用完就死」的峰值工作集；但重算再执行的**目的**恰是重建 backward
      需要的那批 saved 张量，它们**不能**在段末死、必须活到该 island backward 消费完 → 同驻的是
      **整个 saved 集**。取 max over islands 的理由同 ``island_recomp``（反向逐 island 重算、算完
      释放，不同时存活）。两端退化：全选 → `A(层) − ci` == full 式；全不选 → 0 == None。

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
    # P1-07 checkpoint islands（Z1，2026-07-15）：选中 op 的**非连续区段**各是独立重算单元。
    #   n_islands   — 连续选中区段数（全选=1、全不选=0）。
    #   island_recomp — 每 island 的 recomp scratch（各扣**自己**的进入边界）；
    #                   recomp_scratch = max(island_recomp)（反向逐 island 重物化、算完释放、不同时存活）。
    n_islands: int = 0
    island_recomp: tuple = ()
    # 2026-07-25 尾部追加（default=0/()，序不破，照 island_recomp 先例）：重算再物化的 saved 集。
    #   island_remat — 每 island 的 `max(0, activation_saves(island) − pinned_input_boundary(island))`；
    #   remat_saves  — `max(island_remat)`（同 island_recomp 的 ×1 机理）。
    remat_saves: int = 0
    island_remat: tuple = ()


def _checkpoint_islands(ops, is_selected) -> list:
    """把一层 op 序列按**选中连续性**切成 checkpoint islands（P1-07，一等对象）。

    island = 极大连续选中子段。非连续的选中区段（如选 {a,c} 跳 b）→ 多个 island，各是独立的重算
    单元：反向逆序逐 island 重物化其内部激活、算完释放 → 各 island 的 recomp scratch **不同时存活**
    → 峰取 max over islands（见 estimate_select_memory），而非把它们当一整块混算（旧式把某 island
    的进入边界从另一 island 的峰值里错减 → 低估 → OOM 不安全）。

    返回 `list[list[op]]`：全选→单 island（= 旧「一整块」，逐字节复现）；全不选→ `[]`。"""
    islands: list = []
    cur: list = []
    for op in ops:
        if is_selected(op):
            cur.append(op)
        elif cur:
            islands.append(cur)
            cur = []
    if cur:
        islands.append(cur)
    return islands


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


def estimate_select_memory(resolved_ops, is_selected, *, alloc_block_bytes: int = 1,
                           norm_compute_dtype_bytes: int = 0) -> SelectMemory:
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

    sm_sel = estimate_structure_memory(selected, alloc_block_bytes=blk,
                                       norm_compute_dtype_bytes=norm_compute_dtype_bytes)
    sm_non = estimate_structure_memory(nonselected, alloc_block_bytes=blk,
                                       norm_compute_dtype_bytes=norm_compute_dtype_bytes)

    # 层入口 checkpoint_input（重算边界，始终保留）：层「construct 入参」= 第一个有 saves 的 op
    # 的**首个非权重输入**（真实 bf16 dtype，与主 checkpoint_input 同口径；非 mHC 逐字节不变，
    # mHC 从 fp32 污染修正回 bf16 128）；无激活输入时回退首个 save。
    ci_name = None
    ci_bytes = 0
    for op in ops:
        if op.saves:
            _b = next((t for t in op.inputs if not getattr(t, "is_weight", False)), None) or op.saves[0]
            ci_name = _b.name
            ci_bytes = _align_up(_b.local_numel * _b.dtype_bytes, blk)
            break

    # act_live_pinned = 非选中 saves（去重）∪ {ci}。ci 若已在非选中 saves 里则不重复加（按名去重）。
    nonsel_save_names = {s.name for op in nonselected for s in op.saves}
    act_live_pinned = sm_non.activation_saves + (ci_bytes if ci_name not in nonsel_save_names else 0)

    # ── P1-07 checkpoint islands（Z1，2026-07-15）：选中 op 按连续性切成独立重算单元 ──────────
    # 反向逐 island 重物化其内部激活、算完释放 → 各 island recomp **不同时存活** → 峰 = max over
    # islands。每 island 各扣**自己**的进入边界（已 pin 部分，避免与 act_live 双算；未 pin 的细粒度
    # 单 op 边界保留在该 island recomp，OOM 安全）。
    #   - 单 island（连续选中 / 模块 / 全选 / 单 op）：islands=[selected] → 逐字节复现旧「一整块」式
    #     `forward_max_live(选中集) − pinned_boundary(选中集)`（→ 12 锚点连续 island 不动的机理根因）。
    #   - 多 island（非连续）：旧式把某 island 的进入边界从**合并峰值**里错减 → 低估（OOM 不安全）；
    #     新式各 island 只从**自身峰**减**自身边界** → 修正。
    # pinned = 非选中 saves ∪ {层 ci}（ci 恒常驻）。
    pinned_names = nonsel_save_names | ({ci_name} if ci_name is not None else set())
    islands = _checkpoint_islands(ops, is_selected)
    island_recomp = tuple(
        max(0, _forward_max_live(isl, blk) - _pinned_input_boundary(isl, pinned_names, blk))
        for isl in islands)
    recomp_scratch = max(island_recomp) if island_recomp else 0
    # ── 2026-07-25：重算再物化的 saved 集（`remat_saves`，spec §2.2）────────────────────────
    # `island_recomp` 用 `forward_max_live`——「中间量最后一次被用完即死」的峰值工作集；但重算再执行
    # 的**目的**是重建 backward 需要的那批 saved 张量,它们必须活到该 island backward 消费完 → 真正
    # 同驻的是**整个 saved 集**。故每 island 另计 `activation_saves(island) − 已 pin 的进入边界`
    # （边界扣法与 island_recomp 完全同源 `_pinned_input_boundary`,故全选时恰退化为 `A(层) − ci`
    # == full 路径公式 → select-all == full 逐字节保持）。saves 走 `estimate_structure_memory`
    # （全库唯一按名去重点,不另写一份求和）,含 norm-fp32 口径,与 act_live 同尺。
    island_remat = tuple(
        max(0,
            estimate_structure_memory(
                isl, alloc_block_bytes=blk,
                norm_compute_dtype_bytes=norm_compute_dtype_bytes).activation_saves
            - _pinned_input_boundary(isl, pinned_names, blk))
        for isl in islands)
    remat_saves = max(island_remat) if island_remat else 0
    # bwd_working_set = 非选中段 forward_max_live − 非选中 bwd_scratch（与无重算同构，范围收窄）。
    bwd_working_set = max(0, sm_non.forward_max_live - sm_non.bwd_scratch)

    return SelectMemory(
        act_live_pinned=act_live_pinned,
        recomp_scratch=recomp_scratch,
        bwd_working_set=bwd_working_set,
        n_islands=len(islands),
        island_recomp=island_recomp,
        remat_saves=remat_saves,
        island_remat=island_remat,
    )
