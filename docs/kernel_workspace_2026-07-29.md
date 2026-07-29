# Kernel workspace：真机实测 → 建模（2026-07-29）

> 承接 [`census_fix_mhc_rmsnorm_2026-07-29.md`](census_fix_mhc_rmsnorm_2026-07-29.md) §3 ①
> ——那一轮把三个层型的逐层驻留残差（r0 −69.6 / r4 −228.6 / r128 −78.5 MiB/层）**归因为
> kernel workspace**，并写明「快照里读不出，要闭合需真机 profiler 的 kernel workspace 明细」。
> 本轮就是去做那个测量。
>
> **纪律不变**：绝不编造真机数；绝不发明拟合常数；「我跑了并观察到」与「我推断」分开写。
> `REAL*` 常数与 `REAL_SHA256` 指纹**一个字节未动**——本轮的真机数是**新增**测量，不替换任何旧值。

---

## 0. 一句话结论（先说最重要的，因为它推翻了上一轮的假设）

**测出来了，但它不在上一轮以为的位置。**

| 上一轮的假设 | 本轮实测判决 |
|---|---|
| 逐层**驻留**残差（−69.6/−228.6/−78.5 MiB/层）= kernel workspace | **证伪。** 全部 4554（rank0）/5565（rank2）个 workspace 块**寿命恒为 1 个 tracker tick**，在任一 `fwd_exit` 边界上**存活的 workspace 字节数 = 0.00 MiB**（逐层逐微批 8 个边界全查）。workspace **不进驻留**，故它**不可能**解释驻留欠读。 |
| kernel workspace「读不出来、只能留着」 | **可测且已测。** MindSpore 自带 memory tracker 给出逐块 (start,end,size,type,owning-task)；单卡微基准上它逐字节复现 `max_memory_allocated()` 的瞬态。 |
| — | **新发现：真机峰值本身就是 workspace 顶出来的。** rank0 峰 30307.8 MiB、rank2 峰 21019.5 MiB，**两者都由同一笔 730.00 MiB 的瞬态 workspace 分配设定**（该分配落地那一刻 pool 用量正好等于全程 high-water）。模型把 workspace 记 0 → **结构上够不着真机峰**。这才是 OOM-不安全的机制。 |

---

## 1. 测量方法与它的局限

### 1.1 工具

MindSpore 2.10 内置 **memory tracker**：`MS_ALLOC_CONF="memory_tracker:True,memory_tracker_path:<dir>"`。
每个 rank 落 `rank_<N>/{memory_block.csv, task.csv, tracker_graph.ir}`：

- `memory_block.csv` —— **逐次内存池分配**一行：`start_time_stamp,end_time_stamp,size,
  actual_used_memory,actual_peak_memory,type,producer_task,node_name,…`
- `task.csv` —— 逐次 kernel launch 一行：`time_stamp,task_name,node_name,file_name,…`

探针 `run_memprobe_tracker.py`（服务器 `/home/suhaibo/workspace/`）是
`run_memprobe_events.py` 的**严格超集**：同一采样线程、同一
`[MEMPROBE] rank=N peak_alloc_MiB=` 收尾行、同一 `events_rank<N>.csv`，故峰值与
2026-07-25 基线**直接可比**。新增三件事：

1. **逐 rank** 开 tracker 并指到**逐 rank 目录**（8 个 worker 否则互相覆盖）。
   `MS_ALLOC_CONF` 在 `import mindspore` **之前**设好。
2. **`Erfinv` 标记**。`mint.erfinv` 在 `mindformers/` 与 `hyper-parallel/` 全仓 grep **零命中**，
   故 `task.csv` 里每一条 `Erfinv` 都是本探针种下的边界。逐层 fwd/recompute enter+exit、
   逐 pipeline step enter+exit 各一枚；`markers_rank<N>.csv` 按同序记全 (iter,micro,stage,
   step_type,layer,phase)。**序号配对 + 相位类校验**（4/8/12/16 KiB 四种操作数尺寸）。
   实测 rank0 `Erfinv` 任务 162 条 == 标记 162 条（rank2 210 == 210）→ 配对无歧义。
3. **rank 选择**`PROBE_TRACKER_RANKS`（本轮 `0,2`）。其余 rank 逐字节走未插桩的
   `run_memprobe_events.py` 代码路径；因 `dp_shard=2` 令 rank 2k 与 2k+1 同 stage 同层，
   **rank 1 / rank 3 就是 rank 0 / rank 2 的同跑对照**。

**服务器共享源码一行未改**（全部 monkeypatch 在探针文件内）。

### 1.2 workspace 的判据（以及它是怎么被验证的）

`type` 列的取值实测为 `PyNativeOutput / (空) / PyNativeInput / WorkSpace / Other /
ConstantValue / Weight`。判 workspace：

> `type == "WorkSpace"` **或** （`type` 为空 **且** 寿命 ≤ 2 tick）。

**验证（单卡微基准 `probe_ws0.py`，Ascend 910B2 / MS2.10）**：同一段代码里，
「空 type、寿命 1 tick」的块尺寸**逐字节复现** `max_memory_allocated()` 在该 op 期间
高出「输入+输出存活集」的瞬态：

| op | tracker 里的空-type 块 | `max_memory_allocated()` 实测瞬态 |
|---|---:|---:|
| `MatMulExt [4096,4096]×[4096,4096] bf16` | 20972544 B = **20.000 MiB** | **20.00 MiB** |
| `RmsNorm [4096,4096]` | 16777728 B = **16.000 MiB** | **16.00 MiB** |
| `TopkExt k=512 on [4096,2048]` | 8391168 B = **8.002 MiB** | **8.00 MiB** |
| `FlashAttentionScore B1 N64 S4096 D128` | 79693312 B = **76.000 MiB** | 92.00 = 76 ws + 16（softmax_max/sum 两个瞬时输出） |

**端到端复核（真机 8 卡跑）**：`max(actual_used_memory)` over `memory_block.csv`
= **30307.8 MiB**（rank0）/ **21019.5 MiB**（rank2），与探针独立读到的
`ms.runtime.max_memory_allocated()` 收尾值**逐位相同**。即 tracker 的块集合与
分配器计数器**是同一本账**。

### 1.3 局限（必须写在结论前面）

1. **`node_name` 对自定义融合算子是错的（stale-task 标签）。**
   `task.csv` 只记 110 种 kernel，**没有任何一个** `hyper_parallel` 自定义融合算子
   （无 `Mhc*`、无 `SparseFlashMla*`、无 `Indexer*`，也无 `FlashAttentionScore`）——
   它们走 pyboost `op_runner.h` 之外的 launch 路径，不进 `task.csv`。于是它们的
   **输出块与 workspace 块都被记到"上一条 pyboost 任务"名下**。
   证据（rank0, tick 29620-29631，逐行原文见 `analysis4_r0.txt`）：一条名义上的
   `Concat` 任务，其 `PyNativeOutput` 依次是
   `0.000 / 32.000 / 0.063 / 0.250 / 0.063 / 0.375 / 0.016 / 2.500 / 10.000 MiB`
   ——**逐张与融合 mHC `npu_mhc_pre_sinkhorn` 的 8 个输出对上**
   （`h_in` 32.0、`h_post` 0.0625、`h_res` 0.25、`h_pre` 0.0625、`hc_before_norm` 0.375、
   `inv_rms` 0.0156、`sum_out` 2.5、`norm_out` 10.0 —— 见普查收口报告 §0.1 的字节表）。
   **故：块的尺寸与时刻是硬数据，"哪个 kernel"对自定义算子须靠输出签名反推，不能信 `node_name`。**
2. 判据把「任何寿命 ≤2 tick 的无类型池块」都算作 workspace。真机跑里
   **未出现**长寿命无类型块（rank0/rank2 各 0 个），故本轮该判据无歧义；换配置需重验。
3. tracker 只看**设备内存池**。池外分配（HCCL 自带 buffer 等）不在账内——但 §1.2 的
   端到端复核说明本跑的 `max_memory_allocated` 全部落在池内。
4. 标记 `Erfinv` 本身是真算子（4–16 KiB），会占极少量时间/显存；扰动量见 §2。

---

## 2. 扰动对照（强制项）

`c` = fused / 无重算 / L8 / m4 / seq4096 / b=1，`dsv4h_fused_pp4_norecomp_ab.yaml`。
基线 = 2026-07-25 的 `c` 跑（`log_ab_fusion_2026-07-25`）。

| rank | stage | 本轮插桩？ | 本轮 `peak_alloc_MiB` | 2026-07-25 基线 | 偏差 |
|---|---|---|---:|---:|---:|
| 0 | 0 | **tracker+markers** | 30307.8 | 30391.6 | **−0.28 %** |
| 1 | 0 | 无（同 stage 对照） | 30395.0 | 30391.6 | +0.01 % |
| 2 | 1 | **tracker+markers** | 21019.5 | 21019.4 | **+0.0005 %** |
| 3 | 1 | 无（同 stage 对照） | 21010.2 | 21019.4 | −0.04 % |
| 4 | 2 | 无 | 17759.3 | 17759.5 | −0.001 % |
| 5 | 2 | 无 | 17799.9 | 17759.5 | +0.23 % |
| 6 | 3 | 无 | 27720.1 | 27720.1 | **0.00 %** |
| 7 | 3 | 无 | 27821.0 | 27720.1 | +0.36 % |

**判读**：两个插桩 rank 的偏差（−0.28 % / +0.0005 %）**小于未插桩 rank 的跑间散布**
（rank5 +0.23 %、rank7 +0.36 %）。tracker 只在**主机侧**记账，不改设备分配；标记算子
4–16 KiB。**结论：无实质扰动，本跑的数可用于下结论。**
（对照 2026-07-29 事件 tracer 的先例：那次 `g` stage0 被扰动 +2.23 %，根因是强引用
pin 住了其它 stage 的层；本探针沿用其修好的版本，`PROBE_BWD_HOOKS=0` 时不 pin。）

逐层驻留（`alloc(fwd_exit) − alloc(fwd_enter)` 中位数）与 2026-07-29 的逐层直测比对见 §5。

---

## 3. 实测表：workspace 是**瞬态**，不进驻留

`analyze_ws.py` 输出（`log_ws_2026-07-29/analysis_rank0.txt`）：

```
RANK 0   tasks=31827 blocks=24508 markers=162
  block type histogram: [('PyNativeOutput',18906), ('',4554), ('PyNativeInput',668),
                         ('WorkSpace',193), ('Other',132), ('ConstantValue',50), ('Weight',5)]
  untyped-block lifetime (ticks): [(1, 4554)]          <-- 全部恰好 1 tick
  untyped LONG-LIVED blocks: n=0
  WORKSPACE blocks: n=4747  total_alloc=214259.9 MiB  max_single=2228.00 MiB
```

**驻留检查**（每个 `fwd_exit` 标记那一刻还活着的 workspace 块）：

```
layer=0 iter=0 micro=0 : workspace-class blocks still alive = 0  (0.00 MiB)
layer=1 iter=0 micro=0 : workspace-class blocks still alive = 0  (0.00 MiB)
... 8/8 个边界全为 0 ...
```

> **因此：`census_fix_mhc_rmsnorm_2026-07-29.md` §3 ① 的归因是错的。**
> 逐层驻留欠读 −69.6 / −228.6 / −78.5 MiB/层 **不是** kernel workspace。
> 那部分至今**仍未归因**——见 §7。

---

## 4. 实测表：workspace 顶出了峰值

`analyze_ws2.py` §A（`analysis2.txt`）：

| rank | stage | pool high-water | 设定它的那笔分配 | 是 workspace？ | 该刻存活的 workspace |
|---|---|---:|---|---|---:|
| 0 | 0 | **30307.8 MiB** | 730.00 MiB，BWD 中 | **是** | 730.00 MiB |
| 2 | 1 | **21019.5 MiB** | 730.00 MiB，BWD 中 | **是** | 730.00 MiB |

- rank0：`actual_used_memory` 最高的 400 个采样中，**97 个**由 workspace 分配设定。
- rank2：同上 **94 个**。

**这笔 730.00 MiB 是谁的**：按 §1.3 的输出签名反推（rank0 tick 36248–36265 原文）——
紧邻它的那一批「名义 Reshape」输出是
`256.0 / 4.0 / 1.0 / 8.0 / 64.0 / 0.25 / 1.0 / 8.0 MiB`，逐张对上
`csa.py` 融合稀疏 flash-MLA **反向**产出的梯度集
（`query`[B,S,64,512]bf16 = 256、`ori_kv` = 4、`cmp_kv`(r=4) = 1、`query_index` = 64、
`key_index` = 1 …，见仲裁报告 §1.10 的 ctx 表）。
即 **730.00 MiB = 融合稀疏 flash-MLA 反向 kernel 的 workspace**，r4 层独有；rank0（层 0/1）
与 rank2（层 2/3）各有一个 r4 层，两边都测到同一个 730.00 MiB，互为佐证。

### 4.1 逐相位 workspace 极大值（单 kernel）

因为**每块 workspace 只活 1 tick**，任一时刻**至多一笔** workspace 在世（峰值那一刻实测
存活 workspace 块 = 1 笔）。故「该相位的单 kernel workspace 极大值」就是可加到驻留之上的量。

| step_type | rank0 (stage0: r0+r4) | rank2 (stage1: r128+r4) |
|---|---:|---:|
| **FWD** | **304.001 MiB** | **304.001 MiB** |
| **BWD** | **2228.003 MiB** | **730.000 MiB** |
| FSDP_REDUCE_GRAD | 256.002 MiB | 64.002 MiB |
| FWD_SEND / RECV / FSDP_UNSHARD / RESHARD | 0.000 | 0.000 |

### 4.2 逐（层，相位）窗口

| rank | layer | 层型 | 窗口数 | 单 kernel `max_ws`（中位） | 窗口内 workspace 分配总和（中位） |
|---|---|---|---:|---:|---:|
| 0 | 0 | r0 dense | 12 | **304.001 MiB** | 2254.0 MiB |
| 0 | 1 | r4 moe | 12 | **304.001 MiB** | 2521.3 MiB |

（「总和」只作参考——它们**不共存**，OOM 相关的是 max。）

### 4.3 前向 304.001 MiB 是谁的

rank0 layer 0 前向窗口原文（`analysis3_r0.txt`）：

```
t=29616 TASK Erfinv            <-- fwd_enter 标记
t=29618 TASK View
t=29619 TASK Reshape
t=29620 TASK Concat            <-- 名义标签；其"输出"是 mHC 融合 kernel 的 8 个输出
t=29622..29630  blk 0.000/32.000/0.063/0.250/0.063/0.375/0.016/2.500/10.000 MiB
t=29631 >>> WORKSPACE  304.00 MiB
t=29632 TASK Reshape ... t=29635 TASK RmsNorm ...
```

输出签名逐张 == `npu_mhc_pre_sinkhorn` 的 ctx/输出集（普查收口 §0.1）
→ **304.001 MiB = 融合 mHC pre-sinkhorn kernel 的前向 workspace**。
同一模式在该层窗口内出现**两次**（t=29631 与 t=29922，后者紧接 `RmsNorm→MatMulExt→
SplitWithSizeView→SiLU→Mul→MatMulExt` 即 dense FFN）→ 对应一层两次 mHC 调用
（attn 侧 + ffn 侧），**每次 304.001 MiB**，三个层型同值。

### 4.4 全跑最大单笔 workspace

`GatherDGradV2` **2228.003 MiB**，rank0 出现 24 次（= 4 micro × 3 iter × 2），
rank2 同名 kernel 只有 16.03 MiB。`GatherDGradV2` 是**真 pyboost 算子**（在 `task.csv` 里），
故此处 `node_name` 可信。它落地时 pool 用量 21587–30135 MiB，**未**设定 high-water
（峰仍是那笔 730.00）。

### 4.5 shape 可移植性检验：**不成立**

`analyze_ws2.py` §C 逐 kernel 算 `workspace / 该 op 输出字节`：

| kernel | n | min | median | max |
|---|---:|---:|---:|---:|
| GatherDGradV2 | 24 | 1.103 | 127.763 | 127.763 |
| Reshape（含 stale 标签） | 228 | 0.000 | 0.100 | 4.080 |
| Concat | 918 | 0.250 | 0.500 | 6.715 |
| Contiguous | 97 | 0.000 | 0.000 | 524292 |
| HistcExt | 36 | 32771 | 32771 | 32771 |
| MatMulExt | 384 | 0.078 | 0.625 | 4.500 |

比值跨越六个数量级、同一 kernel 内也不稳定 → **「workspace = k(op_type) × 输出字节」
这个可移植模型不被数据支持**。不编。改为 §6 的做法。

---

## 5. 第二重扰动对照：逐层驻留被逐 MiB 复现

tracker 的块台账可以**独立重建**「逐层驻留」这个量：

> `residency(层) = 该层 fwd 窗口内诞生、且在 fwd_exit 那一刻仍然活着的块字节之和`

这与 2026-07-29 事件 tracer 的定义（`memory_allocated()` 在 fwd_exit 减 fwd_enter）是
**两条完全独立的读法**（一条读分配器计数器、一条数内存池块台账）。实测
（`analyze_resid.py`，每层 12 个窗口 = 4 微批 × 3 迭代，取中位）：

| rank | layer | 层型 | 本轮 tracker 重建 | 2026-07-29 逐层直测（存量、未改） | 差 |
|---|---|---|---:|---:|---:|
| 0 | 0 | r0 | **2235.1** | 2235.1 | **0.0** |
| 0 | 1 | r4 | **2356.1** | 2355.3 | +0.8（+0.03 %） |
| 2 | 2 | r128 | **2124.9** | 2124.2 | +0.7（+0.03 %） |
| 2 | 3 | r4 | **2354.2** | 2354.0 | +0.2（+0.01 %） |

r0 **逐位相同**，其余三层 ≤ 0.03 %。这同时证明：① 插桩没有改变逐层驻留；
② `Erfinv` 标记的窗口划分是对的（否则不可能凑出这个数）。

### 5.1 驻留里根本没有 workspace —— 逐块可查

`analyze_resid.py` 列出的「仍然活着」的块**全部**是 `PyNativeOutput` 型；
`analyze_ws.py` 的驻留检查在 8/8 个 `fwd_exit` 边界上给出
`workspace-class blocks still alive = 0 (0.00 MiB)`。两条口径一致。

> **所以逐层驻留欠读（今天：r0 −5.6 / r4 −260.6 / r128 −110.5 MiB/层）与 kernel workspace 无关。**
> 上一轮把它归因为 workspace 是**错的**；本轮用测量推翻它，并给出替代定位（§5.2）。

### 5.2 那 −260.6 MiB（r4）在哪：用块直方图定位到 indexer 链

同一 rank 内比较 r128 与 r4（rank2 的 layer 2 vs layer 3），排除 stage/深度差异：

| 量 | r128 (layer 2) | r4 (layer 3) | 差 |
|---|---:|---:|---:|
| 实测驻留（中位） | 2124.9 | 2354.2 | **+229.3** |
| **64.0005 MiB 块 / 窗口** | **4** | **7** | **+3（= +192.0 MiB）** |
| 32.0005 MiB 块 / 窗口 | 7 | 7 | 0 |
| 256 / 128 MiB 块 / 窗口 | 4 / 2 | 4 / 2 | 0 |
| `<1 MiB` 长尾 / 窗口 | 3.0 | 5.3 | +2.3 |
| **手写普查 `activation_saves`** | **2005.6** | **2080.6** | **+75.0** |

**普查只解释了 229.3 中的 75.0**。缺的 ≈ 154 MiB，实测形态是**三块 64.0005 MiB 中的
约 2.4 块**——`64.0005 MiB` 在本站点尺寸下正是 `[B, S, dsa_indexer_n_heads, index_head_dim]`
bf16（`1×4096×64×128×2 B = 64 MiB`），即 **indexer 链的 query/key 侧张量**。
普查目前只建了其中一张（`idx_query` 64 MiB）。

> 这条线索**本轮不落地**：它属于「普查该建哪些张量」，与本轮的 workspace 口径正交，且
> `dsv4_hybrid.py` 的 indexer 普查此刻正被另一条任务线动。**如实交接**：真机每个 r4 层
> 在 indexer 链上驻留 **3 块 64.0005 MiB**，普查只有 1 块；差额与 r4 相对 r128 的
> 额外欠读（−260.6 vs −110.5 = −150.1）**同量级同方向**。

---

## 6. 建模：一条**加法**的 bwd kernel workspace 通道

### 6.1 为什么不能复用 `bwd_scratch`

`mem_timeline` 对无重算层用
`bwd_working_set = max(0, forward_max_live − bwd_scratch)`
→ 往 `bwd_scratch` 里加字节是**零和**（总量 = `max(bwd_scratch, fml)`），加不进峰值。
而实测的 kernel workspace 是 kernel 向池现要现还的 scratch，**叠在整个反向工作集之上**。
故新开一条通道，落到 `Buckets` 里**独立的 `workspace` 桶**（`total()` 的加项）。

### 6.2 改了什么（全部尾部追加、默认 0 → 未标注者逐字节不变）

| 文件 | 改动 |
|---|---|
| `cost_eval/model_spec.py` | `OpSpec.bwd_workspace`（字符串表达式）+ `bwd_workspace_ref`（TensorRef） |
| `cost_eval/shape_eval.py` | `ResolvedOp.bwd_workspace_bytes`（尾部追加，照 `norm_kind` 先例）；字符串项**刻意不吃**「含 S 就 ÷cp」那条规则（会把常数项也除掉 → cp>1 欠读），线性项走 TensorRef 由 `resolve_tensor` 只对首个 S 维 ÷cp |
| `cost_eval/structure_mem.py` | `StructureMemory.bwd_workspace = max(op.bwd_workspace_bytes)`。**取 max 是实测判据不是保守假设**：块寿命恒 1 tick、峰值那一刻只有一块在世 |
| `cost_eval/mem_timeline.py` | BWD 事件 `B.workspace = sm.bwd_workspace`，`rec()` 后清零 |
| `cost_eval/layers/dsv4_hybrid.py` | `sparse_attn` 挂实测律，**双重门** `fused and enable_indexer` |
| `tests/test_bwd_kernel_workspace.py` | **新增 10 道守卫门**（见 §6.5） |

### 6.3 挂上去的值 —— 逐条测量出处

```python
_FLASHMLA_IDX_BWD_WS = {
    "bwd_workspace":     "209715200",                                   # 200.0 MiB 截距
    "bwd_workspace_ref": TensorRef("flashmla_idx_bwd_ws",
                                   ("B", "S", "33920"), dtype_bytes=4), # B·S·135680 B
}
```

**它只标在一个 op 上**（`sparse_attn`），因为测量把它定位到了一个 kernel；
没有把任何数字摊到层上、也没有为了让总量落位而调过一个字节。

**没测出来的一律留 0**：`r0`（滑窗）、`r128`（无 indexer 的 `FusedSparseFlashMla`）、
`unfused`（小算子链）。两个插桩 rank 的 BWD 单 kernel 极大值都被 r4 那一笔盖住 →
它们各自的值本轮**没有单独测出来**，如实欠读。

### 6.4 逐层型 before → after（实测律 vs 模型）

| 层型 | `bwd_workspace` before | after | 真机实测（同 kernel） | 判 |
|---|---:|---:|---:|---|
| r4（seq 4096） | 0 | **730.000 MiB** | **730.000 MiB** | 逐字节 |
| r4（seq 2048） | 0 | **465.000 MiB** | **465.000 MiB** | 逐字节 |
| r4（seq 1024） | 0 | **332.500 MiB** | **332.500 MiB** | 逐字节 |
| r128 / r0 | 0 | 0 | **未测** | 如实留空 |

（复现：`PYTHONIOENCODING=utf-8 python scratchpad/probe_bwd_workspace.py`）

### 6.5 新增的 10 道门（守的是**测量**，不是模型自洽）

`tests/test_bwd_kernel_workspace.py`：① 三个 seq 的实测律逐字节穿过
`builder → mhc_wrap → ShapeEval → StructureMemory`；② 归属只在 `sparse_attn` 这一个 op 上
（挡「把层级常数摊在层上」）；③ r0/r128/unfused 恒 0（挡「拿 r4 的数去顶」）；
④ 未标注 spec 默认 0；⑤ `workspace` 是 `total()` 的独立加项；
⑥ **`mhc_wrap` 不得静默吞字段**——2026-07-29 真实发生过一次（`_rebuild` 的手写字段清单
把 `workspace_ref` 在全部 DSv4 层上丢掉），本门用「被 mHC 包装的生产层」取数，回潮必红。

---

## 7. 验收

### 7.1 八跑门 before → after

**在隔离 worktree 里各跑一遍**（共享工作树同时被另一条任务线编辑，读数会串；
`git worktree add --detach <dir> HEAD` @ `96209b1`，before = 裸 HEAD、after = HEAD + 本 patch）。
两处读数与共享树一致，故本表可信。

| 指标 | before | **after** |
|---|---|---|
| bucket 聚合 mean / min / max（n=28） | 0.821 / 0.700 / 1.017 | **0.836 / 0.700 / 1.023** |
| hand_spec·chain2 mean / min / max | 0.855 / 0.670 / 1.110 | **0.855 / 0.670 / 1.110（逐字节不变）** |
| hand_spec·dataflow | 0.795 / 0.659 / 1.067 | **同上，逐字节不变** |
| run `c` bucket 逐 stage sim/real | 0.905 / 0.919 / 0.857 / 0.878 | **0.929 / 0.953 / 0.898 / 0.878** |
| run `a` s1 / s2 | 0.813 / 0.826 | **0.863 / 0.878** |
| run `g` s1 | 0.941 | **1.023**（转为轻度过读 = OOM 安全侧，如实记） |
| run `b`/`d`/`f`/`h`（unfused 四跑） | — | **逐 MiB 不变**（该 kernel 只在 fused 分支存在） |
| 两条真机不变量 / 模型 ×1 不变量 / `REAL_SHA256` | PASS | **PASS（一条未动）** |

`hand_spec` 两行**逐字节不变**是本次改动**只动一条口径**的证据：
`cost_eval/liveness/` 按 saves + grad 可达性自建图，不读 `bwd_workspace_bytes`。

run `c` 的 s0/s1/s2 各**恰好** +730.0 MiB —— 就是实测值本身，没有第二个数字介入。
s3 不动，因为它的峰值事件在 head/loss（`bwd@9/bwd4:nll`）。

### 7.2 记分卡锚点

**14 个锚点一个都没动**（`python scorecard_anchors.py` 逐条比对：DSv4-fused 0.9017、
mHC+MTP 0.8914、DSv3 系列全同）。原因**已定位**：这两个 DSv4 锚点的峰值事件是
`bwd@5` / `bwd@6` = **lm_head 段的反向**（`bwd_scratch = 2020.0 MiB` 满 vocab fp32 主导），
不是 decoder 层的反向 → 本轮加在 decoder 层上的 workspace 顶不到它们的峰。
**如实记：本轮没有改善这两个 OOM-不安全锚点。** 要改善它们，需要测 **head/loss 段**
（`GatherDGradV2` 那一档，见 §4.4）的 kernel workspace 并挂到 head 侧 op 上——那是
`layers/head.py`，本轮不属本人可改范围。

### 7.3 测试

```
python -m pytest tests -q   →  1903 passed, 268 warnings
```

基线 **1893**（`96209b1`，含另一条任务线的 7 个提交）+ **10 道新守卫门**。
**没有删除任何一条既有不变量**；新增用例守的是**本轮新引入的机制**（新字段 +
新桶相位），照 O4 `norm_kind` 的先例。

### 7.4 重钉台账（old → new，逐条理由）

**未动**：`REAL*` / `CSV*` 一切真机常数、`REAL_SHA256` 指纹表、两条真机不变量与模型 ×1
不变量、run d 不可评分规则、`nr_moe_frag_factor` / `kept_frag_factor` 两个标定 margin
（**一个字节没动**），以及全部 14 个记分卡锚点。

| 位置 | old → new | 理由 |
|---|---|---|
| `test_acceptance_gate::GOLDEN_BUCKET` | fused 四跑（a/c/e/g）的部分格 **+730.0 整**；unfused 四跑（b/d/f/h）**逐字节不变** | 实测值本身；该 kernel 只在 fused 分支 |
| `...::GOLDEN_AGG` bucket 两行 | 0.821/0.700/1.017 → **0.836/0.700/1.023** | 欠读被真实补上一块；max 上移来自 `g` s1 由 0.941 转 1.023 |
| `...::GOLDEN_AGG` hand_spec 两行 | **未动** | liveness 不读该字段 |
| `test_pp4_recompute_anchor::THEO_ON` | s2 12161.9 → **12448.1**（+286.2） | s0/s1/s3 的峰值事件不在 r4 层 bwd 上；s2 峰值事件因此易主，净上移 286.2 而非 730 |
| `...::THEO_MTP` | s2 同上；s3 28875.0 → **29605.0**（+730.0 整） | MTP 尾 stage 峰值正落在 r4 层 bwd 上 |
| `...::THEO_PP8` | s1 13463.7 → **14193.7**；s4 13079.7 → **13809.7**（各 +730.0 整） | 1 层/stage，只有落在 r4 层且峰值事件在其 bwd 上的 stage 会动 |
| `...::_PP8_OVER_BAND` | (1.00, 1.25) → **(1.00, 1.29)** | s4 由 1.204 → **1.271**。上移**全部**来自一笔实测值；下界仍保 1.00（真翻欠读必须变红） |
| `test_probe185_recon::test_p3p_m8_theoretical_and_gap` | 17966.2 → **18696.2**（+730.0 整） | 同上 |
| `tests/test_bwd_kernel_workspace.py` | **新增**（10 例） | 新机制的守卫门，见 §6.5 |

---

## 8. 仍未解释 / 未测的（诚实清单）

| # | 缺口 | 量 | 为什么没闭合 / 闭合它需要什么 |
|---|---|---:|---|
| ① | **逐层驻留欠读** r0 −5.6 / r4 −260.6 / r128 −110.5 MiB/层 | 见左 | **不是 workspace**（本轮实测推翻）。已定位到 indexer 链的 **3 块 64.0005 MiB**（普查只有 1 块，§5.2）。属普查口径，本轮不动。 |
| ② | **r128 / r0 的反向 kernel workspace** | 未知（≤ 730） | 两个插桩 rank 的 BWD 单 kernel 极大值都被 r4 那一笔盖住。闭合需**逐层 backward 窗口**——把 `PROBE_BWD_HOOKS=1` 的逐层反向钩子与 tracker 同时打开（该钩子本身有已知扰动，须配对照跑）。 |
| ③ | **前向 304.001 MiB**（融合 mHC pre-sinkhorn kernel，每层 2 次） | 每层 max 304.001 | 已测、已归属（输出签名逐张匹配，S=4096/2048 两点确认 **S 无关**），但它的 op 在 `cost_eval/layers/residual.py` —— **本轮不属本人可改范围**。模型当前 fwd `workspace` 只有 16.0 MiB（`_fa_workspace`），欠读 288 MiB/层。**交接项**。 |
| ④ | **head/loss 段 2228.003 MiB**（`GatherDGradV2` = 词表反向，输出 2020.0 MiB == `vocab×H×fp32`） | 2036.0 + S×49152 B（三点线性，斜率 0.046875 MiB/token 逐位相等） | 已测、律已验证，但归属在 embedding/lm_head 段（`layers/head.py` / embedding builder）——**不属本人可改范围**。这正是两个 OOM-不安全 DSv4 锚点（峰在 `bwd@head`）没被改善的原因。**交接项，且是当前最大的单笔未建模 workspace**。<br>**→ 2026-07-30 已闭合，见 [`head_workspace_2026-07-30.md`](head_workspace_2026-07-30.md)。**两点订正：①它是 **word-embedding 的反向**，**不是** head/loss 段（该文 §2 三条证据；本行标题里的「head/loss 段」措辞不准确，S 扫描本身就能把二者分开）；②本行的律是 **H=4096 的特例**，通式为 `4·vocab·H + 16 MiB + 12·H·(B·S) + 3072 B`，已在第二个站点（H=1792/B=2）逐字节复核。**⚠ 由此，两个 DSv4 锚点仍未被改善**——它们的峰在 `bwd@head`，而这笔 workspace 落在 `bwd@<embedding>`。 |
| ⑤ | 实测律在 **B / n_heads / index_topk / v_head_dim** 上的缩放 | — | 只扫了 S（3 点）。B 只测过 1（模型含 B 是 OOM-安全的那一侧）；其余维**未扫**，故系数 135680 B/token **不承诺**随它们变化（它也不能被这些维整除分解：135680 = 512×265，265 = 5×53）。闭合需按维扫描。 |
| ⑥ | tp>1 / cp>1 下的真值 | — | tp：TensorRef 不标 shard → 不切 = **过读 = OOM 安全侧**；cp：常数项/线性项已分开处理（结构正确）但**未实测**。 |
| ⑦ | `node_name` 对自定义融合算子不可信 | — | 它们不进 `task.csv`（走 pyboost 之外的 launch 路径）。本轮靠**输出签名 + S 减半逐张对折**反推，两个 kernel 都对上了；但这是**推断**，不是 tracker 直接告诉我们的。要直接归属需 CANN 侧 profiler 的 kernel 名。 |

**一句话**：本轮把「kernel workspace」从**假设**变成了**测量**——测出它在哪、多大、随什么变，
证明它**不在驻留里**、却**恰恰顶着峰值**；并把其中**归属明确且在本人范围内**的那一笔
（融合稀疏 flash-MLA 反向，三点验证的实测律）建进了模型。**没测的一律留 0 并逐条记账。**

---

## 9. 服务器侧产物（可复现）

`192.168.9.167:/home/suhaibo/workspace/`（探针与分析脚本，**共享源码一行未改**）：

| 文件 | 作用 |
|---|---|
| `probe_ws0.py` | 单卡微基准：验证「空 type + 1 tick」判据 == `max_memory_allocated()` 瞬态 |
| `run_memprobe_tracker.py` | 8 卡 tracker 探针（`run_memprobe_events.py` 的严格超集 + 逐 rank tracker + `Erfinv` 标记） |
| `run_ws_matrix.sh` | 顺序驱动（同 `run_events_matrix.sh` 的启动配方，`PYTHONPATH` **后缀**追加） |
| `analyze_ws.py` / `ws2` / `ws3` / `ws4` / `analyze_resid.py` | 逐 op / 逐相位 / 峰值邻域 / 原始流 / 逐层驻留分解 |
| `dsv4h_fused_pp4_norecomp_s{1024,2048}.yaml`、`gen_dsv4h_data_1024.py`、`dsv4h_data_1024/` | seq 扫描的两个新配置与数据集（**新增文件，未改任何既有配置**） |
| `log_ws_2026-07-29/` | 三次跑的全部原始 CSV 与分析文本 |

```bash
# 三次跑（每次约 100 s，8 卡；跑前先 npu-smi info 确认空闲）
cd /home/suhaibo/workspace && ./run_ws_matrix.sh \
  "c_ws_norecomp_L8_m4:dsv4h_fused_pp4_norecomp_ab.yaml:9360:0,2" \
  "s2048_ws_norecomp_L8_m4:dsv4h_fused_pp4_norecomp_s2048.yaml:9362:0,2" \
  "s1024_ws_norecomp_L8_m4:dsv4h_fused_pp4_norecomp_s1024.yaml:9364:0,2"
docker exec shb_dsv4 bash -lc "cd /home/suhaibo/workspace && \
  python3 analyze_ws2.py log_ws_2026-07-29/c_ws_norecomp_L8_m4 0,2"
```
