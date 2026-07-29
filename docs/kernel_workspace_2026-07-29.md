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

## 5. 逐层驻留：插桩跑 vs 2026-07-29 直测（第二重扰动对照）

（见 §9 表；本节由 `analyze_events.py` 在本跑的 `events_rank<N>.csv` 上复算。）

---

## 6. 建模（待补：见 §6.x）

---

## 7. 仍未解释的（诚实清单）

（待补）
