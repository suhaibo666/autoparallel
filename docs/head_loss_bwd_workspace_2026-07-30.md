# `lm_head` / loss **反向** kernel workspace：实测 → 建模（2026-07-30）

> 承接 [`mhc_fwd_workspace_2026-07-30.md`](mhc_fwd_workspace_2026-07-30.md) §6 ① 与
> [`head_workspace_2026-07-30.md`](head_workspace_2026-07-30.md) §6 ① ——
> 那两轮都把「**`lm_head` / loss 段自己的**反向 kernel workspace」列为
> **未测 → 留 0、已知欠读**，并判它是「当前最要紧的待测项」：量到事件线的 36 条锚点/stage 里，
> 峰值事件落在 **lm_head 反向**的有 **15 条**，其中 **13 条 OOM-不安全**。本轮去测它。
>
> **纪律不变**：绝不编造真机数；绝不发明拟合常数；「我跑了并观察到」与「我推断」分开写。
> `REAL_*` / `CSV_*` 常数与 `REAL_SHA256` 指纹**一个字节未动**（§8 有过滤 diff 佐证）。

复现（服务器侧）：

```bash
# ① 判据自检（单卡，head/loss 那一族 kernel）
docker exec shb_dsv4 bash -lc 'cd /home/suhaibo/workspace && python3 probe_ws_head0.py'
python3 analyze_ws_head0.py /home/suhaibo/workspace/log_ws_head0
# ② 8 卡扫描（S / vocab / H 三轴），跑前 npu-smi 确认 8 卡空闲；wait_and_run_head.sh 是有界等待器
cd /home/suhaibo/workspace && ./wait_and_run_head.sh \
  "S2048_V129280:dsv4h_fused_pp4_norecomp_s2048.yaml:9422:6,4:129280" ...
python3 head_ws_table.py <NAME>=<dir>:<S>:<vocab>:<H>:<B> ...
# ③ 单卡密集 shape 图（确定 operand-copy 选择规律）
docker exec shb_dsv4 bash -lc 'cd /home/suhaibo/workspace && python3 probe_head_matmul_ws.py'
python3 analyze_head_matmul_ws.py log_head_matmul_ws
```

---

## 0. 一句话结论

| | 结论 |
|---|---|
| **测到了，而且它很大** | `lm_head` 反向的两个 `MatMulExt`（dgrad / wgrad）各带**约 1 GiB** 的 kernel workspace。dgrad 那一笔在 DSv4-hybrid 站点（S=4096, vocab=129280, H=4096, B=1）实测 **1147143168 B = 1094.001 MiB**，是该 rank **整个 BWD 相位的单 kernel 极大值**。 |
| **律是 vocab 扫过的**（前作没有一条是） | `bwd_ws(lm_head dgrad) = (2·vocab + 4·H)·(B·S) + 20 MiB + 1024 B`，在 **16 个 vocab=129280 的点**（2 套独立采集 × 6 个 H × 3 个 S × 2 个 B）**逐字节**吻合，并在 **10 个 vocab 值**上成立。 |
| **但 vocab 扫描同时**证伪**了「光滑律」这个假设** | vocab=**64640** 那一点实测只有 `4·H·(B·S) + 20 MiB + 1024`（**`2·vocab` 那一项整项消失**）。这不是噪声——单卡密集扫描逐字节复现。即 operand-copy 是 **kernel 选择**，不是 shape 的光滑函数。**任务书让扫 vocab 是对的：不扫就会把一条会崩的律当成定律。** |
| **⚠ 它不设定该站点的峰** | 同一窗口里，pool high-water 是在 **30 个 tick 之前**由 `_NLLLoss.backward` 的 `ScatterAddExt`（只有 **16.017 MiB** workspace）叠在 **4 张共存的满 vocab fp32 平面**上设定的。1094 MiB 那一笔落地时 pool 离 high-water 还有 **4950.0 MiB**。故本项在既有 `bwd_scratch + max(workspace)` 组合下是**上界**，不是时间分辨的真值——见 §5 的双计分析。 |

---

## 1. 测量方法

### 1.1 工具与探针（复用 2026-07-29 那条链）

MindSpore 2.10 内置 memory tracker：`MS_ALLOC_CONF="memory_tracker:True,memory_tracker_path:<dir>"`，
逐 rank 落 `memory_block.csv`（每次池分配一行：start/end tick、size、type、producer_task、node_name）
与 `task.csv`（每次 kernel launch 一行）。

`run_memprobe_tracker_head.py` **不是抄一份**，而是 `import run_memprobe_tracker as T` 之后
monkeypatch —— 基座探针（采样线程、同一句 `[MEMPROBE] rank=N peak_alloc_MiB=` 收尾行、
同一 `events_rank<N>.csv`、逐层/逐 pipeline-step 的 `Erfinv` 标记）与 2026-07-29 那一轮
**逐字节同一份代码**，故峰值与 2026-07-25 基线直接可比。

**新增的只是标记点**（把 head/loss 段的反向从 tick 流里切出来）：

| 标记 | 机制 | 落点 |
|---|---|---|
| `logsm_fwd_{enter,exit}` / `logsm_bwd_{enter,exit}` | `_Function` **子类**（`mindformers…loss._LogSoftmax` 的子类，`backward` 里种标记），并把 `_LogSoftmaxModule.construct` 换成调子类 | `loss.py:125-152` |
| `nll_fwd_{enter,exit}` / `nll_bwd_{enter,exit}` | 同上，`_NLLLoss` 的子类 | `loss.py:155-186` |
| `head_fwd_{enter,exit}` | 包 `Linear.construct`（按 `output_size == vocab_size` 认 head） | `linear.py:105` |
| `head_in_bwd` | 一个**恒等 `_Function`** 拼在 head 的**输入**上，其 `backward` 种标记 → head 的 dgrad 一落地就触发 = head 反向的远端边界 | `linear.py:105` |

两种机制**先在单卡上验过才用**（`probe_ws_head0.py` PART 2）：
`M1 子类标记 fired=True grad 逐位相同=True`、`M2 恒等标记 fired=True grad 逐位相同=True`。

**这些 patch 只装在 TRACKED rank 上**（`if T.TRACKED:`）→ 其余 rank 逐字节走 2026-07-29 的
代码路径，仍是有效的同跑对照。**服务器共享源码一行未改**。

本轮跟踪 **rank 6（stage 3 = head+loss+层 6,7）与 rank 4（stage 2 = 层 4,5）**。
`compress_ratios=[0,4,128,4,128,4,128,4]` → 层 4/6 都是 r128、层 5/7 都是 r4，
即 **rank 4 与 rank 6 层型完全相同、只差有没有 head** → rank 4 是**层型对照**；
rank 5 / rank 7 是同 stage 的**未插桩对照**。

### 1.2 workspace 判据 —— 在 head/loss 这一族 kernel 上重新验一遍

判据沿用 2026-07-29：`type == "WorkSpace"` **或**（`type` 为空 **且** 寿命 ≤ 2 tick）。
任务书要求「先验判据再信任任何站点数」，故本轮**没有**沿用旧验证，而是在
**head/loss 段自己会 launch 的那一族 kernel** 上重做（单卡 `probe_ws_head0.py`，
N=4096 / V=129280 / H=4096，tracker 与 `max_memory_allocated()` 同进程读同一段代码）：

| 用例（单 op） | tracker 里的「空 type + 1 tick」块 | `max_memory_allocated()` 瞬态 | delta |
|---|---:|---:|---:|
| `matmul [N,H]×[H,V]` bf16（head fwd） | 1113589760 B = **1062.0020 MiB** | **1062.0020 MiB** | **0 B** |
| `matmul [N,V]×[V,H]` bf16（head dgrad 形） | 2206204928 B = **2104.0010 MiB** | **2104.0010 MiB** | **0 B** |
| `matmul [H,N]×[N,V]` bf16（head wgrad 形） | 1113589760 B = **1062.0020 MiB** | **1062.0020 MiB** | **0 B** |
| `max [N,V] fp32 dim1` | 17408 B = 0.0166 MiB | 0.0166 MiB | **0 B** |
| `sum [N,V] fp32` | 1536 B = 0.0015 MiB | 0.0015 MiB | **0 B** |
| `gather [N,V] fp32 dim1` | 16795136 B = 16.0171 MiB | 16.0171 MiB | **0 B** |
| `cast` / `exp` / `mul` [N,V] | 0 | 0.0000 | **0 B** |

三个「有 gap」的用例**全部**是我的 lambda 里含两个 op、gap 逐字节等于那个**中间张量**：
`sub`（含 `max`）差 512 B；`exp(neg(·))` 差 **2118124032 B = 2020.0005 MiB**（= `Neg` 的
`PyNativeOutput`，寿命 6 tick）；`scatter_add`（含 `zeros_like`/`sub_scalar`）差 **16896 B**
（= `ZerosLikeExt` 的输出）。**这三个 gap 正是判据该做的区分**——它把
「kernel 向池要的 scratch」与「算子输出张量」分开，而这一条正是本轮双计分析（§5）的支点。

同跑块普查：`untyped lifetime histogram = [(1, 26), …]`，**26 个无类型块寿命恒 1 tick**；
另有 2 个「永不释放」的无类型块（≤2.0 MiB，H2D staging）。

### 1.3 归因方式（`node_name` 对自定义融合算子不可信，本轮不依赖它）

`MatMulExt` / `ScatterAddExt` / `InplaceAddExt` / `RmsNormGrad` **都是真 pyboost 算子**
（在 `task.csv` 里），故它们的 `node_name` 可信（2026-07-29 §1.3 的 stale-label 限制只针对
`hyper_parallel` 自定义融合算子）。但本轮**仍然不靠名字断言归属**，而是三条独立证据叠加：

1. **标记窗口**：`nll_bwd_enter` → `head_in_bwd` 之间。
2. **输出签名**：dgrad 的 `PyNativeOutput` = **33554944 B**（= `S·B·H·2 + 512`）；
   wgrad 的 = **1059062272 B**（= `H·vocab·2 + 512`）。两者逐字节对上。
3. **rank 4 层型对照**：同层型、无 head 的 rank 4 上，**这三笔一个都不出现**
   （rank 4 的 BWD 单 kernel 极大值是 **730.0005 MiB**，即 2026-07-29 已建模的
   r4 融合稀疏 flash-MLA —— 顺带**逐字节复现**了那一轮的数）。

### 1.4 逐 tick 原文（rank 6，iter 1 / micro 2；`analyze_head_ws.py` 的 C-raw 段）

```
t=49731 TASK Erfinv                         <-- nll_bwd_enter
t=49735 TASK Neg      blk 2118124032 B = 2020.0005 MiB  life=4
t=49737 TASK Exp      blk 2118124032 B = 2020.0005 MiB  life=12
t=49744 TASK ScatterAddExt
t=49747      blk -    16795136 B = 16.0171 MiB life=1   <<< WORKSPACE  (= pool high-water)
t=49752 TASK Mul      blk 2118124032 B = 2020.0005 MiB  life=15
t=49758 TASK Erfinv                         <-- nll_bwd_exit
t=49762 TASK Erfinv                         <-- logsm_bwd_enter
t=49765 TASK Cast     blk 1059062272 B = 1010.0005 MiB  life=21        (_LogSoftmax.backward)
t=49768 TASK Erfinv                         <-- logsm_bwd_exit
t=49774 TASK MatMulExt
t=49776      blk PyNativeOutput   33554944 B =   32.0005 MiB           (dgrad 输出 [S,B,H] bf16)
t=49777      blk -              1147143168 B = 1094.0010 MiB life=1 <<< WORKSPACE
t=49781 TASK MatMulExt
t=49783      blk PyNativeOutput 1059062272 B = 1010.0005 MiB           (wgrad 输出 [H,vocab] bf16)
t=49784      blk -              1113589760 B = 1062.0020 MiB life=1 <<< WORKSPACE
t=49789 TASK Erfinv                         <-- head_in_bwd
t=49791 TASK InplaceAddExt
t=49793      blk -              1059064320 B = 1010.0024 MiB life=1 <<< WORKSPACE  (head 权重梯度累加)
t=49797 TASK RmsNormGrad
t=49799      blk -                16779264 B =   16.0020 MiB life=1 <<< WORKSPACE  (final_norm 反向)
t=49803 TASK InplaceAddExt
...  +687 tick 之后才出现第一笔解码层的 workspace（Reshape 730.0005 MiB = r4 融合 flash-MLA）
```

**模型侧 `bwd@<head>` 事件 = head 层 = `final_norm + lm_head + logsoftmax + nll`**
（`cost_eval/layers/head.py:190-197`）→ 上面这一整段（含 `RmsNormGrad`）正好是它，
**+687 tick 之后**才进解码层。窗口边界不含糊。

**K-profile**（把窗口右端从 `head_in_bwd` 往后推 K 个 tick，看极大值变不变）：

```
K = 0 / 5 / 10 / 20 / 40 / 80 / 160 / 320 / 640
max = 1147143168 B = 1094.0010 MiB (MatMulExt)   —— 九个 K 全同
```

即**本项对窗口右端的选择完全不敏感**（S=4096 站点）。

---

## 2. 多点表（逐字节实测，非转抄）

`head_ws_table.py`，rank 6，每格是 12 个窗口（4 微批 × 3 迭代）的极大值。
`S`/`vocab`/`H` 三轴各自单独扫；**除该轴外全部配置逐字节相同**
（`dsv4h_fused_pp4_norecomp_ab.yaml` = fused / 无重算 / L8 / m4 / pp4·dp2·ep2 / B=1 / tp=1 / cp=1）。

| 跑 | S | vocab | H | dgrad ws (B) | wgrad ws (B) | 权重梯度累加 ws (B) | nll ws (B) | final_norm ws (B) | **层内 max** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S4096 V129280 H4096 | 4096 | 129280 | 4096 | **1147143168** | 1113589760 | 1059064320 | 16795136 | 16779264 | **1147143168** |
| S2048 | 2048 | 129280 | 4096 | 584057856 | 567281664 | 1059064320 | 16786944 | 16779264 | 1059064320 |
| S1024 | 1024 | 129280 | 4096 | 302515200 | 294127616 | 1059064320 | 16782848 | 16778752 | 1059064320 |
| V64640 | 4096 | **64640** | 4096 | **88081408** | 20972544 | 529533440 | 16795136 | 16779264 | 529533440 |
| V32320 | 4096 | **32320** | 4096 | 352846848 | 285737984 | 264768000 | 16795136 | 16779264 | 352846848 |
| H2048 | 4096 | 129280 | **2048** | 1113588736 | 1096812544 | 529533440 | 16795136 | 16779264 | 1113588736 |
| H1792 | 4096 | 129280 | **1792** | 1109394432 | 1094715392 | 463342080 | 16795136 | 16779264 | 1109394432 |

同表（MiB）：dgrad 1094.0010 / 557.0010 / 288.5010 / **84.0010** / 336.5010 / 1062.0010 / 1058.0010。

### 2.1 律与逐字节 delta

```
dgrad :  (2·vocab + 4·H)·(B·S) + 20971520 + 1024      ← 本轮建模的那一条
wgrad :  (2·vocab + 2·H)·(B·S) + 20971520 + 2048
accum :  2·vocab·H + 2560                              ← head 权重梯度累加（S 无关）
nll   :  4·(B·S) + 16777216 + 1536                     ← _NLLLoss.backward 的 ScatterAddExt
```

| 跑 | dgrad delta | wgrad delta | accum delta | nll delta |
|---|---:|---:|---:|---:|
| S4096 V129280 H4096 | **0** | **0** | **0** | **0** |
| S2048 | **0** | **0** | **0** | **0** |
| S1024 | **0** | **0** | **0** | **0** |
| V64640 | **−529530880**（= −2·vocab·B·S） | −563086336 | **0** | **0** |
| V32320 | **0** | −33555456（= −2·H·B·S−1024） | **0** | **0** |
| H2048 | **0** | **0** | **0** | **0** |
| H1792 | **0** | **0** | **0** | **0** |

即 dgrad 律在 **6/7 个 8 卡点上逐字节成立**，唯一例外是 vocab=64640。

### 2.2 第二套独立采集：单卡密集 shape 图（并逐字节复现 8 卡站点）

`probe_head_matmul_ws.py` 在**单卡**上按站点源码逐字复刻 head 的调用
（`linear.py:132-135`：`weight=transpose(w,1,0)` → `matmul(x3d, weight)`），用 `ms.grad`
走**真 autograd**，故那两个 `MatMulExt` 就是站点上的同两个 kernel。

**先看可信度**：与 8 卡站点共有的 **7 个 shape，7/7 逐字节相同**
（1147143168 / 88081408 / 352846848 / 584057856 / 302515200 / 1109394432 / 1113588736）。

`analyze_head_matmul_ws.py` 把每一笔解成 `c_V·N·vocab + c_H·N·H + base + tail`
（`N = B·S`，`c_V,c_H ∈ {0,1,2,4,8}`，`base ∈ {0, 20 MiB}`，穷举精确解）：

| shape (N, vocab, H) | dgrad (c_V,c_H,base,tail) | wgrad (c_V,c_H,base,tail) |
|---|---|---|
| 4096, 129280, 4096 | **2,4,20 MiB,1024** | 2,2,20 MiB,2048 |
| 4096, **64640**, 4096 | **0**,4,20 MiB,1024 | 0,0,20 MiB,1024 |
| 4096, 32320, 4096 | **2,4,20 MiB,1024** | 2,0,20 MiB,1024 |
| 2048 / 1024, 129280, 4096 | **2,4,20 MiB,1024** | 2,2,20 MiB,2048 |
| 4096, **16160 / 24240**, 4096 | 2,**0**,20 MiB,1024 | 2,0,20 MiB,1024 |
| 4096, 40400 / 48480 / 56560, 4096 | **2,4,20 MiB,1024** | 2,0,20 MiB,1024 |
| 4096, 72720 / 80800 / 96960 / 113120 / 121200, 4096 | **2,4,20 MiB,1024** | **4**,2,**0**,2048 |
| 4096, 129280, **1792 / 2048 / 2560 / 3072 / 5120 / 6144** | **2,4,20 MiB,1024** | 2,2,20 MiB,2048 |
| **8192**, 129280, **1792 / 4096**（B=2） | **2,4,20 MiB,1024** | 2,2,20 MiB,2048 |

**dgrad 一列：23 个 shape 里 21 个是同一个解 `(2, 4, 20 MiB, 1024)`。**
例外只有 vocab=64640（`c_V` 掉到 0）与 vocab ≤ 24240（`c_H` 掉到 0，此时本律**过读**）。

### 2.3 第三套独立采集：仓内既有 profiler（H=1792、B=2、另一次 campaign）

`analysis/realmachine/pp2_norecomp/op_816365.csv`（**持有 lm_head 的那个 rank**，DSv3 8L
无重算 pp2/dp1，H=1792、vocab=129280、S=4096、B=2 → `N=8192`；该 campaign 用 AdamW，
与本轮的 Muon 不同，是**另一个 build**）里 `Name == MatMulExt` 的分档：

| Size(KB) | 字节 | n | 判 |
|---:|---:|---:|---|
| 4214785.5 | **4315940352** | 3 | 瞬态（Duration 17–24 µs）→ workspace，**= `4·N·vocab + 4·N·H + 20 MiB + 1536`** |
| 2168457216/1024 = 2117634 | **2168457216** | 3 | **= `(2·vocab + 2·H)·N + 20 MiB + 2048`** —— **wgrad 律逐字节跨站点成立** |
| 2118124032 | 2118124032 | 3 | `2·N·vocab + 512` = 长寿的 logits bf16 本体（不是 workspace） |
| 463340032 | 463340032 | 3 | `2·H·vocab + 512` = wgrad 输出 `[H,vocab]` bf16 |
| 29360640 | 29360640 | 75 | `2·N·H + 512` = dgrad 输出 `[N,H]` bf16 |
| 20972544 | 20972544 | **159** | **`20 MiB + 1024` 的裸 base**，在 H=1792 站点出现 159 次 → **`20 MiB` 与 H 无关**（独立佐证） |

**wgrad 律因此跨 2 个站点、2 个 H、2 个 B、3 个 S 共 8 点中 6 点逐字节成立。**

但 **dgrad 在这一套采集上是 `c_V = 4`**（比本律多整整一份 `2·N·vocab` bf16 拷贝 = 2020 MiB）。
**这不是本律的反例、也不是本律的修正**，而是**另一个 build 的另一个 kernel 选择**：
本轮单卡在**完全相同的 (N=8192, H=1792, vocab=129280)** 上量到的是 **2197816320 B
= 本律的值**（见 §2.2 末行）。故：**同一 shape、不同 build，`c_V` 取 2 或 4。**
如实记为 §7 ② 的欠读。

---

## 3. 归属：这一笔属于哪个 op

| 实测 kernel | tick 位置 | 模型侧 op | 本轮是否建模 |
|---|---|---|---|
| `MatMulExt` dgrad **1094.0010 MiB** | `logsm_bwd_exit` 之后第 6 个 tick | **`lm_head`** | **✔ 建了** |
| `MatMulExt` wgrad 1062.0020 MiB | 同上，再 +7 tick | `lm_head`（同 op） | 被 dgrad 盖住（层内取 max），如实记 |
| `InplaceAddExt` 权重梯度累加 1010.0024 MiB | `head_in_bwd` +4 tick | —— | **✘ 不建**，理由见 §7 ① |
| `ScatterAddExt`（nll 反向）16.0171 MiB | nll 窗口内 | `nll` | ✘（远小于层内 max，建了也不改变任何量） |
| `RmsNormGrad`（final_norm 反向）16.0020 MiB | `head_in_bwd` +8 tick | `final_norm` | ✘（同上） |

**层内取 max 不是保守假设，是实测判据**：本跑 rank 6 的 5384 个无类型块**寿命恒 1 tick**
（`untyped lifetime = [(1, 5382), (2, 1), (4, 1)]`），峰值那一刻在世的 workspace 只有一块。

---

## 4. 扰动对照（强制项）

### 4.1 峰值 vs 2026-07-25 基线（run `c`，`dsv4h_fused_pp4_norecomp_ab.yaml`）

| rank | stage | 本轮插桩？ | 本轮 `peak_alloc_MiB` | 2026-07-25 基线 | 偏差 |
|---|---|---|---:|---:|---:|
| 0 | 0 | 无 | 30391.6 | 30391.6 | **0.000 %** |
| 1 | 0 | 无 | 30395.0 | 30391.6 | +0.011 % |
| 2 | 1 | 无 | 21019.4 | 21019.4 | **0.000 %** |
| 3 | 1 | 无 | 21009.8 | 21019.4 | −0.046 % |
| **4** | 2 | **tracker + 全部标记** | **17759.0** | 17759.5 | **−0.003 %** |
| 5 | 2 | 无（同 stage 对照） | 17799.9 | 17759.5 | +0.228 % |
| **6** | 3 | **tracker + 全部标记（含 head 恒等标记）** | **27720.4** | 27720.1 | **+0.001 %** |
| 7 | 3 | 无（同 stage 对照） | 27821.0 | 27720.1 | +0.364 % |

**判读**：两个插桩 rank 的偏差（−0.003 % / +0.001 %）比未插桩 rank 的跑间散布
（rank5 +0.228 %、rank7 +0.364 %）**小两个数量级**。tracker 只在主机侧记账；
新增标记是 20–60 KiB 的 `Erfinv` 与一个**恒等** `_Function`（单卡验过 grad 逐位相同）。
**结论：无实质扰动。**（对照先例：2026-07-29 的事件 tracer 曾把 `g` stage0 扰动 +2.23 %。）

### 4.2 逐层驻留（`alloc(fwd_exit) − alloc(fwd_enter)` 中位，12 窗口/层）

| rank | stage | 层 | 层型 | 本轮 | 同 stage 未插桩对照 | 2026-07-29 逐层直测 |
|---|---|---|---|---:|---:|---:|
| **4** | 2 | 4 | r128 | **2105.0** | 2144.0（rank5） | 2116.1 |
| **4** | 2 | 5 | r4 | **2370.4** | 2341.1（rank5） | 2341.2 |
| **6** | 3 | 6 | r128 | **2119.3** | 2129.5（rank7） | 2116.1 |
| **6** | 3 | 7 | r4 | **2285.8** | 2425.6（rank7） | 2341.2 |

r128 两个插桩值（2105.0 / 2119.3）与参考 2116.1 差 ≤ 0.5 %。
r4 那一列散布大（2285.8 … 2425.6，**两个未插桩 rank 之间就差 3.6 %**）——
MoE 路由随数据变、专家占用不同，这是**该量固有的散布**，不是插桩造成的；
**扰动的决定性证据是 §4.1 的峰值（≤0.01 %）**，如实记。

---

## 5. ⚠ 双计分析：哪些字节已经在既有 `bwd_scratch` 里

任务书点名要判这一条。**判决：本项与既有 `bwd_scratch` 在字节上完全不重叠，但在时间上不共存。**

### 5.1 字节上不重叠 —— 逐块可查

模型的 head 层 `bwd_scratch = 8·S·B·vocab`（`head.py:187`），并在无重算的 loss 层被改写成
`sm.bwd_scratch // 2 * (K_CE − 1)` = `(K_CE−1)` 份 `4·S·B·vocab` 的**满 vocab fp32 平面**
（`mem_timeline.py:671-677`）。真机 high-water 那一刻（tick 49747）在世的 ≥ 64 MiB 块：

```
2118124032 B = 2020.0005 MiB  PyNativeOutput  Contiguous      life=14023   <- saved logsm（在 act_live）
2118124032 B = 2020.0005 MiB  PyNativeOutput  Log             life=121
2118124032 B = 2020.0005 MiB  PyNativeOutput  Neg             life=12
2118124032 B = 2020.0005 MiB  PyNativeOutput  ScatterAddExt   life=8
size histogram of live blocks >=256 MiB: [(2118124032, 4), (1059062272, 6), (529531392, 2), (268435968, 15)]
```

**4 张满 vocab fp32 平面共存**（1 张 saved + **3 张瞬态**）。它们全部是 `PyNativeOutput` 型；
本项那一笔（1147143168 B）是 `type` 为空、寿命 1 tick 的**另一个块**，在**另一个 tick**。
**故字节不重叠、不双计。**（§1.2 的判据自检正是把这两类分开的那一步。）

> **顺带一条独立发现（不折进本项，如实单列）**：真机 loss 层反向**共存 3 张瞬态满 vocab fp32
> 平面**（+1 张 saved）。模型的 `K_CE − 1` 在 `ce_pynative_lean` 或 `pp==1` 时取 **4−1 = 3**
> —— 与实测**逐张吻合**；但 pp4 这条路走 `pp>1 且非 lean` → `8−1 = 7` 份，比实测**多 4 份
> = 8080 MiB**。这属于 `bwd_scratch` 自己的标定问题，**本轮一个字节没动**
> （任务书要求：若既有 `bwd_scratch` 本身错了，单独列为一条发现，不要静默折进来）。
> 详见 §7 ③。

### 5.2 时间上不共存 —— 量出了余量

`actual_used_memory`（该笔分配落地那一刻的池用量）与全程 high-water 的差：

| 跑 | high-water | dgrad 1094 MiB 落地时池用量 | **余量** | accum 1010 MiB 落地时 | 余量 |
|---|---:|---:|---:|---:|---:|
| S=4096 | 27720.4 | 22770.4 | **4950.0** | 22654.4 | 5066.0 |
| S=2048 | 21737.9 | 19264.9 | 2473.0 | 20206.9 | 1531.0 |
| S=1024 | 19025.1 | 17554.1 | 1471.0 | **19025.1** | **0.0** |

即：在 **S=4096（全部锚点用的 shape）** 上这一笔**不设定峰**，峰在它之前 30 个 tick 由
`ScatterAddExt`（16.017 MiB workspace）叠在 4 张 fp32 平面上设定；
在 **S=1024** 上反而是**权重梯度累加那一笔**恰好设定了峰（余量 0.0）。

**所以必须说清楚**：既有通道的语义是
`peak(BWD 事件) = max_t live(t) + max_t ws(t)`，而真值是 `max_t [live(t) + ws(t)]`
——前者恒 ≥ 后者，本站点的差最多 **4950.0 MiB**。这是该通道**既有**的性质
（2026-07-29 为 r4 建的那一笔同理），本轮**没有改动它**；本项因此是
**上界、OOM-安全侧**，不是时间分辨的真值。**锚点被抬起的一部分是「两个上界互相抵消欠读」，
不是「模型现在能分辨时间线了」。** 这一条写在结论里，不藏在附注里。

---

## 6. 建模

（本节在实测落位后填写，见 §6.1 起。）

---

## 7. 诚实清单：本测量**不**支持什么

| # | 事项 | 状态 |
|---|---|---|
| ① | **参数梯度累加（`InplaceAddExt`）的 workspace** | **实测到、律已定（`2·vocab·H + 2560`，在 3 个 H × 3 个 vocab 共 5 点上逐字节成立），但本轮刻意不建。** 理由：它**不是 head 专有**——同一 rank 上解码层的参数也各有一笔（S=4096 时 `InplaceAddExt 67111424 B = 64.0024 MiB` ×24；S=1024 时另有 `33556992 B` ×12），即「每个参数的梯度累加都带一份自身大小的 scratch」。只挂在 head 上是**误归属**；要闭合它需要一条**逐参数**的新通道，超出本轮范围。**影响面**：仅当 `2·vocab·H > (2·vocab+4·H)·(B·S) + 20 MiB` 时它才是层内 max，即本站点 `B·S ≲ 3775`；今天全部锚点的 head 段 `B·S ≥ 4096`，故**在用锚点上本项不欠读**（S=1024/2048 的扫描点欠读，如实记）。 |
| ② | **`c_V` 的取值规律** | **未定。** dgrad 在 23 个 shape 里 21 个取 `c_V=2`，但 vocab=64640 取 **0**、仓内 DSv3 那次 campaign 取 **4**（同 shape 单卡今天取 2 → 是 build 差异）。这是 aclnn matmul 的**内部 kernel 选择**，从 6 个 vocab 值 + 6 个 H 值看不出判据。**后果**：小概率的 `c_V=0` shape 上本律**过读**（vocab=64640 处过读 1010 MiB = OOM 安全侧）；`c_V=4` 的 build 上**欠读 `2·vocab·B·S`**（DSv3 pp2 站点 = 2020 MiB，OOM-**不安全**）。**闭合它需要**：CANN 侧 aclnn matmul 的 tiling/kernel 选择判据（源码或 profiler 的 kernel 名），tracker 给不出。 |
| ③ | **既有 `bwd_scratch` 的 `K_CE`** | 实测 loss 层反向**共存 3 张瞬态满 vocab fp32 平面**；`K_CE−1` 在 pp>1 非 lean 下取 7 → 多 4 份 = 8080 MiB。**本轮一个字节没动**（单列为发现，不折进本项）。要动它须重钉一批 DSv3-era 冻结口径的锚点，属另一条任务线。 |
| ④ | **vocab ≤ 24240** | dgrad 的 `c_H` 掉到 0 → 本律过读 `4·H·B·S`（安全侧）。今天无锚点在该区间。 |
| ⑤ | **B 轴** | 两点（1 / 2，经 `B·S`）：8 卡 B=1、单卡 B=2 与仓内 profiler B=2 都逐字节吻合 ✓。 |
| ⑥ | **tp / sp 轴** | **未测**。`lm_head` 权重 `Shard(1)`（vocab 维 ÷tp，`head.py:163`）→ tp>1 真值应更小 ⇒ 本式**过读 = OOM 安全侧**。每-token 项走 TensorRef 不标 `sp` → 亦过读。今天所有锚点 tp=1。 |
| ⑦ | **cp 轴** | 结构上分开处理（常数不 ÷cp、每-token 项 ÷cp），cp>1 **未实测**。 |
| ⑧ | **全重算下 head 段** | run `c` 是**无重算**跑；`recompute=full` 时 head 段本身不被重算（loss 层从不重算），故本项在重算配置上同值——这是**推断**（同一 kernel、同一 shape），不是本轮的观测。 |
| ⑨ | **上几轮遗留** | `kernel_workspace_2026-07-29.md` §8 ①（逐层驻留欠读 / indexer 链）与 ②（r0/r128 的**反向** workspace）本轮**一条都没动**，仍未闭合。另 `mhc_fwd_workspace_2026-07-30.md` §6 ②（非融合 mHC 前向 workspace）亦未动。 |

---

## Related

- [`kernel_workspace_2026-07-29.md`](kernel_workspace_2026-07-29.md) —— 本条线的第一轮：方法、判据、「workspace 寿命 1 tick、不进驻留、却顶着峰」这条原理
- [`head_workspace_2026-07-30.md`](head_workspace_2026-07-30.md) —— word-embedding 反向 `GatherDGradV2`；S 扫描把它与 loss 侧分开的那次判据
- [`mhc_fwd_workspace_2026-07-30.md`](mhc_fwd_workspace_2026-07-30.md) —— 前向侧那一笔；「项对、落点对、却够不着锚点」的先例
- [`opdag_walker_core_2026-07-25.md`](opdag_walker_core_2026-07-25.md) §6.6 —— 「保留不变量、只移动举例」的改测试规矩
