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

## 6. 建模：复用既有 BWD 通道，只加一个挂载点

### 6.1 改了什么（全部加法，未标注者逐字节不变）

| 文件 | 改动 |
|---|---|
| `cost_eval/layers/head.py` | 顶部新增 `_HEAD_MATMUL_BWD_WS`（含全部实测出处/边界）；`build_head_and_loss_ops` 的 **`lm_head`** op（tie 与非 tie 两条路同挂）加 `bwd_workspace` + `bwd_workspace_ref` |
| `tests/test_head_bwd_kernel_workspace.py` | **新增 20 道门**（§6.5） |
| `tests/test_bwd_kernel_workspace.py` | 「未标注→0」那道门的**举例**由 `lm_head`（现已实测）移到 head 段里仍未测的 `final_norm`/`logsoftmax`/`nll`（不变量原样） |
| 13 个测试文件 + `scorecard_anchors.py` | 重钉台账，见 §9 |

**通道一个都没新建**：`OpSpec.bwd_workspace{,_ref}` → `ResolvedOp.bwd_workspace_bytes`
→ `StructureMemory.bwd_workspace`（层内取 **max**）→ `Buckets.workspace`（**BWD** 事件）
—— 全是 2026-07-29 那轮为融合稀疏 flash-MLA 建好的同一条路。

### 6.2 两条子通道的分工（照 r4 / embedding 先例）

```python
_HEAD_MATMUL_BWD_WS = {
    "bwd_workspace":     "20971520 + 1024",                    # 与 H/vocab/S 都无关的 matmul base
    "bwd_workspace_ref": TensorRef("head_matmul_bwd_ws",
                                   ("B", "S", "vocab + 2*H"),   # 2·(vocab+2H) = 2·vocab + 4·H B/token
                                   dtype_bytes=2),
}
```

- 常数项走**字符串**通道 → `shape_eval` 刻意不对 `bwd_workspace` 施加「含 S 就 ÷cp」
  （整体 ÷cp 会把常数也除掉 → cp>1 欠读 = OOM-**不安全**）。
- 每-token 项走 **TensorRef** → `resolve_tensor` 只对**首个** S 维 ÷cp（loss/head 区随 cp 切，
  `head.py:59-64` 的 D-1 权威口径）；**不标 `shard`** → tp>1 不 ÷tp = 过读 = 安全侧。
- **只挂一个 op**（`lm_head`），因为测量把它定位到了两个具体 kernel，二者同属该 op；
  层内取 max ⇒ 挂较大的那一笔（dgrad）即精确。**没有把任何数字摊到层上、也没有为了让
  哪个锚点落位而调过一个字节。**
- `20 MiB` 写成常数的依据是**观测**：同一常数（`20971520 = 20 MiB + 1024`）在 H=1792 的
  仓内 profiler 里作为**裸 base 独立出现 159 次**（§2.3），在 167 的 H=4096 站点出现 96 次
  → 它与 H 无关，不是把某个 `f(H)` 拟合成了常数。

### 6.3 层内取 max 的实测依据

rank 6 的 5384 个无类型池块**寿命直方图 = `[(1, 5382), (2, 1), (4, 1)]`** —— 恒 1 tick；
峰值那一刻在世的 workspace 只有一块。故 dgrad / wgrad / 梯度累加 / `ScatterAddExt` /
`RmsNormGrad` **五笔不相加**，层内 = max = dgrad。

### 6.4 逐层型 before → after（结构量）

`PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_head_bwd_ws.py`

| config | 层 | `bwd_workspace` before | after | 真机实测（同 kernel） | 判 |
|---|---|---:|---:|---:|---|
| DSv3 preset（H=1792, S=4096, B=1） | `lm_head` | 0 | **1109394432** | **1109394432** | 逐字节 |
| 同上，S=2048 | `lm_head` | 0 | 565183488 | （未在 8 卡量该 H·S 组合） | 律 |
| 同上，S=1024 | `lm_head` | 0 | 293078016 | （同上） | 律 |
| DSv4-hybrid（H=4096, S=4096, B=1） | `lm_head` | 0 | **1147143168** | **1147143168** | 逐字节 |
| DSv4-hybrid | `embedding` | 2336230400 | **2336230400** | — | 未动 |
| DSv4-hybrid | `dsv4hyb_r4_moe` | 765460480 | **765460480** | — | 未动 |
| DSv4-hybrid + MTP | `mtp`（含共享 head） | 2336230400 | **2336230400** | — | **未动**：层内 `max(embedding 2228.0, sparse_attn 730.0, lm_head 1094.0)` 仍是 embedding 那一笔 |

### 6.5 新增的 20 道门（守的是**测量**，不是模型自洽）

`tests/test_head_bwd_kernel_workspace.py`：

| 组 | 门 | 守什么 |
|---|---|---|
| 值 | 6 例（6 个实测点逐字节穿链） | 实测值穿过 `build_llm_spec → ShapeEval → StructureMemory`；律与常数表分道扬镳必红 |
| 层内 max | 5 例 | wgrad 实测更小 → 层内 max 必须是 dgrad（挡「顺手换成 `2·H` 那条式子」） |
| **反例** | 1 例 | vocab=64640 处本律**过读**，且过读量必须**恰是** `2·vocab·(B·S)`；挡「为了让它对上而删掉 `2·vocab` 项」（删掉 = 其余 10 个 vocab 值全部欠读 = OOM-不安全） |
| B 轴 | 1 例 | per-token 项经 `B·S` 翻倍、常数项不翻倍；且 B=2 那点 == 单卡与仓内 profiler 都量到的 `2197816320` |
| 归属 | 1 例 | 只有 `lm_head`；`final_norm`/`logsoftmax`/`nll` 必须 0 |
| **未建之项** | 1 例 | 梯度累加那一笔刻意不建（§7 ①）+ 钉住它的影响边界（S=4096 时本律必须盖过它、S=1024 时反过来 = 已知欠读） |
| 跨站点 | 1 例 | **直接从 `op_816365.csv` 读**（不转抄）：wgrad 律必须逐字节命中；并钉住该 build 的 dgrad 比本律多整整一份 `2·vocab·N` |
| cp | 1 例（带**非空转自检**） | 常数不吃 ÷cp、per-token 吃；且 cp=2 必须真把 `activation_saves` 切小 |
| tp | 1 例 | 不 ÷tp（真机 `Shard(1)` ⇒ 真值更小 ⇒ 过读 = 安全侧）；挡无实测背书的 ÷tp |
| **重建口吞字段** | 1 例 | MTP 路 `head.py` 的 `dataclasses.replace` / `residual._rebuild` —— 这条 bug class **真的发生过三次** |
| 相位不串 | 1 例 | 本项只占 BWD 通道；head 段 **fwd** `workspace` 必须仍是 0 |

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
| ⑩ | **`nll` 与 `final_norm` 自己的反向 workspace** | **实测到、律已定但刻意不建**（远小于层内 max，建了也不改变任何量）：`nll` 的 `ScatterAddExt` = `4·(B·S) + 16 MiB + 1536`（3 个 S 点逐字节，且与 H / vocab **无关** —— 三个 H、三个 vocab 上都是 16795136 B）；`final_norm` 的 `RmsNormGrad` ≈ 16.002 MiB（S=4096/2048 同值 16779264，S=1024 为 16778752，**3 点不成线性**，故不给律）；`_LogSoftmax.backward` 的 `Cast` **无 workspace**（实测 0）。已由 `tests/test_bwd_kernel_workspace.py` 钉住这三格为 0 = 已知欠读。 |
| ⑪ | **本项不设定该站点的峰** | §5.2：S=4096 上本项落地时 pool 离 high-water **4950.0 MiB**。既有通道 `max_t live + max_t ws` 是上界。**这不是本轮引入的**（r4 那一笔同理），但本轮第一次把余量量出来了。要做到时间分辨须重建 loss 层断面 —— 另一条任务线。 |
| ⑫ | **本轮没有测**：head 段的**前向** workspace（`head_fwd` 的 `MatMulExt` 实测 20.001 MiB、`loss_fwd` 的 `GatherD` 16.001 MiB） | 已测但**未建**：模型 head 段 fwd `workspace` 仍是 0。量级小（≤20 MiB），且 36/36 个锚点的峰全在 BWD（`mhc_fwd_workspace_2026-07-30.md` §2.2）→ 结构上够不着。如实记为已知欠读。 |

---

## 8. 逐锚点 before → after（**全部实跑重算，非转抄**）

复现：`PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_head_bwd_ws_ledger.py`
（在裸 HEAD 的 `git stash` 态与本树各跑一遍逐行 diff）。

| 锚点 | real MiB | before | **after** | Δ MiB | 判 |
|---|---:|---:|---:|---:|---|
| pp4 ON s0 / s1 / s2 | 24153.3 / 14641.7 / 14097.7 | 0.6725 / 0.8265 / 0.8402 | **同左，逐 MiB 不变** | 0 | 峰在解码层 bwd |
| **pp4 ON s3** | 23508.0 | 0.8954 | **0.9420** | **+1094.0** | ↑ 峰在 `bwd@9`(head) |
| pp4 OFF s0 / s1 / s2 | 30395.0 / 21019.4 / 17799.9 | 0.9089 / 0.9404 / 0.8738 | **同左** | 0 | 同上 |
| **pp4 OFF s3** | 27822.0 | 0.8376 | **0.8770** | **+1094.0** | ↑ |
| pp4 MTP s0-s2 | — | 0.6725 / 0.8265 / 0.8401 | **同左** | 0 | 同上 |
| pp4 MTP s3 | 39898.0 | 0.7509 | **0.7509** | 0 | **不动**：末层是 MTP 层，其层内 max 是 embedding 那 2228.0 |
| pp8 s0 | 24759.0 | 0.8274 | **0.8274** | 0 | 峰在 `bwd@1` |
| pp8 s1 / **s2** / **s3** / s4 / s5 / s6 | — | 1.0554 / **1.0007** / **1.0698** / 1.0519 / 1.1244 / 1.1093 | **逐位相同** | 0 | 各 stage 1 层，无 head 段 |
| **pp8 s7** | 26449.0 | 0.9556 | **0.9970** | **+1094.0** | ↑ 峰在 `bwd@9`(head) |
| 185 P3-P m8 s0 | 25343.5 | 0.6909 | **0.6909** | 0 | 峰在 `bwd@2` |
| **185 F0 4L** | 26499.0 | 0.8536 | **0.8747** | **+557.0** | ↑ 峰在 `bwd@5`(head)；seq2048 → 律给 557.0 |
| 185 U1 4L / U2 8L | 40194.0 / 56010.0 | 1.1168 / 1.3719 | **1.1307 / 1.3819** | +557.0 | ↑（本已过读） |
| 185 std ON kv32 s0 / kv8 s0 | 11131.6 / 10747.6 | 0.7466 / 0.7487 | **同左** | 0 | 峰在 `bwd@0`(embedding) |
| **185 std ON kv32 s1 / kv8 s1** | 15370.0 / 14986.0 | 0.9621 / 0.9659 | **1.0312 / 1.0368** | +1062.0 | ↑ **翻过 1.0** |
| 116 std mha pp2 s0 | 15946.1 | 0.9343 | **0.9343** | 0 | 峰不在 head 段 |
| **116 std mha pp2 s1 / pp1 s0** | 19227.3 / 24191.1 | 0.9889 / 0.9903 | **1.0442 / 1.0342** | +1062.0 | ↑ **翻过 1.0** |
| 116 std gqa pp2 s0 | 15116.0 | 0.8523 | **0.8523** | 0 | 同上 |
| **116 std gqa pp2 s1 / pp1 s0** | 18459.3 / 23039.1 | 0.9846 / 0.9747 | **1.0421 / 1.0208** | +1062.0 | ↑ **翻过 1.0** |
| **DSv3 4L full (dp2,sp)** | 12473.1 | 0.9961 | **1.0809** | +1058.0 | ↑ **翻过 1.0** |
| **DSv3 8L full (dp2)** | 13953.3 | 0.9925 | **1.0683** | +1058.0 | ↑ **翻过 1.0** |
| **DSv3 4L full ep=2** | 12474.1 | 0.9926 | **1.0774** | +1058.0 | ↑ **翻过 1.0** |
| **cp2 colossal / ulysses full 4L (B2)** | 12433.0 / 12441.0 | 0.9993 / 0.9986 | **1.0844 / 1.0837** | +1058.0 | ↑ **翻过 1.0** |
| pp2-stage0 (optstep) | 10246.0 | 1.0319 | **1.0319** | 0 | 峰在 `bwd@0`(embedding) |
| **pp2-stage1 (loss,k_ce=8)** | 45655.0 | 0.9990 | **1.0450** | +2096.0 | ↑ **翻过 1.0**（B·S=8192） |
| **cp2-none (loss,k_ce=4)** | 20119.4 | 0.9907 | **1.0433** | +1058.0 | ↑ **翻过 1.0** |
| **DSv3 8L none (dp2)** | 19967.3 | 0.9812 | **1.0342** | +1058.0 | ↑ **翻过 1.0** |
| **select self_attn (keep-FFN)** | 18828.2 | 0.9696 | **1.0258** | +1058.0 | ↑ **翻过 1.0** |
| **select mlp (keep-attn)** | 15764.7 | 0.9322 | **0.9993** | +1058.0 | ↑ 最贴近的一条（仍欠 0.7 MiB） |
| **select both (=full 退化端)** | 13953.3 | 1.0005 | **1.0763** | +1058.0 | ↑ |
| **DSv4-fused (base)** | 15415.5 | 0.9058 | **0.9408** | +539.0 | ↑ 仍 OOM-不安全，缺口收窄 |
| **DSv4 mHC(x4)+MTP** | 21153.1 | 0.8944 | **0.9199** | +539.0 | ↑ 同上 |

**OOM-不安全（ratio<1）的锚点/stage 数：38 → 23。**

### 8.1 边缘位（任务点名要看的）—— **原地不动**

| 位 | before | **after** | 为什么不动 |
|---|---:|---:|---|
| **pp8 s2** | 1.0007（高出真机 8.7 MiB） | **1.0007，逐 MiB 不动** | pp8 每 stage 1 层，s2 上**没有 head 段**（head 恒在末 stage，`parallel_model.py:122`）→ 本项恒 0。**没有越界。** |
| **pp8 s3** | 1.0698 | **1.0698，逐 MiB 不动** | 同上 |

### 8.2 ⚠ 14 个锚点由欠读**翻成过读**（1.02–1.08）—— 成因已定位并量化

**不是本项算错**（本项在 16 个点上逐字节实测），而是这些 config 的其它项**本来就在过读**、
过读量与本项同量级，此前互相抵消。**已量化的那一处**：

| | 真机实测（本轮 tracker，逐块可查） | 模型 |
|---|---|---|
| loss 层反向共存的**满 vocab fp32 平面**数 | **4 张**（1 张 saved `logsm` + **3 张瞬态**） | `mem_timeline.py:671-677` 的 `K_CE`：`pp>1 且非 lean` → `8−1 = 7 张`瞬态 |

即模型在**无重算的 loss stage** 上比实测**多记 4 张**平面。一张平面 = `4·S·B·vocab`
（H=1792/B=2/S=4096 站点 = 4040 MiB）。而代码注释本身就写着 `K_CE=8` 是
「**含当时未建模效应的混合常数**（DSv3-era 冻结口径）…**勿动其锚点**」——
**本项正是那批「当时未建模效应」之一**。

**按任务书要求，`bwd_scratch` 的标定问题单列为发现、本轮一个字节没动**（§7 ③）。
因此这 14 个锚点现在停在过读侧 = **OOM-安全侧**；band 只上移到刚好封住实测落点（两侧都守），
**没有为了凑回 ~1.00 而削本项**。

> 另有一层必须写在明处（§5.2 的量化）：既有通道算的是 `max_t live + max_t ws`，真值是
> `max_t[live + ws]`。在 S=4096 站点，本项落地时 pool 离 high-water 还有 **4950.0 MiB**
> → **本项不设定该站点的峰**。所以「锚点被抬起」在一部分锚点上是**两个上界互相抵消欠读**
> 的结果，**不是**「模型现在能分辨时间线了」。要做到后者必须重建 loss 层的时间分辨断面
> （`bwd_scratch` 与 workspace 按 tick 组合），那是另一条任务线。

---

## 9. 八跑验收门 before → after

`PYTHONIOENCODING=utf-8 PYTHONPATH=. python tools/liveness_ab_validate.py --grad-mode chain2 --deltas`

| 指标 | before | **after** |
|---|---|---|
| bucket 聚合 mean / min / max（n=28） | 0.841 / 0.703 / 1.038 | **0.847 / 0.703 / 1.038** |
| hand_spec·chain2 mean / min / max（n=28） | 0.860 / 0.675 / 1.110 | **0.860 / 0.675 / 1.110（逐位不变）** |
| hand_spec·dataflow | 0.799 / 0.662 / 1.067 | **同上，逐位不变** |
| 32 格逐格 sim/real | — | **只有 5 格动**：`a`/`c`/`d`/`e`/`g` 的 **s3** 各 +1094.0；其余 27 格逐 MiB 不变 |
| `unfused − fused` delta（bucket） | 0.522 | **0.477**（**缺口变大，如实记**：本项只抬 fused 侧 s3，unfused 侧 s3 峰在 `bwd@8:sparse_attn` 一分未动 → delta 净 −1094.0） |
| `unfused − fused` delta（hand_spec） | 0.638 | **0.638（逐位不变）** |
| I1/I2 **真机** ×1（层数 / 微批数） | PASS / PASS | **PASS / PASS** |
| I1/I2 **模型** ×1（bucket & hand_spec，共 4 条） | PASS ×4 | **PASS ×4** |
| `REAL_SHA256` 指纹 | `41e279e591ae4ae9…` PASS | **同值 PASS（一个字节未动）** |

**为什么 min/max 一格不动**：上移的 5 格都不是极值格（min 0.703 = `f`/`b` s1，
max 1.038 = `g` s1，都在解码层 bwd 上）。
**为什么 hand_spec 三行逐位不变**：`cost_eval/liveness/` 按 saves + grad 可达性自建图、
**不读** `bwd_workspace_bytes` —— 这也是本次只动一条口径的证据。

---

## 10. `REAL_*` / `CSV_*` / 指纹未动的 diff 级证明

判据（照前两轮）= 只看 `-` 侧有没有**既有真机常数定义行**被删改：

```bash
$ git diff 148116a HEAD -U0 -- . ':(exclude)docs' ':(exclude)scratchpad' \
    | grep -E '^-' | grep -v '^---' \
    | grep -E '^-\s*(REAL|CSV|_REAL|_CSV|MEASURED)[A-Za-z0-9_]*\s*[:=]'
# → 空（零命中）。**没有任何既有真机常数定义行被删改。**
$ git diff 148116a HEAD -- . ':(exclude)docs' | grep -i sha256
# → 空（`REAL_SHA256` 根本不在 diff 里）。
```

`-` 侧另有 6 处**引用** `REAL_*` 的断言行被改写（`REAL_4L`/`REAL_8L` 的 ±1% 门、
185 std 的 `sim < real` 门、pp2-s1 的欠读带）—— 那些是**断言**，常数本身一个字节未动；
逐条理由见 §11。`+` 侧新增的 `REAL_HEAD_DGRAD_WS_B` / `REAL_HEAD_WGRAD_WS_B` /
`REAL_HEAD_DGRAD_WS_B_ANOMALY` 在**新文件** `tests/test_head_bwd_kernel_workspace.py` 里，
是本轮**新增测量**，不替换任何旧值 —— 与前两轮新增 `REAL_EMB_GATHER_DGRAD_WS_MiB` /
`REAL_MHC_PRE_SINKHORN_FWD_WS_DISPLAY` 同款。

---

## 11. 重钉台账（old → new，逐条理由）

**未动**：`REAL*` / `CSV*` 一切真机常数、`REAL_SHA256` 指纹、两条真机 ×1 不变量与模型 ×1
不变量、run d 不可评分规则、`nr_moe_frag_factor` / `kept_frag_factor` 两个标定 margin、
`K_CE`（**一个字节没动**，§7 ③）。

| 位置 | old → new | 理由 |
|---|---|---|
| `cost_eval/layers/head.py::build_head_and_loss_ops` | 无 → `lm_head` 挂 `bwd_workspace{,_ref}` | 三套独立采集、16 点逐字节的实测律（§2） |
| `tests/test_head_bwd_kernel_workspace.py` | **新增**（20 例） | 新机制的守卫门，§6.5 |
| `tests/test_bwd_kernel_workspace.py::test_default_is_zero_...` | 举例 `lm_head` → **`final_norm`/`logsoftmax`/`nll`** | **不变量原样**（未标注 op → 0）；`lm_head` 现已实测，举例必须移（`opdag_walker_core` §6.6 先例）。新举例顺带钉住 head 段另外三个 op 仍是**已知欠读** |
| `tests/test_dsv3_golden.py::GOLDEN_BREAKDOWN` | `workspace` **0 → 1109394432**；`GOLDEN_PEAK_BYTES` 13093462528 → **14202856960**；锚 12486.9 → **13544.9** | 该桶在 DSv3 冻结口径上**第一次非 0**；+1058.0 = 实测律在 H=1792/S=4096/B=1 上的值 |
| `tests/test_regression_dsv3.py` | 「真机 ±1%」→ **钉新理论值 `THEO_4L/THEO_8L = 13481.9 / 14906.0` + 单列比值断言 1.0809 / 1.0683** | **不放宽成 ±9% 掩盖**；`REAL_4L/REAL_8L` 原值保留、新增 `THEO_*`，任何进一步漂移仍必红 |
| `tests/test_dsv4_preset.py` / `test_migration_outputs.py` / `test_fsdp_prefetch.py` | 同上三处「真机 ±1%」→ 钉理论值 13481.9 / 14906.0 | 同上 |
| `tests/test_from_mindformers.py` | DSv4 13963.8 → **14502.8**（+539.0）；DSv3 12423.9 → **13481.9**（+1058.0） | 实测律在各自 seq/H 上的值 |
| `tests/test_x4_experts_wrap.py` | 12423.9 → **13481.9** | 同上；experts 拆分仍不改峰值（本项与 experts 事件正交） |
| `tests/test_kept_frag_margin.py` | margin-off 15474.5 → **16532.5**；keep-attn 14696.0 → **15754.0**；full 硬门 12423.9 → **13481.9** | margin 仍是唯一变量、仍可关（本项与两个 margin 正交，margin 一个字节未动） |
| `tests/test_ce_optstep.py` | s1 带 `43000-47000` → **`46000-48500`**；DSv3 4L full 12423.9 → **13481.9** | s1 B·S=8192 → 律给 2096.0 |
| `tests/test_cp_activation.py` | band `12000-12600` → **`13000-13600`** | +1058.0；per-token 项**已按 cp 切**、常数项刻意不切 |
| `tests/test_pp4_recompute_anchor.py::THEO_ON` | s3 21049.8 → **22143.8**（+1094.001 整）；s0-s2 未动 | 只有峰值事件在 head 段的 stage 会动 |
| `...::THEO_PP8` | s7 25275.9 → **26369.9**（+1094.001 整）；s0-s6 未动 | 同上（各 stage 1 层，只 s7 带 head 段） |
| `...::_PP8_OVER_BAND` / `_PP8_UNDER` / `BAND_OFF` / `THEO_MTP` | **未动** | pp8 s1-s6 逐位不变；MTP s3 层内 max 仍是 embedding 那一笔 |
| `tests/test_probe185_recon.py::_STD_ON_THEO` | s1 两点 14786.8 → **15848.8** / 14474.8 → **15536.8**；**s0 两点未动** | s0 峰在 `bwd@0`(embedding) |
| `...::test_std_recompute_on_185_framework_gap` | s1 两点的 `sim < real` **不变量被本项翻转** → 改为**钉住翻转后的比值带**（`_STD_ON_GAP_FLIPPED`，两侧都守）；**s0 两点仍守原不变量** | 该 config 的框架缺口（~500 MiB/层）**小于**本项的 1062.0 MiB。**不删断言、不放宽方向**：按 stage 分档并把翻转写进常量名与消息里 |
| `tests/test_x4_p2p_pp.py` | s1 45611.4 → **47707.4**；`_s1_ratio` 带 `[0.995, 1.0]` → **`[1.040, 1.050]`**；s0 未动 | 这一笔正是该锚点自己那次采集的 profiler 量到的（`op_816365.csv` 的 wgrad `2168457216 B` 与本律逐字节吻合） |
| `scorecard_anchors.py` 10 条 band | 上移到刚好封住实测落点（如 `(0.97,1.05)` → `(1.07,1.09)`） | 每条带 `note` 写明 old→new 比值与「过读 = OOM 安全」；**两侧都守**，继续漂移必红 |
| `scorecard_anchors.py` 其余 4 条 band | **未动**（`pp2-stage0` / `DSv4 mHC+MTP` / …） | 逐位不变或仍在带内 |
| `tests/test_scorecard_anchors.py::test_d1_margin_is_present_and_effective` | 举例 `on < 1.0` → **`1.02 ≤ on ≤ 1.05`** | **不变量原样**（不得靠调 margin 凑；由 ① 的 `on > off` 断言守住，margin 仍 0.6）；翻转来自真机实测项，举例必须换 |
| `tests/test_acceptance_gate.py::GOLDEN_BUCKET` | `a`/`c`/`d`/`e`/`g` 的 **s3** 各 +1094.0；**其余 27 格逐字节不变** | 只有峰在 head 段的格会动 |
| `...::GOLDEN_AGG` bucket 两行 | mean 0.841 → **0.847**；min/max **未动** | 上移的 5 格都不是极值格 |
| `...::GOLDEN_AGG` hand_spec 三行 | **未动** | liveness 不读该字段 |
| `...::test_delta_magnitude_gap_is_recorded` | bucket 0.522 → **0.477**；hand_spec **0.638 未动** | **缺口变大，如实记**：unfused 侧 s3 峰不在 head 段 |

**没有删除任何一条不变量、没有删除任何用例**；新增用例守的是本轮新引入的机制。

---

## 12. 验收

```
$ PYTHONIOENCODING=utf-8 python -m pytest tests -q
2030 passed, 268 warnings in 106.41s
```

基线 **2010**（`148116a`）+ **20 道新守卫门**。八跑门 32 格 + 三条聚合 + `REAL_SHA256`
+ 四条 ×1 不变量：**全部 PASS**（§9）。

**真机侧卡态（退出时）**：`ps aux` 里**本人的进程 0 个**（`run_memprobe*` / `probe_*` /
`wait_and_run_head` / `run_head_ws_matrix` 全部退清，无孤儿）。`npu-smi info` 显示 8 卡
**全部被另一租户占用**（`/home/miniconda3/envs/ms29` 的 msrun，11:07 启动，21.5–26.9 GB /
AICore 100 %）—— **不是本人的**。本轮全程遵守「先确认 8 卡空闲再启动、从不抢占」：
另一租户 09:51 占卡 0-3 时本人的三次跑因 cluster-init 超时失败，改用
`wait_and_run_head.sh`（有界等待器，只在 8 卡全空闲时启动）后全部成功。

---

## 13. 服务器侧产物与卡态

`192.168.9.167:/home/suhaibo/workspace/`（**共享源码一行未改**，全部 monkeypatch 在探针文件内）：

| 文件 | 作用 |
|---|---|
| `probe_ws_head0.py` / `analyze_ws_head0.py` | 单卡：head/loss 那一族 kernel 的**判据自检** + 两种标记机制的 smoke test |
| `run_memprobe_tracker_head.py` | 8 卡探针（`import run_memprobe_tracker` + head/loss 标记；只在 TRACKED rank 上装） |
| `run_head_ws_matrix.sh` / `wait_and_run_head.sh` | 顺序驱动 + **有界等待器**（另一租户 09:51 占了卡 0-3，三次跑因 cluster init 超时失败；等待器只在 8 卡全空闲时启动，从不抢占） |
| `analyze_head_ws.py` / `head_ws_table.py` | 逐窗口 / 多跑逐字节表 |
| `probe_head_matmul_ws.py` / `analyze_head_matmul_ws.py` | 单卡密集 shape 图 + `c_V/c_H/base/tail` 穷举精确解 |
| `dsv4h_fused_pp4_norecomp_{v64640,v32320,h2048,h1792}.yaml` | vocab / H 扫描的 4 个新配置（**新增文件，未改任何既有配置**） |
| `log_ws_head_2026-07-30/`、`log_ws_head0/`、`log_head_matmul_ws/` | 全部原始 CSV 与分析文本 |

---

## 14. 本轮之后仍是 **OOM-不安全** 的锚点（23 条，模型欠读真机）

**不调参掩盖。** 按缺口从大到小：

| 锚点 | ratio | 峰值事件 | 本项为何不够 / 够不着 |
|---|---:|---|---|
| pp4 ON s0 / pp4 MTP s0 | 0.673 / 0.673 | `bwd@2`（解码层） | 本项在 head 段，够不着；缺口 7909 MiB |
| 185 P3-P m8 s0 | 0.691 | `bwd@2` | 同上 |
| pp4 MTP s3 | 0.751 | `bwd@9`（**MTP 层**） | MTP 层内 max 是 embedding 那 2228.0，本项被盖住 |
| 185 std ON kv32 s0 / kv8 s0 | 0.747 / 0.749 | `bwd@0`（embedding） | 峰在 embedding 反向，本项够不着 |
| pp4 ON s1 / s2、pp4 MTP s1 / s2 | 0.827 / 0.840 ×2 | `bwd@4` / `bwd@6` | 解码层 bwd |
| pp8 s0 | 0.827 | `bwd@1` | 同上 |
| 116 std gqa pp2 s0 | 0.852 | 解码层 bwd | 同上 |
| **185 F0 4L** | 0.875 | `bwd@5`(head) | **已被本项抬 +557.0**，仍欠 3321 MiB（seq2048 → 律只给 557.0） |
| pp4 OFF s2 | 0.874 | `bwd@6` | 解码层 bwd |
| **pp4 OFF s3** | 0.877 | `bwd@9`(head) | **已抬 +1094.0**，仍欠 3423 MiB —— 这是**唯一有本轮 ground truth 的一格**：真机 27822.0（rank 7）/ 27720.4（rank 6 逐块可查），模型 24398.7 |
| pp4 OFF s0 / s1 | 0.909 / 0.940 | `bwd@2` / `bwd@4` | 解码层 bwd |
| **DSv4 mHC(x4)+MTP** | 0.920 | `bwd@6`(head) | **已抬 +539.0**，仍欠 1695 MiB |
| 116 std mha pp2 s0 | 0.934 | 解码层 bwd | 本项够不着 |
| **DSv4-fused (base)** | 0.941 | `bwd@5`(head) | **已抬 +539.0**，仍欠 913 MiB |
| **pp4 ON s3** | 0.942 | `bwd@9`(head) | **已抬 +1094.0**，仍欠 1364 MiB |
| **pp8 s7** | 0.997 | `bwd@9`(head) | **已抬 +1094.0**，仍欠 79 MiB |
| **select mlp (keep-attn)** | 0.9993 | `bwd@9`(head) | **已抬 +1058.0**，仍欠 **0.7 MiB** —— 全库最贴的一条 |

**过读侧（OOM 安全）**：pp8 s1-s6（1.001–1.124）、185 U1 1.131 / U2 1.382、
pp2-stage0 1.032，以及本轮**新翻过 1.0** 的 14 条（1.021–1.084，见 §8.2）。

### 下一步该测什么（按能抬起多少排序）

1. **`bwd_scratch` 的 `K_CE` 重标定**（§7 ③）—— 现在是**最要紧**的一条：本项已经暴露出
   模型在无重算 loss stage 上多记 4 张满 vocab fp32 平面。修它会把 14 条新过读锚点拉回
   ~1.0，同时不动本项的实测值。需要重钉一批 DSv3-era 冻结口径。
2. **r0 / r128 的反向 kernel workspace**（`kernel_workspace_2026-07-29.md` §8 ②）——
   直接命中上表里**峰在解码层 bwd** 的那 12 条（缺口最大的一批，pp4 ON s0 欠 7909 MiB）。
3. **embedding 反向那一格的剩余缺口** —— 185 std ON s0 两点（0.747/0.749）峰在 `bwd@0`，
   `GatherDGradV2` 已建模却仍欠 ~2800 MiB。
4. **MTP 层的反向 workspace** —— pp4 MTP s3（0.751）峰在 MTP 层，其层内 max 现由 embedding
   那一笔占据；MTP 层自己的 kernel workspace **未测**。
5. 逐层驻留欠读 / indexer 链（`kernel_workspace_2026-07-29.md` §8 ①）—— 不是 workspace 口径。

---

## Related

- [`kernel_workspace_2026-07-29.md`](kernel_workspace_2026-07-29.md) —— 本条线的第一轮：方法、判据、「workspace 寿命 1 tick、不进驻留、却顶着峰」这条原理
- [`head_workspace_2026-07-30.md`](head_workspace_2026-07-30.md) —— word-embedding 反向 `GatherDGradV2`；S 扫描把它与 loss 侧分开的那次判据
- [`mhc_fwd_workspace_2026-07-30.md`](mhc_fwd_workspace_2026-07-30.md) —— 前向侧那一笔；「项对、落点对、却够不着锚点」的先例
- [`opdag_walker_core_2026-07-25.md`](opdag_walker_core_2026-07-25.md) §6.6 —— 「保留不变量、只移动举例」的改测试规矩
