# `K_CE` 重标定：从「混合常数」到**逐块可查的平面份数**（2026-07-30）

> 承接 [`head_loss_bwd_workspace_2026-07-30.md`](head_loss_bwd_workspace_2026-07-30.md) §7 ③ / §8.2 —— 那一轮
> 把 `bwd_scratch` 的 `K_CE` 单列为发现、**一个字节没动**，并判它是「下一步最要紧的一条」。本轮去动它。
>
> **这一轮与之前每一轮方向相反：它是在 *减* 字节。** 因此三条纪律加倍执行：
> ① 绝不编造真机数；② **绝不用另一个猜测替换被拿掉的猜测**——新值必须能从台账逐块数出来或从源码读出来；
> ③「我跑了并观察到」与「我推断」分开写。`REAL_*` / `CSV_*` 常数与 `REAL_SHA256` 指纹**一个字节未动**（§10）。
>
> **本轮没有跑任何真机测量。** 全部实测数据来自**仓内既有台账**：
> `analysis/realmachine/**` 的 9 份 profiler CSV（每份的 high-water 都逐 MiB 命中它自己那条锚点的 `real`，见 §2.1）
> 与前两轮报告里的逐块 dump。

复现（全部离线、纯读数）：

```bash
# ① 从仓内既有 profiler CSV 逐块数「满 vocab 平面」共存份数（不重跑真机）
PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_realmachine_planes.py
# ② 模型侧对应份数（逐锚点：loss 层 bwd 事件的 bwd_scratch ÷ 一张平面）
PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_k_ce_planes.py
# ③ 逐锚点台账 before/after
PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_head_bwd_ws_ledger.py
# ④ 八跑门
PYTHONIOENCODING=utf-8 PYTHONPATH=. python tools/liveness_ab_validate.py --grad-mode chain2 --deltas
```

---

## 0. 一句话结论

| | 结论 |
|---|---|
| **上一轮的归因是错的，必须先订正** | 上一轮写「真机共存 3 张瞬态平面，而 `mem_timeline` 的 `K_CE=8` 记 7 张」。**这两个数不在同一条代码路径上。** 167 那个站点（DSv4-hybrid）模型侧 `cross_entropy_fused=True` → `loss_lids` **空**（`mem_timeline.py:324`）→ **`K_CE` 那一行根本不执行**（实测：该站点 `bwd_scratch` = **2.00** 张平面，`probe_k_ce_planes` 输出）。所以「多记 4 张」这个量**从未在任何锚点上发生过**。 |
| **真正的过读是 1 张平面，且两个 fat 分支都是 1 张** | 从 **9 份仓内 profiler CSV 逐块数**出来的 high-water 在世满 vocab 平面：**7 份 pp=1 的 campaign（跨 DSv3 与 DSv4-align 两个模型族）全部是 3 张 fp32 + 1 张 bf16**；唯一一份 pp>1 的（pp2-stage1，正是 `K_CE=8` 那条锚点自己那次采集）是 **5 张 fp32 + 5 张 bf16**。模型侧对应的量是 `K_CE + 0.5` 张 fp32-等效。解出来：**pp1 该取 3（现 4）、pp>1 该取 7（现 8）**——两边**各过读恰好 1 张**。 |
| **那 1 张就是代码注释里自己写的「+1 保守」** | `mem_timeline.py:664` 逐字：「实测 3 份共存（cp2-none/select），代码取 **4 = 3 观测 + 1 保守**（OOM-安全侧）」。本轮做的事 = **把这条被明示的 padding 拿掉，让常数回到它声称的物理语义**。 |
| **`lean` 分支不动** | `ce_pynative_lean` 的 `K_CE=4` 只有 116/MS2.9 的**峰值差分拟合**背书，**没有逐块台账**（那批探针只记 `max_memory_allocated`）。按任务书「不能确立就保守划界」→ **留 4，并写明什么测量能定它**（§5.3）。 |
| **代价如实记** | 拿掉那 1 张平面 = 在 fat 门开的锚点上减 1 张（N=4096 → 2020 MiB / N=8192 → 4040 MiB）。逐锚点位移与 OOM-不安全数见 §8（实跑重算，非预测）。方向上这是对的：**真值 + 明示余量**，而不是把错常数留着当顺手的 padding。 |

---

## 1. 先订正上一轮的归因（hazard 1 的入口）

上一轮 `head_loss_bwd_workspace_2026-07-30.md` §8.2 的那张两列表：

| | 上一轮写的 | 本轮核查 |
|---|---|---|
| 真机 loss 层反向共存满 vocab fp32 平面数 | **4 张**（1 saved + 3 瞬态）—— 167/MS2.10 DSv4-hybrid pp4 站点 | ✔ 该数没问题（该文 §5.1 逐块 dump） |
| 模型 | 「`mem_timeline.py:671-677` 的 `K_CE`：`pp>1 且非 lean` → `8−1 = 7 张`瞬态」 | ✘ **该站点走不到那一行。** |

**为什么走不到**（`file:line` + 实测）：

1. `cost_eval/mem_timeline.py:324`
   ```python
   loss_lids = (set() if cross_entropy_fused else
                {l.layer_id for l in layers if any(... == "nll" ...)})
   ```
   `cross_entropy_fused=True` → `loss_lids` **空集**。
2. `cost_eval/mem_timeline.py:671` 的门是 `if _stage_no_recompute and lid in loss_lids and sm.bwd_scratch > 0:`
   → `loss_lids` 空 ⇒ **恒不进**，`K_CE` 那一行（`:676`）**不执行**。
3. 167 那个站点（= `pp4 ON/OFF/MTP`、`pp8`、`185 F0/U1/U2`、`185 P3-P` 全族）模型侧
   `cross_entropy_fused=True`：`cost_eval/configs/from_mindformers.py:566-573`
   （dsv4_hybrid 架构默认 `True`，并发 provenance 警告），
   `tests/test_anchor_site_yaml_agreement.py:91` 逐字「对本族锚点恒 0：`cross_entropy_fused=True` →
   `loss_lids` 空」。
4. **实测**（我跑了并观察到，`scratchpad/probe_k_ce_planes.py` 同法的 pp4/pp8 版）：

   ```
   pp4 ON   ce_fused=True ce_lean=False pp=4 ... plane=2020.0 MiB
      s3 peak=bwd@9   bwd_scratch=   4040.0 MiB = 2.00 planes
   pp4 OFF  s3 peak=bwd@9   bwd_scratch=   4040.0 MiB = 2.00 planes
   pp8 ON   s7 peak=bwd@9   bwd_scratch=   4040.0 MiB = 2.00 planes
   ```
   `2.00` = `8·S·B·vocab`（`head.py:258` 的原值，未经 `K_CE` 改写）。

**结论（订正）**：167 站点上模型记 **2 张瞬态**、真机 **3 张瞬态** → 该站点是**欠读 1 张**
（2020 MiB），**不是**过读 4 张。上一轮那句「多记 4 张 = 8080 MiB」在任何锚点上都不成立，
本轮起以本文为准。

> **顺带一条独立发现（不在本轮动）**：167 站点真机的 CE 链是**非融合**的
> —— 该文 §1.1 的 `_LogSoftmax` / `_NLLLoss` 子类标记（`loss.py:125-186`）**真的 fired**，
> 说明现场跑的是 `logsoftmax_nll` 而不是融合 CE kernel；而模型侧该族 `cross_entropy_fused=True`。
> 这是一处**配置口径与现场不符**，且方向是**欠读**（`loss_lids` 空 → 不吃 fat）。
> 改它会把整族 DSv4 锚点搬到 fat 分支上，量级 ≥1 张平面/锚点，**属另一条任务线**，
> 本轮一个字节没动。要闭合它需要：确认 `dsv4h_*` yaml 里到底哪个键决定 CE 融合，
> 以及 167 那个 build 的 `CrossEntropyLoss` 实现标识。

---

## 2. 台账：9 份仓内 profiler CSV 逐块清点

### 2.1 先证「这些 CSV 就是这些锚点自己那次采集」

`analysis/realmachine/**` 的每份 CSV 里 `Allocation Total Allocated(MB)` 的**最大值** = 该 rank 全程
pool high-water。逐份对照 `scorecard_anchors.py` 的 `real`：

| CSV | CSV high-water (MiB) | 锚点 `real` (MiB) | 差 |
|---|---:|---:|---:|
| `pp2_norecomp/op_816365.csv` | **45655.46** | `pp2-stage1 (loss,k_ce=8)` 45655.0 | 0.46 |
| `pp2_norecomp/op_816362.csv` | **10246.24** | `pp2-stage0 (optstep)` 10246.0 | 0.24 |
| `cp2_none/operator_memory.csv` | **20119.37** | `cp2-none (loss,k_ce=4)` 20119.4 | 0.03 |
| `select_ffn/operator_memory.csv` | **19967.28** | `DSv3 8L none (dp2)` 19967.3 | 0.02 |
| `select_attn/operator_memory.csv` | **18828.17** | `select self_attn (keep-FFN)` 18828.2 | 0.03 |
| `select_mlp/operator_memory.csv` | **15764.66** | `select mlp (keep-attn)` 15764.7 | 0.04 |
| `dsv4_fused/operator_memory.csv` | **15415.47** | `DSv4-fused (base)` 15415.5 | 0.03 |
| `operator_memory_rank0.csv`（仓根） | **12473.10** | `DSv3 4L full (dp2,sp)` 12473.1 | 0.00 |
| `cp2_colossal/operator_memory.csv` | **12433.20** | `cp2 colossal full 4L (B2)` 12433.0 | 0.20 |

9/9 命中（≤0.5 MiB）。**故这些 CSV 就是锚点真机值的来源台账，用它们数平面不是「另一次测量」。**
（附带订正两处目录命名误导：`select_ffn/` 那份实际是 **8L-none** 那条锚点的采集；
仓根 `operator_memory_rank0.csv` 是 **4L-full** 那条。以 high-water 对齐为准。）

### 2.2 high-water 那一刻在世的**满 vocab 平面**（逐块数出来）

判据：块尺寸落在 `4·N·vocab ± 8 KiB`（= fp32 满 vocab 平面）或 `2·N·vocab ± 8 KiB`（bf16），
`N = B·S/cp`（loss/head 区随 cp 切，`head.py:201-206` 的 D-1 权威口径）；
「在世」= `Allocation Time ≤ t_hw < Release Time`。

| campaign（= 锚点） | N | pp | 重算 | **fp32 平面** | **bf16 平面** | fp32-等效 | high-water 设定者 |
|---|---:|---:|---|---:|---:|---:|---|
| pp2-stage1 | 8192 | 2 | 无 | **5** | **5** | **7.5** | `MatMulExt` 2168457216 B（= wgrad workspace） |
| cp2-none | 4096 | 1 | 无 | **3** | **1** | **3.5** | `ScatterAddExt` 16795136 B |
| DSv3 8L none | 4096 | 1 | 无 | **3** | **1** | **3.5** | `ScatterAddExt` 16795136 B |
| DSv3 4L full | 4096 | 1 | full | **3** | **1** | **3.5** | `ScatterAddExt` 16795136 B |
| cp2 colossal 4L full | 4096 | 1 | full | **3** | **1** | **3.5** | `ScatterAddExt` 16795136 B |
| select self_attn | 4096 | 1 | select | **3** | **1** | **3.5** | `ScatterAddExt` 16795136 B |
| select mlp | 4096 | 1 | select | **3** | **1** | **3.5** | `ScatterAddExt` 16795136 B |
| **DSv4-align fused** | **2048** | 1 | 无 | **3** | **1** | **3.5** | `ScatterAddExt` 16786944 B |
| pp2-stage0 | 8192 | 2 | 无 | 0 | 0 | 0 | `Mul` 926679552 B（无 lm_head 的 stage） |

> **DSv4-align 那一行是第 7 个独立佐证，而且仓内 2026-07-06 的诊断早就写了这句**：
> `analysis/realmachine/dsv4_fused/DIAGNOSIS.md:20` 逐字
> 「| 1010 fp32 vocab | **3** | 3030 | ✅ `logsm`(saved)+`probs`+`grad_log_softmax`(bwd_scratch)
> ——**恰 3 个**，子代理"漏建 CE buffer"假设被证伪 |」，下一行「| 505 bf16 vocab logits | 1 |」。
> 该站点 `seq2048`（`DIAGNOSIS.md:3`）→ `N = 2048`、一张 fp32 平面 = 1010 MiB。
> **它把「3 fp32 + 1 bf16」推广到了第二个模型族（DSv4-align）与第二个 seq（2048）。**
> 且它是 `cross_entropy_fused=True`（门关）→ 模型 3.5 vs 真机 3.5，**又一处零误差**。

**产出者名字**（同一次清点顺带打出，佐证归属）：

```
pp2-s1  fp32: {'Log': 3, 'Neg': 1, 'ScatterAddExt': 1}   bf16: {'Copy': 4, 'ZerosLikeExt': 1}
pp1 七份 fp32: {'Log': 1, 'Neg': 1, 'ScatterAddExt': 1}   bf16: {'Copy'|'MatMulExt': 1}
```

`{Log, Neg, ScatterAddExt}` 逐条对应 `loss.py:185-196` 的 NLL 反向链
（`head.py:174-177` 的 `bwd_scratch="8*S*B*vocab"` 就是照这三行建的：`Log` = saved
`log_softmax`、`Neg`/`ScatterAddExt` = `probs` 与 `grad_log_softmax` 两张瞬态）；
bf16 那一张 = `logits_lm`（`head.py:213`）。**归属不含糊。**

pp2-s1 的 3 个 `Log`(fp32) 与 3 个 `Copy`(bf16) 里各有 **3 个 / 3 个** 在 CSV 窗口内**无 Release**
（长寿），其余为瞬态 —— 即 **3 组「每微批 loss 区 saved 对（logsm fp32 + logits bf16）」同驻**
＋ 2 张 fp32 瞬态 ＋ 2 张 bf16 瞬态。pp=1 七份则是 **1 组 saved 对 ＋ 2 张 fp32 瞬态**。

### 2.3 pp=1 那七份彼此独立

它们跨 **2 个模型族**（DSv3 / DSv4-align）、**2 个 seq**（2048 / 4096 ⇒ 2 个 `N`：2048 / 4096）、
**2 个 B**（1 / 2）、**2 个 cp**（1 / 2）、**3 种重算态**（none / full / select）、
**2 个层数**（4 / 8）、**2 种 CP 方法**、**融合与非融合 CE 两种口径**、
**3 次不同日期的 campaign**（07-01 / 07-06 / 07-07~08），**份数恒为 3 fp32 + 1 bf16**。
→ 「份数在 pp=1 下不随模型族 / seq / B / cp / 重算态 / 层数 / CE 融合与否变」是**观测**，不是推断。

---

## 3. 模型侧对应的量（实跑读数，非推断）

`scratchpad/probe_k_ce_planes.py`（新增）：取每条锚点 **loss 层 `bwd@<max lid>` 事件**的
`breakdown.bwd_scratch`，除以一张 fp32 平面：

```
锚点                          | peak    | real    | ratio  | bwd_scratch | ÷平面 | act_live | 平面 MiB
DSv3 4L full (dp2,sp)       | 13481.9 | 12473.1 | 1.0809 | 4040.0      | 2.00 | 3114.0   | 2020.0
DSv3 8L full (dp2)          | 14906.0 | 13953.3 | 1.0683 | 4040.0      | 2.00 | 3170.0   | 2020.0
DSv3 4L full ep=2           | 13439.9 | 12474.1 | 1.0774 | 4040.0      | 2.00 | 3114.0   | 2020.0
cp2 colossal full 4L (B2)   | 13481.9 | 12433.0 | 1.0844 | 4040.0      | 2.00 | 3114.0   | 2020.0
cp2 ulysses full 4L (B2)    | 13481.9 | 12441.0 | 1.0837 | 4040.0      | 2.00 | 3114.0   | 2020.0
pp2-stage0 (optstep)        | 10573.0 | 10246.0 | 1.0319 | 0.0         | 0.00 | 4712.8   | 4040.0
pp2-stage1 (loss,k_ce=8)    | 47707.4 | 45655.0 | 1.0450 | 28280.0     | 7.00 | 8708.5   | 4040.0
cp2-none (loss,k_ce=4)      | 20991.1 | 20119.4 | 1.0433 | 6060.0      | 3.00 | 5756.4   | 2020.0
DSv3 8L none (dp2)          | 20649.5 | 19967.3 | 1.0342 | 6060.0      | 3.00 | 5532.4   | 2020.0
select self_attn (keep-FFN) | 19314.0 | 18828.2 | 1.0258 | 4040.0      | 2.00 | 4796.4   | 2020.0
select mlp (keep-attn)      | 15754.0 | 15764.7 | 0.9993 | 4040.0      | 2.00 | 4018.0   | 2020.0
select both (=full,退化端)     | 15018.0 | 13953.3 | 1.0763 | 4040.0      | 2.00 | 3282.0   | 2020.0
```

`act_live` 里的 loss 区份数也实测出来了（相邻 `bwd@` 事件差；`pinned.pop` 在 `rec()` 之后，
`mem_timeline.py:767-778`）：

```
8L none dp2  act_live(bwd@9)=5532.4  act_live(bwd@8)=2474.4  head 层 saves=3058.0 MiB
pp2-s1       act_live(bwd@9)=8708.5  act_live(bwd@8)=2592.5  head 层 saves=6116.0 MiB
8L full      act_live(bwd@9)=3170.0  act_live(bwd@8)= 112.0  head 层 saves=3058.0 MiB
```

`3058.0 = 2020.0(1 张 fp32) + 1010.0(1 张 bf16) + 28.0`，`6116.0 = 4040 + 2020 + 56`
（28/56 = `h_last`+`h_final` 两个 `[S,B,H]` bf16）→ **模型的 loss 区 saved 恒为
「1 张 fp32 + 1 张 bf16」，且 pp2-s1（m=2）也只 pin 1 组**（不是 2 组、不是 3 组）。

**故模型侧 loss 区满 vocab 总量（fp32-等效张数）**：

```
fat 门开：  (K_CE − 1)  瞬态  +  1 saved fp32  +  0.5 saved bf16   =  K_CE + 0.5
fat 门关：      2       瞬态  +  1 saved fp32  +  0.5 saved bf16   =  3.5
```

---

## 4. 对账 → 解出 `K_CE`

| 分支（`mem_timeline.py:676`） | 走这条的锚点 | 模型 fp32-等效 | **台账实测 fp32-等效** | 差 |
|---|---|---:|---:|---:|
| 门关（`cross_entropy_fused` 或 stage 内有重算） | 4L-full / 8L-full / ep2 / cp2-full ×2 / select ×3 / **pp4·pp8·185 全族** | 3.5 | **3.5**（7 份 CSV） | **0（逐张同构：2 瞬态 fp32 + 1 saved fp32 + 1 saved bf16）** |
| `pp == 1 且非 lean` → `K_CE=4` | `cp2-none`、`DSv3 8L none` | 4.5 | **3.5**（这两条自己的 CSV） | **+1.0 张** |
| `pp > 1 且非 lean` → `K_CE=8` | **仅** `pp2-stage1` | 8.5 | **7.5**（该条自己的 CSV） | **+1.0 张** |
| `lean` → `K_CE=4` | 116 std mha/gqa × {pp2-s1, pp1-s0} | 4.5 | **无逐块台账** | 未知 |

解出：`K_CE(pp1) = 3`、`K_CE(pp>1) = 7`。

**两个 fat 分支都恰好过读 1 张**，且这 1 张就是 `mem_timeline.py:664` 自己写的
「**4 = 3 观测 + 1 保守**（OOM-安全侧）」。**门关那条路是逐张同构的零误差基准**
（`8*S*B*vocab` = 2 张瞬态，源自 `head.py:174-177` → `loss.py:185-196`），
这正是「减掉那 1 张之后 pp1 的 fat 退化成门关值」自洽的原因。

---

## 5. 三条 hazard 逐条回答

### 5.1 hazard 1 —— `K_CE=8` 是否还站着别的未建模效应？

**是，但不是上一轮说的那个，而且已经量化到逐块。** 把 `K_CE=7` 之后 pp2-s1 的 6 张瞬态 fp32
与台账逐块摊开：

| 台账实测（pp2-s1 high-water 在世） | 模型用什么承载 |
|---|---|
| 2 张 fp32 瞬态（`Neg`、`ScatterAddExt`） | ✔ `8*S*B*vocab` 本项（物理，`loss.py:185-196`） |
| 1 组 saved 对（`Log` fp32 + `Copy` bf16） | ✔ `act_live` 的 `logsm` + `logits_lm` |
| **另外 2 组 saved 对**（2×(fp32+bf16) = 2 张 fp32-等效） | ✘ 未建模 —— **被 `K_CE` 吸收**。物理成因 = pp 末 stage 多个在途微批的 loss 区 saved 同驻；模型的 `pinned[(mb,lid)]` 在 loss 层只 pin 1 组（§3 实测） |
| **2 张 bf16 瞬态**（`Copy`、`ZerosLikeExt` = 1 张 fp32-等效） | ✘ 未建模 —— **被 `K_CE` 吸收**。op 图里 `logsoftmax`/`nll` 未声明 bf16 满 vocab 瞬态 |

即 `K_CE(pp>1)=7` 仍是**混合常数**，但它现在**混的是两件已定位、已量化的事**（2 组在途微批 saved 对 + 2 张 bf16 瞬态 = 3 张 fp32-等效），
而**不再混那一张纯 padding**。pp1 分支取 3 之后**不再混任何东西**——它逐张等于台账
（2 瞬态 + 1 saved fp32 + 1 saved bf16），是本轮唯一被彻底去混的一格。

**同轮测到但仍留 0 的那几项与本项的关系**（任务书点名）：

| 项 | 是否被 `K_CE` 吸收 | 依据 |
|---|---|---|
| `nll` 的 `ScatterAddExt` workspace（`4·(B·S)+16 MiB+1536`） | **否** | 它就是 pp1/167 两处 high-water 的**设定者本身**（§2.2 表末列），量级 16 MiB；`K_CE` 的单位是 4040/2020 MiB 的平面。两者差 250×，不可能互为标定 |
| `final_norm` 的 `RmsNormGrad`（≈16.002 MiB） | **否** | 同上量级 |
| head **前向** workspace（≤20 MiB） | **否** | 同上量级，且 36/36 锚点峰在 BWD |
| 参数梯度累加 workspace（`2·vocab·H+2560`，5 点） | **否，且已被判非 head 专有** | `head_loss_bwd_workspace_2026-07-30.md` §7 ①：解码层每个参数各有一笔 → 只挂 head 属误归属。它在 pp2-s1 站点 = `2·129280·1792+2560` = 441.9 MiB，**不是**平面量级；且它是 `workspace` 桶（层内 max），已被 dgrad 那 2068 MiB 盖住（该 CSV 的 high-water 设定者正是 dgrad/wgrad 那一笔） |

**结论**：那批「同轮测到、刻意留 0」的项**全部是 16–442 MiB 量级**，而 `K_CE` 的刻度是
**2020 / 4040 MiB 一张**。它们**不可能**是 `K_CE` 里那 1 张平面的成分——把 `K_CE` 减 1 张
**不会**把它们的欠读一起削掉，因为它们从来不在这个桶里。这是本轮敢减的关键判据。

### 5.2 hazard 2 —— 是不是跨模型常数？按 `norm_compute_dtype_bytes` 先例分档

**答：不是跨 build 常数，而且台账直接证伪了跨 build。** 两条独立证据：

1. **份数本身跨 build 变了**：DSv3-era（MS2.9 那批 campaign）pp=1 是
   **1 saved + 2 瞬态 = 3 张 fp32**（`Log`/`Neg`/`ScatterAddExt`，§2.2）；
   167/MS2.10 那个站点是 **1 saved + 3 瞬态 = 4 张 fp32**
   （`head_loss_bwd_workspace_2026-07-30.md` §5.1 的逐块 dump：`Contiguous`(saved,life=14023) +
   `Log` + `Neg` + `ScatterAddExt`）。**同一族 op 链，MS2.10 多留一张。**
2. 仓内既有判词：`tests/test_std_attn_anchor.py:44-46` 逐字「185/MS2.10 同 config 全线低 2-3.3GB
   （**build 差**）」；`analysis/closure_report_2026-07-15.md:107` 逐字「K_CE 只由**同一模型**
   seq 2048/4096 一组差分支持…未满足跨模型稳定性标准」。

**故按 `docs/census_fix_mhc_rmsnorm_2026-07-29.md` 的先例（`norm_compute_dtype_bytes` 那次
不做全局翻转、改成**逐 norm 种类**）——本轮**逐分支**改，不做全局改**：

| 分支 | 动作 | 背书 |
|---|---|---|
| `pp == 1 且非 lean` | 4 → **3** | 7 份 CSV 逐块（跨 2 个模型族 / 2 个 seq），byte + 组成双重同构 |
| `pp > 1 且非 lean` | 8 → **7** | 该分支**唯一**锚点自己那份 CSV 逐块 |
| `lean`（116/MS2.9 build） | **不动，仍 4** | 见 §5.3 |
| 门关（fused CE / 有重算） | **不动**（本来就不过 `K_CE`） | §4 第一行，误差 0 |

**每个分支只被它自己的台账改。** 没有一个分支的值来自另一个分支的观测。

### 5.3 `lean` 分支为什么不动，以及什么测量能定它

`ce_pynative_lean=True` 只有 4 条锚点在用（116 std mha/gqa × {pp2-s1, pp1-s0}，
`tests/test_std_attn_anchor.py:120` 的 `_BUILD_FACTS = {"emb_bytes":"4","ce_lean":"1","sched_wp1":"1"}`）。
它的 `K_CE=4` 出处是 `tests/test_std_attn_anchor.py:27-29`：
「unfused CE 链实测 **~3.3-4 份**满 vocab fp32 co-live 且与 pp 无关（pp1/pp2-s1 差分一致）」
—— 这是**整机峰值差分反解**出来的数（116 那批探针是 `run_memprobe2 safe`，只记
`max_memory_allocated`，仓内**没有** 116 的逐块 CSV），因此它**吸收了该 config 其它所有误差**，
**不能**当平面份数用。「~3.3-4」这个区间本身就横跨 3.5，说明它分辨不到 0.5 张。

- **不动的理由**：把 DSv3-era（MS2.9 DSv3 preset）的逐块份数外推到 116 std build，正是
  §5.2 第 1 条已经证伪的那种外推（MS2.10 就多一张）。**宁可 fail-loud 保留一个明示为拟合的值，
  不用另一个猜测替换它。**
- **定它需要**：在 116（或任何跑 `ce_lean` 那个 build 的机器）上开
  `MS_ALLOC_CONF="memory_tracker:True"` 采一次 **std MHA pp1** 的 `memory_block.csv`，
  在 high-water tick 上数 `4·N·vocab ± 8 KiB` 与 `2·N·vocab ± 8 KiB` 的在世块数
  —— 与 §2.2 完全同一条判据、同一把尺。**一次单跑、一份 CSV 即可闭合**，不需要扫轴。
- **副作用如实记**：`lean` 留 4 而 `pp==1 非 lean` 变 3 之后，`ce_pynative_lean` 这个开关
  在 `pp==1` 下**第一次开始改变数值**（此前两边都是 4，它只在 `pp>1` 下有效）。
  这不是新语义、是旧语义的暴露：该 flag 一直声称「与 pp 无关」，现在它字面上就是这个意思。

### 5.4 hazard 3 —— 份数是否 config 相关？

| 轴 | 判 | 依据 |
|---|---|---|
| 模型族 / `seq` / `B` / `cp` / 层数 / 重算态 / CP 方法 / CE 融合与否 | **不相关**（pp=1 下份数恒 3+1） | §2.3，7 份 CSV 的正交组合 |
| `N = B·S/cp` | **不相关**（份数是计数，`N` 只改一张平面多大） | pp1 七份取 `N`∈{2048, 4096} 份数同；pp2-s1 `N`=8192，份数差异**全部**可由 pp 解释 |
| `pp` | **相关** | 3+1（pp=1）vs 5+5（pp=2）；机理 = 在途微批 saved 组数（§5.1）。**代码本来就按 pp 分档**，本轮保持该分档 |
| `chunk_loss_num`（分块 CE） | **结构上正交，无需分档，且无台账** | `head.py:253-256`：`bwd = "8*S*B*vocab//k"` → `sm.bwd_scratch` 已 ÷k，而 `mem_timeline.py:677` 是 `sm.bwd_scratch // 2 * (K_CE-1)` → fat **自动 ∝1/k**。`K_CE` 是**份数**、不是尺寸，故不吃 k。**份数本身是否随 k 变没有台账**（`chunked` 只出现在 `tests/test_mtp_loss.py` 的结构门里，**无任何真机锚点**）→ 不为它引入分档，如实记为未测 |
| `vocab_parallel_ce` | **既有潜在不一致，本轮不动、单列** | 该变体的 `nll_op` 走 `bwd_scratch=None + bwd_scratch_ref=vp_grad`（`head.py:249-251`）→ `sm.bwd_scratch` 只有 **1 张**（÷tp）而非 2 张，于是 `sm.bwd_scratch // 2 * (K_CE-1)` 的「÷2 = 去掉 2 张里的一张」前提**不成立**（会给出 `(K_CE−1)/2` 张）。这是**改 `K_CE` 之前就存在**的问题，且无锚点触及（同上，只在 `test_mtp_loss.py` 结构门里）→ 如实单列，本轮不动 |
| 融合 CE（`cross_entropy_fused`） | 门关，`K_CE` 不参与 | §1 |

---

## 6. 选定的值

```python
# cost_eval/mem_timeline.py（模块级命名常数，可 grep、可被门钉住）
K_CE_PP1  = 3     # was 4   —— pp==1 且非 lean
K_CE_PP   = 7     # was 8   —— pp>1  且非 lean
K_CE_LEAN = 4     # 不动    —— ce_pynative_lean（无逐块台账）
```

**它现在建模什么**：`K_CE` = 无重算 loss stage 在 loss 反向峰**同驻的满 vocab 平面总数**
（以 fp32 平面为单位；模型另有 1 张 bf16 saved 平面在 `act_live`，故 `K_CE` 承担
`实测 fp32-等效总数 − 0.5`）。

**它仍然吸收什么**（§5.1 已逐块量化）：
- `pp>1` 的 7 里，有 **3 张 fp32-等效**不是 fp32 瞬态 —— 2 组在途微批 loss 区 saved 对
  + 2 张 bf16 满 vocab 瞬态。要去掉这部分混合，须（a）让 `pinned` 在 pp 末 stage 按在途微批数
  pin loss 区 saved（需要该 fork 调度器的在途组数判据），（b）在 op 图里声明
  `logsoftmax`/`nll` 的 bf16 满 vocab 瞬态。二者都不是标定问题，是建模问题。
- `lean` 的 4 **整个**仍是峰值差分拟合值（§5.3）。
- `pp==1 非 lean` 的 3 **不再吸收任何东西**（逐张同构）。
