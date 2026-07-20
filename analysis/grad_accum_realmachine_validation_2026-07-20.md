# 梯度累积 真机 vs 仿真 验证（2026-07-20）

> 起因：给 explorer 加了"非-PP 梯度累积"展示后（commit `94b28e1`），上 116 真机核对仿真准不准。
> 结论提要：**仿真严重欠预测 116 上的 pp=1 梯度累积（m=4 时 0.381、Δ 差 ~21.6×）——但这是
> 版本相关的**：116 的 `feature/pynative-arch-evolution` 分支 pp=1 累积**不省显存**（batch 放大、
> 显存 ∝ m），而本机 `master` 分支**有省显存的微批循环**（与仿真当前模型吻合）。**core/UI 未改，
> 待定「仿真对齐哪个版本行为」后再动**（见 §5 决策）。

## 1. 测试设计（用 Δ 隔离梯度累积显存，消掉框架/池化公共开销）

DSv3 8L, pp=1, dp=2, 无重算, seq=4096, bf16/fp32-AdamW, 2 卡（116 卡 6,7），3 步。
只改 `global_batch_size` 造梯度累积步数 m（`num_accumulation_steps = GBS/(local·dp) = GBS/2`）：
- m=1 → GBS=2；m=2 → GBS=4；m=4 → GBS=8（训练日志 `trainer.py:461` 逐一确认）。
- 真机峰值 = `MemoryProbeCallback` 的 `[MEMPROBE] peak_alloc_MiB`（rank0/1 一致）。
- 仿真峰值 = `Evaluator(...num_microbatches=m...).per_stage[0].peak_bytes`（framework_reserve=0）。

harness：`.claude/skills/real-machine-memory-sim/prep_ds3_sim.py` 加 `SIM_GBS`/`SIM_RECOMPUTE` 两个
env（本次提交），使梯度累积 + 无重算配置可复现。

## 2. 数据

| 配置 | 真机 peak_alloc (MiB) | 真机 reserved (MiB) | 仿真 peak (MiB) | sim/real |
|---|---|---|---|---|
| m=1（无累积） | 19967.3 | 20528 | 20142.3 | **1.009** ✓ |
| m=2 | 32449.4 | 35370 | 21875.1 | 0.674 |
| m=4 | **57413.7** | 58166 | 21875.1 | **0.381** ⚠️ |

- 真机 m=1 = **19967.3** 与既有记分卡锚点 `DSv3 8L none (dp2)` 逐字节一致 → 测量设置可信。
- **真机显存随 m 线性**：+12482.1 MiB/步（m1→m2=+12482.1；m2→m4=+24964.3=2×；预测 m=4=57413.6 vs
  实测 57413.7）。**仿真在 m≥2 饱和**（grad_accum 桶恒 1732.8）。
- 关键差值：**Δ_real(m4−m1)=37446.4 vs Δ_sim(grad_accum)=1732.8 MiB**，仿真只抓 **4.6%**，
  欠预测 **~21.6×（~34.9 GiB）**。Δ_reserved ≈ Δ_alloc（框架/池化在 Δ 里抵消）。
- resting persistent（memory_summary）m=1 与 m=4 **相同 5230 MB** → 整个 Δ 都在瞬态激活峰。

## 3. 基础模型已验证（错的只是"累积语义"）

把真机**每卡实际 batch B=m** 喂给仿真、`num_microbatches=1`：仿真 20142/33647/60655 vs 真机
19967/32449/57414 → **1.009/1.037/1.056**。即 116 的 pp=1 "梯度累积" = **纯 batch 放大**，激活 ∝ m；
仿真的激活模型本身准（1–6%），错的只是把它当成了"省显存顺序累积"。

## 4. 根因 + 版本坑（核实后修正）

- **116 `feature/pynative-arch-evolution`（所有锚点验证目标）**：pp=1 累积**不省显存**——global_batch
  做大后一次前反向跑整个大 batch（子代理读 116 源 `trainer.py:944` 无微批循环；实测线性佐证）。
- **本机 `master`（@377c9c34）**：`training_step`（`trainer.py:1001`）**有省显存微批循环**
  `for micro_step in range(num_accumulation_steps): _split_micro_batch → _forward_backward`，
  优化器只在最后一步 → 显存恒定（**与仿真当前 grad_accum 模型吻合**）。

故"仿真对不对"**取决于 mindformers 版本**：对 master 成立、对 116 当前分支不成立（欠 21×）。
> 上一轮 explorer UI（`94b28e1`，把 pp=1 num_microbatches 当"省显存梯度累积"、显示 grad_accum
> 驻留）是按 **master 语义**建的——**对不上 116 实测**，需随下方决策修正。

## 5. 决策（2026-07-20，已定）：按「正常的多一分 grad」估计

**用户裁定**：仿真建模**正常/省显存的梯度累积语义**——pp=1 累积时**激活保持单微批(不 ∝ m)**，
峰值 = 单微批峰值 **＋ 一份常驻累计梯度**(`grad_accum` 桶)。**不追** 116 feature 分支的线性增长。

**据此 core 无需改——仿真已正确建此模型**（逐值核实）：

| m | 仿真 peak (MiB) | grad_accum (MiB) | Δ vs m=1 |
|---|---|---|---|
| 1 | 20142.3 | 0.0 | — |
| ≥2 | 21875.1 | 1732.8 | **+1732.8 = 恰一份梯度**（m≥2 饱和，就地累加不随步数增长）|

`grad_accum` = 累计 **reduced 梯度分片** = 每卡一份梯度(grad_dtype×params/dp)，pp2 真机探针标定
(P0-01)。**pp>1 已真机验证**：pp2-stage1(m=2) 记分卡 ratio **1.007**——省显存微批循环在 pp>1 下即便
116 也生效，故该桶对流水场景成立。

### 5.1 116 线性增长 = 内存泄漏? （用户新假设，待 subagent 测）
用户提出：116 feature 分支 pp=1 累积的**显存 ∝ m 可能是内存泄漏**（微批循环存在、但各 microstep 的
激活未随 `_split_micro_batch`/step 结束释放 → 累积驻留 → 线性），**而非有意的 batch 放大**。关键辨别：
- **泄漏**：`training_step` 有 `for micro_step in range(num_accumulation_steps)` 循环、`_split_micro_batch`
  切成 local 大小小微批，但激活跨 microstep 不释放 → 单 step 内显存**逐 microstep 阶梯上涨、不回落**（m 个
  递增台阶）。若如此,116 的线性是 **bug**,正常/master 行为(省显存 +1 grad)才对 → **仿真是对的**、116 待修。
- **有意 batch 放大**：无微批循环、一次前反向跑整个 (local×m) 批 → 单 step 内**一个大驼峰**(非 m 个台阶)。
辨别法(subagent 执行)：① 读 116 feature 分支**实际** `training_step` 源码(有无循环、`_split_micro_batch`
是否真切片、喂进 `_forward_backward` 的 batch 是 local 还是 local×m)；② 单 step 内**细粒度**显存 profile
(MindSpore Profiler / 逐 microstep MEMPROBE)看是"m 个不回落台阶"(泄漏)还是"一个大驼峰"(batch 放大)。

**无论结论,仿真按正常语义估的决策不变**；但结论决定 116 那 21× 差异的定性(bug 待报 vs 版本行为差异),
写进本报告并（若泄漏）给 mindformers 侧一个可复现的泄漏证据。

### 5.2 诚实 caveat（务必留档）
- 仿真按正常语义估(省显存 +1 grad)，**故对 116 `feature/pynative-arch-evolution` 分支 pp=1 大
  global_batch 累积会欠预测**(该分支实测显存∝m；m=4 仿真 0.381)。**有意选择**：建正常/正确行为、不建该
  分支的线性增长(疑似泄漏,见 §5.1)。差值=激活按 m 线性放大那部分。
- 上轮 explorer UI(`94b28e1`)按此正常语义建——**与本决策一致,保留**；仅加一句版本 caveat 到 tooltip。
- master 分支省显存微批循环(`trainer.py:1001` `_split_micro_batch`)与仿真模型吻合；116 若合入该循环、
  或修掉 §5.1 的疑似泄漏,其 pp=1 即回到省显存、与仿真一致。

*证据：本次 `[MEMPROBE]` 原始行（m=1/2/4，rank0/1）、训练日志 num_accumulation_steps=1/2/4、
仿真复算、master `trainer.py:1001` 微批循环源码。真机数据全部来自实测，无杜撰。*
