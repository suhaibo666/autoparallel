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

### 5.1 116 线性增长 = 内存泄漏? —— 已测：**否,是有意 batch 放大**（2026-07-20 subagent 定性）

用户假设"线性增长可能是泄漏(微批循环在跑、激活跨 microstep 未释放)"。**双路取证结论：不是泄漏,
是 batch 放大 by design——feature 分支 pp=1 根本没有省显存的微批循环**,泄漏在结构上不可能发生。

**(A) 源码（116 `feature/pynative-arch-evolution` @ `0f4c2f15`，已提交无本地改）**：
- `num_accumulation_steps = GBS/(dp·local)`（`trainer.py:458`）**只喂给 dataset builder**（`:488` 作
  `num_grad_acc`），**不驱动任何前向循环**。
- `_inner_train_loop`（`:719`）：一个 batch → 一次 `training_step` → **无累积循环**。
- pp=1 走 `SingleStageSchedule.step`（`single_stage.py:34`）：**一次 `model(**inputs)` + 一次
  `loss.backward()`**,无循环无切片；`training_step` **每次调用都跑完整 optimizer**。
- 喂进那一次前向的 batch = `GBS/dp = local×m`（`utils.py:378` `dataset.batch(global_batch//dp)`）——
  累积因子完全落在 dataset batch 尺寸里。
- **`_split_micro_batch` 在 feature 分支不存在**（`git grep` 无）。对比 **master**：`training_step`
  `for micro_step in range(num_accumulation_steps): _split_micro_batch(切 local) → _forward_backward`、
  优化器只在末微批 → **峰值 m-恒定(省显存)**。**feature 的 schedule 重构丢掉了这个微批循环**（把 batching
  搬进 dataset + 单 `SingleStageSchedule.step`）→ 峰值 ∝ m。

**(B) 实测（m=4, 8L, dp=2, 无重算, 卡6/7；monkeypatch 探 SingleStageSchedule.step）**：
- **`fwd_calls_total == train_step`**（每步恰 1 次前向,非 4 次）→ **无 m=4 微批循环**。
- **`input_shapes=(4,4096)`** → batch 维 = 4 = local(1)×m(4) = GBS/dp → **整个累积批一次喂入**。
- 单 step 内显存 = **一个大驼峰**：persistent 5230 → after-fwd 41237(单次 +36007 激活) → after-bwd
  6963(激活释放、梯度留),**每步相同、回落到基线,无跨步堆积**。**不是** m 个不回落台阶。
- peak 57413.7 精确复现 m=4;线性拟合 `peak≈7485+12482·m`（floor=persistent+框架,slope=激活∝batch）。

**判据落点**：泄漏需"循环在 ∧ 切片对 ∧ 跨 microstep 不释放";feature 分支前两条都不成立(无循环、无
`_split_micro_batch`),**故无 microstep、无从堆积 → 结构上不可能泄漏**。是 batch 放大 by construction。

**给 mindformers 侧的可行结论（非 bug 单,是行为回归提示）**：feature 分支 pp=1 梯度累积**不再省显存**
（master 靠 `_split_micro_batch` 微批循环保持峰值 m-恒定;schedule 重构换成 `.batch(GBS/dp)`+单
`SingleStageSchedule.step` → 峰值 ∝ m）。若要 pp=1 省显存累积,修法=在 `SingleStageSchedule.step`
内/外恢复类 master 的微批循环。**仿真按正常语义估的决策不变,仿真是对的**。

### 5.2 诚实 caveat（务必留档）
- 仿真按正常语义估(省显存 +1 grad)，**故对 116 `feature/pynative-arch-evolution` 分支 pp=1 大
  global_batch 累积会欠预测**(该分支实测显存∝m；m=4 仿真 0.381)。**有意选择**：建正常/正确行为、不建该
  分支的线性增长(疑似泄漏,见 §5.1)。差值=激活按 m 线性放大那部分。
- 上轮 explorer UI(`94b28e1`)按此正常语义建——**与本决策一致,保留**；仅加一句版本 caveat 到 tooltip。
- master 分支省显存微批循环(`trainer.py:1001` `_split_micro_batch`)与仿真模型吻合；116 若合入该循环、
  或修掉 §5.1 的疑似泄漏,其 pp=1 即回到省显存、与仿真一致。

*证据：本次 `[MEMPROBE]` 原始行（m=1/2/4，rank0/1）、训练日志 num_accumulation_steps=1/2/4、
仿真复算、master `trainer.py:1001` 微批循环源码。真机数据全部来自实测，无杜撰。*

---

## 6. DSv4 补测（用 `deepseek_v4/mindformers` 代码，2026-07-20）

用户要求用 **`/home/suhaibo/workspace/deepseek_v4/mindformers`**（DSv4 主代码,**≠** DSv3 harness 的
`mindformers/mindformers`）跑 DSv4 4L 缩层、梯度累积开时对比。结论：**机制已源码坐实 = 省显存(与仿真
同 regime),但真机数值未测到（当前 HEAD 两处版本漂移回归把训练卡在 step 0,未杜撰数字）。**

**机制（源码,`deepseek_v4/mindformers@master` HEAD `c3df3ffd3`）**：`training_step`（`trainer.py:1106`）
是**省显存微批循环**——`for micro_step in range(num_accumulation_steps): _next_batch()(取 local=1 一
微批) → _forward_backward(各自前反向) …优化器只在末微批`。每微批激活先释放、只累计梯度常驻。**这与
DSv3 那次的 `feature/pynative-arch-evolution`(无循环→batch放大)相反,DSv4 master 行为 = DSv3 master**。
→ DSv4 累积 Δ 应 ≈ 一份梯度,正是仿真所建 regime（Δ_sim=1887.7,memory-saving **MATCH**,非 batch 放大
欠预测）。**仿真对 DSv4 梯度累积在正确 regime,无需 batch-放大修正。**

**真机数值未测到（诚实留白,两处 blocker 均源码举证,均是 2026-07-15/18 后的版本漂移,非 07-06 锚点态）**：
- **FUSED=1** 建模即崩：`csa.py:29` import 新融合算子名(`npu_sparse_flash_mla` 等),而本机 hyper_parallel
  wheel 只导出旧名 → ImportError → fail-loud "DSV4 fused ops unavailable"。csa.py 07-15/18 重写了融合 API,
  wheel 未同步。
- **FUSED=0** 建成但 step0 崩：`AssertionError: hsdp expects uniform original parameter dtype but got
  {bf16, fp32}`（`fully_shard/param_group.py:413`）。模型现有 **15 个 bf16 参数**(MoE/FFN 的
  fc1/fc2/experts/shared/output_layer)与 59 个 fp32 混布(07-15 精度对齐 commit 引入)→ 触发 FSDP 组内
  dtype 一致断言。

**两个 caveat**：① 数值 Δ 未实测,机制结论靠源码(高可信但非实测);② 即便解阻,当前 HEAD 的
**bf16-FFN 权重**已偏离仿真的 uniform-fp32-param 假设 → DSv4 **绝对** persistent/grad 会与仿真的
14971.8 有出入(锚点 15415.5/14971.8 是 07-06 fp32 态标定的)。这属**另一独立漂移**(DSv4 模型精度布局变了、
仿真 DSv4 模型可能需随之更新),与梯度累积 regime 无关。

**harness**：`prep_dsv4align.py` 加 `SIM_GBS`(镜像 DSv3,向后兼容),本次提交;`run_dsv4_gbs.sh` 留在 116。

*证据：`prep_dsv4align.py` 两 GBS 的 `num_accumulation_steps=1/4` 日志、两 blocker 的源码报错(csa.py:29 /
param_group.py:413)、`deepseek_v4/mindformers@c3df3ffd3 trainer.py:1106` 微批循环源码、仿真复算 14971.8/
16859.5。真机峰值**未测到即不报**,无杜撰。*
