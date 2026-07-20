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

## 5. 待定决策（仿真对齐哪个行为）

| 选项 | 做法 | 代价/风险 |
|---|---|---|
| A 查清116意图再改 | 上116确认 batch-放大是有意设计还是待修临时态、会不会很快合入 master 省显存循环 | 最稳，多一轮真机核查 |
| B 对齐116(batch放大) | pp=1 且 num_microbatches>1 时仿真按 batch 放大建模（激活∝m），撤回上轮"省显存 grad_accum" UI 框架 | 匹配当前验证目标；若116后续合master又得反转 |
| C 保持现状(对齐master) | 维持省显存模型，视116 feature为临时/回归，仅文档标注版本差异 | 不改数值；但对116当前分支欠21× |
| D 双模式可切换 | 两种语义都建、配置旗标切、默认对齐116 | 最全但工作量最大 |

**建议**：倾向 A→B（先核实116意图，确认是设计则按 batch 放大改并撤回上轮 UI 框架），但这是
**版本路线判断**，需用户定夺。**在决策前 core/UI 不动**，避免追一个可能的临时状态。

*证据：本次 `[MEMPROBE]` 原始行（m=1/2/4，rank0/1）、训练日志 num_accumulation_steps=1/2/4、
仿真复算、master `trainer.py:1001` 微批循环源码。真机数据全部来自实测，无杜撰。*
