# 新架构下 仿真器 vs 真机 —— 综合差异测试报告（2026-07-09）

> **一句话结论**：当前仿真器（含**源码级 op-DAG 交叉验证** + **B 标定 margin** + fp32-norm 全部修复）
> 对 **12 个真机锚点**，**平均 |ratio−1| = 2.4%**、**9/12 落在 ±5%**、中位比值 0.993。退化/对齐端逐字节；
> 残余 2 个 OOM-不安全欠预测（cp2-none 0.925 / select-mlp 0.940）均为已归因、已文档化的同族残差。

## 1. 测试口径（诚实前提）

- **「新架构」对仿真预测的实际影响**：op-DAG 静态提取（Tasks 1-8/T9/T11）是**离线分析工具**，度量已证实
  它与手写字节模型**逐字节一致**（`opdag_validation.md`）——**不改变**仿真预测，只**交叉验证**。真正改变
  预测的是 **B 标定 margin（select-kept-MoE）** 与更早的 **fp32-norm**。故本报告 = 当前仿真器对全部真机锚点的比值。
- **方法**：本地跑 `Evaluator.evaluate()`（无 MindSpore / 无 NPU / 无网络），比对**已捕获**的真机峰值
  （`analysis/realmachine/*/DIAGNOSIS.md`、参考文档 §14）。真机值为 `max_memory_allocated` MiB。
- **复现脚本**：见附录；每个锚点精确重建其 `ParallelConfig` + `RecomputeSpec`。

## 2. 综合差异矩阵（12 锚点，覆盖 full / cp / pp / select / DSv4）

| 锚点 | regime | 仿真 MiB | 真机 MiB | ratio | 判定 |
|---|---|---|---|---|---|
| DSv3 4L full (dp2,sp) | full | 12409.5 | 12473 | **0.995** | ≈ 对齐 |
| DSv3 8L full (dp2) | full | 13833.1 | 13953 | **0.991** | ≈ |
| DSv3 4L full ep=2 | full+ep | 12367.5 | 12474 | **0.991** | ≈ |
| cp2 colossal full 4L (B2) | cp+full | 12409.5 | 12433 | **0.998** | ✅ 安全 |
| cp2 ulysses full 4L (B2) | cp+full | 12409.5 | 12441 | **0.997** | ✅ |
| pp2-stage0 (optstep) | pp+norecomp | 10794.1 | 10246 | **1.053** | ✅ 过预测(安全) |
| pp2-stage1 (loss,k_ce=8) | pp+norecomp | 43899.1 | 45655 | **0.962** | ≈ |
| **cp2-none (loss,k_ce=4)** | cp+norecomp | 18612.0 | 20119 | **0.925** | ⚠️ 欠(OOM-不安全) |
| **select self_attn (keep-FFN)** | select | 18843.8 | 18828 | **1.001** | ✅ 安全（B margin 修复） |
| **select mlp (keep-attn)** | select | 14821.6 | 15765 | **0.940** | ⚠️ 欠(OOM-不安全) |
| select both (=full,退化端) | select | 13973.1 | 13953 | **1.001** | ✅ |
| DSv4-fused (base) | dsv4+norecomp | 14915.5 | 15416 | **0.968** | ≈ |

## 3. 汇总统计

- 平均 **|ratio−1| = 2.4%**；中位比值 **0.993**。
- **ratio ∈ [0.95, 1.05]：9/12**（75%）。
- 最大**过预测** 1.053（pp2-stage0，OOM-安全）；最大**欠预测** 0.925（cp2-none）。
- **OOM-不安全**（欠预测 <0.95）：仅 2 个 —— `cp2-none`(0.925)、`select mlp`(0.940)。
- **过预测 >1.05**：仅 1 个 —— `pp2-stage0`(1.053，安全侧)。

## 4. 分 regime 解读

- **full / cp-full / select-退化端（5 锚点）：0.991–0.998，逐字节级对齐**。op 图 + 事件模型 + cp÷cp
  （Bug A 修正）在这些制度下机理正确。cp2 full 的 0.998 是**真对齐**（此前 0.996 是 B/cp 数值抵消蒙对）。
- **select self_attn（靶心）：0.823 → 1.001** —— B 标定 margin 把 select-keep-MoE 的碎片长尾补上（OOM-安全）。
- **pp2-stage0：1.053 过预测**（安全侧）：fp32-norm 后无重算逐层反向略超 optstep，峰移 bwd@N。
- **DSv4-fused：0.968**：融合 CE kernel 内部量（D-5，另一族残差，B margin 按设计不触发）。

## 5. 两个 OOM-不安全残差（已归因，未闭合）

| 锚点 | ratio | 根因（已文档化） | 为何未用 B margin 修 |
|---|---|---|---|
| **cp2-none** | 0.925 | 无重算 loss 区 k_ce=4 标定（cp 下满 vocab fp32 共存份数）偏少；`bwd_scratch` 略欠 | B margin 只 gate select-kept-MoE；no-recompute 由 k_ce 制度化平衡，再加会双算（见 `_is_kept`） |
| **select mlp** | 0.940 | keep-attn（**MoE 被重算**）→ 保留的是 attn 碎片尾（比 MoE 小）；MLA 的 fp32-cast/view 中间量欠计 | B margin 只对**保留-MoE**层生效；keep-attn 的 MoE 已重算 → 不触发（避免误伤） |

二者与 DSv4-7%（D-5）、pp2-stage1（0.962）**同族**：无重算/保留模块在 loss 峰的 fp32-cast + 小张量长尾，
**op 图粒度之下**（源码级 op-DAG 提取已证实，`opdag_validation.md`）。可选后续：给 keep-attn / no-recompute
各自标定 margin（同 B 机理，per-module 因子），或 T10 抽显式长尾——本轮未做。

## 6. 未真机核对（仅仿真预测，无真机对比）

cp>2、pp>2 每-stage（受 mindformers 栈 pp>2 优化器 bug 限制，§15）、VPP、swap、TP+MoE、SP+MoE
（栈不支持）——这些配置**仅有仿真预测、无真机数据**，不在本报告 12 锚点内。如需，须在 NPU 服务器
（192.168.9.116）跑新 profiler（`.claude/skills/real-machine-memory-sim/`）。

## 附录：复现
脚本逐锚点重建配置（`build_dsv3_spec` / `validate_dsv4align.evaluate` + `ParallelConfig`/`RecomputeSpec`）。
真机数据源：`analysis/realmachine/{cp2_none,select_attn,select_mlp,dsv4_fused,pp2_norecomp}/`、参考文档 §14。

## 关联
- `specs/2026-07-07-memory-model-reference.md` §14（真机锚点）、§15（诚实边界 + 残差族）
- `analysis/realmachine/opdag_validation.md`（op-DAG 字节级验收 + B margin 决策）
