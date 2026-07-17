# PyNative Cost Evaluator

面向 **MindFormers PyNative**（TorchTitan 移植、对等 Megatron 的 5D 训练栈）的**离线并行策略代价评估器**设计与实施计划。

> 目标：给定一份专家配置（5D 并行 + recompute + swap）+ 一份声明式模型描述（ModelSpec），**纯解析、离线**估算该配置在目标 NPU 集群上的 **① 每卡峰值显存 + OOM 判定 + 构成拆解** 和 **② 单步时间 + 瓶颈 + MFU**。它是自动并行系统里的 **evaluator（评估器）**，与 **optimizer（搜索器）** 解耦——先把评估器做准，搜索器是将来包在它外面的循环。

$$\text{自动并行} = \text{搜索器(optimizer)} \circ \underbrace{\text{评估器(evaluator)}}_{\text{本仓库}}$$

## 快速上手：交互式内存实验台

```bash
python serve_explorer.py     # → http://127.0.0.1:8765
```

改模型结构/并行切分/重算 → 实时重算峰值显存；逐层 op-DAG + 全 stage 内存时间线 + 各桶分解。
**完整使用说明见 [README_explorer.md](README_explorer.md)**。

## 状态

- ✅ 设计定稿（4 份 spec）
- ✅ P0（内存建模）实施计划定稿（TDD，13 任务 + 后续增量）
- ⬜ P0 实现（`cost_eval/` 代码，待执行计划）
- ⬜ P1（时间模型：roofline + bubble + overlap）
- 🔶 P1（时间模型）T0 地基完成：schedule.py 中立化、timesim 骨架+解耦 lint（双向禁令+白名单）、
  opdag 通信提取（comm_probe）+ GPTModel 级段（loss/embedding/head）+ walker 保真（嵌套实参物化/
  opaque_calls）、TimedOpSeq producer（fwd 装配+TP/FSDP/EP/CP 通信注入+SP/feature 分片状态机+
  bwd/recompute 展开+依赖边反转）+ IR 不变量。设计 specs/2026-07-16-step-time-cost-model-design.md，
  计划 plans/2026-07-16-t0-timesim-foundation.md，下一步 T1（op_cost + segment_sim + pipeline_sim）。

## 目录

### `specs/` — 设计文档

| 文档 | 内容 |
|------|------|
| [主架构](specs/2026-06-23-pynative-cost-evaluator-design.md) | 总体设计：声明式 op 图 ModelSpec、CostRecord 内存契约（`params`/`saves`/`workspace`）、5D 切分 / recompute / swap 三套变换、事件驱动内存时间线、验证策略、分阶段 |
| [P0 内存细化](specs/2026-06-29-p0-modelspec-and-memory-design.md) | ModelSpec 数据结构、dense/moe op 图、静态内存（param/grad/opt）与激活内存的逐步计算、事件驱动峰值仿真（recompute 反向尖峰 / FSDP 预取双缓冲 / swap 预取） |
| [实现架构](specs/2026-06-29-evaluator-implementation-design.md) | 9 个逻辑模块（M1–M9）的职责/理论依据、数据流图、时序图、接口设计、可测试性 |
| [核心模块内部](specs/2026-06-29-core-modules-m4-m5-m6-internals.md) | M4 `shape_eval` / M5 `static_mem` / M6 `mem_timeline` 的内部算法、伪代码、关键正确性点（efsdp 不重复算 tp、专家纯 EP） |

### `plans/` — 实施计划

| 文档 | 内容 |
|------|------|
| [P0 内存建模](plans/2026-06-30-p0-memory-modeling.md) | 13 个 TDD 任务（M1→M3→M4→M5→M6→M7 + 端到端），每任务含完整代码、测试、commit；末尾"后续增量"覆盖 FSDP 预取/select 重算/op 级 swap |

## 阅读顺序

1. 主架构 → 建立全局心智模型（评估器=搜索器外的内核、声明式 op 图）。
2. P0 内存细化 → 看具体怎么算显存（静态 + 激活 + 时间线峰值）。
3. 实现架构 → 模块划分与接口。
4. 核心模块内部 → M4/M5/M6 算法细节。
5. P0 计划 → 落地实现。

## 关键设计点

- **声明式 op 图（数据，非脚本/AST/trace）**：模型按层类型声明 op list，每 op 带符号 shape + 切分标注；与框架脚本解耦、零漂移。
- **per-op 内存契约**：每 op 自声明 `params`/`saves`/`workspace`；融合/非融合自动跟随（`saves`=autodiff `save_for_backward`）。
- **事件驱动峰值仿真**：峰值 = 沿 fwd→bwd 时间线取 max，显式建模 recompute 反向尖峰、FSDP all-gather 预取双缓冲、swap 预取。
- **source-grounded**：关键切分语义对照 MindFormers 源码核实（如专家纯 EP 见 `expert_parallel.py:330`），不杜撰。

## 范围

- 覆盖：dense + MoE/EP + 实验性混合注意力（MLA/CSA/hybrid/MTP）；全 5D（DP/TP/PP/CP/EP）+ recompute + swap。
- v1 纯解析（无真机 profiling），预留缩层 profiling 标定接口（`η` 效率因子单列一层，只换常数不动结构）。
