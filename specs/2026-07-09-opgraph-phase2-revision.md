# op-DAG 提取 Phase 2 修订 —— 全 shape 推断 + 扩提取器消 margin

> **承接** `2026-07-08-source-grounded-opgraph-{design,plan}.md`。Tasks 1-8（提取管线）已完成并提交
> （schema→bprop→binder→walker→module_resolver→extractor→MLA→MoE，368 测试绿）。本文档记录 Task 8
> 完成时**新浮现的两个事实**，及据此把 Task 9-11 重排为**最忠实**路线（用户 2026-07-09 裁决）。

## 一、Task 8 完成时浮现的两个事实

**① R3——MoE 有真·不透明部分**（`9f5ab51` 已诚实记录）：
- **可静态提取**（已入 save-set）：`FFNGroupedGEMM` 的 GroupedMatMul×2 + swiglu + 操作数
  （`dispatched_input` permute 后 token 缓冲=18% 的根、专家权重 w1/w2、swiglu 中间量）。子代理发现
  `Morph` 包的是**可读 Python `forward_func`**（非黑盒 kernel），故可达。
- **静态不可分解**：`TopKRouter`（topk/argmax/aux-loss，数据依赖）、`token_dispatcher` 的 AllToAll
  permute/capacity-pad/sort 缓冲、`SharedExpertMLPInterleaved`（`super().construct()` 惯用法未提取）。
  `MoELayer` 顶层在 router 边界 **fail-loud**（不伪造）。

**② shape 是占位符**：walker 发射 `name:?:dtype`——**只跟 dtype，shape=`?`**。算字节（byte 级内存 +
oracle 对标）必须先补 shape。

## 二、忠实路线的关键难点

要"全 shape 推断"，每个 MatMul 需要其 **(in_features, out_features)**——这些不在当前 DAG 里，而在各 Cell
`__init__` 里以 config 符号给出（如 MLA：`linear_q_down_proj: H→q_lora_rank`、
`linear_q_up_proj: q_lora_rank→n_heads·qk_head_dim`、`linear_proj: n_heads·v_head_dim→H`）。故 shape 推断
**依赖先从 `__init__` / `build_module` 调用抠出各 linear 的符号维度**。

## 三、Task 9-13 重排（最忠实）

| # | 任务 | 产物 | 验收 |
|---|---|---|---|
| **T9** | **linear 维度捕获 + 全 shape 推断 pass** | `cost_eval/opdag/shape_infer.py` + extractor 补捕 linear (in,out) 符号维度 | MLA/MLP/MoE DAG 每张量得符号 shape；单测断言关键 shape（q_up 输出=n_heads·qk_head_dim、grouped-GEMM 操作数=E·cap·H 等） |
| **T10** | **扩提取器消 margin** | shared-experts `super().construct()` 内联；permute 缓冲按 capacity_factor 建模；router intermediates 尽量建 | MoELayer 顶层不再 fail-loud（或不透明集显著缩小）；shared-expert MatMul 入 DAG |
| **T11** | **consumer 桥（shape→字节）** | `cost_eval/opdag/consumer.py`：符号 shape 用 DimTable 代入得字节 → OpSpec.saves 兼容 | 单测：MLA/MoE save-set 字节数对 profiler 活跃集合理 |
| **T12** | **DSv3 装配接入 + 回归门** | 装配器 opt-in 吃 DAG（手写 builder 默认/退化参照） | **DSv3 4L 全重算 12409.5 逐字节不破**、cp-full 0.998、select both 0.991 |
| **T13** | **oracle 验收矩阵 + CLI + 文档** | `tools/extract_opdag.py`；`analysis/realmachine/opdag_validation.md`；参考文档 §14/§15 更新 | **self_attn 0.823→≥0.95**；残余不透明部分明示为标定 margin |

## 四、诚实边界（保留）

即便"最忠实"，`TopKRouter` 的 topk/argmax/aux-loss 是**数据依赖控制流**，静态不可完全分解 → 其
routing intermediates（one-hot/gating 缓冲）若无法从 config 符号建模，仍退回**标定 margin（明示为标定常数）**。
目标是把 margin 压到最小，而非假装为零。

## 关联
- `2026-07-08-source-grounded-opgraph-design.md`（三阶段+oracle、R1/R2/R3）、`...-plan.md`（Task 1-8）。
- 残差与锚点：`2026-07-07-memory-model-reference.md` §14/§15。
