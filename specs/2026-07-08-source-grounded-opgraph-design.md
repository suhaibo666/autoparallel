# 源码级 op-DAG 静态提取建模 — 设计文档

> **主线（一条主线）**：把手写的 op 图 builder（`cost_eval/layers/*.py`）换成一个**离线静态分析器**——
> 它读**当前的 mindformers 源码 + 运行 yaml**，解析出模型的 op-DAG（含 dtype/cast 落点），再套一小份
> **稳定的 per-op-类型 bprop 语义库**导出 save-set，喂给现有事件模拟骨架。**代码一改，重跑提取即更新**，
> op-DAG 不再冻结在 Python 里。profiler CSV 只做**校验 oracle**。
>
> **Source baseline**：mindformers @ `E:\97-codes\torch_parallel\mindformers`（真机路径
> `parallel_core/training_graph/transformer/`）；配置 `mf_suhaibo/configs/deepseek3/*.yaml`；
> 评估器 @ 分支 `feat/unified-llm-modelspec`。
> **状态**：设计（待用户过审）→ 实现计划 → 实现。

---

## 0. 背景：为什么重做（动机）

现有 op 图是**手写猜测**：`cost_eval/layers/attention.py` 等里每个 op 的 `OpSpec.saves`（反向保留哪些激活）
是人拍的，再按 dtype 定尺寸。后果——**loss 峰欠预测残差**（真机 2026-07-08 锚点）：

| 锚点 | ratio | 残差根因 |
|---|---|---|
| select `self_attn`（留整个 MoE-FFN） | **0.823** | grouped-GEMM 中间量 + fp32-cast 操作数未建 |
| cp2-none / select `mlp` / DSv4 | 0.911 / 0.940 / 0.968 | 同族：fp32 cast 横切量 + 小中间量欠计 |

根因统一：**手写 op 图没建真机的 cast 落点与中间量**。fp32-layernorm 那一刀只补了 norm 一档，
matmul/激活的 fp32 操作数、grouped-GEMM 的 permute/pad/cast 仍缺。逐 op 手补 = 越补越"固化"，
且 **mindformers 代码一改就烂**。

**用户裁决（本次对齐）**：不逐 op 手补，改为**从代码静态推导 op-DAG**。

---

## 1. 硬约束（已锁定，不可动）

1. **离线静态**：只做 **AST 静态分析** mindformers `.py` 源码 + 运行 yaml。**绝不运行 mindformers**
   （无 MindSpore 环境；不 trace、不 dump IR、不 profile 取图）。
2. **op-DAG = 再生数据工件**（JSON），评估器**消费**它；**不得**把 op-DAG 结构写死在 `cost_eval/layers/*.py`。
   代码变 → 重跑提取器 → 新 JSON → 离线模型零改动自动更新。
3. **profiler CSV 只做校验 oracle**（`analysis/realmachine/*/operator_memory.csv`），不作提取源。
4. **骨架不动**：事件模拟（10 桶、k_ce/k_opt、per-stage 峰）保持不变，只把它消费的 save-set 换成 DAG 导出。
5. **源忠实**：每条 op/save 断言可回指 mindformers `file:line`；解析不了的**fail-loud**，绝不静默产错图。

> [!note] 关于"手维护表"的澄清
> 用户否掉的是"**per-模型冻结的 saves 猜测表**"（随模型烂）。本设计里唯一手维护的是
> §4 的 **per-op-类型 bprop 语义库**（matmul 存两操作数、layernorm 存 input+rstd……约 15 条教科书事实）,
> 它按 **op 类型**而非模型组织，Cell 怎么重构都不变，且有 profiler 兜底真值——与被否的那个正交。

---

## 2. 架构：三阶段 + oracle

```
mindformers 源码树 + 运行 yaml
   │  【阶段1】静态提取器（AST）  ── 见 §3
   ▼
op-DAG.json（节点: op_type / 符号 in-out shape / dtype / cast 落点；边）
   │  【阶段2】per-op-类型 bprop-pin 语义库（~15 稳定规则）  ── 见 §4
   ▼
save-set（哪些张量 pin 到反向 + 什么 dtype）
   │  【阶段3】符号 shape → DimTable 具体化，喂现有事件模拟  ── 见 §5
   ▼
per-stage 峰值  ──【校验】── profiler operator_memory.csv 活跃集 oracle  ── 见 §6
```

**为什么这样分**：阶段1 把"**易变的模型结构**"（op 序列、cast、shape）从当前代码提取——这是代码改动的
唯一入口；阶段2 是"**稳定的自动微分语义**"，正交且长寿；阶段3 复用已验证的事件骨架（full 重算 0.991、
cp-full 0.998 逐字节对齐，不能破）。

---

## 3. 阶段 1 — 静态提取器（AST）

**输入**：mindformers 源码根 + 已解析合并的运行 config（yaml → dict，复用现有 `load_mindformers_yaml`）。
**输出**：`op-DAG.json`（schema 见 §3.4）。**三遍解析**：

### 3.1 Pass A — 模块树解析（config 驱动）

从 config 解析出**本次真机实际实例化的模块树**，定位要走的 construct()：

- 顶层 `GPTModel` → `TransformerLayer` 的 `submodules`（`ModuleSpec`）——决定注意力 Cell
  （`SelfAttention`/`MultiLatentAttention`/`dsa_attention`）与 FFN Cell（`MoELayer`/`MLP`）。
- `build_module(submodules.mlp, ...)`（`transformer_layer.py:149`）这类**动态派发**：解析**装配 spec 的静态
  Python**（`get_gpt_layer_local_spec` 之类的 spec 构造函数本身是静态代码），拿到具体类。
- config 决定的开关：`num_layers`、`moe`/dense、`multi_latent_attention`、`compute_dtype`、
  `layernorm_compute_dtype`、`recompute`（mode/select cell）、`capacity_factor`、`n_routed_experts` 等。

> [!warning] **风险 R1（最大）**：`build_module(submodules.X)` 的静态解析
> submodules 由 spec 构造函数拼装（也是静态 Python，但有 config 分支）。缓解：解析器**顺着 spec 构造函数
> 的 AST + config 值**求解，而非猜。**求解不出的 submodule → fail-loud**，报出哪个 config 键没覆盖。

### 3.2 Pass B — `__init__` 绑定（self.X → op 类型 + dtype/shape 参数）

解析每个 Cell 的 `__init__` AST，建 `self.<name> → (op_type, params)` 绑定表。实测词表（真机路径 grep）：

| `__init__` 里的绑定 | op 类型 | 关键参数（决定 shape/dtype/save） |
|---|---|---|
| `self.cast = P.Cast()` / `ops.Cast()` | **Cast** | 目标 dtype 来自调用点第 2 实参 |
| `self.linear_qkv = ColumnParallelLinear(in,out,compute_dtype=…)` | **MatMul** | in/out feature、compute_dtype、bias |
| `self.linear_proj = RowParallelLinear(…)` | **MatMul** | 同上 |
| `self.activation_func = <SiLU/GeLU/Swiglu>` | **Activation** | 逐元素、compute_dtype |
| `self.core_attention = FlashAttention(…)` | **FlashAttention** | head 数、causal |
| `self.q_layernorm/k_layernorm/input_layernorm = <Norm>` | **Norm** | `layernorm_compute_dtype`(fp32) |
| `self.add/mul/sub = P.Add/Mul/Sub()` | **Elementwise** | 线性/非线性 |
| `self.reshape/transpose/split/shape` | **View/Meta** | 元信息（一般不新增 save） |

dtype 解析源：config 的 `compute_dtype`、`param_init_type`、`layernorm_compute_dtype` + 调用点 `.astype()`/
`self.cast(x, dtype)` 的字面 dtype。

### 3.3 Pass C — construct() 走查（op 序列 + cast 落点 + 符号 shape）

AST 走查每个 Cell 的 `construct()`，按语句序发射 op 节点。要识别的调用惯用法（实测频次）：
`self.cast(` ×48、`self.reshape(` ×67、`ColumnParallelLinear/RowParallelLinear` 调用、`self.add/mul/sub`、
`self.activation_func(`、`self.core_attention(`、`q/k_layernorm(`、`ops.Transpose/SplitWithSize`。

- **dtype 传播**：从入参 dtype 出发，遇 `self.cast(x, t)`/`.astype(t)` 更新该 SSA 值 dtype → **cast 落点显式可见**
  （这是 fp32 残差的根，静态就能抓）。
- **符号 shape**：每个中间值挂符号 shape（`B·S·H`、`B·N·S·D` 等），维度符号来自 config dim。
- **fail-loud**：未知 `self.X` / 未知调用 / 解析不出 dtype → 报错，不产半图。

### 3.4 op-DAG.json schema（草案）

```json
{
  "baseline": {"mf_commit": "...", "config": "pretrain_deepseek3_671b.yaml"},
  "cells": {
    "TransformerLayer": {
      "nodes": [
        {"id": 12, "op": "Cast", "src": "attention.py:207",
         "in": ["h:BSH:bf16"], "out": "h32:BSH:fp32", "to_dtype": "fp32"},
        {"id": 13, "op": "MatMul", "src": "...:210", "module": "ColumnParallelLinear",
         "in": ["h32:BSH:fp32", "Wq:H·rq:fp32"], "out": "q:BS·rq:bf16"}
      ],
      "edges": [[12,13]]
    }
  }
}
```

节点携 `src`（`file:line`，源忠实）、符号 shape、dtype。**saves 不在此产出**——留给阶段2。

---

## 4. 阶段 2 — per-op-类型 bprop-pin 语义库（稳定，~15 条）

遍历 DAG，按 **op 类型**判定其反向 pin 谁，导出 save-set（张量 + dtype，dtype 取该节点**cast 后**的 dtype）：

| op 类型 | pin（save_for_backward） | save dtype | 依据 |
|---|---|---|---|
| MatMul / BMM / **GroupedMatMul** | **两个操作数** | 操作数 compute dtype（升 fp32 则 fp32） | `d a=dy·bᵀ, d b=aᵀ·dy` 都要操作数 |
| Add（残差） | 无 | — | 线性，梯度直传 |
| Mul（gate 逐元素） | 两操作数 | compute dtype | `d x=dy·y` |
| Norm（Layer/RMS） | input +(mean,rstd) | **fp32** | 反向需归一化统计 |
| Softmax | output | compute dtype | `d x=(dy−(dy·y)·1)·y` |
| Activation（SiLU/GeLU/Swiglu） | input | compute dtype | `d x=dy·f'(x)` |
| **Cast** | **无**（只 dtype 元信息） | — | VJP 是把梯度 cast 回去；**其 output 由下游消费者 pin** |
| Reshape/Transpose/Permute | 无 | —（物化的非连续 copy 由下游 pin） | 视图/元信息 |
| FlashAttention | q,k,v + lse | compute dtype | flash 反向重算需 qkv+logsumexp |
| Dropout | mask | 1B | 反向乘同一 mask |
| Gather/Embedding | indices | int | 反向 scatter 用索引 |

**关键机理（这次修 fp32 残差的核心）**：`Cast` 自身不存激活，但它产出的 fp32 buffer 被**下游非线性/matmul
消费者**的 save 钉住 → 该 buffer 归到消费者、按 fp32 计。dedup：一张量被多消费者 pin 只算一次。

**这是唯一手维护件**，但按 op 类型、约 15 条教科书事实、有 §6 profiler 兜底 → 稳定长寿。

---

## 5. 阶段 3 — 参数化 + 接入现有骨架

- **符号 shape → 具体**：DAG 的 `B·S·H` 等符号用 `DimTable` 维度代入，得每个 save 的字节数。
- **接入**：save-set → 现有模拟器的 `activation_saves`/`act_live`（`structure_mem.py`）。事件、10 桶、
  k_ce/k_opt、per-stage 峰**全不动**。等价于把 `OpSpec.saves` 的手写值替换为 DAG 导出值。
- **替换面**：`cost_eval/layers/*.py` 的手写 builder → 退役为"DAG 消费适配"（或保留作退化端参照，见 §8）。

---

## 6. 校验（profiler oracle）

- **逐 op 活跃集重建**：`operator_memory.csv` 用 `alloc ≤ T_peak < release` 重建峰时活跃集（现成技法），
  与 DAG 导出 save-set **逐 op 比**。每条静态解析因此成为**可核验断言**。
- **锚点矩阵**（回归门）：
  - **不许破**：DSv3 4L 全重算 12409.5（逐字节）、cp-full 0.998、select both 0.991。
  - **要修好**：select `self_attn` 0.823→目标 ≥0.95、`mlp` 0.940→↑、DSv4 0.968→↑、cp2-none 0.911→↑。
- 数据：`analysis/realmachine/{cp2_none,select_attn,select_mlp,dsv4_fused,pp2_norecomp,...}/`。

---

## 7. 诚实边界 / 动态性处理

- **控制流/数据依赖**：`recompute` 分支、`capacity_factor`、`n_routed_experts` 等 → **config 绑定后即静态**
  （config 决定分支）。真·数据依赖（如动态 drop）→ fail-loud + 记录声明假设。
- **build_module 动态派发**（R1）：靠 config + spec 构造函数 AST 解析，解不出即报错（不猜）。
- **MindSpore autodiff 不可读**：§4 是"理论 + oracle"，非 MindSpore 源码导出——A 方案的固有折扣，已明示。
- **一条 config 一套 DAG**：MLA/GQA、MoE/dense、DSv4 各自结构不同 → 各出各的 DAG（本就是不同模型）；
  同结构不同 S/B/并行由 §5 符号参数化泛化。

---

## 8. 迁移策略

- 现有 `cost_eval/layers/*.py` 手写 builder：**先并存**（DAG 消费器为主，手写 builder 作退化参照），
  用锚点测试守住"已验证配置逐字节不破"（full 重算 / cp-full）。全绿且残差修好后再退役手写 builder。
- `ATTN_REGISTRY`/`FFN_REGISTRY` 的派发语义保留，只是 builder 内部改为消费 DAG。

---

## 9. 再推导工作流（交付后）

```
mindformers 代码改动
   └─▶ python extract_opdag.py --mf <src> --config <yaml> --out opdag.json   # 纯静态,不跑 mf
          └─▶ 离线评估器消费新 opdag.json,模型自动更新(零 Python 改动)
```

---

## 10. 开放问题 / 风险

- **R1**（§3.1）：`build_module` submodule 的静态解析深度——最大不确定性。先做 DSv3(MLA+MoE) 一条路径打通，
  证明可行再推广。
- **R2**：view 物化（非连续 reshape/transpose 触发 copy）静态判定——保守起见先按"不物化"（元信息），
  由 oracle 校出需修的点。
- **R3**：grouped-GEMM 的 permute/capacity-pad 中间量是否落在 construct() 显式 op（vs 融合 kernel 内部）
  —— 若在 kernel 内部则静态不可见，退回 oracle 标定 margin（明示为标定常数）。
- **粒度**：先 DSv3 单条端到端打通 + 锚点验证，再扩 DSv4/dense/GQA。

## 关联
- 上游残差与整改：`2026-07-07-memory-model-reference.md`（§14 锚点、§15 残差）、`2026-07-06-audit-remediation.md`。
- 现有 op 图设计：`2026-07-01-unified-llm-modelspec-design.md`（§7-10 op 图）。
- 分析方法论：`.claude/skills/source-faithful-analysis/`（源忠实 + 抓本质）。
