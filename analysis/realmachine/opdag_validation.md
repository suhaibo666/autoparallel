# 源码级 op-DAG 提取 —— 字节级验收 + 度量发现（2026-07-09）

> 承接 `specs/2026-07-08-source-grounded-opgraph-*.md`、`specs/2026-07-09-opgraph-phase2-revision.md`。
> 记录 T11 consumer 桥打通后，DAG 导出 save-set 的**真字节**，及"度量优先"揭示的**对 18% 残差的认知重构**。

## 1. DAG 提取已达成（Tasks 1-8 + T9 + T11）

从真 mindformers 源码（`parallel_core/training_graph/transformer/`）**静态**提取 op-DAG（纯 `ast`，
不执行 mindformers），符号 shape 传播 + DimTable 代入得字节。**代码改重跑即更新，未冻结**。417 测试绿。

## 2. 字节级交叉验证（DSv3 缩层：S=4096,B=1,H=1792,E=8,cap=2048,moe_F=1024）

**MLA save-set = 94 MiB/层**（`extract_cell` + `infer_shapes` + `dag_saved_bytes`）：

| save | 符号 shape | dtype | MiB |
|---|---|---|---|
| x（q/kv_down 输入） | S·B·H | bf16 | 14.0 |
| q_compressed（q_layernorm 输入） | S·B·q_lora_rank | **fp32** | 24.0 |
| kv_compressed（kv_layernorm 输入） | S·B·kv_lora_rank | **fp32** | 8.0 |
| query / key（FA） | S·B·n_heads·(qk_head+qk_rope) | bf16 | 12.0 ×2 |
| value（FA） | S·B·n_heads·v_head_dim | bf16 | 12.0 |
| attn_out（proj 输入） | S·B·(n_heads·v_head_dim) | bf16 | 12.0 |

两个 norm 存 **fp32**（32 MiB；bf16 只 16 MiB，+16 是 fp32 残差修正）——机制正确。

**MoE FFNGroupedGEMM 真激活 = 152 MiB/层**（排除权重 w1/w2），与**手写 builder 逐字节一致**：

| DAG | 手写 `build_moe_ffn_ops` | 符号 | MiB |
|---|---|---|---|
| dispatched_input | `disp` | (S·B·topk·C, H) | 56.0 |
| fc1_output | `e_g` | (S·B·topk·C, 2·moe_F) | 64.0 |
| intermediate | `e_act` | (S·B·topk·C, moe_F) | 32.0 |

> [!important] **度量优先的关键发现**：**手写 builder 早已精确计了 grouped-GEMM 中间量**（56/64/32
> 逐字节）。DAG 只是**独立交叉验证它是对的**。故 **Task 8「捕获 grouped-GEMM」并不能闭合 self_attn
> 0.823 缺口——那些字节本来就在账上**。之前「18% 根因是 grouped-GEMM 中间量漏建」的假设**被度量推翻**。

## 3. 18% 残差真身（profiler live-set 归因）

`analysis/realmachine/select_attn/`（DSv3 8L dp=2 keep-FFN，峰=ScatterAddExt loss 反向，
算子级 live=13630 + persistent≈5198 = 18828 真机 vs 估计器 15361，**0.816**）：

| 真机 size 类 | ×数 | 小计 MiB | 估计器 |
|---|---|---|---|
| 2020 fp32 vocab | 3 | 6060 | ✅ loss 区已计 |
| 1010 logits + 883.8 grad + 441.9 | — | 2336 | ✅ 已计 |
| **112 MiB** | **14** | **1568** | ❌ 保留 FFN/MoE 激活（112≈56×2，疑 **fp32 版** dispatched） |
| **<100 MiB 尾** | **313** | **3666** | ❌ fp32 cast 横切 / permute 碎片 / RmsNorm / MatMulExt 中间量 |

缺口 `18828−15361 = 3467 ≈ 313 个小张量尾 3666`。

## 4. 诚实硬结论：残差在 op 图粒度之下

18% 残差**不是**几个大 grouped-GEMM 缓冲（已计），而是 **313 个小张量的长尾**——fp32 cast 副本 +
permute 碎片 + 分配器/autograd 保留物。这些的 **loss 峰存活性是分配器/自动微分保留的现实，不是显式
construct() op 能干净导出的**。

**即"扩提取器消 margin"也无法完全闭合**：T10 顶多把长尾里**显式可建**的部分（token_dispatcher permute、
router one-hot、shared-expert `super().construct()` saves、grouped-GEMM 操作数的 fp32 cast）建出来，
剩余碎片尾仍属**标定 margin**。源（DAG）反驳了"提取能消掉 margin"的期望——这正是源忠实分析该讲的话。

## 5. DAG 这套已赢下的 + 待定

**已赢**：① 可再推导 op-DAG（核心诉求）；② 字节级交叉验证手写模型（grouped-GEMM 对齐、fp32-norm 确认）；
③ 独立证明手写 op 图那部分无误。**闭合不了的**：18% 残差（op 图粒度之下的长尾）。

**方向已定 = B（2026-07-09 用户裁决，已实现）**：给 **select-kept-MoE** 层 loss 峰一个 profiler 标定 margin
`kept_frag_factor`（DSv3 preset=1.9，自 select_attn 18828 标定），明示为标定常数。
- **效果**：select_attn（keep-FFN，靶心）**0.823 → 1.001**（OOM-安全）;keep-attn（MoE 被重算）/full/
  no-recompute（pp2-stage1 0.962）/融合-CE（DSv4）**全不触发、逐字节不动**（`mem_timeline._is_kept`
  gate:仅 `is_select` 且 FFN 未被重算）。硬门 12409.5 不破。421 测试绿。
- **实现**：`mem_timeline` 新增 `kept_frag` 桶（仅 loss-BWD 事件、按 1.9× 当前 kept-MoE 激活计）；
  `DimTable/LLMConfig.kept_frag_factor`（默认 0=回归安全）;preset + 转换器（MoE 且非融合-CE）注入;
  `tests/test_kept_frag_margin.py` 锁定。
- **诚实边界**：1.9 是 coarse 平台常数（自 DSv3 单锚点标定），非 op 图导出;其它 select-kept-MoE 配置
  （不同 S/B/层数）近似覆盖、未逐一真机核对。A（T10 抽显式长尾）为可选后续深化，非本轮。

## 关联
- `specs/2026-07-08-source-grounded-opgraph-design.md`（三阶段+oracle、R1/R2/R3）
- `specs/2026-07-07-memory-model-reference.md` §15（同族残差：DSv4-7% / pp2-stage1 / kept-MoE）
- `analysis/realmachine/select_attn/DIAGNOSIS.md`（原始 live-set 重构）
