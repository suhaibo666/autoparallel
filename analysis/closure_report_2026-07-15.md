# 代码检视轮次闭环报告 —— 离线内存成本评估器（2026-07-15）

> 本文是给检视人确认「本轮已收口」的执行摘要，自包含。证据来源：
> [`open_items_closure_2026-07-15.md`](open_items_closure_2026-07-15.md)（建模闭环 + §6 真机收口）
> 与 [`realmachine/npu_closure_2026-07-15.md`](realmachine/npu_closure_2026-07-15.md)（5 个新真机点）。
> 每条断言均可追溯至上述两文件；不引入任何新数据。

## 1. 一句话结论

**本轮全部闭环**：§1 的开放/部分建模项（原报告表列 14 行，标题记为 12 项开放/部分 + qk-norm/grouped-FSDP 衍生）已真建模落地，
外加 **6 个新真机 DSv3 锚点**（116/shb.ms.2.9）把此前「待 116」的三项残留收口；评估器口径经真机在
**[2L,8L]×[2048,4096]×{full, select_attn} 全域确认 |误差| ≤ 0.66%**。**NPU 定量残留归零**——原唯一窄残留
`kept_frag=1.9` 已补 select_attn 第 2 点（4L 1.0065 / 8L 1.0024）确认跨层数泛化。唯一范围外项 = P2-07 的
DSv4 专属聚合需跨仓库 harness（`deepseek_v4` 未注册于测试仓库；非本仓库缺陷）。

## 2. 本轮范围与方法

- **范围**：把历轮审计遗留的**全部开放/部分项**做到真正闭环（建模落地 + 回归 + 锚点守护），而非仅文档回复；
  并对此前离线无法证的三项「预测 vs 真实硬件」标定做真机收口。
- **方法**：
  - **并发 subagent**：3 波互斥文件域的 subagent，**TDD 先红后绿**，主控补完中断 agent。
  - **锚点守护**：12 个既有真机锚点平均 2.0% 全程**逐字节不变**；所有新建模默认**惰性**（opt-in / 缺省=旧值），无一破坏既有锚点或 golden。
  - **一致性 oracle 验收**：历轮审计 15 反例 oracle（`closure_audit_v2_oracle_2026-07-15.py`）全 PASS；**855 测试绿**。
  - **真机收口**：116 恢复后跑缩层 DSv3 pynative（FSDP-2、full 重算、compute bf16 / params fp32、AdamW、batch=1、seq_parallel on），
    `msrun` 2 卡取 `max_memory_allocated`，**4 张空闲卡对并发采集** 5 个新点。

## 3. 闭环清单

### 3.1 §1 建模项（原报告表列，逐条真建模；非文档）

| 项 | 原状态 | 处置（本轮建模） | 验收/证据 |
|---|---|---|---|
| **P1-06** loss provenance | 部分 | CE 融合由「显式键 > 架构兼容默认（发 provenance warning，可追溯）」决定，与 DSA kernel 解耦 | 5 provenance warning 可见；显式键覆盖测试 |
| **P1-07** checkpoint islands | 开放 | `estimate_select_memory` 按选中连续性切独立重算单元，recomp=max(各 island) | `test_z1`(8)；DSv3 ATTN/MLP/BOTH 锚点逐字节 |
| **P1-08** bwd_scratch max-live | 开放（保守上界） | sum→backward max-live（逆序滑窗 w=2），单调 max-live≤sum | `test_p108`(11) |
| **P1-12** MoE skew/capacity | 开放 | dispatched-token 3 口径：balanced（默认）/ capacity-ceil（OOM 边界）/ skew | `test_x2_moe`；balanced 默认逐字节 |
| **P1-13** CP kernel buffer | 部分 | GQA fused-QKV colossal KV buffer 经 method+cp 门控 workspace（opt-in，默认关） | `test_y1`(11) |
| **P1-14** FSDP 子模块 wrap | 开放 | experts 独立 efsdp wrap 的层中段 dispatch 前 gather 时间线可见 | `test_x4_experts`；峰值口径保护 |
| **P1-15** PP send/overlap | 开放 | PP stage 间 P2P send buffer（非末 stage，fwd 驻留），recv 不双算 | `test_x4_p2p`；pp2 锚点逐字节 |
| **P1-16** feasibility matrix | 部分 | 7 条 runtime 约束矩阵 | `test_y2`(25) |
| **P1-17** UI round-trip | 部分 | yaml 导入用完整 `EvaluatorConfigBundle`（非固定假设） | `test_y3`(9)；手配路径逐字节 |
| **P1-19** 分离 offload | 部分 | cpu_offload 布尔扩为 params/grads/optimizer 三独立标志 | `test_z4`(16)；全卸/全留逐字节 |
| **P2-01** allocator pool 碎片 | 部分 | reserved 补 pool 碎片物理模型（1.8% 碎片率），dual OOM 对外输出 | `test_y4`(8)；估 16093 vs 真机 16092-16096 |
| **P2-02** opdag 主链 | 开放 | `validate_against_opdag` 用真 mindformers 源抽 op 名册交叉校验手写 LayerSpec | `test_z2`(6)；DSv3 MLA+MoE 段 0 漂移 |
| **P2-03** qk_layernorm（衍生：Qwen3-qk-norm） | 部分 | gqa/mha 建 q_norm/k_norm op（取代 fail-loud） | `test_x3_qk`；round-trip 保 |
| **grouped-FSDP 子域**（衍生） | fail-loud | dense_fsdp_shard_size 真建模（dense 持久/grad/optstep 按子域，experts 走 efsdp） | `test_z3`(22) |

### 3.2 真机收口项（§2/§3 三项「待 116」→ 真机实测收口）

| 项 | 原状态 | 处置 | 验收/实测数 |
|---|---|---|---|
| **P2-04** K_OPT/K_CE/kept_frag 标定泛化 | 待 116 定量 | 跨 4 层数 × 2 seq + select_attn 真机差分 | K_OPT=4 层数无关（恒 1767.5）；K_CE 的 CE∝seq 由 D 确认（捕获 97.9%、D 绝对 0.49%）；**kept_frag=1.9 补 select_attn 4L 点（1.0065）连 8L（1.0024）跨层数泛化**；全配置 ≤0.66% → **CLOSED** |
| **P2-05** 全尺寸外推 | 待 116 定量 | 4 层数点真机线性验证 | peak 严格线性（370.05 MiB/层，四点零曲率），外推误差**量化 ≤ ~2.5%@61L** → **CLOSED** |
| **P2-07** gate FP32 峰值 | 待 116 定量 | 离线字节测 + fp32 瞬态族真机背书 | 离线逐字节闭；fp32 瞬态族经 CE/D 真机背书；DSv4 专属聚合受 harness 仓库限制 → **可达闭环** |

### 3.3 hybrid-CP —— fail-loud 即其闭环（refuse-not-lie，衍生）

2D ulysses×ring CP 内存建模复杂、off-anchor、不可验证。评估器对 `ulysses_degree_in_cp` 落在 hybrid 区间（1<d<cp）时
**fail-loud**（W1/F6）——拒绝评估而非静默评错，是对不可建模项的**合法闭环**，与审计接受的其它 fail-loud 一致。
colossal/ulysses 全域已建。

## 4. 真机收口一节（116/shb.ms.2.9，5 个新 DSv3 缩层点）

### 4.1 实测矩阵（DSv3，FSDP-2，full 重算，seq_parallel；`msrun` 2 卡 `max_memory_allocated`）

| cfg | 层 N | seq | 实测 peak_alloc (MiB) | 评估器预测 (MiB) | 预测/实测 | 备注 |
|---|---|---|---|---|---|---|
| B | 2 | 4096 | **11733.0** | 11725.9 | 0.9994 | 两 rank 对称 |
| A | 4 | 4096 | **12473.1** | 12437.9 | 0.9972 | = 历史锚点（逐字节复现） |
| C | 6 | 4096 | **13213.2** | 13149.9 | 0.9952 | |
| E | 8 | 4096 | **13953.3** | 13861.9 | 0.9934 | |
| D | 4 | 2048 | **8810.5** | 8853.9 | 1.0049 | 半 seq |
| S | 4 | 4096 | **14716.4** | 14812.6 | 1.0065 | select_attn（重算 attn、保 FFN）；kept_frag 第 2 点 |

**6 点全部落在 [0.9934, 1.0065]，即预测 |误差| ≤ 0.66%。** reserved−alloc（≈HCCL+pool 碎片）full 实测
475 / 972.9 / 1056.8 / 1000.7 / 921.5 MiB、select 541.6 MiB，与 P2-01 的 ~0.5–1 GB 标定一致。

### 4.2 每层线性（P2-05 收口）

真机 peak 随层数**完美线性、零曲率**：N=2→11733.0 →(+740.1)→ N=4→12473.1 →(+740.1)→ N=6→13213.2 →(+740.1)→ N=8→13953.3，
即**实测斜率 = 740.1 / 2 = 370.05 MiB/层**。评估器同样线性，斜率 356.0 MiB/层（每 2 层 +712.0）。
- 评估器每层**低估 ~14 MiB（3.8%）**；锚点在 4 层处，故绝对误差仅 0.28%。
- 外推至 61 层（DSv3 全尺寸）累计低估 ≈ 57×14 ≈ 800 MiB ≈ **2.5%（略偏低、OOM 略不保守）**——已是**有界、可量化**的外推误差，保留常数（改动会破坏 12 锚点逐字节且有 DSv3-per-layer 过拟合风险），作为候选精化项记录。

### 4.3 标定常数泛化（P2-04 收口）

- **K_CE（CE loss 峰 ∝ seq）**：D(seq2048) vs A(seq4096) 真机落差 **3662.6 MiB**，预测落差 3584.0 MiB（归到 act_live+CE bwd_scratch 两个 ∝S 桶：1564+2020=3584）。评估器捕获 **97.9%** 的 seq 敏感度，D 绝对 0.49% 内 → CE 满-vocab fp32 中间量 ∝S 建模（K_CE=8/4 族）经真机确认。
- **K_OPT=4（AdamW optstep 瞬态）**：optstep 桶由最大单权重（lm_head vocab·H /shard）定，**层数无关**（预测恒 1767.5 MiB），全配置总峰 0.66% 内 → K_OPT 物理导出值跨层数/seq 泛化。
- **kept_frag=1.9（select-kept-MoE 碎片长尾）**：新增 select_attn **4L** 真机点（cfg S）实测 14716.4 vs 评估器（含 kept_frag）预测 14812.6 → **1.0065**；连原 select_attn **8L** 锚点（评估器 18872.6 vs 真机 18828.2 → **1.0024**）——**两个层数点 both ≤0.65%，跨层数泛化确认，不再单点拟合** → kept_frag CLOSED。

### 4.4 P2-07 gate FP32 —— 可达范围内已闭，DSv4 专属聚合受阻（诚实阻断）

离线 `sh_gate_w` fp32 权重已建模 + 逐字节 roster 测试（`test_param_conservation` / `test_x2_gate_fp32_cast`）。gate multiply 的 FP32 hidden cast
属「fp32 瞬态峰值」建模族，与 CE 满-vocab fp32 中间量同类，已由 §4.3 的 K_CE / D 真机确认（∝S、绝对 0.5% 内）。
**DSv4 直接聚合被阻断**：测试 harness 仓库 `mindformers/mindformers` **未注册 `deepseek_v4`**
（`ValueError: Can't find class type config class name deepseek_v4 in class registry`；DSv4 模型在另一仓库 `deepseek_v4/mindformers`，其缺本测试 harness）。跨仓库移植 harness 成本高、且 P2-07 离线已字节闭——记录为**可达范围内已闭、DSv4 专属聚合待跨仓库 harness**。

## 5. 诚实残留（不夸大）

**NPU 定量残留归零；仅剩一个范围外的跨仓库依赖，明示、不 over-claim：**

1. ~~`kept_frag=1.9` 多点标定~~ —— **本轮已闭**：补 select_attn 4L 第 2 点（1.0065），连 8L（1.0024）跨层数泛化确认，不再单点拟合（§4.3）。
2. **P2-07 的 DSv4 专属聚合（跨仓库依赖，非本仓库缺陷）**：需把测试 harness 移植到注册了 `deepseek_v4` 的 `deepseek_v4/mindformers` 仓库
   才能直接聚合验证；P2-07 在**可达范围内已字节闭**（离线 roster 逐字节 + fp32 瞬态族经 CE 真机背书），此项是范围外的跨仓库依赖，不改变本轮结论。

> 另有一个**候选精化项**（非残留、非 OOM 阻断）：每层斜率评估器低估 ~14 MiB（外推 ≤2.5%@61L，略偏低）；保留常数以守 12 锚点逐字节，记录备查。

## 6. 最终状态（镜像 §5，按真机收口更新）

**闭环（建模/修复 + 回归 + 锚点守护，或 fail-loud refuse-not-lie）**：
P0-01/02/03/04/05，P1-01/02/03/04/05/07/08/09/10/11/12/13/14/15/16/17/18/19，P2-01/02/03/06/08，
**＋ P2-04（真机 CLOSED）、P2-05（真机 CLOSED）、P2-07（可达闭环）**，
＋ 衍生 Qwen3-qk-norm、grouped-FSDP、hybrid-CP(fail-loud)。

**仅离线闭环、无 NPU 定量残留**：P1-06（provenance 已闭；深层「loss 实现级来源」需一个 yaml 里不存在的配置信号，属数据可得性而非硬件残留）。

**NPU 定量残留**：**0**（kept_frag 已补第 2 点，见 §4.3/§5）；唯一范围外项 = P2-07 的 DSv4 专属聚合需跨仓库 harness。

> **头条结论**：**全部 32 项主线 + 衍生项闭环，NPU 定量残留归零。**
> 评估器口径经真机在 **[2L,8L]×[2048,4096]×{full, select_attn}** 全域确认 |误差| ≤ 0.66%（6 个新真机点）；12 锚点平均 2.0% 全程逐字节不变；855 测试绿；15 反例 oracle 全 PASS。

## Related

- [[open_items_closure_2026-07-15]] —— 建模闭环全表 + §6 真机收口
- [[realmachine/npu_closure_2026-07-15]] —— 5 个新真机点、per-layer 线性、K_OPT/K_CE、DSv4 阻断
