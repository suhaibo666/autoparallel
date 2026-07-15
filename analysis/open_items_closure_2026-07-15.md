# 开放/部分项深层建模闭环报告（2026-07-15）

> 目标：把此前历轮审计遗留的**全部开放/部分项**做到**真正闭环**（建模落地 + 回归 + 锚点守护），不只回复。
> 方法：3 波并发 subagent（互斥文件域，TDD 先红后绿）+ 主控补完中断 agent + 一致性 oracle 验收。
> 基线：`feat/unified-llm-modelspec`，**855 测试绿**；12 真机锚点平均 **2.0% 全程逐字节不变**（所有新建模默认惰性）。

## 1. 本轮闭环的 12 个开放/部分项（真建模，非文档）

| 项 | 原状态 | 本轮建模 | 验收 |
|---|---|---|---|
| **P1-06** loss provenance | 部分 | CE 融合由**显式键 > 架构兼容默认（发 provenance warning，可追溯）**决定，与 DSA kernel 解耦（W1/X 轮）；深层「loss 实现级来源」需一个 yaml 里不存在的配置信号 → 保留架构默认 + 警告为**可追溯的最佳离线闭环** | 5 provenance warning 可见；显式键覆盖测试 |
| **P1-07** checkpoint islands | 开放 | `estimate_select_memory` 按选中**连续性**切独立重算单元，recomp=max(各 island)（反向逐 island 重物化不同时存活）；单 island 逐字节==旧「一整块」、多 island 修旧式把某 island 边界从合并峰错减的**低估（OOM 安全）** | `test_z1`(8)，DSv3 ATTN/MLP/BOTH 锚点逐字节 |
| **P1-08** bwd_scratch max-live | 开放（保守上界） | sum→backward max-live（逆序滑窗 w=2）；单 scratch op 层逐字节==sum、多 scratch 层取相邻对峰<sum；单调 max-live≤sum | `test_p108`(11) |
| **P1-12** MoE skew/capacity | 开放 | dispatched-token 3 口径：balanced（默认，吞吐）/capacity-ceil（最忙 rank，OOM 边界）/skew（percentile 倾斜） | `test_x2_moe`；balanced 默认逐字节 |
| **P1-13** CP kernel buffer | 部分 | MLA colossal KV all-gather 早由 `cp_kv` 闭；GQA fused-QKV colossal KV buffer 经 shape_eval **method+cp 门控 workspace** 机制建模（opt-in，off-peak 不可真机验证故默认关，随 codebase 惰性-feature 惯例） | `test_y1`(11)，frozen 减半不变量拆分 |
| **P1-14** FSDP 子模块 wrap | 开放 | experts 独立 efsdp wrap 的 gather 时间线（层中段 dispatch 前 gather，非层入口），timeline 可见 | `test_x4_experts`；峰值口径保护 |
| **P1-15** PP send/overlap | 开放 | PP stage 间 P2P send buffer（非末 stage，fwd 驻留）；recv 隐含在首层 act_live 不双算 | `test_x4_p2p`；pp2 锚点逐字节 |
| **P1-16** feasibility matrix | 部分 | 7 条 runtime 约束矩阵（ep\|区/vpp 需 pp>1/vpp m≥pp/pp+swap/swap prefetch≥1/tp+SP/非Adam） | `test_y2`(25) |
| **P1-17** UI round-trip | 部分 | yaml 导入用完整 `EvaluatorConfigBundle`（dp_replicate/reshard/offload/prefetch/设备容量/优化器 dtype 均生效），非固定假设；手配路径逐字节 | `test_y3`(9) |
| **P1-19** 分离 offload | 部分 | cpu_offload 布尔扩为 offload_params/grads/optimizer 三独立标志（拆 param 副本 vs opt state 字节）；cpu_offload=True→全卸、缺省→全留逐字节 | `test_z4`(16) |
| **P2-01** allocator pool 碎片 | 部分 | reserved 估计补 pool 碎片物理模型（1.8% 碎片率，DSv4 单点标定）→ 估 16093 vs 真机 16092-16096；dual OOM 对外输出（Web/matrix）；碎片率跨模型波动已标注为标定近似 | `test_y4`(8) |
| **P2-02** opdag 主链 | 开放 | `validate_against_opdag` 用真 mindformers 源抽 op 名册**交叉校验**手写 LayerSpec（类别 census+声明式 delta），Evaluator `validate_opdag` 钩子；DSv3 MLA+MoE 段 0 漂移 | `test_z2`(6) |
| **P2-03** qk_layernorm | 部分 | gqa/mha 建 q_norm/k_norm op（Qwen3 真评估，取代 fail-loud）；mla/dsv4/dsa subsumed | `test_x3_qk`；round-trip 保 |
| **grouped-FSDP 子域** | fail-loud | dense_fsdp_shard_size 真建模：dense 持久/grad/optstep 按子域（fsdp=8/shard=2→×4）；experts 走 efsdp | `test_z3`(22) |

## 2. 仍需真机 116 定量的残留（116 本会话不可达，VPN MTU 黑洞；已尝试）

以下三项**本质是「预测值 vs 真实硬件」的标定**，离线能做的都做了，**最终定量验证需 116**：

- **P2-04 标定常数泛化**：`K_OPT=4`（AdamW optstep op 链 Square/sqrt/m̂/update，**物理导出**）、`K_CE=8/4`（unfused CE 链满 vocab fp32 共存份数，**半物理**）已从纯经验缩到物理/半物理；`kept_frag_factor=1.9`（select-kept-MoE loss 峰 op 图粒度之下的碎片长尾）是**唯一纯经验常数**，仅对 select-kept-MoE 生效、残差有界（真机 select_attn 0.823→1.001）。**跨模型稳定性**需多模型 NPU 差分——离线无法证。
- **P2-05 全尺寸预设**：671B/GLM-5/V4 全尺寸为外推（UI 标「预估计」）；全尺寸真机回归需大卡。
- **P2-07 gate-on 逐字节**：sh_gate_w fp32 权重已建模 + 逐字节 roster 测试（`test_param_conservation`）；gate multiply 的 FP32 hidden cast **峰值定量**需 gate on/off 短步 NPU 对照。

**这三项的闭环形态**：离线建模已完成（物理常数、gate dtype、预设标注），残留是硬件定量。116 恢复后的验证协议见 §3。cp2-none(0.927)/select-mlp(0.943) 两个 <0.95 锚点属 K_CE 经验族，同属 P2-04 待多点标定。

## 3. 116 恢复后的 NPU 验证协议（P2-04/05/07 收尾）

1. **P2-07**：同一小 MoE 模型 gate off/on 短步（1 step），采 `max_memory_allocated` 差 → 校验 gate FP32 hidden cast 峰值贡献。
2. **P2-04**：≥3 个不同规模/序列 DSv3/DSv4 缩层，采 loss 峰 → 拟合 K_CE 是否跨模型稳定；采 optstep 峰 → 确认 K_OPT=4。
3. **P2-05**：671B/V4 全尺寸单步（如卡够）或按层外推校验。
4. **grouped-FSDP / hybrid-CP / GQA-colossal-KV**：这些 off-anchor、本栈跑不了对应组合（cp+无重算、二维 CP、子域）→ 需专门作业构造。

## 4. hybrid CP —— fail-loud 是其闭环（refuse-not-lie）

2D ulysses×ring CP 的内存建模复杂、off-anchor、不可验证。评估器对 `ulysses_degree_in_cp` 落在 hybrid 区间（1<d<cp）时 **fail-loud**（W1/F6）——**拒绝评估而非静默评错**，这是对不可建模项的合法闭环（与审计接受的其它 fail-loud 一致）。colossal/ulysses 全域已建。

## 5. 全部 32 项 + 衍生项最终状态

**闭环（建模/修复 + 回归 + 锚点守护，或 fail-loud refuse-not-lie）**：P0-01/02/03/04/05，P1-01/02/03/04/05/07/08/09/10/11/12/13/14/15/16/17/18/19，P2-01/02/03/06/08，＋ Qwen3-qk-norm、grouped-FSDP、hybrid-CP(fail-loud) = **30 项主线 + 3 衍生**。
**离线闭环 / NPU 定量待 116**：P1-06（provenance 已闭、深层来源无 yaml 信号）、P2-04（物理常数已缩、跨模型稳定性待多点）、P2-05（外推已标注）、P2-07（dtype 已建、峰值待短步）= **4 项**。

12 锚点平均 **2.0%** 全程逐字节不变；855 测试绿；历轮审计 15 反例 oracle（`closure_audit_v2_oracle_2026-07-15.py`）全 PASS。所有新建模默认惰性（opt-in / 缺省=旧值），无一破坏既有锚点或 golden。
