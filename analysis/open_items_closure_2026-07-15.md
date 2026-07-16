# 开放/部分项深层建模闭环报告（2026-07-15）

> 目标：把此前历轮审计遗留的**全部开放/部分项**做到**真正闭环**（建模落地 + 回归 + 锚点守护），不只回复。
> 方法：3 波并发 subagent（互斥文件域，TDD 先红后绿）+ 主控补完中断 agent + 一致性 oracle 验收。
> 基线：`feat/unified-llm-modelspec`，**855 测试绿**；12 真机锚点平均 **2.0% 全程逐字节不变**（所有新建模默认惰性）。

## 1. 本轮处理的 12 个开放/部分项（真建模；部分仍未闭）

> [!correction] 2026-07-16 复核（见 [`closure_report_verification_2026-07-16.md`](closure_report_verification_2026-07-16.md) §4）
> 原标题「本轮闭环的 12 个」over-claim。复核将 **P1-08 降为开放/高风险**，**P1-13/14/15/16、P2-01 降为部分闭环**（getattr-hack 可达、生命周期不全、非上界、无完整性证明）。各行末已就地标注「复核订正」。P1-06/07/17/19 未见新阻断，维持原判但 P1-06/07 属离线契约（见 §5 订正）。

| 项 | 原状态 | 本轮建模 | 验收 |
|---|---|---|---|
| **P1-06** loss provenance | 部分 | CE 融合由**显式键 > 架构兼容默认（发 provenance warning，可追溯）**决定，与 DSA kernel 解耦（W1/X 轮）；深层「loss 实现级来源」需一个 yaml 里不存在的配置信号 → 保留架构默认 + 警告为**可追溯的最佳离线闭环** | 5 provenance warning 可见；显式键覆盖测试 |
| **P1-07** checkpoint islands | 开放 | `estimate_select_memory` 按选中**连续性**切独立重算单元，recomp=max(各 island)（反向逐 island 重物化不同时存活）；单 island 逐字节==旧「一整块」、多 island 修旧式把某 island 边界从合并峰错减的**低估（OOM 安全）** | `test_z1`(8)，DSv3 ATTN/MLP/BOTH 锚点逐字节 |
| **P1-08** bwd_scratch max-live | 开放（保守上界） | sum→backward max-live（逆序滑窗 w=2）；单 scratch op 层逐字节==sum、多 scratch 层取相邻对峰<sum；单调 max-live≤sum。**→ 复核订正 OPEN/高风险（F10）**：固定 window=2 无 lifetime/profiler 证据；「≤sum」只证更紧、不证仍是实际峰上界；反例 `[4000,0,3000]→4000` 仅在两 scratch 不共存时才安全 → **潜在 OOM 欠估** | `test_p108`(11) |
| **P1-12** MoE skew/capacity | 开放 | dispatched-token 3 口径：balanced（默认，吞吐）/capacity-ceil（最忙 rank，OOM 边界）/skew（percentile 倾斜） | `test_x2_moe`；balanced 默认逐字节 |
| **P1-13** CP kernel buffer | 部分 | MLA colossal KV all-gather 早由 `cp_kv` 闭；GQA fused-QKV colossal KV buffer 经 shape_eval **method+cp 门控 workspace** 机制建模（opt-in，off-peak 不可真机验证故默认关，随 codebase 惰性-feature 惯例）。**→ 复核订正 PARTIAL（F8）**：原仅经测试 getattr-hack (`cp_kv_allgather_buffer`) 可达，公共 `LLMConfig` 不支持该字段（companion commit 已补公共字段） | `test_y1`(11)，frozen 减半不变量拆分 |
| **P1-14** FSDP 子模块 wrap | 开放 | experts 独立 efsdp wrap 的 gather 时间线（层中段 dispatch 前 gather，非层入口），timeline 可见。**→ 复核订正 PARTIAL（F11）**：仅 forward 分裂 experts 子生命周期；backward 仍按整层 `param_full_bytes` gather，无 experts 独立 backward 生命周期 | `test_x4_experts`；峰值口径保护 |
| **P1-15** PP send/overlap | 开放 | PP stage 间 P2P send buffer（非末 stage，fwd 驻留）；recv 隐含在首层 act_live 不双算。**→ 复核订正 PARTIAL（F8/F11）**：forward send 可用；`pipeline_parallel_overlap_p2p` overlap 子功能原仅测试 getattr-hack 可达（companion commit 已补公共字段），且 backward grad-P2P 未建 | `test_x4_p2p`；pp2 锚点逐字节 |
| **P1-16** feasibility matrix | 部分 | 7 条 runtime 约束矩阵（ep\|区/vpp 需 pp>1/vpp m≥pp/pp+swap/swap prefetch≥1/tp+SP/非Adam）。**→ 复核订正 PARTIAL（F12）**：实现所列 **7 条**规则，但无 runtime 枚举的完整性证明 → 称「覆盖 7 条」而非「完整矩阵」 | `test_y2`(25) |
| **P1-17** UI round-trip | 部分 | yaml 导入用完整 `EvaluatorConfigBundle`（dp_replicate/reshard/offload/prefetch/设备容量/优化器 dtype 均生效），非固定假设；手配路径逐字节 | `test_y3`(9) |
| **P1-19** 分离 offload | 部分 | cpu_offload 布尔扩为 offload_params/grads/optimizer 三独立标志（拆 param 副本 vs opt state 字节）；cpu_offload=True→全卸、缺省→全留逐字节 | `test_z4`(16) |
| **P2-01** allocator pool 碎片 | 部分 | reserved 估计补 pool 碎片物理模型（1.8% 碎片率，DSv4 单点标定）→ 估 16093 vs 真机 16092-16096；dual OOM 对外输出（Web/matrix）；碎片率跨模型波动已标注为标定近似。**→ 复核订正 PARTIAL（F5）**：1.8% 是**单点标定**、非可证上界；六点 reserved 误差**变号**且达 ±419 MiB（C=−419）→ 不宣称 reserved OOM 安全，allocated/reserved 应分别给安全余量 | `test_y4`(8) |
| **P2-02** opdag 主链 | 开放 | `validate_against_opdag` 用真 mindformers 源抽 op 名册**交叉校验**手写 LayerSpec（类别 census+声明式 delta），Evaluator `validate_opdag` 钩子；DSv3 MLA+MoE 段 0 漂移 | `test_z2`(6) |
| **P2-03** qk_layernorm | 部分 | gqa/mha 建 q_norm/k_norm op（Qwen3 真评估，取代 fail-loud）；mla/dsv4/dsa subsumed | `test_x3_qk`；round-trip 保 |
| **grouped-FSDP 子域** | fail-loud | dense_fsdp_shard_size 真建模：dense 持久/grad/optstep 按子域（fsdp=8/shard=2→×4）；experts 走 efsdp | `test_z3`(22) |

> [!correction] 2026-07-16 复核：原 `[!done]` 声称「本节三项残留已用真机实测收口」为 over-claim。实际是**部分进展**——取得 5 个新 DSv3 点 + 1 重放（覆盖 6/16 组合），但 **P2-04/05/07 仍开放**（K_OPT 不可辨识、外推未验证、未做 gate off/on）。详见 **§6** 订正与
> [`realmachine/npu_closure_2026-07-15.md`](realmachine/npu_closure_2026-07-15.md)。下方 §2/§3 对「残留是硬件定量、需 116」的描述本身正确，予以保留。

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

> [!correction] 2026-07-16 复核（F5/F8/F10/F11/F12 + §4 三项开放）
> 原「30 项主线闭环」把 P1-08、P1-13/14/15/16、P2-01 计入闭环，且把 P2-04/05/07 计入「待 116」而正文判 CLOSED。订正分档如下。

**离线代码级闭环（无新阻断回归）**：P0-01/02/03/04/05，P1-01/02/03/04/05/09/10/11/12/17/18/19，P2-02(注)/03(注)/06/08，＋ Qwen3-qk-norm、grouped-FSDP、hybrid-CP(fail-loud)。P1-09 更由 `b3e7712` 116 算子探针取得真机证据。
- 注：P2-02 opdag strict 有假绿路径、P2-03 YAML 适配器仍拒 qk-norm，实为**部分**，companion commit 修复中。

**部分闭环（实现存在但不完整 / 不可达 / 非上界）**：**P2-01**（单点标定非上界，±419 MiB）、**P1-08**（开放/高风险，固定 window=2 无 lifetime 证据）、**P1-13**（公共字段原缺，companion 已补）、**P1-14**（仅 forward）、**P1-15**（仅 forward-send，grad-P2P 未建）、**P1-16**（覆盖 7 条非完整矩阵）。

**NPU 定量待真机（仍开放，见 §6 订正）**：**P2-04**（K_OPT 不可辨识、未跨模型）、**P2-05**（仅 2–8L、61L 外推低估 ~800 MiB）、**P2-07**（未做 gate off/on、router dtype 被忽略）；另 P1-06 属可追溯离线降级、P1-07 缺真实 checkpoint 边界证据（离线契约）。

12 锚点平均 **2.0%** 全程逐字节不变；**当前全量 889 测试通过**（本报告提交时为 855，后续提交增用例）；历轮审计 15 反例 oracle（`closure_audit_v2_oracle_2026-07-15.py`）全 PASS。所有新建模默认惰性（opt-in / 缺省=旧值），无一破坏既有锚点或 golden。

## 6. 真机收口（2026-07-15；2026-07-16 复核订正）

> [!correction] 2026-07-16 复核（F1/F2/F3/F4）
> 本节原判 P2-04/05 CLOSED、P2-07「可达闭环」、「NPU 定量残留归零」，均被复核否决。订正后：三项**仍开放**；下表 6 行只覆盖 6/16 组合，A 为重放（新点 5 个）。

§2/§3 的三项「待 116」残留取得**部分真机进展但未闭环**。**5 个新 DSv3 缩层真机点 + 1 个历史锚点重放（cfg A）**（116/shb.ms.2.9，FSDP-2、full 重算、
4 卡对并发采集），这 6 行在**各自已测配置**上预测 vs 实测 |误差| ≤ **0.66%**（仅覆盖 `[2L,8L]×[2048,4096]×{full,select_attn}` 的 6/16 组合，余 10 未验证）。完整证据 [`realmachine/npu_closure_2026-07-15.md`](realmachine/npu_closure_2026-07-15.md)。

| cfg | 层 | seq | 实测 MiB | 预测 MiB | 预测/实测 |
|---|---|---|---|---|---|
| B | 2 | 4096 | 11733.0 | 11725.9 | 0.9994 |
| A | 4 | 4096 | 12473.1（历史锚点重放，非新点） | 12437.9 | 0.9972 |
| C | 6 | 4096 | 13213.2 | 13149.9 | 0.9952 |
| E | 8 | 4096 | 13953.3 | 13861.9 | 0.9934 |
| D | 4 | 2048 | 8810.5 | 8853.9 | 1.0049 |
| S | 4 | 4096 | 14716.4（select_attn） | 14812.6 | 1.0065 |

- **P2-05（全尺寸外推）OPEN**：2–8L 近似线性只量化外推**风险**——评估器每层系统低估 ~14 MiB，外推 61L 累计低估 ~800 MiB（~2.5%，OOM 不安全方向）；未做全尺寸/跨模型验证。线性结果是**风险估计**，非 CLOSED。
- **P2-04（K_OPT/K_CE/kept_frag 泛化）OPEN**：**K_OPT 不可辨识**——optstep 事件（桶 1767.5 MiB）比 loss-BWD 全局峰（bwd@5 12437.9 MiB）低约 5.5 GiB，改错 K_OPT 也不改 `max_memory_allocated`；K_CE 仅同模型 seq 差分、kept_frag 仅同模型 4L/8L，未跨模型/架构；`cp2-none=0.927`、`select-mlp=0.943` 历史欠预测仍未闭。需直接记录 optstep 事件峰 + 第二模型/MoE 架构 + CP-none/select-mlp 锚点。
- **P2-07（gate fp32）OPEN**：**未做 gate off/on 差分**；CE-fp32「类同背书」张量 shape/生命周期/峰值共存位置不同、不可替代；`router_dense_type` 仍被适配器忽略致 gate dtype 不可行使。**harness 存在**——用 `model_type: deepseek_v3` + `experimental_attention_variant: dsv4_hybrid` + `force_unfused_dsa: true`（**不是** `model_type: deepseek_v4`）；原「未注册 deepseek_v4」阻断系用错 model_type。

**最终（订正）**：§2 三项**仍开放**（P2-04/05/07）；**NPU 定量残留不为零**。合法保留的已验证结果：这 6 行在各自配置 ≤0.66%、889 测试通过、15 反例 oracle 全 PASS、P1-09 真机闭环、hybrid-CP fail-loud 闭环。
