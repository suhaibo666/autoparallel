# 代码检视轮次闭环报告 —— 离线内存成本评估器（2026-07-15）

> 本文是给检视人确认「本轮已收口」的执行摘要，自包含。证据来源：
> [`open_items_closure_2026-07-15.md`](open_items_closure_2026-07-15.md)（建模闭环 + §6 真机收口）
> 与 [`realmachine/npu_closure_2026-07-15.md`](realmachine/npu_closure_2026-07-15.md)（5 个新真机点）。
> 每条断言均可追溯至上述两文件；不引入任何新数据。

## 1. 一句话结论

> [!correction] 2026-07-16 复核（见 [`closure_report_verification_2026-07-16.md`](closure_report_verification_2026-07-16.md) §1、F1–F5）
> 原「一句话结论」把少量同模型缩层点的总峰吻合扩张为「全域确认 / 全部闭环 / NPU 定量残留归零」，属 over-claim。以下为订正后的诚实口径；被订正的原文以删除线保留。

**本轮有实质进展，但 NPU 定量残留不为零**：§1 的开放/部分建模项已建模落地并通过离线回归；此外在 116/shb.ms.2.9 上取得
**5 个新 DSv3 缩层真机点 + 1 个历史锚点重放**（cfg A=(4,4096,full) 为重放，非新点）。这 6 行数据在**各自已测配置**上
`peak_alloc` 预测 |误差| ≤ 0.66%——但它们只覆盖 `[2L,8L]×[2048,4096]×{full,select_attn}` 这个 16 组合笛卡尔积中的
**6 个**（`(2,4096,full)/(4,4096,full)/(6,4096,full)/(8,4096,full)/(4,2048,full)/(4,4096,select_attn)`），其余 **10 个组合未验证**。
故 ≤0.66% 只描述这 6 行、不描述评估器锚点全集；12 历史锚点中 `cp2-none=0.927`、`select-mlp=0.943` 仍为 OOM 不安全欠预测。

~~**本轮全部闭环**~~ / ~~**6 个新真机 DSv3 锚点**~~ / ~~**全域确认 |误差| ≤ 0.66%**~~ / ~~**NPU 定量残留归零**~~ 均已订正。
**P2-04（K_OPT 不可辨识）、P2-05（全尺寸外推未验证）、P2-07（未做 gate off/on）仍开放**。P2-07 的正确 harness **存在**
（用 `model_type: deepseek_v3` + `experimental_attention_variant: dsv4_hybrid` + `force_unfused_dsa: true`，**不是** `deepseek_v4`；
原「harness 不存在」的阻断理由系用错 model_type），待共享卡空闲后按正确配置跑 gate-on 差分即可。

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

### 3.2 真机收口项（§2/§3 三项「待 116」→ 部分进展，仍开放）

> [!correction] 2026-07-16 复核（F2/F3/F4）
> 本表原判 P2-04/05 为 CLOSED、P2-07 为「可达闭环」，均被复核否决。订正判定见下表「本轮真实进展 / 仍缺」两列。

| 项 | 原状态 | 本轮真实进展 | 仍缺 → 订正判定 |
|---|---|---|---|
| **P2-04** K_OPT/K_CE/kept_frag 标定泛化 | 待 116 定量 | K_CE 由同模型 seq 2048/4096 差分改善（D 绝对 0.49%）；kept_frag 补同模型 4L/8L 两点 | **K_OPT 不可辨识**：optstep 事件（桶 1767.5 MiB）比 loss-BWD 全局峰（bwd@5 12437.9 MiB）低约 5.5 GiB，改错 K_OPT 也不改 `max_memory_allocated`；未跨模型/架构；`cp2-none=0.927`、`select-mlp=0.943` 仍欠预测 → **OPEN** |
| **P2-05** 全尺寸外推 | 待 116 定量 | 2–8L 四点近似线性，量化了外推**风险** | 评估器每层低估 ~14 MiB，外推 61L 累计低估 ~800 MiB（~2.5%，OOM 不安全方向）；无全尺寸/跨模型验证 → 线性结果是**风险估计**，非 CLOSED → **OPEN** |
| **P2-07** gate FP32 峰值 | 待 116 定量 | 离线 gate 权重 roster 逐字节 | **未做 gate off/on 差分**；CE-fp32「类同背书」张量 shape/生命周期/峰值共存位置不同，不可替代；`router_dense_type` 仍被适配器忽略，gate dtype 无法被行使 → **OPEN**（harness 存在，用错 model_type 才被阻断） |

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

**这 6 行数据（= 5 个新 DSv3 点 + cfg A 历史锚点重放）在各自已测配置上全部落在 [0.9934, 1.0065]，即 `peak_alloc` 预测 |误差| ≤ 0.66%。**

> [!correction] 2026-07-16 复核（F1）
> 这 6 行只覆盖 `[2L,8L]×[2048,4096]×{full,select_attn}` 的 **6/16** 个组合，缺失 10 个（未测层数×seq×重算的其余交叉）；≤0.66% **仅适用于这 6 行**，不能推广为「全域确认」或评估器锚点全集的口径。cfg A=(4,4096,full) 是历史锚点重放，故**新点为 5 个**，全文按此统一。

reserved−alloc（≈HCCL+pool 碎片）full 实测
475 / 972.9 / 1056.8 / 1000.7 / 921.5 MiB、select 541.6 MiB，量级与 P2-01 的 ~0.5–1 GB 一致；但六点 reserved 误差变号且达 ±419 MiB（见 §6 P2-01 订正），量级一致 ≠ 上界成立。

### 4.2 每层线性（P2-05 —— 量化了外推风险，仍 OPEN）

> [!correction] 2026-07-16 复核（F3）
> 2–8L 的「零曲率」只证明这一小型 DSv3 配置在 2–8 层近似线性，**不能**检验全尺寸才会出现的流水/通信/allocator/offload/层异质性/模型族差异。本节结果是**外推风险的量化估计**，不是全尺寸验证，故 P2-05 **保持 OPEN**（原「CLOSED」订正）。

真机 peak 随层数在 2–8L 上**近似线性、零曲率**：N=2→11733.0 →(+740.1)→ N=4→12473.1 →(+740.1)→ N=6→13213.2 →(+740.1)→ N=8→13953.3，
即**实测斜率 = 740.1 / 2 = 370.05 MiB/层**。评估器同样线性，斜率 356.0 MiB/层（每 2 层 +712.0）。
- 评估器每层**系统性低估 ~14 MiB**；锚点在 4 层处，故 4L 绝对误差仅 0.28%。
- 外推至 61 层（DSv3 全尺寸）累计低估 ≈ 57×14 ≈ 800 MiB ≈ **2.5%，且为 OOM 不安全方向**——这是**有界、可量化的外推风险估计**（不等于已验证）。UI/文档应明示「外推且可能低估约 2.5%」并给 OOM 安全余量；未做全尺寸真机回归前不改写为 CLOSED。

### 4.3 标定常数泛化（P2-04 —— 局部改善但常数未泛化，仍 OPEN）

> [!correction] 2026-07-16 复核（F2）
> **K_OPT 在这些作业中不可辨识**：独立 probe 对 DSv3 4L/full 展开时间线显示 global peak = bwd@5 = **12437.9 MiB**，而 optstep total 仅 6874.5 MiB、K_OPT 桶仅 **1767.5 MiB**，比全局峰低约 **5.5 GiB**。真机只记录 `max_memory_allocated`，只要 optstep 低于 loss-BWD，改错 K_OPT 也不改总峰——用「总峰 0.66% 内」背书 K_OPT=4 属不可辨识参数上的错误归因。K_CE 只由**同一模型** seq 2048/4096 一组差分支持；kept_frag 只有**同一模型**同一 seq 的 4L/8L 两点，均未满足「跨模型稳定性」标准。且 12 锚点中 `cp2-none=0.927`、`select-mlp=0.943` 仍是 OOM 不安全欠预测。故 P2-04 **保持 OPEN**（原三处 CLOSED 订正）。至少需：直接记录 optstep 事件峰、补第二个模型/MoE 架构、补 CP-none/select-mlp 锚点。

- **K_CE（CE loss 峰 ∝ seq，局部插值改善）**：D(seq2048) vs A(seq4096) 真机落差 **3662.6 MiB**，预测落差 3584.0 MiB（归到 act_live+CE bwd_scratch 两个 ∝S 桶：1564+2020=3584）。评估器捕获 **97.9%** 的 seq 敏感度，D 绝对 0.49% 内——仅证同模型 seq 插值改善，**未证跨模型稳定**。
- **K_OPT=4（AdamW optstep 瞬态）——不可辨识**：optstep 桶预测恒 1767.5 MiB（层数无关），但该事件比全局峰低约 5.5 GiB，故总峰对 K_OPT 不敏感；此点**不能**作为 K_OPT 泛化的证据（见上方 callout）。
- **kept_frag=1.9（select-kept-MoE 碎片长尾）**：新增 select_attn **4L** 真机点（cfg S）实测 14716.4 vs 预测 14812.6 → **1.0065**，连原 **8L** 锚点（1.0024）——仅**同模型两层数点**改善，非跨模型泛化，故 kept_frag **未 CLOSED**。

### 4.4 P2-07 gate FP32 —— 未做 gate off/on 差分，仍 OPEN

> [!correction] 2026-07-16 复核（F4）
> 本节原判「可达范围内已闭」被否决，且「DSv4 harness 不存在」的阻断理由错误。订正：
> 1. **未做 gate off/on 一步差分**（原验收协议要求同一小 MoE 模型 gate off/on 差分）。CE-fp32「类同背书」**不能替代**——两者张量 shape、生命周期、峰值共存位置都不同。
> 2. **配置语义缺失**：适配器仍把 `router_dense_type` 放入忽略集合（`from_mindformers.py`），probe 输入 `float32` 与 `bfloat16` 得到完全相同的 `LLMConfig` 和 gate dtype，故 gate dtype 无法通过公共配置被行使（companion commit 拟修 qk/router 透传，见响应文档 F7）。
> 3. **harness 存在**（订正原「未注册」说法）：按 skill 应使用 `model_type: deepseek_v3` + `experimental_attention_variant: dsv4_hybrid` + `force_unfused_dsa: true`（**不是** `model_type: deepseek_v4`）；对应 `test_deepseekv4/` harness 与 `dsv4_align` 配置存在，历史 fused 作业已得 `peak_alloc=15415.5 MiB`。原闭环尝试选错 model_type 才被人为阻断。
> 故 P2-07 **保持 OPEN**，待共享卡空闲后按正确 model_type 跑同构 gate-off/on 差分。

离线 `sh_gate_w` fp32 权重已建模 + 逐字节 roster 测试（`test_param_conservation` / `test_x2_gate_fp32_cast`）——这是**离线字节守恒**，不是峰值定量验证。gate multiply 的 FP32 hidden cast 峰值贡献仍需 gate off/on 短步 NPU 对照（上方 callout）。

## 5. 诚实残留（不夸大）

> [!correction] 2026-07-16 复核（F1–F5）
> 原「NPU 定量残留归零」订正为 **不为零**。真实残留如下。

**NPU 定量残留不为零 —— P2-04/05/07 三项仍开放，另有历史欠预测锚点未闭：**

1. **P2-04 常数泛化（OPEN）**：K_OPT 在 loss-BWD 作业中不可辨识（optstep 事件低于全局峰约 5.5 GiB）；K_CE/kept_frag 仅同模型改善，未跨模型/架构。需直接记录 optstep 事件峰 + 第二个模型/MoE 架构 + CP-none/select-mlp 锚点。
2. **P2-05 全尺寸外推（OPEN）**：仅测 2–8L；外推 61L 累计低估 ~800 MiB（~2.5%，OOM 不安全方向），是**风险估计**而非验证。需全尺寸/近全尺寸真机回归。
3. **P2-07 gate FP32（OPEN）**：未做 gate off/on 差分；`router_dense_type` 被适配器忽略致 gate dtype 不可行使。harness **存在**（`model_type: deepseek_v3` + `dsv4_hybrid` + `force_unfused_dsa`，非 `deepseek_v4`），原阻断系用错 model_type。需按正确配置跑同构 gate-off/on 差分。
4. **历史欠预测锚点未闭**：`cp2-none=0.927`、`select-mlp=0.943` 仍 OOM 不安全（属 P2-04 待多点标定）。
5. **P2-01 pool 模型（PARTIAL）**：1.8% 碎片率是单点标定，非可证上界；六点 reserved 误差变号、range 达 ±419 MiB（C=−419）。不宣称 reserved OOM 安全。

> **候选精化项**（属上面 P2-05）：每层斜率评估器低估 ~14 MiB；改常数会破坏 12 锚点逐字节且有 DSv3-per-layer 过拟合风险，故保留常数但**须在 UI 明示外推低估风险**。

## 6. 最终状态（按 2026-07-16 复核订正）

> [!correction] 2026-07-16 复核（F1–F5、F8、F10–F12）
> 原 §6 把 P2-04/05/07 计入「闭环」、宣布「NPU 定量残留归零 / 全部 32 项闭环」，均已订正。P1-08、P1-13/14/15/16、P2-01 由「闭环」降为「开放/部分」。下面是订正后的分档。

**离线代码级闭环（建模/修复 + 回归 + 锚点守护，或 fail-loud refuse-not-lie）**：
P0-01/02/03/04/05，P1-01/02/03/04/05/07/09/10/11/12/17/18/19，P2-02(注)/03(注)/06/08，
＋ 衍生 Qwen3-qk-norm、grouped-FSDP、hybrid-CP(fail-loud)。
- **P1-09 FlashAttention**：`b3e7712` 116 算子级探针直接验证保存集 → **真机闭环**。
- **hybrid-CP**：对未建模混合区间 fail-loud（refuse-not-lie）→ **合法闭环**。
- 注：P2-02（opdag strict 假绿）、P2-03（YAML 适配器仍拒绝 qk-norm）实际为**部分**，companion commit 修复中（见响应文档 F6/F7）。

**部分闭环（实现存在，但公共配置不可达 / 生命周期不全 / 非上界）**：
- **P2-01** allocator pool：单点标定非上界，六点 reserved 误差 ±419 MiB。
- **P1-08** bwd scratch max-live：固定 window=2，无 lifetime 证据，「≤sum」不证上界 → **开放/高风险，潜在 OOM 欠估**。
- **P1-13** CP buffer / **P1-15** overlap：原仅经测试 getattr-hack 可达（companion commit 已补公共字段）；P1-15 仅 forward-send，grad-P2P 未建。
- **P1-14** experts wrap：仅 forward，backward 仍整层 gather。
- **P1-16** feasibility matrix：覆盖 7 条规则，但无 runtime 枚举的完整性证明 → 称「覆盖 7 条」，非「完整矩阵」。

**NPU 定量残留（仍开放）**：**P2-04（K_OPT 不可辨识、未跨模型）、P2-05（仅 2–8L、61L 外推低估 ~800 MiB）、P2-07（未做 gate off/on、router dtype 被忽略）**；另 `cp2-none=0.927`、`select-mlp=0.943` 历史欠预测未闭。

> **头条结论（订正）**：**5 个新 DSv3 真机点 + 1 个历史重放；已测 6/16 组合 ≤0.66%；其余 10 个组合未验证；NPU 定量残留不为零（P2-04/05/07 仍开放）。**
> 合法保留的已验证结果：889 测试通过、15 反例 oracle 全 PASS、这 6 行在各自配置上确 ≤0.66%、P1-09 真机闭环（`b3e7712`）、hybrid-CP fail-loud 闭环。12 锚点平均 2.0% 全程逐字节不变。

## Related

- [[open_items_closure_2026-07-15]] —— 建模闭环全表 + §6 真机收口
- [[realmachine/npu_closure_2026-07-15]] —— 5 个新真机点、per-layer 线性、K_OPT/K_CE、DSv4 阻断
