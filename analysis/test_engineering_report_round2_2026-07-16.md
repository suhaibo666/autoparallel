# 内存仿真器缺陷与功能缺失报告 · 第二轮（闭环复核后，2026-07-16）

> 输入：`closure_report_test_engineering_2026-07-16.md`（对首轮报告 20 条的修复闭环）。
> 方法：**对闭环声明逐条独立复核**（重跑套件/记分卡 + 变异探针 + 门活性探针），不轻信其结论；
> 然后给出复核后的剩余缺陷与功能缺失清单。首轮报告：`test_engineering_defect_report_2026-07-16.md`。

---

## 0. TL;DR

闭环报告的 **5 项修复声明全部独立复核通过、无过度宣称**。修复后，正确性缺陷收敛到
**1 个 OOM-不安全锚点（mHC+MTP 0.920，D2，已门控未修数值）**；防护体系两个盲区（Z1/Z2）
经变异/活性探针确认**真实闭合**。新增 1 条二轮观察（D1 margin 的跨模型迁移风险）。
功能缺失面（未建模残差 / fail-loud 未实现 / 真机栈盲区）与首轮一致，未变化。

---

## 1. 闭环声明复核结果（全部通过）

| 声明 | 复核方式 | 结果 |
|---|---|---|
| 全量 1181 passed | 本地重跑 | ✅ 1181 passed |
| D1：8L-none 0.931→1.009、cp2-none 0.937→1.021，其余 12 锚点逐字节不动 | 重跑记分卡逐行比对 | ✅ 全部吻合（full 域/select/pp2 与修前逐位一致） |
| Z1：crosscheck 层级 norm 校验族 | **变异探针**：删 MoE 层 ln2 → `ok=False, layer_norm_findings=1`；删 ln1 → `ok=False`；完好 spec `ok=True` 不误报 | ✅ 首轮"删 ln2 仍 ok=True"的盲区已闭合 |
| Z2：记分卡入 pytest（单一事实源 `scorecard_anchors.py`） | 跑 22 条门测试 + **门活性探针**：强制 `nr_moe_frag_factor=0` → 8L-none ratio 0.931 < 带下限 0.99 → 必触红 | ✅ 门是活的，margin 被删/关会红 |
| margin 与 select kept_frag 互斥（不双算） | select-attn 锚点在 margin 开启下重算 | ✅ 1.001（不受影响） |
| Z3：mf_root 两级探测 | `test_mf_root_probe.py` 通过（含三分支单测） | ✅ |
| D2 只入卡+门控、不宣称闭环 | 记分卡如实显示 0.920 为唯一 ⚠️；band (0.90, 0.96) 两向防漂移（若回翻 1.088 也触红） | ✅ 诚实 |

D1 margin 实现要点（代码级审查确认）：`nr_moe_frag_factor` 默认 0（回归安全）；标定基
`_nr_moe_act` = 无重算-MoE 非 loss 层驻留激活（判据与 FWD `saved` 口径一致，含 swap 互斥）；
生效四重门 `factor>0 ∧ pp==1 ∧ 本 stage 无任何重算 ∧ loss-BWD`；进 `kept_frag` 桶（与 select
margin 作用域互斥）；fused-CE 时 `loss_lids` 空 → 不触发（故不覆盖 DSv4/mHC+MTP，与声明一致）。

---

## 2. 剩余缺陷（复核后）

### D2（P1，唯一 OOM-不安全项）：mHC+MTP 欠预测 0.920

- sim 19466.7 vs real 21153.1（DSv4 4L + mHC×4 + MTP，fused-CE），缺口 ~1686 MiB。
- 历史：MTP tie 修复前 1.088 过预测 → 修后翻转欠预测；本轮已入记分卡 + band (0.90,0.96) 门控，
  **数值未修**。D1 margin 因 fused-CE 门不覆盖它——正确的设计取舍（乱覆盖会双算/错域）。
- 构成推测：DSv4-base 自身 0.971（−444 MiB）+ MTP 复合层增量欠计（MTP 层含 MoE FFN 段，
  同受碎片长尾影响但走 fused-CE 无 margin；mHC 增量真机仅 +688 MiB、误差贡献小）。
- **建议诊断路径**：116 上对 DSv4 4L 分别跑 `MHC=1 SIM_MTP=0` 与 `MHC=0 SIM_MTP=1`
  （`prep_dsv4align.py` 支持），把 1686 MiB 缺口拆到 mHC/MTP/base 三分量，再决定是显式建模
  （MTP 段 saves 补齐）还是扩 margin 域（fused-CE 下的碎片系数单独标定）。

### D1-R（P2，新增二轮观察）：nr_moe margin 的跨模型迁移风险

margin 本身已诚实标注"2 点标定、非物理"（两点理想 factor 0.53/0.45 差 ~15%），复核补充两点：

1. **迁移面比标定面宽**：`from_mindformers` 对**任意** MoE 非 fused-dsv4 的 YAML 模型注入 0.6
   ——但标定只在 DSv3（MLA+MoE、topk4、S4096）两点上做过。Mixtral/GQA+MoE 类模型的碎片长尾
   比例未必同（碎片 ∝ dispatch/permute 量 ∝ topk/capacity，跨结构未验证）。
2. **直连 API 默认 0**：绕过 preset/YAML 直构 `LLMConfig` 的 MoE 无重算配置**静默回到 0.93x
   欠预测**。这是有意的回归安全默认，但与"OOM 安全"的产品口径有张力。

建议：真机栈解锁后加一个非 DSv3 的 MoE 无重算锚点（如 GQA+MoE）验 0.6 的迁移性；文档面在
README_config 标注直连 API 需自行开 margin。

### D3（P2，维持观察）：pp2-stage0 过预测 1.089

安全方向；门 hi=1.10 只剩 ~1% 余量——这是**有意的紧观察阈**（再保守化会触红逼人收紧 K_OPT），
非缺陷。维持不修。

### 轻微欠预测（带内，门控防漂移）

select-mlp 0.955（lo=0.94）、DSv4-base 0.971（lo=0.95）——±5% 带内，已由门钉住不许再漂。

---

## 3. 防护体系剩余缺口

| 项 | 状态 |
|---|---|
| Z1 层级 norm 校验族 | **已闭合**（本轮复核）；覆盖 ln1/ln2。注意其仍不校验 q/k norm、final_norm、MTP enorm/hnorm 等次级 norm——量级小（saves ~[S,B,head_dim 切片]/[H]），暂可接受 |
| Z2 记分卡门 | **已闭合**；band 需随未来标定更新维护（如修好 D2 后 hi=0.96 会触红逼更新——设计如此） |
| Z3 mf_root | **已闭合**（两级探测 + 三分支单测） |
| Z4 平台常数单点标定 | **未变**：HCCL 200 MiB/域、512B 块、pool 碎片 1.8%（DSv4 单点）——需第二真机点，代码内不可修 |
| z2 无源 skip | 部分缓解（新增一条无源亦跑的结构不变量测试守 ln1/ln2）；op 名册族仍需源码才校验 |

---

## 4. 功能缺失（与首轮一致，未变化）

**未建模残差（有 caveat 文档）**：G2 ring/ulysses CP 在飞双缓冲；G3 GQA colossal KV buffer
（opt-in 默认关、未真机验证）；G4 backward grad-P2P；G5 无重算-MoE 碎片**底层仍未显式建模**
（D1 margin 只覆其 OOM 方向，物理缺口仍在）；G6 VPP m==pp 特例（hyper_parallel 无源）；
G7 MoE 真实倾斜分布（三口径近似）。

**fail-loud 未实现（拒绝评估、不产错数）**：N1 post/sandwich norm；N2 LayerNorm；
N3 learned_absolute/none 位置编码；N4 linear/qkv bias；N5 mla/dsv4/dsa 的 qk_layernorm；
N6 非 Adam 优化器；N7 loss 变体（fused-CE 为布尔近似非显式 op 图）；N8 PP+swap（真机同不支持）；
N9 显式 per-chunk VPP ranges（文档化近似）。

**验证盲区（真机栈限制）**：TP>1 激活、pp>2/VPP、PP×重算、DSA 全域、swap 真机曲线——
均无真机锚点，当前由解析用例（本轮 memval 250 条）+ Megatron/mindformers 逐行 port 锚定。

---

## 5. 最终判定与后续

**判定**：内存仿真功能在被测范围内**正确且 OOM-安全**（14 锚点中 13 个预测≥真机或在 ±5% 带内，
唯一欠预测锚点 D2 已门控留档）；覆盖面（多维混合并行 × 重计算 × 9 类模型结构）由
1181 条测试（250 条解析级 + 22 条记分卡/交叉校验门）+ 14 真机锚点支撑；防护体系经变异/活性
探针验证能拦截已知缺陷类的回潮。

**后续优先级**：
1. D2 数值诊断（116 上 mHC/MTP 分量拆解，见 §2 建议路径）——唯一 OOM-不安全项；
2. D1-R 迁移性验证（非 DSv3 的 MoE 无重算锚点一个）；
3. 真机栈解锁后补 TP>1 / VPP / DSA 锚点 + Z4 第二标定点；
4. （低）Z1 扩展到次级 norm 名册。

*复核证据：1181 passed；记分卡 14 锚点重算逐行比对；变异探针（ln2/ln1 删除）×2；
门活性探针（margin 强制 0 → 0.931 触红）；select 互斥探针（1.001 不动）。*
