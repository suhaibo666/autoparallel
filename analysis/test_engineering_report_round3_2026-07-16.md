# 内存仿真器检视报告 · 第三轮（round2 交付复核 + 终态问题清单，2026-07-16）

> 输入：round2 交付 = `feature_gap_root_cause_analysis_2026-07-16.md`（对本人 round2 §4 功能缺失的
> 根因分析，含对本人报告 2 处分类订正）+ `closure_report_verification_response_2026-07-16.md`
> （对另一线独立复核 F1–F12 的回应，基线 b3e7712，代码由 e5ac9a6 承接）。
> 方法：对两份文档的**可验证声明逐条跑探针**（不轻信）；无代码改动 → 回归确认无漂移；
> 合并两线信息给出**终态**剩余问题清单。

---

## 0. TL;DR

round2 交付为**纯文档/分析**（无代码改动）。其可验证声明**复核全部通过**，其中对本人 round2 报告的
**2 处分类订正成立、本人接受**（N9 实为静默文档化近似而非 fail-loud；N7 是 raise+布尔近似混合体）。
回归确认无漂移（1181 passed；记分卡 14 锚点逐值不变）。

合并另一线（F1–F12）后，**终态问题清单新增 3 项此前未纳入本人报告的开放风险**（F3 层数外推、
F10 window-2 反向上界、F4 router-dtype 差分），正确性侧唯一 OOM-不安全项仍为 **D2 mHC+MTP 0.920**。
「纯遗漏 = 0」的判定经抽查后**认可**。

---

## 1. round2 声明复核结果

| 声明 | 复核方式 | 结果 |
|---|---|---|
| **N9 订正**：`layers_per_stage+interleave` 不 raise、静默连续近似 | 探针：`ParallelModel(pp=2, interleave=2, layers_per_stage=[5,5])` | ✅ 不 raise，chunks=[[0,1,2],[3,4]] 连续切——**本人 round2 把 N9 列 fail-loud 有误，接受订正** |
| 8/8 fail-loud 探针 | 抽查 N1(post-norm)/N6(SGD)/N8(pp+swap) | ✅ 4/4 真拒 |
| N5 是"外科式拒绝" | 双向探针：mla+qk_layernorm → raise；gqa+qk_layernorm → 放行**且真建 q_norm/k_norm op** | ✅ 按能力边界精确画线 |
| **G6**：调度器在外部包无源 | 读 mindformers `pipeline_parallel.py:281-284`：1f1b/interleaved_1f1b/gpipe 三 schedule 全注册指向 `hyper_parallel.core.pipeline_parallel`、惰性 import | ✅ 本地树确无实现，m==pp 特例无从源证，保守通式 warmup 正当 |
| G5 = op 图粒度物理极限 | 依据 `opdag_validation.md` §3-4（313 碎片、源码级 DAG 证不可导出）——此前轮已核 | ✅ 维持 |
| 无代码改动、无回归 | 全量重跑 + 记分卡逐值比对 | ✅ 1181 passed；14 锚点数值与 round2 完全一致（0.920 唯一 ⚠️、1.089 唯一 >1.05） |
| 另一线 F6–F9 已由 e5ac9a6 修入代码 | 本人首轮基线（1155 绿）已含 `test_review2_f7/f8/f9` 与 skew≥1、公共字段（specs.py/llm_config.py 读码确认过） | ✅ 早已在树内 |

**对根因分桶结论的判定**：接受。「35% 有意 fail-loud / ~38% 外部阻塞 / 5% 物理极限（G5）/
20% 已量化推迟 / **0% 纯遗漏**」与本人各轮探针与读码一致；fail-loud 是设计优点而非缺陷的论证成立
（8/8 实证拒绝 + N5 的精确画线是最强证据）。

---

## 2. 终态问题清单（两线合并，按优先级）

### 正确性（产生错数的）

| 级别 | 项 | 状态 | 量化 |
|---|---|---|---|
| **P1** | **D2** mHC+MTP 欠预测 | **开放**（已门控 band 0.90–0.96，数值未修） | 0.920，缺口 ~1686 MiB；藏身处 = N7 的 fused-CE 布尔近似（loss_lids 空 → 一切 loss 峰 margin 不触发）。诊断路径：116 分跑 MHC/MTP 拆分量 |
| **P2** | **F3**（另一线）缩层→全尺寸外推风险 | **开放** | 每层欠 ~14 MiB → 61L 全尺寸 ~800 MiB（~2.5%）欠方向；无全尺寸验证点；UI 应明示外推低估风险并留安全余量 |
| **P2** | **F10**（另一线）bwd_scratch window-2 | **开放/标注高风险** | 逆序滑窗 window=2 无 lifetime 证据；>2 个大 scratch 共存时可欠估（反例 [4000,0,3000]→4000）。现实模型全退化为单 scratch → 当前锚点不受影响；需 profiler lifetime 或 conservative/estimated 双模式 |
| **P2** | **D1-R** margin 迁移面 > 标定面 | **开放** | 0.6 系数 DSv3 两点标定、被注入任意 MoE YAML；直连 LLMConfig 默认 0 静默回 0.93x。需一个非 DSv3 MoE 无重算锚点 |
| P2 | D3 pp2-stage0 过预测 | 观察（安全向） | 1.089，门 hi=1.10 |
| P3 | select-mlp 0.955 / DSv4-base 0.971 | 带内、门控防漂移 | — |

### 标定/验证债（不改代码不能闭的）

| 项 | 状态 |
|---|---|
| **F2**（另一线）常数可辨识性 | 部分缓解：K_OPT 已有 pp2-stage0 专属锚点（峰即 optstep）、cp2-none 已由 D1 margin 闭合（1.021）；仍缺独立 optstep 事件峰记录与第二模型点交叉 |
| **F4**（另一线）router-dtype gate off/on 真机差分 | 开放；阻断理由已订正（harness 存在：`deepseek_v3`+`dsv4_hybrid`+`force_unfused_dsa`），待共享卡空闲 |
| **Z4** 平台常数（HCCL 200MiB/域、512B、pool 1.8%） | 开放；需第二真机点（F5 同判：reserved 不宣称 OOM 安全，allocated/reserved 分别留余量） |
| 真机盲区（TP>1 / pp>2 / VPP / DSA / swap 曲线） | 开放；真机栈限制，锚点可得性问题非建模缺陷 |

### 功能缺失（分类订正后的终态口径）

- **fail-loud 拒绝（正确设计，7 项）**：N1–N3、N5、N6、N8 + N7 的 raise 侧——8/8 探针实证真拒；
  **N4（bias）偏严但可辩护**（内存中性字段，可降级 warning，低优先）。
- **静默文档化近似（订正后归此，2 项）**：**N9**（layers_per_stage+VPP 连续切近似——本人前报告
  误列 fail-loud，已订正；建议补一条 warning，低优先）；**N7 布尔侧**（fused-CE 非显式 op 图
  ——唯一带 OOM 不安全后果的近似 = D2 藏身处，随 D2 修）。
- **已量化推迟（4 项）**：G2（off-peak）/G3（opt-in，迁移未验）/G4（已量化落在余量内）/G7（离线信息极限+三口径）。
- **物理极限（1 项）**：G5 碎片长尾——只能标定 margin 覆 OOM 方向，已如此做（D1）。
- **外部无源（1 项）**：G6 —— 本轮读源确证。
- **纯遗漏：0**（认可）。

---

## 3. 最终判定

- **机理与覆盖**：1181 用例（250 解析级 + 22 门 + 41 交叉校验）双环境绿；14 真机锚点平均 2.2%、
  13/14 OOM-安全或带内；防护网（层级 norm 族、记分卡门）经变异/活性探针证实能拦已知缺陷类回潮。
- **剩余正确性风险收敛到 4 个开放项**：D2（唯一 OOM-不安全锚点）、F3（全尺寸外推）、F10（window-2
  反向上界）、D1-R（margin 迁移）——全部有门控/文档留档，无隐蔽项。
- **功能缺失面定性成立**：fail-loud 优先的设计取舍 + 外部阻塞 + 一个物理极限，纯遗漏为零；
  本人前两轮报告的 2 处分类措辞已随根因分析订正。

**backlog 顺序**：① D2 分量诊断（116 MHC/MTP 拆分）→ ② F3 全尺寸/近全尺寸验证点（与 Z4 第二
标定点同一次真机窗口可并做）→ ③ D1-R 非 DSv3 MoE 锚点 → ④ F10 scratch lifetime（或双模式）→
⑤ F4 router-dtype 差分 → ⑥（低）N9 warning、N4 降级、Z1 次级 norm 名册。

*复核证据：N9 探针（不 raise、连续切分）；fail-loud 抽查 N1/N5×2/N6/N8；G6 源码读证
（pipeline_parallel.py:281-284 外部注册+惰性 import）；1181 passed；记分卡 14 锚点逐值不变。*
