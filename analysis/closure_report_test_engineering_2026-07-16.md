# 测试工程轮检视意见 —— 对抗验证 + 修复闭环报告（2026-07-16）

> 输入：`analysis/test_engineering_defect_report_2026-07-16.md`（20 条 finding）。
> 方法：**逐条对抗验证**（对照源码/代码 + 实跑估计器取真值，不轻信报告结论），只修**真正的问题**；
> 计算类改动全部有探针实证 + 全量回归。**诚实优先**：标定 margin 明示为标定（非物理）；未修的诚实留档。

---

## 0. TL;DR

对报告 20 条逐条对抗验证：**5 条真且可干净修**（D1 / Z1 / Z2 / D2-观测 / Z3，已修并验证），
**1 条真但只余标定手段**（D1 的数值侧，退 2 点标定、诚实标注），**1 条真但不在本轮修**（D2 的
mHC+MTP 数值欠预测，只入卡 + 门控、**不宣称闭环**），其余 13 条为**正确留档/有意 fail-loud/真机栈
限制**，对抗验证后**维持不修**（附理由）。

**净效果**：记分卡从 2 个 OOM-不安全锚点（DSv3 无重算-MoE）降到 **1 个**（mHC+MTP，已门控、诚实
留档）；平均 |ratio−1| 2.6%→**2.2%**（含新入卡的 mHC+MTP）/ 13 卡内口径 1.8%；**全量 1181 passed**
（基线 1159 + 16 记分卡门 + 6 crosscheck/mf_root）。

> [!warning] 不过度宣称
> - **D1 是 2 点标定的 OOM-安全 margin，不是物理建模**。它把无重算-MoE 的 op 图粒度之下碎片长尾
>   （= 报告 G5）退成 `nr_moe_frag_factor` 常数，使两锚点预测≥真机；**底层碎片仍未被显式建模**。
> - **mHC+MTP（D2 数值侧）仍欠预测 0.920、OOM-不安全**。本轮只把它**入记分卡 + 加门**（防再次
>   悄悄漂移/翻转），**未修数值**，**不计入"已闭环"**。

---

## 1. 逐条对抗验证与处置

| 编号 | 报告级别 | 对抗验证结论 | 处置 |
|---|---|---|---|
| F1 | 已修 | 真（前轮 `37f8e3d` 已修 ln2） | 留档，无动作 |
| **D1** | P1 欠(OOM-不安全) | **真**：实跑确认 8L-none 0.931 / cp2-none 0.937；`_is_kept` 明确排除无重算域 | **修**：nr_moe margin（见 §2） |
| **D2** | P1 欠(OOM-不安全) | **真**：实跑确认 mHC+MTP 0.920，且确不在 `sim_vs_real_report.rows` | **半修**：入卡+门控（观测）；数值不修（见 §3） |
| **Z1** | P2 防护缺失 | **真**：代码级证实 MoE 窗口 `[首 moe_gemm…末]` 排除 ln2、MLA 段止于 h1→ln2 落 ffn 段，两族都不 census 它；独立复现"删 ln2 仍 ok=True" | **修**：layer_norms 校验族（见 §4） |
| **Z2** | P2 防护缺失 | **真**：`sim_vs_real_report.py` 是手动脚本、不入 pytest | **修**：参数化门（见 §3） |
| D3 | P2 过(安全向) | 真但**无需修**：pp2-stage0 1.089 是 OOM-**安全**过预测；报告自身"接受观察" | **不修**（门 hi=1.10 观察阈） |
| **Z3** | P3 | 真（小健壮性）：`default_mf_root` 单级、误指仓库根即 33 errors | **修**：两级探测（见 §4） |
| Z4 | P3 | 真但**代码内不可修**：三平台常数需第二个模型点独立测量，非改代码能解 | **不修**（留档，待第二真机点） |
| G2–G7 | P3 | 真且**已文档化的 known-missing physics**（含 G5=D1 根因）；非缺陷、有 caveat | **不修**（D1 只覆 G5 的 OOM 方向，非建模 G5） |
| N1–N9 | P3 | 真但**有意 fail-loud**（NotImplemented，不产生错误数字） | **不修**（设计即拒绝静默） |
| §6 盲区 | — | 真：真机栈限制（TP>1/VPP/DSA/swap 无可跑路径），非仿真器缺陷 | **不修**（栈解锁后补锚点） |

---

## 2. D1（修）——无重算-MoE OOM-安全标定 margin

### 根因对抗验证（systematic-debugging）
- 实跑：8L-none `dsv3(8,None)` = 18596.5 vs 真机 19967.3 → **0.931**；cp2-none = 18852.5 vs 20119.4
  → **0.937**。两点**同 per-device token 数**（dp2·B1 与 cp2·B2 均 = S/卡），缺口 1370.8 / 1266.9 MiB
  近等 → 与"按 token 数缩放的 fp32-cast 碎片"一致。
- 代码：`mem_timeline._is_kept` 仅对 select-kept 计 `kept_act`，注释明确"no-recompute 明确排除"→
  无重算域既无 kept_frag 也无 K_CE 之外补偿。**根因属实**，且是 op 图粒度之下（报告 G5，opdag_validation.md）。

### 隔离性探针（可行性实证，git 可逆）
临时给 pp==1 无重算-MoE-loss 加 margin，扫 factor∈{0,0.5,0.8,1.0,1.2,1.5}：
- **pp==1 gate 完美隔离**：pp2-stage0(1.089)/pp2-stage1(1.007)/select self_attn(1.001)/select
  mlp(0.955)/full(0.997) 在**任意 factor 下逐字节不动**——pp>1 无重算 loss stage 已由 `K_CE=8` 平衡、
  被 gate 排除。
- **单因子拟合两锚点**：factor=0.6 → 8L-none **1.009**、cp2-none **1.021**，均 OOM-安全（预测≥真机）。
  两点理想 factor 0.53/0.45 差 ~15% → **明示为标定常数、非精确物理**。

### 落地
`nr_moe_frag_factor`（默认 0=关；DSv3 preset / from_mindformers MoE 非 fused 注入 0.6）贯穿
`presets/from_mindformers → llm_config → model_spec.DimTable → report → mem_timeline`。margin 进
既有 `kept_frag` 桶（同族碎片、独立作用域、gate 互斥不双算）。**fused-CE(DSv4) loss_lids 空 → 不
触发** → 不覆盖 mHC+MTP（诚实：那是 D2，另档）。

### 复核
- 记分卡：2 个 OOM-不安全锚点 → **0 个 DSv3 无重算**（8L-none 0.931→1.009、cp2-none 0.937→1.021）；
  其余 12 锚点**逐字节不动**。
- 回归：全量 **1181 passed**，无 golden 需改（无既有测试钉 pp==1 无重算 DSv3 峰——恰是 Z2 缺口，
  现由记分卡门补钉）。

---

## 3. Z2 + D2-观测（修）——记分卡入 pytest + mHC+MTP 锚点

- **单一事实源** `scorecard_anchors.py`：脚本 `sim_vs_real_report.py` 与门
  `tests/test_scorecard_anchors.py` 共同消费，杜绝脚本/门 meta 层漂移（Z2 病的元形态）。
- **参数化门**：每锚点 ratio 带（**欠侧比过侧紧**——欠预测=OOM 危险方向）；带均足够紧使 D2 的
  1.088↔0.920（0.168 摆幅）触红（含一条元测试证明 band 非"哑门"）。另加 D1 专项门：DSv3 两无重算
  锚点必须 ratio≥1.0（margin 被删/关即触红）。**16 passed**。
- **D2 观测**：mHC+MTP 入卡（真机 21153.1，当前 **0.920**）。历史 MTP tie 修复后由 1.088 翻转欠预测、
  因未入卡而无人察觉——现入卡+门控杜绝复发。**数值侧不修**（D1 margin 不覆盖 fused-CE），留待单独
  诊断 MTP 段；记分卡汇总现**如实显示它为唯一 OOM-不安全锚点**。

---

## 4. Z1 + Z3（修）——crosscheck 层级 norm 校验族 + mf_root 两级探测

- **Z1 layer_norms 族**（源忠实）：从 `gpt_layer_specs.py`（`resolve_layer_spec` 抽 `input_layernorm`
  /`pre_mlp_layernorm` 绑定，二者均 `get_norm_cls`≠Identity）判定手写每个 decoder-body 层必须含
  ln1（pre-attn NORM）+ ln2（ffn 段首个重算子为 NORM）；缺则 finding、`ok=False`、strict raise。
  **独立复现**：正确 spec ok=True；删 ln2 → ok=False、layer_norm_findings=1、**delta-census 仍 0
  findings（证实旧两族确实盲）**、strict raise 提 pre_mlp。变异探针固化为验收用例 + 一条**无源亦跑**
  的结构不变量测试（防 F1 回潮，CI 无源也守）。
- **Z3**：`default_mf_root` 两级探测（误指仓库根→下降到含 `parallel_core` 的包目录；已是包目录→
  逐字节不变；缺源→原样返回不 fail）。tmp_path 单测覆盖三分支。
- 回归：crosscheck 相关 **41 passed**（旧 8 + 6 新），全量含其内。

---

## 5. 诚实的剩余状态

| 项 | 状态 | 说明 |
|---|---|---|
| DSv3 无重算-MoE（8L-none/cp2-none） | **OOM-安全（标定）** | 数值 1.009/1.021；底层碎片仍未显式建模（标定覆盖 OOM 方向） |
| mHC+MTP（D2 数值） | **仍欠 0.920、未修** | 入卡+门控（防漂移）；D1 不覆盖 fused-CE；留单独诊断 |
| pp2-stage0（D3） | 过 1.089、**安全** | 不修；门 hi=1.10 观察阈 |
| select-mlp / DSv4-base | 轻微欠 0.955/0.971 | ±5% 带内；门 lo 防进一步漂移 |
| Z4 平台常数 | 未独立测量 | 需第二真机点，代码内不可修 |
| G2–G7 / N1–N9 / §6 盲区 | 留档/有意/栈限 | 维持不修（附理由如上表） |

**结论**：报告点名的**唯一可干净修的 OOM-不安全correctness 缺口（DSv3 无重算-MoE）已由保守标定
margin 消除**；防护体系两个真实盲区（Z1 层级 norm、Z2 记分卡未入门）已封；mHC+MTP 的数值欠预测
**如实保留为未闭环项**并纳入门控。无过度宣称。

*产物：`cost_eval/mem_timeline.py`(+report/llm_config/model_spec/presets/from_mindformers) D1 margin；
`cost_eval/opdag/crosscheck.py` Z1/Z3；`scorecard_anchors.py`+`sim_vs_real_report.py`(重构)+
`tests/test_scorecard_anchors.py` Z2/D2；`tests/test_opdag_layer_norms.py`+`tests/test_mf_root_probe.py` Z1/Z3。*
