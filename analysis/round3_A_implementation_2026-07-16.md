# round3 检视意见 · A 档（纯代码可改）实施记录（2026-07-16）

> 输入：`test_engineering_report_round3_2026-07-16.md`。用户指令：把 A 档（离线纯代码即可改、
> 不需真机/上游源）全部做掉。B/C/D 档（需真机标定 / 需真机数据或上游源 / 有意不改）**不在本轮**。
> 原则：全部 OOM-安全方向；默认行为逐字节不变（新旋钮 opt-in）；每项带测试；诚实标注标定/近似。

---

## 完成项（6/6，全绿）

| 项 | 改动 | 性质 | 验证 |
|---|---|---|---|
| **F10** | `structure_mem._backward_max_live` 加 `conservative` 模式（全 scratch 共存 Σ 上界）；`HardwareSpec.bwd_scratch_conservative` opt-in（默认 estimated=window-2） | OOM 安全化 | est=4000 vs cons=7000 on `[4000,0,3000]`；单 scratch 相等；DSv3 真实模型 cons==est（默认锚点逐字节不破） |
| **F3** | `report.evaluate` 在 `n_layers>16`（超缩层锚点已验证尺度）发 `OOMSafetyWarning`：全尺寸外推、每层残差累积（欠方向）、无全尺寸验证点 | 警示 | 20L 触发、8L 不误报 |
| **D1-R** | 直连 `LLMConfig`（MoE+pp1+无重算+非 fused-CE 但 margin=0）发 `OOMSafetyWarning`；`from_mindformers` 注入处补迁移风险留档（0.6 仅 DSv3 两点标定、跨结构未验证外推） | 消静默欠估 | factor=0 触发；preset(0.6)/pp2/full/fused-CE(DSv4) 均静默（不误报） |
| **N4** | `build_llm` `add_bias_linear`/`add_qkv_bias` 从 fail-loud **降级** `ModelingApproxWarning`（bias 内存中性，按无 bias 继续，op 图不变） | 放宽 | 警告并建成、op 图与 base 逐 op 一致；2 处旧"raise"测试改为"warns" |
| **N9** | `parallel_model.stage_chunks` 的 `layers_per_stage`+interleave 连续切近似补 `ModelingApproxWarning` | 加提示 | 该组合下触发 |
| **Z1-次级** | `crosscheck` `layer_norms` 族**扩到源忠实的次级 norm**：MLA latent q/kv norm（条件：源 qk_layernorm 真）、final_norm（无条件，`TransformerBlockSubmodules.layer_norm`）、MTP enorm/hnorm（无条件，`get_mtp_layer_spec`）——各删 1 个即 `ok=False`+strict raise，正确 spec 不误报 | 防护增强 | 独立复现：del q_a_norm→mla_dense/q_layernorm；del final_norm→lm_head/final_norm；均 strict raise。correct DSv3 ok=True |

新增共享模块 `cost_eval/advisories.py`（`ModelingApproxWarning` / `OOMSafetyWarning` 两可测类别 +
`warn_oom_safety`/`warn_modeling_approx`）；`README_config.md §7.5` 集中文档所有 advisory + 保守旋钮
（`bwd_scratch_conservative` / `nr_moe_frag_factor` / `framework_reserve`）。

## 诚实边界

- **F10 conservative / F3 / D1-R 的 nr_moe_frag margin** 都是 OOM-**安全侧**工具（预测≥真机方向），
  不是精确物理：F10 conservative 是严格上界、F3 只警示不改数、D1-R margin 是 DSv3 两点标定。
- **Z1-次级全部源忠实**（无硬编码"总是强制"）；仅校验存在/位置，不校验 saves/shape/dtype；MTP 层
  路由出 MLA/MoE delta 族（其复合结构非那两窗口所建模，跑 delta 会假阳）——诚实边界，仅影响含 MTP 的
  spec（现有套件无，故无行为变化、只除潜在假阳）。GQA q/k norm 有意不强制（无 GQA decoder-body 层、
  qk_layernorm 默认关，强制会假阳）。
- **B/C/D 档未动**：D2 数值（需真机拆 mHC/MTP）、D1-R 迁移验证（需非 DSv3 MoE 锚点）、F2/F4/Z4/
  真机盲区（需真机/第二点）、有意不改项（D3 安全向、N1-3/5/6/8 fail-loud、G5 物理极限、G6 无源）。

## 回归

全量 **1221 passed**（含并发分支已提交的 opdag walker 保真 `3ffd4b1`；本轮 A 新增测试文件
test_bwd_scratch_conservative / test_oom_safety_advisories / test_opdag_layer_norms_secondary +
既有 N4 测试改判）。默认行为逐字节不变——记分卡 14 锚点、golden、full/select/pp2 全部不动。

*本轮只提交本人 A 档 + Z1-次级文件；并发 opdag/timesim 开发（construct_walker/gpt_segments 等）
已由其作者独立提交（3ffd4b1 等），不并入本提交。*
