# `closure_report_2026-07-15.md` 独立复核报告

> 复核日期：2026-07-16  
> 复核对象：`analysis/closure_report_2026-07-15.md`、其对应代码与测试、116 真机证据  
> 代码基线：`b3e771211be22cb277122707c333579caad597b5`（`feat/unified-llm-modelspec`，与远端一致）  
> 结论口径：区分“代码已实现”“离线测试通过”“真机定量闭环”，不以总峰偶然吻合替代分桶可辨识性。

## 1. 结论

**不建议接受原报告的“全部 32 项 + 衍生项闭环、NPU 定量残留归零”结论。**

本轮修复有实质进展：当前全量测试 **889 passed**，旧版 15 个对抗 oracle 全部通过；新增的 5 个 DSv3 真机点（外加 1 个历史点重放）在各自已测配置上，`peak_alloc` 预测误差确实不超过 0.66%；后续提交 `b3e7712` 又以 116 上的算子级探针直接验证了 FlashAttention 保存集。这些证据应保留。

但原报告把“少量同模型缩层点的总峰吻合”扩张成了以下并未被证据支持的结论：

1. 把实际覆盖的 **6/16 个组合**写成 `[2L,8L]×[2048,4096]×{full,select_attn}` “全域确认”；
2. 把 **5 个新点 + 1 个历史重放**写成“6 个新真机点”；
3. 用始终不成为全局峰的 loss-BWD 作业“背书”`K_OPT`，该量在这些作业中不可辨识；
4. 用 DSv3 小模型的层数线性，替代原验收要求的全尺寸或跨模型验证；
5. 未做 gate off/on 差分，而且因使用了已知错误的 `model_type: deepseek_v4` 产生人为阻断，却将 P2-07 判为闭环；
6. 将仍存在配置不可达、适配器分裂、校验器假绿、输入校验缺失的功能判为完整闭环。

综合判定：**P2-04、P2-05、P2-07 仍开放；P2-01/P2-02、P1-06/07/08/12/13/14/15/16、P2-03 至少只能判部分闭环。** 另外发现 6 类新缺陷或集成缺口，见 §4。

## 2. 复核范围与可复现结果

### 2.1 基线和测试

| 检查 | 命令 | 结果 |
|---|---|---|
| 当前代码基线 | `git log -1 --oneline` | `b3e7712 verify(flash-attn)...`，本地分支与 origin 一致 |
| 全量回归 | `python -m pytest -q` | **889 passed, 6 warnings, 18.00s** |
| 原对抗 oracle | `python analysis/closure_audit_v2_oracle_2026-07-15.py` | 所列 oracle 全 PASS |
| 12 锚点回放 | `python sim_vs_real_report.py` | 平均绝对误差 2.0%，±5% 内 9/12；仍有 2 个 OOM 不安全欠预测 |
| 本轮独立反例 | `python analysis/closure_report_verification_probe_2026-07-16.py` | 进程退出码 0；所有反例被稳定复现 |

原报告的“855 测试绿”与其提交时点并不冲突；当前 889 是后续提交增加用例后的结果。问题不在测试是否全绿，而在不少测试只验证了实现内部的自洽性，没有验证公开输入链路或真机生命周期。

### 2.2 仍未消失的历史真机偏差

当前代码重放 12 个历史锚点仍得到：

| 锚点 | 预测/真机 | 判定 |
|---|---:|---|
| `cp2-none (loss,k_ce=4)` | **0.927** | 欠预测 7.3%，OOM 不安全 |
| `select mlp (keep-attn)` | **0.943** | 欠预测 5.7%，OOM 不安全 |
| `pp2-stage0 (optstep)` | 1.057 | 过预测 5.7% |
| `DSv4-fused` | 0.968 | 欠预测 3.2% |

因此“全配置 ≤0.66%”只能描述 §3.1 中那 6 行新/重放数据，不能描述评估器现有锚点全集，更不能据此宣布 P2-04 的标定常数已泛化。

### 2.3 116 真机复核边界

按照项目 `.claude/skills/real-machine-memory-sim/SKILL.md` 的共享机规则检查了 116：SSH 可用，目标容器和仓库可读；检查时 8 张卡均有持续运行的 host Python 进程，因此本轮没有抢占共享卡再启动 gate 对照作业。现有日志与远端 harness 已做只读核对。

远端事实与原报告的 P2-07 “harness 不存在”说法相反：

- 正确仓库是 `/home/suhaibo/workspace/deepseek_v4/mindformers`；
- skill 已明确说明不要使用 `model_type: deepseek_v4`，应使用 `model_type: deepseek_v3` + `experimental_attention_variant: dsv4_hybrid` + `force_unfused_dsa: true`（`SKILL.md:192-209`）；
- 对应 `test_deepseekv4/` harness 和 `dsv4_align` 配置存在，历史 fused 作业也已得到 `peak_alloc=15415.5 MiB`、`peak_reserved=16092/16096 MiB`（`analysis/realmachine/dsv4_fused/memory_summary_rank0.txt:10-11`）；
- 当前基准 YAML 的 `router_dense_type` 为 fp32，但没有 gate-on，因此它不能代替 gate off/on 差分。

所以 P2-07 的直接验证不是“不可达”，而是原闭环尝试选错了模型类型/checkout；在真正的 gate-on 作业完成前应保持开放。

## 3. 对真机闭环论证的逐项审计

### F1（高）：所谓“全域”实际只覆盖 6/16 个笛卡尔积组合

原报告在 `analysis/closure_report_2026-07-15.md:10-13,119-120` 宣称整个
`[2L,8L]×[2048,4096]×{full,select_attn}` 域都在 0.66% 内。按报告表格
`analysis/closure_report_2026-07-15.md:62-75` 还原，实际观察集合只有：

```text
(2,4096,full), (4,4096,full), (6,4096,full), (8,4096,full),
(4,2048,full), (4,4096,select_attn)
```

四个层数 × 两个序列长度 × 两种重算方式共有 16 个组合，缺失 10 个；其中 A `(4,4096,full)` 明确是历史锚点重放，故新增点为 5 个。报告正文 `:62` 自己也写“5 个新 DSv3 缩层点”，但总结 `:120` 又写“6 个新真机点”。

**判定：已测 6 行数据可信；“全域”和“6 个新增”是假阳性/文档计数错误。**

### F2（高）：P2-04 的三个常数都没有按原验收协议被识别

原验收标准要求“≥3 个不同规模/序列 DSv3/DSv4”并分别采 loss 峰和 optstep 峰
（`analysis/open_items_closure_2026-07-15.md:33,41-42`）。实际作业都是同一 DSv3 缩层族，且没有采到 optstep 全局峰。

独立 probe 对报告使用的 DSv3 4L/full 配置展开时间线：

```text
global peak  = bwd@5, 12437.906 MiB
optstep total = 6874.473 MiB
K_OPT bucket  = 1767.500 MiB
```

`K_OPT` 所在事件比全局峰低约 5.56 GiB。真机作业只记录 `max_memory_allocated`，故即使把 `K_OPT` 改错，只要 optstep 仍低于 loss-BWD，报告中的总峰也完全不变。用“所有配置总峰在 0.66% 内”背书 `K_OPT=4`，属于不可辨识参数上的错误归因。

`K_CE` 仅由同一模型的 seq 2048/4096 一组差分支持；`kept_frag=1.9` 只有同一模型、同一 seq 的 4L/8L 两点。它们说明局部插值改善，但没有满足“跨模型稳定性”标准。更重要的是，12 锚点中的 `cp2-none=0.927` 与 `select-mlp=0.943` 仍未闭合。

**判定：P2-04 保持开放。** 至少需要直接记录 optstep 事件峰，并补 DSv4/另一 MoE 架构与 CP/no-recompute、select-mlp 配置。

### F3（高）：P2-05 只是量化了外推风险，没有验证全尺寸

报告自己的数据表明：真机斜率 370.05 MiB/层，评估器斜率 356.0 MiB/层，即每层系统性低估约 14 MiB；外推到 61 层累计低估约 800 MiB（约 2.5%，且为 OOM 不安全方向），见
`analysis/closure_report_2026-07-15.md:78-83`。

四个点的“零曲率”只证明这一小型 DSv3 配置在 2–8 层上近似线性；它不能检验全尺寸才可能出现的流水、通信、allocator、offload、层异质性或模型族差异。原验收标准是“全尺寸真机回归”或至少明确的外推校验（`analysis/open_items_closure_2026-07-15.md:34,43`），本轮没有执行。

**判定：P2-05 保持开放。** 当前结果可以作为风险估计写入 UI/文档，不能改写为 CLOSED。

### F4（高）：P2-07 没有做 gate off/on，而且阻断理由错误

原协议明确要求同一小 MoE 模型 gate off/on 一步差分（`open_items_closure...md:41`）。实际报告只做了 gate 权重的离线 roster，然后以 CE 的 fp32 瞬态“类同背书”gate cast（`closure_report...md:91-96`）。两者张量形状、生命周期和峰值共存位置不同，不能互相替代。

此外适配器仍把 `router_dense_type` 放入忽略集合（`cost_eval/configs/from_mindformers.py:94`），生产 gate 权重和保存张量固定为 4 B（`cost_eval/layers/ffn.py:277-296`）。独立 probe 输入 `router_dense_type=float32` 与 `bfloat16`，得到完全相同的 `LLMConfig` 和 gate dtype；公共配置中也没有 router dtype 字段。

**判定：P2-07 开放，且存在真实配置语义缺失。** 正确 harness 已存在；待卡空闲时应按正确 model_type 跑同构 gate-off/on 差分。

### F5（中高）：P2-01 reserved/pool 模型不是上界，六点离散误差可达 ±419 MiB

生产代码自己标注 1.8% pool fragmentation 是“单点标定、跨模型稳定性待验证”
（`cost_eval/framework.py:76-79`），对外文档也承认它是近似而非可证上界
（`cost_eval/report.py:204`）。按当前公式 `400 MiB HCCL + 1.8% × allocated` 重算报告的六行数据：

| cfg | 预测 `(reserved-alloc)` 减真机，MiB |
|---|---:|
| B | +136.194 |
| A（历史） | -348.384 |
| C | **-418.962** |
| D | -362.911 |
| E | -349.541 |
| S | +123.295 |

误差既变号又随配置波动。“真机差值位于约 0.5–1 GiB”只说明数量级，并不验证预测模型，更不保证 reserved OOM 安全。

**判定：P2-01 仍为部分闭环。** allocated 与 reserved 必须分别给置信区间/安全余量，不能把 1.8% 当通用上界。

## 4. 代码与测试中仍存在的缺陷

### F6（高）：P2-02 opdag `strict=True` 存在两条假绿路径

`CrossCheckReport.ok` 只检查 `findings`（`cost_eval/opdag/crosscheck.py:262-271`）；提取异常被记录到 `uncovered` 后返回成功（`:295-297`）；Evaluator 默认 `validate_opdag=False`（`cost_eval/report.py:240`）。当前交叉检查主要比较粗粒度算子类别计数，没有比较张量 shape/dtype、`saves`、workspace 或生命周期。

独立反例得到：

1. 从 `linear_qb` 删除会影响内存的 `q_a_out` save 后，`strict=True` 仍 `ok=True`、0 findings；
2. 注入 opdag extractor 异常后，`strict=True` 仍返回 `available=True, ok=True, findings=0`，错误只出现在 `uncovered`。

**判定：P2-02 不是闭环；这是会让回归保护失效的真缺陷。** strict 模式应对已声明覆盖族的提取失败/未覆盖直接失败，并对内存契约做比较。

### F7（高）：P2-03 qk_layernorm 出现直连 API 与 YAML 适配器分裂

直连 `LLMConfig(qk_layernorm=True, attn_type="gqa")` 已能生成 `q_norm/k_norm`，对应放行代码在
`cost_eval/build_llm.py:197-208`。但 MindFormers 适配器仍在
`cost_eval/configs/from_mindformers.py:375-380` 抛出“暂未建 op 图”，并在 `:418` 明确不透传字段。

独立 probe 同时验证了这两条路径：直连 API 有两个 norm，等价 YAML 被拒绝。旧 oracle 的“F2 gqa qk_layernorm=True → REJECT”仍 PASS，反而证明 oracle 固化的是旧行为，不是报告宣称的新能力。

**判定：P2-03 部分闭环；主配置导入链路存在真实功能缺陷。**

### F8（高）：P1-13 与 P1-15 的新增能力无法通过公共配置启用

- `LLMConfig(cp_kv_allgather_buffer=True)` 报 `TypeError: unexpected keyword`；测试通过给 `DimTable` 动态挂属性启用（`tests/test_y1_gqa_colossal_kv_buffer.py:171,214`）。
- `ParallelConfig(pipeline_parallel_overlap_p2p=True)` 同样报 `TypeError`；测试在构造后动态赋属性（`tests/test_x4_p2p_pp.py:35-37`），而 YAML 适配器明确 fail-loud（`from_mindformers.py:572`）。

因此测试证明的是内部 `getattr` 分支能运行，不是用户能通过受支持配置到达该功能。

**判定：P1-13 部分闭环；P1-15 的基础 forward send 可用，但 overlap 子功能不可达。**

### F9（高）：MoE skew 接受 `<1`，会主动降低 OOM 估计

配置注释和公式均规定 `moe_skew_factor ≥ 1`（`cost_eval/llm_config.py:69`、`cost_eval/layers/ffn.py:46`），但构建入口没有校验。独立 probe 设置 `mode=skew, factor=0.5`，dispatch tokens 从 balanced 的 16 降到 8，未报错。

**判定：P1-12 存在新增输入校验缺陷。** 应拒绝非有限值、`factor < 1`，并验证 capacity/skew 对 OOM 口径的单调性。

### F10（中高）：P1-08 用“结果 ≤ 旧 sum”证明 OOM 安全，逻辑不成立

`_backward_max_live` 采用固定位置窗口 2（`cost_eval/structure_mem.py:130-160`），没有读取真实依赖或 profiler 生命周期。测试把 `max-live ≤ sum` 称为“OOM 安全”（`tests/test_p108_bwd_scratch_maxlive.py:97`），但从保守 sum 降低只能证明估计更紧，不能证明仍是实际峰的上界。

独立算术反例 `[4000, 0, 3000]` 返回 4000，而旧保守和为 7000；这 3000 B 的删减只有在两个 scratch 确实不重叠时才安全。当前没有运行时 lifetime 证据支撑这个假设。

**判定：P1-08 仍开放，属于潜在 OOM 欠估风险，不宜在取得真实 lifetime 证据前宣称闭环。** 这不是证明当前结果一定错，而是证明现有测试无法证明它正确。

### F11（中）：P1-14/P1-15 只实现了部分生命周期

- experts 子 wrap 仅在 forward 分裂出 `fwd:#experts`（`cost_eval/mem_timeline.py:619-649`）；backward 仍按整层 `sm.param_full_bytes` gather（`:670-690`），没有对应 experts 独立 backward 生命周期。
- P2P 桶文档明确“仅 FWD”，并明确 backward grad-P2P 未建（`:521-524,670-672`）。报告却把整项 P1-15 判为闭环。

**判定：P1-14、P1-15 均为部分闭环。** 单个 pp2 总峰不变化不能证明遗漏的通信缓冲在其它 PP/overlap 配置下无影响。

### F12（中）：部分“闭环”实际上是离线契约或证据缺口

- P1-06：报告自己承认 loss 实现级来源在 YAML 中不可得，只做了默认值 + warning；应标“可追溯的离线降级”，不是功能全闭环。
- P1-07：checkpoint island 由手写 op 邻接/selector 连续性推断，测试是合成图与内部代数，没有对应真实 checkpoint cell 边界证据。
- P1-16：实现了报告列出的 7 条约束，但“完整 runtime feasibility matrix”没有来自真实 runtime 的枚举清单；应标覆盖 7 条而非全矩阵闭环。
- P1-19、grouped-FSDP：代码级与离线回归较完整，本轮未发现新的阻断问题；但没有新增 NPU 定量证据。
- hybrid-CP：对未建模混合区间 fail-loud 是安全闭环，本轮认可。

## 5. 最新提交 `b3e7712` 的影响

该提交不改变上述否决结论，但它为 P1-09 增加了有效的真机证据，应单独认可：

- 116 上 `FlashAttentionScore` 输出的 `softmax_max/softmax_sum` 均为 `[B,N,S,8] fp32`；
- `softmax_out` 是空占位 `(1,)`，没有物化 `S×S`；
- attention output 与评估器保存的 O 对齐；
- 新增 4 个契约测试后全量为 889 passed。

证据见 `analysis/realmachine/flash_attn_activation_validation_2026-07-16.md`。因此 **P1-09 可判真机闭环**；但“无重算总峰偏高 2.53% 不源于 FA”的归因仍应理解为基于算子保存集的合理推断，而非对其它 act-live 项逐一做过消融。

## 6. 建议状态表

| 项目 | 本次建议状态 | 主要理由 |
|---|---|---|
| P1-06 loss provenance | 部分 | 无实现级来源，仅默认 + warning |
| P1-07 checkpoint islands | 部分 | 内部代数闭合，缺真实 checkpoint 边界证据 |
| P1-08 bwd scratch max-live | 开放/高风险 | 固定 window=2，无 lifetime 证据，安全性论证错误 |
| P1-09 FlashAttention saves | **闭环** | `b3e7712` 新增算子级真机证据 |
| P1-12 MoE skew/capacity | 部分 | `<1` 可降低估计且未拒绝 |
| P1-13 CP kernel buffer | 部分 | 内部分支存在，公共配置不可达 |
| P1-14 experts FSDP wrap | 部分 | 仅 forward 子生命周期，backward 仍整层 |
| P1-15 PP send/overlap | 部分 | forward send 有；overlap 不可达，grad-P2P 未建 |
| P1-16 feasibility matrix | 部分 | 7 条规则有测试，完整性无 runtime 枚举证据 |
| P1-17 UI round-trip | 代码级闭环 | 本轮测试通过，未发现新反例 |
| P1-19 分离 offload | 代码级闭环 | 离线路径完整；本轮无新 NPU 证据 |
| P2-01 allocator pool | 部分 | 单点近似非上界，六点 reserved 误差变号 |
| P2-02 opdag 主链 | 开放 | strict 假绿、默认关闭、未校验内存契约 |
| P2-03 qk_layernorm | 部分 | direct API 可用，YAML 适配器仍拒绝 |
| grouped-FSDP 子域 | 代码级闭环 | 离线实现/测试可信，未补 NPU |
| P2-04 常数泛化 | **开放** | K_OPT 不可辨识；未跨模型；旧欠预测锚点仍在 |
| P2-05 全尺寸外推 | **开放** | 只测 2–8L；61L 外推仍低估约 800 MiB |
| P2-07 gate FP32 | **开放** | 无 gate off/on；router dtype 被忽略；原阻断系用错 harness |
| hybrid-CP | **闭环（fail-loud）** | 未建模域明确拒绝，不静默估错 |

未在表中列出的既有 P0/P1 已闭项，本轮全量回归和对抗测试没有发现新的阻断性回归；本报告不因未重新获得每一项的独立真机证据而反向将其全部重开。

## 7. 最小闭环清单

1. 修复 qk norm YAML 透传、router dtype 语义、CP buffer/PP overlap 公共配置字段，并为每项增加“从真实 YAML 到峰值变化”的端到端测试。
2. `moe_skew_factor` 在构建边界强制有限且 `≥1`；增加估计单调性测试。
3. opdag strict 对已支持族的 extractor failure/uncovered 失败，并比较 param/save/workspace 的 shape、dtype 与字节，而不只比较类别计数。
4. P1-08 用真实 profiler/算子依赖获得 scratch lifetime；证据不足时恢复 sum 上界或提供 conservative/estimated 双模式。
5. 补 experts backward gather、grad-P2P 和真实 overlap 生命周期；不要用“当前 pp2 峰未变化”作为其它配置安全性的证明。
6. 真机空闲后用正确 DSv4 harness 跑同构 gate off/on；单独采 optstep 事件峰以识别 K_OPT；补 DSv4/另一 MoE、CP-none、select-mlp 锚点。
7. P2-05 在全尺寸/接近全尺寸上验证；在此之前 UI 明示“外推且可能低估约 2.5%”，并给 OOM 安全余量。
8. 将原报告总结改为：**5 个新 DSv3 点 + 1 个历史重放；已测 6/16 组合 ≤0.66%；其余域未验证；NPU 定量残留不为零。**

## 8. 独立证据入口

- 原闭环报告：`analysis/closure_report_2026-07-15.md`
- 原 NPU 数据：`analysis/realmachine/npu_closure_2026-07-15.md`
- 本轮可执行反例：`analysis/closure_report_verification_probe_2026-07-16.py`
- 12 锚点回放：`sim_vs_real_report.py`
- 最新 FA 真机证据：`analysis/realmachine/flash_attn_activation_validation_2026-07-16.md`

最终验收建议：**拒绝“全部闭环”总判定，保留已验证的局部修复和真机数据，按 §6 状态重新开项。**
