# 对《逐条答复闭环审计》的回应与二轮闭环（2026-07-15）

> 被审对象：`analysis/review_closure_audit_2026-07-14.md`（二次审计，判据：仅 fail-loud/warning/文档化 = 部分闭环，不等同功能实现）。
> 基线：`feat/unified-llm-modelspec`，评估器 **493 passed**（较上轮 475 +18 闭环审计回归）。
> 立场：**审计公正，无假阳性**。上一轮答复文档确有「状态过度归并」——把「已答复/已封口」写成了「已闭环」。本轮逐条接受审计裁定，并把其揭示的核心 API 未闭环反例**逐条修复 + 转正式回归**（`tests/test_closure_audit_2026_07_14.py` 18 项 + `test_param_conservation.py` 逐模块精确对账）。

## 1. 先认账：审计说得对的地方

审计把我上轮的过度声明分成三类，全部属实：
- **代码直接反证回复**：P1-06（仍 `cross_entropy_fused=_dsa_fused`）、P1-14（仍并入整层）、P2-06（无 chunk 字段）、P2-07（非逐模块精确、仍 ±2%）、P2-08（docstring 仍写 strict XFAIL/7×）。
- **只封口未实现能力**：P0-02/P1-02/P1-16/P1-17/P2-01 只在 adapter/UI 封口，核心 API 仍接受。
- **只修一半**：P1-01/P1-09/P1-10/P1-11/P1-19/P2-03。

这些我本轮该修的修、该收窄措辞的收窄。**仍属真·未实现建模的项（P1-07/08/12/13/14/15、P2-02/04/05），本轮不再声称已闭环**，如实列为开放（§4）。

## 2. 本轮二次闭环（把「封口/半修」补成功能实现）

| 审计项 | 审计原判 | 本轮动作 | 证据（回归） |
|---|---|---|---|
| **P0-02** adapter 静默丢语义 | 部分闭环 | training/context/swap 段**全键 schema**，未知键+typo（`nevver`/`local_batch_szie`/`max_device_memry`/`swap.enablee`）fail-loud；非零 dropout fail-loud | `test_training/context/swap_typo_rejected`、`test_nonzero_dropout_rejected` |
| **P0-03** reshard 枚举未校验 | 闭环（有校验缺口） | reshard 枚举进 `ParallelConfig.__post_init__`（`nevver` 不再回落 default） | `test_reshard_enum_typo_rejected_at_config` |
| **P1-02** full 空集/零命中（核心 API） | 部分闭环 | `Evaluator.evaluate` 针对已解析图校验：full 空集 + select 零命中 fail-loud（不只 UI） | `test_full_empty_layers_rejected_at_evaluate`、`test_select_zero_hit_rejected_at_evaluate` |
| **P1-06** CE 与 DSA fusion 耦合 | 部分闭环（代码反证） | `cross_entropy_fused` **解绑 `_dsa_fused`**——改由显式键 + `attn_type` 决定（DSv4 恒 lean、独立于 DSA kernel）；DSv4 锚点 15415.5 不动 | `test_ce_fused_decoupled_from_dsa_fusion`、`test_ce_fused_explicit_key_overrides` |
| **P1-09** FA workspace 不按 TP 切 | 部分闭环 | workspace 字符串 → `workspace_ref` TensorRef（head 维 ÷tp）；TP=8 从高估 8× 变正确（1024→128 KiB） | `test_flash_workspace_is_softmax_lse_formula`（改断言 TP 切） |
| **P1-01** shared_gate 缺 + 守恒非精确 | 部分闭环 | 补 `shared_experts_gate [H,1]`（条件建，DSv3 不用→golden 不变）；参数守恒**改逐模块逐参数精确对账** | `test_dsv3_per_module_exact_roster_and_count`（roster + 逐字节相等） |
| **P1-10** o_groups=0 裸除零 | 部分闭环 | dsv4_hybrid o_groups=0 提前 fail-loud（不再 shape resolve 裸 `ZeroDivisionError`） | `test_dsv4_zero_o_groups_rejected_before_divzero` |
| **P1-11** topk≤0/capacity≤0 | 部分闭环 | 进 `_validate_structure`：负/零 numel 前拦截 | `test_negative_topk_rejected`、`test_zero_capacity_rejected` |
| **P1-16** feasibility（核心 API） | 部分闭环 | `feasibility_errors()` 集中于 `Evaluator`（tp>1+SP、非 Adam、PP+swap），不只 adapter；`check_feasibility=False` 显式绕过通道 | `test_tp_gt1_requires_sequence_parallel_at_evaluator`、`test_pp_plus_swap_rejected` |
| **P1-19** 非 Adam optstep | 部分闭环 | 非 Adam 进 `Evaluator` fail-loud（不再静默生成 AdamW optstep） | `test_non_adam_optimizer_rejected`（param/grad/opt 分离 offload 仍开放，§4） |
| **P1-17** UI 非 round-trip | 部分闭环 | README「评估仍按解析值」订正为「用回填 UI 字段 + 固定假设，**不是**完整解析值；如需完整走 CLI」 | README_explorer.md:35 |
| **P2-01** `.oom` 单口径 | 部分闭环 | 拆 `allocated_oom` / `reserved_oom` 两口径；构造出 `.oom=False` 而 `reserved_oom=True` | `test_reserved_oom_split_from_allocated` |
| **P2-06** timeline 无 chunk | 部分闭环（表述不实） | `TimelineSample` 加 `chunk` 字段 + 事件名 `#c<chunk>`；VPP 下 `(event,mb,chunk)` 唯一 | `test_vpp_timeline_event_mb_chunk_unique` |
| **P2-07** 测试固化（±2%） | 部分闭环 | 新增逐模块精确对账（±2% 仅保留为跨版本观察）；反例转 18 项正式回归 | 见上 |
| **P2-08** 文档漂移 | 部分闭环（有反例） | `test_review_evidence`/`test_ce_optstep` docstring 订正（去「strict XFAIL」「7×」）；CP/K_CE 注释上轮已对齐 | docstring diff |

## 3. 一处需澄清的措辞收窄

- **P2-03 qk_layernorm**：审计指出 qk=true 变 false。核实：qk_layernorm 的内存**真·可忽略**（2 个 `[S,B,head_dim]` 小 RMSNorm 激活）——刻意不建 op 是照抄 `dsv4_align_config()` 的选择，非静默丢关键内存；`build_llm` 对显式 `LLMConfig.qk_layernorm=True` 仍原生 fail-loud。本轮把它在忽略集里的注释改为「已知可忽略、刻意不建」，不再让它看起来像静默 drop。这一项按**已知可忽略**收口，非功能缺口。

## 4. 诚实的仍开放清单（本轮**不**声称闭环）

以下是真·未实现的建模工作，如审计所判，保留开放；本轮只保证不再过度声明：

| 项 | 状态 | 为何暂缓 |
|---|---|---|
| P1-07 checkpoint islands | 开放 | 选择集仍非一等 checkpoint region；跨岛活性需重构 estimate_select_memory |
| P1-08 bwd_scratch max-live | 开放（接受的保守上界） | 仍 `sum(op.bwd_scratch)`；方向 OOM-安全，精化需按 op 时序取 max-live |
| P1-12 MoE skew/capacity | 开放 | 仍 balanced/floor；skew/capacity-ceil/最忙 rank 需真机不均衡路由差分标定 |
| P1-13 CP kernel buffer | 开放 | fused-QKV 的 KV 分量、ring/ulysses 在飞双缓冲仍 caveat |
| P1-14 FSDP wrap 粒度 | 开放 | experts/router 仍并入整层 `param_full_bytes`（layer-group 粒度）；子模块独立 gather 时间线未建 |
| P1-15 PP send/overlap/更多调度 | 开放 | P2P send buffer、overlap 双缓冲、GPipe/zero-bubble、显式 VPP chunk 配额未建 |
| P1-19（残留） | 部分 | 非 Adam 已 fail-loud；param/grad/optimizer **分离** offload 仍不可表达 |
| P2-02 opdag 主链 | 开放 | 交叉验证工具定位（当期设计），Evaluator 不消费 |
| P2-04 标定常数泛化 | 开放 | K_CE/K_OPT/kept_frag 少量 regime 标定；cp2-none 0.927、select-mlp 0.943 仍 <0.95 |
| P2-05 全尺寸预设 | 开放 | 循环 ratios/外推，仅 UI「预估计」标签 |

## 5. 新状态口径

审计的「8 闭环 / 15 部分 / 9 开放」经本轮修复后（不含仍开放的真建模项）：

- **闭环（功能实现 + 守卫 + 回归）**：P0-01/02/03/04/05、P1-01/02/03/04/05/06/09/10/11/16/18、P2-01/06/07/08 —— **20 项**。
- **部分（安全封口 / 已知可忽略，非完整建模，如实标注）**：P1-17、P1-19（offload 分离）、P2-03 —— **3 项**。
- **开放（真建模工作，本轮不声称闭环）**：P1-07/08/12/13/14/15、P2-02/04/05 —— **9 项**。

12 真机锚点平均 |ratio−1| = **2.0%** 全程不动（所有 C1–C5 改动均对锚点配置零字节或仅 TP>1/reserved 口径生效）。真机 P0-01（1889.5 MiB）与 P0-04（vocab local 115834880）证据仍成立，无需复采。
