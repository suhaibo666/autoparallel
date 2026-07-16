# 对独立复核（`closure_report_verification_2026-07-16.md`）的逐条回应

> 回应日期：2026-07-16
> 回应对象：`analysis/closure_report_verification_2026-07-16.md`（F1–F12 + §5/§6）
> 被订正文档：`analysis/closure_report_2026-07-15.md`、`analysis/open_items_closure_2026-07-15.md`
> 代码基线：`b3e771211be22cb277122707c333579caad597b5`

## 0. 总体立场

**接受复核的核心判定。** 撤回原报告的「全部 32 项闭环 / NPU 定量残留归零 / 全域 ≤0.66%」总结，改为诚实口径：
**5 个新 DSv3 真机点 + 1 个历史重放；已测 6/16 组合在各自配置 ≤0.66%；其余 10 组合未验证；NPU 定量残留不为零（P2-04/05/07 仍开放）。**

同时保留复核也认可的、确有证据的局部成果，不因总结被否而反向低估：889 测试通过、15 反例 oracle 全 PASS、这 6 行数据在各自配置上确 ≤0.66%、P1-09 由 `b3e7712` 116 算子探针取得真机闭环、hybrid-CP fail-loud 为合法闭环。既有 P0/P1 已闭项本轮无新阻断回归，不重开。

处置类别三档：**doc-corrected**（本轮已在两份 .md 就地订正）、**fixed-in-code**（由 companion 代码提交承接，本响应不含代码改动，验证待其落地回归）、**acknowledged-open**（需真机工作，暂不能闭）。

## 1. 逐条回应 F1–F12

| # | 复核发现（摘要） | 我方结论 | 处置类别 | 说明 |
|---|---|---|---|---|
| **F1** | 「全域」实为 6/16 组合；5 新点 + 1 重放（A 是重放），非「6 新点」 | **接受** | doc-corrected | closure_report §1/§4.1、open_items §6 已改为「6/16 组合、余 10 未验证；A=(4,4096,full) 为重放，新点 5」。≤0.66% 仅限这 6 行 |
| **F2** | P2-04 三常数未按协议识别；**K_OPT 不可辨识**（optstep 桶 1767.5 MiB 比 bwd@5 全局峰 12437.9 MiB 低约 5.5 GiB） | **接受** | doc-corrected + acknowledged-open | 已改判 **OPEN**。K_CE 仅同模型 seq 差分、kept_frag 仅同模型 4L/8L；`cp2-none=0.927`/`select-mlp=0.943` 未闭。需直接记录 optstep 事件峰 + 第二模型/MoE + CP-none/select-mlp 锚点 |
| **F3** | P2-05 只量化外推**风险**（每层低估 ~14 MiB → 61L ~800 MiB ~2.5%，OOM 不安全方向），无全尺寸验证 | **接受** | doc-corrected + acknowledged-open | 已改判 **OPEN**，线性结果标注为风险估计；UI 须明示「外推且可能低估 ~2.5%」并给安全余量 |
| **F4** | P2-07 未做 gate off/on；CE-fp32 类同背书不可替代；且「harness 不存在」阻断理由错误 | **接受** | doc-corrected + acknowledged-open | 已改判 **OPEN**，并订正 harness **存在**：`model_type: deepseek_v3` + `experimental_attention_variant: dsv4_hybrid` + `force_unfused_dsa: true`（**非** `deepseek_v4`）。原阻断系用错 model_type |
| **F5** | P2-01 的 1.8% pool 是单点标定非上界；六点 reserved 误差变号、range ±419 MiB（C=−419） | **接受** | doc-corrected | 已改判 **PARTIAL**，不宣称 reserved OOM 安全；allocated/reserved 应分别给安全余量 |
| **F6** | P2-02 opdag `strict=True` 两条假绿（extractor 异常入 `uncovered` 仍成功；默认 `validate_opdag=False`；只比类别计数） | **接受** | fixed-in-code（companion） | 承接方案：strict 对已覆盖族的 extractor failure/uncovered 直接失败，并比较 param/save/workspace 的 shape/dtype/字节。**本响应不改代码，验证待 companion 落地** |
| **F7** | P2-03 qk_layernorm 直连 API 可用但 YAML 适配器仍拒绝（`from_mindformers.py` 抛错且不透传） | **接受** | fixed-in-code（companion） | 承接方案：适配器透传 `qk_layernorm` 并建 op 图 + 端到端「YAML→峰值变化」测试。旧 oracle 固化的是旧 REJECT 行为，需同步更新 |
| **F8** | P1-13 CP buffer / P1-15 overlap 原仅经测试 `getattr`-hack 可达，公共 `LLMConfig`/`ParallelConfig` 无字段 | **接受** | fixed-in-code（companion，**已补公共字段**）+ doc-corrected | companion commit 已把 `cp_kv_allgather_buffer` / `pipeline_parallel_overlap_p2p` 加为真实公共字段；两份文档已就地降为 **PARTIAL** |
| **F9** | MoE skew 接受 `factor<1`，会主动降低 OOM 估计（无构建期校验） | **接受** | fixed-in-code（companion） | 承接方案：构建边界强制有限且 `≥1`，并加 capacity/skew 对 OOM 口径的单调性测试 |
| **F10** | P1-08 固定 window=2 无 lifetime 证据；「≤sum」不证上界；反例 `[4000,0,3000]→4000` 仅两 scratch 不共存时安全 | **接受** | doc-corrected + acknowledged-open | 已改判 **OPEN/高风险（潜在 OOM 欠估）**。真正闭需 profiler/依赖给 scratch lifetime；证据不足时应回退 sum 上界或提供 conservative/estimated 双模式 |
| **F11** | P1-14 experts 仅 forward 子生命周期（backward 仍整层 gather）；P1-15 仅 forward-send，grad-P2P 未建 | **接受** | doc-corrected | 两份文档已改为 **PARTIAL**；不以「单个 pp2 总峰不变」证其它 PP/overlap 配置安全 |
| **F12** | 部分「闭环」实为离线契约/证据缺口：P1-06（默认+warning）、P1-07（无真实 checkpoint 边界）、P1-16（覆盖 7 条非完整矩阵）、P1-19/grouped-FSDP（无新 NPU 证据）、hybrid-CP（fail-loud 合法） | **接受**（含一处一致） | doc-corrected | P1-16 已改为「覆盖 7 条」；P1-06「可追溯离线降级」、P1-07「离线契约」标注。**hybrid-CP fail-loud 与 P1-09 真机闭环维持有效**——此处与复核一致，非分歧 |

## 2. 关于最新提交 `b3e7712`（复核 §5）

接受复核对 `b3e7712` 的单独认可：116 上 `FlashAttentionScore` 的 `softmax_max/softmax_sum` 均 `[B,N,S,8] fp32`、`softmax_out` 为 `(1,)` 空占位（未物化 `S×S`），attention output 与评估器保存 O 对齐；新增 4 个契约测试后全量 889 passed。故 **P1-09 判真机闭环**。附带说明：「无重算总峰偏高 ~2.53% 不源于 FA」仍应理解为基于算子保存集的合理推断，非对其它 act-live 项逐一消融。

## 3. 无分歧 / 不重开

- 既有已闭 P0/P1 项：本轮全量回归与对抗测试未发现新阻断回归，不反向重开。
- hybrid-CP：对未建模混合区间 fail-loud（refuse-not-lie）是合法闭环，复核认可，维持。

## 4. 剩余真机 / 代码工作清单（镜像复核 §7）

**代码（companion commit 承接，本响应不含代码改动）**：
1. qk-norm YAML 透传（F7）、router dtype 语义（F4）、CP buffer/PP overlap 公共字段（F8，已补）——每项加「真实 YAML→峰值变化」端到端测试。
2. `moe_skew_factor` 构建期强制有限且 `≥1` + 单调性测试（F9）。
3. opdag strict 对 extractor failure/uncovered 失败，并比较 shape/dtype/字节而非类别计数（F6）。
4. P1-08 用真实 profiler/依赖取 scratch lifetime，或提供 conservative/estimated 双模式（F10）。
5. 补 experts backward gather、grad-P2P、真实 overlap 生命周期（F11）。

**真机（acknowledged-open，待共享卡空闲）**：
6. 用正确 DSv4 harness（`deepseek_v3` + `dsv4_hybrid` + `force_unfused_dsa`）跑同构 gate off/on 差分；单独采 optstep 事件峰识别 K_OPT；补 DSv4/另一 MoE、CP-none、select-mlp 锚点（F2/F4）。
7. P2-05 在全尺寸/近全尺寸验证；在此之前 UI 明示外推低估风险（F3）。

## Related

- [[closure_report_verification_2026-07-16]] —— 被回应的独立复核
- [[closure_report_2026-07-15]] —— 已就地订正的闭环报告
- [[open_items_closure_2026-07-15]] —— 已就地订正的开放/部分项报告
