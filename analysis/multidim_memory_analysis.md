# DeepSeek-V3 (MLA + MoE) 多维并行策略峰值显存分析

离线解析侧（纯 Python，无 MindSpore / 无真机）对 DSv3 缩层训练做 **5D 并行**逐维 + 组合扫描，
预测**单卡峰值 allocated 显存**及 8 桶明细，并锚定真机实测点。

- 评估器入口：`cost_eval.report.Evaluator`；模型构建器：`validate_dsv3.build_dsv3_spec(N)`（复用）。
- 驱动脚本：[`analyze_matrix.py`](../analyze_matrix.py)（模型按 N 构建一次、跨配置复用，仅 ParallelConfig/Evaluator 变化）。
- 固定项：`seq=4096, B=1`，compute bf16 / params fp32（`OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4)`，
  持久 12 B/param），SP=on，`RecomputeSpec(mode="full")` 全 transformer 层重算，
  `HardwareSpec(max_device_memory=59 GiB, framework_reserve=2197 MiB)`，`dp_replicate=cp=1`。
- 列含义：`pred_peak` = 含 framework_reserve 的预测峰值；`struct_peak` = `pred_peak − framework`（剔 2197 MiB 标定残余）；
  其余为峰值时刻 8 桶快照（MiB）；`hccl` = `num_distinct_communicators(pc)`（不同 HCCL 子通信器数，**仅 reserved 池、不进 allocated 峰值**）。
- 全部数据为 `analyze_matrix.py` 实跑输出（单位 MiB）。`swap_buf` 全程为 0（SwapSpec 未启用），故从表中略去。

> **峰值发生位置（关键）**：除 PP 的 stage 0 外，所有配置的峰值事件都落在 **lm_head 的反向（`bwd@<head>`）**——
> 此处 `bwd_scratch`（NLL 的 fp32 probs = `4·S·B·vocab` ≈ 2020 MiB）+ lm_head 的 saved logits(bf16)+logsm(fp32)
> 主导 `act_live`。lm_head **不被 tp/ep 切**、只有一层（不被 pp 拆），是大多数维度下"切不动"的硬底。
> 这解释了下面多数桶为何不随某些并行维变化。

---

## Sweep A — FSDP (`dp_shard`)，N=4，tp=ep=pp=1

| config | pred_peak | struct_peak | persistent | act_live | gather_buf | grad_buf | recomp_scratch | bwd_scratch | workspace | hccl |
|---|---|---|---|---|---|---|---|---|---|---|
| dp_shard=1 | 16302.4 | 14105.4 | 7659.8 | 3100.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 1 |
| dp_shard=2 | 12472.5 | 10275.5 | 3829.9 | 3100.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 2 |
| dp_shard=4 | 10557.6 | 8360.6 | 1914.9 | 3100.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 2 |
| dp_shard=8 |  9600.1 |  7403.1 |  957.5 | 3100.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 2 |

**Trend**：`persistent` 严格 ∝ 1/dp_shard（7659.8 → 3829.9 → 1914.9 → 957.5，每翻倍精确减半），是唯一驱动峰值下降的桶。
`act_live`/`gather_buf`/`grad_buf`/`bwd_scratch` 全部恒定——FSDP 只切持久态（param+opt state），不切激活；
而 `gather_buf`(=441.9, lm_head 全量权重 bf16) 与 `grad_buf`(=883.8, 同权重 fp32 梯度=2×) 建模为**单层 full-unsharded** 缓冲、本就不随 FSDP 缩。符合 ZeRO/FSDP 数学。

## Sweep B — EP on `dp_shard=8`，N=4，tp=pp=1

| config | pred_peak | struct_peak | persistent | act_live | gather_buf | grad_buf | recomp_scratch | bwd_scratch | workspace | hccl |
|---|---|---|---|---|---|---|---|---|---|---|
| ep=1 | 9600.1 | 7403.1 | 957.5 | 3100.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 2 |
| ep=2 | 9600.1 | 7403.1 | 957.5 | 3100.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 3 |
| ep=4 | 9600.1 | 7403.1 | 957.5 | 3100.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 3 |
| ep=8 | 9600.1 | 7403.1 | 957.5 | 3100.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 3 |

**Trend**：**所有桶对 ep 完全不变**，仅 `hccl` 子通信器数 2→3。两条独立原因叠加：
(1) 专家**持久权重对 ep 不变**——`StaticMem` 用 `efsdp = dp_shard·cp·tp // ep` 再切专家，
而 resolve 已先按 ep 切 `n_experts`，二者抵消 ⇒ 专家分片总数恒为 region(`dp_shard·tp`)，与 ep 无关（efsdp 设计）。
(2) 专家**激活**确实 ∝ 1/ep，但在 full 重算下只出现在 transformer 层反向的 `recomp_scratch`，
而峰值落在 **lm_head 反向**（`recomp_scratch=0`），故 ep 切的那部分根本不在峰值时刻 ⇒ 峰值对 ep 免疫。
这与真机 ep=1 vs ep=2 峰值近乎相等（12473→12474）**一致**，是已验证行为，非 bug。
（注：任务预期"EP 让专家 persistent/act ∝1/ep 缩"在本模型下**不成立**——见 Findings。）

## Sweep C — TP，N=4，dp_shard=2，ep=pp=1

| config | pred_peak | struct_peak | persistent | act_live | gather_buf | grad_buf | recomp_scratch | bwd_scratch | workspace | hccl |
|---|---|---|---|---|---|---|---|---|---|---|
| tp=1 | 12472.5 | 10275.5 | 3829.9 | 3100.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 2 |
| tp=2 | 11848.2 |  9651.2 | 3240.6 | 3065.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 3 |
| tp=4 | 11536.0 |  9339.0 | 2945.9 | 3047.5 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 3 |

**Trend**：TP 让 `persistent`（3829.9→3240.6→2945.9）与 `act_live`（3100→3065→3047.5）小幅下降，**收益递减、非 ∝1/tp**。
原因：只有 tp 切的注意力/dense/shared-expert 权重缩，而 embedding、lm_head、专家(ep 切)权重**不随 tp 缩**，构成 tp-不变底座；
`act_live` 里只有 sp 切的 checkpoint 输入(∝1/tp)缩，主导的 lm_head logits/logsm(未切)不动。
`gather_buf`/`grad_buf`/`bwd_scratch` 全不变——峰值在 **lm_head**，其权重不被 tp 切，故峰值处的 gather/grad 缓冲与 tp 无关
（若峰值落在 transformer 层，gather/grad 才会 ∝1/tp）。

## Sweep D — PP，N=8，dp_shard=2，tp=ep=1（报告 per-stage[0]）

| config | pred_peak | struct_peak | persistent | act_live | gather_buf | grad_buf | recomp_scratch | bwd_scratch | workspace | hccl |
|---|---|---|---|---|---|---|---|---|---|---|
| pp=1 | 13896.1 | 11699.1 | 5197.5 | 3156.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 2 |
| pp=2 |  6082.9 |  3885.9 | 2504.2 |   56.0 | 441.9 | 883.8 | 0.0 |    0.0 | 0.0 | 3 |
| pp=4 |  5043.2 |  2846.2 | 1478.5 |   42.0 | 441.9 | 883.8 | 0.0 |    0.0 | 0.0 | 3 |

stage 拓扑：pp=1 → stage0 = 全 10 layer（含 lm_head），warmup mb=0，peak@bwd(lm_head)；
pp=2 → stage0 = layer[0..4]（embedding+1 dense+3 moe，**无 lm_head**），warmup mb=1，peak@bwd(layer0)；
pp=4 → stage0 = layer[0,1]（embedding+dense），warmup mb=3，peak@bwd(layer0)。

**Trend**：per-stage[0] 峰值大降，但**主要不是因为 PP 本身省内存，而是 lm_head 这块巨头(2020 MiB bwd_scratch + ~2.1 GB logits/logsm 激活)被切到了最后一个 stage、离开了 stage 0**：
stage 0 的 `bwd_scratch=0`、`act_live` 暴跌到 ~56 MiB（只剩在飞 microbatch 的 checkpoint 输入）。
`persistent` 随 stage 层数减少而降（5197.5→2504.2→1478.5）。在飞 microbatch（warmup 1、3）在 full 重算下每个只 pin 极小 checkpoint 输入，故只给 `act_live` 加几十 MiB，未抵消省下的量。
**关键 caveat**：stage 0 是**最轻** stage，**不是**真实设备峰值——见下表（tightest stage）。

> **PP tightest-stage（真实单卡峰值）**：评估器 `tightest_stage` 给出的设备峰值远高于 stage 0：
>
> | config | per-stage[0] | tightest stage | tightest peak | 对 pp=1 降幅 |
> |---|---|---|---|---|
> | pp=2 | 6082.9 (stage0) | stage1（含 lm_head） | **11335.9** | −18% |
> | pp=4 | 5043.2 (stage0) | stage3（含 lm_head） | **10980.0** | −21% |
>
> lm_head 单层、`bwd_scratch`(2020) + logits/logsm 激活（~3.1 GB）**不可被 PP 拆分**，恒钉在末 stage，
> 成为 PP 下的真实瓶颈。故 PP 在本缩层 + 巨 vocab 配置下对**设备峰值**只有 ~20% 改善，远小于 per-stage[0] 表面看到的 ~56% 降幅。

## Sweep E — Combinations

| config | pred_peak | struct_peak | persistent | act_live | gather_buf | grad_buf | recomp_scratch | bwd_scratch | workspace | hccl |
|---|---|---|---|---|---|---|---|---|---|---|
| dp_shard=2,ep=2 | 12472.5 | 10275.5 | 3829.9 | 3100.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 3 |
| dp_shard=4,tp=2 | 10227.9 |  8030.9 | 1620.3 | 3065.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 3 |
| dp_shard=2,tp=2,ep=2 | 11848.2 | 9651.2 | 3240.6 | 3065.0 | 441.9 | 883.8 | 0.0 | 2020.0 | 0.0 | 4 |
| dp_shard=2,pp=2 (N=8) | 6082.9 | 3885.9 | 2504.2 | 56.0 | 441.9 | 883.8 | 0.0 | 0.0 | 0.0 | 3 |
| dp_shard=2,tp=2,ep=2,pp=2 (N=8) | 5465.6 | 3268.6 | 1914.9 | 28.0 | 441.9 | 883.8 | 0.0 | 0.0 | 0.0 | 5 |

**Trend**：各维叠加效果**可加且单调下降**，无相互抵消异常。`dp_shard=2,ep=2` ≡ `dp_shard=2`（ep 不动峰值，Sweep B）；
加 tp=2 在其上再省 ~620 MiB（`12472.5→11848.2`）；`dp_shard=4,tp=2` 把 persistent 压到 1620.3（FSDP×TP 双切权重）。
含 PP 的两行报的是 stage 0（最轻），同样需用 tightest-stage 看真实峰值（见 Sweep D caveat）。
`hccl` 随启用的并行域线性增（最多 5），但**不进 allocated 峰值**。

---

## Validation against real machine

真机 rank0 `max_memory_allocated`（MiB）为 ground truth。预测/实测比值：

| anchor config | pred_peak | measured | ratio (pred/meas) |
|---|---|---|---|
| N=4, dp_shard=2 (Sweep A) | 12472.5 | 12473.1 | **1.0000** |
| N=4, dp_shard=2, ep=2 (Sweep E) | 12472.5 | 12474.1 | **0.9999** |
| N=8, dp_shard=2 | 13896.1 | 13953.3 | **0.9959** |

三锚点全部落在 **±0.5%** 内。ep=2 锚点尤为关键：真机 allocated 峰值几乎不随 ep 变（HCCL 涨在 reserved 池），
与评估器"ep 不动 allocated 峰值"完全吻合，佐证了 `framework_reserve(allocated)` **不含** per-comm-domain 的 +200 MB。

---

## Findings / caveats

**已锚定（validated）**：
- **结构分片模型仅在 `dp_shard=2`（ep1/ep2）与 N=8 三点对真机标定**，误差 ≤0.5%。
- HCCL/通信缓冲在 **reserved** 池、**不进 allocated 峰值**：`framework_reserve(allocated)=2197 MiB` 常数，
  **不**按通信域数 +200 MB（ep=2 真机点证实，`cost_eval/framework.py` §8.6）。`hccl` 列仅记录子通信器数供 reserved 侧参考。

**仅预测、未经真机验证（evaluator-only）**：
- **Sweep A 的 dp_shard∈{1,4,8}、Sweep B 全部、Sweep C 全部 TP、Sweep D/E 全部 PP** 均为评估器外推，
  真机验证被机器 HCCL flakiness 阻断，未取点。这些行的趋势在数学上自洽，但绝对值待真机标定。
- `framework_reserve=2197` 假定对所有配置近恒定；其随 seq（flash workspace）、更大 world 的缩放**未验证**（§8.7 开放项）。

**需要关注的建模性结论（非 bug，但与朴素预期相左）**：
- **EP 不改变 allocated 峰值**（Sweep B 完全平坦）。任务预期"EP 让专家 persistent/act ∝1/ep 缩"在本模型下**不成立**：
  (a) efsdp 设计使专家持久态对 ep 恒定（专家始终切满 `dp_shard·tp` region）；
  (b) 专家激活虽 ∝1/ep，但在 full 重算下只活在 transformer 反向 `recomp_scratch`，而峰值在 lm_head 处，二者错峰。
  此结论**已被真机 ep=1≈ep=2 佐证**，是正确行为。若未来关掉 full 重算、或峰值移到 MoE 层，EP 才会显现于峰值。
- **PP 的 per-stage[0] 严重低估设备峰值**：stage 0 不含 lm_head，是最轻 stage；真实单卡峰值由含 lm_head 的末 stage 决定
  （pp=2: 11335.9, pp=4: 10980.0），PP 对设备峰值仅 ~20% 改善。**读 PP 结果务必看 tightest stage，而非 stage 0。**
- **`recomp_scratch`/`workspace` 在所有峰值快照中均为 0**：因峰值统一落在 lm_head 反向（无重算、workspace 已释放）。
  这不代表它们恒为 0——只是从不在峰值时刻；transformer 层反向时 `recomp_scratch` 非 0，但低于 lm_head 峰。

**物理合理性检查**：所有 A–E 配置峰值随分片**单调不增**，**未出现"加分片峰值反升"**的异常 ⇒ 无疑似评估器 bug。

---

## Related artifacts
- 驱动：[`analyze_matrix.py`](../analyze_matrix.py)（复用 `validate_dsv3.build_dsv3_spec`）
- 模型 / 锚点：[`validate_dsv3.py`](../validate_dsv3.py)
- 评估器核心：`cost_eval/{report,mem_timeline,static_mem,framework,parallel_model,shape_eval}.py`
