# DeepSeek-V3 (MLA + MoE) 多维并行策略峰值显存分析

离线解析侧（纯 Python，无 MindSpore / 无真机）对 DSv3 缩层训练做 **5D 并行**逐维 + 组合扫描，
预测**单卡峰值 allocated 显存**及 8 桶明细，并锚定真机实测点。

- 评估器入口：`cost_eval.report.Evaluator`；模型构建器：`validate_dsv3.build_dsv3_spec(N)`（复用）。
- 驱动脚本：[`analyze_matrix.py`](../analyze_matrix.py)（模型按 N 构建一次、跨配置复用，仅 ParallelConfig/Evaluator 变化）。
- 固定项：`seq=4096, B=1`，compute bf16 / params fp32（`OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4)`，
  持久 12 B/param），SP=on，`RecomputeSpec(mode="full")` 全 transformer 层重算，
  `HardwareSpec(max_device_memory=59 GiB, framework_reserve=177 MiB)`，`dp_replicate=cp=1`。
- 列含义：`pred_peak` = 含 framework_reserve 的预测峰值；`struct_peak` = `pred_peak − framework`（剔 **177 MiB** 标定残余）；
  其余为峰值时刻 8 桶快照（MiB）；`hccl` = `num_distinct_communicators(pc)`（不同 HCCL 子通信器数，**仅 reserved 池、不进 allocated 峰值**）。
- 全部数据为 `analyze_matrix.py` 实跑输出（单位 MiB）。`swap_buf` 全程为 0（SwapSpec 未启用），故从表中略去。

> **本次更新（2026-06-30）：把 loss 反向的 fp32 大头从 `framework_reserve` 里拆出来显式建模。**
> 原 `framework_reserve=2197 MiB` 中 **~2020 MiB 其实是 loss 反向漏建的 `grad_log_softmax`**（`scatter_add` 出的满 vocab fp32 梯度，
> 见 `pynative/loss/loss.py:80-82`）。补建进 `nll.bwd_scratch`（`probs + grad_log_softmax = 2×4·S·B·vocab ≈ 4040 MiB`）后，
> `framework_reserve` 收窄到 **177 MiB**，`pred_peak` **不变**（三锚点仍 1.000/0.999/0.996）——证明那 2GB 是 **loss 激活、非框架开销**，
> `framework_reserve` 只是个**记账兜底位**。副带修正：PP 早 stage（无 lm_head）此前被旧常数错收 2020，现降回真实值。

> **峰值发生位置（关键）**：所有配置（含 **PP 的 tightest stage**，即含 lm_head 的末 stage）的峰值事件都落在
> **lm_head 的反向（`bwd@<head>`）**——此处 `bwd_scratch` = NLL 反向同存的 `probs`+`grad_log_softmax`（2 个 fp32 满 vocab ≈ 4040 MiB）
> + `act_live` 里 saved 的 logits(bf16)+log_softmax(fp32) 主导。lm_head **不被 tp/ep 切**、只有一层（**不可被 pp 拆**），
> 是所有维度下"切不动"的硬底，恒钉在末 PP stage。这解释了下面多数桶为何不随某些并行维变化。

> **PP 逐 stage 仿真**：评估器对 PP **逐 stage 仿真**——`PeakMemoryReport.per_stage` 含**全部 stage**
> （层分布由 `parallel_model.py` 切，持久态由 `static_mem.py` 逐 stage 算，激活由 `mem_timeline.py` 按各 stage 的
> 1F1B warmup 深度 `min(pp-1-stage, m)` 仿真），`tightest_stage` 给出**真实单卡设备峰值**所在。
> **不同 stage 负载不一致**（末 stage 含 lm_head 最重、早 stage 在飞 microbatch 多但 full 重算下极轻），
> 故 **PP 的设备峰值必须取 tightest stage，绝不能用 stage 0**（stage 0 会把 PP 收益夸大近 3 倍）。详见 Sweep D。

---

## Sweep A — FSDP (`dp_shard`)，N=4，tp=ep=pp=1

| config | pred_peak | struct_peak | persistent | act_live | gather_buf | grad_buf | recomp_scratch | bwd_scratch | workspace | hccl |
|---|---|---|---|---|---|---|---|---|---|---|
| dp_shard=1 | 16302.4 | 16125.4 | 7659.8 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 1 |
| dp_shard=2 | 12472.5 | 12295.5 | 3829.9 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 2 |
| dp_shard=4 | 10557.6 | 10380.6 | 1914.9 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 2 |
| dp_shard=8 |  9600.1 |  9423.1 |  957.5 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 2 |

**Trend**：`persistent` 严格 ∝ 1/dp_shard（7659.8 → 3829.9 → 1914.9 → 957.5，每翻倍精确减半），是唯一驱动峰值下降的桶。
`act_live`/`gather_buf`/`grad_buf`/`bwd_scratch` 全部恒定——FSDP 只切持久态（param+opt state），不切激活；
`gather_buf`(=441.9, lm_head 全量权重 bf16) 与 `grad_buf`(=883.8, 同权重 fp32 梯度=2×) 建模为**单层 full-unsharded** 缓冲、本就不随 FSDP 缩；
`bwd_scratch`(=4040, loss 反向 probs+grad_log_softmax 两个 fp32 满 vocab) 与 FSDP 无关。符合 ZeRO/FSDP 数学。

## Sweep B — EP on `dp_shard=8`，N=4，tp=pp=1

| config | pred_peak | struct_peak | persistent | act_live | gather_buf | grad_buf | recomp_scratch | bwd_scratch | workspace | hccl |
|---|---|---|---|---|---|---|---|---|---|---|
| ep=1 | 9600.1 | 9423.1 | 957.5 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 2 |
| ep=2 | 9600.1 | 9423.1 | 957.5 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 3 |
| ep=4 | 9600.1 | 9423.1 | 957.5 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 3 |
| ep=8 | 9600.1 | 9423.1 | 957.5 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 3 |

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
| tp=1 | 12472.5 | 12295.5 | 3829.9 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 2 |
| tp=2 | 11848.2 | 11671.2 | 3240.6 | 3065.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 3 |
| tp=4 | 11536.0 | 11359.0 | 2945.9 | 3047.5 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 3 |

**Trend**：TP 让 `persistent`（3829.9→3240.6→2945.9）与 `act_live`（3100→3065→3047.5）小幅下降，**收益递减、非 ∝1/tp**。
原因：只有 tp 切的注意力/dense/shared-expert 权重缩，而 embedding、lm_head、专家(ep 切)权重**不随 tp 缩**，构成 tp-不变底座；
`act_live` 里只有 sp 切的 checkpoint 输入(∝1/tp)缩，主导的 lm_head logits/logsm(未切)不动。
`gather_buf`/`grad_buf`/`bwd_scratch` 全不变——峰值在 **lm_head**，其权重/loss 张量不被 tp 切，故峰值处缓冲与 tp 无关
（若峰值落在 transformer 层，gather/grad 才会 ∝1/tp）。

## Sweep D — PP，N=8，dp_shard=2，tp=ep=1（**全 PP stage 仿真**）

评估器对 PP **逐 stage 仿真**，设备单卡峰值取 **tightest stage**（含 lm_head 的末 stage），**不是 stage 0**。

**① 设备峰值（= tightest stage，OOM 相关的真实单卡峰值）**：

| config | device_peak (tightest) | struct_peak | persistent | act_live | gather_buf | grad_buf | bwd_scratch | hccl | 对 pp=1 降幅 |
|---|---|---|---|---|---|---|---|---|---|
| pp=1 | 13896.1 (stage0) | 13719.1 | 5197.5 | 3156.0 | 441.9 | 883.8 | 4040.0 | 2 | — |
| pp=2 | **11335.9** (stage1) | 11158.9 | 2693.2 | 3100.0 | 441.9 | 883.8 | 4040.0 | 3 | −18% |
| pp=4 | **10980.0** (stage3) | 10803.0 | 2351.3 | 3086.0 | 441.9 | 883.8 | 4040.0 | 3 | −21% |

**② 全 stage 仿真明细**（层分布 / 在飞 microbatch / 各 stage 峰值 / 关键桶）：

`pp=2`（n_stages=2，设备峰值 = stage1 的 11335.9）

| stage | layers | inflight_mb | peak | persist | act_live | gather | grad | bwd_scr | event |
|---|---|---|---|---|---|---|---|---|---|
| 0 | [0–4] | 2 | 4062.9 | 2504.2 | 56.0 | 441.9 | 883.8 | 0.0 | bwd@0 |
| 1 **(tightest)** | [5–9] | 1 | 11335.9 | 2693.2 | 3100.0 | 441.9 | 883.8 | 4040.0 | bwd@9 |

`pp=4`（n_stages=4，设备峰值 = stage3 的 10980.0）

| stage | layers | inflight_mb | peak | persist | act_live | gather | grad | bwd_scr | event |
|---|---|---|---|---|---|---|---|---|---|
| 0 | [0–1] | 4 | 3023.2 | 1478.5 | 42.0 | 441.9 | 883.8 | 0.0 | bwd@0 |
| 1 | [2–3] | 3 | 1648.8 | 683.8 | 84.0 | 114.0 | 227.9 | 0.0 | bwd@3 |
| 2 | [4–5] | 2 | 1620.8 | 683.8 | 56.0 | 114.0 | 227.9 | 0.0 | bwd@5 |
| 3 **(tightest)** | [6–9] | 1 | 10980.0 | 2351.3 | 3086.0 | 441.9 | 883.8 | 4040.0 | bwd@9 |

**Trend / 关键结论**：
- **stage 间负载严重不均**：含 lm_head 的**末 stage 恒为瓶颈**（pp=2 stage1=11336、pp=4 stage3=10980），早 stage 轻得多
  （pp=4 stage0 仅 3023）。lm_head 的 `bwd_scratch`(4040, probs+grad_log_softmax) + logits/logsm 激活(~3.0 GB)**不可被 PP 拆分**，恒钉末 stage。
- **早 stage 的在飞 microbatch 几乎不费内存**：full 重算下每个在飞 microbatch 只 pin 极小 checkpoint 输入
  （pp=4 stage0 有 4 个在飞，`act_live` 仅 42 MiB）→ 1F1B warmup 堆叠**不是**本配置瓶颈，lm_head 才是。
  （若关掉 full 重算，早 stage 的 warmup 堆叠会成为另一个瓶颈来源——届时逐 stage 仿真同样能捕获。）
- **早 stage 峰值随本次 loss-grad 修正下降 ~2020**（pp=2 stage0 6083→4063）：旧的 `framework_reserve=2197` 把 loss 区 2GB
  错误地加到了**每个 stage**，包括无 lm_head 的早 stage；拆出 loss-grad 进 `bwd_scratch`（只在 lm_head stage）后，早 stage 只剩真实的 ~177 残余。
- **PP 对设备峰值仅 ~18–21% 改善**，远小于"只看 stage 0"会误以为的 ~71%（4063/3023）。
  对比 Sweep A：**FSDP `dp_shard=4` 单独就到 10557.6，已优于 PP-4 的 10980**——本（缩层 + 巨 vocab）配置下 PP 不是降单卡峰值的高效手段。

## Sweep E — Combinations

| config | pred_peak | struct_peak | persistent | act_live | gather_buf | grad_buf | recomp_scratch | bwd_scratch | workspace | hccl |
|---|---|---|---|---|---|---|---|---|---|---|
| dp_shard=2,ep=2 | 12472.5 | 12295.5 | 3829.9 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 3 |
| dp_shard=4,tp=2 | 10227.9 | 10050.9 | 1620.3 | 3065.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 3 |
| dp_shard=2,tp=2,ep=2 | 11848.2 | 11671.2 | 3240.6 | 3065.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 4 |
| dp_shard=2,pp=2 (N=8) | 11335.9 | 11158.9 | 2693.2 | 3100.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 3 |
| dp_shard=2,tp=2,ep=2,pp=2 (N=8) | 10617.1 | 10440.1 | 2009.4 | 3065.0 | 441.9 | 883.8 | 0.0 | 4040.0 | 0.0 | 5 |

> 注：含 PP 的两行 `pred_peak` 报 **tightest stage（设备峰值）**，非 stage 0。

**Trend**：各维叠加效果**可加且单调下降**，无相互抵消异常。`dp_shard=2,ep=2` ≡ `dp_shard=2`（ep 不动峰值，Sweep B）；
加 tp=2 在其上再省 ~620 MiB（`12472.5→11848.2`）；`dp_shard=4,tp=2` 把 persistent 压到 1620.3（FSDP×TP 双切权重）。
含 PP 的两行已按 tightest stage 报设备峰值（`dp_shard=2,pp=2`=11335.9，叠 tp2/ep2 进一步到 10617.1，主要省在 persistent）。
`hccl` 随启用的并行域线性增（最多 5），但**不进 allocated 峰值**。

---

## Validation against real machine

真机 rank0 `max_memory_allocated`（MiB）为 ground truth。预测/实测比值：

| anchor config | pred_peak | measured | ratio (pred/meas) |
|---|---|---|---|
| N=4, dp_shard=2 (Sweep A) | 12472.5 | 12473.1 | **1.0000** |
| N=4, dp_shard=2, ep=2 (Sweep E) | 12472.5 | 12474.1 | **0.9999** |
| N=8, dp_shard=2 | 13896.1 | 13953.3 | **0.9959** |

三锚点全部落在 **±0.5%** 内（loss-grad 拆分**不改变** pred，只把 2GB 从 `framework` 桶搬进 `bwd_scratch` 桶）。
ep=2 锚点尤为关键：真机 allocated 峰值几乎不随 ep 变（HCCL 涨在 reserved 池），
与评估器"ep 不动 allocated 峰值"完全吻合，佐证了 `framework_reserve(allocated)` **不含** per-comm-domain 的 +200 MB。

---

## Findings / caveats

**已锚定（validated）**：
- **结构分片模型仅在 `dp_shard=2`（ep1/ep2）与 N=8 三点对真机标定**，误差 ≤0.5%。
- **loss 区 fp32 大头已显式建模、随 seq·vocab 缩放**：NLL 反向 `probs + grad_log_softmax = 2×(4·S·B·vocab) ≈ 4040 MiB`
  （对照 `loss.py:80-82`），不再靠 `framework_reserve` 常数兜。这把 `framework_reserve` 从 2197 收窄到 177，pred 不变。
- HCCL/通信缓冲在 **reserved** 池、**不进 allocated 峰值**：`framework_reserve(allocated)=177 MiB` 小常数，
  **不**按通信域数 +200 MB（ep=2 真机点证实，`cost_eval/framework.py` §8.6）。`hccl` 列仅记录子通信器数供 reserved 侧参考。

**仅预测、未经真机验证（evaluator-only）**：
- **Sweep A 的 dp_shard∈{1,4,8}、Sweep B 全部、Sweep C 全部 TP、Sweep D/E 全部 PP** 均为评估器外推，
  真机验证被机器 HCCL flakiness 阻断，未取点。这些行的趋势在数学上自洽，但绝对值待真机标定。
- `framework_reserve=177` 假定对所有配置近恒定；其随 seq（flash workspace）、更大 world 的缩放**未验证**（§8.7 开放项）。
  注：loss 区 ~4040 现已随 seq·vocab 自动缩放，故 seq 缩放的不确定性已大幅降低，只剩 flash/MoE staging 的小项。

**需要关注的建模性结论（非 bug，但与朴素预期相左）**：
- **EP 不改变 allocated 峰值**（Sweep B 完全平坦）。任务预期"EP 让专家 persistent/act ∝1/ep 缩"在本模型下**不成立**：
  (a) efsdp 设计使专家持久态对 ep 恒定（专家始终切满 `dp_shard·tp` region）；
  (b) 专家激活虽 ∝1/ep，但在 full 重算下只活在 transformer 反向 `recomp_scratch`，而峰值在 lm_head 处，二者错峰。
  此结论**已被真机 ep=1≈ep=2 佐证**，是正确行为。若未来关掉 full 重算、或峰值移到 MoE 层，EP 才会显现于峰值。
- **PP 逐 stage 仿真、设备峰值取 tightest stage**：stage 0 不含 lm_head、是最轻 stage，用它会把 PP 收益夸大近 3 倍；
  真实单卡峰值由含 lm_head 的末 stage 决定（pp=2: 11335.9, pp=4: 10980.0），PP 对设备峰值仅 ~18–21% 改善
  （本缩层+巨 vocab 配置下甚至不及 FSDP-4 的 10557.6）。`analyze_matrix.py` 默认对 PP 报 tightest 并打印**全 stage 明细**。
- **`recomp_scratch`/`workspace` 在所有峰值快照中均为 0**：因峰值统一落在 lm_head 反向（无重算、workspace 已释放）。
  这不代表它们恒为 0——只是从不在峰值时刻；transformer 层反向时 `recomp_scratch` 非 0，但低于 lm_head 峰。
- **`framework_reserve` 是记账兜底位、非物理实体**：本次把其 92%（loss-grad）建模出来后它从 2197→177。
  原则：能建模的项逐一建出来，`framework_reserve` 应持续收窄（理想 → 仅分配器碎片）。

**物理合理性检查**：所有 A–E 配置峰值随分片**单调不增**，**未出现"加分片峰值反升"**的异常 ⇒ 无疑似评估器 bug。

---

## Related artifacts
- 驱动：[`analyze_matrix.py`](../analyze_matrix.py)（复用 `validate_dsv3.build_dsv3_spec`）
- 模型 / 锚点：[`validate_dsv3.py`](../validate_dsv3.py)
- 评估器核心：`cost_eval/{report,mem_timeline,static_mem,framework,parallel_model,shape_eval}.py`
- loss 区建模依据：`mindformers/pynative/loss/loss.py`（`_LogSoftmax`/`_NLLLoss`，§8.3）
