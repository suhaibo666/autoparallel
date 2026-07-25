# 验收台 + `ResolvedLayer` 适配器契约（2026-07-25）

> **性质**：新增**验收门**与**接口契约**，不动任何模型口径。基线 `python -m pytest tests -q`
> → **1583 passed**（本次开工前实测复现，见 §0）。
> 所有权边界：本工作只碰 `tools/**`、`cost_eval/liveness/**`（仅新增文件）、`tests/` 新增件、本文档。
> **不碰** `cost_eval/opdag/**`（另一 agent 正在重写 walker）、`cost_eval/layers/**`、
> `structure_mem.py`、`mem_timeline.py`、`serve_explorer.py`。

## 0. 基线（开工锚点）[RAN]

```
$ python -m pytest tests -q
1583 passed, 268 warnings in 24.90s
```

分支 `feat/unified-llm-modelspec`，HEAD `7b9aa86`。

## 1. 里程碑 1 — graph source 插座（TASK A 的缝）[DONE]

新增 `cost_eval/liveness/sources.py`：一个**注册表**，`fn(model_spec, parallel_model) -> ResolvedGraphLike`。

* `hand_spec`（默认，恒在）= 原 `simulate.py:209` 那一行 `ShapeEval().resolve(spec, pm)`，**逐字节不变**。
* `extracted` = 惰性探测约定入口点 `cost_eval/opdag/to_resolved.py::resolve_graph(model_spec, parallel_model)`；
  文件不存在 → 静默跳过（预期状态）；**文件存在但入口点缺失 → fail-loud**（否则"抽取来源明明在了却
  悄悄不参与 A/B"是最坏的假绿）。第三方也可 `register_graph_source("extracted", fn)`。
* 接线点只有两处：`simulate_liveness(..., graph_source="hand_spec")`、
  `build_stage_graphs(..., graph_source="hand_spec")`。加 `extracted` 是**一次注册**，不是改写。

## 2. 里程碑 2 — `ResolvedLayer` 契约（TASK B）[DONE]

新增 `cost_eval/liveness/contract.py`：Protocol 三层面（layer/op/tensor）+ 18 条编号硬规则 +
`validate_resolved_layer` / `assert_resolved_layer` / `validate_resolved_graph` /
`validate_param_census`，以及派生量 `layer_total_bytes` / `layer_param_bytes` /
`layer_activation_bytes` / `layer_entry_names`。

两个已知硬点被**编成规则**（不是写在文档里靠自觉）：

| 已知硬点（评估文档 §7.1/§7.2/§9 实测） | 规则 |
|---|---|
| 裸抽取 DAG 的 shape 全是 `?` → `total_bytes=0` | **B1** `local_numel>0 且 dtype_bytes>0`、**B2** dtype∈{1,2,4,8}、**B3** `layer_total_bytes>0` |
| 权重被当 activation save（`FFNGroupedGEMM` 236 MiB 里 `w1`+`w2`=88 MiB） | **W2** saves 里不得有 `is_weight=True`；**W4** 权重不得是任何 op 的 output；**W1** params 全是权重；`validate_param_census`（**B4**）挡"权重压根没进 params" |

实测（`hand_spec` 六个 preset × 两组并行度）：**violations=0**，即缝是真的。

## 3. 里程碑 3 — 验收门（TASK A 主体）[DONE]

`tools/liveness_ab_validate.py` 由"复现一张表的脚本"改成**验收门**：

* **输入** = graph source（可多个）× 配置。八跑矩阵模式（默认）从**仓内固化**的两份 base yaml
  `analysis/realmachine/ab_fusion_2026-07-25/dsv4h_{fused,unfused}_pp4_recomp.yaml`
  **程序化派生**其余六跑（只改那 5 个字段），并 `check_derivation()` 逐项**反查标签**：
  fused 位 / 层数 / 微批数（由 `bundle.parallel.num_microbatches` 反查，非信任 gbs 换算）/
  重算域（逐 layer_id 查 `rc.is_full`）/ compress_ratios（值 + 长度 == 层数）/
  pp·dp·ep·tp·cp / seq_length / **每 stage 隐藏层数**（标签里 "2 层/stage" vs "1 层/stage" 的语义）。
  实测 8 跑 × 12 项**全过**。`--config` 为单配置模式（source × 一份 yaml + 契约体检）。
* **输出**：per-stage 峰值、逐格 `sim/real`、聚合、`--deltas` 差值表、
  `--dump-top N` **峰值时刻逐张量 live-set**（含类别标签 + 折算到既有桶名的 breakdown）、
  `--json` 机读、`--gate` 门（非零退出）。
* **并排**：`real / bucket / liveness(hand_spec) / liveness(extracted，可用时)`。列是**动态**的 ——
  `extracted` 落地即自动多一列 + 一组比值，不改这个脚本。
* **两条真机不变量**编成断言，既守 `REAL` 表自身（`REAL_SHA256` 指纹），也作为
  **任何 graph source 都必须满足的结构断言**。

### 3.1 八跑结果表（`--grad-mode chain2`，sources=hand_spec）

```
run                  st        real     bucket lv:hand_sp bucket/re hand_sp/re  peak@hand_spec
a fused   ON  L8 m4  0      24153.3    19307.1    17963.0     0.799     0.744  bwd@1/bwd21:swiglu
                     1      14641.7    14033.1    12592.9     0.958     0.860  bwd@4/bwd30:shared_fc2
                     2      14097.7    13777.1    12336.9     0.977     0.875  bwd@6/bwd30:shared_fc2
                     3      23508.0    22241.8    24261.8     0.946     1.032  bwd@9/bwd4:nll
b unfused ON  L8 m4  0      50187.9    37518.1    41194.8     0.748     0.821  bwd@2/bwd13:sparse_attn
                     1      43940.0    32898.6    36575.3     0.749     0.832  bwd@4/bwd13:sparse_attn
                     2      43407.1    32642.6    36319.3     0.752     0.837  bwd@6/bwd13:sparse_attn
                     3      47888.1    36931.6    40608.3     0.771     0.848  bwd@8/bwd13:sparse_attn
c fused   OFF L8 m4  0      30391.6    40342.2    40734.2     1.327     1.340  bwd@2/bwd30:shared_fc2
                     1      21019.4    29293.3    29429.3     1.394     1.400  bwd@4/bwd30:shared_fc2
                     2      17759.1    21964.1    21844.1     1.237     1.230  bwd@6/bwd30:shared_fc2
                     3      27720.1    27611.6    29887.6     0.996     1.078  bwd@9/bwd4:nll
d unfused OFF L8 m4  0          OOM   136776.2   136141.0         -         -  bwd@2/bwd13:sparse_attn
                     1          OOM   104006.8   105163.6         -         -  bwd@4/bwd13:sparse_attn
                     2          OOM    71773.1    74721.9         -         -  bwd@6/bwd13:sparse_attn
                     3          OOM    52516.1    52744.1         -         -  bwd@9/bwd4:nll
e fused   ON  L8 m8  0      25343.5    19918.6    18478.5     0.786     0.729  bwd@2/bwd30:shared_fc2
                     1      14631.5    14033.1    12592.9     0.959     0.861  bwd@4/bwd30:shared_fc2
                     2      14097.8    13777.1    12336.9     0.977     0.875  bwd@6/bwd30:shared_fc2
                     3      23508.6    22241.8    24261.8     0.946     1.032  bwd@9/bwd4:nll
f unfused ON  L8 m8  0      51378.2    38784.1    42460.8     0.755     0.826  bwd@2/bwd13:sparse_attn
                     1      43940.5    32898.6    36575.3     0.749     0.832  bwd@4/bwd13:sparse_attn
                     2      43407.6    32642.6    36319.3     0.752     0.837  bwd@6/bwd13:sparse_attn
                     3      47888.6    36931.6    40608.3     0.771     0.848  bwd@8/bwd13:sparse_attn
g fused   ON  L4 m4  0      18096.8    16225.6    14881.5     0.897     0.822  bwd@1/bwd21:swiglu
                     1       8867.9    10477.7     9037.5     1.182     1.019  bwd@2/bwd30:shared_fc2
                     2       7845.3    10072.6     8648.4     1.284     1.102  bwd@3/bwd29:shared_fc2
                     3      19623.0    18942.4    20962.4     0.965     1.068  bwd@5/bwd4:nll
h unfused ON  L4 m4  0      19862.6    21468.6    20928.4     1.081     1.054  bwd@1/bwd11:core_attn
                     1      36582.6    29343.2    33019.8     0.802     0.903  bwd@2/bwd13:sparse_attn
                     2      13750.2    16111.6    16211.3     1.172     1.179  bwd@3/bwd12:sparse_attn
                     3      44003.1    33632.2    37308.9     0.764     0.848  bwd@4/bwd13:sparse_attn

-- 聚合 sim/real（可评分格；不可评分 4 格：d@s0, d@s1, d@s2, d@s3）--
  bucket       n=28  mean=0.946  min=0.748  max=1.394
  hand_spec    n=28  mean=0.955  min=0.729  max=1.400
  run d 真机 OOM（s0 在崩前观测到 57712.7-57718.4 MiB，**不是**峰值）→ 该跑**不可评分**，
  模型值仅列出备查，不入任何统计。
```

`--grad-mode dataflow` 的聚合：`bucket n=28 mean=0.946`、
`hand_spec n=28 mean=0.893 min=0.672 max=1.400`。

### 3.2 不变量结论（两个 grad_mode 均 `结论: PASS`）

```
-- 真机不变量（REAL 表自身；这是本项目的两条核心物理发现）--
  PASS  REAL 指纹: sha256=41e279e591ae4ae9... == 预期
  PASS  I1 真机 x1 于层数: stage3 unfused-fused: L8(2 层/stage)=24380.1 L4(1 层/stage)=24380.1 |diff|=0.0 <= 0.1
  PASS  I2 真机 x1 于微批数: stage3 unfused-fused: m4=24380.1 m8=24380.0 |diff|=0.1 <= 0.4
-- 模型不变量（**任何 graph source 都必须满足同样的 x1 结构**）--
  PASS  I1 bucket    x1 于层数: stage3 delta L8=14689.8 L4=14689.8 |diff|=0.0
  PASS  I2 bucket    x1 于微批数: stage3 delta m4=14689.8 m8=14689.8 |diff|=0.0
  PASS  I1 hand_spec x1 于层数: stage3 delta L8=16346.5 L4=16346.5 |diff|=0.0
  PASS  I2 hand_spec x1 于微批数: stage3 delta m4=16346.5 m8=16346.5 |diff|=0.0
-- delta 幅值（**不做断言**，如实报告）--
  bucket    stage3 model=14689.8 real=24380.1 ratio=0.603
  hand_spec stage3 model=16346.5 real=24380.1 ratio=0.670   （chain2；dataflow 为 0.381）
```

> **口径要说准**：断言的是 x1 **结构**（delta 在层数与微批数上不动），**不是**幅值。两条来源
> 今天都**欠读**这个 delta（0.603x / 0.670x）。该缺口写进
> `test_delta_magnitude_gap_is_recorded` 明账，不藏进容差；修好了那个测试会红，提醒更新记录。

### 3.3 峰值时刻逐张量 live-set（示例：run b，stage0）

```
  [b unfused ON  L8 m4] stage0  peak=41194.8 MiB @bwd@2/bwd13:sparse_attn
                                重算工作集峰=20557.1 MiB @bwd@2/rerun13:sparse_attn
    tensor                       layer mb         MiB  category
    kv_g_fp32                    2     1       4096.0  recomp_saved
    ukv_bm                       2     1       4096.0  recomp_saved
    index_scores                 2     1       2048.0  recomp_saved
    kv_gathered                  2     1       2048.0  recomp_saved
    q / q_hnorm_fp32 / attn_weights / uq_f32   各 512.0  recomp_saved
    ...                                        5846.2  (29 个尾部张量)
    breakdown: remat_saves=18505.1, bwd_working_set=9101.1, persistent=6684.6,
               grad_accum=2269.7, gather_buf=2141.9, grad_buf=956.3, optstep=768.0, act_live=768.0
```

## 4. 逐字节中立的**证明**（不是断言）

三条独立证据：

1. **数值逐位比对**：改造前的脚本输出与改造后的输出，主表 + 聚合里的 154 个浮点数、delta 表里的
   60 个浮点数**逐位相同**（两个 grad_mode 各比一次）。
   ```
   chain2   main-table+aggregate floats: 154 154 IDENTICAL   delta-table: 60 60 IDENTICAL
   dataflow main-table+aggregate floats: 154 154 IDENTICAL   delta-table: 60 60 IDENTICAL
   ```
2. **缝的中立性单测**：`simulate_liveness(...)`（不传 `graph_source`）与显式 `"hand_spec"`
   **逐字段**相同 —— 含 `timeline`、峰值 `live_set`、`max_recompute_working_set`
   （`test_default_source_is_byte_identical_to_explicit_hand_spec`）；`build_stage_graphs` 同理；
   且 `resolve_graph(..., "hand_spec") == ShapeEval().resolve(...)`。
3. **golden 锁 + 全量回归**：8x4 桶模型峰值与 8x4x2(grad_mode) liveness 峰值全部锁成 golden
   （0.05 MiB = 打印精度），聚合均值锁到 3 位小数。

**全量回归**（在 `7b9aa86` 的**干净 worktree** 上只叠加本次改动 —— 主工作树里另一 agent 正在
重写 `cost_eval/opdag/`，其 in-flight 状态会让 opdag/timesim 测试暂时红，与本次改动无关）：

```
$ git worktree add --detach <tmp> 7b9aa86 && python -m pytest tests -q
1583 passed, 268 warnings in 32.74s          # 基线复现

$ <只拷入本次改动的 11 个文件 + 2 份 A/B yaml> && python -m pytest tests -q
1676 passed, 268 warnings in 40.45s          # 1583 全过 + 93 项新增
```

新增 93 项 = 验收门 34（`tests/test_acceptance_gate.py`）+ 契约 48
（`tests/test_resolved_layer_contract.py`）+ 缝 11（`tests/test_liveness_graph_source_seam.py`）。

## 5. 契约设计note

完整的接口契约note（逐字段 → 消费者定位 → 抽取侧现状 → 抽取侧必须补什么 → 交接步骤 →
尚存近似）在 [`resolved_layer_contract_2026-07-25.md`](resolved_layer_contract_2026-07-25.md)。

一个**施工中挖出来的真缺口**值得单记：第一版契约只按 `liveness/graph.py` 实际读的字段定
（`name / local_numel / dtype_bytes / is_weight / detached`），写"外部生产者"契约测试时直接炸在
`structure_mem.py:275` 的 `w.is_expert`（**直接属性访问，无 getattr 兜底**）。根因是
`simulate_liveness` 只接管**激活**四桶，其余非 liveness 桶仍走
`structure_mem.estimate_structure_memory` / `static_mem.StaticMem`（`simulate.py:211-213,268-276`）。
故 `is_expert` / `dim0` / `pin_under_recompute` 三项已从"桶模型专用、liveness 不读、缺省即可"
升为硬规则 **T2** —— 缺一项就等于"图能建、显存算不出来"。

## 6. 交付清单

| 项 | 文件 |
|---|---|
| graph source 插座 | `cost_eval/liveness/sources.py`（新）；`simulate.py` / `graph.py` 各接一行 |
| `ResolvedLayer` 契约 | `cost_eval/liveness/contract.py`（新） |
| 验收门 | `tools/liveness_ab_validate.py`（重写）、`tools/__init__.py`（新） |
| 真机 A/B 配置（固化进仓） | `analysis/realmachine/ab_fusion_2026-07-25/dsv4h_{fused,unfused}_pp4_recomp.yaml` |
| 测试 | `tests/test_acceptance_gate.py`、`test_resolved_layer_contract.py`、`test_liveness_graph_source_seam.py` |
| 文档 | 本文 + `docs/resolved_layer_contract_2026-07-25.md` |

## 7. 尚存近似 / 未做（如实列出）

1. **`extracted` 来源尚不存在** —— 本次交付的是插座 + 契约 + 契约测试，`cost_eval/opdag/to_resolved.py`
   由 opdag 侧落地（约定入口点见契约note §6）。表里的 `lv:extracted` 列会在它注册后自动出现。
2. **delta 幅值欠读**：×1 结构成立，但 stage3 的 `unfused−fused` 幅值只到真机的 0.603×（bucket）/
   0.670×（liveness·chain2）。已在测试里明账，未修（属模型侧工作，不属验收台）。
3. **run d 不可评分**：真机 OOM。模型值列出备查，`REAL` 表记 `None`，聚合排除。
   `REAL_D_S0_OBSERVED_RANGE = (57712.7, 57718.4)` 是崩前观测区间，**不是**峰值，不参与任何评分。
4. **×1 于微批数的 stage 作用域**：不变量按任务口径钉在 **stage3**。真机在其它 stage 上该 delta
   移动更多（s0 = 0.1、s2 = 0.4、**s1 = 10.7 MiB**），门里逐 stage 打出来（`--deltas`）但不断言 ——
   s1 那 10.7 MiB 目前**没有解释**，属未决。
5. **shard 正确性无法单层验证**：契约只能验"是不是正数"，除对了没有要靠 B4（param 字节 A/B）+
   逐 stage 峰值比对。
6. **`op.type` 取值域未纳入契约**：`structure_mem` 的 norm/flash/moe_gemm 判据与
   `RecomputeSpec.op_matches` 都按类型字符串工作，抽取侧若换拼法会**静默**走不同分支。
7. **`workspace_bytes` / `bwd_scratch_bytes` 仍是手写/profiler 标定** —— 有意保留的近似
   （kernel 实现细节，源码里读不出来；契约只要求 ≥0 的 int）。
8. **验收门加了约 10 s 测试时长**（8 跑 × 2 grad_mode × 3 次 `main()` 端到端）。

## 8. 提交与最终 pytest 行 [RAN]

两个提交，**各自独立可绿**（在 `7b9aa86` 的干净 worktree 上逐个 checkout 实测）：

| commit | 内容 | `python -m pytest tests -q` |
|---|---|---|
| `c4e5bf4` | `feat(liveness): graph source 插座 + ResolvedLayer 适配器契约` | `1642 passed, 268 warnings in 25.76s`（1583 + 59） |
| `2b77f92` | `feat(tools): 167 A/B 八跑验收门 —— source × config 的验收台` | **`1676 passed, 268 warnings in 38.51s`**（1583 + 93） |

同一干净 worktree 上验收门端到端：`派生反查违规: 0` / `不变量失败: 0 / 7` / `结论: PASS`。

### 主工作树的现状（如实说明）

主工作树里另一 agent 正在重写 `cost_eval/opdag/`（`construct_walker.py` 等 5 个文件、
+1441 行，**未提交**）。在那个 in-flight 状态下跑全量：

```
$ python -m pytest tests -q
14 failed, 1662 passed, 268 warnings in 37.83s
$ python -m pytest tests -q | grep ^FAILED | sed 's/::.*//' | sort | uniq -c
     12 tests/test_opdag_drop_diagnostics.py
      2 tests/test_opdag_gpt_segments.py
```

14 个红全部落在 `test_opdag_*`，**与本次改动无交集**（本次不碰 `cost_eval/opdag/**`）。
本次改动的权威回归结论以上表的干净 worktree 为准：**1676 passed**。
