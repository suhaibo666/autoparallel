# `extracted` 适配器 + 验收门判决（2026-07-25）

> **性质**：施工日志 + 验收判决单。交付 `cost_eval/opdag/to_resolved.py`（把源抽取的 per-Cell
> `OpDAG` 折成满足 `cost_eval/liveness/contract.py` 18 条硬规则的 `ResolvedLayer` 序列），
> 把它注册成 graph source `extracted`，然后跑 167 A/B 八跑验收门并**如实**报告落点。
>
> **权威源**：`E:\97-codes\torch_parallel\mf-src-167\`（`mindformers` @ `26354ff64`、
> `hyper_parallel` @ `41495aa2`）。**不用** `E:\97-codes\torch_parallel\mindformers`。
>
> 记法：**[RAN]** = 实跑（附命令 + 实际输出）；**[SRC]** = 逐字读源（带 `file:line`）。
> 本文按里程碑**逐段 checkpoint**（被打断也留证据）。

---

## 0. 基线锚点 [RAN]

```bash
cd /e/97-codes/torch_parallel/pynative-cost-evaluator && git rev-parse HEAD && python -m pytest tests -q
```
```
48c68a13a54bc89350343710a8740a4db9195505
1795 passed, 268 warnings in 54.78s
```
→ 基线 **1795 passed** 复现；分支 `feat/unified-llm-modelspec`；开工 HEAD = `48c68a1`
（开工瞬间是 `20ae22c`，并行 agent 在读文档期间提交了 `48c68a1`；两者均在本轮**不可改**的
`cost_eval/opdag/` 其它文件上）。

权威快照校验：
```bash
cd /e/97-codes/torch_parallel/mf-src-167 && md5sum \
  mindformers/pynative/transformers/experimental_attention_variant/{csa.py,indexer.py}
```
```
81673be3ad3cdd2191e28dd000a13f0e *csa.py
8e18fee21c33507dd629c89f401986e6 *indexer.py
```

## 0.1 所有权边界（另有 agent 并行）

本轮**只**碰：`cost_eval/opdag/to_resolved.py`（新，本轮独有）、`cost_eval/liveness/**`、
`tools/**`、`tests/` 新增件、本文档。
**不碰** `cost_eval/opdag/` 其余文件、`cost_eval/layers/**`、`structure_mem.py`、
`mem_timeline.py`、`serve_explorer.py`。需要它们改动的，作为**建议项**列在 §9。

---

## 1. 开工前的关键实测 —— 缝的两侧到底差多少 [RAN]

（本节为设计依据，逐段在后续里程碑补齐。）

### 1.1 手写侧（`hand_spec`）的层册

八跑配置（`analysis/realmachine/ab_fusion_2026-07-25/`，pp4/dp2/ep2/tp1/cp1、S=4096、B=1、
L8 时 `compress_ratios=[0,4,128,4,128,4,128,4]`）解出的 `layer_pattern`：

```
0 embedding            stage0   ops=2   param=1010.000 MiB  act=  160.000 MiB
1 dsv4hyb_r0_dense     stage0   ops=24  param= 591.164 MiB  act= 3674.000 / 8921.000 MiB (fused/unfused)
2 dsv4hyb_r4_moe       stage0   ops=32  param= 479.805 MiB  act= 3660.562 / 22530.062 MiB
3 dsv4hyb_r128_moe     stage1   ops=31  param= 455.539 MiB  act= 3582.094 / 9625.094 MiB
… (4..8 交替 r4/r128) …
9 lm_head              stage3   ops=5   param=1010.016 MiB  act= 3222.000 MiB
```

（`cross_entropy_fused=True` → 本配置**没有**独立 loss 伪层，`logsoftmax`/`nll` 在 `lm_head`
层里；`mtp_num_layers=0` → **没有** MTP 伪层。）

### 1.2 抽取侧的起步状态

`extract_cell(HyperConnectionTransformerLayer, recurse=True)` 实测（fused、r4、mHC 融合）：

```
nodes 196  edges 238
param_operands 18 条（去重后 9 个名）：alpha_pre/alpha_post/alpha_res/bias（hyper_connection.py:409/414）、
    ape（compressor.py:208/209）、attn_sink（csa.py:683/687/689）、
    linear_o_group_proj（deepseek_v4_hybrid_attention.py:280/284）、
    weight（router.py:363/364）、tokens_per_expert（moe_layer.py:127）、
    weight1/weight2（experts.py:218/219/221/235）
dims_ctx {}          ← 顶层类 __init__ 求值结果里没有维度符号
detached ['x_detach__i1','qr_detach__i1','q','k','weights','cmp_residual_k','topk_indices','index_scores']
```

**两个结构性发现（决定本轮判决的量级）**：

1. **叶子 `Linear` 的权重不在 `param_operands` 里**。`build_module(submodules.linear_q_down, …)`
   被 `module_resolver.LEAF_OPTYPE` 解成**单个 `MatMul` 节点**（不递归进 `Linear.construct`），
   于是 `self.weight` 这个 `Parameter` 从未被 walker 看到。抽取侧 `param_operands` 只有 9 个名
   ——手写侧 r4 层有 20+ 个权重、479.8 MiB。
   **可补救**：该 `MatMul` 节点带 `attrs["in_dim"]` / `attrs["out_dim"]`（源侧
   `build_module(sub.X, input_size=A, output_size=B)` 逐字读出），而
   `pynative/layers/linear.py:81-83` [SRC] 明写
   `weight_shape = (output_size, input_size)` / `dtype=self.params_dtype`
   → 权重字节 = `out_dim × in_dim × params_dtype`，**是源读出的事实，不是拟合**。本轮据此补。
2. **`dims_ctx` 为空**：`extract_cell` 只装顶层类的 `__init__` 求值（`extractor.py:725`），
   内联子 Cell 里的 `self.<attr>` 解不出 → 必须像 `scratchpad/acceptance.py` 那样
   `merge_dims_ctx(*各子 Cell 的 dims_ctx)`；`param_shapes` / `param_dtypes` 同理。

---

## 2. 交付物与提交 [RAN]

| # | 提交 | 内容 |
|---|---|---|
| 1 | `bc36cb7` | 开工锚点文档（基线 1795 passed + 缝两侧起步实测） |
| 2 | `ad4fbe4` | **验收门按 graph source 隔离故障** + `to_resolved.py` 落地 |
| 3 | `39268c8` | 部分图默认不给数 + param census 与数据流解耦 + 声明式中间量形状 |
| 4 | `d0a64d3` | **两个静默错算的修复**：张量身份按 `(名,尺寸)`、`detached` 按帧精确的节点 attr |
| 5 | `3e91b67` | 30 项门测试（缝 / 纪律 / 回归钉 / 覆盖度明账 / 故障隔离） |

最终回归：

```bash
MINDFORMERS_ROOT=…\mf-src-167\mindformers python -m pytest tests -q
```
```
1808 passed, 268 warnings in 108.81s
```

**逐字节中立 —— diff，不是断言**（`scratchpad/dump_numbers.py`：手写 spec 全字段 dump +
完整 `Evaluator.evaluate()` 的 `peak_event`/`peak_bytes`/全 breakdown）：

```bash
git worktree add -f --detach <scratch>/wt-base eafd17b   # 本轮开工前的 HEAD
cd <wt-base> && python scratchpad/dump_numbers.py > before.txt      # 650 行
cd <main>    && python scratchpad/dump_numbers.py > after.txt       # 650 行
diff before.txt after.txt        ->  *** 0 diff ***
```

即：**新增了一个来源，没有改任何模型一个字节**。验收门里 `bucket` / `hand_spec` 的
8×4 峰值与聚合逐格锁在 `tests/test_acceptance_gate.py` 的 golden 上（0.05 MiB = 打印精度），
全过。

## 2.1 第一优先级：不成熟的来源**不许**把验收门拖下水

开工时 `extracted` 让 3 条 `test_acceptance_gate` 变红（walker 在
`compressor.py:197 cutoff < sq` 上 fail-loud）。fail-loud 本身是**设计要的行为**，不该被吞成
静默跳过、也不该靠放宽 walker 糊过去。正确处置是**按来源隔离**（`tools/liveness_ab_validate.py`）：

* `SourceFailure`：该来源那一列显式打 **`ERR`**（不是 0、不是空白），附异常类型 + 从消息里
  抠出的 `file:line`；其余来源照常评分、照常入统计。
* `Check.skipped` / `Check.advisory`：某来源无值 → 不变量标 **SKIP**（"不知道"和"通过"是两件事）；
  不在 `--expect-green`（默认 `bucket,hand_spec`）里的来源标 **advisory** —— 照常评估打印、
  **不参与退出码**。
* 新增「来源健康」段：逐来源 出数/ERR 跑数 + 首个错误 + 该来源自报的 `Coverage` 报告。
* 与 `liveness/sources.py` 那条「文件不存在→静默跳过；入口点缺失→fail-loud」**不冲突**：
  那条管**注册**坏了（最坏的假绿）；`SourceFailure` 管**模型未完备**（注册好着，图还抽不全）。
  两件事分开，否则「模型未完备」会被误报成「注册坏了」。

反面也钉住了：`test_gate_fails_when_a_required_source_is_red` —— 把一个必然失败的来源提名进
`--expect-green`，门**必须**判死。隔离 ≠ 放过。

---

## 3. 判决：`extracted` **还不能**给出可辩护的峰值

这是本轮最重要的结论，先说清楚。

### 3.1 默认行为：部分图**拒绝交出峰值**

`resolve_graph` 在图不完备时抛 `IncompleteExtraction`，而**不是**返回那个下界。理由：
一个跳过了 N 个节点的图，其峰值是**下界**，不是估计值；一旦它出现在与真机并排的表格里，
就必然被读成"模型读到了这么多"。任务纪律是「绝不把未解析张量零填进峰值；部分解析的图必须
报成 partial，而不是一个低数」，最诚实的落地就是**默认不给数**，把「缺什么、缺在哪一行」
写进异常消息里。

于是默认跑验收门时 `lv:extracted` 那一列是：

```
  extracted    advisory 出数 0/8 跑
      ERR 8 跑：extracted: IncompleteExtraction @hyper_connection.py:408: 抽出的图尚不完备，
      故**拒绝交出峰值**：节点 2568 → op 836（跳过 866）；param 可解析 3290.9 MiB / 真正进图
      1434.0 MiB；saves 未解析 0。级联根：View hyper_connection.py:408（×8）；…
```

要拿下界做诊断：`COST_EVAL_EXTRACTED_ALLOW_PARTIAL=1`（或 `allow_partial=True`）。

### 3.2 下界模式下的八跑表 [RAN]

```bash
COST_EVAL_EXTRACTED_ALLOW_PARTIAL=1 python tools/liveness_ab_validate.py \
    --grad-mode chain2 --deltas --gate
```

> ⚠ `lv:extract` 这一列是**下界**，不是模型峰值。它之所以低，是因为 2568 个节点里有 874 个
> 因字节解不出而被**整个跳过**（绝不零填），且 `workspace`/`bwd_scratch` 按契约恒 0。
> 把它当"抽取模型读到的显存"是错的。

```
run                  st        real     bucket lv:extract lv:hand_sp bucket/re extract/re hand_sp/re
----------------------------------------------------------------------------------------------------
a fused   ON  L8 m4  0      24153.3    19307.1     6124.2    17963.0     0.799     0.254     0.744
                     1      14641.7    14033.1     3112.2    12592.9     0.958     0.213     0.860
                     2      14097.7    13777.1     3048.2    12336.9     0.977     0.216     0.875
                     3      23508.0    22241.8     2984.2    24261.8     0.946     0.127     1.032
b unfused ON  L8 m4  0      50187.9    37518.1     6690.2    41194.8     0.748     0.133     0.821
                     1      43940.0    32898.6     3666.2    36575.3     0.749     0.083     0.832
                     2      43407.1    32642.6     3602.2    36319.3     0.752     0.083     0.837
                     3      47888.1    36931.6     3538.2    40608.3     0.771     0.074     0.848
c fused   OFF L8 m4  0      30391.6    40342.2     8069.8    40734.2     1.327     0.266     1.340
                     1      21019.4    29293.3     4683.8    29429.3     1.394     0.223     1.400
                     2      17759.1    21964.1     3817.8    21844.1     1.237     0.215     1.230
                     3      27720.1    27611.6     2951.8    29887.6     0.996     0.106     1.078
d unfused OFF L8 m4  0          OOM   136776.2     8613.8   136141.0         -         -         -
                     1          OOM   104006.8     4857.8   105163.6         -         -         -
                     2          OOM    71773.1     3973.8    74721.9         -         -         -
                     3          OOM    52516.1     3089.8    52744.1         -         -         -
e fused   ON  L8 m8  0      25343.5    19918.6     6188.2    18478.5     0.786     0.244     0.729
                     1      14631.5    14033.1     3112.2    12592.9     0.959     0.213     0.861
                     2      14097.8    13777.1     3048.2    12336.9     0.977     0.216     0.875
                     3      23508.6    22241.8     2984.2    24261.8     0.946     0.127     1.032
f unfused ON  L8 m8  0      51378.2    38784.1     6754.2    42460.8     0.755     0.131     0.826
                     1      43940.5    32898.6     3666.2    36575.3     0.749     0.083     0.832
                     2      43407.6    32642.6     3602.2    36319.3     0.752     0.083     0.837
                     3      47888.6    36931.6     3538.2    40608.3     0.771     0.074     0.848
g fused   ON  L4 m4  0      18096.8    16225.6     5410.0    14881.5     0.897     0.299     0.822
                     1       8867.9    10477.7     2290.2     9037.5     1.182     0.258     1.019
                     2       7845.3    10072.6     2126.0     8648.4     1.284     0.271     1.102
                     3      19623.0    18942.4     2226.2    20962.4     0.965     0.113     1.068
h unfused ON  L4 m4  0      19862.6    21468.6     5922.0    20928.4     1.081     0.298     1.054
                     1      36582.6    29343.2     2372.2    33019.8     0.802     0.065     0.903
                     2      13750.2    16111.6     2866.0    16211.3     1.172     0.208     1.179
                     3      44003.1    33632.2     2308.2    37308.9     0.764     0.052     0.848

-- 聚合 sim/real（28 个可评分格；run d 真机 OOM → 不入统计）--
  bucket       n=28  mean=0.946  min=0.748  max=1.394      <- 与改造前逐字节相同
  extracted    n=28  mean=0.169  min=0.052  max=0.299      <- **下界**，不是模型值
  hand_spec    n=28  mean=0.955  min=0.729  max=1.400      <- 与改造前逐字节相同
```

### 3.3 两条 ×1 不变量：一条**导出**了，一条**没有** [RAN]

```
-- 真机不变量 --
  PASS  I1 真机 ×1 于层数  : stage3 unfused-fused: L8(2 层/stage)=24380.1 L4(1 层/stage)=24380.1 |diff|=0.0
  PASS  I2 真机 ×1 于微批数: stage3 unfused-fused: m4=24380.1 m8=24380.0 |diff|=0.1
-- 模型不变量 --
  PASS  I1 bucket    x1 于层数  : L8=14689.8 L4=14689.8 |diff|=0.0
  PASS  I2 bucket    x1 于微批数: m4=14689.8 m8=14689.8 |diff|=0.0
  WARN  I1 extracted x1 于层数  : L8=554.0  L4=82.0    |diff|=472.0   [advisory]   <- **不满足**
  PASS  I2 extracted x1 于微批数: m4=554.0  m8=554.0   |diff|=0.0     [advisory]   <- 满足
  PASS  I1 hand_spec x1 于层数  : L8=16346.5 L4=16346.5 |diff|=0.0
  PASS  I2 hand_spec x1 于微批数: m4=16346.5 m8=16346.5 |diff|=0.0
```

* **×1 于微批数：`extracted` 导出了**（`|diff| = 0.0`）。这一条是**仿真器**的结构性结论
  （重算再执行发生在该区域自己的反向步 → 同一时刻只有一个区域在重算），来源无关，抽取图
  接上同一个仿真器就自动继承。
* **×1 于层数：`extracted` 不满足**（L8 554.0 vs L4 82.0）。**这是不完备性的直接症状，不是
  另一个物理发现**：抽取层的激活总量只剩真值的一小截，于是 stage 峰值不再落在「某一层重算
  工作集」那个事件上，而落在多层残留共存的事件上 → delta 随 stage 内的层数累加。
  `hand_spec` 的 stage3 峰值事件是 `bwd@9/bwd4:nll`（fused）/ `bwd@8/bwd13:sparse_attn`
  （unfused），`extracted` 的是 `bwd@8/bwd7:n12_elementwise` —— **峰值事件被移走了**。
  修复方向不是调不变量，而是把 §4 的级联根修掉。

### 3.4 幅值：这项工作存在的那个问题

| 来源 | stage3 `unfused − fused` | 真机 24380.1 MiB 的几分之几 |
|---|---|---|
| bucket | 14689.8 | **0.603×** |
| liveness · hand_spec · chain2 | 16346.5 | **0.670×** |
| **liveness · extracted（下界）** | **554.0** | **0.023×** |

即：**抽取侧今天不但没有补上手写侧欠读的那 0.33，反而只读到 0.023 —— 因为图本身只解析了
约三分之一。** 这就是诚实的答案：*not yet*。缺什么在 §4，逐张量的账在 §5。

---

## 4. 覆盖度：什么解出来了、什么没有 [RAN]

```
覆盖度 extracted（run a，L8 fused）：节点 2568 → op 820（跳过 874）；saves 未解析 0；
    params 已解析 115 / 未解析 33；shape 冲突 0；detach 别名过计 8
  判决: PARTIAL（峰值是下界，不是估计）
  param census: 可解析 3290.877 MiB / 真正进图 1434.000 MiB
  同名异形拆分 99 处（内联后不同帧的同名物理张量，契约 S3）；detach 名册未匹配 20 项
  节点原因码: no_input_shape=513, constant_shape_unknown=96, reduce_axis_unknown=56,
             reshape_unresolved=54, split_size_unresolved=47, no_out_dim=8
  级联根（各 segment 首个被跳过的节点）:
    x8   View           hyper_connection.py:408    no_input_shape
    x1   IndexSelect    vocab_embedding.py:85      constant_shape_unknown
    x1   View           linear.py:132              no_input_shape
    x1   FusedFunction  loss.py:197                constant_shape_unknown
```

逐层（run a / run b，MiB）：

| 层 | ops(抽取/手写) | param 抽取 | param 手写 | act 抽取 | act 手写 |
|---|---|---|---|---|---|
| L0 embedding | 2 / 2 | 0.000 | 1010.000 | 0.016 | 160.000 |
| L1 `dsv4hyb_r0_dense` | 27 / 24 | 460.000 | 591.164 | 1584.0 | 3674.0 |
| L2 `dsv4hyb_r4_moe` (fused) | 69 / 32 | 144.500 | 479.805 | 1092.5 | 3660.6 |
| L2 `dsv4hyb_r4_moe` (unfused) | 71 / 32 | 156.500 | 479.805 | 1177.0 | 22530.1 |
| L3 `dsv4hyb_r128_moe` (fused) | 37 / 31 | 132.000 | 455.539 | 1056.0 | 3582.1 |
| L9 `lm_head` | 2 / 5 | 0.000 | 1010.016 | 0.008 | 3222.0 |

（"ops 抽取 > 手写"不代表更全：手写侧一个 `sparse_attn` op 塌了源侧十几个小算子。）

### 4.1 **级联根只有 4 处源码行** —— 这是本轮最有价值的定位

`no_input_shape=513` 里绝大多数是**连带**。真正要修的是各 segment 执行序上**第一个**解不出的
节点：它的输出一旦有形状，后面整条链就活了。反事实实测（诊断脚本，**不进任何数字路径**）：

```
fused r4 单层（196 节点）：
  as-is                                out-resolved  10/196   gaps=152
  + aggregated_{attn,ffn}=[s,b,H]      out-resolved  75/196   gaps=111      <- 一条种子，7.5x
```

整模型：加这一条声明后 **op 164 → 836**（/2568）。

四处级联根，归结为 `shape_infer` 的**两类结构性盲区**：

| # | 盲区 | 实例 | 为什么源侧其实**知道** |
|---|---|---|---|
| **(a)** | **权重派生节点**：`ins` 为空（全部操作数是 `Parameter`，已按契约 W2/W4 路由去 `param_operands`）→ 无从推形状 | `hyper_connection.py:408` `alpha = concat((alpha_pre, alpha_post, alpha_res), -1)`；`linear.py:132` 权重转置 | `init_dims.param_shapes` **已经**把这些 `Parameter` 的形状从 `__init__` 读出来了（本轮 param census 就用它解出 3290.9 MiB）。缺的只是"把权重形状喂回 shape 推断"这一步 |
| **(b)** | **opaque `Kernel` 的输出**没有 shape 规则 | `hyper_connection.py:413` `npu_mhc_pre_sinkhorn` 的 `h_in / h_post / h_res_flat` | 源侧 docstring 逐字给了：`hyper_connection.py:396-399` `aggregated: [s, b, H]` / `h_res: [s,b,n,n]` / `h_post: [s,b,n,1]`；`parallel_mhc_pre_sinkhorn.py:260-265` 另有一份 |

(a)+(b) 串起来把 mHC 层的**主数据路**掐断：`alpha` → 内核 → `h_in` → 父帧的 `aggregated_*`
→ `input_layernorm_output` → 整个 attention + FFN。

**本轮的处置**（不越界改 `shape_infer.py`，它属并行 agent）：`_DECLARED_SHAPES` 通路 ——
调用方**声明**中间量形状，三重纪律：值逐字来自源（每条带 `file:line`）；名字**不得**被图内
任何节点产出（否则声明会覆盖推断结果 → `fail-loud`）；逐条进 `Coverage.declared_shapes`，
在覆盖度报告里可见，**永不与"推断出来的"混为一谈**（`kernel_saves` 的同一条纪律）。
今天表里只有一条：`aggregated_{attn,ffn} = [s,b,H]`。

### 4.2 param：可解析 3290.9 MiB，真正进图只有 1434.0 MiB

权重形状来自 `__init__`，**与激活数据流无关** → 承载节点被跳过时权重**仍然可解析**，只是进不了
图。故本轮把 param census 与数据流解耦，两个口径都记。差额 1856.9 MiB = 「解出来了但因为
承载它的节点被跳过而丢掉」。

抽取侧权重的两条源侧路径：

1. `Parameter(mint.empty(...))` 声明 → `init_dims.param_shapes` / `param_dtypes`，**按文件名
   消歧**（`Linear.weight` 是 `(vocab,H)`、`TopKRouter.weight` 是 `(E,H)`、`GroupedMLP.weight1`
   是 `[E,H,2·moe_F]` —— 全局按名合并必然撞车；而 `Parameter` 的声明与它的出现点在同一个文件里，
   用文件名消歧是精确的）；
2. **叶子 `Linear`**：`build_module(sub.X, input_size=A, output_size=B)` 被 `LEAF_OPTYPE` 解成
   **单个 `MatMul`**、不递归进 `Linear.construct` → `self.weight` 从未被 walker 看到。由节点
   `attrs["in_dim"]/["out_dim"]` + [SRC] `pynative/layers/linear.py:81-83`
   `weight_shape = (output_size, input_size)` / `dtype=self.params_dtype` 算出。这是**源读**，
   不是拟合。

仍缺的（33 条 `unresolved_params`）：norm 的 gamma、embedding 表、`o_group_proj` 等。

### 4.3 契约层面：0 违约

`validate_resolved_layer` 对 fused / unfused 两支的**全部 10 层**：**violations = 0**
（含 W1–W6 / S1 S3 / B1 B2 B3 / T1 T2 / G2 G3）。即：抽出来的这部分图，形态上是**合格**的
`ResolvedLayer`；问题纯粹是**覆盖度**，不是形态。

### 4.4 本轮修掉的两个**静默错算**（都由"内联后同名"引起）

| # | 症状 | 源侧事实 | 后果 | 修法 |
|---|---|---|---|---|
| 1 | 同基名不同尺寸被**首见定型**吞掉（契约 S3 明文警告的那一条） | fused r4 单层里基名 `q` 是**三个**物理张量：`deepseek_v4:240` 注意力 query（`S·B·n_heads·v_head_dim` = 256 MiB，帧 `self_attention@7`）、`indexer.py:177` indexer query（`S·B·index_n_heads·index_head_dim` = 64 MiB，帧 `…/indexer@2`）、`indexer.py:215` detached cast 复本（帧 `…/indexer@44`）。`weights`/`k`/`topk_indices` 同理 | 后两个静默拿到第一个的尺寸 | 身份键改成 `(基名, local_numel, dtype_bytes)`（消费者 ref 自带 shape ⇒ 无歧义），第 2 个以上签名给 `name#k` 显示名。**实测拆出 99 处** |
| 2 | `detached` 按**裸名**匹配误伤同名张量 | `dag.detached` 只是名册（基名）；节点级 `attrs["detached"]` 才是**帧精确**的（实测二者一一对应） | 注意力的 `q` 被标成 detached（它与 indexer 里那个 detached 的 `q` 同名）→ 掉出 `kept_for_backward` → 反向**少读 256 MiB / r4 层**，且 grad 可达性被错误切断 | 按节点 attr 的输出签名建 `_det_keys`；名册里有而 attr 上匹配不到的记 `Coverage.detach_unmatched`（不静默丢 detach 信息） |

两条都写成了回归钉（`tests/test_to_resolved_adapter.py` 的 C 组）。

---

## 5. 逐张量归因：5 条已登记的手写 census 错，抽取侧各是什么状况 [RAN]

站点：`b unfused ON L8 m4` 的 `dsv4hyb_r4_moe` 层（手写 saves 43 项 / 22045.3 MiB；
抽取 saves 9 项 / 424.0 MiB）。

| # | 台账项 | 手写侧现状 | 抽取侧现状 | 判决 |
|---|---|---|---|---|
| 1 | **`q_hnorm_fp32` +512 MiB/r4 层** | `q_hnorm.saves=[q]` 与 `sparse_attn.saves=[q_hnorm_fp32]` 是**同一个源张量的两个名字**，`structure_mem` 按名去重的字典各计一份 | 只有 **`q` = 256.000 MiB bf16**（`deepseek_v4_hybrid_attention.py:240`），**没有** `q_hnorm_fp32`；`:245` 在抽出的图里是 5 个 `Elementwise`（纯 bf16），**没有任何 `Norm` 节点** | ✅ **抽取侧对**。与源一致（`:245` 无 `.astype(float32)`；`csa.py:689-697` 只把 `weights`/`attn_sink` 抬 fp32） |
| 1b | 同上的**实测量级订正** | 实测 `q` 的 `nbytes_saved` = **512 MiB**（不是 256）——`q` 被 `q_hnorm` 这个 **norm** op 保留，`structure_mem._dt` 把它按 `norm_compute_dtype` 抬到 fp32；加上 `q_hnorm_fp32` 的 512 = **1024 MiB**，源侧只该有 **256** | — | ⚠ 台账记的是 **+512**，实测**有效过读是 +768 MiB / r4 层**（多出的 256 来自 `q` 的 norm-fp32 抬升）。建议更新台账 |
| 2 | **`ukl1`/`ukl2` −2048 MiB/r4 层** | 两项各 1024 MiB fp32、`detached=True`，仍在 `saves` 里 | **没有**这两项 | ⚠ **方向对，但抽取侧不是"导出"的**：那条链（`indexer.py:350` matmul + `:380` fp32 softmax）在抽出的图里**整段没解析出形状**而被跳过。不能记成抽取侧的功劳 |
| 2b | 同上的**口径澄清**（实测） | `liveness(hand_spec)` 侧**已经是对的**：`detached=True` → `requires_grad=False` → `kept_for_backward=False`（实测 `ukl1/ukl2 in_graph=True detached=True rg=False kept=False`） | — | ⚠ 这 −2048 MiB 只对 **bucket** 路径成立（`structure_mem` 把全部 `saves` 求和、不看 `detached`）。liveness 两条来源都不含它 |
| 3 | **漏 `sinks` +256 B/层** | `attn_sink` 只在 `params` 里；源 `csa.py:233` save 的是 `ops.cast(attn_sink, float32)`（`csa.py:687`）—— 一份**独立的 fp32 激活复本** | `csa.py:687/689` 那两个节点**被跳过**（形状未解析）→ 那份 fp32 复本不在图里 | ❌ **还没拿到**。`attn_sink` 本体在抽取侧走 `param_operands`（形态正确，W2/W3/W4 自动成立） |
| 4 | **漏 `sparse_indices` 净 0** | 同名张量已由上游 `indexer` op 声明 saved，`structure_mem` 按名去重 → 净 0 | indexer 链未解析 → 不在图里 | ❌ 还没拿到（但该项净影响本就是 0） |
| 5 | **`ctx.logits`（`loss.py:136`，全模型最大的一块）** | 手写 `lm_head` 层的 `logsoftmax.saves=[logits_lm]`（`S·B·vocab` bf16 = 1010 MiB）**有**这一项 | loss segment 的**首个**节点 `loss.py:197 FusedFunction`（`_LogSoftmax.apply`）就因 `constant_shape_unknown` 被跳过 → `logits` 不在图里 | ❌ 还没拿到。抽取器**认得**它（`bare_ctx_retained=['logits']`，`opdag_component_coverage §3.1 #4`），卡在字节解析而非语义识别 |
| 6 | **新增：`npu_mhc_pre_sinkhorn` 保存自己的输出** | 手写侧无对应物（`attn_hc_sinkhorn` 用一个标定的 `bwd_scratch = 256 MiB` 顶） | 抽取侧的 `saved_ins_idx="all"` 只覆盖**输入侧**（`x`,`alpha`）；那 5 个**自身输出**表达不了 → 显式 unresolved，**不给数** | ⚠ 两侧都欠读。源侧逐字：`ctx.save_for_backward(x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out)`（`custom_op_impl.py:390-391`），而调用点写 `h_in, h_post, h_res_flat, *_ = …`（`hyper_connection.py:413`）—— **`*_` 丢的是 Python 名字，不是设备内存**：autograd ctx 仍持有它们 |

第 6 项可定量的部分（形状逐字来自 `parallel_mhc_pre_sinkhorn.py:262-265`，站点 `S=4096, B=1,
N=hc_mult=4, num_iters=20`）：

```
sum_out  = (2·num_iters, B, S, N)    =   655,360 elems  -> bf16  1.25 MiB / fp32  2.50 MiB
norm_out = (2·num_iters, B, S, N, N) = 2,621,440 elems  -> bf16  5.00 MiB / fp32 10.00 MiB
```

每个 mHC 实例 6.25–12.5 MiB，每层 2 个实例（attn_hc + ffn_hc）→ **12.5–25 MiB/层**，
8 层 → 100–200 MiB/模型。dtype 未在快照里逐字确认，故给两档、**不选一个**。
`h_pre / hc_before_norm / inv_rms` 的末轴长度在未纳入快照的 `.cc` kernel 里 → **不给数**。

### 5.1 峰值时刻逐张量（`--dump-top`，source=extracted）[RAN]

```
[a fused ON L8 m4] stage3  peak=2984.2 MiB @bwd@8/bwd7:n12_elementwise
  q                       8  1   256.0  recomp_saved
  q                       8  1   256.0  grad_act
  x                       8  1   128.0  grad_act
  hidden_states           8  1   128.0  grad_act
  aggregated_attn         7  1    32.0  act_boundary
  aggregated_attn         8  1    32.0  act_boundary
  input_layernorm_output  8  1    32.0  recomp_saved
  aggregated_ffn          8  1    32.0  grad_act
  breakdown: persistent=691.2, bwd_working_set=622.5, gather_buf=553.0, remat_saves=296.0,
             grad_buf=289.0, grad_accum=276.5, optstep=192.0, act_live=64.0
```

对照 `hand_spec` 同格（24261.8 MiB @`bwd@9/bwd4:nll`）：缺口的三大来源，**逐桶**可见 ——

| 桶 | extracted | hand_spec（run b s0 的量级参照） | 差在哪 |
|---|---|---|---|
| `remat_saves` | 296.0 | 18505.1 | 层内 saved 集只解析出一小截（§4.1 的级联根） |
| `persistent` | 691.2 | 6684.6 | param 只有 1434/6300 MiB 进图（§4.2） |
| `bwd_scratch` | **0** | `nll` 4236 MiB + mHC sinkhorn 2×256 MiB + dispatch/combine 2×64 MiB | 契约 §不要求 明确排除（kernel 实现细节，源码读不出）→ 恒 0 |

第三项是**有意**的口径差、不是缺陷：契约把 `workspace_bytes`/`bwd_scratch_bytes` 划为
"继续沿用手写/profiler 标定"。但它确实让 `extracted` 相对 `hand_spec` 少掉那部分已标定量，
故记在 `Coverage.absent_calibration_note` 里，并在此明写。

---

## 6. 诚实清单：还缺什么（按修复价值排序）

| # | 缺口 | 归属 | 预估收益 |
|---|---|---|---|
| **1** | **`shape_infer` 给"权重派生节点"（`ins` 为空、操作数全是 `Parameter`）定型** —— 权重形状 `init_dims.param_shapes` 已经有了，只差喂回去 | `shape_infer.py` / `construct_walker.py`（并行 agent） | 直接解掉级联根 `hyper_connection.py:408` 与 `linear.py:132`；反事实实测：整模型 op **164 → 836** |
| **2** | **opaque `Kernel` 输出的 shape 规则**（`npu_mhc_pre_sinkhorn`）；源侧 docstring 逐字有 | 同上 | 与 #1 同一条链；另可把 `sum_out`/`norm_out` 这 100–200 MiB 纳入 |
| **3** | `constant_shape_unknown = 96`：`mint.arange/full/zeros` 的 shape 实参未记进 attrs（`vocab_embedding.py:85`、`loss.py:197` 两个级联根都是它） | `construct_walker.py`（G4 已做了一半） | 解掉 embedding / loss 两个 segment，`ctx.logits` 那 1010 MiB 随之进图 |
| **4** | `reduce_axis_unknown=56` / `reshape_unresolved=54` / `split_size_unresolved=47` | `shape_infer.py` | dsv4 注意力链的剩余部分 |
| **5** | **`workspace_bytes` / `bwd_scratch_bytes` 仍恒 0** | 契约有意排除；若要 A/B 幅值可比，建议在 `to_resolved` 里加一条**显式的、带出处的**标定直通（今天没做，因为那会把手写标定值洗进"源真值"） | 让 `extracted` 与 `hand_spec` 的 delta 幅值可比 |
| **6** | **`extracted` 的 ×1 于层数不成立** | 是 #1–#4 的症状，不是独立缺陷 | 修完上面几条应自动成立；在此之前它在门里是 advisory，不判死 |
| **7** | `detach 名册未匹配 20 项`：`dag.detached` 里有、但节点级 attr 匹配不到（多因形状未解析） | 随 #1–#4 消失 | — |
| **8** | 同名**同尺寸**但物理不同的张量仍会被合成一个（`(名,尺寸)` 键的残留近似） | `to_resolved.py`；根治要 walker 把 `attrs["frame"]` 纳入 SSA 名 | 未量化 |
| **9** | 激活的 TP/CP placement 抽取侧拿不到 → `tp/cp > 1` **fail-loud**（八跑均 tp=cp=1，不影响验收） | `to_resolved.py` + walker | — |
| **10** | `lm_head` 伪层只覆盖 vocab 投影 + loss 两段；`hc_collapse` / `final_norm` 在 `GPTModel.construct` 里（走查整个 `GPTModel` 是既定的非目标） | — | 1010 MiB 量级 |
| **11** | MTP 通路已实现但**未被本配置触发**（`mtp_num_layers=0`），故**未经验证** | — | — |

---

## 7. 一句话结论

**缝是通的、形态是合格的（18 条硬规则 0 违约，层号/stage 与 `hand_spec` 逐项对齐），
但图还只解析了约三分之一，因此 `extracted` 今天给不出可辩护的峰值 —— 它默认拒绝出数，
而不是给一个 0.023× 的低数冒充答案。** 挡在前面的不是一百个小问题，而是
**两类结构性盲区、四处源码行**（§4.1）；其中最大的一处（`hyper_connection.py:408` →
`npu_mhc_pre_sinkhorn` → `aggregated_*`）一条种子就把整模型的已解析 op 从 164 抬到 836。

## Related

- [`resolved_layer_contract_2026-07-25.md`](resolved_layer_contract_2026-07-25.md) —— 18 条硬规则与交接步骤（本文是它 §6 的落地）。
- [`acceptance_harness_2026-07-25.md`](acceptance_harness_2026-07-25.md) —— 验收门与八跑表的施工记录。
- [`opdag_bytes_2026-07-25.md`](opdag_bytes_2026-07-25.md) / [`opdag_bytes_blockers_2026-07-25.md`](opdag_bytes_blockers_2026-07-25.md) —— 字节解析底座与阻塞项（本文 §4 的级联根引它们）。
- [`opdag_component_coverage_2026-07-25.md`](opdag_component_coverage_2026-07-25.md) —— 各组件的抽取覆盖度。
