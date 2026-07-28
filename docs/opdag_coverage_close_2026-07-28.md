# 抽取图覆盖度收口（2026-07-28）

> **性质**：施工日志 + 验收判决单（**边做边追加**，被打断也留证据）。
> 上游判决：[`to_resolved_adapter_2026-07-25.md`](to_resolved_adapter_2026-07-25.md) —— 缝是通的、
> 形态合格（18 条硬规则 0 违约），但**图只解析了约三分之一**，故 `extracted` 拒绝交出峰值。
> 本轮的任务：**把覆盖度缺口收口**，让 `extracted` 有可能给出可辩护的数；收不完就**如实说到哪儿**。
>
> **权威源**：`E:\97-codes\torch_parallel\mf-src-167\`（`mindformers` @ `26354ff64`、
> `hyper_parallel` @ `41495aa2`）。**不用** `E:\97-codes\torch_parallel\mindformers`。
>
> 记法：**[RAN]** = 实跑（附命令 + 实际输出）；**[SRC]** = 逐字读源（带 `file:line`）。

---

## 0. 基线锚点 [RAN]

```bash
cd /e/97-codes/torch_parallel/pynative-cost-evaluator && git rev-parse HEAD
MINDFORMERS_ROOT=…/mf-src-167/mindformers python -m pytest tests -q
```
```
74d99f4b1b9e1b5b0ba0b1f3fbd5a2a02b6de6d5   (feat/unified-llm-modelspec)
1838 passed, 268 warnings in 121.83s
```

八跑门（下界模式）逐字复现上游判决单的数字：

```bash
COST_EVAL_EXTRACTED_ALLOW_PARTIAL=1 python tools/liveness_ab_validate.py \
    --grad-mode chain2 --deltas --gate
```
```
  bucket       n=28  mean=0.946  min=0.748  max=1.394
  extracted    n=28  mean=0.169  min=0.052  max=0.299     <- 下界，不是模型值
  hand_spec    n=28  mean=0.955  min=0.729  max=1.400
  覆盖度 extracted：节点 2568 → op 820（跳过 874）；params 已解析 115 / 未解析 33
  param census: 可解析 3290.877 MiB / 真正进图 1434.000 MiB
  WARN  I1 extracted ×1 于层数: L8=554.0 L4=82.0 |diff|=472.0   [advisory]  <- 不满足
  PASS  I2 extracted ×1 于微批数: m4=554.0 m8=554.0 |diff|=0.0  [advisory]  <- 满足
  结论: PASS（extracted 为 advisory，不判死）
```

---

## 1. 开工诊断：级联根节点长什么样 [RAN]

诊断脚本只读、**不进任何数字路径**。四处级联根的节点原貌：

| 级联根 | 节点原貌 | 为什么解不出 |
|---|---|---|
| `hyper_connection.py:408` `View/concat` ×8 | `ins=[]`、`out='alpha:?:bf16'`、**`attrs` 里没有 `param_operands`** | 三个操作数 `_to_local(self.alpha_{pre,post,res})` 都是 `Parameter` → 按契约 W2/W4 路由去 `param_operands`，`ins` 空；且它们被记在 **`hyper_connection.py:409`**（`_to_local(...)` 那一行）而节点记在 `:408` → `_emit` 里那句「按**行号相等**匹配」对不上，节点连"我用了哪几个权重"都不知道 |
| `hyper_connection.py:413` `Kernel` | `outs=['h_in','h_post','h_res_flat']`，`shape_infer` 对 `Kernel` 一律记 `constant_shape_unknown` | 融合内核产出形"非源可读" —— 但**源侧 docstring 逐字给了**（`:396-399`） |
| `linear.py:132` `View/transpose` | `ins=[]`、`attrs={'param_operands': ['weight']}` | 同 (a)：权重转置的操作数全是权重。**这一处行号恰好相等**，故 `param_operands` 在 attrs 里 —— 说明缺的只是「把权重形状喂回 shape 推断」 |
| `loss.py:197` `FusedFunction` | `ins=['logits']`、`bare_ctx_retained=['logits']`、`saved_ins_idx=[0]` | `_LogSoftmax.apply` 的**产出**形无规则 → 节点被跳过 → 连带 `ctx.logits`（全模型最大的一块，1010 MiB）拿不到 |
| `vocab_embedding.py:85` `IndexSelect` | `ins=['input_ids']`、`ins_slots=[1]`、`param_operands=['weight']` | `mint.gather(weight, 0, index)` 的产出形 = **index 的形状**（算子定义），而规则里直接记 `constant_shape_unknown` 不推 |

`_init_book.param` 里**已经**有这些权重的形状（本轮开工即实测）：

```
('hyper_connection.py','alpha_pre'/'alpha_post'/'alpha_res') -> (('1',), 'fp32')
('hyper_connection.py','bias')      -> ('(n+n)+(n·n)', 'fp32')
('csa.py','attn_sink')              -> (('n_heads',), 'fp32')
('router.py','weight')              -> (('E','H'), 'fp32')
('experts.py','weight1')            -> (('E','H','2·moe_ffn'), 'compute_dtype')
('experts.py','weight2')            -> (('E','moe_ffn','H'), 'compute_dtype')
…
```

→ **缺的只有三条通路**，不是一百个小问题：
(A) 权重形状喂回 shape 推断；(B) opaque 产出的**声明式**形状（带出处）；(C) gather 的算子定义。

---

## 2. 第一批修复：三条通路 + 三处走查修复 [RAN]

| # | 改动 | 文件 | 为什么是**源读**不是猜 |
|---|---|---|---|
| A1 | **跨行调用的权重归属**：`_emit` 按「param 记录行号 == 节点行号」匹配改成**按发射作用域**认领（本次 `_emit` 处理实参期间新记、且未被内层节点认领的权重就是本节点的） | `construct_walker.py` | `hyper_connection.py:408-411` 的 concat 跨四行，三个 `Parameter` 记在 `:409`；发射作用域是**语法事实**，不是行号巧合 |
| A2 | **权重派生节点的形状通路**：`ins` 为空且操作数全是 `Parameter` 时，用 `init_dims.param_shapes` 当伪输入轴 | `shape_infer.py` + `to_resolved.py` | 权重形状本来就由 `init_dims` 从 `__init__` 的 `Parameter(mint.empty(...))` 逐字读出（本轮之前只喂给 param census） |
| B | **opaque 产出的声明表** `_DECLARED_OPAQUE_OUTS`（8 条，逐条带 `file:line` + 理由，逐条进 `Coverage.declared_shapes`） | `to_resolved.py` | 每条都是源侧 docstring / 注释 / 紧邻 reshape 目标 / `forward` 的 `return` 逐字；轴结构没逐字给出的只声明**元素数**（`~` 前缀），宁少说勿多说 |
| C | **`mint.gather` 产出形 = index 形** | `shape_infer.py` | 算子定义。**只**对 `prim == "mint.gather"` 生效 —— advanced indexing（`csa.py:485`）语义不同，套上去会少算尾轴 |
| D | **数据流边桥接**（`bridge_by_edge`，缺省关） | `shape_infer.py` | 内联子 Cell 的返回值**丢的是名字、不是数据流边**（`h_in` → `aggregated_attn`，边 4→7 在）。缺省关是有意的：`timesim/producer.py:227` 按「几个输入已解出」判 S 分歧，多解出一个就多注入一条 AG —— 那是另一个子系统的口径 |
| E | **归约算子按已记的轴推形**（`reduce_dim` / `keepdim` / **新增** `reduce_all`） | `shape_infer.py` + `construct_walker.py` | 轴 walker 早就记了（G4），`shape_infer` 一直没用；「源侧根本没传 `dim`」是一条**可判定**的源事实（⇒ 全轴归约 ⇒ 标量），与「抠不出轴」必须分开记 |
| F | **RoPE 频率表的形状**（`[max_seq_len, 1, 1, qk_pos_emb_head_dim]`） | `shape_infer.py` | `rotary_pos_embedding.py:109-148` 逐字推导链（见代码注释），`dim = config.qk_pos_emb_head_dim` @ `deepseek_v4_hybrid_attention.py:175-177`；三处调用点共用同一实例（`:88` 层层下传） |
| G | **`init_dims` 支持局部元组 shape** + `Linear`/`VocabEmbedding` 的位置形参种子 | `init_dims.py` + `to_resolved.py` | `linear.py:84-85` `weight_shape = (output_size, input_size)`；种子值逐字来自唯一构造点 `gpt_model.py:252-253` / `language_model_embedding.py:66-68` |
| H | `sym_shape.add` 对**两侧纯常数**折叠 | `sym_shape.py` | `((1+1)+1)` 这种和式原子单元查不到符号映射 → 落 unresolved；两个字面量之和是已知整数 |

### 2.1 覆盖度逐步实测（run a，L8 fused）[RAN]

| 阶段 | 节点 | op | 跳过 | param 可解析 | param 进图 |
|---|---|---|---|---|---|
| 基线（`74d99f4`） | 2568 | 820 | 874 | 3290.9 MiB | 1434.0 MiB |
| + A1/A2 + B（权重派生 + opaque 声明） | 2568 | 986 | 791 | 3290.9 | 1434.0 |
| + D（数据流边桥接） | 2568 | 1094 | 737 | 3290.9 | 1434.0 |
| + E（归约按轴） | 2568 | 1164 | 702 | 3290.9 | 1434.0 |
| + C/F/G（gather / rope / Linear·Embedding 权重） | 2568 | **1226** | **671** | **5310.9** | **3454.0** |

`hand_spec` 侧 param 合计 6300 MiB 量级 → 抽取侧从 **23%** 提到 **55%**。

### 2.2 逐字节中立 —— diff，不是断言 [RAN]

`scratchpad/dump_numbers.py` 依赖从未进版本库的 `validate_dsv3.py`（前一轮的临时脚本），
在任何 clean checkout 上都跑不起来（实测 `ModuleNotFoundError`）。故新增
`scratchpad/dump_numbers_ab.py`：**站点就是验收门那八跑**，dump 手写 `ModelSpec` 全字段
（每个 op 的每个张量的每个字段）+ 桶模型 `Evaluator` 的 `peak/peak_event/全 breakdown`
+ `liveness(hand_spec)` 两种 grad_mode 的逐 stage 峰值。

```bash
git worktree add -f --detach <scratch>/wt-base 74d99f4
(cd <wt-base> && python scratchpad/dump_numbers_ab.py > before.txt)   # 4265 行
python scratchpad/dump_numbers_ab.py > after.txt                       # 4265 行
diff before.txt after.txt        ->  *** 0 diff ***
```

结构上也如此：`bucket` / `hand_spec` 的路径**根本不 import `cost_eval/opdag`**
（唯一的运行期引用是 `liveness/sources.py` 对 `extracted` 入口点的惰性探测）。

### 2.3 台账更新（`tests/test_to_resolved_adapter.py::LEDGER_BLOCKERS`）

2026-07-25 记的 4 处级联根**全部修掉**，测试的**不变量原样保留**（"级联根必须逐条在册；
修好一处就更新台账"），只换了例子 —— 新的两处逐条记在该常量的注释里。

---

## 3. 第二批修复：construct 局部标量 / 差式 / 两轴转置 / MatMul 取权重末轴 [RAN]

| # | 改动 | 文件 | 为什么是**源读**不是猜 |
|---|---|---|---|
| I | **construct 局部整数标量过 `OpDAG` 边界**（新字段 `OpDAG.const_scalars`，由 `_ScalarEnv.exported()` 产出，`shape_infer._scalar` 的**最后一档**） | `construct_walker.py` + `schema.py` + `extractor.py` + `shape_infer.py` | walker **早就**把 `pos_dim = self.config.qk_pos_emb_head_dim`（`deepseek_v4:203`）、`nope_dim = ... - pos_dim`（`:204`）求成了 host 整数（值全由 `config_flags` 派生），只是不过边界。**安全判据**：一个名字被绑成过两个不同的值就不导出（按名取值会取到另一段的值）；排在符号档之后（`S`/`B` 仍拿符号） |
| J | `sym_shape.sub` **差式原子** + `consumer._sym_value` 求值 | `sym_shape.py` + `consumer.py` + `shape_infer.py` | `indexer.py:179-180` `self.index_head_dim - self.qk_pos_emb_head_dim`；与既有 `//` 原子同一条路子（表达式保形，值由 DimTable 决定） |
| K | `mint.transpose(x, d0, d1)` 的**两轴互换**形态：walker 记 `swap_axes`，`shape_infer` 精确换轴（不再退 `numel_only`） | `construct_walker.py` + `shape_infer.py` | `linear.py:132` `self.transpose(weight, 1, 0)` —— 两个轴是源里逐字写着的常数 |
| L | **MatMul 缺 `out_dim` 时取权重末轴**（先 env 里那份当前形状，再退 `param_shapes`） | `shape_infer.py` | 算子定义 `[...,k] @ [k,n] → [...,n]`（与既有 `_bmm` / `_grouped_matmul` 同一条）。`Linear` 作为**顶层** Cell 抽取时没有 `build_module(..., output_size=…)` 那个调用点 ⇒ `out_dim` 缺席 |

### 3.1 覆盖度（run a，L8 fused）[RAN]

| 阶段 | op | 跳过 | param 可解析 | param 进图 |
|---|---|---|---|---|
| 基线 `74d99f4` | 820 | 874 | 3290.9 MiB | 1434.0 MiB |
| 第一批（§2） | 1226 | 671 | 5310.9 | 3454.0 |
| + I（局部标量） | 1370 | 599 | 5310.9 | 3454.0 |
| + J（差式） | 1426 | 571 | 5310.9 | 3454.0 |
| + K/L（转置 / MatMul 末轴） | **1466** | **551** | **5310.9** | **3454.9** |

unfused 支：op 1116 → **1454**，跳过 1430 → 1261。

### 3.2 八跑门（第二批之后）[RAN]

```
  bucket       n=28  mean=0.946  min=0.748  max=1.394     <- 逐字节与基线相同
  extracted    n=28  mean=0.361  min=0.096  max=0.638     <- 基线 0.169
  hand_spec    n=28  mean=0.955  min=0.729  max=1.400     <- 逐字节与基线相同
  PASS  I1 extracted ×1 于层数  : stage3 delta L8=66.0 L4=66.0 |diff|=0.0   [advisory]  <- **由 WARN 转 PASS**
  PASS  I2 extracted ×1 于微批数: stage3 delta m4=66.0 m8=66.0 |diff|=0.0   [advisory]
  结论: PASS
```

> ⚠ **两条 ×1 不变量都过了，但 delta 的幅值反而更小（66.0 / 24380.1 = 0.003×，基线 0.023×）。
> 这不是"更差"，也不是"更好" —— 它说明 stage3 的 `unfused − fused` 差在抽取侧仍然**没有意义**：
> unfused 支的覆盖度（跳过 1261）明显落后于 fused 支（跳过 551），两支的欠读量互相抵消，
> 差值就在 0 附近抖。** 幅值可辩护的前提是**两支都补齐**，见 §4。

---

## 4. 第四批 + 验收判决 [RAN]

第四批只有一条：**同名不同尺寸的权重按 `name#k` 拆开**而不是丢掉第二个。
`Compressor.ape` 在两个构造点的形状本来就不同（`head_dim` 一处 `v_head_dim`、一处
`index_head_dim`），按帧解出后**两个都是对的** —— 此前 `_weights_of` 在冲突时 `continue`
会**少读**第二份。改成与激活侧 `_bykey` 同一条契约 S3 处置（冲突仍逐条记账：记的是
「发现了两个同名物理张量」，不是错误）。

### 4.1 八跑表（最终）[RAN]

```bash
COST_EVAL_EXTRACTED_ALLOW_PARTIAL=1 python tools/liveness_ab_validate.py \
    --grad-mode chain2 --deltas --gate --dump-top 8 --dump-stages all --dump-source extracted
```

```
run                  st        real     bucket lv:extract lv:hand_sp bucket/re extract/re hand_sp/re
a fused   ON  L8 m4  0      24153.3    19307.1    13353.0    17963.0     0.799     0.553     0.744
                     1      14641.7    14033.1     5830.8    12592.9     0.958     0.398     0.860
                     2      14097.7    13777.1     5574.8    12336.9     0.977     0.395     0.875
                     3      23508.0    22241.8    13439.6    24261.8     0.946     0.572     1.032
b unfused ON  L8 m4  0      50187.9    37518.1    13655.0    41194.8     0.748     0.272     0.821
                     1      43940.0    32898.6     5648.9    36575.3     0.749     0.129     0.832
                     2      43407.1    32642.6     5392.9    36319.3     0.752     0.124     0.837
                     3      47888.1    36931.6    13505.6    40608.3     0.771     0.282     0.848
c fused   OFF L8 m4  0      30391.6    40342.2    18662.9    40734.2     1.327     0.614     1.340
                     1      21019.4    29293.3    10627.0    29429.3     1.394     0.506     1.400
                     2      17759.1    21964.1     8343.0    21844.1     1.237     0.470     1.230
                     3      27720.1    27611.6    14870.7    29887.6     0.996     0.536     1.078
d unfused OFF L8 m4  0          OOM   136776.2    15765.9   136141.0         -         -         -
                     1          OOM   104006.8     7703.1   105163.6         -         -         -
                     2          OOM    71773.1     6398.8    74721.9         -         -         -
                     3          OOM    52516.1    13945.1    52744.1         -         -         -
e fused   ON  L8 m8  0      25343.5    19918.6    14020.6    18478.5     0.786     0.553     0.729
                     1      14631.5    14033.1     5830.8    12592.9     0.959     0.399     0.861
                     2      14097.8    13777.1     5574.8    12336.9     0.977     0.395     0.875
                     3      23508.6    22241.8    13439.6    24261.8     0.946     0.572     1.032
f unfused ON  L8 m8  0      51378.2    38784.1    13911.0    42460.8     0.755     0.271     0.826
                     1      43940.5    32898.6     5648.9    36575.3     0.749     0.129     0.832
                     2      43407.6    32642.6     5392.9    36319.3     0.752     0.124     0.837
                     3      47888.6    36931.6    13505.6    40608.3     0.771     0.282     0.848
g fused   ON  L4 m4  0      18096.8    16225.6    12094.3    14881.5     0.897     0.668     0.822
                     1       8867.9    10477.7     4303.6     9037.5     1.182     0.485     1.019
                     2       7845.3    10072.6     3736.9     8648.4     1.284     0.476     1.102
                     3      19623.0    18942.4    12168.3    20962.4     0.965     0.620     1.068
h unfused ON  L4 m4  0      19862.6    21468.6    12342.3    20928.4     1.081     0.621     1.054
                     1      36582.6    29343.2     4121.6    33019.8     0.802     0.113     0.903
                     2      13750.2    16111.6     3735.4    16211.3     1.172     0.272     1.179
                     3      44003.1    33632.2    12234.3    37308.9     0.764     0.278     0.848

-- 聚合 sim/real（28 个可评分格；run d 真机 OOM → 不入统计）--
  bucket       n=28  mean=0.946  min=0.748  max=1.394     <- 与基线**逐字节相同**
  extracted    n=28  mean=0.397  min=0.113  max=0.668     <- 基线 0.169（**仍是下界**）
  hand_spec    n=28  mean=0.955  min=0.729  max=1.400     <- 与基线**逐字节相同**
  验收门结论: PASS（派生反查 0 违规、不变量 0 失败、来源解图失败 0）
```

### 4.2 覆盖度（对基线 2568→820 / 跳过 874）[RAN]

| 口径 | 基线 `74d99f4` | 本轮 | 变化 |
|---|---|---|---|
| 节点 → op（fused） | 2568 → **820** | 2568 → **1542** | **+722（+88%）** |
| 跳过（fused） | 874 | **513** | −361 |
| 节点 → op（unfused） | 1116 | **1922** | +806 |
| 跳过（unfused） | 1430 | **1027** | −403 |
| params 已解析 / 未解析 | 115 / **33** | 196 / **0** | 未解析清零 |
| param 可解析 | 3290.9 MiB | **5823.3 MiB** | +2532.4 |
| param 真正进图 | 1434.0 MiB | **3967.3 MiB** | **+2533.3（×2.77）** |
| 契约违约（10 层 × 两支） | 0 | **0** | 不变 |

原因码：`no_input_shape` 513→310、`constant_shape_unknown` 96→36、`reduce_axis_unknown` 56→14、
`split_size_unresolved` 47→11、`no_out_dim` 8→**0**、`weight_shape_unresolved` **0**；
新出现的 `needs_axis_structure=20` 是**好信号**：它表示「元素数解出来了、轴结构没有」，
而不是「整条不知道」。

### 4.3 两条 ×1 不变量 —— **都过了**，但要把话说清楚 [RAN]

```
  PASS  I1 extracted ×1 于层数  : stage3 delta L8=66.0 L4=66.0 |diff|=0.0   [advisory]  <- 基线 WARN(472.0)
  PASS  I2 extracted ×1 于微批数: stage3 delta m4=66.0 m8=66.0 |diff|=0.0   [advisory]
```

* **×1 于层数：由 WARN 转 PASS。** 机理是实的：stage3 的峰值事件从基线的
  `bwd@8/bwd7:n12_elementwise`（一个被移走了的中间事件）**回到了 loss 那一步** ——
  `bwd@9/bwd3:loss.n2_fusedfunction`，与 `hand_spec` 的 `bwd@9/bwd4:nll` 是**同一个算子**。
  峰值事件归位，正是「覆盖度过了线」的那个信号。
* **但幅值仍然不可辩护。** 逐桶看这 66.0 是什么（stage3，unfused 减 fused）：
  `persistent 4705.0−4675.0 = 30` + `gather_buf 1843.8−1819.8 = 24` +
  `grad_accum 1426.8−1414.8 = 12` = **66.0** —— 全部来自**两支 param 进图量的差**
  （unfused 4015.3 vs fused 3967.3 MiB），**不是**重算工作集。
  真值 24380.1 MiB 的那部分（unfused 的 `_construct_naive` 稀疏注意力链）**还没进图**：
  该层 saves 抽取侧只有 650.5 MiB，而 `hand_spec` 是 22045.3 MiB。
  **所以：不变量的形状对了，量还没有。** 见 §5 第 1 项。

### 4.4 逐张量归因（`--dump-top`，source=extracted）[RAN]

```
[a fused ON L8 m4] stage3  peak=13439.6 @bwd@9/bwd3:loss.n2_fusedfunction
  loss.log_softmax   9  1   1010.0  act_saved
  loss.logits        9  1   1010.0  act_saved      <- **台账第 5 项拿到了**（ctx.logits，loss.py:136）
  loss.log_softmax   9  1   1010.0  grad_act
  x                  7  1    128.0  act_boundary
  x                  8  1    128.0  act_boundary
  lm_head.input_     9  1     32.0  act_boundary
  breakdown: persistent=4675.0, act_live=2308.0, grad_buf=2020.0, gather_buf=1819.8,
             grad_accum=1414.8, bwd_working_set=1010.0, optstep=192.0

[a fused ON L8 m4] stage0  peak=13353.0 @bwd@1/bwd30:n31_fusedfunction
  q 256.0 recomp_saved / __ret__i5 256.0 recomp_saved / output__i1 256.0 grad_act / x 128.0 act_boundary …
  breakdown: persistent=5494.4, gather_buf=2266.6, grad_accum=1742.6, grad_buf=1048.0,
             bwd_working_set=836.3, optstep=768.0, act_live=640.0, remat_saves=557.0
```

对照基线（stage3 peak = 2984.2 MiB @`bwd@8/bwd7:n12_elementwise`，`persistent=691.2`、
`remat_saves=296.0`）：**峰值 ×4.5、persistent ×6.8、峰值事件归位**。

`dsv4hyb_r4_moe` 层的 saves 明细（fused 17 项 / 1115.0 MiB；unfused 16 项 / 650.5 MiB）：

```
fused   : q 256.0 / __ret__i5 256.0 / compressed_kv__i1 256.0 / x 128.0 /
          aggregated_attn 32.0 / input_layernorm_output 32.0 / x_detach__i1 32.0 /
          aggregated_ffn 32.0 / pre_mlp_layernorm_output 32.0 / __ret__i1 32.0 / …
unfused : q 256.0 / x 128.0 / topk_idxs__i1 33.0 / aggregated_attn 32.0 / … / weights 8.0
```

### 4.5 台账逐项复核（对 `to_resolved_adapter_2026-07-25.md` §5）

| # | 台账项 | 本轮状态 |
|---|---|---|
| 1 | `q_hnorm_fp32` **+768 MiB/r4 层**（不是 +512 —— `q` 被 `q_hnorm` 这个 norm op 保留，`structure_mem._dt` 把它也抬 fp32） | ✅ **抽取侧仍然对**：该层只有 `q = 256.000 MiB bf16`，**没有** `q_hnorm_fp32`（源 `deepseek_v4:245` 无 `.astype(float32)`） |
| 2 | `ukl1`/`ukl2` **−2048 MiB/r4 层**，**只对 `bucket` 路径成立**（`liveness(hand_spec)` 已因 `detached=True` 排除） | ✅ 抽取侧同样没有这两项；且现在 indexer 链**已解析**（不再是「因为解不出所以碰巧没有」） |
| 3 | 漏 `sinks` +256 **B**/层 | ❌ **仍未拿到**，但根因变了：`csa.py:687` 的 `ops.cast(attn_sink, fp32)` 现在**形状解得出**，可它的产出被判为**权重派生**（契约 W2/W4：权重永不进 saves）→ 那份 fp32 复本仍不在 saves 里。量级 = `n_heads × 4 B` = **256 B/层**（0.00024 MiB），四舍五入级 |
| 4 | 漏 `sparse_indices` 净 0 | ➖ 净影响本就是 0；抽取侧 `topk_indices` 在 `with _no_grad()` 里、按源正确判为 detached |
| 5 | **`ctx.logits`（`loss.py:136`，全模型最大的一块）** | ✅ **拿到了**：`loss.logits = 1010.0 MiB act_saved`，且它就是 stage3 峰值事件的主体 |
| 6 | `npu_mhc_pre_sinkhorn` 保存自己的 5 个输出（`sum_out`/`norm_out` 12.5–25 MiB/层） | ❌ 仍欠读：`saved_ins_idx` 只表达输入侧。本轮把其中 3 个的**产出形状**声明出来了（h_in/h_post/h_res_flat），但「kernel 保存自身输出」这条语义 `ROp` 表达不了 |

---

## 5. 诚实清单：还缺什么（按修复价值排序），以及**我在哪一条上停手、为什么**

停手判据：**收益变平** —— 前四条以下的每一条都不再是「加一条规则」，而是要给
`sym_shape` / walker 加一项**新能力**（符号整除代数、逐轴 SSA、kernel 自身输出语义）。
四批改动里前三批每批都换来 100–400 个 op；第 4 项之后再动就是重构，故在此停。

| # | 缺口 | 量级 / 证据 | 归属 | 为什么现在停 |
|---|---|---|---|---|
| **1** | **`sym_shape` 不知道 `k·(X//k) == X`（k 整除 X 时）** —— `compressor.py:196-203` `cutoff=(sq//ratio)*ratio; n_compressed=cutoff//ratio; reshape(kv,(n_compressed,ratio,b,-1))`：已知积里是 `S//4`、总积里是 `S`，`-1` 消元约不干净 → 整条压缩链只剩**元素数**（`~`）→ `:216` 的 `.sum(dim=1)` 按轴归约被拒 → **7/10 个 segment 的级联根** | 挡住 unfused 的 `_construct_naive`：该层 saves 抽取侧 650.5 MiB vs `hand_spec` 22045.3 MiB —— **这就是 §4.3 那个「幅值不可辩护」的全部来源** | `sym_shape.floordiv` / `divide` | 需要一个**带 DimTable 的规范化 pass**（或让 `//` 原子在整除可判定时直接约掉）。这是代数层的新能力，不是一条规则 |
| **2** | `no_input_shape = 310`（fused）/ `611`（unfused） | 绝大多数是第 1 条的**连带** | 同上 | 先修 1 再看 |
| **3** | `reshape_unresolved = 103` / `needs_axis_structure = 20` | 同源：轴结构在 `~` 档丢了 | `shape_infer` | 同上 |
| **4** | **`workspace_bytes` / `bwd_scratch_bytes` 恒 0** | `hand_spec` 侧的 `nll` 4236 MiB + mHC sinkhorn 2×256 + dispatch/combine 2×64 —— 抽取侧全 0 | 契约**有意**排除（kernel 实现细节，源码读不出） | 要让 A/B 幅值可比，得在 `to_resolved` 里加一条**显式带出处**的标定直通；本轮**没做**，因为那会把手写标定值洗进「源真值」 |
| **5** | `constant_shape_unknown = 36`：`mint.arange(x.shape[0])`（`router.py:532`）、`Tensor([...])`（`indexer.py:219`）等 | 小张量为主 | `shape_infer` + walker | 收益小 |
| **6** | `reduce_axis_unknown = 14`：`topk` 的 `k`（`indexer.py:262` `k=effective_topk` 是 construct 局部）已可解，剩下的是 `dim` 为变量的少数点 | 小 | `shape_infer` | 收益小 |
| **7** | `slice_bounds_unknown = 11` / `split_size_unresolved = 11` | 小 | `shape_infer` | 收益小 |
| **8** | **kernel 保存自身输出**（`npu_mhc_pre_sinkhorn` 的 `sum_out`/`norm_out`，12.5–25 MiB/层、8 层 100–200 MiB） | 两侧都欠读（手写侧用一个标定的 `bwd_scratch=256 MiB` 顶） | `ROp` / 契约 | `saved_ins_idx` 只表达输入侧；要表达「自身输出」得改契约面 |
| **9** | **权重派生 buffer 不进 saves**（契约 W2/W4）→ `csa.py:687` 的 `sinks` fp32 复本、`experts.py:218` 的 `w1/w2` cast 复本 | `sinks` = 256 **B**/层；`w1/w2` cast 在 bf16→bf16 时是恒等 | 契约 | 契约是**有意**的（权重进 saves 会把 88 MiB 权重当激活计，实测过）。要区分「权重本体」与「权重的 dtype 复本」需要新字段 |
| **10** | `detach 名册未匹配 8 项`、`同名同尺寸但物理不同的张量仍会被合成一个` | 未量化 | `to_resolved` / walker | 根治要 walker 把 `attrs["frame"]` 纳入 SSA 名 |
| **11** | 激活的 TP/CP placement 抽取侧拿不到 → `tp/cp > 1` **fail-loud**；MTP 通路未被本配置触发（`mtp_num_layers=0`）故未验证 | — | — | 八跑均 tp=cp=1，不影响验收 |

## 6. 一句话结论

**缝仍然是通的、形态仍然合格（18 条硬规则 0 违约），覆盖度从「约三分之一」提到了
「约六成」（op 820→1542、param 进图 1434→3967 MiB、params 未解析 33→0），
八跑 `extracted` 的 sim/real 从 0.169 抬到 0.397，`ctx.logits` 那 1010 MiB 进图了，
两条 ×1 不变量**都过了**、stage3 的峰值事件回到了与 `hand_spec` 相同的 loss 算子上。
但 `extracted` **今天仍然给不出可辩护的峰值** —— 它默认照旧拒绝出数：
`unfused − fused` 的 66.0 MiB 被查明是**两支 param 进图量之差**、不是重算工作集，
真正的那 24380.1 MiB（unfused `_construct_naive` 稀疏注意力链）仍被
**一处代数缺口**挡着：`sym_shape` 不知道 `4·(S//4) == S`。**这是唯一挡在前面的大石头。**

## Related

- [`to_resolved_adapter_2026-07-25.md`](to_resolved_adapter_2026-07-25.md) —— 上一轮的判决单（本文是它 §6 诚实清单第 1–4 项的执行）。
- [`resolved_layer_contract_2026-07-25.md`](resolved_layer_contract_2026-07-25.md) —— 18 条硬规则。
- [`opdag_bytes_2026-07-25.md`](opdag_bytes_2026-07-25.md) / [`opdag_bytes_blockers_2026-07-25.md`](opdag_bytes_blockers_2026-07-25.md) —— 字节底座与阻塞项（本轮 G2 的解法落在这里说的那条通路上）。

---

## 7. 第五批（收尾）与**最终**数字 [RAN]

§4/§5 的表来自第四批。收尾又做了三条（都仍在「一条规则」的范围内，故没停）：

| # | 改动 | 为什么是源读 |
|---|---|---|
| M | **`sym_shape.divide_expr`**：reshape 的 `-1` 消元约不干净时形成**整除原子** `(总积)//(已知积)`，而不是整条退 `numel_only`；`consumer._sym_value` 支持**符号分母**（不整除即 None，不取整） | `-1` 位按算子定义**就是** `numel(输入) // ∏(其余目标维)`，两边都是源解出的表达式，值完全由 DimTable 决定 —— 与既有 `S//4` 原子同一条「表达式保形」路子。这正是 §5 第 1 项那块石头的**绕行解**：不需要知道 `4·(S//4)==S`，直接把商保成表达式 |
| N | `consumer._sym_value` 里 `str.strip("()")` → **配对感知**的 `_strip_outer_parens` | 既有隐藏 bug：`2·((a)//(b))·c·(S//4)` 这种乘积项的**配对**尾括号会被 `strip` 剥掉、整串弄坏。整除原子进乘积项后才暴露 |
| O | **过期种子**：内联把子 Cell 形参名永久映射成调用方 ref 时，`ins` 全是「从未被本图产出过」的名字且入边 producer 数与 ins 数相等 → 改用边 | `VocabEmbedding.construct(input_)` 内联后 `:83/:84/:85` 三步的 `ins` 都写作调用方的 `input_ids` → gather 产出被算成 `B·S`，而源 docstring `vocab_embedding.py:76` 逐字写着 `output: (B, S, H)` —— **差 `H` 倍**。修正后 embedding 段的数**变大**了，同时也让若干过读的下游**变小**（见下） |

### 7.1 最终八跑 + 覆盖度 [RAN]

```
  bucket       n=28  mean=0.946  min=0.748  max=1.394     <- 与基线**逐字节相同**
  extracted    n=28  mean=0.390  min=0.113  max=0.674     <- 基线 0.169
  hand_spec    n=28  mean=0.955  min=0.729  max=1.400     <- 与基线**逐字节相同**

  覆盖度 extracted：节点 2568 → op 1564（跳过 502）；saves 未解析 0；
      params 已解析 196 / 未解析 0；shape 冲突 8（= 发现了 8 对同名异形权重，已按 name#k 拆开）
  param census: 可解析 5823.291 MiB / 真正进图 3967.291 MiB
  节点原因码: no_input_shape=285, reshape_unresolved=80, constant_shape_unknown=33,
             needs_axis_structure=16, reduce_axis_unknown=14, slice_bounds_unknown=11,
             split_size_unresolved=8
  级联根: ×4 compressor.py:216 needs_axis_structure / ×3 compressor.py:233 slice_bounds_unknown
         / ×1 deepseek_v4_hybrid_attention.py:205 / ×1 csa.py:485
  PASS  I1 extracted ×1 于层数  : L8=66.0 L4=66.0 |diff|=0.0  [advisory]
  PASS  I2 extracted ×1 于微批数: m4=66.0 m8=66.0 |diff|=0.0  [advisory]
  验收门结论: PASS
```

⚠ **聚合 mean 从第四批的 0.397 微降到 0.390，而覆盖度在涨**（op 1542→1564、
`no_input_shape` 310→285、`reshape_unresolved` 103→80）。这不是退步：
改动 **O** 修掉的是一个**过读**（`input_ids` 这个入口种子被下游误当成中间张量的形状），
若干格（c@s1 10627.0→9064.2、c@s2 8343.0→7296.1、g@s2 3736.9→3240.4）因此**下降**。
本来源给出的仍然是**下界**，所以「更接近真值」不是它的判据；**更接近源**才是。

### 7.2 三张最终对照表（基线 → 现在）

| 口径 | 基线 `74d99f4` | 最终 |
|---|---|---|
| 八跑 `extracted` mean sim/real | 0.169 | **0.390** |
| 节点 → op（fused） | 2568 → 820（跳过 874） | 2568 → **1564**（跳过 **502**） |
| 节点 → op（unfused） | 1116（跳过 1430） | **1956**（跳过 **1010**） |
| params 未解析 | 33 | **0** |
| param 真正进图 | 1434.0 MiB | **3967.3 MiB** |
| `ctx.logits`（1010 MiB） | ❌ 不在图里 | ✅ 在图里，且是 stage3 峰值事件主体 |
| ×1 于层数（`extracted`） | **WARN** \|diff\|=472.0 | **PASS** \|diff\|=0.0 |
| ×1 于微批数（`extracted`） | PASS | PASS |
| 契约违约（10 层 × 两支） | 0 | 0 |
| `bucket` / `hand_spec` | — | **逐字节不变**（4265 行 dump `diff` 为空） |
| `pytest tests` | 1838 passed | **1850 passed** |
