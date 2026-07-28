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
