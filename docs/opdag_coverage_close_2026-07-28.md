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
