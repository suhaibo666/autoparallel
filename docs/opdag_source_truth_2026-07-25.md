# opdag 源真值抽取 + 静默丢弃 fail-loud 化（2026-07-25）

> **性质**：两项可立即交付的工作（评估文档 `docs/opdag_coverage_assessment_2026-07-25.md`
> 路线 A 的落地 + 路线 B 的 P0#1 门）。
>
> **权威源**：`E:\97-codes\torch_parallel\mf-src-167\mindformers\`（commit `26354ff64`）。
> 所有行号以此快照为准。md5 已复核：`csa.py 81673be3ad3cdd2191e28dd000a13f0e`（831 行）、
> `indexer.py 8e18fee21c33507dd629c89f401986e6`（409 行）。
>
> 基线：`python -m pytest tests -q` → **1526 passed**（复核通过）。
>
> 记法：**[RAN]** = 实跑所见；**[SRC]** = 逐字读源（带 file:line）。

---

## 0. 快照与基线 [RAN]

```
md5sum .../experimental_attention_variant/{csa.py,indexer.py}
81673be3ad3cdd2191e28dd000a13f0e *csa.py
8e18fee21c33507dd629c89f401986e6 *indexer.py
   831 csa.py   409 indexer.py   244 compressor.py   300 deepseek_v4_hybrid_attention.py
python -m pytest tests -q  ->  1526 passed, 268 warnings in 23.78s
```

---

## 1. 源真值表（Task 1 抽取目标）—— 手工逐字读结果 [SRC]

快照内全部 `_Function` 子类（`grep -rn "^class .*(_Function)"`，13 个）：

| 类 | 文件:行 | `save_for_backward` | 裸 `ctx.<attr>` |
|---|---|---|---|
| `_AllReduceFunction` | `pynative/distributed/style.py:78` | 无 | — |
| `_AllGatherFunction` | `pynative/distributed/style.py:93` | 无 | — |
| `_ShardFunction` | `pynative/distributed/style.py:114` | 无 | — |
| `AllGatherMatmulFunction` | `pynative/layers/mc2.py:57` | `:83` `(gathered, w)` | — |
| `MatmulReduceScatterFunction` | `pynative/layers/mc2.py:103` | `:129` `(x, w)` | — |
| `_VocabParallelCrossEntropy` | `pynative/loss/loss.py:85` | 无 | — |
| `_LogSoftmax` | `pynative/loss/loss.py:125` | 无 | — |
| `_NLLLoss` | `pynative/loss/loss.py:155` | 无 | — |
| `_ChunkCrossEntropyLoss` | `pynative/loss/loss.py:376` | 无 | — |
| `_ChunkVocabParallelCrossEntropy` | `pynative/loss/loss.py:468` | 无 | — |
| **`FusedSparseFlashMla`** | `csa.py:73` | **`:113` n=8** | `:111,112`（bool）`:123-128`（标量） |
| **`FusedSparseFlashMlaWithIndexerLoss`** | `csa.py:178` | **`:224` n=11** | `:222,223`（bool）`:237-245`（标量） |
| `_DSAIndexerFunction` | `.../dsa_indexer.py:42` | 待抽 | 待抽 |
| `_DSAIndexerGradFunction` | `.../dsa_indexer_loss.py:29` | 待抽 | 待抽 |
| **`_IndexerLossAutoScaler`** | `indexer.py:268` | **无** | **`:276` `ctx.indexer_loss = indexer_loss`（张量！）** |
| `_MoEAuxLossAutoScaler` | `pynative/transformers/moe/moe_utils.py:278` | 待抽 | 待抽 |
| `_MTPLossAutoScaler` | `pynative/transformers/multi_token_prediction.py:53` | 待抽 | 待抽 |

`save_for_backward` 名单（逐字，`csa.py`）：

```
FusedSparseFlashMla                csa.py:113  n=8
  query, ori_kv, cmp_kv, sparse_indices, cmp_residual, sinks, output, softmax_lse
FusedSparseFlashMlaWithIndexerLoss csa.py:224  n=11
  query, ori_kv, cmp_kv, sparse_indices, query_index, key_index, weights,
  cmp_residual, sinks, output, softmax_lse
```

模式（两处一模一样）：`ctx.save_for_backward(*[tensor for tensor in (<元组>) if tensor is not None])`
→ `Starred(ListComp)`，名单在 `generators[0].iter` 的 `Tuple.elts`。

`backward` 侧消费序（`csa.py:134-142` / `:251-262`）与 forward 名单**逐名同序**，
且 `cmp_kv` / `sparse_indices` 两项条件消费（`if ctx.has_cmp_kv` / `if ctx.has_sparse_indices`）
—— 与 forward 的 `if tensor is not None` 过滤一致。这条同序性可作抽取器的自校验。

### detach 站点 [SRC]

`ops.stop_gradient(...)` 快照内 6 处，全在 `csa.py`：

| file:line | 形态 | 被赋名 |
|---|---|---|
| `csa.py:665` | `x_detach = ops.stop_gradient(x)` | `x_detach`（fused 支） |
| `csa.py:666` | `qr_detach = ops.stop_gradient(qr)` | `qr_detach`（fused 支） |
| `csa.py:764` | `x_detach = ops.stop_gradient(x)` | `x_detach`（unfused 支） |
| `csa.py:765` | `qr_detach = ops.stop_gradient(qr)` | `qr_detach`（unfused 支） |
| `csa.py:794` | `self.unfused_indexer_loss(..., ops.stop_gradient(query), ...)` | **无赋名**（内联实参） |
| `csa.py:795` | 同上 `ops.stop_gradient(compressed_kv)` | **无赋名**（内联实参） |

→ 抽取器必须同时覆盖「赋值形」与「内联实参形」；后两处正是 `ukl1/ukl2` 判决的依据。

### `with _no_grad():` 区域 [SRC]

**⚠ 更正评估文档**：`with _no_grad():` 实际在 **`indexer.py:214`**，不是 `:220`
（`:220` 是块内的 `npu_lightning_indexer(` 调用）。`grep -n "_no_grad" indexer.py` → `28`（import）、`214`。
块内产物（`:215-232`）：`q, k, weights, key_length, cmp_residual_k, topk_indices, index_scores`。

---

## 2. Task 1 交付：`cost_eval/opdag/fn_saves.py` + 两个审计测试 [RAN]

**新文件**
| 文件 | 内容 |
|---|---|
| `cost_eval/opdag/fn_saves.py` | 纯 AST 抽取器（~470 行含 docstring）：`scan_source` / `scan_tree` |
| `tests/test_opdag_fn_saves.py` | 22 测试 = 14 合成源单测（永远跑）+ 8 快照源真值断言（md5 门控） |
| `tests/test_source_truth_saves_audit.py` | 11 测试：手写 spec vs 源真值 + 字节影响 + detach 对账 |

`pytest tests -q` → **1559 passed**（基线 1526 + 33 新增，**既有测试一个未改**）。

### 2.1 抽取器设计要点

- 支持的 `save_for_backward` 形态：`plain_args`、`starred_comprehension`
  （`*[t for t in (<元组>) if t is not None]`）、`starred_seq`。
- **看不懂的形态一律进 `unresolved`**（`strict=True` 直接 `ValueError`）—— 与 Task 2 同一条纪律，
  绝不当成「无 saved 集」静默丢。
- `backward` 侧 `next(ctx.saved_tensors)` 消费序独立抽出，与 forward 名单做**同序自校验**
  （两处真源都逐名相等，见 §1）。
- 裸 ctx 属性给**两个判据**，不猜：
  - `tensor_candidate`（**上界**，纯句法）：`kind` 非 `predicate`/`const`；
  - `bwd_tensor_evidence`（**下界**，源证据）：`backward` 里该 attr（或绑它的局部名）被
    `mint.*`/`ops.*` 自由调用、`isinstance(x, DTensor)`、或张量方法（`.to_local()`/`.mul_()`…）
    消费。**纯算术不算证据**（标量也做算术）——刻意保守。
  实测判别力：7 个真张量全命中下界；`csa.py:123-128` 的 6 个 kernel 标量 + `ctx.loss_coeff`
  （只做算术）全部不命中。

### 2.2 源真值表（`experimental_attention_variant/`，`scan_tree` 实跑）[RAN]

**`save_for_backward` 名单**

| 类 | 定位符 | n | 名单（逐字、同序） |
|---|---|---|---|
| `FusedSparseFlashMla` | `csa.py:113` | 8 | query, ori_kv, cmp_kv, sparse_indices, cmp_residual, sinks, output, softmax_lse |
| `FusedSparseFlashMlaWithIndexerLoss` | `csa.py:224` | 11 | query, ori_kv, cmp_kv, sparse_indices, query_index, key_index, weights, cmp_residual, sinks, output, softmax_lse |

两处 `backward` 消费序（`csa.py:134-142` / `:251-262`）与上表**逐名同序**；条件消费项
`cmp_kv` / `sparse_indices`（`if ctx.has_cmp_kv` / `has_sparse_indices`）。

**裸 ctx 张量（绕过 `saved_tensors_hooks`）—— 下界 7 项 / 3 个类**

| 类 | 定位符 | attr | 反向侧证据 |
|---|---|---|---|
| `_IndexerLossAutoScaler` | `indexer.py:276` | `indexer_loss` | `isinstance(…, DTensor)`、`.to_local()`、`mint.ones_like(…)`（`:284-285`） |
| `_DSAIndexerFunction` | `dsa_indexer.py:55/56/57` | `q` / `k` / `weights` | `mint.zeros_like(ctx.q)` 等（`:64-66`） |
| `_DSAIndexerGradFunction` | `dsa_indexer_loss.py:70/71/72` | `d_query_index` / `d_key_index` / `d_weights` | `d_query_index.mul_(grad_scale)` 等（`:87-89`） |

> **新发现（评估文档未记）**：`_DSAIndexerFunction`（`dsa_indexer.py:42`）把 **q/k/weights 三个
> 大张量**整份挂在裸 `ctx.q/k/weights` 上，`_DSAIndexerGradFunction` 把**三份梯度**挂在裸
> `ctx.d_*` 上。两者都**不走** `save_for_backward` → `saved_tensors_hooks`（offload/重算释放）
> 看不见它们。这是 DSA（非 CSA）分支，本 yaml（`is_dsv4_hybrid`→CSA）不在关键路径，但对 DSA
> 配置是实打实的漏计来源。**未改任何数字**，只登记。

**非张量 ctx 属性（不得误判）**：`csa.py:111-112` / `:222-223` bool 谓词（`kind=predicate`）；
`csa.py:123-128` / `:237-245` kernel 标量（`softmax_scale/cmp_ratio/ori_mask_mode/cmp_mask_mode/
ori_win_left/ori_win_right/loss_coeff/layer_number/num_layers`，只作 kwarg 或算术）。

**detach 站点**：见 §1（6 处 `ops.stop_gradient`，4 赋值形 + 2 内联实参形）。

**`_no_grad` 区域**：`indexer.py:214-232`（`CSAIndexer.construct` fused 支），块内绑定
`q, k, weights, key_length, cmp_residual_k, topk_indices, index_scores` —— **整块 detach**。
（**更正评估文档 §3.1/§6.2 记的 `:220`**。）

### 2.3 差异清单 + 字节影响（站点配置 seq4096 / topk512 / 64 heads / v_head_dim 512 / b=1，tp=cp=ep=1）

字节数由 `cost_eval.shape_eval.resolve_tensor` **现算**（评估器自身口径），**不是**真机测量。

手写 fused r4 `sparse_attn.saves` 实跑 = **9 项 / 847.500 MiB**：
`q_hnorm_fp32 512.0` + `kv_a_out 4.0` + `compressed_kv 1.0` + `core_out 256.0` +
`idx_query 64.0` + `idx_key 1.0` + `idx_weights 0.5` + `cmp_residual 8.0` + `softmax_lse 1.0` MiB。

| # | 差异 | 源依据 | 原始字节 | **对 `activation_saves` 的净影响** |
|---|---|---|---|---|
| D1 | 手写缺 `sparse_indices` | `csa.py:228` 逐字 save；映射到手写 `topk_indices` | **8.000 MiB**（[B,S,512] int32） | **0** —— 同名张量已由上游 `indexer` op 声明 saved（`dsv4_hybrid.py:155/159`），而 `structure_mem.py:287` 的 `saves` 是**按名去重的字典**。liveness 侧同样 0：fwd 序 `indexer` 在 `sparse_attn` 之前 → 反向序 `indexer.bwd` 更晚 → 活跃区间本已覆盖，补声明不延长 |
| D2 | 手写缺 `sinks` | `csa.py:233` save 的是 `ops.cast(attn_sink, fp32)`（`csa.py:687`），一份**独立 fp32 激活复本**；手写只把 `attn_sink` 放进 `params`（权重） | **256 B** = 0.000244 MiB | **+256 B/层**（块对齐后 512 B）。量级可忽略，但它证明「saved set 要逐字读、不能靠归类」 |
| D3 | 手写**多**声明 `ukl1`/`ukl2`（unfused r4） | `csa.py:794-795` 两入参过 `ops.stop_gradient` → `indexer.py:350` `matmul*scale` → `:380` fp32 softmax，**链上无任何参数** → 不建 autograd 节点 → 反向无节点读它 | 各 **1024.000 MiB**（[1,4096,64,1024] fp32），合 **2048.000 MiB** | **−2048.000 MiB/r4 层**（= 该 op 全 saves 17664.000 MiB 的 **11.6%**）。**本轮不动** —— 会移动 unfused 锚点 `50187.9/43940.0/43407.1/47888.1` MiB |

**对照项（证明 D3 不是「凡 O(S²) 都不 saved」）**：`CSAIndexer` 自己的 `index_scores`
（`indexer.py:245` bmm → relu → ×weights → sum）经 `linear_wq_b` 参数（`indexer.py:177`）
携带梯度 → `detached=False`、照常 saved，实跑 **2048.000 MiB**（`indexer` op，unfused r4）。

### 2.4 台账（ratchet）而非「直接改 spec」

`tests/test_source_truth_saves_audit.py` 的 `KNOWN_GAPS` 是**双向棘轮**：
- 出现**未登记**的新差异 → `test_sparse_attn_saves_vs_source_truth` 打可读 diff 失败
  （「源说 X / spec 说 Y」+ 期望手写名 + 「改 saves 会移动已标定锚点，请显式决策」）；
- 台账项**被修掉**（spec 补上了）→ `test_known_gaps_ledger_is_still_accurate` 失败，
  迫使删台账项 → 不会留下过期注释。

名字映射 `ALIAS` 的每一条都钉在 `csa.py:689-707` 的 `.apply(...)` **实参位序**上（不是猜同义词），
并有 `test_alias_table_covers_every_source_name` 守住「映射漏项被误报成 spec 缺项」。

### 2.5 `TensorRef.detached` 接线判决

`detached` 字段（`model_spec.py:133`）**已存在**且已由 liveness 消费（`liveness/graph.py`），
手写侧 `ukl1/ukl2` 也**已**标注。故本轮**不新增接线**（无字节中性的增量可做），改为**加对账**：
- `test_detached_flags_are_backed_by_source_detach_sites`：手写每个 `detached=True` 都必须有源侧
  detach 站点背书（实测 `csa.py:794`/`:795` 两条内联 `ops.stop_gradient`，
  `enclosing_call == "self.unfused_indexer_loss"`）；
- `test_no_other_saves_are_silently_detached`：除 `ukl1/ukl2`，dsv4 attn 全 ratio × fused/unfused
  的 saves 不得出现 `detached` —— 防该 flag 悄悄扩散。

把 detach 站点接成 `Detach` 节点 / `FREE_CALL_MAP` 表项（评估文档 P1#11）**属于路线 B**，
不在本轮（它会改 walker 的图形状，不是字节中性）。

---

（Task 2 里程碑将追加到本文件）
