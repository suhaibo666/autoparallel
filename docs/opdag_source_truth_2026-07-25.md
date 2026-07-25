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

## 3. Task 2 交付：静默丢弃全部 fail-loud / 可断言化 [RAN]

`pytest tests -q` → **1583 passed**（基线 1526 + Task 1 的 33 + Task 2 的 24，
**既有测试一个未改**）。新测试文件 `tests/test_opdag_drop_diagnostics.py`（24 测试）。

### 3.1 新增的不变量（四条）

| # | 机制 | 位置 | 默认行为 |
|---|---|---|---|
| ① | **0 节点硬门（恒开，不受 `strict` 影响）** | `construct_walker._run_walker` | construct 有非平凡 body 却抽出 0 节点 → `ExtractionDroppedError`。`return x`/`pass`（恒等/空 Cell）的合法 0 节点由 `_body_is_trivial` 豁免 |
| ② | **结构化计数 `dag.diagnostics`** | `schema.OpDAG.diagnostics` + `construct_walker.DIAG_KINDS` | 逐条带 `src=file:line` + 原文。默认只记录 → 既有路径逐字节不变 |
| ③ | **`strict=True`** | `walk_construct` / `walk_construct_meta` / `extract_cell` | 任一诊断非空即 `ExtractionDroppedError`（默认 False） |
| ④ | **`assert_extraction_clean(dag, allow=(), check_opaque=False)`** | `construct_walker` | 消费方一行门（路线 B `to_resolved.py` 适配器要用的就是这道） |

`ExtractionDroppedError` 继承 `ValueError`：①既有 `pytest.raises(ValueError)` 惯用法照旧；
②`crosscheck._check_segment` 的 `except Exception` 会把它漏斗进 `extraction_failures` →
`ok=False`、strict 下 raise —— 即「已声明覆盖族抽取失败 ≠ 合法无对应」那条既有纪律**自动生效**。

### 3.2 转换的静默丢弃站点清单

| 站点 | 此前 | 现在 |
|---|---|---|
| `walk_stmt` 无 `else` 分支（`construct_walker.py:251-276`） | `With`/`For`/`While`/`Try`/`AugAssign`/`AnnAssign`/`Assert`/`Delete`/`Match` **静默 return**（实测 0 nodes / 0 opaque） | 逐条进 `dropped_stmts`；`with _no_grad():` 另带「整块 detach」note（刻意**不**内联走查——会造出一批无 detach 语义的假节点）；`Pass/Break/Continue/Import/嵌套 def` 明确列为「真·无 op 语义」不记账 |
| `walk_stmt` 的 `ast.Expr` 非 Call 分支 | 静默丢 | 记账（docstring `Expr(Constant)` 豁免） |
| `_handle_assign` 无 `else`（`:329-344`） | `BinOp`/`Compare`/`Subscript`/`Name`/`Attribute`/`ListComp` RHS **静默丢**（评估文档 §6.2 实测 34 处，含 `q_hnorm_fp32`/`attention_scores`/`score_f32`/两个 mask/三处切片） | 进 `dropped_assigns`（带 targets + RHS 形态 + 原文），**并**把目标名记进 `unregistered_targets` |
| `_handle_call` 终端 fallthrough（`:411-413`） | 只记调用文本，**赋值目标从未进 SSA/producer** → 下游拿占位 ref 且**无边**，与合法形参操作数长得一模一样 | 额外记 `unregistered_targets`（`cause="opaque_call"`） |
| `_emit` 占位 ref 分支（`:889-890`） | 未知操作数静默变 `name:?:bf16` —— 这是让**所有上游静默丢弃变得不可见**的汇聚点 | 非 construct 形参、非帧内合成名（`__i<n>`）的落 `unresolved_operands` |
| `init_binder` 只认类实例化（`:95-96`） | pynative 的裸别名（`self.reshape = mint.reshape`）**完全不可见**（评估文档 §2.4：39 原语 / 196 调用点） | 新增 `unbound_aliases(src, cls, file)` 枚举（链首限 `mint/ops/F/mindspore/P/nn`，已被 `_CLS2OP` 绑住的排除）；`extractor` 按 `_init_classes`（derived→base）逐类扫并入 `unbound_aliases` 诊断 |
| 子 Cell 边界 | 子 walker 的诊断不上浮 → 子里丢一大块会被洗白 | `SubExtract.diagnostics` 逐类并入父（镜像既有 `opaque_calls` 传播） |
| `crosscheck._split_decoder` 的 `h1` 探针（`:436-445`） | mHC 下残差名为 `h1_xn`（`residual.py:62` `{name}_xn`）→ 探针失配 → `(None,None)` → 三族 census 全跳过 | 按 `h1` / `h1_*` 前缀匹配（`h2*` 明确不认，非 attn/ffn 边界） |
| `crosscheck` 的空判决 | `covered=0` 照样 `ok=True` | 新增 `attn_layers`（**探针无关**分母：结构性含 `ATTENTION` op 的层段）+ `validated_attn_layers` + `coverage_findings`；含注意力的层段一个都没被任何一族校验到 → `ok=False`、strict 下 raise |
| `init_binder.bind_init` 的 `ast.walk(None)` | 类自身无 `__init__` 时 **AttributeError**（潜伏 bug，被本轮调用暴露） | `init is None` → 返回 `{}`（调用方本就按 MRO 逐类调） |

### 3.3 空判决门为何用「探针无关分母」

第一版规则「有 decoder body 却 `covered=0` → 非绿」**打破了既有契约**
（`test_legitimate_no_correspondence_does_not_trip_strict`：全 GQA/dense 模型**合法**无 MLA/MoE
对应，模块 docstring 的 F6 契约）。且 `decoder_bodies` 来自 `_split_decoder`，是**循环论证**
（同一个失配的探针既定义分母又定义分子——原病症下 `decoder_bodies` 恰恰是 0，规则根本不会触发）。

最终规则：
- **分母** `attn_layers` = 结构性含注意力 op 的层段（用既有 `hand_category(op) == ATTENTION`，
  不引入新结构猜测）—— 探针无关，不可能算错。
- **分子** = 这些层里被**任何一族**（MLA/MoE delta-census 的 `covered`，或 layer_norms 族的
  `layer_norm_checked`）真校验过的。
- 分子为 0 → `coverage_findings` → `ok=False`。缺源（`available=False`）仍 True（既有契约）。

四种情形核对（均有测试）：原病症形状（`covered=0`、`layer_norm_checked=['lm_head']`、
`decoder_bodies=0`）→ **触发**；修复后 DSv4 → 不触发；全 GQA/dense → 不触发；缺源/无注意力层 → 不触发。

### 3.4 验收：`CSAIndexer` 不再 `OK -> 0 nodes` [RAN]

同一段附录 A 复现脚本，`531bcdc` 时的输出 vs 现在：

```
【此前】CSAIndexer   OK  -> 0 nodes   <== 假成功：空图
【现在】CSAIndexer   ExtractionDroppedError
```

完整报错（实跑，权威快照）：

```
construct 走查产出 **0 个节点**,但 CSAIndexer.construct 的 body 非平凡（indexer.py）
—— 整段被丢弃,这**不是**成功。
  诊断计数: dropped_stmts=2, dropped_assigns=0, unregistered_targets=0,
            unresolved_operands=0, unbound_aliases=0, opaque_calls=0
  [dropped_stmts] 2 条:
    - indexer.py:211 Delete: del actual_seq_qlen, actual_seq_klen
        (del 无 op 语义,但是 liveness 的显式释放点(未建模))
    - indexer.py:214 With: with _no_grad():  …
        (`_no_grad()` = 整块 detach:块内产物应标 detached,而非当普通节点发射
         ——故此处刻意**不**内联走查(会造出一批无 detach 语义的假节点))
  —— 请为上列语句/RHS 形态补处理器(评估文档 §10 P0#1/#4/#5)。
```

四个 dsv4 Cell 现在全部**大声**失败（无一个假成功）：

| Cell | 现在 |
|---|---|
| `CSAIndexer` | `ExtractionDroppedError`：0 节点 + `indexer.py:214` `with _no_grad():` |
| `CompressedSparseAttention` | `ExtractionDroppedError`（递归进 `CSAIndexer` 时先命中；此前报 `compressor.py:190`——两者都是 fail-loud，只是现在更早） |
| `Compressor` | `ValueError`：`compressor.py:190` `sq < ratio` 不可判定（不变） |
| `DSv4HybridSelfAttention` | `ValueError`：`deepseek_v4_hybrid_attention.py:233` `self.shape(...)` 未绑定（不变） |

### 3.5 副产物：`h1_*` 探针修复让 DSv4 crosscheck 从空判决变成**真**覆盖 [RAN]

`build_llm_spec(deepseek_v4(6))` + `validate_against_opdag`：

| | 评估文档 §1 实测（修复前） | 现在 |
|---|---|---|
| `covered` | **空** | 3 × `moe_experts`（r4/r128/r0_moe 三层的专家 grouped-GEMM 核逐类别 delta） |
| `layer_norm_checked` | `['lm_head']` | 6 项：4 个 dsv4 decoder body + `mtp` + `lm_head` |
| `decoder_bodies` | （h1 探针失配 → 0） | 4（全部命中 `h1_xn`） |
| `ok` | `True`（**空判决**） | `True`（**真**绿：`coverage_findings == []`、`findings == []`） |

即：连 `ln1`/`ln2` 名册校验都没跑到的那 4 个 decoder body，现在被 layer_norms 族逐层校验了。

### 3.6 副产物：诊断在**抽得通**的 DSv3 MLA 路径上也记到东西 [RAN]

```
MLPInterleaved      8 nodes | 全 0
MLASelfAttention   26 nodes | dropped_assigns=2 unregistered_targets=2 unresolved_operands=1
   [dropped_assigns] multi_latent_attention.py:232 Attribute:  ori_dtype = x.dtype
   [dropped_assigns] multi_latent_attention.py:258 Subscript:  head_dim = query.shape[-1]
   [unresolved_operands] multi_latent_attention.py:307 Cast:   Cast(… ori_dtype …)
```

两条都是**标量/dtype、不是张量** → 对字节无影响；但它们此前完全不可见，且第三条说明
`:307` 那个 `Cast` 的 dtype 操作数是未解析的。这证明计数器不只对失败路径生效。

---

## 4. 留作原样的东西（诚实清单）

| # | 留作原样 | 为什么 |
|---|---|---|
| 1 | **`saves` 名册的三处差异（D1/D2/D3）一个数字都没改** | 任务明确要求：改 `saves` 会移动已标定锚点（unfused `50187.9/43940.0/43407.1/47888.1`、fused `24153.3/14641.7/14097.7/23508.0` MiB），需单独决策。已登记进 `KNOWN_GAPS` 双向棘轮台账 + 字节影响 |
| 2 | **`ast.With` 仍不内联走查**（只记账 + 0 节点门） | `with _no_grad():` 语义是**整块 detach**。走进去发射普通节点会造出一批「看起来梯度可达」的假节点 —— 那是比静默丢更危险的错法（评估文档 §11 纪律）。真正的 `With` 处理器 + 块内 `detached` 标注是路线 B **P0#4**，需与 `TensorRef.detached` / `FREE_CALL_MAP` 一并做 |
| 3 | **`ops.stop_gradient` 仍不建 `Detach` 节点 / 不保边** | 同上，属路线 B **P1#11**（会改 walker 图形状，非字节中性）。本轮只做「detach 站点逐字抽出 + 与手写 `detached` flag 对账」 |
| 4 | **`strict` 默认 False** | 实测既有 DSv3 MLA 路径就有 2 条 `dropped_assigns`（标量/dtype，无害）。默认 True 会让既有 26 节点抽取直接失败 = 破坏 1526 基线。故：**0 节点门恒开**（那一条不可能是无害的），细粒度诊断默认只记录、由消费方用 `assert_extraction_clean(..., allow=...)` 显式决定 |
| 5 | **`_emit` 里 `self.weight` 这类 `Attribute` 操作数仍被丢出 `ins`**（`:881-882`） | 这是 `is_weight` / `op.params` 缺失那件事的同一个根（评估文档 §9，M 级）；补它要同时引入权重操作数概念，属路线 B **P1#14** |
| 6 | **`_handle_chained_call` 的未知链式方法仍当纯视图**（`:461-469`） | 未在本轮 dsv4 链上实测触发；改它要先建 `_VIEW_METHODS` 白名单外的语义表（PIN 侧配套），属路线 B P1#12 |
| 7 | **`tests/conftest.py` 的 `MF_ROOT` 默认仍指非权威树** | 既有 fixture（`mlp_dag`/`mla_dag`）与全部既有 opdag 测试都依赖它，改默认会牵动一大片。本轮新测试**自带 md5 门控**（`authoritative_variant_dir()`），非权威树一律 skip，绝不对着另一个 commit 断言 |
| 8 | **`_DSAIndexerFunction` / `_DSAIndexerGradFunction` 的裸 ctx 张量只登记、未进任何字节口径** | DSA（非 CSA）分支，本 yaml（`is_dsv4_hybrid`→CSA）不在关键路径；进字节口径需要 DSA 配置的形状/锚点，本轮无据可依，不杜撰 |
| 9 | **`del x` 记进 `dropped_stmts` 而非单列 liveness 类** | `del` 无 op 语义但是 liveness 的显式释放点。单列一类需要 liveness 侧配套消费，本轮只保证它可见 |

