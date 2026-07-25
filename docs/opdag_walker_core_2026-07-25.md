# opdag walker 核心补全 —— 让纯 AST 抽取器真正走通目标模型（2026-07-25）

> **性质**:路线 B 的 P0#2/#3/#4/#5 + P1#11 五项(见 `docs/opdag_coverage_assessment_2026-07-25.md` §10)。
> 目标:让 `cost_eval/opdag/` 这条纯 AST 流水线**真正走查** DSv4-Flash 真机跑的 `pynative/` 侧
> 4 个 Cell,而不是只在 `parallel_core/training_graph/` 上抽得通。
>
> **权威源**:`E:\97-codes\torch_parallel\mf-src-167\mindformers\`。**不用**
> `E:\97-codes\torch_parallel\mindformers`(不同 commit)。
>
> 记法:**[RAN]** = 实跑所见(附命令 + 实际输出);**[SRC]** = 逐字读源(带 file:line)。
> 本文按任务的 5 个编号项**逐项 checkpoint**,每项落地即追加。

---

## 0. 环境 / 基线校验 [RAN]

```bash
cd /e/97-codes/torch_parallel/mf-src-167 && md5sum \
  mindformers/pynative/transformers/experimental_attention_variant/{csa.py,indexer.py,compressor.py} \
  && wc -l mindformers/pynative/transformers/experimental_attention_variant/*.py
```
```
81673be3ad3cdd2191e28dd000a13f0e *csa.py          <- 与任务给定一致
8e18fee21c33507dd629c89f401986e6 *indexer.py      <- 与任务给定一致
49ec62ddc358f913438cf1c14cf99f7c *compressor.py
   831 csa.py   409 indexer.py   244 compressor.py   300 deepseek_v4_hybrid_attention.py
```
→ 快照校验通过(831 / 409 / 244 行,md5 与任务描述逐字相同)。

```bash
cd /e/97-codes/torch_parallel/pynative-cost-evaluator && git log --oneline -1 && python -m pytest tests -q
```
```
7b9aa86 fix(opdag): 静默丢弃全部 fail-loud / 可断言化 —— 路线 B 的 P0#1 门
1583 passed, 268 warnings in 24.32s
```
→ 基线 **1583 passed** 复现,HEAD = `7b9aa86`。

---

## 0.1 起步即发现的硬冲突:三组既有测试**钉住的正是本任务要拆掉的病症** [RAN]

任务要求「All 1583 pre-existing tests pass」。**这在字面上不可能同时成立**,因为
`7b9aa86` 新增的 `tests/test_opdag_drop_diagnostics.py` 里有一批测试断言的是
**「这些形态被丢弃」**,而本任务要求的恰恰是**「这些形态必须被处理」**:

| 测试 | 现断言 | 与本任务哪一项冲突 |
|---|---|---|
| `test_unhandled_assign_rhs_recorded_with_targets_and_kind`(:137) | 合成源里 `BinOp/Compare/Subscript/Name/Attribute` 五条 RHS **全进 `dropped_assigns`** | **第 4 项**(RHS 形态必须建节点) |
| `test_dropped_assign_target_is_flagged_unregistered`(:150) | 上述目标全进 `unregistered_targets`,下游拿占位 ref | 第 4 项 |
| `test_strict_reports_every_kind_with_src`(:161) | strict 下报 `c.py:5/6/7`(那三条 RHS) | 第 4 项 |
| `test_diagnostics_summary_counts_every_kind`(:225) | `dropped_assigns == 5` | 第 4 项 |
| `test_assert_extraction_clean_raises_with_readable_listing`(:233) / `..._allows_explicit_waivers`(:241) | 同一合成源必须脏 | 第 4 项 |
| `test_csaindexer_no_longer_reports_ok_zero_nodes`(:412) | 真源 `CSAIndexer` 必须 **raise**(`indexer.py:214` With) | **第 3 项**(With + no-grad 语义) |
| `test_working_dsv3_mla_path_also_surfaces_its_drops`(:430) | DSv3 MLA 真路径 `dropped_assigns >= 2`(`ori_dtype = x.dtype` / `head_dim = query.shape[-1]`) | 第 4 项 |

**处置(不掩盖)**:这些测试的**机制断言**(「看不懂的东西必须记账、必须能 fail-loud」)全部保留
——只把**举例用的形态**从「现已支持」换成「真的还不支持」(见 §6 逐条台账)。凡改动的既有测试
在 §6 逐条列出「原断言 / 新断言 / 为什么这是同一条不变量」。**绝不**删除任何一条不变量。

---

## 1. 第 1 项:裸函数别名绑定(P0#2)—— 显式表 + 未知原语 fail-loud [RAN]

### 1.1 先枚举,再建表(不猜词表)

纯 AST 扫权威快照的 6 个类 / 5 个文件(`scratchpad/enum_prims.py`),实测:

| 口径 | 数量 |
|---|---|
| `self.<attr> = <ns>.<fn>` 裸别名**赋值** | **56** 处(8 个类) |
| 其中**不同的别名目标** | **30** 个 |
| 这些别名的 `self.X(...)` **调用点** | **106** 处 |
| `<ns>.<fn>(...)` **自由调用**的不同点号路径 | **33** 个 |
| 自由调用**调用点** | **116** 处 |
| 两者并集(需要表项的不同原语) | **48** 个 |

评估文档 §2.4 记的「39 原语 / 196 调用点」是另一种口径(别名与自由调用合并计)。两个口径
都指向同一件事:**这不是"少几个表项",是整条通路缺失**。

### 1.2 新文件 `cost_eval/opdag/primitives.py`

**一张显式、可评审的表** `PRIMITIVES: 点号路径 -> (op 类型, attrs)`,每组带定位符注释与
「反向要读什么」的判据(教科书 VJP)。三条纪律写在模块 docstring:
① 只收实测触发点;② **未知原语 fail-loud,绝不归类**;③ 归类按 VJP 事实。

归类决定字节的两个要害(各有测试):
* `Compare`(比较 / `isfinite` / 逻辑)→ **不可微,反向什么都不读**。若误归非线性
  `Elementwise`,`csa.py:779/810` 那两个 O(S·S/r) / O(S·topk) bool mask 就会被错声明为 saved。
* `tensor <op> 标量` → **梯度线性,不存激活**;只有 `tensor <op> tensor` 的 mul/div 才存两操作数。
  这条直接决定 `indexer.py:350` 的 `matmul(q,k) * softmax_scale` 是否把那个 O(S·S/r) fp32
  张量算进 saves(`_binop_kind` 注释 + `test_the_five_named_tensors_now_have_nodes` 钉住)。

`bind_init` 新增三种形态(原两种保留,既有 DSv3 路径逐字不变):
形态 3 **裸别名**;形态 4a **三元构造**(`Hadamard(d) if rotate else IdentityOp()`,
compressor.py:129);形态 4b **工厂方法**(`self._build_rotary_pos_emb(...)`,deepseek_v4:80
—— 两条 return 都是 RoPE 频率表,同一 op 类型才绑);形态 5 **构造函数注入的子模块**
(父在构造点传 `rotary_pos_emb=self.rotary_pos_emb`,indexer.py:131 / csa.py:602/614)。

**fail-loud 面**:能查表的绑上;查不到的留在 `unbound_aliases()` 清单里,**一旦被调用**就抛
`UnknownPrimitiveError`,报错指名道姓说"加哪条表项"。该门在真源上抓到一条:`ops.rms_norm`
(deepseek_v4:158 绑了但本版本未调用)—— 已补表项(`Norm{rms}`)。

### 1.3 并行 agent 的约束:`op.type` 拼写是承载语义的

`OpNode.op`(反向语义词表,`PIN` 的键)与 `model_spec.OpType`(成本模型词表,`structure_mem`
分支 + `RecomputeSpec.op_matches` 子串匹配)**是两套词表**,不能混用:前者必须区分
「`Compare` 不读」与「非线性 `Elementwise` 读两操作数」。故新增
`primitives.OPDAG_OP_TO_OPTYPE` + `to_op_type()`:**显式映射,未映射即 fail-loud**;
未来的 `to_resolved.py` 必须走它,不许现场臆造字符串。

**三个 opdag op 类型没有 `OpType` 对应物 —— 报告为建议项,未擅自新增枚举成员**:
`Detach` / `Compare` / `Constant`,都是「产出张量但不参与梯度」;硬塞进 `elementwise`
会被 `RecomputeSpec.op_matches` 的子串匹配连带选进重算集 = **静默错**。

---

## 2. 第 2 项:跨文件类解析 / MRO(P0#3)[RAN]

新文件 `cost_eval/opdag/module_index.py`:`ClassIndex` 顺 `from ... import ...` / `import ...`
把基类解析到**另一个文件**,`mro(cls, rel)` 给 derived→base 的跨文件 MRO。
`extractor._mro_units` 与 `_Walker._lookup_method` 都改成「有 index 就跨文件,没有就退化为
同文件 BFS」—— **既有 DSv3 路径行为逐字不变**(`cross_file=False` 是默认)。

三处配套(否则跨文件只是半通):
1. `bind_init` / `_named_module_binds` / `unbound_aliases` 必须用**该类自己文件**的源
   (`usrc`/`ufile`),不是派生类的源。首版漏了 `unbound_aliases` 这处,实测报
   `源码里找不到 class MultiLatentAttention` —— 已修。
2. `eval_init_dims(..., mro_units=)`:`__init__` 静态求值也要跨文件。
3. 内联跨文件方法时把 `src_file` 临时切到定义它的文件,否则节点的 `file:line` 是**错的**。

**断言解对了类**(评估文档 §2.1 的"静默解错模型"):`test_opdag_walker_core.py` 的 `targets`
fixture 断言 `HyperConnectionTransformerLayer` / `DSv4HybridSelfAttention` /
`CompressedSparseAttention` 三层都对,再开抽。

### 2.1 一个**主动拒绝**的设计(诚实记录)

评估文档 P0#6 建议「把 `init_dims` 静态求出的 `__init__` 布尔喂给 `walker_flags`」。
**本轮拒绝**,理由是实测反例:`compress_ratio` 的 `__init__` 缺省是 `0`(csa.py:556),
`init_dims._param_defaults` 会据此求出 `self.enable_compress = False`(csa.py:594)——
**与真机(ratio=4 → True)相反**。这正是「静默给错比 fail-loud 更危险」要防的东西。
故 `enable_compress` / `enable_indexer` / `overlap` / `rotate` 一律**显式注入**(与评估文档
附录 A 的既有约定一致),每个注入点在测试里带定位符注释。

---

## 3. 第 3 项:`ast.With` 的块级语义(P0#4)[RAN]

`7b9aa86` 刻意不走查 `With` 的理由是对的(走进去发普通节点 = 造出一批「看起来梯度可达」的
假节点,比丢更危险)。本轮实现的是**正解**:走查块体 + 带块级语义。

两张表(`_WITH_NO_GRAD` / `_WITH_TRANSPARENT`),**其余一切 `with` fail-loud**:

| 分类 | 语义 | 真源触发点 |
|---|---|---|
| `NO_GRAD` | 整块不建 autograd 图 → 块内产物**全部标 `detached`** | `_no_grad()` @ `indexer.py:214`(区域 `:214-232`) |
| `TRANSPARENT` | 只切换派发/布局模式,数学恒等、梯度可达性不变 → 照常走查、不打标 | `SkipDTensorDispatch()` @ `multi_latent_attention.py:244/288`、`flash_attention.py:311` |
| 其它 | **fail-loud**(不猜它改不改梯度可达性) | 测试 `test_unknown_with_context_is_fail_loud` |

实测 `grep -rn "with " pynative/transformers/`:dsv4 链上**只有** `indexer.py:214` 这一处
`with`,且正是 `_no_grad`。

验收(`test_no_grad_block_products_are_detached_not_fabricated`):
```
CSAIndexer(fused)  7 nodes / 6 edges | diag: clean | detached=6
  每个节点的 src 行号都落在 [214, 232] 内,且 attrs.detached is True、no_grad_region is True
  detached 名册 >= {q, k, weights, topk_indices, index_scores}
```
`npu_lightning_indexer`(`indexer.py:220`)发成 `Kernel` 节点:只因**可证明**它在 no-grad 区里
(无反向)才写 `attrs["saved_ins_idx"]=[]`;不在 no-grad 区的内核**缺该 attr → `derive_saves`
fail-loud**(`bprop_rules` 的 `from_attrs` 分支)—— 不给"猜它没有 saved 集"留后门。

---

## 4. 第 4 项:赋值 RHS 形态(P0#5)[RAN]

`_handle_assign` 新增 `BinOp / UnaryOp / Compare / BoolOp / Subscript / Name / Attribute /
Constant / Tuple` 分支 + `AugAssign` / `Delete` / `_ = ...`(显式丢弃)。核心是**判据层**
`_classify()`:把表达式判成 张量 / 标量 / dtype 记号 / None / **权重** / 判不出。

**为什么判据层是关键**:同一批 RHS 形态里既住着这次要修的张量,也住着字节中性的记账 ——
建错任一边都是错:

| 必须建节点(张量) | 必须**不**建节点(标量/dtype) |
|---|---|
| `q = q * rsqrt(mean(q*q,…)+eps)` @ deepseek_v4:245 | `ori_dtype = x.dtype` @ :232 |
| `attention_scores = matmul(q,k) * scale` @ indexer:350 | `head_dim = query.shape[-1]` @ mla:258 |
| `score_f32 = score.astype(fp32) + ape` @ compressor:209 | `ratio = self.compress_ratio` @ compressor:188 |
| `future = cm >= positions // ratio` @ csa:779 | `cutoff = (sq // ratio) * ratio` @ compressor:196 |
| `valid = topk_… < unsqueeze(…)` @ csa:810 | `sq, b, n, d = query.shape` @ csa:466 |
| `kv_flat[flat_indices]` @ csa:485(真 gather) | `d = self.query_projection_size // o_groups` @ deepseek_v4:276 |

判据全部来自**源侧事实**,不是猜:
* `self.<attr>` 的种类由 `init_dims` 静态求值 `__init__` 得(新增 `InitDims.self_kinds`:
  `Parameter(...)` → `param`;`build_module`/类实例化 → `module`;其余 → `scalar`);
* `self.config.<x>` 一定是标量(config 对象里没有张量);
* `<x>.shape[i]` 是**轴长**:值未知但 `>= 1`(新哨兵 `_POSITIVE_DIM` —— **这是张量语义的
  公理,不是编造的尺寸**)。它只让「`n_compressed > 0`」(csa.py:762)这类**只问非空**的条件
  可判定;「`sq < ratio`」(compressor.py:190)要真值,仍 `_UNDECIDED` → fail-loud,除非
  调用方给 `input_axes` 轴种子(与 `infer_shapes(dag, input_shapes)` **同一套契约**)。

### 4.1 `_emit` 里 `self.weight` 这类 `Attribute` 操作数(任务点名的那条)

**处理了,不是延后**,按并行 agent 的 W2/W3 契约:权重(`Parameter`)**不进 `ins`**、
**不是任何节点的 out** → 因此**永不进 `saves`**;改为单列 `OpNode.attrs["param_operands"]`
+ `OpDAG.param_operands`(带 `file:line` 出现点),使其**可见**而非静默丢。
新增 `param_aliases`:`attn_sink = self.attn_sink`(csa.py:683)这类局部名也按权重路由。
验收 `test_weights_never_enter_ins_or_saves`:实测 `{attn_sink, ape, linear_o_group_proj}`
属于 `param_operands`,与 `saves` 交集 = 空,与全体节点 out 交集 = 空。
**留给 P1#14**:`is_expert` / `dim0` / `pin_under_recompute` 三个属性与完整 `ResolvedOp.params`
建模 —— 那是 `to_resolved.py` 的活(本轮非目标)。

### 4.2 一并修掉的 dtype 假值(此前无人可见)

* `self.cast(y, ori_dtype)`(mla:307):`_dtype_name` 对裸 Name 返回**变量名** "ori_dtype"
  = 一个**假 dtype**。现查 dtype 环境得真 dtype。
* `ops.cast(invalid_mask, scores.dtype)`(csa:502):同理返回字面串 "dtype"。现解析 `<x>.dtype`。
* `_emit` 新增「按已解析的 `ins` 推产出 dtype」(实参物化后才知道)—— 否则
  `q_bm = reshape(permute(q_fp32,…))`(csa:494)会把 fp32 中间量错记成 bf16。

---

## 5. 第 5 项:`ops.stop_gradient` → 真 `Detach` 节点(P1#11)[RAN]

评估文档 §5 的三条结论逐条反转:**建节点 + 保边 + 打标**。

| | `7b9aa86` | 现在 |
|---|---|---|
| 节点 | 无(只进 `opaque_calls`) | `Detach` 节点(`PIN["Detach"] = {"inputs": []}`,梯度到此为止) |
| 边 | **静默切断**(赋值目标从未进 SSA,下游拿占位 ref) | 保边:producer→Detach→consumer 都在 `dag.edges` |
| 标记 | 无 | `attrs["detached"]=True` + 名字进 `OpDAG.detached` |

真源 6 处全覆盖(`test_all_six_real_stop_gradient_sites_become_detach_nodes`):
```
fused  支:{csa.py:665, csa.py:666}                            (赋值形)
unfused 支:{csa.py:764, csa.py:765, csa.py:794, csa.py:795}    (后两处是**内联实参形**,无赋名)
```
内联实参形还需额外一处修:`_inline_subcell` 的形参绑定此前对**非 Name 实参**直接给占位 ref
且不连边 —— 正是 `self.unfused_indexer_loss(..., ops.stop_gradient(query), ...)`(csa:794-795,
`ukl1`/`ukl2` 判决的依据)那两处。现改为先物化成节点再按 Name 共享 SSA。

**与 `model_spec.TensorRef.detached` 对齐,且刻意不做传递闭包**:
「detach 的下游是否仍梯度可达」取决于链上有没有**参数**(有 → 仍需 save,如 `index_scores`
经 `linear_wq_b` 携梯度;无 → 不需要,如 `ukl1/ukl2`)。参数操作数建模是 P1#14,故 walker
只给 **detach 边界事实**,闭包留给消费方。**本轮未改 `cost_eval/model_spec.py` 一个字节**
(`detached` 字段早已存在并被 `liveness/graph.py` 消费)。

---

## 6. 验收 [RAN]

### 6.1 四个 dsv4_hybrid Cell x 两条 A/B 配置:零诊断

```bash
python scratchpad/probe_dsv4_walk.py both
```
```
### [FUSED] ratio=4
Compressor                   OK   29 nodes /  31 edges | diag: clean | opaque=0 | detached=0 params=2
CSAIndexer                   OK    7 nodes /   6 edges | diag: clean | opaque=0 | detached=6 params=0
CompressedSparseAttention    OK   90 nodes / 104 edges | diag: clean | opaque=0 | detached=8 params=3
DSv4HybridSelfAttention      OK  123 nodes / 146 edges | diag: clean | opaque=0 | detached=8 params=5

### [UNFUSED] ratio=4
Compressor                   OK   29 nodes /  31 edges | diag: clean | opaque=0 | detached=0 params=2
CSAIndexer                   OK   14 nodes /  13 edges | diag: clean | opaque=0 | detached=0 params=0
CompressedSparseAttention    OK  207 nodes / 240 edges | diag: clean | opaque=1 | detached=4 params=3
DSv4HybridSelfAttention      OK  240 nodes / 281 edges | diag: clean | opaque=1 | detached=4 params=5
```
`diag: clean` = 五类诊断(`dropped_stmts` / `dropped_assigns` / `unregistered_targets` /
`unresolved_operands` / `unbound_aliases`)**全为 0**,`assert_extraction_clean(dag)` 放行,
`strict=True` 亦放行(`test_strict_true_is_clean_on_the_dsv4_path`,8 个参数化组合)。

**对照 `f346b33`(改动前)同一段脚本**:
```
Compressor                   FAIL ValueError: `sq < ratio` 无法由 config 判定（compressor.py:190）
CSAIndexer                   FAIL ExtractionDroppedError: 产出 **0 个节点**（indexer.py 整段被丢弃）
CompressedSparseAttention    FAIL ExtractionDroppedError: 同上（递归进 CSAIndexer 时命中）
DSv4HybridSelfAttention      FAIL ValueError: 调用了未绑定的 self.shape(...)（deepseek_v4:233）
```

### 6.2 唯一残留 opaque(**显式允许清单**,给字节理由)

| 定位符 | 表达式 | 为什么字节中性 |
|---|---|---|
| `csa.py:799` | `save_to_indexer_losses_tracker(indexer_loss, layer_number, num_layers)` | **纯宿主副作用**:把逐层 indexer loss(**标量**)累加进模块级 dict(`utils.py:41-54`),无被消费的返回值、不产张量、不进任何 op 的 ins/out。只在 unfused 支出现(fused 支的等价逻辑在 `_Function.backward` 里,不在前向图上) |

清单由调用方 `host_call_allow=` 显式给出,walker 记 `opaque_calls` 且带
`kind="host_side_effect"`;`test_all_four_dsv4_cells_extract_clean` 逐条断言
`kind == "host_side_effect"` 且表达式含 `tracker` —— **别的 opaque 一律不许出现**。

### 6.3 `unfused_compressed_sparse_attn` —— 从 1 个 op 变成 **45 个节点的真链**

手写 spec 把 `csa.py:464-533` 塌成 **ONE** op + 一张扁平 saves 名单;源侧它是一条真算子链。
实测抽出 **45 个节点**(区间 `csa.py:472-524`;`test_unfused_csa_chain_is_a_real_op_chain_not_one_op`
断言 `>= 40` 且结构关键点齐全):

```
#163 View        csa.py:472  kv_full -> kv_t                      (permute)
#164 Elementwise csa.py:474  clamp(topk_indices, min=0)
#165 Cast        csa.py:474  -> safe_indices : int64
#166 View        csa.py:482  kv_t -> kv_flat                      (reshape [b*sk, d])
#167 Constant    csa.py:483  arange(b) ...
#168 Elementwise csa.py:483  ... * sk -> batch_offset
#169 Elementwise csa.py:484  safe_indices + batch_offset
#170 View        csa.py:484  -> flat_indices                      SAVED(被 #171 的 gather pin)
#171 IndexSelect csa.py:485  kv_flat[flat_indices]   <-- **真正的 gather**
#172 View        csa.py:485  -> kv_gathered
#173/#174        csa.py:489  permute + cast  query -> q : fp32
#175 Cast        csa.py:490  kv_gathered -> kv_g : fp32
#176-#179 View x4 csa.py:494/495  q_bm / kv_bm                    SAVED x2
#180 BMM         csa.py:496  q_bm @ kv_bm            <-- **第 1 次 BMM(scores)**
#181-#183        csa.py:496/497  -> scores,`* softmax_scale`(线性) SAVED
#184 Compare     csa.py:501  topk_indices < 0        <-- **O(S·topk) bool mask**
#185 View        csa.py:501  -> invalid_mask
#186-#188        csa.py:502  cast + `* -1e30` + `scores + ...`     SAVED
#189/#190        csa.py:506  reshape(attn_sink)+cast -> sink : fp32 SAVED(权重经 param_operands)
#191 Elementwise csa.py:508  max(scores, dim=-1) -> scores_max     SAVED
#192 Elementwise csa.py:509  maximum(scores_max, sink)             SAVED
#193/#194        csa.py:511  exp(scores - scores_max) -> exp_scores SAVED
#195/#196        csa.py:512  exp(sink - scores_max) -> exp_sink
#197/#198        csa.py:513  sum(exp_scores) + exp_sink -> sum_exp  SAVED
#199 Elementwise csa.py:514  exp_scores / sum_exp -> attn_weights
#200-#202 View x3 csa.py:517/518  aw_bm / kvo_bm                   SAVED x2
#203 BMM         csa.py:519  aw_bm @ kvo_bm          <-- **第 2 次 BMM(加权和)**
#204-#206        csa.py:519/520/521  reshape / permute / cast(query.dtype)
#207/#208        csa.py:524  permute -> 返回 [sq,b,n,d]
```
即:**手写侧那"一个 op"在源里是 2 次 BMM + 1 次 gather + 手写 softmax(max/exp/sum/div)
+ 1 个 O(S·topk) mask + 十余个 view/cast**。这条链正是「unfused / fused ≈ 2.1–3.1×」那个
真机比值的结构解释,现在它逐节点可见、逐节点带 `file:line`。

### 6.4 逐节点 dump(人工评审用)—— `Compressor`(fused,29 节点)

```
#1  MatMul      compressor.py:193  ins=[x]              out=kv          (linear_wkv)
#2  MatMul      compressor.py:194  ins=[x]              out=score       (linear_wgate)
#3  View        compressor.py:203  ins=[kv]             out=kv          (reshape)
#4  View        compressor.py:204  ins=[score]          out=score
#5  Cast        compressor.py:209  ins=[score]          out=...:fp32
#6  View        compressor.py:209  ins=[]               out=...:bf16  params=['ape']  <- 权重不进 ins
#7  Elementwise compressor.py:209  ins=[...fp32,...bf16] out=score_f32:fp32 params=['ape']
#8-#10  View x3 compressor.py:169/170/172  _overlap_transform(kv)      (chunk/roll/cat)
#11-#13 View x3 compressor.py:169/170/172  _overlap_transform(score_f32)
#14 Softmax     compressor.py:215  ins=[out__i3]        out=weights                 SAVED(反向用输出)
#15 Cast        compressor.py:216  ins=[out__i2]        out=...:fp32                SAVED
#16 Elementwise compressor.py:216  ins=[...fp32,weights] out=...      (kv*weights,两张量→非线性)
#17 Elementwise compressor.py:216  ins=[...]            out=pooled:fp32 (.sum(dim=1),线性)
#18 Cast        compressor.py:218  ins=[pooled]         out=...:bf16                SAVED
#19 Norm        compressor.py:218  ins=[...]            out=pooled:bf16
#20 Constant    compressor.py:231  ins=[]               out=freqs   (RoPE 频率表,无梯度)
#21-#22 View x2 compressor.py:233  freqs 切片(Subscript)
#23-#24 View x2 compressor.py:235/236  unsqueeze / split
#25 Elementwise compressor.py:237  ins=[kv_pe,freqs]    out=kv_pe   (apply_rope,linear)
#26-#27 View x2 compressor.py:243/244  cat / squeeze
#28 Elementwise compressor.py:223  ins=[...]            out=pooled  (Hadamard,linear 正交)
#29 View        compressor.py:224  ins=[pooled]         out=pooled  (unsqueeze)
saves(4): ['__arg__i5', '__arg__i6', 'weights', 'x']
detached: []
```
读法要点(都是本轮设计意图在图上的体现):
* `#6/#7` 的 `params=['ape']`:`Parameter` **不在 `ins` 里**(W2/W3),但**可见**;
* `#7` 的 out 是 **fp32** —— `score.astype(fp32) + ape` 的 dtype 提升算对了(这是真机 OOM
  traceback 命中的那个 `score_f32`,见 `SNAPSHOT.md`);
* `#14 Softmax` 的 saved 是**输出**(`weights`),不是输入;
* `#20 Constant`:RoPE 频率表无参数、无梯度,不进 saves;
* `#28`:`Hadamard` 归 `Elementwise{linear=True}`(常量正交矩阵,反向不存激活),与既有
  `ApplyRotaryPosEmb` 同一条判据。

其余三个 Cell 的 dump 用 `python scratchpad/probe_dsv4_walk.py {fused|unfused} --dump` 复现
(每行含 op 类型 / `file:line` / ins / out / `DETACHED` / `SAVED` / `params=`)。

### 6.5 全量测试 + 字节中性 [RAN]

```bash
python -m pytest tests -q     ->  1713 passed, 268 warnings in 53.71s
```
* 参照基线:`git worktree` 隔离跑 `f346b33` → **1676 passed**(任务书给的 1583 是 `7b9aa86`
  时的数;并行 agent 之后又加了 93 个)。1713 = 1676 + **37 净新增**。
* **字节中性证明(不是断言,是 diff)**:
  1. 手写 spec 全字段 dump(dsv3/dsv4 x L=4/6/8,逐 op 的
     `type/ins/out/params/saves/workspace/bwd_scratch/attrs`,每个 `TensorRef` 的 10 个字段全打)
     → `f346b33` vs 现在:**633 行,0 diff**;
  2. 完整 `Evaluator.evaluate()` 数字(dsv3/dsv4 x L=4/8 x rc None/full,`peak_event` +
     `peak_bytes` + 全 `breakdown`)→ **9 行 / 4029 字节,0 diff**;
  3. 抽取侧的**已有**路径:embedding 段 `derive_saves` 名册前后逐字相同(4 项、同 dtype、
     同 shape),尽管节点数 7→9(见 §6.6 末行)。

### 6.6 既有测试的**期望**变更台账(14 处,逐条给理由)

任务书要求「1583 pre-existing tests pass」。**字面上不可能**:`7b9aa86` 有一批测试断言的
正是本任务要拆掉的病症(见 §0.1)。处置原则:**保留每一条不变量,只换举例形态**。
**没有删除任何一条不变量**;该文件净新增 4 条。

| # | 测试 | 原断言(钉病症) | 新断言(钉不变量) |
|---|---|---|---|
| 1 | `test_with_only_body_raises_instead_of_zero_nodes` → `test_with_no_grad_body_is_walked_and_marked_detached` | 整块 `with` 被丢 → 0 节点门抛 | 块体**走查** + `attrs.detached`/`no_grad_region` + `dag.detached` 名册 |
| 2 | `..._with_is_recorded_as_dropped_stmt...` → `test_with_no_grad_marks_only_the_block_body` | `With` 进 `dropped_stmts`,块内算子被丢 | 块内节点标 detached、**块外不标**(边界要准) |
| 3 | `test_with_dropped_is_loud_under_strict` → `test_unknown_with_context_is_fail_loud` | `_no_grad` 的 With 在 strict 下抛 | **语义未知**的 `with` fail-loud(同一条「不猜」纪律,换到真判不出的形态) |
| 4 | `test_for_while_try_augassign_all_recorded` → `test_for_while_try_all_recorded` + `test_augassign_on_a_tensor_becomes_a_node` | 四类全记丢弃 | `For/While/Try` 仍记(**未支持,不变**);`AugAssign` 建节点(已支持) |
| 5 | `test_unhandled_assign_rhs_recorded_with_targets_and_kind` → `test_tensor_rhs_forms_now_become_nodes` | 5 条 RHS 全进 `dropped_assigns` | BinOp/Compare/Subscript **建节点**且属性正确;只剩 `w = self.weight`(合成源**无 `__init__`** → 判不出是否 Parameter → 保持 fail-loud 记账,**这是对的**) |
| 6 | `test_dropped_assign_target_is_flagged_unregistered` → `test_alias_and_weight_rhs_do_not_fabricate_nodes` + `test_weight_attr_becomes_a_param_operand`(**新增**) | 目标全进 `unregistered_targets`、下游拿占位 ref | 别名不再 unregistered;给了 `self_kinds` 后 `self.weight` 走 `param_operands`(W2/W3 逐条断言) |
| 7 | `test_strict_reports_every_kind_with_src` → `test_still_unsupported_rhs_forms_are_recorded_and_loud` | 那三条 RHS 在 strict 下报行号 | 换到**真未支持**的 `Dict`/`Lambda`/`JoinedStr`,机制(记账 + strict 抛)逐字保留 |
| 8 | `test_downstream_consumer_now_gets_a_real_ref_not_a_placeholder`(**新增**) | — | 反过来钉「边在」:`[q_node.id, act.id] in dag.edges`(此前没有) |
| 9 | `test_diagnostics_summary_counts_every_kind` | `dropped_assigns == 5`(`RHS` 源) | 换到 `UNSUPPORTED_RHS` 源,`== 3`;`set(s)` 断言不变 |
| 10 | `test_assert_extraction_clean_raises_...` / `..._allows_explicit_waivers` | 在 `RHS` 源上必须脏 | 换到 `UNSUPPORTED_RHS` 源,行为断言逐字不变 |
| 11 | `test_bare_function_aliases_are_enumerated` → `..._are_bound_and_unknown_ones_enumerated` | 4 条裸别名**全未绑** | 4 条**绑成正确 op 类型**;`unbound_aliases` 语义收窄为「真·未知原语」,新增 `mint.some_unreviewed_primitive` 守住"不得静默跳过" |
| 12 | `test_calling_an_unknown_bare_alias_is_fail_loud`(**新增**) | — | 未知原语被**调用** → `UnknownPrimitiveError`,报错含原语名 + "primitives" |
| 13 | `test_csaindexer_no_longer_reports_ok_zero_nodes` → `test_csaindexer_fused_branch_is_walked_and_fully_detached` | 真源 `CSAIndexer` 必须 **raise** | 必须**抽出非空图**、节点全在 `[214,232]`、全 detached、`Kernel` 的 saved 集有「可证明无反向」的理由。**同一条不变量的第三次演进**:假成功 → 抛 → 真抽出 |
| 14 | `test_working_dsv3_mla_path_also_surfaces_its_drops` → `..._scalar_forms_are_handled_not_dropped` | DSv3 MLA 路径 `dropped_assigns >= 2` | 该路径**零诊断**、那两行**不产节点**(不造假张量)、`:307` 的 Cast dtype 是**真 dtype**、`derive_saves` 名册逐字钉死 7 项 |
| — | `test_extract_embedding_walks_morph_func` / `..._records_opaque_allreduce`(`test_opdag_gpt_segments.py`) | 精确 7-op census | **9-op** census:`layers.py:152`(`input_ - vocab_start_index`,BinOp)与 `:164`(`input_mask.expand_dims(-1)`)是**两条真算子**,此前被 `_handle_assign` 静默丢。二者 PIN 都不存激活 → **`derive_saves` 逐字节不变**(实测同 4 项、同 dtype/shape,见 §6.5) |

### 6.7 红→绿证据

* 新测试文件整体在 `f346b33` 上**红**:`ImportError: cannot import name 'primitives' from
  'cost_eval.opdag'`(`primitives` / `module_index` 两个模块在改动前不存在);
* 行为层面的红:§6.1 末尾那段 `f346b33` 实跑输出(4 个 Cell / 4 个 FAIL);
* 绿:`python -m pytest tests/test_opdag_walker_core.py -q` → **33 passed**。

---

## 7. 明确**没做**的事(诚实清单)

| # | 未做 | 为什么 |
|---|---|---|
| 1 | **抽出的图的字节解析**(`infer_shapes` 种子/shard、`consumer._SYM2FIELD` 补 dsv4 符号) | 任务明确列为非目标(P2)。图里所有 shape 段仍是 `?` |
| 2 | **`opdag/to_resolved.py` 适配器** / 把 `liveness/` 指向抽出的图 | 任务明确列为非目标。为它预留了 `primitives.to_op_type()`(显式映射 + fail-loud) |
| 3 | **`cost_eval/layers/` 的 `saves` 数字一个没动** | 任务明确要求;三处差异已在 `KNOWN_GAPS` 双向棘轮台账里(D1/D2/D3) |
| 4 | **`init_dims` 的 `__init__` 布尔不喂 `walker_flags`** | 主动拒绝,有实测反例(§2.1):会静默算出与真机相反的 `enable_compress` |
| 5 | **detach 的传递闭包** | 需要参数操作数建模(P1#14);walker 只给 detach 边界事实 |
| 6 | **`is_expert` / `dim0` / `pin_under_recompute`** | `ResolvedOp`/`ResolvedTensor` 侧的契约,属适配器(T2) |
| 7 | **mHC 层 / MTP / pynative MoE / pynative embedding+loss** | 评估文档 P0#8 与 P3;本轮范围是 4 个 dsv4 Cell |
| 8 | **`_handle_chained_call` 的未知链式方法仍当纯视图** | 未在本轮 dsv4 链上实测触发(链式方法现优先走 `primitives.lookup_method`,命中不了才落这条) |
| 9 | **`tests/conftest.py` 的 `MF_ROOT` 默认仍指非权威树** | 不属本轮;新测试**自带 md5 门控**,非权威树一律 skip |
| 10 | **`For`/`While`/`Try` 语句仍不走查** | dsv4 链上零触发(实测);仍逐条记诊断 + strict 下抛 |
| 11 | **`FusedFunction` 的 `saved_internal`(`output`/`softmax_lse`)未解析到 ref** | 它们是 `forward` **内部**张量,没有 ins 对应项;逐字名单已挂在 `attrs["save_for_backward"]` 上(源真值),字节解析属 P2 |

---

## 8. 改动清单

| 文件 | 性质 |
|---|---|
| `cost_eval/opdag/primitives.py` | **新增**:原语表 + `OPDAG_OP_TO_OPTYPE`/`to_op_type()` + `UnknownPrimitiveError` |
| `cost_eval/opdag/module_index.py` | **新增**:跨文件 `ClassIndex` / `mro()` |
| `cost_eval/opdag/construct_walker.py` | With/Delete/AugAssign 处理器;RHS 判据层 + 9 个分支;标量/dtype/权重环境;Detach;跨文件 MRO;`.apply()`/`Kernel`/模块级函数内联;`input_axes`/`runtime_predicates` |
| `cost_eval/opdag/init_binder.py` | `bind_init` 形态 3/4a/4b/5;`_CLS2OP` 补 `IdentityOp`/`Hadamard`/`RotaryEmbedding`/`YarnRotaryEmbedding`;`unbound_aliases` 语义收窄 |
| `cost_eval/opdag/extractor.py` | 跨文件 MRO 单元;`self_kinds`/`fn_classes`/`module_funcs`/`module_consts`;`kw_self` 构造注入传播;新 kwargs |
| `cost_eval/opdag/init_dims.py` | `InitDims.self_kinds`;`mro_units=` 跨文件 |
| `cost_eval/opdag/bprop_rules.py` | PIN 补 9 条(`Compare/Constant/Detach/Where/Scatter/IndexSelect/TopK/FusedFunction/Kernel`);`outputs`/`from_attrs` 支持 + 缺 saved 集 fail-loud |
| `cost_eval/opdag/schema.py` | `OpDAG` 加 `detached` / `param_operands` / `deletes`(纯可加) |
| `tests/test_opdag_walker_core.py` | **新增** 33 测试(md5 门控) |
| `tests/test_opdag_drop_diagnostics.py` | 12 处期望迁移 + 4 条新增(台账见 §6.6) |
| `tests/test_opdag_gpt_segments.py` | 2 处 census 7→9(理由见 §6.6 末行) |
| `scratchpad/probe_dsv4_walk.py` | **新增**:一键复现探针(`--dump` 出逐节点表) |
