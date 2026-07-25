# 让抽出的源图产出**真字节** —— dsv4 四 Cell 的 shape/dtype 解析（2026-07-25）

> **性质**：`docs/opdag_walker_core_2026-07-25.md` §7 第 1 项（"抽出的图的字节解析"）的落地。
> 目标：让 `cost_eval/opdag/` 抽出的 dsv4 图里每个张量拿到 `local_numel > 0` / `dtype_bytes > 0`
> （契约 B1/B2/B3），权重与激活结构性分开（W1–W6 + B4），解不出的**显式**进 `unresolved`。
>
> **权威源**：`E:\97-codes\torch_parallel\mf-src-167\mindformers\`（commit `26354ff64`）。
> **不用** `E:\97-codes\torch_parallel\mindformers`（不同 commit）。
>
> 记法：**[RAN]** = 实跑（附命令 + 实际输出）；**[SRC]** = 逐字读源（带 file:line）。
> 本文按里程碑**逐段 checkpoint**（被打断也留证据）。

---

## 0. 基线校验 [RAN]

```bash
cd /e/97-codes/torch_parallel/mf-src-167 && md5sum \
  mindformers/pynative/transformers/experimental_attention_variant/{csa.py,indexer.py}
```
```
81673be3ad3cdd2191e28dd000a13f0e *csa.py        <- 与任务给定一致
8e18fee21c33507dd629c89f401986e6 *indexer.py    <- 与任务给定一致
```

```bash
cd /e/97-codes/torch_parallel/pynative-cost-evaluator && git log --oneline -1 && python -m pytest tests -q
```
```
a84954e feat(opdag): 让纯 AST 抽取器真正走通 DSv4-Flash 的 pynative 侧四个 Cell
1713 passed, 268 warnings in 59.36s
```
→ 基线 **1713 passed** 复现，HEAD = `a84954e`。

```bash
python scratchpad/probe_dsv4_walk.py both
```
```
### [FUSED] ratio=4
Compressor                   OK   29 nodes /  31 edges | diag: clean | detached=0 params=2
CSAIndexer                   OK    7 nodes /   6 edges | diag: clean | detached=6 params=0
CompressedSparseAttention    OK   90 nodes / 104 edges | diag: clean | detached=8 params=3
DSv4HybridSelfAttention      OK  123 nodes / 146 edges | diag: clean | detached=8 params=5
### [UNFUSED] ratio=4
Compressor                   OK   29 / 31    CSAIndexer  OK 14 / 13
CompressedSparseAttention    OK  207 / 240 (opaque=1)   DSv4HybridSelfAttention OK 240 / 281 (opaque=1)
```
→ 起步状态与任务书一致。

---

## 1. 里程碑 A —— 符号/维度底座:让 `MatMul` 的 out_dim 存在 [RAN]

### 1.1 起步实测:字节链的**第一道**断点不是 shape 规则,是 `linear_dims` 全空

```bash
python scratchpad/probe_dsv4_bytes.py both     # 新探针:抽图 → infer_shapes → dag_saved_bytes
```
```
Compressor                FAIL ValueError: shape_infer: MatMul 输入 shape 已知却缺 out_dim attr
                                （compressor.py:193）—— PART A 漏该 linear 输出维度,fail-loud
CSAIndexer (fused)        nodes=7  out-?=7   resolved=0 unresolved=0 total=0.000 MiB
CSAIndexer (unfused)      nodes=14 out-?=14  resolved=0 unresolved=5 total=0.000 MiB
CompressedSparseAttention FAIL 同上   DSv4HybridSelfAttention FAIL 同上
```

根因(**逐字读源**):`init_dims._do_assign` 只认 `build_module(sub.X, <in>, <out>, ...)` 的
**位序**实参(`len(value.args) >= 3`),而 **pynative 侧一律用关键字**:

| 侧 | 形态 | 定位符 |
|---|---|---|
| `parallel_core/training_graph` | `build_module(sub.x, config.hidden_size, config.q_lora_rank, …)` | `multi_latent_attention.py:634-643` |
| **`pynative`(本任务的目标)** | `build_module(sub.x, input_size=…, output_size=…, …)` | `compressor.py:97-104`、`deepseek_v4_hybrid_attention.py:93-96`、`indexer.py:117-119`、`dsa_indexer.py:169-171` |

→ `linear_dims` **全空** → `_bind_build_module` 不挂 `out_dim` → `shape_infer._matmul` fail-loud。
这不是"少几个符号",是 PART A 在 pynative 侧**整条通路缺失**。

### 1.2 落地(全部在本轮拥有的 5 个文件里)

| 文件 | 改动 | 判据 |
|---|---|---|
| `init_dims.py` | `_build_module_dims`:位序优先、缺位再看 `input_size=`/`output_size=` 关键字 | 上表两侧形态 |
| `init_dims.py` | `INIT_PARAM_SEEDS` 保留键:`__init__` **形参**的维度/值注入 | `head_dim=config.v_head_dim`(csa.py:604)单抽子 Cell 时父不在场;`compress_ratio` 缺省 0 而真机 4/128(csa.py:556) |
| `init_dims.py` | `eval_dim` 末档:结构化解不出但**具体值已知的整数** → 当系数 | `self.coff = 1 + int(self.overlap)`(compressor.py:89-90)—— 纯 host 算术,值由 config_flags 定 |
| `init_dims.py` | `_eval_val` 补 `IfExp` / `BinOp` / `int()`/`bool()` | 同上 |
| `init_dims.py` | `param_shapes` / `param_dtypes`:`Parameter(mint.empty((a,b), dtype=…))` 的**形状与 dtype** | `compressor.py:117` `ape`、`csa.py:589` `attn_sink`(明写 `float32`,**不是** params_dtype) |
| `sym_shape.py` | `CONFIG2SYM` 补 7 条 dsv4 符号;**键是 config 属性名** | `self.index_n_heads = config.dsa_indexer_n_heads`(indexer.py:94)—— 两者不同名,按 self 名建表会被 `_KNOWN_DIM_SYMS` 滤掉 |
| `sym_shape.py` | `floordiv` 第 3 档:系数不整除时形成**整除原子** `S//4` | `compressor.py:196/201`、`csa.py:762` 的压缩序列长 = `S // ratio` |
| `sym_shape.py` | `NUMEL_ONLY`(`~` 前缀)+ `mark_/strip_numel_only` | 见 §3(轴信息未被 walker 记下时的"只知 numel"档) |
| `consumer.py` | `_SYM2FIELD` 补 7 条 → `DimTable` 字段;`_sym_value` 支持 `//` 原子 | `model_spec.py:31-32` |
| `consumer.py` | `_DTYPE_BYTES` 补 `bool`=1 / `int32`=4 / `int64`=8 | `csa.py:779/810` 两个 bool mask、`indexer.py:263` `cast(topk_indices, int32)` |
| `consumer.py` | `local_shape_elems(...)`:TP/EP shard + CP(只切**首个**含 S 的轴)+ `cp_kv` colossal 例外 | 语义逐条对齐 `shape_eval.resolve_tensor`(shape_eval.py:81-120) |

### 1.3 验收 [RAN]

```bash
python -m pytest tests/test_opdag_bytes.py -q     ->  27 passed
```

**隔离回归**(主树被并行 agent 同时改着,故在 `a84954e` 的 `git worktree` 里只放本轮 3 个改动文件
+ 新测试):
```bash
git worktree add -f <scratch>/wt-bytes a84954e
cp cost_eval/opdag/{init_dims,sym_shape,consumer}.py <wt>/cost_eval/opdag/
cp tests/test_opdag_bytes.py <wt>/tests/ && cd <wt> && python -m pytest tests -q
->  1 failed, 1739 passed        # 1713 基线 + 27 新增 − 1
```
唯一失败 = `tests/test_opdag_sym_shape.py::test_floordiv_halves_coeff`,即 §1.4 的**期望变更**。
(主树同时还有 7 个 `tests/test_opdag_components.py` / `test_opdag_drop_diagnostics.py` 失败 ——
`git diff --stat` 显示 `construct_walker.py`/`extractor.py`/`module_resolver.py` 被并行 agent
改着,**不在本轮 5 个文件内**;隔离跑证实与本轮无关。)

### 1.4 既有测试的**期望**变更台账(1 处,唯一)

| 测试 | 原断言 | 新断言 | 为什么是同一条不变量 |
|---|---|---|---|
| `test_opdag_sym_shape.py::test_floordiv_halves_coeff` | `floordiv(parse_axis("H"), 2) is None  # not fabricated` | 拆成 `test_floordiv_keeps_the_expression_when_coeff_does_not_divide`:`render_term == "H//2"`,并把"决不杜撰"搬到它真正该守的两处(`divide()` 的 reshape `-1` 消元仍 None;`n==0` 仍 None) | 原断言把「不能干净约掉」与「编造尺寸」混为一谈:`H//2` 的值**完全由 DimTable 的 H 决定**(4096//2=2048),是源里逐字写着的表达式(`compressor.py:196/201`)。那一档 None 让整条 dsv4 压缩链 shape 全 `?` |

未知符号的整除式仍必须 None —— 由新增 `test_consumer_floordiv_atom_of_unknown_symbol_is_unresolved` 守住。

---

## 2. 里程碑 B —— shape 规则:三处**静默算错**改成显式记账 [RAN]

字节链的第二道断点不是"解不出",而是**解错了还不知道**。改动前实测:`_dispatch` 对 walker 没记
轴信息的原语一律 `_passthrough` 顶替,而这三类的元素数**会变**:

| 原语 | 源定位符 | 旧行为(passthrough)的后果 | 现行为 |
|---|---|---|---|
| 归约 `.sum(dim=1)` / `mint.mean` / `mint.max` | `compressor.py:216`、`deepseek_v4_hybrid_attention.py:245`、`indexer.py:388` | 按**归约前**的大小计 → compressor 那处**多算 2·ratio = 8 倍** | `reduce_axis_unknown` 记账,保 `?` |
| `mint.chunk(t, 2, dim=-1)` | `compressor.py:169` | 每份按整块计 → **多算 n 倍** | 份数由**元组解包元数**证得(`attrs["outs"]` —— Python 元组解包只有恰好 k 份才成立 = 源侧事实),且只在符号上能**精确整除**时才认;否则记账 |
| `mint.cat(list, dim)` | `compressor.py:172/243`、`indexer.py:188`、`deepseek_v4_hybrid_attention.py:215` | 自由调用点**没有** `concat_axis` attr → 默认按轴 0 相加。`cat([kv_nope, kv_pe], dim=-1)`(compressor.py:243)两输入末轴不同(`d−qk_pos_emb_head_dim` vs `qk_pos_emb_head_dim`)→ 给出 `2n·b·(d−64)`,真值 `n·b·d` | 改为**元素数 = 各输入元素数之和**(与轴无关 ⇒ 精确)+ 标 `numel_only` |

### 2.1 新增一档 `numel_only`(`~` 前缀)—— 元素数精确、轴结构未知

契约 **B1** 只要 `local_numel > 0`,故"只知元素数"是**可用**的。关键是**不许**拿一个假轴序往下算:
需要轴结构的算子(MatMul 换末轴 / BMM / split / tile / slice / FlashAttention)遇到 `~` 一律记
`needs_axis_structure` 并拒绝。`reshape` 反过来能**恢复**轴结构(它只需要总积来消 `-1`)。

`reshape` 目标维解不出时也退 `numel_only`(而不是整体放弃)—— **reshape 恒不改变元素数**是算子
定义,不是估计。这一档把 `Compressor` 的两个大 save 从 `?` 变成精确值:

```
weights    ~((B·S·v_head_dim)+(B·S·v_head_dim))  bf16   8.0000 MiB
__arg__i5  ~((B·S·v_head_dim)+(B·S·v_head_dim))  fp32  16.0000 MiB
```
人工核对:`_overlap_transform` 后是 `[n_groups, 2·ratio, b, head_dim]` =
`(4096/4)·8·1·512 = 4194304` 元素 → bf16 = **8 MiB**、fp32 = **16 MiB**,与实测一致。

### 2.2 三个**已算错**的实例(本轮修掉,附证据)

1. **和式原子不能往返** —— `_split_top(s, "·")` 只认括号深度、不认 `+`。把两个乘积裸着相加成
   `"B·S·v_head_dim+B·S·v_head_dim"`,再 `parse_axis` 会切成
   `["B","S","v_head_dim+B","S","v_head_dim"]` —— 冒出假和式单元 `v_head_dim+B`。实测:
   一个 8 MiB 的 concat 产物被算成 `~(B·(B+v_head_dim)·S·S·v_head_dim)` = **8404992 MiB**。
   修法:`sym_shape._sum_term` 给乘积项加括号。旧代码只在**单符号**之间相加故未暴露。
2. **同名重绑后 env 留旧 shape** —— 源里大量 `kv = self.reshape(kv, …)`(compressor.py:203/204)。
   本节点解不出时若 env 仍留着重绑前那个张量的 shape,下游就拿着一个已不成立的形状继续算。
   修法:解不出即作废该名 —— 但**只对本图产出过的名字**;调用方给的种子是外部权威事实
   (实测 `ffn.py:146` `w1 = cast(self.weight1)` 因权重不进 `ins` 而解不出,若连种子一起作废,
   `GroupedMatMul` 就丢了权重末轴,`test_moe_grouped_gemm_operand_is_e_cap_h` 立刻红)。
3. **`numel_only` 泄漏进轴** —— `b, s, n, d = x.shape` 若 src 是 `~` shape,逐轴绑定不成立
   (那种 shape 只有一条"总积"轴)。实测泄漏样例:`q_bm` 被解成 `(B·~S)·n_heads·v_head_dim`。
   修法:src 为 `numel_only` 时逐轴一律绑 None。

### 2.3 dtype:`Compare` 产 **bool = 1 字节**

walker 的 `_emit` 按 ins 推产出 dtype → 比较类算子被记成 compute dtype(bf16)= **2×**。
真源两个大 mask 会被 `mint.where` 的 `PIN{"inputs":[0]}`(bprop_rules.py:55)当 cond **保留**:
`csa.py:779` `future = cm >= positions // ratio`(O(S·S/r))、`csa.py:810`
`valid = topk_… < unsqueeze(…)`(O(S·topk))。故 `infer_shapes` 把 `Compare` 的产出 dtype 订正为
`bool`,**并连带下游 ins ref 一起改**(否则 `derive_saves` 读的是下游那份旧串);
`consumer._DTYPE_BYTES` 补 `bool=1` / `int32=4` / `int64=8`。

### 2.4 `Detach` / `Compare` / `Constant` 的字节处置(任务第 5 点)

| op | 字节处置 | 依据 |
|---|---|---|
| `Detach` | **形状透传 + 别名去重**:`ops.stop_gradient(x)` 不复制存储,`x_detach` 与 `x` 只计一份(`consumer.detach_aliases` 做传递闭包到最原始的名字;判据是"root 也在 saves 里",与顺序无关;被去重的仍逐条列在 `detach_aliased` 里,可见不静默)。`PIN["Detach"]={"inputs":[]}` 本身不 save | 真源 6 处 `csa.py:665/666/764/765/794/795`;实测 CSA 里 `x`/`x_detach__i1` 各 32 MiB,去重后 CSA fused 从 332 → **300 MiB** |
| `Compare` | **形状透传 + dtype=bool(1 B)**;`PIN` 不 save(不可微) | 见 §2.3 |
| `Constant` | **记账 `constant_shape_unknown`**:`mint.arange/full/zeros/ones` 的 shape 实参未被 walker 记进 attrs → 无从得知形状。`PIN` 不 save | `csa.py:483` `arange(b)`、`indexer.py:355` `mint.full`、`:219` RoPE 频率表 |

三者都**没有** `OpType` 对应物(`primitives.to_op_type()` 显式返回 `None` 并 fail-loud),
本轮不新增枚举成员 —— 与 walker 文档 §1.3 的判断一致(硬塞进 `elementwise` 会被
`RecomputeSpec.op_matches` 的子串匹配连带选进重算集 = 静默错)。

---

## 3. 验收:四个 Cell × fused/unfused 的逐节点字节 [RAN]

```bash
# 隔离环境(a84954e worktree + 本轮 4 个文件),权威快照 mf-src-167
PYTHONIOENCODING=utf-8 python scratchpad/acceptance.py
```
站点配置逐字段等于 `tests/test_source_truth_saves_audit.SITE_DIMS`
(seq4096 / topk512 / 64 heads / v_head_dim 512 / b=1,tp=cp=ep=1),故与手写侧可对账。

| Cell | 支 | 节点 | saves 已解析 | unresolved | detach 别名 | **saves 字节** | params | param 字节 | 契约违约 |
|---|---|---|---|---|---|---|---|---|---|
| `Compressor` | fused | 29 | 3 | 1 | 0 | **56.000 MiB** | 1/1 | 0.0078 MiB | **0** |
| `CSAIndexer` | fused | 7 | 0 | 0 | 0 | 0 | 0/0 | 0 | 1(`L1` ops 为空) |
| `CompressedSparseAttention` | fused | 90 | 4 | 5 | 1 | **300.000 MiB** | 2/2 | 0.0081 MiB | **0** |
| `DSv4HybridSelfAttention` | fused | 123 | 4 | 12 | 2 | **312.000 MiB** | 2/3 | 0.0081 MiB | **0** |
| `Compressor` | unfused | 29 | 3 | 1 | 0 | **56.000 MiB** | 1/1 | 0.0078 MiB | **0** |
| `CSAIndexer` | unfused | 14 | 0 | 5 | 0 | 0 | 0/0 | 0 | 1(`L1` ops 为空) |
| `CompressedSparseAttention` | unfused | 207 | 5 | 35 | 1 | **808.500 MiB** | 2/2 | 0.0081 MiB | **0** |
| `DSv4HybridSelfAttention` | unfused | 240 | 4 | 42 | 2 | **312.000 MiB** | 2/3 | 0.0081 MiB | **0** |

逐张量(fused):
```
Compressor                x 32.000(bf16) / __arg__i5 16.000(fp32) / weights 8.000(bf16)
CompressedSparseAttention query 256.000(bf16) / x 32.000 / qr_detach__i1 8.000 / key 4.000
                          ALIAS x_detach__i1 -> x        (同一存储,32.000 不计)
DSv4HybridSelfAttention   q 256.000(bf16) / x 32.000 / q_compressed 16.000(fp32) / kv 8.000(fp32)
                          ALIAS qr_detach__i1 -> q_compressed(8.000) / x_detach__i1 -> x(32.000)
```
逐张量(unfused,CSA 多出的三项):
```
q_bm__i19   ~(B·S·n_heads·v_head_dim)  fp32  512.000 MiB   <- csa.py:494 的 fp32 Q
__arg__i15  S·B·n_heads·v_head_dim     bf16  256.000 MiB   <- csa.py:794 ops.stop_gradient(query)
__arg__i1   ~S·B·index_n_heads         bf16    0.500 MiB   <- indexer.py:251
```

### 3.1 `unresolved` 名单(**完整**,每条带 `file:line` + 原因码)

`infer_shapes(..., report=[])` 的逐节点台账按原因码汇总(fused / unfused):

| 原因码 | Compressor | CSAIndexer | CSA | DSv4Hybrid |
|---|---|---|---|---|
| `no_input_shape`(上游未解出) | 9 / 9 | 5 / 11 | 54 / 108 | 69 / 128 |
| `constant_shape_unknown` | 1 / 1 | 2 / 0 | 6 / 19 | 7 / 20 |
| `reduce_axis_unknown` | 1 / 1 | 0 / 2 | 2 / 11 | 3 / 12 |
| `no_out_dim` | 0 / 0 | 0 / 0 | 4 / 5 | 4 / 4 |
| `reshape_unresolved` | 2 / 2 | 0 / 0 | 1 / 2 | 1 / 1 |
| `split_size_unresolved` | 0 / 0 | 0 / 0 | 0 / 0 | 6 / 6 |
| `needs_axis_structure` | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |

save 级 `unresolved` 逐条(fused `DSv4HybridSelfAttention` 全 12 条;其余由脚本一键复现):
```
__arg__i0 / __arg__i1  <- compressor.py:209 no_input_shape
weights                <- compressor.py:215 no_input_shape
__arg__i5              <- compressor.py:216 no_input_shape
__arg__i6              <- compressor.py:218 no_input_shape
__ret__i5 / __ret__i7  <- deepseek_v4_hybrid_attention.py:215 no_input_shape
compressed_kv__i1      <- csa.py:677 no_input_shape
attn_sink__i1          <- csa.py:687 no_input_shape
cg                     <- deepseek_v4_hybrid_attention.py:286 no_input_shape
wo                     <- deepseek_v4_hybrid_attention.py:287 no_input_shape
intermediate           <- deepseek_v4_hybrid_attention.py:293 no_input_shape
```
**没有任何一条被填了"看起来合理"的数** —— 全部保 `?`,并在 `dag_saved_bytes` 的 `unresolved`
列表与 `infer_shapes` 的 `report` 里双向可见。

### 3.2 契约校验 `validate_resolved_layer` / `validate_param_census`

用一个**只为跑校验器**的最小 `ResolvedLayer` 实现(`scratchpad/acceptance.py` 的 `T/O/L`;
**不是** `to_resolved.py`,不注册 graph source —— 那是本任务的明确非目标),把**已解析**的节点
折成层:

* **`validate_resolved_layer` = 0 违约**,6 个 Cell×支(Compressor / CSA / DSv4Hybrid ×
  fused+unfused)。含 W1/W2/W3/W4/W5/W6、S1/S3、B1/B2/B3、T1/T2、G2/G3 全部通过。
* 唯一违约:`CSAIndexer` 两支 = `[L1] ops 为空`。**解释**:它 7(fused)/14(unfused)个节点
  一个都不可解析(fused 支整块在 `_no_grad()` 里、输入是上游未解出的 indexer 张量),于是折出
  的层为空。这正是契约 **B1/L1** 设计要挡的东西 —— 一张解不出的图**不能**冒充可用图。
* **`validate_param_census`**:`CSAIndexer` PASS;其余 6 个报 `[B4] param 字节 0 != 参考
  8192/8448`。**解释**(不是权重漏册):`dag_param_bytes` 这条**权威 param census** 把权重全部
  解出了(`ape` 8192 B、`attn_sink` 256 B),但那些权重的**承载节点**(compressor.py:208/209、
  csa.py:687)自身 shape 解不出而被整个跳过 → 折出的层里没有它们。差额**逐字节等于**被跳过节点
  上的权重和(8192 / 8448),方向为负 —— 与契约文档 §4.2 描述的"负差 = 权重漏进 params"是同一个
  信号,但根因是**节点跳过**而非权重漏册。修掉 §5 的 G2/G4/G5 后这条自动消失。

### 3.3 权重 vs 激活:结构性成立,88 MiB 那个病在本路径上**不存在**

walker 把 `self.<attr>` 且 `__init__` 里是 `Parameter(...)` 的操作数**刻意排除在 `OpNode.ins`
之外**、单列 `OpDAG.param_operands`(`schema.py:56-59`)→ `derive_saves` 结构上**不可能**把权重
当激活 save,契约 **W2/W4 自动成立**(评估文档 §7.2 那 88 MiB 的病在本路径上没有对应物)。
形状/dtype 由本轮新增的 `init_dims.param_shapes` / `param_dtypes` 从源逐字读出:

```
ape                  4·2·v_head_dim   bf16   0.007812 MiB   <- compressor.py:117
attn_sink            n_heads          fp32   0.000244 MiB   <- csa.py:589(**明写 float32**,不是 params_dtype)
linear_o_group_proj  (o_groups·o_lora_rank)·((n_heads·v_head_dim)//o_groups)  <- deepseek_v4:139
```
`linear_o_group_proj` 在 `DSv4HybridSelfAttention` 上仍 unresolved —— 见 §5 的 G6。

---

## 4. 与手写 spec 的 like-for-like 对账 [RAN] + [SRC]

手写侧数字由**评估器自己的 `resolve_tensor`** 在同一站点配置下算出。手写 fused r4
`sparse_attn` 名册 = **9 项 / 847.500 MiB**(与任务书给定值逐字相同);unfused = 16 项 /
17664.000 MiB。

| 抽取侧 | 手写侧 | 抽取 MiB | 手写 MiB | delta | 判决(**从源**) |
|---|---|---|---|---|---|
| `q`(DSv4Hybrid) | `q`(`q_hnorm.saves`) | 256.000 | 256.000 | **0** | 相等 |
| `x` | `ln1`(`linear_q_down.saves`) | 32.000 | 32.000 | **0** | 相等(同一张量,异名) |
| `qr_detach__i1` | `q_a_out`(`linear_q_up.saves`) | 8.000 | 8.000 | **0** | 相等 |
| `key`(CSA) | `kv_a_out`(`sparse_attn.saves`) | 4.000 | 4.000 | **0** | 相等 |
| `q_bm__i19`(unfused CSA) | `uq_bm` / `uq_f32` | 512.000 | 512.000 | **0** | 相等(`csa.py:494` 的 fp32 Q) |
| **`query`(CSA fused)** | **`q_hnorm_fp32`** | **256.000** | **512.000** | **−256.000** | **抽取侧对** —— 见 §4.1 |
| `q_compressed` | `q_compressed` | 16.000 | 8.000 | +8.000 | **口径差,非分歧** —— 见 §4.2 |
| `kv` | `kv` | 8.000 | 4.000 | +4.000 | 同 §4.2 |
| `__arg__i15`(unfused) | (无对应) | 256.000 | — | — | 仅抽取侧 —— 见 §4.3 |

### 4.1 `q_hnorm_fp32`:手写侧比源**高 256 MiB**(fused r4/层),且与 `q` 重复计

逐字读源:
* `deepseek_v4_hybrid_attention.py:245`
  `q = q * mint.rsqrt(mint.mean(q * q, dim=-1, keepdim=True) + eps)`
  —— **纯 bf16 逐元素**(`q` 来自 `linear_q_up_proj`,compute_dtype=bf16),**没有**
  `.astype(mstype.float32)`。同文件 `:158-161` 确有一个 fp32 `q_rms_gamma` 与
  `self.rms_norm = ops.rms_norm`,但**本快照里 `rms_norm` 从未被调用**(walker 文档 §1.2 的
  fail-loud 门在真源上抓到的正是这一条:"绑了但本版本未调用")→ fp32 归一化路径是死代码;
* `csa.py:674` `query = self.permute(query, (1, 0, 2, 3))` —— 只换轴序,dtype 不变;
* `csa.py:689-697` `FusedSparseFlashMlaWithIndexerLoss.apply(query, key, compressed_kv,
  topk_indices, ops.cast(query_index, bfloat16), ops.cast(key_index, bfloat16),
  ops.cast(weights, float32), attn_sink, …)` —— **只有 `weights` 与 `attn_sink`(`:687`)被 cast
  到 fp32**,`query` 是**原封不动的 bf16**;
* 该 kernel 的 saved 名册(`csa.py:224`,`fn_saves` 逐字读出)第 1 项就是这个 `query`。

⇒ 源侧被 save 的是 **bf16 的 `query` = 256 MiB**,不是 fp32 的 512 MiB。手写侧
`sparse_attn.saves` 里的 `q_hnorm_fp32`(`dtype_bytes=4` → 512 MiB)**比源高 256 MiB**;
更进一步,手写侧 `q_hnorm.saves=[q]`(256 MiB bf16)与 `sparse_attn.saves=[q_hnorm_fp32]`
(512 MiB fp32)**是同一个源张量的两个名字**,而 `structure_mem` 的 saves 是**按名去重的字典**
(structure_mem.py:287)→ 二者都计 = **768 MiB**,源侧只该有 **256 MiB** ⇒ 手写侧
**+512 MiB / r4 层**(fused)。
诚实的保留:`mint.mean`/`rsqrt` 内部若做 fp32 累加,那是 **kernel scratch**(瞬态),不是
`save_for_backward` 的项;判据以 `ctx.save_for_backward` 的逐字名册为准。
**本轮未改任何 `cost_eval/layers/**` 的数字**(任务明确要求),此条作为**新台账项**报出。

### 4.2 `q_compressed` / `kv`:norm 的 fp32 抬升**口径差**,不是分歧

抽取侧 `derive_saves` 对 `Norm` 的输入用 `attrs["ln_compute_dtype"]`(=fp32)**预抬**
(`bprop_rules.py:112-115`);手写侧 `dtype_bytes=None`(bf16),由**消费侧** `structure_mem._dt`
按 op 类型再抬。二者**有效字节相同**,差别只在"在哪里抬"。
按契约 `ResolvedTensorContract` 的明文约定("norm 的 fp32 cast 由消费侧 `structure_mem._dt`
再抬,**不**预抬"),**手写侧是契约口径**;抽取侧若要接 `to_resolved.py`,必须先**去掉**这层预抬,
否则会被抬两次(×2)。→ 已列为 §5 的交接项 T1。

### 4.3 `__arg__i15`(= `ops.stop_gradient(query)`)与 `ukl1`/`ukl2`:两侧**同向过计**

抽取侧把 `csa.py:794` 的 `ops.stop_gradient(query)` 产物计成 256 MiB save。源侧事实:
`csa.py:794-795` 两个入参都过 `stop_gradient` → `indexer.py:350` 的 O(S·S/r) matmul +
`:380` 的 fp32 softmax 这条链上**没有任何参数** → MS 不建 autograd 节点 → 反向无节点读它。
`derive_saves` **刻意不做 detach 的传递闭包**(walker 文档 §5 / §7 第 5 项:闭包需要参数可达性
建模)→ 抽取侧此处过计 256 MiB。手写侧对应的 `ukl1`/`ukl2`(各 1024 MiB fp32、
`detached=True`)同样在 `saves` 里,`KNOWN_GAPS` 台账已记为 **−2048 MiB / r4 层**。
⇒ **两侧同向过计**,根因同一条(detach 闭包未建模);抽取侧量级更小只是因为它的下游链
还没解析出来。修法在 §5 的 T2。

---

## 5. 明确**没做**的事 + 交接项(诚实清单)

| # | 未做 / 未解 | 为什么 | 归属 |
|---|---|---|---|
| **G2** | **构造实参里的维度未传播到子 Cell 的 `__init__` 求值** —— `Compressor(head_dim=config.v_head_dim)`(csa.py:604,=512)与 `Compressor(head_dim=self.index_head_dim)`(indexer.py:128,=128)是**同一个类的两个构造点**,`eval_init_dims(tree, cls, config_flags)` 拿不到构造点上下文。给一个全局 `INIT_PARAM_SEEDS["head_dim"]` 必然把另一处算错 **4×** → **刻意不给**,于是 CSA/DSv4Hybrid 里那 58 个 compressor 节点的 `linear_wkv/wgate` 报 `no_out_dim` | 修法:`extractor._kw_self_args` 已在传播**模块**注入,同样把 `build_module(...)` 的**维度关键字实参**传给子 Cell 的 `eval_init_dims` local 环境 | `extractor.py`(本轮不可改) |
| **G3** | **内联子 Cell 的 `dims_ctx` 未合并**(`dag.dims_ctx` 只有顶层类的)→ `indexer.py:178` 的 `self.index_n_heads/index_head_dim` 在抽 CSA 时解不出 | 本轮提供了**调用方侧**的安全解法 `shape_infer.merge_dims_ctx(*ctxs)`(同名不同值 fail-loud),验收脚本已用它。根治仍应由 extractor 在内联时合并 | 本轮已缓解 / extractor 根治 |
| **G4** | **轴信息未被 walker 记进 attrs**:`permute` 的轴序、归约的 `dim`、`chunk` 的 `dim`、`cat` 的轴、`mint.topk` 的 `k`、`Constant`(arange/full/zeros)的 shape、`slice` 的起止 | 本轮把它们从"静默算错"变成"显式记账"(§2),但**解不出就是解不出**。这是 `unresolved` 里最大的一块 | `construct_walker.py`。建议逐条记 `reduce_dim` / `chunk_dim`+`chunks` / `concat_axis` / `permute_dims` / `topk_k` / `const_shape` / `slice_bounds` |
| **G5** | **construct 局部标量未导出**:`n_compressed = cutoff // ratio`(compressor.py:201)、`seqlen, bsz, _ = x.shape`(indexer.py:173)。walker 内部**已经把它们求成具体整数**(`_scalar_of`;`if sq < ratio` 能判定就是证据),但 `OpDAG` 里没有这张表;且 `SubExtract` **不带 `scalar_binds`** → 内联子 Cell 的 `x.shape` 解包整个丢失(实测 CSA 的 `scalar_binds` 只剩 1 条) | 建议 `OpDAG` 加 `scalars: {name: 符号串}`,并让 `SubExtract` 携带 `scalar_binds`。本轮靠 `reshape` 的 numel 兜底把 `Compressor` 的两个大 save 救了回来 | `construct_walker.py` / `extractor.py` |
| **G6** | `linear_o_group_proj` 的 shape 仍未解出(`deepseek_v4:139`) | `o_chunk = self.query_projection_size // o_groups`(:276)本轮已能解(`init_dims` 的 FloorDiv 支持"除数是已知整数值的名字"),但该 `Parameter` 用的是 `__init__` **局部名** `o_groups`/`o_lora_rank`/`o_chunk`,求值顺序还差一步 | 本轮 `init_dims` 可续修(下一轮) |
| **T1** | **norm 输入的 fp32 预抬要去掉**才能接 `to_resolved.py` | 见 §4.2:契约明文要求"不预抬"。改法在 `bprop_rules.derive_saves`(本轮**未动** —— 它不在本任务拥有的 5 个文件里) | `bprop_rules.py` |
| **T2** | **detach 的传递闭包** | 需要"链上有没有参数"的可达性建模(P1#14)。两侧当前同向过计(§4.3) | `to_resolved.py` / consumer |
| **T3** | `mint.topk` 的**双名冲突**:`topk_indices` 在 `indexer.py:262`(TopK 产出)与 `:263`(`cast(..., int32)`)**同名**,`derive_saves` 按名去重只留首见(TopK 那份,dtype 继承 = bf16)→ 若两者 dtype 不同会取错。手写侧按 int32(4 B)计,与 `:263` 一致 | 本轮未改(`bprop_rules` 的去重口径不在可改范围);在此报出 | `bprop_rules.py` / walker |
| — | **`opdag/to_resolved.py`、把 `liveness/` 指向抽出的图、walker 覆盖度扩展、`cost_eval/layers/**` 的数字** | 任务明确列为非目标 | — |

---

## 6. 回归与字节中性 [RAN]

### 6.1 隔离全量回归(主树被并行 agent 同时改着,故必须隔离)

```bash
git worktree add -f <scratch>/wt-bytes a84954e
cp cost_eval/opdag/{init_dims,sym_shape,consumer,shape_infer}.py <wt>/cost_eval/opdag/
cp tests/test_opdag_{bytes,sym_shape}.py <wt>/tests/ && cd <wt> && python -m pytest tests -q
->  1762 passed, 268 warnings in 52.09s
```
**1762 = 1713 基线 + 49 净新增**,**0 failed**。1713 条**既有**测试全过(唯一一处期望变更见
§1.4;那条测试被改成钉同一条不变量的两个新断言,不变量没有减少)。

### 6.2 字节中性 —— **diff,不是断言**

`scratchpad/dump_numbers.py`(手写 spec 全字段 dump:dsv3 L=4/6/8 逐 op 的
`type/ins/out/params/saves/workspace/bwd_scratch/attrs` + 每个 `TensorRef` 的 10 个字段;
再加完整 `Evaluator.evaluate()`:dsv3 L=4/8 × rc None/full 的 `peak_bytes` + `peak_event` +
**全部 17 个 breakdown 字段**):

```bash
cd <wt-base@a84954e> && python dump_numbers.py > before.txt      # 650 行 / 79088 字节
cd <wt-bytes>        && python dump_numbers.py > after.txt       # 650 行 / 79088 字节
diff before.txt after.txt   ->  *** 0 diff ***
```
即抽取侧仍**完全不在数字路径上**,本轮 4 个文件对既有桶路径**逐字节无影响**。

**主树(与并行 agent 的改动合在一起)同样 0 diff**:
```bash
cd <main tree @ b5502d2> && PYTHONPATH=. python scratchpad/dump_numbers.py > after_main.txt
diff before.txt after_main.txt   ->  *** 0 diff ***   (650 行 / 79088 字节)
python -m pytest tests -q        ->  1780 passed, 268 warnings in 52.52s   (0 failed)
```

### 6.3 主树状态说明

主树(`feat/unified-llm-modelspec` HEAD)同时有并行 agent 对 `construct_walker.py` /
`extractor.py` / `primitives.py` / `module_resolver.py` 的在飞改动,其间实测有 6 个
`tests/test_opdag_components.py` / `test_opdag_drop_diagnostics.py` 失败,且一度
`extract_cell` 直接抛异常。隔离跑证实**与本轮 4 个文件无关**(本轮在 `a84954e` 上
1762 passed / 0 failed)。本轮**未触碰**那 5 个文件,也未触碰
`cost_eval/liveness/**`、`tools/**`、`cost_eval/layers/**`、`structure_mem.py`、
`mem_timeline.py`、`serve_explorer.py`。

---

## 7. 改动清单

| 文件 | 性质 |
|---|---|
| `cost_eval/opdag/init_dims.py` | `build_module` 关键字维度;`INIT_PARAM_SEEDS`;`eval_dim` 整数末档;`_eval_val` 补 IfExp/BinOp/`int()`/`bool()`;FloorDiv 除数可为已知整数名;`param_shapes`/`param_dtypes` |
| `cost_eval/opdag/sym_shape.py` | `CONFIG2SYM` 补 7 条 dsv4 符号;`floordiv` 整除原子;`_sum_term` 括号修正;`NUMEL_ONLY` + `mark_/strip_numel_only` |
| `cost_eval/opdag/shape_infer.py` | `report=` 逐节点记账(11 原因码);reduce/chunk/cat/slice/Constant/TopK/IndexSelect/BMM/Where 规则;`numel_only` 全链;reshape numel 兜底;`Compare` dtype 订正(含下游 ins);同名重绑作废;`merge_dims_ctx` |
| `cost_eval/opdag/consumer.py` | `_SYM2FIELD` 补 7 条;`//` 原子与乘积项求值;`_DTYPE_BYTES` 补 bool/int32/int64;`local_shape_elems`(TP/EP/CP/cp_kv);`dag_local_saved_bytes`;`detach_aliases` 去重;`dag_param_bytes` |
| `tests/test_opdag_bytes.py` | **新增** 49 测试(符号代数 / 维度求值 / dtype / 并行整除 / 记账纪律 / Detach 别名 / param census / merge_dims_ctx) |
| `tests/test_opdag_sym_shape.py` | 1 处期望变更(台账 §1.4),净 +1 条断言 |
| `scratchpad/probe_dsv4_bytes.py` | **新增** 字节探针(`--gaps` 出逐原因码台账) |
| `scratchpad/acceptance.py` | **新增** 验收台(逐节点字节 + param census + 契约校验 + 手写对账) |

