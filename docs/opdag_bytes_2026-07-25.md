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

