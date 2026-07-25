# opdag 抽取覆盖度评估 — DSv4-Flash（2026-07-25）

> **性质**：覆盖度评估（coverage assessment），非重写。目标：判断 `cost_eval/opdag/` 这条
> **纯 AST** 源抽取流水线能否支撑用「从真源抽出的 fwd/bwd 图 + 逐张量 liveness」取代手写
> `cost_eval/layers/*.py` 的 `saves=[...]` 名册。
>
> **权威源**：`E:\97-codes\torch_parallel\mf-src-167\mindformers\`（commit `26354ff64`）。
> 本文所有行号以此快照为准。**不用** `E:\97-codes\torch_parallel\mindformers`（`c1f5e11f5`，内容不同）。
>
> 记法约定：**[RAN]** = 我实际跑了命令并看到该输出；**[INFERRED]** = 由读码推断，未直接跑出。

---

## 0. 环境与基线校验 [RAN]

```bash
cd /e/97-codes/torch_parallel/mf-src-167 && md5sum \
  mindformers/pynative/transformers/experimental_attention_variant/{csa.py,indexer.py} \
  && wc -l mindformers/pynative/transformers/experimental_attention_variant/*.py
```
```
81673be3ad3cdd2191e28dd000a13f0e *.../csa.py
8e18fee21c33507dd629c89f401986e6 *.../indexer.py
   831 csa.py      409 indexer.py      244 compressor.py
   300 deepseek_v4_hybrid_attention.py
```
→ 快照校验通过（md5 与行数均符合）。

```bash
cd /e/97-codes/torch_parallel/pynative-cost-evaluator && python -m pytest tests -q
```
```
1526 passed, 268 warnings in 23.73s
```
→ 基线 **1526 passed** 复现。

`MINDFORMERS_ROOT=E:\97-codes\torch_parallel\mf-src-167` → `default_mf_root()` 的两级探测
（`crosscheck.py:139-148`）正确下降到 `...\mf-src-167\mindformers`（含 `parallel_core`）。**[RAN]**

---

## 1. `crosscheck.py` 今天验什么 + 它对本模型的判决 [RAN]

命令：
```bash
python scratchpad/run_crosscheck_dsv4.py   # 载入 dsv4h_fused_pp4_recomp.yaml → build_llm_spec → validate_against_opdag
```
输出（关键部分）：
```
default_mf_root() -> E:\97-codes\torch_parallel\mf-src-167\mindformers
layer_specs: ['embedding', 'dsv4hyb_r0_dense', 'dsv4hyb_r4_moe', 'dsv4hyb_r128_moe', 'lm_head']
  embedding: 2 ops / dsv4hyb_r0_dense: 24 / dsv4hyb_r4_moe: 32 / dsv4hyb_r128_moe: 31 / lm_head: 5

available: True
ok       : True
覆盖层段=0（无）、未覆盖层=5、漂移=0、提取失败=0、层级pre-norm校验=1、pre-norm缺失=0

covered:            （空）
uncovered:   embedding / dsv4hyb_r0_dense / dsv4hyb_r4_moe / dsv4hyb_r128_moe / lm_head
             —— 全部 "无 opdag 提取源"
extraction_failures: （空）
layer_norm_checked: ['lm_head']
```

**判决：`ok=True` 是空判决（vacuous green）。** DSv4-Flash 的 5 个层段**全部**落进
`uncovered`，`covered=0`，一条也没真比。

根因（源忠实）：`crosscheck._split_decoder`（`crosscheck.py:436-445`）靠找
`op.output.name == "h1"` 切 attn/ffn 段。开了 mHC 后手写侧残差输出名带 `_xn` 后缀（实测
`add1 → h1_xn`、`moe_add → h2_xn`），探针失配 → `(None, None)` → MLA/MoE 两族 census
与第三族 layer_norms 全部跳过。**连 ln1/ln2 名册校验都没跑到**（`layer_norm_checked` 只有
`lm_head` 一项，来自 A5 的 `final_norm` 支）。

`crosscheck` **今天实际能验的**（在 DSv3 上，非本模型）**[RAN]**：
```
opdag_mla_census (DSv3 training_graph MLASelfAttention): {'linear': 5, 'norm': 2, 'attention': 1}
opdag_moe_census (FFNGroupedGEMM):                       {'linear_grouped': 2, 'activation': 1}
layer_norms_source_info: {input_layernorm:T, pre_mlp_layernorm:T, q_layernorm:T, kv_layernorm:T}
final_norm_source_info:  {final_norm: True}
mtp_norm_source_info:    {mtp_enorm: True, mtp_hnorm: True}
```
即：**只**比「重算子类别计数的逐类别 delta」（matmul/norm/linear_grouped/attention/activation），
且提取源是 `parallel_core/training_graph/` 的 **DSv3** MLA + MoE 专家核。它自己的 docstring
（`crosscheck.py:81-86`）已声明：**不**比每张量 `saves`/shape/dtype/workspace/生命周期。
故它**结构上无法**发现本次要修的那一类错误（`saves` 名册错、dtype 错、O(S²) 误声明为 saved）。

---

## 2. 最重要的结构事实：整条流水线是照 `parallel_core/training_graph/` 写的，DSv4-Flash 跑的是 `pynative/`

这是所有下游 gap 的共同根因，先单列。

### 2.1 spec 解析入口写死在 training_graph，且对本模型**静默解错模型** [RAN]

`module_resolver._SPEC_FILES`（`module_resolver.py:52-55`）只登记：
- `parallel_core/training_graph/base_models/gpt/gpt_layer_specs.py`
- `parallel_core/training_graph/base_models/gpt/moe_module_specs.py`

命令：`python scratchpad/probe1_resolve.py`（用真 yaml 的 flags：`is_dsv4_hybrid=True`、
`enable_hyper_connections=True`、`qk_layernorm=True`、`num_experts=8`）

```
### A) default _SPEC_FILES (parallel_core/training_graph) ###
TransformerLayer
  input_layernorm: 'Norm'
  self_attention:
    MLASelfAttentionConcatenated          <-- ！DSv3 MLA，不是 DSv4HybridSelfAttention
      linear_qkv: 'SequenceParallelLinear' / linear_qb / linear_kvb / core_attention: 'FlashAttention' ...
  pre_mlp_layernorm: 'Norm'
  mlp: MoELayer{experts: 'FFNGroupedGEMM', shared_experts: SharedExpertMLPInterleaved}
```

**这是静默错误，不是 fail-loud**：`is_dsv4_hybrid` / `enable_hyper_connections` 被
`module_resolver.py:103` 的 `kwargs = {k: v for k, v in flags.items() if k in param_names}`
按「不是本函数参数」**丢弃**（training_graph 版 `get_gpt_layer_local_spec` 没有这两个参数），
于是解出 DSv3 的 `TransformerLayer` + `MLASelfAttentionConcatenated`，且**返回 `HyperConnection`
之外的普通层**。任何以此为准的抽取都是在给一份真机从未跑过的结构建模。

### 2.2 pynative 有自己的 spec 树；补登记一个文件即可解通 [RAN]

真机路径是 `pynative/base_models/gpt/gpt_layer_specs.py:71 get_gpt_layer_local_spec`，
带 `is_dsv4_hybrid` / `enable_hyper_connections` 参数，`:105-113` 路由到
`get_dsv4_hybrid_module_spec(...)`——该函数在**另一个文件**
`pynative/base_models/gpt/experimental_attention_variant_module_specs.py:41`。

```
### B) _SPEC_FILES 指向 pynative（只 2 个文件） ###
FAIL: ValueError pynative/base_models/gpt/gpt_layer_specs.py:110
      Pass A 遇到未知调用 get_dsv4_hybrid_module_spec(...)（fail-loud）

### C) 再登记 experimental_attention_variant_module_specs.py ###
HyperConnectionTransformerLayer                       <-- 正确！mHC 层
  input_layernorm: 'Norm'
  self_attention:
    DSv4HybridSelfAttention
      q_layernorm: 'Norm' / kv_layernorm: 'Norm'
      linear_q_down_proj: 'Linear' / linear_q_up_proj: 'Linear' / linear_kv_proj: 'Linear'
      core_attention:
        CompressedSparseAttention
          compressor: Compressor{linear_wkv:'Linear', linear_wgate:'Linear', norm:'Norm'}
          indexer:    CSAIndexer{linear_wq_b:'Linear', linear_weights_proj:'Linear',
                                 compressor: Compressor{...}}
      linear_proj: 'Linear'
  pre_mlp_layernorm: 'Norm'
  mlp: MoELayer
```

**结论**：Pass A（模块树解析）对 dsv4_hybrid **完全够用**，只差把 pynative 的 3 个 spec 文件
登记进去（`_SPEC_FILES` 需要参数化，不能全局换掉——现存 DSv3 测试依赖 training_graph 路径）。
工作量 **S**。

### 2.3 `LEAF_OPTYPE` 不含 pynative 的 `Linear` 叶子 [RAN]

`module_resolver.LEAF_OPTYPE`（`:37-44`）只有 `ColumnParallelLinear` / `RowParallelLinear` /
`SequenceParallelLinear` / `FlashAttention` / `Norm` / `Identity`。pynative spec 用的是
`mindformers.pynative.layers.linear.Linear`（见
`experimental_attention_variant_module_specs.py:22,88-90,113-120`）。

```bash
python scratchpad/probe2_extract.py
```
```
### extract_cell(DSv4HybridSelfAttention) → FAIL: 叶子类 'Linear' 不在 LEAF_OPTYPE
    （self.linear_q_down_proj @ deepseek_v4_hybrid_attention.py）—— fail-loud
### extract_cell(CompressedSparseAttention) → FAIL: 'Linear' ... self.linear_wq_b @ indexer.py
### extract_cell(CSAIndexer)                → FAIL: 'Linear' ... self.linear_wq_b @ indexer.py
### extract_cell(Compressor)                → FAIL: 'Linear' ... self.linear_wkv @ compressor.py
```
一行表项修复（`"Linear": "MatMul"`）。工作量 **S**。

### 2.4 `init_binder` 只认「类实例化」惯用法；pynative 用「裸函数别名」惯用法 —— 这是最大的单一 gap [RAN]

`init_binder.bind_init`（`init_binder.py:95-96`）要求
`isinstance(stmt.value, ast.Call)`，再按**类名**查 `_CLS2OP`（`Reshape` / `Cast` / `Mul` / `Concat` / ...）。

两个惯用法的实测对比（同一份快照）：
```bash
grep -n "self.shape\s*=\|self.cast\s*=\|self.reshape\s*=" \
  parallel_core/training_graph/transformer/multi_latent_attention.py
# 160: self.shape = Shape()      161: self.reshape = Reshape()     165: self.cast = Cast()   <-- 类实例化，_CLS2OP 认得

grep -n "self.shape\s*=\|self.cast\s*=\|self.reshape\s*=" \
  pynative/transformers/multi_latent_attention.py
# 127: self.shape = ops.shape    128: self.reshape = mint.reshape  131: self.cast = ops.cast <-- 裸别名，不是 Call → 完全不绑
```
pynative 侧这是**成文约定**，不是偶然：`csa.py:629-630` 注释「Alias the non-trivial mint ops
used in construct/forward per the fine-grained-recompute convention (RFC §3.1 #9)」。

量化（命令：`python scratchpad/probe4_census.py`，逐类统计 `__init__` 里 `self.X=` 的可绑性
与 construct/helper 里 `self.X(...)` 的调用点）：

| 类（文件） | `self.*` 赋值 | 被 `_CLS2OP` 绑住 | `build_module` | **裸别名** | 未绑调用点 |
|---|---|---|---|---|---|
| `DSv4HybridSelfAttention` (deepseek_v4_hybrid_attention.py) | 19 | 1 | 7 | 6 | `split/cat/unsqueeze/bmm` + `shape/reshape/cast/permute/rotary_pos_emb`（后 4 个来自跨文件基类）= 15 处 |
| `CompressedSparseAttention` (csa.py) | 28 | **0** | 0 | 11 | `reshape×3 / permute×4 / cat×2 / unsqueeze×3 / where×2` + `indexer/compressor/unfused_indexer_loss` |
| `CSAIndexer` (indexer.py) | 33 | 1 | 3 | 20 | `reshape×7 / cast×8 / permute×4 / split / cat / bmm / relu / sum / topk / unsqueeze` + `hadamard / rotary_pos_emb` = 26 处 |
| `UnfusedCSAIndexerLoss` (indexer.py) | 19 | **0** | 0 | 15 | `full×5 / sum×4 / where×3 / permute×2 / softmax×2 / log×2 / cast×2 / max×2 / reshape×2 / scatter / matmul / triu / unsqueeze` = 27 处 |
| `Compressor` (compressor.py) | 23 | 1 | 3 | 10 | `reshape×3 / cat×2 / unsqueeze×2 / chunk / roll / softmax / split / squeeze` + `hadamard / rotary_pos_emb` |
| `MultiLatentAttention` (pynative/…/multi_latent_attention.py) | 29 | **0** | 2 | 17 | `shape×2 / reshape×4 / transpose×3` |

即 **dsv4_hybrid 链上几乎每一个结构/逐元素算子都是不可见的**。这不是「少几个表项」，是
`init_binder` 需要新增一条「裸 `mint.*` / `ops.*` 别名 → op 类型」的绑定通路（新表，约
30-40 个 pynative 函数名）。工作量 **M**。

### 2.5 跨文件继承（MRO）不支持 [RAN + INFERRED]

`DSv4HybridSelfAttention(MultiLatentAttention)`——基类在**另一个文件**
（`pynative/transformers/multi_latent_attention.py:60`）。`extractor._init_classes` /
`_find_class`（`extractor.py:37-41, 74-98`）都只在**单个已 parse 的 tree** 内找类，
故基类 `__init__` 的绑定（含 `self.shape`/`self.reshape`/`self.cast`/`self.permute`，
`multi_latent_attention.py:127-131`）永远收不到。**[RAN]**：

```
### DSv4HybridSelfAttention → FAIL: construct 调用了未绑定的 self.shape(...)
    （deepseek_v4_hybrid_attention.py:233）:Pass B(init_binder)未覆盖此名、
     且非本类内部方法/Morph 别名,fail-loud
```
**[INFERRED]** DSv3 路径从未撞上这条，是因为 training_graph 的 `MLASelfAttention` 与其基类
`MultiLatentAttention` 恰在**同一个文件**里。工作量 **M**（要跟 import 解析跨文件基类）。

---

## 3. 逐组件抽取状态

命令（全部经 monkeypatch 打通 §2.2/§2.3 两个 S 级 gap 后跑，无仓库改动）：
`python scratchpad/probe3.py`、`probe5_rest.py`、`probe6_shapes_fn.py`

### 3.1 dsv4_hybrid 注意力 —— **完全没抽出来（0 个组件成功）** [RAN]

| Cell | 源 | 结果 | 卡在哪 |
|---|---|---|---|
| `DSv4HybridSelfAttention` | `deepseek_v4_hybrid_attention.py:48` | **FAIL** | `self.shape(...)` 未绑定 @ `:233` —— 基类 `MultiLatentAttention` 在**另一个文件**，跨文件 MRO 不支持（§2.5） |
| `CompressedSparseAttention` | `csa.py:536` | **FAIL** | 递归进 `Compressor` 后死在 `compressor.py:190` `if sq < ratio:`；单独跑 CSA 时先死在 `csa.py:672` 三元 `self.enable_compress`（`__init__` 派生量，不在 config_flags） |
| `CSAIndexer` | `indexer.py:66` | **fused: OK -> 0 nodes（静默空图）** / **unfused: FAIL** | fused 支整块在 `with _no_grad():`（`indexer.py:220`）里 → walker 无 `ast.With` 处理器 → **整块静默丢，连 opaque_calls 都不记**。unfused 支死在 `self.cast(...)` 未绑定 @ `:234`（裸别名） |
| `Compressor` | `compressor.py` | **FAIL** | `compressor.py:190` `if sq < ratio:` —— `sq` 来自 `sq,b,_ = x.shape`（运行期标量）、`ratio = self.compress_ratio`（局部变量绑 self 属性），两者 walker 都不求值 → `_UNDECIDED`；且非「纯 raise 守卫」（body 是 `return None`）→ fail-loud |
| `UnfusedCSAIndexerLoss` | `indexer.py:294` | 未跑通（同 CSAIndexer 的裸别名问题，27 个未绑调用点） | — |
| `FusedSparseFlashMla` / `FusedSparseFlashMlaWithIndexerLoss` / `_IndexerLossAutoScaler` | `csa.py:73` / `:178` / `indexer.py:268` | **FAIL（结构性）** | 见 §4：`_Function` 只有 `forward`/`backward`，没有 `construct`/`__init__` |
| `unfused_compressed_sparse_attn` | `csa.py:464-533` | **不产节点** | 模块级**自由函数**，非 `self.<method>` → `_handle_call` 终端 fallthrough → 只进 `opaque_calls`。这正是任务描述里「11-op 注意力链」的实体，内部 ~24 个 mint/ops 调用全不可见 |

实测输出摘录：

```
### [FUSED] CSAIndexer
OK -> 0 nodes / 0 edges
  opaque_calls (0):
  derive_saves -> 0
### [FUSED] CompressedSparseAttention
FAIL: construct 的 if 条件无法由 config 判定（compressor.py:190）:`sq < ratio`
### [FUSED] DSv4HybridSelfAttention
FAIL: construct 调用了未绑定的 self.shape(...)（deepseek_v4_hybrid_attention.py:233）
### [UNFUSED] CSAIndexer
FAIL: construct 调用了未绑定的 self.cast(...)（indexer.py:234）
```

两条 A/B 路径（`apply_dsa_kernel_fusion=True|False`）**都**不通。

> **最危险的一条**：`CSAIndexer` fused 支返回 `ok, 0 nodes` —— 这不是 fail-loud，是**假成功**。
> 如果照现状把 opdag 接进数字链路，这一段会贡献 **0 字节**而不报错。

### 3.2 MoE —— 只有 DSv3/training_graph 的专家核抽得出；pynative 侧全 FAIL [RAN]

| 目标 | 结果 |
|---|---|
| `parallel_core/training_graph/.../moe/ffn.py FFNGroupedGEMM`（crosscheck 用的那个） | **OK -> 10 nodes**：`View, Cast x3, View x2, GroupedMatMul, Activation, GroupedMatMul, Cast`；3 条 opaque（`token_dispatcher.token_permutation` / `token_unpermutation` / 一个 reshape） |
| `pynative/transformers/moe/moe_layer.py MoELayer` | **FAIL**：`self.router(...)` 未绑定 @ `moe_layer.py:122` |
| `pynative/transformers/moe/router.py TopKRouter` | **FAIL**：`self.reshape(...)` 未绑定 @ `router.py:353`（裸别名） |
| `pynative/transformers/moe/experts.py GroupedMLP` | **FAIL**：`if isinstance(tokens, DTensor)` 不可判定 @ `experts.py:185` |
| pynative MoE spec 树 | `get_moe_module_spec` 返回 `ModuleSpec(module=MoELayer)`，**submodules 全空**（`pynative/base_models/gpt/moe_module_specs.py:34-37`）→ Pass A 给不出 router/experts/shared_experts 的叶子类型，全靠 `MoELayer.__init__` 按 config 自建 |

router / dispatch / combine 在 crosscheck 里本来就被声明为 **opaque 边界、不比**（`crosscheck.py:34-38`）。实测 `ops.moe_token_permute` / `ops.moe_token_unpermute` 各 1 处调用，均落 opaque。

### 3.3 mHC hyper-connection 残差 —— FAIL [RAN]

```
### mHC HyperConnectionTransformerLayer @ pynative/transformers/transformer_layer.py
FAIL: extractor: build_module(submodules.pre_cross_attn_layernorm) 在 spec 里无对应子模块
      （self.pre_cross_attn_layernorm @ transformer_layer.py）—— fail-loud
```

根因：`extractor._named_module_binds` 用 `ast.walk(init)` **无视 `__init__` 里的 config 分支**，把每一个 `self.X = build_module(submodules.Y, ...)` 都当必存在；pynative `TransformerLayerSubmodules` 真机只填了 4 个槽（`input_layernorm`/`self_attention`/`pre_mlp_layernorm`/`mlp`，见 `pynative/base_models/gpt/gpt_layer_specs.py:105-113`），`pre_cross_attn_layernorm` 这类可选槽为 `None` → 报「spec 里无对应子模块」。所以 **mHC 层的残差/gating/sinkhorn 一个节点都没抽出**。`hc_mult=4` 的 4 路残差流、fused mHC 变体、`attn_hc_*`/`ffn_hc_*` 那 6 个手写 op 在源侧**零覆盖**。

### 3.4 embedding / lm_head / loss [RAN]

| 段 | 结果 |
|---|---|
| `gpt_segments.extract_embedding`（training_graph `VocabParallelEmbedding`） | **OK -> 7 nodes**（`View, Activation, Elementwise x2, Gather, Elementwise, View`），3 条 opaque（含 `ops.AllReduce(group=...)(...)` @ `layers.py:182`，设计上交给 `comm_probe`） |
| `gpt_segments.extract_loss`（training_graph `CrossEntropyLoss`） | **FAIL**：`if self.enable_force_redistribute` 不可判定 @ `loss_func.py:309` |
| `gpt_segments.head_segment_dag()` | **不是抽出来的，是合成的**（4 个手写 `OpNode`，src 钉到 `gpt_model.py:503/505/506/507`）。`verify_gpt_order(MF)` 在快照上**通过**，返回 `[reshape, _preprocess_input_labels_and_masks, language_model, mtp, shared_embedding_or_output_weight, output_layer, transpose, morphed_reshape_logits, cast, compute_language_model_loss]` |
| **pynative** `CrossEntropyLoss` @ `pynative/loss/loss.py:213`（真机跑的那个） | **FAIL**：`if self._tp_group is not None` 不可判定 @ `loss.py:328` |
| **pynative** `VocabEmbedding` | **FAIL**：三元 `isinstance(self.weight, DTensor)` 不可判定 @ `vocab_embedding.py:81` |
| **pynative** `LanguageModelEmbedding` | **FAIL**：`self.word_embeddings(...)` 未绑定 @ `language_model_embedding.py:115` |

即：embedding/lm_head/loss 这三段今天**只在 training_graph 侧有部分覆盖**（embedding 抽得出、head 是合成、loss 抽不出），pynative 侧（真机路径）**三段全 FAIL**。chunked CE 未触及。

### 3.5 MTP —— FAIL [RAN]

```
### pynative MultiTokenPredictionLayer
FAIL: extractor: multi_token_prediction.py self.enorm 的 build_module 首参不是 submodules.<字段>（fail-loud）
```

（本次 yaml `num_nextn_predict_layers: 0`，故 MTP 不在关键路径；但站点配置 =1，届时是硬阻塞。）`crosscheck` 侧对 MTP 只做 `enorm`/`hnorm` 的**存在性**定向 AST 读（`crosscheck.py:422-432`，实测返回 `{mtp_enorm: True, mtp_hnorm: True}`），不抽图。

### 3.6 norms / RoPE

- **norms**：`Norm` 叶子在 Pass A 正常解出（dsv4_hybrid 树里 `q_layernorm`/`kv_layernorm`/compressor 的 `norm` 都解成 `'Norm'`，见 §2.2 输出），`PIN["Norm"]={"inputs":[0]}` 且带 `ln_compute_dtype` fp32 覆盖（`bprop_rules.py:56-58`）——**这一块是现成可用的**。但层级 `input_layernorm`/`pre_mlp_layernorm` 的**节点**要靠抽 `HyperConnectionTransformerLayer` 才拿到，而那个 FAIL（§3.3）。
- **RoPE**：`ApplyRotaryPosEmb` 在 `init_binder._CLS2OP` 里有表项 → `Elementwise(linear=True, rope=True)`（`init_binder.py:22`）。dsv4 侧 `self.apply_rotary_emb = ApplyRotaryPosEmb(config)`（`deepseek_v4_hybrid_attention.py:80`）是 Call 形式，**能绑上**。但 `self.rotary_pos_emb` 是 `RotaryEmbedding(...)`/`YarnRotaryEmbedding(...)` 实例（`:81`，由 `_build_rotary_pos_emb` 返回）→ 不在 `_CLS2OP`、不是 `build_module` → `self.rotary_pos_emb(sq)` 调用点未绑定。inverse-RoPE（`_apply_forward_rope(..., inverse=True)`）走内部方法内联，机制上支持。

---

## 4. 自定义 `_Function` 与 `ctx.save_for_backward(...)` [RAN]

**walker 看不见它们**：

```
walk_construct(FusedSparseFlashMla)                FAIL: class FusedSparseFlashMla(及其基类)缺少 construct 方法
extract_cell(FusedSparseFlashMla)                  FAIL: FusedSparseFlashMla 及其基类均无 __init__
walk_construct(FusedSparseFlashMlaWithIndexerLoss) FAIL: 同上
extract_cell(FusedSparseFlashMlaWithIndexerLoss)   FAIL: 同上
```

入口写死在 `construct`（`construct_walker.py:990` `walker._lookup_method("construct")`），`extract_cell` 更早在 `_init_classes` 空时就 fail（`extractor.py:442-443`）。

**但 `save_for_backward` 的实参名单是逐字可取的** —— 我用 ~10 行裸 AST 就读出来了：

```
FusedSparseFlashMla                 csa.py:113  n=8
  ['query','ori_kv','cmp_kv','sparse_indices','cmp_residual','sinks','output','softmax_lse']
FusedSparseFlashMlaWithIndexerLoss  csa.py:224  n=11
  ['query','ori_kv','cmp_kv','sparse_indices','query_index','key_index','weights',
   'cmp_residual','sinks','output','softmax_lse']
```

（模式固定：`ctx.save_for_backward(*[tensor for tensor in (<名单>) if tensor is not None])`，两处一模一样，`ListComp` 的 `generators[0].iter` 就是那个元组。）

**与手写名册的实测差异**（手写侧 `sparse_attn` op 的 `saves`，取自 `build_llm_spec(...)` 实跑 dump）：

```
手写 9 项: q_hnorm_fp32, kv_a_out, compressed_kv, core_out, idx_query, idx_key,
           idx_weights, cmp_residual, softmax_lse
源侧 11 项（fused+indexer 支，csa.py:224）
对应关系: query->q_hnorm_fp32  ori_kv->kv_a_out  cmp_kv->compressed_kv  output->core_out
          query_index->idx_query  key_index->idx_key  weights->idx_weights
          cmp_residual  softmax_lse                                  = 9 项对上
**手写侧缺**: sparse_indices（[b,sq,topk] int32）、sinks（[n_heads] fp32）
```

→ 这是「源里写着、手写没抄」的**具体两项**，也是「融合算子的 saved set 应当逐字从源抽而不是猜」这条主张的直接证据。工作量：把这两个 `_Function` 的 forward 建成一个「显式 saved-set 节点」是 **S**（模式固定、两处；`backward` 里 `next(saved_tensors)` 的消费序也能同法读出）。

---

## 5. `ops.stop_gradient(...)` 可见性 [RAN]

真源 6 处：`csa.py:665,666`（fused 支 x/qr detach）、`:764,765`（unfused 支）、`:794,795`（unfused indexer KL loss 的 query / compressed_kv detach）。

最小复现（toy）：

```python
xd = ops.stop_gradient(x)
z  = self.mul(xd, y)
```

walker 输出：

```
#1 Elementwise ins=['xd:?:bf16', 'y:?:bf16'] out=z:?:bf16
opaque_calls: [{'src': 'toy.py:7', 'expr': 'ops.stop_gradient(x)'}]
```

逐条结论：

1. **调用文本能看见** —— 落进 `opaque_calls`（`construct_walker.py:411-413`），不是完全静默。
2. **detach 语义完全没建模** —— 没有节点、没有 `detached` 标记、没有任何 attr。
3. **更糟：数据流被静默切断** —— `xd` 没进 SSA/producer（`_handle_assign` 走 `_handle_call`，opaque 分支不登记 target），下游 `self.mul(xd, y)` 拿到的是**占位 ref `xd:?:bf16` 且无边**，到 `x` 的边**丢了**。

这对新设计是**方向相反**的：`liveness/graph.py:60,193-195` 要的是「保留 producer->consumer 边 + 在派生张量上打 `detached=True`」，而现状是「丢边 + 不打标」。好消息：`FREE_CALL_MAP`（`construct_walker.py:61-64`）就是为这类具名自由函数准备的挂钩，加一条 `"ops.stop_gradient": ("Detach", {"detach": True})` + 一条 PIN 表项即可，工作量 **S**。

---

## 6. 控制流 / config 分支处理 [RAN]

### 6.1 能判定什么

`_eval_test`（`construct_walker.py:724-749`）只认三类：`self.<flag>` / `self.config.<flag>`（`_config_flag_name`，`:156-169`）、形参的 present/None、字面量比较。剪枝上下文下**不可判定即 fail-loud**（`:279-291`），唯一例外是「纯 raise 守卫」。

**不能判定**（本模型实测全部撞上）：

| 形态 | 实例 | 后果 |
|---|---|---|
| `__init__` 派生的布尔 | `self.enable_compress`（`csa.py:672`）、`self.enable_indexer`、`self.is_tnd` | fail-loud。**`init_dims` 已经在静态求值 `__init__` 了，但结果只回填 `dag.dims_ctx`（`extractor.py:519`），没喂给 `walker_flags`（`:486-494`）** —— 这是最划算的一处接线 |
| 运行期形状标量 | `if sq < ratio`（`compressor.py:190`）、`if bs > 1`（embedding SP 支） | fail-loud |
| 局部变量绑 self 属性 | `ratio = self.compress_ratio` 后 `sq < ratio` | 局部变量不进 env |
| `isinstance(...)` | `isinstance(self.weight, DTensor)`（`vocab_embedding.py:81`）、`isinstance(tokens, DTensor)`（`experts.py:185`） | fail-loud |
| `hasattr(...)` | `hasattr(attn_sink, "to_local")`（`csa.py:686`, `:468`） | fail-loud（在 if 里） |
| `x is not None` 且 x 非形参 | `self._tp_group is not None`（`loss.py:328`）、`compressed_kv is not None`（`csa.py:674`） | fail-loud |
| 三级 config 路径 | `self.config.moe_config.X` | `_config_flag_name` 只支持 2 级 -> `_UNDECIDED` |
| 逐层 `compress_ratios[i]` | `config.csa_compress_ratios[layer_number]`（`deepseek_v4_hybrid_attention.py:69`）-> Subscript | 不是 Attribute -> 解不出；须**按层实例化**注入 `compress_ratio ∈ {0,4,128}` |

**注入 config 值确实有效** [RAN]：我把 `enable_compress/enable_indexer/is_tnd/compress_ratio/training/...` 当普通 `config_flags` 注入后，`csa.py:672` 那条三元就过了，错误前移到 `compressor.py:190`。所以「派生布尔注入」这条路是通的，只是今天要手喂。

### 6.2 整类语句被**静默丢弃**（无节点、无 opaque 记录）[RAN]

`walk_stmt`（`construct_walker.py:251-276`）只处理 `Assign/Expr/If/Return/Raise`。实测：

```
with           -> 0 nodes, opaque=0   <== SILENT DROP
for            -> 0 nodes, opaque=0   <== SILENT DROP
while          -> 0 nodes, opaque=0   <== SILENT DROP
try            -> 0 nodes, opaque=0   <== SILENT DROP
binop (a=x*y)  -> 0 nodes, opaque=0   <== SILENT DROP
baseline(if)   -> 1 nodes
```

dsv4_hybrid 链里的实际分布：`{'IfExp': 12, 'ListComp': 2, 'With': 1}`（`With` 就是 `indexer.py:220` 的 `_no_grad()`，直接导致 §3.1 的 0-node 假成功）。

**RHS 不受支持的赋值：34 处**，其中恰好包含这次要修的那些张量：

```
deepseek_v4_hybrid_attention.py:245 [BinOp] q = q * mint.rsqrt(mint.mean(q*q, dim=-1, keepdim=True) + eps)
                                            <- 即手写侧的 q_hnorm_fp32
indexer.py:350  [BinOp] attention_scores = self.matmul(query, key) * self.softmax_scale
                                            <- O(S·S/r) fp32（手写侧 ukl1）
indexer.py:387  [BinOp] attention_scores = attention_scores / self.max(self.sum(...), 1e-...)
compressor.py:209 [BinOp] score_f32 = score.astype(mstype.float32) + self.reshape(ape, (1, ratio, 1, -1))
                                            <- fp32 softmax 输入（SNAPSHOT.md 记的真机 OOM traceback 命中处）
csa.py:779  [Compare] future = cm >= positions // ratio
csa.py:810  [Compare] valid  = topk_indices_compressed < self.unsqueeze(n_valid_per_pos, 0)
compressor.py:198/199/233 [Subscript] kv = kv[:cutoff] / score = score[:cutoff] / freqs = freqs[...]
```

即：**「误声明为 saved 的两个 O(S²) fp32 张量」这类问题所涉及的张量，在源侧恰恰都住在 walker 今天不看的 RHS 形态里。** 这是把 saves 从手写换成源抽的**核心技术债**。

---

## 7. shape 解析 -> 字节（`consumer.py`）[RAN]

### 7.1 抽出来的裸 DAG：**全部 unresolved**

`walker._emit` 恒写 `?` 作 shape 段（`construct_walker.py:900`），`extract_cell` **不**调 `infer_shapes`。故直接喂 `consumer`：

```
-- embedding (VocabParallelEmbedding): RAW dag
   total_bytes=0  per_save=0  unresolved=4   （input_, masked_input__i0, output_parallel__i0, input_mask__i0，全 '?'）
-- FFNGroupedGEMM: RAW dag
   total_bytes=0  per_save=0  unresolved=6
```

`infer_shapes(dag, input_shapes, dims_ctx)` 需要**手工给输入种子**（`shape_infer.py:260`），不是自动的。

### 7.2 给了种子之后：DSv3 参考路径**全解** [RAN]

```
=== DSv3 training_graph MLASelfAttention（infer_shapes(dag, {"x":"S·B·H"})）===
nodes=26 total_bytes=98566144 (94.0 MiB)
  OK ('x','S·B·H','bf16',14680064)
  OK ('q_compressed__i0','S·B·q_lora_rank','fp32',25165824)      <- norm 存 fp32，机制在
  OK ('kv_compressed__i0','S·B·kv_lora_rank','fp32',8388608)
  OK ('query','S·B·n_heads·(qk_head_dim+qk_pos_emb_head_dim)','bf16',12582912)
  OK ('key', 同上, 12582912)  OK ('value','S·B·n_heads·v_head_dim',12582912)
  OK ('attn_out','S·B·(n_heads·v_head_dim)','bf16',12582912)
  unresolved: []                                                <- 0 未解析

=== FFNGroupedGEMM（种子 tokens/dispatched_input/w1/w2）===
nodes=10 total_bytes=247463936 (236.0 MiB)
  OK dispatched_input__i0 E·cap·H 58.7MB / w1 / fc1_output__i1 / intermediate_parallel__i1 / w2
  UNRESOLVED ('tokens_per_expert__i0','?','unknown-symbol-or-?')   <- 未种子的外部输入
```

**结论**：符号 shape -> 字节这一段（`sym_shape` + `shape_infer` + `consumer`）**质量不错、可复用**，`unresolved` 清单机制也如设计所说「不杜撰」。它的问题只有两个：

1. `_SYM2FIELD`（`consumer.py:19-34`）只有 14 个符号，**没有** dsv4 需要的 `index_n_heads` / `index_head_dim` / `index_topk` / `compress_ratio` / `csa_window_size` / `o_groups` / `o_lora_rank` / `hc_mult` —— 这些一律会落 `unresolved`（**S**：加表项）。
2. **权重被当成 activation save 计进去了**：`w1`(58.7MB)/`w2`(29.4MB) 占 FFNGroupedGEMM「236 MiB」里的 88 MiB。DAG 没有 `is_weight` 概念（见 §9）。

---

## 8. `bprop_rules.PIN` 覆盖 [RAN]

```
PIN keys (12): Activation BMM Cast Dropout Elementwise FlashAttention Gather
               GroupedMatMul MatMul Norm Softmax View
LEAF_OPTYPE 的 op 类型里不在 PIN 的: ['Identity']
walker 可发射的 op 类型里不在 PIN 的:  ['SubCell']
```

`derive_saves` 对未知 op 类型 **fail-loud**（`bprop_rules.py:44`），所以每个新 op 类型都必须补表项。

dsv4_hybrid 需要的**原语词表实测**（`self.X = <裸 mint/ops 别名>` 且被调用者）：**39 个不同原语、196 个调用点**。按 bprop 语义归类：

| 归类 | 原语（括号内为调用次数） | PIN 现状 |
|---|---|---|
| -> `View`（纯视图，反向不存） | reshape(45) permute(14) split(7) cat(8) unsqueeze(10) squeeze transpose(3) chunk(2) tile roll | 有 |
| -> `Cast` | cast(29) | 有 |
| -> `Elementwise` | add(3) mul(5) div(2) sum(8) mean log(2) sqrt | 有 |
| -> `Softmax` | softmax(4) | 有 |
| -> `Activation` | relu sigmoid softplus | 有 |
| -> `MatMul`/`BMM` | matmul bmm(2) linear | 有 |
| -> `Gather` | gather(3) | 有 |
| **需要新规则** | **topk(4)**（须存 indices）、**where(5)**（须存 cond mask）、**scatter**、**argsort(2)**、**histc(4)**、**cumsum**、**maximum(2)**、arange/full(5)/zeros/ones_like/triu（常量产出，不存） | **全缺** |
| **融合/opaque 内核** | `npu_sparse_flash_mla`(2) `npu_lightning_indexer` `unfused_compressed_sparse_attn` `ops.moe_token_permute`/`unpermute` `_prepare_sparse_flash_mla` `get_{,doc_}{window,compress}_topk_idxs`(4) `ops.stop_gradient`(6) | **全缺** |

即 PIN 侧要补的是 **~8 条真新 bprop 规则 + ~8 个融合/opaque 内核条目 + Identity/SubCell 两条**。其中融合内核的 saved-set **不该靠 PIN 猜**，应走 §4 的「从 `save_for_backward` 逐字读」通路。工作量 **M**（规则本身小，但 topk/where/scatter 的 saved 语义要逐个核对源）。

---

## 9. 抽出的图 vs `liveness/graph.py` 需要的输入 —— 接口该长什么样

`liveness.build_layer_graph(resolved_layer)` today 吃的是
`ResolvedLayer(layer_id, layer_type, ops)`，每个 `ResolvedOp`（`shape_eval.py:187-196`）带
`name / type / inputs / output / params / saves / workspace_bytes / collectives / bwd_scratch_bytes`，
每个 `ResolvedTensor`（`shape_eval.py:62-78`）带
`name / local_numel / dtype_bytes / is_weight / is_expert / pin_under_recompute / dim0 / detached`。

`OpDAG` 侧只有 `OpNode(id, op, src, module, ins: list["name:shape:dtype"], out, attrs)` + `edges`。
逐项对账：

| liveness 需要 | OpDAG 现状 | 缺口 | 量级 |
|---|---|---|---|
| `op.params`（**权重 = 梯度根**，`graph.py:205,220` 用 `bool(op.params)` 决定 `has_bwd`） | **无** —— 权重只是普通 `ins` ref | 必须区分权重操作数（`build_module` 线性的 W、`Parameter(...)` 如 `attn_sink`/`q_rms_gamma`/`linear_o_group_proj`） | **M** |
| `tensor.is_weight` | **无** | 实测后果：FFNGroupedGEMM「236 MiB」里 `w1`(58.7MB)+`w2`(29.4MB)=88 MiB 是权重被当激活 save 计入 | **M** |
| `tensor.detached`（`graph.py:60,193-195`，grad 可达性的切断点） | **无**，且 `stop_gradient` 处**边被切断**（§5） | 反向：要保边 + 打标 | **S** |
| `op.saves`（**逐 op**；`graph.py:223-231` 建 `saved_by[name] -> {op idx}`） | `derive_saves(dag)` 返回**全 DAG 按名去重的扁平表**（`bprop_rules.py:40,59` `saves.setdefault`），只留一个 `op_id` | 改成逐节点 saves 列表（保留全局去重作可选视图） | **S** |
| `tensor.local_numel` / `dtype_bytes`（已按 TP/EP shard 除过） | 符号 shape 串；`consumer` 能算**全局** elems，但**无 shard 标注**（`shape_eval.resolve_tensor` 靠 `TensorRef.shard` + `pm` 做整除） | 需要给抽出的张量补 placement/shard 轴，或在 adapter 里按 op 类型推 | **M** |
| `op.workspace_bytes` / `bwd_scratch_bytes` | **无概念** | 这两项本质上是 kernel 实现细节，源码里读不出来 → 建议**继续保留手写/profiler 标定**，不要求从图里推 | — |
| `layer_id` / `layer_type` / 一层完整 op 序（attn+ffn+residual 拼一起） | 每个 `extract_cell` 只产**单个 Cell** 的图 | 需要用现成的 `recurse=True` + `_inline_subcell`（`construct_walker.py:632-698`）把 `HyperConnectionTransformerLayer → self_attention → CSA → {indexer, compressor}` + `MoELayer` 内联成**一张层图**。该机制已存在且设计正确（id 续编、形参重映射、跨界连边），是这条路上少见的「现成能用」的部分 | **M** |
| 反向节点 | `PIN` + `derive_saves` 给的正是「每个 fwd op → 它反向要读哪些张量」 | **概念上完全对上** —— `BwdNode.reads = fwd[i].saves` 就是 PIN 的语义 | — |

**建议的接口形状**：**不要改 `liveness/`**，写一个 `cost_eval/opdag/to_resolved.py` 适配器：

```
OpDAG(已 infer_shapes) + DimTable + ParallelModel
   → ResolvedLayer(layer_id, layer_type, ops=tuple[ResolvedOp])
```

这样 `liveness/simulate.py`（已实现 full/select 重算的图变换）与 `schedule.py` 事件走查
**一行不用动**，且新旧两条 spec 来源可以在同一个 liveness 上做 A/B 对照（正是交叉校验想要的）。
适配器要负责：ref 串 → `ResolvedTensor`（含 `is_weight`/`detached`/shard 后 numel）、
逐 op saves、以及把 `opaque_calls` 里**未在白名单内**的条目 fail-loud（否则又变成静默丢算子）。

---

## 10. 优先级 gap 清单

「阻塞？」= 不做这条，DSv4-Flash 的显存**根本算不出来**（而不是算得不够准）。

### P0 —— 走查根本走不通（全部阻塞）

| # | gap | 缺什么 | 量级 | 阻塞 |
|---|---|---|---|---|
| 1 | **未处理语句类默认 fail-loud** | `walk_stmt` 对 `With/For/While/Try` 与 `BinOp/Compare/UnaryOp/Subscript` RHS **静默返回**（实测 0 nodes / 0 opaque）。**先把这条改成 fail-loud 或强制记 opaque**，否则后面每一步都在假绿上做 | **S** | 是（今天的 `CSAIndexer → 0 nodes` 就是它） |
| 2 | **pynative 裸别名绑定** | `init_binder` 只认类实例化；pynative 是 `self.reshape = mint.reshape` 这类裸别名。实测 dsv4 链上 **39 个原语 / 196 个调用点**全不可见 | **M** | 是 |
| 3 | **跨文件 MRO 基类绑定** | `DSv4HybridSelfAttention` 的基类 `MultiLatentAttention` 在另一文件，`self.shape/reshape/cast/permute` 收不到 | **M** | 是 |
| 4 | **`ast.With` 走查（含 `_no_grad()` 语义）** | `indexer.py:220` 的 `with _no_grad():` 包住整个 fused indexer 支。不只是要走进去——`_no_grad` 本身就是**整块 detach** 的信号，应当把块内产物标 `detached` | **S** | 是 |
| 5 | **BinOp/Compare/Subscript RHS 建节点** | 34 处。**这次要修的张量恰好全在这里**：`q_hnorm_fp32`(`deepseek_v4_hybrid_attention.py:245`)、O(S·S/r) fp32 `attention_scores`(`indexer.py:350,387`)、fp32 softmax 输入 `score_f32`(`compressor.py:209`)、O(S·S/r) bool mask(`csa.py:779,810`)、切片(`compressor.py:198,199,233`) | **M** | 是 |
| 6 | **`__init__` 派生布尔喂给 walker** | `self.enable_compress/enable_indexer/is_tnd` 等。`init_dims` **已经**在静态求值 `__init__`，只是结果没进 `walker_flags`（`extractor.py:486-494` vs `:519`）。实测手工注入即可过 `csa.py:672` | **S** | 是 |
| 7 | **运行期形状标量 + 局部变量参与 if 判定** | `compressor.py:190 if sq < ratio:`（`sq` 来自 `x.shape` 解包、`ratio = self.compress_ratio` 是局部变量）。给定 S=4096 是**静态可判**的，只是 env 里没有 | **S–M** | 是 |
| 8 | **`build_module` 可选槽为 `None` 时按 config 剪枝** | `_named_module_binds` 用 `ast.walk(init)` 无视 `__init__` 分支 → mHC 层在 `pre_cross_attn_layernorm` 处 fail-loud，**整个 mHC 残差零覆盖** | **S** | 是（mHC 是 `hc_mult=4` 的 4 路残差，量很大） |
| 9 | **`isinstance` / `hasattr` / `<self attr> is not None` 判定** | pynative 到处用 `isinstance(x, DTensor)` / `hasattr(t,"to_local")` 做 DTensor 分支（`vocab_embedding.py:81`、`experts.py:185`、`csa.py:468,686`、`loss.py:328`）。这些是**部署形态**分支，可按 config 注入判定 | **S** | 是（pynative embedding/loss/experts 全卡这里） |

### P1 —— 走通之后 saves 才对（全部阻塞正确性）

| # | gap | 缺什么 | 量级 | 阻塞 |
|---|---|---|---|---|
| 10 | **`_Function.forward` 入口 + `ctx.save_for_backward` 逐字读** | walker 入口写死 `construct`。但名单模式固定、两处、可直接读（实测取到 8 / 11 项）。**这是融合路径唯一正确的 saves 来源**，实测手写侧就漏了 `sparse_indices` 与 `sinks` | **S** | 是（fused 路径的 saves） |
| 11 | **`ops.stop_gradient` → `Detach` 节点 + 保边 + `detached` 标记** | `FREE_CALL_MAP` 就是现成挂钩。**这是全设计里价值最高的一项**（任务描述亦如此定性） | **S** | 是（grad 可达性） |
| 12 | **`PIN` 补规则** | 真新规则 ~8 条（`topk` 存 indices / `where` 存 cond mask / `scatter` / `argsort` / `histc` / `cumsum` / `maximum` / 常量产出类）+ 融合内核 ~8 个条目（后者的 saved-set 走 #10，不靠 PIN 猜）+ `SubCell` | **M** | 是（`derive_saves` 遇未知类型 fail-loud） |
| 13 | **逐 op saves（改造 `derive_saves`）** | 现在是全 DAG 按名去重的扁平表；liveness 要 `saved_by: name → {op idx}` | **S** | 是（liveness 接口） |
| 14 | **`is_weight` / `params` 区分** | 实测权重被当激活 save 计（FFNGroupedGEMM 88/236 MiB）。且 liveness 用 `bool(op.params)` 当梯度根 | **M** | 是 |

### P2 —— 算字节

| # | gap | 缺什么 | 量级 | 阻塞 |
|---|---|---|---|---|
| 15 | **`consumer._SYM2FIELD` 补 dsv4 符号** | 缺 `index_n_heads / index_head_dim / index_topk / compress_ratio / csa_window_size / o_groups / o_lora_rank / hc_mult`；不补就一律落 `unresolved` | **S** | 是 |
| 16 | **shape 种子自动化 + shard 应用** | `infer_shapes(dag, input_shapes)` 今天要手喂种子；抽出的张量无 TP/EP shard 标注 | **M** | 是 |

### P3 —— 本 yaml 不阻塞，但站点配置阻塞 / 或可暂留手写

| # | gap | 量级 | 阻塞 |
|---|---|---|---|
| 17 | **MTP**（`MultiTokenPredictionLayer` 的 `build_module` 首参不是 `submodules.<字段>`） | **M** | 本 yaml `mtp=0` → 否；站点 `mtp=1` → 是 |
| 18 | **pynative MoE**（`MoELayer` / `TopKRouter` / `GroupedMLP`）。router/dispatch/combine 本来就是声明的 opaque 边界；专家 grouped-GEMM 核可先复用 training_graph 那份（实测抽得出 10 nodes） | **M** | 部分（专家核可绕，router/dispatch/combine 需继续手写或 profiler 标定） |
| 19 | **pynative embedding / loss**（含 chunked CE） | **M** | 是（但这两段体量小、结构稳定，可最后做） |
| 20 | **`crosscheck._split_decoder` 的 `h1` 探针** | **S** | 不阻塞算数，但**今天它给出假绿**（§1）。至少应改成「探针失配 → `extraction_failures` 而非 `uncovered`」，让 `ok=False` |

---

## 11. 可行性判决

### 判决：**尚不可行（not yet）** —— 而且要把话说准

> **今天的抽取只覆盖 `parallel_core/training_graph/` 那棵树的 GPT/MLA/MoE-专家 段；
> DSv4-Flash 真正跑的 `pynative/` 侧 dsv4_hybrid 一个 Cell 都没走通（4 个目标全 FAIL，
> 其中 `CSAIndexer` 还是「假成功、0 节点」）；mHC 残差、MTP、pynative 的
> MoE/embedding/loss 同样全 FAIL。** 即：任务描述里那句假设——
> 「extraction covers only the GPT/MLA segments and dsv4_hybrid is not walked at all」——
> **实测成立**，而且比这更严重一点：不只是「没走」，是**其中一条路径静默返回空图**。

三条独立的硬事实支撑这个判决：

1. **结构**（§2）：整条流水线（`_SPEC_FILES` / `LEAF_OPTYPE` / `init_binder._CLS2OP` / 单文件 MRO）
   都是照 `training_graph` 的惯用法写的；pynative 用**另一套惯用法**（裸 `mint.*`/`ops.*` 别名、
   跨文件继承、DTensor 分支、`with _no_grad()`）。实测 dsv4 链上 **39 个原语 / 196 个调用点**
   一个都绑不上。
2. **语义**（§4、§5）：融合路径的 saved-set 与 detach 这两件**最要紧**的事，今天一个看不见
   （`_Function` 无 `construct`）、一个看得见文本但建模方向相反（`stop_gradient` 丢边不打标）。
3. **数值**（§7）：即便抽出图，裸 DAG 的 shape 全是 `?` → `consumer` 给 `total_bytes=0` +
   全部 `unresolved`。要手喂种子；dsv4 的 8 个符号还不在 `_SYM2FIELD` 里。

### 但底子比表面看起来好，三块可以直接复用

- **Pass A（模块树解析）**：加上 §2.2 的文件登记后，dsv4_hybrid 的整棵树（含嵌套两层
  `Compressor`）**完整正确解出**，`HyperConnectionTransformerLayer` 也选对了。这块**够用**。
- **符号 shape → 字节**（`sym_shape` + `shape_infer` + `consumer`）：DSv3 MLA 实测 26 节点、
  7 个 save、**`unresolved == []`**、94.0 MiB。`unresolved` 的「不杜撰」纪律也名副其实。
- **子 Cell 递归内联**（`_inline_subcell`）：id 续编 / 形参重映射 / 跨界连边都实现了，
  正是把 `Layer → attn → CSA → indexer/compressor` 拼成**一张层图**所需的机制。
- 再加 **`PIN` + `derive_saves` 的「每 op → 反向读哪些张量」语义与 `liveness.BwdNode.reads`
  概念上直接对齐**（§9）——目标形状是对的。

### 到 "yes" 的最短路径

**建议分成两条并行推进，因为它们的回报曲线完全不同。**

#### 路线 A（立即拿价值，S+S，不建整图）—— 「源抽 saves 审计器」

不建图，只从源里**逐字读两样东西**去审计/修正现有手写 spec：

1. 所有 `_Function.forward` 里 `ctx.save_for_backward(...)` 的实参名单（模式固定，~10 行 AST，
   实测已取到 `csa.py:113` 的 8 项与 `:224` 的 11 项）；
2. 所有 `ops.stop_gradient(...)` / `with _no_grad():` 的位置与被 detach 的张量名。

**立即产出**：手写 `sparse_attn` 缺 `sparse_indices` / `sinks` 这两项**已经找出来了**；
detach 名单可直接喂 `TensorRef.detached`（该字段 `shape_eval.py:78` 已存在、liveness 已消费）。
把它做成一个**测试**（源里的 `save_for_backward` 名单 ⊆ 手写 saves，否则 fail），
就把「手写名册漂移」这一类错误**永久钉住**——而这恰是 `crosscheck.py` 结构上做不到的
（它只比类别计数，§1）。**量级 S+S，今天就能做，且不依赖后面任何一条。**

#### 路线 B（换掉手写名册的正路）—— 按 P0 → P1 → P2 → 适配器

严格按此序，每一步都可独立验证：

1. **先做 P0#1**（未处理语句类改 fail-loud）。**这一步必须最先做**，否则后面每一步的
   「抽通了」都可能是假绿（现成反例：`CSAIndexer → 0 nodes`）。
2. P0#2/#3（裸别名表 + 跨文件 MRO）—— 这两条一做完，dsv4 链上 196 个调用点就都有归属了。
3. P0#4/#6/#7/#9（`With`+`_no_grad`、`init_dims → walker_flags`、形状标量/局部变量 env、
   `isinstance`/`hasattr` 注入）+ P0#8（可选槽剪枝）→ **验收点：4 个 dsv4 Cell + mHC 层
   在 fused 与 unfused 两条 A/B 配置下都抽出非空图，且 `opaque_calls` 里没有未白名单条目。**
4. P0#5（BinOp/Compare/Subscript RHS）→ **验收点：`q_hnorm_fp32` / `score_f32` /
   `attention_scores` / 两个 mask 这五个张量在图里有节点。**
5. P1 全部（`_Function` 入口 + `stop_gradient` + PIN + 逐 op saves + `is_weight`/`params`）
   → **验收点：`derive_saves` 不 fail-loud；fused 层的 saved 集逐名等于 `csa.py:224` 那 11 项；
   `ukl1`/`ukl2` 这类 detach 下游张量 `requires_grad=False`。**
6. P2（`_SYM2FIELD` + 种子/shard）→ **验收点：全层 `unresolved == []`。**
7. **`opdag/to_resolved.py` 适配器**（§9）→ 复用现成 `liveness/simulate.py` 的重算图变换，
   **不改 `liveness/`**。
8. **最终验收用已有真机数据，不重测**：pp4/dp2/ep2、seq4096、L8、m=4、全重算，
   fused ON `24153.3 / 14641.7 / 14097.7 / 23508.0`、unfused ON
   `50187.9 / 43940.0 / 43407.1 / 47888.1` MiB（`peak_alloc_MiB` per stage）。
   **注意**：unfused/fused ≈ 2.1–3.1× 这个比值本身就是对「fused saved-set 只有那 11 项、
   unfused 要留整条小算子链」的强约束——图抽对了它应当自然出来，抽错了对不上。
   这 8 个数是**验收判据**，不是拟合目标。

#### 粗量级
路线 A：**S + S**（≈ 半天量级，独立可交付）。
路线 B：P0 ≈ 2×M + 5×S；P1 ≈ 2×M + 3×S；P2 ≈ 1×M + 1×S；适配器 1×M —— 合计 **6×M + 9×S**。
P3（MTP / pynative MoE / pynative embedding+loss）另计 3×M，其中只有 MTP 在站点配置下是硬阻塞。

#### 一条必须写进设计的纪律
今天这条流水线在**四个地方静默给错**而不是 fail-loud：
`_SPEC_FILES` 参数过滤（解成 DSv3，§2.1，已修）、未处理语句类（0 节点，§6.2）、
`stop_gradient` 丢边（§5）、`crosscheck` 的 `h1` 探针失配变 `uncovered` 从而 `ok=True`（§1）。
换成源抽 saves 之后，「静默给错」比「手写错」更危险——手写错至少有人看得见那行 `saves=[...]`。
所以新设计的第一条不变量应当是：**任何抽取器看不懂的东西都必须显式出现在
`unresolved` / `opaque_calls` / `extraction_failures` 里，并且这三个列表非空时消费方默认拒绝出数。**

---

## 12. 本次评估过程中做的仓库改动（3 条，单独 commit）

commit `531bcdc` `fix(opdag): 三处最小可加修复 —— 让 pynative spec 树可解 + Identity 不再 fail-loud`
（父 `839968b`，分支 `feat/unified-llm-modelspec`）。全部**可加**、默认路径逐字节不变，
**前后都是 1526 passed**。

| 改动 | 文件 | 为什么 |
|---|---|---|
| `LEAF_OPTYPE` 补 `"Linear": "MatMul"` | `cost_eval/opdag/module_resolver.py` | pynative spec 树统一用 `Linear` 作线性叶子；缺表项时任何 pynative spec 在 `_bind_build_module` fail-loud（§2.3） |
| `resolve_layer_spec(..., spec_files=None)` + 新常量 `PYNATIVE_SPEC_FILES` | `cost_eval/opdag/module_resolver.py` | `_SPEC_FILES` 此前写死 training_graph，对本模型**静默解错模型**（§2.1）。缺省不变，既有 DSv3 调用零影响 |
| `PIN` 补 `"Identity": {"inputs": []}` | `cost_eval/opdag/bprop_rules.py` | 真实存在的 fail-loud：`qk_layernorm=False` → spec 解出 `Identity` → walker 发射 `Identity` 节点 → `derive_saves` 在 `:44` 炸。实测 DSv3 MLA + `qk_layernorm=False` 即触发 |

**这三条不足以让 dsv4_hybrid 抽通** —— 它们只是把最外层两道门打开，让本报告里的诊断可复现。
剩余阻塞见 §10 P0（M 级：跨文件 MRO、pynative 裸别名绑定表、`ast.With`/BinOp 走查）。

---

## 附录 A：一键复现（自包含，无需 monkeypatch）

`531bcdc` 之后，下面这段可直接跑出 §2.2 / §3.1 的结论：

```python
# repro_dsv4_extraction.py  —  在 pynative-cost-evaluator 仓库根跑
import sys
sys.path.insert(0, ".")
MF = r"E:\97-codes\torch_parallel\mf-src-167\mindformers"     # 权威快照的**包目录**

from cost_eval.opdag.module_resolver import resolve_layer_spec, PYNATIVE_SPEC_FILES
from cost_eval.opdag.extractor import extract_cell

SPEC_FLAGS = {   # 取自 dsv4h_fused_pp4_recomp.yaml
    "num_experts": 8, "moe_grouped_gemm": True, "qk_layernorm": True,
    "multi_latent_attention": True, "enable_hyper_connections": True,
    "fused_norm": True, "normalization": "RMSNorm", "is_dsv4_hybrid": True,
}
CELL_FLAGS = {
    "compute_dtype": "bf16", "layernorm_compute_dtype": "fp32",
    "v_head_dim": 512, "qk_pos_emb_head_dim": 64, "q_lora_rank": 1024,
    "num_attention_heads": 64, "hidden_size": 4096, "o_groups": 8, "o_lora_rank": 1024,
    "add_bias_linear": False, "input_layout": "BSND", "num_layers": 8, "mtp_num_layers": 0,
    "index_topk": 512, "index_n_heads": 64, "index_head_dim": 128,
    "csa_window_size": 128, "csa_dense_mode": False,
    # __init__ 派生量（walker 今天算不出，须手喂 —— §6.1 gap P0#6）
    "compress_ratio": 4, "enable_compress": True, "enable_indexer": True,
    "is_tnd": False, "window_size": 128, "training": True,
    "apply_dsa_kernel_fusion": True,
}

top = resolve_layer_spec(MF, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
print("layer      :", top.cell)                     # -> HyperConnectionTransformerLayer
sa  = top.submodules["self_attention"]
csa = sa.submodules["core_attention"]
print("attention  :", sa.cell, "| core:", csa.cell) # -> DSv4HybridSelfAttention | CompressedSparseAttention

REL = "pynative/transformers/experimental_attention_variant"
for name, rel, spec in [
    ("Compressor",                f"{REL}/compressor.py",  csa.submodules["compressor"]),
    ("CSAIndexer",                f"{REL}/indexer.py",     csa.submodules["indexer"]),
    ("CompressedSparseAttention", f"{REL}/csa.py",         csa),
    ("DSv4HybridSelfAttention",   f"{REL}/deepseek_v4_hybrid_attention.py", sa),
]:
    try:
        dag = extract_cell(MF, rel, name, spec, CELL_FLAGS,
                           present_params={"rotary_pos_emb"}, recurse=True, subcell_specs={})
        print(f"{name:26s} OK  -> {len(dag.nodes)} nodes"
              f"{'   <== 假成功：空图' if not dag.nodes else ''}")
    except Exception as e:
        print(f"{name:26s} FAIL -> {str(e)[:110]}")
```

预期输出（实测）：

```
layer      : HyperConnectionTransformerLayer
attention  : DSv4HybridSelfAttention | core: CompressedSparseAttention
Compressor                 FAIL -> construct 的 if 条件无法由 config 判定（compressor.py:190）:`sq < ratio`
CSAIndexer                 OK  -> 0 nodes   <== 假成功：空图
CompressedSparseAttention  FAIL -> construct 的 if 条件无法由 config 判定（compressor.py:190）:`sq < ratio`
DSv4HybridSelfAttention    FAIL -> construct 调用了未绑定的 self.shape(...)（deepseek_v4_hybrid_attention.py:233）
```

## 附录 B：融合 saved-set 逐字读（~10 行，可直接进测试）

```python
import ast
src = open(r"...\mf-src-167\mindformers\pynative\transformers"
           r"\experimental_attention_variant\csa.py", encoding="utf-8").read()
for cls in [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef)]:
    for call in [n for n in ast.walk(cls) if isinstance(n, ast.Call)]:
        f = call.func
        if isinstance(f, ast.Attribute) and f.attr == "save_for_backward":
            for a in call.args:
                if isinstance(a, ast.Starred) and isinstance(a.value, ast.ListComp):
                    it = a.value.generators[0].iter
                    if isinstance(it, (ast.Tuple, ast.List)):
                        print(cls.name, f"csa.py:{call.lineno}",
                              [ast.unparse(e) for e in it.elts])
```

实测输出：

```
FusedSparseFlashMla                csa.py:113 ['query','ori_kv','cmp_kv','sparse_indices',
                                              'cmp_residual','sinks','output','softmax_lse']
FusedSparseFlashMlaWithIndexerLoss csa.py:224 ['query','ori_kv','cmp_kv','sparse_indices',
                                              'query_index','key_index','weights',
                                              'cmp_residual','sinks','output','softmax_lse']
```

对照手写侧 `sparse_attn.saves`（9 项）→ **缺 `sparse_indices` 与 `sinks`**（§4）。
