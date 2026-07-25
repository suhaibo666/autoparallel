# opdag 组件覆盖补全 —— mHC / MoE / embedding / loss / MTP（2026-07-25）

> **性质**：接续 `docs/opdag_walker_core_2026-07-25.md` §7 的「诚实未做清单」第 7 项
> （mHC 层 / MTP / pynative MoE / pynative embedding+loss），把这些组件带到与四个
> dsv4_hybrid 注意力 Cell 相同的标准：**非空**、**零诊断**（或显式允许清单）、`strict=True` 干净、
> 有逐节点 dump。
>
> **权威源**：`E:\97-codes\torch_parallel\mf-src-167\mindformers\`（commit `26354ff64`，
> md5 `csa.py=81673be3ad3cdd2191e28dd000a13f0e` / `indexer.py=8e18fee21c33507dd629c89f401986e6`
> 已实测校验）。**不用** `E:\97-codes\torch_parallel\mindformers`。
>
> 记法：**[RAN]** = 实跑所见（附命令 + 实际输出）；**[SRC]** = 逐字读源（带 file:line）。
> 本文**逐组件 checkpoint**，每完成一个即追加。

---

## 0. 环境 / 基线 [RAN]

```bash
cd /e/97-codes/torch_parallel/mf-src-167 && md5sum \
  mindformers/pynative/transformers/experimental_attention_variant/{csa.py,indexer.py}
```
```
81673be3ad3cdd2191e28dd000a13f0e *csa.py
8e18fee21c33507dd629c89f401986e6 *indexer.py
```
→ 权威快照校验通过（与任务书逐字相同）。

```bash
cd /e/97-codes/torch_parallel/pynative-cost-evaluator && git log --oneline -1 && python -m pytest tests -q
```
```
a84954e feat(opdag): 让纯 AST 抽取器真正走通 DSv4-Flash 的 pynative 侧四个 Cell
1713 passed, 268 warnings in 53.79s
```
→ 基线 **1713 passed** 复现，HEAD = `a84954e`，分支 `feat/unified-llm-modelspec`。

（后续章节按组件追加。）

> **并行 agent 的 in-flight 状态(必须先说清)** [RAN]:本轮进行中,另一 agent 在**同一工作树**
> 提交了 `f4744ae`(字节解析底座)并留有未提交改动
> (`cost_eval/opdag/{shape_infer,consumer,sym_shape}.py`)。因此 `tests` 的绿线以
> **两者叠加**为准。实测:`tests/test_opdag_shape_infer.py::test_moe_grouped_gemm_operand_is_e_cap_h`
> 在**把本轮全部改动 stash 掉之后仍然失败**(命令:`git stash push <本轮 6 个文件>` →
> `pytest tests/test_opdag_shape_infer.py -q` → `1 failed, 17 passed`),即它由并行 agent 的
> `_grouped_matmul`(现 `needs_axis_structure` 时返回 `None`)引起,**不是本轮造成的**。
> 已 `git stash pop` 恢复。本轮自己的绿线在 §7 逐条给。

---

## 1. mHC hyper-connection 残差 —— 从**零覆盖**到整层 193 节点 [RAN]

### 1.1 起点(评估文档 §3.3 的实测复现)

```bash
python scratchpad/probe_components.py mhc     # 改动前
```
```
FusedHyperConnectionModule   OK 5 nodes | diag: unregistered_targets=3, unresolved_operands=2 | opaque=1
HyperConnectionTransformerLayer  FAIL ValueError: extractor: build_module(submodules.pre_cross_attn_layernorm)
                                 在 spec 里无对应子模块（self.pre_cross_attn_layernorm @ transformer_layer.py）
```

### 1.2 终点

```bash
python scratchpad/probe_components.py mhc
```
```
layer spec: HyperConnectionTransformerLayer
  submodules=['cross_attention','input_layernorm','mlp','pre_cross_attn_layernorm',
              'pre_mlp_layernorm','self_attention']
FusedHyperConnectionModule         OK    6 nodes /   5 edges | diag: clean | opaque=0 params=4
HyperConnectionTransformerLayer    OK  193 nodes / 234 edges | diag: clean | opaque=2 | detached=8 params=18
    [opaque] router.py:690: save_to_aux_losses_tracker(...)          kind=host_side_effect
    [opaque] moe_layer.py:127: self.tokens_per_expert.add_(...)      kind=inplace_param_buffer_update
```
非融合支(`use_fused_mhc=False`)`HyperConnectionModule` = **142 节点 / 180 边,零诊断、零 opaque**。

### 1.3 修的六处(每处都带源侧依据,不是"放宽")

| # | 症状 | 源侧事实 | 改动 |
|---|---|---|---|
| 1 | `build_module(submodules.pre_cross_attn_layernorm)` 无对应子模块 → **整个 mHC 层零覆盖** | `TransformerLayerSubmodules` 六个字段的 dataclass 缺省**全是 `IdentityOp`**(`transformer_layer.py:68-75`);dsv4_hybrid spec 只填 4 槽(`gpt_layer_specs.py:105-113`),另两槽真机上**确实**被 `build_module(IdentityOp, ...)` 实例化(`spec_utils.py:76-77,97`) | `module_resolver._declared_defaults`:未显式填的字段用**声明的缺省值**;**缺省是 `None` 的字段不收**(`build_module(None,...)` 在真机上会炸 → 语义是"这条路径不 build 它",填成 Identity 会造出真机不存在的节点) |
| 2 | `self.attn_hc(...)` 未绑定 | `hc_cls = FusedHyperConnectionModule if config.use_fused_mhc else HyperConnectionModule`(`:279`)后 `self.attn_hc = hc_cls(...)`(`:280`) | `extractor._init_class_aliases`:`__init__` 里的**局部类别名**按 config 三元解;**判不出就不收**(两支是 fused/unfused 两条不同算子链) |
| 3 | `self.attn_hc.output_cell(...)` 报「缺少 output_cell 方法」 | `output_cell` 不是方法,是 `HyperConnectionModule.__init__:233` / `FusedHyperConnectionModule.__init__:388` 建的**另一个 Cell** | `extractor._nested_cell_of`:`self.<A>.<B>(...)` 里 B 是 A 的子 Cell 属性时递归进 **B 的 construct**(仅当 A 及其 MRO 都无 `B` 方法) |
| 4 | `self.hidden_states_dropout(...)` 未绑定;递归进去又撞 0 节点硬门 | `Dropout.construct` 首行 `if not self.training or not self.use_dropout: return x`(`pynative/layers/dropout.py:74-75`);yaml 未给 `hidden_dropout`(缺省 0.0)⇒ `use_dropout=False` ⇒ 真机上**就是恒等**,一个 kernel 都不发 | `_took_identity_return`:剪枝后**实际命中**的 return 是「原样返回入参」且诊断/opaque 全空 → 0 节点合法。三重收紧见其 docstring |
| 5 | `self.mapping_proj(...)` 未绑定 | `self.mapping_proj = Linear(input_size=..., ...)`(`hyper_connection.py:212-219`)—— 不经 spec 树的**直接实例化叶子** | `extractor._named_module_binds` 新增 `fname in LEAF_OPTYPE` 分支(与 `build_module(submodules.X)` 解出 `"Linear"` 是同一源侧事实) |
| 6 | `for _ in range(self.iterations - 1)` 整条被丢 | `SinkhornKnopp.construct`(`hyper_connection.py:63`),`iterations = config.mhc_sinkhorn_iterations`(`:206`→`:232`→`:45`),yaml `hc_sinkhorn_iters: 20` | `_handle_for`:**只**展开静态可数的 `range(...)` 与 config 声明的元组;其余照旧记诊断。实测 `iters=20` → 循环体 **114 个节点**(19 轮 × 6),`iters=3/5` → 12/24(线性,测试钉住) |

### 1.4 融合 mHC 内核的 saved 集 —— **快照外,故必须由调用方声明**(诚实记录)

`npu_mhc_pre_sinkhorn` / `npu_mhc_post` 来自 `hyper_parallel.custom_ops.experimental`
(`hyper_connection.py:20-23`,`try/except ImportError`),**不在权威快照里** → 它们的 bprop
读不出来。处置(纪律:绝不猜):

* walker 对未声明的 `Kernel` **不写** `saved_ins_idx` → `derive_saves` 照旧 fail-loud
  (`test_fused_mhc_kernel_saved_set_must_be_declared_not_guessed` 钉住);
* 要出数就必须给 `kernel_saves={内核名: {"saved_ins_idx": ..., "source": ..., "reason": ...}}`,
  缺 `source`/`reason` 即 fail-loud;声明留在节点 attrs 上
  (`saved_declared_by_caller=True` / `saved_source` / `saved_reason`),使「这不是源读出来的」
  **在图上可见**,不与 `FusedFunction`(能从 `ctx.save_for_backward` 逐字读)混为一谈。
* 本轮验收用的声明:`saved_ins_idx="all"`,出处
  `analysis/dsv4_calibration_final_report_2026-07-23.md:137` 记载的实测结论「aclnn 自定义算子
  (mHC/MLA,经 HP DFunction)saves 全走 `save_for_backward`(custom_op_impl.py:331/390/588)」
  → **张量实参全存的保守上界**。**这是调用方声明的上界,不是源真值** —— 列为建议项:
  需要精确值时应把 `hyper_parallel` 一并纳入快照,或在 167 上实测。

### 1.5 逐节点 dump —— 融合 mHC(6 节点)[RAN]

```
#1 View        hyper_connection.py:405  ins=[hidden_states]  out=x        SAVED   (reshape 成 [s,b,n,H])
#2 Cast        hyper_connection.py:406  ins=[x]              out=x:bf16   SAVED
#3 View        hyper_connection.py:408  ins=[]               out=alpha    SAVED   params=[alpha_pre,alpha_post,alpha_res]
#4 Kernel      hyper_connection.py:413  ins=[x, alpha]       out=h_in             (npu_mhc_pre_sinkhorn;saved 集=调用方声明)
#5 View        hyper_connection.py:422  ins=[h_res_flat]     out=h_res
#6 View        hyper_connection.py:423  ins=[h_post]         out=h_post
saves(2): ['alpha', 'x']      param_operands: ['alpha_pre','alpha_post','alpha_res','bias']
```
读法要点:`Parameter` 全在 `param_operands`、**不在 `ins`、不在 saves、不是任何 op 的 out**
(W2/W3/W4,`test_mhc_weights_never_enter_ins_or_saves` 逐条断言);`h_res_flat`/`h_post`
是内核的第 2/3 个输出,`*_` 丢弃项**不登记**(源里 `h_in, h_post, h_res_flat, *_ = ...`,`:413`)。

### 1.6 顺手修掉的一个**假 dtype**(此前无人可见)[RAN]

`x = self.cast(x, self.dtype)`(`hyper_connection.py:406`,`self.dtype = config.compute_dtype`
@ `:205`)—— `self.dtype` 不在 `config_flags` 里时,`_resolve_cast_dtype` 会把
`_dtype_name` 返回的**名字** `"dtype"` 当成一个 dtype,顺着 SSA 传播到下游每个 ref
(实测第一版 dump 里 `#2 ... out=x:?:dtype`)。现新增「合法 dtype 全集」门:解出**不是 dtype
的记号**即 fail-loud 并指名要注入哪个键。

---

## 2. pynative MoE —— 从**三条全 FAIL**到 MoELayer 44 节点 [RAN]

### 2.1 起点 / 终点

```bash
python scratchpad/probe_components.py moe
```
| 目标 | 改动前(评估文档 §3.2 复现) | 现在 |
|---|---|---|
| `MoELayer` | FAIL `self.router(...)` 未绑定 @ `moe_layer.py:122` | **OK 44 节点 / 47 边**,零诊断,2 条justified opaque |
| `TopKRouter` | FAIL `self.reshape(...)` 未绑定 @ `router.py:353` | **OK 30 / 31**,零诊断,1 条 host 副作用 |
| `GroupedMLP` | FAIL `isinstance(tokens, DTensor)` 不可判定 @ `experts.py:185` | **OK 5 / 5**,零诊断、零 opaque |
| `SharedExpertMLP` | 未跑通(spec 树给不出 `linear_fc1`) | **OK 8 / 8**,零诊断、零 opaque |

### 2.2 最重要的一条:一个**静默解错模型**(与评估文档 §2.1 同一病症的第二例)[RAN]

`extractor._find_cell_file` 按 `os.walk` **首个命中**定位类。`class MoELayer` 在快照里有**三份**:

```bash
grep -rn "^class MoELayer" --include=*.py .
./parallel_core/inference/transformer/moe/moe_layer.py:101
./parallel_core/training_graph/transformer/moe/moe_layer.py:86
./pynative/transformers/moe/moe_layer.py:29          <-- 真机跑的这份
```
实测它解到了 `inference` 那份,于是报「`self.router` 的 build_module 首参不是 submodules.<字段>」
—— 那是**另一棵树**的写法(pynative 的 `MoELayer.__init__:49` 是 `self.router = TopKRouter(...)`,
根本没有 `build_module`)。**即:在给一份真机从未跑过的结构建模。**

修法(权威答案在 spec 文件自己的 import 里,不是猜):`ResolvedSpec.origin_rel` ——
Pass A 建 `ModuleSpec` 时顺该 spec 文件的 import 解出类的定义文件
(`pynative/base_models/gpt/moe_module_specs.py:20`
 `from mindformers.pynative.transformers.moe.moe_layer import MoELayer`)。
`_find_cell_file` 同时收紧:多处定义时按调用方位置的**最长公共目录前缀**排序,
并列即 **fail-loud 并列出全部候选**(`test_ambiguous_cell_name_without_origin_is_fail_loud`)。

### 2.3 其余修的十处(逐条源侧依据)

| # | 症状 | 源侧事实 | 改动 |
|---|---|---|---|
| 1 | `if need_dispatch:` 不可判定 @ `experts.py:191` | `need_dispatch = not isinstance(self.weight1, DTensor) or "ep" not in self.weight1.device_mesh.mesh_dim_names`(`:181`);yaml `expert_parallel: 2` ⇒ mesh 有 "ep" ⇒ **False**(ExpertParallel 接管派发) | ① `_eval_compare` 支持 `<字面量> in/not in <x>.mesh_dim_names`,取值由调用方 `runtime_predicates["mesh_dim_names"]` 显式给(缺键 fail-loud 指名要哪个键);② `_handle_compare` 新增「整条 RHS 能判成 bool ⇒ 它是 host 谓词,登记标量不建节点」通路(守卫 `kind != TENSOR`,免得张量比较被误判) |
| 2 | `if not is_in_recompute():` @ `moe_layer.py:126` | `pynative/distributed/activation_checkpoint.py` 的**运行相位**函数;前向图建模的是首次前向 | `runtime_predicates` 支持 `call:<零参函数名>` 键;不给 → fail-loud |
| 3 | `self.tokens_per_expert.add_(...)` @ `moe_layer.py:127` | 接收者是 `Parameter(..., requires_grad=False)`(`:84-88` 逐字) | 新增「非梯度 buffer 原地更新」通路:记 `opaque_calls{kind:"inplace_param_buffer_update"}` + `param_operands`,**不建节点**。字节中性**可证**(不在 autograd 图上 ⇒ 无 saved;原地 ⇒ 不新分配;无赋值目标 ⇒ 无消费者)。`requires_grad=False` 由 extractor 扫 `Parameter(...)` 关键字得到(`self_kinds["__buffer__<attr>"]`),缺证据**不走**这条 |
| 4 | `mint.histc` / `mint.floor_divide` 未知原语 | `router.py:158` / `experts.py:145` | 补 `primitives` 表项,归 `Compare` 桶(**不可微、反向什么都不读** —— 与既有 `argsort`/`argmax` 同一条判据) |
| 5 | `aux_loss_func_map = {...}` → `.get(self.aux_loss_type)` → 调用 @ `router.py:479-490` | **python 级方法派发表**;`aux_loss_type` 来自 yaml `moe_router_load_balancing_type: seq_aux_loss` | walker 新增 `_method_tables` / `_method_refs` 两步解析:键判不出则不解(退回既有兜底);键不在表里 → fail-loud(源里紧随其后就是 `raise ValueError`,`:486`) |
| 6 | `compute_routing_scores_for_aux_loss(...)` @ `router.py:466` 落 opaque | 定义在 `moe_utils.py:343`,由 **相对 import** `from .moe_utils import (...)`(`router.py:19`)引入 | ① `ClassIndex._scan_imports` 支持相对 import(折算成绝对点号路径);② `_module_level_funcs` 顺 import 收**跨文件**模块级 `def`,并带上定义文件 rel(否则内联出的 `file:line` 是错的) |
| 7 | `self.moe_aux_loss_auto_scaler(...)` @ `router.py:708` → 「`MoEAuxLossAutoScaler` 及其基类均无 `__init__`」 | 它是个**纯静态包装 Cell**:只有 `@staticmethod construct`(`moe_utils.py:330-340`),转发 `_MoEAuxLossAutoScaler.apply` | `_extract_meta` 放宽:无 `__init__` 但有入口方法 = 合法(binds 为空);连入口都没有才 fail-loud |
| 8 | `tp_cp_size = get_moe_aux_loss_group_size()` @ `router.py:687` | 该函数返回模块级全局 `_AUX_LOSS_GROUP_SIZE`(`moe_utils.py:261-263`,运行时按 tp×cp 组设置) | `call:<name>` 声明扩展到**任意值**(不只 bool):`_classify_call` 判 SCALAR、`_scalar_of` 给声明值、`_handle_call` 登记标量身份 |
| 9 | `tokens_layout = tokens.layout` 落 `dropped_assigns` → `if tokens_layout is not None:` 随即不可判定 @ `experts.py:186/210` | `.layout` 是 DTensor 的**分片元数据对象**(host 侧),且该分支是 `isinstance(tokens, DTensor)` 判 True 才进来的 ⇒ 它确实存在 | 新增 `_HOST_META_ATTRS`(`layout`/`placements`/`device_mesh`/`mesh`/…)→ 记 `_PRESENT` |
| 10 | `DTensor.from_local(experts_output, ...)` @ `experts.py:211` 落 opaque + `experts_output` 断链 | 它是 `.to_local()`(已在 `PASSTHRU_METHODS`)的**逆**:把已存在的 local 张量包成 DTensor 视图,不复制、数学恒等 | 新增 `_PASSTHRU_FREE_CALLS = {"DTensor.from_local": 0}`;同时修 `_bind_name_rhs` 的**自赋值**分支(`x = f(x)` 此前先 `_forget(x)` 把源身份清掉 → 目标掉出 SSA、下游丢边) |
| 11 | `submodules = MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear)` @ `moe_layer.py:58-62` → 子 `MLP.__init__` 的 `build_module(submodules.linear_fc1)` 无对应子模块 | `get_moe_module_spec` 只返回 `ModuleSpec(module=MoELayer)`,submodules **全空**(`moe_module_specs.py:34-37`)⇒ 这些叶子**只能**从父 `__init__` 读 | `_inline_submodules_spec`:`self.X = <Cls>(config, submodules)` 且 `submodules` 是同一 `__init__` 里 `<XSubmodules>(field=<类名>, ...)` 赋的局部 → 建子 ResolvedSpec 挂到 Binding 的 `attrs["spec"]`,resolver 优先用它 |
| 12 | `super().construct(hidden_states)` @ `shared_experts.py:68` → 抽出 **0 节点** | `SharedExpertMLP(MLP)` 的 fc1→act→fc2 主体全在基类 `MLP.construct` 里 | `_lookup_super_method`:按 MRO **跳过正在走的那一层**(跳过数 = 1 + 内联链里同名次数),绝不回到同一层 |

### 2.4 逐节点 dump —— `GroupedMLP`(5 节点)[RAN]

```
#1 GroupedMatMul experts.py:221  ins=[permuted_local_hidden_states]  out=fc1_output  SAVED  params=['weight1']
#2 View          experts.py:229  ins=[fc1_output]                   out=x0          SAVED  (chunk 2, dim=-1)
#3 Activation    experts.py:230  ins=[x0]                           out=act_out     SAVED
#4 Elementwise   experts.py:231  ins=[act_out, x1]                  out=intermediate_parallel SAVED
#5 GroupedMatMul experts.py:235  ins=[intermediate_parallel]         out=fc2_output         params=['weight2']
param_operands: ['weight1','weight2']   saves ∩ param_operands = ∅
```
`need_dispatch=False`(§2.3 #1)⇒ permute/unpermute 段**不在本 config 的图里**(EP 接管)。

---

## 3. pynative embedding / lm_head / loss / MTP [RAN]

```bash
python scratchpad/probe_components.py emb ; ... loss ; ... mtp ; python scratchpad/probe_head.py
```
| 段 | 改动前(评估文档 §3.4 / §3.5) | 现在 |
|---|---|---|
| `VocabEmbedding` | FAIL 三元 `isinstance(self.weight, DTensor)` 不可判定 | **OK 4 / 3**,零诊断,1 条 host 类型断言 |
| `LanguageModelEmbedding` | FAIL `self.word_embeddings(...)` 未绑定 | **OK 6 / 5**,同上 |
| `Linear`(lm_head vocab 投影) | **合成**(4 个手写 OpNode) | **OK 2 / 0 —— 走查出来的**,零诊断、零 opaque |
| `CrossEntropyLoss`(pynative) | FAIL `self._tp_group is not None` 不可判定 @ `loss.py:328` | **OK 11 / 11**,零诊断、零 opaque |
| `ChunkCrossEntropyLoss` | 未触及 | **OK 1 / 0**(一个 `_ChunkCrossEntropyLoss.apply`),零诊断 |
| `MultiTokenPredictionLayer` | FAIL `self.enorm` 的 build_module 首参不是 `submodules.<字段>` | **OK 232 / 280**,零诊断,3 条justified opaque |

### 3.1 关键修复(逐条源侧依据)

| # | 症状 | 源侧事实 | 改动 |
|---|---|---|---|
| 1 | `Validator.check_type_name(...)` @ `vocab_embedding.py:78` 落无 kind 的 opaque | **纯 host 侧类型断言**:无返回值被消费、不产张量 | `host_call_allow` 的匹配从「裸名」扩到「完整点号路径 / 属性末段」 |
| 2 | embedding gather 的 **saved 集是空的** | `self.embedding(weight, 0, input_)`(`:85`)= `mint.gather(input, dim, index)`,反向是 scatter_add 回 index 位置 ⇒ **必须存 index** | `PIN["IndexSelect"]={"inputs":[1]}` 的 `[1]` 是「**张量操作数**位序」;权重被路由出 `ins` 后 `ins` 只剩 `[input_]`,`[1]` 越界 → saves 空。新增 `attrs["ins_slots"]`(每个存活 `ins` 项的张量操作数位序,只在**确有位被略过**时才写),`derive_saves` 按它取 |
| 3 | `Linear` 三条 `if <dtype> != <dtype>:` 门(`linear.py:126/128/143`)判不出 → lm_head **一个节点都抽不出** | `params_dtype == compute_dtype == bf16`(yaml)⇒ 三条全 False | 新增 `_dtype_value(base)`:权重 → `params_dtype`;SSA 变量 → ref 的 dtype 段;dtype 环境名(`ori_dtype = input_.dtype`)→ 该记号;construct 形参 → `compute_dtype`。`_value` 同时支持**裸 dtype 名**参与比较 |
| 4 | loss 段 `saves` 只有 0 项 | `_LogSoftmax.forward` **只有** `ctx.logits = logits`(`loss.py:136`),没有 `save_for_backward`;`backward` 里 `logits = ctx.logits`(`:151`)—— 内存事实与 `save_for_backward` **相同** | `_fn_class_table` 增 `bare_ctx_params`(rhs 逐字是 forward 形参 ⇒ 能按位映射到 `apply` 实参)与 `bare_ctx_internal`;`_handle_function_apply` 同等计入 saved 集,但在 attrs 上**分开记来源**(`save_for_backward` vs `bare_ctx_retained`),不把「裸 ctx」洗成「save_for_backward」。实测 `CrossEntropyLoss` 的 saves 从 0 → 7 项,含 **logits**(全模型最大的一块,`S·B·vocab`) |
| 5 | `if logits.ndim == 3:` @ `loss.py:363` 判不出 | 轴数是 `input_axes` 种子的**长度**(与 `infer_shapes` 同一套契约) | `_scalar_of` / `_value` 支持 `<入口形参>.ndim`;没种子 → fail-loud(`test_ndim_without_a_seed_is_fail_loud_not_guessed`) |
| 6 | `build_module(self.submodules.enorm, ...)` 首参形态不认 → **MTP 零覆盖** | `MultiTokenPredictionLayer.__init__:286` 先 `self.submodules = submodules`,再从 `self.submodules.<field>` 读 | `_bind_build_module` 接受第二种形态,且**要求**该 `self.<attr>` 能顺 `_param_to_attr` 回溯到 `submodules` 形参(不是任意 `self.x.y` 都放行) |
| 7 | `abs(shifts) >= dim_size` @ `multi_token_prediction.py:178` 判不出 | `roll_tensor` 的 `shape = tensor.shape` / `dim_size = shape[dims]`;`shifts=-1`、`dims=-1`、`seq_length=4096` | ① `shape = <入口形参>.shape` 记成**逐轴值元组**(`_axes_tuple`);② `shape[dims]` 按该元组取值;③ `_classify_call` 的 host 内建集与 `_HOST_BUILTINS` 合并(此前少 `abs`/`slice`/`tuple`/`list` → 左侧判成 UNKNOWN) |
| 8 | `embedding(input_ids=..., position_ids=...)` @ `:447` 落 opaque + 断链 | 这个 `embedding` 是 `gpt_model.py:340` 传进来的 `self.embedding`(= `LanguageModelEmbedding`,`:183`);MTP **真的会**再跑一遍 embedding(输入是 roll 过的 input_ids) | 新增 `param_cells={形参名: 类名}`:调用方顺源里的实参链声明,walker 才递归;不声明照旧落 opaque + `unregistered_targets`(`test_mtp_embedding_param_cell_undeclared_stays_visible_not_silent` 钉住"可见而非静默") |
| 9 | MTP 层 spec 不由入口函数产出 | `get_gpt_mtp_block_spec`(`gpt_layer_specs.py:227-256`)拿 decoder block **最后一层** spec + `hc_head` 调 `get_mtp_layer_spec`(`multi_token_prediction.py:223`);`enable_hc_head` 缺省 `None` ⇒ 跟随 `enable_hyper_connections` = True(`transformer_config.py:2158-2159`) | 新增 `module_resolver.resolve_spec_call(mf_root, entry, ...)` + `MTP_SPEC_FILES` |

### 3.2 逐节点 dump —— lm_head 的 vocab 投影(**走查**,不再合成)[RAN]

```
#1 View   linear.py:132  ins=[]         out=weight  params=['weight']   (weight 转置,weight-derived)
#2 MatMul linear.py:135  ins=[input_]   out=output  params=['weight']
saves(1): ['input_']       <- 只存激活输入;权重与权重派生量都不存
```
对照 `gpt_segments.head_segment_dag()` 的 4 个**合成**节点(src 钉 `gpt_model.py:503-507`):
现在这两个节点的 `file:line` 指向真正的计算体 `pynative/layers/linear.py:132/135`。
**合成段本身未删**(既有数字路径仍在用它),见 §5 的诚实清单。

### 3.3 逐节点 dump —— `CrossEntropyLoss`(11 节点)[RAN]

```
#1  FusedFunction loss.py:197  ins=[logits]                out=...   (_LogSoftmax;bare_ctx_retained=['logits'])
#2  FusedFunction loss.py:210  ins=[log_softmax, label]    out=...   (_NLLLoss)
#3  View          loss.py:342  ins=[input_mask]            out=input_mask       SAVED
#4  Cast          loss.py:343  ins=[input_mask]            out=input_mask:fp32  SAVED
#5  Elementwise   loss.py:344  ins=[loss_reduce, input_mask:fp32]  out=...
#6  Elementwise   loss.py:344  ins=[...]                   out=numerator        SAVED
#7  Elementwise   loss.py:346  ins=[input_mask:fp32]       out=...
#8  Constant      loss.py:347  ins=[]                      out=...     (ops.tuple_to_array((1e-8,)))
#9  Cast          loss.py:347  ins=[...]                   out=...:fp32
#10 Elementwise   loss.py:345  ins=[..., ...:fp32]         out=denominator      SAVED
#11 Elementwise   loss.py:349  ins=[numerator, denominator] out=...
saves(7): ['denominator','input_mask','label','log_softmax','logits','loss_reduce','numerator']
```

---

## 4. 并行 agent 交接项(G4 / T3)—— 在本轮文件里且**防止静默算错**,故一并做了 [RAN]

### 4.1 G4:发射点记录轴/形状元信息

**为什么不是"锦上添花"**:缺轴时下游字节解析对这些算子退化成"直通",于是**不是"未知"而是"错"**
(数量由并行 agent 实测):`.sum(dim=1)`(`compressor.py:216`)**8× 过读**;`chunk` **n× 过读**;
`cat([kv_nope, kv_pe], -1)`(`compressor.py:243`)解成 `2n·b·(d−64)` 而非 `n·b·d`。

新增/补齐的键:`reduce_dim` + `keepdim`、`chunks` + `chunk_dim`、`concat_axis`、`stack_axis`、
`permute_dims`、`squeeze_axis`、`roll_shifts`/`roll_dims`、`broadcast_shape`、`topk_k`/`topk_dim`、
`const_shape`/`const_shape_src`、`slice_bounds`。三个发射通路都挂上了(绑定别名 / 裸别名与自由调用 /
**张量方法**——后者实参不含 receiver,位序要补齐才与命名空间形式一致)。

实测覆盖:
```
Compressor                    29 nodes,  12 带轴元信息
CSAIndexer                     7 nodes,   2
CompressedSparseAttention     90 nodes,  35
DSv4HybridSelfAttention      123 nodes,  43
```
三个被点名的站点逐条命中(`test_reduce_chunk_concat_axes_are_recorded_at_emit_time`):
`compressor.py:216 → reduce_dim=1`、`:169 → chunks=2/chunk_dim=-1`、`:243 → concat_axis=-1`、
`:233 → slice_bounds=[(None,'total_seq_len','self.compress_ratio')]`。
**纪律不变**:抠不出的**一律不写**该键(缺键 = 未知,消费方落 `unresolved`),
绝不填默认轴 —— `test_axis_metadata_is_absent_rather_than_guessed_when_unreadable` 钉住。

### 4.2 T3:多输出算子的 dtype

`topk_scores, topk_indices = self.topk(...)`(`indexer.py:262`)第 2 个输出是 int32 索引。
此前**所有**目标都按同一个 `out_dtype`(bf16)进 SSA,而 `attrs["outs"]` 里写的是 int32 ——
于是下一行 `topk_indices = self.cast(topk_indices, int32)`(`:263`)的 `ins` 带着**错的 dtype**;
`derive_saves` 按名去重只留一个,留下哪个取决于谁先被 pin = **静默取错分支**。
现按 `attrs["outs"]` 逐个输出登记 SSA dtype。实测 `:263` 的 `ins` 从 `topk_indices:?:bf16`
变为 `topk_indices:?:int32`。

### 4.3 顺带做的 W2/W3/W4 那一半:**权重派生量永不进 saves**

评估文档 §7.2 实测:`FFNGroupedGEMM`「236 MiB」里 `w1`(58.7MB)+`w2`(29.4MB)= **88 MiB
是权重被当激活 save 计**。链条是 `w1 = cast(self.weight1, …)`(`ffn.py:146`)→
`w1 = reshape(w1, …)`(`:157`)→ 进 `GroupedMatmul` 的 `ins` → `PIN{"inputs":"all"}` 全存。

修法:walker 传播「权重派生」标记(`ins` 空且本行有权重操作数,**或** `ins` 非空但全部已标记),
在节点上写 `attrs["weight_ins_idx"]`,由 `derive_saves` 排除。
**刻意保留在 `ins` 里** —— 并行 agent 的 `shape_infer._grouped_matmul` 要靠权重末轴推
matmul 输出维,把它从 `ins` 删掉会砸掉字节解析。实测:
```
saves 前: dispatched_input, fc1_output, intermediate_parallel, tokens_per_expert, w1, w2
saves 后: dispatched_input, fc1_output, intermediate_parallel, tokens_per_expert
fc1.attrs["weight_ins_idx"] == [1]   fc2 同   reshape(:157).attrs["weight_ins_idx"] == [0]
```

---

## 5. 验收 [RAN]

```bash
python -m pytest tests -q
```
```
1795 passed, 268 warnings in 54.49s
```
* 参照:并行 agent 的 `1043aa5` → **1780 passed**。1795 = 1780 + **15 净新增**
  (`tests/test_opdag_components.py` 共 31 条)。
* **字节中性(不是断言,是 diff)**:手写 spec 全字段 dump(dsv3 L=4/6/8,逐 op 的
  `type/ws/bws/attrs` + 每个 `TensorRef` 的 10 个字段)+ 完整 `Evaluator.evaluate()` 数字
  (dsv3 L=4/8 × rc None/full 的 `peak_event`/`peak_bytes`/全 `breakdown`)
  → 对 `1043aa5` **652 行 0 diff**(唯一"差异"是混进 stdout 的 warning 里的绝对路径,
  主树 vs worktree 不同,已按路径前缀切掉后逐行相等)。
* 抽取侧四个 dsv4 Cell 的 census 锁在 `test_all_four_dsv4_cells_still_clean_after_axis_capture`。

### 5.1 既有测试的**期望**变更台账(3 处,逐条给理由;**未删除任何一条不变量**)

| # | 测试 | 原断言(钉病症) | 新断言(钉不变量) | 理由 |
|---|---|---|---|---|
| 1 | `test_for_while_try_all_recorded`(`test_opdag_drop_diagnostics.py`) | `LOOPS` 里 `for i in range(3):` 进 `dropped_stmts` | 举例换成 `for t in x:`(**迭代次数由张量决定、静态判不出**),`kinds == ["For","While","Try"]` 逐字保留 | `range(<静态可求值>)` 现按真实轮数展开(真源需要:`SinkhornKnopp` 的 `for _ in range(self.iterations - 1)`,`hc_sinkhorn_iters: 20` ⇒ 19 轮 × 6 节点)。新增支持形态由 `test_static_range_for_is_unrolled` / `test_range_with_undecidable_bound_is_not_unrolled` **单独**钉住(净新增 2 条) |
| 2 | `test_ffn_saveset_includes_previously_missing_grouped_gemm_operands`(`test_opdag_moe_ffn.py`) | `"w1" in names and "w2" in names` —— 断言**权重在 saves 里** | `"w1" not in names and "w2" not in names`;同时断言它们**仍在 `ins` 里**且 `weight_ins_idx == [1]` | 原断言钉的是一个病症(评估文档 §7.2 实测 88/236 MiB)。三条"激活操作数必须被 pin"的不变量(`dispatched_input`/`intermediate_parallel`/`fc1_output`)**逐字保留** |
| 3 | 四个 dsv4 Cell 的 census(fused 支边数 104→103 / 146→145) | — | 见 `test_all_four_dsv4_cells_still_clean_after_axis_capture` 的 docstring | 去掉的是 `[84, 88]`:`FusedSparseFlashMlaWithIndexerLoss.apply(..., attn_sink, ...)` 的 `attn_sink` 操作数。`self.attn_sink` 是 `Parameter`,`csa.py:683-687` 只有 `to_local()` + `cast(fp32)`,全程无激活参与;此前 `:685` 的**自赋值**把权重身份 `_forget` 掉,于是产出一个**看起来是激活**的张量进了融合算子的 `ins`。现走 `param_operands`。字节影响 `[n_heads] fp32` = **256 B**,方向是修正 |

### 5.2 红→绿证据

* 新测试文件整体在 `1043aa5` 上**红**:`TypeError: extract_cell() got an unexpected keyword
  argument 'kernel_saves'`(以及 6 项组件本身的 FAIL,见各节 §x.1 的"起点");
* 行为层面的红:每节起点表格里的实测 FAIL 输出;
* 绿:`python -m pytest tests/test_opdag_components.py -q` → **31 passed**。

---

## 6. 明确**没做**的事(诚实清单)

| # | 未做 | 为什么 |
|---|---|---|
| 1 | **走查整个 `GPTModel.construct`**(把 lm_head 段从"合成"整体换掉) | 已试,并测到前两个阻塞(`mint.not_equal` 未知原语 —— 已补表项;`self.use_attn_mask_compression` 等一批 `__init__` 派生布尔需逐个注入)。`GPTModel.construct` 是**全模型入口**(988 行文件,含 eod/zbv/mtp/pp/return_logits/share_embeddings 多族分支),surface 远大于本轮六个组件。本轮交付的是**vocab 投影本体走查**(`pynative/layers/linear.py:132/135`,全 step 最大的 GEMM)+ loss 段走查;`gpt_segments.head_segment_dag()` 的合成段**原样保留**(既有数字路径在用),`verify_gpt_order` 的段序守卫不变 |
| 2 | **融合 mHC 内核的精确 saved 集** | `npu_mhc_pre_sinkhorn` / `npu_mhc_post` 在 `hyper_parallel.custom_ops.experimental`,**不在权威快照里**。本轮给的是「调用方声明 + 带定位符与理由 + 节点上可评审」的通路,声明值是**保守上界**(见 §1.4)。建议:把 `hyper_parallel` 一并纳入快照,或在 167 上实测 |
| 3 | **G2**(`build_module` 维度关键字按**构造点**传进子 Cell 的 `eval_init_dims`) | 交接说它**阻塞 58 个内联 compressor 节点**的字节解析,且 `Compressor` 有两个构造点(`head_dim` 512 vs 128),单一全局种子**4× 错**。正确做法是 per-construction-site 传播,需要先给 `eval_init_dims` 一个「构造点关键字」入口 —— 而 `init_dims.py` **不在本轮可改文件**内。**留给字节解析方**:extractor 侧的挂钩已就位(`_bind_build_module` 已经能拿到构造点的 `call`,`_kw_self_args` 是同类先例) |
| 4 | **G5**(导出 construct 局部标量环境;`SubExtract` 携 `scalar_binds`) | 需要改 `SubExtract` 契约并在 `_inline_subcell` 里按帧重映射名字(否则子 Cell 的 `x.shape` 解包名与父帧撞形)。这是一次**契约变更**,与本轮的组件覆盖交付相互独立,不在同一提交里混做 |
| 5 | **T1**(去掉 norm 输入的 fp32 预抬,交给 `ResolvedLayer` 的消费方抬) | 它会改动 `bprop_rules.derive_saves` 对 `Norm` 的 dtype 输出,而 `docs/opdag_walker_core_2026-07-25.md` §6.6 第 14 行的「DSv3 MLA `derive_saves` 名册逐字钉死 7 项(同 dtype)」正锁着这个口径。要动就要连同 `to_resolved.py` 的消费侧一起改 —— 那是 T2 的活 |
| 6 | **`ops.rms_norm` / `q_hnorm_fp32` 的 ledger 项** | 交接给的源事实(`ops.rms_norm` 在本快照**从未被调用**;`deepseek_v4_hybrid_attention.py:245` 是纯 bf16 逐元素;`csa.py:689-697` 只把 `weights`/`attn_sink` 抬 fp32)与本轮抽出的图**一致**:`Compressor`/`DSv4HybridSelfAttention` 的图里没有任何 `Norm{rms}` 节点来自 `:245`。按任务约束**未改 `cost_eval/layers/**` 任何数字** |
| 7 | **非融合 mHC 的 `SinkhornKnopp` 展开数与真机对齐** | 展开数严格随注入的 `mhc_sinkhorn_iterations` 线性变化(测试钉住 3/5/20 → 12/24/114 节点)。yaml 给 `hc_sinkhorn_iters: 20`,但**该 key 到 `config.mhc_sinkhorn_iterations` 的映射未在本轮核对**(本 config `use_fused_mhc: true`,不走这条路径) |
| 8 | **`For`/`While`/`Try` 的其余形态** | `for x in <张量>` / `enumerate` / `while` / `try` 仍不走查,逐条记诊断 + strict 下抛(实测 dsv4 + mHC + MoE + MTP 链上零触发,`SinkhornKnopp` 的 `range` 已覆盖) |
| 9 | **`crosscheck.py`** | 任务说「不要把它当验收门」。本轮**未改**它。它的 `_split_decoder` 已在 `a76d192` 之后匹配 `h1_*`;真正的门是 `tools/liveness_ab_validate.py` 对 8 组真机跑 |

---

## 7. 改动清单

| 文件 | 性质 |
|---|---|
| `cost_eval/opdag/construct_walker.py` | `For` 展开;`super()`;方法派发表;host 谓词/零参声明;`mesh_dim_names`;dtype 比较与 `_dtype_value`;`.ndim` / `shape` 元组;`_HOST_META_ATTRS`;`DTensor.from_local`;非梯度 buffer 原地更新;`param_cells`;`kernel_saves`;`ins_slots`;`weight_ins_idx` 传播;多输出 dtype(T3);轴元信息捕获(G4);自赋值修复;`return <BinOp>` 物化;假 dtype 门;0 节点门的恒等豁免 |
| `cost_eval/opdag/extractor.py` | submodules dataclass 缺省;局部类别名;嵌套子 Cell 属性;`self.submodules.<f>`;手搭 submodules;直接实例化叶子;跨文件模块级函数;`origin_rel` 优先 + `_find_cell_file` 多义 fail-loud;无 `__init__` 的静态包装 Cell;`__buffer__` 标记;`bare_ctx_params` |
| `cost_eval/opdag/module_resolver.py` | `ResolvedSpec.origin_rel`;`_declared_defaults` / `submodule_declared_defaults`;`resolve_spec_call` + `MTP_SPEC_FILES` |
| `cost_eval/opdag/module_index.py` | 相对 import 解析(`_relative_to_dotted`) |
| `cost_eval/opdag/primitives.py` | 补 `mint.histc` / `floor_divide` / `nn.functional.sigmoid` / `not_equal` / `equal` / `ops.tuple_to_array` / `ops.zeros|ones|zeros_like|concat|reshape|transpose` |
| `cost_eval/opdag/bprop_rules.py` | `ins_slots` 位序映射;`weight_ins_idx` 排除 |
| `tests/test_opdag_components.py` | **新增** 31 测试(md5 门控) |
| `tests/test_opdag_drop_diagnostics.py` | 1 处期望迁移 + 2 条新增(台账 §5.1) |
| `tests/test_opdag_moe_ffn.py` | 1 处期望迁移(台账 §5.1) |
| `scratchpad/probe_components.py` / `probe_head.py` | **新增**:一键复现探针 |
