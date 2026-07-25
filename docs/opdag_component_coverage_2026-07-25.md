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
