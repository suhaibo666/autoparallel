# 字节解析的剩余阻塞项 —— G2 / G3 / G5 / T1 / T3 + 融合 mHC 内核 saved 集（2026-07-25）

> **性质**：接续 `docs/opdag_bytes_2026-07-25.md` §5 与
> `docs/opdag_component_coverage_2026-07-25.md` §6 的两张诚实清单，把里面
> **归属本轮**的五项（G2 / G3 / G5 / T1 / T3）与「融合 mHC 内核 saved 集」逐项关掉。
>
> **权威源**：`E:\97-codes\torch_parallel\mf-src-167\`，**两个**包：
> * `mindformers/` commit `26354ff64`（533 `.py`）
> * `hyper_parallel/` commit `41495aa2`（480 `.py`，**2026-07-25 新补入**）
>
> **不用** `E:\97-codes\torch_parallel\mindformers`（不同 commit / 不同内容）。
>
> 记法：**[RAN]** = 实跑（附命令 + 实际输出）；**[SRC]** = 逐字读源（带 `file:line`）。
> 本文**逐里程碑 checkpoint**（被打断也留证据）。

---

## 0. 基线 / 快照校验 [RAN]

```bash
cd /e/97-codes/torch_parallel/mf-src-167 && md5sum \
  mindformers/pynative/transformers/experimental_attention_variant/{csa.py,indexer.py} \
  hyper_parallel/core/shard/ops/parallel_mhc_{pre_sinkhorn,post}.py
```
```
81673be3ad3cdd2191e28dd000a13f0e *mindformers/.../csa.py                                 <- 与任务给定一致
8e18fee21c33507dd629c89f401986e6 *mindformers/.../indexer.py                             <- 一致
600cb4961cda0ae4225c708479e3ce77 *hyper_parallel/core/shard/ops/parallel_mhc_pre_sinkhorn.py  <- 一致
f1cbbf2d0c4a60f43c7b3756069a24e8 *hyper_parallel/core/shard/ops/parallel_mhc_post.py          <- 一致
```
文件数：`mindformers` 533、`hyper_parallel` 480，与 `SNAPSHOT.md` 逐字相符。

```bash
cd /e/97-codes/torch_parallel/pynative-cost-evaluator && git log --oneline -1 && python -m pytest tests -q
```
```
20ae22c feat(opdag): G4 发射点轴元信息 + T3 多输出 dtype + 组件覆盖文档
1795 passed, 268 warnings in 57.60s
```
→ 基线 **1795 passed** 复现，HEAD = `20ae22c`，分支 `feat/unified-llm-modelspec`。

---

## 1. 融合 mHC 内核的 saved 集 —— **从源读出来了，且与此前的「保守上界」实质不同** [SRC]

### 1.1 结论先行

此前（`docs/opdag_component_coverage_2026-07-25.md` §1.4 / §6 #2）因 `hyper_parallel` 不在快照里，
只能由**调用方声明**一个上界 `saved_ins_idx="all"`（= 张量实参全存）。
现在 `hyper_parallel` 已在快照里，saved 集**逐字写在源码里**：

| 内核 | 源定位符 | `ctx.save_for_backward(...)` 逐字名单 |
|---|---|---|
| `npu_mhc_post` | `hyper_parallel/platform/mindspore/custom_ops/custom_op_impl.py:331` | `(x, h_res, h_out, h_post)` = **4 个输入全存** |
| `npu_mhc_pre_sinkhorn` | 同文件 `:390-391` | `(x, phi, alpha, bias,` **`h_pre, hc_before_norm, inv_rms, sum_out, norm_out`**`)` = 4 个输入 **+ 自己的 5 个输出** |

**⇒ 旧的「保守上界」在 `npu_mhc_post` 上恰好等于源真值，但在 `npu_mhc_pre_sinkhorn` 上是
**欠读**（不是上界！）**：它漏掉了 5 个被保存的**自身输出**，其中
`sum_out`（`2·num_iters, B, S, N`）与 `norm_out`（`2·num_iters, B, S, N, N`）
是本内核最大的两块，且随 `num_iters`（yaml `hc_sinkhorn_iters`，缺省 20）线性放大。

这正是任务书要求「绝不把上界静默升格为事实」的反面教材：那个上界不但不是事实，**方向也判错了**。

### 1.2 逐字证据链（4 跳，每跳带 file:line）

```
① 调用点（模型侧）
   mindformers/pynative/transformers/hyper_connection.py:413-419
     h_in, h_post, h_res_flat, *_ = npu_mhc_pre_sinkhorn(
         x, _to_local(self.mapping_proj.weight), alpha, _to_local(self.bias),
         hc_mult=n, num_iters=self.sinkhorn_iterations,
         hc_eps=self.hc_eps, norm_eps=self.norm_eps)
   mindformers/pynative/transformers/hyper_connection.py:365
     output = npu_mhc_post(x, h_res, sublayer_out, self.squeeze(h_post, -1))
   import 处:  hyper_connection.py:21
     from hyper_parallel.custom_ops.experimental import npu_mhc_pre_sinkhorn, npu_mhc_post

② 公共包装（无 autograd 语义，纯转发）
   hyper_parallel/custom_ops/experimental/experimental_ops.py:157-166   npu_mhc_post
   hyper_parallel/custom_ops/experimental/experimental_ops.py:169-194   npu_mhc_pre_sinkhorn
     → return _platform.custom_ops.npu_mhc_{post,pre_sinkhorn}(...)

③ 平台派发 → DFunction.apply
   hyper_parallel/platform/mindspore/custom_ops/custom_ops.py:53-56   → NpuMhcPostDFunction.apply
   hyper_parallel/platform/mindspore/custom_ops/custom_ops.py:58-61   → NpuMhcPreSinkhornDFunction.apply

④ **saved 集**（autograd 事实所在）
   hyper_parallel/platform/mindspore/custom_ops/custom_op_impl.py:318-332  (post.forward)
     ctx.save_for_backward(x, h_res, h_out, h_post)                            # :331
   hyper_parallel/platform/mindspore/custom_ops/custom_op_impl.py:367-393  (pre_sinkhorn.forward)
     result = _custom_ops.npu_mhc_pre_sinkhorn(x, phi, alpha, bias, hc_mult,
                                               num_iters, hc_eps, norm_eps, out_flag)   # :386
     _, _, _, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = result                # :389
     ctx.save_for_backward(x, phi, alpha, bias,
                           h_pre, hc_before_norm, inv_rms, sum_out, norm_out)           # :390-391
     ctx.hc_eps = hc_eps                                                                # :392
   反向逐字取回同 9 项:custom_op_impl.py:409（`ctx.saved_tensors` 解包）
                     并全部喂给 `npu_mhc_pre_sinkhorn_backward`（:418-422）
```

### 1.3 `pre_sinkhorn` 的 8 个输出与形状（源侧逐字）

输出名册 `(h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out)`：
`custom_op_impl.py:383-384`（docstring 逐字）与 `parallel_mhc_pre_sinkhorn.py:176`（类 docstring）
**两处独立一致**。形状由 `NpuMhcPreSinkhornDistributedOp.infer_output_layouts`
（`parallel_mhc_pre_sinkhorn.py:236-299`）的注释与 tensor_map 阶数逐字给出：

| # | 输出 | 4-D 输入 `(B,S,N,C)` 时的形状 | 源定位符 | 是否 saved |
|---|---|---|---|---|
| 0 | `h_in` | `(B, S, C)` | `parallel_mhc_pre_sinkhorn.py:260` | ✗ |
| 1 | `h_post` | `(B, S, …)` 3-D | `:261` | ✗ |
| 2 | `h_res` | `(B, S, …)` 3-D | `:261` | ✗ |
| 3 | `h_pre` | `(B, S, …)` 3-D | `:261` | **✓** |
| 4 | `hc_before_norm` | `(B, S, …)` 3-D | `:261` | **✓** |
| 5 | `inv_rms` | `(B, S, …)` 3-D | `:261`（与上同组） | **✓** |
| 6 | `sum_out` | `(2·num_iters, B, S, N)` | `:262-263` | **✓** |
| 7 | `norm_out` | `(2·num_iters, B, S, N, N)` | `:264-265` | **✓** |

> [!important] **诚实边界**：`:261` 那一行把 `h_post / h_res / h_pre / hc_before_norm / inv_rms`
> **五个输出共用**一个 3-D tensor_map `(b_map, s_map, -1)`，第 3 轴写的是 `-1`（= 复制，
> 与分片无关），**它不给出该轴的长度**。也就是说源码给出了这些输出的**阶数**（3-D）和
> 前两轴（`B`、`S`），但**末轴长度不在本快照的 Python 侧**（在 `.cc` kernel 里，
> `custom_op_impl.py:38-41` 列出 `mhc_pre_sinkhorn.cc` 等 4 个 `.cc`，**未纳入快照**）。
> 从模型侧可反推两个：`h_res` 被 reshape 成 `(s,b,n,n)`（`hyper_connection.py:422`）⇒ 末轴 = `N·N`；
> `h_post` 被 reshape 成 `(s,b,n,1)`（`:423`）⇒ 末轴 = `N`。
> 余下 `h_pre / hc_before_norm / inv_rms` 三项的末轴**源侧不确定** → 一律进 `unresolved`
> （原因码 `kernel_output_shape_unknown`），**不给数**。
>
> 数学上可从非融合支类推（`h_pre` 对应 `HyperConnectionModule` 里的 pre 分量、`inv_rms` 是
> RmsNorm 的逐 token 倒数标度 ⇒ 大概率 `N` / `1`），但那是**推测**，不是源。按纪律不写。

`sum_out` / `norm_out` 的形状**是**源侧确定的（`:262-265` 的注释逐字给出
`(2*iters, B, S, N)` / `(2*iters, B, S, N, N)`，且 tensor_map 阶数 4/5 与之一致）。

### 1.4 落地方式

`kernel_saves` 声明契约扩展为可同时声明 **输入位序** 与 **自身输出位序**，且新增
`saved_outs_idx` 与 `saved_out_shapes`（后者只填源侧**确定**的那些）。声明的 `source` 字段
现在指向**真源** `custom_op_impl.py:331` / `:390-391`，而非「实测报告的推断」；
节点 attrs 上 `saved_from_source=True` 与旧的 `saved_declared_by_caller=True` **分开记**，
使「读出来的」与「声明的」在图上仍可区分。

（详见 §6 的改动清单与 §7 的测试。）

---

## 2. G2 / G3 / G5 —— 按构造点传维度 + 按内联帧解符号 [RAN] + [SRC]

### 2.1 三件事其实是**同一个**结构缺陷的三个面

抽取器把子 Cell **内联**进父图后,子节点身上带的还是**子那个类**的符号世界
(`self.head_dim`、局部 `seqlen`),而 `dag.dims_ctx` / `dag.scalar_binds` 只装**顶层**那份。
此前的缓解手段是调用方侧 `shape_infer.merge_dims_ctx(*ctxs)`(同名不同值 fail-loud)——
它在 `Compressor` 上**必然失效**,因为同一个类的两个构造点让 `self.head_dim` 解出
**两个不同**的符号:

| 构造点 | 逐字源 | `head_dim` |
|---|---|---|
| `csa.py:596-603` | `build_module(submodules.compressor, config=config, compress_ratio=self.compress_ratio, head_dim=config.v_head_dim, rotate=False, rotary_pos_emb=self.rotary_pos_emb)` | `v_head_dim` = **512** |
| `indexer.py:126-132` | `build_module(submodules.compressor, config=self.config, compress_ratio=self.compress_ratio, head_dim=self.index_head_dim, rotate=True, rotary_pos_emb=rotary_pos_emb)` | `index_head_dim` = **128** |

⇒ 一张扁平表**装不下**,必须按**内联帧**分作用域。故 G2(构造点维度)、G3(子 `dims_ctx`)、
G5(子 `scalar_binds`)一并落地。

### 2.2 落地(逐文件)

| 文件 | 改动 | 判据 |
|---|---|---|
| `init_dims.py` | `INIT_PARAM_SEEDS` 增第三形态 `{"dim": 符号串, "val": 具体值}` | 构造点 `head_dim=config.v_head_dim` **既**要按符号 `v_head_dim` 参与维度代数(随 DimTable 变)、**又**在 flags 里有 512 可供 `if` 判定;不是二选一 |
| `init_dims.py` | `eval_init_dims(..., param_seeds=)`:**按构造点**逐键覆盖全局种子 | G2 本体 |
| `init_dims.py` | `InitDims.self_seeds`:`self.<attr>` 的可传播求值结果 | 构造点写 `head_dim=self.index_head_dim` 时,子要拿父这一侧求出来的东西 |
| `init_dims.py` | **`_Cell.tainted`** —— 传递地来自 `__init__` **形参缺省**的值**不得**经 `self_seeds` 传播 | 铁律「不得从 `__init__` 缺省推结构」。实测反例:`csa.py:556` `compress_ratio: int = 0`,若传播下去,子 `Compressor` 的 `ratio=0` 让 `cutoff = (sq // ratio) * ratio` 直接判不出 —— 20 条既有测试当场变红,**正是该铁律要挡的东西** |
| `extractor.py` | `_ctor_seeds(call, self_seeds, config_flags, …)`:构造点关键字 → 子形参种子,四种可解形态(字面量 / `config.X` / `self.X` / 裸转发形参),其余**不收** | 见函数 docstring 的逐条源定位符 |
| `extractor.py` | 具体值(int/bool)同时注入子的 `config_flags`,两道门(须是本 MRO 某 `__init__` 的形参 + 值须是 int/bool) | `bind_init` 的三元形态 `self.hadamard = Hadamard(head_dim) if rotate else IdentityOp()`(`compressor.py:129`)按 `config_flags[<形参名>]` 判,只喂 `INIT_PARAM_SEEDS` 判不出 |
| `construct_walker.py` | `SubExtract` 携 `dims_ctx` / `scalar_binds`(G3/G5 的契约变更) | 见该 dataclass 的 docstring |
| `construct_walker.py` | `_inline_subcell`:给每次内联生成**帧标签**(`compressor@28`,嵌套用 `/` 连),写 `attrs["frame"]` / `attrs["dims_ctx"]`;子 `scalar_binds` 按帧上浮且 `src` 重映射到调用方基名 | 同名 `self.<attr>` / 局部 `seqlen` 在父子帧里指不同东西 |
| `shape_infer.py` | `scalar_map` 改按 `(帧, 名)` 键;`_frame_chain` 给可见链(当前帧 → 逐级外层 → 根帧);`_attr_dim` 先查节点自带的 `dims_ctx` 再退顶层 | 同上 |

### 2.3 实测:两个构造点各自解对了 [RAN]

```bash
PYTHONIOENCODING=utf-8 python scratchpad/probe_dsv4_bytes.py fused
```
`CompressedSparseAttention` 里新解出的两项**逐字带 `index_head_dim`**(= indexer 那条链上的
compressor,`indexer.py:128` head_dim=128),而 `Compressor` 单抽时是 `v_head_dim`(=512):

```
CompressedSparseAttention  __arg__i5  ~((B·S·index_head_dim)+(B·S·index_head_dim))  fp32   4.0000 MiB
                           weights    ~((B·S·index_head_dim)+(B·S·index_head_dim))  bf16   2.0000 MiB
Compressor(单抽,= CSA 直挂那个) __arg__i5  ~((B·S·v_head_dim)+(B·S·v_head_dim))    fp32  16.0000 MiB
                                 weights    ~((B·S·v_head_dim)+(B·S·v_head_dim))    bf16   8.0000 MiB
```
比值恰为 512/128 = **4×** —— 正是任务书点名「单一全局种子必然错 4×」的那个量。

| Cell(fused) | 节点 | out 已解析 | 改前 | saves 字节 | 改前 | node-gaps | 改前 |
|---|---|---|---|---|---|---|---|
| `Compressor` | 29 | 15 | 15 | 56.000 MiB | 56.000 | 13 | 13 |
| `CSAIndexer` | 7 | 0 | 0 | 0 | 0 | 7 | 7 |
| `CompressedSparseAttention` | 90 | **43** | 11 | **306.000 MiB** | 300.000 | **45** | 68 |
| `DSv4HybridSelfAttention` | 123 | **53** | 21 | **318.000 MiB** | 312.000 | **68** | 91 |

### 2.4 回归 + 字节中性 [RAN]

```bash
python -m pytest tests -q     ->  3 failed, 1792 passed
```
3 条失败全部是 `tests/test_acceptance_gate.py`,**与本轮无关**:把本轮 4 个改动文件
`git stash` 掉后**同样 3 failed**(并行 agent 的未跟踪 `cost_eval/opdag/to_resolved.py` 一落地就
激活了惰性探测的 `extracted` 源)。即 1792 + 3 = **1795 = 基线**,本轮**零回归**。

字节中性(**diff,不是断言**):
```bash
git worktree add <sc>/wt-base 48c68a1
cd <wt-base> && PYTHONPATH=. python scratchpad/dump_numbers.py > before.txt   # 650 行
cd <主树>    && PYTHONPATH=. python scratchpad/dump_numbers.py > after.txt    # 650 行
diff before.txt after.txt   ->  *** 0 diff ***
```
