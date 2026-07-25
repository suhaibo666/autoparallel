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
