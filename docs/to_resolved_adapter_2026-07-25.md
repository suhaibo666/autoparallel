# `extracted` 适配器 + 验收门判决（2026-07-25）

> **性质**：施工日志 + 验收判决单。交付 `cost_eval/opdag/to_resolved.py`（把源抽取的 per-Cell
> `OpDAG` 折成满足 `cost_eval/liveness/contract.py` 18 条硬规则的 `ResolvedLayer` 序列），
> 把它注册成 graph source `extracted`，然后跑 167 A/B 八跑验收门并**如实**报告落点。
>
> **权威源**：`E:\97-codes\torch_parallel\mf-src-167\`（`mindformers` @ `26354ff64`、
> `hyper_parallel` @ `41495aa2`）。**不用** `E:\97-codes\torch_parallel\mindformers`。
>
> 记法：**[RAN]** = 实跑（附命令 + 实际输出）；**[SRC]** = 逐字读源（带 `file:line`）。
> 本文按里程碑**逐段 checkpoint**（被打断也留证据）。

---

## 0. 基线锚点 [RAN]

```bash
cd /e/97-codes/torch_parallel/pynative-cost-evaluator && git rev-parse HEAD && python -m pytest tests -q
```
```
48c68a13a54bc89350343710a8740a4db9195505
1795 passed, 268 warnings in 54.78s
```
→ 基线 **1795 passed** 复现；分支 `feat/unified-llm-modelspec`；开工 HEAD = `48c68a1`
（开工瞬间是 `20ae22c`，并行 agent 在读文档期间提交了 `48c68a1`；两者均在本轮**不可改**的
`cost_eval/opdag/` 其它文件上）。

权威快照校验：
```bash
cd /e/97-codes/torch_parallel/mf-src-167 && md5sum \
  mindformers/pynative/transformers/experimental_attention_variant/{csa.py,indexer.py}
```
```
81673be3ad3cdd2191e28dd000a13f0e *csa.py
8e18fee21c33507dd629c89f401986e6 *indexer.py
```

## 0.1 所有权边界（另有 agent 并行）

本轮**只**碰：`cost_eval/opdag/to_resolved.py`（新，本轮独有）、`cost_eval/liveness/**`、
`tools/**`、`tests/` 新增件、本文档。
**不碰** `cost_eval/opdag/` 其余文件、`cost_eval/layers/**`、`structure_mem.py`、
`mem_timeline.py`、`serve_explorer.py`。需要它们改动的，作为**建议项**列在 §9。

---

## 1. 开工前的关键实测 —— 缝的两侧到底差多少 [RAN]

（本节为设计依据，逐段在后续里程碑补齐。）

### 1.1 手写侧（`hand_spec`）的层册

八跑配置（`analysis/realmachine/ab_fusion_2026-07-25/`，pp4/dp2/ep2/tp1/cp1、S=4096、B=1、
L8 时 `compress_ratios=[0,4,128,4,128,4,128,4]`）解出的 `layer_pattern`：

```
0 embedding            stage0   ops=2   param=1010.000 MiB  act=  160.000 MiB
1 dsv4hyb_r0_dense     stage0   ops=24  param= 591.164 MiB  act= 3674.000 / 8921.000 MiB (fused/unfused)
2 dsv4hyb_r4_moe       stage0   ops=32  param= 479.805 MiB  act= 3660.562 / 22530.062 MiB
3 dsv4hyb_r128_moe     stage1   ops=31  param= 455.539 MiB  act= 3582.094 / 9625.094 MiB
… (4..8 交替 r4/r128) …
9 lm_head              stage3   ops=5   param=1010.016 MiB  act= 3222.000 MiB
```

（`cross_entropy_fused=True` → 本配置**没有**独立 loss 伪层，`logsoftmax`/`nll` 在 `lm_head`
层里；`mtp_num_layers=0` → **没有** MTP 伪层。）

### 1.2 抽取侧的起步状态

`extract_cell(HyperConnectionTransformerLayer, recurse=True)` 实测（fused、r4、mHC 融合）：

```
nodes 196  edges 238
param_operands 18 条（去重后 9 个名）：alpha_pre/alpha_post/alpha_res/bias（hyper_connection.py:409/414）、
    ape（compressor.py:208/209）、attn_sink（csa.py:683/687/689）、
    linear_o_group_proj（deepseek_v4_hybrid_attention.py:280/284）、
    weight（router.py:363/364）、tokens_per_expert（moe_layer.py:127）、
    weight1/weight2（experts.py:218/219/221/235）
dims_ctx {}          ← 顶层类 __init__ 求值结果里没有维度符号
detached ['x_detach__i1','qr_detach__i1','q','k','weights','cmp_residual_k','topk_indices','index_scores']
```

**两个结构性发现（决定本轮判决的量级）**：

1. **叶子 `Linear` 的权重不在 `param_operands` 里**。`build_module(submodules.linear_q_down, …)`
   被 `module_resolver.LEAF_OPTYPE` 解成**单个 `MatMul` 节点**（不递归进 `Linear.construct`），
   于是 `self.weight` 这个 `Parameter` 从未被 walker 看到。抽取侧 `param_operands` 只有 9 个名
   ——手写侧 r4 层有 20+ 个权重、479.8 MiB。
   **可补救**：该 `MatMul` 节点带 `attrs["in_dim"]` / `attrs["out_dim"]`（源侧
   `build_module(sub.X, input_size=A, output_size=B)` 逐字读出），而
   `pynative/layers/linear.py:81-83` [SRC] 明写
   `weight_shape = (output_size, input_size)` / `dtype=self.params_dtype`
   → 权重字节 = `out_dim × in_dim × params_dtype`，**是源读出的事实，不是拟合**。本轮据此补。
2. **`dims_ctx` 为空**：`extract_cell` 只装顶层类的 `__init__` 求值（`extractor.py:725`），
   内联子 Cell 里的 `self.<attr>` 解不出 → 必须像 `scratchpad/acceptance.py` 那样
   `merge_dims_ctx(*各子 Cell 的 dims_ctx)`；`param_shapes` / `param_dtypes` 同理。

（后续章节按里程碑追加。）
