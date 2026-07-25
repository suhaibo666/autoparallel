# `ResolvedLayer` 适配器契约 —— 抽取侧必须交出什么（2026-07-25）

> **性质**：接口设计note + 交接单。可执行的那一半在 `cost_eval/liveness/contract.py`（Protocol +
> 18 条编号规则 + 校验器），契约测试在 `tests/test_resolved_layer_contract.py`（48 项）。
> 本文只记**为什么这么定**、**抽取侧要补什么**、以及**哪些地方还是近似**。
>
> **权威源**：`E:\97-codes\torch_parallel\mf-src-167\mindformers\`（commit `26354ff64`）。
> 覆盖度实测见 [`opdag_coverage_assessment_2026-07-25.md`](opdag_coverage_assessment_2026-07-25.md)。

## 0. 一句话

`liveness/` **一行不改**；写一个 `cost_eval/opdag/to_resolved.py`，把 `OpDAG` 折成满足本契约的
`ResolvedLayer` 序列，注册成 graph source `extracted`；于是手写 census 与源抽取图可以在
**同一个 liveness 仿真器**上做 A/B —— 差异只可能来自图本身。

```
OpDAG(已 infer_shapes) + DimTable + ParallelModel
    ──[ to_resolved.resolve_graph(model_spec, parallel_model) ]──▶  ResolvedGraph(.stages)
                                                                     │ 满足 contract.py
                                     ┌───────────────────────────────┴───────────────┐
                          liveness/graph.py                       structure_mem / static_mem
                       （激活四桶：逐张量 liveness）              （非 liveness 桶：原样沿用）
```

## 1. 为什么契约面比"liveness 读的字段"更宽 —— 一个实测出来的坑

第一版契约只按 `liveness/graph.py` 实际读的字段定（`name / local_numel / dtype_bytes /
is_weight / detached`）。写"外部生产者"契约测试时**直接炸了**：

```
AttributeError: 'FT' object has no attribute 'is_expert'
  cost_eval/structure_mem.py:275:  divisor = efsdp if w.is_expert else fsdp
```

根因：`simulate_liveness` 只接管**激活**四桶（`act_live` / `recomp_scratch` /
`bwd_working_set` / `remat_saves`），其余非 liveness 桶（`persistent` / `gather_buf` /
`grad_buf` / `bwd_scratch` / `swap_buf` / `optstep` …）**原样沿用**
`structure_mem.estimate_structure_memory` 与 `static_mem.StaticMem`
（`simulate.py:211-213, 268-276`）。所以一个 graph source 必须同时喂得动那两条。

早期设计笔记里"`is_expert`/`dim0`/`pin_under_recompute` 是桶模型专用、liveness 不读、缺省即可"
的说法**实测不成立**，已升为硬规则 **T2**。这就是"先写契约测试，再让抽取侧接上来"的价值：
缝的真实宽度是量出来的，不是想出来的。

## 2. 契约面（逐字段 → 消费者定位 → 抽取侧现状）

| 字段 | 语义（本地量） | 谁读它 | `OpDAG` 现状 | 抽取侧工作量 |
|---|---|---|---|---|
| `layer.layer_id` / `layer_type` | 层号 / 层型（伪层 embedding/lm_head/mtp 亦算层） | `graph.py:300`、`simulate.py` 事件标签 | 每个 `extract_cell` 只产**单个 Cell** 的图 | **M**：用现成 `recurse=True` + `_inline_subcell`（`construct_walker.py:632-698`）把 `HyperConnectionTransformerLayer → self_attention → CSA → {indexer, compressor}` + `MoELayer` 内联成**一张层图**。该机制已存在且设计正确 |
| `layer.ops`（**执行序**） | 前向 op 序列 | 全链 | 有（`nodes` + `edges`） | — |
| `op.name` / `op.type` | op 名 / 规范类型 | `graph.py:240-241`、`structure_mem` 的 norm/flash 判据、`rc.op_matches`（select 重算的选中判据） | 有 | **S** |
| `op.inputs` | 非权重操作数 + tie 权重 | `graph.py:166,187,205,236` | 有（`ins` 串），但**不区分权重** | **M** |
| `op.output` | 单个产出 | `graph.py:166,183,208` | 有 | — |
| `op.params` | 该 op **拥有**的权重（= 梯度根） | `graph.py:205,220`（`bool(op.params)` 决定 `has_bwd`）、`structure_mem:275,301`、`static_mem` | **无** —— 权重只是普通 `ins` | **M** |
| `op.saves` | **逐 op** 的 save_for_backward 候选 | `graph.py:223-231` 建 `saved_by: name → {op idx}` | `derive_saves` 返回**全 DAG 按名去重的扁平表**（`bprop_rules.py:40,59`，只留一个 `op_id`） | **S**：改成逐节点列表（全局去重可留作视图） |
| `op.workspace_bytes` / `bwd_scratch_bytes` | kernel 临时物化 | `graph.py:244,279` | **无概念** | **不要求**从源推 —— 见 §5 |
| `t.local_numel` | 已按 TP/EP shard 与 CP 序列切分**除过**的元素数 | 到处 | 符号 shape 串；`consumer` 能算**全局** elems，**无 shard 标注** | **M**：见 §4.1 |
| `t.dtype_bytes` | 张量**自身** dtype 字节（norm 的 fp32 cast 由消费侧 `structure_mem._dt` 再抬，**不**预抬） | 到处 | 有（`ins` 串第三段） | **S** |
| `t.is_weight` | 是否参数 | `graph.py:176,196,205,228,236,242`、`structure_mem`、`static_mem` | **无** | **M**：见 §4.2 |
| `t.detached` | `stop_gradient` / `with _no_grad()` 后的产物 | `graph.py:176,193-195,211`（grad 可达性的切断点） | **无**，且 `stop_gradient` 处**边被切断** | **S**：`FREE_CALL_MAP` 是现成挂钩；反向要**保边 + 打标** |
| `t.is_expert` | 走 EP-FSDP 还是 dense-FSDP | `structure_mem.py:275,301`（**直读，缺就 AttributeError**） | **无** | **S** |
| `t.dim0` | TP/EP placement 后的**本地首维**（runtime FSDP 按首维切，`parallelize.py:331-350`） | `structure_mem.py:100` | **无** | **S–M**（0 = 未知 → 退回 total-numel 整除口径，与 runtime 可能不一致） |
| `t.pin_under_recompute` | fused 自定义算子 ctx 保存集在全重算下**不释放** | `structure_mem.py:293` | **无** | **S**（当前全库无 spec 设此标志） |

## 3. 18 条硬规则（编号与 `contract.py::CONTRACT_RULES` 一一对应）

结构/打型 **L1 O1 O2 O3 T1 T2** · 字节已解析 **B1 B2 B3**（+ A/B 闸门 **B4**） ·
params vs activations **W1 W2 W3 W4 W5 W6** · saves/反向 **S1 S3** · grad 可达性 **G2 G3**。

每条的判据与理由见 `contract.py` 的 docstring；每条都有一个**违约样本测试**
（`tests/test_resolved_layer_contract.py`），并且有一条元测试
（`test_every_documented_rule_is_reachable`）保证"文档里的规则 ⟺ 校验器真的查的规则"，
不许出现只写不查的规则。

现役手写路径实测：**6 个 preset × 3 组并行度 + 真 DSv4-Flash pp4/dp2/ep2 的 fused/unfused
两支，violations = 0**。即缝是真的，抽取侧接上来时对照的是一个**已被满足**的标准。

## 4. 两个已知硬点：契约怎么**逼**它们，而不是让它们溜过去

### 4.1 裸抽取 DAG 的 shape 全是 `?` → `total_bytes = 0`

`walker._emit` 恒写 `?` 作 shape 段（`construct_walker.py:900`），`extract_cell` **不**调
`infer_shapes`。实测（评估文档 §7.1）：

```
-- embedding (VocabParallelEmbedding): RAW dag   total_bytes=0  per_save=0  unresolved=4
-- FFNGroupedGEMM:                     RAW dag   total_bytes=0  per_save=0  unresolved=6
```

**契约的逼法**：**B1**（逐张量 `local_numel > 0 且 dtype_bytes > 0`）+ **B2**（dtype ∈ {1,2,4,8}）
+ **B3**（`layer_total_bytes > 0`）。一张 `?` 图连契约校验都过不去，**不可能**冒充可用图混进 A/B
表里给出一个"看起来很小"的峰值。

抽取侧要补：
1. `infer_shapes(dag, input_shapes, dims_ctx)` 必须被调，且**种子自动化**（今天要手喂，
   `shape_infer.py:260`）；
2. `consumer._SYM2FIELD`（`consumer.py:19-34`，14 个符号）补 dsv4 缺的 8 个：
   `index_n_heads / index_head_dim / index_topk / compress_ratio / csa_window_size /
   o_groups / o_lora_rank / hc_mult`；不补就一律落 `unresolved`；
3. **shard 应用**：抽出的张量无 TP/EP placement 标注。要么给张量补 shard 轴再走
   `shape_eval.resolve_tensor` 的整除机制，要么在 adapter 里按 op 类型推。
   ⚠ 这一步的正确性**无法**由单层契约验证 → 落在 **B4** 与 A/B 上（§6）。

### 4.2 权重被当成 activation save 计（88 / 236 MiB）

实测（评估文档 §7.2）：给了种子之后 `FFNGroupedGEMM` 算出 `total_bytes = 247463936`（236.0 MiB），
其中 `w1`(58.7 MB) + `w2`(29.4 MB) = **88 MiB 是权重**。`OpDAG` 没有 `is_weight` 概念，
权重只是普通 `ins`，于是 `derive_saves` 把它们当 saved 激活。

**契约的逼法是三层，缺一层就能溜**：

| 溜的方式 | 挡它的规则 |
|---|---|
| 权重进了 `saves`，且**标对了** `is_weight=True` | **W2**：`saves` 里不得有 `is_weight=True` 的张量 |
| 把权重当成某个 op 的 `output`（"算出来的") | **W3** / **W4**：`output.is_weight == False`；权重不得是任何 op 的产物 |
| 权重**压根没进** `params`（于是全程当激活） | **B4** `validate_param_census(layer, 参考字节)` —— param 字节会短一截（负差），A/B 一比即现；差额方向在错误信息里明写 |
| 同名张量一处当权重、一处当激活 | **W5** |
| `params` 里塞了非权重 | **W1** |

⚠ 契约**不**要求 weight 型 `input` 出现在该 op 的 `params` 里：**tie（权重共享）是真实建模形态** ——
`tie_word_embeddings=True` 时 lm_head 复用 embedding 的 `emb_w`，故意 `params=[]` 以免
vocab×H 持久量重复计（`layers/head.py:84-88`、`:220-222`；qwen2/dsv4-mtp 实测各有一处）。
契约只要求它 `is_weight=True`（liveness 的 `alloc()` 据此跳过激活记账）。
这条豁免有专门的测试 `test_tied_weight_input_without_params_is_legal`。

另一个辅助探针：`layer_entry_names(layer)`（非权重且无生产者的张量 = 图入口）。手写侧实测**每层 ≤2 个**
（`attn_streams` / `ffn_streams` / `hc_streams`，全是真激活）。抽取侧漏标的权重会**冒充图入口**
出现在这里 → A/B 时逐层比对该集合即可定位。

## 5. 明确**不要求**从源里推的东西

- `workspace_bytes` / `bwd_scratch_bytes` 是 **kernel 实现细节**，源码里读不出来。契约只要求
  "是个 ≥0 的 int"，**继续沿用手写 / profiler 标定**。这是有意的取舍：把它们也塞进"源真值"
  会逼出一堆编造的常数。
- `collectives`：`liveness` / `structure_mem` 都不读（通信缓冲走 `mem_timeline` 既有标定），可空。
- 反向节点：**不需要**显式给。liveness 从 `saves` + grad 可达性**导出**
  （`BwdNode.reads = fwd[i].saves` 就是 `PIN` 的语义 —— 概念上完全对上）。

## 6. 交接：`extracted` 怎么接上来

1. 在 `cost_eval/opdag/to_resolved.py` 暴露**约定入口点**（大小写、签名都对死）：

   ```python
   def resolve_graph(model_spec, parallel_model):
       """→ 带 `.stages: {stage: [ResolvedLayer, ...]}` 的对象，逐层满足 contract.py。"""
   ```

   `cost_eval/liveness/sources.py::_probe_extracted()` 会在首次询问 graph source 时**惰性探测**
   并注册成 `extracted`。文件不存在 → 静默跳过（预期状态）；**文件存在但入口点缺失 → fail-loud**
   （"抽取来源明明在了却悄悄不参与 A/B"是最坏的假绿）。`EXTRACTED_ENTRY_POINT` 常量与本约定
   由 `test_extracted_entry_point_is_documented_and_probed` 钉住。

2. 自检（不必等验收门）：

   ```python
   from cost_eval.liveness import resolve_graph, validate_resolved_graph
   assert validate_resolved_graph(resolve_graph(spec, pm, "extracted")) == ()
   ```

3. A/B（**同一个仿真器**，逐层比 param 字节 + 逐 stage 比峰值）：

   ```bash
   python tools/liveness_ab_validate.py --sources hand_spec,extracted --grad-mode chain2 \
          --deltas --dump-top 20 --gate
   ```

   验收门会自动多出 `lv:extracted` 一列、多一组 `extracted/re` 比值，并对 `extracted`
   **施加同样的两条 ×1 结构不变量**（`tests/test_acceptance_gate.py::
   test_model_satisfies_the_same_x1_invariants` 已参数化，不需要新写断言）。

4. 逐层 param 字节对账（**B4**）：

   ```python
   from cost_eval.liveness import layer_param_bytes, validate_param_census
   for st in sorted(hand.stages):
       for h, e in zip(hand.stages[st], extracted.stages[st]):
           assert validate_param_census(e, layer_param_bytes(h)) == ()
   ```

## 7. 尚存近似 / 未决（如实列出）

- **shard 正确性无法单层验证**：`local_numel` 是否真的按 TP/EP/CP 除对了，契约管不到（它只看
  "是不是正数"）。唯一探针是 B4 + A/B 逐 stage 峰值比对。抽取侧若"全局 numel 忘了除"，
  param 字节会**成倍**偏大 → B4 立刻炸；但若只错了某一个轴，可能只表现为峰值偏差。
- **`dim0` 的语义强度**：契约要求它在且 ≥0，但 `0` 是合法的"未知"（`structure_mem.py:95` 明确
  退回 total-numel 整除判定）。抽取侧给 0 能过契约、却可能与 runtime 的 FSDP 分片判定不一致。
  今天没有机械手段区分"确实不需要"与"忘了填"。
- **`op.type` 的取值域**没有被契约约束成枚举。`structure_mem` 的 norm/flash/moe_gemm 判据与
  `RecomputeSpec.op_matches` 都按类型字符串工作 → 抽取侧若用了别的拼法，会**静默**走不同分支。
  建议后续把类型词表也纳入契约（目前只在 §2 表里以自然语言记录）。
- **G2/G3 是派生自洽检查**，跑的是 `build_layer_graph`；它们能抓"detached 张量却被保留"，
  但抓不到"该 detach 的没 detach"（那需要源侧真值，属抽取侧 P1#11 的职责）。
- `workspace_bytes` / `bwd_scratch_bytes` 仍是手写/标定值 —— 见 §5，这是有意保留的近似。

## Related

- [`opdag_coverage_assessment_2026-07-25.md`](opdag_coverage_assessment_2026-07-25.md) —— 抽取侧
  逐组件覆盖度实测 + P0/P1/P2 gap 清单（本文 §2 的"抽取侧现状"列全部引它）。
- [`acceptance_harness_2026-07-25.md`](acceptance_harness_2026-07-25.md) —— 验收门与本契约的
  施工记录、八跑结果表、逐字节中立证明。
