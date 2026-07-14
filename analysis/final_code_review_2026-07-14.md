# pynative-cost-evaluator 最终代码检视报告（功能问题、代码缺陷与真机证据汇总）

> 日期：2026-07-14；当前代码：`feat/unified-llm-modelspec` @ `7b217b6`
> 本文是本轮检视的**唯一主报告**。`analysis/code_review_2026-07-14.md` 是早期静态审查快照，`analysis/realmachine/review_evidence_2026-07-14.md` 是实验附件；两者的结论、后续修正和未覆盖问题均已合并到本文。

## 1. 最终结论

该项目已经具备一个结构清晰、可解释、在若干 DeepSeek-V3/DeepSeek-V4 缩层配置上经过标定的显存评估框架；但它目前仍不适合作为“任意多维混合并行策略的安全 OOM 判定器”直接驱动自动搜索。

核心原因不是单一公式误差，而是以下四类问题同时存在：

1. **生命周期缺项**：真机已经证明，反向阶段规约后的梯度会逐层累计并一直驻留到 optimizer；当前时间线却在每层反向后清零梯度，optimizer 事件的 `grad_buf` 为 0。
2. **配置语义丢失**：MindFormers 导入会静默丢失 CP 算法、VPP、reshard、swap/offload、`exclude_op`、loss parallel 等会改变内存的字段；只有 `model` 段做了较严格的未知字段检查。
3. **手写结构图与运行时漂移**：router、TP embedding/lm_head、final norm 等结构不完整；独立 DSA 明确省略 indexer KL loss；源码提取的 op-DAG 尚未成为评估主链路的约束。
4. **抽象粒度不够**：当前以“整层”为 FSDP gather、重计算、反向 scratch 的主要单位，无法忠实表达子模块 wrap、多个 checkpoint island、累积梯度、PP P2P buffer 和不均衡 MoE 路由。

本轮当前代码上可直接复现 **6 个严格 expected-failure**；其中“optimizer 前累计梯度缺失”同时得到两卡 Ascend 910B3 真机确认。测试配置下评估器峰值为 `14915.5 MiB`，真机 `max allocated=15415.5 MiB`，预测/真值为 `0.9676`。这个点只说明该已标定配置误差约 `-3.24%`，不能抵消其他策略上的结构性风险。

### 1.1 影响方向

| 方向 | 主要问题 | 后果 |
|---|---|---|
| OOM 不安全低估 | 累计梯度、独立 DSA KL loss、`exclude_op`/dropout 等被忽略、mHC 张量身份冲突、CP/P2P buffer 缺项 | 设备可能实际 OOM，但报告显示可运行 |
| 保守高估 | TP embedding/head 未切、TP workspace 未缩放、整层 FSDP gather、逐层 `bwd_scratch` 求和 | 错杀可行策略，搜索结果偏离最优解 |
| 双向或排序错误 | 配置字段丢失、reshard 无效、MoE 平均负载、PP/VPP 调度近似、经验标定常数 | 单点数值看似合理，但不同策略的相对排序不可靠 |

## 2. 证据口径

| 标签 | 含义 |
|---|---|
| **NPU-CONFIRMED** | 已在 `192.168.9.116` 的 2×Ascend 910B3 上测得运行时证据 |
| **LOCAL-REPRODUCED** | 当前 `7b217b6` 可由本地命令稳定复现 |
| **SOURCE-CONFIRMED** | 当前实现或已核对的 MindFormers 源码直接表明该行为，但尚无对应差分真机实验 |
| **UNVERIFIED-RISK** | 有明确机理风险，仍需专门真机或 profiler 实验确定幅度 |
| **CONTRACT** | API/定义不完整或容易误用，不能仅凭现有实验判定为数值 bug |

## 3. 当前开放问题总表

### 3.1 P0：应在继续扩大策略搜索前修复

| ID | 类型 | 证据 | 方向 | 问题 |
|---|---|---|---|---|
| P0-01 | 功能缺陷 | **NPU-CONFIRMED + LOCAL-REPRODUCED** | 低估 | 时间线没有“规约后本地分片梯度逐层累计至 optimizer”的生命周期 |
| P0-02 | 功能缺陷/配置缺陷 | **LOCAL-REPRODUCED** | 双向 | MindFormers 并行、重算、swap/offload 等关键字段被默认值静默替换，非 `model` 段也没有完整 fail-loud |
| P0-03 | 死配置 | **LOCAL-REPRODUCED** | 双向 | `ParallelConfig.reshard_after_forward` 存在，但时间线始终按 default 立即释放 gather buffer |
| P0-04 | 并行建模缺陷 | **LOCAL-REPRODUCED + SOURCE-CONFIRMED** | 高估/排序错误 | embedding、lm_head 与默认 loss 没有按当前 MindFormers 的 TP/vocab-parallel 路径切分 |
| P0-05 | 注意力功能缺口 | **SOURCE-CONFIRMED，真机未覆盖** | 可能大幅低估 | 独立 `attn_type=dsa` 明确省略 indexer KL loss 的全量 QK 瞬态 |

### 3.2 P1：会显著影响特定模型或并行策略

| ID | 类型 | 证据 | 方向 | 问题 |
|---|---|---|---|---|
| P1-01 | 结构图缺项 | **LOCAL-REPRODUCED + NPU 旁证** | 低估 | MoE router 权重未放入 `OpSpec.params`；norm/final norm/shared gate 等小参数也不完整 |
| P1-02 | 重计算语义缺陷 | **LOCAL-REPRODUCED + SOURCE-CONFIRMED** | 双向 | `exclude_op` 被丢弃；零命中选择器仍被视为 select；`mode=full` 未给层列表时静默得到空集合 |
| P1-03 | 代码缺陷 | **LOCAL-REPRODUCED** | 低估或错算 | mHC+MTP 同一 resolved layer 内用同名 `x` 表示 `H` 与 `nH` 两种 shape，按名去重错误 |
| P1-04 | 配置/结构缺陷 | **LOCAL-REPRODUCED** | 低估/高估 | `embedding_params_dtype_bytes` 被解析但从未进入 TensorRef；配置为 4B 时实际 resolved 权重仍为 2B |
| P1-05 | 配置入口缺陷 | **LOCAL-REPRODUCED** | 功能不可达 | builder 支持独立 DSA，但 `_infer_attn_type` 只接受 `dsv4_hybrid`，`experimental_attention_variant=dsa` 直接报错 |
| P1-06 | 错误耦合 | **SOURCE-CONFIRMED** | 双向 | `loss_type` 固定为 `logsoftmax_nll`；`cross_entropy_fused` 被错误绑定到 DSA kernel fusion 开关 |
| P1-07 | 重计算模型缺陷 | **SOURCE-CONFIRMED** | 双向 | 非连续选中 op 被压缩成一个 selected 集合，未按真实 checkpoint regions/islands 分段求峰 |
| P1-08 | 生命周期建模缺陷 | **SOURCE-CONFIRMED** | 高估为主 | 一层所有 op 的 `bwd_scratch` 直接求和，而这些 scratch 通常按 op 顺序出现，不一定同时存活 |
| P1-09 | attention 生命周期缺陷 | **SOURCE-CONFIRMED，需 profiler** | 双向 | Flash softmax max/sum 被注释为供反向使用，却建成仅前向存在的 `workspace`；同时 workspace 只按 CP、未按本地 TP heads 缩放 |
| P1-10 | 输入校验缺陷 | **LOCAL-REPRODUCED** | 低估/错图 | DSv4 `compress_ratio=2/3` 会被当作稀疏 HCA 类路径接受；缺 ratio 集合、长度、整除和 `o_groups` 校验 |
| P1-11 | 输入校验缺陷 | **LOCAL-REPRODUCED** | 低估/错图 | `H % n_heads != 0` 时 `head_dim` 直接 floor；MoE freq 长度、top-k、capacity 对齐等也缺统一校验 |
| P1-12 | MoE 功能缺口 | **SOURCE-CONFIRMED，需真机** | 双向 | 路由 token 数只用平均平衡值并 floor，不表达专家倾斜、capacity ceil、padding/对齐和最忙 rank |
| P1-13 | CP 功能缺口 | **SOURCE-CONFIRMED** | 双向 | fused GQA QKV 无法单独标 KV all-gather；ring/ulysses/colossal 的通信双缓冲与 kernel workspace 未完整建模 |
| P1-14 | FSDP 抽象缺口 | **SOURCE-CONFIRMED** | 高估/排序错误 | 以整层参数做 gather/prefetch，未表达 runtime 对 router、experts、output 等子模块独立 wrap 的生命周期 |
| P1-15 | PP/VPP 功能缺口 | **SOURCE-CONFIRMED，VPP 真机未完成** | 双向 | 只建 1F1B/VPP 近似；无 P2P send/recv activation buffer、GPipe/zero-bubble/overlap；显式 stage 配额下 VPP chunk 仍为近似 |
| P1-16 | 策略合法性缺口 | **SOURCE-CONFIRMED** | 排序错误 | 缺 runtime feasibility validator；例如当前测试文档已记录 MindFormers `PP>1` 不支持 activation swap，但评估器仍接受组合 |
| P1-17 | UI 功能缺陷 | **SOURCE-CONFIRMED** | 双向 | YAML 回填后评估仍固定 64 GiB、FP32 AdamW、swap 关闭；`dp_replicate`、reshard、offload、prefetch、loss 类型等无法往返 |
| P1-18 | UI 代码缺陷 | **SOURCE-CONFIRMED** | 导入失败 | 老式 YAML 缺 `num_hidden_layers` 时，`_mf_adapt` 在页面兜底之前用 `N=0` 计算嵌套 offset |
| P1-19 | 优化器/卸载缺口 | **SOURCE-CONFIRMED** | 双向 | 仅支持 Adam/AdamW；optimizer scratch 固定 `K_OPT=6`；一个 `cpu_offload` 布尔量无法表达 param/grad/optimizer 分别卸载 |

### 3.3 P2：契约、工程质量与验证覆盖问题

| ID | 类型 | 证据 | 问题 |
|---|---|---|---|
| P2-01 | API 契约 | **CONTRACT + NPU 口径证据** | `.oom` 只表示 allocated；真实容量还受 reserved/内存池碎片影响，调用方容易把单一布尔值当最终 OOM 结论 |
| P2-02 | 架构债务 | **SOURCE-CONFIRMED** | `cost_eval/opdag/` 只做交叉验证，Evaluator 不消费其结果；手写 LayerSpec 仍是唯一生产事实源 |
| P2-03 | 结构覆盖 | **SOURCE-CONFIRMED** | 主干 final RMSNorm、部分 norm 参数、非零 dropout mask、完整 attention mask、部分 qk norm/偏置等未建或被当作“内存中性”忽略 |
| P2-04 | 标定可移植性 | **SOURCE-CONFIRMED** | `K_CE`、`K_OPT`、`kept_frag_factor=1.9` 来自少量 DSv3/DSv4 regime，跨模型、序列、软件版本的稳定性未证明 |
| P2-05 | 预设可信度 | **SOURCE-CONFIRMED** | DSv4-Flash/Pro 页面预设的 compress ratios 使用循环近似；671B、GLM-5、V4 全尺寸主要是外推而非全尺寸真机回归 |
| P2-06 | 可观测性 | **SOURCE-CONFIRMED** | timeline 事件名只有 `fwd:<layer>`/`bwd@<layer>`，无 microbatch/chunk 标识；重复事件难以和 profiler 对齐 |
| P2-07 | 测试缺口 | **SOURCE-CONFIRMED** | 既有测试以自洽、golden 和 ±2% 参数守恒为主，可能同时固化错误；TP vocab、reshard、累计梯度、张量身份等直到本轮才有反例 |
| P2-08 | 代码/文档漂移 | **SOURCE-CONFIRMED** | `StaticMem` 文档仍写 param+grad+opt，但实现已剔除 grad；CP head 注释、K_CE 注释与当前代码也有冲突，部分 MindFormers 行号已漂移 |

## 4. 关键缺陷详解与证据

### P0-01：累计梯度生命周期缺失

代码位置：

- `cost_eval/specs.py:49-51`：持久态明确剔除 grad。
- `cost_eval/mem_timeline.py:544`：反向仅放入“当前层 full grad”。
- `cost_eval/mem_timeline.py:599`：每层反向事件后立即 `B.grad_buf=0`。
- `cost_eval/mem_timeline.py:621-624`：optimizer 前再次把 `grad_buf` 清零。

两卡 NPU 探针在 optimizer 调用前测得：

```text
rank 0/1: buffers=74, param_grad_buffers=74
logical grad_bytes=3,962,450,432 = 3778.9 MiB
current allocated before optimizer=7566.1 MiB
current allocated after zero_grad=5676.6 MiB
```

FSDP-2 的物理梯度分片应为：

```text
3778.9 / 2 = 1889.45 MiB
7566.1 - 5676.6 = 1889.5 MiB
```

两者吻合。正确语义是：梯度不是跨 step 永久驻留，而是在**本 step 的反向后半段到 optimizer 之间逐层累计驻留**。当前评估器的反例为：

```text
event='optstep', grad_buf=0, optstep=2780037120
```

修复时应把“当前层 reduce-scatter 前 full grad”和“已完成层的本地 reduced grad shard”分成两个桶，前者短暂、后者累积，并在 optimizer/zero_grad 后释放。

### P0-02：配置适配器静默丢失内存语义

`cost_eval/configs/from_mindformers.py:461-496` 只映射部分字段。当前本地输入：

```text
context_parallel_method=ulysses
pipeline_parallel_interleave_num=4
reshard_after_forward_policy=never
```

实际得到：

```text
('colossal', 1, 'default')
```

此外：

- `exclude_op` 在 `_build_recompute` 中没有消费；本地探针显示返回的 selector 仍包含 `flash`。
- `SwapSpec()` 在 `from_mindformers_dict` 中硬编码为 disabled。
- `dense_fsdp_shard_size`、CPU offload、调度/overlap、loss parallel 等无映射。
- 未知 `parallelism` 键（包括拼错的 `context_parallel_methd`）不会报错，而是继续使用 `colossal`。
- `_IGNORED_MODEL_KEYS` 把 dropout、mask compression、多个 fusion flag 当作“内存中性”，但这些字段可能改变保存张量或 workspace；这与显存评估器的目标不一致。

这类错误最危险之处是输出仍然是合理数字，用户无法知道评估的是另一份配置。

### P0-03：reshard 是未接线的死配置

`ParallelConfig` 在 `cost_eval/specs.py:27` 定义 `always|never|default`，但 `cost_eval/mem_timeline.py:511` 无条件执行：

```python
B.gather_buf = 0   # reshard_after_forward(default)
```

本地对 `always` 与 `never` 比较完整 timeline，逐事件 `event/total/gather_buf` signature 完全相同。实际实现还需要按 FSDP wrap 单元区分；例如 output layer/root 模块可能采用与普通 decoder 不同的默认重分片策略。

### P0-04：TP 下 vocab 栈没有按运行时切分

`cost_eval/layers/head.py:21` 的 `emb_w` 和 `:75` 的 `head_w` 没有 TP shard；只有显式设置 `loss_type=vocab_parallel_ce` 时 loss 张量才按 TP 切，但 adapter 在 `from_mindformers.py:382` 始终写死 `logsoftmax_nll`。

本地反例：

```text
TP=1 embedding/head local_numel = 231,669,760
TP=2 embedding/head local_numel = 231,669,760
期望（当前 MindFormers row/column-wise vocab parallel）=115,834,880
```

这通常使 TP 策略被高估，并让 loss 峰对 TP 不敏感，直接破坏策略排序。

### P0-05：独立 DSA 省略 indexer KL loss

`cost_eval/layers/dsa.py:44-50` 明确说明整个 indexer KL loss 未建。静态图路径会产生 QK 分数 `[B,n_heads,S,S]` 的 FP32 瞬态；在 `B=1,n_heads=128,S=4096` 时，单张量理论字节为 `8.0 GiB`，再按真实 TP/CP/kernel 分块方式缩放。

当前 builder 只给 indexer 放入 `4*B*S*S`，没有 `n_heads` 因子，也没有 KL loss 内其他共存量。该路径仍标记为 pre-estimate；本轮 NPU 使用的是 `dsv4_hybrid` fused DSA，不能作为独立 `attn_type=dsa` 的验证。

### P1-01：参数图不守恒且现有测试阈值掩盖遗漏

`cost_eval/layers/ffn.py:123-124` 的 router 没有 params。本地得到所有 router 的参数长度 `[0]`，而真机 optimizer 参数列表包含 `decoder.layers.N.mlp.router.weight`。

类似地，主干 norm/final norm/shared gate 等小参数没有完整进入 LayerSpec。`tests/test_param_conservation.py` 允许 ±2% 误差；router 和 norm 对大模型参数总量占比很小，因此该测试仍会通过。参数守恒应改成逐模块精确对账，而不是总量宽容带。

### P1-02/P1-07：选择性重计算语义不完整

当前问题包括：

1. adapter 丢弃 `exclude_op`。
2. `parse_select_cfg('definitely_missing:0-1')` 返回无错误；`RecomputeSpec.is_select=True`，但实际命中 op 数为 0。
3. 零命中层仍会让 `_stage_no_recompute=False`，从而改变 CE 的经验分支；`_is_kept` 又用 selector 与固定 marker 的**精确集合交集**判断 FFN 是否保留，和真正的子串匹配语义不一致。
4. `mode=full` 但没有 `full_recompute_layer` 时返回 `full_layers=set()`，不报错。
5. `estimate_select_memory` 把所有 selected/nonselected op 各压成一个列表，没有显式构建真实 checkpoint islands；跨岛张量活性和边界输入可能被错误合并。

修复后每个选择器必须报告命中模块/算子数；0 命中应 fail-loud。checkpoint region 应成为一等对象，而不只是 op 的布尔集合。

### P1-03：mHC/MTP 张量身份冲突

`cost_eval/layers/residual.py:53-58` 放大残差张量时保留原 name；MTP 又在 `cost_eval/layers/head.py:132` 使用 `x=[S,B,H]`。本地 resolved MTP layer 中同名 `x` 同时为：

```text
7,340,032 elements
29,360,128 elements
```

`structure_mem._forward_max_live`、params/saves 去重和 `ShapeEval.produced` 都用 name 作为唯一身份，其中第一次出现的 size 会覆盖后续语义。这不仅影响展示，也会直接改变 recompute scratch、激活峰和通信边。

### P1-04：已解析的参数 dtype 没有生效

`LLMConfig.embedding_params_dtype_bytes` 默认/导入值为 4，但 `build_embedding_ops` 与 `build_head_and_loss_ops` 创建权重时未传 `dtype_bytes`。本地输出：

```text
embedding_params_dtype_config = 4
resolved embedding/head weight dtype = 2, 2
```

因此 embedding/head 的 gather bytes 使用 compute dtype，而不是配置指定的参数 dtype。该字段目前是典型 dead configuration。

### P1-06：loss 与 DSA fusion 错误耦合

`from_mindformers.py:382` 固定 `loss_type='logsoftmax_nll'`；`:416-418` 又用 `_dsa_fused(model)` 同时设置 attention fusion 和 `cross_entropy_fused`。两者是独立 kernel/配置：

- attention fused、CE unfused 时会低估 CE 中间量；
- attention unfused、CE fused 时会高估 CE；
- TP 的 vocab-parallel CE 完全无法从 YAML 导入。

### P1-08/P1-09：scratch、workspace 与 saved tensor 生命周期混淆

- `structure_mem.py:171` 对一层所有 `bwd_scratch` 求和；这隐含所有反向临时量同时存活，通常应按逆序事件和依赖关系取 max-live。
- `attention.py:18-25` 说明 softmax max/sum 供 FlashAttentionGrad 反向使用，却把 `64*B*n_heads*S` 建成前向 `workspace`，前向后立即释放；真正保存到反向的 `lse` 又只有 `[S,B,n_heads]` 且默认 dtype。
- `ShapeEval` 对 workspace/bwd scratch 只按 CP 缩放；`FLASH_LSE_WS` 使用全局 `n_heads`，TP>1 时没有除本地头数，产生系统性高估。

应把“kernel 临时 workspace”和“autograd 保存输出”分开，并允许 workspace 表达式声明 TP/CP/EP shard 轴。

### P1-10/P1-11：结构配置缺少统一合法性校验

本地探针已证明：

```text
csa_compress_ratios=(2,3,2,3) 被接受并生成 r2/r3 层
hidden_size=10, num_attention_heads=3 -> head_dim=3（floor，无报错）
```

`dsv4_hybrid.py:73-77` 把任何非 0/1 ratio 都当稀疏路径，仅 ratio 4 开 indexer，其余都近似 HCA；`S//ratio` 也没有整除检查。类似缺口还包括 ratio 列表长度、`o_groups`、MoE layer freq 长度、top-k、capacity/padding 对齐等。

### P1-12：MoE 使用平均负载而非 OOM 最坏 rank

`cost_eval/layers/ffn.py:17-20` 使用：

```text
T_local = S * B * topk * capacity_factor / ep
```

并在 shape 求值时 floor。真实 MoE 会受路由倾斜、expert capacity 的 ceil、token padding、grouped GEMM 对齐和最忙 expert/rank 影响。平均值适合吞吐估计，不足以作为 OOM 安全边界。报告至少应同时给出 `balanced`、`capacity-bound` 和可配置 skew percentile 三种口径。

### P1-13/P1-14：CP 与 FSDP 抽象粒度不足

- `attention.py:51-53` 已承认 fused GQA QKV 无法单独标 KV，因此 colossal CP 的 KV all-gather buffer 欠建。
- CP 目前主要通过张量 numel 除法表达，缺 ring/ulysses 通信在飞缓冲和不同 fused kernel 的 workspace。
- 时间线把整层 `param_full_bytes` 当作 FSDP gather 单元；当前 MindFormers 会对子模块分别 wrap，router、experts、output layer 的 gather/reshard 生命周期并不等同于整层。

这两项会让单个策略的数值和不同 TP/CP/FSDP 策略的相对排序同时偏移。

### P1-15/P1-16：PP 调度和策略可实现性

当前实现覆盖 plain 1F1B 和一个基于 Megatron 的 VPP 近似，但没有：

- stage 间 send/recv activation buffer；
- GPipe、zero-bubble、overlap 等调度；
- 显式 per-chunk layer ranges；
- MindFormers hyper_parallel VPP 的完整真机校验；
- runtime 约束矩阵。

`tests/test_swap_offload.py` 的源码核对已记录 `PP>1` 不支持 activation swap，但 Evaluator/页面仍能组合 PP+swap。搜索器若不先做 feasibility validation，可能把运行时无法构造的策略评为最优。

### P1-17/P1-18：YAML 导入不是保真 round-trip

页面最终评估在 `serve_explorer.py:500-506` 固定或重建：

- `64 GiB` 设备容量；
- `AdamW(params_fp32=True)`；
- `SwapSpec()`；
- 无 `dp_replicate`、reshard、offload、prefetch 等输入。

`_bundle_to_fields` 只回填 UI 可表达的字段。老式 YAML 的嵌套 offset 又在必需模型字段兜底之前计算，导致依赖类内默认层数的配置导入失败。README 中“解析回填全部对话框”的表述应收窄为“回填 UI 支持子集”。

## 5. 真机证据与结论修正

### 5.1 环境

- 服务器：`192.168.9.116`，容器 `shb.ms.2.9`
- NPU：2×Ascend 910B3，物理卡 `6,7`
- MindSpore：2.10
- MindFormers：`/home/suhaibo/workspace/deepseek_v4/mindformers` @ `97466872d`
- `git pull --ff-only`：`Already up to date.`
- 配置：DSv4 hybrid 4L，seq 2048，FSDP-2，TP/CP/EP/PP=1，fused DSA，无重计算，1 step
- NPU 证据采集时评估器为 `9f2372c` 加当时工作区 DSA 改动；当前 `7b217b6` 再次由本地测试确认累计梯度缺陷仍存在

### 5.2 峰值

```text
NPU max allocated : 15415.5 MiB
NPU max reserved  : 16074.0--16096.0 MiB
评估器预测         : 14915.5 MiB @ bwd@5
pred / real       : 0.9676
```

当前配置全局峰由 fused loss 区域主导，因此累计梯度缺失没有把总峰误差扩大到完整的 `1889.5 MiB`。当 TP/vocab-parallel CE 压低 loss、或 optimizer/transformer backward 成为主峰时，该缺陷会更明显。

### 5.3 既有验证矩阵的补充校正

早期真机总结曾写“最差预测/真值约 0.925、仅两个 OOM 不安全点”。早期对抗审查使用 `analysis/realmachine/validation_matrix.md` 保存的锚点并以当时评估器重新补算后，还得到同族更低点：

```text
dp2-none-8L          : 0.919
cp2-ulysses-none     : 0.921
cp2 × select-attn    : 1.003
```

前两项仍属于已归因的 no-recompute/CE-lifetime 误差族；第三项是正面结果，说明 select-attn 的标定 margin 在该 CP 点上也成立。主报告不再沿用“0.925 是最差点”的表述。

### 5.4 必须保留的修正结论

1. **梯度是 step-scoped cumulative，不是跨 step 永久驻留。**
2. **独立 DSA 尚未被本次真机覆盖。** `dsv4_hybrid` fused DSA 结果不能替代 `attn_type=dsa` 的 indexer KL loss 验证。
3. **reserved > allocated 已确认，但不能据此断言现有 `.oom` 布尔一定是 bug。** 当前实验没有在设备容量临界点触发 OOM；应把 allocated 和 reserved 两种判定显式拆开。
4. **测试配置误差约 3.24%，不代表任意策略均在此误差带内。** 该点处于已标定模型/算子族内。

## 6. 本地可执行证据

运行：

```powershell
python -m pytest --runxfail -q tests/test_review_evidence.py
```

当前结果为 `6 failed`，且均是目标断言失败，不是导入或环境失败：

1. 配置实际 `('colossal',1,'default')`，期望 `('ulysses',4,'never')`。
2. `reshard=always` 与 `never` 的完整 timeline signature 相同。
3. router 参数数量 `[0]`。
4. TP=2 embedding/head 仍为 `231669760`，期望 `115834880`。
5. MTP layer 同名 `x` 有 `[7340032,29360128]` 两种 numel。
6. `optstep.breakdown.grad_buf == 0`。

常规回归模式：

```powershell
python -m pytest -q -rxX tests/test_review_evidence.py
```

结果应为 `6 xfailed`、退出码 0。所有标记均为 `strict=True`，实现修复后会先变成 XPASS 并让 suite 失败，提醒维护者移除 xfail。

额外本地源码行为探针还确认：

```text
embedding params config dtype=4, resolved dtype=2
compress_ratio 2/3 accepted
full recompute without layer list -> empty set
unknown parallel key ignored
exclude_op dropped
experimental_attention_variant=dsa rejected
H=10, heads=3 -> head_dim=3 by floor
unknown recompute selector -> is_select=True, matched_ops=0
```

## 7. 早期报告问题的当前状态

| 早期问题 | 当前状态 | 说明 |
|---|---|---|
| K_CE 由全局 mode 控制，per-stage select 低估 | **已修复** | `df078d2` 改为按 stage 是否有重计算判断 |
| YAML select 漏掉 GQA `qkv` | **已修复** | `df078d2` 统一 UI/adapter selector 来源并补 `qkv` |
| 老式 recompute 列表静默转换错误 | **部分修复** | UI 现在明确警告“不支持、不转换”；新式 `exclude_op` 和零命中仍未修 |
| legacy DP 无条件映射 FSDP shard | **已修复** | `9f2372c` 按 `enable_parallel_optimizer` 分流 `dp_shard/dp_replicate` |
| fp32 residual + bf16 norm 被静默忽略 | **已修复** | `9f2372c` 为该组合增加 fail-loud guard |
| VPP 层放置按连续块近似 | **主要路径已修复** | `7b217b6` 改 round-robin；显式 stage 配额下 per-chunk ranges 仍近似 |
| Megatron `m==pp` all-warmup 未实现 | **不再判为已确认 bug** | 当前依据表明 MindFormers hyper_parallel 不一定采用该特例，保留为 VPP 真机待验证项 |
| DSA 只是 MLA 上界别名 | **已演进** | `615660f` 已加入独立 DSA pre-estimate；但 adapter 不可达、KL loss 仍省略、无真机锚点 |
| YAML 嵌套 offset 在缺省层数兜底前计算 | **仍开放** | 见 P1-18 |
| UI 选择器拼错后静默空转 | **仍开放** | 见 P1-02 |
| timeline 无 microbatch 标识 | **仍开放** | 见 P2-06 |
| 真机总结把 0.925 写成最差点 | **已在本文校正** | 验证矩阵还有 0.919/0.921；同时补入正面点 1.003 |
| 多模型真机回归被表述得过宽 | **仍需收窄** | 真机重点仍是少量 DSv3/DSv4 缩层配置，全尺寸模型以外推为主 |

## 8. 修复顺序与验收标准

### 第一阶段：恢复 OOM 与策略排序的基本可信度

1. 增加 `reduced_grad_live` 累计桶；按 dense/expert 的实际 FSDP shard 度计算，optimizer 后清零。
2. 为 adapter 建立字段 schema：所有未映射的 parallel/recompute/swap/offload/loss 字段必须报错或显式 warning，禁止静默默认。
3. 实现 per-wrap `reshard_after_forward`，不能只用全局布尔量。
4. 对齐当前 MindFormers TP embedding、output layer 和 vocab-parallel loss。
5. 独立建模或 fail-loud 禁用 DSA indexer KL loss 路径，直到真机验证完成。

验收：

- optimizer 前 `grad_buf/reduced_grad_live` 与 FSDP-2 真机 `1889.5 MiB` 对齐。
- `context_parallel_method/interleave/reshard` round-trip 字段完全相等。
- `reshard=always/default/never` 的 timeline 产生预期不同的 gather 生命周期。
- TP=2 的 embedding/head local 参数与 vocab loss 主张量均约为 TP=1 的一半。

### 第二阶段：修结构图与重计算

1. 补 router、norm、final norm、shared gate 等参数，参数守恒改为逐模块精确对账。
2. TensorRef 使用稳定唯一 ID，name 只用于显示；同一 ID 的 shape/dtype 必须一致。
3. 把 checkpoint region/island 建成显式结构；实现 `exclude_op` 和命中数校验。
4. 分离 saved tensor、forward workspace、backward scratch 的生命周期，按 op 时间线取峰。
5. 让 workspace/scratch 声明 TP/CP/EP shard 轴。

验收：

- 任一 resolved layer 不再存在“同一 tensor ID 多 shape”。
- selector 0 命中、`full` 空层列表、非法 ratio/维度均 fail-loud。
- select-all/none 两端仍分别退化到 full/None；非连续 islands 有独立单测。

### 第三阶段：策略可实现性与泛化

1. 建立 runtime feasibility matrix，先拒绝 PP+swap 等 MindFormers 不支持组合。
2. 增加 MoE capacity/skew、PP P2P buffer、子模块 FSDP wrap 和 CP kernel buffer。
3. 让 op-DAG 提取结果参与 LayerSpec 校验或生成，减少手写图漂移。
4. 把 `.oom` 拆成 `allocated_oom` 与 `reserved_capacity_exceeded`，同时报告两种余量。
5. 扩充 NPU 差分矩阵并按软件版本保存基线。

## 9. 仍需执行的 NPU 差分矩阵

1. dense TP=1/2：embedding、output layer、vocab-parallel CE 的每卡缩放。
2. FSDP-2 `reshard=always/default/never`：forward-end、optimizer 前 allocated 和全局峰。
3. 独立 `attn_type=dsa`：indexer KL loss 的真实 kernel、分块和保存量。
4. mHC+MTP：修正 tensor identity 前后的 forward/backward 工作集。
5. PP/VPP：至少 pp=2/vpp=2，记录 P2P buffer、每 chunk 在飞微批和 stage 峰。
6. MoE skew：构造不均衡路由，比较平均 rank 与最忙 rank 峰值。
7. 临界 OOM：在可控卡上逐步逼近容量，分别验证 allocated/reserved 判据。

## 10. 复现资产

- 本地 expected-failure：`tests/test_review_evidence.py`
- NPU 梯度探针：`analysis/realmachine/run_main_grad_probe.py`
- NPU 启动脚本：`analysis/realmachine/run_main_grad_probe.sh`
- 真机原始结论附件：`analysis/realmachine/review_evidence_2026-07-14.md`
- 早期声明审查附件：`analysis/code_review_2026-07-14.md`

最终判定：**可继续作为有证据标签的离线显存分析器使用；在 P0 项修复前，不应把单一峰值或 `.oom=False` 当作任意混合并行策略可运行的充分条件，也不应无约束地用其排序全策略空间。**
