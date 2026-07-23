# Pynative Cost Evaluator：FSDP / MoE / HSDP 参数切分审计

> **结论主线**：当前 evaluator 的 dense/HSDP 主分母是对的，但参数只被归为“dense / expert”两类，无法表达当前 MindFormers 运行时的 TP-replicated shared expert、`ep=1` routed expert 退化路径，以及 FSDP `replicate_params`。因此它只在受限条件下准确，不能视为完整的 FSDP/MoE/HSDP 源码等价模型。
>
> **Evaluator 基线**：`pynative-cost-evaluator`，分支 `feat/unified-llm-modelspec`，commit `7b6e2a21dd28ec570898572d338412b05aaaceba`，2026-07-22。
>
> **Runtime 基线**：`mindformers`，分支 `master`，commit `377c9c34423300eed51ae9bf7e247e2a1eaa86dc`，2026-07-13。
>
> **范围**：默认 folded-FSDP 路径，即 `full_dtensor=False`；重点分析参数、优化器状态、规约后梯度与 FSDP gather 缓冲。时间模型单列说明。

---

## 1. 审计结论

当前实现可以分成三档：

| 场景 | 判断 | 说明 |
|---|---|---|
| 普通 dense 参数，TP/FSDP 首维均可整除 | **基本准确** | TP/EP 先切，FSDP 再切的两阶段逻辑正确。 |
| HSDP / dense grouped-FSDP | **主公式准确** | `dp_replicate` 不进入单卡参数分母；dense 子域 `K` 的建模正确。 |
| routed expert，且 `ep > 1` | **基本准确** | EP 与 eFSDP 的乘积分母正确，EP 度数在持久参数公式中抵消。 |
| shared expert，且 `tp > 1` | **不准确，低估** | evaluator 按 TP-sharded 建模；运行时参数在 TP 上 replicated。 |
| routed expert，`ep = 1` 且 `tp > 1` | **不准确，低估** | evaluator 仍走 eFSDP；运行时不创建 expert FSDP wrap，而是随父层走 dense FSDP。 |
| FSDP 首维不可整除 / 特殊 `replicate_params` | **不准确** | evaluator 使用总元素数 `ceil`；运行时 dense 路径复制整个参数，routed-expert 路径也没有该 `ceil` 语义。 |
| HSDP/MoE 通信时间 | **明显不准确** | 时间模型只接受一个 `dp` 度数，未表达 CP-folded FSDP、eFSDP、dense 子域和 `replicate_params`。 |

因此，当前内存公式可靠的适用域是：

1. `full_dtensor=False`；
2. 参数在运行时实际参与 FSDP shard；
3. FSDP shard 维严格可整除；
4. routed expert 必须确实启用 EP，即 `ep > 1`；
5. 不包含 TP-replicated shared-expert FC 权重，或者 `tp=1`；
6. 只讨论内存主公式，不把当前 timesim 当作 HSDP/MoE 的源码等价通信模型。

---

## 2. 变量定义与并行域

定义并行度：

| 符号 | 配置字段 | 含义 |
|---|---|---|
| $R$ | `dp_replicate` | HSDP/DDP 复制轴大小 |
| $S$ | `dp_shard` | 数据并行中的 FSDP shard 轴大小 |
| $C$ | `cp` | Context Parallel 度数 |
| $T$ | `tp` | Tensor Parallel 度数 |
| $E$ | `ep` | Expert Parallel 度数 |
| $Q$ | `pp` | Pipeline Parallel 度数 |
| $W$ | `world_size` | 总进程数 |

默认 folded-FSDP 布局要求：

$$
W = R \cdot S \cdot C \cdot T \cdot Q
$$

MindFormers 将 CP 折入 dense FSDP 轴，因此完整 dense FSDP shard degree 为：

$$
F = S \cdot C
$$

对应源码：

- evaluator：[`cost_eval/parallel_model.py:L51-L52`](../cost_eval/parallel_model.py#L51-L52)
- runtime：[`mindformers/pynative/distributed/parallel_dims.py:L620-L626`](../../mindformers/mindformers/pynative/distributed/parallel_dims.py#L620-L626)

EP 不是额外乘到 world size 上的独立世界轴，而是从单个 `dp_replicate` 副本内的区域

$$
S \cdot C \cdot T
$$

中 carve 出来。因此必须满足：

$$
E \mid (S \cdot C \cdot T)
$$

routed expert 的 eFSDP degree 为：

$$
F_e = \frac{S \cdot C \cdot T}{E}
$$

对应源码：

- evaluator：[`cost_eval/parallel_model.py:L16-L20`](../cost_eval/parallel_model.py#L16-L20)
- runtime 校验：[`mindformers/pynative/distributed/parallel_dims.py:L127-L147`](../../mindformers/mindformers/pynative/distributed/parallel_dims.py#L127-L147)
- runtime mesh：[`mindformers/pynative/distributed/parallel_dims.py:L234-L270`](../../mindformers/mindformers/pynative/distributed/parallel_dims.py#L234-L270)

如果配置了 `dense_fsdp_shard_size`，令其为 $K$；否则使用完整 dense FSDP 域：

$$
K =
\begin{cases}
K_{\mathrm{cfg}}, & \text{if dense\_fsdp\_shard\_size is configured}, \\
F, & \text{otherwise}.
\end{cases}
$$

且必须满足：

$$
1 \le K \le F,
\qquad
K \mid F
$$

对应源码：

- evaluator 配置校验：[`cost_eval/specs.py:L99-L111`](../cost_eval/specs.py#L99-L111)
- evaluator 有效分母：[`cost_eval/parallel_model.py:L54-L66`](../cost_eval/parallel_model.py#L54-L66)
- runtime 子域 mesh：[`mindformers/pynative/distributed/parallel_dims.py:L421-L497`](../../mindformers/mindformers/pynative/distributed/parallel_dims.py#L421-L497)

---

## 3. 参数切分的通用公式

设 $P$ 是一个逻辑参数在 TP/EP 切分前的全局元素数，$P_{\mathrm{layout}}$ 是完成 TP/EP placement 后、尚未应用 FSDP 的本 rank 参数元素数。

### 3.1 TP/EP placement

若参数在 TP 上切分：

$$
P_{\mathrm{layout}} = \frac{P}{T}
$$

若参数在 TP 上复制：

$$
P_{\mathrm{layout}} = P
$$

若 routed-expert 参数在 EP 上切分：

$$
P_{\mathrm{layout}} = \frac{P}{E}
$$

Evaluator 在 [`cost_eval/shape_eval.py:L68-L99`](../cost_eval/shape_eval.py#L68-L99) 中按 `TensorRef.shard` 对 shape 的对应维做整除；这一步只处理 TP/EP 等图内 placement，不处理 FSDP。

### 3.2 FSDP/HSDP 后的本地元素数

对正常参与 dense FSDP shard 的参数：

$$
N_{\mathrm{rank}} = \frac{P_{\mathrm{layout}}}{K}
$$

对正常参与 routed-expert eFSDP shard 的参数：

$$
N_{\mathrm{rank}} = \frac{P_{\mathrm{layout}}}{F_e}
$$

对运行时 `replicate_params`：

$$
N_{\mathrm{rank}} = P_{\mathrm{layout}}
$$

这里最重要的约束是：运行时按某个实际 tensor 维度进行 FSDP shard，而不是只看 `local_numel` 是否能被 degree 整除。因此正确判据应是：

$$
\mathrm{shape}_{\mathrm{layout}}[d] \bmod G = 0
$$

其中 $d$ 是真实 FSDP shard 维，$G$ 是 $K$ 或 $F_e$。仅有

$$
P_{\mathrm{layout}} \bmod G = 0
$$

并不足以证明该参数能按运行时 placement 合法切分。

### 3.3 持久内存、规约梯度和 gather 缓冲

设：

- $b_p$：设备上持久参数副本的每元素字节数；
- $b_o$：优化器状态的每元素字节数；
- $b_g$：规约梯度的每元素字节数；
- $b_c$：计算参数 dtype 的每元素字节数。

则单卡持久内存为：

$$
M_{\mathrm{persistent}}
=
N_{\mathrm{rank}} \cdot (b_p + b_o)
$$

规约完成、驻留到 optimizer step 的梯度分片为：

$$
M_{\mathrm{grad\_shard}}
=
N_{\mathrm{rank}} \cdot b_g
$$

FSDP all-gather 会恢复到 TP/EP placement 后、FSDP 前的参数，因此该参数完整 gather 物化量为：

$$
M_{\mathrm{gather\_full}}
=
P_{\mathrm{layout}} \cdot b_c
$$

也就是说，FSDP/HSDP 会缩小持久参数、优化器状态和规约梯度，但不会按 $K$ 缩小使用期间的完整参数 gather buffer。

Evaluator 的对应实现位于：

- 持久态分母选择：[`cost_eval/static_mem.py:L69-L88`](../cost_eval/static_mem.py#L69-L88)
- 持久态、完整参数和梯度分片：[`cost_eval/structure_mem.py:L222-L252`](../cost_eval/structure_mem.py#L222-L252)
- 梯度分片累计生命周期：[`cost_eval/mem_timeline.py:L636-L645`](../cost_eval/mem_timeline.py#L636-L645)

---

## 4. 各类参数的正确公式与当前实现

下表中的 $P$ 均表示对应逻辑参数在 TP/EP 前的元素数。

| 参数类型 | 当前 MindFormers placement | 正确单卡元素数 | evaluator 当前元素数 | 判断 |
|---|---|---:|---:|---|
| TP-sharded dense 参数 | TP shard，随后 dense FSDP | $\dfrac{P}{T K}$ | $\dfrac{P}{T K}$ | 可整除时正确 |
| TP-replicated dense 参数，如 norm/router | TP replicate，随后 dense FSDP | $\dfrac{P}{K}$ | $\dfrac{P}{K}$ | 一般正确 |
| shared-expert FC 权重 | TP replicate，随后 dense FSDP | $\dfrac{P}{K}$ | $\dfrac{P}{T K}$ | 低估 $T$ 倍 |
| routed expert，$E>1$ | EP shard，随后 eFSDP | $\dfrac{P}{E F_e}=\dfrac{P}{SCT}$ | 相同 | 可整除时正确 |
| routed expert，$E=1$ | TP replicate，随父层走 dense FSDP | $\dfrac{P}{K}$ | $\dfrac{P}{SCT}$ | 默认低估 $T$ 倍 |
| dense `replicate_params` | TP/EP 后本地参数整体复制 | $P_{\mathrm{layout}}$ | $\left\lceil\dfrac{P_{\mathrm{layout}}}{K}\right\rceil$ | 不正确 |

下面分别解释这些结论。

### 4.1 普通 dense 参数

对于典型 TP column/row parallel 权重，TP 已先将参数切成 $1/T$，dense FSDP 再在 $K$ 个 rank 上切分：

$$
N_{\mathrm{dense,TP\text{-}sharded}}
=
\frac{P}{T \cdot K}
$$

对于 norm、router 等 TP-replicated 参数：

$$
N_{\mathrm{dense,TP\text{-}replicated}}
=
\frac{P}{K}
$$

Evaluator 使用图内 `TensorRef.shard` 先处理 TP，再在 `StaticMem` 中统一除 dense FSDP degree，因此这条主路径是正确的。

### 4.2 Routed expert：`ep > 1`

Evaluator 将 routed-expert 权重的 expert 维标成 EP shard：[`cost_eval/layers/ffn.py:L147-L149`](../cost_eval/layers/ffn.py#L147-L149)、[`cost_eval/layers/ffn.py:L183-L192`](../cost_eval/layers/ffn.py#L183-L192)。运行时同样对 `weight1/weight2` 使用 `Shard(0)`：[`mindformers/pynative/distributed/expert_parallel.py:L482-L488`](../../mindformers/mindformers/pynative/distributed/expert_parallel.py#L482-L488)。

先经 EP：

$$
P_{\mathrm{layout}} = \frac{P}{E}
$$

再经 eFSDP：

$$
N_{\mathrm{routed},E>1}
=
\frac{P/E}{F_e}
=
\frac{P}{E \cdot (SCT/E)}
=
\frac{P}{SCT}
$$

所以在合法、可整除的 EP 路径中，routed-expert 持久参数对 $E$ 本身不敏感。增大 EP 会减少每 rank 持有的 expert 数，但 eFSDP degree 同时按 $1/E$ 减小，两者在持久参数乘积分母里抵消。

这不意味着 EP 对激活和通信也没有影响；这里只讨论参数、优化器状态和规约梯度的驻留元素数。

### 4.3 Routed expert：`ep = 1`

当前运行时只有 `ep > 1` 才进入 EP phase：[`mindformers/pynative/base_models/gpt/parallelize.py:L1496-L1520`](../../mindformers/mindformers/pynative/base_models/gpt/parallelize.py#L1496-L1520)。`ep=1` 时，routed-expert 参数在 TP 上显式设为 `Replicate()`：[`mindformers/pynative/base_models/gpt/parallelize.py:L700-L716`](../../mindformers/mindformers/pynative/base_models/gpt/parallelize.py#L700-L716)。同时，只有存在 expert mesh 时才单独 fully-shard experts：[`mindformers/pynative/base_models/gpt/parallelize.py:L1030-L1037`](../../mindformers/mindformers/pynative/base_models/gpt/parallelize.py#L1030-L1037)、[`mindformers/pynative/base_models/gpt/parallelize.py:L1106-L1113`](../../mindformers/mindformers/pynative/base_models/gpt/parallelize.py#L1106-L1113)。

因此运行时公式是：

$$
N_{\mathrm{routed},E=1}^{\mathrm{runtime}}
=
\frac{P}{K}
$$

但 evaluator 只根据 `TensorRef` 是否声明过 EP shard 来设置 `is_expert`，即使 $E=1$ 也继续使用 $F_e=SCT$：[`cost_eval/model_spec.py:L93-L112`](../cost_eval/model_spec.py#L93-L112)、[`cost_eval/shape_eval.py:L97-L99`](../cost_eval/shape_eval.py#L97-L99)。因此：

$$
N_{\mathrm{routed},E=1}^{\mathrm{evaluator}}
=
\frac{P}{SCT}
$$

实际值与估算值之比为：

$$
\frac{N^{\mathrm{runtime}}}{N^{\mathrm{evaluator}}}
=
\frac{SCT}{K}
$$

默认 $K=SC$ 时：

$$
\frac{N^{\mathrm{runtime}}}{N^{\mathrm{evaluator}}} = T
$$

因此 `ep=1, tp>1` 会把 routed-expert 参数、优化器状态和规约梯度低估约 $T$ 倍。若同时配置 $K<SC$ 的 dense FSDP 子域，偏差会进一步放大。

Evaluator 的时间线还用 `efsdp_degree > 1` 判断是否构造独立 expert gather 段：[`cost_eval/mem_timeline.py:L469-L503`](../cost_eval/mem_timeline.py#L469-L503)。因此该问题不仅影响持久态，也影响 `ep=1` 时的 gather 生命周期和峰值位置。

### 4.4 Shared expert

Evaluator 当前把 shared-expert FC1/FC2 权重建成 TP shard：

- FC1：末维除以 TP；
- FC2：首维除以 TP。

源码：[`cost_eval/layers/ffn.py:L242-L268`](../cost_eval/layers/ffn.py#L242-L268)。

但当前 MindFormers 明确说明 shared-expert 参数在 TP 上 replicated，只让 token 激活保留 Sequence Parallel shard：[`mindformers/pynative/base_models/gpt/parallelize.py:L718-L724`](../../mindformers/mindformers/pynative/base_models/gpt/parallelize.py#L718-L724)。这些参数随后仍属于 dense FSDP wrap：[`mindformers/pynative/base_models/gpt/parallelize.py:L1063-L1068`](../../mindformers/mindformers/pynative/base_models/gpt/parallelize.py#L1063-L1068)。

所以正确公式为：

$$
N_{\mathrm{shared}}^{\mathrm{runtime}}
=
\frac{P}{K}
$$

当前 evaluator 为：

$$
N_{\mathrm{shared}}^{\mathrm{evaluator}}
=
\frac{P}{T K}
$$

两者之比：

$$
\frac{N_{\mathrm{shared}}^{\mathrm{runtime}}}
     {N_{\mathrm{shared}}^{\mathrm{evaluator}}}
=
T
$$

这是 OOM 不安全的低估，并且同时影响：

- 参数副本；
- 优化器状态；
- 规约梯度；
- FSDP gather buffer，因为 evaluator 的 `param_full_bytes` 已在错误的 TP placement 后少算了 $T$ 倍。

### 4.5 HSDP 与 dense grouped-FSDP

HSDP 的二维 mesh 由 replicate 轴和 shard 轴组成。对默认完整 dense FSDP：

$$
\mathrm{mesh}_{\mathrm{HSDP}}
=
[R, F]
$$

如果 dense 只在 $K$ 个 rank 的子域上 shard，完整 FSDP 域中剩余的 $F/K$ 也会折入 replicate 轴：

$$
\mathrm{mesh}_{\mathrm{dense\ subgroup}}
=
\left[
R \cdot \frac{F}{K},
K
\right]
$$

运行时正是按下面的公式构造 replicate degree：

$$
R_{\mathrm{effective}}
=
R \cdot \frac{F}{K}
$$

源码：[`mindformers/pynative/distributed/parallel_dims.py:L421-L497`](../../mindformers/mindformers/pynative/distributed/parallel_dims.py#L421-L497)。

因此单卡参数元素数只除 shard 轴：

$$
N_{\mathrm{rank}} = \frac{P_{\mathrm{layout}}}{K}
$$

不能除 replicate 轴：

$$
N_{\mathrm{rank}} \ne
\frac{P_{\mathrm{layout}}}
     {R_{\mathrm{effective}} \cdot K}
$$

从整个 HSDP mesh 看，参数总驻留元素数为：

$$
R_{\mathrm{effective}} \cdot K \cdot
\frac{P_{\mathrm{layout}}}{K}
=
R \cdot \frac{F}{K} \cdot P_{\mathrm{layout}}
$$

这说明减小 $K$ 会增加复制份数和单卡持久态，但梯度规约总域仍保持不变：先在 $K$ 个 shard rank 上 reduce-scatter，再在 $R\cdot F/K$ 个 replica rank 上 all-reduce。运行时对此有直接说明：[`mindformers/pynative/distributed/parallel_dims.py:L424-L433`](../../mindformers/mindformers/pynative/distributed/parallel_dims.py#L424-L433)。

Evaluator 的 `dense_fsdp_degree()` 正确返回 $K$ 或默认 $F$，且 `dp_replicate` 没有进入持久态分母：[`cost_eval/parallel_model.py:L51-L66`](../cost_eval/parallel_model.py#L51-L66)、[`cost_eval/static_mem.py:L69-L88`](../cost_eval/static_mem.py#L69-L88)。因此 HSDP/dense 子域的主参数公式是当前实现中较准确的一部分。

---

## 5. 当前 evaluator 的实际调用链

当前参数内存计算链路是：

```text
TensorRef.shape + TensorRef.shard
        │
        ▼
ShapeEval.resolve_tensor
  └─ 先按 TP / EP 等图内轴缩 local shape
        │
        ▼
ResolvedTensor.local_numel + is_expert
        │
        ▼
ParallelModel
  ├─ dense fsdp degree = K or S·C
  └─ efsdp degree       = S·C·T/E
        │
        ▼
StaticMem.compute
  └─ 将 dense / expert 两个 divisor 传给 structure_mem
        │
        ▼
estimate_structure_memory
  ├─ persistent
  ├─ param_full_bytes
  ├─ grad_full_bytes
  └─ grad_shard_bytes
        │
        ▼
MemTimeline
  ├─ gather_buf
  ├─ grad_buf
  ├─ grad_accum
  └─ optstep
```

关键源码位置：

1. `TensorRef.shard` 与 `has_ep()`：[`cost_eval/model_spec.py:L93-L112`](../cost_eval/model_spec.py#L93-L112)
2. TP/EP shape 求值：[`cost_eval/shape_eval.py:L68-L99`](../cost_eval/shape_eval.py#L68-L99)
3. $F$、$K$、$F_e$：[`cost_eval/parallel_model.py:L16-L20`](../cost_eval/parallel_model.py#L16-L20)、[`cost_eval/parallel_model.py:L51-L66`](../cost_eval/parallel_model.py#L51-L66)
4. StaticMem 选择分母：[`cost_eval/static_mem.py:L69-L88`](../cost_eval/static_mem.py#L69-L88)
5. 持久态与梯度切分：[`cost_eval/structure_mem.py:L222-L252`](../cost_eval/structure_mem.py#L222-L252)
6. gather 与梯度生命周期：[`cost_eval/mem_timeline.py:L469-L503`](../cost_eval/mem_timeline.py#L469-L503)、[`cost_eval/mem_timeline.py:L636-L645`](../cost_eval/mem_timeline.py#L636-L645)

这个设计的根本限制是 `ResolvedTensor.is_expert: bool` 只能表达两种 FSDP 分母：

```text
is_expert = False  → dense FSDP divisor
is_expert = True   → eFSDP divisor
```

而当前运行时至少需要表达以下状态：

1. TP-sharded + dense FSDP；
2. TP-replicated + dense FSDP；
3. EP-sharded + expert FSDP；
4. TP-replicated routed expert + dense FSDP（`ep=1`）；
5. FSDP `replicate_params`；
6. 可选自定义 FSDP shard 维。

因此问题不是简单替换一个分母即可完全解决，而是当前参数 placement 状态空间不足。

---

## 6. 不可整除参数：当前 `ceil` 公式为什么不成立

当前 evaluator 对持久态和规约梯度使用：

$$
N_{\mathrm{rank}}^{\mathrm{evaluator}}
=
\left\lceil
\frac{P_{\mathrm{layout}}}{G}
\right\rceil
$$

其中 $G$ 为 $K$ 或 $F_e$。源码及注释位于 [`cost_eval/structure_mem.py:L222-L252`](../cost_eval/structure_mem.py#L222-L252)，其假设是 FSDP 会把 flat parameter pad 到 group size 的整数倍。

但当前 MindFormers dense 路径按 TP/DTensor 后的参数首维检查：

$$
\mathrm{shape}_{\mathrm{layout}}[0] \bmod K \ne 0
\quad\Longrightarrow\quad
\text{put the whole parameter into replicate\_params}
$$

源码：[`mindformers/pynative/base_models/gpt/parallelize.py:L331-L350`](../../mindformers/mindformers/pynative/base_models/gpt/parallelize.py#L331-L350)。此外，若参数属于特殊小参数，运行时还会显式加入 `replicate_params`：[`mindformers/pynative/base_models/gpt/parallelize.py:L353-L378`](../../mindformers/mindformers/pynative/base_models/gpt/parallelize.py#L353-L378)。

对这类参数，正确公式是：

$$
N_{\mathrm{rank}}^{\mathrm{runtime}}
=
P_{\mathrm{layout}}
$$

而不是：

$$
\left\lceil
\frac{P_{\mathrm{layout}}}{K}
\right\rceil
$$

例如：

$$
P_{\mathrm{layout}}=64,
\qquad
K=256
$$

Evaluator 得到：

$$
\left\lceil \frac{64}{256} \right\rceil = 1
$$

而 dense runtime 路径会复制整个参数：

$$
N_{\mathrm{rank}}^{\mathrm{runtime}}=64
$$

所以“使用 ceil 因而 OOM 安全”并不成立；ceil 只比 floor 保守，却可能远小于运行时复制整个参数的真实驻留。

另外，optimizer-step 瞬态仍使用 floor：

$$
N_{\mathrm{optstep}}
=
\left\lfloor
\frac{P_{\mathrm{layout}}}{G}
\right\rfloor
$$

源码：[`cost_eval/mem_timeline.py:L655-L662`](../cost_eval/mem_timeline.py#L655-L662)。这与持久态/梯度的 ceil 口径本身也不一致。

对 routed-expert 独立 wrap，MindFormers 没有传 `replicate_params`：[`mindformers/pynative/base_models/gpt/parallelize.py:L1109-L1113`](../../mindformers/mindformers/pynative/base_models/gpt/parallelize.py#L1109-L1113)。因此 evaluator 也不应在没有底层源码或真机证据时默认套用 flat-numel ceil；这里至少应做运行时等价的 shard-dimension 校验并 fail loud，或显式建模底层实际策略。

---

## 7. HSDP 配置适配器的 `-1` 语义问题

MindFormers 中 `data_parallel_shard=-1` 表示 pure FSDP：

$$
R=1,
\qquad
S=D
$$

其中 $D$ 是 Trainer 根据 world size 推导出的总 data-parallel degree。源码：

- 默认配置：[`mindformers/pynative/config/config.py:L342-L351`](../../mindformers/mindformers/pynative/config/config.py#L342-L351)
- Trainer 归一化：[`mindformers/pynative/trainer/trainer.py:L430-L454`](../../mindformers/mindformers/pynative/trainer/trainer.py#L430-L454)
- `ParallelDims.from_config`：[`mindformers/pynative/distributed/parallel_dims.py:L65-L96`](../../mindformers/mindformers/pynative/distributed/parallel_dims.py#L65-L96)

但 evaluator adapter 在同时得到 `data_parallel` 且 `data_parallel_shard<=0` 时，会映射为：

$$
S=1,
\qquad
R=D
$$

源码：[`cost_eval/configs/from_mindformers.py:L795-L812`](../cost_eval/configs/from_mindformers.py#L795-L812)。

这会把 raw pre-Trainer 配置中的 pure FSDP 解释成纯复制 DP。若输入已经经过 Trainer 归一化、`data_parallel_shard` 已被回写成正数，则不会触发该分支；因此应在适配器接口上明确区分：

1. raw YAML / pre-Trainer config；
2. Trainer-normalized runtime config。

---

## 8. 数值示例

假设：

$$
R=2,
\quad
S=8,
\quad
C=1,
\quad
T=4,
\quad
E=8,
\quad
K=8
$$

则：

$$
F=S\cdot C=8
$$

$$
F_e=\frac{SCT}{E}
=
\frac{8\cdot1\cdot4}{8}
=4
$$

对一个含 $P$ 个元素的逻辑参数：

### 8.1 TP-sharded dense

$$
N_{\mathrm{rank}}
=
\frac{P}{T K}
=
\frac{P}{4\cdot8}
=
\frac{P}{32}
$$

### 8.2 Router、norm、shared expert

它们在 TP 上 replicated，因此：

$$
N_{\mathrm{rank}}
=
\frac{P}{K}
=
\frac{P}{8}
$$

当前 evaluator 对 shared-expert FC 权重会给出：

$$
\frac{P}{T K}
=
\frac{P}{32}
$$

即低估 4 倍。

### 8.3 Routed expert，`ep=8`

$$
N_{\mathrm{rank}}
=
\frac{P}{E F_e}
=
\frac{P}{8\cdot4}
=
\frac{P}{32}
$$

### 8.4 保持其他配置不变但令 `ep=1`

运行时 routed expert 在 TP 上 replicated，并随父层走 dense FSDP：

$$
N_{\mathrm{rank}}^{\mathrm{runtime}}
=
\frac{P}{8}
$$

Evaluator 仍使用：

$$
F_e = SCT = 32
$$

$$
N_{\mathrm{rank}}^{\mathrm{evaluator}}
=
\frac{P}{32}
$$

因此同样低估 4 倍。

### 8.5 将 dense 子域改为 `K=2`

HSDP mesh 变为：

$$
\left[
R\cdot\frac{F}{K},K
\right]
=
\left[
2\cdot\frac{8}{2},2
\right]
=
[8,2]
$$

此时：

$$
N_{\mathrm{dense,TP\text{-}sharded}}
=
\frac{P}{T K}
=
\frac{P}{8}
$$

$$
N_{\mathrm{dense,TP\text{-}replicated}}
=
\frac{P}{K}
=
\frac{P}{2}
$$

若 `ep>1`，routed expert 仍走独立 eFSDP，因此不受 dense 子域 $K$ 影响，仍为 $P/32$。

---

## 9. 时间模型的额外偏差

内存模型至少分别持有 dense FSDP 和 eFSDP degree；当前 timesim 只有单一 `Degrees.dp`：[`cost_eval/timesim/shard_rules.py:L19-L26`](../cost_eval/timesim/shard_rules.py#L19-L26)。

前向 FSDP all-gather payload 当前按：

$$
V_{\mathrm{AG}}^{\mathrm{current}}
=
\left\lceil
\frac{P_{\mathrm{segment}}}{S}
\right\rceil
$$

源码：[`cost_eval/timesim/frame_comm.py:L45-L64`](../cost_eval/timesim/frame_comm.py#L45-L64)。报告层对所有 segment 都注入同一个 `deg.dp`：[`cost_eval/timesim/report.py:L124-L144`](../cost_eval/timesim/report.py#L124-L144)。这无法表达：

- dense FSDP 的 $F=S\cdot C$；
- dense 子域 $K$；
- routed-expert 的 $F_e$；
- `replicate_params` 无需 FSDP gather；
- `ep=1` routed expert 随父层走 dense FSDP。

HSDP replica 轴的梯度 all-reduce 当前使用完整 stage 参数量：[`cost_eval/timesim/report.py:L175-L185`](../cost_eval/timesim/report.py#L175-L185)。但 HSDP 正常顺序是：

1. 在 shard 轴 $K$ 上 reduce-scatter；
2. 对本地梯度 shard 在 replica 轴 $R\cdot F/K$ 上 all-reduce。

因此普通 sharded 参数的 replica all-reduce payload 应基于：

$$
V_{\mathrm{replica\ AR}}
\propto
\frac{P_{\mathrm{layout}}}{K} \cdot b_g
$$

而不是完整 $P_{\mathrm{layout}}\cdot b_g$。`replicate_params` 则需要单独使用完整参数梯度的 DDP-style all-reduce。

结论是：当前 timesim 的 HSDP/MoE 通信估算不能与内存模型的分片精度等同看待。

---

## 10. 测试与真机证据边界

本次复跑相关测试：

```powershell
python -m pytest -q \
  tests/test_parallel_model.py \
  tests/test_static_mem.py \
  tests/test_z3_grouped_fsdp.py \
  tests/test_from_mindformers.py \
  tests/test_memval_hybrid_recompute.py \
  tests/test_timesim_frame_comm.py \
  tests/test_timesim_report.py
```

结果：

```text
113 passed, 44 warnings
```

这证明相关实现内部测试通过，但不能证明其与当前 MindFormers runtime 完全等价，原因包括：

1. grouped-FSDP 测试在 `ep=1` 时显式要求 routed experts 仍走 eFSDP、且不受 dense 子域影响：[`tests/test_z3_grouped_fsdp.py:L127-L165`](../tests/test_z3_grouped_fsdp.py#L127-L165)。这锁定的是 evaluator 自身假设，不是当前 runtime 的 `ep=1` 退化路径。
2. 混合并行参数守恒测试的 `_MOE_CFG` 没有 shared expert：[`tests/test_memval_hybrid_recompute.py:L234-L239`](../tests/test_memval_hybrid_recompute.py#L234-L239)，因此无法发现 shared expert 的 TP placement 错误。
3. 手工闭式公式直接使用“expert 除以 EP 再除 eFSDP”：[`tests/test_memval_hybrid_recompute.py:L241-L268`](../tests/test_memval_hybrid_recompute.py#L241-L268)，仍属于同模型内部交叉验证。
4. 当前分析文档承认真机锚只覆盖 `dp_shard=2` 的 `ep=1/2` 等少量点，而 TP/HSDP sweep 仍是 evaluator-only 外推：[`analysis/multidim_memory_analysis.md:L155-L168`](multidim_memory_analysis.md#L155-L168)。
5. 已有 `ep=1` 真机锚的 `tp=1`，此时默认 $K=SC$，runtime 与 evaluator 恰好都得到 $P/(SC)$，因此不会暴露 `ep=1,tp>1` 问题。

---

## 11. 建议的修正方向

### P0：从 `is_expert` 升级为显式参数 placement

建议每个 resolved parameter 至少携带：

```text
tp_placement: shard(dim) | replicate
ep_placement: shard(dim) | replicate
fsdp_group: dense | expert | replicate
fsdp_shard_dim: int | None
fsdp_shard_degree: int
```

统一公式变为：

$$
N_{\mathrm{rank}}(p)
=
\begin{cases}
P_{\mathrm{layout}}(p),
& \text{if } p \text{ is FSDP-replicated}, \\
\dfrac{P_{\mathrm{layout}}(p)}{G(p)},
& \text{if the selected shard dimension is divisible}.
\end{cases}
$$

### P0：按 runtime 条件选择 routed-expert 路径

$$
\mathrm{fsdp\_group}(p_{\mathrm{routed}})
=
\begin{cases}
\mathrm{expert}, & E>1, \\
\mathrm{dense}, & E=1.
\end{cases}
$$

并在 $E=1$ 时将 routed-expert TP placement 设为 replicated。

### P0：修正 shared expert TP placement

shared-expert FC 参数应在 TP 上 replicated，激活仍可保持 Sequence Parallel。这样持久态与 gather buffer 都会从 $P/(TK)$ 修正为 $P/K$。

### P0：移除无来源的 total-numel ceil

应先对 TP/EP 后的真实 local shape 和 FSDP shard 维做校验：

$$
\mathrm{shape}_{\mathrm{layout}}[d] \bmod G = 0
$$

然后：

- dense runtime 会复制的参数：标为 `replicate_params`；
- expert runtime 会拒绝的配置：fail loud；
- 若未来底层真正支持 padding：必须按该版本底层的真实 padding 粒度建模，而不是假设对每个参数 total-numel 做 ceil。

### P1：修正 raw MindFormers 配置适配

如果 `data_parallel_shard<0` 且已知总 DP degree $D$，应按 runtime 语义使用：

$$
R=1,
\qquad
S=D
$$

如果不知道 world size，则应明确标记为推断值，不能把 `data_parallel` 自动解释成复制轴。

### P1：让 timesim 使用 per-parameter/per-segment FSDP group

至少区分：

1. dense AG/RS：degree $K$；
2. routed-expert AG/RS：degree $F_e$；
3. HSDP replica AR：payload 为 RS 后 shard；
4. `replicate_params`：无 FSDP AG/RS，梯度走完整 DDP all-reduce。

### 建议新增的源码等价测试

| 测试 | 应验证的公式 |
|---|---|
| `ep=1,tp=4,K=F` routed expert | runtime/evaluator 都应为 $P/K$，不是 $P/(FT)$ |
| `ep=1,tp=4,K<F` routed expert | 应随 dense 子域 $K$ 变化 |
| `ep>1,tp>1` routed expert | 应为 $P/(SCT)$，且不受 dense $K$ 影响 |
| shared expert + `tp=4` | 应为 $P/K$，不是 $P/(4K)$ |
| dense 参数首维不可整除 | 应整参复制，不是 total-numel ceil |
| `R>1` HSDP | 单卡驻留不除 $R$；replica AR payload 除 $K$ |
| raw `data_parallel_shard=-1` | 应映射为 pure FSDP：$R=1,S=D$ |

---

## 12. 最终判断

当前 evaluator 的核心公式不是全错，而是参数分类过粗：

$$
\boxed{
\text{dense/HSDP main path is mostly correct}
}
$$

$$
\boxed{
\text{MoE correctness currently requires } E>1
\text{ and no TP-replicated shared-expert mismatch}
}
$$

$$
\boxed{
\text{replicate\_params and non-divisible sharding are not modeled correctly}
}
$$

对生产 OOM 评估而言，优先级最高的是 shared expert、`ep=1,tp>1` 和 `replicate_params` 三项，因为它们都会让当前模型低估真实的参数、优化器状态、梯度或 gather buffer。

---

## 相关资料

- 当前 evaluator 多维显存分析：[`analysis/multidim_memory_analysis.md`](multidim_memory_analysis.md)
- grouped-FSDP 测试：[`tests/test_z3_grouped_fsdp.py`](../tests/test_z3_grouped_fsdp.py)
- evaluator 参数切分入口：[`cost_eval/shape_eval.py`](../cost_eval/shape_eval.py)、[`cost_eval/static_mem.py`](../cost_eval/static_mem.py)、[`cost_eval/structure_mem.py`](../cost_eval/structure_mem.py)
- MindFormers 并行 mesh：[`mindformers/pynative/distributed/parallel_dims.py`](../../mindformers/mindformers/pynative/distributed/parallel_dims.py)
- MindFormers GPT 并行化：[`mindformers/pynative/base_models/gpt/parallelize.py`](../../mindformers/mindformers/pynative/base_models/gpt/parallelize.py)
- HyperParallel 官方说明：[README](https://gitee.com/mindspore/hyper-parallel/blob/master/README.md)
- HyperParallel `replicate_params` 状态实现：[MindSpore HSDP state coverage source](https://repo-internal.mindspore.cn/mindspore/hyper-parallel/version/202605/20260528/master_20260528020006_339ba7a67be85c79140cac8c108fad809e8fecf9_newest/coverage/htmlcov/z_193f6319e181a474_state_py.html)

---

## 修复答复（2026-07-23）

四项 P0 已全部按 runtime 语义修复（runtime 基线不变:mindformers `377c9c344`,本地
`E:\97-codes\torch_parallel\mindformers`;evaluator 分支 `feat/unified-llm-modelspec`）:

| 项 | 修改点 | 依据行号 |
|---|---|---|
| P0-1 shared expert TP 复制 | `cost_eval/layers/ffn.py::build_shared_expert_ops`:sh_w1/sh_w2 去 tp shard（复制,持久/gather P/(TK)→P/K）;激活 sh_g/sh_act 改按序列 {0:"sp"} 切（SequenceParallel(sequence_dim=0),numel 与旧末维 ÷tp 等价）;输出 sh_o/sh_o_gated 去 partial（权重复制无部分和;为避 mHC 残差承载签名冲突取全量口径,保守） | parallelize.py:718-724, :1063-1068 |
| P0-2 routed expert ep=1 退化 | `cost_eval/parallel_model.py::efsdp_degree`:ep<=1 → 返回 dense_fsdp_degree()=K（expert 随父层 dense wrap）;`cost_eval/mem_timeline.py` `_split_experts` 加 `ep>1` 门（ep=1 不再建独立 expert gather 段）;ep>1 逐字节不变 | parallelize.py:700-716, :1030-1037, :1106-1113, :1496-1520 |
| P0-3 不可整除 → replicate_params | `cost_eval/structure_mem.py::_fsdp_local_count`（新）:dense 按 TP/EP 后**首维** `shape[0] % divisor` 判定,不可整除 → **整参驻留**（persistent/grad_shard/optstep 三口径统一,total-numel ceil 与 optstep floor 全部废除）;expert 独立 wrap（ep>1）不可整除 → fail-loud;`cost_eval/shape_eval.py::ResolvedTensor` 新增 `dim0`（TP/EP 后本地首维）直通 | parallelize.py:331-350, :353-378, :1109-1113 |
| P0-4 adapter `-1` 语义 | `cost_eval/configs/from_mindformers.py`:`data_parallel` 给定且 `data_parallel_shard<=0` → **纯 FSDP**（dp_shard=D, dp_replicate=1）+ warning 注明按 trainer 归一化语义;`shard>0` 分支不动 | trainer.py:449-454, parallel_dims.py:65-96 |

**测试**:§11 矩阵 7 行落成 `tests/test_param_placement_p0.py`（7 用例全过）;§10.1 指出的
`tests/test_z3_grouped_fsdp.py` 错误锁定已重写（`test_ep1_experts_follow_dense_subdomain` +
保留 ep>1 半边 `test_ep2_experts_unaffected_by_dense_subdomain`）;
`tests/test_static_mem.py::test_indivisible_fsdp_ceil_oom_safe` 重写为
`test_indivisible_fsdp_replicates_whole_param`;`tests/test_x4_experts_wrap.py` 拆分用例改
ep=2 并新增 `test_ep1_no_experts_split`;两个参数守恒测试
（test_memval_structure_matrix/test_memval_hybrid_recompute）按 ep=1 专家 TP 复制语义更新期望式。

**回归**:全量 1430 passed;全部真机锚点（scorecard 13+1、pp4 ON/OFF/MTP/pp8、116 std 六锚、
DSv4-Flash roundtrip pin）**逐字节零移动**——锚点均 tp=1（P0-1/P0-2 恒等）且权重整除
（P0-3 整除路径 == 旧 ceil）,与 §10.5 的预判一致。

**已知未修范围**:§9 timesim（HSDP/MoE 通信 payload 的 per-segment FSDP group:dense K /
expert Fe / replica-AR ÷K / replicate_params 全量 DDP-AR）本轮**不修**（用户裁定先修静态
参数口径）,列为后续任务;§11 P1 的 per-parameter placement 状态机(`fsdp_group/shard_dim`
显式化)以 `dim0`+`ep_degree` 的最小实现落地,完整状态机留待 timesim 一并重构。
