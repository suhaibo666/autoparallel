# P0 详细设计：ModelSpec 定义 + 静态/激活内存计算

> 日期: 2026-06-29 · 状态: Draft（待评审）· 父设计: [[2026-06-23-pynative-cost-evaluator-design]]
> P0 目标：dense + MoE 模型在**全 5D 并行 + recompute + swap** 下的**每卡峰值显存 + OOM 判定 + 构成拆解**（时间模型属 P1）。

## 1. P0 范围与产出

**做**：ModelSpec 数据结构 + dense/moe op 图 + 符号 shape 求值 + sharding 代入 + **静态内存（param/grad/opt）** + **激活内存** + PP 时间线峰值 + recompute/swap 对显存的影响 + OOM 判定与拆解。

**不做（→P1）**：单步时间、roofline 时间、集合通信/all-to-all **时间**、overlap/bubble **时间**、swap 传输**时间**。注意：P0 仍需 5D（含 EP）的**切分语义**来算显存，只是不算其时间。

## 2. ModelSpec 数据结构

```python
# ---------- 维度表：架构超参（具体值） ----------
@dataclass
class DimTable:
    H: int            # hidden_size
    F: int            # dense ffn intermediate
    n_heads: int
    n_kv: int         # GQA kv heads（MHA 时 = n_heads）
    head_dim: int
    S: int            # seq_len（单 microbatch）
    B: int            # micro_batch（单 PP microbatch、单 DP rank）
    vocab: int
    n_layers: int
    # MoE（n_experts=0 即 dense 模型）
    n_experts: int = 0
    topk: int = 0
    n_shared: int = 0         # 共享专家数
    moe_F: int = 0            # 专家 ffn intermediate（缺省同 F）
    capacity_factor: float = 1.0   # balanced 路由 = 1.0
    # MLA（可选；0 表示走标准 GQA）
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_dim: int = 0
    qk_nope_dim: int = 0
    v_head_dim: int = 0

# ---------- 张量引用：符号 shape + 切分标注 ----------
@dataclass
class TensorRef:
    name: str
    shape: list[str]          # 符号维表达式，如 ["S","B","H"]、["H","(n_heads+2*n_kv)*head_dim"]
    shard: dict[int, str]     # dim_index -> 并行轴；轴 ∈ {"tp","cp","ep","sp"}
    is_weight: bool = False   # True=持久权重(state)；False=激活
    partial: str | None = None  # 行并行输出在某轴上是部分和，如 "tp"（→ reshard 触发 all-reduce）

# ---------- 算子 ----------
class OpType(Enum):
    MATMUL = auto(); FLASH_ATTN = auto(); ELEMENTWISE = auto(); NORM = auto()
    SOFTMAX = auto(); ROPE = auto(); MOE_ROUTER = auto(); MOE_GEMM = auto()
    DISPATCH = auto(); COMBINE = auto(); COLLECTIVE = auto()

@dataclass
class OpSpec:
    name: str
    type: OpType
    inputs: list[TensorRef]            # matmul 的 weight 也是 input（is_weight=True）
    output: TensorRef
    # ---- 内存契约（每 op 自声明；融合/非融合自动跟随）----
    params: list[TensorRef] = field(default_factory=list)  # 拥有的权重 → param/grad/opt（融合不变）
    saves:  list[TensorRef] = field(default_factory=list)  # 为自己 backward 保留的张量(=save_for_backward；多为输入/少量中间) → 激活（融合敏感）
    workspace: str | None = None                           # 执行期 kernel scratch 的符号字节表达式（瞬时；融合相关）
    attrs: dict = field(default_factory=dict)              # {"act":"swiglu","causal":True,...}

@dataclass
class LayerSpec:
    ops: list[OpSpec]

@dataclass
class ModelSpec:
    name: str
    dims: DimTable
    layer_pattern: list[str]              # 每层类型，如 ["dense"]*3 + ["moe"]*58 + ["mtp"]
    layer_specs: dict[str, LayerSpec]     # layer_type -> op 图
    dtype_bytes: int = 2                  # 计算 dtype（bf16=2）
    tie_embeddings: bool = False
```

### 2.1 内存契约：三类常驻 + 标准聚合 + 融合不变性

每个 op 自声明 `params`/`saves`/`workspace`，三类常驻内存按一套**标准规则**聚合，融合/非融合只是子图替换、估计自动跟随：

| 常驻类 | 来源 | 融合敏感性 | 聚合规则 |
|---|---|---|---|
| 参数+优化器 | op.`params`(is_weight) | **不变** | Σ_op Σparams × 优化器倍数（§6） |
| 激活 | op.`saves` | **敏感** | Σ_**去重**(所有 op 的 saves)；同张量被多 op 需要只算一次 |
| workspace | op.`workspace` | kernel 相关 | 瞬时，进时间线仿真（§8） |

- `saves` = autodiff 的 `save_for_backward`：`Y=X·W` 反向需 X（wgrad）→ **saves X（输入）**，不是 output；W 在 `params`。
- 融合范例：非融合 MLP 三 op saves `X + [S,B,2F] + [S,B,F]`；融合 `fused_swiglu_mlp` 内部重算 → **只 saves X**，中间自动消失；两者 `params` 完全相同。FlashAttn 只 saves Q,K,V,O,LSE（不物化 S×S 分数矩阵）。

## 3. dense_decoder op 图（pre-norm + GQA + SwiGLU，对照 `transformer_layer.py`/`attention.py`/`mlp.py`）

| # | op | inputs(shard) | output(shard, saved) | 备注 |
|---|---|---|---|---|
| 1 | NORM input_ln | x`[S,B,H]{sp:0}` | `[S,B,H]{sp:0}` saved | RMSNorm |
| 2 | MATMUL qkv | ln`[S,B,H]`, w`[H,(n_heads+2n_kv)·hd]{1:tp}`(weight) | `[S,B,(n_heads+2n_kv)·hd]{2:tp}` saved | 融合 QKV，列并行 |
| 3 | ROPE | q,k 子张量 | 同形 | 小 |
| 4 | FLASH_ATTN | q,k,v`{2:tp}` | `[S,B,n_heads·hd]{2:tp}` saved + lse`[S,B,n_heads]{2:tp}` saved | causal；workspace |
| 5 | MATMUL o_proj | attn`[S,B,n_heads·hd]{2:tp}`, w`[n_heads·hd,H]{0:tp}` | `[S,B,H]` partial=tp | 行并行→部分和 |
| 6 | (reshard) | `[S,B,H]` partial(tp)→{sp:0} | `[S,B,H]{sp:0}` | **派生 all-reduce(无SP)/reduce-scatter(SP)** |
| 7 | ELEMENTWISE add | residual | `[S,B,H]{sp:0}` saved | |
| 8 | NORM post_ln | `[S,B,H]{sp:0}` | `[S,B,H]{sp:0}` saved | |
| 9 | MATMUL fc1 | ln, w`[H,2F]{1:tp}` | `[S,B,2F]{2:tp}` saved | 列并行(gated) |
| 10 | ELEMENTWISE swiglu | `[S,B,2F]{2:tp}` | `[S,B,F]{2:tp}` saved | 带宽受限 |
| 11 | MATMUL fc2 | `[S,B,F]{2:tp}`, w`[F,H]{0:tp}` | `[S,B,H]` partial=tp | 行并行 |
| 12 | (reshard) | partial(tp)→{sp:0} | `[S,B,H]{sp:0}` | **派生通信** |
| 13 | ELEMENTWISE add | residual | `[S,B,H]{sp:0}` saved | |

> `{sp:0}` = 在 seq 维(0)上做 sequence-parallel 切分（仅 `sequence_parallel=True` 时生效，否则该维不切、张量在 tp 上复制）。

## 4. moe_decoder op 图（router + EP dispatch + grouped-GEMM，对照 `moe/{moe_layer,router,experts}.py`）

attention 段同 dense（#1–8）。FFN 段换成：

| # | op | inputs(shard) | output(shard, saved) | 备注 |
|---|---|---|---|---|
| 9 | MOE_ROUTER | `[S,B,H]` | logits`[S,B,n_experts]` saved + 路由 | gating |
| 10 | DISPATCH all-to-all | tokens`[S,B,H]` | `[T_local,H]{0:ep}` saved | T_local=本地专家收到的 token 数（见 §7） |
| 11 | MOE_GEMM fc1 | `[T_local,H]`, w`[n_exp_local,H,2·moe_F]{0:ep}` | `[T_local,2·moe_F]` saved | grouped GEMM；**专家纯 EP，不 TP 切**（`expert_parallel.py:330` `weight1:(Shard(0),)`），tp 折进 efsdp |
| 12 | ELEMENTWISE swiglu | `[T_local,2moe_F]` | `[T_local,moe_F]` saved | |
| 13 | MOE_GEMM fc2 | `[T_local,moe_F]`, w`[n_exp_local,moe_F,H]{0:ep}` | `[T_local,H]` | combine 还原 |
| 14 | COMBINE all-to-all | `[T_local,H]` | `[S,B,H]{sp:0}` saved | 还原 + reshard |
| (+) | 共享专家(n_shared) | 同 dense MLP，不走 EP | | 与路由专家并存 |

## 5. 符号 shape 求值 + sharding 代入（核心机制）

给定 `ModelSpec.dims` 的具体值 + `ParallelDims`(tp,cp,ep,dp_shard,pp,dp_replicate)：

1. **求值符号 shape**：把 `[S,B,(n_heads+2n_kv)·hd]` 等表达式用 dims 代入 → 全局 numel。
2. **图内切分（TP/CP/EP/SP）= 代入度数整除**：对 TensorRef 的每个 `shard` 维，numel 除以对应轴度数 → **local numel**。例：`{2:tp}` 的 `[S,B,2F]` → local = S·B·2F/tp。
3. **reshard 检测**：相邻 op 的 output/input sharding 不一致（如 partial(tp)→replicate）→ 标记一次集合通信（P0 只记其**存在与 volume**，时间留 P1）。
4. **外层并行（FSDP/DP/PP）= 聚合层施加**（见 §6/§7），不改 local numel。

## 6. 静态内存计算过程（param / grad / optimizer）

### 6.1 数清权重（从 op 图的 `is_weight` 张量）

- **attn/dense 权重**（按 tp 切）：qkv `H·(n_heads+2n_kv)·hd`、o `n_heads·hd·H`、dense-fc1 `H·2F`、fc2 `F·H`。
- **专家权重**（按 ep 切）：`n_experts · (H·2·moe_F + moe_F·H)`（+共享专家走 dense 计）。
- **复制权重**（不切或仅 vocab 切）：norm `~H`、embedding/lm_head `vocab·H`。
- 记 `N_attn(每层)`、`N_expert(每层)`、`N_repl`。

### 6.2 切分到每卡（用 `parallel_dims.py` 的 mesh 关系：`efsdp·ep = dp_shard·cp·tp`，`fsdp = dp_shard·cp`）

FSDP 策略 `optim_grads_params`（ZeRO-3：param+grad+opt 全分片）下，**每卡持久 param 元素数**：

```
attn 权重:    N_attn_stage  / (tp · dp_shard · cp)
专家权重:    N_expert_stage / (ep · efsdp) = N_expert_stage / (dp_shard · cp · tp)   # ep 被 efsdp 吸收
复制权重:    N_repl_stage   / (dp_shard · cp)        # embedding 若按 vocab 切再 /tp
```
`*_stage` = 仅本 PP stage 承载的层（按 `pipeline_parallel_layers_per_stage`）。

记 `P_dev = attn + 专家 + 复制` 三项之和（元素数）。

### 6.3 乘优化器字节倍数（AdamW 混合精度）

每个**已分片**参数元素的持久字节：

| 项 | dtype | 字节 |
|---|---|---|
| param 分片 | bf16 | 2 |
| grad 分片 | bf16(或fp32) | 2（或4） |
| master 权重 | fp32 | 4 |
| Adam m | fp32 | 4 |
| Adam v | fp32 | 4 |
| **合计 `b_state`** | | **16（或18）** |

> Muon：state 不同（动量 + Newton-Schulz 无 v），`b_state` 改对应值，配置化。

```
S_state(每卡) = P_dev · b_state
```
（FSDP just-in-time all-gather 的整层 bf16 权重属**临时 buffer**，计入 §8 workspace，不入持久 state。）

### 6.4 worked example（dense，Llama2-7B 量纲）

H=4096, n_heads=n_kv=32, hd=128, F=11008, vocab=32000, n_layers=32；TP=8, dp_shard=8, cp=1, pp=1, ep=1, AdamW b_state=16。

- 每层 attn 权重 N_attn = 4096·4096 (qkv, n_kv=n_heads) ·... 简化：≈ 4·H² + 3·H·F ≈ 4·16.8M + 3·45.1M ≈ 202M 元素/层 → 32 层 ≈ 6.45B + embedding 2·(32000·4096)=262M ≈ **6.7B 参数**（对得上 7B）。
- 每卡 P_dev = 6.7B / (tp·dp_shard·cp) = 6.7B/64 ≈ 105M 元素。
- S_state = 105M · 16B ≈ **1.68 GB/卡**。

## 7. 激活内存计算过程

### 7.1 单层激活 = Σ去重(各 op 的 `saves` 的 local 字节)

> §3/§4 表中"saved"列 = 各 op 的 `saves`（§2.1，autodiff 语义，通常为输入张量）；同一张量被多 op 需要只算一次（去重）。

**dense 层**（无重算、bf16=2B、flash-attn）逐项：

| saved output | 全局 numel | 切分 | local numel |
|---|---|---|---|
| 层输入/residual×2 | 2·S·B·H | sp | 2·S·B·H/tp（SP）或 2·S·B·H |
| qkv out | S·B·(n_heads+2n_kv)·hd | tp | /tp |
| flash-attn out | S·B·n_heads·hd | tp | /tp |
| flash-attn lse | S·B·n_heads | tp | /tp（小） |
| post_ln out | S·B·H | sp | /tp（SP） |
| fc1 out | S·B·2F | tp | /tp |
| swiglu out | S·B·F | tp | /tp |

```
A_layer^dense ≈ 2B · S·B · [ c_res·H  +  ((n_heads+2n_kv)·hd + n_heads·hd + 2F + F)/tp ]
   # c_res≈4（层输入 + post_ln + 2×residual）；无 SP 时 H 项不切；SP 时 c_res·H 项也 /tp
```
> 结构与 Megatron 激活公式(arXiv 2205.05198)一致（per-layer ∝ s·b·h/tp + 注意力项），可做交叉验证。

**moe 层**（balanced 路由，capacity_factor=C=1.0）：本地专家收到 token 数

```
T_local = (S·B·topk / n_experts) · (n_experts / ep) · C = S·B·topk·C / ep
```
saved 项：router logits `S·B·n_experts`；dispatched `T_local·H`；fc1 out `T_local·2moe_F`；swiglu `T_local·moe_F`。**专家纯 EP，不 /tp**（`expert_parallel.py:330`）。

```
A_layer^moe ≈ 2B · [ S·B·n_experts + S·B·topk·C/ep·( H + 3·moe_F ) ] + A_attn段
```
> 注：attn 段仍 /tp（注意力恒 TP 切），仅路由专家 FFN 纯 EP；共享专家走 dense MLP（/tp）。

### 7.2 单 microbatch 该 stage 激活（峰值由 §8 时间线仿真给出）

```
A_mb(stage) = Σ_{layer ∈ stage}  A_layer        # 单 microbatch、该 stage、post recompute/swap
```
> 跨 microbatch 的峰值叠加（1F1B 在飞 ≈ PP 份）、recompute 反向尖峰、FSDP/swap 预取双缓冲，**统一由 §8 内存时间线仿真计算**，不在此静态相乘。

### 7.3 recompute / swap 修正（作用于 saved 集）

- `full`（`full_recompute_layer` 命中层）：`A_layer` → 仅留 checkpoint 输入 ≈ `2B·S·B·H/tp`（省 ~10×）。
- `select`（`select_module`）：只减命中子模块的 saved output。
- `exclude_op`：命中 op output 保留（不减）。
- **swap**（`SwapConfig`/`cpu_offload`）：被 offload 的 saved output 从 resident 扣除（计入 prefetch buffer 到 workspace）。
- **三态不变量**：每个 saved output ∈ {resident, recomputed, offloaded} 之一，禁止重复扣减。

### 7.4 worked example（接 §6.4，加 PP=4、full recompute 关）

S=4096, B=1, 续上参数，TP=8, 无 SP。
A_layer ≈ 2B·4096·1·[ 4·4096 + (12288+4096+22016+11008)/8 ] = 2·4096·[16384 + 6176] ≈ **185 MB/层**（H 项未切因无 SP；开 SP 后 H 项 /8 → ~68MB/层）。
PP=4 → 每 stage 8 层；stage0 在飞 ≈ 4 microbatch：峰值激活 ≈ 4·8·185MB ≈ **5.9 GB**（无重算）。
full recompute → 每层降到 checkpoint 输入 ~2B·4096·4096 ≈ 34MB，峰值激活 ≈ 4·8·34MB ≈ **1.1 GB**（+单层重算驻留）。

## 8. 每卡峰值显存：内存时间线仿真（取代静态求和）

峰值是**沿执行时间线取 max**，必须仿真 alloc/free 事件——recompute 反向尖峰、FSDP 预取双缓冲、swap 预取都是**瞬时叠加**，静态求和抓不准。**纯解析**：确定性 walk 一个建模的调度，不执行脚本、不上 NPU。

### 8.1 内存桶（沿时间线维护）

| 桶 | 内容 | 释放时机 |
|---|---|---|
| `persistent` | 分片 param+grad+opt /(fsdp) − cpu_offload | 全程常驻 |
| `act_live` | 当前 pinned 的 `saves`（去重） | 对应 bwd 消费后 |
| `gather_buf` | FSDP all-gather 的整层 bf16 权重 | reshard policy 控（§8.3） |
| `grad_buf` | bwd 中 reduce-scatter 前的整层全量 grad | reduce-scatter 后 |
| `recomp_scratch` | 重算层反向时重物化的该层激活 | 该层 bwd 结束 |
| `swap_buf` | 从 CPU 预取回的激活 | 消费后 |
| `workspace` | 当前 op 的 kernel scratch | op 结束 |

```
peak = O_framework + max over events ( Σ 所有桶 )      # O_framework=allocator 预留+碎片(标定常数)
```

### 8.2 调度与事件 walk

- 1F1B：warm-up（PP−s 个 fwd，累积 `act_live` 到 ≈ PP 份）→ steady（1F1B 交替）→ cool-down；`interleave_num=v` 用交错式。
- per-op 事件（一步数千个，确定性瞬算）：fwd 分配 output/workspace、pin `saves`、释放无用输入；bwd 分配 grad、消费 `saves`、（重算层）重跑 fwd。

### 8.3 三类峰值影响的精确建模

**① recompute 反向尖峰**：fwd 时重算层 `saves` 即释（`act_live` 降），bwd 走到该层重跑 fwd → `recomp_scratch += 1 层完整激活`，叠在 `persistent + 其余 checkpoint 输入 + grad_buf + 该层 gather` 之上 → **全局峰值常在此处，不在 fwd 末**。

**② FSDP all-gather + 预取双缓冲**：
```
gather_buf = (1 + prefetch_depth) · G_layer          # G_layer = 层权重(/tp) gather 成整 bf16
reshard_after_forward = "always": 用完即释、bwd 再 gather（峰低/通信多）
                      = "never" : 各层 gather 全程驻留 → gather_buf 累成 Σ层（峰高）
```
bwd 另有一轮 all-gather 双缓冲 + reduce-scatter 前的 `grad_buf`。

**③ swap 预取**：`swap_buf = default_prefetch · 单次预取激活字节`；offload 的 `saves` 从 `act_live` 扣、预取窗口的重新计入。三态（resident/recomputed/offloaded）互斥，不重复扣减。

### 8.4 输出

报告：**峰值落点事件**（典型两处：fwd 末"在飞激活最满 + 最深预取" vs 首个重算层 bwd"recomp_scratch + grad_buf + gather"）+ 该刻**桶拆解** + 是否 > `ContextConfig.max_device_memory` + 最紧 stage。能解释"为什么 OOM 发生在反向"这类真实现象。

## 9. 验证（P0，纯软件）

1. **参数守恒**：Σ 全局 param = 已知模型参数量（dense 6.7B / MoE 总量）；各卡 P_dev × 切分度数 = 全局。
2. **退化**：tp=cp=ep=dp_shard=pp=1、无重算无 swap → 还原单卡公式。
3. **跨公式互证**：A_layer 对 Megatron 2205.05198 激活公式；S_state 对 ZeRO 16/18 字节/参数律。
4. **单调性**：tp↑→S_state↓、A↓；recompute 开→A↓；ep↑→专家 state↓、MoE 激活↓；pp↑→单 stage state↓ 但在飞激活↑。
5. **MoE balanced 自洽**：Σ_expert T_local = S·B·topk（不丢 token）。

## 10. P0 交付物与边界

**交付**：`model_spec.py`(数据结构 + 内存契约 `params`/`saves`/`workspace`) + `dense`/`moe` 两张 op 图 + `shape_eval`(符号求值+sharding 代入) + `static_mem`(§6) + `act_mem`(§7，Σ去重 saves) + **`mem_timeline.py`(§8 事件驱动峰值仿真：recompute 反向尖峰 / FSDP 预取双缓冲 / swap 预取)** + 验证单测(§9) + 1 dense + 1 MoE worked config。

**边界（→P1）**：所有**时间**（roofline、通信、bubble、swap stall）；EP all-to-all 的**时间**；MLA/CSA/hybrid 的 op 图（P0 先 dense+标准 MoE，§3/§4 的 GQA+SwiGLU 打底）。
