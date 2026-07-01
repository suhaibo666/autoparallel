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

### 6.3 乘优化器字节倍数（持久 = param + opt，**不含 grad**）

> ⚠ **真机修正（2026-06-30）**：grad **不是常驻**——反向 reduce-scatter 后即释（真机训练后常驻只剩 param+m+v）。故持久 `b_state` **剔除 grad**，grad 改算到 §8 的反向瞬态 `grad_buf`。

每个**已分片**参数元素的**持久**字节（按 params_dtype）：

| 项 | bf16 params | fp32 params（如 DSv3） |
|---|---|---|
| param（持久 master） | 2（bf16）+ 4（fp32 master） | 4（fp32 即 master，无额外 bf16 持久） |
| Adam m | 4 | 4 |
| Adam v | 4 | 4 |
| **持久 `b_state`** | **~14** | **12** |
| ~~grad~~（→`grad_buf` 瞬态） | ~~2/4~~ | ~~4~~ |

> 真机实测（DSv3 fp32）：训练后常驻 3862 MiB = 12 B/param × P_dev，**对得上**（§8.7）。Muon：state 不同（动量、无 v），`b_state` 配置化。

```
S_state(每卡, 持久) = P_dev · b_state          # 不含 grad；grad 见 §8 grad_buf
```
（FSDP just-in-time all-gather 的整层权重属**反向瞬态 `gather_buf`**，不入持久 state。）

### 6.4 worked example（dense，Llama2-7B 量纲）

H=4096, n_heads=n_kv=32, hd=128, F=11008, vocab=32000, n_layers=32；TP=8, dp_shard=8, cp=1, pp=1, ep=1, AdamW **持久 b_state=14**（bf16 param2+fp32 master4+m4+v4，**剔 grad**）。

- 每层 attn 权重 N_attn = 4096·4096 (qkv, n_kv=n_heads) ·... 简化：≈ 4·H² + 3·H·F ≈ 4·16.8M + 3·45.1M ≈ 202M 元素/层 → 32 层 ≈ 6.45B + embedding 2·(32000·4096)=262M ≈ **6.7B 参数**（对得上 7B）。
- 每卡 P_dev = 6.7B / (tp·dp_shard·cp) = 6.7B/64 ≈ 105M 元素。
- S_state(持久) = 105M · 14B ≈ **1.47 GB/卡**（grad 1276MiB 另计反向 `grad_buf`）。

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

峰值是**沿执行时间线取 max**，必须仿真 alloc/free 事件——recompute 反向尖峰、FSDP gather/grad、loss 区 fp32 物化都是**瞬时叠加**，静态求和抓不准。**纯解析**：确定性 walk 一个建模的调度，不执行脚本、不上 NPU。
> 本节经 DeepSeek-V3 真机对标修订（2026-06-30，见 §8.7）。

### 8.1 内存桶（沿时间线维护）

| 桶 | 内容 | 释放时机 |
|---|---|---|
| `persistent` | 分片 **param+opt** /(fsdp) − cpu_offload（**不含 grad**，见 §8.4） | 全程常驻 |
| `act_live` | 当前 pinned 的 `saves`（去重；支持 per-tensor dtype，如 fp32 loss 张量） | 对应 bwd 消费后 |
| `gather_buf` | FSDP all-gather 的整层权重（compute dtype） | reshard policy 控 |
| `grad_buf` | bwd reduce-scatter 前的整层 **full grad**（grad dtype，可 fp32） | reduce-scatter 后 |
| `recomp_scratch` | 重算层反向重物化的该层正向激活（见 §8.5 的修正） | 该层 bwd 结束 |
| `bwd_scratch` | op 级反向临时物化（如 loss `probs` fp32，对照 `loss.py`） | 该 op bwd 结束 |
| `swap_buf` | 从 CPU 预取回的激活 | 消费后 |
| `workspace` | 当前 op 的 kernel scratch | op 结束 |

```
peak = max over events ( Σ 上述桶 )  +  framework_reserve(config)   # framework_reserve 见 §8.6，非常数
```

### 8.2 调度与事件 walk

1F1B：warm-up（PP−1−s 个 fwd）→ steady（1F1B 交替）→ cool-down；`interleave_num=v` 用交错式。
- **FWD 每层**：`gather_buf=整层权重`（reshard 后即释）+ `workspace` → 采样；pin `saves`（重算层只 pin checkpoint 输入；swap 层 pin 0）→ `act_live`。
- **BWD 每层（逆序）**：`gather_buf`(re-gather) + `grad_buf`(full grad) + `recomp_scratch`(重算层) + `bwd_scratch`(op) **四者共存** → 采样 → 清零 → 释放该层 pinned `act_live`。

### 8.3 大 vocab 的 fp32 loss 区（真机峰值大头）

对照 `pynative/loss/loss.py`（非 chunk 的 `CrossEntropyLoss` = `_LogSoftmax`+`_NLLLoss`）：
`_LogSoftmax.forward` 把 logits **cast fp32**、返回的 `log_softmax`(fp32) 被 saved（`ctx.log_softmax`）；
**`_NLLLoss.backward`（loss.py:80-82）同时物化** `probs=exp(−log_softmax)`(fp32) 与 `scatter_add` 出的
`grad_log_softmax`(fp32)，二者与 saved `log_softmax` **共存** → 反向峰值瞬时压着 **3 个满 vocab fp32 张量**。
vocab×seq 巨大时这是**真机峰值的绝对大头**。建模：
- `act_live` += logits(bf16) + **log_softmax(fp32, per-tensor dtype=4)**（前向 saved 的两个）。
- nll/loss op 的 `bwd_scratch` = **probs + grad_log_softmax = 2×(4·S·B·vocab)**（反向新物化、不在 saves 里的两个满 vocab fp32 张量）。
- ⚠ **修正（2026-06-30）**：此前只建了 `probs`（1×4·S·B·vocab），**漏了 `scatter_add` 的 `grad_log_softmax`**，
  导致 ~2020 MiB（@seq4096, vocab129280）被错误兜进 `framework_reserve`。补建后 `framework_reserve` 从 **2197→177**（§8.6），
  pred 不变（仍 1.000）——即"这 2GB 是 loss 反向激活、不是框架开销"，详见 §8.6。
- 注：`_ChunkCrossEntropyLoss`（loss.py:249）分块重算正是为消除这多份满 size logits 梯度而设计；本配置未开 chunk，故残余恰 ≈ 1 份满 V_fp32。

### 8.4 grad 不是常驻（真机修正）

fp32 AdamW 训练后常驻 = param(4)+m(4)+v(4) = **12 B/param**（不是 16）；**grad(4B) 是反向瞬态**，reduce-scatter 后即释。故：
- `persistent`（§6）= param + optimizer state（**剔除 grad**）。
- grad 进 `grad_buf`（反向瞬态，full-unsharded，grad dtype）。

### 8.5 recompute 反向的三处修正（⚠ 当前未被真机验证）

`recomp_scratch` = 重算层反向重跑 forward 重物化的正向激活。三个问题须修：
1. **双算 checkpoint 输入**：full 重算层 fwd 时 checkpoint 输入已 pin 进 `act_live`，而 `recomp_scratch=该层 saves 之和`又含层输入 → 应 `recomp_scratch = 该层 saves − checkpoint 输入`。
2. **应取 forward 峰值工作集，非 saves 之和**：重跑 forward 时非 saved 中间量也瞬时活着，峰值可能 > saves；严格应对该层做一次 **mini-forward 时间线求 max-live**（MLA/MoE 多中间量层尤其）。
3. **验证缺口**：DSv3 4L/8L 峰值落在 loss 层反向（`bwd_scratch` 主导），`recomp_scratch=0`——**这条对已验证的 1.000/0.996 毫无贡献，是未验证项**。须设计**让重算 transformer 层成为峰值**的配置（大 hidden / 小 vocab / PP 增在飞 microbatch）单独验证它。

### 8.6 `framework_reserve`：**allocated 峰值的残余，不含 HCCL**（ep=2 真机修正）

> ⚠ **ep=2 真机点证伪了"HCCL×组数"假设（2026-06-30）**：ep=1→2 多一个 EP 通信组，**allocated 峰值不变**（12473→12474），但 **reserved 涨**（13446→13750）。结论：**HCCL 通信缓冲在 reserved 池、不在 allocated 峰值**。评估器预测 `max_memory_allocated`（张量占用，OOM 相关），故 **HCCL 不计入 framework_reserve(allocated)**。
>
> **根因（用户洞见）**：EP/CP/TP 子通信域**复用同一 rank 网格**——`parallel_dims.py: efsdp·ep = dp_shard·cp·tp`，EP 是从 `dp_shard·cp·tp` 区 **carve** 出来的，是 world group 的**子通信器、复用 DP/FSDP 的 rank**，不是新增独立通信域。所以子域是**重叠**的、其 buffer **不可按域数叠加**，且只落 reserved。`num_comm_groups×200MB` 这种线性叠加是错的（已改名 `num_distinct_communicators` 并标注"仅 reserved 上界粗估"）。

| 项 | 归属 | 缩放律 / 现状 |
|---|---|---|
| **HCCL 通信缓冲** | **reserved 池**（≠ allocated 峰值） | `200MB × 通信组数`；仅当预测 reserved 时用（`hccl_reserved_buffer(pc)`） |
| `framework_reserve`(allocated) | allocated 峰值残余 | = flash workspace + MoE all-to-all staging + 分配器块对齐取整；**真机标定 ≈177 MiB @ seq4096**（**原 2197，其中 ~2020 是 loss 反向 `grad_log_softmax`，已于 §8.3 显式建进 `bwd_scratch` 剔出**） |
| 其随 seq 的缩放（flash_ws） | allocated | 待 **seq-varying 真机点**验证（开放项；注：loss 区 ~4040 MiB 现已随 seq·vocab 自动缩放，不再靠常数兜） |

→ 即：HCCL 不进 allocated；loss 反向 fp32 大头（probs+grad_log_softmax）现**显式建模、随 seq·vocab 缩放**；
`framework_reserve(allocated)` 收窄为 ≈177 MiB 的小残余（flash/MoE staging/对齐），经 ep=1/2 验证对 ep 恒定。
**关键认知**：原 2197 的"framework_reserve"并非真·框架开销，其 92% 是漏建的 loss 反向激活——`framework_reserve` 是**记账兜底位**，
不是物理实体；把可建模项逐一建出来后它应持续收窄（理想 →仅分配器碎片）。

### 8.7 真机验证现状（DeepSeek-V3，2026-06-30，3 个数据点）

| 配置 | 静态 persistent | 峰值预测 | 真机 alloc | ratio |
|---|---|---|---|---|
| 4L FSDP-2 ep1 | 3830 vs 3862 | 12472.5 | 12473 | **1.000**（标定点） |
| 8L FSDP-2 ep1 | — | 13896 | 13953 | **0.996**（跨层数） |
| **4L FSDP-2 ep2** | — | 12472.5 | **12474** | **1.000**（跨 ep，证伪 HCCL-in-alloc） |

**已验证**：persistent（<1%）、loss 区 fp32、FSDP gather/grad、跨层数缩放、**跨 ep（framework_reserve 对 ep 恒定 + HCCL 归 reserved）**、
**loss 反向 grad_log_softmax（真机 Profiler 峰值算子 = `ScatterAddExt` 点名，§8.9）**、**内存 timeline 曲线级（峰值+峰值位置，§8.9）**。
**未验证（开放项）**：`recomp_scratch`（§8.5，峰值没踩到；§8.9 观察到真机 transformer 反向台阶比仿真高 ~1500 MiB，属 off-peak，与 §8.5② 欠建一致，待"让重算层成峰值"的配置验证）；
`framework_reserve` 随 **seq/tp** 的缩放（仅 seq4096/tp1，需 seq-varying / tp-varying 点）；swap/select-recompute（无真机点）。

> **逐桶验证原则（已见成效）**：每桶须设计"能让它主导峰值"的真机配置单独验证。ep=2 点正是如此**证伪了一个错误假设**（HCCL 本不在 allocated）——这就是变配置验证的价值。

### 8.8 输出契约：PP **逐 stage 全量 profile**（不是只取 max）

> **核心约定**：内存仿真的产出是**每个 PP stage 各自的完整内存 profile**，而**不是**单一的全局峰值。`tightest_stage` 只是\
> 为 OOM 判定派生出的便利量，**绝不是唯一产出**。每个 PP stage 跑在**不同的物理设备组**上、**负载彼此不同**，\
> 因此必须把**所有 stage**都仿真出来、逐 stage 报，供流水线层划分 / 负载均衡 / 逐设备 OOM 决策使用。

仿真器（`mem_timeline.py: MemTimeline.simulate`）对 **`g.stages` 里的每个 stage 独立**跑一遍 §8.2 的事件 walk，
逐 stage 产出 `StagePeak{stage, peak_bytes, breakdown(8桶快照), peak_event, oom}`。门面（`report.py: Evaluator.evaluate`）
聚合为：

```python
@dataclass(frozen=True)
class PeakMemoryReport:
    per_stage: list[StagePeak]   # ★ 主产出：全部 stage（按 stage 升序），各带完整 8 桶 breakdown
    tightest_stage: int          # 派生便利量：peak_bytes 最大的 stage（仅用于"是否 OOM/选谁卡最紧"）
    oom: bool                    # 任意 stage 超 max_device_memory 即 True（逐 stage 判，非只判 max）
```

**stage 间为何不同**（两个独立来源，缺一不可，都必须逐 stage 建模）：
1. **层分布不同**（`parallel_model.py: _layer_to_stage`）：每个 stage 只承载自己那段层；含 embedding / lm_head / loss 的
   首尾 stage 通常显著重于中间 stage（lm_head 的 fp32 loss 区不可被 PP 拆，恒钉末 stage）。
2. **在飞 microbatch 数不同**（1F1B warmup 深度 `min(PP−1−s, m)`，§8.2）：靠前的 stage warmup 更深、`act_live` 堆叠更多份
   microbatch 的 saved 激活（full 重算时每份仅 checkpoint 输入、堆叠很轻；关重算时这是另一处峰值来源）。

**worked example（DSv3 缩层 N=8, dp_shard=2, full 重算；`analyze_matrix.py` 实跑）** —— 同一配置下各 stage 峰值差到 2 倍：

| pp | stage | 层 | 在飞 mb | 峰值 MiB | 主导 | 备注 |
|---|---|---|---|---|---|---|
| 2 | 0 | [0–4] | 2 | 6082.9 | persistent | 无 lm_head，轻 |
| 2 | 1 | [5–9] | 1 | **11335.9** | lm_head bwd_scratch+act | tightest |
| 4 | 0 | [0–1] | 4 | 5043.2 | persistent | 4 份在飞仅 +42MiB act |
| 4 | 1 | [2–3] | 3 | 3668.8 | persistent | |
| 4 | 2 | [4–5] | 2 | 3640.8 | persistent | |
| 4 | 3 | [6–9] | 1 | **10980.0** | lm_head | tightest |

> 结论与教训：**只读 `per_stage[0]`（最轻 stage）会把 PP 收益夸大近 3 倍**（6083 vs 真实 11336）。消费方（策略搜索 / 选型）
> 应消费**整条 `per_stage`**：逐 stage 判 OOM、看负载是否均衡（首尾 stage 偏重提示需调 `layers_per_stage` 重新切层）。
> `tightest_stage` 仅用于"该配置在最紧设备上是否放得下"的快速判定。

### 8.9 内存 timeline 曲线级验证（真机 Profiler vs 仿真逐事件，2026-07-01）

除峰值标量（§8.7）外，还做了**曲线级**对标：真机用 `ms.Profiler(profile_memory=True)` 采逐时刻 allocated
（`memory_record.csv`）+ 逐算子内存（`operator_memory.csv`）；仿真用 `Evaluator.evaluate(record_timeline=True)`
产出逐事件 `StagePeak.timeline`。runner/脚本见 skill `real-machine-memory-sim` §6 与 `analysis/realmachine/`。

**DSv3 4L FSDP-2 结果**：

| 项 | 真机 | 仿真 | 结论 |
|---|---|---|---|
| persistent 基线 | 3922 MB | 4064 MB | ≈（真机 resident 3862） |
| 峰值 peak | **12473.1**（算子级） | **12472.5** | **1.0000** |
| **峰值算子** | **`ScatterAddExt`** | `bwd@lm_head` | **同一处**（loss 反向） |

> **关键佐证**：真机把 allocated 顶到峰值的算子是 `ScatterAddExt` = `loss.py:82` `_NLLLoss.backward` 的 `scatter_add`，
> 其 `Allocation Total Allocated = 12473.1 MB` = MEMPROBE 峰值。这**独立证实** §8.3 的建模：峰值就在 loss 反向 `grad_log_softmax`
> 物化那一刻，即 §8.6 从 `framework_reserve` 拆进 `bwd_scratch` 的那 ~2020 MiB 是**真实 loss 激活、非框架开销**。

**如何看"真机 vs 仿真曲线波动差异大"**（这是设计层面的口径，不是 bug）：
1. **分辨率**：真机 ~2400 采样/s（**逐 kernel** alloc/free）vs 仿真 **13 点/步**（逐 layer 事件）。真机每个 kernel 的微小 alloc/free 都记，
   仿真把整层塌成 1–2 个采样 → 真机天然锯齿、仿真平滑。**仿真是峰值包络模型，不是逐 kernel tracer**。
2. **范围**：仿真只建**一个稳态 step 的逻辑峰值包络**；真机曲线是整 8.7s 全程——含 warmup + kernel 编译（首步 hump）、
   步间/优化器/数据集空档（~7100 MB 平台且采样稀疏）、以及末尾 3 个 fast step。多数"波动"是仿真**故意不建**的运行时行为。
3. **该匹配的匹配了**：把真机**缩到单个 step**（`comparison_onestep.png`），形状与仿真一致——前向低平 → loss 反向**单尖峰**（ScatterAddExt）
   → transformer 层反向若干小台阶 → 回落。峰值值 + 峰值位置（用于 OOM/选型的两件事）都对上。
4. **一个 off-peak 观察**：真机 transformer 层反向台阶 ~6000–6800 MB，比仿真的 bwd@transformer ~4700–5300 高 ~1500 MiB。
   这在峰值以下、**不影响峰值/OOM 预测**，但与 §8.5② 一致（`recomp_scratch` 当前取 `saves−checkpoint` 而非重跑 forward 的 max-live，**欠建**）——
   若未来配置把峰值移到重算的 transformer 层，需按 §8.5② 修 `recomp_scratch`。曲线级验证正好把这个欠建**看见**了。

> **口径总结**：内存仿真的验收标准是**峰值大小 + 峰值发生位置**（OOM 与选型只关心这两件），已 1.000 + 算子级点名对上；
> 曲线细结构（kernel 级抖动、warmup、步间）不在仿真目标内，差异属**分辨率与范围**、非精度问题。

## 9. 验证（P0，纯软件）

1. **参数守恒**：Σ 全局 param = 已知模型参数量（dense 6.7B / MoE 总量）；各卡 P_dev × 切分度数 = 全局。
2. **退化**：tp=cp=ep=dp_shard=pp=1、无重算无 swap → 还原单卡公式。
3. **跨公式互证**：A_layer 对 Megatron 2205.05198 激活公式；S_state 对 ZeRO **持久 12–14 B/param（剔 grad）** + 真机常驻（§8.7）。
4. **单调性**：tp↑→S_state↓、A↓；recompute 开→A↓；ep↑→专家 state↓、MoE 激活↓；pp↑→单 stage state↓ 但在飞激活↑。
5. **MoE balanced 自洽**：Σ_expert T_local = S·B·topk（不丢 token）。

## 10. P0 交付物与边界

**交付**：`model_spec.py`(数据结构 + 内存契约 `params`/`saves`/`workspace`) + `dense`/`moe` 两张 op 图 + `shape_eval`(符号求值+sharding 代入) + `static_mem`(§6) + `act_mem`(§7，Σ去重 saves) + **`mem_timeline.py`(§8 事件驱动峰值仿真：recompute 反向尖峰 / FSDP 预取双缓冲 / swap 预取)** + **`report.py`(§8.8 门面 `Evaluator` + 输出契约 `PeakMemoryReport`：PP **逐 stage 全量 profile** + 派生 `tightest_stage`/`oom`)** + 验证单测(§9) + 1 dense + 1 MoE worked config。

**边界（→P1）**：所有**时间**（roofline、通信、bubble、swap stall）；EP all-to-all 的**时间**；MLA/CSA/hybrid 的 op 图（P0 先 dense+标准 MoE，§3/§4 的 GQA+SwiGLU 打底）。
