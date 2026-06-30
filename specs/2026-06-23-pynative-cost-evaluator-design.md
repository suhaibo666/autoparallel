# PyNative 并行策略代价评估器 — 设计文档

> 创建: 2026-06-23 · **重大修订: 2026-06-29（模型描述改为"声明式 op 图 ModelSpec"）** · 状态: Draft（待评审）
> 适用: MindFormers `mindformers/pynative/`（TorchTitan 移植，对等 Megatron 的 5D 训练栈）
> 实现架构（模块/时序/接口）: [[2026-06-29-evaluator-implementation-design]] · P0 内存细化: [[2026-06-29-p0-modelspec-and-memory-design]]

## 0. 一句话目标

给定一份**专家配置**（trainer 的 `TrainConfig`：5D 并行 + recompute + swap）+ 一份**模型描述（ModelSpec）**，**离线、纯解析地**估算该配置在目标 NPU 集群上的 **① 每卡峰值显存 + 是否 OOM + 构成拆解** 和 **② 单步时间 + 瓶颈拆解 + MFU**。本评估器是"自动并行"系统里的 **evaluator（评估器）**，与 **optimizer（搜索器）** 解耦——先把评估器做准，搜索器是将来包在它外面的循环。

$$\text{自动并行} = \text{搜索器(optimizer)} \circ \underbrace{\text{评估器(evaluator)}}_{\text{本设计}}$$

## 1. 背景

- 完整"5D(DP/TP/PP/CP/EP) + swap + recompute 自动策略生成"在公开文献里是空白交集（联合搜索系统 Mist/Aceso 停在 DP/TP/PP；生产 5D 系统 Megatron MoE 是手工启发式）。**评估器（代价模型）是任何方案的承重柱**——"代价模型准确性 > 搜索算法先进性"。
- 分阶段：先建评估器（本设计），移除组合搜索复杂度，且评估器在生产中独立有用（OOM 预测、配置对比、容量规划）。

## 2. 范围与非目标

**范围（v1）**：
- 双指标：显存 **和** 时间都要可信。
- 全模型覆盖：dense + MoE/EP + 实验性混合注意力（MLA / CSA / hybrid / indexer / MTP）。
- 全 5D + recompute（None/full/select/exclude_op/recompute_comm）+ swap（激活 offload + FSDP cpu_offload）。
- 纯解析优先（无真机 profiling），预留缩层 profiling 标定接口。

**非目标（v1，YAGNI）**：
- 不做策略搜索/优化（评估器只给单点打分；搜索器是 Phase 2）。
- **不依赖执行/解析模型脚本**：模型由**人写的声明式 op 图数据**描述，不 trace、不实例化真实模块、不 AST 解析脚本、不需 NPU/`hyper_parallel`。
- 不追求声明图之外的 op 级精确性；时间无标定时走 roofline，绝对值可能偏，但**相对排序与瓶颈定位有效**。

## 3. 精度预期（诚实预判）

| 指标 | 无 profiling（纯解析） | 有缩层 profiling 标定后 |
|---|---|---|
| 显存 | 闭式可解，~5–10% | <10% |
| 时间 | roofline + 结构化 overlap/bubble，绝对值可能几十%，**相对/瓶颈有效** | 单层时间 <15% |

架构铁律：**"效率因子 η" 单列一层**。无标定用默认 η，有缩层数据时**只换 η、不动结构**。

## 4. 总体架构

**核心思想**：模型 = **一张声明式 op 图（ModelSpec，数据）**；评估器把符号 shape × 并行度数代入算 roofline，再施加三套变换聚合。

```
输入                          核心层                              输出
┌──────────────┐  ┌────────────────────────────────┐
│ ModelSpec     │  │ 展开层: 按 layer_pattern 实例化   │
│ (声明式 op 图) │─▶│   各层 op 图；ModelSpec 符号维 +  │
│ TrainConfig   │  │   ParallelDims 度数 → local shape │
│ 硬件规格表    │  │     │                            │  ┌──────────────┐
│ (缩层标定)    │  │     ▼                            │─▶│ 每卡峰值显存  │
└──────────────┘  │ 代价层: 每 OpSpec → 子CostRecord  │  │ +OOM+拆解     │
                  │   (roofline FLOPs/bytes/分类;     │  ├──────────────┤
                  │    sharding 代入; reshard→通信)   │─▶│ 单步时间      │
                  │     ▼                            │  │ +瓶颈+MFU     │
                  │ 聚合层: ①切分 ②recompute ③swap    │  └──────────────┘
                  │   → 显存时间线 + 时间/bubble       │
                  └────────────────────────────────┘
```

### 4.1 模型描述 = 声明式 op 图（数据），非脚本/非 AST/非 trace

它**不是**三样东西：不执行脚本、不 AST 解析脚本、不坍缩成单个闭式公式。它**是**人按架构写的声明式数据：

- **符号 shape**：op 的 shape 用 `H/F/n_heads/S/B/tp/cp/ep` 等符号表达 → **一份 spec 覆盖所有模型规模 × 所有并行配置**，评估器代入求值。
- **切分写在 op 的 TensorRef 上**（parallel-strategy-aware），**度数来自 config**（parallel-degree-agnostic）；两者组合 = DTensor 的 placement + mesh degree。
- **集合通信默认由"相邻 op 的 sharding 不匹配"自动派生**（DTensor reshard 式），少数特殊 case 允许在图里显式声明为 op。
- **漂移缓解**：将来可 trace 真实模型，diff 声明图 → 作者错误/漂移告警（仅验证用，非日常依赖）。

## 5. 输入

1. **ModelSpec**（声明式 op 图，见 §5.1）——评估器**只依赖它**描述模型，与 `pynative/transformers/*.py` 解耦。可由 `parallel_core/transformer_config.py` 的 `TransformerConfig` 经 adapter 自动填充（便利），但核心不 import 真实模块。
2. **`TrainConfig`**（复用 `pynative/config/config.py`）：`ParallelismConfig`(5D+SP+mc2+dispatcher) / `RecomputeConfig`+`RecomputeCommConfig` / `SwapConfig`+`cpu_offload` / `OptimizerConfig`(AdamW/Muon) / `TrainingConfig`(batch) / `ContextConfig.max_device_memory`(OOM 阈值)。
3. **硬件规格表**：`peak_FLOPS`(按 dtype)、`HBM_BW`、节点内/节点间互连 α-β、PCIe/offload 带宽、`max_device_memory`。
4. **（可选）CalibrationTable**：缩层 profiling 产出，键 `(op_type, local_shape, dtype)` → `{fwd_us, bwd_us, peak_act_bytes}`。

### 5.1 ModelSpec schema

```python
ModelSpec:
  dims: {H, F, n_heads, n_kv, head_dim, S, B, vocab, ...}    # 符号维
  layer_pattern: ["dense_decoder", "moe_decoder", "mla_decoder", "mtp", ...]  # 每层类型
  layers: { layer_type -> LayerSpec }
  dtype_bytes: int

LayerSpec:
  ops: list[OpSpec]

OpSpec:
  type: "matmul" | "flash_attn" | "swiglu" | "rmsnorm" | "softmax" | "rope"
        | "moe_gemm" | "all_reduce" | "all_to_all" | ...   # 显式集合通信可选
  inputs:  list[TensorRef]      # 含权重(matmul 的 weight 也是一个 TensorRef)
  output:  TensorRef
  # ---- 内存契约(每 op 自声明；融合/非融合自动跟随；详见 P0 §2.1) ----
  params:  list[TensorRef]      # 拥有的权重 → param/grad/opt(融合不变)
  saves:   list[TensorRef]      # 为自己 backward 保留的张量(=save_for_backward；多为输入) → 激活(融合敏感)
  workspace: ShapeExpr          # 执行期 kernel scratch(瞬时；融合相关)
  attrs:   {dtype, is_collective, ...}

TensorRef:
  shape: [Dim, ...]             # 符号,如 [S, B, H]
  shard: {dim_index -> axis}    # 切分标注: {2:"tp"} = H 切 tp; {0:"cp"} = seq 切 cp; weight {1:"tp"}=列并行
  is_weight: bool               # 区分持久权重(state) vs 激活
```

### 5.2 例：dense MLP（SwiGLU）op 图（对照真实 `mlp.py:112-147`）

| op | inputs(shard) | output(shard) | saved | 派生通信 |
|---|---|---|---|---|
| `matmul` fc1 | x`[S,B,H]`, w`[H,2F]{1:tp,is_weight}` | `[S,B,2F]{2:tp}` | ✓ | — |
| `swiglu` | `[S,B,2F]{2:tp}` | `[S,B,F]{2:tp}` | ✓ | — (带宽受限,切分透传) |
| `matmul` fc2 | x`[S,B,F]{2:tp}`, w`[F,H]{0:tp,is_weight}` | `[S,B,H]` Partial(tp) | — | — |
| (派生)reshard | `[S,B,H]` Partial(tp)→Replicate | `[S,B,H]` | — | **mismatch→all-reduce** vol=S·B·H；SP 则→Shard(seq) 插 reduce-scatter |

## 6. CostRecord（按 OpSpec 计算，逐 op 子记录后聚合）

| 组 | 来源 | 说明 |
|---|---|---|
| **state** | op 的 `params`(is_weight) | param_numel/dtype/shard；grad/opt 由聚合层按 optimizer 派生(AdamW 18B/param 量级，Muon 不同)；**融合不变** |
| **compute** | 每 OpSpec | `flops, bytes_in, bytes_out`(从 in/out local shape 算)，`类别`(算力/带宽受限)，`eta_key`(=标定缝)；bwd 默认 GEMM≈2×fwd |
| **comm** | reshard 派生 / 显式 op | `ctype, volume, group_axis(tp/cp/ep), phase` |
| **activation** | op 的 `saves`(去重) | `numel, dtype, recompute_cost, swappable`；**融合敏感**(P0 §2.1) |
| **transient** | 每 OpSpec | `workspace_bytes`(FlashAttn workspace、staging) |

## 7. Op Roofline 代价库（逐 op，符号 shape 上）

```
local_shape = 符号 shape 各维按 shard 标注的 axis 度数整除         # 代入,非推断
t_op = max( FLOPs_op / (peak_FLOPS·d_dtype) , bytes_op / peak_HBM_BW ) / η_op
```

- `η_op∈(0,1]`：无标定用每类 op 保守默认（matmul ~0.6–0.85、FlashAttn ~0.5、elementwise 按可达带宽）；缩层 profiling 反解替换，**键 `eta_key` 共用**。
- 即使无标定，roofline 仍正确分类**算力/带宽受限**，瓶颈定位有效。
- 覆盖 op 类：`matmul/bmm` · `FlashAttn fwd/bwd`(MLA/CSA/hybrid 各形状) · `elementwise/SwiGLU` · `RMSNorm/softmax/rope` · `moe_gemm`(按 tokens-per-expert) · `集合通信`(α-β)。

## 8. 聚合层三套变换（有序：①切分 → ②recompute → ③swap）

### ① 5D 切分 = 代入度数 + reshard 检测

区分两类并行：

- **图内切分(TP/CP/EP)**：写在 op 的 TensorRef `shard` 上 → 代入度数得 local shape；相邻 op `shard` 不匹配处**自动插集合通信**。

  | 轴 | TensorRef 标注体现 | 派生通信 |
  |---|---|---|
  | TP `tensor_parallel` | 权重列/行并行 + 激活 hidden/ffn 维 tp | 行并行 Partial→Replicate=all-reduce；`sequence_parallel`→reduce-scatter+all-gather；`enable_mc2` 改 overlap |
  | CP `context_parallel` | 激活 seq 维 cp | colossal=ring P2P / ulysses=all-to-all(`context_parallel_method`) |
  | EP `expert_parallel` | expert/token 维 ep | dispatch/combine all-to-all，量随**路由分布(输入参数)**；`moe_token_dispatcher_type` 去冗余改量 |

- **外层并行(FSDP/DP/PP)**：聚合层施加，不改图内 local shape：
  - FSDP `data_parallel_shard`：**param+grad+opt 存储 /(dp_shard·cp)**；权重 just-in-time all-gather(fwd+bwd)、grad reduce-scatter；`reshard_after_forward_policy` 控重 gather。
  - DP `dp_replicate`：复制 + grad all-reduce。
  - PP `pipeline_parallel`：按 `layers_per_stage` 把**层分到 stage**（非 per-op 切分）；定在飞 microbatch 时间线。

### ② recompute（作用于 `saves` 的 op output，按 `RecomputeConfig`）

| `mode` | activation | time |
|---|---|---|
| None | 全保存 | 无 |
| full(`full_recompute_layer`) | 该层只留 checkpoint 输入 | +重跑该层 fwd op |
| select(`select_module`) | 仅目标子模块的 op output 丢保存 | +部分重算 |
| `exclude_op` | 命中 op output 保留(MUST_SAVE) | 这些 op 不重跑（含避免重发集合通信，配 `recompute_comm`） |

### ③ swap（作用于 `saves` output 与 state）

| 来源 | 显存 | time |
|---|---|---|
| 激活 swap `SwapConfig`(`layer_swap/op_swap`) | 选中 output 离开峰值 + prefetch buf | +D2H/H2D=bytes/PCIe_BW；`default_prefetch` 决定隐藏 or 暴露 stall |
| FSDP `cpu_offload` | param/grad/opt 下沉 CPU | +param H2D + grad D2H + CPU 优化器步 |

### 不变量

**一个 `saves` output 只能处于 resident / recomputed / offloaded 三态之一**；recompute 与 swap 对同一张量互斥，聚合层禁止重复扣减。

## 9. 显存时间线 & PP bubble

**峰值显存（每 stage，事件驱动内存时间线仿真）**：peak 是沿 fwd→bwd 执行时间线取 max，须仿真 alloc/free 事件——recompute 反向尖峰、FSDP 预取双缓冲、swap 预取均为**瞬时叠加**，静态求和抓不准。详见 [[2026-06-29-p0-modelspec-and-memory-design]] §8。

```
peak(s) = O_framework + max over events ( Σ 桶 )
桶 = persistent(分片 param+grad+opt − offload) + act_live(去重 saves)
   + gather_buf((1+prefetch)·层权重) + grad_buf + recomp_scratch + swap_buf + workspace
```
OOM ⟺ 任一 stage peak > `context.max_device_memory`。

**单步时间**：

```
T_step ≈ (m + PP − 1)/m · Σ t_stage          # 1F1B; bubble 占比 (PP−1)/m
       + T_DP/FSDP_grad_sync^exposed
       + T_opt                               # cpu_offload 时走 CPU + param H2D
t_stage = Σ_op(compute + exposed_comm + recompute + exposed_swap_stall)
exposed_comm = max(0, t_comm − t_可掩盖compute)
```
- `interleave_num=v` → bubble (PP−1)/(v·m)；`enable_dxdw_split`/`overlap_b_f`/`overlap_p2p` 进一步压。
- `MFU = 有效FLOPs / (T_step · peak_FLOPS · world_size)`。

## 10. 验证策略（无真机 → 有缩层数据 的阶梯）

1. **守恒/不变量自检（纯软件单测）**：总 param_numel = 模型真实参数量；总 FLOPs 满足 dense `6ND`(±MoE/attn 修正)；各 rank 切分态求和 = 未切分态；recompute/swap 省字节 ≤ 原激活；**退化测试**：全度=1、无重算无 swap → 还原单卡公式。
2. **跨模型互证**：显存对 **Megatron 激活显存公式(arXiv 2205.05198)**、时间对 **Calculon**，独立推导吻合即可信。
3. **性质测试**：recompute↑→激活↓时间↑；TP↑→per-rank state↓（property-based）。
4. **声明图核对**：（有机器时）trace 真实模型 diff 声明 op 图，抓 shape/sharding 作者错误。
5. **缩层 profiling 对接**：标定填 η → 复测 per-op 时间/per-collective/per-rank 峰值(`profile_memory`)，出分项误差报告（目标:显存<10%、单层时间标定后<15%）。
6. **golden 配置**：公开已知吞吐/显存的配置（如 DeepSeek-V3）端到端比对。

## 11. layer 类型 op 图清单（要按架构声明的层型，对照 `pynative/transformers/` 核实）

| layer 类型 | 对照源文件（核实用） | op 图要点 |
|---|---|---|
| `dense_decoder`(norm+attn+MLP) | `transformer_layer.py` / `mlp.py` / `attention.py` | RMSNorm → QKV → FlashAttn → O → (reshard) → RMSNorm → MLP → (reshard) |
| `moe_decoder` | `moe/{moe_layer,router,experts}.py` | router → dispatch(all-to-all,ep) → moe_gemm → combine(all-to-all) |
| `mla_decoder` | `multi_latent_attention.py` | 低秩压缩 + rope；KV 压缩维 |
| 实验性注意力 | `experimental_attention_variant/{csa,deepseek_v4_hybrid_attention,compressor,indexer}.py` | 各一张 op 图 |
| `mtp` | `multi_token_prediction.py` | 末层、嵌套 transformer_layer |
| embedding / 输出头 / loss | `transformer_block.py` / `loss/loss.py` | 非重复项 |

## 12. 分阶段实施

- **P0**：ModelSpec schema + `dense_decoder` op 图 + Op Roofline 库 + 图内切分(TP/CP) + 外层 FSDP/PP → **显存/OOM 预测器**。
- **P1**：时间模型(compute+comm+overlap+bubble) + recompute/swap 变换 + EP/MoE op 图 → 单步时间 + 瓶颈 + MFU。
- **P1.5**：what-if / 敏感性扫描，给专家候选配置排序。**（"专家配置 + 仿真分析"价值在此交付）**
- **P2**：把搜索器套在已验证的评估器外（从小子空间起步）。

## 13. 开放问题

1. **集合通信默认由 sharding mismatch 派生，少数特殊 case 允许显式声明**（已定方向）。
2. 硬件规格表 η 默认值来源（厂商手册 vs 经验值），逐 op 类填初值。
3. MoE 路由分布：**v1 默认 balanced**（tokens/expert = S·B·topk/n_experts，capacity_factor=1.0）；实测直方图 / 最坏负载因子作为可选输入。**(已定)**
4. ModelSpec 编写方式：手写 op 图 vs 从 `TransformerConfig` adapter 半自动 vs 将来 trace 辅助生成草稿。
5. 闭式 MFU 快速模式：**不纳入 v1（YAGNI）**；先把声明 op 图做准，闭式作为将来可选坍缩。**(已定)**

> P0 详细设计（ModelSpec 定义 + 静态/激活内存计算过程）见 [[2026-06-29-p0-modelspec-and-memory-design]]。
