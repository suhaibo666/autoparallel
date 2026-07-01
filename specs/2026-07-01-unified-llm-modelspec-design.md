# 统一 LLM ModelSpec 构建器 — 设计

> **目标**：把"每个模型手写一个 `build_*_spec`"改成"**一个 config 驱动的 `build_llm_spec(LLMConfig)`**"，
> 像 mindformers/Megatron 用一套 `TransformerConfig` 生成所有 LLM 结构一样，覆盖 Megatron/mindformers
> 主流 + DeepSeek-V4 前沿（DSv4 混合压缩注意力 DSA/CSA/HCA、mHC 残差）的**内存 op 图**。

关联：现有 [`build_dsv3_spec`](../validate_dsv3.py)（将退化为一个 preset）、op 图积木 `cost_eval/layers/{dense,moe,mla}.py`、
内存契约与仿真 [`2026-06-29-p0-modelspec-and-memory-design.md`](2026-06-29-p0-modelspec-and-memory-design.md)（§2 内存契约、§8 时间线）。

---

## 1. 动机与核心原则

- **现状**：`validate_dsv3.build_dsv3_spec(N)` 手工拼 `embedding + mla_dense + mla_moe×(N-1) + lm_head`；每加一个模型就手写一份。`layers/` 里的积木还用 `ops[:6]`/`ops[6:]` 切片硬拆 attn/ffn。
- **诉求**：一套 `LLMConfig` → `build_llm_spec` 生成任意 LLM 的 ModelSpec。

**核心原则：`LLMConfig` = Megatron/mindformers `TransformerConfig` 的「内存结构」子集，不是全量克隆。**
只收**改变 op 图（哪些 op/张量存在、其 shape/saves/params）**的字段。以下**不进** `LLMConfig`（对内存结构无影响，属数值/训练环）：
router score function、load-balancing type、aux/z-loss 系数、token-drop policy、router dtype、expert-bias、dropout、init method、layernorm epsilon、qk-clip 等。
并行（tp/pp/ep/dp/cp）、recompute、swap、dtype 策略、optimizer 由 `ParallelConfig`/`RecomputeSpec`/`SwapSpec`/`OptimizerSpec`/`HardwareSpec` 各自承担，**不进** `LLMConfig`（正交，见 §2）。

---

## 2. 范围与非目标

**In（本设计负责）**：模型**结构** op 图 —— 每层有哪些 op、其输入输出符号 shape、`params`（权重）、`saves`（反向激活）、`bwd_scratch`/`workspace`；层间 pattern（dense/MoE、每层 attn 变体）；embedding/head/loss/MTP/残差结构。产出 `ModelSpec`。

**Out（本设计不碰）**：
- 并行切分、recompute、swap、dtype、optimizer —— 已由现有配置对象承担（`build_llm_spec` 只吐结构，切分在 `ShapeEval`/`StaticMem` 按 `ParallelConfig` 施加）。
- 不改变内存结构的数值项（见 §1）。
- 时间/roofline（P1）。

---

## 3. 三条派发轴（关键架构）

调研 Megatron + mindformers 后，模型结构可正交分解为**三条派发轴**（此前只识别了两条）：

| 轴 | 变体 | op-builder | 派发粒度 |
|---|---|---|---|
| **① 注意力** | `mha` / `gqa` / `mla` / **`dsv4_hybrid`**(DSA·CSA·HCA) | `build_*_attn_ops(d, layer_ctx)` | 每层（dsv4_hybrid 还按 `compress_ratio[layer]` 内部分支） |
| **② FFN** | `dense`(SwiGLU/ungated) / `moe`(+shared) | `build_*_ffn_ops(d)` | 每层（`first_k_dense`/`moe_layer_freq` 决定） |
| **③ 残差（横切）** | `plain` / **`mhc`** | 残差包装器：hidden ×`num_residual_streams` + 每层 HC ops | 全模型统一 |

外加**装配件**：embedding（tie/untie）、lm_head+loss（logsoftmax+NLL / chunked / vocab-parallel-CE）、可选 MTP 头。

---

## 4. `LLMConfig` schema（内存结构子集）

```python
@dataclass(frozen=True)
class LLMConfig:
    # ---- 核心维度 ----
    num_layers: int
    hidden_size: int                       # H
    num_attention_heads: int               # n_heads
    vocab_size: int
    seq_length: int                        # S
    batch_size: int = 1                    # B（local）
    head_dim: int | None = None            # 默认 H // n_heads

    # ---- ① 注意力 ----
    attn_type: str = "gqa"                 # mha | gqa | mla | dsv4_hybrid
    num_query_groups: int | None = None    # GQA 的 KV 组数（None→=n_heads 即 MHA）
    # 滑窗注意力（SWA：Mistral/Qwen/Gemma-2）——对训练峰值**内存中性**（flash 下激活仍 O(S)，
    # 见 §7.4）；留字段以忠实表达模型 + 供 P1 时间/推理 KV-cache 用。
    window_size: int | None = None         # None=全注意力；int=滑窗宽度
    window_pattern: tuple | None = None     # 每层 0=全/1=SWA（Gemma-2 交错）；None=全层同 window_size
    # MLA / dsv4 专用
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0              # = qk_pos_emb_head_dim
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    # dsv4_hybrid（DSA 索引器 + 压缩器 + 稀疏注意力）专用
    csa_compress_ratios: tuple | None = None   # 每层 ∈ {0/1, 4=CSA, 128=HCA}
    csa_window_size: int = 128
    dsa_indexer_n_heads: int = 0
    dsa_indexer_head_dim: int = 0
    dsa_indexer_topk: int = 0
    o_groups: int = 0                      # 分组输出投影
    o_lora_rank: int = 0

    # ---- ② FFN / MoE ----
    ffn_hidden_size: int | None = None     # 默认 4*H
    gated_linear_unit: bool = True         # SwiGLU（vs ungated）
    num_moe_experts: int | None = None     # None→纯 dense
    moe_router_topk: int = 0
    moe_ffn_hidden_size: int | None = None # 专家 FFN 隐藏维
    moe_shared_expert_num: int = 0
    moe_shared_ffn_hidden_size: int = 0
    first_k_dense_replace: int | None = None  # 前 K 层 dense；或用下面的 freq
    moe_layer_freq: tuple | int | None = None # 每层 0=dense/1=MoE（覆盖 first_k）
    moe_capacity_factor: float = 1.0       # 影响 dispatched token 数（内存相关）

    # ---- 归一化 / 位置编码（结构相关部分）----
    normalization: str = "RMSNorm"         # RMSNorm | LayerNorm（op 数同，saves 略异）
    norm_placement: str = "pre"            # pre | post | sandwich（sandwich=多一个 norm）
    qk_layernorm: bool = False             # Q/K 上加 norm（多 2 个小 norm op）
    position_embedding_type: str = "rope"  # rope | learned_absolute(+表) | none

    # ---- 装配：embedding / head / loss ----
    tie_word_embeddings: bool = False      # tie→无独立 lm_head 权重
    loss_type: str = "logsoftmax_nll"      # logsoftmax_nll | chunked | vocab_parallel_ce
    chunk_loss_num: int = 0                # >1：分块 CE，降 loss 区峰值
    embedding_params_dtype_bytes: int = 4  # embedding/输出 fp32（数值稳定）

    # ---- ③ 残差变体（横切）----
    residual_variant: str = "plain"        # plain | mhc
    num_residual_streams: int = 1          # mhc：hidden ×n

    # ---- MTP ----
    mtp_num_layers: int = 0                # DeepSeek-V3/V4 多 token 预测头

    # ---- bias ----
    add_bias_linear: bool = False
    add_qkv_bias: bool = False

    compute_dtype_bytes: int = 2           # bf16
```

> **字段→内存桶映射**（评估器视角，指导取舍）：`num_*/hidden/ffn/vocab/lora/head_dim` → 全桶；`attn_type/csa_*/dsa_*` → attn 段 op 数 + act/saves（index_scores O(S²)、kv_gathered O(S·topk)）；`num_moe_experts/topk/shared/capacity` → 专家 persistent + dispatch/combine act；`moe_layer_freq/first_k_dense` → layer_pattern；`tie_word_embeddings` → head persistent；`loss_type/chunk_loss_num` → loss 区 `act_live`+`bwd_scratch`（§见 P0 §8.3）；`residual_variant/num_residual_streams` → `act_live` ×n；`mtp_num_layers` → 追加 MTP 头层。

---

## 5. `build_llm_spec(cfg) -> ModelSpec` 算法

```
1. dims = _to_dimtable(cfg)                     # LLMConfig → DimTable（现有结构）
2. layer_pattern = _gen_pattern(cfg):           # 生成层序列
     ["embedding"]
     + per-layer decoder key（由 attn_type × (dense/moe by moe_layer_freq/first_k)）
     + ["mtp"] * cfg.mtp_num_layers
     + ["lm_head"]
3. layer_specs = {}:                            # 每种出现的层 key 建一次 LayerSpec
     for each unique decoder key:
        attn_ops = ATTN_REGISTRY[cfg.attn_type](dims, layer_ctx)
        ffn_ops  = FFN_REGISTRY[dense|moe](dims)
        body     = norm+attn+residual + norm+ffn+residual
        if cfg.residual_variant == "mhc":
            body = MHC_WRAP(body, cfg.num_residual_streams)   # ×n hidden + HC ops
        layer_specs[key] = LayerSpec(body)
     layer_specs["embedding"] = _embedding(cfg)
     layer_specs["lm_head"]   = _head_and_loss(cfg)          # tie/loss_type/chunk 感知
     layer_specs["mtp"]       = _mtp_head(cfg)               # 若启用
4. return ModelSpec(name, dims, layer_pattern, layer_specs)
```

**per-layer 上下文 `layer_ctx`**：dsv4_hybrid 需要 `compress_ratio = cfg.csa_compress_ratios[layer_id]`；MoE 需要"本层是否 MoE"。因此层 key 需编码这些，如 `mla_moe`、`dsv4hyb_r4_moe`。`_gen_pattern` 负责把 `first_k_dense`/`moe_layer_freq`/`csa_compress_ratios` 展开成每层 key。

---

## 6. op-builder 注册表 + 现有积木重构

**重构**：现在 `dense.py` 把 attn+ffn 塞进一个 `LayerSpec`（11 op），`moe.py`/`mla.py` 用 `build_dense_decoder(d).ops[:6]`/`[6:]` **切片硬拆**——脆弱。改为**命名 op-builder**，装配器组合：

```
cost_eval/layers/
  attention.py : build_mha_attn_ops / build_gqa_attn_ops / build_mla_attn_ops / build_dsv4_hybrid_attn_ops
  ffn.py       : build_dense_ffn_ops / build_moe_ffn_ops / build_shared_expert_ops
  residual.py  : build_norm_op / apply_residual / mhc_wrap
  head.py      : build_embedding_ops / build_head_and_loss_ops / build_mtp_ops
  registry.py  : ATTN_REGISTRY / FFN_REGISTRY（str→builder）
```

`mla.py`/`dense.py`/`moe.py` 现有实现搬进新文件、去掉切片；行为等价（回归见 §12）。

---

## 7. 注意力变体 op 图（源码 grounded）

### 7.1 `mha`/`gqa`（已实现，`dense.py` attn 段 6 op）
linear_qkv（GQA：KV 用 `num_query_groups`）→ rope → flash_attn → linear_proj，pre-norm + 残差。GQA 由 `num_query_groups<n_heads` 触发，KV 投影维 = `num_query_groups·head_dim`。

### 7.2 `mla`（已实现，`mla.py` 10 op）
linear_q(_a/_b + q_a_ln) + linear_kv(_a/_b + kv_a_ln) + rope + flash_attn + linear_proj。维度：`q_lora_rank/kv_lora_rank/qk_rope/qk_nope/v_head_dim`。DeepSeek-V3 已真机验证（P0 §8.7）。

### 7.3 `dsv4_hybrid`（DSA·CSA·HCA，**新建**，源：`pynative/.../experimental_attention_variant/`）
MLA base 之上，**每层按 `compress_ratio` 分支**（`deepseek_v4_hybrid_attention.py:70`）：

| compress_ratio | 模式 | 分支结构 | rope |
|---|---|---|---|
| 0/1 | 滑窗 | 只 sliding-window（`window_size`）稀疏注意力 | 标准 rope |
| 4 | **CSA** | 压缩器(overlap, coff=2) + **DSA 索引器** top-k + 稀疏注意力 | YaRN(`csa_compress_rotary_base`) |
| 128 | **HCA** | 压缩器(non-overlap, coff=1) + dense 压缩位 | YaRN |

**新增 op（在 MLA base 上）**：
- **索引器**（`indexer.py`，仅 CSA）：`linear_wq_b`(q_lora→n_idx·d_idx)、`linear_weights_proj`(H→n_idx)、压缩器出 k_index；**`index_scores [B,S,S] fp32`（O(S²)，内存大头，可重算不 save）**、`topk_indices [B,S,topk] int32`。
- **压缩器**（`compressor.py`）：`linear_wkv`/`linear_wgate`(H→coff·vd)、`ape`(fp32) → gated pooling → `compressed_kv [S/r, B, 1, vd]`；pooling 中间量 `[S/r, r·coff, B, d]`。
- **稀疏注意力**（`csa.py`）：gather → **`kv_gathered [B,S,topk,vd]`（O(S·topk)，内存大头）**、`scores/attn_weights [B,h,S,topk]`。
- **分组输出**：`linear_o_group_proj`(o_groups×o_lora) + `linear_proj`。

**内存大头**（评估器须建）：`index_scores` O(S²)、`kv_gathered`/`attn_weights` O(S·topk)、`compressed_kv` O(S/r)。config：`csa_compress_ratios/csa_window_size/dsa_indexer_{n_heads,head_dim,topk}/o_groups/o_lora_rank`。
> 实现细节（CSA overlap vs HCA non-overlap 的精确 op 差异）在实施期按 `compressor.py:43` RFC §3.4.2.2 落到 op；本设计先定"一个 `build_dsv4_hybrid_attn_ops(d, ratio)` 按 ratio 内部分支 + 上述内存大头"。

### 7.4 SWA 滑窗注意力（Mistral/Qwen/Gemma-2）——对训练峰值**内存中性**
`mha/gqa/mla` 加 `window_size` 即 SWA。**关键：flash-attention 下 SWA 不改训练激活内存**——
saves 仍是 Q/K/V/O `[S,B,H]` + logsumexp `[S,n_heads]`，`[S,S]` 分数矩阵从不物化；SWA 只缩：
(a) flash **workspace**（block scratch，二阶，现归 framework_reserve）；(b) **推理 KV-cache**（推理，出范围）。
故 **SWA 层 op 图 ≡ 全注意力层 op 图**；`window_size`/`window_pattern` 仅作**忠实表达模型** + 供 P1 时间/推理用，
`build_llm_spec` 对其**不改 op 图**（Gemma-2 的 SWA/全交错也因此不影响每层内存结构）。
> 若将来要建 flash-workspace 随 window 的缩放（现为常数），需 window-varying 真机点标定（类比 P0 §8.7）；属 Tier-2。

---

## 8. FFN / MoE op 图（已实现，微调）
- **dense**：ln2 → fc1(gated=2×) → swiglu → fc2 → 残差（`dense.py` ffn 段 5 op）。`gated_linear_unit=False`→ ungated（fc1 不 2×）。
- **moe**：router → dispatch(all-to-all) → e_fc1/e_swiglu/e_fc2(grouped-GEMM) → combine（`moe.py` 6 op），专家权重 `is_expert`（ep 切）。`moe_capacity_factor` 缩放 dispatched token 数。
- **shared expert**：`build_shared_expert_ops`（DeepSeek 有），并入 MoE 层尾。

---

## 9. mHC 残差包装器（**新建**，源：`pynative/.../hyper_connection.py`, `transformer_block.py`）
`residual_variant="mhc"` 时：
- **block 入口** `expand`：hidden `[S,B,H] → [S,B,n·H]`（n=`num_residual_streams`），**全栈 `act_live` ×n**（残差承载张量）。block 出口 `collapse` 回 `[S,B,H]`。
- **每层 2 个 HyperConnection 模块**（attn 前、ffn 前）：RMSNorm(n·H, saved) + `mapping_proj [n·H, 3n+n²]` + sinkhorn → `h_res [S,B,n,n]`(saved) / `h_post` / `h_pre`；输出 cell 做 `h_res @ streams + h_post·sublayer_out`。
- **内存**：`act_live` 残差 **×n**；每层多 `h_res [S,B,n,n]` 等小 saves；**params 基本不增**。DeepSeek-V4 全层启用。

装配器把 `mhc_wrap` 施加到**每个 decoder 层**，并在 embedding 后 / lm_head 前插 expand/collapse。

---

## 10. loss 变体（内存相关，源：`pynative/loss/loss.py`）
- `logsoftmax_nll`（默认，已建模，P0 §8.3）：logits(bf16) + log_softmax(fp32,saved) + NLL 反向 `probs+grad_log_softmax = 2×(4·S·B·vocab)`（真机 ScatterAddExt 验证，P0 §8.9）。
- `chunked`（`chunk_loss_num>1`）：分块重算，**峰值 loss 区 ÷ chunk 数**（少物化多份满 vocab 张量）。评估器按 chunk 缩 `bwd_scratch`。
- `vocab_parallel_ce`：logits 按 vocab 切（tp/vocab-parallel），loss 区张量 ∝ 1/tp。
config：`loss_type`、`chunk_loss_num`。

---

## 11. Presets（config 工厂）

`cost_eval/presets.py`：返回 `LLMConfig` 的工厂，覆盖主流 + 前沿：

| preset | attn | ffn | 残差 | 备注 |
|---|---|---|---|---|
| `deepseek_v3(N)` | mla | moe(first_k_dense=1)+shared | plain | **= 现 build_dsv3_spec，必须复现锚点** |
| `deepseek_v4(N)` | dsv4_hybrid | moe+shared | **mhc** | csa_compress_ratios、MTP |
| `llama(N)` | gqa | dense-SwiGLU | plain | tie 可选 |
| `qwen2/3(N)` | gqa | dense 或 moe | plain | qk_layernorm(Qwen3) |
| `mixtral(N)` | gqa | moe | plain | 无 shared、无 first_k_dense |

`build_dsv3_spec(N)` → 薄封装 `build_llm_spec(deepseek_v3(N))`。

---

## 12. 验证与迁移（不许回归）

**铁律：`deepseek_v3` preset 必须复现已真机验证的锚点**（P0 §8.7/§8.9）：4L=12472.5、8L=13896.1。
- **回归测试**：`test_unified_builder.py` 断言 `Evaluator(build_llm_spec(deepseek_v3(4)),...).evaluate().per_stage[0].peak_bytes` == 现 `build_dsv3_spec(4)` 的峰值（逐桶相等）。8L 同。
- **结构等价**：断言 `build_llm_spec(deepseek_v3(N))` 的 layer_pattern / 每层 op 序列与现 DSv3 spec 一致。
- **param 守恒**：llama/qwen preset 的 Σparam 对已知模型参数量（如 Llama2-7B）互证（P0 §9.1）。
- **迁移**：现有 `validate_dsv3.py`/`analyze_matrix.py`/`timeline_probe.py` 改调 preset，输出数值不变。

---

## 13. 覆盖分层

**Tier-1（本设计即建 op 图）**：`mha/gqa/mla` × `dense/moe(+shared)` + 每层 pattern（first_k_dense/moe_layer_freq）+ norm(RMS/LN, pre/post/sandwich, qk-norm) + rope/learned/none + tie/untie + loss(logsoftmax_nll/chunked/vocab_parallel_ce) + bias + **SWA 滑窗（`window_size`/`window_pattern`，内存中性字段，§7.4）**。覆盖 Llama/Qwen/Mistral/Gemma-2/Mixtral/DeepSeek-V3。
**Tier-1+（本设计新增，因用户要求前沿）**：`dsv4_hybrid`（DSA/CSA/HCA，按 compress_ratio 分支）、`mhc` 残差、MTP 头。覆盖 DeepSeek-V4。
**Tier-2（仅留 config 字段 + 清晰报错，暂不建 op 图）**：gated_delta_net 线性注意力、yoco、MRoPE、group-limited routing 的内存细节、**flash-workspace 随 SWA window 的精细缩放**（现为常数，见 §7.4）。用到再建。

---

## 14. 非目标 / 已知取舍
- 数值项（router score/aux-loss/dropout/init/eps）不建模——不改内存结构。
- dsv4_hybrid 的 index_scores O(S²) 与 kv_gathered O(S·topk) 目前**无真机点**，属评估器外推（待 DSv4 真机验证，类比 P0 §8.7 逐桶验证原则）。
- mHC 的 `act_live ×n` 未真机验证。
- 精确 CSA-vs-HCA overlap 差异在实施期按源码落 op。
