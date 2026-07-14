"""DSA（DeepSeek Sparse Attention，DSv3.2-Exp / GLM-5）attention op-builder。

> **预估计（pre-estimate）**：本 op 图基于 mindformers **training_graph 静态图** DSA 代码
> （`parallel_core/training_graph/transformer/dsa/{dsa_attention,dsa_indexer}.py` +
> `transformer/multi_latent_attention.py` 的 use_dsa 分支）忠实推导，**无真机锚点**；
> 待 pynative DSA 代码落地后需重校准（saves 口径 / fused kernel 物化行为可能不同）。

结构 = MLA 低秩投影 base + lightning indexer（top-k 选择）+ MQA-absorb 稀疏 flash：

1. **MLA base（q 侧同 MLA，kv 侧不升维）**：linear_qkv → q_a_norm/kv_a_norm → linear_qb
   → rope。与 `build_mla_attn_ops` 的关键差异：DSA 分支**没有 linear_kvb 前向展开**——
   `k_nope = value = compressed_kv_norm`（潜空间 latent，1 个 KV 头，MQA），见
   `multi_latent_attention.py:575-579`。
2. **MQA absorb**（`multi_latent_attention.py:243-254`）：`w_kvb` 权重 reshape/split 为
   q_absorb `[n, qk_nope, kv_lora]` 与 v_absorb `[n, kv_lora, v_head]`；
   `q_nope = bmm(q_nope, q_absorb)` → 每头升到 kv_lora_rank，再 cat q_rope →
   `q_cat [S,B,n·(kv_lora_rank+qk_rope)]`（**比 MLA 的 q 大**：每头 576 vs 192@DSv3 维度）。
   attention 输出后 `attn_lat = bmm(attn_lat, v_absorb)` 回 v_head_dim（:301-305）。
   w_kvb 权重仍存在（param 挂在 q_absorb op 上，只算一次）。
3. **indexer**（`dsa_indexer.py`，输入 x/q_a 已 stop_gradient，:236-237——权重梯度仍需
   保存各 linear 输入）：
     - linear_wq_b：q_lora_rank → n_idx·d_idx（:121-130，ColumnParallelLinear，
       gpt_layer_specs.py:72）；输入为 **q_a_norm 后**的 q_a_out（:543-544/:595）。
     - linear_wk：H → d_idx（:132-141）+ k_norm（LayerNorm，:143-150）。
     - linear_weights_proj：H → n_idx（:152-161）。
     - Hadamard 旋转（:37-61/:323-324）**不建 op**：常量矩阵（scipy hadamard，非 Parameter）
       的 bmm 对输入线性 → 反向无需保存输入，零 saves；rope（:290-295/:311-316）同 MLA
       惯例 in-place 并入投影段不单列。
     - **index_scores O(S²)**：dense warmup 路径（:206-239）物化 fp32 分数——head-sum 后
       `[B,S,1,S]` fp32 → `bwd_scratch="4*B*S*S"`（与 dsv4_hybrid indexer 同公式族）。
       **有意不取** head-sum 前的 `[B,S,n_idx,S]`（:217 bmm 逐头输出）：fused
       `lightning_indexer`（sparse 训练稳态，:257-265）完全不物化 O(S²)（输出仅
       `[T,1,topk]`），dense-warmup 的逐头 bmm 亦在 head-sum 后即可释放——取全量会把
       预估推到非稳态极端。
     - 输出：topk_indices int32 `[B,S,topk]` + index_scores fp32 `[B,S,topk]`
       （fused op `return_value=True` 双输出，:257-265 + infer_shape :199-204，
       二者存活到 indexer loss → 均 save）。
4. **sparse flash（fused sfa kernel）**（`dsa_attention.py:140-149`）：输入
   q(nope+rope)/k(latent+rope)/v(latent)/topk_indices——kernel 内部按 indices 稀疏访问
   **全量 KV**，**不物化 selected-KV gather**（与 dsv4 fused kernel 同行为）。saves 忠实按
   kernel 输入：q_cat 全量 + key_cat/kv_a_out（latent 全量，非 topk 子集）+ topk_indices
   + 输出 attn_lat + softmax_max/sum（2×fp32 `[B,1,S,n]`，:102/:108-118）。

**有意省略（预估计范围外，重校准时再议）**：
  - **indexer KL loss**（`dsa_indexer_loss.py:121`，`multi_latent_attention.py:317-319`）：
    静态图在 loss 内做全量 `bmm(q,k) [B,n_heads,S,S]` fp32 **瞬态**（sparse 模式下随后
    gather 到 topk）。该瞬态 ∝ n_heads·S² fp32（DSv3.2 满维 S=4096 时 ~8.6 GiB/层 ÷tp·cp），
    但评估器 workspace 表达式**不随 tp 切分**（只 ÷cp 一次）→ 直接建模会系统性高估 tp 并行
    下的真值；且 q/k 已 stop_gradient、pynative 实现大概率分块计算。故整个 idx_loss op 不建，
    留待 pynative 代码落地后按真实 kernel 行为补。
  - indexer 的 EOD mask 生成 / actual_seq_len 处理（TND 打包细节，字节量可忽略）。

符号维（`DimTable` 字段，已存在）：`dsa_indexer_n_heads / dsa_indexer_head_dim /
dsa_indexer_topk`；MLA dims（q_lora_rank/kv_lora_rank/qk_rope_head_dim/qk_nope_head_dim/
v_head_dim）。三个 indexer 维必须 >0（fail-loud，不静默产错图）。
"""
from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef
from .attention import FLASH_LSE_WS, QKV_PROJ, QB_OUT, ATTN_OUT

__all__ = ["build_dsa_attn_ops"]

# ── DSA 符号维度表达式（与 DimTable 字段名一致，供 eval_expr 求值）──────────────
# indexer wq_b 输出维：index_n_heads*index_head_dim（dsa_indexer.py:121-130）。
IDX_QB_OUT = "dsa_indexer_n_heads*dsa_indexer_head_dim"
# MQA absorb 后 q：每头 kv_lora_rank(nope) + qk_rope（multi_latent_attention.py:251-253）。
Q_ABS_OUT = "n_heads*(kv_lora_rank+qk_rope_head_dim)"
# key latent+rope（cat_k，multi_latent_attention.py:254；1 个 KV 头）。
KEY_CAT = "kv_lora_rank+qk_rope_head_dim"
# sfa 输出：每头 kv_lora_rank（= q_nope shape，dsa_attention.py:108-118 infer_shape）。
ATTN_LAT = "n_heads*kv_lora_rank"


def build_dsa_attn_ops(d: DimTable) -> list:
    """构造 DSA attention 段的 op 列表（**预估计**，基于 training_graph 静态图代码）。

    op 序列（17 个）：
      ln1 → linear_qkv → q_a_norm → kv_a_norm → linear_qb → rope
      → idx_q → idx_k → idx_k_norm → idx_weights → indexer(top-k)
      → q_absorb → kv_cat → sparse_flash → v_absorb → o_proj → add1

    最后一个 op 输出为 ``h1``（shard={0:'sp'}），可与任意 FFN 尾拼接。

    参数
    ----
    d : DimTable
        需含 MLA dims + `dsa_indexer_n_heads/dsa_indexer_head_dim/dsa_indexer_topk`（>0）。
    """
    if not (getattr(d, "dsa_indexer_n_heads", 0) > 0
            and getattr(d, "dsa_indexer_head_dim", 0) > 0
            and getattr(d, "dsa_indexer_topk", 0) > 0):
        raise ValueError(
            "dsa 需 dsa_indexer_n_heads/dsa_indexer_head_dim/dsa_indexer_topk 三维均 >0"
            "（DSv3.2-Exp: 64/128/2048，GLM-5: 32/128/2048）——缺维将静默产错图，故报错。")

    # ── 激活张量（MLA base 部分与 build_mla_attn_ops 同名同形）──────────────────
    x        = TensorRef("x",        ("S", "B", "H"),      shard={0: "sp"})
    ln1_out  = TensorRef("ln1",      ("S", "B", "H"))
    qkv_out  = TensorRef("qkv_out",  ("S", "B", QKV_PROJ), shard={2: "tp"})
    q_a_in   = TensorRef("q_a_in",   ("S", "B", "q_lora_rank"))
    kv_a_in  = TensorRef("kv_a_in",  ("S", "B", "kv_lora_rank"), cp_kv=True)
    q_a_out  = TensorRef("q_a_out",  ("S", "B", "q_lora_rank"))
    kv_a_out = TensorRef("kv_a_out", ("S", "B", "kv_lora_rank"), cp_kv=True)
    qb_out   = TensorRef("qb_out",   ("S", "B", QB_OUT),   shard={2: "tp"})
    # MQA absorb 后 q（cat nope(kv_lora)+rope，multi_latent_attention.py:251-253）
    q_cat    = TensorRef("q_cat",    ("S", "B", Q_ABS_OUT), shard={2: "tp"})
    # key = cat(kv latent, k_rope)（:254；1 个 KV 头 → 不随 tp 切；cp 语义同 MLA kv 侧）
    key_cat  = TensorRef("key_cat",  ("S", "B", KEY_CAT),  cp_kv=True)
    # sfa 输出（latent 空间，每头 kv_lora_rank）+ softmax_max/sum（2×fp32 [B,1,S,n]，:102）
    attn_lat = TensorRef("attn_lat", ("S", "B", ATTN_LAT), shard={2: "tp"})
    sfa_stats = TensorRef("sfa_stats", ("2", "S", "B", "n_heads"), shard={3: "tp"}, dtype_bytes=4)
    # v_absorb 后输出（回 v_head_dim，:301-305）
    attn_out = TensorRef("attn",     ("S", "B", ATTN_OUT), shard={2: "tp"})
    o        = TensorRef("o",        ("S", "B", "H"),      partial="tp")
    h1       = TensorRef("h1",       ("S", "B", "H"),      shard={0: "sp"})

    # ── indexer 激活（dsa_indexer.py）──────────────────────────────────────────
    idx_q    = TensorRef("idx_q",    ("S", "B", IDX_QB_OUT), shard={2: "tp"})   # :284-297
    idx_k    = TensorRef("idx_k",    ("S", "B", "dsa_indexer_head_dim"))        # :303
    idx_kn   = TensorRef("idx_k_normed", ("S", "B", "dsa_indexer_head_dim"))    # :304 k_norm
    idx_w    = TensorRef("idx_w",    ("S", "B", "dsa_indexer_n_heads"))         # :330
    # fused lightning_indexer 双输出（:257-265 return_value=True）：均存活到 indexer loss
    topk_indices = TensorRef("topk_indices", ("B", "S", "dsa_indexer_topk"), dtype_bytes=4)
    idx_scores   = TensorRef("index_scores", ("B", "S", "dsa_indexer_topk"), dtype_bytes=4)

    # ── 权重张量 ─────────────────────────────────────────────────────────────
    qkv_w   = TensorRef("qkv_w",  ("H",            QKV_PROJ),  shard={1: "tp"}, is_weight=True)
    qb_w    = TensorRef("qb_w",   ("q_lora_rank",  QB_OUT),    shard={1: "tp"}, is_weight=True)
    # w_kvb：DSA 下不做前向 kvb 展开，作为 absorb 权重参与两个 bmm（:246-250）——param 只挂
    # q_absorb op 一次（v_absorb 复用同一物理权重，不重复计）。
    kvb_w   = TensorRef("kvb_w",  ("kv_lora_rank", "n_heads*(qk_nope_head_dim+v_head_dim)"),
                        shard={1: "tp"}, is_weight=True)
    o_w     = TensorRef("o_w",    (ATTN_OUT, "H"),             shard={0: "tp"}, is_weight=True)
    idx_wq_b  = TensorRef("idx_wq_b",  ("q_lora_rank", IDX_QB_OUT), shard={1: "tp"},
                          is_weight=True)                                       # :121-130 列并行
    idx_wk    = TensorRef("idx_wk",    ("H", "dsa_indexer_head_dim"), is_weight=True)   # :132-141
    idx_wproj = TensorRef("idx_wproj", ("H", "dsa_indexer_n_heads"),  is_weight=True)   # :152-161

    return [
        # ── MLA base（同 build_mla_attn_ops 前 6 op；无 linear_kvb）───────────────
        OpSpec("ln1",        OpType.NORM,   [x],               ln1_out, saves=[x]),
        OpSpec("linear_qkv", OpType.MATMUL, [ln1_out, qkv_w],  qkv_out,
               params=[qkv_w], saves=[ln1_out]),
        OpSpec("q_a_norm",   OpType.NORM,   [q_a_in, qkv_out], q_a_out, saves=[q_a_in]),
        OpSpec("kv_a_norm",  OpType.NORM,   [kv_a_in, qkv_out], kv_a_out, saves=[kv_a_in]),
        OpSpec("linear_qb",  OpType.MATMUL, [q_a_out, qb_w],   qb_out,
               params=[qb_w], saves=[q_a_out]),
        OpSpec("rope",       OpType.ROPE,   [qb_out],          qb_out, saves=[]),
        # ── indexer（输入 stop_gradient，但权重梯度仍需保存 linear 输入；saves 同名去重）──
        OpSpec("idx_q",      OpType.MATMUL, [q_a_out, idx_wq_b], idx_q,
               params=[idx_wq_b], saves=[q_a_out]),               # dsa_indexer.py:284
        OpSpec("idx_k",      OpType.MATMUL, [ln1_out, idx_wk],  idx_k,
               params=[idx_wk], saves=[ln1_out]),                 # :303
        OpSpec("idx_k_norm", OpType.NORM,   [idx_k],            idx_kn, saves=[idx_k]),  # :304
        OpSpec("idx_weights", OpType.MATMUL, [ln1_out, idx_wproj], idx_w,
               params=[idx_wproj], saves=[ln1_out]),              # :330
        # top-k 选择：bwd_scratch = dense-warmup 的 head-sum 后 index_scores [B,S,S] fp32
        # （O(S²)，与 dsv4 indexer 同公式族；fused 稳态不物化——预估计取安全侧，见模块 docstring）
        OpSpec("indexer",    OpType.MATMUL, [idx_q, idx_kn, idx_w], topk_indices,
               saves=[topk_indices, idx_scores], bwd_scratch="4*B*S*S"),
        # ── MQA absorb + 稀疏 flash（fused sfa kernel）────────────────────────────
        OpSpec("q_absorb",   OpType.MATMUL, [qb_out, kvb_w],   q_cat,
               params=[kvb_w], saves=[qb_out]),                   # multi_latent_attention.py:251-253
        OpSpec("kv_cat",     OpType.ELEMENTWISE, [kv_a_out, qkv_out], key_cat, saves=[]),  # :254
        # saves 忠实按 fused kernel 输入（dsa_attention.py:140-149）：Q 全量 + latent KV 全量
        # （kernel 按 indices 稀疏访问，**不物化 selected-KV**）+ indices + 输出 + softmax 统计
        OpSpec("sparse_flash", OpType.FLASH_ATTN, [q_cat, key_cat, topk_indices], attn_lat,
               saves=[q_cat, key_cat, kv_a_out, topk_indices, attn_lat, sfa_stats],
               workspace=FLASH_LSE_WS),
        OpSpec("v_absorb",   OpType.MATMUL, [attn_lat],        attn_out,
               saves=[attn_lat]),                                 # :301-305（权重同 kvb_w，不重复计）
        OpSpec("o_proj",     OpType.MATMUL, [attn_out, o_w],   o,
               params=[o_w], saves=[attn_out]),                   # :306 linear_proj
        OpSpec("add1",       OpType.ELEMENTWISE, [o],          h1, saves=[]),
    ]
