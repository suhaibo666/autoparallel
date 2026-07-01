"""Task 2.1：DeepSeek-V4 hybrid 压缩注意力（DSA/CSA/HCA）op-builder。

`build_dsv4_hybrid_attn_ops(d, compress_ratio)` 按 **每层 compress_ratio** 分支
（设计 `specs/2026-07-01-unified-llm-modelspec-design.md` §7.3），忠实映射
mindformers `pynative/.../experimental_attention_variant/` 源码：

  - `deepseek_v4_hybrid_attention.py` — DSv4HybridSelfAttention（顶层 Q/KV/RoPE/分组输出）
  - `indexer.py`                      — CSAIndexer（top-k 选择，产 index_scores O(S²)）
  - `compressor.py`                   — Compressor（门控池化，产 compressed_kv S/ratio）
  - `csa.py`                          — CompressedSparseAttention（gather + 稀疏 attn）

| compress_ratio | 模式 | 分支结构 |
|---|---|---|
| 0 / 1 | 滑窗 | 只 sliding-window 稀疏注意力 → 内存等价 MLA base（§7.4 内存中性），
|       |      | 直接复用 `build_mla_attn_ops`。|
| 4     | CSA  | MLA base + **indexer**（index_scores O(S²)）+ 压缩器(overlap, coff=2)
|       |      | + 稀疏注意力（kv_gathered O(S·topk)）+ 分组输出。|
| 128   | HCA  | MLA base + 压缩器(non-overlap, coff=1) + dense 压缩位（无 top-k indexer）
|       |      | + 稀疏注意力 + 分组输出。|

**内存大头**（评估器建模的重点）：
  - `index_scores [B,S,S] fp32` → O(S²)，建为 indexer op 的 `bwd_scratch="4*B*S*S"`
    （`indexer.py:227-236`，「可重算不 save」→ bwd 期临时物化，fp32=4B）。
  - `kv_gathered [B,S,topk,v_head_dim]` → O(S·topk)，建为 sparse_attn 的 saves
    （`csa.py:208`）。
  - `attn_weights [B,n_heads,S,topk]` → O(S·topk)，sparse_attn saves（`csa.py:237`）。
  - `compressed_kv [S//ratio, B, 1, v_head_dim]`（`compressor.py:221` unsqueeze 后）。

新增符号维（`DimTable` 字段，默认 0）：`dsa_indexer_n_heads/dsa_indexer_head_dim/
dsa_indexer_topk/o_groups/o_lora_rank/csa_window_size`。
"""
from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef
from .attention import build_mla_attn_ops

__all__ = ["build_dsv4_hybrid_attn_ops"]

# ── DSv4 符号维度表达式（与 DimTable 字段名一致，供 eval_expr 求值）──────────
# linear_q_up_proj 输出维：num_attention_heads * q_head_dim，其中 q_head_dim=v_head_dim
# （deepseek_v4_hybrid_attention.py:66 `self.q_head_dim = config.v_head_dim`；:113）。
Q_OUT = "n_heads*v_head_dim"
# 分组输出：linear_o_group_proj 输出 o_groups*o_lora_rank（:139-143 / :285）。
O_GROUP_OUT = "o_groups*o_lora_rank"
# linear_o_group_proj 每组 chunk = query_projection_size // o_groups（:139）。
O_CHUNK = "n_heads*v_head_dim//o_groups"
# 索引器 wq_b 输出维：index_n_heads*index_head_dim（indexer.py:112）。
IDX_QB_OUT = "dsa_indexer_n_heads*dsa_indexer_head_dim"


def build_dsv4_hybrid_attn_ops(d: DimTable, compress_ratio: int) -> list:
    """构造 DSv4 hybrid attention 段的 op 列表（按 compress_ratio 分支）。

    参数
    ----
    d : DimTable
        需含 MLA dims + dsv4 dims（dsa_indexer_* / o_groups / o_lora_rank）。
    compress_ratio : int
        本层压缩比：0/1=滑窗、4=CSA、128=HCA。

    返回的最后一个 op 输出为 ``h1``（shard={0:'sp'}），可与 FFN 尾拼接。
    """
    # ── ratio 0/1：滑窗分支 == MLA base（内存中性，§7.4）──────────────────────
    # DSv4HybridSelfAttention 顶层结构不变，但 core_attention 退化为纯滑窗稠密注意力，
    # flash 下 saves 仍是 Q/K/V/O + lse（[S,S] 分数从不物化），故内存等价 MLA base。
    if compress_ratio in (0, 1):
        return build_mla_attn_ops(d)

    # ratio ∈ {4, 128}：CSA / HCA
    coff = 2 if compress_ratio == 4 else 1               # compressor.py:89-90 overlap→coff=2
    enable_indexer = compress_ratio == 4                 # csa.py:324 仅 ratio==4 启 indexer
    CMP_PROJ_OUT = f"{coff}*v_head_dim"                  # compressor.py:95 proj_out = coff*head_dim
    S_DIV_R = f"S//{compress_ratio}"                     # compressor.py:199 n_compressed

    # ── 激活张量 ──────────────────────────────────────────────────────────────
    x           = TensorRef("x",            ("S", "B", "H"),        shard={0: "sp"})
    ln1         = TensorRef("ln1",          ("S", "B", "H"))
    q_compressed = TensorRef("q_compressed", ("S", "B", "q_lora_rank"))   # :234
    q_a_out     = TensorRef("q_a_out",      ("S", "B", "q_lora_rank"))    # :235
    q           = TensorRef("q",            ("S", "B", Q_OUT), shard={2: "tp"})  # :237 列并行
    kv          = TensorRef("kv",           ("S", "B", "v_head_dim"))     # :249 单共享头
    kv_a_out    = TensorRef("kv_a_out",     ("S", "B", "v_head_dim"))     # :250
    # 稀疏注意力大头激活（csa.py，naive path）
    kv_gathered = TensorRef("kv_gathered",  ("B", "S", "dsa_indexer_topk", "v_head_dim"))  # :208
    attn_weights = TensorRef("attn_weights", ("B", "n_heads", "S", "dsa_indexer_topk"))    # :237
    core_out    = TensorRef("core_out",     ("S", "B", Q_OUT))           # :247 [sq,b,n,vd]
    topk_indices = TensorRef("topk_indices", ("B", "S", "dsa_indexer_topk"), dtype_bytes=4)  # :241 int32
    compressed_kv = TensorRef("compressed_kv", (S_DIV_R, "B", "1", "v_head_dim"))  # :221
    o_group_out = TensorRef("o_group_out",  ("S", "B", O_GROUP_OUT))     # :285
    o           = TensorRef("o",            ("S", "B", "H"))             # :290
    h1          = TensorRef("h1",           ("S", "B", "H"), shard={0: "sp"})

    # ── 权重张量（is_weight=True）─────────────────────────────────────────────
    wq_down = TensorRef("wq_down", ("H", "q_lora_rank"),  is_weight=True)                     # :93-101
    wq_up   = TensorRef("wq_up",   ("q_lora_rank", Q_OUT), shard={1: "tp"}, is_weight=True)    # :110-118 列并行
    wkv     = TensorRef("wkv",     ("H", "v_head_dim"),    is_weight=True)                     # :120-128
    # 索引器权重（indexer.py）
    idx_wq_b  = TensorRef("idx_wq_b",  ("q_lora_rank", IDX_QB_OUT),          is_weight=True)   # indexer.py:109-117
    idx_wproj = TensorRef("idx_wproj", ("H", "dsa_indexer_n_heads"),         is_weight=True)   # indexer.py:126-134
    # 压缩器权重（compressor.py）；ape 为 fp32 可学习参数
    cmp_wkv   = TensorRef("cmp_wkv",   ("H", CMP_PROJ_OUT),                  is_weight=True)    # compressor.py:97-105
    cmp_wgate = TensorRef("cmp_wgate", ("H", CMP_PROJ_OUT),                  is_weight=True)    # compressor.py:107-115
    cmp_ape   = TensorRef("cmp_ape",   (str(compress_ratio), CMP_PROJ_OUT),  is_weight=True, dtype_bytes=4)  # compressor.py:117-120 fp32
    # 稀疏注意力 attn_sink（每头 fp32，csa.py:305）
    attn_sink = TensorRef("attn_sink", ("n_heads",), is_weight=True, dtype_bytes=4)            # csa.py:304-308
    # 分组输出权重
    wo_group = TensorRef("wo_group", (O_GROUP_OUT, O_CHUNK), is_weight=True)                   # :140-143 linear_o_group_proj
    o_w      = TensorRef("o_w",      (O_GROUP_OUT, "H"),     is_weight=True)                    # :145-153 linear_proj

    # ── base：Q 低秩 down→norm→up + 单头 KV down→norm + RoPE ─────────────────
    ops = [
        # 1. Pre-norm
        OpSpec("ln1",           OpType.NORM,   [x],             ln1, saves=[x]),
        # 2. linear_q_down_proj（H → q_lora_rank，:234）
        OpSpec("linear_q_down", OpType.MATMUL, [ln1, wq_down],  q_compressed,
               params=[wq_down], saves=[ln1]),
        # 3. q_layernorm（q_lora_rank 维，:235）
        OpSpec("q_a_norm",      OpType.NORM,   [q_compressed],  q_a_out, saves=[q_compressed]),
        # 4. linear_q_up_proj（q_lora → n_heads*v_head_dim，:237）
        OpSpec("linear_q_up",   OpType.MATMUL, [q_a_out, wq_up], q,
               params=[wq_up], saves=[q_a_out]),
        # 5. linear_kv_proj（H → v_head_dim，单共享头，:249）
        OpSpec("linear_kv",     OpType.MATMUL, [ln1, wkv],      kv,
               params=[wkv], saves=[ln1]),
        # 6. kv_layernorm（v_head_dim 维，:250）
        OpSpec("kv_a_norm",     OpType.NORM,   [kv],            kv_a_out, saves=[kv]),
        # 7. Main Q/K pe-lane RoPE（:256-261，in-place，复用 q 引用）
        OpSpec("rope",          OpType.ROPE,   [q],             q, saves=[]),
    ]

    # ── indexer（仅 CSA ratio==4）：产 index_scores O(S²) + topk_indices ─────────
    if enable_indexer:
        # inputs：x(ln1) 供 linear_weights_proj，qr(q_a_out) 供 linear_wq_b（indexer.py:151-197）。
        # index_scores [B,S,S] fp32 建为 bwd_scratch（可重算不 save，indexer.py:227-236）。
        ops.append(
            OpSpec("indexer", OpType.MATMUL, [ln1, q_a_out], topk_indices,
                   params=[idx_wq_b, idx_wproj], saves=[topk_indices],
                   bwd_scratch="4*B*S*S")
        )

    # ── compressor：门控池化 → compressed_kv [S//ratio,B,1,vd]（compressor.py）─────
    ops.append(
        OpSpec("compressor", OpType.MATMUL, [ln1], compressed_kv,
               params=[cmp_wkv, cmp_wgate, cmp_ape], saves=[compressed_kv])
    )

    # ── sparse attention：gather kv_gathered O(S·topk) + attn（csa.py naive path）──
    ops.append(
        OpSpec("sparse_attn", OpType.FLASH_ATTN, [q, kv_a_out, compressed_kv], core_out,
               params=[attn_sink], saves=[kv_gathered, attn_weights, core_out])
    )

    # ── 分组输出：linear_o_group_proj（bmm）→ linear_proj → 残差 ─────────────────
    ops += [
        # grouped wo_a（:274-287）
        OpSpec("o_group_proj", OpType.MATMUL, [core_out, wo_group], o_group_out,
               params=[wo_group], saves=[core_out]),
        # linear_proj（o_groups*o_lora → H，:290）
        OpSpec("o_proj",       OpType.MATMUL, [o_group_out, o_w], o,
               params=[o_w], saves=[o_group_out]),
        # Residual add（输出 h1，reshard 回 SP）
        OpSpec("add1",         OpType.ELEMENTWISE, [o], h1, saves=[]),
    ]
    return ops
