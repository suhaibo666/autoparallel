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
| 0 / 1 | 滑窗 | 只 sliding-window 稠密注意力（**仍走 DSv4 顶层**：per-head fp32 Q-norm + 分组输出）。|
| 4     | CSA  | +**indexer**（index_scores O(S²)）+ 压缩器(overlap, coff=2) + 稀疏注意力 + 分组输出。|
| 128   | HCA  | + 压缩器(non-overlap, coff=1) + dense 压缩位（无 top-k indexer）+ 稀疏注意力 + 分组输出。|

**关键修正（2026-07-03，真机 Profiler 定位 925 MiB 欠计）**：DSv4HybridSelfAttention 顶层
**对所有 ratio（含 0/1）**都物化两个 fp32 张量到 loss 峰值，此前评估器漏建/错 dtype：
  1. **per-head Query RMSNorm** —— `deepseek_v4_hybrid_attention.py:239-245`
     `q = rms_norm(cast(q, fp32), q_rms_gamma)` → fp32 `q[S,B,n_heads*q_head_dim]`（真机 256 MiB/层）。
  2. **分组输出 bmm 的 fp32 输入** —— `:277-283` `bmm(cast(cg, fp32), cast(wo, fp32))`
     → fp32 `cg[S,B,n_heads*v_head_dim]`（真机 256 MiB/层）。
  ⇒ 故 ratio 0/1 **不再**退化复用 `build_mla_attn_ops`，DSv4 全 ratio 自建 op 图。

**融合 vs 非融合（`d.dsa_fused`，生产默认 True）**：`kv_gathered`/`attn_weights` 是 **unfused
小算子路径**（`csa.py:187` `unfused_compressed_sparse_attn`）才物化的中间量；**fused kernel**
`npu_sparse_attn_shared_kv` 走 scratch **不物化**（真机 15415 里查无此张量）。故 fused 时
sparse_attn **不 save** 它们（否则幻影多算 ~1312 MiB）。

**内存大头**（评估器建模的重点）：
  - `q_hnorm_fp32 [S,B,n_heads*v_head_dim] fp32`（per-head Q-norm，256 MiB/层，:239-245）。
  - `cg_fp32 [S,B,n_heads*v_head_dim] fp32`（分组输出 fp32 输入，256 MiB/层，:277-283）。
  - `index_scores [B,S,S] fp32` → O(S²)，indexer op 的 `bwd_scratch="4*B*S*S"`（indexer.py:227-236）。
  - `kv_gathered [B,S,topk,v_head_dim]` / `attn_weights [B,n_heads,S,topk]` —— **仅 unfused** save（csa.py:208/237）。
  - `compressed_kv [S//ratio, B, 1, v_head_dim]`（compressor.py:221）。

符号维（`DimTable` 字段）：`dsa_indexer_n_heads/dsa_indexer_head_dim/dsa_indexer_topk/
o_groups/o_lora_rank/csa_window_size`；`dsa_fused: bool`（融合开关，默认 True）。
"""
from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef
from .attention import FLASH_LSE_WS

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
        需含 MLA dims + dsv4 dims（dsa_indexer_* / o_groups / o_lora_rank）+ `dsa_fused`。
    compress_ratio : int
        本层压缩比：0/1=滑窗、4=CSA、128=HCA。

    返回的最后一个 op 输出为 ``h1``（shard={0:'sp'}），可与 FFN 尾拼接。
    """
    fused = getattr(d, "dsa_fused", True)          # 融合 DSA kernel（生产默认）：稀疏中间量不物化
    sparse = compress_ratio not in (0, 1)          # 4=CSA / 128=HCA 走稀疏；0/1 滑窗
    coff = 2 if compress_ratio == 4 else 1         # compressor.py:89-90 overlap→coff=2
    enable_indexer = compress_ratio == 4           # csa.py:324 仅 ratio==4 启 indexer
    CMP_PROJ_OUT = f"{coff}*v_head_dim"            # compressor.py:95 proj_out = coff*head_dim
    S_DIV_R = f"S//{compress_ratio}" if sparse else "S"   # compressor.py:199 n_compressed
    # 稀疏注意力 gather 的 KV 位置数（I3）：CSA→top-k；HCA→稠密 window+S//ratio（无 top-k）。
    TOPK_DIM = "dsa_indexer_topk" if enable_indexer else f"csa_window_size + {S_DIV_R}"

    # ── 激活张量 ──────────────────────────────────────────────────────────────
    x            = TensorRef("x",            ("S", "B", "H"),         shard={0: "sp"})
    ln1          = TensorRef("ln1",          ("S", "B", "H"))
    q_compressed = TensorRef("q_compressed", ("S", "B", "q_lora_rank"))          # :234
    q_a_out      = TensorRef("q_a_out",      ("S", "B", "q_lora_rank"))          # :235
    q            = TensorRef("q",            ("S", "B", Q_OUT), shard={2: "tp"})  # :237 列并行(bf16)
    # per-head Q RMSNorm 输出 fp32（:239-245）—— 真机 256 MiB/层，此前漏建
    q_hnorm      = TensorRef("q_hnorm_fp32", ("S", "B", Q_OUT), shard={2: "tp"}, dtype_bytes=4)
    # KV 侧激活（kv/kv_a_out）标 cp_kv=True（D-1 修正）：colossal 下 KV all-gather 到 full-S；
    # 其余 cp 算法随 body ÷cp。dsv4 走 cp=1（DSv4-align 锚点）→ 恒不生效，逐字节不变。
    kv           = TensorRef("kv",           ("S", "B", "v_head_dim"), cp_kv=True)  # :249 单共享头
    kv_a_out     = TensorRef("kv_a_out",     ("S", "B", "v_head_dim"), cp_kv=True)  # :250
    core_out     = TensorRef("core_out",     ("S", "B", Q_OUT))                  # :247 [sq,b,n,vd]
    # 分组输出 bmm 的 fp32 输入 cg（:277-283 cast(cg, fp32)）—— 真机 256 MiB/层，此前错 bf16
    cg_fp32      = TensorRef("cg_fp32",      ("S", "B", Q_OUT), dtype_bytes=4)
    o_group_out  = TensorRef("o_group_out",  ("S", "B", O_GROUP_OUT))            # :285
    o            = TensorRef("o",            ("S", "B", "H"))                    # :290
    h1           = TensorRef("h1",           ("S", "B", "H"), shard={0: "sp"})

    # ── 基础权重（所有 ratio 共有）─────────────────────────────────────────────
    wq_down  = TensorRef("wq_down",  ("H", "q_lora_rank"),   is_weight=True)                    # :93-101
    wq_up    = TensorRef("wq_up",    ("q_lora_rank", Q_OUT), shard={1: "tp"}, is_weight=True)   # :110-118 列并行
    wkv      = TensorRef("wkv",      ("H", "v_head_dim"),    is_weight=True)                    # :120-128
    wo_group = TensorRef("wo_group", (O_GROUP_OUT, O_CHUNK), is_weight=True)                    # :140-143 linear_o_group_proj
    o_w      = TensorRef("o_w",      (O_GROUP_OUT, "H"),     is_weight=True)                    # :145-153 linear_proj

    # ── base：Q 低秩 down→norm→up→**per-head fp32 norm** + 单头 KV down→norm + RoPE ──
    ops = [
        OpSpec("ln1",           OpType.NORM,   [x],              ln1, saves=[x]),                 # 1 Pre-norm
        OpSpec("linear_q_down", OpType.MATMUL, [ln1, wq_down],   q_compressed,                   # 2 :234
               params=[wq_down], saves=[ln1]),
        OpSpec("q_a_norm",      OpType.NORM,   [q_compressed],   q_a_out, saves=[q_compressed]), # 3 :235
        OpSpec("linear_q_up",   OpType.MATMUL, [q_a_out, wq_up], q,                              # 4 :237
               params=[wq_up], saves=[q_a_out]),
        # 5 per-head Query RMSNorm（:239-245）：rms_norm 反向需 bf16 输入 q（128 MiB/层，saved）；
        #   fp32 输出 q_hnorm 由下游 attention save（QK 反向需 Q，256 MiB/层）——两者共存至 loss 峰值。
        OpSpec("q_hnorm",       OpType.NORM,   [q],              q_hnorm, saves=[q]),
        OpSpec("linear_kv",     OpType.MATMUL, [ln1, wkv],       kv, params=[wkv], saves=[ln1]), # 6 :249
        OpSpec("kv_a_norm",     OpType.NORM,   [kv],             kv_a_out, saves=[kv]),          # 7 :250
        OpSpec("rope",          OpType.ROPE,   [q_hnorm],        q_hnorm, saves=[]),             # 8 :256-261 in-place
    ]

    # ── core attention ────────────────────────────────────────────────────────
    if sparse:
        # compressed_kv / kv_gathered 亦 KV 侧（cp_kv=True，D-1 修正）：colossal 下 all-gather 到 full-S。
        compressed_kv = TensorRef("compressed_kv", (S_DIV_R, "B", "1", "v_head_dim"), cp_kv=True)  # :221
        kv_gathered   = TensorRef("kv_gathered",  ("B", "S", TOPK_DIM, "v_head_dim"), cp_kv=True)  # csa.py:208
        attn_weights  = TensorRef("attn_weights", ("B", "n_heads", "S", TOPK_DIM))               # csa.py:237
        topk_indices  = TensorRef("topk_indices", ("B", "S", TOPK_DIM), dtype_bytes=4)           # :241 int32
        idx_wq_b  = TensorRef("idx_wq_b",  ("q_lora_rank", IDX_QB_OUT),         is_weight=True)   # indexer.py:109-117
        idx_wproj = TensorRef("idx_wproj", ("H", "dsa_indexer_n_heads"),        is_weight=True)   # indexer.py:126-134
        cmp_wkv   = TensorRef("cmp_wkv",   ("H", CMP_PROJ_OUT),                 is_weight=True)    # compressor.py:97-105
        cmp_wgate = TensorRef("cmp_wgate", ("H", CMP_PROJ_OUT),                 is_weight=True)    # compressor.py:107-115
        cmp_ape   = TensorRef("cmp_ape",   (str(compress_ratio), CMP_PROJ_OUT), is_weight=True, dtype_bytes=4)  # :117-120 fp32
        attn_sink = TensorRef("attn_sink", ("n_heads",), is_weight=True, dtype_bytes=4)          # csa.py:304-308
        # indexer（仅 CSA ratio==4）：index_scores [B,S,S] fp32 建为 bwd_scratch（可重算不 save，indexer.py:227-236）
        if enable_indexer:
            ops.append(OpSpec("indexer", OpType.MATMUL, [ln1, q_a_out], topk_indices,
                              params=[idx_wq_b, idx_wproj], saves=[topk_indices], bwd_scratch="4*B*S*S"))
        # compressor：门控池化 → compressed_kv [S//ratio,B,1,vd]（compressor.py）
        ops.append(OpSpec("compressor", OpType.MATMUL, [ln1], compressed_kv,
                          params=[cmp_wkv, cmp_wgate, cmp_ape], saves=[compressed_kv]))
        # sparse attention：save Q(=q_hnorm fp32)+O(core_out)；fused kernel 不物化 kv_gathered/
        # attn_weights（走 scratch）；unfused 才 save 它们。
        sparse_saves = ([q_hnorm, core_out] if fused
                        else [q_hnorm, kv_gathered, attn_weights, core_out])
        ops.append(OpSpec("sparse_attn", OpType.FLASH_ATTN, [q_hnorm, kv_a_out, compressed_kv], core_out,
                          params=[attn_sink], saves=sparse_saves, workspace=FLASH_LSE_WS))
    else:
        # 滑窗（ratio 0/1）：纯 flash（sliding-window），saves Q(=q_hnorm)/O(core_out)+lse（[S,S] 从不物化，§7.4）
        ops.append(OpSpec("core_attn", OpType.FLASH_ATTN, [q_hnorm, kv_a_out], core_out,
                          saves=[q_hnorm, core_out], workspace=FLASH_LSE_WS))

    # ── 分组输出：linear_o_group_proj（bmm，fp32 cg save）→ linear_proj → 残差 ────
    ops += [
        # grouped wo_a：bmm 对 core_out cast fp32（cg_fp32 saved，:274-287）—— 真机 256 MiB/层
        OpSpec("o_group_proj", OpType.MATMUL, [core_out, wo_group], o_group_out,
               params=[wo_group], saves=[cg_fp32]),
        OpSpec("o_proj",       OpType.MATMUL, [o_group_out, o_w], o, params=[o_w], saves=[o_group_out]),  # :290
        OpSpec("add1",         OpType.ELEMENTWISE, [o], h1, saves=[]),           # 残差 → h1（reshard SP）
    ]
    return ops
