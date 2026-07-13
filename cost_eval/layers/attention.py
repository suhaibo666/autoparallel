"""M1：命名 attention op-builder（GQA / MLA）。

从 dense.py / mla.py 抽出的可组合 attention 段构件，供 dense/moe/mla decoder
组装器复用（行为不变，op 定义逐字段与旧切片写法一致）。

- ``build_gqa_attn_ops`` = 旧 ``build_dense_decoder(d).ops[:6]``
  （ln1 → qkv → rope → flash → o_proj → add1，对称 GQA）。
- ``build_mla_attn_ops`` = DeepSeek-V3 MLA 融合路径 10 op（原在 mla.py）。
"""
from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef

# ── GQA 符号维度别名（与 DimTable 字段名一致，eval_expr 可求值）──────────────
QKV = "(n_heads+2*n_kv)*head_dim"   # qkv 投影输出维：(H + 2·n_kv)·d_h，对称 GQA
NHD = "n_heads*head_dim"            # o_proj 输入维 = n_heads·head_dim

# ── FlashAttention workspace（机理公式，∝ S·n_heads —— 从 framework_reserve 拆出）──────
# Ascend `FlashAttentionScore`（`flash_attention.py:136-196` MindSpore 算子）除 attention_out
# 外还返回 **softmax_max + softmax_sum**（softmax LSE 统计），每个 shape `[B, n_heads, S, 8]`
# fp32（末维 8 = flash 内层 reduce 分块，CANN 固定），供 `FlashAttentionScoreGrad` 反向复用。
# 工作集 = 2 张量 × 8 × 4B = **64·B·n_heads·S** bytes。**∝ S·n_heads**（随序列缩放——正是要点），
# 取代此前 `S·B·n_heads·head_dim`（把整份 Q/O numel 当字节的经验近似）。block scratch（O(block·d)）
# 二阶、近常数，并入该文档，不单列。仅在该 flash op 的**前向事件**计入（非全局常数、非 loss 峰）。
FLASH_LSE_WS = "64*B*n_heads*S"

# ── MLA 符号维度表达式（与 DimTable 字段名一致，供 eval_expr 求值）──────────
# linear_qkv 输出维（q_lora+kv_lora+k_pe）
QKV_PROJ = "q_lora_rank+kv_lora_rank+qk_rope_head_dim"
# linear_qb 输出维（每头 nope+rope，合并后 n_heads 头）
QB_OUT   = "n_heads*(qk_nope_head_dim+qk_rope_head_dim)"
# linear_kvb 输出维（每头 nope+v，合并后 n_heads 头）
KVB_OUT  = "n_heads*(qk_nope_head_dim+v_head_dim)"
# flash_attn 输出维（n_heads 头，每头 v_head_dim）
ATTN_OUT = "n_heads*v_head_dim"


def build_gqa_attn_ops(d: DimTable) -> list:
    """构造对称 GQA attention 段的 op 列表（6 个 OpSpec）。

    op 序列：ln1 → qkv → rope → flash_attn → o_proj → add1
    权重按 tp 轴切分；flash_attn saves=[qkv, attn, lse]（backward 所需）。
    最后一个 op 输出为 ``h1``（shard={0:'sp'}），可与任意 FFN 尾拼接。

    参数
    ----
    d : DimTable
        模型架构超参（H/n_heads/n_kv/head_dim/S/B …）。
    """
    # ── 激活张量 ───────────────────────────────────────────────────────────────
    # 注：`qkv` 是 Q/K/V **融合**张量（末维 (n_heads+2·n_kv)·d_h），KV 分量不可无损分割 → **不**标
    # cp_kv（D-1 修正）。故 colossal 下 fused-qkv 的 KV 分量仍随 body ÷cp（小幅欠模 colossal 的 KV
    # all-gather buffer）；全重算下该量 off loss 峰、本栈跑不了 cp+无重算故未真机验证——已 caveat。
    x      = TensorRef("x",    ("S", "B", "H"),        shard={0: "sp"})
    ln1    = TensorRef("ln1",  ("S", "B", "H"))
    qkv    = TensorRef("qkv",  ("S", "B", QKV),        shard={2: "tp"})
    attn   = TensorRef("attn", ("S", "B", NHD),        shard={2: "tp"})
    lse    = TensorRef("lse",  ("S", "B", "n_heads"),  shard={2: "tp"})
    o      = TensorRef("o",    ("S", "B", "H"),        partial="tp")
    h1     = TensorRef("h1",   ("S", "B", "H"),        shard={0: "sp"})

    # ── 权重张量（is_weight=True，标注 tp 切分）──────────────────────────────
    qkv_w  = TensorRef("qkv_w", ("H", QKV),   shard={1: "tp"}, is_weight=True)
    o_w    = TensorRef("o_w",   (NHD, "H"),   shard={0: "tp"}, is_weight=True)

    # flash attention workspace = softmax LSE 机理公式（∝ S·n_heads，见 FLASH_LSE_WS）
    fa_ws = FLASH_LSE_WS

    return [
        # 1. Pre-norm（LayerNorm / RMSNorm）
        OpSpec("ln1",    OpType.NORM,        [x],          ln1,
               saves=[x]),
        # 2. QKV 投影
        OpSpec("qkv",    OpType.MATMUL,      [ln1, qkv_w], qkv,
               params=[qkv_w], saves=[ln1]),
        # 3. RoPE（in-place，输出复用 qkv 张量引用）
        OpSpec("rope",   OpType.ROPE,        [qkv],        qkv,
               saves=[]),
        # 4. FlashAttention（saves 存 q/k/v 与 softmax lse，backward 所需）
        OpSpec("flash",  OpType.FLASH_ATTN,  [qkv],        attn,
               saves=[qkv, attn, lse],
               workspace=fa_ws),
        # 5. Output 投影（列并行→行并行）
        OpSpec("o_proj", OpType.MATMUL,      [attn, o_w],  o,
               params=[o_w], saves=[attn]),
        # 6. Residual add（all-reduce 在此隐式完成）
        OpSpec("add1",   OpType.ELEMENTWISE, [o],          h1,
               saves=[]),
    ]


def build_mla_attn_ops(d: DimTable) -> list:
    """构造 MLA attention 段的 op 列表（10 个 OpSpec）。

    融合路径（mla_qkv_concat=True）op 序列：
      ln1 → linear_qkv → [split→] q_a_norm → kv_a_norm → linear_qb → linear_kvb
      → rope → flash_attn → o_proj → add1

    返回的 op 列表最后一个 op 输出为 ``h1``（shard={0:'sp'}），
    可与 dense/moe FFN 尾直接拼接。

    参数
    ----
    d : DimTable
        需包含 q_lora_rank / kv_lora_rank / qk_rope_head_dim /
        qk_nope_head_dim / v_head_dim 字段。
    """
    # ── 激活张量 ────────────────────────────────────────────────────────────
    x        = TensorRef("x",        ("S", "B", "H"),      shard={0: "sp"})
    ln1_out  = TensorRef("ln1",      ("S", "B", "H"))
    # linear_qkv 输出（列并行，末维 ÷ tp）
    qkv_out  = TensorRef("qkv_out",  ("S", "B", QKV_PROJ), shard={2: "tp"})
    # q_a 和 kv_a 切片（split 隐含，建模为无 shard 的新张量）。
    # KV 侧激活（kv_a_in/kv_a_out/kvb_out）标 cp_kv=True（D-1 修正）：colossal 下 KV all-gather
    # 到 full-S（不 ÷cp）；ulysses/ring/hybrid 仍随 body ÷cp。DSv3 走 cp=1 → 恒不生效，逐字节不变。
    q_a_in   = TensorRef("q_a_in",   ("S", "B", "q_lora_rank"))
    kv_a_in  = TensorRef("kv_a_in",  ("S", "B", "kv_lora_rank"), cp_kv=True)
    q_a_out  = TensorRef("q_a_out",  ("S", "B", "q_lora_rank"))
    kv_a_out = TensorRef("kv_a_out", ("S", "B", "kv_lora_rank"), cp_kv=True)
    # linear_qb / linear_kvb 输出
    qb_out   = TensorRef("qb_out",   ("S", "B", QB_OUT),   shard={2: "tp"})
    kvb_out  = TensorRef("kvb_out",  ("S", "B", KVB_OUT),  shard={2: "tp"}, cp_kv=True)
    # flash_attn 输出 + lse
    attn_out = TensorRef("attn",     ("S", "B", ATTN_OUT), shard={2: "tp"})
    lse      = TensorRef("lse",      ("S", "B", "n_heads"), shard={2: "tp"})
    # o_proj 输出（行并行，待 all-reduce / reduce-scatter）
    o        = TensorRef("o",        ("S", "B", "H"),       partial="tp")
    # add1 输出（reshard 后回到 SP 分布）
    h1       = TensorRef("h1",       ("S", "B", "H"),       shard={0: "sp"})

    # ── 权重张量 ─────────────────────────────────────────────────────────────
    qkv_w  = TensorRef("qkv_w",  ("H",           QKV_PROJ), shard={1: "tp"}, is_weight=True)
    qb_w   = TensorRef("qb_w",   ("q_lora_rank", QB_OUT),   shard={1: "tp"}, is_weight=True)
    kvb_w  = TensorRef("kvb_w",  ("kv_lora_rank", KVB_OUT), shard={1: "tp"}, is_weight=True)
    o_w    = TensorRef("o_w",    (ATTN_OUT, "H"),            shard={0: "tp"}, is_weight=True)

    fa_ws = FLASH_LSE_WS   # softmax LSE 机理公式（∝ S·n_heads），MLA 与 GQA 同（不依赖 head_dim）

    return [
        # 1. Pre-norm
        OpSpec("ln1",        OpType.NORM,        [x],               ln1_out,
               saves=[x]),
        # 2. linear_qkv（列并行：H → q_lora+kv_lora+k_pe）
        OpSpec("linear_qkv", OpType.MATMUL,      [ln1_out, qkv_w],  qkv_out,
               params=[qkv_w], saves=[ln1_out]),
        # 3. q_a LayerNorm（在 q_lora_rank 维上；输入为 qkv split 切片）
        #    inputs 含 qkv_out = 切片视图的**数据流依赖**（qkv_out→q_a_norm 边;字节仍按切片 q_a_in 计,
        #    saves 不变——此前名字断链致 op 图出现孤立叶节点）。
        OpSpec("q_a_norm",   OpType.NORM,        [q_a_in, qkv_out], q_a_out,
               saves=[q_a_in]),
        # 4. kv_a LayerNorm（在 kv_lora_rank 维上）
        OpSpec("kv_a_norm",  OpType.NORM,        [kv_a_in, qkv_out], kv_a_out,
               saves=[kv_a_in]),
        # 5. linear_qb（列并行：q_lora → n_heads*(nope+rope)）
        OpSpec("linear_qb",  OpType.MATMUL,      [q_a_out, qb_w],   qb_out,
               params=[qb_w], saves=[q_a_out]),
        # 6. linear_kvb（列并行：kv_lora → n_heads*(nope+v)）
        OpSpec("linear_kvb", OpType.MATMUL,      [kv_a_out, kvb_w], kvb_out,
               params=[kvb_w], saves=[kv_a_out]),
        # 7. RoPE（作用于 qb_out 的 rope 部分，in-place，复用同名引用）
        OpSpec("rope",       OpType.ROPE,        [qb_out],          qb_out,
               saves=[]),
        # 8. FlashAttention（q=qb_out, kv=kvb_out；输出 attn + lse）
        OpSpec("flash",      OpType.FLASH_ATTN,  [qb_out, kvb_out], attn_out,
               saves=[qb_out, kvb_out, attn_out, lse],
               workspace=fa_ws),
        # 9. o_proj（行并行：n_heads*v_head_dim → H，partial=tp）
        OpSpec("o_proj",     OpType.MATMUL,      [attn_out, o_w],   o,
               params=[o_w], saves=[attn_out]),
        # 10. Residual add（all-reduce/reduce-scatter 隐含，输出 h1）
        OpSpec("add1",       OpType.ELEMENTWISE, [o],               h1,
               saves=[]),
    ]
