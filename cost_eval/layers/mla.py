"""M1：DeepSeek-V3 MLA (Multi-Latent Attention) op 图。

融合路径（mla_qkv_concat=True）op 序列：
  ln1 → linear_qkv → [split→] q_a_norm → kv_a_norm → linear_qb → linear_kvb
  → rope → flash_attn → o_proj → add1

TP 切分惯例：
  - 列并行（输出 dim 1 或末维 shard tp）：linear_qkv / linear_qb / linear_kvb
  - 行并行（输出 partial=tp）：o_proj / shared_fc2
  - split 是隐含算子（不单列 op），q_a_in / kv_a_in 建模为独立 TensorRef
"""
from __future__ import annotations

from ..model_spec import DimTable, LayerSpec, OpSpec, OpType, TensorRef
from .dense import build_dense_decoder
from .moe import build_moe_decoder

# ── 符号维度表达式（与 DimTable 字段名一致，供 eval_expr 求值）────────────────
# linear_qkv 输出维（q_lora+kv_lora+k_pe）
QKV_PROJ = "q_lora_rank+kv_lora_rank+qk_rope_head_dim"
# linear_qb 输出维（每头 nope+rope，合并后 n_heads 头）
QB_OUT   = "n_heads*(qk_nope_head_dim+qk_rope_head_dim)"
# linear_kvb 输出维（每头 nope+v，合并后 n_heads 头）
KVB_OUT  = "n_heads*(qk_nope_head_dim+v_head_dim)"
# flash_attn 输出维（n_heads 头，每头 v_head_dim）
ATTN_OUT = "n_heads*v_head_dim"


def build_mla_attn_ops(d: DimTable) -> list:
    """构造 MLA attention 段的 op 列表（10 个 OpSpec）。

    返回的 op 列表最后一个 op 输出为 ``h1``（shard={0:'sp'}），
    可与 dense/moe FFN 尾（从 ops[6:]）直接拼接。

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
    # q_a 和 kv_a 切片（split 隐含，建模为无 shard 的新张量）
    q_a_in   = TensorRef("q_a_in",   ("S", "B", "q_lora_rank"))
    kv_a_in  = TensorRef("kv_a_in",  ("S", "B", "kv_lora_rank"))
    q_a_out  = TensorRef("q_a_out",  ("S", "B", "q_lora_rank"))
    kv_a_out = TensorRef("kv_a_out", ("S", "B", "kv_lora_rank"))
    # linear_qb / linear_kvb 输出
    qb_out   = TensorRef("qb_out",   ("S", "B", QB_OUT),   shard={2: "tp"})
    kvb_out  = TensorRef("kvb_out",  ("S", "B", KVB_OUT),  shard={2: "tp"})
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

    fa_ws = "S*B*n_heads*v_head_dim"

    return [
        # 1. Pre-norm
        OpSpec("ln1",        OpType.NORM,        [x],               ln1_out,
               saves=[x]),
        # 2. linear_qkv（列并行：H → q_lora+kv_lora+k_pe）
        OpSpec("linear_qkv", OpType.MATMUL,      [ln1_out, qkv_w],  qkv_out,
               params=[qkv_w], saves=[ln1_out]),
        # 3. q_a LayerNorm（在 q_lora_rank 维上；输入为 qkv split 切片）
        OpSpec("q_a_norm",   OpType.NORM,        [q_a_in],          q_a_out,
               saves=[q_a_in]),
        # 4. kv_a LayerNorm（在 kv_lora_rank 维上）
        OpSpec("kv_a_norm",  OpType.NORM,        [kv_a_in],         kv_a_out,
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


def build_mla_dense_decoder(d: DimTable) -> LayerSpec:
    """MLA attention + dense MLP FFN 组合的 decoder 层（15 op）。

    attn 段：``build_mla_attn_ops(d)``（10 op）
    FFN 段：``build_dense_decoder(d).ops[6:]``（5 op：ln2/fc1/swiglu/fc2/add2）
    """
    ffn_tail = build_dense_decoder(d).ops[6:]
    return LayerSpec(ops=build_mla_attn_ops(d) + list(ffn_tail))


def build_mla_moe_decoder(d: DimTable) -> LayerSpec:
    """MLA attention + MoE FFN + Shared Expert 组合的 decoder 层（19 op）。

    attn 段：``build_mla_attn_ops(d)``（10 op）
    MoE FFN 段：``build_moe_decoder(d).ops[6:]``（6 op：router/dispatch/e_fc1/e_swiglu/e_fc2/combine）
    Shared expert 段：3 op（shared_fc1/shared_swiglu/shared_fc2），吃 h1，
        intermediate=moe_shared_F，纯 tp 切、不 ep。

    Shared expert 输出名 ``sh_o`` 与 MoE combine 输出 ``comb`` 不同名，避免冲突。
    """
    # ── Shared expert 激活 / 权重 ─────────────────────────────────────────
    hin_sh     = TensorRef("h1",     ("S", "B", "H"),            shard={0: "sp"})
    sh_g       = TensorRef("sh_g",   ("S", "B", "2*moe_shared_F"), shard={2: "tp"})
    sh_act     = TensorRef("sh_act", ("S", "B", "moe_shared_F"),  shard={2: "tp"})
    sh_o       = TensorRef("sh_o",   ("S", "B", "H"),             partial="tp")
    sh_fc1_w   = TensorRef("sh_w1",  ("H",            "2*moe_shared_F"),
                            shard={1: "tp"}, is_weight=True)
    sh_fc2_w   = TensorRef("sh_w2",  ("moe_shared_F", "H"),
                            shard={0: "tp"}, is_weight=True)

    shared_ops = [
        # shared fc1（列并行：H → 2*moe_shared_F）
        OpSpec("shared_fc1",    OpType.MATMUL,      [hin_sh, sh_fc1_w], sh_g,
               params=[sh_fc1_w], saves=[hin_sh]),
        # shared SwiGLU
        OpSpec("shared_swiglu", OpType.ELEMENTWISE, [sh_g],             sh_act,
               saves=[sh_g]),
        # shared fc2（行并行：moe_shared_F → H，partial=tp）
        OpSpec("shared_fc2",    OpType.MATMUL,      [sh_act, sh_fc2_w], sh_o,
               params=[sh_fc2_w], saves=[sh_act]),
    ]

    moe_ffn_tail = list(build_moe_decoder(d).ops[6:])
    return LayerSpec(ops=build_mla_attn_ops(d) + moe_ffn_tail + shared_ops)
