"""M1：dense_decoder op 图（对照 transformer_layer.py / attention.py / mlp.py）。

op 序列：ln1 → qkv → rope → flash_attn → o_proj → add1 → ln2 → fc1 → swiglu → fc2 → add2
权重均按 tp 轴切分；flash_attn saves=[qkv, attn, lse]（backward 所需）。
"""
from __future__ import annotations

from ..model_spec import DimTable, LayerSpec, OpSpec, OpType, TensorRef

# 符号维度别名（保持与 DimTable 字段名一致，eval_expr 可求值）
QKV = "(n_heads+2*n_kv)*head_dim"   # qkv 投影输出维：(H + 2·n_kv)·d_h，对称 GQA
NHD = "n_heads*head_dim"            # o_proj 输入维 = n_heads·head_dim


def build_dense_decoder(d: DimTable) -> LayerSpec:
    """构造 dense Transformer decoder 层的声明式 op 图。

    参数
    ----
    d : DimTable
        模型架构超参（H/F/n_heads/n_kv/head_dim/S/B …）。

    返回
    ----
    LayerSpec
        包含 11 个 OpSpec 的层描述，权重均已标注 tp 切分维。
    """
    # ── 激活张量 ───────────────────────────────────────────────────────────────
    x      = TensorRef("x",    ("S", "B", "H"),        shard={0: "sp"})
    ln1    = TensorRef("ln1",  ("S", "B", "H"))
    qkv    = TensorRef("qkv",  ("S", "B", QKV),        shard={2: "tp"})
    attn   = TensorRef("attn", ("S", "B", NHD),        shard={2: "tp"})
    lse    = TensorRef("lse",  ("S", "B", "n_heads"),  shard={2: "tp"})
    o      = TensorRef("o",    ("S", "B", "H"),        partial="tp")
    h1     = TensorRef("h1",   ("S", "B", "H"),        shard={0: "sp"})
    ln2    = TensorRef("ln2",  ("S", "B", "H"))
    g      = TensorRef("g",    ("S", "B", "2*F"),      shard={2: "tp"})
    act    = TensorRef("act",  ("S", "B", "F"),        shard={2: "tp"})
    o2     = TensorRef("o2",   ("S", "B", "H"),        partial="tp")
    h2     = TensorRef("h2",   ("S", "B", "H"),        shard={0: "sp"})

    # ── 权重张量（is_weight=True，标注 tp 切分）──────────────────────────────
    qkv_w  = TensorRef("qkv_w", ("H", QKV),   shard={1: "tp"}, is_weight=True)
    o_w    = TensorRef("o_w",   (NHD, "H"),   shard={0: "tp"}, is_weight=True)
    fc1_w  = TensorRef("fc1_w", ("H", "2*F"), shard={1: "tp"}, is_weight=True)
    fc2_w  = TensorRef("fc2_w", ("F",  "H"),  shard={0: "tp"}, is_weight=True)

    # flash attention workspace（近似值，用于内存建模）
    fa_ws = "S*B*n_heads*head_dim"

    ops = [
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
        # 7. Pre-FFN norm
        OpSpec("ln2",    OpType.NORM,        [h1],         ln2,
               saves=[h1]),
        # 8. FFN gate 投影（SwiGLU 第一段，输出 2F）
        OpSpec("fc1",    OpType.MATMUL,      [ln2, fc1_w], g,
               params=[fc1_w], saves=[ln2]),
        # 9. SwiGLU 激活
        OpSpec("swiglu", OpType.ELEMENTWISE, [g],          act,
               saves=[g]),
        # 10. FFN down 投影（行并行）
        OpSpec("fc2",    OpType.MATMUL,      [act, fc2_w], o2,
               params=[fc2_w], saves=[act]),
        # 11. Residual add
        OpSpec("add2",   OpType.ELEMENTWISE, [o2],         h2,
               saves=[]),
    ]
    return LayerSpec(ops=ops)
