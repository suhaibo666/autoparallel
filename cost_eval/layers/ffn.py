"""M1：命名 FFN op-builder（dense SwiGLU / MoE / shared expert）。

从 dense.py / moe.py / mla.py 抽出的可组合 FFN 段构件（行为不变，op 定义逐字段
与旧切片写法一致）。

- ``build_dense_ffn_ops``    = 旧 ``build_dense_decoder(d).ops[6:]``
  （ln2 → fc1 → swiglu → fc2 → add2，dense SwiGLU MLP）。
- ``build_moe_ffn_ops``      = 旧 ``build_moe_decoder(d).ops[6:]``
  （router → dispatch → e_fc1 → e_swiglu → e_fc2 → combine，纯 EP）。
- ``build_shared_expert_ops``= 旧 mla ``build_mla_moe_decoder`` 里的 shared 3 op
  （shared_fc1 → shared_swiglu → shared_fc2，纯 tp 切、不 ep）。
"""
from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef

# 每卡 token 数（balanced dispatch，design §7：T_local = S·B·topk·C/ep）。
# capacity_factor=C 影响 dispatched token 数（内存相关）；ep 切分由 shard={0:"ep"} 在
# resolve_tensor 中处理，此处用含 C 的全量符号。C=1.0（DSv3）时退化为 S·B·topk（不变）。
TLOCAL = "S*B*topk*capacity_factor"

# MoE all-to-all **staging 缓冲**（∝ dispatched_tokens·H —— 从 framework_reserve 拆出）。
# 源：experts.py:103-146 `GroupedMLP.permute` 把 token 按专家排序进 `routed_input`
# `[S·B·topk, H]`——ep all-to-all **之前**的发送/置换 staging 缓冲（各卡对自己 S·B 个 token
# 展开 topk 份，**不按 ep 切**，故用全量 S·B·topk）；experts.py:149-173 `unpermute` 反向散射
# （combine staging）。收端 post-a2a token（`disp [TLOCAL,H]{ep}`）已建为 save；此处补的是
# **置换 staging**（compute dtype bf16=2B），dispatch/combine 期瞬时活着 → 建为该 op 的
# workspace（前向逐层临时，非全局常数、非 loss 峰）。= 2×S·B·topk·C·H bytes。
MOE_STAGING_WS = "2*S*B*topk*capacity_factor*H"


def build_dense_ffn_ops(d: DimTable) -> list:
    """构造 dense SwiGLU MLP FFN 段的 op 列表（5 个 OpSpec）。

    op 序列：ln2 → fc1 → swiglu → fc2 → add2
    输入激活为 attn 段输出的 ``h1``（shard={0:'sp'}）。

    参数
    ----
    d : DimTable
        模型架构超参（H/F/S/B …）。
    """
    # ── 激活张量 ───────────────────────────────────────────────────────────────
    h1     = TensorRef("h1",   ("S", "B", "H"),          shard={0: "sp"})
    ln2    = TensorRef("ln2",  ("S", "B", "H"))
    g      = TensorRef("g",    ("S", "B", "2*F"),        shard={2: "tp"})
    act    = TensorRef("act",  ("S", "B", "F"),          shard={2: "tp"})
    o2     = TensorRef("o2",   ("S", "B", "H"),          partial="tp")
    h2     = TensorRef("h2",   ("S", "B", "H"),          shard={0: "sp"})

    # ── 权重张量（is_weight=True，标注 tp 切分）──────────────────────────────
    fc1_w  = TensorRef("fc1_w", ("H", "2*F"), shard={1: "tp"}, is_weight=True)
    fc2_w  = TensorRef("fc2_w", ("F",  "H"),  shard={0: "tp"}, is_weight=True)

    return [
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


def build_moe_ffn_ops(d: DimTable) -> list:
    """构造 MoE FFN 段的 op 列表（6 个 OpSpec，纯 EP）。

    op 序列：router → dispatch(all-to-all) → e_fc1(moe_gemm) → e_swiglu
             → e_fc2(moe_gemm) → combine(all-to-all)
    输入激活为 attn 段输出的 ``h1``（shard={0:'sp'}）。

    专家权重设计（纯 EP）：expert_parallel.py:330 ``weight:(Shard(0),)`` ——
    专家按 ep 轴切分，不做 tp 切分。shard={0:"ep"}，不含 "tp"。
    T_local = S*B*topk//ep（balanced dispatch，capacity=1）。

    参数
    ----
    d : DimTable
        需包含 n_experts / topk / moe_F 字段。
    """
    # ── MoE FFN 激活张量 ────────────────────────────────────────────────────
    # add1 输出（h1）作为 router 与 dispatch 的输入
    hin  = TensorRef("h1",    ("S", "B", "H"),                  shard={0: "sp"})
    # router logits（全 token × 全专家，无切分）
    logits = TensorRef("logits", ("S", "B", "n_experts"))
    # dispatch 后 token 按 ep 分片（all-to-all）
    disp = TensorRef("disp",  (TLOCAL, "H"),                    shard={0: "ep"})
    # 专家 fc1 输出（SwiGLU gate+up 合并，2·moe_F）
    g    = TensorRef("e_g",   (TLOCAL, "2*moe_F"),              shard={0: "ep"})
    # SwiGLU 后激活（moe_F 维）
    act  = TensorRef("e_act", (TLOCAL, "moe_F"),                shard={0: "ep"})
    # 专家 fc2 输出（H 维，仍按 ep 分片）
    eo   = TensorRef("e_o",   (TLOCAL, "H"),                    shard={0: "ep"})
    # combine 输出（all-to-all 还原到原始 token 序列）
    comb = TensorRef("comb",  ("S", "B", "H"),                  shard={0: "sp"})

    # ── 专家权重（纯 EP：dim 0 按 ep 轴切分，不含 tp）─────────────────────
    # shape 用全量维度（n_experts），shard={0:"ep"} 在 resolve_tensor 中做整除
    w1 = TensorRef("e_w1", ("n_experts", "H",     "2*moe_F"), shard={0: "ep"}, is_weight=True)
    w2 = TensorRef("e_w2", ("n_experts", "moe_F", "H"),       shard={0: "ep"}, is_weight=True)

    return [
        # 1. Router（softmax + top-k 选择，输出 logits 存 backward 用）
        OpSpec("router",   OpType.MOE_ROUTER, [hin],       logits,
               saves=[logits]),
        # 2. Dispatch（all-to-all；把 token 路由到各 expert rank）
        #    workspace = 置换发送 staging 缓冲（experts.py permute → routed_input [S·B·topk,H]）
        OpSpec("dispatch", OpType.DISPATCH,   [hin],       disp,
               saves=[disp], workspace=MOE_STAGING_WS),
        # 3. 专家 fc1（grouped GEMM，按 ep 切分的专家矩阵）
        OpSpec("e_fc1",    OpType.MOE_GEMM,   [disp, w1], g,
               params=[w1], saves=[disp]),
        # 4. SwiGLU 激活
        OpSpec("e_swiglu", OpType.ELEMENTWISE, [g],        act,
               saves=[g]),
        # 5. 专家 fc2（grouped GEMM）
        OpSpec("e_fc2",    OpType.MOE_GEMM,   [act, w2],  eo,
               params=[w2], saves=[act]),
        # 6. Combine（all-to-all；把 expert 输出还原到 token 序列）
        #    workspace = 反向散射 staging 缓冲（experts.py unpermute，:149-173）
        OpSpec("combine",  OpType.COMBINE,    [eo],        comb,
               saves=[comb], workspace=MOE_STAGING_WS),
    ]


def build_shared_expert_ops(d: DimTable) -> list:
    """构造 MoE shared expert 段的 op 列表（3 个 OpSpec，纯 tp 切、不 ep）。

    op 序列：shared_fc1 → shared_swiglu → shared_fc2
    吃 attn 段输出的 ``h1``，intermediate=moe_shared_F。
    输出名 ``sh_o`` 与 MoE combine 输出 ``comb`` 不同名，避免冲突。

    参数
    ----
    d : DimTable
        需包含 moe_shared_F 字段。
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

    return [
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
