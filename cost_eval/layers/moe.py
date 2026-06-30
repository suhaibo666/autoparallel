"""M1：moe_decoder op 图（对照 moe/{moe_layer,router,experts}.py）。

attn 段：复用 dense_decoder 前 6 个 op（ln1→qkv→rope→flash→o_proj→add1）。
FFN 段：router → dispatch(all-to-all) → moe_gemm fc1 → swiglu → moe_gemm fc2 → combine。

专家权重设计（纯 EP）
  expert_parallel.py:330 `weight:(Shard(0),)` ——专家按 ep 轴切分，不做 tp 切分。
  shard={0: "ep"}，不含 "tp"。
  T_local = S*B*topk//ep（balanced dispatch，capacity=1）。
"""
from __future__ import annotations

from ..model_spec import DimTable, LayerSpec, OpSpec, OpType, TensorRef
from .dense import build_dense_decoder

# 每卡 token 数（balanced dispatch 假设 capacity_factor=1）
TLOCAL = "S*B*topk//ep"


def build_moe_decoder(d: DimTable) -> LayerSpec:
    """构造 MoE Transformer decoder 层的声明式 op 图。

    attn 段与 dense 共享前 6 个 op；FFN 换为
    router → dispatch → moe_gemm(fc1) → swiglu → moe_gemm(fc2) → combine。

    参数
    ----
    d : DimTable
        模型架构超参，需包含 n_experts / topk / moe_F 字段。

    返回
    ----
    LayerSpec
        完整 MoE decoder 层（attn 段 6 op + FFN 段 6 op = 12 op）。
    """
    # ── 复用 dense attn 段（前 6 个 op：到 add1 含）────────────────────────
    attn_ops = build_dense_decoder(d).ops[:6]

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

    # ── 专家权重（纯 EP：dim 0 = 专家数//ep，不含 tp）──────────────────────
    # shape[0] = n_experts//ep 保证 eval_expr 可求值（整除 ep 轴）
    w1 = TensorRef("e_w1", ("n_experts//ep", "H",      "2*moe_F"), shard={0: "ep"}, is_weight=True)
    w2 = TensorRef("e_w2", ("n_experts//ep", "moe_F",  "H"),       shard={0: "ep"}, is_weight=True)

    ffn_ops = [
        # 1. Router（softmax + top-k 选择，输出 logits 存 backward 用）
        OpSpec("router",   OpType.MOE_ROUTER, [hin],       logits,
               saves=[logits]),
        # 2. Dispatch（all-to-all；把 token 路由到各 expert rank）
        OpSpec("dispatch", OpType.DISPATCH,   [hin],       disp,
               saves=[disp]),
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
        OpSpec("combine",  OpType.COMBINE,    [eo],        comb,
               saves=[comb]),
    ]

    return LayerSpec(ops=list(attn_ops) + ffn_ops)
