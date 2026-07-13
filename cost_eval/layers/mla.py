"""M1：DeepSeek-V3 MLA (Multi-Latent Attention) decoder 组装。

MLA attention 段（10 op，融合路径 mla_qkv_concat=True）现居 :mod:`cost_eval.layers.attention`，
本模块从命名 op-builder 拼出 dense / moe 两种 decoder 组合（行为不变）：

  attn 段     → :mod:`cost_eval.layers.attention` ``build_mla_attn_ops``（10 op）
  dense FFN 段 → :mod:`cost_eval.layers.ffn`       ``build_dense_ffn_ops``（5 op）
  MoE FFN 段   → :mod:`cost_eval.layers.ffn`       ``build_moe_ffn_ops``（6 op）
  shared 段    → :mod:`cost_eval.layers.ffn`       ``build_shared_expert_ops``（3 op）

TP 切分惯例：
  - 列并行（输出 dim 1 或末维 shard tp）：linear_qkv / linear_qb / linear_kvb
  - 行并行（输出 partial=tp）：o_proj / shared_fc2
  - split 是隐含算子（不单列 op），q_a_in / kv_a_in 建模为独立 TensorRef
"""
from __future__ import annotations

from ..model_spec import DimTable, LayerSpec
from .attention import build_mla_attn_ops
from .ffn import (build_dense_ffn_ops, build_moe_ffn_ops, build_shared_expert_ops,
                  build_moe_merge_op)

# 向后兼容：旧代码从 mla 导入 MLA 符号维度别名 / attn builder
from .attention import (  # noqa: F401
    QKV_PROJ, QB_OUT, KVB_OUT, ATTN_OUT, build_mla_attn_ops,
)

__all__ = [
    "build_mla_attn_ops",
    "build_mla_dense_decoder",
    "build_mla_moe_decoder",
]


def build_mla_dense_decoder(d: DimTable) -> LayerSpec:
    """MLA attention + dense MLP FFN 组合的 decoder 层（15 op）。

    attn 段：``build_mla_attn_ops(d)``（10 op）
    FFN 段：``build_dense_ffn_ops(d)``（5 op：ln2/fc1/swiglu/fc2/add2）
    """
    return LayerSpec(ops=build_mla_attn_ops(d) + build_dense_ffn_ops(d))


def build_mla_moe_decoder(d: DimTable) -> LayerSpec:
    """MLA attention + MoE FFN + Shared Expert 组合的 decoder 层（19 op）。

    attn 段：``build_mla_attn_ops(d)``（10 op）
    MoE FFN 段：``build_moe_ffn_ops(d)``（6 op：router/dispatch/e_fc1/e_swiglu/e_fc2/combine）
    Shared expert 段：``build_shared_expert_ops(d)``（3 op：shared_fc1/shared_swiglu/shared_fc2），
        吃 h1，intermediate=moe_shared_F，纯 tp 切、不 ep。

    Shared expert 输出名 ``sh_o`` 与 MoE combine 输出 ``comb`` 不同名，避免冲突。
    尾接 ``moe_add``（2026-07-11 补边）:routed(comb)+shared(sh_o) 合流 → h2（moe_layer 真实语义,
    线性 add saves=[] 零字节;修 op 图孤立叶节点,与 build_llm 装配保持逐字段一致）。
    """
    return LayerSpec(
        ops=build_mla_attn_ops(d) + build_moe_ffn_ops(d) + build_shared_expert_ops(d)
        + [build_moe_merge_op(d)]
    )
