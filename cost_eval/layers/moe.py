"""M1：moe_decoder op 图（对照 moe/{moe_layer,router,experts}.py）。

attn 段：复用 GQA attention 段 6 个 op（ln1→qkv→rope→flash→o_proj→add1）。
FFN 段：router → dispatch(all-to-all) → moe_gemm fc1 → swiglu → moe_gemm fc2 → combine。

专家权重设计（纯 EP）
  expert_parallel.py:330 `weight:(Shard(0),)` ——专家按 ep 轴切分，不做 tp 切分。
  shard={0: "ep"}，不含 "tp"。
  T_local = S*B*topk//ep（balanced dispatch，capacity=1）。

组装从命名 op-builder 拼出（行为不变）：
  attn 段 → :mod:`cost_eval.layers.attention` ``build_gqa_attn_ops``（6 op）
  FFN  段 → :mod:`cost_eval.layers.ffn`       ``build_moe_ffn_ops``（6 op）
"""
from __future__ import annotations

from ..model_spec import DimTable, LayerSpec
from .attention import build_gqa_attn_ops
from .ffn import build_moe_ffn_ops

# 向后兼容：旧代码从 moe 导入每卡 token 符号
from .ffn import TLOCAL  # noqa: F401

__all__ = ["build_moe_decoder", "build_moe_ffn_ops"]


def build_moe_decoder(d: DimTable) -> LayerSpec:
    """构造 MoE Transformer decoder 层的声明式 op 图。

    attn 段与 dense 共享前 6 个 GQA op；FFN 换为
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
    from .transformer import build_transformer_layer
    # GQA MoE：无 shared expert（has_shared=False）——统一经 build_transformer_layer 前插 ln2。
    return LayerSpec(ops=build_transformer_layer(
        d, build_gqa_attn_ops(d), build_moe_ffn_ops(d), is_moe=True, has_shared=False))
