"""M1：dense_decoder op 图（对照 transformer_layer.py / attention.py / mlp.py）。

op 序列：ln1 → qkv → rope → flash_attn → o_proj → add1 → ln2 → fc1 → swiglu → fc2 → add2
权重均按 tp 轴切分；flash_attn saves=[qkv, attn, lse]（backward 所需）。

组装从命名 op-builder 拼出（行为不变）：
  attn 段 → :mod:`cost_eval.layers.attention` ``build_gqa_attn_ops``（6 op）
  FFN  段 → :mod:`cost_eval.layers.ffn`       ``build_dense_ffn_ops``（5 op）
"""
from __future__ import annotations

from ..model_spec import DimTable, LayerSpec
from .attention import build_gqa_attn_ops
from .ffn import build_dense_ffn_ops

# 向后兼容：旧代码从 dense 导入符号维度别名
from .attention import QKV, NHD  # noqa: F401

__all__ = ["build_dense_decoder", "build_gqa_attn_ops", "build_dense_ffn_ops"]


def build_dense_decoder(d: DimTable) -> LayerSpec:
    """构造 dense Transformer decoder 层的声明式 op 图。

    参数
    ----
    d : DimTable
        模型架构超参（H/F/n_heads/n_kv/head_dim/S/B …）。

    返回
    ----
    LayerSpec
        包含 11 个 OpSpec 的层描述（6 GQA attn + 5 dense FFN），权重均已标注 tp 切分维。
    """
    from .transformer import build_transformer_layer
    return LayerSpec(ops=build_transformer_layer(
        d, build_gqa_attn_ops(d), build_dense_ffn_ops(d), is_moe=False))
