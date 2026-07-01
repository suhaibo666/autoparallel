"""op-builder 注册表骨架（设计 §3/§6：三派发轴的 str→builder 映射）。

- ``ATTN_REGISTRY``：注意力变体 → attn op-builder。``mha`` 复用 ``gqa`` builder
  （对称 GQA，num_query_groups=n_heads）。``dsv4_hybrid`` 于 Phase 2 补（Task 2.1）。
- ``FFN_REGISTRY``：FFN 变体 → ffn op-builder。``moe`` 的 shared expert 由装配器
  另调 ``build_shared_expert_ops`` 并入层尾（非独立 FFN 变体）。

装配器 `build_llm_spec` 用 ``ATTN_REGISTRY[cfg.attn_type]`` / ``FFN_REGISTRY[dense|moe]``
取 builder 组合每层 body。
"""
from __future__ import annotations

from .attention import build_gqa_attn_ops, build_mla_attn_ops
from .dsv4_hybrid import build_dsv4_hybrid_attn_ops
from .ffn import build_dense_ffn_ops, build_moe_ffn_ops


def _dsv4_hybrid_entry(d, compress_ratio=None):
    """dsv4_hybrid 注册入口：需 per-layer ``compress_ratio``。

    与其它 attn builder（单参 ``(d)``）不同，dsv4_hybrid 每层还按 compress_ratio
    内部分支。装配器 wiring 由 Task 2.4 负责（届时按 csa_compress_ratios[layer] 传入
    ratio 调 ``build_dsv4_hybrid_attn_ops(d, ratio)``）。在 wiring 就绪前，用单参形式
    误调本入口会抛清晰的 NotImplementedError。
    """
    if compress_ratio is None:
        raise NotImplementedError(
            "dsv4_hybrid 需 per-layer compress_ratio；请直接调用 "
            "build_dsv4_hybrid_attn_ops(d, ratio)（装配器 wiring 见 Task 2.4）。"
        )
    return build_dsv4_hybrid_attn_ops(d, compress_ratio)


# ① 注意力派发轴。mha == 对称 GQA（KV 组数 = n_heads），复用同一 builder。
# dsv4_hybrid（DSA/CSA/HCA）需 compress_ratio → 注册为 _dsv4_hybrid_entry：
# 单参误调抛 NotImplementedError，双参 (d, ratio) 委托 build_dsv4_hybrid_attn_ops。
ATTN_REGISTRY = {
    "mha": build_gqa_attn_ops,
    "gqa": build_gqa_attn_ops,
    "mla": build_mla_attn_ops,
    "dsv4_hybrid": _dsv4_hybrid_entry,
}

# ② FFN 派发轴。
FFN_REGISTRY = {
    "dense": build_dense_ffn_ops,
    "moe": build_moe_ffn_ops,
}
