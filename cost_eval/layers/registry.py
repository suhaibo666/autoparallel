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
from .ffn import build_dense_ffn_ops, build_moe_ffn_ops

# ① 注意力派发轴。mha == 对称 GQA（KV 组数 = n_heads），复用同一 builder。
# dsv4_hybrid（DSA/CSA/HCA）于 Phase 2（Task 2.1）注册，此处暂缺 → build_llm_spec
# 用未实现 key 时抛 KeyError；Phase 2 会填入 build_dsv4_hybrid_attn_ops。
ATTN_REGISTRY = {
    "mha": build_gqa_attn_ops,
    "gqa": build_gqa_attn_ops,
    "mla": build_mla_attn_ops,
}

# ② FFN 派发轴。
FFN_REGISTRY = {
    "dense": build_dense_ffn_ops,
    "moe": build_moe_ffn_ops,
}
