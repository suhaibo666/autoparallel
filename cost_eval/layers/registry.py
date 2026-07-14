"""op-builder 注册表（设计 §3/§6：三派发轴的 str→builder 映射）。

**统一 API（Task 2）**：所有注册入口签名一律 ``REGISTRY[type](dims, ctx)``，`ctx` 为
`LayerContext`（携带 per-layer `compress_ratio` 等结构化字段）：

- ``ATTN_REGISTRY``：注意力变体 → attn op-builder。``mha`` 复用 ``gqa`` builder（对称 GQA，
  num_query_groups=n_heads）。``gqa``/``mla`` 是把原 `(DimTable)`-only builder 适配的薄
  wrapper（**忽略 ctx**）；``dsv4_hybrid`` 从 ``ctx.compress_ratio`` 读 per-layer 压缩比
  （0/1=滑窗、4=CSA、128=HCA）——去掉了旧的「单参报错 / 双参 (d, ratio)」特例 wrapper，
  成为普通 (dims, ctx) 入口。
- ``FFN_REGISTRY``：FFN 变体 → ffn op-builder（``dense``/``moe`` 同样忽略 ctx 的薄 wrapper）。
  ``moe`` 的 shared expert 由装配器另调 ``build_shared_expert_ops`` 并入层尾（非独立 FFN 变体）。

底层 `(DimTable)`-only builder（`build_gqa_attn_ops` 等）与 `(dims, ratio)` builder
（`build_dsv4_hybrid_attn_ops`）仍可直接调用（wrapper 经 ``__wrapped__`` 暴露原函数），
供 validate_dsv3 / 单元测试直接使用。装配器 `build_llm_spec` 统一用
``ATTN_REGISTRY[ctx.attn_type](dims, ctx)`` / ``FFN_REGISTRY[ctx.ffn_type](dims, ctx)``。
"""
from __future__ import annotations

import functools

from .attention import build_gqa_attn_ops, build_mla_attn_ops
from .dsa import build_dsa_attn_ops
from .dsv4_hybrid import build_dsv4_hybrid_attn_ops
from .ffn import build_dense_ffn_ops, build_moe_ffn_ops


def _dims_only(builder):
    """把 `(DimTable)`-only builder 适配成统一 `(dims, ctx)` 注册入口（ctx 未用则忽略）。

    ``functools.wraps`` 令 wrapper 携带 ``__wrapped__``（指回原 builder），既保留可直接
    调用的底层实现，也让绑定关系可测（``ATTN_REGISTRY["gqa"].__wrapped__ is build_gqa_attn_ops``）。
    """
    @functools.wraps(builder)
    def _entry(dims, ctx=None):
        return builder(dims)
    return _entry


def _dsv4_hybrid_entry(dims, ctx=None):
    """dsv4_hybrid 注册入口（统一 (dims, ctx)）：从 ``ctx.compress_ratio`` 读 per-layer 压缩比。

    ctx 须为带 ``compress_ratio`` 的 `LayerContext`（0/1=滑窗、4=CSA、128=HCA）；缺失
    （单参调用 / 非 LayerContext / ratio=None）即抛清晰 NotImplementedError，不静默产错图。
    注意 ``compress_ratio==0`` 合法（滑窗==MLA base），故用 ``is None`` 判缺失而非真值判断。
    """
    ratio = getattr(ctx, "compress_ratio", None)
    if ratio is None:
        raise NotImplementedError(
            "dsv4_hybrid 需 per-layer compress_ratio：请传带 compress_ratio 的 LayerContext"
            "（如 gen_layer_pattern 产出的 ctx），而非单参 / 无 ratio 调用。")
    return build_dsv4_hybrid_attn_ops(dims, ratio)


# ① 注意力派发轴（统一 (dims, ctx)）。mha == 对称 GQA（KV 组数 = n_heads），复用同一 adapter 实例。
# dsa = DSv3.2/GLM-5 稀疏注意力（**预估计**，基于 training_graph 静态图 DSA 代码，见 layers/dsa.py）。
_gqa_entry = _dims_only(build_gqa_attn_ops)
ATTN_REGISTRY = {
    "mha": _gqa_entry,
    "gqa": _gqa_entry,
    "mla": _dims_only(build_mla_attn_ops),
    "dsa": _dims_only(build_dsa_attn_ops),
    "dsv4_hybrid": _dsv4_hybrid_entry,
}

# ② FFN 派发轴（统一 (dims, ctx)，ctx 忽略）。
FFN_REGISTRY = {
    "dense": _dims_only(build_dense_ffn_ops),
    "moe": _dims_only(build_moe_ffn_ops),
}
