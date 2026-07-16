"""TransformerLayer 组装接口（2026-07-16，用户报告）。

**统一** `build_transformer_layer`：一个 decoder 层 =
    attn(含 ln1) → **ln2 前置归一(统一前插)** → dense|moe FFN(消费 ln2) → [moe: shared(消费 ln2) + merge]

修此前的结构 bug：dense 的 FFN builder(`build_dense_ffn_ops`)**内嵌** ln2，而 MoE 的 FFN builder
(`build_moe_ffn_ops`/`build_shared_expert_ops`)直接吃裸 `h1`、**无 ln2** → 每个 MoE 层漏建一份
`[S,B,H]` fp32-cast 常驻（norm_compute=fp32），MoE 模型在无重算/select 下系统性欠预测。把 ln2
hoist 到本层统一前插后：dense 全层拼接逐字节不变；MoE 层补上 ln2、routed+shared 都消费同一 `ln2`
（与真机 `hidden = residual + mlp(post_attention_layernorm(hidden))` 一致，dense/moe 对称）。
"""
from __future__ import annotations

from ..model_spec import DimTable
from .ffn import build_pre_ffn_norm_op, build_shared_expert_ops, build_moe_merge_op


def build_transformer_layer(dims: DimTable, attn_ops, ffn_ops, *,
                            is_moe: bool = False, has_shared: bool = False) -> list:
    """拼装一个 transformer decoder 层的 body op 列表（未套 mHC 残差包装）。

    参数
    ----
    dims       : DimTable —— 架构超参。
    attn_ops   : list[OpSpec] —— attn 段（含 ln1，产出残差输出 `h1`），由调用方从 registry/builder 取。
    ffn_ops    : list[OpSpec] —— dense 或 moe FFN 段（**消费 `ln2`**）。
    is_moe     : bool —— 是否 MoE FFN。
    has_shared : bool —— MoE 且有 shared expert（追加 shared 段 + `moe_add` 合流）。

    返回完整 body op 列表：`attn_ops + [ln2] + ffn_ops (+ shared + moe_add)`。ln2 由
    `build_pre_ffn_norm_op` 统一前插，dense/moe 对称——这是本次修复的核心。
    """
    ops = list(attn_ops)
    ops.append(build_pre_ffn_norm_op(dims))          # ln2：post_attention_layernorm，dense/moe 统一前置
    ops += list(ffn_ops)
    if is_moe and has_shared:
        ops += build_shared_expert_ops(dims)         # shared expert 消费同一 ln2
        ops.append(build_moe_merge_op(dims))         # routed(comb) + shared(sh_o) → h2 合流
    return ops
