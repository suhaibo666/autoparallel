"""统一 LLM ModelSpec 装配器（设计 `specs/2026-07-01-unified-llm-modelspec-design.md` §5）。

一个 config 驱动的 `build_llm_spec(LLMConfig)` 取代"每模型手写一个 build_*_spec"。
本文件 Phase 1（Tier-1）提供：

- `gen_layer_pattern(cfg)`：把 `LLMConfig` 展开成 **`list[LayerContext]`**（结构化逐层身份），
  序列 = `[embedding] + per-layer decoder ctx + [mtp]*mtp_num_layers + [lm_head]`。
- `build_llm_spec(cfg)`：以 `LayerContext`（hashable）为键去重相同层、组装 `LayerSpec`，再用
  `ctx.name` 作 ModelSpec 的字符串层名（输出标签，绝不反解析），返回 `ModelSpec`。

**铁律**：decoder body 由现有命名 op-builder 直接拼接（attn 段已内嵌 ln1/residual，
ffn 段已内嵌 ln2/residual），**不另加 norm op**——否则将偏离 `build_mla_dense_decoder`/
`build_mla_moe_decoder` 而破坏 DSv3 复现硬门（Task 1.3）。
"""
from __future__ import annotations

from .layer_context import LayerContext
from .llm_config import LLMConfig, to_dimtable
from .model_spec import DimTable, LayerSpec, ModelSpec
from .layers.registry import ATTN_REGISTRY, FFN_REGISTRY
from .layers.ffn import build_shared_expert_ops
from .layers.head import build_embedding_ops, build_head_and_loss_ops, build_mtp_ops
from .layers.residual import mhc_wrap, build_hc_expand_op, build_hc_collapse_op


def _check_implemented_dispatch(cfg: LLMConfig) -> None:
    """对**改变 op 图**但当前只建了单一取值的分派字段，非实现取值即显式报错（I2）。

    不静默按已实现取值继续（会产「貌似合理实则错误」的图）。仿 head.py:50（loss_type）。
    """
    if not cfg.gated_linear_unit:
        raise NotImplementedError(
            "gated_linear_unit=False（ungated FFN）暂未建 op 图：ffn.py 硬编码 2*F gated(SwiGLU)。")
    if cfg.norm_placement != "pre":
        raise NotImplementedError(
            f"norm_placement={cfg.norm_placement!r} 暂未建 op 图（仅 'pre' 已实现："
            "attn/ffn 段内嵌 pre-norm）。")
    if cfg.normalization != "RMSNorm":
        raise NotImplementedError(
            f"normalization={cfg.normalization!r} 暂未建 op 图（仅 'RMSNorm' 已实现）。")
    if cfg.position_embedding_type != "rope":
        raise NotImplementedError(
            f"position_embedding_type={cfg.position_embedding_type!r} 暂未建 op 图"
            "（仅 'rope' 已实现；learned_absolute 需额外 pos 表、none 需去 rope op）。")


def _is_moe_layer(cfg: LLMConfig, layer_idx: int) -> bool:
    """本层是否为 MoE FFN 层（否则 dense）。

    优先级（设计 §4/§5）：
      1. 无 MoE（`num_moe_experts` 为 None/0）→ 全 dense。
      2. `moe_layer_freq` 显式给出 → 覆盖 first_k：
         - list/tuple：`moe_layer_freq[layer_idx]` 真值 → MoE；
         - int：每 freq 层一个 MoE（`layer_idx % freq == 0` → MoE，Megatron 语义）。
      3. `first_k_dense_replace`（DSv3 形式）：前 K 层 dense，其后 MoE。
    """
    if not cfg.num_moe_experts:                 # None 或 0 → 纯 dense
        return False
    freq = cfg.moe_layer_freq
    if freq is not None:
        if isinstance(freq, (list, tuple)):
            return bool(freq[layer_idx])
        return (layer_idx % int(freq)) == 0
    k = cfg.first_k_dense_replace or 0
    return layer_idx >= k


def _compress_ratio(cfg: LLMConfig, layer_idx: int) -> int:
    """dsv4_hybrid 本层压缩比 = `cfg.csa_compress_ratios[layer_idx]`（设计 §5/§7.3）。

    未给 ratios 或越界时退化为 0（滑窗 == MLA base，内存中性 §7.4）。
    """
    ratios = cfg.csa_compress_ratios
    if ratios and layer_idx < len(ratios):
        return int(ratios[layer_idx])
    return 0


def gen_layer_pattern(cfg: LLMConfig) -> list:
    """展开层序列为 **`list[LayerContext]`**（设计 §5 步骤 2）。

    `[embedding]` + 每个 transformer 层一个 decoder ctx + `[mtp] * mtp_num_layers`
    + `[lm_head]`。每个 decoder ctx 结构化携带 `attn_type`/`compress_ratio`/`ffn_type`/
    `residual_variant`（不再编码进字符串再反解析）：
      - `ffn_type`（dense/moe）由 `_is_moe_layer` 决定；
      - `compress_ratio` 仅 `dsv4_hybrid` 取 `csa_compress_ratios[layer]`，其它注意力为 None
        （`_build_decoder_body` 据 ctx 字段结构化分支，§7.3）。

    返回 `LayerContext` 列表；其确定性字符串标签（`ctx.name`）在 `build_llm_spec` 里作为
    ModelSpec 的 `layer_pattern`/`layer_specs` 键（输出标签，绝不反解析）。
    """
    pattern = [LayerContext(kind="embedding")]
    for layer_idx in range(cfg.num_layers):
        ffn = "moe" if _is_moe_layer(cfg, layer_idx) else "dense"
        ratio = _compress_ratio(cfg, layer_idx) if cfg.attn_type == "dsv4_hybrid" else None
        pattern.append(LayerContext(
            kind="decoder",
            attn_type=cfg.attn_type,
            compress_ratio=ratio,
            ffn_type=ffn,
            residual_variant=cfg.residual_variant,
        ))
    pattern += [LayerContext(kind="mtp")] * cfg.mtp_num_layers
    pattern.append(LayerContext(kind="lm_head"))
    return pattern


def _use_mhc(cfg: LLMConfig) -> bool:
    """是否启用 mHC 残差包装（设计 §9）。plain 或 n<=1 → False（DSv3 逐字节不变）。"""
    return cfg.residual_variant == "mhc" and cfg.num_residual_streams > 1


def _build_decoder_body(ctx: LayerContext, cfg: LLMConfig, dims: DimTable) -> list:
    """组装一个 decoder 层的 body op（attn 段 + ffn 段，未套 mHC）。

    **统一注册表 API**（结构化 dispatch，读 `ctx` 字段，不解析字符串）：
    - attn 段：`ATTN_REGISTRY[ctx.attn_type](dims, ctx)`——`dsv4_hybrid` 从 `ctx.compress_ratio`
      读 per-layer 压缩比内部分支（§7.3）；`gqa`/`mla` 忽略 ctx。
    - ffn 段：`FFN_REGISTRY[ctx.ffn_type](dims, ctx)`（+ `build_shared_expert_ops` 当 moe & 有 shared expert）。
    """
    attn_ops = list(ATTN_REGISTRY[ctx.attn_type](dims, ctx))
    ffn_ops = list(FFN_REGISTRY[ctx.ffn_type](dims, ctx))
    if ctx.ffn_type == "moe" and cfg.moe_shared_expert_num > 0:
        ffn_ops += build_shared_expert_ops(dims)
    return attn_ops + ffn_ops


def _build_layer_ops(ctx: LayerContext, cfg: LLMConfig, dims: DimTable) -> list:
    """为一个 `LayerContext` 组装 op 列表（设计 §5 步骤 3，dispatch on `ctx.kind`）。

    - `kind=="embedding"` → embedding op（+ mHC 时追加 `hc_expand`：`[S,B,H]→[S,B,n·H]`，§9 stack entry）。
    - `kind=="lm_head"` → head+loss op（+ mHC 时前插 `hc_collapse`：`[S,B,n·H]→[S,B,H]`，§9 stack exit）。
    - `kind=="mtp"` → `build_mtp_ops(cfg)`（embedding + 1 decoder 层 + 共享 head，§10）。
    - `kind=="decoder"` → `_build_decoder_body`；`residual_variant=="mhc"` 时整体套 `mhc_wrap`
      （每层前插 attn_hc/ffn_hc + 残差承载张量 ×n，§9）。

    **不另加 norm op**：attn 段已内嵌 ln1/residual，ffn 段已内嵌 ln2/residual —— plain 路径下
    直接拼接即与 `build_mla_dense_decoder`/`build_mla_moe_decoder` 逐字段一致（1.3 硬门）。
    """
    if ctx.kind == "embedding":
        ops = build_embedding_ops(cfg)
        if _use_mhc(cfg):
            ops = list(ops) + [build_hc_expand_op(dims)]
        return ops
    if ctx.kind == "lm_head":
        ops = build_head_and_loss_ops(cfg)
        if _use_mhc(cfg):
            ops = [build_hc_collapse_op(dims)] + list(ops)
        return ops
    if ctx.kind == "mtp":
        return build_mtp_ops(cfg)

    body = _build_decoder_body(ctx, cfg, dims)
    if _use_mhc(cfg):
        body = mhc_wrap(body, cfg.num_residual_streams, dims)
    return body


def build_llm_spec(cfg: LLMConfig) -> ModelSpec:
    """config 驱动的统一 ModelSpec 装配器（设计 §5，Tier-1）。

    `dims = to_dimtable(cfg)`；`pattern = gen_layer_pattern(cfg)`；对 pattern 中每个
    **唯一** key 组装一份 `LayerSpec`（`_build_layer_ops`）。返回 `ModelSpec`。

    覆盖 Tier-1：`mha/gqa/mla × dense/moe(+shared)` + first_k_dense/moe_layer_freq 决定的
    每层 pattern + embedding/lm_head（tie/loss 感知）。`build_llm_spec(deepseek_v3(N))` 须
    逐桶复现 `build_dsv3_spec(N)`（Task 1.3 硬门）。
    """
    _check_implemented_dispatch(cfg)         # I2：未实现的结构分派项显式报错，不静默产错图
    dims = to_dimtable(cfg)
    pattern = gen_layer_pattern(cfg)           # list[LayerContext]
    # n_layers 必须 = len(layer_pattern)（stage 分配用，ParallelModel._layer_to_stage）。
    # to_dimtable 只算 num_layers+2（embedding+head），未含 MTP 层；此处按实际 pattern 长度校正。
    # mtp_num_layers==0（如 DSv3）时 len(pattern)==num_layers+2 → 无变化，逐字节不变（1.3 硬门）。
    dims.n_layers = len(pattern)
    # 直接用 **LayerContext 本身**（hashable）去重相同层，再把其 `ctx.name` 作为 ModelSpec
    # 的字符串键（输出标签）——构造/分派全程走结构化 ctx，绝不反解析字符串。
    specs_by_ctx = {}
    for ctx in pattern:                        # 保序：dict 记首次出现
        if ctx not in specs_by_ctx:
            specs_by_ctx[ctx] = LayerSpec(_build_layer_ops(ctx, cfg, dims))
    layer_pattern = [ctx.name for ctx in pattern]
    layer_specs = {ctx.name: spec for ctx, spec in specs_by_ctx.items()}
    name = f"llm-{cfg.attn_type}-{cfg.num_layers}L"
    return ModelSpec(name, dims, layer_pattern, layer_specs)
