"""统一 LLM ModelSpec 装配器（设计 `specs/2026-07-01-unified-llm-modelspec-design.md` §5）。

一个 config 驱动的 `build_llm_spec(LLMConfig)` 取代"每模型手写一个 build_*_spec"。
本文件 Phase 1（Tier-1）提供：

- `gen_layer_pattern(cfg)`：把 `LLMConfig` 展开成层序列
  `["embedding"] + per-layer {attn}_{dense|moe} + ["mtp"]*mtp_num_layers + ["lm_head"]`。
- `build_llm_spec(cfg)`：按 pattern 为每种唯一层 key 组装一份 `LayerSpec`，返回 `ModelSpec`。

**铁律**：decoder body 由现有命名 op-builder 直接拼接（attn 段已内嵌 ln1/residual，
ffn 段已内嵌 ln2/residual），**不另加 norm op**——否则将偏离 `build_mla_dense_decoder`/
`build_mla_moe_decoder` 而破坏 DSv3 复现硬门（Task 1.3）。
"""
from __future__ import annotations

from .llm_config import LLMConfig, to_dimtable
from .model_spec import DimTable, LayerSpec, ModelSpec
from .layers.registry import ATTN_REGISTRY, FFN_REGISTRY
from .layers.ffn import build_shared_expert_ops
from .layers.head import build_embedding_ops, build_head_and_loss_ops


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


def gen_layer_pattern(cfg: LLMConfig) -> list:
    """展开层序列（设计 §5 步骤 2）。

    `["embedding"]` + 每个 transformer 层一个 key `f"{attn_type}_{dense|moe}"`
    （ffn 由 `_is_moe_layer` 决定）+ `["mtp"] * mtp_num_layers` + `["lm_head"]`。
    """
    pattern = ["embedding"]
    for layer_idx in range(cfg.num_layers):
        ffn = "moe" if _is_moe_layer(cfg, layer_idx) else "dense"
        pattern.append(f"{cfg.attn_type}_{ffn}")
    pattern += ["mtp"] * cfg.mtp_num_layers
    pattern.append("lm_head")
    return pattern


def _build_layer_ops(key: str, cfg: LLMConfig, dims: DimTable) -> list:
    """为一种层 key 组装 op 列表（设计 §5 步骤 3）。

    - `"embedding"` / `"lm_head"` → 装配件 op-builder（head.py）。
    - `"mtp"` → Phase 2（Task 2.x）；DeepSeek-V3 无 MTP，此处显式报错。
    - decoder key `f"{attn}_{dense|moe}"` → `ATTN_REGISTRY[attn](dims) +
      FFN_REGISTRY[ffn](dims) (+ build_shared_expert_ops(dims) 当 moe & 有 shared expert)`。

    **不另加 norm op**：attn 段已内嵌 ln1/residual，ffn 段已内嵌 ln2/residual —— 直接
    拼接即与 `build_mla_dense_decoder`/`build_mla_moe_decoder` 逐字段一致（1.3 硬门）。
    """
    if key == "embedding":
        return build_embedding_ops(cfg)
    if key == "lm_head":
        return build_head_and_loss_ops(cfg)
    if key == "mtp":
        raise NotImplementedError("mtp 层 op 图属 Phase 2（Task 2.x）；DeepSeek-V3 无 MTP")

    # decoder key：`{attn}_{ffn}`（attn 可含下划线，如未来 dsv4_hybrid，故 rsplit 一次）
    attn, ffn = key.rsplit("_", 1)
    ops = list(ATTN_REGISTRY[attn](dims)) + list(FFN_REGISTRY[ffn](dims))
    if ffn == "moe" and cfg.moe_shared_expert_num > 0:
        ops += build_shared_expert_ops(dims)
    return ops


def build_llm_spec(cfg: LLMConfig) -> ModelSpec:
    """config 驱动的统一 ModelSpec 装配器（设计 §5，Tier-1）。

    `dims = to_dimtable(cfg)`；`pattern = gen_layer_pattern(cfg)`；对 pattern 中每个
    **唯一** key 组装一份 `LayerSpec`（`_build_layer_ops`）。返回 `ModelSpec`。

    覆盖 Tier-1：`mha/gqa/mla × dense/moe(+shared)` + first_k_dense/moe_layer_freq 决定的
    每层 pattern + embedding/lm_head（tie/loss 感知）。`build_llm_spec(deepseek_v3(N))` 须
    逐桶复现 `build_dsv3_spec(N)`（Task 1.3 硬门）。
    """
    dims = to_dimtable(cfg)
    pattern = gen_layer_pattern(cfg)
    layer_specs = {
        key: LayerSpec(_build_layer_ops(key, cfg, dims))
        for key in dict.fromkeys(pattern)          # 唯一 key，保序去重
    }
    name = f"llm-{cfg.attn_type}-{cfg.num_layers}L"
    return ModelSpec(name, dims, pattern, layer_specs)
