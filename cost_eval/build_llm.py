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

from .llm_config import LLMConfig


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
