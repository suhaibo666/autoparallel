"""预设模型工厂：把常见 LLM 的真实超参固化为 `LLMConfig` 工厂函数。

每个工厂返回一份 `LLMConfig`（内存结构子集），交给 `build_llm_spec` 装配成
`ModelSpec`。默认值取自各模型公开的架构超参；调用方可覆盖 `num_layers` 等维度做
缩层/变体研究。

- `deepseek_v3(N)`：DeepSeek-V3 缩层结构（MLA + first_k_dense=1 + 8 专家 top4 +
  1 shared expert），**逐字段等于** `validate_dsv3.build_dsv3_spec(N)[0]` 的 DimTable，
  用于复现真机峰值锚点（Task 1.3 硬门：N=4→12472.5 MiB，N=8→13896.1 MiB）。
"""
from __future__ import annotations

from .llm_config import LLMConfig


def deepseek_v3(num_layers: int = 4) -> LLMConfig:
    """DeepSeek-V3 缩层预设（MLA + MoE），复现真机峰值锚点。

    维度逐字段对齐 `validate_dsv3.build_dsv3_spec` 的 `DimTable`
    （H=1792, MLA lora/rope/nope/v dims, 8 专家 top4, moe_F=1024, 1 shared expert,
    first_k_dense_replace=1, compute=bf16）。层序列展开为
    `["embedding", "mla_dense", "mla_moe"*(N-1), "lm_head"]`，即 1 个 dense MLA 层 +
    (N-1) 个 MoE MLA 层，与 `build_dsv3_spec` 的 `layer_pattern` 一致 → 峰值 byte-identical。

    参数
    ----
    num_layers : int
        transformer 层数 N（不含 embedding / lm_head）。
    """
    return LLMConfig(
        num_layers=num_layers,
        hidden_size=1792,
        num_attention_heads=8,
        num_query_groups=8,
        vocab_size=129280,
        seq_length=4096,
        batch_size=1,
        head_dim=192,
        attn_type="mla",
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=192,
        ffn_hidden_size=3072,
        num_moe_experts=8,
        moe_router_topk=4,
        moe_ffn_hidden_size=1024,
        moe_shared_expert_num=1,
        moe_shared_ffn_hidden_size=1024,
        moe_capacity_factor=1.0,
        first_k_dense_replace=1,
        loss_type="logsoftmax_nll",
        compute_dtype_bytes=2,
    )
