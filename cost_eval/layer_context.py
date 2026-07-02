"""结构化逐层上下文 `LayerContext`（Task 1，设计 `specs/2026-07-01-...design.md` §5）。

取代原来把 per-layer 变体编码进**字符串 key**（`f"dsv4hyb_r{ratio}_{ffn}"`）再用
`rsplit("_",1)` + `int(prefix[...])` 反解析的脆弱做法：装配器把每层展开为一个**冻结、
可哈希**的 `LayerContext`，dispatch 直接读结构化字段（`kind`/`attn_type`/`compress_ratio`/
`ffn_type`），dedup 直接用 ctx 本身作 dict key。

`ModelSpec` 的 API 仍是 `dict[str, LayerSpec]`（`layer_pattern`/`layer_specs` 用字符串层名），
故 `LayerContext.name` 派生一个**确定性字符串标签**——但它只作 ModelSpec 的**输出标签**，
**绝不**被反解析回逻辑（可扩展多轴时不再改解析代码）。名字格式沿用历史命名
（`mla_dense`/`mla_moe`/`dsv4hyb_r{ratio}_{ffn}`/`embedding`/`mtp`/`lm_head`），以保持
DSv3 golden/regression 的 `spec.layer_pattern`、`spec.layer_specs` 键**逐字节不变**。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerContext:
    """一层在层序列中的结构化身份（frozen → 可作 dedup 的 dict/set key）。

    字段
    ----
    kind : str
        层类别：``"embedding"`` | ``"decoder"`` | ``"mtp"`` | ``"lm_head"``。
    attn_type : str | None
        注意力变体：``gqa`` | ``mla`` | ``dsv4_hybrid``（仅 decoder/mtp 有意义）。
    compress_ratio : int | None
        dsv4_hybrid 本层压缩比（0/1=滑窗、4=CSA、128=HCA）；其它注意力为 None。
    ffn_type : str | None
        FFN 变体：``dense`` | ``moe``（仅 decoder/mtp 有意义）。
    residual_variant : str
        残差变体：``plain`` | ``mhc``（横切；不进入 ``name`` 标签，同一 spec 内恒定）。
    """

    kind: str
    attn_type: str | None = None
    compress_ratio: int | None = None
    ffn_type: str | None = None
    residual_variant: str = "plain"

    @property
    def name(self) -> str:
        """派生确定性字符串层名（ModelSpec 的输出标签，历史命名，绝不反解析）。

        - ``decoder`` + ``dsv4_hybrid`` → ``f"dsv4hyb_r{compress_ratio}_{ffn_type}"``；
        - 其它 ``decoder`` → ``f"{attn_type}_{ffn_type}"``（如 ``mla_moe``/``gqa_dense``）；
        - ``embedding`` / ``mtp`` / ``lm_head`` → 同 ``kind``。
        """
        if self.kind == "decoder":
            if self.attn_type == "dsv4_hybrid":
                return f"dsv4hyb_r{self.compress_ratio}_{self.ffn_type}"
            return f"{self.attn_type}_{self.ffn_type}"
        return self.kind
