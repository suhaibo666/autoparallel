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

import math

from .advisories import warn_modeling_approx
from .layer_context import LayerContext
from .llm_config import LLMConfig, to_dimtable
from .model_spec import DimTable, LayerSpec, ModelSpec
from .layers.registry import ATTN_REGISTRY, FFN_REGISTRY
from .layers.ffn import build_shared_expert_ops, build_moe_merge_op
from .layers.transformer import build_transformer_layer
from .layers.head import build_embedding_ops, build_head_and_loss_ops, build_mtp_ops
from .layers.residual import mhc_wrap, build_hc_expand_op, build_hc_collapse_op


def _reject_non_strict_int(name: str, val) -> None:
    """维度/计数字段须为**严格整数**（排除 bool、浮点、其它类型）——否则 fail-loud（含字段/值/类型）。

    closure-audit v2 §F5b（2026-07-15）：Python `bool` 是 `int` 子类（`True==1`），故仅比
    `<1` 的旧校验会放行 `hidden_size=True`，产 `DimTable(H=True)` 类错图。此守卫**先于**数值
    范围检查调用；仅对**非 None** 值调用（None=惰性默认，由调用方跳过）。
    """
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(
            f"{name}={val!r}（类型 {type(val).__name__}）必须是严格整数——"
            "bool/浮点/其它类型会静默产错图（如 DimTable(H=True)），closure-audit v2 §F5b 拒绝。")


def _validate_structure(cfg: LLMConfig) -> None:
    """结构合法性统一校验（P1-10/P1-11，2026-07-14 review + closure-audit v2 2026-07-15）：
    非法/不自洽配置 fail-loud，不静默 floor/近似产错图。"""
    # ── 核心维度正值校验（P1-11，closure-audit v2 §4.6）─────────────────────────────
    # 探针实证：负/零核心维度此前静默产**负参数 numel**（hidden_size=-1792）/**零激活**
    # （seq_length=0）/**零 vocab 张量**（vocab_size=0）/**退化 2 层图**（num_layers=-1）。
    # 整数维度须 ≥1（≥1 而非 >0 因是整数计数）。**放在最前**：保证下方 `H % n_heads` 等取模
    # 的除数为正（n_heads=0 会裸 ZeroDivisionError / n_heads<0 会错算 head_dim）。
    for _f, _v in (("num_layers", cfg.num_layers), ("hidden_size", cfg.hidden_size),
                   ("num_attention_heads", cfg.num_attention_heads),
                   ("vocab_size", cfg.vocab_size), ("seq_length", cfg.seq_length),
                   ("batch_size", cfg.batch_size)):
        _reject_non_strict_int(_f, _v)       # §F5b：先排除 bool/非整数，再比 <1
        if _v < 1:
            raise ValueError(
                f"{_f}({_v}) 必须 ≥1（负/零核心维度会静默产负参数 numel / 零激活 / 退化层图，"
                "见 closure-audit v2 §4.6）。")
    if cfg.ffn_hidden_size is not None:
        _reject_non_strict_int("ffn_hidden_size", cfg.ffn_hidden_size)   # §F5b（None 惰性跳过）
        if cfg.ffn_hidden_size < 1:
            raise ValueError(
                f"ffn_hidden_size({cfg.ffn_hidden_size}) 必须 ≥1（FFN 隐藏维；≤0 会产负/零 dense-FFN "
                "numel）——留 None 取默认 4·hidden_size。")
    # MLA 家族维度正值校验：仅 attn_type ∈ {mla, dsv4_hybrid, dsa} 施加（这三条注意力的
    # op 图用 q_lora_rank/kv_lora_rank/qk_rope/qk_nope/v_head_dim 做低秩/头维——见 layers/
    # {attention.py build_mla_attn_ops, dsv4_hybrid.py, dsa.py} 符号表达式；≤0 会静默产负/零
    # numel）。**gqa/mha 的 MLA 维默认 0 惰性、从不进 op 图 → 不查**（否则误伤合法 GQA 预设）。
    if cfg.attn_type in ("mla", "dsv4_hybrid", "dsa"):
        for _f, _v in (("q_lora_rank", cfg.q_lora_rank), ("kv_lora_rank", cfg.kv_lora_rank),
                       ("qk_rope_head_dim", cfg.qk_rope_head_dim),
                       ("qk_nope_head_dim", cfg.qk_nope_head_dim),
                       ("v_head_dim", cfg.v_head_dim)):
            _reject_non_strict_int(_f, _v)   # §F5b：MLA 低秩/头维亦须严格整数（bool 放行会产错图）
            if _v <= 0:
                raise ValueError(
                    f"attn_type={cfg.attn_type!r} 需 {_f}>0（当前 {_v}）：MLA 低秩/头维进入 attn "
                    "op 图（linear_qkv/qb/kvb/o_proj 等的符号维），≤0 会静默产负/零 numel。")
    # head_dim 静默 floor（P1-11）：H 不被 n_heads 整除且未显式给 head_dim → 此前 to_dimtable
    # 直接 `H // n_heads` 截断（错维产错图）。
    if cfg.head_dim is None and cfg.hidden_size % cfg.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size({cfg.hidden_size}) 不被 num_attention_heads({cfg.num_attention_heads}) "
            "整除且未显式给 head_dim——静默 floor 会产错误 op 图，请显式配置 head_dim。")
    if cfg.num_query_groups is not None and cfg.attn_type in ("gqa", "mha"):
        if cfg.num_query_groups <= 0 or cfg.num_attention_heads % cfg.num_query_groups != 0:
            raise ValueError(
                f"num_query_groups({cfg.num_query_groups}) 须整除 "
                f"num_attention_heads({cfg.num_attention_heads})（GQA 分组约束）。")
    # csa_compress_ratios（P1-10）：此前 2/3 等非法比被当稀疏 HCA 类路径接受（dsv4_hybrid.py
    # sparse = ratio not in (0,1)）——只有 {0,1(滑窗), 4(CSA), 128(HCA)} 是真实实现取值。
    if cfg.csa_compress_ratios is not None:
        if len(cfg.csa_compress_ratios) != cfg.num_layers:
            raise ValueError(
                f"csa_compress_ratios 长度({len(cfg.csa_compress_ratios)}) 必须 == "
                f"num_layers({cfg.num_layers})（每层一个压缩比）。")
        for r in cfg.csa_compress_ratios:
            # §F5c（2026-07-15）：**先拒非整数值**（bool / 非整数浮点）——旧校验 `int(r)` 会把
            # 4.9 静默截断成 4、错走 dsv4hyb_r4_* 图。整数值（含 4.0 这类整数浮点）才继续档位判定。
            if isinstance(r, bool) or not (isinstance(r, int)
                                           or (isinstance(r, float) and float(r).is_integer())):
                raise ValueError(
                    f"csa_compress_ratios 含非整数压缩比 {r!r}（类型 {type(r).__name__}）——拒绝静默 "
                    "int() 截断（如 4.9→4 会错走 dsv4hyb_r4_* 图）；须为整数值 0/1（滑窗）/4（CSA）/128（HCA）。")
            ri = int(r)                        # 此处 r 已确认整数值：4.0→4 无损，4.9 已在上方拒
            if ri not in (0, 1, 4, 128):
                raise NotImplementedError(
                    f"csa_compress_ratios 含非法压缩比 {r}（实现取值：0/1=滑窗、4=CSA、128=HCA；"
                    "其它值会被误当稀疏路径接受、产错图——拒绝）。")
            if ri not in (0, 1) and cfg.seq_length % ri != 0:
                raise ValueError(
                    f"seq_length({cfg.seq_length}) 不被压缩比 {r} 整除（compressor n_compressed "
                    "= S//ratio 需整除，静默 floor 会错算 compressed KV 字节）。")
    if isinstance(cfg.moe_layer_freq, (list, tuple)) and len(cfg.moe_layer_freq) != cfg.num_layers:
        raise ValueError(
            f"moe_layer_freq 长度({len(cfg.moe_layer_freq)}) 必须 == num_layers({cfg.num_layers})。")
    # moe_ffn_hidden_size（若显式设）须 ≥1（P1-11）：≤0 会产负/零专家 FFN numel。留 None 惰性。
    if cfg.moe_ffn_hidden_size is not None:
        _reject_non_strict_int("moe_ffn_hidden_size", cfg.moe_ffn_hidden_size)   # §F5b（None 惰性跳过）
        if cfg.moe_ffn_hidden_size < 1:
            raise ValueError(
                f"moe_ffn_hidden_size({cfg.moe_ffn_hidden_size}) 必须 ≥1（专家 FFN 隐藏维；≤0 产负/零 "
                "MoE numel）。")
    # ── MoE dispatched-token 口径合法性（F9，closure-report v2 2026-07-16）──────────────────
    # 探针实证：`moe_dispatch_mode="skew", moe_skew_factor=0.5` 把 dispatched token 从均衡 16 压到 8
    # （**低于**均衡值）却无报错——skew 建模的是路由倾斜下**最忙 rank** 相对均值的放大，因子 <1 反而
    # **低估** OOM 边界（语义颠倒）；非有限值会产 NaN/inf 尺寸。capacity_factor 同理须 ≥1（capacity 是
    # drop-and-pad 的容量上界，<1 截掉真实 dispatched token、低估 MoE 激活）。dispatch_mode 只有三个
    # 实现取值（layers/ffn.py `_moe_dispatch_token_expr`）。**默认 balanced/1.0/1.0 全部通过 → 惰性、
    # 锚点逐字节不变**；这些是 dispatch 口径的自身不变量，无论是否 MoE 层都无害地统一校验。
    if not math.isfinite(cfg.moe_skew_factor) or cfg.moe_skew_factor < 1.0:
        raise ValueError(
            f"moe_skew_factor({cfg.moe_skew_factor}) 必须为有限值且 ≥1.0：skew 口径建模最忙 rank "
            "相对均值的放大（percentile 倾斜），<1 会把 dispatched token 压到均衡值以下、**低估** OOM "
            "边界（语义颠倒），非有限值会产 NaN/inf 尺寸。")
    if cfg.moe_capacity_factor < 1.0:
        raise ValueError(
            f"moe_capacity_factor({cfg.moe_capacity_factor}) 必须 ≥1.0：capacity 是 drop-and-pad 的"
            "容量上界，<1 会截掉真实 dispatched token、低估 MoE 激活（OOM 不安全）。")
    if cfg.moe_dispatch_mode not in ("balanced", "capacity", "skew"):
        raise ValueError(
            f"moe_dispatch_mode={cfg.moe_dispatch_mode!r} 非法（仅 balanced|capacity|skew，见 "
            "layers/ffn.py `_moe_dispatch_token_expr`）。")
    if cfg.num_moe_experts:
        # §F5b：先拒 bool（`num_moe_experts=True` 是 truthy 会进 MoE 路径当「True 个专家」）——须
        # 先于下方 topk/负数检查，保证 bool 以类型语义而非数值语义 fail-loud。
        _reject_non_strict_int("num_moe_experts", cfg.num_moe_experts)
        # num_moe_experts 正值校验（P1-11）：`if cfg.num_moe_experts:` 对负数为真 → 会走 MoE 路径
        # 产负专家 numel；0/None 视作纯 dense（_is_moe_layer 语义）跳过。故此处只需拒负数。
        if cfg.num_moe_experts < 1:
            raise ValueError(
                f"num_moe_experts({cfg.num_moe_experts}) 必须 ≥1（MoE 专家数；负数会走 MoE 路径产"
                "负专家 numel）——纯 dense 请用 None 或 0。")
        if cfg.moe_router_topk > cfg.num_moe_experts:
            raise ValueError(
                f"moe_router_topk({cfg.moe_router_topk}) > num_moe_experts({cfg.num_moe_experts})。")
        # closure-audit C1（2026-07-15）：topk≤0 会产生**负/零 numel**（TLOCAL=S·B·topk·C/ep），
        # 此前 shape resolve 得 local_numel=-8388608 而不报错。
        if cfg.moe_router_topk <= 0:
            raise ValueError(
                f"moe_router_topk({cfg.moe_router_topk}) 必须 ≥1（MoE 每 token 至少选 1 专家；"
                "≤0 会产生负/零 dispatched-token numel）。")
        if cfg.moe_capacity_factor <= 0:
            raise ValueError(
                f"moe_capacity_factor({cfg.moe_capacity_factor}) 必须 >0"
                "（≤0 会产生零尺寸 dispatched 张量、错算 MoE 激活）。")
    if isinstance(cfg.window_pattern, (list, tuple)) and len(cfg.window_pattern) != cfg.num_layers:
        raise ValueError(
            f"window_pattern 长度({len(cfg.window_pattern)}) 必须 == num_layers({cfg.num_layers})。")
    # closure-audit v2 §4.5（2026-07-15）：dsv4_hybrid 的分组输出投影**依赖** o_groups>0
    # （O_GROUP_OUT=o_groups·o_lora_rank / O_CHUNK=n_heads·v_head_dim//o_groups）。此前用
    # `not cfg.o_groups` 只拒 0——`o_groups=-1` 为 truthy 绕过、最终解析出 o_group_out.local_numel
    # =-4194304 / o_w=-1835008（探针）。改**严格 >0**：负数（产负 numel）与 0（裸除零）都 fail-loud。
    # **须先于整除检查**：保证进入整除分支时 o_groups 已是正数。
    if cfg.attn_type == "dsv4_hybrid" and cfg.o_groups <= 0:
        raise ValueError(
            f"attn_type='dsv4_hybrid' 需 o_groups>0（当前 {cfg.o_groups}）：分组输出投影 "
            "linear_o_group_proj 用它做 O_GROUP_OUT=o_groups·o_lora_rank / "
            "O_CHUNK=n_heads·v_head_dim//o_groups——≤0 会产负 numel 或在 shape 求值裸除零。")
    if cfg.o_groups and cfg.o_groups > 0:      # 整除检查仅对正 o_groups 有意义（负已在上方拒）
        nvd = cfg.num_attention_heads * (cfg.v_head_dim or 0)
        if nvd and nvd % cfg.o_groups != 0:
            raise ValueError(
                f"n_heads·v_head_dim({nvd}) 不被 o_groups({cfg.o_groups}) 整除"
                "（分组输出投影 O_CHUNK 需整除）。")


def _check_implemented_dispatch(cfg: LLMConfig) -> None:
    """对**改变 op 图**但当前只建了单一取值的分派字段，非实现取值即显式报错（I2）。

    不静默按已实现取值继续（会产「貌似合理实则错误」的图）。仿 head.py:50（loss_type）。

    **例外——内存中性字段不 raise**：`window_size` / `window_pattern`（SWA 滑窗）在
    flash-attn 下对训练激活内存中性（设计 §7.4：saves 仍 Q/K/V/O+lse，`[S,S]` 分数从不
    物化 → SWA 层 op 图 ≡ 全注意力层），故**故意不报错**，仅作忠实表达模型（供 P1
    时间/推理）；`build_llm_spec` 对其不改 op 图。
    """
    # gated_linear_unit=False（ungated MLP）：D-6 已建 op 图（ffn.py 按 d.gated_linear_unit 分派，
    # fc1 输出 F 而非 2F）→ 不再 raise。
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
    # ── Task 3：补齐此前 set-but-ignored 的 op-图相关字段。bias 类为**内存中性**字段
    #   （linear/qkv bias 量级 = [out] 一维,相对权重可忽略）→ round3 A(N4)：从 fail-loud **降级为
    #   ModelingApproxWarning**(不再硬拒,继续按无 bias 建图;如需忠实计入在各 MATMUL op 补 bias param)。
    #   其余改 op 图/产错数的字段仍 fail-loud(下方 normalization/PE/qk_layernorm/dsa)。
    if cfg.add_bias_linear:
        warn_modeling_approx(
            "add_bias_linear=True：linear bias 未建为 param（内存中性,量级 = [out] 一维,相对权重"
            "可忽略）→ 按无 bias 继续估计；如需忠实计入请在各 MATMUL op 补 bias param。")
    if cfg.add_qkv_bias:
        warn_modeling_approx(
            "add_qkv_bias=True（Qwen 系）：QKV 投影 bias 未建为 param（内存可忽略）→ 按无 bias 继续"
            "估计；preset 若仅想标注该架构可用 add_qkv_bias=False + 注释（见 presets.qwen2）。")
    # qk_layernorm dispatch（X3 任务 A，2026-07-15）：gqa/mha 的 attn builder 已建 q/k RMSNorm op
    # （attention.py build_gqa_attn_ops 在 getattr(d,"qk_layernorm") 为真时插入 q_norm/k_norm，
    # 字段由 to_dimtable 从 cfg.qk_layernorm 直通）→ **放行**（op 图已建，不再 fail-loud）。**仅当
    # attn builder 确实没建 q/k norm 时才 fail-loud**：mla/dsv4/dsa 的 attn builder 不据 qk_layernorm
    # 建 q/k norm（其 latent/per-head norm 由自身 op 图覆盖，qk_layernorm 对它们语义惰性）→ 若显式置
    # True 则报错，不静默产「设了却没建」的错图。
    if cfg.qk_layernorm and cfg.attn_type not in ("gqa", "mha"):
        raise NotImplementedError(
            f"qk_layernorm=True 仅 gqa/mha 已建 op 图（attention.py build_gqa_attn_ops 补 Q/K "
            f"per-head RMSNorm）；attn_type={cfg.attn_type!r} 的 attn builder 未据 qk_layernorm 建 "
            "q/k norm op（mla/dsv4/dsa 的 latent/per-head norm 已由自身 op 图覆盖、qk_layernorm 对其"
            "惰性）——如需对该注意力忠实建模请在对应 builder 补 q/k norm op，勿静默产错图。")
    if cfg.attn_type == "dsa" and not (cfg.dsa_indexer_n_heads > 0
                                       and cfg.dsa_indexer_head_dim > 0
                                       and cfg.dsa_indexer_topk > 0):
        raise NotImplementedError(
            "attn_type='dsa' 需 dsa_indexer_n_heads/dsa_indexer_head_dim/dsa_indexer_topk 三维"
            "均 >0（DSv3.2-Exp: 64/128/2048，GLM-5: 32/128/2048）——indexer 是 DSA op 图的组成"
            "部分（layers/dsa.py，预估计），缺维会静默产错图，故报错。")
    # window_size / window_pattern：**故意不 raise**（见本函数 docstring「内存中性字段」）。


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
    is_moe = ctx.ffn_type == "moe"
    has_shared = is_moe and cfg.moe_shared_expert_num > 0
    # 统一 transformer 层：attn(ln1) → **ln2 前置归一(统一前插)** → dense|moe FFN(消费 ln2)
    #   → [moe: shared(消费 ln2) + moe_add 合流]。2026-07-16 修 MoE 漏 ln2 的结构 bug（见
    #   layers/transformer.py 与 build_pre_ffn_norm_op）。dense 全层拼接逐字节不变；MoE 补 ln2。
    return build_transformer_layer(dims, attn_ops, ffn_ops, is_moe=is_moe, has_shared=has_shared)


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
    _validate_structure(cfg)                 # P1-10/11：非法/不自洽结构 fail-loud，不静默 floor
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
