"""mindformers pynative 训练 yaml → 评估器配置对象 转换器（D-7，设计
`specs/2026-07-06-audit-remediation.md` §4 D-7）。

`mindformers_yaml (dict 或 path)` → `EvaluatorConfigBundle`
（`LLMConfig` + `ParallelConfig` + `RecomputeSpec` + `SwapSpec` + `OptimizerSpec` + `HardwareSpec`）。

**字段映射逐字复现三处已核验手写映射**（verified source of truth）：
- `validate_dsv4align.py:dsv4_align_config()`（记作 V4）——dsv4 对齐锚点的 `LLMConfig`；
- `.claude/skills/real-machine-memory-sim/prep_dsv4align.py`（记作 P4）——它生成的规范 mindformers dict；
- `cost_eval/presets.py:deepseek_v3()`（记作 D3）——DSv3 缩层锚点。

**保真判据**：把 P4（FUSED 生产）dict 喂进本转换器，得到的 `LLMConfig` 与 `dsv4_align_config()`
**逐字段相等** → 同一评估峰值 14336.5；DSv3 dict → `deepseek_v3()` → 12409.5。

**纯核**：`from_mindformers_dict` 只吃 dict、不 import yaml（`cost_eval` 核无新硬依赖）；仅
`load_mindformers_yaml` 内**惰性** `import yaml`。**fail-loud**：`model` 段任何未映射且未在
「已知内存中性忽略集」的 key → `NotImplementedError`（与 `build_llm._check_implemented_dispatch`
一致，绝不静默产错图）。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..llm_config import LLMConfig
from ..specs import HardwareSpec, OptimizerSpec, ParallelConfig, RecomputeSpec, SwapSpec

GiB = 2 ** 30
MiB = 2 ** 20

# rope 族位置编码：yarn/llama3/dynamic 等都是 rope 的缩放变体，对训练激活内存**等价** rope。
# 非 rope 族（learned_absolute/none）不在此集 → 原样透传给 LLMConfig → build_llm fail-loud。
_ROPE_FAMILY = {"rope", "yarn", "llama3", "dynamic", "linear", "rotary", "default"}

# dtype 名 → 字节。
_DTYPE_BYTES = {
    "float32": 4, "fp32": 4, "float": 4,
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
}


# ── `model.*` 键分类（fail-loud 白名单）───────────────────────────────────────────────
# ① 已映射：被 `_build_llm_config` 消费（含值守卫字段 use_flash_attention / force_unfused_dsa）。
_MAPPED_MODEL_KEYS = {
    "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "vocab_size", "seq_length", "intermediate_size", "head_dim", "qk_head_dim",
    # 同义旧名（_MODEL_KEY_ALIASES 规范化消费,glm4 系）
    "num_layers", "ffn_hidden_size", "padded_vocab_size", "kv_channels",
    "multi_query_group_num", "layernorm_compute_type", "param_init_type",
    "multi_latent_attention", "experimental_attention_variant",
    "kv_lora_rank", "q_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
    "csa_compress_ratios", "csa_window_size", "sliding_window",
    "dsa_indexer_n_heads", "dsa_indexer_head_dim", "dsa_indexer_topk",
    "o_groups", "o_lora_rank",
    "apply_dsa_kernel_fusion", "force_unfused_dsa", "use_flash_attention",
    "gated_linear_unit", "moe_intermediate_size", "moe_capacity_factor",
    "n_routed_experts", "num_experts", "num_experts_per_tok", "n_shared_experts",
    "untie_embeddings_and_output_weights",
    "moe_shared_expert_intermediate_size", "first_k_dense_replace", "moe_layer_freq",
    "enable_hyper_connections", "hc_mult", "num_nextn_predict_layers",
    "add_bias_linear", "add_qkv_bias",
    "normalization", "norm_placement",
    "compute_dtype", "params_dtype", "position_embedding_type", "tie_word_embeddings",
}
# ② 已知「内存中性」忽略集：均**不改内存 op 图**，刻意复现手写映射的省略（如 dsv4_align_config()
#    docstring「qk_layernorm/add_bias omitted（memory-negligible; would fail-loud in build_llm）」——
#    但 qk_layernorm 归到 ① 由 build_llm 原生守卫，见 _build_llm_config）。新增不在 ①∪② 的 model
#    key → fail-loud（不静默吞）。
_IGNORED_MODEL_KEYS = {
    "model_type", "architectures", "max_position_embeddings", "hidden_act", "rms_norm_eps",
    # qk_layernorm：**刻意省略**（内存中性，直接照抄 dsv4_align_config() 的选择——见其 docstring
    # 「qk_layernorm/add_bias omitted（memory-negligible; would fail-loud in build_llm）」）。这是
    # round-trip 到 dsv4_align_config()（qk_layernorm=False）的**必要**条件：P4:88 有 qk_layernorm=True，
    # 若映射到 LLMConfig.qk_layernorm=True 会既破坏字段相等、又触发 build_llm fail-loud。
    "qk_layernorm",
    "mla_qkv_concat", "attention_dropout", "hidden_dropout",
    "layernorm_compute_dtype", "softmax_compute_dtype", "rotary_dtype", "initializer_range",
    "csa_compress_rotary_base", "csa_dense_mode",
    "dsa_indexer_loss_coeff", "dsa_indexer_use_sparse_loss",
    "hc_sinkhorn_iters", "hc_eps", "use_fused_mhc",
    "mtp_loss_scaling_factor",
    "scaling_factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim", "rope_theta",
    "router_dense_type", "routed_scaling_factor",
    "moe_token_dispatcher_type", "moe_grouped_gemm", "moe_router_load_balancing_type",
    "moe_aux_loss_coeff", "scoring_func", "norm_topk_prob", "moe_token_drop_policy",
    "moe_router_enable_expert_bias", "moe_router_bias_update_rate",
    "use_pad_tokens", "topk_group", "n_group",
    # ── 2026-07-11 增补（真机老式 deepseek3 yaml 实测字段,逐项论证内存中性）──────────────
    # kernel 融合 flags:只改瞬时 workspace 形态,不改驻留 saves（融合残差已按 D-5 口径处理）。
    "apply_rope_fusion", "bias_swiglu_fusion", "use_fused_ops_topkrouter",
    # mask 压缩:评估器不建 attention mask 常驻（mask 属 workspace 级）→ 压缩与否不改 op 图。
    "use_attn_mask_compression",
    # 纯 loss 标量权重,无内存效应。
    "mtp_loss_factor",
    # ── 2026-07-14 增补（mf_suhaibo/configs 全量扫描,逐字段论证）──────────────────────
    # 数据管道/输入处理标志,不进 op 图。
    "input_sliced_sig", "use_pad_token", "bos_token_id", "eos_token_id", "pad_token_id",
    # qkv_concat:合并 vs 分开 qkv matmul——saves 总字节相同(同一份激活,1 大 vs 3 小),评估器按
    # 合并建模,内存等量。
    "qkv_concat",
    # fp32 残差链:其**驻留**副本已由 norm-fp32 建模覆盖(ln 存 fp32 输入=残差张量的 fp32 副本,
    # layernorm_compute_dtype 驱动);残差 add 反向直传不存 → 无额外驻留。**该论证仅在
    # layernorm_compute_dtype=fp32 时成立**——fp32_residual=True + ln 非 fp32 的组合由
    # _build_llm_config 值守卫 fail-loud(P0.4,2026-07-14 review)。
    "fp32_residual_connection",
    # router 计算顺序/融合 kernel:logits [S,B,E] 微小且已建,顺序/融合不改驻留。
    "moe_router_pre_softmax", "moe_router_fusion", "use_fused_ops_permute",
    # 数值稳定/结构微变体:scaling 是标量;post-ln 的 norm 输入 saves 与 pre-ln 等量。
    "apply_query_key_layer_scaling", "apply_residual_connection_post_layernorm",
    # attention bias 向量(内存微小,同 add_qkv_bias 口径的省略惯例)。
    "attention_bias", "add_bias_attn",
    # 融合 kernel flags(瞬时 workspace 形态,不改驻留;同 D-5 口径)。
    "use_fused_mla", "use_fused_swiglu", "use_fused_rope",
    # 推理/KV-cache/IO 专用字段,不在训练 op 图。
    "block_size", "num_blocks", "checkpoint_name_or_path", "max_decode_length",
    "repetition_penalty", "temperature", "top_k", "top_p", "do_sample", "is_dynamic",
    "batch_size",   # model 段的推理默认 batch;训练 batch 由 runner/training 段决定
    # GLM 系杂项(池化/激活命名/并行残差标志,结构上已由 attn/ffn 类型与维度捕获)。
    "post_layer_norm", "add_bias_linear_fc", "swiglu", "geglu", "parallel_residual",
    # ── 2026-07-14 第二轮（glm4/qwen3/telechat 全字段）────────────────────────────────
    # softmax fp32 计算(=softmax_compute_dtype 语义,loss 区 fp32 已建模);类注册/生成长度=IO。
    "attention_softmax_in_fp32", "auto_map", "max_length", "type",
    # 融合/排布 flags:字节等量(bias_dropout 融合=瞬时;mlp_concat 同 qkv_concat 论证;
    # contiguous/interleaved 权重排布只改排列不改字节)。
    "bias_dropout_fusion", "mlp_concat", "use_contiguous_weight_layout_attention",
    # gqa 标志(信息已由 num_key_value_heads/multi_query_group_num 表达)。
    "multi_query_attention", "rmsnorm",
    # p-tuning/量化/推理缓存:默认关;启用属推理/微调前缀场景,不在本训练 op 图。
    "pre_seq_len", "prefix_projection", "quantization_bit", "use_past", "use_cache",
    # rope 缩放变体(内存等价 rope);loss 系数;路由均衡策略(容量已由 capacity_factor 建模);部署参数。
    "rope_ratio", "rotary_scaling_factor", "router_aux_loss_coef",
    "moe_router_force_expert_balance", "moe_z_loss_coeff", "npu_nums_per_device",
    "layernorm_epsilon",   # glm4 命名(=rms_norm_eps 同义,数值稳定参数,无内存效应)
}


@dataclass
class EvaluatorConfigBundle:
    """把 6 个评估器配置对象打包（`report.Evaluator(spec, parallel, optimizer, hardware,
    recompute, swap)` 所需）。`llm` 交给 `build_llm_spec` 装配成 `ModelSpec`。"""

    llm: LLMConfig
    parallel: ParallelConfig
    recompute: RecomputeSpec
    swap: SwapSpec
    optimizer: OptimizerSpec
    hardware: HardwareSpec


# ── 小工具 ───────────────────────────────────────────────────────────────────────────
def _dtype_bytes(name, default: int) -> int:
    """dtype 名 → 字节；None/未知 → default（未知非空值 fail-loud）。"""
    if name is None:
        return default
    key = str(name).lower().strip()
    if key not in _DTYPE_BYTES:
        raise NotImplementedError(
            f"未识别的 dtype {name!r}（已知：{sorted(_DTYPE_BYTES)}）——请补充或用标准名。")
    return _DTYPE_BYTES[key]


def _parse_mem_bytes(value) -> int:
    """`context.max_device_memory` → 字节。接受 int（字节）或 "54GB"/"54GiB"/"512MB" 字符串。

    Ascend `max_device_memory` 的 "NGB" 按 **N·2³⁰**（gibibyte）解释（`validate_dsv4align.py:95`
    `54*GiB`，GiB=2³⁰，对齐 P4:49 的 "54GB"）。
    """
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().upper().replace("IB", "B")   # GiB→GB, MiB→MB（统一按 2 的幂）
    mult = {"GB": GiB, "MB": MiB, "KB": 1024, "B": 1}
    for suffix, factor in mult.items():
        if s.endswith(suffix):
            num = s[: -len(suffix)].strip()
            return int(float(num) * factor)
    return int(float(s))   # 纯数字（字节）


def _parse_layer_ranges(spec) -> set:
    """mindformers `full_recompute_layer` → 0-indexed decoder 层集合。

    接受 `["0-3", "5"]`（区间/单值字符串）或 `[0,1,2]`（整数列表）或 `"0-3"`（单串）。
    区间 `"a-b"` 含端点（`range(a, b+1)`）。**注意**：这是 mindformers 的 0-indexed decoder 层号，
    评估器 layer 0=embedding，故调用方须 +1 偏移（见 `_build_recompute`）。
    """
    if spec is None:
        return set()
    if isinstance(spec, (str, int)):
        spec = [spec]
    out = set()
    for item in spec:
        if isinstance(item, int):
            out.add(item)
            continue
        token = str(item).strip()
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(token))
    return out


# ── attn_type 推断 ───────────────────────────────────────────────────────────────────
def _infer_attn_type(model: dict) -> str:
    """`multi_latent_attention` + `experimental_attention_variant` → attn_type（设计 D-7）。

    1. MLA + variant=="dsv4_hybrid" → "dsv4_hybrid"（V4:50）。
    2. MLA（无变体）→ "mla"（D3:39）。
    3. 否则：num_key_value_heads < num_attention_heads → "gqa"；相等/未给 → "mha"（对称，同 builder）。
    4. variant 为其它非空值 → fail-loud（未建 op 图）。
    """
    mla = bool(model.get("multi_latent_attention", False))
    variant = model.get("experimental_attention_variant")
    variant = variant.strip() if isinstance(variant, str) else variant
    if variant:
        if variant != "dsv4_hybrid":
            raise NotImplementedError(
                f"experimental_attention_variant={variant!r} 暂未建 op 图（仅 'dsv4_hybrid' 已实现）。")
        if not mla:
            raise NotImplementedError(
                "experimental_attention_variant='dsv4_hybrid' 需 multi_latent_attention=True。")
        return "dsv4_hybrid"
    if mla:
        return "mla"
    nkv = model.get("num_key_value_heads")
    nh = model.get("num_attention_heads")
    if nkv is not None and nh is not None and int(nkv) < int(nh):
        return "gqa"
    return "mha"


def _query_groups(model: dict, attn_type: str) -> int | None:
    """num_query_groups（n_kv）：显式 `num_key_value_heads` 优先；缺省 MLA 系→1、否则→n_heads。

    MLA/dsv4_hybrid 下 n_kv 对 op 图**惰性**（`attention.py` MLA 分支不用 n_kv）；此默认只为与手写
    锚点 **逐字段相等**：V4:45 `num_query_groups=1`（缺省），D3:34 `num_query_groups=8`（显式）。
    """
    nkv = model.get("num_key_value_heads")
    if nkv is not None:
        return int(nkv)
    if attn_type in ("mla", "dsv4_hybrid"):
        return 1
    return model.get("num_attention_heads")


def _head_dim(model: dict, attn_type: str) -> int | None:
    """head_dim：仅 **plain MLA** 显式取 `qk_nope_head_dim + qk_rope_head_dim`（DSv3=128+64=192，D3:38）。

    `dsv4_hybrid`/GQA/MHA → None（走 `to_dimtable` 的 `H // n_heads` 默认；V4 亦未设 head_dim）。
    MLA 下 head_dim 对 op 图惰性（`attention.py:29-35` 只用 nope/rope/v/lora dims），此值仅为字段相等。
    显式 `head_dim`/`qk_head_dim` 字段（若给）优先。
    """
    explicit = model.get("head_dim", model.get("qk_head_dim"))
    if explicit is not None:
        return int(explicit)
    if attn_type == "mla":
        return int(model.get("qk_nope_head_dim", 0)) + int(model.get("qk_rope_head_dim", 0))
    return None


def _dsa_fused(model: dict) -> bool:
    """dsa_fused = apply_dsa_kernel_fusion；与 force_unfused_dsa 须互反（否则 fail-loud）。

    FUSED 生产（V4 默认 dsa_fused=True，对齐真机 FUSED 锚点 15415.5）：apply=True/force_unfused=False。
    缺省两者都无 → 默认 True（LLMConfig 默认，生产路径）。
    """
    apply = model.get("apply_dsa_kernel_fusion")
    force_unfused = model.get("force_unfused_dsa")
    if apply is not None and force_unfused is not None and bool(apply) == bool(force_unfused):
        raise NotImplementedError(
            f"apply_dsa_kernel_fusion({apply}) 与 force_unfused_dsa({force_unfused}) 须互反"
            "（一个融合、一个不融合）——config 自相矛盾。")
    if apply is not None:
        return bool(apply)
    if force_unfused is not None:
        return not bool(force_unfused)
    return True


# 同义字段规范化（glm4/qwen 等旧命名 → deepseek 系标准名;2026-07-14 全量 configs 扫描）。
# 规范化后后续映射逻辑零改动;旧名计入 _MAPPED（它们被消费）。
_MODEL_KEY_ALIASES = {
    "num_layers": "num_hidden_layers",            # glm4
    "ffn_hidden_size": "intermediate_size",       # glm4
    "padded_vocab_size": "vocab_size",            # glm4
    "kv_channels": "head_dim",                    # glm4
    "multi_query_group_num": "num_key_value_heads",   # glm4 GQA 组数
    "layernorm_compute_type": "layernorm_compute_dtype",  # glm4
    "param_init_type": "params_dtype",            # glm4
}
_REQUIRED_MODEL_KEYS = ("num_hidden_layers", "num_attention_heads", "hidden_size",
                        "vocab_size", "seq_length")


# ── LLMConfig ────────────────────────────────────────────────────────────────────────
def _build_llm_config(model: dict) -> LLMConfig:
    """`model.*` → `LLMConfig`（设计 D-7 字段映射表）。fail-loud：未识别 model key / flash=False。"""
    model = dict(model)
    for old_k, new_k in _MODEL_KEY_ALIASES.items():
        if old_k in model and new_k not in model:
            model[new_k] = model[old_k]
    # 必需结构字段聚合校验（一次列全;缺=该 yaml 依赖 mindformers 类内默认,评估器不猜以免杜撰）。
    miss = [k for k in _REQUIRED_MODEL_KEYS if model.get(k) is None]
    if miss:
        raise ValueError(
            f"yaml 的 model 段缺必需结构字段 {miss}（该 yaml 依赖 mindformers 类内默认值,"
            "评估器不猜默认）——请补全这些字段,或改用页面预设后手工调整。")
    unknown = set(model) - _MAPPED_MODEL_KEYS - _IGNORED_MODEL_KEYS
    if unknown:
        raise NotImplementedError(
            f"未识别的 mindformers `model` 字段（可能改变 op 图但未映射）：{sorted(unknown)}。"
            "已映射见 _MAPPED_MODEL_KEYS，已知内存中性忽略见 _IGNORED_MODEL_KEYS；"
            "如该字段确改内存请在转换器补映射，否则加入忽略集（附‘不改 op 图’论证）。")

    # 值守卫：flash=False 会物化 [S,S] 分数、改 op 图（评估器只建 flash 路径）。
    if model.get("use_flash_attention", True) is False:
        raise NotImplementedError(
            "use_flash_attention=False 暂未建 op 图：非 flash 注意力会物化 [S,S] 分数矩阵"
            "（评估器只建 flash 路径，saves=Q/K/V/O+lse，不物化 [S,S]）——请用 flash 或补 op 图。")

    # 值守卫（P0.4，2026-07-14 review）：fp32_residual_connection 的「内存中性」忽略论证依赖
    # norm-fp32 建模覆盖——layernorm_compute_dtype=fp32 时 norm 保存的 fp32 输入**就是**残差张量的
    # fp32 副本，故 fp32 残差的驻留已被计入。若 fp32_residual=True 而 layernorm dtype 非 fp32
    # （如 bf16），该覆盖不成立：fp32 残差驻留（每层 ~S·B·H·2 额外字节）无人建模 → fail-loud，
    # 不静默低估。当前语料所有 fp32_residual=True 的 yaml（qwen3/telechat3）均配 ln=fp32，不触发。
    if (model.get("fp32_residual_connection")
            and _dtype_bytes(model.get("layernorm_compute_dtype"), 4) < 4):
        raise NotImplementedError(
            "fp32_residual_connection=True 且 layernorm_compute_dtype 非 float32：该组合的 fp32 残差"
            "驻留（每层 ~S·B·H·2 额外字节）未建模。评估器对 fp32 残差的覆盖论证依赖 norm-fp32 建模"
            "（layernorm_compute_dtype=float32 时 norm 保存的 fp32 输入即残差的 fp32 副本），"
            "ln=bf16 时不成立——请把 layernorm_compute_dtype 设为 float32（当前语料均如此），"
            "或为该组合补建模。")

    attn_type = _infer_attn_type(model)
    # n_routed_experts（deepseek 系）/ num_experts（qwen3_moe/general 模板同义字段）
    num_moe_experts = model.get("n_routed_experts") or model.get("num_experts")

    kwargs = dict(
        # 核心维度
        num_layers=int(model["num_hidden_layers"]),
        hidden_size=int(model["hidden_size"]),
        num_attention_heads=int(model["num_attention_heads"]),
        vocab_size=int(model["vocab_size"]),
        seq_length=int(model["seq_length"]),
        batch_size=1,   # local_batch_size 由 training 段定，见 from_mindformers_dict 覆盖
        head_dim=_head_dim(model, attn_type),
        # ① 注意力
        attn_type=attn_type,
        num_query_groups=_query_groups(model, attn_type),
        q_lora_rank=int(model.get("q_lora_rank", 0)),
        kv_lora_rank=int(model.get("kv_lora_rank", 0)),
        qk_rope_head_dim=int(model.get("qk_rope_head_dim", 0)),
        qk_nope_head_dim=int(model.get("qk_nope_head_dim", 0)),
        v_head_dim=int(model.get("v_head_dim", 0)),
        # ② FFN / MoE
        ffn_hidden_size=(int(model["intermediate_size"]) if model.get("intermediate_size") is not None
                         else None),
        gated_linear_unit=bool(model.get("gated_linear_unit", True)),
        num_moe_experts=(int(num_moe_experts) if num_moe_experts else None),
        moe_router_topk=int(model.get("num_experts_per_tok", 0) or 0),
        moe_ffn_hidden_size=(int(model["moe_intermediate_size"])
                             if model.get("moe_intermediate_size") is not None else None),
        moe_shared_expert_num=int(model.get("n_shared_experts", 0) or 0),
        moe_shared_ffn_hidden_size=int(model.get("moe_shared_expert_intermediate_size", 0) or 0),
        moe_capacity_factor=float(model.get("moe_capacity_factor", 1.0)),
        first_k_dense_replace=model.get("first_k_dense_replace"),
        moe_layer_freq=model.get("moe_layer_freq"),
        # 归一化 / 位置编码（结构相关）
        normalization=model.get("normalization", "RMSNorm"),
        norm_placement=model.get("norm_placement", "pre"),
        # qk_layernorm **刻意不映射**（留默认 False）：内存中性、复现 dsv4_align_config() 的省略
        # （见 _IGNORED_MODEL_KEYS 注释）——round-trip 必要条件。
        position_embedding_type=_position_embedding(model),
        # 装配
        # tie:qwen3 系用反义字段 untie_embeddings_and_output_weights（True=不 tie）
        tie_word_embeddings=bool(model.get(
            "tie_word_embeddings",
            not model.get("untie_embeddings_and_output_weights", True))),
        loss_type="logsoftmax_nll",
        embedding_params_dtype_bytes=_dtype_bytes(model.get("params_dtype"), 4),
        # ③ 残差 / MTP / bias / dtype
        residual_variant=("mhc" if model.get("enable_hyper_connections") else "plain"),
        num_residual_streams=(int(model.get("hc_mult", 1))
                              if model.get("enable_hyper_connections") else 1),
        mtp_num_layers=int(model.get("num_nextn_predict_layers", 0) or 0),
        add_bias_linear=bool(model.get("add_bias_linear", False)),
        add_qkv_bias=bool(model.get("add_qkv_bias", False)),
        compute_dtype_bytes=_dtype_bytes(model.get("compute_dtype"), 2),
        # layernorm 计算 dtype（真机 layernorm_compute_dtype，一般 float32）→ norm 激活 fp32（保留 fp32 cast）
        layernorm_compute_dtype_bytes=_dtype_bytes(model.get("layernorm_compute_dtype"), 4),
        # B 标定 margin：select 重算下保留-MoE 层 loss 峰碎片长尾（op 图粒度之下，opdag_validation.md）。
        #   仅 MoE 模型注入（dense 无此 regime）；标定自真机 DSv3 select_attn（1.9× kept-MoE 激活 → OOM-安全）。
        #   仅 select-kept-MoE 生效（mem_timeline._is_kept），full/no-recompute/dense 不触发（锚点不破）。
        #   注：coarse 平台常数（自 DSv3 标定），非 yaml 字段——真机 select-kept-MoE 跑据此避免 18% 欠预测。
        #   融合-CE 模型（DSv4）loss_lids 空 → margin 永不触发 → 设 0（那是另一族融合 kernel 残差，D-5）。
        kept_frag_factor=(1.9 if (num_moe_experts
                                  and not (attn_type == "dsv4_hybrid" and _dsa_fused(model)))
                          else 0.0),
    )

    # dsv4_hybrid 前沿字段（仅该变体设，否则留 LLMConfig 默认 → 对 mla/gqa 惰性）。
    if attn_type == "dsv4_hybrid":
        ratios = model.get("csa_compress_ratios")
        kwargs.update(
            csa_compress_ratios=(tuple(int(r) for r in ratios) if ratios is not None else None),
            csa_window_size=int(model.get("csa_window_size", 128)),
            window_size=int(model.get("csa_window_size", model.get("sliding_window", 128))),
            dsa_indexer_n_heads=int(model.get("dsa_indexer_n_heads", 0)),
            dsa_indexer_head_dim=int(model.get("dsa_indexer_head_dim", 0)),
            dsa_indexer_topk=int(model.get("dsa_indexer_topk", 0)),
            o_groups=int(model.get("o_groups", 0)),
            o_lora_rank=int(model.get("o_lora_rank", 0)),
            dsa_fused=_dsa_fused(model),
            # ①：DSv4 融合生产路径用融合 CE kernel（精简）→ 无重算下不 fat（对齐 dsv4_align_config）。
            cross_entropy_fused=_dsa_fused(model),
        )
    return LLMConfig(**kwargs)


def _position_embedding(model: dict) -> str:
    """position_embedding_type：rope 族 → "rope"（内存等价）；其它原样透传（build_llm fail-loud）。"""
    pe = model.get("position_embedding_type", "rope")
    return "rope" if str(pe).lower() in _ROPE_FAMILY else pe


# ── 并行 / 重算 / 优化器 / 硬件 ────────────────────────────────────────────────────────
def _layers_per_stage(par: dict, num_layers: int, pp: int, mtp: int) -> list | None:
    """`parallelism.{num_layer_list, offset}` → `ParallelConfig.layers_per_stage`（D-8 衔接）。

    per-stage **decoder** 层数 → 评估器每 stage 层数（含伪层）：embedding 归 stage0、(mtp + head) 归末 stage，
    和 == `num_layers + mtp + 2`（= `len(gen_layer_pattern)` = 评估器 n_layers）。缺省（pp==1 或都没给）→ None（均匀切）。

    - `num_layer_list=[n0..n_{pp-1}]`：各 stage decoder 层数，和 == num_hidden_layers。
    - `offset=[o0..]`：各 stage 相对均分 base=`num_layers//pp` 的增量，per-stage=base+offset[i]，和须 == num_layers。
    """
    if pp <= 1:
        return None
    num_layer_list = par.get("num_layer_list")
    offset = par.get("offset")
    if num_layer_list is not None:
        decoder = [int(x) for x in num_layer_list]
    elif offset is not None and isinstance(offset, (list, tuple)):
        base = num_layers // pp
        decoder = [base + int(o) for o in offset]
    else:
        return None   # 无显式配置 → 均匀切（parallel_model 缺省）
    if len(decoder) != pp:
        raise ValueError(f"num_layer_list/offset 长度({len(decoder)}) 必须 == pipeline_parallel({pp})")
    if sum(decoder) != num_layers:
        raise ValueError(
            f"per-stage decoder 层数之和({sum(decoder)}) 必须 == num_hidden_layers({num_layers})")
    stages = list(decoder)
    stages[0] += 1                 # embedding 伪层 → stage 0
    stages[-1] += 1 + mtp          # head 伪层 + mtp 层 → 末 stage
    return stages


def _build_parallel(mf: dict, mtp: int, num_layers: int) -> ParallelConfig:
    par = mf.get("parallelism", {}) or {}
    train = mf.get("training", {}) or {}
    tp = int(par.get("tensor_parallel", 1) or 1)
    ep = int(par.get("expert_parallel", 1) or 1)
    cp = int(par.get("context_parallel", 1) or 1)
    pp = int(par.get("pipeline_parallel", 1) or 1)
    local_bs = int(train.get("local_batch_size", 1) or 1)
    global_bs = train.get("global_batch_size")
    # num_microbatches：pp>1 取 pipeline_parallel_microbatch_size，否则 1
    # （对齐 validate_dsv3.main 的 `mbs = PP if PP>1 else 1`，且 P4:57 pipeline_parallel_microbatch_size=1）。
    ppm = int(par.get("pipeline_parallel_microbatch_size", 1) or 1)
    num_microbatches = ppm if pp > 1 else 1
    # dp_replicate（P0.3，2026-07-14 review）：纯数据并行度——权重/优化器逐 rank **复制**不切分。
    # 评估器正确建模该语义：持久态只 ÷fsdp_degree=dp_shard·cp（static_mem.py:31），dp_replicate 仅进
    # world size（report.py:53）与通信域数（framework.py:52，reserved 池）。老式 yaml 的
    # `parallel_config.data_parallel` + enable_parallel_optimizer=False 由 _mf_adapt 映射到此键。
    dp_repl = int(par.get("data_parallel_replicate", 1) or 1)
    # dp_shard：显式 >0 直取；<=0（auto，P4:55=-1）→ global // (local·num_microbatches·dp_replicate)
    # （global = dp_shard·dp_replicate·local·num_microbatches）。
    dp_cfg = int(par.get("data_parallel_shard", -1))
    if dp_cfg > 0:
        dp_shard = dp_cfg
    elif global_bs is not None:
        dp_shard = max(1, int(global_bs) // (local_bs * num_microbatches * dp_repl))
    else:
        dp_shard = 1
    return ParallelConfig(
        dp_replicate=dp_repl,
        dp_shard=dp_shard,
        cp=cp, tp=tp, pp=pp, ep=ep,
        sequence_parallel=bool(par.get("sequence_parallel", False)),
        num_microbatches=num_microbatches,
        microbatch=ppm,
        layers_per_stage=_layers_per_stage(par, num_layers, pp, mtp),
    )


# mindformers transformer 层的 **cell 名** → 评估器 op 图对应 op 名子串集（`RecomputeSpec.op_matches`
# 做子串匹配）。真机实测（`transformer_layer.py:100/134`）:注意力 cell = `self_attention`、FFN cell
# = **`mlp`**（**非 `feed_forward`**——用错名会静默匹配不到、recompute 不生效,真机 profiler 已证:
# `select_module:feed_forward` 时 GroupedMatmul 仍 live、峰值≈无重算）。子串集清晰隔离 attn vs ffn:
# **cell 边界对齐（真机 transformer_layer.py:92/100/126/134）**:`input_layernorm`(ln1) 与
# `pre_mlp_layernorm`(ln2) 是**独立 cell**,与 `self_attention`/`mlp` 平级 → 重算 self_attention/mlp
# **不含** ln1/ln2（它们保留）；残差 add 也在 cell 外。故子串集**排除 ln1/ln2/add1/add2**（否则会多重算
# layernorm 输出、少估保留量）。q_a/kv_a 的 latent-norm 属 attention cell 内 → 保留在集内。
_SELECT_MODULE_OPS = {
    # self_attention cell:linear_q*/linear_kv*/q_a/kv_a(latent norm)/rope/flash/o_proj（不含 ln1/add1）
    # "qkv" = GQA/MHA 融合投影 op 名(attention.py:74;2026-07-14 review P1.5 补——此前仅 MLA 的
    # linear_q*/linear_kv* 可命中,GQA select 静默漏选 qkv)。
    "self_attention": {"linear_q", "linear_kv", "q_a", "kv_a", "rope", "flash", "o_proj", "qkv"},
    # mlp cell:fc/swiglu/gelu/router/dispatch/e_*/combine/shared（不含 ln2/add2）
    "mlp": {"fc", "swiglu", "gelu", "router", "dispatch", "e_", "combine", "shared"},
}


def _build_recompute(mf: dict) -> RecomputeSpec:
    """`recompute` 段 → `RecomputeSpec`。无段 → mode="None"。full_recompute_layer 0-indexed decoder → +1 偏移。

    **select（D-3，真机配置生效）**:`select_module: {cell_name: [ranges]}` → 逐层 op 子串集。
    cell_name 须是真机 transformer 层的真实 cell 名（`self_attention` / `mlp`）——用错名（如 `feed_forward`）
    真机会静默不重算,故转换器 fail-loud 未知 cell 名,避免"配置貌似生效实则空转"。
    """
    rc = mf.get("recompute")
    if not rc:
        return RecomputeSpec(mode="None", full_layers=set())
    mode = rc.get("mode", "None")
    if mode == "full":
        decoder_layers = _parse_layer_ranges(rc.get("full_recompute_layer"))
        # 评估器 layer 0=embedding、1..N=decoder → mindformers 0-indexed decoder i → 评估器层 i+1。
        return RecomputeSpec(mode="full", full_layers={i + 1 for i in decoder_layers})
    if mode in ("None", None, "none", ""):
        return RecomputeSpec(mode="None", full_layers=set())
    if mode == "select":
        sel_mod = rc.get("select_module") or {}
        select_ops: dict = {}
        for cell_name, ranges in sel_mod.items():
            if cell_name not in _SELECT_MODULE_OPS:
                raise NotImplementedError(
                    f"select_module cell 名 {cell_name!r} 未映射（支持 {sorted(_SELECT_MODULE_OPS)};"
                    "真机 transformer 层 cell 名 = self_attention / mlp,`feed_forward` 是错名、真机不生效）。")
            ops = _SELECT_MODULE_OPS[cell_name]
            for i in _parse_layer_ranges(ranges):        # 0-indexed decoder → 评估器层 i+1
                select_ops.setdefault(i + 1, set()).update(ops)
        return RecomputeSpec(mode="select", select_ops=select_ops)
    raise NotImplementedError(
        f"recompute.mode={mode!r} 暂未映射（转换器支持 'full' / 'None' / 'select'）。")


def _build_optimizer(mf: dict) -> OptimizerSpec:
    """`optimizer` + `model.params_dtype` → `OptimizerSpec`。params_dtype=float32 → params_fp32（state=12）。"""
    opt = mf.get("optimizer", {}) or {}
    model = mf.get("model", {}) or {}
    otype = str(opt.get("type", "AdamW"))
    if otype.lower() not in ("adamw", "adam"):
        raise NotImplementedError(
            f"optimizer.type={otype!r} 暂未映射（评估器持久量模型按 AdamW：master+m+v）。")
    params_fp32 = _dtype_bytes(model.get("params_dtype"), 4) == 4
    return OptimizerSpec.adamw(params_fp32=params_fp32, grad_dtype_bytes=4)


def _build_hardware(mf: dict) -> HardwareSpec:
    """`context.max_device_memory` → `HardwareSpec`。framework_reserve=0（生产默认）、alloc_block=512（平台属性）。"""
    ctx = mf.get("context", {}) or {}
    mdm = ctx.get("max_device_memory")
    max_bytes = _parse_mem_bytes(mdm) if mdm is not None else 54 * GiB
    return HardwareSpec(max_device_memory=max_bytes)


# ── 顶层入口 ─────────────────────────────────────────────────────────────────────────
def from_mindformers_dict(mf: dict) -> EvaluatorConfigBundle:
    """**纯核**：mindformers config dict → `EvaluatorConfigBundle`（不 import yaml）。

    fail-loud：`model` 段未识别字段 / flash=False / 非 dsv4 变体 / dtype 未知 / optimizer 非 AdamW 等。
    """
    if "model" not in mf:
        raise ValueError("mindformers config 缺 `model` 段——无法构建 LLMConfig。")
    model = mf["model"] or {}
    train = mf.get("training", {}) or {}

    llm = _build_llm_config(model)
    # local_batch_size 覆盖 batch_size（training 段，P4:50 local_batch_size=1）。
    llm = dataclasses.replace(llm, batch_size=int(train.get("local_batch_size", 1) or 1))

    parallel = _build_parallel(mf, mtp=llm.mtp_num_layers, num_layers=llm.num_layers)
    recompute = _build_recompute(mf)
    optimizer = _build_optimizer(mf)
    hardware = _build_hardware(mf)
    swap = SwapSpec()   # swap/offload 段映射未实现（见 D-7 限制）→ 默认 disabled。
    return EvaluatorConfigBundle(
        llm=llm, parallel=parallel, recompute=recompute,
        swap=swap, optimizer=optimizer, hardware=hardware,
    )


def load_mindformers_yaml(path) -> EvaluatorConfigBundle:
    """薄 path-loader：**惰性** `import yaml` + `safe_load` + 调纯核 `from_mindformers_dict`。

    yaml 只在此函数内导入 → `cost_eval` 核（`from_mindformers_dict`）不新增硬 yaml 依赖（纯核约束）。
    """
    import yaml   # 惰性：仅路径壳需要，纯核不依赖
    with open(path, "r", encoding="utf-8") as f:
        mf = yaml.safe_load(f)
    return from_mindformers_dict(mf)
