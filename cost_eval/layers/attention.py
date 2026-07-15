"""M1：命名 attention op-builder（GQA / MLA）。

从 dense.py / mla.py 抽出的可组合 attention 段构件，供 dense/moe/mla decoder
组装器复用（行为不变，op 定义逐字段与旧切片写法一致）。

- ``build_gqa_attn_ops`` = 旧 ``build_dense_decoder(d).ops[:6]``
  （ln1 → qkv → rope → flash → o_proj → add1，对称 GQA）。
- ``build_mla_attn_ops`` = DeepSeek-V3 MLA 融合路径 10 op（原在 mla.py）。
"""
from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef

# ── GQA 符号维度别名（与 DimTable 字段名一致，eval_expr 可求值）──────────────
QKV = "(n_heads+2*n_kv)*head_dim"   # qkv 投影输出维：(H + 2·n_kv)·d_h，对称 GQA
NHD = "n_heads*head_dim"            # o_proj 输入维 = n_heads·head_dim

# ── FlashAttention softmax 统计量（P1-09 修正 2026-07-14）────────────────────────────
# Ascend `FlashAttentionScore` 除 attention_out 外还输出 **softmax_max + softmax_sum**，各
# `[B, n_heads, S, 8]` fp32（末维 8 = flash 内层 reduce 分块，CANN 固定），是
# `FlashAttentionScoreGrad` 的**输入**（MindSpeed fusion_attention_v2.py:38-40
# `ctx.save_for_backward(..., softmax_max, softmax_sum, ...)`）→ 必须**从前向驻留到该层反向**。
# 修前误建为 fwd-only workspace + lse [S,B,n_heads] 2B saves（仅真值 1/32，且 workspace 不 ÷tp）。
# 修后：`_fa_stats(...)` TensorRef（2×[B,N,S,8] fp32 = 64·B·n_heads·S 字节，head 维 ÷tp、S 维
# 天然 ÷cp）进 flash op 的 **saves**——无重算层驻留至反向（act_live），重算层随 saves 丢弃重物化。
# flash 前向/重算瞬态 workspace 现由 `_fa_workspace()` **TensorRef 型 workspace_ref** 承载（见下方
# :39-45 与 GQA/MLA flash op :101-103/:187-189）：同为 [2,B,n_heads,S,8] fp32，经 resolve_tensor
# 的 shard/cp 机制**按 TP 切（head 维 ÷tp）+ 按 CP 切（S 维 ÷cp）**——已闭合旧字符串 workspace「只
# ÷cp 不 ÷tp、tp>1 重算瞬态高估 8×」的缺口（closure-audit P1-09/§4.10.1，2026-07-15）。
# 下方 `FLASH_LSE_WS` 字符串常量**不再接入任何 flash op 的 workspace**（各 flash op 已改用
# workspace_ref）：保留仅作 (a) 文档/口径参考值（TP=1/CP=1 时 = _fa_workspace 的 numel×4B），
# (b) test_framework_decomposition 对 `_fa_stats`/`_fa_workspace` numel 的数值交叉校验基准。
FLASH_LSE_WS = "64*B*n_heads*S"


def _fa_stats() -> TensorRef:
    """softmax_max+sum 的驻留张量：[2, B, n_heads, S, 8] fp32，head 维 ÷tp（本地头数），
    首个 S 维由 resolve_tensor 的 cp 规则天然 ÷cp。numel×4B = 64·B·n_heads·S/(tp·cp) 字节。"""
    return TensorRef("fa_stats", ("2", "B", "n_heads", "S", "8"),
                     shard={2: "tp"}, dtype_bytes=4)


def _fa_workspace() -> TensorRef:
    """FlashAttention fwd/重算瞬态 workspace（closure-audit C3，2026-07-15）：改字符串
    `FLASH_LSE_WS` 为 TensorRef 型 workspace_ref，让它**按 TP 切**（head 维 ÷tp）——此前字符串
    workspace 只 ÷cp 不 ÷tp（shape_eval:224-229），TP=8 时重算瞬态高估 8×。与 `_fa_stats` 同形
    但独立 name（workspace_ref 不入 saves/act_live，不与驻留 stats 双计驻留窗）。"""
    return TensorRef("fa_ws", ("2", "B", "n_heads", "S", "8"),
                     shard={2: "tp"}, dtype_bytes=4)

# ── MLA 符号维度表达式（与 DimTable 字段名一致，供 eval_expr 求值）──────────
# linear_qkv 输出维（q_lora+kv_lora+k_pe）
QKV_PROJ = "q_lora_rank+kv_lora_rank+qk_rope_head_dim"
# linear_qb 输出维（每头 nope+rope，合并后 n_heads 头）
QB_OUT   = "n_heads*(qk_nope_head_dim+qk_rope_head_dim)"
# linear_kvb 输出维（每头 nope+v，合并后 n_heads 头）
KVB_OUT  = "n_heads*(qk_nope_head_dim+v_head_dim)"
# flash_attn 输出维（n_heads 头，每头 v_head_dim）
ATTN_OUT = "n_heads*v_head_dim"


def build_gqa_attn_ops(d: DimTable) -> list:
    """构造对称 GQA attention 段的 op 列表（6 个 OpSpec）。

    op 序列：ln1 → qkv → rope → flash_attn → o_proj → add1
    权重按 tp 轴切分；flash_attn saves=[qkv, attn, lse]（backward 所需）。
    最后一个 op 输出为 ``h1``（shard={0:'sp'}），可与任意 FFN 尾拼接。

    参数
    ----
    d : DimTable
        模型架构超参（H/n_heads/n_kv/head_dim/S/B …）。
    """
    # ── 激活张量 ───────────────────────────────────────────────────────────────
    # 注：`qkv` 是 Q/K/V **融合**张量（末维 (n_heads+2·n_kv)·d_h），KV 分量不可无损分割 → **不**标
    # cp_kv（D-1 修正）。故 colossal 下 fused-qkv 的 KV 分量仍随 body ÷cp（小幅欠模 colossal 的 KV
    # all-gather buffer）；全重算下该量 off loss 峰、本栈跑不了 cp+无重算故未真机验证——已 caveat。
    #
    # ── P1-13 CP kernel buffer / fused-QKV KV all-gather 审计（X3 任务 B，2026-07-15）───────────
    # colossal（ulysses_degree=1）需把 attention 的 **KV 分量** all-gather 到 full-S；MLA 侧靠
    # KV 激活标 `cp_kv=True`（build_mla_attn_ops 的 kv_a_in/kv_a_out/kvb_out）表达。**GQA 的 KV
    # 是融合在 `qkv` 里的 (2·n_kv·head_dim)/((n_heads+2·n_kv)·head_dim) 分量，无法从融合张量单独标
    # cp_kv** → colossal 下这段 KV 仍随 body ÷cp、**欠建 full-S all-gather buffer**（量级
    # 2·n_kv·head_dim·S·B·compute_dtype）。ring/ulysses 则是另一套语义：CP 通信在飞双缓冲（KV 块
    # send/recv，量级 ~2·(2·n_kv·head_dim)·(S/cp)·B）——随 cp ÷cp、非 all-gather。
    # **为何本 agent 不在此新增 buffer（可证不可行，非疏漏）**：
    #   (a) builder 只见 DimTable、不知 cp/method；`resolve_tensor` 对 builder 表达的 workspace/save
    #       没有 cp>1 或 method 门，且任何引用 S 的量在 cp=1 时是 full-S ≠ 0（破「cp=1 恒 0」惰性
    #       与 MLA 侧 DSv3 golden 逐字节不变）。shape_eval 不在本 agent 可改文件域。
    #   (b) 冻结 must-green `test_cp_activation.test_cp2_all_activations_halve_including_loss` 断言
    #       GQA decoder body 在 colossal cp2 **精确减半**：对 colossal buffer C 代数即得 C≡0（证见
    #       test_x3_cp_buffer.test_colossal_gqa_buffer_is_provably_forbidden_by_halving）——任何
    #       非零 colossal-cp2 GQA buffer 都会破坏该冻结测试、失守验收「全绿」。
    # 结论：GQA 的 colossal KV all-gather / ring-ulysses 双缓冲**保持 caveat 欠建**（off loss 峰、
    # cp2 锚点走 mla 不受影响）；量化与解锁路径见 test_x3_cp_buffer.py 与交付报告「未尽事项」。
    x      = TensorRef("x",    ("S", "B", "H"),        shard={0: "sp"})
    ln1    = TensorRef("ln1",  ("S", "B", "H"))
    qkv    = TensorRef("qkv",  ("S", "B", QKV),        shard={2: "tp"})
    attn   = TensorRef("attn", ("S", "B", NHD),        shard={2: "tp"})
    fa_st  = _fa_stats()   # softmax_max/sum 驻留（P1-09：FlashAttentionScoreGrad 输入）
    o      = TensorRef("o",    ("S", "B", "H"),        partial="tp")
    h1     = TensorRef("h1",   ("S", "B", "H"),        shard={0: "sp"})

    # ── 权重张量（is_weight=True，标注 tp 切分）──────────────────────────────
    qkv_w  = TensorRef("qkv_w", ("H", QKV),   shard={1: "tp"}, is_weight=True)
    o_w    = TensorRef("o_w",   (NHD, "H"),   shard={0: "tp"}, is_weight=True)
    # norm gamma（P1-01，2026-07-14）：RMSNorm 权重 (H,) fp32——真机每层 norm 因 fp32 精度
    # 独立 FSDP wrap（parallelize.py:318-328/:1140-1142），此前缺失致参数图不守恒。
    ln1_g  = TensorRef("ln1_g", ("H",), is_weight=True, dtype_bytes=4)

    ops = [
        # 1. Pre-norm（LayerNorm / RMSNorm）
        OpSpec("ln1",    OpType.NORM,        [x],          ln1,
               params=[ln1_g], saves=[x]),
        # 2. QKV 投影
        OpSpec("qkv",    OpType.MATMUL,      [ln1, qkv_w], qkv,
               params=[qkv_w], saves=[ln1]),
    ]

    # ── Qwen3 qk-layernorm（X3 任务 A，2026-07-15）──────────────────────────────────────────
    # 真机（mindformers/pynative/transformers/attention.py:311-357）：Qwen3 的 SelfAttention 在
    # linear_qkv 投影后、rope 前，对带 head 维的 Q/K 各作一个 RMSNorm（`dim=head_dim`，per-head、
    # 作用在 head_dim 上、compute=layernorm_compute_dtype=fp32）。get_query_key_value_tensors 里
    # split 出 query/key/value 后立即 `q_layernorm(query.reshape(..,head_dim))` / `k_layernorm(key..)`。
    # 建为两个 NORM op（qkv 后、rope 前，in-place 回写 fused `qkv` 载体，保 rope→flash 数据流不断链）：
    #   - gamma 权重 [head_dim] fp32（真机 dim=head_dim，per-head 共享、不随 tp 切——head_dim 非 tp 分维）；
    #   - saves 各自 per-head 输入切片（q: [S,B,n_heads·head_dim] / k: [S,B,n_kv·head_dim]）——RMSNorm
    #     反向所需的 fp32 cast（norm_compute fp32 语义同 ln1，走 _norm_save_names→fp32）。tp 切特征维、
    #     cp 天然 ÷cp（引用 S）→ 与 body 一致、不破 CP halving。
    # 惰性：`getattr(d,"qk_layernorm",False)` 为假（DSv3 mla / DSv4 / 默认 gqa）→ 不插入，逐字节不变。
    # 字段由 to_dimtable 从 LLMConfig.qk_layernorm 直通（llm_config.py），build_llm dispatch 放行 gqa/mha。
    if getattr(d, "qk_layernorm", False):
        q_norm_in = TensorRef("q_norm_in", ("S", "B", NHD),                shard={2: "tp"})
        k_norm_in = TensorRef("k_norm_in", ("S", "B", "n_kv*head_dim"),    shard={2: "tp"})
        q_norm_g  = TensorRef("q_norm_g", ("head_dim",), is_weight=True, dtype_bytes=4)
        k_norm_g  = TensorRef("k_norm_g", ("head_dim",), is_weight=True, dtype_bytes=4)
        ops += [
            # 2a. Q RMSNorm（per-head head_dim；in-place 回写 qkv 的 Q 分量，saves=Q 切片 fp32 cast）
            OpSpec("q_norm", OpType.NORM, [qkv], qkv, params=[q_norm_g], saves=[q_norm_in]),
            # 2b. K RMSNorm（per-head head_dim；in-place 回写 qkv 的 K 分量，saves=K 切片 fp32 cast）
            OpSpec("k_norm", OpType.NORM, [qkv], qkv, params=[k_norm_g], saves=[k_norm_in]),
        ]

    ops += [
        # 3. RoPE（in-place，输出复用 qkv 张量引用）
        OpSpec("rope",   OpType.ROPE,        [qkv],        qkv,
               saves=[]),
        # 4. FlashAttention（saves 存 q/k/v 与 softmax max/sum 统计——FlashAttentionScoreGrad
        #    的输入，驻留至反向；workspace = 重算路径再物化瞬态，见 FLASH_LSE_WS 注释）
        OpSpec("flash",  OpType.FLASH_ATTN,  [qkv],        attn,
               saves=[qkv, attn, fa_st],
               workspace_ref=_fa_workspace()),
        # 5. Output 投影（列并行→行并行）
        OpSpec("o_proj", OpType.MATMUL,      [attn, o_w],  o,
               params=[o_w], saves=[attn]),
        # 6. Residual add（all-reduce 在此隐式完成）
        OpSpec("add1",   OpType.ELEMENTWISE, [o],          h1,
               saves=[]),
    ]
    return ops


def build_mla_attn_ops(d: DimTable) -> list:
    """构造 MLA attention 段的 op 列表（10 个 OpSpec）。

    融合路径（mla_qkv_concat=True）op 序列：
      ln1 → linear_qkv → [split→] q_a_norm → kv_a_norm → linear_qb → linear_kvb
      → rope → flash_attn → o_proj → add1

    返回的 op 列表最后一个 op 输出为 ``h1``（shard={0:'sp'}），
    可与 dense/moe FFN 尾直接拼接。

    参数
    ----
    d : DimTable
        需包含 q_lora_rank / kv_lora_rank / qk_rope_head_dim /
        qk_nope_head_dim / v_head_dim 字段。
    """
    # ── 激活张量 ────────────────────────────────────────────────────────────
    x        = TensorRef("x",        ("S", "B", "H"),      shard={0: "sp"})
    ln1_out  = TensorRef("ln1",      ("S", "B", "H"))
    # linear_qkv 输出（列并行，末维 ÷ tp）
    qkv_out  = TensorRef("qkv_out",  ("S", "B", QKV_PROJ), shard={2: "tp"})
    # q_a 和 kv_a 切片（split 隐含，建模为无 shard 的新张量）。
    # KV 侧激活（kv_a_in/kv_a_out/kvb_out）标 cp_kv=True（D-1 修正）：colossal 下 KV all-gather
    # 到 full-S（不 ÷cp）；ulysses/ring/hybrid 仍随 body ÷cp。DSv3 走 cp=1 → 恒不生效，逐字节不变。
    q_a_in   = TensorRef("q_a_in",   ("S", "B", "q_lora_rank"))
    kv_a_in  = TensorRef("kv_a_in",  ("S", "B", "kv_lora_rank"), cp_kv=True)
    q_a_out  = TensorRef("q_a_out",  ("S", "B", "q_lora_rank"))
    kv_a_out = TensorRef("kv_a_out", ("S", "B", "kv_lora_rank"), cp_kv=True)
    # linear_qb / linear_kvb 输出
    qb_out   = TensorRef("qb_out",   ("S", "B", QB_OUT),   shard={2: "tp"})
    kvb_out  = TensorRef("kvb_out",  ("S", "B", KVB_OUT),  shard={2: "tp"}, cp_kv=True)
    # flash_attn 输出 + softmax 统计（P1-09）
    attn_out = TensorRef("attn",     ("S", "B", ATTN_OUT), shard={2: "tp"})
    fa_st    = _fa_stats()
    # o_proj 输出（行并行，待 all-reduce / reduce-scatter）
    o        = TensorRef("o",        ("S", "B", "H"),       partial="tp")
    # add1 输出（reshard 后回到 SP 分布）
    h1       = TensorRef("h1",       ("S", "B", "H"),       shard={0: "sp"})

    # ── 权重张量 ─────────────────────────────────────────────────────────────
    qkv_w  = TensorRef("qkv_w",  ("H",           QKV_PROJ), shard={1: "tp"}, is_weight=True)
    qb_w   = TensorRef("qb_w",   ("q_lora_rank", QB_OUT),   shard={1: "tp"}, is_weight=True)
    kvb_w  = TensorRef("kvb_w",  ("kv_lora_rank", KVB_OUT), shard={1: "tp"}, is_weight=True)
    o_w    = TensorRef("o_w",    (ATTN_OUT, "H"),            shard={0: "tp"}, is_weight=True)
    # norm gamma（P1-01）：各 norm 的 RMSNorm 权重 fp32（真机独立 FSDP wrap，parallelize.py:1140-1142）
    ln1_g  = TensorRef("ln1_g",       ("H",),            is_weight=True, dtype_bytes=4)
    qan_g  = TensorRef("q_a_norm_g",  ("q_lora_rank",),  is_weight=True, dtype_bytes=4)
    kvan_g = TensorRef("kv_a_norm_g", ("kv_lora_rank",), is_weight=True, dtype_bytes=4)

    return [
        # 1. Pre-norm
        OpSpec("ln1",        OpType.NORM,        [x],               ln1_out,
               params=[ln1_g], saves=[x]),
        # 2. linear_qkv（列并行：H → q_lora+kv_lora+k_pe）
        OpSpec("linear_qkv", OpType.MATMUL,      [ln1_out, qkv_w],  qkv_out,
               params=[qkv_w], saves=[ln1_out]),
        # 3. q_a LayerNorm（在 q_lora_rank 维上；输入为 qkv split 切片）
        #    inputs 含 qkv_out = 切片视图的**数据流依赖**（qkv_out→q_a_norm 边;字节仍按切片 q_a_in 计,
        #    saves 不变——此前名字断链致 op 图出现孤立叶节点）。
        OpSpec("q_a_norm",   OpType.NORM,        [q_a_in, qkv_out], q_a_out,
               params=[qan_g], saves=[q_a_in]),
        # 4. kv_a LayerNorm（在 kv_lora_rank 维上）
        OpSpec("kv_a_norm",  OpType.NORM,        [kv_a_in, qkv_out], kv_a_out,
               params=[kvan_g], saves=[kv_a_in]),
        # 5. linear_qb（列并行：q_lora → n_heads*(nope+rope)）
        OpSpec("linear_qb",  OpType.MATMUL,      [q_a_out, qb_w],   qb_out,
               params=[qb_w], saves=[q_a_out]),
        # 6. linear_kvb（列并行：kv_lora → n_heads*(nope+v)）
        OpSpec("linear_kvb", OpType.MATMUL,      [kv_a_out, kvb_w], kvb_out,
               params=[kvb_w], saves=[kv_a_out]),
        # 7. RoPE（作用于 qb_out 的 rope 部分，in-place，复用同名引用）
        OpSpec("rope",       OpType.ROPE,        [qb_out],          qb_out,
               saves=[]),
        # 8. FlashAttention（q=qb_out, kv=kvb_out；saves 含 softmax max/sum 统计，驻留至反向）
        OpSpec("flash",      OpType.FLASH_ATTN,  [qb_out, kvb_out], attn_out,
               saves=[qb_out, kvb_out, attn_out, fa_st],
               workspace_ref=_fa_workspace()),
        # 9. o_proj（行并行：n_heads*v_head_dim → H，partial=tp）
        OpSpec("o_proj",     OpType.MATMUL,      [attn_out, o_w],   o,
               params=[o_w], saves=[attn_out]),
        # 10. Residual add（all-reduce/reduce-scatter 隐含，输出 h1）
        OpSpec("add1",       OpType.ELEMENTWISE, [o],               h1,
               saves=[]),
    ]
