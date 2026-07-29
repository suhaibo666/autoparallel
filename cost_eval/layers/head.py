"""M1：装配件 op-builder —— embedding / lm_head+loss（设计 §6/§10）。

`build_embedding_ops` / `build_head_and_loss_ops` 从 `validate_dsv3.build_embedding` /
`build_lm_head` **逐字段移植**（同 TensorRef / dtype / saves / nll 的
`bwd_scratch="8*S*B*vocab"`），使 `build_llm_spec(deepseek_v3(N))` 复现现有
`build_dsv3_spec(N)` 的 embedding/head 段（Task 1.3 硬门）。返回 op 列表（list[OpSpec]），
由装配器包成 `LayerSpec`，与 `build_*_attn_ops`/`build_*_ffn_ops` 约定一致。
"""
from __future__ import annotations

from ..llm_config import LLMConfig
from ..model_spec import OpSpec, OpType, TensorRef, norm_kind_of

# ═══════════════════════════════════════════════════════════════════════════════════════
# word-embedding **反向** kernel workspace（`GatherDGradV2`）—— **两轮独立真机实测**驱动，
#   源码快照里读不出来（契约 §不要求：aclnn kernel 内部 scratch）。
# ═══════════════════════════════════════════════════════════════════════════════════════
# **实测律**（4 个点、2 个站点、2 次独立采集，**逐字节**吻合）：
#
#     bwd_ws(GatherDGradV2) = 4·vocab·H  +  16 MiB  +  12·H·(B·S)  +  3072 B
#     ├─ 4·vocab·H  = 该 op 自己要累加进去的 [vocab,H] **fp32 梯度表的一份影子拷贝**
#     ├─ 16 MiB     = H 无关常数（两个 H 值上都恰为 16.000 MiB）
#     ├─ 12·H/token = 每 token 3 个 fp32 的 hidden 宽副本（H=4096 → 49152 B/token；
#     │               H=1792 → 21504 B/token，两点各自逐字节成立）
#     └─ 3072 B     = 分配器尾（在 167 的 3 位小数显示上就是那个恒定的 "+0.003 MiB"）
#
# ── 采集 ①：167 / MindSpore 2.10 memory-tracker（`docs/kernel_workspace_2026-07-29.md` §4.4/§8④）──
#   DSv4-hybrid 站点：vocab=129280、**H=4096**、**B=1**、tp=1、cp=1、fused/无重算/L8/m4、pp4·dp2·ep2。
#   BWD 相位单 kernel 极大值，**stage 0**（= 持有 embedding 的那个 stage）：
#       S=1024 → 2084.003 MiB ／ S=2048 → 2132.003 MiB ／ S=4096 → **2228.003 MiB**
#   本式在这三点上给 2084.00293 / 2132.00293 / 2228.00293 MiB —— 三位小数逐位相同。
#   同名 kernel 在 rank2（stage 1，无 embedding）只有 16.03 MiB。
#
# ── 采集 ②：`analysis/realmachine/pp2_norecomp/op_816362.csv`（**仓内既有** profiler，DSv3 pp2 无重算）──
#   完全独立的另一次采集、另一个模型、另一组维度：vocab=129280、**H=1792**、**B=2**、S=4096、
#   pp2/dp1、tp=1、cp=1。该文件里 `Name == GatherDGradV2` 共 24 行，按尺寸/寿命分三类：
#     · `Size(KB)=1093379.0` = **1119620096 B**，`Duration(us)≈22.9`（**瞬态** → workspace 类）× 3
#     · `Size(KB)=904960.5`  = 926679552 B，`Duration≈5.59e6 us`（**长寿** → 就是 4·vocab·H
#        = 926679040 B 的 fp32 梯度表本体 + 512 B 分配器尾）× 3
#     · `Size(KB)=16513.5` / `256.5`（小件，另有其主）
#   本式给 4·129280·1792 + 16 MiB + 12·1792·(2·4096) + 3072 = **1119620096 B** —— **delta = 0 B**。
#   对照 rank 文件 `op_816365.csv`（stage 1，无 embedding）：**没有**任何 ≥1 MiB 的
#   `GatherDGradV2` 块（只剩 16513.5 KB / 256.5 KB 小件）—— 与采集 ① 的 rank2 现象同构。
#
# **归属**（这是本项的关键判据，见 `docs/head_workspace_2026-07-30.md` §2）：
#   该 workspace **属于 word-embedding 的反向**（gather 对 [vocab,H] 权重表求导），
#   **不是** lm_head/loss 段的反向。三条独立证据：
#     ① `4·vocab·H` 那一项**与 S 无关**（167 的三点 S 扫描直接证明）。若它是 loss 侧
#        CE-gather 的 [S·B, vocab] fp32 梯度，就必须 ∝S（S=1024 时只剩 1/4）——实测不然。
#        （在 167 站点 S·B = 4096 == H，两种假设的 4096 点数值**恰好相同**；正是 S 扫描把它们分开。）
#     ② 只出现在**持有 embedding 的那个 rank/stage** 上（两次采集一致，pp4 的 stage0 与
#        pp2 的 stage0；lm_head 在 pp 的**末** stage）。
#     ③ `GatherDGradV2` 是 Gather 的 dgrad；本库 [vocab,H] 形状的 Gather 只有 word embedding
#        （lm_head 前向是 MatMul，其 wgrad 由 MatMul 产出，不走 GatherDGrad）。
#
# **缩放的适用边界（超出即未测，不外推）**：
#   · H：**两点验证**（1792 / 4096），常数项与每-token 项各自逐字节成立 ✓
#   · B：**两点验证**（1 / 2，经 B·S 的 token 数）✓
#   · S：3 点验证线性 ✓（1024/2048/4096）
#   · vocab：**未扫**（两次采集都是 129280）。写成 `4·vocab·H` 是**归因推断**——依据是它
#     逐字节等于该 op 自己输出的那张 fp32 梯度表（采集 ② 里那张表**被同一个 CSV 单独看见**）。
#   · tp：`4·vocab·H` 走**字符串**通道 → **不** ÷tp。真机上 `emb_w` 是 Shard(0)（head.py 见下），
#     故 tp>1 时真值应 ÷tp → 本式**过读 = OOM 安全侧**，未实测，如实记。
#   · cp：每-token 项走 TensorRef → 首个 S 维 ÷cp（结构正确）；常数项不缩放。cp>1 **未实测**。
_EMB_GATHER_DGRAD_BWD_WS = {
    # S 无关部分：影子梯度表 4·vocab·H + 16 MiB + 3072 B 分配器尾。
    # 走字符串通道 → 不吃 cp 整除（shape_eval 刻意不对 bwd_workspace 施加「含 S 就 ÷cp」）。
    "bwd_workspace": "4*vocab*H + 16777216 + 3072",
    # 每-token 部分：3 个 fp32 的 hidden 宽副本。走 TensorRef → 首个 S 维按 cp 切。
    "bwd_workspace_ref": TensorRef("emb_gather_dgrad_ws", ("B", "S", "3*H"), dtype_bytes=4),
}


def _cfg_norm_kind(cfg: LLMConfig) -> str:
    """`LLMConfig` → norm 种类（head 段无 DimTable 在手，直接走 `to_dimtable` 的同一条派生）。"""
    from ..llm_config import to_dimtable
    return norm_kind_of(to_dimtable(cfg))


def build_embedding_ops(cfg: LLMConfig) -> list:
    """word embedding 段（1 op），逐字段同 `validate_dsv3.build_embedding`。

    ``emb_w (vocab,H)`` 为 vocab embedding 权重，**vocab 维 ÷tp**（P0-04 修，2026-07-14）：
    pynative TP>1 **无条件**走 RowwiseParallel Shard(0)（parallelize.py:751-759 +
    style.py:583-588 `{"weight": (Shard(0),)}`，每卡 V/tp×H）。旧注释引用的 `vocab_emb_dp`
    只存在于静态图 legacy 路径（pynative 下零命中），已废。dtype 按
    `cfg.embedding_params_dtype_bytes`（P1-04 接线；默认 2 = compute/gather 副本 bf16，
    真机锚点验证口径）。输出 ``emb_out (S,B,H)`` 在 SP 轴分布。ELEMENTWISE（gather 语义），无 saves。
    """
    w = TensorRef("emb_w", ("vocab", "H"), shard={0: "tp"}, is_weight=True,
                  dtype_bytes=cfg.embedding_params_dtype_bytes)
    out = TensorRef("emb_out", ("S", "B", "H"), shard={0: "sp"})
    # `bwd_workspace{,_ref}`（2026-07-30）：`GatherDGradV2` 的反向 kernel workspace，
    # 两轮独立真机实测逐字节吻合——出处/律/适用边界逐条见本文件顶部 `_EMB_GATHER_DGRAD_BWD_WS`。
    # 它**叠在**反向工作集之上（用完即还，寿命 1 tick），与 `grad_buf` 里那张 4·vocab·H 的
    # 梯度表**是两块**（采集 ② 的 CSV 里二者分别可见：瞬态 1119620096 B / 长寿 926679552 B）
    # → 不双计。
    return [OpSpec("embedding", OpType.ELEMENTWISE, [], out, params=[w], saves=[],
                   **_EMB_GATHER_DGRAD_BWD_WS)]


_LOSS_TYPES = ("logsoftmax_nll", "chunked", "vocab_parallel_ce")


def build_head_and_loss_ops(cfg: LLMConfig) -> list:
    """lm_head + loss 段（3 op），按 `cfg.loss_type` 分支（设计 §10）。

    对照 loss.py：logits(bf16) → cast fp32 → log_softmax(fp32,saved) → NLL；
    反向物化 probs(fp32)。NLL 反向同时物化 probs=exp(-log_softmax) 与 scatter_add 出的
    grad_log_softmax，二者皆 fp32 满 vocab、与 saved log_softmax 共存（loss.py:185-196）
    → ``bwd_scratch="8*S*B*vocab"`` = 2×(4·S·B·vocab)。

    **loss 变体（设计 §10，仅改 loss 区，不动 head 权重语义）**：
      - ``logsoftmax_nll``（默认，`_LogSoftmax`+`_NLLLoss`）：**逐字节不变**（DSv3 硬门）。
      - ``chunked``（`_ChunkCrossEntropyLoss`，loss.py:376-465）：分块反向一次只物化 1/k 满
        vocab 梯度（`grad_logits_chunks` 逐块，:442-464）→ `bwd_scratch = 8*S*B*vocab // k`
        （k=`chunk_loss_num`，guard ≥1）。
      - ``vocab_parallel_ce``（`_VocabParallelCrossEntropy`，loss.py:95-134）：logits 按 vocab
        切（tp），`local_logits [N,V_local]`（:105-113），loss 区 logits/logsm/probs 皆 ∝1/tp。
        `ctx.exp_vals`（softmax 分子，:120）保存供反向 → 建为 sharded `probs` save。

    ``tie_word_embeddings=True`` 时 lm_head 复用 embedding 权重（无独立 head_w 参数，
    ``params=[]``，不重复计入 vocab×H 持久量）；DeepSeek-V3 tie=False，走默认路径，
    逐字段等于 `validate_dsv3.build_lm_head`。
    """
    if cfg.loss_type not in _LOSS_TYPES:
        raise NotImplementedError(
            f"loss_type={cfg.loss_type!r} 暂未建 op 图（支持：{_LOSS_TYPES}）")

    # vocab_parallel_ce：loss 区大张量按 vocab 轴（dim 2 of [S,B,vocab]）切 tp；其余变体不切。
    vshard = {2: "tp"} if cfg.loss_type == "vocab_parallel_ce" else {}

    # 对照 loss.py：logits(bf16) → cast fp32 → log_softmax(fp32,saved) → NLL；反向物化 probs(fp32)
    #
    # ── D-1 再修正（2026-07-07，cp2-none profiler 定位 Bug A）：loss/head 区在 cp 下**是 ÷cp**（序列并行）──
    # 之前误判「full-S」：把 cp=2 profiler 的 2020 MiB buffer 误读成 [B=1, full-S]，实为 **[B=2, S/cp=2048]**
    # （full-S·B=1 与 S/cp·B=2 数值都 2020，混淆了 B 与 cp）；cp=2 full 之所以「0.996」是估计器 B=1·full-S
    # 与真机 B=2·S/cp 数值抵消**蒙对**。真机（`analysis/realmachine/cp2_none/`）证 loss buffer=[S/cp,B,V]
    # → loss/head 区随 cp ÷cp（**不**在 head 前 all-gather）。故这些张量恢复默认 `cp_shard=True`（÷cp），
    # nll 的 bwd_scratch 亦随 op.output.cp_shard=True 在 ShapeEval.resolve ÷cp。主 lm_head 与 MTP 头共享。
    x = TensorRef("h_final", ("S", "B", "H"), shard={0: "sp"})   # head 输入随 cp ÷cp（S/(sp·cp)）
    # 主干 final RMSNorm（P1-01/P2-03 补，2026-07-14）：真机 output_layer 前有 final_layernorm
    # （独立 FSDP wrap，parallelize.py:1165-1170），此前整个 op 缺失——其保留输入（fp32 cast，
    # norm_names 机制）在 loss 峰仍存活，且 gamma 参数缺失致参数图不守恒。
    h_last = TensorRef("h_last", ("S", "B", "H"), shard={0: "sp"})
    fn_g = TensorRef("final_norm_g", ("H",), is_weight=True, dtype_bytes=4)
    logits = TensorRef("logits_lm", ("S", "B", "vocab"), shard=dict(vshard))  # bf16, saved, ÷cp
    logsm = TensorRef("logsm", ("S", "B", "vocab"), shard=dict(vshard), dtype_bytes=4)  # fp32, saved, ÷cp
    loss = TensorRef("loss", ("B",))   # nll 输出

    # P0-04（2026-07-14）：lm_head 权重 vocab 维 ÷tp——pynative TP>1 无条件 ColwiseParallel
    # Shard(0)（out-features=vocab；parallelize.py:765-767 + style.py:439-443），且
    # gather_output=False（style.py:425 默认未覆盖）→ logits 每卡 [N, V/tp] 不 all-gather。
    # dtype 同 embedding（P1-04 接线）。
    if cfg.tie_word_embeddings:
        # 复用 embedding 权重：不新增 head_w 参数（vocab×H 只在 embedding 计一次）
        w = TensorRef("emb_w", ("vocab", "H"), shard={0: "tp"}, is_weight=True,
                      dtype_bytes=cfg.embedding_params_dtype_bytes)
        head_op = OpSpec("lm_head", OpType.MATMUL, [x, w], logits, params=[], saves=[x])
    else:
        w = TensorRef("head_w", ("H", "vocab"), shard={1: "tp"}, is_weight=True,
                      dtype_bytes=cfg.embedding_params_dtype_bytes)
        head_op = OpSpec("lm_head", OpType.MATMUL, [x, w], logits, params=[w], saves=[x])

    # NLL 反向：默认/chunked 用 bwd_scratch（满 vocab 瞬态物化）；vocab_parallel 用 sharded probs save。
    if cfg.loss_type == "vocab_parallel_ce":
        # ctx.exp_vals [N,V_local]（loss.py:120）→ softmax 分子，∝1/tp；建为 sharded save。
        # loss/head 区随 cp ÷cp（Bug A 修正）：vocab 按 tp 切，序列维亦 ÷cp。
        # P0-04 补全（2026-07-14）：①手写 backward 物化 grad_local_logits [N,V/tp] fp32
        #   （loss.py:69-82）→ bwd_scratch_ref（TensorRef 才能 ÷tp）；②max/sum-exp/target-logit
        #   三个 [N,1] fp32 all-reduce 项（loss.py:37-66）→ workspace "12*S*B"（3×4B，量级小如实建）。
        #   per-token loss [N] 在 TP 内复制（不切），与源码一致。
        probs = TensorRef("probs", ("S", "B", "vocab"), shard={2: "tp"}, dtype_bytes=4)
        vp_grad = TensorRef("vp_grad_logits", ("S", "B", "vocab"), shard={2: "tp"}, dtype_bytes=4)
        nll_op = OpSpec("nll", OpType.ELEMENTWISE, [logsm], loss,
                        saves=[logsm, probs], bwd_scratch=None,
                        bwd_scratch_ref=vp_grad, workspace="12*S*B")
    else:
        if cfg.loss_type == "chunked":
            # 分块 CE：一次物化 1/k 满 vocab 梯度（loss.py:442-464）→ bwd_scratch ÷ k。
            k = cfg.chunk_loss_num if cfg.chunk_loss_num >= 1 else 1
            bwd = f"8*S*B*vocab//{k}"
        else:
            bwd = "8*S*B*vocab"
        # NLL 反向同时物化 probs 与 grad_log_softmax（fp32 满 vocab，与 saved log_softmax 共存）
        nll_op = OpSpec("nll", OpType.ELEMENTWISE, [logsm], loss, saves=[logsm], bwd_scratch=bwd)

    return [
        # norm_kind：final_layernorm 亦由 `get_norm_cls` 产出（layer_norm.py:187-191）→ 随配置
        # 分辨是否 cast（RMSNorm 不 cast）。`logsoftmax` **不标**：它不是 get_norm_cls 的 norm
        # （softmax_compute_dtype 的事），且其 saves 已按名被 `_norm_save_names` 排除。
        OpSpec("final_norm", OpType.NORM, [h_last], x, params=[fn_g], saves=[h_last],
               norm_kind=_cfg_norm_kind(cfg)),
        head_op,
        OpSpec("logsoftmax", OpType.NORM, [logits], logsm, saves=[logits]),
        nll_op,
    ]


def build_mtp_ops(cfg: LLMConfig) -> list:
    """MTP 头 op 列表（设计 §10「MTP 头 ≈ embedding + 1 decoder 层 + head」）。

    忠实映射 `multi_token_prediction.py` `MultiTokenPredictionLayer`（:245-404）：
      - 共享 embedding（对 roll 后的 input_ids，:441）→ `decoder_input [S,B,H]`。
      - `enorm(decoder_input)` + `hnorm(hidden_states)`（RMSNorm，:375-376）。
      - `cat((decoder_input, hidden_states), -1)` → `[S,B,2H]`（:379），
        `eh_proj` `Linear(2H → H)`（:304-312/:380）→ `[S,B,H]`。
      - 1 个 transformer 层（`cfg.attn_type` 的 attn + dense/moe ffn，:387）。
      - 共享 head + loss（`process_mtp_loss` 用同一 output_layer + CrossEntropyLoss，:629/:647）。

    op 序列：embedding(1) + enorm/hnorm/eh_cat/eh_proj(4) + decoder(attn+ffn) + head+loss(3)。
    """
    from ..llm_config import to_dimtable
    from .registry import ATTN_REGISTRY, FFN_REGISTRY
    from .ffn import build_shared_expert_ops

    dims = to_dimtable(cfg)
    # 共享 embedding（multi_token_prediction.py:379 `embedding(...)` 用主模型 embedding cell）
    # → MTP embedding op **不携带 params**（vocab×H 权重与主 embedding 层 tie，已计一次，C2）。
    ops = list(build_embedding_ops(cfg))
    for op in ops:
        op.params = []

    # ── MTP 专属投影：enorm / hnorm / cat / eh_proj（2H → H，:375-380）─────────────
    dec_in = TensorRef("decoder_input", ("S", "B", "H"), shard={0: "sp"})   # embedding 输出（roll 后）
    hid = TensorRef("mtp_hidden", ("S", "B", "H"), shard={0: "sp"})         # 主干 hidden_states
    en_out = TensorRef("enorm_out", ("S", "B", "H"))
    hn_out = TensorRef("hnorm_out", ("S", "B", "H"))
    eh_cat = TensorRef("eh_cat", ("S", "B", "2*H"))                          # cat → 2H（:379）
    eh_out = TensorRef("x", ("S", "B", "H"), shard={0: "sp"})               # eh_proj 输出 → decoder 输入
    eh_w = TensorRef("eh_w", ("2*H", "H"), is_weight=True)                   # Linear(2H → H，:304-312)
    # enorm inputs 含 emb_out = MTP 共享 embedding 输出(roll 后即 decoder_input)的**数据流依赖**
    # (embedding→enorm 边,2026-07-11 补边;saves 不变零字节)。hnorm 输入 mtp_hidden 为**主干跨层输入**
    # (真实外部入口,层内无 producer 属语义正确,不补假边)。
    emb_out_ref = TensorRef("emb_out", ("S", "B", "H"), shard={0: "sp"})
    en_g = TensorRef("enorm_g", ("H",), is_weight=True, dtype_bytes=4)   # P1-01 norm gamma
    hn_g = TensorRef("hnorm_g", ("H",), is_weight=True, dtype_bytes=4)
    ops += [
        OpSpec("enorm", OpType.NORM, [dec_in, emb_out_ref], en_out, params=[en_g], saves=[dec_in],
               norm_kind=norm_kind_of(dims)),
        OpSpec("hnorm", OpType.NORM, [hid], hn_out, params=[hn_g], saves=[hid],
               norm_kind=norm_kind_of(dims)),
        OpSpec("eh_cat", OpType.ELEMENTWISE, [en_out, hn_out], eh_cat, saves=[]),
        OpSpec("eh_proj", OpType.MATMUL, [eh_cat, eh_w], eh_out, params=[eh_w], saves=[eh_cat]),
    ]

    # ── 1 个 decoder 层（cfg.attn_type 的 attn + dense/moe ffn，:387）─────────────
    if cfg.attn_type == "dsv4_hybrid":
        from .dsv4_hybrid import build_dsv4_hybrid_attn_ops
        ratios = cfg.csa_compress_ratios
        ratio = ratios[-1] if ratios else 0
        attn_ops = build_dsv4_hybrid_attn_ops(dims, ratio)
    else:
        attn_ops = list(ATTN_REGISTRY[cfg.attn_type](dims))
    ffn = "moe" if cfg.num_moe_experts else "dense"
    ffn_ops = list(FFN_REGISTRY[ffn](dims))
    is_moe = ffn == "moe"
    has_shared = is_moe and cfg.moe_shared_expert_num > 0
    # 统一经 build_transformer_layer 前插 ln2（2026-07-16 修：MTP 内层 transformer_layer 的 MoE
    # 同样漏 pre-FFN norm；与主干 _build_decoder_body 同构，含 shared+moe_add 合流）。
    from .transformer import build_transformer_layer
    body = build_transformer_layer(dims, attn_ops, ffn_ops, is_moe=is_moe, has_shared=has_shared)
    # mHC：MTP 的**内层 transformer_layer 同样跑在打包残差流上**（multi_token_prediction.py
    # :381-399：`expand_hyper_connection_streams` → transformer_layer → `collapse_...`，
    # `self.hc = config.enable_hyper_connections`）。故 mHC 开启时 MTP decoder 也要 ×n 包装
    # （残差承载 ×n + 2 个 HC 模块），并前插 expand / 后接 collapse —— 与主干 decoder 同构（§9）。
    # 此前漏建 → MTP 层激活欠算（其 saves 在主 loss 峰值仍存活，因 MTP 反向在 lm_head 之后）。
    if cfg.residual_variant == "mhc" and cfg.num_residual_streams > 1:
        from .residual import mhc_wrap, NH
        mtp_streams = TensorRef("mtp_hc_streams", ("S", "B", NH), shard={0: "sp"})
        eh = TensorRef("x", ("S", "B", "H"), shard={0: "sp"})              # eh_proj 输出
        expand = OpSpec("mtp_hc_expand", OpType.ELEMENTWISE, [eh], mtp_streams, saves=[])
        collapse_out = TensorRef("h_final", ("S", "B", "H"), shard={0: "sp"})
        collapse = OpSpec("mtp_hc_collapse", OpType.ELEMENTWISE, [mtp_streams], collapse_out, saves=[])
        # collapse 的真实入流 = mhc 层尾更新后的残差流(wrapped 末 op 输出);此处先占位,wrap 后补依赖。
        # 2026-07-24 口径切换：MTP 内层 decoder 的 HC 亦不再 pin（纯理论，见 residual.mhc_wrap）。
        wrapped = mhc_wrap(body, cfg.num_residual_streams, dims)
        # expand→attn_hc_norm 补边(2026-07-11):expand 输出 mtp_hc_streams 即 mhc 段入口流
        # (名字断链;inputs 追加引用,saves 不变零字节)。
        # ⚠ 用 `dataclasses.replace` 而非手写字段清单（2026-07-29 三轮教训，同
        #   `residual._rebuild` 的注释）：这里只想**追加一条 inputs 边**，任何漏带的字段都会
        #   被静默清成默认值。`norm_kind` 已经这样丢过一次。
        import dataclasses
        w0 = wrapped[0]
        wrapped[0] = dataclasses.replace(w0, inputs=list(w0.inputs) + [mtp_streams],
                                         attrs=dict(w0.attrs))
        # collapse←层尾更新流 补边(2026-07-11):collapse 规约的是 mhc 更新后的 streams(wrapped 末
        # op 输出,如 moe_add 的 h2×n),非 expand 的原始流——名字断链致 moe_add 孤立。
        collapse = dataclasses.replace(
            collapse, inputs=list(collapse.inputs) + [wrapped[-1].output],
            attrs=dict(collapse.attrs))
        ops += [expand] + wrapped + [collapse]
    else:
        ops += body

    # ── 共享 head + loss（multi_token_prediction.py:393 `output_layer(hidden_states,
    #    weight=output_weight)` 用主模型 output_layer + 其权重）→ MTP head op **不携带 params**
    #    （H×vocab 权重与主 lm_head tie，已计一次，C2）；loss 段（logsoftmax/nll）保留。──
    head_ops = list(build_head_and_loss_ops(cfg))
    # tie 主 head：清 **lm_head** op 的 params（H×vocab 权重与主 head 共享，只计一次，C2）。
    # P1-01 后 head 段首 op 是 final_norm（MTP 有自己的 final norm，parallelize.py:1184-1248
    # MTP 同构 wrap → 其 gamma 保留），故按 op 名定位而非位置。
    for op in head_ops:
        if op.name == "lm_head":
            op.params = []
    ops += head_ops
    return ops
