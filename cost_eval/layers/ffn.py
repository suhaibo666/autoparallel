"""M1：命名 FFN op-builder（dense SwiGLU / MoE / shared expert）。

从 dense.py / moe.py / mla.py 抽出的可组合 FFN 段构件（行为不变，op 定义逐字段
与旧切片写法一致）。

- ``build_dense_ffn_ops``    = 旧 ``build_dense_decoder(d).ops[6:]``
  （ln2 → fc1 → swiglu → fc2 → add2，dense SwiGLU MLP）。
- ``build_moe_ffn_ops``      = 旧 ``build_moe_decoder(d).ops[6:]``
  （router → dispatch → e_fc1 → e_swiglu → e_fc2 → combine，纯 EP）。
- ``build_shared_expert_ops``= 旧 mla ``build_mla_moe_decoder`` 里的 shared 3 op
  （shared_fc1 → shared_swiglu → shared_fc2，纯 tp 切、不 ep）。
"""
from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef

# 每卡 token 数（balanced dispatch，design §7：T_local = S·B·topk·C/ep）。
# capacity_factor=C 影响 dispatched token 数（内存相关）；ep 切分由 shard={0:"ep"} 在
# resolve_tensor 中处理，此处用含 C 的全量符号。C=1.0（DSv3）时退化为 S·B·topk（不变）。
TLOCAL = "S*B*topk*capacity_factor"

# MoE all-to-all **staging 缓冲**（∝ dispatched_tokens·H —— 从 framework_reserve 拆出）。
# 源：experts.py:103-146 `GroupedMLP.permute` 把 token 按专家排序进 `routed_input`
# `[S·B·topk, H]`——ep all-to-all **之前**的发送/置换 staging 缓冲（各卡对自己 S·B 个 token
# 展开 topk 份，**不按 ep 切**，故用全量 S·B·topk）；experts.py:149-173 `unpermute` 反向散射
# （combine staging）。收端 post-a2a token（`disp [TLOCAL,H]{ep}`）已建为 save；此处补的是
# **置换 staging**（compute dtype bf16=2B），dispatch/combine 期瞬时活着 → 建为该 op 的
# workspace（前向逐层临时，非全局常数、非 loss 峰）。= 2×S·B·topk·C·H bytes。
MOE_STAGING_WS = "2*S*B*topk*capacity_factor*H"


def _moe_dispatch_token_expr(d: "DimTable") -> str:
    """MoE 每卡 dispatched-token 数的**全局符号表达式**（shard={0:"ep"} 再 ÷ep 得 per-rank）。

    多口径（P1-12，2026-07-15）——均值（balanced）适合**吞吐**估计，但真实 MoE 受路由倾斜、
    capacity ceil、padding、最忙 rank 影响，均值不足以做 **OOM 安全边界** → 按 `d.moe_dispatch_mode`
    产出不同全局 token 口径：

      - "balanced"（默认，DSv3/DSv4 锚点）：`S·B·topk·C` —— **逐字节等旧 TLOCAL**。物理含义：所有
        expert 收到等量 token 的理想均值（每卡 = /ep），带宽/吞吐估计用。
      - "capacity"：`ceil(S·B·topk·C / n_experts)·n_experts` —— 每 expert 按 capacity 上取整
        （drop-and-pad 到 ceil）后 ×n_experts；shard ÷ep → `ceil(...)·experts_per_rank`。物理含义：
        capacity 丢弃/padding 前的**最忙口径**，OOM 安全边界用（真机 drop-and-pad）。
        `eval_expr` 只有 `+ - * //`（无 `ceil()`）→ 以 `ceil(a/b)=(a+b-1)//b` 表达。divisibility：
        `ceil(...)·n_experts` 恒被 ep 整除（合法 EP 要求 n_experts 被 ep 整除）。
      - "skew"：`S·B·topk·C·moe_skew_factor` —— 均值 × 倾斜因子（percentile 倾斜，≥1）。物理含义：
        路由倾斜下**最忙 rank** 相对均值的放大，OOM 边界用。

    **均值适合吞吐、capacity/skew 适合 OOM 边界**（审计 P1-12 原话）。默认 balanced，新口径经
    `DimTable.moe_dispatch_mode`/`moe_skew_factor` 开启，锚点（gate off + C=1 + balanced）不动。
    """
    mode = getattr(d, "moe_dispatch_mode", "balanced")
    if mode == "balanced":
        return TLOCAL                                    # 逐字节等旧值
    if mode == "capacity":
        # ceil(A/n)·n，A=S·B·topk·C；ceil(a/b)=(a+b-1)//b（受限 eval_expr 无 ceil()）。
        return ("((S*B*topk*capacity_factor + n_experts - 1) // n_experts)"
                " * n_experts")
    if mode == "skew":
        # skew 口径的 factor（须有限且 ≥1，「最忙 rank 相对均值的放大」）由**公共构建入口**
        # `build_llm._validate_structure` 校验（F9 复核 2026-07-16：factor<1 会把 OOM 估计降到均值
        # 以下、OOM 不安全，故 build_llm_spec 直接拒绝）。此低层 expr 沿用「DimTable 不设门、低层
        # 表达式可直调单测」的既有约定（对照 test_x2 用 DimTable('bogus') 期望在上层 raise）。
        return "S*B*topk*capacity_factor*moe_skew_factor"
    raise ValueError(
        f"未知 moe_dispatch_mode: {mode!r}（应为 balanced|capacity|skew）")


def _moe_staging_ws_expr(d: "DimTable") -> str:
    """MoE all-to-all permute/scatter staging 符号（2×dispatched_tokens×H，compute dtype）。

    随 dispatched-token 口径缩放（同 `_moe_dispatch_token_expr`）。balanced 时返回 `MOE_STAGING_WS`
    常量本身 → **逐字节等旧值**（守 test_framework_decomposition 的字符串断言）。
    """
    tok = _moe_dispatch_token_expr(d)
    if tok == TLOCAL:
        return MOE_STAGING_WS                            # 逐字节等旧值
    return f"2*{tok}*H"


def build_pre_ffn_norm_op(d: DimTable) -> "OpSpec":
    """**FFN 前置归一 ln2**（post_attention_layernorm）：`NORM(h1) → ln2`，`saves=[h1]`（fp32 cast）、
    `ln2_g [H] fp32` gamma。**统一** 由 `build_transformer_layer` 前插到 dense/moe FFN 之前。

    2026-07-16 修（用户报告）：此前 ln2 只内嵌在 `build_dense_ffn_ops`，MoE FFN 段（`build_moe_ffn_ops`
    / `build_shared_expert_ops`）直接吃裸 `h1`、**无 ln2** → 每 MoE 层漏建一份 `[S,B,H]` fp32 cast 常驻
    （norm_compute=fp32），MoE 模型在无重算/select 下系统性欠预测。hoist 到统一 transformer 层后：
    dense 全层逐字节不变（同一 ln2 op，只是产出方从 ffn builder 变为本函数）；MoE 层补上 ln2、routed+
    shared 都消费 `ln2`（与真机 `mlp(post_attn_norm(hidden))` 一致）。"""
    h1    = TensorRef("h1",   ("S", "B", "H"), shard={0: "sp"})   # attn 残差输出（sp 切）
    ln2   = TensorRef("ln2",  ("S", "B", "H"))                    # 归一输出（全 S，进 FFN 列并行前 all-gather）
    ln2_g = TensorRef("ln2_g", ("H",), is_weight=True, dtype_bytes=4)   # P1-01 norm gamma（fp32）
    return OpSpec("ln2", OpType.NORM, [h1], ln2, params=[ln2_g], saves=[h1])


def build_dense_ffn_ops(d: DimTable) -> list:
    """构造 dense SwiGLU MLP FFN 段的 op 列表（4 个 OpSpec；**ln2 已 hoist 到 transformer 层**）。

    op 序列：fc1 → swiglu → fc2 → add2（消费上游 `build_pre_ffn_norm_op` 产出的 ``ln2``）。
    2026-07-16：ln2 前置归一由 `build_transformer_layer` 统一前插（见 `build_pre_ffn_norm_op`），
    本段不再自建 ln2 op —— dense 全层拼接后逐字节不变。

    参数
    ----
    d : DimTable
        模型架构超参（H/F/S/B …）。
    """
    # gated（SwiGLU）：fc1 输出 2F（gate+up）→ swiglu → F；ungated（D-6）：fc1 输出 F → gelu → F。
    gated = getattr(d, "gated_linear_unit", True)
    fc1_out = "2*F" if gated else "F"       # fc1 输出维（gate+up 合并 vs 纯 up）
    act_name = "swiglu" if gated else "gelu"

    # ── 激活张量 ───────────────────────────────────────────────────────────────
    ln2    = TensorRef("ln2",  ("S", "B", "H"))          # 上游 ln2 op 产出（本段 fc1 的输入）
    g      = TensorRef("g",    ("S", "B", fc1_out),      shard={2: "tp"})
    act    = TensorRef("act",  ("S", "B", "F"),          shard={2: "tp"})
    o2     = TensorRef("o2",   ("S", "B", "H"),          partial="tp")
    h2     = TensorRef("h2",   ("S", "B", "H"),          shard={0: "sp"})

    # ── 权重张量（is_weight=True，标注 tp 切分）──────────────────────────────
    fc1_w  = TensorRef("fc1_w", ("H", fc1_out), shard={1: "tp"}, is_weight=True)
    fc2_w  = TensorRef("fc2_w", ("F",  "H"),    shard={0: "tp"}, is_weight=True)

    # std 全重算保留集(185 基准,见 attention.py 同名注释):ln2(fc1 保留输入 bf16)在真机全重算
    # 下不释放(g/act/h1-fp32 才释放)→ flag 开启时标 pin。默认关,逐字节不变。
    if getattr(d, "std_recompute_ctx_pin", False):
        ln2.pin_under_recompute = True

    return [
        # FFN 上投影（gated→2F gate+up；ungated→F）
        OpSpec("fc1",    OpType.MATMUL,      [ln2, fc1_w], g,
               params=[fc1_w], saves=[ln2]),
        # 激活（gated=SwiGLU 2F→F；ungated=gelu F→F）
        OpSpec(act_name, OpType.ELEMENTWISE, [g],          act,
               saves=[g]),
        # FFN down 投影（行并行）
        OpSpec("fc2",    OpType.MATMUL,      [act, fc2_w], o2,
               params=[fc2_w], saves=[act]),
        # Residual add
        OpSpec("add2",   OpType.ELEMENTWISE, [o2],         h2,
               saves=[]),
    ]


def build_moe_ffn_ops(d: DimTable) -> list:
    """构造 MoE FFN 段的 op 列表（6 个 OpSpec，纯 EP）。

    op 序列：router → dispatch(all-to-all) → e_fc1(moe_gemm) → e_swiglu
             → e_fc2(moe_gemm) → combine(all-to-all)
    输入激活为 attn 段输出的 ``h1``（shard={0:'sp'}）。

    专家权重设计（纯 EP）：expert_parallel.py:330 ``weight:(Shard(0),)`` ——
    专家按 ep 轴切分，不做 tp 切分。shard={0:"ep"}，不含 "tp"。
    T_local = S*B*topk//ep（balanced dispatch，capacity=1）。

    参数
    ----
    d : DimTable
        需包含 n_experts / topk / moe_F 字段。
    """
    # gated（SwiGLU）：专家 fc1 输出 2·moe_F；ungated（D-6）：moe_F。
    gated = getattr(d, "gated_linear_unit", True)
    e_fc1_out = "2*moe_F" if gated else "moe_F"
    e_act_name = "e_swiglu" if gated else "e_gelu"

    # dispatched-token 全局口径（P1-12，2026-07-15）：balanced（默认，=TLOCAL 逐字节不变）/
    # capacity（ceil 最忙）/ skew（均值×倾斜）。见 `_moe_dispatch_token_expr`。
    tlocal = _moe_dispatch_token_expr(d)
    staging = _moe_staging_ws_expr(d)

    # ── MoE FFN 激活张量 ────────────────────────────────────────────────────
    # **ln2 前置归一输出**（build_pre_ffn_norm_op 产出）作为 router 与 dispatch 的输入。
    # 2026-07-16 修：此前直接吃裸 `h1`（无 pre-FFN norm）→ 漏 ln2；改吃 `ln2`（与 dense fc1 同源）。
    hin  = TensorRef("ln2",   ("S", "B", "H"))
    # router logits（全 token × 全专家，无切分）
    logits = TensorRef("logits", ("S", "B", "n_experts"))
    # dispatch 后 token 按 ep 分片（all-to-all）
    disp = TensorRef("disp",  (tlocal, "H"),                    shard={0: "ep"})
    # 专家 fc1 输出（gated=2·moe_F gate+up；ungated=moe_F）
    g    = TensorRef("e_g",   (tlocal, e_fc1_out),              shard={0: "ep"})
    # 激活后（moe_F 维）
    act  = TensorRef("e_act", (tlocal, "moe_F"),                shard={0: "ep"})
    # 专家 fc2 输出（H 维，仍按 ep 分片）
    eo   = TensorRef("e_o",   (tlocal, "H"),                    shard={0: "ep"})
    # combine 输出（all-to-all 还原到原始 token 序列）
    comb = TensorRef("comb",  ("S", "B", "H"),                  shard={0: "sp"})

    # ── 专家权重（纯 EP：dim 0 按 ep 轴切分，不含 tp）─────────────────────
    # shape 用全量维度（n_experts），shard={0:"ep"} 在 resolve_tensor 中做整除
    w1 = TensorRef("e_w1", ("n_experts", "H",     e_fc1_out), shard={0: "ep"}, is_weight=True)
    w2 = TensorRef("e_w2", ("n_experts", "moe_F", "H"),       shard={0: "ep"}, is_weight=True)
    # router 权重（P1-01，2026-07-14）：[n_experts, H] **fp32**——真机 router 因 fp32 精度
    # 单独 FSDP wrap（parallelize.py:1116-1128），optimizer 参数表含
    # `decoder.layers.N.mlp.router.weight`（NPU 旁证）。此前缺失 → 参数图不守恒。
    # （aux-loss-free 的 expert_bias [n_experts] 是 hook 更新的 buffer 非 optimizer 参数
    # （gpt_model.py:658-671），字节可忽略，不入 params。）
    router_w = TensorRef("router_w", ("n_experts", "H"), is_weight=True, dtype_bytes=4)

    return [
        # 1. Router（softmax + top-k 选择，输出 logits 存 backward 用）
        OpSpec("router",   OpType.MOE_ROUTER, [hin, router_w], logits,
               params=[router_w], saves=[logits]),
        # 2. Dispatch（all-to-all；把 token 路由到各 expert rank）
        #    workspace = 置换发送 staging 缓冲（experts.py permute → routed_input [S·B·topk,H]）
        #    inputs 含 logits = 路由索引的**数据流依赖**（router→dispatch 边;dispatch 按 topk 选路,
        #    字节/saves 不变——此前缺此边致 router 成孤立叶节点）。
        OpSpec("dispatch", OpType.DISPATCH,   [hin, logits], disp,
               saves=[disp], workspace=staging),
        # 3. 专家 fc1（grouped GEMM，按 ep 切分的专家矩阵）
        OpSpec("e_fc1",    OpType.MOE_GEMM,   [disp, w1], g,
               params=[w1], saves=[disp]),
        # 4. 激活（gated=SwiGLU；ungated=gelu）
        OpSpec(e_act_name, OpType.ELEMENTWISE, [g],        act,
               saves=[g]),
        # 5. 专家 fc2（grouped GEMM）
        OpSpec("e_fc2",    OpType.MOE_GEMM,   [act, w2],  eo,
               params=[w2], saves=[act]),
        # 6. Combine（all-to-all；把 expert 输出还原到 token 序列）
        #    workspace = 反向散射 staging 缓冲（experts.py unpermute，:149-173）
        OpSpec("combine",  OpType.COMBINE,    [eo],        comb,
               saves=[comb], workspace=staging),
    ]


def build_moe_merge_op(d: DimTable) -> "OpSpec":
    """MoE 层尾合流 op（2026-07-11 补边）：routed 输出(comb) + shared 输出 相加 → h2。

    真机语义:`moe_layer construct: output = routed + shared` + transformer_layer 残差。此前 moe 层
    未建此 op（dense 层有 add2、moe 层没有,不对称）→ combine/shared_fc2 成 op 图孤立叶节点。
    线性 elementwise:saves=[]（反向直传）→ **激活字节零变化**;仅 forward_max_live 尾部多一个
    live 输出（若动锚点即回退）。

    **shared-expert 门（P1-01，closure-audit §4.2，2026-07-15）**：`moe_shared_gate=True` 时
    shared 分支尾部已产出 gated 输出 `sh_o_gated = sigmoid(sh_gate)·sh_o`（build_shared_expert_ops
    的 shared_gate_mul，源 `shared_experts.py:56-64`），合流须消费 **gated 输出**——否则 gate 只补了
    参数、其数据流成孤立叶节点。gate off（DSv3 默认）走原路径消费裸 `sh_o`，输入/saves 逐字节不变。
    """
    comb = TensorRef("comb", ("S", "B", "H"), shard={0: "sp"})
    h2   = TensorRef("h2",   ("S", "B", "H"), shard={0: "sp"})
    # P0-1（2026-07-23）：shared 输出随权重 TP 复制改为 seq-SP 分布（见 build_shared_expert_ops
    # 注释;同名张量须与产出侧同分布——ShapeEval P1-03 同名同 numel 不变量）。
    if getattr(d, "moe_shared_gate", False):
        shared_in = TensorRef("sh_o_gated", ("S", "B", "H"))
    else:
        shared_in = TensorRef("sh_o", ("S", "B", "H"))
    return OpSpec("moe_add", OpType.ELEMENTWISE, [comb, shared_in], h2, saves=[])


def build_shared_expert_ops(d: DimTable) -> list:
    """构造 MoE shared expert 段的 op 列表（3 个 OpSpec，纯 tp 切、不 ep）。

    op 序列：shared_fc1 → shared_swiglu → shared_fc2
    吃 attn 段输出的 ``h1``，intermediate=moe_shared_F。
    输出名 ``sh_o`` 与 MoE combine 输出 ``comb`` 不同名，避免冲突。

    参数
    ----
    d : DimTable
        需包含 moe_shared_F 字段。
    """
    # gated（SwiGLU）：2·moe_shared_F；ungated（D-6）：moe_shared_F。
    gated = getattr(d, "gated_linear_unit", True)
    sh_fc1_out = "2*moe_shared_F" if gated else "moe_shared_F"
    sh_act_name = "shared_swiglu" if gated else "shared_gelu"

    # ── Shared expert 激活 / 权重 ─────────────────────────────────────────
    # shared expert 与 routed 共同吃 **ln2**（pre-FFN norm 输出）——2026-07-16 修（此前吃裸 h1）。
    #
    # ── P0-1（2026-07-23，runtime 377c9c344）：shared-expert **权重 TP 复制、激活按序列 SP 切** ──
    # runtime `parallelize.py:718-724`：shared expert 参数在 TP 上 **replicated**（注释明言
    # "shared expert weights are replicated across TP"），只对 token 激活用
    # `SequenceParallel(sequence_dim=0)`;权重随后属 dense FSDP wrap（:1063-1068）。修前把
    # sh_w1/sh_w2 按 tp 切（列/行并行）→ 持久/优化器态/梯度/gather 全部欠估 T 倍（audit §4.4,
    # OOM 不安全）。修后:权重去 tp shard（复制,persistent 从 P/(TK)→P/K）;激活 sh_g/sh_act 改
    # 按序列维 {0:"sp"} 切（sequence_dim=0,numel 与旧末维 ÷tp 等价——sp==tp 时;tp=1 锚点两者
    # 恒等,逐字节不变）;输出 sh_o 权重复制下无 partial-sum（各 rank 算自己的 seq 分片）→ 去
    # partial、改 {0:"sp"}。
    # 注:输出 sh_o 真机同为 seq-SP 分布,但 [S,B,H]+{0:"sp"} 恰是 mHC 残差承载签名
    # (residual.py _is_residual_carrier 会将其 ×n 重命名)——sh_o 非 saves、只影响 fml,
    # 故取**全量口径**(≥真实,保守;tp=1 恒等),不标 sp、不标 partial(权重复制下无部分和)。
    hin_sh     = TensorRef("ln2",    ("S", "B", "H"))
    sh_g       = TensorRef("sh_g",   ("S", "B", sh_fc1_out),      shard={0: "sp"})
    sh_act     = TensorRef("sh_act", ("S", "B", "moe_shared_F"),  shard={0: "sp"})
    sh_o       = TensorRef("sh_o",   ("S", "B", "H"))
    sh_fc1_w   = TensorRef("sh_w1",  ("H",            sh_fc1_out), is_weight=True)
    sh_fc2_w   = TensorRef("sh_w2",  ("moe_shared_F", "H"),        is_weight=True)

    ops = [
        # shared fc1（列并行：H → gated 2*moe_shared_F / ungated moe_shared_F）
        OpSpec("shared_fc1",    OpType.MATMUL,      [hin_sh, sh_fc1_w], sh_g,
               params=[sh_fc1_w], saves=[hin_sh]),
        # shared 激活（gated=SwiGLU；ungated=gelu）
        OpSpec(sh_act_name,     OpType.ELEMENTWISE, [sh_g],             sh_act,
               saves=[sh_g]),
        # shared fc2（行并行：moe_shared_F → H，partial=tp）
        OpSpec("shared_fc2",    OpType.MATMUL,      [sh_act, sh_fc2_w], sh_o,
               params=[sh_fc2_w], saves=[sh_act]),
    ]
    # shared-expert 门（P1-01，closure-audit §4.2，2026-07-15）：use_shared_expert_gating=True 时
    # 真机 `shared_out_gated = sigmoid(gate_logits)·shared_expert_out`，`gate_logits = Linear(H→1)(hidden)`
    # （shared_experts.py:56-64）。此前只建了门权重 [H,1]、门激活 [S,B,1] 却**无消费者**（孤立叶节点），
    # 真正合流仍用未乘 gate 的裸 sh_o。此处让门真正进入数据流：
    #   ① shared_gate  (MATMUL)     : Linear(H→1) 产 gate logits `sh_gate [S,B,1]`；输入是把完整
    #       hidden cast 到 router dtype(fp32) 的 [S,B,H] fp32 副本，存活到 gate 反向 → 建为其 saved
    #       激活 `sh_gate_hin_fp32`（任务 A，见下方 if 块内注释）。
    #   ② shared_gate_mul (ELEMENTWISE): `sh_o_gated = sigmoid(sh_gate)·sh_o`；合流 op(moe_add) 改吃它。
    # backward 生命周期（sigmoid·mul 反向）：d/d(sh_o)=sigmoid(sh_gate)、
    #   d/d(sh_gate)=sh_o·sigmoid'(sh_gate) → 须保存 shared 输出 sh_o 与门 logits sh_gate（[S,B,1] 极小；
    #   sigmoid 输出由 sh_gate 重算）。门权重字节可忽略但为参数守恒完整性建。
    # DSv3（moe_shared_gate=False）完全不建这两个 op → op 序列/saves/参数逐字节不变（golden 守卫）。
    if getattr(d, "moe_shared_gate", False):
        # 门权重 dtype=fp32（F3，closure-audit v2 §F3，2026-07-15）：真机 shared_experts_gate =
        # Dense(H→1, dtype=moe_router_dtype)（shared_experts.py:58-62），moe_router_dtype 默认
        # float32（transformer_config.py:1791-1796）——与本文件 router_w 同为 fp32。此前未指定
        # dtype_bytes → 解析后默认 2B/BF16（[H,1] 极小、字节可忽略，但 dtype 应正确）。
        sh_gate_w = TensorRef("sh_gate_w", ("H", "1"), is_weight=True, dtype_bytes=4)
        sh_gate_o = TensorRef("sh_gate", ("S", "B", "1"))
        sh_o_gated = TensorRef("sh_o_gated", ("S", "B", "H"))   # P0-1:随 sh_o 同口径(全量,见上注)
        # ── P1-01（任务 A，2026-07-15）：gate 输入的 **FP32 hidden cast 瞬态** ─────────────
        # 真机 `gate = sigmoid(shared_experts_gate(self.cast(hidden_states, router_dense_type)))`
        # （shared_experts.py:70-71 pynative / :82-83 training_graph）：gate Dense **之前**把
        # **完整 hidden** `[S,B,H]` cast 到 router dtype(fp32)，产生一份 `[S,B,H]` fp32 副本
        # （每 token 全 hidden，非小张量）。gate Dense(matmul) 反向需其输入（= 该 fp32 cast）算
        # 权重梯度 `dW = x^T @ dy` → 该 fp32 hidden **存活到 gate 反向** → 建为 `shared_gate` 的
        # saved 激活（dtype=4B，区别于 shared_fc1 已 save 的 bf16 `h1`；独立张量名，否则 act_live
        # 按名去重不计这份额外显存）。shard/cp 同 `hin_sh`（[S,B,H]{sp}，SP 下随序列切）。
        # gate off（DSv3）完全不进本分支 → 不建此瞬态、op 序列/saves 逐字节不变（golden 守卫）。
        sh_gate_hin_fp32 = TensorRef("sh_gate_hin_fp32", ("S", "B", "H"),
                                     shard={0: "sp"}, dtype_bytes=4)
        ops.append(OpSpec("shared_gate", OpType.MATMUL, [hin_sh, sh_gate_w], sh_gate_o,
                          params=[sh_gate_w], saves=[sh_gate_hin_fp32]))
        ops.append(OpSpec("shared_gate_mul", OpType.ELEMENTWISE, [sh_o, sh_gate_o], sh_o_gated,
                          saves=[sh_o, sh_gate_o]))
    return ops
