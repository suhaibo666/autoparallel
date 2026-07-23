"""Task 2.1：DeepSeek-V4 hybrid 压缩注意力（DSA/CSA/HCA）op-builder。

`build_dsv4_hybrid_attn_ops(d, compress_ratio)` 按 **每层 compress_ratio** 分支
（设计 `specs/2026-07-01-unified-llm-modelspec-design.md` §7.3），忠实映射
mindformers `pynative/.../experimental_attention_variant/` 源码：

  - `deepseek_v4_hybrid_attention.py` — DSv4HybridSelfAttention（顶层 Q/KV/RoPE/分组输出）
  - `indexer.py`                      — CSAIndexer（top-k 选择，产 index_scores O(S²)）
  - `compressor.py`                   — Compressor（门控池化，产 compressed_kv S/ratio）
  - `csa.py`                          — CompressedSparseAttention（gather + 稀疏 attn）

| compress_ratio | 模式 | 分支结构 |
|---|---|---|
| 0 / 1 | 滑窗 | 只 sliding-window 稠密注意力（**仍走 DSv4 顶层**：per-head fp32 Q-norm + 分组输出）。|
| 4     | CSA  | +**indexer**（index_scores O(S²)）+ 压缩器(overlap, coff=2) + 稀疏注意力 + 分组输出。|
| 128   | HCA  | + 压缩器(non-overlap, coff=1) + dense 压缩位（无 top-k indexer）+ 稀疏注意力 + 分组输出。|

**关键修正（2026-07-03，真机 Profiler 定位 925 MiB 欠计）**：DSv4HybridSelfAttention 顶层
**对所有 ratio（含 0/1）**都物化两个 fp32 张量到 loss 峰值，此前评估器漏建/错 dtype：
  1. **per-head Query RMSNorm** —— `deepseek_v4_hybrid_attention.py:239-245`
     `q = rms_norm(cast(q, fp32), q_rms_gamma)` → fp32 `q[S,B,n_heads*q_head_dim]`（真机 256 MiB/层）。
  2. **分组输出 bmm 的 fp32 输入** —— `:277-283` `bmm(cast(cg, fp32), cast(wo, fp32))`
     → fp32 `cg[S,B,n_heads*v_head_dim]`（真机 256 MiB/层）。
  ⇒ 故 ratio 0/1 **不再**退化复用 `build_mla_attn_ops`，DSv4 全 ratio 自建 op 图。

**融合 vs 非融合（`d.dsa_fused`，生产默认 True）**：`kv_gathered`/`attn_weights` 是 **unfused
小算子路径**（`csa.py:187` `unfused_compressed_sparse_attn`）才物化的中间量；**fused kernel**
`npu_sparse_attn_shared_kv` 走 scratch **不物化**（真机 15415 里查无此张量）。故 fused 时
sparse_attn **不 save** 它们（否则幻影多算 ~1312 MiB）。

**内存大头**（评估器建模的重点）：
  - `q_hnorm_fp32 [S,B,n_heads*v_head_dim] fp32`（per-head Q-norm，256 MiB/层，:239-245）。
  - `cg_fp32 [S,B,n_heads*v_head_dim] fp32`（分组输出 fp32 输入，256 MiB/层，:277-283）。
  - `index_scores [B,S,S] fp32` → O(S²)，indexer op 的 `bwd_scratch="4*B*S*S"`（indexer.py:227-236）。
  - `kv_gathered [B,S,topk,v_head_dim]` / `attn_weights [B,n_heads,S,topk]` —— **仅 unfused** save（csa.py:208/237）。
  - `compressed_kv [S//ratio, B, 1, v_head_dim]`（compressor.py:221）。

符号维（`DimTable` 字段）：`dsa_indexer_n_heads/dsa_indexer_head_dim/dsa_indexer_topk/
o_groups/o_lora_rank/csa_window_size`；`dsa_fused: bool`（融合开关，默认 True）。
"""
from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef
from .attention import _fa_workspace

__all__ = ["build_dsv4_hybrid_attn_ops"]

# ── DSv4 符号维度表达式（与 DimTable 字段名一致，供 eval_expr 求值）──────────
# linear_q_up_proj 输出维：num_attention_heads * q_head_dim，其中 q_head_dim=v_head_dim
# （deepseek_v4_hybrid_attention.py:66 `self.q_head_dim = config.v_head_dim`；:113）。
Q_OUT = "n_heads*v_head_dim"
# 分组输出：linear_o_group_proj 输出 o_groups*o_lora_rank（:139-143 / :285）。
O_GROUP_OUT = "o_groups*o_lora_rank"
# linear_o_group_proj 每组 chunk = query_projection_size // o_groups（:139）。
O_CHUNK = "n_heads*v_head_dim//o_groups"
# 索引器 wq_b 输出维：index_n_heads*index_head_dim（indexer.py:112）。
IDX_QB_OUT = "dsa_indexer_n_heads*dsa_indexer_head_dim"


def build_dsv4_hybrid_attn_ops(d: DimTable, compress_ratio: int) -> list:
    """构造 DSv4 hybrid attention 段的 op 列表（按 compress_ratio 分支）。

    参数
    ----
    d : DimTable
        需含 MLA dims + dsv4 dims（dsa_indexer_* / o_groups / o_lora_rank）+ `dsa_fused`。
    compress_ratio : int
        本层压缩比：0/1=滑窗、4=CSA、128=HCA。

    返回的最后一个 op 输出为 ``h1``（shard={0:'sp'}），可与 FFN 尾拼接。
    """
    fused = getattr(d, "dsa_fused", True)          # 融合 DSA kernel（生产默认）：稀疏中间量不物化
    sparse = compress_ratio not in (0, 1)          # 4=CSA / 128=HCA 走稀疏；0/1 滑窗
    coff = 2 if compress_ratio == 4 else 1         # compressor.py:89-90 overlap→coff=2
    enable_indexer = compress_ratio == 4           # csa.py:324 仅 ratio==4 启 indexer
    CMP_PROJ_OUT = f"{coff}*v_head_dim"            # compressor.py:95 proj_out = coff*head_dim
    S_DIV_R = f"S//{compress_ratio}" if sparse else "S"   # compressor.py:199 n_compressed
    # 稀疏注意力 gather 的 KV 位置数（I3）：CSA→top-k；HCA→稠密 window+S//ratio（无 top-k）。
    TOPK_DIM = "dsa_indexer_topk" if enable_indexer else f"csa_window_size + {S_DIV_R}"

    # ── 激活张量 ──────────────────────────────────────────────────────────────
    x            = TensorRef("x",            ("S", "B", "H"),         shard={0: "sp"})
    ln1          = TensorRef("ln1",          ("S", "B", "H"))
    q_compressed = TensorRef("q_compressed", ("S", "B", "q_lora_rank"))          # :234
    q_a_out      = TensorRef("q_a_out",      ("S", "B", "q_lora_rank"))          # :235
    q            = TensorRef("q",            ("S", "B", Q_OUT), shard={2: "tp"})  # :237 列并行(bf16)
    # per-head Q RMSNorm 输出 fp32（:239-245）—— 真机 256 MiB/层，此前漏建
    q_hnorm      = TensorRef("q_hnorm_fp32", ("S", "B", Q_OUT), shard={2: "tp"}, dtype_bytes=4)
    # KV 侧激活（kv/kv_a_out）标 cp_kv=True（D-1 修正）：colossal 下 KV all-gather 到 full-S；
    # 其余 cp 算法随 body ÷cp。dsv4 走 cp=1（DSv4-align 锚点）→ 恒不生效，逐字节不变。
    kv           = TensorRef("kv",           ("S", "B", "v_head_dim"), cp_kv=True)  # :249 单共享头
    kv_a_out     = TensorRef("kv_a_out",     ("S", "B", "v_head_dim"), cp_kv=True)  # :250
    core_out     = TensorRef("core_out",     ("S", "B", Q_OUT))                  # :247 [sq,b,n,vd]
    # 分组输出 bmm 的 fp32 输入 cg（:277-283 cast(cg, fp32)）—— 真机 256 MiB/层，此前错 bf16
    cg_fp32      = TensorRef("cg_fp32",      ("S", "B", Q_OUT), dtype_bytes=4)
    o_group_out  = TensorRef("o_group_out",  ("S", "B", O_GROUP_OUT))            # :285
    o            = TensorRef("o",            ("S", "B", "H"))                    # :290
    h1           = TensorRef("h1",           ("S", "B", "H"), shard={0: "sp"})

    # ── 基础权重（所有 ratio 共有）─────────────────────────────────────────────
    wq_down  = TensorRef("wq_down",  ("H", "q_lora_rank"),   is_weight=True)                    # :93-101
    wq_up    = TensorRef("wq_up",    ("q_lora_rank", Q_OUT), shard={1: "tp"}, is_weight=True)   # :110-118 列并行
    wkv      = TensorRef("wkv",      ("H", "v_head_dim"),    is_weight=True)                    # :120-128
    wo_group = TensorRef("wo_group", (O_GROUP_OUT, O_CHUNK), is_weight=True)                    # :140-143 linear_o_group_proj
    o_w      = TensorRef("o_w",      (O_GROUP_OUT, "H"),     is_weight=True)                    # :145-153 linear_proj
    # norm gamma（P1-01，fp32 独立 wrap，parallelize.py:414-461/:1140-1142）；q_hnorm 为
    # per-head RMSNorm → gamma = 每头维（Q_OUT//n_heads = v_head_dim）
    ln1_g  = TensorRef("ln1_g",       ("H",),            is_weight=True, dtype_bytes=4)
    qan_g  = TensorRef("q_a_norm_g",  ("q_lora_rank",),  is_weight=True, dtype_bytes=4)
    qhn_g  = TensorRef("q_hnorm_g",   ("v_head_dim",),   is_weight=True, dtype_bytes=4)
    kvan_g = TensorRef("kv_a_norm_g", ("v_head_dim",),   is_weight=True, dtype_bytes=4)

    # ── base：Q 低秩 down→norm→up→**per-head fp32 norm** + 单头 KV down→norm + RoPE ──
    ops = [
        OpSpec("ln1",           OpType.NORM,   [x],              ln1, params=[ln1_g], saves=[x]), # 1 Pre-norm
        OpSpec("linear_q_down", OpType.MATMUL, [ln1, wq_down],   q_compressed,                   # 2 :234
               params=[wq_down], saves=[ln1]),
        OpSpec("q_a_norm",      OpType.NORM,   [q_compressed],   q_a_out,
               params=[qan_g], saves=[q_compressed]),                                            # 3 :235
        OpSpec("linear_q_up",   OpType.MATMUL, [q_a_out, wq_up], q,                              # 4 :237
               params=[wq_up], saves=[q_a_out]),
        # 5 per-head Query RMSNorm（:239-245）：rms_norm 反向需 bf16 输入 q（128 MiB/层，saved）；
        #   fp32 输出 q_hnorm 由下游 attention save（QK 反向需 Q，256 MiB/层）——两者共存至 loss 峰值。
        OpSpec("q_hnorm",       OpType.NORM,   [q],              q_hnorm, params=[qhn_g], saves=[q]),
        OpSpec("linear_kv",     OpType.MATMUL, [ln1, wkv],       kv, params=[wkv], saves=[ln1]), # 6 :249
        OpSpec("kv_a_norm",     OpType.NORM,   [kv],             kv_a_out,
               params=[kvan_g], saves=[kv]),                                                     # 7 :250
        OpSpec("rope",          OpType.ROPE,   [q_hnorm],        q_hnorm, saves=[]),             # 8 :256-261 in-place
    ]

    # ── core attention ────────────────────────────────────────────────────────
    if sparse:
        # compressed_kv / kv_gathered 亦 KV 侧（cp_kv=True，D-1 修正）：colossal 下 all-gather 到 full-S。
        compressed_kv = TensorRef("compressed_kv", (S_DIV_R, "B", "1", "v_head_dim"), cp_kv=True)  # :221
        kv_gathered   = TensorRef("kv_gathered",  ("B", "S", TOPK_DIM, "v_head_dim"), cp_kv=True)  # csa.py:485
        # kv_g = **fp32 副本** of kv_gathered（csa.py:490 `ops.cast(kv_gathered, float32)`）——unfused 小算子
        #   路径把 gather 后的 KV 升 fp32 做打分/加权和,反向常驻。是 unfused 最大瞬态（seq²·topk·vd 级）。
        kv_g_fp32     = TensorRef("kv_g_fp32",    ("B", "S", TOPK_DIM, "v_head_dim"), cp_kv=True, dtype_bytes=4)  # csa.py:490
        # attn_weights = softmax 输出 **fp32**（csa.py:519-534 exp/sum/除全在 fp32）。
        attn_weights  = TensorRef("attn_weights", ("B", "n_heads", "S", TOPK_DIM), dtype_bytes=4)  # csa.py:531
        topk_indices  = TensorRef("topk_indices", ("B", "S", TOPK_DIM), dtype_bytes=4)           # :241 int32
        idx_wq_b  = TensorRef("idx_wq_b",  ("q_lora_rank", IDX_QB_OUT),         is_weight=True)   # indexer.py:109-117
        idx_wproj = TensorRef("idx_wproj", ("H", "dsa_indexer_n_heads"),        is_weight=True)   # indexer.py:126-134
        cmp_wkv   = TensorRef("cmp_wkv",   ("H", CMP_PROJ_OUT),                 is_weight=True)    # compressor.py:97-105
        cmp_wgate = TensorRef("cmp_wgate", ("H", CMP_PROJ_OUT),                 is_weight=True)    # compressor.py:107-115
        cmp_ape   = TensorRef("cmp_ape",   (str(compress_ratio), CMP_PROJ_OUT), is_weight=True, dtype_bytes=4)  # :117-120 fp32
        attn_sink = TensorRef("attn_sink", ("n_heads",), is_weight=True, dtype_bytes=4)          # csa.py:304-308
        # indexer（仅 CSA ratio==4）。**fused**：`npu_lightning_indexer` 只返 topk_scores、全阵 [B,S,n,S] 走
        #   kernel scratch 不物化（真机 15415 查无）→ 保持旧 `4*B*S*S` bwd_scratch 口径,fused 锚点不动。
        # **unfused**：`index_scores = bmm(q,k).reshape(b, sq, n_idx, sk)`（indexer.py:245-246）**前向物化**
        #   [B,S,dsa_indexer_n_heads,S]、为 indexer KL loss 反向常驻（dsa_indexer_loss.py）——**含 n_idx_heads
        #   维**（此前 [B,S,S] 漏了 ×n_idx=64,是 unfused 大欠算之一）→ 前向 save,不再当小 scratch。
        if enable_indexer:
            if fused:
                ops.append(OpSpec("indexer", OpType.MATMUL, [ln1, q_a_out], topk_indices,
                                  params=[idx_wq_b, idx_wproj], saves=[topk_indices], bwd_scratch="4*B*S*S"))
            else:
                index_scores = TensorRef("index_scores", ("B", "S", "dsa_indexer_n_heads", "S"))  # indexer.py:246 bf16 bmm
                ops.append(OpSpec("indexer", OpType.MATMUL, [ln1, q_a_out], topk_indices,
                                  params=[idx_wq_b, idx_wproj], saves=[topk_indices, index_scores]))
        # compressor：门控池化 → compressed_kv [S//ratio,B,1,vd]（compressor.py）
        ops.append(OpSpec("compressor", OpType.MATMUL, [ln1], compressed_kv,
                          params=[cmp_wkv, cmp_wgate, cmp_ape], saves=[compressed_kv]))
        # sparse attention：save Q(=q_hnorm fp32)+O(core_out)；fused kernel 不物化 kv_gathered/
        # attn_weights（走 scratch）；unfused 才 save 它们。
        #
        # ── fused ctx 保存集（2026-07-22，185 pp4+全重算锚点定标，交接 §11.5）────────────────
        # fused 自定义算子 SparseFlashMla 每次调用 `ctx.save_for_backward` 存 **11 个张量**
        # （csa.py:224-235：query/ori_kv/cmp_kv/sparse_indices/query_index/key_index/weights/
        # cmp_residual/sinks/output/softmax_lse）。其中 query→q_hnorm、output→core_out、
        # cmp_kv→compressed_kv、sparse_indices→topk_indices 已建；此前缺 ori_kv(→kv_a_out)、
        # query_index/key_index/weights（indexer 内部 Q/K/头权重）、cmp_residual（压缩器残差）、
        # softmax_lse —— 仅 fused 分支补齐（unfused 走小算子路径、锚点 45557 已另标定，不动）。
        # 全部 ctx 张量标 `pin_under_recompute`：MindSpore use_reentrant=False 全重算不释放
        # 自定义 _Function 的 ctx 状态（真机 ON−OFF 净省仅 6.2GB 证实）→ 全重算下仍逐微批常驻。
        if fused:
            idx_query = TensorRef("idx_query", ("B", "S", "dsa_indexer_n_heads", "dsa_indexer_head_dim"))  # indexer.py:112 wq_b 输出
            idx_key = TensorRef("idx_key", ("B", "S", "dsa_indexer_head_dim"), cp_kv=True)   # indexer.py:126 单头 K
            idx_weights = TensorRef("idx_weights", ("B", "S", "dsa_indexer_n_heads"))        # indexer.py:134 头权重
            cmp_residual = TensorRef("cmp_residual", ("S", "B", CMP_PROJ_OUT), cp_kv=True)   # compressor 残差（coff·vd）
            softmax_lse = TensorRef("softmax_lse", ("B", "n_heads", "S"), dtype_bytes=4)     # csa.py:235 fp32
            _ctx_new = ([idx_query, idx_key, idx_weights] if enable_indexer else []) + [cmp_residual, softmax_lse]
            _ctx_all = ([q_hnorm, kv_a_out, compressed_kv, core_out]
                        + ([topk_indices] if enable_indexer else []) + _ctx_new)
            for _t in _ctx_all:
                _t.pin_under_recompute = True                  # ctx 状态：全重算免疫（不释放）
            sparse_saves = [q_hnorm, kv_a_out, compressed_kv, core_out] + _ctx_new
        else:
            # ── unfused 反向图 fp32 复本群（2026-07-23,185 U1 相位合账;DAG 审计嫌疑①②）────────
            # 真机 `unfused_compressed_sparse_attn`（csa.py:187-250）逐 op 的 bprop 持有其输入/输出:
            #   q fp32 cast(:213) + q_bm 重排复本(:218) → 2×[B,n,S,vd] fp32;
            #   kv_bm 重排复本(:219)               → 又一份 [B,S,topk,vd] fp32(kv_g 之外);
            #   scores(:220) + 其 permute(:221) + exp_scores(:235) + aw_bm(:241) → 4×[B,n,S,topk] fp32;
            #   output fp32 + permute(:243-249)     → 2×[B,n,S,vd] fp32。
            # KL 链(仅 r4,csa.py:503-516 unfused_indexer_loss 对全阵 softmax/log fp32,training 每步):
            #   2×[B,S,n_idx,S/r] fp32。185 U1 相位:fwd 末每层驻留实测均值 4990 MiB,修前 census
            #   均值 2827 → 缺口正是本复本群(合账表见 probe 报告)。
            uq_f32   = TensorRef("uq_f32",   ("B", "n_heads", "S", "v_head_dim"), dtype_bytes=4)
            uq_bm    = TensorRef("uq_bm",    ("B", "n_heads", "S", "v_head_dim"), dtype_bytes=4)
            ukv_bm   = TensorRef("ukv_bm",   ("B", "S", TOPK_DIM, "v_head_dim"), cp_kv=True, dtype_bytes=4)
            uscore1  = TensorRef("uscore1",  ("B", "n_heads", "S", TOPK_DIM), dtype_bytes=4)
            uscore2  = TensorRef("uscore2",  ("B", "n_heads", "S", TOPK_DIM), dtype_bytes=4)
            uexp     = TensorRef("uexp",     ("B", "n_heads", "S", TOPK_DIM), dtype_bytes=4)
            uaw_bm   = TensorRef("uaw_bm",   ("B", "n_heads", "S", TOPK_DIM), dtype_bytes=4)
            uout_f32 = TensorRef("uout_f32", ("B", "n_heads", "S", "v_head_dim"), dtype_bytes=4)
            uout_pm  = TensorRef("uout_pm",  ("B", "n_heads", "S", "v_head_dim"), dtype_bytes=4)
            _ucopies = [uq_f32, uq_bm, ukv_bm, uscore1, uscore2, uexp, uaw_bm, uout_f32, uout_pm]
            if enable_indexer:
                ukl1 = TensorRef("ukl1", ("B", "S", "dsa_indexer_n_heads", S_DIV_R), dtype_bytes=4)
                ukl2 = TensorRef("ukl2", ("B", "S", "dsa_indexer_n_heads", S_DIV_R), dtype_bytes=4)
                _ucopies += [ukl1, ukl2]
            sparse_saves = [q_hnorm, kv_gathered, kv_g_fp32, attn_weights, core_out] + _ucopies
        # inputs 含 topk_indices（仅 CSA）= 稀疏选 KV 的**数据流依赖**（indexer→sparse_attn 边;
        # 2026-07-11 补边:此前缺此边致 indexer 成 op 图孤立叶节点。saves/字节不变）。
        sparse_ins = ([q_hnorm, kv_a_out, compressed_kv, topk_indices] if enable_indexer
                      else [q_hnorm, kv_a_out, compressed_kv])
        ops.append(OpSpec("sparse_attn", OpType.FLASH_ATTN, sparse_ins, core_out,
                          params=[attn_sink], saves=sparse_saves, workspace_ref=_fa_workspace()))
    else:
        # 滑窗（ratio 0/1）：纯 flash（sliding-window），saves Q(=q_hnorm)/O(core_out)+lse（[S,S] 从不物化，§7.4）
        # fused（2026-07-22）：滑窗层同走 hyper_parallel 融合注意力 kernel 家族（自定义 _Function），
        # 其 ctx 集 {query, ori_kv, output, softmax_lse} 同样全重算不释放（185 pp4 锚点:含 r0 层的
        # stage0/1/3 欠估仅在 r0 也 pin 时闭合）→ 补 lse/ori_kv save + 全部标 pin。
        if fused:
            sw_lse = TensorRef("softmax_lse", ("B", "n_heads", "S"), dtype_bytes=4)
            for _t in (q_hnorm, kv_a_out, core_out, sw_lse):
                _t.pin_under_recompute = True
            ops.append(OpSpec("core_attn", OpType.FLASH_ATTN, [q_hnorm, kv_a_out], core_out,
                              saves=[q_hnorm, kv_a_out, core_out, sw_lse],
                              workspace_ref=_fa_workspace()))
        else:
            # unfused 滑窗（r0/1）真机同样走 `_construct_naive`（csa.py:438-449:ratio==0 →
            # kv_full=ori_kv、topk=window_idxs → 仍进 unfused_compressed_sparse_attn 的
            # gather+fp32 复本链,TOPK=csa_window_size）——修前按纯 flash 建,漏整个 naive 链
            # (185 U1 相位:r0 层也在 4990 均值口径内)。gather 家族+复本群按 window 尺寸补齐。
            W0 = "csa_window_size"
            r0_kvg  = TensorRef("kv_gathered", ("B", "S", W0, "v_head_dim"), cp_kv=True)
            r0_kvg32 = TensorRef("kv_g_fp32", ("B", "S", W0, "v_head_dim"), cp_kv=True, dtype_bytes=4)
            r0_aw   = TensorRef("attn_weights", ("B", "n_heads", "S", W0), dtype_bytes=4)
            r0_q32  = TensorRef("uq_f32", ("B", "n_heads", "S", "v_head_dim"), dtype_bytes=4)
            r0_qbm  = TensorRef("uq_bm", ("B", "n_heads", "S", "v_head_dim"), dtype_bytes=4)
            r0_kvbm = TensorRef("ukv_bm", ("B", "S", W0, "v_head_dim"), cp_kv=True, dtype_bytes=4)
            r0_sc1  = TensorRef("uscore1", ("B", "n_heads", "S", W0), dtype_bytes=4)
            r0_sc2  = TensorRef("uscore2", ("B", "n_heads", "S", W0), dtype_bytes=4)
            r0_exp  = TensorRef("uexp", ("B", "n_heads", "S", W0), dtype_bytes=4)
            r0_awbm = TensorRef("uaw_bm", ("B", "n_heads", "S", W0), dtype_bytes=4)
            r0_o32  = TensorRef("uout_f32", ("B", "n_heads", "S", "v_head_dim"), dtype_bytes=4)
            r0_opm  = TensorRef("uout_pm", ("B", "n_heads", "S", "v_head_dim"), dtype_bytes=4)
            ops.append(OpSpec("core_attn", OpType.FLASH_ATTN, [q_hnorm, kv_a_out], core_out,
                              saves=[q_hnorm, core_out, r0_kvg, r0_kvg32, r0_aw,
                                     r0_q32, r0_qbm, r0_kvbm, r0_sc1, r0_sc2, r0_exp,
                                     r0_awbm, r0_o32, r0_opm],
                              workspace_ref=_fa_workspace()))

    # ── core_out 逆 RoPE（2026-07-23,185 F 差分合账;DAG 审计嫌疑③）────────────────────
    # 真机对 core attention 输出做 inverse-RoPE（deepseek_v4_hybrid_attention.py:277
    # `_apply_forward_rope(core_out, freqs, inverse=True)`,fused/unfused 两分支都有;inverse
    # 恒走非融合旋转 rope_utils.py:182）。bprop 保留:输出 bf16 复本 [S,B,n·vd] + 旋转 lane 的
    # fp32 cast/rotate_half 各一份 [S,B,n·rope_dim]。185 F 差分:fused 每层真机 3109 vs 修前
    # sim 2960(+149)——本成员 seq2048 记账 128+2×16=160,闭合到 +1.4%。
    inv_out = TensorRef("inv_rope_out", ("S", "B", Q_OUT))
    inv_f32 = TensorRef("inv_rope_f32", ("S", "B", "n_heads*qk_rope_head_dim"), dtype_bytes=4)
    inv_rot = TensorRef("inv_rope_rot", ("S", "B", "n_heads*qk_rope_head_dim"), dtype_bytes=4)
    ops.append(OpSpec("inv_rope", OpType.ROPE, [core_out], core_out,
                      saves=[inv_out, inv_f32, inv_rot]))

    # ── 分组输出：linear_o_group_proj（bmm，fp32 cg save）→ linear_proj → 残差 ────
    ops += [
        # grouped wo_a：bmm 对 core_out cast fp32（cg_fp32 saved，:274-287）—— 真机 256 MiB/层
        OpSpec("o_group_proj", OpType.MATMUL, [core_out, wo_group], o_group_out,
               params=[wo_group], saves=[cg_fp32]),
        OpSpec("o_proj",       OpType.MATMUL, [o_group_out, o_w], o, params=[o_w], saves=[o_group_out]),  # :290
        OpSpec("add1",         OpType.ELEMENTWISE, [o], h1, saves=[]),           # 残差 → h1（reshard SP）
    ]
    return ops
