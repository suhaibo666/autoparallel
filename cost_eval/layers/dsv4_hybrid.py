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
**对所有 ratio（含 0/1）**都物化两个大张量到 loss 峰值，此前评估器漏建：per-head Query
RMSNorm 的输出（`deepseek_v4_hybrid_attention.py:242-245`）与分组输出 bmm 的输入
（`:286-291`），各 `[S,B,n_heads*v_head_dim]`。
  ⇒ 故 ratio 0/1 **不再**退化复用 `build_mla_attn_ops`，DSv4 全 ratio 自建 op 图。

> **⚠ 2026-07-29 dtype 订正（`docs/census_arbitration_2026-07-29.md` §1.1/§1.3）**：这两张
> **不是 fp32**，是 bf16。此前按 fp32 建（各 512 MiB/层）源于对权威快照
> （`mf-src-167`，即真机实跑那份）的**误读**：
>   - `:244-245` 逐字是 `q = q * mint.rsqrt(mint.mean(q * q, dim=-1, keepdim=True) + eps)`
>     —— 纯 `mint` 逐元素算术，**没有 `cast(..., float32)`**；`self.rms_norm = ops.rms_norm`
>     （`:158`）与 `self.q_rms_gamma`（`:159-161`）在 `construct` 里**从未被调用**。
>   - `:288-290` 源注释**显式拒绝** fp32 提升（"stays in the model dtype … A FP32 promotion
>     … is enough to create long-run optimizer drift"），`:291` 是 `bmm(cg, wo)` 直接吃 bf16。
> 纯 AST 抽图独立同意（`q` 256 MiB bf16 @`:240`、`cg` 256 MiB bf16 @`:286`）。
> 真机逐层实测（167/2026-07-29）r4 每层驻留 2355 MiB，而修前普查 3703.8 → 1.58×。

**融合 vs 非融合（`d.dsa_fused`，生产默认 True）**：`kv_gathered`/`attn_weights` 是 **unfused
小算子路径**（`csa.py:187` `unfused_compressed_sparse_attn`）才物化的中间量；**fused kernel**
`npu_sparse_attn_shared_kv` 走 scratch **不物化**（真机 15415 里查无此张量）。故 fused 时
sparse_attn **不 save** 它们（否则幻影多算 ~1312 MiB）。

**内存大头**（评估器建模的重点）：
  - `q_hnorm [S,B,n_heads*v_head_dim] bf16`（per-head Q-norm 输出→RoPE→kernel ctx `query`，
    256 MiB/层，:242-245 + `csa.py:674`）。
  - `cg [S,B,n_heads*v_head_dim] bf16`（分组输出 bmm 的激活操作数，256 MiB/层，:286-291）。
  - `index_scores [B,S,S] fp32` → O(S²)，indexer op 的 `bwd_scratch="4*B*S*S"`（indexer.py:227-236）。
  - `kv_gathered [B,S,topk,v_head_dim]` / `attn_weights [B,n_heads,S,topk]` —— **仅 unfused** save（csa.py:208/237）。
  - `compressed_kv [S//ratio, B, 1, v_head_dim]`（compressor.py:221）。

符号维（`DimTable` 字段）：`dsa_indexer_n_heads/dsa_indexer_head_dim/dsa_indexer_topk/
o_groups/o_lora_rank/csa_window_size`；`dsa_fused: bool`（融合开关，默认 True）。
"""
from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef, norm_kind_of
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
    q            = TensorRef("q",            ("S", "B", Q_OUT), shard={2: "tp"})  # :240 列并行(bf16)
    # per-head Q RMSNorm 输出（:242-245）**bf16**：`q * rsqrt(mean(q*q)+eps)` 无 fp32 cast。
    # 经 RoPE 后由 `csa.py:674` permute 成 BSND 副本进 kernel ctx（`query`）。
    q_hnorm      = TensorRef("q_hnorm",      ("S", "B", Q_OUT), shard={2: "tp"})
    # KV 侧激活（kv/kv_a_out）标 cp_kv=True（D-1 修正）：colossal 下 KV all-gather 到 full-S；
    # 其余 cp 算法随 body ÷cp。dsv4 走 cp=1（DSv4-align 锚点）→ 恒不生效，逐字节不变。
    kv           = TensorRef("kv",           ("S", "B", "v_head_dim"), cp_kv=True)  # :249 单共享头
    kv_a_out     = TensorRef("kv_a_out",     ("S", "B", "v_head_dim"), cp_kv=True)  # :250
    core_out     = TensorRef("core_out",     ("S", "B", Q_OUT))                  # :265 [sq,b,n,vd]
    # 分组输出 bmm 的激活操作数 cg（:286 permute→:291 bmm，**bf16**，源注释 :288-290 拒绝 fp32）
    cg           = TensorRef("cg",           ("S", "B", Q_OUT))
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

    # ── RoPE 反向保留对（rope_utils.py:186-187，**每次调用一对**）────────────────────
    # `output = add(mul(t, cos_), mul(t_rot, sin_))` 的两个 mul 各保留一个操作数 → 每次
    # `apply_rotary_emb` 留下 `t` 与 `t_rot` 两块 `[S,B,n,rot_dim]`，dtype = `config.rotary_dtype`
    # = **fp32**（`models/deepseek4/configuration_deepseek_v4.py:149` `rotary_dtype="fp32"`；
    # `parallel_core/transformer_config.py:166-167` 默认亦 float32；launcher yaml 未覆盖）。
    # 三次调用：q 前向（:260）、key 前向（:261）、core_out 逆向（:271）——**三次都走非融合分支**
    # （`mla_output_remove_interleaving=True` 使 `fused_interleaved_mla` 恒 False，
    # rope_utils.py:160-162）。此前普查只建了逆向那一对，前向两对**漏建**（欠读 130 MiB/层）。
    ROPE_LANE = "n_heads*qk_rope_head_dim"
    q_rope_f32  = TensorRef("q_rope_f32",  ("S", "B", ROPE_LANE), dtype_bytes=4)
    q_rope_rot  = TensorRef("q_rope_rot",  ("S", "B", ROPE_LANE), dtype_bytes=4)
    k_rope_f32  = TensorRef("k_rope_f32",  ("S", "B", "qk_rope_head_dim"), cp_kv=True, dtype_bytes=4)
    k_rope_rot  = TensorRef("k_rope_rot",  ("S", "B", "qk_rope_head_dim"), cp_kv=True, dtype_bytes=4)

    # ── base：Q 低秩 down→norm→up→**per-head norm(bf16)** + 单头 KV down→norm + RoPE ──
    ops = [
        OpSpec("ln1",           OpType.NORM,   [x],              ln1, params=[ln1_g], saves=[x],
               norm_kind=norm_kind_of(d)),                                                                  # 1 Pre-norm
        OpSpec("linear_q_down", OpType.MATMUL, [ln1, wq_down],   q_compressed,                   # 2 :234
               params=[wq_down], saves=[ln1]),
        OpSpec("q_a_norm",      OpType.NORM,   [q_compressed],   q_a_out,
               params=[qan_g], saves=[q_compressed], norm_kind=norm_kind_of(d)),                          # 3 :235
        OpSpec("linear_q_up",   OpType.MATMUL, [q_a_out, wq_up], q,                              # 4 :237
               params=[wq_up], saves=[q_a_out]),
        # 5 per-head Query "RMSNorm"（:242-245）**不是 layernorm 模块**：源逐字
        #   `q = q * mint.rsqrt(mint.mean(q * q, dim=-1, keepdim=True) + eps)` —— 两个 `mul` 的
        #   bprop 各保留 `q`（bf16 256 MiB/层，去重后一份）。故建成 ELEMENTWISE：`OpType.NORM`
        #   会让 `structure_mem._norm_save_names`/`_dt` 按 `layernorm_compute_dtype=fp32` 把 `q`
        #   抬成 4B（+256 MiB/层），而该 yaml 键管不到这段逐元素算术（仲裁 §1.2）。
        #   `q_rms_gamma`（:159-161）虽是真 Parameter，但 `construct` 从不用它 → 仍列 params
        #   （持久态口径不变，2 KiB）。
        OpSpec("q_hnorm",       OpType.ELEMENTWISE, [q],         q_hnorm, params=[qhn_g], saves=[q]),
        OpSpec("linear_kv",     OpType.MATMUL, [ln1, wkv],       kv, params=[wkv], saves=[ln1]), # 6 :249
        OpSpec("kv_a_norm",     OpType.NORM,   [kv],             kv_a_out,
               params=[kvan_g], saves=[kv], norm_kind=norm_kind_of(d)),                                   # 7 :250
        # 8 :260-261 前向 RoPE（q 与 key 各一次）——in-place 语义不动，但**每次调用保留 (t, t_rot)
        #   两块 fp32**（rope_utils.py:186-187，dtype=rotary_dtype=fp32）。此前 saves=[] 漏建。
        OpSpec("rope",          OpType.ROPE,   [q_hnorm],        q_hnorm,
               saves=[q_rope_f32, q_rope_rot, k_rope_f32, k_rope_rot]),
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
        # ── fused ctx 保存集（fused 自定义算子 SparseFlashMla 的 save_for_backward，交接 §11.5）──
        # fused 自定义算子 SparseFlashMla 每次调用 `ctx.save_for_backward` 存 **11 个张量**
        # （csa.py:224-235：query/ori_kv/cmp_kv/sparse_indices/query_index/key_index/weights/
        # cmp_residual/sinks/output/softmax_lse）。其中 query→q_hnorm、output→core_out、
        # cmp_kv→compressed_kv、sparse_indices→topk_indices 已建；此前缺 ori_kv(→kv_a_out)、
        # query_index/key_index/weights（indexer 内部 Q/K/头权重）、cmp_residual（压缩器残差）、
        # softmax_lse —— 仅 fused 分支补齐（unfused 走小算子路径、锚点 45557 已另标定，不动）。
        # 这些是 **save_for_backward** 张量：无重算(OFF)口径下真实驻留(锚点背书)。
        # ⚠ 2026-07-24 口径切换：**不再** pin_under_recompute。受控 A/B(E5/V2/E1b,报告§7.9)证
        # save_for_backward 在 MS use_reentrant=False 全重算下**正常释放** → 纯理论口径全重算只留
        # 层入口 checkpoint_input。真机每微批层驻留 ~1.9G 是「重算边界×在途深度」的框架释放缺口,
        # 显式暴露(见 report §八/FrameworkGapWarning),不再用经验 pin 顶到具名张量上。
        if fused:
            idx_query = TensorRef("idx_query", ("B", "S", "dsa_indexer_n_heads", "dsa_indexer_head_dim"))  # indexer.py:112 wq_b 输出
            idx_key = TensorRef("idx_key", ("B", "S", "dsa_indexer_head_dim"), cp_kv=True)   # indexer.py:126 单头 K
            # `weights` 实参是 `ops.cast(weights, mstype.float32)`（csa.py:696）→ **fp32**（仲裁 §1.8）
            idx_weights = TensorRef("idx_weights", ("B", "S", "dsa_indexer_n_heads"), dtype_bytes=4)  # csa.py:696
            # ⚠ 2026-07-29 shape 订正（仲裁 §1.6）：`cmp_residual` **不是** [S,B,coff·vd]。
            #   `csa.py:64-67`（`_prepare_sparse_flash_mla`，两个 fused _Function 的 forward 首行都调它）
            #   逐字 `cmp_residual = Tensor([int(ori_length) % kernel_cmp_ratio], dtype=mstype.int32)`
            #   —— **1 元素 int32 标量**（4 B，块对齐后 512 B）。此前按 8 MiB(r4)/4 MiB(r128) 记。
            cmp_residual = TensorRef("cmp_residual", ("1",), dtype_bytes=4)                   # csa.py:64-67
            softmax_lse = TensorRef("softmax_lse", ("B", "n_heads", "S"), dtype_bytes=4)     # csa.py:235 fp32
            # `sinks` = `ops.cast(attn_sink, float32)`（csa.py:687）——一份**独立 fp32 激活副本**，
            #   逐字进 ctx（csa.py:113/:224）；此前只有 params 里那个 Parameter（仲裁 §1.7，+256 B）。
            sinks = TensorRef("sinks", ("n_heads",), dtype_bytes=4)                          # csa.py:687
            _ctx_new = ([idx_query, idx_key, idx_weights] if enable_indexer else []) + [
                cmp_residual, sinks, softmax_lse]
            # `sparse_indices = mint.unsqueeze(topk_indices, dim=2)`（csa.py:61-62，视图）亦逐字进 ctx；
            #   同名张量已由上游 indexer 声明 saved、`structure_mem` 按名去重 → **净 0**（仲裁 §1.9）。
            #   源侧 `use_sparse_indices` 要求 `has_cmp and cmp_ratio == 4` → 仅 r4 有。
            _ctx_new += [topk_indices] if enable_indexer else []
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
                # ── indexer-KL **目标分布**分支：`detached=True`（2026-07-25，167 源码核验）───────
                # 真机 `csa.py:794-795` 调用是
                #     self.unfused_indexer_loss(index_scores, topk_indices_compressed,
                #                               ops.stop_gradient(query),
                #                               ops.stop_gradient(compressed_kv), mask=causal_mask)
                # 两个张量输入都被 `ops.stop_gradient` 截断；其内部（`indexer.py:350`
                # `attention_scores = matmul(permute(query), permute(key)) * softmax_scale`,
                # `:380` `softmax(cast(attention_scores, float32), dim=-1)`）整条链**没有一个输入
                # requires_grad** → MS 不为其建 autograd 节点 → **反向没有任何节点会读它** →
                # 它不是 saved 张量、是**纯瞬态**（单卡微基准实测 ~2 MiB/blk 驻留；若真被 save 会显
                # ~128 MiB/blk）。与之对照 `CSAIndexer` 自己的 `index_scores`（`indexer.py:245`
                # `bmm(q,k)` → relu → ×weights → sum）经 indexer 自身 params 携带梯度 → **不**
                # detached、照常 save（上方 `index_scores` TensorRef）。
                # ⚠ 该 flag **只被 `cost_eval/liveness/` 读**：`structure_mem.activation_saves`
                # 等桶路径不读它 → 本次改动对既有桶模型/全部锚点**逐字节无影响**（这两张仍留在
                # `saves` 里，桶模型行为不动）；liveness 交叉校验按 grad 可达性把它们排除。
                ukl1 = TensorRef("ukl1", ("B", "S", "dsa_indexer_n_heads", S_DIV_R),
                                 dtype_bytes=4, detached=True)
                ukl2 = TensorRef("ukl2", ("B", "S", "dsa_indexer_n_heads", S_DIV_R),
                                 dtype_bytes=4, detached=True)
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
        # fused：滑窗层同走 hyper_parallel 融合注意力 kernel 家族（自定义 _Function），其 ctx 集
        # {query, ori_kv, output, softmax_lse} 为 save_for_backward → 补 lse/ori_kv save（OFF 口径真实
        # 驻留）。2026-07-24 口径切换：**不再** pin（save_for_backward 全重算正常释放，报告§7.9）。
        if fused:
            sw_lse = TensorRef("softmax_lse", ("B", "n_heads", "S"), dtype_bytes=4)
            # r0 与 r128 **同走** `FusedSparseFlashMla`（8 项 ctx，csa.py:113）——`enable_indexer`
            # 只在 ratio==4 为真（csa.py:607）；r0 另因 `enable_compress = ratio > 0`（csa.py:593）
            # 使 `cmp_kv=None` 被 `if tensor is not None` 过滤。故 r0 的 ctx =
            # {query, ori_kv, cmp_residual, sinks, output, softmax_lse}。
            sw_cmp_res = TensorRef("cmp_residual", ("1",), dtype_bytes=4)      # csa.py:64-67 标量
            sw_sinks = TensorRef("sinks", ("n_heads",), dtype_bytes=4)         # csa.py:687 fp32 副本
            ops.append(OpSpec("core_attn", OpType.FLASH_ATTN, [q_hnorm, kv_a_out], core_out,
                              saves=[q_hnorm, kv_a_out, core_out, sw_cmp_res, sw_sinks, sw_lse],
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

    # ── core_out 逆 RoPE（deepseek_v4_hybrid_attention.py:271）────────────────────────
    # `_apply_forward_rope(core_out, freqs, 1.0, inverse=True)`,fused/unfused 两分支都有;
    # inverse 恒走非融合旋转（rope_utils.py:186-187）→ bprop 保留旋转 lane 的
    # `t`/`t_rot` 各一份 fp32 [S,B,n·rope_dim]（与前向 rope 同机理，见上方 ROPE_LANE）。
    #
    # ⚠ 2026-07-29 订正（仲裁 §1.4）：**去掉** `inv_rope_out`（输出 bf16 复本 [S,B,n·vd]，256 MiB）。
    #   `:215` 的 `cat([t_nope, t_pe])` 返回值,其**全部**下游是 `:278` reshape → `:286`
    #   reshape+permute,二者 VJP 分别是 reshape 与逆 permute,**都不保留输入**；bmm 真正保留的
    #   操作数是 `cg`（已单列）。抽取图逐 op 佐证：`:215` 的节点 saves=[]，下游首个 save 是 `cg`。
    #   此前依据是 2026-07-23「185 F 差分合账」（seq2048 补 128+2×16=160）——但那次拟合叠在
    #   一个已过读的普查之上（q/q_hnorm_fp32/cg_fp32 三处 dtype 错都在），故不能作为「被保留」的证据。
    inv_f32 = TensorRef("inv_rope_f32", ("S", "B", "n_heads*qk_rope_head_dim"), dtype_bytes=4)
    inv_rot = TensorRef("inv_rope_rot", ("S", "B", "n_heads*qk_rope_head_dim"), dtype_bytes=4)
    ops.append(OpSpec("inv_rope", OpType.ROPE, [core_out], core_out,
                      saves=[inv_f32, inv_rot]))

    # ── 分组输出：linear_o_group_proj（bmm，fp32 cg save）→ linear_proj → 残差 ────
    ops += [
        # grouped wo_a：bmm 的激活操作数是 `cg`（:286 permute 出的 **bf16** 副本，:291 bmm）——
        # 源注释 :288-290 显式拒绝 fp32 提升（仲裁 §1.3）。256 MiB/层。
        OpSpec("o_group_proj", OpType.MATMUL, [core_out, wo_group], o_group_out,
               params=[wo_group], saves=[cg]),
        OpSpec("o_proj",       OpType.MATMUL, [o_group_out, o_w], o, params=[o_w], saves=[o_group_out]),  # :290
        OpSpec("add1",         OpType.ELEMENTWISE, [o], h1, saves=[]),           # 残差 → h1（reshard SP）
    ]
    return ops
