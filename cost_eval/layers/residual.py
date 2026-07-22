"""Task 2.2：mHC 残差包装器（设计 §9）。

`residual_variant="mhc"` 时，decoder 的残差不再是普通 `[S,B,H]` 加法，而是把 hidden
打包成 **n 条残差流** `[S,B,n*H]`（n=`num_residual_streams`）在整栈流动，且每个 decoder 层
在 attention 前、ffn 前各插入一个 **HyperConnection 模块**。本文件提供：

- `build_hyper_connection_ops(prefix, d)` —— 单个 HC 模块的 op（RMSNorm(n·H) + mapping_proj
  + sinkhorn → h_res `[S,B,n,n]`）。
- `mhc_wrap(body_ops, n_streams, d)` —— 把一个 decoder body（attn 段 + ffn 段的 op 列表）
  包装成 mHC：(a) 前插 attn_hc、在 attn/ffn 边界插 ffn_hc；(b) 把所有**残差承载张量**
  （SP 分布的 `[S,B,H]`，即 x/h1/h2）的 H 维乘以 `num_residual_streams` → `[S,B,n*H]`。

源码忠实映射（mindformers pynative）：
  - `hyper_connection.py` `HyperConnectionModule.construct`（:178-233）——
    RMSNorm(fp32, n·H, :194-198) → mapping_proj `Linear(n*H → dim)`（:143-150，
    `dim = n + n + n*n`，:141）→ sinkhorn → `h_res [s,b,n,n]`（:222-223）。
    `rms_weight` 为 fp32 buffer（:157-161），`mapping_proj.weight` 为 fp32（params_dtype=float32）。
  - `transformer_block.py` `expand/collapse_hyper_connection_streams`（:18-38）——
    block 入口 `[s,b,h]→[s,b,n*h]`（tile+reshape）、出口 mean 回 `[s,b,h]`。
  - `transformer_layer.py` `HyperConnectionTransformerLayer.construct`（:287-332）——
    每层 `attn_hc`（attention 前）+ `ffn_hc`（ffn 前）两个 HC 模块，
    `output_cell` 做 `h_res @ streams + h_post * sublayer_out`（残差流更新）。

> 说明：`hyper_connection.py:141` 源码为 `dim = n + n + n*n`（= `2n+n²`；h_pre[n]/h_post[n]/
> h_res[n²]）；设计 §9 记作 `3n+n²`，此处以**源码为准**取 `2n+n²`。mapping_proj 权重量级很小，
> 该差异对内存可忽略。mHC 的 `act_live ×n` 未真机验证（设计 §14）。
"""
from __future__ import annotations

from ..model_spec import DimTable, OpSpec, OpType, TensorRef

__all__ = [
    "build_hyper_connection_ops", "mhc_wrap",
    "build_hc_expand_op", "build_hc_collapse_op",
]

# ── mHC 符号维度表达式（与 DimTable 字段名一致，供 eval_expr 求值）──────────────
# 打包残差流维 n*H（HyperConnectionModule 输入/RMSNorm 维，hyper_connection.py:158）
NH = "num_residual_streams*H"
# mapping_proj 输出维 dim = n + n + n*n（hyper_connection.py:141）
PROJ_OUT = "2*num_residual_streams + num_residual_streams*num_residual_streams"


def _is_residual_carrier(t: TensorRef) -> bool:
    """残差承载张量：SP 分布的 `[S,B,H]`（block 入/出的 hidden，如 x/h1/h2）。

    这些张量在 mHC 下打包为 n 条流 → `[S,B,n*H]`。sublayer 内部张量（o/o2 为 tp-partial、
    ln1/ln2 为归一化中间量）**不是**残差承载，保持 `[S,B,H]`。
    """
    return t.shape == ("S", "B", "H") and t.shard == {0: "sp"}


def _scale_ref(t: TensorRef) -> TensorRef:
    """残差承载张量 → H 维乘以 num_residual_streams（`[S,B,H]→[S,B,n*H]`），其余原样返回。

    P1-03（2026-07-14）：放大后**重命名** `{name}_xn`——此前保留原名导致同一 resolved layer 内
    同名张量两种 numel（MTP 层 `x` 同时 =H 与 =nH），而 structure_mem/ShapeEval.produced 按名
    去重 → 首见 size 覆盖后续语义、字节错算。段内边由 scaled op 集合内的一致重命名保持；
    ShapeEval.resolve 现有同名异 numel fail-loud 不变量防回潮。"""
    if not _is_residual_carrier(t):
        return t
    return TensorRef(f"{t.name}_xn", ("S", "B", NH), shard=dict(t.shard),
                     is_weight=t.is_weight, partial=t.partial, dtype_bytes=t.dtype_bytes)


def _scale_op(op: OpSpec) -> OpSpec:
    """对一个 op 的 inputs/output/params/saves 施加 `_scale_ref`（残差承载 ×n）。"""
    return OpSpec(
        op.name, op.type,
        [_scale_ref(t) for t in op.inputs],
        _scale_ref(op.output),
        params=[_scale_ref(t) for t in op.params],
        saves=[_scale_ref(t) for t in op.saves],
        workspace=op.workspace, bwd_scratch=op.bwd_scratch, attrs=dict(op.attrs),
    )


def build_hyper_connection_ops(prefix: str, d: DimTable, fused_ctx_pin: bool = False) -> list:
    """单个 mHC HyperConnection 模块的 op 列表（3 op，源：`hyper_connection.py:178-233`）。

    prefix ∈ {"attn","ffn"}（`transformer_layer.py:278-285` 的 attn_hc / ffn_hc）。

    fused_ctx_pin（2026-07-22，185 pp4+全重算锚点定标）：fused mHC（fork `use_fused_mhc`，
    hyper_parallel 自定义算子，与 `apply_dsa_kernel_fusion` 同开同关——交接 §7 真机切换口径）
    的 `ctx.save_for_backward` 状态在 MindSpore use_reentrant=False 全重算下**不释放**（与
    fused SparseFlashMla 同机制）→ hc_norm / h_res 标 `pin_under_recompute`。默认 False =
    非 fused mHC（普通小算子路径，全重算正常释放），全部既有 spec 逐字节不变。

    op 序列（吃打包残差流 `streams [S,B,n*H]`）：
      1. `{prefix}_hc_norm`（NORM）: RMSNorm(n·H, fp32) → `hc_norm [S,B,n*H]` fp32（saved，
         mHC 的主激活大头：×n 且 fp32）。权重 `rms_weight [n*H]` fp32 buffer。
      2. `{prefix}_hc_mapping_proj`（MATMUL）: `Linear(n*H → dim)`，权重 fp32 `[n*H, dim]`。
      3. `{prefix}_hc_sinkhorn`（ELEMENTWISE）: sinkhorn 投影 → `h_res [S,B,n,n]`（saved，
         供 output_cell 反向 `h_res @ streams`）。

    aggregate（`h_pre @ streams → [S,B,H]`，:228-231）与 output_cell（:319/:329）不单列 op：
    前者产生的 `[S,B,H]` aggregated 即 sublayer 的输入（由 body 的 ln1 承接），后者的残差流
    更新已由 body 尾部残差承载张量（×n）体现。
    """
    streams = TensorRef(f"{prefix}_streams", ("S", "B", NH), shard={0: "sp"})
    # RMSNorm 在 fp32 下计算（hyper_connection.py:194-198 cast float32）
    hc_norm = TensorRef(f"{prefix}_hc_norm", ("S", "B", NH), dtype_bytes=4)
    h_proj = TensorRef(f"{prefix}_hc_proj", ("S", "B", PROJ_OUT), dtype_bytes=4)
    h_res = TensorRef(f"{prefix}_h_res", ("S", "B", "num_residual_streams", "num_residual_streams"))

    if fused_ctx_pin:
        # fused mHC ctx 持有：RMSNorm fp32 输出（反向 mapping_proj 需之）+ h_res（output_cell
        # 反向 `h_res @ streams` 需之）。流输入(streams=body 承载张量 x_xn/h1_xn)在 mhc_wrap 标。
        hc_norm.pin_under_recompute = True
        h_res.pin_under_recompute = True

    # rms_weight：fp32 buffer（requires_grad=False），量级 n*H（hyper_connection.py:157-161）
    rms_w = TensorRef(f"{prefix}_hc_rms_w", (NH,), is_weight=True, dtype_bytes=4)
    # mapping_proj.weight：fp32（params_dtype=float32），[n*H, dim]（hyper_connection.py:143-150）
    proj_w = TensorRef(f"{prefix}_hc_proj_w", (NH, PROJ_OUT), is_weight=True, dtype_bytes=4)

    return [
        # 1. RMSNorm(n·H) → fp32 归一化流（saved：mapping_proj 反向需其输入）
        OpSpec(f"{prefix}_hc_norm", OpType.NORM, [streams], hc_norm,
               params=[rms_w], saves=[hc_norm]),
        # 2. mapping_proj（n*H → dim = 2n+n²）
        OpSpec(f"{prefix}_hc_mapping_proj", OpType.MATMUL, [hc_norm, proj_w], h_proj,
               params=[proj_w], saves=[]),
        # 3. sinkhorn → h_res [S,B,n,n]（saved：output_cell 反向 h_res @ streams）。
        #    **反向瞬态（bwd_scratch，公式，非常数）**：HyperConnectionOutputCell
        #    （hyper_connection.py:87-112）前向 `new_streams = h_res @ x_streams + h_post*sublayer_out`
        #    产 [s,b,n,H]；其反向物化 ×n 打包残差流梯度 `grad_x_streams [s,b,n*H]` +
        #    重建 `res_part [s,b,n*H]`（`self.matmul(h_res, x_streams)`，:105），二者皆 compute
        #    dtype(bf16=2B) → `bwd_scratch = 2 张量 × 2B × S·B·(n·H) = 4·S·B·n·H`。**∝ num_residual_streams**
        #    （n=1 plain 时退化为普通 [S,B,H] 残差反向，四分之一）。仅在该 mHC 层反向事件计入。
        OpSpec(f"{prefix}_hc_sinkhorn", OpType.ELEMENTWISE, [h_proj], h_res, saves=[h_res],
               bwd_scratch="4*S*B*num_residual_streams*H"),
    ]


def build_hc_expand_op(d: DimTable) -> OpSpec:
    """block 入口 `expand`：hidden `[S,B,H] → [S,B,n·H]`（源：`transformer_block.py:18-38`
    `expand_hyper_connection_streams`，tile+reshape）。

    装配器在 embedding 之后插入本 op，把 SP 分布的 `emb_out [S,B,H]` 打包为 n 条残差流
    `hc_streams [S,B,n·H]`，供 mHC decoder 栈流动（设计 §9 stack entry）。ELEMENTWISE、无 saves
    （反向即 collapse-mean，无大激活）。
    """
    emb = TensorRef("emb_out", ("S", "B", "H"), shard={0: "sp"})
    streams = TensorRef("hc_streams", ("S", "B", NH), shard={0: "sp"})
    return OpSpec("hc_expand", OpType.ELEMENTWISE, [emb], streams, saves=[])


def build_hc_collapse_op(d: DimTable) -> OpSpec:
    """block 出口 `collapse`：`[S,B,n·H] → [S,B,H]`（源：`transformer_block.py:18-38`
    `collapse_hyper_connection_streams`，mean over streams）。

    装配器在 lm_head 之前插入本 op，把 n 条残差流 `hc_streams [S,B,n·H]` 规约回
    `h_final [S,B,H]`（lm_head 段的输入，设计 §9 stack exit）。ELEMENTWISE、无 saves。
    """
    streams = TensorRef("hc_streams", ("S", "B", NH), shard={0: "sp"})
    h_final = TensorRef("h_final", ("S", "B", "H"), shard={0: "sp"})
    return OpSpec("hc_collapse", OpType.ELEMENTWISE, [streams], h_final, saves=[])


def _ffn_split_index(body_ops: list) -> int:
    """attn 段 / ffn 段边界：第一个输出为残差承载张量的 op（attn 的残差 add，产 h1）之后。

    attn 段以残差 add（输出 SP `[S,B,H]` 的 h1）结尾；其后即 ffn 段。找不到则整段视为 attn。
    """
    for i, op in enumerate(body_ops):
        if _is_residual_carrier(op.output):
            return i + 1
    return len(body_ops)


def mhc_wrap(body_ops: list, n_streams: int, d: DimTable,
             fused_ctx_pin: bool = False) -> list:
    """把 decoder body 包装成 mHC（设计 §9）。

    - `n_streams <= 1`：**no-op**，原样返回 `body_ops`（plain 残差，无 mHC）。
    - 否则：
        (a) 前插 `attn_hc` 3 op（attention 前），在 attn/ffn 边界插 `ffn_hc` 3 op；
        (b) body 内所有**残差承载张量**（SP `[S,B,H]`）→ `[S,B,n*H]`（H 维 ×num_residual_streams）。

    参数
    ----
    body_ops : list[OpSpec]
        一个 decoder 层的 op 列表（attn 段 + ffn 段），如
        `build_gqa_attn_ops(d) + build_dense_ffn_ops(d)`。
    n_streams : int
        残差流数 n；仅用于 no-op 门（`<=1` 直接返回）。×n 的符号量来自 `d.num_residual_streams`。
    d : DimTable
        需含 `num_residual_streams`（与 `n_streams` 一致）。
    fused_ctx_pin : bool
        fused mHC（fork `use_fused_mhc` 自定义算子）→ HC 模块 ctx 状态（hc_norm/h_res +
        流输入 x_xn/h1_xn）标 `pin_under_recompute`（全重算不释放，2026-07-22 185 pp4 锚点；
        与 dsv4 `apply_dsa_kernel_fusion` 同开同关）。默认 False 全部既有 spec 逐字节不变。
    """
    if n_streams <= 1:
        return body_ops

    split = _ffn_split_index(body_ops)
    scaled = [_scale_op(op) for op in body_ops]
    if fused_ctx_pin:
        # fused mHC ctx 还持有**流输入**（HC 模块吃的打包残差流 = attn 段入口 x_xn / ffn 段
        # 入口 h1_xn，RMSNorm 反向需其输入）。二者本就被 body 的 ln1/ln2 save（按名去重，
        # activation_saves/OFF 路径零变化）——此处仅补 pin 标志（bf16 原 dtype 计入免疫和;
        # 其 norm-fp32 cast 属重算瞬态、不随 ctx 常驻）。
        for _op in scaled:
            for _s in _op.saves:
                if _s.name in ("x_xn", "h1_xn"):
                    _s.pin_under_recompute = True

    attn_seg, ffn_seg = scaled[:split], scaled[split:]

    attn_hc = build_hyper_connection_ops("attn", d, fused_ctx_pin)
    ffn_hc = build_hyper_connection_ops("ffn", d, fused_ctx_pin)

    # ── 数据流衔接边（2026-07-11 补边;按上方 docstring 已声明的语义,inputs 追加引用、
    #    saves/输出不动 → 零字节;此前名字断链致 mHC 段在 op 图成孤立叶节点）──
    #  ① aggregate「由 body 的 ln1 承接」(:85-87) → 段首 op inputs += hc mapping_proj 输出;
    #  ② output_cell 残差更新由段尾残差 add 体现(:87) → 段尾 op inputs += h_res;
    #  ③ ffn_hc 吃的 ffn_streams = attn 段更新后的残差流(= attn 段尾输出 h1) → inputs += h1。
    def _add_dep(op: OpSpec, *refs) -> OpSpec:
        return OpSpec(op.name, op.type, list(op.inputs) + list(refs), op.output,
                      params=list(op.params), saves=list(op.saves),
                      workspace=op.workspace, bwd_scratch=op.bwd_scratch, attrs=dict(op.attrs))

    def _link(hc, seg, prev_carrier):
        if not seg:
            return seg
        seg = list(seg)
        seg[0] = _add_dep(seg[0], hc[1].output)          # ① proj(聚合来源) → 段首(ln)
        seg[-1] = _add_dep(seg[-1], hc[2].output)        # ② h_res → 段尾残差 add
        return seg

    attn_seg = _link(attn_hc, attn_seg, None)
    if attn_seg:
        ffn_hc = [_add_dep(ffn_hc[0], attn_seg[-1].output)] + ffn_hc[1:]   # ③ h1 → ffn_hc_norm
    ffn_seg = _link(ffn_hc, ffn_seg, None)
    return attn_hc + attn_seg + ffn_hc + ffn_seg
