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
        # norm_kind 必须原样带过（2026-07-29）：mHC 包装重建 OpSpec，丢了它就等于把
        # 被包装层的全部 norm 悄悄退回「抬 fp32」——这正是 x_xn/h1_xn 曾被错抬的通道。
        norm_kind=op.norm_kind,
    )


def build_hyper_connection_ops(prefix: str, d: DimTable) -> list:
    """单个 mHC HyperConnection 模块的 op 列表（**恒 3 op**；融合与否由 `d.use_fused_mhc` 门控）。

    prefix ∈ {"attn","ffn"}（`transformer_layer.py:278-285` 的 attn_hc / ffn_hc）。

    ── **融合门（2026-07-29 普查收口 Fix 1）**───────────────────────────────────────────────
    与 `d.dsa_fused` / `apply_dsa_kernel_fusion` 同一范式：**配置驱动**，两条分支都保留。
      · `use_fused_mhc=False`（默认）→ `_unfused_hc_ops`：非融合 `HyperConnectionModule.construct`
        （`hyper_connection.py:246-299`），`rms_norm(cast(hidden_states, float32), ...)`（`:262-266`）
        **确实**物化 `[S,B,n·H]` fp32（256 MiB/模块），被 `:273` mapping_proj 的 bprop 保留。
      · `use_fused_mhc=True`（站点 `dsv4h_*_pp4_recomp.yaml:109`）→ `_fused_hc_ops`：
        `FusedHyperConnectionModule`（`:368`）走 `npu_mhc_pre_sinkhorn`（`:413-419`）+
        `npu_mhc_post`（`:363`），**没有那张 256 MiB fp32**，改为 ctx 持有一批小张量 + `h_out`。
    两条分支的 **op 数（3）、名字前缀、params 集合完全一致** —— `mhc_wrap._link` 按下标接边、
    `test_param_conservation` 按 params 求和，故切换融合门**不**改 op 图拓扑、**不**改参数守恒。

    2026-07-24 口径切换：去除 `fused_ctx_pin`。fused mHC HyperConnection 的 ctx 走
    `save_for_backward`（custom_op_impl.py:331/390/588），受控 A/B（报告§7.9 E1b：真实
    FusedHyperConnectionModule 重算 ON fwd 末 0、OFF 821）证其在全重算下**正常释放** →
    纯理论口径不 pin，全重算只留层入口 checkpoint_input；真机残留归框架释放缺口显式暴露。

    op 序列（吃打包残差流 `streams [S,B,n*H]`）：见 `_unfused_hc_ops` / `_fused_hc_ops`。

    aggregate（`h_pre @ streams → [S,B,H]`，:228-231）与 output_cell（:319/:329）不单列 op：
    前者产生的 `[S,B,H]` aggregated 即 sublayer 的输入（由 body 的 ln1 承接），后者的残差流
    更新已由 body 尾部残差承载张量（×n）体现。

    > **已知残差（如实记，不补）**：源里 `input_layernorm`/`pre_mlp_layernorm` 吃的是
    > **aggregated `[S,B,H]`**（`transformer_layer.py:311,329`），其 RMSNorm 保留的输入是
    > 32 MiB/次；而普查让 body 的 `ln1`/`ln2` 保留**打包流** `x_xn`/`h1_xn`（`[S,B,n·H]` bf16
    > = 128 MiB/次）——后者数值上正是融合 ctx 的 `x`（`custom_op_impl.py:390` 首项，同一块设备
    > 张量），故融合分支**不再重复声明** `x`，以免双计。代价是那 2×32 MiB 的 aggregated 没人建
    > → **每层欠读 64 MiB**。修它要动 `mhc_wrap` 的残差承载判定（两条分支同时变），超出本轮口径。
    """
    if getattr(d, "use_fused_mhc", False):
        return _fused_hc_ops(prefix, d)
    return _unfused_hc_ops(prefix, d)


def _hc_params(prefix: str) -> tuple:
    """两条分支**共用**的 mHC 参数（`FusedHyperConnectionModule.__init__` 直接 `super().__init__`，
    `hyper_connection.py:387` → 参数名/形状/初始化与非融合完全一致 → 参数守恒逐字节不变）。"""
    # rms_weight：fp32 buffer（requires_grad=False），量级 n*H（hyper_connection.py:157-161）
    rms_w = TensorRef(f"{prefix}_hc_rms_w", (NH,), is_weight=True, dtype_bytes=4)
    # mapping_proj.weight：fp32（params_dtype=float32），[n*H, dim]（hyper_connection.py:143-150）
    proj_w = TensorRef(f"{prefix}_hc_proj_w", (NH, PROJ_OUT), is_weight=True, dtype_bytes=4)
    return rms_w, proj_w


#: `HyperConnectionOutputCell` 反向物化的打包残差流梯度（两条分支同式，见下方逐条注释）。
_HC_OUTPUT_CELL_BWD = "4*S*B*num_residual_streams*H"


def _unfused_hc_ops(prefix: str, d: DimTable) -> list:
    """非融合 `HyperConnectionModule.construct`（`hyper_connection.py:246-299`），3 op：

      1. `{prefix}_hc_norm`（NORM）: RMSNorm(n·H, fp32) → `hc_norm [S,B,n*H]` fp32（saved，
         mHC 的主激活大头：×n 且 fp32）。权重 `rms_weight [n*H]` fp32 buffer。
      2. `{prefix}_hc_mapping_proj`（MATMUL）: `Linear(n*H → dim)`，权重 fp32 `[n*H, dim]`。
      3. `{prefix}_hc_sinkhorn`（ELEMENTWISE）: sinkhorn 投影 → `h_res [S,B,n,n]`（saved，
         供 output_cell 反向 `h_res @ streams`）。

    `hc_norm` 的 fp32 是**源里真有**的：`:262-266` `rms_norm(self.cast(hidden_states, mstype.float32),
    ...)`，`:269-271` 注释逐字说明「MindSpeed keeps the weightless RMSNorm and mHC projection in FP32」。
    该 op 因此**不标** `norm_kind`（保持 CASTING）—— 它确实 cast，与 `FusedRMSNorm` 不同。
    """
    streams = TensorRef(f"{prefix}_streams", ("S", "B", NH), shard={0: "sp"})
    # RMSNorm 在 fp32 下计算（hyper_connection.py:262-266 cast float32）
    hc_norm = TensorRef(f"{prefix}_hc_norm", ("S", "B", NH), dtype_bytes=4)
    h_proj = TensorRef(f"{prefix}_hc_proj", ("S", "B", PROJ_OUT), dtype_bytes=4)
    h_res = TensorRef(f"{prefix}_h_res", ("S", "B", "num_residual_streams", "num_residual_streams"))
    rms_w, proj_w = _hc_params(prefix)

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
               bwd_scratch=_HC_OUTPUT_CELL_BWD),
    ]


def _fused_hc_ops(prefix: str, d: DimTable) -> list:
    """融合 `FusedHyperConnectionModule`（`hyper_connection.py:368-424`），同样 3 op、同样 params。

    **前向只有两个 kernel**：`npu_mhc_pre_sinkhorn`（`:413-419`）与 `npu_mhc_post`（`:363`，
    在 `FusedHyperConnectionOutputCell.construct` 里）。这里把它们摊成 3 个 OpSpec，只是为了让
    `mhc_wrap._link` 的下标接边（hc[1].output→段首 ln、hc[2].output→段尾残差 add）与非融合同构；
    **字节全部落在 `saves` 上**，中间 op 的 output 只是数据流占位（小张量，不 saved）。

    ── ctx 逐字（`hyper_parallel/platform/mindspore/custom_ops/custom_op_impl.py:390-391`）────
        ctx.save_for_backward(x, phi, alpha, bias,
                              h_pre, hc_before_norm, inv_rms, sum_out, norm_out)
    形状逐字（`mhc_pre_sinkhorn.cc:24-50`，`bs=S, seq_len=B, n=num_residual_streams, c=H,
    fusion_size=n²+2n, num_iters=mhc_sinkhorn_iterations`）：

      · `x`        `[S,B,n,H]` bf16 —— **本函数不声明**：它就是打包残差流本身，已由 body 的
                   `x_xn`/`h1_xn` 承担（见 `build_hyper_connection_ops` docstring 的残差说明）。
      · `phi`/`alpha`/`bias` —— **参数**（`rms_w`/`proj_w` 已在 params；alpha/bias 两条分支都没建，
                   量级 ~1.5 MiB 常驻、非激活，保持不建以维持参数守恒逐字节可比）。
      · `h_pre`          `[S,B,n]`                fp32  `.cc:36,40`
      · `hc_before_norm` `[S,B,n²+2n]`            fp32  `.cc:37,41`
      · `inv_rms`        `[S,B,1]`                fp32  `.cc:38,42`
      · `sum_out`        `[2·num_iters,S,B,n]`    fp32  `.cc:38,43`
      · `norm_out`       `[2·num_iters,S,B,n,n]`  fp32  `.cc:39-40,44`

    后 5 项在调用点被 `h_in, h_post, h_res_flat, *_ = npu_mhc_pre_sinkhorn(...)`（`:413`）的 `*_`
    丢掉 Python 名字，**但 ctx 仍强引用** —— `mhc_pre_sinkhorn.cc:61` `ms::TensorAllocate({...})`
    无条件分配全部 8 个输出。**没有 Python 名 ≠ 没有设备内存**。

    另外两项 kernel 输出被下游 `output_cell` 消费、故同样活到反向：
      · `h_res`  `[S,B,n²]` **fp32**（`.cc:33` `kNumberTypeFloat32`；非融合分支建的是 bf16）
      · `h_post` `[S,B,n]`  **fp32**（`.cc:32`）

    `npu_mhc_post` 另存（`custom_op_impl.py:331`）`ctx.save_for_backward(x, h_res, h_out, h_post)`：
    `x`/`h_res`/`h_post` 与上面同名同物（按名去重）；`h_out` = **sublayer 输出** `[S,B,H]` bf16
    （`hyper_connection.py:363` 第 3 实参 `sublayer_out`，`:358` 已 cast 到 compute dtype）——
    普查此前完全没建，这里补上。
    """
    streams = TensorRef(f"{prefix}_streams", ("S", "B", NH), shard={0: "sp"})
    rms_w, proj_w = _hc_params(prefix)
    # 2·num_iters 作为 sum_out/norm_out 的首维（`.cc:38-40`）——符号表达式，随 DimTable 求值。
    IT2 = "2*mhc_sinkhorn_iterations"
    n = "num_residual_streams"

    h_pre = TensorRef(f"{prefix}_hc_h_pre", ("S", "B", n), dtype_bytes=4)
    hc_before_norm = TensorRef(f"{prefix}_hc_before_norm",
                               ("S", "B", f"{n}*{n} + 2*{n}"), dtype_bytes=4)
    inv_rms = TensorRef(f"{prefix}_hc_inv_rms", ("S", "B", "1"), dtype_bytes=4)
    sum_out = TensorRef(f"{prefix}_hc_sum_out", (IT2, "S", "B", n), dtype_bytes=4)
    norm_out = TensorRef(f"{prefix}_hc_norm_out", (IT2, "S", "B", n, n), dtype_bytes=4)
    # h_in：kernel 主输出 aggregated [S,B,H]（`.cc:31`，dtype 同 x = bf16）。**不 saved**
    #   （ctx 里没有它）——只作数据流占位，供段首 ln 接边。
    h_in = TensorRef(f"{prefix}_hc_agg", ("S", "B", "H"), shard={0: "sp"})
    h_res = TensorRef(f"{prefix}_h_res", ("S", "B", n, n), dtype_bytes=4)
    h_post = TensorRef(f"{prefix}_h_post", ("S", "B", n), dtype_bytes=4)
    # npu_mhc_post 的 ctx `h_out` = sublayer 输出 [S,B,H] bf16（custom_op_impl.py:331）。
    h_out = TensorRef(f"{prefix}_h_out", ("S", "B", "H"), shard={0: "sp"})

    return [
        # 1. npu_mhc_pre_sinkhorn 的 ctx 内部量（`*_` 丢名但 ctx 持有）。融合 kernel 非 NORM
        #    —— 建成 ELEMENTWISE，避免被 `_norm_save_names` 误抬 fp32（这些本来就已是 fp32）。
        OpSpec(f"{prefix}_hc_pre_sinkhorn", OpType.ELEMENTWISE, [streams], h_pre,
               params=[rms_w, proj_w],
               saves=[h_pre, hc_before_norm, inv_rms, sum_out, norm_out]),
        # 2. aggregated（h_in）—— 下标位置与非融合的 mapping_proj 对齐，供 `_link` 接段首 ln。
        OpSpec(f"{prefix}_hc_aggregate", OpType.ELEMENTWISE, [streams, h_pre], h_in, saves=[]),
        # 3. h_res / h_post / h_out —— 下标位置与非融合的 sinkhorn 对齐，供 `_link` 接段尾残差 add。
        #    **反向瞬态（bwd_scratch）与非融合同式**：`npu_mhc_post_backward`（custom_op_impl.py:353）
        #    同样要产出打包残差流梯度 `grad_x [s,b,n,H]` 并重建残差项，量级不因融合而变
        #    → 保持 `4·S·B·n·H`，不因换 kernel 就改一个没有源码依据的数。
        OpSpec(f"{prefix}_hc_sinkhorn", OpType.ELEMENTWISE, [h_pre], h_res,
               saves=[h_res, h_post, h_out], bwd_scratch=_HC_OUTPUT_CELL_BWD),
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


def mhc_wrap(body_ops: list, n_streams: int, d: DimTable) -> list:
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

    2026-07-24 口径切换：去除 `fused_ctx_pin`（见 build_hyper_connection_ops docstring）——
    fused mHC ctx 走 save_for_backward、全重算正常释放，纯理论不 pin。
    """
    if n_streams <= 1:
        return body_ops

    split = _ffn_split_index(body_ops)
    scaled = [_scale_op(op) for op in body_ops]

    attn_seg, ffn_seg = scaled[:split], scaled[split:]

    attn_hc = build_hyper_connection_ops("attn", d)
    ffn_hc = build_hyper_connection_ops("ffn", d)

    # ── 数据流衔接边（2026-07-11 补边;按上方 docstring 已声明的语义,inputs 追加引用、
    #    saves/输出不动 → 零字节;此前名字断链致 mHC 段在 op 图成孤立叶节点）──
    #  ① aggregate「由 body 的 ln1 承接」(:85-87) → 段首 op inputs += hc mapping_proj 输出;
    #  ② output_cell 残差更新由段尾残差 add 体现(:87) → 段尾 op inputs += h_res;
    #  ③ ffn_hc 吃的 ffn_streams = attn 段更新后的残差流(= attn 段尾输出 h1) → inputs += h1。
    def _add_dep(op: OpSpec, *refs) -> OpSpec:
        return OpSpec(op.name, op.type, list(op.inputs) + list(refs), op.output,
                      params=list(op.params), saves=list(op.saves),
                      workspace=op.workspace, bwd_scratch=op.bwd_scratch, attrs=dict(op.attrs),
                      norm_kind=op.norm_kind)

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
