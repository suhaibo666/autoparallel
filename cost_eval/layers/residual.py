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

import dataclasses

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
    """残差承载**签名**（必要条件）：SP 分布的 `[S,B,H]`。

    > ⚠ **签名 ≠ 身份**（2026-07-29 普查收口 ③）。本谓词只判形状/切分，**不**判「这张量在
    > 源里真的是打包残差流」。同签名但**不是**流的至少有两个：MoE combine 输出 `comb`
    > （`expert_parallel.py:535`，mlp **内部**张量，`transformer_layer.py:331` 之前从不打包）
    > 与 shared-expert 输出 `sh_o`。此前 `_scale_ref` 直接按本谓词 ×n，把 `comb` 放大到
    > `[S,B,n·H]`（**每 MoE 层过读 96 MiB**）。真正的流由 `_stream_names` 按**数据流位置**
    > 判定，本函数退化为它的形状前置条件。
    """
    return t.shape == ("S", "B", "H") and t.shard == {0: "sp"}


def _stream_names(body_ops: list, split: int) -> frozenset:
    """一层里**真正的打包残差流** `[S,B,n·H]` 的张量名 —— 逐字按源，**只有 3 个**。

    `transformer_layer.py:290-334`（`HyperConnectionTransformerLayer.construct`）逐字：

        streams_before_attn = hidden_states                                    # ① 层入口打包流
        aggregated_attn, h_res_attn, h_post_attn = self.attn_hc(hidden_states) #   → [s,b,H]
        input_layernorm_output = self.input_layernorm(aggregated_attn)         #   ln1 吃 aggregated
        attention_output = self.self_attention(input_layernorm_output, ...)
        hidden_states = self.attn_hc.output_cell(
            h_res_attn, h_post_attn, streams_before_attn, dropout_output)      # ② attn 段尾新流
        streams_before_ffn = hidden_states
        aggregated_ffn, h_res_ffn, h_post_ffn = self.ffn_hc(hidden_states)
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(aggregated_ffn)      #   ln2 吃 aggregated
        mlp_output = self.mlp(pre_mlp_layernorm_output, input_ids=input_ids)   #   mlp 内部全 [s,b,H]
        output = self.ffn_hc.output_cell(
            h_res_ffn, h_post_ffn, streams_before_ffn, dropout_output)         # ③ 层出口新流

    即打包流 = **层入口 + 两个 `output_cell` 的输出**；`mlp` 内部（`comb`/`sh_o`）与两个
    layernorm 的输入（`aggregated`）**全部是 `[s,b,H]`**，一个字节都不 ×n。

    对应到本库的 op 列表：
      · 层入口 = body 里**只被读、无人产**的承载签名张量（`x`；MTP 里由 `eh_proj` 在 body 外产）；
      · 两个 output_cell = attn 段末 op（`add1`→`h1`）与 ffn 段末 op（`add2`/`moe_add`→`h2`）的输出。
    """
    produced = {op.output.name for op in body_ops}
    names = {t.name for op in body_ops for t in op.inputs
             if _is_residual_carrier(t) and t.name not in produced}
    for i in (split - 1, len(body_ops) - 1):
        if 0 <= i < len(body_ops) and _is_residual_carrier(body_ops[i].output):
            names.add(body_ops[i].output.name)
    return frozenset(names)


def _scale_ref(t: TensorRef, streams: frozenset) -> TensorRef:
    """**打包残差流**张量 → H 维乘以 num_residual_streams（`[S,B,H]→[S,B,n*H]`），其余原样返回。

    `streams` = `_stream_names` 判出的名字集（2026-07-29 ③）：只有它们才是源里真的打包流。

    P1-03（2026-07-14）：放大后**重命名** `{name}_xn`——此前保留原名导致同一 resolved layer 内
    同名张量两种 numel（MTP 层 `x` 同时 =H 与 =nH），而 structure_mem/ShapeEval.produced 按名
    去重 → 首见 size 覆盖后续语义、字节错算。段内边由 scaled op 集合内的一致重命名保持；
    ShapeEval.resolve 现有同名异 numel fail-loud 不变量防回潮。"""
    if t.name not in streams or not _is_residual_carrier(t):
        return t
    return TensorRef(f"{t.name}_xn", ("S", "B", NH), shard=dict(t.shard),
                     is_weight=t.is_weight, partial=t.partial, dtype_bytes=t.dtype_bytes)


#: `OpSpec` 上所有 **TensorRef 型**的非主字段（随 OpSpec 演进，用 hasattr 守护向后兼容）。
_REF_FIELDS = ("workspace_ref", "bwd_scratch_ref", "bwd_workspace_ref")


def _rebuild(op: OpSpec, **changes) -> OpSpec:
    """重建一个 OpSpec，**只**改 `changes` 里点名的字段，其余全部原样带过。

    ⚠ **必须用 `dataclasses.replace`，不得手写字段清单**（2026-07-29 三轮教训）：
    `mhc_wrap` 是 mHC 层上**唯一**的 OpSpec 重建通道，手写清单只要漏一个字段，被包装层
    （= 全部 DSv4 decoder 层）就会**静默丢掉**该字段的语义。已经发生过一次：`norm_kind`
    在上一轮被漏掉 → 被包装层的全部 norm 悄悄退回「抬 fp32」，正是 `x_xn`/`h1_xn` 被错抬
    128→256 MiB 的通道。`OpSpec` 还在长字段（`workspace_ref`/`bwd_scratch_ref`/…），
    用 `replace` 让「漏字段」在结构上不可能发生。`attrs` 做防御性浅拷贝（原实现同）。
    """
    changes.setdefault("attrs", dict(op.attrs))
    return dataclasses.replace(op, **changes)


def _scale_op(op: OpSpec, streams: frozenset) -> OpSpec:
    """对一个 op 的 inputs/output/params/saves(+ 各 *_ref) 施加 `_scale_ref`（打包残差流 ×n）。"""
    def _opt(t):
        return None if t is None else _scale_ref(t, streams)

    return _rebuild(
        op,
        inputs=[_scale_ref(t, streams) for t in op.inputs],
        output=_scale_ref(op.output, streams),
        params=[_scale_ref(t, streams) for t in op.params],
        saves=[_scale_ref(t, streams) for t in op.saves],
        # TensorRef 型 workspace/scratch 同样要跟着 ×n（今天恒为 None，但漏了就是下一个
        # norm_kind 式的静默 bug；`_opt` 保证 None 安全）。
        **{f: _opt(getattr(op, f)) for f in _REF_FIELDS if hasattr(op, f)}
    )


def _streams_ref(prefix: str, streams=None) -> TensorRef:
    """本 HC 模块吃的**打包残差流** `[S,B,n·H]`。

    `streams` 由 `mhc_wrap` 传入**层内真名**（attn_hc → `x_xn`；ffn_hc → `h1_xn`），使 HC 模块
    与 body 的残差流**同名同物**（`structure_mem` 按名去重 → 融合 ctx 的 `x` 只算一次）。
    直调（探针/单测）不传时退回占位名 `{prefix}_streams`，字节等价。
    """
    if streams is not None:
        return streams
    return TensorRef(f"{prefix}_streams", ("S", "B", NH), shard={0: "sp"})


def build_hyper_connection_ops(prefix: str, d: DimTable, streams=None) -> list:
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
    **op 2 的 output 恒是 aggregated `{prefix}_hc_agg [S,B,H]`**（两条分支同名同形），
    由 `mhc_wrap._link` 接到段首 layernorm —— 源里 `input_layernorm`/`pre_mlp_layernorm`
    吃的**就是** aggregated（`transformer_layer.py:311,329`），不是打包流。

    ── **收口 ②（2026-07-29 第三轮）：aggregated 与打包流各归各位** ────────────────────────
    此前 body 的 `ln1`/`ln2` 保留的是**打包流** `x_xn`/`h1_xn`（`[S,B,n·H]` bf16 = 128 MiB），
    那是把「RMSNorm 真正保留的 aggregated（32 MiB）」**别名到打包流名上**，好让融合 ctx 的
    `x`（`custom_op_impl.py:390` 首项）不被双计。代价是每层欠读 2×32 MiB。现在改为逐条归位：
      · `ln1`/`ln2` 保留 `{prefix}_hc_agg` `[S,B,H]`（`layer_norm.py:151-155` FusedRMSNorm 输入直通）；
      · 打包流 `x`：**融合分支**由 `npu_mhc_pre_sinkhorn` 的 ctx 逐字持有（`custom_op_impl.py:390`）
        → `_fused_hc_ops` 显式 save 它（仍按名去重，只算一次）；**非融合分支**源里没有任何
        bprop 持有 bf16 打包流本身（`:263`/`:298`/`:109` 三处 `cast` 各自物化 **fp32 副本**，
        被持有的是那些副本）→ `_unfused_hc_ops` **不** save 它，改 save 那些 fp32 副本。
    """
    st = _streams_ref(prefix, streams)
    if getattr(d, "use_fused_mhc", False):
        return _fused_hc_ops(prefix, d, st)
    return _unfused_hc_ops(prefix, d, st)


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


def _unfused_hc_ops(prefix: str, d: DimTable, streams: TensorRef) -> list:
    """非融合 `HyperConnectionModule`（`hyper_connection.py:246-301`）+ 其 `HyperConnectionOutputCell`
    （`:92-125`），3 op：

      1. `{prefix}_hc_norm`（NORM）: RMSNorm(n·H, fp32) → `hc_norm [S,B,n*H]` fp32（saved，
         mHC 的主激活大头：×n 且 fp32）。权重 `rms_weight [n*H]` fp32 buffer。
      2. `{prefix}_hc_mapping_proj`（MATMUL）: `Linear(n*H → dim)`（`:273`）**并**产出
         `aggregated [S,B,H]`（`:296-299`）—— 3 op 拆分是结构抽象（见 `_fused_hc_ops` 同款说明），
         段首 layernorm 消费的是 aggregated，故让它做本 op 的 output。
      3. `{prefix}_hc_sinkhorn`（ELEMENTWISE）: sinkhorn 投影 → `h_res [S,B,n,n]`（saved，
         供 output_cell 反向 `h_res @ streams`）；同时代表 `output_cell` 的前向/反向。

    `hc_norm` 的 fp32 是**源里真有**的：`:262-266` `rms_norm(self.cast(hidden_states, mstype.float32),
    ...)`，`:269-271` 注释逐字说明「MindSpeed keeps the weightless RMSNorm and mHC projection in FP32」。
    该 op 因此**不标** `norm_kind`（保持 CASTING）—— 它确实 cast，与 `FusedRMSNorm` 不同。

    ── **收口 ④（2026-07-29 第三轮）：非融合分支自己的 fp32 打包副本** ─────────────────────
    非融合路径上 **bf16 打包流本身没有任何 bprop 持有**（`reshape` 是视图、`Cast` 的 bprop
    只回 cast、不需输入）；真正被持有的是三处 `self.cast(..., mstype.float32)` 各自物化的
    **fp32 副本**，此前普查一份也没建：

      | 源位置 | 张量 | shape | 谁持有 | 站点字节 |
      |---|---|---|---|---|
      | `:297-298` | `x_streams`   | `[S,B,n,H]` fp32 | `:299` `matmul(h_pre, x_streams)` 的 bprop | 256 MiB |
      | `:108-109` | `x_streams`   | `[S,B,n,H]` fp32 | `:116` `matmul(h_res_t, x_streams)` 的 bprop | 256 MiB |
      | `:119-120` | `sublayer_exp`| `[S,B,1,H]` fp32 | `:122` `mul(h_post, sublayer_exp)` 的 bprop  | 64 MiB |

    三者是**三次独立的 `ops.cast` 调用**（前两者甚至在不同 Cell 里），互不别名 → 三个独立名字。
    （`:262-263` 那份 `cast(hidden_states, fp32)` **不**建：它是否活到反向取决于
    `ops.rms_norm` grad 节点内部持有什么，而那**不在权威快照内** —— 与仲裁 §3.2 脚注同一条纪律，
    读不出来的不猜。逐字节影响见 `docs/census_fix_residual_carrier_2026-07-29.md` §3。）

    融合分支**没有**这三份：`npu_mhc_pre_sinkhorn` / `npu_mhc_post` 的 ctx 逐字只有
    `custom_op_impl.py:390-391` / `:331` 那两组，fp32 中间量在 kernel 内部。
    """
    # RMSNorm 在 fp32 下计算（hyper_connection.py:262-266 cast float32）
    hc_norm = TensorRef(f"{prefix}_hc_norm", ("S", "B", NH), dtype_bytes=4)
    # aggregated：`:299` `cast(squeeze(matmul(h_pre, x_streams), 2), self.dtype)` → [S,B,H] bf16。
    h_agg = TensorRef(f"{prefix}_hc_agg", ("S", "B", "H"), shard={0: "sp"})
    h_res = TensorRef(f"{prefix}_h_res", ("S", "B", "num_residual_streams", "num_residual_streams"))
    rms_w, proj_w = _hc_params(prefix)
    n = "num_residual_streams"
    # `:298` 的 fp32 打包副本（被 `:299` 的 matmul 保留）
    x_f32 = TensorRef(f"{prefix}_hc_x_f32", ("S", "B", n, "H"), shard={0: "sp"}, dtype_bytes=4)
    # `output_cell:109` 的 fp32 打包副本（被 `:116` 的 matmul 保留）——与上者是**不同**的 cast
    cell_x_f32 = TensorRef(f"{prefix}_hc_cell_x_f32", ("S", "B", n, "H"), shard={0: "sp"},
                           dtype_bytes=4)
    # `output_cell:120` 的 fp32 sublayer 输出副本（被 `:122` 的 mul 保留）
    cell_out_f32 = TensorRef(f"{prefix}_hc_cell_out_f32", ("S", "B", "H"), shard={0: "sp"},
                             dtype_bytes=4)

    return [
        # 1. RMSNorm(n·H) → fp32 归一化流（saved：mapping_proj 反向需其输入）
        OpSpec(f"{prefix}_hc_norm", OpType.NORM, [streams], hc_norm,
               params=[rms_w], saves=[hc_norm]),
        # 2. mapping_proj（n*H → dim = 2n+n²，:273）+ 聚合（:296-299）→ aggregated [S,B,H]。
        #    saves = `:298` 的 fp32 打包副本（`:299` 的 matmul 两个操作数都带梯度 → 都保留）。
        OpSpec(f"{prefix}_hc_mapping_proj", OpType.MATMUL, [hc_norm, proj_w], h_agg,
               params=[proj_w], saves=[x_f32]),
        # 3. sinkhorn → h_res [S,B,n,n]（saved：output_cell 反向 h_res @ streams）。
        #    本 op 同时**代表 output_cell**（`_link` 把它的 output 接到段尾残差 add = output_cell）：
        #    故 output_cell 前向的两份 fp32 副本（`:109` / `:120`）也挂在这里。
        #    **反向瞬态（bwd_scratch，公式，非常数）**：HyperConnectionOutputCell
        #    （hyper_connection.py:87-112）前向 `new_streams = h_res @ x_streams + h_post*sublayer_out`
        #    产 [s,b,n,H]；其反向物化 ×n 打包残差流梯度 `grad_x_streams [s,b,n*H]` +
        #    重建 `res_part [s,b,n*H]`（`self.matmul(h_res, x_streams)`，:105），二者皆 compute
        #    dtype(bf16=2B) → `bwd_scratch = 2 张量 × 2B × S·B·(n·H) = 4·S·B·n·H`。**∝ num_residual_streams**
        #    （n=1 plain 时退化为普通 [S,B,H] 残差反向，四分之一）。仅在该 mHC 层反向事件计入。
        OpSpec(f"{prefix}_hc_sinkhorn", OpType.ELEMENTWISE, [hc_norm], h_res,
               saves=[h_res, cell_x_f32, cell_out_f32],
               bwd_scratch=_HC_OUTPUT_CELL_BWD),
    ]


def _fused_hc_ops(prefix: str, d: DimTable, streams: TensorRef) -> list:
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

      · `x`        `[S,B,n,H]` bf16 —— **本函数显式 save**（收口 ②，2026-07-29）：它就是打包残差流
                   本身，`mhc_wrap` 传进来的 `streams` 与 body 的 `x_xn`/`h1_xn` **同名同物**，
                   `structure_mem` 按名去重 → 只算一次。此前靠「让 `ln1`/`ln2` 保留打包流」间接
                   顶上，代价是 aggregated 那 32 MiB/次没人建（每层欠读 64 MiB）。
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
        #    `streams` 逐字是 ctx 首项 `x`（`custom_op_impl.py:390`）→ 收口 ② 后显式 save。
        OpSpec(f"{prefix}_hc_pre_sinkhorn", OpType.ELEMENTWISE, [streams], h_pre,
               params=[rms_w, proj_w],
               saves=[streams, h_pre, hc_before_norm, inv_rms, sum_out, norm_out]),
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
        (b) body 内**真正的打包残差流**（`_stream_names`：层入口 + 两个 output_cell 输出）
            → `[S,B,n*H]`（H 维 ×num_residual_streams）；
        (c) 两个段首 layernorm 的输入/saves 由打包流**改接** HC 的 aggregated `[S,B,H]`。

    ── **收口 ②③（2026-07-29 第三轮）**─────────────────────────────────────────────────────
    源 `transformer_layer.py:290-334` 逐字：`input_layernorm`/`pre_mlp_layernorm` 吃的是
    `aggregated_{attn,ffn}`（`[s,b,H]`），**不是**打包流；打包流只在层入口与两个
    `output_cell` 输出处出现（见 `_stream_names`）。此前 `mhc_wrap`：
      · 把**所有** `[S,B,H]{sp}` 签名张量 ×n → MoE 的 `comb`（mlp 内部量）被误放大到
        `[S,B,n·H]`（**每 MoE 层过读 96 MiB**）；
      · 让 `ln1`/`ln2` 保留打包流（128 MiB）冒充 aggregated（32 MiB），且为不双计而
        **不声明**融合 ctx 的 `x` → **每层欠读 64 MiB**。
    现在两者一起归位：`comb` 保持 `[S,B,H]`；`ln1`/`ln2` 保留 aggregated；打包流由 HC 模块
    按各自分支的源真值声明（融合=ctx 的 `x`；非融合=三份 fp32 副本，见 `_unfused_hc_ops`）。

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
    streams = _stream_names(body_ops, split)
    scaled = [_scale_op(op, streams) for op in body_ops]

    attn_seg, ffn_seg = scaled[:split], scaled[split:]

    # 每段吃的打包流：attn 段 = 层入口流（body 里无人产的承载张量，已 ×n 重命名）；
    # ffn 段 = attn 段尾 output_cell 的输出（attn_seg 末 op 的 output）。
    produced = {op.output.name for op in body_ops}
    attn_stream = next(
        (_scale_ref(t, streams) for op in body_ops for t in op.inputs
         if _is_residual_carrier(t) and t.name not in produced), None)
    ffn_stream = attn_seg[-1].output if attn_seg else None

    attn_hc = build_hyper_connection_ops("attn", d, attn_stream)
    ffn_hc = build_hyper_connection_ops("ffn", d, ffn_stream)

    # ── 数据流衔接边（2026-07-11 补边;此前名字断链致 mHC 段在 op 图成孤立叶节点）──
    #  ① 段首 layernorm 吃的是 aggregated（transformer_layer.py:311,329）→ 把它 inputs/saves
    #     里的打包流**替换**成 hc[1].output（收口 ②：此前是「追加引用」，打包流仍留在 saves 里）;
    #  ② output_cell 残差更新由段尾残差 add 体现 → 段尾 op inputs += h_res;
    #  ③ ffn_hc 吃的打包流 = attn 段尾输出（已由 `ffn_stream` 直接传入，无需补边）。
    def _add_dep(op: OpSpec, *refs) -> OpSpec:
        return _rebuild(op, inputs=list(op.inputs) + list(refs))

    def _swap_dep(op: OpSpec, old, new: TensorRef) -> OpSpec:
        """把 op 的 inputs/saves 里名为 `old` 的引用换成 `new`；`old` 不在 inputs 里就退回追加。

        退回分支保持旧行为（body 段首不消费打包流时，如单测里 ln2 未 hoist 的裸 FFN 段）。
        """
        if old is None or all(t.name != old.name for t in op.inputs):
            return _add_dep(op, new)
        return _rebuild(
            op,
            inputs=[new if t.name == old.name else t for t in op.inputs],
            saves=[new if t.name == old.name else t for t in op.saves])

    def _link(hc, seg, stream):
        if not seg:
            return seg
        seg = list(seg)
        seg[0] = _swap_dep(seg[0], stream, hc[1].output)  # ① 段首 ln 改吃 aggregated
        seg[-1] = _add_dep(seg[-1], hc[2].output)         # ② h_res → 段尾残差 add
        return seg

    attn_seg = _link(attn_hc, attn_seg, attn_stream)
    ffn_seg = _link(ffn_hc, ffn_seg, ffn_stream)
    return attn_hc + attn_seg + ffn_hc + ffn_seg
