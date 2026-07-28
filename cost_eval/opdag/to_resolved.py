# cost_eval/opdag/to_resolved.py
"""**`extracted` graph source 的适配器**：源抽取的 per-Cell `OpDAG` → 满足
`cost_eval/liveness/contract.py` 的 `ResolvedLayer` 序列（2026-07-25）。

为什么有这个文件
----------------
显存今天算自**手写** per-op 名册（`cost_eval/layers/*.py` 里人肉声明的 `saves=[...]`）。那份名册
已被真机反复证伪（`ukl1/ukl2` 把 detached 的 O(S²) fp32 张量当 saved、`q_hnorm_fp32` 抬错 dtype
且与 `q` 同源重复计、漏 `sinks`/`sparse_indices`、漏 `ctx.logits`……逐条见
`fn_saves.KNOWN_GAPS`），故改成从**真 MindFormers 源**抽出的图（`cost_eval/opdag/`，纯 AST）。

逐张量 liveness 仿真器（`cost_eval/liveness/simulate.py`）与两条来源**都无关**：它只要一个
`ResolvedLayer` 序列。本文件就是那道缝的抽取侧实现，注册名 `extracted`（惰性探测入口点见
`cost_eval/liveness/sources.py::EXTRACTED_ENTRY_POINT`）。

    OpDAG(extract_cell) ──infer_shapes──▶ 符号 shape ──consumer.local_shape_elems──▶ 本地字节
        └────────────────── resolve_graph(model_spec, parallel_model) ──────────────▶ ResolvedGraph

三条不可动摇的纪律
------------------
1. **绝不零填**。任何 shape / dtype / 权重形状解不出的东西，**不**拿 0 或"看起来合理的数"顶：
   承载它的节点整个被跳过，并逐条记进 `Coverage`（带 `file:line` + 原因码）。故本来源给出的峰值
   是一个**下界（floor）**，不是估计值 —— `Coverage.is_partial` / `coverage_report()` 让这件事
   在调用点可见。契约的 **B1/B2/B3** 正是为挡住"shape 全 `?` → total_bytes=0 冒充可用图"而设。
2. **绝不杜撰常数**。所有字节都回溯到 `DimTable` 的字段值 + 源里逐字读出的符号表达式。
   `DimTable` 装不下的 yaml 级事实（`use_fused_mhc`、`moe_router_score_function` …）集中在
   `_DECLARED_FACTS`，每条带 `file:line` 或 yaml 键作为出处 —— 与 `extract_cell` 的
   `kernel_saves` / `runtime_predicates` 同一条"调用方声明必须带出处"的纪律。
3. **不可判定即 fail-loud**。不支持的 layer_type、tp/cp > 1（激活的 TP/CP placement 抽取侧
   拿不到）、权威快照 md5 不匹配 —— 一律抛，绝不静默给一个偏小的数。

与手写侧的**已知**口径差（逐条在 `Coverage` 里可见，不隐藏）
--------------------------------------------------------
* `workspace_bytes` / `bwd_scratch_bytes` **恒 0**。契约 §不要求 明确排除它们（kernel 实现细节，
  源码里读不出来，继续沿用手写/profiler 标定）。故 `extracted` 相对 `hand_spec` **少**掉那部分
  已标定量（`nll` 的 4236 MiB、mHC sinkhorn 的 2×256 MiB、dispatch/combine 的 2×64 MiB …）。
  `Coverage.absent_calibration_note` 记着这件事。
* norm 输入的 fp32 **预抬要撤掉**：`bprop_rules.derive_saves` 对 `Norm` 的输入按
  `attrs["ln_compute_dtype"]`（fp32）预抬（`bprop_rules.py:127-130`），而契约明文要求
  「norm 的 fp32 cast 由消费侧 `structure_mem._dt` 再抬，**不**预抬」（`contract.py` 的
  `ResolvedTensorContract` docstring）。不撤会被抬两次（×2）。见 `_undo_norm_prelift`。
* `Detach` 别名（`ops.stop_gradient(x)` 与 `x` 同一块存储）在 `ResolvedLayer` 里**无法**表达
  "零字节别名"，故仍各占一份 → 该部分是**过计**，逐条记在 `Coverage.detach_alias_overcount`。
"""
from __future__ import annotations

import ast
import hashlib
import os
import threading
from dataclasses import dataclass, field, replace

from .bprop_rules import derive_saves
from .consumer import _dtype_bytes, axis_value, detach_aliases, local_shape_elems
from .extractor import extract_cell
from .init_binder import Binding
from .init_dims import INIT_PARAM_SEEDS, eval_init_dims
from .module_resolver import (PYNATIVE_SPEC_FILES, ResolvedSpec,
                              resolve_layer_spec)
from .shape_infer import infer_shapes, merge_dims_ctx
from .sym_shape import parse_axis

__all__ = ["resolve_graph", "ExtractedGraph", "ExtractedLayer", "Coverage",
           "RTensor", "ROp", "mf_root", "SNAPSHOT_MD5", "SUPPORTED_LAYER_KINDS",
           "IncompleteExtraction", "ALLOW_PARTIAL_ENV"]

MiB = 2 ** 20

#: 显式**选择加入**部分图的环境变量（诊断用）。缺省不设 → 部分图 fail-loud，见
#: `IncompleteExtraction` 的论证。
ALLOW_PARTIAL_ENV = "COST_EVAL_EXTRACTED_ALLOW_PARTIAL"


class IncompleteExtraction(RuntimeError):
    """**抽出的图还不完备** —— 于是本来源拒绝交出一个峰值。

    为什么这是默认行为（而不是"给个偏小的数 + 备注"）：一个跳过了 N 个节点的图，其峰值是
    **下界**，不是估计值；一旦它出现在与真机并排的表格里，就必然被当成"模型读到了这么多"。
    任务纪律是「绝不把未解析张量零填进峰值；部分解析的图必须报成 partial，而不是一个低数」，
    最诚实的落地就是：**默认不给数**，改为抛出本异常，把「缺什么、缺在哪一行」写进消息里。
    验收门按来源隔离（`tools/liveness_ab_validate.py::SourceFailure`）→ 该列显示 `ERR` +
    本消息，`bucket` / `hand_spec` 照常出数。

    要拿那个下界做诊断：设 `COST_EVAL_EXTRACTED_ALLOW_PARTIAL=1`，或
    `resolve_graph(spec, pm, allow_partial=True)`。此时返回的图仍带 `.coverage`，
    `coverage.is_partial` 为真。
    """

    def __init__(self, message: str, coverage: "Coverage" = None):
        super().__init__(message)
        self.coverage = coverage

# ═══════════════════════════════════════════════════════════════════════════
# 0. 权威快照
# ═══════════════════════════════════════════════════════════════════════════

#: 权威快照的指纹（`mf-src-167/SNAPSHOT.md`：`mindformers` @ `26354ff64`）。抽取器逐行引用的
#: 行号全部依赖这两个文件的内容；换一份 commit 抽图 = **给一份真机从未跑过的代码建模**。
#: 故不匹配即 fail-loud（`SNAPSHOT.md` 自己的论证）。
SNAPSHOT_MD5 = {
    "pynative/transformers/experimental_attention_variant/csa.py":
        "81673be3ad3cdd2191e28dd000a13f0e",
    "pynative/transformers/experimental_attention_variant/indexer.py":
        "8e18fee21c33507dd629c89f401986e6",
}

_ROOT_HINT = (
    "设 MINDFORMERS_ROOT 指向权威快照的**包目录**"
    r"（E:\97-codes\torch_parallel\mf-src-167\mindformers）；"
    r"不要用 E:\97-codes\torch_parallel\mindformers（不同 commit，见 mf-src-167/SNAPSHOT.md）")


def mf_root() -> str:
    """权威快照的 mindformers **包目录**（含 `pynative/`）。

    与 `crosscheck.default_mf_root()` 的差别（**有意**）：本函数不接受"本机 master"那份缺省 ——
    抽取侧的每一个 `file:line` 都是按 `mf-src-167` 记的，换 commit 会静默错位。故：
    `MINDFORMERS_ROOT`（含两级下降容错）→ 不含 `pynative/` 即 `RuntimeError`。
    """
    cand = os.environ.get("MINDFORMERS_ROOT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "..", "mf-src-167", "mindformers")
    cand = os.path.abspath(cand)
    for c in (cand, os.path.join(cand, "mindformers")):
        if os.path.isdir(os.path.join(c, "pynative")):
            return c
    raise RuntimeError(
        f"抽取来源 `extracted` 需要 mindformers 源包目录，{cand!r} 下没有 `pynative/`。{_ROOT_HINT}")


def _verify_snapshot(root: str) -> None:
    """核对权威快照指纹。缺文件 / md5 不符 → fail-loud（绝不换一份源悄悄出数）。"""
    for rel, want in SNAPSHOT_MD5.items():
        path = os.path.join(root, *rel.split("/"))
        if not os.path.isfile(path):
            raise RuntimeError(f"权威快照缺文件：{path}。{_ROOT_HINT}")
        with open(path, "rb") as fh:
            got = hashlib.md5(fh.read()).hexdigest()
        if got != want:
            raise RuntimeError(
                f"权威快照指纹不符：{rel} md5={got} != {want}（mf-src-167 @ 26354ff64）。"
                f"抽取侧的全部 file:line 依赖这一份内容，换 commit 会静默错位。{_ROOT_HINT}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. `DimTable` 装不下的 yaml / `__init__` 级事实 —— 逐条带出处
# ═══════════════════════════════════════════════════════════════════════════

#: `(flag, value, 出处)`。纪律同 `extract_cell(kernel_saves=...)`：调用方声明**必须带出处**，
#: 且在此集中可评审 —— 不散落在代码里当"默认值"。
#: `yaml:` 前缀 = `analysis/realmachine/ab_fusion_2026-07-25/dsv4h_*_pp4_recomp.yaml` 的键；
#: `src:` 前缀 = 权威快照里逐字读出的源侧事实。
_DECLARED_FACTS = (
    # ── 部署形态 / dtype ────────────────────────────────────────────────────
    ("compute_dtype", "bf16", "yaml:compute_dtype"),
    ("params_dtype", "bf16", "yaml:params_dtype"),
    ("layernorm_compute_dtype", "fp32", "yaml:layernorm_compute_dtype"),
    ("add_bias_linear", False, "yaml:add_bias_linear（Linear.has_bias → 无 bias 参数）"),
    ("input_layout", "BSND", "yaml:input_layout"),
    ("training", True, "训练态建模（前向图 = 首次前向）"),
    ("normalization", "RMSNorm", "yaml:normalization"),
    ("fused_norm", True, "yaml:fused_norm"),
    # ── mHC（hyper-connection 残差）─────────────────────────────────────────
    ("use_fused_mhc", True, "yaml:use_fused_mhc"),
    ("mhc_sinkhorn_iterations", 20, "yaml:hc_sinkhorn_iters"),
    ("iterations", 20, "src:hyper_connection.py:45 SinkhornKnopp.__init__ 形参"),
    ("mhc_layernorm_epsilon", 1e-6, "yaml:layernorm_epsilon"),
    ("apply_residual_connection_post_layernorm", False, "yaml 缺省 False"),
    ("hidden_dropout", 0.0, "yaml 未给 hidden_dropout → Dropout 恒等（dropout.py:74-75）"),
    ("use_dropout", False, "src:dropout.py:74-75 `not self.use_dropout` → 恒等"),
    ("enable_hc_head", True,
     "src:transformer_config.py:2158-2159 enable_hc_head 缺省 None → 跟随 enable_hyper_connections"),
    ("hc", True, "同上"),
    # ── DSv4-Flash 注意力 ───────────────────────────────────────────────────
    ("csa_dense_mode", False, "yaml 缺省 False"),
    ("is_tnd", False, "src:input_layout=BSND ⇒ 非 TND"),
    ("rotate", True, "src:compressor.py:129 `Hadamard(d) if rotate else IdentityOp()`；yaml 缺省 True"),
    ("sparse_loss", True, "yaml:dsa_indexer_loss_coeff 非零 ⇒ 有 indexer loss 分支"),
    ("use_butterfly", False, "yaml 缺省 False"),
    # `enable_compress` **不在这里给** —— 它由 ratio 逐字导出（`csa.py:594`
    # `if compress_ratio > 0 and submodules.compressor is not None:`），见 `_cell_flags`。
    ("dsa_indexer_loss_coeff", 0.001, "yaml:dsa_indexer_loss_coeff"),
    # ── MoE ────────────────────────────────────────────────────────────────
    ("moe_router_dtype", "fp32", "yaml:moe_router_dtype"),
    ("score_func", "softmax", "yaml:moe_router_score_function"),
    ("aux_loss_type", "seq_aux_loss", "yaml:moe_router_load_balancing_type"),
    ("route_norm", True, "yaml:norm_topk_prob"),
    ("route_scale", 1.0, "yaml:moe_router_topk_scaling_factor 缺省 → 1.0（router.py:72-75）"),
    ("num_expert_groups", 0, "yaml:n_group=0"),
    ("is_hash_layer", False, "yaml 缺省 False"),
    ("_debug_force_load_balance", False, "yaml 缺省 False"),
    ("moe_aux_loss_coeff", 0.001, "yaml:moe_aux_loss_coeff"),
    ("calculate_per_token_loss", False, "yaml 缺省 False"),
    ("moe_permute_fusion", False, "yaml 缺省 False"),
    ("use_shared_expert_gating", False, "yaml 缺省 False（DimTable.moe_shared_gate 亦 False）"),
    ("use_shared_expert_gate", False, "src:shared_experts.py:57 派生量"),
    ("score_before_experts", False, "yaml 缺省 False"),
    ("moe_apply_probs_on_input", False, "yaml 缺省 False"),
    ("enable_expert_bias", False, "yaml 缺省 False"),
    ("moe_grouped_gemm", True, "yaml:moe_grouped_gemm"),
    ("activation_type", "silu", "yaml:hidden_act"),
    ("hidden_act", "silu", "yaml:hidden_act"),
    ("activation_type_mlp", "silu", "yaml:hidden_act"),
    ("activation_func_clamp_value", None, "yaml 缺省 None"),
    ("use_clamped_swiglu", False, "src:mlp.py:92 `activation_type=='fusedswiglu' and clamp!=None`"),
    # ── embedding / loss ───────────────────────────────────────────────────
    ("add_position_embedding", False, "yaml:position_embedding_type=rope ⇒ 无 learned absolute"),
    ("num_tokentypes", 0, "yaml 未给 ⇒ 0（language_model_embedding.py:91 ⇒ None）"),
    ("tokentype_embeddings", None, "同上"),
    ("position_embeddings", None, "同上"),
    ("embedding_dropout_prob", 0.0, "yaml 缺省 0.0"),
    ("chunk_loss_num", 1, "yaml 未给 chunk_loss_num ⇒ CrossEntropyLoss（非 Chunk 版）"),
    ("compensate_loss_sense_tp", True, "yaml 缺省 True"),
    # ── Linear（lm_head 构造点，gpt_model.py:252-258）────────────────────────
    ("skip_weight_param_allocation", False, "src:gpt_model.py:252-258 output_layer 构造点"),
    ("has_bias", False, "同上（add_bias_linear=False）"),
    ("skip_add_bias", False, "同上"),
)

#: 部署形态谓词（`hasattr` / `isinstance` / 零参 host 函数）。同 `probe_components.py` 的口径，
#: 逐条是**部署事实**，不是猜：pynative 栈把 `Parameter` 包成 hyper_parallel `DTensor`。
_RUNTIME_PREDICATES = {
    "hasattr:to_local": True,
    "hasattr:detach": True,
    "isinstance:DTensor": True,
    # 前向图建模的是**首次前向**（非重算中那次）—— activation_checkpoint.is_in_recompute()
    "call:is_in_recompute": False,
}
_HOST_ALLOW = ("save_to_indexer_losses_tracker", "get_indexer_loss_tracker",
               "save_to_aux_losses_tracker", "get_moe_layer_wise_logging_tracker",
               "Validator.check_type_name")
_KERNEL_ALLOW = ("npu_lightning_indexer", "npu_mhc_pre_sinkhorn", "npu_mhc_post")

#: 内联进层图的子 Cell（`extract_cell(recurse=True, subcell_specs=...)` 需要它们有个 spec 壳）。
_BARE_CELLS = (
    "UnfusedCSAIndexerLoss", "Hadamard",
    "FusedHyperConnectionModule", "HyperConnectionModule",
    "HyperConnectionOutputCell", "FusedHyperConnectionOutputCell",
    "SinkhornKnopp", "HyperConnectionHead", "Dropout", "IdentityOp",
    "MoELayer", "TopKRouter", "GroupedMLP", "SharedExpertMLP", "SequentialMLP",
    "MLP", "MoEAuxLossAutoScaler", "MoEAlltoAllTokenDispatcher",
    "VocabEmbedding", "LanguageModelEmbedding",
    "_LogSoftmax", "_NLLLoss", "_LogSoftmaxModule", "_NLLLossModule",
    "_VocabParallelCrossEntropy", "_ChunkCrossEntropyLoss",
)

#: 链上各 Cell 的 `__init__` —— 供 `dims_ctx` / `param_shapes` / `param_dtypes` 合并
#: （`extract_cell` 的 `dag.dims_ctx` 只装**顶层**类，`extractor.py:725`；内联子 Cell 里的
#: `self.<attr>` 只在它自己那个类的 `__init__` 里，见 `shape_infer.merge_dims_ctx` 的论证）。
_INIT_CHAIN = (
    ("HyperConnectionTransformerLayer", "pynative/transformers/transformer_layer.py"),
    ("FusedHyperConnectionModule", "pynative/transformers/hyper_connection.py"),
    ("SinkhornKnopp", "pynative/transformers/hyper_connection.py"),
    ("DSv4HybridSelfAttention",
     "pynative/transformers/experimental_attention_variant/deepseek_v4_hybrid_attention.py"),
    ("CompressedSparseAttention",
     "pynative/transformers/experimental_attention_variant/csa.py"),
    ("Compressor", "pynative/transformers/experimental_attention_variant/compressor.py"),
    ("CSAIndexer", "pynative/transformers/experimental_attention_variant/indexer.py"),
    ("MoELayer", "pynative/transformers/moe/moe_layer.py"),
    ("TopKRouter", "pynative/transformers/moe/router.py"),
    ("GroupedMLP", "pynative/transformers/moe/experts.py"),
    ("SharedExpertMLP", "pynative/transformers/moe/shared_experts.py"),
    ("MLP", "pynative/transformers/mlp.py"),
    ("Linear", "pynative/layers/linear.py"),
    ("LanguageModelEmbedding",
     "pynative/base_models/common/embeddings/language_model_embedding.py"),
    ("VocabEmbedding", "pynative/base_models/common/embeddings/vocab_embedding.py"),
    ("CrossEntropyLoss", "pynative/loss/loss.py"),
)

#: 支持的 layer_type 判据（`ModelSpec.layer_pattern` 的取值）。不在此列的 → fail-loud。
SUPPORTED_LAYER_KINDS = ("embedding", "dsv4hyb", "lm_head", "mtp")


# ═══════════════════════════════════════════════════════════════════════════
# 2. 契约面的实现（三层 dataclass，字段一个不多一个不少 + src 溯源）
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RTensor:
    """一个已解析到**本地字节**的张量（`contract.ResolvedTensorContract`）。"""
    name: str
    local_numel: int
    dtype_bytes: int
    is_weight: bool = False
    detached: bool = False
    is_expert: bool = False
    dim0: int = 0
    pin_under_recompute: bool = False
    #: 溯源（非契约字段；报告用）：符号 shape 串 + dtype 串 + 源定位符。
    sym_shape: str = ""
    dtype_name: str = ""
    src: str = ""


@dataclass(frozen=True)
class ROp:
    """一个前向 op（`contract.ResolvedOpContract`）。反向节点由 liveness 从 saves 导出。"""
    name: str
    type: str
    inputs: tuple
    output: RTensor
    params: tuple = ()
    saves: tuple = ()
    #: 契约 §不要求：kernel 实现细节，源码里读不出来 → 恒 0（见模块 docstring）。
    workspace_bytes: int = 0
    bwd_scratch_bytes: int = 0
    collectives: tuple = ()
    src: str = ""


@dataclass(frozen=True)
class ExtractedLayer:
    """一层（或伪层 embedding / lm_head / mtp）的前向 op 序（`contract.ResolvedLayerContract`）。"""
    layer_id: int
    layer_type: str
    ops: tuple
    coverage: "Coverage" = None


# ═══════════════════════════════════════════════════════════════════════════
# 3. 覆盖度台账 —— "解不出的东西必须显式可见"
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Coverage:
    """一个 segment / 一整张图的**覆盖度台账**。

    这是本来源"给出的是下界，不是估计"的凭据。`skipped_ops` 里每条带 `file:line` + 原因码
    （来自 `infer_shapes(report=)` 的 11 个原因码），`unresolved_saves` 是那些**本该**进
    `saves` 却算不出字节的张量名 —— 它们**没有**被零填进峰值。
    """
    tag: str = ""
    n_nodes: int = 0
    n_ops: int = 0
    skipped_ops: list = field(default_factory=list)       # [(node_id, op, src, reason)]
    unresolved_saves: list = field(default_factory=list)  # [(name, sym_shape, src, reason)]
    unresolved_params: list = field(default_factory=list)  # [(name, src, reason)]
    resolved_params: list = field(default_factory=list)   # [(name, sym, dtype, bytes)]
    shape_conflicts: list = field(default_factory=list)   # [(name, sigA, sigB)]  S3 风险
    detach_alias_overcount: list = field(default_factory=list)   # [(alias, root, bytes)]
    node_gap_reasons: dict = field(default_factory=dict)  # reason -> count
    #: **执行序上第一个被跳过的节点** = 级联的**根**（下游 `no_input_shape` 全是它的连带）。
    first_skipped: tuple = None                           # (node_id, op, src, reason)
    #: param census 两个口径（**权重的形状来自 `__init__`，与激活数据流无关**，故即使承载它的
    #: 节点被跳过、权重本身仍可解析）：`census` = 全部可解析的权重字节；`in_graph` = 真正挂在
    #: 存活 op 上、因而**进了显存记账**的那部分。二者的差 = 「解出来了但因节点被跳过而丢掉」。
    param_census_bytes: int = 0
    param_in_graph_bytes: int = 0
    #: **调用方声明的**中间量形状（`_DECLARED_SHAPES`）：`[(name, sym, src, reason)]`。
    #: 与"推断出来的"分开记 —— 声明永远不许被洗成事实（`kernel_saves` 的同一条纪律）。
    declared_shapes: list = field(default_factory=list)
    #: `workspace_bytes` / `bwd_scratch_bytes` 恒 0 的说明（契约 §不要求）。
    absent_calibration_note: str = (
        "workspace_bytes / bwd_scratch_bytes 恒 0：契约 §不要求 明确排除（kernel 实现细节，"
        "源码读不出）→ 相对 hand_spec 少掉那部分已标定量")
    children: list = field(default_factory=list)          # 子 Coverage（整图聚合用）

    # ── 派生 ────────────────────────────────────────────────────────────────
    @property
    def is_partial(self) -> bool:
        return bool(self.skipped_ops or self.unresolved_saves or self.unresolved_params
                    or self.shape_conflicts
                    or any(c.is_partial for c in self.children))

    def totals(self) -> dict:
        """聚合计数（含子 Coverage）。"""
        acc = {"n_nodes": self.n_nodes, "n_ops": self.n_ops,
               "skipped_ops": len(self.skipped_ops),
               "unresolved_saves": len(self.unresolved_saves),
               "unresolved_params": len(self.unresolved_params),
               "resolved_params": len(self.resolved_params),
               "shape_conflicts": len(self.shape_conflicts),
               "detach_alias_overcount": len(self.detach_alias_overcount),
               "param_census_bytes": self.param_census_bytes,
               "param_in_graph_bytes": self.param_in_graph_bytes,
               "declared_shapes": len(self.declared_shapes)}
        for c in self.children:
            for k, v in c.totals().items():
                acc[k] = acc.get(k, 0) + v
        return acc

    def blockers(self, n: int = 5) -> list:
        """**级联的根**：各 segment 执行序上第一个被跳过的节点，按 `src` 归并计数。

        为什么看这个而不是看 `no_input_shape` 的总数：下游 `?` 绝大多数是**连带**。真正要修的
        是最早那一个节点（它的输出一旦有形状，后面整条链就活了）。"""
        cnt: dict = {}
        for c in [self] + self._all_children():
            if c.first_skipped:
                _i, op, src, reason = c.first_skipped
                key = (src, op, reason.split(" @")[0])
                cnt[key] = cnt.get(key, 0) + 1
        return sorted(cnt.items(), key=lambda kv: -kv[1])[:n]

    def report(self, *, top: int = 12) -> str:
        """人读的覆盖度报告（验收报告直接贴这段）。"""
        t = self.totals()
        head = ("覆盖度 %s：节点 %d → op %d（跳过 %d）；saves 未解析 %d；"
                "params 已解析 %d / 未解析 %d；shape 冲突 %d；detach 别名过计 %d"
                % (self.tag or "<graph>", t["n_nodes"], t["n_ops"], t["skipped_ops"],
                   t["unresolved_saves"], t["resolved_params"], t["unresolved_params"],
                   t["shape_conflicts"], t["detach_alias_overcount"]))
        lines = [head, "  判决: %s" % ("PARTIAL（峰值是下界，不是估计）" if self.is_partial
                                      else "FULL")]
        lines.append("  param census: 可解析 %.3f MiB / 真正进图 %.3f MiB（差额 = 权重解出来了"
                     "但承载节点被跳过 → 未进显存记账）"
                     % (t["param_census_bytes"] / MiB, t["param_in_graph_bytes"] / MiB))
        gaps = {}
        for c in [self] + self._all_children():
            for k, v in c.node_gap_reasons.items():
                gaps[k] = gaps.get(k, 0) + v
        if gaps:
            lines.append("  节点原因码: " + ", ".join(
                "%s=%d" % (k, v) for k, v in sorted(gaps.items(), key=lambda kv: -kv[1])))
        decl = {}
        for c in [self] + self._all_children():
            for name, sym, src, _why in c.declared_shapes:
                decl[(name, sym, src)] = decl.get((name, sym, src), 0) + 1
        if decl:
            lines.append("  调用方**声明**的中间量形状（源侧 docstring 逐字，非推断）:")
            for (name, sym, src), k in sorted(decl.items(), key=lambda kv: -kv[1]):
                lines.append("    ×%-3d %-20s = %-22s  <- %s" % (k, name, sym, src))
        bl = self.blockers()
        if bl:
            lines.append("  级联根（各 segment 首个被跳过的节点，按源归并）:")
            for (src, op, reason), k in bl:
                lines.append("    ×%-3d %-14s %-42s %s" % (k, op, src, reason))
        return "\n".join(lines)

    def _all_children(self) -> list:
        out = []
        for c in self.children:
            out.append(c)
            out.extend(c._all_children())
        return out


@dataclass(frozen=True)
class ExtractedGraph:
    """`.stages: {stage: [ExtractedLayer, ...]}` + 整图覆盖度台账。"""
    stages: dict
    coverage: Coverage

    def coverage_report(self, **kw) -> str:
        return self.coverage.report(**kw)


# ═══════════════════════════════════════════════════════════════════════════
# 4. 配置派生（全部来自 `ModelSpec.dims` / `layer_pattern` / `_DECLARED_FACTS`）
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _Site:
    """一个抽取站点的完整配置（cache key）。"""
    fused: bool
    ratio: int
    moe: bool
    dims_key: tuple

    @property
    def kind(self) -> str:
        return "moe" if self.moe else "dense"


def _declared() -> dict:
    return {k: v for k, v, _why in _DECLARED_FACTS}


def _dims_key(dims) -> tuple:
    """`DimTable` 的可哈希签名（**剔除** `n_layers`：它不影响单层图的形状）。"""
    return tuple(sorted((k, v) for k, v in dims.as_dict().items() if k != "n_layers"))


def _spec_flags(dims, *, moe: bool) -> dict:
    """`resolve_layer_spec` 的 spec 级开关（决定 spec 树选哪条实现）。"""
    return {
        "num_experts": int(dims.n_experts) if moe else 0,
        "moe_grouped_gemm": True,
        "qk_layernorm": bool(dims.qk_layernorm),
        "multi_latent_attention": True,
        "enable_hyper_connections": int(dims.num_residual_streams) > 1,
        "fused_norm": True,
        "normalization": "RMSNorm",
        "is_dsv4_hybrid": True,
    }


def _cell_flags(dims, *, fused: bool, ratio: int, moe: bool) -> dict:
    """walker / `eval_init_dims` 的 config 级 flag 表。

    **每一项**要么来自 `DimTable` 的字段值，要么来自 `_DECLARED_FACTS`（带出处），要么由
    `ratio` 逐字导出（源侧就是 `compress_ratio == 4` 这类判据）。没有第四种来源。
    """
    n_stream = int(dims.num_residual_streams)
    # `config.ffn_hidden_size`：dense 层 = `DimTable.F`；**MoE 层里唯一的 MLP 是 shared expert**，
    # 而 `shared_experts.py:52` 在 `super().__init__` 之前把 `config.ffn_hidden_size` 改写成
    # `moe_shared_expert_intermediate_size`（[SRC]）→ MoE 层用 `moe_shared_F`。
    ffn = int(dims.moe_shared_F if moe else dims.F)
    f = dict(_declared())
    f.update({
        # ── 维度（全部来自 DimTable）───────────────────────────────────────
        "hidden_size": int(dims.H), "num_attention_heads": int(dims.n_heads),
        "seq_length": int(dims.S), "micro_batch_size": int(dims.B),
        "vocab_size": int(dims.vocab), "embedding_dim": int(dims.H),
        "q_lora_rank": int(dims.q_lora_rank), "kv_lora_rank": int(dims.kv_lora_rank),
        "v_head_dim": int(dims.v_head_dim),
        "qk_pos_emb_head_dim": int(dims.qk_rope_head_dim),
        "o_groups": int(dims.o_groups), "o_lora_rank": int(dims.o_lora_rank),
        "index_topk": int(dims.dsa_indexer_topk),
        "index_n_heads": int(dims.dsa_indexer_n_heads),
        "index_head_dim": int(dims.dsa_indexer_head_dim),
        "csa_window_size": int(dims.csa_window_size),
        "window_size": int(dims.csa_window_size),
        "num_residual_streams": n_stream, "hc_num_streams": n_stream,
        "hc_hidden_size": int(dims.H),
        "n_hidden": n_stream * int(dims.H),      # mHC 打包流宽度（层 construct 入参末轴）
        "ffn_hidden_size": ffn, "moe_ffn_hidden_size": int(dims.moe_F),
        "moe_shared_expert_intermediate_size": int(dims.moe_shared_F),
        "num_experts": int(dims.n_experts), "num_moe_experts": int(dims.n_experts),
        "moe_router_topk": int(dims.topk), "top_k": int(dims.topk),
        "shared_expert_num": int(dims.n_shared),
        "gated_linear_unit": bool(dims.gated_linear_unit),
        "num_layers": int(dims.n_layers), "mtp_num_layers": 0,
        # ── TopKRouter 的 SP/CP 通信组（tp/cp 由 resolve_graph 门死为 1）─────
        "_tp_size": 1, "_cp_size": 1, "_tp_group": None, "_cp_groups": (),
        # ── ratio 逐字导出的 __init__ 派生量 ───────────────────────────────
        "compress_ratio": int(ratio),
        # src:csa.py:594 `if compress_ratio > 0 and submodules.compressor is not None:`
        "enable_compress": ratio > 0,
        # src:csa.py:608 `if compress_ratio == 4 and not config.csa_dense_mode and …`
        "enable_indexer": ratio == 4,
        "overlap": ratio == 4,                   # src:compressor.py:89
        "coff": 1 + int(ratio == 4),             # src:compressor.py:90
        "apply_dsa_kernel_fusion": bool(fused),  # DimTable.dsa_fused
        "dtype": "bf16",                         # src:hyper_connection.py:205 self.dtype=compute_dtype
        # ── Linear（lm_head 构造点）────────────────────────────────────────
        "input_size": int(dims.H), "output_size": int(dims.vocab),
    })
    return f


def _input_axes(dims) -> dict:
    """入口形参的**逐轴 config-flag 名**（walker 用来判 `.ndim` / `b,s,h = x.shape`）。

    源侧 shape 契约逐字来自 docstring：
      `Compressor.construct(x: [sq, b, hidden_size])`                    compressor.py:179
      `CSA.construct(query:[sq,b,np,v_head_dim], key:[sq,b,1,v_head_dim],
                     x:[sq,b,hidden_size], qr:[sq,b,q_lora_rank])`       csa.py:648-651
      `DSv4HybridSelfAttention.construct(x: [sq, b, hidden_size])`       deepseek_v4:233
      mHC 层 construct 入参是**打包流** [s, b, n·H]                       hyper_connection.py:405
      `logits [b, s, vocab]`（`logits.ndim == 3` @ loss.py:363 靠轴数判定）
    """
    return {
        "x": ("seq_length", "micro_batch_size", "hidden_size"),
        "qr": ("seq_length", "micro_batch_size", "q_lora_rank"),
        "query": ("seq_length", "micro_batch_size", "num_attention_heads", "v_head_dim"),
        "key": ("seq_length", "micro_batch_size", 1, "v_head_dim"),
        "hidden_states": ("seq_length", "micro_batch_size", "n_hidden"),
        "input_": ("seq_length", "micro_batch_size", "hidden_size"),
        "logits": ("micro_batch_size", "seq_length", "vocab_size"),
        "label": ("micro_batch_size", "seq_length"),
        "input_mask": ("micro_batch_size", "seq_length"),
        "input_ids": ("micro_batch_size", "seq_length"),
        "position_ids": ("micro_batch_size", "seq_length"),
    }


#: `infer_shapes` 的入口**符号** shape 种子，**按 segment 分开**（同一个形参名在不同 Cell 里是
#: 不同的东西：`input_` 在 `Linear.construct` 是 `[s,b,H]` 激活、在 `VocabEmbedding.construct`
#: 是 `[b,s]` token id —— 一张扁平表必然把其中一个算错）。
#: 每条的出处是源侧 shape 契约（docstring / 类型注释），与 `_input_axes` 同一批。
_SEED_SHAPES = {
    "decoder": {
        # mHC 层 construct 入参是**打包流** [s, b, n·H]（hyper_connection.py:405 的 reshape）
        "hidden_states": "S·B·hc_mult·H",
        "x": "S·B·H",                       # compressor.py:179 / deepseek_v4:233
        "qr": "S·B·q_lora_rank",            # csa.py:651
        "query": "S·B·n_heads·v_head_dim",   # csa.py:648
        "key": "S·B·1·v_head_dim",           # csa.py:649
        "input_ids": "B·S",
    },
    "embedding": {
        "input_ids": "B·S", "position_ids": "B·S",
        "input_": "B·S",                     # VocabEmbedding.construct(input_) = token id
    },
    "lm_head": {"input_": "S·B·H"},          # Linear.construct(input_)：final-norm 后的 hidden
    "loss": {"logits": "B·S·vocab", "label": "B·S", "input_mask": "B·S"},
    "mtp": {
        "hidden_states": "S·B·hc_mult·H", "input_ids": "B·S", "position_ids": "B·S",
        "x": "S·B·H", "qr": "S·B·q_lora_rank",
        "query": "S·B·n_heads·v_head_dim", "key": "S·B·1·v_head_dim",
    },
}

#: **调用方声明的中间量形状**（`(seg, name) -> (符号串, 出处, 理由)`）。
#:
#: 为什么需要这个通路、以及它为什么不是"杜撰"：`shape_infer` 今天有两类**结构性**盲区 ——
#:   (a) **权重派生节点**（`ins` 为空、全部操作数是 `Parameter`，如
#:       `hyper_connection.py:408` `alpha = concat((alpha_pre, alpha_post, alpha_res), -1)`、
#:       `linear.py:132` 的权重转置）：权重被正确路由去 `param_operands`（契约 W2/W4），
#:       于是 `ins` 空 → 无从推形状，**尽管 `init_dims.param_shapes` 已经把权重形状读出来了**；
#:   (b) **opaque `Kernel` 的输出**（`npu_mhc_pre_sinkhorn`）：`shape_infer` 没有规则。
#: 两者串起来把 mHC 层的**主数据路**掐断：`alpha` → 内核 → `h_in` → 父帧的 `aggregated_*`。
#: 实测（本文档 §3 的反事实）：只补 `aggregated_{attn,ffn}` 一项，fused r4 层的已解析节点
#: 从 **10 → 75**（/196），`no_input_shape` 从 126 → 73。
#:
#: 纪律（三重，缺一条就是杜撰）：
#:   1. 值必须**逐字来自源**（下表每条带 `file:line`）；
#:   2. 该名字**不得被本图任何节点产出** —— 否则就是"覆盖推断结果"，`_declared_seeds` 断言之；
#:   3. 逐条记进 `Coverage.declared_shapes`，在覆盖度报告里**可见**（不与"推断出来的"混为一谈）。
_DECLARED_SHAPES = {
    ("decoder", "aggregated_attn"): (
        "S·B·H", "hyper_connection.py:397",
        "`HyperConnectionModule.construct` 的 Returns docstring 逐字："
        "`aggregated: [s, b, H] weighted input for the sublayer`。它是融合内核 "
        "`npu_mhc_pre_sinkhorn` 的第 0 个输出 `h_in`（:413）经 `:425 aggregated = h_in` 返回；"
        "内核输出无 shape 规则、`alpha`(:408) 又是纯权重派生 → 主数据路在此断开"),
    ("decoder", "aggregated_ffn"): (
        "S·B·H", "hyper_connection.py:397", "同上（同一个 Cell 的第二个实例 ffn_hc）"),
    ("mtp", "aggregated_attn"): ("S·B·H", "hyper_connection.py:397", "同 decoder"),
    ("mtp", "aggregated_ffn"): ("S·B·H", "hyper_connection.py:397", "同 decoder"),
}


def _declared_seeds(seg: str, dag, cov: "Coverage") -> dict:
    """本 segment 可用的**声明式**种子。已被图内某节点产出的名字 → fail-loud（纪律 2）。"""
    produced = {n.out.split(":", 1)[0] for n in dag.nodes if n.out}
    out: dict = {}
    for (s, name), (sym, src, why) in _DECLARED_SHAPES.items():
        if s != seg:
            continue
        if name in produced:
            raise RuntimeError(
                f"声明式 shape 种子 {name!r}（{src}）与图内节点的产出撞名 —— 那会**覆盖**推断"
                f"结果、把声明洗成事实。请改由 shape_infer 推出它，或改名后再声明。")
        out[name] = sym
        cov.declared_shapes.append((name, sym, src, why))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 5. `__init__` 求值的合并（dims_ctx / param 册）
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _InitBook:
    """链上各 Cell `__init__` 求值结果的合并视图。

    * `dims_ctx`：`merge_dims_ctx` 合并（同名不同值 fail-loud）；
    * `param`：**按 (文件名, 属性名) 索引** —— 全局按名合并会撞车（`Linear.weight`
      `(vocab, H)` vs `TopKRouter.weight` `(E, H)`），而 `Parameter` 声明与它的出现点在
      **同一个文件**里（`param_operands` 的 `src` 即该文件）→ 用文件名消歧是精确的。
    """
    dims_ctx: dict
    param: dict          # (basename, attr) -> (axes tuple, dtype str, cls)
    linear: dict         # (basename, attr) -> (in_dim str|None, out_dim str|None)


_TREE_CACHE: dict = {}
_LOCK = threading.RLock()


def _tree(root: str, rel: str):
    key = (root, rel)
    with _LOCK:
        if key not in _TREE_CACHE:
            path = os.path.join(root, *rel.split("/"))
            if not os.path.isfile(path):
                raise RuntimeError(f"权威快照缺文件：{path}。{_ROOT_HINT}")
            with open(path, encoding="utf-8") as fh:
                _TREE_CACHE[key] = ast.parse(fh.read())
        return _TREE_CACHE[key]


def _init_book(root: str, site: _Site, dims) -> _InitBook:
    """跑一遍 `_INIT_CHAIN` 的 `eval_init_dims`，合出 `dims_ctx` + param 册 + linear 册。"""
    flags = _cell_flags(dims, fused=site.fused, ratio=site.ratio, moe=site.moe)
    flags[INIT_PARAM_SEEDS] = {
        # `compress_ratio` 逐层传下去（csa.py:603 / indexer.py:129，缺省 0 而真机 4/128）。
        "compress_ratio": int(site.ratio),
        # `layer_number` 只参与 host 判定（csa.py:552）。
        "layer_number": 1,
        # `head_dim` **刻意不给**：`Compressor` 有两个构造点且 head_dim 不同
        # （csa.py:604 `config.v_head_dim`=512 vs indexer.py:128 `self.index_head_dim`=128），
        # extractor 不传播构造实参 → 给一个全局值必然把另一处算错。见 opdag_bytes §5 的 G2。
    }
    ctxs, pbook, lbook = [], {}, {}
    for cls, rel in _INIT_CHAIN:
        base = rel.rsplit("/", 1)[-1]
        try:
            idm = eval_init_dims(_tree(root, rel), cls, flags)
        except Exception:                       # noqa: BLE001 —— 单个类求不出不该毒死整链
            continue
        ctxs.append(idm.dims_ctx)
        for k, axes in (idm.param_shapes or {}).items():
            pbook.setdefault((base, k), (tuple(axes),
                                         (idm.param_dtypes or {}).get(k), cls))
        for k, v in (idm.linear_dims or {}).items():
            lbook.setdefault((base, k), v)
    return _InitBook(dims_ctx=merge_dims_ctx(*ctxs), param=pbook, linear=lbook)


# ═══════════════════════════════════════════════════════════════════════════
# 6. segment 抽取（每个 (site, segment) 一次，进程内缓存）
# ═══════════════════════════════════════════════════════════════════════════

_INJECTED = {"rotary_pos_emb": Binding("Constant", {"rope_freqs": True})}
_SEG_CACHE: dict = {}


def _bare(extra=()) -> dict:
    out = {n: ResolvedSpec(cell=n, submodules={}) for n in _BARE_CELLS}
    for n in extra:
        out[n] = ResolvedSpec(cell=n, submodules={})
    return out


#: 融合 mHC 内核的 saved 集 —— **源读**（`hyper_parallel` 已在权威快照里 @ `41495aa2`，
#: 4 跳证据链见 `docs/opdag_bytes_blockers_2026-07-25.md` §1.2）：
#:   `npu_mhc_post`          `custom_op_impl.py:331`     `save_for_backward(x, h_res, h_out, h_post)`
#:   `npu_mhc_pre_sinkhorn`  `custom_op_impl.py:390-391` `save_for_backward(x, phi, alpha, bias,
#:                                                        h_pre, hc_before_norm, inv_rms,
#:                                                        sum_out, norm_out)`
#: 两者被保存的**输入**恰好是各自全部张量实参（`phi`/`bias` 是 `Parameter` → 被 walker 路由去
#: `param_operands`，按契约 W2 不进 `saves`）→ `saved_ins_idx="all"` 在**输入侧是源真值**。
#: ⚠ 诚实边界：`pre_sinkhorn` 还保存 **5 个自身输出**（`sum_out` = `2·num_iters·B·S·N`、
#: `norm_out` = `2·num_iters·B·S·N·N` 是其中最大两块）。`saved_ins_idx` 表达不了"自身输出"，
#: 且 `h_pre/hc_before_norm/inv_rms` 的末轴长度在未纳入快照的 `.cc` kernel 里 → 这 5 项
#: **不给数**、进 unresolved（`Coverage.kernel_own_output_saves_note`）。
_KERNEL_SAVES = {
    "npu_mhc_pre_sinkhorn": {
        "saved_ins_idx": "all",
        "source": "hyper_parallel/platform/mindspore/custom_ops/custom_op_impl.py:390-391",
        "reason": "源读：save_for_backward(x, phi, alpha, bias, …) 的输入侧 = 全部张量实参"
                  "（phi/bias 是 Parameter → param_operands，W2 不进 saves）；"
                  "另存的 5 个**自身输出**表达不了 → 显式 unresolved，不给数",
    },
    "npu_mhc_post": {
        "saved_ins_idx": "all",
        "source": "hyper_parallel/platform/mindspore/custom_ops/custom_op_impl.py:331",
        "reason": "源读：save_for_backward(x, h_res, h_out, h_post) = 4 个输入全存",
    },
}


def _kernel_saves(root: str) -> dict:
    return dict(_KERNEL_SAVES)


def _extract(root: str, rel: str, cls: str, spec, flags: dict, **kw):
    return extract_cell(
        root, rel, cls, spec, flags,
        recurse=True, subcell_specs=_bare(kw.pop("extra_bare", ())), cross_file=True,
        runtime_predicates=dict(_RUNTIME_PREDICATES, **kw.pop("predicates", {})),
        host_call_allow=_HOST_ALLOW, kernel_call_allow=_KERNEL_ALLOW,
        kernel_saves=kw.pop("kernel_saves", None), input_axes=kw.pop("input_axes", None),
        **kw)


def _decoder_dag(root: str, site: _Site, dims):
    """一个 dsv4-hybrid 解码层（mHC 残差 + DSv4 稀疏注意力 + MoE/dense FFN）的整层图。"""
    flags = _cell_flags(dims, fused=site.fused, ratio=site.ratio, moe=site.moe)
    flags[INIT_PARAM_SEEDS] = {"compress_ratio": int(site.ratio), "layer_number": 1}
    top = resolve_layer_spec(root, _spec_flags(dims, moe=site.moe),
                             spec_files=PYNATIVE_SPEC_FILES)
    preds = {"mesh_dim_names": ("dp", "ep"), "call:get_moe_aux_loss_group_size": 1}
    return _extract(root, "pynative/transformers/transformer_layer.py",
                    "HyperConnectionTransformerLayer", top, flags,
                    predicates=preds, input_axes=_input_axes(dims),
                    kernel_saves=_kernel_saves(root),
                    injected_binds=_INJECTED, present_params={"rotary_pos_emb"})


def _embedding_dag(root: str, site: _Site, dims):
    flags = _cell_flags(dims, fused=site.fused, ratio=site.ratio, moe=site.moe)
    return _extract(root,
                    "pynative/base_models/common/embeddings/language_model_embedding.py",
                    "LanguageModelEmbedding",
                    ResolvedSpec(cell="LanguageModelEmbedding", submodules={}), flags,
                    input_axes=_input_axes(dims))


def _head_dags(root: str, site: _Site, dims):
    """lm_head 伪层 = vocab 投影（`Linear`）+ 交叉熵（`CrossEntropyLoss`）两段拼一层。

    hand_spec 侧对应 `hc_collapse / final_norm / lm_head / logsoftmax / nll` 五个 op；抽取侧
    **只**覆盖 `lm_head` 与 loss 两段 —— `hc_collapse` / `final_norm` 在 `GPTModel.construct`
    里（走查整个 `GPTModel` 是明确的非目标，见 opdag_component_coverage §6 第 1 项）。
    """
    flags = _cell_flags(dims, fused=site.fused, ratio=site.ratio, moe=site.moe)
    out = [("lm_head", _extract(root, "pynative/layers/linear.py", "Linear",
                                ResolvedSpec(cell="Linear", submodules={}), flags,
                                input_axes=_input_axes(dims)))]
    out.append(("loss", _extract(root, "pynative/loss/loss.py", "CrossEntropyLoss",
                                 ResolvedSpec(cell="CrossEntropyLoss", submodules={}), flags,
                                 input_axes=_input_axes(dims),
                                 present_params={"input_mask"})))
    return out


def _mtp_dag(root: str, site: _Site, dims):
    from .module_resolver import MTP_SPEC_FILES, resolve_spec_call
    flags = _cell_flags(dims, fused=site.fused, ratio=site.ratio, moe=site.moe)
    flags[INIT_PARAM_SEEDS] = {"compress_ratio": int(site.ratio), "layer_number": 1}
    sflags = _spec_flags(dims, moe=site.moe)
    top = resolve_layer_spec(root, sflags, spec_files=PYNATIVE_SPEC_FILES)
    mtp_spec = resolve_spec_call(
        root, "get_mtp_layer_spec", sflags, spec_files=MTP_SPEC_FILES,
        keyword={"transformer_layer_spec": top, "normalization": "RMSNorm",
                 "fused_norm": True, "hc_head": "HyperConnectionHead"})
    preds = {"mesh_dim_names": ("dp", "ep"), "call:get_moe_aux_loss_group_size": 1}
    return _extract(root, "pynative/transformers/multi_token_prediction.py",
                    "MultiTokenPredictionLayer", mtp_spec, flags,
                    predicates=preds, input_axes=_input_axes(dims),
                    kernel_saves=_kernel_saves(root), injected_binds=_INJECTED,
                    param_cells={"embedding": "LanguageModelEmbedding"},
                    present_params={"rotary_pos_emb", "position_ids", "attention_mask",
                                    "embedding", "actual_seq_len"})


def _segment_dags(root: str, kind: str, site: _Site, dims):
    """`(kind, site)` → `[(seg_tag, dag), ...]`（进程内缓存；抽取是纯 AST，无副作用）。"""
    key = (root, kind, site)
    with _LOCK:
        hit = _SEG_CACHE.get(key)
    if hit is not None:
        return hit
    if kind == "embedding":
        got = [("embedding", _embedding_dag(root, site, dims))]
    elif kind == "dsv4hyb":
        got = [("decoder", _decoder_dag(root, site, dims))]
    elif kind == "lm_head":
        got = _head_dags(root, site, dims)
    elif kind == "mtp":
        got = [("mtp", _mtp_dag(root, site, dims))]
    else:                                        # pragma: no cover —— 由 _layer_kind 先挡
        raise RuntimeError(f"不支持的 segment kind {kind!r}")
    with _LOCK:
        _SEG_CACHE[key] = got
    return got


# ═══════════════════════════════════════════════════════════════════════════
# 7. 折叠：`OpDAG` 节点 → `ROp`
# ═══════════════════════════════════════════════════════════════════════════

#: `OpDAG.op`（walker 的规范类型）→ 消费侧读的类型串。**只**对齐三个被真正判据用到的：
#:   `matmul` / `moe_gemm`（`is_muon_matrix_weight`、`moe_lids`）、`norm`（`_norm_save_names`
#:   的 fp32 抬升判据）、`flash_attn`。其余保持 walker 原名的小写（无消费方读它们）。
_OPTYPE = {
    "MatMul": "matmul", "BMM": "matmul", "GroupedMatMul": "moe_gemm",
    "Norm": "norm", "FlashAttention": "flash_attn",
}

#: `Softmax` **刻意不**映射成 `norm`：`structure_mem._norm_save_names` 排除名字含 softmax 的
#: norm op（softmax_compute_dtype 是另一回事）。给它自己的类型串更诚实。
_SOFTMAX_TYPES = ("Softmax",)


def _dim_value(expr: str, dims) -> int | None:
    """一个维度表达式（`n_heads·v_head_dim` / `2·ffn_hidden` / `S//4`）→ 整数值；解不出 None。"""
    if not expr:
        return None
    try:
        return axis_value(parse_axis(str(expr)), dims)
    except Exception:                            # noqa: BLE001
        return None


def _undo_norm_prelift(node, save):
    """撤掉 `derive_saves` 对 `Norm` 输入的 fp32 **预抬**（交接项 T1）。

    契约明文：「norm 的 fp32 cast 由消费侧 `structure_mem._dt` 按 op 类型再抬，**不**预抬」
    （`contract.py` 的 `ResolvedTensorContract` docstring）。而 `bprop_rules.py:127-130` 用
    `attrs["ln_compute_dtype"]` 覆盖了操作数 ref 里的 dtype → 若照抄就会被抬**两次**（×2）。
    这里按该 save 在本节点 `ins` 里的**原始** dtype 段还原；找不到就保留（回归安全）。
    """
    if node.op != "Norm":
        return save.dtype
    ln = node.attrs.get("ln_compute_dtype")
    if not ln or save.dtype != ln:
        return save.dtype
    for ref in node.ins:
        if ref.count(":") == 2 and ref.split(":", 1)[0] == save.name:
            return ref.split(":")[2]
    return save.dtype


class _Folder:
    """把一个 segment 的 `OpDAG` 折成 `[ROp]`，并把每一处"解不出"记进 `Coverage`。"""

    def __init__(self, dag, dims, pm, *, book: _InitBook, tag: str,
                 dims_for_file: dict = None):
        self.dag, self.dims, self.pm, self.book = dag, dims, pm, book
        self.cov = Coverage(tag=tag, n_nodes=len(dag.nodes))
        #: 某些文件里的符号要按**另一张** DimTable 求值（`shared_experts.py:52` 把
        #: `config.ffn_hidden_size` 改写成 shared 尺寸 → MoE 层里 `mlp.py` 链的 `ffn_hidden`
        #: 必须按 `moe_shared_F` 而非 `F` 求值；`consumer._SYM2FIELD` 是全局映射，表达不了改写）。
        self.dims_for_file = dict(dims_for_file or {})
        self._t: dict = {}                       # name -> RTensor（首见定型）
        self.report: list = []
        self._detached: set = set()
        self._census_seen: set = set()            # param census 去重（按名）
        self._graph_seen: set = set()             # 真正进图的权重去重（按名）

    # ── 记账小工具 ────────────────────────────────────────────────────────
    @staticmethod
    def _new_bytes(tensors, seen: set) -> int:
        """这批张量里**首次出现**（按名去重）的字节和。"""
        tot = 0
        for t in tensors:
            if t.name in seen:
                continue
            seen.add(t.name)
            tot += t.local_numel * t.dtype_bytes
        return tot

    def _skip(self, node, reason: str) -> None:
        rec = (node.id, node.op, node.src, reason)
        self.cov.skipped_ops.append(rec)
        if self.cov.first_skipped is None:
            self.cov.first_skipped = rec         # 执行序上第一个 = 级联的根

    # ── 求值上下文 ────────────────────────────────────────────────────────
    def _dims_of(self, src: str):
        base = (src or "").rsplit("/", 1)[-1].split(":", 1)[0]
        return self.dims_for_file.get(base, self.dims)

    # ── 张量 ─────────────────────────────────────────────────────────────
    def _tensor(self, ref: str, node, *, is_weight=False, dtype_override=None):
        """`"name:符号shape:dtype"` → `RTensor`；解不出 → None（**不零填**）。"""
        if not ref or ref.count(":") != 2:
            return None
        name, sym, dt = ref.split(":")
        dt = dtype_override or dt
        prev = self._t.get(name)
        if prev is not None:
            return prev
        dims = self._dims_of(getattr(node, "src", ""))
        try:
            elems = local_shape_elems(sym, dims, self.pm, is_weight=is_weight)
        except ValueError:                        # 不整除 —— 显式记账，不取整
            elems = None
        if not elems or elems <= 0:
            return None
        nb = _dtype_bytes(dt, dims)
        if nb <= 0:
            return None
        det = (not is_weight) and name in self._detached
        t = RTensor(name=name, local_numel=int(elems), dtype_bytes=int(nb),
                    is_weight=is_weight, detached=det,
                    dim0=self._dim0(sym, dims), sym_shape=sym, dtype_name=dt,
                    src=getattr(node, "src", ""))
        self._t[name] = t
        return t

    def _dim0(self, sym: str, dims) -> int:
        """首维（FSDP 判定用，`structure_mem.py:100`）。`~` numel-only 档无轴结构 → 0=未知。"""
        from .sym_shape import strip_numel_only
        s, numel_only = strip_numel_only((sym or "").strip())
        if numel_only or not s or s == "?":
            return 0
        from .sym_shape import parse_shape
        try:
            axes = parse_shape(s)
        except Exception:                         # noqa: BLE001
            return 0
        if not axes:
            return 0
        v = axis_value(axes[0], dims)
        return int(v) if v and v > 0 else 0

    # ── 权重 ─────────────────────────────────────────────────────────────
    def _weights_of(self, node) -> list:
        """该节点**拥有**的权重（梯度根）。两条源侧路径，缺一条就 unresolved（不猜）。

        A. `Parameter(mint.empty(...))` 声明：`init_dims.param_shapes`（按文件名消歧）。
        B. 叶子 `Linear`（`build_module(sub.X, input_size=A, output_size=B)` 被
           `LEAF_OPTYPE` 解成单个 `MatMul`，**不**递归进 `Linear.construct` → `self.weight`
           从未被 walker 看到）：由节点 `attrs["in_dim"]/["out_dim"]` + [SRC]
           `pynative/layers/linear.py:81-83` `weight_shape = (output_size, input_size)`
           / `dtype=self.params_dtype` 算出。
        """
        out = []
        dims = self._dims_of(node.src)
        base = (node.src or "").rsplit("/", 1)[-1].split(":", 1)[0]
        # ── A ──────────────────────────────────────────────────────────────
        for pn in (node.attrs.get("param_operands") or ()):
            rec = self.book.param.get((base, pn))
            if rec is None:
                self.cov.unresolved_params.append(
                    (pn, node.src, "Parameter 形状未由 __init__ 求出（或声明不在同一文件）"))
                continue
            axes, dtn, _cls = rec
            is_expert = node.op == "GroupedMatMul"
            sym = "·".join(axes)
            shard = {0: "ep"} if is_expert else None
            try:
                elems = local_shape_elems(sym, dims, self.pm, shard=shard, is_weight=True)
            except ValueError as e:
                self.cov.unresolved_params.append((pn, node.src, f"分片不整除：{e}"))
                continue
            if not elems:
                self.cov.unresolved_params.append((pn, node.src, f"符号未解析：{sym}"))
                continue
            d0 = self._dim0(sym, dims)
            if is_expert and d0:
                deg = self.pm.degree("ep")
                d0 = d0 // deg if d0 % deg == 0 else d0
            nb = _dtype_bytes(dtn, dims)
            name = f"{pn}@{base}"
            t = RTensor(name=name, local_numel=int(elems), dtype_bytes=int(nb),
                        is_weight=True, is_expert=is_expert, dim0=int(d0),
                        sym_shape=sym, dtype_name=str(dtn), src=node.src)
            prev = self._t.setdefault(name, t)
            if (prev.local_numel, prev.dtype_bytes) != (t.local_numel, t.dtype_bytes):
                self.cov.shape_conflicts.append(
                    (name, (prev.local_numel, prev.dtype_bytes),
                     (t.local_numel, t.dtype_bytes)))
                continue
            out.append(prev)
            self.cov.resolved_params.append(
                (name, sym, str(dtn), prev.local_numel * prev.dtype_bytes))
        # ── B ──────────────────────────────────────────────────────────────
        if node.module == "Linear" and node.op == "MatMul":
            i_dim, o_dim = node.attrs.get("in_dim"), node.attrs.get("out_dim")
            iv, ov = _dim_value(i_dim, dims), _dim_value(o_dim, dims)
            if iv is None or ov is None:
                self.cov.unresolved_params.append(
                    (f"weight@{node.src}", node.src,
                     f"叶子 Linear 的 in_dim/out_dim 未解出（in={i_dim!r} out={o_dim!r}）"))
            else:
                # `params_dtype` 缺省 = `compute_dtype`（linear.py:73-76）→ DimTable.dtype_bytes。
                nb = int(dims.dtype_bytes)
                name = f"w@{node.src}"
                t = RTensor(name=name, local_numel=int(iv * ov), dtype_bytes=nb,
                            is_weight=True, dim0=int(ov),
                            sym_shape=f"({o_dim})·({i_dim})", dtype_name="params_dtype",
                            src=node.src)
                prev = self._t.setdefault(name, t)
                out.append(prev)
                self.cov.resolved_params.append(
                    (name, t.sym_shape, "params_dtype", t.local_numel * nb))
        return out

    # ── 主循环 ───────────────────────────────────────────────────────────
    def fold(self, seed_shapes: dict) -> tuple:
        dag = self.dag
        infer_shapes(dag, seed_shapes, dims_ctx=self.book.dims_ctx, report=self.report)
        for r in self.report:
            k = r.get("reason", "?")
            self.cov.node_gap_reasons[k] = self.cov.node_gap_reasons.get(k, 0) + 1
        why = {}
        for r in self.report:
            why.setdefault(r.get("name") or "", (r.get("src", ""), r.get("reason", "")))
        self._detached = set(dag.detached or ())
        alias = detach_aliases(dag)
        saves_by_op: dict = {}
        for s in derive_saves(dag):
            saves_by_op.setdefault(s.op_id, []).append(s)
        ops: list = []
        for n in dag.nodes:
            # ── param census 先做，**与激活数据流无关** ──────────────────────────
            # 权重形状来自 `__init__`（`Parameter(mint.empty(...))` / `build_module` 的
            # `input_size=/output_size=`），不依赖 shape 推断。故即使承载它的节点因激活
            # 解不出而被跳过，这个权重**仍然可解析** —— 只是它没能进图。两个口径都记，
            # 差额就是「解出来了但丢了」（`Coverage.param_census_bytes` vs `param_in_graph_bytes`）。
            pars = self._weights_of(n)
            self.cov.param_census_bytes += self._new_bytes(pars, self._census_seen)
            if not n.out:
                self._skip(n, "no-output")
                continue
            out = self._tensor(n.out, n)
            ins = [self._tensor(r, n) for r in n.ins]
            if out is None or any(i is None for i in ins):
                bad = ([n.out.split(":")[0]] if out is None else []) + [
                    r.split(":")[0] for r, t in zip(n.ins, ins) if t is None]
                src, reason = why.get(bad[0], ("", "shape-unresolved"))
                self._skip(n, f"{reason} @ {src or n.src} ({','.join(bad[:3])})")
                continue
            self.cov.param_in_graph_bytes += self._new_bytes(pars, self._graph_seen)
            sv = []
            for s in saves_by_op.get(n.id, ()):
                root = alias.get(s.name, s.name)
                dt = _undo_norm_prelift(n, s)
                st = self._tensor(f"{s.name}:{s.sym_shape}:{dt}", n)
                if st is None:
                    src, reason = why.get(s.name, (n.src, "save-shape-unresolved"))
                    self.cov.unresolved_saves.append((s.name, s.sym_shape, src, reason))
                    continue
                if root != s.name:
                    # `stop_gradient` 不复制存储 → 真机上它与 root 同一块内存。`ResolvedLayer`
                    # 表达不了"零字节别名"，故仍各占一份 = **过计**，逐条记账（不静默）。
                    self.cov.detach_alias_overcount.append(
                        (s.name, root, st.local_numel * st.dtype_bytes))
                sv.append(st)
            ops.append(ROp(name=f"n{n.id}_{n.op.lower()}", type=self._optype(n),
                           inputs=tuple(ins), output=out, params=tuple(pars),
                           saves=tuple(sv), src=n.src))
        self.cov.n_ops = len(ops)
        return tuple(ops), self.cov

    def _optype(self, node) -> str:
        if node.op in _SOFTMAX_TYPES:
            return "softmax"
        return _OPTYPE.get(node.op, node.op.lower())


# ═══════════════════════════════════════════════════════════════════════════
# 8. 层装配
# ═══════════════════════════════════════════════════════════════════════════

def _layer_kind(layer_type: str) -> str:
    """`layer_pattern` 的取值 → segment kind。不认识就 fail-loud（绝不产一个空层）。"""
    lt = str(layer_type)
    if lt == "embedding":
        return "embedding"
    if lt.startswith("dsv4hyb"):
        return "dsv4hyb"
    if lt in ("lm_head", "head"):
        return "lm_head"
    if lt.startswith("mtp"):
        return "mtp"
    raise RuntimeError(
        f"graph source 'extracted' 尚不支持 layer_type {layer_type!r}（已支持前缀："
        f"{SUPPORTED_LAYER_KINDS}）。本来源目前只覆盖 pynative DSv4-Flash 模型；"
        f"不认识的层型**不产空层**（空层会静默给出一个偏小的峰值）。")


def _ratio_of(layer_type: str) -> int:
    """`dsv4hyb_r4_moe` → 4。层型串是 `build_llm` 按 `compress_ratios` 逐层生成的权威载体。"""
    for part in str(layer_type).split("_"):
        if part.startswith("r") and part[1:].isdigit():
            return int(part[1:])
    raise RuntimeError(f"从 layer_type {layer_type!r} 里读不出 compress_ratio")


def _build_layer(root: str, dims, pm, *, layer_id: int, layer_type: str, site: _Site):
    kind = _layer_kind(layer_type)
    dims_for_file = {}
    if site.moe:
        # MoE 层里唯一的 MLP 是 shared expert（`shared_experts.py:52` 改写
        # `config.ffn_hidden_size`）→ `mlp.py` 链的 `ffn_hidden` 按 `moe_shared_F` 求值。
        dims_for_file["mlp.py"] = replace(dims, F=int(dims.moe_shared_F or dims.F))
    book = _init_book(root, site, dims)
    ops: list = []
    cov = Coverage(tag=f"L{layer_id}:{layer_type}")
    segs = _segment_dags(root, kind, site, dims)
    for seg_tag, dag in segs:
        folder = _Folder(dag, dims, pm, book=book,
                         tag=f"L{layer_id}:{layer_type}/{seg_tag}",
                         dims_for_file=dims_for_file)
        seeds = dict(_SEED_SHAPES.get(seg_tag, ()))
        seeds.update(_declared_seeds(seg_tag, dag, folder.cov))
        seg_ops, seg_cov = folder.fold(seeds)
        # segment 之间**重命名**：两段拼一层时同名张量（`output` / `weight`）不是同一物理量。
        pre = f"{seg_tag}." if len(segs) > 1 else ""
        ops.extend(_rename(seg_ops, pre) if pre else seg_ops)
        cov.children.append(seg_cov)
        cov.n_nodes += seg_cov.n_nodes
        cov.n_ops += seg_cov.n_ops
    return ExtractedLayer(layer_id=layer_id, layer_type=str(layer_type),
                          ops=tuple(ops), coverage=cov), cov


def _rename(ops, prefix: str):
    memo: dict = {}

    def rn(t: RTensor) -> RTensor:
        key = t.name
        if key not in memo:
            memo[key] = replace(t, name=prefix + t.name)
        return memo[key]

    return tuple(replace(op, inputs=tuple(rn(t) for t in op.inputs), output=rn(op.output),
                         params=tuple(rn(t) for t in op.params),
                         saves=tuple(rn(t) for t in op.saves),
                         name=prefix + op.name)
                 for op in ops)


# ═══════════════════════════════════════════════════════════════════════════
# 9. 约定入口点
# ═══════════════════════════════════════════════════════════════════════════

def _allow_partial(explicit) -> bool:
    if explicit is not None:
        return bool(explicit)
    return str(os.environ.get(ALLOW_PARTIAL_ENV, "")).strip().lower() not in ("", "0",
                                                                             "false", "no")


def resolve_graph(model_spec, parallel_model, *, allow_partial=None) -> ExtractedGraph:
    """**`extracted` graph source 的约定入口点**（`liveness/sources.py::EXTRACTED_ENTRY_POINT`）。

    `(ModelSpec, ParallelModel)` → `ExtractedGraph`（带 `.stages` + `.coverage`），逐层满足
    `cost_eval/liveness/contract.py` 的 18 条硬规则。

    层序 / 层号 / stage 归属**完全沿用** `ModelSpec.layer_pattern` 与 `pm.stage_of(layer_id)`
    ——与 `hand_spec` 逐项对齐，故 A/B 的差异只可能来自**层内的图**，不可能来自层的编排
    （重算域 `rc.is_full(lid)`、loss 层判定都按 layer_id 走）。

    ⚠ **部分解析 ⇒ 默认不给数**：解不出字节的节点被整个跳过（绝不零填）→ 峰值只是**下界**。
    这种图默认**不返回**，而是抛 `IncompleteExtraction`（消息里带缺什么、缺在哪一行）——
    见该异常的论证。要拿下界做诊断：`allow_partial=True` 或设
    `COST_EVAL_EXTRACTED_ALLOW_PARTIAL=1`；返回的图仍带 `.coverage`。
    """
    dims = model_spec.dims
    for axis in ("tp", "cp"):
        if parallel_model.degree(axis) > 1:
            raise RuntimeError(
                f"graph source 'extracted' 尚不支持 {axis}={parallel_model.degree(axis)}>1："
                "抽出的张量**没有 TP/CP placement 标注**（`OpDAG` 里没有 shard 轴概念），"
                "按全局 numel 记账会成倍偏大、按猜的轴切会静默错 → fail-loud。"
                "（167 A/B 八跑均为 tp=cp=1，故不影响验收门。）")
    root = mf_root()
    _verify_snapshot(root)
    stages: dict = {}
    graph_cov = Coverage(tag="extracted")
    dkey = _dims_key(dims)
    for layer_id, ltype in enumerate(model_spec.layer_pattern):
        kind = _layer_kind(ltype)
        site = _Site(fused=bool(getattr(dims, "dsa_fused", True)),
                     ratio=(_ratio_of(ltype) if kind == "dsv4hyb" else 0),
                     moe=(kind == "dsv4hyb" and "moe" in str(ltype)),
                     dims_key=dkey)
        layer, cov = _build_layer(root, dims, parallel_model,
                                  layer_id=layer_id, layer_type=ltype, site=site)
        graph_cov.children.append(cov)
        stages.setdefault(parallel_model.stage_of(layer_id), []).append(layer)
    g = ExtractedGraph(stages=stages, coverage=graph_cov)
    if graph_cov.is_partial and not _allow_partial(allow_partial):
        t = graph_cov.totals()
        bl = "；".join("%s %s（×%d）" % (op, src, k)
                       for (src, op, _r), k in graph_cov.blockers(3)) or "无"
        raise IncompleteExtraction(
            "抽出的图尚不完备，故**拒绝交出峰值**（部分解析的图的峰值是下界，不是估计）："
            f"节点 {t['n_nodes']} → op {t['n_ops']}（跳过 {t['skipped_ops']}）；"
            f"param 可解析 {t['param_census_bytes'] / MiB:.1f} MiB / 真正进图 "
            f"{t['param_in_graph_bytes'] / MiB:.1f} MiB；saves 未解析 {t['unresolved_saves']}。"
            f"级联根：{bl}。"
            f"要拿这个下界做诊断请设 {ALLOW_PARTIAL_ENV}=1 或传 allow_partial=True。\n"
            + graph_cov.report(), graph_cov)
    return g
