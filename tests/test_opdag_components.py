# -*- coding: utf-8 -*-
"""组件覆盖补全 —— mHC / MoE / embedding / lm_head / loss / MTP 的纯 AST 抽取。

背景:`docs/opdag_walker_core_2026-07-25.md` §7 第 7 项把「mHC 层 / MTP / pynative MoE /
pynative embedding+loss」列为**未做**;`docs/opdag_coverage_assessment_2026-07-25.md`
§3.2/§3.3/§3.4/§3.5 逐条给了失败诊断。本文件把这些组件带到与四个 dsv4_hybrid 注意力 Cell
**同一标准**:

  * 抽出**非空**图;
  * **零诊断**(五类 `DIAG_KINDS` 全 0),或显式允许清单(逐条给字节理由);
  * `strict=True` 放行;
  * 逐节点可 dump(op 类型 / `file:line` / ins / out / saves / detached)。

**md5 门控**:凡断言真源行号/结构的测试都先校验权威快照 md5 —— `tests/conftest.py` 的
`MF_ROOT` 默认仍指**非权威**树(`docs/opdag_source_truth_2026-07-25.md` §4 第 7 条),
对着另一个 commit 断言行号是无意义的。
"""
import hashlib
import os

import pytest

from cost_eval.opdag.bprop_rules import derive_saves
from cost_eval.opdag.construct_walker import (
    DIAG_KINDS, ExtractionDroppedError, diagnostics_summary,
)
from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.init_binder import Binding
from cost_eval.opdag.module_resolver import (
    PYNATIVE_SPEC_FILES, ResolvedSpec, resolve_layer_spec,
)

# ── 权威快照门控 ────────────────────────────────────────────────────────────────────
_MD5 = {
    # 锚定件(与 tests/test_opdag_walker_core.py 同口径,证明是同一 commit)
    "pynative/transformers/experimental_attention_variant/csa.py":
        "81673be3ad3cdd2191e28dd000a13f0e",
    "pynative/transformers/experimental_attention_variant/indexer.py":
        "8e18fee21c33507dd629c89f401986e6",
    # 本文件断言的目标件
    "pynative/transformers/hyper_connection.py": "14cf0de51c41df31f5e4439549934660",
    "pynative/transformers/transformer_layer.py": "0b1cce0c7d620544f01a2bea1014fa3a",
    "pynative/transformers/moe/moe_layer.py": "ae0943edc71ba6dbda78077e5df584a2",
    "pynative/transformers/moe/router.py": "24da5963752dc29697ab9fc2f6a2a79c",
    "pynative/transformers/moe/experts.py": "9c53394ade35674fd007866aafdbdae1",
    "pynative/transformers/moe/shared_experts.py": "18d9ba7f8039202801763dbbc89b0b0b",
    "pynative/base_models/common/embeddings/vocab_embedding.py":
        "5d214e4adcd800a7292494209b842349",
    "pynative/base_models/common/embeddings/language_model_embedding.py":
        "dae1316621f327d5ac995f40ffb97d6e",
    "pynative/loss/loss.py": "1370a9f4130c674ca2adea8e45326a33",
    "pynative/transformers/multi_token_prediction.py":
        "34bb535e1c12574f3cf11580a604fe40",
    "pynative/base_models/gpt/gpt_model.py": "0a6f87b1481af7bf71604c4fee123227",
}


def _authoritative_pkg():
    for root in (os.environ.get("MINDFORMERS_ROOT"),
                 r"E:\97-codes\torch_parallel\mf-src-167"):
        if not root:
            continue
        pkg = os.path.join(root, "mindformers")
        ok = True
        for rel, want in _MD5.items():
            p = os.path.join(pkg, *rel.split("/"))
            if not os.path.isfile(p):
                ok = False
                break
            with open(p, "rb") as fh:
                if hashlib.md5(fh.read()).hexdigest() != want:
                    ok = False
                    break
        if ok:
            return pkg
    return None


@pytest.fixture(scope="module")
def mf_pkg():
    pkg = _authoritative_pkg()
    if pkg is None:
        pytest.skip("权威 mindformers 快照缺失/md5 不匹配(绝不对着另一个 commit 断言)")
    return pkg


# ── 抽取配置(逐值取自 dsv4h_fused_pp4_recomp.yaml 或源码逐字)──────────────────────────
SPEC_FLAGS = {
    "num_experts": 8, "moe_grouped_gemm": True, "qk_layernorm": True,
    "multi_latent_attention": True, "enable_hyper_connections": True,
    "fused_norm": True, "normalization": "RMSNorm", "is_dsv4_hybrid": True,
}

# 部署形态谓词(`hasattr` / `isinstance`):真机 pynative 把 Parameter 包成 hyper_parallel
# DTensor,故 `to_local` 存在(`_to_local` @ hyper_connection.py:29-31 就是为此写的)。
# 部署形态 / 运行相位事实表(调用方**显式声明**,缺键 fail-loud —— 绝不默认取某一支)。
RUNTIME_PREDICATES = {
    "hasattr:to_local": True, "hasattr:detach": True, "isinstance:DTensor": True,
    # yaml parallelism: pipeline=4 / dp_shard=2 / expert=2 / tensor=1 / context=1
    # → device mesh 的维名单里**有** "ep"(`moe/experts.py:181` 的 need_dispatch 靠它判)
    "mesh_dim_names": ("dp", "ep"),
    # 前向图建模的是**首次前向**,不是重算中的那次(`moe/moe_layer.py:126`)
    "call:is_in_recompute": False,
    # `moe_utils.py:261-263` 的模块级全局 `_AUX_LOSS_GROUP_SIZE`(tp×cp 组大小);tp=cp=1 → 1
    "call:get_moe_aux_loss_group_size": 1,
}
HOST_ALLOW = ("save_to_indexer_losses_tracker", "get_indexer_loss_tracker",
              # `moe_utils.py:377-...`:把逐层 aux loss(标量)累加进模块级 dict,无返回值张量
              # —— 与 `save_to_indexer_losses_tracker` 完全同一条判据(walker_core 文档 §6.2)。
              "save_to_aux_losses_tracker", "get_moe_layer_wise_logging_tracker")


def cell_flags(**over) -> dict:
    """dsv4h_fused_pp4_recomp.yaml 的逐值 + `__init__` 派生量的**显式注入**。

    为什么派生量必须显式注入(而不是让 `init_dims` 静态求 `__init__`):
    `docs/opdag_walker_core_2026-07-25.md` §2.1 的实测反例 —— `compress_ratio` 的
    `__init__` 缺省是 `0`(csa.py:556),静态求值会得出 `enable_compress=False`,**与真机相反**。
    """
    f = {
        "compute_dtype": "bf16", "layernorm_compute_dtype": "fp32", "params_dtype": "bf16",
        "v_head_dim": 512, "qk_pos_emb_head_dim": 64, "q_lora_rank": 1024,
        "num_attention_heads": 64, "hidden_size": 4096, "o_groups": 8, "o_lora_rank": 1024,
        "add_bias_linear": False, "input_layout": "BSND", "num_layers": 8, "mtp_num_layers": 0,
        "index_topk": 512, "index_n_heads": 64, "index_head_dim": 128,
        "csa_window_size": 128, "csa_dense_mode": False,
        "compress_ratio": 4, "enable_compress": True, "enable_indexer": True,
        "is_tnd": False, "window_size": 128, "training": True,
        "apply_dsa_kernel_fusion": True,
        "rotate": True, "overlap": True, "coff": 2,
        "sparse_loss": True, "use_butterfly": False,
        "seq_length": 4096, "micro_batch_size": 1, "dsa_indexer_loss_coeff": 0.001,
        # ── mHC(yaml:enable_hyper_connections/hc_mult=4/use_fused_mhc/hc_sinkhorn_iters=20)
        "num_residual_streams": 4, "use_fused_mhc": True,
        "mhc_sinkhorn_iterations": 20, "mhc_layernorm_epsilon": 1e-6,
        # `self.dtype = config.compute_dtype`(hyper_connection.py:84/141/205)——
        # `self.cast(x, self.dtype)`(:406/:124/:362)的目标 dtype
        "dtype": "bf16",
        "apply_residual_connection_post_layernorm": False, "hidden_dropout": 0.0,
        "use_dropout": False,       # Dropout.__init__:69 派生量(drop_prob != 0)
        # ── MoE(yaml:n_routed_experts=8 / num_experts_per_tok=2 / n_shared_experts=1 /
        #    norm_topk_prob=true / n_group=0 / moe_aux_loss_coeff=0.001 / seq_aux_loss)
        "moe_router_topk": 2, "gated_linear_unit": True, "activation_type": "silu",
        "moe_ffn_hidden_size": 2048, "moe_shared_expert_intermediate_size": 2048,
        "ffn_hidden_size": 2048, "hidden_act": "silu", "activation_func_clamp_value": None,
        "moe_router_dtype": "fp32", "num_moe_experts": 8, "moe_permute_fusion": False,
        # TopKRouter.__init__ 派生量(显式注入 —— 不静态求 __init__,理由见 cell_flags docstring)
        "score_func": "softmax",            # config 缺省(transformer_config.py:595),yaml 未覆盖
        "is_hash_layer": False,             # moe_n_hash_layers 缺省 0(:1304)
        "num_expert_groups": 0,             # yaml n_group: 0
        "route_norm": True,                 # yaml norm_topk_prob: true
        "route_scale": 1.0,                 # moe_router_topk_scaling_factor 缺省 None → 1.0
        "_debug_force_load_balance": False,  # 缺省 False(:584)
        "moe_aux_loss_coeff": 0.001,        # yaml
        "aux_loss_type": "seq_aux_loss",    # yaml moe_router_load_balancing_type
        "top_k": 2, "calculate_per_token_loss": False,
        "_tp_size": 1, "_cp_size": 1, "_tp_group": None, "_cp_groups": (),
        # MoELayer / MLP / SharedExpertMLP 的 __init__ 派生量
        "shared_expert_num": 1, "enable_expert_bias": False,
        "score_before_experts": False, "moe_apply_probs_on_input": False,
        "use_shared_expert_gating": False, "use_shared_expert_gate": False,
        "use_clamped_swiglu": False,
        # ── embedding(`position_embedding_type='rope'`,gpt_model.py:196)
        "add_position_embedding": False,     # :73 == 'learned_absolute' → False
        "num_tokentypes": 0,
        "tokentype_embeddings": None,        # :91 —— num_tokentypes=0 ⇒ None
        "position_embeddings": None,         # :81
        "embedding_dropout_prob": 0.0, "vocab_size": 129280, "embedding_dim": 4096,
        # ── loss(yaml tensor_parallel=1 ⇒ enable_vocab_parallel 未调用 ⇒ _tp_group is None)
        "chunk_loss_num": 1, "compensate_loss_sense_tp": True,
        # ── MTP(enable_hc_head 缺省 None ⇒ 跟随 enable_hyper_connections = True)
        "hc": True, "hc_num_streams": 4, "hc_hidden_size": 4096, "enable_hc_head": True,
        # `self.dtype = config.compute_dtype`(hyper_connection.py:84/141/205)
        "dtype": "bf16",
    }
    f.update(over)
    # `SinkhornKnopp.__init__:45` 的 `self.iterations = iterations`,实参链:
    #   `config.mhc_sinkhorn_iterations`(hyper_connection.py:206)
    #   → `SinkhornKnopp(self.sinkhorn_iterations)`(:232) → `self.iterations`
    # 循环 `for _ in range(self.iterations - 1)`(:63)按它展开 → 必须显式给。
    f.setdefault("iterations", f["mhc_sinkhorn_iterations"])
    if "mhc_sinkhorn_iterations" in over:
        f["iterations"] = over["mhc_sinkhorn_iterations"]
    return f


INJECTED = {"rotary_pos_emb": Binding("Constant", {"rope_freqs": True})}

# 入口形参的轴符号种子(与 `infer_shapes(dag, input_shapes)` 同契约)。源侧 shape 契约逐字来自
# docstring / 注释:`HyperConnectionModule.construct(hidden_states: [s,b,n*H])`
# (hyper_connection.py:250)、`Compressor.construct(x: [sq,b,hidden_size])`(compressor.py:179)、
# `CompressedSparseAttention.construct(query/key/x/qr)`(csa.py:648-651)。
# **不给种子就是 `_UNKNOWN_SCALAR` → 依赖它的 if 不可判定 → fail-loud(绝不杜撰尺寸)**。
INPUT_AXES = {
    "hidden_states": ("seq_length", "micro_batch_size", "n_hidden"),
    "x":      ("seq_length", "micro_batch_size", "hidden_size"),
    "qr":     ("seq_length", "micro_batch_size", "q_lora_rank"),
    "query":  ("seq_length", "micro_batch_size", "num_attention_heads", "v_head_dim"),
    "key":    ("seq_length", "micro_batch_size", 1, "v_head_dim"),
    # loss 段:`logits [b, s, vocab]`(`gpt_model.py:365` 后 transpose;`logits.ndim == 3`
    # @ `loss.py:363` 靠这份种子的**轴数**判定)
    "logits": ("micro_batch_size", "seq_length", "vocab_size"),
    "label":  ("micro_batch_size", "seq_length"),
    "input_mask": ("micro_batch_size", "seq_length"),
    "labels": ("micro_batch_size", "seq_length"),
    # MTP:`roll_tensor` 取 `shape[-1]`(`multi_token_prediction.py:174`)
    "input_ids": ("micro_batch_size", "seq_length"),
    "position_ids": ("micro_batch_size", "seq_length"),
    "input_": ("micro_batch_size", "seq_length"),
}


def clean(dag) -> None:
    """五类诊断全 0 —— 与四个 dsv4 Cell 同一条验收门。"""
    s = diagnostics_summary(dag)
    bad = {k: s[k] for k in DIAG_KINDS if s[k]}
    assert not bad, f"{dag.cell} 诊断非空: {bad}\n{_diag_text(dag)}"


def _diag_text(dag) -> str:
    from cost_eval.opdag.construct_walker import _format_diagnostics
    s = diagnostics_summary(dag)
    return _format_diagnostics(dag, [k for k in DIAG_KINDS if s[k]])


def dump(dag) -> str:
    """逐节点 dump(人工评审用):op 类型 / `file:line` / ins / out / SAVED / DETACHED / params。"""
    saves = {s.name for s in derive_saves(dag)}
    lines = [f"--- {dag.cell}: {len(dag.nodes)} nodes / {len(dag.edges)} edges ---"]
    for n in dag.nodes:
        out = n.out or "-"
        det = " DETACHED" if n.attrs.get("detached") else ""
        sv = " SAVED" if out.split(":")[0] in saves else ""
        pr = f" params={n.attrs.get('param_operands')}" if n.attrs.get("param_operands") else ""
        lines.append(f"#{n.id:<3d} {n.op:14s} {n.src:44s} ins={list(n.ins)} out={out}{det}{sv}{pr}")
    lines.append(f"saves({len(saves)}): {sorted(saves)}")
    lines.append(f"detached: {list(dag.detached)}")
    lines.append(f"param_operands: {[p.get('param') for p in dag.param_operands]}")
    return "\n".join(lines)


def srcs(dag) -> set:
    return {n.src for n in dag.nodes}


def outs(dag) -> set:
    return {n.out.split(":")[0] for n in dag.nodes if n.out}


# ══════════════════════════════════════════════════════════════════════════════════
# 1. mHC hyper-connection 残差
# ══════════════════════════════════════════════════════════════════════════════════
HC_REL = "pynative/transformers/hyper_connection.py"
LAYER_REL = "pynative/transformers/transformer_layer.py"

# `hyper_parallel.custom_ops.experimental` 的两个融合 mHC 内核**不在权威快照里**
# (`hyper_connection.py:20-23` 是 try/except ImportError 的外部包)。故它们的 saved 集
# **无法从源读出** —— 由调用方显式声明(带定位符与理由),未声明则 `derive_saves` fail-loud。
# 依据:`analysis/dsv4_calibration_final_report_2026-07-23.md:137` 实测记载
# 「aclnn 自定义算子(mHC/MLA,经 HP DFunction)saves 全走 `save_for_backward`
#   (custom_op_impl.py:331,390,588)」→ 张量实参保守全存(**上界声明**,非源读)。
MHC_KERNEL_SAVES = {
    "npu_mhc_pre_sinkhorn": {
        "saved_ins_idx": "all",
        "source": "hyper_parallel custom_op_impl.py:331/390/588（快照外）",
        "reason": "内核 bprop 不在权威快照里；按 HP DFunction 全走 save_for_backward "
                  "取张量实参全存的**保守上界**（调用方声明，非源读）",
    },
    "npu_mhc_post": {
        "saved_ins_idx": "all",
        "source": "hyper_parallel custom_op_impl.py:331/390/588（快照外）",
        "reason": "同上",
    },
}
MHC_KERNEL_ALLOW = tuple(MHC_KERNEL_SAVES)


def _hc_bare(fused: bool) -> dict:
    names = ("SinkhornKnopp", "HyperConnectionOutputCell", "FusedHyperConnectionOutputCell",
             "HyperConnectionModule", "FusedHyperConnectionModule", "HyperConnectionHead")
    return {n: ResolvedSpec(cell=n, submodules={}) for n in names}


def extract_hc(mf_pkg, fused: bool, strict: bool = True):
    cls = "FusedHyperConnectionModule" if fused else "HyperConnectionModule"
    return extract_cell(
        mf_pkg, HC_REL, cls, ResolvedSpec(cell=cls, submodules={}),
        cell_flags(use_fused_mhc=fused), recurse=True, subcell_specs=_hc_bare(fused),
        cross_file=True, runtime_predicates=RUNTIME_PREDICATES,
        host_call_allow=HOST_ALLOW, kernel_call_allow=MHC_KERNEL_ALLOW,
        kernel_saves=MHC_KERNEL_SAVES,
        input_axes=INPUT_AXES, strict=strict,
    )


def test_hyper_connection_module_unfused_extracts_clean(mf_pkg):
    """非融合 mHC(`HyperConnectionModule.construct` @ hyper_connection.py:246-301)全源可见。

    这是 mHC 的**字节真相路径**:每个 op 都在快照里逐行可读(fused 支的内核在快照外,见
    `MHC_KERNEL_SAVES` 的理由)。它同时是手写侧 `cost_eval/layers/residual.py`
    `build_hyper_connection_ops` 那 3 个 op 的源侧对照物。
    """
    dag = extract_hc(mf_pkg, fused=False)
    assert dag.nodes, "非融合 mHC 抽出空图"
    clean(dag)
    # RMSNorm(fp32, n·H) @ :262-266 —— 手写侧 `{prefix}_hc_norm` 的源侧对应
    assert any(n.op == "Norm" for n in dag.nodes), dump(dag)
    # mapping_proj:`Linear(n*H → 2n+n²)` @ :272-273
    assert any(n.op == "MatMul" for n in dag.nodes), dump(dag)
    # `SinkhornKnopp.construct` @ :52-68 —— `for _ in range(self.iterations - 1)` 静态展开
    assert any(n.src.endswith(":65") or n.src.endswith(":67") for n in dag.nodes), dump(dag)
    # aggregate:`h_pre @ x_streams` @ :299
    assert f"hyper_connection.py:299" in srcs(dag), dump(dag)


def test_sinkhorn_range_loop_is_unrolled_by_the_config_iteration_count(mf_pkg):
    """`for _ in range(self.iterations - 1)`(hyper_connection.py:63)按 config 的迭代数展开。

    这不是「猜」:`iterations` 来自 `config.mhc_sinkhorn_iterations`(:206 → :232 构造
    `SinkhornKnopp(self.sinkhorn_iterations)`),yaml 给 `hc_sinkhorn_iters: 20`。展开数
    必须**随注入值线性变化** —— 否则就是把循环体当一次算完(少算 19 轮的 [s,b,n,n] fp32 中间量)。
    """
    def n_loop_nodes(iters):
        dag = extract_cell(
            mf_pkg, HC_REL, "SinkhornKnopp", ResolvedSpec(cell="SinkhornKnopp", submodules={}),
            cell_flags(mhc_sinkhorn_iterations=iters), recurse=True,
            subcell_specs=_hc_bare(False), cross_file=True,
            runtime_predicates=RUNTIME_PREDICATES, strict=True,
        )
        clean(dag)
        return len([n for n in dag.nodes if n.src.split(":")[1] in ("65", "67")])

    n3, n5, n20 = n_loop_nodes(3), n_loop_nodes(5), n_loop_nodes(20)
    # 每轮 6 个节点:两行(`:65` / `:67`)各 `sum` + `add` + `div`。
    assert (n3, n5) == (12, 24), (n3, n5)
    assert n20 == 6 * (20 - 1) == 114, n20        # yaml `hc_sinkhorn_iters: 20` → 19 轮
    # 线性性(不是"恰好对上"的巧合):节点数 = 6·(iters-1)
    assert n5 - n3 == 6 * (5 - 3)


def test_optional_submodule_slots_resolve_to_their_declared_dataclass_default(mf_pkg):
    """`pre_cross_attn_layernorm` / `cross_attention` 未在 spec 里填 → 用**声明的缺省值**。

    源真相(`transformer_layer.py:68-75`):`TransformerLayerSubmodules` 六个字段的 dataclass
    缺省全是 `IdentityOp`。pynative 的 dsv4_hybrid spec 只填 4 个槽
    (`pynative/base_models/gpt/gpt_layer_specs.py:105-113`),另两个槽在真机上**确实被
    实例化**为 `IdentityOp`(`build_module(IdentityOp, ...)` → `IdentityOp(...)`,
    `spec_utils.py:76-77`)—— 不是「不存在」。此前 Pass A 只收显式填的字段,导致
    `_bind_build_module` 在这两个槽上 fail-loud、**整个 mHC 层零覆盖**(评估文档 §3.3)。
    """
    top = resolve_layer_spec(mf_pkg, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    assert top.cell == "HyperConnectionTransformerLayer"
    assert top.submodules["pre_cross_attn_layernorm"] == "Identity"
    assert top.submodules["cross_attention"] == "Identity"
    # 显式填的槽不受影响
    assert top.submodules["input_layernorm"] == "Norm"
    assert top.submodules["self_attention"].cell == "DSv4HybridSelfAttention"


def test_none_default_submodule_slot_stays_fail_loud(mf_pkg):
    """dataclass 缺省是 `None` 的槽**不许**被填成任何东西。

    `MLASelfAttentionSubmodules*` 的字段缺省全是 `None`
    (`parallel_core/training_graph/transformer/multi_latent_attention.py:56-62`)。
    `build_module(None, ...)` 在真机上会炸(`spec_utils.py:76-82` 走到
    `import_module(None.module)`)—— 所以 `None` 缺省的语义是「这条路径不会 build 它」,
    绝不能当成一个 Identity 放行(那会造出一个真机不存在的节点)。
    """
    from cost_eval.opdag.module_resolver import submodule_declared_defaults
    d = submodule_declared_defaults(
        mf_pkg, "MLASelfAttentionSubmodules",
        "parallel_core/training_graph/transformer/multi_latent_attention.py")
    assert d == {}, d
    d2 = submodule_declared_defaults(
        mf_pkg, "TransformerLayerSubmodules", LAYER_REL)
    assert d2 == {k: "Identity" for k in (
        "input_layernorm", "self_attention", "pre_cross_attn_layernorm",
        "cross_attention", "pre_mlp_layernorm", "mlp")}, d2


def _layer_bare(fused_mhc: bool) -> dict:
    """整层递归需要的「裸类 → ResolvedSpec」表:`self.X = <Cls>(...)` 这类**不经 spec 树**的
    直接实例化(惯用法B)必须由调用方提供子 spec,否则维持 fail-loud。"""
    bare = dict(_hc_bare(fused_mhc))
    bare.update({n: ResolvedSpec(cell=n, submodules={}) for n in
                 ("UnfusedCSAIndexerLoss", "Hadamard", "TopKRouter", "GroupedMLP",
                  "SequentialMLP", "MoEAuxLossAutoScaler",
                  # `self.hidden_states_dropout = Dropout(...)`(transformer_layer.py:175)
                  "Dropout", "IdentityOp")})
    # `SharedExpertMLP` 的 submodules 由 `MoELayer.__init__:58-62` 手搭并传入,
    # 走 `extractor._inline_submodules_spec`;此处只需登记类名可递归。
    bare["SharedExpertMLP"] = ResolvedSpec(cell="SharedExpertMLP", submodules={})
    return bare


def extract_mhc_layer(mf_pkg, fused_mhc: bool = True, strict: bool = True):
    top = resolve_layer_spec(mf_pkg, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    return extract_cell(
        mf_pkg, LAYER_REL, "HyperConnectionTransformerLayer", top,
        cell_flags(use_fused_mhc=fused_mhc), present_params={"rotary_pos_emb"},
        recurse=True, subcell_specs=_layer_bare(fused_mhc), cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        kernel_call_allow=MHC_KERNEL_ALLOW + ("npu_lightning_indexer",),
        kernel_saves=MHC_KERNEL_SAVES, injected_binds=INJECTED,
        input_axes=INPUT_AXES, strict=strict,
    )


def test_hyper_connection_layer_extracts_the_whole_mhc_residual_path(mf_pkg):
    """`HyperConnectionTransformerLayer.construct`(:289-336)整层抽通(mHC + attn + MoE)。

    验收要点(逐条对应 `transformer_layer.py` 的行):
      * `self.attn_hc(hidden_states)` @ :309 与 `self.ffn_hc(hidden_states)` @ :327
        —— 两个 mHC 模块都被**递归内联**(不是 opaque);
      * `self.attn_hc.output_cell(...)` @ :323 与 `self.ffn_hc.output_cell(...)` @ :333
        —— 残差流更新(`h_res^T @ x + h_post * sublayer_out`)也在图里;
      * `input_layernorm` @ :311 / `pre_mlp_layernorm` @ :329 两个层级 Norm 有节点
        (评估文档 §3.6:此前它们只能靠抽本层拿到,而本层 FAIL)。
    """
    dag = extract_mhc_layer(mf_pkg)
    assert dag.nodes, "mHC 层抽出空图"
    clean(dag)
    assert any(n.op == "Norm" and n.src == "transformer_layer.py:311" for n in dag.nodes), dump(dag)
    assert any(n.op == "Norm" and n.src == "transformer_layer.py:329" for n in dag.nodes), dump(dag)
    # mHC 模块与 output_cell 的节点都落在 hyper_connection.py 上(内联后 src 切到定义文件)
    hc_srcs = {s for s in srcs(dag) if s.startswith("hyper_connection.py")}
    assert len(hc_srcs) >= 4, sorted(hc_srcs)


def test_mhc_weights_never_enter_ins_or_saves(mf_pkg):
    """W2/W3/W4:mHC 的 `Parameter` 一律走 `param_operands`,不进 `ins`、不是任何 op 的 out。

    mHC 的参数(`hyper_connection.py:220-230`):`alpha_pre` / `alpha_post` / `alpha_res` /
    `bias` / `rms_weight`。它们 fp32、量级小,但**契约是硬的**:权重进了 `ins` 就会被
    `derive_saves` 当激活 save 计(实测 FFNGroupedGEMM「236 MiB」里 88 MiB 就是这个病,
    评估文档 §7.2)。
    """
    dag = extract_hc(mf_pkg, fused=False)
    pnames = {p["param"] for p in dag.param_operands}
    assert {"alpha_pre", "alpha_post", "alpha_res", "bias", "rms_weight"} <= pnames, pnames
    save_names = {s.name for s in derive_saves(dag)}
    ins_names = {i.split(":")[0] for n in dag.nodes for i in n.ins}
    assert not (pnames & save_names), pnames & save_names
    assert not (pnames & outs(dag)), pnames & outs(dag)
    assert not (pnames & ins_names), pnames & ins_names


def test_fused_mhc_kernel_saved_set_must_be_declared_not_guessed(mf_pkg):
    """融合 mHC 内核未声明 saved 集 → `derive_saves` **fail-loud**(不许静默当「无 saved」)。

    `npu_mhc_pre_sinkhorn` 来自 `hyper_parallel.custom_ops.experimental`
    (`hyper_connection.py:20-23`,try/except ImportError),**不在权威快照里** → 它的 bprop
    读不出来。此时唯一诚实的行为是 fail-loud;调用方要出数就必须**显式声明并给理由**。
    """
    dag = extract_cell(
        mf_pkg, HC_REL, "FusedHyperConnectionModule",
        ResolvedSpec(cell="FusedHyperConnectionModule", submodules={}),
        cell_flags(use_fused_mhc=True), recurse=True, subcell_specs=_hc_bare(True),
        cross_file=True, runtime_predicates=RUNTIME_PREDICATES,
        kernel_call_allow=MHC_KERNEL_ALLOW,      # 允许建节点
        # kernel_saves 故意不给
        strict=True,
    )
    assert dag.nodes
    with pytest.raises(ValueError, match="saved"):
        derive_saves(dag)
    # 给了声明就能出数,且声明的定位符/理由挂在节点上(可评审,不是隐形默认)
    dag2 = extract_hc(mf_pkg, fused=True)
    k = next(n for n in dag2.nodes if n.op == "Kernel")
    assert k.attrs["saved_source"], k.attrs
    assert k.attrs["saved_declared_by_caller"] is True, k.attrs
    assert derive_saves(dag2)


def test_fused_and_unfused_mhc_both_extract_clean(mf_pkg):
    """A/B 两条 mHC 路径都抽通且零诊断(`use_fused_mhc` 是 yaml 的一个开关,两支都要能算)。"""
    for fused in (True, False):
        dag = extract_hc(mf_pkg, fused=fused)
        assert dag.nodes, fused
        clean(dag)
        assert not dag.opaque_calls, [c["expr"] for c in dag.opaque_calls]


# ══════════════════════════════════════════════════════════════════════════════════
# 2. pynative MoE(MoELayer / TopKRouter / GroupedMLP / SharedExpertMLP)
# ══════════════════════════════════════════════════════════════════════════════════
MOE_REL = "pynative/transformers/moe/moe_layer.py"
ROUTER_REL = "pynative/transformers/moe/router.py"
EXPERTS_REL = "pynative/transformers/moe/experts.py"


def _moe_bare() -> dict:
    return {n: ResolvedSpec(cell=n, submodules={}) for n in
            ("TopKRouter", "GroupedMLP", "SharedExpertMLP", "SequentialMLP",
             "MoEAuxLossAutoScaler", "Dropout", "IdentityOp")}


def extract_moe(mf_pkg, strict: bool = True):
    """真机走的那个 `MoELayer` —— 经 spec 树解析(`origin_rel` 保证是 pynative 那份)。"""
    top = resolve_layer_spec(mf_pkg, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    mlp = top.submodules["mlp"]
    return extract_cell(
        mf_pkg, MOE_REL, "MoELayer", mlp, cell_flags(), recurse=True,
        subcell_specs=_moe_bare(), cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        input_axes=INPUT_AXES, strict=strict,
    )


def test_moe_layer_spec_resolves_to_the_pynative_class_not_another_tree(mf_pkg):
    """`MoELayer` 必须解到 **pynative** 那份 —— 三棵并行树里同名类的静默错解。

    实测(改动前):`extractor._find_cell_file` 按 `os.walk` 首个命中,把 pynative 的
    `MoELayer` 解成了 `parallel_core/inference/transformer/moe/moe_layer.py` 那份
    —— 于是报「`self.router` 的 build_module 首参不是 submodules.<字段>」(那是**另一棵树**
    的写法)。权威答案在 spec 文件的 import 里:
    `pynative/base_models/gpt/moe_module_specs.py:20`
    `from mindformers.pynative.transformers.moe.moe_layer import MoELayer`。
    """
    top = resolve_layer_spec(mf_pkg, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    mlp = top.submodules["mlp"]
    assert mlp.cell == "MoELayer"
    assert mlp.origin_rel == "pynative/transformers/moe/moe_layer.py", mlp.origin_rel


def test_ambiguous_cell_name_without_origin_is_fail_loud(mf_pkg):
    """没有 `origin_rel` 且调用方位置也定不了唯一时 → **fail-loud 并列出候选**,不掷硬币。"""
    from cost_eval.opdag.extractor import _find_cell_file
    with pytest.raises(ValueError) as ei:
        _find_cell_file(mf_pkg, "MoELayer")          # 不给 prefer_rel
    msg = str(ei.value)
    assert "pynative/transformers/moe/moe_layer.py" in msg
    assert "候选" in msg
    # 给了调用方位置(pynative 树里)→ 唯一确定
    got = _find_cell_file(mf_pkg, "MoELayer", prefer_rel="pynative/transformers/foo.py")
    assert got == "pynative/transformers/moe/moe_layer.py", got


def test_moe_layer_extracts_clean_with_only_two_justified_opaque(mf_pkg):
    """`MoELayer.construct`(`moe_layer.py:113-144`)整段抽通:router + experts + shared。

    允许清单**两条,逐条给字节理由**:
      1. `save_to_aux_losses_tracker(...)` @ `router.py:690` —— 纯宿主副作用(把逐层 aux loss
         **标量**累加进模块级 dict),无被消费的返回值、不产张量。与 walker_core 文档 §6.2 里
         `save_to_indexer_losses_tracker`(csa.py:799)是**同一条判据**。
      2. `self.tokens_per_expert.add_(num_tokens_per_expert)` @ `moe_layer.py:127` ——
         **原地**写进一个 `Parameter(..., requires_grad=False)` 的负载均衡 buffer
         (`moe_layer.py:84-88` 逐字):不在 autograd 图上(⇒ 无 saved 激活)、原地(⇒ 不新分配)、
         无赋值目标(⇒ 无下游消费者)。三条都是源侧可证的,不是"看着像"。
    """
    dag = extract_moe(mf_pkg)
    assert dag.nodes
    clean(dag)
    kinds = sorted(c.get("kind") for c in dag.opaque_calls)
    assert kinds == ["host_side_effect", "inplace_param_buffer_update"], (
        [(c.get("src"), c.get("kind")) for c in dag.opaque_calls])
    # 三族都在图里
    assert any(n.src.startswith("router.py") for n in dag.nodes), dump(dag)
    assert any(n.op == "GroupedMatMul" for n in dag.nodes), dump(dag)      # experts
    assert any(n.src.startswith("mlp.py") for n in dag.nodes), dump(dag)   # shared experts(基类 MLP)


def test_grouped_mlp_experts_are_two_grouped_gemms_plus_gated_activation(mf_pkg):
    """`GroupedMLP.experts_forward`(`experts.py:216-238`):GroupedMatmul → chunk → act → mul
    → GroupedMatmul。**权重不进 ins**(W2/W3):`weight1`/`weight2` 走 `param_operands`。

    `need_dispatch` 由部署形态判定:yaml `expert_parallel: 2` ⇒ `self.weight1` 是带 "ep" 维的
    DTensor ⇒ `not isinstance(...) or "ep" not in ...mesh_dim_names` = **False**
    (`experts.py:181`)⇒ permute/unpermute **不在**本 config 的图里(ExpertParallel 接管派发)。
    """
    dag = extract_cell(
        mf_pkg, EXPERTS_REL, "GroupedMLP", ResolvedSpec(cell="GroupedMLP", submodules={}),
        cell_flags(), recurse=True, subcell_specs=_moe_bare(), cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        input_axes=INPUT_AXES, strict=True,
    )
    clean(dag)
    assert not dag.opaque_calls, [c["expr"] for c in dag.opaque_calls]
    ops = [n.op for n in dag.nodes]
    assert ops.count("GroupedMatMul") == 2, dump(dag)
    assert "Activation" in ops and "Elementwise" in ops, dump(dag)     # swiglu 的 act + mul
    pnames = {p["param"] for p in dag.param_operands}
    assert {"weight1", "weight2"} <= pnames, pnames
    save_names = {s.name for s in derive_saves(dag)}
    ins_names = {i.split(":")[0] for n in dag.nodes for i in n.ins}
    assert not (pnames & save_names) and not (pnames & ins_names) and not (pnames & outs(dag))


def test_topk_router_aux_loss_dispatch_table_is_resolved_not_dropped(mf_pkg):
    """router 的 **python 级方法派发表**必须解开,否则整段 aux-loss 计算(真张量)丢掉。

    源(`router.py:479-490`):`aux_loss_func_map = {"seq_aux_loss": self._apply_seq_aux_loss, ...}`
    → `.get(self.aux_loss_type)` → 调用。`aux_loss_type` 来自 yaml
    (`moe_router_load_balancing_type: seq_aux_loss`)⇒ 静态可解。
    """
    dag = extract_cell(
        mf_pkg, ROUTER_REL, "TopKRouter", ResolvedSpec(cell="TopKRouter", submodules={}),
        cell_flags(), recurse=True, subcell_specs=_moe_bare(), cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        input_axes=INPUT_AXES, strict=True,
    )
    clean(dag)
    # `_apply_seq_aux_loss` 的体被内联:`_reduce_seq_sum_pair` 的 encoded_indices/histc 在图上
    assert any(n.src in ("router.py:535", "router.py:536") for n in dag.nodes), dump(dag)
    # 跨文件模块级函数 `compute_routing_scores_for_aux_loss` @ moe_utils.py:343-367 也被内联
    assert any(n.src.startswith("moe_utils.py") for n in dag.nodes), dump(dag)
    # 唯一 opaque 是那条宿主副作用
    assert [c.get("kind") for c in dag.opaque_calls] == ["host_side_effect"], dag.opaque_calls


def test_router_dispatch_table_with_unknown_config_key_is_fail_loud(mf_pkg):
    """派发表里没有 config 给的键 → fail-loud(源里紧随其后就是 `raise ValueError`,`router.py:486`)。"""
    with pytest.raises(ValueError, match="派发表"):
        extract_cell(
            mf_pkg, ROUTER_REL, "TopKRouter", ResolvedSpec(cell="TopKRouter", submodules={}),
            cell_flags(aux_loss_type="no_such_aux_loss"), recurse=True,
            subcell_specs=_moe_bare(), cross_file=True,
            runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
            input_axes=INPUT_AXES, strict=True,
        )


def test_shared_expert_submodules_come_from_the_parent_init_not_the_spec_tree(mf_pkg):
    """`SharedExpertMLP` 的 `linear_fc1/fc2` 由**父 `__init__` 手搭**并传入,spec 树给不出。

    源(`moe_layer.py:58-62`):`submodules = MLPSubmodules(linear_fc1=Linear, linear_fc2=Linear)`
    → `SharedExpertMLP(config, submodules)`;而 `get_moe_module_spec` 只返回
    `ModuleSpec(module=MoELayer)`,submodules **全空**(`moe_module_specs.py:34-37`)。
    另外 `SharedExpertMLP.construct` 的主体在 `super().construct(...)`(`shared_experts.py:68`)
    —— 不解 `super()` 就抽出 **0 节点**。
    """
    dag = extract_moe(mf_pkg)
    mlp_mm = [n for n in dag.nodes if n.op == "MatMul" and n.src.startswith("mlp.py")]
    assert len(mlp_mm) >= 2, dump(dag)


def test_moe_router_weight_and_expert_weights_never_enter_saves(mf_pkg):
    """W2/W3/W4 在整个 MoE 段上成立:router 的 `weight` 与 experts 的 `weight1/2` 都在
    `param_operands`,不进 `ins`、不进 `saves`、不是任何 op 的 out。

    这正是评估文档 §7.2 那条实测病症(FFNGroupedGEMM「236 MiB」里 88 MiB 是权重被当激活 save)
    在 pynative MoE 侧的对应门。
    """
    dag = extract_moe(mf_pkg)
    pnames = {p["param"] for p in dag.param_operands}
    assert {"weight", "weight1", "weight2"} <= pnames, pnames
    save_names = {s.name for s in derive_saves(dag)}
    ins_names = {i.split(":")[0] for n in dag.nodes for i in n.ins}
    assert not (pnames & save_names), pnames & save_names
    assert not (pnames & ins_names), pnames & ins_names
    assert not (pnames & outs(dag)), pnames & outs(dag)


# ══════════════════════════════════════════════════════════════════════════════════
# 3. pynative embedding(VocabEmbedding / LanguageModelEmbedding)
# ══════════════════════════════════════════════════════════════════════════════════
VOCAB_EMB_REL = "pynative/base_models/common/embeddings/vocab_embedding.py"
LM_EMB_REL = "pynative/base_models/common/embeddings/language_model_embedding.py"

# `Validator.check_type_name('input_ids', input_.dtype, [int32,int64], self.cls_name)`
# @ `vocab_embedding.py:78` —— **纯 host 侧类型断言**:无返回值被消费、不产张量、不改任何
# 张量的值或 dtype。与 `save_to_indexer_losses_tracker` 同一条判据(walker_core 文档 §6.2)。
EMB_HOST_ALLOW = HOST_ALLOW + ("Validator.check_type_name",)


def _emb_bare() -> dict:
    return {n: ResolvedSpec(cell=n, submodules={}) for n in
            ("VocabEmbedding", "Dropout", "IdentityOp")}


def extract_emb(mf_pkg, cls: str, strict: bool = True):
    rel = VOCAB_EMB_REL if cls == "VocabEmbedding" else LM_EMB_REL
    return extract_cell(
        mf_pkg, rel, cls, ResolvedSpec(cell=cls, submodules={}), cell_flags(),
        recurse=True, subcell_specs=_emb_bare(), cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=EMB_HOST_ALLOW,
        input_axes=INPUT_AXES, strict=strict,
    )


def test_vocab_embedding_gather_saves_its_index_not_its_weight(mf_pkg):
    """`VocabEmbedding.construct`(`vocab_embedding.py:62-88`):reshape → tile → gather → reshape。

    两条硬事实:
      * **权重不进 saves**(W2/W3/W4):`weight` 走 `param_operands`;
      * **index 进 saves**:`mint.gather` 的反向是把 dy scatter_add 回零张量的 index 位置
        → 必须存 index。这条以前会**丢**:`self.embedding(weight, 0, input_)`(`:85`)里
        `weight` 被路由出 `ins` 后 `ins` 只剩 `[input_]`,而 `PIN["IndexSelect"]={"inputs":[1]}`
        的 `[1]` 越界 → saved 集变成空。现由 `attrs["ins_slots"]`(张量操作数位序)修正。
    """
    dag = extract_emb(mf_pkg, "VocabEmbedding")
    clean(dag)
    assert [n.op for n in dag.nodes] == ["View", "View", "IndexSelect", "View"], dump(dag)
    gather = next(n for n in dag.nodes if n.op == "IndexSelect")
    assert gather.src == "vocab_embedding.py:85", dump(dag)
    assert gather.attrs["param_operands"] == ["weight"], gather.attrs
    assert gather.attrs["ins_slots"] == [1], gather.attrs      # index 是第 2 个**张量**操作数
    names = {s.name for s in derive_saves(dag)}
    assert "input_" in names, dump(dag)
    assert "weight" not in names, dump(dag)
    # 唯一 opaque:那条 host 侧类型断言
    assert [c.get("kind") for c in dag.opaque_calls] == ["host_side_effect"], dag.opaque_calls


def test_language_model_embedding_walks_the_rope_config_branches(mf_pkg):
    """`LanguageModelEmbedding.construct`(`language_model_embedding.py:103-142`)。

    DSv4 用 rope(`position_embedding_type='rope'`,`gpt_model.py:196`)⇒
    `add_position_embedding = (position_embedding_type == 'learned_absolute')` = **False**
    (`:73`)⇒ 位置嵌入那一支不在图里;`num_tokentypes=0` ⇒ `tokentype_embeddings = None`
    (`:91`)⇒ `:129-131` 的 raise 守卫剪掉;`hidden_dropout=0.0` ⇒ `:137` 的 dropout 不在图里。
    最后是 `transpose([b,s,h] → [s,b,h])` + `cast(compute_dtype)`。
    """
    dag = extract_emb(mf_pkg, "LanguageModelEmbedding")
    clean(dag)
    ops = [n.op for n in dag.nodes]
    assert ops.count("IndexSelect") == 1, dump(dag)           # 只有 word_embeddings 一次 gather
    assert any(n.src == "language_model_embedding.py:134" for n in dag.nodes), dump(dag)  # transpose
    assert any(n.src == "language_model_embedding.py:140" for n in dag.nodes), dump(dag)  # cast
    # 位置嵌入支被剪掉(否则会多一次 gather + 一次 add)
    assert not any(n.src == "language_model_embedding.py:117" for n in dag.nodes), dump(dag)


# ══════════════════════════════════════════════════════════════════════════════════
# 4. lm_head 的 vocab 投影(walked,不再是合成)
# ══════════════════════════════════════════════════════════════════════════════════
LINEAR_REL = "pynative/layers/linear.py"


def head_flags() -> dict:
    """`gpt_model.py:252-258` 的 `self.output_layer = Linear(...)` 构造点逐字。"""
    return cell_flags(
        skip_weight_param_allocation=False, has_bias=False, skip_add_bias=False,
        input_size=4096, output_size=129280,
    )


def test_lm_head_vocab_projection_is_walked_from_source(mf_pkg):
    """lm_head 的 vocab 投影现在是**走查出来的**,不是 `gpt_segments.head_segment_dag()` 合成的。

    源:`self.output_layer = Linear(input_size=hidden_size, output_size=vocab_size, ...)`
    (`pynative/base_models/gpt/gpt_model.py:252-258`),被 `:365` 调用;计算体在
    `pynative/layers/linear.py:105-146`。本 config(`params_dtype == compute_dtype == bf16`,
    `has_bias=False`)下三个 `if <dtype> != <dtype>:` 的 cast 门(`:126/:128/:143`)全为 False
    —— 这三条判定此前判不出来,整个 Linear 一个节点都抽不出。
    """
    dag = extract_cell(
        mf_pkg, LINEAR_REL, "Linear", ResolvedSpec(cell="Linear", submodules={}),
        head_flags(), recurse=True, subcell_specs={}, cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=EMB_HOST_ALLOW,
        input_axes={"input_": ("seq_length", "micro_batch_size", "hidden_size")},
        strict=True,
    )
    clean(dag)
    assert not dag.opaque_calls, [c["expr"] for c in dag.opaque_calls]
    assert [n.op for n in dag.nodes] == ["View", "MatMul"], dump(dag)
    mm = dag.nodes[-1]
    assert mm.src == "linear.py:135", dump(dag)
    assert mm.attrs["param_operands"] == ["weight"], mm.attrs
    names = {s.name for s in derive_saves(dag)}
    assert names == {"input_"}, dump(dag)      # 只存激活输入;权重与权重派生量都不存


def test_weight_derived_tensors_never_enter_saves_even_after_view_and_cast(mf_pkg):
    """**W2/W3/W4 的字节那一半**:权重经 cast/reshape 派生出的张量也不是激活。

    实测病症(评估文档 §7.2):`parallel_core/training_graph/.../moe/ffn.py` 的
    `w1 = cast(self.weight1, ...)`(`:146`)→ `w1 = reshape(w1, ...)`(`:157`)→ 进
    `GroupedMatmul` 的 `ins`,被 `PIN["GroupedMatMul"]={"inputs":"all"}` 当激活 save 计
    —— 「236 MiB」里 `w1`(58.7MB)+`w2`(29.4MB)= **88 MiB**。
    修法保留 `ins`(`shape_infer` 要靠权重末轴推 matmul 输出维),只在
    `attrs["weight_ins_idx"]` 标出来由 `derive_saves` 排除。
    """
    dag = extract_cell(
        mf_pkg, "parallel_core/training_graph/transformer/moe/ffn.py", "FFNGroupedGEMM",
        ResolvedSpec(cell="FFNGroupedGEMM", submodules={}),
        {"gated_linear_unit": True, "activation_type": "silu", "add_bias_linear": False,
         "compute_dtype": "bf16", "moe_token_dispatcher_type": "alltoall"})
    names = {s.name for s in derive_saves(dag)}
    assert "w1" not in names and "w2" not in names, sorted(names)
    fc1, fc2 = [n for n in dag.nodes if n.op == "GroupedMatMul"]
    # 仍在 ins 里(可见、可供 shape 推断),只是被标为权重派生
    assert [i.split(":")[0] for i in fc1.ins][1] == "w1"
    assert fc1.attrs["weight_ins_idx"] == [1] and fc2.attrs["weight_ins_idx"] == [1]
    # 传播链:cast(:146) → reshape(:157) → GroupedMatmul(:178),中间任一跳断掉都会漏回病症
    resh = next(n for n in dag.nodes if n.src == "ffn.py:157")
    assert resh.attrs["weight_ins_idx"] == [0], resh.attrs


# ══════════════════════════════════════════════════════════════════════════════════
# 5. pynative loss(CrossEntropyLoss / ChunkCrossEntropyLoss)
# ══════════════════════════════════════════════════════════════════════════════════
LOSS_REL = "pynative/loss/loss.py"


def _loss_bare() -> dict:
    return {n: ResolvedSpec(cell=n, submodules={}) for n in
            ("_LogSoftmaxModule", "_NLLLossModule")}


def extract_loss(mf_pkg, cls: str = "CrossEntropyLoss", strict: bool = True, **over):
    return extract_cell(
        mf_pkg, LOSS_REL, cls, ResolvedSpec(cell=cls, submodules={}),
        cell_flags(**over),
        # 真机 `compute_language_model_loss(labels, logits, loss_mask)`(`gpt_model.py:509`)
        # 三个实参都传(yaml dataset 的 column_names 含 `loss_mask`)
        present_params={"input_mask"},
        recurse=True, subcell_specs=_loss_bare(), cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        input_axes=INPUT_AXES, strict=strict,
    )


def test_pynative_cross_entropy_extracts_clean(mf_pkg):
    """`CrossEntropyLoss.construct`(`loss.py:326-350`),`_tp_group is None`(tp=1)支。

    源事实:`self._tp_group = None`(`:316`),只有 `enable_vocab_parallel(...)`(`:320`)被调
    才非 None —— yaml `tensor_parallel: 1` ⇒ 不调 ⇒ 走 `log_softmax` + `nll_loss` 那支
    (即手写侧的 `loss_type="logsoftmax_nll"`)。
    """
    dag = extract_loss(mf_pkg)
    assert dag.nodes
    clean(dag)
    assert not dag.opaque_calls, [c["expr"] for c in dag.opaque_calls]
    fns = [n for n in dag.nodes if n.op == "FusedFunction"]
    assert [n.attrs["function"] for n in fns] == ["_LogSoftmax", "_NLLLoss"], dump(dag)


def test_bare_ctx_assignment_is_a_saved_tensor_too(mf_pkg):
    """`ctx.<attr> = <forward 形参>` 与 `ctx.save_for_backward(...)` 的**内存事实相同**。

    `_LogSoftmax.forward` 里**只有** `ctx.logits = logits`(`loss.py:136`),没有
    `save_for_backward`;`backward` 里 `logits = ctx.logits`(`:151`)。这个张量同样被留到反向
    —— 而它是全模型最大的一块(`S·B·vocab` 的 logits)。漏掉它就是漏掉最大的那块。
    节点上分开记来源:`save_for_backward`(源逐字名单)与 `bare_ctx_retained`(裸 ctx 保留)。
    """
    dag = extract_loss(mf_pkg)
    ls = next(n for n in dag.nodes if n.attrs.get("function") == "_LogSoftmax")
    assert ls.attrs["save_for_backward"] == [], ls.attrs      # 源里确实没有 save_for_backward
    assert ls.attrs["bare_ctx_retained"] == ["logits"], ls.attrs
    names = {s.name for s in derive_saves(dag)}
    assert {"logits", "log_softmax", "label"} <= names, sorted(names)


def test_chunked_cross_entropy_is_one_fused_function_by_source(mf_pkg):
    """`ChunkCrossEntropyLoss.construct`(`loss.py:361-373`):3D logits ⇒ 一个
    `_ChunkCrossEntropyLoss.apply`(`:372`)。

    `logits.ndim == 3`(`:363`)由 `input_axes` 给的**轴数**判定(与 `infer_shapes` 同一套种子
    契约);`self._tp_group is None`(tp=1)⇒ 不走 `_ChunkVocabParallelCrossEntropy`。
    这条正是手写侧 `head.py` 的 `loss_type="chunked"` 对应的源侧实体。
    """
    dag = extract_loss(mf_pkg, "ChunkCrossEntropyLoss", chunk_loss_num=4)
    clean(dag)
    assert [n.op for n in dag.nodes] == ["FusedFunction"], dump(dag)
    assert dag.nodes[0].attrs["function"] == "_ChunkCrossEntropyLoss", dag.nodes[0].attrs
    # 该融合反向**只保存原始 logits/labels/mask**(`:408-410` 逐字),分块重算 softmax
    assert set(dag.nodes[0].attrs["bare_ctx_retained"]) == {"logits", "labels", "input_mask"}


def test_ndim_without_a_seed_is_fail_loud_not_guessed(mf_pkg):
    """没给 `input_axes` 种子时 `logits.ndim == 3` **判不出** → fail-loud(不猜维数)。"""
    with pytest.raises(ValueError, match="ndim"):
        extract_cell(
            mf_pkg, LOSS_REL, "ChunkCrossEntropyLoss",
            ResolvedSpec(cell="ChunkCrossEntropyLoss", submodules={}),
            cell_flags(chunk_loss_num=4), present_params={"input_mask"},
            recurse=True, subcell_specs=_loss_bare(), cross_file=True,
            runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
            input_axes={}, strict=True,
        )


# ══════════════════════════════════════════════════════════════════════════════════
# 6. MTP(MultiTokenPredictionLayer)
# ══════════════════════════════════════════════════════════════════════════════════
MTP_REL = "pynative/transformers/multi_token_prediction.py"


def mtp_spec(mf_pkg):
    """MTP 层 spec:`get_gpt_mtp_block_spec`(`gpt_layer_specs.py:227-256`)拿 decoder block 的
    **最后一层** spec 与 `hc_head` 去调 `get_mtp_layer_spec`(`multi_token_prediction.py:223`)。
    `enable_hc_head` 缺省 `None` ⇒ 跟随 `enable_hyper_connections` = True
    (`transformer_config.py:2158-2159`)⇒ `hc_head = HyperConnectionHead`。
    """
    from cost_eval.opdag.module_resolver import MTP_SPEC_FILES, resolve_spec_call
    top = resolve_layer_spec(mf_pkg, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    return resolve_spec_call(
        mf_pkg, "get_mtp_layer_spec", SPEC_FLAGS, spec_files=MTP_SPEC_FILES,
        keyword={"transformer_layer_spec": top, "normalization": "RMSNorm",
                 "fused_norm": True, "hc_head": "HyperConnectionHead"})


def extract_mtp(mf_pkg, strict: bool = True):
    bare = _layer_bare(True)
    bare.update({n: ResolvedSpec(cell=n, submodules={}) for n in
                 ("LanguageModelEmbedding", "VocabEmbedding", "HyperConnectionHead")})
    return extract_cell(
        mf_pkg, MTP_REL, "MultiTokenPredictionLayer", mtp_spec(mf_pkg), cell_flags(),
        # 真机 `MultiTokenPredictionBlock` 逐个传入(`gpt_model.py:337-345`)
        present_params={"rotary_pos_emb", "position_ids", "attention_mask", "embedding",
                        "actual_seq_len"},
        recurse=True, subcell_specs=bare, cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=EMB_HOST_ALLOW,
        kernel_call_allow=MHC_KERNEL_ALLOW + ("npu_lightning_indexer",),
        kernel_saves=MHC_KERNEL_SAVES, injected_binds=INJECTED, input_axes=INPUT_AXES,
        # `embedding` 形参持有的是 `gpt_model.py:183` 建的 LanguageModelEmbedding(`:340` 传入)
        param_cells={"embedding": "LanguageModelEmbedding"},
        strict=strict,
    )


def test_mtp_layer_submodules_come_from_self_submodules(mf_pkg):
    """`build_module(self.submodules.enorm, ...)` 这种首参形态必须认。

    `MultiTokenPredictionLayer.__init__` 先 `self.submodules = submodules`(`:286`),再全部从
    `self.submodules.<field>` 读(`:293/:300/:307/:316/:324`)。此前只认 `submodules.<field>`
    → 「首参不是 submodules.<字段>」fail-loud → **MTP 零覆盖**(评估文档 §3.5)。
    """
    spec = mtp_spec(mf_pkg)
    assert set(spec.submodules) == {"enorm", "hnorm", "eh_proj", "transformer_layer",
                                    "layer_norm", "hc_head"}, sorted(spec.submodules)
    assert spec.submodules["enorm"] == "Norm" and spec.submodules["eh_proj"] == "Linear"
    dag = extract_mtp(mf_pkg)
    assert any(n.op == "Norm" and n.src == "multi_token_prediction.py:380" for n in dag.nodes), \
        dump(dag)      # enorm
    assert any(n.op == "Norm" and n.src == "multi_token_prediction.py:381" for n in dag.nodes), \
        dump(dag)      # hnorm
    assert any(n.op == "MatMul" and n.src == "multi_token_prediction.py:385" for n in dag.nodes), \
        dump(dag)      # eh_proj


def test_mtp_layer_extracts_clean_with_justified_opaque(mf_pkg):
    """整个 MTP 层抽通:roll → embedding → enorm/hnorm → cat → eh_proj → hc expand →
    整个 decoder 层 → hc_head → final_layernorm。允许清单三条,逐条给理由:
      1. `Validator.check_type_name(...)` @ `vocab_embedding.py:78` —— host 侧类型断言;
      2. `save_to_aux_losses_tracker(...)` @ `router.py:690` —— host 侧标量记账;
      3. `self.tokens_per_expert.add_(...)` @ `moe_layer.py:127` —— 原地写非梯度 buffer。
    """
    dag = extract_mtp(mf_pkg)
    assert dag.nodes
    clean(dag)
    assert sorted(c.get("kind") for c in dag.opaque_calls) == [
        "host_side_effect", "host_side_effect", "inplace_param_buffer_update"], (
        [(c.get("src"), c.get("kind")) for c in dag.opaque_calls])
    # MTP 真的**再跑一遍 embedding**(输入是 roll 过的 input_ids)—— 声明了 param_cells 才可见
    assert any(n.src.startswith("vocab_embedding.py") for n in dag.nodes), dump(dag)
    # mHC 流的 expand(:389)与 hc_head 折叠(:401)都在图里
    assert any(n.src == "transformer_block.py:26" for n in dag.nodes), dump(dag)


def test_mtp_embedding_param_cell_undeclared_stays_visible_not_silent(mf_pkg):
    """不声明 `param_cells` 时,`embedding(...)` **落 opaque + unregistered**(可见),而不是静默。"""
    bare = _layer_bare(True)
    bare.update({n: ResolvedSpec(cell=n, submodules={}) for n in
                 ("LanguageModelEmbedding", "VocabEmbedding", "HyperConnectionHead")})
    dag = extract_cell(
        mf_pkg, MTP_REL, "MultiTokenPredictionLayer", mtp_spec(mf_pkg), cell_flags(),
        present_params={"rotary_pos_emb", "position_ids", "attention_mask", "embedding",
                        "actual_seq_len"},
        recurse=True, subcell_specs=bare, cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=EMB_HOST_ALLOW,
        kernel_call_allow=MHC_KERNEL_ALLOW + ("npu_lightning_indexer",),
        kernel_saves=MHC_KERNEL_SAVES, injected_binds=INJECTED, input_axes=INPUT_AXES,
        strict=False,       # 故意不给 param_cells
    )
    assert any(c.get("expr", "").startswith("embedding(") for c in dag.opaque_calls)
    assert any(d.get("target") == "decoder_input__i0" or "decoder_input" in str(d)
               for d in dag.diagnostics["unregistered_targets"]), \
        dag.diagnostics["unregistered_targets"]
    with pytest.raises(ExtractionDroppedError):
        from cost_eval.opdag.construct_walker import assert_extraction_clean
        assert_extraction_clean(dag)


# ══════════════════════════════════════════════════════════════════════════════════
# 7. 发射点的**轴/形状元信息**(G4)与多输出 dtype(T3)—— 缺它们时下游是**错**,不是"未知"
# ══════════════════════════════════════════════════════════════════════════════════
_AXIS_KEYS = ("reduce_dim", "keepdim", "chunk_dim", "chunks", "concat_axis", "permute_dims",
              "topk_k", "topk_dim", "const_shape", "const_shape_src", "slice_bounds",
              "stack_axis", "squeeze_axis", "roll_shifts", "broadcast_shape")

_CSA_REL = "pynative/transformers/experimental_attention_variant"


def _dsv4_bare() -> dict:
    return {n: ResolvedSpec(cell=n, submodules={}) for n in
            ("UnfusedCSAIndexerLoss", "Hadamard")}


def extract_dsv4(mf_pkg, cls: str, fname: str, fused: bool = True, strict: bool = True):
    top = resolve_layer_spec(mf_pkg, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    csa = top.submodules["self_attention"].submodules["core_attention"]
    spec = {"Compressor": csa.submodules["compressor"],
            "CSAIndexer": csa.submodules["indexer"],
            "CompressedSparseAttention": csa,
            "DSv4HybridSelfAttention": top.submodules["self_attention"]}[cls]
    return extract_cell(
        mf_pkg, f"{_CSA_REL}/{fname}", cls, spec,
        cell_flags(apply_dsa_kernel_fusion=fused), present_params={"rotary_pos_emb"},
        recurse=True, subcell_specs=_dsv4_bare(), cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        kernel_call_allow=("npu_lightning_indexer",), input_axes=INPUT_AXES,
        injected_binds=INJECTED, strict=strict,
    )


def test_reduce_chunk_concat_axes_are_recorded_at_emit_time(mf_pkg):
    """三个**曾经静默算错**的算子现在带轴(G4;并行 agent 实测的偏差量写在断言旁)。

    没有轴时下游字节解析对它们退化成"直通",于是**不是"未知"而是"错"**:
      * `(kv.astype(fp32) * weights).sum(dim=1)`(`compressor.py:216`)→ 直通 = **8× 过读**;
      * `self.chunk(x, 2, -1)`(`compressor.py:169`)→ **n× 过读**;
      * `cat((kv_nope, kv_pe), -1)`(`compressor.py:243`)→ 解成 `2n·b·(d−64)` 而非 `n·b·d`。
    纪律:抠不出的**一律不写**该键(消费方按缺键落 `unresolved`),绝不填一个默认轴。
    """
    dag = extract_dsv4(mf_pkg, "Compressor", "compressor.py")
    clean(dag)
    by_src = {}
    for n in dag.nodes:
        by_src.setdefault(n.src, []).append(n)

    # ① `.sum(dim=1)` —— 张量方法形态(实参里没有 receiver,位序要补齐才对得上)
    red = [n for n in by_src["compressor.py:216"] if "reduce_dim" in n.attrs]
    assert red and red[0].attrs["reduce_dim"] == 1, [n.attrs for n in by_src["compressor.py:216"]]
    # ② chunk(chunks=2, dim=-1)
    ch = [n for n in by_src["compressor.py:169"] if "chunks" in n.attrs]
    assert ch and (ch[0].attrs["chunks"], ch[0].attrs["chunk_dim"]) == (2, -1), \
        [n.attrs for n in by_src["compressor.py:169"]]
    # ③ cat 的轴
    cat = [n for n in by_src["compressor.py:243"] if "concat_axis" in n.attrs]
    assert cat and cat[0].attrs["concat_axis"] == -1, [n.attrs for n in by_src["compressor.py:243"]]
    # ④ 切片边界(`freqs[:total:ratio][:n]`,`compressor.py:233`)
    sl = [n for n in by_src["compressor.py:233"] if "slice_bounds" in n.attrs]
    assert len(sl) == 2, [n.attrs for n in by_src["compressor.py:233"]]
    assert sl[0].attrs["slice_bounds"][0][2] == "self.compress_ratio", sl[0].attrs


def test_axis_metadata_is_absent_rather_than_guessed_when_unreadable(mf_pkg):
    """轴抠不出时 **不写该键**(缺键 = 未知),而不是填一个默认轴。

    合成源:`dim` 由一个运行期变量给 —— 填 0 会让下游按错轴算(比"未知"危险)。
    """
    from cost_eval.opdag.construct_walker import walk_construct
    from cost_eval.opdag.init_binder import Binding
    src = '''
class C:
    def construct(self, x, k):
        a = self.sum(x, dim=k)
        b = self.sum(x, dim=2)
        return a, b
'''
    dag = walk_construct(
        src, "C", {"sum": Binding("Elementwise", {"linear": True, "reduce": True})},
        "c.py", config_flags={})
    a, b = dag.nodes
    assert "reduce_dim" not in a.attrs, a.attrs      # 运行期变量 → 不写
    assert b.attrs["reduce_dim"] == 2, b.attrs       # 字面量 → 写


def test_multi_output_dtypes_are_registered_per_output(mf_pkg):
    """多输出算子的**每个**输出按自己的 dtype 进 SSA(T3)。

    `topk_scores, topk_indices = self.topk(index_scores, k=..., dim=-1)`(`indexer.py:262`)
    的第 2 个输出是 int32 索引。此前所有目标都按同一个 `out_dtype`(bf16)进 SSA,而
    `attrs["outs"]` 里写的是 int32 —— 于是下一行 `topk_indices = self.cast(topk_indices, int32)`
    (`:263`)的 `ins` 带着**错的 dtype**;`derive_saves` 按名去重只留一个,留下哪个取决于
    谁先被 pin = **静默取错分支**(与 op 类型拼错同一类)。
    """
    dag = extract_dsv4(mf_pkg, "CSAIndexer", "indexer.py", fused=False)
    clean(dag)
    tk = next(n for n in dag.nodes if n.op == "TopK")
    assert tk.src == "indexer.py:262", tk.src
    assert tk.attrs["outs"][1].endswith(":int32"), tk.attrs["outs"]
    # `k=effective_topk` 是运行期 `min(self.index_topk, int(k.shape[1]))`(`indexer.py:212`)
    # —— 抠不出整数 ⇒ **不写** `topk_k`(缺键 = 未知,而不是编一个 k)
    assert "topk_k" not in tk.attrs, tk.attrs
    # 下一行 cast 的 ins 必须已经是 int32(不是 bf16)
    cst = next(n for n in dag.nodes if n.src == "indexer.py:263")
    assert cst.ins == ["topk_indices:?:int32"], cst.ins
    # saved 的 topk_indices 是 int32(TopK 的第 2 个输出),不是 bf16
    tsave = next(s for s in derive_saves(dag) if s.name == "topk_indices")
    assert tsave.dtype == "int32", tsave


def test_all_four_dsv4_cells_still_clean_after_axis_capture(mf_pkg):
    """四个 dsv4 Cell 的 census 锁(A/B 两支):节点数全不变,零诊断全不变。

    census 基线取自 `docs/opdag_walker_core_2026-07-25.md` §6.1 的实测表。
    **唯一一处变化**(2026-07-25,逐条给理由,不是"顺手放宽"):
      `CompressedSparseAttention` / `DSv4HybridSelfAttention` 的 fused 支边数 104→103 / 146→145。
      去掉的那一条是 `[84, 88]`:`FusedSparseFlashMlaWithIndexerLoss.apply(..., attn_sink, ...)`
      (`csa.py:689-697`)的 `attn_sink` 操作数。源侧 `self.attn_sink` 是一个 `Parameter`
      (`csa.py` 的 `__init__`),`:683-687` 只是 `to_local()` + `cast(fp32)` —— 全程没有任何
      激活参与。此前 `:685` 的**自赋值** `attn_sink = attn_sink.to_local()` 会把权重身份
      `_forget` 掉(见 `_bind_name_rhs` 的自赋值分支注释),于是 `:687` 的 cast 产出一个
      **看起来是激活**的张量并进了融合算子的 `ins` → 按 W2/W3/W4 它本不该在那里。
      现在它走 `param_operands`(仍可见),`ins` 与 saves 都不含它 —— 字节影响
      `[n_heads] fp32` = 64×4 = **256 B**,方向是**修正**。
    """
    census = {
        (True,  "Compressor"): (29, 31),  (True,  "CSAIndexer"): (7, 6),
        (True,  "CompressedSparseAttention"): (90, 103),
        (True,  "DSv4HybridSelfAttention"): (123, 145),
        (False, "Compressor"): (29, 31),  (False, "CSAIndexer"): (14, 13),
        (False, "CompressedSparseAttention"): (207, 240),
        (False, "DSv4HybridSelfAttention"): (240, 281),
    }
    files = {"Compressor": "compressor.py", "CSAIndexer": "indexer.py",
             "CompressedSparseAttention": "csa.py",
             "DSv4HybridSelfAttention": "deepseek_v4_hybrid_attention.py"}
    for (fused, cls), (nn, ne) in census.items():
        dag = extract_dsv4(mf_pkg, cls, files[cls], fused=fused)
        clean(dag)
        assert (len(dag.nodes), len(dag.edges)) == (nn, ne), (fused, cls, len(dag.nodes),
                                                              len(dag.edges))
