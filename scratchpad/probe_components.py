# -*- coding: utf-8 -*-
"""mHC / MoE / embedding / loss / MTP 组件抽取探针(本轮验收用)。

权威快照:E:\\97-codes\\torch_parallel\\mf-src-167\\mindformers
用法:python scratchpad/probe_components.py [mhc|moe|emb|loss|mtp|all] [--dump] [--strict]
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MF = os.environ.get("MF_PKG", r"E:\97-codes\torch_parallel\mf-src-167\mindformers")

from cost_eval.opdag.module_resolver import (
    resolve_layer_spec, PYNATIVE_SPEC_FILES, ResolvedSpec)
from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.construct_walker import (
    diagnostics_summary, DIAG_KINDS, _format_diagnostics)
from cost_eval.opdag.bprop_rules import derive_saves
from cost_eval.opdag.init_binder import Binding
from cost_eval.opdag.fn_saves import mhc_kernel_saves

SPEC_FLAGS = {   # 取自 dsv4h_fused_pp4_recomp.yaml
    "num_experts": 8, "moe_grouped_gemm": True, "qk_layernorm": True,
    "multi_latent_attention": True, "enable_hyper_connections": True,
    "fused_norm": True, "normalization": "RMSNorm", "is_dsv4_hybrid": True,
}

RUNTIME_PREDICATES = {
    "hasattr:to_local": True,
    "hasattr:detach": True,
    "isinstance:DTensor": True,
    # yaml parallelism 段:pp4/dp2/ep2/tp1/cp1 → device mesh 的维名单(部署形态事实)
    "mesh_dim_names": ("dp", "ep"),
    # 前向图建模的是**首次前向**(非重算中的那次)——activation_checkpoint.is_in_recompute()
    "call:is_in_recompute": False,
    # moe_utils.py:261-263 的模块级全局 `_AUX_LOSS_GROUP_SIZE`(tp×cp 组世界大小);
    # yaml tensor_parallel=1 / context_parallel=1 → 1
    "call:get_moe_aux_loss_group_size": 1,
}
HOST_ALLOW = ("save_to_indexer_losses_tracker", "get_indexer_loss_tracker",
              "save_to_aux_losses_tracker", "get_moe_layer_wise_logging_tracker",
              "Validator.check_type_name")
KERNEL_ALLOW = ("npu_lightning_indexer", "npu_mhc_pre_sinkhorn", "npu_mhc_post")
# 融合 mHC 内核的 saved 集 —— **逐字读自 hyper_parallel 源**(2026-07-25 补入快照)。
# 此前只能给「张量实参全存」的保守上界;实测那个上界在 `npu_mhc_post` 上恰等于源真值,
# 但在 `npu_mhc_pre_sinkhorn` 上是**欠读**(漏掉 5 个被保存的自身输出)。
HP_ROOT = os.environ.get(
    "HYPER_PARALLEL_ROOT",
    os.path.join(os.path.dirname(MF), "hyper_parallel"))
try:
    KERNEL_SAVES = mhc_kernel_saves(HP_ROOT)
except ValueError as _e:                    # 快照里没有 hyper_parallel → 明确报出来,不退回猜
    print(f"[warn] 读不到 hyper_parallel 的 mHC saved 集:{_e}")
    KERNEL_SAVES = {}


def cell_flags(fused_mhc: bool = True, ratio: int = 4, fused_dsa: bool = True) -> dict:
    return {
        "compute_dtype": "bf16", "layernorm_compute_dtype": "fp32", "params_dtype": "bf16",
        "v_head_dim": 512, "qk_pos_emb_head_dim": 64, "q_lora_rank": 1024,
        "num_attention_heads": 64, "hidden_size": 4096, "o_groups": 8, "o_lora_rank": 1024,
        "add_bias_linear": False, "input_layout": "BSND", "num_layers": 8, "mtp_num_layers": 0,
        "index_topk": 512, "index_n_heads": 64, "index_head_dim": 128,
        "csa_window_size": 128, "csa_dense_mode": False,
        "compress_ratio": ratio, "enable_compress": True, "enable_indexer": ratio == 4,
        "is_tnd": False, "window_size": 128, "training": True,
        "apply_dsa_kernel_fusion": fused_dsa,
        "rotate": True, "overlap": ratio == 4, "coff": 1 + int(ratio == 4),
        "sparse_loss": True, "use_butterfly": False,
        "seq_length": 4096, "micro_batch_size": 1,
        "dsa_indexer_loss_coeff": 0.001,
        # ── mHC(dsv4h_fused_pp4_recomp.yaml:enable_hyper_connections/hc_mult/use_fused_mhc)
        "num_residual_streams": 4, "use_fused_mhc": fused_mhc,
        "use_dropout": False,
        "mhc_sinkhorn_iterations": 20, "mhc_layernorm_epsilon": 1e-6,
        "iterations": 20,   # SinkhornKnopp.__init__:45 self.iterations = iterations
        # `self.dtype = config.compute_dtype`(hyper_connection.py:84/141/205)——
        # `self.cast(x, self.dtype)`(:406/:124/:362)的目标 dtype
        "dtype": "bf16",
        "apply_residual_connection_post_layernorm": False,
        "hidden_dropout": 0.0,
        # ── MoE(yaml:n_routed_experts=8/num_experts_per_tok=2/n_shared_experts=1/
        #    norm_topk_prob=true/n_group=0/moe_aux_loss_coeff=0.001/seq_aux_loss)
        "moe_router_topk": 2, "moe_grouped_gemm": True, "num_experts": 8,
        "moe_shared_expert_intermediate_size": 2048, "moe_ffn_hidden_size": 2048,
        "gated_linear_unit": True, "activation_type": "silu",
        "moe_router_dtype": "fp32", "num_moe_experts": 8,
        # TopKRouter.__init__ 派生量(显式注入,不静态求 __init__)
        "score_func": "softmax", "is_hash_layer": False, "num_expert_groups": 0,
        "route_norm": True, "route_scale": 1.0, "_debug_force_load_balance": False,
        "moe_aux_loss_coeff": 0.001, "aux_loss_type": "seq_aux_loss",
        "top_k": 2, "calculate_per_token_loss": False,
        # TopKRouter 的 SP/CP 通信组(yaml tensor_parallel=1 / context_parallel=1)
        "_tp_size": 1, "_cp_size": 1, "_tp_group": None, "_cp_groups": (),
        "use_shared_expert_gating": False, "moe_permute_fusion": False,
        "use_shared_expert_gate": False,   # SharedExpertMLP.__init__:57 派生量
        "ffn_hidden_size": 2048, "hidden_act": "silu", "activation_func_clamp_value": None,
        # MLP.__init__:93 派生量:activation_type=='fusedswiglu' and clamp_value is not None
        "use_clamped_swiglu": False, "activation_type_mlp": "silu",
        # embedding(position_embedding_type='rope' → 无 learned_absolute 位置嵌入)
        "add_position_embedding": False, "num_tokentypes": 0,
        "embedding_dropout_prob": 0.0, "vocab_size": 129280, "embedding_dim": 4096,
        # loss(yaml tensor_parallel=1 → enable_vocab_parallel 未调用 → _tp_group is None;
        # 无 chunk_loss_num → CrossEntropyLoss 而非 ChunkCrossEntropyLoss)
        "chunk_loss_num": 1, "compensate_loss_sense_tp": True,
        # MTP(enable_hc_head 缺省 None → 跟随 enable_hyper_connections = True)
        "hc": True, "hc_num_streams": 4, "hc_hidden_size": 4096, "enable_hc_head": True,
        # LanguageModelEmbedding.__init__:91 —— num_tokentypes=0 ⇒ self.tokentype_embeddings = None
        "tokentype_embeddings": None, "position_embeddings": None,
        # MoELayer.__init__ 派生量
        "shared_expert_num": 1, "enable_expert_bias": False,
        "score_before_experts": False, "moe_apply_probs_on_input": False,
    }


INJECTED = {"rotary_pos_emb": Binding("Constant", {"rope_freqs": True})}

REL = "pynative/transformers/experimental_attention_variant"
BARE = {n: ResolvedSpec(cell=n, submodules={}) for n in
        ("UnfusedCSAIndexerLoss", "Hadamard")}

INPUT_AXES = {
    "x":      ("seq_length", "micro_batch_size", "hidden_size"),
    "qr":     ("seq_length", "micro_batch_size", "q_lora_rank"),
    "query":  ("seq_length", "micro_batch_size", "num_attention_heads", "v_head_dim"),
    "key":    ("seq_length", "micro_batch_size", 1, "v_head_dim"),
    "hidden_states": ("seq_length", "micro_batch_size", "n_hidden"),
    # loss 段:`logits [b, s, vocab]`(gpt_model.py:365 后经 transpose,ChunkCrossEntropyLoss
    # 吃 3D;`logits.ndim == 3` @ loss.py:363 靠这份种子的**轴数**判定)
    "logits": ("micro_batch_size", "seq_length", "vocab_size"),
    "label": ("micro_batch_size", "seq_length"),
    "input_mask": ("micro_batch_size", "seq_length"),
    # MTP:`input_ids [b, s]` / `position_ids [b, s]`(gpt_model 传入;`roll_tensor` 取
    # `shape[-1]` = seq_length,`multi_token_prediction.py:174`)
    "input_ids": ("micro_batch_size", "seq_length"),
    "position_ids": ("micro_batch_size", "seq_length"),
}


def show(name, dag, dump=False):
    s = diagnostics_summary(dag)
    bad = ", ".join(f"{k}={s[k]}" for k in DIAG_KINDS if s[k]) or "clean"
    print(f"{name:34s} OK  {len(dag.nodes):3d} nodes / {len(dag.edges):3d} edges "
          f"| diag: {bad} | opaque={s['opaque_calls']} "
          f"| detached={len(dag.detached)} params={len(dag.param_operands)}")
    if s["total"]:
        print(_format_diagnostics(dag, [k for k in DIAG_KINDS if s[k]]))
    for c in dag.opaque_calls:
        print(f"    [opaque] {c.get('src')}: {str(c.get('expr'))[:110]}  kind={c.get('kind')}")
    if dump:
        dump_dag(dag)


def dump_dag(dag):
    try:
        saves = {s.name: s for s in derive_saves(dag)}
        serr = None
    except Exception as e:              # noqa: BLE001
        saves, serr = {}, f"{type(e).__name__}: {e}"
    print(f"    --- node dump ({dag.cell}) ---" + (f"  derive_saves FAIL {serr}" if serr else ""))
    for n in dag.nodes:
        out = n.out or "-"
        det = " DETACHED" if n.attrs.get("detached") else ""
        sv = " SAVED" if out.split(":")[0] in saves else ""
        pr = f" params={n.attrs.get('param_operands')}" if n.attrs.get("param_operands") else ""
        print(f"    #{n.id:<3d} {n.op:14s} {n.src:44s} ins={[i for i in n.ins]} "
              f"out={out}{det}{sv}{pr}")
    print(f"    saves({len(saves)}): {sorted(saves)}")
    print(f"    detached: {dag.detached}")
    print(f"    param_operands: {[p.get('name') for p in dag.param_operands]}")


def _try(name, fn, dump=False):
    try:
        dag = fn()
        show(name, dag, dump)
        return dag
    except Exception as e:              # noqa: BLE001
        print(f"{name:34s} FAIL {type(e).__name__}: {str(e)[:500]}")
        if os.environ.get("PROBE_TB"):
            traceback.print_exc()
        return None


def run_mhc(dump=False, strict=False, fused_mhc=True):
    flags = cell_flags(fused_mhc=fused_mhc)
    top = resolve_layer_spec(MF, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    print(f"  layer spec: {top.cell}  submodules={sorted(top.submodules)}")
    hc_cls = "FusedHyperConnectionModule" if fused_mhc else "HyperConnectionModule"
    bare = dict(BARE)
    for n in (hc_cls, "HyperConnectionOutputCell", "FusedHyperConnectionOutputCell",
              "SinkhornKnopp", "HyperConnectionHead", "Dropout", "IdentityOp",
              "MoELayer", "TopKRouter", "GroupedMLP", "SharedExpertMLP",
              "MoEAuxLossAutoScaler"):
        bare[n] = ResolvedSpec(cell=n, submodules={})
    _try(f"{hc_cls}", lambda: extract_cell(
        MF, "pynative/transformers/hyper_connection.py", hc_cls,
        ResolvedSpec(cell=hc_cls, submodules={}), flags,
        recurse=True, subcell_specs=bare, cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        kernel_call_allow=KERNEL_ALLOW, kernel_saves=KERNEL_SAVES, input_axes=INPUT_AXES, strict=strict,
    ), dump)
    _try("HyperConnectionTransformerLayer", lambda: extract_cell(
        MF, "pynative/transformers/transformer_layer.py", "HyperConnectionTransformerLayer",
        top, flags, recurse=True, subcell_specs=bare, cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        kernel_call_allow=KERNEL_ALLOW, kernel_saves=KERNEL_SAVES, input_axes=INPUT_AXES,
        injected_binds=INJECTED, strict=strict, present_params={"rotary_pos_emb"},
    ), dump)


def run_moe(dump=False, strict=False):
    flags = cell_flags()
    top = resolve_layer_spec(MF, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    mlp = top.submodules.get("mlp")
    print(f"  mlp spec: {mlp}")
    bare = dict(BARE)
    for n in ("TopKRouter", "GroupedMLP", "SharedExpertMLP", "MoEAlltoAllTokenDispatcher",
              "SequentialMLP", "MLP", "MoEAuxLossAutoScaler", "Dropout", "IdentityOp"):
        bare[n] = ResolvedSpec(cell=n, submodules={})
    _try("MoELayer", lambda: extract_cell(
        MF, "pynative/transformers/moe/moe_layer.py", "MoELayer",
        mlp if isinstance(mlp, ResolvedSpec) else ResolvedSpec("MoELayer", {}), flags,
        recurse=True, subcell_specs=bare, cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        kernel_call_allow=KERNEL_ALLOW, kernel_saves=KERNEL_SAVES, input_axes=INPUT_AXES, strict=strict,
    ), dump)
    SUB = {"SharedExpertMLP": {"linear_fc1": "Linear", "linear_fc2": "Linear"}}
    for cls, rel in (("TopKRouter", "pynative/transformers/moe/router.py"),
                     ("GroupedMLP", "pynative/transformers/moe/experts.py"),
                     ("SharedExpertMLP", "pynative/transformers/moe/shared_experts.py")):
        _try(cls, lambda cls=cls, rel=rel: extract_cell(
            MF, rel, cls, ResolvedSpec(cell=cls, submodules=SUB.get(cls, {})), flags,
            recurse=True, subcell_specs=bare, cross_file=True,
            runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
            kernel_call_allow=KERNEL_ALLOW, kernel_saves=KERNEL_SAVES, input_axes=INPUT_AXES, strict=strict,
        ), dump)


def run_emb(dump=False, strict=False):
    flags = cell_flags()
    bare = dict(BARE)
    bare["VocabEmbedding"] = ResolvedSpec(cell="VocabEmbedding", submodules={})
    for cls, rel in (("VocabEmbedding", "pynative/base_models/common/embeddings/vocab_embedding.py"),
                     ("LanguageModelEmbedding",
                      "pynative/base_models/common/embeddings/language_model_embedding.py")):
        _try(cls, lambda cls=cls, rel=rel: extract_cell(
            MF, rel, cls, ResolvedSpec(cell=cls, submodules={}), flags,
            recurse=True, subcell_specs=bare, cross_file=True,
            runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
            kernel_call_allow=KERNEL_ALLOW, kernel_saves=KERNEL_SAVES, input_axes=INPUT_AXES, strict=strict,
        ), dump)


def run_loss(dump=False, strict=False):
    flags = cell_flags()
    bare = dict(BARE)
    for n in ("_LogSoftmax", "_NLLLoss", "_LogSoftmaxModule", "_NLLLossModule",
              "_VocabParallelCrossEntropy", "_ChunkCrossEntropyLoss"):
        bare[n] = ResolvedSpec(cell=n, submodules={})
    for cls in ("CrossEntropyLoss", "ChunkCrossEntropyLoss"):
        _try(f"{cls}(pynative)", lambda cls=cls: extract_cell(
            MF, "pynative/loss/loss.py", cls,
            ResolvedSpec(cell=cls, submodules={}), flags,
            present_params={"input_mask"},
            recurse=True, subcell_specs=bare, cross_file=True,
            runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
            kernel_call_allow=KERNEL_ALLOW, kernel_saves=KERNEL_SAVES,
            input_axes=INPUT_AXES, strict=strict,
        ), dump)
    _try("__unused__", lambda: extract_cell(
        MF, "pynative/loss/loss.py", "CrossEntropyLoss",
        ResolvedSpec(cell="CrossEntropyLoss", submodules={}), flags,
        present_params={"input_mask"},
        recurse=True, subcell_specs=bare, cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        kernel_call_allow=KERNEL_ALLOW, kernel_saves=KERNEL_SAVES, input_axes=INPUT_AXES, strict=strict,
    ), dump)


def run_mtp(dump=False, strict=False):
    from cost_eval.opdag.module_resolver import resolve_spec_call, MTP_SPEC_FILES
    flags = cell_flags()
    top = resolve_layer_spec(MF, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    mtp_spec = resolve_spec_call(
        MF, "get_mtp_layer_spec", SPEC_FLAGS, spec_files=MTP_SPEC_FILES,
        keyword={"transformer_layer_spec": top, "normalization": "RMSNorm",
                 "fused_norm": True, "hc_head": "HyperConnectionHead"})
    print("  mtp spec:", sorted(mtp_spec.submodules))
    bare = dict(BARE)
    for n in ("LanguageModelEmbedding", "VocabEmbedding",
              "HyperConnectionHead", "SinkhornKnopp", "HyperConnectionOutputCell",
              "FusedHyperConnectionOutputCell", "HyperConnectionModule",
              "FusedHyperConnectionModule", "Dropout", "IdentityOp", "TopKRouter",
              "GroupedMLP", "SharedExpertMLP", "SequentialMLP", "MoEAuxLossAutoScaler"):
        bare[n] = ResolvedSpec(cell=n, submodules={})
    _try("MultiTokenPredictionLayer", lambda: extract_cell(
        MF, "pynative/transformers/multi_token_prediction.py", "MultiTokenPredictionLayer",
        mtp_spec, flags,
        # 真机 `MultiTokenPredictionBlock` 传入 input_ids/position_ids/hidden_states/
        # attention_mask/rotary_pos_emb/embedding(gpt_model.py:337-345)——全部"总被传入"
        present_params={"rotary_pos_emb", "position_ids", "attention_mask", "embedding",
                        "actual_seq_len"},
        recurse=True, subcell_specs=bare, cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        kernel_call_allow=KERNEL_ALLOW, kernel_saves=KERNEL_SAVES, input_axes=INPUT_AXES,
        injected_binds=INJECTED, strict=strict,
        param_cells={"embedding": "LanguageModelEmbedding"},
    ), dump)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    dump = "--dump" in sys.argv
    strict = "--strict" in sys.argv
    fns = {"mhc": run_mhc, "moe": run_moe, "emb": run_emb, "loss": run_loss, "mtp": run_mtp}
    for k, fn in fns.items():
        if mode in (k, "all"):
            print(f"\n{'=' * 100}\n### {k.upper()}  strict={strict}\n{'=' * 100}")
            fn(dump=dump, strict=strict)
