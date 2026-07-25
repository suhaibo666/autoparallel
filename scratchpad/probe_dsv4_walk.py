# -*- coding: utf-8 -*-
"""DSv4-Flash 四个 Cell 的抽取探针(路线 B P0#2~#5 + P1#11 验收用)。

权威快照:E:\\97-codes\\torch_parallel\\mf-src-167\\mindformers
用法:python scratchpad/probe_dsv4_walk.py [fused|unfused|both]
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MF = os.environ.get("MF_PKG", r"E:\97-codes\torch_parallel\mf-src-167\mindformers")

from cost_eval.opdag.module_resolver import resolve_layer_spec, PYNATIVE_SPEC_FILES, ResolvedSpec
from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.construct_walker import diagnostics_summary, DIAG_KINDS, _format_diagnostics
from cost_eval.opdag.bprop_rules import derive_saves

SPEC_FLAGS = {   # 取自 dsv4h_fused_pp4_recomp.yaml
    "num_experts": 8, "moe_grouped_gemm": True, "qk_layernorm": True,
    "multi_latent_attention": True, "enable_hyper_connections": True,
    "fused_norm": True, "normalization": "RMSNorm", "is_dsv4_hybrid": True,
}


def cell_flags(fused: bool, ratio: int = 4) -> dict:
    return {
        "compute_dtype": "bf16", "layernorm_compute_dtype": "fp32", "params_dtype": "bf16",
        "v_head_dim": 512, "qk_pos_emb_head_dim": 64, "q_lora_rank": 1024,
        "num_attention_heads": 64, "hidden_size": 4096, "o_groups": 8, "o_lora_rank": 1024,
        "add_bias_linear": False, "input_layout": "BSND", "num_layers": 8, "mtp_num_layers": 0,
        "index_topk": 512, "index_n_heads": 64, "index_head_dim": 128,
        "csa_window_size": 128, "csa_dense_mode": False,
        # __init__ 派生量(walker 不静态求值 __init__ 的布尔——见文档 §2 的拒绝理由,须显式注入)
        "compress_ratio": ratio, "enable_compress": True, "enable_indexer": ratio == 4,
        "is_tnd": False, "window_size": 128, "training": True,
        "apply_dsa_kernel_fusion": fused,
        # Compressor 的 `self.hadamard = Hadamard(d) if rotate else IdentityOp()`(compressor.py:129)
        "rotate": True,
        # Compressor 的 `self.overlap = compress_ratio == 4`(compressor.py:89)/ `self.coff`(:90)
        "overlap": ratio == 4, "coff": 1 + int(ratio == 4),
        "sparse_loss": True, "use_butterfly": False,
        # shape 种子用到的符号(取自 dsv4h_fused_pp4_recomp.yaml:seq_length=4096、mbs=1)
        "seq_length": 4096, "micro_batch_size": 1,
        # yaml 实值(analysis/realmachine/ab_fusion_2026-07-25/dsv4h_*_pp4_recomp.yaml:118)
        "dsa_indexer_loss_coeff": 0.001,
    }


# 部署形态谓词(hasattr/isinstance):真机 pynative 栈把 Parameter 包成 hyper_parallel DTensor,
# 故 `to_local` 存在;`full_tensor` 同理。两支对字节等价,但仍**显式**给值(不默认取某一支)。
RUNTIME_PREDICATES = {
    "hasattr:to_local": True,
    "hasattr:detach": True,
    "isinstance:DTensor": True,
}

# 纯宿主副作用调用(把逐层 indexer loss 记到模块级 dict,utils.py:41-54):无被消费的返回值。
HOST_ALLOW = ("save_to_indexer_losses_tracker", "get_indexer_loss_tracker")

# 融合 NPU 内核自由函数白名单(indexer.py:220 / csa.py:96 内部)。
KERNEL_ALLOW = ("npu_lightning_indexer",)

# 入口形参的轴符号种子(与 `infer_shapes(dag, input_shapes)` 同契约;值经 config_flags 解成数)。
# 源侧 shape 契约逐字来自 docstring / 注释:
#   Compressor.construct(x: [sq, b, hidden_size])                       compressor.py:179
#   CompressedSparseAttention.construct(query: [sq,b,np,v_head_dim],
#       key: [sq,b,1,v_head_dim], x: [sq,b,hidden_size], qr: [sq,b,q_lora_rank])  csa.py:648-651
#   DSv4HybridSelfAttention.construct(x: [sq, b, hidden_size])          deepseek_v4_...:233
#   CSAIndexer.forward_before_topk(x: (S,B,hidden), qr: (S,B,q_lora_rank))       indexer.py:162-163
INPUT_AXES = {
    "x":      ("seq_length", "micro_batch_size", "hidden_size"),
    "qr":     ("seq_length", "micro_batch_size", "q_lora_rank"),
    "query":  ("seq_length", "micro_batch_size", "num_attention_heads", "v_head_dim"),
    "key":    ("seq_length", "micro_batch_size", 1, "v_head_dim"),
}

# 抽取单个子 Cell 时,它的构造函数注入项要由调用方补上(父本会传;单抽时父不在场)。
# 源依据:`build_module(submodules.compressor/indexer, ..., rotary_pos_emb=self.rotary_pos_emb)`
#   —— csa.py:602/614、indexer.py:131、deepseek_v4_hybrid_attention.py:88。
from cost_eval.opdag.init_binder import Binding
INJECTED = {"rotary_pos_emb": Binding("Constant", {"rope_freqs": True})}

REL = "pynative/transformers/experimental_attention_variant"
BARE = {n: ResolvedSpec(cell=n, submodules={}) for n in
        ("UnfusedCSAIndexerLoss", "Hadamard")}


def targets(fused: bool):
    top = resolve_layer_spec(MF, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    sa = top.submodules["self_attention"]
    csa = sa.submodules["core_attention"]
    assert top.cell == "HyperConnectionTransformerLayer", top.cell
    assert sa.cell == "DSv4HybridSelfAttention", sa.cell           # 断言解的是**目标模型**
    assert csa.cell == "CompressedSparseAttention", csa.cell
    return [
        ("Compressor", f"{REL}/compressor.py", csa.submodules["compressor"]),
        ("CSAIndexer", f"{REL}/indexer.py", csa.submodules["indexer"]),
        ("CompressedSparseAttention", f"{REL}/csa.py", csa),
        ("DSv4HybridSelfAttention", f"{REL}/deepseek_v4_hybrid_attention.py", sa),
    ]


def run(fused: bool, dump: bool = False, ratio: int = 4, strict: bool = False):
    flags = cell_flags(fused, ratio)
    tag = "FUSED" if fused else "UNFUSED"
    print(f"\n{'=' * 96}\n### [{tag}] ratio={ratio}  strict={strict}\n{'=' * 96}")
    for name, rel, spec in targets(fused):
        try:
            dag = extract_cell(
                MF, rel, name, spec, flags, present_params={"rotary_pos_emb"},
                recurse=True, subcell_specs=BARE, cross_file=True,
                runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
                kernel_call_allow=KERNEL_ALLOW, input_axes=INPUT_AXES,
                injected_binds=INJECTED, strict=strict,
            )
            s = diagnostics_summary(dag)
            bad = ", ".join(f"{k}={s[k]}" for k in DIAG_KINDS if s[k]) or "clean"
            print(f"{name:28s} OK  {len(dag.nodes):3d} nodes / {len(dag.edges):3d} edges "
                  f"| diag: {bad} | opaque={s['opaque_calls']} "
                  f"| detached={len(dag.detached)} params={len(dag.param_operands)}")
            if s["total"]:
                print(_format_diagnostics(dag, [k for k in DIAG_KINDS if s[k]]))
            if dag.opaque_calls:
                for c in dag.opaque_calls:
                    print(f"    [opaque] {c.get('src')}: {c.get('expr')[:100]}")
            if dump:
                dump_dag(dag)
        except Exception as e:          # noqa: BLE001
            print(f"{name:28s} FAIL {type(e).__name__}: {str(e)[:400]}")
            if os.environ.get("PROBE_TB"):
                traceback.print_exc()


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
        print(f"    #{n.id:<3d} {n.op:14s} {n.src:46s} ins={[i for i in n.ins]} "
              f"out={out}{det}{sv}{pr}")
    print(f"    saves({len(saves)}): {sorted(saves)}")
    print(f"    detached: {dag.detached}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    dump = "--dump" in sys.argv
    if mode in ("fused", "both"):
        run(True, dump=dump)
    if mode in ("unfused", "both"):
        run(False, dump=dump)
