# -*- coding: utf-8 -*-
"""DSv4-Flash 四 Cell 的**字节**探针:抽图 → infer_shapes → derive_saves → dag_saved_bytes。

用法: python scratchpad/probe_dsv4_bytes.py [fused|unfused|both] [--dump]
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_eval.opdag.extractor import extract_cell                     # noqa: E402
from cost_eval.opdag.shape_infer import infer_shapes                   # noqa: E402
from cost_eval.opdag.bprop_rules import derive_saves                   # noqa: E402
from cost_eval.opdag.consumer import dag_saved_bytes                   # noqa: E402
from cost_eval.model_spec import DimTable                              # noqa: E402
from scratchpad.probe_dsv4_walk import (                               # noqa: E402
    MF, cell_flags, targets, BARE, INJECTED, RUNTIME_PREDICATES,
    HOST_ALLOW, KERNEL_ALLOW, INPUT_AXES,
)

MiB = 1024 * 1024

SITE_DIMS = DimTable(
    H=4096, F=4096, n_heads=64, n_kv=1, head_dim=128, S=4096, B=1, vocab=129280,
    n_layers=8, q_lora_rank=1024, kv_lora_rank=512,
    qk_rope_head_dim=64, qk_nope_head_dim=0, v_head_dim=512,
    dsa_indexer_n_heads=64, dsa_indexer_head_dim=128, dsa_indexer_topk=512,
    o_groups=8, o_lora_rank=1024, csa_window_size=128,
)

# infer_shapes 的入口种子(与 probe_dsv4_walk.INPUT_AXES 同源,但这里是**符号串**)。
SEEDS = {
    "x":     "S·B·H",
    "qr":    "S·B·q_lora_rank",
    "query": "S·B·n_heads·v_head_dim",
    "key":   "S·B·1·v_head_dim",
}


def run(fused: bool, dump: bool = False, ratio: int = 4):
    flags = cell_flags(fused, ratio)
    tag = "FUSED" if fused else "UNFUSED"
    print(f"\n{'=' * 100}\n### [{tag}] ratio={ratio}\n{'=' * 100}")
    for name, rel, spec in targets(fused):
        try:
            dag = extract_cell(
                MF, rel, name, spec, flags, present_params={"rotary_pos_emb"},
                recurse=True, subcell_specs=BARE, cross_file=True,
                runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
                kernel_call_allow=KERNEL_ALLOW, input_axes=INPUT_AXES,
                injected_binds=INJECTED,
            )
            infer_shapes(dag, SEEDS)
            n_q = sum(1 for n in dag.nodes if n.out and n.out.split(":")[1] == "?")
            res = dag_saved_bytes(dag, SITE_DIMS)
            print(f"{name:28s} nodes={len(dag.nodes):3d} out-?={n_q:3d} "
                  f"saves={len(res['per_save']) + len(res['unresolved']):3d} "
                  f"resolved={len(res['per_save']):3d} unresolved={len(res['unresolved']):3d} "
                  f"total={res['total_bytes'] / MiB:10.3f} MiB")
            for nm, sh, why in res["unresolved"]:
                print(f"      [UNRESOLVED] {nm:34s} {sh}")
            if dump:
                for n in dag.nodes:
                    print(f"    #{n.id:<3d} {n.op:14s} {n.src:44s} ins={n.ins} out={n.out}")
                for nm, sh, dt, b in sorted(res["per_save"]):
                    print(f"    [save] {nm:34s} {sh:40s} {dt:6s} {b / MiB:10.4f} MiB")
        except Exception as e:                      # noqa: BLE001
            print(f"{name:28s} FAIL {type(e).__name__}: {str(e)[:500]}")
            if os.environ.get("PROBE_TB"):
                traceback.print_exc()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    d = "--dump" in sys.argv
    if mode in ("fused", "both"):
        run(True, dump=d)
    if mode in ("unfused", "both"):
        run(False, dump=d)
