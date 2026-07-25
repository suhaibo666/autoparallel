# -*- coding: utf-8 -*-
"""DSv4-Flash 四 Cell 的**字节**探针:抽图 → infer_shapes(report=) → derive_saves → 字节。

用法: python scratchpad/probe_dsv4_bytes.py [fused|unfused|both] [--dump] [--gaps]
"""
import collections
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_eval.opdag.extractor import extract_cell                     # noqa: E402
from cost_eval.opdag.shape_infer import infer_shapes, merge_dims_ctx    # noqa: E402
from cost_eval.opdag.consumer import dag_saved_bytes                   # noqa: E402
from cost_eval.opdag.init_dims import INIT_PARAM_SEEDS                 # noqa: E402
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


# `__init__` 形参种子(源依据见 init_dims.INIT_PARAM_SEEDS 的 docstring)。
#   compress_ratio —— csa.py:603 / indexer.py:129 逐层传下去(缺省 0,真机 4/128);
#   layer_number   —— csa.py:552(标量,只参与 host 判定);
#   head_dim       —— **刻意不给**:`Compressor` 有两个构造点且 head_dim 不同
#                     (csa.py:604 `config.v_head_dim`=512 vs indexer.py:128
#                     `self.index_head_dim`=128),extractor 不传播构造实参 → 给一个全局值
#                     必然把另一处算错 4×。见 docs/opdag_bytes_2026-07-25.md §4 的 G2。
def seeds(ratio, head_dim=None):
    s = {"compress_ratio": ratio, "layer_number": 1}
    if head_dim is not None:
        s["head_dim"] = head_dim
    return s


SEEDS = {                        # infer_shapes 的入口种子(轴符号串)
    "x":     "S·B·H",
    "qr":    "S·B·q_lora_rank",
    "query": "S·B·n_heads·v_head_dim",
    "key":   "S·B·1·v_head_dim",
}
# 单抽 `Compressor` 时它扮演 CSA 的压缩器(csa.py:604 head_dim=config.v_head_dim)。
STANDALONE_HEAD_DIM = {"Compressor": "v_head_dim"}


def extract(name, rel, spec, fused, ratio):
    flags = dict(cell_flags(fused, ratio))
    flags[INIT_PARAM_SEEDS] = seeds(ratio, STANDALONE_HEAD_DIM.get(name))
    return extract_cell(
        MF, rel, name, spec, flags, present_params={"rotary_pos_emb"},
        recurse=True, subcell_specs=BARE, cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        kernel_call_allow=KERNEL_ALLOW, input_axes=INPUT_AXES,
        injected_binds=INJECTED,
    )


def all_dims_ctx(fused, ratio):
    """链上**各 Cell** 的 dims_ctx 并集(见 `shape_infer.merge_dims_ctx` 的理由:内联进来的
    节点里 `self.<attr>` 属于它自己那个类,顶层 dag.dims_ctx 里没有)。"""
    ctxs = []
    for name, rel, spec in targets(fused):
        ctxs.append(extract(name, rel, spec, fused, ratio).dims_ctx)
    return merge_dims_ctx(*ctxs)


def build(name, rel, spec, fused, ratio, dims_ctx=None):
    dag = extract(name, rel, spec, fused, ratio)
    report = []
    infer_shapes(dag, SEEDS, dims_ctx=dims_ctx, report=report)
    return dag, report


def run(fused: bool, dump=False, gaps=False, ratio=4):
    tag = "FUSED" if fused else "UNFUSED"
    print(f"\n{'=' * 104}\n### [{tag}] ratio={ratio}\n{'=' * 104}")
    dctx = all_dims_ctx(fused, ratio)
    print(f"merged dims_ctx ({len(dctx)}): {dctx}")
    for name, rel, spec in targets(fused):
        try:
            dag, report = build(name, rel, spec, fused, ratio, dims_ctx=dctx)
            n_ok = sum(1 for n in dag.nodes
                       if n.out and n.out.split(":")[1] not in ("?", ""))
            res = dag_saved_bytes(dag, SITE_DIMS)
            print(f"{name:28s} nodes={len(dag.nodes):3d} out-resolved={n_ok:3d} "
                  f"saves={len(res['per_save']) + len(res['unresolved']):3d} "
                  f"(ok={len(res['per_save']):2d} unres={len(res['unresolved']):2d}) "
                  f"total={res['total_bytes'] / MiB:11.3f} MiB  node-gaps={len(report)}")
            for nm, sh, dt, b in sorted(res["per_save"], key=lambda t: -t[3]):
                print(f"      [save] {nm:26s} {sh:46s} {dt:6s} {b / MiB:11.4f} MiB")
            for nm, sh, why in res["unresolved"]:
                print(f"      [UNRESOLVED save] {nm:26s} {sh}")
            if gaps:
                c = collections.Counter((r["reason"], r["prim"]) for r in report)
                for (reason, prim), k in sorted(c.items(), key=lambda t: -t[1]):
                    srcs = sorted({r["src"] for r in report
                                   if r["reason"] == reason and r["prim"] == prim})
                    print(f"      [gap x{k:3d}] {reason:24s} prim={prim:22s} "
                          f"{', '.join(srcs[:6])}{' …' if len(srcs) > 6 else ''}")
            if dump:
                for n in dag.nodes:
                    print(f"    #{n.id:<3d} {n.op:14s} {n.src:44s} ins={n.ins} out={n.out}")
        except Exception as e:                      # noqa: BLE001
            print(f"{name:28s} FAIL {type(e).__name__}: {str(e)[:500]}")
            if os.environ.get("PROBE_TB"):
                traceback.print_exc()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    d, g = "--dump" in sys.argv, "--gaps" in sys.argv
    if mode in ("fused", "both"):
        run(True, dump=d, gaps=g)
    if mode in ("unfused", "both"):
        run(False, dump=d, gaps=g)
