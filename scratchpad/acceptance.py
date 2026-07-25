# -*- coding: utf-8 -*-
"""字节解析的验收台:逐节点字节 + param census + 契约校验 + 手写 spec 对账。"""
import ast
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_eval.opdag.bprop_rules import derive_saves                     # noqa: E402
from cost_eval.opdag.consumer import (                                   # noqa: E402
    dag_local_saved_bytes, dag_param_bytes, detach_aliases,
    resolve_shape_elems, _DTYPE_BYTES, _dtype_bytes,
)
from cost_eval.opdag.init_dims import (                                  # noqa: E402
    eval_init_dims, InitDims, INIT_PARAM_SEEDS)
from cost_eval.opdag.shape_infer import infer_shapes, merge_dims_ctx     # noqa: E402
from cost_eval.liveness.contract import (                                # noqa: E402
    validate_resolved_layer, validate_param_census, layer_param_bytes,
    layer_total_bytes, layer_activation_bytes,
)
from cost_eval.parallel_model import ParallelModel                       # noqa: E402
from cost_eval.specs import ParallelConfig                               # noqa: E402
from scratchpad.probe_dsv4_bytes import (                                # noqa: E402
    MF, SITE_DIMS, SEEDS, extract, targets, seeds, STANDALONE_HEAD_DIM, MiB,
)
from scratchpad.probe_dsv4_walk import cell_flags                        # noqa: E402


def _pm(**kw):
    pc = ParallelConfig(**{"dp_shard": 1, "tp": 1, "cp": 1, "ep": 1, "pp": 1, **kw})
    return ParallelModel(pc, n_layers=8, world_size=1, edge_pseudo=(0, 0))


def init_dims_of(name, rel, fused, ratio):
    path = os.path.join(MF, *rel.split("/"))
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    flags = dict(cell_flags(fused, ratio))
    flags[INIT_PARAM_SEEDS] = seeds(ratio, STANDALONE_HEAD_DIM.get(name))
    return eval_init_dims(tree, name, flags)


def merged_init_dims(fused, ratio):
    """链上各 Cell 的 param_shapes/param_dtypes 并集 —— 与 merge_dims_ctx 同一条理由:
    内联进来的 Parameter(compressor 的 ape、csa 的 attn_sink)属于**它自己那个类**的
    __init__,顶层那份求不出它。同名冲突即断言失败。"""
    ps, pd = {}, {}
    for name, rel, _spec in targets(fused):
        idm = init_dims_of(name, rel, fused, ratio)
        for k, v in (idm.param_shapes or {}).items():
            prev = ps.setdefault(k, v)
            assert prev == v, "param_shapes conflict %s: %s vs %s" % (k, prev, v)
        for k, v in (idm.param_dtypes or {}).items():
            pd.setdefault(k, v)
    return InitDims(param_shapes=ps, param_dtypes=pd)


# ── 契约面的最小实现(**不是** to_resolved.py:只为跑校验器,不注册 graph source)───────
class T:
    def __init__(self, name, numel, dtb, is_weight=False, detached=False, dim0=0):
        self.name, self.local_numel, self.dtype_bytes = name, numel, dtb
        self.is_weight, self.detached = is_weight, detached
        self.is_expert, self.dim0, self.pin_under_recompute = False, dim0, False


class O:
    def __init__(self, name, typ, ins, out, params, saves):
        self.name, self.type = name, typ
        self.inputs, self.output, self.params, self.saves = ins, out, params, saves
        self.workspace_bytes = self.bwd_scratch_bytes = 0


class L:
    def __init__(self, lid, lt, ops):
        self.layer_id, self.layer_type, self.ops = lid, lt, ops


def resolved_layer(dag, dims, idims, layer_id=1):
    """把**已解析**的节点折成一个满足契约面的层(解不出的节点整个跳过 —— 它们过不了 B1,
    这正是契约设计要挡的东西,见 contract.py 的 B1 说明)。"""
    saves_by_op = {}
    for s in derive_saves(dag):
        saves_by_op.setdefault(s.op_id, []).append(s)
    alias = detach_aliases(dag)
    pshapes = dict(getattr(idims, "param_shapes", {}) or {})
    pdtypes = dict(getattr(idims, "param_dtypes", {}) or {})
    tcache, ops, skipped = {}, [], []
    detached_names = set(getattr(dag, "detached", ()) or ())

    def mk(ref, is_weight=False):
        if ref.count(":") != 2:
            return None
        nm, shp, dt = ref.split(":")
        if nm in tcache:
            return tcache[nm]
        el = resolve_shape_elems(shp, dims)
        if el is None or el <= 0:
            return None
        t = T(nm, el, _dtype_bytes(dt, dims), is_weight,
              detached=(not is_weight and nm in detached_names))
        tcache[nm] = t
        return t

    for n in dag.nodes:
        if not n.out:
            continue
        out = mk(n.out)
        ins = [mk(r) for r in n.ins]
        if out is None or any(i is None for i in ins):
            skipped.append((n.id, n.op, n.src))
            continue
        pars = []
        for pn in (n.attrs.get("param_operands") or ()):
            axes = pshapes.get(pn)
            if not axes:
                continue
            pt = mk("%s:%s:%s" % (pn, "·".join(axes), pdtypes.get(pn) or "bf16"),
                    is_weight=True)
            if pt is not None:
                pars.append(pt)
        sv = []
        for s in saves_by_op.get(n.id, ()):
            if alias.get(s.name, s.name) != s.name:
                continue                       # detach 别名:同一块存储,不重复
            st = mk("%s:%s:%s" % (s.name, s.sym_shape, s.dtype))
            if st is not None:
                sv.append(st)
        ops.append(O("n%d_%s" % (n.id, n.op), n.op.lower(), ins, out, pars, sv))
    return L(layer_id, dag.cell, ops), skipped


# 抽取侧名 -> 手写 spec 名(逐条依据 tests/test_source_truth_saves_audit.ALIAS + csa.py 实参位序)
HAND_ALIAS = {
    "query": "q_hnorm_fp32", "key": "kv_a_out", "compressed_kv": "compressed_kv",
    "topk_indices": "topk_indices", "q": "q", "x": "ln1",
    "q_compressed": "q_compressed", "kv": "kv", "qr_detach": "q_a_out",
    "x_detach": "ln1",
}


def hand_saves(fused, ratio=4):
    from cost_eval.layers.dsv4_hybrid import build_dsv4_hybrid_attn_ops
    from cost_eval.model_spec import DimTable
    from cost_eval.shape_eval import resolve_tensor
    d = DimTable(**{**SITE_DIMS.as_dict(), "dsa_fused": fused})
    out = {}
    for op in build_dsv4_hybrid_attn_ops(d, ratio):
        for t in op.saves:
            r = resolve_tensor(t, d, _pm())
            out[t.name] = (op.name, "*".join(t.shape), r.dtype_bytes,
                           r.local_numel * r.dtype_bytes, t.detached)
    return out


def compare(fused):
    dctx = merge_dims_ctx(*[extract(n, r, s, fused, 4).dims_ctx
                            for n, r, s in targets(fused)])
    hand = hand_saves(fused)
    print("")
    print("### 手写 spec vs 抽取图 —— 逐张量对账 (%s)" % ("fused" if fused else "unfused"))
    print("%-26s %-22s %10s %10s %10s  %s"
          % ("extracted", "hand", "ext MiB", "hand MiB", "delta", "verdict"))
    for cell in ("CompressedSparseAttention", "DSv4HybridSelfAttention"):
        name, rel, spec = next(t for t in targets(fused) if t[0] == cell)
        dag = extract(name, rel, spec, fused, 4)
        infer_shapes(dag, SEEDS, dims_ctx=dctx, report=[])
        res = dag_local_saved_bytes(dag, SITE_DIMS, _pm())
        print("-- %s" % cell)
        for nm, sh, dt, b in sorted(res["per_save"], key=lambda t: -t[3]):
            base = nm.split("__i")[0]
            hn = HAND_ALIAS.get(base, base)
            h = hand.get(hn)
            if h is None:
                print("%-26s %-22s %10.3f %10s %10s  仅抽取侧"
                      % (nm, "(no counterpart)", b / MiB, "-", "-"))
                continue
            print("%-26s %-22s %10.3f %10.3f %+10.3f  %s (dt %d vs %d)"
                  % (nm, hn, b / MiB, h[3] / MiB, (b - h[3]) / MiB,
                     "EQUAL" if b == h[3] else "DIFF",
                     _DTYPE_BYTES.get(dt, 2), h[2]))
    sa = {k: v for k, v in hand.items() if v[0] == "sparse_attn"}
    print("-- 手写 sparse_attn 完整名册: %d 项 / %.3f MiB"
          % (len(sa), sum(v[3] for v in sa.values()) / MiB))
    for k, v in sorted(sa.items(), key=lambda kv: -kv[1][3]):
        print("     %-22s %-48s dt=%d det=%s %10.3f MiB"
              % (k, v[1], v[2], v[4], v[3] / MiB))


def main():
    pm = _pm()
    for fused in (True, False):
        print("")
        print("#" * 100)
        print("## [%s] ratio=4  (seq4096 / topk512 / 64 heads / v_head_dim 512 / b=1)"
              % ("FUSED" if fused else "UNFUSED"))
        print("#" * 100)
        dctx = merge_dims_ctx(*[extract(n, r, s, fused, 4).dims_ctx
                                for n, r, s in targets(fused)])
        idims = merged_init_dims(fused, 4)
        for name, rel, spec in targets(fused):
            dag = extract(name, rel, spec, fused, 4)
            rep = []
            infer_shapes(dag, SEEDS, dims_ctx=dctx, report=rep)
            sav = dag_local_saved_bytes(dag, SITE_DIMS, pm)
            par = dag_param_bytes(dag, SITE_DIMS, idims, pm)
            print("")
            print("### %s  (%d nodes)" % (name, len(dag.nodes)))
            print("  saves: %d resolved / %d unresolved / %d detach-aliased   total = %.3f MiB"
                  % (len(sav["per_save"]), len(sav["unresolved"]),
                     len(sav.get("detach_aliased", [])), sav["total_bytes"] / MiB))
            for nm, sh, dt, b in sorted(sav["per_save"], key=lambda t: -t[3]):
                print("    save  %-24s %-44s %-6s %10.4f MiB" % (nm, sh, dt, b / MiB))
            for nm, root, sh, dt, b in sav.get("detach_aliased", []):
                print("    ALIAS %-24s -> %-20s (same storage, not counted) %8.4f MiB"
                      % (nm, root, b / MiB))
            for nm, sh, why in sav["unresolved"]:
                node = next((r for r in rep if r["name"] == nm), None)
                where = (node["src"] + " " + node["reason"]) if node else "no producing node"
                print("    UNRES %-24s %-8s <- %s" % (nm, sh, where))
            print("  params: %d resolved / %d unresolved   total = %.4f MiB"
                  % (len(par["per_param"]), len(par["unresolved"]), par["total_bytes"] / MiB))
            for nm, sh, dt, b in par["per_param"]:
                print("    param %-24s %-32s %-6s %12.6f MiB" % (nm, sh, str(dt), b / MiB))
            for nm, sh, why in par["unresolved"]:
                print("    UNRES-param %-20s %s" % (nm, why))
            layer, skipped = resolved_layer(dag, SITE_DIMS, idims)
            v = validate_resolved_layer(layer)
            print("  contract: ops=%d (skipped %d unresolved) violations=%d  "
                  "total=%.3f MiB param=%.4f act=%.3f"
                  % (len(layer.ops), len(skipped), len(v), layer_total_bytes(layer) / MiB,
                     layer_param_bytes(layer) / MiB, layer_activation_bytes(layer) / MiB))
            for x in v[:8]:
                print("    VIOLATION %s" % x)
            pc = validate_param_census(layer, par["total_bytes"])
            print("  param_census(vs dag_param_bytes) = %s" % ("PASS" if not pc else pc[0]))
            c = collections.Counter(r["reason"] for r in rep)
            print("  node-gaps: " + ", ".join("%s=%d" % (k, n) for k, n in c.most_common()))
        compare(fused)


if __name__ == "__main__":
    main()
