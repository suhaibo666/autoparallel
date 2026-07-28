# -*- coding: utf-8 -*-
"""字节解析的剩余阻塞项 —— G2 / G3 / G5 / T1 / T3 + 融合 mHC 内核 saved 集。

背景:`docs/opdag_bytes_2026-07-25.md` §5 与 `docs/opdag_component_coverage_2026-07-25.md` §6
两张诚实清单里归属本轮的五项。台账在 `docs/opdag_bytes_blockers_2026-07-25.md`。

**md5 门控**:凡断言真源行号/结构的测试都先校验权威快照 md5(`tests/conftest.py` 的 `MF_ROOT`
默认仍指**非权威**树)。本文件**同时**门控 `mindformers/` 与 `hyper_parallel/` 两个包 ——
融合 mHC 内核的 saved 集在后者里(2026-07-25 补入快照)。
"""
import ast
import hashlib
import os

import pytest

from cost_eval.opdag.bprop_rules import derive_saves
from cost_eval.opdag.extractor import extract_cell, _ctor_seeds
from cost_eval.opdag.init_binder import Binding
from cost_eval.opdag.init_dims import INIT_PARAM_SEEDS, eval_init_dims
from cost_eval.opdag.module_resolver import (
    PYNATIVE_SPEC_FILES, ResolvedSpec, resolve_layer_spec,
)
from cost_eval.opdag.shape_infer import infer_shapes, _frame_chain

REL = "pynative/transformers/experimental_attention_variant"

# ── 权威快照门控 ────────────────────────────────────────────────────────────────────
_MD5_MF = {
    f"{REL}/csa.py": "81673be3ad3cdd2191e28dd000a13f0e",
    f"{REL}/indexer.py": "8e18fee21c33507dd629c89f401986e6",
}
_MD5_HP = {
    "core/shard/ops/parallel_mhc_pre_sinkhorn.py": "600cb4961cda0ae4225c708479e3ce77",
    "core/shard/ops/parallel_mhc_post.py": "f1cbbf2d0c4a60f43c7b3756069a24e8",
}


def _md5_ok(pkg, table):
    for rel, want in table.items():
        p = os.path.join(pkg, *rel.split("/"))
        if not os.path.isfile(p):
            return False
        with open(p, "rb") as fh:
            if hashlib.md5(fh.read()).hexdigest() != want:
                return False
    return True


def _snapshot_root():
    for root in (os.environ.get("MINDFORMERS_ROOT"),
                 r"E:\97-codes\torch_parallel\mf-src-167"):
        if root and _md5_ok(os.path.join(root, "mindformers"), _MD5_MF):
            return root
    return None


@pytest.fixture(scope="module")
def mf_pkg():
    root = _snapshot_root()
    if root is None:
        pytest.skip("权威 mindformers 快照缺失/md5 不匹配(绝不对着另一个 commit 断言)")
    return os.path.join(root, "mindformers")


@pytest.fixture(scope="module")
def hp_pkg():
    root = _snapshot_root()
    if root is None or not _md5_ok(os.path.join(root, "hyper_parallel"), _MD5_HP):
        pytest.skip("权威 hyper_parallel 快照缺失/md5 不匹配")
    return os.path.join(root, "hyper_parallel")


# ── 抽取配置(与 tests/test_opdag_walker_core.py 同口径)────────────────────────────────
SPEC_FLAGS = {
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
        "compress_ratio": ratio, "enable_compress": True, "enable_indexer": ratio == 4,
        "is_tnd": False, "window_size": 128, "training": True,
        "apply_dsa_kernel_fusion": fused,
        "rotate": True, "overlap": ratio == 4, "coff": 1 + int(ratio == 4),
        "sparse_loss": True, "use_butterfly": False,
        "seq_length": 4096, "micro_batch_size": 1, "dsa_indexer_loss_coeff": 0.001,
    }


RUNTIME_PREDICATES = {"hasattr:to_local": True, "hasattr:detach": True,
                      "isinstance:DTensor": True}
HOST_ALLOW = ("save_to_indexer_losses_tracker", "get_indexer_loss_tracker")
KERNEL_ALLOW = ("npu_lightning_indexer",)
INPUT_AXES = {
    "x": ("seq_length", "micro_batch_size", "hidden_size"),
    "qr": ("seq_length", "micro_batch_size", "q_lora_rank"),
    "query": ("seq_length", "micro_batch_size", "num_attention_heads", "v_head_dim"),
    "key": ("seq_length", "micro_batch_size", 1, "v_head_dim"),
}
INJECTED = {"rotary_pos_emb": Binding("Constant", {"rope_freqs": True})}
BARE = {n: ResolvedSpec(cell=n, submodules={})
        for n in ("UnfusedCSAIndexerLoss", "Hadamard")}
SEEDS = {"x": "S·B·H", "qr": "S·B·q_lora_rank",
         "query": "S·B·n_heads·v_head_dim", "key": "S·B·1·v_head_dim"}


def _targets(mf_pkg):
    top = resolve_layer_spec(mf_pkg, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    sa = top.submodules["self_attention"]
    csa = sa.submodules["core_attention"]
    assert sa.cell == "DSv4HybridSelfAttention"
    assert csa.cell == "CompressedSparseAttention"
    return {"Compressor": (f"{REL}/compressor.py", csa.submodules["compressor"]),
            "CSAIndexer": (f"{REL}/indexer.py", csa.submodules["indexer"]),
            "CompressedSparseAttention": (f"{REL}/csa.py", csa),
            "DSv4HybridSelfAttention":
                (f"{REL}/deepseek_v4_hybrid_attention.py", sa)}


def _extract(mf_pkg, cls, fused=True, ratio=4, seeds=None):
    rel, spec = _targets(mf_pkg)[cls]
    flags = cell_flags(fused, ratio)
    if seeds:
        flags[INIT_PARAM_SEEDS] = seeds
    return extract_cell(
        mf_pkg, rel, cls, spec, flags, present_params={"rotary_pos_emb"},
        recurse=True, subcell_specs=BARE, cross_file=True,
        runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        kernel_call_allow=KERNEL_ALLOW, input_axes=INPUT_AXES,
        injected_binds=INJECTED)


@pytest.fixture(scope="module")
def csa_dag(mf_pkg):
    dag = _extract(mf_pkg, "CompressedSparseAttention",
                   seeds={"compress_ratio": 4, "layer_number": 1})
    infer_shapes(dag, SEEDS)
    return dag


# ═══ G2:构造点维度关键字**按构造点**传播 ══════════════════════════════════════════════

def test_g2_the_two_compressor_construction_sites_seed_different_head_dims(mf_pkg):
    """`Compressor` 在快照里有两个构造点,`head_dim` 一个 512 一个 128 —— 差 **4×**。

    单一全局 `INIT_PARAM_SEEDS["head_dim"]` 必然把另一处算错,故必须按构造点求。
    逐字源:`csa.py:596-603`(`head_dim=config.v_head_dim`)、
           `indexer.py:126-132`(`head_dim=self.index_head_dim`)。
    """
    def _seed_at(rel, cls, lineno):
        with open(os.path.join(mf_pkg, *rel.split("/")), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        flags = cell_flags(True, 4)
        flags[INIT_PARAM_SEEDS] = {"compress_ratio": 4, "layer_number": 1}
        idm = eval_init_dims(tree, cls, flags)
        call = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and getattr(n, "lineno", 0) == lineno)
        return _ctor_seeds(call, idm.self_seeds, flags)

    csa_site = _seed_at(f"{REL}/csa.py", "CompressedSparseAttention", 596)
    idx_site = _seed_at(f"{REL}/indexer.py", "CSAIndexer", 125)

    # 维度按**符号**传(随 DimTable 变,不写死数),值同时带上供 `if` 判定。
    assert csa_site["head_dim"] == {"dim": "v_head_dim", "val": 512}
    assert idx_site["head_dim"] == {"dim": "index_head_dim", "val": None}
    # 就是这条差异让两侧的压缩产物差 4×(512/128);两个构造点**必须**给出不同的种子。
    assert csa_site["head_dim"]["dim"] != idx_site["head_dim"]["dim"]
    # `rotate` 也是逐构造点的字面量:csa.py:601 False / indexer.py:130 True。
    assert csa_site["rotate"] == {"dim": None, "val": False}
    assert idx_site["rotate"] == {"dim": None, "val": True}


def test_g2_inlined_indexer_compressor_resolves_with_index_head_dim(csa_dag):
    """G2 的**字节级**判据:CSA 图里 indexer 那条链上的 compressor 产物按
    `index_head_dim`(128)解出,而**不是** `v_head_dim`(512)。

    改前这 58 个内联 compressor 节点全报 `no_out_dim`(见 opdag_bytes §5 的 G2)。
    """
    shapes = [n.out.split(":")[1] for n in csa_dag.nodes if n.out.count(":") == 2]
    idx_side = [s for s in shapes if "index_head_dim" in s]
    assert idx_side, "indexer 侧 compressor 一个都没按 index_head_dim 解出 → G2 未生效"
    # 且**没有**把 v_head_dim 的那份也算成 index_head_dim(反之亦然):两者共存 = 分开解对了。
    assert any("v_head_dim" in s for s in shapes)


def test_g2_seeds_never_come_from_an_init_default(mf_pkg):
    """铁律:**不得从 `__init__` 缺省推结构**。`compress_ratio` 在 `csa.py:556` 缺省 `0`,
    真机是 4/128 —— 该缺省绝不能经 `self_seeds` 传播到子 `Compressor` 的构造点
    (实测传下去会让 `cutoff = (sq // ratio) * ratio` 判不出,20 条既有测试当场变红)。"""
    with open(os.path.join(mf_pkg, *f"{REL}/csa.py".split("/")), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    flags = cell_flags(True, 4)          # **不给** INIT_PARAM_SEEDS → 只剩 __init__ 缺省
    idm = eval_init_dims(tree, "CompressedSparseAttention", flags)
    assert "compress_ratio" not in idm.self_seeds, (
        "来自 __init__ 缺省的值泄漏进了 self_seeds —— 会把一个与真机相反的前提扩散给子 Cell")
    # 给了种子(调用方声明的权威事实)就该传播。
    flags[INIT_PARAM_SEEDS] = {"compress_ratio": 4, "layer_number": 1}
    idm2 = eval_init_dims(tree, "CompressedSparseAttention", flags)
    assert idm2.self_seeds["compress_ratio"]["val"] == 4


def test_g2_ctor_seeds_do_not_swallow_dtype_or_module_kwargs(mf_pkg):
    """构造点关键字里 `params_dtype=` / `init_method=` / `rotary_pos_emb=` 不是维度 ——
    塞进 `INIT_PARAM_SEEDS` 会被 `parse_axis` 当成符号维度而污染代数。"""
    with open(os.path.join(mf_pkg, *f"{REL}/csa.py".split("/")), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    flags = cell_flags(True, 4)
    flags[INIT_PARAM_SEEDS] = {"compress_ratio": 4, "layer_number": 1}
    idm = eval_init_dims(tree, "CompressedSparseAttention", flags)
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n, "lineno", 0) == 596)
    seeds = _ctor_seeds(call, idm.self_seeds, flags)
    for bad in ("config", "rotary_pos_emb", "params_dtype", "compute_dtype", "init_method"):
        assert bad not in seeds


# ═══ G3:内联子 Cell 的 dims_ctx 按帧挂到节点上 ═══════════════════════════════════════

def test_g3_inlined_nodes_carry_their_own_class_dims_ctx(csa_dag):
    """内联节点带**它自己那个类**的 `dims_ctx`(顶层那份没有 `index_n_heads` 之类)。"""
    framed = [n for n in csa_dag.nodes if n.attrs.get("frame")]
    assert framed, "没有任何节点带内联帧 → G3 未生效"
    withctx = [n for n in framed if n.attrs.get("dims_ctx")]
    assert withctx, "内联节点没有带 dims_ctx"
    # `indexer.py:178` 的 reshape 用 `self.index_n_heads`/`self.index_head_dim` ——
    # 那两个符号只在 `CSAIndexer.__init__` 里,顶层 CSA 的 dims_ctx 里没有。
    assert "index_n_heads" not in (csa_dag.dims_ctx or {})
    assert any("index_n_heads" in (n.attrs.get("dims_ctx") or {}) for n in withctx)


def test_g3_the_same_attr_resolves_differently_per_frame(csa_dag):
    """**G3 存在的理由**:同一个 `self.head_dim` 在两个 compressor 帧里解出不同符号 ——
    扁平并表(`merge_dims_ctx`)只能 fail-loud 或静默取一个。"""
    seen = {}
    for n in csa_dag.nodes:
        d = n.attrs.get("dims_ctx") or {}
        if "head_dim" in d:
            seen.setdefault(d["head_dim"], set()).add(n.attrs.get("frame", ""))
    assert len(seen) >= 2, f"两个构造点的 head_dim 没有分开:{seen}"
    assert {"v_head_dim", "index_head_dim"} <= set(seen)


def test_g3_frame_chain_is_innermost_first_then_outward():
    """帧的可见链:当前帧 → 逐级外层 → 根帧。内层必须**先**命中。"""
    assert _frame_chain("a@1/b@7") == ["a@1/b@7", "a@1", ""]
    assert _frame_chain("a@1") == ["a@1", ""]
    assert _frame_chain("") == [""]


# ═══ G5:construct 局部标量环境过子 Cell 边界 ══════════════════════════════════════════

def test_g5_scalar_binds_cross_the_subcell_boundary(csa_dag):
    """G5:`seqlen, bsz, _ = x.shape`(`indexer.py:173`)这类解包此前不过边界 →
    内联后子里的 `reshape(q, (seqlen, …))` 整个解不出(实测 CSA 只剩 **1** 条 scalar_binds)。"""
    binds = csa_dag.scalar_binds
    assert len(binds) > 1, f"scalar_binds 仍只有 {len(binds)} 条 → G5 未生效"
    framed = [b for b in binds if b.get("frame")]
    assert framed, "没有任何 scalar_bind 带帧"
    # `src` 必须被重映射成**调用方**那一侧的基名(否则父的 env 里查不到子形参名)。
    caller_names = {n.out.split(":")[0] for n in csa_dag.nodes if n.out}
    caller_names |= set(SEEDS)
    assert any(b["src"] in caller_names for b in framed), (
        f"子 scalar_bind 的 src 没有一个能在父的名字空间里落地:{[b['src'] for b in framed]}")


def test_g5_subextract_declares_the_two_new_scope_fields():
    """契约变更本身钉住:`SubExtract` 必须携 `dims_ctx` 与 `scalar_binds`。"""
    from cost_eval.opdag.construct_walker import SubExtract
    se = SubExtract()
    assert se.dims_ctx == {} and se.scalar_binds == []


# ═══ T3:多输出算子的 dtype ═══════════════════════════════════════════════════════════

def test_t3_topk_indices_is_saved_as_int32_not_the_first_of_two_bindings(mf_pkg):
    """`topk_indices` 在 `indexer.py:262`(TopK 第 2 个输出)与 `:263`(`cast(…, int32)`)
    **同名**;`derive_saves` 按名去重只留首见 → 若两者 dtype 不同就会**静默取错**。

    源侧真值:`mint.topk` 的第 2 个输出是**索引**,`:263` 又显式 cast 成 int32 ⇒ 4 字节。
    若退回 bf16(2 字节)= 少算一半。
    """
    dag = _extract(mf_pkg, "CSAIndexer", fused=False,
                   seeds={"compress_ratio": 4, "layer_number": 1})
    topk = [n for n in dag.nodes if n.op == "TopK"]
    assert len(topk) == 1 and topk[0].src.endswith(":262")
    # 多输出按 `attrs["outs"]` 逐个登记 SSA dtype(第 2 个 = int32)。
    assert topk[0].attrs["outs"][1].split(":")[2] == "int32"
    # 紧随其后的 cast(:263)的 ins 也必须已是 int32(不是继承来的 bf16)。
    cast263 = next(n for n in dag.nodes if n.src.endswith(":263"))
    assert cast263.ins[0].split(":")[2] == "int32"
    # 最终 save 的那一份是 int32。
    s = next(s for s in derive_saves(dag) if s.name == "topk_indices")
    assert s.dtype == "int32", f"topk_indices 存成了 {s.dtype} —— 双名去重取错了分支"
    assert s.op_id == topk[0].id       # 来自 TopK 的 outs[1],不是 Cast


# ═══ 融合 mHC 内核的 saved 集 —— 从 hyper_parallel 源逐字读 ════════════════════════════

def test_mhc_kernel_saved_sets_are_read_from_source_not_bounded(hp_pkg):
    """`npu_mhc_pre_sinkhorn` / `npu_mhc_post` 的 saved 集**逐字写在源码里**
    (`hyper_parallel` 已于 2026-07-25 补入快照),不再需要「调用方声明的保守上界」。

    关键:旧上界 `saved_ins_idx="all"` 在 `post` 上恰等于源真值,但在 `pre_sinkhorn` 上是
    **欠读**而非上界 —— 它漏掉 5 个被保存的**自身输出**。
    """
    from cost_eval.opdag.fn_saves import mhc_kernel_saves
    tbl = mhc_kernel_saves(hp_pkg)

    post = tbl["npu_mhc_post"]
    assert post["saved_ins_idx"] == [0, 1, 2, 3]          # x, h_res, h_out, h_post
    assert post["saved_outs_idx"] == []
    assert post["source"].endswith("custom_op_impl.py:331")

    pre = tbl["npu_mhc_pre_sinkhorn"]
    assert pre["saved_ins_idx"] == [0, 1, 2, 3]           # x, phi, alpha, bias
    # 自身输出 #3..#7 = h_pre, hc_before_norm, inv_rms, sum_out, norm_out
    assert pre["saved_outs_idx"] == [3, 4, 5, 6, 7]
    assert pre["source"].endswith("custom_op_impl.py:390")
    # 按**输出位序**建索引:源里 `h_in, h_post, h_res_flat, *_ = ...`(hyper_connection.py:413)
    # 把 #3..#7 丢弃了,图上没有 ref → 只能靠这张表登记(丢弃 ≠ 不占显存,ctx 仍持有)。
    assert pre["saved_out_names"] == {
        3: "h_pre", 4: "hc_before_norm", 5: "inv_rms", 6: "sum_out", 7: "norm_out"}


def test_mhc_pre_sinkhorn_only_two_saved_outputs_have_a_source_known_shape(hp_pkg):
    """诚实边界:`parallel_mhc_pre_sinkhorn.py:261` 把 5 个输出**共用**一个 3-D tensor_map
    `(b_map, s_map, -1)` —— 末轴写 `-1`(复制,与分片无关),**不给长度**。
    只有 `sum_out`(`:262-263`)与 `norm_out`(`:264-265`)的形状在源侧确定。
    其余三项一律进 `unresolved`,**不给数**。"""
    from cost_eval.opdag.fn_saves import mhc_kernel_saves
    pre = mhc_kernel_saves(hp_pkg)["npu_mhc_pre_sinkhorn"]
    shapes = pre["saved_out_shapes"]
    assert shapes["sum_out"] == "2·num_iters·B·S·N"
    assert shapes["norm_out"] == "2·num_iters·B·S·N·N"
    for unknown in ("h_pre", "hc_before_norm", "inv_rms"):
        assert unknown not in shapes, (
            f"{unknown} 的末轴长度在 .cc kernel 里(未纳入快照),源侧不确定 —— 不许给数")


def test_mhc_declared_bound_stays_labelled_as_a_bound_not_a_fact():
    """调用方声明的上界与「从源读出来的」必须在图上**可区分** —— 绝不静默升格为事实。"""
    from cost_eval.opdag.fn_saves import KERNEL_SAVE_SOURCE_KEYS
    assert "saved_from_source" in KERNEL_SAVE_SOURCE_KEYS
    assert "saved_declared_by_caller" in KERNEL_SAVE_SOURCE_KEYS
