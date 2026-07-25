# -*- coding: utf-8 -*-
"""路线 B P0#2/#3/#4/#5 + P1#11 —— 让纯 AST 抽取器**真正走通** DSv4-Flash 的 `pynative/` 侧。

背景与判决见 `docs/opdag_coverage_assessment_2026-07-25.md` §2/§5/§6 与
`docs/opdag_walker_core_2026-07-25.md`。五项:

  1. **裸函数别名绑定**(P0#2):`self.reshape = mint.reshape` 这类 pynative 成文约定
     (`csa.py:628-630`)此前完全不可见。表在 `primitives.PRIMITIVES`,未知原语 fail-loud。
  2. **跨文件 MRO**(P0#3):`DSv4HybridSelfAttention` 的基类在 `multi_latent_attention.py`,
     基类 `__init__` 的 `self.shape/reshape/cast/permute` 收不到。
  3. **`ast.With` 的块级语义**(P0#4):`with _no_grad():`(`indexer.py:214`)块体要走查,
     块内产物标 `detached`;语义未知的 `with` fail-loud。
  4. **赋值 RHS 形态**(P0#5):`BinOp/Compare/Subscript/Name/Attribute` —— **这次要修的张量
     全在这里**;同时必须把「标量/dtype 记账」与「张量建节点」分开(不造假节点)。
  5. **`ops.stop_gradient` → 真 `Detach` 节点**(P1#11):建节点 + **保边** + 打 `detached` 标。

**md5 门控**:凡断言真源行号/名单的测试都先校验权威快照的 md5(`tests/conftest.py` 的
`MF_ROOT` 默认仍指**非权威**树,见 `docs/opdag_source_truth_2026-07-25.md` §4 第 7 条)。
"""
import hashlib
import os

import pytest

from cost_eval.opdag import primitives as prims
from cost_eval.opdag.bprop_rules import PIN, derive_saves
from cost_eval.opdag.construct_walker import (
    ExtractionDroppedError, assert_extraction_clean, diagnostics_summary, walk_construct,
)
from cost_eval.opdag.init_binder import Binding, bind_init

# ── 权威快照门控(与 tests/test_opdag_fn_saves.py 同口径)────────────────────────────────
_MD5 = {
    "csa.py": "81673be3ad3cdd2191e28dd000a13f0e",
    "indexer.py": "8e18fee21c33507dd629c89f401986e6",
    "compressor.py": "49ec62ddc358f913438cf1c14cf99f7c",
}
_REL = "pynative/transformers/experimental_attention_variant"


def _authoritative_pkg():
    """返回权威 mindformers **包目录**(md5 全中)或 None。"""
    for root in (os.environ.get("MINDFORMERS_ROOT"),
                 r"E:\97-codes\torch_parallel\mf-src-167"):
        if not root:
            continue
        pkg = os.path.join(root, "mindformers")
        d = os.path.join(pkg, *_REL.split("/"))
        if not os.path.isdir(d):
            continue
        ok = True
        for fn, want in _MD5.items():
            p = os.path.join(d, fn)
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


# ── 真源抽取配置(全部取自 dsv4h_fused_pp4_recomp.yaml 或源码逐字)────────────────────────
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
        # `__init__` 派生量:**显式注入**。walker 刻意**不**静态求值 `__init__` 的布尔 ——
        # 反例:`compress_ratio` 的 `__init__` 缺省是 0(csa.py:556),静态求值会得
        # `enable_compress=False`(csa.py:594),与真机(ratio=4)相反 = **静默错**。
        "compress_ratio": ratio, "enable_compress": True, "enable_indexer": ratio == 4,
        "is_tnd": False, "window_size": 128, "training": True,
        "apply_dsa_kernel_fusion": fused,
        "rotate": True,                                   # compressor.py:82/91/129
        "overlap": ratio == 4, "coff": 1 + int(ratio == 4),   # compressor.py:89-90
        "sparse_loss": True, "use_butterfly": False,
        "seq_length": 4096, "micro_batch_size": 1,        # yaml
        "dsa_indexer_loss_coeff": 0.001,                  # yaml:118
    }


# 部署形态谓词:真机 pynative 栈把 Parameter 包成 hyper_parallel DTensor → `to_local` 存在。
# 两支对字节等价,但仍**显式**给值(walker 缺键即 fail-loud,绝不默认取某一支)。
RUNTIME_PREDICATES = {"hasattr:to_local": True, "hasattr:detach": True,
                      "isinstance:DTensor": True}
# 纯宿主副作用调用(逐层 indexer loss 记进模块级 dict,utils.py:41-54):无被消费的返回值。
HOST_ALLOW = ("save_to_indexer_losses_tracker", "get_indexer_loss_tracker")
KERNEL_ALLOW = ("npu_lightning_indexer",)
# 形参轴种子(与 `infer_shapes(dag, input_shapes)` 同契约;逐字来自源侧 docstring)。
INPUT_AXES = {
    "x": ("seq_length", "micro_batch_size", "hidden_size"),          # compressor.py:179
    "qr": ("seq_length", "micro_batch_size", "q_lora_rank"),         # indexer.py:163
    "query": ("seq_length", "micro_batch_size", "num_attention_heads", "v_head_dim"),
    "key": ("seq_length", "micro_batch_size", 1, "v_head_dim"),      # csa.py:648-649
}
# 单抽子 Cell 时父本会传入的构造注入项(csa.py:602/614、indexer.py:131、deepseek_v4:88)。
INJECTED = {"rotary_pos_emb": Binding("Constant", {"rope_freqs": True})}


def _extract(mf_pkg, cls, rel, spec, fused, ratio=4, strict=False):
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import ResolvedSpec
    bare = {n: ResolvedSpec(cell=n, submodules={}) for n in ("UnfusedCSAIndexerLoss", "Hadamard")}
    return extract_cell(
        mf_pkg, rel, cls, spec, cell_flags(fused, ratio),
        present_params={"rotary_pos_emb"}, recurse=True, subcell_specs=bare,
        cross_file=True, runtime_predicates=RUNTIME_PREDICATES, host_call_allow=HOST_ALLOW,
        kernel_call_allow=KERNEL_ALLOW, input_axes=INPUT_AXES, injected_binds=INJECTED,
        strict=strict,
    )


@pytest.fixture(scope="module")
def targets(mf_pkg):
    from cost_eval.opdag.module_resolver import PYNATIVE_SPEC_FILES, resolve_layer_spec
    top = resolve_layer_spec(mf_pkg, SPEC_FLAGS, spec_files=PYNATIVE_SPEC_FILES)
    sa = top.submodules["self_attention"]
    csa = sa.submodules["core_attention"]
    # **断言解出的是我要的那个类**(`_SPEC_FILES` 曾静默解成 DSv3 `MLASelfAttentionConcatenated`,
    # 评估文档 §2.1;`531bcdc` 加了 `spec_files=` 参数化)。
    assert top.cell == "HyperConnectionTransformerLayer"
    assert sa.cell == "DSv4HybridSelfAttention"
    assert csa.cell == "CompressedSparseAttention"
    return {
        "Compressor": (f"{_REL}/compressor.py", csa.submodules["compressor"]),
        "CSAIndexer": (f"{_REL}/indexer.py", csa.submodules["indexer"]),
        "CompressedSparseAttention": (f"{_REL}/csa.py", csa),
        "DSv4HybridSelfAttention": (f"{_REL}/deepseek_v4_hybrid_attention.py", sa),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. 裸函数别名绑定(P0#2)
# ══════════════════════════════════════════════════════════════════════════════
def test_primitive_table_is_an_explicit_reviewable_table():
    """表必须是**显式表**(不是启发式),且每个 op 类型都在 `bprop_rules.PIN` 里有表项。

    后半条是硬约束:`derive_saves` 对未知 op 类型 fail-loud(`bprop_rules.py:49`),
    所以「表里能绑出来的 op 类型」必须与「PIN 覆盖的 op 类型」闭合 —— 否则抽通了也算不出字节。
    """
    ops = {op for op, _ in prims.PRIMITIVES.values()} - {prims.SHAPE_OF}
    missing = sorted(o for o in ops if o not in PIN)
    assert missing == [], f"primitives 能发射但 PIN 没覆盖的 op 类型: {missing}"


def test_unknown_primitive_is_fail_loud_not_guessed():
    """未知原语 **fail-loud**,绝不"猜个类别"——归错类 = saved 集静默错(本项目要杀的那类 bug)。"""
    with pytest.raises(prims.UnknownPrimitiveError) as ei:
        prims.lookup_alias("mint.definitely_not_a_real_op", attr="foo", src="x.py:1")
    msg = str(ei.value)
    assert "primitives" in msg and "mint.definitely_not_a_real_op" in msg
    assert prims.lookup("mint.definitely_not_a_real_op") is None


def test_bare_alias_binding_covers_every_alias_on_the_dsv4_chain(mf_pkg):
    """真源验收:dsv4 链上 6 个类的**全部**裸别名要么绑上,要么在"该加表项"清单里。

    评估文档 §2.4 实测该链上 **39 个原语 / 196 个调用点**一个都绑不上(`init_binder` 只认
    类实例化)。本测试逐类核对:被 `construct`/helper **调用到**的别名必须**全部**绑上。
    """
    import ast
    from cost_eval.opdag.init_binder import unbound_aliases
    units = [
        (f"{_REL}/compressor.py", "Compressor"),
        (f"{_REL}/indexer.py", "CSAIndexer"),
        (f"{_REL}/indexer.py", "UnfusedCSAIndexerLoss"),
        (f"{_REL}/csa.py", "CompressedSparseAttention"),
        (f"{_REL}/deepseek_v4_hybrid_attention.py", "DSv4HybridSelfAttention"),
        ("pynative/transformers/multi_latent_attention.py", "MultiLatentAttention"),
    ]
    total_bound, called_unbound = 0, []
    for rel, cls in units:
        src = open(os.path.join(mf_pkg, *rel.split("/")), encoding="utf-8").read()
        binds = bind_init(src, cls)
        total_bound += len(binds)
        tree = ast.parse(src)
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
        called = {c.func.attr for c in ast.walk(node) if isinstance(c, ast.Call)
                  and isinstance(c.func, ast.Attribute)
                  and isinstance(c.func.value, ast.Name) and c.func.value.id == "self"}
        unknown = {a["attr"] for a in unbound_aliases(src, cls)}
        called_unbound += [f"{cls}.{a}" for a in sorted(called & unknown)]
    assert called_unbound == [], f"被调用却仍未绑的裸别名: {called_unbound}"
    assert total_bound >= 40, total_bound       # 实测 ≥40 个 self.<attr> 绑定


# ══════════════════════════════════════════════════════════════════════════════
# 2. 跨文件 MRO(P0#3)
# ══════════════════════════════════════════════════════════════════════════════
def test_cross_file_mro_resolves_the_base_class(mf_pkg):
    """`DSv4HybridSelfAttention(MultiLatentAttention)` 的基类在**另一个文件**里。"""
    from cost_eval.opdag.module_index import ClassIndex
    rel = f"{_REL}/deepseek_v4_hybrid_attention.py"
    idx = ClassIndex(mf_pkg)
    mro = idx.mro("DSv4HybridSelfAttention", rel)
    names = [rc.name for rc in mro]
    assert names[0] == "DSv4HybridSelfAttention"
    assert "MultiLatentAttention" in names
    base = next(rc for rc in mro if rc.name == "MultiLatentAttention")
    assert base.rel == "pynative/transformers/multi_latent_attention.py"
    assert base.rel != rel                     # **确实**跨了文件


def test_base_class_aliases_reach_the_derived_construct(mf_pkg):
    """基类 `__init__` 的 `self.shape/reshape/cast/permute/transpose`
    (`multi_latent_attention.py:127-131`)必须绑到派生类的 construct 上。

    此前实测:`self.shape(...)` 未绑定 @ `deepseek_v4_hybrid_attention.py:233` → fail-loud。
    """
    src = open(os.path.join(mf_pkg, "pynative", "transformers",
                            "multi_latent_attention.py"), encoding="utf-8").read()
    binds = bind_init(src, "MultiLatentAttention")
    assert binds["shape"].op == prims.SHAPE_OF     # `ops.shape` 产标量元组,**不发节点**
    assert binds["reshape"].op == "View"
    assert binds["cast"].op == "Cast"
    assert binds["permute"].op == "View"
    assert binds["transpose"].op == "View"


def test_shape_call_does_not_fabricate_a_tensor_node(mf_pkg, targets):
    """`sq, bsz, _ = self.shape(x)`(`deepseek_v4_hybrid_attention.py:233`)产出的是 python
    int 元组 —— **绝不发射节点**(发了就是造假节点),只记轴解包 + 标量身份。"""
    rel, spec = targets["DSv4HybridSelfAttention"]
    dag = _extract(mf_pkg, "DSv4HybridSelfAttention", rel, spec, fused=True)
    assert not [n for n in dag.nodes if n.src.endswith("deepseek_v4_hybrid_attention.py:233")]
    assert any(sb["names"][:2] == ["sq", "bsz"] for sb in dag.scalar_binds)


# ══════════════════════════════════════════════════════════════════════════════
# 3. `ast.With` 的块级语义(P0#4)
# ══════════════════════════════════════════════════════════════════════════════
def test_no_grad_block_products_are_detached_not_fabricated(mf_pkg, targets):
    """`with _no_grad():`(`indexer.py:214`,区域 `:214-232`)—— 块体走查 + 块内产物全 detached。

    `7b9aa86` 刻意**不**走查它的理由是对的(发普通节点 = 造出一批"看起来梯度可达"的假节点);
    正解是走查 + 带块级语义。此处验收后者。
    """
    rel, spec = targets["CSAIndexer"]
    dag = _extract(mf_pkg, "CSAIndexer", rel, spec, fused=True)
    assert dag.nodes
    for n in dag.nodes:
        line = int(n.src.split(":")[1])
        assert 214 <= line <= 232, n.src                    # 全在 `_no_grad` 区里
        assert n.attrs.get("detached") is True, n.src
    assert {"q", "k", "weights", "topk_indices", "index_scores"} <= set(dag.detached)
    assert diagnostics_summary(dag)["total"] == 0


def test_transparent_with_is_walked_without_detach_marks():
    """`SkipDTensorDispatch`(只切换派发模式,数学恒等)→ 照常走查,**不**打 detach 标。"""
    src = '''
class C:
    def construct(self, x):
        with SkipDTensorDispatch():
            y = self.mm(x, x)
        return y
'''
    dag = walk_construct(src, "C", {"mm": Binding("MatMul", {})}, "c.py", config_flags={})
    assert [n.op for n in dag.nodes] == ["MatMul"]
    assert dag.nodes[0].attrs.get("detached") is None
    assert dag.detached == []


# ══════════════════════════════════════════════════════════════════════════════
# 4. 赋值 RHS 形态(P0#5)—— 这次要修的张量恰好全在这里
# ══════════════════════════════════════════════════════════════════════════════
def test_the_five_named_tensors_now_have_nodes(mf_pkg, targets):
    """评估文档 §10 P0#5 的验收点,逐个定位符核对(**这五个张量在图里必须有节点**):

      * `deepseek_v4_hybrid_attention.py:245` `q = q * rsqrt(mean(q*q,...) + eps)` → q_hnorm_fp32
      * `indexer.py:350` `attention_scores = matmul(q,k) * softmax_scale` → O(S·S/r) fp32
      * `compressor.py:209` `score_f32 = score.astype(fp32) + reshape(ape,...)` → fp32 softmax 输入
      * `csa.py:779` `future = cm >= positions // ratio`               → O(S·S/r) bool mask
      * `csa.py:810` `valid = topk_indices_compressed < unsqueeze(...)` → O(S·topk) bool mask
    """
    dsv4 = _extract(mf_pkg, "DSv4HybridSelfAttention", *targets["DSv4HybridSelfAttention"],
                    fused=False)
    srcs = {n.src for n in dsv4.nodes}
    for loc in ("deepseek_v4_hybrid_attention.py:245", "indexer.py:350",
                "compressor.py:209", "csa.py:779", "csa.py:810"):
        assert loc in srcs, f"{loc} 没有节点"
    # 两个 mask 必须有 bool 产出的 `Compare` 节点(不可微 → 反向什么都不读)。
    # 注意同一行可能有多个节点:`future = cm >= (positions // ratio)`(csa.py:779)里
    # `positions // ratio` 本身是张量运算,先被物化成一个 Elementwise,再由 Compare 消费。
    for loc in ("csa.py:779", "csa.py:810"):
        cmps = [n for n in dsv4.nodes if n.src == loc and n.op == "Compare"]
        assert cmps, (loc, [n.op for n in dsv4.nodes if n.src == loc])
        assert all(n.out.endswith(":bool") for n in cmps), [n.out for n in cmps]
    # `score_f32` 必须落 fp32(源里 `score.astype(mstype.float32) + ape`)
    sf = next(n for n in dsv4.nodes if n.src == "compressor.py:209")
    assert sf.out.endswith(":fp32"), sf.out
    # `attention_scores = self.matmul(query, key) * self.softmax_scale`(indexer.py:350)拆成两节点:
    #   MatMul(存两操作数)+ Elementwise(tensor × **标量** → 梯度线性 → **不存激活**)。
    # 后半条是字节要害:若把 `* scale` 误判成非线性,那个 O(S·S/r) fp32 张量就会被错声明为 saved。
    ats = [n for n in dsv4.nodes if n.src == "indexer.py:350"]
    assert [n.op for n in ats] == ["MatMul", "Elementwise"], [n.op for n in ats]
    assert ats[1].attrs["linear"] is True


def test_scalar_and_dtype_bookkeeping_does_not_fabricate_tensors(mf_pkg, targets):
    """反向条:**标量 / dtype 记账**的行绝不建节点(建了就是造假张量 → 字节多算)。

      * `deepseek_v4_hybrid_attention.py:232` `ori_dtype = x.dtype`(dtype 记号)
      * `compressor.py:188` `ratio = self.compress_ratio`(config 标量)
      * `compressor.py:196` `cutoff = (sq // ratio) * ratio`(纯标量算术)
      * `csa.py:466` `sq, b, n, d = query.shape`(轴长解包)
    """
    dag = _extract(mf_pkg, "Compressor", *targets["Compressor"], fused=True)
    lines = {int(n.src.split(":")[1]) for n in dag.nodes}
    for scalar_line in (187, 188, 196, 201):
        assert scalar_line not in lines, scalar_line
    dsv4 = _extract(mf_pkg, "DSv4HybridSelfAttention", *targets["DSv4HybridSelfAttention"],
                    fused=True)
    assert not [n for n in dsv4.nodes
                if n.src == "deepseek_v4_hybrid_attention.py:232"]


def test_weights_never_enter_ins_or_saves(mf_pkg, targets):
    """契约 W2/W3:`Parameter` 操作数**不进 `ins`**、**不是任何节点的 out** → 永不进 saves。

    实测反例(评估文档 §7.2):FFNGroupedGEMM「236 MiB」里 88 MiB 是 `w1`/`w2` 被当激活 save 计。
    dsv4 链上的 Parameter:`attn_sink`(csa.py:589)、`ape`(compressor.py:117)、
    `linear_o_group_proj`/`q_rms_gamma`(deepseek_v4:139/159)。
    """
    dag = _extract(mf_pkg, "DSv4HybridSelfAttention", *targets["DSv4HybridSelfAttention"],
                   fused=False)
    params = {r["param"] for r in dag.param_operands}
    assert {"attn_sink", "ape", "linear_o_group_proj"} <= params, params
    save_names = {s.name for s in derive_saves(dag)}
    assert not (params & save_names), params & save_names          # W2
    outs = {n.out.split(":")[0] for n in dag.nodes if n.out}
    assert not (params & outs), params & outs                     # W3


# ══════════════════════════════════════════════════════════════════════════════
# 5. `ops.stop_gradient` → 真 `Detach` 节点(P1#11)
# ══════════════════════════════════════════════════════════════════════════════
def test_stop_gradient_becomes_a_detach_node_preserving_the_edge():
    """评估文档 §5 的三条结论逐条反转:**建节点** + **保边** + **打标**。

    此前:落 `opaque_calls`(无节点、无标记),且**赋值目标从未进 SSA → 边被静默切断**
    (下游拿到占位 ref `xd:?:bf16`,与合法形参操作数长得一模一样)。
    """
    src = '''
class C:
    def construct(self, x, y):
        h = self.mm(x, x)
        xd = ops.stop_gradient(h)
        return self.mul(xd, y)
'''
    binds = {"mm": Binding("MatMul", {}), "mul": Binding("Elementwise", {"linear": False})}
    dag = walk_construct(src, "C", binds, "c.py", config_flags={})
    assert [n.op for n in dag.nodes] == ["MatMul", "Detach", "Elementwise"]
    mm, det, mul = dag.nodes
    assert det.src == "c.py:5" and det.attrs["detached"] is True
    assert det.ins == ["h:?:bf16"] and det.out == "xd:?:bf16"
    assert [mm.id, det.id] in dag.edges            # **保边**:producer → Detach
    assert [det.id, mul.id] in dag.edges           # **保边**:Detach → consumer
    assert dag.detached == ["xd"]
    assert dag.opaque_calls == []                   # 不再只是"看得见文本"
    assert PIN["Detach"] == {"inputs": []}          # 梯度到此为止,反向不读


def test_stop_gradient_inline_argument_form_also_builds_a_node():
    """**内联实参形**(无赋名):`f(..., ops.stop_gradient(q), ...)` —— `csa.py:794/795`
    正是这一形态(`ukl1`/`ukl2` 判决的依据)。此前连目标名都没有,边必然丢。"""
    src = '''
class C:
    def construct(self, x, y):
        h = self.mm(x, x)
        return self.mul(ops.stop_gradient(h), y)
'''
    binds = {"mm": Binding("MatMul", {}), "mul": Binding("Elementwise", {"linear": False})}
    dag = walk_construct(src, "C", binds, "c.py", config_flags={})
    assert [n.op for n in dag.nodes] == ["MatMul", "Detach", "Elementwise"]
    mm, det, mul = dag.nodes
    assert det.ins == ["h:?:bf16"]
    assert [mm.id, det.id] in dag.edges and [det.id, mul.id] in dag.edges
    assert det.out.split(":")[0] in dag.detached


def test_all_six_real_stop_gradient_sites_become_detach_nodes(mf_pkg, targets):
    """真源 6 处 `ops.stop_gradient`(全在 `csa.py`):fused 支 `:665/:666`、
    unfused 支 `:764/:765` + 内联实参 `:794/:795`(`fn_saves` 已独立核过这份名单)。"""
    fused = _extract(mf_pkg, "CompressedSparseAttention", *targets["CompressedSparseAttention"],
                     fused=True)
    fused_det = {n.src for n in fused.nodes if n.op == "Detach"}
    assert fused_det == {"csa.py:665", "csa.py:666"}, fused_det
    unfused = _extract(mf_pkg, "CompressedSparseAttention",
                       *targets["CompressedSparseAttention"], fused=False)
    unfused_det = {n.src for n in unfused.nodes if n.op == "Detach"}
    assert unfused_det == {"csa.py:764", "csa.py:765", "csa.py:794", "csa.py:795"}, unfused_det
    # 每个 Detach 的**数据流都没断**:操作数 ref 在(不是凭空占位),且产出名进 detached 名册。
    # 入边只在被 detach 的张量**在本图内有 producer** 时才存在 —— `csa.py:665/666` detach 的是
    # construct 形参 `x`/`qr`(图的输入边界,天然无 producer),此时"保边"体现为 ins 里有真 ref。
    for dag in (fused, unfused):
        for n in [n for n in dag.nodes if n.op == "Detach"]:
            assert n.ins, n.src
            operand = n.ins[0].split(":")[0]
            has_producer = any(m.out.split(":")[0] == operand for m in dag.nodes if m.id < n.id)
            if has_producer:
                assert any(dst == n.id for _src, dst in dag.edges), n.src
            assert n.out.split(":")[0] in dag.detached, n.src


def test_detached_concept_aligns_with_model_spec_tensorref():
    """与 `cost_eval/model_spec.py` 的 `TensorRef.detached`(已被 `liveness/graph.py` 消费)
    是**同一个概念**:"这张张量的梯度到此为止"。walker 侧只给 detach **边界事实**,
    不做传递闭包 —— 闭包要看链上有没有参数(有参数仍需 save,如 `index_scores` 经
    `linear_wq_b` 携梯度;无参数则不需要,如 `ukl1/ukl2`),那属 P1#14。"""
    import dataclasses
    from cost_eval.model_spec import TensorRef
    fields = {f.name for f in dataclasses.fields(TensorRef)}
    assert "detached" in fields                       # 字段已存在,**本轮未改 model_spec**
    from cost_eval.opdag.schema import OpDAG
    assert "detached" in {f.name for f in dataclasses.fields(OpDAG)}


# ══════════════════════════════════════════════════════════════════════════════
# 6. 验收:四个 Cell × 两条 A/B 配置,零诊断 + strict 干净
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
@pytest.mark.parametrize("cls", ["Compressor", "CSAIndexer", "CompressedSparseAttention",
                                 "DSv4HybridSelfAttention"])
def test_all_four_dsv4_cells_extract_clean(mf_pkg, targets, cls, fused):
    """四个 dsv4_hybrid Cell 在 fused / unfused 两条配置下都抽出**非空图 + 零诊断**。

    残留 opaque 允许清单(逐条给理由,均字节中性):
      * `save_to_indexer_losses_tracker(...)`(`csa.py:799`)—— 纯宿主副作用:把逐层 indexer
        loss(标量)累加进模块级 dict(`utils.py:41-54`),无被消费的返回值、不产张量。
    """
    rel, spec = targets[cls]
    dag = _extract(mf_pkg, cls, rel, spec, fused=fused)
    assert dag.nodes, f"{cls}/{fused} 抽出空图"
    assert diagnostics_summary(dag)["total"] == 0, dag.diagnostics
    assert_extraction_clean(dag)
    for c in dag.opaque_calls:
        assert c.get("kind") == "host_side_effect", c
        assert "tracker" in c["expr"], c


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
@pytest.mark.parametrize("cls", ["Compressor", "CSAIndexer", "CompressedSparseAttention",
                                 "DSv4HybridSelfAttention"])
def test_strict_true_is_clean_on_the_dsv4_path(mf_pkg, targets, cls, fused):
    """`7b9aa86` 立的那道门:`strict=True` 在 dsv4 路径上**放行**(全局默认仍可为 False)。"""
    rel, spec = targets[cls]
    dag = _extract(mf_pkg, cls, rel, spec, fused=fused, strict=True)
    assert dag.nodes


def test_derive_saves_does_not_fail_loud_on_the_extracted_graphs(mf_pkg, targets):
    """`derive_saves` 对未知 op 类型 fail-loud → 抽出的图里每个 op 类型都必须有 PIN 表项。
    融合内核 / `_Function.apply` 的 saved 集**必须来自源**(缺 attrs 即 fail-loud)。"""
    for fused in (True, False):
        for cls in targets:
            dag = _extract(mf_pkg, cls, *targets[cls], fused=fused)
            saves = derive_saves(dag)        # 不抛即通过
            assert isinstance(saves, list)


def test_unfused_csa_chain_is_a_real_op_chain_not_one_op(mf_pkg, targets):
    """**最有价值的结构性产出**:手写 spec 把 `unfused_compressed_sparse_attn`
    (`csa.py:464-533`)塌成**一个** op + 一张扁平 saves 名单;源侧它是一条真算子链。

    此前它是模块级**自由函数**(非 `self.<method>`)→ `_handle_call` 终端 fallthrough →
    只进 `opaque_calls`,内部 ~24 个 mint/ops 调用全不可见。
    """
    dag = _extract(mf_pkg, "CompressedSparseAttention",
                   *targets["CompressedSparseAttention"], fused=False)
    chain = [n for n in dag.nodes
             if n.src.startswith("csa.py:") and 464 <= int(n.src.split(":")[1]) <= 533]
    assert len(chain) >= 40, f"只抽出 {len(chain)} 个节点,不像一条链"
    ops = [n.op for n in chain]
    # 链上的关键结构必须在:两次 BMM(scores / 加权和)、gather、softmax 手写三件套、cast
    assert ops.count("BMM") == 2, ops
    assert "IndexSelect" in ops                       # csa.py:485 `kv_flat[flat_indices]`
    assert "Compare" in ops                           # csa.py:501 invalid mask
    assert ops.count("Cast") >= 4
    # 边是连通的:链上每个非首节点都至少有一条入边或全标量实参
    ids = {n.id for n in chain}
    assert sum(1 for s, d in dag.edges if s in ids and d in ids) >= len(chain) - 5
