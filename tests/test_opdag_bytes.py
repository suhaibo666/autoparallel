"""让抽出的 dsv4 图产出**真字节** —— shape/dtype 解析 + 权重/激活分离 + 并行度本地化。

背景：`docs/opdag_walker_core_2026-07-25.md` §7 第 1 项——walker 已能干净抽出 dsv4 四个 Cell，
但**每个 shape 段仍是 `?`**，`total_bytes = 0`。本文件钉住把它变成真字节的每一条机制，
并且钉住"解不出就显式记账、绝不编造尺寸"这条纪律。

三段：
  1. **合成源单测**（不依赖 mindformers 快照，永远跑）——符号代数 / `__init__` 维度求值 /
     dtype / 并行度整除 / 权重-激活分离的判据；
  2. **权威快照集成**（md5 门控，见 `tests/test_opdag_fn_saves.authoritative_variant_dir`）——
     真 dsv4 四 Cell 的逐节点字节；
  3. **反向不变量**：解不出的一律进 `unresolved` 且带 `file:line`，任何"看起来合理的数"都算失败。
"""
import os

import pytest

from cost_eval.model_spec import DimTable
from cost_eval.opdag import consumer as C
from cost_eval.opdag import shape_infer as SI
from cost_eval.opdag import sym_shape as SS
from cost_eval.opdag.init_dims import eval_init_dims, INIT_PARAM_SEEDS
from cost_eval.parallel_model import ParallelModel
from cost_eval.specs import ParallelConfig

import ast

MiB = 1024 * 1024

# ── 站点配置（seq4096 / topk512 / 64 heads / v_head_dim 512 / b=1）——与
#    `tests/test_source_truth_saves_audit.SITE_DIMS` 逐字段相同，故手写侧/抽取侧可对账。
SITE_DIMS = DimTable(
    H=4096, F=4096, n_heads=64, n_kv=1, head_dim=128, S=4096, B=1, vocab=129280,
    n_layers=8, q_lora_rank=1024, kv_lora_rank=512,
    qk_rope_head_dim=64, qk_nope_head_dim=0, v_head_dim=512,
    dsa_indexer_n_heads=64, dsa_indexer_head_dim=128, dsa_indexer_topk=512,
    o_groups=8, o_lora_rank=1024, csa_window_size=128,
)


# ===========================================================================
# 1. 符号代数：`//`（压缩序列 S//ratio）
# ===========================================================================

def test_floordiv_keeps_exact_when_coeff_divides():
    """系数整除时仍走原路径（`2·v_head_dim // 2 = v_head_dim`）—— 既有行为不变。"""
    f = SS.parse_axis("2·v_head_dim")
    assert SS.render_term(SS.floordiv(f, 2)) == "v_head_dim"


def test_floordiv_forms_atomic_unit_when_not_divisible():
    """`S // 4`：系数不整除 → 形成**原子单元** `S//4`（与和式单元同一套「保持原子」设计）。

    动机（源）：`compressor.py:201` `n_compressed = cutoff // ratio`，
    `csa.py:762` 的压缩 KV 序列长度 = `S // compress_ratio`。此前 `floordiv` 直接返回
    None → 整条压缩链 shape 全 `?`。
    """
    f = SS.floordiv(SS.parse_axis("S"), 4)
    assert f is not None
    assert SS.render_term(f) == "S//4"
    # 再乘回 ratio：`(S//4)·4`（cutoff = (sq//ratio)*ratio, compressor.py:196）
    back = SS.mul(f, SS.Factors(coeff=4))
    assert SS.render_term(back) == "4·(S//4)"
    # 再整除 ratio 回到 S//4（n_compressed = cutoff // ratio, compressor.py:201）
    again = SS.floordiv(back, 4)
    assert SS.render_term(again) == "S//4"


def test_floordiv_atom_round_trips_through_parse():
    f = SS.parse_axis("S//4")
    assert list(f.syms) == ["S//4"] and f.coeff == 1


def test_consumer_values_floordiv_atom():
    """`S//4` 在 DimTable 下取到 4096//4 = 1024（整数除法，不是浮点）。"""
    assert C.resolve_shape_elems("S//4·B", SITE_DIMS) == (4096 // 4) * 1


def test_consumer_floordiv_atom_of_unknown_symbol_is_unresolved():
    assert C.resolve_shape_elems("frobnicate//4", SITE_DIMS) is None


# ===========================================================================
# 2. dsv4 符号 → DimTable 字段（评估文档 §4.1 第 2 条：缺一个就一律落 unresolved）
# ===========================================================================

DSV4_SYMS = {
    "index_n_heads": "dsa_indexer_n_heads",
    "index_head_dim": "dsa_indexer_head_dim",
    "index_topk": "dsa_indexer_topk",
    "csa_window_size": "csa_window_size",
    "o_groups": "o_groups",
    "o_lora_rank": "o_lora_rank",
}


@pytest.mark.parametrize("sym,field", sorted(DSV4_SYMS.items()))
def test_dsv4_symbol_maps_to_dimtable_field(sym, field):
    assert C._SYM2FIELD[sym] == field
    assert C.resolve_shape_elems(sym, SITE_DIMS) == getattr(SITE_DIMS, field)


@pytest.mark.parametrize("sym", sorted(DSV4_SYMS))
def test_dsv4_symbol_is_also_a_config2sym_dim(sym):
    """CONFIG2SYM 决定 `init_dims` 是否把 `self.<attr>` 留进 `dims_ctx`
    （`init_dims._KNOWN_DIM_SYMS = set(CONFIG2SYM.values())`）→ 两张表必须同步。"""
    assert sym in SS.CONFIG2SYM.values()


@pytest.mark.parametrize("cfg_attr,sym", [
    ("dsa_indexer_n_heads", "index_n_heads"),      # indexer.py:94
    ("dsa_indexer_head_dim", "index_head_dim"),    # indexer.py:95
    ("dsa_indexer_topk", "index_topk"),            # indexer.py:96
    ("csa_window_size", "csa_window_size"),        # csa.py:573
    ("o_groups", "o_groups"),                      # deepseek_v4_hybrid_attention.py:136
    ("o_lora_rank", "o_lora_rank"),                # deepseek_v4_hybrid_attention.py:137
])
def test_config2sym_is_keyed_by_the_config_attribute_name(cfg_attr, sym):
    """CONFIG2SYM 的**键是 `config.<attr>` 的属性名**（`_attr_dim`/`_eval_dim` 都按它查）。

    dsv4 侧 config 属性名与源码里的 self 名**不同名**（`self.index_n_heads =
    config.dsa_indexer_n_heads`，indexer.py:94）→ 若按 self 名建表，`config.dsa_indexer_n_heads`
    会落到"保留原名"分支、`init_dims._KNOWN_DIM_SYMS` 再把它滤掉 → dims_ctx 里根本没有它。
    """
    assert SS.CONFIG2SYM[cfg_attr] == sym


def test_unmapped_symbol_still_unresolved_not_zero():
    """未映射的符号**不得**悄悄变成 0/1 —— 必须 None（调用方据此进 unresolved）。"""
    assert C.resolve_shape_elems("S·B·no_such_dim", SITE_DIMS) is None


# ===========================================================================
# 3. `__init__` 维度求值：pynative 侧 build_module 用**关键字** input_size/output_size
# ===========================================================================

_KW_BUILD = '''
class KwLinear:
    def __init__(self, config, submodules, head_dim, compress_ratio=0):
        self.config = config
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        proj_out_dim = self.coff * head_dim
        self.linear_wkv = build_module(
            submodules.linear_wkv,
            input_size=config.hidden_size,
            output_size=proj_out_dim,
            bias=False,
        )
'''


def test_keyword_build_module_dims_are_captured():
    """真源 pynative 侧一律 `build_module(sub.x, input_size=..., output_size=...)`
    （`compressor.py:97-104`、`deepseek_v4_hybrid_attention.py:93-96`、`indexer.py:117-119`）；
    `parallel_core/training_graph` 侧才是位序实参（`multi_latent_attention.py:634-643`）。
    此前 `init_dims` 只认位序 → pynative 侧全部 linear 的 out_dim 缺失 →
    `shape_infer._matmul` fail-loud。"""
    tree = ast.parse(_KW_BUILD)
    got = eval_init_dims(tree, "KwLinear", {
        "hidden_size": 4096,
        INIT_PARAM_SEEDS: {"head_dim": "v_head_dim", "compress_ratio": 4},
    })
    assert got.linear_dims["linear_wkv"] == ("H", "2·v_head_dim")


def test_positional_build_module_dims_unchanged():
    """位序形态（training_graph 侧）逐字不变 —— 回归护栏。"""
    src = '''
class Pos:
    def __init__(self, config, submodules):
        self.config = config
        self.linear_q_proj = build_module(
            submodules.linear_q_proj, config.hidden_size, config.q_lora_rank, bias=False)
'''
    got = eval_init_dims(ast.parse(src), "Pos", {})
    assert got.linear_dims["linear_q_proj"] == ("H", "q_lora_rank")


def test_init_param_seeds_are_required_not_guessed():
    """不给 `INIT_PARAM_SEEDS` → `head_dim` 判不出 → out_dim 记 None（**不**拿 `compress_ratio`
    的 `__init__` 缺省 0 去凑）。这正是 walker 文档 §2.1 拒绝"用 __init__ 缺省喂 flags"的同一条纪律。"""
    got = eval_init_dims(ast.parse(_KW_BUILD), "KwLinear", {"hidden_size": 4096})
    assert got.linear_dims["linear_wkv"] == ("H", None)


def test_parameter_shapes_are_captured_for_param_census():
    """`self.<attr> = Parameter(mint.empty((a, b)))` 的**形状**（B4 param census 要它）。

    真源：`compressor.py:117` `Parameter(mint.empty((compress_ratio, proj_out_dim)))`、
    `csa.py:589` `Parameter(mint.zeros(config.num_attention_heads, dtype=float32))`。
    """
    src = '''
class P:
    def __init__(self, config, head_dim, compress_ratio=0):
        self.config = config
        self.coff = 2
        proj_out_dim = self.coff * head_dim
        self.ape = Parameter(mint.empty((compress_ratio, proj_out_dim),
                                        dtype=config.params_dtype), name="ape")
        self.attn_sink = Parameter(mint.zeros(config.num_attention_heads,
                                              dtype=mstype.float32), name="attn_sink")
'''
    got = eval_init_dims(ast.parse(src), "P", {
        "params_dtype": "bf16",
        INIT_PARAM_SEEDS: {"head_dim": "v_head_dim", "compress_ratio": 4},
    })
    assert got.param_shapes["ape"] == ("4", "2·v_head_dim")
    assert got.param_shapes["attn_sink"] == ("n_heads",)
    # dtype 逐字来自源（`mstype.float32` → fp32；缺省用 params_dtype）
    assert got.param_dtypes["attn_sink"] == "fp32"
    assert got.param_dtypes["ape"] == "bf16"


# ===========================================================================
# 4. 并行度本地化（TP/EP/CP/cp_kv）—— 语义参照 `shape_eval.resolve_tensor`
# ===========================================================================

def _pm(**kw):
    pc = ParallelConfig(**{"dp_shard": 1, "tp": 1, "cp": 1, "ep": 1, "pp": 1, **kw})
    return ParallelModel(pc, n_layers=8, world_size=1, edge_pseudo=(0, 0))


def test_local_bytes_tp_divides_the_head_axis():
    """TP 切 head 维：`S·B·(n_heads·v_head_dim)` 的 n_heads 轴 ÷tp。"""
    got = C.local_shape_elems("S·B·(n_heads·v_head_dim)", SITE_DIMS, _pm(tp=4),
                              shard={"n_heads": "tp"})
    assert got == 4096 * 1 * (64 // 4 * 512)


def test_local_bytes_cp_divides_the_first_S_axis_only():
    """CP 只切**首个**含 S 的轴（query/token 维），其余含 S 的轴保持全量 ——
    与 `shape_eval.resolve_tensor` 的 `break` 语义逐字一致（shape_eval.py:108-114）。"""
    got = C.local_shape_elems("S·B·S", SITE_DIMS, _pm(cp=2))
    assert got == (4096 // 2) * 1 * 4096


def test_local_bytes_cp_kv_stays_full_S_under_colossal():
    """`cp_kv=True` 且 colossal → KV all-gather 到 full-S，**不** ÷cp（shape_eval.py:105-106）。"""
    pm = _pm(cp=2)
    assert C.local_shape_elems("S·B·kv_lora_rank", SITE_DIMS, pm, cp_kv=True) == \
        4096 * 1 * 512
    assert C.local_shape_elems("S·B·kv_lora_rank", SITE_DIMS, pm) == \
        (4096 // 2) * 1 * 512


def test_local_bytes_non_divisible_axis_is_fail_loud():
    """不整除**不许**悄悄取整 —— 与 `resolve_tensor` 同样 fail-loud（shape_eval.py:86-88）。"""
    with pytest.raises(ValueError, match="整除"):
        C.local_shape_elems("S·B·n_heads", SITE_DIMS, _pm(tp=7), shard={"n_heads": "tp"})



# ===========================================================================
# 5. shape 推断：**解不出的一律记账**（绝不 passthrough 顶替）
# ===========================================================================

def _dag(*nodes, **kw):
    from cost_eval.opdag.schema import OpDAG
    return OpDAG(cell="T", nodes=list(nodes), **kw)


def _node(nid, op, src, ins=(), out="", **attrs):
    from cost_eval.opdag.schema import OpNode
    return OpNode(id=nid, op=op, src=src, ins=list(ins), out=out, attrs=dict(attrs))


def test_reduce_without_dim_is_recorded_not_passthrough():
    """`.sum(dim=1)`（`compressor.py:216`）的 `dim` **没被 walker 记进 attrs** →
    被约掉的轴长未知 → 元素数不可知。此前 `_dispatch` 把它当 Elementwise 走 `_passthrough`，
    等于**把归约后的张量按归约前的大小计**（compressor 那处多算 2·ratio=8 倍）。"""
    n = _node(1, "Elementwise", "compressor.py:216", ["kv:S·B·H:fp32"], "pooled:?:fp32",
              reduce=True, linear=True, prim="<tensor>.sum")
    rep = []
    SI.infer_shapes(_dag(n), {"kv": "S·B·H"}, report=rep)
    assert n.out.split(":")[1] == "?"                     # 保 `?`，不顶一个数
    assert [r["reason"] for r in rep] == ["reduce_axis_unknown"]
    assert rep[0]["src"] == "compressor.py:216"           # 带 file:line


def test_concat_is_numel_exact_and_marked_structure_unknown():
    """`cat` 的**元素数 = 各输入元素数之和**（与轴无关 → 精确）；轴未记 → 标 `~`（只知 numel）。

    此前按 `concat_axis` 缺省 0 相加：`compressor.py:243` `cat([kv_nope, kv_pe], dim=-1)`
    两输入末轴不同，按轴 0 合并给 `2n·b·(d−64)`，真值是 `n·b·d` —— 静默算错。
    """
    n = _node(1, "View", "compressor.py:243",
              ["a:S·B·kv_lora_rank:bf16", "b:S·B·v_head_dim:bf16"], "out:?:bf16",
              view="concat", variadic=True, prim="mint.cat")
    SI.infer_shapes(_dag(n), {"a": "S·B·kv_lora_rank", "b": "S·B·v_head_dim"})
    shp = n.out.split(":")[1]
    assert shp.startswith(SS.NUMEL_ONLY)
    got = C.resolve_shape_elems(shp, SITE_DIMS)
    assert got == 4096 * 1 * 512 + 4096 * 1 * 512


def test_matmul_refuses_numel_only_input():
    """上游只知元素数时，MatMul **不许**把"末轴"当成整个乘积去换 —— 记账拒绝。"""
    cat = _node(1, "View", "f.py:1", ["a:S·B·H:bf16", "b:S·B·H:bf16"], "c:?:bf16",
                view="concat", prim="mint.cat")
    mm = _node(2, "MatMul", "f.py:2", ["c:?:bf16"], "y:?:bf16", out_dim="H", module="Linear")
    rep = []
    SI.infer_shapes(_dag(cat, mm), {"a": "S·B·H", "b": "S·B·H"}, report=rep)
    assert mm.out.split(":")[1] == "?"
    assert [r["reason"] for r in rep] == ["needs_axis_structure"]


def test_reshape_with_unresolvable_dim_falls_back_to_exact_numel():
    """reshape **恒不改元素数**（算子定义）→ 目标维解不出时退 `numel_only`，元素数仍精确。

    真源：`compressor.py:203` `reshape(kv, (n_compressed, ratio, b, -1))` —— `n_compressed`/
    `ratio` 是 construct 局部标量，walker **未导出**（docs §4 的 G5）。
    """
    n = _node(1, "View", "compressor.py:203", ["kv:S·B·(2·v_head_dim):bf16"], "kv2:?:bf16",
              view="reshape", prim="mint.reshape",
              reshape_dims=["n_compressed", "ratio", "b", "-1"])
    rep = []
    SI.infer_shapes(_dag(n), {"kv": "S·B·(2·v_head_dim)"}, report=rep)
    shp = n.out.split(":")[1]
    assert shp.startswith(SS.NUMEL_ONLY)
    assert C.resolve_shape_elems(shp, SITE_DIMS) == 4096 * 1 * 2 * 512
    assert [r["reason"] for r in rep] == ["reshape_unresolved"]


def test_chunk_uses_the_unpack_arity_from_source_not_a_guess():
    """`a, b = chunk(x, 2, dim=-1)`：份数由**元组解包元数**证得（`attrs['outs']`），
    元素数 = 总积/2；只在符号上能**精确**整除时才认。"""
    n = _node(1, "View", "compressor.py:169", ["t:S·B·(2·v_head_dim):bf16"], "p:?:bf16",
              view="chunk", prim="mint.chunk", outs=["p:?:bf16", "q:?:bf16"])
    SI.infer_shapes(_dag(n), {"t": "S·B·(2·v_head_dim)"})
    assert C.resolve_shape_elems(n.out.split(":")[1], SITE_DIMS) == 4096 * 1 * 512
    # 两个产出都回填（否则下游拿 `?`）
    assert all(o.split(":")[1] != "?" for o in n.attrs["outs"])


def test_chunk_with_non_divisible_total_is_recorded():
    n = _node(1, "View", "f.py:1", ["t:S·B·H:bf16"], "p:?:bf16",
              view="chunk", prim="mint.chunk", outs=["p:?:bf16", "q:?:bf16", "r:?:bf16"])
    rep = []
    SI.infer_shapes(_dag(n), {"t": "S·B·H"}, report=rep)
    assert n.out.split(":")[1] == "?"
    assert [r["reason"] for r in rep] == ["chunk_axis_unknown"]


def test_stale_same_name_shape_is_invalidated_on_failure():
    """同名重绑（`kv = self.reshape(kv, …)`）失败时，env 里**重绑前**的 shape 必须作废。

    否则下游拿着一个已不成立的形状继续算 —— 实测这是把 `Compressor` 的 concat 产物算成
    8·10⁶ MiB 的两个原因之一（另一个是 `sym_shape._sum_term` 的括号）。
    """
    prod = _node(1, "MatMul", "f.py:0", ["x:S·B·H:bf16"], "kv:?:bf16",
                 out_dim="v_head_dim", module="Linear")           # 本图先产出 kv
    bad = _node(2, "View", "f.py:1", ["kv:?:bf16"], "kv:?:bf16",
                view="slice", index="a,b")                        # 解不出的 slice → 重绑失败
    down = _node(3, "Cast", "f.py:2", ["kv:?:bf16"], "y:?:fp32")
    SI.infer_shapes(_dag(prod, bad, down), {"x": "S·B·H"})
    assert prod.out.split(":")[1] == "S·B·v_head_dim"
    assert down.out.split(":")[1] == "?", "下游不得拿到重绑前的旧 shape"


def test_caller_seeds_are_never_invalidated_by_a_failing_producer():
    """**调用方种子是外部权威事实**，不能被"产出它但自己解不出"的节点抹掉。

    真源实测：`ffn.py:146` `w1 = cast(self.weight1)` —— 权重不进 `ins`（W2/W3 的结构性保证），
    故该 `Cast` 的 ins 为空、解不出；若把种子 `w1` 一并作废，下游 `GroupedMatMul` 就丢了
    权重末轴（`tests/test_opdag_shape_infer.py::test_moe_grouped_gemm_operand_is_e_cap_h`
    实测会红）。
    """
    cast_w = _node(1, "Cast", "ffn.py:146", [], "w1:?:bf16")      # 权重不进 ins → 无输入
    gmm = _node(2, "GroupedMatMul", "ffn.py:178",
                ["d:E·cap·H:bf16", "w1:?:bf16"], "fc1:?:bf16")
    SI.infer_shapes(_dag(cast_w, gmm),
                    {"d": "E·cap·H", "w1": "(E·H)·(2·moe_ffn)"})
    assert gmm.out.split(":")[1] == "E·cap·(2·moe_ffn)"


def test_compare_output_dtype_is_bool_not_inherited():
    """比较产 **bool（1 字节）**；walker 的 `_emit` 按 ins 推 dtype → 记成 bf16 = **2×**。

    真源两个大 mask：`csa.py:779` `future = cm >= positions // ratio`（O(S·S/r)）、
    `csa.py:810` `valid = topk_… < unsqueeze(…)`（O(S·topk)）；它们会被 `mint.where` 的
    `PIN{"inputs":[0]}`（bprop_rules.py:55）当 cond 保留 → dtype 必须对。
    """
    cmp_ = _node(1, "Compare", "csa.py:779", ["cm:S·B·H:bf16"], "future:?:bf16",
                 compare=">=")
    where = _node(2, "Where", "csa.py:781",
                  ["future:?:bf16", "a:S·B·H:bf16", "b:S·B·H:bf16"], "w:?:bf16",
                  prim="mint.where")
    dag = _dag(cmp_, where)
    SI.infer_shapes(dag, {"cm": "S·B·H", "a": "S·B·H", "b": "S·B·H"})
    assert cmp_.out.split(":")[2] == "bool"
    r = C.dag_saved_bytes(dag, SITE_DIMS)
    per = {p[0]: p for p in r["per_save"]}
    assert per["future"][2] == "bool"
    assert per["future"][3] == 4096 * 1 * 4096 * 1          # ×1 字节，不是 ×2


# ===========================================================================
# 6. `Detach` 别名：不许双计
# ===========================================================================

def test_detach_product_aliases_its_input_and_is_not_double_counted():
    """`ops.stop_gradient(x)` 共享 `x` 的存储（`Detach` 的 PIN 也是"什么都不读"）→
    `x_detach` 与 `x` **不得各计一份**。真源 `csa.py:665/666`（fused 支）。"""
    det = _node(1, "Detach", "csa.py:665", ["x:S·B·H:bf16"], "x_detach:?:bf16",
                detach=True, detached=True, prim="ops.stop_gradient")
    mm1 = _node(2, "MatMul", "csa.py:670", ["x:S·B·H:bf16"], "y:?:bf16",
                out_dim="H", module="Linear")
    mm2 = _node(3, "MatMul", "csa.py:671", ["x_detach:?:bf16"], "z:?:bf16",
                out_dim="H", module="Linear")
    dag = _dag(det, mm1, mm2)
    SI.infer_shapes(dag, {"x": "S·B·H"})
    assert C.detach_aliases(dag) == {"x_detach": "x"}
    r = C.dag_local_saved_bytes(dag, SITE_DIMS, None)
    names = {p[0] for p in r["per_save"]}
    assert "x" in names and "x_detach" not in names
    assert [a[:2] for a in r["detach_aliased"]] == [("x_detach", "x")]
    assert r["total_bytes"] == 4096 * 1 * 4096 * 2          # 只算一份
    # 关掉去重则双计 —— 证明这条差异确实是本机制产生的
    r2 = C.dag_local_saved_bytes(dag, SITE_DIMS, None, dedup_detach_aliases=False)
    assert r2["total_bytes"] == 2 * r["total_bytes"]


def test_detach_alias_chain_collapses_to_the_root():
    d1 = _node(1, "Detach", "csa.py:764", ["x:S·B·H:bf16"], "a:?:bf16", detach=True)
    d2 = _node(2, "Detach", "csa.py:765", ["a:?:bf16"], "b:?:bf16", detach=True)
    dag = _dag(d1, d2)
    SI.infer_shapes(dag, {"x": "S·B·H"})
    assert C.detach_aliases(dag) == {"a": "x", "b": "x"}


# ===========================================================================
# 7. 权重 vs 激活（契约 W1..W6 + B4 param census）
# ===========================================================================

def test_weights_are_structurally_outside_saves():
    """walker 把 `Parameter` 操作数**排除在 `ins` 之外**（`schema.py:56-59`）→
    `derive_saves` 结构上**不可能**把权重当激活 save（契约 W2/W4 自动成立）。
    实测 `FFNGroupedGEMM` 那 88 MiB 的病在本路径上不存在。"""
    n = _node(1, "Elementwise", "compressor.py:209", ["s:S·B·H:fp32"], "y:?:fp32",
              linear=True, arith="add", param_operands=["ape"])
    dag = _dag(n, param_operands=[{"src": "compressor.py:209", "param": "ape"}])
    SI.infer_shapes(dag, {"s": "S·B·H"})
    saves = {p[0] for p in C.dag_saved_bytes(dag, SITE_DIMS)["per_save"]}
    assert "ape" not in saves
    assert {p[0] for p in dag.param_operands and [(x["param"],) for x in dag.param_operands]} \
        == {"ape"}


def test_param_bytes_come_from_the_source_parameter_shape():
    """B4 的 param census：形状/dtype 逐字来自 `Parameter(mint.zeros(...), dtype=...)`。"""
    from cost_eval.opdag.init_dims import InitDims
    dag = _dag(_node(1, "Cast", "csa.py:687", [], "s:?:fp32", param_operands=["attn_sink"]),
               param_operands=[{"src": "csa.py:687", "param": "attn_sink"}])
    idims = InitDims(param_shapes={"attn_sink": ("n_heads",)},
                     param_dtypes={"attn_sink": "fp32"})
    r = C.dag_param_bytes(dag, SITE_DIMS, idims)
    assert r["per_param"] == [("attn_sink", "n_heads", "fp32", 64 * 4)]
    assert r["unresolved"] == []


def test_param_without_a_source_shape_is_unresolved_not_guessed():
    dag = _dag(_node(1, "Cast", "csa.py:687", [], "s:?:fp32"),
               param_operands=[{"src": "csa.py:687", "param": "mystery"}])
    r = C.dag_param_bytes(dag, SITE_DIMS, None)
    assert r["total_bytes"] == 0
    assert [u[0] for u in r["unresolved"]] == ["mystery"]


# ===========================================================================
# 8. merge_dims_ctx（内联子 Cell 的 `self.<attr>`）
# ===========================================================================

def test_merge_dims_ctx_unions_and_fails_loud_on_conflict():
    assert SI.merge_dims_ctx({"a": "H"}, {"b": "S"}) == {"a": "H", "b": "S"}
    with pytest.raises(ValueError, match="fail-loud"):
        SI.merge_dims_ctx({"head_dim": "v_head_dim"}, {"head_dim": "index_head_dim"})


def test_local_bytes_equals_hand_path_resolve_tensor():
    """与手写路径**同一张量**逐字节相等（任务要求的等价校验；契约只查"是正数"）。"""
    from cost_eval.model_spec import TensorRef
    from cost_eval.shape_eval import resolve_tensor
    pm = _pm(tp=2, cp=2)
    hand = resolve_tensor(TensorRef("q", ("S", "B", "n_heads*v_head_dim"),
                                    shard={2: "tp"}), SITE_DIMS, pm)
    got = C.local_shape_elems("S·B·(n_heads·v_head_dim)", SITE_DIMS, pm,
                              shard={"n_heads·v_head_dim": "tp"})
    assert got == hand.local_numel
