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
