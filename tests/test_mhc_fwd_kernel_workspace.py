"""**前向** mHC pre-sinkhorn kernel workspace —— 实测项的守卫门（2026-07-30）。

守的是**测量**，不是模型自洽性。167 真机 memory-tracker（MS2.10/CANN9.1/910B2，
`MS_ALLOC_CONF=memory_tracker:True`，run c 家族三个 seq）在 FWD 相位测到的**单 kernel
workspace 极大值 = 304.001 MiB**，归属为融合 mHC 的 `npu_mhc_pre_sinkhorn`
（`docs/kernel_workspace_2026-07-29.md` §4.1/§4.2/§4.3，输出签名逐张反推）。
本文件钉住这笔实测值在模型里的四件事：

1. **值 + S 无关性**逐字节穿过 `builder → mhc_wrap → ShapeEval → StructureMemory`；
2. **归属**只在两个 `*_hc_pre_sinkhorn` op 上（实测「同层窗口内出现两次」= attn_hc + ffn_hc），
   不摊在层上、不挪到别的 op；
3. **没测的分支留 0**：非融合 mHC（小算子链，根本不是这个 kernel）与无 mHC 的模型；
4. **相位不串**：它走 **fwd** 通道（`OpSpec.workspace`），不是 2026-07-29 那轮为反向新建的
   `bwd_workspace{,_ref}`；两条通道在同一层上必须各算各的。

以及那条已经真实发生过两次的失败模式：**OpSpec 重建口静默吞字段**
（`norm_kind` 丢过一次、`workspace_ref` 丢过一次，见 `cost_eval/layers/residual.py::_rebuild`
与 `docs/census_fix_residual_carrier_2026-07-29.md` §5b）。本项挂在
`mhc_wrap` 产出的**第一个 op** 上，而 `cost_eval/layers/head.py:285` 正是拿
`wrapped[0]` 做 `dataclasses.replace` 的那一处 —— 故 MTP 层单列一门。

来源：`docs/mhc_fwd_workspace_2026-07-30.md`、`docs/kernel_workspace_2026-07-29.md`。
"""
import dataclasses
import os
import sys
import warnings

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_TOOLS = os.path.join(_REPO, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
os.environ.setdefault("COST_EVAL_EXTRACTED_ALLOW_PARTIAL", "1")

import liveness_ab_validate as V                       # noqa: E402
import serve_explorer as S                             # noqa: E402
from cost_eval.build_llm import build_llm_spec         # noqa: E402
from cost_eval.parallel_model import ParallelModel     # noqa: E402
from cost_eval.shape_eval import ShapeEval             # noqa: E402
from cost_eval.structure_mem import estimate_structure_memory  # noqa: E402

MiB = 2 ** 20

#: 167 真机 memory-tracker 实测（`docs/kernel_workspace_2026-07-29.md` §4.1 逐相位表、
#: §4.2 逐（层，相位）窗口表、§4.3 归属原文）。**FWD 相位单 kernel workspace 极大值**，
#: 两个插桩 rank（stage0 = r0+r4、stage1 = r128+r4）同值，三个 seq 配置同值 → **S 无关**。
#: **测量值，不得为了让模型好看而编辑**；改它必须先有新的真机测量。
#:
#: ⚠ **它是 tracker 的三位小数原文**，不是精确 MiB 数（304.001 MiB 不是整字节）。故断言
#: 一律按「模型值渲染到**测量本身的精度**」比：`f"{MiB:.3f}" == "304.001"`。想比字节的
#: 见 `MODEL_BYTES` 那道门（那里写清了 观测 / 推断 的分界）。
REAL_MHC_PRE_SINKHORN_FWD_WS_DISPLAY = "304.001"

#: 模型里挂的字节值（`cost_eval/layers/residual.py::_MHC_PRE_SINKHORN_FWD_WS`）。
#: **观测** = 上面那个三位小数；**推断** = 在 `304.001` 的显示窗口 `[318767628, 318768676)`
#: 里，形如 `304 MiB + k·512 B`（分配器块粒度，见该常量注释里的仓内直证）的取值**唯一**：
#: `318767104 + 1024`。残余不确定度 ≤ 1.5 KiB = 0.0005 %。改它同样须先有新的真机测量。
MODEL_BYTES = 318768128

#: 本项**没有**覆盖的分支上，层内 fwd workspace 的既有值（`_fa_workspace` 16.0 /
#: MoE dispatch+combine staging 64.0）——用来钉「非融合分支没有被本项污染」。
_PRE_EXISTING_WS_MiB = {"dsv4hyb_r0_dense": 16.0,
                        "dsv4hyb_r4_moe": 64.0,
                        "dsv4hyb_r128_moe": 64.0}

_FUSED_TAG = "c fused   OFF L8 m4"
_LAYER_TYPES = sorted(_PRE_EXISTING_WS_MiB)
_SEQS = [1024, 2048, 4096]


def _resolve(tag=_FUSED_TAG, seq=None, parallel_over=None):
    """走**生产** builder（八跑 yaml 路，`use_fused_mhc: true`）拿 resolved 图。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = V.BY_TAG[tag]
        mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
        bundle, spec = V.build_bundle(mf)
        if seq is not None:
            spec.dims.S = seq
        p = bundle.parallel
        if parallel_over:
            p = dataclasses.replace(p, **parallel_over)
        world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
        return ShapeEval().resolve(spec, ParallelModel(p, spec.dims.n_layers, world))


def _layer(graph, layer_type):
    for layers in graph.stages.values():
        for lay in layers:
            if lay.layer_type == layer_type:
                return lay
    raise AssertionError(f"layer_type {layer_type} 不在图里；有 "
                         f"{sorted({l.layer_type for ls in graph.stages.values() for l in ls})}")


# ── 扁平 query 路（用来切 mHC 融合门 / 关 mHC / 开 MTP）─────────────────────────────
_DSV4_FLAT = {
    "preset": "dsv4_flash", "attn": "dsv4_hybrid",
    "layers": "8", "seq": "4096", "batch": "1", "mtp": "0",
    "experts": "8", "topk": "2", "dense_k": "1", "heads": "64", "kv_groups": "1",
    "hidden": "4096", "moe_ffn": "2048", "ffn": "16384",
    "q_lora": "1024", "kv_lora": "512", "qk_nope": "448", "qk_rope": "64",
    "v_head": "512", "vocab": "129280",
    "hc": "4", "mhc_fused": "1", "dsa_fused": "1", "ce_fused": "1",
    "compress_ratios": "0,4,128,4,128,4,128,4",
    "dp": "2", "tp": "1", "ep": "2", "pp": "4", "cp": "1", "method": "colossal",
    "optimizer": "muon", "opt_dtype": "fp32", "grad_bytes": "4", "maxdev_gib": "58",
    "select": "attn", "sel_layers": "", "sel_ops": "", "sel_cfg": "", "vpp": "1",
    "mbs": "", "dp_replicate": "1", "reshard": "default", "cpu_offload": "0",
    "prefetch": "1", "sp": "", "recompute": "None",
}


def _spec_from_flat(**over):
    q = dict(_DSV4_FLAT)
    q.update({k: str(val) for k, val in over.items()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errs, cfg, pa = S.parse_and_validate(q)
    assert not errs, errs
    return build_llm_spec(cfg), cfg


def _ws_by_op(ops):
    """{op 名: fwd workspace MiB}，只列非零项。"""
    return {op.name: getattr(op, "workspace_bytes", 0) / MiB
            for op in ops if getattr(op, "workspace_bytes", 0)}


# ═══════════════════════════════════════════════════════════════════════════════════
# 1. 实测值 + S 无关性：逐层型 × 逐 seq 穿过整条链
# ═══════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
@pytest.mark.parametrize("seq", _SEQS)
def test_measured_value_reproduces_end_to_end(layer_type, seq):
    """三个层型 × 三个 seq，层内 fwd `workspace` == 真机实测 304.001 MiB，逐字节。

    这条断言的对象是**测量值**，不是模型自洽性：它红了要么是建模链把实测值丢了/改了，
    要么是有人改了实测常数（后者只能由新的真机测量驱动）。

    · 「三个层型同值」= §4.1 两个插桩 rank（stage0 r0+r4 / stage1 r128+r4）FWD 相位极大值
      同为 304.001 + §4.2 逐窗口表；结构上也必然如此 —— `mhc_wrap` 给每个 decoder 层插的
      是**同一对** HC 模块，与该层的 compress_ratio 无关。
    · 「三个 seq 同值」= 实测 S 无关（该文 §8③）。本项因此是**常数**，不含 S。
    """
    lay = _layer(_resolve(seq=seq), layer_type)
    got = estimate_structure_memory(lay.ops).workspace / MiB
    assert f"{got:.3f}" == REAL_MHC_PRE_SINKHORN_FWD_WS_DISPLAY, (
        f"{layer_type} @S={seq}: 模型 fwd workspace={got:.4f} MiB，渲染成 tracker 的三位小数是 "
        f"{got:.3f}，而 167 真机实测原文是 {REAL_MHC_PRE_SINKHORN_FWD_WS_DISPLAY}。"
        f"这条实测值只能由新的真机测量改动（docs/kernel_workspace_2026-07-29.md §4.1/§4.3）。")


def test_s_independence_is_pinned_byte_exactly():
    """S 折 4 倍，本项**一个字节不动** —— 挡「顺手把它改成 ∝S 的式子」。

    实测在 S=1024/2048/4096 上同为 304.001 MiB（该文 §4.1 三次跑 / §8③）。反向那笔
    （融合稀疏 flash-MLA）**是** ∝S 的（332.5/465.0/730.0），两者不可混淆。"""
    lo = estimate_structure_memory(_layer(_resolve(seq=1024), "dsv4hyb_r4_moe").ops).workspace
    hi = estimate_structure_memory(_layer(_resolve(seq=4096), "dsv4hyb_r4_moe").ops).workspace
    assert lo == hi, f"S 无关性被打破：S=1024 {lo} B vs S=4096 {hi} B"


# ═══════════════════════════════════════════════════════════════════════════════════
# 2. 归属：两个 pre_sinkhorn op，各一份，不摊在层上
# ═══════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
def test_carried_by_the_two_pre_sinkhorn_ops_each_full_value(layer_type):
    """归属：**恰好两个** `*_hc_pre_sinkhorn` op，**各**扛整份 304.001（不是各一半）。

    实测原文（§4.3）：同一层的前向窗口里该模式出现**两次**（t=29631 与 t=29922，后者紧接
    dense FFN 的算子序列）→ attn_hc 与 ffn_hc 各一次，**每次 304.001 MiB**。
    层内取 max（不是 sum）—— 块寿命恒 1 tracker tick、任一时刻至多一块在世（§3/§4.1）。

    这道门排除「把一个层级常数摊在层上」与「只挂一侧」两种做法。"""
    lay = _layer(_resolve(), layer_type)
    ws = _ws_by_op(lay.ops)
    mhc = {k: f"{v:.3f}" for k, v in ws.items() if k.endswith("_hc_pre_sinkhorn")}
    assert mhc == {"attn_hc_pre_sinkhorn": REAL_MHC_PRE_SINKHORN_FWD_WS_DISPLAY,
                   "ffn_hc_pre_sinkhorn": REAL_MHC_PRE_SINKHORN_FWD_WS_DISPLAY}, mhc
    # 既有的 fwd workspace 载体（flash LSE / MoE staging）一个字节没被动。
    others = {k: v for k, v in ws.items() if not k.endswith("_hc_pre_sinkhorn")}
    assert others == pytest.approx(
        {"dsv4hyb_r0_dense": {"core_attn": 16.0},
         "dsv4hyb_r4_moe": {"sparse_attn": 16.0, "dispatch": 64.0, "combine": 64.0},
         "dsv4hyb_r128_moe": {"sparse_attn": 16.0, "dispatch": 64.0, "combine": 64.0},
         }[layer_type], abs=1e-6), others


# ═══════════════════════════════════════════════════════════════════════════════════
# 3. 没测的分支留 0
# ═══════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
def test_unfused_mhc_branch_stays_zero(layer_type):
    """**非融合 mHC 留 0，不拿融合分支的实测数去顶。**

    非融合走 `_unfused_hc_ops` 的 `rms_norm/matmul/sinkhorn` 小算子链
    （`hyper_connection.py:246-301`）——**根本不是** `npu_mhc_pre_sinkhorn` 这个 kernel，
    而 167 那次 tracker 跑的是 fused 配置，**没有采集**非融合分支的 FWD 单 kernel 极大值。
    留 0 是**如实的欠读**，不是「已确认为 0」。照 r0/r128/unfused 反向 workspace 留 0 的先例
    （`tests/test_bwd_kernel_workspace.py::test_unmeasured_branches_stay_zero`）。"""
    spec, cfg = _spec_from_flat(mhc_fused="0")
    assert cfg.use_fused_mhc is False
    ops = spec.layer_specs[layer_type].ops
    assert not [op for op in ops if op.name.endswith("_hc_pre_sinkhorn")], (
        "非融合分支不该有 pre_sinkhorn op")
    assert all(op.workspace is None or "318768128" not in str(op.workspace) for op in ops), (
        "非融合分支上出现了融合分支的实测常数——那是把没测的分支拿测过的数去顶")


@pytest.mark.parametrize("layer_type", _LAYER_TYPES)
def test_unfused_mhc_layer_workspace_falls_back_to_pre_existing(layer_type):
    """非融合分支的层内 fwd workspace == 本项入账**之前**的值（16.0 / 64.0），逐字节。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q = dict(_DSV4_FLAT)
        q["mhc_fused"] = "0"
        errs, cfg, pa = S.parse_and_validate(q)
        assert not errs, errs
        spec = build_llm_spec(cfg)
        pc, _opt, _hw, _swap = S._build_eval_specs(q, pa)
        world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
        g = ShapeEval().resolve(spec, ParallelModel(pc, spec.dims.n_layers, world))
    got = estimate_structure_memory(_layer(g, layer_type).ops).workspace / MiB
    assert got == pytest.approx(_PRE_EXISTING_WS_MiB[layer_type], abs=1e-6), (
        f"非融合 {layer_type} 的 fwd workspace={got:.3f} MiB，应为本项之前的 "
        f"{_PRE_EXISTING_WS_MiB[layer_type]}（_fa_workspace / MoE staging）")


def test_model_without_mhc_is_untouched():
    """无 mHC（`hc=1` → `mhc_wrap` no-op）的模型里不存在本项的载体 op。

    今天 14 个记分卡锚点里 12 个 DSv3 族 + 2 个 DSv4-align（真机跑是**非融合** mHC，
    `docs/fused_mhc_branch_mismatch_2026-07-30.md` §2.2）都落在这一格 → 本项对它们恒 0。"""
    spec, cfg = _spec_from_flat(hc="1")
    names = [op.name for lay in spec.layer_specs.values() for op in lay.ops]
    assert not [n for n in names if "_hc_pre_sinkhorn" in n], names


# ═══════════════════════════════════════════════════════════════════════════════════
# 4. 相位不串：fwd 通道 ≠ bwd 通道
# ═══════════════════════════════════════════════════════════════════════════════════

def test_it_uses_the_fwd_channel_not_the_bwd_channel():
    """本项走 `OpSpec.workspace`（fwd），**不**碰 `bwd_workspace{,_ref}`（bwd）。

    两条通道在 `mem_timeline` 上落在**不同相位的事件**：`workspace` 在 `fwd:<lid>`
    （`mem_timeline.py:583-585`，`rec()` 后立即清零）；`bwd_workspace` 在 `bwd@<lid>`
    （`:659`，同样用完即清）。混用会让一笔前向瞬态跑到反向断面上去 —— 那正是「为了让锚点动
    而挪相位」的失败模式。本门钉住：同一个 r4 层上两笔实测值**各在各的桶里、互不影响**。"""
    from cost_eval.layers.residual import build_hyper_connection_ops
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = V.BY_TAG[_FUSED_TAG]
        mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
        _, spec = V.build_bundle(mf)
    hc = build_hyper_connection_ops("attn", spec.dims)
    pre = [op for op in hc if op.name == "attn_hc_pre_sinkhorn"]
    assert len(pre) == 1
    assert pre[0].workspace is not None, "builder 没挂 fwd 通道"
    assert pre[0].workspace_ref is None, "本项是常数，不该占 TensorRef 通道"
    assert pre[0].bwd_workspace is None and pre[0].bwd_workspace_ref is None, (
        "本项被错挂到 **bwd** 通道上了——它是前向 kernel 的 workspace（实测在 FWD 相位窗口内）")
    sm = estimate_structure_memory(_layer(_resolve(), "dsv4hyb_r4_moe").ops)
    assert sm.workspace == MODEL_BYTES
    assert sm.bwd_workspace / MiB == pytest.approx(730.0, abs=1e-6), (
        "r4 层的 bwd 通道（融合稀疏 flash-MLA 反向 730.0，2026-07-29 实测）被本轮改动波及了")


def test_constant_is_not_divided_by_cp():
    """常数项**不吃** cp 整除 —— 除掉它会在 cp>1 上欠读 = OOM-**不安全**。

    `shape_eval.py:255` 的「含 S 就 ÷cp」只对**引用了 S** 的 workspace 表达式生效；本项是
    纯常数（实测 S 无关），故结构上不触发。cp>1 的真值**未实测**（见 docs 诚实清单）。"""
    l1 = _layer(_resolve(), "dsv4hyb_r4_moe")
    l2 = _layer(_resolve(parallel_over={"cp": 2, "dp_shard": 1}), "dsv4hyb_r4_moe")
    # **非空转自检**：cp=2 必须真的把随 S 切的量切掉一半，否则本门什么也没测。
    a1 = estimate_structure_memory(l1.ops).activation_saves
    a2 = estimate_structure_memory(l2.ops).activation_saves
    assert a2 < a1, f"cp=2 没有生效（activation_saves {a1} → {a2}）—— 本门会空转"
    base = estimate_structure_memory(l1.ops).workspace
    cp2 = estimate_structure_memory(l2.ops).workspace
    assert cp2 == base, (
        f"cp=2 下本项由 {base} B 变成 {cp2} B —— 常数项被 cp 除了，cp>1 将欠读（OOM-不安全）")


# ═══════════════════════════════════════════════════════════════════════════════════
# 5. 「OpSpec 重建口静默吞字段」的定点回归（这条已经真实发生过两次）
# ═══════════════════════════════════════════════════════════════════════════════════

def test_mhc_wrap_does_not_silently_drop_it():
    """`mhc_wrap` 产出的**生产层**必须原样带着本项。

    历史：`_rebuild` 的手写字段清单先后把 `norm_kind`、`workspace_ref` 在**全部 DSv4 层**上
    静默丢掉（`docs/census_fix_residual_carrier_2026-07-29.md` §5b）。现已改用
    `dataclasses.replace` + `_REF_FIELDS`；本门用「被 mHC 包装的生产层」取数，回潮必红。"""
    from cost_eval.layers.residual import build_hyper_connection_ops
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = V.BY_TAG[_FUSED_TAG]
        mf = V.derive_mf_config(V.DEFAULT_BASE_DIR, v)
        _, spec = V.build_bundle(mf)
    raw = [op for op in build_hyper_connection_ops("attn", spec.dims)
           if op.name == "attn_hc_pre_sinkhorn"]
    assert len(raw) == 1 and raw[0].workspace is not None, "裸 builder 没挂本项"
    got = estimate_structure_memory(_layer(_resolve(), "dsv4hyb_r4_moe").ops).workspace
    assert got == MODEL_BYTES, (
        f"被 mHC 包装后 fwd workspace={got} B（应为 {MODEL_BYTES} = "
        f"{REAL_MHC_PRE_SINKHORN_FWD_WS_DISPLAY} MiB）—— mhc_wrap 又在静默吞字段了"
        f"（见 cost_eval/layers/residual.py 的 _rebuild 注释）。")


def test_mtp_layer_does_not_silently_drop_it():
    """**MTP 层单列一门**：`cost_eval/layers/head.py:285` 拿 `mhc_wrap` 的 `wrapped[0]` 做
    `dataclasses.replace` 补一条 inputs 边 —— 而在融合分支下 `wrapped[0]` **正是**
    `attn_hc_pre_sinkhorn`，即本项的载体。这是「重建口吞字段」的第三个现场，
    历史上另两个都真的吞过（`norm_kind` / `workspace_ref`）。"""
    spec, cfg = _spec_from_flat(mtp="1")
    assert cfg.mtp_num_layers == 1
    mtp_keys = [k for k, lay in spec.layer_specs.items()
                if any(op.name == "mtp_hc_expand" for op in lay.ops)]
    assert len(mtp_keys) == 1, f"没找到 MTP 层：{sorted(spec.layer_specs)}"
    ops = spec.layer_specs[mtp_keys[0]].ops
    carriers = {op.name: op.workspace for op in ops
                if op.name.endswith("_hc_pre_sinkhorn")}
    assert set(carriers) == {"attn_hc_pre_sinkhorn", "ffn_hc_pre_sinkhorn"}, carriers
    assert all(w is not None for w in carriers.values()), (
        f"MTP 层的 pre_sinkhorn 丢了 workspace：{carriers} —— head.py:285 的 "
        f"`dataclasses.replace` 被改回手写字段清单了？")
