# -*- coding: utf-8 -*-
"""`extracted` graph source 适配器（`cost_eval/opdag/to_resolved.py`）的门测试。

四类断言：

  A. **缝真的接上了**：约定入口点存在、`extracted` 被注册、返回值逐层满足
     `cost_eval/liveness/contract.py` 的 18 条硬规则（`validate_resolved_graph` == 0）。
  B. **纪律不许退化**（每条都是"少了它就会静默出错"的那种）：
       - 部分解析的图**默认不给数**（抛 `IncompleteExtraction`），而不是返回一个偏小的峰值；
       - 任何张量都不许零填（B1：`local_numel > 0` 且 `dtype_bytes > 0`）；
       - 权威快照 md5 不符 / `tp,cp > 1` / 不认识的 layer_type → fail-loud；
       - 声明式 shape 种子不得与图内节点的产出撞名（否则声明会**覆盖**推断结果）。
  C. **两个已修 bug 的回归钉**（都是"内联后同名"引起的静默错算）：
       - 同基名不同尺寸的张量必须被拆成不同物理张量（契约 S3）；
       - `detached` 必须按**节点级** `attrs["detached"]`（帧精确）判，不许按裸名匹配 ——
         否则注意力的 `q` 会被 indexer 里那个同名 detached 张量误标。
  D. **覆盖度台账**（明账，不是容差）：当前抽出的图**不完备**，缺口逐条写在这里；
     覆盖度改善时这些测试会红，提醒更新记录 —— 与
     `test_acceptance_gate.py::test_delta_magnitude_gap_is_recorded` 同一条纪律。

**md5 门控**：凡断言真源行号/结构的测试都先校验权威快照 —— 对着另一个 commit 断言行号无意义。
"""
from __future__ import annotations

import hashlib
import os

import pytest

from cost_eval.liveness.contract import (layer_param_bytes,
                                         validate_resolved_graph)
from cost_eval.opdag import to_resolved as TR

# ── 权威快照门控（与 tests/test_opdag_components.py 同口径）─────────────────────────
_MD5 = dict(TR.SNAPSHOT_MD5)
_MD5.update({
    "pynative/transformers/hyper_connection.py": "14cf0de51c41df31f5e4439549934660",
    "pynative/transformers/transformer_layer.py": "0b1cce0c7d620544f01a2bea1014fa3a",
})


def _authoritative_pkg():
    for root in (os.environ.get("MINDFORMERS_ROOT"),
                 r"E:\97-codes\torch_parallel\mf-src-167\mindformers",
                 r"E:\97-codes\torch_parallel\mf-src-167"):
        if not root:
            continue
        for pkg in (root, os.path.join(root, "mindformers")):
            ok = True
            for rel, want in _MD5.items():
                p = os.path.join(pkg, *rel.split("/"))
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
        pytest.skip("权威 mindformers 快照缺失/md5 不匹配（绝不对着另一个 commit 断言）")
    os.environ["MINDFORMERS_ROOT"] = pkg
    return pkg


@pytest.fixture(scope="module")
def cfg(mf_pkg):
    """167 A/B 的 fused / unfused 两支（`ModelSpec` + `ParallelModel`）。"""
    import warnings

    from cost_eval.parallel_model import ParallelModel
    from tools.liveness_ab_validate import (BY_TAG, DEFAULT_BASE_DIR,
                                            build_bundle, derive_mf_config)
    warnings.simplefilter("ignore")
    if not os.path.isdir(DEFAULT_BASE_DIR):
        pytest.skip("缺 167 A/B base yaml 目录")
    out = {}
    for key, tag in (("fused", "a fused   ON  L8 m4"), ("unfused", "b unfused ON  L8 m4"),
                     ("fused_l4", "g fused   ON  L4 m4")):
        b, spec = build_bundle(derive_mf_config(DEFAULT_BASE_DIR, BY_TAG[tag]))
        p = b.parallel
        world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
        out[key] = (spec, ParallelModel(p, spec.dims.n_layers, world))
    return out


def _graph(cfg, key="fused"):
    spec, pm = cfg[key]
    return TR.resolve_graph(spec, pm, allow_partial=True)


def _layer(g, ltype):
    for st in sorted(g.stages):
        for l in g.stages[st]:
            if l.layer_type == ltype:
                return l
    raise AssertionError(f"图里没有 layer_type={ltype!r}")


# ═══════════════════════════════════════════════════════════════════════════
# A. 缝真的接上了
# ═══════════════════════════════════════════════════════════════════════════

def test_entry_point_matches_the_registry_contract():
    """入口点的模块路径与属性名是**约定死**的（`liveness/sources.py::EXTRACTED_ENTRY_POINT`）。"""
    from cost_eval.liveness.sources import EXTRACTED_ENTRY_POINT
    mod, attr = EXTRACTED_ENTRY_POINT
    assert mod == TR.__name__ and attr == "resolve_graph"
    assert callable(getattr(TR, attr))


def test_extracted_is_registered_as_a_graph_source(mf_pkg):
    from cost_eval.liveness import available_graph_sources, has_graph_source
    assert has_graph_source("extracted")
    assert "extracted" in available_graph_sources()


def test_resolve_graph_signature_takes_exactly_two_positional_args():
    """插座按 `fn(model_spec, parallel_model)` 调 —— 多一个必填位参就注册不上。"""
    import inspect
    sig = inspect.signature(TR.resolve_graph)
    pos = [p for p in sig.parameters.values()
           if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert len(pos) == 2
    assert all(p.default is p.empty for p in pos)


@pytest.mark.parametrize("key", ["fused", "unfused"])
def test_every_layer_satisfies_the_resolved_layer_contract(cfg, key):
    """18 条硬规则全过 —— 这是"抽取图能被同一个仿真器吃下"的唯一凭据。"""
    v = validate_resolved_graph(_graph(cfg, key))
    assert not v, "契约违约：\n  " + "\n  ".join(str(x) for x in v[:20])


def test_layer_ids_and_stages_line_up_with_hand_spec(cfg):
    """层序 / 层号 / stage 归属必须与 `hand_spec` **逐项相同** —— 否则 A/B 的差异不只来自图
    （重算域 `rc.is_full(lid)`、loss 层判定都按 layer_id 走）。"""
    from cost_eval.liveness import resolve_graph as rg
    spec, pm = cfg["fused"]
    hand, ext = rg(spec, pm, "hand_spec"), _graph(cfg, "fused")
    assert sorted(hand.stages) == sorted(ext.stages)
    for st in sorted(hand.stages):
        assert ([(l.layer_id, l.layer_type) for l in hand.stages[st]]
                == [(l.layer_id, l.layer_type) for l in ext.stages[st]])


# ═══════════════════════════════════════════════════════════════════════════
# B. 纪律
# ═══════════════════════════════════════════════════════════════════════════

def test_partial_graph_refuses_to_give_a_peak_by_default(cfg):
    """**默认不给数**：部分解析的图的峰值是下界，一旦并排进表就必然被当成"模型读到了这么多"。"""
    spec, pm = cfg["fused"]
    with pytest.raises(TR.IncompleteExtraction) as ei:
        TR.resolve_graph(spec, pm, allow_partial=False)
    msg = str(ei.value)
    assert "拒绝交出峰值" in msg
    assert TR.ALLOW_PARTIAL_ENV in msg, "必须告诉调用方怎么显式选择加入下界"
    assert ei.value.coverage is not None and ei.value.coverage.is_partial


def test_allow_partial_env_var_opts_in(cfg, monkeypatch):
    spec, pm = cfg["fused"]
    monkeypatch.setenv(TR.ALLOW_PARTIAL_ENV, "1")
    g = TR.resolve_graph(spec, pm)
    assert g.coverage.is_partial and g.stages


@pytest.mark.parametrize("key", ["fused", "unfused"])
def test_no_tensor_is_ever_zero_filled(cfg, key):
    """契约 B1 的正面断言：图里出现的**每一个**张量都有正的 numel 与 dtype 字节。

    解不出的东西只有两条去处：跳过承载它的节点，或进 `Coverage.unresolved_*` —— 绝不填 0。"""
    for st, layers in _graph(cfg, key).stages.items():
        for l in layers:
            for op in l.ops:
                for t in (*op.inputs, op.output, *op.params, *op.saves):
                    assert t.local_numel > 0 and t.dtype_bytes in (1, 2, 4, 8), \
                        f"{l.layer_type}/{op.name}/{t.name}"


def test_wrong_snapshot_is_fail_loud(tmp_path, monkeypatch):
    """换一份 commit 抽图 = 给一份真机从未跑过的代码建模 → 必须炸，不许悄悄出数。"""
    fake = tmp_path / "mindformers"
    (fake / "pynative").mkdir(parents=True)
    for rel in TR.SNAPSHOT_MD5:
        p = fake / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# not the authoritative snapshot\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="指纹不符"):
        TR._verify_snapshot(str(fake))


def test_missing_root_is_fail_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("MINDFORMERS_ROOT", str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match="pynative"):
        TR.mf_root()


def test_tp_or_cp_gt_one_is_fail_loud(cfg):
    """抽出的张量**没有 TP/CP placement 标注** → 按全局 numel 记账会成倍偏大，按猜的轴切会静默错。"""
    spec, pm = cfg["fused"]

    class _PM:                       # 只需要 degree()/stage_of()
        def __init__(self, inner, **over):
            self._i, self._o = inner, over

        def degree(self, a):
            return self._o.get(a, self._i.degree(a))

        def stage_of(self, lid):
            return self._i.stage_of(lid)

    for axis in ("tp", "cp"):
        with pytest.raises(RuntimeError, match=f"{axis}=2>1"):
            TR.resolve_graph(spec, _PM(pm, **{axis: 2}), allow_partial=True)


def test_unsupported_layer_type_is_fail_loud():
    """不认识的层型**不产空层** —— 空层会静默给出一个偏小的峰值。"""
    with pytest.raises(RuntimeError, match="尚不支持 layer_type"):
        TR._layer_kind("gqa_std")
    assert TR._layer_kind("embedding") == "embedding"
    assert TR._layer_kind("dsv4hyb_r128_moe") == "dsv4hyb"
    assert TR._ratio_of("dsv4hyb_r128_moe") == 128
    assert TR._ratio_of("dsv4hyb_r0_dense") == 0


def test_declared_shape_seed_must_not_collide_with_a_produced_name():
    """声明式种子撞上图内产出 = 声明**覆盖**推断结果 = 把"调用方说的"洗成"源真值"。必须炸。"""
    class _N:
        def __init__(self, out):
            self.out = out

    class _D:
        nodes = [_N("aggregated_attn:S·B·H:bf16")]

    with pytest.raises(RuntimeError, match="撞名"):
        TR._declared_seeds("decoder", _D(), TR.Coverage())


def test_every_declared_shape_carries_a_source_locator():
    """`_DECLARED_SHAPES` 的每一条都必须带 `file:line` 与理由（`kernel_saves` 的同一条纪律）。"""
    assert TR._DECLARED_SHAPES, "至少要有一条，否则这条纪律测试是空转"
    for (seg, name), (sym, src, why) in TR._DECLARED_SHAPES.items():
        assert seg in TR._SEED_SHAPES, seg
        assert sym and name
        assert ".py:" in src, (name, src)
        assert why and len(why) > 20, (name, why)


def test_every_declared_config_fact_carries_a_provenance():
    """`_DECLARED_FACTS` 同理：`DimTable` 装不下的 yaml/`__init__` 级事实必须逐条带出处。"""
    assert len(TR._DECLARED_FACTS) > 20
    for flag, _val, why in TR._DECLARED_FACTS:
        assert flag and why, flag
        assert why.startswith(("yaml:", "src:")) or "缺省" in why or "建模" in why, (flag, why)


def test_workspace_and_bwd_scratch_are_declared_absent_not_guessed(cfg):
    """契约 §不要求 明确排除 kernel 实现细节 → 恒 0，并在台账里说明"因此相对 hand_spec 少了什么"。"""
    g = _graph(cfg, "fused")
    for st, layers in g.stages.items():
        for l in layers:
            for op in l.ops:
                assert op.workspace_bytes == 0 and op.bwd_scratch_bytes == 0
    assert "workspace_bytes" in TR.Coverage().absent_calibration_note


# ═══════════════════════════════════════════════════════════════════════════
# C. 两个已修 bug 的回归钉
# ═══════════════════════════════════════════════════════════════════════════

def test_same_base_name_different_size_becomes_two_physical_tensors(cfg):
    """内联后 `q` 在一层里是**三个**物理张量（注意力 query / indexer query / detached cast 复本）。

    按名首见定型 = 契约 S3 明文警告的"同名异形被静默吞掉"。身份键必须含尺寸。"""
    l = _layer(_graph(cfg, "fused"), "dsv4hyb_r4_moe")
    sizes = {}
    for op in l.ops:
        for t in (*op.inputs, op.output, *op.saves):
            sizes.setdefault(t.name, set()).add((t.local_numel, t.dtype_bytes))
    for name, sigs in sizes.items():
        assert len(sigs) == 1, f"{name} 同名异形未被拆开：{sigs}"
    qs = {t.name: t.local_numel for op in l.ops
          for t in (*op.inputs, op.output) if t.name.split("#")[0] == "q"}
    assert len(qs) >= 2, f"`q` 应该被拆成多个物理张量，实得 {qs}"
    assert len(set(qs.values())) >= 2, f"拆出来的 `q` 尺寸应各不相同，实得 {qs}"


def test_attention_query_is_not_marked_detached(cfg):
    """`detached` 必须按**节点级** `attrs["detached"]`（帧精确）判。

    按裸名匹配会把 `deepseek_v4:240` 的注意力 query 误标成 detached（它与 `indexer.py:215`
    那个 detached 的 `q` 同名）→ 掉出 `kept_for_backward` → 反向少读 256 MiB/r4 层。"""
    l = _layer(_graph(cfg, "fused"), "dsv4hyb_r4_moe")
    attn_q = [t for op in l.ops for t in (*op.inputs, op.output)
              if t.name.split("#")[0] == "q"
              and "deepseek_v4_hybrid_attention.py:240" in (t.src or "")]
    assert attn_q, "没找到 deepseek_v4:240 产出的注意力 query"
    assert all(not t.detached for t in attn_q), \
        "注意力 query 被误标 detached（裸名匹配的老 bug 回潮）"
    det = [t for op in l.ops for t in (*op.inputs, op.output, *op.saves) if t.detached]
    assert det, "`ops.stop_gradient` 的产物必须仍被标 detached（不能一刀切成 False）"
    assert any("csa.py:66" in (t.src or "") for t in det), \
        "csa.py:665/666 的 stop_gradient 产物应在 detached 集里"


def test_norm_fp32_prelift_is_undone(mf_pkg):
    """契约明文：norm 输入的 fp32 cast 由消费侧 `structure_mem._dt` 再抬，**不**预抬。

    `bprop_rules.py:127-130` 用 `attrs["ln_compute_dtype"]` 预抬过了 → 不撤会被抬两次（×2）。"""
    class _N:
        op = "Norm"
        ins = ["x:S·B·H:bf16"]
        attrs = {"ln_compute_dtype": "fp32"}

    class _S:
        name, dtype = "x", "fp32"

    assert TR._undo_norm_prelift(_N(), _S()) == "bf16"
    _N.attrs = {}
    assert TR._undo_norm_prelift(_N(), _S()) == "fp32"   # 没预抬过 → 原样保留


# ═══════════════════════════════════════════════════════════════════════════
# D. 覆盖度台账（明账；覆盖度改善时**应当**变红）
# ═══════════════════════════════════════════════════════════════════════════

#: 当前抽出的图的**级联根**（各 segment 执行序上第一个被跳过的节点）。每条都是一处
#: `shape_infer` 的结构性盲区，逐条见 `docs/to_resolved_adapter_2026-07-25.md` §4。
#: **台账更新（2026-07-28，`docs/opdag_coverage_close_2026-07-28.md`）**：2026-07-25 记的那
#: 4 处级联根**全部修掉**了，逐条对应关系如下 —— 不变量（"级联根必须逐条在册；修好一处就更新
#: 台账"）原样保留，只换了例子：
#:   `hyper_connection.py:408` → `shape_infer` 的**权重派生节点**通路（`param_shapes` 喂回）
#:                               + walker 的**跨行调用权重归属**修复；
#:   `vocab_embedding.py:85`   → `mint.gather` 的产出形 = index 形（算子定义）；
#:   `linear.py:132`           → `init_dims` 支持 `weight_shape = (output_size, input_size)`
#:                               这种局部元组 + `Linear.__init__` 位置形参种子；
#:   `loss.py:197`             → `_LogSoftmax` 的产出形声明（`loss.py:134-143` 逐字）。
#: **台账更新（2026-07-29，`docs/opdag_symbolic_axes_2026-07-29.md`）**：「符号轴结构恢复」
#: 落地后，2026-07-28 记的 4 处**全部出表**，新暴露 5 处。逐条对应关系：
#:   `compressor.py:233` → `_slice` 支持 `":<stop>:<step>"`，且步长整除终点**可证**
#:                         （`total_seq_len = n_compressed·ratio` @ `:230` ⇒ 系数整除）；
#:   `deepseek_v4_hybrid_attention.py:205` → `permute` 用 walker 早就记下的 `permute_dims`
#:                         精确重排（不再退 `~`）⇒ `core_out` 有轴结构 ⇒ split 可解；
#:   `csa.py:485`        → ① `permute` 精确化让 `kv_t` 拿到 `B·1·S·v_head_dim`；
#:                         ② `csa.py:482` 的 reshape 目标里**恰好一个**维（`sk`）没导出，
#:                            由**元素数守恒**唯一确定（与 `-1` 同一条算子定义）；
#:   `compressor.py:216` → `_chunk` 用 walker 记下的 `chunks`/`chunk_dim` 精确切轴
#:                         （`compressor.py:169` `chunk(tensor, 2, dim=-1)`）⇒
#:                         `_overlap_transform` 的产物恢复 `S//4·8·B·v_head_dim` ⇒ 归约可解。
#: **不变量逐字保留**：级联根必须逐条在册、每条带一句"为什么还挡着"；修好一处就更新台账。
LEDGER_BLOCKERS = {
    # ── 新暴露（都是**上游本来就错 / 本来就没解出、此前被 `~` 盖住**的）─────────
    # `mint.arange(n_compressed)`（`csa.py:777` 的 causal mask）：`n_compressed =
    # int(compressed_kv.shape[0])`（`csa.py:747`）是 construct 局部标量，且它依赖
    # 下面那条 `cat` —— 那条 `cat` 一拒，它就无源可解。
    "csa.py:777",
    # `cmp_residual_k = Tensor([int(key_length) % self.compress_ratio], …)`
    # （`indexer.py:219`）：`Tensor([...])` 这种**值列表**构造没有 shape 规则。
    "indexer.py:219",
    # `kv_full = cat([key, compressed_kv], dim=0)`（`csa.py:747`）：两个操作数一个 rank 3
    # （压缩器的**内联返回值**少了 `compressor.py:224` 的 `unsqueeze(pooled, -2)` 那一步）、
    # 一个 rank 4 —— `cat` 的前置条件不成立 ⇒ 至少一处上游 shape 是错的 ⇒ 整条拒绝
    # （`concat_shape_mismatch`）。同一守卫在 `csa.py:818` 上抓到过一个更贵的错：
    # 按轴相加会把 topk 轴从 512 放大成 4224，再经 `csa.py:485` 的元素数守恒传导，
    # 整层 saves 从 ~22 GiB 变成 **86 GiB**（见 docs/opdag_symbolic_axes_2026-07-29.md §3）。
    "csa.py:747",
    "csa.py:818",
    # `pooled = (kv.astype(fp32) * weights).sum(dim=1)`：`compressor.py:169` 的
    # `chunk(tensor, 2, dim=-1)` 此前丢掉轴 → `_overlap_transform` 的产物退 `~` → 归约被拒。
    # 现已解开（`_chunk` 用 walker 记下的 `chunks`/`chunk_dim`），保留在册备查。
    "compressor.py:216",
    # `sink = cast(reshape(attn_sink, (1, n, 1, 1)), fp32)`：`attn_sink` 是 `Parameter`，
    # 按契约 W2/W4 不进 `ins`；该 reshape 节点 `ins` 为空且不是"操作数全是权重"的形态。
    "csa.py:506",
    # `router.py:393` 的 `topk`：`k` 未被 walker 记进 attrs（`reduce_axis_unknown`）。
    "router.py:393",
    # ── 保留在册备查（历史上出现过、现已解开）──────────────────────────────
    "compressor.py:233",
    "deepseek_v4_hybrid_attention.py:205",
    "csa.py:485",
    # `language_model_embedding.py:134` 的 transpose 已随「过期种子」修复解开，保留在册备查。
    "language_model_embedding.py:134",
}


@pytest.mark.parametrize("key", ["fused", "unfused"])
def test_coverage_is_partial_and_the_blockers_are_the_recorded_ones(cfg, key):
    """**如实记账**：图还不完备，级联根逐条在册。修好一处 → 这个测试变红，提醒更新记录。"""
    cov = _graph(cfg, key).coverage
    assert cov.is_partial
    got = {src for (src, _op, _r), _k in cov.blockers(10)}
    assert got <= LEDGER_BLOCKERS, f"出现了台账之外的级联根：{got - LEDGER_BLOCKERS}"
    # **期望变更台账（2026-07-29）**：原断言钉的是 `compressor.py:216`（压缩链 `-1` 消元），
    # 它已被 `_chunk` 的精确切轴解开、**出表**。不变量（"台账不许是空的：必须点名一个仍在
    # 挡着的根，修好了就来改这一行"）逐字保留，只换例子 —— 换成两支各自的当前第一根：
    #   unfused → `csa.py:747`（`cat` 前置条件不成立，`concat_shape_mismatch`）
    #   fused   → `router.py:393`（`topk` 的 `k` 未记进 attrs）
    assert got & {"csa.py:747", "router.py:393", "csa.py:777", "indexer.py:219"}, \
        f"台账里点名的根一个都不在了（修好了就更新台账）：{got}"


def test_param_census_resolves_more_than_it_gets_into_the_graph(cfg):
    """权重形状来自 `__init__`，与激活数据流无关 → 承载节点被跳过时权重仍可解析、但进不了图。

    两个口径的差额是本来源"欠读 param"的**根因说明**，不是容差。"""
    cov = _graph(cfg, "fused").coverage
    t = cov.totals()
    assert t["param_census_bytes"] > t["param_in_graph_bytes"] > 0
    assert t["param_census_bytes"] / 2 ** 20 > 3000     # 实测 3290.9 MiB


def test_extracted_param_bytes_are_still_short_of_hand_spec(cfg):
    """B4 台账：抽取侧的 param 字节今天**短一截**（叶子 `Linear` 之外的权重多数还没进图）。

    契约 §4.2 的负差信号 —— 这里如实记下量级，而不是放宽 `validate_param_census`。"""
    from cost_eval.liveness import resolve_graph as rg
    spec, pm = cfg["fused"]
    hand, ext = rg(spec, pm, "hand_spec"), _graph(cfg, "fused")
    for st in sorted(hand.stages):
        for h, e in zip(hand.stages[st], ext.stages[st]):
            assert layer_param_bytes(e) <= layer_param_bytes(h), \
                f"{h.layer_type}: 抽取侧 param 反而更多 —— 可能重复计入（正差）"


def test_declared_shapes_are_recorded_in_the_coverage_ledger(cfg):
    """调用方声明的形状必须在覆盖度报告里**可见** —— 不许与"推断出来的"混为一谈。"""
    cov = _graph(cfg, "fused").coverage
    names = {n for c in [cov] + cov._all_children() for n, _s, _src, _w in c.declared_shapes}
    assert {"aggregated_attn", "aggregated_ffn"} <= names
    rep = cov.report()
    assert "声明" in rep and "hyper_connection.py:397" in rep
    assert "PARTIAL" in rep and "级联根" in rep


# ═══════════════════════════════════════════════════════════════════════════
# E. 验收门按来源隔离（一个还不成熟的来源**不许**把门拖下水）
# ═══════════════════════════════════════════════════════════════════════════

def _one_run(sources):
    import warnings

    from tools.liveness_ab_validate import (BY_TAG, DEFAULT_BASE_DIR,
                                            evaluate_run)
    warnings.simplefilter("ignore")
    if not os.path.isdir(DEFAULT_BASE_DIR):
        pytest.skip("缺 167 A/B base yaml 目录")
    return evaluate_run(DEFAULT_BASE_DIR, BY_TAG["a fused   ON  L8 m4"], sources, "chain2")


def test_a_failing_source_does_not_take_down_the_other_sources():
    """一个来源抛异常 → 它那一列记 `SourceFailure`（附 `file:line`），**其余来源照常出数**。

    这是验收台存在的意义：`bucket` / `hand_spec` 必须在 `extracted` 成熟过程中持续出数。"""
    from cost_eval.liveness import (register_graph_source,
                                    unregister_graph_source)

    def _boom(spec, pm):
        raise RuntimeError("造出来的失败 @ fake_module.py:123")

    register_graph_source("_test_boom", _boom, override=True)
    try:
        r = _one_run(["hand_spec", "_test_boom"])
    finally:
        unregister_graph_source("_test_boom")
    assert r.ok("hand_spec") and r.ok("bucket")
    assert all(x is not None for x in r.series("hand_spec"))
    assert not r.ok("_test_boom")
    f = r.failed("_test_boom")
    assert f is not None and f.kind == "RuntimeError"
    assert f.locator == "fake_module.py:123", "必须把源定位符从消息里抠出来"
    assert all(x is None for x in r.series("_test_boom")), "失败来源必须是 None，不许填 0"


def test_failing_source_invariants_are_skipped_not_passed():
    """无值 ⇒ 不变量标 **SKIP**（"不知道"和"通过"是两件事），且不参与退出码。"""
    from tools.liveness_ab_validate import PAIRS, Check, model_invariants

    class _R:
        n_stages = 4

        def series(self, key):
            return (None,) * 4

        def failed(self, key):
            from tools.liveness_ab_validate import SourceFailure
            return SourceFailure("x", "RuntimeError", "boom", "a.py:1")

        def ok(self, key):
            return False

    res = {t: _R() for pair in PAIRS for t in pair[1:]}
    checks = model_invariants(res, "x")
    assert checks and all(c.skipped for c in checks)
    assert all(c.status == "SKIP" for c in checks)
    assert not Check("n", False, "d", advisory=True).skipped


def test_gate_passes_when_only_advisory_sources_are_red(capsys):
    """`--gate` 的退出码只反映 `--expect-green` 提名的来源。"""
    import warnings

    from tools.liveness_ab_validate import GATE_REQUIRED_SOURCES, main
    warnings.simplefilter("ignore")
    assert GATE_REQUIRED_SOURCES == ("bucket", "hand_spec")
    rc = main(["--grad-mode", "chain2", "--gate"])
    out = capsys.readouterr().out
    assert rc == 0, out[-2500:]
    assert "来源健康" in out
    assert "结论: PASS" in out


def test_gate_fails_when_a_required_source_is_red(capsys):
    """反面：把一个必然失败的来源提名成"必须绿" → 门必须判死（隔离不等于放过）。"""
    import warnings

    from cost_eval.liveness import (register_graph_source,
                                    unregister_graph_source)
    from tools.liveness_ab_validate import main
    warnings.simplefilter("ignore")

    def _boom(spec, pm):
        raise RuntimeError("造出来的失败 @ fake_module.py:7")

    register_graph_source("_test_boom2", _boom, override=True)
    try:
        rc = main(["--grad-mode", "chain2", "--sources", "hand_spec,_test_boom2",
                   "--expect-green", "bucket,hand_spec,_test_boom2", "--gate"])
    finally:
        unregister_graph_source("_test_boom2")
    out = capsys.readouterr().out
    assert rc == 1, out[-2500:]
    assert "结论: FAIL" in out and "fake_module.py:7" in out
