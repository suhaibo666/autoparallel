# -*- coding: utf-8 -*-
"""**锚点 ↔ 站点 yaml 一致性门**（2026-07-30，`docs/compress_ratios_mismatch_2026-07-30.md` §8）。

## 为什么「分类完备门」拦不住这一类

`tests/test_flat_query_reachability.py` 的判据 ① 问的是：每个 `LLMConfig` 字段**有没有被分类**
（要么有 UI 门控键、要么显式声明为 `llm_json` 专属）。`csa_compress_ratios` **有**分类 ——
它就写在 `_LLM_JSON_ONLY_FIELDS` 里，于是那道门一直是绿的。

> 但那条登记是一句**声明**：「扁平路上它恒取预设值，代价已知可接受」。
> 而这句声明**是假的** —— 锚点比对的站点 yaml 明明给了另一张逐层表
> （`[0,4,128,4,128,4,128,4]` vs 预设的循环近似 `(0,4,128,0,4,128,0,4)`）。
> **门只问「有没有分类」，从不问「分类是不是真的」。** 声明与事实之间没有任何判据。

更根本地：那道门是**字段视角**的，它不知道「锚点」「站点 yaml」这些东西存在，
因此它在结构上就**不可能**发现「锚点扁平 query 建出来的模型 ≠ 那次真机跑的模型」。

## 本门守什么（补上缺的那一问）

对每一对「**归档的真机 launcher yaml** ↔ **对着那次跑打分的锚点扁平 query**」：

① **逐字段一致**：yaml 的权威 `LLMConfig` 与扁平 query 重建的 `LLMConfig` 必须逐字段相等，
   除非该字段进了下面两张**带判据**的表之一。
② **登记不许发霉**：登记的项必须**真的**还在差着（否则是残留豁免，会静默放行真分歧）。
③ **对齐后必须同图**：按两张表对齐后，`LLMConfig` 必须完全相等，且两边 `build_llm_spec` 的
   `DimTable` 与 `layer_pattern` 逐字节相同 —— 「等价」得拿图说话。
④ **`_LLM_JSON_ONLY_FIELDS` 不许收纳假声明**：凡在 ① 里**没登记**又真差着的字段，
   都不许停在那张「已知代价清单」里。这条直接把 2026-07-30 的真实状态钉成红。
⑤ **推断出来的表要与实证的表自洽**：185 4 层相位的 `[0,4,128,4]` 必须是 8 层站点表的前缀。

两张登记表的分工（**都要求证据，不接受口头豁免**）：
  - `_DECLARED_EQUIVALENCES`：写法不同、**语义相同**（判据 ③ 当场用图证明）。
  - `_DECLARED_INERT_DIFFS`：取值**真的不同**，但对该锚点的峰值影响**当场实测为 0**
    （`test_inert_diffs_are_really_byte_neutral` 把 yaml 的值强行灌回去再评一遍）。
    这张表不是豁免，是**带实测的代价声明**；哪天那个差值开始承重，门就红。

**这条门会红于**：新增/改动锚点时用了与站点 yaml 不同的结构；给某字段补 UI 键却忘了在锚点上
显式给值；或有人把一个锚点真需要的字段塞进 `_LLM_JSON_ONLY_FIELDS` 蒙混过去。
"""
import dataclasses
import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import serve_explorer as S
from cost_eval.build_llm import build_llm_spec
from cost_eval.configs.from_mindformers import load_mindformers_yaml

from test_pp4_recompute_anchor import _BASE as PP4_BASE
from test_probe185_recon import _SITE_RATIOS

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AB_DIR = os.path.join(_REPO, "analysis", "realmachine", "ab_fusion_2026-07-25")

#: 「归档真机 launcher yaml ↔ 对着它打分的锚点扁平 query」配对。
#: 认亲证据（`docs/fused_mhc_branch_mismatch_2026-07-30.md` §2.1、本轮 doc §2.1）：
#:   `tests/test_pp4_recompute_anchor.py` 的 `REAL_ON` 逐位等于
#:   `tools/liveness_ab_validate.py` 的 run `a fused ON L8 m4`，而八跑门就是从这两份 yaml 派生的
#:   —— 锚点门与八跑门比的是**同一批真机跑**，两边的模型结构因此必须逐字段相同。
#: 第二对（unfused-DSA）是同族变体：站点两份 yaml 只差 `apply_dsa_kernel_fusion`
#:   （`dsv4h_{fused,unfused}_pp4_recomp.yaml`），而 185 的 U 相位锚点走的正是 `dsa_fused=0`
#:   这条扁平路，故把它一并纳入比对。
_PAIRS = [
    ("dsv4h_fused_pp4_recomp.yaml", "pp4/pp8/MTP 锚点（fused DSA）", dict(PP4_BASE)),
    ("dsv4h_unfused_pp4_recomp.yaml", "185 U 相位同族（unfused DSA）", dict(PP4_BASE, dsa_fused="0")),
]

#: **写法不同、语义相同** —— 每条是 `(判据, 理由)`，判据当场跑；判据 ③ 再用图证明。
_DECLARED_EQUIVALENCES = {
    "ffn_hidden_size": (
        lambda yaml_v, anchor_v, llm: yaml_v is None and anchor_v == 4 * llm.hidden_size,
        "站点 yaml 没写 ffn_hidden_size → 权威 LLMConfig 里是 `None`（惰性语义「取 4·H」）；"
        "而扁平 UI 的 `ffn` 是个可见输入框、必须填一个数，故锚点写显式 4·H=16384。"
        "`_llm_to_fields` 回填时对 None 同样发**生效值** 4·H，两边落到同一张 DimTable 与同一条 "
        "layer_pattern —— 由判据 ③ 当场证明。"),
}

#: **取值真的不同、但对锚点峰值实测为 0** —— 每条是 `(yaml 侧取值, 理由)`。
#: 由 `test_inert_diffs_are_really_byte_neutral` 把 yaml 的值灌回锚点再评一遍来证明。
_DECLARED_INERT_DIFFS = {
    "kept_frag_factor": (
        1.6,
        "**yaml 适配器专属的经验标定 margin**，不是结构：`from_mindformers.py:515-517` 对"
        "「MoE 且非 (dsv4_hybrid ∧ dsa_fused)」注入 1.6，而 `deepseek_v4()` 预设从不设它"
        "（`llm_config.py:92` 默认 0.0；`from_mindformers.py:523` 逐字「直连 LLMConfig(绕过本"
        "适配器)默认 0」——这条不对称是**当时就写下来的已知代价**，不是本轮新漏）。"
        "对本族锚点恒 0：`cross_entropy_fused=True` → `loss_lids` 空 → `mem_timeline.py:748` "
        "的 gate 永不成立。本门当场实测这一点。"),
    "nr_moe_frag_factor": (
        0.6,
        "同上（`from_mindformers.py:526-528` 注入 0.6）。该 margin 只在 pp==1 单 stage 无重算 "
        "loss-BWD 生效（`mem_timeline.py:760`），且同样被融合 CE 挡住。其出处自带风险留档"
        "（`from_mindformers.py:519-525`：0.6 只在 **DSv3** 两锚点标定过，注入到别的结构是"
        "**未经真机验证的外推**）—— 正因为它是经验标定而非物理量，本轮**不**把它搬到扁平路上；"
        "2026-07-24 起本评估器的口径就是去经验补偿的纯理论。它一旦开始承重，本门变红并逼人决定。"),
}

_DECLARED = set(_DECLARED_EQUIVALENCES) | set(_DECLARED_INERT_DIFFS)


def _yaml_llm(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_mindformers_yaml(os.path.join(_AB_DIR, name)).llm


def _anchor_llm(query):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        errs, cfg, pa = S.parse_and_validate(query)
    assert not errs, errs
    assert "llm_json" not in query, "锚点必须走**无 llm_json** 的扁平路（这正是出事的那条路）"
    return cfg


def _diffs(name, query):
    return S._llm_field_diffs(_yaml_llm(name), _anchor_llm(query))


def _peaks(query, **over_cfg):
    """评估该 query 的逐 stage 峰值；`over_cfg` 非空时在 `LLMConfig` 上就地覆盖后再评。"""
    orig = S.parse_and_validate

    def patched(p):
        errs, cfg, pa = orig(p)
        if cfg is not None:
            cfg = dataclasses.replace(cfg, **over_cfg)
        return errs, cfg, pa

    if over_cfg:
        S.parse_and_validate = patched
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = S.eval_config(query)
    finally:
        S.parse_and_validate = orig
    assert r.get("ok"), r.get("errors")
    return {st["stage"]: st["peak"] for st in r["stages"]}


@pytest.mark.parametrize("name,label,query", _PAIRS, ids=[p[1] for p in _PAIRS])
def test_anchor_query_rebuilds_the_site_yaml_model(name, label, query):
    """① 锚点扁平 query 必须重建出**那次真机跑的**结构，未登记的差异一律红。"""
    llm = _yaml_llm(name)
    unexplained = []
    for field, yaml_v, anchor_v in _diffs(name, query):
        if field in _DECLARED_INERT_DIFFS:
            continue
        eq = _DECLARED_EQUIVALENCES.get(field)
        if eq is None or not eq[0](yaml_v, anchor_v, llm):
            unexplained.append((field, yaml_v, anchor_v))
    assert not unexplained, (
        f"{label}：锚点扁平 query 建出来的模型与站点 yaml `{name}` **不是同一份**，"
        f"未解释的字段差异 {unexplained}。\n"
        f"锚点是拿仿真值去对**那次真机跑**的读数打分；结构一旦分歧，打的就是另一个模型的分"
        f"（2026-07-30 的 `use_fused_mhc` 与 `csa_compress_ratios` 两例都是这么发生的）。\n"
        f"请三选一：① 在锚点 query 里显式给上站点 yaml 的值（必要时先给该字段补 UI 键 + "
        f"`_LLM_FIELD_GATE` 登记 + `_llm_to_fields` 回填，照 `compress_ratios`/`mhc_fused` 的样子）；"
        f"② 若是**语义等价**的写法差异，进 `_DECLARED_EQUIVALENCES` 并写清判据；"
        f"③ 若取值真不同但**对峰值影响为 0**，进 `_DECLARED_INERT_DIFFS`（本门会当场实测验证）。")


def test_declared_entries_are_not_stale():
    """② 登记项必须**至少在一对**上真的还在差着 —— 残留的豁免会静默放行真分歧。

    （按**全体配对**判，不逐对判：`kept_frag_factor`/`nr_moe_frag_factor` 只在 unfused 那一对
    上有差异——`from_mindformers.py:515` 的注入条件对 fused-DSv4 就是 0，与预设同值。）"""
    seen = set()
    for name, label, query in _PAIRS:
        seen |= {f for f, _, _ in _diffs(name, query)}
    stale = sorted(_DECLARED - seen)
    assert not stale, (
        f"{stale} 已登记为「已知差异」，但它们在**任何**配对上都不再有差异 —— 请删掉这些登记，"
        f"否则日后真出分歧时会被静默放行。")


@pytest.mark.parametrize("name,label,query", _PAIRS, ids=[p[1] for p in _PAIRS])
def test_declared_entries_leave_the_same_graph(name, label, query):
    """③ 按两张登记表对齐后，`LLMConfig` 必须完全相等，且两边的图逐字节相同。"""
    llm, cfg = _yaml_llm(name), _anchor_llm(query)
    aligned = dataclasses.replace(llm, **{f: getattr(cfg, f) for f in _DECLARED})
    rest = S._llm_field_diffs(aligned, cfg)
    assert not rest, f"{label}：对齐登记项后仍有差异（= 还有没登记的分歧）：{rest}"
    a, b = build_llm_spec(aligned), build_llm_spec(cfg)
    assert a.dims == b.dims, f"{label}：DimTable 不同：{_dim_diffs(a.dims, b.dims)}"
    assert a.layer_pattern == b.layer_pattern, (
        f"{label}：层序列不同（层型分布分歧 = 在给另一个模型打分）：\n"
        f"  yaml   = {a.layer_pattern}\n  anchor = {b.layer_pattern}")


def _dim_diffs(x, y):
    dx, dy = dataclasses.asdict(x), dataclasses.asdict(y)
    return [(k, dx[k], dy[k]) for k in dx if dx[k] != dy[k]]


def test_inert_diffs_are_really_byte_neutral():
    """`_DECLARED_INERT_DIFFS` 的「影响为 0」必须**实测**，不是口头承诺。

    做法：把 yaml 侧的取值强行灌回锚点的 `LLMConfig`，逐 stage 峰值必须**逐 MiB 相同**。
    哪天它开始承重（gate 条件变了 / 有人去掉融合 CE），本判据变红，逼当场做决定 ——
    而不是让一个"已知无害"的差异悄悄长成第四例错配。
    """
    measured = 0
    for name, label, query in _PAIRS:
        seen = {f for f, _, _ in _diffs(name, query)}
        over = {f: v for f, (v, _) in _DECLARED_INERT_DIFFS.items() if f in seen}
        if not over:
            continue                       # 这一对上两侧同值（如 fused-DSv4 两边都是 0.0）
        measured += 1
        base, forced = _peaks(query), _peaks(query, **over)
        assert base == forced, (
            f"{label}：把站点 yaml 的 {over} 灌回锚点后峰值变了（{base} → {forced}）—— "
            f"`_DECLARED_INERT_DIFFS` 声明的「影响为 0」不再成立。这不再是无害差异："
            f"锚点正在用与真机跑不同的 margin 打分，请当场决定是把它接到扁平路上，还是改口径。")
    assert measured, (
        "`_DECLARED_INERT_DIFFS` 非空却一次都没测上 —— 本判据成了空转（同 ② 的残留问题）。")


@pytest.mark.parametrize("name,label,query", _PAIRS, ids=[p[1] for p in _PAIRS])
def test_json_only_registry_holds_no_false_exemption(name, label, query):
    """④ **本轮的核心守卫**：真差着又没登记的字段，不许停在 `_LLM_JSON_ONLY_FIELDS`（= 假声明）。

    2026-07-30 之前的真实状态就是这样：`csa_compress_ratios` 一边登记着「扁平路上恒取预设值、
    代价可接受」，一边与站点 yaml 差着一整张逐层表。分类门问不出这件事，本判据问得出。
    """
    offenders = sorted(({f for f, _, _ in _diffs(name, query)} - _DECLARED)
                       & set(S._LLM_JSON_ONLY_FIELDS))
    assert not offenders, (
        f"{label}：字段 {offenders} 被登记为「`llm_json` 专属 / 扁平路上恒取预设值，代价可接受」，"
        f"但归档的站点 yaml `{name}` 对它们给的值与锚点扁平 query 解析出的**不一样**，"
        f"且没有任何带判据的登记为它背书 —— 那条登记是**假声明**，锚点正在给另一份结构打分。\n"
        f"修法：给该字段补 UI/隐藏字段 + `_LLM_FIELD_GATE` 登记 + `_llm_to_fields` 回填"
        f"（照 `compress_ratios`/`mhc_fused`/`ce_fused` 的样子），并在锚点上显式给站点值。")


def test_this_gate_actually_fires(monkeypatch):
    """**证明本门会响**：把 `compress_ratios` 旋钮拆掉（= 复现 2026-07-30 之前的真实状态）
    —— 锚点立刻退回预设的循环近似，①/④ 两条判据必须同时点名 `csa_compress_ratios`。
    不会响的门等于没有门。"""
    gate = {k: v for k, v in S._LLM_FIELD_GATE.items() if k != "csa_compress_ratios"}
    monkeypatch.setattr(S, "_LLM_FIELD_GATE", gate)
    monkeypatch.setattr(S, "_LLM_JSON_ONLY_FIELDS",
                        S._LLM_JSON_ONLY_FIELDS | {"csa_compress_ratios"})
    # 旋钮拆掉 = `parse_and_validate` 不再消费 `compress_ratios` 键（回到基座的循环近似）。
    monkeypatch.setattr(S, "_parse_compress_ratios", lambda raw: (None, None))
    name, label, query = _PAIRS[0]
    assert "csa_compress_ratios" in {f for f, _, _ in _diffs(name, query)}
    with pytest.raises(AssertionError, match="csa_compress_ratios"):
        test_anchor_query_rebuilds_the_site_yaml_model(name, label, query)
    with pytest.raises(AssertionError, match="csa_compress_ratios"):
        test_json_only_registry_holds_no_false_exemption(name, label, query)


# ── ⑤ 推断出来的 4 层表必须与实证的 8 层站点表自洽 ────────────────────────────────────
def test_probe185_site_ratios_are_consistent_with_the_archived_yaml():
    """`tests/test_probe185_recon._SITE_RATIOS` 的 8 层项必须**逐位等于**归档 yaml；
    4 层项（出处是 `analysis/dsv4_flash_calibration_handoff_2026-07-22.md:121` 的文字记录）
    必须是它的前缀 —— 两处出处互不依赖，这条守它们不许各自漂。"""
    site8 = _yaml_llm("dsv4h_fused_pp4_recomp.yaml").csa_compress_ratios
    got8 = tuple(int(x) for x in _SITE_RATIOS[8].split(","))
    got4 = tuple(int(x) for x in _SITE_RATIOS[4].split(","))
    assert got8 == site8, (got8, site8)
    assert got4 == site8[:4], (
        f"185 的 4 层表 {got4} 不是 8 层站点表 {site8} 的前缀 —— "
        f"4 层表的出处是 handoff 文档的文字记录 `compress_ratios[0,4,128,4]`，"
        f"8 层表的出处是 yaml 逐字；两者本应自洽，不自洽说明至少一处读错了。")
    assert got4 == (0, 4, 128, 4), got4


def test_preset_cycle_and_site_table_really_differ_in_layer_mix():
    """把缺陷本体钉成回归：预设循环与站点表的**层型计数**必须不同（3/3/2 vs 1/4/3）。

    这条不是自洽检查 —— 它是「为什么这件事值得修」的判据：两张表若层型计数相同，
    错配就只是命名问题；正因为不同，锚点才是在给**另一份模型**打分。
    真机三种层型的驻留实测彼此不同（167/2026-07-29 逐层直测：r0 2235.1 / r4 2341.2 /
    r128 2116.1 MiB，`docs/census_fix_residual_carrier_2026-07-29.md` §2）。"""
    from cost_eval.presets import _v4_compress_ratios
    cycle = _v4_compress_ratios(8)
    site = _yaml_llm("dsv4h_fused_pp4_recomp.yaml").csa_compress_ratios
    def mix(t):
        return {r: t.count(r) for r in (0, 4, 128)}
    assert mix(cycle) == {0: 3, 4: 3, 128: 2}, mix(cycle)
    assert mix(site) == {0: 1, 4: 4, 128: 3}, mix(site)
    assert mix(cycle) != mix(site)
