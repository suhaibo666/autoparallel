"""**`ResolvedLayer` 适配器契约**（`cost_eval/liveness/contract.py`）的契约测试（2026-07-25）。

目的：**在抽取侧的生产者（`cost_eval/opdag/to_resolved.py`）存在之前**，先证明这条缝是真的 ——

  * **正向**：现役**手写** census 路径（`hand_spec`）逐层满足契约的全部硬规则，且覆盖真 DSv4-Flash
    配置（pp4/dp2/ep2、fused/unfused 两支）与全部 preset（dsv3 / dsv4 / llama / qwen2 / mixtral）
    × 多组并行度 → 契约不是给抽取侧量身定做的一套空话，它对现有实现就成立。
  * **反向**：逐条规则各造一个**违约样本**，校验器必须逐条抓到（否则规则是装饰性的）。
  * **完整性**：一个**只带契约字段**的外部生产者喂进 `simulate_liveness`，DSv4-Flash pp4 的
    per-stage 峰值与手写路径**逐字节相同** → 契约字段集**够用**（不是"能建图但算不出显存"）。

两个已知硬点（`docs/opdag_coverage_assessment_2026-07-25.md` §7.1/§7.2 实测）被编成可执行规则：
  1. 裸抽取 DAG 的符号 shape 全是 `?` → `consumer` 给 `total_bytes=0` ⇒ **B1/B2/B3**；
  2. 权重被当 activation save 计（`FFNGroupedGEMM` 236 MiB 里 `w1`+`w2`=88 MiB）⇒ **W1/W2/W4**
     + `validate_param_census`（**B4**）。
"""
from __future__ import annotations

import dataclasses
import os

import pytest

from cost_eval import presets
from cost_eval.build_llm import build_llm_spec
from cost_eval.liveness import resolve_graph
from cost_eval.liveness.contract import (CONTRACT_RULES, layer_activation_bytes,
                                         layer_entry_names, layer_param_bytes,
                                         layer_total_bytes,
                                         validate_param_census,
                                         validate_resolved_graph,
                                         validate_resolved_layer)
from cost_eval.parallel_model import ParallelModel
from cost_eval.specs import ParallelConfig

#: 167 A/B（2026-07-25）真跑的两份 launcher 配置（固化进仓；只差 apply_dsa_kernel_fusion）。
AB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "analysis", "realmachine", "ab_fusion_2026-07-25")


def _dsv4_flash_bundle(fused: bool):
    """真 DSv4-Flash 配置 → **bundle-direct** `(bundle, ModelSpec)`（pp4/dp2/ep2、seq4096、全重算）。

    刻意不复用 `tools/liveness_ab_validate.py` —— 契约测试不该依赖工具层（分层反了）。"""
    import warnings

    import yaml

    import serve_explorer as S
    from cost_eval.configs.from_mindformers import from_mindformers_dict
    warnings.simplefilter("ignore")
    fn = "dsv4h_%s_pp4_recomp.yaml" % ("fused" if fused else "unfused")
    with open(os.path.join(AB_DIR, fn), encoding="utf-8") as fh:
        mf = yaml.safe_load(fh)
    m = mf["model"]
    if not m.get("qk_nope_head_dim"):     # launcher 省略；mindformers 由 head_dim − rope 导出
        m["qk_nope_head_dim"] = int(m["head_dim"]) - int(m["qk_rope_head_dim"])
    mf2, _ = S._mf_adapt(mf)
    S._materialize_nested_offset(mf2, [])
    b = from_mindformers_dict(mf2)
    return b, build_llm_spec(b.llm)


def _pm_of(bundle, spec):
    p = bundle.parallel
    world = p.dp_replicate * p.dp_shard * p.cp * p.tp * p.pp
    return ParallelModel(p, spec.dims.n_layers, world)


def _graph(spec, *, pp=1, tp=1, ep=1, dp_shard=1, cp=1, m=2, source="hand_spec"):
    pc = ParallelConfig(pp=pp, tp=tp, ep=ep, dp_shard=dp_shard, cp=cp,
                        num_microbatches=m, sequence_parallel=(tp > 1))
    world = pc.dp_replicate * pc.dp_shard * pc.cp * pc.tp * pc.pp
    return resolve_graph(spec, ParallelModel(pc, spec.dims.n_layers, world), source)


# ---------------------------------------------------------------------------
# 正向：手写路径满足契约
# ---------------------------------------------------------------------------

_PRESETS = {
    "dsv3": lambda: presets.deepseek_v3(4),
    "dsv4_fused": lambda: presets.deepseek_v4(4),
    "dsv4_unfused": lambda: dataclasses.replace(presets.deepseek_v4(4), dsa_fused=False),
    "llama": lambda: presets.llama(num_layers=2),
    "qwen2": lambda: presets.qwen2(num_layers=2),
    "mixtral": lambda: presets.mixtral(num_layers=2),
}


@pytest.mark.parametrize("name", sorted(_PRESETS))
@pytest.mark.parametrize("par", [
    {"pp": 1}, {"pp": 2, "tp": 2, "ep": 2, "dp_shard": 2}, {"pp": 4, "ep": 2, "dp_shard": 2}])
def test_hand_spec_satisfies_contract(name, par):
    """`hand_spec` 的每一层都过全部硬规则 —— 缝对现役实现成立（不是为抽取侧量身定做）。"""
    spec = build_llm_spec(_PRESETS[name]())
    v = validate_resolved_graph(_graph(spec, **par))
    assert not v, "契约违约 %d 条：\n  %s" % (len(v), "\n  ".join(str(x) for x in v[:20]))


@pytest.mark.parametrize("fused", [True, False])
def test_real_dsv4_flash_config_satisfies_contract(fused):
    """真 DSv4-Flash 167 A/B 配置（pp4/dp2/ep2、seq4096、fused/unfused）逐层过契约。"""
    if not os.path.isdir(AB_DIR):
        pytest.skip("缺 A/B base yaml 目录: %s" % AB_DIR)
    b, spec = _dsv4_flash_bundle(fused)
    g = resolve_graph(spec, _pm_of(b, spec), "hand_spec")
    v = validate_resolved_graph(g)
    assert not v, "契约违约 %d 条：\n  %s" % (len(v), "\n  ".join(str(x) for x in v[:20]))
    # 顺带把两个"抽取侧必须交出来"的量落成可比数字（A/B 时逐层比对）。
    for st in sorted(g.stages):
        for layer in g.stages[st]:
            assert layer_total_bytes(layer) > 0
            assert layer_param_bytes(layer) + layer_activation_bytes(layer) \
                   == layer_total_bytes(layer), "param/activation 字节必须互补且不重叠"
            assert len(layer_entry_names(layer)) <= 4, (
                f"{layer.layer_type} 图入口过多：{sorted(layer_entry_names(layer))}"
                " —— 抽取侧漏标的权重会冒充图入口出现在这里")


def test_param_and_activation_bytes_are_disjoint_and_nonzero():
    """`layer_param_bytes` / `layer_activation_bytes` 互补 —— 权重与激活不共用一个字节。"""
    spec = build_llm_spec(presets.deepseek_v4(4))
    g = _graph(spec, pp=2, ep=2, dp_shard=2)
    seen_params = False
    for st in sorted(g.stages):
        for layer in g.stages[st]:
            p, a, t = (layer_param_bytes(layer), layer_activation_bytes(layer),
                       layer_total_bytes(layer))
            assert p + a == t and a > 0
            seen_params |= p > 0
    assert seen_params, "全图零权重？"


def test_param_census_gate_catches_missing_weights():
    """**B4 闸门**：param 字节与参考值不符即违约，且差额方向可读（负 = 权重漏进 params）。"""
    spec = build_llm_spec(presets.deepseek_v4(4))
    layer = next(l for l in _graph(spec).stages[0] if l.layer_type.startswith("dsv4hyb"))
    ref = layer_param_bytes(layer)
    assert validate_param_census(layer, ref) == ()
    short = validate_param_census(layer, ref + 88 * 2 ** 20)
    assert len(short) == 1 and short[0].rule == "B4"
    assert "负差" in short[0].detail          # 抽取侧漏权重的方向
    assert validate_param_census(layer, ref + 1, tol_bytes=1) == ()


# ---------------------------------------------------------------------------
# 反向：逐条规则的违约样本必须被抓到
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class T:
    """可变的契约张量（供造违约样本）。"""
    name: str = "x"
    local_numel: int = 16
    dtype_bytes: int = 2
    is_weight: bool = False
    detached: bool = False
    is_expert: bool = False
    dim0: int = 4
    pin_under_recompute: bool = False


@dataclasses.dataclass
class O:
    name: str = "op"
    type: str = "matmul"
    inputs: tuple = ()
    output: T = None
    params: tuple = ()
    saves: tuple = ()
    workspace_bytes: int = 0
    bwd_scratch_bytes: int = 0
    collectives: tuple = ()


@dataclasses.dataclass
class L:
    layer_id: int = 0
    layer_type: str = "toy"
    ops: tuple = ()


def _good_layer() -> L:
    """一个最小合法层：`w @ x -> y`，`x` 被 save（norm 式），带一个内部中间量。"""
    x, w, y = T("x"), T("w", is_weight=True, dim0=8), T("y")
    mid = T("mid")                       # 只在 saves 里出现 → internal
    op = O(name="mm", inputs=(x, w), output=y, params=(w,), saves=(x, mid))
    return L(ops=(op,))


def _rules_of(layer, **kw):
    return sorted({v.rule for v in validate_resolved_layer(layer, **kw)})


def test_good_layer_is_clean():
    assert validate_resolved_layer(_good_layer()) == ()


def test_rule_B1_unresolved_shape_is_rejected():
    """**B1**：`local_numel=0`（抽取侧 shape='?' 的直接后果）必须被判死。"""
    lay = _good_layer()
    lay.ops[0].inputs[0].local_numel = 0
    assert "B1" in _rules_of(lay)


def test_rule_B2_illegal_dtype_bytes():
    lay = _good_layer()
    lay.ops[0].output.dtype_bytes = 3
    assert "B2" in _rules_of(lay)


def test_rule_B3_all_zero_layer_is_rejected():
    """**B3**：裸抽取 DAG 的典型征状 `total_bytes=0` 整层判死。"""
    lay = _good_layer()
    for t in (*lay.ops[0].inputs, lay.ops[0].output, *lay.ops[0].saves):
        t.local_numel = 0
    r = _rules_of(lay)
    assert "B1" in r and "B3" in r


def test_rule_W2_weight_in_saves_is_rejected():
    """**W2**：权重出现在 `saves` 里 = FFNGroupedGEMM 那 88/236 MiB 的病根。"""
    lay = _good_layer()
    w = next(t for t in lay.ops[0].inputs if t.is_weight)
    lay.ops[0].saves = lay.ops[0].saves + (w,)
    v = validate_resolved_layer(lay)
    assert [x.rule for x in v] == ["W2"] and "88 MiB" in v[0].detail


def test_rule_W1_non_weight_in_params():
    lay = _good_layer()
    lay.ops[0].params = (T("bogus"),)
    assert "W1" in _rules_of(lay)


def test_rule_W3_and_W4_weight_as_output():
    lay = _good_layer()
    lay.ops[0].output.is_weight = True
    r = _rules_of(lay)
    assert "W3" in r and "W4" in r


def test_rule_W5_same_name_flips_is_weight():
    lay = _good_layer()
    x2 = T("x", is_weight=True, dim0=8)
    op2 = O(name="mm2", inputs=(x2,), output=T("z"), params=(x2,))
    lay.ops = lay.ops + (op2,)
    assert "W5" in _rules_of(lay)


def test_rule_W6_detached_weight():
    lay = _good_layer()
    next(t for t in lay.ops[0].inputs if t.is_weight).detached = True
    assert "W6" in _rules_of(lay)


def test_rule_S1_save_not_touched_by_op():
    """**S1**：saves 必须挂在**会在反向读它**的那个 op 上（逐 op saves，非全图扁平表）。"""
    a, b = T("a"), T("b")
    op1 = O(name="o1", inputs=(a,), output=b)
    op2 = O(name="o2", inputs=(b,), output=T("c"), saves=(a,))   # a 不是 o2 的操作数
    assert "S1" in _rules_of(L(ops=(op1, op2)))


def test_rule_S3_same_name_different_bytes():
    lay = _good_layer()
    op2 = O(name="mm2", inputs=(T("y", local_numel=999),), output=T("z"))
    lay.ops = lay.ops + (op2,)
    assert "S3" in _rules_of(lay)


@pytest.mark.parametrize("attr", ["detached", "is_weight", "local_numel", "dtype_bytes", "name"])
def test_rule_T1_missing_core_field(attr):
    """**T1**：核心字段缺一不可；尤其 `detached` **不接受** getattr 兜底。"""
    lay = _good_layer()
    obj = lay.ops[0].inputs[0]
    stripped = type("Stripped", (), {k: v for k, v in vars(obj).items() if k != attr})()
    lay.ops[0].inputs = (stripped, lay.ops[0].inputs[1])
    assert "T1" in _rules_of(lay, check_derived=False)


@pytest.mark.parametrize("attr", ["is_expert", "dim0", "pin_under_recompute"])
def test_rule_T2_missing_bucket_passthrough(attr):
    """**T2**：漏这三项 = "图能建、显存算不出来"（`structure_mem.py:275` 直读 `is_expert`）。"""
    lay = _good_layer()
    obj = lay.ops[0].inputs[0]
    stripped = type("Stripped", (), {k: v for k, v in vars(obj).items() if k != attr})()
    lay.ops[0].inputs = (stripped, lay.ops[0].inputs[1])
    assert "T2" in _rules_of(lay, check_derived=False)


def test_rule_O3_negative_workspace():
    lay = _good_layer()
    lay.ops[0].workspace_bytes = -1
    assert "O3" in _rules_of(lay)


def test_rule_L1_empty_ops():
    assert "L1" in _rules_of(L(ops=()))


def test_rule_G2_detached_tensor_kept_for_backward():
    """**G2**：派生自洽 —— detached 张量不得进 `kept_for_backward`（grad 可达性被切断）。

    这里用一个"标了 detached 却仍被声明成 saves"的样本；`build_layer_graph` 会按 grad 可达性
    把它剔除 → 契约通过（**规则是导出的，不靠特例**）。真正会触发 G2 的是生产者绕过
    grad 可达性直接塞 `kept_for_backward` —— 那是 liveness 内部不可达路径，故此处校验
    "机制在"而非人为造一个不可能的图。"""
    lay = _good_layer()
    lay.ops[0].saves[1].detached = True          # 内部中间量 mid 标 detach
    assert validate_resolved_layer(lay) == ()
    from cost_eval.liveness import build_layer_graph
    lg = build_layer_graph(lay)
    assert "mid" not in lg.kept_for_backward
    assert lg.tensors["mid"].detached and not lg.tensors["mid"].requires_grad


def test_rule_G3_weights_are_gradient_roots():
    lay = _good_layer()
    from cost_eval.liveness import build_layer_graph
    lg = build_layer_graph(lay)
    assert all(t.requires_grad for t in lg.tensors.values() if t.is_weight)


def test_every_documented_rule_is_reachable():
    """`CONTRACT_RULES` 的每条编号都在校验器里真实出现（不许有写在文档里、代码不查的规则）。"""
    import inspect
    import re

    from cost_eval.liveness import contract as C
    src = inspect.getsource(C)
    emitted = set(re.findall(r'Violation\(\s*"([A-Z]\d)"', src))
    for rule in CONTRACT_RULES:
        assert rule in emitted, f"规则 {rule} 只在文档里，校验器不查"
    for rule in emitted:                      # 反向：不许有未文档化的规则
        assert rule in CONTRACT_RULES, f"校验器抛了未文档化的规则 {rule}"


def test_contract_field_set_is_sufficient_on_real_dsv4_flash():
    """**完整性**：一个只带契约字段的外部生产者，在真 DSv4-Flash pp4 上给出**逐字节相同**的
    per-stage 峰值 → 契约字段集**够用**（不是"图能建、显存算不出来"）。

    这条测试挖出过一个真缺口：只按 `liveness/graph.py` 读的字段建契约时，
    `structure_mem.py:275` 的 `w.is_expert` 直读会 `AttributeError` —— 因为 `simulate_liveness`
    只接管**激活**四桶，其余非 liveness 桶仍走 `structure_mem`/`static_mem`。故 T2 是硬要求。"""
    from dataclasses import dataclass, field

    from cost_eval.liveness import (register_graph_source, simulate_liveness,
                                    unregister_graph_source)
    if not os.path.isdir(AB_DIR):
        pytest.skip("缺 A/B base yaml 目录: %s" % AB_DIR)

    @dataclass(frozen=True)
    class FT:                                  # 恰好契约的 8 个字段，一个不多
        name: str
        local_numel: int
        dtype_bytes: int
        is_weight: bool
        detached: bool
        is_expert: bool
        dim0: int
        pin_under_recompute: bool

    @dataclass(frozen=True)
    class FO:
        name: str
        type: str
        inputs: tuple
        output: FT
        params: tuple
        saves: tuple
        workspace_bytes: int
        bwd_scratch_bytes: int
        norm_kind: str          # 契约 O4（2026-07-29）：norm-fp32 抬升按它分辨

    @dataclass(frozen=True)
    class FL:
        layer_id: int
        layer_type: str
        ops: tuple

    @dataclass(frozen=True)
    class FG:
        stages: dict = field(default_factory=dict)

    def _t(t):
        return FT(t.name, t.local_numel, t.dtype_bytes, t.is_weight, t.detached,
                  t.is_expert, t.dim0, t.pin_under_recompute)

    def _foreign(model_spec, pm):
        g = resolve_graph(model_spec, pm, "hand_spec")
        return FG(stages={st: [FL(l.layer_id, l.layer_type,
                                 tuple(FO(op.name,
                                          str(getattr(op.type, "value", op.type)),
                                          tuple(_t(x) for x in op.inputs), _t(op.output),
                                          tuple(_t(x) for x in op.params),
                                          tuple(_t(x) for x in op.saves),
                                          op.workspace_bytes, op.bwd_scratch_bytes,
                                          op.norm_kind)
                                       for op in l.ops))
                             for l in layers]
                         for st, layers in g.stages.items()})

    b, spec = _dsv4_flash_bundle(False)          # unfused 支（小算子链最长，最能暴露漏项）
    register_graph_source("_contract_foreign", _foreign)
    try:
        args = (spec, b.parallel, b.optimizer, b.hardware, b.recompute, b.swap)
        kw = dict(record_timeline=True, grad_mode="chain2")
        hand = simulate_liveness(*args, graph_source="hand_spec", **kw)
        fore = simulate_liveness(*args, graph_source="_contract_foreign", **kw)
        assert [p.peak_bytes for p in hand.per_stage] == [p.peak_bytes for p in fore.per_stage]
        assert [p.peak_substep for p in hand.per_stage] == \
               [p.peak_substep for p in fore.per_stage]
        assert [p.max_recompute_working_set for p in hand.per_stage] == \
               [p.max_recompute_working_set for p in fore.per_stage]
        # 外部生产者的图本身也必须过契约（自证：契约不是只对 ShapeEval 成立）
        assert validate_resolved_graph(_foreign(spec, _pm_of(b, spec))) == ()
    finally:
        unregister_graph_source("_contract_foreign")


def test_tied_weight_input_without_params_is_legal():
    """tie（权重共享）不是违约：`tie_word_embeddings=True` 时 lm_head 复用 `emb_w`，
    故意 `params=[]` 以免 vocab×H 持久量重复计（`layers/head.py:84-88`）。

    契约只要求它 `is_weight=True`（liveness 据此跳过激活记账），**不**要求出现在 `params` 里。"""
    spec = build_llm_spec(presets.qwen2(num_layers=2))
    g = _graph(spec)
    head = next(l for l in g.stages[0] if l.layer_type == "lm_head")
    op = next(o for o in head.ops if o.name == "lm_head")
    tied = [t for t in op.inputs if t.is_weight and t.name not in {p.name for p in op.params}]
    assert tied, "qwen2 应有 tie 的 emb_w 作为 lm_head 的权重输入"
    assert validate_resolved_layer(head) == ()
