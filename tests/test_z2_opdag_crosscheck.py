"""P2-02（closure-wave3，2026-07-15）：opdag 进主链路做一致性校验。

`cost_eval/opdag/` 从纯离线工具升级为**生产链路的可选约束**——Evaluator/独立入口用 opdag 从真
mindformers 源抽出的重算子名册交叉校验手写 LayerSpec，漂移 warn/fail。默认关 → 评估逐字节不变。
（Z2 agent 连接中断未写测试，主控补齐。）
"""
import dataclasses
import os

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.llm_config import LLMConfig
from cost_eval.presets import deepseek_v3
from cost_eval.opdag import crosscheck as _cc
from cost_eval.opdag.crosscheck import (
    OpdagConsistencyError, default_mf_root, hand_category, opdag_category,
    validate_against_opdag,
)

_HAS_SOURCE = os.path.isdir(default_mf_root())
_needs_source = pytest.mark.skipif(not _HAS_SOURCE, reason="mindformers 源不可用（CI）")


def test_missing_source_skips_gracefully():
    """缺 mindformers 源 → available=False、ok=True（无从校验，绝不 fail）。"""
    spec = build_llm_spec(deepseek_v3(4))
    rep = validate_against_opdag(spec, mf_root=r"Z:\no\such\path", warn=False)
    assert rep.available is False and rep.ok is True


@_needs_source
def test_dsv3_handwritten_matches_opdag_no_drift():
    """DSv3 手写 LayerSpec 与真 mindformers 源抽出的 op 名册一致（0 漂移）——闭环主张。"""
    spec = build_llm_spec(deepseek_v3(4))
    rep = validate_against_opdag(spec, warn=False)
    assert rep.available and rep.ok and not rep.findings
    fams = {c.family for c in rep.covered}
    assert "mla_attn" in fams and "moe_experts" in fams        # MLA 段 + MoE 专家段都被校验


@_needs_source
def test_drift_detected_when_handwritten_spec_corrupted():
    """人为在 MLA 段删一个 matmul（qb_w 投影）→ 实际 delta 偏离预期 → 报出漂移（真交叉校验）。"""
    spec = build_llm_spec(deepseek_v3(4))
    # 找含 linear_qb 的 mla 层，删掉该 matmul op（模拟手写 builder 掉一个投影）
    corrupted = False
    for ls in spec.layer_specs.values():
        names = [o.name for o in ls.ops]
        if "linear_qb" in names and "linear_kvb" in names:
            ls.ops = tuple(o for o in ls.ops if o.name != "linear_qb")
            corrupted = True
            break
    assert corrupted, "未找到可破坏的 MLA 层"
    rep = validate_against_opdag(spec, warn=False)
    assert not rep.ok and rep.findings                          # 漂移被报出
    assert any(f.category == "linear" for f in rep.findings)


@_needs_source
def test_strict_mode_raises_on_drift():
    """strict=True：漂移即 raise OpdagConsistencyError。"""
    spec = build_llm_spec(deepseek_v3(4))
    for ls in spec.layer_specs.values():
        if any(o.name == "linear_qb" for o in ls.ops):
            ls.ops = tuple(o for o in ls.ops if o.name != "linear_qb")
            break
    with pytest.raises(OpdagConsistencyError):
        validate_against_opdag(spec, strict=True, warn=False)


@_needs_source
def test_extraction_failure_trips_strict(monkeypatch):
    """F6：已声明覆盖族的 opdag 提取失败 ≠ 合法无对应——它意味着交叉校验**无从验证**该族，
    绝不能静默放行。注入 MLA census 抛异常 → 非 strict 报告 ok=False 且 extraction_failures 记录
    了 mla_attn（不是塞进 uncovered 假装没事）；strict=True 直接 raise OpdagConsistencyError。"""
    spec = build_llm_spec(deepseek_v3(4))

    def boom(_root):
        raise RuntimeError("injected extractor failure")

    monkeypatch.setattr(_cc._MLA_CORR, "census_fn", boom)

    # 非 strict：不 raise，但报告如实反映提取失败（ok=False、extraction_failures 非空、含 mla_attn）
    rep = validate_against_opdag(spec, warn=False)
    assert rep.available is True
    assert rep.ok is False                                     # 提取失败也让 ok 变 False（不只看 findings）
    assert rep.extraction_failures                             # 提取失败被单列，而非混进 uncovered
    assert any(fam == "mla_attn" for (_lt, fam, _err) in rep.extraction_failures)
    # 诚实边界：提取失败不该被降级成 “合法无对应” 的 uncovered 记录
    assert not any("mla_attn" in note for (_lt, note) in rep.uncovered)
    # summary 要点名提取失败（可读的失败原因）
    assert "mla_attn" in rep.summary()

    # strict：提取失败即 raise（这正是此前的假绿路径）
    with pytest.raises(OpdagConsistencyError) as ei:
        validate_against_opdag(spec, strict=True, warn=False)
    assert "mla_attn" in str(ei.value)


@_needs_source
def test_legitimate_no_correspondence_does_not_trip_strict():
    """F6 对照：全 GQA/dense（无 MLA、无 MoE）的层**合法地**没有 opdag 提取源 → 记为 uncovered，
    这不是提取失败，strict 也绝不能因此 raise（否则会把 embedding/lm_head/gqa 误伤为漂移）。"""
    cfg = LLMConfig(
        num_layers=2, hidden_size=8, num_attention_heads=2, num_query_groups=2,
        vocab_size=16, seq_length=8, head_dim=4, attn_type="gqa", ffn_hidden_size=16,
    )
    spec = build_llm_spec(cfg)
    rep = validate_against_opdag(spec, strict=True, warn=False)  # strict 下不 raise
    assert rep.available and rep.ok is True
    assert not rep.covered                                     # 无 MLA/MoE 段可覆盖
    assert not rep.extraction_failures                         # 合法无对应 ≠ 提取失败
    assert rep.uncovered                                       # 全部作为 uncovered log（不 trip strict）


def test_evaluator_default_off_does_not_touch_opdag(monkeypatch):
    """Evaluator 默认 validate_opdag=False → evaluate() 不接触 opdag（默认路径逐字节不变）。"""
    from cost_eval.report import Evaluator
    from cost_eval.specs import (HardwareSpec, OptimizerSpec, ParallelConfig,
                                 RecomputeSpec, SwapSpec)
    spec = build_llm_spec(deepseek_v3(4))
    ev = Evaluator(spec, ParallelConfig(dp_shard=2), OptimizerSpec.adamw(),
                   HardwareSpec(64 * 2**30), RecomputeSpec(), SwapSpec())
    assert ev.validate_opdag is False
    # 默认关 → evaluate 正常出结果（不因缺/有源、不因 opdag 改变数值）
    rep = ev.evaluate()
    assert rep.per_stage[0].peak_bytes > 0


def test_category_maps_are_disjoint_and_documented():
    """类别映射自洽：手写 matmul→linear、moe_gemm→linear_grouped、norm→norm、flash→attention。"""
    from cost_eval.model_spec import OpSpec, OpType, TensorRef
    t = TensorRef("x", ("S", "B", "H"))
    assert hand_category(OpSpec("m", OpType.MATMUL, [t], t)) == "linear"
    assert hand_category(OpSpec("e", OpType.MOE_GEMM, [t], t)) == "linear_grouped"
    assert hand_category(OpSpec("n", OpType.NORM, [t], t)) == "norm"
    assert hand_category(OpSpec("f", OpType.FLASH_ATTN, [t], t)) == "attention"
    assert hand_category(OpSpec("add", OpType.ELEMENTWISE, [t], t)) is None   # 残差 add 不比
    assert opdag_category(type("N", (), {"op": "View"})()) is None            # 结构算子剔除
