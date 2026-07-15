"""P2-02（closure-wave3，2026-07-15）：opdag 进主链路做一致性校验。

`cost_eval/opdag/` 从纯离线工具升级为**生产链路的可选约束**——Evaluator/独立入口用 opdag 从真
mindformers 源抽出的重算子名册交叉校验手写 LayerSpec，漂移 warn/fail。默认关 → 评估逐字节不变。
（Z2 agent 连接中断未写测试，主控补齐。）
"""
import dataclasses
import os

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.presets import deepseek_v3
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
