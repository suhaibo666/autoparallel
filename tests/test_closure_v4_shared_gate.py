"""闭环审计 P1-01（`analysis/closure_audit_verification_2026-07-15.md` §4.2）回归。

审计发现：`moe_shared_gate=True` 时 `build_shared_expert_ops` 追加的 `shared_gate` op 产出的
`sh_gate [S,B,1]` **无任何消费者**（探针对 DSv4 三种 MoE 层 + MTP 均得 consumers=[]），真正合流
仍是 `moe_add = comb + sh_o`（**未乘 gate**）——只补了参数数目，没表达
`sigmoid(gate)·shared_output` 的数据流与 backward 保存生命周期。

真机语义（mindformers `shared_experts.py:56-64` use_shared_expert_gating）：
`shared_out_gated = sigmoid(gate_logits) · shared_expert_out`，`gate_logits = Linear(H→1)(hidden)`。

本组测试：
  - gate ON：`sh_gate` 被 `shared_gate_mul` 消费、gated 输出 `sh_o_gated` 被 `moe_add` 消费
    （数据流闭合，不再是孤立叶节点）；`moe_add` 消费 `sh_o_gated` 而非裸 `sh_o`；
    `shared_gate_mul` 保存 backward 所需的 `[sh_o, sh_gate]`（sigmoid·mul 反向生命周期）。
  - gate OFF（DSv3 默认 `moe_shared_expert_gating=False`）：无 shared_gate/shared_gate_mul，
    `moe_add` 仍消费 `sh_o`，op 序列/saves 逐字节不变（golden 守卫，见 test_dsv3_golden）。
"""
import dataclasses

from cost_eval.build_llm import build_llm_spec
from cost_eval.layers.ffn import build_moe_merge_op, build_shared_expert_ops
from cost_eval.llm_config import to_dimtable
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.report import Evaluator
from cost_eval.specs import (HardwareSpec, OptimizerSpec, ParallelConfig,
                             RecomputeSpec, SwapSpec)

G = 2 ** 30


# ── 图探针（与审计探针 parameter_evidence 同法：某 tensor name 作为下游 op input 出现即为消费者）──
def _consumers(layer_spec, tensor_name):
    return [op.name for op in layer_spec.ops
            if any(inp.name == tensor_name for inp in op.inputs)]


def _find_op(layer_spec, op_name):
    for op in layer_spec.ops:
        if op.name == op_name:
            return op
    return None


def _gated_layer_specs(spec):
    """含 shared_gate op 的层（gate on 时的 MoE 层 + DSv4 的 MTP 层）。"""
    return {lt: ls for lt, ls in spec.layer_specs.items()
            if _find_op(ls, "shared_gate") is not None}


# ── gate ON：数据流闭合（DSv3-like：MLA + MoE + shared，无 mHC/MTP 干扰）─────────────
def test_gate_on_dataflow_closed_dsv3():
    spec = build_llm_spec(dataclasses.replace(deepseek_v3(4), moe_shared_expert_gating=True))
    gated = _gated_layer_specs(spec)
    assert gated, "gate on 时应存在含 shared_gate 的 MoE 层"
    for lt, ls in gated.items():
        # ① sh_gate 有消费者 shared_gate_mul（审计中 consumers=[]）
        assert "shared_gate_mul" in _consumers(ls, "sh_gate"), \
            f"{lt}: sh_gate 无 shared_gate_mul 消费者（仍为孤立叶节点）"
        # ② gated 输出 sh_o_gated 有消费者 moe_add
        assert "moe_add" in _consumers(ls, "sh_o_gated"), \
            f"{lt}: sh_o_gated 未流入 moe_add（gate 未真正合流）"
        # ③ merge 消费 gated 输出而非裸 sh_o
        merge_inputs = {t.name for t in _find_op(ls, "moe_add").inputs}
        assert "sh_o_gated" in merge_inputs and "sh_o" not in merge_inputs, \
            f"{lt}: moe_add 应消费 sh_o_gated 而非 sh_o，实为 {merge_inputs}"


# ── gate ON：审计 §4.2 原样复现（DSv4，mHC + MTP；此前"三种 MoE 层 + MTP"均 consumers=[]）──
def test_gate_on_dataflow_closed_dsv4_audit_repro():
    spec = build_llm_spec(dataclasses.replace(deepseek_v4(4), moe_shared_expert_gating=True))
    gated = _gated_layer_specs(spec)
    assert gated, "DSv4 gate on 时应有含 shared_gate 的层（MoE 层 + MTP）"
    for lt, ls in gated.items():
        # mHC 只重命名 SP [S,B,H] 残差承载张量；sh_gate/sh_o/sh_o_gated（[S,B,1] / partial=tp）不受影响
        assert "shared_gate_mul" in _consumers(ls, "sh_gate"), \
            f"{lt}: DSv4 sh_gate 仍无消费者"
        assert "moe_add" in _consumers(ls, "sh_o_gated"), \
            f"{lt}: DSv4 sh_o_gated 未流入 moe_add"


# ── gate ON：backward 保存生命周期（sigmoid·mul 反向需 shared 输出 + 门 logits）──────────
def test_gate_on_shared_gate_mul_saves_backward_tensors():
    spec = build_llm_spec(dataclasses.replace(deepseek_v3(4), moe_shared_expert_gating=True))
    gated = _gated_layer_specs(spec)
    assert gated
    for lt, ls in gated.items():
        mul = _find_op(ls, "shared_gate_mul")
        assert mul is not None, f"{lt}: 缺 shared_gate_mul op"
        saved = {t.name for t in mul.saves}
        # d/d(sh_o)=sigmoid(sh_gate)；d/d(sh_gate)=sh_o·sigmoid'(sh_gate) → 二者皆须驻留至反向
        assert "sh_o" in saved and "sh_gate" in saved, \
            f"{lt}: shared_gate_mul 应保存 [sh_o, sh_gate]，实为 {saved}"


# ── gate ON：resolve + evaluate 冒烟（图闭合后仍可正常求解出峰值）──────────────────────
def test_gate_on_resolves_and_evaluates():
    spec = build_llm_spec(dataclasses.replace(deepseek_v3(4), moe_shared_expert_gating=True))
    rep = Evaluator(spec, ParallelConfig(dp_shard=2, sequence_parallel=True),
                    OptimizerSpec.adamw(), HardwareSpec(64 * G),
                    RecomputeSpec(), SwapSpec()).evaluate()
    assert rep.per_stage[0].peak_bytes > 0


# ── gate OFF：golden 不变守卫（DSv3 默认逐字节不变）──────────────────────────────────
def test_gate_off_no_gate_ops_and_merge_consumes_sh_o():
    spec = build_llm_spec(deepseek_v3(4))   # 默认 moe_shared_expert_gating=False
    for lt, ls in spec.layer_specs.items():
        op_names = [op.name for op in ls.ops]
        assert "shared_gate" not in op_names, f"{lt}: gate off 不应有 shared_gate"
        assert "shared_gate_mul" not in op_names, f"{lt}: gate off 不应有 shared_gate_mul"
        merge = _find_op(ls, "moe_add")
        if merge is not None:
            merge_inputs = {t.name for t in merge.inputs}
            assert "sh_o" in merge_inputs and "sh_o_gated" not in merge_inputs, \
                f"{lt}: gate off 时 moe_add 应消费 sh_o，实为 {merge_inputs}"


# ── 直接单元测试：builder 在 DimTable.moe_shared_gate on/off 下的 op 结构 ─────────────
def test_build_shared_expert_ops_gate_off_byte_identical():
    d_off = to_dimtable(deepseek_v3(4))
    assert d_off.moe_shared_gate is False
    names = [op.name for op in build_shared_expert_ops(d_off)]
    assert names == ["shared_fc1", "shared_swiglu", "shared_fc2"]


def test_build_shared_expert_ops_gate_on_adds_gate_and_mul():
    d_on = dataclasses.replace(to_dimtable(deepseek_v3(4)), moe_shared_gate=True)
    ops = build_shared_expert_ops(d_on)
    names = [op.name for op in ops]
    assert names == ["shared_fc1", "shared_swiglu", "shared_fc2",
                     "shared_gate", "shared_gate_mul"]
    mul = ops[-1]
    assert [t.name for t in mul.inputs] == ["sh_o", "sh_gate"]
    assert mul.output.name == "sh_o_gated"
    assert mul.output.partial == "tp"   # 与 sh_o 同分布（行并行部分和）


def test_build_moe_merge_op_gate_off_consumes_sh_o():
    d_off = to_dimtable(deepseek_v3(4))
    op = build_moe_merge_op(d_off)
    assert op.name == "moe_add"
    assert [t.name for t in op.inputs] == ["comb", "sh_o"]
    assert list(op.saves) == []          # 线性 add，零激活字节（golden 守卫）


def test_build_moe_merge_op_gate_on_consumes_gated():
    d_on = dataclasses.replace(to_dimtable(deepseek_v3(4)), moe_shared_gate=True)
    op = build_moe_merge_op(d_on)
    assert [t.name for t in op.inputs] == ["comb", "sh_o_gated"]
    assert list(op.saves) == []
