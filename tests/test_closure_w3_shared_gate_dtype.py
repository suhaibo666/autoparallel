"""闭环审计第四轮 §F3（analysis/closure_audit_v2_verification_2026-07-15.md）回归：
shared-expert gate 权重 dtype 应为 fp32。

真机 mindformers `shared_experts_gate = Dense(in_channels=H, out_channels=1, has_bias=False,
dtype=self.router_dense_type)`（shared_experts.py:58-62），`router_dense_type = config.moe_router_dtype`
（:55），`moe_router_dtype` 默认 `"float32"`（transformer_config.py:1791-1796）——故 gate Dense
权重是 fp32(4B)。此前 `sh_gate_w` 未指定 dtype_bytes → 解析后默认 2B/BF16（DimTable.dtype_bytes）。
同文件 `router_w` 已是 fp32(4B)，gate 应对齐。

范围边界：本组只钉 **sh_gate_w 权重 dtype=4** 这一确定项。审计 §F3 另提及 runtime 有「gate Dense
前把 hidden cast 到 router dtype、sigmoid 后 cast 回 compute dtype」的 FP32 hidden cast 生命周期
（shared_experts.py:70-71 `self.cast(...)`）——那个更深的 cast 建模未做，属 partial，不在本次范围。
"""
import dataclasses

from cost_eval.layers.ffn import build_moe_ffn_ops, build_shared_expert_ops
from cost_eval.llm_config import to_dimtable
from cost_eval.presets import deepseek_v3


def _find(ops, name):
    return next(op for op in ops if op.name == name)


def _dtable_gate_on():
    return dataclasses.replace(to_dimtable(deepseek_v3(4)), moe_shared_gate=True)


def test_sh_gate_w_dtype_is_fp32():
    # gate on 时 sh_gate_w 的 dtype_bytes 必须显式为 4（fp32，对齐 moe_router_dtype 默认 float32）
    gate = _find(build_shared_expert_ops(_dtable_gate_on()), "shared_gate")
    sh_gate_w = gate.params[0]
    assert sh_gate_w.name == "sh_gate_w"
    assert sh_gate_w.dtype_bytes == 4


def test_sh_gate_w_aligns_router_w_dtype():
    # gate 权重与 router 权重同为 fp32（真机同源 moe_router_dtype）
    d = _dtable_gate_on()
    sh_gate_w = _find(build_shared_expert_ops(d), "shared_gate").params[0]
    router_w = _find(build_moe_ffn_ops(d), "router").params[0]
    assert router_w.name == "router_w" and router_w.dtype_bytes == 4
    assert sh_gate_w.dtype_bytes == router_w.dtype_bytes


def test_gate_off_has_no_sh_gate_w():
    # gate off（DSv3 默认）不建 shared_gate/sh_gate_w → 不受 dtype 变更影响
    d_off = to_dimtable(deepseek_v3(4))
    assert d_off.moe_shared_gate is False
    names = [op.name for op in build_shared_expert_ops(d_off)]
    assert "shared_gate" not in names
