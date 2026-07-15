"""P1-01（任务 A）：shared-expert gate 的 FP32 hidden-cast 瞬态生命周期。

真机 mindformers `shared_experts.py`（pynative :69-71 / training_graph :82-83）：
    gate = self.sigmoid(self.shared_experts_gate(self.cast(hidden_states, self.router_dense_type)))
gate Dense 之前把**完整 hidden** `[S,B,H]` cast 到 router dtype（fp32），产生一份 `[S,B,H]` fp32
副本。gate Dense（matmul）反向需其输入（= 该 fp32 cast）算权重梯度 `dW = x^T @ dy` → 该 fp32
hidden 存活到 gate 反向。此前 `build_shared_expert_ops` 的 `shared_gate` op `saves=[]`——漏建此
每-token 全 hidden 的 fp32 瞬态。

本组：
  - gate ON：`shared_gate` op 保存**一份** `[S,B,H]`、dtype=4B(fp32) 的 hidden cast 瞬态；
    张量名区别于已 save 的 bf16 `h1`（否则 act_live 按名去重不计额外显存）；能正常 resolve 成
    S·B·H·4 字节。
  - gate OFF（DSv3 默认 `moe_shared_gate=False`）：无 `shared_gate` op → 无此瞬态（golden 守卫）。
"""
import dataclasses

from cost_eval.layers.ffn import build_shared_expert_ops
from cost_eval.llm_config import to_dimtable
from cost_eval.parallel_model import ParallelModel
from cost_eval.presets import deepseek_v3
from cost_eval.shape_eval import resolve_tensor
from cost_eval.specs import ParallelConfig


def _find(ops, name):
    return next((op for op in ops if op.name == name), None)


def _d_gate_on():
    return dataclasses.replace(to_dimtable(deepseek_v3(4)), moe_shared_gate=True)


def _fp32_hidden_saves(op):
    """op.saves 中形如 [S,B,H] 且 dtype=4B 的 fp32 hidden cast 瞬态。"""
    return [s for s in op.saves
            if s.shape == ("S", "B", "H") and s.dtype_bytes == 4]


# ── gate ON：fp32 hidden cast 瞬态建为 shared_gate 的 saved 激活 ─────────────────────
def test_gate_on_shared_gate_saves_one_fp32_hidden_cast():
    gate = _find(build_shared_expert_ops(_d_gate_on()), "shared_gate")
    assert gate is not None, "gate on 时应有 shared_gate op"
    fp32 = _fp32_hidden_saves(gate)
    assert len(fp32) == 1, (
        "shared_gate 应恰保存一份 [S,B,H] fp32 hidden cast 瞬态，实为 "
        f"{[(s.name, s.shape, s.dtype_bytes) for s in gate.saves]}")


def test_fp32_hidden_cast_has_distinct_name_from_h1():
    # 独立张量名（≠h1）以计入额外显存；shape/dtype 为 [S,B,H]/4B。
    gate = _find(build_shared_expert_ops(_d_gate_on()), "shared_gate")
    saved = _fp32_hidden_saves(gate)[0]
    assert saved.name != "h1"
    assert saved.shape == ("S", "B", "H")
    assert saved.dtype_bytes == 4


def test_fp32_hidden_cast_resolves_to_fp32_bytes():
    d = _d_gate_on()
    gate = _find(build_shared_expert_ops(d), "shared_gate")
    saved = _fp32_hidden_saves(gate)[0]
    pm = ParallelModel(ParallelConfig(), n_layers=d.n_layers, world_size=1)
    rt = resolve_tensor(saved, d, pm)
    assert rt.dtype_bytes == 4
    assert rt.local_numel == d.S * d.B * d.H   # tp/cp=1 → 无切分，全量 S·B·H


# ── gate OFF（DSv3 默认）：无 shared_gate、无 fp32 hidden 瞬态（golden 守卫）─────────────
def test_gate_off_no_shared_gate_and_no_fp32_hidden_cast():
    d_off = to_dimtable(deepseek_v3(4))
    assert d_off.moe_shared_gate is False
    ops = build_shared_expert_ops(d_off)
    assert _find(ops, "shared_gate") is None
    # 任何 op 都不应保存 [S,B,H] fp32 瞬态（shared_fc1 save 的 h1 是 bf16/dtype 未指定，不算）
    for op in ops:
        assert _fp32_hidden_saves(op) == [], f"{op.name} 不应有 fp32 hidden cast save"
