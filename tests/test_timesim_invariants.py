# tests/test_timesim_invariants.py
"""IR 层不变量（spec §3.4 / §7.1-L0 的 IR 半边）：GEMM 精确式 + bwd 守恒 + tp 切分守恒 + 词表合法性。

边界（有意为之）：含 module=="" 单入 MatMul 的段（MoE router 路径）经 expand_bwd 会 fail-loud
（ir.py 元数警示）——bwd 守恒不变量对 MoE 段暂不可跑，PART A 记录 router 权重后解除（T1）。"""
import pytest

from cost_eval.model_spec import DimTable
from cost_eval.timesim.ir import (op_flops, COMM_STREAM, STREAM_DEVICE, STREAM_HOST_ONLY)
from cost_eval.timesim.shard_rules import Degrees
from cost_eval.timesim.producer import build_segment
from cost_eval.timesim.bwd_rules import expand_bwd

DIMS = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                S=4096, B=1, vocab=129280, n_layers=4)
_LEGAL_STREAMS = {STREAM_DEVICE, STREAM_HOST_ONLY} | set(COMM_STREAM.values())
_LEGAL_PHASES = {"fwd", "bwd", "recomp"}


def _seg(mlp_dag, deg=None):
    return build_segment("mlp.fwd", mlp_dag, DIMS, deg or Degrees())


def test_mlp_fwd_gemm_flops_exact(mlp_dag):
    """退化(全度=1)精确式：fc1=2·S·B·H·2F + fc2=2·S·B·F·H（spec §7.1-L0③ 的 IR 半边，零公差）。"""
    got = sum(op_flops(o) for o in _seg(mlp_dag).ops)
    S, B, H, F = 4096, 1, 1792, 3072
    assert got == 2 * S * B * H * (2 * F) + 2 * S * B * F * H


def test_bwd_gemm_flops_double_fwd(mlp_dag):
    """bwd GEMM FLOPs = 2×fwd（dX+dW 各一份）——6ND 里 4ND 的来源。须滤 phase（recompute 另加 1×）。"""
    fwd = _seg(mlp_dag)
    bwd = expand_bwd(fwd)
    f = sum(op_flops(o) for o in fwd.ops)
    assert sum(op_flops(o) for o in bwd.ops if o.phase == "bwd") == 2 * f


def test_recompute_prefix_adds_exactly_fwd_gemm(mlp_dag):
    fwd = _seg(mlp_dag)
    seg = expand_bwd(fwd, recompute="full", recomp_comm=False)
    f = sum(op_flops(o) for o in fwd.ops)
    assert sum(op_flops(o) for o in seg.ops if o.phase == "recomp") == f


def test_every_device_fwd_op_has_bwd(mlp_dag):
    fwd = _seg(mlp_dag)
    bwd = expand_bwd(fwd)
    bwd_roots = {o.op_id.rsplit(".b", 1)[0] for o in bwd.ops if o.phase == "bwd"}
    for o in fwd.ops:
        assert o.op_id in bwd_roots, f"{o.op_id}({o.op_type}) 无 bwd 展开"


def test_tp_shard_conserves_global_flops(mlp_dag):
    """性质：tp=2 时 per-rank GEMM FLOPs = 全局/2（切分守恒——spec §7.1-L0④）。"""
    full = sum(op_flops(o) for o in _seg(mlp_dag).ops)
    tp2 = sum(op_flops(o) for o in
              _seg(mlp_dag, Degrees(tp=2, sequence_parallel=True)).ops)
    assert tp2 * 2 == full


def test_stream_and_phase_vocabulary(mlp_dag):
    """词表合法性（Task 7 review 委托：dumb-IR 的验证归宿）——覆盖 fwd/bwd/recomp 全相位。"""
    fwd = _seg(mlp_dag, Degrees(tp=2, sequence_parallel=True))
    seg = expand_bwd(fwd, recompute="full", recomp_comm=True)
    for o in tuple(fwd.ops) + tuple(seg.ops):
        assert o.stream in _LEGAL_STREAMS, f"{o.op_id}: 非法 stream {o.stream!r}"
        assert o.phase in _LEGAL_PHASES, f"{o.op_id}: 非法 phase {o.phase!r}"
        if o.op_type == "CommOp":
            assert o.comm is not None and o.comm.volume_bytes > 0, f"{o.op_id}: CommOp 无载荷"
