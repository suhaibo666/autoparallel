# tests/test_timesim_mla_segment.py
"""MLA（attention 族）段端到端：per-tensor 分片状态 + SequenceParallelLinear（T1 Phase A 靶点，
T0 交接要点5「attention 族 cell 会最先撞上守卫」的闭环验收）。

MLA dims：DSv3 比例缩小——q_lora_rank=1536、kv_lora_rank=512、qk_nope=128、qk_rope=64、
v_head=128、n_heads=8（q_head_dim=192）。"""
import pytest

from cost_eval.model_spec import DimTable
from cost_eval.timesim.shard_rules import Degrees
from cost_eval.timesim.producer import build_segment
from cost_eval.timesim.ir import op_flops

DIMS = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                S=4096, B=1, vocab=129280, n_layers=4,
                q_lora_rank=1536, kv_lora_rank=512,
                qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128)


def test_mla_tp1_builds_end_to_end(mla_dag):
    seg = build_segment("l0.attn.fwd", mla_dag, DIMS, Degrees())
    assert all(o.op_type != "CommOp" for o in seg.ops)
    mm = [o for o in seg.ops if o.op_type == "MatMul"]
    assert [o.module for o in mm] == [
        "SequenceParallelLinear", "SequenceParallelLinear",
        "ColumnParallelLinear", "ColumnParallelLinear", "RowParallelLinear"]
    fa = next(o for o in seg.ops if o.op_type == "FlashAttention")
    # 提取序 (S,B,N,D)：q/k D=192(nope128+rope64)，v D=128
    assert fa.in_shapes[0] == (4096, 1, 8, 192)
    assert fa.in_shapes[2] == (4096, 1, 8, 128)
    assert fa.out_shape == (4096, 1, 8, 128)


def test_mla_tp2_sp_comm_structure(mla_dag):
    """tp=2+SP 的通信结构（per-tensor 语义）：q_up/kv_up 各自 gather 自己的输入（两条
    module-semantics AG——T0 全局位只会发一条）、rope 支在 pe_concat 汇合注入一条
    layout-redistribution AG（源事实4）、proj 后一条 RS。"""
    deg = Degrees(tp=2, sequence_parallel=True)
    seg = build_segment("l0.attn.fwd", mla_dag, DIMS, deg)
    comms = [o for o in seg.ops if o.op_type == "CommOp"]
    kinds = [(o.comm.ctype, o.module) for o in comms]
    assert kinds.count(("all_gather", "injected:module-semantics")) == 2
    assert kinds.count(("all_gather", "injected:layout-redistribution")) == 1
    assert kinds.count(("reduce_scatter", "RowParallelLinear")) == 1
    assert len(comms) == 4
    assert all(o.stream == "comm_tp" for o in comms)


def test_mla_tp2_sp_shapes(mla_dag):
    deg = Degrees(tp=2, sequence_parallel=True)
    seg = build_segment("l0.attn.fwd", mla_dag, DIMS, deg)
    mm = [o for o in seg.ops if o.op_type == "MatMul"]
    # SPL（q_down :797）：SP 驻留——输入 S/2、权重全量、输出 S/2（源事实1）
    q_down = mm[0]
    assert q_down.in_shapes == ((2048, 1, 1792), (1792, 1536))
    assert q_down.out_shape == (2048, 1, 1536)
    # q_up（Column :734）：AG 后全 seq，flat 出轴 carrier=n_heads ÷2 → 8·192/2=768
    q_up = mm[2]
    assert q_up.in_shapes[0] == (4096, 1, 1536)
    assert q_up.out_shape == (4096, 1, 768)
    # FA：heads ÷2、seq 全长
    fa = next(o for o in seg.ops if o.op_type == "FlashAttention")
    assert fa.in_shapes[0] == (4096, 1, 4, 192)
    assert fa.in_shapes[2] == (4096, 1, 4, 128)
    # proj（Row :306）：入 (4096,1,512)、权重 (512,1792)、RS 后 (2048,1,1792)
    proj = mm[4]
    assert proj.in_shapes == ((4096, 1, 512), (512, 1792))
    rs = next(o for o in seg.ops if o.op_type == "CommOp"
              and o.comm.ctype == "reduce_scatter")
    assert rs.out_shape == (2048, 1, 1792)


def test_mla_tp_shard_conserves_gemm_flops(mla_dag):
    """性质（全局守恒）：tp=2+sp 下 per-rank 每个 GEMM 都减半 → 总 per-rank = 全局/2。
    **五个矩乘各自 ÷tp，机理不同但都减半**：SPL（q_down/kv_down）靠序列 S 分片（SP 把 token
    维分到各 rank，probe 实证 q_down 入 (2048,1,1792)——SPL"权重不切"是 weight shape 的性质，
    由 test_mla_tp2_sp_shapes 的 q_down 权重 (1792,1536) 全量断言捕获，**不是** flops 不变）；
    Column（q_up/kv_up）靠输出 feature 分片；Row（proj）靠输入 feature 分片。逐 src 都减半。"""
    full = {o.src: op_flops(o) for o in
            build_segment("s", mla_dag, DIMS, Degrees()).ops if o.op_type == "MatMul"}
    tp2 = {o.src: op_flops(o) for o in
           build_segment("s", mla_dag, DIMS, Degrees(tp=2, sequence_parallel=True)).ops
           if o.op_type == "MatMul"}
    assert set(full) == set(tp2)
    for src, mod_full in full.items():
        assert tp2[src] * 2 == mod_full, src          # 每个矩乘 per-rank 减半（含 SPL）
    assert sum(tp2.values()) * 2 == sum(full.values())   # 全局守恒
