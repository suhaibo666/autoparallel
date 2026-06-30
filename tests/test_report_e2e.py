"""端到端测试：M7 report.Evaluator 门面（Task 13）。"""
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.layers.dense import build_dense_decoder
from cost_eval.layers.moe import build_moe_decoder
from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from cost_eval.report import Evaluator


def test_e2e_dense_breakdown_and_oom():
    D = DimTable(H=4096, F=11008, n_heads=32, n_kv=32, head_dim=128,
                 S=4096, B=1, vocab=32000, n_layers=4)
    spec = ModelSpec("llama-ish", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    ev = Evaluator(
        spec, ParallelConfig(tp=8, dp_shard=8, num_microbatches=1),
        OptimizerSpec.adamw(), HardwareSpec(max_device_memory=60 * 2**30),
        RecomputeSpec("None"), SwapSpec())
    rep = ev.evaluate()
    p = rep.per_stage[0]
    # persistent ≈ (4 层参数) / (tp * fsdp = 64) * 16；非 0、有拆解
    assert p.breakdown.persistent > 0
    assert rep.tightest_stage == 0
    assert rep.oom == (p.peak_bytes > 60 * 2**30)


def test_e2e_moe_runs():
    DM = DimTable(H=512, F=1024, n_heads=8, n_kv=8, head_dim=64, S=512, B=1,
                  vocab=1000, n_layers=2, n_experts=8, topk=2, moe_F=1024)
    spec = ModelSpec("moe", DM, ["moe", "moe"], {"moe": build_moe_decoder(DM)})
    ev = Evaluator(spec, ParallelConfig(ep=4, tp=2, dp_shard=2, num_microbatches=1),
                   OptimizerSpec.adamw(), HardwareSpec(max_device_memory=80 * 2**30),
                   RecomputeSpec("None"), SwapSpec())
    rep = ev.evaluate()
    assert rep.per_stage[0].peak_bytes > 0
