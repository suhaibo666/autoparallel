"""DSv3 4层缩层配置：评估器预测 vs 真机实测峰值（真机 rank0 peak_alloc=12473.1 MiB）。

对应真机 run：pynarive_ds3.yaml 缩层4层，FSDP-2(dp_shard=2)、tp=ep=pp=cp=1、SP=on、full 重算、
compute=bf16/params=fp32、seq=4096、local_batch=1。

本模块也作为**可导入的 DSv3 缩层模型构建器**：`build_dsv3_spec(N)` 返回 (spec, d, full_layers)，
供 analyze_matrix.py 在多维并行扫描中复用同一份模型定义（构建一次、跨配置复用）。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from cost_eval.model_spec import LayerSpec
from cost_eval.build_llm import build_llm_spec
from cost_eval.presets import deepseek_v3
from cost_eval.layers.head import build_embedding_ops, build_head_and_loss_ops
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

MiB = 2 ** 20
GiB = 2 ** 30
import os
# 真机实测 (peak_alloc_MiB, resident_MiB) by (层数, ep)
MEASURED = {(4, 1): (12473.1, 3862.0), (8, 1): (13953.3, None), (4, 2): (12474.1, None)}
# framework_reserve(allocated) = residual（**不含 HCCL**，ep=2 修正 §8.6）。
# 原 2197 里 ~2020 MiB 是 loss 反向漏建的 grad_log_softmax(fp32 满 vocab)，现已显式建进
# nll.bwd_scratch（probs + grad 共 2×4·S·B·vocab，loss.py:80-82）；残余 ~177 = flash workspace
# + MoE all-to-all staging + 分配器块对齐取整。HCCL 在 reserved 池不计入 allocated 峰值。
RESIDUAL_MiB = 177


def build_embedding(d):
    """word embedding 段（1 op），委托统一装配件 `cost_eval.layers.head.build_embedding_ops`。

    保留 `(d: DimTable) -> LayerSpec` 签名（`test_build_llm_tier1` 等仍以 DimTable 调用），
    但 op 构造已统一到与 `build_llm_spec` 同源的 builder（逐字段一致）。embedding op 与
    cfg 无关，用 `deepseek_v3()` 作 DSv3 驱动的规范 cfg。
    """
    return LayerSpec(build_embedding_ops(deepseek_v3()))


def build_lm_head(d):
    """lm_head + loss 段（3 op），委托 `cost_eval.layers.head.build_head_and_loss_ops`。

    保留 `(d: DimTable) -> LayerSpec` 签名；DSv3 走默认 `logsoftmax_nll` + `tie=False` 路径，
    与原手写 `build_lm_head` 逐字段一致（`test_build_llm_tier1` 的 verbatim 断言）。
    """
    return LayerSpec(build_head_and_loss_ops(deepseek_v3()))


def build_dsv3_spec(N):
    """构建 DSv3 缩层 ModelSpec（N 个 transformer 层），改走统一 preset 装配路径。

    等价于 `build_llm_spec(deepseek_v3(N))`：层序列
    `["embedding", "mla_dense", "mla_moe"*(N-1), "lm_head"]`（1 个 dense MLA + (N-1) 个 MoE
    MLA + embedding/head，共 N+2 个 layer），与原手写 `build_dsv3_spec` **逐字节一致**
    （`tests/test_regression_dsv3.py` 硬门：N=4→12472.5 MiB，N=8→13896.1 MiB）。

    返回 `(spec, dims, full_layers)`：`dims = spec.dims`（DimTable），
    `full_layers = set(range(1, N+1))`（transformer 层；embedding=0 / head=N+1 不重算）。
    签名不变，供 `analyze_matrix` / `timeline_probe` 解包复用。
    """
    spec = build_llm_spec(deepseek_v3(N))
    full_layers = set(range(1, N + 1))   # transformer 层（embedding=0, head=N+1 不重算）
    return spec, spec.dims, full_layers


def main():
    N = int(os.environ.get("SIM_LAYERS", "4"))   # transformer 层数（与真机 SIM_LAYERS 对齐）
    EP = int(os.environ.get("SIM_EP", "1"))       # 专家并行度（变配置验证）
    spec, d, full_layers = build_dsv3_spec(N)

    # 任意并行配置（env 驱动，用于 FSDP/EP/TP/PP 组合矩阵研究）
    TP = int(os.environ.get("SIM_TP", "1"))
    PP = int(os.environ.get("SIM_PP", "1"))
    WORLD = int(os.environ.get("SIM_WORLD", "2"))
    DPSHARD = int(os.environ.get("SIM_DPSHARD", "-1"))
    dp_shard = DPSHARD if DPSHARD > 0 else max(WORLD // (TP * PP), 1)   # dp_replicate=cp=1
    mbs = PP if PP > 1 else 1   # PP 时在飞 microbatch 数

    pc = ParallelConfig(dp_shard=dp_shard, tp=TP, ep=EP, pp=PP, cp=1, sequence_parallel=True,
                        num_microbatches=mbs)
    opt = OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4)
    # framework_reserve(allocated)=residual(剔HCCL); 研究模式下也看"结构峰值"(剔framework)
    ev = Evaluator(spec, pc, opt,
                   HardwareSpec(max_device_memory=59 * GiB,
                                framework_reserve=RESIDUAL_MiB * MiB),
                   RecomputeSpec(mode="full", full_layers=full_layers), SwapSpec())
    rep = ev.evaluate()
    p = rep.per_stage[0]
    b = p.breakdown
    pk = p.peak_bytes / MiB
    struct = pk - b.framework / MiB    # 结构峰值（剔 framework_reserve）
    from cost_eval.framework import num_distinct_communicators
    print(f"=== DSv3 {N}L  world={WORLD} dp_shard={dp_shard} tp={TP} ep={EP} pp={PP}  (hccl子通信器={num_distinct_communicators(pc)}) ===")
    print(f"[峰值] 预测(含reserve) = {pk:8.1f} MiB ; 结构(剔reserve) = {struct:8.1f} MiB ; persistent={b.persistent/MiB:.0f}")
    print(f"--- 预测 breakdown (MiB) ---")
    for k in ("persistent", "act_live", "gather_buf", "grad_buf", "recomp_scratch",
              "bwd_scratch", "swap_buf", "workspace", "framework"):
        print(f"  {k:16s} = {getattr(b, k)/MiB:9.1f}")


if __name__ == "__main__":
    main()
