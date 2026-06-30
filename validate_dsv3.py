"""DSv3 4层缩层配置：评估器预测 vs 真机实测峰值（真机 rank0 peak_alloc=12473.1 MiB）。

对应真机 run：pynarive_ds3.yaml 缩层4层，FSDP-2(dp_shard=2)、tp=ep=pp=cp=1、SP=on、full 重算、
compute=bf16/params=fp32、seq=4096、local_batch=1。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from cost_eval.model_spec import DimTable, TensorRef, OpSpec, OpType, LayerSpec, ModelSpec
from cost_eval.layers.mla import build_mla_dense_decoder, build_mla_moe_decoder
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator

MiB = 2 ** 20
GiB = 2 ** 30
import os
N = int(os.environ.get("SIM_LAYERS", "4"))   # transformer 层数（与真机 SIM_LAYERS 对齐）
EP = int(os.environ.get("SIM_EP", "1"))       # 专家并行度（变配置验证）
# 真机实测 (peak_alloc_MiB, resident_MiB) by (层数, ep)
MEASURED = {(4, 1): (12473.1, 3862.0), (8, 1): (13953.3, None), (4, 2): (12474.1, None)}
# framework_reserve(allocated) = residual（MoE staging+flash+cast+碎片，**不含 HCCL**，ep=2 修正 §8.6）
# 真机 ep=1/2、层4/8 下近恒定 2197；HCCL 在 reserved 池不计入 allocated 峰值
RESIDUAL_MiB = 2197

d = DimTable(
    H=1792, F=3072, n_heads=8, n_kv=8, head_dim=192, S=4096, B=1, vocab=129280,
    n_layers=N + 2,                   # = len(layer_pattern)（含 embedding + head）
    n_experts=8, topk=4, n_shared=1, moe_F=1024, capacity_factor=1.0,
    q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128,
    v_head_dim=192, moe_shared_F=1024,
    dtype_bytes=2,                    # compute bf16
)


def build_embedding(d):
    w = TensorRef("emb_w", ("vocab", "H"), is_weight=True)        # vocab_emb_dp：tp=1 不切
    out = TensorRef("emb_out", ("S", "B", "H"), shard={0: "sp"})
    return LayerSpec(ops=[OpSpec("embedding", OpType.ELEMENTWISE, [], out, params=[w], saves=[])])


def build_lm_head(d):
    # 对照 loss.py：logits(bf16) → cast fp32 → log_softmax(fp32,saved) → NLL；反向物化 probs(fp32)
    x = TensorRef("h_final", ("S", "B", "H"), shard={0: "sp"})
    w = TensorRef("head_w", ("H", "vocab"), is_weight=True)
    logits = TensorRef("logits_lm", ("S", "B", "vocab"))                      # bf16, saved(ctx.logits)
    logsm = TensorRef("logsm", ("S", "B", "vocab"), dtype_bytes=4)            # fp32, saved
    loss = TensorRef("loss", ("B",))
    return LayerSpec(ops=[
        OpSpec("lm_head", OpType.MATMUL, [x, w], logits, params=[w], saves=[x]),
        OpSpec("logsoftmax", OpType.NORM, [logits], logsm, saves=[logits]),
        # NLL 反向 probs=exp(-log_softmax) 物化 fp32（loss.py:80）
        OpSpec("nll", OpType.ELEMENTWISE, [logsm], loss, saves=[logsm], bwd_scratch="4*S*B*vocab"),
    ])


layer_specs = {
    "embedding": build_embedding(d),
    "mla_dense": build_mla_dense_decoder(d),
    "mla_moe": build_mla_moe_decoder(d),
    "lm_head": build_lm_head(d),
}
layer_pattern = ["embedding"] + ["mla_dense"] + ["mla_moe"] * (N - 1) + ["lm_head"]
spec = ModelSpec(f"dsv3-{N}L", d, layer_pattern, layer_specs)
full_layers = set(range(1, N + 1))   # transformer 层（embedding=0, head=N+1 不重算）

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
