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
MEASURED_ALLOC_MiB = 12473.1   # 真机 rank0

d = DimTable(
    H=1792, F=3072, n_heads=8, n_kv=8, head_dim=192, S=4096, B=1, vocab=129280,
    n_layers=6,                       # = len(layer_pattern)（含 embedding + head）
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
    x = TensorRef("h_final", ("S", "B", "H"), shard={0: "sp"})
    w = TensorRef("head_w", ("H", "vocab"), is_weight=True)
    logits = TensorRef("logits_lm", ("S", "B", "vocab"))          # 巨大，saved 给 loss
    return LayerSpec(ops=[OpSpec("lm_head", OpType.MATMUL, [x, w], logits, params=[w], saves=[x, logits])])


layer_specs = {
    "embedding": build_embedding(d),
    "mla_dense": build_mla_dense_decoder(d),
    "mla_moe": build_mla_moe_decoder(d),
    "lm_head": build_lm_head(d),
}
layer_pattern = ["embedding", "mla_dense", "mla_moe", "mla_moe", "mla_moe", "lm_head"]
spec = ModelSpec("dsv3-4L", d, layer_pattern, layer_specs)

# 真机数据点（4层DSv3 缩层 run）
MEASURED_PEAK_MiB = 12473.1     # max_memory_allocated（训练反向中峰值）
MEASURED_RESIDENT_MiB = 3862.0  # 训练后当前 allocated（= param+m+v 常驻，grad 已释）

pc = ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1, sequence_parallel=True,
                    num_microbatches=1)
# fp32 AdamW 常驻 = param(4)+m(4)+v(4) = 12 B/param；grad(4B) 是反向瞬态不常驻
opt = OptimizerSpec(type="AdamW", state_bytes_per_param=12)
ev = Evaluator(spec, pc, opt, HardwareSpec(max_device_memory=59 * GiB),
               RecomputeSpec(mode="full", full_layers={1, 2, 3, 4}), SwapSpec())
rep = ev.evaluate()
p = rep.per_stage[0]
b = p.breakdown
print("=== DSv3 4L  FSDP-2  full-recompute（fp32 params, 常驻=param+m+v=12B/param）===")
print(f"[静态验证] 预测常驻 persistent = {b.persistent/MiB:8.1f} MiB   vs 真机常驻 {MEASURED_RESIDENT_MiB:.0f} MiB"
      f"   误差 {abs(b.persistent/MiB-MEASURED_RESIDENT_MiB)/MEASURED_RESIDENT_MiB*100:.1f}%")
print(f"[峰值对标] 预测 peak = {p.peak_bytes/MiB:8.1f} MiB   vs 真机 peak {MEASURED_PEAK_MiB:.0f} MiB"
      f"   ratio {(p.peak_bytes/MiB)/MEASURED_PEAK_MiB:.3f}")
gap = MEASURED_PEAK_MiB - p.peak_bytes / MiB
print(f"[缺口] 真机峰值 - 预测 = {gap:8.1f} MiB  ~ 框架反向瞬态(FSDP all-gather全参/full grad/大vocab loss区/MoE all-to-all/hccl+flash workspace)")
print(f"--- 预测 breakdown (MiB) ---")
for k in ("persistent", "act_live", "gather_buf", "grad_buf", "recomp_scratch",
          "swap_buf", "workspace", "framework"):
    print(f"  {k:16s} = {getattr(b, k)/MiB:9.1f}")
