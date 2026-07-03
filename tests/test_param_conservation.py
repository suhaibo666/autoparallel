"""Task 7 — 全局参数守恒（会抓住 C2 幻影参数类回归）。

全 1 并行下 local_numel == global_numel，sum 所有 op 的 param 张量即全局参数量。
与逐结构手算 / 冻结 golden 在 ±2% 内一致 → op 图把权重建全、且无重复计入。
"""
from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.build_llm import build_llm_spec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import ParallelConfig


def _global_param_count(spec) -> int:
    pc = ParallelConfig(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1,
                        sequence_parallel=False)
    pm = ParallelModel(pc, spec.dims.n_layers, world_size=1)
    g = ShapeEval().resolve(spec, pm)
    return sum(
        w.local_numel
        for layers in g.stages.values()
        for layer in layers
        for op in layer.ops
        for w in op.params
    )


def _rel_err(got: int, ref: int) -> float:
    return abs(got - ref) / ref


def test_dsv3_param_conservation():
    """DSv3(4) 逐结构手算（独立于 build_llm_spec）：
      emb vocab·H = 129280·1792 = 231,669,760；lm_head 同 = 231,669,760
      MLA attn/层 = qkv(H·2112)+qb(1536·1536)+kvb(512·2560)+o(1536·H) = 10,207,232
      dense ffn = fc1(H·2·3072)+fc2(3072·H) = 11,010,048+5,505,024 = 16,515,072
      moe ffn = 8·(H·2·1024)+8·(1024·H) + shared(H·2·1024+1024·H)
              = 44,040,192 + 5,505,024 = 49,545,216
      总 = 2·231,669,760 + (10,207,232+16,515,072) + 3·(10,207,232+49,545,216) = 669,319,168
    """
    ref = 669_319_168
    n = _global_param_count(build_llm_spec(deepseek_v3(4)))
    assert _rel_err(n, ref) < 0.02, f"DSv3(4) params={n:,} vs ref {ref:,} err={_rel_err(n, ref):.4%}"


def test_dsv4_param_conservation_no_phantom():
    """DSv4(4) 冻结 golden（tie 后无幻影 vocab·H；逐层：emb 231.670M + r0_dense 53.262M
      + r4_moe 199.458M + r128_moe 186.096M + r0_moe 185.383M + mtp 191.805M
      + lm_head 231.670M = 1,279,344,144）。
    若 C2 复发（MTP untie）会再 +2·vocab·H≈463M → ~1,742M，超 ±2% → 本测试挂（守住 C2）。

    2026-07-03 修订：ref 1,200,440,848 → 1,279,344,144（+78.9M）。**非幻影**——三个
    **ratio-0 注意力块**（r0_dense/r0_moe/mtp）从 MLA-base 权重切到 DSv4-own 权重
    （各 +~26.18M）：真机 Profiler 定位 DSv4HybridSelfAttention 对**所有 ratio**（含滑窗
    0/1）都用同一顶层模块（per-head fp32 Q-norm + 单共享 KV + 分组输出），滑窗只是其中一个
    分支、非独立 MLA 模块。故 ratio-0 不再退化复用 build_mla_attn_ops。
    佐证无幻影：emb + lm_head 两项 **逐字节不变**（各 231.670M）；r4_moe/r128_moe
    （本就 DSv4-own）不变。delta 全部落在 3 个 r0 注意力块，无 vocab·H 混入。
    """
    ref = 1_279_344_144
    n = _global_param_count(build_llm_spec(deepseek_v4(4)))
    assert _rel_err(n, ref) < 0.02, f"DSv4(4) params={n:,} vs ref {ref:,} err={_rel_err(n, ref):.4%}"
    # 直接守卫：vocab 权重只应计两次（emb + lm_head），不得混入 MTP 幻影。
    vocab_h = 129280 * 1792
    assert n < ref + vocab_h, "DSv4 param 疑似含 MTP 幻影 vocab 权重（C2 回归）"
