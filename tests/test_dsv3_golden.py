"""Task 7 [I6] — DSv3 N=4 独立冻结 golden（防自指回归门）。

`test_regression_dsv3` 把 `build_llm_spec(deepseek_v3(4))` 与 `build_dsv3_spec(4)` 互比，
但后者已委托前者（`validate_dsv3.build_dsv3_spec → build_llm_spec(deepseek_v3)`）——**自指**：
两边一起改就一起错、抓不住回归。这里把 DSv3 N=4 的 **op 名序列 + 逐桶 breakdown** 冻结为
硬编码常量，独立守卫（与 build 无关，改了 build 就会挂）。数值锚点断言保留。
"""
from cost_eval.presets import deepseek_v3
from cost_eval.build_llm import build_llm_spec
from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from cost_eval.report import Evaluator

MiB, GiB = 2 ** 20, 2 ** 30

# ── 冻结 golden：层序列 + 各唯一层 key 的 op 名序列 ─────────────────────────────
GOLDEN_PATTERN = ["embedding", "mla_dense", "mla_moe", "mla_moe", "mla_moe", "lm_head"]
GOLDEN_OPS = {
    "embedding": ["embedding"],
    "mla_dense": ["ln1", "linear_qkv", "q_a_norm", "kv_a_norm", "linear_qb", "linear_kvb",
                  "rope", "flash", "o_proj", "add1", "ln2", "fc1", "swiglu", "fc2", "add2"],
    "mla_moe": ["ln1", "linear_qkv", "q_a_norm", "kv_a_norm", "linear_qb", "linear_kvb",
                "rope", "flash", "o_proj", "add1",
                # ln2(2026-07-16 补建):pre-FFN norm(post_attention_layernorm)。此前 MoE 层漏此 op
                # （dense 有、moe 无），每 MoE 层欠一份 [S,B,H] fp32-cast 常驻 → 无重算/select 欠预测。
                # build_transformer_layer 统一前插；routed(router/dispatch)+shared(shared_fc1) 都消费 ln2。
                "ln2", "router", "dispatch", "e_fc1",
                "e_swiglu", "e_fc2", "combine", "shared_fc1", "shared_swiglu", "shared_fc2",
                # moe_add(2026-07-11 补边):routed+shared 合流(moe_layer 真实语义,线性 add saves=[]
                # 零字节;修 op 图 combine/shared_fc2 孤立叶节点)。
                "moe_add"],
    "lm_head": ["final_norm", "lm_head", "logsoftmax", "nll"],   # P1-01: 补 final_norm(真机实存)
}
# ── 冻结 golden：N=4 峰值逐桶字节（framework_reserve=177 MiB、full 重算 1..4、FSDP dp_shard=2、
#    **prefetch_depth=0**——旧单缓冲回归路径：depth=0 逐字节复现预取建模前的 breakdown，
#    守卫「depth=0 复现旧行为」；默认 depth=1 的拆解路径另见 test_fsdp_prefetch.py）─
# P1-01（2026-07-14）：+router fp32 权重/norm gamma（persistent/gather/grad +371712 B 级）
# + final_norm op（act_live + 其 fp32 保留输入）——真机实存、修前欠算；峰 12472.5 → 12500.9 MiB
# （真机 12473.1 → 1.0022，0.2% 保守侧）。
# 2026-07-16：pre-FFN norm(ln2) 补建 → 3 个 MoE 层各 +1 个 ln2_g([H]fp32) 参数 → persistent
#   +32256 B（4016286720→4016318976）；full 重算下 ln2 saved 激活丢弃 → act_live/recomp/峰值事件
#   逐字节不变；峰 +32256 B（仍 12500.9 MiB，真机 12473.1 → 1.0022 不变）。
# **2026-07-29 二次重钉**（`docs/census_fix_mhc_rmsnorm_2026-07-29.md` Fix 2）：
#   `FusedRMSNorm.construct`（`layer_norm.py:151-155`）**不** cast，`:149` 的 self.cast 是死属性；
#   DSv3 站点 `normalization="RMSNorm"`（`configuration_deepseek_v3.py:149`）→ `get_norm_cls`
#   （`:190-191`）恒返回 `FusedRMSNorm` → 其保留输入**不产生 fp32 副本**。norm-fp32 抬升改为
#   按 norm 种类成立后，DSv3 的 ln1/ln2/q_a_norm/kv_a_norm/final_norm 全部落回自身 dtype。
#   act_live 3279945728 → 3265265664（−14680064 B = −14.0 MiB）；峰 12500.9 → 12486.9 MiB
#   （真机 12473.1 → 1.0022 → **1.0011**，仍在保守侧）。逐桶之和 == 峰值的不变量不动。
GOLDEN_BREAKDOWN = {
    "persistent": 4016318976, "act_live": 3265265664, "gather_buf": 463346688,
    "grad_buf": 926686208, "recomp_scratch": 0, "bwd_scratch": 4236247040,
    "swap_buf": 0, "workspace": 0, "framework": 185597952,
}
GOLDEN_PEAK_EVENT = "bwd@5"
GOLDEN_PEAK_BYTES = 13093462528        # = 12486.9 MiB（真机 12473.1 → 1.0011）


def _spec():
    return build_llm_spec(deepseek_v3(4))


def _eval(spec):
    ev = Evaluator(
        spec,
        ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1, sequence_parallel=True,
                       prefetch_depth=0),   # 回归路径：复现预取建模前的旧单缓冲 breakdown
        OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
        HardwareSpec(max_device_memory=59 * GiB, framework_reserve=177 * MiB),
        RecomputeSpec(mode="full", full_layers=set(range(1, 5))),
        SwapSpec(),
    )
    return ev.evaluate().per_stage[0]


def test_dsv3_op_structure_frozen():
    spec = _spec()
    assert spec.layer_pattern == GOLDEN_PATTERN
    for key, names in GOLDEN_OPS.items():
        assert [op.name for op in spec.layer_specs[key].ops] == names, key


def test_dsv3_breakdown_frozen_and_anchor():
    p = _eval(_spec())
    b = p.breakdown
    for k, v in GOLDEN_BREAKDOWN.items():
        assert getattr(b, k) == v, (k, getattr(b, k), v)
    assert p.peak_event == GOLDEN_PEAK_EVENT
    assert p.peak_bytes == GOLDEN_PEAK_BYTES
    assert abs(p.peak_bytes / MiB - 12486.9) < 0.1
    # 逐桶之和恰为峰值（无遗漏/重复）
    assert sum(GOLDEN_BREAKDOWN.values()) == GOLDEN_PEAK_BYTES
