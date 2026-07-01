# 统一 LLM ModelSpec 构建器 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** 用一个 config 驱动的 `build_llm_spec(LLMConfig)` 替代 per-model 手写 spec，覆盖设计
[`2026-07-01-unified-llm-modelspec-design.md`](2026-07-01-unified-llm-modelspec-design.md) 的 Tier-1 + 前沿（dsv4_hybrid、mHC、MTP）。

**Architecture:** 三派发轴（注意力/FFN/残差）+ 注册表；现有 `layers/{dense,moe,mla}.py` 重构为命名 op-builder；`LLMConfig` = 内存结构子集。

**Tech Stack:** Python 3, `cost_eval/` 纯解析包，pytest。无 MindSpore（真机部分用 skill `real-machine-memory-sim`）。

**铁律（贯穿全程）：** `deepseek_v3` preset 必须逐桶复现已真机验证的锚点 **4L=12472.5 / 8L=13896.1 MiB**（P0 §8.7/§8.9）。Phase 1 结束设**硬门**，未过不进 Phase 2。

**分支：** `feat/unified-llm-modelspec`（已建）。每个 Task 完成即 commit。

---

## Phase 0 — 积木重构（行为不变）+ 骨架

**目标：** 把 attn/ffn op 从"一个 LayerSpec + 切片"拆成命名 op-builder；建 `LLMConfig` 与注册表骨架。**不改任何数值**——现有 52 测试必须全绿。

### Task 0.1: 抽出命名 attn/ffn op-builder（`cost_eval/layers/`）

**Files:**
- Create: `cost_eval/layers/attention.py`, `cost_eval/layers/ffn.py`
- Modify: `cost_eval/layers/dense.py`, `moe.py`, `mla.py`（改为从新模块 re-export，保持旧导入路径）
- Test: `tests/test_layers_refactor.py`

- [ ] **Step 1: 写失败测试**——断言重构后 op 序列与旧完全一致

```python
# tests/test_layers_refactor.py
from cost_eval.model_spec import DimTable
from cost_eval.layers.attention import build_gqa_attn_ops, build_mla_attn_ops
from cost_eval.layers.ffn import build_dense_ffn_ops, build_moe_ffn_ops
from cost_eval.layers import dense, moe, mla   # 旧路径仍可用

D = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=192, S=4096, B=1,
             vocab=129280, n_layers=6, dtype_bytes=2)

def _names(ops): return [o.name for o in ops]

def test_gqa_attn_ops_equal_old_dense_prefix():
    # 旧 build_dense_decoder.ops[:6] == 新 build_gqa_attn_ops
    assert _names(build_gqa_attn_ops(D)) == _names(dense.build_dense_decoder(D).ops[:6])

def test_dense_ffn_ops_equal_old_suffix():
    assert _names(build_dense_ffn_ops(D)) == _names(dense.build_dense_decoder(D).ops[6:])

def test_mla_decoder_recomposed_equal():
    # 新装配（mla_attn + dense_ffn）op 序列 == 旧 build_mla_dense_decoder
    from cost_eval.layers.mla import build_mla_dense_decoder
    recomposed = _names(build_mla_attn_ops(D)) + _names(build_dense_ffn_ops(D))
    assert recomposed == _names(build_mla_dense_decoder(D).ops)
```

- [ ] **Step 2: 跑测试确认失败**（新模块不存在）：`python -m pytest tests/test_layers_refactor.py -x` → ImportError。
- [ ] **Step 3: 实现**——把 `dense.py` 的 attn 段（前 6 op）搬进 `attention.py:build_gqa_attn_ops`、ffn 段（后 5 op）搬进 `ffn.py:build_dense_ffn_ops`；`mla.py` 的 `build_mla_attn_ops` 移入 `attention.py`；`moe.py` 的 ffn 段移入 `ffn.py:build_moe_ffn_ops` + `build_shared_expert_ops`。旧文件 re-export 新符号并保留 `build_*_decoder` 用新 builder 组合。**逐字节保持 op 定义**。
- [ ] **Step 4: 跑全量测试**：`python -m pytest -q` → **52 + 新测试全绿**。
- [ ] **Step 5: commit** `refactor(layers): 抽出命名 attn/ffn op-builder（行为不变）`

### Task 0.2: `LLMConfig` 数据结构 + `_to_dimtable`

**Files:** Create `cost_eval/llm_config.py`; Test `tests/test_llm_config.py`

- [ ] **Step 1: 失败测试**——LLMConfig 默认值 + `_to_dimtable` 映射

```python
# tests/test_llm_config.py
from cost_eval.llm_config import LLMConfig, to_dimtable
def test_defaults_and_dimtable():
    c = LLMConfig(num_layers=4, hidden_size=1792, num_attention_heads=8, vocab_size=129280, seq_length=4096)
    assert c.attn_type == "gqa" and c.residual_variant == "plain"
    d = to_dimtable(c)
    assert d.H == 1792 and d.vocab == 129280 and d.S == 4096
```

- [ ] **Step 2:** 跑 → 失败。
- [ ] **Step 3:** 实现 `LLMConfig`（设计 §4 全字段）+ `to_dimtable(cfg)->DimTable`（字段映射，head_dim 默认 H//n_heads）。
- [ ] **Step 4:** `python -m pytest tests/test_llm_config.py -q` 绿。
- [ ] **Step 5: commit** `feat(llm_config): LLMConfig 数据结构 + to_dimtable`

### Task 0.3: 注册表骨架

**Files:** Create `cost_eval/layers/registry.py`; Test `tests/test_registry.py`

- [ ] **Step 1: 失败测试**：`ATTN_REGISTRY["gqa"]` / `["mla"]` 返回 callable；`FFN_REGISTRY["dense"]/["moe"]`。
- [ ] **Step 2:** 跑 → 失败。
- [ ] **Step 3:** 实现 `ATTN_REGISTRY = {"mha":..,"gqa":build_gqa_attn_ops,"mla":build_mla_attn_ops}`、`FFN_REGISTRY = {"dense":build_dense_ffn_ops,"moe":build_moe_ffn_ops}`（dsv4_hybrid/mhc 占位，Phase 2 填）。
- [ ] **Step 4:** 绿。 **Step 5: commit** `feat(registry): attn/ffn op-builder 注册表骨架`

---

## Phase 1 — `build_llm_spec` 核心 + Tier-1 + deepseek_v3 preset【硬门】

### Task 1.1: layer_pattern 生成器

**Files:** Create `cost_eval/build_llm.py`; Test `tests/test_layer_pattern.py`

- [ ] **Step 1: 失败测试**——first_k_dense / moe_layer_freq 展开

```python
from cost_eval.build_llm import gen_layer_pattern
from cost_eval.llm_config import LLMConfig
def test_dsv3_pattern():
    c = LLMConfig(num_layers=4, hidden_size=1, num_attention_heads=1, vocab_size=1, seq_length=1,
                  attn_type="mla", num_moe_experts=8, first_k_dense_replace=1)
    # embedding + 1 dense + 3 moe + lm_head
    assert gen_layer_pattern(c) == ["embedding","mla_dense","mla_moe","mla_moe","mla_moe","lm_head"]
def test_mtp_appended():
    c = LLMConfig(num_layers=2, hidden_size=1, num_attention_heads=1, vocab_size=1, seq_length=1,
                  attn_type="mla", num_moe_experts=8, first_k_dense_replace=1, mtp_num_layers=1)
    assert gen_layer_pattern(c)[-2:] == ["mtp","lm_head"]
```

- [ ] **Step 2:** 跑 → 失败。 **Step 3:** 实现 `gen_layer_pattern`：embedding + 每层 key（`{attn_type}_{dense|moe}`，由 first_k_dense_replace / moe_layer_freq 决定 dense/moe；dsv4_hybrid 编码 `compress_ratio`）+ `mtp`×n + lm_head。 **Step 4:** 绿。 **Step 5: commit**

### Task 1.2: `build_llm_spec` 装配器（Tier-1）

**Files:** Modify `cost_eval/build_llm.py`; Create `cost_eval/layers/head.py`（embedding/head/loss）; Test `tests/test_build_llm_tier1.py`

- [ ] **Step 1: 失败测试**——build_llm_spec 出 ModelSpec，层数/键正确，可过 Evaluator。
- [ ] **Step 2:** 跑 → 失败。
- [ ] **Step 3:** 实现 `build_llm_spec(cfg)`（设计 §5）：`to_dimtable` → `gen_layer_pattern` → 每唯一 key 用注册表组 `norm+attn+residual + norm+ffn+residual`（`residual.py` 提供 norm/residual op；plain 残差）；`head.py:build_embedding_ops`（tie 感知）、`build_head_and_loss_ops`（loss_type=logsoftmax_nll，复用现 `validate_dsv3.build_lm_head` 的 logits/logsm/nll + bwd_scratch=8·S·B·vocab）。
- [ ] **Step 4:** 绿。 **Step 5: commit** `feat(build_llm): build_llm_spec 装配器（Tier-1）`

### Task 1.3【硬门】: deepseek_v3 preset 复现真机锚点

**Files:** Create `cost_eval/presets.py`; Test `tests/test_regression_dsv3.py`; Modify `validate_dsv3.py`

- [ ] **Step 1: 失败测试**——preset 逐桶等于旧 build_dsv3_spec + 命中锚点

```python
# tests/test_regression_dsv3.py
from cost_eval.presets import deepseek_v3
from cost_eval.build_llm import build_llm_spec
from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
from cost_eval.report import Evaluator
from validate_dsv3 import build_dsv3_spec   # 旧手写 spec（保留作 oracle）
MiB, GiB = 2**20, 2**30

def _peak(spec, N):
    full = set(range(1, N+1))
    ev = Evaluator(spec, ParallelConfig(dp_shard=2, tp=1, ep=1, pp=1, cp=1, sequence_parallel=True),
                   OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                   HardwareSpec(max_device_memory=59*GiB, framework_reserve=177*MiB),
                   RecomputeSpec(mode="full", full_layers=full), SwapSpec())
    return ev.evaluate().per_stage[0]

def test_preset_equals_oracle_and_anchor_4L():
    new = _peak(build_llm_spec(deepseek_v3(4)), 4)
    old = _peak(build_dsv3_spec(4)[0], 4)
    assert abs(new.peak_bytes - old.peak_bytes) < 1   # 逐桶等价
    assert abs(new.peak_bytes/MiB - 12472.5) < 0.5    # 真机锚点

def test_anchor_8L():
    new = _peak(build_llm_spec(deepseek_v3(8)), 8)
    assert abs(new.peak_bytes/MiB - 13896.1) < 1.0
```

- [ ] **Step 2:** 跑 → 失败。
- [ ] **Step 3:** 实现 `deepseek_v3(N)`→LLMConfig（H=1792,n_heads=8,mla dims,vocab=129280,moe 8/topk4/shared1/moe_F=1024,first_k_dense_replace=1,loss=logsoftmax_nll,params fp32）。调至逐桶等于 oracle。
- [ ] **Step 4【硬门】:** `python -m pytest tests/test_regression_dsv3.py -q` → 两测试绿。**未过不进 Phase 2。**
- [ ] **Step 5: commit** `feat(presets): deepseek_v3 preset 复现真机锚点 12472.5/13896.1【硬门】`

### Task 1.4: Tier-1 presets（llama/qwen/mixtral）+ param 守恒

**Files:** Modify `cost_eval/presets.py`; Test `tests/test_presets_tier1.py`

- [ ] **Step 1: 失败测试**——`llama(32)` 的 Σparam 对已知（如 Llama2-7B ≈ 6.7B，±2%）。 **Step 2:** 跑→失败。 **Step 3:** 实现 llama(gqa+dense)、qwen2/3(gqa+dense/moe,qk_layernorm)、mixtral(gqa+moe 无 shared)。 **Step 4:** 绿。 **Step 5: commit**

---

## Phase 2 — 前沿变体（dsv4_hybrid / mHC / MTP / loss 变体）

### Task 2.1: dsv4_hybrid 注意力 op-builder

**Files:** Create `cost_eval/layers/dsv4_hybrid.py`; Test `tests/test_dsv4_hybrid.py`
参考：设计 §7.3 + 源 `pynative/.../experimental_attention_variant/{deepseek_v4_hybrid_attention,csa,indexer,compressor}.py`。

- [ ] **Step 1: 失败测试**——按 compress_ratio 分支，内存大头张量存在且 shape 正确

```python
# 断言 ratio=4(CSA) 图含 index_scores[B,S,S] fp32 与 kv_gathered[B,S,topk,vd]；
# ratio=128(HCA) 图含 compressed_kv[S/128,...] 且无 index_scores；ratio=0 只滑窗。
```

- [ ] **Step 2:** 跑→失败。 **Step 3:** 实现 `build_dsv4_hybrid_attn_ops(d, compress_ratio)`：MLA base + 索引器（ratio=4）+ 压缩器 + 稀疏注意力 + 分组输出；`index_scores` 建为 op 的 `workspace`/`bwd_scratch`（O(S²)，可重算不 save），`kv_gathered` 建为 saves（O(S·topk)）。注册进 `ATTN_REGISTRY["dsv4_hybrid"]`。 **Step 4:** 绿。 **Step 5: commit**

### Task 2.2: mHC 残差包装器

**Files:** Create `cost_eval/layers/residual.py`（扩展）; Test `tests/test_mhc.py`
参考：设计 §9 + 源 `pynative/.../hyper_connection.py`,`transformer_block.py`。

- [ ] **Step 1: 失败测试**——mhc_wrap 使层 `act_live` 残差 ×num_residual_streams；每层多 `h_res[S,B,n,n]` saves。
- [ ] **Step 2:** 跑→失败。 **Step 3:** 实现 `mhc_wrap(body_ops, n)`：把残差承载 hidden 的 saves ×n（expand/collapse 在 embedding 后/head 前，由 build_llm_spec 插）；每层加 2 个 HC 模块的 norm+proj+sinkhorn op（h_res saved）。 **Step 4:** 绿。 **Step 5: commit**

### Task 2.3: MTP 头 + loss 变体（chunked / vocab_parallel_ce）

**Files:** Modify `cost_eval/layers/head.py`; Test `tests/test_mtp_loss.py`
参考：设计 §10 + 源 `pynative/loss/loss.py`（chunk）、`multi_token_prediction.py`。

- [ ] **Step 1: 失败测试**——`chunk_loss_num=k` 使 loss 区 `bwd_scratch` ÷k；MTP 头追加 embedding+1层+head 结构。
- [ ] **Step 2:** 跑→失败。 **Step 3:** 实现 chunked（bwd_scratch=8·S·B·vocab/chunk）、vocab_parallel_ce（loss 张量按 tp 切）、`build_mtp_ops`。 **Step 4:** 绿。 **Step 5: commit**

### Task 2.4: deepseek_v4 preset + SWA 字段

**Files:** Modify `cost_eval/presets.py`, `cost_eval/build_llm.py`; Test `tests/test_dsv4_preset.py`

- [ ] **Step 1: 失败测试**——`deepseek_v4(N)` 出 spec（dsv4_hybrid + mhc + mtp），可过 Evaluator 无异常；`window_size` 不改 op 图（SWA 内存中性，§7.4）。
- [ ] **Step 2:** 跑→失败。 **Step 3:** 实现 deepseek_v4 preset（csa_compress_ratios、mhc n=4、mtp=1）；build_llm_spec 对 window_size 不改图。 **Step 4:** 绿。 **Step 5: commit**

---

## Phase 3 — 迁移现有驱动到 preset（输出不变）

### Task 3.1: validate_dsv3 / analyze_matrix / timeline_probe 改调 preset

**Files:** Modify `validate_dsv3.py`, `analyze_matrix.py`, `timeline_probe.py`; Test `tests/test_migration_outputs.py`

- [ ] **Step 1: 失败测试**——迁移后 4L/8L 峰值、analyze_matrix 各行数值、timeline 峰值**逐字节不变**。
- [ ] **Step 2:** 跑→失败（若有差）。 **Step 3:** 把 `build_dsv3_spec` 内部改为 `build_llm_spec(deepseek_v3(N))`（保留函数签名）。 **Step 4:** 绿 + `python -m pytest -q` 全绿。 **Step 5: commit**

---

## Phase 4 — 真机验证（DeepSeek-V4 新结构）

> 用 skill `real-machine-memory-sim`（§6 timeline + §4 峰值）。共享机礼仪：1–2 卡、缩层、跑完即清。

### Task 4.1: dsv4_hybrid + mHC 缩层真机峰值 vs 预测

- [ ] **Step 1:** 在 116/shb.ms.2.9 用 mindformers deepseek_v4 配置缩层（4 层）跑 FSDP-2，采 `max_memory_allocated` + 峰值算子（Profiler operator_memory）。
- [ ] **Step 2:** 评估器 `deepseek_v4(4)` 预测峰值，对比 ratio；峰值算子是否落在 dsv4 稀疏注意力 / loss 区（印证内存大头）。
- [ ] **Step 3:** 若偏差 >10%，回看 dsv4_hybrid/mHC op 图哪个桶与真实不符（对照源码，勿杜撰），修 op（**不动已验证的 dense/MLA/loss**）。标定 framework_reserve（只换常数）。
- [ ] **Step 4:** 把真机点写进设计 §8.7 验证表 + skill 结果表。 **Step 5: commit**

---

## 收尾
- [ ] 全量 `python -m pytest -q` 绿；`deepseek_v3` 锚点 1.000 未回归。
- [ ] final code review（superpowers:requesting-code-review）。
- [ ] 合并决策：Phase 1 硬门过 + Phase 4 真机点通过后，再考虑合入（用户历次要求：验证后才合）。

## 自查（writing-plans self-review）
- **Spec 覆盖**：Tier-1（0-1）、dsv4_hybrid（2.1）、mHC（2.2）、MTP/loss（2.3）、SWA（2.4）、presets（1.3/1.4/2.4）、迁移（3）、真机（4）——全覆盖设计 §3-§13。
- **类型一致**：`build_llm_spec`/`gen_layer_pattern`/`to_dimtable`/`LLMConfig` 全程同名。
- **无占位**：每 Task 有 test-first + 命令 + 期望；前沿 op 细节引设计 §7-§10 + 源码（非杜撰）。
- **硬门**：Task 1.3 deepseek_v3 复现真机锚点未过不进 Phase 2。
