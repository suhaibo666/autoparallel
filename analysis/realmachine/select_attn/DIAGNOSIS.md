# 选择性重算真机验证 —— D-3 系统性欠预测（2026-07-07）

真机：`ascend116` / `shb.ms.2.9` / cards 0,1 / DSv3 8L MLA+MoE、dp=2、SP off、GBS=2、pp=1、无 recompute 全存基线。
mindformers 配置：`recompute: {mode: select, select_module: {<module>: [0-7]}}`（`RecomputeConfig` config.py:704/710）。

## 扫描结果（select 一个模块 = 重算它、保留其余）

| select_module | 真机 MiB | 估计器 MiB | ratio | 备注 |
|---|---|---|---|---|
| `self_attention` | **18828.2** | 15361.5 | **0.816** | 重算 attn、保 FFN saves |
| `feed_forward` | **19967.3** | 14553.6 | **0.729** | 重算 FFN、保 attn saves |
| both（attn+ffn ≈ full） | 13953（≈full 锚点） | 13833.1 | **0.991** ✅ | 退化 = full 重算，精确 |
| 参照 full | 13953 | 13833.1 | 0.99 | |
| 参照 none | 26182（est） | 26182.0 | — | |

**关键**：**退化端（both ≈ full）精确 0.991**，但**部分选择系统性欠预测**（0.82 / 0.73，OOM-不安全向）。
且真机呈现一个**反转**：真机重算 FFN 比重算 attn 省得多（19967>18828 → 说明 **attn saves 更大**），
而估计器预测反了（15361>14553）——估计器把 attn/FFN 的保留激活量级估反了。

## 峰值 live-set 重构（`operator_memory.csv`，self_attention，alloc≤T_peak<release）

峰值算子 = **ScatterAddExt（loss 反向）**，算子级 live = **13630 MiB**（+ persistent ~5198 = 18828）。

| 真机 size 类 | ×数 | 小计 | 估计器 |
|---|---|---|---|
| 2020 fp32 vocab | 3 | 6060 | ✅ logsm/probs/grad |
| 1010 bf16 logits + 883.8 grad + 441.9 | — | 2336 | ✅ |
| **112 MiB** | **14** | **1568** | ❌ 保留的 FFN/MoE 层激活 |
| **<100 尾** | **313** | **3666** | ❌ **保留的 FFN/MoE 无重算激活尾 + attn 重算工作集** |

## 归因（byte 级）

`18828 − 15361 = 3467 ≈ 真机 <100 小张量尾 3666`。估计器 `estimate_select_memory`（D-3）：
- **退化端正确**（both==full 0.991，select-none==none 已验）——机制对。
- **部分选择欠计**：选择性重算下，**非选中模块（FFN/MoE）的保留激活 + 反向工作集**在 loss 峰值处
  与 loss 区**共存**，真机呈 14×112 + 大量小张量尾（GroupedMatmul/MatMulExt/RmsNorm = FFN 前向 saves）。
  估计器 `act_live`(4684) + 反向三桶把这部分算小了 —— **与 pp stage1 / DSv4-fused 同族**（§D-10 ①、D-5）：
  **无重算/保留的 MoE 层激活 + 反向工作集在峰值处欠计**。

## 结论

- **D-3 选择性重算机制正确**（退化端逐字节复现 full/none），但**部分选择系统性欠预测 ~18-27%**（OOM-不安全）。
- **根因非 D-3 本身**，而是**保留模块（尤其 MoE）的无重算激活/反向工作集**在 loss 峰值处的欠计——
  同 §D-10 ①（unfused CE 已修 loss 区本身）、DSv4-7%（D-5）。选择性重算把「保留一部分层激活」放大暴露了它。
- **未修**（本轮为验证+定位）：要 OOM-safe 需把保留 MoE 层的 forward saves + bwd_working_set 口径按真机校准
  （§8.5② MoE 分支），或对 select/no-recompute 配置设显式 margin。列入待办（同族欠预测统一处理）。

## 产物
`operator_memory.csv`（19251 行，self_attention profiler）。真机配方沉淀见 `.claude/skills/real-machine-memory-sim/SKILL.md`。
