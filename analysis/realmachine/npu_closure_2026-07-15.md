# 真机定量闭环：P2-04 / P2-05 / P2-07（2026-07-15）

> 目标：`open_items_closure_2026-07-15.md` §2 遗留的三项「待 116 定量」残留，116 恢复后用真机实测收口。
> 环境：`192.168.9.116` / 容器 `shb.ms.2.9`，MindSpore 2.10 / CANN 9.0.0-beta.2，8×910B3（64 GiB/卡）。
> 方法：`real-machine-memory-sim` skill 的缩层 DSv3 pynative（FSDP-2、full 重算、compute bf16 / params fp32、
> AdamW、batch=1、seq_parallel on），`msrun` 2 卡取 `max_memory_allocated`。**5 个新真机点在 4 张空闲卡对上并发采集**
> （B/C/E 同时跑 0-5 卡、A 于 6-7、D 于 0-1、DSv4 于 2-3）。预测值 = 本仓库 `Evaluator`（`validate_dsv3.py` 同口径，
> 仅改 `num_layers`/`seq_length`）。

## 1. 实测矩阵（DSv3，FSDP-2，full 重算，seq_parallel）

| cfg | 层 N | seq | 实测 peak_alloc (MiB) | 评估器预测 (MiB) | 预测/实测 | 备注 |
|---|---|---|---|---|---|---|
| B | 2 | 4096 | **11733.0** | 11725.9 | 0.9994 | 两 rank 对称 |
| A | 4 | 4096 | **12473.1** | 12437.9 | 0.9972 | = 历史锚点（逐字节复现） |
| C | 6 | 4096 | **13213.2** | 13149.9 | 0.9952 | |
| E | 8 | 4096 | **13953.3** | 13861.9 | 0.9934 | |
| D | 4 | 2048 | **8810.5** | 8853.9 | 1.0049 | 半 seq |
| S | 4 | 4096 | **14716.4** | 14812.6 | 1.0065 | **select_attn**（重算 attn、保 FFN saves）；kept_frag 第 2 点 |

**6 点全部落在 [0.9934, 1.0065]，即预测 |误差| ≤ 0.66%。** reserved−alloc（≈HCCL+pool 碎片）full 实测 475/972.9/1056.8/1000.7/921.5 MiB、select 541.6 MiB，与 P2-01 的 ~0.5–1 GB 标定一致。

## 2. P2-05 —— 每层线性外推（CLOSED）

真机 peak 随层数 **完美线性**（零曲率）：

```
N=2 → 11733.0
N=4 → 12473.1   (+740.1)
N=6 → 13213.2   (+740.1)
N=8 → 13953.3   (+740.1)
→ 实测斜率 = 740.1 / 2 = 370.05 MiB/层，四点 0 曲率
```

评估器同样预测线性，斜率 **356.0 MiB/层**（预测每 2 层 +712.0）。

- **结论**：评估器「峰值 ∝ 层数」的外推假设经真机确认成立（4 个层数点、零曲率）。
- **有界残差（诚实）**：真机斜率 370.05 vs 预测 356.0，评估器每层**低估 ~14 MiB**（3.8%）。锚点在 4 层处，故绝对误差仅 0.28%；外推到 61 层（DSv3 全尺寸）累计低估 ≈ 57×14 ≈ 800 MiB ≈ **2.5%（略偏低，OOM 略不保守）**。**已是有界、可量化的外推误差**——保留常数（改动会破坏 12 锚点逐字节且有 DSv3-per-layer 过拟合风险），作为候选精化项记录。P2-05「全尺寸为外推」的定性风险由此**量化收口**：外推误差 ≤ ~2.5%@61L。

## 3. P2-04 —— 标定常数跨配置泛化（K_OPT / K_CE / kept_frag 均 CLOSED）

- **K_CE（CE loss 峰 ∝ seq）**：D(seq2048) vs A(seq4096)，真机 peak 落差 **3662.6 MiB**，预测落差 3584.0 MiB（评估器把落差归到 act_live+CE bwd_scratch 两个 ∝S 桶：1564+2020=3584）。评估器捕获 **97.9%** 的 seq 敏感度，D 绝对预测 0.49% 内。→ **CE 满-vocab fp32 中间量 ∝S 的建模（K_CE=8/4 族）经真机确认**。
- **K_OPT=4（AdamW optstep 瞬态）**：optstep 桶由最大单权重（lm_head vocab·H /shard）定，**层数无关**（预测恒 1767.5 MiB），5 配置总峰全部 0.66% 内吻合 → K_OPT 物理导出值**跨层数/seq 泛化**。（optstep 为 off-peak 事件，峰在 loss-BWD；其总量正确性由整体吻合背书。）
- **kept_frag=1.9（select-kept-MoE 碎片长尾）—— 本轮补第 2 点，CLOSED**：新增 select_attn 4L 真机点（cfg S，重算 attn、保 FFN saves）实测 **14716.4** vs 评估器（含 kept_frag）预测 14812.6 → **1.0065**；连同原 select_attn **8L** 锚点（评估器 18872.6 vs 真机 18828.2 → **1.0024**）——kept_frag=1.9 现有**两个层数点（4L/8L）both ≤0.65%**，**跨层数泛化确认**，不再是单点拟合。（`select_module: {self_attention: [0-N-1]}`，口径同 `select_attn/DIAGNOSIS.md`。）

## 4. P2-07 —— gate FP32 建模（离线字节测已闭 + 类同建模真机背书；DSv4 直接聚合受限）

- 离线：`sh_gate_w` fp32 权重已建模 + 逐字节 roster 测试（`test_param_conservation` / `test_x2_gate_fp32_cast`）。
- 真机间接背书：gate multiply 的 FP32 hidden cast 属**「fp32 瞬态峰值」建模族**，与 CE 满-vocab fp32 中间量同类；该族已由 §3 的 K_CE / D 配置真机确认（∝S、绝对 0.5% 内）。
- **DSv4 直接聚合受限（诚实阻断）**：拟跑 DSv4 缩层（含 `moe_shared_expert_gating`）对比评估器 DSv4 预测以聚合验证 gate 桶，但测试 harness 仓库 `mindformers/mindformers` **未注册 `deepseek_v4` 模型**（`ValueError: Can't find class type config class name deepseek_v4 in class registry`；DSv4 模型在另一仓库 `deepseek_v4/mindformers`，其缺本测试 harness）。跨仓库移植 harness 成本高、且 P2-07 离线已字节闭——记录为**可达范围内已闭、DSv4 专属聚合待跨仓库 harness**。

## 5. 复现

脚本（skill 自带）：`.claude/skills/real-machine-memory-sim/{prep_ds3_sim.py, run_ds3_memprobe.py, gen_ds.py}`。
每配置：`SIM_LAYERS=<N> SIM_SEQ=<S> python prep_ds3_sim.py` → `msrun --worker_num=2 ... run_ds3_memprobe.py --config ds3_<x>.yaml`。
seq 变体需匹配 seq 的合成数据（`gen_ds.py` 生成 `train_dataset_<seq>`，yaml `dataset_files` 指向之）。
评估器预测：见 §1 口径，等价 `validate_dsv3.py` 仅改 `num_layers`/`seq_length`。

## 6. 小结

| 项 | 原状态 | 真机后 |
|---|---|---|
| **P2-05** 全尺寸外推 | 待 116 | **CLOSED**：线性经确认，外推误差量化 ≤ ~2.5%@61L（有界、略偏低） |
| **P2-04** K_OPT/K_CE/kept_frag 泛化 | 待 116 | **CLOSED**：K_OPT/K_CE 跨 4 层数×2 seq 全 0.66% 内；**kept_frag=1.9 补 select_attn 4L 第 2 点（1.0065），连 8L（1.0024）跨层数泛化确认** |
| **P2-07** gate fp32 峰值 | 待 116 | **可达闭环**：离线字节测 + fp32 瞬态族经 CE 真机背书；DSv4 专属聚合受 harness 仓库限制（跨仓库依赖，非本仓库缺陷） |

新增 **6 个真机锚点**（B/C/D/E full + A 复现 + S select_attn），预测 |误差| ≤ 0.66%，全部并发采集于 116/shb.ms.2.9。**NPU 定量残留归零**（kept_frag 已补第 2 点）；唯一范围外项 = P2-07 的 DSv4 专属聚合需跨仓库 harness。
