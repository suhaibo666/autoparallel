# 真机内存 timeline vs 仿真内存 timeline（DSv3 4L FSDP-2）

用 **MindSpore Profiler(`profile_memory=True`)** 采真机逐时刻 allocated/reserved 曲线，
对标 `cost_eval` 仿真器的逐事件内存曲线（`Evaluator.evaluate(record_timeline=True)`）。

- 真机：`192.168.9.116` / 容器 `shb.ms.2.9`，cards 0,1，DSv3 4 层、FSDP-2、seq4096、full 重算、compute bf16/params fp32，3 步。
- runner：`.claude/skills/real-machine-memory-sim/run_ds3_memtimeline.py`（trainer + `ms.Profiler(profile_memory=True)`）。
- 仿真：`analysis/realmachine/compare_realmachine_timeline.py`（解析 profiler CSV + 建仿真曲线 + 出图）。
- 图：`comparison.png`（上=真机 allocated-vs-time 3 步；下=仿真逐事件）。

## 关键结果

| 台阶（MB） | 真机 | 仿真 | 真机实测锚点 |
|---|---|---|---|
| 基线 / persistent | 3922.3 | 4063.8 | 3862 |
| **峰值 peak** | **12473.1**（算子级）| **12472.5** | 12473.1 |
| 仿真/实测峰值比 | — | — | **1.0000** |

- **峰值算子 = `ScatterAddExt`**，其 `Allocation Total Allocated = 12473.1 MB` = MEMPROBE 峰值。
  这正是 `pynative/loss/loss.py:82` `_NLLLoss.backward` 里的 `mint.scatter_add(probs, ...)`——
  **真机独立证实：单卡峰值就发生在 loss 反向的 scatter_add（grad_log_softmax 物化）那一刻**，
  即前几轮从 `framework_reserve` 拆进 `nll.bwd_scratch` 的那块 ~2020 MiB 是**真实 loss 反向激活、非框架开销**。
- 曲线**形状一致**：前向低平（full 重算压掉激活）→ 进 loss 区抬起 → **loss 反向一根尖峰** → 反向逐层回落，周期性 3 步。
- `memory_record.csv` 的曲线全局 max = 12772.5 MB（含 ~300 MB profiler 自身开销）；干净模型峰值以**算子级 12473.1** 与 MEMPROBE 为准。

## 如何看"真机 vs 仿真曲线波动差异大"（`comparison.png` 全程 vs `comparison_onestep.png` 单步）

不是 bug，是**分辨率 + 范围**差异（设计层面的口径，详见设计 §8.9）：

1. **分辨率**：真机 ~2400 采样/s（逐 kernel alloc/free）vs 仿真 13 点/步（逐 layer 事件）→ 真机天然锯齿、仿真平滑。
   **仿真是峰值包络模型，不是逐 kernel tracer**。
2. **范围**：仿真只建**一个稳态 step** 的逻辑峰值包络；真机曲线是整 8.7s——含 warmup+编译（首步大 hump）、
   步间/优化器/数据空档（~7100 MB 稀疏平台）、末尾 3 个 fast step。多数波动是仿真**故意不建**的运行时行为。
3. **该匹配的匹配了**：`comparison_onestep.png` 把真机缩到峰值所在单步，形状与仿真一致——
   前向低平 → loss 反向**单尖峰**（ScatterAddExt）→ transformer 反向若干小台阶 → 回落。峰值值+位置都对上。
4. **一个 off-peak 观察**：真机 transformer 反向台阶 ~6000–6800 MB 比仿真 bwd@transformer ~4700–5300 高 ~1500 MiB，
   在峰值以下不影响 OOM 预测，但与设计 §8.5② 一致（`recomp_scratch` 欠建）——曲线级验证把这个欠建看见了。

> 验收口径：内存仿真只对**峰值大小 + 峰值位置**负责（OOM/选型只需这两件），已 1.000 + 算子级点名对上；细结构差异属分辨率与范围。

## 复现

```bash
CK=/home/suhaibo/workspace/mindformers/mindformers
REF=$CK/tests/st/test_multi_cards_cases/test_pynative/test_models/test_deepseek3
# 1) 传 prep + timeline runner（见 SKILL.md §3 传法），SIM_LAYERS=4 SIM_STEPS=3 python prep_ds3_sim.py
# 2) 跑 profiler：
ASCEND_RT_VISIBLE_DEVICES=0,1 msrun --worker_num=2 --local_worker_num=2 --master_port=8126 \
  --log_dir=$REF/log_tl --join=True \
  $REF/run_ds3_memtimeline.py --config $REF/ds3_sim.yaml --prof_out $REF/prof_mem
# 3) 拉 memory_record.csv / operator_memory.csv 到 analysis/realmachine/，跑 compare_realmachine_timeline.py
```

> 原始 profiler CSV（memory_record / operator_memory，各数 MB）已 gitignore，不入库；本 README + `comparison.png` 为留存证据。
