---
name: real-machine-memory-sim
description: >-
  Run MindFormers PyNative on the real Ascend NPU server (192.168.9.116, container shb.ms.2.9) to
  MEASURE actual per-device peak memory, and compare it against the cost_eval evaluator's PREDICTED
  peak to validate and calibrate the simulator. USE THIS whenever the task is: validate / calibrate
  the memory evaluator against real hardware, capture真机显存 / measured peak memory for a parallel
  config, run a mindformers pynative job on 116/shb.ms.2.9 to collect memory, or close the
  "P0 §10 验证阶梯 step 4-5"（缩层 profiling 对接 / framework_reserve 标定）. It owns the verified
  connect path, Ascend env sourcing, the MindSpore memory API, and the measured-vs-predicted loop.
---

# Real-Machine Memory Simulation (MindFormers PyNative on Ascend 116)

把 `cost_eval` 评估器**预测的峰值显存**接到 **mindformers 真机实测峰值**上，做验证与标定。这是仿真器开发的"真机仿真"环节（对应设计 P0 §10 验证阶梯 step 4–5）。

```
评估器预测 peak_bytes   ──对比──>   真机实测 max_memory_allocated
   (cost_eval.Evaluator)              (mindformers PyNative on Ascend)
                         └── 差值 → 标定 HardwareSpec.framework_reserve (O_framework) / η ──┘
```

## 0. 环境（已实测验证 2026-06-30）

| 项 | 值 |
|---|---|
| 服务器 | `192.168.9.116`（hostname `ascend116`），SSH 用户 **root**（key 已配，`ssh 192.168.9.116`） |
| 容器 | `shb.ms.2.9`（镜像 `mindspore2.9:9.0.0-beta.2`，`docker exec` 默认 root） |
| 代码 | `/home/suhaibo/workspace/deepseek_v4/mindformers` |
| 加速卡 | **8× Ascend NPU**，单卡 HBM ≈ **64 GB**（实测 total 65452113920 B；config 默认 `max_device_memory: "59GB"`）—— **共享机器，探针只用 1–2 卡、只跑几步** |
| 框架 | MindSpore **2.10.0**，CANN **9.0.0-beta.2** |

### 连接配方（逐层）
```bash
ssh 192.168.9.116                                              # root@ascend116（key 已配）
docker exec -it shb.ms.2.9 bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh            # ★ 必须，否则 Ascend 跑不起来
cd /home/suhaibo/workspace/deepseek_v4/mindformers
```
单条非交互（本机直接驱动，推荐脚本化）：
```bash
ssh 192.168.9.116 'docker exec shb.ms.2.9 bash -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh; cd /home/suhaibo/workspace/deepseek_v4/mindformers; <CMD>"'
```

### ⚠️ 关键坑（实测踩过）
1. **非交互 shell 不带 Ascend 环境**：`ASCEND_HOME_PATH` 默认空，直接 `import mindspore; set_device("Ascend")` 会报 `libge_runner.so: cannot open shared object file`。**先 `source set_env.sh`**（设好后 `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0-beta.2`）。
2. **116 SSH 用户是 root**，不是 suhaibo（用 suhaibo 会 publickey 拒绝）。容器内代码归属 `/home/suhaibo/...` 但 `docker exec` 进去是 root。
3. 网络抖动找 `fix-116-network` skill；本 skill 假设已连通。

## 1. 显存采集 API（已实测验证）

MindSpore `ms.runtime`（与 `ms.hal` 等价）提供：
```
reset_peak_memory_stats()    # 清零峰值统计（每次测量前调）
max_memory_allocated()       # 峰值"已分配"字节  ← 对标评估器 peak_bytes（实张量占用）
max_memory_reserved()        # 峰值"已预留"字节（含内存池/碎片，≈评估器 peak + framework_reserve）
memory_allocated() / memory_reserved()
memory_summary()             # 文本拆解（可解析出各类占用）
```
**自检脚本**（确认采集机制能用，单卡、~256MiB、秒级）：见 `probe_device_mem.py`。**已实测跑通**，输出 `[PROBE] device=0 max_allocated_MiB=256.0 max_reserved_MiB=258.0`。运行（stdin 管道**必须 `docker exec -i`**）：
```bash
ssh 192.168.9.116 'docker exec -i shb.ms.2.9 bash -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh; python -"' \
  < .claude/skills/real-machine-memory-sim/probe_device_mem.py
```
选空闲卡：`PROBE_DEVICE=<id>`（跑前 `npu-smi info` 看哪张空；卡 0 常被占）。

> **对标关系**：评估器 `report.per_stage[s].peak_bytes` ↔ 真机该 rank 的 `max_memory_allocated()`；评估器的 `O_framework`(framework_reserve) ↔ `max_memory_reserved() − max_memory_allocated()`（内存池预留+碎片）。

## 2. 在 mindformers 训练里采集（TrainerCallback 挂点）

pynative trainer 有回调系统：`mindformers/pynative/callback/callback.py:22` `TrainerCallback`，钩子 `on_step_begin(self, args, state, **kwargs)` / `on_step_end(...)`（参考已有 `LossCallback`、`MaxLogitsMonitor`）。

用 `mem_probe_callback.py` 里的 `MemoryProbeCallback`：在 warmup 步后 `reset_peak_memory_stats()`，下一步 `on_step_end` 读 `max_memory_allocated()` 并按 rank 打印 `[MEMPROBE] rank=.. peak_alloc_MiB=.. peak_reserved_MiB=..`。把它加入 trainer 的 callbacks 列表即可（构造 `Trainer(..., callbacks=[MemoryProbeCallback()])`，或按该版本 trainer 的注册方式）。

> ⚠ 首次使用确认 `state` 的步计数属性名（grep `mindformers/pynative/callback/callback.py` 的 `state` 用法；常见 `state.global_step`）。脚本里已做容错（多候选属性名）。

## 3. 启动 pynative（缩层）训练

路由：`run_mindformer.py:62` —— YAML 顶层 `mode: 1` → `PynativeTrainer(config.config).train()`（否则走 GRAPH_MODE）。
- 单机多卡：`bash scripts/msrun_launcher.sh "python run_mindformer.py --config <yaml>" <WORKER_NUM> ...`（msrun 拉起多 rank；具体参数见脚本头）。
- **缩层**：把 YAML 的模型 `num_layers` 改小（如 61→2~4 层）做"缩层规格"——per-layer 同质，峰值可按层数外推，省卡省时（设计 P0/缩层 profiling 的核心）。
- 仓库内 `configs/`、`research/deepseek3/` 只有**推理** yaml；**pynative 训练 yaml 需用用户自己的**（位置首次运行时与用户确认；结构 = 顶层 `mode:1` + `config:` 块，含 `training/parallelism/recompute/swap/model/context`，对应 `mindformers/pynative/config/config.py:TrainConfig`）。

## 4. 测量 vs 预测：验证 / 标定回路

1. 选一个配置 C（模型维度 + 5D 并行度 + recompute + swap），**缩层**。
2. **真机**：按 §3 跑几步，§2 回调采每 rank `peak_alloc` / `peak_reserved`（或 §1 自检式直接读）。
3. **评估器**：用**同一** C 构造 `cost_eval` 的 `ModelSpec`+`ParallelConfig`+`HardwareSpec`+`RecomputeSpec`+`SwapSpec`，`Evaluator(...).evaluate()` 取 `report.per_stage[s].peak_bytes` 与 `breakdown`。
4. **对比**：`measured_alloc` vs `predicted peak_bytes`（应接近）；`measured_reserved − measured_alloc` → 标定 `HardwareSpec.framework_reserve`（O_framework）。分项偏差大时回看 §1 `memory_summary()` 定位（激活 / 参数 / 临时）。
5. **回填**：把标定出的 `framework_reserve`（及将来 η）写回评估器默认值——**只换常数、不动结构**（设计铁律）。

期望：标定后显存预测 <10%。差异系统性偏大→检查评估器某条 op 的 `saves`/切分是否与真实实现不符（对照 `mindformers/pynative/` 源码核实，勿杜撰）。

## 5. 安全 / 礼仪（共享机）
- 8 卡共享：探针/缩层只占 1–2 卡（msrun `WORKER_NUM` 小、或单卡），只跑 **3–10 步**即够采峰值。
- 跑前 `npu-smi info` 看哪些卡空闲，选空闲卡（`ASCEND_RT_VISIBLE_DEVICES=<id>`）。
- 不要长跑、不要占满 8 卡、跑完即停；临时脚本用完删。

## 关联
- 评估器：`cost_eval/`（本仓库），预测侧。
- 设计：`specs/2026-06-29-p0-modelspec-and-memory-design.md` §10 验证阶梯、`specs/2026-06-23-...-design.md` §3 精度预期（η 标定缝）。
- 网络故障：`fix-116-network` skill。
