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

## 3. 启动 pynative（缩层）训练 —— 已验证 DSv3 端到端（2026-06-30）

**入口**：`PynativeTrainer(config="<yaml>").train()`（参考测试 `run_deepseek3.py` 的用法）。yaml 顶层直接是 `checkpoint/training/parallelism/recompute/model/...`（**无** `mode:`/`config:` 块）；trainer 内部**自动补** `context.mode:1` + `max_device_memory:"59GB"`(=OOM 阈值)，并把 model 转成 megatron `TransformerConfig`。（`run_mindformer.py:62` 的 `mode==1` 是另一条等价入口。）

**参考配置**（DSv3 MLA+MoE）：`<CK>/tests/st/test_multi_cards_cases/test_pynative/test_models/test_deepseek3/pynarive_ds3.yaml`，同目录 `run_deepseek3.py`、`test_two_cards.py`（含 `generate_mindrecord_file` 造合成数据）。
> 116 上有两份 mindformers：`deepseek_v4/mindformers`(主代码) 与 `mindformers/mindformers`(含本测试 harness，本 skill 在此跑通)。下文 `CK=/home/suhaibo/workspace/mindformers/mindformers`。

**缩层**：改 model `num_hidden_layers`（如 61→4）—— per-layer 同质、峰值按层数外推，省卡省时。

### 已验证一键跑法（bundled `prep_ds3_sim.py` + `run_ds3_memprobe.py`）
```bash
CK=/home/suhaibo/workspace/mindformers/mindformers
REF=$CK/tests/st/test_multi_cards_cases/test_pynative/test_models/test_deepseek3
# 传脚本（本机 → 容器，stdin 必须 docker exec -i）：
ssh 192.168.9.116 "docker exec -i shb.ms.2.9 bash -c 'cat > $REF/prep_ds3_sim.py'"     < prep_ds3_sim.py
ssh 192.168.9.116 "docker exec -i shb.ms.2.9 bash -c 'cat > $REF/run_ds3_memprobe.py'" < run_ds3_memprobe.py
# 跑（关键 env：source set_env + conda bin 上 PATH(找 msrun) + 仓库根上 PYTHONPATH）：
ssh 192.168.9.116 "docker exec shb.ms.2.9 bash -lc '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export PATH=/root/miniconda3/envs/mindspore2.10/bin:\$PATH
  export PYTHONPATH=$CK:\$PYTHONPATH
  cd $CK && SIM_LAYERS=4 SIM_STEPS=3 python $REF/prep_ds3_sim.py
  msrun --worker_num=2 --local_worker_num=2 --master_port=8124 --log_dir=$REF/log_sim --join=True \
        $REF/run_ds3_memprobe.py --config $REF/ds3_sim.yaml
'"
# 取结果：
ssh 192.168.9.116 "docker exec shb.ms.2.9 bash -c 'grep -h MEMPROBE $REF/log_sim/worker_*.log'"
```

### 实测结果（4 层 DSv3，FSDP-2，seq4096，full 重算，compute=bf16/params=fp32）
```
[MEMPROBE] rank=0 peak_alloc_MiB=12473.1 peak_reserved_MiB=13446.0 framework_reserve_MiB=972.9
[MEMPROBE] rank=1 peak_alloc_MiB=12473.1 peak_reserved_MiB=13474.0 framework_reserve_MiB=1000.9
```
→ 每卡峰值 **~12.47 GB 已分配 / ~13.45 GB 预留**，**framework_reserve ≈ 0.97–1.0 GB**（即评估器 `O_framework` 标定值）。首步 6.6s（kernel 编译）、后续 ~430ms/步，exit 0。

### 关键 env 坑（实测踩过）
- **msrun 不在默认 PATH** → 在 `/root/miniconda3/envs/mindspore2.10/bin/`，需 `export PATH=...:$PATH`。
- **PYTHONPATH 必须含 mindformers 仓库根**（否则 `tests.utils` / 本仓库 `mindformers` import 失败）。
- 嵌套 `ssh→docker exec bash -lc '...'` 里 echo **别带 `()` 等元字符**（inner bash 会语法报错）。

## 4. 测量 vs 预测：验证 / 标定回路

1. 选一个配置 C（模型维度 + 5D 并行度 + recompute + swap），**缩层**。
2. **真机**：按 §3 跑几步，§2 回调采每 rank `peak_alloc` / `peak_reserved`（或 §1 自检式直接读）。
3. **评估器**：用**同一** C 构造 `cost_eval` 的 `ModelSpec`+`ParallelConfig`+`HardwareSpec`+`RecomputeSpec`+`SwapSpec`，`Evaluator(...).evaluate()` 取 `report.per_stage[s].peak_bytes` 与 `breakdown`。
4. **对比**：`measured_alloc` vs `predicted peak_bytes`（应接近）；`measured_reserved − measured_alloc` → 标定 `HardwareSpec.framework_reserve`（O_framework）。分项偏差大时回看 §1 `memory_summary()` 定位（激活 / 参数 / 临时）。
5. **回填**：把标定出的 `framework_reserve`（及将来 η）写回评估器默认值——**只换常数、不动结构**（设计铁律）。

期望：标定后显存预测 <10%。差异系统性偏大→检查评估器某条 op 的 `saves`/切分是否与真实实现不符（对照 `mindformers/pynative/` 源码核实，勿杜撰）。

### 已做的 DSv3 对标 + 标定（2026-06-30，`validate_dsv3.py`）
MLA op 图（`cost_eval/layers/mla.py`）+ 框架反向瞬态建模已加。对 4 层 DSv3 / FSDP-2 / full 重算：

| 对标项 | 评估器预测 | 真机实测 | 结果 |
|---|---|---|---|
| **常驻 persistent**（param+m+v, fp32=12B/param, /dp_shard） | 3829.9 MiB | 3862 MiB | **误差 0.8% ✅** |
| **峰值 peak（纯结构）** | 10275 MiB | 12473 MiB | ratio **0.82** |
| **峰值 peak（+ framework_reserve=2197）** | 12472.5 MiB | 12473 MiB | ratio **1.000** ✅ |
| **8L 峰值（同一 reserve，泛化复验）** | 13896.1 MiB | 13953.3 MiB | ratio **0.996** ✅ |
| **4L ep=2 峰值（跨 ep，证伪 HCCL 假设）** | 12472.5 MiB | 12474.1 MiB | ratio **1.000** ✅ |

> **ep=2 真机点的关键发现**：多一个 EP 通信组，**allocated 峰值不变**（12473→12474）、**reserved 涨**（13446→13750）→ **HCCL 缓冲在 reserved 池、不在 allocated 峰值**。评估器预测 `max_memory_allocated`(OOM 相关) 故 **HCCL 不计入 framework_reserve**；framework_reserve 经 ep=1/2 验证**对 ep 恒定**。这证伪了之前"HCCL×组数进 allocated"的假设——变配置真机验证的价值。

峰值预测演进：**0.39 →(加 loss 区 fp32)→ 0.66 →(加 FSDP gather/grad + bwd_scratch)→ 0.82 →(标定 reserve)→ 1.00**。

**结构 breakdown（4L，MiB）**：persistent 3830 + act_live 3100(logits bf16 + **log_softmax fp32**) + gather_buf 442(FSDP 全参 bf16) + grad_buf 884(full grad fp32) + **bwd_scratch 2020(loss probs fp32)** + framework 2197。

**关键发现 / 已修**：
1. **参数量精确**（669M = 手算 669.35M），MLA/MoE/embedding/lm_head op 图无误。
2. **grad 非常驻**：fp32 AdamW 常驻=param+m+v=**12B/param**（grad 是反向瞬态）。已分离 持久(param+opt) vs 瞬态(grad)。
3. **大 vocab loss 区是峰值大头**：`loss.py` log_softmax/probs 走 **fp32**，vocab=129280×seq4096 → log_softmax(saved 2118MiB)+probs(反向 2118MiB)≈4.2GB。已用 per-tensor dtype + `OpSpec.bwd_scratch` 建模。
4. **残余 2.2GB = 框架不可解析瞬态**（MoE all-to-all/hccl 200MB×组/flash workspace/碎片）→ 收进标定的 `framework_reserve`（每平台标一次）。

> 结论：**评估器已真机验证（2 数据点）**：静态 0.8%；峰值标定后 **4L=1.000 / 8L=0.996**。`framework_reserve=2197` 从 4 层标定、**泛化到 8 层仅差 0.4%**（结构项随层数 persistent 3830→5198 自动升、reserve 恒定）。结构项全部 source-grounded，仅 1 个平台常数。

## 5. 安全 / 礼仪（共享机）
- 8 卡共享：探针/缩层只占 1–2 卡（msrun `WORKER_NUM` 小、或单卡），只跑 **3–10 步**即够采峰值。
- 跑前 `npu-smi info` 看哪些卡空闲，选空闲卡（`ASCEND_RT_VISIBLE_DEVICES=<id>`）。
- 不要长跑、不要占满 8 卡、跑完即停；临时脚本用完删。

## 关联
- 评估器：`cost_eval/`（本仓库），预测侧。
- 设计：`specs/2026-06-29-p0-modelspec-and-memory-design.md` §10 验证阶梯、`specs/2026-06-23-...-design.md` §3 精度预期（η 标定缝）。
- 网络故障：`fix-116-network` skill。
