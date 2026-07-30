# 交接文档 —— 显存模型标定与源码抽图（2026-07-25 ~ 07-30）

> 自包含：读完可直接接手，不需要之前的对话上下文。
> 分支 `feat/unified-llm-modelspec`，HEAD `9053686`，**2030 passed**，树干净，**未 push**。
> 铁律（全程）：**绝不杜撰真机数字；绝不用拟合常数顶数字；判不出就 fail-loud 并记账，不猜。**

---

## 0. TL;DR

起点是一个诊断问题（"重算没生效还是 hyper 没生效？"），终点是一次**模型标定级重构**：

| | 起点 | 现在 |
|---|---|---|
| 逐层驻留 r0 / r4 / r128（模型/真机） | 1.677 / 1.582 / 1.713 | **0.9975 / 0.9434 / 0.9478** |
| 8 组门：平均绝对误差 | 17.3% | **15.9%** |
| 8 组门：最大单格误差 | **39.4%** | **29.7%** |
| 过读格数 | 7/28 | **4/28** |
| 区间 | [0.748, **1.394**] | [0.703, **1.038**] |
| 测量导出的 workspace 律 | 0 条 | **4 条** |
| 测试 | 1463 | **2030** |

**注意均值反而"变差"（0.946 → 0.847），这是对的**：0.946 是**过读抵消欠读**的假象。真指标是
最大偏离与过读格数，两者都大幅收窄。误差从**双向**变成基本**单向欠读**——可解释、可继续收敛。

---

## 1. 原始问题的答案：**H1 和 H2 都不成立**

用户最初问：真机激活偏大，是 **H1**（mindformers 自定义反向存在时重算没生效）还是 **H2**（hyper 没生效）？

**都不是。重算生效了，hyper 也生效了。** 167/MS2.10 单卡微基准（`log_release_probe/`）：

- rc=ON 下 `fwd_end ≡ 0`（所有锚法 × NBLK=2/4/8）→ 重算**确实**释放
- 裸 `ctx.attr` 只钉住**它实际存的那个张量**；`_IndexerLossAutoScaler` 存的是 **kl 标量**，可忽略
- rc=ON 的 `fwd_peak` **与 NBLK 无关**（复刻 552 / 真模块 403，恒定）→ 剩余是**单区域工作集，×1**
- 真模块最大的 `attention_scores` 由 `stop_gradient(query/kv)`（`csa.py:794-795`）算出 → **不需梯度、不被保存、纯瞬态**

**真答案 = 未建模的「单层重算工作集」。** 真机多卡独立验证（8 组 A/B，`REAL_SHA256=41e279e5…`）：

- **×1 于层数**：stage3 `unfused−fused` delta 在 **L8（2 层/stage）= 24380.1**、**L4（1 层/stage）= 24380.1**，|diff| = **0.0**
- **×1 于微批数**：m4↔m8 移动 ≤ **0.4 MiB**

---

## 2. 贯穿全程的系统性发现

### 2.1 聚合指标会骗人（本项目最重要的方法论教训）

均值 0.946 掩盖了**方向相反**的两类误差（无重算过读 24–39%、有重算欠读 25%）。
**每一层聚合都可能重演这个病**：后来"unfused 欠读 10.4 GiB"这个聚合值本身也是按层型反号的假象。
→ **报告一律用 MAE / 最大单格误差 / 过读格数**，不用均值。

### 2.2 「锚点在跟错误的模型对账」出现了**三次**

| 实例 | 表现 | 状态 |
|---|---|---|
| `_bundle_to_fields` 丢 **11 个** `LLMConfig` 字段 | 锚点/UI 静默取预设值（实测 35% 分歧） | 已修 + 建往返不变量 |
| `use_fused_mhc` 无扁平 query 键 | 锚点建 **unfused**、真机跑 **fused**（~960 MiB/层 × ~20 锚点） | 已修 |
| `csa_compress_ratios` 用预设**循环**而非逐层表 | 层型配比 3×r0/3×r4/2×r128 vs 真实 1×r0/4×r4/3×r128 | 已修 |

**第二次的修法制造了第三次的盲区**：`llm_json` 让"新字段自动搭车"，于是往返不变量对
"字段是否被携带"**结构上恒为真**；而锚点/探针路径**没有 `llm_json`**，基底是预设。
已在守卫处闭掉：`_LLM_JSON_ONLY_FIELDS` + `unclassified_llm_fields()` +
`tests/test_flat_query_reachability.py`（含反向测试证明闸门会响）。

---

## 3. 已完成的修复

### 3.1 census 从高估 1.64× 收到 0.94–1.00

误差**几乎全在 `act_live` 一个桶**（其余 9 桶合计只差 1.3–8.5%）。逐条源码判决（全带 `file:line`）：

| 项 | 性质 | 影响 |
|---|---|---|
| `q_hnorm_fp32` → `q_hnorm` | **dtype 错**（`deepseek_v4_hybrid_attention.py:244-245` 是纯 bf16 逐元素；`self.rms_norm` 从未被调用） | −256/层 |
| `q` 的 op `NORM`→`ELEMENTWISE` | 虚假 `_dt` 抬升（`:245` 不是 layernorm 模块） | −256/层 |
| `cg_fp32` → `cg` | dtype 错（`:288-290` 有**明确拒绝 fp32 提升**的源码注释） | −256/层 |
| `inv_rope_out` | **根本不存在**（`:215` 的 `cat` 只喂 reshape/permute，VJP 无保留） | −256/层 |
| `cmp_residual` | **形状错**（`csa.py:64-67` 是 1 元素 int32 标量） | −8 / −4 |
| 前向 RoPE `(t,t_rot)` 对 | **漏了**（`rope_utils.py:169,186-187`，`rotary_dtype=fp32`） | **+130/层** |
| 融合 mHC ctx | 建模成 unfused 而现场 `use_fused_mhc: true`；每模块 256.125 → 45.266 | −421/层 |
| `FusedRMSNorm` 不 cast | `layer_norm.py:151-155` 直接喂 `ops.rms_norm`，`self.cast`(:149) 是**死代码**（`FusedLayerNorm` :93-101 才 cast）→ 按 **norm 种类**改，`norm_compute_dtype_bytes` 全局口径未动 | −268/层 |
| indexer 内部 RoPE 保留对 | `apply_rope_fusion` 默认 False → 走非融合分支，两个 `mul` 各留一操作数 | **+128/r4 层** |
| `comb_xn` 错误放大 / RMSNorm 聚合 save 别名 | `mhc_wrap._is_residual_carrier` 同根 | −96 / +64 |

### 3.2 四条**测量导出**的 workspace 律（无拟合常数）

**原理（实测）**：workspace 生命周期**恰好 1 tick**，`fwd_exit` 边界在世量 **0.00 MiB** →
**不进驻留，但决定峰值**。方法：`MS_ALLOC_CONF=memory_tracker:True`，分类器先在单卡微基准上验到逐字节。

| # | 项 | 律 | 验证 |
|---|---|---|---|
| 1 | embedding 反向 `GatherDGradV2` | `4·vocab·H + 16 MiB + 12·H·(B·S) + 3072 B` | 4 点 / 2 站点 / 2 次活动 / 2 个 H / 2 个 B，逐字节 |
| 2 | 融合稀疏 flash-MLA 反向（r4） | `209715200 B + B·S·135680 B` | 3 点线性，斜率两区间逐位相同 |
| 3 | 融合 mHC 前向 | **304.001 MiB，与 S 无关** | 3 个 S 点 |
| 4 | `lm_head` 反向 dgrad / wgrad | `(2·vocab+4·H)·(B·S) + 20 MiB + 1024 B` / `(2·vocab+2·H)·(B·S) + 20 MiB + 2048 B` | **16 点**（6 个 H / 3 个 S / 2 个 B）+ 单卡 7/7 逐位 + 第三站点独立确认 |

**第 4 条的 vocab 扫描是最有价值的否定结果**：vocab=64640 时 `2·vocab` 项**整项消失**；
23 形状扫描显示 `c_V ∈ {0,2,4}` 是 aclnn **内核选择**、非光滑函数。
**不扫 vocab 就会交付一条静默失效的"律"。**

### 3.3 `K_CE` 重标定（`K_CE_PP1` 4→3、`K_CE_PP` 8→7、`K_CE_LEAN` 留 4）

依据 **9 份仓内 profiler CSV**（每份池高水位命中对应锚点 `real` 到 <0.5 MiB → 它们就是那些锚点的源账本）。
门关路径（融合 CE / 任何重算）**逐平面零误差**；两条非 lean 路径各**多 1 个平面**——
正是代码注释自认的"3 观测 + 1 保守"。`K_CE_LEAN` 无逐块台账，**未动**。

---

## 4. 当前真机 vs 仿真（刚实测）

`python tools/liveness_ab_validate.py --grad-mode chain2 --deltas`

| 跑 | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| a fused ON | 0.720 | 0.871 | 0.887 | 0.991 |
| **b unfused ON** | **0.708** | **0.703** | **0.706** | **0.730** |
| c fused **OFF** | 0.946 | 0.972 | 0.913 | 0.922 |
| e fused ON m8 | 0.736 | 0.872 | 0.887 | 0.991 |
| **f unfused ON m8** | **0.716** | **0.703** | **0.706** | **0.730** |
| g fused ON L4 | 0.784 | 1.038 | 1.012 | 1.020 |
| h unfused ON L4 | 0.979 | **0.747** | 1.017 | **0.719** |

`unfused − fused` delta（×1 工作集的量），bucket/real @L8：**0.697 / 0.619 / 0.619 / 0.477**。

**逐层驻留**：r0 2229.5/2235.1 = **0.9975**；r4 2208.6/2341.2 = **0.9434**；r128 2005.6/2116.1 = **0.9478**。

**⚠ 运维含义**：模型现在**系统性欠读**，约 23–26 条锚点 OOM-unsafe。
**在补上剩余项前，理论值不能直接当 OOM 判据，必须留余量。**

---

## 5. **未完成**：unfused 反向梯度工作集（推导完成，实现待做）

**这是当前最大的单项。** 推导已成文：`docs/unfused_bwd_working_set_derivation_2026-07-30.md`。

- **为何看不见**：`_forward_max_live` 只遍历 `op.inputs`/`op.output`，unfused fp32 复本群只在 `saves`
  → 对 `fml` 贡献**恰好 0**（实测 r4 层 `fml` 在 fused/unfused **都是 808.1 MiB**）。
- **可辩护 8192 MiB**：`grad_kv_bm`(4096) 与 `grad_kv_g` 累加器(4096) **必然共生**（VJP 强制）。
- **源码判不出 +4096**：步⑦逆 permute 是否独立缓冲，取决于 MS 的 permute-bprop 是否与
  `AccumulateGrad` 融合。**缺口"正好对得上"不构成理由**——那是拟合。
- **⚠ 实现陷阱**：`bwd_working_set = max(0, fml − bwd_scratch)` 与 `bwd_scratch` **互补**，
  加字节会被抵消，**必须实测净位移**。
- **自检项**：**fused 格子必须零位移**（fused 无 `kv_g`，走已实测的 730 MiB kernel scratch）。
- r0 滑窗分支（`dsv4_hybrid.py:406-411`）走同一函数、同样缺此项，按 `W0` 而非 `topk` 缩放，未覆盖。

---

## 6. 其余未决（按价值）

1. **12 个过读无解释**（`K_CE` 只解释 3 个；其中 5 个已**证明不在** loss 区平面桶里）。
2. **`extracted`（源码抽图）仍拒绝出数**：`IncompleteExtraction @indexer.py:219`，2568 节点→2056 op、**跳过 256**。
   它**拒绝给峰值**因为部分解析图的峰值是下界不是估计——**这条纪律不要放宽**。
3. **共享主干多声明 1 块 64 MiB**：模型 8/5 vs 账本 7/4（差值对得上、绝对值各多一）。需 tracker 输出 r128 前向窗口那 4 块的**诞生 op**。
4. `K_CE_LEAN = 4` 无逐块台账（一次 std MHA pp1 的 tracker 跑即可闭合）。
5. `vocab_parallel_ce` 既存潜在不一致（其 `bwd_scratch` 为 1 平面 → `//2` 前提失效）。
6. `nll`/`final_norm`/head 前向 workspace 已测但留 0；梯度累加 workspace（`2·vocab·H + 2560`，5 点）刻意未建模（非 head 特有）。

---

## 7. ⚠ 已被推翻的说法（**不要重新采纳**）

| 曾经的说法 | 判决 |
|---|---|
| "框架不释放 / 全重算只释放约 30% 激活" | **实测证伪**（`FrameworkGapWarning` 文案已改写） |
| "剩余残差是 kernel workspace" | **证伪**——workspace 1 tick，不进驻留；但**决定峰值** |
| "`q` 与 `q_hnorm_fp32` 是同一张量被数两次" | **错**——是两个真实 buffer，问题是 **2 次 dtype 判错** |
| "误差按层型反号（r0 2.97× 过读…）" | **无效**——`h−g` 不是干净单层隔离；真机 g/h 峰值落在**不同事件**（stage1/3，恰是主导差值的两个） |
| "`K_CE=8` 多算 4 个平面" | **错**——真实是 **+1 平面**，且现场配置**根本不走那行代码**（融合 CE → `loss_lids` 空） |
| "有个未解释的第三块 64 MiB" | **不存在**——第三块是 `query_index`，census 本来就有 |
| 逐层型总量 3225.2/3179.8/3101.3 | **漏了 `_dt` 抬升**，正确是 3749.2/3703.8/3625.3 |

---

## 8. 接手须知（操作）

**权威源码**：固定快照 `E:\97-codes\torch_parallel\mf-src-167\`，含 `mindformers/`（commit **`26354ff64`**）
与 `hyper_parallel/`（**`41495aa2`**），见其 `SNAPSHOT.md`。
**绝不用** `E:\97-codes\torch_parallel\mindformers`（不同 commit、内容不同 —— 本机 master 与真机不是一份代码）。

**真机**：`ssh 192.168.9.167` → `docker exec shb_dsv4`（MS2.10/CANN9.1，8×910B2）。
- `PYTHONPATH` **必须后缀** `:$PYTHONPATH`（裸赋值抹掉 CANN 路径 → 全 worker 死于 hccl）
- 信 `worker_<N>.log` 的 `[MEMPROBE]` 行，**别信** `peak_rank0.json`（scheduler 与 rank0 共享）
- **共享机器**：launch 前 `npu-smi info` 查空闲，有别的租户就等（`wait_and_run.sh`），**不抢占**
- 探针：`run_memprobe_events.py`（逐(微批,层,阶段) tracer，纯 monkeypatch，**未改共享源码**）

**验收门**：`tools/liveness_ab_validate.py --grad-mode chain2 --deltas`（8 组 × 4 列 + 不变量 + delta + 来源健康）。
真机表由 `REAL_SHA256` 指纹保护，**`REAL_*`/`CSV_*` 常量绝对不许改**。
核验：`git diff <base> HEAD -U0 -- . ':(exclude)docs' ':(exclude)scratchpad' | grep -nE '^[-+].*(REAL|CSV|MEASURED|sha256)'` → 应零命中。

**纪律**：测试期望必须改动时照 `docs/opdag_walker_core_2026-07-25.md` §6.6 先例——
**保留每一条不变量、只换举例形态**，逐条记理由。锚点重钉给 old → new + 一行理由。

**Windows**：跑 CLI 前 `export PYTHONIOENCODING=utf-8`（GBK 控制台，报告里有 `−`）。

**本轮 agent 死亡的四种模式**（勤提交 + 边写文档不是形式主义）：网络断连、TLS 拦截窗口
（企业代理 MITM，症状 `Self-signed certificate detected` / SSH `kex_exchange_identification`）、
会话额度上限、`529 Overloaded`。我曾两次手工抢救未提交的工作。

---

## 9. 关键文档索引

| 文档 | 内容 |
|---|---|
| `analysis/sim_vs_real_gap_decomposition_2026-07-28.md` | 差分拆解 + §5.5 真机逐事件判决 + §5.6 **逐桶差值表** |
| `analysis/handoff_2026-07-25_unfused-recompute-norelease-mechanism.md` | H1/H2 机制判决（§3 保留了被证伪的假设供追溯） |
| `docs/census_arbitration_2026-07-29.md` / `census_fix_mhc_rmsnorm_` / `census_fix_residual_carrier_` / `r4_indexer_census_` | census 逐条源判 |
| `docs/kernel_workspace_2026-07-29.md` / `head_workspace_` / `mhc_fwd_workspace_` / `head_loss_bwd_workspace_` | 四条 workspace 律 |
| `docs/k_ce_recalibration_2026-07-30.md` | `K_CE` 重标定（含前作归因订正） |
| `docs/unfused_bwd_working_set_derivation_2026-07-30.md` | **未完成项**的推导 |
| `docs/opdag_*`（10 份） | 源码抽图：覆盖度评估 → walker 核心 → 组件覆盖 → 字节 → 收口 |
| `docs/fused_mhc_branch_mismatch_2026-07-30.md` / `compress_ratios_mismatch_` | 两次"锚点对错模型" |
| 复现脚本 | `scratchpad/{decompose_gap,bucket_gap,per_layer_tensors,census_threeway,axis_probe_*}.py` |
