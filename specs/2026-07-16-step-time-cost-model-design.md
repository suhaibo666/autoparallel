# Step Time Cost Model（P1 时间模型）— 设计文档

> 创建: 2026-07-16 · 状态: 设计定稿（用户逐节过审）→ 实现计划 → 实现
> 父设计: [[2026-06-23-pynative-cost-evaluator-design]]（§9 单步时间草图、M8/M9 扩展点）
> 依赖事实源: [[2026-07-08-source-grounded-opgraph-design]]（opdag 提取器）
> 对标方法论: 内存侧真机锚点阶梯（[[2026-07-07-memory-model-reference]] §14，DSv3 4L=1.000）

## 0. 一句话目标

给定 专家配置（5D 并行 + recompute）+ opdag（源码抽取的 op-DAG）+ OpTimeLibrary（op 时间经验库），
**离线**估算目标 NPU 集群上的 **单步时间 T_step + 瓶颈拆解（host/算力/带宽/通信/bubble）+ MFU/HFU**。
这是评估器双指标（显存+时间）的第二半，落进实现架构预留的 M8/M9 扩展点。

## 1. 已定决策记录（设计对话 2026-07-16，逐项过审）

| # | 决策 | 内容 |
|---|------|------|
| D1 | 精度定位 | **标定后绝对预测**：单层 <15%，端到端建真机锚点矩阵（对标内存侧 1.000 阶梯）；无标定退化为 roofline 相对排序 |
| D2 | host 建模 | **双流仿真一等公民**：host 发射流与 device 计算流分开建；host-bound 是仿真涌现结果，不可静态判定 |
| D3 | v1 范围 | 5D 全轴 + 基础 overlap（FSDP 预取/DP grad sync/EP all-to-all/CP/PP 1F1B+VPP）+ recompute 时间惩罚；**swap、PP 高级调度（dxdw/zero-bubble/overlap_p2p）、mc2、cpu_offload → v1.5** |
| D4 | 总体方案 | **分层离散事件仿真**（op 级双流 → pipeline 级调度仿真）；闭式公式降级为退化交叉验证 |
| D5 | 解耦硬约束 | **时间仿真器与内存仿真器不耦合、独立演进**（用户强调；§2 五条契约 + import-lint 测试强制） |
| D6 | IR 唯一来源 | **全量走 opdag，无 LayerSpec 兜底**（用户裁决：兜底=两个事实源，重新引入手写猜测问题）；唯一退化是 fail-loud 拒绝出数 |
| D7 | 经验库 | 时间数值主来源是**持续积累的 OpTimeLibrary**（实测点+每 op 类经验模型）；理论 FLOPs 退居无数据下界与 roofline 分类依据（用户提出） |
| D8 | 可视化 | 时间侧四面板长进现有 serve_explorer（同一配置区驱动内存+时间 what-if），面板间无共享内部状态（用户提出） |

## 2. 总体架构与解耦契约

### 2.1 分层数据流

```
opdag JSON(源码抽取)      ParallelConfig 等         HardwareSpec + OpTimeLibrary
      │                        │                          │
      ▼                        ▼                          │
 [timed_ir]  op序列 IR 化：并行代入(local shape) + 通信注入 + 流标注    │
      │      （TimedOpSeq：中立数据结构，时间仿真唯一入口）              │
      ▼                                                   ▼
 [op_cost]   每 op → OpCost{t_host, t_dev 或 t_comm, 算术强度, bound, provenance}   ← M8
      │
      ▼
 [segment_sim] 段内多流 DES：H(host发射)/D(device计算)/C_tp/C_ep/C_dp
      │        → SegmentTime{时长 + host空洞/exposed_comm 归因}         ← M9-L1
      ▼
 [pipeline_sim] (stage×microbatch[×chunk]) 全局 DES：调度序 + p2p 依赖
      │         + grad sync 尾 + opt step → T_step、实际 bubble、关键路径 ← M9-L2
      ▼
 [report]    StepTimeReport：T_step + MFU/HFU + 瓶颈拆解，并入 M7 门面
```

模块落位：**`cost_eval/timesim/` 独立子包**（`timed_ir.py / op_cost.py / segment_sim.py /
pipeline_sim.py / library.py`）。M8/M9 落进 [[2026-06-29-evaluator-implementation-design]] §8
预留的扩展点，M4/M6 产物不动。

### 2.2 解耦契约（D5，五条硬约束）

1. **双向 import 禁令**：`timesim/**` 不得 import `mem_timeline / structure_mem / static_mem`，
   反向亦然。**用 import-graph lint 单测强制**，与 12 内存锚点一起进回归门——解耦是测试守住的边界，
   不是口头约定。
2. **共享面只有三样中立资产**：`specs.py`（配置数据类）、`schedule.py`（新模块，见下）、上游 IR
   （opdag JSON）。**`schedule.py` = 把 1F1B/VPP 纯调度代数从 `mem_timeline` 搬家**（`build_1f1b` /
   `interleaved_warmup` / `get_schedule_table` / `interleaved_virtual_order`——逐行 port Megatron
   `schedules.py` 且过真机 VPP 验证的部分），无任何内存/时间语义。搬家是纯重构：`mem_timeline`
   保留 re-export，验收 = 12 锚点逐字节不破。
3. **各自独立 walk**：时间仿真自建事件推进，不复用内存 walk 的桶/事件对象。内存侧改桶、加事件，
   时间侧零感知；反之亦然。
4. **标定命名空间分离**：时间标定（经验库/host 单价/α-β）不回流内存；内存的 `framework_reserve` /
   margin 不进时间。
5. **门面组合**：`Evaluator` 各自可单跑（memory / time），任一失败不影响另一个的结果可用性。

## 3. TimedOpSeq IR（timesim 唯一输入）

### 3.1 opdag 现状事实（设计前核实）

- opdag（`cost_eval/opdag/schema.py`）= 纯静态数据流 DAG：`OpDAG{cell, nodes, edges, baseline,
  scalar_binds, dims_ctx}`，`OpNode{id, op, src(file:line), module, ins/out("名:符号shape:dtype"),
  attrs}`。**无 timeline、无 phase（仅 fwd）、无通信节点**。
- **执行顺序隐式可用**：`construct_walker` 按 construct() 语句序发射节点 → node id 序 ≈ 程序序 =
  host 发射序；`edges` = 数据依赖 = DES 跨流依赖来源。
- **覆盖缺口**（`crosscheck.py:38`）：「GQA/dense/embedding/lm_head/mtp 层无 opdag 提取源」；
  **通信原语在整个 `opdag/` 中零涉及**（grep AllReduce/all_gather/reduce_scatter 无命中）——
  通信提取是纯新增能力，工作量如实计入 v1。

### 3.2 数据结构

```python
@dataclass(frozen=True)
class TimedOp:
    op_id: str                  # 回指 opdag 节点 id → mindformers file:line（源忠实链完整）
    op_type: str                # MatMul/GroupedMatMul/FlashAttention/Norm/Activation/
                                #   Elementwise/Cast/View/CommOp（复用 opdag schema 词表）
    phase: str                  # fwd | bwd | recomp
    in_shapes: tuple; out_shape: tuple   # 已代入 local shape（DimTable + tp/cp/ep 度数）
    dtype: str
    stream: str                 # device | comm_tp | comm_ep | comm_dp | comm_pp | host_only
    deps: tuple[str, ...]       # 跨流依赖（同流按发射序隐式 FIFO）
    comm: CommSpec | None       # ctype / volume_bytes / group_axis / group_size

@dataclass(frozen=True)
class TimedSegment:             # pipeline_sim 消费的粒度
    seg_id: str                 # "stage2.mb0.fwd" / "layer_3.bwd" / "loss.bwd" 等
    ops: tuple[TimedOp, ...]
```

### 3.3 producer 五步管线（opdag JSON → TimedOpSeq）

**a. 覆盖扩展（v1 工作项，D6 的代价兑现）**：extractor 从 TransformerLayer 级扩到 **GPTModel 级
walk**——`embedding(gather) → layer 栈 → final norm → lm_head(matmul, vocab-parallel 语义) → loss`。
loss 段走真机已证实路径（`log_softmax + nll + scatter_add`，内存峰值算子 `ScatterAddExt` 即在此段
反向）。MTP 视排期 v1.5。提取不出 → **fail-loud 报缺口位置，拒绝出数**（07-08 设计硬约束 5 延续）。

**b. 并行代入**：符号 shape × DimTable → local shape。复用 `opdag/consumer.py` 已有符号→维表机制。

**c. 通信注入**（两类来源，均源忠实）：
- **源码内显式通信（新增提取能力）**：TP/SP 的 all-reduce / reduce-scatter / all-gather 在
  `ColumnParallelLinear/RowParallelLinear` construct 里是显式调用——extractor 识别其惯用法发射
  CommOp 节点；识别不出按模块语义注入并 fail-loud 记录。
- **框架层通信**（不在 layer construct 源码里）：FSDP per-layer all-gather（fwd 预取 + bwd 重
  gather，随 `reshard_after_forward`）、grad reduce-scatter、DP grad all-reduce——按 FSDP/DDP
  语义注入为**段边界 CommOp**，挂 `comm_dp` 流。MoE dispatcher 是 opaque 段 → 按
  `moe_token_dispatcher_type` 注入 dispatch/combine 两个 all-to-all，量按 **balanced 路由**
  （v1 假设，继承内存侧口径）。CP 按 `context_parallel_method` 注入 ring P2P / all-to-all 到
  FlashAttention 两侧。
**d. bwd 序列展开**：fwd DAG × **per-op-type bwd 规则库** → 逆拓扑序 bwd op 序列。`bprop_rules`
（save-set 用）的姊妹件——同按 op 类型、~15 条教科书事实：MatMul → dX+dW 两个 matmul；
GroupedMatMul 同理；FA → 一个 FA-grad kernel；Norm/Activation/Cast → 对应 grad kernel（带宽类）；
View → host_only；**CommOp → 反向对偶通信**（all-gather↔reduce-scatter、all-to-all↔all-to-all、
all-reduce↔all-reduce）——SP/TP 反向通信自动涌现，不手拍。bwd op 数目由展开决定，host 单价用
bwd 相位标定（autodiff engine 发射，与 fwd Python 发射不同价）。

**e. recompute 展开**：按 RecomputeSpec 把该层 fwd 序列（或 select 子段）以 `phase=recomp` 前插进
bwd 段；`recompute_comm`/`exclude_op` 语义 = 重放序列中剔除/保留对应 CommOp。语义继承内存侧已核实
口径，**独立实现在 timesim**（契约 3）。

### 3.4 IR 层自带验收

- 不变量：Σ MatMul FLOPs(fwd) ≈ 6ND±MoE/attn 修正——进 op_cost 前纯 IR 层可测。
- **时间侧自己的 census 对账**：TimedOpSeq op 名册 vs 真机 profiler kernel 名册
  （`kernel_details.csv`）逐类别 delta——与内存侧 crosscheck 同哲学、不同实现、互不依赖。

## 4. op_cost（M8）与 OpTimeLibrary（经验库，D7）

### 4.1 OpCost

```python
@dataclass(frozen=True)
class OpCost:
    t_host_us: float          # host 发射成本（所有 op 都有，含 View/Cast）
    t_dev_us: float           # device 计算成本（View=0）
    t_comm_us: float          # 通信成本（CommOp）
    flops: int; bytes_rw: int
    arith_intensity: float    # FLOPs / bytes_rw（通信 op 为 0）
    ridge: float              # 机器平衡点 = peak_FLOPS(dtype)/HBM_BW，判据留痕
    bound: str                # compute | memory | comm | host —— 静态 roofline 分类（AI vs ridge）；
                              # "host"=View/host_only（无 device 分量、分类无意义），**≠**下文 D2 讲的
                              # segment_sim 涌现态 host-bound（T1 实现补，Task 6 review 同步）
    host_dominated: bool      # t_host > t_dev（该 op 发射比执行贵：View/Cast/小elementwise 典型）
    eta_key: str
    provenance: str           # hit | model(置信度) | theory —— 该数字的可信来源
```

**分类口径的关键区分**（D2 推论）：`bound`/`host_dominated` 是 op 内禀静态属性；**真正的
host-bound（device 空洞）不是 op 属性**——host 流是流水的，同一 op 序列在不同上下文里空洞可有可无，
只能由 segment_sim 仿真涌现（§5.3 三态归因）。

**FLOPs/bytes 公式按 op_type 内置**：MatMul `2MKN`；GroupedMatMul 按组求和（balanced 口径）；
FA fwd `4·B·N·S²·D·causal系数`（bwd ≈2.5×）；Norm/Activation/Elementwise/Cast 为 bytes-only；View 计 0。
理论 FLOPs 的角色（D7）：① 无数据时保守下界；② roofline compute/memory 分类依据。**时间数值主来源
是经验库**。

**host 单价**：`t_host(op_type, phase)`——op 类型 × 相位常数单价，与 shape 无关；fwd（Python/pyboost
发射路径）与 bwd（autodiff engine 发射路径）两套分开标。

**通信成本**：`t_comm = α(ctype, topo) + volume · hops(ctype, n) / BW_link`；ring all-reduce
`2(n−1)/n`、all-gather/reduce-scatter `(n−1)/n`、all-to-all `(n−1)/n`（对分带宽口径）、P2P 直传。
`topo` 由 group 在 mesh 上落位推断（节点内 HCCS / 跨机 RoCE / 分层混合）。

### 4.2 OpTimeLibrary：持久积累的经验库（D7，取代一次性 CalibrationTable）

定位：**跨 run 积累、版本化的实测 kernel 时间资产**（JSON 工件族，按 `device × ms_version` 分库），
与 opdag JSON 同哲学——再生数据工件，评估器只读。每跑一次真机，库变厚一分。

```
OpTimeLibrary (device=910B, ms=2.9):
  samples:  {(op_type, shape, dtype) → {t_fwd_us, t_bwd_us, n_samples, var, source_run}}
  models:   {op_type → 拟合模型（从 samples 定期重拟合，缓存）}
  comm:     {(ctype, topo, group_size) → {alpha, beta}}
  host:     {(op_type, phase) → unit_us}
```

**每 op 类经验模型**（对"纯理论 FLOPs"的实质替代；NPU kernel 时间随 shape 是阶跃非平滑的）：

| op 类 | 模型形态 | 理由 |
|---|---|---|
| GEMM/GroupedGEMM | `(M,K,N)` 网格有效 FLOPS 曲面插值 + 对齐阶跃修正（16/32 档） | cube 利用率随 shape 阶跃，单 η 差 2–3× |
| FlashAttention | `(B,N,S,D,causal)` 参数化拟合（S² 主项系数实测） | kernel 实现系数理论算不准 |
| 带宽类 | `t = launch + bytes/eff_BW`，eff_BW 分段拟合 | 小尺寸 launch 主导，大尺寸逼近带宽 |
| 通信 | α-β per (ctype, topo, group)，多 size 点拟合 | — |

**三级退化**（provenance 对应三态）：
1. **精确命中**：一个训练 step 里不同 shape 的 op 很少（几十种，跨层/微批重复）→ 锚点配置命中率天然高；
2. **经验模型内插**：what-if 改 tp/ep → local shape 变 → 落进模型插值，**置信度按到实测支撑点距离**；
3. **roofline × 默认 η**：库无数据的 op 类才落到理论下界（matmul 0.6–0.85 / FA 0.4–0.5 /
   带宽类按可达带宽），退化为原 P1 纯解析，相对排序仍有效。

**采集双路**：
- 被动：每次锚点/验证 run 的 profiler `kernel_details.csv` 自动入库（append + 去重 + 方差累积）；
- 主动：**op-level microbench runner**（116，独立脚本）对关键 op 类扫 shape 网格（GEMM 在模型
  shape 邻域扫 M/K/N、FA 扫 S、带宽类扫 bytes）——"缩层 profiling"概念的资产化。
- host 单价两法互证：① tiny-shape op 循环 microbench（device→0，wall≈host）；② 段级最小二乘
  （段 wall − device busy = host 空洞，对各 op 类单价回归）。

**经验库自身验证**：leave-one-out 交叉验证（抽一实测点，用余点模型预测之）→ per op class 插值误差
分布 = provenance 置信度的量化来源（验证阶梯 L2 扩展档）。

**结构与常数分离铁律延续**：换硬件/换 MindSpore 版本 = 换库 JSON，代码零改动。

### 4.3 可信度可见

provenance 逐 op 向上聚合：StepTimeReport 报告"本预测 X% 时间来自实测命中、Y% 经验模型内插、
Z% 理论默认"——绝对预测的可信度逐数字可追溯。

## 5. segment_sim：段内多流离散事件仿真（M9-L1）

### 5.1 流定义

| 流 | 内容 | 推进语义 |
|---|---|---|
| **H**（host 发射） | 全部 TimedOp，按语句序（fwd=opdag 程序序，bwd=展开序） | 串行 `h_end_i = h_start_i + t_host_i`；**默认不等 device**（PyNative 异步下发），仅显式同步点停 |
| **D**（device 计算） | 计算 op（View/host_only 不占用） | FIFO：`d_start = max(已被H发射, D前op结束, 跨流deps完成)` |
| **C_tp/C_ep/C_dp/C_pp**（通信按 group 轴分道） | CommOp 按 `stream` 入道 | 同道 FIFO + 跨流 deps；不同轴通信域可并发（HCCL 多流一阶近似） |

跨流依赖只来自 `TimedOp.deps`（= opdag 数据边）；同流顺序 FIFO 隐含。多通信流带宽争抢 v1 不建（§7 边界）。

### 5.2 仿真单元：连续 pass，不是孤立的层

**PyNative 的 host 不在层边界停**——device 跑第 i 层时 host 已在发射第 i+1 层。L1 仿真单元是
**一个 (stage, phase, microbatch[, chunk]) 的完整连续 pass**，host 流一条贯到底；"层"只是报告标记。
按层孤立仿真再求和会掐断 host run-ahead，系统性高估。pass 边界（p2p recv、microbatch 切换）天然是
同步点，跨 pass run-ahead 截断——v1 有界近似，如实记录。

### 5.3 归因：三态积分（涌现，非拍定）

对 D 流在 `[0, makespan]` 逐时刻积分，每刻恰属一态（判据次序：先查发射、再查依赖，无二义）：

```
D 在忙：执行 op x            → 按 x.bound 记入 t_compute 或 t_membound
D 空闲：下一 op 未被 H 发射   → t_host_gap（真 host-bound 空洞）
D 空闲：已发射但 comm dep 未完 → t_exposed_comm[该通信轴]
```

`Σ 三态 = makespan`，守恒可测（L0）。

### 5.4 输出

```python
@dataclass(frozen=True)
class SegmentTime:
    seg_id: str
    duration_us: float                # makespan
    t_compute: float; t_membound: float   # device 忙，按 op 静态类劈分
    t_host_gap: float
    t_exposed_comm: dict[str, float]  # 按通信轴
    host_len_us: float                # H 流总长；host_len≈duration ⇒ 整段 host-bound
    per_layer: dict                   # 层标记聚合（报告用）
    top_contributors: list[str]       # 占时 top-N op_id，回指 file:line
```

### 5.5 结构性 overlap 从哪来（位置，不是系数）

- **FSDP 预取**：第 i+1 层 param all-gather 注入第 i 层序列头部（`comm_dp` 道）→ 重叠自然发生；
  `reshard_after_forward` 决定 bwd 是否再注入。
- **grad reduce-scatter**：注入每层 bwd 序列尾部 → 与后续层 bwd 重叠，压不掉部分自动 exposed。
- **recompute**：`phase=recomp` 前缀直接加长 bwd pass 两条流，惩罚=序列长度本身，无需公式。
- **同步点 v1 白名单**：pass 边界、optimizer step、已知强制同步 op（loss 标量化）。balanced 假设下
  MoE 无数据依赖同步；router 若实际有 `.item()` 类同步 → v1.5 修正。

## 6. pipeline_sim：全局调度仿真（M9-L2）

### 6.1 事件与调度来源

事件 = `(stage, microbatch, phase[, chunk])`，时长取 L1 SegmentTime。每 stage 事件序取自
`schedule.py`（§2.2 契约 2 搬家产物），时间侧不重新发明调度。段时长按**稳态口径**每
`(stage, phase, chunk)` 算一次：stage 层构成不同（首 stage embedding、末 stage head+loss、MoE 落位）
→ 时长天然不同，**非均匀 stage 是自然输入，无需特判**。

### 6.2 全局 DES 规则

- stage 资源：每 stage 同刻只执行一个事件，按调度序取下一个；
- 跨 stage 依赖：`F(s,i)` 需 `F(s−1,i)` 完成 + p2p；`B(s,i)` 需 `B(s+1,i)` 完成 + p2p；VPP 按 chunk 同构；
- p2p：`t = α + act_bytes/BW`，v1 作**依赖边时延**（不占 device 资源）；`overlap_p2p/dxdw` 等 v1.5
  换调度生成器即可（架构预留）；
- `T_step` = 全 stage makespan；**bubble = 每 stage device 时间线空闲积分**——仿真结果非公式。
  闭式 `(m+PP−1)/m` 降级为退化交叉验证（L0③）。

### 6.3 步收尾

1. **grad sync 尾**：per-layer grad 通信已在 L1 注入（可遮盖部分自动遮盖）；pipeline_sim 加 barrier
   ——**optimizer step 等全部 `comm_dp` 道排空**，遮不住的尾巴自动暴露成关键路径。
2. **optimizer step**：v1 = device elementwise 带宽类（`(param+grad+opt_state) 分片字节 / HBM_BW /
   η_opt`）+ host 发射；cpu_offload 随 v1.5。
3. **per-step 固定开销**（dataloader、step 边界框架成本）：锚点 `实测 − 仿真` 反解的标定常数，
   **单列、不混进 η**（口径同内存侧 framework_reserve 哲学，独立命名空间）。

### 6.4 输出

```python
StepTimeReport:
  t_step_us
  per_stage: {stage → busy / bubble / exposed_comm{轴} / host_gap}
  bubble_fraction: 仿真实际值（附闭式参考值；两者差=非均匀/调度效应，本身即洞察）
  critical_path: [(stage, mb, phase)...]
  mfu / hfu             # MFU=有效FLOPs(不含recompute)；HFU=含（Megatron 惯例分开报）
  provenance_mix        # 实测命中/模型内插/理论默认 占比（§4.3）
  bottleneck_ranking    # host / comm轴 / bubble / compute 排序
```

recompute↑ 时 MFU↓/HFU↑ 剪刀差是性质测试点（L0④）。

### 6.5 快速通道（P2 搜索器）

L1 结果按 `(stage, phase, chunk)` 缓存；同模型扫并行配置只重算受影响段；单次评估目标毫秒级。

## 7. 验证阶梯与诚实边界

### 7.1 验证阶梯（内存侧方法论的时间版）

| 级 | 内容 | 判据 |
|---|---|---|
| **L0 纯软件不变量** | ① 三态归因守恒 `Σ=makespan`；② FLOPs≈6ND±修正；③ 退化还原：host=0+无通信+均匀 stage → 闭式 `(m+PP−1)/m·Σt`；全度=1+无重算 → 单卡串行 roofline 和；④ 性质：m↑→bubble%↓、recompute↑→t_step↑且 MFU↓/HFU↑、tp↑→GEMM t↓通信↑ | 单测常绿 |
| **L1 文献互证** | dense 同配置对 **Calculon** 独立推导；公开 MFU 数量级 sanity | 吻合即可信 |
| **L2 逐 kernel oracle** | `kernel_details.csv` vs op_cost：命中项恒等自检；模型内插项 LOO 误差分布 per op class | per-class 误差报告 |
| **L3 段级** | profiler step trace vs SegmentTime：fwd/bwd pass 时长、host 空洞 vs device 利用率 gap | 单层 <15%（标定后） |
| **L4 端到端锚点矩阵** | 与内存锚点**同配置族、同 run 双收**（内存+时间证据一次采齐；验证脚本各自独立）：DSv3 4L/8L、cp2、pp2（bubble 实测）、select/full 重算、ep 变体 | 标定后 t_step ≤15%；**相对排序 100% 正确** |

### 7.2 诚实边界（7 条）

1. 跨机 α-β 未标定（116 单机 8 卡；厂商规格代入，report 标记为未标定参数）；
2. 多通信流带宽争抢不建（多流并发理想化）；
3. MoE balanced 路由（继承内存口径；实测直方图 v1.5 可选输入）；
4. host run-ahead 跨 pass 截断（有界近似）；
5. 同步点白名单外的强制同步不建（偏乐观）；
6. **profiler 观测污染**：采集拖慢 host（内存侧已见 ~300MB 同类效应）——标定 run 用轻量采集 + 污染扣除；
7. 降频/cache 效应统一进经验模型/η，不单列。

## 8. 可视化：explorer 时间侧（D8）

沿用现有交互范式（改配置→防抖重算），**同一配置区同时驱动内存+时间**——P1.5 what-if 的交付形态。
新增四面板：

1. **多流泳道图（段内 Gantt，核心）**：H/D/C_tp/C_ep/C_dp 泳道，op 色块（色=bound 类），
   **host 空洞红、exposed_comm 橙**；点 op → 右栏详情（file:line、OpCost、算术强度落点、provenance）。
   即"仿真版 profiler timeline"，可与真机 MindSpore Profiler 时间线并排肉眼比对（L3 的可视化形态）。
2. **Pipeline 调度图**：stage 行 × 时间轴，F/B/recomp 色块 per microbatch，bubble 即空白，VPP chunk
   分色；点色块下钻到该段泳道图。
3. **Roofline 散点图**：每 op 按（算术强度, 达成 FLOPS）落点 + ridge 线，色=bound、形=provenance。
4. **T_step 瀑布图 + KPI**：compute/membound/host_gap/exposed(按轴)/bubble/收尾 瀑布分解 +
   top-N 耗时 op 表（回指源码）；顶部 KPI 行加 **T_step / MFU / bubble%**。

**解耦在可视化层延伸**：时间面板只消费 timesim JSON（`/eval_time` 端点），内存面板走现有端点；
serve_explorer 只是门面路由，两侧无共享内部状态；timesim 不可用/无标定时时间面板独立降级
（provenance 警示条），不影响内存面板。

## 9. 分阶段交付

- **T0 地基**：`schedule.py` 搬家（12 内存锚点逐字节回归门）+ `timesim/` 包骨架 + **import-lint
  解耦测试**；opdag 覆盖扩展（GPTModel 级 walk + **通信提取新增能力**）；TimedOpSeq producer +
  IR 层不变量。
- **T1 仿真**：op_cost（默认 η）+ segment_sim + pipeline_sim + StepTimeReport；L0/L1 全绿；
  explorer 时间面板上线（标"未标定"口径）。
- **T2 经验库+真机**：microbench/收割 runner（116）+ OpTimeLibrary 工件 + L2–L4 锚点矩阵 +
  分项误差报告；explorer 显示 provenance。
- **v1.5+**：swap/offload 时间、PP 高级调度（dxdw/zero-bubble/overlap_p2p）、mc2、CPU 优化器、
  实测路由分布输入、MTP 段、router 同步修正。

## 10. 开放问题 / 风险

- **R1（最大）**：`Column/RowParallelLinear` construct 内通信调用惯用法的静态提取深度——通信提取是
  全新能力（§3.1 核实零基础）。缓解：先 DSv3 一条路径打通；识别不出按模块语义注入 + fail-loud。
- **R2**：host 单价的稳定性——同 op 类不同调用点的发射成本方差；两法互证（microbench vs 段级回归）
  兜底，方差过大时按调用点细分键。
- **R3**：MindSpore profiler 的 host 侧时间线（框架层采集）可用性/口径——影响 host 标定路①；
  路②（段级回归）不依赖之。
- **R4**：GroupedMatMul 的 permute/pad 中间 kernel 是否在 profiler 中独立可见（同内存侧 R3 教训）
  ——不可见则该段按融合 kernel 整体入库。
- **R5**：`schedule.py` 搬家虽是纯重构，仍触及内存侧文件——须在独立 commit、12 锚点回归门下做，
  且先于一切 timesim 功能。

## 关联

- 主架构: [[2026-06-23-pynative-cost-evaluator-design]]（§7 roofline、§9 时间草图、§12 P1 定义）
- 实现架构: [[2026-06-29-evaluator-implementation-design]]（M8/M9 扩展点约定）
- opdag: [[2026-07-08-source-grounded-opgraph-design]]（提取器三阶段、fail-loud 铁律）
- 内存参考: [[2026-07-07-memory-model-reference]]（锚点矩阵方法论）
- 真机通路: `.claude/skills/real-machine-memory-sim/`（116 连接/环境/采集，时间 runner 独立新增）
