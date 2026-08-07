# Target Design v2 工程预测语义重置设计

> 日期：2026-08-07  
> 状态：设计已在对话中批准，待书面审阅  
> 仓库基线：`feat/unified-llm-modelspec` @ `6f8e9cf`，工作区含既有未提交修改  
> 正文基线：`docs/target-design-v2/src/index.template.html` 当前工作区快照（2026-08-07）  
> 修订对象：目标方案文档，不修改 `cost_eval/` 实现代码

## 1. 目标与结论

`target-design-v2` 的产品契约从“形式化 OOM 证据与描述性时间样本”重置为：

> 给定模型、并行、重算、优化器、调度和硬件配置，构造统一 IR，并输出允许存在模型误差的
> **每卡预测峰值显存**与**预测 step time**。结果用于工程分析和配置比较，不宣称数学证明。

当前正文把显存输出定义为单侧下界和三值 `oom_verdict`，把时间输出定义为不可比较的
`Sample[step_time]`。这些定义分别见当前正文 §10.2、§10.3、§14.3
（`src/index.template.html:2050`、`:2430`、`:3683`），与本目标冲突，必须整链替换。

本次不采用双轨 `SoundResult + EstimateResult`。产品只提供工程预测结果；证据来源、覆盖率和假设
作为诊断元数据存在，不形成第二套证明型输出。

## 2. 保留与删除的架构

### 2.1 保留

- 源码偏特化、纯配置算子语义和逐层改写 pass 构成的统一 IR 主链。
- 结构、放置、存储实例和执行目标使用独立 ID。
- 三个物理隔离的注册域，以及未知算子或缺少所需语义面时 fail-loud。
- 并行、重算、优化器和调度配置 DSL。
- Memory/Time 后端只读消费 `SchedIR` 的自包含投影，互不回写 IR。
- 每个预测量的来源、规则、覆盖范围、配置摘要和硬件摘要可追溯。

### 2.2 删除

- `Q[T]`、`Q3[Bool]`、`Sample[T]` 作为最终产品语义。
- `peak_interval`、`peak_allocated.lo/hi`、`oom_verdict`、`C_eff` 比较和
  `VERDICT-WITHDRAW`。
- `S_∞`、`absence`、`unprovable_ratio`、条件式下界以及围绕“假 OOM”的防逃逸链。
- “不回答装得下”“时间结果不可配置间比较”等旧能力边界。
- 为生成区间端点而做的敏感性扫描。配置 sweep 改为多次独立仿真，每个配置各自产生点预测。

## 3. 横切证据契约

数值使用点预测，证据只承担可追溯性：

```text
Estimate<T>:
  value: T
  evidence: [EvidenceTag]
  coverage: Coverage
  assumptions: [Assumption]
  model_digest: Digest

EvidenceTag:
  basis: source | measured | calibrated | modeled | assumed
  rule_id: String
  source_ref: SourceRef | None
  calibration_domain: Domain | None
```

`EvidenceTag` 沿数据流取并，用于解释预测依赖了哪些源码事实、标定值、复刻规则或假设。它不产生
上下界，也不改变结果类型。缺少后端必需语义面时阻断；已有语义面的证据等级较弱时仍可出预测值，
但必须在 `assumptions` 和 `coverage` 中暴露。

## 4. 内存仿真设计

### 4.1 输入契约

`Pass_sched` 为内存后端生成确定的 `MemoryEventView`。每个存储实例使用
`StorageInstanceId + epoch` 标识，事件至少包含：

```text
Allocate(storage, bytes, pool, event_id)
Bind(tensor, storage, offset, view_shape, event_id)
Use(tensor, op_id, event_id)
Free(storage, event_id)
```

- 只有 `Allocate` 增加已分配字节，只有 `Free` 减少已分配字节。
- alias、view 和原地结果通过 `Bind` 指向已有 `StorageInstanceId`，没有新的 `Allocate` 就不重复计数。
- 需要新物理存储的 tensor 必须有新的 `Allocate`；旧 storage 被复用时进入新 epoch。
- 参数、优化器状态等持久实例可不在 step 内 `Free`，其生命周期延续到 step 边界。
- 事件顺序由当前配置对应的调度计划确定；本后端不枚举其它交织，也不构造跨交织上下界。

alias、lifetime 或事件生成规则判断错误属于普通模型误差。方案不建立 `must_alias/may_alias` 关系求
所有可行划分的极小值，也不把当前 storage 划分宣称为事实证明。

### 4.2 峰值算法

```text
live_allocated(0) = persistent_allocated_at_step_start

Allocate(s, b): live_allocated += b
Free(s):        live_allocated -= bytes(s)

peak_allocated = max_event live_allocated(event)
```

分桶拆解与总量使用同一事件状态，必须逐事件守恒。输出为：

```text
MemoryEstimate:
  peak_allocated_bytes
  peak_event_id
  peak_breakdown
  timeline
  per_rank_or_rank_class
  coverage
  assumptions
  model_digest
```

本期不输出 `reserved`、allocator 外部碎片、余量或 OOM/fit verdict。以后若引入 allocator 模型，
它应形成独立的 `ReservedMemoryEstimate`，不能改变 `peak_allocated_bytes` 的定义。

### 4.3 内存结构门

- 同一 storage epoch 恰有一次 `Allocate`，至多一次 `Free`。
- 禁止 double allocate、double free、free-before-allocate 和负 live bytes。
- `Bind/Use` 必须发生在 storage 生命周期内，offset 与 view 范围不得越界。
- 非持久 storage 在计划要求的生命周期结束后必须释放；持久例外由类型给出，不由 waiver 给出。
- 每个事件的 breakdown 之和等于 `live_allocated`，报告峰值等于 timeline 最大值。

这些门验证仿真器内部执行是否自洽，不证明事件规则等同于真机。

## 5. 时间仿真设计

### 5.1 输入与调度规则

时间后端消费 OpDAG、执行目标、stream、资源请求、collective 关联和时长点估计。对每个 op：

```text
ready_time(op) = max(
  max(end_time(dep)),
  stream_available(op.stream),
  resource_available(op.resource_request),
  collective_ready(op.collective_correlation_id)
)

start_time(op) = ready_time(op)
end_time(op)   = start_time(op) + predicted_duration(op)
step_time      = max(end_time) - step_start_time
```

DES 必须表达：

- OpDAG 数据依赖和控制依赖；
- 同 stream 串行与跨 stream 显式依赖；
- compute、HBM、设备内拷贝和链路等资源容量；
- collective 的跨 rank rendezvous 与 correlation；
- PP/VPP、microbatch、重算和优化器阶段的执行关系。

时长可以来自实测、插值、族模型或 roofline；无值或非法值不能静默降为零。

### 5.2 输出与可比较性

```text
StepTimeEstimate:
  predicted_step_time
  op_timeline
  critical_path
  stream_overlap
  pipeline_bubble
  stage_breakdown
  resource_breakdown
  coverage
  assumptions
  model_digest
```

`predicted_step_time` 是每个配置的一次标量点预测。同一模型快照、硬件快照和 metric 定义下，不同
配置的结果可以比较，报告同时给出绝对差与相对差。比较结果仍是模型预测，不提升为真机事实。

### 5.3 时间结构门

- OpDAG 无环，所有依赖指向存在节点；生成的执行序满足全部依赖。
- 同一 stream 的执行区间不重叠，资源占用不超过声明容量。
- collective 的参与 rank、group、次序、字节和 correlation 完整匹配。
- 每个节点满足 `end = start + duration`，duration 有限且非负。
- makespan 等于所有 step 节点的最大结束时间减起点；相同输入摘要重复运行结果确定。

真机 op 序列、逐 op duration 和端到端 step time 是可选的标定与误差评估数据。没有包含
`start/end/stream/rank` 的调度级 trace 不阻断仿真，只意味着无法把调度误差归因到具体 overlap 或
stream 规则。端到端误差在有数据时必须报告，但不作为“预测值是否存在”的前置条件。

## 6. 报告与配置 sweep

首页只保留对工程使用直接有意义的字段：

- `MemoryEstimate` 与 `StepTimeEstimate` 的核心数值、构成和时间线入口；
- 模型、配置、环境、注册表和标定集摘要；
- op、storage、bytes 和 duration 的覆盖率；
- 未建模项、假设、越域项和 fallback 计数；
- 同口径配置比较表。

配置 sweep 是对每个配置完整执行一次构造和仿真：

```text
for config in configs:
    result[config] = simulate(config)
compare(result)
```

它不再生成区间端点或 `Sample` 极值集。敏感性分析使用预测值之间的差分，并携带双方的覆盖率与
假设变化，避免把“模型规则改变”误读成纯数值扰动。

## 7. 正文修订范围

主修改落在 `docs/target-design-v2/src/index.template.html`：

1. 标题升为 v4，§0 改写产品契约和前提。
2. §1 保留分层主干，更新全局输出不变量。
3. §2 全章从区间/provenance 代数改写为 `Estimate + EvidenceTag + Coverage`。
4. §3–§8 保留事实源、注册域和改写链，删除所有为区间端点、`oom_verdict` 或防逃逸服务的支线。
5. §9.2 改成确定的 memory lifecycle 投影与 OpDAG 调度投影。
6. §10 按本设计重写为 MemoryEstimate、StepTimeEstimate、报告和配置比较。
7. §11 保留未知算子和缺语义面阻断，删除围绕 `undetermined` 的补救体系。
8. §12 用内存/时间结构门和可选精度评估替换证明型门。
9. §13 保留规模预算，配置扫描改成重复仿真。
10. §14 重写诚实边界：结果可能因结构、alias、lifetime、stream、资源和时长模型错误而偏高或偏低。

同步修改：

- `docs/target-design-v2/tools/verify_gates.py`：更新门表计数和新增的全文契约断言。
- `docs/target-design-v2/HANDOFF.md`：只更新当前目标、输出和工作流摘要，不改历史对抗记录。
- 运行构建脚本生成 `index.html` 与 `artifact.html`；不手改生成物。
- `ADVERSARIAL-RECORD.md`、`DEFECT-LEDGER.md`、各轮 refute 文件作为历史记录保留，继续以正文为准。

## 8. 验收

### 8.1 正向契约

- 正文明确说产品是允许误差的工程预测仿真器。
- 内存公式只按显式 `Allocate/Free` 生命周期求 `peak_allocated`。
- 时间公式从 OpDAG、stream、资源与 collective 关系求标量 `predicted_step_time`。
- 正文明确允许同口径配置比较，并标注它是预测比较。
- alias/view 不产生新 allocation 的规则、持久 storage 例外和 lifecycle 结构门均有唯一落点。

### 8.2 负向契约

正文规范性内容不得残留下列旧产品语义：

```text
oom_verdict
peak_interval
peak_allocated.lo
peak_allocated.hi
SoundResult
Sample[step_time]
VERDICT-WITHDRAW
可证下界
配置间不可比较
本工具不回答哪个配置更快
```

历史文件允许保留这些词，核验范围只覆盖唯一正文源和生成页面的正文。

### 8.3 构建与一致性

- `python tools/build_doc.py` 成功。
- `python tools/verify_gates.py` 成功。
- HTML 受保护块外标签配平，所有 `href="#..."` 均有目标锚点。
- `index.html`、`artifact.html` 与模板的正文一致。
- 人工复核 §0、§2、§9、§10、§12、§14，确认没有同时存在新旧两套输出契约。

## 9. 非目标

- 不修改或迁移现有实现代码。
- 不承诺具体误差阈值；阈值需要真实配置族的数据后另行制定。
- 不增加形式化 OOM 证明、容量 verdict 或 reserved/碎片估计。
- 不要求先采集调度级 trace 才允许输出 step time。
- 不借本次语义重置扩大模型族或算子覆盖范围。
