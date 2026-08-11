# 交接文档 · PyNative 多维并行成本评估目标方案 v4.2

> 最后更新：2026-08-10
>
> 当前任务性质：目标方案设计与审查，不是 `cost_eval` 实现迁移
>
> 唯一权威正文：`src/index.template.html`
>
> 历史 spec/review 只记录形成过程；冲突时一律以权威正文为准

## 0. 三十秒版本

这是一个纯代码驱动的多维混合并行训练内存与 step-time 工程预测仿真器。

权威模块架构只维护一份：见权威模板 [`src/index.template.html`](src/index.template.html) 中
`data-diagram-id="layered-module-architecture"` 的 Mermaid 源，以及由它生成的 `index.html`/`artifact.html`
同名静态 SVG。HANDOFF 不再复制一份可能漂移的 ASCII 架构图；下文 `text` 代码块仅用于 schema、公式或
伪代码，不是第二套架构/流程图。artifact 的规范装配顺序见正文 §13.2，不由模块架构图重复定义。
图 1 实线统一从 dependency provider 指向 runtime caller，并机械反转 §1.2 的 allowed-dependency pair；
虚线只表示 immutable DTO/data lineage，不授予运行时调用权。

- 内存只按确定的 logical kernel order 重放 `Allocate/Bind/Use/Free`，不读取 duration 或完成时刻。
- 时间使用无资源竞争的 global progress DAG；只建模依赖、stream 次序、P2P/collective rendezvous。
- compute duration 来自用户预采集、冻结且 MeasurementKey 精确匹配的唯一记录；communication duration
  来自唯一选中的版本化理论公式与 route。
- logical-rank→device 映射来自显式 ExecutionDeployment；stream 使用 device-qualified PhysicalStreamId 和
  唯一 StreamBinding，不由实现临时选择。
- kernel variant 在 core 中由 hardware-free selector 与 HardwareBindingPolicy 唯一解析；memory workspace
  和 time MeasurementKey 共用同一 KernelVariantBinding，但缺 duration record 仍只阻断 time。
- RuntimeEventPlan 已物化规范 `LogicalRankContextSet`、`ResolvedEventSemantic`（含唯一 EffectSummary）、Tensor/Storage instance binding、memory key 与
  structural report facts；projection 不按 local ID 暗读 CodeIR/registry 进程对象。
- blocker 使用 `BlockerInstanceId + BlockerCode + BlockerRecord` 统一表示，Diagnostic 只能规范派生；
  Ready view 内嵌规范 `EstimateContext`；source/runtime/core/projection blocker 必须按 backend scope 传播，
  affected blocker 非空时不可能 Ready。
- 待预测运行的 raw trace 不是生产输入；独立 TraceFixture 只用于离线/CI conformance。

### 0.1 十张权威 Mermaid 图索引

图的唯一有效源均位于 `src/index.template.html` 的 `mermaid-source`；`tools/build_doc.py` 在构建期将其渲染为
静态 SVG。图负责解释关系，邻接 schema、公式与伪代码仍是规范真值。

| diagram ID | 权威位置与责任 |
|---|---|
| `layered-module-architecture` | §1.1/§1.2；L0–L4 十模块分层、provider→caller 实线依赖、DTO 虚线与离线 conformance sidecar |
| `facts-and-digests` | §2.4/§3.1；输入事实、模型/运行时/后端摘要和验证旁路 |
| `per-rank-codeir` | §4.2/§4.3；逐 logical rank 求值、obligation disposition 与 CodeIR 汇总 |
| `value-storage-identity` | §5.2/§5.3；prototype、event occurrence、Tensor/Storage instance、binding 与 initial state |
| `semantic-effect-closure` | §6.3–§6.5；语义注册/校验、effect 闭包与 blocker 状态机 |
| `runtime-event-expansion` | §7/§8.1–§8.3；forward/backward/recompute/optimizer 与 P2P/collective intent 展开 |
| `plan-projection-module-architecture` | §9.1；上游 DTO 虚线、plan-projection 三段边界与 provider→coordinator/finalizer 注入 ports |
| `memory-logical-replay` | §9.2/§10.1；逻辑 Allocate/Bind/Use/Free 重放与 peak |
| `time-progress-des` | §9.3/§10.3；精确 compute record、理论 communication、progress DAG 与无竞争 DES |
| `result-gate-comparison` | §10.4/§10.5/§12.2；source/value/seal authority、ledger、comparison 与 conformance 旁路 |

### 0.2 十组模块契约索引

每组契约的 HTML anchor 即下表 locator；其中集中列出模块责任、核心结构、唯一生产者/消费者、正式入口、
显式结果联合、blocker/InternalContractViolation 边界以及缓存与序列化不变量。

| module ID / locator | 核心结构与正式入口 |
|---|---|
| `input-facts` / `mc-input-facts` | Source/Registry/Request snapshots；`build_request_snapshot(...) -> RequestSnapshotBuildResult` |
| `code-ir` / `mc-code-ir` | CodeIR、RankCodeIR、obligation、leaf IR types；`evaluate_source(...) -> RankCodeIRBuildResult` |
| `runtime-events` / `mc-runtime-events` | RuntimeEventPlan、ExecEvent、semantics、autograd/lifetime/intents；`expand_runtime_semantics(...) -> RuntimeBuildResult` |
| `plan-projection` / `mc-plan-projection` | SimulationPlanCore、typed ProjectionCandidate/Result 与 ProjectionBundleBuild；`bind_core(...)`、`evaluate_and_finalize_projection_bundle(...)` |
| `memory-backend` / `mc-memory-backend` | Storage/Workspace bindings、MemoryEventView/Estimate；`build_memory_projection_candidate(...)` 与 `run_memory_backend(...)` |
| `time-backend` / `mc-time-backend` | stream/cost/communication bindings、TimeEventView/StepTimeEstimate；`build_time_projection_candidate(...)` 与 `run_time_backend(...)` |
| `result-sealing` / `mc-result-sealing` | candidate 与 source/value/seal authority；`run_backend_build_candidate_and_seal(...)` |
| `gate-system` / `mc-gate-system` | GateManifest/Clause/Ledger/Blocker；`compile_gate_manifest(...)`、`run_gate_domain(...)`、ledger extension |
| `comparison` / `mc-comparison` | typed arms、basis/source/candidate/seal；basis 派生、authority 构造与逐 metric comparison |
| `conformance` / `mc-conformance` | `BasicOfflineReportArtifact` / `AttestedConformanceSealArtifact`；`run_basic_offline_conformance(...)`、`run_attested_release_conformance(...)` 与 `apply_release_policy(...)` |

## 1. v4.2 已定边界

### 1.1 逐 rank CodeIR

- trainer 按当前配置分配 transformer block，因此每个 logical rank 使用由 P 派生的
  `RankEvalContext(r)` 独立求值。
- 顶层 `CodeIR(P)` 是规范 rank 顺序的有序映射，不是一张带 rank guard 的母图。
- CodeIR 创建后不可变；不做算子融合、消除、替换或通用图优化。
- 对外引用使用 `(logical_rank, local_id)` 限定；physical device/stream 只在 SimulationPlanCore/投影出现。
- 独立句法 pass 为每个候选调用 occurrence 生成 SourceObligation；guard 只作为 branch evidence，
  不单独成为 obligation。每项恰为 CodeNode、Residual 或有证据的 ProvenNotExecuted，且双向闭合；
  分母不能从已经生成的 IR 反推。
- PySub 世界由 SourceSnapshot、CompileEnvFacts 与冻结 registry 封闭。未快照 hook/dispatch/monkey
  patch 若可能改变调用图，必须阻断，不能静默静态绑定。

### 1.2 microbatch 与初态身份

```text
ValuePrototypeRef = RankTensorRef | RuntimeTensorRef
StoragePrototypeRef = RankStorageRef | RuntimeStorageRef | WorkspaceRef

TensorInstanceId =
  (ValuePrototypeRef, EventId | InitialTensorRef, output_ordinal)

StorageInstanceId =
  (StoragePrototypeRef, allocation_occurrence_id)
```

`allocation_occurrence_id` 区分 microbatch、recompute、prefetch、rematerialization 与 persistent live-in。
ExecEvent 输入/输出、saved tensor、gradient relation 和 Memory Bind/Use 都引用 TensorInstanceId，
不能使用裸 TensorId 混同不同 microbatch。

RuntimeTensor/StorageRef 使用 `RuntimeRuleInvocationRef(rule_id, owner_origin, event_instance_key)`，承载
backward、optimizer、通信等 rule 的每次派生对象；role-local ordinal 不能跨 invocation 复用。
每个 storage-bearing TensorInstance 都有唯一 `TensorStorageBinding`；shape/dtype/relevant attrs 与
`MemoryBindingKey` 随 RuntimeEventPlan 序列化。WorkspaceRef 只在 memory projection 由独立
`WorkspaceBinding` 创建，不依赖 duration CostBinding，
不能伪造 CodeIR identity。`LogicalInitialState` 随 RuntimeEventPlan
传递；`SimulationInitialState` 再显式给出绑定后的 persistent live storage、初始 tensor binding 和 state version。
每个 StorageInstance 必须恰来自 initial live-in 或一次 Allocate，二者互斥。
`cold_start` 指参数已常驻但 lazy optimizer/runtime state 尚未创建的首个训练 step；
`steady_state` 把已声明 persistent state 全部作为 live-in；模型构造不属于 step metric。

### 1.3 正反向与 event obligation

- 每个 forward occurrence 逐 rank、逐 microbatch 实例化。
- 每个要求梯度的 forward TensorInstance 必须有闭合的 saved-tensor、gradient producer/consumer、
  accumulation 与参数更新关系。
- 每个 runtime occurrence 先生成 EventObligation；每个 obligation 恰为
  `EventId | BlockerInstanceId | NotApplicable`，coverage 按三分支守恒，每个 Event 也恰有一个反向授权。
- `AutogradLink` 显式关联 forward/backward/recompute、saved tensor、gradient、accumulation 与 update；
  重复 origin/microbatch 不能按位置猜配对。缺规则返回 `BLK-EVENT-SEMANTICS`。
- recompute 产生新的 EventId、TensorInstanceId 和必要的 StorageInstanceId，但继续引用原
  RankCodeNodeRef。
- LogicalScheduleSpec 中必须影响实际发射/完成的条件全部 lower 为 `ScheduleConstraintEdge`；
  `logical_kernel_order` 只是内存重放/tie-break 全序，不能替代跨 stream 执行边。

### 1.4 最小 effect 顺序规则

`StateRef` 是 canonical logical state location；tensor mutation 归一化到 alias root，RNG generator
使用具名 StateRef。

```text
conflict(a,b) :=
  W(a) intersects (R(b) union W(b))
  or W(b) intersects (R(a) union W(a))

overlap_permitted(a,b) :=
  not conflict(a,b)
  and disjoint(nonoverlap_domains(a), nonoverlap_domains(b))

base_event_graph = data/required-source-control/ScheduleConstraintEdges
canonical_base_order = stable_topological_order(base_event_graph, canonical instance key)
for every conflicting unordered pair:
  orient the effect edge from earlier to later in canonical_base_order
dependencies = base edges union all required effect edges
```

`EffectDomainRef` 精确声明不可重入互斥域，不能把所有 `reentrant=false` 事件全局串行，也不能只凭
semantic_id 猜域。必须遵守的 source order 先成为 control edge；所有 conflict 沿 base DAG 的同一拓扑全序
定向，避免局部 pair 方向把可串行化 DAG 人为造环；phase/None/EventId 的 tie 全序固定。
`overlap_permitted=false` 必须由已有 path 或确定 EventEdge 物化。RNG recompute 必须有 save/restore
与版本闭合。无法确定 StateRef、顺序或物化后成环时返回 `BLK-EFFECT`。本版不建立通用 effect algebra。

### 1.5 layout/stride

- 不建立精确 layout/stride/逐元素地址模型。
- transpose/view 按注册语义作为 metadata alias，不预期隐式 reorder。
- 只有明确声明为 new、dense、无 padding 的 storage 才能用 shape × dtype 派生 bytes。
- 若 new/view 判断依赖未建模的 implicit contiguous/materialization，返回
  `BLK-UNSUPPORTED-LAYOUT`，不能默认 contiguous。

## 2. 两个后端

### 2.1 MemoryEventView

`MemoryEventView.per_rank` 是覆盖全部 logical rank 的 OrderedMap；每个 Runtime storage 先得到唯一
`StorageBinding`，payload bytes 再按目标 device alignment 规范化；每个 `RankMemoryEventView` 含绑定后的
initial state、BoundStorageInstance、完整 RuntimeEventPlan logical order 和规范 MemoryEvent 序列。
`ExpectedMemoryProjection` 由 runtime plan、hardware facts、memory registry、唯一 StorageBindings/WorkspaceBindings
纯函数式生成，要求 rank/storage/binding/use/event 集合和排序完全相等，禁止幽灵 workspace 或漏事件。

```text
LogicalAnchor =
  step_start | before(event_id) | at(event_id) | after(event_id) | step_end

stable order:
  step_start
  → for event in logical_kernel_order: before → at → after
  → step_end

same anchor:
  Allocate → Bind → Use → Free → memory_event_id
```

- 每个 rank view 的 logical_kernel_order 字节等于 RuntimeEventPlan 的权威顺序；MemoryEventId 是
  `(rank, anchor, kind, subject ids, ordinal)` 的规范 tagged tuple。
- projection 只读取冻结 hardware alignment 并把 aligned bytes 写入 view；后端重放不再读取 raw hardware、
  physical stream、duration、start/end、resource state 或 TimeEstimate。
- 本版不建模多 stream 完成次序、跨 stream allocator reuse 或由完成时刻触发的提前 free。
- 每个 RuntimeEventPlan StorageInstance 恰有一个 `LogicalLifetime(release_event_id|None)`；多个 release candidate 阻断。
  `dependency_safe(u,r) := u==r || path_length>=1(u,r)`。非空 release 必须对全部 Use 安全；只有 None
  才取 step_end 并记录 assumption，不搜索或任选 logical join。
- Runtime storage 使用 RuntimeEventPlan.LogicalLifetime；workspace 使用 WorkspaceBinding 中唯一固定的
  `before(owner event) → after(owner event)` transient lifetime，两种来源互斥。跨 event/persistent 临时区
  必须建模成 RuntimeStorageRef。
- peak 可发生在 `INITIAL_STATE`。
- 峰值 tie order 为 `INITIAL_STATE < canonical MemoryEventId`；输出权威值是逐 rank peak map，cluster max
  只是按最小 LogicalRank 打破并列的派生摘要，rank class 必须可无损展开。

### 2.2 P2P 与 collective

`P2PIntent` 显式记录 src/dst、send/recv EventId、channel、sequence、bytes、payload TensorInstance、
buffer owner/lifetime 与 `protocol_semantics_ref`。公式/route 只在 time projection 的硬件成本绑定中选择。
PP activation/gradient 和 CP ring 不得只用跨 rank
dependency 暗示通信。

v4.2 支持 `matched_rendezvous`：两端 endpoint 及其前驱都 ready 后 cohort 启动；完成后两端后继
才能推进。eager buffering、后台 progress 或协议阈值未显式建模时阻断。P2P buffer owner/lifetime
必须降低为明确 StorageInstance 与 LogicalAnchor，不能只留在通信标签里。

collective 保留 participant endpoint/group/sequence/bytes/correlation；time projection 为整个 cohort 恰选择
一个对全部 endpoint 适用的 algorithm/formula。本版只支持 all-participant rendezvous；非阻塞/background
progress 未升级语义时阻断。全局 wait-for graph 同时包含：

- explicit EventEdge；
- 相邻 stream sequence；
- P2P endpoint/channel sequence；
- collective rendezvous/group sequence。

只检查各 rank 的本地 DAG 不够；联合 graph 必须无环并可结束。

通信内存只含 intent/runtime rule 显式声明的 payload、endpoint、bucket/buffer。随 algorithm 变化的通信库
内部 scratch 本版不建模；发现依赖时阻断 memory，不能让 memory/time 暗选不同算法。

### 2.3 TimeEventView

```text
ProgressNode =
  BoundComputeEvent | BoundP2P | BoundCollective

event_projection: EventId → ProgressNodeId
  non-communication event → own node
  every matched communication endpoint → the one cohort node
  image(event_projection) == progress_nodes; no extra node

PhysicalStreamId = (device_id, local_stream_id)
stream_bindings = unique_stream_assignments(
  runtime_plan, HardwareProjectionFacts, StreamAssignmentPolicy)

progress_edges == canonicalize(
  typed_project(RuntimeEventPlan.EventEdge)
  ∪ stream adjacency
  ∪ collective group sequence
  ∪ P2P channel sequence)

step_start_ns = 0
for node in stable_topological_order(global_progress_graph):
  start_ns(node) = max({0} ∪ {end_ns(pred)})
  end_ns(node) = checked_add(start_ns(node), bound_duration_ns(node))
empty graph → step_time=0, timeline=[], critical_path=[]
```

TimeEventView 内嵌 exact measurement+protocol snapshots，或 normalized formula AST/inputs/route 与
NumericPolicySnapshot，并在投影时复核 digest、复算 `bound_duration_ns`；DES 只消费已校验整数，
不回读 core/CalibrationSet/registry。endpoint quotient 后不再单独计时；只有同 intent 的
`communication_internal` self-loop 可丢弃，其他被 quotient 成自环的 data/control/schedule/effect edge 阻断。
ProgressNodeId 为 `tag(compute, EventId)` 或 `tag(communication, intent kind/id)`；并列拓扑/关键路径/
step-end 按它打破。busy/overlap/stage bubble 用半开 interval union 定义，service 按 compute event 或
communication cohort 恰计一次，并由 G-TIME6 重算。

TimeEventView 还内嵌 `selected_device_domain`、`pipeline_stage_domain`，stage domain 唯一由
RuntimeEventPlan 中完整 LogicalRankContextSet 派生而非从实际 event 反推；每个通信 node 携带规范 endpoint
descriptor `(EventId, rank, device, stage, PhysicalStreamId)`；因此零事件 stage/空图的零值 map 与通信跨 stage 归因都不回读 core。
View 还内嵌全局 timeline NumericPolicySnapshot 与 NumericRangeWitness；Ready 前用任意精度执行同一 DES 和
全部 interval/busy/bubble/service 公式，任一实际输出标量越界则 `BLK-NUMERIC-RANGE`。这不会用并行节点简单总和误杀合法图，
也保证后端不会在 Ready 后首次溢出。
`ExpectedTimeProjection` 还逐字段固定每个 BoundComputeEvent 的 origin/rank/phase/microbatch/stage/chunk、
StreamBinding 与 CostBinding；只保留 EventId/duration 后篡改分组字段不能过门。
`communication_internal` edge 只有在同 intent endpoint quotient 成 self-loop 时可丢弃，其他情形必须阻断，
不会流入不支持该 kind 的 ProgressEdge。

没有 `resource_ready`、capacity reservation、resource request 或 contention arbitration。不同 stream
且无 progress edge 的节点可完全重叠，即使物理上共享 compute/HBM/link；并发降速不建模。

## 3. 成本数据、lineage 与摘要

### 3.1 compute measurement

`CalibrationSet` 是用户在待预测请求之前采集、规范化、冻结并版本化的 compute duration 记录与
MeasurementProtocol snapshots；workspace 规则/记录属于独立 MemoryRegistrySnapshot。
Structure/Runtime 只生成硬件无关 `MeasurementKeyTemplate`；core 用 HardwareProjectionFacts 与
HardwareBindingPolicy 生成 shared KernelVariantBinding，time projection 再结合 TimeCostPolicy 补全
`MeasurementKey`。完整 key 覆盖 semantic/kernel、精确
shape/dtype/attrs、硬件/runtime 与 measurement_protocol_digest；
`MeasurementProtocol` 要求隔离、同步、warmup/repetitions/statistic 和 `co_runner_none`。
active record 的 key protocol digest、ObservationRef protocol ref、selected protocol digest 与
TimeCostPolicy.required protocol digest 必须四者相等。
每个 key 必须恰有一个 active record，或使用冻结且进入摘要的 reducer 预先归并。

`MemoryRegistrySnapshot` 冻结普通 storage size/category、compute workspace 与 alignment 规则；每个 runtime
storage 形成唯一 StorageBinding，每个 compute workspace 形成唯一 WorkspaceBinding。`align_up(0,a)=0`，
其余 bytes 用 checked ceil-div/multiply，不能由实现选择 raw/aligned 两套口径。

时间后端只允许 exact measurement hit；禁止插值、roofline/FLOPs/零值 fallback。0 个候选返回
`BLK-MISSING-TIME`，多个同等候选返回 `BLK-AMBIGUOUS-COST`。生产 lookup 只消费 calibration_train；
holdout 只评估采集/适用性误差，不参与 duration 定价；conformance_fixture 只用于发布验证。

### 3.2 communication formula

`CommunicationModelSnapshot` 固定 formula id/version/digest、kind、algorithm、适用域及所需
bytes/participants/path/link-latency/bandwidth。公式只生成单个通信 cohort 的固定 duration；
不预留 link capacity，也不使并发通信彼此降速。
完整 NumericPolicySnapshot（Decimal precision、rounding、integer ns range、NaN/Inf/negative 规则）随
CostBinding 嵌入；仅有 numeric policy digest 不足以复算。

### 3.3 数据隔离

`sample_key = hash(raw artifact digest, coordinates/window, collector, hardware/runtime, quantity, unit)`，
不含 split label；`split_group_key = hash(run/config/protocol/hardware group)`。
`calibration_train`、`holdout`、`conformance_fixture` 必须按 sample_key、split_group_key 与 lineage ancestor
两两不相交；当前待预测运行不能出现在生产测量 lineage。生产进程不读取 raw profiler/TraceFixture 路径。
digest/lineage 只能证明内容一致，不能证明测量真实或 split 独立；需要更强保证时依赖方案外的签名采集器/
attestation 信任根，否则标为“用户提供的测量事实”。

```text
model_input_digest =
  source/model/P/rank/code-shape bindings/compile facts
  + evaluator/PySub/descriptor/dispatch/canonical-ID/structure-registry semantics
model_digest = model_input_digest + canonical_payload_without_derived_digests(CodeIR)
runtime_input_digest = model_digest + scenario + runtime semantics
runtime_plan_digest = runtime_input_digest + canonical_payload_without_derived_digests(RuntimeEventPlan)
simulation_core_digest = runtime_plan_digest + hardware + ExecutionDeployment + HardwareBindingPolicy
                         + shared KernelVariantBindings
                         + canonical_payload_without_derived_digests(SimulationPlanCore)
memory_simulation_digest = core + MemoryRegistry + StorageBindings + WorkspaceBindings
                           + projection/identity/policy/assumptions/backend semantics
time_simulation_digest = core + CalibrationSet + communication snapshot + TimeCostPolicy
                         + exact CostBindings + StreamBindings + assumptions/backend semantics
result_digest = hash(Estimate除result_digest自身之外的value、evidence、coverage、assumptions、
                     EstimateContext digest与上游digests)
```

所有内容摘要统一使用逐 schema 排除表：CodeIR 只排除 model_digest、RuntimePlan 只排除 runtime_plan_digest、
Core 只排除 simulation_core_digest，各 Binding/Snapshot/Policy/EstimateContext/GateManifest/GateExecutionLedger/Estimate 只排除
自己的派生 digest；RequestSnapshot、EvaluationInstanceIdentity、ProductionSubject/ValidationContext、
ProjectionCandidate/ProjectionBundleAuthority、
EstimateCandidate、backend source/value/seal authority/candidate/artifact 与 comparison source/candidate/seal authority
也各自只排除 schema 明列的自身 digest。ExecutionDeployment、MeasurementProtocol、CommunicationFormulaRef 与
ResolvedEventSemantic 也各自只排除自己的 deployment/protocol/formula/semantic digest。嵌套输入 digest
保留；KernelVariantBinding 的自身字段固定名为 `kernel_variant_binding_digest`；禁止把自身置零后 hash 或由实现自行删字段。

TraceFixture、AttestedConformanceReport 和 ConformanceVerdict 不进入生产摘要、比较 basis 或缓存键。
requested backend set 与 NotRequested 也不进入仿真摘要；memory/time result 使用独立 cache namespace。
EvaluationInstanceIdentity 只是 request/config capability provenance，不进入 model/runtime/backend simulation digest、
Estimate.result_digest 或模型结果 cache key。缓存命中的 backend value 只能在当前 evaluation 的
joint projection bundle 已做 scope 闭包且该侧 Ready 后作为 proposed value，必须重跑当前 value/result-seal clauses 并形成带
`(request_digest,evaluation_instance_digest,backend)` 的新 BackendSealArtifact；comparison 不消费裸缓存结果。

## 4. 输出与严格比较

```text
BackendResult<T> =
  Ok { estimate: Estimate<T> }
  | Blocked { diagnostics: NonEmpty[Diagnostic] }
  | NotRequested
```

Blocked 不得携带 null/0 value、空 timeline 或旧缓存值。memory/time 可独立成功；time blocked 不影响
duration-independent MemoryEventView。
所有构建阶段 Blocked 都携带 `BlockerRecord`，最终 Diagnostic 的 instance/code/stage/scope/context 必须逐字派生；
`BlockerCode` 只分类，`BlockerInstanceId` 区分同码的不同根因，scope 由规范 BlockerScopePolicy 求值。
`RequestSnapshot.canonical_production_evaluation_inputs.configurations` 以 ConfigRef 唯一绑定正则 P 与该配置的
scenario/deployment input slice；`requested_backend_inputs` 对 memory/time 分别携带 `Requested{...}` 或无 payload/digest
的 `NotRequested` 标签，未请求侧事实不得进入 request digest。`EvaluationInstanceIdentity` 由 request digest、config_ref 与 slice digest 派生。两个不含
gate outcome 的 `ProjectionCandidate` 必须绑定同一 identity，再进入同一 `ProjectionBundleAuthority`。
bundle 同时保留权威 RuntimeBuildResult 与 CoreBuildResult：Runtime Ready 时 core 必须精确等于对该 plan 的
bind_core 结果，因此 plan.blocker_index 在 core 后续 Blocked 时仍然可读；Runtime Blocked 时 core 只能携相同 blocker。
任何使 RuntimeEventPlan/Core 或 shared G-IR invocation 无法 Ready/Pass 的 blocker 必须精确 scope 到
`{memory,time}`；单侧结构 face 问题必须保留 shared artifact，下沉到该侧 candidate/view gate。
joint projection ledger 恰执行一次 shared structure domain，加 memory/time 两个 view domain；在产生任一
ProjectionResult 前，必须先规范聚合 pre-plan/core、RuntimeEventPlan、两个 construction candidate 与两侧 gate 的
全部 InputBlocker occurrence，再按 `affected_backends` 投影。这同时保证 CandidateBlocked 不丢已有 blocker，以及
time/P2P gate 产生 `{memory,time}` scope 时 memory 不会先行 seal 为 Ok。只影响 time 的 blocker 仍允许 memory Ready；
未请求侧始终 NotRequested；`RequestSnapshot.requested_backends` 必须精确等于 tagged map 中 Requested arm 的 key 集，
memory-only 请求不要求或指纹化 time facts，time-only 请求同样不要求 MemoryRegistrySnapshot。
Projection Ready 当且仅当已请求、scope-closed `affected_blockers` 为空、candidate 为 View，且显式
`ready_prerequisites(memory|time)` 全过。Ready 之后的 backend value/report gate 失败不是用户 blocker。两组
view prerequisite 分别是 G-IR1..3 + G-MEM1..4 / G-TIME1..5。后端取得 Ready view 后先生成不可外泄的
`EstimateCandidate`；其 digest 覆盖 value 与 execution witness。`BackendValueGateAuthority` 绑定 source authority 与这一个
具体 candidate，G-MEM5/6 或 G-TIME6 的每条记录必须绑定该 authority digest，不能先验正确 value 后替换。
随后形成三态 `BackendResultCandidate(ProposedOk|ProposedBlocked|ProposedNotRequested)`；G-REP1/3 对当前 candidate
统一校验，全部通过才形成 `BackendSealArtifact`，对外 API/缓存只暴露 artifact.result，比较器消费完整 artifact capability。
该 capability 的身份是 `(request_digest,evaluation_instance_digest,backend)`，不可由 API 输入或反序列化构造，
唯一构造器是当前请求/配置/backend 的 seal；digest 只证明内容完整性，不能把
用户传入的同形 JSON 变成已验证 artifact。
value/REP gate 自身失败进入规范排序的 `InternalContractViolation` 全集并终止请求，不得递归包装成未经同门验证的 Blocked。
`BackendResultSourceAuthority` 固化完整 ProjectionBundleAuthority（含 Runtime/Core 两层权威结果）、该侧最终
ProjectionResult 与 joint projection ledger；
pre-seal ledger 必须字节级保留 shared+memory+time 的完整 bundle prefix，再只追加当前 backend value stage。
candidate 绑定 source/value authority 与 pre-seal ledger digest；
result-seal 记录再绑定 BackendSealAuthority+完整 BackendResultCandidate digest。sealed ledger 只作为 request validation
audit，不进入模型 evidence/result/cache identity。不能用 builder 调用路径、局部自洽对象或未绑定 loose records 冒充 provenance。
ProductionSubject 必须从 build artifact、evaluator/registry、production policy、backend semantics 四组 constituent digest
复算；ProductionValidationContext 与 candidate/source authority digest 绑定 lineage/exclusion facts。
G-REP2 仅按比较请求走独立 comparison seal，因此不存在 Ready→Estimate→gate→Ready 的环。deployment/core build 失败传播到
两个已请求后端，未请求侧仍为 NotRequested。
Diagnostic.subject_ref 覆盖 request/config/deployment/source/residual/event-obligation/runtime-rule/binding 等阶段；
source_ref 可空，但每条诊断至少有一个与 blocker stage 匹配的稳定 context ref。
每个 Ready view 内嵌唯一 `EstimateContext(evidence, coverage, assumptions, model/runtime/backend digests)`；
projection gate 对照规范 producer，后端只能逐字复制并生成 value/result_digest，不能自由重填 metadata。
MemoryEstimate/StepTimeEstimate 是纯 payload；coverage、assumptions、evidence 与 digests 只在外层
Estimate 出现，不能在一个 Ok 内维护两套 metadata。
MemoryEstimate 的权威 payload 是 `OrderedMap<LogicalRank, RankMemoryEstimate>`；cluster max/rank class
仅为可重算派生字段。StepTimeEstimate 的 busy/overlap/bubble/service 字段都由 timeline 半开区间公式产生。

```text
ComparisonResult = {
  comparison_axes,
  memory: MetricComparison<MemoryDelta>,
  time: MetricComparison<TimeDelta>
}

MetricComparison<T> =
  ComparableDelta { comparison_basis_digest,
                    left_simulation_digest, right_simulation_digest,
                    delta, coverage_delta }
  | Incomparable { basis_mismatches }
  | Unavailable { left_status, right_status, diagnostics }
  | NotRequested
```

`ComparisonBasis` 有规范 schema 与 hash；每个 metric 的 `ComparisonBasisPair` 同时保存 left/right basis，
不等时从该 pair 的规范结构差异派生 Incomparable，相等时才形成 common digest。`masked_config_evaluation_input`
从完整 `CanonicalConfigEvaluationInput` 开始；normalized P/ExecutionScenario 仅在校验过的 comparison_axes 及其
schema-declared cross-field deterministic derivation closure 上用 sentinel mask，其余字段（包括未落入闭包的
logical rank context、deployment 与 `other_config_indexed_production_inputs`）必须相等。microbatch 通过
P.batching 作为 axis；未声明的 scenario 变化仍不可比。每个 metric 独立判断：basis 相同且两侧均 Ok 才返回 ComparableDelta；basis 不同返回
Incomparable；任一侧阻断/单侧未请求返回 Unavailable；两侧均未请求返回 NotRequested。只有
ComparableDelta 携带数值 delta。
time 两臂必须使用同一 measurement_protocol_digest；不同统计协议 Incomparable。relative delta 固定为
`(right-left)/left`；左基线为 0 时返回 `UndefinedZeroBaseline`，不能输出 Inf/NaN；MemoryDelta 要求
logical-rank id set 相同。

ComparisonResult 不能由裸 `compare_per_metric` 直接外泄。显式 comparison request 先构造
`ComparisonArmAuthority(config_ref,evaluation_identity,projection_bundle_authority_digest,artifacts)`，并严格要求每个 artifact 的
`(request_digest,evaluation_instance_digest,backend)` 与所在臂一致。
每臂的 memory/time artifact 还必须共享字节相同的 ProjectionBundleAuthority，arm 中的 bundle digest 与
nested authority 等式可复算；该 bundle 的 ProductionValidationContext/GateManifest 必须与当前 comparison source 字节相同，
不能在同 request/evaluation identity 下混用旧 subject。随后构造 `ComparisonSourceAuthority`，
绑定 RequestSnapshot、左右 typed arm、逐 metric basis pair、ProductionValidationContext
与 GateManifest；生成不可见的 `ComparisonResultCandidate` 后，G-REP2 的逐 clause ledger 必须绑定
ComparisonSealAuthority+当前 candidate digest。全部通过才返回 ComparisonResult；失败固定为
`CV-COMPARISON-CONTRACT` 且无结果、缓存或后续报告。成功时形成 ComparisonSealArtifact，外部只暴露
artifact.result；没有 comparison request 时不构造比较 candidate/artifact。

## 5. 18 道结构门

### IR · 3

- `G-IR1`：rank/qualified identity 完整；SourceObligation 三分支双向闭合，Residual 引用唯一 BlockerRecord 且 scope 合法。
- `G-IR2`：摘要排除表可复算；model_input 覆盖 evaluator/PySub/ID 语义，CodeIR/model digest 硬件独立。
- `G-IR3`：EventObligation、ResolvedEventSemantic、TensorStorageBinding、structural report facts、正反向关系
  与规范 data/control/schedule/effect 边集完整；effect 沿 base DAG 的规范拓扑全序定向。

### Memory · 6

- `G-MEM1`：Storage/WorkspaceBinding 唯一，alignment 等式、per-rank expected set 与 origin XOR 成立。
- `G-MEM2`：runtime LogicalLifetime 与 workspace BoundWorkspaceLifetime 各自唯一互斥；Free/release 合法。
- `G-MEM3`：producer 与 TensorStorageBinding 唯一，Bind/Use 指向 live storage，range/release 安全。
- `G-MEM4`：view order/initial/storage/events 等于 ExpectedMemoryProjection；live bytes 与集合一致。
- `G-MEM5`：作为 EstimateCandidate 后置门，每 anchor/category 精确等于该类 live bytes，分类与总和均守恒。
- `G-MEM6`：作为 EstimateCandidate 后置门，逐 rank peak/tie、cluster max 与规范 RankClass ID/member/representative/展开可重算。

### Time · 6

- `G-TIME1`：deployment/device domain 合法、stage domain 来自完整 rank contexts；progress node 恰等于
  ExpectedTimeProjection，compute 复制字段和 endpoint quotient 无漂移/幽灵；communication_internal/typed self-loop/expected edge 等式成立，图可结束。
- `G-TIME2`：每 Event 唯一绑定 PhysicalStreamId；StreamBinding/sequence 与冻结 policy/计划精确对应；
  schedule/effect edge 生效，cohort 占用全部 endpoint stream。
- `G-TIME3`：P2P send/recv/src/dst/channel/sequence/bytes/protocol/buffer contract 一一匹配。
- `G-TIME4`：collective participant/group/sequence/bytes/correlation/algorithm 跨 rank 匹配。
- `G-TIME5`：ResolvedEventSemantic 的 hardware-free key template 正确补全；compute record/key/ObservationRef/policy
  protocol identity 四者闭合，comm formula/route+numeric snapshot 唯一且嵌入成本可复算；任意精度 preflight 证明全部输出标量不越界。
- `G-TIME6`：作为 EstimateCandidate 后置门，只从 Ready view/witness 重算起点0/空图0、checked ns、规范 tie 与全部聚合，无暗读 core/contention。

### Report · 3

- `G-REP1`：封装前按 BackendSealAuthority 验证 source/value/seal authority、BackendResultCandidate 及三分支；
  `(request_digest,evaluation_instance_digest,backend)` 全链相等；请求只读 RequestSnapshot，Ok 的完整
  value/witness 与 value authority 相等，Blocked 精确来自 scope-closed bundle，NotRequested 请求状态闭合。
- `G-REP2`：只在 typed comparison arms→source→candidate→ledger→seal 路径对当前 candidate 执行；
  左右 config_ref/evaluation identity/artifact 不可交换；同臂 artifact 共享完整 bundle/current ProductionSubject/GateManifest；
  完整 CanonicalConfigEvaluationInput 仅 mask P/scenario axis 及跨字段派生闭包，basis pair、逐 metric sum type、
  协议与相对差公式合法。
- `G-REP3`：同一 evaluation 的两个 ProjectionCandidate 在 bundle authority 上联合逐 clause 验证；bundle 同时保留
  RuntimeBuildResult/CoreBuildResult，plan blocker 不因 bind_core 失败丢失；shared build/G-IR blocker 必须双侧，shared invocation
  恰执行一次，全部 InputBlocker occurrence 先全局聚合再按 affected_backends 派生两侧 Result；GateExecutionLedger
  对 bundle/backend/branch/stage/clause domain 无增漏，完整 bundle prefix/context 不可覆写，value/result records 绑定当前
  candidate；BlockerRecord/Diagnostic scope 不分叉，contract failure 不伪装 blocker；
  sample/split/lineage/train-only 合法；fixture/verdict 不污染生产。

权威正文把每个复合 gate 分解为 `GateClause`；GateManifest 以
`GateInvocationId=(GateId,backend,branch)` 表达条件化 stage/dependency DAG。每个适用 invocation 的
`clause_outcomes: OrderedMap<GateClauseId, GateClauseExecutionRecord>` 必须精确覆盖全部 clause；manifest map key、
GateManifestEntry.invocation_id、record.invocation_id 三方相等，clause map key、record.clause_id 与 manifest clause_id 三方相等。
GateClause.failure_disposition 是唯一权威，ClauseFail 只携带 failure_instances，不能重标 disposition。仅 all-pass 才派生 Pass，
任一失败即保留完整 failed-clause set，多个 InputBlocker 全量输出，任一 InternalViolation 优先终止但仍保留完整 violation set。
每个 gate/clause record 绑定 stage evaluation-input digest、clause id 与 predicate implementation digest；NotApplicable
也由具内容摘要的 applicability predicate 与证据产生。账本的 stage-context map key 等于 covered stages；projection、value、
result-seal 与 comparison 分别绑定自己的 immutable authority/candidate；projection ledger 的 domain 是
shared structure∪memory view∪time view，backend pre-seal 保留完整 bundle prefix 后仅加本侧 value domain，sealed ledger
再仅加本侧 result domain。extension 不得覆盖旧 prefix/context，从而
memory/time、Ok/Blocked/NotRequested 不互相制造假依赖，也不能用游离 record 或漏跑 clause 冒充已执行门。
每条 gate record 的 invocation key、manifest-entry payload digest、subject 与 runner 还必须逐字回对 manifest/ledger。
每个 clause 绑定 predicate implementation digest、正例、边界负例、assertion，以及全函数
`GateFailureDisposition = InputBlocker(BlockerCode,scope) | InternalViolation(code)`。backend value、result seal
和 comparison gate 失败一律是 InternalViolation。GateManifest 还绑定 subject、runner 与自身 digest。
`verify_gates.py` 只做文档 lint，不证明运行时 predicate；发布测试必须逐 clause 执行。

## 6. conformance

### 6.1 部署 profile 闭包与共享事实

部署侧选择是一个无摘要、不可序列化且不会进入 `RequestSnapshot` 的闭联合：

```text
ConformanceDeploymentProfile :=
  BasicOfflineConformance
  | AttestedReleaseConformance {
      conformance_trust_root: ConformanceTrustRootCapability,
      release_trust_root: ReleaseApprovalTrustRootCapability
    }
default_conformance_deployment_profile := BasicOfflineConformance
```

没有显式 profile 时只运行 Basic；显式选择 Attested 却缺少任一受保护 capability 是
`InternalContractViolation`，不得静默降级。两条分支复用唯一的 TraceFixture、FixtureBinding/Set、
ValidationPolicy、VerifierRunner、ProductionSubject、GateManifest、ConformanceFinding、
ConformanceObservedOutput 与确定性 finding/verdict 规则。
Figure 10 先以 OfflineConformanceRequested opt 表达整个离线 sidecar 可选，其内再以 Basic(default)/Attested 单一 alt 表达 profile 互斥；gate system 只返回 clause observations 与
`GateExecutionLedger`，Basic/Attested validator 各自构造 `BasicOfflineReportArtifact` / `AttestedConformanceSealArtifact`，
只有 Attested 分支随后调用 `apply_release_policy`。

### 6.2 BasicOfflineConformance

`ConformanceVerdict = pass | fail | insufficient`。Basic 的唯一入口与结果边界是：

```text
run_basic_offline_conformance(
  authority: BasicOfflineConformanceAuthority
) -> BasicOfflineRunResult
```

`BasicOfflineConformanceAuthority`、`BasicOfflineExecutionRecord`、`BasicOfflineExecutionLedger`、
`BasicOfflineReport`、`BasicOfflineReportArtifact` 与 `BasicOfflineRunResult` 中，前五种是 content-addressed payload，使用 `profile_schema_id="basic-offline/v1"` 与彼此独立的 `basic-offline-*/v1` hash domain；`BasicOfflineRunResult` 只是无 own digest、无 profile field 的 transport union。
Basic 执行精确 fixture domain，产出 content-addressed `offline_verdict`；内容摘要只证明 identity/integrity，
不证明 runner authenticity，也不能授权发布、附签升级或转换为 Attested artifact。

### 6.3 AttestedReleaseConformance

只有部署显式选择 `AttestedReleaseConformance` 才启用受保护 measurement session、runner attestation、
single-use nonce、外部 approval trust root 与发布策略。它的两个公开入口精确为：

```text
run_attested_release_conformance(
  authority: AttestedConformanceInvocationAuthority,
  profile: AttestedReleaseConformance,
  measured_environment: MeasuredExecutionEnvironment
) -> AttestedConformanceRunResult

apply_release_policy(
  artifact: AttestedConformanceSealArtifact,
  approval: ReleaseApprovalArtifact,
  policy: ReleasePolicy,
  profile: AttestedReleaseConformance
) -> ReleasePolicyApplicationResult

ReleasePolicyApplicationResult :=
  Completed { decision: ReleaseDecision }
  | InternalViolation { violation: InternalContractViolation }
```

Attested authority、report 与 seal payload 固定 `profile_schema_id="attested-release/v1"`；创建它必须使用新的
authority、fresh measured session、已消费 nonce 并完整重跑，不存在接受 Basic artifact 或公共 artifact union 的重载。
只有 `run_basic_offline_conformance`、`run_attested_release_conformance`、`apply_release_policy` 是公开端口；
envelope validation、seal validation、report/approved-set derivation helper 均为 internal/non-exported。

### 6.4 判定、批准与隔离

两条 profile 共享以下判定含义：

- missing/extra/unresolved node、错误 branch/dependency/communication identity 是结构性 fail；
- peak/step/start/end 等数值误差按版本化 validation policy；
- 没有独立 fixture 时为 insufficient，不能称 pass；
- Attested 分支的 `AttestedConformanceReport` 必须绑定 content-addressed subject、fixture set、validation policy、GateManifest 和
  verifier runner digests；
- subject 必须与 ProductionValidationContext 共用同一 ProductionSubject，并从 build artifact、evaluator/registry、
  production policy、backend semantics 四组实际 constituent digest 复算，不能接受 opaque 自填字符串；
- `GateManifest.subject_digest == AttestedConformanceReport.subject_digest`，runner digest 也同时匹配报告和批准清单；
- pass 必须先调用 `derive_approved_digest_set(artifact, policy)`，逐字重算并匹配以下全部十个字段：
  `subject_digest`、`fixture_set_digest`、`validation_policy_digest`、`gate_manifest_digest`、
  `verifier_runner_digest`、`runner_attestation_digest`、`conformance_execution_ledger_digest`、
  `conformance_report_digest`、`conformance_seal_artifact_digest`、`release_policy_digest`；
- pass 还必须由部署/verifier-owned、请求/fixture/policy/approval 无法构造的
  `ReleaseApprovalTrustRootCapability` 固定完整 ReleasePolicy 与 approval-store snapshot，限制支持的 scheme，
  并只使用 root-owned trusted approver public key material 验证覆盖 pinned policy/store identity 与上述十字段的签名；
  self-owned key、wrong store、unsupported scheme 或 substituted policy 即使内部 digest/signature 自洽也不得发布；
- `apply_release_policy` 必须先从当前 `profile.conformance_trust_root` 解析 store/policy，逐字匹配 artifact authority 的
  trust-store/policy refs，复算 store key registry/store/policy own digest 与 scheme 集合，并逐字闭合 environment、attestation、
  ledger、report 到 authority/runner/observed 的全部重复 identity；使用该 root 的 measurement-session authority 重验 evidence，并从 current-root nonce registry
  查询绑定精确 measured-execution-environment digest 的 terminal consumption receipt，
  并使用 root-owned key/scheme/message 重验 seal 内 runner attestation；这一步不再次消费 measurement session/nonce。
  另一 conformance domain 产生的 seal 即使 release approval 自洽也必须返回 InternalViolation；
- terminal receipt 的唯一规范 key 覆盖 nonce、invocation/runner/executable、session/process/container、time window 与
  measured-execution-environment digest；envelope validator 原子消费时写入该 key，seal 校验同一 receipt，release 只读查询同一 key；
- 每次 trust-store、approval-store 或 approver-key map lookup 前必须先验证 key membership；缺 attestation/approver key
  返回 InternalViolation，可信 approval store 中缺当前 subject 则返回 Completed(ReleaseBlocked/ApprovalMissing)；
- Attested 的 fail 必须阻止精确 subject；insufficient 的发布行为由显式版本化 ReleasePolicy 决定；
- verdict 不改变生产 BackendResult，生产摘要也不反向依赖报告。

CodeIR conformance 按 rank 对比并聚合 rank set；event conformance 覆盖 backward/recompute/optimizer、
TensorInstance、P2P/collective、microbatch/pipeline 与 logical allocation timeline。

Attested-only 会话边界已经闭合：`MeasuredExecutionEnvironment` 是 protected measurer 发行的
不可重放 envelope，摘要覆盖 invocation、runner/executable、process/container/session identity、single-use nonce、
execution time window、measured payload 与 measurer identity/freshness evidence；所有 fixture binding 只能在
`ValidatedMeasurementSessionCapability` 对应的同一调用域执行，runner attestation 签完整 envelope digest，
validator 在读取 observed/ledger 前复算并消费 nonce。跨会话重放或只替换 provenance/freshness 都是
InternalContractViolation。这个闭包仍以部署侧 `ConformanceTrustRootCapability`、`ReleaseApprovalTrustRootCapability`、
protected measurer、可信单调时钟、进程/容器身份提供者与 trusted approver public-key provisioning 为外部信任根；
摘要与签名不能自行证明这些物理根可信，也不能把 Attested offline conformance
表述为脱离该信任根的宿主隔离证明。

## 7. 缓存闭包

- Source parse：source + parser/evaluator semantics + compile parser facts。
- CodeIR lookup：`model_input_digest`，命中后重验 `model_digest`。
- RuntimeEventPlan lookup：`runtime_input_digest`，命中后重验 `runtime_plan_digest`。
- SimulationPlanCore：仅缓存 Ready core，以 `simulation_core_digest` 为键；CoreBuild Blocked 不伪造 core。
- Memory/Time result：分别使用 `(memory|time_simulation_digest, result_schema_version)`。
- 缓存 value 只能在当前 EvaluationInstanceIdentity 的 joint projection blocker closure 判定该侧 Ready 后作为 proposed value；
  必须在当前 `(request,evaluation,backend)` 上重建 source/value/seal authority 与 BackendSealArtifact。
- 完整 API response 默认不缓存；若缓存必须加入 requested backend set。`NotRequested` 不是 estimate cache value。

## 8. 文件与工作流

| 路径 | 状态与责任 |
|---|---|
| `src/index.template.html` | 唯一权威正文与 Mermaid 业务图源，只编辑这里 |
| `src/style.css`、`src/mermaid.config.json` | 页面样式与固定 Mermaid 配置 |
| `tools/render_mermaid.py` | 只调用本地 mmdc，规范化、加前缀并校验 SVG |
| `tools/build_doc.py` | 唯一活动构建入口 |
| `tools/verify_gates.py` | 文档合同与禁用语义 verifier；不是 runtime gate runner |
| `index.html`、`artifact.html` | `python tools/build_doc.py` 生成，不手改 |
| `stage0/` | 探针与阶段证据，不是生产输入 |

### 8.1 Mermaid 11.16.0 离线构建闭包

- `package.json` 与 `package-lock.json` 精确锁定 `@mermaid-js/mermaid-cli@11.16.0`。Node 需满足
  `^18.19 || >=20.0`。
- 新环境先运行 `npm ci`；若 npm cache 已含 lockfile 的全部 tarball，可使用 `npm ci --offline`。
  这是依赖 bootstrap，读者打开生成 HTML 不需要 Node、JavaScript 或网络。
- `python tools/build_doc.py` 只经 `tools/render_mermaid.py` 调用项目本地
  `node_modules/.bin/mmdc`（Windows 为对应 `.cmd` launcher），不回退到 CDN、在线 Mermaid 服务或全局 mmdc。
- `src/mermaid.config.json` 固定 `securityLevel=strict`、`htmlLabels=false`、
  `deterministicIds=true`、`deterministicIDSeed=target-design-v2` 与 `handDrawnSeed=271828`。后者固定
  Mermaid 11.16 复杂 class shape 内部使用的 rough geometry；渲染器再规范化 XML、属性、ID 和本地引用；
  确定性验收必须连续构建两次并比较两个产物的 SHA-256。
- SVG 校验拒绝 `script`、事件属性、`foreignObject`、外部 `href/src`、远程字体、`@import` 与 CSS
  `url()`；要求 `viewBox`、唯一 `title/desc`、原生 `text/tspan`、页面级唯一 ID 和闭合本地引用。

以下为 PowerShell 的活动构建与验证路径；只能由 `build_doc.py` 写 `index.html`/`artifact.html`：

实现阶段必须另提供内容寻址的 `validation/gate-manifest.json` 与 runtime gate runner；它们不是当前设计文档
linter 的产物。Basic profile 的 CI 只给离线质量结论；显式选择 Attested profile 的发布 CI 才以
AttestedConformanceReport 中的 manifest/runner digest 绑定精确制品，且不能只运行 `verify_gates.py`。

从 `docs/target-design-v2`：

```powershell
$env:PYTHONIOENCODING='utf-8'
npm ci
python tools/verify_gates.py
python tools/build_doc.py
python tools/verify_gates.py
python -m unittest discover -s tools -p "test_*.py" -v
```

为复核确定性，连续执行两轮 `python tools/build_doc.py`，每轮紧接：

```powershell
Get-FileHash -Algorithm SHA256 index.html, artifact.html
```

两轮的 `index.html` hash 必须相同，两轮的 `artifact.html` hash 也必须相同。

### 8.2 legacy 图资产边界

以下资产仅为历史审查/回溯而保留，均为 legacy、非权威且不参与当前构建：

- `tools/gen_diagrams.py`
- `diagrams/*.excalidraw`
- `build/svg/*.svg`

活动模板不得包含 `{{SVG:*}}`，`tools/build_doc.py` 不得 import/call `gen_diagrams`，也不得读取上述目录。
不要删除这些用户已有文件；若未来退役，应走单独的数据保留决策，而不是构建脚本顺手清理。

继续实现审查时重点检查：

1. 是否真的逐 rank 求值，而不是代表 rank 或母图；
2. source/event obligation 是否有独立分母和双向映射；
3. microbatch/recompute 是否使用 Tensor/Storage instance；
4. ResolvedEventSemantic/TensorStorageBinding 是否让序列化 RuntimePlan 独立于隐藏 CodeIR/object table；
5. forward/backward/saved tensor/grad accumulation 是否闭合；
6. effect 是否沿 base DAG 的同一稳定拓扑序定向，schedule constraint 是否成为真实 edge；
7. per-rank MemoryEventView 是否等于 ExpectedMemoryProjection 且完全不读 duration/start/end；
8. Storage/WorkspaceBinding 的 bytes/alignment/lifetime 是否唯一，memory/time 是否可独立 Ready/Blocked；
9. ExecutionDeployment、shared KernelVariantBinding 与 device-qualified StreamBinding 是否唯一且进入摘要；
10. PP/CP P2P/collective 是否经 endpoint descriptor quotient 与 typed internal-edge 规则进入联合 graph；
11. TimeEventView 是否自带 device/stage domain、protocol/numeric/bound cost，且没有 contention 隐式入口；
12. Ready EstimateContext 与 blocker propagation 是否有唯一 producer；
13. exact measurement/formula/route、split group 与摘要排除表是否闭合；
14. bundle 是否同时保留 Runtime/Core result，plan blocker 在 bind_core Blocked 后是否仍完整；shared build/G-IR blocker 是否强制双侧；两侧 projection blocker 是否全局并集并按 affected_backends 闭包，且闭包前未开始任一 backend seal；
15. comparison basis 是否从完整 CanonicalConfigEvaluationInput 构造，只 mask 已声明 axis 及跨字段确定性派生闭包；仅 `other_config_indexed_production_inputs.x` 不同是否必为 Incomparable 并定位该路径；
16. config_ref→EvaluationInstanceIdentity→BackendSealArtifact→ComparisonArmAuthority 是否全链可复算，A/B artifact 是否不可交换，同臂 artifact 是否共享字节相同 bundle/current subject/manifest；
17. GateClause 的 map key/record/manifest identity 是否三方相等，failure disposition 是否只从 manifest 派生；
18. 逐 metric comparison、basis pair、scenario axis、零 baseline、分层 cache 与 NotRequested 是否保持类型闭包；
19. Basic 是否只能生成不可发布的离线报告；AttestedConformanceReport 是否绑定批准 subject/fixture/policy/manifest/runner，GateClause 是否逐子句有正反例。

不要为本版重新引入编译器图优化、精确 layout/stride、多流完成时刻内存复用或资源竞争模型。
