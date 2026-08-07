# Target Design v2 P1：PyNative CodeIR 架构修订设计

**日期：** 2026-08-07  
**状态：** 已在对话中确认  
**适用范围：** `docs/target-design-v2` 的架构与验证契约  
**不涉及：** 本轮不修改 `cost_eval` 实现，也不回退已确认的 P0 工程估算器产品契约

## 1. 决策摘要

本轮按 PyNative 动态图的真实边界修订设计：给定源码快照、模型规格和当前归一化配置，直接求值得到该配置对应的、硬件无关且不可变的 `CodeIR(P)`。文档不再把策略单位元 `π₀` 生成的图作为权威结构，也不再设计编译器式融合、替换、图版本或 superseded 节点。

训练期派生行为不通过改写 `CodeIR` 表达，而由运行时语义展开生成 `RuntimeEventPlan`。反向、重计算、优化器、通信、microbatch 和 pipeline 实例都是执行事件；同一源码节点可以被多个事件引用，但源码结构始终只有一份。

硬件信息在 `CodeIR` 之后才参与绑定和成本求值。真实 trace 仅作为离线或 CI 的一致性测试 oracle，不是仿真器输入，不进入摘要、缓存键或数值预测路径。

## 2. 权威数据流

```text
SourceSnapshot + ModelSpec + normalized config P + CompileEnvFacts
                         ↓ source evaluation
                      CodeIR(P)
                         ↓ runtime semantic expansion
                  RuntimeEventPlan
                         ↓ hardware binding + cost evaluation
       HardwareProfile + CalibrationSet + Registry
                         ↓
                   SimulationPlan
                      ├─ MemoryEventView
                      └─ TimeEventView
```

测试链与生产链分离：

```text
Production:
Source + Config + Registry + HardwareProfile + CalibrationSet
    → memory/time estimates

Validation:
same Source + Config → predicted CodeIR / RuntimeEventPlan
real run             → TraceFixture
both                 → IRConformance / ScheduleConformance report
```

`TraceFixture` 缺失不得阻断生产预测，不得改变预测值、状态或覆盖率。CI 可以根据一致性测试结果阻止版本发布，但运行时仿真器不得查询 trace。

## 3. `CodeIR(P)`：按当前配置直接复刻源码

### 3.1 输入边界

`CodeIR(P)` 的合法输入只有：

- `SourceSnapshot`：参与求值的 Python 源码及可追溯文件摘要；
- `ModelSpec`：模型结构与训练配置；
- 当前归一化配置 `P`：TP、PP、EP、重计算、microbatch 等实际取值；
- `CompileEnvFacts`：框架/运行时版本、源码 feature flag、可静态确定的导入和配置事实。

`HardwareProfile`、设备容量、带宽、拓扑、allocator 常量、kernel 标定数据不得进入 `CodeIR(P)` 的构造或摘要。

### 3.2 源码结构规则

- 每个目标配置都从源码和该配置直接生成一张 `CodeIR(P)`，不从 `π₀` 图手工重建策略分支。
- 源码当前配置实际选择的分支进入图；无法静态决定且会改变结构的 guard 形成 residual/blocked 诊断，不得猜测另一条分支。
- 源码显式写成一个 fused operator 时，`CodeIR` 中保留一个对应节点。
- 源码写成多个普通 operator 时，`CodeIR` 中保留多个节点；不得由设计侧增加 fusion pass。
- kernel 内部实现、workspace 或性能特征属于成本证据，不得反向改变源码节点结构。
- `CodeIR(P)` 创建后不可变。设计中不引入 `GraphVersion`、`RewriteMap`、active/superseded/replacement 状态。

### 3.3 身份与摘要

`CodeIR` 至少稳定区分：

- `code_node_id`：源码可见的调用/语义节点；
- `tensor_id`：逻辑张量值；
- `storage_id`：逻辑存储对象及其 alias 关系；
- 数据依赖、控制依赖、shape、dtype 和源码位置。

改变 `HardwareProfile` 必须保持 `CodeIR` 内容和 `model_digest` 不变；改变源码或影响结构的当前配置必须改变摘要。

## 4. `RuntimeEventPlan`：运行时语义展开，不是图改写

`RuntimeEventPlan` 把不可变源码结构展开为一次训练 step 中实际需要仿真的事件实例。它可以包含：

- forward 与 autograd 派生的 backward；
- optimizer/state update；
- rematerialization/recompute；
- collective 与 point-to-point communication；
- microbatch、pipeline stage、rank/device 上的执行实例；
- alloc/free、workspace、同步和依赖事件。

每个 `ExecEvent` 必须保留 `event_id`、`code_node_id`（没有直接源码节点时保留派生来源）、phase、microbatch、rank/device、依赖和语义来源。重计算表示新的 `ExecEvent` 再次引用原 `code_node_id`，不得复制、替换或修改 `CodeIR` 节点。

运行时语义展开只能实例化框架训练语义和显式并行语义，不能发明编译器优化。硬件 stream、资源域和代价在后续 `SimulationPlan` 中绑定。

## 5. 显式 Effect 模型

每个算子语义必须提供 `EffectSummary`：

```text
EffectSummary:
  reads
  writes
  mutates
  rng
  collective
  host_side_effect
  replayable
  deterministic
  reentrant
```

重计算、重复执行、事件重排和并发合法性必须由 effect 约束，而不是仅由 shape 或数据边推断：

- RNG/Dropout 需要明确状态保存或可复现规则；
- mutation、状态更新和 host side effect 默认不可自由重放；
- collective 必须保留参与集合、顺序域和 correlation identity；
- 不可重入 kernel 不得生成重叠执行；
- effect 缺失或冲突时，受影响事件进入 blocked/unknown 诊断，不得以默认纯函数继续。

融合不在本设计的变换集合内，因此 effect 不用于证明融合合法性。

## 6. 硬件边界

环境事实拆分为：

```text
CompileEnvFacts:
  framework/runtime/version
  source feature flags
  static import/config facts

HardwareProfile:
  devices/capacity
  topology/link bandwidth
  peak throughput
  allocator/alignment constants
  resource-domain definitions
```

`CompileEnvFacts` 可以参与源码求值，`HardwareProfile` 只能进入硬件绑定、`SimulationPlan` 和内存/时间后端。`CodeIR`、源码语义注册表与其 digest 不得携带具体设备、容量、带宽或拓扑字段。

## 7. 逐表达式证据与 epistemics

“Native”只表示所有权或来源，不能把整个算子的所有语义面自动提升为 exact。所有会影响结构、内存或时间的表达式使用统一包装：

```text
SemanticValue<T>:
  value: T | unknown
  quality: exact | derived | calibrated | estimated | unknown
  evidence_ref
  assumptions
  coverage_tags
```

`shape`、`dtype`、`storage/alias`、`autograd`、`placement`、`resource/cost`、`workspace` 和 `effects` 分别携带自己的 `SemanticValue`。一个算子可以同时具有 exact shape、derived storage、calibrated duration 和 unknown effect；报告必须保留这种差异，不能压成单一 `self_certainty`。

## 8. Source-to-DAG 一致性验证

Stage0 的 AST 覆盖率只证明语法构件可识别，不能证明部分求值和调用解析正确。新增 `IRConformance` 测试链，对代表性 Dense、MoE、PP、重计算、分布式优化器和显式 fused-op 配置进行源码到真实执行的一致性验证。

`CodeIR` 层至少比较：

- operator identity、数量与稳定顺序；
- 数据依赖与关键控制依赖；
- shape、dtype、配置分支和源码位置映射；
- 预测中缺失、多余或无法解析的节点及覆盖率。

`RuntimeEventPlan` 层可进一步比较：

- backward/recompute/optimizer 事件数量；
- collective 参与者、相关标识和顺序；
- rank/device、stream、开始/结束时间与依赖；
- microbatch/pipeline 实例和 makespan 误差。

默认 fixture 应通过标准 profiler、memory tracker 和独立 runner/launch 配置采集。修改模型业务源码或框架不是必需路径；可选 marker/hook 只能用于诊断，且必须进行 A/B 扰动检查。

## 9. Trace 与 CalibrationSet 的严格分界

原始 `TraceFixture`：

- 只存在于测试/验证目录和 conformance runner；
- 不属于生产 API、`SpecBundle` 或仿真请求；
- 不进入 `model_digest`、缓存键、coverage 或 confidence 计算；
- 缺失时不改变任何生产输出；
- 可由 CI 用于发现设计或实现偏差。

`CalibrationSet` 是可选、版本化且可审计的成本模型数据，例如算子 duration 或 workspace 标定。它可以作为生产估算输入，但必须记录来源、硬件适用范围和留出集误差。不得把某次待预测运行的 raw trace 包装为 `CalibrationSet`，从而形成隐式真机依赖或数据泄漏。

## 10. 与 P0 契约的关系

本修订只改变结构生成、训练事件表达、硬件边界、证据粒度和验证方式。以下 P0 决策保持不变：

- 产品是允许误差的工程估算器，不回答数学意义上的“是否装得下”；
- 内存后端按 tensor/storage 的 alloc/free 生命周期计算 logical allocated peak；
- 不输出 reserved、碎片、allocator OOM 或 soundness verdict；
- 时间后端根据依赖正确的 event DAG 做 DES，输出预测 step time；
- 多配置结果可以比较，但必须同时报告误差、覆盖率、假设和适用域。

## 11. 文档修改范围

本轮只修改：

- `docs/target-design-v2/src/index.template.html`：权威设计正文；
- `docs/target-design-v2/HANDOFF.md`：实现交接摘要；
- `docs/target-design-v2/tools/verify_gates.py`：结构和禁用语义门禁；
- 必要的 verifier 单元测试；
- 由模板重建的 `index.html` 与 `artifact.html`。

不修改 `cost_eval`、Stage0 证据、既有真机 trace，也不伪造尚未执行的一致性结果。

## 12. 验收条件

修订后的权威模板必须同时满足：

1. 明确出现 `CodeIR(P)`、`RuntimeEventPlan`、`SimulationPlan`、`EffectSummary`、`CompileEnvFacts`、`HardwareProfile`、`SemanticValue`、`TraceFixture` 和 `IRConformance`。
2. 明确声明当前配置直接从源码生成 IR，且 CodeIR 硬件无关、不可变。
3. 明确声明无 compiler fusion、无通用 graph rewrite、无 `GraphVersion`/replacement/superseded 机制。
4. 明确声明重计算是事件实例复用 `code_node_id`，不是复制源码图。
5. 明确声明 raw trace 仅用于测试，不进入生产输入、摘要或缓存键，缺失不改变预测。
6. 18 道结构门覆盖 CodeIR、event/effect/evidence、内存、时间和报告/验证边界。
7. 重建 HTML 后通过 verifier、工具测试、全仓回归和浏览器视觉检查。

## 13. 非目标

- 不设计静态图编译器或图优化框架；
- 不引入算子融合、消除、替换或版本化图改写；
- 不让 trace 成为在线预测依赖；
- 不承诺现阶段已经达到真实框架的全部 DAG、调度或数值精度；
- 不在本轮实现新的采集器或修改模型业务代码。
