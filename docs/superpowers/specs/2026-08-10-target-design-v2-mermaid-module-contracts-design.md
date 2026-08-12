# Target Design v2：Mermaid 图与模块契约说明改造设计

**日期：** 2026-08-10  
**状态：** 已在对话中确认方案 A  
**适用范围：** `pynative-cost-evaluator/docs/target-design-v2`  
**方案基线：** `src/index.template.html` SHA256 `51B005E126653B2F7D59E90C15C66F668EA69542AB8BF6BF5F18E7DFFD231AF0`；`HANDOFF.md` SHA256 `F198F7F227CB18BF121726E3B016BA0DDF807E242AA0BD8F83207F241A27694A`  
**不涉及：** 不读取或修改仿真器实现，不改变 v4.2 已冻结的产品边界、18 道结构门或 memory/time 数值语义。

## 1. 决策摘要

采用方案 A：在权威模板中维护 Mermaid 源，构建期使用固定版本 Mermaid CLI 生成静态 SVG，再把 SVG 内联到 `index.html` 与 `artifact.html`。读者端不加载 Mermaid、JavaScript、CDN、字体或图片资源；断网和禁用 JavaScript 时仍能看到全部图。

文档新增 10 张承重图和 9 组模块契约说明。图负责解释层次、关系和控制流；相邻 schema、公式、伪代码及结构门仍是规范真值。不会把每个叶子 schema 或每行算法强行图形化。

## 2. 当前问题与依据

1. 当前权威生产数据流是 32 行 ASCII 图，构图、运行时展开、双投影、门、seal 与 comparison 混在一块，见 `src/index.template.html:68-99`。
2. 模板、`index.html` 和 `artifact.html` 当前均没有 Mermaid、`<figure>` 或 `<svg>`；七份旧 Excalidraw/SVG 未被模板引用，而且包含旧版 `CoreIR/TrainIR` 等术语，不得恢复为当前图源。
3. 构建器已有 SVG 内联通道，但 caption 把来源硬编码为 `.excalidraw`，见 `tools/build_doc.py:121-126`。
4. `index.html` 构建会全局抽取所有 `<title>/<style>`，见 `tools/build_doc.py:201-206`。Mermaid SVG 自带同名子元素时会被误移入 HTML head，必须先修正。
5. 核心 schema 很完整，但模块入口和数据结构责任说明分散。例如 CodeIR schema 位于 `src/index.template.html:588-631`，RuntimeEventPlan 位于 `:911-1017`，双投影与 bundle 位于 `:1266-1807`，result/seal 位于 `:1953-2464`；读者难以从“字段列表”恢复模块的生产者、消费者与失败边界。

## 3. 目标与非目标

### 3.1 目标

- 所有承担架构、流程、生命周期、状态机或 authority 交接说明的可视化均由 Mermaid 源生成。
- 生成 HTML 完全离线、自包含、无读者端 Mermaid/JS 依赖。
- 每张图有稳定 ID、可访问标题/描述、图题和可搜索的 SVG 文本。
- 九个核心模块均有统一的结构/接口契约卡，能回答“做什么、输入什么、产出什么、谁消费、保持什么、怎样失败”。
- 补齐正文已经引用、但尚无正式 schema 或唯一接口签名的最小类型与辅助接口，不扩张产品能力。
- 明暗主题、窄屏、打印/PDF 均可读；相同输入连续构建产生规范化后一致的图。

### 3.2 非目标

- 不把精确公式、排除表、结构门谓词和算法伪代码改写成图。
- 不引入交互节点、缩放控件或客户端重新渲染。
- 不删除用户已有的 `diagrams/*.excalidraw`、`build/svg/*.svg` 或 `tools/gen_diagrams.py`；它们仅标记为 legacy，且不再参与当前构建。
- 不因补说明改变 CodeIR、RuntimeEventPlan、logical memory replay、no-contention DES、cost binding 或 blocker scope 的语义。

## 4. Mermaid 作者与构建架构

```mermaid
flowchart LR
    A[权威模板<br/>Mermaid 源 + 正文] --> B[build_doc.py]
    B --> C[render_mermaid.py]
    C --> D[@mermaid-js/mermaid-cli 11.16.0]
    D --> E[静态 SVG]
    E --> F[安全校验与 ID 前缀化]
    F --> G[index.html<br/>完整离线 HTML]
    F --> H[artifact.html<br/>宿主 fragment]
    I[src/style.css] --> B
    J[mermaid.config.json] --> C
```

### 4.1 图源格式

Mermaid 源直接放在 `src/index.template.html` 的图位置，使用稳定的自描述容器：

```html
<figure class="mermaid-figure" data-diagram-id="production-pipeline">
  <pre class="mermaid-source">flowchart TD
    request[RequestSnapshot] --> codeir[CodeIR]
    codeir --> runtime[RuntimeEventPlan]
  </pre>
  <figcaption>图题；相邻 schema/伪代码为规范</figcaption>
</figure>
```

模板是正文与图语义的唯一真值。生成 SVG 是可重建产物，不手工编辑；不另建一套会与正文漂移的 `.mmd` 业务图目录。

### 4.2 固定工具链

- [`@mermaid-js/mermaid-cli`](https://www.npmjs.com/package/%40mermaid-js/mermaid-cli) 固定为 `11.16.0`，Node 要求 `^18.19 || >=20.0`，通过目标目录内的 `package.json` 与 `package-lock.json` 固定完整依赖。
- `src/mermaid.config.json` 固定 `securityLevel=strict`、`htmlLabels=false`、确定性 ID seed、透明背景、字体栈、最大宽度和统一曲线。
- `tools/render_mermaid.py` 只调用本地 `node_modules/.bin/mmdc`；缺少依赖时给出 `npm ci` 的明确错误，不回退 CDN 或在线服务。
- `tools/build_doc.py` 提取每个 `mermaid-source`，以临时文件调用 renderer，校验并内联 SVG；构建结束不保留临时图。

### 4.3 SVG 闭包与安全

每个 SVG 必须满足：

- 有 `viewBox`、唯一 `<title>` 与 `<desc>`，并由 figure 的稳定 ID 关联；
- 文本保留为 `<text>/<tspan>`，不得全部转成 path；
- 所有内部 ID 加 `diagram-id` 前缀，页面级不重复；`url(#local-id)` 与 `href="#local-id"` 可解析；
- 不含 `<script>`、事件属性、`foreignObject`、外部 `href/src`、远程字体、`@import` 或 CSS `url()`；
- Mermaid 的内部 `<title>/<style>` 保留在 SVG 子树，构建器只抽取文档级 title/CSS；
- 图节点使用 `input/semantic/ir/plan/backend/blocker/note` 语义 class，由页面 CSS 变量适配明暗主题，不能只靠颜色传递含义。

## 5. 十张承重图

| ID | 位置与来源基线 | Mermaid 类型 | 说明责任 | 保留的规范文本 |
|---|---|---|---|---|
| `production-pipeline` | 1.1，替换 `src/index.template.html:68-99` | `flowchart TD` | 从请求、逐 rank CodeIR 到 runtime/core、双投影、backend seal 和 comparison 的主链；同时标出 validation 旁路不进入生产 | 相邻 1.1 解释段 |
| `facts-and-digests` | 2.4/3.1，依据 `:219-317`、`:332-417` | `flowchart LR` | 输入事实归属、model/runtime/core/backend digest 的单向依赖 | 全部 hash 公式和排除表 |
| `per-rank-codeir` | 4.2/4.3，依据 `:579-641` | `flowchart LR`，按 rank 分 subgraph | 每个 rank 独立求值、CodeIR containment、obligation 与 blocker 双向授权 | CodeIR schema、guard 规则 |
| `value-storage-identity` | 5.2/5.3，依据 `:689-771` | `classDiagram` | prototype、Tensor/Storage instance、binding、initial state 与 alias/view/in-place 关系 | 最小 alias 示例与 identity schema |
| `semantic-effect-closure` | 6.3–6.5，依据 `:816-906` | `stateDiagram-v2` | 注册、schema/effect 验证、binding/planning 以及任一步失败到 blocker；附 effect materialization 注释 | effect 算法伪代码 |
| `runtime-event-expansion` | 7/8.1–8.3，依据 `:911-1131` | `flowchart LR` | forward/backward/recompute/optimizer/microbatch 事件与 P2P/collective intent 的展开 | Event/Intent schema 与具体规则 |
| `core-dual-projection` | 8.5/9.1，依据 `:1143-1546` | `flowchart TD` | bind_core、shared variant、memory/time candidate、joint gate、scope closure 与三态结果 | binding schema、finalize 伪代码 |
| `memory-logical-replay` | 9.2/10.1，依据 `:1561-1644`、`:1825-1872` | `flowchart TD` | Runtime storage→bindings→MemoryEventView→logical replay→per-rank/cluster peak | ExpectedMemoryProjection 与 replay 算法 |
| `time-progress-des` | 9.3/10.3，依据 `:1646-1807`、`:1874-1949` | `flowchart TD` | endpoint quotient、stream/schedule edges、global progress DAG、cost validation、DES 与 aggregates | ExpectedTimeProjection、DES 公式 |
| `result-gate-comparison` | 10.4/10.5/12.2，依据 `:1953-2464`、`:2578-2778` | `sequenceDiagram` | source/value/seal authority、不可外泄 candidate、ledger prefix、可选双臂 comparison 与 conformance 发布旁路 | 全部 authority schema、seal 与 gate 等式 |

每图限制为约 8–12 个主节点。节点超过上限时优先使用 subgraph，不把两张语义不同的图硬拼成一张。

## 6. 九组模块契约说明

每组说明在对应详细 schema 前增加“模块契约”块，采用同一格式：

```text
模块责任：一句话定义权威边界。
核心数据结构：名称、用途、唯一生产者、消费者、关键不变量。
生产接口：完整签名、规范输入及 digest、返回 sum type。
失败方式：BlockerCode/scope 或 InternalContractViolation。
缓存与序列化：可进入的摘要层和禁止携带的输入。
```

新增接口使用以下显式结果类型，不以异常或空值代替规范分支：

```text
RequestSnapshotBuildResult :=
  Ready { request_snapshot: RequestSnapshot }
  | Blocked { blockers: NonEmpty[BlockerRecord] }

RankCodeIRBuildResult :=
  Ready { rank_ir: RankCodeIR }
  | Blocked { blockers: NonEmpty[BlockerRecord] }

BackendExecution<T, W> :=
  Completed { value: T, witness: W }
  | InternalViolation { violations: NonEmpty[InternalContractViolation] }
```

| 模块 | 核心结构 | 必须正式说明或补齐的接口/叶子类型 |
|---|---|---|
| 输入与事实 | `SourceSnapshot`、各 RegistrySnapshot、`CanonicalConfigEvaluationInput`、`RequestSnapshot` | `build_request_snapshot(CommonProductionInputs, OrderedMap<ConfigRef, CanonicalConfigEvaluationInput>, OrderedSet<Backend>, ComparisonRequest \| None) -> RequestSnapshotBuildResult`；Source/Structure/Runtime registry 的内容寻址 schema 与验证失败 |
| CodeIR | `CodeIR`、`RankCodeIR`、`SourceObligation`、`ResidualSite`、`CodeNode` | `TensorValue`、`LogicalStorage`、`DataEdge`、`ControlEdge`；`evaluate_source(SourceSnapshot, ModelSpec, normalized config P, LogicalRankContext, CodeInputShapeDtypeBindings, CompileEnvFacts, StructureRegistrySnapshot) -> RankCodeIRBuildResult` |
| Runtime | `RuntimeEventPlan`、`ExecEvent`、`ResolvedEventSemantic`、`AutogradLink`、`LogicalLifetime`、通信 intent | `expand_runtime_semantics(CodeIR, normalized config P, ExecutionScenario, RuntimeRegistrySnapshot) -> RuntimeBuildResult` 的前置、后置与 blocker 传播 |
| Memory | `StorageBinding`、`WorkspaceBinding`、`MemoryEventView`、`MemoryEstimate` | `MemoryTimelineEntry`、`MemoryExecutionWitness`；`build_memory_projection_candidate(RequestSnapshot, EvaluationInstanceIdentity, SimulationPlanCore, MemoryRegistrySnapshot) -> ProjectionCandidate<MemoryEventView>`；`run_memory_backend(MemoryEventView) -> BackendExecution<MemoryEstimate, MemoryExecutionWitness>` |
| Time | `StreamBinding`、`CostBinding`、`BoundComputeEvent`、`BoundCommunication`、`TimeEventView`、`StepTimeEstimate` | `BoundP2P`、`BoundCollective`、`RouteBinding`、`TimeTimelineEntry`、`TimeExecutionWitness`；`build_time_projection_candidate(RequestSnapshot, EvaluationInstanceIdentity, SimulationPlanCore, CalibrationSet, CommunicationModelSnapshot, TimeCostPolicy) -> ProjectionCandidate<TimeEventView>`；`run_time_backend(TimeEventView) -> BackendExecution<StepTimeEstimate, TimeExecutionWitness>` |
| Result | `EstimateCandidate`、source/value/seal authority、`BackendSealArtifact` | `run_backend_build_candidate_and_seal(BackendResultSourceAuthority) -> BackendSealArtifact \| NonEmpty[InternalContractViolation]`，作为现有 source→candidate→seal 三步的唯一组合入口 |
| Gate | `GateManifest`、`GateClause`、`GateExecutionLedger`、`BlockerRecord` | `BlockerScopePolicySnapshot`；`compile_gate_manifest(ProductionValidationContext, GateSpecificationSet, GateRunnerSnapshot) -> GateManifest`；`run_gate_domain(GateExecutionAuthority, OrderedSet<GateInvocationId>) -> GateExecutionLedger \| NonEmpty[InternalContractViolation]`；`extend_gate_ledger_without_overwrite(GateExecutionLedger, GateExecutionRecordSet, StageContextMap) -> GateExecutionLedger` |
| Comparison | `ComparisonBasisPair`、arms/source/candidate/seal artifact | `CanonicalSchemaPathSet`、`BasisMismatch`、`CoverageDelta`；`derive_comparison_basis_pair(RequestSnapshot, ComparisonArmAuthority, ComparisonArmAuthority, Metric) -> ComparisonBasisPair`；`compare_per_metric_from_authority(ComparisonSourceAuthority) -> ComparisonResult`；`build_comparison_source_authority(RequestSnapshot, ComparisonArmAuthority, ComparisonArmAuthority, OrderedMap<Metric, ComparisonBasisPair>, ProductionValidationContext, GateManifest) -> ComparisonSourceAuthority` |
| Conformance | `TraceFixture`、`IRConformance`、`ConformanceReport`、`ReleasePolicy` | `ConformanceFinding`、`ApprovedDigestSet`；`run_conformance(ProductionSubject, FixtureSet, ValidationPolicy, GateManifest, VerifierRunner) -> ConformanceReport`；`apply_release_policy(ConformanceReport, ApprovedDigestSet, ReleasePolicy) -> ReleaseDecision` |

这些补充只为已存在的引用提供唯一类型和入口，不增加在线 trace、资源竞争、隐式 layout 重排或新的仿真 face。

## 7. 页面表现规范

- figure 采用现有页面的细边框、无大圆角、等宽图题；图内字号不小于 12px。
- 总体/长流程用 `TD`，短链、fork 和 ownership 用 `LR`；状态生命周期使用 `stateDiagram-v2`；authority 交接使用 `sequenceDiagram`；containment/cardinality 使用 `classDiagram`。
- 语义色沿用当前 CSS：input 灰、semantic 蓝、IR 绿、plan 橙、backend 紫、blocker 红；每个节点同时显示文字类别或形状差异。
- 宽图在窄屏横向滚动，不整体压缩到不可读；打印时允许分页前保持 figure 完整。
- 每个 caption 都包含“解释图；相邻 schema/伪代码为规范”，避免图成为第二套规范。

## 8. 文件变更边界

### 新增

- `docs/target-design-v2/package.json`
- `docs/target-design-v2/package-lock.json`
- `docs/target-design-v2/src/mermaid.config.json`
- `docs/target-design-v2/tools/render_mermaid.py`
- Mermaid 构建与闭包测试文件（沿用现有 `tools/test_*.py` 组织）

### 修改

- `src/index.template.html`：Mermaid 源、十张图、九组模块契约与最小叶子 schema/接口说明。
- `src/style.css`：Mermaid figure、语义 class、明暗/窄屏/打印样式。
- `tools/build_doc.py`：构建期渲染、SVG 校验与安全内联；修复嵌套 title/style 抽取。
- `tools/test_build_doc.py`、`tools/verify_gates.py`、`tools/test_verify_gates.py`：新增图源、接口说明与资源闭包回归。
- `HANDOFF.md`：Mermaid 唯一有效图源、Node/npm bootstrap、模块接口索引、构建与验证命令。
- `index.html`、`artifact.html`：由模板重建。

### 保留但退出构建

- `diagrams/*.excalidraw`
- `build/svg/*.svg`
- `tools/gen_diagrams.py`

这些文件不得继续被模板或构建器引用，并在 HANDOFF 中标记为 legacy。

## 9. 验证设计

### 9.1 单元与结构测试

1. 每个模板 Mermaid figure 有唯一 `data-diagram-id`、非空 source、caption 和可访问标题/描述。
2. `index.html` 与 `artifact.html` 图数量、图 ID 和 SVG 语义文本一致。
3. SVG 内部 `<title>/<style>` 未被移到 HTML head；文档级 title/CSS 仍唯一。
4. 页面无外部 `src/href/xlink:href`、`@import`、CSS `url()`；SVG 无 script、事件属性或 `foreignObject`。
5. 所有 SVG/marker/clipPath ID 页面级唯一，内部引用全部可解析。
6. SVG 有 `viewBox`，关键标签是 DOM text，不是纯 path。
7. 删除任一 Mermaid 源、模块契约关键词或正式接口签名，verifier 必须失败。
8. 同一冻结输入连续构建两次，规范化后的 SVG/HTML hash 一致。
9. 现有 18 道结构门、163 项必需契约和 93 项禁用语义仍通过；如必需契约数增加，旧项不得减少。

### 9.2 视觉验证

- 1440px、768px、375px 三种宽度；light、dark、system 三种主题。
- 检查十张图的文字溢出、箭头遮挡、subgraph 边界、横向滚动和 caption。
- 打印/PDF 预览确认图不被裁切，黑白打印仍可凭标签和形状区分。
- 断网并禁用 JavaScript后直接打开 `index.html`；artifact 在最小宿主壳中打开，全部图仍可见。

## 10. 风险与处理

| 风险 | 处理 |
|---|---|
| Mermaid/Puppeteer 构建依赖较重 | 仅构建端安装；版本与 lockfile 固定；读者端零依赖 |
| Mermaid 输出 ID 或布局漂移 | 固定 11.16.0、配置、seed 和图 ID；规范化 SVG 后做双构建测试 |
| 明暗主题覆盖 Mermaid 内联样式困难 | `htmlLabels=false`，使用稳定语义 class 与 `!important` 的页面级最小覆盖；做浏览器截图验收 |
| 图与规范正文漂移 | 图源紧邻 schema；caption 明示非规范；verifier 检查关键节点标签和相邻接口名 |
| 补接口说明引入新语义 | 只定义已被正文调用的类型/入口；完成后重新做 IR、simulation、validation 三路 Critical/High 审查 |

## 11. 验收条件

1. 十张承重图全部由 Mermaid 11.16.0 构建，生成 HTML 不含 ASCII 架构/流程图或旧 Excalidraw/SVG 引用。
2. 九个模块均有结构与接口契约说明，列出的缺失叶子类型和辅助接口具有唯一正式定义。
3. `index.html`/`artifact.html` 离线、无 JS 可显示全部静态图，资源闭包和 SVG 安全测试通过。
4. 明暗、窄屏、打印视觉检查通过，无不可读文字、裁切、重叠或错误颜色依赖。
5. 原 v4.2 语义、18 门、blocker scope、partial result、digest/cache 与 comparison contract 无回归。
6. 不读取或修改仿真器实现；全部变更限于目标设计文档和文档构建/验证工具。
