# Target Design v2 Mermaid 与模块契约 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `target-design-v2` 的承重架构图和流程图统一为构建期渲染的 Mermaid 静态 SVG，并为九个核心模块补齐可实现、可验证的数据结构与接口契约说明。

**Architecture:** 权威内容仍只维护在 `src/index.template.html`。模板内保存 Mermaid 源码，`tools/build_doc.py` 调用新的 `tools/render_mermaid.py`，通过锁定版本的 Mermaid CLI 在构建期渲染、校验、消毒并以内联 SVG 写入 `index.html` 与 `artifact.html`。模块契约以结构化 HTML section 表达，同时由 `verify_gates.py` 和单元测试检查类型、接口、阻断分支与跨模块闭包。

**Tech Stack:** Python 3 `unittest`、HTML/CSS、Mermaid CLI 11.16.0、Node.js/npm、XML/SVG 校验。

## Global Constraints

- 只修改 `pynative-cost-evaluator/docs/target-design-v2`、对应设计说明与本实施计划；不读取或修改仿真器实现代码。
- `src/index.template.html` 是内容权威；`index.html` 与 `artifact.html` 只能由构建生成，不手工修补。
- 不使用 CDN、浏览器端 Mermaid 或外部图像资源；最终两个 HTML 必须离线、自包含、零网络依赖。
- Mermaid 图必须包含 `viewBox`、可访问的 `title`/`desc`、原生 SVG 文本，并通过脚本、事件属性、`foreignObject`、外链、CSS `url()`/`@import`、悬空 ID 引用检查。
- `diagrams/*.excalidraw`、`build/svg/*.svg` 与 `tools/gen_diagrams.py` 保留为 legacy，不删除，但退出活动构建路径。
- 目标目录当前完全未被 Git 跟踪，不执行 `git add` 或提交；每个任务以 SHA-256 快照、测试输出和计划勾选项作为检查点。
- 测试驱动：每项功能先添加能证明缺口的失败测试，观察 RED，再实现最小闭合改动并观察 GREEN。
- 同一时刻只允许一个实现者编辑共享权威模板；其他代理只能只读审查，避免共享工作区覆盖。
- 禁止未决标记、占位接口或无定义名词进入最终文档。

---

## Task 1: 建立 Mermaid 离线渲染与安全校验工具链

**Files:**

- Create: `pynative-cost-evaluator/docs/target-design-v2/package.json`
- Create: `pynative-cost-evaluator/docs/target-design-v2/package-lock.json`
- Create: `pynative-cost-evaluator/docs/target-design-v2/src/mermaid.config.json`
- Create: `pynative-cost-evaluator/docs/target-design-v2/tools/render_mermaid.py`
- Create: `pynative-cost-evaluator/docs/target-design-v2/tools/test_mermaid_render.py`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/build_doc.py`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/test_build_doc.py`

- [x] 记录 `src/index.template.html`、`HANDOFF.md`、`tools/build_doc.py` 和当前生成文件的 SHA-256。
- [x] 在 `test_mermaid_render.py` 先写失败测试，覆盖：提取 `<figure class="mermaid-figure" data-diagram-id>`；相同输入产生字节稳定输出；每图 ID 前缀唯一；所有 `url(#id)`/`href="#id"` 引用闭合；拒绝 `script`、`foreignObject`、`on*`、外部 `href`、CSS `url(http...)` 与 `@import`；输出必须有 `viewBox/title/desc` 和原生 `text/tspan`。
- [x] 在 `test_build_doc.py` 增加回归测试：内联 SVG 中的 `<title>` 与 `<style>` 不得被文档级 head 提取逻辑搬走；构建产物不得残留 Mermaid 源码或 `{{SVG:*}}`。
- [x] 运行 `python -m unittest tools.test_mermaid_render tools.test_build_doc -v`，确认新增测试因模块/行为缺失而失败。
- [x] 添加锁定依赖 `@mermaid-js/mermaid-cli@11.16.0`，生成并保留 lockfile；`package.json` 只暴露确定性 render/verify 脚本，不引入前端运行时依赖。
- [x] 配置 `securityLevel: strict`、`htmlLabels: false`、确定性 ID/seed、透明背景及中性主题变量。
- [x] 实现 `render_mermaid.py`：
  - `extract_mermaid_figures(template_html) -> list[MermaidFigureSource]`
  - `render_mermaid_source(source, diagram_id, config_path, output_dir) -> str`
  - `sanitize_and_prefix_svg(svg_text, diagram_id, title, description) -> str`
  - `validate_inline_svg(svg_text, diagram_id) -> None`
  - 使用临时目录和参数数组调用本地 `node_modules/.bin/mmdc`，不拼接 shell 字符串。
  - XML 级改写全部 element ID 与本地引用，移除生成器元数据，规范属性顺序与换行，保证重复构建字节稳定。
- [x] 重构 `build_doc.py`：只从模板文档 head 提取顶层 `<title>/<style>`；随后解析并替换 Mermaid figure，避免扫描内联 SVG 子树；移除 `gen_diagrams` 的活动 import 和 `{{SVG:*}}` 构建分支。
- [x] 运行目标测试直至 GREEN，再运行现有 9 个基线测试和 `verify_gates.py`，确保没有旧契约回归。
- [x] 双次构建并比较 `index.html`、`artifact.html` SHA-256；若不一致，定位并消除 Mermaid CLI 的随机 ID、时间戳或序列化差异。
- [x] 记录阶段 SHA 与实际命令输出。

## Task 2: 用 Mermaid 重绘十张承重架构图和流程图

**Files:**

- Modify: `pynative-cost-evaluator/docs/target-design-v2/src/index.template.html`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/src/style.css`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/test_build_doc.py`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/verify_gates.py`

- [x] 先增加失败测试，要求模板与两个生成产物都精确包含以下十个唯一 diagram ID：`production-pipeline`、`facts-and-digests`、`per-rank-codeir`、`value-storage-identity`、`semantic-effect-closure`、`runtime-event-expansion`、`core-dual-projection`、`memory-logical-replay`、`time-progress-des`、`result-gate-comparison`。
- [x] 测试每图在两个产物中各出现一次、具有唯一 SVG ID 命名空间、`title/desc/figcaption`、可见文本节点和闭合本地引用；确认测试先 RED。
- [x] 将总生产流程的 ASCII/pre 图替换为 `production-pipeline` Mermaid，明确 source/config/facts → per-rank CodeIR → RuntimeEventPlan → SimulationPlanCore → memory/time 双投影 → gates/seal → comparison/conformance；validation 为旁路，不进入生产摘要。
- [x] 添加 `facts-and-digests`，显示事实域、派生对象、各层 digest 与缓存键的单向依赖，并标出不可把 TraceFixture/请求时 profiler 混入生产链。
- [x] 添加 `per-rank-codeir`，用 rank 子图表达 MindFormers/Trainer 配置驱动的 block 分配、逐 rank source evaluation、obligation disposition 与 rank IR 集合。
- [x] 添加 `value-storage-identity` class diagram，区分 prototype、microbatch/repeat instance、runtime-only value/storage、tensor-storage binding 与 initial live-in。
- [x] 添加 `semantic-effect-closure` state/flow diagram，表达 semantic lookup、shape/effect/autograd closure、unknown/ambiguous blocker 和 effect-domain precedence。
- [x] 添加 `runtime-event-expansion`，表达 forward/backward/recompute/optimizer、不同 microbatch、P2P send/recv 与 collective intent 的规范展开。
- [x] 添加 `core-dual-projection`，表达共享 core、硬件绑定、memory/time candidate、联合 blocker scope reconciliation 和独立 Ready/Blocked/NotRequested。
- [x] 添加 `memory-logical-replay`，只按 canonical logical kernel order 回放 Allocate/Bind/Use/Free，明确不读取完成时刻、不建模跨流复用竞争。
- [x] 添加 `time-progress-des`，表达 compute 精确用户测量、communication 理论公式、P2P/collective quotient、stream/schedule edges、无竞争 DES 与聚合结果。
- [x] 添加 `result-gate-comparison` sequence diagram，覆盖 source authority、逐 clause ledger、backend witness/value gates、seal、按 metric comparison 与 conformance。
- [x] 在 CSS 添加语义类 `input/semantic/ir/plan/backend/blocker/note` 的浅色/深色变量、图框、窄屏横向滚动和打印规则；不只依赖颜色表达状态。
- [x] 运行图形结构测试、全量构建、verifier 与全部单测直至 GREEN；抽查每张 SVG 中关键节点文本与 authority 正文术语一致。
- [x] 记录阶段 SHA 与图 ID/尺寸清单。

## Task 3: 补齐输入、事实、CodeIR 与 Runtime 模块契约

**Files:**

- Modify: `pynative-cost-evaluator/docs/target-design-v2/src/index.template.html`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/test_verify_gates.py`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/verify_gates.py`

- [x] 先写结构测试，要求 `section.module-contract[data-module]` 中存在 `input-facts`、`code-ir`、`runtime-events`，且各自含“职责边界、核心数据结构、接口定义、成功/阻断语义、不变量”五类子节。
- [x] 让 verifier 要求下列新增类型/接口文字与签名；运行并观察 RED。
- [x] 输入与事实模块补充：`SourceSnapshot`、registry snapshots、`CanonicalConfigEvaluationInput`、`RequestSnapshot`、`RequestSnapshotBuildResult`；定义 `build_request_snapshot(...) -> RequestSnapshotBuildResult`，并说明 normalized P、ExecutionScenario、request backend set、fact snapshot 与 digest 的唯一生产者。
- [x] CodeIR 模块补充叶子结构：`TensorValue`、`LogicalStorage`、`DataEdge`、`ControlEdge`、source obligations/dispositions、rank-local stable identity；定义：

```text
evaluate_source(
  source: SourceSnapshot,
  model: ModelSpec,
  config: NormalizedParallelConfig,
  rank: LogicalRankContext,
  bindings: CodeInputShapeDtypeBindings,
  env: CompileEnvFacts,
  registry: StructureRegistrySnapshot
) -> RankCodeIRBuildResult
```

- [x] 明确 `RankCodeIRBuildResult := Ready{rank_ir} | Blocked{NonEmpty[BlockerRecord]}`，逐 rank 构建可用 Trainer 配置分配规律，但每个候选调用与 guard obligation 必须有规范 disposition；CodeIR 不读取硬件或执行时间。
- [x] Runtime 模块补充 `RuntimeRuleInvocationRef`、`ExecEvent`、`ResolvedEventSemantic`、`TensorInstance`、`StorageInstance`、`TensorStorageBinding`、`LogicalInitialState`、`P2PIntent`、`CollectiveIntent` 与 autograd link；定义：

```text
expand_runtime_semantics(
  code_ir: CodeIR,
  config: NormalizedParallelConfig,
  scenario: ExecutionScenario,
  registry: RuntimeRegistrySnapshot
) -> RuntimeBuildResult
```

- [x] 说明 microbatch/repeat identity、正反向双向闭合、P2P/collective 一一匹配、effect/schedule precedence、runtime blocker 的 backend scope。
- [x] 运行结构测试、verifier 与全量单测直至 GREEN，并检查三个模块的术语只引用已定义类型。
- [x] 记录阶段 SHA。

## Task 4: 补齐 Memory、Time 与 Result 模块契约

**Files:**

- Modify: `pynative-cost-evaluator/docs/target-design-v2/src/index.template.html`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/test_verify_gates.py`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/verify_gates.py`

- [x] 先增加 `memory-backend`、`time-backend`、`result-sealing` 三个 module-contract 的失败测试。
- [x] Memory 模块补充 `MemoryProjectionCandidate`、`MemoryEventView`、`MemoryTimelineEntry`、`MemoryExecutionWitness`、`MemoryEstimate` 与 `WorkspaceBinding`；定义：

```text
build_memory_projection_candidate(
  request: RequestSnapshot,
  evaluation_identity: EvaluationInstanceIdentity,
  core: SimulationPlanCore,
  memory_registry: MemoryRegistrySnapshot
) -> ProjectionCandidate<MemoryEventView>

run_memory_backend(view: MemoryEventView)
  -> BackendExecution<MemoryEstimate, MemoryExecutionWitness>
```

- [x] 明确 logical order replay、initial-state anchor、每 storage 唯一 release rule、rank map/cluster max、breakdown 守恒和本轮不建模跨流 allocator 竞争。
- [x] Time 模块补充 `BoundP2P`、`BoundCollective`、`RouteBinding`、`CostBinding`、`PhysicalStreamId`、`TimeTimelineEntry`、`TimeExecutionWitness`、`StepTimeEstimate`；定义：

```text
build_time_projection_candidate(
  request: RequestSnapshot,
  evaluation_identity: EvaluationInstanceIdentity,
  core: SimulationPlanCore,
  calibration: CalibrationSet,
  communication: CommunicationModelSnapshot,
  policy: TimeCostPolicy
) -> ProjectionCandidate<TimeEventView>

run_time_backend(view: TimeEventView)
  -> BackendExecution<StepTimeEstimate, TimeExecutionWitness>
```

- [x] 明确 compute 仅 exact user measurement record 命中，communication 由冻结理论公式求值；P2P/collective 语义、stream/schedule edge、step-relative zero、无资源竞争 DES、timeline 和 breakdown 公式闭合。
- [x] Result 模块补充 `BackendExecution<T,W>`、`EstimateCandidate`、`BackendResultCandidate`、`BackendSealArtifact`、`InternalContractViolation`；定义：

```text
run_backend_build_candidate_and_seal<V: BackendReadyView>(
  source: BackendResultSourceAuthority<V>
) -> BackendSealBuildResult<EstimateOf<V>, V>
```

- [x] 说明 Ready/Blocked/NotRequested provenance、value subject digest、逐 clause gate records、result seal、外层 estimate context/result digest 以及 contract violation 不伪装成输入 blocker。
- [x] 运行结构测试、verifier 与全量单测直至 GREEN。
- [x] 记录阶段 SHA。

## Task 5: 补齐 Gate、Comparison 与 Conformance 模块契约

**Files:**

- Modify: `pynative-cost-evaluator/docs/target-design-v2/src/index.template.html`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/test_module_contracts.py`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/test_verify_gates.py`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/verify_gates.py`

- [x] 先增加 `gate-system`、`comparison`、`conformance` 三个 module-contract 的失败测试，并增加 scope-policy、invocation-domain、schema closure、fixture 隔离、ICV aggregate 与 conformance seal 的 section-local 反例。
- [x] Gate 模块补充 content-addressed `BlockerScopePolicySnapshot` / `ScopePolicyRef`、`GateManifest`、`GateInvocationId`、`GateClauseExecutionRecord`、`GateExecutionLedger`、`GateEvaluationAuthority` 与 `ExpectedInvocationDomain`；定义：

```text
compile_gate_manifest(context: ProductionValidationContext, specifications: GateSpecificationSet, runner: GateRunnerSnapshot) -> GateManifest
build_gate_evaluation_authority(subject: GateEvaluationSubject, manifest: GateManifest, runner: GateRunnerSnapshot) -> GateEvaluationAuthority | InternalContractViolation
expected_invocation_domain(authority: GateEvaluationAuthority, base: GateExecutionLedger) -> ExpectedInvocationDomain
run_gate_domain(authority: GateEvaluationAuthority, base: GateExecutionLedger, invocation_domain: OrderedSet<GateInvocationId>) -> GateExecutionLedger | InternalContractViolation
extend_gate_ledger_without_overwrite(authority: GateEvaluationAuthority, base: GateExecutionLedger, extension: GateExecutionRecordSet, current_stage_contexts: StageContextMap) -> GateExecutionLedger | InternalContractViolation
```

- [x] 明确 request-owned blocker policy 与 specification/manifest/ProjectionBundleAuthority/GateEvaluationAuthority payload+digest 逐字闭合；GateEvaluationSubject 是具体 Memory/Time 泛型闭合的 tagged union，`gate_evaluation_context_digest(subject)` 是唯一 context 真值；expected domain 将 covered-stage 全部规范 backend/branch sibling 与 dependency closure 纳入 `audit_invocations`，typed coordinates 只派生 `applicable_invocations`，负例使用独立 literal reference derivation。manifest clause key 与 execution record clause_id、failure disposition 逐 clause 相等，InputBlocker 全量规范并集、单一 canonical InternalViolation 优先终止；GateClause 仅持无 fixture ID 的 `ClauseCoverageRequirement`。
- [x] Comparison 模块补充 request-owned `ComparisonSchemaSnapshot`、绑定 snapshot digest 且 closure 只可派生的 `CanonicalSchemaPathSet`、`ComparisonBasisPair`、`BasisMismatch`、`CoverageDelta`、`ComparisonSourceAuthority`、`ComparisonResultCandidate`；定义：

```text
derive_comparison_basis_pair(request: RequestSnapshot, left: ComparisonArmAuthority, right: ComparisonArmAuthority, metric: memory | time) -> ComparisonBasisPair
build_comparison_source_authority(request: RequestSnapshot, left: ComparisonArmAuthority, right: ComparisonArmAuthority, basis_pairs: OrderedMap<memory | time, ComparisonBasisPair>, context: ProductionValidationContext, manifest: GateManifest) -> ComparisonSourceAuthority
compare_per_metric_from_authority(source: ComparisonSourceAuthority) -> ComparisonResult
```

- [x] 明确 A/B config identity、左右 basis、同 metric 可比性、零基线 undefined、registry/calibration/fallback/assumption 差异不伪装成策略收益；basis pair 与 G-REP2 均从当前 RequestSnapshot 重算 schema closure，禁止调用方注入路径。
- [x] Conformance 模块补充 `FixtureBinding`、`ConformanceInvocationAuthority`、`ConformanceFinding`、`ConformanceObservedOutput`、签名 `RunnerAttestation`、`ConformanceExecutionLedger`、`ConformanceReport`、`ConformanceSealArtifact`、`ConformanceRunResult`、`ReleaseApprovalArtifact`、`ReleaseDecision`；定义：

```text
run_conformance(authority: ConformanceInvocationAuthority) -> ConformanceRunResult
derive_conformance_report(authority: ConformanceInvocationAuthority, observed: ConformanceObservedOutput, ledger: ConformanceExecutionLedger, attestation: RunnerAttestation) -> ConformanceReport
validate_and_seal_conformance(authority: ConformanceInvocationAuthority, observed: ConformanceObservedOutput, ledger: ConformanceExecutionLedger, attestation: RunnerAttestation) -> ConformanceRunResult
apply_release_policy(artifact: ConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy) -> ReleaseDecision
```

- [x] 说明 FixtureSet 是唯一 fixture-to-(invocation, clause) 映射且完全位于生产摘要/cache 链外；record input 与 canonical observed output 从 invocation authority 复算，runner attestation 和 approval store artifact 均验签；report findings/verdict 从 observed output 与版本化 policy 唯一派生，approval 始终使用 policy key+scheme 验签；subject/fixture/policy/manifest/runner/attestation/output/report/ledger/seal digest 闭合，无执行伪造 pass、伪造 attestation/ApprovedDigestSet、record/output/verdict 篡改均失败；insufficient 仅表示完整合法执行后的证据不足。所有 gate failure 统一返回单一 canonical InternalViolation，并有全模板 forbidden 回归。
- [x] 运行结构测试、verifier 与全量单测直至 GREEN：Task 5 目标测试 54/54；全量 82/82；standalone verifier 204 required / 105 forbidden / 18 gates。
- [x] 记录阶段 SHA：`index.template.html` `31590DA8BF3EFE731CFB97D5FC8921F1D723574CD51F005753CC6DB167F1C85D`；`test_module_contracts.py` `5FE1D18BFEDE4A5CD565D92BD7FCC4B190C59A283B0D5024BB3F55CC5A3F2E52`；`test_verify_gates.py` `A92E49DA1347168E7B7B0419AFA4A6ADF5CEF8C52E4A01DC4C6FA41BDC76EF2C`；`verify_gates.py` `D07CE0B94160E106CD809B180135B343C3314AE0E195B8BA6FE3C347F01E3B46`。

### Task 5 review fix round 4（2026-08-10）

- [x] Finding 1（Critical）按严格 TDD 将跨 stage domain 拆成 `required_prefix_invocations` 与 `new_execution_invocations`：RED 7 tests / 35 failures / 0 errors；GREEN 7/7。`base` ledger 显式进入 expected/run/extend，dependency closure 只验证不可覆盖前缀中的 Pass/NotApplicable，current root stage 才执行完整 backend/branch siblings；base records/context byte-preserved。memory ProposedOk fixture 固定 G-MEM5/6 的 V，仅 result-seal 新执行使用 R，未选/未请求 siblings 保持 NotApplicable。
- [x] Finding 2（Critical）补齐 Task 5 own-digest exclusion table 与 verifier 精确逐 owner/own-field/唯一行检查，明确 `VerifierRunner` 公式且不递归删除 nested digests：首轮 RED 3 tests / 115 failures / 0 errors，重复/过宽补强 RED 1 test / 30 failures / 0 errors；GREEN 4/4。新增类型 `TrustStoreSnapshot`、`RunnerAttestationPolicy` 同步纳入精确表，自引用、漏项、重复/过宽行均拒绝。
- [x] Finding 3（High）闭合 gate compile provenance：RED 5 tests / 24 failures / 0 errors；GREEN 5/5；final self-review 对 runtime subject/context 复算补强 RED 2 tests / 6 failures / 0 errors，GREEN 2/2。request/common production inputs 持有可信 `GateSpecificationSet` payload/ref，compile 与 runtime builder 复算 subject/context/specification/policy/runner/manifest 两侧 digest，且 `manifest.entries == specifications.entries`；manifest 自报不能成为权威。
- [x] Finding 4（High）将 `ConformanceReport.findings` 与 observed output 统一为 `OrderedMap<FindingId, ConformanceFinding>`：RED 3 tests / 3 failures / 0 errors；GREEN 3/3。
- [x] Finding 5（High）定义并统一引用 `derive_approved_digest_set(artifact, policy)`，精确派生当前 `ApprovedDigestSet` 全部十个字段：RED 5 tests / 21 failures / 0 errors；GREEN 5/5。模块正文、`ReleasePolicy` 与 12.6 不再使用四/六字段子集。
- [x] Finding 6（High）新增版本化 authority-owned `RunnerAttestationPolicy` / `TrustStoreSnapshot`，固定 trust registry/key material/supported schemes；7 元 signed message 覆盖 environment/executable/key/scheme/invocation/observed/runner，validator 逐字绑定重复字段并仅用 policy-selected scheme 与 store key material 验签：RED 5 tests / 37 failures / 0 errors；GREEN 5/5。wrong store、unsupported scheme、tampered environment、runner self-owned key 四类负例闭合。
- [x] Round 4 fresh verification：Task 5 目标测试 54/54；全量 82/82；standalone verifier 204 required / 105 forbidden / 18 gates。
- [x] Round 4 final SHA：`index.template.html` `E72CC0BDE389A4A8DEC123D07F615A41C1F57016D4BD8889E7EA0A3B984AB371`；`test_module_contracts.py` `536D223DB00A8359DE02BFDF876CB564813B61FFE57F58A41529CEA920DA068B`；`test_verify_gates.py` `D80FB467AFFCF3ABDFBF992CCB2006A71AACFC436B597BD0B5ADDD571C6959F3`；`verify_gates.py` `234AEA30E872F200C4CCDD0D9295EDB9F50D1A44A2E02860C271DED33CE23251`。

## Task 6: 同步 HANDOFF、清理活动构建路径并做最终验收

**Files:**

- Modify: `pynative-cost-evaluator/docs/target-design-v2/HANDOFF.md`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/build_doc.py`
- Modify: `pynative-cost-evaluator/docs/target-design-v2/tools/test_build_doc.py`
- Regenerate: `pynative-cost-evaluator/docs/target-design-v2/index.html`
- Regenerate: `pynative-cost-evaluator/docs/target-design-v2/artifact.html`

- [x] 将 HANDOFF 中重复的 ASCII 架构流程改为指向权威 Mermaid 图的精确说明；同步新增模块契约、工具链、10 个 diagram ID、legacy 资产边界和构建命令。
- [x] 添加测试确认活动构建脚本不 import/call `gen_diagrams`，模板不再含 `{{SVG:*}}`，但 legacy 文件仍存在且 HANDOFF 明确其非权威状态。
- [x] 从干净依赖状态运行 `npm ci`，再运行文档构建、合同 verifier、全部单元测试；最终为 99/99 tests、212 required / 108 forbidden / 18/18 gates。
- [x] 连续构建两次并验证 `index.html`/`artifact.html` 字节哈希一致：分别为 `D88AAFE9B87191071035B8F1C1154025736EA977EB811EDC1B7C6A28F88DEC4B` 与 `922182AE54979F5E734D75DDDD4E5BF083EB9C77127B87E62B9C4ED959E85C16`。
- [x] 对两个产物运行静态审计：10 图唯一、SVG 引用闭合、无 SVG script/事件属性/foreignObject/外部网络、9 个 module-contract 完整、无未决标记、无未解析模板标记。
- [ ] 使用本地静态服务器和浏览器检查宽屏、窄屏、深色、打印预览：本地服务器已启动，但 in-app Browser 的管理员策略校验服务持续不可用；遵守浏览器安全策略未绕过。Mermaid 真实渲染、可见原生文本、窄屏横滚/深色/打印 CSS 与静态产物检查均已通过，仍建议人工打开最终 HTML 做一次视觉抽查。
- [x] 由三名只读审查者分别复核 CodeIR/runtime、memory/time、gate/result/comparison/conformance 与渲染闭包；发现的 backend face 耦合、compute stream accessor、图示分支、session/release trust-root 等 Critical/High 候选均已按 TDD 修复并复验，无未处置 Critical/High。
- [x] 执行 verification-before-completion：fresh dependency install、fresh build、fresh verifier、fresh tests、fresh hash、最终文件清单均已有证据。

### Task 5 review fix round 5 (2026-08-10)

- [x] A — validator substitution closure: `validate_and_seal_conformance` recomputes the authority and every nested subject/fixture/policy/manifest/runner digest as its first operation, before consuming observed output, ledger, attestation, measured environment, or selected trust material. The old-invocation-plus-substituted-policy/store fixture is required.
- [x] B — external trust boundary: deployment/verifier-owned `ConformanceTrustRootCapability` supplies expected trust-store and policy payloads; the request-owned authority carries only refs/digests. `MeasuredExecutionEnvironment` is protected measurer output, and the runner signs its recomputed digest rather than the policy's expected value.
- [x] C — product orchestration uses only typed staged Gate calls: projection advances structure → memory view → time view from a canonical empty ledger, then value/result/comparison stages each build the exact immutable subject and authority before `run_gate_domain`; non-requested siblings remain `NotApplicable`.
- [x] D — applicable dependency closure is computed from `applicable_new_invocations`; every record in that closure must be `Pass`. A `NotApplicable` record proves only its own non-applicable sibling and cannot authorize an applicable dependent invocation.
- [x] E — request/runtime manifest closure: `CommonProductionInputs` owns the `GateRunnerSnapshot`; request construction independently checks specification policy/ref and every applicability/clause predicate digest; runtime authority construction requires the request-owned runner/specification/policy and rejects E+P2 or wrong-runner substitutions.
- [x] F — the derived-digest exclusion table contains no owner wildcard. Every concrete owner has exactly one own-field exclusion; extra/missing/duplicate/overbroad/self-referential rows and unmapped typed canonical-payload owners fail the standalone scan.
- [x] Round 5 TDD evidence: independent A–F RED failures preceded the minimal edits; fresh target tests 54/54, full discovery 82/82, and standalone verifier 204 required / 105 forbidden / 18 gates are GREEN.

## Final Acceptance Criteria

- [x] 十张承重架构/流程图全部由模板中的 Mermaid 源生成，两个 HTML 内联静态 SVG，无活动 Excalidraw/SVG legacy 依赖。
- [x] 十个模块均有可定位的结构化契约说明，核心结构、接口、结果联合、阻断与不变量闭合。
- [x] compute 时间严格来自用户采集的 exact record；communication 时间严格来自冻结理论公式；P2P/PP/CP 通信被建模；本轮明确不建模资源竞争与跨流 allocator 复用竞争。
- [x] `index.html` 与 `artifact.html` 已通过离线、自包含、可访问性、浅/深色、窄屏横滚、打印 CSS 和双构建确定性自动检查；受管理员策略限制的 live-browser 人工抽查不作为本次离线交付阻断项。
- [x] 全量 verifier、单元测试、SVG 安全校验与三路只读审查均无未处置 Critical/High 问题。

## 2026-08-11 Review Reconciliation

### Task 7: Close the core modeling and comparison contracts

**Files:**
- Modify: `docs/target-design-v2/src/index.template.html`
- Modify: `docs/target-design-v2/tools/test_module_contracts.py`
- Modify: `docs/target-design-v2/tools/test_verify_gates.py`
- Modify: `docs/target-design-v2/tools/verify_gates.py`

- [x] Add failing behavioral verifier cases for compute-profile versus communication-formula cost ownership, memory/time face independence, comparison outcome priority, typed Coverage conservation, split-specific calibration manifests, capacity exclusion, and Chapter 0/14 boundary equivalence.
- [x] Make those cases fail against the committed `ea07e95` baseline with zero test errors.
- [x] Apply the minimal normative changes specified in design §12.1–§12.2.
- [x] Run the focused tests and standalone verifier to GREEN (`70/70` contract tests; `212` required / `110` forbidden / `18/18` gates; final scoped review APPROVED at `af6096f`).

### Task 8: Replace artifact flows with layered module architecture

**Files:**
- Modify: `docs/target-design-v2/src/index.template.html`
- Modify: `docs/target-design-v2/HANDOFF.md`
- Modify: `docs/target-design-v2/tools/test_build_doc.py`
- Modify: `docs/target-design-v2/tools/test_module_contracts.py`
- Modify: `docs/target-design-v2/tools/test_verify_gates.py`
- Modify: `docs/target-design-v2/tools/verify_gates.py`

- [x] Add failing tests requiring the two architecture views to use module/layer abstractions, disallow artifact-only architecture nodes, and require a tenth `plan-projection` module contract.
- [x] Make those cases fail before changing the Mermaid source.
- [x] Replace figures 1 and 7 without changing the total figure count; preserve the other eight behavioral views.
- [x] Add the plan/projection module boundary, authoritative type references, typed ports, result branches, dependency DAG, blocker scope and invariants; synchronize HANDOFF.
- [x] Render all Mermaid figures with the pinned renderer and run focused tests to GREEN (`80/80` module/verifier; `35/35` Mermaid/build; `10/10` pinned CLI renders; final two-reviewer APPROVED at `f69890b`).

### Task 9: Make conformance hardening an optional deployment profile

**Files:**
- Modify: `docs/target-design-v2/src/index.template.html`
- Modify: `docs/target-design-v2/HANDOFF.md`
- Modify: `docs/target-design-v2/tools/test_module_contracts.py`
- Modify: `docs/target-design-v2/tools/test_verify_gates.py`
- Modify: `docs/target-design-v2/tools/verify_gates.py`

- [x] Add failing cases proving BasicOfflineConformance cannot authorize release and AttestedReleaseConformance alone owns trust roots/session replay protection.
- [x] Make those cases fail on the mandatory-hardened baseline with zero test errors.
- [x] Add the closed profile union and profile-specific interfaces while keeping both profiles outside production Memory/Time inputs, digests, caches, and comparison basis.
- [x] Run focused, full, build determinism, SVG safety, and standalone verifier checks; regenerate `index.html` and `artifact.html` twice and require byte-stable output (`149/149` full discover, `66/66` verifier, `35/35` Mermaid/build, `212/110/18` standalone).
- [x] Perform a final source-faithful review of the seven findings and the two architecture views; final consistency review: `0 Critical / 0 High / 0 Important`, `APPROVED`.
