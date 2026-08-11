# docs/target-design-v2/tools/verify_gates.py
"""核验 target-design-v2 v4.2 的产品、PyNative IR 与结构门契约。

本脚本不判断预测是否等同于真机；它防止文档在编辑过程中重新混入 v3 的
证明型输出、π₀/pass/rewrite 架构或运行时 trace 依赖，也防止内存/时间后端
失去必需的规范文字。它是文档 linter，不证明运行时 gate predicate 已实现；
运行时语义必须由 GateManifest 对应的正反例测试验证。

用法：python tools/verify_gates.py
任一契约缺失、旧语义残留、章节或门集合漂移时退出码非 0。
"""
from __future__ import annotations

import os
import re
import sys
from html import unescape
from html.parser import HTMLParser


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src", "index.template.html")
HANDOFF = os.path.join(os.path.dirname(HERE), "HANDOFF.md")

REQUIRED_TEXT = (
    "目标方案设计 v4.2",
    "工程预测仿真器",
    "BackendResult",
    "MemoryEstimate",
    "StepTimeEstimate",
    "peak_allocated_bytes",
    "predicted_step_time",
    "Allocate",
    "Bind",
    "Free",
    "同口径配置可比较",
    "Incomparable",
    "CodeIR(P)",
    "RankCodeIR",
    "LogicalRankContextSet",
    'data-module="input-facts"',
    'data-module="code-ir"',
    'data-module="runtime-events"',
    "StructureRegistrySnapshot",
    "RuntimeRegistrySnapshot",
    "CommonProductionInputs",
    "RequestedBackendInput",
    "requested_backend_inputs",
    "RequestSnapshotBuildResult",
    "build_request_snapshot(",
    "canonical_utf8_source_bytes: ByteString",
    "RankCodeIRBuildInput:",
    "TensorValue:",
    "LogicalStorage:",
    "TensorStorageRelation:",
    "DataEdge:",
    "ControlEdge:",
    "RankCodeIRBuildResult",
    "CodeIRBuildResult",
    "evaluate_source(",
    "assemble_code_ir(",
    "rank_build_blocker_union(",
    "RuntimeRuleSnapshot:",
    "expand_runtime_semantics(",
    'data-module="memory-backend"',
    'data-module="time-backend"',
    'data-module="result-sealing"',
    'data-module="gate-system"',
    'data-module="comparison"',
    'data-module="conformance"',
    "MemoryTimelineEntry:",
    "MemoryExecutionWitness:",
    "MemoryProjectionSemanticsSnapshot:",
    "run_memory_backend(",
    "RouteBinding:",
    "BoundP2P extends BoundCommunicationCommon:",
    "BoundCollective extends BoundCommunicationCommon:",
    "TimeTimelineEntry:",
    "TimeExecutionWitness:",
    "run_time_backend(",
    "BackendExecution&lt;T, W&gt; :=",
    "BackendSealBuildResult&lt;T, V&gt; :=",
    "run_backend_build_candidate_and_seal(",
    "SourceObligation",
    "EventObligation",
    "TensorInstanceId",
    "SimulationInitialState",
    "RuntimeEventPlan",
    "SimulationPlan",
    "SimulationPlanCore",
    "ProjectionResult",
    "Opaque capabilities ConformanceDeploymentProfile, ConformanceTrustRootCapability, ReleaseApprovalTrustRootCapability and ValidatedMeasurementSessionCapability are non-serializable, own no derived digest and therefore have no exclusion-table row",
    "ExecutionDeployment",
    "HardwareProjectionFacts",
    "HardwareBindingPolicySnapshot",
    "KernelVariantBinding",
    "kernel_variant_binding_digest",
    "RuntimeBuildResult",
    "CoreBuildResult",
    "ResolvedEventSemantic",
    "TensorStorageBinding",
    "StorageBinding",
    "EffectSummary",
    "RuntimeRuleInvocationRef",
    "RuntimeTensorRef",
    "AutogradLink",
    "ScheduleConstraintEdge",
    "LogicalLifetime",
    "dependency_safe",
    "WorkspaceBinding",
    "BoundWorkspaceLifetime",
    "align_up",
    "ExpectedMemoryProjection",
    "P2PIntent",
    "matched_rendezvous",
    "all_participant_rendezvous",
    "global_progress_graph",
    "CompileEnvFacts",
    "HardwareProfile",
    "CommunicationModelSnapshot",
    "SemanticValue",
    "ObservationRef",
    "MeasurementKey",
    "MeasurementKeyTemplate",
    "split_group_key",
    "exact measurement record",
    "PhysicalStreamId",
    "StreamBinding",
    "CommunicationEndpointDescriptor",
    "ExpectedTimeProjection",
    "occupied_streams(node) for every progress node",
    "occupied_streams(compute)==OrderedSet{physical_stream_id}",
    "occupied_streams(node) accessor",
    "selected_device_domain",
    "pipeline_stage_domain",
    "timeline_numeric_policy",
    "NumericRangeWitness",
    "unbounded_integer_preflight",
    "NumericPolicySnapshot",
    "EstimateContext",
    "EstimateCandidate",
    "BackendResultCandidate",
    "BackendSealAuthority",
    "RequestSnapshot",
    "CanonicalConfigEvaluationInput",
    "EvaluationInstanceIdentity",
    "evaluation_instance_digest",
    "ProductionSubject",
    "backend_semantics_digests := requested_backend_semantics_digests(request.requested_backend_inputs)",
    "requested_backend_semantics_digests excludes every NotRequested arm",
    "ProductionValidationContext",
    "ProjectionCandidate",
    "ProjectionBundleAuthority",
    "projection_bundle_authority_digest",
    "ProjectionBundleBuild",
    "runtime_result: RuntimeBuildResult",
    "affected_backends=={memory,time}",
    "evaluate_and_finalize_projection_bundle",
    "base_projection_blockers",
    "all_projection_blockers",
    "estimate_candidate_digest",
    "BackendValueGateAuthority",
    "value_gate_authority_digest",
    "backend_result_candidate_digest",
    "BackendSealArtifact",
    "backend_seal_artifact_digest",
    "ComparisonSourceAuthority",
    "ComparisonArmAuthority",
    "arm_bundle",
    "unique byte-identical nested ProjectionBundleAuthority",
    "ComparisonBasisPair",
    "masked_config_evaluation_input",
    "canonical_comparison_basis_pairs",
    "derive_comparison_basis_pair",
    "ComparisonResultCandidate",
    "ComparisonSealAuthority",
    "ComparisonSealArtifact",
    "comparison_seal_artifact_digest",
    "build_and_seal_comparison",
    "InternalContractViolation",
    "ContractViolationCode",
    "view_blockers",
    "affected_blockers",
    "ready_prerequisites",
    "backend_value_postconditions",
    "result_seal_postconditions",
    "BlockerInstanceId",
    "BlockerCode",
    "BlockerRecord",
    "BlockerScopePolicy",
    "estimate_context_digest",
    "fractional_change",
    "context_refs",
    "RankClass",
    "canonical_payload_without_derived_digests",
    "model_input_digest",
    "memory_simulation_digest",
    "time_simulation_digest",
    "event_projection",
    "expected_progress_edges",
    "MetricComparison",
    "ComparisonBasis",
    "UndefinedZeroBaseline",
    "BLK-EVENT-SEMANTICS",
    "BLK-AMBIGUOUS-COST",
    "BLK-NUMERIC-RANGE",
    "AttestedConformanceReport",
    "GateManifestEntry",
    "GateStage",
    "GateInvocationId",
    "GateOutcome",
    "GateExecutionRecord",
    "GateExecutionLedger",
    "gate_execution_ledger_digest",
    "BackendResultSourceAuthority",
    "GateClauseId",
    "GateClauseOutcome",
    "GateClauseExecutionRecord",
    "manifest_clause",
    "normative_clause_ref",
    "effective_failure_disposition",
    "ClauseFail carries no disposition",
    "entry.invocation_id == invocation_key",
    "clause_record.clause_id == clause_key == clause.clause_id",
    "FailureInstanceKey",
    "FailureWitness",
    "stage_evaluation_context_digests",
    "GateFailureDisposition",
    "GateClause",
    "BLK-UNSUPPORTED-LAYOUT",
    "TraceFixture",
    "IRConformance",
    "TraceFixture 只用于测试",
    "TraceFixture 不进入生产输入",
    "TraceFixture 不进入 model_digest",
    "TraceFixture 不进入 simulation_digest 或缓存键",
    "本版不模拟共享资源竞争或并发降速",
)

FORBIDDEN_TEXT = (
    "InternalContractViolation set",
    "InternalContractViolation(CV-COMPARISON-CONTRACT set)",
    "return InternalContractViolation set",
    "produce InternalContractViolation set",
    "return InternalContractViolation(CV-REPORT-CONTRACT set)",
    "evaluation_subject_digest(subject: GateEvaluationSubject) :=",
    "result_seal_subject_digest :=",
    "seal_context_digest :=",
    "comparison_context_digest :=",
    "oom_verdict",
    "peak_interval",
    "peak_allocated.lo",
    "peak_allocated.hi",
    "SoundResult",
    "Sample[step_time]",
    "VERDICT-WITHDRAW",
    "可证下界",
    "配置间不可比较",
    "本工具不回答哪个配置更快",
    "CoreIR = PE(Source, ModelSpec, EnvFacts, π₀)",
    "策略单位元 π₀",
    "Pass_impl_select",
    "实现选择与融合",
    "折叠到一个 kernel",
    "replacement 映射",
    "只增不改",
    "Gk --rewrite--> Gk+1",
    "GraphVersion",
    "TraceFixture 作为生产输入",
    "trace 作为预测输入",
    "trace 进入 model_digest",
    "trace 进入缓存键",
    "TraceFixture → SimulationPlan",
    "目标方案设计 v4.1",
    "resource_ready   := earliest_capacity",
    "resource_requests\n",
    "arbitration_rule\n",
    "capacity-constrained execution target",
    "memory_capacity",
    "source_nodes_planned",
    "Bind(tensor_id, storage_instance_id",
    "StorageInstanceId := (StorageId, logical_rank, materialization_epoch)",
    "从 direct measurement 回退到 calibrated model",
    "exact_or_declared_in_domain_user_measurement",
    "域内插值",
    "下一个 logical join",
    "next logical join",
    "bound_compute_events",
    "SimulationPlan/result</td><td>simulation_digest",
    "WorkspaceRef :=\n  (EventId, workspace_role, CostBindingDigest)",
    "Map[StreamId",
    "reentrant: bool",
    "effects_compatible",
    "reentrancy_satisfied",
    "rank_or_rank_class",
    "with quotient self-loops removed",
    "assert progress_edges == expected_progress_edges",
    "GateManifest := Map",
    "subject_digest := hash(build_id",
    "stream_overlap",
    "wait_breakdown",
    "compute duration/workspace",
    "event class       → stream_id",
    "source order first",
    "otherwise base reachability",
    "memory_policy)",
    "Defined { ratio: Decimal }",
    "BlockerId",
    "  binding_digest, evidence",
    "every gate for that backend passes",
    "core_result.diagnostics",
    "hash(backend-specific simulation digest",
    "hash(all Estimate fields except result_digest)",
    "  effects\n  resolved_semantic_ref",
    "G-MEM4, G-MEM5, G-MEM6",
    "G-TIME4, G-TIME5, G-TIME6",
    "finish_ready_backend",
    "backend_postcondition | comparison_only",
    "expected_failure_code",
    "backend execution/value-postcondition/result-seal blockers if produced",
    "append canonical BlockerRecord(s) for every value failure",
    "depends_on: OrderedSet[GateId]",
    "entries: Map[GateId, GateManifestEntry]",
    "validate_and_seal_backend_result(backend, result_candidate)",
    "aggregate_duration_upper_bound_ns",
    "requested_backend_set",
    "build_memory_projection_and_gate_ledger",
    "build_time_projection_and_gate_ledger",
    "comparison = compare_per_metric",
    "GateOutcome :=\n  Pass { assertion_witness_digest }",
    "value_failures := run",
    "ProjectionGateEvaluationAuthority",
    "projection_gate_authority_digest",
    "evaluate_and_finalize_projection(",
    "canonical_union(bs, gate_blockers)",
    "memory_artifacts[P]",
    "time_artifacts[P]",
    "for P in normalized_configs",
    "left_seal_artifacts:",
    "right_seal_artifacts:",
    "canonical_comparison_bases:",
    "ClauseFail {\n      failure_disposition",
    "microbatch 数不能改变它",
    "relation_to_root: new | alias | view | inplace",
    "authority: ProjectionGateEvaluationAuthority",
    "workspace_registry: WorkspaceRegistrySnapshot",
    "stream_policy: StreamAssignmentPolicySnapshot",
    "runtime and backend facts resolve uniquely",
    "InputBlocker / no synthesized zero or pure effect",
    "node.occupied_streams) for node in order",
)

EXPECTED_GATES = {
    "G-IR1",
    "G-IR2",
    "G-IR3",
    "G-MEM1",
    "G-MEM2",
    "G-MEM3",
    "G-MEM4",
    "G-MEM5",
    "G-MEM6",
    "G-TIME1",
    "G-TIME2",
    "G-TIME3",
    "G-TIME4",
    "G-TIME5",
    "G-TIME6",
    "G-REP1",
    "G-REP2",
    "G-REP3",
}

TASK7_COVERAGE_FIELDS = {
    "source_obligation_universe": "SourceObligationUniverse",
    "source_obligations_total": "SourceObligationCount",
    "source_obligations_planned": "SourceObligationCount",
    "source_obligations_residual": "SourceObligationCount",
    "source_obligations_proven_not_executed": "SourceObligationCount",
    "event_obligation_universe": "EventObligationUniverse",
    "event_obligations_total": "EventObligationCount",
    "event_obligations_planned": "EventObligationCount",
    "event_obligations_blocked": "EventObligationCount",
    "event_obligations_not_applicable": "EventObligationCount",
    "op_occurrence_universe": "OpOccurrenceUniverse",
    "op_occurrences_total": "OpOccurrenceCount",
    "op_occurrences_modeled": "OpOccurrenceCount",
    "op_occurrences_blocked": "OpOccurrenceCount",
    "storage_instance_universe": "StorageInstanceUniverse",
    "storage_instances_total": "StorageInstanceCount",
    "storage_instances_modeled": "StorageInstanceCount",
    "byte_universe": "KnownByteUniverse",
    "bytes_total_known": "ByteCount",
    "bytes_modeled": "ByteCount",
    "time_event_universe": "TimeEventUniverse",
    "time_events_total": "TimeEventCount",
    "time_events_priced": "TimeEventCount",
    "emitted_codeir_node_count": "CodeIRNodeCount",
}

TASK7_COMMON_PRODUCTION_INPUT_FIELDS = {
    "source_snapshot": "SourceSnapshot",
    "model_spec": "ModelSpec",
    "compile_env_facts": "CompileEnvFacts",
    "structure_registry_snapshot": "StructureRegistrySnapshot",
    "runtime_registry_snapshot": "RuntimeRegistrySnapshot",
    "hardware_profile": "HardwareProfile",
    "hardware_binding_policy_snapshot": "HardwareBindingPolicySnapshot",
    "blocker_scope_policy_snapshot": "BlockerScopePolicySnapshot",
    "gate_specification_set_snapshot": "GateSpecificationSet",
    "gate_specification_set_ref": "GateSpecificationSetRef",
    "gate_runner_snapshot": "GateRunnerSnapshot",
    "comparison_schema_snapshot": "ComparisonSchemaSnapshot",
    "production_policy_snapshots": "ProductionPolicySnapshots",
}

TASK7_CANONICAL_PRODUCTION_EVALUATION_INPUT_FIELDS = {
    "common_inputs": "CommonProductionInputs",
    "configurations": (
        "NonEmptyOrderedMap[ConfigRef, CanonicalConfigEvaluationInput]"
    ),
}

TASK7_REQUEST_SNAPSHOT_FIELDS = {
    "canonical_production_evaluation_inputs": "CanonicalProductionEvaluationInputs",
    "requested_backend_inputs": (
        "OrderedMap[memory | time, RequestedBackendInput]"
    ),
    "requested_backends": "OrderedSet[memory | time]",
    "comparison_request": "ComparisonRequest | None",
    "request_digest": "Digest",
}

TASK7_REQUESTED_BACKEND_INPUT_ARMS = {
    "MemoryRequested": {
        "memory_registry_snapshot": "MemoryRegistrySnapshot",
    },
    "TimeRequested": {
        "calibration_set": "CalibrationSet",
        "calibration_train_manifest_snapshot": "CalibrationTrainManifest",
        "communication_model_snapshot": "CommunicationModelSnapshot",
        "time_cost_policy": "TimeCostPolicy",
    },
    "NotRequested": {},
}

TASK7_COMPARISON_BASIS_FIELDS = {
    "metric": "memory | time",
    "metric_basis_schema_digest": "Digest",
    "source_snapshot_digest": "Digest",
    "model_architecture_and_workload_digest": "Digest",
    "logical_rank_id_set": "OrderedSet[LogicalRank]",
    "relevant_hardware_digest": "Digest",
    "relevant_registry_and_cost_dataset_digests": "OrderedSet[Digest]",
    "selected_measurement_protocol_digest": "Digest | NotApplicable",
    "fallback_policy": "FallbackPolicySnapshot",
    "assumption_policy": "AssumptionPolicySnapshot",
    "canonical_resolved_fallbacks_and_assumptions": (
        "CanonicalResolvedFallbacksAndAssumptions"
    ),
    "backend_and_numeric_semantic_versions": "SemanticVersionSet",
    "masked_config_evaluation_input": "MaskedCanonicalConfigEvaluationInput",
}

TASK7_TIME_SIMULATION_INPUT_FIELDS = {
    "simulation_core_digest": "Digest",
    "time_registry_snapshot": "TimeRegistrySnapshot",
    "calibration_set": "CalibrationSet",
    "calibration_train_manifest_snapshot": "CalibrationTrainManifest",
    "communication_model_snapshot": "CommunicationModelSnapshot",
    "time_cost_policy": "TimeCostPolicy",
    "cost_bindings": "ExactCostBindings",
    "stream_bindings": "ExactStreamBindings",
    "resolved_time_fallbacks_and_assumptions": (
        "CanonicalResolvedFallbacksAndAssumptions"
    ),
    "progress_semantics": "ProgressSemanticsSnapshot",
    "projection_witness_digest": "Digest",
    "time_backend_semantic_version": "SemanticVersion",
    "time_numeric_semantic_version": "SemanticVersion",
}

TASK7_PRODUCTION_CACHE_ROWS = {
    "source-parse": (
        "Source parse",
        "source digest + parser/evaluator semantic digest + CompileEnvFacts parser subset",
        "hardware、calibration、fixture",
    ),
    "code-ir-lookup": (
        "CodeIR lookup",
        "model_input_digest；命中 artifact 必须重验其 model_digest",
        "HardwareProfile、成本数据、fixture",
    ),
    "runtime-event-plan-lookup": (
        "RuntimeEventPlan lookup",
        "runtime_input_digest；命中 artifact 必须重验其 runtime_plan_digest",
        "raw profiler output、硬件成本",
    ),
    "simulation-plan-core": (
        "SimulationPlanCore",
        "simulation_core_digest",
        "CalibrationSet、通信公式、backend 请求状态",
    ),
    "memory-result": (
        "Memory result",
        "(memory_simulation_digest, memory_result_schema_version)",
        "time projection、NotRequested、验证制品",
    ),
    "time-result": (
        "Time result",
        "(time_simulation_digest, time_result_schema_version)",
        "memory projection、NotRequested、验证制品",
    ),
}

TASK7_NON_GOAL_IDS = {
    "NG-CAPACITY",
    "NG-ALLOCATOR-RESERVED",
    "NG-PROOF",
    "NG-STATIC-GRAPH-OPTIMIZATION",
    "NG-ONLINE-TRACE",
    "NG-AUTOMATIC-SEARCH",
    "NG-MULTISTREAM-MEMORY-REUSE",
    "NG-LAYOUT-REORDER",
    "NG-CONTENTION",
    "NG-COMMUNICATION-SCRATCH",
}

TASK7_DECISION_IDS = {f"D{index}" for index in range(1, 20)}

INCOMPATIBLE_DIAGRAMS = (
    "{{SVG:01-layering}}",
    "{{SVG:02-provenance}}",
    "{{SVG:04-structure-backward}}",
    "{{SVG:05-identities}}",
    "{{SVG:06-placement-execution}}",
)

TASK8_DIAGRAM_IDS = (
    "layered-module-architecture",
    "facts-and-digests",
    "per-rank-codeir",
    "value-storage-identity",
    "semantic-effect-closure",
    "runtime-event-expansion",
    "plan-projection-module-architecture",
    "memory-logical-replay",
    "time-progress-des",
    "result-gate-comparison",
)
TASK8_ARCHITECTURE_DIAGRAM_IDS = {
    "layered-module-architecture",
    "plan-projection-module-architecture",
}
TASK8_MODULE_DEPENDENCIES = {
    "input-facts": ("L0", ""),
    "code-ir": ("L1", ""),
    "runtime-events": ("L1", ""),
    "plan-projection": ("L2", "gate-system,memory-backend,time-backend"),
    "gate-system": ("L2", ""),
    "memory-backend": ("L3", ""),
    "time-backend": ("L3", ""),
    "result-sealing": (
        "L4",
        "gate-system,memory-backend,plan-projection,time-backend",
    ),
    "comparison": ("L4", "gate-system,result-sealing"),
    "conformance": ("OFFLINE", "comparison,gate-system,result-sealing"),
}
TASK8_MODULE_ROWS = {
    "input-facts": ("L0", "冻结生产事实与请求", "CommonProductionInputs, RequestSnapshot", "build_request_snapshot", "无"),
    "code-ir": ("L1", "逐 rank 源码求值", "RankCodeIR, CodeIR", "evaluate_source; assemble_code_ir", "无"),
    "runtime-events": ("L1", "展开训练与通信事件", "RuntimeEventPlan", "expand_runtime_semantics", "无"),
    "plan-projection": ("L2", "绑定共享核并联合闭合双投影", "SimulationPlanCore, ProjectionBundleBuild", "bind_core; evaluate_and_finalize_projection_bundle", "gate-system, memory-backend, time-backend"),
    "gate-system": ("L2", "执行 staged gate 与 ledger", "GateEvaluationAuthority, GateExecutionLedger", "compile_gate_manifest; build_gate_evaluation_authority; expected_invocation_domain; run_gate_domain", "无"),
    "memory-backend": ("L3", "构造并重放内存视图", "MemoryEventView, MemoryEstimate", "build_memory_projection_candidate; run_memory_backend", "无"),
    "time-backend": ("L3", "构造时间视图并执行 DES", "TimeEventView, StepTimeEstimate", "build_time_projection_candidate; run_time_backend", "无"),
    "result-sealing": ("L4", "生成并封装后端结果", "BackendSealArtifact", "run_backend_build_candidate_and_seal", "gate-system, memory-backend, plan-projection, time-backend"),
    "comparison": ("L4", "同口径配置比较", "ComparisonResult", "derive_comparison_basis_pair; build_comparison_source_authority; compare_per_metric_from_authority", "gate-system, result-sealing"),
    "conformance": ("离线", "可选离线对照与加固发布门", "BasicOfflineReportArtifact, AttestedConformanceSealArtifact", "run_basic_offline_conformance; run_attested_release_conformance; apply_release_policy", "comparison, gate-system, result-sealing"),
}
TASK8_ARCHITECTURE_NODES = {
    "layered-module-architecture": {
        "l0label": "L0 · Input facts", "inputFacts": "input-facts",
        "l1label": "L1 · IR and events", "codeIr": "code-ir", "runtimeEvents": "runtime-events",
        "l2label": "L2 · Coordination", "planProjection": "plan-projection", "gateSystem": "gate-system",
        "l3label": "L3 · Backends", "memoryBackend": "memory-backend", "timeBackend": "time-backend",
        "l4label": "L4 · Product outputs", "resultSealing": "result-sealing", "comparison": "comparison",
        "offlineLabel": "Offline sidecar · outside production digest chain", "conformance": "conformance",
    },
    "plan-projection-module-architecture": {
        "upstreamLabel": "Upstream modules", "inputFacts": "input-facts", "runtimeEvents": "runtime-events",
        "boundaryLabel": "plan-projection boundary", "coreBinder": "Core binder", "faceCoordinator": "Face coordinator", "bundleFinalizer": "Bundle finalizer",
        "portsLabel": "Injected ports", "memoryBackend": "memory-backend", "timeBackend": "time-backend", "gateSystem": "gate-system",
    },
}
TASK8_LAYERED_DOTTED_EDGES = {
    ("inputFacts", "-.->", "codeIr"),
    ("inputFacts", "-.->", "runtimeEvents"),
    ("codeIr", "-.->", "runtimeEvents"),
    ("runtimeEvents", "-.->", "planProjection"),
    ("inputFacts", "-.->", "gateSystem"),
}
TASK8_PLAN_ARCHITECTURE_SOLID_EDGES = {
    ("coreBinder", "-->", "faceCoordinator"),
    ("faceCoordinator", "-->", "bundleFinalizer"),
    ("memoryBackend", "-->", "faceCoordinator"),
    ("timeBackend", "-->", "faceCoordinator"),
    ("gateSystem", "-->", "faceCoordinator"),
    ("gateSystem", "-->", "bundleFinalizer"),
}
TASK8_PLAN_ARCHITECTURE_DOTTED_EDGES = {
    ("inputFacts", "-.->", "coreBinder"),
    ("runtimeEvents", "-.->", "coreBinder"),
}
TASK8_PLAN_AUTHORITATIVE_CONTRACT_REFS = (
    "SimulationPlanCore",
    "CoreBuildResult<SimulationPlanCore>",
    "ProjectionCandidate<MemoryEventView>",
    "ProjectionCandidate<TimeEventView>",
    "ProjectionBundleAuthority",
    "ProjectionBundleBuild",
    "ProjectionResult<MemoryEventView>",
    "ProjectionResult<TimeEventView>",
)

CONFORMANCE_LOCAL_BLOCKS = {
    ("scope", "shared"),
    ("scope", "deployment"),
    ("profile", "basic-offline"),
    ("profile", "attested-release"),
}
CONFORMANCE_BASIC_TYPES = (
    "BasicOfflineConformanceAuthority",
    "BasicOfflineExecutionRecord",
    "BasicOfflineExecutionLedger",
    "BasicOfflineReport",
    "BasicOfflineReportArtifact",
    "BasicOfflineRunResult",
)
CONFORMANCE_BASIC_CONTENT_TYPES = CONFORMANCE_BASIC_TYPES[:-1]
CONFORMANCE_BASIC_FORBIDDEN = (
    r"trust[_ -]?(?:store|root)",
    r"measured[_ -]?(?:execution[_ -]?)?environment",
    r"(?:^|[^A-Za-z0-9])session(?:$|[^A-Za-z0-9])",
    r"nonce",
    r"time[_ -]?window",
    r"attestation",
    r"signature",
    r"ApprovedDigestSet",
    r"seal",
    r"approval",
    r"ReleasePolicy",
    r"ReleaseDecision",
    r"apply_release_policy",
    r"derive_approved_digest_set",
)
CONFORMANCE_SHARED_TYPES = (
    "TraceFixture",
    "FixtureBinding",
    "FixtureSet",
    "ValidationPolicy",
    "VerifierRunner",
    "ConformanceFinding",
    "ObservedClauseOutput",
    "ConformanceObservedOutput",
)
CONFORMANCE_OWNER_INVENTORY = {
    ("scope", "shared"): CONFORMANCE_SHARED_TYPES,
    ("scope", "deployment"): ("ConformanceDeploymentProfile",),
    ("profile", "basic-offline"): CONFORMANCE_BASIC_TYPES,
    ("profile", "attested-release"): (
        "TrustStoreSnapshot",
        "RunnerAttestationPolicy",
        "TrustStoreSnapshotRef",
        "RunnerAttestationPolicyRef",
        "ConformanceTrustRootCapability",
        "MeasuredExecutionEnvironment",
        "ValidatedMeasurementSessionCapability",
        "AttestedConformanceInvocationAuthority",
        "RunnerAttestation",
        "AttestedConformanceExecutionRecord",
        "AttestedConformanceExecutionLedger",
        "ApprovedDigestSet",
        "ReleaseApprovalStoreSnapshot",
        "ReleaseApprovalTrustRootCapability",
        "ReleaseApprovalAuthority",
        "ReleaseApprovalArtifact",
        "AttestedConformanceReport",
        "ReleasePolicy",
        "AttestedConformanceSealArtifact",
        "AttestedConformanceRunResult",
        "ReleaseBlockReason",
        "ReleaseDecision",
        "ReleasePolicyApplicationResult",
    ),
}
CONFORMANCE_ATTESTED_ONLY_RULES = (
    "measured_payload_digest == policy.expected_execution_environment_digest",
    "atomically consumed exactly once",
    "verify_measurer_identity_and_freshness_evidence(",
    "verify_signature(",
    "release_trust_root := profile.release_trust_root",
    "trust_root := profile.conformance_trust_root",
)
CONFORMANCE_ATTESTED_SECURITY_RULES = (
    "negative fixture: no execution but forged all-pass",
    "negative fixture: forged runner attestation",
    "negative fixture: forged ApprovedDigestSet",
    "negative fixture: tampered execution record or observed output",
    "negative fixture: valid signature plus failing observed output cannot be sealed with forged pass",
    "negative fixture: release approval wrong signature scheme",
    "negative fixture: unsupported release approval signature scheme",
    "negative fixture: release approval self-owned approver key",
    "negative fixture: release approval wrong store snapshot",
    "negative fixture: release approval unsupported signature scheme",
    "negative fixture: release approval substituted policy",
    "negative fixture: wrong runner attestation trust store",
    "negative fixture: unsupported runner attestation signature scheme",
    "negative fixture: tampered runner execution environment",
    "negative fixture: runner self-owned attestation key",
    "negative fixture: release artifact from different conformance trust root",
    "negative fixture: old invocation digest plus substituted policy or trust-store reference",
    "negative fixture: replay valid measurement envelope from another execution session",
    "negative fixture: change only measurer identity or freshness evidence while retaining old envelope digest",
)
CONFORMANCE_BASIC_CLOSURE_RULES = (
    "require record.fixture_ref == record_key",
    "require record.target_invocation_id == binding.target_invocation_id",
    "require record.target_clause_id == binding.target_clause_id",
    "require record.fixture_digest == binding.fixture.fixture_digest",
    "require record.observed_clause_output == observed_clause_outputs[record_key]",
    'require record.evaluation_input_digest == hash("basic-offline-record-input/v1", authority.basic_offline_authority_digest, canonical(binding), binding.fixture_binding_digest)',
    'require record.basic_offline_execution_record_digest == hash("basic-offline-record/v1",',
    "require ledger.expected_binding_domain == keys(authority.fixture_set.bindings)",
    "require keys(ledger.execution_records) == ledger.expected_binding_domain",
    "require ledger.basic_offline_authority_digest == authority.basic_offline_authority_digest",
    "require ledger.subject_digest == authority.production_subject.subject_digest",
    "require ledger.fixture_set_digest == authority.fixture_set.fixture_set_digest",
    "require ledger.validation_policy_digest == authority.validation_policy.validation_policy_digest",
    "require ledger.gate_manifest_digest == authority.gate_manifest.gate_manifest_digest",
    "require ledger.verifier_runner_digest == authority.verifier_runner.verifier_runner_digest",
    "require ledger.observed_output == observed",
    'require ledger.basic_offline_execution_ledger_digest == hash("basic-offline-ledger/v1",',
    "require report.production_subject == authority.production_subject",
    "require report.subject_digest == authority.production_subject.subject_digest",
    "require report.basic_offline_authority_digest == authority.basic_offline_authority_digest",
    "require report.observed_output_digest == observed.observed_output_digest",
    "require report.fixture_set_digest == authority.fixture_set.fixture_set_digest",
    "require report.validation_policy_digest == authority.validation_policy.validation_policy_digest",
    "require report.gate_manifest_digest == authority.gate_manifest.gate_manifest_digest",
    "require report.verifier_runner_digest == authority.verifier_runner.verifier_runner_digest",
    "require report.basic_offline_execution_ledger_digest == ledger.basic_offline_execution_ledger_digest",
    "require report.offline_verdict == derived_offline_verdict",
    "require report.findings == observed.findings",
    'require report.basic_offline_report_digest == hash("basic-offline-report/v1",',
    "require artifact.authority == authority",
    "require artifact.observed_output == observed",
    "require artifact.execution_ledger == ledger",
    "require artifact.report == report",
    "require artifact.basic_offline_report_artifact_digest == hash(",
    "recompute every Basic artifact nested digest before Completed",
    "return Completed iff every Basic closure equation holds; otherwise InternalViolation",
)
CONFORMANCE_BASIC_ALGORITHM_LINES = tuple(
    """  recompute authority and every nested content digest before any binding
  require authority.basic_offline_authority_digest == hash("basic-offline-authority/v1",
    canonical_payload_without_derived_digests(authority))
  require authority.gate_manifest.subject_digest == authority.production_subject.subject_digest
  require keys(authority.fixture_set.bindings) is the exact expected binding domain
  for every (record_key, binding) in authority.fixture_set.bindings:
    execute binding exactly once and build record: BasicOfflineExecutionRecord
    require record.fixture_ref == record_key
    require record.target_invocation_id == binding.target_invocation_id
    require record.target_clause_id == binding.target_clause_id
    require record.fixture_digest == binding.fixture.fixture_digest
    require record.evaluation_input_digest == hash("basic-offline-record-input/v1",
      authority.basic_offline_authority_digest,
      canonical(binding), binding.fixture_binding_digest)
    require record.observed_clause_output == observed_clause_outputs[record_key]
    require record.basic_offline_execution_record_digest == hash("basic-offline-record/v1",
      canonical_payload_without_derived_digests(record))
  observed := ConformanceObservedOutput(observed_clause_outputs, findings, coverage_gaps)
  ledger := BasicOfflineExecutionLedger(authority identity fields,
    execution_records, observed, keys(authority.fixture_set.bindings))
  require ledger.expected_binding_domain == keys(authority.fixture_set.bindings)
  require keys(ledger.execution_records) == ledger.expected_binding_domain
  require ledger.basic_offline_authority_digest == authority.basic_offline_authority_digest
  require ledger.subject_digest == authority.production_subject.subject_digest
  require ledger.fixture_set_digest == authority.fixture_set.fixture_set_digest
  require ledger.validation_policy_digest == authority.validation_policy.validation_policy_digest
  require ledger.gate_manifest_digest == authority.gate_manifest.gate_manifest_digest
  require ledger.verifier_runner_digest == authority.verifier_runner.verifier_runner_digest
  require ledger.observed_output == observed
  require ledger.basic_offline_execution_ledger_digest == hash("basic-offline-ledger/v1",
    canonical_payload_without_derived_digests(ledger))
  (derived_findings, derived_offline_verdict) := derive_conformance_findings_and_verdict(
    authority.validation_policy, observed,
    exact_evidence_or_coverage_gaps(authority, observed, ledger))
  report := BasicOfflineReport(authority/observed/ledger identity fields,
    derived_offline_verdict, derived_findings)
  require report.production_subject == authority.production_subject
  require report.subject_digest == authority.production_subject.subject_digest
  require report.basic_offline_authority_digest == authority.basic_offline_authority_digest
  require report.observed_output_digest == observed.observed_output_digest
  require report.fixture_set_digest == authority.fixture_set.fixture_set_digest
  require report.validation_policy_digest == authority.validation_policy.validation_policy_digest
  require report.gate_manifest_digest == authority.gate_manifest.gate_manifest_digest
  require report.verifier_runner_digest == authority.verifier_runner.verifier_runner_digest
  require report.basic_offline_execution_ledger_digest == ledger.basic_offline_execution_ledger_digest
  require report.offline_verdict == derived_offline_verdict
  require report.findings == observed.findings
  require report.basic_offline_report_digest == hash("basic-offline-report/v1",
    canonical_payload_without_derived_digests(report))
  artifact := BasicOfflineReportArtifact(authority, observed, ledger, report)
  require artifact.authority == authority
  require artifact.observed_output == observed
  require artifact.execution_ledger == ledger
  require artifact.report == report
  recompute every Basic artifact nested digest before Completed
  require artifact.basic_offline_report_artifact_digest == hash(
    "basic-offline-report-artifact/v1",
    canonical_payload_without_derived_digests(artifact))
  content hashes prove identity/integrity only, not runner authenticity
  return Completed iff every Basic closure equation holds; otherwise InternalViolation""".splitlines()
)
CONFORMANCE_RELEASE_FIRST_OPERATION_LINES = tuple(
    """  conformance_trust_root := profile.conformance_trust_root
  release_trust_root := profile.release_trust_root
  recompute artifact and every nested invocation/report/ledger/attestation/environment digest
  approval_authority := approval.authority
  approved := approval_authority.approved
  recompute approval_authority.store_snapshot and every nested ApprovedDigestSet value
  require approval_authority.store_snapshot.release_approval_store_snapshot_digest ==
    hash(canonical_payload_without_derived_digests(approval_authority.store_snapshot))
  recompute policy.release_policy_digest
  require policy.release_policy_digest ==
    hash(canonical_payload_without_derived_digests(policy))
  recompute approval_authority.release_approval_authority_digest
  require approval_authority.release_approval_authority_digest ==
    hash(canonical_payload_without_derived_digests(approval_authority))
  recompute approval.release_approval_artifact_digest
  require approval.release_approval_artifact_digest ==
    hash(canonical_payload_without_derived_digests(approval))
  subject_digest := artifact.invocation_authority.production_subject.subject_digest
  derived_approved := derive_approved_digest_set(artifact, policy)
  approved_mismatches := canonical_schema_path_diff(approved, derived_approved)
  require canonical(approved) == canonical(approval_authority.approved)
  require derived_approved == derive_approved_digest_set(artifact, policy)
  require every stored nested digest equals its own enclosing payload recomputation; no approved/derived cross-equality
  require all equations above execute before resolving either profile trust root key material, verifying any signature or reading verdict""".splitlines()
)
CONFORMANCE_RELEASE_AFTER_OPERATION_LINES = tuple(
    """apply_release_policy equations after first operation:
  (current_conformance_store, current_conformance_policy) :=
    resolve_conformance_trust_root(conformance_trust_root)
  require conformance_trust_root was issued by deployment/verifier configuration and
          cannot be constructed by request deserialization, fixture, artifact,
          approval store or approval artifact
  require current_conformance_store.key_registry_digest ==
    hash(canonical current_conformance_store.trusted_key_material_by_id)
  require current_conformance_store.trust_store_snapshot_digest ==
    hash(canonical_payload_without_derived_digests(current_conformance_store))
  require current_conformance_policy.runner_attestation_policy_digest ==
    hash(canonical_payload_without_derived_digests(current_conformance_policy))
  require canonical(current_conformance_store) ==
          canonical(conformance_trust_root.expected_trust_store_snapshot)
  require canonical(current_conformance_policy) ==
          canonical(conformance_trust_root.expected_runner_attestation_policy)
  require artifact.invocation_authority.trust_store_snapshot_ref.trust_store_snapshot_digest ==
          current_conformance_store.trust_store_snapshot_digest ==
          conformance_trust_root.expected_trust_store_snapshot_digest
  require artifact.invocation_authority.runner_attestation_policy_ref.runner_attestation_policy_digest ==
          current_conformance_policy.runner_attestation_policy_digest ==
          conformance_trust_root.expected_runner_attestation_policy_digest
  require current_conformance_policy.trust_store_snapshot_digest ==
          current_conformance_store.trust_store_snapshot_digest
  require current_conformance_policy.supported_signature_schemes ==
          current_conformance_store.supported_signature_schemes
  require current_conformance_policy.attestation_signature_scheme in
          current_conformance_policy.supported_signature_schemes
  require artifact.measured_environment.conformance_invocation_digest ==
          artifact.runner_attestation.conformance_invocation_digest ==
          artifact.execution_ledger.conformance_invocation_digest ==
          artifact.report.conformance_invocation_digest ==
          artifact.invocation_authority.conformance_invocation_digest
  require artifact.measured_environment.verifier_runner_digest ==
          artifact.runner_attestation.verifier_runner_digest ==
          artifact.execution_ledger.verifier_runner_digest ==
          artifact.report.verifier_runner_digest ==
          artifact.invocation_authority.verifier_runner.verifier_runner_digest
  require artifact.measured_environment.executable_artifact_digest ==
          artifact.runner_attestation.executable_artifact_digest ==
          artifact.invocation_authority.verifier_runner.executable_artifact_digest
  require artifact.runner_attestation.measured_execution_environment_digest ==
          artifact.execution_ledger.measured_execution_environment_digest ==
          artifact.report.measured_execution_environment_digest ==
          artifact.measured_environment.measured_execution_environment_digest
  require artifact.runner_attestation.observed_output_digest ==
          artifact.execution_ledger.observed_output_digest ==
          artifact.report.observed_output_digest
  require artifact.execution_ledger.subject_digest ==
          artifact.report.subject_digest ==
          artifact.invocation_authority.production_subject.subject_digest
  require artifact.execution_ledger.fixture_set_digest ==
          artifact.report.fixture_set_digest ==
          artifact.invocation_authority.fixture_set.fixture_set_digest
  require artifact.execution_ledger.validation_policy_digest ==
          artifact.report.validation_policy_digest ==
          artifact.invocation_authority.validation_policy.validation_policy_digest
  require artifact.execution_ledger.gate_manifest_digest ==
          artifact.report.gate_manifest_digest ==
          artifact.invocation_authority.gate_manifest.gate_manifest_digest
  require artifact.execution_ledger.runner_attestation_digest ==
          artifact.report.runner_attestation_digest ==
          artifact.runner_attestation.runner_attestation_digest
  require artifact.report.conformance_execution_ledger_digest ==
          artifact.execution_ledger.conformance_execution_ledger_digest
  require artifact.report.production_subject ==
          artifact.invocation_authority.production_subject
  require artifact.execution_ledger.observed_output_digest ==
          artifact.execution_ledger.observed_output.observed_output_digest
  require artifact.report.findings ==
          artifact.execution_ledger.observed_output.findings
  require current_conformance_policy.trusted_attestation_key_id in
          current_conformance_store.trusted_key_material_by_id; otherwise InternalViolation
  current_conformance_trusted_key_material :=
    current_conformance_store.trusted_key_material_by_id[
      current_conformance_policy.trusted_attestation_key_id]
  require artifact.runner_attestation.trusted_attestation_key_id ==
          current_conformance_policy.trusted_attestation_key_id
  require artifact.runner_attestation.attestation_signature_scheme ==
          current_conformance_policy.attestation_signature_scheme
  require current_conformance_policy.attestation_signature_scheme in
          current_conformance_store.supported_signature_schemes
  current_measurement_evidence_message := canonical_tuple(
    artifact.measured_environment.conformance_invocation_digest,
    artifact.measured_environment.verifier_runner_digest,
    artifact.measured_environment.executable_artifact_digest,
    artifact.measured_environment.execution_session_id,
    artifact.measured_environment.process_identity,
    artifact.measured_environment.container_identity,
    artifact.measured_environment.verifier_nonce,
    artifact.measured_environment.execution_time_window,
    canonical(artifact.measured_environment.measured_environment_payload))
  require verify_measurer_identity_and_freshness_evidence(
    conformance_trust_root.measurement_session_authority,
    current_measurement_evidence_message,
    artifact.measured_environment.measurer_identity_and_freshness_evidence)
  current_measured_payload_digest :=
    hash(canonical artifact.measured_environment.measured_environment_payload)
  require current_measured_payload_digest ==
          current_conformance_policy.expected_execution_environment_digest
  require conformance_trust_root.measurement_session_authority.has_terminal_consumption_receipt(
    terminal_consumption_receipt_key(artifact.measured_environment))
  current_conformance_signed_message := canonical_tuple(
    artifact.runner_attestation.measured_execution_environment_digest,
    artifact.runner_attestation.executable_artifact_digest,
    artifact.runner_attestation.trusted_attestation_key_id,
    artifact.runner_attestation.attestation_signature_scheme,
    artifact.runner_attestation.conformance_invocation_digest,
    artifact.runner_attestation.observed_output_digest,
    artifact.runner_attestation.verifier_runner_digest)
  require artifact.runner_attestation.signed_message ==
          current_conformance_signed_message
  require verify_signature(current_conformance_trusted_key_material,
    current_conformance_policy.attestation_signature_scheme,
    current_conformance_signed_message, artifact.runner_attestation.signature)
  reverify sealed runner attestation under current profile root without consuming a measurement session or nonce
  root := resolve_release_approval_trust_root(release_trust_root)
  require release_trust_root was issued by deployment/verifier configuration and
          cannot be constructed by request deserialization, fixture, policy,
          approval store or approval artifact
  require root.expected_release_policy_digest == policy.release_policy_digest
  require canonical(root.expected_release_policy) == canonical(policy)
  require root.expected_release_approval_store_snapshot_digest ==
          approval_authority.store_snapshot.release_approval_store_snapshot_digest
  require canonical(root.expected_release_approval_store_snapshot) ==
          canonical(approval_authority.store_snapshot)
  if subject_digest not in approval_authority.store_snapshot.approved_by_subject:
    return Completed { decision: ReleaseBlocked { subject_digest, reasons: {ApprovalMissing} } }
  require approval_authority.store_snapshot.approved_by_subject[subject_digest] == approved
  require approval_authority.trusted_approver_key_id ==
          policy.trusted_approver_key_id
  require approval_authority.approval_signature_scheme == policy.approval_signature_scheme
  require policy.approval_signature_scheme in
          root.supported_release_approval_signature_schemes
  require policy.trusted_approver_key_id in root.trusted_approver_public_key_material_by_id; otherwise InternalViolation
  trusted_approver_public_key_material :=
    root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]
  release_approval_signed_message := canonical_tuple(
    root.expected_release_policy_digest,
    root.expected_release_approval_store_snapshot_digest,
    policy.trusted_approver_key_id, policy.approval_signature_scheme,
    canonical(approved))
  require approval_authority.signed_message == release_approval_signed_message
  require verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)
  verify release approval signature only with external-root public key material
  if approved_mismatches is NonEmpty:
    return Completed { decision: ReleaseBlocked { subject_digest, reasons:
      map(approved_mismatches, path -> ApprovedDigestMismatch(path)) } }
  require approved == derived_approved
  require all ten ApprovedDigestSet fields are byte-equal only on this authenticated empty-diff path
  decision := apply the versioned pass/fail/insufficient rule after comparing all ten approved digest fields
  return Completed { decision }""".splitlines()
)
CONFORMANCE_RELEASE_APPLY_LINES = (
    CONFORMANCE_RELEASE_FIRST_OPERATION_LINES
    + CONFORMANCE_RELEASE_AFTER_OPERATION_LINES
)
CONFORMANCE_CALLABLE_INVENTORY = {
    ("scope", "shared"): ("derive_conformance_findings_and_verdict",),
    ("scope", "deployment"): (),
    ("profile", "basic-offline"): ("run_basic_offline_conformance",),
    ("profile", "attested-release"): (
        "run_attested_release_conformance",
        "apply_release_policy",
        "validate_and_seal_attested_conformance",
        "validate_and_consume_measurement_envelope",
        "derive_attested_conformance_report",
        "derive_approved_digest_set",
    ),
}
CONFORMANCE_ATTESTED_AUTHORITY_FIELDS = {
    "production_subject": "ProductionSubject",
    "fixture_set": "FixtureSet",
    "validation_policy": "ValidationPolicy",
    "gate_manifest": "GateManifest",
    "verifier_runner": "VerifierRunner",
    "trust_store_snapshot_ref": "TrustStoreSnapshotRef",
    "runner_attestation_policy_ref": "RunnerAttestationPolicyRef",
    "conformance_invocation_digest": "Digest",
}
CONFORMANCE_RELEASE_APPROVAL_AUTHORITY_FIELDS = {
    "store_snapshot": "ReleaseApprovalStoreSnapshot",
    "approved": "ApprovedDigestSet",
    "trusted_approver_key_id": "ApproverKeyId",
    "approval_signature_scheme": "SignatureScheme",
    "signed_message": "CanonicalMessage",
    "release_approval_authority_digest": "Digest",
}
CONFORMANCE_ATTESTED_SEAL_ARTIFACT_FIELDS = {
    "invocation_authority": "AttestedConformanceInvocationAuthority",
    "measured_environment": "MeasuredExecutionEnvironment",
    "report": "AttestedConformanceReport",
    "execution_ledger": "AttestedConformanceExecutionLedger",
    "runner_attestation": "RunnerAttestation",
    "conformance_seal_artifact_digest": "Digest",
}
CONFORMANCE_RELEASE_RESULT_TYPES = (
    "ReleaseDecision",
    "AttestedConformanceSealArtifact",
    "ReleasePolicyApplicationResult",
)
CONFORMANCE_ATTESTED_REQUIRED = (
    "AttestedConformanceInvocationAuthority:",
    "AttestedConformanceExecutionRecord:",
    "AttestedConformanceExecutionLedger:",
    "AttestedConformanceReport:",
    "AttestedConformanceSealArtifact:",
    "AttestedConformanceRunResult :=",
    'profile_schema_id := "attested-release/v1"',
    'hash("attested-conformance-authority/v1",',
    'hash("attested-conformance-record/v1",',
    'hash("attested-conformance-ledger/v1",',
    'hash("attested-conformance-report/v1",',
    'hash("attested-conformance-seal-artifact/v1",',
    "run_attested_release_conformance( authority: AttestedConformanceInvocationAuthority, profile: AttestedReleaseConformance, measured_environment: MeasuredExecutionEnvironment ) -> AttestedConformanceRunResult",
    "trust_root := profile.conformance_trust_root",
    "apply_release_policy( artifact: AttestedConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy, profile: AttestedReleaseConformance ) -> ReleasePolicyApplicationResult",
    "release_trust_root := profile.release_trust_root",
    "ValidatedMeasurementSessionCapability:",
    "validate_and_consume_measurement_envelope(",
    "validate_and_seal_attested_conformance(",
    "ApprovedDigestSet:",
    "atomically consumed exactly once",
    "verify_signature(",
    "negative fixture: forged runner attestation",
    "negative fixture: forged ApprovedDigestSet",
    "negative fixture: replay valid measurement envelope from another execution session",
    "negative fixture: release approval self-owned approver key",
    "新建 AttestedConformanceInvocationAuthority",
    "fresh measured session",
    "完整重新执行",
)

TASK9_PRODUCTION_OWNER_FIELDS = {
    "EvaluationInstanceIdentity": {
        "request_digest": "Digest",
        "config_ref": "ConfigRef",
        "normalized_config_digest": "Digest",
        "config_input_digest": "Digest",
        "evaluation_instance_digest": "Digest",
    },
    "ProductionSubject": {
        "build_artifact_content_digest": "Digest",
        "evaluator_and_registry_digests": "OrderedSet[Digest]",
        "production_policy_digests": "OrderedSet[Digest]",
        "backend_semantics_digests": "OrderedMap[memory | time, Digest]",
        "subject_digest": "Digest",
    },
    "Estimate<T>": {
        "value": "T",
        "evidence": "[EvidenceTag]",
        "coverage": "Coverage",
        "assumptions": "[Assumption]",
        "model_digest": "Digest",
        "runtime_plan_digest": "Digest",
        "simulation_digest": "BackendSpecificDigest",
        "estimate_context_digest": "Digest",
        "result_digest": "Digest",
    },
    "EstimateCandidate<T>": {
        "estimate": "Estimate<T>",
        "backend_execution_witness": "MemoryExecutionWitness | TimeExecutionWitness",
        "estimate_candidate_digest": "Digest",
    },
    "BackendResultCandidate<T>": {
        "backend": "memory | time",
        "source_authority_digest": "Digest",
        "value_gate_authority_digest": "Digest",
        "pre_seal_gate_execution_ledger_digest": "Digest",
        "branch": "BackendResultBranch<T>",
        "backend_result_candidate_digest": "Digest",
    },
    "BackendResultSourceAuthority<V>": {
        "projection_bundle_authority": "ProjectionBundleAuthority",
        "backend": "memory | time",
        "source_projection_result": "ProjectionResult<V>",
        "source_projection_digest": "Digest",
        "projection_gate_ledger": "GateExecutionLedger",
        "source_authority_digest": "Digest",
    },
    "BackendValueGateAuthority<T,V>": {
        "source_authority": "BackendResultSourceAuthority<V>",
        "value_subject": "BackendValueSubject<T>",
        "value_gate_authority_digest": "Digest",
    },
    "BackendSealAuthority<T,V>": {
        "value_gate_authority": "BackendValueGateAuthority<T,V>",
        "pre_seal_gate_ledger": "GateExecutionLedger",
        "backend_seal_authority_digest": "Digest",
    },
    "BackendSealArtifact<T,V>": {
        "request_digest": "Digest",
        "evaluation_instance_digest": "Digest",
        "backend": "memory | time",
        "seal_authority": "BackendSealAuthority<T,V>",
        "result_candidate": "BackendResultCandidate<T>",
        "sealed_gate_ledger": "GateExecutionLedger",
        "result": "BackendResult<T>",
        "backend_seal_artifact_digest": "Digest",
    },
    "ComparisonResult": {
        "comparison_axes": "CanonicalSchemaPathSet",
        "memory": "MetricComparison<MemoryDelta>",
        "time": "MetricComparison<TimeDelta>",
    },
    "ComparisonArmAuthority": {
        "config_ref": "ConfigRef",
        "evaluation_identity": "EvaluationInstanceIdentity",
        "projection_bundle_authority_digest": "Digest",
        "seal_artifacts": "OrderedMap[memory | time, BackendSealArtifact]",
    },
    "ComparisonSourceAuthority": {
        "request_snapshot": "RequestSnapshot",
        "left_arm": "ComparisonArmAuthority",
        "right_arm": "ComparisonArmAuthority",
        "canonical_comparison_basis_pairs": "OrderedMap[memory | time, ComparisonBasisPair]",
        "production_validation_context": "ProductionValidationContext",
        "gate_manifest": "GateManifest",
        "comparison_source_authority_digest": "Digest",
    },
    "ComparisonResultCandidate": {
        "source_authority_digest": "Digest",
        "value": "ComparisonResult",
        "comparison_result_candidate_digest": "Digest",
    },
    "ComparisonSealArtifact": {
        "seal_authority": "ComparisonSealAuthority",
        "comparison_gate_ledger": "GateExecutionLedger",
        "result": "ComparisonResult",
        "comparison_seal_artifact_digest": "Digest",
    },
    "ComparisonSealAuthority": {
        "source_authority": "ComparisonSourceAuthority",
        "result_candidate": "ComparisonResultCandidate",
        "comparison_seal_authority_digest": "Digest",
    },
}

MODULE_CONTRACT_SUBSECTIONS = {
    "职责边界",
    "核心数据结构",
    "接口定义",
    "成功与阻断语义",
    "不变量",
}

MODULE_CONTRACT_REQUIRED_TEXT = {
    "input-facts": (
        "canonical_utf8_source_bytes: ByteString",
        "content_digest := hash(canonical_utf8_source_bytes)",
        "blocker_scope_policy_snapshot: BlockerScopePolicySnapshot",
        "gate_specification_set_snapshot: GateSpecificationSet",
        "gate_specification_set_ref: GateSpecificationSetRef",
        "gate_runner_snapshot: GateRunnerSnapshot",
        "RequestedBackendInput :=",
        "MemoryRequested { memory_registry_snapshot: MemoryRegistrySnapshot }",
        "TimeRequested { calibration_set: CalibrationSet calibration_train_manifest_snapshot: CalibrationTrainManifest communication_model_snapshot: CommunicationModelSnapshot time_cost_policy: TimeCostPolicy }",
        "| NotRequested",
        "requested_backend_inputs: RequestedBackendInputMap",
        "domain(requested_backend_inputs) == {memory,time}",
        "requested_backends == keys tagged MemoryRequested or TimeRequested",
        "NotRequested carries no payload and no digest",
        "memory-only request neither requires nor fingerprints CalibrationSet, CommunicationModelSnapshot or TimeCostPolicy",
        "time-only request neither requires nor fingerprints MemoryRegistrySnapshot",
        "RequestSnapshotBuildResult :=",
        "Ready { request_snapshot: RequestSnapshot }",
        "| Blocked { blockers: NonEmpty<BlockerRecord> }",
        "build_request_snapshot( common: CommonProductionInputs, configurations: NonEmptyOrderedMap<ConfigRef, CanonicalConfigEvaluationInput>, requested_backends: OrderedSet<memory | time>, requested_backend_inputs: RequestedBackendInputMap, comparison_request: ComparisonRequest | None ) -> RequestSnapshotBuildResult | InternalContractViolation",
        "build_request_snapshot gate specification equations:",
        "request_policy := common.blocker_scope_policy_snapshot",
        "request_specifications := common.gate_specification_set_snapshot",
        "request_runner := common.gate_runner_snapshot",
        "require request_specifications.blocker_scope_policy == request_policy",
        "require common.gate_specification_set_ref.gate_specification_set_digest == request_specifications.gate_specification_set_digest",
        "require entry.applicability_predicate_digest == request_runner.predicate_implementation_digests[entry.applicability_predicate_ref]",
        "require clause.predicate_implementation_digest == request_runner.predicate_implementation_digests[clause.exact_predicate_ref]",
        "negative fixture: E+P2 request specification embeds different blocker policy",
    ),
    "code-ir": (
        "RankCodeIRBuildInput:",
        "rank_build_input_digest :=",
        "TensorStorageRelation:",
        "CodeNode.storage_relations: [TensorStorageRelation]",
        "RankCodeIRBuildResult :=",
        "Ready { rank_build_input_digest: Digest",
        "| Blocked { rank_build_input_digest: Digest",
        "evaluate_source( input: RankCodeIRBuildInput ) -> RankCodeIRBuildResult | InternalContractViolation",
        "assemble_code_ir( rank_inputs: OrderedMap<LogicalRank, RankCodeIRBuildInput>, rank_results: OrderedMap<LogicalRank, RankCodeIRBuildResult> ) -> CodeIRBuildResult | InternalContractViolation",
        "rank_build_blocker_union(rank_inputs, rank_results) :=",
    ),
    "runtime-events": (
        "RuntimeRuleSnapshot:",
        "expand_runtime_semantics( code_ir: CodeIR, config: NormalizedParallelConfig, scenario: ExecutionScenario, registry: RuntimeRegistrySnapshot ) -> RuntimeBuildResult | InternalContractViolation",
    ),
    "plan-projection": (
        "SimulationPlanCore",
        "CoreBuildResult<SimulationPlanCore>",
        "ProjectionCandidate<MemoryEventView>",
        "ProjectionCandidate<TimeEventView>",
        "ProjectionBundleAuthority",
        "ProjectionBundleBuild",
        "ProjectionResult<MemoryEventView>",
        "ProjectionResult<TimeEventView>",
        "bind_core(",
        "evaluate_and_finalize_projection_bundle(",
        "build_memory_projection_candidate(",
        "build_time_projection_candidate(",
        "run_gate_domain(",
        "exactly one candidate per backend/evaluation identity",
        "shared Runtime/Core/G-IR blockers affect both faces",
        "face-local blockers affect only that face",
        "missing exact compute profile blocks time only",
        "missing memory-only storage/alias/lifetime/explicit-workspace facts blocks memory only",
        "missing shared compute shape yields BLK-MISSING-SHAPE and affects both faces",
        "unrequested arms are empty and unread",
        "blocker union/scope closure occurs before both results",
        "shared G-IR runs once",
        "ledgers append without prefix overwrite",
        "no capacity, completion-time or contention reads",
        "one canonical InternalContractViolation and no partial bundle",
    ),
    "memory-backend": (
        "MemoryProjectionSemanticsSnapshot:",
        "memory_projection_semantics_snapshot_digest := hash(canonical_payload_without_derived_digests( MemoryProjectionSemanticsSnapshot))",
        "unique_memory_projection_semantics_snapshot(MEMORY_BACKEND_SEMANTIC_VERSION)",
        "MemoryTimelineEntry:",
        "MemoryExecutionWitness:",
        "memory_execution_input_digest",
        "build_memory_projection_candidate( request: RequestSnapshot, evaluation_identity: EvaluationInstanceIdentity, core: SimulationPlanCore ) -> ProjectionCandidate<MemoryEventView> | InternalContractViolation",
        "memory_inputs := require MemoryRequested at request.requested_backend_inputs[memory]",
        "expected_memory_projection(",
        "memory_projection_semantics_snapshot: MemoryProjectionSemanticsSnapshot",
        "view.memory_projection_semantics_digest == memory_projection_semantics_snapshot.memory_projection_semantics_snapshot_digest",
        "run_memory_backend( view: MemoryEventView ) -> BackendExecution<MemoryEstimate, MemoryExecutionWitness>",
        'MEMORY_BACKEND_SEMANTIC_VERSION := "memory-backend/v1"',
        'MEMORY_REPLAY_SEMANTICS_VERSION := "logical-kernel-order-replay/v1"',
        "expected_memory_execution( view: MemoryEventView ) -> Completed<MemoryEstimate, MemoryExecutionWitness>:",
        "timeline[r] := replay(view.per_rank[r])",
        "value.per_rank[r].timeline == timeline[r]",
        "witness.per_rank_replay_order[r] == map(entry.position, timeline[r])",
        "witness.per_rank_timeline_digest[r] == hash(canonical timeline[r])",
        "witness.per_rank_terminal_live_storage_ids[r] == last(timeline[r]).live_storage_instance_ids",
        "witness.memory_execution_input_digest == hash(canonical(view))",
        "witness.memory_projection_semantics_digest == view.memory_projection_semantics_digest",
        "hash(MEMORY_REPLAY_SEMANTICS_VERSION, witness.memory_projection_semantics_digest)",
        "require run_memory_backend(view) == expected_memory_execution(view)",
        "INITIAL_STATE < all canonical MemoryEventId",
    ),
    "time-backend": (
        "RouteBinding:",
        "BoundP2P extends BoundCommunicationCommon:",
        "BoundCollective extends BoundCommunicationCommon:",
        "match_key: (src, dst, channel, sequence, bytes, payload_tensor_instances, protocol)",
        "node.common.endpoint_descriptors == [node.send_endpoint, node.recv_endpoint]",
        "node.common.endpoint_preimage == [node.send_endpoint.event_id, node.recv_endpoint.event_id]",
        "node.common.endpoint_descriptors == canonical(node.participant_endpoints)",
        "node.common.endpoint_preimage == canonical(event_id of node.participant_endpoints)",
        "route.communication_id == node.common.communication_id",
        "route.kind == branch_tag(node)",
        "route.endpoint_descriptors == node.common.endpoint_descriptors",
        "route.p2p_ordered_device_and_link_path == node.p2p_ordered_device_and_link_path",
        "route.collective_topology_and_participant_order == node.collective_topology_and_participant_order",
        "selected_formula_snapshot.algorithm is the only communication algorithm truth",
        "occupied_streams(node: BoundComputeEvent | BoundCommunication)",
        "BoundComputeEvent -> OrderedSet{node.physical_stream_id}",
        "BoundCommunication -> node.common.occupied_streams",
        "occupied_streams(expected_bound_compute(e, core, cost_bindings, stream_bindings)) == OrderedSet{stream_bindings[e.event_id].physical_stream_id}",
        "TimeTimelineEntry:",
        "TimeExecutionWitness:",
        "build_time_projection_candidate( request: RequestSnapshot, evaluation_identity: EvaluationInstanceIdentity, core: SimulationPlanCore ) -> ProjectionCandidate<TimeEventView> | InternalContractViolation",
        "time_inputs := require TimeRequested at request.requested_backend_inputs[time]",
        "run_time_backend( view: TimeEventView ) -> BackendExecution<StepTimeEstimate, TimeExecutionWitness>",
        "expected_time_execution( view: TimeEventView ) -> Completed<StepTimeEstimate, TimeExecutionWitness>:",
        "occupied_streams(node)) for node in order",
        "value.event_timeline == timeline",
        "value.critical_path == critical_path",
        "derive_time_aggregates_by_exact_10_4_equations(aggregate_interval_inputs)",
        "witness.projection_witness_digest == view.projection_witness_digest",
        "witness.stable_topological_order == order",
        "witness.predecessor_end_max_by_node == predecessor_end_max_by_node",
        "witness.critical_predecessor_by_node == critical_predecessor_by_node",
        "witness.step_end_node == step_end_node",
        "witness.timeline_digest == hash(canonical timeline)",
        "witness.aggregate_interval_inputs_digest == hash(canonical aggregate_interval_inputs)",
        "require run_time_backend(view) == expected_time_execution(view)",
        "CostBinding communication arm",
    ),
    "result-sealing": (
        "BackendReadyView := MemoryEventView | TimeEventView",
        "EstimateOf<MemoryEventView> := MemoryEstimate",
        "EstimateOf<TimeEventView> := StepTimeEstimate",
        "BackendExecution<T, W> :=",
        "Completed { value: T, witness: W }",
        "| InternalViolation { violation: InternalContractViolation }",
        "BackendSealBuildResult<T, V> :=",
        "Sealed { artifact: BackendSealArtifact<T, V> }",
        "canonical_internal_violation_union( inputs: NonEmpty<InternalContractViolation> ) -> InternalContractViolation:",
        "flatten every inputs[i].violations",
        "canonical_deduplicate_and_sort(flattened_contract_violations)",
        "不得把 InternalContractViolation 与 ContractViolation 混在同一层",
        "不得只取 first item",
        "run_backend_build_candidate_and_seal( source: BackendResultSourceAuthority<V> ) -> BackendSealBuildResult<EstimateOf<V>, V>",
        "require (source.backend, V, EstimateOf<V>) in { (memory, MemoryEventView, MemoryEstimate), (time, TimeEventView, StepTimeEstimate) }",
    ),
    "gate-system": (
        "BlockerScopePolicySnapshot:",
        "ScopePolicyRef := (blocker_scope_policy_digest, rule_id)",
        "GateEvaluationAuthority:",
        "GateManifest",
        "GateInvocationId",
        "GateClauseExecutionRecord",
        "GateExecutionLedger",
        "compile_gate_manifest( context: ProductionValidationContext, specifications: GateSpecificationSet, runner: GateRunnerSnapshot ) -> GateManifest",
        "run_gate_domain( authority: GateEvaluationAuthority, base: GateExecutionLedger, invocation_domain: OrderedSet<GateInvocationId> ) -> GateExecutionLedger | InternalContractViolation",
        "extend_gate_ledger_without_overwrite( authority: GateEvaluationAuthority, base: GateExecutionLedger, extension: GateExecutionRecordSet, current_stage_contexts: StageContextMap ) -> GateExecutionLedger | InternalContractViolation",
        "canonical_internal_violation_union(gate_internal_violations)",
        "clause_record.clause_id == clause_key == clause.clause_id",
        "effective_failure_disposition",
        "exact invocation domain",
        "gate_evaluation_context_digest",
        "canonical union of every InputBlocker failure occurrence",
        "InternalViolation takes priority",
        "must not be converted to InputBlocker",
        "GateSpecificationSet: schema_version blocker_scope_policy: BlockerScopePolicySnapshot",
        "manifest.blocker_scope_policy_digest == authority.blocker_scope_policy.blocker_scope_policy_digest",
        "manifest.blocker_scope_policy_digest == specifications.blocker_scope_policy.blocker_scope_policy_digest",
        "unique rule selected by scope_policy_ref.rule_id",
        "selected_rule.blocker_code == disposition.blocker_code",
        "negative fixture: policy digest mismatch",
        "negative fixture: dangling scope rule",
        "ExpectedInvocationDomain:",
        "required_prefix_invocations: OrderedSet<GateInvocationId>",
        "new_execution_invocations: OrderedSet<GateInvocationId>",
        "applicable_new_invocations: OrderedSet<GateInvocationId>",
        "current_root_stage: GateStage",
        "expected_invocation_domain( authority: GateEvaluationAuthority, base: GateExecutionLedger ) -> ExpectedInvocationDomain",
        "require invocation_domain == expected_domain.new_execution_invocations",
        "keys(base.records) == expected_domain.required_prefix_invocations",
        "keys(ledger.records) == expected_domain.required_prefix_invocations union expected_domain.new_execution_invocations",
        "ledger.stage_evaluation_context_digests == byte_preserved(base.stage_evaluation_context_digests) union expected_domain.current_stage_contexts",
        "negative fixture: invocation subset",
        "negative fixture: invocation superset",
        "negative fixture: wrong stage context",
        "ClauseCoverageRequirement:",
        "coverage_requirement: ClauseCoverageRequirement",
        "ClauseCoverageRequirement: require_positive_example: true require_negative_boundary_example: true // contains no fixture identifiers",
        "GateEvaluationSubject :=",
        "ProjectionBundleGateSubject { authority: ProjectionBundleAuthority }",
        "MemoryBackendValueGateSubject { authority: BackendValueGateAuthority<MemoryEstimate,MemoryEventView> }",
        "TimeBackendValueGateSubject { authority: BackendValueGateAuthority<StepTimeEstimate,TimeEventView> }",
        "ResultSealSubject<T,V>:",
        "ComparisonGateSubject { authority: ComparisonSealAuthority }",
        "GateSubjectCoordinates:",
        "derive_gate_subject_coordinates( subject: GateEvaluationSubject ) -> GateSubjectCoordinates",
        "build_gate_evaluation_authority( subject: GateEvaluationSubject, manifest: GateManifest, runner: GateRunnerSnapshot ) -> GateEvaluationAuthority | InternalContractViolation",
        "subject_request_snapshot(subject)",
        "request.canonical_production_evaluation_inputs.common_inputs.blocker_scope_policy_snapshot",
        "manifest.blocker_scope_policy == request_policy == authority.blocker_scope_policy",
        "gate_specification_set_digest: Digest := hash(canonical_payload_without_derived_digests(GateSpecificationSet))",
        "require manifest.gate_specification_set_digest == specifications.gate_specification_set_digest",
        "GateSpecificationSetRef:",
        "require manifest.entries == specifications.entries",
        "recompute context.production_subject.subject_digest",
        "recompute context.production_validation_context_digest",
        "recompute specifications.gate_specification_set_digest",
        "recompute specifications.blocker_scope_policy.blocker_scope_policy_digest",
        "recompute runner.verifier_runner_digest",
        "require manifest.subject_digest == context.production_subject.subject_digest",
        "require manifest.production_validation_context_digest == context.production_validation_context_digest",
        "require manifest.verifier_runner_digest == runner.verifier_runner_digest",
        "recompute manifest.gate_manifest_digest",
        "request_specifications := request.canonical_production_evaluation_inputs.common_inputs.gate_specification_set_snapshot",
        "request_specification_ref := request.canonical_production_evaluation_inputs.common_inputs.gate_specification_set_ref",
        "subject_context := subject_production_validation_context(subject)",
        "recompute subject_context.production_subject.subject_digest",
        "recompute subject_context.production_validation_context_digest",
        "manifest.subject_digest == subject_context.production_subject.subject_digest",
        "manifest.production_validation_context_digest == subject_context.production_validation_context_digest",
        "authority.gate_specification_set == request_specifications",
        "authority.gate_specification_set_ref == request_specification_ref",
        "manifest.entries == request_specifications.entries",
        "runtime builder compares request-owned ref and payload; it never trusts manifest self-report",
        "request_runner := request.canonical_production_evaluation_inputs.common_inputs.gate_runner_snapshot",
        "require runner == request_runner",
        "require entry.applicability_predicate_digest == runner.predicate_implementation_digests[entry.applicability_predicate_ref]",
        "require clause.predicate_implementation_digest == runner.predicate_implementation_digests[clause.exact_predicate_ref]",
        "negative fixture: runtime manifest uses wrong predicate runner",
        "ProjectionBundleAuthority.request_snapshot.canonical_production_evaluation_inputs.common_inputs.blocker_scope_policy_snapshot == request_policy",
        "current_root_stage := normative_current_root_stage(subject, base)",
        "new_execution_invocations := normative_stage_siblings(manifest.entries, current_root_stage)",
        "required_prefix_invocations := keys(base.records)",
        "applicable_dependency_closure := dependency_transitive_closure(manifest.entries, applicable_new_invocations)",
        "applicable_dependency_closure is a subset of required_prefix_invocations",
        "require every record in applicable_dependency_closure is Pass",
        "NotApplicable only records the non-applicable sibling itself",
        "NotApplicable never satisfies a dependency of an applicable invocation",
        "negative fixture: applicable invocation depends on NotApplicable sibling",
        "current_stage_contexts == singleton(current_root_stage, gate_evaluation_context_digest(subject))",
        "all normative backend/branch siblings in exactly current_root_stage",
        "dependency closure is validated only in the immutable base ledger",
        "base records and stage contexts are copied byte-for-byte",
        "no required prefix invocation is executed or pasted again",
        "i in expected_domain.applicable_new_invocations -> Pass | Fail",
        "i in expected_domain.new_execution_invocations - expected_domain.applicable_new_invocations -> NotApplicable",
        "negative fixture: memory-only request retains time view sibling as NotApplicable",
        "negative fixture: ProposedOk retains ProposedBlocked and ProposedNotRequested siblings as NotApplicable",
        "negative fixture: memory ProposedOk prefix keeps G-MEM5/6 value context V while result seal stage uses R only for new execution",
        "G-MEM5/6 records and value-stage context remain byte-identical V",
        "R appears only at result_seal_postcondition",
        "literal hand-derived expected domains",
        "must not call expected_invocation_domain",
        "negative fixture: wrong subject arm",
        "gate_evaluation_context_digest(subject: GateEvaluationSubject) :=",
        "ProjectionBundleGateSubject -> authority.projection_bundle_authority_digest",
        "MemoryBackendValueGateSubject | TimeBackendValueGateSubject -> authority.value_gate_authority_digest",
        "MemoryResultSealGateSubject | TimeResultSealGateSubject -> hash(subject.backend_seal_authority.backend_seal_authority_digest, subject.backend_result_candidate.backend_result_candidate_digest)",
        "ComparisonGateSubject -> hash(authority.comparison_seal_authority_digest, authority.result_candidate.comparison_result_candidate_digest)",
        "gate_record.evaluation_input_digest == gate_evaluation_context_digest(subject)",
        "clause_record.evaluation_input_digest == gate_evaluation_context_digest(subject)",
        "legal ResultSealSubject context == hash(backend_seal_authority_digest, backend_result_candidate_digest)",
        "legal ComparisonGateSubject context == hash(comparison_seal_authority_digest, comparison_result_candidate_digest)",
        "base.stage_evaluation_context_digests[backend_value_postcondition] == V",
        "expected.current_stage_contexts[result_seal_postcondition] == R",
    ),
    "comparison": (
        "CanonicalSchemaPathSet:",
        "ComparisonSchemaSnapshot:",
        "comparison_schema_snapshot_digest: Digest",
        "declared_paths: OrderedSet<CanonicalSchemaPath>",
        "derivation_closure == derive_closure(snapshot, declared_paths)",
        "BasisMismatch:",
        "CoverageDelta:",
        "ComparisonBasisPair",
        "ComparisonSourceAuthority",
        "ComparisonResultCandidate",
        "BackendSealArtifact",
        "derive_comparison_basis_pair( request: RequestSnapshot, left: ComparisonArmAuthority, right: ComparisonArmAuthority, metric: memory | time ) -> ComparisonBasisPair",
        "build_comparison_source_authority( request: RequestSnapshot, left: ComparisonArmAuthority, right: ComparisonArmAuthority, basis_pairs: OrderedMap<memory | time, ComparisonBasisPair>, context: ProductionValidationContext, manifest: GateManifest ) -> ComparisonSourceAuthority",
        "compare_per_metric_from_authority( source: ComparisonSourceAuthority ) -> ComparisonResult",
        "same RequestSnapshot",
        "request.canonical_production_evaluation_inputs.common_inputs.comparison_schema_snapshot",
        "derive_comparison_basis_pair recomputes",
        "G-REP2 recomputes",
        "negative fixture: inject other_config_indexed_production_inputs.x into derivation closure",
        "BasisMismatch.schema_path == other_config_indexed_production_inputs.x",
        "config_ref",
        "evaluation_instance_digest",
        "canonical_comparison_basis_pairs",
        "ComparableDelta",
        "Incomparable",
        "Unavailable",
        "NotRequested",
        "UndefinedZeroBaseline",
        "registry/calibration/fallback/assumption",
    ),
    "conformance": (
        "TraceFixture:",
        "FixtureSet:",
        "ConformanceFinding:",
        "ApprovedDigestSet:",
        "AttestedConformanceReport",
        "ReleaseDecision",
        "run_basic_offline_conformance( authority: BasicOfflineConformanceAuthority ) -> BasicOfflineRunResult",
        "run_attested_release_conformance( authority: AttestedConformanceInvocationAuthority, profile: AttestedReleaseConformance, measured_environment: MeasuredExecutionEnvironment ) -> AttestedConformanceRunResult",
        "apply_release_policy( artifact: AttestedConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy, profile: AttestedReleaseConformance ) -> ReleasePolicyApplicationResult",
        "derive_approved_digest_set( artifact: AttestedConformanceSealArtifact, policy: ReleasePolicy ) -> ApprovedDigestSet",
        "derive_approved_digest_set fields:",
        "subject_digest := recompute artifact.invocation_authority.production_subject.subject_digest",
        "fixture_set_digest := recompute artifact.invocation_authority.fixture_set.fixture_set_digest",
        "validation_policy_digest := recompute artifact.invocation_authority.validation_policy.validation_policy_digest",
        "gate_manifest_digest := recompute artifact.invocation_authority.gate_manifest.gate_manifest_digest",
        "verifier_runner_digest := recompute artifact.invocation_authority.verifier_runner.verifier_runner_digest",
        "runner_attestation_digest := recompute artifact.runner_attestation.runner_attestation_digest",
        "conformance_execution_ledger_digest := recompute artifact.execution_ledger.conformance_execution_ledger_digest",
        "conformance_report_digest := recompute artifact.report.conformance_report_digest",
        "conformance_seal_artifact_digest := recompute artifact.conformance_seal_artifact_digest",
        "release_policy_digest := recompute policy.release_policy_digest",
        "derived_approved := derive_approved_digest_set(artifact, policy)",
        "require approved == derived_approved",
        "approved_mismatches := canonical_schema_path_diff(approved, derived_approved)",
        "map(approved_mismatches, path -> ApprovedDigestMismatch(path))",
        "all ten approved digest fields",
        "positive fixture",
        "negative-boundary fixture",
        "subject_digest",
        "fixture_set_digest",
        "validation_policy_digest",
        "gate_manifest_digest",
        "verifier_runner_digest",
        "release_policy_digest",
        "outside production digests and cache keys",
        "does not alter BackendResult",
        "FixtureBinding:",
        "bindings: OrderedMap<FixtureRef, FixtureBinding>",
        "FixtureSet is the sole fixture-to-(invocation, clause) mapping",
        "FixtureSet: bindings: OrderedMap<FixtureRef, FixtureBinding> derived_coverage_by_clause := derive_fixture_coverage(bindings)",
        "TrustStoreSnapshot:",
        "trusted_key_material_by_id: OrderedMap<AttestationKeyId, TrustedPublicKeyMaterial>",
        "key_registry_digest := hash(canonical trusted_key_material_by_id)",
        "supported_signature_schemes: OrderedSet<SignatureScheme>",
        "RunnerAttestationPolicy:",
        "expected_execution_environment_digest",
        "trusted_attestation_key_id, attestation_signature_scheme",
        "trust_store_snapshot_digest",
        "runner_attestation_policy_digest := hash(canonical_payload_without_derived_digests(RunnerAttestationPolicy))",
        "RunnerAttestation:",
        "signed_message := canonical_tuple( measured_execution_environment_digest, executable_artifact_digest, trusted_attestation_key_id, attestation_signature_scheme, conformance_invocation_digest, observed_output_digest, verifier_runner_digest)",
        "AttestedConformanceExecutionRecord:",
        "AttestedConformanceExecutionLedger:",
        "conformance_execution_record_digest :=",
        "AttestedConformanceRunResult := Completed { artifact: AttestedConformanceSealArtifact } | InternalViolation { violation: InternalContractViolation }",
        'hash("attested-conformance-seal-artifact/v1", canonical_payload_without_derived_digests(AttestedConformanceSealArtifact))',
        "runner crash, schema failure, digest failure or conservation failure",
        "complete, valid execution",
        "record.fixture_ref == record_key",
        "record.target_invocation_id == binding.target_invocation_id",
        "record.target_clause_id == binding.target_clause_id",
        "AttestedConformanceInvocationAuthority:",
        "conformance_invocation_digest :=",
        'evaluation_input_digest := hash("attested-conformance-record-input/v1", authority.conformance_invocation_digest, session.measured_execution_environment_digest, canonical(binding), binding.fixture_binding_digest) observed_clause_output: ObservedClauseOutput',
        "ConformanceObservedOutput:",
        "ConformanceObservedOutput: observed_clause_outputs: OrderedMap<FixtureRef, ObservedClauseOutput> findings: OrderedMap<FindingId, ConformanceFinding> coverage_gaps: OrderedSet<GateClauseId> observed_output_digest := hash(canonical observed_clause_outputs, findings, coverage_gaps)",
        "ledger.observed_output_digest == report.observed_output_digest == attestation.observed_output_digest",
        "verifier_runner_digest := hash(canonical_payload_without_derived_digests(VerifierRunner))",
        "trust_store_snapshot_ref: TrustStoreSnapshotRef",
        "runner_attestation_policy_ref: RunnerAttestationPolicyRef",
        "ConformanceTrustRootCapability: opaque deployment/verifier-owned capability",
        "cannot be constructed by request deserialization",
        "expected_trust_store_snapshot: TrustStoreSnapshot",
        "expected_runner_attestation_policy: RunnerAttestationPolicy",
        "MeasuredExecutionEnvironment:",
        "protected_environment_measurer",
        "opaque protected_environment_measurer-issued non-replayable measurement envelope",
        "conformance_invocation_digest",
        "verifier_runner_digest",
        "executable_artifact_digest",
        "execution_session_id: ExecutionSessionId",
        "process_identity: ProcessIdentity",
        "container_identity: ContainerIdentity | BareProcess",
        "verifier_nonce: SingleUseVerifierNonce",
        "execution_time_window: ClosedInterval<MonotonicTimestamp>",
        "measured_environment_payload",
        "measurer_identity_and_freshness_evidence",
        "measured_execution_environment_digest := hash(canonical_payload_without_derived_digests(MeasuredExecutionEnvironment))",
        "ValidatedMeasurementSessionCapability:",
        "opaque call-scoped, non-serializable and non-transferable capability",
        "validate_and_consume_measurement_envelope( authority: AttestedConformanceInvocationAuthority, runner: VerifierRunner, trust_root: ConformanceTrustRootCapability, measured_environment: MeasuredExecutionEnvironment ) -> ValidatedMeasurementSessionCapability | InternalContractViolation",
        "(store, policy) := resolve_conformance_trust_root(trust_root)",
        "authority.trust_store_snapshot_ref.trust_store_snapshot_digest == store.trust_store_snapshot_digest",
        "authority.runner_attestation_policy_ref.runner_attestation_policy_digest == policy.runner_attestation_policy_digest",
        "policy.attestation_signature_scheme in store.supported_signature_schemes",
        "trusted_key_material := store.trusted_key_material_by_id[policy.trusted_attestation_key_id]",
        "verify_signature(trusted_key_material, policy.attestation_signature_scheme, signed_message, attestation.signature)",
        "measurement_envelope_digest := recompute measured_environment.measured_execution_environment_digest",
        "measurement_envelope_digest == hash(canonical_payload_without_derived_digests(measured_environment))",
        "measured_environment.conformance_invocation_digest == authority.conformance_invocation_digest",
        "measured_environment.verifier_runner_digest == runner.verifier_runner_digest",
        "measured_environment.executable_artifact_digest == runner.executable_artifact_digest",
        "current_execution_session_context() == canonical_tuple(measured_environment.execution_session_id, measured_environment.process_identity, measured_environment.container_identity)",
        "terminal_consumption_receipt_key( environment: MeasuredExecutionEnvironment ) := canonical_tuple( environment.verifier_nonce, environment.conformance_invocation_digest, environment.verifier_runner_digest, environment.executable_artifact_digest, environment.execution_session_id, environment.process_identity, environment.container_identity, environment.execution_time_window, environment.measured_execution_environment_digest)",
        "atomically consumed exactly once while persisting a terminal receipt under that exact key",
        "session.nonce_consumption_receipt proves the terminal registry contains exact terminal_consumption_receipt_key(measured_environment)",
        "verify_measurer_identity_and_freshness_evidence(",
        "measured_payload_digest := hash(canonical measured_environment.measured_environment_payload)",
        "measured_payload_digest == policy.expected_execution_environment_digest",
        "execute the exact binding once under session.capability",
        "session.measured_execution_environment_digest",
        "attestation.measured_execution_environment_digest == measurement_envelope_digest",
        "runner signs canonical_tuple( measurement_envelope_digest",
        "attestation.executable_artifact_digest == runner.executable_artifact_digest",
        "attestation.trusted_attestation_key_id == policy.trusted_attestation_key_id",
        "attestation.attestation_signature_scheme == policy.attestation_signature_scheme",
        "validate_and_seal_attested_conformance( authority: AttestedConformanceInvocationAuthority, trust_root: ConformanceTrustRootCapability, session: ValidatedMeasurementSessionCapability, observed: ConformanceObservedOutput, ledger: AttestedConformanceExecutionLedger, attestation: RunnerAttestation ) -> AttestedConformanceRunResult",
        "validate_and_seal_attested_conformance first operation:",
        "recompute authority.production_subject.subject_digest",
        "recompute every authority.fixture_set binding, nested fixture and fixture_set_digest",
        "recompute authority.validation_policy.validation_policy_digest",
        "recompute authority.gate_manifest and every nested entry/clause digest",
        "recompute authority.gate_manifest.blocker_scope_policy.blocker_scope_policy_digest",
        "recompute authority.verifier_runner.verifier_runner_digest",
        "recompute authority.conformance_invocation_digest only after all nested recomputations",
        "negative fixture: old invocation digest plus substituted policy or trust-store reference",
        "derive_attested_conformance_report( authority: AttestedConformanceInvocationAuthority, observed: ConformanceObservedOutput, ledger: AttestedConformanceExecutionLedger, attestation: RunnerAttestation ) -> AttestedConformanceReport",
        "AttestedConformanceReport.findings: OrderedMap<FindingId, ConformanceFinding>",
        "report.findings == observed.findings",
        "failure_findings := policy_classified_failure_findings(validation_policy, findings)",
        "verdict := fail iff failure_findings is non-empty",
        "else insufficient iff exact_evidence_gaps or observed.coverage_gaps is non-empty",
        "else pass",
        "execute -> observed -> runner attestation -> ledger -> derived report -> seal",
        "negative fixture: valid signature plus failing observed output cannot be sealed with forged pass",
        "ReleaseApprovalTrustRootCapability: opaque deployment/verifier-owned capability",
        "expected_release_policy: ReleasePolicy",
        "expected_release_policy_digest :=",
        "expected_release_approval_store_snapshot: ReleaseApprovalStoreSnapshot",
        "expected_release_approval_store_snapshot_digest :=",
        "supported_release_approval_signature_schemes: OrderedSet<SignatureScheme>",
        "trusted_approver_public_key_material_by_id: OrderedMap<ApproverKeyId, TrustedPublicKeyMaterial>",
        "cannot be constructed by request deserialization, fixture, policy, approval store or approval artifact",
        "release_trust_root := profile.release_trust_root",
        "apply_release_policy first operation:",
        "recompute artifact and every nested invocation/report/ledger/attestation/environment digest",
        "recompute approval_authority.store_snapshot and every nested ApprovedDigestSet value",
        "recompute policy.release_policy_digest",
        "recompute approval_authority.release_approval_authority_digest",
        "recompute approval.release_approval_artifact_digest",
        "before resolving either profile trust root key material, verifying any signature or reading verdict",
        "root := resolve_release_approval_trust_root(release_trust_root)",
        "root.expected_release_policy_digest == policy.release_policy_digest",
        "canonical(root.expected_release_policy) == canonical(policy)",
        "root.expected_release_approval_store_snapshot_digest == approval_authority.store_snapshot.release_approval_store_snapshot_digest",
        "canonical(root.expected_release_approval_store_snapshot) == canonical(approval_authority.store_snapshot)",
        "require approval_authority.approval_signature_scheme == policy.approval_signature_scheme",
        "policy.approval_signature_scheme in root.supported_release_approval_signature_schemes",
        "trusted_approver_public_key_material := root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]",
        "release_approval_signed_message := canonical_tuple(",
        "canonical(approved)",
        "verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)",
        "negative fixture: release approval wrong signature scheme",
        "negative fixture: unsupported release approval signature scheme",
        "negative fixture: release approval self-owned approver key",
        "negative fixture: release approval wrong store snapshot",
        "negative fixture: release approval unsupported signature scheme",
        "negative fixture: release approval substituted policy",
        "ReleaseApprovalStoreSnapshot:",
        "ReleaseApprovalAuthority:",
        "ReleaseApprovalArtifact:",
        "verify release approval signature",
        "negative fixture: no execution but forged all-pass",
        "negative fixture: forged runner attestation",
        "negative fixture: forged ApprovedDigestSet",
        "negative fixture: tampered execution record or observed output",
        "negative fixture: wrong runner attestation trust store",
        "negative fixture: unsupported runner attestation signature scheme",
        "negative fixture: tampered runner execution environment",
        "negative fixture: runner self-owned attestation key",
        "negative fixture: replay valid measurement envelope from another execution session",
        "negative fixture: change only measurer identity or freshness evidence while retaining old envelope digest",
        "change only measurer_identity_and_freshness_evidence changes measured_execution_environment_digest",
    ),
}

MODULE_CONTRACT_REQUIRED_OCCURRENCES = {
    "conformance": {
        "supported_signature_schemes: OrderedSet<SignatureScheme>": 2,
        "expected_execution_environment_digest": 5,
        "trusted_attestation_key_id, attestation_signature_scheme": 2,
        "trust_store_snapshot_digest": 21,
        "cannot be constructed by request deserialization": 6,
        "protected_environment_measurer": 1,
        "(store, policy) := resolve_conformance_trust_root(trust_root)": 3,
        "authority.trust_store_snapshot_ref.trust_store_snapshot_digest == store.trust_store_snapshot_digest": 2,
        "authority.runner_attestation_policy_ref.runner_attestation_policy_digest == policy.runner_attestation_policy_digest": 2,
        "policy.attestation_signature_scheme in store.supported_signature_schemes": 2,
        "trusted_key_material := store.trusted_key_material_by_id[policy.trusted_attestation_key_id]": 2,
        "measurement_envelope_digest := recompute measured_environment.measured_execution_environment_digest": 2,
        "measurement_envelope_digest == hash(canonical_payload_without_derived_digests(measured_environment))": 2,
        "measured_environment.conformance_invocation_digest == authority.conformance_invocation_digest": 2,
        "measured_environment.verifier_runner_digest == runner.verifier_runner_digest": 2,
        "measured_environment.executable_artifact_digest == runner.executable_artifact_digest": 2,
        "current_execution_session_context() == canonical_tuple(measured_environment.execution_session_id, measured_environment.process_identity, measured_environment.container_identity)": 2,
        "verify_measurer_identity_and_freshness_evidence(": 3,
        "measured_payload_digest := hash(canonical measured_environment.measured_environment_payload)": 2,
        "measured_payload_digest == policy.expected_execution_environment_digest": 2,
        "measurer_identity_and_freshness_evidence": 8,
        "session.measured_execution_environment_digest": 4,
    },
}

MODULE_CONTRACT_FORBIDDEN_TEXT = {
    "gate-system": (
        "GateExecutionLedger | NonEmpty<InternalContractViolation>",
        "positive_fixture_refs",
        "negative_boundary_fixture_refs",
        "expected_invocation_domain(authority).audit_invocations",
        "audit_stage_sibling_dependency_closure(manifest.entries, root_stages)",
        "every derived stage context := gate_evaluation_context_digest(subject)",
        "builder validates only manifest, request-owned policy, typed subject and runner",
        "manifest entries may differ from specifications entries",
        "trust manifest gate_specification_set_digest without request payload",
    ),
    "time-backend": (
        "selected_algorithm",
        "match_key: (src, dst, channel, sequence, bytes, payload, protocol)",
    ),
    "result-sealing": (
        "InternalViolation { violations: NonEmpty<InternalContractViolation> }",
        "EstimateOf<MemoryEventView> := StepTimeEstimate",
        "use only first(inputs).violations",
    ),
    "conformance": (
        "positive_fixture_refs",
        "negative_boundary_fixture_refs",
        "validate_and_seal_attested_conformance( authority: AttestedConformanceInvocationAuthority, observed: ConformanceObservedOutput, ledger: AttestedConformanceExecutionLedger, report: AttestedConformanceReport",
        "apply_release_policy( report: AttestedConformanceReport",
        "run_attested_release_conformance( subject: ProductionSubject",
        "apply_release_policy( artifact: AttestedConformanceSealArtifact, approved: ApprovedDigestSet",
        "apply_release_policy( artifact: AttestedConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy ) -> ReleaseDecision",
        "verify_signature(runner.trusted_attestation_key_id, runner.attestation_signature_scheme",
        "VerifierRunner: runner_id, runner_version, executable_artifact_digest trusted_attestation_key_id, attestation_signature_scheme",
        "measured_execution_environment_digest := hash(canonical measured_environment_payload)",
        "signed_message := canonical_tuple( execution_environment_digest",
        "attestation.execution_environment_digest == policy.expected_execution_environment_digest",
        "verify_signature(policy.trusted_approver_key_id",
        "policy.approval_signature_scheme in SUPPORTED_RELEASE_APPROVAL_SIGNATURE_SCHEMES",
    ),
}

DERIVED_DIGEST_EXCLUSIONS = (
    ("SourceFileSnapshot", "content_digest"),
    ("SourceSnapshot", "source_snapshot_digest"),
    ("StructureRegistrySnapshot", "structure_registry_digest"),
    ("RuntimeRegistrySnapshot", "runtime_registry_digest"),
    ("RankCodeIRBuildInput", "rank_build_input_digest"),
    ("RuntimeRuleSnapshot", "rule_digest"),
    ("CodeIR", "model_digest"),
    ("RuntimeEventPlan", "runtime_plan_digest"),
    ("SimulationPlanCore", "simulation_core_digest"),
    ("ExecutionDeployment", "deployment_digest"),
    ("HardwareBindingPolicySnapshot", "policy_digest"),
    ("MeasurementProtocol", "protocol_digest"),
    ("CalibrationTrainManifest", "calibration_train_manifest_digest"),
    ("TimeCostPolicy", "policy_digest"),
    ("NumericPolicySnapshot", "numeric_policy_digest"),
    ("CommunicationFormulaRef", "digest"),
    ("CommunicationModelSnapshot", "communication_model_digest"),
    ("MemoryRegistrySnapshot", "memory_registry_digest"),
    ("ResolvedEventSemantic", "resolved_semantic_digest"),
    ("KernelVariantBinding", "kernel_variant_binding_digest"),
    ("StreamAssignmentPolicySnapshot", "policy_digest"),
    ("StreamBinding", "stream_binding_digest"),
    ("StorageBinding", "storage_binding_digest"),
    ("WorkspaceBinding", "workspace_binding_digest"),
    ("RouteBinding", "route_binding_digest"),
    ("CostBinding", "cost_binding_digest"),
    (
        "MemoryProjectionSemanticsSnapshot",
        "memory_projection_semantics_snapshot_digest",
    ),
    ("EstimateContext", "estimate_context_digest"),
    ("RequestSnapshot", "request_digest"),
    ("EvaluationInstanceIdentity", "evaluation_instance_digest"),
    ("ProductionSubject", "subject_digest"),
    ("ProductionValidationContext", "production_validation_context_digest"),
    ("ProjectionCandidate", "projection_candidate_digest"),
    ("ProjectionBundleAuthority", "projection_bundle_authority_digest"),
    ("ComparisonSchemaSnapshot", "comparison_schema_snapshot_digest"),
    ("CanonicalSchemaPathSet", "canonical_schema_path_set_digest"),
    ("ComparisonBasis", "comparison_basis_digest"),
    ("CoverageDelta", "coverage_delta_digest"),
    ("ComparisonSourceAuthority", "comparison_source_authority_digest"),
    ("ComparisonResultCandidate", "comparison_result_candidate_digest"),
    ("ComparisonSealAuthority", "comparison_seal_authority_digest"),
    ("ComparisonSealArtifact", "comparison_seal_artifact_digest"),
    ("BlockerScopePolicySnapshot", "blocker_scope_policy_digest"),
    ("GateSpecificationSet", "gate_specification_set_digest"),
    ("GateRunnerSnapshot", "verifier_runner_digest"),
    ("GateEvaluationAuthority", "gate_evaluation_authority_digest"),
    ("GateManifest", "gate_manifest_digest"),
    ("GateExecutionLedger", "gate_execution_ledger_digest"),
    ("EstimateCandidate", "estimate_candidate_digest"),
    ("BackendResultSourceAuthority", "source_authority_digest"),
    ("BackendValueGateAuthority", "value_gate_authority_digest"),
    ("BackendResultCandidate", "backend_result_candidate_digest"),
    ("BackendSealAuthority", "backend_seal_authority_digest"),
    ("BackendSealArtifact", "backend_seal_artifact_digest"),
    ("TraceFixture", "fixture_digest"),
    ("FixtureBinding", "fixture_binding_digest"),
    ("FixtureSet", "fixture_set_digest"),
    ("ValidationPolicy", "validation_policy_digest"),
    ("VerifierRunner", "verifier_runner_digest"),
    ("BasicOfflineConformanceAuthority", "basic_offline_authority_digest"),
    ("BasicOfflineExecutionRecord", "basic_offline_execution_record_digest"),
    ("BasicOfflineExecutionLedger", "basic_offline_execution_ledger_digest"),
    ("BasicOfflineReport", "basic_offline_report_digest"),
    ("BasicOfflineReportArtifact", "basic_offline_report_artifact_digest"),
    ("TrustStoreSnapshot", "trust_store_snapshot_digest"),
    ("RunnerAttestationPolicy", "runner_attestation_policy_digest"),
    ("MeasuredExecutionEnvironment", "measured_execution_environment_digest"),
    ("AttestedConformanceInvocationAuthority", "conformance_invocation_digest"),
    ("RunnerAttestation", "runner_attestation_digest"),
    ("AttestedConformanceExecutionRecord", "conformance_execution_record_digest"),
    ("ConformanceObservedOutput", "observed_output_digest"),
    ("AttestedConformanceExecutionLedger", "conformance_execution_ledger_digest"),
    ("ReleaseApprovalStoreSnapshot", "release_approval_store_snapshot_digest"),
    ("ReleaseApprovalAuthority", "release_approval_authority_digest"),
    ("ReleaseApprovalArtifact", "release_approval_artifact_digest"),
    ("AttestedConformanceSealArtifact", "conformance_seal_artifact_digest"),
    ("AttestedConformanceReport", "conformance_report_digest"),
    ("ReleasePolicy", "release_policy_digest"),
    ("Estimate", "result_digest"),
)

PRODUCT_GATE_STAGING_REQUIRED_TEXT = (
    "projection_gate_subject := ProjectionBundleGateSubject { authority }",
    "projection_gate_ledger := canonical_empty_gate_ledger(projection_gate_authority)",
    "for expected_stage in [structure_prerequisite, memory_view_prerequisite, time_view_prerequisite]:",
    "projection_gate_ledger := run_gate_domain(",
    "unrequested memory/time view stage still executes its full sibling domain as NotApplicable",
    "run_backend_value_gate_stage(value_authority):",
    "value_gate_runtime_authority := build_gate_evaluation_authority(",
    "pre_seal_ledger := run_gate_domain(",
    "result_seal_gate_subject := exact MemoryResultSealGateSubject or",
    "result_seal_runtime_authority := build_gate_evaluation_authority(",
    "sealed_gate_ledger := run_gate_domain(",
    "comparison_gate_subject := ComparisonGateSubject { authority: seal_authority }",
    "comparison_base := canonical_empty_gate_ledger(comparison_runtime_authority)",
    "comparison_gate_ledger := run_gate_domain(",
)


class _ModuleContractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.contracts: list[dict[str, object]] = []
        self.current: dict[str, object] | None = None
        self.section_depth = 0
        self.heading: tuple[str, str, list[str]] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attr_map = {name: value or "" for name, value in attrs}
        if tag == "section":
            classes = set(attr_map.get("class", "").split())
            if self.current is None and "module-contract" in classes:
                self.current = {
                    "attrs": attr_map,
                    "headings": [],
                    "text_parts": [],
                }
                self.section_depth = 1
                return
            if self.current is not None:
                self.section_depth += 1
        if self.current is not None and re.fullmatch(r"h[3-6]", tag):
            self.heading = (tag, attr_map.get("id", ""), [])

    def handle_data(self, data: str) -> None:
        if self.current is None:
            return
        text_parts = self.current["text_parts"]
        assert isinstance(text_parts, list)
        text_parts.append(data)
        if self.heading is not None:
            self.heading[2].append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.current is None:
            return
        if self.heading is not None and tag == self.heading[0]:
            level, heading_id, parts = self.heading
            headings = self.current["headings"]
            assert isinstance(headings, list)
            headings.append((level, heading_id, _normalize_contract_text(parts)))
            self.heading = None
        if tag == "section":
            self.section_depth -= 1
            if self.section_depth == 0:
                self.contracts.append(self.current)
                self.current = None


class _ConformanceLocalBlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.conformance_depth: int | None = None
        self.active: dict[str, object] | None = None
        self.blocks: list[dict[str, object]] = []
        self.outside_parts: list[str] = []
        self.malformed = False

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.depth += 1
        attr_map = {name: value or "" for name, value in attrs}
        markers = [
            ("scope", attr_map["data-conformance-scope"])
            for _ in [0]
            if "data-conformance-scope" in attr_map
        ] + [
            ("profile", attr_map["data-conformance-profile"])
            for _ in [0]
            if "data-conformance-profile" in attr_map
        ]
        if (
            tag == "section"
            and "module-contract" in attr_map.get("class", "").split()
            and attr_map.get("data-module") == "conformance"
        ):
            if self.conformance_depth is not None:
                self.malformed = True
            self.conformance_depth = self.depth

        if markers and (self.conformance_depth is None or tag != "div"):
            self.malformed = True
        if self.conformance_depth is None:
            return
        if len(markers) > 1 or (markers and self.active is not None):
            self.malformed = True
        if len(markers) == 1:
            kind, value = markers[0]
            self.active = {
                "kind": kind,
                "value": value,
                "tag": tag,
                "depth": self.depth,
                "text_parts": [],
            }

    def handle_data(self, data: str) -> None:
        if self.active is not None:
            parts = self.active["text_parts"]
            assert isinstance(parts, list)
            parts.append(data)
        elif self.conformance_depth is not None:
            self.outside_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if (
            self.active is not None
            and self.active["tag"] == tag
            and self.active["depth"] == self.depth
        ):
            self.blocks.append(self.active)
            self.active = None
        if tag == "section" and self.conformance_depth == self.depth:
            if self.active is not None:
                self.malformed = True
            self.conformance_depth = None
        self.depth -= 1


def _normalize_contract_text(parts: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def check_module_contracts(html: str, errors: list[str]) -> None:
    parser = _ModuleContractParser()
    parser.feed(html)
    by_module: dict[str, list[dict[str, object]]] = {}
    for contract in parser.contracts:
        attrs = contract["attrs"]
        assert isinstance(attrs, dict)
        module = str(attrs.get("data-module", ""))
        by_module.setdefault(module, []).append(contract)

    unexpected_modules = sorted(
        set(by_module) - set(MODULE_CONTRACT_REQUIRED_TEXT)
    )
    for module in unexpected_modules:
        errors.append(f"unexpected module-contract: {module}")

    for module, required_texts in MODULE_CONTRACT_REQUIRED_TEXT.items():
        matches = by_module.get(module, [])
        if len(matches) != 1:
            errors.append(
                f"module-contract {module} 应恰出现一次，实际 {len(matches)} 次"
            )
            continue
        contract = matches[0]
        attrs = contract["attrs"]
        headings = contract["headings"]
        text_parts = contract["text_parts"]
        assert isinstance(attrs, dict)
        assert isinstance(headings, list)
        assert isinstance(text_parts, list)

        labelled_by = str(attrs.get("aria-labelledby", ""))
        heading_ids = {heading_id for _, heading_id, _ in headings}
        if not labelled_by or labelled_by not in heading_ids:
            errors.append(
                f"module-contract {module} 的 aria-labelledby 未引用本节标题"
            )
        subsection_titles = {
            title for level, _, title in headings if level in {"h4", "h5"}
        }
        for subsection in sorted(MODULE_CONTRACT_SUBSECTIONS):
            if subsection not in subsection_titles:
                errors.append(
                    f"module-contract {module} 缺少局部子节：{subsection}"
                )

        local_text = _normalize_contract_text(text_parts)
        for required_text in required_texts:
            if required_text not in local_text:
                errors.append(
                    f"module-contract {module} 缺少局部契约：{required_text}"
                )
        for required_text, required_count in MODULE_CONTRACT_REQUIRED_OCCURRENCES.get(
            module, {}
        ).items():
            actual_count = local_text.count(required_text)
            if actual_count != required_count:
                errors.append(
                    f"module-contract {module} occurrence contract mismatch: "
                    f"{required_text}; expected {required_count}, actual {actual_count}"
                )
        for forbidden_text in MODULE_CONTRACT_FORBIDDEN_TEXT.get(module, ()):
            if forbidden_text in local_text:
                errors.append(
                    f"module-contract {module} 残留局部禁用语义：{forbidden_text}"
                )

        if module == "input-facts":
            common_match = re.search(
                r"CommonProductionInputs:\s*(.*?)\s*RequestSnapshotBuildResult\s*:=",
                local_text,
            )
            if common_match is None:
                errors.append(
                    "module-contract input-facts 的 CommonProductionInputs schema 不可局部提取"
                )
            else:
                common_schema = common_match.group(1)
                for backend_field in (
                    "memory_registry_snapshot",
                    "calibration_set",
                    "calibration_train_manifest_snapshot",
                    "communication_model_snapshot",
                    "time_cost_policy",
                ):
                    if backend_field in common_schema:
                        errors.append(
                            "module-contract input-facts 的 CommonProductionInputs "
                            f"混入 backend-specific field：{backend_field}"
                        )

        if module == "time-backend":
            execution_match = re.search(
                r"expected_time_execution\(\s*view:\s*TimeEventView\s*\)"
                r"(.*?)require run_time_backend\(view\) == expected_time_execution\(view\)",
                local_text,
            )
            if execution_match is None:
                errors.append(
                    "module-contract time-backend 的 expected_time_execution 不可局部提取"
                )
            elif "node.occupied_streams" in execution_match.group(1):
                errors.append(
                    "module-contract time-backend 的 timeline 必须调用 occupied_streams(node)"
                )


def _flat_schema_fields(html: str, schema_name: str) -> dict[str, str] | None:
    match = re.search(
        rf"(?m)^{re.escape(schema_name)}:\s*\r?\n"
        rf"(?P<body>(?:  [^\r\n]*(?:\r?\n|$))*)",
        html,
    )
    if match is None:
        return None
    fields: dict[str, str] = {}
    for name, field_type in re.findall(
        r"(?m)^  ([a-z][a-z0-9_]*):\s*([^\r\n/]+?)(?:\s*//.*)?$",
        match.group("body"),
    ):
        fields[name] = field_type.strip()
    return fields


def _exact_flat_schema_fields(html: str, schema_name: str) -> dict[str, str] | None:
    match = re.search(
        rf"(?m)^{re.escape(schema_name)}:\s*\r?\n"
        rf"(?P<body>(?:  [^\r\n]+(?:\r?\n|$))*)",
        html,
    )
    if match is None:
        return None
    fields: dict[str, str] = {}
    for line in match.group("body").splitlines():
        field = re.fullmatch(
            r"  ([a-z][a-z0-9_]*):\s*(\S(?:.*\S)?)\s*",
            line,
        )
        if field is None or field.group(1) in fields:
            return None
        fields[field.group(1)] = unescape(field.group(2))
    return fields


def _task9_owner_block(document_text: str, schema_name: str) -> str | None:
    owner_pattern = re.escape(schema_name)
    if "<" not in schema_name:
        owner_pattern += r"(?:<[^>\r\n]+>)?"
    match = re.search(
        rf"(?m)(?:^|>){owner_pattern}:[ \t]*\r?\n"
        rf"(?P<body>(?:[ \t]+[^\r\n]*(?:\r?\n|$))*)",
        document_text,
    )
    return match.group("body") if match is not None else None


def _task9_strip_line_comments(contract_text: str) -> str:
    return re.sub(r"(?m)//[^\r\n]*$", "", contract_text)


def _task9_owner_assignments(
    document_text: str, schema_name: str
) -> tuple[tuple[str, str], ...] | None:
    body = _task9_owner_block(document_text, schema_name)
    if body is None:
        return None
    lines = _task9_strip_line_comments(body).splitlines()
    assignments: list[tuple[str, str]] = []
    current_name: str | None = None
    current_rhs: list[str] = []

    def finish_assignment() -> None:
        nonlocal current_name, current_rhs
        if current_name is not None:
            assignments.append(
                (current_name, _normalize_contract_text(current_rhs))
            )
        current_name = None
        current_rhs = []

    for line in lines:
        assignment = re.fullmatch(
            r"  ([a-z][a-z0-9_]*)[ \t]*:=[ \t]*(.*)", line
        )
        if assignment is not None:
            finish_assignment()
            current_name = assignment.group(1)
            current_rhs = [assignment.group(2)]
            continue
        if current_name is not None and line.startswith("    "):
            current_rhs.append(line.strip())
            continue
        finish_assignment()
    finish_assignment()
    return tuple(assignments)


def _task9_exact_owner_fields(
    document_text: str, schema_name: str
) -> dict[str, str] | None:
    body = _task9_owner_block(document_text, schema_name)
    if body is None:
        return None
    fields: dict[str, str] = {}
    for line in body.splitlines():
        if re.fullmatch(r"\s*//.*", line):
            continue
        if re.fullmatch(r"  [a-z][a-z0-9_]*\s*:=.*", line):
            continue
        if line.startswith("    "):
            continue
        field = re.fullmatch(
            r"  ([a-z][a-z0-9_]*):\s*(\S(?:.*\S)?)\s*", line
        )
        if field is None or field.group(1) in fields:
            return None
        fields[field.group(1)] = field.group(2)
    return fields


def _requested_backend_input_arms(
    html: str,
) -> dict[str, dict[str, str]] | None:
    definition = re.search(
        r"RequestedBackendInput\s*:=\s*(?P<body>.*?)"
        r"RequestedBackendInputMap\s*:=",
        html,
        re.DOTALL,
    )
    if definition is None:
        return None
    body = definition.group("body")
    arm_pattern = re.compile(
        r"(?P<tag>MemoryRequested|TimeRequested|NotRequested)"
        r"(?:\s*\{(?P<fields>[^{}]*)\})?"
    )
    matches = list(arm_pattern.finditer(body))
    if [match.group("tag") for match in matches] != [
        "MemoryRequested",
        "TimeRequested",
        "NotRequested",
    ]:
        return None
    arms: dict[str, dict[str, str]] = {}
    cursor = 0
    for match in matches:
        if re.sub(r"[|\s]", "", body[cursor : match.start()]):
            return None
        tag = match.group("tag")
        if tag in arms:
            return None
        fields_text = match.group("fields")
        if (tag == "NotRequested") != (fields_text is None):
            return None
        fields: dict[str, str] = {}
        field_cursor = 0
        if fields_text is not None:
            for field in re.finditer(
                r"([a-z][a-z0-9_]*):\s*([A-Z][A-Za-z0-9_]*)",
                fields_text,
            ):
                if fields_text[field_cursor : field.start()].strip():
                    return None
                name = field.group(1)
                if name in fields:
                    return None
                fields[name] = field.group(2)
                field_cursor = field.end()
            if fields_text[field_cursor:].strip():
                return None
        arms[tag] = fields
        cursor = match.end()
    if re.sub(r"[|\s]", "", body[cursor:]):
        return None
    return arms


def _production_cache_rows(html: str) -> dict[str, tuple[str, str, str]] | None:
    section = _section_between(html, "c13-3", "c13-4")
    if section is None:
        return None
    rows: dict[str, tuple[str, str, str]] = {}
    for match in re.finditer(
        r"<tr\b(?P<attrs>[^>]*)>(?P<body>.*?)</tr>", section, re.DOTALL
    ):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", match.group("body"), re.DOTALL)
        if not cells:
            continue
        cache_id_match = re.search(
            r'\bdata-cache-id="([a-z0-9-]+)"', match.group("attrs")
        )
        if cache_id_match is None or len(cells) != 3:
            return None
        cache_id = cache_id_match.group(1)
        if cache_id in rows:
            return None
        normalized_cells = tuple(
            re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", cell))).strip()
            for cell in cells
        )
        rows[cache_id] = (
            normalized_cells[0],
            normalized_cells[1],
            normalized_cells[2],
        )
    return rows


def _section_between(html: str, start_id: str, end_id: str) -> str | None:
    match = re.search(
        rf'<h3 id="{re.escape(start_id)}"[^>]*>.*?</h3>'
        rf'(?P<body>.*?)<h3 id="{re.escape(end_id)}"',
        html,
        re.DOTALL,
    )
    return match.group("body") if match is not None else None


def check_task7_modeling_contracts(html: str, errors: list[str]) -> None:
    common_inputs = _exact_flat_schema_fields(html, "CommonProductionInputs")
    if common_inputs != TASK7_COMMON_PRODUCTION_INPUT_FIELDS:
        errors.append("Task7 CommonProductionInputs exact schema 不闭合")

    canonical_evaluation_inputs = _exact_flat_schema_fields(
        html, "CanonicalProductionEvaluationInputs"
    )
    if (
        canonical_evaluation_inputs
        != TASK7_CANONICAL_PRODUCTION_EVALUATION_INPUT_FIELDS
    ):
        errors.append("Task7 CanonicalProductionEvaluationInputs exact schema 不闭合")

    request_snapshot = _exact_flat_schema_fields(html, "RequestSnapshot")
    if request_snapshot != TASK7_REQUEST_SNAPSHOT_FIELDS:
        errors.append("Task7 RequestSnapshot exact schema 不闭合")
    request_digest = re.search(
        r"(?m)^request_digest\s*:=\s*(?P<body>.*?)"
        r"^RequestSnapshot requested-backend equations:",
        html,
        re.DOTALL,
    )
    if request_digest is None or _normalize_contract_text(
        [request_digest.group("body")]
    ) != "hash(canonical_payload_without_derived_digests(RequestSnapshot))":
        errors.append("Task7 request_digest 必须覆盖 exact RequestSnapshot schema")

    requested_arms = _requested_backend_input_arms(html)
    if requested_arms != TASK7_REQUESTED_BACKEND_INPUT_ARMS:
        errors.append("Task7 RequestedBackendInput exact union 不闭合")

    comparison_basis = _exact_flat_schema_fields(html, "ComparisonBasis")
    if comparison_basis != TASK7_COMPARISON_BASIS_FIELDS:
        errors.append("Task7 ComparisonBasis exact schema 不闭合")

    time_input_domain = _exact_flat_schema_fields(html, "TimeSimulationInputDomain")
    if time_input_domain != TASK7_TIME_SIMULATION_INPUT_FIELDS:
        errors.append("Task7 TimeSimulationInputDomain exact schema 不闭合")
    time_digest = re.search(
        r"time_simulation_digest\s*:=\s*(?P<body>.*?)result_digest\s*:=",
        html,
        re.DOTALL,
    )
    if time_digest is None or _normalize_contract_text(
        [time_digest.group("body")]
    ) != "hash(canonical(TimeSimulationInputDomain))":
        errors.append("Task7 time_simulation_digest 必须只哈希 exact input domain")

    cache_rows = _production_cache_rows(html)
    if cache_rows != TASK7_PRODUCTION_CACHE_ROWS:
        errors.append("Task7 production cache exact schema 不闭合")

    coverage = _flat_schema_fields(html, "Coverage")
    if coverage is None:
        errors.append("Task7 Coverage schema 不可提取")
    else:
        for field_name, field_type in TASK7_COVERAGE_FIELDS.items():
            if coverage.get(field_name) != field_type:
                errors.append(
                    "Task7 Coverage field/type mismatch: "
                    f"{field_name}: {field_type}"
                )

    calibration = _flat_schema_fields(html, "CalibrationSet")
    expected_calibration = {
        "active_compute_records": "Map[MeasurementKey, ComputeMeasurementRecord]",
        "measurement_protocols": "Map[ProtocolDigest, MeasurementProtocol]",
        "calibration_train_manifest_digest": "Digest",
    }
    if calibration != expected_calibration:
        errors.append(
            "Task7 CalibrationSet 必须只持 active records/protocols 与 train manifest digest"
        )

    train_manifest = _flat_schema_fields(html, "CalibrationTrainManifest")
    expected_train_fields = {
        "split": "calibration_train",
        "active_measurement_keys": "OrderedSet[MeasurementKey]",
        "active_protocol_digests": "OrderedSet[ProtocolDigest]",
        "source_dataset_manifest_digests": "OrderedSet[Digest]",
        "calibration_train_manifest_digest": "Digest",
    }
    if train_manifest != expected_train_fields:
        errors.append("Task7 CalibrationTrainManifest payload/digest schema 不闭合")

    holdout = _flat_schema_fields(html, "HoldoutEvaluationManifest")
    if holdout is None or holdout.get("split") != "holdout":
        errors.append("Task7 HoldoutEvaluationManifest 必须是独立 offline split")

    hardware = _flat_schema_fields(html, "HardwareProfile")
    expected_hardware = {
        "devices": "DeviceCatalog",
        "topology": "TopologySnapshot",
        "bandwidth": "BandwidthSnapshot",
        "link_latency": "LinkLatencySnapshot",
        "allocator_alignment": "AllocatorAlignmentSnapshot",
        "physical_stream_catalog": "PhysicalStreamCatalog",
    }
    if hardware != expected_hardware:
        errors.append("Task7 HardwareProfile production schema 必须排除 capacity")

    production_inputs = re.search(
        r"CanonicalConfigEvaluationInput:(.*?)EvaluationInstanceIdentity:",
        html,
        re.DOTALL,
    )
    if production_inputs is None or "PlanningMetadata" in production_inputs.group(1):
        errors.append("Task7 PlanningMetadata 只能属于上层，不能进入生产输入")

    comparison = re.search(
        r"compare_metric_outcome\(left_status, right_status, basis_pair\):"
        r"(?P<body>.*?)end compare_metric_outcome",
        html,
        re.DOTALL,
    )
    expected_branches = (
        "if left_status == NotRequested and right_status == NotRequested:",
        "return NotRequested",
        "if left_status != Ok or right_status != Ok:",
        "return Unavailable",
        "if basis_pair.left != basis_pair.right:",
        "return Incomparable",
        "return ComparableDelta",
    )
    if comparison is None:
        errors.append("Task7 comparison outcome pseudocode 不可提取")
    else:
        body = _normalize_contract_text([comparison.group("body")])
        positions = [body.find(branch) for branch in expected_branches]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            errors.append("Task7 comparison outcome branch priority/order 不闭合")

    non_goals = _section_between(html, "c0-2", "c0-3")
    decisions = _section_between(html, "c14-1", "c14-2")
    non_goal_ids = (
        re.findall(r'<tr\b[^>]*data-non-goal-id="([A-Z0-9-]+)"', non_goals)
        if non_goals is not None
        else []
    )
    decision_refs = (
        re.findall(r'<tr\b[^>]*data-non-goal-ref="([A-Z0-9-]+)"', decisions)
        if decisions is not None
        else []
    )
    decision_ids = (
        re.findall(r'<tr\b[^>]*data-decision-id="([A-Z0-9-]+)"', decisions)
        if decisions is not None
        else []
    )
    if len(non_goal_ids) != 10 or set(non_goal_ids) != TASK7_NON_GOAL_IDS:
        errors.append("Task7 Chapter 0 必须有十个唯一稳定 non-goal ID")
    if len(decision_refs) != 10 or set(decision_refs) != TASK7_NON_GOAL_IDS:
        errors.append("Task7 Chapter 14 决策 ref 必须与十个 non-goal ID 机械等价")
    if len(decision_ids) != 19 or set(decision_ids) != TASK7_DECISION_IDS:
        errors.append("Task7 Chapter 14 必须保留唯一稳定 decision ID")

    capability_section = re.search(
        r'<h3 id="c14-5"[^>]*>.*?</h3>(.*?)</main>', html, re.DOTALL
    )
    capability_scopes = re.findall(
        r'data-capability-scope="([a-z-]+)"',
        capability_section.group(1) if capability_section is not None else "",
    )
    if capability_scopes != ["production", "offline-validation"]:
        errors.append("Task7 14.5 必须分离 production 与 offline-validation 能力")


def check_task9_production_identity_isolation(
    html: str, errors: list[str]
) -> None:
    document_text = unescape(html)
    normalized_document = _normalize_contract_text([document_text])
    release_result_union = "|".join(
        re.escape(result_type) for result_type in CONFORMANCE_RELEASE_RESULT_TYPES
    )
    basic_to_release_flow = re.compile(
        r"(?ms)(?:^|>)[A-Za-z][A-Za-z0-9_]*[ \t]*(?::=[ \t]*)?\("
        r"[^)]*\bBasicOfflineReportArtifact\b[^)]*\)[ \t]*->[ \t]*(?:"
        + release_result_union
        + r")\b"
    )
    if basic_to_release_flow.search(document_text):
        errors.append("Basic artifact to release type-flow is forbidden")

    production_subject_block = _task9_owner_block(document_text, "ProductionSubject")
    if production_subject_block is None or re.search(
        r"(?m)^  subject_digest := hash\(all four constituent fields in schema order\)"
        r"[ \t]*(?:\r?\n)?\Z",
        production_subject_block,
    ) is None:
        errors.append("ProductionSubject subject_digest exact equation mismatch")

    conformance_digest_owners = {
        owner
        for owners in CONFORMANCE_OWNER_INVENTORY.values()
        for owner in owners
    }
    shared_type_union = "|".join(
        re.escape(type_name) for type_name in CONFORMANCE_SHARED_TYPES
    )
    shared_alias_union = "|".join(
        re.escape(
            re.sub(r"(?<!^)(?=[A-Z])", "_", type_name).lower()
        )
        for type_name in CONFORMANCE_SHARED_TYPES
    )
    production_conformance_field = re.compile(
        r"(?im)^[ \t]+(?:conformance|attested|basic_offline|release_approval|"
        r"release_policy)[a-z0-9_]*[ \t]*:(?!=)|^[ \t]+(?:"
        + shared_alias_union
        + r")(?:_digest|_ref)?[ \t]*:(?!=)|\b(?:"
        r"ConformanceDeploymentProfile|BasicOfflineReport|"
        r"BasicOfflineReportArtifact|AttestedConformanceReport|"
        r"AttestedConformanceSealArtifact|AttestedConformanceRunResult|"
        r"ReleasePolicyApplicationResult|ConformanceTrustRootCapability|"
        r"ReleaseApprovalTrustRootCapability|"
        r"ValidatedMeasurementSessionCapability|"
        + shared_type_union
        + r")\b"
    )
    for owner, _ in DERIVED_DIGEST_EXCLUSIONS:
        if owner in conformance_digest_owners:
            continue
        owner_block = _task9_owner_block(document_text, owner)
        if owner_block is not None and production_conformance_field.search(owner_block):
            errors.append(
                f"production owner conformance isolation mismatch: {owner}"
            )
    for owner, expected_fields in TASK9_PRODUCTION_OWNER_FIELDS.items():
        definitions = re.findall(
            rf"(?m)(?:^|>){re.escape(owner)}:[ \t]*$", document_text
        )
        actual_fields = _task9_exact_owner_fields(document_text, owner)
        if len(definitions) != 1 or actual_fields != expected_fields:
            errors.append(f"Task9 {owner} exact production schema mismatch")

    digest_section = re.search(
        r'<h3 id="c2-4"[^>]*>.*?</h3>.*?'
        r'<pre><code>(?P<body>.*?)</code></pre>',
        html,
        re.DOTALL,
    )
    digest_text = unescape(digest_section.group("body") if digest_section else "")
    formulas = (
        (
            "model_input_digest",
            "model_digest",
            "hash(SourceSnapshot, ModelSpec, normalized config P, LogicalRankContextSet, code-relevant ExecutionScenario bindings, CompileEnvFacts, PySub/evaluator/descriptor/dispatch semantics, source-obligation and canonical-ID/order rules, structure semantic registry)",
        ),
        (
            "model_digest",
            "runtime_input_digest",
            "hash(model_input_digest, canonical_payload_without_derived_digests(CodeIR))",
        ),
        (
            "runtime_input_digest",
            "runtime_plan_digest",
            "hash(model_digest, ExecutionScenario, runtime semantic registry, runtime expander/EventId/order semantic version)",
        ),
        (
            "runtime_plan_digest",
            "simulation_core_digest",
            "hash(runtime_input_digest, canonical_payload_without_derived_digests(RuntimeEventPlan))",
        ),
        (
            "simulation_core_digest",
            "memory_simulation_digest",
            "hash(runtime_plan_digest, HardwareProfile, ExecutionDeployment, HardwareBindingPolicySnapshot, binding semantics, canonical_payload_without_derived_digests(SimulationPlanCore))",
        ),
        (
            "memory_simulation_digest",
            "TimeSimulationInputDomain:",
            "hash(simulation_core_digest, memory registry, StorageBindings, WorkspaceBindings, MemoryProjectionSemanticsSnapshot, resolved memory fallbacks/assumptions, logical-order semantics, memory projection/identity semantics, memory backend and numeric semantic versions)",
        ),
        (
            "time_simulation_digest",
            "result_digest",
            "hash(canonical(TimeSimulationInputDomain))",
        ),
    )
    for owner, next_owner, expected in formulas:
        match = re.search(
            rf"(?ms)^{re.escape(owner)}\s*:=\s*(?P<body>.*?)"
            rf"(?=^{re.escape(next_owner)}\s*(?::=|$))",
            digest_text,
        )
        actual = _normalize_contract_text([match.group("body")]) if match else ""
        if actual != expected:
            errors.append(f"Task9 production digest exact input closure mismatch: {owner}")
    result_match = re.search(
        r"(?ms)^result_digest\s*:=\s*(?P<body>.*)$", digest_text
    )
    if (
        result_match is None
        or _normalize_contract_text([result_match.group("body")])
        != "hash(canonical_payload_without_derived_digests(Estimate))"
    ):
        errors.append("Task9 production digest exact input closure mismatch: result_digest")

    backend_result = re.search(
        r"(?ms)^BackendResult<T>\s*:=\s*(?P<body>.*?)^SemanticValue<T>:",
        document_text,
    )
    expected_backend_result = (
        "Ok { estimate: Estimate<T> } | Blocked { diagnostics: "
        "NonEmpty[Diagnostic] } | NotRequested"
    )
    if (
        backend_result is None
        or _normalize_contract_text([backend_result.group("body")])
        != expected_backend_result
    ):
        errors.append("Task9 BackendResult exact three-arm production union mismatch")


def check_task7_behavioral_contracts(html: str, errors: list[str]) -> None:
    request_input = re.search(
        r"RequestedBackendInput\s*:=\s*(.*?)RequestedBackendInputMap\s*:=",
        html,
        re.DOTALL,
    )
    request_text = (
        _normalize_contract_text([request_input.group(1)])
        if request_input is not None
        else ""
    )
    request_tokens = (
        "calibration_set: CalibrationSet",
        "calibration_train_manifest_snapshot: CalibrationTrainManifest",
        "communication_model_snapshot: CommunicationModelSnapshot",
        "time_cost_policy: TimeCostPolicy",
    )
    if request_input is None or any(token not in request_text for token in request_tokens):
        errors.append("Task7 TimeRequested 未携带独立 train manifest authority input")

    request_builder = re.search(
        r"build_request_snapshot gate specification equations:"
        r"(?P<body>.*?)</code></pre>",
        html,
        re.DOTALL,
    )
    request_builder_text = (
        _normalize_contract_text([request_builder.group("body")])
        if request_builder is not None
        else ""
    )
    request_closure_tokens = (
        "calibration := requested_backend_inputs[time].calibration_set",
        "train_manifest := requested_backend_inputs[time].calibration_train_manifest_snapshot",
        "calibration.calibration_train_manifest_digest == train_manifest.calibration_train_manifest_digest",
        "keys(calibration.active_compute_records) == train_manifest.active_measurement_keys",
        "keys(calibration.measurement_protocols) == train_manifest.active_protocol_digests",
    )
    if request_builder is None or any(
        token not in request_builder_text for token in request_closure_tokens
    ):
        errors.append("Task7 RequestSnapshot train manifest authority closure 不闭合")

    memory_constructor = re.search(
        r"build_memory_projection_candidate\(request, evaluation_identity, core\):"
        r"(?P<body>.*?)build_time_projection_candidate\(",
        html,
        re.DOTALL,
    )
    memory_text = (
        _normalize_contract_text([memory_constructor.group("body")])
        if memory_constructor is not None
        else ""
    )
    if (
        memory_constructor is None
        or "construct only from memory_inputs.memory_registry_snapshot" not in memory_text
        or any(
            forbidden in memory_text
            for forbidden in (
                "calibration_set",
                "calibration_train_manifest_snapshot",
                "communication_model_snapshot",
                "time_cost_policy",
                "requested_backend_inputs[time]",
            )
        )
    ):
        errors.append("Task7 memory face authority 必须只读取 memory input arm")

    time_constructor = re.search(
        r"build_time_projection_candidate\(request, evaluation_identity, core\):"
        r"(?P<body>.*?)evaluate_and_finalize_projection_bundle\(authority\):",
        html,
        re.DOTALL,
    )
    time_text = (
        _normalize_contract_text([time_constructor.group("body")])
        if time_constructor is not None
        else ""
    )
    time_tokens = (
        "calibration := time_inputs.calibration_set",
        "train_manifest := time_inputs.calibration_train_manifest_snapshot",
        "communication_model := time_inputs.communication_model_snapshot",
        "time_policy := time_inputs.time_cost_policy",
        "calibration.calibration_train_manifest_digest == train_manifest.calibration_train_manifest_digest",
        "keys(calibration.active_compute_records) == train_manifest.active_measurement_keys",
        "keys(calibration.measurement_protocols) == train_manifest.active_protocol_digests",
    )
    if time_constructor is None or any(token not in time_text for token in time_tokens):
        errors.append("Task7 time authority 未闭合 calibration records/protocols 与 train manifest")
    if "memory_registry_snapshot" in time_text or "HoldoutEvaluationManifest" in time_text:
        errors.append("Task7 time face authority 混入 memory/holdout input")

    bundle = re.search(
        r"evaluate_and_finalize_projection_bundle\(authority\):"
        r"(?P<body>.*?)</code></pre>",
        html,
        re.DOTALL,
    )
    bundle_text = (
        _normalize_contract_text([bundle.group("body")]) if bundle is not None else ""
    )
    shared_scope_tokens = (
        "each construction blocker contains candidate.backend in affected_backends",
        "every BlockerRecord that prevents RuntimeBuildResult or CoreBuildResult from being Ready has affected_backends=={memory,time}",
        "every InputBlocker occurrence from a shared structure/G-IR invocation has affected_backends=={memory,time}",
    )
    if bundle is None or any(token not in bundle_text for token in shared_scope_tokens):
        errors.append("Task7 backend-local 与 shared failure scope 不闭合")

    bundle_source = unescape(bundle.group("body")) if bundle is not None else ""
    runtime_branch = re.search(
        r"match authority\.runtime_result:\s*"
        r"Blocked \{ blockers=bs \}:\s*(?P<blocked>.*?)\s*"
        r"Ready \{ runtime_plan=plan \}:\s*(?P<ready>.*?)\s*"
        r"(?=match authority\.core_result)",
        bundle_source,
        re.DOTALL,
    )
    runtime_blocked_text = re.sub(
        r"\s+", " ", runtime_branch.group("blocked") if runtime_branch else ""
    ).strip()
    if runtime_blocked_text != (
        "require authority.core_result == Blocked { blockers=bs }"
    ):
        errors.append("Task8 runtime-blocked core propagation 不闭合")

    core_branches = re.search(
        r"match authority\.core_result \(exhaustive, mutually exclusive\):\s*"
        r"Ready \{ core=core \}:\s*(?P<ready>.*?)\s*"
        r"Blocked \{ blockers=blockers \}:\s*(?P<blocked>.*?)\s*"
        r"require exactly one candidate-construction branch executed",
        bundle_source,
        re.DOTALL,
    )
    ready_core_text = core_branches.group("ready") if core_branches else ""
    blocked_core_text = core_branches.group("blocked") if core_branches else ""
    ready_bindings = sorted(
        re.findall(
            r"require\s+authority\.(memory|time)_candidate\s+is byte-equal to\s+"
            r"build_(memory|time)_projection_candidate\(\s*"
            r"authority\.request_snapshot,\s*authority\.evaluation_identity,\s*"
            r"core\s*\)",
            ready_core_text,
        )
    )
    blocked_bindings = sorted(
        re.findall(
            r"require\s+authority\.(memory|time)_candidate\s+is byte-equal to\s+"
            r"expected_projection_candidate\(\s*authority\.request_snapshot,\s*"
            r"authority\.evaluation_identity,\s*(memory|time),\s*"
            r"None if (memory|time) not in\s*"
            r"authority\.request_snapshot\.requested_backends else\s*"
            r"CandidateBlocked \{\s*construction_blockers=([A-Za-z_]+)\s*\}\s*\)",
            blocked_core_text,
        )
    )
    exact_backends = [("memory", "memory"), ("time", "time")]
    exact_blocked_backends = [
        ("memory", "memory", "memory", "blockers"),
        ("time", "time", "time", "blockers"),
    ]
    if (
        core_branches is None
        or bundle_source.count("match authority.core_result") != 1
        or bundle_source.count(
            "require exactly one candidate-construction branch executed"
        )
        != 1
        or ready_bindings != exact_backends
        or "expected_projection_candidate(" in ready_core_text
        or blocked_bindings != exact_blocked_backends
        or "build_memory_projection_candidate(" in blocked_core_text
        or "build_time_projection_candidate(" in blocked_core_text
        or blocked_core_text.count("do not call either backend candidate builder")
        != 1
        or "both candidates are byte-equal to the exact outputs of the declared"
        in bundle_source
    ):
        errors.append("Task8 exhaustive core-result candidate branches 不闭合")

    comparison_basis = re.search(
        r"ComparisonBasis:\s*(?P<body>.*?)comparison_basis_digest\s*:=",
        html,
        re.DOTALL,
    )
    if comparison_basis is None or re.search(
        r"(?m)^\s*logical_rank_id_set:\s*OrderedSet\[LogicalRank\]\s*$",
        comparison_basis.group("body"),
    ) is None:
        errors.append("Task7 ComparisonBasis 缺少 logical_rank_id_set world boundary")

    coverage_section = re.search(
        r"Coverage:\s*(?P<body>.*?)<h3 id=\"c2-4\"",
        html,
        re.DOTALL,
    )
    coverage_text = coverage_section.group("body") if coverage_section is not None else ""
    coverage_equations = (
        "source_obligations_total = source_obligations_planned + source_obligations_residual + source_obligations_proven_not_executed",
        "event_obligations_total = event_obligations_planned + event_obligations_blocked + event_obligations_not_applicable",
    )
    if coverage_section is None or any(
        equation not in coverage_text for equation in coverage_equations
    ):
        errors.append("Task7 Coverage obligation conservation equations 不闭合")

def check_derived_digest_exclusions(html: str, errors: list[str]) -> None:
    table_match = re.search(
        r"derived-digest exclusion table \(exact\):(.*?)"
        r"all nested input digests not named above remain in the payload",
        html,
        re.DOTALL,
    )
    table = table_match.group(1) if table_match is not None else ""
    for wildcard in ("*Binding", "*Snapshot / *Policy"):
        if wildcard in table:
            errors.append(
                f"derived-digest exclusion table forbids wildcard owner: {wildcard}"
            )
    expected = dict(DERIVED_DIGEST_EXCLUSIONS)
    parsed_rows = re.findall(
        r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)\s*-&gt;\s*"
        r"\{\s*([a-z][a-z0-9_]*)\s*\}\s*$",
        table,
    )
    parsed = dict(parsed_rows)
    if len(parsed_rows) != len(expected) or len(parsed) != len(parsed_rows):
        errors.append(
            "derived-digest exclusion table must contain each concrete owner exactly once"
        )
    for owner_type in sorted(set(parsed) - set(expected)):
        errors.append(
            f"derived-digest exclusion table has undeclared concrete owner: {owner_type}"
        )
    for owner_type, own_field in DERIVED_DIGEST_EXCLUSIONS:
        owner_row = re.compile(
            rf"(?m)^\s*{re.escape(owner_type)}\s*-&gt;\s*\{{[^\r\n]*\}}\s*$"
        )
        exact_row = re.compile(
            rf"(?m)^\s*{re.escape(owner_type)}\s*-&gt;\s*"
            rf"\{{\s*{re.escape(own_field)}\s*\}}\s*$"
        )
        owner_rows = owner_row.findall(table)
        exact_rows = exact_row.findall(table)
        if len(owner_rows) != 1 or len(exact_rows) != 1:
            errors.append(
                "derived-digest exclusion table 缺少唯一精确 own-field："
                f"{owner_type} -> {{ {own_field} }}"
            )

    canonical_owner_calls = {
        owner
        for owner in re.findall(
            r"canonical_payload_without_derived_digests\(\s*"
            r"([A-Z][A-Za-z0-9_]*)\s*\)",
            html,
        )
        if owner != "X"
    }
    for owner_type in sorted(canonical_owner_calls - set(expected)):
        errors.append(
            "derived-digest exclusion table missing canonical payload owner: "
            f"{owner_type}"
        )


def check_product_gate_staging(html: str, errors: list[str]) -> None:
    for required_text in PRODUCT_GATE_STAGING_REQUIRED_TEXT:
        if required_text not in html:
            errors.append(
                f"product gate staging missing typed staged call: {required_text}"
            )


def _conformance_local_texts(
    html: str, errors: list[str]
) -> dict[tuple[str, str], str]:
    parser = _ConformanceLocalBlockParser()
    parser.feed(html)
    parser.close()
    if parser.active is not None or parser.conformance_depth is not None:
        parser.malformed = True
    grouped: dict[tuple[str, str], list[str]] = {}
    for block in parser.blocks:
        key = (str(block["kind"]), str(block["value"]))
        parts = block["text_parts"]
        assert isinstance(parts, list)
        grouped.setdefault(key, []).append(_normalize_contract_text(parts))
    actual = set(grouped)
    if parser.malformed or actual != CONFORMANCE_LOCAL_BLOCKS:
        errors.append(
            "conformance local blocks must be exact shared/deployment/basic-offline/attested-release"
        )
    return {key: " ".join(parts) for key, parts in grouped.items()}


def _conformance_local_raw_texts(html: str) -> dict[tuple[str, str], str]:
    parser = _ConformanceLocalBlockParser()
    parser.feed(html)
    parser.close()
    grouped: dict[tuple[str, str], list[str]] = {}
    for block in parser.blocks:
        key = (str(block["kind"]), str(block["value"]))
        parts = block["text_parts"]
        assert isinstance(parts, list)
        grouped.setdefault(key, []).append("".join(parts))
    return {key: "\n".join(parts) for key, parts in grouped.items()}


def _conformance_outside_local_text(html: str) -> str:
    parser = _ConformanceLocalBlockParser()
    parser.feed(html)
    parser.close()
    return _normalize_contract_text(parser.outside_parts)


def check_conformance_deployment_profiles(html: str, errors: list[str]) -> None:
    local = _conformance_local_texts(html, errors)
    raw_local = _conformance_local_raw_texts(html)
    outside_local = _conformance_outside_local_text(html)
    parsed_inventory: dict[tuple[str, str], tuple[str, ...]] = {}
    for key, raw_text in raw_local.items():
        parsed_inventory[key] = tuple(
            re.findall(
                r"(?m)^([A-Z][A-Za-z0-9_]*)\s*(?::=|:)", raw_text
            )
        )
    if parsed_inventory != CONFORMANCE_OWNER_INVENTORY:
        errors.append("conformance owner inventory must exactly match every local block")

    parsed_callables: dict[tuple[str, str], tuple[str, ...]] = {}

    def parse_callable_names(raw_text: str) -> tuple[str, ...]:
        names: list[str] = []
        lines = raw_text.splitlines()
        for index, line in enumerate(lines):
            start = re.match(r"^([a-z][a-z0-9_]*)\s*\(", line)
            if start is None:
                continue
            name = start.group(1)
            same_line_tail = line[start.end() :]
            if re.search(r"\)\s*->", same_line_tail):
                names.append(name)
                continue
            if re.search(r"\)\s*:=", same_line_tail):
                continue
            for continuation in lines[index + 1 :]:
                close = re.match(r"^\)\s*(->|:=)", continuation)
                if close is None:
                    continue
                if close.group(1) == "->":
                    names.append(name)
                break
        return tuple(names)

    for key, raw_text in raw_local.items():
        parsed_callables[key] = parse_callable_names(raw_text)
    if parsed_callables != CONFORMANCE_CALLABLE_INVENTORY:
        errors.append("whole conformance callable owner inventory mismatch")
    export_lines = sorted(
        line.strip()
        for raw_text in raw_local.values()
        for line in raw_text.splitlines()
        if re.search(r"(?i)\b(?:export(?:ed)?|alias)\b", line)
    )
    expected_export_lines = sorted(
        [
            "public exported ports (exact):",
            "public exported ports (exact):",
            "internal non-exported ports:",
        ]
    )
    if export_lines != expected_export_lines or re.search(
        r"(?i)\b(?:export(?:ed)?|alias)\b", outside_local
    ):
        errors.append("whole conformance exported port inventory mismatch")

    profile_specific_owners = tuple(
        owner
        for key, owners in CONFORMANCE_OWNER_INVENTORY.items()
        if key != ("scope", "shared")
        for owner in owners
    )
    outside_identifiers = profile_specific_owners + (
        "run_basic_offline_conformance",
        "run_attested_release_conformance",
        "apply_release_policy",
        "validate_and_seal_attested_conformance",
        "validate_and_consume_measurement_envelope",
        "derive_attested_conformance_report",
        "derive_approved_digest_set",
    )
    if any(
        re.search(rf"\b{re.escape(identifier)}\b", outside_local)
        for identifier in outside_identifiers
    ) or any(rule in outside_local for rule in CONFORMANCE_ATTESTED_ONLY_RULES):
        errors.append("conformance profile-specific declaration outside local block")

    document_text = unescape(html)
    normalized_document = _normalize_contract_text([document_text])
    conformance_section = re.search(
        r'<section\b[^>]*data-module="conformance"[^>]*>.*?</section>',
        document_text,
        re.DOTALL,
    )
    if conformance_section is None:
        errors.append("conformance module is not globally extractable")
        outside_document = document_text
    else:
        outside_document = (
            document_text[: conformance_section.start()]
            + document_text[conformance_section.end() :]
        )
    outside_callable_names = re.findall(
        r"(?m)(?:^|>)([a-z][a-z0-9_]*)\s*\([^\r\n]*?\)\s*->",
        outside_document,
    )
    if any(
        re.search(r"(?i)basic|attested|conformance|release", name)
        for name in outside_callable_names
    ):
        errors.append("conformance declaration outside module")
    for owners in CONFORMANCE_OWNER_INVENTORY.values():
        for owner in owners:
            definitions = re.findall(
                rf"(?m)(?:^|>){re.escape(owner)}\s*(?::=|:)", document_text
            )
            if len(definitions) != 1:
                errors.append(
                    f"global conformance owner uniqueness mismatch: {owner}"
                )
    unknown_profile_declarations = re.findall(
        r"(?m)(?:^|>)([A-Z][A-Za-z0-9_]*Conformance)\s*(?::=|:)",
        document_text,
    )
    if unknown_profile_declarations:
        errors.append("closed conformance profile declaration set mismatch")
    profile_union_definitions = re.findall(
        r"(?m)(?:^|>)([A-Z][A-Za-z0-9_]*Profile)\s*:=", document_text
    )
    conformance_profile_aliases = re.findall(
        r"(?m)(?:^|>)([A-Z][A-Za-z0-9_]*)[ \t]*:=[ \t]*[^\r\n]*"
        r"(?:ConformanceDeploymentProfile|BasicOfflineConformance|"
        r"AttestedReleaseConformance)",
        document_text,
    )
    if (
        profile_union_definitions != ["ConformanceDeploymentProfile"]
        or conformance_profile_aliases
        or document_text.count("ConformanceDeploymentProfile :=") != 1
        or document_text.count(
            "default_conformance_deployment_profile := BasicOfflineConformance"
        )
        != 1
    ):
        errors.append("closed conformance profile definition inventory mismatch")

    deployment = local.get(("scope", "deployment"), "")
    union_match = re.search(
        r"ConformanceDeploymentProfile\s*:=\s*(.*?)\s*"
        r"default_conformance_deployment_profile\s*:=",
        deployment,
    )
    exact_union_body = re.compile(
        r"BasicOfflineConformance\s*\|\s*AttestedReleaseConformance\s*\{\s*"
        r"conformance_trust_root:\s*ConformanceTrustRootCapability\s*"
        r"release_trust_root:\s*ReleaseApprovalTrustRootCapability\s*\}"
    )
    if union_match is None or exact_union_body.fullmatch(union_match.group(1)) is None:
        errors.append("conformance deployment profile union must have exact two arms")
    for token in (
        "default_conformance_deployment_profile := BasicOfflineConformance",
        "non-serializable",
        "owns no digest",
        "never a RequestSnapshot field",
        "without both protected capabilities -> InternalContractViolation",
        "never silently downgrade",
    ):
        if token not in deployment:
            errors.append(f"conformance deployment profile missing contract: {token}")
    if re.search(r"\bDisabled\b|Optional\s*<|None\s*\|", deployment):
        errors.append("conformance deployment profile must not add disabled/nullable arms")

    shared = local.get(("scope", "shared"), "")
    whole_local = " ".join(local.values())
    for type_name in CONFORMANCE_SHARED_TYPES:
        definition = f"{type_name}:"
        if shared.count(definition) != 1 or whole_local.count(definition) != 1:
            errors.append(f"conformance shared type must have one shared owner: {type_name}")
    heavy_owners = CONFORMANCE_OWNER_INVENTORY[("profile", "attested-release")]
    if re.search(r"\b(?:BasicOffline|AttestedConformance)\w*", shared) or any(
        re.search(rf"\b{re.escape(owner)}\b", shared) for owner in heavy_owners
    ):
        errors.append("conformance shared block references a profile-specific type")
    for token in (
        "derive_conformance_findings_and_verdict(",
        "deterministic finding/verdict rules:",
        "observed_clause_outputs: OrderedMap<FixtureRef, ObservedClauseOutput>",
    ):
        if shared.count(token) != 1 or whole_local.count(token) < 1:
            errors.append(f"conformance shared block missing single authority: {token}")

    basic = local.get(("profile", "basic-offline"), "")
    raw_basic = raw_local.get(("profile", "basic-offline"), "")
    for type_name in CONFORMANCE_BASIC_TYPES:
        if f"{type_name}:" not in basic and f"{type_name} :=" not in basic:
            errors.append(f"basic-offline block missing concrete type: {type_name}")
    for token in (
        'profile_schema_id := "basic-offline/v1"',
        'hash("basic-offline-authority/v1",',
        'hash("basic-offline-record/v1",',
        'hash("basic-offline-ledger/v1",',
        'hash("basic-offline-report/v1",',
        'hash("basic-offline-report-artifact/v1",',
        "run_basic_offline_conformance( authority: BasicOfflineConformanceAuthority ) -> BasicOfflineRunResult",
        "offline_verdict: pass | fail | insufficient",
        "identity/integrity only, not runner authenticity",
    ):
        if token not in basic:
            errors.append(f"basic-offline block missing contract: {token}")
    if basic.count('profile_schema_id := "basic-offline/v1"') != 5:
        errors.append("basic-offline every digest-owning concrete schema must bind profile_schema_id")
    for forbidden_pattern in CONFORMANCE_BASIC_FORBIDDEN:
        if re.search(forbidden_pattern, basic, re.IGNORECASE):
            errors.append(
                "basic-offline block contains attested/release token: "
                f"{forbidden_pattern}"
            )
    if any(re.search(rf"\b{re.escape(owner)}\b", basic) for owner in heavy_owners):
        errors.append("basic-offline block references an Attested owner")
    basic_result = re.search(
        r"BasicOfflineRunResult\s*:=\s*(.*?)\s*public exported ports \(exact\):",
        basic,
    )
    if (
        basic_result is None
        or basic_result.group(1)
        != "Completed { artifact: BasicOfflineReportArtifact } | InternalViolation { violation: InternalContractViolation }"
    ):
        errors.append("BasicOfflineRunResult exact two-arm union mismatch")
    basic_content_exclusions = {
        owner
        for owner, _ in DERIVED_DIGEST_EXCLUSIONS
        if owner.startswith("BasicOffline")
    }
    basic_content_boundary = (
        "Basic 五种 content-addressed payload 的 own digest 只使用 "
        "basic-offline/v1 schema 与各自 basic-offline-*/v1 hash domain；"
        "BasicOfflineRunResult 是无 own digest、无 profile_schema_id field 的 "
        "transport union"
    )
    if not (
        basic.count(basic_content_boundary) == 1
        and basic_result is not None
        and basic_result.group(1)
        == "Completed { artifact: BasicOfflineReportArtifact } | InternalViolation { violation: InternalContractViolation }"
        and basic.count('profile_schema_id := "basic-offline/v1"') == 5
        and basic_content_exclusions == set(CONFORMANCE_BASIC_CONTENT_TYPES)
        and "BasicOfflineRunResult" not in basic_content_exclusions
    ):
        errors.append(
            "Basic content-addressed payload and transport result boundary mismatch"
        )
    basic_algorithm_match = re.search(
        r"(?ms)^run_basic_offline_conformance equations:[ \t]*$\n"
        r"(?P<body>.*?"
        r"^  return Completed iff every Basic closure equation holds; otherwise InternalViolation[ \t]*$)",
        raw_basic,
    )
    raw_basic_algorithm = (
        basic_algorithm_match.group("body") if basic_algorithm_match else ""
    )
    basic_algorithm = _normalize_contract_text([raw_basic_algorithm])
    basic_algorithm_lines = tuple(
        line.rstrip()
        for line in raw_basic_algorithm.splitlines()
        if line.strip()
    )
    if basic_algorithm_lines != CONFORMANCE_BASIC_ALGORITHM_LINES:
        errors.append("conformance basic closed control grammar mismatch")
    for token in CONFORMANCE_BASIC_CLOSURE_RULES:
        statement_head = token
        if "hash(" in token and ", " in token:
            statement_head = token.split(", ", 1)[0] + ","
        canonical_statement_count = len(
            re.findall(
                rf"(?m)^[ \t]*{re.escape(statement_head)}", raw_basic_algorithm
            )
        )
        if token not in basic_algorithm or canonical_statement_count != 1:
            errors.append(
                f"module-contract conformance basic closure missing: {token}"
            )

    attested = local.get(("profile", "attested-release"), "")
    raw_attested = raw_local.get(("profile", "attested-release"), "")
    for token in CONFORMANCE_ATTESTED_REQUIRED:
        if token not in attested:
            errors.append(f"attested-release block missing contract: {token}")
    for owner, next_owner in (
        ("AttestedConformanceInvocationAuthority", "RunnerAttestation"),
        ("AttestedConformanceReport", "ReleasePolicy"),
        ("AttestedConformanceSealArtifact", "AttestedConformanceRunResult"),
    ):
        owner_schema = re.search(
            rf"{owner}:\s*(.*?)\s*{next_owner}(?::|\s*:=)",
            attested,
        )
        if (
            owner_schema is None
            or owner_schema.group(1).count(
                'profile_schema_id := "attested-release/v1"'
            )
            != 1
        ):
            errors.append(
                f"attested {owner} schema must bind profile_schema_id exactly once"
            )
    if (
        _task9_exact_owner_fields(
            raw_attested, "AttestedConformanceInvocationAuthority"
        )
        != CONFORMANCE_ATTESTED_AUTHORITY_FIELDS
    ):
        errors.append("AttestedConformanceInvocationAuthority exact schema mismatch")
    if (
        _task9_exact_owner_fields(raw_attested, "ReleaseApprovalAuthority")
        != CONFORMANCE_RELEASE_APPROVAL_AUTHORITY_FIELDS
    ):
        errors.append("ReleaseApprovalAuthority exact schema mismatch")
    release_approval_authority_schema = _normalize_contract_text(
        [_task9_owner_block(raw_attested, "ReleaseApprovalAuthority") or ""]
    )
    expected_release_schema_signed_message = (
        "signed_message := canonical_tuple( "
        "approved.release_policy_digest, "
        "store_snapshot.release_approval_store_snapshot_digest, "
        "trusted_approver_key_id, approval_signature_scheme, canonical(approved))"
    )
    if (
        release_approval_authority_schema.count(
            expected_release_schema_signed_message
        )
        != 1
    ):
        errors.append("release approval signed-message approved binding mismatch")
    if (
        _task9_exact_owner_fields(raw_attested, "AttestedConformanceSealArtifact")
        != CONFORMANCE_ATTESTED_SEAL_ARTIFACT_FIELDS
    ):
        errors.append("AttestedConformanceSealArtifact exact schema mismatch")
    if re.search(
        r"(?:ConformanceArtifact|BasicOfflineReportArtifact)\s*[,)]\s*"
        r"approval:",
        attested,
    ):
        errors.append("apply_release_policy accepts a Basic/common artifact")
    approved_match = re.search(
        r"ApprovedDigestSet:\s*(.*?)\s*ReleaseApprovalStoreSnapshot:",
        attested,
    )
    expected_approved_fields = [
        "subject_digest",
        "fixture_set_digest",
        "validation_policy_digest",
        "gate_manifest_digest",
        "verifier_runner_digest",
        "runner_attestation_digest",
        "conformance_execution_ledger_digest",
        "conformance_report_digest",
        "conformance_seal_artifact_digest",
        "release_policy_digest",
    ]
    approved_raw = re.search(
        r"(?ms)^ApprovedDigestSet:\s*$\n(?P<body>.*?)"
        r"^ReleaseApprovalStoreSnapshot:\s*$",
        raw_local.get(("profile", "attested-release"), ""),
    )
    approved_fields = [
        line.strip()
        for line in (approved_raw.group("body") if approved_raw else "").splitlines()
        if line.strip()
    ]
    if approved_fields != expected_approved_fields:
        errors.append("Attested ApprovedDigestSet exact ten-field schema mismatch")

    authority_assignments = _task9_owner_assignments(
        raw_attested, "AttestedConformanceInvocationAuthority"
    )
    authority_validator = re.search(
        r"validate_and_seal_attested_conformance first operation:\s*"
        r"(?P<body>.*?)\s*validate_and_seal_attested_conformance equations after first operation:",
        attested,
    )
    expected_authority_assignments = (
        ("profile_schema_id", '"attested-release/v1"'),
        (
            "conformance_invocation_digest",
            'hash("attested-conformance-authority/v1", '
            "canonical_payload_without_derived_digests("
            "AttestedConformanceInvocationAuthority))",
        ),
    )
    validator_authority_hashes = re.findall(
        r'hash\("([^"]+)",\s*canonical_payload_without_derived_digests\('
        r"(authority)\)\)",
        authority_validator.group("body") if authority_validator else "",
    )
    if authority_assignments != expected_authority_assignments or validator_authority_hashes != [
        ("attested-conformance-authority/v1", "authority")
    ]:
        errors.append("Attested authority schema/validator domain equality mismatch")
    record_assignments = _task9_owner_assignments(
        raw_attested, "AttestedConformanceExecutionRecord"
    )
    attested_algorithms = attested[
        attested.find("run_attested_release_conformance equations:") :
    ]
    expected_record_assignments = (
        (
            "evaluation_input_digest",
            'hash("attested-conformance-record-input/v1", '
            "authority.conformance_invocation_digest, "
            "session.measured_execution_environment_digest, canonical(binding), "
            "binding.fixture_binding_digest)",
        ),
        (
            "conformance_execution_record_digest",
            'hash("attested-conformance-record/v1", '
            "canonical_payload_without_derived_digests("
            "AttestedConformanceExecutionRecord))",
        ),
    )
    record_algorithm_hashes = re.findall(
        r'evaluation_input_digest := hash\("([^"]+)",\s*'
        r"(authority\.conformance_invocation_digest, "
        r"session\.measured_execution_environment_digest, canonical\(binding\), "
        r"binding\.fixture_binding_digest)\)",
        attested_algorithms,
    )
    expected_record_hash = (
        "attested-conformance-record-input/v1",
        "authority.conformance_invocation_digest, session.measured_execution_environment_digest, canonical(binding), binding.fixture_binding_digest",
    )
    if record_assignments != expected_record_assignments or record_algorithm_hashes != [
        expected_record_hash,
        expected_record_hash,
    ]:
        errors.append("Attested record-input three-site normalized equality mismatch")

    authority_accesses = set(
        re.findall(r"(?<![A-Za-z0-9_.])authority\.([a-z][a-z0-9_]*)", attested)
    )
    if not authority_accesses.issubset(CONFORMANCE_ATTESTED_AUTHORITY_FIELDS):
        errors.append("caller-owned Attested trust fallback is forbidden")
    approval_authority_accesses = set(
        re.findall(r"\bapproval_authority\.([a-z][a-z0-9_]*)", attested)
    )
    if not approval_authority_accesses.issubset(
        CONFORMANCE_RELEASE_APPROVAL_AUTHORITY_FIELDS
    ):
        errors.append("caller-owned release approval key fallback is forbidden")
    approval_accesses = set(re.findall(r"\bapproval\.([a-z][a-z0-9_]*)", attested))
    if not approval_accesses.issubset(
        {"authority", "signature", "release_approval_artifact_digest"}
    ):
        errors.append("caller-owned release approval key fallback is forbidden")
    if (
        attested.count(
            "trusted_key_material := store.trusted_key_material_by_id[policy.trusted_attestation_key_id]"
        )
        != 2
        or len(re.findall(r"\btrusted_key_material\s*:=", attested)) != 2
    ):
        errors.append("caller-owned Attested trust fallback is forbidden")
    if (
        attested.count(
            "trusted_approver_public_key_material := root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]"
        )
        != 1
        or len(
            re.findall(r"\btrusted_approver_public_key_material\s*:=", attested)
        )
        != 1
    ):
        errors.append("caller-owned release approval key fallback is forbidden")

    security_source_counts = {
        "trust_root := profile.conformance_trust_root": 1,
        "conformance_trust_root := profile.conformance_trust_root": 1,
        "release_trust_root := profile.release_trust_root": 1,
        "(store, policy) := resolve_conformance_trust_root(trust_root)": 3,
        "(current_conformance_store, current_conformance_policy) := resolve_conformance_trust_root(conformance_trust_root)": 1,
        "root := resolve_release_approval_trust_root(release_trust_root)": 1,
        "session := validate_and_consume_measurement_envelope(": 1,
        "measured_environment := session.measured_environment": 1,
    }
    security_assignment_counts = {
        r"(?<![A-Za-z0-9_])trust_root\s*:=": 1,
        r"(?<![A-Za-z0-9_])conformance_trust_root\s*:=": 1,
        r"(?<![A-Za-z0-9_])release_trust_root\s*:=": 1,
        r"(?<![A-Za-z0-9_])root\s*:=": 1,
        r"(?<![A-Za-z0-9_])session\s*:=": 1,
        r"(?<![A-Za-z0-9_])measured_environment\s*:=": 1,
        r"(?<![A-Za-z0-9_])store\s*:=": 0,
        r"(?<![A-Za-z0-9_])policy\s*:=": 0,
    }
    security_sources_valid = all(
        len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(token)}", attested))
        == count
        for token, count in security_source_counts.items()
    ) and all(
        len(re.findall(pattern, attested)) == count
        for pattern, count in security_assignment_counts.items()
    )
    if security_sources_valid:
        trust_root_position = attested.index(
            "trust_root := profile.conformance_trust_root"
        )
        store_positions = [
            match.start()
            for match in re.finditer(
                re.escape(
                    "(store, policy) := resolve_conformance_trust_root(trust_root)"
                ),
                attested,
            )
        ]
        session_position = attested.index(
            "session := validate_and_consume_measurement_envelope("
        )
        measured_position = attested.index(
            "measured_environment := session.measured_environment"
        )
        release_root_position = attested.index(
            "release_trust_root := profile.release_trust_root"
        )
        conformance_release_root_position = attested.index(
            "conformance_trust_root := profile.conformance_trust_root"
        )
        conformance_release_resolve_position = attested.index(
            "(current_conformance_store, current_conformance_policy) := "
            "resolve_conformance_trust_root(conformance_trust_root)"
        )
        root_position = attested.index(
            "root := resolve_release_approval_trust_root(release_trust_root)"
        )
        security_sources_valid = (
            all(trust_root_position < position for position in store_positions)
            and store_positions[0] < session_position < measured_position
            and conformance_release_root_position
            < conformance_release_resolve_position
            < root_position
            and release_root_position < root_position
        )
    if not security_sources_valid:
        errors.append("Attested security variable source/dominance mismatch")

    run_algorithm_match = re.search(
        r"(?ms)^run_attested_release_conformance equations:\s*$\n"
        r"(?P<body>.*?)"
        r"(?=^validate_and_consume_measurement_envelope equations, in this exact order)",
        raw_attested,
    )
    run_algorithm = _normalize_contract_text(
        [run_algorithm_match.group("body") if run_algorithm_match else ""]
    )
    run_root_flow_tokens = (
        "trust_root := profile.conformance_trust_root",
        "(store, policy) := resolve_conformance_trust_root(trust_root)",
        "session := validate_and_consume_measurement_envelope( authority, runner, trust_root, measured_environment)",
        "return validate_and_seal_attested_conformance( authority, trust_root, session, observed, ledger, attestation)",
    )
    if not (
        run_algorithm_match is not None
        and all(run_algorithm.count(token) == 1 for token in run_root_flow_tokens)
        and [run_algorithm.index(token) for token in run_root_flow_tokens]
        == sorted(run_algorithm.index(token) for token in run_root_flow_tokens)
        and "profile.release_trust_root" not in run_algorithm
    ):
        errors.append("Attested conformance root source-to-sink flow mismatch")

    release_result = re.search(
        r"ReleasePolicyApplicationResult\s*:=\s*(.*?)\s*"
        r"AttestedConformanceReport\.findings:",
        attested,
    )
    if (
        release_result is None
        or release_result.group(1)
        != "Completed { decision: ReleaseDecision } | InternalViolation { violation: InternalContractViolation }"
    ):
        errors.append("ReleasePolicyApplicationResult exact two-arm union mismatch")

    attested_result = re.search(
        r"AttestedConformanceRunResult\s*:=\s*(.*?)\s*ReleaseBlockReason\s*:=",
        attested,
    )
    if (
        attested_result is None
        or attested_result.group(1)
        != "Completed { artifact: AttestedConformanceSealArtifact } | InternalViolation { violation: InternalContractViolation }"
    ):
        errors.append("AttestedConformanceRunResult exact two-arm union mismatch")
    release_decision = re.search(
        r"ReleaseDecision\s*:=\s*(.*?)\s*ReleasePolicyApplicationResult\s*:=",
        attested,
    )
    if (
        release_decision is None
        or release_decision.group(1)
        != "ReleaseAllowed { subject_digest, approved: ApprovedDigestSet } | ReleaseBlocked { subject_digest, reasons: NonEmpty<ReleaseBlockReason> }"
    ):
        errors.append("ReleaseDecision exact two-arm union mismatch")

    basic_public = re.search(
        r"public exported ports \(exact\):\s*(.*?)\s*"
        r"run_basic_offline_conformance equations:",
        basic,
    )
    expected_basic_public = (
        "run_basic_offline_conformance( authority: "
        "BasicOfflineConformanceAuthority ) -> BasicOfflineRunResult"
    )
    attested_public = re.search(
        r"public exported ports \(exact\):\s*(.*?)\s*"
        r"internal non-exported ports:",
        attested,
    )
    expected_attested_public = (
        "run_attested_release_conformance( authority: "
        "AttestedConformanceInvocationAuthority, profile: "
        "AttestedReleaseConformance, measured_environment: "
        "MeasuredExecutionEnvironment ) -> AttestedConformanceRunResult "
        "apply_release_policy( artifact: AttestedConformanceSealArtifact, "
        "approval: ReleaseApprovalArtifact, policy: ReleasePolicy, profile: "
        "AttestedReleaseConformance ) -> ReleasePolicyApplicationResult"
    )
    if (
        basic_public is None
        or basic_public.group(1) != expected_basic_public
        or attested_public is None
        or attested_public.group(1) != expected_attested_public
    ):
        errors.append("conformance exact three public conformance ports mismatch")
    for public_signature in (
        expected_basic_public,
        "run_attested_release_conformance( authority: "
        "AttestedConformanceInvocationAuthority, profile: "
        "AttestedReleaseConformance, measured_environment: "
        "MeasuredExecutionEnvironment ) -> AttestedConformanceRunResult",
        "apply_release_policy( artifact: AttestedConformanceSealArtifact, "
        "approval: ReleaseApprovalArtifact, policy: ReleasePolicy, profile: "
        "AttestedReleaseConformance ) -> ReleasePolicyApplicationResult",
    ):
        if normalized_document.count(public_signature) != 1:
            errors.append("whole conformance exported port inventory mismatch")

    internal_ports = re.search(
        r"internal non-exported ports:\s*(.*?)\s*"
        r"derive_approved_digest_set fields:",
        attested,
    )
    internal_names = re.findall(
        r"\b([a-z][a-z0-9_]*)\(.*?\)\s*->",
        internal_ports.group(1) if internal_ports else "",
    )
    if internal_names != [
        "validate_and_seal_attested_conformance",
        "validate_and_consume_measurement_envelope",
        "derive_attested_conformance_report",
        "derive_approved_digest_set",
    ]:
        errors.append("conformance internal non-exported port inventory mismatch")

    apply_signature = (
        "apply_release_policy( artifact: AttestedConformanceSealArtifact, "
        "approval: ReleaseApprovalArtifact, policy: ReleasePolicy, profile: "
        "AttestedReleaseConformance ) -> ReleasePolicyApplicationResult"
    )
    if (whole_local + " " + outside_local).count(apply_signature) != 1:
        errors.append("conformance unique exact Attested apply_release_policy mismatch")

    membership_counts = {
        "require policy.trusted_attestation_key_id in store.trusted_key_material_by_id; otherwise InternalViolation": 2,
        "if subject_digest not in approval_authority.store_snapshot.approved_by_subject: return Completed { decision: ReleaseBlocked { subject_digest, reasons: {ApprovalMissing} } }": 1,
        "require policy.trusted_approver_key_id in root.trusted_approver_public_key_material_by_id; otherwise InternalViolation": 1,
    }
    if any(attested.count(token) != count for token, count in membership_counts.items()):
        errors.append("conformance map lookup membership closure mismatch")

    attestation_membership = (
        "require policy.trusted_attestation_key_id in "
        "store.trusted_key_material_by_id; otherwise InternalViolation"
    )
    attestation_lookup = (
        "trusted_key_material := "
        "store.trusted_key_material_by_id[policy.trusted_attestation_key_id]"
    )
    approval_key_membership = (
        "require policy.trusted_approver_key_id in "
        "root.trusted_approver_public_key_material_by_id; otherwise InternalViolation"
    )
    approval_key_lookup = (
        "trusted_approver_public_key_material := "
        "root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]"
    )
    approval_subject_membership = (
        "if subject_digest not in "
        "approval_authority.store_snapshot.approved_by_subject:"
    )
    approval_subject_lookup = (
        "require approval_authority.store_snapshot."
        "approved_by_subject[subject_digest] == approved"
    )

    def token_positions(token: str) -> list[int]:
        return [match.start() for match in re.finditer(re.escape(token), attested)]

    attestation_memberships = token_positions(attestation_membership)
    attestation_lookups = token_positions(attestation_lookup)
    approval_key_memberships = token_positions(approval_key_membership)
    approval_key_lookups = token_positions(approval_key_lookup)
    approval_subject_memberships = token_positions(approval_subject_membership)
    approval_subject_lookups = token_positions(approval_subject_lookup)
    if not (
        len(attestation_memberships) == len(attestation_lookups) == 2
        and all(
            membership < lookup
            for membership, lookup in zip(
                attestation_memberships, attestation_lookups, strict=True
            )
        )
        and len(approval_key_memberships) == len(approval_key_lookups) == 1
        and approval_key_memberships[0] < approval_key_lookups[0]
        and len(approval_subject_memberships) == len(approval_subject_lookups) == 1
        and approval_subject_memberships[0] < approval_subject_lookups[0]
    ):
        errors.append("map lookup membership dominance mismatch")
    approval_missing_branch = (
        "if subject_digest not in approval_authority.store_snapshot.approved_by_subject: "
        "return Completed { decision: ReleaseBlocked { subject_digest, reasons: {ApprovalMissing} } }"
    )
    if attested.count(approval_missing_branch) != 1 or re.search(
        r"if subject_digest not in approval_authority\.store_snapshot\.approved_by_subject:.*?ReleaseAllowed",
        attested,
    ):
        errors.append("ApprovalMissing unique blocked branch mismatch")

    raw_apply_match = re.search(
        r"(?ms)^apply_release_policy first operation:\s*$\n"
        r"(?P<body>.*?)"
        r"(?=^  any failed recomputation, root, scheme, key-membership or signature equation)",
        raw_attested,
    )
    raw_apply = raw_apply_match.group("body") if raw_apply_match else ""
    apply_algorithm = _normalize_contract_text([raw_apply])
    apply_after_marker = "apply_release_policy equations after first operation:"
    apply_after_index = raw_apply.find(apply_after_marker)
    raw_apply_first_operation = (
        raw_apply[:apply_after_index] if apply_after_index >= 0 else ""
    )
    apply_first_operation_lines = tuple(
        line.rstrip()
        for line in raw_apply_first_operation.splitlines()
        if line.strip()
    )
    apply_lines = tuple(
        line.rstrip() for line in raw_apply.splitlines() if line.strip()
    )
    independent_recomputation_tokens = (
        "require canonical(approved) == canonical(approval_authority.approved)",
        "require derived_approved == derive_approved_digest_set(artifact, policy)",
        "require every stored nested digest equals its own enclosing payload recomputation; no approved/derived cross-equality",
    )
    late_approved_equality = (
        "require all ten ApprovedDigestSet fields are byte-equal only on this "
        "authenticated empty-diff path"
    )
    independent_recomputation_valid = (
        apply_after_index >= 0
        and apply_first_operation_lines
        == CONFORMANCE_RELEASE_FIRST_OPERATION_LINES
        and all(
            _normalize_contract_text([raw_apply_first_operation]).count(token) == 1
            for token in independent_recomputation_tokens
        )
        and re.search(
            r"(?im)^[ \t]*require .*every one of the ten ApprovedDigestSet fields "
            r"equals the recomputed value above[ \t]*$",
            raw_apply_first_operation,
        )
        is None
        and apply_algorithm.count(late_approved_equality) == 1
    )
    if not independent_recomputation_valid:
        errors.append("release approval independent recomputation boundary mismatch")

    expected_apply_signed_message = (
        "release_approval_signed_message := canonical_tuple( "
        "root.expected_release_policy_digest, "
        "root.expected_release_approval_store_snapshot_digest, "
        "policy.trusted_approver_key_id, policy.approval_signature_scheme, "
        "canonical(approved))"
    )
    current_closure_start_line = (
        "  require current_conformance_store.key_registry_digest =="
    )
    current_closure_end_line = (
        "          artifact.execution_ledger.observed_output.findings"
    )
    expected_current_closure_start = CONFORMANCE_RELEASE_AFTER_OPERATION_LINES.index(
        current_closure_start_line
    )
    expected_current_closure_end = CONFORMANCE_RELEASE_AFTER_OPERATION_LINES.index(
        current_closure_end_line
    )
    expected_current_closure_lines = CONFORMANCE_RELEASE_AFTER_OPERATION_LINES[
        expected_current_closure_start : expected_current_closure_end + 1
    ]
    raw_current_closure_match = re.search(
        r"(?ms)^  require current_conformance_store\.key_registry_digest ==\s*$\n"
        r"(?P<body>.*?)"
        r"(?=^  require current_conformance_policy\.trusted_attestation_key_id in\s*$)",
        raw_apply,
    )
    actual_current_closure_lines = tuple(
        line.rstrip()
        for line in (
            (current_closure_start_line + "\n" + raw_current_closure_match.group("body"))
            if raw_current_closure_match
            else ""
        ).splitlines()
        if line.strip()
    )
    if actual_current_closure_lines != expected_current_closure_lines:
        errors.append("release current conformance closure mismatch")

    current_conformance_root_tokens = (
        "conformance_trust_root := profile.conformance_trust_root",
        "(current_conformance_store, current_conformance_policy) := "
        "resolve_conformance_trust_root(conformance_trust_root)",
        "require current_conformance_policy.trusted_attestation_key_id in "
        "current_conformance_store.trusted_key_material_by_id; otherwise "
        "InternalViolation",
        "current_conformance_trusted_key_material := "
        "current_conformance_store.trusted_key_material_by_id[ "
        "current_conformance_policy.trusted_attestation_key_id]",
        "require verify_measurer_identity_and_freshness_evidence( "
        "conformance_trust_root.measurement_session_authority, "
        "current_measurement_evidence_message, artifact.measured_environment."
        "measurer_identity_and_freshness_evidence)",
        "require conformance_trust_root.measurement_session_authority."
        "has_terminal_consumption_receipt( terminal_consumption_receipt_key("
        "artifact.measured_environment))",
        "current_conformance_signed_message := canonical_tuple( "
        "artifact.runner_attestation.measured_execution_environment_digest, "
        "artifact.runner_attestation.executable_artifact_digest, artifact."
        "runner_attestation.trusted_attestation_key_id, artifact.runner_attestation."
        "attestation_signature_scheme, artifact.runner_attestation."
        "conformance_invocation_digest, artifact.runner_attestation."
        "observed_output_digest, artifact.runner_attestation.verifier_runner_digest)",
        "require verify_signature(current_conformance_trusted_key_material, "
        "current_conformance_policy.attestation_signature_scheme, "
        "current_conformance_signed_message, artifact.runner_attestation.signature)",
        "reverify sealed runner attestation under current profile root without "
        "consuming a measurement session or nonce",
    )
    current_conformance_positions = [
        apply_algorithm.find(token) for token in current_conformance_root_tokens
    ]
    if not (
        all(position >= 0 for position in current_conformance_positions)
        and current_conformance_positions == sorted(current_conformance_positions)
        and len(set(current_conformance_positions))
        == len(current_conformance_positions)
        and all(
            apply_algorithm.count(token) == 1
            for token in current_conformance_root_tokens
        )
    ):
        errors.append("release current conformance root revalidation mismatch")
    receipt_formula_tokens = (
        "terminal_consumption_receipt_key( environment: MeasuredExecutionEnvironment ) := "
        "canonical_tuple( environment.verifier_nonce, environment.conformance_invocation_digest, "
        "environment.verifier_runner_digest, environment.executable_artifact_digest, "
        "environment.execution_session_id, environment.process_identity, "
        "environment.container_identity, environment.execution_time_window, "
        "environment.measured_execution_environment_digest)",
        "terminal_consumption_receipt_key_value := "
        "terminal_consumption_receipt_key(measured_environment)",
        "atomically consumed exactly once while persisting a terminal receipt under that exact key",
        "session.nonce_consumption_receipt proves the terminal registry contains exact "
        "terminal_consumption_receipt_key(measured_environment)",
        "has_terminal_consumption_receipt( terminal_consumption_receipt_key("
        "artifact.measured_environment))",
    )
    receipt_positions = [attested.find(token) for token in receipt_formula_tokens]
    if not (
        all(position >= 0 for position in receipt_positions)
        and receipt_positions == sorted(receipt_positions)
        and all(attested.count(token) == 1 for token in receipt_formula_tokens)
    ):
        errors.append("terminal receipt production/seal/query closure mismatch")
    apply_tokens = (
        current_conformance_root_tokens[0],
        "release_trust_root := profile.release_trust_root",
        "derived_approved := derive_approved_digest_set(artifact, policy)",
        "approved_mismatches := canonical_schema_path_diff(approved, derived_approved)",
        *independent_recomputation_tokens,
        apply_after_marker,
        *current_conformance_root_tokens[1:],
        "root := resolve_release_approval_trust_root(release_trust_root)",
        approval_subject_membership,
        approval_subject_lookup,
        approval_key_membership,
        approval_key_lookup,
        expected_apply_signed_message,
        "require verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)",
        "if approved_mismatches is NonEmpty:",
        "require approved == derived_approved",
        late_approved_equality,
        "decision := apply the versioned pass/fail/insufficient rule after comparing all ten approved digest fields",
        "return Completed { decision }",
    )
    apply_positions = [apply_algorithm.find(token) for token in apply_tokens]
    exact_mismatch_guard = re.findall(
        r"(?m)^  if approved_mismatches is NonEmpty:[ \t]*$", raw_apply
    )
    exact_mismatch_return = re.findall(
        r"(?m)^    return Completed \{ decision: ReleaseBlocked \{ subject_digest, reasons:[ \t]*$\n"
        r"^      map\(approved_mismatches, path -> ApprovedDigestMismatch\(path\)\) \} \}[ \t]*$",
        raw_apply,
    )
    exact_missing_branch = re.findall(
        r"(?m)^  if subject_digest not in approval_authority\.store_snapshot\.approved_by_subject:[ \t]*$\n"
        r"^    return Completed \{ decision: ReleaseBlocked \{ subject_digest, reasons: \{ApprovalMissing\} \} \}[ \t]*$",
        raw_apply,
    )
    actual_return_statements = tuple(
        line.strip()
        for line in raw_apply.splitlines()
        if re.match(r"^[ \t]+return\b", line)
    )
    expected_return_statements = (
        "return Completed { decision: ReleaseBlocked { subject_digest, reasons: {ApprovalMissing} } }",
        "return Completed { decision: ReleaseBlocked { subject_digest, reasons:",
        "return Completed { decision }",
    )
    signature_to_mismatch_is_closed = re.search(
        r"(?m)^  verify release approval signature only with external-root public key material[ \t]*$\n"
        r"^  if approved_mismatches is NonEmpty:[ \t]*$",
        raw_apply,
    ) is not None
    release_control_valid = (
        raw_apply_match is not None
        and all(position >= 0 for position in apply_positions)
        and apply_positions == sorted(apply_positions)
        and len(set(apply_positions)) == len(apply_positions)
        and len(exact_mismatch_guard) == 1
        and len(exact_mismatch_return) == 1
        and len(exact_missing_branch) == 1
        and apply_lines == CONFORMANCE_RELEASE_APPLY_LINES
        and actual_return_statements == expected_return_statements
        and signature_to_mismatch_is_closed
        and "ReleaseAllowed" not in apply_algorithm
        and apply_algorithm.count(
            "decision := apply the versioned pass/fail/insufficient rule after comparing all ten approved digest fields"
        )
        == 1
        and apply_algorithm.count("return Completed { decision }") == 1
    )
    if not release_control_valid:
        errors.append("release decision authenticated control-flow mismatch")
    if apply_algorithm.count(expected_apply_signed_message) != 1:
        errors.append("release approval signed-message approved binding mismatch")
    if len(exact_mismatch_guard) != 1 or len(exact_mismatch_return) != 1:
        errors.append("ApprovedDigestMismatch reachable blocked branch mismatch")

    if any(
        attested.count(rule) != 1 or outside_local.count(rule) != 0
        for rule in CONFORMANCE_ATTESTED_SECURITY_RULES
    ):
        errors.append("Attested security rule inventory mismatch")

    chapter_12_4 = _section_between(html, "c12-4", "c12-5")
    chapter_12_4_text = _normalize_contract_text([unescape(chapter_12_4 or "")])
    if chapter_12_4_text.count(
        "apply_release_policy(..., profile: AttestedReleaseConformance) -> ReleasePolicyApplicationResult"
    ) != 1:
        errors.append("Chapter 12.4 release result wrapper mismatch")
    if re.search(
        r"(?:ConformanceArtifact\s*:=|BasicOfflineReportArtifact\s*->\s*"
        r"AttestedConformanceSealArtifact)",
        whole_local,
    ):
        errors.append("conformance profiles must not expose a common artifact or upgrade path")


def check_conformance_trust_boundary(html: str, errors: list[str]) -> None:
    local_text = _conformance_local_texts(html, []).get(
        ("profile", "attested-release"), ""
    )

    authority_match = re.search(
        r"AttestedConformanceInvocationAuthority:\s*(.*?)\s*RunnerAttestation:",
        local_text,
        re.DOTALL,
    )
    if authority_match is None:
        errors.append("conformance authority schema is not locally extractable")
    else:
        authority_schema = authority_match.group(1)
        for forbidden_field in (
            "trust_store_snapshot: TrustStoreSnapshot",
            "runner_attestation_policy: RunnerAttestationPolicy",
        ):
            if forbidden_field in authority_schema:
                errors.append(
                    "AttestedConformanceInvocationAuthority must hold only external-root refs: "
                    f"{forbidden_field}"
                )

    first_marker = "validate_and_seal_attested_conformance first operation:"
    after_marker = "validate_and_seal_attested_conformance equations after first operation:"
    first_index = local_text.find(first_marker)
    after_index = local_text.find(after_marker, first_index + 1)
    first_operation_tokens = (
        "recompute authority.production_subject.subject_digest",
        "recompute every authority.fixture_set binding, nested fixture and fixture_set_digest",
        "recompute authority.validation_policy.validation_policy_digest",
        "recompute authority.gate_manifest and every nested entry/clause digest",
        "recompute authority.gate_manifest.blocker_scope_policy.blocker_scope_policy_digest",
        "recompute authority.verifier_runner.verifier_runner_digest",
        "recompute authority.conformance_invocation_digest only after all nested recomputations",
    )
    if first_index < 0 or after_index < 0:
        errors.append("validate_and_seal_attested_conformance first-operation boundary is missing")
    else:
        for token in first_operation_tokens:
            token_index = local_text.find(token, first_index, after_index)
            if token_index < 0:
                errors.append(
                    "validate_and_seal_attested_conformance first operation missing nested "
                    f"recomputation: {token}"
                )
    if (
        "runner signs canonical_tuple( policy.expected_execution_environment_digest"
        in local_text
    ):
        errors.append(
            "runner attestation must sign protected measured environment, not policy expected value"
        )


def check_conformance_report_findings(html: str, errors: list[str]) -> None:
    attested = _conformance_local_texts(html, []).get(
        ("profile", "attested-release"), ""
    )
    exact_schema = re.compile(
        r"AttestedConformanceReport:\s*.*?"
        r"verdict:\s*pass\s*\|\s*fail\s*\|\s*insufficient\s*"
        r"findings:\s*OrderedMap<FindingId,\s*ConformanceFinding>",
        re.DOTALL,
    )
    if exact_schema.search(attested) is None:
        errors.append(
            "AttestedConformanceReport.findings 必须与 observed output 同型："
            "OrderedMap<FindingId, ConformanceFinding>"
        )


def check_release_approved_digest_contract(html: str, errors: list[str]) -> None:
    for legacy_pattern in (r"六个\s+approved digest", r"四个\s+digest"):
        match = re.search(legacy_pattern, html)
        if match is not None:
            errors.append(
                "derive_approved_digest_set 禁止 release digest 子集："
                f"{match.group(0)}"
            )

    attested = _conformance_local_texts(html, []).get(
        ("profile", "attested-release"), ""
    )
    release_policy = re.search(
        r"ReleasePolicy:\s*.*?release_policy_digest\s*:=",
        attested,
        re.DOTALL,
    )
    if release_policy is None or not re.search(
        r"approved\s*==\s*derive_approved_digest_set\(artifact,\s*self\)",
        release_policy.group(0),
    ):
        errors.append(
            "ReleasePolicy.pass 必须引用 derive_approved_digest_set(artifact, self)"
        )

    isolation = re.search(
        r'<h3 id="c12-6">(.*?)<h2 id="c13">', html, re.DOTALL
    )
    if isolation is None or not re.search(
        r"derive_approved_digest_set\(artifact,\s*policy\).*?"
        r"all ten approved digest fields",
        isolation.group(1),
        re.DOTALL,
    ):
        errors.append(
            "12.6 必须以 derive_approved_digest_set 覆盖 all ten approved digest fields"
        )


def check_required(html: str, errors: list[str]) -> None:
    for text in REQUIRED_TEXT:
        if text not in html:
            errors.append(f"缺少 v4.2 必需契约：{text}")
    check_derived_digest_exclusions(html, errors)
    check_product_gate_staging(html, errors)
    check_conformance_deployment_profiles(html, errors)
    check_conformance_trust_boundary(html, errors)
    check_conformance_report_findings(html, errors)
    check_release_approved_digest_contract(html, errors)
    check_module_contracts(html, errors)
    check_task7_modeling_contracts(html, errors)
    check_task7_behavioral_contracts(html, errors)
    check_task9_production_identity_isolation(html, errors)


def check_forbidden(html: str, errors: list[str]) -> None:
    for text in FORBIDDEN_TEXT:
        if text in html:
            errors.append(f"残留禁用语义：{text}")


def parse_gates(html: str) -> list[str]:
    return re.findall(r'<tr data-gate="([A-Z0-9-]+)">', html)


def check_chapters(html: str, errors: list[str]) -> None:
    for chapter in range(15):
        anchor = f'id="c{chapter}"'
        count = html.count(anchor)
        if count != 1:
            errors.append(f"章节锚点 {anchor} 应恰出现一次，实际 {count} 次")


def check_gates(html: str, errors: list[str]) -> None:
    gates = parse_gates(html)
    duplicates = sorted({gate for gate in gates if gates.count(gate) > 1})
    if duplicates:
        errors.append(f"结构门编号重复：{duplicates}")

    actual = set(gates)
    missing = sorted(EXPECTED_GATES - actual)
    extra = sorted(actual - EXPECTED_GATES)
    if missing:
        errors.append(f"缺少结构门：{missing}")
    if extra:
        errors.append(f"出现计划外结构门：{extra}")


def check_diagrams(html: str, errors: list[str]) -> None:
    for placeholder in INCOMPATIBLE_DIAGRAMS:
        if placeholder in html:
            errors.append(f"引用了仍含 v3 证明语义的图：{placeholder}")
    figures = re.findall(
        r'<figure\b[^>]*data-diagram-id="([^"]+)"[^>]*>.*?'
        r'<pre class="mermaid-source"><code>(.*?)</code></pre>',
        html,
        re.DOTALL,
    )
    if tuple(diagram_id for diagram_id, _ in figures) != TASK8_DIAGRAM_IDS:
        errors.append("Task8 authoritative diagram ID/order 不闭合")
    sources = dict(figures)
    result_source = unescape(sources.get("result-gate-comparison", ""))
    result_participants = re.findall(
        r"(?m)^  participant ([A-Za-z][A-Za-z0-9_]*) as ([^\r\n]+)$",
        result_source,
    )
    expected_result_participants = [
        ("PB", "ProjectionBundleBuild"),
        ("SA", "BackendResultSourceAuthority"),
        ("BE", "Backend"),
        ("VA", "BackendValueGateAuthority"),
        ("GL", "GateManifest and GateExecutionLedger"),
        ("RC", "BackendResultCandidate"),
        ("BA", "BackendSealArtifact"),
        ("ARM", "ComparisonArmAuthority"),
        ("CS", "ComparisonSourceAuthority"),
        ("CC", "ComparisonResultCandidate"),
        ("CA", "ComparisonSealArtifact"),
        ("BO", "Basic offline validator"),
        ("AR", "Attested release validator"),
    ]
    result_profile_match = re.search(
        r"(?ms)^  opt OfflineConformanceRequested[ \t]*$\n"
        r"^    alt BasicOfflineConformance \(default\)[ \t]*$\n"
        r"(?P<body>.*?)"
        r"^    end[ \t]*$\n"
        r"^  end[ \t]*$",
        result_source,
    )
    result_profile_lines = [
        line.rstrip()
        for line in (
            result_profile_match.group("body") if result_profile_match else ""
        ).splitlines()
        if line.strip()
    ]
    expected_result_profile_lines = [
        "      BO->>GL: exact subject manifest fixtures policy and runner",
        "      GL-->>BO: clause observations and GateExecutionLedger",
        "      BO->>BO: construct BasicOfflineReportArtifact",
        "    else AttestedReleaseConformance",
        "      AR->>GL: same fixtures plus protected measured session",
        "      GL-->>AR: clause observations and GateExecutionLedger",
        "      AR->>AR: construct AttestedConformanceSealArtifact",
        "      AR->>AR: apply_release_policy with current profile roots",
    ]
    result_sequence_lines = tuple(
        line.rstrip()
        for line in result_source.splitlines()
        if line.strip() and not line.startswith("%%{init:")
    )
    expected_result_sequence_lines = (
        "sequenceDiagram",
        "  participant PB as ProjectionBundleBuild",
        "  participant SA as BackendResultSourceAuthority",
        "  participant BE as Backend",
        "  participant VA as BackendValueGateAuthority",
        "  participant GL as GateManifest and GateExecutionLedger",
        "  participant RC as BackendResultCandidate",
        "  participant BA as BackendSealArtifact",
        "  participant ARM as ComparisonArmAuthority",
        "  participant CS as ComparisonSourceAuthority",
        "  participant CC as ComparisonResultCandidate",
        "  participant CA as ComparisonSealArtifact",
        "  participant BO as Basic offline validator",
        "  participant AR as Attested release validator",
        "  PB->>SA: exact ProjectionResult and projection ledger",
        "  alt Ready",
        "    SA->>BE: immutable Ready view",
        "    BE-->>VA: EstimateCandidate and witness",
        "  else Blocked or NotRequested",
        "    SA->>VA: matching non-value branch",
        "  end",
        "  VA->>GL: exact backend-value clauses",
        "  GL-->>VA: pre-seal ledger prefix",
        "  VA->>RC: assemble bound branch",
        "  Note right of RC: candidate remains internal until seal",
        "  RC->>GL: result-seal clauses",
        "  GL-->>BA: sealed ledger and BackendResult",
        "  opt ComparisonRequest",
        "    BA->>ARM: exact comparison arms",
        "    ARM->>CS: identities and ComparisonBasisPair",
        "    CS->>CC: compare per metric",
        "    CC->>GL: G-REP2 clauses",
        "    GL-->>CA: sealed ComparisonResult",
        "  end",
        "  opt OfflineConformanceRequested",
        "    alt BasicOfflineConformance (default)",
        "      BO->>GL: exact subject manifest fixtures policy and runner",
        "      GL-->>BO: clause observations and GateExecutionLedger",
        "      BO->>BO: construct BasicOfflineReportArtifact",
        "    else AttestedReleaseConformance",
        "      AR->>GL: same fixtures plus protected measured session",
        "      GL-->>AR: clause observations and GateExecutionLedger",
        "      AR->>AR: construct AttestedConformanceSealArtifact",
        "      AR->>AR: apply_release_policy with current profile roots",
        "    end",
        "  end",
        "  Note over BO,AR: optional offline sidecars excluded from production digests",
    )
    if result_sequence_lines != expected_result_sequence_lines:
        errors.append("Task9 Figure 10 exact sequence structure mismatch")
    if (
        not re.search(r"(?m)^sequenceDiagram\s*$", result_source)
        or result_participants != expected_result_participants
        or result_profile_match is None
        or result_profile_lines != expected_result_profile_lines
        or result_source.count("opt OfflineConformanceRequested") != 1
        or "opt BasicOfflineConformance" in result_source
        or "opt AttestedReleaseConformance" in result_source
        or result_source.count(
            "Note over BO,AR: optional offline sidecars excluded from production digests"
        )
        != 1
    ):
        errors.append(
            "Task9 Figure 10 exact mutually-exclusive profile ownership mismatch"
        )

    architecture_edges: dict[str, set[tuple[str, str, str]]] = {}
    for diagram_id in TASK8_ARCHITECTURE_DIAGRAM_IDS:
        source = unescape(sources.get(diagram_id, ""))
        if '"useGradient": false' not in source or not re.search(
            r"(?m)^block-beta\s*$", source
        ):
            errors.append(f"Task8 architecture diagram kind mismatch: {diagram_id}")
        if re.search(r"(?m)^flowchart\b", source):
            errors.append(f"Task8 architecture diagram forbids flowchart: {diagram_id}")
        nodes = dict(re.findall(r'(?m)^\s+([A-Za-z][A-Za-z0-9]*)\["([^"]+)"\]\s*$', source))
        edges = set(
            re.findall(
                r"(?m)^\s+([A-Za-z][A-Za-z0-9]*)\s+(-->|-\.->)\s+"
                r"([A-Za-z][A-Za-z0-9]*)\s*$",
                source,
            )
        )
        architecture_edges[diagram_id] = edges
        if nodes != TASK8_ARCHITECTURE_NODES[diagram_id]:
            errors.append(f"Task8 architecture exact node/label set mismatch: {diagram_id}")
        if diagram_id == "plan-projection-module-architecture":
            solid = {edge for edge in edges if edge[1] == "-->"}
            dotted = {edge for edge in edges if edge[1] == "-.->"}
            if solid != TASK8_PLAN_ARCHITECTURE_SOLID_EDGES:
                errors.append("Task8 plan architecture exact provider-to-caller edges mismatch")
            if dotted != TASK8_PLAN_ARCHITECTURE_DOTTED_EDGES:
                errors.append("Task8 plan architecture exact dotted DTO lineage mismatch")

    table_match = re.search(
        r'<table\b[^>]*data-module-architecture="normative"[^>]*>'
        r'(?P<body>.*?)</table>',
        html,
        re.DOTALL,
    )
    parsed_rows: dict[str, tuple[str, str, str, str, str]] = {}
    dependency_rows: dict[str, tuple[str, str]] = {}
    malformed = table_match is None
    if table_match is not None:
        header = re.search(r"<tr>(.*?)</tr>", table_match.group("body"), re.DOTALL)
        header_cells = [
            re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", cell))).strip()
            for cell in re.findall(r"<th[^>]*>(.*?)</th>", header.group(1) if header else "", re.DOTALL)
        ]
        malformed = header_cells != ["层", "模块 ID", "职责", "核心契约类型", "正式入口 port", "允许调用依赖"]
        for row in re.finditer(r"<tr\b(?P<attrs>[^>]*)>(?P<body>.*?)</tr>", table_match.group("body"), re.DOTALL):
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row.group("body"), re.DOTALL)
            if not cells:
                continue
            attrs = dict(re.findall(r'([a-z-]+)="([^"]*)"', row.group("attrs")))
            module = attrs.get("data-module-id", "")
            normalized = tuple(
                re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", cell))).strip()
                for cell in cells
            )
            if len(normalized) != 6 or not module or module in parsed_rows:
                malformed = True
                continue
            layer = attrs.get("data-layer", "")
            deps = attrs.get("data-allowed-dependencies", "")
            parsed_rows[module] = (normalized[0], normalized[2], normalized[3], normalized[4], normalized[5])
            dependency_rows[module] = (layer, deps)
            if normalized[1] != module or normalized[0] != ("离线" if layer == "OFFLINE" else layer):
                malformed = True
    if malformed or parsed_rows != TASK8_MODULE_ROWS or dependency_rows != TASK8_MODULE_DEPENDENCIES:
        errors.append("Task8 Chapter 1 exact six-cell module table 不闭合")

    layered_module_nodes = {
        label: node
        for node, label in TASK8_ARCHITECTURE_NODES[
            "layered-module-architecture"
        ].items()
        if label in TASK8_MODULE_DEPENDENCIES
    }
    expected_layered_solid = {
        (layered_module_nodes[dependency], "-->", layered_module_nodes[caller])
        for caller, (_, dependencies) in dependency_rows.items()
        if caller in layered_module_nodes
        for dependency in dependencies.split(",")
        if dependency in layered_module_nodes
    }
    layered_edges = architecture_edges.get("layered-module-architecture", set())
    layered_solid = {edge for edge in layered_edges if edge[1] == "-->"}
    layered_dotted = {edge for edge in layered_edges if edge[1] == "-.->"}
    if (
        set(layered_module_nodes) != set(dependency_rows)
        or layered_solid != expected_layered_solid
    ):
        errors.append("Task8 layered solid edges must reverse table dependencies")
    if layered_dotted != TASK8_LAYERED_DOTTED_EDGES:
        errors.append("Task8 layered dotted DTO lineage mismatch")

    graph = {
        module: tuple(filter(None, deps.split(",")))
        for module, (_, deps) in dependency_rows.items()
    }
    graph_invalid = set(graph) != set(TASK8_MODULE_DEPENDENCIES) or any(
        dep not in graph for deps in graph.values() for dep in deps
    )
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(module: str) -> bool:
        if module in visiting:
            return False
        if module in visited:
            return True
        visiting.add(module)
        if any(not visit(dep) for dep in graph.get(module, ())):
            return False
        visiting.remove(module)
        visited.add(module)
        return True
    if graph_invalid or any(not visit(module) for module in graph):
        errors.append("Task8 module dependency DAG 不闭合")

    for module, row in TASK8_MODULE_ROWS.items():
        section = re.search(
            rf'<section\b[^>]*data-module="{re.escape(module)}"[^>]*>(?P<body>.*?)</section>',
            html,
            re.DOTALL,
        )
        interface = re.search(
            rf'<h4\b[^>]*id="mc-{re.escape(module)}-interface"[^>]*>.*?</h4>'
            rf'(?P<body>.*?)(?=<h4\b|$)',
            section.group("body") if section else "",
            re.DOTALL,
        )
        interface_text = _normalize_contract_text(
            [unescape(re.sub(r"<[^>]+>", " ", interface.group("body")))]
        ) if interface else ""
        for port in row[3].split("; "):
            if f"{port}(" not in interface_text:
                errors.append(f"Task8 table port missing from {module} interface: {port}")

    plan_section = re.search(
        r'<section\b[^>]*data-module="plan-projection"[^>]*>(?P<body>.*?)</section>',
        html,
        re.DOTALL,
    )
    plan_body = plan_section.group("body") if plan_section else ""
    core_data = re.search(
        r'<h4\b[^>]*id="mc-plan-projection-data"[^>]*>.*?</h4>'
        r'(?P<body>.*?)(?=<h4\b|$)',
        plan_body,
        re.DOTALL,
    )
    core_data_body = core_data.group("body") if core_data else ""
    reference_lists = list(
        re.finditer(
            r'<ul\b(?P<attrs>[^>]*)>(?P<body>.*?)</ul>',
            core_data_body,
            re.DOTALL,
        )
    )
    refs: list[str] = []
    refs_malformed = len(reference_lists) != 1
    if reference_lists:
        list_attrs = dict(
            re.findall(r'([:\w-]+)="([^"]*)"', reference_lists[0].group("attrs"))
        )
        refs_malformed = refs_malformed or list_attrs.get(
            "data-authoritative-contract-references"
        ) != "plan-projection"
        items = list(
            re.finditer(
                r'<li\b(?P<attrs>[^>]*)>(?P<body>.*?)</li>',
                reference_lists[0].group("body"),
                re.DOTALL,
            )
        )
        for item in items:
            item_attrs = dict(
                re.findall(r'([:\w-]+)="([^"]*)"', item.group("attrs"))
            )
            ref = unescape(item_attrs.get("data-contract-ref", ""))
            visible = re.sub(
                r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", item.group("body")))
            ).strip()
            if not ref or ref != visible:
                refs_malformed = True
            refs.append(ref)
    if refs_malformed or tuple(refs) != TASK8_PLAN_AUTHORITATIVE_CONTRACT_REFS:
        errors.append("Task8 plan-projection exact authoritative contract references 不闭合")
    core_data_text = unescape(re.sub(r"<[^>]+>", "\n", core_data_body))
    if "PlanProjectionOwnedContracts" in plan_body or re.search(
        r"(?m)^\s*[A-Za-z][A-Za-z0-9_]*(?:<[^>\n]+>)?\s*(?::=|:)(?:\s|$)",
        core_data_text,
    ):
        errors.append("Task8 plan-projection forbids local wrapper schema")
    if re.search(r"(?:CoreBuildResult|ProjectionResult|ProjectionCandidate)[^\r\n]*:=", unescape(plan_body)):
        errors.append("Task8 plan-projection forbids local result/union second truth")
    plan_text = _normalize_contract_text([unescape(re.sub(r"<[^>]+>", " ", plan_body))])
    required_plan_ports = (
        "bind_core( plan: RuntimeEventPlan, hardware: HardwareProfile, deployment: ExecutionDeployment, policy: HardwareBindingPolicySnapshot ) -> CoreBuildResult<SimulationPlanCore> | InternalContractViolation",
        "evaluate_and_finalize_projection_bundle( authority: ProjectionBundleAuthority ) -> ProjectionBundleBuild | InternalContractViolation",
        "build_memory_projection_candidate( request: RequestSnapshot, evaluation_identity: EvaluationInstanceIdentity, core: SimulationPlanCore ) -> ProjectionCandidate<MemoryEventView> | InternalContractViolation",
        "build_time_projection_candidate( request: RequestSnapshot, evaluation_identity: EvaluationInstanceIdentity, core: SimulationPlanCore ) -> ProjectionCandidate<TimeEventView> | InternalContractViolation",
        "build_gate_evaluation_authority( subject: GateEvaluationSubject, manifest: GateManifest, runner: GateRunnerSnapshot ) -> GateEvaluationAuthority | InternalContractViolation",
        "expected_invocation_domain( authority: GateEvaluationAuthority, base: GateExecutionLedger ) -> ExpectedInvocationDomain",
        "run_gate_domain( authority: GateEvaluationAuthority, base: GateExecutionLedger, invocation_domain: OrderedSet<GateInvocationId> ) -> GateExecutionLedger | InternalContractViolation",
    )
    if any(port not in plan_text for port in required_plan_ports):
        errors.append("Task8 plan-projection typed port signature 不闭合")

    document_text = unescape(html)
    pure_candidate_equation = re.findall(
        r"(?m)^expected_projection_candidate\(request, evaluation_identity, backend,\s*\n"
        r"\s+requested_branch\) :=\s*$",
        document_text,
    )
    candidate_postconditions = re.findall(
        r"require\s+returned_candidate\s*==\s*expected_projection_candidate\(",
        document_text,
    )
    if (
        len(pure_candidate_equation) != 1
        or len(candidate_postconditions) != 2
        or "finalize_projection_candidate" in document_text
        or "projection_candidate_from_request" in document_text
        or re.search(r"\bcall\s+expected_projection_candidate", document_text)
    ):
        errors.append("Task8 candidate construction requires pure expected_projection_candidate equation")

    builder_binding_invalid = False
    for backend in ("memory", "time"):
        builder = re.search(
            rf"(?ms)^build_{backend}_projection_candidate\(request, evaluation_identity, core\):"
            rf"(?P<body>.*?)(?=^build_(?:memory|time)_projection_candidate|"
            rf"^evaluate_and_finalize_projection_bundle)",
            document_text,
        )
        postcondition_backends = re.findall(
            r"require\s+returned_candidate\s*==\s*"
            r"expected_projection_candidate\(\s*request,\s*evaluation_identity,\s*"
            r"(memory|time),\s*requested_branch\s*\)",
            builder.group("body") if builder else "",
        )
        if postcondition_backends != [backend]:
            builder_binding_invalid = True
    if builder_binding_invalid:
        errors.append("Task8 candidate builder postcondition binding mismatch")

    sweep = re.search(
        r'<h3\b[^>]*id="c13-2"[^>]*>.*?</h3>\s*'
        r'<pre><code>(?P<body>.*?)</code></pre>',
        html,
        re.DOTALL,
    )
    sweep_text = unescape(sweep.group("body") if sweep else "")
    blocked_candidate_bindings = sorted(
        re.findall(
            r"(memory|time)_projection_candidate\s*=\s*"
            r"expected_projection_candidate\(\s*request_snapshot,\s*"
            r"evaluation_identity,\s*(memory|time),",
            sweep_text,
        )
    )
    ready_candidate_bindings = sorted(
        re.findall(
            r"(memory|time)_projection_candidate\s*=\s*"
            r"build_(memory|time)_projection_candidate\(",
            sweep_text,
        )
    )
    expected_candidate_bindings = [("memory", "memory"), ("time", "time")]
    if (
        blocked_candidate_bindings != expected_candidate_bindings
        or ready_candidate_bindings != expected_candidate_bindings
    ):
        errors.append("Task8 product orchestration candidate binding mismatch")

    for backend_module in ("memory-backend", "time-backend"):
        backend_section = re.search(
            rf'<section\b[^>]*data-module="{backend_module}"[^>]*>(?P<body>.*?)</section>',
            html,
            re.DOTALL,
        )
        if re.search(
            r"\b(?:expected|finalize)_projection_candidate\s*\(",
            unescape(backend_section.group("body") if backend_section else ""),
        ):
            errors.append(
                f"Task8 {backend_module} backend runtime-calls plan-owned helper"
            )

    if (
        "shape/storage/lifetime 语义只阻断内存侧" in document_text
        or document_text.count(
            "缺共享 compute shape 以 BLK-MISSING-SHAPE 同时阻断两侧"
        ) != 1
        or document_text.count(
            "缺 memory-only storage/alias/lifetime/explicit-workspace 事实只阻断内存侧"
        ) != 1
    ):
        errors.append("Task8 memory blocker scope mismatch")


def check_handoff_conformance_sync(markdown: str, errors: list[str]) -> None:
    rows = re.findall(
        r"(?m)^\| `conformance` / `mc-conformance` \| (?P<body>[^|]*)\|$",
        markdown,
    )
    required_row_tokens = (
        "BasicOfflineReportArtifact",
        "AttestedConformanceSealArtifact",
        "run_basic_offline_conformance(...)",
        "run_attested_release_conformance(...)",
        "apply_release_policy(...)",
    )
    if len(rows) != 1 or any(rows[0].count(token) != 1 for token in required_row_tokens):
        errors.append("HANDOFF conformance module locator/profile ports 不闭合")

    section_match = re.search(
        r"(?ms)^## 6\. conformance\s*$\n(?P<body>.*?)(?=^## 7\. )",
        markdown,
    )
    if section_match is None:
        errors.append("HANDOFF conformance section 不可结构化提取")
        return
    section = section_match.group("body")
    headings = re.findall(r"(?m)^### (6\.[1-4]) ([^\r\n]+)$", section)
    if headings != [
        ("6.1", "部署 profile 闭包与共享事实"),
        ("6.2", "BasicOfflineConformance"),
        ("6.3", "AttestedReleaseConformance"),
        ("6.4", "判定、批准与隔离"),
    ]:
        errors.append("HANDOFF conformance profile subsections 不闭合")

    union_match = re.search(
        r"(?ms)^ConformanceDeploymentProfile :=\s*\n"
        r"(?P<body>.*?)^default_conformance_deployment_profile := "
        r"BasicOfflineConformance\s*$",
        section,
    )
    exact_union = re.compile(
        r"\s*BasicOfflineConformance\s*\n"
        r"\s*\| AttestedReleaseConformance \{\s*\n"
        r"\s*conformance_trust_root: ConformanceTrustRootCapability,\s*\n"
        r"\s*release_trust_root: ReleaseApprovalTrustRootCapability\s*\n"
        r"\s*\}\s*\n?"
    )
    if union_match is None or exact_union.fullmatch(union_match.group("body")) is None:
        errors.append("HANDOFF conformance deployment union must have exact two arms/default")

    basic_match = re.search(
        r"(?ms)^### 6\.2 BasicOfflineConformance\s*$\n"
        r"(?P<body>.*?)(?=^### 6\.3 )",
        section,
    )
    attested_match = re.search(
        r"(?ms)^### 6\.3 AttestedReleaseConformance\s*$\n"
        r"(?P<body>.*?)(?=^### 6\.4 )",
        section,
    )
    basic = re.sub(r"\s+", " ", basic_match.group("body") if basic_match else "")
    attested = re.sub(r"\s+", " ", attested_match.group("body") if attested_match else "")
    required_basic = (
        "run_basic_offline_conformance( authority: BasicOfflineConformanceAuthority ) -> BasicOfflineRunResult",
        'profile_schema_id="basic-offline/v1"',
        "BasicOfflineReportArtifact",
        "offline_verdict",
        "identity/integrity",
        "不能授权发布",
    )
    if basic_match is None or any(token not in basic for token in required_basic):
        errors.append("HANDOFF BasicOffline profile boundary 不闭合")
    handoff_basic_boundary = (
        "前五种是 content-addressed payload，使用 "
        '`profile_schema_id="basic-offline/v1"` 与彼此独立的 '
        "`basic-offline-*/v1` hash domain；`BasicOfflineRunResult` 只是无 own "
        "digest、无 profile field 的 transport union"
    )
    if basic.count(handoff_basic_boundary) != 1:
        errors.append("HANDOFF BasicOffline payload/result boundary mismatch")
    if re.search(
        r"(?i)trust[_ -]?(?:root|store)|measured[_ -]?environment|"
        r"(?:^|[^A-Za-z0-9])session(?:$|[^A-Za-z0-9])|nonce|attestation|signature|ApprovedDigestSet|"
        r"Release(?:Policy|Decision)|apply_release_policy|seal",
        basic,
    ):
        errors.append("HANDOFF BasicOffline profile contains attested/release capability")

    required_attested = (
        "run_attested_release_conformance( authority: AttestedConformanceInvocationAuthority, profile: AttestedReleaseConformance, measured_environment: MeasuredExecutionEnvironment ) -> AttestedConformanceRunResult",
        "apply_release_policy( artifact: AttestedConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy, profile: AttestedReleaseConformance ) -> ReleasePolicyApplicationResult",
        "ReleasePolicyApplicationResult := Completed { decision: ReleaseDecision } | InternalViolation { violation: InternalContractViolation }",
        'profile_schema_id="attested-release/v1"',
        "fresh measured session",
        "已消费 nonce",
        "完整重跑",
        "internal/non-exported",
    )
    if attested_match is None or any(token not in attested for token in required_attested):
        errors.append("HANDOFF AttestedRelease profile boundary 不闭合")
    if "BasicOfflineReportArtifact" in attested or "ConformanceArtifact" in attested:
        errors.append("HANDOFF Attested release accepts Basic/common artifact")

    handoff_figure_boundary = (
        "Figure 10 先以 OfflineConformanceRequested opt 表达整个离线 sidecar 可选，"
        "其内再以 Basic(default)/Attested 单一 alt 表达 profile 互斥"
    )
    handoff_current_root_tokens = (
        "profile.conformance_trust_root",
        "逐字匹配 artifact authority 的 trust-store/policy refs",
        "current-root nonce registry",
        "measured-execution-environment digest",
        "重验 seal 内 runner attestation",
        "不再次消费 measurement session/nonce",
        "另一 conformance domain 产生的 seal",
    )
    if section.count(handoff_figure_boundary) != 1:
        errors.append("HANDOFF Figure 10 profile ownership boundary mismatch")
    normalized_section = re.sub(r"\s+", " ", section)
    if any(token not in normalized_section for token in handoff_current_root_tokens):
        errors.append("HANDOFF current conformance root release boundary mismatch")

    for isolation_token in (
        "verdict 不改变生产 BackendResult",
        "生产摘要也不反向依赖报告",
    ):
        if isolation_token not in section:
            errors.append(f"HANDOFF conformance isolation missing: {isolation_token}")


def main() -> int:
    with open(SRC, encoding="utf-8") as source:
        html = source.read()
    with open(HANDOFF, encoding="utf-8") as handoff_source:
        handoff = handoff_source.read()

    errors: list[str] = []
    check_required(html, errors)
    check_forbidden(html, errors)
    check_chapters(html, errors)
    check_gates(html, errors)
    check_diagrams(html, errors)
    check_handoff_conformance_sync(handoff, errors)

    gates = parse_gates(html)
    print(f"v4.2 必需契约 {len(REQUIRED_TEXT)} 项")
    print(f"禁用语义 {len(FORBIDDEN_TEXT)} 项")
    print(f"结构门 {len(gates)} / 期望 {len(EXPECTED_GATES)}")

    if errors:
        print("\n✗ target-design-v2 v4.2 契约不一致：")
        for error in errors:
            print("   -", error)
        return 1

    print("\n[OK] target-design-v2 v4.2 产品、PyNative IR、章节与结构门一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
