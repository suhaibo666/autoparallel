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
from html.parser import HTMLParser


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src", "index.template.html")

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
    "Opaque capabilities ConformanceTrustRootCapability, ReleaseApprovalTrustRootCapability and ValidatedMeasurementSessionCapability are non-serializable, own no derived digest and therefore have no exclusion-table row",
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
    "ConformanceReport",
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

INCOMPATIBLE_DIAGRAMS = (
    "{{SVG:01-layering}}",
    "{{SVG:02-provenance}}",
    "{{SVG:04-structure-backward}}",
    "{{SVG:05-identities}}",
    "{{SVG:06-placement-execution}}",
)

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
        "TimeRequested { calibration_set: CalibrationSet communication_model_snapshot: CommunicationModelSnapshot time_cost_policy: TimeCostPolicy }",
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
        "ConformanceReport",
        "ReleaseDecision",
        "run_conformance( authority: ConformanceInvocationAuthority, trust_root: ConformanceTrustRootCapability, measured_environment: MeasuredExecutionEnvironment ) -> ConformanceRunResult",
        "apply_release_policy( artifact: ConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy, release_trust_root: ReleaseApprovalTrustRootCapability ) -> ReleaseDecision",
        "derive_approved_digest_set( artifact: ConformanceSealArtifact, policy: ReleasePolicy ) -> ApprovedDigestSet",
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
        "require approved == derive_approved_digest_set(artifact, policy)",
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
        "ConformanceExecutionRecord:",
        "ConformanceExecutionLedger:",
        "conformance_execution_record_digest :=",
        "ConformanceRunResult := Completed { artifact: ConformanceSealArtifact } | InternalViolation { violation: InternalContractViolation }",
        "conformance_seal_artifact_digest := hash(canonical_payload_without_derived_digests(ConformanceSealArtifact))",
        "runner crash, schema failure, digest failure or conservation failure",
        "complete, valid execution",
        "record.fixture_ref == record_key",
        "record.target_invocation_id == binding.target_invocation_id",
        "record.target_clause_id == binding.target_clause_id",
        "ConformanceInvocationAuthority:",
        "conformance_invocation_digest :=",
        "fixture_digest evaluation_input_digest := hash(authority.conformance_invocation_digest, session.measured_execution_environment_digest, canonical(binding), binding.fixture_binding_digest) observed_clause_output_digest",
        "ConformanceObservedOutput:",
        "ConformanceObservedOutput: execution_records: OrderedMap<FixtureRef, ConformanceExecutionRecord> findings: OrderedMap<FindingId, ConformanceFinding> coverage_gaps: OrderedSet<GateClauseId> observed_output_digest := hash(canonical execution_records, findings, coverage_gaps)",
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
        "validate_and_consume_measurement_envelope( authority: ConformanceInvocationAuthority, runner: VerifierRunner, trust_root: ConformanceTrustRootCapability, measured_environment: MeasuredExecutionEnvironment ) -> ValidatedMeasurementSessionCapability | InternalContractViolation",
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
        "measured_environment.verifier_nonce is active, bound to this exact invocation/runner/executable/session/time-window tuple, and atomically consumed exactly once",
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
        "validate_and_seal_conformance( authority: ConformanceInvocationAuthority, trust_root: ConformanceTrustRootCapability, session: ValidatedMeasurementSessionCapability, observed: ConformanceObservedOutput, ledger: ConformanceExecutionLedger, attestation: RunnerAttestation ) -> ConformanceRunResult",
        "validate_and_seal_conformance first operation:",
        "recompute authority.production_subject.subject_digest",
        "recompute every authority.fixture_set binding, nested fixture and fixture_set_digest",
        "recompute authority.validation_policy.validation_policy_digest",
        "recompute authority.gate_manifest and every nested entry/clause digest",
        "recompute authority.gate_manifest.blocker_scope_policy.blocker_scope_policy_digest",
        "recompute authority.verifier_runner.verifier_runner_digest",
        "recompute authority.conformance_invocation_digest only after all nested recomputations",
        "negative fixture: old invocation digest plus substituted policy or trust-store reference",
        "derive_conformance_report( authority: ConformanceInvocationAuthority, observed: ConformanceObservedOutput, ledger: ConformanceExecutionLedger, attestation: RunnerAttestation ) -> ConformanceReport",
        "ConformanceReport.findings: OrderedMap<FindingId, ConformanceFinding>",
        "report.findings == observed.findings",
        "failure_findings := policy_classified_failure_findings(authority.validation_policy, observed.findings)",
        "verdict := fail iff failure_findings is non-empty",
        "else insufficient iff exact_evidence_or_coverage_gaps(authority, observed, ledger) is non-empty",
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
        "apply_release_policy first operation:",
        "recompute artifact and every nested invocation/report/ledger/attestation/environment digest",
        "recompute approval_authority.store_snapshot and every nested ApprovedDigestSet value",
        "recompute policy.release_policy_digest",
        "recompute approval_authority.release_approval_authority_digest",
        "recompute approval.release_approval_artifact_digest",
        "before resolving release_trust_root key material, verifying signature or reading verdict",
        "root := resolve_release_approval_trust_root(release_trust_root)",
        "root.expected_release_policy_digest == policy.release_policy_digest",
        "canonical(root.expected_release_policy) == canonical(policy)",
        "root.expected_release_approval_store_snapshot_digest == approval_authority.store_snapshot.release_approval_store_snapshot_digest",
        "canonical(root.expected_release_approval_store_snapshot) == canonical(approval_authority.store_snapshot)",
        "require approval_authority.approval_signature_scheme == policy.approval_signature_scheme",
        "policy.approval_signature_scheme in root.supported_release_approval_signature_schemes",
        "trusted_approver_public_key_material := root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]",
        "release_approval_signed_message := canonical_tuple(",
        "canonical(derived_approved)",
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
        "expected_execution_environment_digest": 4,
        "trusted_attestation_key_id, attestation_signature_scheme": 2,
        "trust_store_snapshot_digest": 15,
        "cannot be constructed by request deserialization": 5,
        "protected_environment_measurer": 1,
        "(store, policy) := resolve_conformance_trust_root(trust_root)": 2,
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
        "verify_measurer_identity_and_freshness_evidence(": 2,
        "measured_payload_digest := hash(canonical measured_environment.measured_environment_payload)": 2,
        "measured_payload_digest == policy.expected_execution_environment_digest": 2,
        "measurer_identity_and_freshness_evidence": 6,
        "session.measured_execution_environment_digest": 3,
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
        "validate_and_seal_conformance( authority: ConformanceInvocationAuthority, observed: ConformanceObservedOutput, ledger: ConformanceExecutionLedger, report: ConformanceReport",
        "apply_release_policy( report: ConformanceReport",
        "run_conformance( subject: ProductionSubject",
        "apply_release_policy( artifact: ConformanceSealArtifact, approved: ApprovedDigestSet",
        "apply_release_policy( artifact: ConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy ) -> ReleaseDecision",
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
    ("TrustStoreSnapshot", "trust_store_snapshot_digest"),
    ("RunnerAttestationPolicy", "runner_attestation_policy_digest"),
    ("MeasuredExecutionEnvironment", "measured_execution_environment_digest"),
    ("ConformanceInvocationAuthority", "conformance_invocation_digest"),
    ("RunnerAttestation", "runner_attestation_digest"),
    ("ConformanceExecutionRecord", "conformance_execution_record_digest"),
    ("ConformanceObservedOutput", "observed_output_digest"),
    ("ConformanceExecutionLedger", "conformance_execution_ledger_digest"),
    ("ReleaseApprovalStoreSnapshot", "release_approval_store_snapshot_digest"),
    ("ReleaseApprovalAuthority", "release_approval_authority_digest"),
    ("ReleaseApprovalArtifact", "release_approval_artifact_digest"),
    ("ConformanceSealArtifact", "conformance_seal_artifact_digest"),
    ("ConformanceReport", "conformance_report_digest"),
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


def check_conformance_trust_boundary(html: str, errors: list[str]) -> None:
    parser = _ModuleContractParser()
    parser.feed(html)
    matches = []
    for contract in parser.contracts:
        attrs = contract["attrs"]
        assert isinstance(attrs, dict)
        if attrs.get("data-module") == "conformance":
            matches.append(contract)
    if len(matches) != 1:
        return
    text_parts = matches[0]["text_parts"]
    assert isinstance(text_parts, list)
    local_text = _normalize_contract_text(text_parts)

    authority_match = re.search(
        r"ConformanceInvocationAuthority:\s*(.*?)\s*RunnerAttestation:",
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
                    "ConformanceInvocationAuthority must hold only external-root refs: "
                    f"{forbidden_field}"
                )

    first_marker = "validate_and_seal_conformance first operation:"
    after_marker = "validate_and_seal_conformance equations after first operation:"
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
        errors.append("validate_and_seal_conformance first-operation boundary is missing")
    else:
        for token in first_operation_tokens:
            token_index = local_text.find(token, first_index, after_index)
            if token_index < 0:
                errors.append(
                    "validate_and_seal_conformance first operation missing nested "
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
    exact_schema = re.compile(
        r"ConformanceReport:\s*.*?"
        r"verdict:\s*pass\s*\|\s*fail\s*\|\s*insufficient\s*"
        r"findings:\s*OrderedMap&lt;FindingId,\s*ConformanceFinding&gt;",
        re.DOTALL,
    )
    if exact_schema.search(html) is None:
        errors.append(
            "ConformanceReport.findings 必须与 observed output 同型："
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

    release_policy = re.search(
        r"ReleasePolicy:\s*.*?release_policy_digest\s*:=",
        html,
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
    check_conformance_trust_boundary(html, errors)
    check_conformance_report_findings(html, errors)
    check_release_approved_digest_contract(html, errors)
    check_module_contracts(html, errors)


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


def main() -> int:
    with open(SRC, encoding="utf-8") as source:
        html = source.read()

    errors: list[str] = []
    check_required(html, errors)
    check_forbidden(html, errors)
    check_chapters(html, errors)
    check_gates(html, errors)
    check_diagrams(html, errors)

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
