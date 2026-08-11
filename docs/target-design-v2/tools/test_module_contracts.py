from __future__ import annotations

import pathlib
import re
import unittest
from dataclasses import dataclass, field
from html import unescape
from html.parser import HTMLParser


ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "index.template.html"
TARGET_MODULES = (
    "input-facts",
    "code-ir",
    "runtime-events",
    "plan-projection",
    "memory-backend",
    "time-backend",
    "result-sealing",
    "gate-system",
    "comparison",
    "conformance",
)
REQUIRED_SUBSECTIONS = {
    "职责边界",
    "核心数据结构",
    "接口定义",
    "成功与阻断语义",
    "不变量",
}
PLAN_AUTHORITATIVE_CONTRACT_REFS = (
    "SimulationPlanCore",
    "CoreBuildResult<SimulationPlanCore>",
    "ProjectionCandidate<MemoryEventView>",
    "ProjectionCandidate<TimeEventView>",
    "ProjectionBundleAuthority",
    "ProjectionBundleBuild",
    "ProjectionResult<MemoryEventView>",
    "ProjectionResult<TimeEventView>",
)


@dataclass
class Contract:
    attrs: dict[str, str]
    headings: list[tuple[str, str, str]] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.text_parts)).strip()


class ContractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.contracts: list[Contract] = []
        self.current: Contract | None = None
        self.section_depth = 0
        self.heading: tuple[str, str, list[str]] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attr_map = {name: value or "" for name, value in attrs}
        if tag == "section":
            classes = set(attr_map.get("class", "").split())
            if self.current is None and "module-contract" in classes:
                self.current = Contract(attr_map)
                self.section_depth = 1
                return
            if self.current is not None:
                self.section_depth += 1
        if self.current is not None and re.fullmatch(r"h[3-6]", tag):
            self.heading = (tag, attr_map.get("id", ""), [])

    def handle_data(self, data: str) -> None:
        if self.current is None:
            return
        self.current.text_parts.append(data)
        if self.heading is not None:
            self.heading[2].append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.current is None:
            return
        if self.heading is not None and tag == self.heading[0]:
            level, heading_id, parts = self.heading
            text = re.sub(r"\s+", " ", " ".join(parts)).strip()
            self.current.headings.append((level, heading_id, text))
            self.heading = None
        if tag == "section":
            self.section_depth -= 1
            if self.section_depth == 0:
                self.contracts.append(self.current)
                self.current = None


def parse_contracts() -> dict[str, list[Contract]]:
    parser = ContractParser()
    parser.feed(TEMPLATE.read_text(encoding="utf-8"))
    by_module: dict[str, list[Contract]] = {}
    for contract in parser.contracts:
        by_module.setdefault(contract.attrs.get("data-module", ""), []).append(contract)
    return by_module


def diagram_source(diagram_id: str) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    match = re.search(
        rf'<figure\b[^>]*data-diagram-id="{re.escape(diagram_id)}"[^>]*>'
        rf'.*?<pre class="mermaid-source"><code>(.*?)</code></pre>',
        template,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing Mermaid diagram {diagram_id!r}")
    return unescape(match.group(1))


class ModuleContractStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts = parse_contracts()

    def contract_text(self, module: str) -> str:
        matches = self.contracts.get(module, [])
        self.assertEqual(len(matches), 1, f"{module}: {len(matches)} sections")
        return matches[0].text

    def test_task3_contract_sections_are_complete_and_accessible(self) -> None:
        for module in TARGET_MODULES:
            with self.subTest(module=module):
                matches = self.contracts.get(module, [])
                self.assertEqual(len(matches), 1, f"{module}: {len(matches)} sections")
                contract = matches[0]
                labelled_by = contract.attrs.get("aria-labelledby", "")
                self.assertTrue(labelled_by, f"{module} lacks aria-labelledby")
                heading_ids = {heading_id for _, heading_id, _ in contract.headings}
                self.assertIn(labelled_by, heading_ids)
                subsection_titles = {
                    text for level, _, text in contract.headings if level in {"h4", "h5"}
                }
                self.assertTrue(
                    REQUIRED_SUBSECTIONS <= subsection_titles,
                    f"{module} missing {sorted(REQUIRED_SUBSECTIONS - subsection_titles)}",
                )

    def test_input_facts_contract_has_closed_entry_and_result_union(self) -> None:
        text = self.contracts["input-facts"][0].text
        for token in (
            "SourceSnapshot",
            "StructureRegistrySnapshot",
            "RuntimeRegistrySnapshot",
            "CommonProductionInputs",
            "RequestSnapshotBuildResult",
            "CanonicalConfigEvaluationInput",
            "RequestSnapshot",
            "MemoryRegistrySnapshot",
            "TimeCostPolicy",
            "build_request_snapshot(",
            "requested_backends",
            "InternalContractViolation",
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"RequestSnapshotBuildResult\s*:=\s*Ready\s*\{\s*request_snapshot:\s*RequestSnapshot\s*\}\s*\|\s*Blocked",
        )
        self.assertIn("Blocked 仅", text)
        self.assertIn("唯一生产", text)

    def test_plan_projection_contract_has_existing_typed_ports_and_closed_outcomes(self) -> None:
        text = self.contract_text("plan-projection")
        for token in (
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
            "missing exact compute profile blocks time only",
            "missing memory-only storage/alias/lifetime/explicit-workspace facts blocks memory only",
            "missing shared compute shape yields BLK-MISSING-SHAPE and affects both faces",
            "no capacity, completion-time or contention reads",
            "one canonical InternalContractViolation and no partial bundle",
        ):
            self.assertIn(token, text)
        template = TEMPLATE.read_text(encoding="utf-8")
        contract = re.search(
            r'<section\b[^>]*data-module="plan-projection"[^>]*>(.*?)</section>',
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(contract)
        reference_list = re.search(
            r'<ul\b[^>]*data-authoritative-contract-references="plan-projection"[^>]*>'
            r'(.*?)</ul>',
            contract.group(1),
            re.DOTALL,
        )
        self.assertIsNotNone(reference_list)
        refs = tuple(
            unescape(ref)
            for ref in re.findall(
                r'<li\b[^>]*data-contract-ref="([^"]+)"[^>]*>',
                reference_list.group(1),
            )
        )
        self.assertEqual(refs, PLAN_AUTHORITATIVE_CONTRACT_REFS)
        self.assertNotIn("PlanProjectionOwnedContracts", contract.group(1))
        self.assertNotRegex(contract.group(1), r"(?m)^\s*[A-Za-z][A-Za-z0-9_<>]*\s*:=")
        self.assertNotIn("CoreBuildResult<SimulationPlanCore> :=", text)
        self.assertNotIn("ProjectionResult<V> :=", text)
        self.assertNotIn("ProjectionCandidate<MemoryEventView> :=", text)

    def test_chapter1_module_table_is_exact_and_dependency_closed(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")
        rows = re.findall(
            r'<tr\b[^>]*data-module-id="([a-z-]+)"[^>]*data-layer="([A-Z0-9-]+)"'
            r'[^>]*data-allowed-dependencies="([a-z,-]*)"[^>]*>',
            template,
        )
        expected = {
            "input-facts": ("L0", ""),
            "code-ir": ("L1", ""),
            "runtime-events": ("L1", ""),
            "plan-projection": ("L2", "gate-system,memory-backend,time-backend"),
            "gate-system": ("L2", ""),
            "memory-backend": ("L3", ""),
            "time-backend": ("L3", ""),
            "result-sealing": ("L4", "gate-system,memory-backend,plan-projection,time-backend"),
            "comparison": ("L4", "gate-system,result-sealing"),
            "conformance": ("OFFLINE", "comparison,gate-system,result-sealing"),
        }
        self.assertEqual({module: (layer, deps) for module, layer, deps in rows}, expected)

    def test_architecture_edges_encode_reverse_dependencies_and_dto_lineage(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")
        table_rows = re.findall(
            r'<tr\b[^>]*data-module-id="([a-z-]+)"[^>]*'
            r'data-allowed-dependencies="([a-z,-]*)"[^>]*>',
            template,
        )
        source = diagram_source("layered-module-architecture")
        nodes = dict(
            re.findall(r'(?m)^\s+([A-Za-z][A-Za-z0-9]*)\["([^"]+)"\]\s*$', source)
        )
        module_nodes = {
            label: node for node, label in nodes.items() if label in dict(table_rows)
        }
        solid = set(re.findall(r"(?m)^\s+(\w+)\s+(-->)\s+(\w+)\s*$", source))
        dotted = set(re.findall(r"(?m)^\s+(\w+)\s+(-\.->)\s+(\w+)\s*$", source))
        expected_solid = {
            (module_nodes[dependency], "-->", module_nodes[caller])
            for caller, dependencies in table_rows
            for dependency in dependencies.split(",")
            if dependency
        }
        self.assertEqual(solid, expected_solid)
        self.assertEqual(
            dotted,
            {
                ("inputFacts", "-.->", "codeIr"),
                ("inputFacts", "-.->", "runtimeEvents"),
                ("codeIr", "-.->", "runtimeEvents"),
                ("runtimeEvents", "-.->", "planProjection"),
                ("inputFacts", "-.->", "gateSystem"),
            },
        )

        plan_source = diagram_source("plan-projection-module-architecture")
        plan_solid = set(
            re.findall(r"(?m)^\s+(\w+)\s+(-->)\s+(\w+)\s*$", plan_source)
        )
        plan_dotted = set(
            re.findall(r"(?m)^\s+(\w+)\s+(-\.->)\s+(\w+)\s*$", plan_source)
        )
        self.assertEqual(
            plan_solid,
            {
                ("coreBinder", "-->", "faceCoordinator"),
                ("faceCoordinator", "-->", "bundleFinalizer"),
                ("memoryBackend", "-->", "faceCoordinator"),
                ("timeBackend", "-->", "faceCoordinator"),
                ("gateSystem", "-->", "faceCoordinator"),
                ("gateSystem", "-->", "bundleFinalizer"),
            },
        )
        self.assertEqual(
            plan_dotted,
            {
                ("inputFacts", "-.->", "coreBinder"),
                ("runtimeEvents", "-.->", "coreBinder"),
            },
        )

    def test_candidate_builders_use_pure_expected_postcondition(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertNotIn("finalize_projection_candidate", template)
        self.assertIn("expected_projection_candidate(", template)
        self.assertEqual(
            template.count(
                "require returned_candidate ==\n    expected_projection_candidate("
            ),
            2,
        )
        self.assertNotRegex(template, r"call\s+expected_projection_candidate")

    def test_projection_bundle_constructor_is_total_over_core_result(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")
        evaluator = re.search(
            r"evaluate_and_finalize_projection_bundle\(authority\):"
            r"(?P<body>.*?)</code></pre>",
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(evaluator)
        body = evaluator.group("body") if evaluator else ""
        branches = re.search(
            r"match authority\.core_result \(exhaustive, mutually exclusive\):\s*"
            r"Ready \{ core=core \}:\s*(?P<ready>.*?)\s*"
            r"Blocked \{ blockers=blockers \}:\s*(?P<blocked>.*?)\s*"
            r"require exactly one candidate-construction branch executed",
            body,
            re.DOTALL,
        )
        self.assertIsNotNone(branches)
        ready = re.sub(r"\s+", " ", branches.group("ready") if branches else "")
        blocked = re.sub(r"\s+", " ", branches.group("blocked") if branches else "")
        for backend in ("memory", "time"):
            self.assertIn(
                f"authority.{backend}_candidate is byte-equal to "
                f"build_{backend}_projection_candidate( authority.request_snapshot, "
                "authority.evaluation_identity, core)",
                ready,
            )
            self.assertIn(
                f"authority.{backend}_candidate is byte-equal to "
                "expected_projection_candidate( authority.request_snapshot, "
                f"authority.evaluation_identity, {backend}, None if {backend} not in "
                "authority.request_snapshot.requested_backends else CandidateBlocked "
                "{ construction_blockers=blockers })",
                blocked,
            )
        self.assertNotIn("build_memory_projection_candidate(", blocked)
        self.assertNotIn("build_time_projection_candidate(", blocked)
        self.assertIn("do not call either backend candidate builder", blocked)
        self.assertNotIn(
            "both candidates are byte-equal to the exact outputs of the declared candidate constructors",
            re.sub(r"\s+", " ", body),
        )

    def test_memory_blocker_scope_matches_shared_shape_authority(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertNotIn("shape/storage/lifetime 语义只阻断内存侧", template)
        self.assertIn(
            "缺共享 compute shape 以 BLK-MISSING-SHAPE 同时阻断两侧", template
        )
        self.assertIn(
            "缺 memory-only storage/alias/lifetime/explicit-workspace 事实只阻断内存侧",
            template,
        )

    def test_backend_specific_request_inputs_are_tagged_and_independent(self) -> None:
        text = self.contract_text("input-facts")
        common_match = re.search(
            r"CommonProductionInputs:(.*?)RequestSnapshotBuildResult\s*:=",
            text,
        )
        self.assertIsNotNone(common_match)
        common = common_match.group(1) if common_match else ""
        for backend_field in (
            "memory_registry_snapshot",
            "calibration_set",
            "calibration_train_manifest_snapshot",
            "communication_model_snapshot",
            "time_cost_policy",
        ):
            self.assertNotIn(backend_field, common)

        for token in (
            "RequestedBackendInput :=",
            "MemoryRequested { memory_registry_snapshot: MemoryRegistrySnapshot }",
            "TimeRequested { calibration_set: CalibrationSet calibration_train_manifest_snapshot: CalibrationTrainManifest communication_model_snapshot: CommunicationModelSnapshot time_cost_policy: TimeCostPolicy }",
            "| NotRequested",
            "requested_backend_inputs: OrderedMap<memory | time, RequestedBackendInput>",
            "domain(requested_backend_inputs) == {memory,time}",
            "requested_backends == keys tagged MemoryRequested or TimeRequested",
            "NotRequested carries no payload and no digest",
            "memory-only request neither requires nor fingerprints CalibrationSet, CommunicationModelSnapshot or TimeCostPolicy",
            "time-only request neither requires nor fingerprints MemoryRegistrySnapshot",
        ):
            self.assertIn(token, text)

    def test_request_subject_and_candidate_builders_bind_only_requested_faces(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")
        for token in (
            "backend_semantics_digests := requested_backend_semantics_digests(request.requested_backend_inputs)",
            "requested_backend_semantics_digests excludes every NotRequested arm",
            "memory_inputs := require MemoryRequested at request.requested_backend_inputs[memory]",
            "time_inputs := require TimeRequested at request.requested_backend_inputs[time]",
        ):
            self.assertIn(token, template)
        self.assertRegex(
            template,
            r"build_memory_projection_candidate\(\s*request:\s*RequestSnapshot,\s*evaluation_identity:\s*EvaluationInstanceIdentity,\s*core:\s*SimulationPlanCore\s*\)\s*-&gt;\s*ProjectionCandidate&lt;MemoryEventView&gt;",
        )
        self.assertRegex(
            template,
            r"build_time_projection_candidate\(\s*request:\s*RequestSnapshot,\s*evaluation_identity:\s*EvaluationInstanceIdentity,\s*core:\s*SimulationPlanCore\s*\)\s*-&gt;\s*ProjectionCandidate&lt;TimeEventView&gt;",
        )

    def test_time_occupied_streams_accessor_is_total_and_normative(self) -> None:
        text = self.contract_text("time-backend")
        for token in (
            "occupied_streams(node: BoundComputeEvent | BoundCommunication)",
            "BoundComputeEvent -> OrderedSet{node.physical_stream_id}",
            "BoundCommunication -> node.common.occupied_streams",
            "occupied_streams(expected_bound_compute(e, core, cost_bindings, stream_bindings)) == OrderedSet{stream_bindings[e.event_id].physical_stream_id}",
            "occupied_streams(node)) for node in order",
        ):
            self.assertIn(token, text)
        execution = re.search(
            r"expected_time_execution\((.*?)require run_time_backend\(view\) == expected_time_execution\(view\)",
            text,
        )
        self.assertIsNotNone(execution)
        self.assertNotIn("node.occupied_streams", execution.group(1) if execution else "")

        template = TEMPLATE.read_text(encoding="utf-8")
        for gate_token in (
            "G-TIME1",
            "occupied_streams(compute)==OrderedSet{physical_stream_id}",
            "G-TIME2",
            "occupied_streams(node) accessor",
            "occupied_streams(node) for every progress node",
        ):
            self.assertIn(gate_token, template)

    def test_mermaid_semantic_closure_binds_runtime_facts_only(self) -> None:
        source = diagram_source("semantic-effect-closure")
        self.assertIn(
            "SemanticValidated --> RuntimeBound : runtime facts resolve uniquely",
            source,
        )
        self.assertIn("RuntimeBound --> Planned : base DAG then required effect edges", source)
        self.assertIn("Stage-local Blocked or InternalContractViolation outcome", source)
        self.assertNotIn("backend facts", source)
        self.assertNotIn("InputBlocker / no synthesized zero or pure effect", source)

    def test_mermaid_per_rank_flow_separates_residual_and_rank_build_failure(self) -> None:
        source = diagram_source("per-rank-codeir")
        for token in (
            'readyResiduals["Ready residual_blocker_records"]',
            'blockerIndex["CodeIR.blocker_index when all rank builds are Ready"]',
            'rankBlocked["RankCodeIRBuildResult.Blocked / rank-build blockers"]',
            'topBlocked["CodeIRBuildResult.Blocked / canonical blocker union"]',
            "readyResiduals --> blockerIndex",
            "rankBlocked --> topBlocked",
        ):
            self.assertIn(token, source)
        self.assertNotIn(
            'disposition --> blockers["BlockerRecord in CodeIR.blocker_index"]',
            source,
        )

    def test_source_file_snapshot_owns_canonical_bytes(self) -> None:
        text = self.contracts["input-facts"][0].text
        for token in (
            "canonical_utf8_source_bytes: ByteString",
            "content_digest := hash(canonical_utf8_source_bytes)",
            "canonical_path: DiagnosticPath",
            "canonical_path 仅用于诊断",
            "不得隐式读取文件系统",
        ):
            self.assertIn(token, text)

    def test_code_ir_contract_closes_rank_build_and_keeps_runtime_identity_out(self) -> None:
        text = self.contracts["code-ir"][0].text
        for token in (
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
            "Trainer",
            "layer_and_expert_ownership",
            "HardwareProfile",
            "TimeCostPolicy",
            "RuntimeEventPlan",
            "InternalContractViolation",
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"evaluate_source\(\s*input:\s*RankCodeIRBuildInput\s*\)\s*->\s*RankCodeIRBuildResult",
        )
        self.assertIn("microbatch occurrence identity", text)
        self.assertIn("Blocked 仅", text)

    def test_rank_build_input_digest_closes_every_result_branch(self) -> None:
        text = self.contracts["code-ir"][0].text
        for token in (
            "source: SourceSnapshot",
            "model: ModelSpec",
            "config: NormalizedParallelConfig",
            "rank: LogicalRankContext",
            "bindings: CodeInputShapeDtypeBindings",
            "env: CompileEnvFacts",
            "registry: StructureRegistrySnapshot",
            "rank_build_input_digest :=",
            "hash(canonical_payload_without_derived_digests( RankCodeIRBuildInput))",
            "result.rank_build_input_digest == rank_input.rank_build_input_digest",
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"RankCodeIRBuildResult\s*:=\s*Ready\s*\{\s*rank_build_input_digest:\s*Digest.*?\}\s*\|\s*Blocked\s*\{\s*rank_build_input_digest:\s*Digest",
        )
        self.assertRegex(
            text,
            r"assemble_code_ir\(\s*rank_inputs:\s*OrderedMap<LogicalRank,\s*RankCodeIRBuildInput>,\s*rank_results:\s*OrderedMap<LogicalRank,\s*RankCodeIRBuildResult>\s*\)",
        )
        self.assertNotIn("model_input_digest: Digest", text)

    def test_storage_root_is_separate_from_tensor_storage_relation(self) -> None:
        text = self.contracts["code-ir"][0].text
        match = re.search(r"LogicalStorage:(.*?)TensorStorageRelation:", text)
        self.assertIsNotNone(match)
        logical_storage = match.group(1) if match is not None else ""
        self.assertNotIn("relation_to_root", logical_storage)
        self.assertNotIn("offset", logical_storage)
        self.assertNotIn("range", logical_storage)
        for token in (
            "relation: new | alias | view | inplace",
            "base_storage_id",
            "offset, range",
            "CodeNode.storage_relations: [TensorStorageRelation]",
        ):
            self.assertIn(token, text)

    def test_assembly_unions_ready_residuals_with_blocked_rank_failures(self) -> None:
        text = self.contracts["code-ir"][0].text
        for token in (
            "rank_build_blocker_union(rank_inputs, rank_results) :=",
            "canonical_union",
            "Ready residual_blocker_records",
            "Blocked blockers",
            "model_input_digest 按 2.4 节现行公式从 rank_inputs 重算",
            "r0 Ready residual {A} + r1 Blocked {B}",
            "Blocked {A, B}",
        ):
            self.assertIn(token, text)

    def test_batching_can_change_code_ir_without_copying_occurrence_identity(self) -> None:
        text = self.contracts["code-ir"][0].text
        self.assertNotIn("microbatch 数不能改变它", text)
        for token in (
            "microbatch occurrence identity 不复制进 CodeIR",
            "完整 NormalizedParallelConfig 仍是 evaluate_source 输入",
            "shape/guard 相关 batching 字段可以改变 CodeIR",
        ):
            self.assertIn(token, text)

    def test_runtime_contract_closes_training_and_communication_expansion(self) -> None:
        text = self.contracts["runtime-events"][0].text
        for token in (
            "RuntimeRuleSnapshot:",
            "expand_runtime_semantics(",
            "RuntimeBuildResult",
            "RuntimeRuleInvocationRef",
            "ExecEvent",
            "ResolvedEventSemantic",
            "TensorInstance",
            "StorageInstance",
            "TensorStorageBinding",
            "LogicalInitialState",
            "AutogradLink",
            "P2PIntent",
            "CollectiveIntent",
            "send",
            "recv",
            "PP",
            "CP",
            "affected_backends",
            "InternalContractViolation",
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"expand_runtime_semantics\(\s*code_ir:\s*CodeIR,\s*config:\s*NormalizedParallelConfig,\s*scenario:\s*ExecutionScenario,\s*registry:\s*RuntimeRegistrySnapshot\s*\)\s*->\s*RuntimeBuildResult",
        )
        self.assertIn("正反向双向闭合", text)
        self.assertIn("microbatch", text)
        self.assertIn("Blocked 仅", text)

    def test_memory_backend_contract_uses_current_projection_and_replay_truth(self) -> None:
        text = self.contract_text("memory-backend")
        for token in (
            "MemoryTimelineEntry:",
            "MemoryExecutionWitness:",
            "MemoryEventView",
            "MemoryEstimate",
            "WorkspaceBinding",
            "build_memory_projection_candidate(",
            "run_memory_backend(",
            "logical_kernel_order",
            "INITIAL_STATE < all canonical MemoryEventId",
            "per_rank.keys",
            "cluster_max_peak_allocated_bytes",
            "breakdown",
            "不读取 duration",
            "不读取 physical stream",
            "不建模跨 stream allocator reuse 或竞争",
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"build_memory_projection_candidate\(\s*request:\s*RequestSnapshot,\s*evaluation_identity:\s*EvaluationInstanceIdentity,\s*core:\s*SimulationPlanCore\s*\)\s*->\s*ProjectionCandidate<MemoryEventView>\s*\|\s*InternalContractViolation",
        )
        self.assertRegex(
            text,
            r"run_memory_backend\(\s*view:\s*MemoryEventView\s*\)\s*->\s*BackendExecution<MemoryEstimate,\s*MemoryExecutionWitness>",
        )
        self.assertNotIn("ProjectionGateEvaluationAuthority", text)
        self.assertNotIn("WorkspaceRegistrySnapshot", text)

    def test_memory_execution_is_the_unique_field_complete_replay(self) -> None:
        text = self.contract_text("memory-backend")
        for token in (
            'MEMORY_REPLAY_SEMANTICS_VERSION := "logical-kernel-order-replay/v1"',
            "expected_memory_execution(",
            "memory_execution_input_digest",
            "timeline[r] := replay(view.per_rank[r])",
            "value.per_rank[r].timeline == timeline[r]",
            "witness.per_rank_replay_order[r] == map(entry.position, timeline[r])",
            "witness.per_rank_timeline_digest[r] == hash(canonical timeline[r])",
            "witness.per_rank_terminal_live_storage_ids[r] == last(timeline[r]).live_storage_instance_ids",
            "witness.memory_execution_input_digest == hash(canonical(view))",
            "witness.memory_projection_semantics_digest == view.memory_projection_semantics_digest",
            "hash(MEMORY_REPLAY_SEMANTICS_VERSION, witness.memory_projection_semantics_digest)",
            "require run_memory_backend(view) == expected_memory_execution(view)",
        ):
            self.assertIn(token, text)

    def test_memory_projection_semantics_are_a_frozen_snapshot(self) -> None:
        text = self.contract_text("memory-backend")
        for token in (
            "MemoryProjectionSemanticsSnapshot:",
            "semantic_version",
            "anchor_order_rule_id",
            "memory_event_id_rule_id",
            "lifetime_rule_id",
            "workspace_lifetime_rule_id",
            "rank_projection_rule_id",
            "memory_projection_semantics_snapshot_digest := hash(canonical_payload_without_derived_digests( MemoryProjectionSemanticsSnapshot))",
            "unique_memory_projection_semantics_snapshot(MEMORY_BACKEND_SEMANTIC_VERSION)",
            "expected_memory_projection(",
            "memory_projection_semantics_snapshot: MemoryProjectionSemanticsSnapshot",
            "view.memory_projection_semantics_digest == memory_projection_semantics_snapshot.memory_projection_semantics_snapshot_digest",
        ):
            self.assertIn(token, text)
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertRegex(
            template,
            r"(?s)memory_simulation_digest\s*:=\s*hash\(.*?MemoryProjectionSemanticsSnapshot.*?memory backend and numeric semantic versions\)",
        )

    def test_time_backend_contract_uses_exact_costs_and_no_contention_des(self) -> None:
        text = self.contract_text("time-backend")
        for token in (
            "RouteBinding:",
            "BoundP2P extends BoundCommunicationCommon:",
            "BoundCollective extends BoundCommunicationCommon:",
            "TimeTimelineEntry:",
            "TimeExecutionWitness:",
            "StepTimeEstimate",
            "CostBinding communication arm",
            "build_time_projection_candidate(",
            "run_time_backend(",
            "unique exact user measurement record",
            "禁止插值、roofline",
            "frozen theoretical formula",
            "PP/CP",
            "endpoint quotient",
            "step-relative zero",
            "no contention",
            "empty graph",
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"build_time_projection_candidate\(\s*request:\s*RequestSnapshot,\s*evaluation_identity:\s*EvaluationInstanceIdentity,\s*core:\s*SimulationPlanCore\s*\)\s*->\s*ProjectionCandidate<TimeEventView>\s*\|\s*InternalContractViolation",
        )
        self.assertRegex(
            text,
            r"run_time_backend\(\s*view:\s*TimeEventView\s*\)\s*->\s*BackendExecution<StepTimeEstimate,\s*TimeExecutionWitness>",
        )
        self.assertNotIn("ProjectionGateEvaluationAuthority", text)
        self.assertNotIn("WorkspaceRegistrySnapshot", text)
        self.assertNotIn("StreamAssignmentPolicySnapshot", text)

    def test_time_execution_and_communication_bindings_are_field_complete(self) -> None:
        text = self.contract_text("time-backend")
        for token in (
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
            "expected_time_execution(",
            "witness.stable_topological_order == order",
            "witness.predecessor_end_max_by_node == predecessor_end_max_by_node",
            "witness.critical_predecessor_by_node == critical_predecessor_by_node",
            "witness.step_end_node == step_end_node",
            "witness.timeline_digest == hash(canonical timeline)",
            "witness.aggregate_interval_inputs_digest == hash(canonical aggregate_interval_inputs)",
            "value.event_timeline == timeline",
            "value.critical_path == critical_path",
            "derive_time_aggregates_by_exact_10_4_equations(aggregate_interval_inputs)",
            "require run_time_backend(view) == expected_time_execution(view)",
        ):
            self.assertIn(token, text)
        collective = re.search(r"BoundCollective extends BoundCommunicationCommon:(.*?)TimeTimelineEntry:", text)
        self.assertIsNotNone(collective)
        self.assertNotIn("selected_algorithm", collective.group(1) if collective else "")

    def test_result_sealing_contract_closes_execution_and_all_three_branches(self) -> None:
        text = self.contract_text("result-sealing")
        for token in (
            "BackendReadyView := MemoryEventView | TimeEventView",
            "EstimateOf<MemoryEventView> := MemoryEstimate",
            "EstimateOf<TimeEventView> := StepTimeEstimate",
            "BackendExecution<T, W> :=",
            "Completed { value: T, witness: W }",
            "| InternalViolation { violation: InternalContractViolation }",
            "BackendSealBuildResult<T, V> :=",
            "Sealed { artifact: BackendSealArtifact<T, V> }",
            "run_backend_build_candidate_and_seal(",
            "BackendResultSourceAuthority<V>",
            "Ready 分支才执行对应 backend",
            "Blocked 与 NotRequested",
            "candidate + seal",
            "未封装 value 不得外泄",
            "不得伪装成 BlockerRecord",
            "canonical_internal_violation_union",
            "inputs: NonEmpty<InternalContractViolation>",
            "flatten every inputs[i].violations",
            "不得把 InternalContractViolation 与 ContractViolation 混在同一层",
            "不得只取 first item",
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"run_backend_build_candidate_and_seal\(\s*source:\s*BackendResultSourceAuthority<V>\s*\)\s*->\s*BackendSealBuildResult<EstimateOf<V>,\s*V>",
        )
        self.assertNotIn(
            "InternalViolation { violations: NonEmpty<InternalContractViolation> }",
            text,
        )


    def test_gate_contract_closes_manifest_clause_domain_and_failure_priority(self) -> None:
        text = self.contract_text("gate-system")
        for token in (
            "BlockerScopePolicySnapshot:",
            "GateEvaluationAuthority:",
            "GateManifest",
            "GateInvocationId",
            "GateClauseExecutionRecord",
            "GateExecutionLedger",
            "compile_gate_manifest(",
            "run_gate_domain(",
            "extend_gate_ledger_without_overwrite(",
            "clause_record.clause_id == clause_key == clause.clause_id",
            "effective_failure_disposition",
            "exact invocation domain",
            "gate_evaluation_context_digest",
            "canonical union of every InputBlocker failure occurrence",
            "InternalViolation takes priority",
            "must not be converted to InputBlocker",
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"compile_gate_manifest\(\s*context:\s*ProductionValidationContext,\s*specifications:\s*GateSpecificationSet,\s*runner:\s*GateRunnerSnapshot\s*\)\s*->\s*GateManifest",
        )
        self.assertRegex(
            text,
            r"run_gate_domain\(\s*authority:\s*GateEvaluationAuthority,\s*base:\s*GateExecutionLedger,\s*invocation_domain:\s*OrderedSet<GateInvocationId>\s*\)\s*->\s*GateExecutionLedger\s*\|\s*InternalContractViolation",
        )
        self.assertRegex(
            text,
            r"extend_gate_ledger_without_overwrite\(\s*authority:\s*GateEvaluationAuthority,\s*base:\s*GateExecutionLedger,\s*extension:\s*GateExecutionRecordSet,\s*current_stage_contexts:\s*StageContextMap\s*\)\s*->\s*GateExecutionLedger\s*\|\s*InternalContractViolation",
        )
        self.assertIn("canonical_internal_violation_union(gate_internal_violations)", text)
        self.assertNotIn(
            "GateExecutionLedger | NonEmpty<InternalContractViolation>", text
        )
        self.assertNotIn("GatePolicySnapshot", text)
        template = TEMPLATE.read_text(encoding="utf-8")
        for product_token in (
            "for expected_stage in [structure_prerequisite, memory_view_prerequisite, time_view_prerequisite]:",
            "projection_gate_authority := build_gate_evaluation_authority(",
            "projection_gate_ledger := canonical_empty_gate_ledger(projection_gate_authority)",
            "projection_domain := expected_invocation_domain(projection_gate_authority, projection_gate_ledger)",
            "require projection_domain.current_root_stage == expected_stage",
            "projection_gate_ledger := run_gate_domain(",
            "unrequested memory/time view stage still executes its full sibling domain as NotApplicable",
            "value_gate_runtime_authority := build_gate_evaluation_authority(",
            "value_gate_domain := expected_invocation_domain(value_gate_runtime_authority, source.projection_gate_ledger)",
            "pre_seal_ledger := run_gate_domain(",
            "result_seal_runtime_authority := build_gate_evaluation_authority(",
            "result_seal_domain := expected_invocation_domain(result_seal_runtime_authority, authority.pre_seal_gate_ledger)",
            "sealed_gate_ledger := run_gate_domain(",
            "comparison_runtime_authority := build_gate_evaluation_authority(",
            "comparison_base := canonical_empty_gate_ledger(comparison_runtime_authority)",
            "comparison_domain := expected_invocation_domain(comparison_runtime_authority, comparison_base)",
            "comparison_gate_ledger := run_gate_domain(",
        ):
            self.assertIn(product_token, template)
        for loose_product_call in (
            "run every manifest invocation in the exact union",
            "value_gate_records := run exact backend-value invocation domain",
            "value_gate_records := run every backend_value_postcondition clause",
            "seal_gate_records := run result_seal_postconditions",
            "comparison_gate_ledger := run the exact comparison/comparison_only",
        ):
            self.assertNotIn(loose_product_call, template)

    def test_gate_scope_policy_identity_and_clause_resolution_are_closed(self) -> None:
        text = self.contract_text("gate-system")
        for token in (
            "ScopePolicyRef :=",
            "(blocker_scope_policy_digest, rule_id)",
            "blocker_scope_policy: BlockerScopePolicySnapshot",
            "manifest.blocker_scope_policy_digest ==",
            "authority.blocker_scope_policy.blocker_scope_policy_digest",
            "manifest.blocker_scope_policy_digest == specifications.blocker_scope_policy.blocker_scope_policy_digest",
            "unique rule selected by scope_policy_ref.rule_id",
            "selected_rule.blocker_code == disposition.blocker_code",
            "negative fixture: policy digest mismatch",
            "negative fixture: dangling scope rule",
        ):
            self.assertIn(token, text)
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertRegex(
            template,
            r"(?s)GateManifest:\s*schema_version\s*subject_digest\s*production_validation_context_digest\s*verifier_runner_digest\s*gate_specification_set_digest:\s*Digest\s*blocker_scope_policy:\s*BlockerScopePolicySnapshot\s*blocker_scope_policy_digest\s*:=",
        )
        self.assertIn(
            "scope_policy_ref: ScopePolicyRef",
            template,
        )

    def test_gate_invocation_domain_is_derived_and_stage_context_closed(self) -> None:
        text = self.contract_text("gate-system")
        for token in (
            "ExpectedInvocationDomain:",
            "required_prefix_invocations: OrderedSet<GateInvocationId>",
            "new_execution_invocations: OrderedSet<GateInvocationId>",
            "applicable_new_invocations: OrderedSet<GateInvocationId>",
            "current_root_stage: GateStage",
            "expected_invocation_domain(",
            "authority: GateEvaluationAuthority",
            "base: GateExecutionLedger",
            "require invocation_domain == expected_domain.new_execution_invocations",
            "keys(base.records) == expected_domain.required_prefix_invocations",
            "keys(ledger.records) == expected_domain.required_prefix_invocations union expected_domain.new_execution_invocations",
            "ledger.stage_evaluation_context_digests == byte_preserved(base.stage_evaluation_context_digests) union expected_domain.current_stage_contexts",
            "negative fixture: invocation subset",
            "negative fixture: invocation superset",
            "negative fixture: wrong stage context",
        ):
            self.assertIn(token, text)

    def test_gate_authority_is_request_owned_typed_and_reference_derived(self) -> None:
        text = self.contract_text("gate-system")
        for token in (
            "GateEvaluationSubject :=",
            "ProjectionBundleGateSubject { authority: ProjectionBundleAuthority }",
            "MemoryBackendValueGateSubject { authority: BackendValueGateAuthority<MemoryEstimate,MemoryEventView> }",
            "TimeBackendValueGateSubject { authority: BackendValueGateAuthority<StepTimeEstimate,TimeEventView> }",
            "ResultSealSubject<T,V>:",
            "backend_seal_authority: BackendSealAuthority<T,V>",
            "backend_result_candidate: BackendResultCandidate<T>",
            "ComparisonGateSubject { authority: ComparisonSealAuthority }",
            "GateSubjectCoordinates:",
            "derive_gate_subject_coordinates(",
            "build_gate_evaluation_authority(",
            "subject_request_snapshot(subject)",
            "request.canonical_production_evaluation_inputs.common_inputs.blocker_scope_policy_snapshot",
            "manifest.blocker_scope_policy == request_policy == authority.blocker_scope_policy",
            "ProjectionBundleAuthority.request_snapshot.canonical_production_evaluation_inputs.common_inputs.blocker_scope_policy_snapshot == request_policy",
            "current_root_stage := normative_current_root_stage(subject, base)",
            "new_execution_invocations := normative_stage_siblings(manifest.entries, current_root_stage)",
            "required_prefix_invocations := keys(base.records)",
            "applicable_dependency_closure := dependency_transitive_closure(manifest.entries, applicable_new_invocations)",
            "applicable_dependency_closure is a subset of required_prefix_invocations",
            "current_stage_contexts == singleton(current_root_stage, gate_evaluation_context_digest(subject))",
            "literal hand-derived expected domains",
            "must not call expected_invocation_domain",
            "negative fixture: wrong subject arm",
        ):
            self.assertIn(token, text)

    def test_gate_compile_and_runtime_use_request_owned_specification_provenance(self) -> None:
        text = self.contract_text("gate-system")
        input_text = self.contract_text("input-facts")
        for token in (
            "GateSpecificationSetRef:",
            "gate_specification_set_digest: Digest",
            "require manifest.gate_specification_set_digest == specifications.gate_specification_set_digest",
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
            "require manifest.blocker_scope_policy == request_policy == authority.blocker_scope_policy",
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
        ):
            self.assertIn(token, text)
        for token in (
            "gate_runner_snapshot: GateRunnerSnapshot",
            "build_request_snapshot gate specification equations:",
            "request_policy := common.blocker_scope_policy_snapshot",
            "request_specifications := common.gate_specification_set_snapshot",
            "request_runner := common.gate_runner_snapshot",
            "require request_specifications.blocker_scope_policy == request_policy",
            "require common.gate_specification_set_ref.gate_specification_set_digest == request_specifications.gate_specification_set_digest",
            "require entry.applicability_predicate_digest == request_runner.predicate_implementation_digests[entry.applicability_predicate_ref]",
            "require clause.predicate_implementation_digest == request_runner.predicate_implementation_digests[clause.exact_predicate_ref]",
            "negative fixture: E+P2 request specification embeds different blocker policy",
        ):
            self.assertIn(token, input_text)
        builder = text.split("build_gate_evaluation_authority equations:", 1)[1].split(
            "expected_invocation_domain(", 1
        )[0]
        self.assertIn("request_specifications", builder)
        self.assertIn("request_specification_ref", builder)
        self.assertRegex(
            text,
            r"build_gate_evaluation_authority\(\s*subject:\s*GateEvaluationSubject,\s*manifest:\s*GateManifest,\s*runner:\s*GateRunnerSnapshot\s*\)\s*->\s*GateEvaluationAuthority\s*\|\s*InternalContractViolation",
        )
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertRegex(
            template,
            r"(?s)CommonProductionInputs:.*?blocker_scope_policy_snapshot:\s*BlockerScopePolicySnapshot.*?gate_specification_set_snapshot:\s*GateSpecificationSet.*?gate_specification_set_ref:\s*GateSpecificationSetRef.*?RequestSnapshotBuildResult :=",
        )

    def test_gate_executes_only_current_stage_and_preserves_prefix(self) -> None:
        text = self.contract_text("gate-system")
        for token in (
            "new_execution_invocations: OrderedSet<GateInvocationId>",
            "required_prefix_invocations: OrderedSet<GateInvocationId>",
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
            "applicable_dependency_closure := dependency_transitive_closure(manifest.entries, applicable_new_invocations)",
            "applicable_dependency_closure is a subset of required_prefix_invocations",
            "require every record in applicable_dependency_closure is Pass",
            "NotApplicable only records the non-applicable sibling itself",
            "NotApplicable never satisfies a dependency of an applicable invocation",
            "negative fixture: applicable invocation depends on NotApplicable sibling",
        ):
            self.assertIn(token, text)
        self.assertNotIn(
            "require each dependency record in base is Pass or an authority-consistent NotApplicable",
            text,
        )

    def test_gate_evaluation_context_has_one_branch_function_and_record_truth(self) -> None:
        text = self.contract_text("gate-system")
        for token in (
            "gate_evaluation_context_digest(subject: GateEvaluationSubject) :=",
            "ProjectionBundleGateSubject -> authority.projection_bundle_authority_digest",
            "MemoryBackendValueGateSubject | TimeBackendValueGateSubject -> authority.value_gate_authority_digest",
            "MemoryResultSealGateSubject | TimeResultSealGateSubject -> hash(subject.backend_seal_authority.backend_seal_authority_digest, subject.backend_result_candidate.backend_result_candidate_digest)",
            "ComparisonGateSubject -> hash(authority.comparison_seal_authority_digest, authority.result_candidate.comparison_result_candidate_digest)",
            "current_stage_contexts == singleton(current_root_stage, gate_evaluation_context_digest(subject))",
            "gate_record.evaluation_input_digest == gate_evaluation_context_digest(subject)",
            "clause_record.evaluation_input_digest == gate_evaluation_context_digest(subject)",
            "legal ResultSealSubject context == hash(backend_seal_authority_digest, backend_result_candidate_digest)",
            "legal ComparisonGateSubject context == hash(comparison_seal_authority_digest, comparison_result_candidate_digest)",
            "base.stage_evaluation_context_digests[backend_value_postcondition] == V",
            "expected.current_stage_contexts[result_seal_postcondition] == R",
        ):
            self.assertIn(token, text)
        template = TEMPLATE.read_text(encoding="utf-8")
        for forbidden in (
            "evaluation_subject_digest(subject: GateEvaluationSubject) :=",
            "result_seal_subject_digest :=",
            "seal_context_digest :=",
            "comparison_context_digest :=",
        ):
            self.assertNotIn(forbidden, template)

    def test_gate_fixture_coverage_is_declarative_and_offline_owned(self) -> None:
        gate_text = self.contract_text("gate-system")
        conformance_text = self.contract_text("conformance")
        for token in (
            "ClauseCoverageRequirement:",
            "coverage_requirement: ClauseCoverageRequirement",
            "contains no fixture identifiers",
        ):
            self.assertIn(token, gate_text)
        for token in (
            "FixtureBinding:",
            "bindings: OrderedMap<FixtureRef, FixtureBinding>",
            "FixtureSet is the sole fixture-to-(invocation, clause) mapping",
            "derived_coverage_by_clause := derive_fixture_coverage(bindings)",
        ):
            self.assertIn(token, conformance_text)
        template = TEMPLATE.read_text(encoding="utf-8")
        gate_clause = re.search(
            r"(?s)GateClause:\s*(.*?)\s*GateClauseOutcome :=", template
        )
        self.assertIsNotNone(gate_clause)
        assert gate_clause is not None
        self.assertIn(
            "coverage_requirement: ClauseCoverageRequirement", gate_clause.group(1)
        )
        self.assertNotIn("positive_fixture_refs", gate_clause.group(1))
        self.assertNotIn("negative_boundary_fixture_refs", gate_clause.group(1))

    def test_comparison_contract_consumes_only_same_request_sealed_artifacts(self) -> None:
        text = self.contract_text("comparison")
        for token in (
            "CanonicalSchemaPathSet:",
            "BasisMismatch:",
            "CoverageDelta:",
            "ComparisonBasisPair",
            "ComparisonSourceAuthority",
            "ComparisonResultCandidate",
            "BackendSealArtifact",
            "derive_comparison_basis_pair(",
            "build_comparison_source_authority(",
            "compare_per_metric_from_authority(",
            "same RequestSnapshot",
            "config_ref",
            "evaluation_instance_digest",
            "canonical_comparison_basis_pairs",
            "ComparableDelta",
            "Incomparable",
            "Unavailable",
            "NotRequested",
            "UndefinedZeroBaseline",
            "registry/calibration/fallback/assumption",
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"derive_comparison_basis_pair\(\s*request:\s*RequestSnapshot,\s*left:\s*ComparisonArmAuthority,\s*right:\s*ComparisonArmAuthority,\s*metric:\s*memory\s*\|\s*time\s*\)\s*->\s*ComparisonBasisPair",
        )
        self.assertRegex(
            text,
            r"compare_per_metric_from_authority\(\s*source:\s*ComparisonSourceAuthority\s*\)\s*->\s*ComparisonResult",
        )
        self.assertNotRegex(
            text,
            r"compare_per_metric_from_authority\([^)]*\)\s*->\s*ComparisonSealArtifact",
        )

    def test_task7_schema_and_boundary_contracts_are_structural(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")

        def fields(schema: str) -> dict[str, str]:
            match = re.search(
                rf"(?m)^{re.escape(schema)}:\s*\n"
                rf"(?P<body>(?:  [^\n]*(?:\n|$))*)",
                template,
            )
            self.assertIsNotNone(match, schema)
            assert match is not None
            return {
                name: field_type.strip()
                for name, field_type in re.findall(
                    r"(?m)^  ([a-z][a-z0-9_]*):\s*([^\n/]+?)(?:\s*//.*)?$",
                    match.group("body"),
                )
            }

        coverage = fields("Coverage")
        expected_numerators = {
            "op_occurrences_modeled": "OpOccurrenceCount",
            "op_occurrences_blocked": "OpOccurrenceCount",
            "storage_instances_modeled": "StorageInstanceCount",
            "bytes_modeled": "ByteCount",
            "time_events_priced": "TimeEventCount",
        }
        for name, field_type in expected_numerators.items():
            self.assertEqual(coverage.get(name), field_type, name)
        for universe in (
            "source_obligation_universe",
            "event_obligation_universe",
            "op_occurrence_universe",
            "storage_instance_universe",
            "byte_universe",
            "time_event_universe",
        ):
            self.assertIn(universe, coverage)

        self.assertEqual(
            fields("CalibrationSet"),
            {
                "active_compute_records": "Map[MeasurementKey, ComputeMeasurementRecord]",
                "measurement_protocols": "Map[ProtocolDigest, MeasurementProtocol]",
                "calibration_train_manifest_digest": "Digest",
            },
        )
        self.assertEqual(fields("HoldoutEvaluationManifest").get("split"), "holdout")
        self.assertEqual(
            fields("CommonProductionInputs"),
            {
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
            },
        )
        self.assertEqual(
            fields("CanonicalProductionEvaluationInputs"),
            {
                "common_inputs": "CommonProductionInputs",
                "configurations": "NonEmptyOrderedMap[ConfigRef, CanonicalConfigEvaluationInput]",
            },
        )
        self.assertEqual(
            fields("RequestSnapshot"),
            {
                "canonical_production_evaluation_inputs": "CanonicalProductionEvaluationInputs",
                "requested_backend_inputs": "OrderedMap[memory | time, RequestedBackendInput]",
                "requested_backends": "OrderedSet[memory | time]",
                "comparison_request": "ComparisonRequest | None",
                "request_digest": "Digest",
            },
        )
        self.assertEqual(
            fields("ComparisonBasis"),
            {
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
                "canonical_resolved_fallbacks_and_assumptions": "CanonicalResolvedFallbacksAndAssumptions",
                "backend_and_numeric_semantic_versions": "SemanticVersionSet",
                "masked_config_evaluation_input": "MaskedCanonicalConfigEvaluationInput",
            },
        )
        self.assertEqual(
            fields("TimeSimulationInputDomain"),
            {
                "simulation_core_digest": "Digest",
                "time_registry_snapshot": "TimeRegistrySnapshot",
                "calibration_set": "CalibrationSet",
                "calibration_train_manifest_snapshot": "CalibrationTrainManifest",
                "communication_model_snapshot": "CommunicationModelSnapshot",
                "time_cost_policy": "TimeCostPolicy",
                "cost_bindings": "ExactCostBindings",
                "stream_bindings": "ExactStreamBindings",
                "resolved_time_fallbacks_and_assumptions": "CanonicalResolvedFallbacksAndAssumptions",
                "progress_semantics": "ProgressSemanticsSnapshot",
                "projection_witness_digest": "Digest",
                "time_backend_semantic_version": "SemanticVersion",
                "time_numeric_semantic_version": "SemanticVersion",
            },
        )
        self.assertEqual(
            set(fields("HardwareProfile")),
            {
                "devices",
                "topology",
                "bandwidth",
                "link_latency",
                "allocator_alignment",
                "physical_stream_catalog",
            },
        )

        comparison = re.search(
            r"compare_metric_outcome\(left_status, right_status, basis_pair\):"
            r"(?P<body>.*?)end compare_metric_outcome",
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(comparison)
        assert comparison is not None
        body = re.sub(r"\s+", " ", comparison.group("body"))
        branches = (
            "if left_status == NotRequested and right_status == NotRequested:",
            "return NotRequested",
            "if left_status != Ok or right_status != Ok:",
            "return Unavailable",
            "if basis_pair.left != basis_pair.right:",
            "return Incomparable",
            "return ComparableDelta",
        )
        positions = [body.find(branch) for branch in branches]
        self.assertNotIn(-1, positions)
        self.assertEqual(positions, sorted(positions))

        non_goals = re.search(
            r'<h3 id="c0-2".*?</h3>(.*?)<h3 id="c0-3"',
            template,
            re.DOTALL,
        )
        decisions = re.search(
            r'<h3 id="c14-1".*?</h3>(.*?)<h3 id="c14-2"',
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(non_goals)
        self.assertIsNotNone(decisions)
        assert non_goals is not None and decisions is not None
        ids = re.findall(r'data-non-goal-id="([A-Z0-9-]+)"', non_goals.group(1))
        refs = re.findall(r'data-non-goal-ref="([A-Z0-9-]+)"', decisions.group(1))
        decision_ids = re.findall(
            r'data-decision-id="([A-Z0-9-]+)"', decisions.group(1)
        )
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(refs), 10)
        self.assertEqual(set(ids), set(refs))
        self.assertEqual(decision_ids, [f"D{index}" for index in range(1, 20)])

    def test_task7_authority_and_behavior_equations_are_closed(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")

        request_input = re.search(
            r"RequestedBackendInput\s*:=\s*(.*?)RequestedBackendInputMap\s*:=",
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(request_input)
        assert request_input is not None
        request_text = re.sub(r"\s+", " ", request_input.group(1))
        self.assertIn(
            "calibration_train_manifest_snapshot: CalibrationTrainManifest",
            request_text,
        )
        self.assertNotIn("HoldoutEvaluationManifest", request_text)

        time_constructor = re.search(
            r"build_time_projection_candidate\(request, evaluation_identity, core\):"
            r"(?P<body>.*?)evaluate_and_finalize_projection_bundle\(authority\):",
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(time_constructor)
        assert time_constructor is not None
        time_text = re.sub(r"\s+", " ", time_constructor.group("body"))
        for equation in (
            "calibration := time_inputs.calibration_set",
            "train_manifest := time_inputs.calibration_train_manifest_snapshot",
            "calibration.calibration_train_manifest_digest == train_manifest.calibration_train_manifest_digest",
            "keys(calibration.active_compute_records) == train_manifest.active_measurement_keys",
            "keys(calibration.measurement_protocols) == train_manifest.active_protocol_digests",
        ):
            self.assertIn(equation, time_text)

        coverage_region = re.search(
            r"Coverage:\s*(?P<body>.*?)<h3 id=\"c2-4\"",
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(coverage_region)
        assert coverage_region is not None
        for equation in (
            "source_obligations_total = source_obligations_planned + source_obligations_residual + source_obligations_proven_not_executed",
            "event_obligations_total = event_obligations_planned + event_obligations_blocked + event_obligations_not_applicable",
        ):
            self.assertIn(equation, coverage_region.group("body"))

        basis = re.search(
            r"ComparisonBasis:\s*(?P<body>.*?)comparison_basis_digest\s*:=",
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(basis)
        assert basis is not None
        self.assertIn("logical_rank_id_set", basis.group("body"))
        self.assertNotIn("HoldoutEvaluationManifest", basis.group("body"))

        bundle = re.search(
            r"evaluate_and_finalize_projection_bundle\(authority\):"
            r"(?P<body>.*?)</code></pre>",
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(bundle)
        assert bundle is not None
        bundle_text = re.sub(r"\s+", " ", bundle.group("body"))
        for scope_equation in (
            "each construction blocker contains candidate.backend in affected_backends",
            "every BlockerRecord that prevents RuntimeBuildResult or CoreBuildResult from being Ready has affected_backends=={memory,time}",
            "every InputBlocker occurrence from a shared structure/G-IR invocation has affected_backends=={memory,time}",
        ):
            self.assertIn(scope_equation, bundle_text)

        time_digest = re.search(
            r"time_simulation_digest\s*:=\s*(?P<body>.*?)result_digest\s*:=",
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(time_digest)
        assert time_digest is not None
        self.assertNotIn("HoldoutEvaluationManifest", time_digest.group("body"))

    def test_comparison_schema_derivation_and_task5_digest_exclusions_are_closed(self) -> None:
        text = self.contract_text("comparison")
        for token in (
            "ComparisonSchemaSnapshot:",
            "root_schema",
            "allowed_axis_paths",
            "cross_field_deterministic_derivation_rules",
            "comparison_schema_semantic_version",
            "comparison_schema_snapshot_digest :=",
            "comparison_schema_snapshot_digest: Digest",
            "declared_paths: OrderedSet<CanonicalSchemaPath>",
            "derivation_closure == derive_closure(snapshot, declared_paths)",
            "request.canonical_production_evaluation_inputs.common_inputs.comparison_schema_snapshot",
            "derive_comparison_basis_pair recomputes",
            "G-REP2 recomputes",
            "negative fixture: inject other_config_indexed_production_inputs.x into derivation closure",
            "BasisMismatch.schema_path == other_config_indexed_production_inputs.x",
        ):
            self.assertIn(token, text)
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertRegex(
            template,
            r"(?s)CommonProductionInputs:.*?comparison_schema_snapshot:\s*ComparisonSchemaSnapshot.*?RequestSnapshot:",
        )
        exact_exclusions = {
            "SourceFileSnapshot": "content_digest",
            "SourceSnapshot": "source_snapshot_digest",
            "StructureRegistrySnapshot": "structure_registry_digest",
            "RuntimeRegistrySnapshot": "runtime_registry_digest",
            "RankCodeIRBuildInput": "rank_build_input_digest",
            "RuntimeRuleSnapshot": "rule_digest",
            "CodeIR": "model_digest",
            "RuntimeEventPlan": "runtime_plan_digest",
            "SimulationPlanCore": "simulation_core_digest",
            "ExecutionDeployment": "deployment_digest",
            "HardwareBindingPolicySnapshot": "policy_digest",
            "MeasurementProtocol": "protocol_digest",
            "CalibrationTrainManifest": "calibration_train_manifest_digest",
            "TimeCostPolicy": "policy_digest",
            "NumericPolicySnapshot": "numeric_policy_digest",
            "CommunicationFormulaRef": "digest",
            "CommunicationModelSnapshot": "communication_model_digest",
            "MemoryRegistrySnapshot": "memory_registry_digest",
            "ResolvedEventSemantic": "resolved_semantic_digest",
            "KernelVariantBinding": "kernel_variant_binding_digest",
            "StreamAssignmentPolicySnapshot": "policy_digest",
            "StreamBinding": "stream_binding_digest",
            "StorageBinding": "storage_binding_digest",
            "WorkspaceBinding": "workspace_binding_digest",
            "RouteBinding": "route_binding_digest",
            "CostBinding": "cost_binding_digest",
            "MemoryProjectionSemanticsSnapshot": "memory_projection_semantics_snapshot_digest",
            "EstimateContext": "estimate_context_digest",
            "RequestSnapshot": "request_digest",
            "EvaluationInstanceIdentity": "evaluation_instance_digest",
            "ProductionSubject": "subject_digest",
            "ProductionValidationContext": "production_validation_context_digest",
            "ProjectionCandidate": "projection_candidate_digest",
            "ProjectionBundleAuthority": "projection_bundle_authority_digest",
            "ComparisonSchemaSnapshot": "comparison_schema_snapshot_digest",
            "CanonicalSchemaPathSet": "canonical_schema_path_set_digest",
            "ComparisonBasis": "comparison_basis_digest",
            "CoverageDelta": "coverage_delta_digest",
            "ComparisonSourceAuthority": "comparison_source_authority_digest",
            "ComparisonResultCandidate": "comparison_result_candidate_digest",
            "ComparisonSealAuthority": "comparison_seal_authority_digest",
            "ComparisonSealArtifact": "comparison_seal_artifact_digest",
            "BlockerScopePolicySnapshot": "blocker_scope_policy_digest",
            "GateSpecificationSet": "gate_specification_set_digest",
            "GateRunnerSnapshot": "verifier_runner_digest",
            "GateEvaluationAuthority": "gate_evaluation_authority_digest",
            "GateManifest": "gate_manifest_digest",
            "GateExecutionLedger": "gate_execution_ledger_digest",
            "EstimateCandidate": "estimate_candidate_digest",
            "BackendResultSourceAuthority": "source_authority_digest",
            "BackendValueGateAuthority": "value_gate_authority_digest",
            "BackendResultCandidate": "backend_result_candidate_digest",
            "BackendSealAuthority": "backend_seal_authority_digest",
            "BackendSealArtifact": "backend_seal_artifact_digest",
            "TraceFixture": "fixture_digest",
            "FixtureBinding": "fixture_binding_digest",
            "FixtureSet": "fixture_set_digest",
            "ValidationPolicy": "validation_policy_digest",
            "VerifierRunner": "verifier_runner_digest",
            "TrustStoreSnapshot": "trust_store_snapshot_digest",
            "RunnerAttestationPolicy": "runner_attestation_policy_digest",
            "MeasuredExecutionEnvironment": "measured_execution_environment_digest",
            "ConformanceInvocationAuthority": "conformance_invocation_digest",
            "RunnerAttestation": "runner_attestation_digest",
            "ConformanceExecutionRecord": "conformance_execution_record_digest",
            "ConformanceObservedOutput": "observed_output_digest",
            "ConformanceExecutionLedger": "conformance_execution_ledger_digest",
            "ReleaseApprovalStoreSnapshot": "release_approval_store_snapshot_digest",
            "ReleaseApprovalAuthority": "release_approval_authority_digest",
            "ReleaseApprovalArtifact": "release_approval_artifact_digest",
            "ConformanceSealArtifact": "conformance_seal_artifact_digest",
            "ConformanceReport": "conformance_report_digest",
            "ReleasePolicy": "release_policy_digest",
            "Estimate": "result_digest",
        }
        table = template.split("derived-digest exclusion table (exact):", 1)[1].split(
            "all nested input digests not named above remain in the payload", 1
        )[0]
        self.assertNotIn("*Binding", table)
        self.assertNotIn("*Snapshot / *Policy", table)
        parsed_rows = re.findall(
            r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)\s*-&gt;\s*\{\s*([a-z][a-z0-9_]*)\s*\}\s*$",
            table,
        )
        self.assertEqual(len(parsed_rows), len(exact_exclusions))
        self.assertEqual(dict(parsed_rows), exact_exclusions)
        self.assertEqual(len({owner for owner, _ in parsed_rows}), len(parsed_rows))
        for owner_type, own_field in exact_exclusions.items():
            with self.subTest(owner_type=owner_type):
                self.assertRegex(
                    template,
                    rf"(?m)^\s*{re.escape(owner_type)}\s*-&gt;\s*\{{\s*{re.escape(own_field)}\s*\}}\s*$",
                )
        canonical_owner_calls = {
            name
            for name in re.findall(
                r"canonical_payload_without_derived_digests\(([A-Z][A-Za-z0-9_]*)\)",
                template,
            )
            if name != "X"
        }
        self.assertLessEqual(canonical_owner_calls, set(exact_exclusions))

    def test_conformance_contract_is_offline_digest_closed_and_release_versioned(self) -> None:
        text = self.contract_text("conformance")
        for token in (
            "TraceFixture:",
            "FixtureSet:",
            "ConformanceFinding:",
            "ApprovedDigestSet:",
            "ConformanceReport",
            "ReleaseDecision",
            "run_conformance(",
            "apply_release_policy(",
            "derive_approved_digest_set(",
            "derive_approved_digest_set fields:",
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
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"run_conformance\(\s*authority:\s*ConformanceInvocationAuthority,\s*trust_root:\s*ConformanceTrustRootCapability,\s*measured_environment:\s*MeasuredExecutionEnvironment\s*\)\s*->\s*ConformanceRunResult",
        )
        self.assertRegex(
            text,
            r"apply_release_policy\(\s*artifact:\s*ConformanceSealArtifact,\s*approval:\s*ReleaseApprovalArtifact,\s*policy:\s*ReleasePolicy,\s*release_trust_root:\s*ReleaseApprovalTrustRootCapability\s*\)\s*->\s*ReleaseDecision",
        )
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertRegex(
            template,
            r"(?s)ReleasePolicy:\s*schema_version.*?pass:.*?derive_approved_digest_set\(artifact,\s*self\).*?release_policy_digest\s*:=\s*hash\(canonical_payload_without_derived_digests\(ReleasePolicy\)\)",
        )
        self.assertRegex(
            template,
            r"(?s)<h3 id=\"c12-6\">.*?derive_approved_digest_set\(artifact,\s*policy\).*?all ten approved digest fields.*?<h2 id=\"c13\"",
        )
        self.assertNotRegex(template, r"六个\s+approved digest|四个\s+digest")

    def test_release_approval_uses_external_trust_root_capability(self) -> None:
        """A policy/store/key tuple supplied by the approval cannot authorize itself."""
        text = self.contract_text("conformance")
        for token in (
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
            "policy.approval_signature_scheme in root.supported_release_approval_signature_schemes",
            "trusted_approver_public_key_material := root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]",
            "release_approval_signed_message := canonical_tuple(",
            "canonical(derived_approved)",
            "verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)",
            "negative fixture: release approval self-owned approver key",
            "negative fixture: release approval wrong store snapshot",
            "negative fixture: release approval unsupported signature scheme",
            "negative fixture: release approval substituted policy",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

        self.assertRegex(
            text,
            r"apply_release_policy\(\s*artifact:\s*ConformanceSealArtifact,\s*approval:\s*ReleaseApprovalArtifact,\s*policy:\s*ReleasePolicy,\s*release_trust_root:\s*ReleaseApprovalTrustRootCapability\s*\)\s*->\s*ReleaseDecision",
        )
        self.assertNotIn(
            "apply_release_policy( artifact: ConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy ) -> ReleaseDecision",
            text,
        )
        self.assertNotIn(
            "verify_signature(policy.trusted_approver_key_id",
            text,
        )
        self.assertNotIn(
            "policy.approval_signature_scheme in SUPPORTED_RELEASE_APPROVAL_SIGNATURE_SCHEMES",
            text,
        )

    def test_opaque_trust_root_capabilities_are_not_digest_table_records(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")
        table = template.split("derived-digest exclusion table (exact):", 1)[1].split(
            "all nested input digests not named above remain in the payload", 1
        )[0]
        for capability in (
            "ConformanceTrustRootCapability",
            "ReleaseApprovalTrustRootCapability",
            "ValidatedMeasurementSessionCapability",
        ):
            with self.subTest(capability=capability):
                self.assertNotIn(f"{capability} -&gt;", table)
        self.assertIn(
            "Opaque capabilities ConformanceTrustRootCapability, ReleaseApprovalTrustRootCapability and ValidatedMeasurementSessionCapability are non-serializable, own no derived digest and therefore have no exclusion-table row",
            template,
        )

    def test_conformance_execution_is_attested_sealed_and_recomputed(self) -> None:
        text = self.contract_text("conformance")
        for token in (
            "RunnerAttestation:",
            "ConformanceExecutionRecord:",
            "ConformanceExecutionLedger:",
            "runner_attestation_digest",
            "conformance_execution_ledger_digest :=",
            "conformance_execution_record_digest :=",
            "conformance_report_digest",
            "ConformanceSealArtifact:",
            "conformance_seal_artifact_digest :=",
            "ConformanceRunResult :=",
            "Completed { artifact: ConformanceSealArtifact }",
            "| InternalViolation { violation: InternalContractViolation }",
            "runner crash, schema failure, digest failure or conservation failure",
            "complete, valid execution",
            "record.fixture_ref == record_key",
            "record.target_invocation_id == binding.target_invocation_id",
            "record.target_clause_id == binding.target_clause_id",
            "derived_approved := derive_approved_digest_set(artifact, policy)",
            "require approved == derived_approved",
            "validate_and_seal_conformance first operation:",
            "recompute authority.production_subject.subject_digest",
            "recompute every authority.fixture_set binding, nested fixture and fixture_set_digest",
            "recompute authority.validation_policy.validation_policy_digest",
            "recompute authority.gate_manifest and every nested entry/clause digest",
            "recompute authority.verifier_runner.verifier_runner_digest",
            "recompute authority.conformance_invocation_digest only after all nested recomputations",
            "before reading observed, ledger, attestation, session.measured_environment or trust_root",
            "negative fixture: old invocation digest plus substituted policy or trust-store reference",
        ):
            self.assertIn(token, text)
        self.assertNotRegex(
            text,
            r"run_conformance\([^)]*\)\s*->\s*ConformanceReport",
        )
        self.assertNotRegex(
            text,
            r"apply_release_policy\(\s*report:\s*ConformanceReport",
        )
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertRegex(
            template,
            r"(?s)ConformanceReport:.*?conformance_report_digest\s*:=\s*hash\(canonical_payload_without_derived_digests\(ConformanceReport\)\)",
        )

    def test_conformance_provenance_is_signed_observed_and_release_approved(self) -> None:
        text = self.contract_text("conformance")
        for token in (
            "ConformanceInvocationAuthority:",
            "production_subject: ProductionSubject",
            "fixture_set: FixtureSet",
            "validation_policy: ValidationPolicy",
            "gate_manifest: GateManifest",
            "verifier_runner: VerifierRunner",
            "trust_store_snapshot_ref: TrustStoreSnapshotRef",
            "runner_attestation_policy_ref: RunnerAttestationPolicyRef",
            "conformance_invocation_digest :=",
            "evaluation_input_digest := hash(authority.conformance_invocation_digest, session.measured_execution_environment_digest, canonical(binding), binding.fixture_binding_digest)",
            "ConformanceObservedOutput:",
            "observed_output_digest := hash(canonical execution_records, findings, coverage_gaps)",
            "ledger.observed_output_digest == report.observed_output_digest == attestation.observed_output_digest",
            "TrustStoreSnapshot:",
            "trusted_key_material_by_id: OrderedMap<AttestationKeyId, TrustedPublicKeyMaterial>",
            "key_registry_digest := hash(canonical trusted_key_material_by_id)",
            "supported_signature_schemes: OrderedSet<SignatureScheme>",
            "RunnerAttestationPolicy:",
            "expected_execution_environment_digest",
            "trusted_attestation_key_id, attestation_signature_scheme",
            "trust_store_snapshot_digest",
            "runner_attestation_policy_digest := hash(canonical_payload_without_derived_digests(RunnerAttestationPolicy))",
            "ConformanceTrustRootCapability: opaque deployment/verifier-owned capability",
            "cannot be constructed by request deserialization",
            "expected_trust_store_snapshot: TrustStoreSnapshot",
            "expected_runner_attestation_policy: RunnerAttestationPolicy",
            "MeasuredExecutionEnvironment:",
            "protected_environment_measurer",
            "measured_execution_environment_digest := hash(canonical_payload_without_derived_digests(MeasuredExecutionEnvironment))",
            "signed_message := canonical_tuple( measured_execution_environment_digest, executable_artifact_digest, trusted_attestation_key_id, attestation_signature_scheme, conformance_invocation_digest, observed_output_digest, verifier_runner_digest)",
            "verifier_runner_digest := hash(canonical_payload_without_derived_digests(VerifierRunner))",
            "(store, policy) := resolve_conformance_trust_root(trust_root)",
            "authority.trust_store_snapshot_ref.trust_store_snapshot_digest == store.trust_store_snapshot_digest",
            "authority.runner_attestation_policy_ref.runner_attestation_policy_digest == policy.runner_attestation_policy_digest",
            "policy.attestation_signature_scheme in store.supported_signature_schemes",
            "trusted_key_material := store.trusted_key_material_by_id[policy.trusted_attestation_key_id]",
            "verify_signature(trusted_key_material, policy.attestation_signature_scheme, signed_message, attestation.signature)",
            "measurement_envelope_digest := recompute measured_environment.measured_execution_environment_digest",
            "measured_payload_digest := hash(canonical measured_environment.measured_environment_payload)",
            "attestation.measured_execution_environment_digest == measurement_envelope_digest",
            "measured_payload_digest == policy.expected_execution_environment_digest",
            "attestation.executable_artifact_digest == runner.executable_artifact_digest",
            "attestation.trusted_attestation_key_id == policy.trusted_attestation_key_id",
            "attestation.attestation_signature_scheme == policy.attestation_signature_scheme",
            "negative fixture: wrong runner attestation trust store",
            "negative fixture: unsupported runner attestation signature scheme",
            "negative fixture: tampered runner execution environment",
            "negative fixture: runner self-owned attestation key",
            "validate_and_seal_conformance(",
            "ReleaseApprovalStoreSnapshot:",
            "ReleaseApprovalAuthority:",
            "ReleaseApprovalArtifact:",
            "trusted_approver_key_id",
            "approval_signature_scheme",
            "verify release approval signature",
            "negative fixture: no execution but forged all-pass",
            "negative fixture: forged runner attestation",
            "negative fixture: forged ApprovedDigestSet",
            "negative fixture: tampered execution record or observed output",
        ):
            self.assertIn(token, text)
        self.assertRegex(
            text,
            r"validate_and_seal_conformance\(\s*authority:\s*ConformanceInvocationAuthority,\s*trust_root:\s*ConformanceTrustRootCapability,\s*session:\s*ValidatedMeasurementSessionCapability,\s*observed:\s*ConformanceObservedOutput,\s*ledger:\s*ConformanceExecutionLedger,\s*attestation:\s*RunnerAttestation\s*\)\s*->\s*ConformanceRunResult",
        )
        self.assertRegex(
            text,
            r"run_conformance\(\s*authority:\s*ConformanceInvocationAuthority,\s*trust_root:\s*ConformanceTrustRootCapability,\s*measured_environment:\s*MeasuredExecutionEnvironment\s*\)\s*->\s*ConformanceRunResult",
        )
        authority_schema = re.search(
            r"(?s)ConformanceInvocationAuthority:\s*(.*?)\s*RunnerAttestation:",
            text,
        )
        self.assertIsNotNone(authority_schema)
        assert authority_schema is not None
        self.assertNotIn(
            "trust_store_snapshot: TrustStoreSnapshot", authority_schema.group(1)
        )
        self.assertNotIn(
            "runner_attestation_policy: RunnerAttestationPolicy",
            authority_schema.group(1),
        )
        self.assertNotIn(
            "runner signs canonical_tuple( policy.expected_execution_environment_digest",
            text,
        )
        self.assertNotIn(
            "verify_signature(runner.trusted_attestation_key_id, runner.attestation_signature_scheme",
            text,
        )

    def test_conformance_measurement_envelope_is_session_bound_and_non_replayable(self) -> None:
        text = self.contract_text("conformance")
        envelope_match = re.search(
            r"MeasuredExecutionEnvironment:\s*(.*?)\s*ValidatedMeasurementSessionCapability:",
            text,
        )
        self.assertIsNotNone(envelope_match)
        assert envelope_match is not None
        envelope_schema = envelope_match.group(1)
        for field in (
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
        ):
            with self.subTest(field=field):
                self.assertIn(field, envelope_schema)

        for token in (
            "ValidatedMeasurementSessionCapability:",
            "opaque call-scoped, non-serializable and non-transferable capability",
            "validate_and_consume_measurement_envelope( authority: ConformanceInvocationAuthority, runner: VerifierRunner, trust_root: ConformanceTrustRootCapability, measured_environment: MeasuredExecutionEnvironment ) -> ValidatedMeasurementSessionCapability | InternalContractViolation",
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
            "negative fixture: replay valid measurement envelope from another execution session",
            "negative fixture: change only measurer identity or freshness evidence while retaining old envelope digest",
            "change only measurer_identity_and_freshness_evidence changes measured_execution_environment_digest",
        ):
            with self.subTest(token=token):
                self.assertIn(token, text)

        self.assertNotIn(
            "measured_execution_environment_digest := hash(canonical measured_environment_payload)",
            text,
        )
        self.assertNotIn(
            "signed_message := canonical_tuple( execution_environment_digest",
            text,
        )

    def test_conformance_report_is_uniquely_derived_from_observed_output(self) -> None:
        text = self.contract_text("conformance")
        for token in (
            "derive_conformance_report( authority: ConformanceInvocationAuthority, observed: ConformanceObservedOutput, ledger: ConformanceExecutionLedger, attestation: RunnerAttestation ) -> ConformanceReport",
            "ConformanceReport.findings: OrderedMap<FindingId, ConformanceFinding>",
            "report.findings == observed.findings",
            "failure_findings := policy_classified_failure_findings(authority.validation_policy, observed.findings)",
            "verdict := fail iff failure_findings is non-empty",
            "else insufficient iff exact_evidence_or_coverage_gaps(authority, observed, ledger) is non-empty",
            "else pass",
            "execute -> observed -> runner attestation -> ledger -> derived report -> seal",
            "negative fixture: valid signature plus failing observed output cannot be sealed with forged pass",
        ):
            self.assertIn(token, text)
        self.assertNotRegex(
            text,
            r"validate_and_seal_conformance\([^)]*report:\s*ConformanceReport",
        )
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertRegex(
            template,
            r"(?s)ConformanceReport:\s*.*?verdict:\s*pass\s*\|\s*fail\s*\|\s*insufficient\s*findings:\s*OrderedMap&lt;FindingId,\s*ConformanceFinding&gt;",
        )

    def test_release_approval_uses_only_external_root_key_and_scheme(self) -> None:
        text = self.contract_text("conformance")
        for token in (
            "derive_approved_digest_set( artifact: ConformanceSealArtifact, policy: ReleasePolicy ) -> ApprovedDigestSet",
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
            "require approval_authority.approval_signature_scheme == policy.approval_signature_scheme",
            "policy.approval_signature_scheme in root.supported_release_approval_signature_schemes",
            "release_approval_signed_message := canonical_tuple(",
            "trusted_approver_public_key_material := root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]",
            "verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)",
            "negative fixture: release approval wrong signature scheme",
            "negative fixture: unsupported release approval signature scheme",
        ):
            self.assertIn(token, text)
        self.assertNotIn(
            "verify_signature(policy.trusted_approver_key_id",
            text,
        )
        self.assertEqual(text.count("derive_approved_digest_set fields:"), 1)

    def test_all_gate_failures_return_one_canonical_internal_violation(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")
        canonical = (
            "InternalViolation { violation: "
            "canonical_internal_violation_union(all_internal_violations) }"
        )
        self.assertGreaterEqual(template.count(canonical), 5)
        self.assertNotIn("InternalContractViolation set", template)
        self.assertNotIn(
            "InternalContractViolation(CV-COMPARISON-CONTRACT set)", template
        )


if __name__ == "__main__":
    unittest.main()
