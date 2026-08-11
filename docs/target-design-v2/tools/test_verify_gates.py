from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import unittest
from html import escape, unescape

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERIFY_GATES = ROOT / "tools" / "verify_gates.py"

VERIFY_GATES_SPEC = importlib.util.spec_from_file_location(
    "target_design_v2_verify_gates",
    VERIFY_GATES,
)
assert VERIFY_GATES_SPEC is not None and VERIFY_GATES_SPEC.loader is not None
verify_gates = importlib.util.module_from_spec(VERIFY_GATES_SPEC)
VERIFY_GATES_SPEC.loader.exec_module(verify_gates)


P1_REQUIRED_TEXT = "\n".join(verify_gates.REQUIRED_TEXT)

P1_GATES = (
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
)


def p1_document(extra: str = "") -> str:
    chapters = "".join(f'<h2 id="c{chapter}"></h2>' for chapter in range(15))
    gates = "".join(f'<tr data-gate="{gate}"></tr>' for gate in P1_GATES)
    return P1_REQUIRED_TEXT + chapters + gates + extra


def validate(html: str) -> list[str]:
    errors: list[str] = []
    verify_gates.check_required(html, errors)
    verify_gates.check_forbidden(html, errors)
    verify_gates.check_chapters(html, errors)
    verify_gates.check_gates(html, errors)
    verify_gates.check_diagrams(html, errors)
    return errors


MODULE_SECTION_PATTERN = re.compile(
    r'(?P<open><section\b(?P<attrs>[^>]*)>)(?P<body>.*?)(?P<close></section>)',
    re.DOTALL,
)
ATTRIBUTE_PATTERN = re.compile(r'([:\w-]+)\s*=\s*(["\'])(.*?)\2', re.DOTALL)


def module_section_match(html: str, module: str) -> re.Match[str]:
    matches: list[re.Match[str]] = []
    for match in MODULE_SECTION_PATTERN.finditer(html):
        attrs = {
            name: value
            for name, _, value in ATTRIBUTE_PATTERN.findall(match.group("attrs"))
        }
        if (
            "module-contract" in attrs.get("class", "").split()
            and attrs.get("data-module") == module
        ):
            matches.append(match)
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one module-contract {module!r}, found {len(matches)}"
        )
    return matches[0]


def rehost_from_module(html: str, module: str, needle: str) -> str:
    match = module_section_match(html, module)
    if needle not in match.group("body"):
        raise AssertionError(f"missing fixture token {needle!r} in {module}")
    body = match.group("body").replace(needle, "REMOVED_FROM_MODULE", 1)
    rewritten = (
        html[: match.start()]
        + match.group("open")
        + body
        + match.group("close")
        + html[match.end() :]
    )
    return rewritten + "\n" + needle


def replace_in_module(
    html: str, module: str, original: str, replacement: str
) -> str:
    match = module_section_match(html, module)
    raw_original = escape(original, quote=False)
    raw_replacement = escape(replacement, quote=False)
    original_parts = re.split(r"\s+", raw_original.strip())
    original_pattern = re.compile(r"\s+".join(map(re.escape, original_parts)))
    if original_pattern.search(match.group("body")) is None:
        raise AssertionError(f"missing fixture token {original!r} in {module}")
    rewritten_body = original_pattern.sub(raw_replacement, match.group("body"), count=1)
    return (
        html[: match.start()]
        + match.group("open")
        + rewritten_body
        + match.group("close")
        + html[match.end() :]
    )


class VerifyGatesContractTest(unittest.TestCase):
    def test_task9_round9_figure10_wraps_profiles_in_optional_sidecar(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
        wrapped = (
            "  opt OfflineConformanceRequested\n"
            "    alt BasicOfflineConformance (default)\n"
            "      BO-&gt;&gt;GL: exact subject manifest fixtures policy and runner\n"
            "      GL--&gt;&gt;BO: clause observations and GateExecutionLedger\n"
            "      BO-&gt;&gt;BO: construct BasicOfflineReportArtifact\n"
            "    else AttestedReleaseConformance\n"
            "      AR-&gt;&gt;GL: same fixtures plus protected measured session\n"
            "      GL--&gt;&gt;AR: clause observations and GateExecutionLedger\n"
            "      AR-&gt;&gt;AR: construct AttestedConformanceSealArtifact\n"
            "      AR-&gt;&gt;AR: apply_release_policy with current profile roots\n"
            "    end\n"
            "  end"
        )
        self.assertEqual(template.count(wrapped), 1)
        handoff_boundary = (
            "Figure 10 先以 OfflineConformanceRequested opt 表达整个离线 sidecar 可选，"
            "其内再以 Basic(default)/Attested 单一 alt 表达 profile 互斥"
        )
        self.assertEqual(handoff.count(handoff_boundary), 1)

        unwrapped = "\n".join(
            line[2:] if line.startswith("  ") else line
            for line in wrapped.splitlines()[1:-1]
        )
        errors = validate(template.replace(wrapped, unwrapped, 1))
        self.assertTrue(
            any("Task9 Figure 10 exact sequence structure" in error for error in errors),
            errors,
        )
        stale_handoff = handoff.replace(
            handoff_boundary,
            "Figure 10 直接以 Basic(default)/Attested alt 表达 profile",
            1,
        )
        handoff_errors: list[str] = []
        verify_gates.check_handoff_conformance_sync(stale_handoff, handoff_errors)
        self.assertTrue(
            any("HANDOFF Figure 10 profile ownership boundary" in error for error in handoff_errors),
            handoff_errors,
        )

    def test_task9_round9_release_revalidates_complete_conformance_closure(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        closure_tokens = (
            "  require current_conformance_store.key_registry_digest ==\n"
            "    hash(canonical current_conformance_store.trusted_key_material_by_id)",
            "  require current_conformance_store.trust_store_snapshot_digest ==\n"
            "    hash(canonical_payload_without_derived_digests(current_conformance_store))",
            "  require current_conformance_policy.runner_attestation_policy_digest ==\n"
            "    hash(canonical_payload_without_derived_digests(current_conformance_policy))",
            "  require current_conformance_policy.trust_store_snapshot_digest ==\n"
            "          current_conformance_store.trust_store_snapshot_digest",
            "  require current_conformance_policy.supported_signature_schemes ==\n"
            "          current_conformance_store.supported_signature_schemes",
            "  require current_conformance_policy.attestation_signature_scheme in\n"
            "          current_conformance_policy.supported_signature_schemes",
            "  require artifact.measured_environment.conformance_invocation_digest ==\n"
            "          artifact.runner_attestation.conformance_invocation_digest ==\n"
            "          artifact.execution_ledger.conformance_invocation_digest ==\n"
            "          artifact.report.conformance_invocation_digest ==\n"
            "          artifact.invocation_authority.conformance_invocation_digest",
            "  require artifact.measured_environment.verifier_runner_digest ==\n"
            "          artifact.runner_attestation.verifier_runner_digest ==\n"
            "          artifact.execution_ledger.verifier_runner_digest ==\n"
            "          artifact.report.verifier_runner_digest ==\n"
            "          artifact.invocation_authority.verifier_runner.verifier_runner_digest",
            "  require artifact.measured_environment.executable_artifact_digest ==\n"
            "          artifact.runner_attestation.executable_artifact_digest ==\n"
            "          artifact.invocation_authority.verifier_runner.executable_artifact_digest",
            "  require artifact.runner_attestation.measured_execution_environment_digest ==\n"
            "          artifact.execution_ledger.measured_execution_environment_digest ==\n"
            "          artifact.report.measured_execution_environment_digest ==\n"
            "          artifact.measured_environment.measured_execution_environment_digest",
            "  require artifact.runner_attestation.observed_output_digest ==\n"
            "          artifact.execution_ledger.observed_output_digest ==\n"
            "          artifact.report.observed_output_digest",
            "  require artifact.execution_ledger.subject_digest ==\n"
            "          artifact.report.subject_digest ==\n"
            "          artifact.invocation_authority.production_subject.subject_digest",
            "  require artifact.execution_ledger.fixture_set_digest ==\n"
            "          artifact.report.fixture_set_digest ==\n"
            "          artifact.invocation_authority.fixture_set.fixture_set_digest",
            "  require artifact.execution_ledger.validation_policy_digest ==\n"
            "          artifact.report.validation_policy_digest ==\n"
            "          artifact.invocation_authority.validation_policy.validation_policy_digest",
            "  require artifact.execution_ledger.gate_manifest_digest ==\n"
            "          artifact.report.gate_manifest_digest ==\n"
            "          artifact.invocation_authority.gate_manifest.gate_manifest_digest",
            "  require artifact.execution_ledger.runner_attestation_digest ==\n"
            "          artifact.report.runner_attestation_digest ==\n"
            "          artifact.runner_attestation.runner_attestation_digest",
            "  require artifact.report.conformance_execution_ledger_digest ==\n"
            "          artifact.execution_ledger.conformance_execution_ledger_digest",
            "  require artifact.report.production_subject ==\n"
            "          artifact.invocation_authority.production_subject",
            "  require artifact.execution_ledger.observed_output_digest ==\n"
            "          artifact.execution_ledger.observed_output.observed_output_digest",
            "  require artifact.report.findings ==\n"
            "          artifact.execution_ledger.observed_output.findings",
        )
        for token in closure_tokens:
            self.assertEqual(template.count(token), 1, token)

        mutations = (
            (closure_tokens[0], closure_tokens[0].replace("current_conformance_store.trusted_key_material_by_id", "approval_authority.store_snapshot.approved_by_subject")),
            (closure_tokens[4], closure_tokens[4].replace("current_conformance_store.supported_signature_schemes", "root.supported_release_approval_signature_schemes")),
            (closure_tokens[6], closure_tokens[6].replace("artifact.invocation_authority.conformance_invocation_digest", "approval_authority.release_approval_authority_digest")),
            (closure_tokens[8], closure_tokens[8].replace("artifact.invocation_authority.verifier_runner.executable_artifact_digest", "artifact.report.conformance_report_digest")),
            (closure_tokens[10], closure_tokens[10].replace("artifact.report.observed_output_digest", "artifact.report.conformance_report_digest")),
            (closure_tokens[19], closure_tokens[19].replace("artifact.execution_ledger.observed_output.findings", "artifact.report.findings")),
        )
        for original, replacement in mutations:
            with self.subTest(mutation=replacement.splitlines()[-1].strip()):
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(
                    any("release current conformance closure mismatch" in error for error in errors),
                    errors,
                )

    def test_task9_round9_terminal_receipt_has_one_formula_at_all_three_stages(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        formula = (
            "terminal_consumption_receipt_key(\n"
            "  environment: MeasuredExecutionEnvironment\n"
            ") := canonical_tuple(\n"
            "  environment.verifier_nonce, environment.conformance_invocation_digest,\n"
            "  environment.verifier_runner_digest, environment.executable_artifact_digest,\n"
            "  environment.execution_session_id, environment.process_identity,\n"
            "  environment.container_identity, environment.execution_time_window,\n"
            "  environment.measured_execution_environment_digest)"
        )
        producer = (
            "  terminal_consumption_receipt_key_value :=\n"
            "    terminal_consumption_receipt_key(measured_environment)\n"
            "  require measured_environment.verifier_nonce is active, bound to this exact key,\n"
            "    and atomically consumed exactly once while persisting a terminal receipt under that exact key"
        )
        seal = (
            "  require session.nonce_consumption_receipt proves the terminal registry contains exact\n"
            "    terminal_consumption_receipt_key(measured_environment)"
        )
        query = (
            "  require conformance_trust_root.measurement_session_authority.has_terminal_consumption_receipt(\n"
            "    terminal_consumption_receipt_key(artifact.measured_environment))"
        )
        for token in (formula, producer, seal, query):
            self.assertEqual(template.count(token), 1, token)

        mutations = (
            (formula, formula.replace("environment.measured_execution_environment_digest", "environment.verifier_runner_digest")),
            (producer, producer.replace("terminal_consumption_receipt_key(measured_environment)", "canonical(measured_environment.verifier_nonce)")),
            (seal, seal.replace("terminal_consumption_receipt_key(measured_environment)", "canonical(measured_environment.verifier_nonce)")),
            (query, query.replace("terminal_consumption_receipt_key(artifact.measured_environment)", "canonical(artifact.measured_environment.verifier_nonce)")),
        )
        for original, replacement in mutations:
            errors = validate(template.replace(original, replacement, 1))
            self.assertTrue(
                any("terminal receipt production/seal/query closure mismatch" in error for error in errors),
                errors,
            )

    def test_task9_round9_current_root_resolve_mutations_are_total_failures(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        resolve = (
            "  (current_conformance_store, current_conformance_policy) :=\n"
            "    resolve_conformance_trust_root(conformance_trust_root)"
        )
        self.assertEqual(template.count(resolve), 1)
        for replacement in (
            resolve.replace("current_conformance_store", "aliased_store", 1),
            "  current conformance root resolution deleted",
        ):
            mutated = template.replace(resolve, replacement, 1)
            errors = validate(mutated)
            self.assertTrue(
                any(
                    "Attested security variable source/dominance mismatch" in error
                    for error in errors
                ),
                errors,
            )

    def test_task9_round8_basic_payload_owners_exclude_transport_result(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        template_boundary = (
            "Basic 五种 content-addressed payload 的 own digest 只使用 "
            "basic-offline/v1 schema 与各自 basic-offline-*/v1 hash domain；"
            "BasicOfflineRunResult 是无 own digest、无 profile_schema_id field 的 "
            "transport union"
        )
        handoff_boundary = (
            "前五种是 content-addressed payload，使用 "
            '`profile_schema_id="basic-offline/v1"` 与彼此独立的 '
            "`basic-offline-*/v1` hash domain；`BasicOfflineRunResult` 只是无 own "
            "digest、无 profile field 的 transport union"
        )
        self.assertEqual(template.count(template_boundary), 1)
        self.assertEqual(handoff.count(handoff_boundary), 1)

        result_union = (
            "BasicOfflineRunResult :=\n"
            "  Completed { artifact: BasicOfflineReportArtifact }\n"
            "  | InternalViolation { violation: InternalContractViolation }"
        )
        self.assertEqual(template.count(result_union), 1)
        for extra in (
            '  profile_schema_id := "basic-offline/v1"\n',
            "  basic_offline_run_result_digest := hash(result)\n",
        ):
            with self.subTest(extra=extra.strip()):
                mutated = template.replace(
                    result_union,
                    result_union.replace(
                        "  Completed", extra + "  Completed", 1
                    ),
                    1,
                )
                errors = validate(mutated)
                self.assertTrue(
                    any(
                        "Basic content-addressed payload and transport result boundary"
                        in error
                        for error in errors
                    ),
                    errors,
                )

        stale_template = template.replace(
            "Basic 五种 content-addressed payload", "Basic 六种 concrete type 与 own digest", 1
        )
        stale_errors = validate(stale_template)
        self.assertTrue(
            any(
                "Basic content-addressed payload and transport result boundary" in error
                for error in stale_errors
            ),
            stale_errors,
        )
        stale_handoff = handoff.replace("前五种是 content-addressed payload", "六种均有 own digest", 1)
        handoff_errors: list[str] = []
        verify_gates.check_handoff_conformance_sync(stale_handoff, handoff_errors)
        self.assertTrue(
            any("BasicOffline payload/result boundary" in error for error in handoff_errors),
            handoff_errors,
        )

    def test_task9_round8_figure10_profiles_are_mutually_exclusive_and_own_artifacts(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        figure = re.search(
            r'<figure\b[^>]*data-diagram-id="result-gate-comparison"[^>]*>.*?'
            r'<pre class="mermaid-source"><code>(?P<body>.*?)</code></pre>',
            template,
            re.DOTALL,
        )
        self.assertIsNotNone(figure)
        assert figure is not None
        source = unescape(figure.group("body"))
        profile_branch = (
            "    alt BasicOfflineConformance (default)\n"
            "      BO->>GL: exact subject manifest fixtures policy and runner\n"
            "      GL-->>BO: clause observations and GateExecutionLedger\n"
            "      BO->>BO: construct BasicOfflineReportArtifact\n"
            "    else AttestedReleaseConformance\n"
            "      AR->>GL: same fixtures plus protected measured session\n"
            "      GL-->>AR: clause observations and GateExecutionLedger\n"
            "      AR->>AR: construct AttestedConformanceSealArtifact\n"
            "      AR->>AR: apply_release_policy with current profile roots\n"
            "    end"
        )
        self.assertEqual(source.count(profile_branch), 1)
        self.assertNotIn("opt BasicOfflineConformance", source)
        self.assertNotIn("opt AttestedReleaseConformance", source)
        self.assertIn(
            "Figure 10 先以 OfflineConformanceRequested opt 表达整个离线 sidecar 可选，其内再以 Basic(default)/Attested 单一 alt 表达 profile 互斥",
            handoff,
        )

        for original, replacement in (
            ("    alt BasicOfflineConformance (default)", "    opt BasicOfflineConformance (default)"),
            ("    else AttestedReleaseConformance", "    end\n    opt AttestedReleaseConformance"),
            (
                "GL--&gt;&gt;BO: clause observations and GateExecutionLedger",
                "GL--&gt;&gt;BO: BasicOfflineReportArtifact",
            ),
            (
                "BO-&gt;&gt;BO: construct BasicOfflineReportArtifact",
                "GL-&gt;&gt;BO: construct BasicOfflineReportArtifact",
            ),
            (
                "GL--&gt;&gt;AR: clause observations and GateExecutionLedger",
                "GL--&gt;&gt;AR: AttestedConformanceSealArtifact",
            ),
        ):
            with self.subTest(mutation=original):
                mutated = template.replace(original, replacement, 1)
                errors = validate(mutated)
                self.assertTrue(
                    any(
                        "Task9 Figure 10 exact mutually-exclusive profile ownership"
                        in error
                        for error in errors
                    ),
                    errors,
                )

    def test_task9_round8_figure10_full_sequence_structure_is_exact(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        profile_branch = (
            "  opt OfflineConformanceRequested\n"
            "    alt BasicOfflineConformance (default)\n"
            "      BO-&gt;&gt;GL: exact subject manifest fixtures policy and runner\n"
            "      GL--&gt;&gt;BO: clause observations and GateExecutionLedger\n"
            "      BO-&gt;&gt;BO: construct BasicOfflineReportArtifact\n"
            "    else AttestedReleaseConformance\n"
            "      AR-&gt;&gt;GL: same fixtures plus protected measured session\n"
            "      GL--&gt;&gt;AR: clause observations and GateExecutionLedger\n"
            "      AR-&gt;&gt;AR: construct AttestedConformanceSealArtifact\n"
            "      AR-&gt;&gt;AR: apply_release_policy with current profile roots\n"
            "    end\n"
            "  end"
        )
        self.assertEqual(template.count(profile_branch), 1)
        mutations = (
            ("  alt Ready", "  opt Ready"),
            ("    BE--&gt;&gt;VA: EstimateCandidate and witness", "    BE-&gt;&gt;VA: EstimateCandidate and witness"),
            (profile_branch, profile_branch + "\n" + profile_branch),
            (
                "  participant PB as ProjectionBundleBuild",
                "  participant X1 as Rogue\n  participant PB as ProjectionBundleBuild",
            ),
            (
                "  participant PB as ProjectionBundleBuild",
                "  participant rogue as Rogue\n  participant PB as ProjectionBundleBuild",
            ),
        )
        for original, replacement in mutations:
            with self.subTest(mutation=replacement.splitlines()[0]):
                mutated = template.replace(original, replacement, 1)
                self.assertNotEqual(mutated, template)
                errors = validate(mutated)
                self.assertTrue(
                    any(
                        "Task9 Figure 10 exact sequence structure" in error
                        for error in errors
                    ),
                    errors,
                )

    def test_task9_round8_attested_root_flow_call_actuals_are_exact(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        consume_call = (
            "  session := validate_and_consume_measurement_envelope(\n"
            "    authority, runner, trust_root, measured_environment)"
        )
        seal_call = (
            "  return validate_and_seal_attested_conformance(\n"
            "    authority, trust_root, session, observed, ledger, attestation)"
        )
        for original in (consume_call, seal_call):
            self.assertEqual(template.count(original), 1)
            mutated = template.replace(
                original, original.replace("trust_root", "profile.release_trust_root"), 1
            )
            errors = validate(mutated)
            self.assertTrue(
                any("Attested conformance root source-to-sink flow mismatch" in error for error in errors),
                errors,
            )

    def test_task9_round8_terminal_receipt_binds_exact_measurement_envelope(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        receipt_binding = (
            "terminal_consumption_receipt_key(\n"
            "  environment: MeasuredExecutionEnvironment\n"
            ") := canonical_tuple(\n"
            "  environment.verifier_nonce, environment.conformance_invocation_digest,\n"
            "  environment.verifier_runner_digest, environment.executable_artifact_digest,\n"
            "  environment.execution_session_id, environment.process_identity,\n"
            "  environment.container_identity, environment.execution_time_window,\n"
            "  environment.measured_execution_environment_digest)"
        )
        self.assertEqual(template.count(receipt_binding), 1)
        for replacement in (
            receipt_binding.replace(
                "  environment.container_identity, environment.execution_time_window,\n"
                "  environment.measured_execution_environment_digest)",
                "  environment.container_identity, environment.execution_time_window)",
            ),
            receipt_binding.replace(
                "environment.measured_execution_environment_digest",
                "environment.verifier_runner_digest",
                1,
            ),
        ):
            mutated = template.replace(receipt_binding, replacement, 1)
            self.assertNotEqual(mutated, template)
            errors = validate(mutated)
            self.assertTrue(
                any("terminal receipt production/seal/query closure mismatch" in error for error in errors),
                errors,
            )

    def test_task9_round8_release_revalidates_artifact_under_current_conformance_root(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        required = (
            "  conformance_trust_root := profile.conformance_trust_root",
            "  (current_conformance_store, current_conformance_policy) :=\n"
            "    resolve_conformance_trust_root(conformance_trust_root)",
            "  require artifact.invocation_authority.trust_store_snapshot_ref.trust_store_snapshot_digest ==\n"
            "          current_conformance_store.trust_store_snapshot_digest ==\n"
            "          conformance_trust_root.expected_trust_store_snapshot_digest",
            "  require artifact.invocation_authority.runner_attestation_policy_ref.runner_attestation_policy_digest ==\n"
            "          current_conformance_policy.runner_attestation_policy_digest ==\n"
            "          conformance_trust_root.expected_runner_attestation_policy_digest",
            "  current_conformance_trusted_key_material :=\n"
            "    current_conformance_store.trusted_key_material_by_id[\n"
            "      current_conformance_policy.trusted_attestation_key_id]",
            "  current_conformance_signed_message := canonical_tuple(\n"
            "    artifact.runner_attestation.measured_execution_environment_digest,",
            "  require verify_measurer_identity_and_freshness_evidence(\n"
            "    conformance_trust_root.measurement_session_authority,\n"
            "    current_measurement_evidence_message,\n"
            "    artifact.measured_environment.measurer_identity_and_freshness_evidence)",
            "  require conformance_trust_root.measurement_session_authority.has_terminal_consumption_receipt(\n"
            "    terminal_consumption_receipt_key(artifact.measured_environment))",
            "  require verify_signature(current_conformance_trusted_key_material,\n"
            "    current_conformance_policy.attestation_signature_scheme,\n"
            "    current_conformance_signed_message, artifact.runner_attestation.signature)",
            "  reverify sealed runner attestation under current profile root without consuming a measurement session or nonce",
        )
        for token in required:
            self.assertEqual(template.count(token), 1, token)

        mutations = (
            (
                required[0],
                "  conformance_trust_root := artifact.invocation_authority.trust_root",
            ),
            (
                required[2],
                required[2].replace(
                    "conformance_trust_root.expected_trust_store_snapshot_digest",
                    "release_trust_root.expected_trust_store_snapshot_digest",
                ),
            ),
            (
                required[4],
                "  current_conformance_trusted_key_material :=\n"
                "    artifact.runner_attestation.self_owned_key_material",
            ),
            (
                required[9],
                "  consume artifact measurement session and nonce again",
            ),
            (
                required[6],
                required[6].replace(
                    "conformance_trust_root.measurement_session_authority",
                    "release_trust_root.measurement_session_authority",
                ),
            ),
        )
        for original, replacement in mutations:
            with self.subTest(mutation=original.splitlines()[0]):
                mutated = template.replace(original, replacement, 1)
                errors = validate(mutated)
                self.assertTrue(
                    any(
                        marker in error
                        for error in errors
                        for marker in (
                            "release current conformance root revalidation mismatch",
                            "release current conformance closure mismatch",
                        )
                    ),
                    errors,
                )

    def test_task9_round7_release_pre_authentication_block_is_closed(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        anchor = (
            "  require derived_approved == derive_approved_digest_set(artifact, policy)\n"
        )
        self.assertEqual(template.count(anchor), 1)
        for injected_statement in (
            "  require canonical(approved) == canonical(derived_approved)\n",
            "  require approved.subject_digest == derived_approved.subject_digest\n",
            "  approved_alias := derived_approved\n",
        ):
            with self.subTest(statement=injected_statement.strip()):
                mutated = template.replace(
                    anchor, anchor + injected_statement, 1
                )
                errors = validate(mutated)
                self.assertTrue(
                    any(
                        "release approval independent recomputation boundary"
                        in error
                        for error in errors
                    ),
                    errors,
                )

    def test_task9_round7_attested_owner_assignments_are_exact_and_single(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        authority_assignment = (
            "  conformance_invocation_digest :=\n"
            '    hash("attested-conformance-authority/v1",\n'
            "      canonical_payload_without_derived_digests(AttestedConformanceInvocationAuthority))\n"
        )
        self.assertEqual(template.count(authority_assignment), 1)
        authority_mutations = (
            authority_assignment
            + "  conformance_invocation_digest := wrong_authority_digest\n",
            authority_assignment
            + "  unknown_authority_digest := wrong_authority_digest\n",
            (
                "  conformance_invocation_digest := wrong_authority_digest\n"
                "  shadow_invocation_digest :=\n"
                '    hash("attested-conformance-authority/v1",\n'
                "      canonical_payload_without_derived_digests(AttestedConformanceInvocationAuthority))\n"
            ),
        )
        for replacement in authority_mutations:
            with self.subTest(authority=replacement.splitlines()[-1]):
                mutated = template.replace(authority_assignment, replacement, 1)
                errors = validate(mutated)
                self.assertTrue(
                    any(
                        "Attested authority schema/validator domain equality"
                        in error
                        for error in errors
                    ),
                    errors,
                )

        record_assignment = (
            '  evaluation_input_digest := hash("attested-conformance-record-input/v1",\n'
            "    authority.conformance_invocation_digest,\n"
            "    session.measured_execution_environment_digest,\n"
            "    canonical(binding), binding.fixture_binding_digest)\n"
        )
        self.assertEqual(template.count(record_assignment), 1)
        record_mutations = (
            record_assignment
            + "  evaluation_input_digest := wrong_record_input_digest\n",
            record_assignment
            + "  unknown_record_input_digest := wrong_record_input_digest\n",
            (
                "  evaluation_input_digest := wrong_record_input_digest\n"
                '  shadow_evaluation_input_digest := hash("attested-conformance-record-input/v1",\n'
                "    authority.conformance_invocation_digest,\n"
                "    session.measured_execution_environment_digest,\n"
                "    canonical(binding), binding.fixture_binding_digest)\n"
            ),
        )
        for replacement in record_mutations:
            with self.subTest(record=replacement.splitlines()[-1]):
                mutated = template.replace(record_assignment, replacement, 1)
                errors = validate(mutated)
                self.assertTrue(
                    any(
                        "Attested record-input three-site normalized equality"
                        in error
                        for error in errors
                    ),
                    errors,
                )

    def test_task9_round7_basic_run_control_grammar_is_closed(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        record_rule = "    require record.fixture_ref == record_key\n"
        self.assertEqual(template.count(record_rule), 1)
        hidden_rule = template.replace(
            record_rule,
            "    if false:\n      require record.fixture_ref == record_key\n",
            1,
        )
        loop_anchor = (
            "  for every (record_key, binding) in authority.fixture_set.bindings:\n"
        )
        duplicate_loop = template.replace(loop_anchor, loop_anchor + loop_anchor, 1)
        terminal_anchor = (
            "  observed := ConformanceObservedOutput(observed_clause_outputs, findings, coverage_gaps)\n"
        )
        extra_terminal = template.replace(
            terminal_anchor,
            "  return InternalViolation { violation: SyntheticBasicStop }\n"
            + terminal_anchor,
            1,
        )
        for mutated in (hidden_rule, duplicate_loop, extra_terminal):
            errors = validate(mutated)
            self.assertTrue(
                any("conformance basic closed control grammar" in error for error in errors),
                errors,
            )

    def test_task9_round7_release_apply_statement_sequence_is_closed(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        signature_requirement = (
            "  require verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)\n"
        )
        self.assertEqual(template.count(signature_requirement), 1)
        false_after_signature = template.replace(
            signature_requirement,
            signature_requirement + "  require false\n",
            1,
        )
        root_assignment = (
            "  root := resolve_release_approval_trust_root(release_trust_root)\n"
        )
        self.assertEqual(template.count(root_assignment), 1)
        unknown_assignment = template.replace(
            root_assignment,
            root_assignment + "  synthetic_release_alias := approved\n",
            1,
        )
        for mutated in (false_after_signature, unknown_assignment):
            errors = validate(mutated)
            self.assertTrue(
                any("release decision authenticated control-flow" in error for error in errors),
                errors,
            )

    def test_task9_round6_approved_and_derived_sets_remain_independent_until_authenticated_diff(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        independent_closure = (
            "  require canonical(approved) == canonical(approval_authority.approved)\n"
            "  require derived_approved == derive_approved_digest_set(artifact, policy)\n"
            "  require every stored nested digest equals its own enclosing payload recomputation; no approved/derived cross-equality\n"
        )
        authenticated_equality = (
            "  if approved_mismatches is NonEmpty:\n"
            "    return Completed { decision: ReleaseBlocked { subject_digest, reasons:\n"
            "      map(approved_mismatches, path -&gt; ApprovedDigestMismatch(path)) } }\n"
            "  require approved == derived_approved\n"
            "  require all ten ApprovedDigestSet fields are byte-equal only on this authenticated empty-diff path\n"
        )
        self.assertEqual(template.count(independent_closure), 1)
        self.assertEqual(template.count(authenticated_equality), 1)

        early_cross_equality = template.replace(
            independent_closure,
            independent_closure
            + "  require every one of the ten ApprovedDigestSet fields equals the recomputed value above\n",
            1,
        )
        early_errors = validate(early_cross_equality)
        self.assertTrue(
            any("release approval independent recomputation boundary" in error for error in early_errors),
            early_errors,
        )

        missing_late_equality = template.replace(
            authenticated_equality,
            authenticated_equality.replace(
                "  require all ten ApprovedDigestSet fields are byte-equal only on this authenticated empty-diff path\n",
                "",
            ),
            1,
        )
        late_errors = validate(missing_late_equality)
        self.assertTrue(
            any("release approval independent recomputation boundary" in error for error in late_errors),
            late_errors,
        )

    def test_task9_round6_rejects_owner_comments_and_post_algorithm_basic_decoys(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        authority_formula = (
            '    hash("attested-conformance-authority/v1",\n'
            "      canonical_payload_without_derived_digests(AttestedConformanceInvocationAuthority))"
        )
        self.assertEqual(template.count(authority_formula), 1)
        authority_comment_decoy = template.replace(
            authority_formula,
            '    digest("wrong-authority-domain/v1",\n'
            "      canonical_payload_without_derived_digests(AttestedConformanceInvocationAuthority))\n"
            '    // Schema evidence: hash("attested-conformance-authority/v1", '
            "canonical_payload_without_derived_digests(AttestedConformanceInvocationAuthority))",
            1,
        )
        authority_errors = validate(authority_comment_decoy)
        self.assertTrue(
            any("Attested authority schema/validator domain equality" in error for error in authority_errors),
            authority_errors,
        )

        record_formula = (
            '  evaluation_input_digest := hash("attested-conformance-record-input/v1",\n'
            "    authority.conformance_invocation_digest,\n"
            "    session.measured_execution_environment_digest,\n"
            "    canonical(binding), binding.fixture_binding_digest)"
        )
        self.assertEqual(template.count(record_formula), 1)
        record_comment_decoy = template.replace(
            record_formula,
            record_formula.replace(" := hash(", " := digest(").replace(
                '"attested-conformance-record-input/v1"', '"wrong-record-domain/v1"'
            )
            + '\n    // Schema evidence: evaluation_input_digest := hash("attested-conformance-record-input/v1", '
            + "authority.conformance_invocation_digest, session.measured_execution_environment_digest, "
            + "canonical(binding), binding.fixture_binding_digest)",
            1,
        )
        record_errors = validate(record_comment_decoy)
        self.assertTrue(
            any("Attested record-input three-site normalized equality" in error for error in record_errors),
            record_errors,
        )

        basic_rule = "    require record.fixture_ref == record_key\n"
        basic_end = (
            "  return Completed iff every Basic closure equation holds; otherwise InternalViolation</code></pre>"
        )
        self.assertEqual(template.count(basic_rule), 1)
        self.assertEqual(template.count(basic_end), 1)
        post_algorithm_decoy = template.replace(
            basic_rule,
            "    demand record.fixture_ref == record_key\n",
            1,
        ).replace(
            basic_end,
            basic_end + "\n<p><code>require record.fixture_ref == record_key</code></p>",
            1,
        )
        basic_errors = validate(post_algorithm_decoy)
        self.assertTrue(
            any("require record.fixture_ref == record_key" in error and "conformance basic closure" in error for error in basic_errors),
            basic_errors,
        )

    def test_task9_round6_rejects_shared_offline_types_and_extra_release_terminals(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        projection_header = "ProjectionBundleAuthority:\n"
        self.assertEqual(template.count(projection_header), 1)
        for field in (
            "trace_fixture: TraceFixture",
            "observed_output: ConformanceObservedOutput",
            "fixture_set_digest: Digest",
            "validation_policy_ref: ValidationPolicyRef",
        ):
            with self.subTest(field=field):
                mutated = template.replace(
                    projection_header, projection_header + f"  {field}\n", 1
                )
                errors = validate(mutated)
                self.assertTrue(
                    any("production owner conformance isolation" in error for error in errors),
                    errors,
                )

        signature_success = (
            "  verify release approval signature only with external-root public key material\n"
        )
        self.assertEqual(template.count(signature_success), 1)
        extra_terminal = template.replace(
            signature_success,
            signature_success
            + "  return InternalViolation { violation: SyntheticAlwaysStop }\n",
            1,
        )
        terminal_errors = validate(extra_terminal)
        self.assertTrue(
            any("release decision authenticated control-flow" in error for error in terminal_errors),
            terminal_errors,
        )

    def test_task9_round5_release_signature_authenticates_submitted_approved_set(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        schema_tuple = (
            "  signed_message := canonical_tuple(\n"
            "    approved.release_policy_digest,\n"
            "    store_snapshot.release_approval_store_snapshot_digest,\n"
            "    trusted_approver_key_id, approval_signature_scheme, canonical(approved))"
        )
        apply_tuple = (
            "  release_approval_signed_message := canonical_tuple(\n"
            "    root.expected_release_policy_digest,\n"
            "    root.expected_release_approval_store_snapshot_digest,\n"
            "    policy.trusted_approver_key_id, policy.approval_signature_scheme,\n"
            "    canonical(approved))"
        )
        self.assertEqual(template.count(schema_tuple), 1)
        self.assertEqual(template.count(apply_tuple), 1)

        schema_derived = template.replace(
            schema_tuple,
            schema_tuple.replace("canonical(approved)", "canonical(derived_approved)"),
            1,
        )
        schema_errors = validate(schema_derived)
        self.assertTrue(
            any("release approval signed-message approved binding" in error for error in schema_errors),
            schema_errors,
        )

        apply_derived = template.replace(
            apply_tuple,
            apply_tuple.replace("canonical(approved)", "canonical(derived_approved)"),
            1,
        )
        apply_errors = validate(apply_derived)
        self.assertTrue(
            any("release approval signed-message approved binding" in error for error in apply_errors),
            apply_errors,
        )

        signed_before_mismatch = (
            "  require verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)\n"
            "  verify release approval signature only with external-root public key material\n"
            "  if approved_mismatches is NonEmpty:\n"
        )
        self.assertEqual(template.count(signed_before_mismatch), 1)

    def test_task9_round4_rejects_basic_to_release_type_flows_and_seal_schema_swaps(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        type_flows = (
            "adapt(payload: BasicOfflineReportArtifact) -&gt; ReleaseDecision",
            "NeutralBridge := (BasicOfflineReportArtifact) -&gt; AttestedConformanceSealArtifact",
            "translate(payload: BasicOfflineReportArtifact) -&gt; ReleasePolicyApplicationResult",
        )
        for declaration in type_flows:
            with self.subTest(type_flow=declaration):
                mutated = template.replace(
                    "</main>", f"<pre><code>{declaration}</code></pre>\n</main>", 1
                )
                errors = validate(mutated)
                self.assertTrue(
                    any("Basic artifact to release type-flow" in error for error in errors),
                    errors,
                )

        swapped_report = template.replace(
            "  report: AttestedConformanceReport\n",
            "  report: BasicOfflineReportArtifact\n",
            1,
        )
        swapped_errors = validate(swapped_report)
        self.assertTrue(
            any("AttestedConformanceSealArtifact exact schema" in error for error in swapped_errors),
            swapped_errors,
        )

        seal_digest = (
            '    hash("attested-conformance-seal-artifact/v1",\n'
            "      canonical_payload_without_derived_digests(AttestedConformanceSealArtifact))"
        )
        self.assertEqual(template.count(seal_digest), 1)
        post_digest_field = template.replace(
            seal_digest,
            seal_digest + "\n  basic_report: BasicOfflineReportArtifact",
            1,
        )
        post_digest_errors = validate(post_digest_field)
        self.assertTrue(
            any("AttestedConformanceSealArtifact exact schema" in error for error in post_digest_errors),
            post_digest_errors,
        )

    def test_task9_round4_security_variable_sources_and_map_membership_dominate(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        source_mutations = (
            (
                "  (store, policy) := resolve_conformance_trust_root(trust_root)\n",
                "  (store, policy) := resolve_conformance_trust_root(trust_root)\n"
                "  store := authority.fixture_set\n",
            ),
            (
                "  trust_root := profile.conformance_trust_root\n",
                "  trust_root := profile.conformance_trust_root\n"
                "  trust_root := authority.fixture_set\n",
            ),
            (
                "  root := resolve_release_approval_trust_root(release_trust_root)\n",
                "  root := resolve_release_approval_trust_root(release_trust_root)\n"
                "  root := approval_authority.store_snapshot\n",
            ),
            (
                "  session := validate_and_consume_measurement_envelope(\n",
                "  session := authority.fixture_set\n"
                "  session := validate_and_consume_measurement_envelope(\n",
            ),
            (
                "  measured_environment := session.measured_environment\n",
                "  measured_environment := approval_authority.store_snapshot\n"
                "  measured_environment := session.measured_environment\n",
            ),
        )
        for original, replacement in source_mutations:
            with self.subTest(source=original.strip()):
                self.assertGreaterEqual(template.count(original), 1)
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(
                    any("Attested security variable source/dominance" in error for error in errors),
                    errors,
                )

        attestation_membership = (
            "  require policy.trusted_attestation_key_id in store.trusted_key_material_by_id; otherwise InternalViolation\n"
            "  trusted_key_material := store.trusted_key_material_by_id[policy.trusted_attestation_key_id]\n"
        )
        self.assertEqual(template.count(attestation_membership), 1)
        reordered_attestation = template.replace(
            attestation_membership,
            "  trusted_key_material := store.trusted_key_material_by_id[policy.trusted_attestation_key_id]\n"
            "  require policy.trusted_attestation_key_id in store.trusted_key_material_by_id; otherwise InternalViolation\n",
            1,
        )
        attestation_errors = validate(reordered_attestation)
        self.assertTrue(
            any("map lookup membership dominance" in error for error in attestation_errors),
            attestation_errors,
        )

        approval_membership = (
            "  require policy.trusted_approver_key_id in root.trusted_approver_public_key_material_by_id; otherwise InternalViolation\n"
            "  trusted_approver_public_key_material :=\n"
            "    root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]\n"
        )
        self.assertEqual(template.count(approval_membership), 1)
        reordered_approval = template.replace(
            approval_membership,
            "  trusted_approver_public_key_material :=\n"
            "    root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]\n"
            "  require policy.trusted_approver_key_id in root.trusted_approver_public_key_material_by_id; otherwise InternalViolation\n",
            1,
        )
        approval_errors = validate(reordered_approval)
        self.assertTrue(
            any("map lookup membership dominance" in error for error in approval_errors),
            approval_errors,
        )

    def test_task9_round4_release_control_flow_and_production_subject_digest_are_exact(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        missing_branch = (
            "  if subject_digest not in approval_authority.store_snapshot.approved_by_subject:\n"
            "    return Completed { decision: ReleaseBlocked { subject_digest, reasons: {ApprovalMissing} } }\n"
            "  require approval_authority.store_snapshot.approved_by_subject[subject_digest] == approved\n"
        )
        self.assertEqual(template.count(missing_branch), 1)
        lookup_first = template.replace(
            missing_branch,
            "  require approval_authority.store_snapshot.approved_by_subject[subject_digest] == approved\n"
            "  if subject_digest not in approval_authority.store_snapshot.approved_by_subject:\n"
            "    return Completed { decision: ReleaseBlocked { subject_digest, reasons: {ApprovalMissing} } }\n",
            1,
        )
        lookup_errors = validate(lookup_first)
        self.assertTrue(
            any("release decision authenticated control-flow" in error for error in lookup_errors),
            lookup_errors,
        )

        explicit_allow = template.replace(
            "  if subject_digest not in approval_authority.store_snapshot.approved_by_subject:\n",
            "  return Completed { decision: ReleaseAllowed { subject_digest, approved } }\n"
            "  if subject_digest not in approval_authority.store_snapshot.approved_by_subject:\n",
            1,
        )
        explicit_allow_errors = validate(explicit_allow)
        self.assertTrue(
            any("release decision authenticated control-flow" in error for error in explicit_allow_errors),
            explicit_allow_errors,
        )

        missing_to_allow = template.replace(
            "    return Completed { decision: ReleaseBlocked { subject_digest, reasons: {ApprovalMissing} } }\n",
            "    return Completed { decision: ReleaseAllowed { subject_digest, approved } }\n",
            1,
        )
        missing_to_allow_errors = validate(missing_to_allow)
        self.assertTrue(
            any("release decision authenticated control-flow" in error for error in missing_to_allow_errors),
            missing_to_allow_errors,
        )

        versioned_decision = (
            "  decision := apply the versioned pass/fail/insufficient rule after comparing all ten approved digest fields\n"
        )
        self.assertEqual(template.count(versioned_decision), 1)
        direct_allow = template.replace(
            versioned_decision,
            "  decision := ReleaseAllowed { subject_digest, approved }\n"
            + versioned_decision,
            1,
        )
        direct_allow_errors = validate(direct_allow)
        self.assertTrue(
            any("release decision authenticated control-flow" in error for error in direct_allow_errors),
            direct_allow_errors,
        )

        subject_digest = "  subject_digest := hash(all four constituent fields in schema order)\n"
        self.assertEqual(template.count(subject_digest), 1)
        digest_extension = template.replace(
            subject_digest,
            subject_digest + "    + hash(canonical ConformanceDeploymentProfile)\n",
            1,
        )
        digest_errors = validate(digest_extension)
        self.assertTrue(
            any("ProductionSubject subject_digest exact equation" in error for error in digest_errors),
            digest_errors,
        )

    def test_task9_round4_rejects_profile_prose_decoys_and_isolates_all_production_owners(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        authority_schema_formula = (
            'hash("attested-conformance-authority/v1",\n'
            "      canonical_payload_without_derived_digests(AttestedConformanceInvocationAuthority))"
        )
        self.assertEqual(template.count(authority_schema_formula), 1)
        authority_decoy = template.replace(
            authority_schema_formula,
            'digest("wrong-authority-domain/v1",\n'
            "      canonical_payload_without_derived_digests(AttestedConformanceInvocationAuthority))",
            1,
        ).replace(
            "RunnerAttestation:\n",
            "Schema evidence: " + authority_schema_formula.replace("\n", " ") + "\n\n"
            "RunnerAttestation:\n",
            1,
        )
        authority_decoy_errors = validate(authority_decoy)
        self.assertTrue(
            any("Attested authority schema/validator domain equality" in error for error in authority_decoy_errors),
            authority_decoy_errors,
        )

        record_schema_formula = (
            '  evaluation_input_digest := hash("attested-conformance-record-input/v1",\n'
            "    authority.conformance_invocation_digest,\n"
            "    session.measured_execution_environment_digest,\n"
            "    canonical(binding), binding.fixture_binding_digest)"
        )
        self.assertEqual(template.count(record_schema_formula), 1)
        record_decoy = template.replace(
            record_schema_formula,
            record_schema_formula.replace(" := hash(", " := digest(").replace(
                '"attested-conformance-record-input/v1"', '"wrong-record-domain/v1"'
            ),
            1,
        ).replace(
            "derive_approved_digest_set fields:\n",
            "Schema evidence: "
            + record_schema_formula.strip().replace("\n", " ")
            + "\n\nderive_approved_digest_set fields:\n",
            1,
        )
        record_decoy_errors = validate(record_decoy)
        self.assertTrue(
            any("Attested record-input three-site normalized equality" in error for error in record_decoy_errors),
            record_decoy_errors,
        )

        basic_rule = "require record.fixture_ref == record_key"
        basic_tail = (
            "return Completed iff every Basic closure equation holds; otherwise InternalViolation"
        )
        self.assertEqual(template.count(basic_rule), 1)
        basic_decoy = template.replace(basic_rule, "demand record.fixture_ref == record_key", 1).replace(
            basic_tail,
            basic_tail + "\nprose evidence: " + basic_rule,
            1,
        )
        basic_decoy_errors = validate(basic_decoy)
        self.assertTrue(
            any(basic_rule in error and "conformance basic closure" in error for error in basic_decoy_errors),
            basic_decoy_errors,
        )

        for owner, field in (
            ("ProjectionBundleAuthority", "release_profile: ConformanceDeploymentProfile"),
            ("GateManifest", "attested_seal: AttestedConformanceSealArtifact"),
        ):
            with self.subTest(production_owner=owner):
                header = f"{owner}:\n"
                self.assertEqual(template.count(header), 1)
                mutated = template.replace(header, header + f"  {field}\n", 1)
                errors = validate(mutated)
                self.assertTrue(
                    any("production owner conformance isolation" in error for error in errors),
                    errors,
                )

    def test_task9_round3_rejects_export_aliases(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        derive_tail = "  release_policy_digest := recompute policy.release_policy_digest"
        self.assertEqual(template.count(derive_tail), 1)
        exported_alias = template.replace(
            derive_tail,
            derive_tail
            + "\npublic exported alias: seal_attested := "
            + "validate_and_seal_attested_conformance",
            1,
        )
        alias_errors = validate(exported_alias)
        self.assertTrue(
            any("whole conformance exported port inventory" in error for error in alias_errors),
            alias_errors,
        )

    def test_task9_round3_rejects_profile_union_extensions(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        extended_profile = (
            "ExtendedConformanceProfile := "
            "ConformanceDeploymentProfile | UnknownProfile"
        )
        outside_extension = template.replace(
            "</main>", f"<pre><code>{extended_profile}</code></pre>\n</main>", 1
        )
        extension_errors = validate(outside_extension)
        self.assertTrue(
            any("closed conformance profile definition inventory" in error for error in extension_errors),
            extension_errors,
        )

    def test_task9_round2_whole_section_port_union_and_global_owner_closure(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        attested_result = (
            "AttestedConformanceRunResult :=\n"
            "  Completed { artifact: AttestedConformanceSealArtifact }\n"
            "  | InternalViolation { violation: InternalContractViolation }"
        )
        self.assertEqual(template.count(attested_result), 1)
        extra_arm = template.replace(
            attested_result,
            attested_result + "\n  | BasicAccepted { artifact: BasicOfflineReportArtifact }",
            1,
        )
        extra_arm_errors = validate(extra_arm)
        self.assertTrue(
            any("AttestedConformanceRunResult exact two-arm union" in error for error in extra_arm_errors),
            extra_arm_errors,
        )

        release_decision = (
            "ReleaseDecision :=\n"
            "  ReleaseAllowed { subject_digest, approved: ApprovedDigestSet }\n"
            "  | ReleaseBlocked { subject_digest, reasons: NonEmpty&lt;ReleaseBlockReason&gt; }"
        )
        self.assertEqual(template.count(release_decision), 1)
        release_extra_arm = template.replace(
            release_decision,
            release_decision + "\n  | Deferred { subject_digest }",
            1,
        )
        release_extra_errors = validate(release_extra_arm)
        self.assertTrue(
            any("ReleaseDecision exact two-arm union" in error for error in release_extra_errors),
            release_extra_errors,
        )

        basic_tail = (
            "return Completed iff every Basic closure equation holds; "
            "otherwise InternalViolation"
        )
        alias_in_marker = template.replace(
            basic_tail,
            basic_tail
            + "\nupgrade_basic_for_release(payload: Bytes) -&gt; Bytes",
            1,
        )
        alias_in_marker_errors = validate(alias_in_marker)
        self.assertTrue(
            any("whole conformance callable owner inventory" in error for error in alias_in_marker_errors),
            alias_in_marker_errors,
        )

        outside_declarations = (
            (
                "ReleasePolicyApplicationResult :=\n"
                "  Completed { decision: ReleaseDecision }\n"
                "  | InternalViolation { violation: InternalContractViolation }",
                "global conformance owner uniqueness",
            ),
            (
                "upgrade_basic_for_release(payload: Bytes) -&gt; Bytes",
                "conformance declaration outside module",
            ),
            (
                "ExperimentalConformance := BasicOfflineConformance",
                "closed conformance profile declaration set",
            ),
        )
        for declaration, expected_error in outside_declarations:
            with self.subTest(declaration=declaration):
                mutated = template.replace(
                    "</main>", f"<pre><code>{declaration}</code></pre>\n</main>", 1
                )
                errors = validate(mutated)
                self.assertTrue(
                    any(expected_error in error for error in errors), errors
                )

    def test_task9_round2_schema_extent_trust_dataflow_and_missing_approval(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        post_digest_mutations = (
            (
                "  subject_digest := hash(all four constituent fields in schema order)\n",
                "  subject_digest := hash(all four constituent fields in schema order)\n"
                "  conformance_profile: ConformanceDeploymentProfile\n",
                "ProductionSubject exact production schema",
            ),
            (
                "    hash(canonical_payload_without_derived_digests(BackendSealArtifact))\n",
                "    hash(canonical_payload_without_derived_digests(BackendSealArtifact))\n"
                "  basic_report: BasicOfflineReportArtifact\n",
                "BackendSealArtifact<T,V> exact production schema",
            ),
            (
                "           ComparisonResultCandidate))\n",
                "           ComparisonResultCandidate))\n"
                "  conformance_profile: ConformanceDeploymentProfile\n",
                "ComparisonResultCandidate exact production schema",
            ),
            (
                "    hash(canonical_payload_without_derived_digests(ComparisonSealArtifact))\n",
                "    hash(canonical_payload_without_derived_digests(ComparisonSealArtifact))\n"
                "  attested_seal: AttestedConformanceSealArtifact\n",
                "ComparisonSealArtifact exact production schema",
            ),
        )
        for original, replacement, expected_error in post_digest_mutations:
            with self.subTest(schema=expected_error):
                self.assertEqual(template.count(original), 1)
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(
                    any(expected_error in error for error in errors), errors
                )

        additional_production_owners = (
            "EvaluationInstanceIdentity",
            "EstimateCandidate&lt;T&gt;",
            "BackendResultCandidate&lt;T&gt;",
            "BackendResultSourceAuthority&lt;V&gt;",
            "BackendValueGateAuthority&lt;T,V&gt;",
            "BackendSealAuthority&lt;T,V&gt;",
            "ComparisonArmAuthority",
            "ComparisonSourceAuthority",
            "ComparisonSealAuthority",
        )
        for owner in additional_production_owners:
            with self.subTest(production_owner=owner):
                header = f"{owner}:\n"
                self.assertEqual(template.count(header), 1)
                mutated = template.replace(
                    header,
                    header + "  conformance_profile: ConformanceDeploymentProfile\n",
                    1,
                )
                errors = validate(mutated)
                self.assertTrue(
                    any(
                        f"{unescape(owner)} exact production schema" in error
                        for error in errors
                    ),
                    errors,
                )

        authority_fallback = template.replace(
            "  runner_attestation_policy_ref: RunnerAttestationPolicyRef\n",
            "  runner_attestation_policy_ref: RunnerAttestationPolicyRef\n"
            "  fallback_trust_store: TrustStoreSnapshot\n"
            "  fallback_runner_policy: RunnerAttestationPolicy\n",
            1,
        )
        authority_fallback_errors = validate(authority_fallback)
        self.assertTrue(
            any("AttestedConformanceInvocationAuthority exact schema" in error for error in authority_fallback_errors),
            authority_fallback_errors,
        )

        fallback_dataflow = template.replace(
            "  (store, policy) := resolve_conformance_trust_root(trust_root)\n",
            "  (store, policy) := resolve_conformance_trust_root(trust_root)\n"
            "  if store is None: store := authority.fallback_trust_store\n"
            "  if policy is None: policy := authority.fallback_runner_policy\n",
            1,
        )
        fallback_dataflow_errors = validate(fallback_dataflow)
        self.assertTrue(
            any("caller-owned Attested trust fallback" in error for error in fallback_dataflow_errors),
            fallback_dataflow_errors,
        )

        approval_fallback = template.replace(
            "  approved: ApprovedDigestSet\n",
            "  approved: ApprovedDigestSet\n"
            "  fallback_key_material: TrustedPublicKeyMaterial\n",
            1,
        )
        approval_fallback_errors = validate(approval_fallback)
        self.assertTrue(
            any("ReleaseApprovalAuthority exact schema" in error for error in approval_fallback_errors),
            approval_fallback_errors,
        )

        key_assignment = (
            "  trusted_approver_public_key_material :=\n"
            "    root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]\n"
        )
        self.assertEqual(template.count(key_assignment), 1)
        fallback_key = template.replace(
            key_assignment,
            key_assignment
            + "  if trusted_approver_public_key_material is None:\n"
            + "    trusted_approver_public_key_material := approval.fallback_key_material\n",
            1,
        )
        fallback_key_errors = validate(fallback_key)
        self.assertTrue(
            any("caller-owned release approval key fallback" in error for error in fallback_key_errors),
            fallback_key_errors,
        )

        missing_branch = (
            "  if subject_digest not in approval_authority.store_snapshot.approved_by_subject:\n"
            "    return Completed { decision: ReleaseBlocked { subject_digest, reasons: {ApprovalMissing} } }\n"
        )
        self.assertEqual(template.count(missing_branch), 1)
        missing_override = template.replace(
            missing_branch,
            missing_branch
            + "  if subject_digest not in approval_authority.store_snapshot.approved_by_subject:\n"
            + "    return Completed { decision: ReleaseAllowed { subject_digest, approved } }\n",
            1,
        )
        missing_override_errors = validate(missing_override)
        self.assertTrue(
            any("ApprovalMissing unique blocked branch" in error for error in missing_override_errors),
            missing_override_errors,
        )

    def test_task9_round2_release_digest_mismatch_is_reachable_and_chapter_syncs(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")

        mismatch_declaration = (
            "  derived_approved := derive_approved_digest_set(artifact, policy)\n"
            "  approved_mismatches := canonical_schema_path_diff(approved, derived_approved)\n"
        )
        authenticated_branch = (
            "  require verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)\n"
            "  verify release approval signature only with external-root public key material\n"
            "  if approved_mismatches is NonEmpty:\n"
            "    return Completed { decision: ReleaseBlocked { subject_digest, reasons:\n"
            "      map(approved_mismatches, path -&gt; ApprovedDigestMismatch(path)) } }\n"
            "  require approved == derived_approved\n"
            "  require all ten ApprovedDigestSet fields are byte-equal only on this authenticated empty-diff path\n"
            "  decision := apply the versioned pass/fail/insufficient rule after comparing all ten approved digest fields\n"
        )
        self.assertEqual(template.count(mismatch_declaration), 1)
        self.assertEqual(template.count(authenticated_branch), 1)
        removed_branch = template.replace(
            authenticated_branch,
            "  require verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)\n"
            "  verify release approval signature only with external-root public key material\n"
            "  require approved == derived_approved\n"
            "  require all ten ApprovedDigestSet fields are byte-equal only on this authenticated empty-diff path\n"
            "  decision := apply the versioned pass/fail/insufficient rule after comparing all ten approved digest fields\n",
            1,
        )
        removed_errors = validate(removed_branch)
        self.assertTrue(
            any("ApprovedDigestMismatch reachable blocked branch" in error for error in removed_errors),
            removed_errors,
        )

        early_return = template.replace(
            mismatch_declaration,
            mismatch_declaration
            + "  if approved_mismatches is NonEmpty:\n"
            + "    return Completed { decision: ReleaseBlocked { subject_digest, reasons:\n"
            + "      map(approved_mismatches, path -&gt; ApprovedDigestMismatch(path)) } }\n",
            1,
        ).replace(
            authenticated_branch,
            "  require verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)\n"
            "  verify release approval signature only with external-root public key material\n"
            "  require approved == derived_approved\n"
            "  require all ten ApprovedDigestSet fields are byte-equal only on this authenticated empty-diff path\n"
            "  decision := apply the versioned pass/fail/insufficient rule after comparing all ten approved digest fields\n",
            1,
        )
        stale_errors = validate(early_return)
        self.assertTrue(
            any("release decision authenticated control-flow" in error for error in stale_errors),
            stale_errors,
        )

        false_guard = template.replace(
            "  if approved_mismatches is NonEmpty:\n",
            "  if False and approved_mismatches is NonEmpty:\n",
            1,
        ).replace(
            "  decision := apply the versioned pass/fail/insufficient rule after comparing all ten approved digest fields\n",
            "  decision := apply the versioned pass/fail/insufficient rule after comparing all ten approved digest fields\n"
            "  invariant evidence: if approved_mismatches is NonEmpty:\n",
            1,
        )
        false_guard_errors = validate(false_guard)
        self.assertTrue(
            any("release decision authenticated control-flow" in error for error in false_guard_errors),
            false_guard_errors,
        )

        chapter_12_4 = verify_gates._section_between(template, "c12-4", "c12-5")
        self.assertIsNotNone(chapter_12_4)
        self.assertEqual(
            chapter_12_4.count("-&gt; ReleasePolicyApplicationResult"), 1
        )
        stale_chapter = template.replace(
            "  -&gt; ReleasePolicyApplicationResult</code></pre>",
            "  -&gt; ReleaseDecision</code></pre>",
            1,
        )
        stale_chapter_errors = validate(stale_chapter)
        self.assertTrue(
            any("Chapter 12.4 release result wrapper" in error for error in stale_chapter_errors),
            stale_chapter_errors,
        )

    def test_task9_round2_domain_basic_security_and_handoff_inventories(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        domain_mutations = (
            (
                'hash("attested-conformance-authority/v1",\n'
                "      canonical_payload_without_derived_digests(AttestedConformanceInvocationAuthority))",
                'hash("wrong-authority-domain/v1",\n'
                "      canonical_payload_without_derived_digests(AttestedConformanceInvocationAuthority))",
                "Attested authority schema/validator domain equality",
            ),
            (
                'evaluation_input_digest := hash("attested-conformance-record-input/v1",\n'
                "    authority.conformance_invocation_digest,",
                'evaluation_input_digest := hash("wrong-record-domain/v1",\n'
                "    authority.conformance_invocation_digest,",
                "Attested record-input three-site normalized equality",
            ),
        )
        for original, replacement, expected_error in domain_mutations:
            with self.subTest(domain=expected_error):
                self.assertEqual(template.count(original), 1)
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(
                    any(expected_error in error for error in errors), errors
                )

        basic_closure_rules = (
            "require record.fixture_ref == record_key",
            "require record.target_invocation_id == binding.target_invocation_id",
            "require record.target_clause_id == binding.target_clause_id",
            "require record.fixture_digest == binding.fixture.fixture_digest",
            "require record.observed_clause_output == observed_clause_outputs[record_key]",
            "require record.basic_offline_execution_record_digest == hash(\"basic-offline-record/v1\",",
            "require ledger.expected_binding_domain == keys(authority.fixture_set.bindings)",
            "require keys(ledger.execution_records) == ledger.expected_binding_domain",
            "require ledger.basic_offline_authority_digest == authority.basic_offline_authority_digest",
            "require ledger.subject_digest == authority.production_subject.subject_digest",
            "require ledger.fixture_set_digest == authority.fixture_set.fixture_set_digest",
            "require ledger.validation_policy_digest == authority.validation_policy.validation_policy_digest",
            "require ledger.gate_manifest_digest == authority.gate_manifest.gate_manifest_digest",
            "require ledger.verifier_runner_digest == authority.verifier_runner.verifier_runner_digest",
            "require ledger.observed_output == observed",
            "require ledger.basic_offline_execution_ledger_digest == hash(\"basic-offline-ledger/v1\",",
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
            "require report.basic_offline_report_digest == hash(\"basic-offline-report/v1\",",
            "require artifact.authority == authority",
            "require artifact.observed_output == observed",
            "require artifact.execution_ledger == ledger",
            "require artifact.report == report",
            "require artifact.basic_offline_report_artifact_digest == hash(",
        )
        for rule in basic_closure_rules:
            with self.subTest(basic_rule=rule):
                self.assertIn(rule, template)
                errors = validate(template.replace(rule, "REMOVED_BASIC_RULE", 1))
                self.assertTrue(
                    any(rule in error and "conformance basic closure" in error for error in errors),
                    errors,
                )

        rehost_rule = "negative fixture: release approval substituted policy"
        self.assertEqual(template.count(rehost_rule), 1)
        removed = replace_in_module(
            template, "conformance", rehost_rule, "REHOSTED_SECURITY_RULE"
        )
        conformance = module_section_match(removed, "conformance")
        rehosted = (
            removed[: conformance.start()]
            + conformance.group("open")
            + conformance.group("body")
            + f"<p>{rehost_rule}</p>"
            + conformance.group("close")
            + removed[conformance.end() :]
        )
        rehosted_errors = validate(rehosted)
        self.assertTrue(
            any("Attested security rule inventory" in error for error in rehosted_errors),
            rehosted_errors,
        )

        handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
        basic_signature = (
            "run_basic_offline_conformance(\n"
            "  authority: BasicOfflineConformanceAuthority\n"
            ") -> BasicOfflineRunResult"
        )
        self.assertEqual(handoff.count(basic_signature), 1)
        mutated_handoff = handoff.replace(
            basic_signature,
            basic_signature + "\nsession: Bytes",
            1,
        )
        handoff_errors: list[str] = []
        verify_gates.check_handoff_conformance_sync(mutated_handoff, handoff_errors)
        self.assertTrue(
            any("BasicOffline profile contains attested/release capability" in error for error in handoff_errors),
            handoff_errors,
        )

    def test_task9_round1_domain_tags_and_basic_closure_are_explicit(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        local_errors: list[str] = []
        local = verify_gates._conformance_local_texts(template, local_errors)
        self.assertEqual(local_errors, [])
        attested = local[("profile", "attested-release")]
        basic = local[("profile", "basic-offline")]

        authority_formula = (
            'hash("attested-conformance-authority/v1", '
            "canonical_payload_without_derived_digests(authority))"
        )
        record_input_formula = (
            'evaluation_input_digest := hash("attested-conformance-record-input/v1", '
            "authority.conformance_invocation_digest, "
            "session.measured_execution_environment_digest, canonical(binding), "
            "binding.fixture_binding_digest)"
        )
        self.assertEqual(attested.count(authority_formula), 1)
        self.assertEqual(attested.count(record_input_formula), 3)

        basic_closure_tokens = (
            "require record.fixture_ref == record_key",
            "require record.target_invocation_id == binding.target_invocation_id",
            "require record.target_clause_id == binding.target_clause_id",
            "require record.fixture_digest == binding.fixture.fixture_digest",
            'record.evaluation_input_digest == hash("basic-offline-record-input/v1", authority.basic_offline_authority_digest, canonical(binding), binding.fixture_binding_digest)',
            "require ledger.expected_binding_domain == keys(authority.fixture_set.bindings)",
            "require keys(ledger.execution_records) == ledger.expected_binding_domain",
            "require ledger.basic_offline_authority_digest == authority.basic_offline_authority_digest",
            "require ledger.observed_output == observed",
            "require report.production_subject == authority.production_subject",
            "require report.offline_verdict == derived_offline_verdict",
            "require report.findings == observed.findings",
            "require artifact.authority == authority",
            "require artifact.observed_output == observed",
            "require artifact.execution_ledger == ledger",
            "require artifact.report == report",
            "recompute every Basic artifact nested digest before Completed",
            "return Completed iff every Basic closure equation holds; otherwise InternalViolation",
        )
        for token in basic_closure_tokens:
            with self.subTest(token=token):
                self.assertIn(token, basic)

        expected_owner_inventory = {
            ("scope", "shared"): (
                "TraceFixture",
                "FixtureBinding",
                "FixtureSet",
                "ValidationPolicy",
                "VerifierRunner",
                "ConformanceFinding",
                "ObservedClauseOutput",
                "ConformanceObservedOutput",
            ),
            ("scope", "deployment"): ("ConformanceDeploymentProfile",),
            ("profile", "basic-offline"): (
                "BasicOfflineConformanceAuthority",
                "BasicOfflineExecutionRecord",
                "BasicOfflineExecutionLedger",
                "BasicOfflineReport",
                "BasicOfflineReportArtifact",
                "BasicOfflineRunResult",
            ),
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
        self.assertEqual(
            getattr(verify_gates, "CONFORMANCE_OWNER_INVENTORY", {}),
            expected_owner_inventory,
        )

        attested_boundary_tokens = (
            "ReleasePolicyApplicationResult := Completed { decision: ReleaseDecision } | InternalViolation { violation: InternalContractViolation }",
            "apply_release_policy( artifact: AttestedConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy, profile: AttestedReleaseConformance ) -> ReleasePolicyApplicationResult",
            "public exported ports (exact):",
            "internal non-exported ports:",
            "require policy.trusted_attestation_key_id in store.trusted_key_material_by_id; otherwise InternalViolation",
            "if subject_digest not in approval_authority.store_snapshot.approved_by_subject: return Completed { decision: ReleaseBlocked { subject_digest, reasons: {ApprovalMissing} } }",
            "require policy.trusted_approver_key_id in root.trusted_approver_public_key_material_by_id; otherwise InternalViolation",
        )
        for token in attested_boundary_tokens:
            with self.subTest(attested_boundary=token):
                self.assertIn(token, attested)

    def test_task9_round1_rejects_unmarked_ports_extra_result_arms_and_owner_fields(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        basic_publish = template.replace(
            "BasicOfflineRunResult :=\n"
            "  Completed { artifact: BasicOfflineReportArtifact }\n"
            "  | InternalViolation { violation: InternalContractViolation }",
            "BasicOfflineRunResult :=\n"
            "  Completed { artifact: BasicOfflineReportArtifact }\n"
            "  | InternalViolation { violation: InternalContractViolation }\n"
            "  | PublishAuthorized { subject_digest: Digest }",
            1,
        )
        publish_errors = validate(basic_publish)
        self.assertTrue(
            any("BasicOfflineRunResult exact two-arm union" in error for error in publish_errors),
            publish_errors,
        )

        release_result_arm = (
            "ReleasePolicyApplicationResult :=\n"
            "  Completed { decision: ReleaseDecision }\n"
            "  | InternalViolation { violation: InternalContractViolation }"
        )
        self.assertEqual(template.count(release_result_arm), 1)
        release_publish = template.replace(
            release_result_arm,
            release_result_arm + "\n  | PublishAuthorized { subject_digest: Digest }",
            1,
        )
        release_publish_errors = validate(release_publish)
        self.assertTrue(
            any("ReleasePolicyApplicationResult exact two-arm union" in error for error in release_publish_errors),
            release_publish_errors,
        )

        naked_session = template.replace(
            "BasicOfflineConformanceAuthority:\n",
            "BasicOfflineConformanceAuthority:\n  session: Bytes\n",
            1,
        )
        naked_session_errors = validate(naked_session)
        self.assertTrue(
            any("basic-offline block contains attested/release token" in error for error in naked_session_errors),
            naked_session_errors,
        )

        shared_heavy = template.replace(
            "ConformanceObservedOutput:\n",
            "ConformanceObservedOutput:\n"
            "  runner_attestation_ref: RunnerAttestation\n",
            1,
        )
        shared_heavy_errors = validate(shared_heavy)
        self.assertTrue(
            any("shared block references a profile-specific type" in error for error in shared_heavy_errors),
            shared_heavy_errors,
        )

        conformance = module_section_match(template, "conformance")
        unmarked_overload = (
            "<pre><code>apply_release_policy(\n"
            "  artifact: BasicOfflineReportArtifact,\n"
            "  approval: ReleaseApprovalArtifact,\n"
            "  policy: ReleasePolicy,\n"
            "  profile: AttestedReleaseConformance\n"
            ") -&gt; ReleaseDecision</code></pre>"
        )
        unmarked_body = conformance.group("body") + unmarked_overload
        unmarked = (
            template[: conformance.start()]
            + conformance.group("open")
            + unmarked_body
            + conformance.group("close")
            + template[conformance.end() :]
        )
        unmarked_errors = validate(unmarked)
        self.assertTrue(
            any("profile-specific declaration outside local block" in error for error in unmarked_errors),
            unmarked_errors,
        )

        exact_signature = (
            "apply_release_policy(\n"
            "  artifact: AttestedConformanceSealArtifact,\n"
            "  approval: ReleaseApprovalArtifact,\n"
            "  policy: ReleasePolicy,\n"
            "  profile: AttestedReleaseConformance\n"
            ") -&gt; ReleasePolicyApplicationResult"
        )
        duplicate_body = conformance.group("body") + f"<pre><code>{exact_signature}</code></pre>"
        duplicate = (
            template[: conformance.start()]
            + conformance.group("open")
            + duplicate_body
            + conformance.group("close")
            + template[conformance.end() :]
        )
        duplicate_errors = validate(duplicate)
        self.assertTrue(
            any("unique exact Attested apply_release_policy" in error for error in duplicate_errors),
            duplicate_errors,
        )

        public_helper = template.replace(
            "public exported ports (exact):\n",
            "public exported ports (exact):\n"
            "validate_and_seal_attested_conformance(\n"
            "  authority: AttestedConformanceInvocationAuthority\n"
            ") -&gt; AttestedConformanceRunResult\n",
            1,
        )
        public_helper_errors = validate(public_helper)
        self.assertTrue(
            any("exact three public conformance ports" in error for error in public_helper_errors),
            public_helper_errors,
        )

        wrong_apply_result = template.replace(
            ") -&gt; ReleasePolicyApplicationResult\n\ninternal non-exported ports:",
            ") -&gt; ReleaseDecision\n\ninternal non-exported ports:",
            1,
        )
        wrong_apply_errors = validate(wrong_apply_result)
        self.assertTrue(
            any("unique exact Attested apply_release_policy" in error for error in wrong_apply_errors),
            wrong_apply_errors,
        )

        membership_tokens = (
            "require policy.trusted_attestation_key_id in store.trusted_key_material_by_id; otherwise InternalViolation",
            "if subject_digest not in approval_authority.store_snapshot.approved_by_subject:",
            "require policy.trusted_approver_key_id in root.trusted_approver_public_key_material_by_id; otherwise InternalViolation",
        )
        for token in membership_tokens:
            with self.subTest(membership=token):
                self.assertIn(token, template)
                membership_errors = validate(template.replace(token, "MISSING_MEMBERSHIP_CHECK", 1))
                self.assertTrue(
                    any("map lookup membership closure" in error for error in membership_errors),
                    membership_errors,
                )

        attested_rule = (
            "measured_payload_digest == policy.expected_execution_environment_digest"
        )
        self.assertEqual(template.count(attested_rule), 2)
        rule_removed = replace_in_module(
            template,
            "conformance",
            attested_rule,
            "REHOSTED_MEASURED_PAYLOAD_RULE",
        )
        rule_removed = replace_in_module(
            rule_removed,
            "conformance",
            attested_rule,
            "REHOSTED_MEASURED_PAYLOAD_RULE",
        )
        removed_section = module_section_match(rule_removed, "conformance")
        rule_rehosted_body = (
            removed_section.group("body")
            + f"<p>{attested_rule}; {attested_rule}</p>"
        )
        rule_rehosted = (
            rule_removed[: removed_section.start()]
            + removed_section.group("open")
            + rule_rehosted_body
            + removed_section.group("close")
            + rule_removed[removed_section.end() :]
        )
        rule_rehosted_errors = validate(rule_rehosted)
        self.assertTrue(
            any("profile-specific declaration outside local block" in error for error in rule_rehosted_errors),
            rule_rehosted_errors,
        )

        extra_approved_field = template.replace(
            "  release_policy_digest\n\nReleaseApprovalStoreSnapshot:",
            "  release_policy_digest\n  profile_schema_id\n\n"
            "ReleaseApprovalStoreSnapshot:",
            1,
        )
        extra_approved_errors = validate(extra_approved_field)
        self.assertTrue(
            any("ApprovedDigestSet exact ten-field schema" in error for error in extra_approved_errors),
            extra_approved_errors,
        )

        owner_mutations = (
            (
                "ProductionSubject:\n",
                "ProductionSubject:\n  conformance_profile: ConformanceDeploymentProfile\n",
                "ProductionSubject exact production schema",
            ),
            (
                "Estimate&lt;T&gt;:\n",
                "Estimate&lt;T&gt;:\n  basic_report: BasicOfflineReportArtifact\n",
                "Estimate<T> exact production schema",
            ),
            (
                "BackendSealArtifact&lt;T,V&gt;:\n",
                "BackendSealArtifact&lt;T,V&gt;:\n"
                "  conformance_seal: AttestedConformanceSealArtifact\n",
                "BackendSealArtifact<T,V> exact production schema",
            ),
            (
                "ComparisonResult:\n",
                "ComparisonResult:\n  conformance_profile: ConformanceDeploymentProfile\n",
                "ComparisonResult exact production schema",
            ),
            (
                "ComparisonResultCandidate:\n",
                "ComparisonResultCandidate:\n"
                "  basic_report: BasicOfflineReportArtifact\n",
                "ComparisonResultCandidate exact production schema",
            ),
            (
                "ComparisonSealArtifact:\n",
                "ComparisonSealArtifact:\n"
                "  conformance_seal: AttestedConformanceSealArtifact\n",
                "ComparisonSealArtifact exact production schema",
            ),
        )
        for original, replacement, expected_error in owner_mutations:
            with self.subTest(owner=original):
                self.assertEqual(template.count(original), 1)
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(
                    any(expected_error in error for error in errors), errors
                )

    def test_task9_handoff_profiles_and_ports_are_structurally_synchronized(self) -> None:
        handoff = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
        errors: list[str] = []
        verify_gates.check_handoff_conformance_sync(handoff, errors)
        self.assertEqual(errors, [])

        mutations = (
            (
                "  BasicOfflineConformance\n  | AttestedReleaseConformance {",
                "  BasicOfflineConformance\n  | ExperimentalConformance\n  | AttestedReleaseConformance {",
                "exact two arms/default",
            ),
            (
                "default_conformance_deployment_profile := BasicOfflineConformance",
                "default_conformance_deployment_profile := AttestedReleaseConformance",
                "exact two arms/default",
            ),
            (
                "artifact: AttestedConformanceSealArtifact,",
                "artifact: BasicOfflineReportArtifact,",
                "AttestedRelease profile boundary",
            ),
            (
                "`run_basic_offline_conformance(...)`、",
                "",
                "module locator/profile ports",
            ),
        )
        for original, replacement, expected_error in mutations:
            with self.subTest(original=original):
                self.assertIn(original, handoff)
                mutated_errors: list[str] = []
                verify_gates.check_handoff_conformance_sync(
                    handoff.replace(original, replacement, 1), mutated_errors
                )
                self.assertTrue(
                    any(expected_error in error for error in mutated_errors),
                    mutated_errors,
                )

    def test_task9_conformance_profiles_are_structural_and_branch_local(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        mutations = (
            (
                'data-conformance-profile="basic-offline"',
                'data-conformance-profile="unknown"',
                "exact shared/deployment/basic-offline/attested-release",
            ),
            (
                'profile_schema_id := "basic-offline/v1"',
                'profile_schema_id := "attested-release/v1"',
                "basic-offline every digest-owning concrete schema",
            ),
            (
                "BasicOfflineConformanceAuthority:",
                "BasicOfflineConformanceAuthority:\n  signature: Bytes",
                "basic-offline block contains attested/release token",
            ),
            (
                "BasicOfflineConformanceAuthority:",
                "BasicOfflineConformanceAuthority:\n  trust_root: Bytes",
                "basic-offline block contains attested/release token",
            ),
            (
                "BasicOfflineConformanceAuthority:",
                "BasicOfflineConformanceAuthority:\n  execution_session_id: Bytes",
                "basic-offline block contains attested/release token",
            ),
            (
                "| AttestedReleaseConformance {",
                "| Disabled\n  | AttestedReleaseConformance {",
                "exact two arms",
            ),
            (
                "  }\n\ndefault_conformance_deployment_profile := BasicOfflineConformance",
                "  }\n  | ExperimentalConformance\n\n"
                "default_conformance_deployment_profile := BasicOfflineConformance",
                "exact two arms",
            ),
            (
                'data-conformance-scope="shared"',
                'data-conformance-scope="shared" '
                'data-conformance-profile="basic-offline"',
                "exact shared/deployment/basic-offline/attested-release",
            ),
            (
                "apply_release_policy(\n  artifact: AttestedConformanceSealArtifact,\n  approval: ReleaseApprovalArtifact,\n  policy: ReleasePolicy,\n  profile: AttestedReleaseConformance\n) -&gt; ReleasePolicyApplicationResult",
                "apply_release_policy(\n  artifact: BasicOfflineReportArtifact,\n  approval: ReleaseApprovalArtifact,\n  policy: ReleasePolicy,\n  profile: AttestedReleaseConformance\n) -&gt; ReleasePolicyApplicationResult",
                "attested-release block missing contract",
            ),
            (
                "GL--&gt;&gt;BO: clause observations and GateExecutionLedger",
                "GL--&gt;&gt;BO: AttestedConformanceSealArtifact",
                "Task9 Figure 10 exact mutually-exclusive profile ownership",
            ),
            (
                "    else AttestedReleaseConformance\n      AR-&gt;&gt;GL:",
                "    else BasicOfflineConformance\n      AR-&gt;&gt;GL:",
                "Task9 Figure 10 exact mutually-exclusive profile ownership",
            ),
            (
                "  release_policy_digest\n\nReleaseApprovalStoreSnapshot:",
                "  release_policy_digest\n  profile_digest\n\n"
                "ReleaseApprovalStoreSnapshot:",
                "ApprovedDigestSet exact ten-field schema",
            ),
        )
        for original, replacement, expected_error in mutations:
            with self.subTest(original=original):
                self.assertIn(original, template)
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(
                    any(expected_error in error for error in errors), errors
                )

        rehost_token = (
            "negative fixture: replay valid measurement envelope from another "
            "execution session"
        )
        self.assertEqual(template.count(rehost_token), 1)
        removed = replace_in_module(
            template, "conformance", rehost_token, "REHOSTED_ATTESTED_RULE"
        )
        conformance = module_section_match(removed, "conformance")
        rehosted_body = (
            conformance.group("body")
            + f"<p>{rehost_token}</p>"
        )
        rehosted = (
            removed[: conformance.start()]
            + conformance.group("open")
            + rehosted_body
            + conformance.group("close")
            + removed[conformance.end() :]
        )
        rehosted_errors = validate(rehosted)
        self.assertTrue(
            any("attested-release block missing contract" in error for error in rehosted_errors),
            rehosted_errors,
        )

    def test_final_review_contracts_are_section_local_and_mutation_sensitive(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        cases = (
            (
                "input-facts",
                "RequestedBackendInput :=",
            ),
            (
                "input-facts",
                "requested_backends == keys tagged MemoryRequested or TimeRequested",
            ),
            (
                "time-backend",
                "occupied_streams(node: BoundComputeEvent | BoundCommunication)",
            ),
            (
                "time-backend",
                "occupied_streams(node)) for node in order",
            ),
        )

        for module, token in cases:
            with self.subTest(module=module, token=token):
                self.assertIn(
                    token,
                    verify_gates.MODULE_CONTRACT_REQUIRED_TEXT[module],
                    "section-local verifier does not yet own the reviewed contract",
                )
                baseline_errors = validate(template)
                self.assertEqual(baseline_errors, [])
                errors = validate(replace_in_module(template, module, token, "MUTATED"))
                self.assertTrue(
                    any(module in error and token in error for error in errors),
                    errors,
                )

    def test_task7_core_modeling_contracts_reject_local_regressions(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        mutations = (
            (
                "coverage numerator type",
                "op_occurrences_modeled: OpOccurrenceCount",
                "op_occurrences_modeled: ByteCount",
                "Coverage field/type mismatch",
            ),
            (
                "calibration embeds manifest payload",
                "CalibrationSet:\n  active_compute_records: Map[MeasurementKey, ComputeMeasurementRecord]\n  measurement_protocols: Map[ProtocolDigest, MeasurementProtocol]\n  calibration_train_manifest_digest: Digest",
                "CalibrationSet:\n  active_compute_records: Map[MeasurementKey, ComputeMeasurementRecord]\n  measurement_protocols: Map[ProtocolDigest, MeasurementProtocol]\n  calibration_train_manifest: CalibrationTrainManifest\n  calibration_train_manifest_digest: Digest",
                "CalibrationSet 必须只持",
            ),
            (
                "train manifest authority field",
                "      calibration_train_manifest_snapshot: CalibrationTrainManifest",
                "      calibration_train_manifest_snapshot: Digest",
                "TimeRequested 未携带独立 train manifest",
            ),
            (
                "request manifest closure",
                "calibration := requested_backend_inputs[time].calibration_set",
                "calibration := requested_backend_inputs[time].unvalidated_calibration_set",
                "RequestSnapshot train manifest authority closure",
            ),
            (
                "train manifest moved to common inputs",
                "comparison_schema_snapshot: ComparisonSchemaSnapshot",
                "comparison_schema_snapshot: ComparisonSchemaSnapshot\n  calibration_train_manifest_snapshot: CalibrationTrainManifest",
                "CommonProductionInputs exact schema",
            ),
            (
                "active record closure",
                "keys(calibration.active_compute_records) ==\n            train_manifest.active_measurement_keys\n    require keys(calibration.measurement_protocols) ==\n            train_manifest.active_protocol_digests\n    require every record",
                "keys(calibration.active_compute_records) !=\n            train_manifest.active_measurement_keys\n    require keys(calibration.measurement_protocols) ==\n            train_manifest.active_protocol_digests\n    require every record",
                "time authority 未闭合",
            ),
            (
                "hardware field",
                "devices: DeviceCatalog",
                "devices: DeviceCatalog\n  memory_capacity: ByteCount",
                "HardwareProfile production schema",
            ),
            (
                "comparison branch",
                "return Unavailable",
                "return ComparableDelta",
                "comparison outcome branch priority/order",
            ),
            (
                "comparison world boundary",
                "  logical_rank_id_set: OrderedSet[LogicalRank]\n",
                "  world_size_only: Count\n",
                "logical_rank_id_set world boundary",
            ),
            (
                "coverage conservation",
                "source_obligations_total = source_obligations_planned + source_obligations_residual + source_obligations_proven_not_executed",
                "source_obligations_total = source_obligations_planned - source_obligations_residual + source_obligations_proven_not_executed",
                "Coverage obligation conservation",
            ),
            (
                "memory reads time arm",
                "construct only from memory_inputs.memory_registry_snapshot",
                "construct only from memory_inputs.memory_registry_snapshot and request.requested_backend_inputs[time]",
                "memory face authority",
            ),
            (
                "time reads memory arm",
                "construct only from calibration, train_manifest, communication_model and time_policy",
                "construct only from calibration, train_manifest, communication_model, time_policy and memory_inputs.memory_registry_snapshot",
                "time face authority",
            ),
            (
                "backend-local blocker scope",
                "each construction blocker\n          contains candidate.backend in affected_backends",
                "each construction blocker\n          excludes candidate.backend from affected_backends",
                "shared failure scope",
            ),
            (
                "shared runtime scope",
                "every BlockerRecord that prevents RuntimeBuildResult or\n          CoreBuildResult from being Ready has affected_backends=={memory,time}",
                "every BlockerRecord that prevents RuntimeBuildResult or\n          CoreBuildResult from being Ready has affected_backends=={time}",
                "shared failure scope",
            ),
            (
                "shared G-IR scope",
                "every InputBlocker occurrence from a shared structure/G-IR invocation\n          has affected_backends=={memory,time}",
                "every InputBlocker occurrence from a shared structure/G-IR invocation\n          has affected_backends=={memory}",
                "shared failure scope",
            ),
            (
                "holdout in request payload",
                "      calibration_train_manifest_snapshot: CalibrationTrainManifest",
                "      calibration_train_manifest_snapshot: CalibrationTrainManifest\n"
                "      holdout_manifest: HoldoutEvaluationManifest",
                "RequestedBackendInput exact union",
            ),
            (
                "holdout in time digest",
                "  projection_witness_digest: Digest\n",
                "  projection_witness_digest: Digest\n  holdout_manifest_digest: Digest\n",
                "TimeSimulationInputDomain exact schema",
            ),
            (
                "holdout in comparison basis",
                "  logical_rank_id_set: OrderedSet[LogicalRank]\n",
                "  logical_rank_id_set: OrderedSet[LogicalRank]\n  holdout_manifest: HoldoutEvaluationManifest\n",
                "ComparisonBasis exact schema",
            ),
            (
                "holdout in cache key",
                "<tr data-cache-id=\"time-result\"><td>Time result</td><td>(time_simulation_digest, time_result_schema_version)</td>",
                "<tr data-cache-id=\"time-result\"><td>Time result</td><td>(time_simulation_digest, time_result_schema_version, HoldoutEvaluationManifest)</td>",
                "production cache exact schema",
            ),
            (
                "non-goal id",
                'data-non-goal-id="NG-AUTOMATIC-SEARCH"',
                'data-non-goal-id="NG-AUTOMATIC-SEARCH-MUTATED"',
                "十个唯一稳定 non-goal ID",
            ),
            (
                "decision ref",
                'data-non-goal-ref="NG-ALLOCATOR-RESERVED"',
                'data-non-goal-ref="NG-CAPACITY"',
                "决策 ref",
            ),
            (
                "decision id",
                'data-decision-id="D18"',
                'data-decision-id="D17"',
                "唯一稳定 decision ID",
            ),
        )
        for label, original, replacement, expected_error in mutations:
            with self.subTest(label=label):
                self.assertEqual(template.count(original), 1, original)
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(
                    any(expected_error in error for error in errors), errors
                )

    def test_task7_production_domains_reject_holdout_alias_fields_structurally(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        mutations = (
            (
                "common holdout alias",
                "  comparison_schema_snapshot: ComparisonSchemaSnapshot\n",
                "  comparison_schema_snapshot: ComparisonSchemaSnapshot\n"
                "  holdout_manifest: Digest\n",
                "CommonProductionInputs exact schema",
            ),
            (
                "common holdout digest alias",
                "  comparison_schema_snapshot: ComparisonSchemaSnapshot\n",
                "  comparison_schema_snapshot: ComparisonSchemaSnapshot\n"
                "  holdout_manifest_digest: Digest\n",
                "CommonProductionInputs exact schema",
            ),
            (
                "memory request extra payload",
                "    memory_registry_snapshot: MemoryRegistrySnapshot }",
                "    memory_registry_snapshot: MemoryRegistrySnapshot\n"
                "    evaluation_manifest_digest: Digest }",
                "RequestedBackendInput exact union",
            ),
            (
                "not-requested payload",
                "  | NotRequested\n\nRequestedBackendInputMap :=",
                "  | NotRequested { evaluation_manifest_digest: Digest }\n\n"
                "RequestedBackendInputMap :=",
                "RequestedBackendInput exact union",
            ),
            (
                "comparison alias",
                "  logical_rank_id_set: OrderedSet[LogicalRank]\n",
                "  logical_rank_id_set: OrderedSet[LogicalRank]\n"
                "  evaluation_manifest_digest: Digest\n",
                "ComparisonBasis exact schema",
            ),
            (
                "time digest alias",
                "  projection_witness_digest: Digest\n",
                "  projection_witness_digest: Digest\n"
                "  evaluation_manifest_digest: Digest\n",
                "TimeSimulationInputDomain exact schema",
            ),
            (
                "time digest bypasses domain",
                "hash(canonical(TimeSimulationInputDomain))",
                "hash(canonical(TimeSimulationInputDomain, evaluation_manifest_digest))",
                "time_simulation_digest 必须只哈希 exact input domain",
            ),
            (
                "cache alias",
                "(time_simulation_digest, time_result_schema_version)",
                "(time_simulation_digest, time_result_schema_version, evaluation_manifest_digest)",
                "production cache exact schema",
            ),
        )
        for label, original, replacement, expected_error in mutations:
            with self.subTest(label=label):
                self.assertEqual(template.count(original), 1, original)
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(
                    any(expected_error in error for error in errors), errors
                )

    def test_task7_request_snapshot_rejects_direct_payload_aliases_structurally(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        anchor = (
            "  comparison_request: ComparisonRequest | None\n"
            "  request_digest: Digest"
        )
        for alias in (
            "holdout_manifest_digest: Digest",
            "evaluation_manifest_digest: Digest",
        ):
            with self.subTest(alias=alias):
                self.assertEqual(template.count(anchor), 1)
                mutated = template.replace(
                    anchor,
                    "  comparison_request: ComparisonRequest | None\n"
                    f"  {alias}\n"
                    "  request_digest: Digest",
                    1,
                )
                errors = validate(mutated)
                self.assertTrue(
                    any("RequestSnapshot exact schema" in error for error in errors),
                    errors,
                )

    def test_task9_conformance_cannot_enter_any_production_identity_surface(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        mutations = (
            (
                "common profile field",
                "  comparison_schema_snapshot: ComparisonSchemaSnapshot\n",
                "  comparison_schema_snapshot: ComparisonSchemaSnapshot\n"
                "  conformance_profile: ConformanceDeploymentProfile\n",
                "CommonProductionInputs exact schema",
            ),
            (
                "request report artifact",
                "  comparison_request: ComparisonRequest | None\n  request_digest: Digest",
                "  comparison_request: ComparisonRequest | None\n"
                "  basic_report_artifact: BasicOfflineReportArtifact\n"
                "  request_digest: Digest",
                "RequestSnapshot exact schema",
            ),
            (
                "requested arm seal artifact",
                "    memory_registry_snapshot: MemoryRegistrySnapshot }",
                "    memory_registry_snapshot: MemoryRegistrySnapshot\n"
                "    conformance_seal: AttestedConformanceSealArtifact }",
                "RequestedBackendInput exact union",
            ),
            (
                "comparison basis profile",
                "  logical_rank_id_set: OrderedSet[LogicalRank]\n",
                "  logical_rank_id_set: OrderedSet[LogicalRank]\n"
                "  conformance_profile: ConformanceDeploymentProfile\n",
                "ComparisonBasis exact schema",
            ),
            (
                "time input report",
                "  projection_witness_digest: Digest\n",
                "  projection_witness_digest: Digest\n"
                "  basic_report_artifact: BasicOfflineReportArtifact\n",
                "TimeSimulationInputDomain exact schema",
            ),
            (
                "model digest profile",
                "canonical_payload_without_derived_digests(CodeIR))",
                "canonical_payload_without_derived_digests(CodeIR), "
                "ConformanceDeploymentProfile)",
                "production digest exact input closure mismatch: model_digest",
            ),
            (
                "runtime digest report",
                "canonical_payload_without_derived_digests(RuntimeEventPlan))",
                "canonical_payload_without_derived_digests(RuntimeEventPlan), "
                "BasicOfflineReportArtifact)",
                "production digest exact input closure mismatch: runtime_plan_digest",
            ),
            (
                "core digest seal",
                "canonical_payload_without_derived_digests(SimulationPlanCore))",
                "canonical_payload_without_derived_digests(SimulationPlanCore), "
                "AttestedConformanceSealArtifact)",
                "production digest exact input closure mismatch: simulation_core_digest",
            ),
            (
                "memory digest profile",
                "memory backend and numeric semantic versions)",
                "memory backend and numeric semantic versions, "
                "ConformanceDeploymentProfile)",
                "production digest exact input closure mismatch: memory_simulation_digest",
            ),
            (
                "time digest report",
                "hash(canonical(TimeSimulationInputDomain))",
                "hash(canonical(TimeSimulationInputDomain), BasicOfflineReportArtifact)",
                "production digest exact input closure mismatch: time_simulation_digest",
            ),
            (
                "result digest seal",
                "hash(canonical_payload_without_derived_digests(Estimate))",
                "hash(canonical_payload_without_derived_digests(Estimate), "
                "AttestedConformanceSealArtifact)",
                "production digest exact input closure mismatch: result_digest",
            ),
            (
                "cache profile",
                "(memory_simulation_digest, memory_result_schema_version)",
                "(memory_simulation_digest, memory_result_schema_version, "
                "ConformanceDeploymentProfile)",
                "production cache exact schema",
            ),
            (
                "backend result report arm",
                "  | NotRequested\n\nSemanticValue&lt;T&gt;:",
                "  | NotRequested\n  | ConformanceReport { artifact: "
                "BasicOfflineReportArtifact }\n\nSemanticValue&lt;T&gt;:",
                "BackendResult exact three-arm production union",
            ),
        )
        for label, original, replacement, expected_error in mutations:
            with self.subTest(label=label):
                self.assertEqual(template.count(original), 1, original)
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(
                    any(expected_error in error for error in errors), errors
                )

        digest_formula = "hash(canonical_payload_without_derived_digests(RequestSnapshot))"
        self.assertEqual(template.count(digest_formula), 1)
        errors = validate(
            template.replace(
                digest_formula,
                "hash(canonical(RequestSnapshot.comparison_request))",
                1,
            )
        )
        self.assertTrue(
            any("request_digest 必须覆盖 exact RequestSnapshot schema" in error for error in errors),
            errors,
        )

    def test_task8_architecture_and_dependency_mutations_are_rejected(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        mutations = (
            (
                'data-diagram-id="layered-module-architecture"',
                'data-diagram-id="production-pipeline"',
                "diagram ID/order",
            ),
            ("block-beta", "flowchart TD", "architecture diagram kind mismatch"),
            (
                'inputFacts["input-facts"]',
                'inputFacts["input-facts / RequestSnapshot"]',
                "exact node/label set mismatch",
            ),
            (
                'data-module-id="plan-projection" data-layer="L2" data-allowed-dependencies="gate-system,memory-backend,time-backend"',
                'data-module-id="plan-projection" data-layer="L2" data-allowed-dependencies="comparison,gate-system,memory-backend,time-backend"',
                "exact six-cell module table",
            ),
            (
                '<tr data-module-id="conformance"',
                '<tr data-module-id="unknown-module"',
                "exact six-cell module table",
            ),
        )
        for original, replacement, expected in mutations:
            with self.subTest(expected=expected):
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(any(expected in error for error in errors), errors)

        mutated = replace_in_module(
            template,
            "plan-projection",
            "exactly one candidate per backend/evaluation identity",
            "REHOSTED_OUTSIDE_CONTRACT",
        )
        errors = validate(mutated + "\nexactly one candidate per backend/evaluation identity")
        self.assertTrue(any("plan-projection" in error for error in errors), errors)

    def test_task8_plan_ports_generics_and_dependency_dag_mutations_are_rejected(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        mutations = (
            (
                "CoreBuildResult<SimulationPlanCore>",
                "CoreBuildResult<MemoryEventView>",
                "authoritative contract references",
            ),
            (
                "policy: HardwareBindingPolicySnapshot",
                "binding_policy: HardwareBindingPolicySnapshot",
                "plan-projection typed port",
            ),
            (
                "expected_invocation_domain(",
                "missing_gate_domain_port(",
                "plan-projection typed port",
            ),
        )
        for original, replacement, expected in mutations:
            with self.subTest(expected=expected):
                mutated = replace_in_module(
                    template, "plan-projection", original, replacement
                )
                self.assertNotEqual(mutated, template)
                errors = validate(mutated)
                self.assertTrue(any(expected in error for error in errors), errors)

        cycle_anchor = (
            'data-module-id="plan-projection" data-layer="L2" '
            'data-allowed-dependencies="gate-system,memory-backend,time-backend"'
        )
        self.assertEqual(template.count(cycle_anchor), 1)
        errors = validate(
            template.replace(
                cycle_anchor,
                'data-module-id="plan-projection" data-layer="L2" '
                'data-allowed-dependencies="gate-system,memory-backend,result-sealing,time-backend"',
                1,
            )
        )
        self.assertTrue(any("dependency DAG" in error for error in errors), errors)

    def test_task8_edge_reference_scope_and_helper_mutations_are_rejected(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])

        mutations = (
            (
                "memoryBackend --&gt; planProjection",
                "planProjection --&gt; memoryBackend",
                "solid edges must reverse table dependencies",
            ),
            (
                "inputFacts -.-&gt; codeIr",
                "inputFacts --&gt; codeIr",
                "solid edges must reverse table dependencies",
            ),
            (
                "runtimeEvents -.-&gt; planProjection",
                "runtimeEvents -.-&gt; gateSystem",
                "dotted DTO lineage",
            ),
            (
                "expected_projection_candidate(request, evaluation_identity, backend,\n"
                "                              requested_branch) :=",
                "finalize_projection_candidate(request, evaluation_identity, backend,\n"
                "                              requested_branch) :=",
                "pure expected_projection_candidate equation",
            ),
            (
                "缺共享 compute shape 以 BLK-MISSING-SHAPE 同时阻断两侧",
                "shape/storage/lifetime 语义只阻断内存侧",
                "memory blocker scope",
            ),
            (
                "expected_projection_candidate(request, evaluation_identity, memory,\n"
                "                                  requested_branch)",
                "expected_projection_candidate(request, evaluation_identity, time,\n"
                "                                  requested_branch)",
                "candidate builder postcondition binding",
            ),
            (
                "memory_projection_candidate = expected_projection_candidate(\n"
                "          request_snapshot, evaluation_identity, memory,",
                "memory_projection_candidate = expected_projection_candidate(\n"
                "          request_snapshot, evaluation_identity, time,",
                "product orchestration candidate binding",
            ),
        )
        for original, replacement, expected in mutations:
            with self.subTest(expected=expected):
                self.assertEqual(template.count(original), 1, original)
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(any(expected in error for error in errors), errors)

        for module in ("memory-backend", "time-backend"):
            with self.subTest(module=module):
                mutated = replace_in_module(
                    template,
                    module,
                    "when time is not requested:"
                    if module == "time-backend"
                    else "run_memory_backend(",
                    "candidate := expected_projection_candidate(...)\n"
                    + (
                        "when time is not requested:"
                        if module == "time-backend"
                        else "run_memory_backend("
                    ),
                )
                errors = validate(mutated)
                self.assertTrue(
                    any("backend runtime-calls plan-owned helper" in error for error in errors),
                    errors,
                )

        wrapper_anchor = '<h4 id="mc-plan-projection-data">核心数据结构</h4>'
        self.assertEqual(template.count(wrapper_anchor), 1)
        errors = validate(
            template.replace(
                wrapper_anchor,
                wrapper_anchor + "\n<pre><code>PlanProjectionOwnedContracts:</code></pre>",
                1,
            )
        )
        self.assertTrue(
            any("forbids local wrapper schema" in error for error in errors), errors
        )

    def test_task8_projection_bundle_core_branch_mutations_are_rejected(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        self.assertEqual(validate(template), [])
        mutations = (
            (
                "    Blocked { blockers=blockers }:\n",
                "    REMOVED_BLOCKED_BRANCH:\n",
                "core-result candidate branches",
            ),
            (
                "authority.memory_candidate is byte-equal to\n"
                "              expected_projection_candidate(\n"
                "                authority.request_snapshot, authority.evaluation_identity, memory,",
                "authority.memory_candidate is byte-equal to\n"
                "              build_memory_projection_candidate(\n"
                "                authority.request_snapshot, authority.evaluation_identity, memory,",
                "core-result candidate branches",
            ),
            (
                "authority.memory_candidate is byte-equal to\n"
                "              expected_projection_candidate(\n"
                "                authority.request_snapshot, authority.evaluation_identity, memory,",
                "authority.memory_candidate is byte-equal to\n"
                "              expected_projection_candidate(\n"
                "                authority.request_snapshot, authority.evaluation_identity, time,",
                "core-result candidate branches",
            ),
            (
                "None if memory not in\n"
                "                  authority.request_snapshot.requested_backends else\n"
                "                  CandidateBlocked { construction_blockers=blockers })",
                "None if memory not in\n"
                "                  authority.request_snapshot.requested_backends else\n"
                "                  CandidateBlocked { construction_blockers=EMPTY })",
                "core-result candidate branches",
            ),
            (
                "require authority.core_result == Blocked { blockers=bs }",
                "require authority.core_result == Ready { core=bs }",
                "runtime-blocked core propagation",
            ),
        )
        for original, replacement, expected in mutations:
            with self.subTest(expected=expected, replacement=replacement):
                self.assertEqual(template.count(original), 1, original)
                errors = validate(template.replace(original, replacement, 1))
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_module_fixture_locator_is_attribute_order_independent(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        original_tag = (
            '<section class="module-contract" data-module="memory-backend" '
            'aria-labelledby="mc-memory-backend">'
        )
        reordered_tag = (
            '<section aria-labelledby="mc-memory-backend" '
            'data-module="memory-backend" class="module-contract">'
        )
        self.assertIn(original_tag, template)
        reordered = template.replace(original_tag, reordered_tag, 1)

        try:
            rehosted = rehost_from_module(
                reordered, "memory-backend", "run_memory_backend("
            )
            replaced = replace_in_module(
                reordered,
                "memory-backend",
                "run_memory_backend(",
                "run_memory_backend_MUTATED(",
            )
        except AssertionError as error:
            self.fail(f"attribute-order-independent locator failed: {error}")

        self.assertTrue(any("memory-backend" in error for error in validate(rehosted)))
        self.assertTrue(any("memory-backend" in error for error in validate(replaced)))

    def test_module_fixture_missing_token_is_a_hard_failure(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "missing fixture token"):
            rehost_from_module(template, "memory-backend", "NOT_A_REAL_FIXTURE")
        with self.assertRaisesRegex(AssertionError, "missing fixture token"):
            replace_in_module(
                template,
                "memory-backend",
                "NOT_A_REAL_FIXTURE",
                "replacement",
            )
        self.assertNotIn("rehost_if_present", globals())

    def test_module_contract_tokens_cannot_be_rehosted_globally(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        cases = (
            ("input-facts", "build_request_snapshot("),
            ("input-facts", "RequestSnapshotBuildResult :="),
            ("code-ir", "evaluate_source("),
            ("code-ir", "RankCodeIRBuildResult :="),
            ("runtime-events", "expand_runtime_semantics("),
            ("memory-backend", "build_memory_projection_candidate("),
            ("memory-backend", "run_memory_backend("),
            ("time-backend", "build_time_projection_candidate("),
            ("time-backend", "run_time_backend("),
            ("result-sealing", "BackendExecution&lt;T, W&gt; :="),
            ("result-sealing", "run_backend_build_candidate_and_seal("),
        )

        for module, needle in cases:
            with self.subTest(module=module, needle=needle):
                errors = validate(rehost_from_module(template, module, needle))
                self.assertTrue(
                    any(
                        module in error and unescape(needle) in error
                        for error in errors
                    ),
                    errors,
                )

    def test_module_contract_subsections_must_be_section_local(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        heading = '<h4 id="mc-code-ir-invariants">不变量</h4>'
        rewritten = rehost_from_module(template, "code-ir", heading)

        errors = validate(rewritten)

        self.assertTrue(
            any("code-ir" in error and "不变量" in error for error in errors),
            errors,
        )

    def test_task4_module_subsections_must_be_section_local(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        cases = (
            ("memory-backend", '<h4 id="mc-memory-backend-invariants">不变量</h4>'),
            ("time-backend", '<h4 id="mc-time-backend-invariants">不变量</h4>'),
            ("result-sealing", '<h4 id="mc-result-sealing-invariants">不变量</h4>'),
        )

        for module, heading in cases:
            with self.subTest(module=module):
                errors = validate(rehost_from_module(template, module, heading))
                self.assertTrue(
                    any(module in error and "不变量" in error for error in errors),
                    errors,
                )

    def test_task4_witness_and_binding_equations_reject_local_mutation(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        cases = (
            (
                "memory-backend",
                "view.memory_projection_semantics_digest == memory_projection_semantics_snapshot.memory_projection_semantics_snapshot_digest",
                "view.memory_projection_semantics_digest == supplied_digest",
            ),
            (
                "memory-backend",
                "witness.memory_execution_input_digest == hash(canonical(view))",
                "witness.memory_execution_input_digest == supplied_digest",
            ),
            (
                "memory-backend",
                "witness.per_rank_replay_order[r] == map(entry.position, timeline[r])",
                "witness.per_rank_replay_order[r] == arbitrary_order",
            ),
            (
                "memory-backend",
                "witness.per_rank_timeline_digest[r] == hash(canonical timeline[r])",
                "witness.per_rank_timeline_digest[r] == supplied_digest",
            ),
            (
                "memory-backend",
                "witness.per_rank_terminal_live_storage_ids[r] == last(timeline[r]).live_storage_instance_ids",
                "witness.per_rank_terminal_live_storage_ids[r] == supplied_ids",
            ),
            (
                "memory-backend",
                "require run_memory_backend(view) == expected_memory_execution(view)",
                "allow run_memory_backend(view) != expected_memory_execution(view)",
            ),
            (
                "time-backend",
                "witness.critical_predecessor_by_node == critical_predecessor_by_node",
                "witness.critical_predecessor_by_node == supplied_predecessors",
            ),
            (
                "time-backend",
                "witness.timeline_digest == hash(canonical timeline)",
                "witness.timeline_digest == supplied_digest",
            ),
            (
                "time-backend",
                "witness.aggregate_interval_inputs_digest == hash(canonical aggregate_interval_inputs)",
                "witness.aggregate_interval_inputs_digest == supplied_digest",
            ),
            (
                "time-backend",
                "node.common.endpoint_descriptors == [node.send_endpoint, node.recv_endpoint]",
                "node.common.endpoint_descriptors == [node.recv_endpoint, node.send_endpoint]",
            ),
            (
                "time-backend",
                "route.endpoint_descriptors == node.common.endpoint_descriptors",
                "route.endpoint_descriptors may differ from node.common.endpoint_descriptors",
            ),
            (
                "time-backend",
                "require run_time_backend(view) == expected_time_execution(view)",
                "allow run_time_backend(view) != expected_time_execution(view)",
            ),
            (
                "result-sealing",
                "EstimateOf<MemoryEventView> := MemoryEstimate",
                "EstimateOf<MemoryEventView> := StepTimeEstimate",
            ),
            (
                "result-sealing",
                "run_backend_build_candidate_and_seal( source: BackendResultSourceAuthority<V> ) -> BackendSealBuildResult<EstimateOf<V>, V>",
                "run_backend_build_candidate_and_seal( source: BackendResultSourceAuthority<V> ) -> BackendSealBuildResult<StepTimeEstimate, V>",
            ),
            (
                "result-sealing",
                "| InternalViolation { violation: InternalContractViolation }",
                "| InternalViolation { violations: NonEmpty<InternalContractViolation> }",
            ),
            (
                "result-sealing",
                "flatten every inputs[i].violations",
                "use only first(inputs).violations",
            ),
        )

        for module, original, replacement in cases:
            with self.subTest(module=module, original=original):
                self.assertIn(
                    original,
                    verify_gates.MODULE_CONTRACT_REQUIRED_TEXT[module],
                )
                errors = validate(
                    replace_in_module(template, module, original, replacement)
                )
                self.assertTrue(
                    any(module in error for error in errors),
                    errors,
                )

    def test_task3_module_contracts_are_required(self) -> None:
        required_contracts = (
            'data-module="input-facts"',
            'data-module="code-ir"',
            'data-module="runtime-events"',
            "StructureRegistrySnapshot",
            "RuntimeRegistrySnapshot",
            "CommonProductionInputs",
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
            "MemoryTimelineEntry:",
            "MemoryExecutionWitness:",
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
        )

        for required_contract in required_contracts:
            with self.subTest(required_contract=required_contract):
                self.assertIn(required_contract, verify_gates.REQUIRED_TEXT)

    def test_missing_codeir_contract_is_rejected(self) -> None:
        html = p1_document().replace("CodeIR(P)", "CodeIR")

        errors = validate(html)

        self.assertTrue(any("CodeIR(P)" in error for error in errors), errors)

    def test_pi0_and_rewrite_architectures_are_rejected(self) -> None:
        stale_contracts = (
            "CoreIR = PE(Source, ModelSpec, EnvFacts, π₀)",
            "策略单位元 π₀",
            "Pass_impl_select",
            "实现选择与融合",
            "折叠到一个 kernel",
            "replacement 映射",
            "只增不改",
            "Gk --rewrite--> Gk+1",
            "GraphVersion",
        )

        for stale_contract in stale_contracts:
            with self.subTest(stale_contract=stale_contract):
                errors = validate(p1_document("\n" + stale_contract))
                self.assertTrue(
                    any(stale_contract in error for error in errors),
                    errors,
                )

    def test_trace_as_prediction_dependency_is_rejected(self) -> None:
        invalid_trace_contracts = (
            "TraceFixture 作为生产输入",
            "trace 作为预测输入",
            "trace 进入 model_digest",
            "trace 进入缓存键",
            "TraceFixture → SimulationPlan",
        )

        for invalid_contract in invalid_trace_contracts:
            with self.subTest(invalid_contract=invalid_contract):
                errors = validate(p1_document("\n" + invalid_contract))
                self.assertTrue(
                    any(invalid_contract in error for error in errors),
                    errors,
                )

    def test_v41_resource_contention_contracts_are_rejected(self) -> None:
        stale_contracts = (
            "目标方案设计 v4.1",
            "resource_ready   := earliest_capacity",
            "capacity-constrained execution target",
            "Bind(tensor_id, storage_instance_id",
            "StorageInstanceId := (StorageId, logical_rank, materialization_epoch)",
            "从 direct measurement 回退到 calibrated model",
        )

        for stale_contract in stale_contracts:
            with self.subTest(stale_contract=stale_contract):
                errors = validate(p1_document("\n" + stale_contract))
                self.assertTrue(
                    any(stale_contract in error for error in errors),
                    errors,
                )

    def test_v42_partial_result_and_exact_cost_regressions_are_rejected(self) -> None:
        stale_contracts = (
            "exact_or_declared_in_domain_user_measurement",
            "域内插值",
            "下一个 logical join",
            "next logical join",
            "bound_compute_events",
            "SimulationPlan/result</td><td>simulation_digest",
        )

        for stale_contract in stale_contracts:
            with self.subTest(stale_contract=stale_contract):
                errors = validate(p1_document("\n" + stale_contract))
                self.assertTrue(
                    any(stale_contract in error for error in errors),
                    errors,
                )

    def test_v42_projection_identity_regressions_are_rejected(self) -> None:
        stale_contracts = (
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
        )

        for stale_contract in stale_contracts:
            with self.subTest(stale_contract=stale_contract):
                errors = validate(p1_document("\n" + stale_contract))
                self.assertTrue(
                    any(stale_contract in error for error in errors),
                    errors,
                )

    def test_v42_projection_closure_regressions_are_rejected(self) -> None:
        required_contracts = (
            "ResolvedEventSemantic",
            "TensorStorageBinding",
            "StorageBinding",
            "BoundWorkspaceLifetime",
            "CommunicationEndpointDescriptor",
            "ExpectedTimeProjection",
            "timeline_numeric_policy",
            "NumericRangeWitness",
            "unbounded_integer_preflight",
            "EstimateContext",
            "EstimateCandidate",
            "BackendResultCandidate",
            "BackendSealAuthority",
            "RequestSnapshot",
            "CanonicalConfigEvaluationInput",
            "EvaluationInstanceIdentity",
            "evaluation_instance_digest",
            "ProductionSubject",
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
            "BLK-NUMERIC-RANGE",
            "kernel_variant_binding_digest",
            "RuntimeBuildResult",
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
        )
        for required_contract in required_contracts:
            with self.subTest(required_contract=required_contract):
                html = p1_document().replace(required_contract, "REMOVED")
                errors = validate(html)
                self.assertTrue(
                    any(required_contract in error for error in errors),
                    errors,
                )

        stale_contracts = (
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
        )
        for stale_contract in stale_contracts:
            with self.subTest(stale_contract=stale_contract):
                errors = validate(p1_document("\n" + stale_contract))
                self.assertTrue(
                    any(stale_contract in error for error in errors),
                    errors,
                )


    def test_task5_module_contracts_are_required_section_locally(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        cases = {
            "gate-system": (
                "BlockerScopePolicySnapshot:",
                "ScopePolicyRef := (blocker_scope_policy_digest, rule_id)",
                "GateSpecificationSet: schema_version blocker_scope_policy: BlockerScopePolicySnapshot",
                "compile_gate_manifest( context: ProductionValidationContext, specifications: GateSpecificationSet, runner: GateRunnerSnapshot ) -> GateManifest",
                "run_gate_domain( authority: GateEvaluationAuthority, base: GateExecutionLedger, invocation_domain: OrderedSet<GateInvocationId> ) -> GateExecutionLedger | InternalContractViolation",
                "extend_gate_ledger_without_overwrite( authority: GateEvaluationAuthority, base: GateExecutionLedger, extension: GateExecutionRecordSet, current_stage_contexts: StageContextMap ) -> GateExecutionLedger | InternalContractViolation",
                "canonical_internal_violation_union(gate_internal_violations)",
                "clause_record.clause_id == clause_key == clause.clause_id",
                "canonical union of every InputBlocker failure occurrence",
                "InternalViolation takes priority",
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
            ),
            "comparison": (
                "CanonicalSchemaPathSet:",
                "ComparisonSchemaSnapshot:",
                "comparison_schema_snapshot_digest: Digest",
                "declared_paths: OrderedSet<CanonicalSchemaPath>",
                "derivation_closure == derive_closure(snapshot, declared_paths)",
                "BasisMismatch:",
                "CoverageDelta:",
                "derive_comparison_basis_pair( request: RequestSnapshot, left: ComparisonArmAuthority, right: ComparisonArmAuthority, metric: memory | time ) -> ComparisonBasisPair",
                "compare_per_metric_from_authority( source: ComparisonSourceAuthority ) -> ComparisonResult",
                "same RequestSnapshot",
                "UndefinedZeroBaseline",
                "request.canonical_production_evaluation_inputs.common_inputs.comparison_schema_snapshot",
                "derive_comparison_basis_pair recomputes",
                "G-REP2 recomputes",
                "negative fixture: inject other_config_indexed_production_inputs.x into derivation closure",
                "BasisMismatch.schema_path == other_config_indexed_production_inputs.x",
            ),
            "conformance": (
                "ConformanceFinding:",
                "ApprovedDigestSet:",
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
                "outside production digests and cache keys",
                "does not alter BackendResult",
                "FixtureBinding:",
                "bindings: OrderedMap<FixtureRef, FixtureBinding>",
                "FixtureSet is the sole fixture-to-(invocation, clause) mapping",
                "FixtureSet: bindings: OrderedMap<FixtureRef, FixtureBinding> derived_coverage_by_clause := derive_fixture_coverage(bindings)",
                "RunnerAttestation:",
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
                "trust_store_snapshot_ref: TrustStoreSnapshotRef",
                "runner_attestation_policy_ref: RunnerAttestationPolicyRef",
                "conformance_invocation_digest :=",
                'evaluation_input_digest := hash("attested-conformance-record-input/v1", authority.conformance_invocation_digest, session.measured_execution_environment_digest, canonical(binding), binding.fixture_binding_digest) observed_clause_output: ObservedClauseOutput',
                "ConformanceObservedOutput:",
                "ConformanceObservedOutput: observed_clause_outputs: OrderedMap<FixtureRef, ObservedClauseOutput> findings: OrderedMap<FindingId, ConformanceFinding> coverage_gaps: OrderedSet<GateClauseId> observed_output_digest := hash(canonical observed_clause_outputs, findings, coverage_gaps)",
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
                "validate_and_seal_attested_conformance( authority: AttestedConformanceInvocationAuthority, trust_root: ConformanceTrustRootCapability, session: ValidatedMeasurementSessionCapability, observed: ConformanceObservedOutput, ledger: AttestedConformanceExecutionLedger, attestation: RunnerAttestation ) -> AttestedConformanceRunResult",
                "validate_and_seal_attested_conformance first operation:",
                "recompute every authority.fixture_set binding, nested fixture and fixture_set_digest",
                "recompute authority.gate_manifest and every nested entry/clause digest",
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
                "require approval_authority.approval_signature_scheme == policy.approval_signature_scheme",
                "policy.approval_signature_scheme in root.supported_release_approval_signature_schemes",
                "release_approval_signed_message := canonical_tuple(",
                "trusted_approver_public_key_material := root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]",
                "verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)",
                "negative fixture: release approval wrong signature scheme",
                "negative fixture: unsupported release approval signature scheme",
                "ReleaseApprovalStoreSnapshot:",
                "ReleaseApprovalAuthority:",
                "ReleaseApprovalArtifact:",
                "verify release approval signature",
                "negative fixture: no execution but forged all-pass",
                "negative fixture: forged runner attestation",
                "negative fixture: forged ApprovedDigestSet",
                "negative fixture: tampered execution record or observed output",
            ),
        }

        for module, required_tokens in cases.items():
            with self.subTest(module=module):
                self.assertIn(module, verify_gates.MODULE_CONTRACT_REQUIRED_TEXT)
                for token in required_tokens:
                    with self.subTest(token=token):
                        self.assertIn(
                            token,
                            verify_gates.MODULE_CONTRACT_REQUIRED_TEXT[module],
                        )
                        errors = validate(
                            replace_in_module(
                                template,
                                module,
                                token,
                                "REMOVED_FROM_MODULE",
                            )
                        )
                        self.assertTrue(
                            any(module in error and token in error for error in errors),
                            errors,
                        )

    def test_input_contract_requires_request_owned_gate_policy_and_specification(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        for token in (
            "blocker_scope_policy_snapshot: BlockerScopePolicySnapshot",
            "gate_specification_set_snapshot: GateSpecificationSet",
            "gate_specification_set_ref: GateSpecificationSetRef",
            "gate_runner_snapshot: GateRunnerSnapshot",
            "build_request_snapshot gate specification equations:",
            "request_runner := common.gate_runner_snapshot",
            "require request_specifications.blocker_scope_policy == request_policy",
            "negative fixture: E+P2 request specification embeds different blocker policy",
        ):
            with self.subTest(token=token):
                self.assertIn(
                    token, verify_gates.MODULE_CONTRACT_REQUIRED_TEXT["input-facts"]
                )
                errors = validate(
                    replace_in_module(
                        template, "input-facts", token, "REMOVED_FROM_MODULE"
                    )
                )
                self.assertTrue(
                    any("input-facts" in error and token in error for error in errors),
                    errors,
                )

    def test_module_contract_domain_is_exactly_the_nine_declared_modules(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        extra = """
<section class="module-contract" data-module="unexpected-tenth" aria-labelledby="mc-unexpected">
<h3 id="mc-unexpected">unexpected</h3>
<h4>职责边界</h4><h4>核心数据结构</h4><h4>接口定义</h4>
<h4>成功与阻断语义</h4><h4>不变量</h4>
</section>
"""

        errors = validate(template + extra)

        self.assertTrue(
            any("unexpected module-contract" in error for error in errors),
            errors,
        )

    def test_gate_contract_rejects_noncanonical_aggregate_and_prefix_reexecution(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        for forbidden in (
            "GateExecutionLedger | NonEmpty<InternalContractViolation>",
            "expected_invocation_domain(authority).audit_invocations",
            "audit_stage_sibling_dependency_closure(manifest.entries, root_stages)",
            "every derived stage context := gate_evaluation_context_digest(subject)",
            "builder validates only manifest, request-owned policy, typed subject and runner",
            "manifest entries may differ from specifications entries",
            "trust manifest gate_specification_set_digest without request payload",
        ):
            with self.subTest(forbidden=forbidden):
                mutated = replace_in_module(
                    template,
                    "gate-system",
                    "本模块把",
                    f"{forbidden} 本模块把",
                )
                errors = validate(mutated)
                self.assertTrue(
                    any("gate-system" in error and forbidden in error for error in errors),
                    errors,
                )

        for required in (
            "projection_gate_subject := ProjectionBundleGateSubject { authority }",
            "for expected_stage in [structure_prerequisite, memory_view_prerequisite, time_view_prerequisite]:",
            "run_backend_value_gate_stage(value_authority):",
            "result_seal_gate_subject := exact MemoryResultSealGateSubject or",
            "comparison_gate_subject := ComparisonGateSubject { authority: seal_authority }",
        ):
            with self.subTest(required_product_stage=required):
                self.assertIn(required, verify_gates.PRODUCT_GATE_STAGING_REQUIRED_TEXT)
                self.assertIn(required, template)
                errors = validate(template.replace(required, "REMOVED_STAGE_CALL", 1))
                self.assertTrue(
                    any("product gate staging" in error and required in error for error in errors),
                    errors,
                )

    def test_task5_rejects_production_fixture_identifiers_section_locally(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        for module, forbidden in (
            ("gate-system", "positive_fixture_refs"),
            ("conformance", "negative_boundary_fixture_refs"),
        ):
            with self.subTest(module=module, forbidden=forbidden):
                mutated = replace_in_module(
                    template,
                    module,
                    "本模块",
                    f"{forbidden} 本模块",
                )
                errors = validate(mutated)
                self.assertTrue(
                    any(module in error and forbidden in error for error in errors),
                    errors,
                )

    def test_conformance_contract_rejects_unsealed_report_interfaces(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        for forbidden in (
            "validate_and_seal_attested_conformance( authority: AttestedConformanceInvocationAuthority, observed: ConformanceObservedOutput, ledger: AttestedConformanceExecutionLedger, report: AttestedConformanceReport",
            "apply_release_policy( report: AttestedConformanceReport",
            "run_attested_release_conformance( subject: ProductionSubject",
            "apply_release_policy( artifact: AttestedConformanceSealArtifact, approved: ApprovedDigestSet",
            "verify_signature(runner.trusted_attestation_key_id, runner.attestation_signature_scheme",
            "VerifierRunner: runner_id, runner_version, executable_artifact_digest trusted_attestation_key_id, attestation_signature_scheme",
        ):
            with self.subTest(forbidden=forbidden):
                mutated = replace_in_module(
                    template,
                    "conformance",
                    "本模块只",
                    f"{forbidden} 本模块只",
                )
                errors = validate(mutated)
                self.assertTrue(
                    any("conformance" in error and forbidden in error for error in errors),
                    errors,
                )

        report_findings = re.compile(
            r"(?s)(AttestedConformanceReport:\s*.*?verdict:\s*pass\s*\|\s*fail\s*\|\s*insufficient\s*)"
            r"findings:\s*OrderedMap&lt;FindingId,\s*ConformanceFinding&gt;"
        )
        self.assertIsNotNone(report_findings.search(template))
        mutated = report_findings.sub(
            r"\1findings: [ConformanceFinding]", template, count=1
        )
        errors = validate(mutated)
        self.assertTrue(
            any("AttestedConformanceReport.findings" in error for error in errors), errors
        )

        authority_ref = "  trust_store_snapshot_ref: TrustStoreSnapshotRef"
        self.assertIn(authority_ref, template)
        mutated = template.replace(
            authority_ref,
            authority_ref + "\n  trust_store_snapshot: TrustStoreSnapshot",
            1,
        )
        errors = validate(mutated)
        self.assertTrue(
            any("external-root refs" in error for error in errors), errors
        )

        measured_signing = (
            "attestation := runner signs canonical_tuple(\n"
            "    measurement_envelope_digest,"
        )
        self.assertIn(measured_signing, template)
        mutated = template.replace(
            measured_signing,
            "attestation := runner signs canonical_tuple(\n"
            "    policy.expected_execution_environment_digest,",
            1,
        )
        errors = validate(mutated)
        self.assertTrue(
            any("protected measured environment" in error for error in errors), errors
        )

        nested_recompute = (
            "  recompute every authority.fixture_set binding, nested fixture and fixture_set_digest\n"
        )
        after_first_operation = (
            "validate_and_seal_attested_conformance equations after first operation:"
        )
        self.assertIn(nested_recompute, template)
        mutated = template.replace(nested_recompute, "", 1).replace(
            after_first_operation,
            after_first_operation + "\n" + nested_recompute.rstrip(),
            1,
        )
        errors = validate(mutated)
        self.assertTrue(
            any("first operation missing nested recomputation" in error for error in errors),
            errors,
        )

    def test_conformance_measurement_session_envelope_is_section_local(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        required_tokens = (
            "opaque protected_environment_measurer-issued non-replayable measurement envelope",
            "execution_session_id: ExecutionSessionId",
            "process_identity: ProcessIdentity",
            "container_identity: ContainerIdentity | BareProcess",
            "verifier_nonce: SingleUseVerifierNonce",
            "execution_time_window: ClosedInterval<MonotonicTimestamp>",
            "measurer_identity_and_freshness_evidence",
            "measured_execution_environment_digest := hash(canonical_payload_without_derived_digests(MeasuredExecutionEnvironment))",
            "ValidatedMeasurementSessionCapability:",
            "validate_and_consume_measurement_envelope( authority: AttestedConformanceInvocationAuthority, runner: VerifierRunner, trust_root: ConformanceTrustRootCapability, measured_environment: MeasuredExecutionEnvironment ) -> ValidatedMeasurementSessionCapability | InternalContractViolation",
            "measurement_envelope_digest == hash(canonical_payload_without_derived_digests(measured_environment))",
            "measured_environment.conformance_invocation_digest == authority.conformance_invocation_digest",
            "measured_environment.verifier_runner_digest == runner.verifier_runner_digest",
            "measured_environment.executable_artifact_digest == runner.executable_artifact_digest",
            "current_execution_session_context() == canonical_tuple(measured_environment.execution_session_id, measured_environment.process_identity, measured_environment.container_identity)",
            "terminal_consumption_receipt_key( environment: MeasuredExecutionEnvironment ) := canonical_tuple( environment.verifier_nonce, environment.conformance_invocation_digest, environment.verifier_runner_digest, environment.executable_artifact_digest, environment.execution_session_id, environment.process_identity, environment.container_identity, environment.execution_time_window, environment.measured_execution_environment_digest)",
            "atomically consumed exactly once while persisting a terminal receipt under that exact key",
            "session.nonce_consumption_receipt proves the terminal registry contains exact terminal_consumption_receipt_key(measured_environment)",
            "verify_measurer_identity_and_freshness_evidence(",
            "measured_payload_digest == policy.expected_execution_environment_digest",
            "execute the exact binding once under session.capability",
            "runner signs canonical_tuple( measurement_envelope_digest",
            "negative fixture: replay valid measurement envelope from another execution session",
            "negative fixture: change only measurer identity or freshness evidence while retaining old envelope digest",
            "change only measurer_identity_and_freshness_evidence changes measured_execution_environment_digest",
        )
        forbidden_tokens = (
            "measured_execution_environment_digest := hash(canonical measured_environment_payload)",
            "signed_message := canonical_tuple( execution_environment_digest",
            "attestation.execution_environment_digest == policy.expected_execution_environment_digest",
        )

        for token in required_tokens:
            with self.subTest(required=token):
                self.assertIn(
                    token, verify_gates.MODULE_CONTRACT_REQUIRED_TEXT["conformance"]
                )
                errors = validate(
                    replace_in_module(
                        template, "conformance", token, "REMOVED_FROM_MODULE"
                    )
                )
                self.assertTrue(
                    any("conformance" in error and token in error for error in errors),
                    errors,
                )

        for token in forbidden_tokens:
            with self.subTest(forbidden=token):
                self.assertIn(
                    token, verify_gates.MODULE_CONTRACT_FORBIDDEN_TEXT["conformance"]
                )
                mutated = replace_in_module(
                    template,
                    "conformance",
                    "本模块只",
                    f"{token} 本模块只",
                )
                errors = validate(mutated)
                self.assertTrue(
                    any("conformance" in error and token in error for error in errors),
                    errors,
                )

    def test_release_approval_external_trust_root_is_section_local_and_mutation_sensitive(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        required_tokens = (
            "ReleaseApprovalTrustRootCapability: opaque deployment/verifier-owned capability",
            "expected_release_policy: ReleasePolicy",
            "expected_release_approval_store_snapshot: ReleaseApprovalStoreSnapshot",
            "supported_release_approval_signature_schemes: OrderedSet<SignatureScheme>",
            "trusted_approver_public_key_material_by_id: OrderedMap<ApproverKeyId, TrustedPublicKeyMaterial>",
            "cannot be constructed by request deserialization, fixture, policy, approval store or approval artifact",
            "apply_release_policy( artifact: AttestedConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy, profile: AttestedReleaseConformance ) -> ReleasePolicyApplicationResult",
            "release_trust_root := profile.release_trust_root",
            "apply_release_policy first operation:",
            "recompute artifact and every nested invocation/report/ledger/attestation/environment digest",
            "recompute approval_authority.store_snapshot and every nested ApprovedDigestSet value",
            "before resolving either profile trust root key material, verifying any signature or reading verdict",
            "root := resolve_release_approval_trust_root(release_trust_root)",
            "root.expected_release_policy_digest == policy.release_policy_digest",
            "canonical(root.expected_release_policy) == canonical(policy)",
            "root.expected_release_approval_store_snapshot_digest == approval_authority.store_snapshot.release_approval_store_snapshot_digest",
            "canonical(root.expected_release_approval_store_snapshot) == canonical(approval_authority.store_snapshot)",
            "policy.approval_signature_scheme in root.supported_release_approval_signature_schemes",
            "trusted_approver_public_key_material := root.trusted_approver_public_key_material_by_id[policy.trusted_approver_key_id]",
            "release_approval_signed_message := canonical_tuple(",
            "verify_signature(trusted_approver_public_key_material, policy.approval_signature_scheme, release_approval_signed_message, approval.signature)",
            "negative fixture: release approval self-owned approver key",
            "negative fixture: release approval wrong store snapshot",
            "negative fixture: release approval unsupported signature scheme",
            "negative fixture: release approval substituted policy",
        )
        forbidden_tokens = (
            "apply_release_policy( artifact: AttestedConformanceSealArtifact, approval: ReleaseApprovalArtifact, policy: ReleasePolicy ) -> ReleaseDecision",
            "verify_signature(policy.trusted_approver_key_id",
            "policy.approval_signature_scheme in SUPPORTED_RELEASE_APPROVAL_SIGNATURE_SCHEMES",
        )

        for token in required_tokens:
            with self.subTest(required=token):
                self.assertIn(
                    token, verify_gates.MODULE_CONTRACT_REQUIRED_TEXT["conformance"]
                )
                errors = validate(
                    replace_in_module(
                        template, "conformance", token, "REMOVED_FROM_MODULE"
                    )
                )
                self.assertTrue(
                    any("conformance" in error and token in error for error in errors),
                    errors,
                )

        for token in forbidden_tokens:
            with self.subTest(forbidden=token):
                self.assertIn(
                    token, verify_gates.MODULE_CONTRACT_FORBIDDEN_TEXT["conformance"]
                )
                mutated = replace_in_module(
                    template,
                    "conformance",
                    "本模块只",
                    f"{token} 本模块只",
                )
                errors = validate(mutated)
                self.assertTrue(
                    any("conformance" in error and token in error for error in errors),
                    errors,
                )

    def test_global_verifier_rejects_noncanonical_internal_violation_sets(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        for forbidden in (
            "InternalContractViolation set",
            "InternalContractViolation(CV-COMPARISON-CONTRACT set)",
            "return InternalContractViolation set",
            "produce InternalContractViolation set",
            "return InternalContractViolation(CV-REPORT-CONTRACT set)",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertIn(forbidden, verify_gates.FORBIDDEN_TEXT)
                errors = validate(template + "\n" + forbidden)
                self.assertTrue(
                    any(forbidden in error for error in errors),
                    errors,
                )
        for legacy_release_subset in (
            "六个 approved digest",
            "四个 digest",
        ):
            with self.subTest(legacy_release_subset=legacy_release_subset):
                errors = validate(template + "\n" + legacy_release_subset)
                self.assertTrue(
                    any(
                        "derive_approved_digest_set" in error
                        and legacy_release_subset in error
                        for error in errors
                    ),
                    errors,
                )

    def test_global_verifier_rejects_legacy_context_and_bad_digest_exclusions(self) -> None:
        template = (ROOT / "src" / "index.template.html").read_text(encoding="utf-8")
        for forbidden in (
            "evaluation_subject_digest(subject: GateEvaluationSubject) :=",
            "result_seal_subject_digest :=",
            "seal_context_digest :=",
            "comparison_context_digest :=",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertIn(forbidden, verify_gates.FORBIDDEN_TEXT)
                errors = validate(template + "\n" + forbidden)
                self.assertTrue(
                    any(forbidden in error for error in errors),
                    errors,
                )

        task5_exclusions = {
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
            "BasicOfflineConformanceAuthority": "basic_offline_authority_digest",
            "BasicOfflineExecutionRecord": "basic_offline_execution_record_digest",
            "BasicOfflineExecutionLedger": "basic_offline_execution_ledger_digest",
            "BasicOfflineReport": "basic_offline_report_digest",
            "BasicOfflineReportArtifact": "basic_offline_report_artifact_digest",
            "TrustStoreSnapshot": "trust_store_snapshot_digest",
            "RunnerAttestationPolicy": "runner_attestation_policy_digest",
            "MeasuredExecutionEnvironment": "measured_execution_environment_digest",
            "AttestedConformanceInvocationAuthority": "conformance_invocation_digest",
            "RunnerAttestation": "runner_attestation_digest",
            "AttestedConformanceExecutionRecord": "conformance_execution_record_digest",
            "ConformanceObservedOutput": "observed_output_digest",
            "AttestedConformanceExecutionLedger": "conformance_execution_ledger_digest",
            "ReleaseApprovalStoreSnapshot": "release_approval_store_snapshot_digest",
            "ReleaseApprovalAuthority": "release_approval_authority_digest",
            "ReleaseApprovalArtifact": "release_approval_artifact_digest",
            "AttestedConformanceSealArtifact": "conformance_seal_artifact_digest",
            "AttestedConformanceReport": "conformance_report_digest",
            "ReleasePolicy": "release_policy_digest",
            "Estimate": "result_digest",
        }
        self.assertEqual(
            dict(verify_gates.DERIVED_DIGEST_EXCLUSIONS), task5_exclusions
        )
        for wildcard in ("*Binding", "*Snapshot / *Policy"):
            with self.subTest(wildcard=wildcard):
                mutated = template.replace(
                    "all nested input digests not named above remain in the payload",
                    f"  {wildcard} -&gt; {{ wildcard_digest }}\n\n"
                    "all nested input digests not named above remain in the payload",
                    1,
                )
                errors = validate(mutated)
                self.assertTrue(
                    any("wildcard owner" in error and wildcard in error for error in errors),
                    errors,
                )
        for owner_type, own_field in task5_exclusions.items():
            pattern = re.compile(
                rf"(?m)^\s*{re.escape(owner_type)}\s*-&gt;\s*\{{\s*{re.escape(own_field)}\s*\}}\s*$"
            )
            with self.subTest(owner_type=owner_type, mutation="missing"):
                matches = list(pattern.finditer(template))
                self.assertEqual(len(matches), 1, f"{owner_type}: {len(matches)}")
                mutated = pattern.sub("", template, count=1)
                errors = validate(mutated)
                self.assertTrue(
                    any(owner_type in error and own_field in error for error in errors),
                    errors,
                )
            with self.subTest(owner_type=owner_type, mutation="self-reference"):
                mutated = pattern.sub(
                    f"  {owner_type} -> {{ nested_input_digest }}", template, count=1
                )
                errors = validate(mutated)
                self.assertTrue(
                    any(owner_type in error and own_field in error for error in errors),
                    errors,
                )
            with self.subTest(owner_type=owner_type, mutation="overbroad"):
                mutated = pattern.sub(
                    f"  {owner_type} -> {{ {own_field}, nested_input_digest }}",
                    template,
                    count=1,
                )
                errors = validate(mutated)
                self.assertTrue(
                    any(owner_type in error and own_field in error for error in errors),
                    errors,
                )
            with self.subTest(owner_type=owner_type, mutation="duplicate-overbroad"):
                mutated = pattern.sub(
                    f"  {owner_type} -&gt; {{ {own_field} }}\n"
                    f"  {owner_type} -&gt; {{ nested_input_digest }}",
                    template,
                    count=1,
                )
                errors = validate(mutated)
                self.assertTrue(
                    any(owner_type in error and own_field in error for error in errors),
                    errors,
                )


class VerifyGatesConsoleTest(unittest.TestCase):
    def test_verifier_succeeds_on_gbk_console(self) -> None:
        """Adding a non-GBK status glyph must break this test."""
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "gbk"
        completed = subprocess.run(
            [sys.executable, str(VERIFY_GATES)],
            cwd=ROOT,
            env=env,
            capture_output=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("gbk", errors="replace"),
        )


if __name__ == "__main__":
    unittest.main()
