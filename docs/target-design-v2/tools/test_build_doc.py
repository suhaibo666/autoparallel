from __future__ import annotations

import ast
import pathlib
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET

from tools.build_doc import build_document
from tools.render_mermaid import extract_mermaid_figures, validate_inline_svg


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "index.template.html"
CSS = ROOT / "src" / "style.css"
MERMAID_CONFIG = ROOT / "src" / "mermaid.config.json"
HANDOFF = ROOT / "HANDOFF.md"
BUILD_SCRIPT = ROOT / "tools" / "build_doc.py"

EXPECTED_DIAGRAM_TEXT = {
    "production-pipeline": ("RankCodeIR(P,r)", "RuntimeEventPlan", "BackendSealArtifact"),
    "facts-and-digests": ("model_input_digest", "memory_simulation_digest", "TraceFixture"),
    "per-rank-codeir": ("SourceObligation", "RankCodeIR(P,r)", "CodeIR(P)"),
    "value-storage-identity": ("ValuePrototypeRef", "TensorInstanceId", "SimulationInitialState"),
    "semantic-effect-closure": (
        "SEMANTIC_VALIDATED",
        "RUNTIME_BOUND",
        "PLANNED",
        "InternalContractViolation",
    ),
    "runtime-event-expansion": ("AutogradLink", "P2PIntent", "CollectiveIntent"),
    "core-dual-projection": ("KernelVariantBinding", "ProjectionBundleAuthority", "NotRequested"),
    "memory-logical-replay": ("logical_kernel_order", "Allocate", "MemoryEstimate"),
    "time-progress-des": ("ComputeMeasurementRecord", "CommunicationModelSnapshot", "No-contention DES"),
    "result-gate-comparison": ("BackendResultSourceAuthority", "GateExecutionLedger", "G-REP2"),
}
EXPECTED_DIAGRAM_IDS = tuple(EXPECTED_DIAGRAM_TEXT)
FIGURE_RE = re.compile(r"<figure\b(?P<attrs>[^>]*)>(?P<body>.*?)</figure>", re.I | re.S)
SVG_RE = re.compile(r"<svg\b.*?</svg>", re.I | re.S)
CAPTION_RE = re.compile(r"<figcaption\b[^>]*>(?P<body>.*?)</figcaption>", re.I | re.S)


def _figure_blocks(html: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    for match in FIGURE_RE.finditer(html):
        diagram_id = re.search(r'\bdata-diagram-id="([^"]+)"', match.group("attrs"), re.I)
        if diagram_id:
            blocks.setdefault(diagram_id.group(1), []).append(match.group(0))
    return blocks


def _caption_text(block: str) -> str:
    match = CAPTION_RE.search(block)
    if not match:
        return ""
    return re.sub(r"<[^>]+>", "", match.group("body")).strip()


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _mermaid_sources() -> dict[str, str]:
    return {
        figure.diagram_id: figure.source
        for figure in extract_mermaid_figures(SOURCE.read_text(encoding="utf-8"))
    }


def _write_fixture(directory: pathlib.Path, template: str) -> tuple[pathlib.Path, pathlib.Path]:
    source = directory / "fixture.template.html"
    css = directory / "fixture.css"
    source.write_text(template, encoding="utf-8")
    css.write_text("body { color: black; }\n", encoding="utf-8")
    return source, css


class BuildDocArtifactTest(unittest.TestCase):
    def test_facts_digest_source_has_independent_runtime_inputs(self) -> None:
        """Making runtime facts a model-digest derivative must fail."""
        source = _mermaid_sources()["facts-and-digests"]

        self.assertRegex(source, r"\bmodelDigest\s*-->\s*runtimeInput\b")
        self.assertRegex(source, r"\bruntimeFacts\s*-->\s*runtimeInput\b")
        self.assertNotRegex(source, r"\bmodelDigest\s*-->\s*runtimeFacts\b")

    def test_value_storage_identity_source_names_occurrence_dimensions(self) -> None:
        """Hiding microbatch, repeat, or recompute identity dimensions must fail."""
        source = _mermaid_sources()["value-storage-identity"]

        self.assertRegex(source, r"\bclass\s+EventInstanceKey\b")
        for dimension in ("microbatch_id", "repeat_ordinal", "recompute"):
            self.assertIn(dimension, source)
        self.assertRegex(source, r"\bEventInstanceKey\s*-->\s*TensorInstanceId\b")
        self.assertRegex(source, r"\bEventInstanceKey\s*-->\s*StorageInstanceId\b")

    def test_runtime_event_source_correlates_send_and_receive_endpoints(self) -> None:
        """Collapsing P2P endpoints without their exact matching key must fail."""
        source = _mermaid_sources()["runtime-event-expansion"]

        self.assertRegex(source, r"\bsendEndpoint\s*\[[^\]]*SendEndpoint")
        self.assertRegex(source, r"\brecvEndpoint\s*\[[^\]]*RecvEndpoint")
        self.assertRegex(source, r"\bp2p\s*\[[^\]]*P2PIntent")
        for endpoint in ("sendEndpoint", "recvEndpoint"):
            self.assertRegex(
                source,
                rf"\b{endpoint}\s*-->\s*\|"
                rf"(?=[^|]*channel_id)(?=[^|]*sequence_in_channel)"
                rf"(?=[^|]*bytes_expr)[^|]*\|\s*p2p\b",
            )

    def test_time_des_source_lists_every_progress_edge_family(self) -> None:
        """Omitting any semantic, stream, or communication edge family must fail."""
        source = _mermaid_sources()["time-progress-des"]
        edge_node = re.search(r'\bedges\s*\["(?P<label>[^"]+)"\]', source)

        self.assertIsNotNone(edge_node)
        label = edge_node.group("label")
        for edge_family in (
            "data",
            "control",
            "effect",
            "schedule",
            "stream",
            "communication sequence",
        ):
            self.assertIn(edge_family, label)

    def test_authority_template_declares_exactly_the_ten_approved_diagrams(self) -> None:
        """Removing, duplicating, or under-describing an approved figure must fail."""
        template = SOURCE.read_text(encoding="utf-8")
        figures = extract_mermaid_figures(template)
        blocks = _figure_blocks(template)

        self.assertEqual(tuple(figure.diagram_id for figure in figures), EXPECTED_DIAGRAM_IDS)
        self.assertEqual(set(blocks), set(EXPECTED_DIAGRAM_IDS))
        for figure in figures:
            with self.subTest(diagram_id=figure.diagram_id):
                self.assertTrue(figure.title)
                self.assertTrue(figure.description)
                self.assertTrue(figure.source.strip())
                self.assertEqual(len(blocks[figure.diagram_id]), 1)
                self.assertTrue(_caption_text(blocks[figure.diagram_id][0]))

    def test_built_outputs_inline_each_approved_diagram_with_closed_svg(self) -> None:
        """Broken rendering, accessibility, ID closure, or semantic labels must fail."""
        template = SOURCE.read_text(encoding="utf-8")
        metadata = {
            figure.diagram_id: (figure.title, figure.description)
            for figure in extract_mermaid_figures(template)
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            index, artifact = build_document(
                SOURCE,
                CSS,
                output_dir,
                mermaid_config_path=MERMAID_CONFIG,
            )
            outputs = {
                "index": index.read_text(encoding="utf-8"),
                "artifact": artifact.read_text(encoding="utf-8"),
            }

        for output_name, html in outputs.items():
            with self.subTest(output=output_name):
                blocks = _figure_blocks(html)
                self.assertEqual(set(blocks), set(EXPECTED_DIAGRAM_IDS))
                self.assertEqual(sum(map(len, blocks.values())), len(EXPECTED_DIAGRAM_IDS))
                self.assertNotRegex(
                    html,
                    r'<pre\b[^>]*\bclass=["\'][^"\']*\bmermaid-source\b',
                )
                remaining_pre_blocks = "\n".join(
                    re.findall(r"<pre\b[^>]*>.*?</pre>", html, re.I | re.S)
                )
                for raw_token in (
                    "flowchart TD",
                    "flowchart LR",
                    "stateDiagram-v2",
                    "classDiagram",
                    "sequenceDiagram",
                ):
                    self.assertNotIn(raw_token, remaining_pre_blocks)

                page_ids: set[str] = set()
                for diagram_id, required_text in EXPECTED_DIAGRAM_TEXT.items():
                    block = blocks[diagram_id][0]
                    svg_match = SVG_RE.search(block)
                    self.assertIsNotNone(svg_match, diagram_id)
                    svg = svg_match.group(0)
                    validate_inline_svg(svg, diagram_id)
                    root = ET.fromstring(svg)
                    svg_ids = {
                        value
                        for element in root.iter()
                        if (value := element.get("id"))
                    }
                    self.assertTrue(page_ids.isdisjoint(svg_ids), diagram_id)
                    page_ids.update(svg_ids)

                    direct = list(root)
                    title = next(
                        element for element in direct if _local_name(element.tag) == "title"
                    )
                    description = next(
                        element for element in direct if _local_name(element.tag) == "desc"
                    )
                    self.assertEqual("".join(title.itertext()), metadata[diagram_id][0])
                    self.assertEqual("".join(description.itertext()), metadata[diagram_id][1])
                    self.assertTrue(_caption_text(block))

                    visible_text = " ".join(
                        "".join(element.itertext())
                        for element in root.iter()
                        if _local_name(element.tag) == "text"
                    )
                    compact_visible_text = re.sub(r"\s+", "", visible_text)
                    for token in required_text:
                        self.assertIn(
                            re.sub(r"\s+", "", token),
                            compact_visible_text,
                            (diagram_id, token),
                        )
                    for element in root.iter():
                        for raw_name, value in element.attrib.items():
                            if _local_name(raw_name) in {"href", "src"} and value:
                                self.assertTrue(value.startswith("#"), (diagram_id, value))

    def test_directly_opened_artifact_declares_utf8(self) -> None:
        """Removing the artifact charset declaration must break this test."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            _, artifact = build_document(
                SOURCE,
                CSS,
                output_dir,
                mermaid_config_path=MERMAID_CONFIG,
            )
            prefix = artifact.read_bytes()[:1024].lower()

        self.assertIn(b'<meta charset="utf-8">', prefix)

    def test_repeated_full_build_is_byte_stable(self) -> None:
        """Randomized Mermaid geometry must not change either generated artifact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            first_index, first_artifact = build_document(
                SOURCE,
                CSS,
                root / "first",
                mermaid_config_path=MERMAID_CONFIG,
            )
            second_index, second_artifact = build_document(
                SOURCE,
                CSS,
                root / "second",
                mermaid_config_path=MERMAID_CONFIG,
            )

            self.assertEqual(first_index.read_bytes(), second_index.read_bytes())
            self.assertEqual(first_artifact.read_bytes(), second_artifact.read_bytes())

    def test_svg_title_and_style_remain_inside_svg_in_index(self) -> None:
        template = """<title>Document title</title>
{{CSS}}
<div class="wrap">{{TOC}}<main>
<h2 id="c1"><span class="cn">Chapter 1</span>Body</h2>
<svg viewBox="0 0 10 10"><title>Diagram title</title>
<style>.diagram-node { fill: red; }</style><text>Node</text></svg>
</main></div>
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = pathlib.Path(temp_dir)
            source, css = _write_fixture(directory, template)
            index, _ = build_document(
                source,
                css,
                directory / "out",
                mermaid_config_path=MERMAID_CONFIG,
            )
            html = index.read_text(encoding="utf-8")

        svg_start = html.index("<svg")
        svg_end = html.index("</svg>", svg_start)
        svg = html[svg_start:svg_end]
        head = html[html.index("<head>"):html.index("</head>")]
        self.assertIn("<title>Diagram title</title>", svg)
        self.assertIn("<style>.diagram-node { fill: red; }</style>", svg)
        self.assertNotIn("Diagram title", head)
        self.assertNotIn("diagram-node", head)

    def test_mermaid_source_is_replaced_by_inline_svg_in_both_outputs(self) -> None:
        template = """<title>Mermaid document</title>
{{CSS}}
<div class="wrap">{{TOC}}<main>
<h2 id="c1"><span class="cn">Chapter 1</span>Flow</h2>
<figure class="mermaid-figure" data-diagram-id="build-flow"
        data-title="Build flow" data-description="Alpha to beta">
  <pre class="mermaid-source"><code>flowchart LR
    A[Alpha] --&gt; B[Beta]
  </code></pre>
  <figcaption>Build flow</figcaption>
</figure>
</main></div>
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = pathlib.Path(temp_dir)
            source, css = _write_fixture(directory, template)
            index, artifact = build_document(
                source,
                css,
                directory / "out",
                mermaid_config_path=MERMAID_CONFIG,
            )
            outputs = [
                index.read_text(encoding="utf-8"),
                artifact.read_text(encoding="utf-8"),
            ]

        for html in outputs:
            with self.subTest(output=html[:40]):
                self.assertIn("<svg", html)
                self.assertIn("Alpha", html)
                self.assertNotIn("mermaid-source", html)
                self.assertNotIn("flowchart LR", html)
                self.assertNotIn("{{SVG:", html)

    def test_legacy_svg_placeholder_is_rejected_before_writing_outputs(self) -> None:
        template = """<title>Legacy placeholder</title>
{{CSS}}
<div class="wrap">{{TOC}}<main>
<h2 id="c1"><span class="cn">Chapter 1</span>Legacy</h2>
{{SVG:legacy}}
</main></div>
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = pathlib.Path(temp_dir)
            source, css = _write_fixture(directory, template)
            output_dir = directory / "out"

            with self.assertRaisesRegex(ValueError, "unresolved template placeholder"):
                build_document(
                    source,
                    css,
                    output_dir,
                    mermaid_config_path=MERMAID_CONFIG,
                )

            self.assertFalse((output_dir / "index.html").exists())
            self.assertFalse((output_dir / "artifact.html").exists())


class BuildDocActivityPathTest(unittest.TestCase):
    def test_legacy_diagram_assets_exist_but_are_outside_the_active_build(self) -> None:
        """Reactivating gen_diagrams/SVG placeholders or hiding legacy assets must fail."""
        build_source = BUILD_SCRIPT.read_text(encoding="utf-8")
        build_tree = ast.parse(build_source)
        imported_modules = {
            alias.name
            for node in ast.walk(build_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module or ""
            for node in ast.walk(build_tree)
            if isinstance(node, ast.ImportFrom)
        )
        called_names = {
            node.func.id
            for node in ast.walk(build_tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        called_names.update(
            node.func.attr
            for node in ast.walk(build_tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        )
        template = SOURCE.read_text(encoding="utf-8")
        handoff = HANDOFF.read_text(encoding="utf-8")

        self.assertNotIn("gen_diagrams", imported_modules)
        self.assertNotIn("gen_diagrams", called_names)
        self.assertNotIn("{{SVG:", template)
        self.assertTrue((ROOT / "tools" / "gen_diagrams.py").is_file())
        self.assertTrue(tuple((ROOT / "diagrams").glob("*.excalidraw")))
        self.assertTrue(tuple((ROOT / "build" / "svg").glob("*.svg")))
        for legacy_path in (
            "tools/gen_diagrams.py",
            "diagrams/*.excalidraw",
            "build/svg/*.svg",
        ):
            with self.subTest(legacy_path=legacy_path):
                self.assertIn(legacy_path, handoff)
        self.assertIn("legacy", handoff.lower())
        self.assertIn("不参与当前构建", handoff)
        self.assertIn("非权威", handoff)

    def test_handoff_indexes_every_authoritative_diagram_and_module_contract(self) -> None:
        """Omitting an approved diagram or module-contract locator must fail."""
        handoff = HANDOFF.read_text(encoding="utf-8")
        expected_modules = (
            "input-facts",
            "code-ir",
            "runtime-events",
            "memory-backend",
            "time-backend",
            "result-sealing",
            "gate-system",
            "comparison",
            "conformance",
        )

        for diagram_id in EXPECTED_DIAGRAM_IDS:
            with self.subTest(diagram_id=diagram_id):
                self.assertIn(f"`{diagram_id}`", handoff)
        for module_id in expected_modules:
            with self.subTest(module_id=module_id):
                self.assertIn(f"`{module_id}`", handoff)

    def test_handoff_documents_the_offline_deterministic_safe_build(self) -> None:
        """An unpinned, network-backed, or unaudited build recipe must fail."""
        handoff = HANDOFF.read_text(encoding="utf-8")
        required_phrases = (
            "@mermaid-js/mermaid-cli@11.16.0",
            "npm ci",
            "python tools/build_doc.py",
            "python tools/verify_gates.py",
            'python -m unittest discover -s tools -p "test_*.py" -v',
            "node_modules/.bin/mmdc",
            "securityLevel=strict",
            "htmlLabels=false",
            "deterministicIds=true",
            "deterministicIDSeed=target-design-v2",
            "handDrawnSeed=271828",
            "不回退到 CDN",
            "连续构建两次",
            "SHA-256",
        )

        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, handoff)

    def test_handoff_replaces_load_bearing_ascii_flows_with_authority_links(self) -> None:
        """Restoring the duplicated top-level ASCII pipeline or file tree must fail."""
        handoff = HANDOFF.read_text(encoding="utf-8")
        overview = handoff.split("## 1.", 1)[0]
        fenced_text = "\n".join(re.findall(r"```text\s*(.*?)```", overview, re.S))
        all_fenced_blocks = "\n".join(
            re.findall(r"```[^\n]*\n(.*?)```", handoff, re.S)
        )

        self.assertNotRegex(fenced_text, r"RequestSnapshot[\s\S]*RuntimeEventPlan")
        self.assertNotRegex(all_fenced_blocks, r"[├└│]|(?:^|\n)\s*\+--")
        self.assertIn("`production-pipeline`", overview)
        self.assertIn("`src/index.template.html`", overview)

    def test_handoff_does_not_overclaim_conformance_session_isolation(self) -> None:
        """Treating digest closure as proof of execution-session isolation must fail."""
        handoff = HANDOFF.read_text(encoding="utf-8")

        for phrase in (
            "MeasuredExecutionEnvironment",
            "不可重放 envelope",
            "invocation、runner/executable、process/container/session identity、single-use nonce",
            "ValidatedMeasurementSessionCapability",
            "同一调用域",
            "摘要与签名不能自行证明这些物理根可信",
            "脱离该信任根的宿主隔离证明",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, handoff)


if __name__ == "__main__":
    unittest.main()
