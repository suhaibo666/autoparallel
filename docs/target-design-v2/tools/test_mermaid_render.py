from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from tools.render_mermaid import (
    MermaidRenderError,
    extract_mermaid_figures,
    render_mermaid_source,
    sanitize_and_prefix_svg,
    validate_inline_svg,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "mermaid.config.json"
SVG_NS = "http://www.w3.org/2000/svg"


def _svg(body: str, *, view_box: str = ' viewBox="0 0 120 40"') -> str:
    return f'<svg xmlns="{SVG_NS}"{view_box}>{body}</svg>'


def _ids(svg_text: str) -> set[str]:
    root = ET.fromstring(svg_text)
    return {value for element in root.iter() if (value := element.get("id"))}


class MermaidFigureExtractionTest(unittest.TestCase):
    def test_extracts_escaped_source_and_accessibility_metadata(self) -> None:
        template = """
<figure class="wide mermaid-figure" data-diagram-id="pipeline"
        data-title="Production pipeline"
        data-description="Source to sealed result">
  <pre class="mermaid-source"><code>flowchart LR
    A[&quot;Source &amp; config&quot;] --&gt; B[Result]
  </code></pre>
  <figcaption>Production pipeline</figcaption>
</figure>
"""

        figures = extract_mermaid_figures(template)

        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0].diagram_id, "pipeline")
        self.assertEqual(figures[0].title, "Production pipeline")
        self.assertEqual(figures[0].description, "Source to sealed result")
        self.assertIn('A["Source & config"] --> B[Result]', figures[0].source)

    def test_rejects_duplicate_diagram_ids(self) -> None:
        figure = """
<figure class="mermaid-figure" data-diagram-id="same"
        data-title="Same" data-description="Duplicate">
  <pre class="mermaid-source"><code>flowchart LR
    A --&gt; B
  </code></pre>
</figure>
"""

        with self.assertRaisesRegex(ValueError, "duplicate diagram id"):
            extract_mermaid_figures(figure + figure)


class MermaidSvgContractTest(unittest.TestCase):
    def test_prefixes_ids_per_diagram_and_rewrites_local_references(self) -> None:
        raw = _svg(
            """
<defs>
  <marker id="arrow"><path d="M0 0 L5 2.5 L0 5 z" /></marker>
  <clipPath id="clip"><rect width="120" height="40" /></clipPath>
</defs>
<g id="node" clip-path="url(#clip)">
  <a href="#node"><text>Node</text></a>
  <path marker-end="url(#arrow)" d="M0 0 L100 0" />
</g>
"""
        )

        first = sanitize_and_prefix_svg(raw, "first", "First", "First graph")
        second = sanitize_and_prefix_svg(raw, "second", "Second", "Second graph")

        first_ids = _ids(first)
        second_ids = _ids(second)
        self.assertTrue(first_ids)
        self.assertTrue(all(value.startswith("mmd-first--") for value in first_ids))
        self.assertTrue(all(value.startswith("mmd-second--") for value in second_ids))
        self.assertTrue(first_ids.isdisjoint(second_ids))
        self.assertIn("url(#mmd-first--clip)", first)
        self.assertIn('href="#mmd-first--node"', first)
        self.assertIn("url(#mmd-first--arrow)", first)
        validate_inline_svg(first, "first")
        validate_inline_svg(second, "second")

    def test_sanitization_is_byte_stable_for_identical_input(self) -> None:
        raw = _svg('<g id="b"><text z="2" a="1">Stable</text></g>')

        one = sanitize_and_prefix_svg(raw, "stable", "Stable", "Stable graph")
        two = sanitize_and_prefix_svg(raw, "stable", "Stable", "Stable graph")

        self.assertEqual(one, two)

    def test_rejects_active_or_network_capable_svg(self) -> None:
        unsafe = {
            "script": _svg("<script>alert(1)</script><text>Unsafe</text>"),
            "foreignObject": _svg(
                '<foreignObject><div xmlns="http://www.w3.org/1999/xhtml">x</div>'
                "</foreignObject><text>Unsafe</text>"
            ),
            "event attribute": _svg('<text onclick="alert(1)">Unsafe</text>'),
            "external href": _svg('<a href="https://example.invalid/"><text>Unsafe</text></a>'),
            "external css url": _svg(
                "<style>.x{fill:url(https://example.invalid/a.svg)}</style><text>Unsafe</text>"
            ),
            "css import": _svg(
                '<style>@import "https://example.invalid/a.css";</style><text>Unsafe</text>'
            ),
        }

        for label, svg_text in unsafe.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    sanitize_and_prefix_svg(svg_text, "unsafe", "Unsafe", "Unsafe graph")

    def test_rejects_css_escape_obfuscated_external_url(self) -> None:
        raw = _svg(
            r"<style>.x{fill:u\72l(https://example.invalid/a.svg)}</style>"
            "<text>Unsafe</text>"
        )

        with self.assertRaises(ValueError):
            sanitize_and_prefix_svg(raw, "escaped-url", "Unsafe", "Unsafe graph")

    def test_rejects_css_escape_obfuscated_import(self) -> None:
        raw = _svg(
            r'<style>@im\70ort "https://example.invalid/a.css";</style>'
            "<text>Unsafe</text>"
        )

        with self.assertRaises(ValueError):
            sanitize_and_prefix_svg(raw, "escaped-import", "Unsafe", "Unsafe graph")

    def test_rejects_smil_set_that_assigns_external_href(self) -> None:
        raw = _svg(
            '<set attributeName="href" to="https://example.invalid/a.svg" />'
            "<text>Unsafe</text>"
        )

        with self.assertRaises(ValueError):
            sanitize_and_prefix_svg(raw, "smil-set", "Unsafe", "Unsafe graph")

    def test_rejects_any_nonempty_src_attribute(self) -> None:
        unsafe = {
            "https": "https://example.invalid/a.svg",
            "protocol-relative": "//example.invalid/a.svg",
            "data": "data:image/svg+xml;base64,PHN2Zy8+",
            "relative": "embedded/a.svg",
        }

        for label, source in unsafe.items():
            with self.subTest(label=label):
                raw = _svg(f'<image src="{source}" /><text>Unsafe</text>')
                with self.assertRaises(ValueError):
                    sanitize_and_prefix_svg(raw, "image-src", "Unsafe", "Unsafe graph")

    def test_rejects_animate_syncbase_reference(self) -> None:
        raw = _svg(
            '<g id="target"><text>Target</text></g>'
            '<animate attributeName="opacity" begin="target.click" dur="1s" />'
        )

        with self.assertRaises(ValueError):
            sanitize_and_prefix_svg(raw, "smil-begin", "Unsafe", "Unsafe graph")

    def test_rejects_all_other_unsupported_animation_elements(self) -> None:
        unsafe = {
            "animateMotion": '<animateMotion dur="1s" path="M0 0 L1 1" />',
            "animateTransform": (
                '<animateTransform attributeName="transform" type="rotate" dur="1s" />'
            ),
            "mpath": '<path id="target" d="M0 0" /><mpath href="#target" />',
        }

        for label, payload in unsafe.items():
            with self.subTest(label=label):
                raw = _svg(payload + "<text>Unsafe</text>")
                with self.assertRaises(ValueError):
                    sanitize_and_prefix_svg(raw, "smil-extra", "Unsafe", "Unsafe graph")

    def test_rejects_css_comments_and_all_at_rules(self) -> None:
        unsafe = {
            "comment": (
                "<style>.x{fill:u/**/rl(https://example.invalid/a.svg)}</style>"
                "<text>Unsafe</text>"
            ),
            "at-rule": "<style>@media screen {.x{fill:red}}</style><text>Unsafe</text>",
        }

        for label, svg_body in unsafe.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    sanitize_and_prefix_svg(
                        _svg(svg_body), "css-grammar", "Unsafe", "Unsafe graph"
                    )

    def test_rejects_quoted_local_css_url(self) -> None:
        raw = _svg(
            '<defs><clipPath id="clip"><rect width="1" height="1" /></clipPath></defs>'
            '<style>.x{clip-path:url("#clip")}</style><text>Unsafe</text>'
        )

        with self.assertRaises(ValueError):
            sanitize_and_prefix_svg(raw, "quoted-url", "Unsafe", "Unsafe graph")

    def test_rejects_css_animation_declaration(self) -> None:
        raw = _svg(
            "<style>.x{animation:dash 1s linear infinite}</style><text>Unsafe</text>"
        )

        with self.assertRaises(ValueError):
            sanitize_and_prefix_svg(raw, "css-animation", "Unsafe", "Unsafe graph")

    def test_rejects_dangling_local_reference(self) -> None:
        raw = _svg('<path marker-end="url(#missing)" /><text>Dangling</text>')

        with self.assertRaisesRegex(ValueError, "dangling SVG reference"):
            sanitize_and_prefix_svg(raw, "dangling", "Dangling", "Dangling graph")

    def test_validated_output_requires_viewbox_title_desc_and_native_text(self) -> None:
        missing = {
            "viewBox": _svg(
                "<title>T</title><desc>D</desc><text>Node</text>", view_box=""
            ),
            "title": _svg("<desc>D</desc><text>Node</text>"),
            "description": _svg("<title>T</title><text>Node</text>"),
            "native text": _svg("<title>T</title><desc>D</desc><path d=\"M0 0\" />"),
        }

        for label, svg_text in missing.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    validate_inline_svg(svg_text, "required")


class MermaidCliIntegrationTest(unittest.TestCase):
    def test_mmdc_timeout_is_bounded_and_wrapped_with_diagram_context(self) -> None:
        observed_timeouts: list[float | None] = []

        def time_out(command: list[str], **kwargs: object) -> None:
            timeout = kwargs.get("timeout")
            observed_timeouts.append(timeout if isinstance(timeout, (int, float)) else None)
            raise subprocess.TimeoutExpired(command, timeout)

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("tools.render_mermaid.subprocess.run", side_effect=time_out):
                try:
                    render_mermaid_source(
                        "flowchart LR\n    A --> B\n",
                        "timeout-diagram",
                        CONFIG,
                        pathlib.Path(temp_dir),
                    )
                except Exception as exc:
                    self.assertIsInstance(exc, MermaidRenderError)
                    self.assertIn("timeout-diagram", str(exc))
                    self.assertIsInstance(exc.__cause__, subprocess.TimeoutExpired)
                else:
                    self.fail("render_mermaid_source did not surface an mmdc timeout")

        self.assertEqual(observed_timeouts, [60])

    def test_same_mermaid_source_renders_to_identical_sanitized_svg(self) -> None:
        source = """flowchart LR
    A[Alpha] --> B[Beta]
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            raw_one = render_mermaid_source(source, "stable", CONFIG, output_dir)
            raw_two = render_mermaid_source(source, "stable", CONFIG, output_dir)

        self.assertNotIn("@keyframes", raw_one)
        self.assertNotIn("animation:", raw_one)
        one = sanitize_and_prefix_svg(raw_one, "stable", "Stable", "Alpha to beta")
        two = sanitize_and_prefix_svg(raw_two, "stable", "Stable", "Alpha to beta")

        self.assertEqual(one, two)
        self.assertNotIn("foreignObject", one)
        self.assertIn("<text", one)
        validate_inline_svg(one, "stable")


if __name__ == "__main__":
    unittest.main()
