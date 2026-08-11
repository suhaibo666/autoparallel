from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
from html.parser import HTMLParser
import os
import pathlib
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "src" / "index.template.html"
DEFAULT_CONFIG = ROOT / "src" / "mermaid.config.json"
DEFAULT_OUTPUT_DIR = ROOT / "build" / "mermaid"
MMDC_TIMEOUT_SECONDS = 60
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
DIAGRAM_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
FIGURE_RE = re.compile(r"<figure\b(?P<attrs>[^>]*)>(?P<body>.*?)</figure>", re.I | re.S)
PRE_RE = re.compile(r"<pre\b(?P<attrs>[^>]*)>(?P<body>.*?)</pre>", re.I | re.S)
CODE_RE = re.compile(r"^\s*<code\b[^>]*>(?P<body>.*?)</code>\s*$", re.I | re.S)
LOCAL_URL_RE = re.compile(r"url\(#(?P<target>[-A-Za-z0-9_:.]+)\)", re.I)
URL_FUNCTION_RE = re.compile(r"(?<![-\w])url\s*\(", re.I)
ANIMATION_DECL_RE = re.compile(
    r"(?i)(?<![-\w])(?:-[a-z]+-)?animation(?:-[a-z-]+)?\s*:[^;}]*;?"
)
FORBIDDEN_SVG_ELEMENTS = {
    "animate",
    "animatemotion",
    "animatetransform",
    "foreignobject",
    "mpath",
    "script",
    "set",
}
CSS_VALUE_ATTRIBUTES = {
    "clip-path",
    "color",
    "cursor",
    "fill",
    "filter",
    "marker",
    "marker-end",
    "marker-mid",
    "marker-start",
    "mask",
    "stroke",
    "style",
}


class MermaidRenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class MermaidFigureSource:
    diagram_id: str
    title: str
    description: str
    source: str
    source_start: int
    source_end: int


class _StartTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.attrs: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.attrs is None:
            self.attrs = {name.lower(): value or "" for name, value in attrs}


def _tag_attrs(tag: str) -> dict[str, str]:
    parser = _StartTagParser()
    parser.feed(tag)
    if parser.attrs is None:
        raise ValueError("invalid HTML start tag")
    return parser.attrs


def _class_tokens(attrs: dict[str, str]) -> set[str]:
    return set(attrs.get("class", "").split())


def extract_mermaid_figures(template_html: str) -> list[MermaidFigureSource]:
    figures: list[MermaidFigureSource] = []
    seen: set[str] = set()
    for figure_match in FIGURE_RE.finditer(template_html):
        opening = template_html[figure_match.start():figure_match.start("body")]
        attrs = _tag_attrs(opening)
        if "mermaid-figure" not in _class_tokens(attrs):
            continue

        diagram_id = attrs.get("data-diagram-id", "").strip()
        title = attrs.get("data-title", "").strip()
        description = attrs.get("data-description", "").strip()
        if not DIAGRAM_ID_RE.fullmatch(diagram_id):
            raise ValueError(f"invalid Mermaid diagram id: {diagram_id!r}")
        if diagram_id in seen:
            raise ValueError(f"duplicate diagram id: {diagram_id}")
        if not title or not description:
            raise ValueError(f"Mermaid figure {diagram_id!r} requires data-title and data-description")

        source_match = None
        for candidate in PRE_RE.finditer(figure_match.group("body")):
            opening_end = candidate.start("body") - candidate.start()
            candidate_opening = candidate.group(0)[:opening_end]
            if "mermaid-source" in _class_tokens(_tag_attrs(candidate_opening)):
                if source_match is not None:
                    raise ValueError(f"Mermaid figure {diagram_id!r} has multiple source blocks")
                source_match = candidate
        if source_match is None:
            raise ValueError(f"Mermaid figure {diagram_id!r} has no mermaid-source block")

        encoded_source = source_match.group("body")
        code_match = CODE_RE.match(encoded_source)
        if code_match:
            encoded_source = code_match.group("body")
        if re.search(r"<[^>]+>", encoded_source):
            raise ValueError(f"Mermaid source {diagram_id!r} must be escaped plain text")
        source = html.unescape(encoded_source).strip()
        if not source:
            raise ValueError(f"Mermaid figure {diagram_id!r} has empty source")

        body_offset = figure_match.start("body")
        figures.append(
            MermaidFigureSource(
                diagram_id=diagram_id,
                title=title,
                description=description,
                source=source + "\n",
                source_start=body_offset + source_match.start(),
                source_end=body_offset + source_match.end(),
            )
        )
        seen.add(diagram_id)
    return figures


def _mmdc_path() -> pathlib.Path:
    name = "mmdc.cmd" if os.name == "nt" else "mmdc"
    path = ROOT / "node_modules" / ".bin" / name
    if not path.is_file():
        raise MermaidRenderError(
            f"local Mermaid CLI is missing at {path}; run `npm ci` in {ROOT}"
        )
    return path


def render_mermaid_source(
    source: str,
    diagram_id: str,
    config_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
) -> str:
    if not DIAGRAM_ID_RE.fullmatch(diagram_id):
        raise ValueError(f"invalid Mermaid diagram id: {diagram_id!r}")
    if not source.strip():
        raise ValueError("Mermaid source must not be empty")
    config = pathlib.Path(config_path).resolve()
    if not config.is_file():
        raise MermaidRenderError(f"Mermaid config is missing: {config}")
    work_root = pathlib.Path(output_dir).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"mmd-{diagram_id}-", dir=work_root) as temp_dir:
        temp = pathlib.Path(temp_dir)
        input_path = temp / f"{diagram_id}.mmd"
        output_path = temp / f"{diagram_id}.svg"
        input_path.write_text(source.rstrip() + "\n", encoding="utf-8", newline="\n")
        command = [
            str(_mmdc_path()),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--configFile",
            str(config),
            "--backgroundColor",
            "transparent",
            "--quiet",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=MMDC_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise MermaidRenderError(
                f"mmdc timed out after {MMDC_TIMEOUT_SECONDS}s for {diagram_id!r}"
            ) from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise MermaidRenderError(
                f"mmdc failed for {diagram_id!r} with exit code {completed.returncode}: {detail}"
            )
        if not output_path.is_file():
            raise MermaidRenderError(f"mmdc produced no SVG for {diagram_id!r}")
        return _staticize_mmdc_svg(output_path.read_text(encoding="utf-8"))


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _css_block_end(css: str, opening_brace: int) -> int:
    depth = 1
    quote: str | None = None
    escaped = False
    index = opening_brace + 1
    while index < len(css):
        char = css[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise MermaidRenderError("mmdc emitted an unterminated CSS at-rule block")


def _strip_mmdc_keyframes(css: str) -> str:
    output: list[str] = []
    cursor = 0
    while True:
        at = css.find("@", cursor)
        if at < 0:
            output.append(css[cursor:])
            break
        output.append(css[cursor:at])
        if not css[at:].lower().startswith("@keyframes"):
            raise MermaidRenderError("mmdc emitted an unsupported CSS at-rule")
        opening_brace = css.find("{", at + len("@keyframes"))
        if opening_brace < 0:
            raise MermaidRenderError("mmdc emitted an unterminated @keyframes rule")
        cursor = _css_block_end(css, opening_brace)
    return "".join(output)


def _staticize_css(css: str) -> str:
    static_css = _strip_mmdc_keyframes(css)
    static_css = ANIMATION_DECL_RE.sub("", static_css)
    _validate_css(static_css)
    return static_css


def _staticize_mmdc_svg(svg_text: str) -> str:
    if re.search(r"<!DOCTYPE|<!ENTITY", svg_text, re.I):
        raise MermaidRenderError("mmdc emitted a forbidden SVG document type or entity")
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise MermaidRenderError(f"mmdc emitted invalid SVG XML: {exc}") from exc
    if _local_name(root.tag).lower() != "svg":
        raise MermaidRenderError("mmdc output root is not svg")
    for element in root.iter():
        if _local_name(element.tag).lower() == "style":
            element.text = _staticize_css("".join(element.itertext()))
        style = element.get("style")
        if style is not None:
            element.set("style", _staticize_css(style))
    return _serialize_svg(root)


def _validate_css(css: str) -> None:
    if ANIMATION_DECL_RE.search(css):
        raise ValueError("SVG CSS animation declarations are forbidden")
    if "\\" in css:
        raise ValueError("SVG CSS escapes are forbidden")
    if "/*" in css or "*/" in css:
        raise ValueError("SVG CSS comments are forbidden")
    if "@" in css:
        raise ValueError("SVG CSS at-rules are forbidden")
    without_local_urls = LOCAL_URL_RE.sub("", css)
    if URL_FUNCTION_RE.search(without_local_urls):
        raise ValueError("SVG CSS only permits url(#local-id)")


def _reject_unsafe_svg(root: ET.Element) -> None:
    for element in root.iter():
        local_tag = _local_name(element.tag).lower()
        if local_tag in FORBIDDEN_SVG_ELEMENTS:
            raise ValueError(f"unsafe SVG element: {local_tag}")
        if local_tag == "style":
            _validate_css("".join(element.itertext()))

        for raw_name, value in element.attrib.items():
            local_name = _local_name(raw_name).lower()
            if local_name.startswith("on"):
                raise ValueError(f"unsafe SVG event attribute: {local_name}")
            if local_name == "href" and value and not value.startswith("#"):
                raise ValueError(f"external SVG href: {value}")
            if local_name == "src" and value:
                raise ValueError(f"embedded SVG src is forbidden: {value}")
            if local_name in CSS_VALUE_ATTRIBUTES or URL_FUNCTION_RE.search(value):
                _validate_css(value)


def _parse_svg(svg_text: str) -> ET.Element:
    if re.search(r"<!DOCTYPE|<!ENTITY", svg_text, re.I):
        raise ValueError("SVG document types and entities are forbidden")
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError(f"invalid SVG XML: {exc}") from exc
    if _local_name(root.tag).lower() != "svg":
        raise ValueError("inline diagram root must be svg")
    _reject_unsafe_svg(root)
    return root


def _remove_generator_metadata(root: ET.Element) -> None:
    for parent in root.iter():
        for child in list(parent):
            if _local_name(child.tag).lower() == "metadata":
                parent.remove(child)


def _rewrite_reference_value(value: str, id_map: dict[str, str]) -> str:
    def replace_url(match: re.Match[str]) -> str:
        old = match.group("target")
        return f"url(#{id_map.get(old, old)})"

    rewritten = LOCAL_URL_RE.sub(replace_url, value)
    if rewritten.startswith("#"):
        old = rewritten[1:]
        rewritten = "#" + id_map.get(old, old)
    return rewritten


def _rewrite_style_text(value: str, id_map: dict[str, str]) -> str:
    rewritten = _rewrite_reference_value(value, id_map)
    for old, new in sorted(id_map.items(), key=lambda item: len(item[0]), reverse=True):
        rewritten = re.sub(
            rf"(?<![\w-])#{re.escape(old)}(?![\w-])",
            f"#{new}",
            rewritten,
        )
    return rewritten


def _sort_attributes(root: ET.Element) -> None:
    for element in root.iter():
        if len(element.attrib) > 1:
            ordered = sorted(element.attrib.items())
            element.attrib.clear()
            element.attrib.update(ordered)


def _serialize_svg(root: ET.Element) -> str:
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    _sort_attributes(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", short_empty_elements=True).rstrip() + "\n"


def sanitize_and_prefix_svg(
    svg_text: str,
    diagram_id: str,
    title: str,
    description: str,
) -> str:
    if not DIAGRAM_ID_RE.fullmatch(diagram_id):
        raise ValueError(f"invalid Mermaid diagram id: {diagram_id!r}")
    if not title.strip() or not description.strip():
        raise ValueError("SVG title and description must not be empty")
    root = _parse_svg(svg_text)
    _remove_generator_metadata(root)

    prefix = f"mmd-{diagram_id}--"
    id_map: dict[str, str] = {}
    for element in root.iter():
        old = element.get("id")
        if old is None:
            continue
        if old in id_map:
            raise ValueError(f"duplicate SVG id: {old}")
        id_map[old] = old if old.startswith(prefix) else prefix + old

    for element in root.iter():
        old = element.get("id")
        if old is not None:
            element.set("id", id_map[old])
        for raw_name, value in list(element.attrib.items()):
            local_name = _local_name(raw_name).lower()
            if local_name in {"aria-labelledby", "aria-describedby"}:
                element.set(
                    raw_name,
                    " ".join(id_map.get(token, token) for token in value.split()),
                )
            elif local_name != "id":
                element.set(raw_name, _rewrite_reference_value(value, id_map))
        if _local_name(element.tag).lower() == "style" and element.text:
            element.text = _rewrite_style_text(element.text, id_map)

    for child in list(root):
        if _local_name(child.tag).lower() in {"title", "desc"}:
            root.remove(child)

    root_id = root.get("id") or prefix + "svg"
    if not root_id.startswith(prefix):
        root_id = prefix + root_id
    root.set("id", root_id)
    title_id = prefix + "title"
    description_id = prefix + "desc"
    if title_id in set(id_map.values()) or description_id in set(id_map.values()):
        raise ValueError("generated accessibility ID collides with SVG input")
    root.set("role", "img")
    root.set("aria-labelledby", f"{title_id} {description_id}")
    title_element = ET.Element(f"{{{SVG_NS}}}title", {"id": title_id})
    title_element.text = title.strip()
    desc_element = ET.Element(f"{{{SVG_NS}}}desc", {"id": description_id})
    desc_element.text = description.strip()
    root.insert(0, desc_element)
    root.insert(0, title_element)

    sanitized = _serialize_svg(root)
    validate_inline_svg(sanitized, diagram_id)
    return sanitized


def _collect_references(root: ET.Element) -> set[str]:
    references: set[str] = set()
    for element in root.iter():
        for raw_name, value in element.attrib.items():
            local_name = _local_name(raw_name).lower()
            if local_name == "href" and value.startswith("#"):
                references.add(value[1:])
            if local_name in {"aria-labelledby", "aria-describedby"}:
                references.update(value.split())
            for match in LOCAL_URL_RE.finditer(value):
                references.add(match.group("target"))
        if _local_name(element.tag).lower() == "style" and element.text:
            for match in LOCAL_URL_RE.finditer(element.text):
                references.add(match.group("target"))
    return references


def validate_inline_svg(svg_text: str, diagram_id: str) -> None:
    root = _parse_svg(svg_text)
    view_box = root.get("viewBox")
    if not view_box or len(view_box.split()) != 4:
        raise ValueError("inline SVG requires a four-value viewBox")

    direct_children = list(root)
    titles = [child for child in direct_children if _local_name(child.tag).lower() == "title"]
    descriptions = [child for child in direct_children if _local_name(child.tag).lower() == "desc"]
    if not titles or not "".join(titles[0].itertext()).strip():
        raise ValueError("inline SVG requires a non-empty title")
    if not descriptions or not "".join(descriptions[0].itertext()).strip():
        raise ValueError("inline SVG requires a non-empty description")
    native_text = [
        element
        for element in root.iter()
        if _local_name(element.tag).lower() in {"text", "tspan"}
        and "".join(element.itertext()).strip()
    ]
    if not native_text:
        raise ValueError("inline SVG requires native text or tspan labels")

    prefix = f"mmd-{diagram_id}--"
    ids: set[str] = set()
    for element in root.iter():
        value = element.get("id")
        if value is None:
            continue
        if value in ids:
            raise ValueError(f"duplicate SVG id: {value}")
        if not value.startswith(prefix):
            raise ValueError(f"SVG id lacks diagram prefix {prefix!r}: {value}")
        ids.add(value)
    missing = sorted(_collect_references(root) - ids)
    if missing:
        raise ValueError(f"dangling SVG reference(s): {missing}")


def render_mermaid_figures(
    template_html: str,
    config_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
) -> str:
    rendered = template_html
    figures = extract_mermaid_figures(template_html)
    for figure in reversed(figures):
        raw_svg = render_mermaid_source(
            figure.source,
            figure.diagram_id,
            config_path,
            output_dir,
        )
        svg = sanitize_and_prefix_svg(
            raw_svg,
            figure.diagram_id,
            figure.title,
            figure.description,
        )
        replacement = f'<div class="fw">\n{svg.rstrip()}\n</div>'
        rendered = rendered[:figure.source_start] + replacement + rendered[figure.source_end:]
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render and validate template Mermaid figures")
    parser.add_argument("--template", type=pathlib.Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    template = args.template.read_text(encoding="utf-8")
    figures = extract_mermaid_figures(template)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for figure in figures:
        raw = render_mermaid_source(figure.source, figure.diagram_id, args.config, args.output_dir)
        svg = sanitize_and_prefix_svg(raw, figure.diagram_id, figure.title, figure.description)
        path = args.output_dir / f"{figure.diagram_id}.svg"
        path.write_text(svg, encoding="utf-8", newline="\n")
        print(f"rendered {figure.diagram_id}: {path}")
    print(f"validated {len(figures)} Mermaid figure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
