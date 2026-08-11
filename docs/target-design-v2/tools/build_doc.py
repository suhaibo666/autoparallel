# docs/target-design-v2/tools/build_doc.py
"""模板 + 构建期 Mermaid SVG + CSS → 两份 HTML。

替换的占位符：
  {{CSS}}            → <style> src/style.css </style>
  {{TOC}}            → 由文档标题**自动生成**的导航（不手维护，故不会漏章节）
  <figure class="mermaid-figure" ...> → 构建期渲染、校验并内联静态 SVG

另做两件构建期修正：
  1. **自动给缺 id 的 h2/h3 补 id**（由标题里的「N.M」推），保证每个小节可锚定；
  2. **消掉中文句中的源码换行**。模板为可读性在中文句子中间折行，而 HTML 会把换行
     折成一个空格 → 汉字之间凭空多出空隙（实测 72 处）。故在构建期把「CJK 换行 CJK」
     「CJK 换行 <tag>」「</tag> 换行 CJK」三种情形的换行去掉；中英之间的空格**保留**
     （那是中文排版里想要的）。pre / style / script / svg 区域一律不动。

产出两份，内容同源：
  index.html     独立文件（含 doctype/html/head + 主题切换），浏览器直接打开
  artifact.html  charset/title/style/正文（Artifact 发布用，宿主自带外壳）
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "index.template.html")
CSS = os.path.join(ROOT, "src", "style.css")
MERMAID_CONFIG = os.path.join(ROOT, "src", "mermaid.config.json")

if __package__:
    from .render_mermaid import extract_mermaid_figures, render_mermaid_figures
else:
    sys.path.insert(0, HERE)
    from render_mermaid import extract_mermaid_figures, render_mermaid_figures

#: CJK / 全角标点。用于判定「此处换行会不会在汉字间造出空隙」。
CJK = r"[⺀-鿿豈-﫿︰-﹏＀-￯　-〿]"
#: 构建期不得改动的区域（代码、样式、脚本、图）。
PROTECT = re.compile(r"<(pre|style|script|svg)\b.*?</\1>", re.S | re.I)


def collapse_cjk_newlines(html: str) -> tuple[str, int]:
    """去掉落在中文文本内部的源码换行。返回 (结果, 修正处数)。"""
    keep: list[str] = []

    def stash(m):
        keep.append(m.group(0))
        return "\x00%d\x00" % (len(keep) - 1)

    body = PROTECT.sub(stash, html)
    n = 0
    for pat in (CJK + r")\s*\n[ \t]*(" + CJK,       # 汉字 ↵ 汉字
                CJK + r")\s*\n[ \t]*(<",            # 汉字 ↵ <tag>
                r">)\s*\n[ \t]*(" + CJK):           # </tag> ↵ 汉字
        rx = re.compile("(" + pat + ")")
        while True:
            body, k = rx.subn(r"\1\2", body)
            n += k
            if not k:                                # 相邻多处需反复扫
                break
    body = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], body)
    return body, n


def _slug(text: str, fallback: str) -> str:
    """从「2.1 是什么」推出 `s2-1`；推不出用 fallback。"""
    m = re.match(r"\s*(\d+)\.(\d+)", text)
    return f"s{m.group(1)}-{m.group(2)}" if m else fallback


def assign_ids(html: str) -> tuple[str, int]:
    """给缺 id 的 h2/h3 补上 id（已有的一律不动，免得改锚点）。"""
    n = [0]

    def fix(m):
        tag, attr, inner = m.group(1), m.group(2), m.group(3)
        if re.search(r'\bid=', attr):
            return m.group(0)
        text = re.sub(r"<[^>]+>", "", inner).strip()
        n[0] += 1
        sid = _slug(text, f"{tag}-auto{n[0]}")
        return f'<{tag} id="{sid}"{attr}>{inner}</{tag}>'

    out = re.sub(r"<(h2|h3)((?:\s[^>]*)?)>(.*?)</\1>", fix, html, flags=re.S)
    return out, n[0]


def build_toc(html: str) -> str:
    """按文档里 h2/h3 的实际顺序生成导航。分部标签取 h2 的 `data-part`。"""
    rows: list[str] = []
    for m in re.finditer(r"<(h2|h3)((?:\s[^>]*)?)>(.*?)</\1>", html, re.S):
        tag, attr, inner = m.group(1), m.group(2), m.group(3)
        sid = re.search(r'id="([^"]+)"', attr).group(1)
        part = re.search(r'data-part="([^"]+)"', attr)
        # h2 的章号在 <span class="cn">Chapter N</span> 里，标题在其后
        cn = re.search(r'<span class="cn">\s*Chapter\s+(\d+)\s*</span>', inner)
        text = re.sub(r"<[^>]+>", "", inner).strip()
        if tag == "h2":
            title = re.sub(r"^\s*Chapter\s+\d+\s*", "", text).strip()
            num = cn.group(1) if cn else ""
            if part:
                rows.append(f'<li class="grp">{part.group(1)}</li>')
            rows.append(
                f'<li><a href="#{sid}"><span class="num">{num}</span>'
                f'<span class="ttl">{title}</span></a></li>')
        else:
            m2 = re.match(r"\s*([\d.]+)\s*(.*)$", text, re.S)
            num, title = (m2.group(1), m2.group(2).strip()) if m2 else ("", text)
            rows.append(
                f'<li class="sub"><a href="#{sid}" title="{num} {title}">'
                f'<span class="num">{num}</span>'
                f'<span class="ttl">{title}</span></a></li>')
    return ('<nav aria-label="目录">\n  <p class="navtitle">目录</p>\n  <ol>\n    '
            + "\n    ".join(rows) + "\n  </ol>\n</nav>")


WRAPPER_HEAD = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
"""

WRAPPER_TAIL = """
<button id="tt" aria-label="切换明暗主题" title="切换明暗主题">◐</button>
<style>
#tt{position:fixed;right:18px;bottom:18px;z-index:60;width:38px;height:38px;
  border:1px solid var(--rule);background:var(--panel);color:var(--ink2);
  font-size:16px;cursor:pointer}
#tt:hover{color:var(--accent);border-color:var(--accent)}
@media print{#tt{display:none}}
</style>
<script>
(function(){
  var r=document.documentElement, k='cev-doc-theme', s=localStorage.getItem(k);
  if(s){r.setAttribute('data-theme',s);}
  document.getElementById('tt').addEventListener('click',function(){
    var cur=r.getAttribute('data-theme');
    if(!cur){cur=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';}
    var nxt=cur==='dark'?'light':'dark';
    r.setAttribute('data-theme',nxt); localStorage.setItem(k,nxt);
  });
})();
</script>
</body>
</html>
"""


def _extract_document_title(template_html: str) -> tuple[str, str]:
    """Extract only the leading document title, never nested SVG titles."""
    match = re.match(r"\s*(<title>.*?</title>)\s*", template_html, re.S | re.I)
    if not match:
        raise ValueError("template must start with one document-level <title>")
    return match.group(1), template_html[match.end():]


def build_document(
    source_path: os.PathLike[str] | str = SRC,
    css_path: os.PathLike[str] | str = CSS,
    output_dir: os.PathLike[str] | str = ROOT,
    *,
    mermaid_config_path: os.PathLike[str] | str = MERMAID_CONFIG,
) -> tuple[pathlib.Path, pathlib.Path]:
    source = pathlib.Path(source_path)
    stylesheet = pathlib.Path(css_path)
    destination = pathlib.Path(output_dir)

    with source.open("r", encoding="utf-8") as fh:
        tpl = fh.read()

    tpl, n_ids = assign_ids(tpl)
    document_title, tpl = _extract_document_title(tpl)
    if tpl.count("{{CSS}}") != 1:
        raise ValueError("template must contain exactly one {{CSS}} placeholder")
    with stylesheet.open("r", encoding="utf-8") as fh:
        document_style = "<style>\n" + fh.read().rstrip() + "\n</style>"
    tpl = tpl.replace("{{CSS}}", "", 1)

    toc = build_toc(tpl)
    body = tpl.replace("{{TOC}}", toc)
    body = render_mermaid_figures(body, mermaid_config_path, destination)

    left = re.findall(r"\{\{[^}]*\}\}", body)
    if left:
        raise ValueError(f"unresolved template placeholder(s): {left[:5]}")

    body, n_nl = collapse_cjk_newlines(body)
    artifact_html = (
        '<meta charset="utf-8">\n'
        + document_title
        + "\n"
        + document_style
        + body
    )
    index_html = (
        WRAPPER_HEAD
        + document_title
        + "\n"
        + document_style
        + "\n</head>\n<body>\n"
        + body.strip()
        + WRAPPER_TAIL
    )

    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "artifact.html"
    index = destination / "index.html"
    artifact.write_text(artifact_html, encoding="utf-8", newline="\n")
    index.write_text(index_html, encoding="utf-8", newline="\n")

    build_document.last_stats = {
        "chapters": toc.count('<li><a'),
        "subsections": toc.count('class="sub"'),
        "assigned_ids": n_ids,
        "collapsed_newlines": n_nl,
        "diagrams": len(extract_mermaid_figures(tpl)),
    }
    return index, artifact


build_document.last_stats = {}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build target-design-v2 HTML documents")
    parser.add_argument("--source", type=pathlib.Path, default=pathlib.Path(SRC))
    parser.add_argument("--css", type=pathlib.Path, default=pathlib.Path(CSS))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path(ROOT))
    parser.add_argument(
        "--mermaid-config",
        type=pathlib.Path,
        default=pathlib.Path(MERMAID_CONFIG),
    )
    args = parser.parse_args(argv)
    index, artifact = build_document(
        args.source,
        args.css,
        args.output_dir,
        mermaid_config_path=args.mermaid_config,
    )

    stats = build_document.last_stats
    print(
        f"  导航自动生成：{stats['chapters']} 章 + {stats['subsections']} 小节"
        f"（补 id {stats['assigned_ids']} 个）"
    )
    print(f"  中文句中换行修正：{stats['collapsed_newlines']} 处")
    print(f"  Mermaid 静态图已内联：{stats['diagrams']}")
    for path in (index, artifact):
        print(f"  {path.name:<34} {path.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
