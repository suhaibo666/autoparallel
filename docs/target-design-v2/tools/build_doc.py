# docs/target-design-v2/tools/build_doc.py
"""模板 + 同源 SVG → 两份 HTML。

`{{SVG:<name>}}` 被替换成一个 <figure>：内联 SVG + 图题 + 可编辑源路径。
产出两份，内容同源：
  index.html     独立文件（含 doctype/html/head + 主题切换），浏览器直接打开
  artifact.html  仅 title/style/正文（Artifact 发布用，宿主自带外壳）
改图后重跑 gen_diagrams.py 再跑本脚本即同步，两边不会漂移。
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src", "index.template.html")
CSS = os.path.join(ROOT, "src", "style.css")
SVG_DIR = os.path.join(ROOT, "build", "svg")

sys.path.insert(0, HERE)
import gen_diagrams as G  # noqa: E402

TITLES = {name: fn().title for name, fn in G.DIAGRAMS}
FIGNO = {name: i + 1 for i, (name, _fn) in enumerate(G.DIAGRAMS)}


def figure(name: str) -> str:
    path = os.path.join(SVG_DIR, name + ".svg")
    with open(path, "r", encoding="utf-8") as fh:
        svg = fh.read()
    title = TITLES.get(name, name)
    return (f'<figure id="fig-{name}">\n<div class="fw">\n{svg}\n</div>\n'
            f'<figcaption><span><b>图 {FIGNO[name]}</b>　{title}</span>'
            f'<span class="src">diagrams/{name}.excalidraw</span></figcaption>\n</figure>')


WRAPPER_HEAD = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
"""

WRAPPER_TAIL = """
<button id="tt" aria-label="切换明暗主题" title="切换明暗主题">◐</button>
<style>
#tt{position:fixed;right:16px;bottom:16px;z-index:50;width:40px;height:40px;
  border-radius:50%;border:1px solid var(--line);background:var(--panel);
  color:var(--ink2);font-size:17px;cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.12)}
#tt:hover{color:var(--ink)}
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


def main():
    with open(SRC, "r", encoding="utf-8") as fh:
        tpl = fh.read()

    used, missing = [], []

    def sub(m):
        name = m.group(1)
        if not os.path.isfile(os.path.join(SVG_DIR, name + ".svg")):
            missing.append(name)
            return f'<p><b>[缺图 {name}]</b></p>'
        used.append(name)
        return figure(name)

    body = re.sub(r"\{\{SVG:([\w-]+)\}\}", sub, tpl)
    with open(CSS, "r", encoding="utf-8") as fh:
        body = body.replace("{{CSS}}", "<style>\n" + fh.read().rstrip() + "\n</style>")

    if missing:
        raise SystemExit(f"build_doc: 模板引用了不存在的图：{missing}（先跑 gen_diagrams.py）")
    left = re.findall(r"\{\{[^}]*\}\}", body)
    if left:
        raise SystemExit(f"build_doc: 未替换的占位符 {left[:5]}")

    # 1) Artifact 用：无外壳
    art = os.path.join(ROOT, "artifact.html")
    with open(art, "w", encoding="utf-8") as fh:
        fh.write(body)

    # 2) 独立文件：加外壳。<title>/<style> 从正文提到 head。
    head_parts = re.findall(r"<title>.*?</title>|<style>.*?</style>", body, re.S)
    rest = body
    for p in head_parts:
        rest = rest.replace(p, "", 1)
    std = WRAPPER_HEAD + "\n".join(head_parts) + "\n</head>\n<body>\n" \
        + rest.strip() + WRAPPER_TAIL
    idx = os.path.join(ROOT, "index.html")
    with open(idx, "w", encoding="utf-8") as fh:
        fh.write(std)

    unused = [n for n, _ in G.DIAGRAMS if n not in used]
    print(f"  图已内联 {len(used)}/{len(G.DIAGRAMS)}" + (f"  未用：{unused}" if unused else ""))
    for p in (idx, art):
        print(f"  {os.path.relpath(p, os.path.dirname(ROOT)):<34} "
              f"{os.path.getsize(p) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
