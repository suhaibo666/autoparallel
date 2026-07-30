# docs/target-design-v2/tools/gen_diagrams.py
"""一份几何描述 → 同时产出 `.excalidraw`(可编辑源) 与内联 SVG(HTML 用)。

存在的理由：图有两个消费者(excalidraw 编辑 / 网页阅读)，两边手画必然漂移。
故几何只写一次，两个 emitter 各自渲染。SVG 用 CSS class 上色，由页面主题驱动明暗。
"""
from __future__ import annotations

import json
import os
import re

OUT_EXC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diagrams")
OUT_SVG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "svg")

# ── 调色：语义色板（excalidraw 用具体色值，SVG 用 class）────────────────────────
PALETTE = {
    "input":   ("#e9ecef", "#495057"),   # 输入/外部事实
    "sem":     ("#e7f5ff", "#1971c2"),   # 语义层
    "ir":      ("#d3f9d8", "#2f9e44"),   # 不可变 IR
    "plan":    ("#fff3bf", "#e8590c"),   # 计划层
    "backend": ("#f3d9fa", "#9c36b5"),   # 后端
    "block":   ("#ffe3e3", "#c92a2a"),   # 阻断
    "note":    ("#f8f9fa", "#868e96"),   # 旁注
    "derived": ("#e5dbff", "#6741d9"),   # 派生层
}
FRAME_STROKE = "#adb5bd"
TEXT = "#1e1e1e"


def _tw(s: str, size: float) -> float:
    """文本像素宽估算：CJK/全角 1.0em，其余 0.55em。"""
    w = 0.0
    for ch in s:
        w += 1.0 if ord(ch) > 0x2E7F else 0.55
    return w * size


#: 分词：单个非 ASCII 字符（CJK 处处可断）| 一段 ASCII 可见字符（拉丁词不许断开）| 空白。
#: **必须覆盖全部字符**——早先的 `[\x00-\x7F]+?(?:\s+|$)` 在「ASCII 串紧跟中文标点」处
#: 匹配失败，`findall` 会直接跳过那段 → **静默丢字**（实测把「同 SemanticId、新 NodeId」
#: 渲染成「同 、新 、」）。故下面的断言恒开。
_TOK = re.compile(r"[^\x00-\x7F]|[!-~]+|[ \t]+")


def _wrap(text: str, size: float, avail: float) -> list:
    """按估算宽度折行；显式 `\\n` 恒为硬换行。不丢字（由断言保证）。"""
    out = []
    for hard in text.split("\n"):
        if _tw(hard, size) <= avail:
            out.append(hard)
            continue
        toks = _TOK.findall(hard)
        assert "".join(toks) == hard, f"分词丢字: {hard!r}"
        cur = ""
        for tok in toks:
            if cur and _tw(cur + tok, size) > avail:
                out.append(cur.rstrip())
                cur = tok if tok.strip() else ""
            else:
                cur += tok
        if cur.strip():
            out.append(cur.rstrip())
    # 防丢字总校验（忽略折行处被吃掉的空格）
    a = re.sub(r"\s", "", "".join(out))
    b = re.sub(r"\s", "", text)
    assert a == b, f"折行丢字:\n  in : {b!r}\n  out: {a!r}"
    return out


class D:
    """极小图元集合：box / arrow / label / frame。坐标单位 = px，原点左上。"""

    def __init__(self, name: str, title: str, w: int, h: int):
        self.name, self.title, self.w, self.h = name, title, w, h
        self.items: list = []
        self._n = 0

    def _id(self, p="e"):
        self._n += 1
        return f"{self.name}-{p}{self._n}"

    def frame(self, x, y, w, h, label="", tone="note"):
        self.items.append(dict(k="frame", x=x, y=y, w=w, h=h, label=label, tone=tone,
                               id=self._id("f")))
        return self.items[-1]

    def box(self, x, y, w, h, label, tone="sem", sub="", mono=False):
        self.items.append(dict(k="box", x=x, y=y, w=w, h=h, label=label, sub=sub,
                               tone=tone, mono=mono, id=self._id("b")))
        return self.items[-1]

    def arrow(self, pts, label="", dashed=False, tone="note", lpos=None):
        """`lpos`：显式标签位置。多段折线的形心常落在别的图元上，此时必须显式给。"""
        self.items.append(dict(k="arrow", pts=pts, label=label, dashed=dashed,
                               tone=tone, lpos=lpos, id=self._id("a")))
        return self.items[-1]

    def label(self, x, y, text, size=13, anchor="start", tone="note", mono=False):
        self.items.append(dict(k="label", x=x, y=y, text=text, size=size,
                               anchor=anchor, tone=tone, mono=mono, id=self._id("t")))
        return self.items[-1]

    # ── emitters ────────────────────────────────────────────────────────────
    def to_excalidraw(self) -> dict:
        els, seed = [], [7]

        def nxt():
            seed[0] = (seed[0] * 1103515245 + 12345) & 0x7FFFFFFF
            return seed[0]

        def base(i, typ, x, y, w, h, stroke, bg, dashed=False):
            return {
                "id": i, "type": typ, "x": float(x), "y": float(y),
                "width": float(w), "height": float(h), "angle": 0,
                "strokeColor": stroke, "backgroundColor": bg, "fillStyle": "solid",
                "strokeWidth": 2, "strokeStyle": "dashed" if dashed else "solid",
                "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
                "roundness": {"type": 3} if typ == "rectangle" else None,
                "seed": nxt(), "version": 1, "versionNonce": nxt(),
                "isDeleted": False, "boundElements": None, "updated": 1,
                "link": None, "locked": False,
            }

        def bound_text(cid, txt, x, y, w, h, size=16, mono=False, color=TEXT):
            tid = cid + "-t"
            n = txt.count("\n") + 1
            return tid, {
                **base(tid, "text", x + 6, y + max(0, (h - size * 1.25 * n) / 2),
                       w - 12, size * 1.25 * n, color, "transparent"),
                "roundness": None,
                "text": txt, "fontSize": size, "fontFamily": 3 if mono else 2,
                "textAlign": "center", "verticalAlign": "middle",
                "containerId": cid, "originalText": txt, "lineHeight": 1.25,
            }

        for it in self.items:
            if it["k"] == "frame":
                e = base(it["id"], "rectangle", it["x"], it["y"], it["w"], it["h"],
                         FRAME_STROKE, "transparent", dashed=True)
                els.append(e)
                if it["label"]:
                    els.append({
                        **base(it["id"] + "-l", "text", it["x"] + 10, it["y"] - 24,
                               260, 20, PALETTE[it["tone"]][1], "transparent"),
                        "roundness": None, "text": it["label"], "fontSize": 16,
                        "fontFamily": 2, "textAlign": "left", "verticalAlign": "top",
                        "containerId": None, "originalText": it["label"],
                        "lineHeight": 1.25,
                    })
            elif it["k"] == "box":
                bg, st = PALETTE[it["tone"]]
                e = base(it["id"], "rectangle", it["x"], it["y"], it["w"], it["h"], st, bg)
                txt = it["label"] + (("\n" + it["sub"]) if it["sub"] else "")
                tid, te = bound_text(it["id"], txt, it["x"], it["y"], it["w"], it["h"],
                                     size=15, mono=it["mono"])
                e["boundElements"] = [{"id": tid, "type": "text"}]
                els.append(e)
                els.append(te)
            elif it["k"] == "arrow":
                pts = it["pts"]
                x0, y0 = pts[0]
                rel = [[float(px - x0), float(py - y0)] for px, py in pts]
                xs = [p[0] for p in rel]
                ys = [p[1] for p in rel]
                e = base(it["id"], "arrow", x0, y0,
                         max(xs) - min(xs), max(ys) - min(ys),
                         PALETTE[it["tone"]][1], "transparent", dashed=it["dashed"])
                e.update({"roundness": {"type": 2}, "points": rel,
                          "lastCommittedPoint": None, "startBinding": None,
                          "endBinding": None, "startArrowhead": None,
                          "endArrowhead": "arrow"})
                els.append(e)
                if it["label"]:
                    if it.get("lpos"):
                        mx, my = it["lpos"]
                    else:
                        mx = sum(p[0] for p in pts) / len(pts)
                        my = sum(p[1] for p in pts) / len(pts)
                    els.append({
                        **base(it["id"] + "-l", "text", mx + 6, my - 20, 200, 18,
                               PALETTE[it["tone"]][1], "transparent"),
                        "roundness": None, "text": it["label"], "fontSize": 13,
                        "fontFamily": 2, "textAlign": "left", "verticalAlign": "top",
                        "containerId": None, "originalText": it["label"],
                        "lineHeight": 1.25,
                    })
            elif it["k"] == "label":
                els.append({
                    **base(it["id"], "text", it["x"], it["y"], 340,
                           it["size"] * 1.3 * (it["text"].count("\n") + 1),
                           PALETTE[it["tone"]][1], "transparent"),
                    "roundness": None, "text": it["text"], "fontSize": it["size"],
                    "fontFamily": 3 if it["mono"] else 2,
                    "textAlign": it["anchor"] if it["anchor"] != "middle" else "center",
                    "verticalAlign": "top", "containerId": None,
                    "originalText": it["text"], "lineHeight": 1.3,
                })
        return {"type": "excalidraw", "version": 2,
                "source": "pynative-cost-evaluator/docs/target-design-v2",
                "elements": els, "files": {},
                "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"}}

    def to_svg(self) -> str:
        def esc(s):
            return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

        o = [f'<svg class="dg" viewBox="0 0 {self.w} {self.h}" '
             f'xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="{esc(self.title)}">',
             '<defs>',
             '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
             'markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" class="dg-ah"/></marker>',
             '</defs>']
        for it in self.items:
            if it["k"] == "frame":
                o.append(f'<rect class="dg-frame" x="{it["x"]}" y="{it["y"]}" '
                         f'width="{it["w"]}" height="{it["h"]}" rx="10"/>')
                if it["label"]:
                    o.append(f'<text class="dg-ftitle t-{it["tone"]}" x="{it["x"] + 12}" '
                             f'y="{it["y"] - 8}">{esc(it["label"])}</text>')
            elif it["k"] == "box":
                cx = it["x"] + it["w"] / 2
                o.append(f'<rect class="dg-box b-{it["tone"]}" x="{it["x"]}" y="{it["y"]}" '
                         f'width="{it["w"]}" height="{it["h"]}" rx="8"/>')
                # 框内文本按框宽折行（主标 15px / 副标 12px），否则会溢出框体。
                lines = [(t, 15, True) for t in _wrap(it["label"], 15, it["w"] - 16)]
                if it["sub"]:
                    lines += [(t, 12, False) for t in _wrap(it["sub"], 12, it["w"] - 14)]
                total_h = sum(15 if b else 14 for _t, _s, b in lines)
                y = it["y"] + (it["h"] - total_h) / 2 + 12
                for ln, sz, is_main in lines:
                    cls = "dg-blabel" if is_main else "dg-bsub"
                    mono = ' dg-mono' if (it["mono"] and is_main) or not is_main else ''
                    o.append(f'<text class="{cls}{mono} t-{it["tone"]}" x="{cx}" '
                             f'y="{y}" font-size="{sz}">{esc(ln)}</text>')
                    y += 15 if is_main else 14
            elif it["k"] == "arrow":
                pts = " ".join(f"{x},{y}" for x, y in it["pts"])
                dash = ' dg-dash' if it["dashed"] else ''
                o.append(f'<polyline class="dg-arrow a-{it["tone"]}{dash}" points="{pts}" '
                         f'marker-end="url(#ah)"/>')
                if it["label"]:
                    if it.get("lpos"):
                        mx, my = it["lpos"]
                    else:
                        mx = sum(p[0] for p in it["pts"]) / len(it["pts"])
                        my = sum(p[1] for p in it["pts"]) / len(it["pts"])
                    o.append(f'<text class="dg-alabel t-{it["tone"]}" x="{mx + 8}" '
                             f'y="{my - 6}">{esc(it["label"])}</text>')
            elif it["k"] == "label":
                anch = {"start": "start", "middle": "middle", "end": "end"}[it["anchor"]]
                mono = ' dg-mono' if it["mono"] else ''
                # SVG text 不换行 → 按估算宽度折行（CJK≈1.0em、拉丁≈0.55em），
                # 否则长断言会横向溢出 viewBox。
                avail = (self.w - it["x"] - 30) if anch == "start" else self.w * 0.9
                for i, ln in enumerate(_wrap(it["text"], it["size"], avail)):
                    o.append(f'<text class="dg-note{mono} t-{it["tone"]}" x="{it["x"]}" '
                             f'y="{it["y"] + 14 + i * (it["size"] + 4)}" '
                             f'text-anchor="{anch}" font-size="{it["size"]}">'
                             f'{esc(ln)}</text>')
        o.append("</svg>")
        return "\n".join(o)


# ══════════════════════════════════════════════════════════════════════════════
# D1 — 总体分层与数据流
# ══════════════════════════════════════════════════════════════════════════════
def d1():
    g = D("d1", "总体分层与数据流", 1180, 1010)
    CX, BW, BH = 430, 320, 56
    rows = [
        (30,  "S0 输入", "input", [
            ("MindFormers 源码", "只读，不执行"),
        ]),
    ]
    # 逐层：(y, 层名, tone, 主产物, 副标)
    layers = [
        (40,  "S1 语义层", "sem",     "TypedRegistrySnapshot", "op 语义词典（冻结、带 digest）"),
        (170, "S2 结构层", "ir",      "CoreIR",                "逻辑事实：op / 值 / 存储关系"),
        (300, "S3 反向层", "derived", "BackwardIR",            "前向+反向+重算，同一节点类型"),
        (430, "S4 布局层", "plan",    "PlacementPlan",         "在哪：mesh / shard / local shape"),
        (560, "S5 执行层", "plan",    "ExecutionPlan",         "何时：event / lifetime / alloc-free"),
    ]
    for y, name, tone, art, sub in layers:
        g.frame(70, y, 1040, 92, name, tone)
        g.box(CX, y + 18, BW, BH, art, tone, sub=sub, mono=True)

    # 输入侧。**注意**：源码喂的是 S2 的 DraftGraph，不是 S1 的注册快照 —— 两者是
    # 「结构」与「语义」两个正交事实源（§4.2 的二分），画反了会让整张图的论点失效。
    g.box(90, 58, 250, 40, "Native / User / Patch", "input", sub="三个物理隔离注册域")
    g.box(820, 58, 250, 40, "FrameworkRuntimeSnapshot", "input", sub="版本参与 digest")
    g.box(90, 188, 250, 40, "源码 → DraftGraph", "input", sub="符号 / 数据流 / attrs 字面值")
    g.box(820, 318, 250, 40, "RecomputeSpec", "input", sub="重算策略（派生输入）")
    g.box(820, 448, 250, 40, "ParallelConfig", "input", sub="tp / pp / ep / cp / dp")
    g.box(820, 578, 250, 40, "BufferCalibration", "input", sub="measured 量的载体")

    # 主干箭头
    for y0, y1 in ((114, 158), (244, 288), (374, 418), (504, 548), (634, 690)):
        g.arrow([(CX + BW / 2, y0), (CX + BW / 2, y1)])
    # 侧向注入
    g.arrow([(340, 78), (CX, 78)], tone="input")
    g.arrow([(820, 78), (CX + BW, 78)], tone="input")
    g.arrow([(340, 208), (CX, 208)], tone="input")
    g.arrow([(820, 338), (CX + BW, 338)], tone="input")
    g.arrow([(820, 468), (CX + BW, 468)], tone="input")
    g.arrow([(820, 598), (CX + BW, 598)], tone="input")

    # S6 后端
    g.frame(70, 690, 1040, 120, "S6 只读后端", "backend")
    g.box(150, 716, 330, 72, "MemorySimulator", "backend",
          sub="allocator / liveness / OOM")
    g.box(700, 716, 330, 72, "TimeSimulator", "backend",
          sub="cost / contention / pipeline DES")
    g.arrow([(CX + 40, 634), (315, 700), (315, 712)])
    g.arrow([(CX + BW - 40, 634), (865, 700), (865, 712)])
    g.label(590, 740, "互不 import\n不回写上游", 13, "middle", "block")

    # S7 报告
    g.frame(70, 838, 1040, 96, "S7 报告", "note")
    g.box(CX, 862, BW, 52, "UnifiedReport", "note",
          sub="结果 + provenance + confidence", mono=True)
    g.arrow([(315, 788), (CX, 880)])
    g.arrow([(865, 788), (CX + BW, 880)])

    g.label(90, 952, "不变量：每层只增加一类信息；下游只读上游，从不回写。"
                     "改并行配置只失效 S4/S5，CoreIR 不动（what-if 的交互性由此保证）。", 14)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D2 — 知识分级（provenance）
# ══════════════════════════════════════════════════════════════════════════════
def d2():
    g = D("d2", "知识分级：量的 provenance 而非层的退化档", 1160, 700)
    cols = [
        (60,  "source_derived", "ir",
         ["shape / dtype", "alias / inplace", "saved 结构", "数据流与依赖"],
         "源码 + 算子定义\n可判定", "缺失 ⇒ 阻断"),
        (450, "measured", "plan",
         ["workspace_bytes", "bwd_scratch", "kernel duration", "allocator 参数"],
         "只能实测标定\n源码里不存在", "缺 key ⇒ 阻断\n命中 ⇒ 带 confidence"),
        (840, "assumed", "backend",
         ["MoE 每专家负载", "专家 capacity", "非规则化 placement"],
         "需要建模假设\n值依赖运行期数据", "必须显式声明\n缺声明 ⇒ 阻断"),
    ]
    for x, name, tone, items, why, rule in cols:
        g.frame(x, 60, 280, 470, name, tone)
        g.label(x + 16, 74, why, 13, "start", tone)
        for i, s in enumerate(items):
            g.box(x + 20, 132 + i * 52, 240, 40, s, tone, mono=True)
        g.box(x + 20, 132 + 4 * 52 + 14, 240, 60, "判据", "block", sub=rule)

    g.label(60, 18, "同一条契约字段 provenance 贯穿 S1→S7；分界线是"
                    "「源码可判定 / 不可判定」，横穿内存与时间两侧。", 15)
    g.box(390, 578, 380, 56, "UnifiedReport", "note",
          sub="按 provenance 分别统计覆盖率与 confidence", mono=True)
    for x in (200, 590, 980):
        g.arrow([(x, 530), (x, 560), (580, 560), (580, 574)])
    g.label(60, 660, "关键：assumed 不是 fallback。未声明仍然阻断；"
                     "声明了才放行，且 basis 与方向（上界/下界/标称）进报告。", 14)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D3 — 语义层：三注册域 → 冻结快照 → 绑定
# ══════════════════════════════════════════════════════════════════════════════
def d3():
    g = D("d3", "语义层：注册域、DSL 编译与调用绑定", 1160, 620)
    g.frame(60, 60, 300, 250, "三个物理隔离注册域", "sem")
    g.box(80, 92,  260, 44, "NativeOpRegistry", "sem", sub="随版本发布，只读", mono=True)
    g.box(80, 152, 260, 44, "UserOpRegistry", "sem", sub="只能新增 native 未覆盖", mono=True)
    g.box(80, 212, 260, 44, "NativePatchRegistry", "sem",
          sub="整条替换 + hash 门禁", mono=True)

    g.box(440, 92, 260, 44, "RegistryLoader", "sem", sub="冲突 / 版本 / 完整性")
    g.box(440, 168, 260, 44, "DSL 编译", "sem", sub="parse → typecheck → 模板展开")
    g.box(440, 244, 260, 60, "TypedRegistrySnapshot", "ir",
          sub="不可变 + digest", mono=True)
    g.box(790, 92, 300, 44, "FrameworkRuntimeSnapshot", "input",
          sub="framework / runtime 版本", mono=True)
    g.box(790, 168, 300, 44, "TemplateLibrary", "input", sub="固定模板，非第二执行路径")

    g.arrow([(340, 114), (436, 114)])
    g.arrow([(570, 140), (570, 164)])
    g.arrow([(570, 216), (570, 240)])
    g.arrow([(790, 114), (704, 114)], tone="input")
    g.arrow([(790, 190), (704, 190)], tone="input")

    g.frame(60, 360, 1030, 230, "调用绑定（快照冻结后，不再解析 YAML、不再展开模板）", "ir")
    g.box(90, 402, 210, 48, "原子源码调用", "input", sub="symbol + inputs + attrs")
    g.box(350, 402, 190, 48, "native 命中？", "sem")
    g.box(350, 500, 190, 48, "user 命中？", "sem")
    g.box(610, 402, 190, 48, "有有效 patch？", "sem")
    g.box(860, 402, 190, 48, "ResolvedCall", "ir",
          sub="definition + attrs + provenance", mono=True)
    g.box(610, 500, 190, 48, "E_OP_UNREGISTERED", "block", sub="阻断 + 生成骨架", mono=True)

    g.arrow([(300, 426), (346, 426)])
    g.arrow([(540, 426), (606, 426)], label="是")
    g.arrow([(800, 426), (856, 426)])
    g.arrow([(445, 450), (445, 496)], label="否")
    g.arrow([(540, 524), (606, 524)], label="否")
    # user 命中 → 从**底边**绕行进 ResolvedCall，避免与「否」分支同点出发。
    # 折线形心会落在 E_OP_UNREGISTERED 上，故标签位置显式给。
    g.arrow([(445, 548), (445, 570), (960, 570), (960, 454)], tone="sem",
            label="是（user 绑定）", lpos=(660, 590))
    g.label(806, 396, "patch 命中则优先", 12, "middle", "sem")
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D4 — 结构层与反向层：source 给结构、registry 给语义；重算嵌进反向图
# ══════════════════════════════════════════════════════════════════════════════
def d4():
    g = D("d4", "结构层与反向层", 1160, 780)
    g.frame(60, 60, 1030, 190, "S2 结构层：二分法", "ir")
    g.box(90, 100, 240, 48, "SourceFrontend", "input", sub="AST 内联可展开调用")
    g.box(90, 172, 240, 48, "DraftGraph", "input",
          sub="symbol / 数据流 / attrs 字面值", mono=True)
    g.box(430, 136, 240, 48, "SemanticResolver", "sem")
    g.box(430, 60 + 8, 240, 40, "TypedRegistrySnapshot", "ir",
          sub="(输入shape, attrs) → 语义", mono=True)
    g.box(780, 136, 280, 60, "CoreIR", "ir",
          sub="OpInstance / TensorValue / Storage", mono=True)
    g.arrow([(210, 148), (210, 168)])
    g.arrow([(330, 196), (426, 170)])
    g.arrow([(550, 108), (550, 132)], tone="ir")
    g.arrow([(670, 160), (776, 160)])
    g.label(90, 232, "源码只给三样：符号、数据流、attrs 字面值。"
                     "注册表给全部语义。两者不重叠 ⇒ 结构上不可能出现「两个真相源」。", 13)

    g.frame(60, 300, 1030, 300, "S3 反向层：BackwardIR（派生，CoreIR 保持 policy-free）", "derived")
    # 前向链（CoreIR）
    fy = 352
    for nm, x in (("op1", 100), ("op2", 260), ("op3", 420)):
        g.box(x, fy, 120, 42, nm, "ir", mono=True)
    for x0, x1 in ((220, 256), (380, 416)):
        g.arrow([(x0, fy + 21), (x1, fy + 21)], tone="ir")
    g.label(100, fy - 32, "CoreIR 前向", 13, "start", "ir")

    # 反向链：右→左。重算节点**就在链上**（op3′ → op2ʳ → op2′ → op1′），
    # 不是旁挂 —— 这正是「三类节点同构、无独立 lowering 路径」的图示。
    by = 480
    g.box(100, by, 120, 42, "op1′", "derived", mono=True)
    g.box(260, by, 120, 42, "op2′", "derived", mono=True)
    g.box(420, by, 150, 42, "op2ʳ", "plan", sub="phase=recompute", mono=True)
    g.box(610, by, 120, 42, "op3′", "derived", mono=True)
    for x0, x1 in ((610, 574), (420, 384), (260, 224)):
        g.arrow([(x0, by + 21), (x1, by + 21)], tone="derived")
    g.label(100, by + 58, "BackwardIR 反向（右→左）", 13, "start", "derived")
    # 复制关系
    g.arrow([(320, fy + 42), (320, 442), (495, 442), (495, by - 4)],
            dashed=True, tone="plan", label="复制")

    g.box(790, 396, 280, 128, "统一节点类型", "ir",
          sub="前向 / 反向 / 重算 三类节点同构，\n共用同一套形状·类型·布局·资源机器")
    g.label(610, by + 58, "重算节点 = 前向 op2 的副本：同 SemanticId、新 NodeId、"
                          "origin 回指。插在该区域反向节点之前。", 13, "start", "plan")

    g.label(60, 640, "为什么 BackwardIR 是派生层而不是写进 CoreIR：CoreIR 必须 policy-free。"
                     "否则改一次重算配置就失效 CoreIR 缓存 —— 而它是最贵的一层。", 14)
    g.label(60, 686, "净效果是删类：AutogradContract / GradientValueSpec / "
                     "GradientAccumulationSpec（fan-in 累加本来就是个 add 节点）"
                     "全部退化为图里的普通节点与边。", 14)
    g.label(60, 732, "不变量：BackwardIR 只派生 CoreIR，不回写；RecomputeSpec 是它的输入，"
                     "digest 独立。", 14)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D5 — 四层身份
# ══════════════════════════════════════════════════════════════════════════════
def d5():
    g = D("d5", "四层身份：为什么名称去重必须废掉", 1160, 640)
    rows = [
        (70,  "TensorId", "ir", "逻辑值身份（一次赋值 = 一个 ID）",
         "同名重绑 ⇒ 新 ID。名称不能推断物理复用"),
        (190, "StorageId", "ir", "逻辑存储 / alias 等价类",
         "alias / view / inplace 归并到同一 root"),
        (310, "StoragePlacementId", "plan", "rank / stage 上的放置模板",
         "同一逻辑存储在不同 rank 是不同放置"),
        (430, "StorageInstanceId", "plan", "某次执行 lifetime 的物理分配",
         "唯一进入内存计数的身份；epoch 区分重算再物化"),
    ]
    for y, nm, tone, what, why in rows:
        g.box(70, y, 300, 60, nm, tone, sub=what, mono=True)
        g.label(400, y + 12, why, 14, "start", tone)
        if y < 430:
            g.arrow([(220, y + 60), (220, y + 108)])

    g.frame(70, 520, 1020, 84, "一个具体例子", "note")
    g.label(90, 536,
            "x = f(a)            → TensorId t1, StorageId s1\n"
            "y = x.view(...)     → TensorId t2, StorageId s1（alias，零 Allocate）\n"
            "x = g(y)            → TensorId t3, StorageId s2（同名，但是新逻辑值）",
            13, "start", "note", mono=True)
    g.label(70, 620, "今天按 tensor name 首见定型：t1/t3 会被当成同一个（少算），"
                     "t1/t2 会被当成两次分配（多算）。四层身份把这两类错误一起消掉。", 14)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D6 — 布局传播与执行事件
# ══════════════════════════════════════════════════════════════════════════════
def d6():
    g = D("d6", "布局传播与执行事件", 1160, 720)
    g.frame(60, 60, 1030, 228, "S4 布局：沿数据流传播 per-value 分片状态", "plan")
    chain = (("段入口 x", 90), ("Column", 300), ("act", 500), ("Row", 660), ("out", 870))
    for nm, x in chain:
        g.box(x, 120, 150, 44, nm, "plan", mono=True)
    for x0, x1 in ((240, 296), (450, 496), (650, 656), (810, 866)):
        g.arrow([(x0, 142), (x1, 142)], tone="plan")
    states = (('{"S": tp}', 90), ('{carrier: tp}', 300), ('{carrier: tp}', 500),
              ('{} + RS/AR', 660), ('{}', 870))
    for s, x in states:
        g.label(x + 75, 176, s, 12, "middle", "note", mono=True)
    g.box(300, 232, 150, 44, "注入 AG", "block", sub="入携 S")
    g.box(660, 232, 150, 44, "注入 RS / AR", "block", sub="sp / 非 sp")
    g.arrow([(375, 164), (375, 228)], tone="block")
    g.arrow([(735, 164), (735, 228)], tone="block")
    g.label(60, 296, "规则来自模块语义（在 Registry 的 placement 面），不是猜。"
                     "carrier 歧义、Column∘Column、Row 前无 Column ⇒ 阻断。", 13)

    g.frame(60, 360, 1030, 268, "S5 执行：生命周期 → 显式事件", "plan")
    tl = 420
    g.arrow([(110, tl + 120), (1050, tl + 120)], tone="note")
    ev = (("Allocate", 130), ("Bind", 270), ("Use", 400), ("Pin", 530),
          ("Retire", 700), ("Free", 850))
    for nm, x in ev:
        g.box(x, tl + 40, 130, 40, nm, "plan", mono=True)
        g.arrow([(x + 65, tl + 80), (x + 65, tl + 114)], tone="plan")
    g.label(130, tl - 8, "每个 StorageInstanceId 的事件序；非 persistent 恰好一次 "
                         "Allocate / Free，结束前 FinalAudit。", 13, "start", "plan")
    g.box(130, tl + 150, 400, 40, "MemoryEventView", "backend",
          sub="自包含 dependency / stream 偏序", mono=True)
    g.box(620, tl + 150, 400, 40, "ResourceRequest", "backend",
          sub="自包含 PricingDescriptor", mono=True)
    g.label(60, 664, "为什么两个视图都必须自包含：后端一旦回读上游，"
                     "profile / cache key 就随上游结构变化而失效，实测复用会静默失灵。", 14)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D7 — 未知 op 阻断闭环
# ══════════════════════════════════════════════════════════════════════════════
def d7():
    g = D("d7", "未知 op 的阻断与用户闭环", 1100, 470)
    st = ("DISCOVERED", "UNREGISTERED", "USER_DESCRIBED", "SCHEMA_VALIDATED",
          "SEMANTIC_VALIDATED", "COMPILED", "SIMULATED")
    x = 40
    for i, s in enumerate(st):
        tone = "block" if i == 1 else ("ir" if i >= 5 else "sem")
        g.box(x, 120, 130, 52, s, tone, mono=True)
        if i < len(st) - 1:
            g.arrow([(x + 130, 146), (x + 148, 146)])
        x += 148
    g.label(40, 76, "任一步失败都不进入下一状态；不产生部分计划。", 14)

    g.box(188, 230, 400, 60, "scaffold 生成骨架", "note",
          sub="只填可从调用证明的 selector / inputs / attrs，其余 REQUIRED 占位")
    g.arrow([(253, 172), (253, 226)], tone="block")
    g.box(640, 230, 420, 60, "cost-eval semantics validate / explain", "note",
          sub="用户补完 → digest 更新 → 缓存失效 → 重新编译", mono=True)
    g.arrow([(588, 260), (636, 260)])
    g.arrow([(850, 230), (850, 200), (484, 200), (484, 176)], dashed=True, tone="sem")

    g.label(40, 350, "阻断只针对「未注册 op」。已注册但含 measured / assumed 量的 ⇒ "
                     "放行，并在报告里逐项标注 —— 这两件事必须分开，", 14)
    g.label(40, 380, "否则 assumed 通道无处落脚。", 14)
    g.label(40, 424, "已知失效模式：用户为跑通而在骨架里填一个看似合理的数，"
                     "该数与真值不可区分。故 assumed 必须带 basis 且独立统计。", 14, "start", "block")
    return g


DIAGRAMS = [("01-layering", d1), ("02-provenance", d2), ("03-semantic-layer", d3),
            ("04-structure-backward", d4), ("05-identities", d5),
            ("06-placement-execution", d6), ("07-unknown-op-loop", d7)]


def main():
    os.makedirs(OUT_EXC, exist_ok=True)
    os.makedirs(OUT_SVG, exist_ok=True)
    for name, fn in DIAGRAMS:
        g = fn()
        with open(os.path.join(OUT_EXC, name + ".excalidraw"), "w", encoding="utf-8") as fh:
            json.dump(g.to_excalidraw(), fh, ensure_ascii=False, indent=1)
        with open(os.path.join(OUT_SVG, name + ".svg"), "w", encoding="utf-8") as fh:
            fh.write(g.to_svg())
        print(f"  {name:<26} {len(g.items):>3} 图元  {g.w}x{g.h}")
    print(f"\nexcalidraw → {OUT_EXC}\nsvg        → {OUT_SVG}")


if __name__ == "__main__":
    main()
