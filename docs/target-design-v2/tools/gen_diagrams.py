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
    # provenance 三色。**刻意与"层"色分开成另一类视觉对象**（描边不填色，
    # 与正文 `.pill` 同款）：层色表达"在哪一层"，provenance 表达"这个量算哪种知识"，
    # 两者横切。若共用填色块，读者会把「绿=CoreIR 层」误读成「绿=source_derived」。
    "sd":      ("transparent", "#1f7a3d"),
    "me":      ("transparent", "#b35309"),
    "as":      ("transparent", "#7c3aed"),
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
    """总体分层。左列 = 每层的**独有不变量**（G-L1 的断言点），右列 = 策略输入。

    这个左右分工不是排版选择：G-L1 说「一层存在当且仅当它是某条不变量唯一能被陈述的位置」，
    所以左列就是这张图要论证的东西本身。策略从右侧逐层注入，而 G0 右侧**必须空着** ——
    那是 policy-free 的视觉表达。
    """
    g = D("d1", "总体分层：左＝该层独有不变量（G-L1 断言点），右＝策略输入", 1180, 1046)
    CX, BW, BH = 392, 356, 46
    LX, LW = 62, 300           # 左列：独有不变量
    RX, RW = 782, 320          # 右列：策略输入

    # ── S0 事实源 ────────────────────────────────────────────────────────────
    g.frame(62, 26, 1040, 78, "S0 事实源（只有这五类产生新事实；表外字段 ⇒ BLK-UNCLASSIFIED）",
            "input")
    for i, (t, s) in enumerate([
            ("框架源码", "AST + 版本"), ("算子语义快照", "三注册域"),
            ("EnvFacts", "HBM / 设备"), ("CalibrationSet", "域片 + 实测点"),
            ("八份 *Spec", "使用者配置")]):
        g.box(76 + i * 206, 52, 196, 40, t, "input", sub=s)

    # ── IR 链 ────────────────────────────────────────────────────────────────
    layers = [
        (128, "G0",   "ir",      "CoreIR", "PE(Src, ModelSpec, EnvFacts, π₀)",
         "policy(·) ∉ roots\ndigest 在策略扰动下不变", None),
        (210, "G0.5", "sem",     "ImplIR", "impl_select ∘ fusion_logical ∘ precision_fwd",
         "op 集合自此定稿", "ImplSpec · PrecisionSpec"),
        (292, "G1",   "derived", "TrainIR", "Pass_train_step",
         "shape(grad_out[i]) == shape(in_i)\n每条梯度边恰一 producer", None),
        (374, "G1.7", "derived", "PrecIR", "Pass_precision_opt",
         "新增节点下游不含 grad_role\n持久字节守恒", "PrecisionSpec"),
        (456, "G2",   "plan",    "ShardIR", "Pass_shard",
         "placement 无失配\n桶数 = 真机归约 op 数",
         "ParallelSpec · PlacementAnnotation\nDistOptSpec"),
        (538, "G2.5", "plan",    "—", "Pass_fusion_dist",
         "融合前后通信节点语义多重集守恒", None),
        (620, "G3",   "plan",    "RematIR", "Pass_remat",
         "stage_role==remat ⇒ origin 是前向节点", "RematSpec"),
        (702, "G4",   "plan",    "SchedIR", "Pass_sched",
         "事件不重不漏、偏序无环\nroots(t_free)∋calib ⟺ timing_dependent",
         "ScheduleSpec · CalibrationSet"),
    ]
    for y, tag, tone, art, defn, inv, spec in layers:
        g.frame(62, y, 1040, 70, "", tone)
        title = tag if art in ("—", None) else f"{tag}　{art}"
        g.box(CX, y + 12, BW, BH, title, tone, sub=defn, mono=True)
        g.label(LX, y + 16, inv, 11.5, "start", tone)
        if spec:
            g.box(RX, y + 14, RW, 42, spec, "input", mono=True)
            g.arrow([(RX, y + 35), (CX + BW, y + 35)], tone="input")
        # 主干
        if y > 128:
            g.arrow([(CX + BW / 2, y - 12), (CX + BW / 2, y + 8)])
    g.arrow([(CX + BW / 2, 92), (CX + BW / 2, 124)], tone="input")

    g.label(LX, 108, "独有不变量（断言点）", 11, "start", "note", mono=True)
    g.label(RX, 108, "策略输入", 11, "start", "note", mono=True)
    g.label(RX, 146, "（G0 右侧空着 —— 这就是 policy-free）", 11.5, "start", "block")

    # ── 两后端 ───────────────────────────────────────────────────────────────
    g.frame(62, 800, 1040, 108, "两个只读后端：零新事实（不得引入新的 root 种类）", "backend")
    g.box(120, 828, 380, 62, "MemorySimulator", "backend",
          sub="liveness / allocator / peak 区间", mono=True)
    g.box(664, 828, 380, 62, "TimeSimulator", "backend",
          sub="定价 / 竞争 / 两级 DES", mono=True)
    g.arrow([(CX + 60, 772), (310, 812), (310, 824)])
    g.arrow([(CX + BW - 60, 772), (854, 812), (854, 824)])
    g.label(582, 862, "互不 import\n不回写上游", 12, "middle", "block")

    # ── 报告 ─────────────────────────────────────────────────────────────────
    g.frame(62, 928, 1040, 84, "报告", "note")
    g.box(CX, 950, BW, 50, "oom_verdict : Q3[Bool]", "note",
          sub="peak 只有区间形态；undetermined 附判定阻碍分解", mono=True)
    g.arrow([(310, 890), (CX, 968)])
    g.arrow([(854, 890), (CX + BW, 968)])
    g.label(76, 962, "显存链：无外部裁判\n（全部为自洽性检查）", 11.5, "start", "block")
    g.label(1040, 962, "时间链：op 级序列\n与时长为唯一 oracle", 11.5, "end", "sd")
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D2 — provenance 代数：两条正交轴 + readability 分派器
# ══════════════════════════════════════════════════════════════════════════════
def d2():
    """v2 把两件正交的事压成了一条三值枚举，这是全篇病灶。

    这张图的任务是让「正交」一眼可见：左轴回答**形状**（点还是区间），右轴回答**归因**
    （错了谁负责）。下方的 readability 分派器是把「人的判断」与「源码事实」分开的机器判据。
    """
    g = D("d2", "provenance 代数：certainty × roots 两条正交轴，由 readability 机器分派", 1180, 968)

    g.label(60, 16, "旧代数把 certainty 定义成 roots 的函数（「roots 必含标定才算 modeled」）。"
                    "这一条同时造成两处症状：roofline 被判成事实；通信字节的嗅探门恒不可满足。", 13.5)

    # ── 左轴：certainty ──────────────────────────────────────────────────────
    g.frame(60, 68, 470, 300, "轴一 · certainty —— 即使输入全对，我们施加的函数是不是真的那个函数",
            "note")
    g.label(78, 96, "决定**形状**：答案该是点还是区间", 12.5, "start", "note")
    for i, (nm, tone, sub) in enumerate([
            ("exact", "sd", "只有工具自带的 ~15 个算术原语"),
            ("modeled", "me", "经过一个复刻别人机制的算子"),
            ("assumed", "as", "取值依赖运行期数据")]):
        g.box(82, 128 + i * 72, 300, 52, nm, tone, sub=sub, mono=True)
    g.label(400, 150, "⊔ = max\n只升不降（P8）", 12, "start", "note")
    g.label(400, 232, "注册域拿不到\nexact（值域裁剪）", 12, "start", "block")
    g.label(400, 300, "⇒ direction\n必单侧（P6）", 12, "start", "as")

    # ── 右轴：roots ──────────────────────────────────────────────────────────
    g.frame(566, 68, 554, 300, "轴二 · roots —— 输入的叶子事实来自哪里", "note")
    g.label(584, 96, "决定**归因**：这条根错了，多少字节会变", 12.5, "start", "note")
    groups = [
        ("事实类", "sd", ["source", "configured", "transcribed", "calib"]),
        ("复刻类", "me", ["replication"]),
        ("人工判断类", "as", ["declared_semantics", "abstraction"]),
        ("数据依赖", "as", ["assumption"]),
    ]
    y = 126
    for gname, tone, members in groups:
        g.label(584, y + 4, gname, 11.5, "start", tone, mono=True)
        for j, m in enumerate(members):
            g.box(668 + j * 112, y, 106, 30, m, tone, mono=True)
        y += 44
    g.label(584, 306, "闭集八种；⊔ = ∪。按 root 聚合是**覆盖量不是划分**"
                      "（一个字节可有多个根）⇒ 报表禁止饼图。", 12, "start", "note")

    # ── readability 分派器 ───────────────────────────────────────────────────
    g.frame(60, 396, 1060, 214,
            "readability —— PE 在阻断时刻计算并写入，用户不可填（AST 在不在是客观事实）", "block")
    g.box(90, 434, 220, 44, "readability(target)", "block", mono=True)
    rows = [
        (492, "FULL", "PySub 可解析全部可达体", "abstraction", "强制 reconstructed"),
        (536, "PARTIAL", "入口可解析，体内有阻断站点", "abstraction（未求值站点）", "强制 reconstructed"),
        (580, "NONE", "无 AST（C 扩展 / 二进制 kernel）", "declared_semantics", "可为 calibrated"),
    ]
    g.label(96, 496, "readability", 11, "start", "note", mono=True)
    g.label(300, 496, "判据", 11, "start", "note", mono=True)
    g.label(660, 496, "分派的根", 11, "start", "note", mono=True)
    g.label(910, 496, "evidence.grade", 11, "start", "note", mono=True)
    for yy, nm, crit, root, grade in rows:
        g.label(96, yy + 24, nm, 12.5, "start", "block", mono=True)
        g.label(300, yy + 24, crit, 12.5, "start", "note")
        g.label(660, yy + 24, root, 12.5, "start", "as", mono=True)
        g.label(910, yy + 24, grade, 12.5, "start", "me", mono=True)
    g.arrow([(200, 478), (200, 508)], tone="block")
    g.label(90, 466, "self_certainty ≥ modeled，三档一律如此", 11.5, "start", "block")

    # ── P4 / P5 ──────────────────────────────────────────────────────────────
    g.frame(60, 638, 520, 290, "P4 存在位下传 —— 承重墙，一条律解三处", "sd")
    g.box(84, 676, 300, 40, "Prov(q) ⊒ Prov(exist(n))", "sd", mono=True)
    for i, t in enumerate([
            "被抽象节点的字节继承 abstraction 根",
            "通信节点的字节继承 replication 根",
            "overlap 下 Free 时刻带标定根 ⇒ 显存继承时间根"]):
        g.label(84, 740 + i * 46, "· " + t, 12.5, "start", "note")
    g.label(84, 886, "没有它，抽象只污染「这是什么算子」，"
                     "不污染「它产出多少字节」。", 12, "start", "block")

    g.frame(614, 638, 506, 290, "P5 own_residual 的四个来源 + 兜底", "me")
    for i, t in enumerate([
            "① G-B1a 抽象残差 hull(S_P, R_P)",
            "② 反向双候选 hull(a, b)",
            "③ 标定残差 residual_rel（grade=calibrated）",
            "④ declaration_unverified_bound"]):
        g.label(638, 682 + i * 34, t, 12.5, "start", "me", mono=True)
    g.box(638, 824, 458, 56, "兜底：无一适用 ⇒ 取单侧", "block",
          sub="**不得保持点值** —— 否则最不可信的那部分对区间宽度贡献为零")
    g.label(638, 906, "⇒ peak.hi 不可计算 ⇒ verdict = undetermined ⇒ 判定阻碍分解列出该标定谁",
            12, "start", "block")
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D3 — 语义层：三注册域 → 冻结快照 → 绑定
# ══════════════════════════════════════════════════════════════════════════════
def d3():
    g = D("d3", "语义层：注册域、DSL 编译与调用绑定", 1160, 620)
    g.frame(60, 60, 300, 250, "三个物理隔离注册域", "sem")
    g.box(80, 92,  260, 44, "NativeOpRegistry", "sem", sub="随版本发布，只读", mono=True)
    g.box(80, 152, 260, 44, "UserOpRegistry", "sem", sub="仅补充 Native 未覆盖项", mono=True)
    g.box(80, 212, 260, 44, "NativePatchRegistry", "sem",
          sub="完整替换 + hash 门禁", mono=True)

    g.box(440, 92, 260, 44, "RegistryLoader", "sem", sub="校验冲突、版本与完整性")
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
    g = D("d4", "G0 CoreIR 与 G1 TrainIR：源码给结构，注册域给语义", 1160, 780)
    g.frame(60, 60, 1030, 190, "G0 · CoreIR = PE(Src, ModelSpec, EnvFacts, π₀)", "ir")
    g.box(90, 100, 240, 48, "PE · 偏特化", "input", sub="在 π₀ 处内联并折叠 guard")
    g.box(90, 172, 240, 48, "π₀ 残差", "input",
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
    g.label(90, 232, "源码仅提供三类信息：符号、数据流、attrs 字面值。"
                     "注册表提供完整语义。两者不重叠，因此结构上不会出现「两个真相源」。", 13)

    g.frame(60, 300, 1030, 300, "G1 · TrainIR（反向）与 G3 · RematIR（重算副本）：三类节点同构", "derived")
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
    g.box(420, by, 150, 42, "op2ʳ", "plan", sub="stage_role=remat", mono=True)
    g.box(610, by, 120, 42, "op3′", "derived", mono=True)
    for x0, x1 in ((610, 574), (420, 384), (260, 224)):
        g.arrow([(x0, by + 21), (x1, by + 21)], tone="derived")
    g.label(100, by + 58, "TrainIR 反向（右→左）", 13, "start", "derived")
    # 复制关系
    g.arrow([(320, fy + 42), (320, 442), (495, 442), (495, by - 4)],
            dashed=True, tone="plan", label="复制")

    g.box(790, 396, 280, 128, "统一节点类型", "ir",
          sub="前向 / 反向 / 重算三类节点同构，\n共用同一套形状·类型·布局·资源计算机制")
    g.label(610, by + 58, "重算节点 = 前向 op2 的副本：同 SemanticId、新 NodeId、"
                          "origin 回指。插在该区域反向节点之前。", 13, "start", "plan")

    g.label(60, 640, "反向必须是派生层而不能写进 G0：G0 的定义就是 policy-free"
                     "（∀f: policy(·) ∉ f.roots），而重算范围是策略。这条不变量是 CoreIR "
                     "唯一的断言点 —— 写进去就没地方陈述它了（G-L1）。", 14)
    g.label(60, 686, "一组本来要写成「契约字段」的东西因此变成图里的普通节点与边："
                     "autograd 契约就是边，梯度值规格就是 TensorValue + StorageRelation，"
                     "fan-in 累加就是一个 add 节点。", 14)
    g.label(60, 732, "不变量：改写 pass 只读上游、从不回写；RematSpec 是 G3 的输入；"
                     "各层 digest 独立（G0 必须用内容 digest，否则其断言恒真）。", 14)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D5 — 四层身份
# ══════════════════════════════════════════════════════════════════════════════
def d5():
    g = D("d5", "四层身份：为什么必须禁止按名称去重", 1160, 640)
    rows = [
        (70,  "TensorId", "ir", "逻辑值身份（一次赋值 = 一个 ID）",
         "同名重新赋值 ⇒ 新 ID。名称不能推断物理复用"),
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
    g.label(70, 620, "若按 tensor name 采用首次出现时的定义，t1/t3 会被视为同一个值（少算），"
                     "t1/t2 会被视为两次分配（多算）。四层身份可以同时消除这两类错误。", 14)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D6 — 布局传播与执行事件
# ══════════════════════════════════════════════════════════════════════════════
def d6():
    g = D("d6", "G2 分片传播与 G4 执行事件", 1160, 720)
    g.frame(60, 60, 1030, 228, "G2 · Pass_shard：传播逐值 placement，失配处插 redistribute", "plan")
    chain = (("段入口 x", 90), ("Column", 300), ("act", 500), ("Row", 660), ("out", 870))
    for nm, x in chain:
        g.box(x, 120, 150, 44, nm, "plan", mono=True)
    for x0, x1 in ((240, 296), (450, 496), (650, 656), (810, 866)):
        g.arrow([(x0, 142), (x1, 142)], tone="plan")
    states = (('{"S": tp}', 90), ('{carrier: tp}', 300), ('{carrier: tp}', 500),
              ('{} + RS/AR', 660), ('{}', 870))
    for s, x in states:
        g.label(x + 75, 176, s, 12, "middle", "note", mono=True)
    g.box(300, 232, 150, 44, "注入 AG", "block", sub="输入携带 S")
    g.box(660, 232, 150, 44, "注入 RS / AR", "block", sub="sp / 非 sp")
    g.arrow([(375, 164), (375, 228)], tone="block")
    g.arrow([(735, 164), (735, 228)], tone="block")
    g.label(60, 296, "规则来自 S1 的 placement 面，而非推测。图上的「跑完无失配」"
                     "只是 Pass_shard 自己的终止条件 —— 它是后置条件不是门；规则表相对 "
                     "placement 格的完备性，由装载期的 G-P1 枚举检出。", 13)

    g.frame(60, 360, 1030, 268, "G4 · Pass_sched：生命周期 → 显式事件", "plan")
    tl = 420
    g.arrow([(110, tl + 120), (1050, tl + 120)], tone="note")
    ev = (("Allocate", 130), ("Bind", 270), ("Use", 400), ("Pin", 530),
          ("Retire", 700), ("Free", 850))
    for nm, x in ev:
        g.box(x, tl + 40, 130, 40, nm, "plan", mono=True)
        g.arrow([(x + 65, tl + 80), (x + 65, tl + 114)], tone="plan")
    g.label(130, tl - 8, "每个 StorageInstanceId 均有明确事件序；非 persistent 实例恰好执行一次 "
                         "Allocate / Free，并在结束前执行 FinalAudit。", 13, "start", "plan")
    g.box(130, tl + 150, 400, 40, "MemoryEventView", "backend",
          sub="自包含 dependency / stream 偏序", mono=True)
    g.box(620, tl + 150, 400, 40, "ResourceRequest", "backend",
          sub="自包含 PricingDescriptor", mono=True)
    g.label(60, 664, "两个视图都必须自包含：后端一旦回读上游，"
                     "profile / cache key 就会随上游结构变化而失效，实测画像复用也会静默失效。", 14)
    return g


# ══════════════════════════════════════════════════════════════════════════════
# D7 — 未知 op 阻断闭环
# ══════════════════════════════════════════════════════════════════════════════
def d7():
    g = D("d7", "未知 Op 的阻断与用户闭环", 1100, 470)
    st = ("DISCOVERED", "UNREGISTERED", "USER_\nDESCRIBED", "SCHEMA_\nVALIDATED",
          "SEMANTIC_\nVALIDATED", "COMPILED", "SIMULATED")
    x = 40
    for i, s in enumerate(st):
        tone = "block" if i == 1 else ("ir" if i >= 5 else "sem")
        g.box(x, 120, 130, 52, s, tone, mono=True)
        if i < len(st) - 1:
            g.arrow([(x + 130, 146), (x + 148, 146)])
        x += 148
    g.label(40, 76, "任一步失败都不进入下一状态；不产生部分计划。", 14)

    g.box(188, 230, 400, 60, "scaffold 生成骨架", "note",
          sub="只填写可从调用证明的 selector / inputs / attrs，其余字段保留 REQUIRED 占位")
    g.arrow([(253, 172), (253, 226)], tone="block")
    g.box(640, 230, 420, 60, "cost-eval semantics validate / explain", "note",
          sub="用户补充完整 → digest 更新 → 缓存失效 → 重新编译", mono=True)
    g.arrow([(588, 260), (636, 260)])
    g.arrow([(850, 230), (850, 200), (484, 200), (484, 176)], dashed=True, tone="sem")

    g.label(40, 350, "阻断只针对「未注册 Op」。已注册但含 measured / assumed 量的 Op 可以继续执行，"
                     "并在报告中逐项标注。两类情况必须区分，", 14)
    g.label(40, 380, "否则 assumed 将缺少合法的表达通道。", 14)
    g.label(40, 424, "已知失效模式：用户为使仿真继续执行而在骨架中填写一个表面合理的数值，"
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
