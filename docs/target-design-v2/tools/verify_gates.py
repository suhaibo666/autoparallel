# docs/target-design-v2/tools/verify_gates.py
"""从门表机器核对门数与切分，并断言正文里印出来的数与之一致。

存在的理由：这些数是 §10.5 强制上首页的字段，而**手数已经错过三次**
（26/40/43 各一次）。一个自己算错的首页数字，正是 `G-M1` 要消灭的那种虚假安全感，
所以它不能靠人数，必须有一个会失败的检查器。

用法：python tools/verify_gates.py        （不一致时退出码非 0）
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src", "index.template.html")

#: 独立性折扣：门 → 它归并到谁（共用 oracle 或是其实例）
NOT_INDEPENDENT = {"G-T2": "G-T1", "G-N2": "G-T1"}

#: oracle 分组。前缀不足以区分的（G-N* 分散在三组）在此显式指定。
GROUP_OF = {
    "G-P1": "快照自身", "G-P2": "快照自身", "G-P3": "快照自身", "G-N1": "快照自身",
    "G-T0": "真机 op 级 time", "G-T0b": "真机 op 级 time", "G-T1": "真机 op 级 time",
    "G-T2": "真机 op 级 time", "G-N2": "真机 op 级 time",
    "G-B0": "provenance 代数", "G-B1a": "provenance 代数", "G-B4": "provenance 代数",
    "G-R1": "实现与规模", "G-R2": "实现与规模", "G-R2b": "实现与规模",
    "G-R3": "实现与规模", "G-R4": "实现与规模", "G-N3": "实现与规模",
    "G-M1": "元判据", "G-M2a": "元判据", "G-M2b": "元判据", "G-Z1a": "元判据",
}

#: 组名 → 正文分布表里的行标签（正文带说明性后缀）
ROW_LABEL = {
    "快照自身": "快照自身（装载期）",
    "元判据": "元判据 / 项目级",
}

#: 检模型事实（其余为自律门）
CHECKS_MODEL_FACTS = {
    "G-P1", "G-P2", "G-C1", "G-C3", "G-D1", "G-D3", "G-D4", "G-D5", "G-E6",
    "G-B1a", "G-N2", "G-N5", "G-N6", "G-N7", "G-T1", "G-T2",
}


def group(gid: str) -> str:
    if gid in GROUP_OF:
        return GROUP_OF[gid]
    if gid.startswith("G-X"):
        return "schema 自身"
    return "图自身或框架源码文本"


def parse(html: str):
    """从 §12.2 的门表抽出 (编号, 类别)。展开 `G-N4–G-N7` 这种区间行。"""
    sec = html[html.index('id="c12-2"'):html.index('id="c12-3"')]
    rows = []
    for m in re.finditer(
        r'<tr><td class="nw">((?:<code>G-[\w\d]+</code>[/–]?)+)</td>(.*?)</tr>', sec, re.S
    ):
        ids = re.findall(r"<code>(G-[\w\d]+)</code>", m.group(1))
        tds = re.findall(r'<td class="nw">(.*?)</td>', m.group(2), re.S)
        kind = re.sub(r"<[^>]+>", "", tds[-1]).strip() if tds else "?"
        if "–" in m.group(1):                       # G-N4–G-N7 代表四道
            a, b = ids
            n = int(re.sub(r"\D", "", b)) - int(re.sub(r"\D", "", a)) + 1
            pre = re.match(r"(G-[A-Z]+)", a).group(1)
            ids = [f"{pre}{int(re.sub(r'\D', '', a)) + k}" for k in range(n)]
        rows.extend((i, kind) for i in ids)
    return rows


def main() -> int:
    html = open(SRC, encoding="utf-8").read()
    rows = parse(html)

    gates = [i for i, k in rows if "后置条件" not in k and "报告项" not in k]
    post = [i for i, k in rows if "后置条件" in k]
    rep = [i for i, k in rows if "报告项" in k]
    indep = [g for g in gates if g not in NOT_INDEPENDENT]

    by_group: dict[str, list[str]] = {}
    for g in gates:
        by_group.setdefault(group(g), []).append(g)

    facts = [g for g in gates if g in CHECKS_MODEL_FACTS]
    external = [g for g in gates if group(g) == "真机 op 级 time"]

    print(f"条目 {len(rows)}   真门 {len(gates)}   后置条件 {len(post)}   报告项 {len(rep)}")
    print(f"独立真门 {len(indep)}   （折扣：{', '.join(f'{k}→{v}' for k, v in NOT_INDEPENDENT.items())}）")
    print("\noracle 分布：")
    for k, v in sorted(by_group.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:<16} {len(v):>3}   {' '.join(sorted(v))}")
    print(f"\n检模型事实 {len(facts)} : 自律 {len(gates) - len(facts)}")
    print(f"外部 oracle {len(external)}；其中检模型事实 "
          f"{len([g for g in external if g in CHECKS_MODEL_FACTS])}；"
          f"其中相互独立 {len([g for g in external if g in indep and g in CHECKS_MODEL_FACTS])}")

    # ── 断言：正文里印出来的数必须与上面一致 ──────────────────────────────
    errs = []

    def want(pat: str, val: int, what: str):
        m = re.search(pat, html)
        if not m:
            errs.append(f"正文找不到「{what}」的印数")
        elif int(m.group(1)) != val:
            errs.append(f"{what}：正文印 {m.group(1)}，实际 {val}")

    want(r"共 <b>(\d+)</b> 个条目", len(rows), "条目总数")
    want(r"<b>真门 (\d+)</b>", len(gates), "真门数")
    want(r"<b>独立真门 (\d+)</b>", len(indep), "独立真门数")
    want(r'但"(\d+) 道门"这个数本身', len(gates), "标题里的门数")
    want(r"全部 (\d+) 道真门</b>上", len(gates), "切分的基数")
    for k, v in by_group.items():
        lbl = ROW_LABEL.get(k, k)
        want(rf'<tr><td class="nw">(?:<b>)?{re.escape(lbl)}(?:</b>)?</td>'
             rf'<td class="nw">(?:<b>)?(\d+)(?:</b>)?</td>', len(v), f"分布·{k}")

    tot = sum(len(v) for v in by_group.values())
    if tot != len(gates):
        errs.append(f"分布之和 {tot} ≠ 真门数 {len(gates)}")

    if errs:
        print("\n✗ 不一致：")
        for e in errs:
            print("   -", e)
        return 1
    print("\n✓ 正文印出来的数与门表一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
