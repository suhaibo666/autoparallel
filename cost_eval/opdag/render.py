# cost_eval/opdag/render.py
"""op-DAG / 内存时间线 → **自包含 SVG**（纯 Python,零依赖:无 graphviz/matplotlib,浏览器直接开）。

- `dag_to_svg(dag, dims=None)`：op-DAG 分层布局。节点按 op 类型着色;**`derive_saves` 判定为 saved
  的张量高亮（红框 + 字节数）**——即"反向要存进内存"的激活;view/cast 等**转瞬即释/重算**的灰显。
- `timeline_to_svg(samples)`：内存时间线 8+桶**堆叠面积** + total 折线,峰值事件标 ★。

设计动机:让"哪些 tensor 存内存、哪些重算"一眼可见,并把仿真的内存曲线可视化对齐真机。
"""
from __future__ import annotations

from collections import deque

from .bprop_rules import derive_saves

# op 类型 → 填充色（语义分组:matmul 蓝 / norm 绿 / 激活橙 / cast 灰 / view 浅灰 / flash 红 / 逐元素 紫）。
_OP_COLOR = {
    "MatMul": "#4e79a7", "BMM": "#4e79a7", "GroupedMatMul": "#2f4b7c",
    "Norm": "#59a14f", "Softmax": "#8cd17d",
    "Activation": "#f28e2b", "Elementwise": "#b07aa1",
    "Cast": "#9c9c9c", "View": "#d7d7d7", "Gather": "#bab0ac",
    "FlashAttention": "#e15759", "Dropout": "#ff9da7",
}
MiB = 2 ** 20


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _ref_name(ref):
    return ref.split(":")[0] if ref else ""


def _ref_shape_dtype(ref):
    p = (ref or "").split(":")
    return (p[1] if len(p) > 1 else "?"), (p[2] if len(p) > 2 else "")


def _levels(dag):
    """最长路径分层（DAG 拓扑深度）。孤立/环节点 → level 0。"""
    succ, indeg = {}, {n.id: 0 for n in dag.nodes}
    for a, b in dag.edges:
        if a in indeg and b in indeg:
            succ.setdefault(a, []).append(b)
            indeg[b] += 1
    level = {n.id: 0 for n in dag.nodes}
    ind = dict(indeg)
    q = deque([nid for nid in ind if ind[nid] == 0])
    while q:
        u = q.popleft()
        for v in succ.get(u, []):
            level[v] = max(level[v], level[u] + 1)
            ind[v] -= 1
            if ind[v] == 0:
                q.append(v)
    return level


def _saved_bytes_by_name(dag, dims):
    """{saved 张量名 → (bytes|None, dtype, sym_shape)}（derive_saves;有 dims 则算字节）。"""
    out = {}
    try:
        from .consumer import save_bytes
    except Exception:
        save_bytes = None
    for s in derive_saves(dag):
        b = None
        if dims is not None and save_bytes is not None:
            try:
                b = save_bytes(s, dims)
            except Exception:
                b = None
        out[s.name] = (b, s.dtype, s.sym_shape)
    return out


def dag_to_svg(dag, dims=None, title=None) -> str:
    """op-DAG → SVG 串。saved 张量（反向要存内存）红框高亮 + 字节;transient 灰显。"""
    COLW, ROWH, NW, NH, MX, MY = 250, 78, 210, 52, 24, 64
    level = _levels(dag)
    saved = _saved_bytes_by_name(dag, dims)

    # 分层排位：同 level 的节点按 id 顺序堆行。
    by_level = {}
    for n in sorted(dag.nodes, key=lambda x: x.id):
        by_level.setdefault(level[n.id], []).append(n)
    pos = {}
    for lv, ns in by_level.items():
        for row, n in enumerate(ns):
            pos[n.id] = (MX + lv * COLW, MY + row * ROWH)
    maxlv = max(by_level) if by_level else 0
    maxrow = max((len(ns) for ns in by_level.values()), default=1)
    W = MX * 2 + maxlv * COLW + NW
    H = MY + maxrow * ROWH + 40

    total_saved = sum(b for b, _, _ in saved.values() if b) if dims is not None else None
    ttl = title or f"op-DAG: {dag.cell}"
    if total_saved is not None:
        ttl += f"  —  saved(激活驻留) 合计 {total_saved / MiB:.1f} MiB"

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Consolas,Menlo,monospace" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="#fbfbfb"/>',
        '<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
        '<path d="M0,0 L7,3 L0,6 Z" fill="#8a8a8a"/></marker></defs>',
        f'<text x="{MX}" y="28" font-size="16" font-weight="bold" fill="#222">{_esc(ttl)}</text>',
    ]
    # 边（先画，压在节点下）
    for a, b in dag.edges:
        if a not in pos or b not in pos:
            continue
        ax, ay = pos[a]; bx, by = pos[b]
        out.append(f'<line x1="{ax+NW}" y1="{ay+NH//2}" x2="{bx}" y2="{by+NH//2}" '
                   f'stroke="#c2c2c2" stroke-width="1.2" marker-end="url(#ar)"/>')
    # 节点
    for n in dag.nodes:
        x, y = pos[n.id]
        fill = _OP_COLOR.get(n.op, "#e8e8e8")
        oname = _ref_name(n.out)
        shp, dt = _ref_shape_dtype(n.out)
        is_saved = oname in saved
        if is_saved:
            b, sdt, _ = saved[oname]
            stroke, sw = "#c0392b", 3
            byte_lbl = (f"💾 {b/MiB:.1f} MiB" if b else f"💾 {sdt}")
            dtype_disp = sdt                       # saved 张量的 dtype（norm 存 fp32,可能 ≠ 节点输出 dtype）
        else:
            stroke, sw = "#9a9a9a", 1
            byte_lbl = "↻ transient"               # 非 saved = 转瞬即释 / 反向重算
            dtype_disp = dt
        out.append(f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="7" '
                   f'fill="{fill}" fill-opacity="0.82" stroke="{stroke}" stroke-width="{sw}"/>')
        label = f"{n.op}" + (f" · {n.module}" if n.module else "")
        out.append(f'<text x="{x+9}" y="{y+18}" font-weight="bold" fill="#fff">{_esc(label)}</text>')
        out.append(f'<text x="{x+9}" y="{y+33}" fill="#f4f4f4">{_esc(oname+":"+shp)}</text>')
        lblcol = "#ffe08a" if is_saved else "#eee"
        out.append(f'<text x="{x+9}" y="{y+47}" fill="{lblcol}" font-size="11">{_esc(byte_lbl)} · {_esc(dtype_disp)}</text>')

    # 图例
    ly = H - 22
    out.append(f'<rect x="{MX}" y="{ly-13}" width="16" height="12" rx="2" fill="none" stroke="#c0392b" stroke-width="3"/>')
    out.append(f'<text x="{MX+22}" y="{ly-3}" fill="#333">saved = 反向要存进内存（激活驻留 fwd→bwd）</text>')
    out.append(f'<rect x="{MX+320}" y="{ly-13}" width="16" height="12" rx="2" fill="none" stroke="#9a9a9a" stroke-width="1"/>')
    out.append(f'<text x="{MX+342}" y="{ly-3}" fill="#333">transient = 转瞬即释 / 反向重算（不占峰值）</text>')
    out.append('</svg>')
    return "\n".join(out)


# ── 内存时间线 → SVG（8+桶堆叠面积 + total 折线 + 峰值 ★）─────────────────────────────
_BUCKETS = [
    ("persistent", "#6b6b6b"), ("act_live", "#4e79a7"), ("kept_frag", "#c0392b"),
    ("gather_buf", "#59a14f"), ("grad_buf", "#f28e2b"), ("recomp_scratch", "#b07aa1"),
    ("bwd_scratch", "#e15759"), ("bwd_working_set", "#8cd17d"), ("swap_buf", "#76b7b2"),
    ("workspace", "#bab0ac"), ("optstep", "#ff9da7"), ("framework", "#d7d7d7"),
]


def timeline_to_svg(samples, title=None) -> str:
    """内存时间线 → 堆叠面积 SVG。samples = list[TimelineSample]（record_timeline=True 得）。"""
    if not samples:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40"><text x="10" y="24">empty timeline</text></svg>'
    W, H, PL, PR, PT, PB = 1080, 460, 70, 210, 44, 56
    plotw, ploth = W - PL - PR, H - PT - PB
    n = len(samples)
    peak = max(s.total_bytes for s in samples)
    ymax = peak * 1.08
    xs = [PL + (i / max(1, n - 1)) * plotw for i in range(n)]

    def yv(v):
        return PT + ploth - (v / ymax) * ploth

    def bval(bd, name):
        return getattr(bd, name, 0) or 0

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Consolas,Menlo,monospace" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="#fbfbfb"/>',
        f'<text x="{PL}" y="26" font-size="16" font-weight="bold" fill="#222">'
        f'{_esc(title or "内存时间线（仿真）")}  —  峰值 {peak/MiB:.0f} MiB</text>',
    ]
    # y 轴网格 + 刻度
    for k in range(5):
        v = ymax * k / 4
        y = yv(v)
        out.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{PL+plotw}" y2="{y:.1f}" stroke="#ececec" stroke-width="1"/>')
        out.append(f'<text x="{PL-8}" y="{y+4:.1f}" text-anchor="end" fill="#888">{v/MiB:.0f}</text>')
    # 堆叠面积（自底向上累加各桶）
    base = [PT + ploth] * n   # 每个 x 当前累计顶（像素 y）
    cum = [0.0] * n
    for name, color in _BUCKETS:
        top = []
        for i, s in enumerate(samples):
            cum[i] += bval(s.breakdown, name)
            top.append(yv(cum[i]))
        # 多边形：上沿(top,正序) + 下沿(base,逆序)
        pts = [f"{xs[i]:.1f},{top[i]:.1f}" for i in range(n)] + \
              [f"{xs[i]:.1f},{base[i]:.1f}" for i in range(n - 1, -1, -1)]
        out.append(f'<polygon points="{" ".join(pts)}" fill="{color}" fill-opacity="0.85" stroke="none"/>')
        base = top
    # total 折线
    tl = " ".join(f"{xs[i]:.1f},{yv(samples[i].total_bytes):.1f}" for i in range(n))
    out.append(f'<polyline points="{tl}" fill="none" stroke="#111" stroke-width="1.6"/>')
    # 峰值 ★
    pi = max(range(n), key=lambda i: samples[i].total_bytes)
    px, py = xs[pi], yv(samples[pi].total_bytes)
    out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#c0392b"/>')
    out.append(f'<text x="{px+7:.1f}" y="{py-6:.1f}" fill="#c0392b" font-weight="bold">★ {_esc(samples[pi].event)} {peak/MiB:.0f}</text>')
    # x 轴：稀疏事件标签
    step = max(1, n // 14)
    for i in range(0, n, step):
        out.append(f'<text x="{xs[i]:.1f}" y="{PT+ploth+16}" text-anchor="middle" fill="#999" font-size="10" '
                   f'transform="rotate(35 {xs[i]:.1f} {PT+ploth+16})">{_esc(samples[i].event)}</text>')
    # 图例
    lx, lyv = PL + plotw + 16, PT
    for name, color in _BUCKETS:
        out.append(f'<rect x="{lx}" y="{lyv-10}" width="13" height="13" rx="2" fill="{color}" fill-opacity="0.85"/>')
        out.append(f'<text x="{lx+18}" y="{lyv+1}" fill="#333" font-size="11">{_esc(name)}</text>')
        lyv += 20
    out.append('</svg>')
    return "\n".join(out)
