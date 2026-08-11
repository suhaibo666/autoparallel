# -*- coding: utf-8 -*-
"""合成图空转基准 —— 把「每事件成本」从假设变成测量值。

设计文档 §13.1 自曝：对账表四个输入数「全部是推的，不是测的」，其中**最危险的是每事件
成本**——按「SoA + 编译热循环」给 20 µs，但 allocator 模拟的最坏实现（线性扫描找空洞、
每次分配遍历空闲链）单事件可到毫秒级，那样整张表是**三个数量级**的作废。

本基准**不涉及任何模型语义**：假事件、假 Storage、假依赖。它只测五个热循环 + allocator
在两种实现下的每事件成本，以及 RSS/实例数（`G-R1` 的 64 B 门）。

五个热循环（§13.1 R20）：实例化 / liveness×2 / 段内 DES / 报告聚合。
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import psutil

PROC = psutil.Process(os.getpid())
RNG = np.random.default_rng(20260804)


def rss_mb() -> float:
    return PROC.memory_info().rss / 1024 / 1024


class T:
    """计时器：返回墙钟秒与该段内 RSS 增量。"""

    def __init__(self, name, n):
        self.name, self.n = name, n

    def __enter__(self):
        gc.collect()
        self.r0 = rss_mb()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.dt = time.perf_counter() - self.t0
        self.dr = rss_mb() - self.r0
        self.us = self.dt / self.n * 1e6
        print(f"    {self.name:<34} {self.dt:8.3f} s   {self.us:9.3f} µs/事件   ΔRSS {self.dr:+8.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# 合成图：折叠图 → 实例化 → 事件流
# ══════════════════════════════════════════════════════════════════════════════
def make_folded(n_nodes: int):
    """折叠图：n_nodes 个节点，每节点产出 1 个 storage，带 def/lastuse 的相对偏移。"""
    size = RNG.integers(4096, 64 << 20, n_nodes, dtype=np.int64)      # 4KB–64MB
    span = RNG.integers(1, max(2, n_nodes // 8), n_nodes)             # 存活跨度
    persistent = RNG.random(n_nodes) < 0.08                           # 8% 持久
    return size, span, persistent


# ── 实现 A · 朴素：每实例一个宿主对象 ──────────────────────────────────────────
class NaiveEvent:
    __slots__ = ("kind", "target", "stream", "sid", "size", "t")

    def __init__(self, kind, target, stream, sid, size, t):
        self.kind, self.target, self.stream = kind, target, stream
        self.sid, self.size, self.t = sid, size, t


def naive_instantiate(size, span, persistent, mb, ranks):
    ev = []
    n = len(size)
    t = 0
    for r in range(ranks):
        for m in range(mb):
            for i in range(n):
                sid = (r * mb + m) * n + i
                ev.append(NaiveEvent(0, r, 0, sid, int(size[i]), t)); t += 1
                if not persistent[i]:
                    ev.append(NaiveEvent(1, r, 0, sid, int(size[i]), t + int(span[i]))); t += 1
    return ev


def naive_liveness(ev):
    live = {}
    cur = peak = 0
    for e in sorted(ev, key=lambda x: x.t):
        if e.kind == 0:
            live[e.sid] = e.size; cur += e.size
            if cur > peak: peak = cur
        else:
            cur -= live.pop(e.sid, 0)
    return peak


def naive_aggregate(ev):
    acc = {}
    for e in ev:
        k = e.sid % 8                       # 假的 root 种类
        acc[k] = acc.get(k, 0) + e.size
    return acc


# ── 实现 B · SoA：平铺数组 + 向量化 ────────────────────────────────────────────
def soa_instantiate(size, span, persistent, mb, ranks):
    n = len(size)
    reps = mb * ranks
    sid = np.arange(n * reps, dtype=np.int64)
    sz = np.tile(size, reps)
    sp = np.tile(span, reps)
    pers = np.tile(persistent, reps)
    t_alloc = np.arange(n * reps, dtype=np.int64)
    t_free = t_alloc + sp
    t_free[pers] = np.iinfo(np.int64).max            # 持久：不释放
    return sid, sz, t_alloc, t_free


def soa_liveness(sz, t_alloc, t_free):
    """扫描线：+size @alloc，−size @free，前缀和取 max。O(n log n)，全向量化。"""
    fin = t_free != np.iinfo(np.int64).max
    ts = np.concatenate([t_alloc, t_free[fin]])
    dv = np.concatenate([sz, -sz[fin]])
    o = np.argsort(ts, kind="stable")
    return int(np.cumsum(dv[o]).max())


def soa_aggregate(sid, sz):
    return np.bincount(sid % 8, weights=sz, minlength=8)


def soa_des(n_seg, per_seg):
    """段级 DES：每段一个 (start, dur)，按依赖串起来。"""
    dur = RNG.integers(1, 1000, n_seg).astype(np.int64)
    start = np.cumsum(np.concatenate([[0], dur[:-1]]))
    return int((start + dur).max())


# ══════════════════════════════════════════════════════════════════════════════
# allocator 模拟：被设计点名为最大风险的那一格
# ══════════════════════════════════════════════════════════════════════════════
def alloc_linear(sz, t_alloc, t_free, align=512):
    """最坏实现：free list 线性扫描找 best-fit。这是设计担心的毫秒级那条。"""
    free = []                                       # [(off, len)]
    live = {}
    top = 0
    peak = 0
    order = np.argsort(np.concatenate([t_alloc, t_free]), kind="stable")
    n = len(sz)
    for k in order:
        if k < n:                                   # alloc
            need = int((sz[k] + align - 1) // align * align)
            best = -1; bl = 1 << 62
            for j, (off, ln) in enumerate(free):    # ← 线性扫描
                if ln >= need and ln < bl:
                    best, bl = j, ln
            if best >= 0:
                off, ln = free.pop(best)
                live[k] = (off, need)
                if ln > need:
                    free.append((off + need, ln - need))
            else:
                live[k] = (top, need); top += need
                if top > peak: peak = top
        else:                                       # free
            i = k - n
            if i in live:
                off, ln = live.pop(i)
                free.append((off, ln))
    return peak


def alloc_bucketed(sz, t_alloc, t_free, align=512, nb=32):
    """分桶 free list：按 size 的 log2 分桶，桶内取首个。O(1) 均摊。"""
    buckets = [[] for _ in range(nb)]
    live = {}
    top = 0
    peak = 0
    order = np.argsort(np.concatenate([t_alloc, t_free]), kind="stable")
    n = len(sz)
    for k in order:
        if k < n:
            need = int((sz[k] + align - 1) // align * align)
            b = min(nb - 1, max(0, need.bit_length() - 1))
            hit = None
            for bb in range(b, nb):
                if buckets[bb]:
                    hit = buckets[bb].pop(); break
            if hit is not None:
                off, ln = hit
                live[k] = (off, need)
                if ln > need:
                    rb = min(nb - 1, max(0, (ln - need).bit_length() - 1))
                    buckets[rb].append((off + need, ln - need))
            else:
                live[k] = (top, need); top += need
                if top > peak: peak = top
        else:
            i = k - n
            if i in live:
                off, ln = live.pop(i)
                b = min(nb - 1, max(0, ln.bit_length() - 1))
                buckets[b].append((off, ln))
    return peak


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Python {sys.version.split()[0]}  numpy {np.__version__}  cores {os.cpu_count()}")
    print(f"起始 RSS {rss_mb():.1f} MB\n")

    N_FOLDED = 4100                      # ≈ 对账表 R8 的 G3 折叠节点数
    print("═" * 96)
    print(f"折叠图 {N_FOLDED} 节点（≈ 对账表 R8）")

    size, span, persistent = make_folded(N_FOLDED)

    # ── B · SoA，全尺度 ───────────────────────────────────────────────────────
    for mb, ranks, tag in ((32, 16, "A 列 mb=32 × rank类=16"),):
        n_ev = N_FOLDED * mb * ranks * 2
        print(f"\n【SoA + numpy 向量化】 {tag} ⇒ 事件数 ≈ {n_ev/1e6:.2f}×10⁶")
        r0 = rss_mb()
        with T("① 实例化", n_ev) as t1:
            sid, sz, ta, tf = soa_instantiate(size, span, persistent, mb, ranks)
        with T("② liveness @lo", n_ev) as t2:
            p_lo = soa_liveness(sz, ta, tf)
        with T("③ liveness @hi", n_ev) as t3:
            p_hi = soa_liveness((sz * 1.1).astype(np.int64), ta, tf)
        with T("④ 段内 DES", n_ev) as t4:
            soa_des(mb * ranks * 2, N_FOLDED)
        with T("⑤ 报告聚合", n_ev) as t5:
            soa_aggregate(sid, sz)
        tot = t1.us + t2.us + t3.us + t4.us + t5.us
        rss = rss_mb() - r0
        print(f"    {'合计（5 遍）':<34} {'':>8}     {tot:9.3f} µs/事件")
        print(f"    峰值 lo/hi = {p_lo/2**30:.1f} / {p_hi/2**30:.1f} GiB")
        print(f"    RSS 增量 {rss:.0f} MB  ⇒  {rss*1024*1024/n_ev:.1f} B/事件"
              f"   （G-R1 门：≤ 64 B ⇒ {'通过' if rss*1024*1024/n_ev <= 64 else '不通过'}）")

        # allocator 两种实现（在 10 万事件的子集上测，再外推）
        SUB = 100_000
        print(f"\n  allocator 模拟（子集 {SUB//1000}k alloc + 同量 free）：")
        with T("  best-fit 线性扫描（最坏实现）", SUB * 2) as ta1:
            alloc_linear(sz[:SUB], ta[:SUB], tf[:SUB])
        with T("  分桶 free list", SUB * 2) as ta2:
            alloc_bucketed(sz[:SUB], ta[:SUB], tf[:SUB])
        print(f"    线性扫描 / 分桶 = {ta1.us/ta2.us:.0f}×")

    # ── A · 朴素宿主对象，小尺度后外推 ────────────────────────────────────────
    SMALL_MB, SMALL_R = 2, 2
    n_small = N_FOLDED * SMALL_MB * SMALL_R * 2
    print(f"\n【朴素：每实例一个宿主对象】 mb={SMALL_MB} × rank类={SMALL_R} ⇒ 事件数 {n_small/1e3:.0f}k"
          f"（小尺度实测后外推）")
    r0 = rss_mb()
    with T("① 实例化", n_small) as u1:
        ev = naive_instantiate(size, span, persistent, SMALL_MB, SMALL_R)
    with T("② liveness", n_small) as u2:
        naive_liveness(ev)
    with T("⑤ 报告聚合", n_small) as u5:
        naive_aggregate(ev)
    rss_n = rss_mb() - r0
    print(f"    RSS 增量 {rss_n:.0f} MB  ⇒  {rss_n*1024*1024/n_small:.0f} B/事件"
          f"   （G-R1 门：≤ 64 B ⇒ {'通过' if rss_n*1024*1024/n_small <= 64 else '不通过'}）")
    print(f"    外推到 A 列 7.2×10⁵ 事件：RSS ≈ {rss_n*1024*1024/n_small*7.2e5/1e9:.1f} GB")
    del ev
    gc.collect()
    print("\n" + "═" * 96)


if __name__ == "__main__":
    main()
