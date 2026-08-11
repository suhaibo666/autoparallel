# -*- coding: utf-8 -*-
"""空转基准 · 第二轮：补测朴素实现的真实内存与全尺度 allocator。

第一轮的两处未测准：
  · 朴素实现在 33k 事件尺度上被 gc 干扰，RSS 增量测成了负数
  · allocator 只在 100k 子集上测，未验证全尺度是否线性
"""
from __future__ import annotations

import gc
import os
import time

import numpy as np
import psutil

PROC = psutil.Process(os.getpid())
RNG = np.random.default_rng(20260804)
GiB = 1 << 30


def rss():
    return PROC.memory_info().rss


class Event:
    __slots__ = ("kind", "target", "stream", "sid", "size", "t")

    def __init__(self, kind, target, stream, sid, size, t):
        self.kind, self.target, self.stream = kind, target, stream
        self.sid, self.size, self.t = sid, size, t


def bucketed(sz, ta, tf, align=512, nb=48):
    buckets = [[] for _ in range(nb)]
    live, top, peak = {}, 0, 0
    order = np.argsort(np.concatenate([ta, tf]), kind="stable")
    n = len(sz)
    for k in order:
        if k < n:
            need = int((int(sz[k]) + align - 1) // align * align)
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
                peak = max(peak, top)
        else:
            i = k - n
            if i in live:
                off, ln = live.pop(i)
                buckets[min(nb - 1, max(0, ln.bit_length() - 1))].append((off, ln))
    return peak


def main():
    print("═" * 92)
    print("【1】朴素宿主对象的真实内存占用（分尺度实测，验证线性）")
    print(f"{'事件数':>10} {'构造 s':>9} {'µs/事件':>10} {'RSS 增量 MB':>13} {'B/事件':>9}")
    per_ev = []
    for n_ev in (50_000, 200_000, 800_000):
        gc.collect(); r0 = rss(); t0 = time.perf_counter()
        ev = [Event(i & 1, i % 16, 0, i >> 1, 1 << 20, i) for i in range(n_ev)]
        dt = time.perf_counter() - t0
        gc.collect(); dr = rss() - r0
        b = dr / n_ev
        per_ev.append(b)
        print(f"{n_ev:>10,} {dt:>9.3f} {dt/n_ev*1e6:>10.3f} {dr/1e6:>13.1f} {b:>9.0f}")
        del ev; gc.collect()
    b_naive = per_ev[-1]
    print(f"\n  ⇒ 稳定值 ≈ {b_naive:.0f} B/事件（G-R1 门 ≤ 64 B ⇒ "
          f"{'通过' if b_naive <= 64 else '**不通过**'}）")
    print(f"  ⇒ 外推 A 列 7.2×10⁵ 事件：{b_naive*7.2e5/1e9:.2f} GB；"
          f"B 列 1.75×10⁶：{b_naive*1.75e6/1e9:.2f} GB")

    print("\n" + "═" * 92)
    print("【2】allocator 分桶实现的全尺度线性性")
    N = 4100 * 32 * 16
    sz = RNG.integers(4096, 64 << 20, N, dtype=np.int64)
    ta = np.arange(N, dtype=np.int64)
    tf = ta + RNG.integers(1, N // 8, N)
    print(f"{'事件数(alloc)':>14} {'秒':>9} {'µs/事件':>10}")
    prev = None
    for frac in (0.1, 0.4, 1.0):
        k = int(N * frac)
        t0 = time.perf_counter(); bucketed(sz[:k], ta[:k], tf[:k]); dt = time.perf_counter() - t0
        us = dt / (2 * k) * 1e6
        flag = "" if prev is None else f"   （相对上一档 µs/事件 ×{us/prev:.2f}）"
        print(f"{k:>14,} {dt:>9.3f} {us:>10.3f}{flag}")
        prev = us

    print("\n" + "═" * 92)
    print("【3】单次编译的真实墙钟（SoA + 分桶 allocator）")
    FIVE_PASS_US = 0.090          # 第一轮实测
    ALLOC_US = prev               # 本轮全尺度实测
    for tag, E in (("A 列 grouped", 7.2e5), ("B 列 ungrouped", 1.75e6),
                   ("本基准合成规模", float(N))):
        core = (FIVE_PASS_US + ALLOC_US) * E / 1e6
        print(f"  {tag:<16} E={E:>10,.0f}   单核 {core:7.2f} s   8 核 {core/8:6.2f} s"
              f"   （5 遍 {FIVE_PASS_US*E/1e6:.2f} s + allocator {ALLOC_US*E/1e6:.2f} s）")

    print("\n" + "═" * 92)
    print("【4】对账表 R22 的实测替代值")
    print(f"  设计假设            20 µs/事件")
    print(f"  实测（5 遍）         {FIVE_PASS_US:.3f} µs/事件      ⇒ 假设保守 {20/FIVE_PASS_US:.0f}×")
    print(f"  实测（含 allocator） {FIVE_PASS_US+ALLOC_US:.3f} µs/事件      ⇒ 假设保守 {20/(FIVE_PASS_US+ALLOC_US):.1f}×")
    print(f"  最坏 allocator      1294 µs/事件（第一轮实测）⇒ 比假设**差 {1294/20:.0f}×**")
    print("═" * 92)


if __name__ == "__main__":
    main()
