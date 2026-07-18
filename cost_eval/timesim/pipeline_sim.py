"""全局调度仿真（M9-L2，spec §6）。

事件 = (stage, kind, mb, chunk)，时长 = durations[(stage, kind, chunk)]（稳态口径：每
(stage, phase, chunk) 一个时长——非均匀 stage 是自然输入，缺 key = KeyError fail-loud）。
每 stage 事件序取自 cost_eval.schedule（契约2 中立模块）：v<=1 → build_1f1b；v>1 →
interleaved_virtual_order（group_size 默认 pp，Megatron model_parallel_config.py:519-520）。

跨 stage 依赖（§6.2；虚拟 stage 号 k = chunk·pp + stage，Megatron round-robin 放置）：
  F(k) 需 F(k−1)：s>0 → (s−1,同c)；s==0 且 c>0 → (pp−1, c−1)；
  B(k) 需 B(k+1)：s<pp−1 → (s+1,同c)；s==pp−1 且 c<v−1 → (0, c+1)；
  B 对自身 F 的依赖由同 stage 调度序 FIFO 隐含（schedule 序保证 F 先于其 B）。

**VPP 反向 chunk 反转（v>1）**：跨 stage 反向依赖传递地要求同一物理 stage 内**深 chunk 先反向**
（B(c,s)→B(c,s+1)→…→B(c,pp−1)→B(c+1,0)→…→B(c+1,s)，故 B(c,s) 需 B(c+1,s) 先完成 → 同
stage 反向 chunk 须**降序**）。而 schedule.interleaved_virtual_order 为 mem_timeline 设计，
其 docstring 明言"forward/backward 均按同一 schedule table FIFO"——反向 chunk 是**升序**。
时间侧消费时按 Megatron `get_model_chunk_id(forward=False)=v−1−chunk` 反转反向事件的 chunk，
使之与因果序一致（契约3「各自独立 walk」：mem 侧不反转、不受影响；时间侧自建消费）。v=1 时
v−1−c≡c 反转为恒等，plain 路径逐字节不变。
**分组限制（v1）**：仅深度优先分组 group_size==pp（Megatron 默认
microbatch_group_size_per_vp_stage=pp）经验证自洽（跨 pp/v 广泛探针守恒+无死锁）；gs≠pp 的
广度优先变体调度序与本反向依赖模型不自洽（pp≥4 实测仍死锁）→ fail-loud，留 v1.5。
p2p：作**依赖边时延**加在跨 stage 依赖上（不占 device 资源，v1；overlap_p2p/dxdw 等 v1.5
换调度生成器）；同 stage 的 chunk 转移（pp=1 退化）不加 p2p。
推进：逐 stage 顺序取下一事件、依赖满足即调度（stage 同刻只执行一个事件）；一整轮无进展 =
调度死锁 → fail-loud（调度序或依赖规则被打破）。
bubble = 每 stage 在 [0, T] 的空闲积分（仿真结果非公式）；闭式 (pp−1)/(m·v+pp−1) 只作
report 参考值（§6.2）。critical_path：从全局最晚事件沿"决定 start 的约束"回溯。"""
from __future__ import annotations

from dataclasses import dataclass

from ..schedule import build_1f1b, interleaved_virtual_order


@dataclass(frozen=True)
class StageEvent:
    stage: int
    kind: str          # "FWD" | "BWD"
    mb: int
    chunk: int
    start: float
    end: float


@dataclass(frozen=True)
class PipelineResult:
    t_total_us: float
    events: tuple
    per_stage_busy: tuple
    per_stage_bubble: tuple
    bubble_fraction: float
    critical_path: tuple      # ((stage, kind, mb, chunk), ...) 首→尾


def _dep_of(kind: str, s: int, mb: int, c: int, pp: int, v: int):
    if kind == "FWD":
        if s > 0:
            return ("FWD", s - 1, mb, c)
        if c > 0:
            return ("FWD", pp - 1, mb, c - 1)
        return None
    if s < pp - 1:
        return ("BWD", s + 1, mb, c)
    if c < v - 1:
        return ("BWD", 0, mb, c + 1)
    return None


def simulate_pipeline(durations: dict, pp: int, m: int, *, v: int = 1,
                       group_size: int | None = None, p2p_us: float = 0.0) -> PipelineResult:
    gs = group_size or pp
    if v > 1 and gs != pp:
        raise ValueError(
            f"pipeline_sim: VPP(v={v}) 时间仿真 v1 仅支持深度优先分组 group_size==pp"
            f"（Megatron 默认 microbatch_group_size_per_vp_stage=pp）；gs={gs}≠pp={pp} 的广度"
            f"优先变体调度序与跨 stage 反向依赖不自洽（pp≥4 实测死锁）——v1.5，fail-loud")
    orders = []
    for s in range(pp):
        if v <= 1:
            evs = [(e.kind, e.mb, 0) for e in build_1f1b(s, pp, m)]
        else:
            # VPP 反向 chunk 反转（模块 docstring）：Megatron get_model_chunk_id(forward=False)
            # = v−1−chunk，使同 stage 反向 chunk 降序、与跨 stage 因果序自洽（mem 侧不反转）。
            evs = [(k, mb, (v - 1 - c) if k == "BWD" else c)
                   for (k, mb, c) in interleaved_virtual_order(s, pp, m, v, gs)]
        orders.append(evs)

    done: dict[tuple, float] = {}
    blocker: dict[tuple, tuple | None] = {}
    prev_ev: dict[int, tuple | None] = {s: None for s in range(pp)}
    ptr = [0] * pp
    clock = [0.0] * pp
    events: list[StageEvent] = []
    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for s in range(pp):
            while ptr[s] < len(orders[s]):
                kind, mb, c = orders[s][ptr[s]]
                dep = _dep_of(kind, s, mb, c, pp, v)
                if dep is not None:
                    dkey = (dep[1], dep[0], dep[2], dep[3])   # (stage, kind, mb, chunk)
                    if dkey not in done:
                        break
                    lat = p2p_us if dep[1] != s else 0.0
                    ready = max(clock[s], done[dkey] + lat)
                    blk = dkey if done[dkey] + lat >= clock[s] else prev_ev[s]
                else:
                    ready = clock[s]
                    blk = prev_ev[s]
                fin = ready + durations[(s, kind, c)]
                key = (s, kind, mb, c)
                done[key] = fin
                blocker[key] = blk
                events.append(StageEvent(s, kind, mb, c, ready, fin))
                clock[s] = fin
                prev_ev[s] = key
                ptr[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("pipeline_sim: 调度死锁（依赖环/调度序被打破）——fail-loud")

    t_total = max(clock)
    busy = [0.0] * pp
    for e in events:
        busy[e.stage] += e.end - e.start
    bubble = [t_total - b for b in busy]
    bf = sum(bubble) / (pp * t_total) if t_total else 0.0

    last = max(events, key=lambda e: e.end)
    path = []
    k = (last.stage, last.kind, last.mb, last.chunk)
    seen = set()
    while k is not None and k not in seen:
        path.append(k)
        seen.add(k)
        k = blocker.get(k)
    return PipelineResult(t_total, tuple(events), tuple(busy), tuple(bubble), bf,
                          tuple(reversed(path)))
