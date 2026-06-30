"""Task 11 + 12 tests for M6 mem_timeline (build_1f1b + simulate)."""
from cost_eval.mem_timeline import build_1f1b, Event


# ---------------------------------------------------------------------------
# Task 11: 1F1B 调度构建
# ---------------------------------------------------------------------------

def test_1f1b_warmup_depth():
    # PP=4, stage 0: warmup = PP-1-0 = 3 个前向先行
    evs = build_1f1b(stage=0, pp=4, m=8)
    fwd_prefix = []
    for e in evs:
        if e.kind == "FWD":
            fwd_prefix.append(e)
        else:
            break
    assert len(fwd_prefix) == 4    # warmup(3) + steady 第一个 F = 4 个 F 才出现 B


def test_event_counts_balanced():
    evs = build_1f1b(stage=1, pp=4, m=8)
    assert sum(e.kind == "FWD" for e in evs) == 8
    assert sum(e.kind == "BWD" for e in evs) == 8
