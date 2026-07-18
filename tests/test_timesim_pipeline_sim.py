"""全局调度 DES（M9-L2，spec §6.1/6.2）：调度序取自 schedule.py（时间侧不重新发明调度），
p2p=依赖边时延（不占 device 资源），bubble=仿真空闲积分（非公式）。"""
import pytest

from cost_eval.timesim.pipeline_sim import simulate_pipeline


def _uniform(pp, tf=10.0, tb=20.0, v=1):
    return {(s, k, c): (tf if k == "FWD" else tb)
            for s in range(pp) for k in ("FWD", "BWD") for c in range(v)}


def test_plain_1f1b_closed_form_exact():
    """L0③退化还原：均匀 stage + p2p=0 → T = (m+pp−1)·(tf+tb)（逐值精确，零公差）。"""
    r = simulate_pipeline(_uniform(2), pp=2, m=4)
    assert r.t_total_us == pytest.approx((4 + 2 - 1) * 30.0)
    # 均匀退化的 bubble 恒等式：每 stage busy = m·(tf+tb)
    assert all(b == pytest.approx(4 * 30.0) for b in r.per_stage_busy)
    assert r.per_stage_bubble[0] == pytest.approx(150.0 - 120.0)
    assert r.bubble_fraction == pytest.approx((2 - 1) / (4 + 2 - 1))


def test_bubble_shrinks_with_m():
    """L0④性质：m↑ → bubble%↓。"""
    b4 = simulate_pipeline(_uniform(4), pp=4, m=4).bubble_fraction
    b16 = simulate_pipeline(_uniform(4), pp=4, m=16).bubble_fraction
    assert b16 < b4


def test_p2p_latency_stretches_pipeline():
    r0 = simulate_pipeline(_uniform(2), pp=2, m=4, p2p_us=0.0)
    r5 = simulate_pipeline(_uniform(2), pp=2, m=4, p2p_us=5.0)
    assert r5.t_total_us > r0.t_total_us


def test_vpp_reduces_bubble():
    """VPP（v=2，chunk 时长=整段一半）应压 bubble → T 更短（性质，非精确式）。"""
    t1 = simulate_pipeline(_uniform(4, 10.0, 20.0, v=1), pp=4, m=8, v=1).t_total_us
    t2 = simulate_pipeline(_uniform(4, 5.0, 10.0, v=2), pp=4, m=8, v=2).t_total_us
    assert t2 < t1


def test_vpp_event_count_and_conservation():
    r = simulate_pipeline(_uniform(2, 5.0, 10.0, v=2), pp=2, m=4, v=2)
    per_stage = [len([e for e in r.events if e.stage == s]) for s in range(2)]
    assert per_stage == [2 * 4 * 2] * 2                       # 2·m·v 事件/stage
    for s in range(2):
        assert r.per_stage_busy[s] + r.per_stage_bubble[s] == pytest.approx(r.t_total_us)


def test_vpp_v3_conserves_and_no_deadlock():
    """VPP 反向 chunk 反转的泛化锁（v=3、pp=3）：无死锁 + 事件计数 2·m·v + 逐 stage 守恒。
    （反向 chunk 反转 = Megatron get_model_chunk_id(forward=False)，深度优先 gs=pp——Task 9
    调度死锁根因修复的回归；v=1 反转为恒等故 plain 路径不受影响。）"""
    r = simulate_pipeline(_uniform(3, 10.0, 20.0, v=3), pp=3, m=9, v=3)
    for s in range(3):
        assert len([e for e in r.events if e.stage == s]) == 2 * 9 * 3
        assert r.per_stage_busy[s] + r.per_stage_bubble[s] == pytest.approx(r.t_total_us)
    # 每个 (stage,kind,mb,chunk) 事件恰一次（反转不丢/不重）
    keys = [(e.stage, e.kind, e.mb, e.chunk) for e in r.events]
    assert len(keys) == len(set(keys)) == 3 * 2 * 9 * 3


def test_vpp_nondefault_group_size_fail_loud():
    """VPP + group_size≠pp（广度优先变体，v1 未建模）→ fail-loud（不静默死锁/不出未验证数）。"""
    with pytest.raises(ValueError, match="group_size"):
        simulate_pipeline(_uniform(4, 10.0, 20.0, v=2), pp=4, m=8, v=2, group_size=2)


def test_vpp_closed_form_bubble_exact():
    """L0③ VPP 版（review 补强）：均匀 stage + p2p=0 → bubble_fraction 精确 =
    (pp−1)/(m·v+pp−1)（plain 版 v=1 的自然推广，虚拟微批数 m·v）——零公差,锁反转后调度的
    bubble 数值正确性（原 test_vpp_reduces_bubble 只弱断言 t2<t1）。"""
    for pp, m, v in [(2, 4, 2), (4, 8, 2), (3, 6, 2), (2, 4, 3)]:
        r = simulate_pipeline(_uniform(pp, 10.0, 20.0, v), pp=pp, m=m, v=v, p2p_us=0.0)
        assert r.bubble_fraction == pytest.approx((pp - 1) / (m * v + pp - 1))


def test_vpp_m_not_divisible_by_pp_fail_loud():
    """VPP 要求 m%pp==0（Megatron 交错硬约束）→ 不整除 fail-loud（首个出时间数的消费者补验，
    否则对不可跑配置出貌似合理的数，review Important）。"""
    with pytest.raises(ValueError, match="整除"):
        simulate_pipeline(_uniform(4, 10.0, 20.0, v=2), pp=4, m=6, v=2)   # 6%4≠0


def test_missing_duration_fail_loud():
    with pytest.raises(KeyError):
        simulate_pipeline({(0, "FWD", 0): 1.0}, pp=2, m=2)


def test_critical_path_ends_at_last_event():
    r = simulate_pipeline(_uniform(2), pp=2, m=4)
    last = max(r.events, key=lambda e: e.end)
    assert r.critical_path[-1] == (last.stage, last.kind, last.mb, last.chunk)
    assert len(r.critical_path) >= 2
