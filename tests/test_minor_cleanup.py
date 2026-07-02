"""Task 8 [MINOR] — 小问题清理：pp>n_layers 守卫 / TLOCAL capacity_factor / cp 重复计数。"""
import pytest

from cost_eval.model_spec import DimTable
from cost_eval.shape_eval import eval_expr
from cost_eval.specs import ParallelConfig
from cost_eval.parallel_model import ParallelModel
from cost_eval.framework import num_distinct_communicators
from cost_eval.layers.ffn import build_moe_ffn_ops


# ── 8.1 pp > n_layers 守卫（此前 ZeroDivisionError）────────────────────────────
def test_pp_exceeds_n_layers_raises_valueerror():
    with pytest.raises(ValueError):
        ParallelModel(ParallelConfig(pp=8), n_layers=4, world_size=8)


def test_pp_le_n_layers_ok():
    pm = ParallelModel(ParallelConfig(pp=2), n_layers=4, world_size=2)
    assert pm.stage_of(0) == 0 and pm.stage_of(3) == 1


# ── 8.2 MoE TLOCAL 含 capacity_factor（design §7：S·B·topk·C/ep）─────────────────
def _moe_disp_tlocal(d):
    ops = build_moe_ffn_ops(d)
    disp = next(op for op in ops if op.name == "dispatch").output
    return eval_expr(disp.shape[0], d)


def test_moe_tlocal_includes_capacity_factor():
    base = dict(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=1,
                n_experts=4, topk=2, moe_F=8)
    d1 = DimTable(capacity_factor=1.0, **base)
    d2 = DimTable(capacity_factor=2.0, **base)
    n1 = _moe_disp_tlocal(d1)
    n2 = _moe_disp_tlocal(d2)
    # C=1.0：退化为 S·B·topk（DSv3 不变）
    assert n1 == d1.S * d1.B * d1.topk
    # C=2.0：dispatched token 数翻倍（capacity 影响显存）
    assert n2 == 2 * n1


# ── 8.4 num_distinct_communicators 不重复计 cp（cp 已含在 dp_shard*cp FSDP 组）──────
def test_num_communicators_no_cp_double_count():
    # cp=2, 其余=1：world(1) + FSDP 组 dp_shard*cp=2(>1) = 2 个；cp 不再单列。
    assert num_distinct_communicators(ParallelConfig(cp=2)) == 2
