"""P1-12（任务 B）：MoE dispatched-token 多口径（balanced / capacity / skew）。

`TLOCAL` 现为 balanced 均值口径 `S·B·topk·C`（每卡 = /ep）。审计 P1-12：真实 MoE 受路由倾斜、
capacity ceil、padding、最忙 rank 影响，均值适合**吞吐**估计但不足以做 **OOM 安全边界**。故给
dispatched-token 数增加可配置口径 `DimTable.moe_dispatch_mode`：

  - "balanced"（默认，DSv3/DSv4 锚点）：`S·B·topk·C` —— 逐字节等旧 TLOCAL（理想均分）。
  - "capacity"：`ceil(S·B·topk·C / n_experts)·n_experts` —— 每 expert 按 capacity 上取整（padding
    到 ceil）再 ×n_experts；shard ÷ep → `ceil(...)·experts_per_rank`。**最忙口径**，OOM 边界用。
  - "skew"：`S·B·topk·C·skew_factor` —— 均值 × 倾斜因子（percentile 倾斜）。OOM 边界用。

默认必须保持 balanced（锚点逐字节不动）；capacity/skew（factor>1）严格 > balanced。
"""
import math

import pytest

from cost_eval.layers.ffn import MOE_STAGING_WS, TLOCAL, build_moe_ffn_ops
from cost_eval.model_spec import DimTable
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import eval_expr, resolve_tensor
from cost_eval.specs import ParallelConfig

# capacity ceil 要**严格** > balanced 需 n_experts ∤ (S·B·topk·C)。
# 选 n_experts=6、S·B·topk=8 → A=8，ceil(8/6)·6 = 2·6 = 12 > 8。
BASE = dict(H=16, F=32, n_heads=4, n_kv=2, head_dim=8, S=8, B=1, vocab=32, n_layers=3,
            n_experts=6, topk=1, moe_F=32, moe_shared_F=32)


def _disp_op(d):
    return next(op for op in build_moe_ffn_ops(d) if op.name == "dispatch")


def _disp_tokens(d):
    """全局 dispatched-token 数（disp 张量 dim0 表达式求值，未 ÷ep）。"""
    return eval_expr(_disp_op(d).output.shape[0], d)


# ── balanced（默认）：逐字节等旧常量 ────────────────────────────────────────────────
def test_balanced_is_default():
    d = DimTable(**BASE)
    assert getattr(d, "moe_dispatch_mode", "balanced") == "balanced"


def test_balanced_disp_and_staging_byte_identical_to_old():
    d = DimTable(**BASE)
    disp = _disp_op(d)
    # disp dim0 与 staging workspace 必须与旧模块常量**逐字节相同**（守 test_minor_cleanup /
    # test_framework_decomposition 的字符串断言）。
    assert disp.output.shape[0] == TLOCAL == "S*B*topk*capacity_factor"
    assert disp.workspace == MOE_STAGING_WS == "2*S*B*topk*capacity_factor*H"


def test_balanced_value_equals_mean():
    d = DimTable(**BASE)
    assert _disp_tokens(d) == d.S * d.B * d.topk   # C=1.0


# ── capacity：ceil 上取整口径，严格 > balanced ─────────────────────────────────────
def test_capacity_ceil_formula_and_greater_than_balanced():
    d_bal = DimTable(**BASE)
    d_cap = DimTable(moe_dispatch_mode="capacity", **BASE)
    A = d_cap.S * d_cap.B * d_cap.topk            # C=1.0 → 8
    expected = math.ceil(A / d_cap.n_experts) * d_cap.n_experts   # ceil(8/6)*6 = 12
    assert _disp_tokens(d_cap) == expected
    assert _disp_tokens(d_cap) > _disp_tokens(d_bal)


def test_capacity_staging_scales_with_tokens():
    d_cap = DimTable(moe_dispatch_mode="capacity", **BASE)
    disp = _disp_op(d_cap)
    assert eval_expr(disp.workspace, d_cap) == 2 * _disp_tokens(d_cap) * d_cap.H


def test_capacity_disp_resolves_and_divides_by_ep():
    # 全局 12 tokens，n_experts=6；ep=2（n_experts 可被 ep 整除）→ per-rank 12/2 = 6，×H。
    d_cap = DimTable(moe_dispatch_mode="capacity", **BASE)
    disp_out = _disp_op(d_cap).output
    pm = ParallelModel(ParallelConfig(dp_shard=2, ep=2), n_layers=d_cap.n_layers, world_size=2)
    rt = resolve_tensor(disp_out, d_cap, pm)
    assert rt.local_numel == (12 // 2) * d_cap.H


# ── skew：均值 × 倾斜因子，factor>1 时 > balanced ────────────────────────────────────
def test_skew_scales_balanced():
    d_bal = DimTable(**BASE)
    d_skew = DimTable(moe_dispatch_mode="skew", moe_skew_factor=1.5, **BASE)
    bal = _disp_tokens(d_bal)
    assert _disp_tokens(d_skew) == int(bal * 1.5)
    assert _disp_tokens(d_skew) > bal


def test_skew_factor_one_equals_balanced():
    # skew_factor=1.0（默认）时 skew 口径退化为 balanced 值（字段惰性）。
    d_bal = DimTable(**BASE)
    d_skew1 = DimTable(moe_dispatch_mode="skew", moe_skew_factor=1.0, **BASE)
    assert _disp_tokens(d_skew1) == _disp_tokens(d_bal)


# ── 未知口径 fail-loud ───────────────────────────────────────────────────────────
def test_unknown_dispatch_mode_raises():
    d = DimTable(moe_dispatch_mode="bogus", **BASE)
    with pytest.raises(ValueError):
        build_moe_ffn_ops(d)
