# cost_eval/timesim/shard_rules.py
"""并行代入（spec §3.3b）：符号 shape → local 具体 shape。

符号 token 求值复用 opdag.consumer 的机制（上游共享 IR 资产，契约 §2.2-2 允许——注意这不是
内存仿真模块）。轴语义规则：
  - 含 S 的轴 ÷cp；sp_active（producer 的 SP 状态机给出，Task 9）时再 ÷tp；
  - 含 E 的轴 ÷ep；
  - feature 轴按调用方给的 feat_div_last（模块语义：Column 出 ÷tp、Row 入已 ÷tp）；
  - 权重切分轴随模块语义（weight_local：Column 末轴 / Row 轴0），不是一律末轴。
整除性 fail-loud：并行度不整除即报错（评估合法性校验，serve_explorer 同款规则族的时间侧）。"""
from __future__ import annotations

from dataclasses import dataclass

from ..opdag.consumer import axis_value as _axis_value
from ..opdag.sym_shape import parse_shape


@dataclass(frozen=True)
class Degrees:
    tp: int = 1
    cp: int = 1
    ep: int = 1
    dp: int = 1
    pp: int = 1
    sequence_parallel: bool = False   # 配置位；localize 只看 sp_active（producer 状态机据此推导）


def axis_values(sym_shape: str, dims) -> list[int]:
    """符号 shape → 逐轴具体值（纯符号求值，不做并行代入）。"""
    axes = parse_shape(sym_shape)
    if not axes:
        raise ValueError(f"shard_rules: 无法解析符号 shape {sym_shape!r}（fail-loud）")
    vals = []
    for f in axes:
        v = _axis_value(f, dims)
        if v is None:
            raise ValueError(f"shard_rules: {sym_shape!r} 含未解析 token（fail-loud）")
        vals.append(v)
    return vals


def _div_exact(v: int, d: int, what: str) -> int:
    if d <= 1:
        return v
    if v % d:
        raise ValueError(f"shard_rules: {what} 维 {v} 不被并行度 {d} 整除（fail-loud）")
    return v // d


def localize(vals: list[int], sym_shape: str, deg: Degrees, *,
             feat_div_last: int = 1, sp_active: bool = False) -> list[int]:
    """轴语义代入：含 S 的轴 ÷cp（再按 sp_active ÷tp）、含 E 的轴 ÷ep、末轴按 feat_div_last 另除。

    `feat_div_last` 由**调用方**按模块语义给出（本函数不猜）——Column 出 = deg.tp、Row 出 = 1
    （Row 入已在上游 ÷tp，不重复除）。`deg.sequence_parallel` 在本函数是惰性配置位，只有显式
    传入 `sp_active=True` 时才会触发该次 ÷tp（sp_active 由 producer 的 SP 状态机逐段推导，见
    Task 9）。"""
    axes = parse_shape(sym_shape)
    out = list(vals)
    for i, f in enumerate(axes):
        syms = set(f.syms)
        if "S" in syms:
            out[i] = _div_exact(out[i], deg.cp, "seq(cp)")
            if sp_active:
                out[i] = _div_exact(out[i], deg.tp, "seq(sp)")
        if "E" in syms:
            out[i] = _div_exact(out[i], deg.ep, "expert(ep)")
    if feat_div_last > 1:
        last_syms = set(axes[-1].syms) if axes else set()
        if last_syms & {"S", "E"}:
            raise ValueError(
                "shard_rules: feat_div_last>1 且末轴含 S/E 语义——防误用双除（fail-loud）")
        out[-1] = _div_exact(out[-1], feat_div_last, "feature(tp)")
    return out


def weight_local(sym: str, dims, module: str, tp: int) -> tuple[int, ...]:
    """线性层**权重**的 local shape（切分轴随模块语义，不是一律末轴）：
    ColumnParallelLinear   权重 [in, out] → out(末轴) ÷tp；
    RowParallelLinear      权重 [in, out] → in(轴0)  ÷tp；
    SequenceParallelLinear 权重不切（layers.py:819「A is not parallelized」，:850 布局
    ("None","None")——T1 Task 3）。"""
    vals = axis_values(sym, dims)
    if tp > 1:
        if module == "ColumnParallelLinear":
            vals[-1] = _div_exact(vals[-1], tp, "col-weight out")
        elif module == "RowParallelLinear":
            vals[0] = _div_exact(vals[0], tp, "row-weight in")
        elif module == "SequenceParallelLinear":
            pass                                   # 权重全量（每 rank 复制）
    return tuple(vals)
