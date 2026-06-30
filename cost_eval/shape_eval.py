"""M4：符号 shape 求值 + 切分代入 + reshard 检测。"""
from __future__ import annotations
import ast
import operator
from dataclasses import dataclass
from math import prod
from .model_spec import DimTable, ModelSpec, TensorRef

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.floordiv,
}


def eval_expr(expr: str, dims: DimTable) -> int:
    """对 dim 符号做受限算术求值；仅允许 + - * // 与已知符号/整数。"""
    env = dims.as_dict()

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"未知符号: {node.id}")
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_ev(node.left), _ev(node.right))
        raise ValueError(f"非法表达式节点: {ast.dump(node)}")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"非法表达式语法: {expr!r}") from exc
    return int(_ev(tree))


# ---------------------------------------------------------------------------
# Task 5 — ResolvedTensor + resolve_tensor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedTensor:
    name: str
    local_numel: int
    dtype_bytes: int
    is_weight: bool
    is_expert: bool = False


def resolve_tensor(t: TensorRef, dims: DimTable, pm) -> ResolvedTensor:
    """符号求值 → 按 shard 维整除轴度数 → 不整除抛 ValueError。"""
    sizes = [eval_expr(e, dims) for e in t.shape]
    for dim_idx, axis in t.shard.items():
        deg = pm.degree(axis)
        if sizes[dim_idx] % deg != 0:
            raise ValueError(
                f"{t.name} dim{dim_idx}={sizes[dim_idx]} 不被 {axis}={deg} 整除")
        sizes[dim_idx] //= deg
    numel = prod(sizes) if sizes else 1
    return ResolvedTensor(t.name, numel, dims.dtype_bytes, t.is_weight, t.has_ep())
