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
