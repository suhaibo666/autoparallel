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


# ---------------------------------------------------------------------------
# Task 6 — Placement algebra + CommSpec + detect_reshard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    """Placement 描述张量在通信轴上的分布状态。

    shard: tuple of (dim_index, axis) pairs（内部存为有序 tuple，可哈希）。
    构造时 shard 入参可接受 dict 或 tuple；partial 为未规约轴名或 None。
    """
    shard: tuple = ()
    partial: str = None

    def __init__(self, shard=(), partial=None):
        items = tuple(sorted(shard.items())) if isinstance(shard, dict) else tuple(shard)
        object.__setattr__(self, "shard", items)
        object.__setattr__(self, "partial", partial)

    @staticmethod
    def of(t: TensorRef) -> "Placement":
        return Placement(t.shard, t.partial)


@dataclass(frozen=True)
class CommSpec:
    ctype: str          # all_reduce | reduce_scatter | all_gather | all_to_all
    volume_bytes: int
    group_axis: str
    phase: str = "fwd"


def detect_reshard(src: Placement, dst: Placement,
                   numel: int, dtype_bytes: int):
    """推导两个 placement 之间的 collective；相等或 src=None 返回 None。"""
    if src is None or src == dst:
        return None
    # 推断 collective 轴
    if src.partial:
        axis = src.partial
    elif src.shard:
        axis = src.shard[0][1]
    elif dst.shard:
        axis = dst.shard[0][1]
    else:
        return None
    if src.partial and not dst.partial and not dst.shard:
        ctype = "all_reduce"
    elif src.partial and dst.shard:
        ctype = "reduce_scatter"
    elif src.shard and not dst.shard and not dst.partial:
        ctype = "all_gather"
    elif src.shard and dst.shard:
        ctype = "all_to_all"
    else:
        return None
    return CommSpec(ctype, numel * dtype_bytes, axis)


# ---------------------------------------------------------------------------
# Task 7 — ResolvedOp / ResolvedLayer / ResolvedGraph + ShapeEval
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedOp:
    name: str
    type: str
    inputs: tuple
    output: ResolvedTensor
    params: tuple
    saves: tuple
    workspace_bytes: int
    collectives: tuple


@dataclass(frozen=True)
class ResolvedLayer:
    layer_id: int
    layer_type: str
    ops: tuple


@dataclass(frozen=True)
class ResolvedGraph:
    stages: dict          # int -> list[ResolvedLayer]


class ShapeEval:
    def resolve(self, spec: ModelSpec, pm) -> ResolvedGraph:
        """遍历 layer_pattern → stage_of 分组；逐 op 解析 inputs/output/params/saves，
        算 workspace，按相邻 op 的 placement 不匹配派生 collectives。"""
        stages: dict = {}
        for layer_id, ltype in enumerate(spec.layer_pattern):
            stage = pm.stage_of(layer_id)
            lspec = spec.get_layer(ltype)
            r_ops = []
            produced: dict = {}          # tensor name -> Placement of producer
            for op in lspec.ops:
                r_in = tuple(resolve_tensor(t, spec.dims, pm) for t in op.inputs)
                r_out = resolve_tensor(op.output, spec.dims, pm)
                r_par = tuple(resolve_tensor(t, spec.dims, pm) for t in op.params)
                r_sav = tuple(resolve_tensor(t, spec.dims, pm) for t in op.saves)
                ws = eval_expr(op.workspace, spec.dims) if op.workspace else 0
                comms = []
                for t in op.inputs:
                    src = produced.get(t.name)
                    c = detect_reshard(
                        src,
                        Placement.of(t),
                        resolve_tensor(t, spec.dims, pm).local_numel,
                        spec.dims.dtype_bytes,
                    )
                    if c:
                        comms.append(c)
                r_ops.append(ResolvedOp(
                    op.name, op.type.value,
                    r_in, r_out, r_par, r_sav,
                    ws, tuple(comms),
                ))
                produced[op.output.name] = Placement.of(op.output)
            stages.setdefault(stage, []).append(
                ResolvedLayer(layer_id, ltype, tuple(r_ops))
            )
        return ResolvedGraph(stages)
