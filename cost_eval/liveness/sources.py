"""**graph source 插座**：liveness 仿真器吃的 op 图从哪来 —— 手写 census 还是源抽取（2026-07-25）。

动机
----
显存今天算自**手写**逐 op 名册（`cost_eval/layers/*.py` 里人肉声明的 `saves=[...]`）。这份名册
已被真机反复证伪（漏 `sparse_indices`/`sinks`、把 detached 的 O(S²) fp32 张量当 saved……），
故决定改成从**真 MindFormers 源**抽出的图（`cost_eval/opdag/`，纯 AST）。逐张量 liveness 仿真器
（`simulate.py`）与两者都**无关**：它只要一个 `ResolvedLayer` 序列。

本模块把「图从哪来」变成一个**注册项**而不是一次改写：

    resolve_graph(spec, pm, "hand_spec")   # 今天：ShapeEval 解析手写 census
    resolve_graph(spec, pm, "extracted")   # 明天：opdag/to_resolved.py 解析抽取图

两条来源喂**同一个** `simulate_liveness` → 可在同一仿真器上做 A/B 对照（这正是交叉校验要的：
差异只能来自图本身，不可能来自仿真口径）。

契约
----
一个 graph source 是可调用对象 ``fn(model_spec, parallel_model) -> ResolvedGraphLike``，
返回值只需带 ``.stages: dict[int, Sequence[ResolvedLayerLike]]``；每个 layer 必须满足
`cost_eval/liveness/contract.py` 的 `ResolvedLayerContract`（见该模块 docstring 的逐条规则）。

**逐字节中立**：``"hand_spec"`` 就是原来 `simulate.py` 里那一行 `ShapeEval().resolve(spec, pm)`，
默认值不变 → 既有 liveness 数值与桶路径数值一字节不动（`tests/test_liveness_graph_source_seam.py`
逐字段比对默认路径与显式路径证明之）。

`extracted` 的落地方式
---------------------
**不改本模块**。`cost_eval/opdag/to_resolved.py` 落地后暴露

    def resolve_graph(model_spec, parallel_model) -> ResolvedGraph

即可；`available_graph_sources()` 会在首次询问时惰性探测该入口并自动注册（见
`_probe_extracted`）。也可由任意第三方显式 `register_graph_source("extracted", fn)`
（如实验分支、或先喂一份离线 JSON 图）。
"""
from __future__ import annotations

from typing import Callable

__all__ = [
    "HAND_SPEC", "EXTRACTED", "EXTRACTED_ENTRY_POINT",
    "register_graph_source", "unregister_graph_source", "graph_source",
    "has_graph_source", "available_graph_sources", "resolve_graph",
]

HAND_SPEC = "hand_spec"
EXTRACTED = "extracted"
#: `extracted` 来源的约定入口点（模块路径, 属性名）——由 opdag 侧提供，本模块只惰性探测。
EXTRACTED_ENTRY_POINT = ("cost_eval.opdag.to_resolved", "resolve_graph")

_REGISTRY: dict = {}
_probed = False


def _hand_spec(model_spec, parallel_model):
    """手写 census 来源：`ModelSpec` → `ShapeEval` 符号求值 → `ResolvedGraph`。

    这与改造前 `simulate.py:209` 的那一行**完全相同**，故默认路径逐字节不变。"""
    from ..shape_eval import ShapeEval
    return ShapeEval().resolve(model_spec, parallel_model)


_REGISTRY[HAND_SPEC] = _hand_spec


def register_graph_source(name: str, fn: Callable, *, override: bool = False) -> None:
    """注册一个 graph source。

    参数
    ----
    name     : 来源名（`"hand_spec"` / `"extracted"` / 任意实验名）。
    fn       : ``fn(model_spec, parallel_model) -> ResolvedGraphLike``。
    override : 已存在时是否允许覆盖（默认 False → 抛 ValueError，避免静默换口径）。
    """
    if not callable(fn):
        raise TypeError(f"graph source {name!r} 必须可调用，实得 {type(fn).__name__}")
    if name in _REGISTRY and not override:
        raise ValueError(f"graph source {name!r} 已注册；要替换请传 override=True")
    _REGISTRY[name] = fn


def unregister_graph_source(name: str) -> None:
    """注销（主要供测试用；注销 `hand_spec` 会让默认路径失效，故禁止）。"""
    if name == HAND_SPEC:
        raise ValueError("hand_spec 是默认来源，不可注销")
    _REGISTRY.pop(name, None)


def _probe_extracted() -> None:
    """惰性探测 `cost_eval/opdag/to_resolved.py` 的约定入口点并注册为 `extracted`。

    **静默跳过只在「模块不存在」这一种情形**（该文件尚未落地是预期状态）；模块存在但入口点缺失
    则 fail-loud —— 否则「抽取来源明明在了却悄悄不参与 A/B」是最坏的假绿。"""
    global _probed
    if _probed:
        return
    _probed = True
    mod_path, attr = EXTRACTED_ENTRY_POINT
    try:
        mod = __import__(mod_path, fromlist=["*"])
    except ImportError:
        return                                    # 尚未落地 —— 预期状态，不报错
    fn = getattr(mod, attr, None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"{mod_path} 已存在但没有可调用的 {attr}(model_spec, parallel_model)；"
            f"graph source {EXTRACTED!r} 无法注册（见 cost_eval/liveness/contract.py 的契约）")
    _REGISTRY.setdefault(EXTRACTED, fn)


def has_graph_source(name: str) -> bool:
    """该来源当前是否可用（会触发 `extracted` 的惰性探测）。"""
    if name in _REGISTRY:
        return True
    if name == EXTRACTED:
        _probe_extracted()
    return name in _REGISTRY


def available_graph_sources() -> tuple:
    """当前可用来源名（升序；`hand_spec` 恒在，`extracted` 视 opdag 侧是否落地）。"""
    _probe_extracted()
    return tuple(sorted(_REGISTRY))


def graph_source(name: str = HAND_SPEC) -> Callable:
    """取来源可调用对象；未注册则 fail-loud（附当前可用清单，不静默退回 hand_spec）。"""
    if callable(name):
        return name
    if not has_graph_source(name):
        raise KeyError(f"未知 graph source {name!r}；当前可用：{available_graph_sources()}")
    return _REGISTRY[name]


def resolve_graph(model_spec, parallel_model, source=HAND_SPEC):
    """`(ModelSpec, ParallelModel, source)` → `ResolvedGraphLike`（带 `.stages`）。

    `source` 可以是注册名，也可以直接是一个 ``fn(spec, pm)``（便于一次性实验，不必注册）。"""
    g = graph_source(source)(model_spec, parallel_model)
    if not hasattr(g, "stages"):
        raise TypeError(f"graph source {source!r} 的返回值缺少 `.stages`（得到 {type(g).__name__}）")
    return g
