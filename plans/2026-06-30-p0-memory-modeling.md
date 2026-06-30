# P0 内存建模评估器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 P0 内存预测器：给定 ModelSpec(声明式 op 图) + 并行配置，纯解析算出 dense/MoE 模型在 5D 并行 + recompute + swap 下的每卡峰值显存、OOM 判定与构成拆解。

**Architecture:** 分层模块 M1`model_spec`→M3`parallel_model`→M4`shape_eval`(符号求值+切分代入+reshard)→M5`static_mem`(持久态)→M6`mem_timeline`(事件驱动峰值仿真)→M7`report`。纯 Python、离线、不 import 真实模型/不上 NPU。

**Tech Stack:** Python 3.9+ dataclass，pytest。无第三方依赖（标准库 `dataclasses`/`enum`/`math`/`re`）。

**设计依据:** [[2026-06-23-pynative-cost-evaluator-design]] / [[2026-06-29-p0-modelspec-and-memory-design]] / [[2026-06-29-evaluator-implementation-design]] / [[2026-06-29-core-modules-m4-m5-m6-internals]]

**代码位置:** `mindformers/pynative/cost_eval/`，测试 `tests/st/test_ut/test_pynative/test_cost_eval/`。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `cost_eval/__init__.py` | 包导出 |
| `cost_eval/model_spec.py` | M1：DimTable/TensorRef/OpSpec/LayerSpec/ModelSpec/OpType + placement 工具 |
| `cost_eval/layers/dense.py` | M1：dense_decoder op 图 builder |
| `cost_eval/layers/moe.py` | M1：moe_decoder op 图 builder |
| `cost_eval/specs.py` | ParallelConfig/OptimizerSpec/HardwareSpec/RecomputeSpec/SwapSpec |
| `cost_eval/parallel_model.py` | M3：度数/mesh/stage 分配 |
| `cost_eval/shape_eval.py` | M4：eval_expr/resolve_tensor/detect_reshard/ShapeEval + Resolved* |
| `cost_eval/static_mem.py` | M5：持久 param/grad/opt |
| `cost_eval/mem_timeline.py` | M6：Buckets/调度/simulate + StagePeak/MemBreakdown |
| `cost_eval/report.py` | M7：Report/Evaluator/PeakMemoryReport |

每个测试文件 `test_<module>.py` 与模块同名。**每个 Task 末尾 commit。**

---

## Task 0: 包脚手架

**Files:**
- Create: `mindformers/pynative/cost_eval/__init__.py`
- Create: `tests/st/test_ut/test_pynative/test_cost_eval/__init__.py`
- Test: `tests/st/test_ut/test_pynative/test_cost_eval/test_smoke.py`

- [ ] **Step 1: 建包 + 冒烟测试**

`cost_eval/__init__.py`:
```python
"""离线并行策略代价评估器（P0：内存）。"""
__all__ = []
```

`test_cost_eval/__init__.py`: 空文件。

`test_smoke.py`:
```python
def test_import_package():
    import mindformers.pynative.cost_eval as ce
    assert ce is not None
```

- [ ] **Step 2: 跑测试**

Run: `pytest tests/st/test_ut/test_pynative/test_cost_eval/test_smoke.py -q`
Expected: 1 passed

- [ ] **Step 3: Commit**

```bash
git add mindformers/pynative/cost_eval tests/st/test_ut/test_pynative/test_cost_eval
git commit -m "feat(cost_eval): scaffold package (P0 Task0)"
```

---

## Task 1: M1 model_spec 数据结构

**Files:**
- Create: `mindformers/pynative/cost_eval/model_spec.py`
- Test: `tests/st/test_ut/test_pynative/test_cost_eval/test_model_spec.py`

- [ ] **Step 1: 写失败测试**

`test_model_spec.py`:
```python
from mindformers.pynative.cost_eval.model_spec import (
    DimTable, TensorRef, OpSpec, OpType, LayerSpec, ModelSpec)

def test_dimtable_as_dict_exposes_symbols():
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)
    assert d.as_dict()["H"] == 8 and d.as_dict()["n_kv"] == 2

def test_tensorref_defaults():
    t = TensorRef("x", ("S", "B", "H"))
    assert t.shard == {} and t.is_weight is False and t.partial is None

def test_opspec_holds_memory_contract():
    w = TensorRef("w", ("H", "F"), shard={1: "tp"}, is_weight=True)
    o = OpSpec("fc", OpType.MATMUL, inputs=[TensorRef("x", ("S", "B", "H"))],
               output=TensorRef("y", ("S", "B", "F"), shard={2: "tp"}),
               params=[w], saves=[TensorRef("x", ("S", "B", "H"))])
    assert o.params[0].is_weight and o.saves[0].name == "x"

def test_modelspec_layer_lookup():
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)
    ls = LayerSpec(ops=[])
    m = ModelSpec("toy", d, layer_pattern=["dense", "dense"], layer_specs={"dense": ls})
    assert m.get_layer("dense") is ls
```

- [ ] **Step 2: 跑测试看失败**

Run: `pytest tests/st/test_ut/test_pynative/test_cost_eval/test_model_spec.py -q`
Expected: FAIL（ImportError: model_spec）

- [ ] **Step 3: 实现 model_spec.py**

```python
"""M1：声明式 op 图数据结构 + 内存契约。"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OpType(Enum):
    MATMUL = "matmul"
    FLASH_ATTN = "flash_attn"
    ELEMENTWISE = "elementwise"
    NORM = "norm"
    ROPE = "rope"
    MOE_ROUTER = "moe_router"
    MOE_GEMM = "moe_gemm"
    DISPATCH = "dispatch"
    COMBINE = "combine"


@dataclass
class DimTable:
    """架构超参（具体值）。"""
    H: int; F: int; n_heads: int; n_kv: int; head_dim: int
    S: int; B: int; vocab: int; n_layers: int
    n_experts: int = 0; topk: int = 0; n_shared: int = 0; moe_F: int = 0
    capacity_factor: float = 1.0
    dtype_bytes: int = 2

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TensorRef:
    """符号 shape + 切分标注。shard: {dim_index -> axis('tp'|'cp'|'ep'|'sp')}。"""
    name: str
    shape: tuple
    shard: dict = field(default_factory=dict)
    is_weight: bool = False
    partial: Optional[str] = None      # 该张量在此轴上是未规约部分和

    def has_ep(self) -> bool:
        return "ep" in self.shard.values()


@dataclass
class OpSpec:
    """算子 + 内存契约。params/saves 都是 TensorRef 列表。"""
    name: str
    type: OpType
    inputs: list
    output: TensorRef
    params: list = field(default_factory=list)   # is_weight 张量 → param/grad/opt
    saves: list = field(default_factory=list)     # save_for_backward → 激活
    workspace: Optional[str] = None               # 符号字节表达式
    attrs: dict = field(default_factory=dict)


@dataclass
class LayerSpec:
    ops: list


@dataclass
class ModelSpec:
    name: str
    dims: DimTable
    layer_pattern: list           # list[str]
    layer_specs: dict             # str -> LayerSpec

    def get_layer(self, layer_type: str) -> LayerSpec:
        return self.layer_specs[layer_type]
```

- [ ] **Step 4: 跑测试通过**

Run: `pytest tests/st/test_ut/test_pynative/test_cost_eval/test_model_spec.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/model_spec.py tests/st/test_ut/test_pynative/test_cost_eval/test_model_spec.py
git commit -m "feat(cost_eval): M1 model_spec data structures + memory contract (P0 Task1)"
```

---

## Task 2: specs（配置数据结构）

**Files:**
- Create: `mindformers/pynative/cost_eval/specs.py`
- Test: `tests/st/test_ut/test_pynative/test_cost_eval/test_specs.py`

- [ ] **Step 1: 写失败测试**

```python
from mindformers.pynative.cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)

def test_optimizer_adamw_bytes():
    assert OptimizerSpec.adamw().state_bytes_per_param == 16

def test_parallelconfig_defaults_single():
    pc = ParallelConfig()
    assert pc.tp == 1 and pc.dp_shard == 1 and pc.reshard_after_forward == "default"

def test_recompute_full_layers():
    rc = RecomputeSpec(mode="full", full_layers={0, 1})
    assert rc.is_full(0) and not rc.is_full(2)
```

- [ ] **Step 2: 跑失败** — `pytest .../test_specs.py -q` → FAIL（ImportError）

- [ ] **Step 3: 实现 specs.py**

```python
"""并行/优化器/硬件/重算/swap 配置。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParallelConfig:
    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    sequence_parallel: bool = False
    reshard_after_forward: str = "default"   # always|never|default
    cpu_offload: bool = False
    microbatch: int = 1
    interleave: int = 1
    layers_per_stage: Optional[list] = None  # None=均匀切
    prefetch_depth: int = 1
    num_microbatches: int = 1                # m = global_batch/(dp*microbatch)，由 adapter 算好


@dataclass
class OptimizerSpec:
    type: str = "AdamW"
    state_bytes_per_param: int = 16          # bf16 param2+grad2+master4+m4+v4

    @classmethod
    def adamw(cls, fp32_grad: bool = False) -> "OptimizerSpec":
        return cls("AdamW", 18 if fp32_grad else 16)


@dataclass
class HardwareSpec:
    max_device_memory: int                   # bytes（来自 ContextConfig.max_device_memory）
    framework_reserve: int = 0               # O_framework 标定常数


@dataclass
class RecomputeSpec:
    mode: str = "None"                       # None|full
    full_layers: set = field(default_factory=set)

    def is_full(self, layer_id: int) -> bool:
        return self.mode == "full" and layer_id in self.full_layers


@dataclass
class SwapSpec:
    enable: bool = False
    default_prefetch: int = 1
    swap_layers: set = field(default_factory=set)

    def swaps(self, layer_id: int) -> bool:
        return self.enable and layer_id in self.swap_layers
```

- [ ] **Step 4: 跑通过** — `pytest .../test_specs.py -q` → 3 passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/specs.py tests/st/test_ut/test_pynative/test_cost_eval/test_specs.py
git commit -m "feat(cost_eval): config specs (P0 Task2)"
```

---

## Task 3: M3 parallel_model

**Files:**
- Create: `mindformers/pynative/cost_eval/parallel_model.py`
- Test: `tests/st/test_ut/test_pynative/test_cost_eval/test_parallel_model.py`

理论依据：镜像 `parallel_dims.py`——`fsdp=dp_shard·cp`，`efsdp=dp_shard·cp·tp//ep`，`degree("sp")=tp if sequence_parallel else 1`。

- [ ] **Step 1: 写失败测试**

```python
import pytest
from mindformers.pynative.cost_eval.specs import ParallelConfig
from mindformers.pynative.cost_eval.parallel_model import ParallelModel

def test_degrees_and_groups():
    pc = ParallelConfig(tp=8, dp_shard=8, cp=1, ep=4, pp=1, sequence_parallel=True)
    pm = ParallelModel(pc, n_layers=4, world_size=8 * 8)
    assert pm.degree("tp") == 8 and pm.degree("sp") == 8     # sp follows tp when SP on
    assert pm.fsdp_degree() == 8                              # dp_shard*cp
    assert pm.efsdp_degree() == 8 * 1 * 8 // 4                # dp_shard*cp*tp//ep = 16

def test_sp_off_degree_is_one():
    pm = ParallelModel(ParallelConfig(tp=8, sequence_parallel=False), 4, 8)
    assert pm.degree("sp") == 1

def test_even_stage_assignment():
    pm = ParallelModel(ParallelConfig(pp=2), n_layers=4, world_size=2)
    assert pm.stage_of(0) == 0 and pm.stage_of(3) == 1
    assert pm.stage_layers(1) == [2, 3]

def test_indivisible_efsdp_raises():
    with pytest.raises(ValueError):
        ParallelModel(ParallelConfig(tp=2, dp_shard=1, cp=1, ep=4), 4, 8)  # ep>dp_shard*cp*tp
```

- [ ] **Step 2: 跑失败** — FAIL（ImportError）

- [ ] **Step 3: 实现 parallel_model.py**

```python
"""M3：并行度数 + mesh 关系 + stage→层 分配（镜像 parallel_dims.py）。"""
from __future__ import annotations
from .specs import ParallelConfig


class ParallelModel:
    def __init__(self, pc: ParallelConfig, n_layers: int, world_size: int):
        self.pc = pc
        self.n_layers = n_layers
        self.world_size = world_size
        region = pc.dp_shard * pc.cp * pc.tp
        if region % pc.ep != 0:
            raise ValueError(
                f"ep({pc.ep}) 必须整除 dp_shard*cp*tp={region}（专家在该区内分片）")
        self._efsdp = region // pc.ep

    def degree(self, axis: str) -> int:
        if axis == "sp":
            return self.pc.tp if self.pc.sequence_parallel else 1
        return {
            "tp": self.pc.tp, "cp": self.pc.cp, "ep": self.pc.ep,
            "dp_shard": self.pc.dp_shard, "dp_replicate": self.pc.dp_replicate,
            "pp": self.pc.pp,
        }[axis]

    def fsdp_degree(self) -> int:
        return self.pc.dp_shard * self.pc.cp

    def efsdp_degree(self) -> int:
        return self._efsdp

    def stage_of(self, layer_id: int) -> int:
        return self._layer_to_stage()[layer_id]

    def stage_layers(self, stage: int) -> list:
        return [l for l, s in enumerate(self._layer_to_stage()) if s == stage]

    def _layer_to_stage(self) -> list:
        pp = self.pc.pp
        if self.pc.layers_per_stage:
            mapping = []
            for stage, count in enumerate(self.pc.layers_per_stage):
                mapping += [stage] * count
            return mapping
        per = self.n_layers // pp
        return [min(l // per, pp - 1) for l in range(self.n_layers)]
```

- [ ] **Step 4: 跑通过** — 4 passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/parallel_model.py tests/st/test_ut/test_pynative/test_cost_eval/test_parallel_model.py
git commit -m "feat(cost_eval): M3 parallel_model (P0 Task3)"
```

---

## Task 4: M4 shape_eval — eval_expr（符号算术）

**Files:**
- Create: `mindformers/pynative/cost_eval/shape_eval.py`
- Test: `tests/st/test_ut/test_pynative/test_cost_eval/test_shape_eval.py`

- [ ] **Step 1: 写失败测试**

```python
from mindformers.pynative.cost_eval.model_spec import DimTable
from mindformers.pynative.cost_eval.shape_eval import eval_expr

DIMS = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)

def test_eval_plain_symbol():
    assert eval_expr("H", DIMS) == 8

def test_eval_arithmetic():
    assert eval_expr("(n_heads+2*n_kv)*head_dim", DIMS) == (2 + 2 * 2) * 4   # 24

def test_eval_integer_literal():
    assert eval_expr("2*F", DIMS) == 32

def test_eval_rejects_unknown_name():
    import pytest
    with pytest.raises(ValueError):
        eval_expr("import os", DIMS)
```

- [ ] **Step 2: 跑失败** — FAIL（ImportError）

- [ ] **Step 3: 实现 eval_expr（受限算术，禁任意代码）**

写入 `shape_eval.py`:
```python
"""M4：符号 shape 求值 + 切分代入 + reshard 检测。"""
from __future__ import annotations
import ast
import operator
from dataclasses import dataclass, field
from math import prod
from .model_spec import DimTable, ModelSpec, TensorRef

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv,
           ast.Div: operator.floordiv}


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

    return int(_ev(ast.parse(expr, mode="eval")))
```

- [ ] **Step 4: 跑通过** — 4 passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/shape_eval.py tests/st/test_ut/test_pynative/test_cost_eval/test_shape_eval.py
git commit -m "feat(cost_eval): M4 eval_expr restricted arithmetic (P0 Task4)"
```

---

## Task 5: M4 shape_eval — resolve_tensor（切分代入）

**Files:**
- Modify: `mindformers/pynative/cost_eval/shape_eval.py`
- Test: `tests/st/test_ut/test_pynative/test_cost_eval/test_shape_eval.py`（追加）

- [ ] **Step 1: 追加失败测试**

```python
from mindformers.pynative.cost_eval.model_spec import TensorRef
from mindformers.pynative.cost_eval.specs import ParallelConfig
from mindformers.pynative.cost_eval.parallel_model import ParallelModel
from mindformers.pynative.cost_eval.shape_eval import resolve_tensor

def _pm(**kw):
    return ParallelModel(ParallelConfig(**kw), n_layers=2, world_size=64)

def test_resolve_shards_tp_dim():
    pm = _pm(tp=8, dp_shard=8)
    t = TensorRef("y", ("S", "B", "2*F"), shard={2: "tp"})
    rt = resolve_tensor(t, DIMS, pm)               # S*B*(2*16/8) = 4*1*4 = 16
    assert rt.local_numel == 4 * 1 * (2 * 16 // 8)

def test_resolve_marks_expert():
    pm = _pm(ep=4, tp=2, dp_shard=2)
    w = TensorRef("w", ("n_experts", "H"), shard={0: "ep"}, is_weight=True)
    import mindformers.pynative.cost_eval.model_spec as ms
    d = ms.DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2, n_experts=8)
    rt = resolve_tensor(w, d, pm)
    assert rt.is_expert and rt.local_numel == (8 // 4) * 8

def test_resolve_indivisible_raises():
    import pytest
    pm = _pm(tp=3)
    with pytest.raises(ValueError):
        resolve_tensor(TensorRef("y", ("H",), shard={0: "tp"}), DIMS, pm)  # 8%3
```

- [ ] **Step 2: 跑失败** — FAIL（resolve_tensor 未定义）

- [ ] **Step 3: 追加 ResolvedTensor + resolve_tensor 到 shape_eval.py**

```python
@dataclass(frozen=True)
class ResolvedTensor:
    name: str
    local_numel: int
    dtype_bytes: int
    is_weight: bool
    is_expert: bool = False


def resolve_tensor(t: TensorRef, dims: DimTable, pm) -> ResolvedTensor:
    sizes = [eval_expr(e, dims) for e in t.shape]
    for dim_idx, axis in t.shard.items():
        deg = pm.degree(axis)
        if sizes[dim_idx] % deg != 0:
            raise ValueError(
                f"{t.name} dim{dim_idx}={sizes[dim_idx]} 不被 {axis}={deg} 整除")
        sizes[dim_idx] //= deg
    numel = prod(sizes) if sizes else 1
    return ResolvedTensor(t.name, numel, dims.dtype_bytes, t.is_weight, t.has_ep())
```

- [ ] **Step 4: 跑通过** — 测试全 passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/shape_eval.py tests/st/test_ut/test_pynative/test_cost_eval/test_shape_eval.py
git commit -m "feat(cost_eval): M4 resolve_tensor sharding substitution (P0 Task5)"
```

---

## Task 6: M4 shape_eval — detect_reshard（placement 代数）

**Files:**
- Modify: `mindformers/pynative/cost_eval/shape_eval.py`
- Test: `test_shape_eval.py`（追加）

- [ ] **Step 1: 追加失败测试**

```python
from mindformers.pynative.cost_eval.shape_eval import detect_reshard, Placement

def test_partial_to_replicate_allreduce():
    src = Placement(shard={}, partial="tp")
    dst = Placement(shard={}, partial=None)
    c = detect_reshard(src, dst, numel=128, dtype_bytes=2)
    assert c.ctype == "all_reduce" and c.group_axis == "tp" and c.volume_bytes == 256

def test_shard_to_shard_alltoall():
    src = Placement(shard={0: "ep"}, partial=None)
    dst = Placement(shard={1: "ep"}, partial=None)
    assert detect_reshard(src, dst, 64, 2).ctype == "all_to_all"

def test_no_reshard_when_equal():
    p = Placement(shard={2: "tp"}, partial=None)
    assert detect_reshard(p, p, 64, 2) is None
```

- [ ] **Step 2: 跑失败** — FAIL（Placement/detect_reshard 未定义）

- [ ] **Step 3: 追加 Placement / CommSpec / detect_reshard**

```python
@dataclass(frozen=True)
class Placement:
    shard: tuple = ()          # 用 tuple(sorted(shard.items())) 以可哈希；构造见下
    partial: str = None

    @staticmethod
    def of(t: TensorRef) -> "Placement":
        return Placement(tuple(sorted(t.shard.items())), t.partial)


@dataclass(frozen=True)
class CommSpec:
    ctype: str
    volume_bytes: int
    group_axis: str
    phase: str = "fwd"


def detect_reshard(src: Placement, dst: Placement, numel: int, dtype_bytes: int):
    if src is None or src == dst:
        return None
    axis = src.partial or (src.shard[0][1] if src.shard else dst.shard[0][1])
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
```

> 测试里 `Placement(shard={0:"ep"})` 传 dict 不可哈希——改测试用 `Placement.of(TensorRef(...))` 或直接传 `tuple`。本任务测试已用 `shard={...}` 形式时，需将 `Placement.shard` 接受 dict 并在 `__post_init__` 转 tuple。改用下方实现以兼容测试：

```python
@dataclass(frozen=True)
class Placement:
    shard: tuple = ()
    partial: str = None

    def __init__(self, shard=(), partial=None):
        items = tuple(sorted(shard.items())) if isinstance(shard, dict) else tuple(shard)
        object.__setattr__(self, "shard", items)
        object.__setattr__(self, "partial", partial)

    @staticmethod
    def of(t: TensorRef) -> "Placement":
        return Placement(t.shard, t.partial)
```

- [ ] **Step 4: 跑通过** — 3 passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/shape_eval.py tests/st/test_ut/test_pynative/test_cost_eval/test_shape_eval.py
git commit -m "feat(cost_eval): M4 placement algebra + reshard detection (P0 Task6)"
```

---

## Task 7: M4 shape_eval — ResolvedGraph + ShapeEval.resolve

**Files:**
- Modify: `mindformers/pynative/cost_eval/shape_eval.py`
- Test: `test_shape_eval.py`（追加）

- [ ] **Step 1: 追加失败测试（用一条手搭 2-op 层）**

```python
from mindformers.pynative.cost_eval.model_spec import OpSpec, OpType, LayerSpec, ModelSpec
from mindformers.pynative.cost_eval.shape_eval import ShapeEval

def _toy_dense_layer():
    x = TensorRef("x", ("S", "B", "H"))
    w1 = TensorRef("w1", ("H", "2*F"), shard={1: "tp"}, is_weight=True)
    h = TensorRef("h", ("S", "B", "2*F"), shard={2: "tp"})
    op = OpSpec("fc1", OpType.MATMUL, inputs=[x, w1], output=h, params=[w1], saves=[x])
    return LayerSpec(ops=[op])

def test_resolve_graph_groups_by_stage():
    spec = ModelSpec("toy", DIMS, ["dense", "dense"], {"dense": _toy_dense_layer()})
    pm = _pm(tp=8, dp_shard=8, pp=2)
    g = ShapeEval().resolve(spec, pm)
    assert set(g.stages.keys()) == {0, 1}
    op0 = g.stages[0][0].ops[0]
    assert op0.params[0].local_numel == 8 * (2 * 16 // 8)   # H * (2F/tp)
    assert op0.saves[0].name == "x"
```

- [ ] **Step 2: 跑失败** — FAIL

- [ ] **Step 3: 追加 Resolved* + ShapeEval**

```python
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
        stages = {}
        for layer_id, ltype in enumerate(spec.layer_pattern):
            stage = pm.stage_of(layer_id)
            lspec = spec.get_layer(ltype)
            r_ops, produced = [], {}
            for op in lspec.ops:
                r_in = tuple(resolve_tensor(t, spec.dims, pm) for t in op.inputs)
                r_out = resolve_tensor(op.output, spec.dims, pm)
                r_par = tuple(resolve_tensor(t, spec.dims, pm) for t in op.params)
                r_sav = tuple(resolve_tensor(t, spec.dims, pm) for t in op.saves)
                ws = eval_expr(op.workspace, spec.dims) if op.workspace else 0
                comms = []
                for t in op.inputs:
                    src = produced.get(t.name)
                    c = detect_reshard(src, Placement.of(t),
                                       resolve_tensor(t, spec.dims, pm).local_numel,
                                       spec.dims.dtype_bytes)
                    if c:
                        comms.append(c)
                r_ops.append(ResolvedOp(op.name, op.type.value, r_in, r_out,
                                        r_par, r_sav, ws, tuple(comms)))
                produced[op.output.name] = Placement.of(op.output)
            stages.setdefault(stage, []).append(ResolvedLayer(layer_id, ltype, tuple(r_ops)))
        return ResolvedGraph(stages)
```

- [ ] **Step 4: 跑通过** — passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/shape_eval.py tests/st/test_ut/test_pynative/test_cost_eval/test_shape_eval.py
git commit -m "feat(cost_eval): M4 ResolvedGraph + ShapeEval.resolve (P0 Task7)"
```

---

## Task 8: M1 layers/dense — dense_decoder op 图

**Files:**
- Create: `mindformers/pynative/cost_eval/layers/__init__.py`
- Create: `mindformers/pynative/cost_eval/layers/dense.py`
- Test: `tests/st/test_ut/test_pynative/test_cost_eval/test_layers.py`

理论依据：P0 §3。op 图 = norm→qkv→rope→flash_attn→o_proj→add→norm→fc1→swiglu→fc2→add。`saves` 取每 op backward 所需（多为输入）。

- [ ] **Step 1: 写失败测试（验证参数量闭式）**

```python
from mindformers.pynative.cost_eval.model_spec import DimTable, OpType
from mindformers.pynative.cost_eval.layers.dense import build_dense_decoder

D = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)

def test_dense_param_numel_matches_formula():
    layer = build_dense_decoder(D)
    from mindformers.pynative.cost_eval.shape_eval import eval_expr
    total = sum(eval_expr(w.shape[0], D) * eval_expr(w.shape[1], D)
               for op in layer.ops for w in op.params)
    # 4H^2 + 3HF  (qkv=H*(n_heads+2n_kv)*hd=H*3H/... 这里 n_heads=n_kv → qkv=3H*H? 用实际维)
    qkv = D.H * (D.n_heads + 2 * D.n_kv) * D.head_dim
    o = D.n_heads * D.head_dim * D.H
    fc1 = D.H * 2 * D.F
    fc2 = D.F * D.H
    assert total == qkv + o + fc1 + fc2

def test_dense_has_flash_attn_and_saves():
    layer = build_dense_decoder(D)
    types = [op.type for op in layer.ops]
    assert OpType.FLASH_ATTN in types
    fa = next(op for op in layer.ops if op.type == OpType.FLASH_ATTN)
    assert any(s.name.startswith("qkv") or s.name in ("q", "k", "v") for s in fa.saves)
```

- [ ] **Step 2: 跑失败** — FAIL（ImportError）

- [ ] **Step 3: 实现 layers/__init__.py（空）+ layers/dense.py**

```python
"""M1：dense_decoder op 图（对照 transformer_layer.py/attention.py/mlp.py）。"""
from ..model_spec import TensorRef, OpSpec, OpType, LayerSpec, DimTable

QKV = "(n_heads+2*n_kv)*head_dim"
NHD = "n_heads*head_dim"


def build_dense_decoder(d: DimTable) -> LayerSpec:
    x = TensorRef("x", ("S", "B", "H"), shard={0: "sp"})
    ln1 = TensorRef("ln1", ("S", "B", "H"))
    qkv_w = TensorRef("qkv_w", ("H", QKV), shard={1: "tp"}, is_weight=True)
    qkv = TensorRef("qkv", ("S", "B", QKV), shard={2: "tp"})
    attn = TensorRef("attn", ("S", "B", NHD), shard={2: "tp"})
    lse = TensorRef("lse", ("S", "B", "n_heads"), shard={2: "tp"})
    o_w = TensorRef("o_w", (NHD, "H"), shard={0: "tp"}, is_weight=True)
    o = TensorRef("o", ("S", "B", "H"), partial="tp")
    h1 = TensorRef("h1", ("S", "B", "H"), shard={0: "sp"})
    ln2 = TensorRef("ln2", ("S", "B", "H"))
    fc1_w = TensorRef("fc1_w", ("H", "2*F"), shard={1: "tp"}, is_weight=True)
    g = TensorRef("g", ("S", "B", "2*F"), shard={2: "tp"})
    act = TensorRef("act", ("S", "B", "F"), shard={2: "tp"})
    fc2_w = TensorRef("fc2_w", ("F", "H"), shard={0: "tp"}, is_weight=True)
    o2 = TensorRef("o2", ("S", "B", "H"), partial="tp")
    h2 = TensorRef("h2", ("S", "B", "H"), shard={0: "sp"})
    fa_ws = "S*B*n_heads*head_dim"   # flash workspace 近似
    ops = [
        OpSpec("ln1", OpType.NORM, [x], ln1, saves=[x]),
        OpSpec("qkv", OpType.MATMUL, [ln1, qkv_w], qkv, params=[qkv_w], saves=[ln1]),
        OpSpec("rope", OpType.ROPE, [qkv], qkv, saves=[]),
        OpSpec("flash", OpType.FLASH_ATTN, [qkv], attn, saves=[qkv, attn, lse], workspace=fa_ws),
        OpSpec("o_proj", OpType.MATMUL, [attn, o_w], o, params=[o_w], saves=[attn]),
        OpSpec("add1", OpType.ELEMENTWISE, [o], h1, saves=[]),
        OpSpec("ln2", OpType.NORM, [h1], ln2, saves=[h1]),
        OpSpec("fc1", OpType.MATMUL, [ln2, fc1_w], g, params=[fc1_w], saves=[ln2]),
        OpSpec("swiglu", OpType.ELEMENTWISE, [g], act, saves=[g]),
        OpSpec("fc2", OpType.MATMUL, [act, fc2_w], o2, params=[fc2_w], saves=[act]),
        OpSpec("add2", OpType.ELEMENTWISE, [o2], h2, saves=[]),
    ]
    return LayerSpec(ops=ops)
```

- [ ] **Step 4: 跑通过** — 2 passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/layers tests/st/test_ut/test_pynative/test_cost_eval/test_layers.py
git commit -m "feat(cost_eval): M1 dense_decoder op graph (P0 Task8)"
```

---

## Task 9: M1 layers/moe — moe_decoder op 图

**Files:**
- Create: `mindformers/pynative/cost_eval/layers/moe.py`
- Test: `test_layers.py`（追加）

理论依据：P0 §4（attn 段同 dense；FFN 换 router→dispatch→moe_gemm×2→combine；专家纯 EP `{0:ep}`，`T_local=S·B·topk·C/ep`）。

- [ ] **Step 1: 追加失败测试**

```python
from mindformers.pynative.cost_eval.layers.moe import build_moe_decoder

DM = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10,
              n_layers=2, n_experts=8, topk=2, moe_F=16)

def test_moe_expert_weight_is_ep_only():
    layer = build_moe_decoder(DM)
    gemms = [op for op in layer.ops if op.type.name == "MOE_GEMM"]
    assert gemms, "should have grouped GEMM"
    for op in gemms:
        for w in op.params:
            assert "ep" in w.shard.values() and "tp" not in w.shard.values()
```

- [ ] **Step 2: 跑失败** — FAIL

- [ ] **Step 3: 实现 layers/moe.py**

```python
"""M1：moe_decoder op 图（对照 moe/{moe_layer,router,experts}.py）。
专家纯 EP（expert_parallel.py:330 `weight:(Shard(0),)`），不 TP 切。"""
from ..model_spec import TensorRef, OpSpec, OpType, LayerSpec, DimTable
from .dense import build_dense_decoder

TLOCAL = "S*B*topk//ep"   # balanced，capacity=1


def build_moe_decoder(d: DimTable) -> LayerSpec:
    attn_ops = build_dense_decoder(d).ops[:6]   # 复用 attn 段（到 add1）
    hin = TensorRef("h1", ("S", "B", "H"), shard={0: "sp"})
    logits = TensorRef("logits", ("S", "B", "n_experts"))
    disp = TensorRef("disp", (TLOCAL, "H"), shard={0: "ep"})
    w1 = TensorRef("e_w1", ("n_experts//ep", "H", "2*moe_F"), shard={0: "ep"}, is_weight=True)
    g = TensorRef("e_g", (TLOCAL, "2*moe_F"), shard={0: "ep"})
    act = TensorRef("e_act", (TLOCAL, "moe_F"), shard={0: "ep"})
    w2 = TensorRef("e_w2", ("n_experts//ep", "moe_F", "H"), shard={0: "ep"}, is_weight=True)
    eo = TensorRef("e_o", (TLOCAL, "H"), shard={0: "ep"})
    comb = TensorRef("comb", ("S", "B", "H"), shard={0: "sp"})
    ffn = [
        OpSpec("router", OpType.MOE_ROUTER, [hin], logits, saves=[logits]),
        OpSpec("dispatch", OpType.DISPATCH, [hin], disp, saves=[disp]),
        OpSpec("e_fc1", OpType.MOE_GEMM, [disp, w1], g, params=[w1], saves=[disp]),
        OpSpec("e_swiglu", OpType.ELEMENTWISE, [g], act, saves=[g]),
        OpSpec("e_fc2", OpType.MOE_GEMM, [act, w2], eo, params=[w2], saves=[act]),
        OpSpec("combine", OpType.COMBINE, [eo], comb, saves=[comb]),
    ]
    return LayerSpec(ops=list(attn_ops) + ffn)
```

- [ ] **Step 4: 跑通过** — passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/layers/moe.py tests/st/test_ut/test_pynative/test_cost_eval/test_layers.py
git commit -m "feat(cost_eval): M1 moe_decoder op graph (pure-EP experts) (P0 Task9)"
```

---

## Task 10: M5 static_mem

**Files:**
- Create: `mindformers/pynative/cost_eval/static_mem.py`
- Test: `tests/st/test_ut/test_pynative/test_cost_eval/test_static_mem.py`

理论依据：M4 已对 params 施加图内切分（dense→/tp、expert→/ep）；M5 再除 FSDP 组（dense `fsdp`、expert `efsdp`）× 优化器倍数。守恒可验。

- [ ] **Step 1: 写失败测试（守恒 + 退化）**

```python
from mindformers.pynative.cost_eval.model_spec import DimTable, ModelSpec
from mindformers.pynative.cost_eval.layers.dense import build_dense_decoder
from mindformers.pynative.cost_eval.specs import ParallelConfig, OptimizerSpec
from mindformers.pynative.cost_eval.parallel_model import ParallelModel
from mindformers.pynative.cost_eval.shape_eval import ShapeEval, eval_expr
from mindformers.pynative.cost_eval.static_mem import StaticMem

D = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=4, B=1, vocab=10, n_layers=2)

def _spec():
    return ModelSpec("toy", D, ["dense", "dense"], {"dense": build_dense_decoder(D)})

def _global_param_numel():
    layer = build_dense_decoder(D)
    return 2 * sum(eval_expr(w.shape[0], D) * eval_expr(w.shape[1], D)
                   for op in layer.ops for w in op.params)   # 2 层

def test_single_device_equals_global_times_bytes():
    pm = ParallelModel(ParallelConfig(), n_layers=2, world_size=1)
    g = ShapeEval().resolve(_spec(), pm)
    out = StaticMem().compute(g, OptimizerSpec.adamw(), pm, cpu_offload=False)
    assert out[0] == _global_param_numel() * 16

def test_conservation_under_sharding():
    pm = ParallelModel(ParallelConfig(tp=2, dp_shard=2), n_layers=2, world_size=4)
    g = ShapeEval().resolve(_spec(), pm)
    out = StaticMem().compute(g, OptimizerSpec.adamw(), pm, cpu_offload=False)
    per_dev_numel = out[0] // 16
    assert per_dev_numel * (2 * 2) == _global_param_numel()   # tp*fsdp = 4

def test_cpu_offload_zeroes_persistent():
    pm = ParallelModel(ParallelConfig(), 2, 1)
    g = ShapeEval().resolve(_spec(), pm)
    out = StaticMem().compute(g, OptimizerSpec.adamw(), pm, cpu_offload=True)
    assert out[0] == 0
```

- [ ] **Step 2: 跑失败** — FAIL（ImportError）

- [ ] **Step 3: 实现 static_mem.py**

```python
"""M5：持久 param/grad/opt。dense→/fsdp，expert→/efsdp（tp 已在 efsdp，不重复除）。"""
from __future__ import annotations


class StaticMem:
    def compute(self, g, opt, pm, cpu_offload: bool) -> dict:
        out = {}
        for stage, layers in g.stages.items():
            numel = 0
            for layer in layers:
                for op in layer.ops:
                    for w in op.params:        # w.local_numel: M4 已 /tp 或 /ep
                        fsdp = pm.efsdp_degree() if w.is_expert else pm.fsdp_degree()
                        numel += w.local_numel // fsdp
            out[stage] = 0 if cpu_offload else numel * opt.state_bytes_per_param
        return out
```

- [ ] **Step 4: 跑通过** — 3 passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/static_mem.py tests/st/test_ut/test_pynative/test_cost_eval/test_static_mem.py
git commit -m "feat(cost_eval): M5 static_mem with efsdp correctness (P0 Task10)"
```

---

## Task 11: M6 mem_timeline — 调度构建（1F1B 事件）

**Files:**
- Create: `mindformers/pynative/cost_eval/mem_timeline.py`
- Test: `tests/st/test_ut/test_pynative/test_cost_eval/test_mem_timeline.py`

- [ ] **Step 1: 写失败测试**

```python
from mindformers.pynative.cost_eval.mem_timeline import build_1f1b, Event

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
```

- [ ] **Step 2: 跑失败** — FAIL（ImportError）

- [ ] **Step 3: 实现调度部分**

写入 `mem_timeline.py`:
```python
"""M6：事件驱动内存时间线仿真。"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    kind: str      # "FWD" | "BWD"
    mb: int
    layer: int = -1


def build_1f1b(stage: int, pp: int, m: int):
    """返回 (FWD/BWD, microbatch) 事件序列（层粒度在 simulate 内展开）。
    warmup=pp-1-stage 个前向先行，然后 1F1B 交替，最后 cooldown。"""
    warmup = min(pp - 1 - stage, m)
    evs = [Event("FWD", i) for i in range(warmup)]
    fwd_i, bwd_i = warmup, 0
    while bwd_i < m:
        if fwd_i < m:
            evs.append(Event("FWD", fwd_i)); fwd_i += 1
        evs.append(Event("BWD", bwd_i)); bwd_i += 1
    return evs
```

- [ ] **Step 4: 跑通过** — 2 passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/mem_timeline.py tests/st/test_ut/test_pynative/test_cost_eval/test_mem_timeline.py
git commit -m "feat(cost_eval): M6 1F1B schedule builder (P0 Task11)"
```

---

## Task 12: M6 mem_timeline — simulate（桶 + recompute + FSDP 预取 + swap）

**Files:**
- Modify: `mindformers/pynative/cost_eval/mem_timeline.py`
- Test: `test_mem_timeline.py`（追加）

理论依据：M4/M5/M6 内部设计 §M6.2/M6.3。峰值落点：无重算→`fwd_end`；开 full 重算→`bwd_recompute@*` 且总峰值下降。

- [ ] **Step 1: 追加失败测试（峰值落点迁移 + 单调性）**

```python
from mindformers.pynative.cost_eval.model_spec import DimTable, ModelSpec
from mindformers.pynative.cost_eval.layers.dense import build_dense_decoder
from mindformers.pynative.cost_eval.specs import ParallelConfig, RecomputeSpec, SwapSpec
from mindformers.pynative.cost_eval.parallel_model import ParallelModel
from mindformers.pynative.cost_eval.shape_eval import ShapeEval
from mindformers.pynative.cost_eval.static_mem import StaticMem
from mindformers.pynative.cost_eval.specs import OptimizerSpec
from mindformers.pynative.cost_eval.mem_timeline import MemTimeline

D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=4)

def _setup(pp=1):
    spec = ModelSpec("toy", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    pm = ParallelModel(ParallelConfig(pp=pp), n_layers=4, world_size=pp)
    g = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(g, OptimizerSpec.adamw(), pm, False)
    return g, pm, persistent

def test_full_recompute_lowers_peak_and_moves_event():
    g, pm, persistent = _setup(pp=1)
    mt = MemTimeline()
    none = mt.simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                       framework_reserve=0, max_device_memory=10**12)
    full = mt.simulate(g, RecomputeSpec("full", {0, 1, 2, 3}), SwapSpec(), pm, persistent,
                       framework_reserve=0, max_device_memory=10**12)
    assert full[0].peak_bytes < none[0].peak_bytes
    assert none[0].peak_event == "fwd_end"
    assert full[0].peak_event.startswith("bwd_recompute")

def test_oom_flag():
    g, pm, persistent = _setup(pp=1)
    r = MemTimeline().simulate(g, RecomputeSpec("None"), SwapSpec(), pm, persistent,
                               framework_reserve=0, max_device_memory=1)
    assert r[0].oom is True
```

- [ ] **Step 2: 跑失败** — FAIL（MemTimeline 未定义）

- [ ] **Step 3: 追加 Buckets/StagePeak/MemBreakdown + MemTimeline.simulate**

```python
@dataclass
class Buckets:
    persistent: int = 0
    act_live: int = 0
    gather_buf: int = 0
    grad_buf: int = 0
    recomp_scratch: int = 0
    swap_buf: int = 0
    workspace: int = 0

    def total(self) -> int:
        return (self.persistent + self.act_live + self.gather_buf + self.grad_buf
                + self.recomp_scratch + self.swap_buf + self.workspace)


@dataclass(frozen=True)
class MemBreakdown:
    persistent: int; act_live: int; gather_buf: int; grad_buf: int
    recomp_scratch: int; swap_buf: int; workspace: int; framework: int


@dataclass(frozen=True)
class StagePeak:
    stage: int
    peak_bytes: int
    breakdown: MemBreakdown
    peak_event: str
    oom: bool


def _layer_saves_bytes(layer) -> int:
    return sum(s.local_numel * s.dtype_bytes for op in layer.ops for s in op.saves)


def _checkpoint_input_bytes(layer) -> int:
    """full 重算时仅保留层输入（首 op 的首个 save）。"""
    for op in layer.ops:
        if op.saves:
            s = op.saves[0]
            return s.local_numel * s.dtype_bytes
    return 0


def _layer_param_bytes(layer) -> int:
    return sum(w.local_numel * w.dtype_bytes for op in layer.ops for w in op.params)


def _layer_workspace(layer) -> int:
    return max([op.workspace_bytes for op in layer.ops] + [0])


class MemTimeline:
    def simulate(self, g, recompute, swap, pm, static_persistent,
                 framework_reserve, max_device_memory) -> dict:
        res = {}
        pp = pm.degree("pp")
        m = pm.pc.num_microbatches
        for stage, layers in g.stages.items():
            layer_ids = [l.layer_id for l in layers]
            by_id = {l.layer_id: l for l in layers}
            B = Buckets(persistent=static_persistent.get(stage, 0))
            peak, peak_ev, peak_bd = -1, None, None

            def rec(tag):
                nonlocal peak, peak_ev, peak_bd
                t = B.total() + framework_reserve
                if t > peak:
                    peak, peak_ev = t, tag
                    peak_bd = MemBreakdown(B.persistent, B.act_live, B.gather_buf,
                                           B.grad_buf, B.recomp_scratch, B.swap_buf,
                                           B.workspace, framework_reserve)

            pinned = {}                       # (mb, layer_id) -> saved bytes
            for ev in build_1f1b(stage, pp, m):
                if ev.kind == "FWD":
                    for lid in layer_ids:
                        layer = by_id[lid]
                        B.workspace = _layer_workspace(layer); rec(f"fwd:{lid}")
                        B.workspace = 0
                        if recompute.is_full(lid):
                            saved = _checkpoint_input_bytes(layer)
                        elif swap.swaps(lid):
                            saved = 0
                        else:
                            saved = _layer_saves_bytes(layer)
                        pinned[(ev.mb, lid)] = saved
                        B.act_live += saved
                    rec("fwd_end")
                else:  # BWD（逆序层）
                    for lid in reversed(layer_ids):
                        layer = by_id[lid]
                        if recompute.is_full(lid):
                            B.recomp_scratch = _layer_saves_bytes(layer)
                            rec(f"bwd_recompute@{lid}")
                            B.recomp_scratch = 0
                        B.grad_buf = _layer_param_bytes(layer); rec(f"bwd_grad@{lid}")
                        B.grad_buf = 0
                        B.act_live -= pinned.pop((ev.mb, lid))
            res[stage] = StagePeak(stage, peak, peak_bd, peak_ev,
                                   oom=peak > max_device_memory)
        return res
```

> 说明：P0 此版把 FSDP `gather_buf` 简化为 0（dense 已 /tp 持久；预取双缓冲在 Task 13 后续增量补 `_ensure_gather`），swap 仅做"激活离开 act_live"。`gather_buf` 完整建模与 `select/exclude_op`、op 级 swap 列为 P0 收尾增量（见末尾"后续增量"）。

- [ ] **Step 4: 跑通过** — 2 passed

- [ ] **Step 5: Commit**

```bash
git add mindformers/pynative/cost_eval/mem_timeline.py tests/st/test_ut/test_pynative/test_cost_eval/test_mem_timeline.py
git commit -m "feat(cost_eval): M6 timeline simulate (recompute spike + swap) (P0 Task12)"
```

---

## Task 13: M7 report + Evaluator 门面 + 端到端

**Files:**
- Create: `mindformers/pynative/cost_eval/report.py`
- Test: `tests/st/test_ut/test_pynative/test_cost_eval/test_report_e2e.py`

- [ ] **Step 1: 写失败测试（端到端 + 最紧 stage）**

```python
from mindformers.pynative.cost_eval.model_spec import DimTable, ModelSpec
from mindformers.pynative.cost_eval.layers.dense import build_dense_decoder
from mindformers.pynative.cost_eval.layers.moe import build_moe_decoder
from mindformers.pynative.cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from mindformers.pynative.cost_eval.report import Evaluator

def test_e2e_dense_breakdown_and_oom():
    D = DimTable(H=4096, F=11008, n_heads=32, n_kv=32, head_dim=128,
                 S=4096, B=1, vocab=32000, n_layers=4)
    spec = ModelSpec("llama-ish", D, ["dense"] * 4, {"dense": build_dense_decoder(D)})
    ev = Evaluator(
        spec, ParallelConfig(tp=8, dp_shard=8, num_microbatches=1),
        OptimizerSpec.adamw(), HardwareSpec(max_device_memory=60 * 2**30),
        RecomputeSpec("None"), SwapSpec())
    rep = ev.evaluate()
    p = rep.per_stage[0]
    # persistent ≈ (4层参数)/(tp*fsdp=64) * 16；非 0、有拆解
    assert p.breakdown.persistent > 0
    assert rep.tightest_stage == 0
    assert rep.oom == (p.peak_bytes > 60 * 2**30)

def test_e2e_moe_runs():
    DM = DimTable(H=512, F=1024, n_heads=8, n_kv=8, head_dim=64, S=512, B=1,
                  vocab=1000, n_layers=2, n_experts=8, topk=2, moe_F=1024)
    spec = ModelSpec("moe", DM, ["moe", "moe"], {"moe": build_moe_decoder(DM)})
    ev = Evaluator(spec, ParallelConfig(ep=4, tp=2, dp_shard=2, num_microbatches=1),
                   OptimizerSpec.adamw(), HardwareSpec(max_device_memory=80 * 2**30),
                   RecomputeSpec("None"), SwapSpec())
    rep = ev.evaluate()
    assert rep.per_stage[0].peak_bytes > 0
```

- [ ] **Step 2: 跑失败** — FAIL（ImportError）

- [ ] **Step 3: 实现 report.py**

```python
"""M7：组装报告 + Evaluator 门面。"""
from __future__ import annotations
from dataclasses import dataclass
from .parallel_model import ParallelModel
from .shape_eval import ShapeEval
from .static_mem import StaticMem
from .mem_timeline import MemTimeline, StagePeak


@dataclass(frozen=True)
class PeakMemoryReport:
    per_stage: list          # list[StagePeak]
    tightest_stage: int
    oom: bool


class Evaluator:
    def __init__(self, model_spec, parallel_config, optimizer, hardware,
                 recompute, swap):
        self.spec = model_spec
        self.pc = parallel_config
        self.opt = optimizer
        self.hw = hardware
        self.recompute = recompute
        self.swap = swap

    def evaluate(self) -> PeakMemoryReport:
        world = (self.pc.dp_replicate * self.pc.dp_shard * self.pc.cp
                 * self.pc.tp * self.pc.pp)
        pm = ParallelModel(self.pc, self.spec.dims.n_layers, world)
        g = ShapeEval().resolve(self.spec, pm)
        persistent = StaticMem().compute(g, self.opt, pm, self.pc.cpu_offload)
        peaks = MemTimeline().simulate(
            g, self.recompute, self.swap, pm, persistent,
            self.hw.framework_reserve, self.hw.max_device_memory)
        per_stage = [peaks[s] for s in sorted(peaks)]
        tightest = max(per_stage, key=lambda p: p.peak_bytes).stage
        return PeakMemoryReport(per_stage, tightest, any(p.oom for p in per_stage))
```

- [ ] **Step 4: 跑通过** — 2 passed

- [ ] **Step 5: 跑全部测试**

Run: `pytest tests/st/test_ut/test_pynative/test_cost_eval/ -q`
Expected: 全部 passed（约 20+）

- [ ] **Step 6: Commit**

```bash
git add mindformers/pynative/cost_eval/report.py tests/st/test_ut/test_pynative/test_cost_eval/test_report_e2e.py
git commit -m "feat(cost_eval): M7 report + Evaluator facade + e2e dense/moe (P0 Task13)"
```

---

## 后续增量（P0 收尾，单独小 Task，可在主链跑通后做）

- **FSDP `gather_buf` + 预取双缓冲**：在 `MemTimeline` 加 `_ensure_gather`（按 `reshard_after_forward`/`prefetch_depth` 维护 `gathered` 集合），dense 权重 just-in-time all-gather 计入 `gather_buf`。测试：reshard `never` vs `always` 峰值不同。
- **recompute `select`/`exclude_op`**：`RecomputeSpec` 加 `select_modules`/`exclude_ops`，`_saves_after_recompute_swap` 按模块路径/op 名细化（不再整层）。
- **op 级 swap + prefetch buffer**：`SwapSpec` 支持 op 级 + `swap_buf = default_prefetch · 单次预取`。
- **config_adapter**：从 `TrainConfig` YAML 自动构建 ModelSpec/ParallelConfig/HardwareSpec（P0 暂在测试里手搭）。

---

## Self-Review（写完即查）

- **Spec 覆盖**：M1(Task1,8,9) / M3(Task3) / M4(Task4-7) / M5(Task10) / M6(Task11-12) / M7(Task13) 全覆盖；FSDP gather/预取与 select/op-swap 显式列入"后续增量"（不在主链遗漏）。
- **占位符**：无 TODO；所有 step 含完整代码与可跑命令。
- **类型一致**：`ResolvedTensor.local_numel`/`is_expert`、`ParallelModel.degree/fsdp_degree/efsdp_degree`、`StagePeak.peak_bytes/peak_event/breakdown`、`Buckets.total()`、`Evaluator.evaluate()` 跨任务命名一致。
- **已知简化**（在 Task12 标注）：P0 主链 `gather_buf=0`、swap 仅"离开 act_live"；完整建模在"后续增量"。
