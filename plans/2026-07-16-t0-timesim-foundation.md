# T0 地基：timesim foundation 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 spec `2026-07-16-step-time-cost-model-design.md` 的 T0 地基：schedule.py 搬家（解耦契约）、timesim 包骨架 + import-lint、opdag 通信提取与 GPTModel 级覆盖扩展、TimedOpSeq producer（fwd/bwd/recompute）+ IR 层不变量。

**Architecture:** 时间仿真器（`cost_eval/timesim/`）与内存仿真器双向 import 禁令，只共享 `specs.py`/新抽出的 `schedule.py`/opdag 上游 IR。opdag 新增 comm_probe（TP 通信惯用法静态探测）与 loss/embedding/lm_head 段提取；timesim 的 producer 把 OpDAG 装配成带 local shape、通信注入、bwd/recompute 展开的 TimedOpSeq。

**Tech Stack:** Python 3.13 + pytest + stdlib ast（绝不 import mindspore/mindformers）。mindformers 源根 `E:\97-codes\torch_parallel\mindformers\mindformers`（环境变量 `MINDFORMERS_ROOT` 可覆盖，测试无源则 skip——沿用 test_opdag_* 既有模式）。

**执行约定：**
- 仓库根：`E:\97-codes\torch_parallel\pynative-cost-evaluator`，分支 `feat/unified-llm-modelspec`（沿用本仓惯例，不建 worktree）。
- 每个任务末跑**全量** `python -m pytest tests/ -q`——12 内存锚点（test_dsv3_golden / test_regression_dsv3 等）是硬回归门。
- 所有源事实引用（`file:line`）以执行时实际源码为准；若行号漂移，以**惯用法**定位并更新注释里的行号。

---

## 文件结构总览

```
cost_eval/schedule.py                # 新：纯 1F1B/VPP 调度代数（自 mem_timeline 搬家）
cost_eval/mem_timeline.py            # 改：删调度函数 → re-export
cost_eval/timesim/__init__.py        # 新：包骨架
cost_eval/timesim/ir.py              # 新：TimedOp/TimedSegment/CommSpec + op_flops
cost_eval/timesim/shard_rules.py     # 新：并行代入（轴语义×度数）
cost_eval/timesim/producer.py        # 新：OpDAG→TimedSegment（装配+TP通信注入+sp状态机）
cost_eval/timesim/frame_comm.py      # 新：框架层通信注入（FSDP/EP/CP）
cost_eval/timesim/bwd_rules.py       # 新：per-op-type bwd 展开 + recompute 前缀
cost_eval/opdag/comm_probe.py        # 新：TP 通信惯用法静态探测
cost_eval/opdag/gpt_segments.py      # 新：loss/embedding/lm_head 段 + GPTModel 段序核对
cost_eval/opdag/extractor.py         # 改：直接实例化子 Cell 绑定（subcell_specs 惯用法B）
cost_eval/opdag/init_binder.py       # 改：_CLS2OP 增补（loss/embedding 用到的原语）
tests/test_schedule_move.py          # 新
tests/test_timesim_decoupling.py     # 新（import-lint，进回归门）
tests/test_opdag_comm_probe.py       # 新
tests/test_opdag_gpt_segments.py     # 新
tests/test_timesim_ir.py             # 新
tests/test_timesim_producer.py       # 新
tests/test_timesim_bwd_rules.py      # 新
tests/test_timesim_invariants.py     # 新（6ND + bwd 守恒）
```

---

# Phase A — 解耦地基

## Task 1: 测试基线记录

**Files:** 无改动。

- [ ] **Step 1: 跑全量测试，记录基线**

Run: `python -m pytest tests/ -q`
Expected: 全绿。记下通过数 N_baseline（后续每任务对照，不允许减少）。

## Task 2: schedule.py 搬家（纯重构，锚点回归门）

**Files:**
- Create: `cost_eval/schedule.py`
- Modify: `cost_eval/mem_timeline.py:8-179`
- Test: `tests/test_schedule_move.py`

- [ ] **Step 1: 写搬家回归测试（先写，此刻 import 失败即"红"）**

```python
# tests/test_schedule_move.py
"""schedule.py 搬家（spec §2.2 契约2）：调度代数移入中立模块，mem_timeline re-export 兼容。
验收 = ①新模块可独立 import 且不依赖任何内存模块;②旧 import 路径同一对象(is);③行为逐值一致。"""
import ast
import inspect


def test_schedule_module_standalone():
    import cost_eval.schedule as sch
    for name in ("Event", "build_1f1b", "interleaved_warmup", "build_interleaved_1f1b",
                 "get_schedule_table", "interleaved_virtual_order", "chunk_layer_ids"):
        assert hasattr(sch, name), name


def test_schedule_has_no_mem_imports():
    import cost_eval.schedule as sch
    src = inspect.getsource(sch)
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.update(a.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.update(node.module.split("."))
    assert not names & {"mem_timeline", "structure_mem", "static_mem"}


def test_mem_timeline_reexports_same_objects():
    import cost_eval.schedule as sch
    import cost_eval.mem_timeline as mt
    for name in ("Event", "build_1f1b", "interleaved_warmup", "build_interleaved_1f1b",
                 "get_schedule_table", "interleaved_virtual_order", "chunk_layer_ids"):
        assert getattr(mt, name) is getattr(sch, name), name


def test_schedule_behavior_spotcheck():
    from cost_eval.schedule import build_1f1b, interleaved_virtual_order
    evs = build_1f1b(stage=0, pp=2, m=4)          # warmup=1
    kinds = [(e.kind, e.mb) for e in evs]
    assert kinds == [("FWD", 0), ("FWD", 1), ("BWD", 0), ("FWD", 2), ("BWD", 1),
                     ("FWD", 3), ("BWD", 2), ("BWD", 3)]
    order = interleaved_virtual_order(stage=0, pp=2, m=4, v=2, group_size=2)
    assert len(order) == 2 * 4 * 2 and order[0][0] == "FWD"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_schedule_move.py -q`
Expected: FAIL（`ModuleNotFoundError: cost_eval.schedule`）。

- [ ] **Step 3: 创建 `cost_eval/schedule.py`（逐行搬移，不改一字）**

文件头：

```python
"""纯 1F1B/VPP 调度代数（自 mem_timeline 搬家，spec 2026-07-16 §2.2 契约2）。

**中立模块**：无任何内存/时间语义，mem_timeline（内存仿真）与 timesim（时间仿真）各自消费，
两者互不 import（tests/test_timesim_decoupling.py 强制）。逐行 port Megatron schedules.py
的注释与 file:line 引用随函数原样保留。"""
from __future__ import annotations
from dataclasses import dataclass
```

正文 = 现 `cost_eval/mem_timeline.py` 第 **12–179 行原文剪切搬移，一字不改**（`Event`、
`_1f1b_from_warmup`、`build_1f1b`、`interleaved_warmup`、`build_interleaved_1f1b`、
`get_schedule_table`、`interleaved_virtual_order`、`chunk_layer_ids` 共 8 个定义，含全部注释块）。
搬移后确认这些函数体内**不引用** `structure_mem`/桶类（它们是纯函数，仅用 stdlib）。

- [ ] **Step 4: 改 `mem_timeline.py` 为 re-export**

删除第 8–179 行（分节注释 + 8 个定义），原位替换为：

```python
# ---------------------------------------------------------------------------
# 1F1B/VPP 调度代数已搬家至 cost_eval/schedule.py（spec 2026-07-16 §2.2 契约2：
# 时间/内存两仿真器共享的中立调度模块）。此处 re-export 保持全部旧 import 路径兼容。
# ---------------------------------------------------------------------------
from .schedule import (                                    # noqa: F401
    Event, _1f1b_from_warmup, build_1f1b, interleaved_warmup,
    build_interleaved_1f1b, get_schedule_table, interleaved_virtual_order,
    chunk_layer_ids,
)
```

- [ ] **Step 5: 跑新测试 + 全量回归**

Run: `python -m pytest tests/test_schedule_move.py -q` → PASS（4 个）。
Run: `python -m pytest tests/ -q` → 通过数 = N_baseline + 4（锚点全绿；任何内存测试失败 = 搬家动了语义，回退重查）。

- [ ] **Step 6: Commit**

```bash
git add cost_eval/schedule.py cost_eval/mem_timeline.py tests/test_schedule_move.py
git commit -m "refactor(schedule): 1F1B/VPP 调度代数搬家至中立模块 schedule.py(T0-1,契约2,锚点逐字节不动)"
```

## Task 3: timesim 包骨架 + import-lint 解耦测试

**Files:**
- Create: `cost_eval/timesim/__init__.py`
- Test: `tests/test_timesim_decoupling.py`

- [ ] **Step 1: 写 import-lint 测试（含 linter 自检）**

```python
# tests/test_timesim_decoupling.py
"""解耦契约 §2.2-1（进回归门）：cost_eval/timesim/** 与内存仿真模块双向 import 禁令。
共享白名单只有 specs / schedule / opdag（上游 IR）——契约 §2.2-2。"""
import ast
import glob
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cost_eval")
MEM_MODULES = {"mem_timeline", "structure_mem", "static_mem"}


def _imported_names(src: str) -> set[str]:
    """import 语句涉及的全部名字段集合（模块点分段 + from-import 的符号名）。"""
    toks: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                toks.update(a.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                toks.update(node.module.split("."))
            for a in node.names:
                toks.add(a.name)
    return toks


def _read(p):
    with open(p, "r", encoding="utf-8") as fh:
        return fh.read()


def test_linter_helper_catches_violations():
    """linter 自检：三种走私写法都要被抓到。"""
    assert "mem_timeline" in _imported_names("from ..mem_timeline import build_1f1b")
    assert "mem_timeline" in _imported_names("from cost_eval import mem_timeline")
    assert "mem_timeline" in _imported_names("import cost_eval.mem_timeline as mt")


def test_timesim_never_imports_mem_simulator():
    files = glob.glob(os.path.join(ROOT, "timesim", "**", "*.py"), recursive=True)
    assert files, "timesim 包不存在？"
    for p in files:
        bad = _imported_names(_read(p)) & MEM_MODULES
        assert not bad, f"{p} 走私 import 内存模块: {bad}"


def test_mem_simulator_never_imports_timesim():
    for mod in MEM_MODULES:
        p = os.path.join(ROOT, f"{mod}.py")
        assert "timesim" not in _imported_names(_read(p)), f"{p} 反向 import timesim"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_timesim_decoupling.py -q`
Expected: FAIL（`test_timesim_never_imports_mem_simulator` 断言 "timesim 包不存在？"）。

- [ ] **Step 3: 创建包骨架**

```python
# cost_eval/timesim/__init__.py
"""timesim：step time 仿真器（spec specs/2026-07-16-step-time-cost-model-design.md）。

与内存仿真器（mem_timeline/structure_mem/static_mem）**双向 import 禁令**（契约 §2.2-1，
tests/test_timesim_decoupling.py 强制）。允许共享：cost_eval.specs / cost_eval.schedule /
cost_eval.opdag（上游 IR 资产）。"""
```

- [ ] **Step 4: 跑测试 + 全量**

Run: `python -m pytest tests/test_timesim_decoupling.py tests/ -q` → 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add cost_eval/timesim/__init__.py tests/test_timesim_decoupling.py
git commit -m "feat(timesim): 包骨架 + import-lint 双向解耦测试(T0-2,契约1)"
```

---

# Phase B — opdag 扩展（通信提取 + GPTModel 级覆盖）

## Task 4: comm_probe——TP 通信惯用法静态探测（R1 打通）

**Files:**
- Create: `cost_eval/opdag/comm_probe.py`
- Test: `tests/test_opdag_comm_probe.py`

**已核实的源事实**（mindformers `parallel_core/training_graph/tensor_parallel/layers.py`）：
- `RowParallelLinear.__init__`：`self.reduce_scatter = ops.ReduceScatter(group=...)`（:545）、
  `self.all_reduce = ops.AllReduce(group=...)`（:547）；方法体 `if self.sequence_parallel:
  input_ = self.reduce_scatter(input_) else: input_ = self.all_reduce(input_)`（:619/:621 与 :646/:648 两处）。
- `VocabParallelEmbedding.embedding_func`：sp 支**内联** `ops.ReduceScatter(group=self.group)(output_parallel)`；
  docstring 声明非 sp 支为 AllReduce。
- `ColumnParallelLinear`：construct/morphed 方法**无显式通信**（Morph shard layout 是图模式机制）
  → probe 返回空，SP 前置 all-gather 由 producer 按模块语义注入（Task 9，spec §3.3c 允许）。

- [ ] **Step 1: 写测试**

```python
# tests/test_opdag_comm_probe.py
"""comm_probe：从真 mindformers 源静态提取 TP 集合通信调用点（spec §3.3c 惯用法A/B，R1）。"""
import os
import pytest

from cost_eval.opdag.comm_probe import probe_cell_comm

MF_ROOT = os.environ.get("MINDFORMERS_ROOT",
                         r"E:\97-codes\torch_parallel\mindformers\mindformers")
TP_REL = "parallel_core/training_graph/tensor_parallel/layers.py"


def _require_mf():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")


def test_row_parallel_linear_sites():
    _require_mf()
    sites = probe_cell_comm(MF_ROOT, TP_REL, "RowParallelLinear")
    kinds = {(s.ctype, s.guard) for s in sites}
    assert ("reduce_scatter", "sequence_parallel") in kinds       # :619/:646
    assert ("all_reduce", "!sequence_parallel") in kinds          # :621/:648
    assert all(s.src.startswith("layers.py:") for s in sites)


def test_vocab_parallel_embedding_sites():
    _require_mf()
    sites = probe_cell_comm(MF_ROOT, TP_REL, "VocabParallelEmbedding")
    ctypes = {s.ctype for s in sites}
    assert "reduce_scatter" in ctypes                             # embedding_func sp 支（内联惯用法B）
    assert "all_reduce" in ctypes                                 # 非 sp 支（docstring 声明）
    rs = [s for s in sites if s.ctype == "reduce_scatter"]
    assert any("sequence_parallel" in s.guard for s in rs)


def test_column_parallel_linear_has_no_explicit_comm():
    _require_mf()
    assert probe_cell_comm(MF_ROOT, TP_REL, "ColumnParallelLinear") == []
```

- [ ] **Step 2: 跑测试确认失败**（`ModuleNotFoundError: comm_probe`）

- [ ] **Step 3: 实现 comm_probe.py**

```python
# cost_eval/opdag/comm_probe.py
"""TP 集合通信惯用法静态探测（spec §3.3c 第一类「源码内显式通信」，风险 R1）。

读真 mindformers Cell 源（AST，绝不 import），对给定类提取**全部方法体**（含 Morph 包裹的
forward 方法）里的集合通信调用点。两种惯用法：
  A) __init__ 绑定：`self.X = ops.AllReduce(group=...)`（链式 .set_prim_instance_name 等剥壳）
     → 方法体里 `self.X(t)`；
  B) 内联：`ops.ReduceScatter(group=...)(t)` 直接构造调用（VocabParallelEmbedding.embedding_func）。
guard = 包围调用的 if 链条件的可识别合取："sequence_parallel"/"!sequence_parallel"，多层用 "&"
连接；不可识别的条件记 "?"（消费方 producer 对含 "?" 的 guard fail-loud，不猜）。
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass

from .extractor import _find_class

COMM_CLS = {
    "AllReduce": "all_reduce", "ReduceScatter": "reduce_scatter",
    "AllGather": "all_gather", "AlltoAll": "all_to_all", "AlltoAllV": "all_to_all",
}


@dataclass(frozen=True)
class CommSite:
    ctype: str      # all_reduce | reduce_scatter | all_gather | all_to_all
    guard: str      # "" 无条件 | "sequence_parallel" | "!sequence_parallel" | 含 "?" 不可识别
    src: str        # "layers.py:619"
    method: str     # 调用点所在方法名


def _comm_ctor(call: ast.Call) -> str | None:
    """Call 是否为 ops.<Comm>()/P.<Comm>()/<Comm>() 构造；是则返回 ctype。"""
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return COMM_CLS.get(f.attr)
    if isinstance(f, ast.Name):
        return COMM_CLS.get(f.id)
    return None


def _unwrap_chain(node: ast.AST) -> ast.Call | None:
    """剥 `.set_prim_instance_name(...)/.shard(...)` 链，取最内层 Call。"""
    while isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
            node = f.value
            continue
        return node
    return None


def _mro_classes(tree: ast.AST, cls_name: str) -> list[ast.ClassDef]:
    """derived→base 顺序的同文件 MRO 类节点（找不到基类源即止，够 layers.py 用）。"""
    out, seen, queue = [], set(), [cls_name]
    while queue:
        cname = queue.pop(0)
        if cname in seen:
            continue
        seen.add(cname)
        cls = _find_class(tree, cname)
        if cls is None:
            continue
        out.append(cls)
        for base in cls.bases:
            if isinstance(base, ast.Name):
                queue.append(base.id)
    return out


def _guard_of(test: ast.AST) -> str:
    """if 条件 → 可识别 guard：self.<flag> → flag；not self.<flag> → !flag；其余 ?。"""
    if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name) \
            and test.value.id == "self":
        return test.attr
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _guard_of(test.operand)
        return "!" + inner if not inner.startswith("?") else "?"
    return "?"


class _MethodScan(ast.NodeVisitor):
    def __init__(self, binds: dict[str, str], src_file: str, method: str):
        self.binds, self.src_file, self.method = binds, src_file, method
        self.stack: list[str] = []
        self.sites: list[CommSite] = []

    def _guard(self) -> str:
        return "&".join(self.stack)

    def visit_If(self, node: ast.If):
        g = _guard_of(node.test)
        self.stack.append(g)
        for s in node.body:
            self.visit(s)
        self.stack.pop()
        self.stack.append("!" + g if not g.startswith("?") else "?")
        for s in node.orelse:
            self.visit(s)
        self.stack.pop()

    def visit_Call(self, node: ast.Call):
        f = node.func
        # 惯用法A：self.X(...) 且 X 是 __init__ 里的通信绑定
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                and f.value.id == "self" and f.attr in self.binds:
            self.sites.append(CommSite(self.binds[f.attr], self._guard(),
                                       f"{self.src_file}:{node.lineno}", self.method))
        # 惯用法B：ops.<Comm>(group=...)(t) 内联构造调用
        if isinstance(f, ast.Call):
            ct = _comm_ctor(f)
            if ct:
                self.sites.append(CommSite(ct, self._guard(),
                                           f"{self.src_file}:{node.lineno}", self.method))
        self.generic_visit(node)


def probe_cell_comm(mf_root: str, rel: str, cls_name: str) -> list[CommSite]:
    path = os.path.join(mf_root, *rel.split("/"))
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    src_file = os.path.basename(path)
    classes = _mro_classes(tree, cls_name)
    if not classes:
        raise ValueError(f"comm_probe: {src_file} 找不到 class {cls_name}（fail-loud）")
    # 1) 收集 __init__ 里的通信绑定（惯用法A）
    binds: dict[str, str] = {}
    for cls in reversed(classes):                       # base 先，derived 覆盖
        init = next((n for n in cls.body
                     if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is None:
            continue
        for stmt in ast.walk(init):
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                continue
            tgt = stmt.targets[0]
            if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self" and isinstance(stmt.value, ast.Call)):
                continue
            inner = _unwrap_chain(stmt.value)
            ct = _comm_ctor(inner) if inner is not None else None
            if ct:
                binds[tgt.attr] = ct
    # 2) 扫全部方法体的调用点（含 Morph 目标方法——它们就是普通方法）
    sites: list[CommSite] = []
    seen_methods: set[str] = set()
    for cls in classes:                                 # derived 先，同名方法不重扫
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name in seen_methods \
                    or fn.name == "__init__":
                continue
            seen_methods.add(fn.name)
            scan = _MethodScan(binds, src_file, fn.name)
            for s in fn.body:
                scan.visit(s)
            sites.extend(scan.sites)
    return sites
```

- [ ] **Step 4: 跑测试**

Run: `python -m pytest tests/test_opdag_comm_probe.py -q` → 3 PASS。
若 `test_vocab_parallel_embedding_sites` 失败：读 layers.py 实际分支，按真源修 guard 断言
（只许改**测试期望到实际源**，不许改探测器去凑）。若 Column 意外探出通信 → 更新 Task 9 的注入策略
（改用 probe 结果，删模块语义注入），并把发现记进 commit message。

- [ ] **Step 5: 全量回归 + Commit**

```bash
python -m pytest tests/ -q
git add cost_eval/opdag/comm_probe.py tests/test_opdag_comm_probe.py
git commit -m "feat(opdag): comm_probe TP 通信惯用法静态探测(T0-3,R1 打通,Row/Embedding 双惯用法)"
```

## Task 5: extractor 直接实例化子 Cell 绑定（惯用法B）

**Files:**
- Modify: `cost_eval/opdag/extractor.py:440`（`combined = {...}` 之后插入）
- Modify: `cost_eval/opdag/init_binder.py:9-26`（`_CLS2OP` 增补）
- Test: `tests/test_opdag_gpt_segments.py`（本任务先写 loss 部分）

**动机**：loss 组合是 `self._log_softmax = _LogSoftmax(config)`（gpt_model.py:373 →
loss_func.py:279-280）——直接实例化，不走 `build_module`，现有 extractor 绑不上。

- [ ] **Step 1: 写失败测试（loss 段端到端提取）**

```python
# tests/test_opdag_gpt_segments.py
"""GPTModel 级段提取（spec §3.3a）：loss / embedding / lm_head + construct 段序核对。"""
import os
import pytest

from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.module_resolver import ResolvedSpec

MF_ROOT = os.environ.get("MINDFORMERS_ROOT",
                         r"E:\97-codes\torch_parallel\mindformers\mindformers")
LOSS_REL = "parallel_core/training_graph/loss_func.py"


def _require_mf():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")


LOSS_FLAGS = {"compute_dtype": "bf16", "add_bias_linear": False}


def test_extract_cross_entropy_loss_inlines_subcells():
    _require_mf()
    dag = extract_cell(
        MF_ROOT, LOSS_REL, "CrossEntropyLoss",
        ResolvedSpec(cell="CrossEntropyLoss", submodules={}),
        LOSS_FLAGS, recurse=True,
        subcell_specs={
            "_LogSoftmax": ResolvedSpec(cell="_LogSoftmax", submodules={}),
            "_NLLLoss": ResolvedSpec(cell="_NLLLoss", submodules={}),
        },
    )
    assert len(dag.nodes) >= 5                              # 两子 Cell 已内联（非 2 个 SubCell 占位）
    assert all(n.op != "SubCell" for n in dag.nodes)
    assert all(n.src.split(":")[0] == "loss_func.py" for n in dag.nodes)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_opdag_gpt_segments.py -q`
Expected: FAIL——fail-loud 报 `_log_softmax` 未知绑定（或 SubCell 未解析）。

- [ ] **Step 3: extractor 加惯用法B 绑定**

在 `extractor.py` 的 `_extract_meta` 中 `combined = {**base_binds, **named}` 之后插入：

```python
    # 1b) 直接实例化的子 Cell（惯用法B，spec §3.3a）：`self.X = <ClsName>(...)` 且
    #     ClsName ∈ subcell_specs → SubCell(bare)。build_module 之外的第二种子 Cell 组合方式
    #     （loss: `self._log_softmax = _LogSoftmax(config)`，loss_func.py:279）。
    #     subcell_specs 未提供（内存侧全部既有调用）时零行为变化。
    if recurse and subcell_specs:
        from .init_binder import _base_call_name
        for cname in reversed(init_classes):
            cls_node = _find_class(tree, cname)
            init_fn = _method_of(cls_node, "__init__")
            if init_fn is None:
                continue
            for stmt in ast.walk(init_fn):
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                    continue
                tgt = stmt.targets[0]
                if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self" and isinstance(stmt.value, ast.Call)):
                    continue
                ctor = _base_call_name(stmt.value)
                if ctor in subcell_specs and tgt.attr not in combined:
                    combined[tgt.attr] = Binding(
                        op="SubCell", attrs={"cell": ctor, "field": tgt.attr, "bare": True})
```

（`Binding` 已由 `from .init_binder import bind_init, Binding` 在文件头引入；`_base_call_name`
就地 import 避免顶部循环。）

- [ ] **Step 4: 跑测试，按 fail-loud 输出增补 `_CLS2OP`**

Run: `python -m pytest tests/test_opdag_gpt_segments.py -q`
预期此时 fail-loud 会报 `_LogSoftmax`/`_NLLLoss` construct 里的未知原语绑定。按下表把**实际报出的**
类名加进 `init_binder._CLS2OP`（只加报出的，不预加）：

```python
    # loss/embedding 段原语（T0-4 增补;语义=教科书 VJP 分类,与 bprop_rules 口径一致）
    "Exp": ("Elementwise", {"linear": False}),
    "Log": ("Elementwise", {"linear": False}),
    "Neg": ("Elementwise", {"linear": True}),
    "ReduceSum": ("Elementwise", {"linear": True, "reduce": True}),
    "ReduceMax": ("Elementwise", {"linear": False, "reduce": True}),
    "OneHot": ("Elementwise", {"linear": True}),
    "GatherD": ("Gather", {}),
    "ReLU": ("Activation", {"activation_type": "relu"}),
    "Minimum": ("Elementwise", {"linear": False}),
    "Equal": ("Elementwise", {"linear": True}),
    "StridedSlice": ("View", {"view": "slice"}),
```

若报的是**自由函数调用**（非 `self.X`，如 `mint.*`/`ops.functional`）→ 记录该调用行到测试文件
docstring，并在 `construct_walker` 的既有未知调用处理处确认其 fail-loud 信息可定位；这类缺口若
阻塞 loss 主链（log_softmax→nll 主路径断边）则本任务 blocked，上报用户裁决；只断旁路（如
label smoothing 分支被 config 剪枝掉）则按剪枝处理继续。

- [ ] **Step 5: 测试过 + 全量回归（内存侧零变化：`subcell_specs` 新语义只在显式传参时激活）**

Run: `python -m pytest tests/ -q` → N 增长，无回退。

- [ ] **Step 6: Commit**

```bash
git add cost_eval/opdag/extractor.py cost_eval/opdag/init_binder.py tests/test_opdag_gpt_segments.py
git commit -m "feat(opdag): 直接实例化子Cell绑定(惯用法B)+loss原语_CLS2OP增补,CrossEntropyLoss 段可提取(T0-4)"
```

## Task 6: gpt_segments——embedding / lm_head 段 + GPTModel 段序核对

**Files:**
- Create: `cost_eval/opdag/gpt_segments.py`
- Test: `tests/test_opdag_gpt_segments.py`（追加）

- [ ] **Step 1: 追加失败测试**

```python
# tests/test_opdag_gpt_segments.py 追加

def test_extract_embedding_walks_morph_func():
    _require_mf()
    from cost_eval.opdag.gpt_segments import extract_embedding
    dag = extract_embedding(MF_ROOT, {"compute_dtype": "bf16"})
    ops = [n.op for n in dag.nodes]
    assert "Gather" in ops or "Embedding" in ops          # mint embedding lookup
    assert all(n.src.split(":")[0] == "layers.py" for n in dag.nodes)


def test_lm_head_segment_source_pinned():
    _require_mf()
    from cost_eval.opdag.gpt_segments import head_segment_dag
    dag = head_segment_dag()
    assert [n.op for n in dag.nodes] == ["MatMul", "View", "View", "Cast"]
    assert dag.nodes[0].module == "ColumnParallelLinear"
    assert all(n.src.startswith("gpt_model.py:") for n in dag.nodes)
    assert dag.nodes[-1].attrs.get("to_dtype") == "fp32"  # logits cast fp32（gpt_model.py:509）


def test_verify_gpt_order():
    _require_mf()
    from cost_eval.opdag.gpt_segments import verify_gpt_order
    order = verify_gpt_order(MF_ROOT)
    lm = order.index("language_model")
    head = order.index("output_layer")
    loss = order.index("compute_language_model_loss")
    assert lm < head < loss
```

- [ ] **Step 2: 跑测试确认失败**（`ModuleNotFoundError: gpt_segments`）

- [ ] **Step 3: 实现 gpt_segments.py**

```python
# cost_eval/opdag/gpt_segments.py
"""GPTModel 级非重复段（spec §3.3a）：embedding / lm_head / loss 提取 + construct 段序核对。
transformer 层段沿用既有 per-cell 提取。所有节点 src 回指真源（源忠实链完整）。

事实源（mindformers @ parallel_core/training_graph，行号为 2026-07-16 基线，漂移时按惯用法重定位）：
  embedding: tensor_parallel/layers.py VocabParallelEmbedding——construct 调 self.embedding_morph
             （P.Morph(self.embedding_func)，__init__ 内绑定）→ 走既有 _morph_aliases 内联机制。
  lm_head:   gpt_model.py:364 `self.output_layer = ColumnParallelLinear(hidden→vocab)`；
             construct :505-509 logits = output_layer(h) → transpose → morphed_reshape → cast(fp32)。
             GPTModel.construct 含 mtp/eod/zbv 等大量 config 分支，T0 不整体走查——head 段由本模块
             按上述已核实调用序**合成**（每节点 src 钉到真源行,顺序由 verify_gpt_order 的 AST 断言
             守护;整体走查待 v1.5 MTP 时一并做）。
  loss:      loss_func.py CrossEntropyLoss（_LogSoftmax + _NLLLoss 直接实例化组合,Task 5 惯用法B）。
"""
from __future__ import annotations

import ast
import os

from .extractor import extract_cell, _find_class, _method_of
from .module_resolver import ResolvedSpec
from .schema import OpDAG, OpNode

TP_REL = "parallel_core/training_graph/tensor_parallel/layers.py"
GPT_REL = "parallel_core/training_graph/base_models/gpt/gpt_model.py"
LOSS_REL = "parallel_core/training_graph/loss_func.py"


def extract_loss(mf_root: str, config_flags: dict) -> OpDAG:
    return extract_cell(
        mf_root, LOSS_REL, "CrossEntropyLoss",
        ResolvedSpec(cell="CrossEntropyLoss", submodules={}),
        config_flags, recurse=True,
        subcell_specs={
            "_LogSoftmax": ResolvedSpec(cell="_LogSoftmax", submodules={}),
            "_NLLLoss": ResolvedSpec(cell="_NLLLoss", submodules={}),
        },
    )


def extract_embedding(mf_root: str, config_flags: dict) -> OpDAG:
    return extract_cell(
        mf_root, TP_REL, "VocabParallelEmbedding",
        ResolvedSpec(cell="VocabParallelEmbedding", submodules={}),
        config_flags,
    )


def head_segment_dag() -> OpDAG:
    """lm_head 段（合成，逐节点钉真源行；结构由 verify_gpt_order 守护——见模块 docstring）。"""
    nodes = [
        OpNode(id=0, op="MatMul", src="gpt_model.py:505", module="ColumnParallelLinear",
               ins=["h:S·B·H:bf16", "W_head:H·vocab:bf16"], out="logits:S·B·vocab:bf16"),
        OpNode(id=1, op="View", src="gpt_model.py:507",
               ins=["logits:S·B·vocab:bf16"], out="logits_t:B·S·vocab:bf16",
               attrs={"view": "transpose"}),
        OpNode(id=2, op="View", src="gpt_model.py:508",
               ins=["logits_t:B·S·vocab:bf16"], out="logits_2d:S·B·vocab:bf16",
               attrs={"view": "reshape"}),
        OpNode(id=3, op="Cast", src="gpt_model.py:509",
               ins=["logits_2d:S·B·vocab:bf16"], out="logits32:S·B·vocab:fp32",
               attrs={"to_dtype": "fp32"}),
    ]
    return OpDAG(cell="GPTModel.head", nodes=nodes, edges=[[0, 1], [1, 2], [2, 3]])


def verify_gpt_order(mf_root: str) -> list[str]:
    """AST 读 GPTModel.construct，按语句序返回 `self.X(...)`/`self.X(...)` 调用的 attr 名列表；
    language_model → output_layer → compute_language_model_loss 缺一或错序 → fail-loud。"""
    path = os.path.join(mf_root, *GPT_REL.split("/"))
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    cls = _find_class(tree, "GPTModel")
    construct = _method_of(cls, "construct") if cls else None
    if construct is None:
        raise ValueError("gpt_segments: GPTModel.construct 定位失败（fail-loud）")
    order: list[str] = []
    for node in ast.walk(construct):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            order.append(node.func.attr)
    for need in ("language_model", "output_layer", "compute_language_model_loss"):
        if need not in order:
            raise ValueError(f"gpt_segments: GPTModel.construct 里找不到 self.{need}(...) —— "
                             f"源结构变了，段合成失效（fail-loud）")
    if not (order.index("language_model") < order.index("output_layer")
            < order.index("compute_language_model_loss")):
        raise ValueError("gpt_segments: GPTModel 段序变化（fail-loud）")
    return order
```

- [ ] **Step 4: 跑测试，按 fail-loud 分诊 embedding**

Run: `python -m pytest tests/test_opdag_gpt_segments.py -q`
embedding 走查预期缺口与处置（同 Task 5 纪律：只修报出的）：
- `ReLU/Minimum/Equal` 未绑 → Task 5 的 `_CLS2OP` 表已含，确认生效；
- `mint.nn.functional.embedding(...)` 自由函数调用 → 在 `construct_walker` 的调用识别处增加一条
  惯用法：`mint.nn.functional.embedding` / `F.embedding` → 发射 `OpNode(op="Gather",
  attrs={"embedding": True})`（教科书语义：embedding lookup = 行 gather）。加在既有"未知调用
  fail-loud"分支之前，模式与频次注释引用 layers.py 实际行号；
- `output_parallel.transpose(1,0,2)` 方法式调用 → 若 walker 未识别，同处加 `.transpose/.reshape`
  方法调用 → `View` 节点（这与 §3.2 词表一致）。
若缺口超出上述三类且断主链 → blocked，上报。

- [ ] **Step 5: 全量回归 + Commit**

```bash
python -m pytest tests/ -q
git add cost_eval/opdag/gpt_segments.py cost_eval/opdag/construct_walker.py tests/test_opdag_gpt_segments.py
git commit -m "feat(opdag): embedding/lm_head/loss 段提取 + GPTModel 段序核对(T0-5,覆盖缺口 crosscheck.py:38 关闭三项)"
```

---

# Phase C — TimedOpSeq producer

## Task 7: timesim/ir.py 数据结构

**Files:**
- Create: `cost_eval/timesim/ir.py`
- Test: `tests/test_timesim_ir.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_timesim_ir.py
"""TimedOp/TimedSegment/CommSpec（spec §3.2）+ op_flops 纯函数。"""
import dataclasses
import pytest

from cost_eval.timesim.ir import TimedOp, TimedSegment, CommSpec, op_flops


def _op(**kw):
    base = dict(op_id="n0", op_type="MatMul", phase="fwd",
                in_shapes=((4096, 1, 1792), (1792, 3072)), out_shape=(4096, 1, 3072),
                dtype="bf16", stream="device", src="mlp.py:1")
    base.update(kw)
    return TimedOp(**base)


def test_timedop_frozen():
    top = _op()
    with pytest.raises(dataclasses.FrozenInstanceError):
        top.phase = "bwd"


def test_op_flops_matmul():
    # 2·M·K·N：M=4096·1（batch 摊平）、K=1792、N=3072
    assert op_flops(_op()) == 2 * 4096 * 1 * 1792 * 3072


def test_op_flops_grouped_matmul_sums_groups():
    top = _op(op_type="GroupedMatMul",
              in_shapes=((8, 512, 1792), (8, 1792, 1024)), out_shape=(8, 512, 1024))
    assert op_flops(top) == 8 * (2 * 512 * 1792 * 1024)


def test_op_flops_nonmatmul_zero():
    assert op_flops(_op(op_type="Norm")) == 0        # FA/带宽类 flops 归 op_cost（T1）


def test_commspec_on_comm_op():
    c = CommSpec(ctype="reduce_scatter", volume_bytes=4096 * 1792 * 2,
                 group_axis="tp", group_size=2)
    top = _op(op_type="CommOp", stream="comm_tp", comm=c, in_shapes=(), out_shape=())
    assert top.comm.ctype == "reduce_scatter"


def test_segment_holds_ops():
    seg = TimedSegment(seg_id="layer_0.fwd", ops=(_op(),))
    assert seg.ops[0].op_type == "MatMul"
```

- [ ] **Step 2: 跑确认失败** → **Step 3: 实现**

```python
# cost_eval/timesim/ir.py
"""TimedOpSeq IR（spec §3.2）：timesim 的唯一输入数据结构 + op_flops 纯函数。

流标注（spec §5.1，cp 按"通信按 group 轴分道"补为独立道）：
  device | host_only | comm_tp | comm_cp | comm_ep | comm_dp | comm_pp
op_flops 只算 GEMM 族（6ND 不变量的主体）；FA/带宽类的时间成本由 op_cost（T1）经经验库给出，
不在 IR 层杜撰系数。"""
from __future__ import annotations

from dataclasses import dataclass

STREAM_DEVICE = "device"
STREAM_HOST_ONLY = "host_only"
COMM_STREAM = {"tp": "comm_tp", "cp": "comm_cp", "ep": "comm_ep",
               "dp": "comm_dp", "pp": "comm_pp"}


@dataclass(frozen=True)
class CommSpec:
    ctype: str            # all_reduce | reduce_scatter | all_gather | all_to_all | p2p
    volume_bytes: int     # 本 rank 载荷字节（(n-1)/n 等算法系数归 op_cost，T1）
    group_axis: str       # tp | cp | ep | dp | pp
    group_size: int


@dataclass(frozen=True)
class TimedOp:
    op_id: str                       # 回指 opdag 节点（"<cell>#<id>" / 展开后缀 ".bK"/".rK"）
    op_type: str                     # opdag 词表 + CommOp/…Grad
    phase: str                       # fwd | bwd | recomp
    in_shapes: tuple                 # tuple[tuple[int,...],...]，已代入 local
    out_shape: tuple
    dtype: str
    stream: str
    src: str = ""                    # mindformers file:line（源忠实）
    deps: tuple = ()                 # 跨流依赖的 op_id（同流 FIFO 隐含）
    comm: CommSpec | None = None
    module: str = ""                 # ColumnParallelLinear 等（shard/通信语义键）


@dataclass(frozen=True)
class TimedSegment:
    seg_id: str                      # "layer_3.fwd" / "embedding.fwd" / "loss.bwd" …
    ops: tuple


def op_flops(top: TimedOp) -> int:
    """GEMM 族 FLOPs = **2 · numel(A) · N_out**（A=in_shapes[0]，N_out=out_shape 末轴）。

    该式对 fwd（C=A·B：numel(A)=M·K）、bwd 的 dX=dy·Bᵀ（numel(dy)=M·N，N_out=K）、
    dW=Aᵀ·dy（numel(A)=M·K，N_out=N）**一致成立**——收缩维总在 numel(A) 里，无需分情况；
    GroupedMatMul 同式（expert 维在 numel(A) 里）。朴素 `k=in[0][-1]` 启发式对 dW 会取错
    收缩维（k 取成 K 而非 M），故弃用。其余 op 返回 0（见模块 docstring）。"""
    if top.op_type in ("MatMul", "GroupedMatMul") \
            and top.in_shapes and top.in_shapes[0] and top.out_shape:
        a = 1
        for d in top.in_shapes[0]:
            a *= d
        return 2 * a * top.out_shape[-1]
    return 0
```

- [ ] **Step 4: 跑测试 + 全量** → **Step 5: Commit**

```bash
git add cost_eval/timesim/ir.py tests/test_timesim_ir.py
git commit -m "feat(timesim): TimedOp/TimedSegment/CommSpec IR + op_flops(T0-6,spec §3.2)"
```

## Task 8: shard_rules——并行代入

**Files:**
- Create: `cost_eval/timesim/shard_rules.py`
- Test: `tests/test_timesim_producer.py`（先写 shard 部分）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_timesim_producer.py
"""shard_rules + producer（spec §3.3 b/c）。"""
import pytest

from cost_eval.timesim.shard_rules import Degrees, axis_values, localize
from cost_eval.model_spec import DimTable

DIMS = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                S=4096, B=1, vocab=129280, n_layers=4)


def test_axis_values_resolves_symbols():
    assert axis_values("S·B·H", DIMS) == [4096, 1, 1792]


def test_axis_values_fail_loud_on_unknown():
    with pytest.raises(ValueError):
        axis_values("S·B·unknown_dim", DIMS)


def test_localize_seq_axis_cp_and_sp():
    deg = Degrees(tp=2, cp=2, sequence_parallel=True)
    # sp 驻留区：S 轴 ÷(cp·tp)；feature 不动
    assert localize([4096, 1, 1792], "S·B·H", deg, sp_active=True) == [1024, 1, 1792]
    # 非 sp 驻留区：只 ÷cp
    assert localize([4096, 1, 1792], "S·B·H", deg, sp_active=False) == [2048, 1, 1792]


def test_localize_feature_and_expert_axes():
    deg = Degrees(tp=2, ep=4)
    # Column 出激活末轴 ÷tp。gated fc1 出维是带系数单轴 "(2·ffn_hidden)"——sym_shape 语法：
    # 顶层 `·` 分轴，系数轴须括号（sym_shape.py 模块头）。
    assert localize([4096, 1, 6144], "S·B·(2·ffn_hidden)", deg, feat_div_last=deg.tp) \
        == [4096, 1, 3072]
    # expert 轴（E）÷ep
    assert localize([8, 512, 1024], "E·cap·moe_ffn", deg) == [2, 512, 1024]


def test_weight_local_divides_correct_axis():
    from cost_eval.timesim.shard_rules import weight_local
    # Column 权重 [H, 2F]：out 维=末轴 ÷tp；Row 权重 [F, H]：in 维=轴0 ÷tp（Megatron 语义）
    assert weight_local("H·(2·ffn_hidden)", DIMS, "ColumnParallelLinear", 2) == (1792, 3072)
    assert weight_local("ffn_hidden·H", DIMS, "RowParallelLinear", 2) == (1536, 1792)
```

- [ ] **Step 2: 跑确认失败** → **Step 3: 实现**

```python
# cost_eval/timesim/shard_rules.py
"""并行代入（spec §3.3b）：符号 shape → local 具体 shape。

符号 token 求值复用 opdag.consumer 的机制（上游共享 IR 资产，契约 §2.2-2 允许——注意这不是
内存仿真模块）。轴语义规则：
  - 含 S 的轴 ÷cp；sp_active（producer 的 SP 状态机给出）时再 ÷tp；
  - 含 E 的轴 ÷ep；
  - feature 轴按调用方给的 feat_div_last（模块语义：Column 出 ÷tp、Row 入已 ÷tp）。
整除性 fail-loud：并行度不整除即报错（评估合法性校验，serve_explorer 同款规则的时间侧）。"""
from __future__ import annotations

from dataclasses import dataclass

from ..opdag.consumer import _axis_value
from ..opdag.sym_shape import parse_shape


@dataclass(frozen=True)
class Degrees:
    tp: int = 1
    cp: int = 1
    ep: int = 1
    dp: int = 1
    pp: int = 1
    sequence_parallel: bool = False


def axis_values(sym_shape: str, dims) -> list[int]:
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
        out[-1] = _div_exact(out[-1], feat_div_last, "feature(tp)")
    return out


def weight_local(sym: str, dims, module: str, tp: int) -> tuple:
    """线性层**权重**的 local shape（切分轴随模块语义，不是一律末轴）：
    ColumnParallelLinear 权重 [in, out] → out(末轴) ÷tp；
    RowParallelLinear    权重 [in, out] → in(轴0)  ÷tp。"""
    vals = axis_values(sym, dims)
    if tp > 1:
        if module == "ColumnParallelLinear":
            vals[-1] = _div_exact(vals[-1], tp, "col-weight out")
        elif module == "RowParallelLinear":
            vals[0] = _div_exact(vals[0], tp, "row-weight in")
    return tuple(vals)
```

- [ ] **Step 4: 跑测试 + 全量** → **Step 5: Commit**

```bash
git add cost_eval/timesim/shard_rules.py tests/test_timesim_producer.py
git commit -m "feat(timesim): shard_rules 并行代入(T0-7,轴语义×度数,整除 fail-loud)"
```

## Task 9: producer——fwd 段装配 + TP 通信注入 + SP 状态机

**Files:**
- Create: `cost_eval/timesim/producer.py`
- Test: `tests/test_timesim_producer.py`（追加）

- [ ] **Step 1: 追加失败测试（用真提取的 MLP DAG）**

```python
# tests/test_timesim_producer.py 追加
import os

MF_ROOT = os.environ.get("MINDFORMERS_ROOT",
                         r"E:\97-codes\torch_parallel\mindformers\mindformers")


def _mlp_dag():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import ResolvedSpec
    return extract_cell(
        MF_ROOT, "parallel_core/training_graph/transformer/mlp.py", "MLPInterleaved",
        ResolvedSpec(cell="MLPInterleaved",
                     submodules={"linear_fc1": "ColumnParallelLinear",
                                 "linear_fc2": "RowParallelLinear"}),
        {"gated_linear_unit": True, "activation_type": "silu",
         "add_bias_linear": False, "compute_dtype": "bf16"})


def test_build_segment_tp2_sp_injects_comm():
    from cost_eval.timesim.producer import build_segment
    deg = Degrees(tp=2, cp=1, sequence_parallel=True)
    seg = build_segment("layer_0.mlp.fwd", _mlp_dag(), DIMS, deg)
    kinds = [(o.op_type, o.stream) for o in seg.ops]
    comms = [o for o in seg.ops if o.op_type == "CommOp"]
    # Column 前 sp all-gather（模块语义注入）+ Row 后 reduce_scatter（comm_probe 源惯用法）
    assert [c.comm.ctype for c in comms] == ["all_gather", "reduce_scatter"]
    assert all(c.stream == "comm_tp" for c in comms)
    # fc1 出 feature ÷tp：gated 2F=6144 → 3072；且 all_gather 后 sp 退出 → seq 全长
    fc1 = next(o for o in seg.ops if o.op_type == "MatMul")
    assert fc1.out_shape == (4096, 1, 3072)
    # View 全部 host_only
    assert all(o.stream == "host_only" for o in seg.ops if o.op_type == "View")
    # reduce_scatter 载荷 = 全 seq 输出字节（S·B·H·2B）
    assert comms[-1].comm.volume_bytes == 4096 * 1 * 1792 * 2


def test_build_segment_tp1_has_no_comm():
    from cost_eval.timesim.producer import build_segment
    seg = build_segment("layer_0.mlp.fwd", _mlp_dag(), DIMS, Degrees())
    assert all(o.op_type != "CommOp" for o in seg.ops)
```

- [ ] **Step 2: 跑确认失败** → **Step 3: 实现 producer.py**

```python
# cost_eval/timesim/producer.py
"""TimedOpSeq producer（spec §3.3：b 并行代入 + c 通信注入装配）。

单元 = 一个 cell DAG → 一个 TimedSegment（fwd）。SP 状态机（spec §5.5 结构性 overlap 的
"位置"基础）：sp_active 从段边界起为 config.sequence_parallel；
  - ColumnParallelLinear 且 sp_active 且 tp>1：矩乘**前**注入 all_gather（模块语义注入——
    Column 源无显式通信，comm_probe 已证空，spec §3.3c 允许并记 injected="module-semantics"），
    sp 退出（激活恢复全 seq、feature 分片）；
  - RowParallelLinear 且 tp>1：矩乘**后**注入 reduce_scatter（sp，进入 sp_active）或
    all_reduce（非 sp）——源惯用法 layers.py:619/:621（comm_probe 实证）。
DAG 内联的显式 CommOp 节点（embedding 的 ReduceScatter）直接映射。
dtype 取 opdag 节点 cast 后标注；View → host_only。deps = opdag 数据边。"""
from __future__ import annotations

from .ir import (TimedOp, TimedSegment, CommSpec,
                 STREAM_DEVICE, STREAM_HOST_ONLY, COMM_STREAM)
from .shard_rules import Degrees, axis_values, localize, weight_local

_COL = "ColumnParallelLinear"
_ROW = "RowParallelLinear"


def _parse_ref(ref: str):
    """opdag 的 'name:符号shape:dtype' → (name, sym_shape, dtype)。"""
    parts = ref.split(":")
    if len(parts) != 3:
        raise ValueError(f"producer: 非法 TensorRef {ref!r}（fail-loud）")
    return parts[0], parts[1], parts[2]


def _bytes_of(shape, dtype: str) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * (4 if dtype == "fp32" else 2)


def _local(sym: str, dims, deg: Degrees, sp_active: bool, feat_div: int = 1):
    vals = axis_values(sym, dims)
    return tuple(localize(vals, sym, deg, feat_div_last=feat_div, sp_active=sp_active))


def build_segment(seg_id: str, dag, dims, deg: Degrees, *, phase: str = "fwd") -> TimedSegment:
    cell = dag.cell
    id2opid = {n.id: f"{cell}#{n.id}" for n in dag.nodes}
    in_edges: dict[int, list[int]] = {}
    for s, d in dag.edges:
        in_edges.setdefault(d, []).append(s)

    ops: list[TimedOp] = []
    sp_active = deg.sequence_parallel
    for n in dag.nodes:
        deps = tuple(id2opid[s] for s in in_edges.get(n.id, ()))
        out_name, out_sym, out_dt = _parse_ref(n.out) if n.out else ("", "", "bf16")
        module = n.module or ""

        # —— Column 前置 sp all-gather（模块语义注入）——
        if module == _COL and deg.tp > 1 and sp_active and n.op == "MatMul":
            x_sym = _parse_ref(n.ins[0])[1]
            gather_in = _local(x_sym, dims, deg, sp_active=True)
            ops.append(TimedOp(
                op_id=id2opid[n.id] + ".ag", op_type="CommOp", phase=phase,
                in_shapes=(gather_in,), out_shape=_local(x_sym, dims, deg, sp_active=False),
                dtype=_parse_ref(n.ins[0])[2], stream=COMM_STREAM["tp"], src=n.src,
                deps=deps, module="injected:module-semantics",
                comm=CommSpec("all_gather", _bytes_of(gather_in, _parse_ref(n.ins[0])[2]),
                              "tp", deg.tp)))
            deps = (ops[-1].op_id,)
            sp_active = False

        # —— 本体节点 ——
        feat_div = deg.tp if (module == _COL and n.op == "MatMul" and deg.tp > 1) else 1
        if n.op == "CommOp" or n.op in ("AllReduce", "ReduceScatter", "AllGather"):
            # DAG 内联显式通信（embedding 惯用法B 提取产物）
            ctype = n.attrs.get("ctype") or {"AllReduce": "all_reduce",
                                             "ReduceScatter": "reduce_scatter",
                                             "AllGather": "all_gather"}.get(n.op, "all_reduce")
            in_sym = _parse_ref(n.ins[0])[1]
            shp = _local(in_sym, dims, deg, sp_active=sp_active)
            ops.append(TimedOp(op_id=id2opid[n.id], op_type="CommOp", phase=phase,
                               in_shapes=(shp,), out_shape=shp, dtype=out_dt or "bf16",
                               stream=COMM_STREAM["tp"], src=n.src, deps=deps, module=module,
                               comm=CommSpec(ctype, _bytes_of(shp, out_dt or "bf16"),
                                             "tp", deg.tp)))
            if ctype == "reduce_scatter":
                sp_active = deg.sequence_parallel
            continue

        stream = STREAM_HOST_ONLY if n.op == "View" else STREAM_DEVICE
        row_matmul = module == _ROW and n.op == "MatMul" and deg.tp > 1
        col_matmul = module == _COL and n.op == "MatMul" and deg.tp > 1
        in_shapes = []
        for i, ref in enumerate(n.ins):
            _, sym, _dt = _parse_ref(ref)
            if not sym or sym == "?":
                in_shapes.append(())          # `?` 只容忍于 View/host_only（见下）
                continue
            if (row_matmul or col_matmul) and i == 1:
                # 权重切分轴随模块语义（Column 末轴 / Row 轴0），不是一律末轴
                in_shapes.append(weight_local(sym, dims, module, deg.tp))
                continue
            # 激活输入：Row 的输入 feature 已按 tp 分片（上游 Column 出）→ 末轴 ÷tp
            div = deg.tp if (row_matmul and i == 0) else 1
            in_shapes.append(_local(sym, dims, deg, sp_active=sp_active, feat_div=div))
        # GroupedMatMul 的 E 轴已由 localize 按 ep 除
        out_shape = _local(out_sym, dims, deg, sp_active=sp_active, feat_div=feat_div) \
            if out_sym and out_sym != "?" else ()
        if stream == STREAM_DEVICE and n.op != "View" and out_sym == "?":
            raise ValueError(f"producer: device op {n.src} 输出 shape 未解析（fail-loud）")
        ops.append(TimedOp(op_id=id2opid[n.id], op_type=n.op, phase=phase,
                           in_shapes=tuple(in_shapes), out_shape=out_shape,
                           dtype=out_dt or "bf16", stream=stream, src=n.src,
                           deps=deps, module=module))

        # —— Row 后置 reduce_scatter / all_reduce（源惯用法 layers.py:619/:621）——
        if row_matmul:
            full = _local(out_sym, dims, deg, sp_active=False)
            ctype = "reduce_scatter" if deg.sequence_parallel else "all_reduce"
            out_after = _local(out_sym, dims, deg, sp_active=deg.sequence_parallel)
            ops.append(TimedOp(op_id=id2opid[n.id] + ".rs", op_type="CommOp", phase=phase,
                               in_shapes=(full,), out_shape=out_after, dtype=out_dt,
                               stream=COMM_STREAM["tp"],
                               src="layers.py:619" if ctype == "reduce_scatter" else "layers.py:621",
                               deps=(id2opid[n.id],), module=module,
                               comm=CommSpec(ctype, _bytes_of(full, out_dt), "tp", deg.tp)))
            sp_active = deg.sequence_parallel
    return TimedSegment(seg_id=seg_id, ops=tuple(ops))
```

**实现注意**：`?` shape（loss 段个别 View）→ 空 tuple 占位；device op 输出遇 `?` → fail-loud
（代码里已含该分支，报节点 src）。

- [ ] **Step 4: 跑测试（两条都过）+ 全量** → **Step 5: Commit**

```bash
git add cost_eval/timesim/producer.py tests/test_timesim_producer.py
git commit -m "feat(timesim): producer fwd 段装配+TP通信注入+SP状态机(T0-8,spec §3.3b/c)"
```

## Task 10: 框架层通信注入（FSDP / EP / CP）

**Files:**
- Create: `cost_eval/timesim/frame_comm.py`
- Test: `tests/test_timesim_producer.py`（追加）

- [ ] **Step 1: 追加失败测试**

```python
# tests/test_timesim_producer.py 追加

def test_fsdp_gather_injected_at_segment_head():
    from cost_eval.timesim.producer import build_segment
    from cost_eval.timesim.frame_comm import inject_fsdp
    seg = build_segment("layer_0.mlp.fwd", _mlp_dag(), DIMS, Degrees())
    seg2 = inject_fsdp(seg, dp_shard=4, dims=DIMS)
    first = seg2.ops[0]
    assert first.op_type == "CommOp" and first.comm.ctype == "all_gather"
    assert first.stream == "comm_dp" and first.comm.group_axis == "dp"
    # 载荷 = 段内权重字节（fc1 H·2F + fc2 F·H，bf16）
    w = (1792 * 2 * 3072 + 3072 * 1792) * 2
    assert first.comm.volume_bytes == w


def test_cp_ring_p2p_wraps_flash_attention():
    from cost_eval.timesim.ir import TimedOp, TimedSegment
    from cost_eval.timesim.frame_comm import inject_cp
    fa = TimedOp(op_id="a#0", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224),) * 3, out_shape=(2048, 1, 8, 224),
                 dtype="bf16", stream="device", src="attention.py:1")
    seg = inject_cp(TimedSegment("l.fwd", (fa,)), cp=2, method="colossal")
    kinds = [o.comm.ctype for o in seg.ops if o.op_type == "CommOp"]
    assert kinds == ["p2p"]                       # ring：每 FA 伴一跳 kv p2p（cp-1 跳归 op_cost 系数）
    assert seg.ops[0].stream == "comm_cp"
```

- [ ] **Step 2: 跑确认失败** → **Step 3: 实现**

```python
# cost_eval/timesim/frame_comm.py
"""框架层通信注入（spec §3.3c 第二类——不在 layer construct 源码里的通信）。

  FSDP：per-segment 权重 all-gather 注入段**头部**（fwd 预取语义 → pipeline 期与上段计算重叠，
        spec §5.5）；载荷 = 段内 MatMul/GroupedMatMul 权重操作数（ins[1]，is-weight 惯用名 W*/
        weight）字节 ——由 TimedOp.in_shapes 直接求和，不引入内存侧任何模块。
  EP  ：GroupedMatMul 段两侧注入 dispatch/combine all-to-all（balanced 口径：载荷=本 rank
        dispatched tokens 字节 = GroupedMatMul 输入激活字节）。
  CP  ：colossal ring → 每 FlashAttention 前注入一条 kv p2p（cp-1 跳/对分系数归 op_cost，T1）；
        ulysses → FA 前后各一条 all_to_all。
bwd 对偶由 bwd_rules 的 CommOp 对偶规则统一处理，此处只管 fwd 位置。"""
from __future__ import annotations

from .ir import TimedOp, TimedSegment, CommSpec, COMM_STREAM


def _weight_bytes(seg: TimedSegment) -> int:
    total = 0
    for o in seg.ops:
        if o.op_type in ("MatMul", "GroupedMatMul") and len(o.in_shapes) >= 2:
            n = 1
            for d in o.in_shapes[1]:
                n *= d
            total += n * (4 if o.dtype == "fp32" else 2)
    return total


def _act_bytes(shape, dtype="bf16") -> int:
    n = 1
    for d in shape:
        n *= d
    return n * (4 if dtype == "fp32" else 2)


def inject_fsdp(seg: TimedSegment, dp_shard: int, dims) -> TimedSegment:
    if dp_shard <= 1:
        return seg
    w = _weight_bytes(seg)
    gather = TimedOp(op_id=f"{seg.seg_id}.fsdp_ag", op_type="CommOp", phase=seg.ops[0].phase,
                     in_shapes=(), out_shape=(), dtype="bf16", stream=COMM_STREAM["dp"],
                     src="fsdp:module-semantics",
                     comm=CommSpec("all_gather", w, "dp", dp_shard))
    return TimedSegment(seg.seg_id, (gather,) + seg.ops)


def inject_ep(seg: TimedSegment, ep: int) -> TimedSegment:
    if ep <= 1:
        return seg
    ops = list(seg.ops)
    idx = [i for i, o in enumerate(ops) if o.op_type == "GroupedMatMul"]
    if not idx:
        return seg
    first, last = idx[0], idx[-1]
    tok = _act_bytes(ops[first].in_shapes[0], ops[first].dtype)
    ph = ops[first].phase
    disp = TimedOp(op_id=f"{seg.seg_id}.ep_disp", op_type="CommOp", phase=ph,
                   in_shapes=(), out_shape=(), dtype=ops[first].dtype,
                   stream=COMM_STREAM["ep"], src="moe_dispatcher:module-semantics",
                   comm=CommSpec("all_to_all", tok, "ep", ep))
    comb = TimedOp(op_id=f"{seg.seg_id}.ep_comb", op_type="CommOp", phase=ph,
                   in_shapes=(), out_shape=(), dtype=ops[last].dtype,
                   stream=COMM_STREAM["ep"], src="moe_dispatcher:module-semantics",
                   comm=CommSpec("all_to_all", tok, "ep", ep))
    return TimedSegment(seg.seg_id,
                        tuple(ops[:first]) + (disp,) + tuple(ops[first:last + 1])
                        + (comb,) + tuple(ops[last + 1:]))


def inject_cp(seg: TimedSegment, cp: int, method: str = "colossal") -> TimedSegment:
    if cp <= 1:
        return seg
    out = []
    for o in seg.ops:
        if o.op_type == "FlashAttention":
            kv = _act_bytes(o.in_shapes[1], o.dtype) + _act_bytes(o.in_shapes[2], o.dtype)
            ctype = "p2p" if method == "colossal" else "all_to_all"
            out.append(TimedOp(op_id=o.op_id + ".cp", op_type="CommOp", phase=o.phase,
                               in_shapes=(), out_shape=(), dtype=o.dtype,
                               stream=COMM_STREAM["cp"], src=f"cp[{method}]:module-semantics",
                               comm=CommSpec(ctype, kv, "cp", cp)))
        out.append(o)
    return TimedSegment(seg.seg_id, tuple(out))
```

- [ ] **Step 4: 跑测试 + 全量** → **Step 5: Commit**

```bash
git add cost_eval/timesim/frame_comm.py tests/test_timesim_producer.py
git commit -m "feat(timesim): 框架层通信注入 FSDP/EP/CP(T0-9,spec §3.3c 第二类,位置即overlap语义)"
```

## Task 11: bwd_rules——bwd/recompute 展开

**Files:**
- Create: `cost_eval/timesim/bwd_rules.py`
- Test: `tests/test_timesim_bwd_rules.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_timesim_bwd_rules.py
"""bwd 展开规则库（spec §3.3d/e）：bprop_rules 的姊妹件，逆拓扑序 + 通信对偶 + recompute 前缀。"""
from cost_eval.timesim.ir import TimedOp, TimedSegment, CommSpec, op_flops
from cost_eval.timesim.bwd_rules import expand_bwd


def _mm():
    return TimedOp(op_id="c#0", op_type="MatMul", phase="fwd",
                   in_shapes=((4096, 1, 1792), (1792, 3072)), out_shape=(4096, 1, 3072),
                   dtype="bf16", stream="device", src="mlp.py:1")


def _rs():
    return TimedOp(op_id="c#1.rs", op_type="CommOp", phase="fwd", in_shapes=((4096, 1, 1792),),
                   out_shape=(2048, 1, 1792), dtype="bf16", stream="comm_tp", src="layers.py:619",
                   comm=CommSpec("reduce_scatter", 4096 * 1792 * 2, "tp", 2))


def _view():
    return TimedOp(op_id="c#2", op_type="View", phase="fwd", in_shapes=((4096, 1, 1792),),
                   out_shape=(4096, 1792), dtype="bf16", stream="host_only", src="mlp.py:2")


def test_matmul_expands_to_dx_dw_with_2x_flops():
    seg = expand_bwd(TimedSegment("l.fwd", (_mm(),)))
    bw = [o for o in seg.ops if o.phase == "bwd"]
    assert [o.op_type for o in bw] == ["MatMul", "MatMul"]        # dX + dW
    assert sum(op_flops(o) for o in bw) == 2 * op_flops(_mm())
    assert bw[0].out_shape == _mm().in_shapes[0]                  # dX shape=输入
    assert bw[1].out_shape == _mm().in_shapes[1]                  # dW shape=权重


def test_comm_dual_and_reverse_order():
    seg = expand_bwd(TimedSegment("l.fwd", (_mm(), _rs())))
    bw = [o for o in seg.ops if o.phase == "bwd"]
    assert bw[0].op_type == "CommOp" and bw[0].comm.ctype == "all_gather"   # RS↔AG 对偶，先反向
    assert bw[0].stream == "comm_tp"


def test_view_stays_host_only():
    seg = expand_bwd(TimedSegment("l.fwd", (_view(),)))
    assert all(o.stream == "host_only" for o in seg.ops if o.phase == "bwd")


def test_recompute_prefix_full_and_comm_drop():
    fwd = (_mm(), _rs(), _view())
    seg = expand_bwd(TimedSegment("l.fwd", fwd), recompute="full", recomp_comm=False)
    rc = [o for o in seg.ops if o.phase == "recomp"]
    assert [o.op_type for o in rc] == ["MatMul", "View"]          # 重放 fwd 序，剔 CommOp
    assert seg.ops[:len(rc)] == tuple(rc)                          # 前缀在 bwd 之前
    seg2 = expand_bwd(TimedSegment("l.fwd", fwd), recompute="full", recomp_comm=True)
    assert [o.op_type for o in seg2.ops if o.phase == "recomp"] == ["MatMul", "CommOp", "View"]
```

- [ ] **Step 2: 跑确认失败** → **Step 3: 实现**

```python
# cost_eval/timesim/bwd_rules.py
"""per-op-type bwd 展开（spec §3.3d，~15 条教科书事实——opdag.bprop_rules(save-set) 的姊妹件）
+ recompute 前缀（§3.3e）。

规则表（bwd op 数目忠实是 host 双流的命门,spec D2）：
  MatMul/GroupedMatMul → dX + dW 两个同族 GEMM（dX=dy·Wᵀ,dW=xᵀ·dy,FLOPs 各≈fwd）
  FlashAttention       → 1 个 FlashAttentionGrad kernel
  CommOp               → 对偶通信：AG↔RS、AR↔AR、A2A↔A2A、p2p↔p2p
  View/host_only       → host_only（视图反向仍是元信息）
  其余（Norm/Activation/Elementwise/Cast/Gather）→ 单个 <op>Grad（带宽类）
recompute：fwd 序以 phase=recomp 前插 bwd 段;recomp_comm=False → 剔 CommOp
（`recompute_comm`/exclude 语义,继承内存侧口径、独立实现——契约 §2.2-3）。"""
from __future__ import annotations

from dataclasses import replace

from .ir import TimedOp, TimedSegment, CommSpec, STREAM_HOST_ONLY

_DUAL = {"all_reduce": "all_reduce", "reduce_scatter": "all_gather",
         "all_gather": "reduce_scatter", "all_to_all": "all_to_all", "p2p": "p2p"}


def _bwd_of(o: TimedOp) -> list[TimedOp]:
    b = dict(phase="bwd", deps=(), src=o.src, dtype=o.dtype, module=o.module)
    if o.op_type == "CommOp":
        dual = CommSpec(_DUAL[o.comm.ctype], o.comm.volume_bytes,
                        o.comm.group_axis, o.comm.group_size)
        return [TimedOp(op_id=o.op_id + ".b0", op_type="CommOp",
                        in_shapes=(o.out_shape,), out_shape=o.in_shapes[0] if o.in_shapes else (),
                        stream=o.stream, comm=dual, **b)]
    if o.stream == STREAM_HOST_ONLY or o.op_type == "View":
        return [TimedOp(op_id=o.op_id + ".b0", op_type="View",
                        in_shapes=(o.out_shape,), out_shape=o.in_shapes[0] if o.in_shapes else (),
                        stream=STREAM_HOST_ONLY, **b)]
    if o.op_type in ("MatMul", "GroupedMatMul"):
        x, w = o.in_shapes[0], o.in_shapes[1]
        dx = TimedOp(op_id=o.op_id + ".b0", op_type=o.op_type,
                     in_shapes=(o.out_shape, w), out_shape=x, stream=o.stream, **b)
        dw = TimedOp(op_id=o.op_id + ".b1", op_type=o.op_type,
                     in_shapes=(x, o.out_shape), out_shape=w, stream=o.stream, **b)
        return [dx, dw]
    if o.op_type == "FlashAttention":
        return [TimedOp(op_id=o.op_id + ".b0", op_type="FlashAttentionGrad",
                        in_shapes=o.in_shapes, out_shape=o.out_shape, stream=o.stream, **b)]
    return [TimedOp(op_id=o.op_id + ".b0", op_type=o.op_type + "Grad",
                    in_shapes=(o.out_shape,) + o.in_shapes, out_shape=o.out_shape,
                    stream=o.stream, **b)]


def expand_bwd(seg: TimedSegment, *, recompute: str | None = None,
               recomp_comm: bool = False) -> TimedSegment:
    """fwd 段 → bwd 段（含可选 recompute 前缀）。seg_id 后缀 .bwd。"""
    prefix: list[TimedOp] = []
    if recompute == "full":
        for o in seg.ops:
            if o.op_type == "CommOp" and not recomp_comm:
                continue
            prefix.append(replace(o, op_id=o.op_id + ".r", phase="recomp"))
    bwd: list[TimedOp] = []
    for o in reversed(seg.ops):
        bwd.extend(_bwd_of(o))
    return TimedSegment(seg.seg_id.replace(".fwd", "") + ".bwd", tuple(prefix + bwd))
```

- [ ] **Step 4: 跑测试 + 全量** → **Step 5: Commit**

```bash
git add cost_eval/timesim/bwd_rules.py tests/test_timesim_bwd_rules.py
git commit -m "feat(timesim): bwd 展开规则库+recompute 前缀(T0-10,spec §3.3d/e,通信对偶自动涌现)"
```

## Task 12: IR 层不变量（6ND + bwd 守恒）+ 收尾

**Files:**
- Test: `tests/test_timesim_invariants.py`
- Modify: `README.md`（状态区一行）

- [ ] **Step 1: 写不变量测试**

```python
# tests/test_timesim_invariants.py
"""IR 层不变量（spec §3.4 / §7.1-L0）：GEMM FLOPs 精确式 + 全模型 6ND 量级 + bwd 守恒。"""
import os
import pytest

from cost_eval.model_spec import DimTable
from cost_eval.timesim.ir import op_flops
from cost_eval.timesim.shard_rules import Degrees
from cost_eval.timesim.producer import build_segment
from cost_eval.timesim.bwd_rules import expand_bwd

MF_ROOT = os.environ.get("MINDFORMERS_ROOT",
                         r"E:\97-codes\torch_parallel\mindformers\mindformers")
DIMS = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                S=4096, B=1, vocab=129280, n_layers=4)


def _mlp_seg():
    if not os.path.isdir(MF_ROOT):
        pytest.skip("mindformers 源根不存在")
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import ResolvedSpec
    dag = extract_cell(
        MF_ROOT, "parallel_core/training_graph/transformer/mlp.py", "MLPInterleaved",
        ResolvedSpec(cell="MLPInterleaved",
                     submodules={"linear_fc1": "ColumnParallelLinear",
                                 "linear_fc2": "RowParallelLinear"}),
        {"gated_linear_unit": True, "activation_type": "silu",
         "add_bias_linear": False, "compute_dtype": "bf16"})
    return build_segment("mlp.fwd", dag, DIMS, Degrees())


def test_mlp_fwd_gemm_flops_exact():
    """退化(全度=1)精确式：fc1=2·S·B·H·2F + fc2=2·S·B·F·H（spec §7.1-L0③ 的 IR 半边）。"""
    got = sum(op_flops(o) for o in _mlp_seg().ops)
    S, B, H, F = 4096, 1, 1792, 3072
    assert got == 2 * S * B * H * (2 * F) + 2 * S * B * F * H


def test_bwd_gemm_flops_double_fwd():
    """bwd GEMM FLOPs = 2×fwd（dX+dW 各一份）——6ND 里 4ND 的来源。"""
    fwd = _mlp_seg()
    bwd = expand_bwd(fwd)
    f = sum(op_flops(o) for o in fwd.ops)
    assert sum(op_flops(o) for o in bwd.ops if o.phase == "bwd") == 2 * f


def test_every_device_fwd_op_has_bwd():
    fwd = _mlp_seg()
    bwd = expand_bwd(fwd)
    bwd_roots = {o.op_id.rsplit(".b", 1)[0] for o in bwd.ops if o.phase == "bwd"}
    for o in fwd.ops:
        assert o.op_id in bwd_roots, f"{o.op_id}({o.op_type}) 无 bwd 展开"


def test_tp_shard_conserves_global_flops():
    """性质：tp=2 时 per-rank GEMM FLOPs = 全局/2（切分守恒——spec §7.1-L0④ tp↑→单卡 GEMM↓）。"""
    if not os.path.isdir(MF_ROOT):
        pytest.skip("mindformers 源根不存在")
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import ResolvedSpec
    dag = extract_cell(
        MF_ROOT, "parallel_core/training_graph/transformer/mlp.py", "MLPInterleaved",
        ResolvedSpec(cell="MLPInterleaved",
                     submodules={"linear_fc1": "ColumnParallelLinear",
                                 "linear_fc2": "RowParallelLinear"}),
        {"gated_linear_unit": True, "activation_type": "silu",
         "add_bias_linear": False, "compute_dtype": "bf16"})
    full = sum(op_flops(o) for o in build_segment("s", dag, DIMS, Degrees()).ops)
    tp2 = sum(op_flops(o) for o in
              build_segment("s", dag, DIMS, Degrees(tp=2, sequence_parallel=True)).ops)
    assert tp2 * 2 == full
```

- [ ] **Step 2: 跑测试**（应直接 PASS——前序任务已就绪；若 FAIL 按 systematic-debugging 处置，
  不许调公差凑数：精确式测试的意义就是零公差）

- [ ] **Step 3: README 状态更新**

`README.md` 状态区追加一行：

```markdown
- 🔶 P1（时间模型）T0 地基完成：schedule.py 中立化、timesim 骨架+解耦 lint、opdag 通信提取
  （comm_probe）+ GPTModel 级段（loss/embedding/head）、TimedOpSeq producer（fwd/bwd/recompute）
  + IR 不变量。设计见 specs/2026-07-16-step-time-cost-model-design.md，下一步 T1（op_cost +
  segment_sim + pipeline_sim）。
```

- [ ] **Step 4: 全量回归 + Commit**

```bash
python -m pytest tests/ -q
git add tests/test_timesim_invariants.py README.md
git commit -m "test(timesim): IR 不变量——GEMM 精确式/bwd 守恒/tp 切分守恒(T0-11)+README 状态"
```

---

## 完成判据（对照 spec §9-T0）

| spec T0 项 | 任务 | 验收 |
|---|---|---|
| schedule.py 搬家（12 锚点回归门） | Task 2 | 全量 pytest 无回退 + re-export 同一性 |
| timesim 骨架 + import-lint | Task 3 | 双向禁令测试进回归门 |
| opdag 通信提取（新增能力） | Task 4 | Row/Embedding 双惯用法实证提取 |
| GPTModel 级覆盖（embedding/head/loss） | Task 5, 6 | 三段可提取/合成+段序核对，crosscheck.py:38 缺口关闭三项（GQA/dense/mtp 留 v1.5） |
| TimedOpSeq producer 五步管线 | Task 8, 9, 10, 11 | b 并行代入 / c 双类通信注入 / d bwd 展开 / e recompute 前缀 |
| IR 层不变量 | Task 12 | GEMM 精确式 + bwd 守恒 + tp 守恒 |

**明示不在 T0**（spec 对应后续阶段）：op_cost/OpTimeLibrary（T1/T2）、segment_sim/pipeline_sim（T1）、
经验库采集 runner（T2）、MTP/GQA-dense 段、DP grad-sync per-bucket 注入细化、
`interleaved_virtual_order` 在时间侧的消费（T1 pipeline_sim）。
