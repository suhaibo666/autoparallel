# 源码级 op-DAG 静态提取建模 — 实现计划

> **For agentic workers:** 用 subagent-driven-development 或 executing-plans 逐任务实现。步骤用 `- [ ]`。
> **设计依据**：`specs/2026-07-08-source-grounded-opgraph-design.md`（三阶段 + oracle，硬约束见其 §1）。

**Goal（一句话）**：建一个**离线静态分析器**，读当前 mindformers 源码 + 运行 yaml，解析出 op-DAG，套稳定的
per-op-类型 bprop 规则导出 save-set，喂现有事件模拟骨架——代码改即重跑，DAG 不冻结在 Python。

**Architecture**：`cost_eval/opdag/` 新模块。数据流 `mf 源码+yaml →[Stage1 AST]→ op-DAG →[Stage2 规则]→
save-set →[Stage3 参数化]→ 现有 structure_mem 事件模拟`。profiler CSV 仅校验。

**Tech Stack**：Python `ast`（标准库，纯静态，**不 import mindspore/mindformers、不执行**）、现有
`DimTable`/`structure_mem`/`OpSpec`、pytest。

**硬约束（不可破）**：① 绝不运行 mindformers；② DAG 是消费的数据工件、不写死；③ profiler 仅 oracle；
④ 事件骨架不动（DSv3 4L 全重算 12409.5 逐字节、cp-full 0.998、select both 0.991 回归门）；⑤ 解析不了 fail-loud。

**粒度**：本计划只打通 **DSv3（MLA+MoE）单条端到端**（设计 §7）；DSv4/dense/GQA 扩展列为 Phase 6 占位。

---

## 文件结构

- Create `cost_eval/opdag/__init__.py` — 导出 `extract_opdag`, `derive_saves`, `OpDAG`。
- Create `cost_eval/opdag/schema.py` — `OpNode`/`OpDAG` 数据类 + JSON 序列化（Task 1）。
- Create `cost_eval/opdag/bprop_rules.py` — per-op-类型 pin 规则 + `derive_saves(dag)`（Task 2）。
- Create `cost_eval/opdag/init_binder.py` — Pass B：`__init__` AST → `self.X`→op 类型绑定（Task 3）。
- Create `cost_eval/opdag/construct_walker.py` — Pass C：construct() AST → op 序列（Task 4, 7, 8）。
- Create `cost_eval/opdag/module_resolver.py` — Pass A：config → 模块树（R1，Task 6）。
- Create `cost_eval/opdag/extractor.py` — 三遍编排 + `extract_opdag(mf_root, cfg)`（Task 5）。
- Create `cost_eval/opdag/consumer.py` — 桥：op-DAG save-set → `OpSpec.saves`（Task 9）。
- Create `tools/extract_opdag.py` — CLI `--mf --config --out`（Task 10）。
- Create `tests/test_opdag_*.py` — 各 Task 的测试。
- Modify `cost_eval/layers/registry.py` / 装配器 — DSv3 走 DAG 消费（Task 9），手写 builder 保留作退化参照。

---

## Task 1：op-DAG schema（数据类 + JSON）

**Files**：Create `cost_eval/opdag/schema.py`；Test `tests/test_opdag_schema.py`

- [ ] **Step 1：写失败测试**

```python
# tests/test_opdag_schema.py
from cost_eval.opdag.schema import OpNode, OpDAG

def test_opnode_roundtrip_json():
    n = OpNode(id=13, op="MatMul", src="mlp.py:141", module="ColumnParallelLinear",
               ins=["h:S·B·H:bf16", "Wfc1:H·F:bf16"], out="i:S·B·F:bf16", attrs={"bias": False})
    dag = OpDAG(cell="MLP", nodes=[n], edges=[])
    js = dag.to_json()
    back = OpDAG.from_json(js)
    assert back.nodes[0].op == "MatMul"
    assert back.nodes[0].ins == ["h:S·B·H:bf16", "Wfc1:H·F:bf16"]
    assert back.nodes[0].src == "mlp.py:141"          # 源忠实：locator 必须往返保真
```

- [ ] **Step 2：运行验证失败** — `pytest tests/test_opdag_schema.py -v` → FAIL（模块不存在）。

- [ ] **Step 3：最小实现**

```python
# cost_eval/opdag/schema.py
"""op-DAG 数据 schema（设计 §3.4）。每节点携 src=file:line（源忠实）、符号 shape、dtype。
saves 不在此产出——由 bprop_rules 导出。纯数据类,无行为,便于 JSON 往返 + 版本化。"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json

@dataclass
class OpNode:
    id: int
    op: str                       # 规范 op 类型: MatMul/Cast/Norm/Activation/Elementwise/View/FlashAttention/...
    src: str                      # "file.py:line" —— 回指 mindformers 源
    module: str = ""              # 具名模块(ColumnParallelLinear 等),无则空
    ins: list[str] = field(default_factory=list)   # "name:符号shape:dtype"
    out: str = ""                 # 同上
    attrs: dict = field(default_factory=dict)      # bias/causal/to_dtype/linear 等

@dataclass
class OpDAG:
    cell: str
    nodes: list[OpNode] = field(default_factory=list)
    edges: list[list[int]] = field(default_factory=list)  # [src_id, dst_id]
    baseline: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "OpDAG":
        d = json.loads(s)
        nodes = [OpNode(**n) for n in d.pop("nodes", [])]
        return cls(nodes=nodes, **d)
```

- [ ] **Step 4：运行验证通过** — `pytest tests/test_opdag_schema.py -v` → PASS。
- [ ] **Step 5：提交** — `git add cost_eval/opdag/ tests/test_opdag_schema.py && git commit -m "feat(opdag): op-DAG schema(数据类+JSON 往返,携 src locator)"`

---

## Task 2：Stage2 — per-op-类型 bprop-pin 规则 + save-set 导出

**先做 Stage2**：它是纯函数（DAG→save-set），可用手搭 DAG 夹具独立 TDD，不依赖 Stage1。规则表见设计 §4。

**Files**：Create `cost_eval/opdag/bprop_rules.py`；Test `tests/test_opdag_bprop.py`

- [ ] **Step 1：写失败测试**（覆盖三个关键机理：matmul 存两操作数、cast 不自存但输出被下游 pin、norm 存 fp32）

```python
# tests/test_opdag_bprop.py
from cost_eval.opdag.schema import OpNode, OpDAG
from cost_eval.opdag.bprop_rules import derive_saves

def _dag(*nodes):
    return OpDAG(cell="T", nodes=list(nodes),
                 edges=[[nodes[i].id, nodes[i+1].id] for i in range(len(nodes)-1)])

def test_matmul_pins_both_operands():
    d = _dag(OpNode(id=1, op="MatMul", src="x:1", ins=["a:S·B·H:bf16", "W:H·F:bf16"], out="y:S·B·F:bf16"))
    saves = derive_saves(d)
    names = {s.name for s in saves}
    assert "a" in names and "W" in names          # d a/d b 都要操作数

def test_cast_not_self_saved_but_output_pinned_by_norm_consumer_at_fp32():
    # Cast(bf16->fp32) 自身不 save；其输出 h32 被下游 Norm 消费 → h32 按 fp32 计入 save
    cast = OpNode(id=1, op="Cast", src="x:1", ins=["h:S·B·H:bf16"], out="h32:S·B·H:fp32", attrs={"to_dtype": "fp32"})
    norm = OpNode(id=2, op="Norm", src="x:2", ins=["h32:S·B·H:fp32"], out="n:S·B·H:bf16")
    d = OpDAG(cell="T", nodes=[cast, norm], edges=[[1, 2]])
    saves = derive_saves(d)
    h32 = [s for s in saves if s.name == "h32"]
    assert h32 and h32[0].dtype == "fp32"          # fp32 残差修复的核心断言
    assert not any(s.op_id == 1 for s in saves)    # cast 节点自身不产 save

def test_add_residual_pins_nothing():
    d = _dag(OpNode(id=1, op="Elementwise", src="x:1", module="AddExt",
                    ins=["a:S·B·H:bf16", "b:S·B·H:bf16"], out="y:S·B·H:bf16", attrs={"linear": True}))
    assert derive_saves(d) == []                   # 线性 add 反向不存激活
```

- [ ] **Step 2：运行验证失败** — FAIL（模块不存在）。

- [ ] **Step 3：最小实现**（规则表 = 设计 §4 的 15 条；`Save(name, dtype, op_id, sym_shape)`）

```python
# cost_eval/opdag/bprop_rules.py
"""Stage2:per-op-类型 bprop-pin 规则(设计 §4)。唯一手维护件,但按 op 类型(非模型)、
教科书自动微分事实、有 profiler 兜底 → 稳定。derive_saves(dag) 遍历节点,按类型判 pin 谁。

关键:Cast 自身不 save,但其输出被下游非线性/matmul 消费者 pin(fp32 buffer 归消费者、按 fp32 计)。
实现:先对每节点按其"消费的输入"判 pin(消费者存输入),故 cast 的 fp32 输出自然被下游 norm/matmul 收进 save。"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Save:
    name: str
    dtype: str
    op_id: int          # 由哪个消费者 pin
    sym_shape: str

def _parse(ref: str):
    # "name:S·B·H:dtype" → (name, shape, dtype)
    name, shape, dtype = ref.split(":")
    return name, shape, dtype

# 每个 op 类型:它的反向 pin 哪些"输入位序"(操作数),以及是否 pin 输出。
# 'inputs':[idx...] 存这些输入; 'output':True 存输出; 空 = 不 pin(线性/视图)。
PIN = {
    "MatMul":        {"inputs": "all"},              # d a=dy·bᵀ, d b=aᵀ·dy
    "BMM":           {"inputs": "all"},
    "GroupedMatMul": {"inputs": "all"},
    "Norm":          {"inputs": [0]},                # 归一化统计需 input(fp32)
    "Softmax":       {"output": True},
    "Activation":    {"inputs": [0]},                # d x=dy·f'(x)
    "FlashAttention":{"inputs": [0, 1, 2]},          # q,k,v(+lse 由 kernel,略)
    "Gather":        {"inputs": [0]},                # 索引(小)
    "Dropout":       {"inputs": []},                 # mask 由 attrs 另计(1B),此处不按输入
    "Cast":          {"inputs": []},                 # 自身不存;输出由下游 pin
    "Elementwise":   {},                             # 见下:linear→无;非线性 mul→两操作数
    "View":          {"inputs": []},                 # reshape/transpose 元信息
}

def derive_saves(dag) -> list[Save]:
    saves: dict[tuple, Save] = {}          # dedup by (name)
    for n in dag.nodes:
        spec = PIN.get(n.op)
        if spec is None:
            raise ValueError(f"未知 op 类型 '{n.op}' @ {n.src}：bprop 规则未覆盖(fail-loud)")
        idxs = []
        if n.op == "Elementwise":
            if not n.attrs.get("linear", False):     # 非线性(mul/gate)存两操作数;linear(add)不存
                idxs = list(range(len(n.ins)))
        elif spec.get("inputs") == "all":
            idxs = list(range(len(n.ins)))
        elif isinstance(spec.get("inputs"), list):
            idxs = spec["inputs"]
        for i in idxs:
            name, shape, dtype = _parse(n.ins[i])
            saves.setdefault(name, Save(name, dtype, n.id, shape))
        if spec.get("output"):
            name, shape, dtype = _parse(n.out)
            saves.setdefault(name, Save(name, dtype, n.id, shape))
    return list(saves.values())
```

- [ ] **Step 4：运行验证通过** — `pytest tests/test_opdag_bprop.py -v` → PASS（三测全绿）。
- [ ] **Step 5：提交** — `git commit -m "feat(opdag): Stage2 bprop-pin 规则(15 条 per-op-类型)+ save-set 导出(cast 输出由消费者 pin)"`

---

## Task 3：Pass B — `__init__` 绑定（`self.X` → op 类型）

**Files**：Create `cost_eval/opdag/init_binder.py`；Test `tests/test_opdag_init_binder.py`
**夹具**：`mlp.py` 的 `MLP.__init__`——`self.add=AddExt()`、`self.mul=Mul()`、`self.split=SplitWithSize()`，
且这些类**从 `mindspore.ops.auto_generate` import**（按导入名 → 规范 op 类型）。

- [ ] **Step 1：写失败测试**（用内联最小源码串，不读真文件，测纯 AST 逻辑）

```python
# tests/test_opdag_init_binder.py
from cost_eval.opdag.init_binder import bind_init

SRC = '''
from mindspore.ops.auto_generate import Mul, AddExt, SplitWithSize, Reshape, Transpose
class MLP:
    def __init__(self, config, submodules):
        self.mul = Mul()
        self.add = AddExt()
        self.split = SplitWithSize()
        self.reshape = Reshape().recompute(True)
        self.compute_dtype = config.compute_dtype
'''

def test_bind_self_names_to_canonical_optype():
    b = bind_init(SRC, "MLP")
    assert b["mul"].op == "Elementwise" and b["mul"].attrs.get("linear") is False
    assert b["add"].op == "Elementwise" and b["add"].attrs.get("linear") is True
    assert b["split"].op == "View"
    assert b["reshape"].op == "View"        # .recompute(True) 链式调用不影响类型解析
    assert "compute_dtype" not in b          # 非 op 绑定不入表
```

- [ ] **Step 2：运行验证失败** — FAIL。

- [ ] **Step 3：实现**（AST：收集 import 别名 → 类名；扫 `__init__` 里 `self.X = <Call>`；类名→规范 op 类型映射表；
  `AddExt`/`Add`→Elementwise linear=True，`Mul`/`Sub`→Elementwise linear=False，`Reshape`/`Transpose`/`SplitWithSize`→View，
  `Cast`→Cast……；链式 `.recompute()`/`.shard()` 剥到基 Call。未知类名 → fail-loud）。

```python
# cost_eval/opdag/init_binder.py
"""Pass B(设计 §3.2):解析 Cell.__init__ 的 AST,建 self.<name> → Binding(op 类型,attrs)。
按 import 的类名解析规范 op 类型;链式(.recompute/.shard)剥到基 Call。未知类名 fail-loud。"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field

# 规范 op 类型映射(mindspore 类名 → 我们的 op 类型)。linear=反向是否线性。
_CLS2OP = {
    "AddExt": ("Elementwise", {"linear": True}),  "Add": ("Elementwise", {"linear": True}),
    "Sub":   ("Elementwise", {"linear": True}),
    "Mul":   ("Elementwise", {"linear": False}),
    "Cast":  ("Cast", {}),
    "Reshape": ("View", {}), "Transpose": ("View", {}), "SplitWithSize": ("View", {}),
    "Shape": ("View", {}),
    # 具名 linear/attention/norm/activation 由 construct 调用点解析(build_module/get_activation),此处不绑。
}

@dataclass
class Binding:
    op: str
    attrs: dict = field(default_factory=dict)

def _base_call_name(node):
    """剥链式 .recompute()/.shard() 等,取最内层 Call 的类名。"""
    while isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            # X().method() → 递归到 X()
            if isinstance(f.value, ast.Call):
                node = f.value
                continue
            return None
        return None
    return None

def bind_init(src: str, cls_name: str) -> dict[str, Binding]:
    tree = ast.parse(src)
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
    if cls is None:
        raise ValueError(f"源码里找不到 class {cls_name}(fail-loud)")
    init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    out: dict[str, Binding] = {}
    for stmt in ast.walk(init):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self"):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        clsname = _base_call_name(stmt.value)
        if clsname in _CLS2OP:
            op, attrs = _CLS2OP[clsname]
            out[tgt.attr] = Binding(op=op, attrs=dict(attrs))
    return out
```

- [ ] **Step 4：运行验证通过** — PASS。
- [ ] **Step 5：提交** — `git commit -m "feat(opdag): Pass B __init__ 绑定(self.X→op 类型,链式剥离,未知 fail-loud)"`

---

## Task 4：Pass C — construct() 走查（MLP 打通，第一条端到端）

**Files**：Create `cost_eval/opdag/construct_walker.py`；Test `tests/test_opdag_walk_mlp.py`
**夹具**：`mlp.py:134` `MLP.construct`——`linear_fc1(x)` → `add(i, bias)` → `activation_func(i)`(swiglu) → `linear_fc2(i)`。

- [ ] **Step 1：写失败测试**（内联 MLP construct 最小串 + 已绑定的 init 表 + build_module 解析结果，验证 op 序列）

```python
# tests/test_opdag_walk_mlp.py
from cost_eval.opdag.init_binder import Binding
from cost_eval.opdag.construct_walker import walk_construct

MLP_CONSTRUCT = '''
class MLP:
    def construct(self, hidden_states):
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias
'''

def test_walk_mlp_emits_matmul_activation_matmul():
    # linear_fc1/fc2 由 build_module 解析为 MatMul(module=ColumnParallelLinear);activation_func=Activation
    binds = {
        "linear_fc1": Binding(op="MatMul", attrs={"module": "ColumnParallelLinear", "compute_dtype": "bf16"}),
        "activation_func": Binding(op="Activation", attrs={"kind": "swiglu"}),
        "linear_fc2": Binding(op="MatMul", attrs={"module": "RowParallelLinear", "compute_dtype": "bf16"}),
    }
    dag = walk_construct(MLP_CONSTRUCT, "MLP", binds, src_file="mlp.py")
    ops = [n.op for n in dag.nodes]
    assert ops == ["MatMul", "Activation", "MatMul"]
    assert dag.nodes[0].src.startswith("mlp.py:")     # 每节点带真 locator
```

- [ ] **Step 2：运行验证失败** — FAIL。
- [ ] **Step 3：实现** — AST 走 `construct` 体，遇 `self.<name>(...)` 调用：查 `binds` → 发 `OpNode`（op 类型、
  `src=f"{src_file}:{node.lineno}"`、ins/out 变量名）；SSA 变量名跟踪连边；未知 `self.X` 调用 → fail-loud。
  dtype 传播：遇 `self.cast(x, dt)` / `.astype(dt)` 更新目标 SSA 值 dtype。**（完整代码在实现时按此结构写；
  控制流按 config 绑定后取定分支——见 Task 6 传入的 flags。）**
- [ ] **Step 4：验证通过** — PASS。
- [ ] **Step 5：提交** — `git commit -m "feat(opdag): Pass C construct 走查(MLP 打通,SSA 连边,未知调用 fail-loud)"`

---

## Task 5：extractor 编排（读真 `mlp.py` 产 DAG）

**Files**：Create `cost_eval/opdag/extractor.py`；Test `tests/test_opdag_extract_mlp_real.py`
- [ ] **Step 1：失败测试** — `extract_cell(mf_root, "MLP")` 读**真** `mlp.py`，断言产出 `[MatMul, Activation, MatMul]`
  且每节点 `src` 指向 `mlp.py` 真实行号（对 grep 到的行号）。
- [ ] **Step 2：FAIL**。
- [ ] **Step 3：实现** — 组合 Pass B(`bind_init`) + build_module 解析(Task 6 的 `resolve_call`) + Pass C(`walk_construct`)，
  读真文件文本喂 AST。
- [ ] **Step 4：PASS**。
- [ ] **Step 5：提交** — `git commit -m "feat(opdag): extractor 编排(读真 mlp.py 产 DAG,行号回指)"`

---

## Task 6：Pass A — 模块树/`build_module` 解析（R1）

**Files**：Create `cost_eval/opdag/module_resolver.py`；Test `tests/test_opdag_module_resolver.py`
**入口**：`base_models/gpt/gpt_layer_specs.py`（`get_gpt_layer_local_spec` 静态装配 submodules）+ DSv3 yaml。
- [ ] **Step 1：失败测试** — 给定 DSv3 config flags（`multi_latent_attention=True, num_moe_experts=N`），
  `resolve_layer_spec(mf_root, cfg)` 返回 `{"self_attention": "MultiLatentAttention", "mlp": "MoELayer", ...}`。
- [ ] **Step 2：FAIL**。
- [ ] **Step 3：实现** — 静态解析 spec 构造函数 AST + config 分支求值（只求 config 决定的 if）；
  `build_module(submodules.X)` → 顺 spec 找具体类。**解不出的 submodule → fail-loud，报缺的 config 键**（设计 R1）。
- [ ] **Step 4：PASS**。
- [ ] **Step 5：提交** — `git commit -m "feat(opdag): Pass A 模块树解析(config 驱动 build_module,解不出 fail-loud)"`

---

## Task 7：MLA 注意力 Cell 提取

**Files**：Modify `construct_walker.py`（补 rope/flash/latent 惯用法）；Test `tests/test_opdag_mla.py`
**夹具**：`multi_latent_attention.py` construct（`get_query_key_value_tensors`、`apply_rotary_pos_emb`、`core_attention`）。
- [ ] **Step 1：先读** `multi_latent_attention.py` 的 `__init__`+`construct`，列出全部 `self.X` 调用与 cast 落点。
- [ ] **Step 2：失败测试** — 断言 DAG 含 q_a/kv_a 降维 MatMul、q/k_layernorm(Norm,fp32)、FlashAttention、o_proj，
  且 rope/latent 的 cast 落点被记为 fp32（若源码如此）。
- [ ] **Step 3：FAIL → 实现补惯用法 → PASS**。
- [ ] **Step 4：oracle 校验** — 对 `analysis/realmachine/select_attn/`（重算 attn、留 FFN）的活跃集，MLA 前向 saves 应与 profiler 一致。
- [ ] **Step 5：提交** — `git commit -m "feat(opdag): MLA Cell 提取(latent/rope/flash/norm-fp32)"`

---

## Task 8：MoE FFN Cell 提取（grouped-GEMM 中间量，18% 残差的靶心）

**Files**：Modify `construct_walker.py`（补 router/permute/GroupedMatMul/combine）；Test `tests/test_opdag_moe.py`
**夹具**：`moe/moe_layer.py`+`moe/experts.py` construct。
- [ ] **Step 1：先读** moe_layer/experts 的 construct，定位 permute/capacity-pad/cast/GroupedMatMul 是否**显式 op**
  （若在融合 kernel 内 → 记 R3，退 oracle margin，明示为标定常数）。
- [ ] **Step 2：失败测试** — 断言 DAG 含 router(MatMul)、dispatch/permute(View 或显式 copy)、GroupedMatMul×2、
  combine，且 GroupedMatMul 操作数按 §4 存两操作数。
- [ ] **Step 3：FAIL → 实现 → PASS**。
- [ ] **Step 4：oracle 校验** — 对 `analysis/realmachine/select_attn/`（留整个 MoE-FFN），导出 MoE 前向 saves
  应把 select `self_attn` 从 0.823 拉到 ≥0.95（**本计划核心验收**）。
- [ ] **Step 5：提交** — `git commit -m "feat(opdag): MoE Cell 提取(grouped-GEMM 中间量,闭合 self_attn 18% 残差)"`

---

## Task 9：consumer 桥 + 装配接入（Stage3）

**Files**：Create `cost_eval/opdag/consumer.py`；Modify DSv3 装配路径；Test `tests/test_opdag_consumer.py`
- [ ] **Step 1：失败测试** — `saves_to_opspec(dag, dims)` 把符号 shape 用 `DimTable` 代入得字节,
  产出与现有 `OpSpec.saves` 同接口;装配器 DSv3 路径改吃 DAG 后,**回归门锚点不破**（DSv3 4L 全重算 12409.5）。
- [ ] **Step 2：FAIL**。
- [ ] **Step 3：实现** — 符号 shape 解析器(`S·B·H`→dims 乘积)+ dtype→字节;接入装配器(手写 builder 留作退化参照,
  用 flag 切换,默认 DAG)。
- [ ] **Step 4：PASS + 全量回归** — `pytest tests/ -q` 全绿,特别是 `test_ce_optstep.py`/`test_dsv4align_*`/cp/pp 锚点。
- [ ] **Step 5：提交** — `git commit -m "feat(opdag): Stage3 consumer 桥 + DSv3 装配接入(锚点回归全绿)"`

---

## Task 10：CLI + 再推导工作流

**Files**：Create `tools/extract_opdag.py`；Test `tests/test_extract_cli.py`
- [ ] **Step 1：失败测试** — `python tools/extract_opdag.py --mf <root> --config <dsv3.yaml> --out /tmp/dag.json`
  产出可被 `OpDAG.from_json` 读回、含 TransformerLayer 全 Cell。
- [ ] **Step 2：FAIL → 实现 CLI(纯静态,不 import mindformers)→ PASS**。
- [ ] **Step 3：更新** 设计文档 §9 工作流为实际命令 + `analysis/realmachine/` 加 opdag 验收记录。
- [ ] **Step 4：提交** — `git commit -m "feat(opdag): extract_opdag CLI + 再推导工作流(纯静态)"`

---

## Task 11：oracle 验收矩阵 + 文档

**Files**：Modify `specs/2026-07-07-memory-model-reference.md` §14/§15；Create `analysis/realmachine/opdag_validation.md`
- [ ] 跑全锚点,填对比表:**不破**（DSv3 全重算 12409.5、cp-full 0.998、select both 0.991）；
  **要修好**（self_attn 0.823→?、mlp 0.940→?、DSv4 0.968→?、cp2-none 0.911→?）。
- [ ] 更新参考文档:op 图来源从"手写 builder"改为"静态 DAG 提取",标注 §D-10 残差闭合情况。
- [ ] 未闭合的（R2 view 物化 / R3 融合 kernel 内部量）→ 明示为已知残差 + 标定 margin。
- [ ] **提交** — `git commit -m "docs(opdag): oracle 验收矩阵 + 参考文档更新(残差闭合记录)"`

---

## Phase 6（占位，本计划不做）：扩展到 DSv4 / dense / GQA

DSv3 端到端 + 锚点验收通过后，按同 Task 7/8 模式补 `dsa_attention.py`(DSv4 hybrid)、dense `MLP`、GQA `SelfAttention`
的 construct 惯用法。每个新结构一条 DAG + 一次 oracle 校验。**不在本计划范围**——先证 DSv3 一条路走通。

---

## 自检（writing-plans）

- **覆盖**：设计三阶段 → Task 1(schema)/2(Stage2)/3-8(Stage1)/9(Stage3)/6(R1)/11(oracle) 全覆盖；硬约束①(纯静态)贯穿。
- **占位符**：Task 4/7/8 的"先读源码再实现"是**真实工作步骤**（静态提取器必须先读具体 construct），非 TBD 占位；
  基础 Task 1/2/3 给了完整可跑代码。
- **类型一致**：`OpNode`/`OpDAG`(Task1)→`derive_saves`(Task2)→`bind_init`/`Binding`(Task3)→`walk_construct`(Task4)
  →`extract_cell`(Task5)→`resolve_layer_spec`(Task6)→`saves_to_opspec`(Task9) 接口链贯通。
- **验收硬指标**：Task 8 把 select `self_attn` 0.823→≥0.95 是本计划成败判据；回归门锚点(12409.5/0.998/0.991)不破。
