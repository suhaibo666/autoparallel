# 统一的源码级 Model IR — 设计文档

> 创建: 2026-07-30 · 状态: 设计（待用户过审）→ 实现计划 → 实现
> **代码基线**: `feat/unified-llm-modelspec` @ `148116a`（工作区干净）；`pytest tests -q -k opdag` → 479 passed
> **源码基线**: `mf-src-167`（`mindformers` @ `26354ff64` + `hyper_parallel` @ `41495aa2`，md5 门见
> `to_resolved.py:101-106`）
> 父设计: `2026-07-08-source-grounded-opgraph-design.md`（opdag 抽取器）
> 相关: `2026-07-16-step-time-cost-model-design.md`（timesim；本文与其 D5/§2.2 的关系见 §2）
> 依据实测: `docs/census_arbitration_2026-07-29.md`（三方仲裁）、`docs/opdag_symbolic_axes_2026-07-29.md`

## 0. 一句话目标

把今天**两条独立的 op 图构造链**（显存吃手写 `model_spec` 图、时间吃源码抽取 `OpDAG`）合成
**一个 Model IR**：`ModelIR = (Structure, Placement, Registry)`——结构从源码抽取（唯一结构源）、
布局由并行配置推导（一份实现）、**op 语义由纯 YAML 配置描述**（原子 op，用户可扩展，
未定义即阻断）。显存与时间都是它的消费者。

## 1. 已定决策记录（设计对话 2026-07-30，逐项过审）

| # | 决策 | 内容 |
|---|------|------|
| U1 | 手写图终局 | `cost_eval/layers/*.py` 的 **op 结构与 `saves=[...]` 退役**；只保留源码读不出的**标定量**，并迁入 Registry 的 `kernels:` 段 |
| U2 | 单一结构源 | 所有图结构从**源码抽取**（`cost_eval/opdag/`）。理由见 §3.1 |
| U3 | 共享面 | 方案 B：mem/time 共享 **Structure + Placement**，各自只保留"口径壳"（mem 要字节、time 要时间）。**不**共享到 local IR（会造 god object） |
| U4 | op 描述形态 | **纯 YAML**。描述里**不出现任何代码引用**；op 类型拆到**原子级**（一个类型 = 一条形状关系） |
| U5 | 未知 op | **阻断运行**。用户通过追加 op 描述配置文件定义；阻断时输出可粘贴的 stub |
| U6 | op 名对齐 | `OpNode.qname` = 内联帧路径 + 调用点 `self.<attr>` 名，取代合成名 `n{id}_{op}` |
| U7 | 迁移棘轮 | 按桶切换，不整体切（§9） |

### 1.1 本设计自行裁决的四点（可推翻，附理由）

| # | 裁决 | 理由 | 如何推翻 |
|---|------|------|---------|
| A1 | CP 切轴以 **mem 侧语义**为准：只切**首个引用 S 的轴**，其余保持全量 | `shape_eval.py:103-114` 写明"切 query/token 维、key/context 维保持全量"（CP 下 context 维聚齐）。time 侧"每个 S 轴都切"无源侧论证 | 给出 CP 实现中 context 维也被切的证据 |
| A2 | `faces` **按消费者分档**：mem 需 `{optype,bprop,shape,placement}`，time 另需 `{bwd,cost}` | 一刀切会让"某个时间系数未标定"卡死显存工作，而那个系数对显存零影响 | 配置项 `registry.strict_all_faces: true` 一键切回一刀切 |
| A3 | Registry **覆盖须显式** `override: true`，否则重复定义即报错 | 防用户文件静默改掉内置语义（与 `from_mindformers.py` 未知键 fail-loud 同源） | — |
| A4 | `qname` 用**全限定帧路径**（`csa.indexer.linear_wq_b`）而非短名 | 短名跨帧会撞：`Compressor` 有两个构造点（`csa.py:604` / `indexer.py:128`），同名不同物 | — |

## 2. 与既有裁决 D5（timesim 解耦）的关系

`2026-07-16-step-time-cost-model-design.md` D5 记录了用户裁决"**时间仿真器与内存仿真器不耦合、
独立演进**"，并由 `tests/test_timesim_decoupling.py` 强制。**本设计不推翻它**：

| D5 §2.2 契约 | 本设计 |
|---|---|
| 1. 双向 import 禁令（`timesim/**` ⊥ `mem_timeline`/`structure_mem`/`static_mem`） | **不动**。placement 与轴代入落在 `opdag/`，不引入任一禁止项 |
| 2. 共享面只有 `specs` / `schedule` / **opdag（上游 IR）** | **加厚这一条**。placement 与轴代入本就属"上游 IR + 并行代入"；`shard_rules.py:15` 今天已 import `opdag.consumer`，先例已在。白名单需**新增 `opdefs`**（配置包） |
| 3. 各自独立 walk（不复用桶/事件对象） | **不动**。共享止于 IR，事件推进各自为政 |
| 4. 标定命名空间分离 | **不动**。时间标定不回流内存，反之亦然；Registry 的 `cost:` 面只由 time 读、`bprop:` 面只由 mem 读 |
| 5. 门面各自可跑 | **不动** |

> **为什么"独立轴代入"从来不在 D5 的意图内**：D5 §2.2-3 说的是"独立**事件推进**"。而轴代入
> （符号 shape → local）是纯函数，两侧独立实现的结果是**已经分道了**——见 §7 缺陷 ①，同一张量
> 同一配置差 2×，今天无人发现。共享它是在恢复 D5 的本意，不是违反它。

## 3. 架构

```
S0   输入      mf-src-167 快照(md5 门) │ yaml→LLMConfig→DimTable │ specs.py(并行/重算/优化器/硬件)
                    │
S0.5 词典      opdefs/*.yaml ─▶ OpRegistry        【新·纯 YAML·用户可扩展·未定义即阻断】
                    │   加载自检 + 建图后完备性门
                    ▼
S1   结构      cost_eval/opdag/ ─▶ OpDAG(+qname)   【唯一结构源】识别面/词表面 ← Registry
                    │
S2   布局      opdag/placement.py(OpDAG, Degrees) ─▶ Placement   布局面 ← Registry
                    │
S3   代入      opdag/localize.py: local_axes(sym, dims, placement) ─▶ 逐轴 local 值【唯一一份】
                    ├─ mem : prod() → RTensor/ROp → ResolvedLayer → mem_timeline 桶 / liveness
                    └─ time: 逐轴 list → TimedOp   → segment_sim(L1) / pipeline_sim(L2)
                             反向/声明面 ← Registry      展开/定价面 ← Registry
```

`ModelIR = (Structure=OpDAG, Placement, Registry)`。两个消费者都是它的纯函数。

### 3.1 为什么结构源是源码抽取而不是手写 ModelSpec（U2 论证）

| | 源码抽取 | 手写 `model_spec` |
|---|---|---|
| 可证伪 | 每节点带 `src=file:line` | `saves=[...]` 全部依据是一句注释"最坏情况：每个 op 的 bprop 都持有其输入/输出"；实测 **1.582–1.713× 真机**（`census_arbitration` §0） |
| 失效可见 | 看不懂即 fail-loud + 逐条进 `diagnostics`/`Coverage` | 漏 op **无任何机制发现**——实测漏过 `sinks`/`sparse_indices`/`ctx.logits`，且把 detached 的 O(S²) fp32 当 saved |
| 跟随性 | 重跑抽取即更新 | 结构变了数字还在，静默失效 |
| op 名 | 真源属性名（`self.linear_q_down_proj`） | 截断近似（`linear_q_down`），无人检查 |

**手写侧唯一不可替代的**是"源码里读不出的量"（`workspace`/`bwd_scratch`/`bwd_workspace`，在 `.cc`
kernel 里或只能真机标定）→ 迁入 Registry 的 `kernels:` 段（§5.3）。`TensorRef.shard` **被 S2 取代**
（placement 是可推导的，不是要人声明的事实）；`model_spec.OpType` 枚举**保留**为消费侧词表。

## 4. 各层设计

### 4.1 S0.5 — OpRegistry（`cost_eval/opdefs/`）

**职责**：op 语义的唯一词典。今天散在 **8 张 Python 表**：`primitives.PRIMITIVES`(98) /
`OPDAG_OP_TO_OPTYPE`(22) / `bprop_rules.PIN`(22) / `init_binder._CLS2OP`(29) /
`module_resolver.LEAF_OPTYPE`(7) / `construct_walker.DIRECT_OP_MAP` / `FREE_CALL_MAP` /
`to_resolved._KERNEL_SAVES`+`_DECLARED_OPAQUE_OUTS`——加一个 op 要改 3–5 处 Python，且用户无法扩展。

**纯核原则**（沿用 `configs/from_mindformers.py:15-16`）：`OpRegistry.from_dict` 只吃 dict、不 import
yaml；仅 `load_opdefs` 内惰性 `import yaml`。未知键 fail-loud。

**分层加载**：`opdefs/builtin/*.yaml`（随仓库）→ `opdefs/site/*.yaml`（站点）→ `--opdefs <path>` /
`COST_EVAL_OPDEFS`。后者覆盖前者，但**必须显式 `override: true`**（A3）。

**加载自检**（硬门）：schema 校验；每条必须带 `src`（`file:line` 或标定报告路径）与 `why`；
`optype` 必须是 `model_spec.OpType` 的合法值；`shape.rel` 必须是组合子库里存在的名字；
`guards` 必须是谓词库里存在的名字。任一不满足 → `OpRegistryError`。

### 4.2 S1 — 结构层（`cost_eval/opdag/`）

**职责**：源码 → `OpDAG`。只回答"有哪些 op、怎么连、符号形状是什么"，**不回答**并行度/字节/时间。

**变更**：

| # | 变更 | 落点 | 理由 |
|---|---|---|---|
| a | `OpNode` 加 `qname` | `schema.py`；`_handle_self_call:1946` 的 `name` 透到 `_emit:3092` | U6。名字已在 walker 手里，今天在 `:2003` 被丢弃 |
| b | op 类型**拆到原子级** | Registry + `primitives`/`shape_infer`/`bprop_rules` | U4。见 §4.2.1 |
| c | 删 `to_resolved._OPTYPE`，统一走 Registry 的 `optype` 面 | `to_resolved.py:975-978, 1338-1341` | 缺陷 ②（§7） |
| d | 识别面/词表面改由 Registry 供给（表内容迁 YAML，代码只留查询） | `primitives.py` / `init_binder.py` / `module_resolver.py` | U4 |

**不动**：fail-loud 纪律、`diagnostics` 五键、`opaque_calls`、0 节点硬门、快照 md5 门、
`Coverage` 的"绝不零填"。这些是 opdag 最值钱的部分。

#### 4.2.1 原子 op 拆分（U4 的落地）

今天 `View` 一个类型塞 **14 种语义**、`IndexSelect` 塞 2 种、`Constant` 塞多种 ctor，代码在
`shape_infer._dispatch:1418-1470` 用 `if attrs["view"] == ...` **再分派**。

**关键事实**：识别面**今天已经是声明式的**——`primitives.py:70-84` 就写着
`"mint.reshape": ("View", {"view": "reshape"})`、`"mint.permute": ("View", {"view": "permute"})`……
代码的再分派只是在重读配置里已有的判别字段，**没有比配置多知道任何东西**。

拆分即：`("View", {"view":"reshape"})` → `Reshape`。得到的原子类型：

```
Reshape Permute Transpose Squeeze ExpandDims Concat Stack Split Chunk Slice Tile Roll
BroadcastTo Contiguous | Gather AdvancedIndex | Zeros Ones Full Empty Arange EyeLike
```

连带：`bprop_rules.PIN`、`OPDAG_OP_TO_OPTYPE`、`to_resolved` 类型映射、`timesim/bwd_rules` 分派
全部跟着拆——但**它们都已迁进同一份 YAML**，所以是"一处拆细"而非"四处同步"。

拆分本身是净收益：今天 `IndexSelect` 下 `Gather` 与 `AdvancedIndex` 共用 `bprop: {"inputs":[1]}`，
**碰巧**两者 index 都是 ins[1]；碰巧对不是设计。

### 4.3 S2 — 布局层（`cost_eval/opdag/placement.py`，新）

**职责**：`(OpDAG, Degrees) → Placement`。回答"每个张量的每根轴被谁切了、哪里要插通信"。
实现来自 `timesim/producer.py:8-45` 的 per-tensor 分片状态传播，**剥出来共享**。

```python
@dataclass(frozen=True)
class Placement:
    shard:      dict   # tensor_key -> {轴键: 'tp'|'ep'|'cp'|'sp'}   ← 直接对上 local_shape_elems 的既有入参
    sp_active:  dict   # tensor_key -> bool
    cp_shard:   dict   # tensor_key -> bool（kv/context 维为 False）
    comm_sites: tuple  # (node_id, ctype, group_axis, why, src)
    coverage:   Coverage

def infer_placement(dag, degrees, registry, *, strict: bool) -> Placement: ...
```

**为什么几乎不改消费者**：`consumer.local_shape_elems(..., shard=...)` 的 `shard` 槽
**本来就存在**（`consumer.py:217-228`），只是 `_Folder._sig:1078` **从不传**（默认 `None`）——
这就是 `resolve_graph` 对 `tp/cp>1` fail-loud（`to_resolved.py:1444-1450`）的全部原因。
placement pass 就是那个缺失的供给者。

**传播规则**（沿用 producer 已验证的，规则本身写进 Registry 的 `placement:` 面）：
段入口种子（含 S 轴 + `sequence_parallel` + tp>1 → `{"S": tp}`）；Column 出 carrier、入携 S 则注入
AG；Row 消 carrier、出注入 RS(sp)/AR(非 sp)；SPL 权重全量无通信；多输入汇合 carrier 取并集、
S 分歧注入 layout-redistribution AG；carrier 歧义 fail-loud。

**两档严格度**：`strict=True`（time 侧，沿用现行 fail-loud）；`strict=False`（mem 侧，
推不出的张量进 `Coverage` 后跳过，**不默认成全量**——沿用"绝不零填"）。

**收益**：mem 侧 `tp/cp>1` 从 fail-loud 变为支持；显存侧首次看得到通信缓冲；AG/RS 判据只有一份。

### 4.4 S3 — 代入层（`cost_eval/opdag/localize.py`，新）

**职责**：`(符号 shape, DimTable, Placement) → 逐轴 local 值`。纯函数、无状态、**唯一一份**。

```python
def local_axes(sym, dims, deg, *, shard=None, sp_active=False,
               cp_shard=True, cp_kv=False, is_weight=False) -> list[int] | None
```

规则按 A1 以 mem 侧语义为准；轴识别统一用 `consumer._axis_refs_S` 的**标识符匹配**（能看穿
`csa_window_size+S//4` 这类复合轴），不用 `shard_rules` 的 `"S" in syms` 字面匹配。

消费者各加一层薄壳：mem = `prod(local_axes(...))`（`consumer.local_shape_elems` 收敛成一行，
`_Folder._sig` 补传 `shard=`）；time = 直接用逐轴 list（`shard_rules.localize` 删除；
`weight_local` **保留**——权重切分轴随模块语义，是另一件事）。

### 4.5 S4 — 消费者（口径壳，不动核心）

mem：`to_resolved._Folder` → `ResolvedLayer`（`liveness/contract.py` 18 条硬规则不动）→
`mem_timeline` 18 桶 / `liveness/simulate`。
time：`timesim/producer` 剥掉 placement 后只剩"发 TimedOp"→ `op_cost` → `segment_sim` →
`pass_builder` → `pipeline_sim`。

## 5. op 描述配置（U4/U5 的落地）

### 5.1 七个面

| 面 | 消费者 | 形态 |
|---|---|---|
| 识别 名字→op 类型 | S1 | `primitives:` 段 |
| 词表 op→`OpType` | mem | `ops.<op>.optype` |
| 反向 pin 哪些张量 | mem | `ops.<op>.bprop` |
| 形状 out shape 关系 | 两侧 | `ops.<op>.shape` |
| **dtype 产出 dtype 关系** | 两侧 | `ops.<op>.dtype` |
| 布局 placement 传播 | 两侧 | `ops.<op>.placement` |
| 展开+定价 | time | `ops.<op>.bwd` / `.cost` |

#### 5.1.1 dtype 面（缺陷 ⑥ 的归口）

**今天 dtype 推断散在至少 9 处**，且推错过一次真实的 2× 字节错误：

| 位置 | 干什么 |
|---|---|
| `construct_walker.py:1840, 1990` | 显式 cast 取目标 dtype（`_resolve_cast_dtype`） |
| `:1618` `_promote_dtype` + `_emit:3163-3172` | 按输入提升序推——**两份 rank 表** |
| `:1701, 1714, 1992` | `attrs.get("compute_dtype") or "bf16"` 兜底——**三处重复** |
| `:1470, 1506, 1829` | `"bool" if op == "Compare" else None`——**三处重复** |
| `shape_infer.py:1272-1281` | **第二道**把 Compare 改 bool。注释写明动因：「walker 的 `_emit` 按 ins 推产出 dtype → 这批节点被记成 bf16 = **2×**」（`csa.py:779` / `:810` 两个 O(S²) mask，被 `Where` 的 `PIN{"inputs":[0]}` 当 cond 保留 ⇒ dtype 必须对） |
| `bprop_rules.py:127-130` → `to_resolved._undo_norm_prelift` → `structure_mem._dt` | Norm 输入**预抬 fp32 → 适配器撤销 → 消费侧再抬**，一个三段往返，只因 dtype 语义没有单一归属 |

**归口规则**（声明式，优先级自上而下）：

```yaml
Cast:        {dtype: attrs.to_dtype}
Compare:     {dtype: fixed(bool)}
Norm:        {dtype: config.layernorm_compute_dtype}    # 单点归属，取消预抬-撤销-再抬
TopK:        {dtype: {out0: same_as(in0), out1: fixed(int32)}}
Elementwise: {dtype: promote(in*)}                      # 唯一一份提升序表（引擎侧）
Constant:    {dtype: config.compute_dtype}
MatMul:      {dtype: same_as(in0)}
```

dtype 组合子（引擎侧，与形状组合子同性质）：`same_as(inK)` · `promote(in*)` · `fixed(<dtype>)` ·
`attrs.<key>` · `config.<key>`。**取消 `"bf16"` 硬兜底**——缺 dtype 依据的节点走完备性门阻断，
不默认成 bf16（那正是 2× 错误的形态）。

### 5.2 形状面：纯声明的边界

```
YAML（op 描述）= 形状关系 + 参数来源(attrs) + 前置条件 + 退化档
代码（引擎）   = 形状组合子 + sym_shape 符号代数 + 谓词库
```

组合子是**封闭的一小组**（不到 20 个），跨 op 复用（9 个原子 op 共用 `same_as`）：

`same_as(inK)` · `broadcast(in*)` · `replace_last(in0, attrs.out_dim)` · `bmm` · `halve_last` ·
`reshape` · `permute` · `swap_axes` · `squeeze` · `expand_dims` · `concat` · `stack` · `split` ·
`chunk` · `slice` · `tile` · `literal(attrs.X)` · `range_len` · `index_compose` · `declared`

**为什么组合子在代码里不算"op 与代码关联"**：组合子是**表达式求值器的原语**，与 op 无关——
就像 YAML 里写 `2*x` 不会让人说"这关联了代码"。证据：`_reshape:1029` 的 `-1`/单缺维消元
整个在 `sym_shape.resolve_reshape:644-662`（通用机器）；`_permute:940` 是一条置换关系；
`_concat:774` 是"逐轴相等校验 + 该轴相加"。op 独有的知识只有**关系名 + 参数来源 + 守卫**，全是数据。

### 5.3 Schema

加载器选项（**不是** op 描述的一部分，放 `opdefs/registry.yaml`）：

```yaml
# opdefs/registry.yaml —— 加载器行为，与 op 语义正交
version: 1
strict_all_faces: false    # A2：true = 任一面缺即全阻断（不分 mem/time 消费者）
```

op 描述本体：

```yaml
version: 1

# ① 识别面
primitives:
  "mint.reshape":         {op: Reshape,  src: "csa.py:628"}
  "mint.permute":         {op: Permute,  src: "csa.py:472"}
  "ops.stop_gradient":    {op: Detach,   src: "csa.py:665"}
  "ColumnParallelLinear": {op: MatMul, module: ColumnParallelLinear, src: "layers.py:619"}

# ② op 语义（原子级：一个类型 = 一条形状关系）
ops:
  Norm:
    optype: norm
    bprop:  {inputs: [0], why: "归一化统计需 input(fp32)"}
    shape:  {rel: same_as, args: {input: in0}}
    placement: passthrough
    bwd:  single_grad
    cost: {class: bandwidth}

  MatMul:
    optype: matmul
    bprop:  {inputs: all, why: "dA=dy·Bᵀ, dB=Aᵀ·dy"}
    shape:  {rel: replace_last, args: {input: in0, last: attrs.out_dim}}
    placement: linear
    bwd:  two_gemm
    cost: {class: gemm}

  Reshape:
    optype: elementwise
    bprop:  {inputs: [], why: "视图，反向是逆视图"}
    shape:
      rel:  reshape
      args: {input: in0, target: attrs.reshape_dims}
      on_unresolved: numel_only     # reshape 恒不改 numel（算子定义）→ 退元素数档比整条放弃更准确
      why:  "csa.py:482 `mint.reshape(kv_t, (b*sk, d))`：单缺维由元素数守恒唯一确定"
    placement: passthrough
    bwd:  host_only
    cost: {class: host_only}

  Permute:
    optype: elementwise
    bprop:  {inputs: []}
    shape:
      rel:    permute
      args:   {input: in0, dims: attrs.permute_dims}
      guards: [dims_is_permutation]
      on_guard_fail: numel_only
      why:    "out.shape[i]=x.shape[dims[i]]（算子定义）；轴序 walker 已抠进 attrs"

  AdvancedIndex:
    optype: elementwise
    bprop:  {inputs: [1], why: "反向 = dy scatter_add 回零张量的 index 位置"}
    shape:
      rel:    index_compose
      args:   {index: in1, base: in0}
      guards: [exactly_two_tensor_operands, index_dtype_is_integer, base_rank_ge_1]
      on_guard_fail: reject          # 布尔掩码语义不同（产出长度取决于值）→ 宁 `?` 勿错
      why:    "csa.py:485 `kv_flat[flat_indices]` → idx.shape ++ x.shape[1:]（NumPy/torch 定义）"

# ③ 声明面（源码读不出的量；今天的 _KERNEL_SAVES + _DECLARED_OPAQUE_OUTS + 手写 workspace 合一）
kernels:
  npu_lightning_indexer:
    op: Kernel
    saves:     {ins_idx: all, src: "hyper_parallel/.../custom_op_impl.py:390-391"}
    out_shape: {declared: "~S·B·index_topk", src: "indexer.py:219"}
    workspace:
      bwd_workspace: "209715200 + B*S*33920*4"
      src:    "analysis/… 标定报告"
      bounds: "S 三点线性✓(1024/2048/4096)；cp>1 未实测；tp 不切=OOM 安全侧；B 只测过 1"
```

**四条纪律**（三段共用，沿用 `_KERNEL_SAVES` 现有的）：① 每条带 `src` + `why`；② 逐项进
`Coverage.declared_*`（报告可见）；③ **永不**与"推断出来的"混为一谈；④ 读不出的位留 `None`，不给数。

**键为什么用 kernel/Function 名而非行号**：workspace 是**算子实现**的属性，跟着 kernel 走不跟着
调用点走；对行号漂移免疫，且快照 md5 门天然守护。`construct_walker.py:1734-1735` 已在节点上写了
`attrs = {"kernel": func.id}`。

## 6. 阻断机制（U5）

**建图后完备性门**：

```python
def assert_registry_complete(dag, registry, *, faces) -> None:
    """遍历 OpDAG 收集出现过的每个 (原语路径 | op 类型 | kernel 名)，逐个查 registry
    是否给全 `faces` 要求的面；缺任何一条 → OpRegistryIncomplete，拒绝出数。"""
```

`faces` 按 A2 分档。阻断时**输出可粘贴 stub**（`src` 由 walker 自动填，语义位留 `TODO` 并附提问）：

```
OpRegistryIncomplete: 3 个 op 缺定义，仿真拒绝出数。
把下面这段存成 opdefs/site/custom.yaml 并填完 TODO：

ops:
  ScatterAdd:                    # 出现 4 次，首次 @ moe/router.py:412
    optype: TODO                 # 从 {matmul,moe_gemm,norm,flash_attn,elementwise,...} 选
    bprop:  {inputs: TODO, why: TODO}   # 反向要读哪些操作数？（教科书 VJP，写清依据）
    shape:  TODO                 # {rel: <组合子>, args: {...}}；组合子清单见 opdefs/README.md
    src:    "moe/router.py:412"  # 已自动填
```

**组合子扩展点的治理**：`rel:` 引用不存在的组合子 → 加载即 fail-loud，报"需新增组合子 `<name>`，
请说明其算子定义"。组合子是可数、要写理由的显式扩展点，不是后门。

## 7. 已发现的缺陷（本设计一并修复；每条均已核实）

| # | 缺陷 | 证据 | 修复落点 |
|---|---|---|---|
| ① | **两个 localizer 的 CP 规则分道**：张量 `S·B·S`、cp=2 → mem `local_shape_elems` 得 numel **8,388,608**（切首个 S 轴后 break，`consumer.py:255-261`）；time `localize` 得 **4,194,304**（每个 S 轴都切，`shard_rules.py:60-65`）。**同一物理量差 2×，今天无断言把两者放在一起看** | 实跑复现 | S3 唯一实现 + `test_localizer_parity` |
| ② | `primitives.to_op_type` / `OPDAG_OP_TO_OPTYPE`（`primitives.py:275-309`）是为"防 `OpType` 拼错静默走错分支"而写的 fail-loud 表，**全仓零调用者**；现役是 `to_resolved._OPTYPE.get(op, op.lower())` 静默兜底。分叉后果：`FusedFunction`/`Kernel` 权威表映射 `flash_attn`、现役落 `"fusedfunction"`/`"kernel"` → `RecomputeSpec.op_matches`（`specs.py:288-291`，子串匹配；调用点 `liveness/simulate.py:527,565`）在两条来源下**命中不同的 op**。今天八跑只有 full/off，故**潜伏未发生** | 代码核实 | S1-c |
| ③ | `timesim/bwd_rules.py:86-88` 对**任何**未知 op 类型静默产 `<op_type>Grad` | 代码核实 | §6 完备性门 |
| ④ | `Coverage.totals()`（`to_resolved.py:382-400`）递归加子 Coverage，而 `_build_layer:1395-1396` 已把 segment 计数累加进层级 cov → `n_nodes`/`n_ops` **双计**。报 `2568→2056`，真值 `1284→1028`。不影响字节，但让跳过率看起来好一倍（真值 256/1284≈20%） | 实跑核实 | 随 S1 修 |
| ⑤ | 手写 op 名是真源属性名的**截断近似**（`linear_q_down` vs `self.linear_q_down_proj` @ `deepseek_v4_hybrid_attention.py:92,237`），无人检查 | 源码核实 | S1-a（U6） |
| ⑥ | **dtype 推断散在至少 9 处**（walker 6 + `shape_infer` 1 + `bprop_rules` 1 + 适配器 1 处撤销），含两份提升序表、三处 `"bf16"` 硬兜底、三处 Compare→bool 重复；且 `shape_infer.py:1272-1281` 的注释记载它推错过一次真实的 **2×** 字节错误（O(S²) mask 被记成 bf16）。Norm 的 fp32 更是「预抬→撤销→再抬」三段往返 | 代码核实 | §5.1.1 dtype 面归口 |

## 8. 测试与验收门

| 门 | 内容 |
|---|---|
| `test_optype_vocab` | Registry 的 op 类型键集合 == `bprop` 面键集合 == `bwd` 面键集合（三面同表后退化为 schema 校验，仍保留防回归） |
| `test_localizer_parity` | 一组代表性符号 shape（含 `S·B·S`、`E·cap·H`、`n_heads·v_head_dim`、含 `-1` 消元的）× 一组并行度：mem 路径 numel ≡ time 路径逐轴之积 |
| `test_registry_completeness` | 全模型抽图后 `assert_registry_complete` 通过；人为删一条 op 描述 → 阻断且 stub 含正确 `src` |
| `test_dtype_single_source` | 全图无任何张量 dtype 来自硬兜底（取消 `"bf16"` 默认后，缺依据即阻断）；`csa.py:779` / `:810` 两个 O(S²) mask 恒为 `bool`（缺陷 ⑥ 的 2× 防回归）；Norm 输入 fp32 **只被抬一次** |
| `test_registry_load` | schema 校验 / 缺 `src` 拒绝 / 未知组合子拒绝 / 无 `override` 的重复定义拒绝 |
| `test_timesim_decoupling` | **更新**：白名单加 `opdefs`；其余四条契约不变（§2） |
| `tools/liveness_ab_validate.py` | 167 A/B 八跑，`REAL_SHA256` 钉真机；每步迁移的主门 |
| 既有 479 opdag 测试 + 12 内存锚点 | 全程不破 |

## 9. 迁移路径（U7）

| 步 | 动作 | 门 |
|---|---|---|
| 0 | S1-a/c/d（`qname`、删 `_OPTYPE`、识别面迁 YAML）+ 缺陷 ④ | 八跑**逐字节不变**（纯改名/改表） |
| 1 | S3 唯一 localizer + parity 门 | 显存八跑**逐字节不变**（A1 采 mem 语义 ⇒ mem 侧零变化）；**time 侧会变**——缺陷 ① 的修正使多 S 轴张量的 local shape 改变，须逐条列出受影响算子并记进变更说明，不得混在"纯重构"里过门 |
| 2 | S1-b 原子 op 拆分 + Registry 全面接管**七个面**（含 §5.1.1 dtype 归口：取消三处 `"bf16"` 硬兜底、合并两份提升序表、拆掉 Norm 的预抬-撤销-再抬往返） | 479 opdag 测试不破；八跑逐字节不变（dtype 归口若改变任何张量的 dtype，须逐条列出并单独记账，不得混在"纯重构"里过门） |
| 3 | §6 完备性门开启（先 warn 一轮观察缺口，再切 raise） | 缺口清单进 `Coverage` |
| 4 | S2 placement 落地，mem 侧补传 `shard` | `tp=cp=1` 八跑逐字节不变；新增 `tp=2` 用例从 fail-loud 变出数 |
| 5 | Registry `kernels:` 段建表，标定量从 `layers/*.py` 迁入 | `extracted` 的 workspace/bwd_scratch 从恒 0 变有值，与 `hand_spec` 逐桶对照 |
| 6 | 收两个级联根（`indexer.py:219` Constant ×4、`router.py:393` TopK ×3），跳过 256→~0 | `Coverage.skipped_ops` |
| 7 | **`act_live` 桶**切到 extracted，其余 11 个非-liveness 桶仍用标定 | 八跑上 `act_live` 的 sim/real 不劣于手写现值（1.62–1.64×），且聚合不劣化 |
| 8 | `layers/*.py` op 结构 + `saves` 删除；`shape_eval.ShapeEval` 退役 | 全绿后 |

## 10. 诚实边界 / 非目标

- **不承诺 extracted 立刻达标**。今天它是**下界**（r0 0.750× / r4 0.518× / r128 0.554× 真机，
  `census_arbitration` §0），因 workspace 恒 0、256 op 被跳过、fused kernel 自身输出 saves 不给数。
  第 5/6 步是为了闭合前两项；**kernel 自身输出的 saves**（`.cc` 里的末轴）结构上读不出，
  继续显式 unresolved，不给数。
- **不扩抽取覆盖面**。抽取图今天只覆盖 `embedding` / `dsv4hyb` / `lm_head` / `mtp`
  （`to_resolved._layer_kind:1348` 对其余层型 fail-loud）。DSv3/GQA/dense 层型的抽取源是**独立工作**，
  不在本设计范围——故 12 锚点里非 DSv4 的那些**继续走手写图**，第 8 步只删 DSv4 相关的 builder。
- **不改出数口径本身**：`mem_timeline` 18 桶、`liveness` 7 类、`segment_sim`/`pipeline_sim`、
  `schedule` 全部不动。本设计只改"图和布局从哪来"。
- **不做运行时探针驱动的通信注入**（timesim spec 的 v1.5 项）。S2 只把今天硬编码的模块语义
  注入规则搬到共享层并写进 Registry，不改其判据。
