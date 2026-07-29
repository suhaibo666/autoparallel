"""**`ResolvedLayer` 适配器契约** —— 抽取图要长成什么样，`liveness/graph.py` 才能一行不改地吃下。

为什么要有这份契约
------------------
显存今天算自**手写** op 名册（`cost_eval/layers/*.py` 的 `saves=[...]`）。改成从真 MindFormers 源
抽（`cost_eval/opdag/`，纯 AST）之后，`liveness/simulate.py`（已实现 full/select 重算的图变换）与
`schedule.py` 事件走查**都不该动**：只需一个适配器 `cost_eval/opdag/to_resolved.py` 把抽出来的
`OpDAG` 折成 `ResolvedLayer`。本模块把「折成什么」写成**可执行的契约**（而不是散落在设计文档里的
散句），并且用它**先验证现有手写路径**——先证明缝是真的，再让抽取侧接上来
（`tests/test_resolved_layer_contract.py`）。

契约面（三层，逐层的字段语义与 `shape_eval.py:62-204` 的现役实现一致）
--------------------------------------------------------------------
``ResolvedLayerContract``  ``layer_id: int``, ``layer_type: str``, ``ops: Sequence[op]``
``ResolvedOpContract``     ``name``, ``type``, ``inputs``, ``output``, ``params``, ``saves``,
                           ``workspace_bytes``, ``bwd_scratch_bytes``, ``norm_kind``
``ResolvedTensorContract`` ``name``, ``local_numel``, ``dtype_bytes``, ``is_weight``, ``detached``,
                           ``is_expert``, ``pin_under_recompute``, ``dim0``

⚠ **契约面比 `liveness/graph.py` 读的字段更宽** —— 这不是冗余：`simulate_liveness` 只接管
**激活**四桶，其余非 liveness 桶（`persistent` / `gather_buf` / `grad_buf` / `bwd_scratch` /
`swap_buf` …）原样沿用 `structure_mem.estimate_structure_memory` 与 `static_mem.StaticMem`
（`simulate.py:211-213,268-276`）。故一个 graph source 必须同时喂得动那两条：
`structure_mem.py:275,301` **直接属性访问** `w.is_expert`（无 getattr 兜底，缺就 AttributeError）；
`:100` 读 `dim0`（FSDP 按参数**首维**切，`dim0=0` 会静默退回 total-numel 整除口径 → 分片判定
可能与 runtime 不一致）；`:293` 读 `pin_under_recompute`。这三项由本契约的 **T2** 覆盖
（缺一项就等于"图能建、显存算不出来"）。

**硬规则（两条来源都必须满足；现有手写路径实测全过，见 §测试）**

===== 结构 / 打型 =====
- **L1** `layer_id` 是 ≥0 的 int；`layer_type` 是非空 str；`ops` 非空。
- **O1** 每个 op 有非空 `name`；`type` 可取字符串（enum 走 `.value`）。
- **O2** `inputs` / `params` / `saves` 是序列，`output` 是**单个**张量。
- **O3** `workspace_bytes` / `bwd_scratch_bytes` 是 ≥0 的 int（**不**要求能从源推出，见 §不要求）。
- **O4**（2026-07-29）**`type == "norm"` 的 op 必须真实带** `norm_kind` ∈ {`"layernorm"`,
  `"rmsnorm"`}（与 `detached` 同理由，不接受 getattr 兜底）。它是**非 liveness 桶的直通**：
  `structure_mem._norm_save_names` 按它决定该 norm op 的 saves 要不要按 `norm_compute_dtype_bytes`
  抬成 fp32 —— `FusedLayerNorm` 真 cast（`layer_norm.py:93-101`）要抬、`FusedRMSNorm` 输入直通
  （`:151-155`，`:149` 的 self.cast 是死属性）不抬。缺它 = 「图能建、显存算错」，故与 T2 同级强制。
  非 norm 的 op 可省（`_norm_save_names` 先按 type 过滤 → 无副作用）；**给了就必须合法**
  （拼错字符串会静默退回「抬 fp32」，故非法值一律违约）。
- **T1** 每个张量有非空 `name`、`local_numel: int`、`dtype_bytes: int`、`is_weight: bool`、
  `detached: bool`。`detached` 必须**真实存在**（不接受靠 `getattr(t,"detached",False)` 兜底：
  「没标 detach」和「标了不 detach」在 grad 可达性上是两件事，必须由生产者显式表态）。
- **T2** 非 liveness 桶的直通字段也必须在：`is_expert: bool`（`structure_mem.py:275,301`
  **直接属性访问**）、`dim0: int >= 0`（`:100`，FSDP 按参数首维切；0 = 未知 → 退回 total-numel
  整除口径，与 runtime 可能不一致）、`pin_under_recompute: bool`（`:293`）。
  只满足 T1 的图**能建 LayerGraph、但算不出显存**（实测：外部生产者漏 `is_expert` 直接
  `AttributeError` 于 `structure_mem.py:275`）。

===== 字节必须已解析（挡住抽取侧「shape 全是 `?` → total_bytes=0」）=====
- **B1** `local_numel > 0` **且** `dtype_bytes > 0`，逐张量。
  裸抽取 DAG 今天 `_emit` 恒写 `?`（`construct_walker.py:900`），`consumer` 给
  `total_bytes=0` + 全部 `unresolved`（评估文档 §7.1 实测）。B1 让这种图**不能**冒充可用图。
- **B2** `dtype_bytes ∈ {1,2,4,8}`。
- **B3** `layer_total_bytes(layer) > 0`（B1 的层级推论，单独暴露以便一眼判死）。
- **B4** 字节是**本地**（local）量：TP/EP/CP 切分与 FSDP 分片**已经除过**。契约无法从单层直接
  验证「除对了没有」，故这条落在 A/B 上：抽取侧与手写侧的 `param_bytes(layer)` 必须一致
  （`validate_param_census`），不一致就是 shard 没应用或权重册漏项。

===== params vs activations（挡住抽取侧「权重被当 activation save 计」）=====
- **W1** `op.params` 里每个张量 `is_weight=True`。
- **W2** **`op.saves` 里不得有 `is_weight=True` 的张量** —— 权重不是激活 save。
  实测后果：`FFNGroupedGEMM` 那「236 MiB」里 `w1`(58.7 MB)+`w2`(29.4 MB)=**88 MiB 是权重**被当
  激活 save 计入（评估文档 §7.2/§9）。`OpDAG` 没有 `is_weight` 概念，权重只是普通 `ins`。
- **W3** `op.output.is_weight == False`（op 不产出参数）。
- **W4** `is_weight=True` 的张量不得是任何 op 的 `output`（权重是叶子，不是算出来的）。
- **W5** 同名张量在同层内 `is_weight` 一致（不能一处当权重一处当激活）。
- **W6** `detached=True` ⇒ `is_weight=False`（params 是梯度根，不可能被 detach）。
- ⚠ **不要求** weight 型 `input` 必须出现在该 op 的 `params` 里：**tie（权重共享）**是真实建模
  形态 —— `tie_word_embeddings=True` 时 lm_head 复用 embedding 的 `emb_w`，故意 `params=[]`
  以免 vocab×H 持久量重复计一次（`layers/head.py:84-88`、`:220-222`）。契约只要求它
  `is_weight=True`（liveness 的 `alloc()` 据此跳过激活记账）。
  **W2+W4+`validate_param_census` 三条合起来**才是「权重不许混进激活」的完整闸门：W2 挡住
  「权重进了 saves 且标对了」，`param_census` 挡住「权重压根没进 params」（则 param 字节短
  一截，A/B 一比即现），W4 挡住「把权重当某 op 的产物」。

===== saves / 反向 =====
- **S1** `saves ⊆ inputs ∪ {output} ∪ internal`：每个 save 必须是该 op **碰过**的张量，或该 op
  的**内部中间量**（只在 `saves` 里出现、不是数据流边）。挡住「把不相干张量挂到某 op 名下」。
- **S2** `saves` 是**逐 op** 的。`opdag.bprop_rules.derive_saves` 今天返回**全 DAG 按名去重的
  扁平表**（`bprop_rules.py:40,59` 的 `saves.setdefault` 只留一个 `op_id`）；liveness 要
  `saved_by: name -> {op idx}`（`graph.py:223-231`）→ 适配器必须逐节点给 saves，同名可被多个
  op 保留。契约的可检形式：`saves` 出现在**它的读者 op** 上（S1 已蕴含 op 碰过它）。
- **S3** 同层同名张量的 `(local_numel, dtype_bytes)` 必须一致 —— liveness 建册用
  `raw.setdefault(t.name, t)`（`graph.py:164-167`），首见定型，冲突会静默取首个。

===== grad 可达性（新设计的最高价值输出）=====
- **G1** `detached` 由生产者从**源**填（`ops.stop_gradient` / `with _no_grad():` 块内产物），
  不是默认值。契约侧只能验存在性（T1）与自洽性（G2/G3）。
- **G2** 派生自洽：`build_layer_graph(layer)` 必须能跑通，且 `detached` 张量**不得**出现在
  `kept_for_backward` 里（stop_gradient 切断 ⇒ 反向无节点读它）。
- **G3** 派生自洽：所有 `is_weight` 张量 `requires_grad=True`（params 是梯度根）。
- 真机实例（规则通用，不硬编码）：`csa.py:794-795` 的 `unfused_indexer_loss(...,
  ops.stop_gradient(query), ops.stop_gradient(compressed_kv))` → `indexer.py:350` 的 O(S·S/r)
  `matmul` + `:380` 的 fp32 `softmax` 全在 detached 侧 → 纯瞬态。手写 census
  （`layers/dsv4_hybrid.py:207-210`）曾把这对张量声明成 `saves`。

§不要求（明确排除，免得抽取侧被逼去源里找不存在的东西）
------------------------------------------------------
- `workspace_bytes` / `bwd_scratch_bytes` 是**kernel 实现细节**，源码里读不出来 → 契约只要求
  「是个 ≥0 的 int」，**继续沿用手写 / profiler 标定**（评估文档 §9 同结论）。
- `collectives`：`liveness` / `structure_mem` 都不读它（通信缓冲走 `mem_timeline` 的既有标定），
  可为空。
- ⚠ 与早期设计笔记的口径修正：`is_expert` / `dim0` / `pin_under_recompute` 曾被记成"桶模型专用、
  liveness 不读、缺省即可"。**实测不成立** —— `simulate_liveness` 内部就在调
  `estimate_structure_memory`/`StaticMem`，缺 `is_expert` 直接 `AttributeError`。已升为 T2 硬要求。

用法
----
    from cost_eval.liveness.contract import validate_resolved_layer, assert_resolved_layer
    for st, layers in graph.stages.items():
        for layer in layers:
            assert_resolved_layer(layer)          # 违约即抛，附逐条 rule 编号 + 定位
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from ..model_spec import NORM_KIND_CASTING, NORM_KIND_NONCASTING

#: O4 合法取值（见模块 docstring）。
_NORM_KINDS = frozenset({NORM_KIND_CASTING, NORM_KIND_NONCASTING})

__all__ = [
    "ResolvedTensorContract", "ResolvedOpContract", "ResolvedLayerContract",
    "Violation", "validate_resolved_layer", "assert_resolved_layer",
    "validate_resolved_graph", "layer_total_bytes", "layer_param_bytes",
    "layer_activation_bytes", "layer_entry_names", "layer_internal_save_names",
    "validate_param_census", "CONTRACT_RULES",
]

_LEGAL_DTYPE_BYTES = (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# 契约面（Protocol —— 结构化打型，不要求继承）
# ---------------------------------------------------------------------------

@runtime_checkable
class ResolvedTensorContract(Protocol):
    """一个已解析到**本地字节**的张量。

    `local_numel` 已按 TP/EP shard 与 CP 序列切分除过（`shape_eval.resolve_tensor`）；
    `dtype_bytes` 是该张量**自身** dtype 的字节数（norm op 保留输入的 fp32 cast 由
    `structure_mem._dt` 在消费侧按 op 类型再抬，**不**在这里预抬）。

    后三项是**非 liveness 桶**的直通（T2）：`is_expert` 决定权重走 EP-FSDP 还是 dense-FSDP
    分片（`structure_mem.py:275,301` 直接属性访问）；`dim0` 是 TP/EP placement 后的**本地首维**，
    runtime FSDP 按首维切（`parallelize.py:331-350`），0 = 未知 → 退回 total-numel 整除；
    `pin_under_recompute` 标记 fused 自定义算子 ctx 保存集在全重算下**不释放**。
    """
    name: str
    local_numel: int
    dtype_bytes: int
    is_weight: bool
    detached: bool
    is_expert: bool
    dim0: int
    pin_under_recompute: bool


@runtime_checkable
class ResolvedOpContract(Protocol):
    """一个前向 op。反向节点由 liveness 从 `saves` + grad 可达性**导出**，不需要显式给。"""
    name: str
    type: object              # str 或带 `.value` 的 enum
    inputs: Sequence
    output: ResolvedTensorContract
    params: Sequence
    saves: Sequence
    workspace_bytes: int
    bwd_scratch_bytes: int
    #: O4：norm 的种类（"layernorm" = 真 cast 输入 / "rmsnorm" = 输入直通）。决定
    #: `structure_mem._dt` 的 norm-fp32 抬升对该 op 是否成立。见模块 docstring O4。
    norm_kind: str


@runtime_checkable
class ResolvedLayerContract(Protocol):
    """一层（或伪层 embedding / lm_head / mtp）的完整前向 op 序，按**执行序**。"""
    layer_id: int
    layer_type: str
    ops: Sequence


# ---------------------------------------------------------------------------
# 违约记录
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    """一条违约：`rule` = 上面 docstring 的规则编号，`where` = 定位，`detail` = 实测值。"""
    rule: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.where}: {self.detail}"


#: 规则编号 → 一句话说明（供报告/错误信息自解释；与 docstring 同步）。
CONTRACT_RULES = {
    "L1": "layer_id/layer_type/ops 打型与非空",
    "O1": "op.name 非空、op.type 可取字符串",
    "O2": "inputs/params/saves 是序列，output 是单张量",
    "O3": "workspace_bytes/bwd_scratch_bytes 是 >=0 的 int",
    "O4": "norm_kind ∈ {layernorm, rmsnorm} 且必须真实存在（norm-fp32 抬升按它分辨）",
    "T1": "张量必备字段 name/local_numel/dtype_bytes/is_weight/detached（detached 不可缺省兜底）",
    "T2": "非 liveness 桶直通字段 is_expert/dim0/pin_under_recompute 必须在（structure_mem 直读）",
    "B1": "local_numel>0 且 dtype_bytes>0（挡住 shape='?' → 0 字节的裸抽取图）",
    "B2": "dtype_bytes ∈ {1,2,4,8}",
    "B3": "layer_total_bytes > 0",
    "B4": "param 字节与参考来源一致（validate_param_census；挡'权重压根没进 params'）",
    "W1": "params 里每个张量 is_weight=True",
    "W2": "saves 里不得有 is_weight=True 的张量（权重不是激活 save）",
    "W3": "op.output.is_weight == False",
    "W4": "is_weight 张量不得是任何 op 的 output（权重是叶子）",
    "W5": "同名张量的 is_weight 在层内一致",
    "W6": "detached=True ⇒ is_weight=False",
    "S1": "saves ⊆ inputs ∪ {output} ∪ internal",
    "S3": "同名张量的 (local_numel, dtype_bytes) 在层内一致",
    "G2": "detached 张量不得进 kept_for_backward",
    "G3": "is_weight 张量 requires_grad=True",
}


# ---------------------------------------------------------------------------
# 派生量（也供 A/B 报告直用）
# ---------------------------------------------------------------------------

def _nbytes(t) -> int:
    return int(t.local_numel) * int(t.dtype_bytes)


def _op_type_str(op) -> str:
    t = getattr(op, "type", "")
    return str(getattr(t, "value", t))


def layer_flow_names(layer) -> frozenset:
    """数据流边名集合（inputs ∪ outputs）—— `internal` 的补集判据。"""
    names: set = set()
    for op in layer.ops:
        for t in op.inputs:
            names.add(t.name)
        names.add(op.output.name)
    return frozenset(names)


def layer_internal_save_names(layer) -> frozenset:
    """只在 `saves` 里出现的**op 内部中间量**名（非数据流边、非权重）。"""
    flow = layer_flow_names(layer)
    return frozenset(s.name for op in layer.ops for s in op.saves
                     if s.name not in flow and not s.is_weight)


def layer_entry_names(layer) -> frozenset:
    """图入口激活名：非权重、且不被该层任何 op 产出（层 construct 入参 / 上一 stage 送来的激活）。

    与 `graph.py:187-188` 的 `entries` 同判据。抽取侧若把权重漏标成激活，它们会**冒充图入口**
    出现在这里 —— 故 A/B 时应逐层比对本集合（手写侧实测每层 ≤2 个，全是真激活）。"""
    produced = {op.output.name for op in layer.ops}
    produced |= layer_internal_save_names(layer)
    return frozenset(t.name for op in layer.ops for t in op.inputs
                     if not t.is_weight and t.name not in produced)


def _unique_tensors(layer) -> dict:
    """层内张量首见定型表（与 `graph.py:164-167` 的 `raw.setdefault` 同口径）。"""
    out: dict = {}
    for op in layer.ops:
        for t in (*op.inputs, op.output, *op.params, *op.saves):
            out.setdefault(t.name, t)
    return out


def layer_total_bytes(layer) -> int:
    """层内**去重后**全部张量的本地字节和（含权重）。裸抽取图这里会是 0 → B3 判死。"""
    return sum(_nbytes(t) for t in _unique_tensors(layer).values())


def layer_param_bytes(layer) -> int:
    """层内去重后**权重**字节和。A/B 的 B4 闸门：抽取侧与手写侧必须一致。"""
    return sum(_nbytes(t) for t in _unique_tensors(layer).values() if t.is_weight)


def layer_activation_bytes(layer) -> int:
    """层内去重后**非权重**字节和（激活 + 内部中间量）。与 `layer_param_bytes` 互补且不重叠。"""
    return sum(_nbytes(t) for t in _unique_tensors(layer).values() if not t.is_weight)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def _check_tensor(t, where: str, out: list) -> bool:
    """逐张量 T1/B1/B2/W6。返回 False 表示打型都没过（后续规则跳过它，免得报一串连带错）。"""
    ok = True
    for attr in ("name", "local_numel", "dtype_bytes", "is_weight", "detached"):
        if not hasattr(t, attr):
            out.append(Violation("T1", where, f"张量缺字段 `{attr}`（{type(t).__name__}）"))
            ok = False
    if not ok:
        return False
    if not isinstance(t.name, str) or not t.name:
        out.append(Violation("T1", where, f"张量 name 非法：{t.name!r}"))
        ok = False
    if not isinstance(t.local_numel, int) or isinstance(t.local_numel, bool):
        out.append(Violation("T1", f"{where}/{t.name}",
                             f"local_numel 必须是 int，实得 {type(t.local_numel).__name__}"))
        ok = False
    elif t.local_numel <= 0:
        out.append(Violation("B1", f"{where}/{t.name}",
                             f"local_numel={t.local_numel} —— 符号 shape 未解析（抽取侧 '?'）"
                             f"或该张量为空，字节记账无从谈起"))
    if not isinstance(t.dtype_bytes, int) or isinstance(t.dtype_bytes, bool):
        out.append(Violation("T1", f"{where}/{t.name}",
                             f"dtype_bytes 必须是 int，实得 {type(t.dtype_bytes).__name__}"))
        ok = False
    elif t.dtype_bytes <= 0:
        out.append(Violation("B1", f"{where}/{t.name}", f"dtype_bytes={t.dtype_bytes}"))
    elif t.dtype_bytes not in _LEGAL_DTYPE_BYTES:
        out.append(Violation("B2", f"{where}/{t.name}",
                             f"dtype_bytes={t.dtype_bytes} 不在 {_LEGAL_DTYPE_BYTES}"))
    if not isinstance(t.is_weight, bool):
        out.append(Violation("T1", f"{where}/{t.name}",
                             f"is_weight 必须是 bool，实得 {type(t.is_weight).__name__}"))
        ok = False
    if not isinstance(t.detached, bool):
        out.append(Violation("T1", f"{where}/{t.name}",
                             f"detached 必须是 bool，实得 {type(t.detached).__name__}"))
        ok = False
    if ok and t.detached and t.is_weight:
        out.append(Violation("W6", f"{where}/{t.name}",
                             "detached=True 且 is_weight=True —— params 是梯度根，不可能被 detach"))
    # ── T2：非 liveness 桶（structure_mem / static_mem）直读的三项 ────────────────
    for attr, want, note in (
            ("is_expert", bool, "structure_mem.py:275,301 直接属性访问，缺则 AttributeError"),
            ("dim0", int, "structure_mem.py:100，FSDP 按参数首维切；0=未知（退回 numel 整除）"),
            ("pin_under_recompute", bool, "structure_mem.py:293，fused ctx 全重算下不释放")):
        if not hasattr(t, attr):
            out.append(Violation("T2", f"{where}/{getattr(t, 'name', '?')}",
                                 f"缺字段 `{attr}` —— {note}"))
            continue
        v = getattr(t, attr)
        if want is bool and not isinstance(v, bool):
            out.append(Violation("T2", f"{where}/{t.name}",
                                 f"{attr} 必须是 bool，实得 {type(v).__name__}"))
        elif want is int and (not isinstance(v, int) or isinstance(v, bool) or v < 0):
            out.append(Violation("T2", f"{where}/{t.name}",
                                 f"{attr}={v!r} 必须是 >=0 的 int"))
    return ok


def validate_resolved_layer(layer, *, check_derived: bool = True) -> tuple:
    """校验一个 `ResolvedLayer`（或任何满足契约面的对象），返回 `tuple[Violation]`（空 = 合约）。

    参数
    ----
    check_derived : 是否额外跑 G2/G3（要 `build_layer_graph` 走一遍，稍慢但覆盖 grad 可达性）。
    """
    out: list = []
    lname = getattr(layer, "layer_type", "<no layer_type>")
    tag = f"L{getattr(layer, 'layer_id', '?')}:{lname}"

    # ── L1 ──────────────────────────────────────────────────────────────────
    if not isinstance(getattr(layer, "layer_id", None), int) or layer.layer_id < 0:
        out.append(Violation("L1", tag, f"layer_id 非法：{getattr(layer, 'layer_id', None)!r}"))
    if not isinstance(lname, str) or not lname:
        out.append(Violation("L1", tag, f"layer_type 非法：{lname!r}"))
    ops = getattr(layer, "ops", None)
    if ops is None or isinstance(ops, (str, bytes)) or not hasattr(ops, "__iter__"):
        out.append(Violation("L1", tag, f"ops 不是序列：{type(ops).__name__}"))
        return tuple(out)
    ops = list(ops)
    if not ops:
        out.append(Violation("L1", tag, "ops 为空 —— 空层无法参与 liveness"))
        return tuple(out)

    typed: dict = {}          # name -> (numel, dtype_bytes, is_weight)  首见定型
    produced_names: set = set()
    okmap: dict = {}          # id(tensor) -> 打型是否过（过不了就不参与后续规则）
    ops_ok: list = []         # 打型过关、可进 pass B 的 op

    # ── pass A：op 打型（O1/O2/O3）+ 逐张量打型（T1/T2/B1/B2/W6）─────────────────
    for i, op in enumerate(ops):
        where = f"{tag}/op{i}"
        # ── O1 / O2 / O3 ────────────────────────────────────────────────────
        oname = getattr(op, "name", None)
        if not isinstance(oname, str) or not oname:
            out.append(Violation("O1", where, f"op.name 非法：{oname!r}"))
        else:
            where = f"{tag}/op{i}:{oname}"
        try:
            _op_type_str(op)
        except Exception as e:                                # pragma: no cover - 防御
            out.append(Violation("O1", where, f"op.type 不可取字符串：{e}"))
        missing = [a for a in ("inputs", "output", "params", "saves") if not hasattr(op, a)]
        if missing:
            out.append(Violation("O2", where, f"op 缺字段 {missing}"))
            continue
        for attr in ("inputs", "params", "saves"):
            v = getattr(op, attr)
            if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
                out.append(Violation("O2", where, f"op.{attr} 不是序列：{type(v).__name__}"))
        if hasattr(op.output, "__iter__") and not isinstance(op.output, str):
            out.append(Violation("O2", where, "op.output 必须是**单个**张量，不是序列"))
            continue
        for attr in ("workspace_bytes", "bwd_scratch_bytes"):
            v = getattr(op, attr, 0)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                out.append(Violation("O3", where, f"op.{attr}={v!r} 必须是 >=0 的 int"))
        # O4：**norm 类型的 op** 必须显式表态 norm 种类（缺 = 图能建、显存按错 dtype 算）。
        #   非 norm 的 op 给不给都行（`_norm_save_names` 先按 type 过滤 → 给了也无副作用）；
        #   给了就必须合法，防拼错字符串静默退回「抬 fp32」。
        _is_norm = _op_type_str(op) == "norm"
        if _is_norm and not hasattr(op, "norm_kind"):
            out.append(Violation("O4", where,
                                 "norm op 缺 norm_kind（norm-fp32 抬升按它分辨，不接受兜底）"))
        elif hasattr(op, "norm_kind") and getattr(op, "norm_kind") not in _NORM_KINDS:
            out.append(Violation("O4", where,
                                 f"op.norm_kind={getattr(op, 'norm_kind')!r} 不在 {sorted(_NORM_KINDS)}"))

        ins = list(op.inputs)
        pars = list(op.params)
        savs = list(op.saves)
        for slot, seq in (("in", ins), ("out", [op.output]), ("param", pars), ("save", savs)):
            for t in seq:
                okmap[id(t)] = _check_tensor(t, f"{where}[{slot}]", out)
        ops_ok.append((i, where, op, ins, pars, savs))

    # 有张量连 `name` 都取不到 → 后续按名做的规则（S1/S3/W*/B3/G*）全无意义，早退。
    if any(not hasattr(t, "name") for _i, _w, op, ins, pars, savs in ops_ok
           for t in (*ins, op.output, *pars, *savs)):
        return tuple(out)

    # ── pass B：按名的语义规则 ──────────────────────────────────────────────────
    flow = layer_flow_names(layer)
    for i, where, op, ins, pars, savs in ops_ok:
        # ── W1 / W3 / W4 / W2 ───────────────────────────────────────────────
        for p in pars:
            if okmap.get(id(p)) and not p.is_weight:
                out.append(Violation("W1", f"{where}/{p.name}",
                                     "出现在 op.params 里但 is_weight=False"))
        if okmap.get(id(op.output)) and op.output.is_weight:
            out.append(Violation("W3", f"{where}/{op.output.name}",
                                 "op.output.is_weight=True —— op 不产出参数"))
        for s in savs:
            if okmap.get(id(s)) and s.is_weight:
                out.append(Violation(
                    "W2", f"{where}/{s.name}",
                    "权重出现在 op.saves 里 —— 权重不是激活 save（实测 FFNGroupedGEMM 的"
                    " w1+w2=88 MiB 曾被这样计入 236 MiB）"))
        if okmap.get(id(op.output)):
            produced_names.add(op.output.name)

        # ── S1：saves ⊆ inputs ∪ {output} ∪ internal ─────────────────────────
        touched = {t.name for t in ins} | {op.output.name}
        for s in savs:
            if not okmap.get(id(s)):
                continue
            if s.name in touched:
                continue
            if s.name not in flow:               # internal（只在 saves 里出现）→ 合法
                continue
            out.append(Violation(
                "S1", f"{where}/{s.name}",
                "save 既不是该 op 的 input/output，也不是它的内部中间量 —— "
                "saves 必须挂在**会在反向读它**的那个 op 上（逐 op saves，非全图扁平表）"))

        # ── S3 / W5：同名一致性 ─────────────────────────────────────────────
        for t in (*ins, op.output, *pars, *savs):
            if not okmap.get(id(t)):
                continue
            sig = (t.local_numel, t.dtype_bytes, t.is_weight)
            prev = typed.setdefault(t.name, sig)
            if prev != sig:
                if prev[2] != sig[2]:
                    out.append(Violation("W5", f"{where}/{t.name}",
                                         f"is_weight 前后不一致：{prev[2]} vs {sig[2]}"))
                if prev[:2] != sig[:2]:
                    out.append(Violation(
                        "S3", f"{where}/{t.name}",
                        f"(local_numel, dtype_bytes) 前后不一致：{prev[:2]} vs {sig[:2]}"
                        " —— liveness 建册首见定型（graph.py:164-167），冲突会被静默吞掉"))

    # ── W4：权重不得是任何 op 的产物 ─────────────────────────────────────────
    for name, sig in typed.items():
        if sig[2] and name in produced_names:
            out.append(Violation("W4", f"{tag}/{name}",
                                 "is_weight=True 却是某 op 的 output —— 权重是叶子，不是算出来的"))

    # ── B3 ──────────────────────────────────────────────────────────────────
    if not any(v.rule in ("T1", "T2", "L1", "O2") for v in out):
        if layer_total_bytes(layer) <= 0:
            out.append(Violation("B3", tag,
                                 "layer_total_bytes=0 —— 整层字节未解析（裸抽取 DAG 的典型征状）"))

    # ── G2 / G3：派生自洽（跑一遍 build_layer_graph）─────────────────────────
    if check_derived and not out:
        from .graph import build_layer_graph
        try:
            lg = build_layer_graph(layer)
        except Exception as e:
            out.append(Violation("G2", tag, f"build_layer_graph 失败：{type(e).__name__}: {e}"))
            return tuple(out)
        det = {n for n, t in lg.tensors.items() if t.detached}
        leak = sorted(det & set(lg.kept_for_backward))
        if leak:
            out.append(Violation("G2", tag,
                                 f"detached 张量进了 kept_for_backward：{leak}"))
        nog = sorted(n for n, t in lg.tensors.items() if t.is_weight and not t.requires_grad)
        if nog:
            out.append(Violation("G3", tag, f"权重未 requires_grad：{nog}"))
    return tuple(out)


def assert_resolved_layer(layer, *, check_derived: bool = True) -> None:
    """`validate_resolved_layer` 的 fail-loud 版本：违约即抛 `AssertionError`（逐条列出）。"""
    v = validate_resolved_layer(layer, check_derived=check_derived)
    if v:
        head = f"ResolvedLayer 契约违约 {len(v)} 条（cost_eval/liveness/contract.py）："
        raise AssertionError(head + "\n  " + "\n  ".join(str(x) for x in v))


def validate_resolved_graph(graph, *, check_derived: bool = True) -> tuple:
    """校验整张 `ResolvedGraph`（`.stages: {stage: [layer, ...]}`），返回全部违约。"""
    out: list = []
    if not hasattr(graph, "stages"):
        return (Violation("L1", "<graph>", f"缺少 `.stages`（{type(graph).__name__}）"),)
    for st in sorted(graph.stages):
        for layer in graph.stages[st]:
            out.extend(validate_resolved_layer(layer, check_derived=check_derived))
    return tuple(out)


def validate_param_census(layer, expected_param_bytes: int, *, tol_bytes: int = 0) -> tuple:
    """**B4 闸门**：该层权重字节必须与参考值一致（默认零容差）。

    这是「权重压根没进 `params`」的唯一可靠探针 —— W2 只能挡住「权重进了 saves 且标对了」。
    参考值取自另一条来源（手写侧）或权威参数册。差额为负 = 抽取侧漏权重（评估文档 §7.2 的
    `w1`+`w2`=88 MiB 就是这种漏法）；为正 = 重复计入（如 tie 权重被两个 op 各算一次）。"""
    got = layer_param_bytes(layer)
    if abs(got - int(expected_param_bytes)) <= tol_bytes:
        return ()
    d = got - int(expected_param_bytes)
    return (Violation(
        "B4", f"L{getattr(layer, 'layer_id', '?')}:{getattr(layer, 'layer_type', '?')}",
        f"param 字节 {got} != 参考 {expected_param_bytes}（差 {d:+d} = {d / 2 ** 20:+.1f} MiB）"
        f"{'；负差 = 权重漏进 params（会被当激活计）' if d < 0 else '；正差 = 权重重复计入'}"),)
