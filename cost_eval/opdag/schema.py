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
    # 标量 shape 绑定(`seq, bs, h = x.shape` 这类不产 op 的解包):{"names":[...], "src": "<var>"}。
    # shape 推断按 src 的已知 shape 逐轴取,填 reshape/split 表达式里的标量名。
    scalar_binds: list = field(default_factory=list)
    # self.<attr> → 符号 token 串(由 __init__ 维度求值得,供 shape 推断解析 reshape/split 里的 self.X)。
    dims_ctx: dict = field(default_factory=dict)
    # T0-6.5 Fix2:_handle_call 终端 fallthrough(既非四种已知调用形态、也非内部方法/Morph 别名)命中的
    # 调用点显式记录(镜像 comm_probe.CommSite.opaque_guards 的"opacity 显式携带,不静默丢"先例)。
    # 每条 {"src": "file.py:line", "expr": "<ast.unparse 的调用原文>"}。该调用**不**产 DAG 节点
    # (如 embedding 段的 inline AllReduce 由 comm_probe 另侧覆盖,walker 若也发射会双重计数)——
    # 消费方需对未在已知白名单内的条目自行判断是否 fail-loud,不猜其语义。
    opaque_calls: list = field(default_factory=list)
    # ── 抽取诊断(Task 2 / 评估文档 P0#1,2026-07-25)────────────────────────────────────────
    # 走查过程中**看不懂而没建节点**的一切,逐条带 `src=file:line` + 原文。纪律(评估文档 §11):
    # 「任何抽取器看不懂的东西都必须显式出现在 unresolved / opaque_calls / extraction_failures 里」
    # ——本字段就是前者在 walker 侧的落地。键见 `construct_walker.DIAG_KINDS`:
    #   dropped_stmts        walk_stmt 不处理的语句类(With/For/While/Try/AugAssign/…);
    #                        `with _no_grad():` 另带 note(整块 detach 信号,路线 B P0#4)
    #   dropped_assigns      _handle_assign 不支持的 RHS 形态(BinOp/Compare/Subscript/Name/…)
    #   unregistered_targets 赋值目标从未进 SSA(被丢的赋值 / opaque 调用的目标)——下游会拿占位 ref、丢边
    #   unresolved_operands  _emit 里落占位 ref `name:?:bf16` 的操作数(非 construct 形参)
    #   unbound_aliases      __init__ 里的裸函数别名(`self.reshape = mint.reshape`),_CLS2OP 绑不上
    # 每条是 dict(纯数据,JSON 往返安全)。空字典 = 尚未走查 / 无诊断。
    diagnostics: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "OpDAG":
        d = json.loads(s)
        nodes = [OpNode(**n) for n in d.pop("nodes", [])]
        return cls(nodes=nodes, **d)
