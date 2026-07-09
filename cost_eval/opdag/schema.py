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

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "OpDAG":
        d = json.loads(s)
        nodes = [OpNode(**n) for n in d.pop("nodes", [])]
        return cls(nodes=nodes, **d)
