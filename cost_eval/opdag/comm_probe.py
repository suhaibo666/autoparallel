# cost_eval/opdag/comm_probe.py
"""TP 集合通信惯用法静态探测（spec §3.3c 第一类「源码内显式通信」，风险 R1）。

读真 mindformers Cell 源（AST，绝不 import），对给定类提取**全部方法体**（含 Morph 包裹的
forward 方法，如 `RowParallelLinear.forward_func`/`forward_func_with_bias`——它们本身就是
普通 FunctionDef，`self.morphed_forward(...)` 只是间接触发；直接扫方法体即可，无需还原 Morph 别名）
里的集合通信调用点。两种惯用法：
  A) __init__ 绑定：`self.X = ops.AllReduce(group=...)`（链式 `.set_prim_instance_name(...)` 等剥壳）
     → 方法体里 `self.X(t)`；
  B) 内联：`ops.ReduceScatter(group=...)(t)` 直接构造调用（VocabParallelEmbedding.embedding_func）。

guard 语义（关键设计点，由 RowParallelLinear 真实嵌套结构倒推确认）：
  只累计**可识别**的包围条件（`self.<flag>` / `not self.<flag>`），按外→内以 "&" 连接；
  不可识别的包围条件不计入 guard 合取——它们通常是与通信变体选择正交的结构性前提
  （如 `RowParallelLinear.forward_func` 里 `if self.tp != 1: if self.sequence_parallel: ...`——
  外层 `tp != 1` 只是"这些 comm 绑定是否存在"的前提，已被 __init__ 里同一条件蕴含，不参与
  reduce_scatter/all_reduce 的二选一），因此 :619/:646 处 reduce_scatter 的 guard 就是精确的
  "sequence_parallel"，不被外层 "tp != 1" 污染成 "?&sequence_parallel"。
  若调用点**全程只有不可识别条件包围、一个可识别条件都没有** → guard = "?"；调用点完全无
  if 包围 → guard = ""（无条件）。

early-return 互斥性建模（quality review 修洞）：
  块内某 `if` 的 body（对称地 orelse）以 `return` 终结时，块内**其余语句**只在该条件为假
  （真）时可达——等价于被取反（正向）条件包围，同样计入 guard/opaque_guards。典型：
  `VocabParallelEmbedding.embedding_func` 的 sp 支 :179 提前 return，故 :182 的 AllReduce
  实际 guard = "!sequence_parallel&enable_embedding_tp"（真源 :144-145 明文两者互斥二选一），
  而非仅 "enable_embedding_tp"——否则 producer 在 sp=True 时会把两个集合通信都注入。

opaque_guards（fail-loud 恢复，spec review 裁决——被跳过的条件不静默丢弃）：
  每个未计入 guard 的不可识别外层 if 条件，按外→内顺序以 `ast.unparse(test)` 原文记录在
  `CommSite.opaque_guards`（orelse 支记 `"not (<原文>)"`）。消费方（producer）必须对
  opaque_guards 中不在已知冗余白名单（如 `self.tp != 1`）内的条目 fail-loud，不猜其真值。

已知未建模形态（现源无此模式,出现时需重估）：elif-return 链只传播首条件；loop/with 体内的
continue/break 与 raise 终结不做尾部取反。
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
    guard: str      # "" 无条件 | "sequence_parallel" | "!sequence_parallel" | "?" 全程不可识别
    src: str        # "layers.py:619"
    method: str     # 调用点所在方法名
    opaque_guards: tuple[str, ...] = ()   # 被跳过的不可识别外层 if 条件原文(外→内;orelse 支带 "not (…)")


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
    # extractor._unwrap_to_call 的去谓词原语版

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
    """if 条件 → 可识别 guard：self.<flag> → flag；not self.<flag> → !flag；其余 "?"（不可识别）。"""
    if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name) \
            and test.value.id == "self":
        return test.attr
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _guard_of(test.operand)
        return "!" + inner if inner != "?" else "?"
    return "?"


class _MethodScan(ast.NodeVisitor):
    """两条栈 = 当前调用点的可识别(stack)/不可识别(opaque)包围条件前缀，与遍历深度同步压弹。"""

    def __init__(self, binds: dict[str, str], src_file: str, method: str):
        self.binds, self.src_file, self.method = binds, src_file, method
        self.stack: list[str] = []       # 仅可识别层级(外→内)
        self.opaque: list[str] = []      # 不可识别层级的条件原文(外→内,orelse 支带 "not (…)")
        self.sites: list[CommSite] = []

    def _guard(self) -> str:
        if self.stack:
            return "&".join(self.stack)
        if self.opaque:
            return "?"
        return ""

    def _push_cond(self, test: ast.AST, negate: bool) -> list[str]:
        """把 if 条件(negate=其取反)压入对应栈，返回该栈以便调用方对称弹出。"""
        g = _guard_of(test)
        if g != "?":
            if negate:
                # 双重否定归一："!flag" 的否定是 "flag",不产 "!!flag"。
                g = g[1:] if g.startswith("!") else "!" + g
            self.stack.append(g)
            return self.stack
        txt = ast.unparse(test)
        self.opaque.append("not (" + txt + ")" if negate else txt)
        return self.opaque

    def _visit_block(self, stmts: list[ast.stmt]):
        """逐语句遍历一个语句块；块内 `if` 的 body/orelse 以 Return 终结时（early-return
        惯用法），为块内**其余语句**压入取反/正向条件（多个串联叠加，块末平衡弹出）。"""
        tails: list[list[str]] = []      # 本块因 early-return 追加的压栈(记其所在栈)
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                self.visit_If(stmt)
                if stmt.body and isinstance(stmt.body[-1], ast.Return):
                    tails.append(self._push_cond(stmt.test, negate=True))
                if stmt.orelse and isinstance(stmt.orelse[-1], ast.Return):
                    tails.append(self._push_cond(stmt.test, negate=False))
            else:
                self.visit(stmt)
        for st in reversed(tails):
            st.pop()

    def visit_If(self, node: ast.If):
        st = self._push_cond(node.test, negate=False)
        self._visit_block(node.body)
        st.pop()
        st = self._push_cond(node.test, negate=True)
        self._visit_block(node.orelse)
        st.pop()

    def _emit(self, ctype: str, node: ast.Call):
        self.sites.append(CommSite(ctype, self._guard(),
                                   f"{self.src_file}:{node.lineno}", self.method,
                                   tuple(self.opaque)))

    def visit_Call(self, node: ast.Call):
        f = node.func
        # 惯用法A：self.X(...) 且 X 是 __init__ 里的通信绑定
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                and f.value.id == "self" and f.attr in self.binds:
            self._emit(self.binds[f.attr], node)
        # 惯用法B：ops.<Comm>(group=...)(t) 内联构造调用
        if isinstance(f, ast.Call):
            ct = _comm_ctor(f)
            if ct:
                self._emit(ct, node)
        self.generic_visit(node)


def probe_cell_comm(mf_root: str, rel: str, cls_name: str) -> list[CommSite]:
    """静态探测 `cls_name`（含同文件基类 MRO）全部方法体里的集合通信调用点。

    参数
      mf_root  — mindformers 源根（仅 ast 读，绝不 import）。
      rel      — Cell 源文件相对 mf_root 的路径，用 "/" 分隔。
      cls_name — 目标 Cell 类名。
    返回按（类 MRO derived→base、方法定义序、块内出现序）排列的 CommSite 列表。
    文件里找不到 class cls_name → raise ValueError（fail-loud）。
    """
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
            scan._visit_block(fn.body)
            sites.extend(scan.sites)
    return sites
