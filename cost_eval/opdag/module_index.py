# cost_eval/opdag/module_index.py
"""跨文件类解析 / MRO(路线 B P0#3,2026-07-25)。

## 问题(实测)

`extractor._find_class` 与 `construct_walker._Walker._find_class` 都只在**单个已 parse 的 tree**
里找 `ClassDef`。`DSv4HybridSelfAttention(MultiLatentAttention)` 的基类在**另一个文件**
(`pynative/transformers/multi_latent_attention.py:60`),于是基类 `__init__` 里的
`self.shape = ops.shape` / `self.reshape = mint.reshape` / `self.cast = ops.cast` / `self.permute`
(`:127-131`)**永远收不到** → `construct` 里 `self.shape(x)`(`deepseek_v4_hybrid_attention.py:233`)
未绑定 → fail-loud。

DSv3 路径从未撞上这条,因为 training_graph 的 `MLASelfAttention` 与其基类 `MultiLatentAttention`
恰在**同一个文件**里 —— 单文件 MRO 够用纯属巧合。

## 机制

沿**import 语句**解析基类名:纯 `ast`,不 import 任何东西。
  * `from mindformers.pynative.transformers.multi_latent_attention import MultiLatentAttention`
    → 模块路径 → 文件路径(去掉与包目录同名的首段)→ parse → 找 `ClassDef`;
  * `from ... import X as Y` → 记 `Y -> (module, X)`;
  * `import a.b.c` + 基类写作 `c.Cls` / `a.b.c.Cls` → 同法;
  * 同文件优先(基类与派生类同源文件时不走 import)。

**解析不到就返回 None**(调用方按「同文件 MRO」的旧行为继续,或 fail-loud)——绝不臆造基类。
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedClass:
    """一个被解析到的类:类节点 + 它所在文件的 tree/相对路径/basename。"""
    name: str
    node: ast.ClassDef
    tree: ast.AST
    rel: str            # 相对 mf_root,用 "/" 分隔
    src: str            # 该文件全文(供 bind_init 等按文本重 parse 的既有接口)

    @property
    def file(self) -> str:
        return os.path.basename(self.rel)


class ClassIndex:
    """按需(lazy)解析 mf_root 下的模块,提供「类名 → ResolvedClass」的跨文件查找。"""

    def __init__(self, mf_root: str):
        self.mf_root = mf_root
        self._pkg = os.path.basename(os.path.normpath(mf_root))
        self._files: dict[str, tuple[ast.AST, str]] = {}     # rel -> (tree, src)
        self._imports: dict[str, dict[str, tuple[str, str]]] = {}  # rel -> {local: (module, orig)}

    # ── 文件级 ────────────────────────────────────────────────────────────────
    def load(self, rel: str, tree: ast.AST | None = None, src: str | None = None):
        """登记/取一个文件的 (tree, src)。已登记则直接返回(不重复 parse)。"""
        rel = rel.replace(os.sep, "/")
        if rel in self._files:
            return self._files[rel]
        if tree is None or src is None:
            path = os.path.join(self.mf_root, *rel.split("/"))
            if not os.path.isfile(path):
                return None
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            tree = ast.parse(src, filename=path)
        self._files[rel] = (tree, src)
        self._imports[rel] = self._scan_imports(tree)
        return self._files[rel]

    @staticmethod
    def _scan_imports(tree: ast.AST) -> dict:
        """收 `from M import A as B` / `import M as N` → {本地名: (模块路径, 原名)}。
        `import a.b.c`(无 as)记两把钥匙:`a.b.c` 与末段 `c`,让 `c.Cls` 也解得开。"""
        out: dict[str, tuple[str, str]] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                if n.module is None or n.level:
                    continue                    # 相对 import:本库源里未出现,不臆测
                for a in n.names:
                    out[a.asname or a.name] = (n.module, a.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname:
                        out[a.asname] = (a.name, "")
                    else:
                        out[a.name] = (a.name, "")
                        out[a.name.split(".")[-1]] = (a.name, "")
        return out

    def _module_to_rel(self, module: str) -> list[str]:
        """`mindformers.pynative.x.y` → 候选相对路径(模块文件 + 包 __init__)。"""
        parts = module.split(".")
        if parts and parts[0] == self._pkg:
            parts = parts[1:]
        if not parts:
            return []
        base = "/".join(parts)
        return [f"{base}.py", f"{base}/__init__.py"]

    # ── 类级 ──────────────────────────────────────────────────────────────────
    def _in_file(self, rel: str, cls_name: str) -> ResolvedClass | None:
        got = self.load(rel)
        if got is None:
            return None
        tree, src = got
        node = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == cls_name), None)
        if node is None:
            return None
        return ResolvedClass(name=cls_name, node=node, tree=tree, rel=rel, src=src)

    def resolve(self, cls_name: str, from_rel: str, _depth: int = 0) -> ResolvedClass | None:
        """在 `from_rel` 的作用域里解析类名 `cls_name`(先同文件,再跟 import)。"""
        if _depth > 6:          # import 转发链上限(防 __init__.py 互转发成环)
            return None
        from_rel = from_rel.replace(os.sep, "/")
        hit = self._in_file(from_rel, cls_name)
        if hit is not None:
            return hit
        imports = self._imports.get(from_rel) or {}
        target = imports.get(cls_name)
        if target is None:
            return None
        module, orig = target
        want = orig or cls_name
        for rel in self._module_to_rel(module):
            got = self._in_file(rel, want)
            if got is not None:
                # `from M import X as Y`:节点名是 X,但调用方按 Y 找 —— 记 X(真实定义名)。
                return got
            # 该模块存在但类不在其中(典型:包 __init__.py 只做转发)→ 顺着它的 import 再找一层
            if self.load(rel) is not None:
                deeper = self.resolve(want, rel, _depth + 1)
                if deeper is not None:
                    return deeper
        return None

    def base_names(self, cls: ast.ClassDef) -> list[str]:
        """基类名列表:`Name` 取 id;`Attribute`(如 `nn.Cell`)取末段 attr。"""
        out: list[str] = []
        for b in cls.bases:
            if isinstance(b, ast.Name):
                out.append(b.id)
            elif isinstance(b, ast.Attribute):
                out.append(b.attr)
        return out

    def mro(self, cls_name: str, from_rel: str) -> list[ResolvedClass]:
        """derived→base 的 BFS「MRO」(**跨文件**),解析不到的基类(如 `nn.Cell`)自然止步。"""
        order: list[ResolvedClass] = []
        seen: set[tuple[str, str]] = set()
        queue: list[tuple[str, str]] = [(cls_name, from_rel)]
        while queue:
            name, rel = queue.pop(0)
            rc = self.resolve(name, rel)
            if rc is None:
                continue
            key = (rc.rel, rc.name)
            if key in seen:
                continue
            seen.add(key)
            order.append(rc)
            for b in self.base_names(rc.node):
                queue.append((b, rc.rel))
        return order
