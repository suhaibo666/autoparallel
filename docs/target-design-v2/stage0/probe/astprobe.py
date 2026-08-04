"""
astprobe.py -- 纯静态 AST 探针库。
绝不 import 被分析的框架，只用 ast.parse 解析源码文本。

每个公开函数对应产出报告里的一个数，报告中会注明 "astprobe.<func>"。
"""
from __future__ import annotations

import ast
import io
import os
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------- 文件发现


def iter_py_files(root, exclude_dirs=()):
    """遍历 root 下所有 .py。exclude_dirs 是相对 root 的 posix 前缀。"""
    root = os.path.abspath(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ('__pycache__', '.git', '.mypy_cache', '.pytest_cache')]
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace('\\', '/')
            if any(rel == e or rel.startswith(e.rstrip('/') + '/') for e in exclude_dirs):
                continue
            out.append((rel, full))
    out.sort()
    return out


def parse_all(files):
    """返回 (ok: {rel: (tree, src, nlines)}, failures: [(rel, errtype, msg)])"""
    ok = {}
    failures = []
    for rel, full in files:
        try:
            with open(full, 'rb') as f:
                raw = f.read()
            src = raw.decode('utf-8', errors='strict')
        except Exception as e:  # noqa: BLE001
            failures.append((rel, 'decode:' + type(e).__name__, str(e)[:200]))
            continue
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError as e:
            failures.append((rel, 'SyntaxError', '%s (line %s)' % (e.msg, e.lineno)))
            continue
        except (ValueError, RecursionError, MemoryError) as e:
            failures.append((rel, type(e).__name__, str(e)[:200]))
            continue
        ok[rel] = (tree, src, src.count('\n') + 1)
    return ok, failures


# ---------------------------------------------------------- 语法构件标签化

# "核心内核": 任何可用的 PySub 都必须包含的基础构件。用于曲线 B 的起点。
CORE_TAGS = frozenset({
    'stmt.Assign', 'stmt.Return', 'stmt.Expr', 'stmt.If', 'stmt.For', 'stmt.While',
    'stmt.Pass', 'stmt.Break', 'stmt.Continue', 'stmt.AugAssign', 'stmt.AnnAssign',
    'expr.Name', 'expr.Attribute', 'expr.Call', 'expr.Constant', 'expr.BinOp',
    'expr.UnaryOp', 'expr.BoolOp', 'expr.Compare', 'expr.Subscript', 'expr.Slice',
    'expr.Tuple', 'expr.List', 'expr.Dict', 'expr.Set', 'expr.JoinedStr',
    'def.func', 'def.class', 'arg.default', 'arg.kwonly',
})

_DYN_ATTR = {'getattr', 'setattr', 'hasattr', 'delattr'}
_DYN_EXEC = {'exec', 'eval', 'compile', '__import__'}
_INPLACE_METHODS = {'append', 'extend', 'insert', 'update', 'add', 'discard',
                    'pop', 'remove', 'clear', 'setdefault', 'popitem', 'sort'}


def _fmt_target(node):
    """把 Name/Attribute 链渲染成点号限定名; 其他返回 None。"""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return '.'.join(reversed(parts))
    return None


def _attr_chain_len(node):
    n = 0
    cur = node
    while isinstance(cur, ast.Attribute):
        n += 1
        cur = cur.value
    return n if isinstance(cur, ast.Name) else n  # 起点非 Name 也计链长


class ConstructTagger(ast.NodeVisitor):
    """把一棵子树里出现的语法构件转成 (tag -> count)。"""

    def __init__(self):
        self.counts = Counter()
        self.attr_chain_hist = Counter()
        self._in_attr = 0

    def hit(self, tag, n=1):
        self.counts[tag] += n

    # --- 统一入口: 记录原生节点类型 ------------------------------------
    def generic_visit(self, node):
        super().generic_visit(node)

    def visit(self, node):
        self.counts['raw.' + type(node).__name__] += 1
        return super().visit(node)

    # --- 推导式 / 生成器 ------------------------------------------------
    def visit_ListComp(self, node):
        self.hit('comp.list')
        self._comp_extra(node)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.hit('comp.set')
        self._comp_extra(node)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.hit('comp.dict')
        self._comp_extra(node)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.hit('comp.genexp')
        self._comp_extra(node)
        self.generic_visit(node)

    def _comp_extra(self, node):
        if len(node.generators) > 1:
            self.hit('comp.multi_for')
        for g in node.generators:
            if g.ifs:
                self.hit('comp.filter')
            if getattr(g, 'is_async', 0):
                self.hit('comp.async')

    # --- 函数 / 装饰器 / 参数 -------------------------------------------
    def visit_FunctionDef(self, node):
        self._funcdef(node, 'def.func')

    def visit_AsyncFunctionDef(self, node):
        self.hit('def.async_func')
        self._funcdef(node, 'def.func')

    def _funcdef(self, node, tag):
        self.hit(tag)
        for d in node.decorator_list:
            self.hit('decorator.func')
            nm = _fmt_target(d.func if isinstance(d, ast.Call) else d)
            if nm:
                self.hit('decoratorname.' + nm)
            if isinstance(d, ast.Call):
                self.hit('decorator.func_parameterized')
        a = node.args
        if a.vararg is not None:
            self.hit('arg.vararg')
        if a.kwarg is not None:
            self.hit('arg.kwarg')
        if a.defaults or a.kw_defaults:
            self.hit('arg.default')
        if a.kwonlyargs:
            self.hit('arg.kwonly')
        if a.posonlyargs:
            self.hit('arg.posonly')
        if node.name in ('__getattr__', '__getattribute__'):
            self.hit('dyn.def___getattr__')
        if node.name == '__setattr__':
            self.hit('dyn.def___setattr__')
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.hit('def.class')
        for d in node.decorator_list:
            self.hit('decorator.class')
        nbases = len(node.bases)
        if nbases == 0:
            self.hit('class.bases0')
        elif nbases == 1:
            self.hit('class.bases1')
        else:
            self.hit('class.multi_inherit')
        for kw in node.keywords:
            if kw.arg == 'metaclass':
                self.hit('class.metaclass')
            else:
                self.hit('class.kwarg_other')
        for b in node.bases:
            if isinstance(b, ast.Call):
                self.hit('class.dynamic_base')  # 基类由函数调用产生
            if isinstance(b, ast.Starred):
                self.hit('class.starred_base')
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.hit('expr.Lambda')
        self.generic_visit(node)

    def visit_Nonlocal(self, node):
        self.hit('scope.nonlocal')
        self.generic_visit(node)

    def visit_Global(self, node):
        self.hit('scope.global')
        self.generic_visit(node)

    # --- yield / await --------------------------------------------------
    def visit_Yield(self, node):
        self.hit('gen.yield')
        self.generic_visit(node)

    def visit_YieldFrom(self, node):
        self.hit('gen.yield_from')
        self.generic_visit(node)

    def visit_Await(self, node):
        self.hit('async.await')
        self.generic_visit(node)

    # --- try / with / raise ---------------------------------------------
    def visit_Try(self, node):
        self.hit('ctl.try')
        for h in node.handlers:
            self.hit('ctl.except_handler')
            if h.type is None:
                self.hit('ctl.bare_except')
        if node.finalbody:
            self.hit('ctl.finally')
        if node.orelse:
            self.hit('ctl.try_else')
        self.generic_visit(node)

    def visit_TryStar(self, node):
        self.hit('ctl.try_star')
        self.generic_visit(node)

    def visit_Raise(self, node):
        self.hit('ctl.raise')
        self.generic_visit(node)

    def visit_With(self, node):
        self.hit('ctl.with')
        if len(node.items) > 1:
            self.hit('ctl.with_multi')
        self.generic_visit(node)

    def visit_AsyncWith(self, node):
        self.hit('ctl.async_with')
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.hit('ctl.assert')
        self.generic_visit(node)

    def visit_Match(self, node):
        self.hit('ctl.match')
        self.generic_visit(node)

    def visit_Delete(self, node):
        self.hit('ctl.del')
        self.generic_visit(node)

    # --- 解包 / 星号 -----------------------------------------------------
    def visit_Starred(self, node):
        # 区分调用实参解包 vs 赋值目标解包 由父节点决定; 这里只记通用
        self.hit('unpack.starred')
        self.generic_visit(node)

    def visit_Call(self, node):
        fname = _fmt_target(node.func)
        short = fname.rsplit('.', 1)[-1] if fname else None
        if any(isinstance(a, ast.Starred) for a in node.args):
            self.hit('unpack.call_star')
        if any(k.arg is None for k in node.keywords):
            self.hit('unpack.call_dstar')
        if short in _DYN_ATTR:
            self.hit('dyn.' + short)
            # 动态与否: 第二个实参是否常量字符串
            if short in ('getattr', 'setattr', 'hasattr', 'delattr') and len(node.args) >= 2:
                a1 = node.args[1]
                if isinstance(a1, ast.Constant) and isinstance(a1.value, str):
                    self.hit('dyn.%s_conststr' % short)
                else:
                    self.hit('dyn.%s_dynamic' % short)
        if short in _DYN_EXEC:
            self.hit('dyn.' + short)
        if short == 'super':
            self.hit('oop.super_call')
            if not node.args:
                self.hit('oop.super_zeroarg')
        if short in ('type',) and len(node.args) == 3:
            self.hit('dyn.type3_classfactory')
        if short in ('import_module', 'reload'):
            self.hit('dyn.importlib_call')
        if short in ('globals', 'locals', 'vars'):
            self.hit('dyn.' + short)
        if fname and fname.startswith('functools.partial'):
            self.hit('dyn.functools_partial')
        self.generic_visit(node)

    def visit_Dict(self, node):
        if any(k is None for k in node.keys):
            self.hit('unpack.dict_star')
        self.generic_visit(node)

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, (ast.Tuple, ast.List)):
                self.hit('unpack.assign_seq')
                if any(isinstance(e, ast.Starred) for e in t.elts):
                    self.hit('unpack.assign_star')
        if len(node.targets) > 1:
            self.hit('assign.chained')
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.hit('expr.IfExp')
        self.generic_visit(node)

    def visit_NamedExpr(self, node):
        self.hit('expr.walrus')
        self.generic_visit(node)

    def visit_Compare(self, node):
        if len(node.ops) > 1:
            self.hit('expr.chained_compare')
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        self.hit('expr.JoinedStr')
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if self._in_attr == 0:
            self.attr_chain_hist[_attr_chain_len(node)] += 1
        self._in_attr += 1
        self.generic_visit(node)
        self._in_attr -= 1

    # --- import ----------------------------------------------------------
    def visit_Import(self, node):
        self.hit('imp.import')
        for a in node.names:
            if a.name.split('.')[0] == 'importlib':
                self.hit('dyn.importlib_import')
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.hit('imp.from_import')
        if node.level:
            self.hit('imp.relative')
        if any(a.name == '*' for a in node.names):
            self.hit('imp.star')
        if node.module and node.module.split('.')[0] == 'importlib':
            self.hit('dyn.importlib_import')
        self.generic_visit(node)


def _base_tags_from_raw(counts):
    """把部分 raw.* 归入语义 tag(核心内核用)。"""
    m = {
        'raw.Assign': 'stmt.Assign', 'raw.Return': 'stmt.Return', 'raw.Expr': 'stmt.Expr',
        'raw.If': 'stmt.If', 'raw.For': 'stmt.For', 'raw.While': 'stmt.While',
        'raw.Pass': 'stmt.Pass', 'raw.Break': 'stmt.Break', 'raw.Continue': 'stmt.Continue',
        'raw.AugAssign': 'stmt.AugAssign', 'raw.AnnAssign': 'stmt.AnnAssign',
        'raw.Name': 'expr.Name', 'raw.Attribute': 'expr.Attribute', 'raw.Call': 'expr.Call',
        'raw.Constant': 'expr.Constant', 'raw.BinOp': 'expr.BinOp', 'raw.UnaryOp': 'expr.UnaryOp',
        'raw.BoolOp': 'expr.BoolOp', 'raw.Compare': 'expr.Compare', 'raw.Subscript': 'expr.Subscript',
        'raw.Slice': 'expr.Slice', 'raw.Tuple': 'expr.Tuple', 'raw.List': 'expr.List',
        'raw.Dict': 'expr.Dict', 'raw.Set': 'expr.Set',
    }
    out = Counter()
    for k, v in counts.items():
        if k in m:
            out[m[k]] += v
    return out


def tag_subtree(node):
    """返回 (Counter(tag->count), Counter(chain_len->count))，含语义 tag 与派生核心 tag。"""
    t = ConstructTagger()
    for child in ast.iter_child_nodes(node):
        t.visit(child)
    c = Counter(t.counts)
    c.update(_base_tags_from_raw(t.counts))
    return c, t.attr_chain_hist


def repo_construct_stats(parsed):
    """① 全仓语法构件频次。
    返回 {tag: {'count': n, 'files': m}}, attr_chain_hist, raw_node_stats
    """
    agg = Counter()
    filecnt = Counter()
    chain = Counter()
    for rel, (tree, src, nl) in parsed.items():
        c, ch = tag_subtree(tree)
        agg.update(c)
        for k in c:
            filecnt[k] += 1
        chain.update(ch)
    return {k: {'count': agg[k], 'files': filecnt[k]} for k in agg}, chain


# ------------------------------------------------- 函数级 profile 与 PySub 曲线

def _iter_scopes(tree, rel):
    """产出 (qualname, node, kind)。kind in {'func','module'}。
    module 伪作用域 = 文件顶层语句(不含 def/class 体)。
    """
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = prefix + '.' + child.name if prefix else child.name
                out.append((qn, child, 'func'))
                walk(child, qn)
            elif isinstance(child, ast.ClassDef):
                qn = prefix + '.' + child.name if prefix else child.name
                walk(child, qn)
            else:
                walk(child, prefix)

    walk(tree, '')
    # module 伪作用域
    mod = ast.Module(body=[s for s in tree.body
                           if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))],
                     type_ignores=[])
    out.append(('<module>', mod, 'module'))
    return [(rel + '::' + qn, n, k) for qn, n, k in out]


def function_profiles(parsed, include_module_scope=True):
    """每个函数(及 <module> 伪作用域)用到的 tag 集合。
    注意: 嵌套函数的 tag 同时计入外层函数(PE 内联时确实要处理)。
    返回 [(qualname, kind, frozenset(tags))]
    """
    profs = []
    for rel, (tree, src, nl) in parsed.items():
        for qn, node, kind in _iter_scopes(tree, rel):
            if kind == 'module' and not include_module_scope:
                continue
            c, _ = tag_subtree(node)
            tags = frozenset(k for k in c
                             if not k.startswith('raw.') and not k.startswith('decoratorname.'))
            if kind == 'func':
                # 函数自身的装饰器/参数特征来自其定义节点本身
                own, _ = tag_subtree(ast.Module(body=[node], type_ignores=[]))
                tags = tags | frozenset(k for k in own
                                        if not k.startswith('raw.')
                                        and not k.startswith('decoratorname.'))
            profs.append((qn, kind, tags))
    return profs


def pysub_curve(profiles, seed_tags=frozenset(), max_steps=None):
    """② PySub 覆盖率曲线。
    贪心: 每步加入"能新增覆盖最多作用域"的 tag。
    返回 [(step, added_tag, covered, total, frac)]
    """
    masks = {}
    order = []

    def bit(t):
        if t not in masks:
            masks[t] = 1 << len(order)
            order.append(t)
        return masks[t]

    profile_masks = Counter()
    for qn, kind, tags in profiles:
        m = 0
        for t in tags:
            m |= bit(t)
        profile_masks[m] += 1

    total = sum(profile_masks.values())
    supported = 0
    for t in seed_tags:
        if t in masks:
            supported |= masks[t]

    remaining = [t for t in order if t not in seed_tags]
    curve = []
    step = 0
    covered = sum(n for m, n in profile_masks.items() if (m & ~supported) == 0)
    curve.append((0, '<seed:%d tags>' % len(seed_tags & set(order)), covered, total,
                  covered / total if total else 0.0))
    live = {m: n for m, n in profile_masks.items() if (m & ~supported) != 0}
    while remaining and (max_steps is None or step < max_steps):
        best, best_gain = None, -1
        for t in remaining:
            cand = supported | masks[t]
            g = 0
            for m, n in live.items():
                if (m & ~cand) == 0:
                    g += n
            if g > best_gain:
                best, best_gain = t, g
        step += 1
        supported |= masks[best]
        remaining.remove(best)
        covered += best_gain
        live = {m: n for m, n in live.items() if (m & ~supported) != 0}
        curve.append((step, best, covered, total, covered / total if total else 0.0))
        if covered >= total:
            break
    return curve


def blocker_ranking(profiles, seed_tags):
    """若 PySub 不支持某 tag，会阻断多少个作用域(独占阻断/参与阻断)。"""
    total = len(profiles)
    involved = Counter()      # 用到该 tag 的作用域数
    sole = Counter()          # 该 tag 是该作用域唯一的超纲构件
    for qn, kind, tags in profiles:
        extra = tags - seed_tags
        for t in extra:
            involved[t] += 1
        if len(extra) == 1:
            sole[next(iter(extra))] += 1
    return involved, sole, total


# ------------------------------------------------ ③ import 期 monkey-patch

def _module_level_stmts(tree):
    """import 期真正会执行的顶层语句(穿透 if/try/with/for，不进 def/class)。"""
    out = []

    def walk(body, depth):
        for s in body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            out.append((s, depth))
            for fld in ('body', 'orelse', 'finalbody'):
                sub = getattr(s, fld, None)
                if isinstance(sub, list):
                    walk(sub, depth + 1)
            if isinstance(s, ast.Try):
                for h in s.handlers:
                    walk(h.body, depth + 1)
    walk(tree.body, 0)
    return out


def module_imported_names(tree):
    """顶层 import 引入的名字 -> ('module'|'name', 源)。"""
    names = {}
    for s, _ in _module_level_stmts(tree):
        if isinstance(s, ast.Import):
            for a in s.names:
                bind = a.asname or a.name.split('.')[0]
                names[bind] = ('module', a.name)
        elif isinstance(s, ast.ImportFrom):
            mod = ('.' * (s.level or 0)) + (s.module or '')
            for a in s.names:
                if a.name == '*':
                    continue
                bind = a.asname or a.name
                names[bind] = ('name', mod + '.' + a.name)
    return names


def find_monkey_patches(parsed):
    """③ 模块顶层对已 import 模块/名字的属性赋值。
    返回 [(rel, lineno, target_qualname, root, root_kind, origin, form, guarded)]
    """
    hits = []
    for rel, (tree, src, nl) in parsed.items():
        imported = module_imported_names(tree)
        toplevel_assigned = set()
        for s, _ in _module_level_stmts(tree):
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    if isinstance(t, ast.Name):
                        toplevel_assigned.add(t.id)
        for s, depth in _module_level_stmts(tree):
            targets = []
            form = None
            if isinstance(s, ast.Assign):
                targets = s.targets
                form = 'assign'
            elif isinstance(s, ast.AnnAssign) and s.value is not None:
                targets = [s.target]
                form = 'annassign'
            elif isinstance(s, ast.AugAssign):
                targets = [s.target]
                form = 'augassign'
            elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
                f = _fmt_target(s.value.func)
                if f and f.rsplit('.', 1)[-1] in ('setattr', 'delattr'):
                    a = s.value.args
                    root = _fmt_target(a[0]) if a else None
                    attr = None
                    if len(a) >= 2 and isinstance(a[1], ast.Constant):
                        attr = a[1].value
                    qn = '%s.%s' % (root, attr if attr else '<dynamic>')
                    r0 = root.split('.')[0] if root else None
                    kind, origin = imported.get(r0, (None, None))
                    hits.append((rel, s.lineno, qn, r0, kind or ('local' if r0 in toplevel_assigned else 'unknown'),
                                 origin, 'setattr_call', depth > 0))
                    continue
                # 常见的 patch 入口函数调用
                if f and re.search(r'(patch|apply|register|monkey|replace|override|inject)',
                                   f.rsplit('.', 1)[-1], re.I):
                    hits.append((rel, s.lineno, f + '()', f.split('.')[0],
                                 imported.get(f.split('.')[0], ('unknown', None))[0],
                                 None, 'patchfn_call', depth > 0))
                continue
            for t in targets:
                if not isinstance(t, ast.Attribute):
                    continue
                qn = _fmt_target(t)
                if qn is None:
                    continue
                r0 = qn.split('.')[0]
                kind, origin = imported.get(r0, (None, None))
                if kind is None:
                    kind = 'local_toplevel' if r0 in toplevel_assigned else 'unknown'
                hits.append((rel, t.lineno, qn, r0, kind, origin, form, depth > 0))
    return hits


# ------------------------------- ④ 并行状态访问器 & 模块级可变全局

ACCESSOR_PATTERNS = [
    re.compile(r'^get_.*_(world_size|rank|group|size|ranks|src_rank|global_ranks)$'),
    re.compile(r'^get_.*parallel.*$'),
    re.compile(r'^is_.*(first|last)_.*$'),
    re.compile(r'^(get|is)_.*(pipeline|tensor|expert|context|sequence|data)_.*$'),
    re.compile(r'^get_(world_size|rank|group|data_parallel|model_parallel).*$'),
    re.compile(r'^is_(initialized|rank_in|pipeline|inside).*$'),
]


def _is_accessor_name(name):
    return any(p.match(name) for p in ACCESSOR_PATTERNS)


def module_level_bindings(tree):
    """顶层绑定的名字 -> 出现行号列表(仅 Name 目标)。"""
    binds = defaultdict(list)
    for s, _ in _module_level_stmts(tree):
        if isinstance(s, ast.Assign):
            for t in s.targets:
                if isinstance(t, ast.Name):
                    binds[t.id].append(t.lineno)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for e in t.elts:
                        if isinstance(e, ast.Name):
                            binds[e.id].append(e.lineno)
        elif isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
            binds[s.target.id].append(s.lineno)
        elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
            binds[s.target.id].append(s.lineno)
        elif isinstance(s, ast.For) and isinstance(s.target, ast.Name):
            binds[s.target.id].append(s.lineno)
    return binds


def _is_final_annotated(tree, name):
    for s, _ in _module_level_stmts(tree):
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id == name:
            a = _fmt_target(s.annotation) or ''
            if 'Final' in a:
                return True
    return False


def analyze_globals(parsed):
    """④ 模块级可变全局。
    返回 per-file dict: {rel: {name: {...evidence...}}}
    """
    res = {}
    for rel, (tree, src, nl) in parsed.items():
        binds = module_level_bindings(tree)
        if not binds:
            res[rel] = {}
            continue
        # 证据收集: global 声明 / 下标赋值 / 属性赋值 / 就地方法
        global_decl = Counter()
        global_rebind = Counter()      # global X 且函数内对 X 赋值
        subs_assign = Counter()
        attr_assign = Counter()
        inplace = Counter()

        class Ev(ast.NodeVisitor):
            def __init__(self):
                self.gscope = []

            def visit_FunctionDef(self, node):
                self.gscope.append(set())
                self.generic_visit(node)
                self.gscope.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Global(self, node):
                for n in node.names:
                    global_decl[n] += 1
                    if self.gscope:
                        self.gscope[-1].add(n)
                self.generic_visit(node)

            def _mark_target(self, t):
                if isinstance(t, ast.Name):
                    if self.gscope and t.id in self.gscope[-1]:
                        global_rebind[t.id] += 1
                elif isinstance(t, ast.Subscript):
                    r = _fmt_target(t.value)
                    if r:
                        subs_assign[r.split('.')[0]] += 1
                elif isinstance(t, ast.Attribute):
                    r = _fmt_target(t)
                    if r:
                        attr_assign[r.split('.')[0]] += 1
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for e in t.elts:
                        self._mark_target(e)

            def visit_Assign(self, node):
                for t in node.targets:
                    self._mark_target(t)
                self.generic_visit(node)

            def visit_AugAssign(self, node):
                self._mark_target(node.target)
                self.generic_visit(node)

            def visit_AnnAssign(self, node):
                self._mark_target(node.target)
                self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr in _INPLACE_METHODS:
                    r = _fmt_target(node.func.value)
                    if r:
                        inplace[r.split('.')[0]] += 1
                self.generic_visit(node)

        Ev().visit(tree)

        d = {}
        for name, lines in binds.items():
            if name.startswith('__') and name.endswith('__'):
                continue
            ev = {
                'lines': lines,
                'n_toplevel_binds': len(lines),
                'global_decl': global_decl.get(name, 0),
                'global_rebind': global_rebind.get(name, 0),
                'subscript_assign': subs_assign.get(name, 0),
                'attr_assign': attr_assign.get(name, 0),
                'inplace_method': inplace.get(name, 0),
                'is_final': _is_final_annotated(tree, name),
                'allcaps': bool(re.fullmatch(r'_*[A-Z0-9_]+', name)),
            }
            ev['mutable'] = bool(
                ev['global_rebind'] or ev['subscript_assign'] or ev['attr_assign']
                or ev['inplace_method'] or ev['n_toplevel_binds'] > 1
            )
            ev['mutable_strong'] = bool(ev['global_rebind'])
            d[name] = ev
        res[rel] = d
    return res


def find_accessors(parsed, globals_by_file):
    """④ 访问器函数定义 + 它读到的模块级全局。"""
    out = []
    for rel, (tree, src, nl) in parsed.items():
        modglobals = set(globals_by_file.get(rel, {}).keys())
        mutable = {k for k, v in globals_by_file.get(rel, {}).items() if v['mutable']}

        def walk(node, prefix):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qn = (prefix + '.' + child.name) if prefix else child.name
                    if _is_accessor_name(child.name):
                        reads = set()
                        gdecl = set()
                        for n in ast.walk(child):
                            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                                if n.id in modglobals:
                                    reads.add(n.id)
                            if isinstance(n, ast.Global):
                                gdecl.update(n.names)
                        callees = set()
                        for n in ast.walk(child):
                            if isinstance(n, ast.Call):
                                f = _fmt_target(n.func)
                                if f:
                                    callees.add(f)
                        out.append({
                            'file': rel, 'qualname': rel + '::' + qn, 'name': child.name,
                            'lineno': child.lineno,
                            'reads_module_globals': sorted(reads),
                            'reads_mutable': sorted(reads & mutable),
                            'global_decls': sorted(gdecl),
                            'n_callees': len(callees),
                            'callees': sorted(callees)[:12],
                            'nlines': (child.end_lineno or child.lineno) - child.lineno + 1,
                        })
                    walk(child, qn)
                elif isinstance(child, ast.ClassDef):
                    qn = (prefix + '.' + child.name) if prefix else child.name
                    walk(child, qn)
                else:
                    walk(child, prefix)

        walk(tree, '')
    return out


def find_accessor_callsites(parsed, accessor_names):
    """访问器在代码里的调用点计数。"""
    cnt = Counter()
    perfile = defaultdict(Counter)
    for rel, (tree, src, nl) in parsed.items():
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = _fmt_target(n.func)
                if not f:
                    continue
                short = f.rsplit('.', 1)[-1]
                if short in accessor_names:
                    cnt[short] += 1
                    perfile[rel][short] += 1
    return cnt, perfile


# --------------------------------------- ⑤ policy-tainted guard

POLICY_GROUPS = {
    '并行度': [
        r'tensor_model_parallel', r'pipeline_model_parallel', r'expert_model_parallel',
        r'expert_tensor_parallel', r'context_parallel', r'sequence_parallel',
        r'data_parallel', r'model_parallel', r'hierarchical_context_parallel',
        r'^tp$', r'^pp$', r'^ep$', r'^cp$', r'^dp$', r'^etp$',
        r'tp_size', r'pp_size', r'ep_size', r'cp_size', r'dp_size',
        r'world_size', r'tp_group', r'^tp_comm', r'moe_extended_tp',
    ],
    '重算': [r'recompute', r'checkpoint', r'gradient_checkpointing', r'activation_offload',
             r'cpu_offloading', r'selective_recompute'],
    '融合/实现选择': [r'fusion', r'^fused_', r'_fused$', r'use_flash', r'flash_attn',
                r'attention_backend', r'_impl$', r'_impl_', r'grouped_', r'te_version',
                r'^use_te', r'transformer_impl', r'kernel', r'backend'],
    '精度': [r'^fp8', r'fp8_', r'bf16', r'fp16', r'dtype', r'master_weights',
             r'main_grad', r'params_dtype', r'autocast', r'quant'],
    '分布式优化器': [r'zero_stage', r'use_distributed_optimizer', r'overlap_', r'bucket',
                r'ddp_config', r'grad_reduce', r'fsdp', r'reduce_scatter', r'all_gather'],
    '调度': [r'virtual_pipeline', r'num_microbatches', r'microbatch', r'interleav',
             r'pipeline_dtype', r'first_stage', r'last_stage', r'is_pipeline'],
    'MoE': [r'^moe_', r'_moe$', r'num_experts', r'expert_', r'router', r'token_dispatcher',
            r'^n_routed', r'shared_expert'],
}

_POLICY_RE = {g: re.compile('|'.join(pats)) for g, pats in POLICY_GROUPS.items()}


def _test_identifiers(test):
    """测试表达式里出现的所有标识符(Name.id / Attribute.attr / 常量字符串)。"""
    ids = []
    for n in ast.walk(test):
        if isinstance(n, ast.Name):
            ids.append(n.id)
        elif isinstance(n, ast.Attribute):
            ids.append(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 60:
            ids.append(n.value)
    return ids


def _match_policy(ids):
    groups = set()
    fields = set()
    for i in ids:
        for g, rx in _POLICY_RE.items():
            if rx.search(i):
                groups.add(g)
                fields.add(i)
    return groups, fields


def _calls_in(stmts):
    c = Counter()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Call):
                f = _fmt_target(n.func)
                c[f or '<expr>()'] += 1
    return c


def _branch_kind(body):
    if not body:
        return 'empty'
    if all(isinstance(s, ast.Pass) for s in body):
        return 'pass_only'
    if all(isinstance(s, (ast.Return, ast.Raise, ast.Continue, ast.Break, ast.Pass))
           for s in body):
        return 'early_exit'
    return 'nonempty'


def find_policy_guards(parsed):
    """⑤ 在给定文件集合上找 policy-tainted if / IfExp。"""
    guards = []
    for rel, (tree, src, nl) in parsed.items():
        # 计算每个 If 的嵌套深度(仅统计 if/ifexp 嵌套) 与 policy-guard 嵌套深度
        def walk(node, if_depth, pol_depth, fnqn):
            for child in ast.iter_child_nodes(node):
                nf = fnqn
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nf = (fnqn + '.' + child.name) if fnqn else child.name
                if isinstance(child, ast.If):
                    ids = _test_identifiers(child.test)
                    groups, fields = _match_policy(ids)
                    is_pol = bool(groups)
                    if is_pol:
                        body_calls = _calls_in(child.body)
                        else_calls = _calls_in(child.orelse)
                        bk = _branch_kind(child.body)
                        ek = _branch_kind(child.orelse)
                        only_assign = all(
                            isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Pass))
                            for s in list(child.body) + list(child.orelse))
                        callset_delta = (set(body_calls) != set(else_calls))
                        if ek in ('empty',):
                            struct = 'single_sided'
                        elif bk in ('early_exit', 'pass_only') or ek in ('early_exit', 'pass_only'):
                            struct = 'one_side_trivial'
                        else:
                            struct = 'both_nonempty'
                        if only_assign and not callset_delta:
                            cls = 'assign_only'
                        elif only_assign:
                            cls = 'assign_only_with_call_delta'
                        elif callset_delta:
                            cls = 'call_set_changes'
                        else:
                            cls = 'same_calls_diff_args'
                        guards.append({
                            'file': rel, 'lineno': child.lineno, 'node': 'If',
                            'func': nf, 'groups': sorted(groups), 'fields': sorted(fields)[:6],
                            'struct': struct, 'cls': cls,
                            'body_kind': bk, 'else_kind': ek,
                            'n_body_stmts': len(child.body), 'n_else_stmts': len(child.orelse),
                            'body_calls': len(body_calls), 'else_calls': len(else_calls),
                            'added_calls': sorted(set(body_calls) - set(else_calls))[:5],
                            'removed_calls': sorted(set(else_calls) - set(body_calls))[:5],
                            'if_depth': if_depth, 'pol_depth': pol_depth,
                            'test_src': _safe_unparse(child.test),
                        })
                    walk(child, if_depth + 1, pol_depth + (1 if is_pol else 0), nf)
                elif isinstance(child, ast.IfExp):
                    ids = _test_identifiers(child.test)
                    groups, fields = _match_policy(ids)
                    if groups:
                        bc = _calls_in([ast.Expr(value=child.body)])
                        ec = _calls_in([ast.Expr(value=child.orelse)])
                        guards.append({
                            'file': rel, 'lineno': getattr(child, 'lineno', 0), 'node': 'IfExp',
                            'func': nf, 'groups': sorted(groups), 'fields': sorted(fields)[:6],
                            'struct': 'ifexp',
                            'cls': 'call_set_changes' if set(bc) != set(ec) else 'assign_only',
                            'body_kind': 'expr', 'else_kind': 'expr',
                            'n_body_stmts': 1, 'n_else_stmts': 1,
                            'body_calls': len(bc), 'else_calls': len(ec),
                            'added_calls': sorted(set(bc) - set(ec))[:5],
                            'removed_calls': sorted(set(ec) - set(bc))[:5],
                            'if_depth': if_depth, 'pol_depth': pol_depth,
                            'test_src': _safe_unparse(child.test),
                        })
                    walk(child, if_depth, pol_depth, nf)
                else:
                    walk(child, if_depth, pol_depth, nf)

        walk(tree, 0, 0, '')
    return guards


def _safe_unparse(node):
    try:
        s = ast.unparse(node)
    except Exception:  # noqa: BLE001
        return '<unparse-failed>'
    return s if len(s) <= 140 else s[:137] + '...'


def count_all_ifs(parsed):
    n_if = 0
    n_ifexp = 0
    for rel, (tree, src, nl) in parsed.items():
        for x in ast.walk(tree):
            if isinstance(x, ast.If):
                n_if += 1
            elif isinstance(x, ast.IfExp):
                n_ifexp += 1
    return n_if, n_ifexp


# --------------------------------------------- 规模 & 类继承深度

def repo_scale(parsed):
    nfiles = len(parsed)
    nlines = sum(v[2] for v in parsed.values())
    nfuncs = 0
    nclasses = 0
    for rel, (tree, src, nl) in parsed.items():
        for x in ast.walk(tree):
            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nfuncs += 1
            elif isinstance(x, ast.ClassDef):
                nclasses += 1
    return {'files': nfiles, 'lines': nlines, 'functions': nfuncs, 'classes': nclasses}


def class_hierarchy(parsed):
    """按类名近似的仓内继承深度(跨文件同名类会合并, 已知近似)。"""
    bases = {}
    for rel, (tree, src, nl) in parsed.items():
        for x in ast.walk(tree):
            if isinstance(x, ast.ClassDef):
                bl = []
                for b in x.bases:
                    f = _fmt_target(b)
                    if f:
                        bl.append(f.rsplit('.', 1)[-1])
                bases.setdefault(x.name, set()).update(bl)
    memo = {}

    def depth(n, seen):
        if n in memo:
            return memo[n]
        if n in seen or n not in bases:
            return 0
        seen = seen | {n}
        d = 0
        for b in bases[n]:
            d = max(d, depth(b, seen) + 1)
        memo[n] = d
        return d

    hist = Counter()
    for n in bases:
        hist[depth(n, frozenset())] += 1
    return hist, len(bases)


def import_closure(parsed, entry_rels, pkg_prefixes):
    """从入口文件出发的仓内 import 闭包(模块级 import, 含 def 内 import)。"""
    rel_by_mod = {}
    for rel in parsed:
        mod = rel[:-3].replace('/', '.')
        if mod.endswith('.__init__'):
            mod = mod[:-len('.__init__')]
        rel_by_mod[mod] = rel

    def resolve(rel, node):
        out = []
        cur_mod = rel[:-3].replace('/', '.')
        cur_pkg = cur_mod.rsplit('.', 1)[0] if '.' in cur_mod else ''
        if cur_mod.endswith('.__init__'):
            cur_pkg = cur_mod[:-len('.__init__')]
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = cur_pkg.split('.')
                base = '.'.join(parts[:len(parts) - (node.level - 1)]) if node.level > 1 else cur_pkg
                m = base + ('.' + node.module if node.module else '')
            else:
                m = node.module or ''
            out.append(m)
            for a in node.names:
                if a.name != '*':
                    out.append(m + '.' + a.name)
        return out

    seen = set()
    frontier = [r for r in entry_rels if r in parsed]
    seen.update(frontier)
    while frontier:
        nxt = []
        for rel in frontier:
            tree = parsed[rel][0]
            for n in ast.walk(tree):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    for m in resolve(rel, n):
                        if not any(m.startswith(p) for p in pkg_prefixes):
                            continue
                        cand = rel_by_mod.get(m)
                        if cand is None:
                            cand = rel_by_mod.get(m.rsplit('.', 1)[0]) if '.' in m else None
                        if cand and cand not in seen:
                            seen.add(cand)
                            nxt.append(cand)
        frontier = nxt
    return seen
