# cost_eval/opdag/extractor.py
"""端到端抽取器(设计 §Task5):读一个**真** mindformers Cell 源文件,组合 Pass B(init 绑定)
+ 具名模块解析(build_module / get_activation)+ Pass C(construct 走查,按 config 剪枝),
产出该 Cell 的 op-DAG。

组合关系:
  1. `bind_init(src, <定义 __init__ 的类>)` → 基础绑定(逐元素 / view / cast 类)。
  2. 解析 __init__ 的具名模块赋值,扩展绑定:
       * `self.<X> = build_module(submodules.<Y>, ...)`:
           - spec.submodules[<Y>] 是叶子类名(str) → op 类型 = LEAF_OPTYPE[叶子];
           - spec.submodules[<Y>] 是 ResolvedSpec(子 Cell)→ op="SubCell"(attrs.cell=子 Cell 名),
             为后续任务的递归展开预留挂钩(本任务 MLP 的 fc1/fc2 走叶子 MatMul 路径)。
       * `self.<X> = get_activation(...)` → Binding(op="Activation")。
       * 解不出的具名模块赋值 → fail-loud。
  3. 从 config_flags / spec 推 walker 的 none_vars / param_defaults / config_flags:
       * add_bias_linear=False → linear 调用元组目标的第二位(bias 变量)记为 None(启发式,见 _bias_none_vars);
       * construct 形参缺省即 None → param_defaults(顶部 `if x is not None: raise` 守卫得以剪掉);
       * activation_func 是否为 None 由 activation_type 决定,并注入 walker config_flags(供三元判定)。
  4. `walk_construct(..., config_flags, none_vars, param_defaults)` → 剪枝后的 op-DAG。

**绝不** import / 执行 mindformers / mindspore —— 只 `ast` + stdlib 静态读源。
"""
from __future__ import annotations

import ast
import os

from .schema import OpDAG
from .init_binder import bind_init, Binding
from .construct_walker import walk_construct
from .module_resolver import ResolvedSpec, LEAF_OPTYPE


# ── AST 小工具 ────────────────────────────────────────────────────────────────
def _find_class(tree: ast.AST, cls_name: str) -> ast.ClassDef | None:
    return next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls_name),
        None,
    )


def _method_of(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    return next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name),
        None,
    )


def _defining_class(tree: ast.AST, cls_name: str, method: str) -> str | None:
    """返回**实际定义** `method` 的类名:从 cls_name 起,沿其(同文件内可见的)基类做 BFS。

    用于处理继承——如 MLPInterleaved 自身只重写 construct,__init__ 继承自基类 MLP。
    """
    seen: set[str] = set()
    queue = [cls_name]
    while queue:
        cname = queue.pop(0)
        if cname in seen:
            continue
        seen.add(cname)
        cls = _find_class(tree, cname)
        if cls is None:
            continue
        if _method_of(cls, method) is not None:
            return cname
        for base in cls.bases:
            if isinstance(base, ast.Name):
                queue.append(base.id)
    return None


# ── __init__ 里的具名模块绑定 ─────────────────────────────────────────────────
def _named_module_binds(
    tree: ast.AST, init_cls: str, spec: ResolvedSpec, config_flags: dict, src_file: str
) -> dict[str, Binding]:
    cls = _find_class(tree, init_cls)
    init = _method_of(cls, "__init__")
    if init is None:  # 理论上 init_cls 已保证有 __init__
        raise ValueError(f"extractor: {init_cls} 无 __init__（fail-loud）")

    compute_dtype = config_flags.get("compute_dtype", "bf16")
    # activation 是否存在:未显式给 activation_type 视作存在;显式给 None → 不绑定(activation_func=None)。
    act_type_given = "activation_type" in config_flags
    act_type = config_flags.get("activation_type")
    activation_present = (not act_type_given) or (act_type is not None)

    binds: dict[str, Binding] = {}
    for stmt in ast.walk(init):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        name = tgt.attr
        fn = stmt.value.func
        fname = fn.id if isinstance(fn, ast.Name) else None

        if fname == "build_module":
            binds[name] = _bind_build_module(stmt.value, name, spec, compute_dtype, src_file)
        elif fname == "get_activation":
            if activation_present:
                binds[name] = Binding(
                    op="Activation",
                    attrs={"activation_type": act_type if act_type_given else None},
                )
            # activation_type 显式为 None → 不绑定(下游按 None 剪枝)
    return binds


def _bind_build_module(
    call: ast.Call, self_name: str, spec: ResolvedSpec, compute_dtype: str, src_file: str
) -> Binding:
    if not call.args:
        raise ValueError(
            f"extractor: {src_file} self.{self_name} = build_module(...) 缺首个 submodules 实参（fail-loud）"
        )
    a0 = call.args[0]
    if not (isinstance(a0, ast.Attribute) and isinstance(a0.value, ast.Name)):
        raise ValueError(
            f"extractor: {src_file} self.{self_name} 的 build_module 首参不是 submodules.<字段>（fail-loud）"
        )
    field = a0.attr
    leaf = spec.submodules.get(field)
    if leaf is None:
        raise ValueError(
            f"extractor: build_module(submodules.{field}) 在 spec 里无对应子模块"
            f"（self.{self_name} @ {src_file}）—— fail-loud"
        )
    if isinstance(leaf, ResolvedSpec):
        # 子 Cell:预留递归展开挂钩(本任务不触发)。
        return Binding(op="SubCell", attrs={"cell": leaf.cell, "field": field})
    if isinstance(leaf, str):
        op = LEAF_OPTYPE.get(leaf)
        if op is None:
            raise ValueError(
                f"extractor: 叶子类 {leaf!r} 不在 LEAF_OPTYPE（self.{self_name} @ {src_file}）—— fail-loud"
            )
        return Binding(op=op, attrs={"module": leaf, "compute_dtype": compute_dtype})
    raise ValueError(
        f"extractor: submodules.{field} 解出非法类型 {leaf!r}（self.{self_name}）—— fail-loud"
    )


# ── none_vars / param_defaults 推断 ───────────────────────────────────────────
def _bias_none_vars(tree: ast.AST, cls_name: str, binds: dict) -> set[str]:
    """启发式:construct 里 `<out>, <bias> = self.<linear>(...)`(<linear> 绑定为 MatMul)时,
    add_bias_linear=False ⇒ 该元组第二位(bias 变量)恒为 None。收集这些 bias 变量名。"""
    dcls = _defining_class(tree, cls_name, "construct")
    cls = _find_class(tree, dcls) if dcls else None
    construct = _method_of(cls, "construct") if cls else None
    out: set[str] = set()
    if construct is None:
        return out
    for stmt in ast.walk(construct):
        if not (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)):
            continue
        f = stmt.value.func
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == "self"):
            continue
        b = binds.get(f.attr)
        if b is None or b.op != "MatMul":
            continue
        for tgt in stmt.targets:
            if (isinstance(tgt, (ast.Tuple, ast.List)) and len(tgt.elts) >= 2
                    and isinstance(tgt.elts[1], ast.Name)):
                out.add(tgt.elts[1].id)
    return out


def _construct_none_defaults(tree: ast.AST, cls_name: str) -> dict:
    """construct 形参中缺省值恰为 None 的 → {形参名: None}(供顶部 `if x is not None: raise` 守卫剪枝)。"""
    dcls = _defining_class(tree, cls_name, "construct")
    cls = _find_class(tree, dcls) if dcls else None
    construct = _method_of(cls, "construct") if cls else None
    out: dict = {}
    if construct is None:
        return out
    args = construct.args.args
    defaults = construct.args.defaults
    n, nd = len(args), len(defaults)
    for i, a in enumerate(args):
        j = i - (n - nd)
        if j >= 0:
            d = defaults[j]
            if isinstance(d, ast.Constant) and d.value is None:
                out[a.arg] = None
    return out


# ── 对外入口 ──────────────────────────────────────────────────────────────────
def extract_cell(
    mf_root: str,
    cell_file_relpath: str,
    cls_name: str,
    spec: ResolvedSpec,
    config_flags: dict,
) -> OpDAG:
    """读真 mindformers Cell 源,按 config 剪枝,产出其 op-DAG。

    参数
      mf_root           — mindformers 源根(仅 ast 读,绝不 import)。
      cell_file_relpath — Cell 源文件相对 mf_root 的路径(用 "/" 分隔)。
      cls_name          — 目标 Cell 类名(其 construct 被走查;__init__ 可继承自基类)。
      spec              — 该 Cell 的 ResolvedSpec(submodules 字段 → 叶子类名 / 子 ResolvedSpec)。
      config_flags      — config 派生的 flags(gated_linear_unit / activation_type / add_bias_linear / ...)。
    行为
      找不到源 / 无法解析具名模块 / 剪枝时遇不可判定 if → fail-loud(ValueError)。
    """
    path = os.path.join(mf_root, *cell_file_relpath.split("/"))
    if not os.path.isfile(path):
        raise ValueError(f"extractor: 找不到 Cell 源 {path}（fail-loud）")
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    src_file = os.path.basename(path)

    if _find_class(tree, cls_name) is None:
        raise ValueError(f"extractor: {src_file} 里找不到 class {cls_name}（fail-loud）")

    # 1) 找到实际定义 __init__ 的类(可能是基类),做基础 + 具名绑定。
    init_cls = _defining_class(tree, cls_name, "__init__")
    if init_cls is None:
        raise ValueError(f"extractor: {cls_name} 及其基类均无 __init__（fail-loud）")
    base_binds = bind_init(src, init_cls)
    named = _named_module_binds(tree, init_cls, spec, config_flags, src_file)
    combined = {**base_binds, **named}

    # 2) walker 的 config_flags:透传 + 注入 activation_func 真值(由 activation_type 决定其是否为 None)。
    walker_flags = dict(config_flags)
    act_type_given = "activation_type" in config_flags
    activation_present = (not act_type_given) or (config_flags.get("activation_type") is not None)
    for name, b in named.items():
        if b.op == "Activation":
            walker_flags[name] = True  # 已绑定 → 该模块非 None
    if not activation_present:
        # activation_type 为 None:construct 里 `self.activation_func` 恒 falsy。
        walker_flags["activation_func"] = False

    # 3) none_vars(bias 关闭)/ param_defaults(缺省即 None 的形参)。
    if config_flags.get("add_bias_linear", False):
        none_vars: set[str] = set()
    else:
        none_vars = _bias_none_vars(tree, cls_name, combined)
    param_defaults = _construct_none_defaults(tree, cls_name)

    # 4) 走查 + 剪枝。
    return walk_construct(
        src,
        cls_name,
        combined,
        src_file,
        config_flags=walker_flags,
        none_vars=none_vars,
        param_defaults=param_defaults,
    )
