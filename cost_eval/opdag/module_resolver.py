# cost_eval/opdag/module_resolver.py
"""Pass A(设计 §3.1,风险 R1):**静态解释** mindformers 的 spec 构造函数,把 config flags
解成"本次真机实际实例化的模块树"。

为什么是解释而不是硬编码:`get_gpt_layer_local_spec(...)` 是**普通静态 Python**——它按 config
开关走 `if`/三元,`return ModuleSpec(module=Cell, submodules=SubmodulesCls(field=LeafClass|子ModuleSpec))`。
若 mindformers 改了接线,重新 parse 就能跟上。所以我们**读真源、parse AST、按绑定的参数值求值**:

  * 绑参:函数签名 param ← flags(缺的用函数自带默认);
  * 剪 if:`if <param(可判定)>:` 只走命中支;`X if cond else Y` 同理;undecidable → fail-loud;
  * 建 ModuleSpec:`ModuleSpec(module=Cls, submodules=Sub(field=...))` → `ResolvedSpec(cell, {field: ...})`;
  * 展开 helper:`get_mlp_module_spec` / `get_attention_module_spec` / `get_moe_module_spec`(在
    `moe_module_specs.py`)当函数调用递归求值;
  * 叶子:裸类名 → 类名字符串;`get_norm_cls(...)` → "Norm";`IdentityOp` → "Identity"。

**绝不** import / 执行 mindspore / mindformers——只用 `ast` + stdlib。任何**解不出**(if 引用未绑定名 /
非 config 表达式)、**命中 raise**(如 `qk_l2_norm` 未实现)、**未知构造** → 抛 `ValueError`,点名
`file:line` 与解不出的东西,决不静默返回半棵树。
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Union


# ── 对外数据结构 ──────────────────────────────────────────────────────────────
@dataclass
class ResolvedSpec:
    """一个被实例化的 Cell:cell=类名;submodules[field] = 叶子类名(str)或嵌套 ResolvedSpec。"""
    cell: str
    submodules: dict = field(default_factory=dict)  # dict[str, Union["ResolvedSpec", str]]


# 叶子类名 → 规范 op 类型(供后续 Pass 用;复合 Cell 不在此表——它们被递归展开)。
LEAF_OPTYPE = {
    "ColumnParallelLinear": "MatMul",
    "RowParallelLinear": "MatMul",
    "SequenceParallelLinear": "MatMul",
    "FlashAttention": "FlashAttention",
    "Norm": "Norm",
    "Identity": "Identity",
}

# 叶子类名归一(源里的类名 → 我们的规范叶子名)。
_NAME_ALIAS = {
    "IdentityOp": "Identity",
}

# spec 构造函数所在文件(相对 mf_root),按序解析并登记其顶层函数。
_SPEC_FILES = (
    "parallel_core/training_graph/base_models/gpt/gpt_layer_specs.py",
    "parallel_core/training_graph/base_models/gpt/moe_module_specs.py",
)

# 入口函数名。
_ENTRY = "get_gpt_layer_local_spec"


# ── 内部标记 ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _ClassRef:
    """一个自由类名引用(未绑定名),例如 `ColumnParallelLinear` / `MLASelfAttention`。
    与"已绑定的 config 原始值(bool/int/None)"区分开——这样条件里若碰到类名引用能 fail-loud。"""
    name: str


class _Return:
    """函数体求值命中 `return` 的信号(携返回值),用于从 if 分支里冒泡出去。"""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


# ── 解释器 ───────────────────────────────────────────────────────────────────
class _Interp:
    def __init__(self, mf_root: str):
        self.mf_root = mf_root
        # name -> (FunctionDef, src_file_relpath)
        self.funcs: dict[str, tuple[ast.FunctionDef, str]] = {}
        for rel in _SPEC_FILES:
            path = os.path.join(mf_root, *rel.split("/"))
            if not os.path.isfile(path):
                # gpt_layer_specs.py 缺失是硬错;moe 缺失只在真正调用 get_moe_module_spec 时才 fail。
                if rel.endswith("gpt_layer_specs.py"):
                    raise ValueError(f"Pass A 找不到 spec 源: {path}(fail-loud)")
                continue
            with open(path, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    self.funcs.setdefault(node.name, (node, rel))

    # --- 入口:绑 flags 到 get_gpt_layer_local_spec,解释其体 ---
    def resolve(self, flags: dict) -> ResolvedSpec:
        if _ENTRY not in self.funcs:
            raise ValueError(f"Pass A 源里找不到入口函数 {_ENTRY}(fail-loud)")
        fdef, rel = self.funcs[_ENTRY]
        param_names = {a.arg for a in fdef.args.args}
        # 只取签名里认得的 flag;其余(更宽的 config 键)忽略(它们不是本函数的参数)。
        kwargs = {k: v for k, v in flags.items() if k in param_names}
        result = self._call(fdef, rel, positional=[], keyword=kwargs)
        if not isinstance(result, ResolvedSpec):
            raise ValueError(
                f"{rel}:{fdef.lineno} 入口 {_ENTRY} 未解出 ModuleSpec(得到 {result!r})(fail-loud)"
            )
        return result

    # --- 函数调用:绑参(默认→位置→关键字)后解释函数体 ---
    def _call(self, fdef: ast.FunctionDef, rel: str, positional: list, keyword: dict):
        args = fdef.args.args
        defaults = fdef.args.defaults
        n_args, n_def = len(args), len(defaults)
        env: dict = {}
        # 1) 默认值(尾部对齐)
        for i, a in enumerate(args):
            j = i - (n_args - n_def)
            if j >= 0:
                env[a.arg] = self._eval(defaults[j], {}, rel)
        # 2) 位置实参
        for i, val in enumerate(positional):
            if i >= n_args:
                raise ValueError(f"{rel}:{fdef.lineno} 调用 {fdef.name} 位置实参过多(fail-loud)")
            env[args[i].arg] = val
        # 3) 关键字实参
        for k, val in keyword.items():
            env[k] = val
        # 4) 解释体
        for stmt in fdef.body:
            r = self._exec(stmt, env, rel)
            if isinstance(r, _Return):
                return r.value
        raise ValueError(f"{rel}:{fdef.lineno} 函数 {fdef.name} 走到底未 return(fail-loud)")

    # --- 语句执行:返回 _Return 表示命中 return(冒泡),否则 None ---
    def _exec(self, stmt: ast.stmt, env: dict, rel: str):
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return None  # docstring / 裸常量
        if isinstance(stmt, ast.Return):
            return _Return(self._eval(stmt.value, env, rel))
        if isinstance(stmt, ast.Assign):
            val = self._eval(stmt.value, env, rel)
            for tgt in stmt.targets:
                if not isinstance(tgt, ast.Name):
                    raise ValueError(f"{rel}:{stmt.lineno} 不支持的赋值目标(fail-loud)")
                env[tgt.id] = val
            return None
        if isinstance(stmt, ast.If):
            branch = stmt.body if self._truthy(stmt.test, env, rel) else stmt.orelse
            for s in branch:
                r = self._exec(s, env, rel)
                if isinstance(r, _Return):
                    return r
            return None
        if isinstance(stmt, ast.Raise):
            raise ValueError(
                f"{rel}:{stmt.lineno} spec 构造函数命中 raise(此 config 组合不被支持):"
                f"{self._raise_text(stmt)} —— fail-loud"
            )
        raise ValueError(
            f"{rel}:{stmt.lineno} Pass A 遇到未支持的语句 {type(stmt).__name__}(fail-loud)"
        )

    # --- 表达式求值:返回 ResolvedSpec / 叶子 str / _ClassRef / config 原始值(bool/int/None/str) ---
    def _eval(self, node, env: dict, rel: str):
        if node is None:
            raise ValueError(f"{rel}: 空表达式(fail-loud)")
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            # 不在 env:视为自由类名引用(叶子类 / Cell 类)。若被用于条件会在 _truthy 里 fail-loud。
            return _ClassRef(node.id)
        if isinstance(node, ast.IfExp):
            chosen = node.body if self._truthy(node.test, env, rel) else node.orelse
            return self._eval(chosen, env, rel)
        if isinstance(node, ast.BoolOp):
            return self._eval_boolop(node, env, rel)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self._to_bool(self._eval(node.operand, env, rel), node, rel)
        if isinstance(node, ast.Compare):
            return self._eval_compare(node, env, rel)
        if isinstance(node, ast.Call):
            return self._eval_call(node, env, rel)
        raise ValueError(
            f"{rel}:{getattr(node, 'lineno', '?')} Pass A 无法求值的表达式 "
            f"{type(node).__name__}(fail-loud)"
        )

    def _eval_call(self, call: ast.Call, env: dict, rel: str):
        func = call.func
        if not isinstance(func, ast.Name):
            raise ValueError(
                f"{rel}:{call.lineno} 不支持的调用形态 {ast.dump(func)}(fail-loud)"
            )
        name = func.id
        if name == "ModuleSpec":
            return self._build_modulespec(call, env, rel)
        if name == "get_norm_cls":
            return "Norm"  # 归一化层类由 fused_norm 决定,规范为叶子 "Norm"
        if name in self.funcs:  # 本地 helper:get_mlp/attention/moe_module_spec → 递归
            fdef, frel = self.funcs[name]
            positional = [self._eval(a, env, rel) for a in call.args]
            keyword = {kw.arg: self._eval(kw.value, env, rel) for kw in call.keywords if kw.arg}
            return self._call(fdef, frel, positional, keyword)
        raise ValueError(
            f"{rel}:{call.lineno} Pass A 遇到未知调用 {name}(...)——不是 ModuleSpec/helper/get_norm_cls"
            f"(fail-loud;缺覆盖或非静态构造)"
        )

    def _build_modulespec(self, call: ast.Call, env: dict, rel: str) -> ResolvedSpec:
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        if "module" not in kw:
            raise ValueError(f"{rel}:{call.lineno} ModuleSpec 缺 module=(fail-loud)")
        cell = self._as_leaf(self._eval(kw["module"], env, rel), call, rel)
        submods: dict = {}
        sub_node = kw.get("submodules")
        if sub_node is not None:
            if not isinstance(sub_node, ast.Call):
                raise ValueError(
                    f"{rel}:{call.lineno} ModuleSpec.submodules 不是 Submodules(...) 调用(fail-loud)"
                )
            for k in sub_node.keywords:
                if k.arg is None:
                    raise ValueError(f"{rel}:{sub_node.lineno} submodules 含 **kwargs(fail-loud)")
                submods[k.arg] = self._resolve_field(k.value, env, rel)
        return ResolvedSpec(cell=cell, submodules=submods)

    def _resolve_field(self, node, env: dict, rel: str) -> Union[ResolvedSpec, str]:
        val = self._eval(node, env, rel)
        if isinstance(val, ResolvedSpec):
            return val
        if isinstance(val, _ClassRef):
            return _NAME_ALIAS.get(val.name, val.name)
        if isinstance(val, str):  # 如 get_norm_cls → "Norm"
            return val
        raise ValueError(
            f"{rel}:{getattr(node, 'lineno', '?')} submodule 字段解出非类/非 ModuleSpec 值 "
            f"{val!r}(fail-loud)"
        )

    def _as_leaf(self, val, node, rel: str) -> str:
        """module= 的取值必是一个类名(可能来自三元 / 变量绑定的 _ClassRef)。"""
        if isinstance(val, _ClassRef):
            return _NAME_ALIAS.get(val.name, val.name)
        if isinstance(val, str):
            return val
        raise ValueError(
            f"{rel}:{getattr(node, 'lineno', '?')} ModuleSpec.module 解出非类名 {val!r}(fail-loud)"
        )

    # --- 条件求值(必须可判定为 bool,否则 fail-loud) ---
    def _truthy(self, test, env: dict, rel: str) -> bool:
        return self._to_bool(self._eval(test, env, rel), test, rel)

    def _to_bool(self, val, node, rel: str) -> bool:
        if isinstance(val, (bool, int, float)) or val is None:
            return bool(val)
        if isinstance(val, bool):  # 冗余保护
            return val
        # _ClassRef / ResolvedSpec / 其它 → 条件不可由 config 判定
        raise ValueError(
            f"{rel}:{getattr(node, 'lineno', '?')} 条件不可由已绑定的 config 参数判定"
            f"(得到 {val!r};引用了未绑定名或非 config 表达式)—— fail-loud"
        )

    def _eval_boolop(self, node: ast.BoolOp, env: dict, rel: str):
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not self._truthy(v, env, rel):
                    return False
            return True
        # Or
        for v in node.values:
            if self._truthy(v, env, rel):
                return True
        return False

    def _eval_compare(self, node: ast.Compare, env: dict, rel: str):
        if len(node.ops) != 1:
            raise ValueError(f"{rel}:{node.lineno} 不支持链式比较(fail-loud)")
        left = self._eval(node.left, env, rel)
        right = self._eval(node.comparators[0], env, rel)
        for side in (left, right):
            if isinstance(side, (ResolvedSpec, _ClassRef)):
                raise ValueError(
                    f"{rel}:{node.lineno} 比较操作数不可判定({side!r})—— fail-loud"
                )
        op = node.ops[0]
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is not right
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        raise ValueError(
            f"{rel}:{node.lineno} 不支持的比较运算 {type(op).__name__}(fail-loud)"
        )

    @staticmethod
    def _raise_text(stmt: ast.Raise) -> str:
        exc = stmt.exc
        if isinstance(exc, ast.Call):
            head = exc.func.id if isinstance(exc.func, ast.Name) else "?"
            msg = ""
            if exc.args and isinstance(exc.args[0], ast.Constant):
                msg = str(exc.args[0].value)
            return f"raise {head}({msg!r})" if msg else f"raise {head}(...)"
        if isinstance(exc, ast.Name):
            return f"raise {exc.id}"
        return "raise"


# ── 对外入口 ──────────────────────────────────────────────────────────────────
def resolve_layer_spec(mf_root: str, flags: dict) -> ResolvedSpec:
    """读真 `gpt_layer_specs.py`,按 `flags` 静态解释 `get_gpt_layer_local_spec`,返回解出的模块树。

    参数
      mf_root — mindformers 源根(含 `parallel_core/...`);仅被 `ast` 读取,绝不 import。
      flags   — config 派生的 kwargs(见 DSv3 示例);缺的参数用函数自带默认。
    行为
      解不出的 if / 命中 raise / 未知构造 → 抛 ValueError 点名 file:line。
    """
    return _Interp(mf_root).resolve(flags)
