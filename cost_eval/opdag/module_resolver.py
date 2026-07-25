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

from .module_index import ClassIndex


# ── 对外数据结构 ──────────────────────────────────────────────────────────────
@dataclass
class ResolvedSpec:
    """一个被实例化的 Cell:cell=类名;submodules[field] = 叶子类名(str)或嵌套 ResolvedSpec。

    `origin_rel`(2026-07-25 新增,可选):**定义该 Cell 类的文件**相对 mf_root 的路径,
    由 Pass A 顺 spec 文件自己的 `import` 解出。存在的理由是一条实测的**静默解错**:
    mindformers 有三棵并行的树,`class MoELayer` 在 `pynative/transformers/moe/moe_layer.py`、
    `parallel_core/training_graph/transformer/moe/moe_layer.py`、
    `parallel_core/inference/transformer/moe/moe_layer.py` 各有一份。`extractor._find_cell_file`
    按 `os.walk` **首个命中**定位,实测把 pynative 的 `MoELayer` 解成了 `inference` 那份
    (于是报「`self.router` 的 build_module 首参不是 submodules.<字段>」—— 那是另一棵树的写法)。
    spec 文件的 import 是**唯一权威**的答案:`pynative/base_models/gpt/moe_module_specs.py:20`
    写着 `from mindformers.pynative.transformers.moe.moe_layer import MoELayer`。
    """
    cell: str
    submodules: dict = field(default_factory=dict)  # dict[str, Union["ResolvedSpec", str]]
    origin_rel: str | None = None


# 叶子类名 → 规范 op 类型(供后续 Pass 用;复合 Cell 不在此表——它们被递归展开)。
LEAF_OPTYPE = {
    "ColumnParallelLinear": "MatMul",
    "RowParallelLinear": "MatMul",
    "SequenceParallelLinear": "MatMul",
    # pynative 侧的统一线性叶子(`mindformers.pynative.layers.linear.Linear`):
    # training_graph 用 Column/Row/SequenceParallelLinear 三分,pynative 只有一个 `Linear`
    # (见 pynative/base_models/gpt/gpt_layer_specs.py 与
    #  pynative/base_models/gpt/experimental_attention_variant_module_specs.py 的 submodules 填充)。
    # 缺此表项时 pynative 任何 spec 树都在 extractor._bind_build_module 处 fail-loud。
    "Linear": "MatMul",
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

# pynative(真机 PyNative 训练路径)的 spec 文件集。DSv4-Flash 走的是这一支:
# `pynative/.../gpt_layer_specs.py:get_gpt_layer_local_spec` 带 `is_dsv4_hybrid` /
# `enable_hyper_connections` 两个参数(training_graph 版**没有**这两个参数,传进去会被
# `_Interp.resolve` 的 param_names 过滤**静默丢弃** → 解出 DSv3 的 MLASelfAttentionConcatenated
# 而非 DSv4HybridSelfAttention)。第三个文件是必需的:`:110` 调
# `get_dsv4_hybrid_module_spec(...)`,该 helper 定义在那里,不登记则 `_eval_call` fail-loud。
PYNATIVE_SPEC_FILES = (
    "pynative/base_models/gpt/gpt_layer_specs.py",
    "pynative/base_models/gpt/moe_module_specs.py",
    "pynative/base_models/gpt/experimental_attention_variant_module_specs.py",
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
    def __init__(self, mf_root: str, spec_files=None):
        self.mf_root = mf_root
        self._index: ClassIndex | None = None    # 懒建:只在需要解 submodules dataclass 时用
        # name -> (FunctionDef, src_file_relpath)
        self.funcs: dict[str, tuple[ast.FunctionDef, str]] = {}
        for rel in (spec_files or _SPEC_FILES):
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
            # ── **未显式填的字段用它在 dataclass 里声明的缺省值**(2026-07-25)────────────
            # 这不是"补一个默认"——真机上 `TransformerLayerSubmodules()` 的 6 个字段缺省全是
            # `IdentityOp`(`pynative/transformers/transformer_layer.py:68-75`),而 dsv4_hybrid
            # 的 spec 只填 4 个槽(`pynative/base_models/gpt/gpt_layer_specs.py:105-113`),
            # 另两个槽在真机上**确实被实例化**为 `IdentityOp`
            # (`build_module(IdentityOp, ...)` → `IdentityOp(...)`,`spec_utils.py:76-77,97`)。
            # 此前 Pass A 只收显式填的字段 → `extractor._bind_build_module` 在
            # `self.pre_cross_attn_layernorm` 处 fail-loud → **整个 mHC 层零覆盖**
            # (评估文档 §3.3)。缺省是 `None` 的字段**不收**(见 `_declared_defaults` 的理由)。
            sub_cls = sub_node.func.id if isinstance(sub_node.func, ast.Name) else None
            if sub_cls:
                submods.update(self._declared_defaults(sub_cls, rel))
            for k in sub_node.keywords:
                if k.arg is None:
                    raise ValueError(f"{rel}:{sub_node.lineno} submodules 含 **kwargs(fail-loud)")
                submods[k.arg] = self._resolve_field(k.value, env, rel)
        return ResolvedSpec(cell=cell, submodules=submods,
                            origin_rel=self._origin_rel(cell, rel))

    def _origin_rel(self, cell: str, rel: str) -> str | None:
        """`cell` 类的定义文件(顺 `rel` 这个 spec 文件的 import 解);解不到 → None。"""
        if self._index is None:
            self._index = ClassIndex(self.mf_root)
        rc = self._index.resolve(cell, rel)
        return rc.rel if rc is not None else None

    # --- submodules dataclass 的**声明缺省值** ---
    def _declared_defaults(self, sub_cls: str, from_rel: str) -> dict:
        if self._index is None:
            self._index = ClassIndex(self.mf_root)
        return _declared_defaults(self._index, sub_cls, from_rel)

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


# ── submodules dataclass 的声明缺省值(2026-07-25)──────────────────────────────────
def _declared_defaults(index: ClassIndex, sub_cls: str, from_rel: str) -> dict:
    """`<XSubmodules>` dataclass 的字段声明缺省 → `{field: 叶子类名}`。

    **只收能确定成"一个类"的缺省**:
      * `field: T = IdentityOp` → `{"field": "Identity"}`(经 `_NAME_ALIAS` 归一);
      * `field: T = None` → **不收**。理由是源侧事实,不是保守:`build_module(None, ...)`
        在真机上会走到 `import_module(None.module)` 而抛 AttributeError
        (`parallel_core/utils/spec_utils.py:76-82`)—— 所以「缺省 None」的语义是
        「这条代码路径不会对它调 build_module」。把它填成 Identity 会造出一个真机
        **不存在**的节点(`MLASelfAttentionSubmodules` 的 7 个字段缺省全是 None,
        `parallel_core/training_graph/transformer/multi_latent_attention.py:56-62`);
      * 其它缺省形态(`field(default_factory=...)` / 调用 / 下标 …)→ **不收**,
        让 `extractor._bind_build_module` 保持 fail-loud(绝不猜一个类出来)。

    解析不到该 dataclass(如它定义在快照外)→ 返回 `{}`,同样退回 fail-loud。
    """
    rc = index.resolve(sub_cls, from_rel)
    if rc is None:
        return {}
    out: dict = {}
    for stmt in rc.node.body:
        if not isinstance(stmt, ast.AnnAssign) or stmt.value is None:
            continue
        if not isinstance(stmt.target, ast.Name):
            continue
        v = stmt.value
        if isinstance(v, ast.Name):
            out[stmt.target.id] = _NAME_ALIAS.get(v.id, v.id)
        elif isinstance(v, ast.Attribute):
            out[stmt.target.id] = _NAME_ALIAS.get(v.attr, v.attr)
        # Constant(None) / 其它形态:不收(见 docstring)
    return out


def submodule_declared_defaults(mf_root: str, sub_cls: str, from_rel: str) -> dict:
    """对外可测入口:见 `_declared_defaults`。"""
    return _declared_defaults(ClassIndex(mf_root), sub_cls, from_rel)


# ── 对外入口 ──────────────────────────────────────────────────────────────────
def resolve_layer_spec(mf_root: str, flags: dict, spec_files=None) -> ResolvedSpec:
    """读真 `gpt_layer_specs.py`,按 `flags` 静态解释 `get_gpt_layer_local_spec`,返回解出的模块树。

    参数
      mf_root    — mindformers 源根(含 `parallel_core/...`);仅被 `ast` 读取,绝不 import。
      flags      — config 派生的 kwargs(见 DSv3 示例);缺的参数用函数自带默认。
      spec_files — 可选。spec 构造函数所在文件集(相对 mf_root)。缺省 `_SPEC_FILES`
                   (`parallel_core/training_graph/...`,与既有 DSv3 调用逐字节一致);
                   传 `PYNATIVE_SPEC_FILES` 解 pynative(真机 PyNative)那一支——
                   dsv4_hybrid / hyper-connection 只在那里可解。
    行为
      解不出的 if / 命中 raise / 未知构造 → 抛 ValueError 点名 file:line。
    """
    return _Interp(mf_root, spec_files).resolve(flags)


# MTP 层 spec 构造函数所在文件(`get_mtp_layer_spec` @ `multi_token_prediction.py:223`)。
MTP_SPEC_FILES = PYNATIVE_SPEC_FILES + (
    "pynative/transformers/multi_token_prediction.py",
)


def resolve_spec_call(mf_root: str, entry: str, flags: dict, spec_files=None,
                      positional=(), keyword=None) -> ResolvedSpec:
    """静态解释**任意一个** spec 构造函数(不只入口 `get_gpt_layer_local_spec`)。

    存在的理由:MTP 的层 spec 不由入口函数产出 —— `get_gpt_mtp_block_spec`
    (`pynative/base_models/gpt/gpt_layer_specs.py:227-256`)拿 decoder block 的**最后一层**
    spec 与 `hc_head` 去调 `get_mtp_layer_spec(...)`(`multi_token_prediction.py:223`)。
    调用方把已解好的 decoder 层 `ResolvedSpec` 当位置/关键字实参传进来即可,解释逻辑复用同一套
    (`ResolvedSpec` 实参在 `_resolve_field` 里原样通过)。

    `keyword` 的值可以是 `ResolvedSpec` / 叶子类名字符串 / config 原始值。
    """
    interp = _Interp(mf_root, spec_files or MTP_SPEC_FILES)
    if entry not in interp.funcs:
        raise ValueError(f"Pass A 源里找不到函数 {entry}(fail-loud)")
    fdef, rel = interp.funcs[entry]
    param_names = {a.arg for a in fdef.args.args}
    kw = {k: v for k, v in (keyword or {}).items() if k in param_names}
    for k, v in (flags or {}).items():
        if k in param_names and k not in kw:
            kw[k] = v
    result = interp._call(fdef, rel, positional=list(positional), keyword=kw)
    if not isinstance(result, ResolvedSpec):
        raise ValueError(
            f"{rel}:{fdef.lineno} {entry} 未解出 ModuleSpec(得到 {result!r})(fail-loud)")
    return result
