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
import re

from .schema import OpDAG
from .init_binder import bind_init, Binding
from .construct_walker import walk_construct_meta, SubExtract
from .module_resolver import ResolvedSpec, LEAF_OPTYPE
from .init_dims import eval_init_dims


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


def _init_classes(tree: ast.AST, cls_name: str) -> list[str]:
    """MRO(derived→base)里**所有定义了 __init__ 的类**,按 BFS 顺序返回。

    MLA 需要:`MLASelfAttention.__init__`(建 q/kv 投影、rope、split 等)+ 基类
    `MultiLatentAttention.__init__`(建 core_attention、linear_proj、shape/cast/reshape 等)——
    两级 __init__ 的绑定都要合并,否则 construct 里 `self.shape` / `self.cast` 会缺绑 fail-loud。
    (MLPInterleaved 自身无 __init__ → 只返回基类 MLP,与旧 _defining_class 行为一致。)
    """
    order: list[str] = []
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
        if _method_of(cls, "__init__") is not None:
            order.append(cname)
        for base in cls.bases:
            if isinstance(base, ast.Name):
                queue.append(base.id)
    return order


# ── Morph(self.<method>) 别名侦测 ─────────────────────────────────────────────
def _unwrap_to_call(node: ast.AST, name: str) -> ast.Call | None:
    """剥链式 `.add_prim_attr(...)/.shard(...)` 等,取最内层 Call;其 func 为 Name==name 时返回该 Call,否则 None。"""
    while isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return node if f.id == name else None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
            node = f.value
            continue
        return None
    return None


def _morph_aliases(tree: ast.AST, init_cls: str) -> dict[str, str]:
    """__init__ 里 `self.<X> = Morph(self.<method>, ...)`(mindspore 自定义融合原语):记录别名 X→method。
    construct 里 self.X(...) 实为调用被 Morph 包裹的 self.<method>(...) 的计算 → walker 内联该方法以还原其算子
    (如 FFNGroupedGEMM.morphed_forward → forward_func → GroupedMatmul×2 + swiglu;permute/unpermute 在 forward_func 内)。"""
    cls = _find_class(tree, init_cls)
    init = _method_of(cls, "__init__") if cls else None
    out: dict[str, str] = {}
    if init is None:
        return out
    for stmt in ast.walk(init):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        morph = _unwrap_to_call(stmt.value, "Morph")
        if morph is None or not morph.args:
            continue
        a0 = morph.args[0]
        if (isinstance(a0, ast.Attribute) and isinstance(a0.value, ast.Name)
                and a0.value.id == "self"):
            out[tgt.attr] = a0.attr
    return out


# dtype 名归一(与 construct_walker 的短标签一致)。
_DTYPE_SHORT = {
    "float32": "fp32", "fp32": "fp32", "float": "fp32",
    "bfloat16": "bf16", "bf16": "bf16", "bfloat": "bf16",
    "float16": "fp16", "fp16": "fp16", "half": "fp16",
}


# ── __init__ 里的具名模块绑定 ─────────────────────────────────────────────────
def _named_module_binds(
    tree: ast.AST, init_cls: str, spec: ResolvedSpec, config_flags: dict, src_file: str,
    recurse: bool = False, subcell_specs: dict | None = None,
    linear_dims: dict | None = None,
) -> dict[str, Binding]:
    cls = _find_class(tree, init_cls)
    init = _method_of(cls, "__init__")
    if init is None:  # 理论上 init_cls 已保证有 __init__
        raise ValueError(f"extractor: {init_cls} 无 __init__（fail-loud）")

    compute_dtype = config_flags.get("compute_dtype", "bf16")
    # 归一化层的计算/保存 dtype(fp32-残差机制);缺省 fp32(norm 常在 fp32 统计)。
    ln_compute_dtype = _DTYPE_SHORT.get(
        config_flags.get("layernorm_compute_dtype", "fp32"), "fp32"
    )
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
            binds[name] = _bind_build_module(
                stmt.value, name, spec, compute_dtype, ln_compute_dtype, src_file,
                recurse, subcell_specs, linear_dims,
            )
        elif fname == "get_activation":
            if activation_present:
                binds[name] = Binding(
                    op="Activation",
                    attrs={"activation_type": act_type if act_type_given else None},
                )
            # activation_type 显式为 None → 不绑定(下游按 None 剪枝)
    return binds


def _bind_build_module(
    call: ast.Call, self_name: str, spec: ResolvedSpec, compute_dtype: str,
    ln_compute_dtype: str, src_file: str,
    recurse: bool = False, subcell_specs: dict | None = None,
    linear_dims: dict | None = None,
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
        # 子 Cell(submodules 已解成嵌套 ResolvedSpec):挂 SubCell,recurse 时按 field 取该子 spec 递归。
        return Binding(op="SubCell", attrs={"cell": leaf.cell, "field": field, "bare": False})
    if isinstance(leaf, str):
        op = LEAF_OPTYPE.get(leaf)
        if op is None:
            # 裸 Cell 类名(如 experts="FFNGroupedGEMM"):recurse 且 subcell_specs 提供其 ResolvedSpec
            # → 挂 SubCell(bare),由 resolver 按类名从 subcell_specs 取子 spec 递归;否则维持 fail-loud。
            if recurse and subcell_specs and leaf in subcell_specs:
                return Binding(op="SubCell", attrs={"cell": leaf, "field": field, "bare": True})
            raise ValueError(
                f"extractor: 叶子类 {leaf!r} 不在 LEAF_OPTYPE（self.{self_name} @ {src_file}）—— fail-loud"
            )
        attrs = {"module": leaf, "compute_dtype": compute_dtype}
        if op == "Norm":
            # 归一化层在 fp32 计算并存 fp32 输入(fp32-残差机制)→ 供 bprop_rules 把该 Norm 输入按 fp32 计。
            attrs["ln_compute_dtype"] = ln_compute_dtype
        # PART A:linear 的 (in,out) 符号维度(build_module 位序 1/2,由 __init__ 求值得)。
        dims = (linear_dims or {}).get(self_name)
        if dims is not None:
            in_d, out_d = dims
            if in_d is not None:
                attrs["in_dim"] = in_d
            if out_d is not None:
                attrs["out_dim"] = out_d
        return Binding(op=op, attrs=attrs)
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


# ── 子 Cell 定位 + 递归 resolver ──────────────────────────────────────────────
def _find_cell_file(mf_root: str, cell_name: str) -> str:
    """在 mf_root 里搜 `class <cell_name>` 的定义源,返回相对 mf_root 的路径(用 "/" 分隔)。
    多处定义取首个命中;找不到 → fail-loud。"""
    pat = re.compile(rf"^\s*class\s+{re.escape(cell_name)}\b", re.M)
    for dirpath, _dirs, files in os.walk(mf_root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    txt = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
            if pat.search(txt):
                return os.path.relpath(p, mf_root).replace(os.sep, "/")
    raise ValueError(
        f"extractor: 在 {mf_root} 找不到定义 class {cell_name} 的源文件（fail-loud）"
    )


def _make_subcell_resolver(
    mf_root: str, parent_spec: ResolvedSpec, config_flags: dict,
    subcell_specs: dict | None, stack_with_self: set,
):
    """构造给 walker 的 resolver:遇 SubCell 调用点 → 取子 spec、定位子文件、递归抽取,返回 SubExtract。"""
    def resolver(cell_name: str, field: str, bare: bool) -> SubExtract:
        if bare:
            sub_spec = (subcell_specs or {}).get(cell_name)
            if not isinstance(sub_spec, ResolvedSpec):
                raise ValueError(
                    f"extractor: 裸类 SubCell {cell_name!r} 未由 subcell_specs 提供 ResolvedSpec（fail-loud）"
                )
        else:
            sub_spec = parent_spec.submodules.get(field)
            if not isinstance(sub_spec, ResolvedSpec):
                raise ValueError(
                    f"extractor: SubCell 字段 {field!r} 在父 spec 里非 ResolvedSpec（得到 {sub_spec!r}）—— fail-loud"
                )
        rel = _find_cell_file(mf_root, sub_spec.cell)
        dag, params, returns = _extract_meta(
            mf_root, rel, sub_spec.cell, sub_spec, config_flags,
            recurse=True, subcell_specs=subcell_specs, _stack=stack_with_self,
        )
        return SubExtract(nodes=dag.nodes, edges=dag.edges, param_names=params, returns=returns)
    return resolver


# ── 对外入口 ──────────────────────────────────────────────────────────────────
def extract_cell(
    mf_root: str,
    cell_file_relpath: str,
    cls_name: str,
    spec: ResolvedSpec,
    config_flags: dict,
    present_params: set | None = None,
    recurse: bool = False,
    subcell_specs: dict | None = None,
) -> OpDAG:
    """读真 mindformers Cell 源,按 config 剪枝,产出其 op-DAG。

    参数
      mf_root           — mindformers 源根(仅 ast 读,绝不 import)。
      cell_file_relpath — Cell 源文件相对 mf_root 的路径(用 "/" 分隔)。
      cls_name          — 目标 Cell 类名(其 construct 被走查;__init__ 可继承自基类)。
      spec              — 该 Cell 的 ResolvedSpec(submodules 字段 → 叶子类名 / 子 ResolvedSpec)。
      config_flags      — config 派生的 flags(gated_linear_unit / activation_type / q_lora_rank /
                          compute_dtype / layernorm_compute_dtype / ...)。
      present_params    — 可选。construct 形参中"缺省 None 但真机总被传入"的名字(如 MLA 的
                          rotary_pos_emb):从 param_defaults 移除并标记为 present。
      recurse           — 打开子 Cell 递归内联:当 self.<X>(...) 绑定为 SubCell(submodules 的子
                          ResolvedSpec,或裸 Cell 类名+subcell_specs 提供其 spec)→ 递归抽取该子
                          Cell DAG 并在调用点内联(id 续编、形参重映射、跨界连边)。
      subcell_specs     — 可选。{类名: ResolvedSpec}:为裸 Cell 类名(不在 LEAF_OPTYPE)提供其解析树。
    行为
      找不到源 / 无法解析具名模块 / 剪枝时遇不可判定 if / 子 Cell 递归环 → fail-loud(ValueError)。
    """
    dag, _params, _returns = _extract_meta(
        mf_root, cell_file_relpath, cls_name, spec, config_flags,
        present_params=present_params, recurse=recurse, subcell_specs=subcell_specs,
    )
    return dag


def _extract_meta(
    mf_root: str,
    cell_file_relpath: str,
    cls_name: str,
    spec: ResolvedSpec,
    config_flags: dict,
    present_params: set | None = None,
    recurse: bool = False,
    subcell_specs: dict | None = None,
    _stack: set | None = None,
):
    """extract_cell 的内核,额外返回 (OpDAG, construct 形参名, 返回值分类)——供 resolver 递归内联。
    `_stack` 携当前正在抽取的 Cell 类名链,用于子 Cell 递归环 fail-loud。"""
    stack = set(_stack or ())
    if cls_name in stack:
        raise ValueError(
            f"extractor: 检测到子 Cell 递归环——Cell {cls_name!r} 直接/间接自指（链 {sorted(stack)}）,fail-loud"
        )

    path = os.path.join(mf_root, *cell_file_relpath.split("/"))
    if not os.path.isfile(path):
        raise ValueError(f"extractor: 找不到 Cell 源 {path}（fail-loud）")
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    src_file = os.path.basename(path)

    if _find_class(tree, cls_name) is None:
        raise ValueError(f"extractor: {src_file} 里找不到 class {cls_name}（fail-loud）")

    # 0) recurse 时给 walker 备一个子 Cell resolver(携带把自己压栈后的递归链)。
    resolver = None
    if recurse:
        resolver = _make_subcell_resolver(
            mf_root, spec, config_flags, subcell_specs, stack | {cls_name}
        )

    # 1) 沿 __init__ 的 MRO 链(base→derived)做基础 + 具名绑定并合并(derived 覆盖 base)。
    init_classes = _init_classes(tree, cls_name)   # derived→base
    if not init_classes:
        raise ValueError(f"extractor: {cls_name} 及其基类均无 __init__（fail-loud）")
    # PART A:一次求值 __init__(全 MRO)得各 linear 的符号 (in,out) 维度 + self.<attr> dims_ctx。
    init_dims = eval_init_dims(tree, cls_name, config_flags)

    base_binds: dict[str, Binding] = {}
    named: dict[str, Binding] = {}
    method_aliases: dict[str, str] = {}
    for cname in reversed(init_classes):           # base 先绑,derived 后绑覆盖
        base_binds.update(bind_init(src, cname))
        named.update(_named_module_binds(
            tree, cname, spec, config_flags, src_file, recurse, subcell_specs,
            linear_dims=init_dims.linear_dims,
        ))
        method_aliases.update(_morph_aliases(tree, cname))  # Morph(self.method) 别名
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

    # 3) none_vars(bias 关闭)/ param_defaults(缺省即 None 的形参)/ present_vars(真机总传入的形参)。
    if config_flags.get("add_bias_linear", False):
        none_vars: set[str] = set()
    else:
        none_vars = _bias_none_vars(tree, cls_name, combined)
    param_defaults = _construct_none_defaults(tree, cls_name)
    present = set(present_params or ())
    for p in present:
        param_defaults.pop(p, None)   # present 覆盖"缺省 None":该形参按存在处理

    # 4) 走查 + 剪枝(recurse 时携 resolver:SubCell 调用点递归内联)。
    dag, params, returns = walk_construct_meta(
        src,
        cls_name,
        combined,
        src_file,
        config_flags=walker_flags,
        none_vars=none_vars,
        param_defaults=param_defaults,
        present_vars=present,
        subcell_resolver=resolver,
        method_aliases=method_aliases,
    )
    dag.dims_ctx = init_dims.dims_ctx           # self.<attr> → 符号 token(供 shape 推断解析)
    return dag, params, returns
