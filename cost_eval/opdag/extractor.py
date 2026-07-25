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
from .init_binder import bind_init, Binding, _base_call_name, unbound_aliases
from .construct_walker import (
    walk_construct_meta, SubExtract, assert_extraction_clean,
)
from .module_resolver import ResolvedSpec, LEAF_OPTYPE
from .init_dims import eval_init_dims
from .module_index import ClassIndex
from . import fn_saves


# ── 跨文件 MRO 单元(P0#3)────────────────────────────────────────────────────
def _mro_units(mf_root: str, tree: ast.AST, src: str, rel: str, cls_name: str,
               index: ClassIndex | None):
    """返回 **derived→base** 的 `[(cls_name, tree, src, rel)]`。

    有 `index` 时跨文件解析(`DSv4HybridSelfAttention` → `MultiLatentAttention`
    @ `pynative/transformers/multi_latent_attention.py:60`);没有时退化为同文件 BFS
    —— 既有 DSv3 调用路径行为逐字不变。
    """
    if index is None:
        return [(c, tree, src, rel) for c in _init_classes(tree, cls_name)]
    out = []
    for rc in index.mro(cls_name, rel):
        if _method_of(rc.node, "__init__") is not None:
            out.append((rc.name, rc.tree, rc.src, rc.rel))
    return out


def _module_level_consts(tree: ast.AST) -> dict:
    """模块级 `NAME = <字面量>`(如 `_BF16_MIN = -3.3895313892515355e38`,indexer.py:47)。
    这些是 **host 侧标量**;不登记的话它们会以"未知名"进 `ins`,变成一个假张量操作数。"""
    out: dict = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if isinstance(n.value, ast.Constant):
                out[n.targets[0].id] = n.value.value
            elif (isinstance(n.value, ast.UnaryOp) and isinstance(n.value.op, ast.USub)
                  and isinstance(n.value.operand, ast.Constant)):
                out[n.targets[0].id] = -n.value.operand.value
    return out


def _module_level_funcs(tree: ast.AST) -> dict:
    """模块级 `def`(**同文件**)→ 供 walker 在调用点内联。
    `unfused_compressed_sparse_attn`(csa.py:464)是这条路径上最重要的一个:它是那条
    ~11-op 注意力链的**实体**,此前作为「非 self.<method> 的自由函数」只进 opaque_calls。"""
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _fn_class_table(src_file: str, src: str, extra: dict | None = None) -> dict:
    """`_Function` 子类 → `{saves, forward_params, bare_ctx}`(源真值,复用已落地的 `fn_saves`)。

    `<Cls>.apply(...)` 的 saved 集**只能**从源里 `ctx.save_for_backward(...)` 逐字读
    (评估文档 §4 已实测手写侧漏了 `sparse_indices`/`sinks` 两项)—— 绝不让 PIN 去猜。
    """
    out = dict(extra or {})
    truth = fn_saves.scan_source(src_file, src)
    tree = ast.parse(src)
    params: dict[str, list] = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        fwd = _method_of(cls, "forward")
        if fwd is not None:
            params[cls.name] = [a.arg for a in fwd.args.args if a.arg != "ctx"]
    for rec in truth.functions:
        out[rec.cls] = {
            "saves": list(rec.saved.names) if rec.saved is not None else [],
            "forward_params": params.get(rec.cls, []),
            "bare_ctx": [c.attr for c in rec.ctx_attrs if c.kind not in ("predicate", "const")],
            "src": rec.src,
        }
    return out


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
    """剥链式 `.add_prim_attr(...)/.shard(...)` 等,取最内层 Call;其 func 为 Name==name,
    或 Attribute(attr==name)(如 `P.Morph(...)`——mindspore 常见的模块限定调用形式,
    VocabParallelEmbedding.embedding_morph 即此写法,layers.py:108)时返回该 Call,否则 None。"""
    while isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return node if f.id == name else None
        if isinstance(f, ast.Attribute) and f.attr == name:
            return node
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
                recurse, subcell_specs, linear_dims, _param_to_attr(init),
            )
        elif fname == "get_activation":
            if activation_present:
                binds[name] = Binding(
                    op="Activation",
                    attrs={"activation_type": act_type if act_type_given else None},
                )
            # activation_type 显式为 None → 不绑定(下游按 None 剪枝)
    return binds


def _param_to_attr(init_fn: ast.FunctionDef) -> dict:
    """`__init__` 里 `self.<X> = <形参名>` → `{形参名: X}`(构造函数注入项的转发追踪)。"""
    out: dict = {}
    if init_fn is None:
        return out
    for stmt in ast.walk(init_fn):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        t, v = stmt.targets[0], stmt.value
        if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                and t.value.id == "self" and isinstance(v, ast.Name)):
            out.setdefault(v.id, t.attr)
    return out


def _kw_self_args(call: ast.Call, param_to_attr: dict | None = None) -> dict:
    """构造点里形如 `<kw>=self.<attr>` 的关键字实参 -> `{kw: attr}`。

    这是**构造函数注入的子模块**的源侧证据链:父把自己那个已绑好的 `self.<attr>` 传给子,
    子在 `__init__` 里 `self.<kw> = <kw>`。实测必需:
      `build_module(submodules.compressor, ..., rotary_pos_emb=self.rotary_pos_emb)`
      (indexer.py:131、csa.py:602/614、deepseek_v4_hybrid_attention.py:88)
    -> 子(`Compressor` / `CSAIndexer` / `CompressedSparseAttention`)的
      `self.rotary_pos_emb = rotary_pos_emb`(compressor.py:92、indexer.py:87、csa.py:562)。
    """
    out = {kw.arg: kw.value.attr for kw in call.keywords
           if kw.arg and isinstance(kw.value, ast.Attribute)
           and isinstance(kw.value.value, ast.Name) and kw.value.value.id == "self"}
    # 第二种形态:`rotary_pos_emb=rotary_pos_emb` —— 直接转发**自己的 __init__ 形参**
    # (indexer.py:131)。此时要顺着 `self.<X> = <该形参>` 找回父侧的 self 名
    # (indexer.py:87 `self.rotary_pos_emb = rotary_pos_emb`)。`param_to_attr` 由调用方给。
    for kw in call.keywords:
        if kw.arg and isinstance(kw.value, ast.Name) and kw.arg not in out:
            attr = (param_to_attr or {}).get(kw.value.id)
            if attr is not None:
                out[kw.arg] = attr
    return out


def _bind_build_module(
    call: ast.Call, self_name: str, spec: ResolvedSpec, compute_dtype: str,
    ln_compute_dtype: str, src_file: str,
    recurse: bool = False, subcell_specs: dict | None = None,
    linear_dims: dict | None = None, param_to_attr: dict | None = None,
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
        return Binding(op="SubCell", attrs={"cell": leaf.cell, "field": field, "bare": False,
                                            "kw_self": _kw_self_args(call, param_to_attr)})
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


def _entry_defaults(tree: ast.AST, cls_name: str, method: str = "construct") -> dict:
    """入口方法形参的**字面量缺省**:{形参名: 值}(含 None)。"""
    dcls = _defining_class(tree, cls_name, method)
    cls = _find_class(tree, dcls) if dcls else None
    fn = _method_of(cls, method) if cls else None
    out: dict = {}
    if fn is None:
        return out
    args = fn.args.args
    defaults = fn.args.defaults
    n, nd = len(args), len(defaults)
    for i, a in enumerate(args):
        j = i - (n - nd)
        if j >= 0 and isinstance(defaults[j], ast.Constant):
            out[a.arg] = defaults[j].value
    return out


def _construct_none_defaults(tree: ast.AST, cls_name: str, method: str = "construct") -> dict:
    """construct 形参中缺省值恰为 None 的 → {形参名: None}(供顶部 `if x is not None: raise` 守卫剪枝)。"""
    return {k: v for k, v in _entry_defaults(tree, cls_name, method).items() if v is None}


def _construct_literal_defaults(tree: ast.AST, cls_name: str, method: str = "construct") -> dict:
    """construct 形参中缺省值是**非 None 字面量**的 → 标量环境种子。
    实测必需:`Compressor.construct(self, x, rope_pos_offset: int = 0)`(compressor.py:175)
    —— `rope_pos_offset` 是**标量形参**,不能当张量(当张量就会造出一条假操作数)。"""
    return {k: v for k, v in _entry_defaults(tree, cls_name, method).items() if v is not None}


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
    parent_tree: ast.AST | None = None, parent_rel: str | None = None,
    cross_file: bool = False, runtime_predicates: dict | None = None,
    host_call_allow: tuple = (), class_index=None,
    kernel_call_allow: tuple = (), input_axes: dict | None = None,
):
    """构造给 walker 的 resolver:遇 SubCell 调用点 → 取子 spec、定位子文件、递归抽取,返回 SubExtract。

    定位子文件优先同文件（parent_tree/parent_rel,惯用法B 典型——私有 helper Cell 与其使用者同源文件,
    如 `_LogSoftmax`/`_NLLLoss` 与 `CrossEntropyLoss` 同在 loss_func.py）,避免 `_find_cell_file` 的
    全树"首个命中"在类名跨模块重名时（下划线前缀名尤易重名,如 mindformers 另有
    `core/loss/loss.py` / `pynative/loss/loss.py` 各自的同名旧实现）误定位到无关源文件。
    """
    def resolver(cell_name: str, field: str, bare: bool,
                 method: str = "construct", injected_binds: dict | None = None) -> SubExtract:
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
        if (parent_tree is not None and parent_rel is not None
                and _find_class(parent_tree, sub_spec.cell) is not None):
            rel = parent_rel
        elif class_index is not None:
            rc = (class_index.resolve(sub_spec.cell, parent_rel)
                  if parent_rel else None)
            rel = rc.rel if rc is not None else _find_cell_file(mf_root, sub_spec.cell)
        else:
            rel = _find_cell_file(mf_root, sub_spec.cell)
        dag, params, returns = _extract_meta(
            mf_root, rel, sub_spec.cell, sub_spec, config_flags,
            recurse=True, subcell_specs=subcell_specs, _stack=stack_with_self,
            cross_file=cross_file, runtime_predicates=runtime_predicates,
            host_call_allow=host_call_allow, entry_method=method,
            kernel_call_allow=kernel_call_allow, input_axes=input_axes,
            injected_binds=injected_binds, _class_index=class_index,
        )
        return SubExtract(nodes=dag.nodes, edges=dag.edges, param_names=params, returns=returns,
                          opaque_calls=dag.opaque_calls, diagnostics=dag.diagnostics,
                          detached=list(getattr(dag, "detached", ()) or ()),
                          param_operands=list(getattr(dag, "param_operands", ()) or ()))
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
    strict: bool = False,
    cross_file: bool = False,
    runtime_predicates: dict | None = None,
    host_call_allow: tuple = (),
    kernel_call_allow: tuple = (),
    input_axes: dict | None = None,
    entry_method: str = "construct",
    injected_binds: dict | None = None,
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
      strict            — 可选(Task 2 / 评估文档 P0#1)。True:走查诊断(`dag.diagnostics`)非空即
                          `ExtractionDroppedError`。默认 False → 只记录,既有路径逐字节不变。
    行为
      找不到源 / 无法解析具名模块 / 剪枝时遇不可判定 if / 子 Cell 递归环 → fail-loud(ValueError)。
      **0 节点硬门恒开**(与 strict 无关):construct 有非平凡 body 却抽出 0 节点 →
      `ExtractionDroppedError`(实测反例 `CSAIndexer` fused 支,见 construct_walker._run_walker)。
    """
    dag, _params, _returns = _extract_meta(
        mf_root, cell_file_relpath, cls_name, spec, config_flags,
        present_params=present_params, recurse=recurse, subcell_specs=subcell_specs,
        strict=strict, cross_file=cross_file, runtime_predicates=runtime_predicates,
        host_call_allow=host_call_allow, kernel_call_allow=kernel_call_allow,
        input_axes=input_axes, entry_method=entry_method,
        injected_binds=injected_binds,
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
    strict: bool = False,
    cross_file: bool = False,
    runtime_predicates: dict | None = None,
    host_call_allow: tuple = (),
    kernel_call_allow: tuple = (),
    input_axes: dict | None = None,
    entry_method: str = "construct",
    injected_binds: dict | None = None,
    _class_index=None,
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

    # 0a) 跨文件类索引(P0#3)。`cross_file=False`(默认)时为 None → 既有单文件行为逐字不变。
    class_index = _class_index
    if cross_file and class_index is None:
        class_index = ClassIndex(mf_root)
    if class_index is not None:
        class_index.load(cell_file_relpath, tree=tree, src=src)

    # 0) recurse 时给 walker 备一个子 Cell resolver(携带把自己压栈后的递归链)。
    resolver = None
    if recurse:
        resolver = _make_subcell_resolver(
            mf_root, spec, config_flags, subcell_specs, stack | {cls_name},
            parent_tree=tree, parent_rel=cell_file_relpath,
            cross_file=cross_file, runtime_predicates=runtime_predicates,
            host_call_allow=host_call_allow, class_index=class_index,
            kernel_call_allow=kernel_call_allow, input_axes=input_axes,
        )

    # 1) 沿 __init__ 的 MRO 链(base→derived)做基础 + 具名绑定并合并(derived 覆盖 base)。
    #    **跨文件(P0#3)**:cross_file=True 时用 ClassIndex 顺 import 解析基类。
    units = _mro_units(mf_root, tree, src, cell_file_relpath, cls_name, class_index)
    init_classes = [u[0] for u in units]           # derived→base
    if not units:
        raise ValueError(f"extractor: {cls_name} 及其基类均无 __init__（fail-loud）")
    # PART A:一次求值 __init__(全 MRO,跨文件)得 linear (in,out) 维度 + dims_ctx + self_kinds。
    init_dims = eval_init_dims(tree, cls_name, config_flags,
                               mro_units=[(u[1], u[0]) for u in units])

    base_binds: dict[str, Binding] = {}
    named: dict[str, Binding] = {}
    method_aliases: dict[str, str] = {}
    alias_unknown: dict[str, dict] = {}
    for cname, utree, usrc, urel in reversed(units):    # base 先绑,derived 后绑覆盖
        ufile = os.path.basename(urel)
        base_binds.update(bind_init(usrc, cname, config_flags=config_flags, file=ufile,
                                    init_param_binds=injected_binds))
        named.update(_named_module_binds(
            utree, cname, spec, config_flags, ufile, recurse, subcell_specs,
            linear_dims=init_dims.linear_dims,
        ))
        method_aliases.update(_morph_aliases(utree, cname))  # Morph(self.method) 别名
        for a in unbound_aliases(usrc, cname, file=ufile):
            alias_unknown[a["attr"]] = a
    combined = {**base_binds, **named}

    # 1b) 直接实例化的子 Cell（惯用法B，spec §3.3a）：`self.X = <ClsName>(...)` 且
    #     ClsName ∈ subcell_specs → SubCell(bare)。build_module 之外的第二种子 Cell 组合方式
    #     （loss: `self._log_softmax = _LogSoftmax(config)`，loss_func.py:279）。
    #     subcell_specs 未提供（内存侧全部既有调用）时零行为变化。
    # 同时收集 `_Function` 子类表(供 `<Cls>.apply(...)` 挂源真值 saved 集)与
    # `self.<attr> = <Cls>()` 的类名映射(供 `self.<attr>.apply(...)` 定位类)。
    fn_classes = _fn_class_table(src_file, src)
    attr_ctor: dict[str, str] = {}
    for cname, utree, usrc, urel in units:             # derived→base
        cls_node = _find_class(utree, cname)
        init_fn = _method_of(cls_node, "__init__") if cls_node else None
        if init_fn is None:
            continue
        for stmt in ast.walk(init_fn):
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
                continue
            tgt = stmt.targets[0]
            if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self" and isinstance(stmt.value, ast.Call)):
                continue
            ctor = _base_call_name(stmt.value)
            if ctor is None:
                continue
            attr_ctor.setdefault(tgt.attr, ctor)
            # T0-6.5 Fix3:units 序已是 derived→base;此处 first-wins(`tgt.attr not in combined`)
            # 与步骤 1 的 `reversed(...)+update 覆盖`(等效 derived wins)MRO 方向一致。
            if recurse and subcell_specs and ctor in subcell_specs and tgt.attr not in combined:
                combined[tgt.attr] = Binding(
                    op="SubCell", attrs={"cell": ctor, "field": tgt.attr, "bare": True,
                                         "kw_self": _kw_self_args(
                                             stmt.value, _param_to_attr(init_fn))})
            # `_Function` 子类(自定义反向)在别的文件里定义时,顺 import 解析并抽它的源真值。
            if class_index is not None and ctor not in fn_classes:
                rc = class_index.resolve(ctor, urel)
                if rc is not None and any(
                        b in ("_Function", "Function") for b in class_index.base_names(rc.node)):
                    fn_classes.update(_fn_class_table(os.path.basename(rc.rel), rc.src))

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
    param_defaults = _construct_none_defaults(tree, cls_name, entry_method)
    present = set(present_params or ())
    for p in present:
        param_defaults.pop(p, None)   # present 覆盖"缺省 None":该形参按存在处理
    param_literals = _construct_literal_defaults(tree, cls_name, entry_method)

    # 3b) walker 需要的其余上下文(P0#2~#5)。
    self_kinds = dict(init_dims.self_kinds)
    for attr, ctor in attr_ctor.items():
        self_kinds[f"__cls__{attr}"] = ctor

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
        strict=False,     # 先不抛:下面还要并入 __init__ 侧的裸别名诊断,再统一判 strict
        class_index=class_index,
        cls_rel=cell_file_relpath,
        module_funcs=_module_level_funcs(tree),
        module_consts=_module_level_consts(tree),
        runtime_predicates=runtime_predicates,
        fn_classes=fn_classes,
        self_kinds=self_kinds,
        param_literals=param_literals,
        alias_unknown={k: v for k, v in alias_unknown.items() if k not in combined},
        host_call_allow=tuple(host_call_allow or ()),
        kernel_call_allow=tuple(kernel_call_allow or ()),
        input_axes=input_axes,
        entry_method=entry_method,
    )
    dag.dims_ctx = init_dims.dims_ctx           # self.<attr> → 符号 token(供 shape 推断解析)
    # ── Task 2(P0#1):把 `__init__` 侧的**裸函数别名**(pynative 惯用法,_CLS2OP 绑不上)并入诊断。
    # 调用点若真用到它们,`_handle_self_call` 会 fail-loud;但**没被调用**的那些今天完全不可见
    # (评估文档 §2.4:dsv4 链上 39 原语 / 196 调用点)。此处让它们出现在计数里,不静默跳过。
    # 与 `base_binds` 同口径:按**跨文件 MRO**(derived→base)逐类扫 —— 必须用**该类自己文件**
    # 的源(`usrc`/`ufile`),不能用派生类的源(基类可能在另一个文件里,见 `_mro_units`)。
    # P0#2 落地后此处只剩「真·未知原语」(能查表的已由 bind_init 形态 3 绑定)。
    aliases = [a for _cname, _ut, _us, _ur in units
               for a in unbound_aliases(_us, _cname, file=os.path.basename(_ur))
               if a["attr"] not in combined]
    if aliases:
        dag.diagnostics.setdefault("unbound_aliases", []).extend(aliases)
    if strict:
        assert_extraction_clean(dag)
    return dag, params, returns
