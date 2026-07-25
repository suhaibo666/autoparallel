# cost_eval/opdag/consumer.py
"""T11 consumer 桥：把 op-DAG save-set 的**符号 shape** 用 `DimTable` 代入算**字节**。

`derive_saves(dag)` 产出的每个 `Save.sym_shape` 是符号串（"S·B·q_lora_rank"、"E·cap·H"…）。本桥
把每个符号 token 映射到 DimTable 的值、按 `sym_shape` 代数求积得元素数，再乘 dtype 字节。

**不杜撰**：任一 token 无映射 / 值为 0 / shape 含 `?` 轴 → 该 save 归入 `unresolved` 列表（供后续
标定 margin），绝不编造 size。纯符号代入，不 import mindspore/mindformers。
"""
from __future__ import annotations

import math
import re

from .bprop_rules import derive_saves
from .sym_shape import (parse_shape, parse_axis, render_term, strip_numel_only,
                        _split_top)


# 符号 token → DimTable 属性名。token 集来自 sym_shape.CONFIG2SYM（提取器/推断产出的符号）。
_SYM2FIELD = {
    "H": "H",
    "ffn_hidden": "F",
    "moe_ffn": "moe_F",
    "n_heads": "n_heads",
    "E": "n_experts",
    "q_lora_rank": "q_lora_rank",
    "kv_lora_rank": "kv_lora_rank",
    "v_head_dim": "v_head_dim",
    "S": "S",
    "B": "B",
    "vocab": "vocab",
    # MLA：DAG 里 qk_head_dim=nope 段、qk_pos_emb_head_dim=rope 段。
    "qk_head_dim": "qk_nope_head_dim",
    "qk_pos_emb_head_dim": "qk_rope_head_dim",
    # ── DSv4-Flash（pynative dsa_indexer + compressor + CSA），2026-07-25 ─────────────
    # 与 `sym_shape.CONFIG2SYM` 的新增项一一对应（那张表决定符号能否留进 `dims_ctx`，
    # 本表决定符号能否取到值；**缺任一张都会让该张量落 unresolved**，见评估文档 §4.1 第 2 条）。
    # DimTable 字段名见 `cost_eval/model_spec.py:31-32`。
    "index_n_heads": "dsa_indexer_n_heads",
    "index_head_dim": "dsa_indexer_head_dim",
    "index_topk": "dsa_indexer_topk",
    "csa_window_size": "csa_window_size",
    "o_groups": "o_groups",
    "o_lora_rank": "o_lora_rank",
    # mHC 残差流倍数：`DimTable.num_residual_streams`（model_spec.py:64）。
    "hc_mult": "num_residual_streams",
}

# dtype 串 → 字节。未知回退 dims.dtype_bytes（compute dtype）。
_DTYPE_BYTES = {
    "fp32": 4, "float32": 4,
    "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2,
    "fp8": 1, "int8": 1, "uint8": 1,
    # bool：比较 / 逻辑 / isfinite 的产出（`csa.py:779` 的 O(S·S/r) mask、`:810` 的 O(S·topk)
    # mask）。MindSpore bool_ 张量 1 字节/元素。缺此项时会退回 compute dtype（bf16）= **2×**。
    "bool": 1, "bool_": 1,
    "int32": 4, "uint32": 4, "int64": 8, "uint64": 8, "fp64": 8, "float64": 8,
}


def _cap_value(dims):
    """MoE 每专家容量：capacity_factor × (S·B·topk) / n_experts（标准 top-k 容量式）。

    注：这是**标准容量公式**（tokens=S·B、每 token 选 topk、均摊到 E 专家）；若真机 dispatcher
    的容量定义不同，此处需据真机调整（已在 T11 报告 flag）。任一因子缺失 → None（不杜撰）。
    """
    E = getattr(dims, "n_experts", 0)
    topk = getattr(dims, "topk", 0)
    cf = getattr(dims, "capacity_factor", 0)
    if not E or not topk or not cf:
        return None
    tokens = dims.S * dims.B
    return math.ceil(cf * tokens * topk / E)


def _sym_value(sym, dims):
    """一个原子 token → 整数值。token 可能是和式 'a+b'（concat 出来的）或整除式 'S//4'
    （`sym_shape.floordiv` 第 3 档，压缩序列长度）。未知/0 → None。"""
    sym = sym.strip()
    parts = [p.strip() for p in _split_top(sym, "+")]
    if len(parts) > 1:                       # 和式单元：逐项求和
        vals = [_sym_value(p, dims) for p in parts]
        return sum(vals) if all(v is not None for v in vals) else None
    if "//" in sym:                          # 整除式单元：`<term>//<n>`（右结合地剥最外层）
        base, _, denom = sym.rpartition("//")
        if not denom.strip().lstrip("-").isdigit():
            return None
        n = int(denom.strip())
        if n == 0:
            return None
        b = _axis_value(parse_axis(base), dims) if base.strip() else None
        return None if b is None else b // n
    if sym == "cap":
        return _cap_value(dims)
    field = _SYM2FIELD.get(sym)
    if field is None:
        # **乘积项**（concat 的"元素数之和"里每一项都是一个乘积，见 `sym_shape._sum_term`）：
        # 不是单符号 → 按轴重新解析求积。纯符号但无映射的仍落 None（不杜撰）。
        stripped = sym.strip("()").strip()
        if _split_top(stripped, "·")[1:]:
            return _axis_value(parse_axis(stripped), dims)
        return None                          # 无映射 → 未解析（不杜撰）
    v = getattr(dims, field, None)
    return int(v) if v else None             # 0/None → 未解析


def _axis_value(f, dims):
    """一个轴 Factors（coeff × ∏ syms^mult）→ 整数值。任一单元未解析 → None。"""
    val = f.coeff
    for unit, mult in f.syms.items():
        u = _sym_value(unit, dims)
        if u is None:
            return None
        val *= u ** mult
    return val


# 公共名（T0→T1 交接要点6）：timesim.shard_rules 等跨包消费方用此名；
# 私名 _axis_value 保留（本模块内部/存量引用兼容）。
axis_value = _axis_value


def resolve_shape_elems(sym_shape, dims):
    """符号 shape 串 → 元素总数（各轴之积）。空/`?`/任一轴未解析 → None（不杜撰）。

    `~` 前缀（`sym_shape.NUMEL_ONLY`）= 只知元素数、不知轴结构 —— 元素数**仍精确**，故照常求值
    （轴结构只有 shape 推断的下游算子在意，字节记账不在意）。
    """
    if sym_shape is None:
        return None
    s, _numel_only = strip_numel_only(sym_shape.strip())
    s = s.strip()
    if s == "" or s == "?":
        return None
    axes = parse_shape(s)
    if not axes:
        return None
    total = 1
    for f in axes:
        v = _axis_value(f, dims)
        if v is None:
            return None
        total *= v
    return total


# ---------------------------------------------------------------------------
# 并行度本地化（TP/EP/CP/cp_kv）—— 语义**逐条对齐** `shape_eval.resolve_tensor`
# ---------------------------------------------------------------------------
#: 轴串里"引用序列符号 S"的判据。`shape_eval._refs_symbol` 走 AST Name 判定（`"S//4"` 命中、
#: `"kv_lora_rank"` 不误伤）；符号侧没有 AST，故按**标识符 token 精确等于 `S`** 判，等价。
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _axis_refs_S(f) -> bool:
    """一个轴（Factors）是否引用序列符号 S（含 `S//4`、`csa_window_size+S//4` 这类原子）。"""
    for unit in f.syms:
        if "S" in _IDENT.findall(unit):
            return True
    return False


def _shard_axis_index(axes, key):
    """`shard` 的键 → 轴下标。键可是轴下标（int），或**轴项串 / 该轴里的一个符号单元**（str）。

    与 `TensorRef.shard`（`{dim_index -> axis}`，model_spec.py:100）同语义，但符号侧的轴常是
    乘积（`n_heads·v_head_dim`），按符号名指定更贴源（"TP 切 head 维"）。找不到 → None。
    """
    if isinstance(key, int) and not isinstance(key, bool):
        return key if 0 <= key < len(axes) else None
    for i, f in enumerate(axes):
        if render_term(f) == key or key in f.syms:
            return i
    return None


def local_shape_elems(sym_shape, dims, pm, *, shard=None, is_weight=False,
                      cp_shard=True, cp_kv=False):
    """符号 shape 串 → **本地**元素数（TP/EP shard 与 CP 序列切分已除过）。

    这是契约 **B4**（"字节是 local 量"）在抽取侧的落地。语义参照现役手写路径
    `shape_eval.resolve_tensor`（shape_eval.py:81-120），逐条对齐：

      * `shard`：`{轴键: 'tp'|'ep'|'cp'|'sp'}` → 该轴 ÷`pm.degree(axis)`；
        **不整除即 `ValueError`**（shape_eval.py:86-88 同款 fail-loud，不许悄悄取整）；
      * CP：非权重、`cp_shard=True`、且不是 colossal 下的 `cp_kv` → **只切首个引用 S 的轴**
        （shape_eval.py:103-114 的 `break`：切 query/token 维，key/context 维保持全量）；
      * 任一轴解不出 → None（调用方据此进 `unresolved`，**绝不**拿 1 或 0 顶）。
    """
    if sym_shape is None:
        return None
    s, _numel_only = strip_numel_only(sym_shape.strip())
    s = s.strip()
    if s in ("", "?"):
        return None
    axes = parse_shape(s)
    if not axes:
        return None
    sizes = []
    for f in axes:
        v = _axis_value(f, dims)
        if v is None:
            return None
        sizes.append(v)
    for key, axis in (shard or {}).items():
        i = _shard_axis_index(axes, key)
        if i is None:
            raise ValueError(f"local_shape_elems: shard 键 {key!r} 在 shape {s!r} 里找不到对应轴")
        deg = pm.degree(axis)
        if sizes[i] % deg != 0:
            raise ValueError(f"{s} 轴 {key!r}={sizes[i]} 不被 {axis}={deg} 整除")
        sizes[i] //= deg
    cp = pm.degree("cp")
    method = getattr(getattr(pm, "pc", None), "context_parallel_method", "colossal")
    if cp > 1 and not is_weight and cp_shard and not (method == "colossal" and cp_kv):
        for i, f in enumerate(axes):
            if _axis_refs_S(f):
                if sizes[i] % cp != 0:
                    raise ValueError(f"{s} 序列轴 dim{i}={sizes[i]} 不被 cp={cp} 整除")
                sizes[i] //= cp
                break
    total = 1
    for v in sizes:
        total *= v
    return total


def _dtype_bytes(dtype, dims):
    return _DTYPE_BYTES.get(dtype, getattr(dims, "dtype_bytes", 2))


def save_bytes(save, dims):
    """一个 Save → 字节数（元素数 × dtype 字节）。shape 未解析 → None。"""
    elems = resolve_shape_elems(save.sym_shape, dims)
    if elems is None:
        return None
    return elems * _dtype_bytes(save.dtype, dims)


# ---------------------------------------------------------------------------
# `Detach` 别名去重 —— `stop_gradient` 不复制存储
# ---------------------------------------------------------------------------

def detach_aliases(dag) -> dict:
    """`{detach 产物名: 其输入名}`。

    `ops.stop_gradient(x)` **不复制存储**：它返回一个共享同一块内存、只是不参与 autograd 的
    别名（`Detach` 的 `PIN` 也是 `{"inputs": []}`，`bprop_rules.py:53`）。故字节记账里
    **产物与输入不得各计一份** —— 真源 6 处 detach（`csa.py:665/666/764/765/794/795`）里
    `x_detach`/`qr_detach` 与 `x`/`qr` 就是同一块内存，`derive_saves` 按名去重看不出这层。
    传递闭包（detach 的 detach）按链一路指向最原始的那个名字。
    """
    direct: dict = {}
    for n in dag.nodes:
        if n.op != "Detach" or not n.out or not n.ins:
            continue
        src = n.ins[0].split(":", 1)[0]
        dst = n.out.split(":", 1)[0]
        if dst != src:
            direct[dst] = src
    out: dict = {}
    for dst in direct:
        seen, cur = {dst}, direct[dst]
        while cur in direct and cur not in seen:
            seen.add(cur)
            cur = direct[cur]
        out[dst] = cur
    return out


# ---------------------------------------------------------------------------
# 权重（params）—— 与激活**结构性**分开（契约 W1..W6 + B4）
# ---------------------------------------------------------------------------

def dag_param_bytes(dag, dims, init_dims=None, pm=None):
    """DAG 的**权重**字节。返回 `{total_bytes, per_param, unresolved}`。

    权重从哪来（结构性，不是筛名字）：walker 把 `self.<attr>` 且 `__init__` 里是 `Parameter(...)`
    的操作数**刻意排除在 `OpNode.ins` 之外**，单列在 `OpDAG.param_operands`
    （`schema.py:56-59`；正因如此 `derive_saves` 结构上**不可能**把权重当激活 save —— 契约 W2/W4
    自动成立，实测 `FFNGroupedGEMM` 那 88 MiB 的病在本路径上不存在）。形状/dtype 由
    `init_dims.param_shapes` / `param_dtypes` 从 `Parameter(mint.empty((...), dtype=...))`
    逐字读出（`compressor.py:117`、`csa.py:589`、`deepseek_v4_hybrid_attention.py:139/159`）。

    `init_dims` 不给（或某个权重没形状）→ 该权重进 `unresolved`（**不猜**）。
    """
    shapes = dict(getattr(init_dims, "param_shapes", None) or {})
    dtypes = dict(getattr(init_dims, "param_dtypes", None) or {})
    names, seen = [], set()
    for p in getattr(dag, "param_operands", None) or ():
        nm = p.get("param") if isinstance(p, dict) else str(p)
        if nm and nm not in seen:
            seen.add(nm)
            names.append((nm, p.get("src") if isinstance(p, dict) else ""))
    per, unres, total = [], [], 0
    for nm, src in names:
        axes = shapes.get(nm)
        if not axes:
            unres.append((nm, "?", f"Parameter 形状未由 __init__ 求出（{src}）"))
            continue
        shp = "·".join(axes)
        elems = (local_shape_elems(shp, dims, pm, is_weight=True) if pm is not None
                 else resolve_shape_elems(shp, dims))
        if elems is None:
            unres.append((nm, shp, f"符号未解析（{src}）"))
            continue
        b = elems * _dtype_bytes(dtypes.get(nm), dims)
        per.append((nm, shp, dtypes.get(nm), b))
        total += b
    return {"total_bytes": total, "per_param": per, "unresolved": unres}


def dag_saved_bytes(dag, dims):
    """DAG 的 save-set 逐张量算字节。返回 {total_bytes, per_save, unresolved}。

    per_save: [(name, sym_shape, dtype, bytes)]；unresolved: [(name, sym_shape, reason)]
    （derive_saves 已按 name 去重，故此处天然去重）。
    """
    return dag_local_saved_bytes(dag, dims, None)


def dag_local_saved_bytes(dag, dims, pm, *, shard=None, cp_kv_names=(),
                          dedup_detach_aliases=True):
    """`dag_saved_bytes` 的**并行度感知**版本：每个 save 的字节按 TP/EP/CP 切分**除过**（契约 B4）。

    `pm is None` → 全局口径（与 `dag_saved_bytes` 逐字节相同，既有调用方不变）。
    `shard`：`{张量名: {轴键: 'tp'|'ep'|...}}`（轴键语义见 `local_shape_elems`）。
    `cp_kv_names`：attention KV 侧激活名（colossal CP 下保持 full-S，`shape_eval.py:105-106`）。
    `dedup_detach_aliases`：`Detach` 产物与其输入共享存储 → 只计一份（见 `detach_aliases`）。
    """
    alias = detach_aliases(dag) if dedup_detach_aliases else {}
    saves = list(derive_saves(dag))
    save_names = {s.name for s in saves}
    per_save, unresolved, aliased = [], [], []
    total = 0
    for s in saves:
        root = alias.get(s.name, s.name)
        elems = (resolve_shape_elems(s.sym_shape, dims) if pm is None else
                 local_shape_elems(s.sym_shape, dims, pm,
                                   shard=(shard or {}).get(s.name),
                                   cp_kv=s.name in set(cp_kv_names)))
        if elems is None:
            unresolved.append((s.name, s.sym_shape, "unknown-symbol-or-?"))
            continue
        b = elems * _dtype_bytes(s.dtype, dims)
        if root != s.name and root in save_names:
            # `stop_gradient` 的产物与其输入是**同一块存储**，且那块内存已由 root 记过一份
            # → 不重复计（顺序无关：判据是"root 也在 saves 里"，不是"root 已先算过"）。
            # 仍逐条列出（可见，不静默丢）。
            aliased.append((s.name, root, s.sym_shape, s.dtype, b))
            continue
        per_save.append((s.name, s.sym_shape, s.dtype, b))
        total += b
    out = {"total_bytes": total, "per_save": per_save, "unresolved": unresolved}
    if dedup_detach_aliases:
        out["detach_aliased"] = aliased
    return out
