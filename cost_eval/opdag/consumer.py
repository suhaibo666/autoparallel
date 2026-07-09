# cost_eval/opdag/consumer.py
"""T11 consumer 桥：把 op-DAG save-set 的**符号 shape** 用 `DimTable` 代入算**字节**。

`derive_saves(dag)` 产出的每个 `Save.sym_shape` 是符号串（"S·B·q_lora_rank"、"E·cap·H"…）。本桥
把每个符号 token 映射到 DimTable 的值、按 `sym_shape` 代数求积得元素数，再乘 dtype 字节。

**不杜撰**：任一 token 无映射 / 值为 0 / shape 含 `?` 轴 → 该 save 归入 `unresolved` 列表（供后续
标定 margin），绝不编造 size。纯符号代入，不 import mindspore/mindformers。
"""
from __future__ import annotations

import math

from .bprop_rules import derive_saves
from .sym_shape import parse_shape, _split_top


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
}

# dtype 串 → 字节。未知回退 dims.dtype_bytes（compute dtype）。
_DTYPE_BYTES = {
    "fp32": 4, "float32": 4,
    "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2,
    "fp8": 1, "int8": 1, "uint8": 1,
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
    """一个原子 token → 整数值。token 可能是和式 'a+b'（concat 出来的）。未知/0 → None。"""
    sym = sym.strip()
    parts = [p.strip() for p in _split_top(sym, "+")]
    if len(parts) > 1:                       # 和式单元：逐项求和
        vals = [_sym_value(p, dims) for p in parts]
        return sum(vals) if all(v is not None for v in vals) else None
    if sym == "cap":
        return _cap_value(dims)
    field = _SYM2FIELD.get(sym)
    if field is None:
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


def resolve_shape_elems(sym_shape, dims):
    """符号 shape 串 → 元素总数（各轴之积）。空/`?`/任一轴未解析 → None（不杜撰）。"""
    if sym_shape is None:
        return None
    s = sym_shape.strip()
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


def _dtype_bytes(dtype, dims):
    return _DTYPE_BYTES.get(dtype, getattr(dims, "dtype_bytes", 2))


def save_bytes(save, dims):
    """一个 Save → 字节数（元素数 × dtype 字节）。shape 未解析 → None。"""
    elems = resolve_shape_elems(save.sym_shape, dims)
    if elems is None:
        return None
    return elems * _dtype_bytes(save.dtype, dims)


def dag_saved_bytes(dag, dims):
    """DAG 的 save-set 逐张量算字节。返回 {total_bytes, per_save, unresolved}。

    per_save: [(name, sym_shape, dtype, bytes)]；unresolved: [(name, sym_shape, reason)]
    （derive_saves 已按 name 去重，故此处天然去重）。
    """
    per_save = []
    unresolved = []
    total = 0
    for s in derive_saves(dag):
        b = save_bytes(s, dims)
        if b is None:
            unresolved.append((s.name, s.sym_shape, "unknown-symbol-or-?"))
        else:
            per_save.append((s.name, s.sym_shape, s.dtype, b))
            total += b
    return {"total_bytes": total, "per_save": per_save, "unresolved": unresolved}
