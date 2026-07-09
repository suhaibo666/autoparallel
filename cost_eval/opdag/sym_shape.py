# cost_eval/opdag/sym_shape.py
"""符号 shape 代数(T9 底座)。

一个 shape = 一列**轴**,以 `·` 连接成串(如 "S·B·H");乘积轴用 `(...)` 包裹以消歧
(如 "S·B·(n_heads·qk_head_dim)" 是 3 轴,末轴为一个乘积)。每个轴 = 一个 `Factors`:
  * `coeff` —— 整数系数(如 `2·ffn_hidden` 的 2);
  * `syms`  —— 原子因子多重集(dict: 单元→重数)。单元是**纯符号**(H/n_heads)或**和式**
    (如 "qk_head_dim+v_head_dim",内部不再分解,规范化=按 `+` 项排序)。

为什么"和式保持原子":reshape 的 -1 需按乘积因子**消元**(整除),把和式当单个不可分因子,
消元就退化成多重集差 + 系数整除——干净且忠实。任何**不能干净消元/整除**的 → 返回 None
(交调用方保留 `?`,决不杜撰维度)。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# reshape 目标里的 -1(待消元)哨兵。
NEG1 = object()


# config 属性名 → 规范符号维度 token(供 __init__ 维度捕获 + reshape/split 表达式解析共用)。
# 未在此表的 config 属性在维度上下文里保留原名(flag=raw),决不映成数字。
CONFIG2SYM = {
    "hidden_size": "H",
    "ffn_hidden_size": "ffn_hidden",
    "moe_ffn_hidden_size": "moe_ffn",
    "num_attention_heads": "n_heads",
    "num_moe_experts": "E",
    "num_local_experts": "E",
    "q_lora_rank": "q_lora_rank",
    "kv_lora_rank": "kv_lora_rank",
    "qk_head_dim": "qk_head_dim",
    "qk_pos_emb_head_dim": "qk_pos_emb_head_dim",
    "v_head_dim": "v_head_dim",
}


@dataclass
class Factors:
    """一个轴 = coeff × ∏ syms(syms: 原子单元→重数)。"""
    coeff: int = 1
    syms: dict = field(default_factory=dict)

    def copy(self) -> "Factors":
        return Factors(self.coeff, dict(self.syms))


# ── 顶层 `·` / `+` 切分(尊重括号深度)────────────────────────────────────────
def _split_top(s: str, sep: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        # 确认这对括号是"最外层配对"(而非 "(a)·(b)")
        depth = 0
        matched = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    matched = False
                    break
        if matched:
            s = s[1:-1].strip()
        else:
            break
    return s


def _canon_sum(unit: str) -> str:
    """和式单元规范化:按 `+` 顶层项排序后重连(使相等和式串相等)。"""
    terms = [t.strip() for t in _split_top(unit, "+")]
    if len(terms) == 1:
        return terms[0]
    return "+".join(sorted(terms))


# ── 解析 ──────────────────────────────────────────────────────────────────────
def parse_axis(tok: str) -> Factors:
    """把一个轴串解析成 Factors。轴可含系数(2·ffn_hidden)、多因子(n_heads·qk_head_dim)、
    和式((qk_head_dim+v_head_dim))。"""
    tok = _strip_outer_parens(tok)
    f = Factors()
    for unit in _split_top(tok, "·"):
        unit = _strip_outer_parens(unit.strip())
        if unit == "":
            continue
        if _split_top(unit, "+")[1:]:            # 顶层有 `+` → 和式单元(原子)
            u = _canon_sum(unit)
            f.syms[u] = f.syms.get(u, 0) + 1
        elif unit.lstrip("-").isdigit():         # 整数 → 并入系数
            f.coeff *= int(unit)
        else:                                    # 纯符号
            f.syms[unit] = f.syms.get(unit, 0) + 1
    return f


def parse_shape(s: str) -> list[Factors]:
    s = s.strip()
    if s == "" or s == "?":
        return []
    return [parse_axis(a) for a in _split_top(s, "·")]


# ── 渲染 ──────────────────────────────────────────────────────────────────────
def _render_unit(unit: str) -> str:
    return f"({unit})" if _split_top(unit, "+")[1:] else unit


def render_term(f: Factors) -> str:
    """一个轴渲成乘积串(不含最外层括号):2·ffn_hidden / n_heads·(qk_head_dim+v_head_dim) / S。
    单一单元(纯符号或和式)且系数为 1 时**裸出**(不加括号,外层由 render_shape 决定)。"""
    units: list[str] = []
    for unit in sorted(f.syms):
        units.extend([unit] * f.syms[unit])
    if f.coeff == 1 and len(units) == 1:
        return units[0]
    parts: list[str] = []
    if f.coeff != 1 or not units:
        parts.append(str(f.coeff))
    parts.extend(_render_unit(u) for u in units)   # 乘积中的和式单元加括号
    return "·".join(parts) if parts else "1"


def _is_composite(f: Factors) -> bool:
    """作为 shape 中的一个轴时是否需要外层括号(=乘积/和式)。纯整数轴不加括号。"""
    num_units = sum(f.syms.values())
    if num_units == 0:
        return False                               # 纯整数轴(如 reshape 目标里的 2)
    if f.coeff != 1 or num_units > 1:
        return True
    (unit,) = list(f.syms.keys())
    return bool(_split_top(unit, "+")[1:])         # 单个和式单元 → 也要括号


def render_shape(axes: list[Factors]) -> str:
    out = []
    for f in axes:
        t = render_term(f)
        out.append(f"({t})" if _is_composite(f) else t)
    return "·".join(out)


# ── 代数 ──────────────────────────────────────────────────────────────────────
def mul(a: Factors, b: Factors) -> Factors:
    r = a.copy()
    r.coeff *= b.coeff
    for u, c in b.syms.items():
        r.syms[u] = r.syms.get(u, 0) + c
    return r


def product_of(axes: list[Factors]) -> Factors:
    r = Factors()
    for f in axes:
        r = mul(r, f)
    return r


def divide(total: Factors, denom: Factors) -> Factors | None:
    """total / denom(消元)。要求 denom 的每个因子在 total 里数量足够、系数整除;否则 None。"""
    if denom.coeff == 0 or total.coeff % denom.coeff != 0:
        return None
    syms = dict(total.syms)
    for u, c in denom.syms.items():
        if syms.get(u, 0) < c:
            return None
        syms[u] -= c
        if syms[u] == 0:
            del syms[u]
    return Factors(total.coeff // denom.coeff, syms)


def floordiv(a: Factors, n: int) -> Factors | None:
    """轴整除标量 n:仅当系数整除(如 2·ffn_hidden // 2 = ffn_hidden);否则 None。"""
    if n == 0 or a.coeff % n != 0:
        return None
    return Factors(a.coeff // n, dict(a.syms))


def add(a: Factors, b: Factors) -> Factors:
    """两个轴相加(concat 末轴 / q_head_dim=qk_head_dim+qk_pos_emb_head_dim):合成一个和式原子单元。"""
    ta, tb = render_term(a), render_term(b)
    unit = _canon_sum(f"{ta}+{tb}")
    return Factors(1, {unit: 1})


def resolve_reshape(input_axes: list[Factors], target) -> list[Factors] | None:
    """按 reshape 目标(Factors 列表,-1 用 NEG1 哨兵)算出输出各轴;单个 -1 靠总积消元填补。
    无法解析(多个 -1 / 消元不干净)→ None。"""
    neg_positions = [i for i, t in enumerate(target) if t is NEG1]
    if not neg_positions:
        return [t.copy() for t in target]
    if len(neg_positions) != 1:
        return None
    total = product_of(input_axes)
    known = product_of([t for t in target if t is not NEG1])
    filled = divide(total, known)
    if filled is None:
        return None
    out = [t if t is not NEG1 else filled for t in target]
    return [t.copy() for t in out]
