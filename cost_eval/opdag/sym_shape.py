# cost_eval/opdag/sym_shape.py
"""符号 shape 代数(T9 底座)。

一个 shape = 一列**轴**,以 `·` 连接成串(如 "S·B·H");乘积轴用 `(...)` 包裹以消歧
(如 "S·B·(n_heads·qk_head_dim)" 是 3 轴,末轴为一个乘积)。每个轴 = 一个 `Factors`:
  * `coeff` —— 整数系数(如 `2·ffn_hidden` 的 2);
  * `syms`  —— 原子因子多重集(dict: 单元→重数)。单元是**纯符号**(H/n_heads)、**和式**
    (如 "qk_head_dim+v_head_dim")或**整除式**(如 "S//4"),内部不再分解;
    规范化 = 和式按 `+` 项排序、整除式保持源写法。

为什么"和式/整除式保持原子":reshape 的 -1 需按乘积因子**消元**(整除),把它们当单个不可分
因子,消元就退化成多重集差 + 系数整除——干净且忠实。任何**不能干净消元/整除**的 → 返回 None
(交调用方保留 `?`,决不杜撰维度)。

整除原子(`//`)的动机(2026-07-25,dsv4 压缩链):`compressor.py:196` `cutoff = (sq // ratio)
* ratio`、`:201` `n_compressed = cutoff // ratio`、`csa.py:762` 的压缩 KV 序列长度都是
`S // compress_ratio`。`floordiv` 此前只在**系数整除**时可解(`2·ffn_hidden // 2`),
`S // 4` 直接返回 None → 整条压缩链 shape 全 `?`。现在系数不整除时改为形成原子单元
`"<term>//<n>"`,由 `consumer._sym_value` 按 DimTable 值做**整数除法**求值。
注意这是"表达式保形",**不是**编造尺寸:`S//4` 的值仍完全由 DimTable 的 S 决定。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# reshape 目标里的 -1(待消元)哨兵。
NEG1 = object()

#: **只知元素数、不知轴结构**的 shape 串前缀(2026-07-25)。
#:
#: 为什么需要这一档:某些原语的轴信息在 DAG 里**没有被记下来**(walker attrs 缺 `dim`/轴序)——
#: 如 `mint.cat(..., dim=-1)`(`compressor.py:243`)只记了 `view="concat"`,没记轴。这种情况下:
#:   * 元素数**仍然精确可知**(concat 的 numel = 各输入 numel 之和,与轴无关);
#:   * 轴结构**不可知** → 任何"猜一个轴"的做法都可能算错(实测:`cat([kv_nope, kv_pe], -1)`
#:     的两个输入末轴不同,按轴 0 合并会给出错的 numel)。
#: 契约(`liveness/contract.py` B1)只要 `local_numel > 0`,故"只知 numel"是**可用**的;
#: 但下游需要轴结构的算子(MatMul 换末轴 / split / expand_dims / tile)遇到它必须**拒绝**
#: (记 unresolved),不许拿着一个假轴序往下算。串形如 `"~S·B·H"` —— `~` 让它在任何 dump /
#: 报表里**一眼可见**,不会被误当成普通 shape。
NUMEL_ONLY = "~"


def mark_numel_only(shape: str) -> str:
    """把一个 shape 串标记为"只知元素数"(幂等)。"""
    return shape if shape.startswith(NUMEL_ONLY) else NUMEL_ONLY + shape


def strip_numel_only(shape: str):
    """→ `(裸 shape 串, 是否只知元素数)`。"""
    if isinstance(shape, str) and shape.startswith(NUMEL_ONLY):
        return shape[len(NUMEL_ONLY):], True
    return shape, False


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
    # ── DSv4-Flash(pynative dsa_indexer + compressor + CSA)的维度符号 ────────────────
    # ⚠ 本表的**键是 config 属性名**(`config.<键>`),值是规范符号 token。dsv4 侧二者不同名
    # (config 里叫 `dsa_indexer_n_heads`、源码里的 self 名叫 `index_n_heads`),必须按属性名收。
    # 每条都实测出现在**维度上下文**里(reshape/split 表达式或 build_module 的 output_size):
    #   dsa_indexer_n_heads / dsa_indexer_head_dim —— `indexer.py:94-95`
    #     `self.index_n_heads = config.dsa_indexer_n_heads`;`:118-119`
    #     `output_size=self.index_n_heads * self.index_head_dim`;
    #   dsa_indexer_topk —— `indexer.py:96` / `:262` `self.topk(index_scores, self.index_topk, ...)`
    #     的 topk 轴(csa.py:474/501 的 O(S·topk) mask/gather 都按它);
    #   csa_window_size —— `csa.py:571-573` `self.window_size = config.csa_window_size`
    #     (滑窗上下文位置维;与 `S//ratio` 相加成 TOPK_DIM);
    #   o_groups / o_lora_rank —— `deepseek_v4_hybrid_attention.py:136-137/146`
    #     `input_size=o_groups * o_lora_rank`;`:274-276` `o_chunk = ... // o_groups`;
    #   hc_mult —— mHC 的残差流倍数(消费侧 `DimTable.num_residual_streams`)。
    # 不在本表 = 在 dims_ctx 里会被 `init_dims._KNOWN_DIM_SYMS` 滤掉 →
    # shape 推断解不出 `self.<attr>` → 该张量保 `?`(而非拿个数糊上去)。
    "dsa_indexer_n_heads": "index_n_heads",
    "dsa_indexer_head_dim": "index_head_dim",
    "dsa_indexer_topk": "index_topk",
    "csa_window_size": "csa_window_size",
    "o_groups": "o_groups",
    "o_lora_rank": "o_lora_rank",
    "hc_mult": "hc_mult",
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


def _is_atom_expr(unit: str) -> bool:
    """该单元是否是**表达式原子**(和式或整除式)—— 出现在乘积里时要加括号以免读歧义。"""
    return bool(_split_top(unit, "+")[1:]) or "//" in unit


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
    return f"({unit})" if _is_atom_expr(unit) else unit


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
    """轴整除标量 n。三档(**都不杜撰**,值仍完全由 DimTable 决定):

      1. 系数整除 → 直接约掉(`2·ffn_hidden // 2 = ffn_hidden`)——既有行为逐字不变;
      2. 纯整数轴 → 直接整数除(`8 // 4 = 2`);
      3. 否则 → 形成**整除原子** `"<term>//<n>"`(`S // 4` → `S//4`),由
         `consumer._sym_value` 按 DimTable 值做整数除法。
         动机:`compressor.py:196/201`、`csa.py:762` 的压缩序列长度 = `S // compress_ratio`;
         第 3 档缺失时整条压缩链 shape 全 `?`(见模块 docstring)。

    `n == 0` → None(不定义)。
    """
    if n == 0:
        return None
    if a.coeff % n == 0:
        return Factors(a.coeff // n, dict(a.syms))
    if not a.syms:                                   # 纯整数轴:精确整数除
        return Factors(a.coeff // n)
    return Factors(1, {f"{render_term(a)}//{n}": 1})


def _sum_term(f: Factors) -> str:
    """和式的一项在**串形式**里的写法:**乘积项必须加括号**。

    为什么(2026-07-25 实测的一个静默错):`_split_top(s, "·")` 只认括号深度,不认 `+`。
    若把两个乘积项裸着相加成 `"B·S·v_head_dim+B·S·v_head_dim"`,再 `parse_axis` 时会被按 `·`
    切成 `["B", "S", "v_head_dim+B", "S", "v_head_dim"]` —— 中间冒出个假和式单元
    `v_head_dim+B`,值算成 `B·(B+v_head_dim)·S·S·v_head_dim`(实测把一个 8 MiB 的 concat
    产物算成 8·10⁶ MiB)。旧代码只在**单符号**之间相加(`qk_head_dim+v_head_dim`)故没暴露;
    concat 的"元素数 = 各输入元素数之和"要相加**乘积**,必须补括号才能往返。
    """
    t = render_term(f)
    return f"({t})" if (_split_top(t, "·")[1:] or _split_top(t, "+")[1:]) else t


def add(a: Factors, b: Factors) -> Factors:
    """两个轴相加(concat 末轴 / q_head_dim=qk_head_dim+qk_pos_emb_head_dim):合成一个和式原子单元。

    **两侧都是纯常数时直接折叠**(2026-07-28):和式原子单元是给"含符号、约不掉"的情形准备的
    (整个串被当成一个不可解的符号名);两个字面量相加的结果是**已知整数**,包成 `((1+1)+1)`
    这种原子单元反而让 `consumer._sym_value` 查不到映射 → 落 unresolved。真源:
    `hyper_connection.py:408` `concat((alpha_pre, alpha_post, alpha_res), -1)` 三个
    `Parameter` 形状都是 `(1,)`,末轴相加应当就是 3。
    """
    if not a.syms and not b.syms:
        return Factors(coeff=a.coeff + b.coeff)
    unit = _canon_sum(f"{_sum_term(a)}+{_sum_term(b)}")
    return Factors(1, {unit: 1})


def sub(a: Factors, b: Factors) -> Factors | None:
    """两个轴相减,合成一个**差式原子单元**(2026-07-28)。

    真源:`indexer.py:179-180`
    `self.split(q, [self.index_head_dim - self.qk_pos_emb_head_dim, self.qk_pos_emb_head_dim], -1)`
    与 `deepseek_v4_hybrid_attention.py:204` `nope_dim = self.config.v_head_dim - pos_dim`。
    两侧都是纯常数时直接折叠(同 `add`);差 ≤ 0 → None(轴长非正 = 解错了,宁 `?` 勿错)。

    差式**不排序**(减法不可交换),由 `consumer._sym_value` 按 DimTable 求值 —— 与 `//`
    原子同一条路子:表达式保形,值仍完全由 DimTable 决定,不是编造。
    """
    if not a.syms and not b.syms:
        d = a.coeff - b.coeff
        return Factors(coeff=d) if d > 0 else None
    return Factors(1, {f"{_sum_term(a)}-{_sum_term(b)}": 1})


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
