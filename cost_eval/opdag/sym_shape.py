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
    """该单元是否是**表达式原子**(和式 / 差式 / 整除式)—— 出现在乘积里时要加括号以免读歧义。

    **差式也算**(2026-07-29):`sub()` 产出的 `v_head_dim-64` 此前不加括号,进乘积后成为
    `B·v_head_dim-64` —— `consumer._sym_value` 的差式档按**最后一个** `-` 左右切,会把它读成
    `B·v_head_dim − 64`(真值是 `B·(v_head_dim−64)`)。加括号即消歧。
    """
    return bool(_split_signed(unit)[1:]) or "//" in unit


def _canon_sum(unit: str) -> str:
    """和式单元规范化:按 `+` 顶层项排序后重连(使相等和式串相等)。"""
    terms = [t.strip() for t in _split_top(unit, "+")]
    if len(terms) == 1:
        return terms[0]
    return "+".join(sorted(terms))


def top_floordiv(sym: str) -> int:
    """最外层(depth 0)**最后**一个 `//` 的下标;没有则 -1。右结合,与 `rpartition` 同向;
    但**括号感知** —— `(a)//((b)//c)` 的最外层是第一个 `//`,`rpartition` 会切错。

    (原实现在 `consumer._top_floordiv`,2026-07-29 上提到本模块,规范化与求值共用同一条切法。)
    """
    depth, last, i = 0, -1, 0
    while i < len(sym):
        c = sym[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "/" and depth == 0 and sym[i + 1:i + 2] == "/":
            last = i
            i += 1
        i += 1
    return last


def _split_signed(s: str) -> list:
    """把表达式串按**顶层** `+`/`-` 拆成 `[(符号, 项串), ...]`(括号内不拆)。

    顶层出现 `//` 时**整串当一项**返回 —— 与 `consumer._sym_value` 的优先级一致
    (`a-b//c` 读作 `(a-b)//c`),避免把整除式的被除数拆散。
    """
    if top_floordiv(s) >= 0:
        return [(1, s.strip())]
    out: list = []
    depth, cur, sign = 0, [], 1
    for ch in s:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif depth == 0 and ch in "+-":
            head = "".join(cur).strip()
            if head:
                out.append((sign, head))
                sign = 1 if ch == "+" else -1
                cur = []
            elif ch == "-":                     # 前导一元负号
                sign = -sign
            # 前导 `+` 无意义,忽略
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append((sign, tail))
    return out or [(1, s.strip())]


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
    # 单个和式/差式单元 → 要括号;**单个单元内部含顶层 `·` 也要**(2026-07-29 实测的一个静默错):
    # `floordiv` 造的整除原子 `n_heads·v_head_dim//8` 是**一个**单元,但它内部有顶层 `·`,
    # 裸着放进 shape 串 `8·1024·n_heads·v_head_dim//8` 后,`parse_shape` 按 `·` 切会得到
    # **4 条轴**而不是 3 —— rank 凭空多一条,`permute`/`transpose` 的置换校验随之判错。
    return bool(_split_signed(unit)[1:]) or bool(_split_top(unit, "·")[1:])


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
    return f"({t})" if (_split_top(t, "·")[1:] or _split_signed(t)[1:]) else t


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


def divide_expr(total: Factors, denom: Factors) -> Factors | None:
    """`total / denom` —— 先走精确消元 `divide`;**约不干净时形成整除原子** `(total)//(denom)`。

    为什么这不是杜撰:reshape 的 `-1` 位按定义**就是** `numel(输入) // ∏(其余目标维)`,
    两边都是已由源解出的符号表达式,值完全由 DimTable 决定 —— 与 `floordiv` 第 3 档
    (`S//4`)是同一条"表达式保形"的路子,只是分母也可以是符号。

    动机(实测):`compressor.py:196-203`
        cutoff = (sq // ratio) * ratio ;  n_compressed = cutoff // ratio
        kv = self.reshape(kv, (n_compressed, ratio, b, -1))
    已知积里是 `S//4`、总积里是 `S` —— `sym_shape` 不知道 `4·(S//4) == S`(那要 4 | S),
    于是消元失败、整条压缩链退成"只知元素数",`:216` 的按轴归约随之被拒。
    形成整除原子后由 `consumer._sym_value` 用 DimTable 求值(不整除 → None,不取整)。
    """
    d = divide(total, denom)
    if d is not None:
        return d
    if denom.coeff == 0:
        return None
    return Factors(1, {f"({render_term(total)})//({render_term(denom)})": 1})


# ── 整除事实 + 表达式规范化(2026-07-29,「符号轴结构恢复」)──────────────────────
#
# 动机(`docs/next_fix_adversarial_review_2026-07-28.md` §3.5):四个级联根里三个归约到
# 同一项能力缺口 —— 符号层**只会往前累积表达式,从不化简**,于是:
#   * `csa.py:482` 的 `kv_flat` 末轴是 `64+v_head_dim-64`(源侧 split→cat 原样拼回),
#     串上带着两个互相抵消的 64 → 下游的 advanced-index / BMM 认不出它就是 `v_head_dim`;
#   * `compressor.py:203` 的 `-1` 位消元约不干净(`S` vs `4·(S//4)`)→ 整条压缩链退 `~`,
#     `:216` 的按轴归约随之被拒。
#
# 两条纪律:
#   ① **只做恒等变形**。折叠前后代入同一份 DimTable 求值必须逐字节相同(测试钉住)。
#   ② **整除身份要先核验**。`n·(X//n) == X` 仅当 `n | X`;该前提由 `DimTable` 的**实际值**
#      核验后才生成为一条 `DivFacts`,核验不过就**没有**这条事实 —— 表达式保持保守形态,
#      下游按轴改形的算子照旧拒绝并记 `file:line`。**绝不**拿"大概能整除"往下算。

@dataclass(frozen=True)
class DivFacts:
    """**已核验**的整除事实集合。每条 `(n, term)` 表示 `n | term`(term 是一个符号串)。

    `rejected` 记下**核验不通过**的候选(`(n, term, 实际值)`),供调用方显式报告 ——
    "这条身份不成立"必须是可见的,而不是静默退化。
    """
    facts: frozenset = frozenset()
    rejected: tuple = ()

    def has(self, n: int, term: str) -> bool:
        return (int(n), str(term)) in self.facts


#: 空事实集(缺省)。**没有事实时一律不折叠整除身份** —— 这是"宁 `?` 勿错"的默认档。
NO_DIV_FACTS = DivFacts()


def assert_divisible(n: int, term: str, values: dict) -> bool:
    """显式断言 `n | term`;**不成立就 fail-loud**(ValueError,带实际值)。

    给"我确信这条前提成立"的调用点用 —— 比如源侧逐字写着
    `assert seq_length % compress_ratio == 0`。值查不到也 fail-loud(不许拿"查不到"当通过)。
    """
    v = values.get(str(term))
    if v is None:
        raise ValueError(
            f"sym_shape.assert_divisible: 断言 {n} | {term} 但 {term!r} 没有已知值 —— "
            "不许把'查不到'当成'成立'")
    if int(n) <= 0 or int(v) % int(n) != 0:
        raise ValueError(
            f"sym_shape.assert_divisible: {n} | {term} **不成立**（{term}={int(v)}，"
            f"{int(v)} % {n} = {int(v) % int(n) if int(n) else '除零'}）")
    return True


def div_facts_from_values(pairs, values: dict) -> DivFacts:
    """按**实际值**核验一批候选 `(n, term)`,只把成立的那些收成事实。

    * 值未知 → 既不成为事实、也不算被拒(无从判定,保守跳过);
    * 值已知且不整除 → 进 `rejected`(可见),**不**成为事实。
    """
    ok, bad = set(), []
    for n, term in pairs:
        n = int(n)
        v = values.get(str(term))
        if v is None or n <= 0:
            continue
        if int(v) % n == 0:
            ok.add((n, str(term)))
        else:
            bad.append((n, str(term), int(v)))
    return DivFacts(frozenset(ok), tuple(bad))


_NORM_MAX_DEPTH = 12          # 递归护栏(病态串不许把栈打穿)
_NORM_MAX_ROUNDS = 8          # 整除身份重写的迭代上限


def _lin_paren(t: str) -> str:
    """和式里的一项:含顶层 `·` / `+` / `-` / `//` 就加括号(否则往返解析会读错分组)。"""
    if _split_top(t, "·")[1:] or _split_signed(t)[1:] or top_floordiv(t) >= 0:
        return f"({t})"
    return t


def _render_linear(items: list, const: int) -> str:
    """`[(系数, syms), ...] + 常数` → 规范和式串:正项(升序)在前,负项(升序)在后。

    再过一遍 `_canon_sum` ⇒ 与 `parse_axis` 的规范化**同一个不动点**(幂等)。
    """
    pos, neg = [], []
    for c, syms in items:
        t = _lin_paren(render_term(Factors(abs(c), dict(syms))))
        (pos if c > 0 else neg).append(t)
    if const > 0:
        pos.append(str(const))
    elif const < 0:
        neg.append(str(-const))
    pos.sort()
    neg.sort()
    return _canon_sum("+".join(pos) + "".join("-" + t for t in neg))


def _expand_linear(ft: Factors, facts: DivFacts, depth: int):
    """`Factors` → `(常数, [(系数, syms), ...])`,把**线性原子单元**递归摊平(乘法分配律)。

    为什么必须摊平:`sub`/`add` 会把线性式包成一个**不可分的原子串**,嵌套一层就藏住了
    可抵消的项 —— `(v_head_dim-64)+64` 若不摊平,外层只看见 `v_head_dim-64` 与 `64`
    两个不同的 key,抵消不掉。摊平后按单项归并,`-64` 与 `+64` 才碰得上。
    """
    if depth > _NORM_MAX_DEPTH:
        return (ft.coeff, []) if not ft.syms else (0, [(ft.coeff, dict(ft.syms))])
    if not ft.syms:
        return ft.coeff, []
    if len(ft.syms) == 1:
        ((unit, mult),) = ft.syms.items()
        if mult == 1 and top_floordiv(unit) < 0:
            parts = _split_signed(_strip_outer_parens(unit))
            if len(parts) > 1 or parts[0][0] < 0:
                const, mons = 0, []
                for s, t in parts:
                    c2, m2 = _expand_linear(_norm_unit(t, facts, depth + 1), facts, depth + 1)
                    const += s * ft.coeff * c2
                    mons.extend((s * ft.coeff * c, syms) for c, syms in m2)
                return const, mons
    return 0, [(ft.coeff, dict(ft.syms))]


def _norm_linear(terms: list, facts: DivFacts, depth: int) -> Factors:
    const = 0
    groups: dict = {}
    for sign, t in terms:
        c0, mons = _expand_linear(_norm_unit(t, facts, depth + 1), facts, depth + 1)
        const += sign * c0
        for c, syms in mons:
            key = render_term(Factors(1, dict(syms)))
            g = groups.setdefault(key, [0, dict(syms)])
            g[0] += sign * c
    items = [(c, syms) for c, syms in groups.values() if c != 0]
    if not items:
        return Factors(coeff=const)
    if len(items) == 1 and const == 0 and items[0][0] > 0:
        c, syms = items[0]
        return Factors(c, dict(syms))
    return Factors(1, {_render_linear(items, const): 1})


def _norm_floordiv(nb: Factors, nd: Factors, facts: DivFacts) -> Factors:
    """规范化后的 `nb // nd`:能**精确消元**就消,否则原样重造整除原子。

    重造的串形与 `floordiv`(整数分母)/ `divide_expr`(符号分母)**逐字一致** ——
    什么都没折叠时,规范化必须是恒等映射(报表/台账不许无谓地抖)。
    """
    q = divide(nb, nd)
    if q is not None:
        return q
    if not nd.syms:
        n = nd.coeff
        if n == 0:
            return Factors(1, {f"{render_term(nb)}//0": 1})
        if not nb.syms:
            return Factors(coeff=nb.coeff // n)
        return Factors(1, {f"{render_term(nb)}//{n}": 1})
    return Factors(1, {f"({render_term(nb)})//({render_term(nd)})": 1})


def _norm_unit(unit: str, facts: DivFacts, depth: int = 0) -> Factors:
    """一个原子单元串 → 规范化后的 `Factors`(可能不再是原子:折叠掉了就散成乘积)。"""
    u = _strip_outer_parens(str(unit).strip())
    if u == "" or depth > _NORM_MAX_DEPTH:
        return Factors(1, {u: 1}) if u else Factors()
    cut = top_floordiv(u)
    if cut >= 0:
        nb = normalize_axis(parse_axis(u[:cut]), facts, depth + 1)
        nd = normalize_axis(parse_axis(u[cut + 2:]), facts, depth + 1)
        return _norm_floordiv(nb, nd, facts)
    terms = _split_signed(u)
    if len(terms) > 1 or terms[0][0] < 0:
        return _norm_linear(terms, facts, depth)
    if _split_top(u, "·")[1:]:
        return normalize_axis(parse_axis(u), facts, depth + 1)
    if u.lstrip("-").isdigit():
        return Factors(coeff=int(u))
    return Factors(1, {u: 1})


def _apply_div_facts(f: Factors, facts: DivFacts, depth: int) -> Factors:
    """在一个**乘积**里用已核验的整除身份消元:`coeff` 含 `n` 且 `n | X` ⇒ `n·(X//n) → X`。

    真源:`compressor.py:196-203`
        cutoff = (sq // ratio) * ratio ;  n_compressed = cutoff // ratio
        kv = self.reshape(kv, (n_compressed, ratio, b, -1))
    `-1` 位的分母是 `n_compressed·ratio·b` = `(S//ratio)·ratio·B`。`ratio | S` 成立时它
    **恒等于** `B·S`,消元退化成精确的多重集差;不成立时源侧 `:197-199` 真的会截断,
    身份不成立 —— 所以这一步**必须**由事实门控。
    """
    if not facts.facts:
        return f
    for _ in range(_NORM_MAX_ROUNDS):
        hit = None
        for unit, mult in f.syms.items():
            if mult <= 0:
                continue
            cut = top_floordiv(unit)
            if cut < 0:
                continue
            denom = unit[cut + 2:].strip()
            if not denom.lstrip("-").isdigit():
                continue
            n = int(denom)
            base = _strip_outer_parens(unit[:cut].strip())
            if n > 0 and f.coeff % n == 0 and facts.has(n, base):
                hit = (unit, n, base)
                break
        if hit is None:
            return f
        unit, n, base = hit
        syms = dict(f.syms)
        syms[unit] -= 1
        if syms[unit] == 0:
            del syms[unit]
        f = mul(Factors(f.coeff // n, syms), _norm_unit(base, facts, depth + 1))
    return f


def normalize_axis(f: Factors, facts: DivFacts = NO_DIV_FACTS, depth: int = 0) -> Factors:
    """一个轴的规范化:逐单元化简 + 整除身份消元。**恒等变形**,值不变。"""
    if depth > _NORM_MAX_DEPTH:
        return f.copy()
    acc = Factors(coeff=f.coeff)
    for unit, mult in f.syms.items():
        nu = _norm_unit(unit, facts, depth + 1)
        for _ in range(mult):
            acc = mul(acc, nu)
    return _apply_div_facts(acc, facts, depth)


def normalize_axes(axes, facts: DivFacts = NO_DIV_FACTS) -> list:
    return [normalize_axis(a, facts) for a in (axes or ())]


def resolve_reshape(input_axes: list[Factors], target, *,
                    allow_expr: bool = False) -> list[Factors] | None:
    """按 reshape 目标(Factors 列表,-1 用 NEG1 哨兵)算出输出各轴;单个 -1 靠总积消元填补。
    无法解析(多个 -1 / 消元不干净)→ None。

    `allow_expr=True`(2026-07-28):消元不干净时改用 `divide_expr` 形成整除原子,
    而不是整条放弃(见 `divide_expr` 的论证)。缺省关 —— 既有调用方逐字不变。"""
    neg_positions = [i for i, t in enumerate(target) if t is NEG1]
    if not neg_positions:
        return [t.copy() for t in target]
    if len(neg_positions) != 1:
        return None
    total = product_of(input_axes)
    known = product_of([t for t in target if t is not NEG1])
    filled = divide_expr(total, known) if allow_expr else divide(total, known)
    if filled is None:
        return None
    out = [t if t is not NEG1 else filled for t in target]
    return [t.copy() for t in out]
