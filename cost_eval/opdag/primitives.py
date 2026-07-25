# cost_eval/opdag/primitives.py
"""**唯一**的「原语名 → op 类型」表(路线 B P0#2,2026-07-25)。

## 为什么需要这张表

`init_binder._CLS2OP` 认的是 `parallel_core/training_graph/` 的惯用法——**类实例化**
(`self.reshape = Reshape()`,一个 `ast.Call`,按类名可查)。DSv4-Flash 真机跑的 `pynative/`
用的是**另一套成文约定**:`self.reshape = mint.reshape`(**裸函数别名**,不是 Call)
——见 `csa.py:628-630` 注释「Alias the non-trivial mint ops used in construct/forward per the
fine-grained-recompute convention (RFC §3.1 #9)」。实测(`docs/opdag_walker_core_2026-07-25.md`
§1)dsv4 链上 **30 个不同裸别名目标 / 106 个 `self.X(...)` 调用点** +
**33 个不同 `<ns>.<fn>(...)` 自由调用路径 / 116 个调用点**,一个都绑不上。

## 三条纪律

1. **只收实测触发点**。每条表项都能在权威快照里指到 `file:line`(与 `construct_walker.FREE_CALL_MAP`
   的既有纪律一致)。不预置「以后可能有」的名字——没实测过的语义等于猜。
2. **未知原语 fail-loud,绝不归类**。归错类 = saved 集静默错 = 这件事要杀的那类 bug。
   `lookup_alias` 对未知名抛 `UnknownPrimitiveError`(带 file:line + 该加什么表项)。
3. **op 类型的 bprop 语义按教科书 VJP,不按「大概像谁」**。每条表项的注释写清「反向要读什么」,
   因为这直接决定 `bprop_rules.derive_saves` 存哪些张量 = 直接决定字节。

## 归类判据(逐类的反向事实)

| op 类型 | 反向读什么 | 归入的原语 |
|---|---|---|
| `View` | 什么都不读(反向是逆视图 / 切片 / 拼接) | reshape permute transpose squeeze unsqueeze split chunk cat concat tile roll broadcast_to flatten |
| `Cast` | 自身不存(fp32 buffer 归下游消费者 pin) | ops.cast |
| `Elementwise{linear:True}` | 什么都不读(dy 直通 / 广播) | add sub neg sum mean triu cumsum |
| `Elementwise{linear:False}` | 两个操作数(dy·f'(x) 需要 x) | mul div exp log sqrt rsqrt abs pow clamp maximum minimum max min |
| `Activation` | 输入(dy·f'(x)) | relu sigmoid silu gelu softplus |
| `Softmax` | **输出**(dy·y − y·Σ) | softmax log_softmax |
| `MatMul`/`BMM` | 两个操作数(dA=dy·Bᵀ / dB=Aᵀ·dy) | matmul bmm linear |
| `IndexSelect` | **index**(反向是 scatter_add 到零张量) | gather |
| `Where` | **cond mask**(反向按 mask 分流) | where |
| `Scatter` | **index** | scatter |
| `TopK` | **indices**(自己的第 2 个输出) | topk |
| `Compare` | 什么都不读(**不可微**,bool 产出) | eq ne gt ge lt le isfinite isnan logical_* any all argsort |
| `Constant` | 什么都不读(常量产出,无梯度) | arange zeros ones full empty *_like eye |
| `Detach` | 什么都不读(梯度到此为止,但**保边**) | ops.stop_gradient |
| `ShapeOf` | **不发射节点**(产出的是标量元组,不是张量) | ops.shape |
"""
from __future__ import annotations


class UnknownPrimitiveError(ValueError):
    """裸别名 / 自由调用命中了一个**表里没有**的原语。

    刻意继承 `ValueError`——与 walker 其余 fail-loud 同一漏斗
    (`crosscheck._check_segment` 的 `except Exception` 会把它计进 `extraction_failures`)。
    """


# `ShapeOf` 是**伪 op 类型**:命中它的调用不发射节点,只做「标量轴解包」记账
# (`sq, b, _ = self.shape(x)` 产出的是 python int 元组,不是张量——发射节点就是**造假节点**)。
SHAPE_OF = "ShapeOf"

_VIEW = ("View", {})
_LIN = ("Elementwise", {"linear": True})
_NONLIN = ("Elementwise", {"linear": False})
_REDUCE_LIN = ("Elementwise", {"linear": True, "reduce": True})
_REDUCE_NONLIN = ("Elementwise", {"linear": False, "reduce": True})
_CONST = ("Constant", {})
_CMP = ("Compare", {})

# ── 主表:点号路径 → (op 类型, attrs)。定位符见每组注释。 ────────────────────────────────
# `attrs["view"]` 子类型供 `_view_capture` 抠变换元信息(reshape 目标 / split 尺寸 / perm)。
PRIMITIVES: dict[str, tuple[str, dict]] = {
    # —— 纯视图(反向不新增激活)。csa.py:631-634 / indexer.py:146-149 / compressor.py:137-145 ——
    "mint.reshape":      ("View", {"view": "reshape"}),
    "mint.view":         ("View", {"view": "reshape"}),
    "mint.permute":      ("View", {"view": "permute"}),
    "mint.transpose":    ("View", {"view": "transpose"}),
    "mint.squeeze":      ("View", {"view": "squeeze"}),
    "mint.unsqueeze":    ("View", {"view": "expand_dims"}),
    "mint.split":        ("View", {"view": "split"}),
    "mint.chunk":        ("View", {"view": "chunk"}),
    "mint.cat":          ("View", {"variadic": True, "view": "concat"}),
    "mint.concat":       ("View", {"variadic": True, "view": "concat"}),
    "mint.stack":        ("View", {"variadic": True, "view": "stack"}),
    "mint.tile":         ("View", {"view": "tile"}),
    "mint.roll":         ("View", {"view": "roll"}),
    "mint.broadcast_to": ("View", {"view": "broadcast"}),
    "mint.flatten":      ("View", {"view": "reshape"}),
    "mint.contiguous":   ("View", {"view": "contiguous"}),

    # —— dtype ——(`Cast` 自身不 save;fp32 buffer 由下游非线性/matmul 消费者 pin,
    #    见 bprop_rules 模块 docstring)
    "ops.cast":          ("Cast", {}),
    "mint.cast":         ("Cast", {}),

    # —— RMSNorm(`ops.rms_norm`,deepseek_v4_hybrid_attention.py:158 绑定;本版本 construct
    #    未调用它——Q-head 归一化在 :245 用 `q * mint.rsqrt(mint.mean(q*q,...) + eps)` 手写)。
    #    归一化统计需要输入 → 与既有 `Norm` 叶子同口径。
    "ops.rms_norm":      ("Norm", {"rms": True}),

    # —— 形状查询:**不发射节点**(产出 python 标量元组)。multi_latent_attention.py:127 ——
    "ops.shape":         (SHAPE_OF, {}),

    # —— 逐元素 · 梯度线性(反向 dy 直通/广播,不需存输入)——
    "mint.add":          _LIN,
    "mint.sub":          _LIN,
    "mint.neg":          _LIN,
    # 归约求和/均值:dx = broadcast(dy)/N —— 与输入值无关 → 不存。indexer.py:308-309 ——
    "mint.sum":          _REDUCE_LIN,
    "mint.mean":         _REDUCE_LIN,
    "mint.cumsum":       _REDUCE_LIN,
    # triu:结构性掩蔽(与输入值无关的 0/1 掩码),dx = triu(dy) → 线性。indexer.py:314 ——
    "mint.triu":         _LIN,
    "mint.tril":         _LIN,

    # —— 逐元素 · 非线性(dy·f'(x) 需要操作数)——
    "mint.mul":          _NONLIN,
    "mint.div":          _NONLIN,
    "mint.exp":          _NONLIN,     # csa.py:511-512
    "mint.log":          _NONLIN,     # indexer.py:310
    "mint.sqrt":         _NONLIN,
    "mint.rsqrt":        _NONLIN,     # deepseek_v4_hybrid_attention.py:245
    "mint.abs":          _NONLIN,
    "mint.pow":          _NONLIN,
    "mint.sign":         _LIN,
    # clamp:反向按「是否落在区间内」的掩码分流 → 需要输入。csa.py:455 / indexer.py:156 ——
    "mint.clamp":        _NONLIN,
    # maximum/minimum(逐元素二元):反向按「哪一侧胜出」分流 → 需要两操作数。csa.py:509 ——
    "mint.maximum":      _NONLIN,
    "mint.minimum":      _NONLIN,
    # max/min(归约,返回 (values, indices)):反向只回传 argmax 位置 → 保守存输入
    # (与既有 `ArgMaxWithValue` 口径一致,init_binder.py:30)。csa.py:508 ——
    "mint.max":          _REDUCE_NONLIN,
    "mint.min":          _REDUCE_NONLIN,

    # —— 激活(dy·f'(x) → 存输入)——
    "mint.nn.functional.relu":     ("Activation", {"activation_type": "relu"}),   # indexer.py:152
    "mint.relu":                   ("Activation", {"activation_type": "relu"}),
    "mint.sigmoid":                ("Activation", {"activation_type": "sigmoid"}),
    "mint.nn.functional.silu":     ("Activation", {"activation_type": "silu"}),
    "mint.nn.functional.gelu":     ("Activation", {"activation_type": "gelu"}),
    "mint.nn.functional.softplus": ("Activation", {"activation_type": "softplus"}),

    # —— softmax:反向用**输出** ——  indexer.py:307 / compressor.py:138
    "mint.softmax":                    ("Softmax", {}),
    "mint.nn.functional.softmax":      ("Softmax", {}),
    "mint.log_softmax":                ("Softmax", {"log": True}),

    # —— 矩阵乘(两操作数都要存)——  indexer.py:151/306 / csa.py:496 / utils.py:112
    "mint.matmul":                 ("MatMul", {}),
    "mint.bmm":                    ("BMM", {}),
    "mint.nn.functional.linear":   ("MatMul", {"functional_linear": True}),

    # —— 索引类 ——
    # gather(input, dim, index):反向 = 把 dy scatter_add 回零张量的 index 位置 → 存 **index**。
    # `ins` 只收可追踪张量操作数(dim 是 int,被 _emit 略过)→ index 落 ins[1]。indexer.py:390 ——
    "mint.gather":                     ("IndexSelect", {}),
    "mint.index_select":               ("IndexSelect", {}),
    "mint.nn.functional.embedding":    ("Gather", {"embedding": True}),
    # scatter(input, dim, index, src):反向对 input 是「index 处置零」,对 src 是 gather → 存 index。
    # ins = [input, index](dim/常量 src 被略过)。indexer.py:315 ——
    "mint.scatter":                    ("Scatter", {}),
    # where(cond, a, b):反向按 cond 分流 → 存 **cond**。csa.py:443 / indexer.py:316 ——
    "mint.where":                      ("Where", {}),
    # topk:反向把 dy scatter 到选中位置 → 存**自己的第 2 个输出**(indices)。indexer.py:155 ——
    "mint.topk":                       ("TopK", {}),
    "mint.sort":                       ("TopK", {"sort": True}),
    # argsort/argmax:只产索引,**不可微** ——
    "mint.argsort":                    _CMP,
    "mint.argmax":                     _CMP,

    # —— 比较 / 逻辑:bool 产出,**不可微**,反向什么都不读 ——
    "mint.eq": _CMP, "mint.ne": _CMP, "mint.gt": _CMP, "mint.ge": _CMP,
    "mint.lt": _CMP, "mint.le": _CMP,
    "mint.isfinite": _CMP,          # csa.py:807 / indexer.py:401
    "mint.isnan": _CMP,
    "mint.logical_and": _CMP, "mint.logical_or": _CMP, "mint.logical_not": _CMP,
    "mint.any": _CMP, "mint.all": _CMP,     # indexer.py:317

    # —— 常量产出(无梯度、反向不读任何输入;`*_like` 的张量实参只用于取 shape)——
    "mint.arange":     _CONST,      # csa.py:435/439/452-453
    "mint.zeros":      _CONST,      # csa.py:590/782
    "mint.ones":       _CONST,      # deepseek_v4_hybrid_attention.py:160
    "mint.full":       _CONST,      # csa.py:442/457/780/811 / indexer.py:313
    "mint.empty":      _CONST,      # compressor.py:118
    "mint.eye":        _CONST,
    "mint.zeros_like": _CONST,
    "mint.ones_like":  _CONST,      # indexer.py:285
    "mint.full_like":  _CONST,

    # —— detach:**建节点 + 保边 + 打 detached 标**(路线 B P1#11;csa.py:665/666/764/765/794/795)——
    "ops.stop_gradient": ("Detach", {"detach": True}),
}

# 张量方法链(`<expr>.<method>(...)`)里语义上是**别名/直通**的方法:不新增激活、不建节点,
# 目标承接上游 producer(保边)。
#   to_local / full_tensor —— DTensor 局部分片视图(hyper_parallel);compressor.py:208、
#                             deepseek_v4_hybrid_attention.py:280、csa.py:468/685
#   contiguous            —— 内存布局重排,数学恒等;csa.py:265
#   detach                —— **不**在此列(它有梯度语义,走 Detach 节点)
PASSTHRU_METHODS = frozenset({"to_local", "full_tensor", "contiguous"})

# 张量方法链里视作纯视图的方法(既有 `construct_walker._VIEW_METHODS` 的超集,合并到此处统一维护)。
VIEW_METHODS = frozenset({
    "reshape", "view", "transpose", "swapaxes", "flatten", "expand_dims",
    "tile", "permute", "squeeze", "unsqueeze", "broadcast_to", "repeat",
})

# 已知张量算子命名空间(链首)。链首不在此集合的 `self.x = a.b` **不算算子别名**
# (如 `self.n = config.num_heads` —— 配置读取)。
ALIAS_NAMESPACES = ("mint", "ops", "F", "mindspore", "P", "nn")


# 张量**方法**调用 `<expr>.<method>(...)` 的语义:与 `mint.<method>` 完全同一条判据,
# 故直接按方法名映射到主表里的 `mint.*` 键(单点维护,不另开一张表)。
# 实测触发点:`(kv.astype(fp32) * weights).sum(dim=1)`(compressor.py:216)、
#   `pooled.astype(x.dtype)`(:218)、`mask.to(mstype.float32)`(indexer.py:359)、
#   `mint.arange(b, ...).unsqueeze(1).unsqueeze(2)`(csa.py:483)。
TENSOR_METHODS = (
    "sum", "mean", "max", "min", "abs", "exp", "log", "sqrt", "rsqrt", "sigmoid",
    "softmax", "clamp", "gather", "scatter", "topk", "cumsum", "any", "all",
    "isfinite", "argsort", "argmax", "triu", "tril", "matmul", "bmm", "roll", "chunk",
    "split", "cat", "concat", "stack", "where", "pow", "neg", "add", "sub", "mul", "div",
)


def lookup_method(name: str):
    """张量方法名 → `(op, attrs)`;不在 `TENSOR_METHODS` 里返回 None。"""
    if name not in TENSOR_METHODS:
        return None
    return lookup(f"mint.{name}")


# ── opdag op 类型 → `model_spec.OpType` 的**显式**映射(2026-07-25 并行 agent 的约束)──────────
#
# **两套词表是分开的,这不是笔误**:
#   * `OpNode.op`(本模块 / `bprop_rules.PIN` 的键)是**反向语义**词表 —— 它必须区分
#     「`Compare` 反向什么都不读」与「非线性 `Elementwise` 反向存两操作数」,否则 saved 集就错;
#   * `model_spec.OpType`(`structure_mem` 的 norm/flash-attn/moe_gemm 判据、
#     `specs.RecomputeSpec.op_matches` 的子串匹配)是**成本模型**词表,粒度粗得多。
#
# 风险(并行 agent 实测):`OpType` 的**字符串值是承载语义的**,拼错会**静默走错分支**。
# 故这里给一张**显式、可评审**的映射,并让 `to_op_type()` 对未映射的 op **fail-loud** ——
# 未来的 `to_resolved.py` 适配器(**非本轮交付**)必须走它,不许现场臆造字符串。
#
# 三个 opdag op 类型**没有**合适的 `OpType` 成员(报告为建议项,不擅自新增枚举成员):
#   `Detach` / `Compare` / `Constant` —— 它们都是「产出张量但不参与梯度」的类别,
#   现有 9 个成员里没有语义对应物;硬塞进 `ELEMENTWISE` 会让 `RecomputeSpec.op_matches`
#   的子串匹配把它们连带选进重算集,是**静默错**。故映射到 `None` 并由调用方显式决策。
OPDAG_OP_TO_OPTYPE: dict[str, str | None] = {
    "MatMul": "matmul", "BMM": "matmul", "GroupedMatMul": "moe_gemm",
    "Norm": "norm",
    "FlashAttention": "flash_attn",
    "Softmax": "elementwise", "Activation": "elementwise",
    "Elementwise": "elementwise", "Cast": "elementwise",
    "View": "elementwise", "Identity": "elementwise",
    "Gather": "elementwise", "IndexSelect": "elementwise",
    "Where": "elementwise", "Scatter": "elementwise", "TopK": "elementwise",
    "Dropout": "elementwise",
    # 融合注意力内核:`npu_sparse_flash_mla` / `npu_lightning_indexer` 都是 attention 内核。
    "FusedFunction": "flash_attn", "Kernel": "flash_attn",
    # ↓ 无对应枚举成员 —— 适配器必须显式决策(见上方注释)。
    "Detach": None, "Compare": None, "Constant": None,
    # 结构占位(递归内联后不应残留)。
    "SubCell": None,
}


class UnmappedOpTypeError(ValueError):
    """opdag op 类型没有 `model_spec.OpType` 对应物 —— 拒绝猜一个字符串出来。"""


def to_op_type(op: str) -> str:
    """opdag op 类型 → `model_spec.OpType` 的**值字符串**;未映射/无对应物即 fail-loud。

    存在的唯一理由:`OpType` 的值参与 `structure_mem` 的分支与 `RecomputeSpec.op_matches`
    的子串匹配 —— 拼错不会报错,只会**静默走另一条分支**。
    """
    if op not in OPDAG_OP_TO_OPTYPE:
        raise UnmappedOpTypeError(
            f"opdag op 类型 {op!r} 不在 `OPDAG_OP_TO_OPTYPE` 表里 —— 拒绝现场编一个 "
            f"OpType 字符串(它的值参与 structure_mem 分支与 RecomputeSpec 子串匹配,"
            f"拼错是**静默**走错分支)。请显式加映射并说明理由。"
        )
    val = OPDAG_OP_TO_OPTYPE[op]
    if val is None:
        raise UnmappedOpTypeError(
            f"opdag op 类型 {op!r} 在 `model_spec.OpType` 的 9 个成员里**没有语义对应物**"
            f"(它产出张量但不参与梯度)。硬塞进 'elementwise' 会被 "
            f"`RecomputeSpec.op_matches` 的子串匹配连带选中 = 静默错。"
            f"需要新枚举成员时请显式决策,不要在适配器里猜。"
        )
    return val


def lookup(dotted: str):
    """点号路径 → `(op, attrs)`;表里没有则 `None`(调用方决定是 fallthrough 还是 fail-loud)。"""
    hit = PRIMITIVES.get(dotted)
    if hit is None:
        return None
    op, attrs = hit
    return op, dict(attrs)


def is_alias_namespace(dotted: str) -> bool:
    return bool(dotted) and dotted.split(".")[0] in ALIAS_NAMESPACES


def lookup_alias(dotted: str, *, attr: str = "", src: str = ""):
    """裸别名专用:**未知即 fail-loud**(不猜类别)。

    `self.<attr> = <dotted>` 且 `<dotted>` 在已知张量算子命名空间里,却不在 `PRIMITIVES` 表里
    —— 说明这是个**没人核对过 bprop 语义**的原语。归错类会静默产出错的 saved 集,故拒绝放行。
    """
    hit = lookup(dotted)
    if hit is None:
        raise UnknownPrimitiveError(
            f"未知原语 `{dotted}`(self.{attr} @ {src or '?'}):"
            f"它在已知张量算子命名空间里,但 primitives.PRIMITIVES 无表项 —— "
            f"**拒绝猜它属于哪一类**(归错类 = saved 集静默错)。"
            f"请在 primitives.py 加一条表项,并在注释里写清「反向要读什么」+ 定位符。"
        )
    return hit
