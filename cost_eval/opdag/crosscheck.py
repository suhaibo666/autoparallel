"""P2-02 闭环：把 op-DAG 从**离线交叉验证工具**升级为**生产链路的一致性校验器**。

背景（审计判据）：`cost_eval/opdag/` 从**真** mindformers 源码（AST 静态读，绝不 import）抽出
每个 Cell 的 op 图，此前只在 `tests/test_opdag_*` 里离线核对，Evaluator/build_llm 主链路**不消费**
它——手写 `LayerSpec`（`cost_eval/layers/*`）仍是唯一的生产事实源，opdag 与它是否漂移无人把关。

本模块让 opdag **真正进入主链路**：给 Evaluator（或独立入口 `validate_against_opdag(spec)`）一个
**默认旁路、可选启用**的一致性钩子——对**有 opdag 提取源的层段**（DSv3 的 MLA 注意力段、MoE 专家
grouped-GEMM 段），把手写 `LayerSpec` 的**重算子名册（op census）**与 opdag 从源码抽出的名册做
**类别级交叉校验**；漂移即 `warn`（`strict=True` 则 `raise OpdagConsistencyError`）。

**为什么是「类别级 census + 声明式 delta」而不是「逐 op 相等」**（核心设计）：两侧粒度/边界本就不同，
逐 op 相等会满屏假阳性——

  1. **结构算子粒度**：opdag 把 reshape/split/transpose 建成 `View`、dtype 转换建成 `Cast`，手写侧
     不单列这些（隐式）→ 从两侧 census 中**剔除** View/Cast（及 opdag 用 Elementwise 表达的 rope、
     手写用 ELEMENTWISE 表达的残差 add）——只比**重算子**：线性(matmul)/分组 GEMM/归一化/FlashAttn/
     非线性激活。
  2. **模块边界**：mindformers 的 MLA `MLASelfAttention` **不含** pre-norm(input_layernorm) 与
     residual add——它们在外层 `TransformerLayer`；手写 `build_mla_attn_ops` 把 ln1/add1 一并纳入。
     故手写 NORM 比 opdag 多 1（ln1）。以**声明式 `expected_delta[NORM]=-1`** 显式对账（有据、可读）。
  3. **融合选择**：手写走融合路径（`mla_qkv_concat=True` 语义：`linear_qkv` 把 q_down+kv_down 合成
     一次 matmul）；opdag 提取夹具用 `mla_qkv_concat=False`（q_down/kv_down 分列）→ opdag matmul 比
     手写多 1。以 `expected_delta[LINEAR]=+1` 显式对账。

于是校验断言 **`opdag_census[cat] - hand_census[cat] == expected_delta[cat]`**（逐类别）。这是**真**
交叉校验：手写 builder 若掉了 `linear_qb`（少一个 matmul）→ 实际 delta 变 +2≠+1 → 报出；mindformers
源结构变了 → opdag census 变 → 报出。`expected_delta` 里的非零项是**文档化的结构事实**（融合 / 模块
边界）的一次性对账，不是拟合。

**诚实的覆盖边界**（不杜撰、显式 log）：
  - MLA 段：只交叉校验 LINEAR/NORM/ATTENTION；rope（opdag=Elementwise vs 手写=ROPE，粒度不同）与
    residual add 不比。
  - MoE 段：只交叉校验专家 grouped-GEMM 核（LINEAR_GROUPED/ACTIVATION）。`router`/`dispatch`/
    `combine` 是 token_dispatcher 的 opaque all-to-all + 数据依赖的 TopKRouter（opdag 在 `MoELayer`
    的 router 处 fail-loud，见 `test_opdag_moe_ffn.py` R3 结论）→ **不在 opdag 提取范围**，作为 opaque
    边界 log，不比。
  - GQA/dense/embedding/lm_head/mtp 层**无 opdag 提取源** → 作为 uncovered 层 log。

**校验口径的诚实边界（不可过度宣称）**：本模块是一个**算子类别 census 的一致性校验器**——它只比较
两侧「重算子类别计数」（matmul/norm/linear_grouped/attention/activation…）之间的**逐类别 delta**，
**不**比较每个张量的 `saves`（保存清单）、shape、dtype、workspace 或生命周期。因此它**无法**发现
一类不改变 op 类别计数的内存契约漂移——例如从某个 matmul op 删掉一条 `saves` 记录（op 数不变、类别
census 不变）就**逃不出**本校验。要覆盖这类 save 级/内存契约漂移，需要另建**逐张量的 save-level
对账**（本模块不做，也不假装做）。一句话：这是**类别 census 一致性检查**，不是**完整内存契约验证器**。

**提取失败 ≠ 合法无对应（strict 语义关键）**：某层「合法地没有 opdag 提取源」（embedding/lm_head/
mtp/gqa/dense）记为 `uncovered`，是设计内的诚实边界，strict **不**因此失败。但一个**已声明覆盖**的族
（MLA/MoE）在抽取时**抛异常**是另一回事——它意味着交叉校验**无从验证**该族，若仍静默返回 `ok=True`
就是**假绿**。故此类失败单列进 `extraction_failures`，让 `ok=False`，strict 下直接 `raise`。

**默认不改变评估行为**：`Evaluator(..., validate_opdag=False)` 默认关；本模块也可独立调用。缺 mindformers
源（如 CI）→ 报告标 `available=False`、静默跳过（`warn` 一次），绝不因缺源而 fail。
"""
from __future__ import annotations

import collections
import functools
import os
import warnings
from dataclasses import dataclass, field

from .extractor import extract_cell
from .module_resolver import ResolvedSpec, resolve_layer_spec


# ── 规范重算子类别（两侧共享词表）────────────────────────────────────────────────
LINEAR = "linear"                 # dense matmul 投影
LINEAR_GROUPED = "linear_grouped"  # 分组 GEMM（MoE 专家）
NORM = "norm"                     # 归一化
ATTENTION = "attention"           # FlashAttention 核
ACTIVATION = "activation"         # 非线性门（swiglu/gelu/…）

# 手写激活 op 名启发式（ELEMENTWISE 里区分「非线性激活」与「残差 add」）。
_ACT_NAME_HINTS = ("swiglu", "gelu", "silu", "relu", "geglu")


class OpdagConsistencyError(ValueError):
    """strict 模式下 opdag 一致性校验发现漂移时抛出。"""


class OpdagDriftWarning(UserWarning):
    """非 strict 模式下 opdag 一致性校验发现漂移时的告警类别。"""


def default_mf_root() -> str:
    """mindformers 源根：环境变量 MINDFORMERS_ROOT 优先，否则本机默认路径。"""
    return os.environ.get(
        "MINDFORMERS_ROOT",
        r"E:\97-codes\torch_parallel\mindformers\mindformers",
    )


# ── 类别映射 ──────────────────────────────────────────────────────────────────
def hand_category(op):
    """手写 `OpSpec` → 规范类别（None = 不参与交叉校验：rope/residual/router/dispatch/combine…）。"""
    t = getattr(op.type, "value", op.type)
    if t == "matmul":
        return LINEAR
    if t == "moe_gemm":
        return LINEAR_GROUPED
    if t == "norm":
        return NORM
    if t == "flash_attn":
        return ATTENTION
    if t == "elementwise":
        name = (op.name or "").lower()
        if any(h in name for h in _ACT_NAME_HINTS):
            return ACTIVATION
    return None


_OPDAG_CAT = {
    "MatMul": LINEAR,
    "GroupedMatMul": LINEAR_GROUPED,
    "Norm": NORM,
    "FlashAttention": ATTENTION,
    "Activation": ACTIVATION,
    # View / Cast / Elementwise(rope) / Gather / Dropout → None（结构/瞬态/opaque，剔除）。
}


def opdag_category(node):
    """opdag `OpNode.op` → 规范类别（None = 结构/瞬态，剔除）。"""
    return _OPDAG_CAT.get(node.op)


def _census_opdag(dag) -> collections.Counter:
    c = collections.Counter()
    for n in dag.nodes:
        cat = opdag_category(n)
        if cat is not None:
            c[cat] += 1
    return c


# ── opdag 提取（从真 mindformers 源，缓存；夹具与 test_opdag_mla/moe_ffn 一致）───────────
_MLA_REL = "parallel_core/training_graph/transformer/multi_latent_attention.py"
_MLA_SPEC_FLAGS = {
    "multi_latent_attention": True, "mla_qkv_concat": False, "num_experts": 256,
    "qk_layernorm": True, "sparse_attention": False, "fused_norm": True,
    "moe_grouped_gemm": True, "use_contiguous_weight_layout_attention": False,
    "use_interleaved_weight_layout_mlp": True,
}
_MLA_CELL_FLAGS = {
    "use_dsa": False, "use_flash_attention": True, "use_eod_attn_mask_compression": False,
    "cp": 1, "cp_ds": 1, "input_layout": "BNSD", "q_lora_rank": 1536,
    "compute_dtype": "bf16", "layernorm_compute_dtype": "fp32",
}
_FFN_REL = "parallel_core/training_graph/transformer/moe/ffn.py"
_FFN_FLAGS = {
    "moe_token_dispatcher_type": "alltoall", "compute_dtype": "bf16", "add_bias_linear": False,
}


@functools.lru_cache(maxsize=None)
def opdag_mla_census(mf_root: str) -> collections.Counter:
    """真 `MLASelfAttention`（DSv3, mla_qkv_concat=False）的重算子 census。"""
    top = resolve_layer_spec(mf_root, _MLA_SPEC_FLAGS)
    mla = top.submodules["self_attention"]
    dag = extract_cell(mf_root, _MLA_REL, "MLASelfAttention", mla, _MLA_CELL_FLAGS,
                       present_params={"rotary_pos_emb"})
    return _census_opdag(dag)


@functools.lru_cache(maxsize=None)
def opdag_moe_census(mf_root: str) -> collections.Counter:
    """真 `FFNGroupedGEMM`（DSv3 MoE 专家 grouped-GEMM 核）的重算子 census。"""
    spec = ResolvedSpec(cell="FFNGroupedGEMM", submodules={})
    dag = extract_cell(mf_root, _FFN_REL, "FFNGroupedGEMM", spec, _FFN_FLAGS)
    return _census_opdag(dag)


# ── 层段 ↔ opdag 提取源的对应契约 ─────────────────────────────────────────────────
@dataclass
class Correspondence:
    """一个手写层段 ↔ opdag 提取源的对应契约。"""
    family: str                      # "mla_attn" / "moe_experts"
    checked: tuple                   # 参与交叉校验的类别
    expected_delta: dict             # {category: opdag_count - hand_count}（结构事实对账）
    census_fn: object                # callable(mf_root) -> Counter（opdag 侧）
    source: str                      # 人读的 opdag 提取源描述
    opaque_note: str                 # opdag 不覆盖/不比的部分（诚实边界）


_MLA_CORR = Correspondence(
    family="mla_attn",
    checked=(LINEAR, NORM, ATTENTION),
    # opdag - hand：LINEAR +1（手写融合 q_down+kv_down→linear_qkv）；NORM -1（pre-norm ln1 在
    # TransformerLayer，不在 MLASelfAttention 模块内）；ATTENTION 0（flash 恒 1==1）。
    expected_delta={LINEAR: +1, NORM: -1, ATTENTION: 0},
    census_fn=opdag_mla_census,
    source="MLASelfAttention @ multi_latent_attention.py (mla_qkv_concat=False)",
    opaque_note=("rope（opdag=Elementwise vs 手写=ROPE，粒度不同）与 residual add 不交叉校验；"
                 "pre-norm ln1 归 TransformerLayer（delta NORM=-1）；q/kv down 手写融合（delta LINEAR=+1）"),
)

_MOE_CORR = Correspondence(
    family="moe_experts",
    checked=(LINEAR_GROUPED, ACTIVATION),
    expected_delta={LINEAR_GROUPED: 0, ACTIVATION: 0},   # 专家核 grouped-GEMM×2 + 激活×1，逐个相等
    census_fn=opdag_moe_census,
    source="FFNGroupedGEMM @ moe/ffn.py",
    opaque_note=("router/dispatch/combine 是 token_dispatcher 的 opaque all-to-all + 数据依赖 "
                 "TopKRouter（opdag 在 MoELayer.router 处 fail-loud，R3 边界）→ 不在提取范围、不比"),
)


# ── 手写层段切分 + census ─────────────────────────────────────────────────────────
def _split_decoder(ops):
    """把一个 decoder 层的 ops 按残差输出 `h1` 切成 (attn 段, ffn 段)。

    手写 decoder 的 op 序列为 [attn 段…(以 add1→h1 结尾)] + [ffn 段…]。无 `h1` 产出者（embedding/
    lm_head/mtp 等非 decoder body）→ 返回 (None, None)。
    """
    for i, op in enumerate(ops):
        if getattr(op.output, "name", None) == "h1":
            return list(ops[:i + 1]), list(ops[i + 1:])
    return None, None


def _census_hand(ops, categories) -> collections.Counter:
    c = collections.Counter()
    for op in ops:
        cat = hand_category(op)
        if cat in categories:
            c[cat] += 1
    return c


def _moe_expert_window(ffn_ops):
    """MoE ffn 段里专家 grouped-GEMM 核的 op 窗口 = [首个 moe_gemm … 末个 moe_gemm]（含其间激活）。

    这样只圈住 opdag `FFNGroupedGEMM` 对应的专家核（e_fc1/e_swiglu/e_fc2），把 router/dispatch/
    combine（前）与 shared expert（后）排除在窗口外——它们的类别不进本次比较（诚实边界）。
    无 moe_gemm → None（非 MoE ffn）。
    """
    idx = [i for i, op in enumerate(ffn_ops) if hand_category(op) == LINEAR_GROUPED]
    if not idx:
        return None
    return ffn_ops[idx[0]:idx[-1] + 1]


# ── 报告数据结构 ──────────────────────────────────────────────────────────────────
@dataclass
class Finding:
    """一条漂移：某覆盖层段的某类别，opdag−hand 的实际 delta 偏离 expected_delta。"""
    layer_type: str
    family: str
    category: str
    hand_count: int
    opdag_count: int
    expected_delta: int
    actual_delta: int
    source: str

    @property
    def message(self) -> str:
        return (f"[opdag 漂移] 层 {self.layer_type!r} 的 {self.family} 段 类别 {self.category}："
                f"手写={self.hand_count}、opdag={self.opdag_count}（源 {self.source}）；"
                f"实际 delta(opdag-hand)={self.actual_delta:+d} ≠ 预期 {self.expected_delta:+d}"
                f"——手写 LayerSpec 与源码 op 图结构不一致。")


@dataclass
class Covered:
    layer_type: str
    family: str
    source: str
    opaque_note: str


@dataclass
class CrossCheckReport:
    available: bool                       # mindformers 源可用（否则跳过）
    mf_root: str
    findings: list = field(default_factory=list)     # list[Finding]（漂移）
    covered: list = field(default_factory=list)       # list[Covered]
    uncovered: list = field(default_factory=list)     # list[(layer_type, note)]（合法无对应）
    # 已声明覆盖族在抽取时抛异常 → list[(layer_type, family, error)]。这**不是**合法无对应
    # （见模块 docstring）——它意味着该族无从验证，故让 ok=False、strict 下 raise。
    extraction_failures: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """无漂移**且**无提取失败即通过（不可用也视作「无从校验、无可报」→ True）。

        注意：提取失败也让 ok=False——一个已声明覆盖的族抽取失败意味着交叉校验无从验证它，
        若仍返回 True 就是假绿（见模块 docstring「提取失败 ≠ 合法无对应」）。
        """
        return not self.findings and not self.extraction_failures

    def summary(self) -> str:
        if not self.available:
            return f"opdag 一致性校验：跳过（mindformers 源不可用：{self.mf_root!r}）"
        cov = ", ".join(sorted({c.family for c in self.covered})) or "（无）"
        status = "通过" if self.ok else "发现漂移/提取失败"
        head = (f"opdag 一致性校验：{status}；"
                f"覆盖层段={len(self.covered)}（{cov}）、未覆盖层={len(self.uncovered)}、"
                f"漂移={len(self.findings)}、提取失败={len(self.extraction_failures)}")
        lines = [head]
        for f in self.findings:
            lines.append("  - " + f.message)
        for layer_type, family, error in self.extraction_failures:
            lines.append(
                f"  - [opdag 提取失败] 层 {layer_type!r} 的 {family} 段：声明覆盖但无法从源抽取 op 图"
                f"——交叉校验无从验证该族（{error}）")
        return "\n".join(lines)


# 手写层段 → 对应契约（按检测顺序）。
_CORRESPONDENCES = (_MLA_CORR, _MOE_CORR)


def _check_segment(layer_type, seg_ops, corr, mf_root, report):
    """对一个手写层段跑对应契约的交叉校验，把 Covered / Finding 写进 report。返回是否命中该族。"""
    try:
        ocen = corr.census_fn(mf_root)
    except Exception as e:  # 已声明覆盖族的 opdag 提取失败（源结构变动/夹具不匹配/抽取器异常）。
        # **不**降级成 uncovered（那会被误读为「合法无对应」而放行）——单列进 extraction_failures，
        # 让 ok=False、strict 下 raise：交叉校验无从验证该族，静默放行就是假绿（F6）。不 crash。
        report.extraction_failures.append((layer_type, corr.family, str(e)))
        return True
    hcen = _census_hand(seg_ops, corr.checked)
    report.covered.append(Covered(layer_type, corr.family, corr.source, corr.opaque_note))
    for cat in corr.checked:
        actual = ocen.get(cat, 0) - hcen.get(cat, 0)
        expected = corr.expected_delta.get(cat, 0)
        if actual != expected:
            report.findings.append(Finding(
                layer_type, corr.family, cat, hcen.get(cat, 0), ocen.get(cat, 0),
                expected, actual, corr.source))
    return True


def validate_against_opdag(spec, *, mf_root: str | None = None,
                           strict: bool = False, warn: bool = True) -> CrossCheckReport:
    """把手写 `ModelSpec` 的重算子名册与 opdag 从 mindformers 源抽出的名册做**类别 census** 一致性校验。

    **口径**：只比较两侧「重算子类别计数」的逐类别 delta（matmul/norm/linear_grouped/attention/
    activation），**不**比较每张量 `saves`/shape/dtype/workspace/生命周期。故它是**类别 census 一致性
    检查**，不是**完整内存契约验证器**——不改变 op 类别计数的 save 级漂移（如从某 matmul 删一条 `saves`）
    逃得出本校验（详见模块 docstring）。

    参数
      spec    — 手写 `ModelSpec`（`build_llm_spec` / `build_dsv3_spec` 产出）。
      mf_root — mindformers 源根；缺省用 `default_mf_root()`。不存在 → 报告标 `available=False`、
                （warn 时）告警一次后**静默跳过**（绝不因缺源 fail；strict 也不 raise，无从校验）。
      strict  — True：发现**漂移**或**已声明覆盖族的提取失败**即 `raise OpdagConsistencyError`。
                False（默认）：`warn` 时发 `OpdagDriftWarning`，把漂移/提取失败记进报告返回。
      warn    — 是否在缺源/漂移/提取失败时 `warnings.warn`（默认 True）。

    返回 `CrossCheckReport`（findings/covered/uncovered/extraction_failures/available）。
    合法无对应的层（embedding/lm_head/mtp/gqa/dense）记 `uncovered`，**不**触发 strict；只有已声明
    覆盖族（MLA/MoE）抽取失败才记 `extraction_failures` 并触发 strict（见模块 docstring）。
    """
    mf_root = mf_root or default_mf_root()
    report = CrossCheckReport(available=os.path.isdir(mf_root), mf_root=mf_root)
    if not report.available:
        if warn:
            warnings.warn(
                f"opdag 一致性校验跳过：mindformers 源不存在于 {mf_root!r}"
                "（设 MINDFORMERS_ROOT 或传 mf_root 启用；缺源不影响评估）。",
                OpdagDriftWarning, stacklevel=2)
        return report

    for ltype, ls in spec.layer_specs.items():
        attn_ops, ffn_ops = _split_decoder(ls.ops)
        matched = False
        # MLA 注意力段：signature = 段内含 `linear_kvb`（MLA 独有；GQA 走融合 `qkv` 无此名）。
        if attn_ops is not None and any(op.name == "linear_kvb" for op in attn_ops):
            matched |= _check_segment(ltype, attn_ops, _MLA_CORR, mf_root, report)
        # MoE 专家段：signature = ffn 段内含 moe_gemm（grouped-GEMM）op。
        if ffn_ops is not None:
            window = _moe_expert_window(ffn_ops)
            if window is not None:
                matched |= _check_segment(ltype, window, _MOE_CORR, mf_root, report)
        if not matched:
            report.uncovered.append(
                (ltype, "无 opdag 提取源（embedding/lm_head/mtp/gqa/dense 未抽取）"))

    # 漂移 **或** 已声明覆盖族的提取失败都是「非绿」——strict 下都要 raise（提取失败静默放行=假绿）。
    if report.findings or report.extraction_failures:
        if strict:
            if report.extraction_failures and not report.findings:
                fams = ", ".join(sorted({f for _lt, f, _e in report.extraction_failures}))
                head = (f"opdag 一致性校验无法完成：已声明覆盖族 [{fams}] 的 op 图抽取失败——"
                        f"无从验证，strict 下不得静默放行。\n")
                raise OpdagConsistencyError(head + report.summary())
            raise OpdagConsistencyError(report.summary())
        if warn:
            warnings.warn(report.summary(), OpdagDriftWarning, stacklevel=2)
    return report
