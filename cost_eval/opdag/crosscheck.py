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

**第三族 `layer_norms`（层级 pre-norm 名册，Z1 修复 2026-07-16）**：前两族（MLA/MoE）的比较窗口
把**层级两个强制 pre-norm** 漏在外——`moe_experts` 窗口是 `[首个 moe_gemm … 末个 moe_gemm]`，排除了
`ln2`（pre_mlp_layernorm，位于 router 之前）；`mla_attn` 段以 `add1→h1` 收尾，`ln2` 落在 ffn 段，
被两族都不 census。于是删掉某 MoE 层的 `ln2` 竟能 `ok==True`（F1 类漏建：少一个 pre-norm）。第三族补这个
洞：对每个 decoder-body 层（`_split_decoder` 同时给出 attn/ffn 段者），用 `resolve_layer_spec` 从**真**
`get_gpt_layer_local_spec` 静态解析 `TransformerLayerSubmodules` 的 `input_layernorm` / `pre_mlp_layernorm`
两槽是否为**真归一类**（源码在 MLA 与非 MLA 两支**无条件** `= get_norm_cls(fused_norm)`，见
`gpt_layer_specs.py:162/164`（MLA）与 `:172/183`（非 MLA）；`IdentityOp` 只用于**可选** q/k layernorm）——
若为真 norm，则断言手写 attn 段含一个 pre-attn NORM（ln1）、ffn 段**首个重算子是 NORM**（ln2）；缺失即
`LayerNormFinding`（让 `ok=False`、strict 下 raise）。**诚实边界**：本族只校验这两个 pre-norm 的**存在/
位置**，**不**校验它们的 saves/shape/dtype（与全模块「类别 census、非内存契约验证器」的口径一致）。

**A5 扩展（2026-07-16）：第三族再补三类源码强制的「次级 norm」**——两个层级 pre-norm 之外，源码还
无条件/条件地强制若干次级归一层，前两族窗口同样 census 不到，静默删掉即又一 F1 类漏建。逐条源忠实：

  1. **MLA 潜向量 q/kv norm**（条件强制）：MLA `self_attention` 的 `q_layernorm`/`kv_layernorm`
     `= get_norm_cls(fused_norm) if qk_layernorm else IdentityOp`（`gpt_layer_specs.py:155/156`）。用
     **与 `opdag_mla_census` 同一** `_MLA_SPEC_FLAGS`（`qk_layernorm=True`）解析——此路径下二者是真 norm，
     故对 MLA 手写 attn 段（signature=含 `linear_kvb`）断言含 `q_a_norm`/`kv_a_norm` 两个 NORM；源若解成
     IdentityOp（qk 关）则**不**要求（无假阳）。（与 `_MLA_CORR` 的 NORM 总数 delta 语义正交：那是计数
     对账、此处点名具体潜向量 norm 的在场，可读且抗计数守恒的替换。）**诚实边界**：GQA 支的
     `q_layernorm`/`k_layernorm`（→ 手写 `q_norm`/`k_norm`，`gpt_layer_specs.py:179/180`）本族**不**单独
     强制——建模到的 spec 里没有 GQA decoder-body 层（DSv4 混合注意力层是非-body/uncovered；纯 GQA 默认
     `qk_layernorm=False`），且若用固定 qk-on 的 MLA 解析去要求 GQA，会对合法 qk-off 的 GQA 层误报假阳。
  2. **final_norm**（无条件强制）：decoder block 末、lm_head 前的 `TransformerBlockSubmodules.layer_norm
     = get_norm_cls(config.fused_norm)`（`gpt_layer_specs.py:254`，无 if 门控）。该源在 block-spec 函数里、
     **不**经 `get_gpt_layer_local_spec` 入口 → 用**定向 AST 静态读**确认其绑定为 `get_norm_cls`（非
     IdentityOp）。手写 head/final 段（非 decoder-body、含 `lm_head` 投影）须**首个重算子是 NORM**
     （final_norm，位于 lm_head 之前）；缺失即报出。
  3. **MTP enorm/hnorm**（无条件强制）：`get_mtp_layer_spec` 无条件绑定 `enorm`/`hnorm`
     `= get_norm_cls(fused_norm)`（`multi_token_prediction.py:102/103`）。同样在 MTP 层 spec 函数里、非主
     入口 → 定向 AST 读确认。手写 MTP 段（signature=含 `eh_proj`）须含 `enorm`/`hnorm` 两个 NORM。**MTP
     层被路由出前两族 delta 家族**（其 embedding/enorm/hnorm/eh_proj 前缀 + 内层 decoder + 共享 head 的
     包装结构不被 MLA/MoE 窗口建模，逐 op delta 会满屏假阳）——只校验这两个 MTP 专属 norm 的在场，记
     `uncovered`（诚实边界；MTP 内层 decoder 的 ln1/ln2/final_norm 本族不重复校验）。

  三条同 ln1/ln2：只校验**存在/位置**，不校验 saves/shape/dtype。final_norm/MTP 的定向 AST 读失败=已声明
  覆盖族无从验证 → `extraction_failures`（fail-loud，非静默放行）；MTP 源仅在 spec 真含 MTP 层时才读（无
  MTP 层的 spec 绝不因 MTP 源读失败而 fail）。

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

import ast
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
    """mindformers 源根：环境变量 MINDFORMERS_ROOT 优先，否则本机默认路径。

    **两级探测（Z3 健壮性，2026-07-16）**：抽取器要求根是**包目录**——含 `parallel_core` 的那层
    （即 `…/mindformers/mindformers`），而非仓库根。常见误配是把 `MINDFORMERS_ROOT` 指到仓库根
    `…/mindformers`（其下才是 `mindformers/parallel_core`）→ 抽取器找不到 `parallel_core` 而失败。
    故：候选（env 或默认）若自身**不含** `parallel_core` 但 `<候选>/mindformers/parallel_core` 存在，
    下降一层到 `mindformers`。**向后兼容**：候选已含 `parallel_core` → 原样返回（与当前默认逐字节一致）；
    两级都不含（如 CI 缺源）→ 原样返回候选（`available` 判定照旧为 False，绝不因探测改变缺源行为）。
    """
    cand = os.environ.get(
        "MINDFORMERS_ROOT",
        r"E:\97-codes\torch_parallel\mindformers\mindformers",
    )
    if os.path.isdir(os.path.join(cand, "parallel_core")):
        return cand                                   # 已是包目录 → 逐字节不变
    nested = os.path.join(cand, "mindformers")
    if os.path.isdir(os.path.join(nested, "parallel_core")):
        return nested                                 # 误指仓库根 → 下降一层到包目录
    return cand                                       # 两级都无（缺源）→ 原样返回，行为不变


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


# ── 第三族 layer_norms：层级 pre-norm 名册（Z1 修复 2026-07-16）─────────────────────────
# 源侧解析复用 DSv3 MLA 的 spec flags（同一 `get_gpt_layer_local_spec` 入口、同一 `TransformerLayer`
# 结构）。**源事实**：`input_layernorm`/`pre_mlp_layernorm` 在 MLA 与非 MLA 两支都**无条件**
# `= get_norm_cls(fused_norm)`（`gpt_layer_specs.py:162/164` 与 `:172/183`），故用此单一解析代表所有
# decoder 层是**源忠实**的；`IdentityOp` 只出现在**可选** q/k layernorm（qk_layernorm 门控）。
_LAYER_NORMS_SPEC_FLAGS = _MLA_SPEC_FLAGS
_LAYER_NORMS_SOURCE_DESC = (
    "TransformerLayerSubmodules.input_layernorm / pre_mlp_layernorm "
    "@ gpt_layer_specs.py get_gpt_layer_local_spec")
_LAYER_NORMS_OPAQUE_NOTE = (
    "只校验两个层级 pre-norm 的**存在/位置**（attn 段有 pre-attn NORM=ln1、ffn 段首个重算子是 "
    "NORM=ln2）；不校验其 saves/shape/dtype（类别 census 边界，见模块 docstring）")

# A5：final_norm / MTP enorm-hnorm 的源在 block-spec / mtp-spec 函数里，**不**经
# get_gpt_layer_local_spec 入口 → 用定向 AST 静态读（绝不 import mindformers）。
_GPT_LAYER_SPECS_REL = "parallel_core/training_graph/base_models/gpt/gpt_layer_specs.py"
_MTP_REL = "parallel_core/training_graph/transformer/multi_token_prediction.py"

# 每个 norm 槽位的源码定位（供 LayerNormFinding 自描述，源忠实、可读）。
_NORM_SLOT_SOURCE = {
    "input_layernorm":
        "get_gpt_layer_local_spec 无条件绑定 input_layernorm=get_norm_cls(fused_norm)"
        "（gpt_layer_specs.py:162/172）",
    "pre_mlp_layernorm":
        "get_gpt_layer_local_spec 无条件绑定 pre_mlp_layernorm=get_norm_cls(fused_norm)"
        "（gpt_layer_specs.py:164/183）",
    "q_layernorm":
        "MLA self_attention.q_layernorm=get_norm_cls(fused_norm) if qk_layernorm else IdentityOp"
        "（qk_layernorm=True 时真 norm，gpt_layer_specs.py:155）",
    "kv_layernorm":
        "MLA self_attention.kv_layernorm=get_norm_cls(fused_norm) if qk_layernorm else IdentityOp"
        "（qk_layernorm=True 时真 norm，gpt_layer_specs.py:156）",
    "final_norm":
        "TransformerBlockSubmodules.layer_norm=get_norm_cls(config.fused_norm)"
        "（gpt_layer_specs.py:254，无条件）",
    "mtp_enorm":
        "get_mtp_layer_spec 无条件绑定 enorm=get_norm_cls(fused_norm)"
        "（multi_token_prediction.py:102）",
    "mtp_hnorm":
        "get_mtp_layer_spec 无条件绑定 hnorm=get_norm_cls(fused_norm)"
        "（multi_token_prediction.py:103）",
}


def _is_real_norm(resolved) -> bool:
    """判定 `resolve_layer_spec` 解出的一个 submodule 槽位是否为**真归一类**（vs IdentityOp/缺失）。

    `resolve_layer_spec` 把 `get_norm_cls(...)` 归一为叶子字符串 `"Norm"`、`IdentityOp` 归一为
    `"Identity"`（module_resolver `_NAME_ALIAS`）；也可能是嵌套 `ResolvedSpec`（取 `.cell`）。稳健起见对
    str 叶子 / ResolvedSpec / 缺失都判：类名（不区分大小写）含 `"norm"` 且不含 `"identity"` → 真 norm。
    """
    if resolved is None:
        return False
    if isinstance(resolved, str):
        name = resolved
    elif isinstance(resolved, ResolvedSpec):
        name = resolved.cell
    else:  # 兜底：类对象 / 其它 → 取 __name__ 或字符串化
        name = getattr(resolved, "__name__", None) or str(resolved)
    if not name:
        return False
    low = name.lower()
    return ("norm" in low) and ("identity" not in low)


@functools.lru_cache(maxsize=None)
def layer_norms_source_info(mf_root: str) -> dict:
    """从真 `gpt_layer_specs.py` 静态解析 DSv3 `TransformerLayer` 的层级 pre-norm 与 MLA 潜向量 norm。

    返回 `{"input_layernorm", "pre_mlp_layernorm", "q_layernorm", "kv_layernorm"}` → bool（True=真归一
    类；False=IdentityOp/缺失）。前两者是 `TransformerLayer` 顶层的两个 pre-norm；后两者是 MLA
    `self_attention` 子模块的潜向量 q/kv norm（A5，`gpt_layer_specs.py:155/156`；用**与 `opdag_mla_census`
    同一** `_MLA_SPEC_FLAGS` 解析，`qk_layernorm=True` → 解成真 norm）。
    **诚实边界**：只判「是不是真 norm」，不解析其具体归一类型/shape/dtype。解析失败照全模块 fail-loud
    约定向上抛（由 `validate_against_opdag` 记 `extraction_failures`——已声明覆盖族无从验证 ≠ 合法无对应）。
    """
    top = resolve_layer_spec(mf_root, _LAYER_NORMS_SPEC_FLAGS)
    subs = top.submodules
    # MLA 潜向量 norm 在 self_attention 子树里（qk_layernorm 门控）。非 ResolvedSpec（缺/异常）→ 空子树。
    sa = subs.get("self_attention")
    sa_subs = sa.submodules if isinstance(sa, ResolvedSpec) else {}
    return {
        "input_layernorm": _is_real_norm(subs.get("input_layernorm")),
        "pre_mlp_layernorm": _is_real_norm(subs.get("pre_mlp_layernorm")),
        # MLA：q_layernorm/kv_layernorm（mla_qkv_concat=False 支的子模块键，见探测）。
        "q_layernorm": _is_real_norm(sa_subs.get("q_layernorm")),
        "kv_layernorm": _is_real_norm(sa_subs.get("kv_layernorm")),
    }


# ── A5：final_norm / MTP norm 的定向 AST 静态读（源在非主入口函数里，绝不 import）──────────────
def _norm_kwarg_is_real(value, rel: str, key: str) -> bool:
    """判定一个 `*Submodules(...)` 构造里某关键字的**绑定表达式**是否为真归一类。

    `get_norm_cls(...)` → True（真 norm）；`IdentityOp`（名）/ 缺失（None）→ False；其它未知构造 → 抛
    `ValueError`（fail-loud，交由 `extraction_failures`——已声明覆盖但解不出 ≠ 合法放行）。
    """
    if value is None:
        return False
    if isinstance(value, ast.Call):
        fn = value.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name == "get_norm_cls":
            return True
        raise ValueError(f"{rel}: 关键字 {key} 绑定到未知调用 {name}(...)（A5 fail-loud）")
    if isinstance(value, ast.Name):
        if value.id in ("IdentityOp", "Identity"):
            return False
        raise ValueError(f"{rel}: 关键字 {key} 绑定到未知名 {value.id}（A5 fail-loud）")
    raise ValueError(
        f"{rel}: 关键字 {key} 绑定到不可判定表达式 {type(value).__name__}（A5 fail-loud）")


def _ast_norm_kwarg_bits(mf_root: str, rel: str, func_name: str, ctor_name: str,
                         keys) -> dict:
    """静态读真源 `rel` 里 `func_name` 函数体内对 `ctor_name(...)` 构造的若干 norm 关键字绑定。

    返回 `{key: bool}`（True=真 norm）。只用 `ast`，**绝不** import mindformers。找不到文件/函数/构造、
    或关键字绑定到未知构造 → 抛 `ValueError`（fail-loud）。这与 `module_resolver` 的静态读同源精神：
    读真绑定、判是不是 `get_norm_cls`，源码若改成条件/IdentityOp 就能跟上（不硬编码「一定强制」）。
    """
    path = os.path.join(mf_root, *rel.split("/"))
    if not os.path.isfile(path):
        raise ValueError(f"A5 源读失败：找不到 {path}（fail-loud）")
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    fdef = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    if fdef is None:
        raise ValueError(f"{rel}: 找不到函数 {func_name}（A5 fail-loud）")
    ctor = None
    for node in ast.walk(fdef):
        if isinstance(node, ast.Call):
            fn = node.func
            nm = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if nm == ctor_name:
                ctor = node
                break
    if ctor is None:
        raise ValueError(f"{rel}:{func_name} 未找到 {ctor_name}(...) 构造（A5 fail-loud）")
    kw = {k.arg: k.value for k in ctor.keywords if k.arg}
    return {key: _norm_kwarg_is_real(kw.get(key), rel, key) for key in keys}


@functools.lru_cache(maxsize=None)
def final_norm_source_info(mf_root: str) -> dict:
    """block 末 final_norm 是否为真 norm（`TransformerBlockSubmodules.layer_norm`，无条件 get_norm_cls）。

    返回 `{"final_norm": bool}`。定向 AST 读 `get_gpt_decoder_block_spec`（源在 block-spec 函数、非主入口）。
    """
    bits = _ast_norm_kwarg_bits(
        mf_root, _GPT_LAYER_SPECS_REL, "get_gpt_decoder_block_spec",
        "TransformerBlockSubmodules", ("layer_norm",))
    return {"final_norm": bits["layer_norm"]}


@functools.lru_cache(maxsize=None)
def mtp_norm_source_info(mf_root: str) -> dict:
    """MTP 层 enorm/hnorm 是否为真 norm（`get_mtp_layer_spec` 无条件 get_norm_cls）。

    返回 `{"mtp_enorm": bool, "mtp_hnorm": bool}`。定向 AST 读 `get_mtp_layer_spec`（源在 MTP-spec 函数、
    非主入口）。仅在 spec 真含 MTP 层时才被调用（无 MTP 层的 spec 不因此源读失败而 fail）。
    """
    bits = _ast_norm_kwarg_bits(
        mf_root, _MTP_REL, "get_mtp_layer_spec",
        "MultiTokenPredictionLayerSubmodules", ("enorm", "hnorm"))
    return {"mtp_enorm": bits["enorm"], "mtp_hnorm": bits["hnorm"]}


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


# ── 第三族 layer_norms：手写侧 pre-norm 在场判定（Z1 修复 2026-07-16）───────────────────
def _has_pre_attn_norm(attn_ops) -> bool:
    """attn 段是否含一个作为 **pre-attn 归一** 的 NORM（`ln1`）。

    判据：段内存在一个 `hand_category == NORM` 的 op，其输入含该层的**输入残差流**——层输入取 attn 段
    首个 op 的首个输入张量名（正常层 = `x`；`ln1` 即 `NORM(x)→ln1`，恒在场）。删掉 `ln1` 后，段首变
    linear_qkv/qkv（其输入是 `ln1` 载体），已无 NORM 以层输入为输入 → False（被报出）。只判在场，不判
    saves/shape/dtype（诚实边界）；MLA 段里的 q/kv_a_norm 输入是 lora 切片、非层输入，不误判为 pre-attn。
    """
    if not attn_ops:
        return False
    first_in = attn_ops[0].inputs[0] if attn_ops[0].inputs else None
    layer_in = getattr(first_in, "name", None)
    if layer_in is None:
        return False
    for op in attn_ops:
        if hand_category(op) == NORM:
            in_names = {getattr(t, "name", None) for t in (op.inputs or [])}
            if layer_in in in_names:
                return True
    return False


def _ffn_leads_with_norm(ffn_ops) -> bool:
    """ffn 段是否以 **pre_mlp_layernorm**（`ln2`, NORM）打头。

    判据：段内**首个重算子类别**（`hand_category` 非 None：LINEAR/LINEAR_GROUPED/NORM/ATTENTION/
    ACTIVATION）的 op 必须是 NORM。router/dispatch/combine/residual 等结构/opaque op 类别为 None、跳过。
    删掉 `ln2` 后，dense 段首个重算子变 `fc1`(MATMUL)、MoE 段变 `e_fc1`(MOE_GEMM) → 非 NORM → 报出
    （正是 Z1 的删-ln2 突变）。只判位置在场，不判 saves/shape/dtype（诚实边界）。
    """
    for op in ffn_ops:
        cat = hand_category(op)
        if cat is None:
            continue                     # 跳过 router/dispatch/combine/residual add（非重算子）
        return cat == NORM               # 首个重算子必须是 NORM=ln2
    return False


def _is_mla_attn(attn_ops) -> bool:
    """attn 段是否为 **MLA**（signature=含 `linear_kvb`；GQA 走融合 `qkv` 无此名）。与主循环里 MLA 段
    的判据一致。"""
    return bool(attn_ops) and any(getattr(op, "name", None) == "linear_kvb" for op in attn_ops)


def _seg_norm_names(ops) -> set:
    """段内**归一类**（`hand_category==NORM`）op 的名字集合（供次级 norm 按名点名在场判定）。"""
    return {op.name for op in (ops or []) if hand_category(op) == NORM}


def _is_head_segment(ops) -> bool:
    """段是否为 **head/final 段**（signature=含 `lm_head` 输出投影 op）。embedding/dsv4hyb 无此 op，故不误
    判；MTP 段虽也含 `lm_head`，但主循环先按 `eh_proj` 拦截 MTP 并 continue，不会走到这里。"""
    return any(getattr(op, "name", None) == "lm_head" for op in (ops or []))


def _head_leads_with_norm(head_ops) -> bool:
    """head/final 段是否以 **final_norm**（NORM）打头——即**首个重算子**是 NORM，位于 `lm_head` 投影之前。

    判据同 `_ffn_leads_with_norm`：跳过非重算子，首个 `hand_category` 非 None 的 op 必须是 NORM。head 段
    op 序=[final_norm(NORM), lm_head(LINEAR), logsoftmax(NORM), nll]；删 final_norm 后首个重算子变
    `lm_head`(LINEAR) → 报出（`logsoftmax` 那个 NORM 在 lm_head **之后**，不算 final_norm）。
    """
    for op in head_ops:
        cat = hand_category(op)
        if cat is None:
            continue
        return cat == NORM
    return False


def _is_mtp_layer(ops) -> bool:
    """段是否为 **MTP 层**（signature=含 `eh_proj`——MTP 专属的 embedding-hidden 投影）。

    用 `eh_proj`（而非 enorm/hnorm 名）判定：删 enorm/hnorm 之一做突变时，`eh_proj` 仍在 → 层仍被识别为
    MTP 而正确报出缺失（若用 enorm/hnorm 名判定，删掉后反而认不出 MTP 层）。MLA/GQA/head/embedding 段
    均无 `eh_proj`，不误判。
    """
    return any(getattr(op, "name", None) == "eh_proj" for op in (ops or []))


def _check_layer_norms(layer_type, attn_ops, ffn_ops, source_info, report):
    """第三族 **layer_norms** 的交叉校验：对一个 decoder-body 层，断言源码强制的层级 pre-norm 与（A5）
    MLA 潜向量 norm 在手写 op 图里**存在且就位**——

      - 源 `input_layernorm` 为真 norm → attn 段必须含一个 pre-attn NORM（`ln1`，输入为层输入残差流）；
        缺失 → `LayerNormFinding`。（注：`_MLA_CORR` 的 NORM delta 是**总 NORM 计数**对账，此族专管
        **具体 norm 的在场**，语义正交、不重复报同一 finding。）
      - 源 `pre_mlp_layernorm` 为真 norm → ffn 段**首个重算子是 NORM**（`ln2`）；否则 → `LayerNormFinding`
        （这正是 `moe_experts`/`mla_attn` 两族都 census 不到的删-ln2 突变）。
      - **A5 MLA 潜向量 q/kv norm**：仅当 attn 段是 MLA（含 `linear_kvb`）时——源 `q_layernorm`/`kv_layernorm`
        为真 norm（`_MLA_SPEC_FLAGS` 的 qk_layernorm=True 下解成真 norm）→ attn 段须含 `q_a_norm`/`kv_a_norm`
        两个 NORM；源解成 IdentityOp（qk 关）则**不**要求（无假阳）。

    **诚实边界**：只校验这些 norm 的**存在/位置**，不校验其 saves/shape/dtype。
    """
    report.layer_norm_checked.append(layer_type)
    if source_info.get("input_layernorm") and not _has_pre_attn_norm(attn_ops):
        report.layer_norm_findings.append(LayerNormFinding(
            layer_type, "input_layernorm",
            "attn 段缺少以层输入为输入的 pre-attn NORM（ln1）"))
    if source_info.get("pre_mlp_layernorm") and not _ffn_leads_with_norm(ffn_ops):
        report.layer_norm_findings.append(LayerNormFinding(
            layer_type, "pre_mlp_layernorm",
            "ffn 段首个重算子不是 NORM（缺失 pre_mlp_layernorm / ln2）"))
    # A5：MLA 潜向量 q/kv norm（仅 MLA attn 段；源真 norm 时才要求）。
    if _is_mla_attn(attn_ops):
        norm_names = _seg_norm_names(attn_ops)
        if source_info.get("q_layernorm") and "q_a_norm" not in norm_names:
            report.layer_norm_findings.append(LayerNormFinding(
                layer_type, "q_layernorm",
                "MLA attn 段缺少潜向量 q norm（q_a_norm）"))
        if source_info.get("kv_layernorm") and "kv_a_norm" not in norm_names:
            report.layer_norm_findings.append(LayerNormFinding(
                layer_type, "kv_layernorm",
                "MLA attn 段缺少潜向量 kv norm（kv_a_norm）"))


def _check_final_norm(layer_type, ops, final_info, report):
    """A5 **final_norm**：head/final 段（含 `lm_head`）须以 NORM（final_norm）打头，位于 lm_head 投影之前。

    源 `final_norm` 为真 norm（`TransformerBlockSubmodules.layer_norm`，无条件 get_norm_cls）→ 缺失即
    `LayerNormFinding`。只判存在/位置，不判 saves/shape/dtype（诚实边界）。
    """
    report.layer_norm_checked.append(layer_type)
    if final_info.get("final_norm") and not _head_leads_with_norm(ops):
        report.layer_norm_findings.append(LayerNormFinding(
            layer_type, "final_norm",
            "head/final 段首个重算子不是 NORM（缺失 final_norm，应在 lm_head 投影之前）"))


def _check_mtp_norms(layer_type, ops, mtp_info, report):
    """A5 **MTP enorm/hnorm**：MTP 段（含 `eh_proj`）须含 `enorm`/`hnorm` 两个 NORM。

    源 `enorm`/`hnorm` 为真 norm（`get_mtp_layer_spec` 无条件 get_norm_cls）→ 缺失即 `LayerNormFinding`。
    只判存在，不判 saves/shape/dtype（诚实边界）。MTP 内层 decoder 的 ln1/ln2/final_norm 本族不重复校验。
    """
    report.layer_norm_checked.append(layer_type)
    norm_names = _seg_norm_names(ops)
    if mtp_info.get("mtp_enorm") and "enorm" not in norm_names:
        report.layer_norm_findings.append(LayerNormFinding(
            layer_type, "mtp_enorm", "MTP 段缺少 embedding-norm（enorm）"))
    if mtp_info.get("mtp_hnorm") and "hnorm" not in norm_names:
        report.layer_norm_findings.append(LayerNormFinding(
            layer_type, "mtp_hnorm", "MTP 段缺少 hidden-norm（hnorm）"))


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
class LayerNormFinding:
    """一条**归一名册**漂移（第三族 layer_norms，Z1 + A5）：某手写层段缺失了源码强制的某 NORM op。

    `norm_slot` 词表（每项都溯源到真 mindformers 源，见 `_NORM_SLOT_SOURCE`）：
      - Z1 两个层级 pre-norm：`"input_layernorm"`（ln1）/ `"pre_mlp_layernorm"`（ln2）。
      - A5 次级 norm：`"q_layernorm"`/`"kv_layernorm"`（MLA 潜向量 q/kv norm，条件强制）、`"final_norm"`
        （block 末、lm_head 前，无条件）、`"mtp_enorm"`/`"mtp_hnorm"`（MTP 层两个 norm，无条件）。

    诚实边界：本条只表达该 norm 的**存在/位置**缺失，**不**涉及其 saves/shape/dtype。
    """
    layer_type: str
    norm_slot: str          # 见上词表
    detail: str

    @property
    def message(self) -> str:
        src = _NORM_SLOT_SOURCE.get(self.norm_slot, "源码强制该 norm")
        return (f"[opdag 归一名册缺失] 层 {self.layer_type!r} 的 {self.norm_slot}："
                f"{self.detail}——源据：{src}；手写 LayerSpec 却缺对应 NORM op。")


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
    # 第三族 layer_norms（Z1）：层级 pre-norm 缺失 → list[LayerNormFinding]（让 ok=False、strict raise）。
    layer_norm_findings: list = field(default_factory=list)
    # 被 layer_norms 族校验过的 decoder-body 层名（可读覆盖记录；与 delta-census 的 `covered` 分列，
    # 保持 `covered` 仅指 MLA/MoE 两族的语义——GQA/dense 层也参与 layer_norms 但不进 `covered`）。
    layer_norm_checked: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """无漂移**且**无提取失败**且**无层级 pre-norm 缺失即通过（不可用也视作「无从校验、无可报」→ True）。

        注意：提取失败也让 ok=False——一个已声明覆盖的族抽取失败意味着交叉校验无从验证它，
        若仍返回 True 就是假绿（见模块 docstring「提取失败 ≠ 合法无对应」）。layer_norms 族的缺失
        （删 ln1/ln2）同样让 ok=False（Z1 修复）。
        """
        return (not self.findings and not self.extraction_failures
                and not self.layer_norm_findings)

    def summary(self) -> str:
        if not self.available:
            return f"opdag 一致性校验：跳过（mindformers 源不可用：{self.mf_root!r}）"
        cov = ", ".join(sorted({c.family for c in self.covered})) or "（无）"
        status = "通过" if self.ok else "发现漂移/提取失败"
        head = (f"opdag 一致性校验：{status}；"
                f"覆盖层段={len(self.covered)}（{cov}）、未覆盖层={len(self.uncovered)}、"
                f"漂移={len(self.findings)}、提取失败={len(self.extraction_failures)}、"
                f"层级pre-norm校验={len(self.layer_norm_checked)}、"
                f"pre-norm缺失={len(self.layer_norm_findings)}")
        lines = [head]
        for f in self.findings:
            lines.append("  - " + f.message)
        for lf in self.layer_norm_findings:
            lines.append("  - " + lf.message)
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

    返回 `CrossCheckReport`（findings/covered/uncovered/extraction_failures/layer_norm_findings/
    available）。合法无对应的层（embedding/lm_head/mtp/gqa/dense）记 `uncovered`，**不**触发 strict；
    只有已声明覆盖族（MLA/MoE/layer_norms）抽取失败才记 `extraction_failures` 并触发 strict；
    layer_norms 族发现删 ln1/ln2 记 `layer_norm_findings` 并触发 strict（见模块 docstring）。
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

    # 第三族 layer_norms 的源侧信息（两个层级 pre-norm 是否真 norm）——解析一次，对所有 decoder-body
    # 层适用（DSv3 每层同一 TransformerLayer 结构；源事实在 MLA/非 MLA 两支一致）。解析失败=已声明覆盖
    # 族无从验证 → extraction_failures（不静默放行；见模块 docstring「提取失败 ≠ 合法无对应」）。
    ln_source_info = None
    try:
        ln_source_info = layer_norms_source_info(mf_root)
    except Exception as e:
        report.extraction_failures.append(("<decoder layers>", "layer_norms", str(e)))

    # A5 final_norm 源（block 末、lm_head 前的无条件 norm）——每个 spec 都有 head 段，故 eager 读一次。
    final_info = None
    try:
        final_info = final_norm_source_info(mf_root)
    except Exception as e:
        report.extraction_failures.append(("<final_norm>", "layer_norms", str(e)))

    # A5 MTP 源（enorm/hnorm）——仅在 spec **真含** MTP 层时才读（无 MTP 层的 spec 绝不因 MTP 源读失败
    # 而 fail；诚实边界：不为一个不存在的层段引入提取失败）。
    mtp_info = None
    if any(_is_mtp_layer(ls.ops) for ls in spec.layer_specs.values()):
        try:
            mtp_info = mtp_norm_source_info(mf_root)
        except Exception as e:
            report.extraction_failures.append(("<mtp layers>", "layer_norms", str(e)))

    for ltype, ls in spec.layer_specs.items():
        ops = ls.ops
        # A5：MTP 层先拦截——其 embedding/enorm/hnorm/eh_proj 前缀 + 内层 decoder + 共享 head 的包装结构
        # 不被 MLA/MoE 窗口建模（逐 op delta 会满屏假阳），故**路由出前两族 delta 家族**，只校验
        # enorm/hnorm 两个 MTP 专属 norm 的在场，记 uncovered（诚实边界）。
        if _is_mtp_layer(ops):
            if mtp_info is not None:
                _check_mtp_norms(ltype, ops, mtp_info, report)
            report.uncovered.append(
                (ltype, "MTP 层：enorm/hnorm 名册已校验；MLA/MoE delta 家族不建模 MTP 包装结构"))
            continue
        attn_ops, ffn_ops = _split_decoder(ops)
        matched = False
        # MLA 注意力段：signature = 段内含 `linear_kvb`（MLA 独有；GQA 走融合 `qkv` 无此名）。
        if attn_ops is not None and any(op.name == "linear_kvb" for op in attn_ops):
            matched |= _check_segment(ltype, attn_ops, _MLA_CORR, mf_root, report)
        # MoE 专家段：signature = ffn 段内含 moe_gemm（grouped-GEMM）op。
        if ffn_ops is not None:
            window = _moe_expert_window(ffn_ops)
            if window is not None:
                matched |= _check_segment(ltype, window, _MOE_CORR, mf_root, report)
        # 第三族 layer_norms：每个 decoder-body 层（同时有 attn/ffn 段，即有 h1 残差）都校验层级 pre-norm
        # 与（A5）MLA 潜向量 q/kv norm 的在场——这**不**改 matched/covered（保持 covered 仅指 MLA/MoE 两族；
        # GQA/dense 层仍记 uncovered），只在缺 norm 时记 layer_norm_findings。前两族窗口都 census 不到 ln2。
        if ln_source_info is not None and attn_ops is not None and ffn_ops is not None:
            _check_layer_norms(ltype, attn_ops, ffn_ops, ln_source_info, report)
        # A5 final_norm：head/final 段（非 decoder-body、含 `lm_head` 投影）须以 final_norm 打头。
        if final_info is not None and attn_ops is None and _is_head_segment(ops):
            _check_final_norm(ltype, ops, final_info, report)
        if not matched:
            report.uncovered.append(
                (ltype, "无 opdag 提取源（embedding/lm_head/mtp/gqa/dense 未抽取）"))

    # 漂移 **或** 提取失败 **或** 层级 pre-norm 缺失都是「非绿」——strict 下都要 raise（静默放行=假绿）。
    if report.findings or report.extraction_failures or report.layer_norm_findings:
        if strict:
            if (report.extraction_failures and not report.findings
                    and not report.layer_norm_findings):
                fams = ", ".join(sorted({f for _lt, f, _e in report.extraction_failures}))
                head = (f"opdag 一致性校验无法完成：已声明覆盖族 [{fams}] 的 op 图抽取失败——"
                        f"无从验证，strict 下不得静默放行。\n")
                raise OpdagConsistencyError(head + report.summary())
            raise OpdagConsistencyError(report.summary())
        if warn:
            warnings.warn(report.summary(), OpdagDriftWarning, stacklevel=2)
    return report
