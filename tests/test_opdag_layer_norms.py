"""Z1（2026-07-16）：opdag 交叉校验的**第三族 layer_norms**——层级 pre-norm 名册守卫。

背景（已核验的真缺陷）：前两族（`mla_attn` / `moe_experts`）的比较窗口把**层级两个强制 pre-norm**
漏在外——`moe_experts` 窗口 `[首个 moe_gemm … 末个 moe_gemm]` 排除了 `ln2`（pre_mlp_layernorm，
位于 router 之前）；`mla_attn` 段以 `add1→h1` 收尾，`ln2` 落 ffn 段被两族都不 census。于是删掉某 MoE
层的 `ln2`（F1 类漏建：少一个 pre-norm）竟能让 `validate_against_opdag(...).ok == True`、0 findings。

源忠实依据（真源 `gpt_layer_specs.py:get_gpt_layer_local_spec`）：每个 decoder 层的
`TransformerLayerSubmodules` 在 MLA（:162/164）与非 MLA（:172/183）两支都**无条件**绑定
`input_layernorm=get_norm_cls(fused_norm)` 与 `pre_mlp_layernorm=get_norm_cls(fused_norm)` 两个**真**
norm；`IdentityOp` 只用于**可选** q/k layernorm。第三族据此断言这两个层级 pre-norm 在手写 op 图里在场。

本文件两测：
  1. **源门控**的突变探针（缺 mindformers 源则 skip）：correct spec 通过；删 ln2 → 报 pre_mlp 缺失、
     ok False、strict raise（且前两族 delta 依旧 0 findings——坐实这是第三族补的洞）。
  2. **源无关**的结构不变量（恒跑）：每个 decoder-body 层 attn 段有 pre-attn NORM、ffn 段首 op 是
     NORM（ln2）——即便无源的机器也守住突变。
"""
import copy
import os

import pytest

from validate_dsv3 import build_dsv3_spec
from cost_eval.model_spec import OpType
from cost_eval.opdag.crosscheck import (
    NORM, OpdagConsistencyError, default_mf_root, hand_category,
    validate_against_opdag, _split_decoder, _has_pre_attn_norm, _ffn_leads_with_norm,
)

_HAS_SOURCE = os.path.isdir(default_mf_root())
_needs_source = pytest.mark.skipif(not _HAS_SOURCE, reason="mindformers 源不可用（CI）")


def _delete_ln2_from_a_moe_layer(spec):
    """在 spec 的某 MoE 层（ops 含 moe_gemm）里删掉名为 `ln2` 的 op，返回被改的层名（未找到 → None）。"""
    for ltype, ls in spec.layer_specs.items():
        if any(op.type == OpType.MOE_GEMM for op in ls.ops):
            new_ops = [op for op in ls.ops if op.name != "ln2"]
            assert len(new_ops) == len(ls.ops) - 1, f"{ltype} 未找到 ln2 可删"
            ls.ops = new_ops
            return ltype
    return None


@_needs_source
def test_correct_spec_passes_but_deleted_ln2_is_reported():
    """源门控回归守卫（F1 类）：correct DSv3 spec 通过；删某 MoE 层 ln2 → 第三族报出 pre_mlp 缺失。"""
    spec, _d, _full = build_dsv3_spec(4)

    # ── 基线：正确 spec 通过（两个层级 pre-norm 都在场）──────────────────────────────
    rep = validate_against_opdag(spec, warn=False)
    assert rep.ok is True
    assert not rep.layer_norm_findings
    assert rep.layer_norm_checked                       # decoder-body 层确实被第三族校验过

    # ── 突变：删一个 MoE 层的 ln2（pre_mlp_layernorm）──────────────────────────────
    mutated = copy.deepcopy(spec)
    changed = _delete_ln2_from_a_moe_layer(mutated)
    assert changed is not None, "未找到含 moe_gemm 的 MoE 层"

    rep2 = validate_against_opdag(mutated, warn=False)
    assert rep2.ok is False                             # 现在报出（此前假绿）
    assert rep2.layer_norm_findings                     # 第三族抓到缺失
    lf = rep2.layer_norm_findings
    assert any(f.norm_slot == "pre_mlp_layernorm" and f.layer_type == changed for f in lf)
    # 报告文本点名 ffn / pre_mlp（可读失败原因）
    assert "pre_mlp" in rep2.summary()
    # 坐实这是第三族补的洞：前两族的 delta-census 对删 ln2 **依旧盲**（0 delta findings）。
    assert not rep2.findings

    # strict：删 ln2 即 raise OpdagConsistencyError，且消息点名 pre_mlp。
    with pytest.raises(OpdagConsistencyError) as ei:
        validate_against_opdag(mutated, strict=True, warn=False)
    assert "pre_mlp" in str(ei.value)


def test_structural_invariant_every_decoder_body_has_both_prenorms():
    """源无关的结构不变量（恒跑，不依赖 mindformers 源）：DSv3 每个 decoder-body 层（`_split_decoder`
    同时给出 attn/ffn 两段）都有一个 pre-attn NORM（ln1）、且 ffn 段首个 op 是 NORM（ln2）。

    这守住删-ln2 突变即便在无源机器上：只要 build_dsv3_spec 的 op 图结构漂移（漏 ln1/ln2）就在此失守。
    """
    spec, _d, _full = build_dsv3_spec(4)
    body_layers = []
    for ltype, ls in spec.layer_specs.items():
        attn_ops, ffn_ops = _split_decoder(ls.ops)
        if attn_ops is None or ffn_ops is None:
            continue                                    # embedding/lm_head 非 body → 跳过
        body_layers.append(ltype)
        # ln1：attn 段含以层输入为输入的 pre-attn NORM
        assert _has_pre_attn_norm(attn_ops), f"{ltype}: attn 段缺 pre-attn NORM（ln1）"
        # ln2：ffn 段**首个 op** 就是 NORM（pre_mlp_layernorm）
        assert ffn_ops, f"{ltype}: ffn 段为空"
        assert hand_category(ffn_ops[0]) == NORM, f"{ltype}: ffn_ops[0] 不是 NORM（ln2）"
        assert _ffn_leads_with_norm(ffn_ops), f"{ltype}: ffn 段首个重算子不是 NORM"
    assert body_layers, "DSv3 应至少有一个 decoder-body 层"


def test_helpers_detect_missing_prenorms_directly():
    """辅助判定单测（源无关）：删掉 ln1 → `_has_pre_attn_norm` False；删掉 ln2 → `_ffn_leads_with_norm`
    False。守住第三族判定不因重构而失灵。"""
    spec, _d, _full = build_dsv3_spec(4)
    # 取任一 body 层
    attn_ops = ffn_ops = None
    for ls in spec.layer_specs.values():
        a, f = _split_decoder(ls.ops)
        if a is not None and f is not None:
            attn_ops, ffn_ops = a, f
            break
    assert attn_ops is not None
    # 正常：两者都在场
    assert _has_pre_attn_norm(attn_ops) and _ffn_leads_with_norm(ffn_ops)
    # 删 ln1（attn 段首个 NORM=层输入归一）→ 判为缺失
    attn_no_ln1 = [op for op in attn_ops if op.name != "ln1"]
    assert not _has_pre_attn_norm(attn_no_ln1)
    # 删 ln2（ffn 段首个 NORM）→ 判为缺失
    ffn_no_ln2 = [op for op in ffn_ops if op.name != "ln2"]
    assert not _ffn_leads_with_norm(ffn_no_ln2)
