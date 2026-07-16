"""A5（2026-07-16）：opdag 交叉校验第三族 layer_norms 的**次级 norm 守卫**。

背景：Z1 已补上两个层级 pre-norm（ln1/ln2）；但源码还强制若干**次级 norm**，前两族（mla_attn/
moe_experts）窗口同样 census 不到，静默删掉即又一 F1 类漏建。本文件守 A5 新加的三类次级 norm：

  1. **MLA 潜向量 q/kv norm**（`q_a_norm`/`kv_a_norm`）——源 `MLASelfAttention.q_layernorm/kv_layernorm`
     `= get_norm_cls(fused_norm) if qk_layernorm else IdentityOp`（gpt_layer_specs.py:155/156）；用与
     `opdag_mla_census` 同一 `_MLA_SPEC_FLAGS`（qk_layernorm=True）解析 → 真 norm，故要求在场。
  2. **final_norm**——源 `TransformerBlockSubmodules.layer_norm=get_norm_cls(config.fused_norm)`
     （gpt_layer_specs.py:254，无条件）；手写 head/final 段须以 final_norm 打头。
  3. **MTP enorm/hnorm**——源 `get_mtp_layer_spec` 无条件绑定 `enorm/hnorm=get_norm_cls(fused_norm)`
     （multi_token_prediction.py:102/103）；手写 MTP 段须含二者。

测法：
  - **源门控**（缺源则 skip）：correct spec 通过；删任一次级 norm → 报出对应 slot、ok False、strict raise。
  - **源无关**（恒跑）：detector（`_is_mla_attn`/`_is_head_segment`/`_is_mtp_layer`）与位置判定
    （`_head_leads_with_norm`）的结构不变量；qk 关时的**不要求**（无假阳）门控。
"""
import copy
import dataclasses
import os

import pytest

from cost_eval.build_llm import build_llm_spec
from cost_eval.presets import deepseek_v3, deepseek_v4
from validate_dsv3 import build_dsv3_spec
from cost_eval.opdag.crosscheck import (
    CrossCheckReport, OpdagConsistencyError, default_mf_root,
    validate_against_opdag, _split_decoder, _is_mla_attn, _is_mtp_layer,
    _is_head_segment, _head_leads_with_norm, _check_layer_norms,
)

_HAS_SOURCE = os.path.isdir(default_mf_root())
_needs_source = pytest.mark.skipif(not _HAS_SOURCE, reason="mindformers 源不可用（CI）")


def _mtp_spec():
    """DSv3(4 层) + 1 个 MTP 层：一个 spec 同时含 MLA 层（q_a_norm/kv_a_norm）、MTP 层（enorm/hnorm）、
    head 段（final_norm）——A5 三类次级 norm 都能在此单一 spec 上探针。spec 构造不读 mindformers 源。"""
    return build_llm_spec(dataclasses.replace(deepseek_v3(4), mtp_num_layers=1))


def _is_mla_ls(ls):  return any(o.name == "linear_kvb" for o in ls.ops)
def _is_mtp_ls(ls):  return any(o.name == "eh_proj" for o in ls.ops)
def _is_head_ls(ls): return any(o.name == "lm_head" for o in ls.ops) and not _is_mtp_ls(ls)


def _del_op(spec, pred, opname):
    """在 spec 里首个 `pred(ls)` 为真且含名为 `opname` 的 op 的层里删掉该 op，返回被改层名。"""
    for lt, ls in spec.layer_specs.items():
        if pred(ls) and any(o.name == opname for o in ls.ops):
            ls.ops = [o for o in ls.ops if o.name != opname]
            return lt
    raise AssertionError(f"未找到含 {opname} 的目标层")


def _slots(rep):
    return sorted(f.norm_slot for f in rep.layer_norm_findings)


# ══════════════════════════ 源门控：correct 通过 + 每类删除报出 ══════════════════════════
@_needs_source
def test_correct_specs_have_no_secondary_norm_false_positives():
    """零假阳基线：DSv3、DSv3+MTP、DSv4 三个 correct spec 都 ok=True，且无任何 pre-norm/次级-norm 缺失。"""
    sp3, _d, _f = build_dsv3_spec(4)
    r3 = validate_against_opdag(sp3, warn=False)
    assert r3.ok is True
    assert not r3.layer_norm_findings and not r3.findings

    spm = _mtp_spec()
    rm = validate_against_opdag(spm, warn=False)
    assert rm.ok is True
    assert not rm.layer_norm_findings
    # MLA 层、MTP 层、head 段都被第三族校验过（次级 norm 三类全覆盖）。
    assert set(rm.layer_norm_checked) >= {"mla_dense", "mla_moe", "mtp", "lm_head"}

    sp4 = build_llm_spec(deepseek_v4(4))          # 含 MTP + 混合注意力（非 MLA signature）+ head
    r4 = validate_against_opdag(sp4, warn=False)
    assert r4.ok is True
    assert not r4.layer_norm_findings


@_needs_source
def test_delete_mla_latent_q_norm_is_reported():
    """删 MLA 潜向量 q_a_norm → 报 q_layernorm 缺失、ok False、strict raise（源 qk_layernorm=True 时强制）。"""
    spec = _mtp_spec()
    changed = _del_op(spec, _is_mla_ls, "q_a_norm")
    rep = validate_against_opdag(spec, warn=False)
    assert rep.ok is False
    assert any(f.norm_slot == "q_layernorm" and f.layer_type == changed
               for f in rep.layer_norm_findings)
    assert "q_layernorm" in rep.summary()
    with pytest.raises(OpdagConsistencyError) as ei:
        validate_against_opdag(spec, strict=True, warn=False)
    assert "q_layernorm" in str(ei.value)


@_needs_source
def test_delete_mla_latent_kv_norm_is_reported():
    """删 MLA 潜向量 kv_a_norm → 报 kv_layernorm 缺失、ok False、strict raise。"""
    spec = _mtp_spec()
    changed = _del_op(spec, _is_mla_ls, "kv_a_norm")
    rep = validate_against_opdag(spec, warn=False)
    assert rep.ok is False
    assert any(f.norm_slot == "kv_layernorm" and f.layer_type == changed
               for f in rep.layer_norm_findings)
    with pytest.raises(OpdagConsistencyError):
        validate_against_opdag(spec, strict=True, warn=False)


@_needs_source
def test_delete_final_norm_is_reported():
    """删 head/final 段 final_norm → 报 final_norm 缺失、ok False、strict raise（源无条件强制）。
    head 段是 uncovered（非 MLA/MoE）→ 此缺失**只**由第三族抓到，前两族 delta 依旧 0 findings。"""
    spec = _mtp_spec()
    changed = _del_op(spec, _is_head_ls, "final_norm")
    rep = validate_against_opdag(spec, warn=False)
    assert rep.ok is False
    assert any(f.norm_slot == "final_norm" and f.layer_type == changed
               for f in rep.layer_norm_findings)
    assert not rep.findings                        # head 段无 delta 家族覆盖 → 只第三族补此洞
    assert "final_norm" in rep.summary()
    with pytest.raises(OpdagConsistencyError) as ei:
        validate_against_opdag(spec, strict=True, warn=False)
    assert "final_norm" in str(ei.value)


@_needs_source
def test_delete_mtp_enorm_is_reported():
    """删 MTP enorm → 报 mtp_enorm 缺失、ok False、strict raise（源无条件强制；MTP 段路由出 delta 家族）。"""
    spec = _mtp_spec()
    changed = _del_op(spec, _is_mtp_ls, "enorm")
    rep = validate_against_opdag(spec, warn=False)
    assert rep.ok is False
    assert any(f.norm_slot == "mtp_enorm" and f.layer_type == changed
               for f in rep.layer_norm_findings)
    assert not rep.findings                        # MTP 段不进 MLA/MoE delta 家族
    with pytest.raises(OpdagConsistencyError) as ei:
        validate_against_opdag(spec, strict=True, warn=False)
    assert "mtp_enorm" in str(ei.value)


@_needs_source
def test_delete_mtp_hnorm_is_reported():
    """删 MTP hnorm → 报 mtp_hnorm 缺失、ok False、strict raise。"""
    spec = _mtp_spec()
    changed = _del_op(spec, _is_mtp_ls, "hnorm")
    rep = validate_against_opdag(spec, warn=False)
    assert rep.ok is False
    assert any(f.norm_slot == "mtp_hnorm" and f.layer_type == changed
               for f in rep.layer_norm_findings)
    with pytest.raises(OpdagConsistencyError):
        validate_against_opdag(spec, strict=True, warn=False)


@_needs_source
def test_missing_mtp_source_does_not_fail_non_mtp_spec(monkeypatch):
    """诚实边界：无 MTP 层的 spec（DSv3）**不**读 MTP 源——即便 MTP 源读会抛，DSv3 也不因此 fail。
    注入 `mtp_norm_source_info` 抛异常，跑 DSv3（无 MTP 层）仍 ok=True、无 extraction_failure。"""
    import cost_eval.opdag.crosscheck as _cc

    def boom(_root):
        raise RuntimeError("injected mtp source failure")

    monkeypatch.setattr(_cc, "mtp_norm_source_info", boom)
    sp3, _d, _f = build_dsv3_spec(4)              # 无 MTP 层
    rep = validate_against_opdag(sp3, warn=False)
    assert rep.ok is True
    assert not any(fam == "layer_norms" and "mtp" in lt.lower()
                   for (lt, fam, _e) in rep.extraction_failures)


# ══════════════════════════ 源无关：detector / 位置判定 / qk 门控（恒跑） ══════════════════════════
def test_mla_detector_specific_to_mla_attn():
    """`_is_mla_attn`：只对含 `linear_kvb` 的 MLA attn 段为真；head/embedding 段为假。"""
    spec = _mtp_spec()
    for lt, ls in spec.layer_specs.items():
        a, f = _split_decoder(ls.ops)
        if a is not None and _is_mla_ls(ls):
            assert _is_mla_attn(a) is True
    # head 段（非 body）不是 MLA
    head = next(ls for ls in spec.layer_specs.values() if _is_head_ls(ls))
    assert _is_mla_attn(list(head.ops)) is False


def test_head_detector_and_final_norm_position():
    """`_is_head_segment` 精确命中 head（含 lm_head），排除 embedding；`_head_leads_with_norm` 判位置：
    正常 head 首个重算子是 final_norm(NORM)；删 final_norm → 首个重算子变 lm_head(LINEAR) → False。"""
    spec = _mtp_spec()
    head_ls = next(ls for ls in spec.layer_specs.values() if _is_head_ls(ls))
    head_ops = list(head_ls.ops)
    assert _is_head_segment(head_ops) is True
    assert _head_leads_with_norm(head_ops) is True
    # 删 final_norm → 首个重算子不再是 NORM
    no_fn = [o for o in head_ops if o.name != "final_norm"]
    assert _head_leads_with_norm(no_fn) is False
    # embedding 段（无 lm_head）不被判为 head
    emb_ls = spec.layer_specs["embedding"]
    assert _is_head_segment(list(emb_ls.ops)) is False


def test_mtp_detector_survives_norm_deletion():
    """`_is_mtp_layer` 用 `eh_proj`（非 enorm/hnorm 名）判定 → 删掉 enorm/hnorm 后仍认得出 MTP 层
    （否则删掉后反而识别不出、报不出缺失）。MLA/head/embedding 段无 `eh_proj` → 不误判。"""
    spec = _mtp_spec()
    mtp_ls = next(ls for ls in spec.layer_specs.values() if _is_mtp_ls(ls))
    ops = list(mtp_ls.ops)
    assert _is_mtp_layer(ops) is True
    stripped = [o for o in ops if o.name not in ("enorm", "hnorm")]
    assert _is_mtp_layer(stripped) is True          # eh_proj 仍在 → 仍识别为 MTP
    # 非 MTP 段不误判
    mla_ls = next(ls for ls in spec.layer_specs.values() if _is_mla_ls(ls))
    assert _is_mtp_layer(list(mla_ls.ops)) is False
    assert _is_mtp_layer(list(spec.layer_specs["embedding"].ops)) is False


def test_qk_layernorm_off_source_requires_no_latent_norm():
    """无假阳门控（源无关）：源 `q_layernorm`/`kv_layernorm` 解成 IdentityOp（qk_layernorm 关）时，
    即便手写 MLA attn 段没有 q_a_norm/kv_a_norm，`_check_layer_norms` 也**不**报缺失。"""
    spec = build_dsv3_spec(4)[0]                    # 不需 mindformers 源
    attn = ffn = None
    for ls in spec.layer_specs.values():
        a, f = _split_decoder(ls.ops)
        if a is not None and _is_mla_ls(ls):
            attn, ffn = a, f
            break
    assert attn is not None
    stripped = [o for o in attn if o.name not in ("q_a_norm", "kv_a_norm")]
    rep = CrossCheckReport(available=True, mf_root="x")
    src_off = {"input_layernorm": True, "pre_mlp_layernorm": True,
               "q_layernorm": False, "kv_layernorm": False}
    _check_layer_norms("mla_x", stripped, ffn, src_off, rep)
    assert not any(f.norm_slot in ("q_layernorm", "kv_layernorm")
                   for f in rep.layer_norm_findings)

    # 对照：源开启（True）且缺 q_a/kv_a_norm → 两条都报出。
    rep2 = CrossCheckReport(available=True, mf_root="x")
    src_on = {"input_layernorm": True, "pre_mlp_layernorm": True,
              "q_layernorm": True, "kv_layernorm": True}
    _check_layer_norms("mla_x", stripped, ffn, src_on, rep2)
    assert _slots(rep2) == ["kv_layernorm", "q_layernorm"]
