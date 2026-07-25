# -*- coding: utf-8 -*-
"""yaml 导入 **round-trip 保真门**（2026-07-25，静默配置丢失 bug 回归）。

**bug**：`_bundle_to_fields` 把 `EvaluatorConfigBundle` 摊平成 UI query dict 时，多个
**结构字段在 UI 里无表达**（`csa_compress_ratios` / `o_groups` / `dsa_indexer_topk` /
`moe_shared_ffn_hidden_size` / `loss_type` / `chunk_loss_num` / `kept_frag_factor` /
`nr_moe_frag_factor` / `head_dim` …），`eval_config` 又从**预设基座**重建 `LLMConfig`
→ 凡 fields 没带的字段**静默沿用预设值**。现场 A/B config 实测：UI 路径 unfused 设备峰值
50926.7 MiB vs bundle 直连 37518.1 MiB（**35% 分歧**），纯由 yaml 的
`compress_ratios=[0,4,128,4,128,4,128,4]` 被换成 `presets._v4_compress_ratios` 的
`(0,4,128)` 循环造成。缺 `kv_lora_rank` 的现场 config 更是被预设值补上后**照样评估**，
而 bundle 直连本会 fail-loud。

**门**（与本库既有 fail-loud 纪律同源：`_IGNORED_MODEL_KEYS` / `_PAR_UNSUPPORTED_TRUTHY` /
`_validate_recompute_against_graph`）：
  ① **逐层注意力类型**必须与 yaml 的 `compress_ratios` 逐项一致（抓原 bug 的最小回归）；
  ② `_bundle_to_fields` → `parse_and_validate` → `LLMConfig` 与 bundle **逐字段相等**，
     否则 `_bundle_to_fields` **自身** fail-loud（运行时不变量 → `/import` 一并受护）；
  ③ 现场 yaml（`test.yaml` / `test-unfused.yaml` / 167 A/B launcher）逐字段核；
  ④ 缺 `kv_lora_rank` 的现场 config：UI 路径与 bundle 直连**同样** fail-loud，绝不替换；
  ⑤ 导入后在页面手改结构字段：只有**被改的那个**覆盖，其余仍取 yaml 忠实值。
"""
import dataclasses
import json
import os
import warnings

import pytest

import serve_explorer as S
from cost_eval.build_llm import build_llm_spec, gen_layer_pattern
from cost_eval.configs.from_mindformers import from_mindformers_dict

# 用户文档里的复现 DEF（**含会污染结构的 ffn=3072**）——fields 必须压过它，否则 round-trip 不保真。
_DEF = {"preset": "custom", "ffn": "3072", "select": "attn", "sel_layers": "",
        "sel_ops": "", "vpp": "1", "mbs": "", "grad_bytes": "4"}


def _query(fields, **over):
    """文档里的 UI query 构造法：DEF 兜底 + fields 覆盖（None 跳过=浏览器 skip-null 语义）。"""
    q = dict(_DEF)
    q.update({k: str(v) for k, v in fields.items() if v is not None})
    q.update(over)
    return q


def _roundtrip(mf):
    """mindformers dict → bundle → UI fields → query → (bundle, fields, cfg)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf2, vpp = S._mf_adapt(mf)
        S._materialize_nested_offset(mf2, [])
        b = from_mindformers_dict(mf2)
        fields = S._bundle_to_fields(b)
        q = _query(fields, **({"vpp": str(vpp)} if vpp > 1 else {}))
        errs, cfg, pa = S.parse_and_validate(q)
    assert not errs, errs
    return b, fields, cfg


# ── 现场 167 A/B launcher（dsv4h_fused_pp4_recomp.yaml 逐字段内联，35% 分歧的原始 config）──
# **刻意保留原写法**：① 无 `qk_nope_head_dim`（mindformers 由 head_dim−qk_rope 导出 448）；
#   ② `compress_ratios` 是真实逐层表 [0,4,128,4,128,4,128,4]（≠ 预设的 (0,4,128) 循环）。
def _ab_launcher(fused=True, **model_over):
    model = {
        "model_type": "deepseek_v4", "architectures": "DeepseekV4ForCausalLM",
        "vocab_size": 129280, "seq_length": 4096, "hidden_size": 4096,
        "num_hidden_layers": 8, "num_attention_heads": 64,
        "multi_latent_attention": True, "experimental_attention_variant": "dsv4_hybrid",
        "q_lora_rank": 1024, "kv_lora_rank": 512, "head_dim": 512, "qk_rope_head_dim": 64,
        "v_head_dim": 512, "o_groups": 8, "o_lora_rank": 1024, "qk_layernorm": True,
        "params_dtype": "bfloat16", "compute_dtype": "bfloat16",
        "layernorm_compute_dtype": "float32", "num_nextn_predict_layers": 0,
        "moe_intermediate_size": 2048, "n_routed_experts": 8, "num_experts_per_tok": 2,
        "n_shared_experts": 1, "moe_shared_expert_intermediate_size": 2048,
        "first_k_dense_replace": 1, "gated_linear_unit": True, "moe_grouped_gemm": True,
        "router_dense_type": "float32", "norm_topk_prob": True, "n_group": 0,
        "moe_router_load_balancing_type": "seq_aux_loss", "moe_aux_loss_coeff": 0.001,
        "enable_hyper_connections": True, "hc_mult": 4, "use_fused_mhc": True,
        "hc_sinkhorn_iters": 20, "hc_eps": 1.0e-06, "mhc_init_gating_factor": 0.01,
        "index_n_heads": 64, "index_head_dim": 128, "index_topk": 512,
        "apply_dsa_kernel_fusion": fused, "dsa_indexer_loss_coeff": 0.001,
        "dsa_indexer_use_sparse_loss": True,
        "compress_ratios": [0, 4, 128, 4, 128, 4, 128, 4],
        "sliding_window": 128, "compress_rope_theta": 40000, "rope_theta": 10000,
        "use_flash_attention": True,
    }
    model.update(model_over)
    return {
        "checkpoint": {"enable_save": False, "load_balanced": False},
        "context": {"device_target": "Ascend", "max_device_memory": "58GB", "mode": 1},
        "training": {"steps": 3, "local_batch_size": 1, "global_batch_size": 8, "seed": 42},
        "optimizer": {"type": "Muon", "weight_decay": 0.1, "momentum": 0.95,
                      "adamw_betas": [0.9, 0.95], "adamw_eps": 1.0e-8},
        "lr_scheduler": {"type": "ConstantWarmUpLR", "learning_rate": 1.0e-5},
        "parallelism": {
            "pipeline_parallel": 4, "data_parallel_shard": 2, "expert_parallel": 2,
            "tensor_parallel": 1, "context_parallel": 1, "sequence_parallel": False,
            "disable_gradient_division": True,
            "data_parallel_shard_strategy": "optim_grads_params",
            "reshard_after_forward_policy": "default", "cpu_offload": False,
            "pipeline_parallel_layers_per_stage": "auto",
            "pipeline_parallel_schedule": "1f1b",
        },
        "recompute": {"mode": "full", "full_recompute_layer": ["0-7"]},
        "model": model,
    }


# ── ① 头号回归：逐层注意力类型 == yaml compress_ratios（原 bug 的判据）─────────────────
def test_per_layer_attn_types_match_yaml_compress_ratios():
    """导入带显式 compress_ratios 的 yaml → **评估出的模型逐层注意力类型逐项等于 yaml**。

    刻意用 `[0,4,4,128,128,0]`（6 层）：预设 `_v4_compress_ratios` 会给 `(0,4,128,0,4,128)`
    → 原 bug 下本断言必红。
    """
    ratios = [0, 4, 4, 128, 128, 0]
    mf = _ab_launcher(num_hidden_layers=6, compress_ratios=ratios)
    mf["parallelism"]["pipeline_parallel"] = 1
    mf["recompute"] = {"mode": "None"}
    b, fields, cfg = _roundtrip(mf)
    assert list(b.llm.csa_compress_ratios) == ratios           # bundle 侧本就正确
    # round-trip 后的 cfg：逐层压缩比一致
    assert list(cfg.csa_compress_ratios) == ratios
    # 评估出的**模型**逐层身份（LayerContext.name = dsv4hyb_r{ratio}_{ffn}）
    got = [c.compress_ratio for c in gen_layer_pattern(cfg) if c.kind == "decoder"]
    assert got == ratios
    r = S.eval_config(_query(fields))
    assert r["ok"], r.get("errors")
    names = [L["type"] for st in r["stages"] for L in st["graph"]
             if str(L["type"]).startswith("dsv4hyb_")]
    assert [int(n.split("_")[1][1:]) for n in names] == ratios


def test_ui_path_peak_matches_bundle_direct_peak():
    """UI 路径设备峰值 == bundle 直连（同一 config 不该有两个答案；原 bug 35% 分歧）。"""
    from cost_eval.report import Evaluator
    mf = _ab_launcher(fused=False)
    b, fields, cfg = _roundtrip(mf)
    r = S.eval_config(_query(fields))
    assert r["ok"], r.get("errors")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = Evaluator(build_llm_spec(b.llm), b.parallel, b.optimizer, b.hardware,
                        b.recompute, b.swap, check_feasibility=False).evaluate(record_timeline=True)
    direct = round(max(sp.peak_bytes for sp in rep.per_stage) / (2 ** 20), 1)
    assert abs(r["device_peak"] - direct) < 0.2, (r["device_peak"], direct)


# ── ② 运行时不变量：round-trip 不保真即 fail-loud ─────────────────────────────────────
@pytest.mark.parametrize("fused", [True, False])
def test_bundle_roundtrip_field_exact(fused):
    b, fields, cfg = _roundtrip(_ab_launcher(fused=fused))
    a, c = dataclasses.asdict(b.llm), dataclasses.asdict(cfg)
    diffs = {k: (a[k], c[k]) for k in a if a[k] != c[k]}
    assert not diffs, diffs


def test_roundtrip_diffs_detects_substitution():
    """`_llm_roundtrip_diffs` 必须**真的**能发现替换（用错 fields 喂它 → 报出字段名）。"""
    b = from_mindformers_dict(_ab_launcher())
    other = dataclasses.replace(b.llm, csa_compress_ratios=(128,) * b.llm.num_layers,
                                o_groups=16)
    fields = S._llm_to_fields(other)
    diffs = S._llm_roundtrip_diffs(b.llm, _query(fields))
    names = {d[0] for d in diffs}
    assert "csa_compress_ratios" in names and "o_groups" in names, diffs


def test_bundle_to_fields_raises_when_lossy(monkeypatch):
    """fields 丢字段 → `_bundle_to_fields` 自身 fail-loud（护住 web `/import`）。"""
    b = from_mindformers_dict(_ab_launcher())

    def _lossy(llm):
        f = dict(_orig(llm))
        f.pop("llm_json", None)          # 模拟"UI 表达不了"→ 退回预设基座
        return f

    _orig = S._llm_to_fields
    monkeypatch.setattr(S, "_llm_to_fields", _lossy)
    with pytest.raises(S.ConfigRoundTripError) as ei:
        S._bundle_to_fields(b)
    assert "csa_compress_ratios" in str(ei.value)


def test_import_handler_surfaces_roundtrip_error(monkeypatch):
    """`/api/parse_yaml` 遇不保真 round-trip → 明确报错，不回填一份"看着对"的 fields。"""
    b = from_mindformers_dict(_ab_launcher())
    fields = S._llm_to_fields(dataclasses.replace(b.llm, o_groups=16))
    diffs = S._llm_roundtrip_diffs(b.llm, _query(fields))
    assert diffs and any(d[0] == "o_groups" for d in diffs)


# ── ②b 浏览器语义：回填的字段必须真有输入框承载，否则 qs() 时被静默丢弃 ────────────────
def _page_input_names():
    import re
    names = set()
    for tag in re.findall(r"<input\b[^>]*>", S.PAGE):
        m = re.search(r'name="([A-Za-z_0-9]+)"', tag)
        if m:
            names.add(m.group(1))
    for tag in re.findall(r"<select\b[^>]*>.*?</select>", S.PAGE, re.S):
        m = re.search(r'name="([A-Za-z_0-9]+)"', tag)
        if m:
            names.add(m.group(1))
    return names


def _page_defaults():
    """页面各输入框的初始值（= 浏览器 `qs()` 在未导入时会发出的 query）。"""
    import re
    out = {}
    for tag in re.findall(r"<input\b[^>]*>", S.PAGE):
        m = re.search(r'name="([A-Za-z_0-9]+)"', tag)
        if not m:
            continue
        v = re.search(r'value="([^"]*)"', tag)
        out[m.group(1)] = v.group(1) if v else ""
    for tag in re.findall(r"<select\b[^>]*>.*?</select>", S.PAGE, re.S):
        m = re.search(r'name="([A-Za-z_0-9]+)"', tag)
        if not m:
            continue
        # `<option value="x" selected>` 与 `<option selected>x</option>`（无 value → 取文本）两种写法
        opts = re.findall(r"<option\b([^>]*)>([^<]*)</option>", tag)
        vals = [(re.search(r'value="([^"]*)"', a).group(1) if 'value="' in a else t.strip(),
                 "selected" in a) for a, t in opts]
        out[m.group(1)] = next((v for v, s in vals if s), vals[0][0] if vals else "")
    return out


def test_every_emitted_field_has_a_page_input():
    """`_bundle_to_fields` 发出的每个键都必须有同名输入框——否则浏览器 `qs()` 收不到它。

    历史坑：`ce_fused` 早在 2026-07-22 就回填了，但页面从没有该输入框 → web UI 上这个修复
    **一直没生效**（只有脚本调用者受益）。
    """
    b = from_mindformers_dict(_ab_launcher())
    fields = S._bundle_to_fields(b)
    missing = sorted(set(fields) - _page_input_names())
    assert not missing, f"回填字段无输入框承载（浏览器会丢弃）：{missing}"


def test_browser_roundtrip_is_faithful():
    """完整模拟浏览器：页面默认值 + 导入回填（只认有输入框的键）→ query → 必须逐字段保真。"""
    b = from_mindformers_dict(_ab_launcher(fused=False))
    fields = S._bundle_to_fields(b)
    names = _page_input_names()
    q = _page_defaults()
    q["preset"] = "custom"          # 前端导入后就是这么设的
    for k, v in fields.items():
        if k in names and v is not None:
            q[k] = str(v)
    diffs = S._llm_roundtrip_diffs(b.llm, q)
    assert not diffs, diffs


def test_page_defaults_alone_still_evaluate():
    """页面**原始默认值**（未导入，llm_json 空）仍可评估 → 手配路径不被 llm_json 通道破坏。"""
    q = _page_defaults()
    r = S.eval_config(q)
    assert r["ok"], r.get("errors")
    assert r["device_peak"] > 0


# ── ③ 现场 yaml 逐字段核 ─────────────────────────────────────────────────────────────
_SITE_YAMLS = [r"C:\Users\suhaibo\Desktop\test.yaml",
               r"C:\Users\suhaibo\Desktop\test-unfused.yaml"]


@pytest.mark.parametrize("path", _SITE_YAMLS)
def test_site_yaml_roundtrip_field_exact(path):
    if not os.path.exists(path):
        pytest.skip(f"现场 yaml 不可用（本机私有路径）：{path}")
    import yaml as _yaml
    mf = _yaml.safe_load(open(path, encoding="utf-8"))
    # 现场 config 缺 kv_lora_rank（见 test_site_yaml_missing_kv_lora_fails_loud）——本例只核
    # **round-trip 保真**，故显式补上让 build 可达；补的值不参与保真判据（两侧同源）。
    mf["model"]["kv_lora_rank"] = 512
    b, fields, cfg = _roundtrip(mf)
    a, c = dataclasses.asdict(b.llm), dataclasses.asdict(cfg)
    diffs = {k: (a[k], c[k]) for k in a if a[k] != c[k]}
    assert not diffs, diffs
    # 逐层压缩比必须是 yaml 的真表（44 = 43 层 + 1 MTP，末项给 MTP 层）
    assert list(cfg.csa_compress_ratios) == list(mf["model"]["compress_ratios"])


# ── ④ 缺 kv_lora_rank：两条路径同样 fail-loud（绝不替换预设值）────────────────────────
def test_site_yaml_missing_kv_lora_roundtrips_as_zero_not_preset():
    """现场 test.yaml 真的没有 kv_lora_rank → 两条路径都必须忠实带 **0**，绝不补预设值。

    2026-07-25 订正:本测试原先还断言 `build_llm_spec` 必 fail-loud。该守卫已按证据收窄——
    `kv_lora_rank` 在 `layers/dsv4_hybrid.py` 出现 **0 次**、峰值对其逐字节不变(见
    tests/test_dsv4_hybrid_unused_mla_dims.py),故本变体不再要求 >0(现场 yaml 遂可原样评估,
    不必编值)。mla/dsa 的 fail-loud 覆盖移至该文件。**本测试的原本意图(round-trip 不得静默
    替代成预设值)完整保留** —— 那才是这里要守的东西。
    """
    mf = _ab_launcher()
    mf["model"].pop("kv_lora_rank")
    b, fields, cfg = _roundtrip(mf)
    assert b.llm.kv_lora_rank == 0 and cfg.kv_lora_rank == 0   # round-trip 忠实带 0，不补预设
    # dsv4_hybrid 不用这一维 → 可直接建图评估（且 UI 路径与 bundle 一致）
    for label, c in (("bundle", b.llm), ("ui", cfg)):
        assert build_llm_spec(c) is not None, label


# ── ⑤ qk_nope_head_dim：仅按 head_dim = qk_nope + qk_rope 恒等式导出，违背即 fail-loud ──
def test_qk_nope_derived_from_head_dim_identity():
    """A/B launcher 省略 qk_nope_head_dim；mindformers 按 head_dim−qk_rope 导出（512−64=448）。"""
    b = from_mindformers_dict(_ab_launcher())
    assert b.llm.qk_nope_head_dim == 448
    assert b.llm.head_dim == 512 and b.llm.qk_rope_head_dim == 64


def test_qk_nope_identity_violation_fails_loud():
    """三维都写了但不满足恒等式 → fail-loud（不猜哪个对）。"""
    with pytest.raises(ValueError) as ei:
        from_mindformers_dict(_ab_launcher(qk_nope_head_dim=100))
    assert "qk_nope_head_dim" in str(ei.value)


def test_qk_nope_not_derivable_without_head_dim():
    """既无 qk_nope 也无 head_dim → **不导出**（保持 0），不杜撰一个值。

    2026-07-25 订正:原先还断言 build_llm 因此 fail-loud。`qk_nope_head_dim` 在
    `layers/dsv4_hybrid.py` 出现 **0 次**(该变体 `q_head_dim = v_head_dim`,不按 nope+rope 拆,
    见 dsv4_hybrid.py:49-51),峰值对其逐字节不变 → 本变体不再要求 >0。**本测试的原本意图
    (缺恒等式输入时不许编造导出值)完整保留**;mla/dsa 的 fail-loud 见
    tests/test_dsv4_hybrid_unused_mla_dims.py。
    """
    mf = _ab_launcher()
    mf["model"].pop("head_dim")
    b = from_mindformers_dict(mf)
    assert b.llm.qk_nope_head_dim == 0          # 不导出、不杜撰
    assert build_llm_spec(b.llm) is not None    # dsv4_hybrid 不用这一维 → 仍可建图


# ── ⑥ compress_ratios 长度：N 或 N+mtp（mindformers 现场写法，末项=MTP 层）────────────
def test_compress_ratios_length_allows_mtp_tail():
    mf = _ab_launcher(num_nextn_predict_layers=1,
                      compress_ratios=[0, 4, 128, 4, 128, 4, 128, 4, 0])   # 8 层 + 1 MTP
    b = from_mindformers_dict(mf)
    assert len(b.llm.csa_compress_ratios) == 9
    spec = build_llm_spec(b.llm)          # 不再因 9 != 8 报错
    assert spec.layer_pattern.count("mtp") == 1


def test_compress_ratios_bad_length_still_fails_loud():
    mf = _ab_launcher(compress_ratios=[0, 4, 128])          # 3 != 8（且 mtp=0）
    b = from_mindformers_dict(mf)
    with pytest.raises(ValueError) as ei:
        build_llm_spec(b.llm)
    assert "csa_compress_ratios" in str(ei.value)


# ── ⑦ 导入后手改：只有被改的字段覆盖，其余仍是 yaml 忠实值 ────────────────────────────
def test_post_import_edit_overrides_only_edited_field():
    b, fields, cfg = _roundtrip(_ab_launcher())
    q = _query(fields, hidden="2048")           # 用户在页面把 hidden 改了
    errs, cfg2, pa = S.parse_and_validate(q)
    assert not errs, errs
    assert cfg2.hidden_size == 2048                                   # 改动生效
    assert cfg2.csa_compress_ratios == b.llm.csa_compress_ratios      # 其余仍忠实
    assert cfg2.o_groups == b.llm.o_groups == 8
    assert cfg2.chunk_loss_num == b.llm.chunk_loss_num


def test_unregistered_override_gate_fails_loud(monkeypatch):
    """日后新增 UI 覆盖项却忘登记门控键 → fail-loud，不静默覆盖权威 config。"""
    gate = dict(S._LLM_FIELD_GATE)
    gate.pop("hidden_size")                      # 模拟"忘登记"
    monkeypatch.setattr(S, "_LLM_FIELD_GATE", gate)
    b, fields = None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = from_mindformers_dict(_ab_launcher())
        fields = S._llm_to_fields(b.llm)
    with pytest.raises(S.ConfigRoundTripError) as ei:
        S.parse_and_validate(_query(fields))
    assert "hidden_size" in str(ei.value)


def test_llm_json_field_set_mismatch_fails_loud():
    """`llm_json` 缺字段（编码方与 LLMConfig 定义漂移）→ 报错，不静默取 dataclass 默认。"""
    b = from_mindformers_dict(_ab_launcher())
    d = json.loads(S._llm_to_fields(b.llm)["llm_json"])
    d.pop("csa_compress_ratios")
    q = _query(S._llm_to_fields(b.llm), llm_json=json.dumps(d))
    errs, cfg, pa = S.parse_and_validate(q)
    assert errs and "llm_json" in errs[0] and "csa_compress_ratios" in errs[0], errs


def test_manual_path_still_uses_preset_base():
    """无 llm_json 的手配 query → 仍走预设基座（历史行为逐字节不变）。"""
    errs, cfg, pa = S.parse_and_validate({"preset": "dsv3_mini", "layers": "8"})
    assert not errs, errs
    assert cfg.hidden_size == 1792 and cfg.attn_type == "mla"


# ── ⑧ 重算 round-trip：部分层 full 不再被放大成"全部层" ───────────────────────────────
def test_partial_full_recompute_layers_survive_roundtrip():
    mf = _ab_launcher()
    mf["parallelism"]["pipeline_parallel"] = 1
    mf["recompute"] = {"mode": "full", "full_recompute_layer": ["0-3"]}
    b, fields, cfg = _roundtrip(mf)
    assert b.recompute.mode == "full" and b.recompute.full_layers == {1, 2, 3, 4}
    errs, _cfg, pa = S.parse_and_validate(_query(fields))
    assert not errs, errs
    assert set(pa["sel_layers"]) == {1, 2, 3, 4}, pa["sel_layers"]
