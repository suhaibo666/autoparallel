"""Review-2 F7：qk_layernorm 直连 API 与 YAML 适配器不再 split-brain。

`build_llm.py` 早已为 gqa/mha `qk_layernorm=True` 建 `q_norm`/`k_norm`（X3）；此前 YAML 适配器
`from_mindformers.py:375-380` 却对 gqa/mha fail-loud → 同一已建能力经 YAML 路径**不可达**（探针
`...probe_2026-07-16.py::qk_norm_adapter_split_brain`：direct API 有 q/k_norm，yaml_adapter 却
NotImplementedError）。

修复：适配器移除该 fail-loud，并**仅对 gqa/mha** 透传 `qk_layernorm=True`；mla/dsv4/dsa 仍
subsumed（透传 False、不 raise，DSv4align round-trip 锚点不破）。
"""
from cost_eval.build_llm import build_llm_spec
from cost_eval.configs.from_mindformers import from_mindformers_dict


def _mf(model_extra):
    model = {"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 1024,
             "vocab_size": 1000, "seq_length": 512, "compute_dtype": "bfloat16"}
    model.update(model_extra)
    return {"model": model, "training": {"local_batch_size": 1}}


def _op_names(spec):
    return {o.name for ls in spec.layer_specs.values() for o in ls.ops}


def test_gqa_qk_layernorm_adapter_accepts_and_builds_norm_ops():
    bundle = from_mindformers_dict(_mf({"num_key_value_heads": 4, "qk_layernorm": True}))
    assert bundle.llm.attn_type == "gqa"
    assert bundle.llm.qk_layernorm is True                 # gqa → 透传
    names = _op_names(build_llm_spec(bundle.llm))
    assert "q_norm" in names and "k_norm" in names


def test_mha_qk_layernorm_adapter_accepts_and_builds_norm_ops():
    bundle = from_mindformers_dict(_mf({"qk_layernorm": True}))   # 无 kv_heads → mha
    assert bundle.llm.attn_type == "mha"
    assert bundle.llm.qk_layernorm is True
    names = _op_names(build_llm_spec(bundle.llm))
    assert "q_norm" in names and "k_norm" in names


def test_mla_qk_layernorm_still_subsumed_no_extra_norm():
    bundle = from_mindformers_dict(_mf({
        "multi_latent_attention": True, "q_lora_rank": 512, "kv_lora_rank": 256,
        "qk_rope_head_dim": 64, "qk_nope_head_dim": 64, "v_head_dim": 128,
        "qk_layernorm": True}))
    assert bundle.llm.attn_type == "mla"
    assert bundle.llm.qk_layernorm is False                # subsumed → 不透传
    names = _op_names(build_llm_spec(bundle.llm))
    assert "q_norm" not in names and "k_norm" not in names
