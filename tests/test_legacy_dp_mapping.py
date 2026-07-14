"""P0.3（2026-07-14 review）：老式 `parallel_config.data_parallel` 的映射须按
`parallel.enable_parallel_optimizer`（epo，mindformers/mindspore 默认 **False**）分流：

- epo=True  → 权重/优化器按 dp 切分（zero/FSDP 类）→ `data_parallel_shard`；
- epo=False → 纯数据并行,权重逐 dp rank **复制** → `data_parallel_replicate`
  （评估器已正确建模：持久态只 ÷fsdp_degree=dp_shard·cp，绝不 ÷dp_replicate，static_mem.py:31）。

此前无条件映射 dp_shard 会把 epo=False + dp>1 的持久内存**静默低估 ~dp 倍**。
"""
from cost_eval.configs.from_mindformers import from_mindformers_dict
from serve_explorer import _mf_adapt


def _legacy_mf(dp=4, epo=None):
    """最小老式 mindformers yaml dict（parallel_config + model.model_config 嵌套）。"""
    mf = {
        "runner_config": {"batch_size": 1},
        "parallel_config": {"data_parallel": dp, "model_parallel": 1, "pipeline_stage": 1},
        "model": {"model_config": {
            "num_hidden_layers": 2, "hidden_size": 256, "num_attention_heads": 8,
            "vocab_size": 1000, "seq_length": 512, "intermediate_size": 512,
            "compute_dtype": "bfloat16", "params_dtype": "float32",
        }},
    }
    if epo is not None:
        mf["parallel"] = {"parallel_mode": 1, "enable_parallel_optimizer": epo}
    return mf


def test_epo_true_maps_dp_to_shard():
    mf, _ = _mf_adapt(_legacy_mf(dp=4, epo=True))
    par = mf["parallelism"]
    assert par["data_parallel_shard"] == 4
    assert par.get("data_parallel_replicate", 1) == 1


def test_epo_false_maps_dp_to_replicate():
    mf, _ = _mf_adapt(_legacy_mf(dp=4, epo=False))
    par = mf["parallelism"]
    assert par["data_parallel_shard"] == 1
    assert par["data_parallel_replicate"] == 4


def test_epo_absent_defaults_false_maps_to_replicate():
    """`parallel` 段缺省 → epo 取 mindformers/mindspore 默认 False → 纯数据并行。"""
    mf, _ = _mf_adapt(_legacy_mf(dp=4, epo=None))
    par = mf["parallelism"]
    assert par["data_parallel_shard"] == 1
    assert par["data_parallel_replicate"] == 4


def test_dp1_mapping_invariant_regardless_of_epo():
    for epo in (True, False, None):
        mf, _ = _mf_adapt(_legacy_mf(dp=1, epo=epo))
        par = mf["parallelism"]
        assert par["data_parallel_shard"] == 1
        assert par.get("data_parallel_replicate", 1) == 1


def test_end_to_end_legacy_epo_false_gives_dp_replicate():
    bundle = from_mindformers_dict(_mf_adapt(_legacy_mf(dp=4, epo=False))[0])
    assert bundle.parallel.dp_replicate == 4
    assert bundle.parallel.dp_shard == 1


def test_end_to_end_legacy_epo_true_gives_dp_shard():
    """锁定 671b 场景（pretrain_deepseek3_671b: dp=4 + epo=True → dp_shard=4 切分）。"""
    bundle = from_mindformers_dict(_mf_adapt(_legacy_mf(dp=4, epo=True))[0])
    assert bundle.parallel.dp_shard == 4
    assert bundle.parallel.dp_replicate == 1


def test_new_style_yaml_untouched_by_adapt():
    """新式 yaml（已有 parallelism 段）不经 legacy 分流——原样通过。"""
    mf = {"parallelism": {"data_parallel_shard": 2, "tensor_parallel": 1},
          "training": {"local_batch_size": 1},
          "model": _legacy_mf()["model"]["model_config"]}
    out, _ = _mf_adapt(mf)
    assert out["parallelism"]["data_parallel_shard"] == 2
    assert "data_parallel_replicate" not in out["parallelism"]
