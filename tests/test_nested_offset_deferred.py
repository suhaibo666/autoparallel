"""P1-18（final review 2026-07-14）：老式 yaml 的**嵌套 offset**（VPP per-chunk）换算须在
必需模型字段兜底**之后**进行。

修前 `_mf_adapt` 在 adapt 时刻读 `num_hidden_layers`（yaml 依赖 mindformers 类内默认时缺失→0）
算 `num_layer_list`，得到 sum(offset)≠N 的恒错列表 → `finetune_deepseek3_671b.yaml` 这类配置
永远导不进（fail-loud 兜住但根因是换算时机错）。修后：N 缺失时暂存 `_nested_offset`，由
`_materialize_nested_offset` 在页面兜底注入 N 后再换算。
"""
from serve_explorer import _mf_adapt, _materialize_nested_offset


def _legacy_nested(with_layers: bool):
    mc = {"hidden_size": 256, "num_attention_heads": 8, "vocab_size": 1000,
          "seq_length": 512, "intermediate_size": 512,
          "offset": [[1, 0], [0, -1]], "pp_interleave_num": 2}
    if with_layers:
        mc["num_hidden_layers"] = 8
    return {
        "runner_config": {"batch_size": 1},
        "parallel_config": {"data_parallel": 1, "model_parallel": 1, "pipeline_stage": 2},
        "model": {"model_config": mc},
    }


def test_nested_offset_with_layers_materializes_immediately():
    """N 在场：行为与修前一致——立即换算。N=8,pp=2,v=2,base=2 → [ (2+1)+(2+0), (2+0)+(2-1) ]=[5,3]。"""
    mf, vpp = _mf_adapt(_legacy_nested(with_layers=True))
    assert vpp == 2
    assert mf["parallelism"]["num_layer_list"] == [5, 3]
    assert "_nested_offset" not in mf["parallelism"]


def test_nested_offset_without_layers_is_deferred_not_wrong():
    """N 缺失：不得用 N=0 算出错列表；暂存 `_nested_offset` 等兜底。"""
    mf, _ = _mf_adapt(_legacy_nested(with_layers=False))
    par = mf["parallelism"]
    assert "num_layer_list" not in par            # 修前这里会是 [1, -1](N=0 恒错)
    assert par["_nested_offset"] == [[1, 0], [0, -1]]


def test_materialize_after_backfill_yields_correct_layer_list():
    """页面兜底注入 num_hidden_layers 后物化 → 与 N 在场路径同结果。"""
    mf, _ = _mf_adapt(_legacy_nested(with_layers=False))
    mf["model"]["num_hidden_layers"] = 8          # 模拟 do_POST 的页面值兜底
    warnings = []
    _materialize_nested_offset(mf, warnings)
    assert mf["parallelism"]["num_layer_list"] == [5, 3]
    assert "_nested_offset" not in mf["parallelism"]
    assert warnings == []


def test_materialize_without_layers_drops_with_warning():
    """兜底仍无 N → 丢弃并警告，绝不静默错算。"""
    mf, _ = _mf_adapt(_legacy_nested(with_layers=False))
    warnings = []
    _materialize_nested_offset(mf, warnings)
    assert "num_layer_list" not in mf["parallelism"]
    assert "_nested_offset" not in mf["parallelism"]
    assert warnings and "offset" in warnings[0]
