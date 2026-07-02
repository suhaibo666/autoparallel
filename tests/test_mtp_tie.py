"""Task 2 [C2] — MTP ties embedding + output head (no phantom vocab×H params).

DeepSeek-V4 MTP shares (ties) the main model's embedding and output layer
(`mindformers/pynative/transformers/multi_token_prediction.py`: `construct` takes
`embedding`/`output_layer`/`output_weight` from the main model, :336-338/:379/:393).
So `build_mtp_ops` must NOT re-declare `emb_w (vocab,H)` / `head_w (H,vocab)` as params
— otherwise `static_mem` (which sums params per-op-occurrence) counts ~2·vocab·H phantom
params for the MTP layer (the measured DSv4 over-prediction).
"""
import dataclasses

from cost_eval.presets import deepseek_v4
from cost_eval.build_llm import build_llm_spec
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import ParallelConfig


def _global_param_count(spec) -> int:
    """全 1 并行下解析，sum 所有 op 的 param 张量 local_numel（== 全局 numel）。"""
    pc = ParallelConfig(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1,
                        sequence_parallel=False)
    pm = ParallelModel(pc, spec.dims.n_layers, world_size=1)
    g = ShapeEval().resolve(spec, pm)
    return sum(
        w.local_numel
        for layers in g.stages.values()
        for layer in layers
        for op in layer.ops
        for w in op.params
    )


def test_mtp_layer_params_exclude_vocab_weights():
    """MTP 层的 op params 里不得含 emb_w / head_w（tie 主模型，不重复建权重）。"""
    spec = build_llm_spec(deepseek_v4(4))
    assert "mtp" in spec.layer_specs
    param_names = {w.name for op in spec.layer_specs["mtp"].ops for w in op.params}
    assert "emb_w" not in param_names
    assert "head_w" not in param_names


def test_mtp_param_delta_excludes_phantom_vocab_head():
    """启用 MTP 的全局 param 增量 ≈ 一个 decoder 层 + eh_proj，
    绝不含 2·vocab·H 幻影参数（C2）。"""
    cfg = deepseek_v4(4)
    with_mtp = build_llm_spec(cfg)
    without_mtp = build_llm_spec(dataclasses.replace(cfg, mtp_num_layers=0))

    n_with = _global_param_count(with_mtp)
    n_without = _global_param_count(without_mtp)
    delta = n_with - n_without

    vocab_h = cfg.vocab_size * cfg.hidden_size
    # 幻影上界：untied 会多算 emb_w + head_w = 2·vocab·H
    assert delta > 0
    assert delta < 2 * vocab_h, (
        f"MTP 增量 {delta:,} 含 ~2·vocab·H={2*vocab_h:,} 幻影参数（未 tie）")

    # 自洽：增量应恰等于 MTP 层实际 op params 之和（tie 后 = eh_w + decoder 层权重）。
    pc = ParallelConfig(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1,
                        sequence_parallel=False)
    pm = ParallelModel(pc, with_mtp.dims.n_layers, world_size=1)
    g = ShapeEval().resolve(with_mtp, pm)
    mtp_layer = next(
        l for layers in g.stages.values() for l in layers if l.layer_type == "mtp")
    mtp_param_numel = sum(w.local_numel for op in mtp_layer.ops for w in op.params)
    assert delta == mtp_param_numel
