"""Task 2.4 — integrate dsv4_hybrid / mHC / MTP into build_llm_spec + deepseek_v4 preset (TDD).

Wires the frontier op-builders into the assembler (`build_llm.py`):
  1. dsv4_hybrid per-layer compress_ratio encoded in the layer key
     (`dsv4hyb_r{ratio}_{dense|moe}`) and dispatched to
     `build_dsv4_hybrid_attn_ops(dims, ratio)`.
  2. `residual_variant="mhc"` → each decoder body wrapped by `mhc_wrap`, with an
     expand op after embedding and a collapse op before lm_head (design §9).
  3. `mtp` layer registered via `build_mtp_ops(cfg)`.
  4. SWA (`window_size`/`window_pattern`) memory-neutral: does NOT change the op
     graph → identical peak (design §7.4).

Hard gate: the plain/DSv3 path stays byte-identical (see tests/test_regression_dsv3.py).
"""
import dataclasses

from cost_eval.presets import deepseek_v3, deepseek_v4
from cost_eval.build_llm import build_llm_spec, gen_layer_pattern
from cost_eval.model_spec import ModelSpec
from cost_eval.specs import (
    ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec)
from cost_eval.report import Evaluator
from validate_dsv3 import build_dsv3_spec, RESIDUAL_MiB

MiB, GiB = 2 ** 20, 2 ** 30


def _peak(spec, num_transformer_layers, ep=1):
    full = set(range(1, num_transformer_layers + 1))   # transformer layers only
    # 默认 depth=1 预取双缓冲 + 拆解后的 RESIDUAL_MiB(=177−ΔP)
    ev = Evaluator(
        spec,
        ParallelConfig(dp_shard=2, tp=1, ep=ep, pp=1, cp=1, sequence_parallel=True),
        OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
        HardwareSpec(max_device_memory=59 * GiB, framework_reserve=RESIDUAL_MiB * MiB),
        RecomputeSpec(mode="full", full_layers=full),
        SwapSpec(),
    )
    return ev.evaluate().per_stage[0]


# ---------------------------------------------------------------------------
# (1) deepseek_v4(4) builds + evaluates
# ---------------------------------------------------------------------------

def test_deepseek_v4_builds_modelspec_and_evaluates():
    cfg = deepseek_v4(4)
    spec = build_llm_spec(cfg)
    assert isinstance(spec, ModelSpec)
    p = _peak(spec, cfg.num_layers)
    assert p.peak_bytes > 0


# ---------------------------------------------------------------------------
# (2) layer_pattern encodes per-layer compress_ratio
# ---------------------------------------------------------------------------

def test_layer_pattern_encodes_compress_ratios():
    cfg = deepseek_v4(4)
    # gen_layer_pattern 返回 list[LayerContext]；其派生字符串标签仍编码 per-layer ratio。
    pattern = [c.name for c in gen_layer_pattern(cfg)]
    assert pattern[0] == "embedding"
    assert pattern[-1] == "lm_head"
    # each transformer layer key encodes its ratio from csa_compress_ratios
    ratios = cfg.csa_compress_ratios
    assert ratios is not None and len(ratios) == cfg.num_layers
    for i, ratio in enumerate(ratios):
        key = pattern[1 + i]
        assert key.startswith(f"dsv4hyb_r{ratio}_"), (i, ratio, key)
        assert key.endswith("_dense") or key.endswith("_moe")
    # mixed ratios exercise all three branches
    assert any("_r4_" in k for k in pattern)      # CSA (indexer)
    assert any("_r128_" in k for k in pattern)    # HCA
    assert any("_r0_" in k for k in pattern)      # sliding == MLA base


def test_dsv4_layer_key_dispatches_to_branch_ops():
    cfg = deepseek_v4(4)
    spec = build_llm_spec(cfg)
    # ratio-4 key -> indexer present (CSA); ratio-128 key -> compressor, no indexer.
    r4 = next(k for k in spec.layer_pattern if "_r4_" in k)
    r128 = next(k for k in spec.layer_pattern if "_r128_" in k)
    r4_names = [op.name for op in spec.layer_specs[r4].ops]
    r128_names = [op.name for op in spec.layer_specs[r128].ops]
    assert "indexer" in r4_names and "compressor" in r4_names and "sparse_attn" in r4_names
    assert "compressor" in r128_names and "indexer" not in r128_names


# ---------------------------------------------------------------------------
# (3) mHC wrapping applied
# ---------------------------------------------------------------------------

def test_mhc_wrapping_applied():
    cfg = deepseek_v4(4)
    spec = build_llm_spec(cfg)
    op_names = [op.name for ls in spec.layer_specs.values() for op in ls.ops]
    tensor_names = [
        t.name
        for ls in spec.layer_specs.values() for op in ls.ops
        for t in (list(op.inputs) + [op.output] + list(op.saves) + list(op.params))
    ]
    # HC modules (attn_hc / ffn_hc) present + h_res saved
    assert any(n.endswith("_hc_sinkhorn") for n in op_names)
    assert any(n.endswith("_hc_norm") for n in op_names)
    assert any(n.endswith("_h_res") for n in tensor_names)
    # residual carriers packed to n*H at the stack boundaries (expand / collapse)
    assert "hc_expand" in op_names and "hc_collapse" in op_names
    assert any(t == "num_residual_streams*H"
               for ls in spec.layer_specs.values() for op in ls.ops
               for tref in (list(op.inputs) + [op.output] + list(op.saves))
               for t in tref.shape)


def test_plain_path_has_no_mhc_ops():
    """deepseek_v3 (plain residual) must contain no HC/expand/collapse ops."""
    spec = build_llm_spec(deepseek_v3(4))
    op_names = [op.name for ls in spec.layer_specs.values() for op in ls.ops]
    assert not any("_hc_" in n or n in ("hc_expand", "hc_collapse") for n in op_names)


# ---------------------------------------------------------------------------
# (4) MTP layer present
# ---------------------------------------------------------------------------

def test_mtp_layer_present_and_registered():
    cfg = deepseek_v4(4)
    assert cfg.mtp_num_layers == 1
    spec = build_llm_spec(cfg)
    assert "mtp" in spec.layer_pattern
    assert "mtp" in spec.layer_specs
    mtp_names = [op.name for op in spec.layer_specs["mtp"].ops]
    assert "embedding" in mtp_names and "eh_proj" in mtp_names and "lm_head" in mtp_names


# ---------------------------------------------------------------------------
# (SWA) window_size is memory-neutral (design §7.4)
# ---------------------------------------------------------------------------

def test_swa_memory_neutral_equal_peak():
    base = deepseek_v3(4)                         # MLA, no window
    swa = dataclasses.replace(base, window_size=128,
                              window_pattern=(0, 1, 1, 1))
    p_base = _peak(build_llm_spec(base), base.num_layers)
    p_swa = _peak(build_llm_spec(swa), swa.num_layers)
    assert p_swa.peak_bytes == p_base.peak_bytes


# ---------------------------------------------------------------------------
# (DSv3 hard gate) plain path stays byte-identical to the oracle
# ---------------------------------------------------------------------------

def test_dsv3_preset_still_byte_identical():
    new = _peak(build_llm_spec(deepseek_v3(4)), 4)
    old = _peak(build_dsv3_spec(4)[0], 4)
    assert abs(new.peak_bytes - old.peak_bytes) < 1
    assert abs(new.peak_bytes / MiB - 12472.5) < 0.5
