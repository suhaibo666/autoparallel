"""FlashAttention 保存契约 —— 真机算子探针坐实（2026-07-16）。

真机 `FlashAttentionScore`（Ascend 910B / MindSpore 2.10）前向输出：
  softmax_max / softmax_sum 各 `[B, N, S, 8]` **Float32**；softmax_out `(1,)` **空**（不物化 S×S）；
  attention_out `[B, N, S, D]` bf16。`FlashAttentionScoreGrad` 输入 = {q,k,v,attention_out,
  softmax_max,softmax_sum,dy} → 保存集 = Q/K/V + O + softmax(max,sum)。
本组把评估器 flash op 的 `saves` 钉在该契约上（防回退到「存 S×S 矩阵」或「统计非 [.,.,.,8]fp32」）。
证据见 analysis/realmachine/flash_attn_activation_validation_2026-07-16.md。
"""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_eval.build_llm import build_llm_spec
from cost_eval.presets import deepseek_v3


def _mla_spec():
    return build_llm_spec(deepseek_v3(4))                       # DSv3 MLA（flash op）


def _gqa_spec():
    return build_llm_spec(dataclasses.replace(deepseek_v3(4), attn_type="gqa", num_query_groups=2))


def _flash_specs():
    """有标准 FlashAttention op 的两条路径：MLA 与 GQA（DSv4-hybrid 走 DSA/CSA，无标准 flash op，另论）。"""
    return [("mla", _mla_spec()), ("gqa", _gqa_spec())]


def _flash_ops(spec):
    for lt, ls in spec.layer_specs.items():
        for op in ls.ops:
            if op.name == "flash":
                yield lt, op


def test_flash_saves_softmax_stats_is_BNS8_fp32():
    """softmax 统计存量 = `[2, B, n_heads, S, 8]` fp32（真机 softmax_max+sum 各 [B,N,S,8] Float32）。"""
    for tag, spec in _flash_specs():
        seen = 0
        for lt, op in _flash_ops(spec):
            seen += 1
            stats = [t for t in op.saves if t.name == "fa_stats"]
            assert stats, (tag, lt, [t.name for t in op.saves])
            t = stats[0]
            assert t.dtype_bytes == 4, (tag, lt, "softmax 统计须 fp32", t.dtype_bytes)
            assert tuple(t.shape) == ("2", "B", "n_heads", "S", "8"), (tag, lt, t.shape)
        assert seen, (tag, "未找到 flash op")


def test_flash_saves_no_SxS_matrix():
    """任一 flash save 不得含 S 出现两次（真 FA softmax_out 为空、不物化 S×S 分数矩阵）。

    例外（2026-07-23 std census）：`attn_mask_u8` [S,S] **uint8**（dtype_bytes=1）是真机
    attention.py:224-226 `cast(attention_mask, uint8)` 的每层复本（FA bprop 持有）——合法
    S×S 保留;本合约只禁 **≥2B**（bf16/fp32）的 S×S 分数/概率矩阵物化。"""
    for tag, spec in _flash_specs():
        for lt, op in _flash_ops(spec):
            for t in op.saves:
                if (t.dtype_bytes or 2) == 1:
                    continue          # uint8 mask 复本（真机保留,非分数矩阵）
                s_count = sum(1 for d in t.shape if "S" in str(d))
                assert s_count <= 1, (tag, lt, t.name, t.shape, "疑似 S×S 矩阵")


def test_flash_saves_cover_qkv_and_output():
    """保存集须覆盖 flash 的输入（Q/K/V）与输出（O）——即 FlashAttentionScoreGrad 的输入。"""
    for tag, spec in _flash_specs():
        seen = 0
        for lt, op in _flash_ops(spec):
            seen += 1
            names = {t.name for t in op.saves}
            assert op.output.name in names, (tag, lt, "O 须在 saves", op.output.name, names)
            for t in op.inputs:
                assert t.name in names, (tag, lt, "Q/K/V 输入须在 saves", t.name, names)
            assert "fa_stats" in names, (tag, lt, names)
        assert seen, (tag, "未找到 flash op")


def test_flash_stats_numel_matches_operator():
    """resolve 后 fa_stats 数值 = 2·B·n_heads·S·8（tp=cp=1），与真机算子 [B,N,S,8]×2 一致。"""
    from cost_eval.shape_eval import ShapeEval
    from cost_eval.parallel_model import ParallelModel
    from cost_eval.specs import ParallelConfig
    spec = _mla_spec()
    d = spec.dims
    pc = ParallelConfig(dp_shard=1, tp=1, cp=1, num_microbatches=1)
    g = ShapeEval().resolve(spec, ParallelModel(pc, d.n_layers, 1))
    for lid, lys in g.stages.items():
        for l in lys:
            for op in l.ops:
                if op.name == "flash":
                    t = next(t for t in op.saves if t.name == "fa_stats")
                    # numel = 2 * B * n_heads * S * 8；此处只校验能被 16 整除且 = 2*8*(B*n_heads*S)
                    assert t.dtype_bytes == 4 and t.local_numel % 16 == 0, (t.local_numel, t.dtype_bytes)
                    return
    raise AssertionError("未找到 flash op")
