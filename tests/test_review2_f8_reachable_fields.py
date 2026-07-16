"""Review-2 F8：两个已建模特性经**公开构造字段**可达（此前只能动态挂属性 → TypeError）。

探针（`...probe_2026-07-16.py::modeled_features_not_publicly_reachable`）实证：
- `LLMConfig(cp_kv_allgather_buffer=True)` → TypeError（消费点 shape_eval.py:265 靠
  `getattr(spec.dims, "cp_kv_allgather_buffer", False)`，但从不是真字段）。
- `ParallelConfig(pipeline_parallel_overlap_p2p=True)` → TypeError（消费点 mem_timeline.py:527
  靠 `getattr(pm.pc, "pipeline_parallel_overlap_p2p", False)`，同样非真字段）。

修复：二者成为真 dataclass 字段，默认 False；`cp_kv_allgather_buffer` 经 `to_dimtable` 直通到
`DimTable`。默认 False → 逐字节等此前 getattr 兜底行为（锚点不动）。
"""
from cost_eval.llm_config import LLMConfig, to_dimtable
from cost_eval.specs import ParallelConfig


def _base_llm(**over):
    base = dict(num_layers=2, hidden_size=8, num_attention_heads=2, num_query_groups=2,
                vocab_size=16, seq_length=16, batch_size=1, head_dim=4, attn_type="gqa",
                ffn_hidden_size=16)
    base.update(over)
    return LLMConfig(**base)


def test_llmconfig_accepts_cp_kv_allgather_buffer_field():
    cfg = _base_llm(cp_kv_allgather_buffer=True)   # 无 TypeError
    assert cfg.cp_kv_allgather_buffer is True


def test_cp_kv_buffer_propagates_to_dimtable():
    assert to_dimtable(_base_llm(cp_kv_allgather_buffer=True)).cp_kv_allgather_buffer is True


def test_parallelconfig_accepts_overlap_p2p_field():
    pc = ParallelConfig(pp=2, pipeline_parallel_overlap_p2p=True)   # 无 TypeError
    assert pc.pipeline_parallel_overlap_p2p is True


def test_defaults_are_false_byte_identical():
    """默认关：DimTable/ParallelConfig 的这两个字段默认 False → 与此前 getattr 兜底逐字节等价。"""
    assert _base_llm().cp_kv_allgather_buffer is False
    assert to_dimtable(_base_llm()).cp_kv_allgather_buffer is False
    assert ParallelConfig().pipeline_parallel_overlap_p2p is False
