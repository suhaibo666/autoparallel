"""Task 3 [MODULARITY] — 每结构内存 rollup（StructureMemory）。

每个基本结构（attention / ffn / moe / …）有一份可组合的内存估计
`estimate_structure_memory(resolved_ops) -> StructureMemory`，对 params 与 saves
**按名去重**，是全库唯一的去重点（static_mem / mem_timeline 均组装它）。
"""
from cost_eval.model_spec import DimTable, ModelSpec, LayerSpec
from cost_eval.layers.attention import build_gqa_attn_ops
from cost_eval.layers.ffn import build_dense_ffn_ops, build_moe_ffn_ops
from cost_eval.specs import ParallelConfig
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.structure_mem import estimate_structure_memory, StructureMemory

D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100,
             n_layers=1, n_experts=8, topk=2, moe_F=128)


def _resolve(op_list, pc=None):
    """把裸 op 列表解析成 ResolvedOp 列表（单卡默认）。"""
    pc = pc or ParallelConfig()
    spec = ModelSpec("t", D, ["x"], {"x": LayerSpec(list(op_list))})
    pm = ParallelModel(pc, n_layers=1, world_size=1)
    return ShapeEval().resolve(spec, pm).stages[0][0].ops, pm


def test_per_structure_breakdown_sensible():
    """attn / ffn / moe 各自 rollup：persistent 加性、saves 去重、workspace=max。"""
    attn, _ = _resolve(build_gqa_attn_ops(D))
    sa = estimate_structure_memory(attn)
    assert isinstance(sa, StructureMemory)
    assert sa.param_full_bytes > 0            # qkv_w + o_w
    assert sa.activation_saves > 0
    assert sa.workspace > 0                   # flash workspace
    # attn 的 saves 去重：flash+o_proj 双 save 的 attn 只算一次
    naive = sum(s.local_numel * s.dtype_bytes for op in attn for s in op.saves)
    attn_t = next(s for op in attn for s in op.saves if s.name == "attn")
    assert sa.activation_saves == naive - attn_t.local_numel * attn_t.dtype_bytes


def test_moe_expert_params_flagged():
    """MoE 专家权重 is_expert=True → param_full_bytes 含专家权重。"""
    moe, _ = _resolve(build_moe_ffn_ops(D))
    sm = estimate_structure_memory(moe)
    assert sm.param_full_bytes > 0
    # 专家 rollup 的 persistent 用 efsdp 分母（单卡=1，退化为 numel*opt_bytes）
    sm2 = estimate_structure_memory(moe, fsdp=1, efsdp=1, opt_state_bytes=14)
    assert sm2.persistent > 0


def test_compose_equals_whole_layer():
    """attn ⊕ ffn 的逐桶和 == 整层（attn+ffn 拼接后）的 rollup（模块化组装自洽）。"""
    attn_ops = build_gqa_attn_ops(D)
    ffn_ops = build_dense_ffn_ops(D)
    attn, pm = _resolve(attn_ops)
    ffn, _ = _resolve(ffn_ops)
    whole, _ = _resolve(list(attn_ops) + list(ffn_ops))

    sa = estimate_structure_memory(attn, fsdp=2, efsdp=2, opt_state_bytes=14)
    sf = estimate_structure_memory(ffn, fsdp=2, efsdp=2, opt_state_bytes=14)
    sw = estimate_structure_memory(whole, fsdp=2, efsdp=2, opt_state_bytes=14)

    assert sa.param_full_bytes + sf.param_full_bytes == sw.param_full_bytes
    assert sa.grad_full_bytes + sf.grad_full_bytes == sw.grad_full_bytes
    assert sa.activation_saves + sf.activation_saves == sw.activation_saves
    assert sa.persistent + sf.persistent == sw.persistent
    assert sa.bwd_scratch + sf.bwd_scratch == sw.bwd_scratch
    assert max(sa.workspace, sf.workspace) == sw.workspace
