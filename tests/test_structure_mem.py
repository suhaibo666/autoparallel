"""Task 3 [MODULARITY] — 每结构内存 rollup（StructureMemory）。

每个基本结构（attention / ffn / moe / …）有一份可组合的内存估计
`estimate_structure_memory(resolved_ops) -> StructureMemory`，对 params 与 saves
**按名去重**，是全库唯一的去重点（static_mem / mem_timeline 均组装它）。
"""
from cost_eval.model_spec import DimTable, ModelSpec, LayerSpec, OpSpec, OpType, TensorRef
from cost_eval.layers.attention import build_gqa_attn_ops
from cost_eval.layers.ffn import build_dense_ffn_ops, build_moe_ffn_ops
from cost_eval.layers.mla import build_mla_dense_decoder
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


def _ref_forward_max_live(ops):
    """独立参照：mini forward 时间线求 max-live（activation-only，去重按名，含 op workspace）。

    与被测实现分开写，守卫 estimate_structure_memory 的 forward_max_live 在真实 op 上正确。
    活性区间 = [首次出现 op, 末次出现 op]（作为 input 或 output）；权重（is_weight）不计。
    """
    byt, first, last = {}, {}, {}
    for i, op in enumerate(ops):
        for t in [t for t in op.inputs if not t.is_weight] + [op.output]:
            byt.setdefault(t.name, t.local_numel * t.dtype_bytes)
            first.setdefault(t.name, i)
            last[t.name] = i
    return max(
        sum(b for n, b in byt.items() if first[n] <= i <= last[n]) + op.workspace_bytes
        for i, op in enumerate(ops)
    )


def test_forward_max_live_exact_tiny_layer():
    """3-op 玩具层已知活性模式 → forward_max_live 精确值（§8.5②）。

    op p: [X(act), W(weight)] → A     （X 死于 p；W 是权重，**不计入**激活工作集）
    op q: [A]                 → Bt    （A 跨 p..r 存活）
    op r: [A, Bt, W]          → C     （A/Bt/C 三者共存，+ workspace）
    活字节（dtype=4，无对齐取整）：X=512, A=Bt=C=32768。
      p: X+A               = 33280
      q: A+Bt              = 65536
      r: A+Bt+C + ws(1000) = 99304  ← 峰
    权重 W 若被误计（16384）峰会变 115688 → 精确 99304 守卫「权重不计入」。
    """
    X = TensorRef("X", ("S", "B"), dtype_bytes=4)
    W = TensorRef("W", ("H", "H"), is_weight=True, dtype_bytes=4)
    A = TensorRef("A", ("S", "B", "H"), dtype_bytes=4)
    Bt = TensorRef("Bt", ("S", "B", "H"), dtype_bytes=4)
    C = TensorRef("C", ("S", "B", "H"), dtype_bytes=4)
    ops, _ = _resolve([
        OpSpec("p", OpType.MATMUL, [X, W], A, params=[W], saves=[X]),
        OpSpec("q", OpType.MATMUL, [A], Bt, saves=[A]),
        OpSpec("r", OpType.ELEMENTWISE, [A, Bt, W], C, params=[W], workspace="1000"),
    ])
    sm = estimate_structure_memory(ops)
    assert sm.forward_max_live == 99304
    # 峰 ≥ 任一单点活集（tautology 但守回归）
    assert sm.forward_max_live >= 33280 and sm.forward_max_live >= 65536


def test_forward_max_live_is_true_peak_on_mla_layer():
    """真实 MLA dense 层：forward_max_live == 独立参照 mini-walk，且 ≥ checkpoint_input。

    并记录关键事实：**forward_max_live < activation_saves**（峰值是单时刻共存，saves 是整层
    去重之和；故 `fml − saves` 恒为 0 —— 这正是 bwd_working_set 不能用 `fml−saves` 的原因）。
    """
    Dmla = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100,
                    n_layers=1, q_lora_rank=32, kv_lora_rank=16,
                    qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16)
    ops, _ = _resolve(build_mla_dense_decoder(Dmla).ops)
    sm = estimate_structure_memory(ops)
    assert sm.forward_max_live == _ref_forward_max_live(ops)
    assert sm.forward_max_live > 0
    assert sm.forward_max_live >= sm.checkpoint_input
    assert sm.forward_max_live < sm.activation_saves      # 峰 < Σ去重saves（本层实测）


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
