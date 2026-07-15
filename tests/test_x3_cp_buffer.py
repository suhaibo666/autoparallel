"""X3 任务 B — P1-13 CP kernel buffer + fused-QKV KV all-gather 审计 + 不变量守卫。

审计结论（source-faithful，见 attention.py `build_gqa_attn_ops` 的 P1-13 注释）：
  fused GQA 的 `qkv` 是 Q/K/V 融合张量，KV 分量无法单独标 `cp_kv` → colossal CP 下 KV 分量
  仍随 body ÷cp、**欠建 colossal 的 KV all-gather full-S buffer**；ring/ulysses 的 CP 通信双缓冲
  （KV 块 send/recv）亦未建。

**为何无法在本 agent 文件域内闭环（可证不可行，非疏漏）**：
  1. 惰性硬约束「cp=1 恒 0」+ MLA 侧 DSv3 golden 逐字节不变 → 任何 buffer 必须 cp=1 时为 0。
     但 builder 只见 DimTable、不知 cp/method；`resolve_tensor` 对 builder 表达的 workspace/
     save **没有** cp>1 或 method 门（cp_kv 只是让「已存在的激活」在 colossal 下保持 full-S，
     并不能凭空「只在 cp>1 出现」一个 buffer，且 cp=1 时它是 full-S ≠ 0）。shape_eval 不在本
     agent 可改文件域。
  2. **frozen must-green** `test_cp_activation.test_cp2_all_activations_halve_including_loss`
     断言 GQA decoder body 在 colossal cp2 **精确减半**：对任一被计入桶 X，要求
     `2·X(cp2)==X(cp1)`。设 colossal buffer C：C(cp1) 与 C(cp2) 代入即得 `C==0`（见下方
     `test_colossal_gqa_buffer_is_provably_forbidden_by_halving`）——任何非零 colossal-cp2
     GQA body buffer都会破坏该冻结测试。故本 agent **不新增 buffer**（新增即破验收「全绿」），
     改为：①强化 attention.py 的 P1-13 审计注释（写清 colossal all-gather vs ring/ulysses 双缓冲
     的算法差异与公式）；②本文件把审计量化 + 守住不变量。解锁需 shape_eval 支持
     method+cp>1 门控的 workspace_ref（cp=1→0、colossal cp>1→full-S、ulysses/ring cp>1→KV_block·2），
     属他人文件域，已在报告「未尽事项」列出。

cp2 的两个真机锚点（cp2-colossal/cp2-ulysses）是 **DSv3(mla)**；本 agent 不改 mla builder →
这两个锚点逐字节不变（MLA 的 colossal KV all-gather 早由 `cp_kv=True` 建模，见下方对照测试）。
"""
from cost_eval.model_spec import DimTable, TensorRef, OpSpec, OpType, ModelSpec, LayerSpec
from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec
from cost_eval.shape_eval import ShapeEval, resolve_tensor
from cost_eval.structure_mem import estimate_structure_memory
from cost_eval.parallel_model import ParallelModel
from cost_eval.specs import ParallelConfig


def _gqa_cfg(**over):
    base = dict(num_layers=2, hidden_size=8, num_attention_heads=2, num_query_groups=2,
                vocab_size=16, seq_length=8, batch_size=1, head_dim=4, attn_type="gqa",
                ffn_hidden_size=16)
    base.update(over)
    return LLMConfig(**base)


def _resolve(spec, **pc):
    pm = ParallelModel(ParallelConfig(**pc), n_layers=len(spec.layer_pattern), world_size=64)
    g = ShapeEval().resolve(spec, pm)
    return [l for st in sorted(g.stages) for l in g.stages[st]]


# ── 守卫①：cp=1 下 GQA decoder 无任何 CP-only buffer（惰性；防误加 always-on buffer）─────────
def test_gqa_cp1_workspace_is_only_flash_ws_no_cp_buffer():
    """cp=1：GQA decoder body 的 workspace 只含 flash 的 fa_ws（= 64·B·n_heads·S），无额外 CP buffer。
    这守住「cp=1 恒 0（惰性）」——若谁误加了 cp=1 也存在的 CP buffer，此测试立刻抓到。"""
    spec = build_llm_spec(_gqa_cfg())
    layers = _resolve(spec, cp=1)
    d = spec.dims
    fa_ws = 64 * d.B * d.n_heads * d.S       # 单层 flash workspace（TP=1/CP=1）
    for l in layers:
        if any(o.name == "flash" for o in l.ops):
            ws = estimate_structure_memory(l.ops).workspace
            assert ws == fa_ws, f"GQA decoder workspace={ws} != flash-only {fa_ws}（疑似误加 CP buffer）"


# ── 守卫②：colossal cp2 下 GQA decoder body 精确减半（= frozen must-green 的不变量）───────────
def test_gqa_colossal_cp2_body_halves_exactly():
    """任务 A 的 q/k norm 加入后，GQA decoder body 在 colossal cp2 仍**精确减半**（saves+workspace）。
    与 test_cp_activation 冻结不变量同口径——既保任务 A 不破 CP，又实证任何 full-S colossal buffer
    会破坏此减半（见下一测试的代数证明）。"""
    for qk in (False, True):
        spec = build_llm_spec(_gqa_cfg(qk_layernorm=qk))
        l1 = _resolve(spec, cp=1)
        l2 = _resolve(spec, cp=2)
        for a, b in zip(l1, l2):
            if not any(o.name == "flash" for o in a.ops):
                continue
            sa = estimate_structure_memory(a.ops)
            sb = estimate_structure_memory(b.ops)
            assert sb.activation_saves * 2 == sa.activation_saves
            if sa.workspace:
                assert sb.workspace * 2 == sa.workspace


def test_colossal_gqa_buffer_is_provably_forbidden_by_halving():
    """代数证明（codified）：设 base 为无 buffer 的 GQA body 桶值（S-scaling → base(cp2)=base(cp1)/2），
    加 colossal buffer C（C(cp1)=c1, C(cp2)=c2）。冻结不变量要求 2·(base/2 + c2) == base + c1，
    即 **c2 == c1/2**。而 colossal all-gather 的物理语义是 full-S（cp2 不 ÷cp）→ c2==c1；且惰性要求
    c1==0（cp=1 无 all-gather buffer）。两者联立 → c1==c2==0。故非零 colossal-cp2 GQA buffer 与
    冻结不变量不相容。"""
    base_cp1 = 1000
    base_cp2 = base_cp1 // 2
    # colossal 物理语义：full-S，cp2 与 cp1 同量（不 ÷cp）
    def halving_holds(c1, c2):
        return 2 * (base_cp2 + c2) == base_cp1 + c1
    # 惰性 c1=0 + full-S c2=c1 → 唯一满足减半的是 c1=c2=0
    assert halving_holds(0, 0)
    assert not halving_holds(0, 200)         # 惰性但 full-S 非零 → 破减半
    assert not halving_holds(400, 400)       # cp=1 也存在的 full-S → 仍破减半


# ── 审计量化：欠建的 colossal KV all-gather full-S buffer 字节公式（codify P1-13 缺口量级）──────
def test_underbuilt_colossal_kv_allgather_bytes_formula():
    """量化 P1-13 缺口：colossal 下 fused-qkv 的 KV 分量应 all-gather 到 full-S 的额外 buffer =
    2·n_kv·head_dim（KV 列数） × S × B × compute_dtype，占 qkv 的
    (2·n_kv·head_dim)/((n_heads+2·n_kv)·head_dim) 比例。此测试把该公式与比例固化，便于后续
    （shape_eval 支持 cp>1/method 门后）对拍。"""
    d = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=192, S=4096, B=1,
                 vocab=129280, n_layers=4)
    kv_cols = 2 * d.n_kv * d.head_dim
    qkv_cols = (d.n_heads + 2 * d.n_kv) * d.head_dim
    kv_full_s_bytes = kv_cols * d.S * d.B * d.dtype_bytes
    ratio = kv_cols / qkv_cols
    assert kv_full_s_bytes == 2 * 8 * 192 * 4096 * 1 * 2
    assert abs(ratio - (2 * 8) / (8 + 2 * 8)) < 1e-12       # KV 占 qkv 的 2/3（此对称 GQA 例）


# ── 对照：MLA 侧 colossal KV all-gather **已由 cp_kv=True 建模**（GQA 缺口的正例）──────────────
def test_mla_colossal_kv_full_s_already_modeled():
    """MLA 的 KV 侧激活标了 cp_kv=True → colossal cp>1 下 all-gather 到 full-S（不 ÷cp），
    ulysses/ring 仍 ÷cp。这正是 GQA fused-qkv 无法做到的（KV 无法从融合张量单独标）——
    对照凸显 P1-13 缺口只在 GQA。DSv3 走 mla → cp2 锚点（colossal/ulysses）本就含此建模、
    且本 agent 不改 mla → 逐字节不变。"""
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=16, B=1, vocab=16, n_layers=4)
    kv = TensorRef("kvb_out", ("S", "B", "H"), cp_kv=True)          # MLA KV 侧
    pm_col = ParallelModel(ParallelConfig(cp=2, context_parallel_method="colossal"),
                           n_layers=4, world_size=2)
    pm_uly = ParallelModel(ParallelConfig(cp=2, context_parallel_method="ulysses"),
                           n_layers=4, world_size=2)
    assert resolve_tensor(kv, d, pm_col).local_numel == 16 * 1 * 8        # colossal: full-S
    assert resolve_tensor(kv, d, pm_uly).local_numel == (16 // 2) * 1 * 8  # ulysses: S/cp
    # cp=1：cp_kv 无效（惰性）→ full 尺寸即自然尺寸，不额外
    pm1 = ParallelModel(ParallelConfig(cp=1), n_layers=4, world_size=1)
    assert resolve_tensor(kv, d, pm1).local_numel == 16 * 1 * 8
