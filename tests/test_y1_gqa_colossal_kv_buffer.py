"""Y1 — P1-13 闭环：GQA fused-QKV 的 colossal CP KV all-gather full-S buffer。

背景（X3 审计，见 `attention.py build_gqa_attn_ops` 的 P1-13 注释 + `test_x3_cp_buffer.py`）：
  MLA 靠**独立 KV 激活**标 `cp_kv=True` 表达 colossal 下 KV all-gather 到 full-S；GQA 的 KV
  是**融合**在 `qkv`（末维 (n_heads+2·n_kv)·head_dim）里的 (2·n_kv·head_dim) 分量，**无法从融合
  张量单独标 cp_kv** → colossal 下这段 KV 仍随 body ÷cp、**欠建 full-S all-gather buffer**。

Y1 补的机制（shape_eval + attention）：**method+cp 门控的 workspace**。
  - builder（attention.py）**无条件**把 `attrs["colossal_kv_ws"]` 挂在 GQA flash op（builder 不知
    cp/method，只表达「若走 colossal all-gather，额外多这么多 full-S 字节」）。
  - shape_eval.resolve **三重门**决定是否把它加进 `workspace_bytes`（缺一则 0，惰性）：
      ① `context_parallel_method == "colossal"` 且 `cp > 1`；
      ② op.attrs 带 `"colossal_kv_ws"`；
      ③ `getattr(spec.dims, "cp_kv_allgather_buffer", False)`（modeling opt-in，默认关）。
  - 门通过 → `ws += eval_expr(colossal_kv_ws)`（**full-S，不 ÷cp**，因是 all-gather 到 full-S）。

公式（X3 已固化，见 `test_x3_cp_buffer.test_underbuilt_colossal_kv_allgather_bytes_formula`）：
  `2·n_kv·head_dim · S · B · dtype`（KV 列数 × full-S × B × compute dtype），占 fused qkv 的
  (2·n_kv·head_dim)/((n_heads+2·n_kv)·head_dim) 比例。ring/ulysses/hybrid 是**另一套语义**
  （CP 通信在飞双缓冲 KV 块 send/recv，随 body ÷cp、非 all-gather buffer）→ 门 ① 挡掉。

**为何 opt-in 默认关（③）**：本量 off loss 峰、本栈跑不了 cp+无重算 → **未真机验证**；且 12 golden
锚点走 mla/dsv4、不经 GQA builder。默认关 → golden 逐字节不变、守住 X3 冻结不变量
（`test_gqa_colossal_cp2_body_halves_exactly`：未 opt-in 的 GQA colossal cp2 仍精确减半）；开启即
按公式计入。这是「机制已就绪、默认口径待标定」的保守工程口径（同库内 qk_layernorm/moe_dispatch_mode/
kept_frag_factor 等惰性特性）。
"""
from cost_eval.model_spec import DimTable, TensorRef, OpSpec, OpType, ModelSpec, LayerSpec
from cost_eval.shape_eval import ShapeEval, resolve_tensor
from cost_eval.parallel_model import ParallelModel
from cost_eval.specs import ParallelConfig
from cost_eval.llm_config import LLMConfig
from cost_eval.build_llm import build_llm_spec
from cost_eval.structure_mem import estimate_structure_memory
from cost_eval.layers.attention import build_gqa_attn_ops, build_mla_attn_ops


# ── X3 固化公式（Y1 复用；builder 应挂同串）─────────────────────────────────────
COLOSSAL_KV_WS = "2*n_kv*head_dim*S*B*dtype_bytes"


def _dims(**over):
    d = DimTable(H=8, F=16, n_heads=2, n_kv=2, head_dim=4, S=16, B=1, vocab=16, n_layers=4)
    for k, v in over.items():
        setattr(d, k, v)
    return d


def _flash_op_with_attr():
    """最小 GQA-like flash op：只挂 colossal_kv_ws attr、**无** workspace_ref（隔离 buffer 贡献
    → workspace_bytes 恰等于 buffer 值或 0）。"""
    qkv = TensorRef("qkv", ("S", "B", "(n_heads+2*n_kv)*head_dim"), shard={2: "tp"})
    attn = TensorRef("attn", ("S", "B", "n_heads*head_dim"), shard={2: "tp"})
    return OpSpec("flash", OpType.FLASH_ATTN, [qkv], attn, saves=[qkv, attn],
                  attrs={"colossal_kv_ws": COLOSSAL_KV_WS})


def _resolve_ws(d, cp, method="colossal", world_size=8):
    op = _flash_op_with_attr()
    spec = ModelSpec("t", d, ["x"], {"x": LayerSpec([op])})
    pm = ParallelModel(ParallelConfig(cp=cp, context_parallel_method=method),
                       n_layers=4, world_size=world_size)
    return ShapeEval().resolve(spec, pm).stages[0][0].ops[0].workspace_bytes


# ═══════════════════════════════════════════════════════════════════════════════
# A. shape_eval 门控机制（单 op 隔离）
# ═══════════════════════════════════════════════════════════════════════════════

def test_flag_off_never_adds_buffer():
    """③ opt-in 默认关：任何 (cp, method) 组合下 buffer=0（workspace_bytes=0，因无 workspace_ref）。"""
    d = _dims()                                          # cp_kv_allgather_buffer 未设 → getattr False
    for method in ("colossal", "ulysses", "ring", "hybrid"):
        for cp in (1, 2, 4):
            assert _resolve_ws(d, cp, method) == 0, f"flag-off {method} cp{cp} 不应有 buffer"


def test_flag_on_colossal_cp_gt1_adds_full_s_buffer():
    """①②③ 全通过：colossal cp>1 → buffer = full-S = 2·n_kv·head_dim·S·B·dtype（**不 ÷cp**）。"""
    d = _dims(cp_kv_allgather_buffer=True)
    full_s = 2 * d.n_kv * d.head_dim * d.S * d.B * d.dtype_bytes
    assert _resolve_ws(d, 2, "colossal") == full_s
    # full-S 不 ÷cp：cp=4 与 cp=2 同值（all-gather 恒到 full-S）
    assert _resolve_ws(d, 4, "colossal") == full_s
    assert full_s == 2 * 2 * 4 * 16 * 1 * 2              # 固化数值


def test_flag_on_cp1_no_buffer():
    """① cp 门：cp=1 无 all-gather → buffer=0（即便 flag on / method colossal）。"""
    d = _dims(cp_kv_allgather_buffer=True)
    assert _resolve_ws(d, 1, "colossal") == 0


def test_flag_on_non_colossal_no_buffer():
    """① method 门：ulysses/ring/hybrid 是双缓冲 send/recv（随 body ÷cp）、**非** all-gather
    buffer → 该项 0。"""
    d = _dims(cp_kv_allgather_buffer=True)
    for method in ("ulysses", "ring", "hybrid"):
        assert _resolve_ws(d, 2, method) == 0, f"{method} 不应有 colossal all-gather buffer"


def test_buffer_adds_on_top_of_existing_flash_workspace():
    """② buffer 与 flash op 已有 workspace_ref（fa_ws）**共存相加**（同 op workspace_bytes）——
    真机 all-gather 的 KV 须在 flash 计算期驻留 → SUM，非 max。"""
    d = _dims(cp_kv_allgather_buffer=True)
    qkv = TensorRef("qkv", ("S", "B", "(n_heads+2*n_kv)*head_dim"), shard={2: "tp"})
    attn = TensorRef("attn", ("S", "B", "n_heads*head_dim"), shard={2: "tp"})
    fa_ws = TensorRef("fa_ws", ("2", "B", "n_heads", "S", "8"), shard={2: "tp"}, dtype_bytes=4)
    op = OpSpec("flash", OpType.FLASH_ATTN, [qkv], attn, saves=[qkv, attn],
                workspace_ref=fa_ws, attrs={"colossal_kv_ws": COLOSSAL_KV_WS})
    spec = ModelSpec("t", d, ["x"], {"x": LayerSpec([op])})

    def ws(cp, method="colossal"):
        pm = ParallelModel(ParallelConfig(cp=cp, context_parallel_method=method),
                           n_layers=4, world_size=8)
        return ShapeEval().resolve(spec, pm).stages[0][0].ops[0].workspace_bytes

    fa_full = 64 * d.B * d.n_heads * d.S                  # fa_ws numel×4B (TP=1/CP=1)
    kv_full = 2 * d.n_kv * d.head_dim * d.S * d.B * d.dtype_bytes
    assert ws(1, "colossal") == fa_full                  # cp1：只 fa_ws（buffer 惰性）
    assert ws(2, "colossal") == fa_full // 2 + kv_full   # colossal cp2：减半 fa_ws + full-S KV buffer
    assert ws(2, "ulysses") == fa_full // 2              # ulysses cp2：只减半 fa_ws（无 buffer）


# ═══════════════════════════════════════════════════════════════════════════════
# B. builder 契约（attention.py）
# ═══════════════════════════════════════════════════════════════════════════════

def test_gqa_flash_op_carries_colossal_kv_ws_attr():
    """GQA builder 的 flash op **无条件**挂 colossal_kv_ws attr（值 = X3 固化公式串）。"""
    d = _dims()
    ops = build_gqa_attn_ops(d)
    flash = next(o for o in ops if o.name == "flash")
    assert flash.attrs.get("colossal_kv_ws") == COLOSSAL_KV_WS


def test_mla_flash_op_has_no_colossal_kv_ws_attr():
    """MLA 侧**不**用此 buffer（KV all-gather 早由独立 KV 激活的 cp_kv=True 建模）→ MLA flash op
    无 colossal_kv_ws attr。这是 GQA 缺口的正例，也保 MLA 锚点（DSv3 cp2）不受 Y1 影响。"""
    d = _dims(q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=2,
              qk_nope_head_dim=2, v_head_dim=4)
    ops = build_mla_attn_ops(d)
    flash = next(o for o in ops if o.name == "flash")
    assert "colossal_kv_ws" not in flash.attrs


# ═══════════════════════════════════════════════════════════════════════════════
# C. 集成（build_llm GQA / MLA）
# ═══════════════════════════════════════════════════════════════════════════════

def _gqa_spec(**over):
    base = dict(num_layers=2, hidden_size=8, num_attention_heads=2, num_query_groups=2,
                vocab_size=16, seq_length=16, batch_size=1, head_dim=4, attn_type="gqa",
                ffn_hidden_size=16)
    base.update(over)
    return build_llm_spec(LLMConfig(**base))


def _layers(spec, cp, method="colossal"):
    pm = ParallelModel(ParallelConfig(cp=cp, context_parallel_method=method),
                       n_layers=len(spec.layer_pattern), world_size=64)
    g = ShapeEval().resolve(spec, pm)
    return [l for st in sorted(g.stages) for l in g.stages[st]]


def test_integration_gqa_colossal_buffer_and_gates():
    """集成：build_llm GQA + opt-in on。flash 层：colossal cp2 = 减半 fa_ws + full-S KV buffer；
    ulysses cp2 精确减半；cp1 无 buffer。body 激活 saves 三路径皆精确 ÷cp。"""
    spec = _gqa_spec(cp_kv_allgather_buffer=True)        # opt-in 经 LLMConfig 真字段（F8 订正）
    d = spec.dims
    l1 = _layers(spec, 1, "colossal")
    l2 = _layers(spec, 2, "colossal")
    l2u = _layers(spec, 2, "ulysses")
    kv_buf = 2 * d.n_kv * d.head_dim * d.S * d.B * d.dtype_bytes
    seen_flash = False
    for a, b, bu in zip(l1, l2, l2u):
        if not any(o.name == "flash" for o in a.ops):
            continue
        seen_flash = True
        sa, sb, sbu = (estimate_structure_memory(x.ops) for x in (a, b, bu))
        # body 激活 saves 精确 ÷cp（无论 colossal/ulysses；all-gather 只进 workspace、不进 saves）
        assert sb.activation_saves * 2 == sa.activation_saves
        assert sbu.activation_saves * 2 == sa.activation_saves
        # colossal cp2 workspace = 减半 fa_ws + full-S KV buffer（非减半项）
        assert sb.workspace == sa.workspace // 2 + kv_buf
        # ulysses cp2 workspace 精确减半（无 all-gather buffer）
        assert sbu.workspace * 2 == sa.workspace
    assert seen_flash


def test_integration_flag_off_is_byte_identical_halving():
    """opt-in **默认关**（不 setattr）：GQA colossal cp2 flash workspace 仍**精确减半**——守 X3 冻结
    不变量 `test_gqa_colossal_cp2_body_halves_exactly` 与 12 golden 锚点逐字节不变。"""
    spec = _gqa_spec()                                   # 默认不设 cp_kv_allgather_buffer
    l1 = _layers(spec, 1, "colossal")
    l2 = _layers(spec, 2, "colossal")
    for a, b in zip(l1, l2):
        if not any(o.name == "flash" for o in a.ops):
            continue
        sa, sb = estimate_structure_memory(a.ops), estimate_structure_memory(b.ops)
        assert sb.workspace * 2 == sa.workspace          # 无 buffer → 精确减半


def test_integration_mla_unaffected_by_opt_in():
    """MLA 走 build_mla_attn_ops（flash op 无 colossal_kv_ws attr）→ 即便 opt-in on，colossal cp2
    workspace 与 opt-in off 逐字节相同。守 DSv3(mla) cp2 锚点不受 Y1 影响。"""
    base = dict(num_layers=2, hidden_size=8, num_attention_heads=2, vocab_size=16,
                seq_length=16, batch_size=1, head_dim=4, attn_type="mla", ffn_hidden_size=16,
                q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=2, qk_nope_head_dim=2,
                v_head_dim=4)
    spec_off = build_llm_spec(LLMConfig(**base))
    spec_on = build_llm_spec(LLMConfig(**base, cp_kv_allgather_buffer=True))   # 真字段（F8 订正）
    for a, b in zip(_layers(spec_off, 2, "colossal"), _layers(spec_on, 2, "colossal")):
        sa, sb = estimate_structure_memory(a.ops), estimate_structure_memory(b.ops)
        assert sa.workspace == sb.workspace              # MLA 不受 opt-in 影响


# ═══════════════════════════════════════════════════════════════════════════════
# D. 公式 / 比例固化
# ═══════════════════════════════════════════════════════════════════════════════

def test_kv_allgather_formula_and_ratio():
    """固化 P1-13 公式与 KV 占比（对称 GQA 大例，对齐 X3 test_underbuilt_...）。"""
    d = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=192, S=4096, B=1,
                 vocab=129280, n_layers=4)
    kv_cols = 2 * d.n_kv * d.head_dim
    qkv_cols = (d.n_heads + 2 * d.n_kv) * d.head_dim
    kv_full_s = kv_cols * d.S * d.B * d.dtype_bytes
    assert kv_full_s == 2 * 8 * 192 * 4096 * 1 * 2
    assert abs(kv_cols / qkv_cols - (2 * 8) / (8 + 2 * 8)) < 1e-12   # KV 占 qkv 的 2/3
