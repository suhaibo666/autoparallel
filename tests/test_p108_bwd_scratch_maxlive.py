"""P1-08 — bwd_scratch 从「求和上界」精化为 backward **max-live**（TDD）。

审计 P1-08（`analysis/review_closure_audit_2026-07-14.md:102`）判「仍开放（接受的保守上界）」：
`estimate_structure_memory` 里 `bwd_scratch = Σ op.bwd_scratch_bytes` 把一层所有 op 的反向
临时物化**直接求和**，隐含它们同时存活。真机上这些 scratch 是**op 内瞬态**——在各自 op 的反向步
物化、步末即释，不同时存活（loss 链 probs+grad 的真实共存已编码进**单个** `nll` op 的
`8·S·B·vocab`=2 份里）。故求和是保守高估。

本模块把它精化为 **backward max-live**（`_forward_max_live` 的反向镜像）：

模型（**逆序滑窗 window=2**，OOM 安全侧，见 `structure_mem._backward_max_live` docstring）——
op i 的反向 scratch 活在其自身反向步，并**保守地**延续到其反向消费者（前一 forward op =
后一 backward op）的反向步 → 任一 backward 时间点同时存活 = **相邻两 op 的 bwd_scratch 之和**。

关键性质（本模块守卫）：
  - 单 bwd_scratch op 层（loss 的 nll、DSA/dsv4 的 indexer）：max-live == sum（**逐字节不变** →
    K_CE fat 锚点、DSv3/DSv4 golden 不破）。
  - 多 bwd_scratch op 层（分离的 mHC sinkhorn ×2 / 3+ 相邻）：max-live < sum（更准）。
  - 单调性：max-live ≤ sum **恒成立**（OOM 安全，绝不欠估破 OOM 门）。
"""
from cost_eval.model_spec import DimTable, ModelSpec, LayerSpec, OpSpec, OpType, TensorRef
from cost_eval.llm_config import LLMConfig, to_dimtable
from cost_eval.layers.head import build_head_and_loss_ops
from cost_eval.specs import ParallelConfig
from cost_eval.parallel_model import ParallelModel
from cost_eval.shape_eval import ShapeEval
from cost_eval.structure_mem import estimate_structure_memory, _backward_max_live


D = DimTable(H=64, F=128, n_heads=4, n_kv=4, head_dim=16, S=128, B=1, vocab=100, n_layers=1)


class _Op:
    """最小 op 替身：`_backward_max_live` 只读 `bwd_scratch_bytes`。"""

    def __init__(self, bwd_scratch_bytes=0):
        self.bwd_scratch_bytes = bwd_scratch_bytes


def _resolve(op_list, dims=D):
    spec = ModelSpec("t", dims, ["x"], {"x": LayerSpec(list(op_list))})
    pm = ParallelModel(ParallelConfig(), n_layers=1, world_size=1)
    return ShapeEval().resolve(spec, pm).stages[0][0].ops


def _naive_sum(ops):
    return sum(getattr(op, "bwd_scratch_bytes", 0) for op in ops)


# ---------------------------------------------------------------------------
# (1) 纯模型单元测试 —— _backward_max_live（不依赖 op 解析）
# ---------------------------------------------------------------------------

def test_empty_is_zero():
    assert _backward_max_live([]) == 0


def test_single_op_equals_its_value_equals_sum():
    """单 op 层（loss 的 nll / DSA indexer）：max-live == 该 op 值 == sum。**loss 锚点不变的根因。**"""
    ops = [_Op(0), _Op(0), _Op(0), _Op(4236247040)]     # final_norm/lm_head/logsm/nll 形态
    assert _backward_max_live(ops) == 4236247040
    assert _backward_max_live(ops) == _naive_sum(ops)


def test_single_nonzero_in_middle_equals_sum():
    """仅一个非零 scratch（被零邻居包夹）：max-live == 该值 == sum（唯一非零，无叠加）。"""
    ops = [_Op(0), _Op(777), _Op(0)]
    assert _backward_max_live(ops) == 777
    assert _backward_max_live(ops) == _naive_sum(ops)


def test_two_adjacent_is_conservative_pair_sum():
    """相邻两非零 scratch：window=2 → 保守取二者之和（== sum）。文档化「宁可略保守」的选择：
    真机 op 内瞬态其实不叠加（纯 max 更紧），但对相邻对我们保留一步握手余量（OOM 安全）。"""
    ops = [_Op(100), _Op(60)]
    assert _backward_max_live(ops) == 160
    assert _backward_max_live(ops) == _naive_sum(ops)


def test_three_adjacent_maxlive_below_sum():
    """3 个相邻非零 scratch：max-live = max(相邻对和) < sum（精化生效）。"""
    ops = [_Op(100), _Op(60), _Op(40)]
    # 相邻对：100+60=160、60+40=100 → max=160；sum=200 → 160 < 200
    assert _backward_max_live(ops) == 160
    assert _backward_max_live(ops) < _naive_sum(ops)


def test_separated_scratch_reduces_to_pure_max():
    """两非零 scratch 被无-scratch op 分离（真实 mHC：attn_hc / ffn_hc sinkhorn 隔着 body）：
    window=2 退化为纯 max（各自与零邻居配对）→ max-live = max(a,c) < a+c。"""
    ops = [_Op(4000), _Op(0), _Op(3000)]
    assert _backward_max_live(ops) == 4000          # max(4000+0, 0+3000)
    assert _backward_max_live(ops) < _naive_sum(ops)  # 4000 < 7000


def test_monotonic_maxlive_le_sum_always():
    """单调性：任意 bwd_scratch 向量,max-live ≤ sum 恒成立（OOM 安全，绝不欠估）。"""
    import itertools
    for combo in itertools.product([0, 10, 300, 5000], repeat=4):
        ops = [_Op(v) for v in combo]
        assert _backward_max_live(ops) <= _naive_sum(ops)
        # 且 ≥ 任一单 op（不塌到 0，仍覆盖单峰）
        assert _backward_max_live(ops) >= max(combo)


# ---------------------------------------------------------------------------
# (2) 端到端 —— estimate_structure_memory.bwd_scratch 采用 max-live
# ---------------------------------------------------------------------------

def _mk(name, tin, tout, bwd=None, saves=None):
    return OpSpec(name, OpType.ELEMENTWISE, [tin], tout,
                  saves=saves if saves is not None else [tin],
                  **({"bwd_scratch": bwd} if bwd else {}))


def test_single_big_op_layer_maxlive_equals_sum():
    """单大 op 主导层（loss 形态）：sm.bwd_scratch == sum == 该 op 值（不变）。"""
    x = TensorRef("x", ("S", "B", "H"), dtype_bytes=2)
    a = TensorRef("a", ("S", "B", "H"), dtype_bytes=2)
    ops = _resolve([_mk("big", x, a, bwd="8*S*B*vocab")])
    sm = estimate_structure_memory(ops)
    assert sm.bwd_scratch == _naive_sum(ops)
    assert sm.bwd_scratch == 8 * 128 * 1 * 100     # 102400 > 0


def test_separated_multi_scratch_layer_below_sum():
    """多 bwd_scratch op 层（分离，仿 mHC）：sm.bwd_scratch < 朴素求和（max-live 更准），
    且 == max(单 op)（分离 → 纯 max）。"""
    x = TensorRef("x", ("S", "B", "H"), dtype_bytes=2)
    a = TensorRef("a", ("S", "B", "H"), dtype_bytes=2)
    m = TensorRef("m", ("S", "B", "H"), dtype_bytes=2)
    c = TensorRef("c", ("S", "B", "H"), dtype_bytes=2)
    ops = _resolve([
        _mk("scr_a", x, a, bwd="4*S*B*H"),          # 4*128*64 = 32768
        _mk("mid",   a, m),                          # no scratch
        _mk("scr_c", m, c, bwd="2*S*B*H"),          # 2*128*64 = 16384
    ])
    sm = estimate_structure_memory(ops)
    assert _naive_sum(ops) == 32768 + 16384          # 49152
    assert sm.bwd_scratch == 32768                   # max(32768, 16384)
    assert sm.bwd_scratch < _naive_sum(ops)


def test_three_adjacent_scratch_layer_below_sum():
    """3 个相邻 bwd_scratch op：sm.bwd_scratch = 相邻对和峰 < 三者求和。"""
    x = TensorRef("x", ("S", "B", "H"), dtype_bytes=2)
    a = TensorRef("a", ("S", "B", "H"), dtype_bytes=2)
    m = TensorRef("m", ("S", "B", "H"), dtype_bytes=2)
    c = TensorRef("c", ("S", "B", "H"), dtype_bytes=2)
    ops = _resolve([
        _mk("s1", x, a, bwd="4*S*B*H"),   # 32768
        _mk("s2", a, m, bwd="4*S*B*H"),   # 32768
        _mk("s3", m, c, bwd="4*S*B*H"),   # 32768
    ])
    sm = estimate_structure_memory(ops)
    assert _naive_sum(ops) == 3 * 32768              # 98304
    assert sm.bwd_scratch == 2 * 32768               # 相邻对峰 65536
    assert sm.bwd_scratch < _naive_sum(ops)


# ---------------------------------------------------------------------------
# (3) loss 层不变量 —— K_CE fat 锚点 / DSv3 golden 的机理根因
# ---------------------------------------------------------------------------

def test_real_loss_layer_bwd_scratch_unchanged_equals_nll():
    """真实 head+loss 层（final_norm/lm_head/logsoftmax/nll）：唯一大 bwd_scratch 是 nll →
    max-live == sum == nll.bwd_scratch_bytes。守卫 mem_timeline 的 K_CE 逻辑
    （`sm.bwd_scratch // 2 * (K_CE-1)`）在改后逐字节不变（loss 峰锚点不破）。"""
    cfg = LLMConfig(num_layers=2, hidden_size=16, num_attention_heads=2, vocab_size=64,
                    seq_length=8, batch_size=1, attn_type="gqa")
    ops = _resolve(build_head_and_loss_ops(cfg), dims=to_dimtable(cfg))
    nll = next(op for op in ops if op.name == "nll")
    sm = estimate_structure_memory(ops)
    assert nll.bwd_scratch_bytes > 0
    assert sm.bwd_scratch == nll.bwd_scratch_bytes            # 单 op 主导
    assert sm.bwd_scratch == _naive_sum(ops)                 # == 旧求和（逐字节不变）
