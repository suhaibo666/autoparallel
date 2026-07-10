"""opdag.render：op-DAG / 时间线 → 自包含 SVG（纯 Python）。验证 SVG 结构 + saved 高亮 + 时间线折线。"""
from cost_eval.opdag.schema import OpNode, OpDAG
from cost_eval.opdag.render import dag_to_svg, timeline_to_svg


def _dag():
    # cast(bf16→fp32) → norm(存 fp32 输入) → matmul(存操作数)。norm 的 fp32 输入是 saved。
    return OpDAG(cell="T", nodes=[
        OpNode(id=1, op="Cast", src="x:1", ins=["h:S·B·H:bf16"], out="h32:S·B·H:fp32", attrs={"to_dtype": "fp32"}),
        OpNode(id=2, op="Norm", src="x:2", ins=["h32:S·B·H:fp32"], out="n:S·B·H:bf16"),
        OpNode(id=3, op="MatMul", src="x:3", module="ColumnParallelLinear",
               ins=["n:S·B·H:bf16", "W:H·F:bf16"], out="y:S·B·F:bf16", attrs={"out_dim": "F"}),
    ], edges=[[1, 2], [2, 3]])


def test_dag_svg_is_wellformed():
    svg = dag_to_svg(_dag())
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg
    # 每个节点一个 op 标签
    for op in ("Cast", "Norm", "MatMul"):
        assert op in svg


def test_dag_svg_highlights_saved_vs_transient():
    svg = dag_to_svg(_dag())
    # norm 的输入 h32 被 pin（saved）→ 至少一个红框（stroke #c0392b width 3）
    assert 'stroke="#c0392b" stroke-width="3"' in svg
    # cast 输出 h32 是 saved（被 norm 消费者 pin）；n 非 saved → 有 transient 标记
    assert "transient" in svg


def test_dag_svg_shows_bytes_when_dims_given():
    class Dims:
        S, B, H, F = 4096, 1, 1792, 3072
        dtype_bytes = 2
        n_experts = topk = 0
        capacity_factor = 1.0
    svg = dag_to_svg(_dag(), Dims())
    assert "MiB" in svg                      # 有 dims → saved 张量给出字节
    assert "saved(激活驻留) 合计" in svg      # 标题含总 saved 字节


def test_timeline_svg_from_real_evaluate():
    from validate_dsv3 import build_dsv3_spec
    from cost_eval.specs import ParallelConfig, OptimizerSpec, HardwareSpec, RecomputeSpec, SwapSpec
    from cost_eval.report import Evaluator
    spec, d, fl = build_dsv3_spec(4)
    pc = ParallelConfig(dp_shard=2, cp=1, tp=1, pp=1, sequence_parallel=True, num_microbatches=1)
    r = Evaluator(spec, pc, OptimizerSpec.adamw(params_fp32=True, grad_dtype_bytes=4),
                  HardwareSpec(max_device_memory=64 * 2 ** 30, framework_reserve=0),
                  RecomputeSpec("full", full_layers=fl), SwapSpec()).evaluate(record_timeline=True)
    svg = timeline_to_svg(r.per_stage[0].timeline, title="test")
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "<polygon" in svg and "<polyline" in svg      # 堆叠面积 + total 折线
    assert "★" in svg                                     # 峰值标记


def test_timeline_svg_empty_safe():
    svg = timeline_to_svg([])
    assert svg.startswith("<svg") and "empty" in svg
