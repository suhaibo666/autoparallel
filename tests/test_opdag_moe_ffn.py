# tests/test_opdag_moe_ffn.py
"""ACCEPTANCE:抽取**真** mindformers MoE FFN Cell(`FFNGroupedGEMM`),还原 grouped-GEMM 中间量
——某 18% 显存低估的根因。夹具 = 真源(仅 ast 静态读,绝不 import/执行 mindspore/mindformers)。

────────────────────────────────────────────────────────────────────────────
R3 判定(grouped-GEMM 中间量:显式 op 还是 opaque 融合核?)—— 关键结论
────────────────────────────────────────────────────────────────────────────
`FFNGroupedGEMM.construct`(ffn.py:135)里 experts 计算被 **Morph 自定义融合原语**
`self.morphed_forward = Morph(self.forward_func, ...)`(ffn.py:124)包裹。construct 顶层显式 op 仅:
  Shape(141) → Cast tokens(145) / Cast w1(146) / Cast w2(147) → **Morph(149,opaque)** → Cast out(152)。
但 Morph 包的 `forward_func`(ffn.py:155)是**可静态读的 Python**(非编译黑核),内部:
  * reshape w1/w2(157/158)= 显式 View;
  * `self.token_dispatcher.token_permutation(...)`(160)= **AllToAll 派发 + 排序/容量填充**——
    子模块方法、数据依赖 → 本库标准 op 粒度下 **opaque**(不发射);
  * `self.experts_forward(...)`(163)= 分组 GEMM,**显式**:
      `GroupedMatmul(...)([permuted],[w1],...)`(178)+ `Swiglu`(180)+ `GroupedMatmul(...)([...],[w2],...)`(181);
  * `self.token_dispatcher.token_unpermutation(...)`(165)= **combine**,同 opaque。

⇒ **VERDICT**:grouped-GEMM 的两个 `GroupedMatmul` + `Swiglu` + 其操作数(permute 后的 token 缓冲
`dispatched_input`、专家权重 w1/w2、swiglu 输入/输出)**是 Python 源里的显式 op,可静态还原**
(经 Morph→forward_func→experts_forward 的方法内联 + 直接实例化算子 `GroupedMatmul(...)(...)` 识别)。
它们正是先前被当作 Morph 黑盒而**漏掉的 18% 中间量**——现被 `derive_saves` 按 GroupedMatMul 规则 pin 住。

而 **token permute/unpermute(AllToAll 派发、容量填充、排序缓冲)与 TopKRouter 的 topk/argmax/aux-loss
是数据依赖 / opaque**:permute/combine 是 token_dispatcher 的子模块方法(不发射);TopKRouter 的
`routing()` 用 `ops.one_hot/masked_fill/repeat_interleave/TopkExt`(非标准 op)且 `self.gating_activation`
经字典查表绑定 → **无法静态忠实分解**。故顶层 `MoELayer` 抽取会在 `self.router`(moe_layer.py:219)
**fail-loud**(见 test_moelayer_extraction_is_blocked_at_data_dependent_router)。

⇒ 这部分残差(容量填充/排序缓冲 + 路由中间量)**需 Task-11 标定 margin**,不静默省略。
"""
import os

import pytest

from cost_eval.opdag.extractor import extract_cell
from cost_eval.opdag.module_resolver import ResolvedSpec
from cost_eval.opdag.bprop_rules import derive_saves


MF_ROOT = os.environ.get(
    "MINDFORMERS_ROOT",
    r"E:\97-codes\torch_parallel\mindformers\mindformers",
)
FFN_REL = "parallel_core/training_graph/transformer/moe/ffn.py"
MOE_REL = "parallel_core/training_graph/transformer/moe/moe_layer.py"

# DSv3 MoE 派生 flags(见 configs/deepseek3/pretrain_deepseek3_671b.yaml:137/146/162):
#   moe_token_dispatcher_type="alltoall"(deredundency 支被剪)、compute_dtype=bf16、add_bias_linear=False。
DSV3_FFN_FLAGS = {
    "moe_token_dispatcher_type": "alltoall",
    "compute_dtype": "bf16",
    "add_bias_linear": False,
}


def _require_mf():
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")


@pytest.fixture(scope="module")
def ffn_spec():
    # FFNGroupedGEMM 无 build_module 子模块(权重是 Parameter、token_dispatcher 直接实例化)。
    return ResolvedSpec(cell="FFNGroupedGEMM", submodules={})


@pytest.fixture(scope="module")
def ffn_dag(ffn_spec):
    _require_mf()
    # grouped-GEMM 靠 Morph→forward_func 内联 + 直接实例化算子识别,不依赖子 Cell 递归(recurse 无关)。
    return extract_cell(MF_ROOT, FFN_REL, "FFNGroupedGEMM", ffn_spec, DSV3_FFN_FLAGS)


# ── grouped-GEMM 结构(真 ffn.py 逐行对齐)──────────────────────────────────────
def test_ffn_full_op_sequence(ffn_dag):
    ops = [n.op for n in ffn_dag.nodes]
    assert ops == [
        "View",                    # 141 shape
        "Cast", "Cast", "Cast",    # 145 tokens→bf16, 146 w1→bf16, 147 w2→bf16
        "View", "View",            # 157/158 reshape w1/w2 到 [E,h,H]/[E,H,h](Morph.forward_func)
        "GroupedMatMul",           # 178 fc1(experts_forward)
        "Activation",              # 180 swiglu
        "GroupedMatMul",           # 181 fc2
        "Cast",                    # 152 output→原 dtype
    ]


def test_ffn_src_lines_align_with_ffn_py(ffn_dag):
    assert all(n.src.startswith("ffn.py:") for n in ffn_dag.nodes)
    lines = [int(n.src.split(":")[1]) for n in ffn_dag.nodes]
    assert lines == [141, 145, 146, 147, 157, 158, 178, 180, 181, 152]
    # AllToAll 派发(160)/ combine(165)不在标准 op 粒度里 → 不发射(opaque,见文件头 R3)。
    assert 160 not in lines and 165 not in lines


def test_ffn_has_two_grouped_matmuls_and_one_activation(ffn_dag):
    gmm = [n for n in ffn_dag.nodes if n.op == "GroupedMatMul"]
    assert len(gmm) == 2                                  # 分组专家 fc1 + fc2
    assert [int(n.src.split(":")[1]) for n in gmm] == [178, 181]
    acts = [n for n in ffn_dag.nodes if n.op == "Activation"]
    assert len(acts) == 1 and int(acts[0].src.split(":")[1]) == 180   # swiglu(门控)


def test_ffn_grouped_matmul_operands_include_permuted_tokens_and_weights(ffn_dag):
    gmm = [n for n in ffn_dag.nodes if n.op == "GroupedMatMul"]
    fc1, fc2 = gmm
    # fc1 消费 permute 后的 token 缓冲(dispatched_input)+ 专家权重 w1;fc2 消费 swiglu 输出 + w2。
    fc1_names = [i.split(":")[0] for i in fc1.ins]
    fc2_names = [i.split(":")[0] for i in fc2.ins]
    assert any(nm.startswith("dispatched_input") for nm in fc1_names)   # 先前被漏的 permute 后 token
    assert "w1" in fc1_names
    assert any(nm.startswith("intermediate_parallel") for nm in fc2_names)  # swiglu 输出
    assert "w2" in fc2_names


# ── save-set:grouped-GEMM 操作数被 pin(先前缺失的 18% 中间量)────────────────────
def test_ffn_saveset_includes_previously_missing_grouped_gemm_operands(ffn_dag):
    saves = derive_saves(ffn_dag)
    names = {s.name for s in saves}
    prefixes = {n.split("__")[0] for n in names}   # 去内联帧后缀(__i0/__i1)
    # GroupedMatMul(inputs=all)pin:permute 后 token(dispatched_input)、专家权重 w1/w2、
    # swiglu 输入(fc1_output)与输出(intermediate_parallel)——正是先前当 Morph 黑盒漏掉的中间量。
    assert "dispatched_input" in prefixes        # ★ 先前缺失:permute 后的 token 激活(18% 根因)
    assert "intermediate_parallel" in prefixes   # swiglu 输出(喂 fc2)
    assert "fc1_output" in prefixes              # swiglu 输入(Activation pin)
    # ── 2026-07-25 期望迁移(W2/W3/W4;台账见 `docs/opdag_component_coverage_2026-07-25.md`)──
    # 原断言 `"w1" in names and "w2" in names` 钉的是一个**病症**:专家权重被当激活 save 计。
    # 评估文档 §7.2 已实测其字节代价 —— FFNGroupedGEMM「236 MiB」里 `w1`(58.7MB)+`w2`(29.4MB)
    # = **88 MiB 是权重**。契约 W2/W3/W4 要求权重永不进 `saves`。
    # 不变量(「GroupedMatMul 的**激活**操作数必须被 pin」)逐字保留在上面三条;
    # 这里反过来钉住权重**不**在 saves 里,同时确认它们仍**在 `ins` 里可见**
    # (`shape_infer._grouped_matmul` 要靠权重末轴推输出维,`w1`/`w2` 不能从 ins 消失)。
    assert "w1" not in names and "w2" not in names
    fc1, fc2 = [n for n in ffn_dag.nodes if n.op == "GroupedMatMul"]
    assert "w1" in [i.split(":")[0] for i in fc1.ins]
    assert "w2" in [i.split(":")[0] for i in fc2.ins]
    # 权重派生的 ins 下标由 walker 显式标出(不是"恰好没被选中")
    assert fc1.attrs["weight_ins_idx"] == [1] and fc2.attrs["weight_ins_idx"] == [1]


def test_ffn_saveset_permuted_tokens_pinned_by_grouped_matmul(ffn_dag):
    saves = derive_saves(ffn_dag)
    gmm_ids = {n.id for n in ffn_dag.nodes if n.op == "GroupedMatMul"}
    permuted = [s for s in saves if s.name.startswith("dispatched_input")]
    assert len(permuted) == 1
    # 该 permute 后 token 缓冲由某个 GroupedMatMul(fc1)pin —— 这条 save 是先前 18% 低估被补回的核心。
    assert permuted[0].op_id in gmm_ids
    assert permuted[0].dtype == "bf16"


def test_ffn_grouped_gemm_dataflow_edges(ffn_dag):
    e = ffn_dag.edges
    ids = {n.id: n for n in ffn_dag.nodes}
    gmm = [n.id for n in ffn_dag.nodes if n.op == "GroupedMatMul"]
    act = next(n.id for n in ffn_dag.nodes if n.op == "Activation")
    fc1, fc2 = gmm
    # fc1 → swiglu → fc2(门控 GEMM 主链)
    assert [fc1, act] in e and [act, fc2] in e


# ── R3 边界:顶层 MoELayer 在数据依赖的 TopKRouter 处 fail-loud(不杜撰)──────────────
def test_moelayer_extraction_is_blocked_at_data_dependent_router():
    """顶层 MoELayer.construct 可分解 transpose(212)/shape(213)/local_reshape_permute(216, Morph→View),
    随后 `self.router(x_reshaped)`(219)= 数据依赖的 TopKRouter(topk/one_hot/masked_fill/字典查表激活)
    —— 标准 op 粒度无法忠实分解 → **fail-loud**(而非静默产错 DAG)。这是 R3 记录的 opaque 边界:
    路由中间量 + AllToAll permute/combine 的容量填充/排序缓冲 需 Task-11 标定 margin。"""
    _require_mf()
    spec = ResolvedSpec(cell="MoELayer", submodules={
        "experts": "FFNGroupedGEMM",
        "shared_experts": "SharedExpertMLPInterleaved",
    })
    subcell_specs = {
        "FFNGroupedGEMM": ResolvedSpec(cell="FFNGroupedGEMM", submodules={}),
        "SharedExpertMLPInterleaved": ResolvedSpec(
            cell="SharedExpertMLPInterleaved",
            submodules={"linear_fc1": "ColumnParallelLinear", "linear_fc2": "RowParallelLinear"},
        ),
    }
    flags = dict(DSV3_FFN_FLAGS)
    with pytest.raises(ValueError) as ei:
        extract_cell(MF_ROOT, MOE_REL, "MoELayer", spec, flags,
                     recurse=True, subcell_specs=subcell_specs)
    msg = str(ei.value)
    assert "router" in msg and "moe_layer.py:219" in msg
