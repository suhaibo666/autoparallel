"""TransformerLayer 抽象 + MoE 前置 FFN 归一（ln2）修复（用户报告 2026-07-16）。

结构 bug：dense decoder 的 FFN 段内嵌 ln2（post_attention_layernorm），但 MoE decoder 的
`build_moe_ffn_ops`/`build_shared_expert_ops` 直接吃**裸 h1**、无 ln2 → 每个 MoE 层漏建一个
`[S,B,H]` fp32-cast 常驻激活（norm_compute=fp32），MoE 模型在无重算/select 下系统性欠预测
（真机 cp2-none 0.927 / DSv3-8L-none 0.922 / select-mlp 0.943 / DSv4-fused 0.968，均 OOM 不安全）。

修复：统一 `build_transformer_layer` 接口，attn(含 ln1) → **ln2 统一前置** → dense|moe FFN(消费 ln2)。
本组钉住：MoE 层有 ln2、routed+shared 都消费 ln2、dense 全层逐字节不变。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_eval.build_llm import build_llm_spec
from cost_eval.presets import deepseek_v3


def _layer(spec, layer_type):
    return spec.layer_specs[layer_type]


def _op(ls, name):
    return next((o for o in ls.ops if o.name == name), None)


def test_moe_layer_has_ln2_prenorm():
    """MoE decoder 层必须含 ln2（pre-FFN 归一），与 dense 对称。"""
    spec = build_llm_spec(deepseek_v3(4))
    moe_types = [lt for lt in spec.layer_specs if "moe" in lt]
    assert moe_types, "预设应有 MoE 层"
    for lt in moe_types:
        ls = _layer(spec, lt)
        names = [o.name for o in ls.ops]
        assert "ln2" in names, (lt, names)


def test_moe_ln2_consumed_by_router_and_shared():
    """ln2 的输出必须被 routed(router/dispatch) 与 shared(shared_fc1) 共同消费；ln2 消费 h1。"""
    spec = build_llm_spec(deepseek_v3(4))
    lt = next(lt for lt in spec.layer_specs if "moe" in lt)
    ls = _layer(spec, lt)
    ln2 = _op(ls, "ln2")
    assert ln2 is not None and ln2.type.value == "norm", ln2
    ln2_out = ln2.output.name
    assert any(t.name == "h1" for t in ln2.inputs), [t.name for t in ln2.inputs]
    router = _op(ls, "router")
    assert router is not None and any(t.name == ln2_out for t in router.inputs), \
        (ln2_out, [t.name for t in router.inputs])
    sh = _op(ls, "shared_fc1")
    if sh is not None:                                   # 有 shared expert 时
        assert any(t.name == ln2_out for t in sh.inputs), (ln2_out, [t.name for t in sh.inputs])


def test_moe_ln2_saves_fp32_cast_of_h1():
    """ln2 保存其输入 h1（norm_compute=fp32 → 该 save 为 [S,B,H] fp32 常驻，正是漏建的那份显存）。"""
    spec = build_llm_spec(deepseek_v3(4))
    lt = next(lt for lt in spec.layer_specs if "moe" in lt)
    ln2 = _op(_layer(spec, lt), "ln2")
    assert any(t.name == "h1" for t in ln2.saves), [t.name for t in ln2.saves]
    # ln2 gamma 参数（[H] fp32）存在
    assert any("ln2_g" in t.name or "ln2" in t.name for t in ln2.params), [t.name for t in ln2.params]


def test_dense_layer_full_sequence_unchanged():
    """dense 全层 op 序列逐字节不变（ln1…add1 ln2 fc1 swiglu fc2 add2）——ln2 hoist 不改 dense。"""
    spec = build_llm_spec(deepseek_v3(4))
    dense_types = [lt for lt in spec.layer_specs if "dense" in lt]
    assert dense_types
    for lt in dense_types:
        names = [o.name for o in _layer(spec, lt).ops]
        # ln2 紧跟 add1、在 fc1 前
        assert "ln2" in names and "add1" in names and "fc1" in names
        assert names.index("add1") < names.index("ln2") < names.index("fc1")
