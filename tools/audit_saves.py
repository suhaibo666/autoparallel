"""saves 审计器（问题1 的机械化"举一反三"）：按反向依赖规则逐 op 推导**应存集合**，
与手写 builder 声明的 `saves` 对比，报 漏存(MISS)/多存(EXTRA)/一致(OK)。

规则（= opdag bprop_rules.PIN 思路,按手写 OpType 映射）:
  matmul/moe_gemm → 存全部**激活**操作数(权重在 params,持久不计);norm → 存输入;
  flash_attn → 存 q/k/v 输入 + 输出 + lse(flash 反向语义,比教科书多 O/lse);
  elementwise → **非线性**(swiglu/gelu/mul/sigmoid/silu)存输入;线性(add/cat/expand/collapse/embedding 查表)不存;
  nll → 特判(存 logsm,fp32 物化走 bwd_scratch);rope → 不存(线性旋转,cos/sin 是常数表);
  moe_router → 存输出 logits(softmax);dispatch/combine → 不存(重排,反向是逆重排)。

用法: PYTHONPATH=. python tools/audit_saves.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from cost_eval.presets import deepseek_v3
from cost_eval.build_llm import build_llm_spec

# 线性 elementwise（反向不需输入）名单;其余 elementwise 视为非线性（需存输入）。
_LINEAR_EW = ("add", "cat", "expand", "collapse", "embedding", "scale", "residual")
_NONLINEAR_EW = ("swiglu", "gelu", "silu", "mul", "sigmoid", "act")


def expected_saves(op):
    """按反向依赖规则推导该 op 应存的激活名集合。返回 (set, note)。"""
    t = op.type.value if hasattr(op.type, "value") else op.type
    ins = [x.name for x in op.inputs if not getattr(x, "is_weight", False)]
    outn = op.output.name
    if t in ("matmul", "moe_gemm"):
        return set(ins), "matmul 存激活操作数"
    if t == "norm":
        return set(ins), "norm 存输入"
    if t == "flash_attn":
        return set(ins) | {outn}, "flash 存 qkv+输出(+lse 另计)"
    if t == "rope":
        return set(), "rope 线性(cos/sin 常数)"
    if t == "moe_router":
        return {outn}, "softmax 存输出"
    if t in ("dispatch", "combine"):
        return set(), "重排/通信,反向=逆重排"
    if t == "elementwise":
        name = op.name.lower()
        if op.name == "nll":
            return None, "nll 特判(logsm+bwd_scratch,已真机校准)"
        if any(k in name for k in _NONLINEAR_EW):
            return set(ins), "非线性 elementwise 存输入"
        if any(k in name for k in _LINEAR_EW) or not ins:
            return set(), "线性 elementwise 不存"
        return None, f"elementwise {op.name!r} 线/非线性未知——人工判"
    return None, f"未知类型 {t}"


def audit(cfg, label):
    spec = build_llm_spec(cfg)
    print(f"\n{'='*90}\n== {label} ==")
    seen_types = set()
    n_ok = n_miss = n_extra = n_manual = 0
    for ltype in spec.layer_pattern:
        if ltype in seen_types:
            continue
        seen_types.add(ltype)
        lspec = spec.get_layer(ltype)
        print(f"\n-- layer type: {ltype} --")
        for op in lspec.ops:
            exp, note = expected_saves(op)
            got = {s.name for s in op.saves}
            # lse 是 flash 的元信息张量(小),从比较里剔除
            got_cmp = {g for g in got if g != "lse"}
            if exp is None:
                print(f"  ?  {op.name:<16} declared={sorted(got)}  [{note}]")
                n_manual += 1
                continue
            miss = exp - got_cmp
            extra = got_cmp - exp
            if not miss and not extra:
                n_ok += 1
                continue
            flag = []
            if miss:
                flag.append(f"MISS(漏存)={sorted(miss)}")
                n_miss += 1
            if extra:
                flag.append(f"EXTRA(多存)={sorted(extra)}")
                n_extra += 1
            print(f"  ✗  {op.name:<16} declared={sorted(got)} expected={sorted(exp)}  {' '.join(flag)}  [{note}]")
    print(f"\n汇总[{label}]: OK={n_ok}  漏存={n_miss}  多存={n_extra}  需人工判={n_manual}")


if __name__ == "__main__":
    audit(deepseek_v3(4), "DSv3 (MLA + MoE + shared)")
    import dataclasses
    gqa = dataclasses.replace(deepseek_v3(4), attn_type="gqa", num_query_groups=4,
                              first_k_dense_replace=4, num_moe_experts=None)
    audit(gqa, "GQA 纯 dense")
