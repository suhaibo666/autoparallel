# cost_eval/opdag/bprop_rules.py
"""Stage2:per-op-类型 bprop-pin 规则(设计 §4)。唯一手维护件,但按 op 类型(非模型)、
教科书自动微分事实、有 profiler 兜底 → 稳定。derive_saves(dag) 遍历节点,按类型判 pin 谁。

关键:Cast 自身不 save,但其输出被下游非线性/matmul 消费者 pin(fp32 buffer 归消费者、按 fp32 计)。
实现:先对每节点按其"消费的输入"判 pin(消费者存输入),故 cast 的 fp32 输出自然被下游 norm/matmul 收进 save。"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Save:
    name: str
    dtype: str
    op_id: int          # 由哪个消费者 pin
    sym_shape: str

def _parse(ref: str):
    # "name:S·B·H:dtype" → (name, shape, dtype)
    name, shape, dtype = ref.split(":")
    return name, shape, dtype

# 每个 op 类型:它的反向 pin 哪些"输入位序"(操作数),以及是否 pin 输出。
# 'inputs':[idx...] 存这些输入; 'output':True 存输出; 空 = 不 pin(线性/视图)。
PIN = {
    "MatMul":        {"inputs": "all"},              # d a=dy·bᵀ, d b=aᵀ·dy
    "BMM":           {"inputs": "all"},
    "GroupedMatMul": {"inputs": "all"},
    "Norm":          {"inputs": [0]},                # 归一化统计需 input(fp32)
    "Softmax":       {"output": True},
    "Activation":    {"inputs": [0]},                # d x=dy·f'(x)
    "FlashAttention":{"inputs": [0, 1, 2]},          # q,k,v(+lse 由 kernel,略)
    "Gather":        {"inputs": [0]},                # 索引(小)
    "Dropout":       {"inputs": []},                 # mask 由 attrs 另计(1B),此处不按输入
    "Cast":          {"inputs": []},                 # 自身不存;输出由下游 pin
    "Elementwise":   {},                             # 见下:linear→无;非线性 mul→两操作数
    "View":          {"inputs": []},                 # reshape/transpose 元信息
    # IdentityOp:恒等映射,反向 dx=dy 直通,不需存任何激活。
    # 它**会**真出现在 DAG 里:`qk_layernorm=False` 时 spec 把 q_layernorm/k_layernorm 解成
    # `Identity`(module_resolver._NAME_ALIAS / LEAF_OPTYPE),walker 照常发射 Identity 节点。
    # 缺此表项时 derive_saves 在 :44 fail-loud(实测 DSv3 MLA + qk_layernorm=False 即触发)。
    "Identity":      {"inputs": []},
}

def derive_saves(dag) -> list[Save]:
    saves: dict[str, Save] = {}          # dedup by name
    for n in dag.nodes:
        spec = PIN.get(n.op)
        if spec is None:
            raise ValueError(f"未知 op 类型 '{n.op}' @ {n.src}：bprop 规则未覆盖(fail-loud)")
        idxs = []
        if n.op == "Elementwise":
            if not n.attrs.get("linear", False):     # 非线性(mul/gate)存两操作数;linear(add)不存
                idxs = list(range(len(n.ins)))
        elif spec.get("inputs") == "all":
            idxs = list(range(len(n.ins)))
        elif isinstance(spec.get("inputs"), list):
            idxs = spec["inputs"]
        for i in idxs:
            name, shape, dtype = _parse(n.ins[i])
            if n.op == "Norm":
                # fp32-残差机制:layernorm_compute_dtype=fp32 时归一化在 fp32 计算并存 fp32 输入,
                # 覆盖操作数 ref 里传播来的 compute_dtype(bf16)。无该 attr 则用 ref dtype(回归安全)。
                dtype = n.attrs.get("ln_compute_dtype", dtype)
            saves.setdefault(name, Save(name, dtype, n.id, shape))
        if spec.get("output"):
            name, shape, dtype = _parse(n.out)
            saves.setdefault(name, Save(name, dtype, n.id, shape))
    return list(saves.values())
