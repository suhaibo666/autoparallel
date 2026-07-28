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

    # ── 路线 B P0#4/#5 + P1#11 新增的 op 类型(2026-07-25)。每条的判据 = 教科书 VJP,
    #    定位符见 `primitives.py` 的表项注释。────────────────────────────────────────────
    # Compare:比较 / 逻辑 / isfinite —— **不可微**(bool 产出),反向什么都不读。
    #   `future = cm >= positions // ratio`(csa.py:779)、
    #   `valid = topk_indices_compressed < ...`(csa.py:810)—— 两个 O(S·S/r) bool mask。
    "Compare":       {"inputs": []},
    # Constant:arange / zeros / ones / full / *_like / RoPE 频率表 —— 常量产出,无梯度。
    "Constant":      {"inputs": []},
    # Detach:`ops.stop_gradient(...)` —— 梯度到此为止,反向不读任何输入(但**边仍在**,
    #   见 schema.OpDAG.detached 的说明:节点存在 = 数据流可追溯)。
    "Detach":        {"inputs": []},
    # Where(cond, a, b):反向按 cond 把 dy 分流到 a/b → **必须存 cond**(ins[0])。
    "Where":         {"inputs": [0]},
    # Scatter(input, dim, index, src):反向对 input 是「index 处置零」、对 src 是 gather
    #   → 存 **index**。`ins` 只收可追踪张量操作数(dim/常量 src 被 _emit 略过)→ index = ins[1]。
    "Scatter":       {"inputs": [1]},
    # IndexSelect = `mint.gather(input, dim, index)` / advanced indexing `x[idx]`:
    #   反向 = dy scatter_add 回零张量的 index 位置 → 存 **index**(ins[1])。
    #   csa.py:485 `kv_flat[flat_indices]`、indexer.py:390 `mint.gather(...)`。
    "IndexSelect":   {"inputs": [1]},
    # TopK:反向把 dy scatter 到被选中位置 → 存**自己的第 2 个输出(indices)**,不是输入。
    #   `topk_scores, topk_indices = self.topk(index_scores, ...)`(indexer.py:262)。
    "TopK":          {"outputs": [1]},
    # FusedFunction = mindspore `_Function.apply(...)`:saved 集**逐字来自源**
    #   (`ctx.save_for_backward(...)`),挂在 attrs 上,**绝不由本表猜**。见 derive_saves 的分支。
    "FusedFunction": {"from_attrs": True},
    # Kernel = 白名单里的融合 NPU 内核自由函数(`npu_lightning_indexer` 等):它的 saved 集
    #   同样**不由本表猜**。只有 walker 能**证明**该调用无反向(在 `_no_grad()` 区里)时才会
    #   写 `attrs["saved_ins_idx"]=[]`;否则缺该 attr → 下面 fail-loud。
    "Kernel":        {"from_attrs": True},
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
        elif spec.get("from_attrs"):
            if "saved_ins_idx" not in n.attrs:
                raise ValueError(
                    f"op '{n.op}' @ {n.src} 的 saved 集必须**来自源**"
                    f"(`ctx.save_for_backward(...)` 或「可证明无反向」),但节点上没有 "
                    f"attrs['saved_ins_idx'] —— 拒绝当成「无 saved 集」静默放行(fail-loud)"
                )
            # `FusedFunction`:saved 集来自源里 `ctx.save_for_backward(...)` 的逐字名单
            # (`attrs["saved_ins_idx"]` = 能定位到 ins 的那些项)。`attrs["saved_internal"]`
            # 里是 forward **内部**张量(如 `output`/`softmax_lse`),没有 ins 对应项 ——
            # 它们的字节解析属 P2,此处只保证不被静默当成"无 saved 集"。
            idxs = [i for i in (n.attrs.get("saved_ins_idx") or ()) if i < len(n.ins)]
        elif spec.get("inputs") == "all":
            idxs = list(range(len(n.ins)))
        elif isinstance(spec.get("inputs"), list):
            # PIN 的下标按「**张量操作数**位序」写(`mint.gather(input, dim, index)` 的 `dim`
            # 是 int、不计位,故 index = `[1]`)。当某个前置张量操作数是**权重**时它被路由去
            # `param_operands`(W2/W3)、不进 `ins`,`ins` 位序随之左移 —— `attrs["ins_slots"]`
            # 记着每个存活 `ins` 项的张量操作数位序,有它就按它取,没有(位序恒等)就直接用,
            # 既有路径逐字不变。
            slots = n.attrs.get("ins_slots")
            if slots:
                pos = {slot: i for i, slot in enumerate(slots)}
                idxs = [pos[k] for k in spec["inputs"] if k in pos]
            else:
                idxs = spec["inputs"]
        out_idxs = list(spec["outputs"]) if isinstance(spec.get("outputs"), list) else []
        # 融合内核**保存自己输出**的情形:`npu_mhc_pre_sinkhorn` 的
        # `ctx.save_for_backward(x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms,
        #  sum_out, norm_out)`(custom_op_impl.py:390-391)—— 后 5 项是它**自己的输出**。
        # 源里调用点写 `h_in, h_post, h_res_flat, *_ = npu_mhc_pre_sinkhorn(...)`
        # (hyper_connection.py:413),`*_` 把它们丢了 —— 但 autograd ctx 仍持有引用,
        # 显存是**真实占用**的。Python 侧没名字 ≠ 不占显存。
        out_idxs += [k for k in (n.attrs.get("saved_outs_idx") or ()) if k not in out_idxs]
        if out_idxs:
            # 存**自己的第 k 个输出**(TopK 的 indices):多输出 ref 在 attrs["outs"] 里。
            outs = n.attrs.get("outs") or ([n.out] if n.out else [])
            declared_names = n.attrs.get("saved_out_names") or {}
            declared_shapes = n.attrs.get("saved_out_shapes") or {}
            for k in sorted(out_idxs):
                if k < len(outs) and outs[k].count(":") == 2:
                    name, shape, dtype = _parse(outs[k])
                    saves.setdefault(name, Save(name, dtype, n.id, shape))
                    continue
                # 该输出在图上**没有 ref**(源里被 `*_` 丢弃)→ 用声明里的名字/形状登记,
                # 形状不确定就留 `?`,由消费方计进 `unresolved`(**绝不**编一个数)。
                nm = declared_names.get(k) if isinstance(declared_names, dict) else None
                if nm is None:
                    continue
                sname = f"{nm}__k{n.id}"
                saves.setdefault(sname, Save(
                    sname,
                    n.out.split(":")[2] if n.out.count(":") == 2 else "bf16",
                    n.id, declared_shapes.get(nm, "?")))
        # **权重派生的操作数不是激活**(W2/W3/W4):`w1 = cast(self.weight1, ...)`(ffn.py:146)
        # 之后 `w1` 进 GroupedMatmul 的 ins,若照 `inputs:"all"` 计入 saves 就是把权重当激活
        # —— 实测 FFNGroupedGEMM「236 MiB」里的 88 MiB(评估文档 §7.2)。walker 在
        # `attrs["weight_ins_idx"]` 里标了这些下标(判据:该项的产出节点操作数全是权重)。
        wix = set(n.attrs.get("weight_ins_idx") or ())
        for i in idxs:
            if i >= len(n.ins) or i in wix:
                continue
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
