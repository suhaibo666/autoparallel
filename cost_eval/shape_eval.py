"""M4：符号 shape 求值 + 切分代入 + reshard 检测。"""
from __future__ import annotations
import ast
import operator
from dataclasses import dataclass
from math import prod
from .model_spec import DimTable, ModelSpec, TensorRef

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.floordiv,
}


def eval_expr(expr: str, dims: DimTable) -> int:
    """对 dim 符号做受限算术求值；仅允许 + - * // 与已知符号/整数。"""
    env = dims.as_dict()

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"未知符号: {node.id}")
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_ev(node.left), _ev(node.right))
        raise ValueError(f"非法表达式节点: {ast.dump(node)}")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"非法表达式语法: {expr!r}") from exc
    return int(_ev(tree))


def _refs_symbol(expr: str, sym: str) -> bool:
    """符号维/字符串表达式是否**引用**符号 `sym`（作为标识符 Name，而非裸子串匹配）。

    用于 context-parallel 识别「序列（token/query）维」：某维/某 workspace 表达式引用 `S` 即随
    序列长度缩放，CP 下应 ÷cp。走 AST Name 判定 → `"S//4"`/`"S*B*topk*capacity_factor"` 命中，
    `"kv_lora_rank"`/`"n_heads"`（含小写 s 但无 Name `S`）不误伤。非法/空表达式 → False。"""
    if not expr:
        return False
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Name) and n.id == sym for n in ast.walk(tree))


# ---------------------------------------------------------------------------
# Task 5 — ResolvedTensor + resolve_tensor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedTensor:
    name: str
    local_numel: int
    dtype_bytes: int
    is_weight: bool
    is_expert: bool = False
    # 重算免疫标志直通（TensorRef.pin_under_recompute，2026-07-22）：fused 自定义算子
    # ctx 保存集在全重算下不释放 → structure_mem.recompute_pinned_saves 按此聚合。
    pin_under_recompute: bool = False
    # TP/EP placement 后的**本地首维**（2026-07-23 P0-3，replicate_params 判定）：runtime FSDP
    # 按参数首维切（parallelize.py:331-350, commit 377c9c344），`shape[0] % shard_size != 0` →
    # 整参不做 FSDP（replicate_params，全量驻留）。structure_mem 以此替代 total-numel ceil。
    # 0 = 未知（旧构造/测试 stub → structure_mem 退回 numel 整除判定,兼容）。
    dim0: int = 0
    # grad 可达性直通（TensorRef.detached，2026-07-25）：stop_gradient 后的子计算产物 →
    # 无 autograd 节点 → 反向无节点读它。**只被 cost_eval/liveness 读**，桶路径逐字节不变。
    detached: bool = False


def resolve_tensor(t: TensorRef, dims: DimTable, pm) -> ResolvedTensor:
    """符号求值 → 按 shard 维整除轴度数 → 不整除抛 ValueError。"""
    sizes = [eval_expr(e, dims) for e in t.shape]
    for dim_idx, axis in t.shard.items():
        deg = pm.degree(axis)
        if sizes[dim_idx] % deg != 0:
            raise ValueError(
                f"{t.name} dim{dim_idx}={sizes[dim_idx]} 不被 {axis}={deg} 整除")
        sizes[dim_idx] //= deg
    # ── context-parallel（cp）：全局序列切分——每个**非权重激活**的 token/query 维 ÷cp（D-1）──
    # cp 是独立于 sp 的全局序列并行域（ring/context attention）：切 query（token）维，key/context
    # 维保持全量（每 rank 的 S/cp 个 query 仍见完整上下文）。故只切**首个引用符号 S 的维**（token
    # 维），其余含 S 的维（仅 dsv4 `TOPK_DIM`=window+S//ratio 这类上下文位置维）不切 → `break`。
    # 与 SP 组合：SP 张量已在上面 ÷sp（=tp），此处再 ÷cp → S/(tp·cp)；非 SP → S/cp。权重无 S
    # （已核）→ is_weight 跳过；参数分片仍走 fsdp_degree=dp_shard·cp（static_mem.py:8），不双算。
    #
    # ── D-1 修正（2026-07-07；P2-08 对齐 2026-07-14）：cp 下**保持 full-S** 的张量──────────────
    #   1. cp_shard=False：预留通道——**当前全库无张量使用**。早期把 loss/head 区列为此类的
    #      "cp=2 满 vocab full-S 实测"已被 Bug A 复核推翻（B=2·S/cp 误读为 B=1·full-S），
    #      loss/head 区现随 cp ÷cp（权威口径 head.py:59-64）。
    #   2. cp_kv=True 且 method==colossal：attention KV 侧激活——colossal（ulysses_degree=1）
    #      all-gather KV 到 full-S（额外 KV buffer）；其余算法（ulysses/ring/hybrid）KV 仍随 body ÷cp。
    cp = pm.degree("cp")
    method = getattr(pm.pc, "context_parallel_method", "colossal")
    cp_applies = (cp > 1 and not t.is_weight and t.cp_shard
                  and not (method == "colossal" and t.cp_kv))
    if cp_applies:
        for dim_idx, e in enumerate(t.shape):
            if _refs_symbol(e, "S"):
                if sizes[dim_idx] % cp != 0:
                    raise ValueError(
                        f"{t.name} 序列维 dim{dim_idx}={sizes[dim_idx]} 不被 cp={cp} 整除")
                sizes[dim_idx] //= cp
                break
    numel = prod(sizes) if sizes else 1
    dtype_bytes = t.dtype_bytes if t.dtype_bytes is not None else dims.dtype_bytes
    return ResolvedTensor(t.name, numel, dtype_bytes, t.is_weight, t.has_ep(),
                          getattr(t, "pin_under_recompute", False),
                          int(sizes[0]) if sizes else 1,
                          getattr(t, "detached", False))


# ---------------------------------------------------------------------------
# Task 6 — Placement algebra + CommSpec + detect_reshard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    """Placement 描述张量在通信轴上的分布状态。

    shard: tuple of (dim_index, axis) pairs（内部存为有序 tuple，可哈希）。
    构造时 shard 入参可接受 dict 或 tuple；partial 为未规约轴名或 None。
    """
    shard: tuple = ()
    partial: str = None

    def __init__(self, shard=(), partial=None):
        items = tuple(sorted(shard.items())) if isinstance(shard, dict) else tuple(shard)
        object.__setattr__(self, "shard", items)
        object.__setattr__(self, "partial", partial)

    @staticmethod
    def of(t: TensorRef) -> "Placement":
        return Placement(t.shard, t.partial)


@dataclass(frozen=True)
class CommSpec:
    ctype: str          # all_reduce | reduce_scatter | all_gather | all_to_all
    volume_bytes: int
    group_axis: str
    phase: str = "fwd"


def detect_reshard(src: Placement, dst: Placement,
                   numel: int, dtype_bytes: int):
    """推导两个 placement 之间的 collective；相等或 src=None 返回 None。"""
    if src is None or src == dst:
        return None
    # 推断 collective 轴
    if src.partial:
        axis = src.partial
    elif src.shard:
        axis = src.shard[0][1]
    elif dst.shard:
        axis = dst.shard[0][1]
    else:
        return None
    if src.partial and not dst.partial and not dst.shard:
        ctype = "all_reduce"
    elif src.partial and dst.shard:
        ctype = "reduce_scatter"
    elif src.shard and not dst.shard and not dst.partial:
        ctype = "all_gather"
    elif src.shard and dst.shard:
        ctype = "all_to_all"
    else:
        return None
    return CommSpec(ctype, numel * dtype_bytes, axis)


# ---------------------------------------------------------------------------
# Task 7 — ResolvedOp / ResolvedLayer / ResolvedGraph + ShapeEval
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedOp:
    name: str
    type: str
    inputs: tuple
    output: ResolvedTensor
    params: tuple
    saves: tuple
    workspace_bytes: int
    collectives: tuple
    bwd_scratch_bytes: int = 0


@dataclass(frozen=True)
class ResolvedLayer:
    layer_id: int
    layer_type: str
    ops: tuple


@dataclass(frozen=True)
class ResolvedGraph:
    stages: dict          # int -> list[ResolvedLayer]


class ShapeEval:
    def resolve(self, spec: ModelSpec, pm) -> ResolvedGraph:
        """遍历 layer_pattern → stage_of 分组；逐 op 解析 inputs/output/params/saves，
        算 workspace，按相邻 op 的 placement 不匹配派生 collectives。"""
        stages: dict = {}
        for layer_id, ltype in enumerate(spec.layer_pattern):
            stage = pm.stage_of(layer_id)
            lspec = spec.get_layer(ltype)
            r_ops = []
            produced: dict = {}          # tensor name -> Placement of producer
            for op in lspec.ops:
                r_in = tuple(resolve_tensor(t, spec.dims, pm) for t in op.inputs)
                r_out = resolve_tensor(op.output, spec.dims, pm)
                r_par = tuple(resolve_tensor(t, spec.dims, pm) for t in op.params)
                r_sav = tuple(resolve_tensor(t, spec.dims, pm) for t in op.saves)
                ws = eval_expr(op.workspace, spec.dims) if op.workspace else 0
                bws = eval_expr(op.bwd_scratch, spec.dims) if op.bwd_scratch else 0
                # context-parallel（cp）：含符号 S 的 workspace/bwd_scratch 也 ÷cp 一次（D-1）——
                # flash-ws(∝S)/loss(∝S)/MoE-staging(∝S)/mHC(∝S) → S/cp；index_scores `4·B·S·S`
                # → S²/cp（去掉 query 那个 S 因子，与张量口径一致：切 query 维、key 维保持全量）。
                # 每个带 S-workspace 的 op 必有含 S 的激活张量（flash 有 qkv、nll 有 logsm…），
                # 故非法 S%cp≠0 已在上面 resolve_tensor 先 raise → 此处合法配置下整除，floordiv 安全。
                #
                # ── cp_shard=False 的 op 保持 full-S 门（P2-08 对齐 2026-07-14）：预留通道——
                # loss/head 区已按 Bug A 复核恢复 cp_shard=True（÷cp，head.py:59-64），全库当前无
                # cp_shard=False 张量 → 此门对现有图空转；保留供未来真 full-S 语义 op 使用。
                cp = pm.degree("cp")
                method = getattr(pm.pc, "context_parallel_method", "colossal")
                if cp > 1 and op.output.cp_shard:
                    if _refs_symbol(op.workspace, "S"):
                        ws //= cp
                    if _refs_symbol(op.bwd_scratch, "S"):
                        bws //= cp
                # TensorRef 型 workspace / bwd_scratch（P0-05/P0-04）：走 resolve_tensor 的
                # shard（÷tp/ep）+ cp 机制（首个 S 维 ÷cp）求 local 字节，解决字符串表达式
                # 不随 tp 切分的表达缺口。
                if getattr(op, "workspace_ref", None) is not None:
                    wr = resolve_tensor(op.workspace_ref, spec.dims, pm)
                    ws += wr.local_numel * wr.dtype_bytes
                if getattr(op, "bwd_scratch_ref", None) is not None:
                    br = resolve_tensor(op.bwd_scratch_ref, spec.dims, pm)
                    bws += br.local_numel * br.dtype_bytes
                # ── P1-13（Y1，2026-07-15）：method+cp 门控 workspace —— GQA fused-qkv 的 colossal
                # CP KV all-gather full-S buffer ─────────────────────────────────────────────────
                # MLA 靠**独立 KV 激活**标 `cp_kv=True` 表达 colossal 下 KV all-gather 到 full-S；GQA 的
                # KV **融合**在 `qkv`（末维 (n_heads+2·n_kv)·head_dim）里，无法从融合张量单独标 cp_kv →
                # 需一条**只在 colossal cp>1 生效、否则 0**的 workspace 通道（TensorRef 的 cp_kv 做不到：
                # 它让「已存在的激活」在 colossal 保持 full-S，而非凭空「只在 cp>1 出现」一个 buffer，
                # 且 cp=1 时它是 full-S≠0，破惰性）。故走 OpSpec.attrs 携带 full-S 字节表达式、在此按门加进
                # workspace_bytes（builder 无条件挂串、shape_eval 门控——**estimate_structure_memory 口径
                # 无侵入**：它照常读 ResolvedOp.workspace_bytes 的 max）。
                #
                # **三重门**（缺一则贡献 0，惰性）：
                #   ① `method=="colossal" 且 cp>1`：colossal（ulysses_degree=1）才 all-gather KV 到
                #      full-S；ring/ulysses/hybrid 是 CP 通信在飞的**双缓冲**（KV 块 send/recv，量级
                #      ~2·(2·n_kv·head_dim)·(S/cp)·B）——随 body ÷cp、**非** all-gather buffer，故此门挡掉
                #      （ring/ulysses 双缓冲尚未建，见交付报告「未尽事项」）。
                #   ② op.attrs 带 "colossal_kv_ws"（builder 无条件挂在 GQA flash op；MLA flash op 不挂）。
                #   ③ `spec.dims.cp_kv_allgather_buffer`（modeling opt-in，**默认 False**）：本量 off loss
                #      峰、本栈跑不了 cp+无重算故**未真机验证**；且 12 golden 锚点走 mla/dsv4 不经 GQA
                #      builder。默认关 → golden 逐字节不变 + 守 X3 冻结不变量（未 opt-in 的 GQA colossal
                #      cp2 仍精确减半）；开启即按公式计入。
                # full-S 值（**不 ÷cp**，因 all-gather 到 full-S）= 2·n_kv·head_dim·S·B·dtype（X3 固化公式，
                # 占 fused qkv 的 (2·n_kv·head_dim)/((n_heads+2·n_kv)·head_dim) 比例）；与 flash 已有
                # fa_ws 共存**相加**（同 op workspace_bytes，物理上 gathered KV 须在 flash 计算期驻留）。
                # 注：此串按全 n_kv 不 ÷tp（对齐 X3 固化公式；本量 cp2 锚点 tp=1 时精确，tp>1 为保守上界）。
                ckv_ws = op.attrs.get("colossal_kv_ws") if getattr(op, "attrs", None) else None
                if (ckv_ws and method == "colossal" and cp > 1
                        and getattr(spec.dims, "cp_kv_allgather_buffer", False)):
                    ws += eval_expr(ckv_ws, spec.dims)
                comms = []
                for t in op.inputs:
                    src = produced.get(t.name)
                    c = detect_reshard(
                        src,
                        Placement.of(t),
                        resolve_tensor(t, spec.dims, pm).local_numel,
                        spec.dims.dtype_bytes,
                    )
                    if c:
                        comms.append(c)
                r_ops.append(ResolvedOp(
                    op.name, op.type.value,
                    r_in, r_out, r_par, r_sav,
                    ws, tuple(comms), bws,
                ))
                produced[op.output.name] = Placement.of(op.output)
            # P1-03 不变量（2026-07-14 fail-loud）：同一 resolved layer 内张量名 → local_numel
            # 必须唯一——structure_mem/act_live 的按名去重把同名异 size 首见覆盖、静默错算
            # （mHC/MTP 的 `x` 曾同时 =H 与 =nH）。发现即报错防回潮。
            sizes_by_name: dict = {}
            for rop in r_ops:
                for rt in (*rop.inputs, rop.output, *rop.params, *rop.saves):
                    prev = sizes_by_name.setdefault(rt.name, rt.local_numel)
                    if prev != rt.local_numel:
                        raise ValueError(
                            f"layer {layer_id}({ltype}) 张量 {rt.name!r} 出现两种 numel "
                            f"{prev} != {rt.local_numel}（同名异形会被按名去重静默错算，"
                            "P1-03——请在 builder 侧重命名消歧）")
            stages.setdefault(stage, []).append(
                ResolvedLayer(layer_id, ltype, tuple(r_ops))
            )
        return ResolvedGraph(stages)
