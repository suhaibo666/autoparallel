# M4 / M5 / M6 核心模块内部逻辑设计

> 日期: 2026-06-29 · 状态: Draft（待评审）
> 父: [[2026-06-29-evaluator-implementation-design]] · [[2026-06-29-p0-modelspec-and-memory-design]]
> 展开评估器三个核心模块的内部算法、伪代码、关键正确性点。

---

## M4 `shape_eval` — 符号求值 + 切分代入 + reshard 检测

**目标**：ModelSpec + ParallelModel → ResolvedGraph（每 stage 每层每 op 带 local shape + 内存契约字节 + 派生通信）。

### M4.1 主流程

```python
def resolve(spec, pm) -> ResolvedGraph:
    stages = defaultdict(list)
    for layer_id, ltype in enumerate(spec.layer_pattern):
        stage = pm.stage_of(layer_id)                      # PP: layers_per_stage
        lspec = get_layer_spec(ltype, spec.dims)           # M1 注册表
        rops, last_placement = [], {}                      # tid -> placement(生产端)
        for op in lspec.ops:
            # 1) 解析 + 切分输入/输出/params/saves
            r_in  = [resolve_tensor(t, spec.dims, pm) for t in op.inputs]
            r_out = resolve_tensor(op.output, spec.dims, pm)
            r_par = [resolve_tensor(t, spec.dims, pm) for t in op.params]
            r_sav = [resolve_tensor(t, spec.dims, pm) for t in op.saves]
            ws    = eval_expr(op.workspace, spec.dims, pm) if op.workspace else 0
            # 2) reshard 检测：本 op 各输入的"期望 placement" vs 其生产端 placement
            comms = []
            for t in op.inputs:
                src = last_placement.get(t.name)
                c = detect_reshard(src, placement_of(t), pm)
                if c: comms.append(c)
            rops.append(ResolvedOp(op.name, op.type, r_in, r_out,
                                   r_par, r_sav, ws, comms))
            last_placement[op.output.name] = placement_of(op.output)
        stages[stage].append(ResolvedLayer(layer_id, ltype, rops))
    return ResolvedGraph(stages)
```

### M4.2 符号 shape 求值 + 切分代入

```python
def resolve_tensor(t: TensorRef, dims, pm) -> ResolvedTensor:
    sizes = [eval_expr(d, dims) for d in t.shape]       # 符号 → 全局 int
    for dim_idx, axis in t.shard.items():               # 切分代入
        deg = pm.degree(axis)
        if sizes[dim_idx] % deg != 0:
            raise InvalidConfig(f"{t.name} dim{dim_idx}={sizes[dim_idx]} 不被 {axis}={deg} 整除")
        sizes[dim_idx] //= deg
    return ResolvedTensor(t.name, prod(sizes), dims.dtype_bytes, t.is_weight)
```
- `eval_expr`：对 `"(n_heads+2*n_kv)*head_dim"` 这类表达式，在 dims 符号字典上做**受限算术求值**（仅 + - * // 与 dim 名，无任意代码）。
- 注意：M4 只做**图内切分（tp/cp/ep/sp）**；外层 FSDP/DP/PP 不在此（M5/M6 施加）。

### M4.3 placement 代数 + reshard 检测（派生通信）

placement = 每张量维可选 `Shard(axis)` + 可选 `Partial(axis)`（未规约部分和）+ 缺省 `Replicate`。相邻 op 对同一 tid 的生产/消费 placement 不一致 → 派生集合通信：

| 生产端 → 消费端 | 集合通信 | 典型来源 |
|---|---|---|
| `Partial(a)` → `Replicate` | **all-reduce(a)** | 行并行收尾（无 SP） |
| `Partial(a)` → `Shard(d,a)` | **reduce-scatter(a)** | 行并行 + SP |
| `Shard(d,a)` → `Replicate` | **all-gather(a)** | SP→非SP 进注意力 |
| `Shard(d1,a)` → `Shard(d2,a)` | **all-to-all(a)** | EP dispatch / CP ulysses |
| `Replicate` → `Shard(d,a)` | 本地切片（无通信） | 进 SP 区 |

```python
def detect_reshard(src, dst, pm) -> CommSpec | None:
    if src is None or src == dst: return None
    a = src.partial_axis or dst.shard_axis
    ctype = RESHARD_TABLE[(kind(src), kind(dst))]       # 上表
    vol   = numel(dst) * dtype_bytes                    # P0 只记 volume，时间留 P1
    return CommSpec(ctype, vol, group_axis=a, phase="fwd")
```
> P0 记录 collectives 用于 M6 的 staging buffer 计账；其**时间**由 P1 的 M8/M9 计算。

### M4.4 边界与校验
- 整除校验（tp 须整除头数/ffn、cp 整除 seq、ep 整除专家数）→ 非法配置即报。
- SP 开关：`sequence_parallel=False` 时 `{sp:0}` 退化为不切（张量在 tp 上复制）。
- MoE：专家权重 placement = `{0:ep}`（纯 EP，见 `expert_parallel.py:330`），**不带 tp**。

---

## M5 `static_mem` — 持久 param/grad/opt

**目标**：每 stage 持久态字节。M4 已对 params 施加了**图内切分**（dense→/tp、expert→/ep）；M5 再施加 **FSDP 存储切分** + 优化器倍数。

### M5.1 关键正确性点（efsdp 不能重复算 tp）

权重持有"不同分片的设备数" = 图内切分轴 × 其 FSDP 组：

| 权重类 | 图内切分(M4 已做) | FSDP 组(M5 施加) | 每卡净除 |
|---|---|---|---|
| attn/dense | tp | `fsdp = dp_shard·cp` | `/(tp·dp_shard·cp)` |
| **expert** | ep | **`efsdp = dp_shard·cp·tp/ep`** | `/(ep·efsdp) = /(dp_shard·cp·tp)` |
| replicated(norm) | — | `fsdp = dp_shard·cp` | `/(dp_shard·cp)` |

> **陷阱**：专家**不要再 ×/tp**——tp 已在 `efsdp` 内（`parallel_dims.py`: `efsdp = fsdp·tp//ep`），专家不单独 TP 切（`expert_parallel.py:330` `weight:(Shard(0),)`）。若误把专家标 `{0:ep,2:tp}` 再除 efsdp 会多除一个 tp。

### M5.2 伪代码

```python
def compute(g, opt, pm, cpu_offload) -> dict[int, int]:
    out = {}
    for stage, layers in g.stages.items():
        numel = 0
        for layer in layers:
            for op in layer.ops:
                for w in op.params:                 # w.local_numel: M4 已 /tp 或 /ep
                    fsdp = pm.efsdp_degree() if is_expert(w) else pm.fsdp_degree()
                    numel += w.local_numel // fsdp
        out[stage] = 0 if cpu_offload else numel * opt.state_bytes_per_param
    return out
```
- `is_expert(w)`：该权重 placement 含 `ep`（来自 M4 标记）。
- `state_bytes_per_param`：AdamW=16（bf16 param2+grad2+master4+m4+v4）或 18（fp32 grad）；Muon 另算。
- `cpu_offload=True`：持久态下沉 CPU → 设备持久≈0，仅 M6 的 `gather_buf` 瞬时驻留。

### M5.3 自检
- 守恒：`Σ_stage Σ_device numel × 切分度数 == 全局参数量`。
- 退化：tp=cp=ep=dp_shard=1 → numel = 全局 stage 参数。
- 对 ZeRO：dense 7B / (tp·dp_shard·cp) × 16 应落在 ZeRO-3 已知量级。

---

## M6 `mem_timeline` — 事件驱动峰值仿真

**目标**：沿 1F1B 执行时间线 walk alloc/free 事件，`peak = O_framework + max_t Σ桶`，显式建模 recompute 反向尖峰 / FSDP 预取双缓冲 / swap 预取。

### M6.1 调度模型（1F1B）

每 stage `s`（0-indexed，共 PP 个）、`m` 个 microbatch（`m = global_batch / (dp · microbatch)`）：

```
warmup  : N_w = PP - 1 - s 个 FWD（只前向，累积激活）
steady  : m - N_w 个 (FWD, BWD) 对（1F1B 交替）
cooldown: N_w 个 BWD
在飞激活峰 ≈ (N_w + 1) 份 microbatch（stage 0 最深 ≈ PP）
interleave_num=v: 每 rank v 个 chunk，warmup 更深、单 chunk 激活更小
```
事件序列 = 上述展开为 `[FWD(mb, layer), BWD(mb, layer), ...]`（layer 在 stage 内顺序/逆序）。

### M6.2 桶状态机（7 桶）

```python
def simulate(g, recompute, swap, pc, static_persistent) -> dict[int, StagePeak]:
    res = {}
    for stage, layers in g.stages.items():
        B = Buckets(persistent=static_persistent[stage])   # M5
        peak, peak_ev = 0, None
        gathered = {}                # layer_id -> G_layer（FSDP 已 gather 的）
        pinned   = {}                # (mb, layer_id) -> 该层 post-recompute/swap 的 saves 字节
        for ev in build_1f1b(stage, pc, m=num_microbatches(pc)):
            if ev.kind == "FWD":
                _ensure_gather(gathered, ev.layer, pc, B)         # 见 M6.3①
                for op in layers[ev.layer].ops:
                    B.workspace = op.workspace_bytes              # 瞬时
                    peak, peak_ev = _rec(B, peak, peak_ev, f"fwd:{ev.layer}")
                    B.workspace = 0
                saved = _saves_after_recompute_swap(ev.layer, recompute, swap)
                pinned[(ev.mb, ev.layer)] = saved
                B.act_live += saved
                if pc.reshard_after_forward == "always":
                    _free_gather(gathered, ev.layer, B)
                peak, peak_ev = _rec(B, peak, peak_ev, "fwd_end")
            else:  # BWD
                _ensure_gather_bwd(gathered, ev.layer, pc, B)
                if _is_recomputed(ev.layer, recompute):           # 见 M6.3②
                    B.recomp_scratch = _full_saves(ev.layer)
                    peak, peak_ev = _rec(B, peak, peak_ev, f"bwd_recompute@{ev.layer}")
                    B.recomp_scratch = 0
                B.grad_buf = _grad_bytes(ev.layer)                # reduce-scatter 前
                peak, peak_ev = _rec(B, peak, peak_ev, f"bwd_grad@{ev.layer}")
                B.grad_buf = 0
                B.act_live -= pinned.pop((ev.mb, ev.layer))
                _free_gather(gathered, ev.layer, B)
        res[stage] = StagePeak(stage, peak + B.framework, breakdown_at(peak_ev),
                               peak_ev, oom=(peak + B.framework) > hw.max_device_memory)
    return res
```

### M6.3 三类峰值影响的精确规则

**① FSDP all-gather + 预取双缓冲**（`_ensure_gather`）
```
进入 layer i 计算前：gather 自身 + 预取后 d 层 → gathered = {i, i+1, ..., i+d}
G_layer = (层 dense 权重 local_numel /tp 已含) gather 成整 bf16 = layer_param_after_tp · 2
gather_buf = Σ_{l∈gathered} G_l
reshard_after_forward="always": FWD 用完即 _free_gather(i)，BWD 再 gather
                      ="never" : 不释放 → gathered 累积到整 stage（峰值大）
```

**② recompute 反向尖峰**（`_is_recomputed` + `_full_saves`）
- `full`（命中 `full_recompute_layer`）：FWD 时 `_saves_after_recompute_swap` 只留 checkpoint 输入（`act_live` 大降）；BWD 时 `recomp_scratch = _full_saves(layer)`（该层完整激活）瞬时叠加 → **常是全局峰值**。
- `select`/`exclude_op`：`_saves_after_recompute_swap` 按 `select_module`/`exclude_op` 只减/保留部分。

**③ swap 预取**（`_saves_after_recompute_swap` + swap_buf）
- offload 的 saves 从 `act_live` 扣；接近其 BWD 时 `swap_buf += default_prefetch · 单次预取字节`（H2D 预取窗口）。
- 三态互斥：每个 saves ∈ {resident(act_live) / recomputed(recomp_scratch) / offloaded(swap_buf)}，仿真器保证只进一个桶。

### M6.4 输出
`StagePeak{peak_bytes, breakdown(7桶+framework), peak_event, oom}`。`peak_event` 标出峰值落点（如 `"fwd_end"` 或 `"bwd_recompute@layer_40"`），使报告能解释**为什么 OOM 发生在反向**。

### M6.5 自检
- 无重算 + reshard=always：峰值应落在 `fwd_end`（在飞激活最满）。
- 开 full 重算：峰值落点应**迁移到** `bwd_recompute@*`，且总峰值下降（激活降 > recomp_scratch 升）。
- reshard `never` vs `always`：`gather_buf` 应显著不同。
- 单调性：`default_prefetch`↑ → `swap_buf`↑、暴露 stall↓（stall 属 P1 时间）。
