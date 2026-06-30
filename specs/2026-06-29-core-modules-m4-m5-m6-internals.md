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
- `state_bytes_per_param`：**持久 = param+opt（剔除 grad，真机修正）**——bf16 params ~14、fp32 params(DSv3) 12；grad 是反向瞬态(`grad_buf`，§M6.3①)。Muon 另算。详见 §P0 6.3/8.4。
- `cpu_offload=True`：持久态下沉 CPU → 设备持久≈0，仅 M6 的 `gather_buf` 瞬时驻留。

### M5.3 自检
- 守恒：`Σ_stage Σ_device numel × 切分度数 == 全局参数量`。
- 退化：tp=cp=ep=dp_shard=1 → numel = 全局 stage 参数。
- 对 ZeRO：dense 7B / (tp·dp_shard·cp) × 16 应落在 ZeRO-3 已知量级。

---

## M6 `mem_timeline` — 事件驱动峰值仿真

**目标**：沿 1F1B 执行时间线 walk alloc/free 事件，`peak = max_t Σ桶 + framework_reserve(config)`，显式建模 recompute 反向尖峰 / FSDP gather+full grad / loss 区 fp32 / swap（BWD 共存采峰）。

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
            else:  # BWD（逆序层）—— gather + full grad + recompute + bwd_scratch 共存采峰
                B.gather_buf = _layer_param_bytes(ev.layer)           # FSDP re-gather 整层(compute dtype)
                B.grad_buf = _layer_grad_bytes(ev.layer, grad_dtype)  # full grad(grad dtype, 可 fp32)
                if _is_recomputed(ev.layer, recompute):
                    B.recomp_scratch = _recomp_workingset(ev.layer)   # §M6.3② 修正
                B.bwd_scratch = _layer_bwd_scratch(ev.layer)          # op 反向物化(loss probs)
                peak, peak_ev = _rec(B, peak, peak_ev, f"bwd@{ev.layer}")   # 四者一次采样
                B.gather_buf = B.grad_buf = B.recomp_scratch = B.bwd_scratch = 0
                B.act_live -= pinned.pop((ev.mb, ev.layer))
        fr = framework_reserve(pc)                                    # §M6.4, 按配置分解
        res[stage] = StagePeak(stage, peak + fr, breakdown_at(peak_ev),
                               peak_ev, oom=(peak + fr) > hw.max_device_memory)
    return res
```

### M6.3 反向桶的精确规则（2026-06-30 真机修订）

**关键修正：BWD 一层的 `gather_buf` / `grad_buf` / `recomp_scratch` / `bwd_scratch` 必须共存采峰**（旧版分开采样、且先释放 `act_live` 再算 grad → 漏算反向叠加，导致峰值低估约 2×）。

**① FSDP gather / grad**：`gather_buf=整层权重(compute dtype)`、`grad_buf=full grad(grad dtype，可 fp32)`；`reshard_after_forward` 控 FWD 是否即释（"never" 累积，峰高）。**grad 是反向瞬态、不入持久**（§8.4）。

**② recompute（三处修正，⚠ 未被真机验证，见 §8.5）**：
- `recomp_scratch = 该层 saves − checkpoint 输入`（**去双算**——输入已在 `act_live`）。
- 严格应为 `_recomp_workingset` = 该层 forward 的 **max-live**（非 saves 之和；对单层做 mini-forward 时间线求峰；MLA/MoE 多中间量层尤其）。
- **验证缺口**：DSv3 峰值落在 loss 层，`recomp_scratch=0`，这条没被踩到 → 须用"重算 transformer 层成峰值"的配置（大 hidden / 小 vocab / PP 增在飞 microbatch）单独验证。

**③ bwd_scratch（大 vocab fp32 loss 区）**：lm_head/loss op `probs=4·S·B·vocab`；配合 per-tensor dtype 的 fp32 `log_softmax`(saved)。大 vocab 时是峰值大头（对照 `loss.py`）。

**④ swap**：offload 的 saves 从 `act_live` 扣 + `swap_buf` 预取窗口；三态互斥。

### M6.4 输出 + `framework_reserve(config)`（按配置分解，非固定常数）

`StagePeak{peak_bytes, breakdown(8桶+framework), peak_event, oom}`。
```
framework_reserve(pc) = hccl(200MB × 通信组数(pc)) + moe_comm(op图 a2a 量) + flash_ws(seq×heads) + frag(平台小常数)
```
随并行配置自动缩放（§8.6）；仅 `frag` 留作每平台标定。`peak_event` 典型 `"bwd@loss"` / `"bwd@layer_i"`。

### M6.5 自检 + 真机验证现状

- full 重算降低峰值、峰值落 `bwd@*`（gather/grad 共存使反向 > fwd_end）。
- reshard never vs always → `gather_buf` 不同。
- **真机对标（§8.7）**：DSv3 4L **1.000** / 8L **0.996**（已验证 persistent<1% / loss区 fp32 / gather / grad / 跨层数缩放）。
- **逐桶验证原则**：每桶须设计"让它主导峰值"的配置单独验证。**开放项**：`recomp_scratch`（§M6.3②）、`framework_reserve` 分解（§M6.4，仅 1 个并行配置未跨配置标定）、swap / select-recompute。
