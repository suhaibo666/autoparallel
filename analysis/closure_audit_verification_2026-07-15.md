# `closure_audit_response_2026-07-15.md` 可信度复核报告

> 复核日期：2026-07-15  
> 当前代码：`feat/unified-llm-modelspec` @ `68f6cd2`  
> 被核主张：`analysis/closure_audit_response_2026-07-15.md`  
> 修复提交：`5c7fdd3`、`e8eaeb7`；当前 HEAD 另含不属于该答复主体的 UI 提交 `68f6cd2`  
> 判据：沿用上一轮审计口径——原问题的功能语义、输入域和对外结果均落实且有有效回归才算闭环；只覆盖一个反例、只 fail-loud、只改文档或只在核心对象新增但未传到调用方，记为部分闭环。

## 1. 结论

这份修复答复**部分可信，但“20 项闭环”明显过度**。

- `493 passed` 属实；新增定向测试也全部通过。
- FA workspace 按 TP 缩放、VPP timeline 的 chunk 身份、reshard 枚举校验等修复真实生效。
- 12 个已有真机锚点在当前代码上的重放结果为平均 `|ratio-1| = 2.0201%`，答复写的 **2.0% 属实**；这只是用已保存真机值重算当前评估器，不是本轮重新采集 NPU 数据。
- 原审查的 9 个开放建模项仍如实开放，这部分答复诚实。
- 但是，被答复列为闭环的 20 项里有 **10 项仍只能判为部分闭环**：P0-02、P1-01/02/06/10/11/16、P2-01/07/08。

严格按原问题边界重算状态：

| 状态 | 本次复核 | 答复自报 | 项目 |
|---|---:|---:|---|
| 闭环 | **10** | 20 | P0-01/03/04/05，P1-03/04/05/09/18，P2-06 |
| 部分闭环 | **13** | 3 | P0-02，P1-01/02/06/10/11/16/17/19，P2-01/03/07/08 |
| 开放 | **9** | 9 | P1-07/08/12/13/14/15，P2-02/04/05 |

没有发现原审查的整项问题是假阳性；发现的是这份最新答复中的 **10 个“假闭环/过度归并”**。

## 2. 实际执行的验证

| 验证 | 结果 | 能证明什么 |
|---|---|---|
| `python -m pytest -q tests/test_closure_audit_2026_07_14.py tests/test_param_conservation.py -vv` | **20 passed** | 答复新增的 17 个闭环用例及 3 个参数用例均绿 |
| `python -m pytest -q` | **493 passed in 4.38s** | 当前正式回归无失败，答复基线数字真实 |
| `python -m compileall -q cost_eval serve_explorer.py` | 通过 | Python 文件可编译 |
| `python analysis/review_closure_probe_2026-07-14.py` | 在第一个已修反例 `reshard='nevver'` 处按预期抛 `ValueError` 后退出 | 旧探针已不能作为批量复核器；它没有逐项捕获异常 |
| `python analysis/closure_audit_verification_probe_2026-07-15.py` | 运行通过并复现下述边界反例 | 穿透新增测试未覆盖的输入域，并验证正向修复 |
| 12 锚点本地重放 | 平均绝对相对误差 **2.0201%**，**9/12** 在 ±5% | 答复的“2.0%”可信；不是新 NPU 采样 |

独立探针是本次新增的只读验证脚本，不修改生产实现。其关键输出如下：

```text
FA workspace bytes: TP1=2097152, TP2=1048576, TP8=262144
recompute full_layers={999}: accepted, peak == no-recompute
recompute select_ops={}: accepted, peak == no-recompute
recompute select_ops={999:{missing}}: accepted, peak == no-recompute
o_groups=-1: accepted; o_group_out=-4194304, o_w=-1835008 numel
attention_dropout=-0.1: accepted
parallelism.pipeline_parallel_overlap_p2p=True: accepted
optimizer key/segment typo: accepted and silently defaults to AdamW
interleave=0 / prefetch_depth=-1 / num_microbatches=0: all accepted and return peaks
router dtype 4B→2B: “exact” numel roster unchanged, actual byte roster changed
shared_gate outputs: all have zero consumers
VPP (PP,V,m)=(2,2,4)/(2,3,7)/(3,2,8): every stage (event,mb,chunk) unique
```

## 3. 已确认真实闭环或真实生效的修复

### 3.1 P0-03：reshard 枚举校验

`ParallelConfig.__post_init__` 已在核心配置入口拒绝非 `always|never|default` 的值（`cost_eval/specs.py:38-50`）。正式用例 `tests/test_closure_audit_2026_07_14.py:28-30` 和旧探针的提前失败共同证明 `nevver` 不再回落到 default。

### 3.2 P1-09：FlashAttention workspace 的 TP 缩放

workspace 已由不带并行轴的字符串改为 `TensorRef("fa_ws", ..., shard={2: "tp"})`（`cost_eval/layers/attention.py:39-45`），并通过 `workspace_ref` 接入 GQA/MLA flash op（`:101-103,187-189`）。独立探针得到 TP=1/2/8 的 `2 MiB / 1 MiB / 256 KiB`，严格按 `1/tp` 缩放。因此 P1-09 原剩余缺口已闭环。

### 3.3 P2-06：VPP timeline chunk 身份

`TimelineSample` 已有结构化 `chunk` 字段（`cost_eval/mem_timeline.py:227-240`），`rec()` 同时把 chunk 写入字段和事件标签（`:493-508`），VPP 调度步骤保留 chunk id（`:536-551`）。除作者只测的 stage0 外，本次又覆盖三组 PP/V/m 和全部 stage，所有 `(event, mb, chunk)` 均唯一。

### 3.4 P2-01 的核心对象确实已有双口径

`PeakMemoryReport` 已增加 `allocated_oom` 与 `reserved_oom`（`cost_eval/report.py:91-109`），作者构造的 allocated 未超、reserved 超限用例真实通过（`tests/test_closure_audit_2026_07_14.py:149-160`）。这证明**核心对象的属性拆分**是真修复；之所以整项仍判部分闭环，是调用链仍只输出 allocated，见 §4.8。

### 3.5 12 个既有真机锚点重放

| 锚点 | 当前预测 MiB | 已存真机 MiB | ratio |
|---|---:|---:|---:|
| DSv3 4L full | 12437.9 | 12473.1 | 0.9972 |
| DSv3 8L full | 13861.9 | 13953.3 | 0.9935 |
| DSv3 4L full, EP2 | 12395.9 | 12474.1 | 0.9937 |
| CP2 colossal full | 12437.9 | 12433.0 | 1.0004 |
| CP2 ulysses full | 12437.9 | 12441.0 | 0.9998 |
| PP2 stage0 | 10826.0 | 10246.2 | 1.0566 |
| PP2 stage1 | 45767.3 | 45655.5 | 1.0024 |
| CP2 none | 18656.4 | 20119.4 | 0.9273 |
| select attention | 18872.6 | 18828.0 | 1.0024 |
| select MLP | 14865.9 | 15764.7 | 0.9430 |
| select both | 14001.9 | 13953.3 | 1.0035 |
| DSv4 fused | 14929.8 | 15415.5 | 0.9685 |

均值是 `2.0201%`，答复 `analysis/closure_audit_response_2026-07-15.md:65` 的 2.0% 数字可信。CP2-none 和 select-MLP 仍低于 0.95，答复也已在 P2-04 开放项中承认（`:54`），不存在掩盖。

## 4. 被过度声明为闭环的 10 项

### 4.1 P0-02：adapter 静默丢语义 —— 仍是部分闭环

新增 training/context/swap 键表能抓住答复列出的三个 typo，但没有形成完整 adapter schema，而且新守卫本身存在布尔值漏洞：

1. `_PAR_UNSUPPORTED_TRUTHY` 明确把 PP overlap、CP async、EP async D2H 等列为未建模（`cost_eval/configs/from_mindformers.py:531-540`），但判断把 `1` 放进允许集合（`:579-582`）。Python 中 `True == 1`，所以 `pipeline_parallel_overlap_p2p=True` 被静默接受；独立探针已复现。
2. `_build_optimizer` 只读取 `type`，不校验 optimizer 余下键（`:760-769`）；`tyep: SGD` 或顶层 `optimzier:` 都会静默回落到 AdamW。
3. 段级 schema 只调用 training/context/swap 三段（`:781-825`），没有顶层 schema。
4. 答复声称“非零 dropout fail-loud”（`analysis/closure_audit_response_2026-07-15.md:20`），实现却只拒绝 `> 0`（`cost_eval/configs/from_mindformers.py:331-339`）；`-0.1` 这个非法非零值被接受。

所以列出的四个正例确实修了，但“adapter 静默丢语义”这一原问题没有全域闭环。

### 4.2 P1-01：shared gate 与“逐模块逐字节” —— 仍是部分闭环

`sh_gate_w [H,1]` 参数确实已加入（`cost_eval/layers/ffn.py:206-213`），但实现只把 `shared_gate` 作为尾部孤立 op：其输出 `sh_gate` 没有任何消费者；真正合流仍直接用 `comb + sh_o`（`:154-165`）。独立图探针对 DSv4 三种 MoE 层和 MTP 均得到 `consumers=[]`。这只补了参数数目，没有表达 sigmoid gate 乘 shared-expert 输出及其 backward 保存生命周期。

答复引用的“逐模块逐参数精确对账”也不支持 shared gate 闭环：

- 测试只构建默认 `deepseek_v3(4)`，而 DSv3 的 `moe_shared_expert_gating=False`，预期名册没有 `sh_gate_w`（`tests/test_param_conservation.py:96-127`）。
- helper 和断言只比较 `local_numel`（`:71-93,126-127`），没有乘 `dtype_bytes`。本次把 `router_w` 从 4B 人为改成 2B 后，测试所用 numel roster 完全不变，但真实参数字节已变化。

因此参数补丁本身有效，但 shared-gate 功能图和所谓“逐字节”证据都不完整。

### 4.3 P1-02：full 空集/select 零命中 —— 仍是部分闭环

`_validate_recompute_against_graph` 能拒绝 full 的**空集合**和“存在层上的 selector 零命中”，但对不存在层号直接 `continue`（`cost_eval/report.py:43-69`）。正式测试只覆盖 layer 1 上的 missing selector（`tests/test_closure_audit_2026_07_14.py:51-59`）。本次三个反例均被接受并与无重算逐字节相同：

- `RecomputeSpec("full", {999})`
- `RecomputeSpec("select", select_ops={})`
- `RecomputeSpec("select", select_ops={999: {"definitely_missing"}})`

这仍属于“配置看起来启用、实际空转”。

### 4.4 P1-06：CE 与 DSA fusion —— 直接耦合已拆，独立 provenance 未闭环

切换 `apply_dsa_kernel_fusion` 已不会再切换 CE，显式 `cross_entropy_fused` 也能覆盖，作者两条测试有效。但是无显式键时，代码仍用 `attn_type == "dsv4_hybrid"` 推断 CE=True（`cost_eval/configs/from_mindformers.py:449-462`）。

这比旧的 `_dsa_fused(model)` 绑定更好，却仍不是“由独立、可追溯的 loss 配置决定”：相同 DSv4 attention 结构若换普通 CE 实现，adapter 仍会推断 lean；当前测试只是把这条启发式固化为 `True`（`tests/test_closure_audit_2026_07_14.py:126-135`），没有测试真实 loss 实现来源。本项应记“直接 bug 修复、设计闭环不足”。

### 4.5 P1-10：`o_groups` 校验 —— 只挡 0，负数仍产负 numel

实现用 `if cfg.o_groups:` 做整除检查，再用 `not cfg.o_groups` 拒绝 0（`cost_eval/build_llm.py:76-88`）。`o_groups=-1` 两个条件都不会拒绝，最终解析出：

```text
o_group_out.local_numel = -4194304
o_w.local_numel         = -1835008
```

正式测试只测 `o_groups=0`（`tests/test_closure_audit_2026_07_14.py:72-74`）。原问题是结构合法性，不应把“0 的裸除零变清晰错误”等同于输入域闭环。

### 4.6 P1-11：通用结构校验 —— 只补了 topk/capacity 两个边界

`moe_router_topk<=0` 和 `moe_capacity_factor<=0` 的 guard 已生效（`cost_eval/build_llm.py:59-72`），但 `_validate_structure` 仍没有核心维度正值约束。本次实测以下配置都能建图/解析：

- `hidden_size=-1792`：产生负参数 numel；
- `seq_length=0`：产生零激活；
- `vocab_size=0`：产生零 vocab 张量；
- `num_layers=-1`：退化为仅 embedding/head 的 2 层图。

所以答复中的两个具体反例已修，但 P1-11“通用结构校验”仍不能算完整 validator。

### 4.7 P1-16：feasibility validator —— 只有三个特例，不是约束矩阵

`feasibility_errors()` 目前只有 PP+swap、TP>1+SP=false、非 Adam 三条（`cost_eval/report.py:16-40`）。这三条在核心 Evaluator 的确生效；但原问题还包含 runtime constraint matrix。独立探针证明：

```text
ParallelConfig(interleave=0)          -> 接受并返回峰值
ParallelConfig(prefetch_depth=-1)     -> 接受并返回峰值
ParallelConfig(num_microbatches=0)    -> 接受并返回峰值
```

这些值在调度中被退化、空循环或产生无意义峰值。答复把“三个守卫实现”升级成 P1-16 整项闭环，不成立。

### 4.8 P2-01：双 OOM 口径 —— 核心属性有了，用户输出仍是单口径

核心 `PeakMemoryReport` 的拆分是真实的，但现有主要消费者没有输出它：

- Web explorer stage JSON 仍只给 `"oom": sp.oom`（`serve_explorer.py:535-548`）；
- 策略矩阵仍只给 `"oom": rep.oom`（`analyze_matrix.py:75-85`）；
- `reserved_estimate_bytes` 只加 HCCL buffer（`cost_eval/report.py:91-93`），并未实际估算 docstring 提到的 allocator pool 碎片。

因此“核心 API 有双属性”已完成，“对外报告拆成两口径、避免用户误判容量”尚未完成。

### 4.9 P2-07：测试固化风险 —— 新反例有价值，但“逐模块逐字节 exact”不成立

新增 17 条测试成功阻止上一轮多数反例回潮，这是实质改进；但本轮独立反例又穿过全部 493 个测试，说明测试仍主要覆盖单点：不存在层号、负 `o_groups`、unsupported bool、基本负/零维度和调度非法值均无回归。

尤其 `test_dsv3_per_module_exact_roster_and_count`：

- 只测 DSv3，不测新增 shared gate；
- 只测 numel，不测 dtype/bytes；
- 测试文件顶部仍明确保留全局 ±2%（`tests/test_param_conservation.py:1-5,31-64`）。

所以可以说“新增反例回归”，不能说参数守恒已由“逐模块逐字节 exact”完整替代。

### 4.10 P2-08：文档漂移 —— 指定的两处修了，但当前代码仍有直接矛盾

`test_review_evidence` 和 `test_ce_optstep` 的旧措辞确已订正，但同一修复提交留下了新的/未清理的矛盾：

1. `cost_eval/layers/attention.py:26-29` 仍写 workspace 是字符串、只除 CP 不除 TP；实际下方 `:39-45` 已是按 TP shard 的 TensorRef。
2. `cost_eval/configs/from_mindformers.py:69-72` 写 `qk_layernorm` 归入 mapped 并由 build_llm 守卫，实际它就在 ignored set（`:73-82`）且 adapter 会把 true 变成 false。

因此指定 docstring diff 真实，但 P2-08“文档漂移整项闭环”不真实。

## 5. 32 项逐项最终状态

| 项 | 本次状态 | 相对最新答复 | 简要依据 |
|---|---|---|---|
| P0-01 | 闭环 | 同意 | 累计梯度逻辑未被 C1-C5 改动；既有本地逐字节和 NPU 生命周期证据仍有效 |
| P0-02 | 部分 | **下调** | unsupported `True`、optimizer/top-level typo、负 dropout 仍静默 |
| P0-03 | 闭环 | 同意 | 核心枚举校验 + 生命周期接线 |
| P0-04 | 闭环 | 同意 | TP vocab 栈代码未被本轮改动；既有 DTensor 分片证据仍有效 |
| P0-05 | 闭环 | 同意 | DSA KL TensorRef 接线未回归 |
| P1-01 | 部分 | **下调** | gate 参数存在但数据流孤立；exact 测试不测 gate/bytes |
| P1-02 | 部分 | **下调** | 空 select、越界层号仍空转 |
| P1-03 | 闭环 | 同意 | 同名异 shape 不变量仍有回归 |
| P1-04 | 闭环 | 同意 | embedding dtype 接线未回归 |
| P1-05 | 闭环 | 同意 | 独立 DSA 入口未回归 |
| P1-06 | 部分 | **下调** | DSA fusion 直接耦合已拆；默认仍从 attention 类型推断 loss kernel |
| P1-07 | 开放 | 同意 | checkpoint islands 未建 |
| P1-08 | 开放 | 同意 | bwd scratch 仍求和上界 |
| P1-09 | 闭环 | 同意 | saves 生命周期和 workspace TP 缩放均落实 |
| P1-10 | 部分 | **下调** | 0 被拒，负 `o_groups` 仍产负 numel |
| P1-11 | 部分 | **下调** | topk/capacity 两点已修，基础维度 validator 仍缺 |
| P1-12 | 开放 | 同意 | MoE skew/capacity-bound 未建 |
| P1-13 | 开放 | 同意 | CP kernel buffer 未建全 |
| P1-14 | 开放 | 同意 | FSDP 子模块 wrap 时间线未建 |
| P1-15 | 开放 | 同意 | PP send/overlap/更多调度未建 |
| P1-16 | 部分 | **下调** | 三条 guard 有效，不是完整 feasibility matrix |
| P1-17 | 部分 | 同意 | README 已诚实收窄，UI 仍非 round-trip |
| P1-18 | 闭环 | 同意 | nested offset 延迟物化未回归 |
| P1-19 | 部分 | 同意 | 非 Adam 核心拒绝；分离 offload 仍不可表达 |
| P2-01 | 部分 | **下调** | 核心双属性；UI/矩阵仍只展示 allocated `.oom` |
| P2-02 | 开放 | 同意 | opdag 仍非主链 |
| P2-03 | 部分 | 同意 | qk true 仍被 adapter 变 false；仅以“可忽略”收窄 |
| P2-04 | 开放 | 同意 | 泛化/低于 0.95 锚点仍在 |
| P2-05 | 开放 | 同意 | 全尺寸预设仍属外推 |
| P2-06 | 闭环 | 同意 | chunk 结构字段及多组合唯一性成立 |
| P2-07 | 部分 | **下调** | 新回归有价值；exact/输入域覆盖声明过度 |
| P2-08 | 部分 | **下调** | 两处旧文档修了，但仍有源码注释直接反证实现 |

## 6. 假阳性与证据边界

### 6.1 是否存在原审查假阳性

**没有发现整项假阳性。** 上一轮“8 闭环 / 15 部分 / 9 开放”的方向仍正确；本轮确实把其中若干缺口推进了，但没有推进到答复宣称的 20 项全闭环。

### 6.2 本轮为什么没有重新跑 NPU

本轮 C1-C5 主要变更是离线配置校验、report 属性、timeline 元数据、参数名册测试和 FA workspace 的 TP 公式。配置 typo/非法值是否被拒绝、chunk 是否唯一、workspace 是否按 TP 切，均可由当前评估器代码确定；在 NPU 上跑一个非法配置不会比核心单元测试提供更多证据。

P0-01 的梯度生命周期和 P0-04 的 TP vocab 分片已有上一轮重新采集的 NPU 决定性证据，而本轮提交没有修改相应 `grad_accum` 状态转移或 `layers/head.py` 分片实现，故没有重复消耗真机。12 锚点表是基于已存真机值的**本地重放**，报告不把它表述成新真机采样。

若要把 P1-06 和 P1-01 从“部分”升级为“闭环”，下一轮真正有价值的 NPU/源码验证应是：同一 DSv4 attention 下分别运行 fused/unfused/chunked CE，确认 loss kernel provenance；以及开启 `use_shared_expert_gating=True` 后采 gate multiply 的保存张量与峰值，而不是重复已有 12 锚点。

## 7. 建议的真正关闭门槛

1. 修 `_PAR_UNSUPPORTED_TRUTHY`：布尔 True 必须拒绝；给 optimizer 和顶层段增加 schema。
2. graph-aware recompute 校验必须拒绝空 select、full/select 越界层号，并验证每个目标层至少一项命中。
3. CE 类型应来自独立 loss 配置/运行时实现标识；架构类型最多作为显式兼容默认并输出 warning/provenance。
4. `o_groups > 0`，并为 layers/H/S/vocab/heads/并行度/microbatch/prefetch 等建立统一正值和整除 validator。
5. shared gate 要进入 `sigmoid(gate) * shared_output -> merge` 数据流；参数测试按运行时模块清单覆盖 DSv3/DSv4/gate on/off，并比较 `numel * dtype_bytes`。
6. Web/CLI/矩阵同时输出 `allocated_oom`、`reserved_oom`、对应余量及 reserved 估计边界。
7. 把本次独立探针中的反例转正式回归，并清理 attention/qk_layernorm 的矛盾注释后再关闭 P2-07/P2-08。
