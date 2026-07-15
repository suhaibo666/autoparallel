# `closure_audit_v2_response_2026-07-15.md` 可信度复核与回归审计

> 日期：2026-07-15  
> 被审对象：`analysis/closure_audit_v2_response_2026-07-15.md`  
> 固定代码基线：`feat/unified-llm-modelspec` @ `b4b8579`，审计开始时与
> `origin/feat/unified-llm-modelspec` 一致且 worktree clean  
> 对比基线：`68f6cd2..b4b8579`  
> 独立探针：`analysis/closure_audit_v2_verification_probe_2026-07-15.py`

## 1. 结论先行

**v2 答复只能判“部分可信”，不能接受其“18 项闭环”的最终结论。**

可以确认的事实：

- 完整测试确为 **596 passed**；
- v2 点名的 15 个旧反例所对应的正式回归均能通过；
- shared-gate 的孤立数据流已接通，gate-on 输出已有消费者；
- 12 个已存真机锚点本地重放的平均绝对相对误差仍为 **2.0%**，9/12 在 ±5%；
- `o_groups=-1`、负/零核心维度、full/select 的三个已知空转样例、Web/矩阵双口径字段，均有实质推进。

不能确认、且已被反例推翻的部分：

- v2 声称本轮新增闭环的 8 项中，至少 **7 项仍只能算部分闭环**；
- P0-02 的 `dense_fsdp_shard_size` 修复引入了新的**静默语义丢失和假拒绝**；
- `ulysses_degree_in_cp=1` 从旧版可接受变成了假拒绝，是另一项新回归；
- shared-gate 只闭合了拓扑，未闭合真机 dtype、FP32 cast 和保存生命周期；
- `qk_layernorm` 被标成“真·可忽略”不符合 MindFormers 源码，也会让 Qwen3 配置被静默改写；
- core API 仍接受重算 mode typo、空字符串 selector、布尔维度和布尔并行度；
- `reserved_oom` 仍是“不含 allocator pool 碎片的下界判定”，并非完整 reserved OOM。

按 v2 自己的“功能语义 + 输入域 + 对外结果 + 回归”闭环标准，**“18 closed”不成立**。若保留此前已复核的
10 项闭环，并只把本轮窄义的 `o_groups<=0` 子问题计为闭环，则当前 closed 数量上限是 **11**；其余本轮
新增项应回退为“部分”。不建议在修复下面问题前继续给出新的总闭环数。

## 2. 实测矩阵

| 验证项 | 结果 | 判读 |
|---|---|---|
| `python -m pytest -q` | **596 passed**, 5 warnings, 9.20s | v2 的测试数属实 |
| `python sim_vs_real_report.py` | 12 锚点，平均 `|ratio-1|=2.0%`，9/12 在 ±5% | v2 的锚点数字属实；是已存数据重放，不是新 NPU 采样 |
| 旧探针 `analysis/closure_audit_verification_probe_2026-07-15.py` | 直接在 `ParallelConfig(interleave=0)` 抛错并退出 1 | 修后不能完整运行，v2 的“15 探针全拒”缺少可直接复跑的统一产物 |
| 新独立探针 | exit 0；复现 dense-FSDP、Ulysses、重算、标量、gate dtype、qk norm 反例 | 证明正式测试存在定向覆盖盲区 |
| 116/NPU | 网络诊断 L0/L1/L2 PASS，L3 VPN unreachable；安全修复后仍失败 | 本轮无法新增真机采样；不影响纯配置/源码反例，gate cast 峰值仍需后续 NPU 定量 |

完整测试的 5 个 warning 均来自 DSv4 缺显式 `cross_entropy_fused` 时的架构启发式，和 v2 对 P1-06
“仍为部分”的表述一致。

## 3. 严重问题

### F1 — P0：`dense_fsdp_shard_size` 修复是上下文无关的错误判定，并引入静默低估

v2 把该数值键处理成“`=1` 合法、`>1` 未建模而拒绝”（答复 `:26`；实现
`cost_eval/configs/from_mindformers.py:561-565,611-615`）。这个规则忽略了完整 FSDP 域大小。

MindFormers 的真实约束是：

- `shard_size` 必须为正整数；
- 它必须整除 `fsdp = dp_shard * cp`；
- **只有 `shard_size == fsdp` 才复用完整 FSDP mesh、与默认语义一致**；
- `shard_size=1` 且 `fsdp>1` 表示 dense 权重不在完整域上分片，而不是“中性值”。

源码依据：`../mindformers/mindformers/pynative/distributed/parallel_dims.py:443-470`。

独立探针得到：

| 输入 | 当前 adapter | MindFormers 语义 | 结论 |
|---|---|---|---|
| `data_parallel_shard=8, dense_fsdp_shard_size=1` | **接受**，返回与未配置该键完全相同的 `ParallelConfig(dp_shard=8)` | dense 权重 shard 域为 1，显存语义改变 | **静默丢语义，可能低估 dense 参数/聚合峰值** |
| `data_parallel_shard=8, dense_fsdp_shard_size=8` | **拒绝** | 等于完整 FSDP 域，复用默认 mesh | **假阳性** |
| `data_parallel_shard=8, dense_fsdp_shard_size=0` | **接受** | runtime 明确拒绝非正整数 | **假阴性** |

新增测试只测了 `dense_fsdp_shard_size=1` 且未配置 FSDP>1
（`tests/test_closure_v1_adapter.py:78-86`），因此恰好避开了语义发生变化的条件。

这是 **b4b8579 新引入的功能回归**。正确守卫至少应先算出 `fsdp=dp_shard*cp`：

- `value == fsdp`：中性，可接受；
- `1 <= value < fsdp` 且整除：合法但当前未建模，应 fail-loud；
- 其它值/类型：按 runtime 约束拒绝。

### F2 — P0/P1：`qk_layernorm` 不是“真·可忽略”，adapter 正在静默改写 Qwen3 结构

v2 把 P2-08 判为闭环，依据是注释已改成“qk_layernorm 在忽略集、不映射”
（答复 `:34`）。实现进一步声称它是“内存真·可忽略”，且形状只是两个
`[S,B,head_dim]` 小激活（`cost_eval/configs/from_mindformers.py:79-86`），随后在构造
`LLMConfig` 时确实不传该字段（`:404-405`）。

这个分类不符合真实实现：

- MindFormers 会真正构造 `q_layernorm` / `k_layernorm`，并作用于带 head/group 维的 query/key，
  见 `../mindformers/mindformers/pynative/transformers/attention.py:311-357`；
- Qwen3 PyNative 显式强制 `qk_layernorm=True`，见
  `../mindformers/mindformers/models/qwen3/modeling_qwen3_train_pynative.py:47-50`；
- 本地 Qwen3-32B 配置为 S=4096、64 Q heads、8 KV heads、head_dim=128、64 层，并显式配置
  `qk_layernorm: True`，见
  `../mindformers/configs/qwen3/pretrain_qwen3_32b_4k.yaml:124-152`；
- core builder 对直接传入的 `qk_layernorm=True` 反而 fail-loud，明确承认两个 norm op 未建模，见
  `cost_eval/build_llm.py:167-170`。

独立探针证明，adapter 输入 `qk_layernorm=False/True` 后得到的 `LLMConfig` 和 `ModelSpec` **完全相同**。
这不是注释问题，而是 adapter 绕过 core fail-loud、静默评估另一份模型。

仅按 Qwen3-32B 的 Q/K norm 输入张量体积计算：

- 每层 BF16 extent = **72 MiB**；FP32 extent = **144 MiB**；
- 64 层无重算累计 extent = **4.5–9 GiB**。

这不是对最终峰值的直接断言，但足以推翻“两个 `[S,B,head_dim]`、真·可忽略”的分类。实际峰值还取决于
norm 实现保存什么、重算和释放时机，应当建模或 fail-loud，不能静默忽略。

此问题主要是**旧缺陷未闭合**；b4b8579 新增的注释和正式测试
`tests/test_closure_v1_adapter.py:183-196` 把错误行为固化成了 oracle，属于本轮新增的测试债务。

### F3 — P1-01/P2-07：shared-gate 只闭合数据流，未闭合真机逐字节语义

本轮新增的 `shared_gate_mul` 和 merge 消费链是正确推进：

- `sh_gate` 已被 `shared_gate_mul` 消费；
- `sh_o_gated` 已被 `moe_add` 消费；
- multiply 保存了 `sh_o` 和 `sh_gate`。

代码见 `cost_eval/layers/ffn.py:214-231`。因此“孤立叶”子问题可以确认已修。

但逐字节语义仍不对：

1. MindFormers 的 gate Dense 权重 dtype 是 `config.moe_router_dtype`，默认 **float32**，见
   `../mindformers/mindformers/parallel_core/transformer_config.py:1791-1796` 和
   `../mindformers/mindformers/pynative/transformers/moe/shared_experts.py:55-62`。
2. 评估器的 `sh_gate_w` 未指定 dtype，解析后是默认 **2B/BF16**，见
   `cost_eval/layers/ffn.py:224-229`；独立 resolved-graph 探针实测 `sh_gate_w.dtype_bytes == 2`。
3. runtime 在 gate Dense 前显式把完整 hidden states cast 到 router dtype，并在 sigmoid 后再 cast 回
   compute dtype（`shared_experts.py:69-71`）。当前 op 图没有这条 FP32 hidden cast，也没有为 gate
   MatMul 权重梯度单独表达其 FP32 输入生命周期。
4. adapter 把 `router_dense_type` 放进忽略集（`cost_eval/configs/from_mindformers.py:97`），但
   MindFormers 会把它映射为 `moe_router_dtype`，见
   `../mindformers/mindformers/parallel_core/transformer_config_utils.py:353-357`。该字段显然改变权重、
   router/gate 激活和 cast 字节，不是内存中性字段。

新增“逐字节 roster”测试只证明辅助函数能发现**人为腐蚀 router_w 4B→2B**，并没有给 gate-on 的
`sh_gate_w` 建立 4B oracle；gate-on 测试仍只断言 numel 和消费者
（`tests/test_param_conservation.py:154-195`）。因此 P2-07 的“逐字节/gate on 覆盖”是测试存在，
不是实际字节正确性的证明。

本项应为 **部分闭环**。FP32 cast 对最终峰值的定量影响仍需 116 恢复后做 gate on/off 短步真机对照；
权重 dtype 错误则已由源码和 resolved graph 直接证实，无需等待 NPU 才能定性。

## 4. 中等问题

### F4 — P1-02：core 重算输入域仍可静默空转/误选

`_validate_recompute_against_graph` 只处理 `mode in {full, select}`；其它 mode 直接返回
（`cost_eval/report.py:57-59`）。独立探针：

- `RecomputeSpec("ful", {1})` 被接受；
- 其峰值与 `RecomputeSpec("None")` **逐字节相同**（17102764032 bytes）；
- `RecomputeSpec("select", select_ops={1:{""}})` 也被接受；因为空字符串是所有 op 名/类型的子串，
  它会意外命中整层，而不是被当作非法 selector 拒绝。

v2 点名的 full 越界、select 空 map、select 越界、零命中反例确已封住
（`cost_eval/report.py:68-100`），但 core API 的 mode/selector 类型域仍未闭合。P1-02 应继续标“部分”。

### F5 — P1-11/P1-16：新增正值 validator 不等于整数域 validator

`ParallelConfig.__post_init__` 使用 `isinstance(v, int)`（`cost_eval/specs.py:54-68`），而 Python 的
`bool` 是 `int` 子类。独立探针确认以下配置均被接受：

- `ParallelConfig(tp=True)`；
- `ParallelConfig(num_microbatches=True)`。

`build_llm._validate_structure` 只比较 `<1`，没有类型守卫（`cost_eval/build_llm.py:29-58`），因此
`hidden_size=True, num_attention_heads=True` 也能生成 `DimTable(H=True,n_heads=True)`。

另一个转换缺口是浮点压缩比：

- `csa_compress_ratios=(4.9,1,1,4)` 通过校验；
- validator 和 `_compress_ratio` 都执行 `int(r)`（`cost_eval/build_llm.py:72-85,202-210`），
  把 4.9 静默变成 4，最终生成 `dsv4hyb_r4_*` 图。

因此 v2 列举的负/零样例确实修了，但“核心维度/并行标量输入域闭环”仍是过度声明。

### F6 — P0-02：`ulysses_degree_in_cp=1` 被误当布尔真值拒绝

当前代码把 `ulysses_degree_in_cp` 放进 `_PAR_UNSUPPORTED_TRUTHY`
（`cost_eval/configs/from_mindformers.py:553-560`），故显式值 1 被拒。

MindFormers 的真实逻辑是：

- colossal CP 直接返回有效度 1；
- ulysses CP 要求 degree 等于 CP size，CP=1 时 degree=1 合法；
- 只有 hybrid CP 才要求 `1 < degree < cp_size`。

源码见 `../mindformers/mindformers/pynative/distributed/context_parallel.py:248-265`。

独立探针确认 `context_parallel_method=colossal, ulysses_degree_in_cp=1` 被当前 adapter 假拒。这是
b4b8579 将旧的统一允许集合拆分时**新引入的兼容性回归**。该键应按数值和 CP method 联合判定，不能放在
布尔 truthy 集合里。

## 5. 较低但必须修正的问题

### F7 — P2-01：输出链已补齐，但 `reserved_oom` 仍只是下界判定

Web 和矩阵输出双字段确已接通，相关测试也通过。这里的剩余问题是命名和口径：

- `reserved_estimate_bytes` 文档正确承认只加 HCCL、**不含 allocator pool 碎片**，是 reserved 下界，见
  `cost_eval/report.py:122-129`；
- 紧接着 `reserved_oom` 文档又称它包含“allocated + HCCL + 池碎片近似”，见 `:137-141`，与实现矛盾；
- UI/JSON 直接命名为 `reserved_oom`，会让调用方误以为 False 表示真实 reserved 不超容。

现有 DSv4 已存真机点为 allocated 15415.5 MiB、reserved 16092–16096 MiB，差约 676.5–680.5 MiB；
当前模型同类 FSDP-2 的 HCCL 估计为 400 MiB，仍缺约 277–281 MiB allocator/pool 分量。因此临界区
确实存在 `reserved_oom=False` 但真实 reserved 已超的可能。

建议字段改成 `reserved_lower_bound_oom`，或补全 allocator 模型后再保留当前名称。P2-01 应标“对外链路闭合、
真实 reserved 判定部分闭环”。

### F8 — 验证产物不可直接复跑

同一提交加入的 `analysis/closure_audit_verification_probe_2026-07-15.py` 在修后代码上直接退出：它在收集
结果前就构造 `ParallelConfig(interleave=0)`，现在该构造会抛异常。v2 的 15 个旧反例可以从正式测试逐项
确认，但“一个命令打印 15 个 REJECTED”的展示无法由仓库中的原探针复现。

这不是生产功能 bug，但会削弱审计可重复性。本报告新增的探针将每个案例包在独立 capture 中，修复前后均能
完整输出所有结果。

## 6. 对 v2 十项答复的重新判定

| 项 | v2 最终状态 | 本次复核 | 原因 |
|---|---|---|---|
| P0-02 adapter | closed | **部分；且有新回归** | dense-FSDP 上下文判定错误、Ulysses=1 假拒、qk/router dtype 仍静默丢失 |
| P1-01 shared gate | closed | **部分** | 数据流闭合；gate dtype、FP32 cast、保存生命周期未闭合 |
| P1-02 recompute | closed | **部分** | 点名的越界/空 map 已修；mode typo、空字符串 selector 仍通过 |
| P1-06 CE provenance | partial | **部分，表述可信** | 仍由 dsv4 架构启发式默认决定；v2 已诚实承认 |
| P1-10 o_groups | closed | **窄义子问题闭合** | `o_groups<=0` 已封；相邻 CSA 比例类型仍可被静默截断 |
| P1-11 structure dims | closed | **部分** | 负/零已封；bool/非整数域未封 |
| P1-16 feasibility | partial | **部分，但“标量闭环”也不成立** | v2 已承认组合矩阵未完；bool 标量仍通过 |
| P2-01 dual OOM | closed | **部分** | 对外字段已接通；reserved 只是下界且文档自相矛盾 |
| P2-07 exact bytes | closed | **部分** | helper 能检测人为 dtype 腐蚀，但 gate-on 实际 dtype oracle 错/缺失 |
| P2-08 comments | closed | **未闭合** | qk “可忽略”分类与源码相反；reserved 文档仍漂移 |

## 7. 假阳性与新问题结论

### v2 对上一轮审计“没有假阳性”的答复

就上一轮点名的 10 个下调问题而言，**没有发现上一轮审计是假阳性**。v2 对这一点的承认是可信的。

### v2 本轮自身是否有假阳性/假阴性

有：

- 假阴性（接受不该接受/静默丢语义）：FSDP=8 下 dense shard=1、dense shard=0、重算 mode typo、
  空 selector、bool 维度/并行度、CSA 4.9、qk_layernorm=True；
- 假阳性（拒绝合法/中性配置）：FSDP=8 下 dense shard=8、colossal CP 下 Ulysses degree=1。

### 是否引入新问题

明确新引入：

1. `dense_fsdp_shard_size` 的上下文无关分流导致一组误接受/误拒绝；
2. `ulysses_degree_in_cp=1` 新的兼容性假拒绝；
3. qk_layernorm 的错误“真·可忽略”注释和回归测试把既有语义丢失固化为期望行为；
4. reserved property 的新文档与下界实现不一致。

shared-gate dtype/cast、重算 typo、bool 标量主要是**旧缺口未随本轮宣称一起闭合**，不宜都称为新回归；但它们
足以推翻本轮 closed 判定。

## 8. 建议的最小修复与验收顺序

1. **先修 adapter 语义安全**：dense-FSDP 按完整 FSDP 域判定；Ulysses 按 CP method 判定；
   `qk_layernorm`/`router_dense_type` 要么映射并建模，要么 fail-loud。
2. **补 core 输入类型**：排除 bool，要求整数维度/并行度；拒绝非整数 CSA ratio；给
   `RecomputeSpec` mode 和 selector 做构造期 schema。
3. **补 gate-on 逐字节模型**：显式 `moe_router_dtype_bytes`，gate weight/hidden cast/sigmoid/cast-back
   生命周期入图；正式 byte roster 必须对 gate-on 建立外部 oracle，而不是只比 numel。
4. **修 reserved 命名/模型**：在 allocator 未建模前明确输出 lower-bound，避免 `False` 被解释成安全。
5. **恢复 116 后做两个短步 NPU 对照**：同一小模型 gate off/on；同一模型 router dtype bf16/fp32。
   采 `max_memory_allocated`、`max_memory_reserved` 和 profiler live tensors，用来确认 FP32 cast 保存时长。

## 9. 复现命令

```powershell
cd E:\97-codes\torch_parallel\pynative-cost-evaluator

# 正式回归
python -m pytest -q

# 本报告独立反例
python analysis/closure_audit_v2_verification_probe_2026-07-15.py

# 既有 12 个真机锚点本地重放
python sim_vs_real_report.py

# 证明旧探针在修后不能完整运行
python analysis/closure_audit_verification_probe_2026-07-15.py
```

## 10. 最终验收意见

**不建议接受 `closure_audit_v2_response_2026-07-15.md` 的闭环总数和最终状态表。**

可以接受其测试数、旧反例修复事实、锚点重放结果，以及对 P1-06/P1-16 大边界仍未完成的诚实说明；但应将
P0-02、P1-01、P1-02、P1-11、P2-01、P2-07、P2-08 全部回退为“部分”，并先处理 F1/F2/F3。
其中 F1 是会直接改变多维并行显存结论的 P0 缺陷，优先级最高。
