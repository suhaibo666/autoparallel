# 内存实验台 · 配置参数字典（serve_explorer）

本文件是 **[README_explorer.md](README_explorer.md)** 的配套「参数字典」——把网页 http://127.0.0.1:8765
上每一个可填框逐项说明：**含义 / 类型·范围 / 默认（手工配置）/ 示例 / 约束 / 对应 mindformers yaml 键**。
usage/页面布局/口径与诚实边界见 README_explorer；本文件只讲「每个框填什么」。

> 改任意框 → 自动防抖重算刷新（无需按钮）。非法配置 → 页面顶部红框列出全部错误（中文）。
> 所有校验先行、报错前不评估。

---

## 0. 三步上手

1. **选模型** —— 顶部「模型预设」选一个（如 `DSv3-mini`），或选 `Custom` 从零填结构。选中即填满全部结构/维度框；**手改任意结构字段自动跳回 Custom**（并行/重算字段不算偏离）。也可用「yaml 导入」直接解析一份 mindformers 训练 yaml（完整回填，不写回文件）。
2. **配并行** —— 填 `dp_shard / tp / ep / pp / cp`（+可选 `pp 层分配 / vpp / microbatch`）。右上 KPI 实时给出 `world = dp_replicate·dp_shard·tp·pp·cp` 和最紧 stage 的设备峰值。
3. **配重算**（可选） —— 下拉 `recompute` 选 `full`/`select`/`custom`，用 `重算层范围` 限定作用层；细粒度场景用 `细粒度重算` 文本框。**左侧「模型结构」逐层激活会随重算配置实时变化**（见 §5）。

**最小示例**（DSv3-mini，2 卡 FSDP，全重算前 2 层）：`预设=DSv3-mini`，`dp_shard=2`，`recompute=full`，`重算层范围=1-2`。

---

## 1. 模型预设 / yaml 导入

| 框 | name | 含义 | 说明 |
|---|---|---|---|
| 模型预设 | `preset` | 结构模板 | `Custom` / `DSv3-mini`（仓库缩层锚点，真机 12473）/ `DeepSeek-V3 671B` / `V3.2-Exp` / `V4-Flash` / `V4-Pro` / `GLM-5`。结构字段抓自 HF `config.json`。选中即填全部框；**改结构字段→自动跳 Custom** |
| yaml 导入 | （文件选择） | 解析 mindformers 训练 yaml | **完整 round-trip**：`parallelism`/`recompute` 新式段与 `parallel_config`/`recompute_config`/`model.model_config` 老式段均支持；`data_parallel_replicate`/`reshard`/`cpu_offload`/`prefetch`/设备容量/优化器 dtype 全部解析回填并生效（§6）。**不写回文件**。缺必需结构字段→用页面当前值兜底 + 黄条警告。`swap.enable=True` 会 fail-loud |

---

## 2. 模型结构

| 框 | name | 含义 | 类型·默认 | 约束 |
|---|---|---|---|---|
| attn | `attn` | 注意力结构 | `mla`（默认）/`gqa`/`mha`/`dsa`/`dsv4_hybrid` | DSA=MLA 结构、稀疏 indexer 按 full-attn **上界**；dsv4_hybrid=逐层 DSA/CSA/HCA 混合 |
| layers | `layers` | transformer 层数 N | int ≥1，默认 8 | — |
| dense 层数 | `dense_k` | 前 K 层 dense、其余 MoE（`first_k_dense_replace`） | int ≥0，默认 1 | ≤layers；=layers→纯 dense（experts 无效） |
| experts | `experts` | 路由专家数 | int ≥0，默认 8 | 纯 dense 时无效 |
| topk | `topk` | 每 token 选取专家数 | int ≥1，默认 4 | `topk ≤ experts` |
| heads | `heads` | 注意力头数 | int ≥1，默认 8 | `tp \| heads` |
| kv_groups | `kv_groups` | GQA 的 KV 组数 | int ≥1，默认=heads | `kv_groups \| heads`；MHA=heads；MLA/DSA 下惰性 |
| seq | `seq` | 训练序列长 | int ≥1，默认 4096 | `cp \| seq` |
| batch | `batch` | 每卡 micro-batch | int ≥1，默认 1 | — |
| mtp 层数 | `mtp` | MTP（`num_nextn_predict_layers`） | int ≥0，默认 0 | **计入可切分总层数**（pp 分配和=layers+mtp），位于层序末端 |

## 3. 结构维度（Custom / 预设微调）

`hidden` / `ffn` / `moe_ffn` / `q_lora` / `kv_lora` / `qk_nope` / `qk_rope` / `v_head` / `vocab` —— 全部开放，
留空 = 用预设/基座默认；填值 = **最终覆盖**预设。`head_dim` 自动 = `qk_nope + qk_rope`。每项 int ≥1。

## 4. 并行切分

| 框 | name | 含义 | 默认 | 校验规则（非法即红框） |
|---|---|---|---|---|
| dp_shard | `dp` | FSDP 分片度 | 2 | ≥1，无上限 |
| tp | `tp` | 张量并行 | 1 | `tp \| heads`；GQA 还需 `tp \| kv_groups` |
| ep | `ep` | 专家并行 | 1 | `ep \| experts` 且 `ep \| dp·cp·tp`；纯 dense 须=1 |
| pp | `pp` | 流水线并行 | 1 | `pp ≤ layers+mtp` |
| pp 层分配 | `pp_split` | 每 stage 的 transformer(+mtp) 层数，逗号分隔（`num_layer_list`） | 空=均匀切 | 段数==pp、和==layers+mtp、每段≥1；**embedding→stage0、head→末 stage 自动归置（不占配额）**；空时余数**前置**分摊（N=8 pp=3→3,3,2） |
| vpp | `vpp` | 虚拟流水交错数（`pp_interleave_num`） | 1 | ≥1；>1 需 pp>1、microbatch≥pp |
| microbatch | `mbs` | 流水 microbatch 数 | 空=auto（pp>1 时=pp，否则 1） | ≥1 或留空 |
| cp | `cp` | 上下文并行 | 1 | `cp \| seq` |
| cp 算法 | `method` | `colossal`（KV all-gather full-S）/`ulysses`/`ring`/`hybrid` | colossal | — |

> `world = dp_replicate · dp_shard · tp · pp · cp`（KPI 下方显示）。手配默认 `dp_replicate=1`（§6）。

## 5. 重算（recompute）—— 重点

改重算任一项 → **左侧「模型结构」逐层激活实时变化**：被重算的 op 画**紫色虚线框**、标「↻重算·不存 省 X M」、
其激活**不计入**该层 stored 总量；层头显示 `↻全重算/选择性重算 · 存 X MiB（层入口锚点 Z）/ 全量 Y`。
点开该层看 op-DAG，被重算 op 的「要存的激活」明细划为「↻ 不存」。

### 5.1 四个档位（下拉 `recompute`）

| 档位 | 配法 | 每层 stored 激活 |
|---|---|---|
| **无** | 下拉「无」 | 全量 saves 常驻（最大） |
| **full** | 下拉「full」 | 层整层重算 → 仅存**层入口锚点**（checkpoint_input，上一层输出） |
| **select（模块）** | 下拉「select」+ `select 模块`选 `self_attn`/`mlp`/`both` | 选中 cell 的 op 重算，其余常驻（`both`≈full，真机退化端 0.991） |
| **custom（图上选）** | 下拉「custom」，或直接**在左图 op 节点点 ↻**（自动切 custom） | 勾选的 op 重算；chips 行可移除 |

### 5.2 `重算层范围`（`sel_layers`）

层号（1-indexed，含端点），**对 full / select / custom 均生效**，**空 = 全部层 1-N**。
**支持多段不连续**：`1-8;12-13;23-25`（分隔符 `,` / `;` / 全角皆可）→ 取这些层的并集。
例：`recompute=full` + `1-2` → 只前 2 层整层重算；`select` + `1-8;23-25` → 这 11 层按 select 模块重算，其余层保留全量激活。
> 2026-07-15：① 该框对 full/select 生效（此前仅 custom 消费，用户报告 #1）；② 支持多段不连续层（用户报告，此前仅单个 `a-b`）。

### 5.3 `细粒度重算`（`sel_cfg`）—— 统一入口

**一个文本框，按每段 key 自动识别两种坐标系（不可混用，混用报错）**，非空即优先于图上勾选/select 模块：

| 写法 | key 形态 | 语义 | 示例 |
|---|---|---|---|
| **按 PP stage** | `s0` / `s1-2` | stage→层映射跟当前 pp 切分（含 pp 层分配 / mtp）；**stage 号支持多段** `s0,2-3` | `s0:both; s1-2:self_attention; s3:none` |
| **按绝对层号** | cell/op 名 | mindformers `select_module` 口径，层范围 **0-indexed**，**每 pattern 层范围支持多段不连续** | `self_attention:0-7,11-12,22-24; flash:4-7,9` |

- **模式 / pattern**：`none` / `self_attention` / `mlp` / `both`（≈full）/ 任意 op 名子串（`flash`/`e_fc1`/`ln2`…）。
- 分号 `;` 分隔多条 pattern；**每 pattern 的层范围支持多段不连续**（逗号分段）`0-3,6,10-12`；stage 号同理支持多段 `s0,2-3`。
- 2026-07-15 合并：旧「per-stage 选重」与「细粒度选重(mf 口径)」两个框归一（本就产出同一 `select_ops`、且互斥）。旧字段 `sel_stage` 保留为**隐藏兼容别名**（`sel_cfg` 为空时回落读取）。

## 6. 运行时 / 硬件（非 UI 子集，yaml 导入 round-trip 落点）

mindformers 训练配置里非模型/非切分、但影响每卡显存/OOM 判定的项。手配用「默认」列，yaml 导入按解析值生效。

| 框 | name | 含义 | 默认(手配) | yaml 来源 |
|---|---|---|---|---|
| dp_replicate | `dp_replicate` | 纯数据并行度（权重/优化器逐 rank **复制**、不切分；单卡峰值同 dp_shard=1，进 world/HCCL 域） | 1 | `parallelism.data_parallel // data_parallel_shard`（或老式 `data_parallel`+`enable_parallel_optimizer=False`） |
| reshard | `reshard` | `reshard_after_forward_policy`：`default`/`always`/`never`——改 gather 生命周期（fsdp=dp_shard·cp>1 时生效） | default | `parallelism.reshard_after_forward_policy` |
| cpu_offload | `cpu_offload` | 参数/优化器状态卸载 CPU（开→持久态=0、优化器 step 无设备瞬态） | 关 | `parallelism.cpu_offload` |
| prefetch | `prefetch` | FSDP 参数预取深度（0=单缓冲，≥1=下 N 层双缓冲） | 1 | 当前 yaml 无对应键，恒 1 |
| 设备容量(GiB) | `maxdev_gib` | 设备 HBM（`max_device_memory`）——OOM/reserved 余量判据用它 | 64 | `context.max_device_memory`（缺省 54 GiB） |
| 优化器 dtype | `opt_dtype` | AdamW params dtype：`fp32`（state 12 B/param）/`bf16`（+compute 副本=14 B/param） | fp32 | `model.params_dtype` |

> 另有隐藏字段 `sequence_parallel`（yaml 按 `parallelism.sequence_parallel`；手配缺省时按 `dp_shard>1 或 tp>1` 推导）。选任一预设=手配路径→这些 extra 复位为上表默认（不残留上次导入值）。

## 7. 常见报错速查

| 报错关键字 | 处置 |
|---|---|
| `tp(x) 必须整除注意力头数` / `kv_groups` / `cp(x) 必须整除 seq` | §4 校验规则，按提示改并行度 |
| `pp 层分配之和(x) 必须 == 可切分总层数 transformer+mtp(y)` | 和里记得算上 mtp（或 mtp 框设 0） |
| `重算层范围 'a-b' 非法（1-N 内的 a-b）` | §5.2，范围须落在 1-N、a≤b |
| `细粒度重算不可混用 stage(s0:…) 与层号(mlp:0-3)` | §5.3，一次只用一种坐标系 |
| `选重 pattern [...] 在层 k 的 op 图零命中` | pattern 拼错/该层无此 op；报错会列出该层可用 op 名 |
| `未识别的 mindformers model 字段 [...]` | 转换器 fail-loud，新字段需在 `from_mindformers.py` 补映射 |
| `此并行组合可能不被 mindformers 栈支持` | 评估器算得出但真机栈有已知限制（SP+MoE / TP+MoE / pp>2 优化器 bug） |

## 7.5 OOM-安全咨询与保守旋钮（round3 A，2026-07-16）

估计器在若干**已知欠预测风险**处会发**非致命咨询告警**（`warnings`，不改数值、不产错数，只提示——
欠预测=误报"放得下"却 OOM，是最危险方向）。分两类（`cost_eval/advisories.py`）：

| 类别 | 何时发 | 含义 / 处置 |
|---|---|---|
| `OOMSafetyWarning`（F3） | `n_layers` > 16（远超缩层锚点 ≤8L 的已验证尺度） | 全尺寸预测是**外推**,每层小残差随层数累积（欠方向）、无全尺寸真机验证点。留安全余量:见下方旋钮 |
| `OOMSafetyWarning`（D1-R） | MoE + 单 stage(pp=1) + 无重算 + 非 fused-CE 但 `nr_moe_frag_factor=0`（直连 `LLMConfig` 绕过 preset/adapter） | 无重算-MoE loss 峰碎片长尾未补 → 欠 ~0.93x。经 `presets.deepseek_v3` / `from_mindformers` 构建会自动注入 0.6;直连请显式设 `dims.nr_moe_frag_factor`（DSv3 标定 0.6） |
| `ModelingApproxWarning`（N4） | `add_bias_linear` / `add_qkv_bias`=True | bias 内存中性、未建为 param → 按无 bias 继续（不再 fail-loud） |
| `ModelingApproxWarning`（N9） | `layers_per_stage` + `interleave`>1 同给 | VPP 用「连续均衡切 v 段」近似,层不均匀时 chunk 归属可能有偏 |

**保守旋钮（opt-in，默认关=逐字节复现旧行为）**：

| 旋钮 | 位置 | 作用 |
|---|---|---|
| `bwd_scratch_conservative` | `HardwareSpec`（默认 False） | 反向瞬态取**全 scratch 共存严格上界 Σ**（而非默认 window=2）。现实模型全退化为单 scratch → 开关无差;仅当有 >2 个非相邻大 scratch 共存且要"最保守 OOM 上界"时用（F10） |
| `nr_moe_frag_factor` | `DimTable`（默认 0；DSv3 preset/adapter 注入 0.6） | 无重算-MoE OOM-安全标定 margin（**DSv3 两点标定,非物理**；非 DSv3 结构为未验证外推——见 `from_mindformers.py` D1-R 留档） |
| `framework_reserve` | `HardwareSpec`（默认 0） | 显式冗余余量（审计/回归旋钮,也可作外推安全余量） |

## 8. 相关

- 使用/布局/口径与诚实边界：[README_explorer.md](README_explorer.md)
- 模型/内存口径参考：`specs/2026-07-07-memory-model-reference.md`
- 真机验证：`analysis/realmachine/`
