# LLM 内存实验台（serve_explorer）使用说明

交互式训练显存实验台：改模型结构 / 并行切分 / 重算策略 → **实时重算**每卡峰值显存，展示逐层
op-DAG（每算子标注要存的激活及其 shape 计算式）、全 stage 内存时间线、峰值各桶分解。
后端 = 本仓库评估器（真机锚点校准，见 `specs/2026-07-07-memory-model-reference.md` §14）。

## 1. 启动

```bash
cd pynative-cost-evaluator
python serve_explorer.py            # 默认端口 8765
python serve_explorer.py 9000       # 指定端口
```

浏览器打开 **http://127.0.0.1:8765**。零第三方依赖（stdlib `http.server`；`yaml` 仅 yaml 导入用）。
改任意配置项 → 自动防抖重算刷新（无需按钮）。配置非法时页面顶部红框列出全部错误（中文）。

## 2. 页面布局

| 区域 | 内容 | 交互 |
|---|---|---|
| 顶部配置区 | 模型预设 / yaml 导入 / 模型结构 / 结构维度 / 并行切分 / 重算 | 改动即重算；右上 KPI = 设备峰值（最紧 stage）+ world 数 |
| Stage 标签行 | `Stage N · 峰值 · 层数`（pp>1 时多个） | 点击切换左图/联动高亮；⚠OOM 红标 |
| 左：模型结构 op-DAG | 逐层折叠组（`L0 embedding … lm_head`），展开 = 该层真 SVG DAG（拓扑分层 + 连线箭头，分支/汇合可见） | 点层头展开/折叠；**悬停算子** = 高亮上下游（入边蓝/出边红）+ 右栏详情；算子右上 **↻** = 标记该 op 重算（custom） |
| 左下：内存时间线 | **全部 stage** 各一块堆叠面积图（12 桶 + total 黑线 + 峰值★） | 点块标题切 stage；点图内任意时刻 = 右栏显示该事件各桶 |
| 右：详情面板 | 算子详情（要存的激活 + `(符号shape)=数值 ÷切分 ·dtype` 计算式、源码位置、上下游）或 timeline 事件的各桶分解 | 上下游列表可点跳转 |

## 3. 参数说明

> 逐框「填什么/默认/约束/yaml 来源」的**完整参数字典**见 **[README_config.md](README_config.md)**；下面是分区速览。

### 3.1 模型预设 / yaml 导入

| 项 | 说明 |
|---|---|
| **模型预设** | `Custom` / `DSv3-mini`（仓库缩层锚点，真机验证 12473）/ `DeepSeek-V3 671B` / `DeepSeek-V3.2-Exp` / `DeepSeek-V4-Flash` / `DeepSeek-V4-Pro` / `GLM-5`。结构字段抓自 **HF config.json**（2026-07，来源显示在 KPI 下方）。选中即填充全部字段；**手改任意结构字段自动跳回 Custom**（并行/重算字段不算偏离） |
| **yaml 导入** | 选 mindformers 训练 yaml → 解析回填**完整字段**（**不写回文件**;P1-17/§4.8 **完整 round-trip**，closure-audit 2026-07-15）。**页面点「评估」时按解析出的完整 `EvaluatorConfigBundle` 算**——`data_parallel_replicate` 进 world/HCCL 域、`reshard_after_forward_policy` 改 gather 生命周期、`cpu_offload`、`prefetch`、**设备容量**（`context.max_device_memory`，缺省 54 GiB）、**优化器 dtype**（`model.params_dtype`）全部生效，**不再**固定假设 64 GiB/AdamW-fp32。这些非 UI 子集的解析值回填到「运行时/硬件」行（§3.6，可见可改）+ 隐藏字段（`sequence_parallel`），随页面评估请求一起回传。支持新式段（`parallelism`/`recompute`）与老式段（`parallel_config`/`recompute_config`/`model.model_config` 嵌套、`offset`（含 VPP 嵌套列表）/`pp_interleave_num`）。glm4/qwen 系字段名自动别名映射。**缺必需结构字段**（yaml 依赖 mindformers 类内默认）→ 用页面当前值兜底 + 黄条警告逐字段列出，请核对。select 重算配置会逆向渲染成「细粒度选重」文本。`swap` 段导入侧恒关（`swap.enable=True` 会 fail-loud，需在评估器侧手构 `SwapSpec`）。**手工配置路径**（不导入 yaml，从预设/Custom 起手）仍用页面字段 + 合理默认（64 GiB/AdamW-fp32/dp_replicate=1/reshard=default/offload 关），行为不变 |

### 3.2 模型结构

| 参数 | 含义 | 约束/说明 |
|---|---|---|
| `attn` | `MLA` / `GQA` / `MHA` / `DSA` / `DSv4-hybrid` | **DSA**（V3.2/GLM-5）= MLA 结构，稀疏 indexer 未建模 → 按 full-attention **上界**（OOM 安全侧）；`DSv4-hybrid` = DSA/CSA/HCA 逐层混合（compress_ratios 循环近似）+ mHC + o_groups |
| `layers` | transformer 层数 N | ≥1 |
| `dense 层数` | 前 K 层 dense、其余 MoE（`first_k_dense_replace`） | =layers 则纯 dense；纯 dense 时 experts 无效 |
| `experts` / `topk` | 路由专家数 / 每 token 选取数 | `topk ≤ experts` |
| `heads` / `kv_groups` | 注意力头数 / GQA 的 KV 组数 | `kv_groups \| heads`；MHA = kv_groups==heads；MLA/DSA 下 kv_groups 惰性 |
| `seq` / `batch` | 训练序列长 / 每卡 micro-batch | `cp \| seq` |
| `mtp 层数` | MTP（`num_nextn_predict_layers`） | **计入可切分总层数**（pp 分配的和 = layers+mtp），位于层序列末端 |

### 3.3 结构维度（Custom / 预设微调）

`hidden / ffn / moe_ffn / q_lora / kv_lora / qk_nope / qk_rope / v_head / vocab` —— 全部开放；
UI 值**最终覆盖**预设；`head_dim` 自动 = qk_nope + qk_rope。

### 3.4 并行切分

| 参数 | 含义 | 校验规则（非法即红框报错） |
|---|---|---|
| `dp_shard` | FSDP 分片度 | ≥1，无上限 |
| `tp` | 张量并行 | `tp \| heads`；GQA 下还需 `tp \| kv_groups` |
| `ep` | 专家并行 | `ep \| experts` 且 `ep \| dp·cp·tp`（专家在该区内分片）；纯 dense 时须为 1 |
| `pp` | 流水线并行 | `pp ≤ layers+mtp` |
| `pp 层分配` | 每 stage 的 transformer(+mtp) 层数，逗号分隔（mindformers `num_layer_list` 口径），如 `5,5,6,6,6,6,5,5` | 段数==pp、和==layers+mtp、每段≥1；**embedding 自动归 stage0、head 自动归末 stage（不占配额）**；空 = 均匀切（余数**前置**分摊：N=8 pp=3 → 3,3,2） |
| `vpp` | 虚拟流水交错数（`pp_interleave_num`） | ≥1 |
| `cp` / `cp 算法` | 上下文并行 / `colossal`（KV all-gather full-S）`ulysses` `ring` `hybrid` | `cp \| seq` |

world = dp_replicate·dp_shard·tp·pp·cp（显示在 KPI 下；手配默认 dp_replicate=1，yaml 导入按解析值）。

### 3.5 重算（recompute）

> 改重算档位/层范围/细粒度配置 → **左侧「模型结构」逐层激活实时随之变化**（2026-07-15）：被重算的 op 标**紫色虚线**「↻重算·不存」、其激活**不计入**该层 stored 总量；层头显示 `↻全重算/选择性重算 · 存 X MiB（层入口锚点 Z）/ 全量 Y`。完整参数字典见 **[README_config.md](README_config.md) §5**。

| 档位 | 配法 | 说明 |
|---|---|---|
| 无 | 下拉「无」 | 全激活保存（act_live 最大） |
| full | 下拉「full」；`重算层范围` 限定作用层 | 层整层重算（仅存层入口锚点） |
| select(模块) | 下拉「select」+ 模块选 `self_attn`/`mlp`/`both`；`重算层范围` 限定作用层 | 按 cell 重算（`both`≈full，真机退化端 0.991） |
| custom（图上选） | 下拉「custom」或直接**在左图 op 节点点 ↻**（自动切 custom）；`重算层范围` 限定层 | 任意 op 集合 × 层号集；chips 行可移除 |
| 细粒度重算（统一入口） | `细粒度重算` 输入框，**一个框、两种写法自动识别、互斥**：① 按 PP stage `s0:both; s1-2:self_attention; s3:none`（stage→层跟当前 pp 切分）；② 按绝对层号（mf `select_module`，0-indexed）`self_attention:0-7,11-12,22-24; flash:4-7` | 非空即优先于图上勾选/select 模块；模式/pattern = `none`/`self_attention`/`mlp`/`both`/op 名子串；**每 pattern 层范围支持多段不连续** |

> - **`重算层范围`**（`sel_layers`）对 **full / select / custom 均生效**（空 = 全部层 1-N），**支持多段不连续** `1-8;12-13;23-25`（`,`/`;`/全角皆可）——2026-07-15：① 改 full/select 生效（此前仅 custom）；② 支持多段（此前仅单个 `a-b`）。
> - **细粒度重算**合并了旧「per-stage 选重」与「细粒度选重(mf 口径)」两个框（2026-07-15）：二者本就产出同一 `select_ops`、且互斥；旧 `sel_stage` 保留为隐藏兼容别名。stage 写法的"full"用 `both` 表达（评估器 full/select 为全局互斥模式，`both` 真机退化端 ≈full）。

### 3.6 运行时 / 硬件（非 UI 子集，yaml 导入 round-trip 的落点）

这些是 mindformers 训练配置里**非模型/非切分**、但影响每卡显存/OOM 判定的项。**手工配置**时用合理默认（下方「默认」列）；**yaml 导入**时按解析值回填、页面评估直接生效（P1-17/§4.8 闭环）。

| 参数 | 含义 | 默认（手配） | yaml 来源 |
|---|---|---|---|
| `dp_replicate` | 纯数据并行度（权重/优化器逐 rank **复制**、不切分）——单卡峰值与 `dp_shard=1` 相同，进 world/HCCL 域 | 1 | `parallelism.data_parallel // data_parallel_shard`（或老式 `data_parallel` + `enable_parallel_optimizer=False`） |
| `reshard` | `reshard_after_forward_policy`：`default`（PP 整体不 reshard；非 PP 除 output 均前向后即 reshard）/ `always`（前向后即 reshard、反向 re-gather）/ `never`（unsharded 权重驻留至本模块反向）——改 **gather 生命周期**（`fsdp=dp_shard·cp>1` 时生效） | default | `parallelism.reshard_after_forward_policy` |
| `cpu_offload` | 参数/优化器状态卸载 CPU：开 → 该 stage 持久态=0、优化器 step 无设备瞬态 | 关 | `parallelism.cpu_offload` |
| `prefetch` | FSDP 参数预取深度（`prefetch_depth`）：0=单缓冲，≥1=下 N 层双缓冲 | 1 | （当前 yaml 无对应键，恒 1；可手改建模更深预取） |
| `设备容量(GiB)` | 设备 HBM 容量（`HardwareSpec.max_device_memory`）——**OOM/reserved 余量判据用它**，不再硬编 64 GiB | 64 | `context.max_device_memory`（缺省 54 GiB） |
| `优化器 dtype` | AdamW params dtype：`fp32`（state=master+m+v=12 B/param）/ `bf16`（+compute 副本 2 B=14 B/param） | fp32 | `model.params_dtype` |

> 另有隐藏字段 `sequence_parallel`（yaml 导入按 `parallelism.sequence_parallel` 解析值；手配缺省时按 `dp_shard>1 或 tp>1` 推导）。选任一模型预设 = 手配路径 → 这些 extra 复位为上表默认（不残留上次导入值）。

## 4. 口径与诚实边界

- **数值口径**：`max_memory_allocated` 每卡峰值（MiB）；HCCL 通信缓冲在 reserved 池、单独显示不计入峰值。
- **校准**：DSv3 缩层真机锚点（full 0.995 / cp-full 0.998 / select-both 0.991 / select-attn 1.001 含 B margin）；综合 12 锚点平均 |误差| 2.4%（`analysis/realmachine/sim_vs_real_report_2026-07-09.md`）。
- **kept_frag（B margin）**：select 保留-MoE 的 loss 峰碎片长尾标定项（1.9×kept-MoE 激活，自真机标定，仅 select-kept-MoE 触发）。轻量 custom select（如只重算 flash）时该 margin 属保守外推（OOM 安全侧）。
- **未建模/上界**：DSA/GLM-5 稀疏 indexer（按 full-attn 上界）；V4 `compress_ratios` 逐层全列表未抓取（0/4/128 循环近似）；预设中 GLM-5 mtp 未在 HF config 确认（填 0）。
- 真机未覆盖的配置（cp>2、pp>2 逐 stage、VPP、swap）只有仿真预测、无真机对比。

## 5. 常见报错

| 报错 | 原因/处置 |
|---|---|
| `tp(x) 必须整除注意力头数` 等 | §3.4 校验规则，按提示改并行度 |
| `pp 层分配之和(x) 必须 == 可切分总层数 transformer+mtp(y)` | 和里记得算上 mtp（或把 mtp 框设 0） |
| `yaml 的 model 段缺必需结构字段 [...]` | 该 yaml 依赖 mindformers 类内默认——正常情况下会自动用页面值兜底+警告；报此错说明连兜底值都无效 |
| `未识别的 mindformers model 字段 [...]` | 转换器 fail-loud（防静默错图）：新字段需在 `from_mindformers.py` 补映射或论证加入忽略集 |
| `此并行组合可能不被 mindformers 栈支持` | 评估器算得出，但真机栈有已知限制（SP+MoE / TP+MoE / pp>2 优化器 bug，见参考文档 §15） |

## 6. 相关

- 模型/内存口径：`specs/2026-07-07-memory-model-reference.md`
- 真机验证：`analysis/realmachine/`（锚点数据 + sim-vs-real 报告）
- 静态 op-DAG 提取（交叉验证工具）：`cost_eval/opdag/`、`specs/2026-07-08-source-grounded-opgraph-design.md`
- 其它可视化生成器：`build_memory_dashboard.py`（静态面板）、`build_opdag_explorer.py` / `build_model_explorer.py`（静态 HTML）
