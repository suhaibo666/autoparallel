# 功能检视报告（7 项声明 × 7 并发对抗式 review，2026-07-14）

> 方法：每项声明一个独立 reviewer agent，对抗式审查（读代码 + 实测复现 + 找声明与现实的差距），
> 全部基于 `feat/unified-llm-modelspec` @ b4e3ecb。综合判定：**4 项 CONFIRMED，3 项 PARTIALLY**。

## 一、逐项结论

| # | 声明 | 判定 | 一句话 |
|---|---|---|---|
| 1 | mindformers 代码解读 → op-DAG 建模 | **CONFIRMED** | 纯 AST 解析真源码（8 个硬行号实证吻合）、123 测试绿、不可提取段明文披露；**但 op-DAG 生产角色 = 交叉验证手写建图器，非模拟器运行图** |
| 2 | fsdp/tp/ep/pp 混合并行 + vpp/1f1b | **CONFIRMED**（限内存口径） | 四轴真实切分（dp4·tp2·ep2·pp2 实测+逐轴消融方向全对）、plain-1F1B 在飞微批逐点=Megatron、VPP 实测生效；无时间/气泡建模 |
| 3 | 五种 attn 结构建模 | **PARTIALLY** | gqa/mla/dsv4_hybrid（三分支真不同图）=真实建模、mha=数学精确特例；**dsa=MLA 上界别名**（三处诚实标注，但并列称"建模"拔高） |
| 4 | per-stage 内存建模 + timeline 展示 | **CONFIRMED** | pp2 锚点逐位命中（10794.1/43899.1）、persistent 真 per-stage、warmup 深度按 stage 正确、UI 全 stage 可点渲染 |
| 5 | 细粒度选择重算（mindformers 口径） | **PARTIALLY** | 机制真实（退化端逐字节==full/none、13973 复现、三入口等价验证）；**但只覆盖新式 select_module 的 attn/mlp 子集**，exclude_op/通信重算/老式列表缺失且多为静默丢弃 |
| 6 | yaml 导入解析 | **CONFIRMED**（带 4 缺陷） | 21/24 configs 通过、671b 嵌套 offset/VPP 求和数学验证精确、select 往返正确 |
| 7 | 多种模型经真机回归 | **PARTIALLY** | 12 锚点逐行复现（2.4%）、60+ 真机硬门测试；**但"多种模型"实为 2 个缩层配置**，671B/GLM-5/V4 全尺寸纯外推、3/12 锚点是 in-sample 标定 |

## 二、发现的问题（按危害分级）

### P0 —— 会产生方向性低估（OOM 不安全侧），应修
1. **K_CE 门按全局 mode 而非按层**（`mem_timeline.py:537`，reviewer-5 实测）：per-stage select（如 `s0:both; s1:none`）时，未重算的 loss stage 峰从 22669 → 12569（**-45% 低估**）——全局 mode 变 select 关掉了 loss 层 k_ce=8 的 fat。
2. **yaml `exclude_op` 被静默忽略**（`_build_recompute` 只读三个键）：exclude_op 语义=重算中保留指定 op（真机比评估器多驻留）→ 低估方向。
3. **dp→dp_shard 无条件按 FSDP 映射**（`_mf_adapt`）：不读 `enable_parallel_optimizer`；遇 dp>1 + epo=False 的 yaml 持久内存**静默低估 dp 倍**（本语料侥幸全为 epo=True 或 dp=1）。
4. **fp32_residual_connection 的忽略论证有未设防组合**：论证依赖 layernorm=fp32 覆盖；若 yaml 配 `fp32_residual=True + layernorm=bf16`（语料中存在 ln=bf16 的 yaml），fp32 残差驻留（~40 MiB/层@qwen3-32b）无建模且无守卫。

### P1 —— 功能缺陷/入口分叉
5. **`_SEL_ATTN` 与转换器 `_SELECT_MODULE_OPS` 口径漂移**（reviewer-5/6 独立发现）：serve 多 `"qkv"` → **GQA/MHA 模型经 yaml 导入的 select 静默漏选 qkv 投影**，与 UI 手配同一意图结果分叉（MLA 不受影响）。
6. **老式 `recompute: [per-stage 层数列表]` / `select_recompute: True/[regex]` 被静默丢弃 → mode=None**（`_mf_adapt` 只认 `recompute: True`）——内存关键配置无警告丢失，违背本仓 fail-loud 哲学。
7. **`_mf_adapt` 嵌套 offset 计算先于缺字段兜底**：`finetune_deepseek3_671b.yaml`（缺 num_hidden_layers）用 N=0 算 offset → 和 13≠61 → 该 yaml 永远导不进（fail-loud 兜住、非静默，但需把 offset 计算挪到兜底后）。
8. **UI sel_cfg 错拼 op 子串 = "越错越贵"的静默空转**：`feed_forward:0-7` 匹配零 op 但层被标 select → kept_frag margin 生效 → 峰 21304 > none 18356（真机应≈none）。yaml 路径 fail-loud，UI 路径不校验命中数。

### P2 —— 措辞/文档/忠实度缺口
9. **Megatron `m==pp` 的 all-warmup VPP 特例未实现**，而实验台默认恰为 m=pp → VPP 路径末 stage 峰偏低（reviewer-2）。
10. **真机报告漏报 2 个同族更差点**（reviewer-7 从 validation_matrix 补算）：dp2-none-8L **0.919**、cp2-ulysses-none **0.921**（均为已归因的 no-recompute k_ce 族）——"最差 0.925/仅 2 个 OOM 不安全"表述不完整。同时发现 1 个未报告的**正面**点：cp2×sel_attn=1.003（B margin 在 cp 下亦成立）。
11. **DSA"上界"定性在大 S 下存疑**（reviewer-3）：被忽略的 indexer 自带 O(S²) scratch，长序列可能侵蚀 full-attn 余量。
12. 手写 builder 的 mindformers 行号注释已随源码更新漂移（机制仍对）；轻量 select 的 kept_frag 反直觉非单调（已文档化）；timeline 事件无 microbatch 序号；微批数硬编码=pp。

## 三、跨 reviewer 交叉确认（高置信）
- `qkv` 口径漂移与老式 recompute 静默丢弃由 **两个 reviewer 独立发现**；
- 全部 7 个 reviewer 各自实测复现锚点/退化端/等价性，无一数值失配；
- 全套 426 测试在各 reviewer 环境独立跑均绿。

## 四、总体评价
核心声明**大体成立且诚实文档化程度高**（多处"反向宣传"式披露边界）；主要风险集中在
**yaml 导入路径的静默近似/丢弃**（P0.2/3/4、P1.5/6/7）与 **per-stage select 的 K_CE 全局门**（P0.1）。
建议优先修 P0.1（k_ce 按层判定）与 P1.5/6（口径统一 + 老式列表 fail-loud），其余按需。
