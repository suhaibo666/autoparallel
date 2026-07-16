# 功能缺失根因分析 —— "是设计问题还是什么原因？"（2026-07-16）

> 范围：`test_engineering_report_round2_2026-07-16.md §4` 点名的三个功能缺失族——
> **未建模残差 G2–G7**、**fail-loud 未实现 N1–N9**、**验证盲区（真机栈限制）**。
> 方法：**逐项定位真实代码/源码证据**（每条断言带 `file:line`），把根因归入单一主桶，判"未实现是否正当"。
> 立场：源忠实——报告本身的措辞若与代码不符则显式订正（本轮订正 2 处：**N9 实为静默近似而非 fail-loud**、
> **N7 为"raise + 布尔近似"混合体**）。fail-loud 拒绝真实性经**可运行探针**实证（见 §4）。

---

## 0. TL;DR（直接回答用户）

**绝大多数功能缺失不是设计缺陷，而是一个"源忠实 + fail-loud + 离线 + 真机栈受限"评估器的预期形状。**

按主根因分桶（20 项，含验证盲区 5 域）：

| 主根因桶 | 项数 | 占比 | 一句话性质 |
|---|---|---|---|
| ① 有意 fail-loud（拒绝猜、不产错数） | 7 | 35% | 改 op 图/持久态但只建了一种取值 → 显式 raise，正确设计 |
| ② 外部阻塞·无上游源码 | 1.5 | ~8% | 调度器在外部包 `hyper_parallel` 本地无源（G6；N9 部分） |
| ③ 外部阻塞·真机栈不支持/不可跑 → 无锚点 | 6 | 30% | 真机自己也拒（N8）或跑不起来（5 盲区域） |
| ④ 根本建模极限（op 图粒度之下） | 1 | 5% | G5：313 个 <100MiB 碎片长尾，源码级证实不可显式导出 |
| ⑤ 可建模但推迟（off-peak / opt-in / 已量化无害） | 4 | 20% | G2/G3/G4/G7，均有 caveat + 量化理由 |
| ⑥ 遗漏 / 真缺陷（应做而没做、无正当理由） | **0** | **0%** | 无一项属纯遗漏 |

**判定**：这些缺口**不构成设计质量问题**。N 族体现"fail-loud > 静默错"的正确取舍（宁可拒绝也不产"貌似合理实则错"
的数）；G 族是已文档化的 off-peak / opt-in 残差或一个真实的 op 图粒度物理极限（G5）；验证盲区是真机栈解锁前的
**锚点可得性**问题、非仿真器建模缺陷。**唯一需要修正的是报告的分类措辞**（N9 被误列为 fail-loud，实为静默近似）。
弱正当的推迟有 3 处需诚实点名（G3 迁移面、N4 硬拒零量级 bias、N7 布尔近似是 D2 欠预测藏身处），但均已在别处留档，非未被察觉的遗漏。

---

## 1. 逐项根因分类表

图例：主桶①–⑥同上；**正当？** = 不实现是否合理（是/部分/否）。

| 项 | 主桶 | 一句话证据（file:line） | 正当？ |
|---|---|---|---|
| **N1** post/sandwich norm | ① | `build_llm.py:200-203` `norm_placement!='pre'` → `NotImplementedError`（探针实证 raise） | 是 |
| **N2** 非 RMSNorm（LayerNorm） | ① | `build_llm.py:204-206` raise；LN 多存 mean/var + fp32 残差，静默套 RMSNorm 图=错数 | 是 |
| **N3** learned_absolute/none 位置编码 | ① | `build_llm.py:207-210` raise；learned 需额外 pos 表 param、none 需删 rope op | 是 |
| **N4** add_bias_linear/add_qkv_bias | ① | `build_llm.py:212-219` raise；量级可忽略，escape hatch 见 `presets.py:215-221`(qwen2 False+注释) | **部分**（可降级为 warning） |
| **N5** qk_layernorm 于 mla/dsv4/dsa | ① | `build_llm.py:226-231` raise；**gqa/mha 已建则放行**（`attention.py build_gqa_attn_ops`），仅这三种 builder 没建 q/k norm op 时拒 | 是（精准外科拒绝） |
| **N6** 非 Adam 优化器 | ① | `report.py:91-96 feasibility_errors`；K_OPT/state/optstep 全自 AdamW 链导出，SGD 会全错（探针实证 1 err） | 是 |
| **N7** loss_type 三种之外 + fused-CE 布尔近似 | ①+⑤ | `head.py:55-57` raise 未知 loss_type（探针实证）；但 `cross_entropy_fused` 是**布尔**（`mem_timeline.py:471-473` 置空 loss_lids）**非显式 op 图** | 是（raise 侧）/ 部分（布尔侧=D2 藏身处） |
| **N8** PP>1 + activation swap | ③ | `report.py:76-79 feasibility_errors`（真机 `activation_checkpoint.py:898-900` 直接 raise）→ **忠实拒绝**（探针实证 swap-hit） | 是 |
| **N9** 显式 per-chunk VPP ranges | ②+⑤ | **`parallel_model.py:112-154` 不 raise、静默"连续均衡切 v 段"文档化近似**（`:118`,`:129`）——报告把它列 fail-loud **是措辞错** | 部分（近似正当、标签错） |
| **G2** ring/ulysses CP 在飞双缓冲 | ⑤ | `attention.py:106-116` off loss 峰；async 开关 `from_mindformers.py:575 _PAR_UNSUPPORTED_TRUTHY` fail-loud | 是（off-peak） |
| **G3** GQA colossal KV all-gather buffer | ⑤+③ | 已建公式 `attention.py:39-50`，**opt-in 默认关**（`model_spec.py:44 cp_kv_allgather_buffer=False`，`shape_eval.py:255,263-265` 三重门）；cp2 精确减半冻结测试代数禁其非零（`attention.py:113-115`）→ 只能 opt-in；真机跑不了 cp+无重算故未验 | 是（但迁移未验，见 §3） |
| **G4** backward grad-P2P | ⑤ | `mem_timeline.py:530-531`：量级 [S,B,H]，**已量化落在 pp2-stage0 现有过预测余量内**，OOM 安全、文档化保守残差 | 是（量化无害） |
| **G5** 无重算-MoE 碎片长尾（=D1 根因） | ④ | `opdag_validation.md §3-4`：313 个 <100MiB 碎片（3666 MiB），源码级 DAG 提取证实**在 op 图粒度之下、不可 construct() 导出**；D1 margin 只覆 OOM 方向非物理 | 是（真实物理极限） |
| **G6** VPP m==pp 调度特例 | ② | `mem_timeline.py:45-50`：mindformers pynative 调度器在**外部包** `hyper_parallel.core.pipeline_parallel`（本地无源，`pipeline_parallel.py:282-284,310` 惰性 import 证实），经 mindformers 侧确认"非全前向先行"→ 有意不移植 Megatron 特例、统一走通式 warmup | 是（无源下保守） |
| **G7** MoE 真实倾斜分布 | ⑤+④ | `ffn.py:35-50`+`build_llm.py:125-142`：三口径 balanced/capacity/skew，真实 skew 因子**须用户给**（`moe_skew_factor≥1`）——离线不可知路由倾斜 | 是（离线信息极限 + 给保守口径） |
| **盲区** TP>1 激活 | ③ | 报告 §6：SP+MoE 不支持 / TP+MoE Detach layout bug / DSA+TP fork kernel bug（`int - tuple`）→ 真机无可跑路径 | N/A（非缺陷） |
| **盲区** pp>2 / VPP | ③ | `pp_select_vpp_validation.md:23,65`：pp 仅 `NORECOMP=1` 可跑；run_axis.sh 无 INTERLEAVE env → VPP 不可驱动；pynative AdamW 中间 stage param/state 配对崩 | N/A |
| **盲区** PP×重算 | ③ | `pp_select_vpp_validation.md:23`：本 build 互斥（NORECOMP=1 才能跑 PP） | N/A |
| **盲区** DSA 全域 | ③ | `dsa.py:3,52,83` 全标**预估计**（基于 training_graph 静态图，无 pynative DSA 可跑路径）；`build_llm.py:238` 缺维 fail-loud | N/A（模型自标预估计） |
| **盲区** swap 真机曲线 | ③ | 报告 §6：未采（评估器侧有解析用例）；N8 更进一步真机根本不支持 PP+swap | N/A |

---

## 2. 分桶详解（为什么归此桶）

### ① 有意 fail-loud（N1–N7，7 项）—— 正确设计取舍
统一哲学写在 `build_llm.py:188-197 _check_implemented_dispatch` docstring：对**改变 op 图**但只建了单一取值的分派字段，
"不静默按已实现取值继续（会产『貌似合理实则错误』的图）"。这是源忠实分析的核心原则在评估器里的落地——
**宁可拒绝，也不产一个偏移的数字骗过 OOM 判断**。判据的正确性：
- N2/N3 会**实打实改字节**（LayerNorm 多 fp32 残差、learned pos 表是持久 param）→ 静默套错图 = 错数，**必须拒**。
- N5 是**精准外科拒绝**：gqa/mha 的 builder 确实建了 q/k RMSNorm op（`build_llm.py:220-225` 放行），只对
  mla/dsv4/dsa 这三种没建的 builder 拒——不是一刀切，证明 fail-loud 是**按能力边界精确画线**的。
- N6 非 Adam：K_OPT/optstep 瞬态与 persistent 全从 AdamW op 链导出（`report.py:48-49,91-96`），SGD 无动量态、
  Adam-mini 等状态形态全异 → 按 AdamW 近似会**全错**，拒绝正确。
- **弱点（N4）**：`add_qkv_bias`/`add_bias_linear` 量级可忽略（`presets.py:216` 自承"显存量级可忽略"），却硬 raise。
  论据：与"设了却被静默忽略"的反模式一致对待、且给了 escape hatch（`presets.py:219` False+注释）。**可辩护但偏严**——
  对一个内存中性字段，降级为 warning 亦合理。记为**部分正当**。

### ② 外部阻塞·无上游源码（G6；N9 部分）
- **G6**：`mem_timeline.py:47` 明写"mindformers pynative 的调度器在 `hyper_parallel.core.pipeline_parallel`
  （外部库,本地无源码）"。已核实：`mindformers/pynative/distributed/pipeline_parallel.py:282-284` 把三种
  schedule 类（`Schedule1F1B/ScheduleInterleaved1F1B/ScheduleGPipe`）指向该外部包，`:310` 惰性 import——本地树
  **确无**该调度器实现。故 m==pp 的 all-warmup 与否无法从源确证，评估器保守走通式 warmup（`interleaved_warmup`
  `mem_timeline.py:51-79`，锚定 Megatron 公式）→ **无源下的忠实保守**，正当。
- **N9 订正**：报告 §5 把 N9 列入"fail-loud NotImplemented（拒绝评估、不产错数）"，**但代码不 raise**——
  `parallel_model.stage_chunks` 对 `layers_per_stage + interleave` 组合**静默**走"stage 内连续均衡切 v 段"
  文档化近似（`:118`,`:129`）。**它产一个近似数字、不拒绝**。故 N9 应归 G 族（已文档化近似），非 N 族（fail-loud）。
  另一半事实：mindformers 的 per-chunk range **配置格式其实本地可见**（`pipeline_parallel.py:151-178` 每 rank
  逗号分隔 range 解析），评估器默认路径也实现了**有源**的 round-robin（`:257-258 chunk_id*pp+rank`，见
  `mem_timeline.py:98-100`）——只在"显式 layers_per_stage + 交错"这个**罕见组合**上近似。故 N9 = 外部（外部调度器
  warmup 精确语义未确证）+ 推迟（罕见组合，⑤）的混合，近似本身正当，**唯标签需订正**。

### ③ 外部阻塞·真机栈不支持 → 无锚点（N8 + 5 盲区，6 项）
关键区分：**这些不是"没建模"，是"建了但无真机锚点可验"**（报告 §6 明确：由解析用例 + Megatron/mindformers 逐行
port 锚定）。真机栈（116, MS2.10）本身跑不起来这些配置：
- N8 是**忠实拒绝**：真机 `activation_checkpoint.py:898-900` 自己 raise PP+swap → 评估器 fail-loud 是**复刻真机行为**，
  不是能力缺失。
- 盲区 5 域各有真机侧硬阻塞（TP+MoE layout bug / DSA+TP fork kernel `int-tuple` bug / pp 仅 NORECOMP=1 /
  pynative 无 DSA 可跑路径 / swap 曲线未采），全在报告 §6 与 `pp_select_vpp_validation.md` 有据。DSA 更由模型
  **自标"预估计"**（`dsa.py:3,52`）诚实降级。**这是环境约束，非设计缺陷。**

### ④ 根本建模极限（G5，1 项）
`opdag_validation.md` 是本桶的铁证：§2 用源码级 DAG 提取**字节级交叉验证**手写 builder（grouped-GEMM 56/64/32
逐字节吻合），§3 定位 18% 残差 = **313 个 <100MiB 小张量长尾**（fp32 cast 横切 / permute 碎片 / 分配器保留），
§4 硬结论"这些的 loss 峰存活性是分配器/自动微分保留的现实,不是显式 construct() op 能干净导出的",
"即『扩提取器消 margin』也无法完全闭合"。D1 的 `nr_moe_frag_factor` margin（`mem_timeline.py:763-775`）
**明示为 2 点标定、非物理**，只 gate 到 OOM 方向。**这是模型 op 图粒度的真实物理极限，不是能补的漏。**

### ⑤ 可建模但推迟（G2/G3/G4/G7，4 项）—— 均有量化理由
- **G2**（CP 在飞双缓冲）：`attention.py:99,106-116` off loss 峰（OOM 发生在 BWD/loss 峰，此 buffer 是 FWD 侧在飞）。
  推迟正当（不动峰）；对应 async 开关仍 fail-loud（`from_mindformers.py:575`）防静默错。
- **G3**（colossal KV all-gather buffer）：已建（`attention.py:39-50`），opt-in 默认关是因为 cp2 精确减半冻结锚点
  **代数上禁止**非零 colossal buffer（`attention.py:113-115`），默认开会破锚点。推迟到 opt-in 正当。
- **G4**（backward grad-P2P）：`mem_timeline.py:530-531` **已量化**落在 pp2-stage0 现有 ~580MiB 过预测余量内 →
  OOM 安全。这是"推迟正当且证明了它不动峰"的范例。
- **G7**（MoE 真实倾斜）：真实 skew 是运行时路由/数据依赖，**离线根本不可知**（④的信息极限成分）；评估器的正确回应
  是给 balanced（吞吐）/capacity/skew 三个 OOM 边界口径 + 用户 skew 因子旋钮（`ffn.py:35-50`）。正当。

### ⑥ 遗漏 / 真缺陷 —— 0 项
逐项排查后**未发现纯遗漏**。最接近的三处均已在别处留档、非未察觉：N9 的标签错（本文订正）、N4 的偏严（部分正当）、
N7 布尔近似藏着 D2 的 0.920 欠预测（已作为 D2 独立留档、门控）。无一条属"应做、没做、且无人知道"。

---

## 3. 诚实点名：3 处弱正当的推迟（虽正当但值得盯）

1. **G3 迁移面 > 标定面**（与 `report_round2 §D1-R` 一致）：`from_mindformers` 对**任意** MoE 非 fused 模型注入
   margin，但 colossal KV buffer 公式只在 GQA 结构上代数验证过、opt-in 且**从未真机验**。它不动当前锚点（默认关），
   但产品口径若默认开需先补一个 GQA+cp 真机点。
2. **N4 硬拒零量级 bias**：对内存中性字段用 `NotImplementedError` 偏严；一致性上可辩护，但降级 warning 亦无损 OOM 安全。
3. **N7 fused-CE 布尔近似 = D2 欠预测藏身处**：`cross_entropy_fused=True` 时 `loss_lids` 置空（`mem_timeline.py:471-473`）
   → D1 的碎片 margin 不覆盖它（`mem_timeline.py:772` 明注）→ mHC+MTP 停在 **0.920 OOM 不安全**（D2）。
   这是**唯一一处"文档化近似当前带来 OOM 不安全后果"**的地方，虽已作 D2 门控留档，是三族里最该优先补的一块
   （补法见 `report_round2 §2 建议路径`：MTP 段显式 saves 或 fused-CE 单独标定碎片系数）。

---

## 4. fail-loud 真实性验证（可运行探针，非仅文档）

对代表性 N 项**实际喂违规 config 并断言 raise**（`cost_eval` 直连 API，`deepseek_v3(4)` 为基）：

| 探针 | 结果 |
|---|---|
| N1 `norm_placement='post'` / `'sandwich'` | `NotImplementedError`（both）✅ |
| N2 `normalization='LayerNorm'` | `NotImplementedError` ✅ |
| N3 `position_embedding_type='learned_absolute'` / `'none'` | `NotImplementedError`（both）✅ |
| N4 `add_bias_linear=True` / `add_qkv_bias=True` | `NotImplementedError`（both）✅ |
| N5 `qk_layernorm=True`（base attn_type=mla） | `NotImplementedError` ✅ |
| N6 `OptimizerSpec(type='SGD')` | `feasibility_errors` 返 1 err ✅ |
| N7 `loss_type='focal'` | `NotImplementedError` ✅ |
| N8 `ParallelConfig(pp=2)` + `SwapSpec(enable=True)` | `feasibility_errors` swap-hit ✅ |

**8/8 代表性 fail-loud 均实证触发**——拒绝是真的、不是仅文档承诺。（N9 反向验证：`stage_chunks` 对
`layers_per_stage+interleave` **不** raise，产近似数字——印证 §2 的标签订正。）

---

## 5. 结论

**回答"是设计问题还是什么原因？"：**

**不是设计质量问题。** 20 项功能缺失里——
- **35% 是有意 fail-loud**（N1–N7）：一个源忠实评估器面对"改 op 图/持久态但只建了一种取值"的正确反应就是
  拒绝而非猜，实证 8/8 真拒。这是**设计优点**不是缺陷。
- **~38% 是外部阻塞**（②③：G6/N9-部分/N8 + 5 盲区）：上游调度器在外部包无源、或真机栈自己跑不起来/也拒 →
  是**环境与锚点可得性约束**，非建模能力缺失；评估器已用解析用例 + 逐行 port 兜底。
- **~5% 是根本极限**（G5）：op 图粒度之下的 313 碎片长尾，源码级 DAG **证明**不可显式导出，只能标定 margin 覆 OOM 方向。
- **~20% 是已量化的 off-peak/opt-in 推迟**（G2/G3/G4/G7）：每项都有 caveat + 为何不动峰的量化论证。
- **0% 是纯遗漏。**

**总判定**：这些缺口是一个 **"源忠实 + fail-loud + 离线 + 真机栈受限"** 评估器的**预期形状**，
而非设计缺陷清单。真正需要动的只有两类：(a) 报告的**分类措辞订正**——N9 是静默文档化近似、不是 fail-loud
（N7 是 raise+布尔近似混合）；(b) 唯一带 OOM 不安全后果的文档化近似——**N7 布尔 fused-CE 下的 D2（mHC+MTP 0.920）**，
应作为 backlog 优先补（其余 G/N 项均属"正确地被拒/被 caveat"，无需当作缺陷去填）。

*证据：`build_llm.py:188-238`、`report.py:16-97`、`head.py:31-57`、`parallel_model.py:112-154`、
`mem_timeline.py:45-104/471-473/530-531/579-775`、`attention.py:39-116`、`ffn.py:35-50`、`shape_eval.py:254-265`、
`opdag_validation.md §2-4`、mindformers `pipeline_parallel.py:151-178/257-258/282-310`、`dsa.py:3-83`；
8/8 fail-loud 可运行探针实证。*
