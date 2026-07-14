# 对《最终代码检视报告》的逐条对抗审视与答复（2026-07-14）

> 被审对象：`analysis/final_code_review_2026-07-14.md`（专家模型检视，32 项：P0×5 / P1×19 / P2×8）。
> 方法：4 个并发只读核查 agent 对照 **mindformers pynative 源码**（本地 `E:\97-codes\torch_parallel\mindformers` @ 377c9c344 + 容器 deepseek_v4 fork @ 97466872d 双版本复核）、hyper_parallel @ cecf192、
> training_graph DSA 参考实现、MindSpeed FA 契约逐条核实；6 条 LOCAL-REPRODUCED 证据（`tests/test_review_evidence.py`，strict xfail）本地全部复现。
> 答复四态：**[认可-已修]** / **[认可-部分修]** / **[部分认可-已修]**（意见有误差但主体成立）/ **[有保留]**（意见部分不成立，给反证）/ **[认可-暂缓]**（属实但按边界文档化，非本轮修）。
> 修复状态与回归/真机数字见 §3；本节先给**核实裁定**。

## 1. P0 逐条裁定

### P0-01 累计梯度生命周期缺失 —— [认可-已修]
真机证据无可辩驳（两卡探针 7566.1−5676.6=1889.5 MiB ≡ 3778.9/2，算术自洽）；正确语义采纳检视的修正表述——**step-scoped cumulative**（反向后半段逐层累计驻留至 optimizer，zero_grad 释放），非跨 step 永久驻留。
对抗审视发现的**修复约束**（检视未提）：现有 optstep 桶的 K_OPT 常数当年是对真机 optimizer 峰标定的，而真机该峰**已含**累计梯度——直接加桶存在双计风险，必须先做 K_OPT 分解再加桶（详见 §3 修复设计与锚点回归）。

### P0-02 配置适配器静默丢失内存语义 —— [认可-已修，且比检视更严重]
核查确认 4 个键全部真实存在（`pynative/config/config.py:341-484`）且被静默丢弃。**对抗核查额外发现三处检视未覆盖的更深缺陷**：
1. adapter 读的 `num_layer_list`/`offset` 在 pynative 新式 schema 中是**幻影键**——真键是 `pipeline_parallel_layers_per_stage`（config.py:397，`"0-3,8-11|4-7,12-15"` 格式），且 `ParallelismConfig.allow_extra=False`（config.py:112-116）下含幻影键的 yaml 连 mindformers 自己都加载不了；
2. `data_parallel_replicate` 不是 yaml 键，是 trainer 运行时派生（`dp_replicate = data_parallel // data_parallel_shard`，trainer.py:449-454）；
3. `pipeline_parallel_microbatch_size` 的 yaml 值会被 trainer 覆盖为 `global_batch_size // (dp·local_batch_size)`（trainer.py:472-475）。
修复按核查产出的全键分类表做 fail-loud schema（一档 fail-loud / 二档警告 / 中性忽略集，各键注明 mindformers 消费点行号）。

### P0-03 reshard 死配置 —— [认可-已修（按核查修正后的语义接线）]
「未接线」属实（timeline 无条件 `gather_buf=0`）。但检视给的三态语义不完整，hook 级核查修正为（hyper_parallel `hsdp_scheduler.py:153-250`、`state.py:505-537`、`fsdp.py:42-74`、`parallelize.py:1172-1182`）：
- `always`：gather 驻留 = [fwd 入→fwd 出] ∪ [bwd re-gather→bwd 出]；
- `never` 与 `default+PP`：驻留 = [fwd 入→**本模块** post_backward] 连续段（**非整 step**——`reshard_after_backward` 默认 True）；
- `default`（非 PP）：同 always，**除** output_layer（恒 never 语义，parallelize.py:1176 显式传参）；norm 组与 root 残余组**不随 policy**（恒 always 语义，未传参走 API 默认）。
检视猜测的「root 可能特判」方向反了：root 组是恒 True（always 语义），output_layer 才是 never 语义。评估器按此接线（层残余组粒度，norm 组字节量可忽略、文档化）。

### P0-04 TP 下 vocab 栈未切分 —— [认可-已修]
双版本源码复核（377c9c344 + 容器 97466872d）：pynative TP>1 **无条件、无开关**走 embedding RowwiseParallel Shard(0)（parallelize.py:751-759）+ output_layer ColwiseParallel Shard(0) `gather_output=False`（:765-767）+ vocab-parallel CE（loss.py:326-336，max/sum-exp/target-logit 三个 [N,1] 项 all-reduce，全程不物化全量 logits）。
评估器 head.py:21 原注释引用的 `vocab_emb_dp` 语义**只存在于静态图 legacy 路径**（pynative 下零命中）——建模依据过时，检视方向完全正确。
一处措辞修正（不影响结论）：被 TP 切的是 loss 的**输入 logits**（N×V/tp）；per-token loss 向量 [N] 在 TP 内是复制的。
另证实 `enable_loss_parallel`（config.py:444-449）在 pynative 全库**无消费者**（死键，旧 DTensor 流残留）——即 vocab-parallel 行为无法关闭。12 个真机锚点全部 tp=1，修复不动锚点。

### P0-05 独立 DSA 省略 indexer KL loss —— [认可-已修（预估计），一处单位订正]
training_graph 参考实现核实：KL 的 QK bmm 产生 `[B, n_heads(主注意力=128), S, S]` fp32 瞬态（dsa_indexer_loss.py:118-120；n=config.num_attention_heads :62——检视写 128 **无误**，我们此前存疑方向错了）；B1/S4096 恰 **8.00 GiB**；tp（head 维）/cp（S 维）双切（:184,193）；q/k 均 stop_gradient（multi_latent_attention.py:308-310）→ 纯前向瞬态、零反向保存。sparse_loss 模式也先物化全量 bmm 再 gather（:120→:131），省不掉。
订正：评估器 dsa.py 旧注释「~8.6 GiB/层」是 **GB 误标 GiB**（8.59e9 B = 8.0 GiB）；检视报告的"8 GiB"反而正确。
另澄清（检视 §4 亦承认）：这是评估器**书面声明的有意省略**（dsa.py:44-50），非静默漏建——但既然影响可达 8 GiB/层量级，按检视意见改为显式建模（预估计口径，pynative 代码可用后重校准）。

## 2. P1/P2 关键项裁定（凡有出入处注明）

### P1-06 loss 与 DSA fusion 错误耦合 —— [部分认可-已修]
耦合确实要拆，但检视的前提「CE 有 fused/unfused 两态」**不成立**：pynative LM CE **恒为 unfused 小算子组合**（`_LogSoftmax`+`_NLLLoss` / `_VocabParallelCrossEntropy`，loss/loss.py 全文无 fused CE 开关；chunked 变体的 "fused backward" 是分块重算语义、由 `chunk_loss_num` 独立控制）。`apply_dsa_kernel_fusion`（transformer_config.py:2853，默认 True）只管 attention kernel 与 **DSA 自带 indexer loss** 的融合路径。
修法：`cross_entropy_fused` 与 `_dsa_fused` 解耦、pynative 恒 unfused；tp>1 → `loss_type=vocab_parallel_ce`（与 P0-04 联动）；`chunk_loss_num` 键接入。

### P1-09 FA softmax 统计量生命周期 —— [认可-已修]
核实属实且定量确认：softmax_max/sum 各 [B,N,S,8] fp32 是 FlashAttentionScoreGrad 输入（MindSpeed fusion_attention_v2.py:38-40 `save_for_backward` 佐证），须驻留 fwd→bwd。现状三点全中：①建成 fwd-only workspace ②lse 只有真值 1/32（shape 缺 ×8、dtype 2B）③workspace 只 ÷cp 不 ÷tp（TP=8 时反向高估 8×）。B1/N128/S4096 每无重算层欠 **31 MiB**。
补充发现：DSA 的 sparse_flash 已正确建 sfa_stats（SFA 契约无 ×8 内层块），但同时多挂了 FLASH_LSE_WS——修复时一并去除防双算。
锚点影响（预期物理效应）：no-recompute/keep-attn 族欠预测锚点（0.925/0.940/0.962/0.968）向 1.0 移动；full/select-both 族（saves 被重算丢弃）不变。

### P1-14 FSDP wrap 粒度 —— [认可-暂缓（边界已文档化）]
属实并拿到 wrap 单元全表（parallelize.py:992-1266）：experts（独立 efsdp mesh）、router（fp32 单独组）、embedding、每层 5 种 norm、final_layernorm、output_layer、MTP 各自独立 wrap；**但 attention 主体 + dense-FFN 仍在整层残余组**——「所有子模块独立」需修正为「router/experts/norm/output 独立，大头仍整层」。评估器当前按整层建 gather 的误差主要落在 experts 的 gather 时机（层中段而非层入口）与 reshard 粒度；本轮先在 P0-03 接线中把 output_layer 特判与 experts 独立组建进去，逐 norm 组的 gather 时间线（字节量 ~H 级）不建，文档化。

### P1-18 嵌套 offset 计算时机 —— [认可-已修]
`_materialize_nested_offset`：缺 `num_hidden_layers` 时暂存 `_nested_offset`，页面兜底注入后物化；仍无 N → 丢弃+警告，绝不静默错算（tests/test_nested_offset_deferred.py ×4）。

### 其余各项裁定摘要
| 项 | 裁定 | 一句话 |
|---|---|---|
| P1-01 router/norm 参数缺失 | 认可-已修 | router 权重真机为 **fp32 单独 wrap**（parallelize.py:1116-1128），补参数时连 dtype 一起修；参数守恒改逐模块对账 |
| P1-02/07 重算语义（exclude_op/零命中/full 空集/islands） | 认可-已修（exclude_op fail-loud；islands 暂缓文档化） | exclude_op「重算中保留」语义评估器暂不可表达→拒绝导入而非静默低估；checkpoint islands 一等对象化列入后续 |
| P1-03 张量身份冲突 | 认可-已修 | 重命名消歧 + resolve 期同名异 numel fail-loud 不变量 |
| P1-04 embedding dtype 死配置 | 认可-已修 | 接线并按真机 dtype 核定默认值 |
| P1-05 dsa 入口不可达 | 认可-已修 | `experimental_attention_variant=dsa` 放行 |
| P1-08 bwd_scratch 求和 | 有保留-暂缓 | 逐 op 求和是**有意的 OOM-安全上界**（保守方向），非低估缺陷；max-live 精化列入后续，量级评估后再动 |
| P1-10/11 输入校验缺失 | 认可-已修 | 统一 `_validate`：非法 ratio/floor head_dim/长度/整除全 fail-loud |
| P1-12 MoE 平均负载 | 认可-部分修 | 均值口径如实标注；capacity/skew 多口径报告列入后续（需真机不均衡路由差分实验支撑标定） |
| P1-13 CP buffer 缺口 | 认可-暂缓 | fused QKV 无法单独标 KV 的限制已文档化；ring/ulysses 双缓冲需 profiler 定位后建模 |
| P1-15 PP P2P buffer / 更多调度 | 认可-暂缓 | P2P recv 激活已隐含在首层输入 act_live；send buffer 与 overlap 双缓冲未建（量级 S·B·H 级/微批）；gpipe/zero-bubble 导入侧 fail-loud |
| P1-16 可行性校验缺失 | 认可-已修 | PP>1+swap 拒绝；tp>1 强制 sequence_parallel（config.py:471-477） |
| P1-17 UI 非保真 round-trip | 认可-已修（措辞+警告） | README 收窄为「回填 UI 支持子集」；固定假设（64GiB/AdamW-fp32/swap 关）作为 warning 返回 |
| P1-19 优化器单一/K_OPT 固定 | 认可-部分修 | P0-01 修复自带 K_OPT 分解（grad 显式化）；Adam 之外的优化器 fail-loud 列入后续 |
| P2-01 .oom 口径 | 认可-已修（口径拆分文档化） | allocated 与 reserved 判定分开报告；真机 reserved−allocated ≈658-680 MiB 已记录 |
| P2-02 opdag 未进主链路 | 认可-暂缓 | 交叉验证定位是当期设计（T13 收尾后评估作为 LayerSpec 校验器接入） |
| P2-03 结构覆盖缺口 | 认可-部分修 | final norm 参数补入（P1-01）；dropout=0（本模型族配置）与压缩 mask 中性性已注明依据 |
| P2-04 标定常数可移植性 | 认可 | P1-09 修复把 no-recompute 族欠预测的**物理成分**（fa_stats 驻留）从 K_CE 经验常数中剥出——常数覆盖面随修复缩小，方向正确；跨模型稳定性结论维持检视原文 |
| P2-05 预设可信度 | 认可 | UI 已标"预估计"；全尺寸外推定位不变，措辞维持 |
| P2-06 timeline 无 microbatch 标识 | 认可-已修 | 事件名加 microbatch/chunk 序号 |
| P2-07 测试固化风险 | 认可-部分修 | 本轮新增反例向测试（xfail→正向翻绿制）；参数守恒改逐模块精确对账 |
| P2-08 文档漂移 | 认可-已修 | StaticMem docstring/K_CE/CP 注释对齐实现 |

## 3. 修复实施、回归与真机验证

### 3.1 修复提交清单（4 commit，全部推送前先本轮回归）
| commit | 内容 | 关键文件 |
|---|---|---|
| `5c247b1` | P0-01 grad_accum 桶（K_OPT 6→4 重标）+ P0-03 reshard 三态接线 + P0-04 TP vocab 栈切分 + P0-05 DSA KL 主项 + P1-09 fa_stats + P1-04 dtype 接线 + P1-18 offset 延迟物化 | mem_timeline/structure_mem/shape_eval/model_spec/layers/head+attention+dsa/llm_config |
| `f82fbf5` | P1-01 router/norm gamma/final_norm 参数入图 + P1-03 张量身份重命名 + resolve 同名异形不变量 | layers/ffn+attention+dsa+dsv4_hybrid+head+residual、shape_eval |
| `5e9bd31` | P0-02 parallelism/swap/recompute fail-loud schema + P1-02/05/06/10/11/16/17 + P2-01/08 文档 | configs/from_mindformers、build_llm、report、serve_explorer、README |
| （本 commit） | 逐条答复文档 + 真机验证记录 | analysis/ |

### 3.2 12 锚点 sim-vs-real（修前 → 修后）
| 锚点 | 修前 ratio | 修后 ratio | 主因 |
|---|---|---|---|
| DSv3 4L full | 0.995 | **0.997** | +router/norm/final_norm 参数 |
| DSv3 8L full | 0.991 | **0.993** | 同上 |
| DSv3 4L full ep2 | 0.991 | **0.994** | 同上 |
| cp2 colossal full | 0.998 | **1.000** | 同上 |
| cp2 ulysses full | 0.997 | **1.000** | 同上 |
| pp2-stage0 (optstep) | 1.053 | **1.057** | grad_accum + K_OPT 重标（净微增，仍 OOM 安全） |
| **pp2-stage1 (loss)** | 0.962 | **1.002** | **grad_accum 显式化（检视 P0-01 靶心，欠预测消除）** |
| cp2-none (loss) | 0.925 | 0.927 | fa_stats/参数微增（仍 <0.95，见 §4 残留） |
| select self_attn | 1.001 | 1.002 | fa_stats keep 层 |
| select mlp | 0.940 | 0.943 | fa_stats attn 保留层 |
| select both (=full) | 1.001 | 1.003 | 参数增 |
| DSv4-fused | 0.968 | 0.968 | loss 区主导，累计梯度非峰 |
| **汇总** | 平均 2.4% | **平均 2.0%、中位 0.998** | — |

### 3.3 测试
- 评估器全套 **475 passed**（修前 455）；新增 `test_nested_offset_deferred`(4)、`test_structure_validation`(9)、`test_parallel_adapter_rejects_unknown_and_unmodeled_keys`(1)。
- `tests/test_review_evidence.py`：6 条 strict-xfail **全部翻绿转正向断言**（P0-01 grad_accum / P0-02 adapter round-trip + 拒绝 / P0-03 reshard / P0-04 TP 减半 / P1-01 router / P1-03 张量身份）。

### 3.4 真机复测（192.168.9.116 容器 shb.ms.2.9，2×910B3 卡 6/7，MindSpore 2.10，deepseek_v4 fork @ 97466872d）
**Job A — DSv4 hybrid 4L FSDP-2 锚点 + 累计梯度探针复跑**（修后评估器 `9f2372c`+本轮改动）：
```
optimizer 前 grad_bytes=3778.9 MiB  current_alloc=7566.1
zero_grad 后 current_alloc=5676.6   → 差 1889.5 MiB ≡ grad 3778.9/2（FSDP-2）
峰值 allocated=15415.5  reserved=16074.0（framework 658.5）
```
→ P0-01 累计梯度公式（grad_accum = Σ shard，DSv4 =1887.6 预测 vs 1889.5 真机 = 0.999）**真机再确认**；且梯度采样含 `input_layernorm.weight (1792,) Float32` → 佐证 P1-01 norm gamma 是 fp32 独立参数（此前欠建）。DSv4 总峰锚点 15415.5 不变（评估器 14929.8 → 0.968，累计梯度非该配置的全局峰）。

**Job B — TP=2 vocab 栈差分**（P0-04）：见 §3.5。

### 3.5 TP=2 vocab 栈真机结果（P0-04 决定性确认）
DSv4 4L TP=2，DTensor 分片探针（`param.shape` 报逻辑全局 shape，故读 `to_local()`/placements）：
```
embedding.word_embeddings.weight: global (129280,1792), placements (StridedShard(dim=0,split=2), Shard(dim=0)),
                                  local_shape (64640,1792)  local_numel 115834880
output_layer.weight:              global (129280,1792), placements (StridedShard(dim=0,split=2), Shard(dim=0)),
                                  local_shape (64640,1792)  local_numel 115834880
```
- 两权重均 **`Shard(dim=0)`（vocab 维）÷tp**，local_numel = 115834880 = 全量 231669760 / 2 ——
  **与检视报告预测的"期望值 115834880"逐字节吻合**；`StridedShard(split=2)` 是其上叠的 FSDP 层。
- 日志明确打印 `vocab_emb_dp is not supported in MCore, converted to False`——印证 `vocab_emb_dp`
  是静态图 legacy 键、pynative/MCore 路径忽略它（旧建模依据过时，检视方向正确）。
- **评估器修后同口径逐字节一致**：`tp=1 → emb_w/head_w local_numel 231669760；tp=2 → 115834880`。
- 注：TP=2 训练步因 fork 的 DSA+TP kernel 自身 bug（`int - tuple`，与 vocab 栈无关）崩溃，故未取
  TP=2 运行时峰值；但**权重分片这一 P0-04 靶心已由 DTensor placements 决定性确认**（init 即定，
  不依赖训练步）。

## 4. 未修项与残留（诚实边界）

以下项经核实**属实但按边界文档化、非本轮修**（连同检视 §8「修复顺序」的二/三阶段）：
- **P1-08 bwd_scratch 逐 op 求和**：有意的 OOM-安全上界（保守方向），非低估缺陷；max-live 精化列入后续。
- **P1-12 MoE 平均负载**：均值口径已注明；capacity/skew 多口径需真机不均衡路由差分标定。
- **P1-13 CP kernel buffer / P1-14 子模块 wrap 时间线 / P1-15 P2P send buffer & 更多调度**：
  已文档化限制；需 profiler 定位量级后建模。experts/router/output_layer 独立 wrap 的 gather
  时机差异（层中段 vs 层入口）本轮未细分，字节量级 ~experts 单组、方向保守。
- **P1-19 非 Adam 优化器 / 分离 offload**：Adam 之外 fail-loud（未静默按 AdamW 近似）列入后续。
- **P2-02 opdag 未进主链路**：交叉验证定位是当期设计（T13 收尾后作为 LayerSpec 校验器接入）。
- **P2-04/05 标定常数可移植性 / 全尺寸外推**：kept_frag=1.9、K_CE 覆盖面随 P1-09 物理成分剥离而
  缩小（no-recompute 族欠预测的 fa_stats 分量已从经验常数拆出）；跨模型稳定性结论维持检视原文。
- **cp2-none 0.927 / select-mlp 0.943 仍 <0.95**：属检视 §5.3 已归因的 no-recompute CE-lifetime
  残差族；本轮 fa_stats 修复使其向 1.0 移动但未越 0.95——根因是 unfused CE 链 K_CE 经验常数的
  标定余量，需专门 profiler 逐份计数（P2-04 范畴），非结构缺陷。

## 5. 逐条覆盖核对（32 项）
P0-01..05 全修（§1）；P1-01..07/09/10/11/16/17/18 已修，P1-08/12/13/14/15/19 暂缓（§4，附依据）；
P2-01/06/07/08 已修（口径文档 + microbatch 标识 + 反例向测试 + 注释对齐），P2-02/03/04/05 部分修/维持
（§4）。**无一条未答复**。
