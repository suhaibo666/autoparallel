# 内存仿真器缺陷与功能缺失综合报告（测试工程轮，2026-07-16）

> 范围：`cost_eval` 内存仿真全链路（结构建模 → 并行切分 → 静态/时间线 → 报告口径）。
> 证据：本地 + 116 容器双侧 **1159 用例全绿**（含本轮新增 250 条手工解析用例）；
> **14 个真机锚点**（13 记分卡 + 1 记分卡外）；116 真机源码 grep 定性 + 新采无重算探针；
> z2 交叉校验对 live 分支（feature/pynative-arch-evolution）0 漂移 + 变异探针。

---

## 0. TL;DR

机理正确性经解析级用例与真机双重验证成立：full 重算 / select 退化端 / PP loss 峰 / CP / EP
诸域全部落在 ±2% 内。**剩余缺陷收敛为一个物理域 + 一个防护域**：

1. **无重算-MoE 保留态欠预测**（OOM 不安全方向）——3 个锚点 0.920–0.937，每点缺口 ~1.3–1.7 GiB，
   根因是 op 图粒度之下的 fp32-cast/dispatch 碎片（F1 修复已回收其中可显式化的 ~196 MiB）。
2. **z2 交叉校验拦不住层级 norm 类缺口**（变异探针实证）——F1 这类 bug 若回潮，现有防护网不报警。

其余为文档化的功能缺失（fail-loud，不产生错误数字）与真机栈限制导致的验证盲区。

### 风险矩阵

| 级别 | 编号 | 问题 | 方向 | 量化 |
|---|---|---|---|---|
| ~~P1~~ 已修 | F1 | MoE decoder 缺 pre_mlp_layernorm | 欠 | +196 MiB@8L，`37f8e3d` 修复并复核 ✅ |
| **P1** | D1 | 无重算-MoE 保留态激活尾欠计 | **欠（OOM 不安全）** | 8L-none **0.931**（−1371 MiB）、cp2-none **0.937**（−1267 MiB） |
| **P1** | D2 | mHC+MTP 锚点欠预测且**不在记分卡** | **欠（OOM 不安全）** | **0.920**（sim 19466.7 vs real 21153.1，DSv4 4L） |
| **P2** | Z1 | z2 交叉校验盲区：不查层级 norm 名册 | 防护缺失 | 变异探针：删 ln2 后 `ok=True, 0 findings` |
| P2 | D3 | pp2-stage0 optstep 过预测 | 过（安全向） | 1.057 → F1 后 **1.089** |
| P2 | Z2 | 13 锚点记分卡是脚本非测试，不入 pytest 门 | 防护缺失 | `sim_vs_real_report.py` 手动跑 |
| P2 | G1 | DSA 内存模型无真机锚点（预估计） | 未知 | 仅 `dsa_peak ≥ mla_peak` 断言 |
| P3 | G2–G7 | 未建模残差/单点标定常数（见 §4） | 已文档化 | — |
| P3 | N1–N9 | 未实现功能（fail-loud，见 §5） | 拒绝评估 | — |

---

## 1. 证据基线

| 证据 | 结果 |
|---|---|
| 全套件（本地 Windows / 116 容器 mindspore2.10） | **1159 passed** 双侧一致 |
| 本轮新增解析用例 | `test_memval_hybrid_recompute.py` 25 条（TP×CP 逐字节、tp2·ep2·dp2·pp2 端到端持久/grad_accum/optstep 精确字节、VPP×full/select 重算、9 配置集群参数守恒）+ `test_memval_structure_matrix.py` 225 条（9 结构 × 6 并行 × 3 重算不变量矩阵：会计闭合/非负/峰值一致/重算偏序/守恒） |
| z2 源码交叉校验（116 live 分支） | mla_attn + moe_experts 两族 **0 漂移**（注意 §3 盲区） |
| 真机新锚点 | DSv3 8L dp2 无重算 = **19967.3 MiB**（卡 6/7，两 rank 一致；并证实旧 "feed_forward 19967" 错名数据实为真无重算测量） |
| F1 源码定性 | `gpt_layer_specs.py:110/:129` MLA/GQA 分支均无条件绑定 `pre_mlp_layernorm=RMSNorm`；`transformer_layer.py:126/:198` 平级 cell 先 norm 再 mlp |

### 记分卡现状（F1 修复后，13 锚点 + 1 卡外）

| 域 | 锚点 | ratio | 判定 |
|---|---|---|---|
| full 重算（4L/8L/ep2/cp2-colossal/cp2-ulysses） | 5 | 0.993–1.000 | ✅ |
| select（attn / both 退化端） | 2 | 1.001 / 1.018 | ✅ |
| select-mlp（keep-attn） | 1 | 0.955 | ✅（F1 后进安全带） |
| pp2（optstep / loss k_ce=8） | 2 | 1.089 / 1.007 | ✅（stage0 保守） |
| DSv4-fused base | 1 | 0.971 | ≈ |
| **无重算-MoE**（8L-none / cp2-none） | 2 | **0.931 / 0.937** | ⚠️ D1 |
| **mHC+MTP（记分卡外）** | 1 | **0.920** | ⚠️ D2 |

汇总（13 卡内）：平均 |ratio−1| = 2.6%，±5% 命中 10/13，中位 0.997。

---

## 2. 缺陷清单（正确性）

### F1（已修复，留档）：MoE decoder 缺 pre-FFN norm

- **现象**：dense FFN 段内嵌 ln2，MoE FFN 段（`build_moe_ffn_ops`/`build_shared_expert_ops`）直接吃裸
  `h1`——每 MoE 层漏一份 `[S,B,H]` fp32-cast 常驻（norm_compute=fp32）+ [H] gamma 参数。
- **真机确证**：`gpt_layer_specs.py:110/:129`（pre_mlp_layernorm 对 dense/MoE 统一绑定 RMSNorm）。
- **修复**（`37f8e3d`）：新 `layers/transformer.py build_transformer_layer` 统一前插 ln2，5 个装配点收口；
  kept_frag_factor 1.9→1.6 复标防双计。
- **复核**：select-mlp 锚点 +196.1 MiB 与诊断值逐位吻合；full 域 5 锚点逐字节不动；1159 全绿。

### D1（P1，未修）：无重算-MoE 保留态激活尾欠计 ~1.3 GiB/锚点

- **锚点**：DSv3 8L dp2 none **0.931**（sim 18596.5 vs real 19967.3）；cp2-none **0.937**（18852.5 vs 20119.4）。
- **根因**（排除 F1 后的剩余）：无重算下 MoE 保留态的 dispatch/permute/grouped-GEMM fp32-cast 横切 +
  小张量长尾——profiler live-set 313 个 <100 MiB 碎片，**在 op 图粒度之下**（opdag_validation.md），
  非显式 op 可导出。
- **为何现在没有补偿**：`kept_frag_factor` margin 刻意只 gate 到 select-keep-FFN（防与 K_CE fat 双算），
  no-recompute 域被明确排除（mem_timeline `_is_kept` 注释）——该域于是既无 margin 也无 K_CE 之外的补偿。
- **建议**：二选一——(a) 把 kept_frag 类 margin 扩展到 no-recompute 域，与 K_CE 解耦后用
  8L-none + cp2-none 两点联合标定（两点同族，可互为留出验证）；(b) 对 profiler live-set 再挖一轮
  可显式化项（如 grouped-GEMM 输入 fp32 cast 是否可建为 op saves）。

### D2（P1，未修）：mHC+MTP 锚点 0.920 且不在记分卡

- **现状**：DSv4 4L + mHC(×4) + MTP 真机 21153.1（2026-07-01 采），当前仿真 19466.7 → **0.920**。
  历史上该点是 1.088 过预测（MTP untie 幻影参数），MTP tie 修复后翻转为欠预测，
  **但 `sim_vs_real_report.py` 未收录此锚点**，翻转无人察觉。
- **构成推测**：DSv4 base 本身 0.971（−444 MiB）+ MTP/mHC 增量部分欠计（MTP 层是"embedding+decoder+head"
  复合层，其 MoE FFN 段同受 D1 碎片影响；mHC 增量真机仅 +688 MiB、建模误差小）。
- **建议**：先把该锚点加入记分卡（一行代码）；随 D1 修复重看——若 D1 margin 扩展后该点回到 ≥0.95，
  则无独立缺陷；仍欠则单独诊断 MTP 段。

### D3（P2，接受观察）：pp2-stage0 optstep 过预测 1.089

F1 前 1.057，ln2 进 op 图后无重算 BWD 峰共存项整体变保守所致（提交注明）。安全方向，
K_OPT=4 本就取"3 观测 +1 保守"。建议观察，若后续 >1.10 再收紧。

---

## 3. 防护/测试体系缺陷

### Z1（P2）：z2 交叉校验拦不住"层级 norm 缺失"类缺口

- **实证**（本轮变异探针）：从 DSv3 MoE 层删掉刚补的 ln2 op → `validate_against_opdag` 仍
  `ok=True, findings=[]`。crosscheck 只对 `mla_attn`/`moe_experts` 两族做 op 名册对比，
  不校验 `TransformerLayerSubmodules` 级别的 norm roster——F1 这类 bug 回潮时防护网静默。
- **建议**：给 crosscheck 增加第三族 `layer_norms`：从 `gpt_layer_specs.py` 抽
  `input_layernorm`/`pre_mlp_layernorm` 绑定（非 IdentityOp 即须在手写层 op 图中存在对应 NORM op）。
  本轮变异探针可直接固化为其验收用例。

### Z2（P2）：13 锚点记分卡不在 pytest 门里

`sim_vs_real_report.py` 是手动脚本；pytest 内只有 `test_regression_dsv3.py`（2 锚点 ±1%）等零散守卫。
锚点比值漂移（如 D2 的 1.088→0.920 翻转）不会让任何测试变红。
**建议**：把记分卡包成参数化测试（每锚点 ratio 带上下带），欠预测带比过预测带严。

### Z3（P3）：z2 在无 mindformers 源的机器上整体 skip

CI/本地无源码时 8 条 crosscheck 用例 skip——源忠实性在默认跑法下无守卫。另有踩坑记录：
`MINDFORMERS_ROOT` 必须指**包目录**（`.../mindformers/mindformers/mindformers`），
指仓库根会 33 errors（extractor 找不到 `parallel_core`）。建议 `default_mf_root` 做两级探测。

### Z4（P3）：三个平台常数自证而非独立测量

HCCL 200 MiB/通信域、alloc_block 512 B、pool 碎片率 1.8%（DSv4 单点标定，跨模型稳定性未验证，
测试 docstring 自己承认）。`reserved_oom` 因此两侧皆近似。

---

## 4. 未建模残差（known-missing physics，均有文档 caveat）

| 编号 | 项 | 影响域 | 现状 |
|---|---|---|---|
| G2 | ring/ulysses CP 的 KV 块 send/recv 在飞双缓冲（~2·(2·n_kv·hd)·(S/cp)·B） | CP 非 colossal | 未建；off loss 峰 |
| G3 | GQA colossal KV all-gather full-S buffer | GQA+colossal cp>1 | 已建公式但 opt-in 默认关、**未真机验证**（冻结的 cp2 精确减半测试与其代数互斥） |
| G4 | backward 方向 grad-P2P（量级 [S,B,H]） | pp>1 | 未建；在 pp2-stage0 现有过预测余量内，文档化为保守残差 |
| G5 | 无重算-MoE 碎片长尾（= D1 根因） | no-recompute MoE | 唯一造成 <0.95 锚点的未建模项 |
| G6 | VPP m==pp 的调度特例（Megatron all-warmup vs hyper_parallel） | VPP 边界 | hyper_parallel 无源码，未确证，统一走通式 warmup |
| G7 | MoE dispatch 真实倾斜分布（现为 balanced/capacity/skew 三口径近似） | MoE OOM 边界 | skew 因子须用户给 |

---

## 5. 未实现功能（fail-loud NotImplemented——不产生错误数字，但是功能缺失）

| 编号 | 功能 | 拒绝点 |
|---|---|---|
| N1 | `norm_placement` post / sandwich | `build_llm._check_implemented_dispatch` |
| N2 | `normalization` ≠ RMSNorm（LayerNorm 等） | 同上 |
| N3 | `position_embedding_type` learned_absolute / none | 同上 |
| N4 | `add_bias_linear` / `add_qkv_bias`（Qwen 系 bias） | 同上（量级小但拒绝静默忽略） |
| N5 | `qk_layernorm=True` 于 mla/dsv4/dsa（仅 gqa/mha 已建） | 同上 |
| N6 | 非 Adam/AdamW 优化器（K_OPT/state 均自 AdamW 导出） | `feasibility_errors` |
| N7 | loss_type 三种之外（如 fused-CE 变体的显式 op 图；现走 `cross_entropy_fused` 布尔近似） | `head.py` |
| N8 | PP>1 + activation swap（真机也不支持，属忠实拒绝） | `feasibility_errors` |
| N9 | mindformers 显式 per-chunk VPP ranges（`layers_per_stage` + interleave 时为"stage 内连续均衡切"文档化近似） | `parallel_model.stage_chunks` |

---

## 6. 验证盲区（真机栈限制所致，非仿真器缺陷）

本 build（116, feature/pynative-arch-evolution + MS2.10）实测限制，导致以下域**只有解析验证、无真机锚点**：

| 域 | 阻塞原因 |
|---|---|
| **TP>1 激活**（tp 切激活的真机峰值） | SP+MoE 不支持；TP+MoE Detach layout bug；DSA+TP fork kernel bug（`int - tuple`） |
| **pp>2 / VPP** | pynative AdamW 对 decoder-only 中间 stage param/state 配对崩（MLA/GQA 同崩） |
| **PP × 重计算** | 本 build 互斥（NORECOMP=1 才能跑 PP） |
| **DSA 全域** | 无 pynative DSA 可跑路径（模型自标"预估计"） |
| **swap/offload 真机曲线** | 未采（评估器侧有解析用例） |

这些域当前由本轮解析用例（VPP×重算、TP×CP 逐字节、混合并行端到端）+ Megatron/mindformers
逐行 port 锚定；真机栈解锁后应优先补锚点。

---

## 7. 建议行动（优先级序）

1. **D1**：no-recompute 域残差标定（8L-none + cp2-none 两点联合，与 K_CE 解耦）→ 预期把
   0.931/0.937 拉回 ≥0.95，消灭全部 OOM 不安全锚点。
2. **D2**：mHC+MTP 锚点入记分卡（立即，一行）；随 D1 复看。
3. **Z1**：z2 增加 layer_norms 校验族 + 固化变异探针为测试（防 F1 回潮）。
4. **Z2**：记分卡入 pytest（每锚点 ratio 带）。
5. **G1/盲区**：真机栈解锁后补 TP>1、VPP、DSA 锚点；GQA colossal KV buffer 找一个
   GQA+cp 可跑配置做单点验证。
6. **Z4**：pool 碎片率找第二个模型点交叉标定。

---

*测试工程轮产物：`tests/test_memval_hybrid_recompute.py`（25）、`tests/test_memval_structure_matrix.py`（225）、
新真机锚点 DSv3-8L-none=19967.3（已入 `sim_vs_real_report.py`）、F1 定性与修复复核、本报告。*
