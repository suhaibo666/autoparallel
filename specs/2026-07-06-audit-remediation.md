# 内存评估器审计 + 整改设计（2026-07-06）

> **baseline**: `feat/unified-llm-modelspec` @ `5a5034f`（审计时），后续整改在其上迭代。
> 审计方法：source-faithful，每条断言带已核验 `file:line`。整改总原则：**每项守住 DSv3 锚点
> `validate_dsv3.py` = 12409.5 逐字节 + `pytest` 全绿**（所有整改都在 DSv3 config 不走的路径上，
> 故此不变量必然成立——是每步的硬门）。**docs-first**：每个子代理先补本文档对应 §，再改代码。

## 1. 7 目标达成度（审计结论）

| # | 目标 | 判定 | 关键证据 |
|---|---|---|---|
| 1 | 全/选择/细粒度选择重算 + swap | ✅（细粒度有偏差→D-3） | `mem_timeline.py:376,378,382`；`structure_mem.py:219` |
| 2 | mha/gqa/mla/dsa/hca/csa + dense/moe | ✅（ungated 缺→D-6） | `registry.py:57`；`dsv4_hybrid.py:73`（csa=r4/hca=r128/dsa=r4 indexer 子件） |
| 3 | fsdp2/tp/ep/pp/cp | ⚠️→D-1 | cp 只切参数不切激活（全库 `'cp'` 作 shard key 0 处） |
| 4 | HCCL group 开销 | ⚠️→D-2 | `framework.py:57` 有模型但 `report.py:52` 未接入 |
| 5 | 内存 timeline 图 | ✅ | `timeline_probe.py`（ASCII+PNG）；`record_timeline` |
| 6 | PP 每个 stage | ✅（VPP 过估→D-4） | `mem_timeline.py:305`；`report.py:58` |
| 7 | 配置文件灵活仿真 | ⚠️→D-7 | 对象驱动（LLMConfig+presets），无 yaml 加载器 |

## 2. 发现清单 + 整改决策（用户 2026-07-06 定夺）

| # | 发现 | 决策 | 整改要点 |
|---|---|---|---|
| **D-1** | cp 只切参数、不切激活（over-count） | **修** | cp = **全局域序列切分**：整体激活值都 ÷cp（含 flash-ws）。cp 是独立于 sp 的全局 SP 域 |
| **D-2** | HCCL 未接入报告 | **修** | 按**通信域**估计，接入报告（reserved 预测） |
| **D-3** | 细粒度单 op 选择性重算 under-count | **修** | 细粒度选重**按单个 op 各自估计自己**（每个选中 op 用自身 in/saves 估自己的重算，不依赖段边界已 pin 假设） |
| **D-4** | VPP(v>1) 峰值 over ~V× | **修** | 按 **VPP 理论公式**估 micro 数；整体激活按**实际层数（所有 chunk 层数之和）**估 |
| **D-5** | DSv4-fused 已知 UNDER ~7% | **维持现状** | 融合 kernel 内部量，源码级已定位、已 caveat |
| **D-6** | ungated FFN 硬编码 2*F | **修** | 为 `gated_linear_unit=False` **构建 op 图**（fc1 不 2×） |
| **D-7** | 无配置文件加载器 | **修** | 构建 **mindformers 配置文件 → LLMConfig 转换器** |
| **D-8** | 非整除 PP 层分配堆最后 stage | **修** | 按 mindformers 设计**显式配置每 stage 层数**（从用户 config 读，不自行推测） |
| **D-9** | DSv3 ~0.5% sub-block 残差 | **不处理** | rope cos/sin、norm rstd、cast 临时量 |

## 3. 整改执行顺序（子代理，逐项 docs-first + DSv3 硬门）

1. **并行域核**（同改 `shape_eval/parallel_model/mem_timeline/attention`，故一组内序化）：D-8 → D-1 → D-4。
2. **独立项**（文件不相交，可分批）：D-3（`structure_mem`）、D-2（`framework`+`report`）、D-6（`ffn`+`build_llm`）、D-7（新 `configs/` 转换器）。

各子代理落地机制细节到下方 §4+（每项一节，先写后码）。

## 4. 各项整改机制（子代理填充，source-faithful）

<!-- D-8 / D-1 / D-4 / D-3 / D-2 / D-6 / D-7 各起一节，含 mindformers/Megatron 源码定位 + 公式 + DSv3 复核 -->

### D-8：PP 每 stage 层数显式配置（不自行推测）

**审计发现**：`ParallelConfig.layers_per_stage`（显式每 stage 层数）已支持并被 `parallel_model.py:45-49`
读取，但**无校验**——错长度/错和的列表会静默产生错误映射；缺省的均匀切 `per=n_layers//pp`
（`:50-51`）把非整除 remainder 全堆**末 stage**，可能错判 tightest stage。

**决策（用户）**：忠实 mindformers 流水线设计——**直接配置每 stage 多少层，不自行推测**：显式让用户
配、或从用户配置读。mindformers 用 `offset`/`num_layer_list` 指定各 stage 的层数偏移；本库 `layers_per_stage`
是其等价低层接口（每 stage 层数列表，**含 embedding + head 两个伪层**，故和 == `n_layers` = num_layers+2）。

**整改**（`parallel_model.py.__init__` + `specs.py`）：
- 显式 `layers_per_stage` 升为**首选路径**，`__init__` 加三重校验：`len == pp`、`sum == n_layers`、
  每项 `> 0`，任一不满足即 `raise ValueError`（不静默）。
- 均匀切保留为**缺省 fallback**（`layers_per_stage=None`），文档标注 pp>1 非整除时建议显式配。
- 高层 mindformers 语义（decoder 层→stage + emb 归 stage0 / head 归末 stage）由 D-7 配置转换器映射到此列表。

**DSv3 复核**：DSv3 走 `pp=1, layers_per_stage=None` → 不进显式分支、均匀切 `per=n_layers` 全归 stage0
（与旧一致）→ `12409.5` 逐字节不变、`pytest` 全绿。新增 `tests/test_stage_layers_explicit.py`（6 例：
非均匀映射 + 三重校验 raise + None 均匀 + remainder 归末）。

### D-1：context-parallel（cp）必须切激活（activations ÷cp）

**审计发现（已核）**：`cp` 目前**只切参数**（折进 `fsdp_degree = dp_shard*cp`，`static_mem.py:8`/
`parallel_model.py:32`），**不切任何激活**——全库 `'cp'` 作 shard key 出现 **0 次**（`grep -rn "'cp'"
cost_eval/layers` 无命中；所有 TensorRef 的 `shard` 只用 `tp`/`ep`/`sp`）。故 `cp>1` 时每卡激活、
flash workspace、loss/index bwd_scratch 全部 **over-count ×cp**。

> [!deprecated] 已被 2026-07-07 真机 + 算子级 Profiler **部分推翻**，修正见本节末「#### D-1 修正
> （2026-07-07）」。要点：**loss/head 区对所有 cp 算法都是 full-S（不 ÷cp）**——旧「整体 ÷cp（含
> loss/head）」错半了满 vocab 的 loss 张量，cp=2 峰值**欠估 ~29%**（旧 8839 vs 真机 12433）；且
> body ÷cp 与否**依赖 `context_parallel_method`**（colossal 额外 all-gather KV 到 full-S）。以下
> 2026-07-06 段落保留原始决策记录（body ÷cp 对 ulysses/ring/hybrid 仍成立、对 loss/head 被推翻）。

**忠实模型（S_effective = S / cp）**。cp 是**全局域的序列并行**（ring / context attention）：整条序列
沿 token 维切成 cp 份，每 rank 只拥有 **S/cp 个 token（query）**；key/context 经 ring P2P 轮转，每个
query 仍见**完整上下文**（用户 2026-07-06 定夺原话：「cp 是全局域的序列切分，整体的激活值都会切分为
原有的 1/cp」）。这正是本库设计文档早已写下、但**只设计未落地**的意图：

- `specs/2026-06-23-pynative-cost-evaluator-design.md:147` —— 「**CP `context_parallel`｜激活 seq 维
  cp**｜colossal=ring P2P / ulysses=all-to-all」。
- 同文件 `:102` —— 「`{0:"cp"}` = seq 切 cp」（TensorRef.shard 的既定语义，只是没有一个张量真的带上它）。
- `specs/2026-06-29-core-modules-m4-m5-m6-internals.md:80` —— 「**cp 整除 seq** → 非法配置即报」
  （整除校验口径，本次照此 raise）。
- 参数侧不动：同 `:151` 「FSDP `data_parallel_shard`：param+grad+opt 存储 /(dp_shard·cp)」——参数的
  cp 折进 `fsdp_degree`（`static_mem.py:8`）**已实现且正确**，本次**只碰激活**，不双算。

**SP × CP 组合**（两个正交的序列并行域，degree 相乘）：
- **SP 标注的激活**（`shard={0:'sp'}`，`sequence_parallel` 时 → S/tp）：叠加 cp → **S/(tp·cp)**。
- **非 SP 激活**（attention 内部全序列，如 MLA `qb_out`/`kvb_out`、GQA `qkv`）：→ **S/cp**。
- **权重无 S**（已逐一核验：embedding `[vocab,H]`、head `[H,vocab]`、attn `qkv_w[H,·]`/`o_w[·,H]`、
  MLA `qb_w/kvb_w`、ffn `fc1_w/fc2_w`、专家 `e_w1/e_w2`、dsv4 `wq_*/wkv/wo_*/idx_*/cmp_*`、mHC
  `rms_w/proj_w` —— 无一含符号 `S`）→ 天然不受 S→S/cp 影响，参数分片仍走既有 `fsdp_degree`。

**index_scores（S²）在 CP 下的取舍（关键决策）**。dsv4 DSA 索引器把 `index_scores [B,S,S] fp32`
物化为 `bwd_scratch="4*B*S*S"`（`dsv4_hybrid.py:136`，源 `indexer.py:227-236`）。它是 `[B, S_query,
S_key]` 的打分矩阵。CP 下 **query 维切分、key 维保持全量**（每 rank 的 S/cp 个 query 对完整 S 个 key
打分，key 经 ring 轮转补齐）→ 每卡 `[B, S/cp, S]` = **`4*B*S*S / cp` = S²/cp（去掉一个 S 因子，一次
cp）**，**不是** S²/cp²。理由：ring/context attention 只切 query（token）维；被 attend 的 key/context
维不切。这与用户「整体激活 ÷cp（一次）」以及设计「seq 维 cp（单维）」完全一致。同理 dsv4 稀疏 gather
张量 `kv_gathered [B,S,TOPK,vd]` / `attn_weights [B,n_heads,S,TOPK]` 只切 query 维 `S`，**不切**
`TOPK`（= window + S//ratio，每 query 见的上下文位置数，ring 补齐后仍全量）。

**统一机制 = 「切 query/token 维一次」**，落在两处：
1. **张量**（`shape_eval.resolve_tensor`）：非权重张量，切**首个引用符号 `S` 的维**（token/query 维）
   ÷cp；其余含 S 的维（仅 `TOPK_DIM` 这类上下文维）保持全量（故用 `break` 只切第一维）。含 S 的
   token 维覆盖：`[S,B,·]`（dim0）、MoE `disp/e_* [TLOCAL=S·B·topk·C, ·]`（dim0，dispatched token
   ∝S）、dsv4 `compressed_kv [S//ratio,·]`（dim0，压缩 token ∝S）、`kv_gathered/attn_weights/
   topk_indices` 的 query 维（首个 `S`）。
2. **workspace / bwd_scratch 字符串**（`shape_eval.ShapeEval.resolve`）：含符号 `S` 的表达式，求值后
   **整体 ÷cp 一次**。单 S 项 → S/cp（`FLASH_LSE_WS=64·B·n_heads·S`、loss `8·S·B·vocab`、MoE
   `MOE_STAGING_WS=2·S·B·topk·C·H`、mHC `4·S·B·n·H`）；双 S 的 index_scores `4·B·S·S` → S²/cp（去一
   个 S 因子 = query 切分，与张量口径一致）。全库 workspace/bwd_scratch **全部含 S**（已核），故「除
   一次」对每一项都恰为「去掉 query 那一个 S 因子」。

**整除**（与 `shape_eval.py:60-62` 同口径 raise，OOM 安全，不静默 floor）：张量侧对被切的 token 维值
做 `% cp != 0 → raise`（SP 张量先 ÷sp 再查 `(S/tp)%cp`，等价于 `(tp·cp)|S`；非 SP 查 `S%cp`）。因每个
带 S-workspace 的 op 必有含 S 的激活输入/输出（flash 有 `qkv`、nll 有 `logsm`、indexer 有 `ln1`…），
`resolve_tensor` 会**先**在这些张量上 raise → 非法 `S%cp≠0` 配置在算 workspace 前已被拒；故 workspace/
scratch 侧用 floordiv 安全（合法配置下 `S%cp==0` 时 S 与 S² 均整除，逐字节精确）。

**DSv3 = cp=1 无操作（硬门论证）**。两处改动均 `if cp > 1:` 守卫，`cp = pm.degree("cp")`；DSv3 anchor
（`validate_dsv3.py:84`）走 `cp=1`，`degree("cp")=1` → **两处分支完全不进入**，`resolve_tensor` /
`ShapeEval.resolve` 逐字节复现旧行为 → DSv3 4L 恒 `12409.5`、`pytest` 全绿。cp 是本次唯一新增的激活
分母，且只在 `>1` 时生效，故对全部现有 cp=1 配置（anchors/golden/preset）是**精确 no-op**。

#### D-1 修正（2026-07-07，真机 + 算子级 Profiler 确认）：loss/head 区 full-S + 按 cp 算法分派

真机 cp=2 实测 + 算子级 Profiler（`analysis/realmachine/cp2_colossal/operator_memory.csv`，
DSv3 4L、seq=4096、vocab=129280、full-recompute、`context_parallel_method=colossal`）**推翻**了上面
「整体激活 ÷cp（含 loss/head）」的口径，给出两条修正：

**Correction A（已真机验证，最高优先）——loss/head 区对所有 cp 算法都是 full-S（不 ÷cp）。**
- 证据：cp=2 峰值 **`Allocated=12433.2 MiB`**，落在 **loss 反向（`ScatterAddExt`/loss backward）**。
  此刻 vocab 三张量均为 **full-S fp32 满 vocab，各 2020 MiB**：`log_softmax`(saved) + `probs`
  (=exp(-logsm)) + `grad_log_softmax`(scatter_add 出的梯度)。`4·S·B·vocab=4·4096·129280=2020 MiB`，
  三份共 **6060 MiB 全为 full-S**——**不是** S/cp（S/cp 仅 1010 MiB/份）。Profiler 里 `ScatterAddExt`/
  `Log`/`Neg`/`SubExt`/`MaxDim` 五个 loss 反向算子全部实测 2020.0 MiB，逐一坐实 full-S。
- 机理：mindformers 各 cp 算法在 **LM head 前把 hidden all-gather 回 full-S**（head+loss 跑在完整
  序列上），故 `h_final`/`logits_lm`/`logsm`/`probs`/`loss` 及 nll 反向 `grad_log_softmax`
  （`bwd_scratch=8·S·B·vocab`，含 chunked `8·S·B·vocab//k`）**全为 full-S、所有 cp 算法一致**。
  旧「整体 ÷cp」把它们错半 → cp=2 峰值 **欠估 ~29%（OOM 不安全）**：旧模型报 8839，真机 12433。
- **注**：`embedding` 输出 `emb_out` 与所有 **decoder 层激活**仍 ÷cp（是 S/cp）——**只有 head/loss 区**
  full-S。二者边界 = LM head 前的 all-gather。

**Correction B（源码忠实，colossal-KV 分支暂未真机验证）——body ÷cp 依 `context_parallel_method`。**
核于 mindformers `pynative/distributed/{context_parallel,style}.py`：

| method | body 激活 | attention KV | loss/head |
|---|---|---|---|
| `ulysses`（`style.py:170,198`：all-to-all，seq-shard→head-shard） | ÷cp | ÷cp（随 body） | **full-S** |
| `ring`（Q,K,V 全切 + ring-pass 分块） | ÷cp | ÷cp | **full-S** |
| `hybrid`（ring×ulysses 组合） | ÷cp | ÷cp | **full-S** |
| `colossal`（`style.py:168,188`：`ulysses_degree=1`，all-gather KV 到 full-S） | ÷cp | **full-S（额外 full-S KV buffer）** | **full-S** |

**落地机制（2026-07-07）：**
- `ParallelConfig.context_parallel_method: str = "colossal"`（默认对齐 mindformers DSv3 yaml），
  `__post_init__` 校验 ∈ {colossal, ulysses, ring, hybrid}（fail-loud，仿 head.py:50 的 loss_type）。
- `TensorRef.cp_shard: bool = True`（False=full-S 不 ÷cp）；`TensorRef.cp_kv: bool = False`
  （True=colossal 下 all-gather 到 full-S、其余 method 仍 ÷cp）。
- **A**（`layers/head.py`）：`h_final`/`logits_lm`/`logsm`/`loss`/`probs`(vocab_parallel 支) 标
  `cp_shard=False`。`resolve_tensor` 仅当 `t.cp_shard` 才 ÷cp；`ShapeEval.resolve` 仅当
  `op.output.cp_shard` 才对 workspace/bwd_scratch ÷cp（nll 输出 `loss` cp_shard=False → 其
  `bwd_scratch=8·S·B·vocab` 保持 full-S）。MTP 头经共享 builder `build_head_and_loss_ops` 自动覆盖。
- **B**（`layers/{attention,mla,dsv4_hybrid}.py`）：KV 侧激活标 `cp_kv=True`（MLA
  `kv_a_in`/`kv_a_out`/`kvb_out`；dsv4 `kv`/`kv_a_out`/`compressed_kv`/`kv_gathered`）；
  `resolve_tensor` 在 `method=="colossal"` 下对 `cp_kv` 张量跳过 ÷cp（其余 method 仍 ÷cp，随 body）。
  GQA 的 `qkv` 是 Q/K/V **融合**张量、KV 分量不可无损分割 → **不单独标**（colossal 下 fused-qkv 仍按
  body ÷cp，属小幅欠模；全重算下该 KV 量 off-peak、且本栈跑不了 cp+无重算故未真机验证——已 caveat）。
- **验证**：DSv3 走 cp=1 → 两处 cp 分支 `if cp>1` 不进入 → 逐字节复现旧行为 → DSv3 4L 恒 `12409.5`。
  cp=2 4L full-recompute 预测 **8839 → 12381.5**（Δ = Δlogsm 1010 + Δlogits 505 + Δbwd_scratch 2020
  + Δh_final 7 ≈ +3542）；colossal==ulysses（KV all-gather 在全重算层、off loss 峰）→ ratio 0.996 vs
  真机 colossal 12433 / ulysses 12441（≈ dp=2 的 12474）。

### D-4：VPP（交错式 1F1B, v>1）激活峰值 —— 从 ~V× 过估改为**按 chunk 忠实累加**

**审计发现（已核，`mem_timeline.py:86-95`/`:363-387` 旧行为）**：交错式 1F1B（VPP，`interleave = v > 1`）
的 simulate 走**物理微批粒度**——沿用 plain-1F1B 的事件循环，每个 FWD 事件把**整个物理 stage 的全部 L 层**
（遍历全 `layer_ids`）pin 进 `act_live`（`:365-387`），却又把 warmup 加深到交错式深度
（`build_interleaved_1f1b` / `interleaved_warmup`）。于是峰值 ≈ `(warmup+1) 个在飞微批 × L 层`，而**真实
VPP 每个在飞（虚拟）步只驻留一个 chunk（L/V 层）**——旧式每微批高估 **V 倍**（旧 docstring 自述「偏高约
V 倍」，列为文档化后续项）。`v=1` 时该式退回 plain-1F1B、逐字节一致（`:97-99`），**唯 v>1 过估**。

**忠实模型（Megatron 交错式，按 chunk 累加）**。VPP 下每个物理 device（PP rank）持 **V 个 model chunk**，
每 chunk **L/V 层**（L = 该物理 stage 层数；全模型总层数 = `pp·L`，**不是** `pp·V·L`）。用户 2026-07-06
定夺原话：「vpp 应该按照 vpp 的理论公式估计 micro size，整个激活按照实际的层数（**所有的 chunk 层数之和**）
来估计」。落成两条，均**逐行锚定 Megatron `pipeline_parallel/schedules.py`**：

1. **micro 数 = VPP 理论 warmup 公式（虚拟步 = chunk-forward 粒度）**。取 `get_pp_rank_microbatches`
   （`schedules.py:877-878`）：
   $$\text{warmup} = (pp - \text{rank} - 1)\cdot 2 + (V-1)\cdot G,\qquad G=\texttt{microbatch\_group\_size\_per\_vp\_stage}$$
   默认 `G = pp`（深度优先，`model_parallel_config.py:519-520`），**clamp 到 `total = m·V`**（虚拟微批总数，
   `:889-890`）——注意是 `m·V` 不是 `m`：warmup 计的是**虚拟步（chunk-forward）**，本库旧 `interleaved_warmup`
   clamp 到 `m` 是物理粒度口径，虚拟路径须用 `m·V`。
2. **整体激活 = 实际层数（Σ chunk 层数）忠实累加，每步只驻留一个 chunk**。交错式的 (microbatch, chunk)
   下发顺序逐行 port 自 Megatron `get_schedule_table`（`:902-929`，deep-first：每组连跑 `G` 个微批的同一
   chunk 再切下一 chunk，末组把剩余微批一次排完）+ `convert_schedule_table_to_order`（`:932-955`：前 warmup
   步纯 FWD，随后 steady 逐 `(FWD_i, BWD_{i-warmup})` 交替，末尾 warmup 个 BWD 收尾；forward/backward 同表
   FIFO → 每 `(mb,chunk)` 前向一次、反向一次）。每个**虚拟 FWD 步 pin 一个 chunk（L/V 层）**、其对应 BWD 步
   释放该 chunk。峰值（warmup 结束、第一个 BWD 前的最深处，`warmup+1` 个在飞虚拟步）：
   $$\text{peak act\_live} = \sum_{c=0}^{V-1} n_c \cdot (\text{chunk }c\text{ 的 saves}),\qquad \sum_c n_c = \text{warmup}+1$$
   其中 $n_c$ = 峰时 chunk $c$ 的在飞微批数（**热身期早 chunk 在飞更多** → 分布非均匀）。层均匀时
   $= (\text{warmup}+1)\cdot(L/V)\cdot s$（= 旧物理峰 ÷V，V× 过估恰好消去）；层非均匀时（embedding 归首
   chunk、loss 归末 chunk）按各 chunk **实际** saves 加权 → **非「盲目 ÷V」**，loss 层巨大的 `bwd_scratch`
   仍落在其所在单一 chunk。**关键不变式**：`Σ_c len(chunk_c) == L`（所有 chunk 层数之和 = 实际 device 层数），
   任一微批走过全 V chunk 最多 pin **L 层**，**绝不 `V·L`**。

**per-chunk 分布（已手算+脚本双验）**。`pp=2, v=2, m≥4, stage0`：`warmup=(2-0-1)·2+(2-1)·2=4`，峰 `5` 个
在飞虚拟步，分布 `chunk0=3 / chunk1=2`（deep-first 表 `[(0,0)(1,0)(0,1)(1,1)(2,0)(3,0)(2,1)(3,1)]` 前
5 个前向）→ `peak = 5·(L/2)·s`。对照：v=1（plain）`warmup=1`、峰 `2` 微批 × L 层 = `2L·s`；旧物理过估
v=2 = `5 微批 × L 层 = 5L·s`（= 忠实值的 **V=2 倍**）。故忠实峰 `5·(L/2)=2.5L·s` 落在 plain(`2L`) 与
旧 V×(`5L`) 之间 —— **方向（>plain，交错确实更吃激活）与幅度（<V×，过估消去）同时修正**。

**峰值随 V 非单调（bulge at V=2）**。忠实模型下净激活膨胀比 ≈ `1 + (pp-1)/(pp·V)`（旧 docstring 已述），
**V=2 处最大、随 V 增大回落**：`pp=2, L=4, stage0` 的峰 act_live（层单位）= v1:`8` → v2:`(4+1)·2=10` →
v4:`(8+1)·1=9`，故 `v2 > v4 > v1`（旧 simulate 是单调增，属过估的副产物）。整改后原
`test_interleave_peak_monotonic_in_v_stage0`（断言单调增）改写为 `bulge at v2 + 界定 <V×`。

**实现**（`mem_timeline.py`，`v>1` 专属分支，`v<=1` 走原路径）：
- 新增 `get_schedule_table` / `interleaved_virtual_order` / `chunk_layer_ids` 三个纯函数（前两者逐行 port
  Megatron，后者均衡切层且带 `Σ==L` 不变式）。
- simulate 事件循环改为遍历统一的 `steps = [(kind, mb, ev_layers)]`：`v>1` → `ev_layers = chunk`（一个虚拟步
  一个 chunk）；`v<=1` → `ev_layers = layer_ids`（整 stage，逐字节复现 `build_interleaved_1f1b`）。逐层 pin/
  gather/workspace/BWD 桶**机体一字未改**，仅把作用域从「全 stage」收窄到「当前步的 chunk」，`pinned` 仍以
  `(mb, lid)` 为键（chunk 互斥 → 无碰撞、无泄漏）。

**v=1 == plain-1F1B + DSv3 no-op（硬门论证）**。simulate 对 `v<=1` 直接用 `[(ev.kind, ev.mb, layer_ids)
for ev in build_interleaved_1f1b(stage,pp,m,v)]`，`build_interleaved_1f1b(·,v<=1)` 委托 `build_1f1b`，`ev_layers`
恒 = 整个 `layer_ids`（同一列表对象）→ 事件序列、pin 集合、`_prefetch_*` 全逐字节等价旧循环。DSv3（`pp=1,
v=1`，`interleave` 默认 1）与所有现有 anchors/golden/preset 均 `v=1` → **不进 v>1 分支** → `12409.5` 恒定、
`pytest` 全绿。v>1 是本次唯一改动路径，对 v=1 是**精确 no-op**。

### D-6：ungated MLP（`gated_linear_unit=False`）—— 构建 op 图

**审计发现**：`build_llm._check_implemented_dispatch`（`build_llm.py:36-38`）对 `gated_linear_unit=False`
**fail-loud raise**——ffn.py 硬编码 `2*F`（SwiGLU gate+up），不支持 ungated MLP（GPT-2/OPT 式 fc1→F→gelu→fc2）。

**忠实模型**：gated（SwiGLU）fc1 输出 **2·F**（gate+up 合并），swiglu 取 gate⊙up→F；ungated（plain MLP）
fc1 输出 **F**，纯 gelu/relu→F。二者 op 数相同（ln2→fc1→act→fc2→add2），**唯 fc1 输出维 2F↔F、fc1_w
[H,2F]↔[H,F]、激活 swiglu↔gelu** 不同。同理 MoE 专家（`2*moe_F↔moe_F`）与 shared expert
（`2*moe_shared_F↔moe_shared_F`）。

**整改**（`model_spec.py`/`llm_config.py`/`ffn.py`/`build_llm.py`）：
- `DimTable` 加 `gated_linear_unit: bool = True`；`to_dimtable` 透传 `cfg.gated_linear_unit`。
- `build_dense_ffn_ops`/`build_moe_ffn_ops`/`build_shared_expert_ops` 按 `d.gated_linear_unit` 分派 fc1
  输出维（`2*F`/`F` 等）+ 激活名（swiglu/gelu）。
- 删 `build_llm.py` 对 `gated_linear_unit=False` 的 raise。

**DSv3 复核**：默认 `gated_linear_unit=True`（DSv3/全 preset 皆 SwiGLU）→ 走原 2F 分支、`12409.5` 逐字节
不变。新增 `tests/test_ungated_ffn.py`（6 例：dense/moe/shared 的 F↔2F + fc1_w 减半 + build_llm 不再 raise）；
`test_unimplemented_dispatch.py` 移除 gated_linear_unit 用例（已实现）。

### D-2：HCCL 通信缓冲接入报告（按通信域数，reserved 口径）

**审计发现**：`framework.hccl_reserved_buffer(pc)`（`framework.py:57`，= `200MB × num_distinct_communicators`）
**有模型但从未被 `Evaluator.evaluate` 调用**（`report.py:52` 只算 `framework_reserve`=0）→ 目标 4「涵盖 HCCL」
在报告层面未落地：`evaluate()` 返回里 HCCL=0。

**忠实模型**：HCCL 每个**不同子通信器**（world + 各启用并行域：FSDP=dp_shard·cp / tp / ep / pp / dp_replicate）
在 **reserved 池**预留一份 `hcclBufferSize=200MB`（CANN 9.0 真机日志，平台属性）。**不进 allocated 峰值**
（ep=2 真机实证：加 EP 组 allocated 不变、reserved 涨）。设备 HBM 真实约束是 **reserved ≈ allocated_peak
+ HCCL (+ 池碎片)**。

**整改**（`report.py`）：`Evaluator.evaluate` 调 `hccl_reserved_buffer(pc)`，surfaced 为
`PeakMemoryReport.hccl_reserved_bytes`（world-level 同值）；加 `reserved_estimate_bytes(stage) = 该 stage
allocated 峰值 + HCCL` 供 reserved 口径 OOM 余量核查。**不改** `peak_bytes`/`oom`（仍 allocated 口径，真机验证）
——HCCL 只作独立字段，故 allocated 峰值不变。

**DSv3 复核**：`peak_bytes` 逐字节不变 `12409.5`（HCCL 是新增独立字段，不进 allocated）；`hccl_reserved_bytes`
= world(1)+FSDP(dp_shard=2>1) = 2×200MB（DSv3 dp_shard=2）。新增 `tests/test_hccl_reserved.py`（5 例：
world-only / tp / fsdp 缩放 + 不进 allocated + reserved 估计）。

### D-3：细粒度选择性重算 —— 每个选中 op **自估自身**重算足迹（单 op 不再低估）

**审计发现（已核）**：`estimate_select_memory`（`structure_mem.py:239`）把选中段的反向重物化算成
`recomp_scratch = max(0, sm_sel.forward_max_live − sm_sel.checkpoint_input)`——**无条件**扣掉选中段
「首个 save」(`sm_sel.checkpoint_input`)，其隐含假设是「该段入口边界已被前驱非选中 op / 层入口 pin 进
act_live，故不重复计」。代码自己已 flag 该假设的适用边界（`structure_mem.py:194-197`）：mindformers 选择
粒度是**模块 / cell**（`self_attention`、`feed_forward`），其边界恰是层 / 模块入口（已 pin），扣段边界精确；
但若选**更细的单个 op**（如仅 `flash`），该 op 的输入边界**未必已 pin** → 无条件扣段边界会把一份**并未
常驻**的输入当成「已提供」减掉 → `recomp` 被**低估**（OOM **不**安全方向）。**只在模块 / cell 粒度精确**。

**用户定夺（原话）**：「细粒度的选重还是应该根据单个 op 自己去估计自己来看」——细粒度选择性重算应
**由每个选中 op 各自、独立地估计自己的重算成本**，而不是共享一个「段边界已 pin」的假设。

**忠实模型（每 op 自估）**。选择性重算在反向时**逐个**重跑选中 op / region 以复现其输出供求梯度；
框架把每个选中的 op / 模块**各自**包进一个 checkpoint region（各自 save 自己的输入、丢弃内部激活、反向
时独立重物化）。故重算单个 op 的显存 = **它自己的重算足迹** = 该 op 此刻活着的输入 + 输出 + workspace
（= 对该单 op 求 `forward_max_live([op])`，即 inputs+output+workspace）。选中 op 在反向**不同时刻**被
各自重算（backward 顺着层逆序走，任一时刻只有一个 region 在重算）→ 层的重算工作集峰 = **各 region 峰
的沿时间线复用**（`forward_max_live(选中集)` 天然给出：**连续选中岛内**共存取峰、**跨岛**取 max），
再减去**其中确实已 pin 的进入边界输入**（已在 act_live 计过，避免双算）——**而非**无条件减「段首 save」。

- **进入边界**（须常驻才能起算的输入）= 被选中 op 消费、但**不由任何选中 op 产出**的非权重输入（段外
  产出或层入口）。其中**已 pin**（∈ 非选中 saves ∪ {层 checkpoint_input}）→ 减；**未 pin**（细粒度单 op
  的输入，唯一 saver 是它自己且已随选中丢弃）→ **保留**在 recomp（该 op 须自付其输入的重物化）。
- 与**旧式的区别**：旧式减 `sm_sel.checkpoint_input`（选中段**首个 save**，且**无条件**假设已 pin）；
  新式减 `_pinned_input_boundary`（选中段**进入边界输入**中**实际已 pin** 的部分）。两者在模块 / 全选
  粒度**逐字节相等**（边界 = 层入口 `x`，既是段首 save 又是 `ci`、确已 pin → 都减 `x` 字节）；只在
  细粒度单 op（边界未 pin）**分叉**：旧减、新不减 → 新 `recomp` 更大（**修正低估**，OOM 安全方向）。

**源码定位（selective remat = 逐 region 独立重跑，各持自身输入）**：
- mindformers `RecomputeConfig.select_module` / `exclude_op`（`config.py:759-796`）：按**模块路径** /
  **算子名子串**选一组 op；`activation_checkpoint.py` 的 `_clean_and_parse_config`（`:478-518`）反转成
  `{layer_id: [module_names]}`，并对**每个命中的 op / 模块各自**包 checkpoint 包装（`:534-540`
  `needle in op_name` 命中即独立包裹）→ 每个被包的 region **各自** save 输入、反向各自重跑，彼此**不**
  共享一个「整层已 pin」的边界。本库 op 图是扁平叶子模块，故「模块」= 一组 op，用子串命中匹配。
- Megatron `recompute_granularity="selective"` + `recompute_modules`（`transformer_config.py:526-534,559`，
  默认 `core_attn`）：selective 把**每个**被选 region（默认 core-attn）用 `tensor_parallel/random.py`
  的 `checkpoint(fn, ...)` 包住——`checkpoint` **只 save 该 region 的输入**、反向用保存的输入**重跑该
  region**（内部激活是重算期瞬态），每个 region 独立。这正是「每 op 自估自身、其输入是自己的 checkpoint
  边界」的语义来源。

**两端退化（byte-identical，load-bearing）**。`_pinned_input_boundary` 在两个极端与旧式逐字节相等，故
`select-all == full`、`select-none == None` 不变：
- **全选**（选中 = 全部 op）：非选中 = ∅ → 进入边界 = 层入口 `x`（唯一段外输入），`x` = `ci`、已 pin →
  减 `x` 字节 = `sm.checkpoint_input`；`forward_max_live(全) − x` **==** `full` 路径的
  `sm.forward_max_live − sm.checkpoint_input`（`mem_timeline.py:507-508`）。`bwd_working_set=0`、
  `act_live=ci` 均不动。
- **全不选**（选中 = ∅）：`selected=∅` → `forward_max_live(∅)=0` → `recomp=0`；`act_live=activation_saves`、
  `bwd_working_set=forward_max_live(全)−bwd_scratch` == `None` 路径，不动。
- **模块粒度**（选中 = 整个 attention 段）：进入边界 = 层入口 `x`（已 pin）→ 减 `x`，与旧
  `sm_attn.forward_max_live − sm_attn.checkpoint_input` 逐字节相等（既有测试不变）。

**单 op 修正（数值）**。toy dense 层（`H=64,S=128,B=1,n_kv=4,head_dim=16`，bf16，blk 不影响均 512 整除）
选 `{flash}`：`forward_max_live([flash]) = qkv(49152)+attn(16384)+ws(32768) = 98304`；flash 的输入边界
`qkv` 的唯一 saver 是 flash 自己（已随选中丢弃）、`qkv ∉ {非选中 saves ∪ ci=x}` → **未 pin** → 不减。
**旧** `recomp = 98304 − sm_sel.checkpoint_input(=qkv 49152) = 49152`（低估）；**新** `recomp = 98304 − 0
= 98304`（+49152，= 被错减的 qkv 边界）。方向：新 ≥ 旧，修正 OOM 不安全的低估。

**DSv3 复核（本项对 DSv3 完全 no-op）**：DSv3 走 `RecomputeSpec(mode="full", ...)`（`validate_dsv3.py:91`），
**不是** `select` → `mem_timeline.py:392-399` 仅对 `recompute.is_select(lid)` 的层建 `select_mem_by_id`，
DSv3 无 select 层 → `estimate_select_memory` **根本不在 DSv3 路径上**。故 `peak_bytes` 逐字节不变 `12409.5`、
`pytest` 全绿。改动仅落 `structure_mem.estimate_select_memory` + 新增 `_pinned_input_boundary` helper；更新
`tests/test_structure_mem.py::test_select_core_attn_scoped_to_selected_ops`（该断言原编码旧低估
`fml−checkpoint_input`，改为修正值 `fml`（边界未 pin 不减），附 D-3 注释）+ 新增单 op 自估足迹用例。

### D-7：mindformers 配置文件 → 评估器配置对象 转换器

**审计发现**：评估器完全**对象驱动**——`LLMConfig`（`llm_config.py`）+ `ParallelConfig`/`RecomputeSpec`/
`SwapSpec`/`OptimizerSpec`/`HardwareSpec`（`specs.py`）+ 预设工厂（`presets.py`），`cost_eval/` 内 **0 处**
`yaml`/`json.load`（`grep -rn "yaml\|json.load" cost_eval` 无命中）。故目标 7「按配置文件灵活仿真」只在
**对象层**达成：拿到一份真实 mindformers 训练 yaml 的用户必须**手工翻译**成上述对象。三处手写翻译
（`validate_dsv3.py`/`validate_dsv4align.py:dsv4_align_config()`/`presets.py`）就是这份手工映射被反复做的证据。

**决策（用户原话「构建 mindformers 的配置文件转换器」）**：新增 `cost_eval/configs/from_mindformers.py`，
把 mindformers **pynative 训练 yaml** 映射到评估器配置对象。**核心保持纯**（只吃 dict，`cost_eval` 核不新增
硬 yaml 依赖）；薄 path-loader 内**惰性** `import yaml`。字段映射**逐字复现**上述三处已核验手写映射
（它们是 verified source of truth），并以「喂回锚点 config → 得到与 `dsv4_align_config()`/`deepseek_v3()`
**逐字段相等**的 `LLMConfig` → 同一评估峰值」作为保真判据。

**模块 API**：

| 符号 | 签名 | 说明 |
|------|------|------|
| `EvaluatorConfigBundle` | dataclass(`llm, parallel, recompute, swap, optimizer, hardware`) | 打包 6 个评估器对象 |
| `from_mindformers_dict(mf: dict)` | `dict → EvaluatorConfigBundle` | **纯核**（不 import yaml），全部映射逻辑在此 |
| `load_mindformers_yaml(path)` | `path → EvaluatorConfigBundle` | 薄壳：**函数内**惰性 `import yaml` + `safe_load` + 调纯核 |

**`model.*` → `LLMConfig` 字段映射**（定位符 = `prep_dsv4align.py:NN`（记作 P4:NN）+ `validate_dsv4align.py:
dsv4_align_config()`（记作 V4:NN）+ `presets.py:deepseek_v3()`（记作 D3:NN））：

| mindformers `model.*` | LLMConfig 字段 | 规则 / 定位 |
|---|---|---|
| `hidden_size` | `hidden_size` | 直通（P4:72；V4:43） |
| `num_hidden_layers` | `num_layers` | 直通（P4:74） |
| `num_attention_heads` | `num_attention_heads` | 直通（P4:77；V4:44） |
| `num_key_value_heads` | `num_query_groups` | 缺省：MLA 系→`1`（V4:45 `num_query_groups=1`）/ 否则→`num_attention_heads`；DSv3 显式 `8`（D3:34） |
| `vocab_size` | `vocab_size` | 直通（P4:70；V4:46） |
| `seq_length` | `seq_length` | 直通（P4:71；V4:47） |
| `intermediate_size` | `ffn_hidden_size` | 直通（P4:73；V4:66 `ffn_hidden_size=3072`） |
| `multi_latent_attention` + `experimental_attention_variant` | `attn_type` | 见下「attn_type 推断」（P4:81/98；V4:50） |
| `kv_lora_rank`/`q_lora_rank`/`qk_rope_head_dim`/`qk_nope_head_dim`/`v_head_dim` | 同名直通 | P4:83-87；V4:51-55 |
| （MLA 派生）`qk_nope_head_dim + qk_rope_head_dim` | `head_dim` | **仅 `attn_type="mla"`** 设为 nope+rope（DSv3=128+64=192，D3:38）；`dsv4_hybrid`/GQA **留 None**（V4 未设 head_dim；MLA 下 `head_dim` 对 op 图**惰性**，`attention.py:29-35` 只用 nope/rope/v/lora dims） |
| `csa_compress_ratios` | `csa_compress_ratios` | `list→tuple`（P4:101；V4:57 `_cycle_ratios`） |
| `csa_window_size` | `csa_window_size`（+ `dsv4_hybrid` 时 `window_size`） | P4:102；V4:58/64 |
| `dsa_indexer_n_heads`/`dsa_indexer_head_dim`/`dsa_indexer_topk` | 同名直通 | P4:105-107；V4:59-62 |
| `o_groups`/`o_lora_rank` | 同名直通 | P4:110-111；V4:63/64 |
| `apply_dsa_kernel_fusion` | `dsa_fused` | `bool()`；与 `force_unfused_dsa`（P4:100）互反，不一致则 fail-loud。**FUSED 生产**=True（V4 默认 `dsa_fused=True`，对齐真机 FUSED 锚点 15415.5） |
| `gated_linear_unit` | `gated_linear_unit` | 直通（P4:128；D-6） |
| `moe_intermediate_size` | `moe_ffn_hidden_size` | 直通（P4:129；V4:69） |
| `n_routed_experts` | `num_moe_experts` | 直通（P4:130；V4:67） |
| `num_experts_per_tok` | `moe_router_topk` | 直通（P4:130；V4:68） |
| `n_shared_experts` | `moe_shared_expert_num` | 直通（P4:131；V4:70） |
| `moe_shared_expert_intermediate_size` | `moe_shared_ffn_hidden_size` | 直通（P4:131；V4:71） |
| `first_k_dense_replace` | `first_k_dense_replace` | 直通（P4:130；V4:73） |
| `enable_hyper_connections` / `hc_mult` | `residual_variant` / `num_residual_streams` | True→`"mhc"` + `num_residual_streams=hc_mult`；False→`"plain"`+`1`（P4:113-114；V4:75-76） |
| `num_nextn_predict_layers` | `mtp_num_layers` | 直通（P4:121；V4:78） |
| `add_bias_linear` | `add_bias_linear` | 直通（False=默认；True→build_llm fail-loud）（P4:79） |
| `compute_dtype` | `compute_dtype_bytes` | `bfloat16/float16→2`，`float32→4`（P4:92；V4:80=2） |
| `params_dtype` | `embedding_params_dtype_bytes` + OptimizerSpec | `float32→4`（=默认）；同时定 `params_fp32`（见下）（P4:91） |
| `position_embedding_type` | `position_embedding_type` | rope 族（`rope/yarn/llama3/dynamic/linear`）→`"rope"`（内存等价）；其它→原样透传（build_llm fail-loud）（P4:124=`yarn`） |
| `tie_word_embeddings` | `tie_word_embeddings` | 缺省 False（两锚点均独立 lm_head） |
| `moe_capacity_factor`（缺省） | `moe_capacity_factor` | 无 mf 字段 → 默认 `1.0`（V4:72） |

**attn_type 推断规则**（源：`multi_latent_attention`/`experimental_attention_variant` 如何表达，P4:81/98；
落到 V4:50 `dsv4_hybrid`、D3:39 `mla`）：

1. `multi_latent_attention=True` **且** `experimental_attention_variant="dsv4_hybrid"` → `attn_type="dsv4_hybrid"`。
2. `multi_latent_attention=True`（无 dsv4 变体）→ `attn_type="mla"`。
3. 否则：给了 `num_key_value_heads < num_attention_heads` → `"gqa"`（`num_query_groups=num_key_value_heads`）；
   相等/未给 → `"mha"`（对称，`num_query_groups=num_attention_heads`）。
4. `experimental_attention_variant` 为其它非空值（非 dsv4_hybrid）→ **fail-loud**（未建 op 图）。

**ffn/moe 推断**：`n_routed_experts>0` → 该模型有 MoE 层，逐层 dense/moe 由 `first_k_dense_replace`
（前 K 层 dense）决定（`build_llm._is_moe_layer` 既有语义）；`n_routed_experts` 缺省/0 → 纯 dense。
`gated_linear_unit` 决定 fc1 是否 2×（D-6）。

**非 `model` 段映射**：

| mindformers | 评估器对象 | 规则 / 定位 |
|---|---|---|
| `parallelism.tensor_parallel`/`expert_parallel`/`context_parallel`/`pipeline_parallel` | `ParallelConfig.tp`/`ep`/`cp`/`pp` | 直通（P4:56-57） |
| `parallelism.sequence_parallel` | `ParallelConfig.sequence_parallel` | 直通（P4:58） |
| `parallelism.data_parallel_shard` | `ParallelConfig.dp_shard` | `>0` 直取；`<=0`（auto）→ `global_batch // (local_batch·num_microbatches)`（P4:55；FSDP-only，`dp_replicate=1`）→ 锚点 `2//1=2` |
| `parallelism.pipeline_parallel_microbatch_size` | `ParallelConfig.num_microbatches` | `pp>1` 取该值、否则 `1`（对齐 `validate_dsv3.main` 的 `mbs=PP if PP>1 else 1`）（P4:57） |
| `parallelism.{offset, num_layer_list}` | `ParallelConfig.layers_per_stage` | **D-8 衔接**：per-stage decoder 层数 → 加 embedding@stage0 + (mtp+head)@末 stage（和==`num_layers+mtp+2`）；缺省 None（均匀切） |
| `recompute.mode` + `full_recompute_layer` | `RecomputeSpec.mode`/`full_layers` | 无 recompute 段→`"None"`；`mode="full"`+`["0-K"]`（0-indexed decoder）→ `full_layers={i+1}`（评估器 layer 0=embedding，故 **+1 偏移**）（P4:60；V4:97） |
| `optimizer.type` + `params_dtype` | `OptimizerSpec` | `type→AdamW`；`params_dtype=float32→params_fp32=True`（`state_bytes=12`）；`grad_dtype_bytes=4`（默认 fp32 grad）（P4:52/91；V4:92） |
| `context.max_device_memory` | `HardwareSpec.max_device_memory` | `"54GB"→54·2³⁰`（P4:49；不影响峰值，只判 OOM）；`framework_reserve=0`、`alloc_block_bytes=512` 用默认 |
| （无 swap/offload 段） | `SwapSpec()` | 默认 disabled（swap/offload 映射未实现，见「限制」） |

**fail-loud 策略**（与 `build_llm._check_implemented_dispatch` 一致——不静默产错图）：
- **`model` 段严格白名单**：每个 key 必须落在「已映射集」∪「已知内存中性忽略集」，否则
  `raise NotImplementedError`（列出未识别 key）。这保证「改变 op 图但未映射」的新结构字段**必炸**。
- **值守卫**：`use_flash_attention=False`（会物化 `[S,S]` 分数、改 op 图）→ fail-loud；
  `experimental_attention_variant` 非 `dsv4_hybrid` 的非空值 → fail-loud；`apply_dsa_kernel_fusion`
  与 `force_unfused_dsa` 不互反 → fail-loud。
- **透传守卫**：`add_bias_linear=True`/`add_qkv_bias=True`/`qk_layernorm=True`（若不在忽略集）/
  `normalization≠RMSNorm`/`norm_placement≠pre`/`position_embedding_type` 非 rope 族 → 映射到
  `LLMConfig` 后由 `build_llm` 原生 fail-loud（DRY，不在转换器重复）。
- **非 `model` 段**（`parallelism`/`recompute`/`training`/`optimizer`/`context`/`checkpoint`/`lr_scheduler`/
  `train_dataset`）：只读已知 key、忽略其余（这些只改训练循环/IO，不改 op 图）。

**已知「内存中性」忽略集（`model.*`）= 刻意复现手写映射的省略**（每项均**不改内存 op 图**；`qk_layernorm`
的省略直接照抄 `dsv4_align_config()` docstring「qk_layernorm/add_bias omitted（memory-negligible; would
fail-loud in build_llm）」）：`model_type`/`architectures`/`max_position_embeddings`/`hidden_act`/`rms_norm_eps`/
`mla_qkv_concat`/`qk_layernorm`/`attention_dropout`/`hidden_dropout`/`layernorm_compute_dtype`/
`softmax_compute_dtype`/`rotary_dtype`/`initializer_range`/`csa_compress_rotary_base`/`csa_dense_mode`/
`dsa_indexer_loss_coeff`/`dsa_indexer_use_sparse_loss`/`hc_sinkhorn_iters`/`hc_eps`/`use_fused_mhc`/
`mtp_loss_scaling_factor`/`scaling_factor`/`beta_fast`/`beta_slow`/`mscale`/`mscale_all_dim`/`rope_theta`/
`router_dense_type`/`routed_scaling_factor`/`moe_token_dispatcher_type`/`moe_grouped_gemm`/
`moe_router_load_balancing_type`/`moe_aux_loss_coeff`/`scoring_func`/`norm_topk_prob`/`moe_token_drop_policy`/
`moe_router_enable_expert_bias`/`moe_router_bias_update_rate`/`use_pad_tokens`/`topk_group`/`n_group`/
`force_unfused_dsa`（作 `apply_dsa_kernel_fusion` 的互反校验用，不单独映射）。**新增未在此集也未映射的
`model` key → fail-loud**（不静默吞）。

**round-trip 保真论证**：
- **DSv4-align**：喂 `prep_dsv4align.py`（`N=4, SEQ=2048, MHC=0, MTP=0, FUSED=1`）生成的 dict → 转换器产
  `LLMConfig` 与 `dsv4_align_config(4)` **逐字段相等**（上表全部 40+ 字段核对，`qk_layernorm:True` 落忽略集→
  默认 False、`params_dtype:float32→4`=默认、`apply_dsa_kernel_fusion:True→dsa_fused=True`、`num_query_groups`
  缺省→1、`head_dim` dsv4_hybrid→None、`position_embedding_type:yarn→rope`）→ 同一 bundle（dp_shard=2/
  no-recompute/54GB）→ **峰值 14336.5**。
- **DSv3**：喂重构的 DSv3 mindformers dict（`multi_latent_attention:True` 无 dsv4 变体、`num_key_value_heads:8`、
  MLA dims 128/64/192、`n_routed_experts:8`/`num_experts_per_tok:4`、`recompute.mode:full`+`["0-3"]`、`59GB`）→
  转换器产 `LLMConfig` 与 `deepseek_v3(4)` 逐字段相等（`head_dim=128+64=192`、`num_query_groups=8`、
  `attn_type="mla"`）→ 同一 bundle（dp_shard=2/full-recompute {1,2,3,4}/59GB）→ **峰值 12409.5**。
- 若 round-trip **不能**复现锚点 `LLMConfig`/峰值 → **停并报差异**（静默偏离 verified 手写映射比没有转换器更糟）。

**DSv3 硬门（本模块对 eval core 零改动）**：新增 `cost_eval/configs/`（转换器）+ `tests/test_from_mindformers.py`，
**不碰** `cost_eval` 核任何评估路径 → `validate_dsv3.py` 恒 `12409.5`、`pytest` 全绿必然成立（仍显式跑核验）。
纯核 `from_mindformers_dict` 不 `import yaml`；仅 `load_mindformers_yaml` 内惰性导入 → `cost_eval` 核无新硬依赖。

## 5. 整改完成状态（2026-07-06）

7 项决策全部落地（D-5/D-9 按定夺维持现状），逐项 docs-first + DSv3 硬门 + 独立 commit：

| # | 状态 | commit | 关键效果 |
|---|---|---|---|
| D-1 cp 切激活 | ✅（→ 2026-07-07 修正） | `28229e4` | 激活 S_eff=S/cp（含 flash-ws/loss/index）；cp=1 no-op。**修正**：loss/head 区 full-S（真机 12433，见 §D-1 修正）+ 按 `context_parallel_method` 分派（colossal KV all-gather） |
| D-8 显式 stage 层数 | ✅ | `2d66940` | `layers_per_stage` 首选 + 三重校验，不静默错映射 |
| D-4 VPP 按 chunk | ✅ | `2ac2e4b` | 去 ~V× 过估，按实际 chunk 层数累加；v=1 == 1F1B |
| D-6 ungated FFN | ✅ | `7a2594c` | `gated_linear_unit=False` 建 op 图（fc1 不 2×） |
| D-2 HCCL 接入 | ✅ | `30febaa` | 按通信域数 surfaced 到报告（reserved 口径） |
| D-3 细粒度选重 per-op | ✅ | `73dab0a` | 单 op 自估自身足迹（`_pinned_input_boundary`），修低估 |
| D-7 配置转换器 | ✅ | `4cc33dc` | mindformers yaml → 评估器配置对象；round-trip 两锚点精确 |
| D-5 / D-9 | 维持 | — | DSv4-fused 7% caveat / DSv3 ~0.5% sub-block 残差 |

**不变量守住**：DSv3 锚点全程逐字节 `12409.5`；DSv4-align `14336.5`（0.930，D-5 caveat）；
`pytest` `233 → 288`（+55 例，全绿）。7 目标：审计时 5✅2⚠️ → 整改后 cp/HCCL/配置文件/VPP/
细粒度选重/ungated/显式 stage 全部补齐。
