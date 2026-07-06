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
