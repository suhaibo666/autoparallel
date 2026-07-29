# 下一个该修的问题：诊断与判决（2026-07-28）

> **性质**：诊断书 + 可证伪的量化预言。**只读**——本轮不改任何源码。
> **输入**：`analysis/sim_vs_real_gap_decomposition_2026-07-28.md`（差分拆解）、
> `analysis/handoff_2026-07-25_unfused-recompute-norelease-mechanism.md`（真机微基准机制）、
> `docs/opdag_coverage_close_2026-07-28.md` §5/§7（抽取路径剩余缺口）、
> `docs/to_resolved_adapter_2026-07-25.md`（适配器判决 + 手写 census 台账）。
> **基线**：`feat/unified-llm-modelspec` @ `8849f97`，`python -m pytest tests -q` → 1864 passed。
>
> 记法：**[RAN]** = 我实跑并粘了输出；**[SRC]** = 逐字读源（带 `file:line`）；
> **[INFER]** = 由前两者推断，**不是**观测。铁律：真机数字一律来自 `REAL`（SHA256 指纹保护），
> 不编造、不拟合常数；分辨不了的写「无法判定」。

---

## 0. 结论先行

**下一个该修的问题（唯一一个）：让抽取器能给 `csa.py:485`
（`kv_flat[flat_indices]`，advanced indexing）定型。**
它是 `dsv4hyb_r0_dense`（unfused）层里**唯一**的级联根，闸住的是
[SRC] `csa.py:464-525` `unfused_compressed_sparse_attn` **整个函数体**——
而这个函数被 [SRC] `csa.py:823-825` 对 **r0 / r4 / r128 三种 ratio 共用**，
正是模型 fused↔unfused **全部**差异信号所依赖的那份手写 `saves` 清单
（`cost_eval/layers/dsv4_hybrid.py:188-229`）唯一对应的源码。

**为什么是它、而不是那四条「机制误差」**：我实测到一条拆解文档没有的事实（§2.3）——
unfused 的误差**按层型符号相反**：滑窗层 **过读 2.97×**、HCA 层 **准（1.02×）**、
CSA 层 **欠读 0.68×**。所以「unfused 欠读 10.4 GiB」**不是一个机制**，
而**任何**改公式、加字节的修法都会同时弄坏另外两个层型
（朴素修法量化后果见 §3.4：三格全翻过读，滑窗层到 **5.94×**）。
这份清单必须先被**独立测量**，而唯一不依赖真机、不引入拟合的测法就是让抽取器读源。

**它今天关闭的缺口是 0 MiB**（`extracted` 是 advisory 列）。我认为这是正确的取舍，
论证在 §6.1/§6.2；量化预言钉在覆盖度与逐层 `activation_saves` 上（§5）。

**同时报告一条对输入文档的订正**（§2.1）：差分拆解 §5 第 3 条写「单层成本欠读 40–50%，
**与 (2) 同源（unfused 链的梯度侧）**」——**这是错的**。`a` 与 `g` **两跑都是 fused**
（`tools/liveness_ab_validate.py:115,121` `Variant(..., True, ...)`），故 `a−g` 隔离的是
**多一个 fused 层**的成本，与 unfused fp32 链**没有任何关系**。这条错误归因如果被
下一轮当作「修 (2) 顺带修 (3)」的依据，会直接把收益预期算错。

---

## 1. 我实跑的基线 [RAN]

```bash
cd /e/97-codes/torch_parallel/pynative-cost-evaluator
PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas --gate
```

逐格与 `analysis/sim_vs_real_gap_decomposition_2026-07-28.md` 一致：

```
  bucket       n=28  mean=0.946  min=0.748  max=1.394
  extracted    n=0  （该来源无任何可评分格：解不出图或全 OOM）
  hand_spec    n=28  mean=0.955  min=0.729  max=1.400
  验收门结论: PASS
```

`extracted` 在门里 8/8 跑 `ERR`：`IncompleteExtraction @compressor.py:216`（级联根
`Elementwise compressor.py:216 ×4`）。即差分拆解 §6 与 `opdag_coverage_close` §5 第 1 项
所说的那块石头，在 HEAD 上仍然原样挡着（默认不给数；`COST_EVAL_EXTRACTED_ALLOW_PARTIAL=1`
才出下界）。

---

## 2. 我新测到的、拆解文档里没有的东西

### 2.1 订正：对比量 (3)「每多一层的成本」**不是** unfused 机制，是 **fused** 路径的问题 [SRC]

`tools/liveness_ab_validate.py:115` `Variant("a fused   ON  L8 m4", True, 8, 4, True, 8)`
`tools/liveness_ab_validate.py:121` `Variant("g fused   ON  L4 m4", True, 8, 4, True, 4)`

第 2 个字段是 `fused`。**`a` 与 `g` 都是 `True`**。所以 `a − g` 隔离的是「同一 stage 多一个
**fused** transformer 层」的边际成本，**与 unfused fp32 复本链无关**。

拆解文档 §5 第 3 条「单层成本：欠读 40–50%，**与 (2) 同源（unfused 链的梯度侧）**」
**归因错误**。(3) 是一条**独立**的机制误差（fused 全重算下每层边际成本欠读 ~2200–3000 MiB），
必须单独立项，不能算进 (2) 的收益里。

### 2.2 `fml` 看不见 saves —— 机制**属实**，且可以逐字节点名 [RAN][SRC]

[SRC] `cost_eval/structure_mem.py:163-170`：

```python
for i, op in enumerate(ops):
    acts = [t for t in op.inputs if not t.is_weight]
    acts.append(op.output)                      # output 恒为激活
    for t in acts:
        ...
```

`op.saves` **从不出现**在这个 walk 里。故任何**只在 `saves` 里声明、从不作为某 op 的
input/output 出现**的张量，对 `forward_max_live` 的贡献**恰好是 0**。

[RAN] 我逐张量点了名（`scratchpad/probe_layers.py`，只读，直调 `ShapeEval().resolve` +
`estimate_structure_memory`）：

| 层 | fused：只在 saves（fml 看不见） | unfused：只在 saves | 差 |
|---|---:|---:|---:|
| `dsv4hyb_r0_dense` | 897.0 MiB / 5 项 | **6144.0 MiB / 16 项** | +5247.0 |
| `dsv4hyb_r128_moe` | 901.0 MiB / 6 项 | **6944.0 MiB / 16 项** | +6043.0 |
| `dsv4hyb_r4_moe` | 970.5 MiB / 9 项 | **19840.0 MiB / 19 项** | +18869.5 |

unfused r4 的那 19840.0 MiB 逐项（`cost_eval/layers/dsv4_hybrid.py:197-228` 声明）：
`kv_g_fp32` 4096 / `ukv_bm` 4096 / `index_scores` 2048 / `kv_gathered` 2048 /
`ukl1` 1024 / `ukl2` 1024 / `attn_weights`·`cg_fp32`·`uaw_bm`·`uexp`·`uout_f32`·`uout_pm`·
`uq_bm`·`uq_f32`·`uscore1`·`uscore2` 各 512 / `inv_rope_*` 等小项。

对应 [SRC] `mf-src-167/.../csa.py:490` `kv_g = ops.cast(kv_gathered, mstype.float32)`，
被 `:495` `kv_bm = reshape(permute(kv_g,(0,1,3,2)), (b·sq,d,topk))`（permute 后 reshape →
**真复本**）与 `:518` `kvo_bm = reshape(kv_g,(b·sq,topk,d))`（连续 reshape → **视图**）
两处消费；`b=1,sq=4096,topk=512,d=512` fp32 = **4096 MiB** 逐字节对上。

**但——这个机制不能按「加梯度字节」去修。** 见 §3.4。

### 2.3 **决定性新数据**：unfused 误差按层型**符号相反** [RAN]

L4（每 stage 恰好 1 个 transformer 层）把「一个层型的 unfused 增量」隔离得干干净净。
`h − g` 在 s0/s1/s2 **峰值事件同为 `bwd@<该层>`、除 `remat_saves` 外每个桶逐字节相同**
（我逐桶打印核对过），所以这三格是**真正的单层隔离**：

| stage | 层型 | 模型 Δ（= Δ`remat_saves`） | 真机 Δ | 模型/真机 | 误差 |
|---|---|---:|---:|---:|---:|
| s0 | `dsv4hyb_r0_dense`（滑窗 unfused） | **5243.0** | 19862.6−18096.8 = **1765.8** | **2.97× 过读** | −3477.2 |
| s1 | `dsv4hyb_r4_moe`（CSA，带 indexer） | **18865.5** | 36582.6−8867.9 = **27714.7** | **0.68× 欠读** | +8849.2 |
| s2 | `dsv4hyb_r128_moe`（HCA，无 indexer） | **6039.0** | 13750.2−7845.3 = **5904.9** | **1.02× ≈ 准** | −134.1 |

（`g`/`h` 各 stage 的基线残差本身在 ±2300 量级：g = +1871.2/−1609.8/−2227.3/+680.6。
故 r128 的 −134.1 **在噪声内、无信号**；r0 的 −3477.2 与 r4 的 +8849.2 **在噪声外**。）

**这一条推翻了「unfused 欠读 10.4 GiB 是一个机制」的框架。** 它是**三个**误差之和，
其中两个符号相反。聚合数 −10.4 GiB 是抵消后的结果，和总比值 0.946 是同一种假象。

**同时它也解释了 `hand_spec`（liveness·chain2，显式建梯度）为什么不是纯改善**：

| 层型 | bucket Δ | hand_spec Δ | 真机 Δ |
|---|---:|---:|---:|
| r0 | 5243.0（2.97× 过） | **6046.9（3.42× 过）** | 1765.8 |
| r4 | 18865.5（0.68× 欠） | **23982.3（0.87× 欠）** | 27714.7 |
| r128 | 6039.0（1.02×） | **7562.8（1.28× 过）** | 5904.9 |

liveness 的显式梯度**方向是对的**（r4 从 0.68 抬到 0.87，补回 5116.8 的 58%），
但它把梯度加在一个**本身就错的 saves 清单**上 → r0/r128 反而更过读。
**梯度公式不是根因；根因在那张清单。**

### 2.4 抽取路径：级联根**不是一个**，是**三个，与层型一一对应** [RAN]

`resolve_graph(spec, pm, allow_partial=True)` 逐层 `Coverage`（run b，unfused L8）：

```
CHILD tag=L0:embedding            ops=12   skipped=0
CHILD tag=L1:dsv4hyb_r0_dense     ops=162  skipped=42    x1 IndexSelect  csa.py:485        constant_shape_unknown
CHILD tag=L2:dsv4hyb_r4_moe       ops=280  skipped=173   x1 Elementwise  compressor.py:216 needs_axis_structure
CHILD tag=L3:dsv4hyb_r128_moe     ops=212  skipped=92    x1 View         compressor.py:233 slice_bounds_unknown
CHILD tag=L4/L6/L8 (r4)           同 L2                  x1 Elementwise  compressor.py:216
CHILD tag=L5/L7   (r128)          同 L3                  x1 View         compressor.py:233
CHILD tag=L9:lm_head              ops=26   skipped=0
TOTALS: n_nodes 3976 → n_ops 1956（skipped 1010）；unresolved_saves 0；unresolved_params 0
逐层 extracted activation_saves：r0 1067.6 / r4 650.5 / r128 566.9 / lm_head 2052.0 MiB
（对照 hand_spec：r0 8468.2 / r4 22045.3 / r128 9140.3 / lm_head 3094.0）
```

**订正 `opdag_coverage_close_2026-07-28.md` §6 的「唯一挡在前面的大石头」**：
`sym_shape` 的 `4·(S//4)==S` 只挡住 **4 个 r4 层**；**r128 的 3 个层**卡在
`compressor.py:233`（rope 频率表切片 `freqs[:total_seq_len:ratio][:n_compressed]`，
`slice_bounds_unknown`），**r0 的 1 个层**卡在 `csa.py:485`（advanced index，
`constant_shape_unknown`）。三条**互不相干**。

---

## 3. 候选排序与判决

### 3.0 排序表

| # | 候选 | 关闭的缺口 | 根因确信度 | 破锚风险 | 在抽取架构关键路径上？ | 成本 | 判决 |
|---|---|---|---|---|---|---|---|
| **(1)** | 重算节省高估 2.4–3.4× | 名义 14796.8（s0） | — | — | 否 | — | **不是一个问题**：拆解文档自己写明 = (4) 的过读 + run a 的欠读之和。排除 |
| **(2)** | unfused 物化欠读 10.4 GiB | 8–12 格 × ~10.4 GiB | 机制 [RAN] 属实；**修法被实测证伪**（§3.4） | **高** | 否 | 中 | 今天**无可辩护落地形态**。降级为「等测量」 |
| **(3)** | 每多一层欠读 40–50% | 586–2975/层 | 低（**归因被我推翻**，§2.1；真根因未定） | 中 | 否 | ? | 需先重新立项，本轮无法判 |
| **(4)** | 无重算 `act_live` 过读 1.4–1.6× | 4 格，最大 9950.6 | **无法判定**（深度 vs 每层驻留，拆解文档已声明；我复核确认 §3.3） | 高 | 否 | 需真机探针 | 需数据，不能猜 |
| **(5)** | `sym_shape` 不知 `4·(S//4)==S`（`compressor.py:216`） | **0 MiB**（advisory 列） | 高 | **零** | **是** | **高**（新代数能力） | 好，但**不是最优的那一条**（§3.5） |
| **(5′)** | **`csa.py:485` advanced-index 定型（本文推荐）** | **0 MiB**（advisory 列） | **高，且我把 file:line 全钉住了** | **零** | **是** | **低**（一条 walker 规则） | ✅ **本轮判决** |

### 3.1 判决：**(5′) —— 修 `csa.py:485` 的 advanced-index 定型**

**一句话**：模型在 unfused/fused 上的**全部**差异信号，都来自一个手写猜测的
`saves` 清单；那份清单在滑窗层过读 2.97×、在 CSA 层欠读 1.47×、在 HCA 层刚好 ——
**没有任何公式改动能同时修好这三个**。要修它必须先**独立测**它，而唯一不依赖真机、
不引入拟合的测法是让抽取器读源；抽取器读它的入口被**一个节点**挡着，就是 `csa.py:485`。

### 3.2 为什么整个 unfused 信号都住在那份清单里 [RAN]

逐桶截面（`scratchpad/probe_stages.py`）：`h`（unfused L4）与 `g`（fused L4）在
s0/s1/s2 **峰值事件相同**，且 `persistent` / `act_live` / `gather_buf` / `grad_buf` /
`recomp_scratch` / `bwd_scratch` / `bwd_working_set` / `optstep` / `grad_accum`
**逐字节相同**，唯一不同的桶是 `remat_saves`：

```
h s0 remat_saves 8864.2 | g s0 3621.2      Δ 5243.0
h s1 remat_saves 22441.3| g s1 3575.8      Δ 18865.5
h s2 remat_saves 9536.3 | g s2 3497.3      Δ 6039.0
```

而 `remat_saves = max(0, activation_saves − checkpoint_input)`
（[SRC] `cost_eval/mem_timeline.py:709-710`），`checkpoint_input` 两支相同（128.0 MiB）
→ **Δ`remat_saves` ≡ Δ`activation_saves` ≡ 那份手写清单的差**。

### 3.3 为什么不选 (4)：我复核了「深度 vs 每层」，确认**判不了** [RAN]

按「其余桶正确」反解真机 `act_live`（run c；模型 `act_live` = 深度 × 层数 × 单位，由实测 s1 `act_live`=21987.5 与深度 3 × 2 层
反解单位 = **3664.58 MiB/(层×微批)**；s0 8×3664.58=29316.6 vs 实测 29812.2 含 embedding 项，
s2 4×3664.58=14658.3 ✓ 逐格对上）：

| stage | 模型 层×微批 | 反解真机 层×微批 | 比 |
|---|---:|---:|---:|
| 0 | 8 | 5.42 | 0.678 |
| 1 | 6 | 3.74 | 0.623 |
| 2 | 4 | 2.85 | 0.713 |

**既不是整数偏移**（8→6→4 的整数下调给 6/4/2 = 0.75/0.667/0.50，对不上），
**也不是常数比例**（0.678/0.623/0.713 三点不齐）。我另试了「深度对、每层驻留小」
与「(depth−1)+部分」两种线性假设，三点都拟不上（(8,5.42)(6,3.74)(4,2.85) 非线性）。
→ **拆解文档 §4.1「不硬判」的结论我复核成立**。且 run d OOM ⇒ 无重算只剩 run c
**4 个可评分格**，任何在 4 点上定的改动就是拟合。**排除。**

### 3.4 为什么不能按「让 `fml` 看见 saves」去修 (2) —— 量化反证 [RAN]

我实算了这个最朴素的落地形态（把 `saves` 也纳入 max-live 走查，存活区间取
「产出 op → 前向末」，即"必须活到反向消费"的语义）：

| 层 | 现 `fml` | 反事实 `fml⁺saves` | 现 `bwd_working_set` | 反事实 |
|---|---:|---:|---:|---:|
| r0 fused / unfused | 800.1 / 800.1 | 2617.1 / 7860.1 | 544.1 / 544.1 | 2361.1 / 7604.1 |
| r128 fused / unfused | 800.1 / 800.1 | 2461.2 / 8500.1 | 544.1 / 544.1 | 2205.2 / 8244.1 |
| r4 fused / unfused | 808.1 / 808.1 | 2523.6 / 21388.1 | 552.1 / 552.1 | 2267.6 / 21132.1 |

于是 `h − g` 的模型 Δ 变成（现值 + 反事实增量）：

| stage | 现模型 Δ | 反事实 Δ | 真机 Δ | 反事实/真机 |
|---|---:|---:|---:|---:|
| s0 (r0) | 5243.0（2.97×） | **10486.0** | 1765.8 | **5.94× 过读** |
| s1 (r4) | 18865.5（0.68×） | **37730.0** | 27714.7 | **1.36× 过读** |
| s2 (r128) | 6039.0（1.02×） | **12077.9** | 5904.9 | **2.05× 过读** |

**三格全部翻成过读**，且原本准确的 r128 被打成 2.05×。
（fused 侧也会崩：run a s1 的 `bwd_working_set` 552.1 → 2267.6，峰值 14033.1 → 15748.6，
`sim/real` 0.958 → **1.076**。）

**结论：「fml 看不见 saves」是真的代码事实，但它不是可修的那一端。**
`remat_saves` 已经把这批字节收了一次；再从 `fml` 收一次就是双算。

### 3.5 为什么选 `csa.py:485` 而不是 `compressor.py:216`

| | `compressor.py:216`（= 候选 5） | **`csa.py:485`（= 5′，推荐）** |
|---|---|---|
| 挡住的层 | 4 个 **r4** | 1 个 **r0** |
| 挡住的东西 | 压缩器（上游管道） | **`unfused_compressed_sparse_attn` 函数体本身**（[SRC] `csa.py:464-525`），即那份手写清单**唯一**对应的源码 |
| 该函数被谁调用 | — | [SRC] `csa.py:823-825` `_construct_naive` **对 r0/r4/r128 全部 ratio 调用同一个函数** → 读通一次，三种 ratio 的清单都可校 |
| 修法性质 | `sym_shape` 需要**带 DimTable 的整除规范化 pass**（`opdag_coverage_close` §5 自己写明「是代数层的**新能力**，不是一条规则」，作者在此停手） | walker 把常量构造的 shape 实参记进 attrs，喂给**已经存在**的 `_advanced_index`（[SRC] `shape_infer.py:1195-1201`）—— **一条规则** |
| 修完还剩多少 | r4 修通后**紧接着**还会撞上 `csa.py:485`（同一函数体） | r0 层只有这一个级联根（`skipped 42/162`，26%） |
| 真机对照的干净度 | r4 在 L8 各 stage 与其它层共存 | **`h s0 − g s0` 是全表最干净的单层隔离**：峰值事件相同、除 `remat_saves` 外逐桶逐字节相同 |

即：**`compressor.py:216` 是 `csa.py:485` 的前置，反过来不成立。** 先修便宜的那一个，
拿到共享函数体的源码判决，再决定 `compressor.py:216` 值不值得那份代数能力。

---

## 4. 根因，逐条钉到代码

### 4.1 直接根因（要修的那一条）[RAN][SRC]

[RAN] run b 的 `L1:dsv4hyb_r0_dense` 逐层覆盖度报告：

```
覆盖度 L1:dsv4hyb_r0_dense：节点 246 → op 162（跳过 42）；saves 未解析 0；params 已解析 13 / 未解析 0
  节点原因码: no_input_shape=29, reshape_unresolved=11, constant_shape_unknown=1
  级联根: ×1  IndexSelect  csa.py:485  constant_shape_unknown
```

**整个 L1 层里 `constant_shape_unknown` 只有 1 个节点，就是 `csa.py:485`。**
另外 40 个被跳过的节点（`no_input_shape=29` + `reshape_unresolved=11`）是它的**连带**
——数量与形态正好是 `unfused_compressed_sparse_attn` 函数体
（[SRC] `csa.py:489-525`：permute/reshape ×11、bmm/exp/div/max/sum 等 ×~29）。

[SRC] 被卡的节点：`csa.py:485`
```python
kv_gathered = mint.reshape(kv_flat[flat_indices], (b, sq, topk, d))
```
`kv_flat[flat_indices]` 是 **advanced indexing**。

[SRC] 处理它的规则**已经存在**：`cost_eval/opdag/shape_infer.py:707-731` `_advanced_index`，
docstring 逐字写着「真源 `csa.py:485` `kv_flat[flat_indices]`」，规则 =
`idx.shape ++ x.shape[1:]`。它返回 `None` 时落到
`shape_infer.py:1195-1201`：
```python
r = _gather(n, in_axes_list, in_numel_only)
if r is None:
    r = _advanced_index(n, in_axes_list, in_numel_only)
if r is not None:
    return r
_note(ctx, n, "constant_shape_unknown",
      "gather/advanced-index 的产出形由 index 形状决定,attrs 未记轴")
```

`_advanced_index` 有四道守卫（`shape_infer.py:719-729`）：① `attrs["advanced_index"]` 为真；
② 恰两个张量操作数；③ 两侧轴结构可信（非 `~`）；④ index dtype 是整数。

**[INFER]**（我未能实测判定是哪一道）：记下的消息「attrs **未记轴**」指向 ①/③。
index 侧的生产链是 [SRC] `csa.py:483-484`
```python
batch_offset = mint.arange(b, dtype=mstype.int64).unsqueeze(1).unsqueeze(2) * sk
flat_indices = mint.reshape(safe_indices + batch_offset, (-1,))
```
`mint.arange(b, …)` 属 `Constant`，其 shape 实参若未被 walker 记进 attrs，
`_constant`（`shape_infer.py:630+`）返回 `None` → `batch_offset` 无轴 → 广播加无轴 →
index 无轴 → 守卫 ③ 失败。**实现者第一步应该打印这个节点的 `attrs`，确认是 ① 还是 ③
——我不猜。**

### 4.2 为什么这一个节点值这么多 [SRC]

[SRC] `csa.py:823-825`（`_construct_naive`）：
```python
output = unfused_compressed_sparse_attn(
    query, kv_full, self.attn_sink, topk_idxs, self.softmax_scale
)
```
**对 r0 / r4 / r128 三种 ratio 调用的是同一个函数**（分支只在上游造 `kv_full` / `topk_idxs`：
`:745-751` compressor 只在 `enable_compress` 时接、`:762-820` indexer 只在 `ratio>1` 时接）。
而 `cost_eval/layers/dsv4_hybrid.py:197-206` 的那 9–12 个 `_ucopies`（`uq_f32`/`uq_bm`/
`ukv_bm`/`uscore1`/`uscore2`/`uexp`/`uaw_bm`/`uout_f32`/`uout_pm`）+ `kv_g_fp32`/
`kv_gathered`/`attn_weights` 就是**这个函数体的逐 op 手写普查**，其依据只有一行注释
（`dsv4_hybrid.py:189` 「真机 `unfused_compressed_sparse_attn` 逐 op 的 bprop **持有其输入/输出**」）
—— 那是**最坏情形假设**，不是源读。

而本仓库**已经有**逐 op 类型的 VJP 保留规则：[SRC] `cost_eval/opdag/bprop_rules.py:22-72`
`PIN` 表 + `:74-155` `derive_saves`，其中与本函数相关的：
- `"View": {"inputs": []}`（reshape/permute **不保留任何东西**）
- `"Cast": {"inputs": []}`（cast 自身不存，其输出由下游消费者 pin）
- `"BMM": {"inputs": "all"}`（两个操作数都保留）
- `"IndexSelect": {"inputs": [1]}`（只存 **index**，不存被索引的数据；注释里逐字点名 `csa.py:485`）
- `Elementwise`：`derive_saves` `:80-82` —— `attrs["linear"]` 为真（加法）不存，非线性（乘/除）存全部操作数

**这两套东西对同一段源码给出不同答案。** 手写普查说「每个中间量都留」；仓库自己的
VJP 表说「View/Cast 链上的复本不留」。真机（§2.3）站在 VJP 表那一边（r0 过读 2.97×）。
**修 `csa.py:485` 就是让第二套去实测第一套。**

### 4.3 派生根因（不修，但要记账）

- [RAN][SRC] `structure_mem.py:163-170` `_forward_max_live` 不遍历 `op.saves`
  → `bwd_working_set` = 552.1 而非 GiB 级。**机制属实，但 §3.4 已证不能从这一端修。**
- [SRC] `mem_timeline.py:681-710`：全重算层的三个桶
  （`recomp_scratch` = `fml − ckpt`、`bwd_working_set` = `fml − bwd_scratch`、
  `remat_saves` = `activation_saves − ckpt`）**全部**由 `fml`/`saves` 派生且
  代码注释自认「部分重叠」（`:707-708`）。这是 §3.4 双算风险的出处。

---

## 5. 可证伪的量化预言

> ⚠ 本条修**不移动任何一格 8 跑数字**（`extracted` 是 advisory 列，且修完仍
> `IncompleteExtraction`）。这**不是**回避：预言改钉在**可直接核对的抽取量**上，
> 见 P1/P2；同时 P3 把「不许动的东西」也写成硬预言。

### P1 — 覆盖度（`resolve_graph(..., allow_partial=True)`，run b）

| 量 | 现值 [RAN] | 预言 | 证伪条件 |
|---|---:|---|---|
| `L1:dsv4hyb_r0_dense` 跳过 op | **42** | **≤ 5** | 若 > 22（即降幅 < 50%），「一个节点闸住整个函数体」**错** |
| L1 `constant_shape_unknown` | **1** | **0** | > 0 → 没修到点上 |
| L1 `no_input_shape` | **29** | **≤ 3** | > 12 → 连带链没打通，另有独立根 |
| L1 `reshape_unresolved` | **11** | **≤ 2** | > 6 → 同上 |
| 全局 `skipped_ops` | **1010** | **968 ± 8** | 出界 → 影响面与预期不符 |
| 全局级联根 | `×4 compressor.py:216` / `×3 compressor.py:233` / `×1 csa.py:485` | **`csa.py:485` 那条消失**，只剩 ×4 + ×3 | 仍在 → 未修好 |

### P2 — 抽取侧 r0 层的 `activation_saves`（**这一条才是判决书**）

| 量 | 现值 [RAN] | 预言 | 依据 |
|---|---:|---|---|
| `extracted` L1 `activation_saves` | **1067.6 MiB** | **落在 [3000, 5000]，中心 ≈ 3900**；且**严格小于** `hand_spec` 的 **8468.2** | [INFER] 按 `bprop_rules.PIN` 手工过 `csa.py:464-525` |

逐张量预言（r0，W = csa_window_size = **128**，b=1、sq=4096、d=512，由
`kv_gathered [B,S,W,vd]` bf16 = 512.0 MiB 反解得 W=128）：

**应当出现在 extracted saves 里**（≈ 2944 MiB）：
`q_bm`(bmm 操作数) 512 · `kv_bm`(bmm 操作数) 1024 · `kv_g`/`kvo_bm`(第二个 bmm 的操作数) 1024 ·
permute 后的 `scores`（被 `*softmax_scale` 这个非线性 Elementwise pin）128 ·
`exp_scores`（被 `/sum_exp` pin）128 · `aw_bm`(bmm 操作数) 128

**应当 _不_ 出现**（手写普查有、共 **2304 MiB**）：
`uq_f32`(=`csa.py:489` 的 `q`，Cast 产出、只被 View 消费) 512 ·
`kv_gathered`(:485 产出、只被 Cast 消费) 512 · `attn_weights`(:514 产出、只被 View 消费) 128 ·
`uscore1`(:496 bmm 产出，BMM 不存自身输出) 128 · `uout_f32`(:519) 512 · `uout_pm`(:520) 512

**可能新增**（手写普查没有）：`scores − scores_max`（:511 `exp` 的输入，`Activation:{"inputs":[0]}`）128。

⇒ 净预言：源读的函数体 saves ≈ **5248 − 2304 + 128 = 3072 MiB**，而手写普查是 **5248**。
把它接回模型，`h s0` 的 Δ`remat_saves` 会从 **5243.0** 降到 ≈ **3067**，真机是 **1765.8**
→ `2.97×` 过读收敛到约 **1.74×**（**不是**修到 1.0；剩下的 ~1300 MiB 另有出处，见 §7）。

### P3 — 必须**不**动的（纪律预言，任何一条破了就退回）

| 量 | 必须保持 |
|---|---|
| 8 跑 × 4 stage 的 `bucket` 峰值 | **逐格 ≤0.05 MiB 不变**（打印精度） |
| 8 跑 × 4 stage 的 `hand_spec` 峰值 | **逐格 ≤0.05 MiB 不变** |
| 聚合 | `bucket 0.946 / min 0.748 / max 1.394`、`hand_spec 0.955 / min 0.729 / max 1.400` 一字不改 |
| 真机不变量 I1/I2 + `REAL_SHA256` | PASS，指纹不变 |
| `bucket` / `hand_spec` 的 ×1 不变量 | 全 PASS（L8=L4=14689.8 / 16346.5；m4=m8） |
| `extracted` 在门里 | **仍然 8/8 ERR**（`compressor.py:216`/`:233` 还在）→ I1/I2 仍是 **SKIP**，**不得**变成 PASS（那意味着有人拿下界冒充了峰值） |
| `scratchpad/dump_numbers.py` before/after | `diff` **为空** |

---

## 6. 反对我自己这个判决的最强论证，以及什么能证伪根因

### 6.1 最强反对：「它今天关闭 0 MiB 缺口」

**成立。** 我的回答分两层：

1. 今天**每一条**能移动格子的候选，其根都落在同一份**手写猜测清单**上，而这份清单在
   三个层型上的误差**符号相反**（§2.3）。在这种结构下，让聚合数变好**恰恰是危险的**
   —— `0.946` 这个"好看的均值"就是这么来的（拆解文档 §0 自己说「总比值是假象」）。
2. 项目铁律是「绝不编造拟合常数 / 源忠实」。在只有 28 格、其中 4 格无重算、
   12 格 unfused 的样本上去调一份 15 项/层的张量清单，**定义上就是拟合**。

### 6.2 第二强反对：「同样的信息，直接手读 `csa.py:464-525` 对着 `bprop_rules.PIN`
改 `dsv4_hybrid.py` 就行，根本不用动 walker，还便宜 10 倍、还能移动格子」

**这条最难反驳，我先承认它在"本周把数字弄好看"的目标下是对的。**
但我把它的后果算出来了 —— 它会**弄坏两个层型**：

| 层型 | 手写普查现值 | 按 `PIN` 手改后 [INFER] | 模型 Δ 现 → 改后 | 真机 Δ | 比值 现 → 改后 |
|---|---:|---:|---|---:|---|
| r0 | 5248 | 3072 | 5243.0 → **3067** | 1765.8 | 2.97× → **1.74×** ✅ 好转 |
| r128 | 6048 | 3712 | 6039.0 → **3703** | 5904.9 | 1.02× → **0.63×** ❌ 变坏 |
| r4 | 18944 | 14848 | 18865.5 → **14770** | 27714.7 | 0.68× → **0.53×** ❌ 变坏 |

（`PIN` 判定：保留 bmm 的两个操作数、非线性 Elementwise 的操作数、`Activation` 的输入；
丢掉 View/Cast 链上的复本与 bmm 自身输出。逐张量清单见 §5 P2。）

**三分之二变坏。** 而且它顺带**证明了一件重要的事**：r4/r128 的欠读**不可能**是
saved 集的问题 —— 按源规则算，saved 集只会更小。所以 (2) 里 r4 那 +8849.2
**必然住在「工作集/梯度」侧**，而不是 `remat_saves` 侧。这正是拆解文档 §4.2
猜对了方向（"在反向工作集"）但归错了修法（不能从 `fml` 那端加）。

⇒ **手改可以做，但绝不能单独 ship**：门（28 格、三层型误差反号）**没有能力**裁决它。
先让抽取器读出那份清单，再决定手写侧改成什么。

### 6.3 其它反对

- **「r0 只是 8 层里的 1 层，还不是生产配置的主力层型」** —— 属实。回答：被闸住的
  `unfused_compressed_sparse_attn` 是 [SRC] `csa.py:823-825` 对**三种 ratio 共用的同一个函数**，
  读通一次的判决可以直接搬到 r4/r128 的手写清单上；而 r0 是**唯一**离抽取器只差一个节点的层。
- **「(4) 值 9950.6，是你 r0 发现的 3 倍，你却不碰」** —— 因为它在 4 个格上判不了（§3.3），
  碰它就是发明常数。它需要的是真机逐事件/逐微批驻留探针，那是**并行**可做的事，不是"下一个修"。
- **「P2 是你手工套 PIN 表推的，不是实测」** —— 承认。所以我把预言写成**逐张量在场/不在场**
  清单，而不是只给一个总数：实现者可以逐条核，错在哪一条一眼可见。

### 6.4 什么能证伪我的根因（具体、可执行）

| # | 观测 | 若成立，则 |
|---|---|---|
| **E1** | 补上 advanced-index 定型后，L1 的 `skipped_ops` 仍 > 22（现 42） | 「一个节点闸住整个函数体」**错**：另有独立闸门（最可能是 `csa.py:440-461` `get_window_topk_idxs` 里 `:457 mint.full(matrix.shape, -1, …)`）。整条推荐的性价比崩塌 → 改推 §6.2 的手改 |
| **E2** | 抽取侧 L1 `activation_saves` ≥ `hand_spec` 的 8468.2 | 「手写普查在 r0 上过读」**错** → r0 那 −3477.2 必须另找出处（最可能是峰值事件迁移，见 E4） |
| **E3** | `uout_f32`/`uout_pm`/`uq_f32`/`kv_gathered` 的对应张量**出现在**抽取侧 saves 里 | 我对 `bprop_rules.PIN`（View/Cast 不 pin）的读法**错**，§6.2 的表全废 |
| **E4** | 真机逐事件内存曲线显示 `g s0` 与 `h s0`（或 `g s2`/`h s2`）**峰在不同事件** | §2.3 那张「按层型符号相反」的表**不是单层隔离** → 本文最重要的那条证据作废，判决要重做。**这是唯一能一击掀翻全文的观测** |
| **E5** | 有人在 167 复跑 a–h，`peak_alloc_MiB` 与 `REAL` 表不符 | `REAL_SHA256` 必须显式改，本文全部差分重算 |

---

## 7. 我排除不掉的干扰项

1. **【最大】峰值事件迁移**。差分法默认「真机在两跑里峰在同一时刻」。模型侧我核对过
   s0/s1/s2 事件相同，但**真机侧没有逐事件数据**（`REAL` 只有 `peak_alloc_MiB` 一个标量）。
   `h s3` 的模型峰值事件确实**从 `bwd@5` 移到了 `bwd@4`** —— 证明事件迁移在这个量级上
   **真会发生**。所以 §2.3 的三个数各自带一个**未知大小**的迁移误差。→ E4。
2. **(2) 与 (3) 是不是同源？** —— **(3) 与 (2) 肯定不同源**（a、g 都 fused，§2.1，已定案）。
   但 **(2) 里 r4 的 +8849.2 与 (3) 的 +2218…2975/层 有可能同源**：两者都可以由
   「真机上相邻层的重算工作集没有严格 ×1、有重叠」解释。**我判不了。**
   约束：真机 I1（stage3 的 unfused−fused delta 在 L8/L4 都是 24380.1）说**那个差**是 ×1 的，
   但它不排除 fused 基线本身有一项随层数走的量。
3. **修 (2) 会不会弄坏别的**：会，而且我量化了 —— §3.4（朴素 fml 法：三格全翻过读，
   r0 到 5.94×）与 §6.2（PIN 手改：r128 1.02→0.63、r4 0.68→0.53）。
   现有**过读**格子一旦被"只加字节"的修法推得更远：`g s1 −1609.8`、`g s2 −2227.3`、
   `h s0 −1606.0`、`h s2 −2361.4`、`c s0/s1/s2 −9950.6/−8273.9/−4205.0`，以及
   L4 s0 的 delta 已经过读 **2.97×**。
4. **口径混用**：我的逐层 `activation_saves`（3179.8 / 22045.3 …）是
   `norm_compute_dtype_bytes=0` 档；`mem_timeline` 内部用 norm-fp32 抬升档
   （3703.8 / 22569.3）。**两档的差值逐字节相同**（我用的全是差值，且与门里
   `remat_saves` 的差逐格对上），但**绝对值不可跨档比较**。
5. **`extracted` 的结构性天花板**：`workspace_bytes` / `bwd_scratch_bytes` 按契约恒 0
   （`to_resolved_adapter_2026-07-25.md` §5.1、`opdag_coverage_close` §5 第 4 项）。
   即使图全解出来，`extracted` 的**峰值**也不能与真机比。**所以我的预言 P2 特意钉在
   逐层 `activation_saves` 上，而不是峰值。**
6. **抽取图与手写图的粒度不同**：抽取把手写的一个 `sparse_attn` op 摊成 ~30 个节点，
   全局有 168 处 `name_disambiguations`（同名异形按 `name#k` 拆）。这会让两侧的
   saves 总数在**未量化**的幅度上偏移。P2 的区间 [3000, 5000] 已经留了这个余量。
7. **run d 永久不可评分**（真机 OOM）→ 「unfused × 无重算」这一格**永远**没有真值，
   (2) 只能在重算制度下被验证。
8. **`csa.py:485` 卡在四道守卫里的哪一道，我没实测判定**（§4.1）。若是守卫 ①
   （walker 没打 `advanced_index` 标记），修法在 `construct_walker.py`；若是守卫 ③
   （index 无轴），修法在常量构造的 attrs 记录。**两者工作量不同，实现者要先打 attrs。**

---

## 8. 怎么验收这个修

### 8.1 硬门（必须全绿）

```bash
cd pynative-cost-evaluator
PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas --gate
python -m pytest tests -q            # 基线 1864 passed
```

* **28 格 `bucket` + 28 格 `hand_spec` 逐格 ≤0.05 MiB 不变**，聚合 0.946 / 0.955 不变（P3）。
* `REAL_SHA256` 不变；真机 I1/I2 PASS。
* `bucket`/`hand_spec` 的 **×1 于层数 / ×1 于微批数** 全 PASS（14689.8 / 16346.5；m4≡m8）。
* `extracted` **仍然 8/8 ERR**（`compressor.py:216`/`:233` 未修）→ 其 I1/I2 仍是 **SKIP**。
  **若它变成 PASS，立即退回** —— 那意味着有人让一个下界冒充了峰值
  （`to_resolved.py:1432-1443` 的 `IncompleteExtraction` 纪律被绕过）。
* 逐字节中立：`git worktree` 起 `8849f97`，两侧跑 `scratchpad/dump_numbers.py`，`diff` **必须为空**。

### 8.2 收益门（P1/P2，用 `allow_partial` 诊断口径）

```bash
COST_EVAL_EXTRACTED_ALLOW_PARTIAL=1 PYTHONIOENCODING=utf-8 \
  python tools/liveness_ab_validate.py --grad-mode chain2 --deltas --gate
# 逐层：scratchpad 里的只读探针（本轮用的两个）
```
按 §5 的 P1 表逐行核；P2 按**逐张量在场/不在场清单**核，不只看总数。

### 8.3 可能**合理**需要重钉的测试

| 文件 | 为什么 |
|---|---|
| `tests/test_to_resolved_adapter.py` | 覆盖度明账（op 数 / skipped / 原因码 / 级联根）会变 —— **这是预期的**，按新实测值重钉 |
| `4c42d63` 引入的 14 条 opdag 规则测试 | 若新增了 advanced-index / 常量 shape 的规则，应**加**新钉，不改旧钉 |
| `tests/test_acceptance_gate.py` | **不该动**。它一旦要改 golden，说明修改泄漏进了 `bucket`/`hand_spec` 共享路径 → 退回 |

### 8.4 不变量的语义检查（比数字更重要）

- **×1 于微批数**：仿真器的结构性结论（重算再执行发生在该区域自己的反向步）。
  抽取图接同一个仿真器 → 自动继承。修完必须仍成立。
- **×1 于层数**：真机事实（stage3 delta 在 L8/L4 都是 24380.1）。
  `extracted` 今天因不给数而 SKIP；**不要**为了让它 PASS 去放宽 `IncompleteExtraction`。

---

## 9. 我没能判定的事（明写，不猜）

1. `csa.py:485` 具体卡在 `_advanced_index` 四道守卫的哪一道（§4.1、§7-8）。
2. r4 的 +8849.2 MiB 由什么组成。拆解文档给的
   「`grad_kv_bm` + `grad_kvo_bm` + `grad_kv_g` ≈ 12288 MiB」在**量级**上说得通，
   但 (a) 数值差 3439，(b) 同一机制在 r0/r128 上会分别要求 +2048/+2560，
   而真机在那两层要求的是 **−3477.2 / −134.1** —— **该假设与 L4 差分不自洽**。
   我给不出替代分解。
3. §4.1 (4) 的「深度过深 vs 每层驻留过大」—— 复核后确认**判不了**，需真机逐微批探针。
4. 真机峰值事件位置（全表最大的干扰项，§7-1）。

---

## 10. 与既有文档的关系

- 修正 `analysis/sim_vs_real_gap_decomposition_2026-07-28.md` §5 第 3 条的归因（§2.1）
  与 §2.2/§4.2「unfused 欠读是一个机制」的框架（§2.3）。
- 修正 `docs/opdag_coverage_close_2026-07-28.md` §6「唯一挡在前面的大石头」（§2.4：三个根，
  与层型一一对应）与 §5 第 5 项对 `constant_shape_unknown`「收益小」的评估（§4.1：
  它是 r0 层的**唯一**级联根，且闸住共享函数体）。
- 承接 `analysis/handoff_2026-07-25_...-norelease-mechanism.md` §7「峰值必须加一项
  ×1 单层重算工作集」：本文给出该项在**多卡全尺寸**下的第一个量化约束
  —— 按层型分别是 r0 **1765.8**、r128 **5904.9**、r4 **27714.7** MiB（unfused−fused，
  L4 单层隔离），而模型现给 5243.0 / 6039.0 / 18865.5。
