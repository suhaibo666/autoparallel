# 对抗性复核：`next_fix_diagnosis_2026-07-28.md` 的判决是否站得住（2026-07-28）

> **性质**：对抗性复核（adversary）。默认立场是**怀疑** —— 除非证据逼我让步，否则假定推荐是错的。
> **源码只读**；探针全在 `scratchpad/adv_probe_*.py`（一次性，不进任何数字路径）。
> **被攻击对象**：`docs/next_fix_diagnosis_2026-07-28.md`（570 行）。
> **基线**：`feat/unified-llm-modelspec` @ `8849f97`。
>
> 记法：**[RAN]** = 我实跑并粘了输出；**[SRC]** = 逐字读源（带 `file:line`）；
> **[INFER]** = 推断，不是观测。铁律：真机数字一律来自 `REAL`（`REAL_SHA256` 指纹保护），
> 不编造、不拟合；分辨不了的写「无法判定」。

## 判决速览

| # | 诊断书的断言 | 我的判决 |
|---|---|---|
| A | `csa.py:485` 卡在 `_advanced_index` 的守卫①（walker 没记 `advanced_index`）或③ | **证伪**：①④ 都过，卡在 **③c**，且在**被索引侧**（`kv_flat` 是 `~`），不在 index 侧 |
| B | 修法 = 「一条 walker 规则」，喂给已存在的 `_advanced_index` | **证伪**：规则已被调用，失败原因是**输入没有轴结构** —— 与 `compressor.py:216` 同一项能力缺口 |
| C | `csa.py:485` 是 r0 层**唯一**的级联根 | **证伪**：r0 层第二道独立闸门 = `deepseek_v4_hybrid_attention.py:205`，独占 42 个跳过 op 中的 **13** 个 |
| D | P1：L1 跳过 42 → **≤5** | **证伪**（反事实实测 **36**）；**诊断书自己的 E1（>22）触发** |
| E | P2：L1 `activation_saves` 1067.6 → **[3000, 5000]** | **证伪**（反事实实测 **1071.6**，+4.0 MiB） |
| F | §2.3「按层型符号相反」是决定性单层隔离数据 | **证伪其决定性**：差分法**结构性只能低估**真机增量 → 「过读」方向**原理上不可由该法建立**；且 r0 的 −3477.2 按诊断书自己的噪声口径只有 **1.45σ** |
| G | 订正 (a)：`a`/`g` 两跑都 fused，拆解文档 §5-3 归因错 | **确认属实** |
| H | 订正 (b)：级联根是三个不是一个 | **确认属实**（unfused 支），但被过度外推成 C（见上） |
| I | §6.2 推论：r4 的 +8849.2 必然住在工作集/梯度侧，不在 saved 集 | **确认，且被我的不等式加强** |

**最终：REFUTE。** 详见 §8。

---

## 0. 基线复现 [RAN]

```bash
cd /e/97-codes/torch_parallel/pynative-cost-evaluator
PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas --gate
```

逐字复现诊断书 §1：

```
  bucket       n=28  mean=0.946  min=0.748  max=1.394
  extracted    n=0  （该来源无任何可评分格：解不出图或全 OOM）
  hand_spec    n=28  mean=0.955  min=0.729  max=1.400
  验收门结论: PASS
  extracted 8/8 ERR: IncompleteExtraction @compressor.py:216
```

delta 表 L4 m4 逐格也复现：

```
L4 m4    0       1765.8     5243.0   ...  2.969
         1      27714.7    18865.5   ...  0.681
         2       5904.9     6039.0   ...  1.023
         3      24380.1    14689.8   ...  0.603      <- 诊断书 §2.3 把这一行删掉了
```

**可复现性缺陷**：诊断书 §2.2 / §3.2 / §3.4 的 [RAN] 证据引用
`scratchpad/probe_layers.py` 与 `scratchpad/probe_stages.py` —— **这两个文件在 HEAD 上不存在**
（`ls scratchpad/` 只有 acceptance / decompose_gap / dump_numbers / dump_numbers_ab /
enum_prims / probe_components / probe_dsv4_bytes / probe_dsv4_walk / probe_head）。
其 [RAN] 数字**不可按文复跑**。下面全部由我独立重新实现。

---

## 1. 攻击点 2 —— 三层型表的算术：**逐格重算，结论：算术对，取样有偏** [RAN]

### 1.1 `g` 与 `h` 确实只差融合位 [RAN][SRC]

```bash
diff analysis/realmachine/ab_fusion_2026-07-25/dsv4h_{fused,unfused}_pp4_recomp.yaml
117c117
<   apply_dsa_kernel_fusion: true
---
>   apply_dsa_kernel_fusion: false
```

**两份 base yaml 只差第 117 行**，`derive_mf_config` 对 g/h 又给同样的
`global_batch_size=8 / n_layers=4 / compress_ratios=[0,4,128,4] / recompute=full`
（[SRC] `tools/liveness_ab_validate.py:121-122`）。**确认。**

### 1.2 L4 的 stage→层型映射 [RAN]

`scratchpad/adv_probe_stages.py` PART 1（直接读 `resolve_graph(...).stages`）：

```
g fused ON L4 m4 / h unfused ON L4 m4  （两跑逐字相同）
   s0: [(0,'embedding'), (1,'dsv4hyb_r0_dense')]
   s1: [(2,'dsv4hyb_r4_moe')]
   s2: [(3,'dsv4hyb_r128_moe')]
   s3: [(4,'dsv4hyb_r4_moe'), (5,'lm_head')]
```

诊断书说的 s0=r0 / s1=r4 / s2=r128 **属实**。

> ⚠ **但 s3 也恰好是 1 个 `dsv4hyb_r4_moe` 层。** 诊断书的三行表把它删了。
> 按诊断书自己的框架，s1 与 s3 隔离的是**同一个量**（一个 r4 层的 unfused 增量），
> 真机却给出 **27714.7 vs 24380.1**（差 **3334.6 MiB / 13.7%**），
> 模型给出 **18865.5 vs 14689.8**（差 **4175.7**）。**这就是下面 §2 的入口。**

### 1.3 逐桶截面：诊断书 §3.2 **属实**，但它是桶模型的**构造性恒等式** [RAN][SRC]

`scratchpad/adv_probe_stages.py` PART 2（桶模型 `Evaluator`，16 个桶逐个打）：

```
s0: g peak 16225.6 @bwd@1 | h peak 21468.6 @bwd@1 | SAME EVENT=True
    唯一不同的桶: remat_saves 3621.2 -> 8864.2   (Δ 5243.0)   其余 15 桶逐字节相同
s1: g 10477.7 @bwd@2 | h 29343.2 @bwd@2 | SAME=True
    remat_saves 3575.8 -> 22441.3 (Δ 18865.5)     其余 15 桶逐字节相同
s2: g 10072.6 @bwd@3 | h 16111.6 @bwd@3 | SAME=True
    remat_saves 3497.3 -> 9536.3  (Δ 6039.0)      其余 15 桶逐字节相同
s3: g 18942.4 @bwd@5 | h 33632.2 @bwd@4 | SAME=False
    act_live / gather_buf / grad_buf / recomp_scratch / bwd_scratch / bwd_working_set /
    remat_saves **七个桶全都不同**
```

**§3.2 的三行我逐字复现，确认。** 但必须把它的地位说清楚 —— 它**不是一条观测发现，
而是桶模型构造的必然**：

[SRC] `cost_eval/mem_timeline.py:709-710` `B.remat_saves = max(0, sm.activation_saves −
sm.checkpoint_input)`，注释 [SRC] `:37`/`:707` 逐字写「**逐层赋在该层自己的 `bwd@lid`
事件上**」，且 [SRC] `:760` `B.remat_saves = 0  # 跑完即清`。
L4 每 stage **恰好 1 个隐藏层** ⇒ **整条时间线上只有一个事件带 remat_saves**。
往一个「本来就是最大值」的事件上加 X，最大值必然抬高 X，且峰值事件不变 ——
这是算术，不是关于真机的证据。s3 就是「那个事件本来不是最大值」的反例，
于是七个桶一起变。

---

## 2. 攻击点 1 —— **E4 是承重假设，而它是关于真机的**：差分法有**单向偏置**

### 2.1 一条不需要真机数据就能证明的不等式

设 `g(e)`、`h(e)` 为两跑在同一事件 `e` 的占用，`δ(e) = h(e) − g(e)`。
`Δ_obs = max_e h(e) − max_e g(e)`。若 `δ(e) ≥ 0`（unfused 逐点不小于 fused），则

```
δ(argmax g)  ≤  Δ_obs  ≤  max_e δ(e)
```

右半边是关键：**观测到的差分是「真实最大单层增量」的下界**，衰减量
`att = max_e δ − Δ_obs ≥ 0` 取决于两跑各自峰值落在哪里。

### 2.2 前提逐点验证：`h ≥ g` 处处成立 [RAN]

`scratchpad/adv_probe_dom.py`，桶模型 + `hand_spec` 两条来源、L4 与 L8、全 4 stage：

```
  #events where h<g = 0        （每一格都是 0，无一例外）
```

**前提成立**，故不等式可用。

### 2.3 模型自己就展示了衰减，而且量级和 r0 的「过读」同阶 [RAN]

`adv_probe_dom.py` 的 `ATTENUATION = max_e δ − Δ_obs`：

| 配置 | stage | 来源 | `Δ_obs` | `max_e δ` | **衰减** | 峰值事件相同? |
|---|---|---|---:|---:|---:|---|
| L4 m4 | s0/s1/s2 | bucket | 5243.0 / 18865.5 / 6039.0 | 同左 | **0.0（0%）** | 是 |
| L4 m4 | **s3** | bucket | 14689.8 | 18865.5 | **4175.7（22.1%）** | 否（g@bwd@5, h@bwd@4） |
| L4 m4 | **s3** | hand_spec | 16346.5 | 23982.3 | **7635.8（31.8%）** | 否 |
| L8 m4 | s0 | bucket | 18211.0 | 18865.5 | 654.5（3.5%） | 否（g@bwd@1, h@bwd@2） |
| L8 m4 | s0 | hand_spec | 23231.8 | 23982.3 | 750.5（3.1%） | 否 |

**模型内部，同一个 r4 层的增量在 s1 是 18865.5、在 s3 被衰减成 14689.8。**
这正是真机 s1(27714.7) 与 s3(24380.1) 差 3334.6 的同一机理。

### 2.4 更细的模型分辨率上，g 与 h 的峰**本来就不在同一时刻** [RAN]

`hand_spec` 的 `peak_substep`（`bucket` 每层反向只有 1 个采样点，`hand_spec` 逐 op）：

| stage | g（fused） | h（unfused） |
|---|---|---|
| s0 | `bwd@1/`**`bwd21:swiglu`** | `bwd@1/`**`bwd11:core_attn`** |
| s1 | `bwd@2/`**`bwd30:shared_fc2`** | `bwd@2/`**`bwd13:sparse_attn`** |
| s2 | `bwd@3/`**`bwd29:shared_fc2`** | `bwd@3/`**`bwd12:sparse_attn`** |
| s3 | `bwd@5/bwd4:nll` | `bwd@4/bwd13:sparse_attn` |

**四个 stage 全都在不同微步达峰。** 真机分配器的分辨率是**算子级**（比 `hand_spec` 还细），
比桶模型的「每层反向一个采样点」细一到两个数量级。
所以「峰值事件相同」这条**只在桶模型那一档粒度上成立**，
而那一档粒度恰恰是它**不可能不成立**的那一档（§1.3）。

### 2.5 结论：**「过读」这个方向，差分法原理上建立不了**

`Δ_obs(model) / Δ_obs(real)` 的分子在 L4 s0/s1/s2 衰减为 0（§2.3），分母的衰减 `att_real ≥ 0`
未知。故

```
观测比值  =  max δ_model / (max δ_real − att_real)   ≥   max δ_model / max δ_real  =  真实比值
```

**观测比值是真实比值的上界。** 于是：

* **欠读（比值 < 1）是稳的** —— 真实欠读只会更严重。r4 的 0.681×：真实 ≤ 0.681。
* **过读（比值 > 1）不成立** —— r0 的 2.97× 只是上界，真实可能是 1.0。
* r128 的 1.023× 同理只是上界 —— 它**可能也在欠读**，不是「准」。

要让 r0 的真实比值落到 1.0，需要 `att_real / max δ_real = 3477.2/5243.0 = 66%`。
模型能展示的最大衰减是 31.8%（§2.3）。**所以衰减不足以完全解释 r0 的差，但足以解释其中很大一部分**
—— 而且 s0 是**warmup 最深**的 stage（4 个微批在途），真机在此处峰值落在别处的概率
恰恰是全表最高的。**我判不了 r0 的真实比值；我能判的是它不是 2.97×，且「符号相反」没有被建立。**

### 2.6 独立第二条：按诊断书**自己的噪声口径**，r0 的误差只有 1.45σ [RAN]

诊断书 §2.3 括注：「`g` 各 stage 的基线残差本身在 ±2300 量级 …
故 r128 的 −134.1 在噪声内、r0 的 −3477.2 与 r4 的 +8849.2 在噪声外」。

`scratchpad/adv_probe_noise.py`：run g 四格残差 `+1871.2 / −1609.8 / −2227.3 / +680.6`
→ **rms = 1696.8**。而 delta 是**两格之差**，独立误差下 `σ_diff = 1696.8·√2 = 2399.6`：

| 层型 | delta 误差 | σ_diff 倍数 | 判定 |
|---|---:|---:|---|
| r0 (s0) | −3477.2 | **1.45σ** | **不显著** |
| r4 (s1) | +8849.2 | 3.69σ | 显著 |
| r128 (s2) | −134.1 | 0.06σ | 不显著 |
| r4 (s3) | +9690.3 | 4.04σ | 显著 |

诊断书拿单格噪声尺度去裁判一个**差分量**，少算了 √2；按一致的标准，
**r0 那格不在噪声外**。两条独立论证（§2.5 单向偏置、§2.6 显著性）指向同一结论。

> **§2.3「决定性新数据」被降级**：真正被建立的只有「r4 严重欠读（≥32%，两 stage 一致）」，
> 这与拆解文档原本的结论方向相同。「三个层型符号相反、所以任何加字节的修法都会弄坏两个」
> —— **未被建立**。而这是 §0 / §3.1 / §6.1 全部论证的地基。

### 2.7 什么能真正判定 E4（无法用现有数据判） — 需要什么真机测量

`REAL` 只有 `peak_alloc_MiB` 一个标量（[SRC] `liveness_ab_validate.py:65-85`），
**没有逐事件曲线**，故 E4 在现有数据上**不可判**。能判它的最小测量：

1. 在 167 的 `g`/`h` 两跑上加 **per-op 内存快照**（MS `memory_allocated()` 逐算子 hook，
   或 `ms.runtime` 的 memory tracker dump），rank0/rank2/rank4 各一份；
2. 只需两条曲线的 **argmax 位置**与 **`h` 在 `g` 峰位置处的值** —— 就能直接算出
   `att_real`，把 §2.5 的不等式收紧成等式；
3. 成本极低（不改配置、不改模型，只加探针），且**同一次跑**顺带产出
   拆解文档 §6 要的「逐微批驻留探针」（能判 (4) 的 9950.6 MiB）。

**[RAN] 我尝试连 167 读既有日志，失败**（严格只读、未启动任何训练）：

```
ping 192.168.9.167   ->  3/3 回复, <1ms
ssh -vv 192.168.9.167 -> debug1: Connection established.
                         kex_exchange_identification: Connection closed by remote host
（连续 6 次重试，全部同一症状；TCP 可连、sshd 在 banner 阶段关闭）
```

故 `/home/suhaibo/workspace/log_ab_fusion_2026-07-25/` 里**是否已存在**逐事件数据，
**我无法确认**（见 §7 未能核对清单）。

---

## 3. 攻击点 3 —— `csa.py:485` 是不是 r0 的唯一闸门？**不是**，而且修法性质被说错了

### 3.1 覆盖度逐字复现 [RAN]

`scratchpad/adv_probe_cov.py`（run b，unfused L8）：

```
TOTALS n_nodes=3976 n_ops=1956 skipped=1010   unresolved_saves=0  unresolved_params=0
  L0:embedding          ops= 12 skipped=  0
  L1:dsv4hyb_r0_dense   ops=162 skipped= 42
  L2:dsv4hyb_r4_moe     ops=280 skipped=173
  L3:dsv4hyb_r128_moe   ops=212 skipped= 92
  ...
  级联根: x4 Elementwise compressor.py:216 needs_axis_structure
          x3 View        compressor.py:233 slice_bounds_unknown
          x1 IndexSelect csa.py:485        constant_shape_unknown
  逐层 extracted activation_saves: r0 1067.6 / r4 650.5 / r128 566.9 / lm_head 2052.0
```

**诊断书 §2.4 的每一个数字我都复现了。** 其对 `opdag_coverage_close` §6
「唯一挡在前面的大石头」的订正（三根，不是一根）**成立**（判决 H）。

fused 支（run a）我另测了一次 —— 它的第三根**不是** `csa.py:485`：

```
a fused: 级联根 x4 compressor.py:216 / x3 compressor.py:233
                x1 View deepseek_v4_hybrid_attention.py:205 needs_axis_structure
```

### 3.2 **卡在哪一道守卫：实测判定**（诊断书 §9-1 明写「我没判定」）[RAN]

`scratchpad/adv_probe_e1.py` PART C：在 `SI._advanced_index` 上挂只读 spy，
打出真实的 `attrs` / `ins` / 已推轴 / `numel_only`：

```
node=47  src=csa.py:485        (compress_ratio=0，即 r0 站点)
  ins        = ['kv_flat__i10:~((64+v_head_dim-64)·B·S):bf16',
                'flat_indices__i10:(128·S):int64']
  in_axes    = [[Factors(coeff=1, syms={'64+v_head_dim-64':1,'B':1,'S':1})],
                [Factors(coeff=128, syms={'S':1})]]
  numel_only = [True, False]
  attrs      = {'advanced_index': True, 'frame': 'self_attention@7/core_attention@20', ...}
  -> GUARD3c FAIL: numel_only=[True, False] ('~' axis structure)
（另有 node=198 ratio=4、node=83 ratio=128 两个站点，**同样只败在 3c**）
```

逐条对照 [SRC] `shape_infer.py:718-729` 的四道守卫：

| 守卫 | 实测 | 诊断书 §4.1 的 [INFER] |
|---|---|---|
| ① `attrs["advanced_index"]` | **`True`，通过** | 猜「可能是 ①」 |
| ② 恰两个张量操作数 | 通过（2/2） | — |
| ③ 两侧轴结构可信 | **失败**，但败在 `numel_only[0]=True`，即**被索引侧 `kv_flat`** | 猜「index 侧无轴 → ③ 失败」 |
| ④ index dtype 是整数 | **`int64`，通过** | — |

**诊断书的机制推断是反的**：index 侧 `flat_indices` 轴结构**完整**（`(128·S)`，`numel_only=False`），
是**数据侧** `kv_flat` 只有元素数。故 §4.1 那条「`mint.arange(b,…)` 的常量 shape 没进 attrs
→ index 无轴」的推断链**与实测不符**，§3.5 表里「修法 = walker 把常量构造的 shape 实参记进 attrs
—— **一条规则**」**落空**：它去修的是一道**没坏的**守卫。

真正坏的地方 [SRC] `mf-src-167/.../csa.py:482`：

```python
kv_flat = mint.reshape(kv_t, (b * sk, d))       # kv_t = permute(kv_full,(1,2,0,3)) @ :472
```

`~((64+v_head_dim-64)·B·S)` 这个串同时暴露两件事：
(i) 4 维 permute 后按 `(b·sk, d)` 的**两轴 reshape 没能保住轴结构**；
(ii) `64+v_head_dim-64` **没被规范化成 `v_head_dim`**。
**这两件事恰好就是 `opdag_coverage_close` §5 第 1 项作者停手的那项「代数层新能力」**，
不是「一条规则」。

### 3.3 **E1 直接实测：r0 层有第二道独立闸门** [RAN]

`scratchpad/adv_probe_e1b.py`（把 42 个跳过 op 逐条按 `(src, op, reason)` 归并）：

```
b unfused / L1:dsv4hyb_r0_dense   ops=162 skipped=42
   在 csa.py 内 : 29 个（:485,:490,:495,:496,:497,:502,:506,:508,:509,:511,:512,:513,:514,
                          :517,:518,:519,:520,:521,:524）
   不在 csa.py  : 13 个
      x2 View  deepseek_v4_hybrid_attention.py:286   no_input_shape
      x1 View  deepseek_v4_hybrid_attention.py:205   shape-unresolved
      x1 Elementwise :206 / x1 View :215 / x1 View :278 / x1 BMM :291 /
      x1 View :294 / x1 View :293 / x1 MatMul :298 / x1 Cast :299 /
      x1 Cast hyper_connection.py:363 / x1 Kernel hyper_connection.py:365

a fused   / L1:dsv4hyb_r0_dense   ops=120 skipped=13
   first_skipped = View deepseek_v4_hybrid_attention.py:205 needs_axis_structure
   跳过清单 = **与上面那 13 条逐条同 file:line、同 op 类型、同计数**
   （唯一差别：`dsv4:205` 的原因码 fused 是 `needs_axis_structure`、unfused 是
     `shape-unresolved` —— 后者是它的输入被已死的 csa 链污染所致，闸门本身是同一个）
```

**这 13 条在反事实里确实活下来了，不是推断** [RAN]：§3.4 修好 `csa.py:485` 后
L1 跳过 42 → 36，恢复的 **6 个全在 `csa.py` 内**（23 + 13 = 36），
`dsv4:205` 那道闸门与它的 12 个连带**原封不动**。

**决定性**：unfused r0 的 42 个跳过 op 里，有 **13 个**与 `csa.py` 完全无关 ——
它们**就是 fused r0 层那 13 个**（fused 路径根本不调 `unfused_compressed_sparse_attn`，
`csa.py:485` 在那支不存在）。所以它们是 `csa.py:485` 的**独立闸门**，不是它的连带。

那道闸门是 [SRC] `mf-src-167/.../deepseek_v4_hybrid_attention.py:205`：

```python
t_nope, t_pe = self.split(t, [nope_dim, pos_dim], dim=-1)     # _apply_forward_rope
```

即 **逆 RoPE 的按维切分**，原因码 `needs_axis_structure` —— 又是同一项能力。

> 诊断书 §3.5 写「r0 层只有这一个级联根」，§4.1 写「另外 40 个被跳过的节点是它的**连带**」。
> 这两句都错。错因是方法论的：[SRC] `to_resolved.py:344` `first_skipped` 定义为
> 「**执行序上第一个**被跳过的节点」，`blockers()`（`:398-411`）**只统计 `first_skipped`**。
> 一个 segment 里的**第二、第三个独立根**，这个 API **结构上看不见**。
> 诊断书直接把 `blockers()` 的 `×1` 读成了「该层只有一个闸门」。

### 3.4 **反事实实测：把 `csa.py:485` 修好（而且比提案更慷慨），会发生什么** [RAN]

`scratchpad/adv_probe_cf.py`：monkeypatch `SI._advanced_index`，在 `csa.py:485` 处
**直接按算子定义交出正确形状** —— `idx.shape ++ [v_head_dim]`，尾轴由
[SRC] `csa.py:482` `reshape(kv_t, (b*sk, d))` 逐字读出。
（这比诊断书提的「记 index 侧常量 shape」**更强**：它连数据侧的轴都白送了。
所以下面的数是该修法**能达到的上界**。源码零改动。）

```
                          BASELINE           COUNTERFACTUAL
  全局 n_ops / skipped     1956 / 1010        2046 / 965
  L1:dsv4hyb_r0_dense      162 / 42           174 / 36
  L2:dsv4hyb_r4_moe        280 / 173          292 / 167
  L3:dsv4hyb_r128_moe      212 / 92           222 / 87
  级联根                    x4 compressor.py:216       x4 compressor.py:216
                           x3 compressor.py:233       x3 compressor.py:233
                           x1 csa.py:485              x1 **BMM csa.py:496 needs_axis_structure**
  extracted saves  r0      1067.6 MiB         **1071.6 MiB**
                   r4       650.5 MiB           782.5 MiB
                   r128     566.9 MiB           566.9 MiB
```

逐条对诊断书 §5 的预言：

| 预言 | 值 | 实测 | 判定 |
|---|---|---|---|
| **P1** L1 跳过 42 → **≤5** | ≤5 | **36** | **证伪**；**诊断书自己的 E1（>22）触发** |
| P1 全局跳过 1010 → 968±8 | 968±8 | 965 | 成立（但阈值本身就宽） |
| P1 `csa.py:485` 那条根消失 | 消失 | 消失，**换成 `csa.py:496`** | 字面成立，实质是换了一块石头 |
| **P2** L1 saves 1067.6 → **[3000,5000]**，中心 3900 | [3000,5000] | **1071.6（+4.0）** | **证伪** |

修完后 L1 saves 里新增的**只有** `flat_indices__i10  4.0 MiB  (128·S)  csa.py:484` 一项。
诊断书 P2 点名「应当出现」的 `q_bm`(512) / `kv_bm`(1024) / `kv_g`·`kvo_bm`(1024) /
`scores`(128) / `exp_scores`(128) / `aw_bm`(128) ≈ 2944 MiB —— **一个都没出现**，
因为链条在 **11 行之后**（`csa.py:496` `scores = reshape(bmm(q_bm, kv_bm), …)`）就断了，
断因是 `:494-495` 那两个 `reshape(permute(...), (b·sq, n, d))` 的**轴合并**解不出 ——
**又是 `needs_axis_structure`**。

> **E3 未触发**：`uout_f32`/`uq_f32`/`kv_gathered` 也确实没进抽取侧 saves。
> 但那不是因为 `bprop_rules.PIN` 的读法被验证了，而是因为**那些 op 根本没被解析**。
> 诊断书对 PIN 表的读法**既没被证伪，也没被检验**。

**⇒ 攻击点 3 的答复：是的，这就是回到规则跑步机。** 前一轮覆盖度 agent 的判断
（「收益变平；剩下的每一条都要新能力，不是一条规则」，`opdag_coverage_close` §5 停手判据）
**被本反事实实测证实**。而且我现在能把它说得更准：

**四个级联根里有三个是同一项能力缺口。**

| 级联根 | 原因码 | 归约到 |
|---|---|---|
| `compressor.py:216` ×4 | `needs_axis_structure` | **符号轴结构恢复**（reshape/permute + 表达式规范化） |
| `deepseek_v4_hybrid_attention.py:205` ×1（fused r0） | `needs_axis_structure` | 同上 |
| `csa.py:485` ×1（unfused r0） | 守卫 3c = 输入 `~`（无轴） | 同上（`64+v_head_dim-64` 未归一 + 两轴 reshape 丢轴） |
| `csa.py:496`（修完 485 后的下一块） | `needs_axis_structure` | 同上 |
| `compressor.py:233` ×3 | `slice_bounds_unknown` | 另一项（切片边界） |

---

## 4. 攻击点 5 —— 两条对输入文档的订正

### 4.1 订正 (a)：`a` 与 `g` 都是 fused —— **确认属实** [SRC]

[SRC] `tools/liveness_ab_validate.py:115` `Variant("a fused   ON  L8 m4", True, 8, 4, True, 8)`
[SRC] `:121` `Variant("g fused   ON  L4 m4", True, 8, 4, True, 4)`
[SRC] `:100-107` `Variant` 的第 2 个字段是 `fused: bool`。**两跑都是 `True`。**

故 `a − g` 隔离的是「同 stage 多一个 **fused** 层」，与 unfused fp32 复本链无关。
`analysis/sim_vs_real_gap_decomposition_2026-07-28.md:137` 第 3 条
「单层成本：欠读 40–50%，**与 (2) 同源（unfused 链的梯度侧）**」**归因确实错**。
**诊断书这条订正成立，应当采纳。**

（补充一条它没说的：由 §2.1 的不等式，`a−g` 同样是 peak-差，也带未知衰减 ——
实测 L8 s0 的模型侧衰减就有 654.5 MiB。所以 (3) 的「欠读 40–50%」本身也是上界。）

### 4.2 订正 (b)：级联根是三个不是一个 —— **确认属实，但被过度外推** [RAN]

三根的实测见 §3.1，**成立**。但诊断书把它外推成「r0 层只有这一个级联根」（§3.5）
与「另外 40 个是它的连带」（§4.1）—— **这两条被 §3.3 证伪**（13/42 属独立闸门）。

---

## 5. 攻击点 4 —— 价值论证：它连自己宣称的**唯一收益**都交付不了

诊断书对「关闭 0 MiB」的辩护（§6.1）是：**「唯一不依赖真机、不引入拟合的测法就是让抽取器读源；
入口被一个节点挡着」**。也就是说它承认不关缺口，但主张换来一次**独立测量**。

**实测：这次测量不会发生。**（§3.4）修完之后 r0 抽取侧 saves = **1071.6 MiB**，
手写普查是 **8500.2 MiB**（[RAN] `adv_probe_census.py`，34 项；诊断书引的 8468.2 是
另一档 `norm_compute_dtype` 口径，差 32.0）。**覆盖 12.6%**，没有任何一项 `_ucopies`
进图。r4 从 650.5 → 782.5，对 22109.3 的手写普查覆盖 **3.5%**。
**零 MiB 缺口 + 零测量 = 纯支出。**

与被它否掉的替代项逐条对比：

| 候选 | 关闭缺口 | 能否**产出判决** | 我的复核 |
|---|---|---|---|
| **(5′) `csa.py:485`（本文被攻击的推荐）** | 0 MiB | **否**（实测 +4.0 MiB saves，下一块石头 11 行之后） | **性价比 = 0** |
| (5) `compressor.py:216` | 0 MiB | 部分（r4 层，`needs_axis_structure` 同族能力） | 与 (5′) **相互独立**，不存在「(5′) 是 (5) 的便宜前置」——我的反事实证明修 485 对 216 **零影响**（×4 原样在列） |
| **(5″) 符号轴结构恢复能力（我的替代推荐）** | 0 MiB | **是**：一次打掉 `compressor.py:216`(×4) + `dsv4:205`(×1) + `csa.py:485/496`(r0/r4/r128 共用函数体) | 见 §6 |
| **(P) 167 逐事件 / 逐微批内存探针** | 直接判 (4)（最大 9950.6）**并**判 E4 | **是**，且是**唯一**能判 E4 的东西 | 见 §2.7；今天 167 SSH 不通 |
| (6.2) 按 `PIN` 手改手写普查 | 可移动格子 | 否（会把 r4/r128 推得更差，见下） | **不反对它的结论** |

关于 §6.2 —— 我**确认并加强**它：按 §2.5 的不等式，r4/r128 的真实比值 ≤ 观测比值
（0.681 / 1.023），故把 census 改小只会更差，**这个方向的结论对衰减偏置免疫**。
连带地，§6.2 的推论「r4 的 +8849.2 **必然住在工作集/梯度侧**、不在 `remat_saves` 侧」
**成立且被加强** —— 这是整份诊断书里最有价值、且经得起攻击的一条。

关于 §3.3（为什么不选 (4)）：我复核了它的反解算术
（单位 3664.58 MiB/(层·微批)，反解 5.42/3.74/2.85 对 8/6/4），
**同意「在 4 个格上判不了、碰它就是发明常数」**。但注意：它需要的那个真机探针
（§2.7 的同一次跑）**顺带就把 E4 判了**。诊断书把它划成「并行可做，不是下一个修」，
可它自己推荐的那一条实测收益是 0 —— **排序应当反过来**。

---

## 6. 我的替代推荐

**不要单点修 `csa.py:485`。改为：给 `sym_shape` / `shape_infer` 加一项
「符号轴结构恢复」能力，验收锚点钉在四个级联根上。**

理由（全部有 §3 的实测支撑）：

1. **它是真正的单一根**：四个级联根里三个（`compressor.py:216` ×4、
   `deepseek_v4_hybrid_attention.py:205` ×1、`csa.py:485` ×1 + 其后继 `csa.py:496`）
   归约到同一项能力（§3.3 末表）。单点修任何一个都只换来个位数 op（实测 6 个）。
2. **它才是「让抽取器读那份清单」的必要条件**：`unfused_compressed_sparse_attn`
   函数体（`csa.py:464-525`）里 **:494/:495/:517/:518** 全是
   `reshape(permute(x, perm), (b·sq, …))` 形态；不解决轴合并，函数体读不通，
   诊断书 §0/§3.1 的整个论证目的达不成。
3. **它兑现诊断书自己的目标**：三种 ratio 共用同一个函数体这条 [SRC] `csa.py:823-825`
   我核实**属实**（且我的守卫探针实测到 **三个 `csa.py:485` 节点，
   `dims_ctx.compress_ratio` 分别是 0 / 4 / 128** —— 这是对该论点最直接的证据，
   比诊断书给的更硬）。读通一次确实三种 ratio 都受益 —— 但前提是**真读通**。
4. **验收可证伪且不需要真机**：`L1 unfused 跳过 42 → ≤13`（13 是 `dsv4:205` 那道闸门的
   独立下界，除非该能力同时打掉它，那就 →0）；`r0 抽取侧 saves ≥ 5000 MiB`；
   `compressor.py:216 ×4` 与 `csa.py:496` 双双从级联根表消失。

**并行、且优先级不低于它的**：§2.7 的 167 逐事件 + 逐微批内存探针。
它是**唯一**能判 E4 的东西，也是**唯一**能判拆解文档 (4)（9950.6 MiB，全表最大单项）的东西，
一次跑同时交付两件。今天 167 SSH 不通（§2.7），这是排期约束，不是价值判断。

**纪律**：诊断书 §5 的 P3（28 格 `bucket` + 28 格 `hand_spec` 逐格 ≤0.05 MiB 不变、
`REAL_SHA256` 不变、`extracted` 仍 8/8 ERR / I1·I2 仍 SKIP）**我完全赞同，原样保留**。
它写得对，且与我的替代推荐兼容。

---

## 7. 我**没能**核对的（明写，不猜）

| # | 诊断书的断言 | 为什么没核对 | 要核对需要什么 |
|---|---|---|---|
| 1 | **E4 本身**：真机 `g s0` 与 `h s0` 是否峰在同一事件 | `REAL` 只有 `peak_alloc_MiB` 标量；167 SSH 六次重试全部 `kex_exchange_identification: Connection closed`（TCP 可连、ping 3/3） | §2.7 的 per-op 内存快照探针；或先修通 167 SSH 后读 `/home/suhaibo/workspace/log_ab_fusion_2026-07-25/` 看是否已有逐事件数据 |
| 2 | §2.2 的「只在 saves / `fml` 看不见」逐张量表（897.0/6144.0/6944.0/19840.0 MiB） | 其探针 `scratchpad/probe_layers.py` 不在库里；我复核的是 `activation_saves` 全量（8500.2 / 22109.3），**没有**逐张量重算「只在 saves」这个子集 | 重写该探针；但 [SRC] `structure_mem.py:163-175` 的机制我已逐字确认：`_forward_max_live` 只走 `op.inputs`/`op.output`，**确实不遍历 `op.saves`** |
| 3 | §3.4 的反事实（`fml⁺saves` → 三格全翻过读、r0 到 5.94×） | 同上，探针不在库里；且这条是**否定性**结论（不该那样修），与本轮判决不冲突 | 重写探针 |
| 4 | §6.2 手改后的逐层数（r0 3072 / r128 3712 / r4 14848） | 是诊断书自己标 [INFER] 的手工套表，不是实测；其**方向性**结论我已在 §5 独立确认 | 需要先有 §6 的能力，让抽取器给出源读答案 |
| 5 | 真机 `g`/`h` 各 stage 的**衰减量** `att_real` | 同 #1 | 同 #1 |
| 6 | 诊断书 §7-2「(2) 里 r4 的 +8849.2 与 (3) 的 +2218…2975/层 是否同源」 | 我也判不了；它诚实标了「我判不了」 | 逐微批驻留探针 |

**顺手记两条 file:line 订正**（诊断书自称「每条断言带已核验定位符」）：

* §0 / §4.2 把 r0 层的手写 saves 清单定位到 `cost_eval/layers/dsv4_hybrid.py:188-229` /
  `:197-206`。那是 **`if sparse:` 稀疏分支**（r4/r128）的 `_ucopies`。
  **r0 走的是 `else:` 滑窗分支**，其 unfused 清单在 [SRC] `dsv4_hybrid.py:246-268`
  （`sparse = compress_ratio not in (0,1)` @ `:73`）。§5 P2 逐张量预言里的
  `uq_f32`/`kv_gathered`/`attn_weights`/`uscore1`/`uout_f32`/`uout_pm` 都在 `:255-263`。
  张量内容对得上，**定位符指错了分支**。
* §4.1 引 `shape_infer.py:707-731 _advanced_index` 与 `:1195-1201` 的落点 ——
  这两处 [SRC] **逐字准确**，我核实无误。

---

## 8. 判决

# REFUTE

**推荐（修 `csa.py:485` 的 advanced-index 定型）不应作为下一个修的项。**

驱动判决的证据（全部 [RAN]，命令与输出见 §0/§1/§2/§3）：

1. **反事实实测直接证伪它自己的两条量化预言**（`scratchpad/adv_probe_cf.py`，源码零改动）：
   L1 跳过 **42 → 36**（预言 ≤5），extracted L1 `activation_saves`
   **1067.6 → 1071.6 MiB**（预言 [3000,5000]）。我的反事实**比提案更慷慨**
   （直接按算子定义交出正确形状，连数据侧尾轴都白送），所以这是该修法的**上界**。
2. **诊断书自己写的 E1 触发**（36 > 22）：「『一个节点闸住整个函数体』**错**：另有独立闸门 …
   整条推荐的性价比崩塌」。独立闸门实测为
   `deepseek_v4_hybrid_attention.py:205`（不是它猜的 `csa.py:457`），
   独占 42 个跳过 op 中的 13 个 —— 证据是 fused r0 层的 13 个跳过 op 与之**逐条同源**。
3. **修法性质被说错**：守卫探针实测 `attrs['advanced_index']=True`、index `int64` 且轴完整，
   失败的是守卫 **3c**、在**被索引侧** `kv_flat`（`~((64+v_head_dim-64)·B·S)`）。
   提案要加的「walker 记 index 侧常量 shape」修的是一道**没坏的守卫**；
   真正要的是符号轴结构恢复 —— 与 `compressor.py:216` **同一项新能力**，
   §3.5 表里「一条规则 vs 新能力」的成本对比**不成立**。
4. **它宣称的唯一收益（独立测量那份 census）不会兑现**：修完 r0 抽取侧覆盖手写普查的
   **12.6%**（1071.6 / 8500.2），r4 覆盖 **3.5%**（782.5 / 22109.3）。0 MiB + 0 测量。
5. **其中心证据 §2.3 的「符号相反」未被建立**：
   (a) 逐点实测 `h ≥ g` 处处成立（0 例外）⇒ `Δ_obs ≤ max_e δ` ⇒ 观测比值是真实比值的**上界**
   ⇒ **「过读」方向原理上不可由差分法建立**，只有「欠读」是稳的；
   (b) 模型自己在 L4 s3 展示 **22.1%（bucket）/ 31.8%（hand_spec）** 的衰减，真机 s1 vs s3
   对同一层型给出 27714.7 vs 24380.1（差 3334.6）；
   (c) 按诊断书自己的噪声口径（run g rms 1696.8，差分 σ=2399.6），r0 的 −3477.2 只有
   **1.45σ，不显著**（r4 是 3.69σ/4.04σ，显著）。
   「三层型符号相反 ⇒ 任何加字节的修法都会弄坏两个」这条 §0/§3.1/§6.1 的地基，**塌了**。

**推荐改为**（§6）：把 `sym_shape`/`shape_infer` 的**符号轴结构恢复**立项
（reshape/permute 轴合并 + 表达式规范化），验收钉在
「`compressor.py:216`×4 与 `csa.py:496` 双双出级联根表；L1 unfused 跳过 ≤13；
r0 抽取侧 saves ≥5000 MiB」；**并行**推 167 的逐事件/逐微批内存探针（§2.7）——
它是唯一能判 E4 的东西，也是唯一能判拆解文档 (4)（9950.6 MiB）的东西。
诊断书 §5 的 P3 纪律条款**原样保留**。

**我确认（不是为了显得公允，是证据支持的）**：
诊断书的 §2.1（`a`/`g` 都 fused，拆解文档 §5-3 归因错）**属实**；
§2.4（三个级联根，订正 `opdag_coverage_close` §6「唯一一块大石头」）**属实**；
§3.2（L4 s0/s1/s2 逐桶只差 `remat_saves`）**逐字复现**；
§3.3（(4) 在 4 个格上判不了）**复核成立**；
§6.2 的推论（r4 的 +8849.2 住在工作集/梯度侧、不在 saved 集）**成立且被我的不等式加强**；
§5 的 P3 纪律**写得对**。
`csa.py:823-825` 三种 ratio 共用同一函数体 —— 我不但确认，还给出了更硬的证据：
抽取器里实测到**三个** `csa.py:485` 节点，`dims_ctx.compress_ratio` 分别是 **0 / 4 / 128**。
问题不在这个论点，在于**读通它需要的是能力、不是规则**。

---

## 附：本文用到的一次性探针（`scratchpad/`，源码零改动）

| 文件 | 干什么 |
|---|---|
| `adv_probe_stages.py` | L4 stage→层型映射；g/h 逐桶截面（16 桶）；事件迁移分解；`max_e δ` vs peak-差 |
| `adv_probe_cov.py` | 逐层覆盖度 + 全部级联根（fused/unfused 两支）；`csa.py:485` 节点原貌 |
| `adv_probe_e1.py` | 四道守卫逐条判定（只读 spy） |
| `adv_probe_e1b.py` | 42 个跳过 op 逐条归并 + fused/unfused 对照（E1 直测） |
| `adv_probe_cf.py` | **反事实**：monkeypatch 让 `csa.py:485` 按算子定义解出，重测 P1/P2 |
| `adv_probe_dom.py` | 逐点 `h ≥ g` 验证 + 衰减量 + `hand_spec` 峰值微步 |
| `adv_probe_noise.py` | 峰值锐度（到第二名的余量）+ 残差噪声尺度与显著性 |
| `adv_probe_census.py` | 手写 census 逐张量复核（r0 8500.2 / 34 项；`_ucopies` 合 5248） |
