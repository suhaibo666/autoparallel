# 修配置错配（第三例）：`csa_compress_ratios` 旋钮补齐 —— 锚点不再拿错**层型分布**对真机（2026-07-30）

> 承接 [`fused_mhc_branch_mismatch_2026-07-30.md`](fused_mhc_branch_mismatch_2026-07-30.md) §9
> ——上一轮**发现并量化**了这条错配（"留给下一轮的第一优先项"），但按当时授权没有修。本轮修它。
>
> **纪律不变**：绝不编造真机数；绝不发明拟合常数；每条结构性断言带 `file:line`；
> 「我跑了并观察到」与「我推断」分开写。`REAL_*` / `CSV_*` 真机常数与 `REAL_SHA256`
> 指纹**一个字节未动**（本文 §7 给出 diff 级证明）。**无 NPU 访问**，一切真机对照均用已存档实测数。

复现：

```bash
PYTHONIOENCODING=utf-8 python scratchpad/probe_compress_ratios_repin.py   # 逐锚点 before→after
PYTHONIOENCODING=utf-8 PYTHONPATH=. python tools/liveness_ab_validate.py --grad-mode chain2 --deltas
PYTHONIOENCODING=utf-8 python -m pytest tests -q
```

---

## 0. 一句话结论

`serve_explorer.parse_and_validate` 缺一个 `compress_ratios` 旋钮（对照 `dsa_fused` / `ce_fused` /
`mhc_fused` 都有），于是**所有走扁平 query 的 dsv4 锚点**恒取预设 `dsv4_flash` 的 0/4/128
**循环近似** `(0,4,128,0,4,128,0,4)`，而它们比对的真机跑站点 yaml 是**逐层表**
`[0,4,128,4,128,4,128,4]`。层型分布 **3×r0/3×r4/2×r128（锚点）vs 1×r0/4×r4/3×r128（真机）** ——
三种层型的驻留是真机直测出来彼此不同的（r0 2235.1 / r4 2341.2 / r128 2116.1 MiB）。
补上旋钮 + 在这批锚点显式给站点表后，pp4-OFF s1/s2 由 0.867/0.815 升到 **0.922/0.859**，
pp4-ON s1/s2 由 0.746/0.790 升到 **0.818/0.831** —— 与上一轮的 what-if 预测**逐位一致**。

**关于守卫（任务书第 3 点）**：分类门没响，**不是因为漏登记，恰恰因为登记了** ——
`csa_compress_ratios` 当时就写在 `_LLM_JSON_ONLY_FIELDS` 里（= 声明「扁平路上恒取预设值，
代价已知可接受」），而那条声明**是假的**。详见 §8。

---

## 1. 缺陷本体（源判）

| 项 | `dsa_fused` | `mhc_fused` | `compress_ratios`（本轮前） |
|---|---|---|---|
| UI/隐藏字段 | `serve_explorer.py:1658` | `:1666` | **无** |
| `_llm_to_fields` 回填 | `:1193` | `:1196` | **无** |
| `_LLM_FIELD_GATE` 门控键 | `:318` | `:327` | **无**（反而在 `_LLM_JSON_ONLY_FIELDS` 里） |
| `parse_and_validate` 解析 | `:631` | `:652-653` | **无** |
| 结果 | yaml/UI 值直达 | yaml/UI 值直达 | 恒 = `PRESETS["dsv4_flash"]` 基座的循环近似 |

（行号是**本轮改动后**的当前值，便于按图索骥。）

消费侧**早就存在**且逐层忠实：`cost_eval/build_llm.py:294` `_compress_ratio` 按
`csa_compress_ratios[layer_idx]` 取每层比，`gen_layer_pattern` 展成 `dsv4hyb_r{ratio}_*` 层 key
（`cost_eval/build_llm.py:307`）；`cost_eval/layers/head.py:250-252` 用 `ratios[-1]` 给 MTP 层。
缺的只是从扁平 query dict 到 `LLMConfig` 这一段线。yaml 导入侧早已接通
（`cost_eval/configs/from_mindformers.py:359,539-541`），扁平 query 侧没接。

**「预设是近似」是预设自己写的**：`serve_explorer.py` `PRESETS["dsv4_flash"]["source"]` 逐字
「compress_ratios 逐层按 0/4/128 循环**近似**」（`PRESETS["dsv4_pro"]` 同）。
近似值不该被真机锚点当口径用 —— 而扁平路没有任何办法拒绝它。

---

## 2. 站点表是对的那一份（逐条源判）

### 2.1 8 层（pp4 / pp8 / MTP / 185 的 8L 相位）

1. **站点 yaml 逐字**：`analysis/realmachine/ab_fusion_2026-07-25/dsv4h_fused_pp4_recomp.yaml:121`
   与 `dsv4h_unfused_pp4_recomp.yaml:121` **都是** `compress_ratios: [0, 4, 128, 4, 128, 4, 128, 4]`
   （上一行注释逐字 `# CSA (length must equal num_hidden_layers = 8)`）。
2. **恒等认亲**（沿用上一轮 §2.1 的证据链，本轮未新增假设）：
   `tests/test_pp4_recompute_anchor.py:34` 的 `REAL_ON` **逐位等于**
   `tools/liveness_ab_validate.py:74` 的 run `a fused ON L8 m4`；八跑门就是从上面那两份 yaml
   派生的（`liveness_ab_validate.py:62-63`）。→ pp4 锚点比的**就是**八跑门比的那几跑，
   而八跑门走 yaml 路、用的正是这张逐层表。
3. **真机逐层直测自证是站点分布**：`docs/census_fix_residual_carrier_2026-07-29.md` §2 的
   167/2026-07-29 逐 (微批, 层, 阶段) 直测，其每层型数值是按
   「r4 = **4 层**的均值 2341.2 / r128 = **3 层**的均值 2116.1」算出来的 —— 4/3 正是**站点表**
   的层型计数（1×r0/4×r4/3×r128），不是预设循环的 3/3/2。**测量本身带着站点的层型分布。**

### 2.2 4 层（185 U1 / F0 相位，seq2048）

`analysis/dsv4_flash_calibration_handoff_2026-07-22.md:121` 逐字记着那次跑的配置：

> **缩小 dsv4_hybrid 配置**（185 跑的那个，改 `apply_dsa_kernel_fusion`/`use_fused_mhc` 切
> fused/unfused）：… 4层 / seq2048 / hidden4096 / heads64 / … / **compress_ratios[0,4,128,4]** /
> 8专家 topk2 moe_inter2048 / first_k_dense1 / pp1 dp2 ep2 / no-recompute / …

同一文档 `:69` 第 4 行给出该配置的真机数 `26499 / 29826` —— 正是
`tests/test_probe185_recon.py:117` 的 F0 锚 `26499`。故 **4L 站点表 = `[0,4,128,4]`**，
是 8L 表的前 4 项（而**不是**预设循环的 `(0,4,128,0)`）。

### 2.3 8L @seq2048（185 U2 / F1 相位）—— **推断**，非逐位实证

U2/F1 是 U1/F0 同一 launcher 家族按层数放大（`_dsv4_q(8, …)`），仓内没有它们自己的 yaml。
按 §2.1 的 8L 站点表 + §2.2 的 4L 记录（后者是前者的前缀）取 `[0,4,128,4,128,4,128,4]`。
**这是推断**：若日后翻出 U/F 8L 相位的原始 yaml 与此不符，§4.2 的 U2/F1 读数须回滚重钉。

---

## 3. 落地的改动

| # | 文件:行 | 改动 |
|---|---|---|
| ① | `serve_explorer.py:328-337` | `_LLM_FIELD_GATE` 加 `"csa_compress_ratios": "compress_ratios"` |
| ② | `serve_explorer.py:350-357` | 从 `_LLM_JSON_ONLY_FIELDS` **移除** `csa_compress_ratios`；把原来那条 ⚠「已知仍在错」的散文注释换成**本表是声明不是证明**的教训 + 指向新守卫 |
| ③ | `serve_explorer.py:407-430` | 新增 `_parse_compress_ratios(raw)`：逗号串 → 元组，只校验「能不能读成整数序列」，档位/长度/整除性仍由 `build_llm._validate_structure` 单一来源 fail-loud |
| ④ | `serve_explorer.py:461-467` | `parse_and_validate` 早段解析（进 `errs` 聚合），空/缺键 → `cr_over=None` |
| ⑤ | `serve_explorer.py:654-663` | `if cr_over is not None: over["csa_compress_ratios"] = cr_over` —— **与 `ce_fused`/`mhc_fused` 同款语义**（缺省/空 → 保留基座 → 手配路径逐字节不变） |
| ⑥ | `serve_explorer.py:1197-1201` | `_llm_to_fields` 回填逗号串（`None` → 空串）；canon 与回填同源 → yaml 导入后未手改判「未改」→ 权威元组原样保留 |
| ⑦ | `serve_explorer.py:1667` / `:1978` | 页面隐藏输入 `<input type="hidden" name="compress_ratios" value="">` + `RT_DEFAULTS` 复位项（否则修复只对脚本调用者生效——2026-07-25 `ce_fused`、2026-07-30 `mhc_fused` 都踩过这个坑） |

**为什么不改预设基座**（`PRESETS["dsv4_flash"]` / `_v4_compress_ratios`）：那张循环表是
**HF config 缩层的通用近似**（`cost_eval/presets.py:71-78` 逐字「缩层预设改混三档循环，
以在小 N 下同时覆盖三条注意力分支」），改它会把每个 dsv4_flash 消费者（含非锚点测试、
`test_dsv4_preset.py` 的三分支覆盖门）一起搬走。而「这一跑的逐层表长什么样」是**跑的属性**，
不是模型族的属性。按 `dsa_fused`/`mhc_fused` 的既有形状，逐锚点显式给。

---

## 4. 锚点重钉台账（old → new，逐条理由 + 对真机比值）

真机常数一个字节未动；下表 real 列全部是**已存档实测值**。
全部读数出自 `scratchpad/probe_compress_ratios_repin.py`（我跑了并观察到），随后由 pytest 复核。

### 4.1 `tests/test_pp4_recompute_anchor.py` —— pp4（2 层/stage，stage s = 层 2s,2s+1）

层型对照：预设循环 `(0,4,128,0,4,128,0,4)` vs 站点表 `(0,4,128,4,128,4,128,4)`
→ s0=(r0,r4) **两表相同**；s1 (r128,r0)→(r128,**r4**)；s2 (r4,r128)→(**r128**,**r4**)；s3 (r0,r4)→(**r128**,r4)。

| 锚点 | old | **new** | real | ratio old → **new** | 理由 |
|---|---:|---:|---:|---|---|
| `THEO_ON` s0 | 16143.8 | **16143.8** | 24153.3 | 0.668 → **0.668** | 两表在 L0/L1 同档 → **逐 MiB 恒等** |
| `THEO_ON` s1 | 10924.6 | **11973.1** | 14641.7 | 0.746 → **0.818** | L3 r0→r4（+1048.5，全重算下只经 `remat_saves`/`recomp_scratch` 进峰） |
| `THEO_ON` s2 | 11133.6 | **11717.1** | 14097.7 | 0.790 → **0.831** | L4,L5 (r4,r128)→(r128,r4)，净 +583.5 |
| `THEO_ON` s3 | 21008.9 | **21049.8** | 23508.0 | 0.894 → **0.895** | 峰在 head/loss 段；+40.9 = L6 由 r0 换 r128 的净额 |
| `BAND_OFF` s0 | 实测 0.892 | **0.892** | 30395.0 | band (0.85,0.94) **未动** | 同上恒等（27112.7 逐 MiB 不变） |
| `BAND_OFF` s1 | 0.867 | **0.922** | 21019.4 | band (0.83,0.91) → **(0.88,0.96)** | 无重算=全量 saves，L3 r0→r4 直接进峰 +1158.0 |
| `BAND_OFF` s2 | 0.815 | **0.859** | 17799.9 | band (0.78,0.86) → **(0.82,0.90)** | 同上 +786.8 |
| `BAND_OFF` s3 | 0.832 | **0.833** | 27822.0 | band (0.79,0.88) **未动** | +32.7，峰在 head/loss 段 |
| `THEO_MTP` s0–s2 | 同 `THEO_ON` | 同 `THEO_ON` | — | — | — |
| `THEO_MTP` s3 | 29789.8 | **29830.6** | 39898.0 | 0.747 → **0.748** | MTP 层比取 `ratios[-1]`（`head.py:250-252`），两表末项**都是 4** → MTP 层不换档；+40.8 与无 MTP 的 s3 同源 |
| `THEO_PP8` s0 | 20486.2 | **20486.2** | 24759.0 | 0.827 → **0.827** | 1 层/stage：层 0 两表都 r0 → 恒等 |
| `THEO_PP8` s1 | 12879.2 | **12879.2** | 12324.0 | 1.045 → **1.045** | 层 1 两表都 r4 → 恒等 |
| `THEO_PP8` s2 | 11686.7 | **11686.7** | 11678.0 | 1.001 → **1.001** | 层 2 两表都 r128 → **恒等**（边缘位，见 §5） |
| `THEO_PP8` s3 | 11477.1 | **12623.2** | 11919.0 | 0.963 → **1.059** | 层 3 r0→**r4**（+1146.1，含该层 bwd 上实测的 730.0 融合稀疏 flash-MLA workspace）；**翻回过读**，见 §5 |
| `THEO_PP8` s4 | 12495.2 | **11430.7** | 10867.0 | 1.150 → **1.052** | 层 4 r4→**r128**（−1064.5，退掉那 730.0 与索引器 saves） |
| `THEO_PP8` s5 | 11302.7 | **12367.2** | 11113.0 | 1.017 → **1.113** | 层 5 r128→**r4**（+1064.5） |
| `THEO_PP8` s6 | 11093.1 | **11174.7** | 10074.0 | 1.101 → **1.109** | 层 6 r0→**r128**（+81.6） |
| `THEO_PP8` s7 | 25275.9 | **25275.9** | 26449.0 | 0.956 → **0.956** | 层 7 两表都 r4 且峰在 head/loss 段 → 恒等 |
| `_PP8_UNDER` | `{0,3,7}` | **`{0,7}`** | — | — | s3 由欠读翻回过读（1.059）——**如实跟随读数**，与上一轮把它移进来同一条纪律 |
| `_PP8_OVER_BAND` | (1.00,1.16) | **(1.00,1.12)** | — | — | 过读上界由 1.150(s4) 变 1.113(s5)；下界仍刻意守 1.00 |

### 4.2 `tests/test_probe185_recon.py`

| 锚点 | old | **new** | real | ratio old → **new** | 理由 |
|---|---:|---:|---:|---|---|
| U1 峰（4L unfused-DSA seq2048） | 38834.8 | **44760.9** | 40194.0 | 0.966 → **1.114** | 4 层站点表末层 r4（循环是 r0）→ unfused 多一个 r4 层；unfused r4 要物化 `index_scores[B,S,64,S]` fp32 + CSA fp32 副本群 ≈ 5.9 GB/层 @seq2048。band (0.95,1.08) → **(1.05,1.18)**，**带宽 0.13 不变、只平移** |
| U2 峰（8L unfused-DSA） | 70475.6 | **76585.3** | OOM@56010 | 1.258 → **1.367** | 同上（+1 个 r4、+1 个 r128、−2 个 r0）；断言 `>56010`（OOM 翻正）**未动**，仍绿 |
| F0 绝对（4L fused-DSA） | 22357.8 | **22492.9** | 26499.0 | 0.844 → **0.849** | fused 分支上三档差很小（稀疏中间量走 kernel scratch）；band (0.83,1.02) **未动**，仍绿 |
| F1（8L fused-DSA） | 31730.6 | **31849.4** | 38936.0 | 0.815 → **0.818** | 同上 |
| `test_fused_per_layer_increment` 记录值 | 2343.2 | **2339.1** | 差分锚 3109 | 0.754 → **0.752** | (F1−F0)/4，−4.1；记录带 (0.70,0.90) **未动** |
| `test_p3p_m8_theoretical_and_gap` s0 | 17381.8 | **17381.8** | 25343.5 | 0.686 → **0.686** | pp4 s0 = (L0,L1)，两表同档 → **逐 MiB 恒等** |

`_std_on_q`（185 std MHA/GQA 四点）**逐字节不变**：`attn` 是 mha/gqa，`csa_compress_ratios`
在这条路上根本不进图（`build_llm.py:307`「`compress_ratio` 仅 `dsv4_hybrid` 取」）。

### 4.3 `scorecard_anchors.py` / DSv4-align 族 —— **数值一个字节不动，且必须不动**

`DSv4 mHC(x4)+MTP`（真机 21153.1）与 `DSv4-fused (base)` 走的是 `dsv4_align_config()`，
不经 `serve_explorer` 扁平 query。而它们那次真机跑的 yaml 是
`.claude/skills/real-machine-memory-sim/prep_dsv4align.py:29` 生成的：

```python
_CYCLE = [0, 4, 128]
compress_ratios = [_CYCLE[i % 3] for i in range(N)] + [0] * MTP
```

—— **那一跑用的就是 0/4/128 循环**。所以「预设循环」对这批锚点是**对的口径**，
本轮一个字节都不该动它们，也确实没动（§7 的 diff 证明）。
> 这条同时说明：错的从来不是「循环」这张表本身，而是**把它套到用逐层表跑的那批真机上**。

### 4.4 八跑门（`tests/test_acceptance_gate.py`）—— **逐字节不变**

八跑走 yaml → `from_mindformers` 路，`compress_ratios` 早已接通
（`cost_eval/configs/from_mindformers.py:359,539-541`），本轮不触及。§6 给出实跑 diff。

---

## 5. 两个边缘位落在哪（任务书点名要看的）

| 位 | before | **after** | 判 |
|---|---:|---:|---|
| **pp8 s2** | **1.001**（高出真机 8.7 MiB） | **1.001，逐 MiB 不动** | pp8 每 stage 1 层，s2 = 层 2，**两张表都是 r128** → 本轮修复对它**恒等**。**没有越界。** |
| **pp8 s3** | 0.963（欠读） | **1.059（过读）** | **越过边缘、翻回过读**。上一轮的 what-if 预测 **1.059**，**逐位命中**。 |

- s3 的处理是**把它移出 `_PP8_UNDER`**（回到过读带断言），与上一轮把它移进去是同一条纪律：
  **如实跟随读数**，不为守住某个方向调参。它在两轮里来回一次，两次都有逐位可复现的机理
  （上一轮 mHC 分支 −1314.4；本轮层型 r0→r4 +1146.1）。
- **过读是 OOM 安全侧，但不代表更准。** s3 现在高出真机 704.2 MiB。
- 新的过读上界是 **s5 = 1.113**（前次是 s4 = 1.150）；带上界随之收到 1.12。
- 我**没有**为任何 stage 调过参。

---

## 6. 八跑验收门 before → after

`PYTHONIOENCODING=utf-8 PYTHONPATH=. python tools/liveness_ab_validate.py --grad-mode chain2 --deltas`

（before = 本轮改动前的 HEAD `43ff9ab`；after = 本轮改动后。两份输出**实跑落盘后 `diff`**，
不是"看着应该一样"就下结论——任务书要求确认而非假设。）

| 指标 | before | **after** |
|---|---|---|
| bucket 聚合 mean / min / max（n=28） | 0.836 / 0.700 / 1.023 | **0.836 / 0.700 / 1.023** |
| hand_spec·chain2 mean / min / max（n=28） | 0.855 / 0.670 / 1.110 | **0.855 / 0.670 / 1.110** |
| 32 格逐格 sim/real（8 跑 × 4 stage） | — | **逐位不变** |
| `unfused − fused` delta @stage3（bucket / hand_spec） | 0.517 / 0.633 | **0.517 / 0.633** |
| 两条真机不变量 I1 / I2 | PASS / PASS | **PASS / PASS** |
| 模型 ×1 不变量（bucket I1/I2、hand_spec I1/I2） | PASS ×4 | **PASS ×4** |
| `REAL_SHA256` 指纹 | `41e279e591ae4ae9…` PASS | **`41e279e591ae4ae9…` PASS** |
| 整份输出 | — | **`diff` 为空（逐行相同）** |

**为什么一格不动**：八跑门走 yaml → `from_mindformers_dict` 这条路，`compress_ratios`
在 `from_mindformers.py:359`（异名映射）+ `:539-541`（转 tuple）早已接通，**它一直用的就是
站点逐层表**。本轮补的是**扁平 query** 这条路。两条路现在读同一张表 —— 这正是本修复的意义：
**同一批真机跑此前被两条模型路径用两张不同的压缩比表建模**（与上一轮 mHC 分支错配同款自证）。

---

## 7. `REAL_*` / `CSV_*` / 指纹未动的 diff 级证明

```
$ git diff 43ff9ab HEAD -U0 -- . ':!docs' ':!scratchpad' \
    | grep -E "^[+-]" | grep -v "^[+-][+-]" | grep -E "REAL|CSV|MEASURED|sha256|SHA256"
+#:   `tests/test_pp4_recompute_anchor.py` 的 `REAL_ON` 逐位等于
```

唯一命中是**一行新增注释**（在新文件 `tests/test_anchor_site_yaml_agreement.py` 里，
复述认亲证据时提到了 `REAL_ON` 这个名字），**没有任何 `REAL_*` / `CSV_*` / `MEASURED` /
`REAL_SHA256` 常数行被增删改**。八跑门的指纹校验也在 §6 里 PASS（`41e279e591ae4ae9…`）。

改动只落在 5 个源文件 + 1 个新测试文件：
`serve_explorer.py` / `tests/test_flat_query_reachability.py` /
`tests/test_pp4_recompute_anchor.py` / `tests/test_probe185_recon.py` /
`tests/test_yaml_roundtrip_fidelity.py` / **新增** `tests/test_anchor_site_yaml_agreement.py`。

---

## 8. 为什么上一轮新装的**分类门**没有拦住它（任务书第 3 点）

### 8.1 它没漏判，是**判据成立而声明为假**

`csa_compress_ratios` **登记在** `_LLM_JSON_ONLY_FIELDS` 里
（本轮之前的 `serve_explorer.py:346`，那一行逐字：
`"window_size", "window_pattern", "csa_compress_ratios", "csa_window_size",`）。
于是 `unclassified_llm_fields()` 返回空、`test_every_llmconfig_field_is_classified` 一直绿。

**这不是漏登记，恰恰是登记了。** 而 `_LLM_JSON_ONLY_FIELDS` 的语义是一句**声明**：

> 「这个字段扁平 UI 表达不了 → 扁平路上恒取预设值 → 这个代价我认了。」

对 `csa_compress_ratios` 而言，这句话的前半段是真的（逐层表确实塞不进一个数字输入框），
**后半段是假的**：pp4/pp8/MTP/185 那批锚点比对的站点 yaml 明明给了另一张表，代价不是"可接受"，
是"锚点在给另一个模型打分"。

> **门只问「有没有分类」，从不问「分类是不是真的」。**
> 一条**没有任何判据背书的自我声明**，被当成了通过条件。

更刺眼的是：上一轮的作者**知道**这条是假的 —— 本轮之前 `serve_explorer.py:340-345` 就写着
一段 ⚠「**已知仍在错**的一条：`csa_compress_ratios` …」的散文注释，紧挨着那张表。
**没有任何测试会读散文。** 于是「代码里写着白纸黑字的缺陷自白」与「守卫全绿」同时成立了两天。

### 8.2 更根本的：那道门是**字段视角**的，看不见锚点

`test_flat_query_reachability.py` 的四条判据全部只认识 `LLMConfig` 的字段集与两张登记表。
它**不知道**「站点 yaml」「锚点」这些东西存在，因此在结构上就不可能回答本 bug class 的那一问：

> **锚点扁平 query 建出来的模型，是不是那次真机跑的模型？**

`use_fused_mhc`（第二例）与 `csa_compress_ratios`（第三例）都不是「字段没接线」，
而是「**接了线但锚点没给值 / 根本没有线可给**」——两者在字段视角下都可以是"已分类"。

### 8.3 在守卫处闭环：`tests/test_anchor_site_yaml_agreement.py`（新增 12 例）

新增的不是"再比一次字段集"，而是把缺的那一问变成机器判据：**归档的真机 launcher yaml
与对着它打分的锚点扁平 query，逐字段比对**。

| 判据 | 守什么 | 会红于 |
|---|---|---|
| ① 逐字段一致 | yaml 权威 `LLMConfig` == 扁平 query 重建的 `LLMConfig`（未登记差异一律红） | **2026-07-29/30 的真实状态**（`use_fused_mhc` 与 `csa_compress_ratios` 都会被点名） |
| ② 登记不发霉 | 登记的差异必须**至少在一对上**真的还在 | 残留豁免（会静默放行日后的真分歧） |
| ③ 对齐后同图 | 按登记表对齐后 `LLMConfig` 全等，且两边 `DimTable`/`layer_pattern` 逐字节相同 | 「口头等价、实际不等价」 |
| ④ **无假声明** | 真差着又没登记的字段，**不许**停在 `_LLM_JSON_ONLY_FIELDS` | 有人把锚点真需要的字段塞进"已知代价清单"蒙混过去 |
| ⑤ 门会响（负例） | monkeypatch 拆掉 `compress_ratios` 旋钮 → ①/④ 必须同时点名该字段 | 门本身失效 |
| ⑥ 推断自洽 | 185 的 4 层表必须是 8 层站点表的前缀（两处出处互不依赖） | 两处出处各自漂 |
| ⑦ 缺陷本体回归 | 预设循环与站点表的**层型计数**必须不同（3/3/2 vs 1/4/3） | 有人"顺手把预设改成站点表"来消除告警（那会搬走每个 dsv4_flash 消费者） |

**关键设计：豁免必须带证据。** 两张登记表都不接受口头承诺：
- `_DECLARED_EQUIVALENCES`（写法不同、语义相同）→ 判据 ③ 当场用**图**证明；
- `_DECLARED_INERT_DIFFS`（取值真不同、影响为 0）→ `test_inert_diffs_are_really_byte_neutral`
  当场把 yaml 的值**灌回锚点再评一遍**，逐 stage 峰值必须逐 MiB 相同。

这正是上一轮缺的那一环：`_LLM_JSON_ONLY_FIELDS` 收的是**声明**，新表收的是**带判据的声明**。

### 8.4 这道门当场抓到的第二件事（**新发现，已量化，未处理**）

把 unfused 那一对纳入比对，立刻多出两条差异：

| 字段 | 站点 yaml | 锚点扁平路 | 出处 |
|---|---:|---:|---|
| `kept_frag_factor` | **1.6** | 0.0 | `from_mindformers.py:515-517` 对「MoE 且非 (dsv4_hybrid ∧ dsa_fused)」注入；`deepseek_v4()` 预设从不设它（`llm_config.py:92` 默认 0.0） |
| `nr_moe_frag_factor` | **0.6** | 0.0 | `from_mindformers.py:526-528`，同上 |

形式上这与本轮的缺陷**同一个 class**（yaml 路注入、扁平路取默认），而且它落在真实锚点上
（185 U 相位 = dsv4_hybrid + MoE + unfused DSA + pp1 + 无重算，正好是这两个 margin 的作用域）。

**我跑了并观察到：它对这批锚点的影响是 0.0 MiB**（把 1.6/0.6 强行灌回锚点，U1 44760.9、
U2 76585.3、F0 22492.9 **逐 MiB 不变**）。原因：两个桶都被融合 CE 挡住
（`cross_entropy_fused=True` → `loss_lids` 空 → `mem_timeline.py:748` / `:760` 的 gate 不成立）。

**为什么不"顺手修掉"**：这两个是**经验标定 margin**，不是结构量。`from_mindformers.py:519-525`
自带风险留档，逐字写着 0.6「**仅在 DSv3(MLA+MoE、topk4、S4096) 两锚点标定过**，注入到别的结构
是**未经真机验证的外推**」；而本评估器自 2026-07-24 起的口径就是**去经验补偿的纯理论**
（`tests/test_pp4_recompute_anchor.py` 模块 docstring 逐字）。把一个 DSv3 标定常数搬到 DSv4
扁平路上，等于凭空给锚点加一笔没有真机背书的量 —— 违反「绝不发明拟合常数」。

处理方式：登记进 `_DECLARED_INERT_DIFFS`，**并让门每次跑都实测一遍那个 0**。
它哪天开始承重，门就红，逼当场做决定。

---

## 9. 现在已知为 **OOM-不安全** 的锚点（模型欠读真机）

**不调参掩盖。** 记分卡（`python sim_vs_real_report.py`）与 116 std 族**本轮逐字节未动**
——它们要么走 `dsv4_align_config()`（不经扁平 query），要么是 mha/gqa（`csa_compress_ratios`
在那条路上根本不进图，`build_llm.py:307`）。我跑了记分卡确认，14 锚一字不差。

| 锚点 | 本轮前 | **本轮后** | 变化 |
|---|---:|---:|---|
| `pp2-stage1 (loss,k_ce=8)` | 0.999 | 0.999 | 未动（DSv3，无 CSA） |
| `cp2-none (loss,k_ce=4)` | 0.991 | 0.991 | 未动（同上） |
| `DSv3 8L none (dp2)` | 0.981 | 0.981 | 未动（同上） |
| `select self_attn (keep-FFN)` | 0.970 | 0.970 | 未动（同上） |
| `select mlp (keep-attn)` | 0.932 | 0.932 | 未动（同上） |
| `std 116 mha pp2 s0` | 0.934 | 0.934 | 未动（mha 不走 CSA） |
| `std 116 gqa pp2 s0` | 0.852 | 0.852 | 未动（同上） |
| `DSv4-fused (base)` | 0.902 | 0.902 | 未动（`dsv4_align_config`，不经扁平 query） |
| `DSv4 mHC(x4)+MTP` | 0.891 | 0.891 | 未动（同上；且那次跑**本来就用循环**，见 §4.3） |
| `pp4 OFF` s0 | 0.892 | **0.892** | 层型同档 → 逐 MiB 恒等 |
| `pp4 OFF` s1 | 0.867 | **0.922** | **收窄**（what-if 预测 0.922，逐位命中） |
| `pp4 OFF` s2 | 0.815 | **0.859** | **收窄**（预测 0.859，逐位命中） |
| `pp4 OFF` s3 | 0.832 | **0.833** | 几乎不动 |
| `pp4 ON` s0–s3（框架缺口档） | 0.668/0.746/0.790/0.894 | **0.668/0.818/0.831/0.895** | s1/s2 **收窄** |
| `pp8` s0 | 0.827 | **0.827** | 层 0 同档 → 恒等 |
| `pp8` s7 | 0.956 | **0.956** | 峰在 head/loss 段 → 恒等 |
| ~~`pp8` s3~~ | 0.963 | **1.059（离开本表）** | 翻回过读（OOM 安全侧），见 §5 |
| `185 P3-P s0`（框架缺口档） | 0.686 | **0.686** | 恒等 |
| `185 F 每层差分` | 0.754 | **0.752** | 记录门；仍低于 3109 |
| `185 F0 绝对` | 0.844 | **0.849** | 微收窄 |
| ~~`185 U1 峰`~~ | 0.966 | **1.114（离开本表）** | 翻成过读；**本轮唯一"离 1.00 更远"的锚点**，见下 |
| `185 std ON MHA/GQA s0`（框架缺口档） | 0.747 / 0.749 | 同左 | 未动（mha/gqa） |

**本轮没有把任何一个锚点推进 OOM-不安全**；两个离开（pp8 s3、185 U1，均翻到 OOM 安全侧）。

### 9.1 唯一"变差"的一条：185 U1（0.966 → 1.114）

如实记：|ratio−1| 由 0.034 涨到 0.114。**但方向是过读 = OOM 安全侧**，且我没有为它调任何参数。
成因是**结构性**的：4 层站点表 `[0,4,128,4]` 的末层是 r4（循环是 r0），而 U 相位是 unfused DSA
—— unfused 的 r4 层要物化 `index_scores [B,S,64,S]` fp32 与 CSA 的 fp32 副本群
（`cost_eval/layers/indexer.py:245` / `csa.py:490,531`），单层就值 ~5.9 GB @seq2048。
**unfused r4 的逐桶归因从来没闭过**：`analysis/dsv4_flash_calibration_handoff_2026-07-22.md:68`
第 3 行逐字「80%（**未闭，缺逐桶归因**）」。本轮把层型对齐，等于把这个一直存在的过读
**从"被一个错的层型分布掩盖着"变成"直接可见"**。要收它，需要 unfused r4 的逐桶 micro-anchor，
不是调层型表。

### 9.2 最大的一笔仍未被处理（**与本轮无关，不由我补偿**）

`lm_head` / loss 段自己的**反向 kernel workspace 从未被测量、留 0**
（`docs/head_workspace_2026-07-30.md` §6 ①）—— 它是上表前 9 行的峰值事件（`bwd@head`，
由 2020 MiB 满 vocab fp32 `bwd_scratch` 主导）。**本轮一个字节都没有往上凑**。
要抬它必须实测 MatMul wgrad / log_softmax / CE 链的 kernel scratch。

---

## 10. 验收

```
$ PYTHONIOENCODING=utf-8 python -m pytest tests -q
1964 passed, 268 warnings in 99.26s (0:01:39)
```

`1949 → 1964`：**新增 15 例**，**没有删除任何用例、没有删除任何不变量**：

| 新增 | 处 | 例数 |
|---|---|---:|
| 扁平 query 接线用例（`csa_compress_ratios`）| `test_flat_query_reachability.py` `_WIRING_CASES` | 1 |
| 逐层压缩比定点回归（落到 r{ratio} 层 key / 缺省保留预设循环 / 非整数 fail-loud） | 同上 | 3 |
| 锚点 ↔ 站点 yaml 一致性门（①②③④⑤⑥⑦，含 2 组配对参数化） | **新文件** `test_anchor_site_yaml_agreement.py` | 11 |

既有测试只按 `docs/opdag_walker_core_2026-07-25.md` §6.6 精神**移动举例值**，每条都在测试文件里
就地写了理由：
- `test_pp4_recompute_anchor.py`：`THEO_ON` / `THEO_MTP` / `THEO_PP8` 三组 pin 值、`BAND_OFF`
  的 s1/s2 两条带、`_PP8_UNDER`、`_PP8_OVER_BAND`；
- `test_probe185_recon.py`：U1 带（**带宽 0.13 不变、只平移**）、每层差分记录值；
- `test_yaml_roundtrip_fidelity.py::test_bundle_to_fields_raises_when_lossy`：判据不变
  （丢 `llm_json` 必须 fail-loud 并点名被替换字段），只把举例字段从 `csa_compress_ratios`
  换成 `o_groups`/`dsa_indexer_topk` —— 因为前者现在**能过桥了**（这正是本轮修复的目的）。

---

## Related

- [`fused_mhc_branch_mismatch_2026-07-30.md`](fused_mhc_branch_mismatch_2026-07-30.md) §9 ——
  本缺陷的发现处与 what-if 预测（本轮逐位对账见 §4.1：pp4-OFF s1/s2、pp8 s3 全部命中）；
  §8 是上一轮装的分类门，本文 §8 说明它为什么绿着放行了这一例
- [`census_fix_residual_carrier_2026-07-29.md`](census_fix_residual_carrier_2026-07-29.md) §2 ——
  167 逐 (微批, 层, 阶段) 直测；「r4 = 4 层均值 / r128 = 3 层均值」是站点层型分布的旁证
- [`head_workspace_2026-07-30.md`](head_workspace_2026-07-30.md) §6 ① / §9 ——
  仍未测的 `lm_head`/loss 反向 workspace，是剩余欠读的最大一笔（本轮未触碰）
- [`opdag_walker_core_2026-07-25.md`](opdag_walker_core_2026-07-25.md) §6.6 ——
  「保留不变量、只移动举例」的改测试规矩
