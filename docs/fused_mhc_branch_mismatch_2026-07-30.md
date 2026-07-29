# 修配置错配：`use_fused_mhc` 旋钮补齐 —— 锚点不再拿错 mHC 分支对真机（2026-07-30）

> 承接 [`census_fix_residual_carrier_2026-07-29.md`](census_fix_residual_carrier_2026-07-29.md) §5
> ——上一轮**发现并逐条注解**了这条错配（"下一轮第一优先项"），但按当时授权没有修。本轮修它。
>
> **纪律不变**：绝不编造真机数；绝不发明拟合常数；每条改动带 `file:line`；
> 「我跑了并观察到」与「我推断」分开写。`REAL_*` / `CSV_*` 真机常数与 `REAL_SHA256`
> 指纹**一个字节未动**（本文 §7 给出 diff 级证明）。**无 NPU 访问**，一切真机对照均用已存档实测数。

复现：

```bash
PYTHONIOENCODING=utf-8 python scratchpad/probe_mhc_fused_repin.py      # 逐锚点 OFF→ON 读数
PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas
PYTHONIOENCODING=utf-8 python -m pytest tests -q
```

---

## 0. 一句话结论

`serve_explorer.parse_and_validate` 缺一个 `use_fused_mhc` 旋钮（对照 `dsa_fused` / `ce_fused`
都有），于是**所有走扁平 query 的锚点**（pp4×2 / pp8 / MTP / 185 探针 4 组）恒取
`LLMConfig.use_fused_mhc` 默认 `False` = **非融合** mHC，而它们比对的真机跑站点 yaml 是
`use_fused_mhc: true` = **融合** mHC。补上旋钮 + 在这批锚点显式置 1 后，pp4-OFF 由
1.239/1.243/1.111/0.926 落到 **0.892/0.867/0.815/0.832**，与 167 逐层直测（0.89–0.99×）同侧同量级。

**一条与上一轮注解相反的订正**：记分卡 `DSv4 mHC(x4)+MTP`（真机 21153.1，2026-07-01）
**不是**错配 —— 那次真机跑的容器 vendor OPP 里根本没有 `aclnnMhcPreSinkhorn` kernel，
mHC **本来就走非融合**（三处独立源，§2.2）。上一轮把它和 167/185 的 dsv4h 站点 yaml 混为一谈了。
该锚点因此**一个字节不动**，只把注解订正 + 把旋钮显式化。

---

## 1. 缺陷本体（源判）

| 项 | `dsa_fused` | `ce_fused` | `use_fused_mhc`（本轮前） |
|---|---|---|---|
| UI/隐藏字段 | `serve_explorer.py:1598` `<input hidden name="dsa_fused">` | `:1607` `ce_fused` | **无** |
| `_llm_to_fields` 回填 | `:1138` | `:1143` | **无** |
| `_LLM_FIELD_GATE` 门控键 | `:318` | `:323` | **无** |
| `parse_and_validate` 解析 | `:586` `_x_flag(p,"dsa_fused",True)` | `:593-594` | **无** |
| 结果 | yaml/UI 值直达 | yaml/UI 值直达 | 恒 = `LLMConfig` 默认 `False`（`cost_eval/llm_config.py:117`） |

（`serve_explorer.py` 的行号是**本轮改动后**的当前值，便于按图索骥。）

`DimTable.use_fused_mhc`（`cost_eval/model_spec.py:97`）与它驱动的分支
（`cost_eval/layers/residual.py:192` `if getattr(d,"use_fused_mhc",False): _fused_hc_ops else _unfused_hc_ops`）
**早就存在**；缺的只是从 query dict 到 `LLMConfig` 这一段线。yaml 导入侧上一轮已接
（`cost_eval/configs/from_mindformers.py:499`），扁平 query 侧没接。

**两条分支的字节差**（站点 S=4096,B=1,H=4096,n=4，`docs/census_fix_residual_carrier_2026-07-29.md` §1）：

```
融合   /模块:  −96(ln 吃 aggregated) + 128(ctx 的 x)      = +32
非融合 /模块:  −96(同上)             + 576(三份 fp32 副本) = +480
差                                                        = 448 /模块 = 896 /层
```

（三份 fp32 = `hyper_connection.py:297-298` / `:108-109` / `:119-120`；融合分支的 ctx 逐字只有
`custom_op_impl.py:390-391` / `:331`。）

---

## 2. 每个锚点的真机跑到底是哪条分支（逐条源判，不靠推广）

### 2.1 pp4 / pp8 / MTP / 185 探针 —— **融合 mHC**（改）

1. **站点 yaml 逐字**：`analysis/realmachine/ab_fusion_2026-07-25/dsv4h_fused_pp4_recomp.yaml:109`
   与 `dsv4h_unfused_pp4_recomp.yaml:109` **都是** `use_fused_mhc: true`；两份的差别只在
   `apply_dsa_kernel_fusion`（:117 附近），**不是** mHC 融合。
2. **真机数值恒等交叉认亲**（我跑了并观察到 —— 逐位比对已存档常数）：
   - `tests/test_pp4_recompute_anchor.py:34` `REAL_ON = {24153.3, 14641.7, 14097.7, 23508.0}`
     **逐位等于** `tools/liveness_ab_validate.py:75` 的 run `a fused ON L8 m4`
     `{24153.3, 14641.7, 14097.7, 23508.0}`；
   - `tests/test_probe185_recon.py` P3-P 的真机 `25343.5` **逐位等于** run `e fused ON L8 m8` 的 s0。
   run `a`/`e` 就是从上面那两份 yaml 派生的（`liveness_ab_validate.py:62-63`
   "仓内固化的两份 base yaml = 167 上真跑的 launcher 配置"）。
   → pp4/P3-P 锚点比的**就是**八跑门比的那几跑，而八跑门走 yaml 路、已经是融合 mHC。
   **同一个真机跑被两条模型路径用两条不同的 mHC 分支建模** —— 这是本缺陷最硬的自证。
3. `tests/test_pp4_recompute_anchor.py:5` 模块 docstring 自己逐字写着真机是
   "8 层 dsv4_hybrid **FUSED**（DSA kernel + **fused mHC** + fused CE）"。
4. *推断*（非逐位实证）：185 探针 U/F 两相位（seq2048、4L/8L、`apply_dsa_kernel_fusion` 两值）
   与 P3-P 同属 `dsv4h_*` 站点 launcher 家族，故同为 `use_fused_mhc: true`。依据是同家族的
   fused/unfused 两份 yaml **都**写 true（第 1 条）+ P3-P 的逐位认亲（第 2 条）。
   若日后翻出 U/F 相位的原始 yaml 与此不符，本条须回滚。

### 2.2 记分卡 `DSv4 mHC(x4)+MTP`（真机 21153.1）—— **非融合 mHC**（不改，订正注解）

上一轮 `scorecard_anchors.py:140-141` 写「站点 yaml 为 `use_fused_mhc: true` → 配置错配」。
**这条是错的**，三处独立源一致指向非融合：

| 源 | 逐字 |
|---|---|
| `.claude/skills/real-machine-memory-sim/prep_dsv4align.py:120-122`（生成该跑 yaml 的脚本） | 「容器 vendor OPP 无 `aclnnMhcPreSinkhorn` 融合 kernel → mHC 走 unfused」；键值 `"use_fused_mhc": os.environ.get("FUSED_MHC") == "1"` |
| `.claude/skills/real-machine-memory-sim/SKILL.md` §7 第 6 条 | 「融合 mHC kernel（`aclnnMhcPreSinkhorn`）容器 vendor OPP **没有** → mHC 走 unfused。`FUSED_MHC=1` 可强开（需该 kernel）」 |
| `specs/2026-07-01-unified-llm-modelspec-design.md:297` | 锚点标题逐字「mHC + MTP 真机锚点（2026-07-01，fused DSA + **unfused mHC** + MTP）」；:299「**仍待验证**：融合 mHC（…本次 mHC 走 unfused）」 |

`git log -L 118,124:.claude/skills/real-machine-memory-sim/prep_dsv4align.py` 显示这三行是
**与该锚点同一个 commit**（`5200afa realmachine(dsv4): mHC+MTP 真机锚点`）引入的 —— 即该跑用的
就是这份带 `FUSED_MHC` 默认关的脚本。

旁证（本仓自带的保真门）：`tests/test_from_mindformers.py:52` 的 P4 参照 dict 里
`"use_fused_mhc": False`，`:126` 断言 `bundle.llm == dsv4_align_config(4)` —— 这条门本来就把
DSv4-align 的 mHC 分支钉在非融合上，而它一直是绿的。

**结论**：`validate_dsv4align` 该配的就是 `use_fused_mhc=False`。本轮给它加旋钮
（默认 False = 该真机跑的口径、数值一个字节不动），是为了把「隐式默认」变成「显式带出处的选择」。

---

## 3. 落地的改动

| # | 文件:行 | 改动 |
|---|---|---|
| ① | `serve_explorer.py:322-327` | `_LLM_FIELD_GATE` 加 `"use_fused_mhc": "mhc_fused"`（未登记门控键时 `_gate_edited` 会 fail-loud，故这行是 llm_json 基座下能覆盖它的前提） |
| ② | `serve_explorer.py:599-608` | `parse_and_validate`：`if (p.get("mhc_fused") or "").strip() != "": over["use_fused_mhc"] = _x_flag(p,"mhc_fused",base.use_fused_mhc)` —— **与 `ce_fused` 同款语义**（缺省/空 → 保留基座 → 手配路径逐字节不变） |
| ③ | `serve_explorer.py:1139-1141` | `_llm_to_fields` 回填 `"mhc_fused": int(bool(llm.use_fused_mhc))` |
| ④ | `serve_explorer.py:1606` / `:1917` | 页面隐藏输入 `<input type="hidden" name="mhc_fused" value="">` + `RT_DEFAULTS` 复位项（否则 web UI 上这条修复只对脚本调用者生效——与 2026-07-25 `ce_fused` 踩过的坑同款） |
| ⑤ | `validate_dsv4align.py:36,91,100,104,123-130` | `dsv4_align_config(..., use_fused_mhc=False)` / `evaluate(..., use_fused_mhc=False)` / `main()` 读 `FUSED_MHC` env（与 `prep_dsv4align.py:122` 同名同义）+ 打印该位 |
| ⑥ | `scorecard_anchors.py` | `_dsv4_sim` 显式传 `use_fused_mhc=False` 并带出处；订正上一轮的错配注解 |
| ⑦ | `tests/test_pp4_recompute_anchor.py:_BASE` / `tests/test_probe185_recon.py:_dsv4_q` | 加 `"mhc_fused": "1"`（站点 yaml `:109`） |

**为什么不改预设基座**（`PRESETS["dsv4_flash"]` / `deepseek_v4()`）：那会把每个 dsv4_flash 消费者
（含非锚点测试）一起搬走，且「这一跑是不是融合 mHC」是**跑的属性**不是**模型族的属性**
（同一份站点 yaml 家族里 DSA 融合与否就是两跑）。按 `dsa_fused` 的既有形状，逐锚点显式给。

---

## 4. 锚点重钉台账（old → new，逐条理由 + 对真机比值）

真机常数一个字节未动；下表 real 列全部是**已存档实测值**。

### 4.1 `tests/test_pp4_recompute_anchor.py`

| 锚点 | old | **new** | real | ratio old → **new** | 理由 |
|---|---:|---:|---:|---|---|
| `THEO_ON` s0 | 17458.2 | **16143.8** | 24153.3 | 0.723 → **0.668** | 全重算下每层只留 ci，分支差只经 `remat_saves`/`recomp_scratch` 进峰 → −1314.4 |
| `THEO_ON` s1 | 12239.0 | **10924.6** | 14641.7 | 0.836 → **0.746** | 同上 |
| `THEO_ON` s2 | 12448.1 | **11133.6** | 14097.7 | 0.883 → **0.790** | 同上 |
| `THEO_ON` s3 | 21005.9 | **21008.9** | 23508.0 | 0.894 → **0.894** | 峰在 head/loss 段；+3.0 = 融合分支 ctx 的净额（不是解码层 saves） |
| `BAND_OFF` s0 | 实测 1.239 | **0.892** | 30395.0 | band (1.16,1.27) → **(0.85,0.94)** | 无重算全量 saves，−896/层 ×8 层 ×在途 → −10538.7 |
| `BAND_OFF` s1 | 1.243 | **0.867** | 21019.4 | band (1.19,1.29) → **(0.83,0.91)** | 同上 |
| `BAND_OFF` s2 | 1.111 | **0.815** | 17799.9 | band (1.06,1.16) → **(0.78,0.86)** | 同上 |
| `BAND_OFF` s3 | 0.926 | **0.832** | 27822.0 | band (0.88,0.97) → **(0.79,0.88)** | 同上 |
| `THEO_MTP` s0–s2 | 同 `THEO_ON` | 同 `THEO_ON` | — | — | — |
| `THEO_MTP` s3 | 29605.0 | **28291.7** | 39898.0 | 0.742 → **0.709** | 尾 stage 峰在 MTP decoder 层 bwd（是解码层）→ 吃满 −1313.3 |
| `THEO_PP8` s0 | 21800.6 | **20486.2** | 24759.0 | 0.881 → **0.827** | 1 层/stage，−1314.4/stage |
| `THEO_PP8` s1 | 14193.7 | **12879.2** | 12324.0 | 1.152 → **1.045** | 同上 |
| `THEO_PP8` s2 | 13001.1 | **11686.7** | 11678.0 | 1.113 → **1.001** | 同上；**新的"边缘位"**（+8.7 MiB） |
| `THEO_PP8` s3 | 12791.6 | **11477.1** | 11919.0 | 1.073 → **0.963** | 同上；**翻成欠读**，见 §5 |
| `THEO_PP8` s4 | 13809.7 | **12495.2** | 10867.0 | 1.271 → **1.150** | 同上 |
| `THEO_PP8` s5 | 12617.1 | **11302.7** | 11113.0 | 1.135 → **1.017** | 同上 |
| `THEO_PP8` s6 | 12407.6 | **11093.1** | 10074.0 | 1.232 → **1.101** | 同上 |
| `THEO_PP8` s7 | 25272.9 | **25275.9** | 26449.0 | 0.956 → **0.956** | 峰在 head/loss 段，+3.0（同 s3 的 THEO_ON） |
| `_PP8_UNDER` | `{0,7}` | **`{0,3,7}`** | — | — | s3 由过读翻为欠读（0.963）——**如实记账**，不为守住 1.00 而调参 |
| `_PP8_OVER_BAND` | (1.00,1.29) | **(1.00,1.16)** | — | — | 余下过读 stage 实测 1.001/1.045/1.150/1.017/1.101；下界仍刻意守 1.00 |

### 4.2 `tests/test_probe185_recon.py`

| 锚点 | old | **new** | real | ratio old → **new** | 理由 |
|---|---:|---:|---:|---|---|
| U1 峰（4L unfused-DSA seq2048） | 41464.3 | **38834.8** | 40194.0 | 1.032 → **0.966** | −2629.5 = 448/模块 ×2 模块 ×4 层 @seq2048(=站点的一半)；**band (0.95,1.08) 未动**，仍绿 |
| U2 峰（8L unfused-DSA） | 75734.4 | **70475.6** | OOM@56010 | 1.352 → **1.258** | 同上 ×8 层；断言 `>56010`（OOM 翻正）**未动**，仍绿 |
| F0 绝对（4L fused-DSA） | 24987.3 | **22357.8** | 26499.0 | 0.943 → **0.844** | 同上；band (0.83,1.02) **未动**，仍绿 |
| `test_fused_per_layer_increment` 记录值 | 3000.5 | **2343.2** | 差分锚 3109 | 0.965 → **0.754** | (F1−F0)/4；记录门按 §6.6 只移动举例，记录带 (0.80,1.05) → **(0.70,0.90)** |
| `test_p3p_m8_theoretical_and_gap` s0 | 18696.2 | **17381.8** | 25343.5 | 0.738 → **0.686** | 全重算，−1314.4；断言的不变量（理论 < 真机）方向不变 |

`_std_on_q`（185 std MHA/GQA 四点）**逐字节不变**：该配置 `hc: "1"` = 无 mHC
（`test_probe185_recon.py:109`），本轮改动与它无关。

### 4.3 `scorecard_anchors.py` —— **数值一个字节不动**

| 锚点 | old | new | 理由 |
|---|---|---|---|
| `DSv4 mHC(x4)+MTP` | ratio 0.891 / band (0.86,0.93) | **不变** | §2.2：该真机跑本来就是非融合 mHC，模型侧无需翻分支。只把旋钮显式化 + 订正上一轮的错配注解 |
| `DSv4-fused (base)` | 0.902 / (0.87,0.93) | **不变** | `_dsv4_sim(0,0)` → `residual_variant="plain"`，无 mHC |
| 其余 DSv3 族 12 锚 | — | **不变** | 无 mHC |

### 4.4 八跑门（`tests/test_acceptance_gate.py`）—— **逐字节不变**

八跑走 yaml → `from_mindformers` 路，`use_fused_mhc: true` 上一轮已接通，本轮不触及。
`tools/liveness_ab_validate.py --grad-mode chain2 --deltas` 的完整输出在改动前后
**逐行 diff 为空**（§6）。

---

## 5. `pp8 s3`：从 1.0007 的边缘**越过**边缘，落在 0.963

上一轮点名「`pp8 s3` 只剩 1.0007（高出真机 8.6 MiB），错配一修就会退回边缘」。
**实测落点：0.963**（sim 11477.1 vs 真机 11919.0，**低 441.9 MiB**）—— 不是"回到边缘"，
是**越过边缘、翻成欠读**。

- `_PP8_OVER_BAND` 的下界 1.00 是上一轮**刻意**设的「真翻成欠读必须变红」，本轮它如期变红。
  正确处理是**把 s3 移进 `_PP8_UNDER`**（改断言为 `sim < real`，与 s0/s7 同一条不变量），
  **不是**把下界调到 0.96 把红压回绿。已按前者做。
- **新的边缘位是 s2 = 1.001**（sim 11686.7 vs 11678.0，高 **8.7 MiB**）。巧合的是它与上一轮
  s3 的 8.6 MiB 几乎同量 —— 我没有为此做任何调整，如实记，下一轮盯它。
- 我**没有**为 s3 调任何参数。它欠读 441.9 MiB 属 §8 的 OOM-不安全清单。

---

## 6. 八跑验收门 before → after

`PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas`
（before = `git stash` 掉本轮 `serve_explorer.py`/`validate_dsv4align.py` 后跑）：

| 指标 | before | **after** |
|---|---|---|
| bucket 聚合 mean / min / max（n=28） | 0.836 / 0.700 / 1.023 | **0.836 / 0.700 / 1.023** |
| hand_spec·chain2 mean / min / max | 0.855 / 0.670 / 1.110 | **0.855 / 0.670 / 1.110** |
| `unfused − fused` delta @stage3（bucket / hand_spec） | 0.517 / 0.633 | **0.517 / 0.633** |
| 两条真机不变量 I1 / I2 | PASS / PASS | **PASS / PASS** |
| 模型 ×1 不变量（bucket I1/I2、hand_spec I1/I2） | PASS ×4 | **PASS ×4** |
| `REAL_SHA256` 指纹 | `41e279e591ae4ae9…` PASS | **`41e279e591ae4ae9…` PASS** |
| 整份输出 | — | **`diff` 为空（逐行相同）** |

原因：八跑门走的是 yaml → `from_mindformers_dict` 这条路，`use_fused_mhc` 上一轮已在
`from_mindformers.py:499` 接通；本轮补的是**扁平 query** 这条路。两条路现在读同一个值。

---

## 7. `REAL_*` / `CSV_*` / 指纹未动的 diff 级证明

```
$ git diff 4bcbad1 -U0 -- . ':!docs' ':!scratchpad' \
    | grep -E "^[+-]" | grep -v "^[+-][+-]" | grep -E "REAL|CSV|MEASURED|sha256|SHA256"
+#   **恒等认亲**：本文件 `REAL_ON` 与 `tools/liveness_ab_validate.py:75` 的 run `a fused ON L8 m4`
```

唯一命中是一行**新增注释**（提到 `REAL_ON` 这个名字），没有任何 `REAL_*` / `CSV_*` /
`MEASURED` / `REAL_SHA256` **常数行**被增删改。八跑门的指纹校验也在 §6 里 PASS。

---

## 8. 为什么 `_assert_bundle_roundtrip` 没能拦住它（任务书第 2 项）

### 8.1 它**覆盖**了这个字段，却**问错了问题**

`_assert_bundle_roundtrip`（`serve_explorer.py:1175-1196`）用 `_llm_field_diffs`
（`:1151-1154` `dataclasses.asdict` 逐字段比）核 bundle 与 round-trip 的 `LLMConfig` ——
**`use_fused_mhc` 从加进 `LLMConfig` 那天起就在被比的字段集里**。所以答案不是"没覆盖"。

没拦住的原因是那一轮的**修法**：`_llm_to_fields`（`:1147`）把**权威 `LLMConfig` 整体**
JSON 化塞进隐藏字段 `llm_json`，`parse_and_validate`（`:545-552`）见到 `llm_json` 就以它为
基座。于是

> **任何**新增字段都自动随 `llm_json` 过桥 → 该判据对「这个字段在扁平 dict 里有没有承载」
> **结构上永远绿**。

`_llm_to_fields` 的 docstring 自己把这写成了优点（`:1112`「此后**新增 LLMConfig 字段
自动被带上**，不必再逐个加 UI 字段」）—— 对**导入路**确实是优点；代价是它**同时**熄灭了
「这个字段在扁平路上没人管」这个信号。判据不是漏判，是**问错了问题**：
它问「导入的 config 有没有被静默替换」，而这次的病是「**没有导入**的那条路上，字段恒取预设值」。

### 8.2 两条路的不对称（这才是缺陷的容身处）

| | yaml 导入路 | **扁平 query 路**（锚点 / 探针 / 页面手配） |
|---|---|---|
| 结构基座 | `llm_json` 里的权威 `LLMConfig` | `PRESETS[preset]` |
| 没有 UI 键的字段 | 随 `llm_json` 忠实过桥 | **静默取预设/默认值** |
| 有守卫吗 | 有（`_assert_bundle_roundtrip`，运行时） | **本轮之前：没有** |

`dsa_fused` / `ce_fused` 之所以有 UI 键，正是因为**锚点需要它们**（`ce_fused` 那次还额外踩了
「回填了但页面没输入框」的坑，见 `_bundle_to_fields` 上方注释）。`use_fused_mhc` 在
`census_fix_mhc_rmsnorm_2026-07-29` 那轮进 `LLMConfig` 时**只接了 yaml 一条路**
（`from_mindformers.py:499`），而**没有任何机制**逼作者对第二条路做决定 —— 于是默认沉默。

### 8.3 在守卫处闭环：`tests/test_flat_query_reachability.py`（新增 37 例）

新增的不是"再比一次同样的东西"，而是**补上缺的那个问题**：

| 判据 | 守什么 | 会红于 |
|---|---|---|
| ① 分类完备 | 每个 `LLMConfig` 字段**要么**在 `_LLM_FIELD_GATE`（有 UI 键）**要么**在新增的 `_LLM_JSON_ONLY_FIELDS`（显式承认扁平路上恒取预设值） | 新增字段两边都不登记 —— **这正是 2026-07-29 的真实状态** |
| ② 两表互斥 / ③ 无残留字段 | 登记表与 `LLMConfig` 定义不许漂 | 改名/删字段后残留 |
| ④ 接线为真（27 个字段 × 2 取值） | 已登记的 UI 键在**无 `llm_json`** 的扁平 query 上**真的**改得动对应字段 | 「登记了但没接 `parse_and_validate`」 |
| ⑤ 门会响（负例） | monkeypatch 拿掉 `use_fused_mhc` 的登记 → `unclassified_llm_fields()` 必须点名它 | 门本身失效 |
| ⑥ mHC 分支定点回归 | `mhc_fused=1/0` 必须切到 `_fused_hc_ops` / `_unfused_hc_ops` 两条**不同**的 op 链；缺省/空保留基座 | 有人把旋钮摘掉或接错 |

`_LLM_JSON_ONLY_FIELDS`（`serve_explorer.py:344-365`）里现有 **32** 个字段，逐组带了原因。
它不是"豁免清单"，是**已知代价清单** —— §9 就是照着它读出来的第二个错配。

---

## 9. ⚠ 同一 bug class 的**第二例**：`csa_compress_ratios`（本轮**发现但未修**）

> **[2026-07-30 后记]** 已在下一轮修复：[`compress_ratios_mismatch_2026-07-30.md`](compress_ratios_mismatch_2026-07-30.md)。
> 采用了下面的改法 ①（补 UI/隐藏字段 + 门控键）；本节的 what-if 数字
> （pp4-OFF s1 0.922 / s2 0.859、pp8 s3 1.059）**逐位命中**实测。
> 并回答了本文 §8 没能回答的那一问 —— 分类门当时是**绿的**，因为该字段**登记在**
> `_LLM_JSON_ONLY_FIELDS` 里；门只问「有没有分类」，不问「分类是不是真的」。
> 新增 `tests/test_anchor_site_yaml_agreement.py` 补上后一问。

把 §8.3 的 `_LLM_JSON_ONLY_FIELDS` 逐条对着站点 yaml 读，立刻撞上一条**仍在错**的：

| | 值 | 层型分布（L8） |
|---|---|---|
| 锚点扁平路实际解析出的（预设 `dsv4_flash` 的**循环**） | `(0, 4, 128, 0, 4, 128, 0, 4)` | **3×r0 / 3×r4 / 2×r128** |
| 站点 yaml `dsv4h_fused_pp4_recomp.yaml` 的**逐层表** | `[0, 4, 128, 4, 128, 4, 128, 4]` | **1×r0 / 4×r4 / 3×r128** |

（我跑了并观察到：`scratchpad/probe_compress_ratios_whatif.py` 首行逐字打印这两个元组。
旁证：`census_fix_residual_carrier_2026-07-29.md` §2 的真机逐层直测正是「r4 **4 层**均
2341.2 / r128 **3 层**均 2116.1」= 站点表的分布；而 `test_pp4_recompute_anchor.py` 里
2026-07-29 那条注释写「`compress_ratios=[0,4,128,4,128,4,128,4]` → 只有 s1/s3/s5/s7 是 r4 层」
—— 那是**站点**的表，不是模型实际用的那个。）

**what-if 量化**（monkeypatch 覆盖该字段，**不改源**）：

| 锚点 | 今天（预设循环） | what-if（站点逐层表） | Δ |
|---|---|---|---:|
| pp4-ON s1 / s2 | 0.746 / 0.790 | **0.818 / 0.831** | +1048.5 / +583.5 |
| pp4-OFF s1 / s2 | 0.867 / 0.815 | **0.922 / 0.859** | +1158.0 / +786.8 |
| pp8 s3 / s4 / s5 | 0.963 / 1.150 / 1.017 | **1.059 / 1.052 / 1.113** | +1146.1 / −1064.5 / +1064.5 |
| pp4 s0（两口径） | 0.892 / 0.668 | 不变 | 0.0 |

方向上**也是向真机收敛**（pp4 的 s1/s2 各 +0.05~0.07）。

**我没有修它**，理由与上一轮不修 mHC 错配同款：这是又一次**口径变更**，会再次一次性移动
pp4×2 + pp8 + MTP + 185 共 ~20 个锚点，且本轮任务书授权的是 `use_fused_mhc`。
**这是我留给下一轮的第一优先项**；改法两条可选：
① 给 `csa_compress_ratios` 一个 UI/隐藏字段（逐层表用逗号串编码）+ 登记门控键；
② 或者让 pp4/pp8/185 锚点**改走 yaml→bundle 路**（八跑门已经是这么干的），从根上不再有第二份结构。
个人倾向 ②：锚点与八跑门比的本来就是**同一批真机跑**（§2.1 的恒等认亲），却各自维护一份结构。

---

## 10. 验收

```
$ PYTHONIOENCODING=utf-8 python -m pytest tests -q
1940 passed, 268 warnings in 123.43s (0:02:03)
```

`1903 → 1940`：**新增 37 例**（全部在新文件 `tests/test_flat_query_reachability.py`：
完备/互斥/无残留/门会响 4 例 + 接线为真 27 例（逐字段 parametrize）+ 用例自检 2 例 +
mHC 分支定点回归 4 例）。**没有删除任何用例，没有删除任何不变量**；
既有测试只按 `docs/opdag_walker_core_2026-07-25.md` §6.6 精神**移动举例值**
（`test_fused_per_layer_increment` 的记录值与记录带、`test_pp8_framework_gap` 的方向档举例），
每条都在测试文件里就地写了理由。

---

## Related

- [`census_fix_residual_carrier_2026-07-29.md`](census_fix_residual_carrier_2026-07-29.md) §5 —— 本缺陷的发现处与 what-if 预测（本轮逐位对账见 §4.1）
- [`census_fix_mhc_rmsnorm_2026-07-29.md`](census_fix_mhc_rmsnorm_2026-07-29.md) —— `use_fused_mhc` 进 `LLMConfig` 的那一轮（只接了 yaml 一条路）
- [`kernel_workspace_2026-07-29.md`](kernel_workspace_2026-07-29.md) —— 上一轮（+730 实测 workspace），本轮 s0 与 what-if 预测的差额来源
- [`opdag_walker_core_2026-07-25.md`](opdag_walker_core_2026-07-25.md) §6.6 —— 「保留不变量、只移动举例」的改测试规矩

---

## 8. 现在已知为 **OOM-不安全** 的锚点（模型欠读真机）

**不调参掩盖。**

| 锚点 | 本轮前 | **本轮后** | 变化 |
|---|---:|---:|---|
| `pp2-stage1 (loss,k_ce=8)` | 0.999 | 0.999 | 未动（DSv3，无 mHC） |
| `cp2-none (loss,k_ce=4)` | 0.991 | 0.991 | 未动（同上） |
| `DSv3 8L none (dp2)` | 0.981 | 0.981 | 未动（同上） |
| `select self_attn (keep-FFN)` | 0.970 | 0.970 | 未动（同上） |
| `select mlp (keep-attn)` | 0.932 | 0.932 | 未动（同上） |
| `std 116 mha pp2 s0` | 0.934 | 0.934 | 未动（无 mHC） |
| `std 116 gqa pp2 s0` | 0.852 | 0.852 | 未动（同上） |
| `DSv4-fused (base)` | 0.902 | 0.902 | 未动（无 mHC） |
| `DSv4 mHC(x4)+MTP` | 0.891 | **0.891** | **未动**（§2.2：本来就是对的分支） |
| `pp4 OFF` s0 | 1.239（过读） | **0.892** | **翻回欠读** —— 错配修掉后的真读数 |
| `pp4 OFF` s1 | 1.243（过读） | **0.867** | 同上 |
| `pp4 OFF` s2 | 1.111（过读） | **0.815** | 同上 |
| `pp4 OFF` s3 | 0.926 | **0.832** | 一直欠读，更欠 |
| `pp8` s0 | 0.881 | **0.827** | 仍欠读 |
| `pp8` s3 | 1.073（过读） | **0.963** | **新翻成欠读**（§5） |
| `pp8` s7 | 0.956 | **0.956** | 仍欠读（+3.0，head/loss 段） |
| `185 P3-P s0` | 0.738 | **0.686** | 仍欠读 |
| `185 F 每层差分` | 0.965 | **0.754** | 记录门；仍低于 3109 |
| `185 F0 绝对` | 0.943 | **0.844** | 仍欠读 |
| `185 U1 峰` | 1.032（过读） | **0.966** | **新翻成欠读**（band (0.95,1.08) 未动，仍绿） |

**为什么这批集体下移仍是"更对"而不是"更差"**：167/2026-07-29 逐 (微批, 层, 阶段) **直测**
给出融合口径每层驻留 r0 2235.1 / r4 2341.2 / r128 2116.1 MiB，模型逐层型
2229.5 / 2080.6 / 2005.6 → **0.89–0.99×**（`census_fix_residual_carrier_2026-07-29.md` §2）。
本轮后 pp4-OFF 的 0.82–0.89 与之同侧同量级；本轮前的 1.11–1.24 与之**矛盾**。
剩余欠读的归因（kernel workspace 等）见 `kernel_workspace_2026-07-29.md`，本轮一个字节没有往上凑。
