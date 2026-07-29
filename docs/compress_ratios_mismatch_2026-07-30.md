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

