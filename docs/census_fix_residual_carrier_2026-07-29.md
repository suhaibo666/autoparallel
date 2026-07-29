# 普查收口：mHC 残差承载判定归位（2026-07-29，第三轮）

> 承接 [`census_fix_mhc_rmsnorm_2026-07-29.md`](census_fix_mhc_rmsnorm_2026-07-29.md) §3 的
> **②③④** 三条「刻意留下的缺陷」。上一轮的约束是「非融合 mHC 普查必须逐字节不变」，故这三条
> 无法落地；本轮该约束**已解除**，三条一并按源判决落地。
>
> **纪律不变**：绝不编造真机数；绝不为把比值凑到 1.0 而发明拟合常数；每条改动带
> **权威快照**（`E:\97-codes\torch_parallel\mf-src-167\`，含 `mindformers/` 与 `hyper_parallel/`）的
> `file:line` 与实测字节影响；「我跑了并观察到」与「我推断」分开写。**无 NPU 访问**（167 由并行的
> kernel-workspace 任务独占），一切真机对照均用**已存档**的实测数。

复现：
```bash
PYTHONIOENCODING=utf-8 MINDFORMERS_ROOT=E:\97-codes\torch_parallel\mf-src-167\mindformers \
  python scratchpad/census_threeway.py                 # 逐层型逐张量
PYTHONIOENCODING=utf-8 MINDFORMERS_ROOT=... python scratchpad/bucket_gap.py   # run c 逐桶 act_live
PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas
PYTHONIOENCODING=utf-8 python scratchpad/probe_mhc_carrier.py                 # ×n 重命名了哪些张量
PYTHONIOENCODING=utf-8 python scratchpad/probe_pp4_fused_mhc_whatif.py        # §5 的 what-if
python -m pytest tests -q
```

---

## 0. 三条缺陷的源码判决（先判后改）

**②③ 同根**：`residual.py` 的残差承载判定按「形状/切分**签名**」`[S,B,H] + shard={0:"sp"}`
认张量，而源里「哪些张量是打包残差流」是**数据流位置**决定的，不是形状决定的。

`mindformers/pynative/transformers/transformer_layer.py:290-334`
（`HyperConnectionTransformerLayer.construct`）逐字：

```
307:        streams_before_attn = hidden_states                                    # ① 层入口打包流
308:        aggregated_attn, h_res_attn, h_post_attn = self.attn_hc(hidden_states)
310:        input_layernorm_output = self.input_layernorm(aggregated_attn)         #   ln1 吃 aggregated
311:        attention_output = self.self_attention(input_layernorm_output, ...)
319:        dropout_output = self.hidden_states_dropout(attention_output)
320:        hidden_states = self.attn_hc.output_cell(
321:            h_res_attn, h_post_attn, streams_before_attn, dropout_output)      # ② attn 段尾新流
323:        streams_before_ffn = hidden_states
324:        aggregated_ffn, h_res_ffn, h_post_ffn = self.ffn_hc(hidden_states)
326:        pre_mlp_layernorm_output = self.pre_mlp_layernorm(aggregated_ffn)      #   ln2 吃 aggregated
327:        mlp_output = self.mlp(pre_mlp_layernorm_output, input_ids=input_ids)   #   mlp 内部全 [s,b,H]
329:        dropout_output = self.hidden_states_dropout(mlp_output)
330:        output = self.ffn_hc.output_cell(
331:            h_res_ffn, h_post_ffn, streams_before_ffn, dropout_output)         # ③ 层出口新流
```

**一层里的打包流恰好 3 个**：层入口 + 两个 `output_cell` 输出。
`input_layernorm`/`pre_mlp_layernorm` 吃的是 `aggregated_*` `[s,b,H]`；`self.mlp` 内部
（`comb`/`sh_o`/`o`/`o2`）全是 `[s,b,H]`。

### 0.1 ② —— 两个 RMSNorm 的 aggregated 被**别名到打包流名上**（−64 MiB/层）

普查现状（本轮前）：`ln1.saves = [x_xn]`、`ln2.saves = [h1_xn]`，各 `[S,B,n·H]` bf16 = 128 MiB。
真值是两笔**不同**的东西：

| 真值 | 源 | 站点字节（S=4096,B=1,H=4096,n=4） |
|---|---|---:|
| `input_layernorm` 保留的输入 = `aggregated_attn [s,b,H]` bf16 | `transformer_layer.py:310` + `layer_norm.py:151-155`（`FusedRMSNorm` 输入直通） | **32.0** |
| 融合 mHC ctx 首项 `x` = 打包流 `[s,b,n,H]` bf16 | `custom_op_impl.py:390` `ctx.save_for_backward(x, phi, alpha, bias, ...)` | **128.0** |

普查只记了一笔 128 —— 用打包流的名字承担 aggregated 的角色，正是为了**不双计**融合 ctx 的 `x`。
代价：每模块欠 32 MiB，**每层欠 64 MiB**。

### 0.2 ③ —— MoE 的 `comb` 被误当打包流放大（+96 MiB/MoE 层，本轮前模型里唯一的过读项）

`comb` 是 `combine` 的输出（`cost_eval/layers/ffn.py:189,223`，对应源
`pynative/transformers/moe/expert_parallel.py:535`），形状 `[S,B,H]`、标了 `shard={0:"sp"}`
（非 mHC 模型下它**确实**随序列切，这个标注本身没错）—— 恰好撞上 `_is_residual_carrier`
的签名，被 `_scale_ref` 放大成 `[S,B,n·H]` 并改名 `comb_xn`：128 MiB，真值 32 MiB。

源侧它显然不是流：`transformer_layer.py:327` 的 `mlp_output` 要到 `:331` 才被 `output_cell`
打包，`comb` 在 `self.mlp` **内部**、且还要先和 shared 输出相加。
（改前 `ffn.py` 里 `sh_o` 旁的注释「`residual.py _is_residual_carrier` 会将其 ×n 重命名」
正是这条判定过宽的自述 —— 本轮已就地订正。）

### 0.3 ④ —— 非融合分支自己的 fp32 打包副本（本轮前一份没建）

非融合路径上 **bf16 打包流本身没有任何 bprop 持有**（`reshape` 是视图；`Cast` 的 bprop 只是
反向 cast，不需要输入）。真正被持有的是三处 `self.cast(..., mstype.float32)` 各自物化的 fp32 副本：

| 源位置（`mindformers/pynative/transformers/hyper_connection.py`） | 张量 | shape | 谁持有它 | 站点字节 |
|---|---|---|---|---:|
| `:297-298` `x_streams = reshape(hidden_states,...)`; `cast(..., float32)` | `x_streams` | `[S,B,n,H]` fp32 | `:299` `matmul(h_pre, x_streams)` 的 bprop（两操作数都带梯度） | **256.0** |
| `:108-109`（`HyperConnectionOutputCell`）同款 cast | `x_streams` | `[S,B,n,H]` fp32 | `:116` `matmul(h_res_t, x_streams)` 的 bprop | **256.0** |
| `:119-120` `sublayer_exp = cast(reshape(sublayer_out,...), float32)` | `sublayer_exp` | `[S,B,1,H]` fp32 | `:122` `mul(h_post, sublayer_exp)` 的 bprop | **64.0** |

三者是**三次独立的 `ops.cast` 调用**（前两者甚至在不同 `nn.Cell` 里），互不别名 → 三个独立名字。

> **刻意不建的第四份**：`:262-263` `rms_norm(self.cast(hidden_states, mstype.float32), ...)` 的
> 那份 fp32 输入副本（256 MiB）。它是否活到反向取决于 `ops.rms_norm` **grad 节点内部**持有什么，
> 而那**不在权威快照内** —— 与仲裁 §3.2 脚注同一条纪律：读不出来的不猜。（普查已建的
> `hc_norm` 是该 norm 的**输出** `norm_x`，其保留依据是 `:273` mapping_proj 的 bprop，源里读得出。）

融合分支**没有**这三份：`npu_mhc_pre_sinkhorn` / `npu_mhc_post` 的 ctx 逐字只有
`custom_op_impl.py:390-391` / `:331` 那两组，fp32 中间量在 kernel 内部。

---

## 1. 落地的改动与实测字节影响

全部改动在 `cost_eval/layers/residual.py`（+ `cost_eval/layers/ffn.py` 的两处注释订正）。

| # | 改动 | 源依据 | 实测 Δ |
|---|---|---|---:|
| ②a | `ln1`/`ln2` 的 inputs/saves 里的打包流 → 换成 `{prefix}_hc_agg [S,B,H]`（`mhc_wrap._swap_dep`） | `transformer_layer.py:310,326` | −96.0 /模块 |
| ②b | `_fused_hc_ops` 的 `pre_sinkhorn` 显式 `saves += [streams]`；`mhc_wrap` 把 `{prefix}_streams` 换成层内真名（`x_xn`/`h1_xn`）以便按名去重只算一次 | `custom_op_impl.py:390` | +128.0 /模块（融合） |
| ③ | 新增 `_stream_names(body_ops, split)`：打包流 = 层入口 + 两段末 op 输出；`_is_residual_carrier` 降级为形状必要条件 | `transformer_layer.py:290-334` | −96.0 /MoE 层 |
| ④ | `_unfused_hc_ops` 新增三份 fp32 副本 saves；且**不再** save bf16 打包流 | `hyper_connection.py:298` / `:109` / `:120` | +576.0 /模块（非融合） |
| — | `_unfused_hc_ops` op2 的 output 由 `h_proj` 改为 `{prefix}_hc_agg`（两条分支同名同形，`_link` 按下标接边） | `hyper_connection.py:299` | 0（`h_proj` 本就不 saved） |

**逐分支净额**（S=4096,B=1,H=4096,n=4）：

```
融合分支   /模块:  −96(ln 改吃 aggregated) + 128(ctx 的 x) = +32   →  +64 /层
非融合分支 /模块:  −96(同上)               + 576(三份 fp32) = +480  → +960 /层
MoE 层再叠 ③:                                                −96 /层（dense 层无 comb，不叠）
```

---

## 2. 落点：逐层型 `activation_saves` vs 真机逐层直测

`scratchpad/census_threeway.py`（run `c` = fused / 无重算 / L8 / m4 / seq4096，`use_fused_mhc: true`）。
真机 = 167 / 2026-07-29 逐 (微批, 层, 阶段) 轨迹，**存档实测值，本轮未上机**。

| 层型 | 本轮前 | **本轮后** | 真机实测 | 比（前 → 后） | 任务给的预测 |
|---|---:|---:|---:|---|---:|
| r0 `dsv4hyb_r0_dense` | 2165.5 | **2229.5** | 2235.1 | 0.969 → **0.998** | ≈2133（0.954） |
| r4 `dsv4hyb_r4_moe` | 2112.6 | **2080.6** | 2341.2（4 层均） | 0.902 → **0.889** | ≈2081（0.889）✅ |
| r128 `dsv4hyb_r128_moe` | 2037.6 | **2005.6** | 2116.1（3 层均） | 0.963 → **0.948**※ | ≈2006（0.948）✅ |
| lm_head 段 | 3094.0 | **3094.0** | —（真机无逐层可比口径） | 不变 | — |

※ r128 逐层比值随所比的具体层不同（L3/L5/L7 真机 2124.2/2105.0/2119.2），对 3 层均值 2116.1 = 0.948。

**逐条 Δ 与任务预测的对账**：

| 条目 | 任务预测 | **实测** | 判定 |
|---|---:|---:|---|
| ② | +64 MiB/层 | **+64.0** | ✅ 逐 MiB 吻合 |
| ③ | −96 MiB/层 | **−96.0** | ✅ 逐 MiB 吻合，但**只在 MoE 层** |

> ⚠ **与任务预测表的唯一出入：r0 行。** 任务预测 r0 → ≈2133（即对 dense 层也减 96），实测
> 2229.5。原因是 **③ 的 `comb` 只存在于 MoE FFN 段**（`build_moe_ffn_ops` 的 `combine` 输出）；
> dense 层的 FFN 是 `fc1→swiglu→fc2→add2`，**没有 `comb`**，所以只吃 ② 的 +64。
> 探针 `scratchpad/probe_mhc_carrier.py` 逐字给出：r0 层带 `_xn` 后缀的张量只有
> `x_xn/h1_xn/h2_xn` 三个，r4 层多一个 `comb_xn` —— 后者正是 ③ 消掉的那一个。
> 按任务的指示「trust your source reading and report the discrepancy」，此处以源为准。
> 方向上这是**好消息**：r0 由 0.969 收到 **0.998**，是三个层型里最接近真机的一个。

### run `c` 的 `act_live` 逐 stage（`scratchpad/bucket_gap.py`，真机 act_live 为存档实测）

| stage | 真机 `act_live`（深度 × 逐层，实测） | 本轮前 | 比 | **本轮后** | **比** |
|---|---:|---:|---:|---:|---:|
| 0 | 18361.6 | 17112.5 | 0.932× | **17240.5** | **0.939×** |
| 1 | 13434.6 | 12450.7 | 0.927× | **12258.7** | **0.912×** |
| 2 | 8950.0 | 8300.4 | 0.927× | **8172.4** | **0.913×** |
| 3（含 head/loss，不并列） | 4404.7 | 7244.2 | 1.645× | **7180.2** | **1.630×** |

stage0 上升是因为它含 r0（dense）层（+64/层）；stage1/2 全是 MoE 层（−32/层）。
**这正是任务预期的落点**：模型整体略微更欠读，剩余缺口是 kernel workspace，
**本轮一个字节都没有往上凑**。

---

## 3. 八跑门 before → after（`tools/liveness_ab_validate.py --grad-mode chain2 --deltas`）

八跑 yaml **全部** `use_fused_mhc: true`（`analysis/realmachine/ab_fusion_2026-07-25/
dsv4h_{fused,unfused}_pp4_recomp.yaml:109`；两份的差别是 `apply_dsa_kernel_fusion`，
**不是** mHC 融合）→ 本表**只吃 ②③，完全不吃 ④**。

| 指标 | 本轮前 | **本轮后** |
|---|---|---|
| bucket 聚合 mean / min / max（n=28） | 0.822 / 0.701 / 1.019 | **0.821 / 0.700 / 1.017** |
| hand_spec·chain2 mean / min / max | 0.862 / 0.672 / 1.116 | **0.855 / 0.670 / 1.110** |
| hand_spec·dataflow mean / min / max | 0.800 / 0.658 / 1.067 | **0.794 / 0.659 / 1.067** |
| run `c` bucket 逐 stage sim/real | 0.901 / 0.928 / 0.865 / 0.880 | **0.905 / 0.919 / 0.857 / 0.878** |
| run `a`（fused+重算） | 0.713 / 0.815 / 0.828 / 0.945 | 0.715 / 0.813 / 0.826 / 0.945 |
| run `b`（DSA-unfused+重算） | 0.706 / 0.701 / 0.704 / 0.728 | 0.705 / 0.700 / 0.703 / 0.727 |
| `unfused − fused` delta @stage3（bucket / hand_spec） | 0.518 / 0.636 | **0.517 / 0.633** |
| `b − a` @stage1（bucket，真机 29298.3） | 18873.0（0.6442） | **18873.0（0.6442）—— 逐 MiB 不变** |
| 两条真机不变量（I1/I2） | PASS | **PASS** |
| 模型 ×1 不变量（bucket / hand_spec，I1/I2） | PASS | **PASS** |
| `REAL_SHA256` 指纹 | `41e279e591ae4ae9…` | **`41e279e591ae4ae9…`（一个字节没动）** |

`b − a`（DSA-unfused 物化的 ×1 量）**逐 MiB 不变** —— ②③ 在 `a`/`b` 两侧同幅作用、差值抵消。

---

## 4. 锚点重钉台账（old → new，逐条理由）

**未动**：一切 `REAL*` / `CSV*` 真机常数；`REAL_SHA256` 指纹表；两条真机不变量与模型 ×1
不变量；run `d` 不可评分规则；`nr_moe_frag_factor` / `kept_frag_factor` 两个标定 margin
（**一个字节没动**）。**没有删除任何一条不变量。**

### 4.1 八跑门（`tests/test_acceptance_gate.py`）—— 只吃 ②③

| 位置 | old → new | 理由 |
|---|---|---|
| `GOLDEN_BUCKET` / `GOLDEN_LIVENESS_{DATAFLOW,CHAIN2}` | 8×4 表整体移动（如 `c` s1 19503.6→19311.6、`c` s0 27389.6→27517.6） | MoE 层 −32 / dense 层 +64 |
| `GOLDEN_AGG` | bucket 0.822/0.701/1.019 → **0.821/0.700/1.017**；hand_spec·chain2 0.862/0.672/1.116 → **0.855/0.670/1.110** | 同上 |
| `test_delta_magnitude_gap_is_recorded` | 0.518/0.636 → **0.517/0.633** | 两侧同幅，差值几乎不动 |
| `test_unfused_on_cell_ratios_match_recorded_reference` | [0.804,0.813,0.817,0.830] → **[0.802,0.811,0.815,0.828]**；bucket 带 0.701–0.728 → **0.700–0.727** | 同上 |

### 4.2 pp4 / pp8 / MTP / 185 / scorecard —— **走非融合分支**，吃 ②③④（见 §5 的错配说明）

| 位置 | old → new | 理由 |
|---|---|---|
| `test_pp4_recompute_anchor::THEO_ON` | s0 16498.2→**17458.2** / s1 11375.0→**12239.0** / s2 11297.9→**12161.9** / s3 21005.9 **不变** | ④ 打在非融合分支（+960/层）；s3 峰在 head/loss 段。四 stage 仍全部 `sim < real`（不变量方向不动） |
| `...::BAND_OFF` | 实测 0.975/0.996/0.917/0.864 → **1.215/1.243/1.111/0.927**；带 (0.93,1.02)/(0.95,1.05)/(0.87,0.97)/(0.82,0.91) → **(1.16,1.27)/(1.19,1.29)/(1.06,1.16)/(0.88,0.97)** | 同上。s0–s2 由欠读**翻回过读**（OOM 安全侧但**不准**）；s3 仍欠读 |
| `...::THEO_MTP` | s0–s2 同 `THEO_ON`；s3 28011.0→**28875.0** | 同上 |
| `...::THEO_PP8` | 全 stage +960（s0 20840.6→21800.6 … s6 11543.6→12407.6）；s7 25272.9 **不变** | 同上 |
| `...::_PP8_OVER_BAND` | (1.00,1.20) → **(1.00,1.25)** | 实测 s1 1.093 / s2 1.113 / s3 1.073 / s4 1.204 / s5 1.135 / s6 1.232。⚠ 上一轮记的「s3 只剩 1.0007、在翻转边缘」变成 1.073 —— 那是**错配被放大**的结果、不是精度变好；下界仍刻意守 1.00 |
| `test_probe185_recon::test_fused_per_layer_increment` | 2568.5 → **3000.5**（/3109 = 0.826 → **0.965**） | 同一记录门，样例移动；该探针的 `fused=True` 相位真机是融合 mHC，模型走非融合（错配） |
| `...::test_p3p_m8_theoretical_and_gap` | 17102.2 → **17966.2**（/25343.5 = 0.675 → **0.709**，**仍欠读**） | 同上；本门断言的不变量（理论 < 真机）方向不变 |
| `scorecard_anchors` `DSv4 mHC(x4)+MTP` | ratio 0.846 → **0.891**；band (0.81,0.88) → **(0.86,0.93)** | 同上（`validate_dsv4align` 亦无 `use_fused_mhc` 旋钮）。**仍 OOM-不安全**，仍排除历史翻转 1.088 |

### 4.3 一条测试按 `docs/opdag_walker_core_2026-07-25.md` §6.6 精神**移动举例**

`tests/test_mhc.py::test_wrap_scales_residual_hidden_by_n`

| | 原 | 新 |
|---|---|---|
| **不变量** | 层入口残差承载张量被打包成 `[S,B,n·H]`，数值恰为 n× | **同一条，未变** |
| **举例** | 从 `ln1.saves` 里取 `x_xn` | 从 `attn_hc` 的**入流**里取 `x_xn`（源 `transformer_layer.py:307-310`：ln1 吃的是 aggregated，打包流是送进 attn_hc 的那个） |
| **新增** | — | 新用例 `test_prenorm_saves_aggregated_not_packed_stream`：正向钉住 ②（`ln1.saves` 必须含 `attn_hc_agg [S,B,H]`、**不得**含 `x_xn`），有人把它调回去会变红 |

**没有删除任何一条不变量**；新增 1 个用例（`1892 → 1893 passed`）—— 这是唯一的用例数变化，
目的是让 ② 的修复不可被静默回退。

---

## 5. ⚠ 本轮最重要的发现：pp4/pp8/185/scorecard 锚点建的是**非融合** mHC，真机跑的是**融合**

**不是本轮引入的，是本轮才可见。**

- `serve_explorer.parse_and_validate`（pp4/pp8/MTP/185 探针 + 记分卡的公共入口）**没有**
  `use_fused_mhc` 旋钮 —— 对照 `dsa_fused`（`serve_explorer.py:541` `_x_flag(p,"dsa_fused",True)`）
  与 `ce_fused`（`:548-549`）都有 —— 故恒取 `LLMConfig.use_fused_mhc` 默认 **False**。
  `validate_dsv4align.py`（记分卡 DSv4 两条锚）同样没有。实测见探针
  `scratchpad/probe_mhc_branch_of_anchors.py`：
  `pp4 anchor … residual_variant='mhc' n_streams=4 use_fused_mhc=False dsa_fused=True`。
- 而这批锚点比对的真机跑**是融合 mHC**：`tests/test_pp4_recompute_anchor.py` 自己的模块
  docstring 逐字「8 层 dsv4_hybrid **FUSED**（DSA kernel + **fused mHC** + fused CE）」；
  站点 yaml `analysis/realmachine/ab_fusion_2026-07-25/dsv4h_*_pp4_recomp.yaml:109`
  `use_fused_mhc: true`。
- 上一轮（`census_fix_mhc_rmsnorm_2026-07-29.md` Fix 1）加融合门时，只把它接到 **yaml 导入**
  一条路（`from_mindformers.py:499`），没接到 explorer/validate 这条路。当时两条分支的声明字节
  接近，错配不可见；本轮 ④ 只打在非融合分支（+960 MiB/层），错配被放大到无法忽视。

**what-if（探针 `scratchpad/probe_pp4_fused_mhc_whatif.py`，monkeypatch 打开开关，不改源）：**

| 锚点 | 今天（非融合，已重钉） | **what-if（融合）** | 真机 |
|---|---|---|---|
| pp4 OFF s0–s3 | 1.215 / 1.243 / 1.111 / 0.927 | **0.868 / 0.867 / 0.815 / 0.832** | 30395.0 / 21019.4 / 17799.9 / 27822.0 |
| pp4 ON s0–s3 | 0.723 / 0.836 / 0.863 / 0.894 | 0.668 / 0.746 / 0.769 / 0.894 | 24153.3 / 14641.7 / 14097.7 / 23508.0 |
| pp8 s0–s7 | 0.881 / 1.093 / 1.113 / 1.073 / 1.204 / 1.135 / 1.232 / 0.956 | 0.827 / 0.986 / 1.001 / 0.963 / 1.083 / 1.017 / 1.101 / 0.956 | — |

融合口径下 pp4-OFF 落在 **0.82–0.87**，与 167 逐层直测（0.89–0.99×）**同侧同量级**；
非融合口径下的 1.11–1.24 与直测**矛盾**。

**我没有修它**：把 explorer/validate 的 mHC 分支切到融合是一次**口径变更**（与上一轮把
`use_fused_mhc` 从「内存中性忽略集」移进 `_MAPPED_MODEL_KEYS` 同级），会一次性移动
pp4×2 + pp8 + MTP + 185×2 + 记分卡×1 共 ~20 个锚点，量级（≈960 MiB/层）远大于本轮三条修正，
且本轮任务书没有授权。**这是我留给下一轮的第一优先项**，改法只有三行：
给 `serve_explorer` 加一个与 `dsa_fused` 同款的隐藏字段并直通 `LLMConfig.use_fused_mhc`，
`validate_dsv4align` 同理。修它时上表「what-if」列就是预期落点。

---

## 6. 现在已知为 **OOM-不安全** 的锚点（模型欠读真机）

**不调参掩盖。** 唯一正确的补法是基于真机 profiler 的 kernel workspace 项，不是普查充气。

| 锚点 | 本轮前 | **本轮后** | 变化 |
|---|---:|---:|---|
| `pp2-stage1 (loss,k_ce=8)` | 0.999 | 0.999 | 未动（DSv3，无 mHC） |
| `cp2-none (loss,k_ce=4)` | 0.991 | 0.991 | 未动（同上） |
| `DSv3 8L none (dp2)` | 0.981 | 0.981 | 未动（同上） |
| `select self_attn (keep-FFN)` | 0.970 | 0.970 | 未动（同上） |
| `select mlp (keep-attn)` | 0.932 | 0.932 | 未动（同上） |
| `std 116 mha pp2 s0` | 0.934 | 0.934 | 未动（无 mHC） |
| `std 116 gqa pp2 s0` | 0.852 | 0.852 | 未动（同上） |
| `DSv4-fused (base)` | 0.902 | 0.902 | 未动（`_dsv4_sim(0,0)` → `residual_variant="plain"`，**无 mHC**） |
| `DSv4 mHC(x4)+MTP` | 0.846 | **0.891** | ④ 上移；**仍不安全** |
| `pp4 no-recompute OFF` s0/s1/s2 | 0.975 / 0.996 / 0.917 | **1.215 / 1.243 / 1.111** | **翻回 OOM-安全侧**，但见 §5：这是错配放大的结果，不是精度变好 |
| `pp4 no-recompute OFF` s3 | 0.864 | **0.927** | 仍不安全 |
| `pp8` s0 / s7 | 0.842 / 0.956 | **0.881** / 0.956 | 仍不安全（`_PP8_UNDER` 门保持） |
| `185 P3-P s0` | 0.675 | **0.709** | 仍不安全 |
| `185 fused 每层差分` | 0.826 | **0.965** | 记录门；仍低于 3109 |

**本轮没有任何锚点新翻转到 OOM-不安全侧**（②③ 的净额太小、④ 的方向向上）。
上一轮点名的「`pp8 s3` 只剩 1.0007、离翻转 8.6 MiB」现在是 **1.073** —— 但那是 §5 错配放大
的结果；**错配一修就会退回边缘**，届时必须重新盯住。

---

## 7. 我**刻意留下**的欠读（逐条，带量）

| # | 缺口 | 量 | 为什么不补 |
|---|---|---:|---|
| ① | kernel workspace（compressor / indexer / flash / mHC kernel 内部 scratch） | r0 −5.6 / r4 −260.6 / r128 −110.5 MiB/层 | **快照里读不出**。要闭合需真机 profiler 的 kernel workspace 明细。补它 = 发明拟合常数。**这正是并行的另一条任务线在做的事**（`cost_eval/layers/{attention,dsa,dsv4_hybrid}.py` + `mem_timeline/structure_mem`，本轮未碰） |
| ② | 非融合 mHC 的 `rms_norm` **输入** fp32 副本（`hyper_connection.py:262-263`） | 256 MiB/模块（仅非融合） | 是否活到反向取决于 `ops.rms_norm` grad 节点内部持有什么，**不在权威快照内**。与仲裁 §3.2 脚注同一条纪律 |
| ③ | 非融合 mHC 的 `SinkhornKnopp` 循环中间量（`hyper_connection.py:52-68`，`iterations=20` 轮 softmax/div/sum，每轮若干 `[s,b,n,n]` fp32 = 0.25 MiB） | **未量化**（*推断*：与融合 ctx 的 `sum_out`+`norm_out` = 12.5 MiB 同阶，非源读出） | 逐轮到底保留几张要展开 `mint.div`/`mint.sum` 的 bprop 约定，**快照里读不出确定条数** —— 不猜。方向已知（欠读），如实记 |
| ④ | `unfused − fused` delta 的欠读（DSA 侧） | bucket 0.517 / hand_spec 0.633 | 根因是 DSA-unfused 的**反向梯度工作集**（gathered-KV fp32 的三份梯度，拆解报告 §4.2），与本轮改动无关 |
| ⑤ | **mHC 分支配置错配**（§5） | pp4-OFF 的 1.21 vs what-if 的 0.87，≈ 960 MiB/层 | 口径变更，需要一次决策；不在本轮授权内。**下一轮第一优先** |

---

## 8. 下一次真机回归应当验证什么（我不负责跑，但请照这几条核）

新一轮 167 真机回归（fused / no-recompute / L8 / m4 / seq4096，即 run `c` 口径）应当能证伪/证实：

1. **逐层型驻留**：r0 应 ≈ 2235 MiB（模型 2229.5，**0.998×**）；r4 ≈ 2341（模型 2080.6，0.889×）；
   r128 ≈ 2116（模型 2005.6，0.948×）。若 r0 也显著高于模型，说明 dense 层同样有 kernel
   workspace，需要与 r4/r128 的缺口一起归因。
2. **②的可证伪点**：融合 mHC 下，每个 HC 模块应能观察到**两块独立**驻留：一块 `[S,B,n·H]`
   bf16（128 MiB，`npu_mhc_pre_sinkhorn` 的 ctx `x`）+ 一块 `[S,B,H]` bf16（32 MiB，
   `input_layernorm` 保留的 aggregated）。若 profiler 只看到 128 而没有那 32，② 的修法就是错的。
3. **③的可证伪点**：MoE 层的 combine 输出应是 `[S,B,H]` = 32 MiB，**不是** 128 MiB。
   这是本轮唯一的**减法**，也是最容易被真机 per-tensor 明细直接判定的一条。
4. **④只能在 `use_fused_mhc: false` 的跑上验证**：需要一次专门的 unfused-mHC 真机跑
   （站点默认是 true）。若不打算跑，请至少先修 §5 的配置错配，否则那批锚点的读数不代表任何
   真机配置。
5. **剩余缺口**（r4 −260.6 / r128 −110.5 / r0 −5.6 MiB/层）应当**完全**由 kernel workspace
   解释。r0 的 −5.6 已接近噪声，是「注意力主干 + mHC + FFN 的 Python 层张量已基本枚举干净」的
   最强证据；r4 的 −260.6 集中在 indexer/compressor 侧（与仲裁 §1.10 的层型差分一致）。

---

## 9. 验收

```
python -m pytest tests -q   →  1893 passed, 268 warnings
```

1892 → 1893：新增 1 个用例 `test_prenorm_saves_aggregated_not_packed_stream`（正向钉住 ②，
防静默回退）；`test_wrap_scales_residual_hidden_by_n` 按 §6.6 精神**只移动举例位置**、不变量未变。
无删除用例、无删除不变量。

---

## Related

- [`census_arbitration_2026-07-29.md`](census_arbitration_2026-07-29.md) —— 三方仲裁（注意力主干）
- [`census_fix_mhc_rmsnorm_2026-07-29.md`](census_fix_mhc_rmsnorm_2026-07-29.md) —— 上一轮（融合 mHC ctx + RMSNorm 不 cast），本轮 ②③④ 的出处
- [`opdag_walker_core_2026-07-25.md`](opdag_walker_core_2026-07-25.md) §6.6 —— 「保留不变量、只移动举例」的改测试规矩
