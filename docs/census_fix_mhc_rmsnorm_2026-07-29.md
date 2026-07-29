# 普查收口：融合 mHC + RMSNorm 不 cast（2026-07-29）

> 承接 [`census_arbitration_2026-07-29.md`](census_arbitration_2026-07-29.md) §3.1 / §3.2 ——
> 上一轮**刻意搁置**的两条「知道在哪、但改动跨口径、需单独决策」的源码级缺陷，本轮决策已下：**落地**。
>
> **纪律不变**：绝不编造真机数；绝不为把比值凑到 1.0 而发明拟合常数；每条改动带
> **权威快照**（`E:\97-codes\torch_parallel\mf-src-167\`，含 `mindformers/` 与 `hyper_parallel/`）的
> `file:line` 与实测字节影响；「我跑了并观察到」与「我推断」分开写。

复现：
```bash
PYTHONIOENCODING=utf-8 MINDFORMERS_ROOT=E:\97-codes\torch_parallel\mf-src-167\mindformers \
  python scratchpad/census_threeway.py            # 逐层型逐张量
PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas
python -m pytest tests -q
```

---

## 0. 两条缺陷的源码判决（先判后改）

### 0.1 站点跑**融合 mHC**，普查按**非融合**建（≈ −421.7 MiB/层）

站点 yaml（`analysis/realmachine/ab_fusion_2026-07-25/dsv4h_{fused,unfused}_pp4_recomp.yaml:109-110`）
逐字 `use_fused_mhc: true` + `hc_sinkhorn_iters: 20` → `FusedHyperConnectionModule`
（`mindformers/pynative/transformers/hyper_connection.py:368`），前向走
`npu_mhc_pre_sinkhorn`（`:413-419`）+ `npu_mhc_post`（`:363`）。

`npu_mhc_pre_sinkhorn` 的 ctx 逐字（`hyper_parallel/platform/mindspore/custom_ops/custom_op_impl.py:390-391`）：

```
390:        ctx.save_for_backward(x, phi, alpha, bias,
391:                              h_pre, hc_before_norm, inv_rms, sum_out, norm_out)
```

形状逐字在 `hyper_parallel/platform/mindspore/custom_ops/mhc_pre_sinkhorn.cc:24-50`
（`bs=S=4096, seq_len=B=1, n=4, c=H=4096, fusion_size=n²+2n=24, num_iters=20`）：

| ctx 张量 | `.cc` 行 | shape | dtype | MiB |
|---|---|---|---|---:|
| `x`（打包残差流） | `:30`(入参) | `[4096,1,4,4096]` | bf16 | 128.0 |
| `h_pre` | `:36,40` | `[4096,1,4]` | fp32 | 0.0625 |
| `hc_before_norm` | `:37,41` | `[4096,1,24]` | fp32 | 0.375 |
| `inv_rms` | `:38,42` | `[4096,1,1]` | fp32 | 0.0156 |
| `sum_out` | `:38,43` | `[40,4096,1,4]` | fp32 | 2.5 |
| `norm_out` | `:39-40,44` | `[40,4096,1,4,4]` | fp32 | 10.0 |
| `h_res`（kernel 输出，被下游 `output_cell` 用） | `:33` | `[4096,1,16]` | **fp32** | 0.25 |
| `h_post`（同上） | `:32` | `[4096,1,4]` | **fp32** | 0.0625 |

后 5 项（`h_pre`…`norm_out`）在调用点被 `h_in, h_post, h_res_flat, *_ = npu_mhc_pre_sinkhorn(...)`
（`hyper_connection.py:413`）的 `*_` **丢掉 Python 名字，但仍被 ctx 强引用** ——
`mhc_pre_sinkhorn.cc:61` `ms::TensorAllocate({h_in, h_post, h_res, h_pre, hc_before_norm, inv_rms,
sum_out, norm_out})` **无条件分配全部 8 个输出**，`custom_op_impl.py:390` 又逐字 save 其中 5 个。
**没有 Python 名 ≠ 没有设备内存。**

`npu_mhc_post` 另存（`custom_op_impl.py:331`）`ctx.save_for_backward(x, h_res, h_out, h_post)`，
其中 `h_out` = sublayer 输出 `[S,B,H]` bf16 = **32.0 MiB**，普查**完全没建**。

对照普查现状 `cost_eval/layers/residual.py:110-126`：每个 HC 模块建了一张
`{prefix}_hc_norm [S,B,n·H] fp32` = **256.0 MiB** + `h_res` bf16 0.125 = 256.125 MiB/模块。
该 256 MiB 只在**非融合** `HyperConnectionModule.construct` 里真实存在
（`hyper_connection.py:262-266` `rms_norm(cast(hidden_states, float32), ...)`，其输出被 `:273`
`mapping_proj` 的 bprop 保留）——**非融合分支必须保留**，故本条按**配置门控**落地，不硬切。

### 0.2 `FusedRMSNorm` **不做** fp32 cast（−268.0 MiB/层 + lm_head −32.0）

`mindformers/pynative/layers/layer_norm.py:151-155` 逐字：

```
151:    def construct(self, x):
152:        """Apply fused RMS Normalization (on the local, sequence-sharded activation)."""
153:        # Norm weight is a Replicate DTensor; run the kernel on the local copy.
154:        output = self.norm(x, _to_local(self.weight), self.eps)[0]
155:        return output
```

`x` **直通** `ops.rms_norm`（`self.norm = rms_norm`，`:148`），**没有任何 cast**；
`self.cast = cast`（`:149`）是**死属性**（全类只此一次出现，`construct` 从不调用）。

对照 `FusedLayerNorm.construct`（`:93-101`）**才有**真 cast：

```
 95:        original_type = x.dtype
 96:        compute_type = self.compute_type
 97:        x = self.cast(x, compute_type)
...
101:        return self.cast(output, original_type)
```

`get_norm_cls`（`:187-191`）按 `normalization` 返回二者之一。站点 `normalization: RMSNorm` →
**全部** `input_layernorm` / `pre_mlp_layernorm` / `q_layernorm` / `kv_layernorm` / `final_layernorm`
（`transformer_layer.py:126,159,311,329`、`deepseek_v4_hybrid_attention.py:102-106,129-133,238,250`）
都是 `FusedRMSNorm` → `layernorm_compute_dtype: float32` 在这里**只决定 gamma 参数 dtype**
（`layer_norm.py:146` `Parameter(mint.empty(dim, dtype=self.compute_type))`），
**不产生任何 fp32 激活副本**。

而评估器 `cost_eval/structure_mem.py:118-139` 的 `_norm_save_names` + `_dt` 把**所有** `OpType.NORM`
op 的 saves 无差别按 `norm_compute_dtype_bytes=4` 抬升。**这对 RMSNorm 是错的、对 LayerNorm 是对的**
→ 必须**按 norm 种类**分辨，不能整体翻 `norm_compute_dtype_bytes`（那是跨模型口径，会连
「真的 cast 的 norm」一起关掉）。

---

## 1. 落地的改动与实测字节影响

### 1.1 Fix 1 —— 融合门（**配置驱动，两条分支都保留**，照 `dsa_fused` 的范式）

新增两个 `DimTable` 字段（**加法式、带默认 → 无关模型逐字节不变**）：

| 字段 | 默认 | 来源 | 作用 |
|---|---|---|---|
| `use_fused_mhc` | `False` | yaml `use_fused_mhc`（站点 `:109` = true） | 选 `_fused_hc_ops` / `_unfused_hc_ops` |
| `mhc_sinkhorn_iterations` | `20` | yaml `hc_sinkhorn_iters`（站点 `:110` = 20）；`transformer_config.py:2084` 默认亦 20 | `sum_out`/`norm_out` 的首维 `2·num_iters` |

`cost_eval/configs/from_mindformers.py`：二者（含同义名 `mhc_sinkhorn_iterations`）由
「内存中性忽略集」**移入** `_MAPPED_MODEL_KEYS` —— 此前判它们中性是错的。

**两条分支实测**（`scratchpad/probe_mhc_branch.py`，B=1/S=4096/H=4096/n=4/iters=20）：

```
use_fused_mhc=False (3 ops)                        use_fused_mhc=True (3 ops)
  attn_hc_norm         norm        hc_norm 256.0     attn_hc_pre_sinkhorn elementwise
  attn_hc_mapping_proj matmul                          h_pre 0.0625 / before_norm 0.3750
  attn_hc_sinkhorn     elementwise h_res     0.125     inv_rms 0.0156 / sum_out 2.5 / norm_out 10.0
                                                     attn_hc_aggregate    elementwise  (no saves)
                                                     attn_hc_sinkhorn     elementwise
                                                       h_res 0.25 / h_post 0.0625 / h_out 32.0
  == 每模块 256.1250 MiB                             == 每模块  45.2656 MiB
```

- 非融合分支**逐字节、逐 op、逐 param 与改动前完全一致**（3 op、同名、同 params、同
  `bwd_scratch`）→ `use_fused_mhc: false` 仍产生今天的非融合普查。
- 两条分支 **op 数（3）/ params 集合完全相同** → `mhc_wrap._link` 的下标接边不变、
  `test_param_conservation` 逐字节不变（`FusedHyperConnectionModule.__init__` 直接
  `super().__init__`，`hyper_connection.py:387`，参数名/形状完全继承）。
- Δ = **−210.86 MiB/模块 × 2 模块 = −421.7 MiB/层**（实测普查 2855.3 → 2433.5 = **−421.8**，
  差 0.1 是块对齐）。

### 1.2 Fix 2 —— norm-fp32 抬升**按种类**成立（不是整体翻 `norm_compute_dtype_bytes`）

| 位置 | 改动 |
|---|---|
| `cost_eval/model_spec.py` | 新增 `NORM_KIND_CASTING`/`NORM_KIND_NONCASTING` 常量、`norm_kind_of(d)` 助手；`DimTable.norm_kind`（默认 CASTING）、`OpSpec.norm_kind`（默认 CASTING） |
| `cost_eval/shape_eval.py` | `ResolvedOp.norm_kind` 直通（frozen dataclass，str 可哈希） |
| `cost_eval/structure_mem.py` | `_norm_save_names` 跳过 `norm_kind == rmsnorm` 的 op |
| `cost_eval/llm_config.py` | `to_dimtable` 从 `LLMConfig.normalization` 派生（RMSNorm→noncasting、LayerNorm→casting，忠实 `get_norm_cls` `layer_norm.py:187-191`） |
| `cost_eval/layers/{attention,dsa,dsv4_hybrid,ffn,head}.py` | 逐 op 盖在**由 `get_norm_cls` 产出的** 14 个 norm 上 |
| `cost_eval/layers/{residual,head}.py` | `_scale_op` / `_add_dep` / MTP 的两处 OpSpec 重建**带过** `norm_kind`（丢了它等于把被包装层的 norm 悄悄退回抬 fp32 —— 这正是 `x_xn`/`h1_xn` 曾被错抬的通道） |

**刻意不盖**的两个 norm：

- `logsoftmax`（`head.py`）—— 不是 `get_norm_cls` 的 norm，归 `softmax_compute_dtype`；
  且其 saves 早已按名被 `_norm_save_names` 排除。
- 非融合 mHC 的 `hc_norm`（`residual.py`）—— 源里 `hyper_connection.py:262-266` 是
  `rms_norm(self.cast(hidden_states, mstype.float32), ...)`，**显式 cast**，`:269-271` 的注释
  逐字说明 MindSpeed 刻意把它留在 FP32。它**真的** cast，故保持 CASTING。

**为什么不是整体翻 `norm_compute_dtype_bytes`**：那是**跨模型**口径，翻它会把
`FusedLayerNorm`（`layer_norm.py:93-101` 真 cast）一并关掉。`transformer_config.py:232-233`
的 `normalization` 默认就是 `"LayerNorm"` → 这条分支并非空谈。

**实测**：DSv4 每层 −268.0 MiB（`x_xn` −128 / `h1_xn` −128 / `q_compressed` −8 / `kv` −4）、
lm_head 段 −32.0（`h_last`）。

### 1.3 契约 O4 —— 是**测试主动发现**的，不是顺手加的

`tests/test_resolved_layer_contract.py::test_contract_field_set_is_sufficient_on_real_dsv4_flash`
在改完 Fix 2 后**变红**：按契约字段重建的「外部生产者」图比 `hand_spec` 每层多 12 MiB。
这正是该门存在的意义 —— `norm_kind` 决定真实字节，契约字段集必须包含它，否则
「图能建、显存算错」。故新增 **O4**：`type == "norm"` 的 op 必须真实带
`norm_kind ∈ {layernorm, rmsnorm}`（与 `detached` 同理由，不接受 getattr 兜底）；
非 norm 可省，**给了就必须合法**（拼错字符串会静默退回抬 fp32）。
`opdag/to_resolved.ROp` 默认 `rmsnorm` —— 本库语料 `normalization` 恒 `"RMSNorm"`
（`configuration_deepseek_v3.py:149` / `configuration_deepseek_v4.py:155`）。

---

## 2. 落点：逐层型 activation_saves vs 真机逐层直测

`PYTHONIOENCODING=utf-8 MINDFORMERS_ROOT=...\mf-src-167\mindformers python scratchpad/census_threeway.py`

| 层型 | 本轮前 | Fix 1 后 | **Fix 1+2 后** | 真机实测 | **模型/真机** | 残差 |
|---|---:|---:|---:|---:|---:|---:|
| r0 `dsv4hyb_r0_dense` | 2855.3 | 2433.5 | **2165.5** | **2235.1** | **0.969×** | −69.6 |
| r4 `dsv4hyb_r4_moe` | 2802.3 | 2380.6 | **2112.6** | **2341.2**（4 层均） | **0.902×** | −228.6 |
| r128 `dsv4hyb_r128_moe` | 2727.3 | 2305.6 | **2037.6** | **2116.1**（3 层均） | **0.963×** | −78.5 |
| lm_head 段 | 3126.0 | 3126.0 | **3094.0** | —（真机无逐层可比口径） | — | — |

**三个层型全部转为欠读**，与仲裁 §3.4 的源侧枚举预测（约 2085/2157/2027，0.92–0.96×）
同向、同量级。**残差刻意不补**：它是 kernel workspace（快照里读不出，需真机 profiler 明细）；
往上凑等于发明拟合常数，正是本轮工作要消灭的失败模式。

### run `c` 的 `act_live` 比值（本轮成功判据）

| stage | 真机 `act_live`（深度×逐层，实测） | 本轮前 | 比 | **本轮后** | **比** |
|---|---:|---:|---:|---:|---:|
| 0 | 18361.6 | 22630.3 | 1.232× | **17112.5** | **0.932×** |
| 1 | 13434.6 | 16589.0 | 1.235× | **12450.7** | **0.927×** |
| 2 | 8950.0 | 11059.3 | 1.236× | **8300.4** | **0.927×** |
| 3（含 head/loss，不并列） | 4404.7 | 8655.7 | 1.965× | 7244.2 | 1.645× |

**过读被完全消掉**（1.232–1.236 → 0.927–0.932），落在欠侧 −7%。

---

## 3. 我**刻意留下**的欠读（逐条，带量）

| # | 缺口 | 量 | 为什么不补 |
|---|---|---:|---|
| ① | kernel workspace（compressor/indexer/flash/mHC kernel 内部 scratch） | r0 −69.6 / r4 −228.6 / r128 −78.5 MiB/层 | **快照里读不出**。要闭合需真机 profiler 的 kernel workspace 明细。补它 = 发明拟合常数。 |
| ② | 两个 RMSNorm 真正保留的 **aggregated `[S,B,H]`** | −32 MiB × 2 = **−64 MiB/层** | 源里 `input_layernorm`/`pre_mlp_layernorm` 吃的是 aggregated（`transformer_layer.py:311,329`），32 MiB/次；普查让 `ln1`/`ln2` 保留**打包流** `x_xn`/`h1_xn`（128 MiB bf16）—— 后者数值上正是融合 ctx 的 `x`（`custom_op_impl.py:390` 首项，同一块设备张量），故融合分支**不重复声明** `x` 以免双计。代价即这 64 MiB。修它要动 `mhc_wrap` 的残差承载判定，**两条分支同时变** → 会破坏「`use_fused_mhc: false` 保持今天的非融合普查」这条本轮约束。 |
| ③ | `comb_xn` 被 `_scale_op` 当成残差承载放大到 `[S,B,n·H]` | **+96 MiB/层 过读**（真值：MoE routed 输出 `comb` 是 `[S,B,H]` = 32 MiB） | 与 ② 同根（`_is_residual_carrier` 的判定），同样两条分支同时变。**方向与 ①② 相反**，是本轮残差里唯一的过读项 —— 一并留待「mHC 残差承载判定」专项。 |
| ④ | 非融合 mHC 分支自身的欠读 | 约 −320 MiB/模块 | 源里 `:297-299` 另有一份 `x_streams` **fp32 副本**（`[S,B,n,H]` fp32 = 256 MiB，被 `:299` 的 matmul 保留）+ `output_cell` 的 bf16 打包流（128）。今天的非融合普查靠 `ln1` 的 fp32 抬升「凑巧」顶上其中 128。本轮约束是**保持非融合分支不变**，故不动，如实记。八跑门全部 `use_fused_mhc: true`，不触及。 |
| ⑤ | `unfused − fused` delta 的欠读 | bucket 0.518 / hand_spec 0.636 | 根因是 unfused 的**反向梯度工作集**（gathered-KV fp32 的三份梯度，拆解报告 §4.2），与本轮改动无关；本轮只减 fused 侧 → 缺口继续变大，如实记。 |

---

## 4. 八跑门 before → after（`tools/liveness_ab_validate.py --grad-mode chain2 --deltas`）

| 指标 | 本轮前 | **本轮后** |
|---|---|---|
| bucket 聚合 mean / min / **max**（n=28） | 0.867 / 0.717 / **1.125** | **0.822 / 0.701 / 1.019** |
| hand_spec·chain2 mean / min / **max** | 0.899 / 0.694 / **1.143** | **0.862 / 0.672 / 1.116** |
| run `c` bucket 逐 stage sim/real | 1.083 / 1.125 / 1.020 / 0.931 | **0.901 / 0.928 / 0.865 / 0.880** |
| run `a`（fused+重算） | 0.741 / 0.862 / 0.877 / 0.946 | 0.713 / 0.815 / 0.828 / 0.945 |
| run `b`（unfused+重算） | 0.720 / 0.717 / 0.720 / 0.742 | 0.706 / 0.701 / 0.704 / 0.728 |
| `unfused − fused` delta @stage3（bucket / hand_spec） | 0.545 / 0.644 | **0.518 / 0.636** |
| `b − a` @stage1（bucket，真机 29298.3） | 18873.0（0.6442） | **18873.0（0.6442）—— 逐 MiB 不变** |
| 两条真机不变量 / 模型 ×1 不变量 / `REAL_SHA256` 指纹 | PASS | **PASS（一条未动）** |

**均值继续下降是预期且如实的**：拆解报告 §0 已证误差在不同场景**反向**（无重算端过读、
有重算端与 unfused 端欠读）。本轮把过读那一端**修到 0.93**（此前 1.23），抵消彻底消失，
欠读全部露出来。**不为让均值好看而保留一个源码上错误的字节。**

`b − a`（unfused 物化的 ×1 量）**逐 MiB 不变** —— `a` 与 `b` 同幅下降、差值抵消；
`remat_saves` 同源下降只体现在各跑绝对值上。

---

## 5. 锚点重钉台账（old → new，逐条理由）

**未动**：`REAL*` / `CSV*` 一切真机常数；`REAL_SHA256` 指纹表；两条真机不变量与模型 ×1
不变量；run d 不可评分规则；`nr_moe_frag_factor` / `kept_frag_factor` 两个标定 margin
（**一个字节没动** —— 本轮变化全部来自去掉一处真实过读）。

### 5.1 DSv4 侧

| 位置 | old → new | 理由 |
|---|---|---|
| `test_acceptance_gate::GOLDEN_BUCKET` / `GOLDEN_LIVENESS_{DATAFLOW,CHAIN2}` | 8×4 表整体下移（如 `c` s1 23638.8→19503.6） | 普查按源订正，saves 降 689.8 MiB/层 |
| `...::GOLDEN_AGG` | bucket 0.867/0.717/1.125 → **0.822/0.701/1.019**；hand_spec·chain2 0.899/0.694/1.143 → **0.862/0.672/1.116** | max 首次跌破 1.02 = 过读被完全消掉 |
| `...::test_delta_magnitude_gap_is_recorded` | 0.545/0.644 → **0.518/0.636** | 只修 fused 侧，delta 更欠，如实记 |
| `...::test_unfused_on_cell_ratios...` | [0.808,0.818,0.822,0.835] → **[0.804,0.813,0.817,0.830]**；bucket 0.717–0.742 → 0.701–0.728 | RMSNorm 一条也作用在 unfused 跑上 |
| `test_pp4_recompute_anchor::THEO_ON` | s0 16766.2→16498.2 / s1 11643.0→11375.0 / s2 11565.9→11297.9 / s3 21037.9→21005.9 | `remat_saves` 同源；s3 峰在 head 段只吃 −32.0 |
| `...::THEO_MTP` | s0-s2 同上；s3 28375.0→**28011.0** | 同上 |
| `...::THEO_PP8` | 全 stage −268.0（s7 −32.0） | 同上 |
| `...::BAND_OFF` | 实测 1.045/1.073/0.977/0.885 → **0.975/0.996/0.917/0.864**；带 (1.00,1.10)/(1.02,1.12)/(0.93,1.03)/(0.85,0.93) → **(0.93,1.02)/(0.95,1.05)/(0.87,0.97)/(0.82,0.91)** | s0/s1 由过读收到约 1.0（目标达成）；s0/s2/s3 已转欠读 = **OOM-不安全**，如实记带 |
| `...::_PP8_OVER_BAND` | (1.00,1.25) → **(1.00,1.20)** | 过读幅度再压；⚠ **s3 只剩 1.0007（高出真机 8.6 MiB），在翻转边缘**；下界刻意保持 1.00，真翻转必须变红 |
| `test_probe185_recon::test_fused_per_layer_increment` | 2702.5 → **2568.5**（/3109 = 0.826） | 同一记录门，样例移动 |
| `...::_STD_ON_THEO` | MHA 8074.8/14802.8 → **8042.8/14786.8**；GQA 7594.8/14490.8 → **7562.8/14474.8** | RMSNorm 不 cast 也作用在 std 路径 |
| `...::test_p3p_m8_theoretical_and_gap` | 17370.2 → **17102.2**（/25343.5 = 0.675，仍欠读） | 同 `remat_saves` |
| `scorecard_anchors` DSv4-fused (base) | band (0.88,0.94)→**(0.87,0.93)**，ratio 0.908→**0.902** | 同上；**OOM-不安全，如实记** |
| `scorecard_anchors` DSv4 mHC(x4)+MTP | band (0.83,0.90)→**(0.81,0.88)**，ratio 0.862→**0.846** | 同上；仍排除历史翻转值 1.088 |
| `test_from_mindformers::test_dsv4align_roundtrip_same_peak` | 13994.8 → **13899.8** | 同上 |

### 5.2 DSv3 侧（**因为 DSv3 站点同样是 RMSNorm**）

判据不是推断而是快照实证：`configuration_deepseek_v3.py:149` `normalization="RMSNorm"` →
`gpt_layer_specs.py:109,111,125,126,132,134,142,149,150,153,220` 与
`multi_token_prediction.py:243,244` 全部走 `get_norm_cls(normalization, fused_norm)` →
`layer_norm.py:190-191` 返回 `FusedRMSNorm` → `:151-155` 不 cast。仲裁 §3.2 提出的
「关掉它需要的证据」由此闭合。

| 位置 | old → new | 理由 |
|---|---|---|
| `test_dsv3_golden::GOLDEN_BREAKDOWN` | `act_live` 3279945728 → **3265265664**（−14.0 MiB） | 同上；逐桶之和 == 峰值的不变量保持 |
| `...::GOLDEN_PEAK_BYTES` | 13108142592 → **13093462528**（12500.9 → 12486.9 MiB；真机 12473.1 → 1.0022 → **1.0011**） | 同上 |
| `test_ce_optstep` / `test_from_mindformers`(DSv3) / `test_kept_frag_margin` / `test_x4_experts_wrap` | DSv3 4L full 12437.9 → **12423.9** | 同一硬门的四处引用 |
| `test_kept_frag_margin::test_margin_off_reproduces_pre_fix_underprediction` | 15712.5 → **15474.5** | margin-off 基线，margin 仍是唯一变量 |
| `...::test_select_mlp_keepattn_unchanged_moe_recomputed` | 15062.0 → **14696.0**（真机 15765 → 0.955 → **0.932**） | 已跌出 ±5% |
| `test_x4_p2p_pp::test_pp2_..._anchor` | s0/s1 11162.1/45991.4 → **10458.1/45611.4** | 同上 |
| `test_std_attn_anchor::BAND` | mha pp2 s0 (0.95,1.05)→**(0.90,1.00)**（0.981→**0.934**）；gqa pp2 s0 (0.86,1.05)→**(0.82,0.92)**（0.876→**0.852**） | 两个 s0 跌出 ±5%；残差①（真机 s0 GQA约等于 MHA、KV 缩水未兑现）依旧未修，与本次叠加 |
| `scorecard_anchors` pp2-stage0 (optstep) | band (1.05,1.10)→**(1.00,1.06)**，1.089→**1.021** | 仍 OOM-安全，余量大幅收窄；lo=1.00 守住「仍在安全侧」 |
| `scorecard_anchors` pp2-stage1 | 1.007→**0.999**（band 未动） | **刚翻到 OOM-不安全**（欠 43.6 MiB / 0.10%） |
| `scorecard_anchors` cp2-none | band (1.00,1.06)→**(0.96,1.02)**，1.007→**0.991** | **OOM-不安全**；D1 margin 仍开仍 0.6 |
| `scorecard_anchors` DSv3 8L none | band (0.99,1.06)→**(0.95,1.01)**，1.003→**0.981** | 同上 |
| `scorecard_anchors` select self_attn | band (0.98,1.05)→**(0.94,1.00)**，1.001→**0.970** | **OOM-不安全**；`kept_frag` margin 未重标 |
| `scorecard_anchors` select mlp | band (0.94,1.00)→**(0.90,0.96)**，0.955→**0.932** | 同上，已跌出 ±5% |

### 5.3 一条测试被**改写**（机制保留、样例移动，§6.6 精神）

`tests/test_scorecard_anchors.py::test_scorecard_has_no_oom_unsafe_dsv3_no_recompute`
→ `test_d1_margin_is_present_and_effective`

| | 原 | 新 |
|---|---|---|
| **不变量** | D1 `nr_moe_frag_factor=0.6` margin 必须在、必须有效 | **同一条，未变** |
| **举例** | 用「两锚点 ratio ≥ 1.0」证明它没被误删 | 用「margin **开 > 关**」+ 幅度带证明；对照点 0.9130/0.9172 是**实测**（`scratchpad/probe_d1_margin_off.py`），非拟合 |
| **新增** | — | 第三条断言 `ratio < 1.0`：把 **OOM-不安全**状态如实钉住；有人把它「调回」安全侧会变红 |

`tests/test_x4_p2p_pp.py` 的 `s1 >= 45655.0` 同理：s0 保留 OOM-安全硬断言（1.021），
s1 改为「已记录欠读带 `(0.995, 1.0]`」，双向守卫（下界防恶化、上界防被调回去）。

**没有删除任何一条不变量；没有新增/删除任何用例**（`1892 passed` 与改动前同数）。

---

## 6. 现在已知为 **OOM-不安全** 的锚点（模型欠读真机）

**不调参掩盖。** 唯一正确的补法是**基于真机 profiler 的 kernel workspace 项**，不是普查充气。

| 锚点 | ratio | 缺口 |
|---|---:|---|
| `pp2-stage1 (loss,k_ce=8)` | 0.999 | 43.6 MiB（**本轮新翻转**） |
| `cp2-none (loss,k_ce=4)` | 0.991 | 186 MiB（**本轮新翻转**） |
| `DSv3 8L none (dp2)` | 0.981 | 376 MiB（**本轮新翻转**） |
| `select self_attn (keep-FFN)` | 0.970 | 572 MiB（**本轮新翻转**） |
| `select mlp (keep-attn)` | 0.932 | 1069 MiB（原 0.955，本轮跌出 ±5%） |
| `std 116 mha pp2 s0` | 0.934 | 1047 MiB（原 0.981，**本轮新跌出 ±5%**） |
| `std 116 gqa pp2 s0` | 0.852 | 2233 MiB（原 0.876，叠加已知残差①） |
| `DSv4-fused (base)` | 0.902 | 1516 MiB（上一轮已不安全，本轮更欠） |
| `DSv4 mHC(x4)+MTP` | 0.846 | 3264 MiB（上一轮已不安全，本轮更欠） |
| `pp4 no-recompute OFF` s0/s2/s3 | 0.975 / 0.917 / 0.864 | 三个 stage（**本轮新翻转**） |

仍 OOM-安全的：`pp2-stage0` 1.021（余量已很薄）、`select both` 1.001、`pp8` s1–s6
1.00–1.15（**s3 仅 1.0007，翻转边缘**）、DSv3 full 系列 0.99–1.00 带内。

---

## 7. 验收

```
python -m pytest tests -q   →  1892 passed, 268 warnings
```

与改动前同数（无新增/删除用例，只移动样例值 + 一处「同一不变量换举例方式」的改写）。
