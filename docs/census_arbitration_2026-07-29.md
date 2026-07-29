# 手写激活普查的三方仲裁（2026-07-29）

> **问题**：fused + 无重算（run `c`）下，模型 `act_live` 桶比真机大 **1.62–1.64×**；其余九个桶
> **合计**只差 279–1503 MiB（1.3–8.5%）。`act_live` ≡ `structure_mem.activation_saves` ≡
> `cost_eval/layers/dsv4_hybrid.py` 那张手写 `saves=[...]` 名册，其唯一依据是一句
> 「最坏情况：每个 op 的 bprop 都持有其输入/输出」的注释。见
> `analysis/sim_vs_real_gap_decomposition_2026-07-28.md` §5.6。
>
> **方法**：三方对账 —— ①真机逐层实测；②手写普查；③纯 AST 源码抽图（`cost_eval/opdag/`）。
> 逐张量判决，每条带**权威快照**（`E:\97-codes\torch_parallel\mf-src-167\`，
> `mindformers` @ `26354ff64`，即真机实跑那份）的 `file:line`。
>
> **纪律**：绝不编造真机数；绝不为把比值凑到 1.0 而发明常数；源里读不出来的就留着并如实报残差。

复现：
```bash
PYTHONIOENCODING=utf-8 MINDFORMERS_ROOT=E:\97-codes\torch_parallel\mf-src-167\mindformers \
  python scratchpad/census_threeway.py --extracted     # 三方逐张量表
PYTHONIOENCODING=utf-8 MINDFORMERS_ROOT=E:\97-codes\torch_parallel\mf-src-167\mindformers \
  python scratchpad/census_ops_dump.py 2               # 抽取侧逐 op（判「不在场」是 VJP 说不留 vs 节点被跳过）
PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas
```

---

## 0. 三方逐层型对照（run `c` = fused / 无重算 / L8 / m4 / seq4096 / b=1）

层号映射：评估器 `layer_id` 里 0 是 embedding → **真机 decoder 层 k = layer_id − 1**。
`compress_ratios=[0,4,128,4,128,4,128,4]` → r0={L0}、r4={L1,L3,L5,L7}、r128={L2,L4,L6}。

| 层型 | 真机实测（MiB/层） | 手写普查 `activation_saves` | 手写/真机 | 抽取图（**下界**） | 抽取/真机 |
|---|---:|---:|---:|---:|---:|
| **r0** `dsv4hyb_r0_dense` | **2235.1**（单样本 L0） | **3749.2** | **1.677×** | 1677.2 | 0.750× |
| **r4** `dsv4hyb_r4_moe` | **2355.3 / 2354.0 / 2370.0 / 2285.5**（均 2341.2） | **3703.8** | **1.582×** | 1212.4 | 0.518× |
| **r128** `dsv4hyb_r128_moe` | **2124.2 / 2105.0 / 2119.2**（均 2116.1） | **3625.3** | **1.713×** | 1173.2 | 0.554× |

- 模型的「每(层×微批)」蕴含值 = (3703.8+3625.3)/2 = **3664.55**；真机 stage1 实测
  (2124.2+2354.0)/2 = **2239.1** → **1.637×**，与 §5.6 的 1.637 逐格吻合 ✅（对账口径正确）。
- **抽取图是有声明的下界**：`workspace_bytes`/`bwd_scratch_bytes` 恒 0、rope 内部未展开、
  fused kernel 自身的 `output` 未计入 saves、`permute` 被当成零字节视图 —— 故它只能用来
  **证伪某张量「不被任何 bprop 需要」**，不能用来定总量。

> ⚠ **口径提醒**：若逐张量求和时**不**施加 `structure_mem._dt` 的 norm-fp32 抬升
> （`structure_mem.py:286-288`），三个层型分别少 **524.0 MiB**（3225.2 / 3179.8 / 3101.3）。
> `act_live` 用的是**抬升后**的值，本文全部按抬升后记。524.0 = `q`(+256) + `x_xn`(+128) +
> `h1_xn`(+128) + `q_compressed`(+8) + `kv`(+4)。

---

## 1. 权威快照对手写普查的**逐条判决**（注意力主干，三层型共有）

分类：**absent**（声明了但源里没有任何 bprop 需要它）/ **wrong dtype** / **wrong shape** /
**double-counted identity**（同一源张量两个名字）/ **missing**（源里保留、普查没建）。

### 1.1 `q_hnorm_fp32` —— **wrong dtype**，512 → 256 MiB

普查依据（`dsv4_hybrid.py:20-22, 32, 88`）自称：
`deepseek_v4_hybrid_attention.py:239-245` 是 `q = rms_norm(cast(q, fp32), q_rms_gamma)`。
**权威快照里没有这段代码。** 逐字：

```
244:        eps = self.config.layernorm_epsilon
245:        q = q * mint.rsqrt(mint.mean(q * q, dim=-1, keepdim=True) + eps)
```

—— 纯 `mint` 逐元素算术，**没有任何 `cast(..., float32)`**。全文件只有两处 `self.cast`
（`:253` `cast(kv_4d, compute_dtype)`、`:299` `cast(output, ori_dtype)`），都不是这里。
`self.rms_norm = ops.rms_norm`（`:158`）与 `self.q_rms_gamma`（`:159-161`）在 `construct`
里**从未被调用**（全文件只在 `:167` `reset_parameter` 出现 `fill_`）→ 那条注释描述的是
另一个 commit 的代码。输出 dtype = 输入 dtype = `compute_dtype` = **bf16**。

抽取侧独立同意：`q` @ `deepseek_v4_hybrid_attention.py:240` = **256.0 MiB bf16**，全层无 fp32 孪生。

**影响：−256.0 MiB/层 × 8 层。**

### 1.2 `q`（`q_hnorm` op 的 saves）—— **wrong dtype**（伪 norm 抬升），512 → 256 MiB

`q_hnorm` 被声明成 `OpType.NORM`（`dsv4_hybrid.py:124`）→ `structure_mem._norm_save_names`
把 `q` 收进 norm 集（`structure_mem.py:118-133`）→ `_dt` 按 `layernorm_compute_dtype=fp32`
把它从 2 B 抬到 4 B（`structure_mem.py:135-139`）。但 `:245` **不是 layernorm/RMSNorm 模块**，
是 bf16 逐元素乘 —— `layernorm_compute_dtype` 这条 yaml 根本管不到它。抽取侧把 `:245` 判成
**5 个 `elementwise` 节点**（探针 `census_ops_dump.py 2` 的 op 11–15），与源一致。

**影响：−256.0 MiB/层 × 8 层。**

> **对前一轮线索的订正**：`q` 与 `q_hnorm_fp32` **不是**同一张量的两个名字（不是
> double-counted identity）。源里确实有两块 `[S,B,64·512]`：
> ①`q` = `linear_q_up_proj` 输出，被 `:245` 的两个 `mul` 的 bprop 保留；
> ②`query` = `csa.py:674` `permute(q_roped,(1,0,2,3))` 的 **BSND 副本**，被
> `ctx.save_for_backward`（`csa.py:113`/`:224`）保留。
> 二者共 **512 MiB bf16**，普查记 **1024 MiB**（两张都错抬 fp32）→ 净过读 **512**，
> 机制是 **2× wrong dtype**，不是重复计数。

### 1.3 `cg_fp32` —— **wrong dtype**，512 → 256 MiB

源 `deepseek_v4_hybrid_attention.py:286-291`：

```
286:        cg = self.permute(self.reshape(core_grouped, (sq * bsz, o_groups, d)), (1, 0, 2))
287:        wo = self.permute(wo_a_3d, (0, 2, 1))
288:        # MindSpeed's grouped ``torch.einsum`` stays in the model dtype.  A
289:        # FP32 promotion changes a few BF16 rounding decisions before wo_b and
290:        # is enough to create long-run optimizer drift.
291:        inter = self.bmm(cg, wo)
```

`:288-290` 是源码作者**显式拒绝** fp32 提升的注释；普查（`dsv4_hybrid.py:33, 95, 284-286`）
声明的 `bmm(cast(cg, fp32), cast(wo, fp32))` 在快照里不存在。抽取侧：`cg` 256.0 MiB **bf16**
@ `:286`，被 `matmul :291` 保留。

**影响：−256.0 MiB/层 × 8 层。**

### 1.4 `inv_rope_out` —— **absent**，−256 MiB

`:271` `core_out = self._apply_forward_rope(core_out, freqs, 1.0, inverse=True)`，返回值是
`:215` 的 `self.cat([t_nope, t_pe], dim=-1)`（一块新的 `[S,B,64,512]` bf16 = 256 MiB）。
它的**全部**下游是 `:278` `reshape` → `:286` `reshape`+`permute`，其 VJP 分别是 reshape 与
逆 permute，**都不需要保留输入**；bmm 真正保留的操作数是 `cg`（§1.3，已单列）。
抽取侧逐 op 佐证：`census_ops_dump.py 2` 的 op 118 `view :215 → __ret__i9#3 saves=[-]`，
下游第一个 save 是 op 124 的 `cg`。

> **与 2026-07-23「185 F 差分合账」冲突**（`dsv4_hybrid.py:270-275`）：那次在 185/seq2048 上
> 用差分补了 `128 + 2×16 = 160`。但那次拟合是**叠在一个已经过读的普查之上**做的（当时
> `q`/`q_hnorm_fp32`/`cg_fp32` 三处 dtype 错误都在），故「+149 欠读」不能作为
> 「`inv_rope_out` 被保留」的证据 —— 典型的拟合常数吸收误差。本轮以**逐层直测 + 源 VJP** 为准。

### 1.5 前向 rope 的保留对 —— **missing**，+130 MiB/层

`rope_utils.py` `ApplyRotaryPosEmb.construct` 对**三次**调用（q 前向 `:260`、key 前向 `:261`、
core_out 逆向 `:271`）走的是同一条非融合分支（`mla_output_remove_interleaving=True` 使
`fused_interleaved_mla` 恒 False，`rope_utils.py:160-162`）：

```
169:        t = self.cast(t, self.rotary_dtype)
186:            t_rot = self._rotate_half(t, rotary_interleaved)
187:            output = self.add(self.mul(t, cos_), self.mul(t_rot, sin_))
```

两个 `mul` 的 bprop 各保留一个操作数 → 每次调用留下 `t` 与 `t_rot` 两块
`[S,B,n,rot_dim]`，dtype = `config.rotary_dtype` = **fp32**
（`models/deepseek4/configuration_deepseek_v4.py:149` `rotary_dtype="fp32"`，
`parallel_core/transformer_config.py:166-167` 默认亦 `"float32"`；launcher yaml 未覆盖）。

普查只建了**逆向**那一对（`inv_rope_f32`/`inv_rope_rot`，2×64 MiB），**漏了前向 q 的一对**
（2×64 = 128 MiB）与 key 的一对（2×[S,B,1,64] fp32 = 2 MiB）。

**影响：+130.0 MiB/层 × 8 层**（方向与前四条相反 —— 这是**欠读**，如实补上）。

### 1.6 `cmp_residual` —— **wrong shape**，8 MiB(r4) / 4 MiB(r128) → **4 B**

源 `csa.py:64-70`（`_prepare_sparse_flash_mla`，两个 fused `_Function` 的 forward 第一行都调它）：

```
 64:    cmp_residual = Tensor(
 65:        [int(ori_length) % kernel_cmp_ratio],
 66:        dtype=mstype.int32,
 67:    )
```

—— **一个 1 元素 int32 标量**（4 字节），不是普查声明的 `[S,B,coff·v_head_dim]`
（`dsv4_hybrid.py:183`「压缩器残差（coff·vd）」）。块对齐后占 512 B。

**影响：−8.0 MiB（r4）/ −4.0 MiB（r128）；r0 原本没建，补上 +512 B。**

### 1.7 `sinks` —— **missing**，+256 B

`csa.py:687` `attn_sink = ops.cast(attn_sink, mstype.float32)`，该 **fp32 副本**（不是那个
Parameter 本身）逐字进 `ctx.save_for_backward`（`csa.py:113` / `:224`）。`[n_heads]=64` fp32
= 256 B。此前登记在 `tests/test_source_truth_saves_audit.py::KNOWN_GAPS`。

### 1.8 `idx_weights` —— **wrong dtype**（r4 独有），0.5 → 1.0 MiB

`csa.py:696` 实参是 `ops.cast(weights, mstype.float32)`；普查按默认 bf16 建。**+0.5 MiB/r4 层**。

### 1.9 `sparse_indices` —— **missing 但净 0**（r4 独有）

`csa.py:61-62` `sparse_indices = mint.unsqueeze(cmp_sparse_indices, dim=2)`（视图，无拷贝），
逐字 save。同名张量已由上游 `indexer` op 声明 saved，而 `structure_mem` 的 saves 按**名**
去重（`structure_mem.py:261-262`）→ 补声明后 **净 0**。同时源侧 `use_sparse_indices` 要求
`has_cmp and cmp_ratio == 4`（`csa.py:61`）→ **r128/r0 不保留它**，与普查现状一致 ✅。

### 1.10 fused `_Function` 选择（层型差异的源依据）

`csa.py:688-716`：**只有 `enable_indexer`（`compress_ratio == 4`，`csa.py:607`）** 走
`FusedSparseFlashMlaWithIndexerLoss`（11 项 saved，`csa.py:224`）；**r0 与 r128 都走**
`FusedSparseFlashMla`（8 项 saved，`csa.py:113`）。`enable_compress = compress_ratio > 0`
（`csa.py:593-595`）→ r0 的 `cmp_kv` 为 `None`，被 `if tensor is not None` 过滤掉。

于是三层型 kernel-ctx 的**源真值**（B=1,S=4096,n=64,vd=512）：

| ctx 名 | r0 | r4 | r128 | 字节依据 |
|---|---:|---:|---:|---|
| `query` [B,S,64,512] bf16 | 256 | 256 | 256 | `csa.py:674` permute |
| `ori_kv` [B,S,1,512] bf16 | 4 | 4 | 4 | `csa.py:675` |
| `cmp_kv` [B,S/r,1,512] bf16 | — | 1.0 | 0.031 | `csa.py:677`；r0 为 None |
| `sparse_indices` [B,S,1,512] int32 | — | 8（视图,净 0） | — | `csa.py:61-62` |
| `query_index` bf16 | — | 64 | — | `csa.py:694` |
| `key_index` bf16 | — | 1 | — | `csa.py:695` |
| `weights` **fp32** | — | 1 | — | `csa.py:696` |
| `cmp_residual` int32 标量 | 4 B | 4 B | 4 B | `csa.py:64-67` |
| `sinks` [64] fp32 | 256 B | 256 B | 256 B | `csa.py:687` |
| `output` [B,S,64,512] bf16 | 256 | 256 | 256 | kernel 输出 |
| `softmax_lse` fp32 | 1 | 1 | 1 | kernel 第 2 输出 |

**这解释了 r128 比 r4 低 ~239 MiB 中的 74 MiB**（`query_index` 64 + `sparse_indices` 8 +
`key_index` 1 + `weights` 1 = 74，全是 indexer 侧）；其余要到 indexer/compressor 内部找
（见 §3 未归因）。

---

## 2. 已落地的修改与字节影响（commit `537df67`）

改的全在 `cost_eval/layers/dsv4_hybrid.py`（注意力主干）。每条的分类、依据、字节：

| # | 张量 | 分类 | 源依据 | 修前 | 修后 | Δ/层 |
|---|---|---|---|---:|---:|---:|
| 1 | `q_hnorm_fp32` → `q_hnorm` | wrong dtype | `deepseek_v4_hybrid_attention.py:244-245`；`:158-161` 死代码 | 512.0 fp32 | 256.0 bf16 | **−256.0** |
| 2 | `q`（`q_hnorm` op 由 NORM→ELEMENTWISE） | wrong dtype（伪 norm 抬升） | 同上 + `structure_mem.py:118-139` | 512.0 | 256.0 | **−256.0** |
| 3 | `cg_fp32` → `cg` | wrong dtype | `:286-291`（注释 `:288-290` 拒绝 fp32） | 512.0 fp32 | 256.0 bf16 | **−256.0** |
| 4 | `inv_rope_out` | **absent** | `:215`/`:271` + `:278`/`:286` 的 VJP | 256.0 | 0 | **−256.0** |
| 5 | 前向 RoPE `(t, t_rot)`（q + key） | **missing** | `rope_utils.py:169,186-187` + `rotary_dtype=fp32` | 0 | 130.0 | **+130.0** |
| 6 | `cmp_residual` | wrong shape | `csa.py:64-67`（1 元素 int32） | 8.0(r4)/4.0(r128) | 512 B | **−8.0 / −4.0** |
| 7 | `sinks` | missing | `csa.py:687` + `:113`/`:224` | 0 | 256 B | +0.0002 |
| 8 | `idx_weights` | wrong dtype | `csa.py:696` | 0.5 | 1.0 | +0.5（仅 r4） |
| 9 | `sparse_indices`(=`topk_indices`) | missing 但净 0 | `csa.py:61-62` 视图 + `structure_mem.py:261-262` 按名去重 | — | — | 0（仅 r4） |

**逐层型合计**

| 层型 | 修前 | 修后 | 真机 | 修前比 | **修后比** |
|---|---:|---:|---:|---:|---:|
| r0 | 3749.2 | **2855.3** | 2235.1 | 1.677× | **1.277×** |
| r4 | 3703.8 | **2802.3** | 2341.2（4 层均） | 1.582× | **1.197×** |
| r128 | 3625.3 | **2727.3** | 2116.1（3 层均） | 1.713× | **1.289×** |

### run `c` 的 `act_live` 比值（本任务的成功判据）

| stage | 真机 `act_live`（深度×逐层，实测） | 模型（修前） | 比（修前） | 模型（修后） | **比（修后）** |
|---|---:|---:|---:|---:|---:|
| 0 | 18361.6 | 29812.2 | 1.624× | 22630.3 | **1.232×** |
| 1 | 13434.6 | 21987.5 | 1.637× | 16589.0 | **1.235×** |
| 2 | 8950.0 | 14658.3 | 1.638× | 11059.3 | **1.236×** |
| 3（含 head/loss，不并列） | 4404.7 | 10455.2 | 2.374× | 8655.7 | 1.965× |

**过读量减少 62%**（stage1：+8552.9 → +3154.4 MiB）。

### 八跑门 before → after（`tools/liveness_ab_validate.py --grad-mode chain2 --deltas`）

| 指标 | before | after |
|---|---|---|
| bucket 聚合 mean / min / **max**（n=28） | 0.946 / 0.748 / **1.394** | 0.867 / 0.717 / **1.125** |
| hand_spec·chain2 mean / min / **max** | 0.955 / 0.729 / **1.400** | 0.899 / 0.694 / **1.143** |
| run `c` bucket 逐 stage sim/real | 1.327 / 1.394 / 1.237 / 0.996 | **1.083 / 1.125 / 1.020 / 0.931** |
| run `a`（fused+重算） | 0.799 / 0.958 / 0.977 / 0.946 | 0.741 / 0.862 / 0.877 / 0.946 |
| run `b`（unfused+重算） | 0.748 / 0.749 / 0.752 / 0.771 | 0.720 / 0.717 / 0.720 / 0.742 |
| `unfused − fused` delta @stage3（bucket/real） | 0.603 | 0.545 |
| 两条真机不变量 / 模型 ×1 不变量 / REAL 指纹 | PASS | **PASS（一条未动）** |

**均值下降是预期且如实的**：拆解报告 §0 已证误差在不同场景**反向**（无重算端过读、有重算端与
unfused 端欠读），总比值 0.946 是二者互相抵消的假象。本次只修了过读那一端 → 抵消消失，
欠读露出来。**不为了让均值好看而保留一个源码上错误的字节。**

### 重算跑受到的影响（`remat_saves = activation_saves − checkpoint_input` 同源）

- `remat_saves` 随之下降 → run `a`/`b`/`e`/`f`/`g`/`h` 全部更欠读（见上表）。
- **`b − a`（unfused 物化）几乎没动**：stage1 bucket 18865.5 → **18873.0**
  （真机 29298.3，比值 0.6439 → 0.6442）—— `a` 与 `b` 同幅下降、差值抵消。
  **既没变好也没变坏**：该欠读的根因是 unfused 的**反向梯度工作集**
  （gathered-KV fp32 的三份梯度，拆解报告 §4.2），与本次改动无关。
- stage3 的 `unfused − fused` delta 变差（14689.8 → 13283.8 / 真机 24380.1，0.603 → 0.545），
  同一原因：只减了 fused 侧。

---

## 3. 未归因 / 已识别但**未**落地的残差

修后 r4 仍高 **461 MiB**（2802.3 vs 2341.2）、r128 高 **611 MiB**、r0 高 **620 MiB**。
按源逐项枚举，**这些不是「不知道在哪」，而是「知道在哪但改动跨口径、需单独决策」**：

### 3.1 mHC 用的是**非融合**建模，而站点跑的是**融合** kernel —— 最大单项 ≈ −421 MiB/层

站点 `use_fused_mhc: true` → `FusedHyperConnectionModule`（`hyper_connection.py:369`）+
`npu_mhc_pre_sinkhorn` / `npu_mhc_post`。`cost_eval/layers/residual.py:113-115` 却按
**非融合** `HyperConnectionModule`（`hyper_connection.py:262-266`）建了一张
`{prefix}_hc_norm [S,B,n·H] fp32` = **256 MiB × 2 次/层 = 512 MiB**。

融合 kernel 的 ctx 真值（`hyper_parallel/.../custom_op_impl.py:390-391`
`ctx.save_for_backward(x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out)`；
形状逐字在 `hyper_parallel/.../mhc_pre_sinkhorn.cc:24-50`，
`bs=s=4096, seq_len=b=1, n=4, c=H=4096, fusion_size=n²+2n=24, num_iters=20`）：

| ctx 张量 | shape | dtype | MiB |
|---|---|---|---:|
| `x`（打包残差流） | `[4096,1,4,4096]` | bf16 | 128.0 |
| `h_pre` | `[4096,1,4]` | fp32 | 0.0625 |
| `hc_before_norm` | `[4096,1,24]` | fp32 | 0.375 |
| `inv_rms` | `[4096,1,1]` | fp32 | 0.0156 |
| `sum_out` | `[40,4096,1,4]` | fp32 | 2.5 |
| `norm_out` | `[40,4096,1,4,4]` | fp32 | **10.0** |
| `phi`/`bias`（参数） | — | fp32 | 1.5（持久，不计激活） |

后 5 项在调用点被 `h_in, h_post, h_res_flat, *_ = npu_mhc_pre_sinkhorn(...)`
（`hyper_connection.py:413`）用 `*_` **丢掉了 Python 名字，但仍被 ctx 强引用**
（`mhc_pre_sinkhorn.cc:61` 无条件分配全部 8 个输出）——**没有 Python 名 ≠ 没有设备内存**。
`npu_mhc_post` 另存 `(x, h_res, h_out, h_post)`（`custom_op_impl.py:331`），其中
`h_out` = sublayer 输出 `[S,B,H]` bf16 **32 MiB**（普查完全没建）。

→ 融合真值 ≈ 每次调用 `12.95(内部) + 0.31(h_res/h_post) + 32(h_out)` = **45.3 MiB**
（`x` 128 MiB 已由普查的 `x_xn`/`h1_xn` 承担）；普查记 256 → **每层 −485.5 + 64 = −421.5 MiB**。

**为什么没落地**：需要给 `DimTable` 增 `use_fused_mhc` / `mhc_sinkhorn_iterations` 两个字段
并改 `build_hyper_connection_ops` 的 op 结构（非融合分支必须保留 —— 那条路径**确实**物化
256 MiB fp32，见 `hyper_connection.py:262-266,297-298,108-116`）。这是一次**独立的口径变更**，
且会把已经转欠侧的 DSv4 scorecard 锚推到 ~0.75，应与 unfused 欠读一起决策。

### 3.2 `FusedRMSNorm` **不做** fp32 cast —— `structure_mem._dt` 的抬升每层多读 **268 MiB**（本轮前 524）

`layer_norm.py:151-155` 逐字：

```
    def construct(self, x):
        output = self.norm(x, _to_local(self.weight), self.eps)[0]
        return output
```

`x` 直通，**没有** cast；`self.cast`（`layer_norm.py:149`）是**死属性**。对照
`FusedLayerNorm.construct`（`layer_norm.py:93-101`）**才有** `x = self.cast(x, compute_type)`。
站点 `normalization: RMSNorm` + `fused_norm` → `get_norm_cls` 返回 `FusedRMSNorm`
（`layer_norm.py:190-191`），故 `layernorm_compute_dtype: float32` 在这里**只影响 gamma 参数
dtype**（`layer_norm.py:146`），**不产生任何 fp32 激活**。

而 `structure_mem._dt`（`structure_mem.py:135-139`）把所有 norm op 的 saves 按
`norm_compute_dtype_bytes` 抬到 4 B。DSv4 每层被抬的是：
`x_xn` +128、`h1_xn` +128、`q` +256（本轮已按 §1.2 单独修掉）、`q_compressed` +8、`kv` +4
= **524.0 MiB/层**（本轮修掉其中 256，剩 **268.0 MiB/层**）。

**为什么没落地**：`norm_compute_dtype_bytes` 是**跨模型**口径（DSv3 等锚点都是在带抬升的前提下
标定的），改它会同时移动一批**在别的真机上标定**的锚点，本轮无法验证。
**关掉它需要的证据**：DSv3 站点也用 `RMSNorm`，若同样成立，则 DSv3 全部锚点需要重标。

> 另注：`ops.rms_norm` 的 grad 节点内部保留什么（约定的 `RmsNormGrad(dy, x, rstd, gamma)`
> 意味着 bf16 `x` + fp32 `rstd`）**不在快照内**，只能断言 Python 层不产生 fp32 副本。

### 3.3 抽取侧的结构性下界（不是本次能修的）

`extracted` 逐层给 1677.2(r0) / 1212.4(r4) / 1173.2(r128) —— 分别是真机的
0.750 / 0.515 / 0.554×。已知**成因**（`to_resolved.py` 模块 docstring 逐条声明）：
`workspace_bytes`/`bwd_scratch_bytes` 恒 0、fused kernel 自身的 `output` 不计 saves、
`permute` 被当零字节视图（`csa.py:674` 的 BSND 副本因此丢了 256 MiB）、rope 内部未展开、
MoE 的 grouped-GEMM 支路被跳过。故它只可用于**证伪「某张量被保留」**，不可定总量。

### 3.4 逐层型源真值枚举 vs 实测（收口检验）

把 §1.10 + §3.1 + MoE 侧源枚举加总（B=1,S=4096,H=4096,n=4,E=8,K=2,F_moe=2048）：

| 组成 | 依据 | r0 | r4 | r128 |
|---|---|---:|---:|---:|
| 注意力主干（含 kernel ctx、三对 rope、cg、intermediate、两个 lora norm） | §1.10 + `deepseek_v4_hybrid_attention.py` | ~1130 | ~1330 | ~1200 |
| mHC（2 次调用：打包流 2×128 + 内部 2×12.95 + h_res/h_post + h_out 2×32） | `mhc_pre_sinkhorn.cc:24-50` / `custom_op_impl.py:331,390` | 346.5 | 346.5 | 346.5 |
| 两个 RMSNorm 保留的 aggregated `[S,B,H]` bf16 | `transformer_layer.py:311,329` + `layer_norm.py:151-155` | 64 | 64 | 64 |
| FFN / MoE | 见下 | 544（dense, F=16384） | 416.9 | 416.9 |
| **合计（源）** | | **~2085** | **~2157** | **~2027** |
| **真机实测** | 167/2026-07-29 | **2235.1** | **2341.2** | **2116.1** |
| 源/真机 | | 0.93 | 0.92 | 0.96 |

MoE 侧源枚举（`pynative/transformers/moe/`，逐条 `file:line`）：router 的
`cast(x, moe_router_dtype=fp32)` 副本 **64 MiB**（`router.py:360` 产、`:364` 的 matmul 保留）、
softmax 输出 `scores` 0.125（`router.py:276`）、重排后进 grouped-GEMM 的 token 块 64.1
（`expert_parallel.py:223` 产、`experts.py:221` 保留）、`fc1_output` 64.1（`experts.py:229-231`）、
`act_out` 32.06 + `intermediate_parallel` 32.06（`experts.py:231,235`）、combine 前的
`routed_output` 64.1（`expert_parallel.py:535` 的 `mul` 因 `probs` 带梯度而**两个操作数都存**）、
共享专家 96.0（`mlp.py:121,139-141,145`）。dense 层（`first_k_dense_replace: 1`）
= `32 + F_d/32` MiB，站点 `ffn_hidden_size = 4H = 16384` → 544。
（专家侧 64.1 假定路由**均衡**：真实 `N = sum(output_splits)` 与数据相关，见
`expert_parallel.py:288-291`；combine 侧的 64.1 只依赖 shape，是精确的。）

**源侧合计比实测低 4–8%** —— 缺口方向一致、量级小，主要落在 kernel workspace 与
compressor/indexer 内部（源里能读出的部分已计入，读不出的是 kernel 内部）。
**结论：按源逐张量枚举能把真机每层驻留解释到 92–96%；剩下的 4–8% 需要真机 profiler 的
kernel workspace 明细才能闭合** —— 这就是「还差什么证据」的答案。

---

## 4. 锚点重钉台账（old → new，逐条理由）

**未动**：`REAL` / `REAL_SHA256` / `REAL_OFF` / `REAL_PP8` / `REAL_MTP` 等一切真机常数；
两条真机不变量与模型 ×1 不变量；run d 不可评分规则。

| 位置 | old → new | 理由 |
|---|---|---|
| `tests/test_acceptance_gate.py::GOLDEN_BUCKET` | 8×4 表整体下移（如 `c` s1 29293.3→23638.8） | 普查按源订正，saves 降 ~901 MiB/层 |
| `…::GOLDEN_LIVENESS_{DATAFLOW,CHAIN2}` | 同上 | 同一张图喂 liveness |
| `…::GOLDEN_AGG` | bucket 0.946/0.748/1.394 → 0.867/0.717/1.125；hand_spec·chain2 0.955/0.729/1.400 → 0.899/0.694/1.143 | max 下降=过读被修；mean 下降=抵消消失 |
| `…::test_delta_magnitude_gap_is_recorded` | 0.603/0.670 → 0.545/0.644 | 只修 fused 侧 → delta 更欠，如实记 |
| `…::test_unfused_on_cell_ratios…` | [0.821,0.832,0.837,0.848] → [0.808,0.818,0.822,0.835]；bucket 0.748–0.771 → 0.717–0.742 | 同上 |
| `tests/test_pp4_recompute_anchor.py::THEO_ON` | s0 18172.2→16766.2 / s1 13049.0→11643.0 / s2 12975.9→11565.9 / s3 不变 | `remat_saves` 同源 |
| `…::THEO_MTP` | s0-s2 同上；s3 29276.5→28375.0 | 同上 |
| `…::THEO_PP8` | s0 22514.6→21108.6 … s6 13217.6→11811.6；s7 不变 | 同上 |
| `…::BAND_OFF` | (0.95,1.34)/(0.95,1.39)/(0.95,1.25)/(0.90,1.05) → (1.00,1.10)/(1.02,1.12)/(0.93,1.03)/(0.85,0.93) | 前三 stage 由 1.3× 过读**收敛到 ~1.0**（目标达成）；s3 落到 0.885，**OOM-不安全，如实记带** |
| `…::_PP8_OVER_BAND` | (1.10,1.35) → (1.00,1.25) | 过读幅度被压掉大半（s1 1.16→1.044、s2 1.18→1.062、s3 1.14→1.023、s5 1.21→1.082），方向仍是过读 |
| `tests/test_probe185_recon.py::test_fused_per_layer_increment` | 断言「±5% 命中 3109」→ 记录值 2702.5 + 宽方向带 | **两个真机数不自洽**：3109 是 `(L8−L4)` **全过程峰值差** @seq2048，2239.1 是**逐层直测** @seq4096（seq 翻倍反而更小）；差分法「两侧峰在可比事件」的前提已被拆解报告 §5.5 实测证伪（g/h 对翻车）。以直测为准，本锚降级为记录门 |
| `…::test_p3p_m8_theoretical_and_gap` | 18783.7 → 17370.2 | 同 `remat_saves`；仍 < 真机 25343.5（不变量方向不变） |
| `scorecard_anchors.py` DSv4-fused (base) | band (0.97,1.05) → (0.88,0.94)，ratio 1.024→0.908 | 同 185 差分锚的理由；**转欠侧 = OOM-不安全，如实记，不调参掩盖** |
| `scorecard_anchors.py` DSv4 mHC(x4)+MTP | band (0.92,0.99) → (0.83,0.90)，ratio 0.968→0.862 | 同上；仍排除历史翻转值 1.088 |
| `tests/test_from_mindformers.py::test_dsv4align_roundtrip_same_peak` | 15788.6 → 13994.8 | 同上 |
| `tests/test_dsv4_hybrid.py` fp32 saves 测试 | 断言 fp32 → 断言 **bf16**（其余不变量全保留） | §1.1/§1.3 |
| `tests/test_source_truth_saves_audit.py::KNOWN_GAPS` | `{sparse_indices, sinks}` → **`{}`**（两条按棘轮规则删除，改为正向「11 项逐名在场」+ 防回退断言） | §1.7/§1.9 |

`python -m pytest tests -q` → **1892 passed**（与改动前同数：无新增/删除用例，只移动样例值）。
