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

## 2. 已落地的修改与字节影响

（本节随提交滚动追加。）

## 3. 未归因残差

（本节随提交滚动追加。）
