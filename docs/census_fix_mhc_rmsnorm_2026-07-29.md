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
