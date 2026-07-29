# r4 层驻留缺口：indexer 链的三块 64.0005 MiB（2026-07-30）

> 上一轮（`docs/kernel_workspace_2026-07-29.md` §5.2 / §8 ①）用 167 真机 memory-tracker 的
> **块台账**把 r4 层的驻留欠读定位到了 indexer 链，但没落地。本轮把它按**源真值**逐张量认领并
> 建进普查。**本轮不产生任何新的真机测量**——用的是上一轮已存档的块台账。

复现：

```bash
PYTHONIOENCODING=utf-8 python scratchpad/census_threeway.py                    # 逐层型逐张量普查
PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas   # 八跑门
PYTHONIOENCODING=utf-8 python -m pytest tests -q
```

权威源码快照：`E:\97-codes\torch_parallel\mf-src-167\`（`mindformers/` + `hyper_parallel/`，
即 167 真机实跑那一份）。**不得**用 `E:\97-codes\torch_parallel\mindformers`（另一个 commit）。

---

## 0. 一句话结论

r4 层比 r128 层多出的**三块 64.0005 MiB** 是 indexer 的 **Q 链**上的三张 saved 张量：

| # | 张量 | 源位置 | 谁在持有它（保留机制） | 站点字节 | 普查此前 |
|---|---|---|---|---:|---|
| ① | `query_index`（permute 后的 indexer Q，BSND） | `indexer.py:199` → `csa.py:694` | 融合算子 `FusedSparseFlashMlaWithIndexerLoss` 的 `ctx.save_for_backward`（`csa.py:224-235`，第 5 项 `query_index` @ `:229`） | 64.000 MiB | **已建**（`idx_query`） |
| ② | indexer 内部 RoPE 的 `t`（fp32） | `indexer.py:182-187` → `rope_utils.py:169,187` | `mul(t, cos_)` 的 bprop（`rope_utils.py:187`） | 64.000 MiB | **缺** |
| ③ | indexer 内部 RoPE 的 `t_rot`（fp32） | 同上 → `rope_utils.py:186-187` | `mul(t_rot, sin_)` 的 bprop（`rope_utils.py:187`） | 64.000 MiB | **缺** |

补 ②③ = **+128.000 MiB / r4 层**。r0 / r128 **逐字节不动**（`enable_indexer` 只在 ratio==4 为真）。

---

## 1. 逐块认领（源真值 → 为什么它活到 fwd_exit）

### 1.1 测量给的是什么（上一轮，未重测）

`docs/kernel_workspace_2026-07-29.md` §5.2，run `c`（fused / 无重算 / L8 / m4 / seq4096，
b=1、tp=cp=1），**同 rank 内**比 r128（rank2 layer2）与 r4（rank2 layer3）——排除 stage/深度差异：

| 量 | r128 | r4 | 差 |
|---|---:|---:|---:|
| 实测驻留（中位） | 2124.9 | 2354.2 | **+229.3** |
| **64.0005 MiB 块 / 窗口** | **4** | **7** | **+3** |
| 手写普查 `activation_saves` | 2005.6 | 2080.6 | +75.0 |

「块」的判据是 tracker 的块台账：**在该层 fwd 窗口内诞生、且在 fwd_exit 那一刻仍然活着**。
64.0005 MiB = 64 MiB 净载荷 + 512 B 池块对齐（`_align_up`，`structure_mem.py:83-91` 同口径）。

### 1.2 站点尺寸下，「64 MiB 一块」是什么

站点（`.claude/skills/real-machine-memory-sim/prep_dsv4align.py:71-113`）：
`S=4096`、`B=1`、`num_attention_heads=64`、`qk_rope_head_dim=64`、`v_head_dim=512`、
`dsa_indexer_n_heads=64`、`dsa_indexer_head_dim=128`、`rotary_dtype="float32"`。

- `[S,B,n_heads·qk_rope_head_dim]` fp32 = 4096·1·(64·64)·4 B = **64.000 MiB**
- `[B,S,index_n_heads,index_head_dim]` bf16 = 4096·64·128·2 B = **64.000 MiB**
- `[S,B,index_n_heads,qk_pos_emb_head_dim]` fp32 = 4096·1·(64·64)·4 B = **64.000 MiB**

三者在这个站点上**恰好同尺寸**，块台账区分不了它们——所以下面按**源**认，不按尺寸猜。

### 1.3 r128 那 4 块是什么（对照组，先立标尺）

r128（和 r0、r4）共有的顶层 attention 走**三次** `ApplyRotaryPosEmb`
（`deepseek_v4_hybrid_attention.py` 的 q 前向 / key 前向 / core_out 逆向），
每次落到 `rope_utils.py:186-187`：

```python
t_rot = self._rotate_half(t, rotary_interleaved)
output = self.add(self.mul(t, cos_), self.mul(t_rot, sin_))
```

两个 `mul` 各保留一个操作数 → 每次留下 `t` 与 `t_rot`。`t` 在 `:169` 被
`cast(t, self.rotary_dtype)` 抬成 **fp32**。三次调用的 lane：

| 调用 | lane | fp32 字节 |
|---|---|---:|
| q 前向 | `n_heads·qk_rope_head_dim` = 4096 | 64.000 MiB × 2 |
| key 前向 | `qk_rope_head_dim` = 64（单共享头） | 1.000 MiB × 2 |
| core_out 逆向 | `n_heads·qk_rope_head_dim` = 4096 | 64.000 MiB × 2 |

→ **4 块 64 MiB**（`q_rope_f32` / `q_rope_rot` / `inv_rope_f32` / `inv_rope_rot`，普查已建），
与 r128 实测的 4 块**块数相等**。这条标尺很重要：它说明「rope 一次调用留两块 lane-fp32」
这条机制**已经被真机块台账在 r128 上独立验过一遍**，本轮只是把同一条机制套到 indexer 的
那次调用上。

> **判据订正（重要）**：`dsv4_hybrid.py` 原注释把「走非融合分支」归因为
> 「`mla_output_remove_interleaving=True` 使 `fused_interleaved_mla` 恒 False」。这条**推不出**
> 结论——`rope_utils.py:182` 的条件是 `if self.apply_rope_fusion and not inverse:`，里面**没有**
> `fused_interleaved_mla`。真正的判据是 **`apply_rope_fusion` 默认 `False`**
> （`parallel_core/transformer_config.py:1578`，站点 yaml 未设 `apply_rope_fusion` / `use_fused_rope`）。
> 结论（三次都走 `:186-187`）不变，判据已在代码注释里就地补正。

### 1.4 ① `query_index` —— 唯一有显式 `save_for_backward` 的那块

`csa.py:660-707` 的融合路径：

- `:665-666` `x_detach = ops.stop_gradient(x)` / `qr_detach = ops.stop_gradient(qr)`
- `:667` `query_index, key_index, weights = self.indexer.forward_before_topk(x_detach, qr_detach)`
- `:694` `ops.cast(query_index, mstype.bfloat16)` 作为第 5 个实参进
  `FusedSparseFlashMlaWithIndexerLoss.apply`
- `:224-235` `ctx.save_for_backward(*[tensor for tensor in (query, ori_kv, cmp_kv,
  cmp_sparse_indices, query_index, key_index, weights, cmp_residual, sinks, output,
  softmax_lse) if tensor is not None])` —— `query_index` 在 `:229`

`query_index` 的形状是 `indexer.py:199` `q = self.permute(q, (1, 0, 2, 3))` 之后的
`[B,S,index_n_heads,index_head_dim]` bf16 = 64.000 MiB。**普查此前已建**（`idx_query`，
`dsv4_hybrid.py` 的 `sparse_saves`）。

### 1.5 ②③ indexer 内部 RoPE 的保留对 —— 本轮新建的两块

`indexer.py:158-207` `CSAIndexer.forward_before_topk` 的 Q 路：

```
:177  q = self.linear_wq_b(qr)                    # [S,B, n_idx*d_idx] bf16
:178  q = self.reshape(q, (S, B, n_idx, d_idx))
:179  q_nope, q_pe = self.split(q, [d_idx - qk_pos_emb_head_dim, qk_pos_emb_head_dim], dim=-1)
:182  q_pe = self.apply_rope(q_pe, freqs, 1, ..., multi_latent_attention=True,
                             mla_output_remove_interleaving=True)
:188  q = self.cat([q_nope, q_pe], dim=-1)
:189  q = self.hadamard(q)
:199  q = self.permute(q, (1, 0, 2, 3))           # → query_index
```

- **形状**：`index_head_dim=128`、`qk_pos_emb_head_dim=64` → `q_pe = [S,B,64,64]`。
  进 `rope_utils.construct` 后 `rot_dim = freqs.shape[-1] = 64 == head_dim` →
  `:152-153` 走 `t_not_rotary = None` 分支，**整块**参与旋转。
- **dtype**：`:169` `t = self.cast(t, self.rotary_dtype)` → fp32（站点
  `rotary_dtype: "float32"`；`configuration_deepseek_v4.py:149` 与
  `transformer_config.py:166-167` 亦默认 fp32）。
- **保留**：`:187` 两个 `mul` → `t`、`t_rot` 各一块
  `[S,B,index_n_heads,qk_pos_emb_head_dim]` fp32 = **各 64.000 MiB**。
- **梯度确实到达**（否则不是 saved 张量、是纯瞬态）：入参虽被 `stop_gradient`
  截断（`csa.py:665-666`），但 `linear_wq_b` 是 Parameter（`indexer.py:116-124`）→
  `q` 起就带梯度；且融合算子的 `backward` **真的吐** `d_query_index`（`csa.py:314`），
  经 `:694` 的 cast 回流到整条 Q 链。这与 `ukl1`/`ukl2` 的判决（`csa.py:794-795`
  两个输入都 detach、链上无任何参数 → 从不 saved）是同一把尺子的两侧。

### 1.6 为什么**只有**这两块，而不是 Q 链上另外四张 64 MiB

Q 链在站点尺寸下共有 **7** 张 64 MiB 量级的张量。逐张判：

| 候选 | 源位置 | dtype | 谁会持有它 | 判 |
|---|---|---|---|---|
| `linear_wq_b` 输出 | `indexer.py:177` | bf16 | 下游是 `reshape`(`:178`) 与 `split`(`:179`)：VJP 分别是 reshape / concat，**都不读前向输入** | 不保留（且 `q_nope`/`q_pe` 在 `:188` 之后即死，view 语义下也不延寿） |
| `q_pe` 的 rope `t` | `rope_utils.py:169` | **fp32** | `mul(t, cos_)` bprop | **保留** ② |
| `q_pe` 的 rope `t_rot` | `rope_utils.py:186` | **fp32** | `mul(t_rot, sin_)` bprop | **保留** ③ |
| `cat` 输出 | `indexer.py:188` | bf16 | 下游 `Hadamard`：`mint.nn.functional.linear(x, H)`，`H = self.hadamard_mat.astype(x.dtype)` 是**普通 Tensor 常量**（`utils.py:138-140`，非 Parameter）→ 不需要 `dH` → 不需要保留 `x` | 不保留 |
| `Hadamard` linear 输出 | `utils.py:140` | bf16 | 下游 `x * self.scale`（`:141`，Python 标量）→ `dx = dout·scale`，不读 `x` | 不保留 |
| `Hadamard` scale 输出 | `utils.py:141` | bf16 | 下游 `permute`（`:199`），VJP 是逆 permute，不读输入 | 不保留 |
| `permute` 输出 = `query_index` | `indexer.py:199` | bf16 | `ctx.save_for_backward`（`csa.py:229`） | **保留** ① |

判「不保留」用的是**本普查自己一贯的规则**，不是新规则：矩阵乘只为**权重梯度**保留激活操作数
（如 `linear_q_up` 的 `saves=[q_a_out]`），而 Hadamard 的「权重」是常量、没有权重梯度；
reshape / split / concat / permute 的 VJP 都是纯下标重排，不读前向输入。

**结果：源侧数出来正好 3 张活到 fwd_exit，与实测的 3 块逐块相符。** 这是本轮唯一的
数量级交叉验证，它不是拟合出来的——候选集是源码枚举的，保留判据是逐条独立的。

### 1.7 `enable_indexer` 之外的层型确实没有这条链

`csa.py:608` `if compress_ratio == 4 and not config.csa_dense_mode and submodules.indexer is not None:`
→ `self.enable_indexer = True`，否则 `:623-626` 全置 `None`。r0 / r128 **没有 `CSAIndexer` 对象**，
因此没有 `forward_before_topk`、没有那次 rope 调用。

模型侧已按输出确认（不是假设）：见 §3 的逐层型表，r0 / r128 的 `activation_saves`
**逐字节不变**；`tests/test_indexer_rope_saves.py` 另有一道门直接断言 r0/r128（fused 与
unfused 四种组合）的 op 图里**不存在**任何 `idx_rope_*` 张量。

### 1.8 fused 与 unfused 两侧都建

`forward_before_topk` 在两条路径上被**逐字**调用：`csa.py:667`（`_construct_fused`）与
`csa.py:766`（`_construct_naive`）。rope 调用在它内部，与 `apply_dsa_kernel_fusion` 无关
——该开关只切 `CSAIndexer.construct` 里的**打分实现**（`indexer.py:213` 融合 kernel vs
`:233` 小算子 bmm）。故 ②③ 在 fused / unfused **两侧都声明**。

> 对照：`_FLASHMLA_IDX_BWD_WS`（反向 kernel workspace）是**双重门** `fused and enable_indexer`，
> 因为那是**实测**且只在那一个 kernel 上；本项是**源结构**且两侧同在，判据不同。

---

## 2. 改了什么

| 文件 | 改动 |
|---|---|
| `cost_eval/layers/dsv4_hybrid.py` | `indexer` op 的 `saves` 追加 `idx_rope_f32` / `idx_rope_rot`（`("S","B","dsa_indexer_n_heads*qk_rope_head_dim")`，`dtype_bytes=4`），fused / unfused 两分支；就地补正「走非融合 rope 分支」的判据 |
| `tests/test_indexer_rope_saves.py` | **新增守卫门**（见 §5） |

`REAL_*` / `CSV_*` / `REAL_SHA256` 指纹表 **一个字节没动**（验证见 §6）。

---

## 3. 逐层型 before → after（模型 vs 真机）

真机为 167 / 2026-07-29 存档实测（**未重测**）：r0 = 2235.1、r4 = 2341.2（4 层均）、
r128 = 2116.1（3 层均）MiB/层。

| 层型 | 模型 before | 模型 after | 真机 | before 比 | **after 比** |
|---|---:|---:|---:|---:|---:|
| r0 `dsv4hyb_r0_dense` | 2229.5 | **2229.5** | 2235.1 | 0.998 | **0.998** |
| r4 `dsv4hyb_r4_moe` | 2080.6 | **2208.6** | 2341.2 | 0.889 | **0.943** |
| r128 `dsv4hyb_r128_moe` | 2005.6 | **2005.6** | 2116.1 | 0.948 | **0.948** |

r4 由 **0.889 → 0.943**，与 r128 的 0.948 落到同一档（r4 相对 r128 的**层型专属**欠读被闭掉）。

**块台账口径的对账**（§1.1 的同 rank 比较）：

| 量 | before | after | 实测 |
|---|---:|---:|---:|
| 普查 r4 − r128 | +75.0 | **+203.0** | **+229.3** |
| 模型里的 64 MiB 张量数 r4 − r128 | 6 − 5 = 1 | **8 − 5 = 3** | **7 − 4 = 3** |

块**数差**逐块对上；**字节差**仍差 26.3 MiB（见 §7）。

---

## 4. 八跑门 before → after

`PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas`

### 4.1 聚合

| 来源 | before | after |
|---|---|---|
| `bucket`（必须绿） | n=28 mean=**0.836** min=**0.700** max=**1.023** | n=28 mean=**0.845** min=**0.706** max=**1.036** |
| `hand_spec`（必须绿） | n=28 mean=**0.855** min=**0.670** max=**1.110** | n=28 mean=**0.864** min=**0.674** max=**1.124** |
| `extracted`（advisory） | 8 跑 ERR（`IncompleteExtraction @indexer.py:219`） | 同（不受影响） |

run `d` 四格恒不可评分（真机 OOM），before/after 同。

### 4.2 逐格（`bucket/real`）

| run | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| a fused ON L8 m4 | 0.715 → **0.715** | 0.863 → **0.881** | 0.878 → **0.896** | 0.945 → **0.945** |
| b unfused ON L8 m4 | 0.705 → **0.713** | 0.700 → **0.706** | 0.703 → **0.709** | 0.727 → **0.735** |
| c fused OFF L8 m4 | 0.929 → **0.937** | 0.953 → **0.965** | 0.898 → **0.898** | 0.878 → **0.887** |
| e fused ON L8 m8 | 0.731 → **0.731** | 0.863 → **0.881** | 0.878 → **0.896** | 0.945 → **0.945** |
| f unfused ON L8 m8 | 0.714 → **0.721** | 0.700 → **0.706** | 0.703 → **0.709** | 0.727 → **0.735** |
| g fused ON L4 m4 | 0.784 → **0.784** | 1.023 → **1.036** | 1.012 → **1.012** | 0.964 → **0.964** |
| h unfused ON L4 m4 | 0.979 → **0.979** | 0.744 → **0.752** | 1.017 → **1.017** | 0.716 → **0.725** |

**方向一致**：所有动了的格都**上移**（欠读被真实补上一块），没有一格从欠读翻成过读；
`bucket` 的 max 由 0.723 (g@s1 1.023) 抬到 1.036，仍在既有量级。

### 4.3 不变量

| 不变量 | before | after |
|---|---|---|
| REAL 指纹 `sha256=41e279e591ae4ae9…` | PASS | **PASS** |
| I1 真机 ×1 于层数 | PASS | **PASS** |
| I2 真机 ×1 于微批数 | PASS | **PASS** |
| I1 / I2 `bucket` ×1 | PASS | **PASS** |
| I1 / I2 `hand_spec` ×1 | PASS | **PASS** |
| I1 / I2 `extracted` ×1 | SKIP（图不完备） | **SKIP**（同因） |

---

## 5. 新增的守卫门（`tests/test_indexer_rope_saves.py`）

守的是**认领**（源结构 + 实测块数），不是模型自洽：

1. 站点尺寸下 `idx_rope_f32` / `idx_rope_rot` **各恰好 64.000 MiB**、`dtype_bytes == 4`
   —— 数值来自 `resolve_tensor`，抓「有人把 lane 写错 / 把 fp32 写回 bf16」。
2. lane 表达式恒等于 `dsa_indexer_n_heads*qk_rope_head_dim`（≠ 顶层的 `n_heads*qk_rope_head_dim`）
   —— 两者在本站点碰巧同值，故用**表达式**钉，防「站点巧合」被固化。
3. fused **与** unfused 两侧都在场（`forward_before_topk` 两条路径共用，§1.8）。
4. **r0 / r128 恒无**（fused × unfused 四种组合），且它们的 `activation_saves`
   与 before 逐字节相等 —— 挡「拿 r4 的东西去顶别的层型」。
5. 名字不与顶层 rope 的六张撞 —— `structure_mem` 按名去重，撞名会把新增静默吞成 0。
6. **实测块数门**：模型里 r4 比 r128 多出的「恰好 64.000 MiB」张量**数**必须 == **3**
   （真机块台账，`docs/kernel_workspace_2026-07-29.md` §5.2）。这道门守的是**测量**：
   再往 r4 的 64 MiB 档里塞一张、或把 ②③ 删掉，都会变红。
7. `activation_saves` 净增**恰好 128.000 MiB / r4 层**（去重后），r0/r128 净增 0。

---

## 6. 「没动真机常数」的验证

```bash
git diff <base> HEAD -U0 -- . ':(exclude)docs' ':(exclude)scratchpad' \
  | grep -nE '^[-+].*(REAL|CSV|MEASURED|sha256)'
```

结果见 §9 的执行记录：**只有** `tests/test_pp4_recompute_anchor.py` / `tests/test_scorecard_anchors.py`
等文件里**注释行**提到 `REAL_*` 名字（说明重钉理由），没有任何 `REAL_*` / `CSV_*` 常量本体、
也没有 `REAL_SHA256` 指纹被修改。八跑门里的 `REAL 指纹` 断言 PASS 是第二重独立验证。

---

## 7. 诚实清单：剩下的没闭合的

| # | 缺口 | 量 | 为什么没闭合 / 闭合它需要什么 |
|---|---|---:|---|
| ① | **r4 相对 r128 的残差** | **26.3 MiB/层**（块口径：229.3 实测 − 203.0 普查）；层型均值口径 132.6 − 110.5 = **22.1 MiB/层** | 实测形态是「`<1 MiB` 长尾 +2.3 块/窗口」+ 未列进 §5.2 直方图的中间档。**源侧最可能的落点是 indexer 自带的那个 `Compressor`**（`indexer.py:125-132`，`head_dim=128`/`ratio=4`/`rotate=True`，r4 独有）：`compressor.py:216` `(kv.astype(fp32) * weights).sum(dim=1)` 的 `mul` 两个操作数都带梯度 → 各留一块 `[S/ratio, coff·ratio, B, head_dim]` fp32（站点 ≈4 MiB × 2），加 `softmax` 输出（`:215`，≈4 MiB）。**但这是推断，不是测量**：块台账没有把这一档拆开，本轮**不建**。闭合需要按块尺寸重跑 §5.2 的直方图并把 1–16 MiB 档也列出来。 |
| ② | **r128 / r0 共有的稀疏路欠读** | r128 −110.5、r0 −5.6 MiB/层 | 与 r4 无关（r4 也吃这一份）。同 ① 的 `compressor` 机理：普查的 `compressor` op 只 `saves=[compressed_kv]`，而源 `compressor.py:203-218` 的 `softmax` 输出与 `kv` 的 fp32 副本都被 `mul` 的 bprop 持有（r128 各 ≈8 MiB、r4 各 ≈16 MiB）。**未测、未建**，理由同 ①：改它会同时移动 r0/r4/r128 三个层型与全部锚点，属于要单独决策的口径变更。 |
| ③ | **模型里 5 张「共有的 64 MiB」 vs 实测 r128 只有 4 块** | 1 块 = 64 MiB | 模型对**所有** DSv4 层型都声明 5 张 64 MiB 张量（`q_rope_f32`/`q_rope_rot`/`inv_rope_f32`/`inv_rope_rot`/`o_group_out`），实测 r128 侧只有 **4** 块。**块数差 3 是对的、绝对块数差 1**。这条差异是**共有部分**的、不是 r4 专属，且方向是「模型多一张」；本轮**不动**——动它会同时移动 r0/r4/r128。闭合需要 tracker 侧把这 4 块的诞生 op 逐块打出来（本轮的存档只给了直方图计数）。 |
| ④ | **`indexer` 内 `Compressor` 的 rope 对**（`compressor.py:237-242`，r4 独有一份、主压缩器另有一份） | 站点各 ≈0.25 MiB | 同 §1.5 的机理，但落在 `<1 MiB`–MiB 档，块台账区分不了。**未建**，因为「按机理推、无块级证据」正是本轮想避免的那类补法。 |
| ⑤ | **`idx_key` 形状可能过读** | ≤0.75 MiB/层 | 普查按 `[B,S,dsa_indexer_head_dim]` 建（1.000 MiB）；源侧 `key_index = self.compressor(x)`（`indexer.py:190`）经 `compressor.py:225` 返回 `[S//ratio, B, 1, head_dim]` → permute 后应为 `[B, S//4, 1, 128]` = 0.250 MiB。**未改**：它进的是 `sparse_attn.saves`，受 `tests/test_source_truth_saves_audit.py` 的 11 项名单门管辖，改形状要单独决策；且 0.75 MiB 远在本轮块口径的分辨率之下。 |
| ⑥ | **OOM-不安全锚点未被改善** | — | 最大已知成因（`lm_head`/loss 反向 kernel workspace 未测、留 0）与本轮无关，**刻意不在此处补偿**。 |

**`tests/test_source_truth_saves_audit.py::KNOWN_GAPS` 保持为空**，这是正确的：该台账的键**必须**是
`csa.py:224` `ctx.save_for_backward` 名单里的源名（`test_known_gaps_ledger_is_still_accurate`
第 164 行断言 `src_name in source_saved_names`），而上面 ①–⑤ 全都**不是** `sparse_attn` 的 ctx 项
（它们在 `indexer` / `compressor` 上）。往那个台账里塞会让它自己的棘轮红。故如实登记在本节。

---

## 8. 重钉台账（old → new，逐条理由）

**未动**：`REAL*` / `CSV*` 一切真机常数、`REAL_SHA256` 指纹表、两条真机不变量与模型 ×1
不变量、run `d` 不可评分规则、`nr_moe_frag_factor` / `kept_frag_factor` 两个标定 margin。

（逐条见 §9 的执行记录表。）

---

## 9. 执行记录

（见文末，按执行顺序追加。）
