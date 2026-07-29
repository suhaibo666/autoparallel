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

> **注意「三块」的口径**：三块**都有** `file:line` 出处，但**本轮只新建 ②③**——① 在本轮之前
> 就已经建好了（模型名 `idx_query`，before 树的普查输出里就有）。真正**没有**出处的是另一件
> 反方向的事：模型对每个 DSv4 层型都声明 **5** 张 64 MiB，而块台账在 r128 只看到 **4** 块
> → 绝对块数 **模型 8/5 vs 实测 7/4**，**差对上了、绝对数各多 1**。那一块在**共有**部分、
> 与 r4 无关，**未闭合**、不编造。详见 §7 ③ 与其上方的口径澄清框。

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
| `tests/test_indexer_rope_saves.py` | **新增守卫门**（见 §5，22 条） |
| `tests/test_acceptance_gate.py` | 八跑 golden 三表 + 聚合 + delta 幅值 + unfused 参照比值重钉（§8 #1–#5） |
| `tests/test_pp4_recompute_anchor.py` | `THEO_ON` / `THEO_MTP` / `THEO_PP8` / `_PP8_OVER_BAND` 重钉（§8 #6–#9） |
| `tests/test_probe185_recon.py` | F 相位差分 + F0 带记录 + P3-P s0 重钉，U1/U2 读数记录（§8 #10–#11） |
| `tests/test_from_mindformers.py` | dsv4-align round-trip 峰重钉（§8 #12） |
| `scratchpad/probe_idx_rope_repin.py` | 只读探针：逐锚点 peak + **peak_event**（用来解释非整数幅度） |

`REAL_*` / `CSV_*` / `REAL_SHA256` 指纹表 **一个字节没动**（验证见 §6）。

---

## 3. 逐层型 before → after（模型 vs 真机）

真机为 167 / 2026-07-29 存档实测（**未重测**）：r0 = 2235.1、r4 = 2341.2（4 层均）、
r128 = 2116.1（3 层均）MiB/层。

> **✅ 本节已实跑核实**（2026-07-30 收口轮，`scratchpad/census_threeway.py` 在 before/after
> 两棵树各跑一遍）。下表的 after 列是**观测值**，不是估算。

| 层型 | 模型 before | 模型 after | 真机 | before 比 | **after 比** | after 残差 |
|---|---:|---:|---:|---:|---:|---:|
| r0 `dsv4hyb_r0_dense` | 2229.5 | **2229.5** | 2235.1 | 0.9975 | **0.9975** | −5.6 |
| r4 `dsv4hyb_r4_moe` | 2080.6 | **2208.6** | 2341.2 | 0.8887 | **0.9434** | **−132.6** |
| r128 `dsv4hyb_r128_moe` | 2005.6 | **2005.6** | 2116.1 | 0.9478 | **0.9478** | −110.5 |

r4 由 **0.889 → 0.943**，与 r128 的 0.948 落到同一档（r4 相对 r128 的**层型专属**欠读被闭掉）。
r0/r128 **逐字节不动**，这是**从模型输出核实的**（不是从 `enable_indexer` 推的）：三个层型的
`activation_saves` 与 4 个 r4 层 / 3 个 r128 层的逐层值在 before/after 两次运行里逐行对比，
r0 与全部 r128 层的数字一字不差。`tests/test_indexer_rope_saves.py` 另有一道门直接断言
r0/r128 的 op 图里不存在任何 `idx_rope_*`（fused × unfused 四种组合），并把三个层型的
注意力段去重 saves 绝对值钉死。

**r4 剩下的 132.6 MiB/层**里，只有 **22.1** 是 r4 专属的（132.6 − 110.5）；另外 110.5 是
r0/r4/r128 **共有**的欠读（见 §7 ②），不属于本轮的 r4 层型问题。

**块台账口径的对账**（§1.1 的同 rank 比较）：

| 量 | before | after | 实测 |
|---|---:|---:|---:|
| 普查 r4 − r128 | +75.0 | **+203.0** | **+229.3** |
| 模型里的 64 MiB 张量数 r4 − r128 | 6 − 5 = 1 | **8 − 5 = 3** | **7 − 4 = 3** |

块**数差**逐块对上；**字节差**仍差 26.3 MiB（见 §7）。

---

## 4. 八跑门 before → after

`PYTHONIOENCODING=utf-8 python tools/liveness_ab_validate.py --grad-mode chain2 --deltas`

> **⚠ 订正（2026-07-30 收口轮）**：本节原先登记的 after 列是**没有实跑的手工估算**
> （按「每 stage 2 个 r4 层各 +64」推的），与实跑不符。下表已全部替换为**实跑观测值**
> （before 与 after 各跑一次 `tools/liveness_ab_validate.py --grad-mode chain2 --deltas`）。
> 保留这条订正而不是抹掉旧数，是因为「估算被实跑推翻」本身是要记的事实。

### 4.1 聚合（实跑）

| 来源 | before | after |
|---|---|---|
| `bucket`（必须绿） | n=28 mean=**0.836** min=**0.700** max=**1.023** | n=28 mean=**0.841** min=**0.703** max=**1.038** |
| `hand_spec`（必须绿，chain2） | n=28 mean=**0.855** min=**0.670** max=**1.110** | n=28 mean=**0.860** min=**0.675** max=**1.110** |
| `hand_spec`（dataflow） | n=28 mean=**0.795** min=**0.659** max=**1.067** | n=28 mean=**0.799** min=**0.662** max=**1.067** |
| `extracted`（advisory） | 8 跑 ERR（`IncompleteExtraction @indexer.py:219`） | 同（不受影响） |

run `d` 四格恒不可评分（真机 OOM），before/after 同。

⚠ 与上一轮（bwd kernel workspace）**不同**：那一项只走桶模型的 `bwd_workspace_bytes` 通道，
`hand_spec` 当时逐字节不动；**本项改的是 spec 自己的 `saves`**，`cost_eval/liveness/` 按 saves
建图 → **两条来源同时移动**。

### 4.2 逐格（`bucket/real`，实跑）

| run | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| a fused ON L8 m4 | 0.715 → **0.720** | 0.863 → **0.871** | 0.878 → **0.887** | 0.945 → **0.945** |
| b unfused ON L8 m4 | 0.705 → **0.708** | 0.700 → **0.703** | 0.703 → **0.706** | 0.727 → **0.730** |
| c fused OFF L8 m4 | 0.929 → **0.946** | 0.953 → **0.972** | 0.898 → **0.913** | 0.878 → **0.883** |
| e fused ON L8 m8 | 0.731 → **0.736** | 0.863 → **0.872** | 0.878 → **0.887** | 0.945 → **0.945** |
| f unfused ON L8 m8 | 0.714 → **0.716** | 0.700 → **0.703** | 0.703 → **0.706** | 0.727 → **0.730** |
| g fused ON L4 m4 | 0.784 → **0.784** | 1.023 → **1.038** | 1.012 → **1.012** | 0.964 → **0.964** |
| h unfused ON L4 m4 | 0.979 → **0.979** | 0.744 → **0.747** | 1.017 → **1.017** | 0.716 → **0.719** |

**方向一致**：28 个可评分格里 **0 格下移**（`bucket` 21 上移 / 7 不动；`hand_spec`·chain2
20 上移 / 8 不动），没有一格从欠读翻成过读 —— `g` s1 / `g` s2 / `h` s2 本来就在过读侧。
`bucket` 的 max 由 1.023 (g@s1) 抬到 **1.038**，仍在既有量级。

**逐 stage 幅度是可核对的整数**（MiB，桶模型）：每 stage 的上移 = 该 stage **峰值事件上在世
的 r4 层数 × 128.0**。无重算两跑（c/d）因逐微批常驻，幅度 = **在途微批深度 × 128**
（s0..s3 = +512 / +384 / +256 / +128）。唯一的非整数差是 `a` s0 的 **+100.0**，成因是
**峰值事件易主**（实测 before `bwd@2`=17250.6 < `bwd@1`=17278.6；after `bwd@2`=17378.6
越过之），r4 层那一格实涨仍是整 128.0 —— 不是本项只算了一部分。

### 4.3 不变量

| 不变量 | before | after |
|---|---|---|
| REAL 指纹 `sha256=41e279e591ae4ae9…` | PASS | **PASS** |
| I1 真机 ×1 于层数（stage3 unfused−fused：L8=24380.1 / L4=24380.1，\|diff\|=0.0） | PASS | **PASS** |
| I2 真机 ×1 于微批数（m4=24380.1 / m8=24380.0，\|diff\|=0.1） | PASS | **PASS** |
| I1 / I2 `bucket` ×1（模型 delta L8=L4=12722.4；m4=m8=12722.4） | PASS | **PASS** |
| I1 / I2 `hand_spec` ×1（模型 delta L8=L4=15549.8；m4=m8=15549.8） | PASS | **PASS** |
| I1 / I2 `extracted` ×1 | SKIP（图不完备） | **SKIP**（同因） |

**delta 幅值**（如实记录、不入断言容差）：`bucket` 12594.4 → **12722.4**（/24380.1 =
0.517 → **0.522**）、`hand_spec` 15421.8 → **15549.8**（0.633 → **0.638**）。
**方向与前三轮相反、缺口收窄** —— 本项在 `forward_before_topk` 内部，fused/unfused
两条路径共用，unfused 侧同样 +128.0，而 stage3 的 fused 侧峰在 head 段（`bwd@9`）不动。

---

## 5. 新增的守卫门（`tests/test_indexer_rope_saves.py`）

> **✅ 2026-07-30 收口轮落地，22 条全绿。** 上一轮本节只写了规格、文件没建（§1.7/§2 因此在
> 引用一个不存在的门）。**红→绿证据**：把该文件拷进 `00eab88`（未打 census 改动）的 worktree
> 里跑 → **12 failed / 10 passed**；打上改动后 → **22 passed**。通过的那 10 条正是
> 「r0/r128 恒无 + 不撞名」这类本就不该动的事实，它们在两侧都绿是**对的**。

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
git diff 00eab88 HEAD -U0 -- . ':(exclude)docs' ':(exclude)scratchpad' \
  | grep -nE '^[-+].*(REAL|CSV|MEASURED|sha256)'
```

**实跑结果：零命中**（grep 退出码 1）。本轮连注释行都没有提到 `REAL_*` 名字，比上一轮更干净。
非 doc / 非 scratchpad 的改动只有 4 个测试文件 + `cost_eval/layers/dsv4_hybrid.py`；
`REAL_ON` / `REAL_OFF` / `REAL_MTP` / `REAL_PP8` / `REAL_116` / `REAL` 八跑表 / `REAL_SHA256`
指纹表 **一个字节没动**。八跑门里的 `PASS REAL 指纹: sha256=41e279e591ae4ae9…` 是第二重
独立验证（before/after 均 PASS）。

---

## 7. 诚实清单：剩下的没闭合的

| # | 缺口 | 量 | 为什么没闭合 / 闭合它需要什么 |
|---|---|---:|---|
| ① | **r4 相对 r128 的残差** | **26.3 MiB/层**（块口径：229.3 实测 − 203.0 普查）；层型均值口径 132.6 − 110.5 = **22.1 MiB/层** | 实测形态是「`<1 MiB` 长尾 +2.3 块/窗口」+ 未列进 §5.2 直方图的中间档。**源侧最可能的落点是 indexer 自带的那个 `Compressor`**（`indexer.py:125-132`，`head_dim=128`/`ratio=4`/`rotate=True`，r4 独有）：`compressor.py:216` `(kv.astype(fp32) * weights).sum(dim=1)` 的 `mul` 两个操作数都带梯度 → 各留一块 `[S/ratio, coff·ratio, B, head_dim]` fp32（站点 ≈4 MiB × 2），加 `softmax` 输出（`:215`，≈4 MiB）。**但这是推断，不是测量**：块台账没有把这一档拆开，本轮**不建**。闭合需要按块尺寸重跑 §5.2 的直方图并把 1–16 MiB 档也列出来。 |
| ② | **r128 / r0 共有的稀疏路欠读** | r128 −110.5、r0 −5.6 MiB/层 | 与 r4 无关（r4 也吃这一份）。同 ① 的 `compressor` 机理：普查的 `compressor` op 只 `saves=[compressed_kv]`，而源 `compressor.py:203-218` 的 `softmax` 输出与 `kv` 的 fp32 副本都被 `mul` 的 bprop 持有（r128 各 ≈8 MiB、r4 各 ≈16 MiB）。**未测、未建**，理由同 ①：改它会同时移动 r0/r4/r128 三个层型与全部锚点，属于要单独决策的口径变更。 |
| ③ | **模型里 5 张「共有的 64 MiB」 vs 实测 r128 只有 4 块** | 1 块 = 64 MiB | 模型对**所有** DSv4 层型都声明 5 张 64 MiB 张量（`q_rope_f32`/`q_rope_rot`/`inv_rope_f32`/`inv_rope_rot`/`o_group_out`），实测 r128 侧只有 **4** 块。**块数差 3 是对的、绝对块数差 1**。这条差异是**共有部分**的、不是 r4 专属，且方向是「模型多一张」；本轮**不动**——动它会同时移动 r0/r4/r128。闭合需要 tracker 侧把这 4 块的诞生 op 逐块打出来（本轮的存档只给了直方图计数）。 |
| ④ | **`indexer` 内 `Compressor` 的 rope 对**（`compressor.py:237-242`，r4 独有一份、主压缩器另有一份） | 站点各 ≈0.25 MiB | 同 §1.5 的机理，但落在 `<1 MiB`–MiB 档，块台账区分不了。**未建**，因为「按机理推、无块级证据」正是本轮想避免的那类补法。 |
| ⑤ | **`idx_key` 形状可能过读** | ≤0.75 MiB/层 | 普查按 `[B,S,dsa_indexer_head_dim]` 建（1.000 MiB）；源侧 `key_index = self.compressor(x)`（`indexer.py:190`）经 `compressor.py:225` 返回 `[S//ratio, B, 1, head_dim]` → permute 后应为 `[B, S//4, 1, 128]` = 0.250 MiB。**未改**：它进的是 `sparse_attn.saves`，受 `tests/test_source_truth_saves_audit.py` 的 11 项名单门管辖，改形状要单独决策；且 0.75 MiB 远在本轮块口径的分辨率之下。 |
| ⑥ | **OOM-不安全锚点未被闭合**（缺口收窄，但一个都没翻到安全侧） | 见 §10 | 最大已知成因（`lm_head`/loss 反向 kernel workspace **未测、留 0**，是其中 9 个锚点的峰值事件所在，见 `docs/head_workspace_2026-07-30.md` §9）与本轮无关，**刻意不在此处补偿** —— 把 r4 层的 saves 加大去顶 head 段的欠读，会让两个错误互相掩盖。 |

**`tests/test_source_truth_saves_audit.py::KNOWN_GAPS` 保持为空**，这是正确的：该台账的键**必须**是
`csa.py:224` `ctx.save_for_backward` 名单里的源名（`test_known_gaps_ledger_is_still_accurate`
第 164 行断言 `src_name in source_saved_names`），而上面 ①–⑤ 全都**不是** `sparse_attn` 的 ctx 项
（它们在 `indexer` / `compressor` 上）；且 `test_hand_saves_count_matches_source_eleven` 第 176 行
逐字断言 `set(KNOWN_GAPS) == set()`（棘轮）。往那个台账里塞会让它自己变红。故如实登记在本节。

> **⚠ 关于「三块里的第三块」的口径澄清（2026-07-30 收口轮）**
>
> 有一种说法是「源侧只解释了 2 块（`t` / `t_rot`），第三块没有出处」。**这条不成立**，
> 三块**都有** `file:line` 出处，只是第 ① 块**本轮之前就已经建好了**：
> `query_index` 在 `csa.py:224` 的 `ctx.save_for_backward(...)` 名单里、第 5 项 @ `csa.py:229`；
> 梯度确实到达（`csa.py:314` 的 `d_query_index`）；模型侧名为 `idx_query`，
> **在打本轮改动之前的普查输出里就已经存在**（before 树实跑：r4 层 `64.0 MiB idx_query
> by=sparse_attn`，r128 层无此项）。`tests/test_indexer_rope_saves.py::
> test_the_third_block_is_idx_query_and_predates_this_round` 把这条认领钉住。
>
> **真正没有出处的是另一件事**（= 下表 ③，方向相反）：模型对**每一个** DSv4 层型都声明
> **5** 张 64 MiB 张量，而块台账在 r128 层只看到 **4** 块。于是绝对块数是
> **模型 8 / 5 vs 实测 7 / 4** —— **差（3）对上了，绝对数各多 1**。多出来的那一块在
> **三个层型共有**的部分，与 r4 无关，**本轮不动**（动它会同时移动 r0/r4/r128 与全部锚点）。
> **要识别它需要的证据**：让 tracker 把 r128 层 fwd 窗口里那 4 块 64.0005 MiB 的**诞生 op**
> 逐块打出来（本轮用的存档只给了直方图**计数**，给不出归属）。做法与
> `docs/kernel_workspace_2026-07-29.md` 同：`MS_ALLOC_CONF=memory_tracker:True` 重跑 run c，
> 在 r128 层的 fwd 窗口里按块地址关联 alloc 事件的 `node_name`/调用栈。
> 已把这条钉进守卫门：`test_block_count_delta_matches_the_real_machine_ledger` 除了断言
> 差 == 3，还断言绝对数 8/5 —— 一旦 ③ 被闭合（模型少一张），该门立刻变红并强制更新本节。

---

## 8. 重钉台账（old → new，逐条理由）

**未动**：`REAL*` / `CSV*` 一切真机常数、`REAL_SHA256` 指纹表、两条真机不变量与模型 ×1
不变量、run `d` 不可评分规则、`nr_moe_frag_factor` / `kept_frag_factor` 两个标定 margin。
**原则**（`docs/opdag_walker_core_2026-07-25.md` §6.6）：保留每一条不变量，只换举例形态；
读数不出带就**不平移带**。

**24 个变红的用例，逐条：**

| # | 用例 | old → new | 理由（幅度为什么是这个数） |
|---|---|---|---|
| 1 | `test_acceptance_gate::test_bucket_peaks_byte_neutral[dataflow/chain2]`（2 例） | `GOLDEN_BUCKET` 8×4 | 每格 = 峰值事件上在世的 r4 层数 ×128.0；无重算两跑按在途微批深度 4/3/2/1 → +512/+384/+256/+128 |
| 2 | `…::test_liveness_peaks_byte_neutral[dataflow-golden0 / chain2-golden1]`（2 例） | `GOLDEN_LIVENESS_*` 8×4 | 同上。⚠ 与上一轮不同，本项改的是 spec 的 `saves` → `liveness` 也动 |
| 3 | `…::test_aggregate_regression[bucket/hand_spec × dataflow/chain2]`（4 例） | bucket 0.836→**0.841**（min 0.700→0.703，max 1.023→**1.038**）；hand_spec·chain2 0.855→**0.860**（min 0.670→0.675，max **1.110 不动**）；hand_spec·dataflow 0.795→**0.799** | 28 格 0 格下移；bucket 21 上移/7 不动，hand_spec 20 上移/8 不动 |
| 4 | `…::test_delta_magnitude_gap_is_recorded` | bucket 0.517→**0.522**；hand_spec 0.633→**0.638** | stage3 fused 侧峰在 head 段不动、unfused 侧 `bwd@8` 吃满 +128.0 → delta +128.0。**缺口收窄**（与前三轮方向相反） |
| 5 | `…::test_unfused_on_cell_ratios_match_recorded_reference` | 0.802/0.811/0.815/0.828 → **0.804/0.814/0.818/0.831**；bucket 带 0.700–0.727 → **0.703–0.730** | unfused 侧同吃：`forward_before_topk` 在 `csa.py:667`/`:766` 两条路径逐字共用 |
| 6 | `test_pp4_recompute_anchor::test_full_recompute_theoretical_value[0/1/2]`（3 例） | `THEO_ON` s0 16143.8→**16243.8**、s1 11973.1→**12101.1**、s2 11717.1→**11845.1**（s3 21049.8 不变，峰在 head/loss 段） | pp4=2 层/stage、站点表下每 stage 恰好 1 个 r4 层 → +128.0。**s0 的 +100.0 是峰值事件由 `bwd@1`(r0) 换成 `bwd@2`(r4)**（实测 before bwd@2=16115.8 < bwd@1=16143.8） |
| 7 | `…::test_mtp_tail_stage_theoretical[0/1/2/3]`（4 例） | `THEO_MTP` s0-s2 同 `THEO_ON`；s3 29830.6→**29958.6** | MTP 层压缩比取 `ratios[-1]`=4（`head.py:250-252`）→ 它自己就是 r4 层，尾 stage 峰在它的 bwd 上 → +128.0 |
| 8 | `…::test_pp8_theoretical_value[1/3/5]`（3 例） | `THEO_PP8` s1 12879.2→**13007.2**、s3 12623.2→**12751.2**、s5 12367.2→**12495.2** | pp8=1 层/stage，站点表下只有 s1/s3/s5/s7 是 r4；s7 峰在 head 段 → 只动三格，各整 128.0 |
| 9 | `…::test_pp8_framework_gap[5]` | `_PP8_OVER_BAND` (1.00,1.12) → **(1.00,1.13)** | 上界随读数：s5 由 1.113 → **1.124**，留 ~0.006 余量（与上两轮 1.150→1.16 / 1.113→1.12 同款）。下界仍守 1.00（真翻成欠读必须变红）。`_PP8_UNDER` 仍 {0,7}，**没有移动任何 stage 的档** |
| 10 | `test_probe185_recon::test_fused_per_layer_increment` | 2339.1 → **2371.1**（/3109 = 0.752→0.763） | pp1/m1/无重算 → 差分就是层型计数差：F0 +128.0（2 个 r4 层 ×64.0@seq2048）、F1 +256.0（4 个）→ (256−128)/4 = **+32.0** 整。记录带 (0.70,0.90) 与 F0 带 (0.83,1.02) **都没动** |
| 11 | `…::test_p3p_m8_theoretical_and_gap` | 17381.8 → **17509.8**（/25343.5 = 0.686→0.691） | s0=(r0,r4)，峰值事件仍是 `bwd@2`（未易主）→ 整 128.0 |
| 12 | `test_from_mindformers::test_dsv4align_roundtrip_same_peak` | 13899.8 → **13963.8**（/15415.5 = 0.902→0.906） | 该配置 `cyc=[0,4,128]`、n=4 → `compress=[0,4,128,0]`，**只有 1 个 r4 层** → +64.0@seq2048 |

**没变红、但读数移动了、按规矩不动带的**（如实记录）：

| 锚点 | old → new | 带 |
|---|---|---|
| `BAND_OFF` s0–s3（pp4 无重算） | 0.892/0.922/0.859/0.833 → **0.909/0.940/0.874/0.838**（+512/+384/+256/+128） | 四条带**全不动**（新读数都还在原带内） |
| 185 `U1` unfused 峰 | 44760.9 → **44888.9**（/40194 = 1.114→**1.117**） | (1.05,1.18) **不动** |
| 185 `U2` unfused 峰 | 76585.3 → **76841.3** | `>56010` 断言不动 |
| 185 `F0` 绝对 | 22492.9 → **22620.9**（/26499 = 0.849→**0.854**） | (0.83,1.02) **不动** |
| 记分卡 `DSv4-fused (base)` | 13899.8 → **13963.8**（0.902→**0.906**） | (0.87,0.93) **不动** |
| 记分卡 `DSv4 mHC(x4)+MTP` | 18855.2 → **18919.2**（0.891→**0.894**） | (0.86,0.93) **不动** |

**边缘位落点**（不调参）：

- `pp8 s2 = 1.001` —— **逐 MiB 不动**（sim 11686.7 vs 真机 11678.0，仍高 8.7 MiB）。
  s2 是 **r128 层**、没有 `CSAIndexer` 对象 → 本轮对它**恒等**。**没有越界。**
- `pp8 s3` —— 任务书给的 0.963 是**五次重钉时的旧值**；六次重钉（逐层压缩比修复）已把它
  抬回 **1.059** 并移出 `_PP8_UNDER`。本轮再 +128.0 → **1.070**，仍在过读侧、离边缘更远。
- 本轮**没有任何 stage 跨过 1.00**（两个方向都没有）。

---

## 9. 执行记录

**环境**：Windows 11 / 本仓 `feat/unified-llm-modelspec`；权威源码快照
`E:\97-codes\torch_parallel\mf-src-167\`；before 对照树 = `00eab88` 的 `git worktree`。

**逐步（按执行顺序）：**

| # | 动作 | 观测 |
|---|---|---|
| 1 | `pytest tests -q` @ `00eab88` | **1964 passed** in 101.81s（基线） |
| 2 | `census_threeway.py` @ before | r0 **2229.5** / r4 **2080.6** / r128 **2005.6**；r4 有 6 张 64.0 MiB、r128 有 5 张 |
| 3 | `liveness_ab_validate --grad-mode chain2 --deltas` @ before | bucket n=28 mean=**0.836** min=0.700 max=1.023；hand_spec mean=**0.855** min=0.670 max=1.110；REAL 指纹 / I1 / I2 全 PASS |
| 4 | 源核对（`mf-src-167`） | `csa.py:224` ctx / `:229` `query_index` / `:314` `d_query_index`；`indexer.py:177,179,182-183,188-189,199`；`rope_utils.py:153,169,182,186,187`；`transformer_config.py:1578-1579` `apply_rope_fusion: bool = field(default=False, …)` —— **六处逐行核实** |
| 5 | `git merge --ff-only wip/r4-indexer-census` + `--amend` 去掉 WIP 措辞 | `f188a72` |
| 6 | `pytest tests -q` | **24 failed, 1940 passed**（与任务书给的 24 逐条对上） |
| 7 | `census_threeway.py` @ after | r0 **2229.5**（不变）/ r4 **2208.6**（+128.0）/ r128 **2005.6**（不变）；r4 8 张 64.0 MiB、r128 5 张、r0 5 张 |
| 8 | 峰值事件溯源（before/after 两棵树 dump `bwd@*`） | pp4 ON s0：before `bwd@1`=16143.8 / `bwd@2`=16115.8 → after `bwd@2`=**16243.8**（易主）。八跑 run a s0 同签名：before `bwd@1`=17278.6 / `bwd@2`=17250.6 → after `bwd@2`=**17378.6** |
| 9 | 重钉 `test_acceptance_gate.py`（`scratchpad/regen_gate_goldens.py` 生成，不手抄） | **34 passed** → 提交 `ff1f2b4` |
| 10 | 重钉 `test_pp4_recompute_anchor.py` | **39 passed** → 提交 `fee7729` |
| 11 | 重钉 `test_probe185_recon.py` + `test_from_mindformers.py` | 与 `test_scorecard_anchors.py` 合跑 **60 passed** → 提交 `d5c0fa8` |
| 12 | `pytest tests -q` | **1964 passed**（与基线同数） |
| 13 | 建 §5 守卫门 `tests/test_indexer_rope_saves.py` | after **22 passed**；拷进 before 树 → **12 failed / 10 passed**（红→绿证据） → 提交 `c157042` |
| 14 | `git diff 00eab88 HEAD -U0 -- . ':(exclude)docs' ':(exclude)scratchpad' \| grep -nE '^[-+].*(REAL\|CSV\|MEASURED\|sha256)'` | **零命中** |
| 15 | `liveness_ab_validate --grad-mode chain2 --deltas` @ after | bucket mean=**0.841** min=0.703 max=1.038；hand_spec mean=**0.860** min=0.675 max=1.110；REAL 指纹 / I1 / I2 **全 PASS** |
| 16 | `pytest tests -q`（收口） | **1986 passed**（1964 + 守卫门 22） |

---

## 10. 本轮之后仍是 **OOM-不安全** 的锚点（模型欠读真机）

**不调参掩盖。** 本轮**没有把任何一个锚点推进 OOM-不安全**，也**没有任何一个离开**
（所有移动都是同向收窄）。

| 锚点 | 本轮前 | **本轮后** | 峰值事件 | 变化 |
|---|---:|---:|---|---|
| `pp2-stage1 (loss,k_ce=8)` | 0.999 | 0.999 | `bwd@9` = lm_head | 未动（DSv3，无 CSA） |
| `cp2-none (loss,k_ce=4)` | 0.991 | 0.991 | `bwd@9` | 未动（同上） |
| `DSv3 8L none (dp2)` | 0.981 | 0.981 | `bwd@9` | 未动（同上） |
| `select self_attn (keep-FFN)` | 0.970 | 0.970 | `bwd@9` | 未动（同上） |
| `select mlp (keep-attn)` | 0.932 | 0.932 | `bwd@9` | 未动（同上） |
| `DSv3 4L full (dp2,sp)` / `8L full` / `4L full ep=2` | 0.996 / 0.993 / 0.993 | 同左 | 解码层 | 未动（DSv3） |
| `cp2 colossal full 4L` / `cp2 ulysses full 4L` | 0.999 / 0.999 | 同左 | 解码层 | 未动（DSv3） |
| `std 116 mha pp2 s0` | 0.934 | 0.934 | `bwd@4` | 未动（mha 不走 CSA） |
| `std 116 gqa pp2 s0` | 0.852 | 0.852 | `bwd@4` | 未动（同上） |
| `DSv4-fused (base)` | 0.902 | **0.906** | `bwd@5` = lm_head | 收窄（+64.0，1 个 r4 层 @seq2048） |
| `DSv4 mHC(x4)+MTP` | 0.891 | **0.894** | `bwd@6` = lm_head | 收窄（+64.0） |
| `pp4 OFF` s0–s3 | 0.892 / 0.922 / 0.859 / 0.833 | **0.909 / 0.940 / 0.874 / 0.838** | 解码层 bwd | 收窄（在途深度 ×128） |
| `pp4 ON` s0–s3（框架缺口档） | 0.668 / 0.818 / 0.831 / 0.895 | **0.673 / 0.827 / 0.840 / 0.895** | 解码层 bwd | 收窄（s3 峰在 head 段，恒等） |
| `pp8` s0 | 0.827 | 0.827 | 解码层 bwd | 恒等（s0 是 r0 层） |
| `pp8` s7 | 0.956 | 0.956 | head/loss 段 | 恒等 |
| `185 P3-P s0`（框架缺口档） | 0.686 | **0.691** | `bwd@2` | 收窄（+128.0） |
| `185 F 每层差分` | 0.752 | **0.763** | — | 记录门；仍低于差分锚 3109 |
| `185 F0 绝对` | 0.849 | **0.854** | — | 微收窄 |
| `185 std ON MHA/GQA s0`（框架缺口档） | 0.747 / 0.749 | 同左 | — | 未动（mha/gqa 不走 CSA） |

**最大已知成因不属本轮**：下面这些锚点的峰值事件就是 **`lm_head` / loss 段的反向**，由满 vocab
fp32 的 `bwd_scratch`（2020 MiB）主导 ——
`pp2-stage1`、`cp2-none`、`DSv3 8L none`、`select self_attn`、`select mlp`（皆 `bwd@9`，
出处 `docs/head_workspace_2026-07-30.md` §9）、`DSv4-fused`（`bwd@5`）、`DSv4 mHC(x4)+MTP`
（`bwd@6`），再加本轮实跑观测到的 `pp4 ON s3` / `pp4 OFF s3` / `pp8 s7`（皆 `bwd@9`）——
**共 10 条**。而 **`lm_head` / loss 段自己的反向 kernel workspace 至今未测、在模型里留 0**
（`tests/test_bwd_kernel_workspace.py::test_default_is_zero_and_byte_neutral_for_untagged_specs`
把这一格钉为 0）。
**本轮刻意不在 r4 层这里补偿它** —— 那会让「r4 欠读」与「head 欠读」两个独立错误互相掩盖，
而它们的配置依赖完全不同。要抬它们，得按 `docs/head_workspace_2026-07-30.md` §9 的清单去测
**MatMul wgrad / log_softmax / CE 链**的 kernel scratch。

**下一步能抬起本表的（按覆盖面排序）**：① `lm_head`/loss 段反向 kernel workspace（命中上面那 10 条）；
② r0 / r128 的反向 workspace（抬 pp4/pp8 的解码层 stage）；③ 本文 §7 ①②的 `Compressor` 保留对
（抬全部 DSv4 层型，需先按块尺寸重跑 §5.2 的直方图并列出 1–16 MiB 档）。
