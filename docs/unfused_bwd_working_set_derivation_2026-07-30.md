# unfused 稀疏注意力链的反向梯度工作集 —— 源码级推导（2026-07-30）

> **状态**：**推导完成，尚未实现**。本文由协调者在三次 `529 Overloaded` 导致实现 agent 无法启动
> 后先行落盘，避免分析丢失。实现 + 锚点重钉是后续工作。
> **权威源码**：固定快照 `E:\97-codes\torch_parallel\mf-src-167\`（**不是**本机 master 检出）。
> 站点尺寸：`sq=4096, b=1, n=64, d=512(v_head_dim), topk=512, sk=1024`。

---

## 1. 为什么现有公式集看不见它

`structure_mem._forward_max_live` **只遍历 `op.inputs` 与 `op.output`**。unfused 那套 fp32 复本群
**只声明在 `saves`** 里（`cost_eval/layers/dsv4_hybrid.py:334-343` 的 `_ucopies`），
故对 `fml` 贡献**恰好 0**。

实测佐证：r4 层的 `fml` 在 fused 与 unfused 下**都是 808.1 MiB**，而 `activation_saves`
分别是 3703.8 / **22569.3**。而唯一的反向工作集项是
`bwd_working_set = max(0, fml − bwd_scratch)` → **结构上不可能看见 unfused 的反向代价**。

前向侧声明其实**已经完备**；缺的是**反向侧的梯度缓冲**：unfused 的 `sparse_attn` op 目前有
`workspace_ref`（前向 kernel scratch）和 `bwd_workspace`（2026-07-29 实测，**fused 专属双重门**），
但**没有任何 `bwd_scratch`**。

---

## 2. 前向链逐行事实（`csa.py:464-533`）

| 行 | 张量 | 形状 | dtype | 站点字节 | 性质 |
|---|---|---|---|---:|---|
| :487 | `kv_gathered` | `[b,sq,topk,d]` | bf16 | 2048 MiB | gather 物化 |
| :490 | `q` | `[b,n,sq,d]` | fp32 | 512 MiB | `cast(permute(query))` |
| **:491** | **`kv_g`** | `[b,sq,topk,d]` | **fp32** | **4096 MiB** | `cast(kv_gathered, fp32)` |
| :495 | `q_bm` | `[b*sq,n,d]` | fp32 | 512 MiB | `reshape(permute(q))` → 副本 |
| **:496** | **`kv_bm`** | `[b*sq,d,topk]` | fp32 | **4096 MiB** | `reshape(permute(kv_g,(0,1,3,2)))` —— **交换末两轴 → 真实转置副本** |
| :497 | `scores` | `[b,sq,n,topk]` | fp32 | 512 MiB | `bmm(q_bm, kv_bm)` ← **消费者 ①** |
| :498/:503/:509/:512 | `scores·scale` / masked / `exp_scores` / `attn_weights` | `[b,n,sq,topk]` | fp32 | 512 MiB × 4 | softmax 链 |
| :519 | `aw_bm` | `[b*sq,n,topk]` | fp32 | 512 MiB | `reshape(permute(attn_weights))` → 副本 |
| **:520** | **`kvo_bm`** | `[b*sq,topk,d]` | fp32 | **0（view）** | `reshape(kv_g)` —— **纯 reshape，无 permute → 别名 `kv_g`，不占新内存** |
| :521 | `output` | `[b,sq,n,d]` | fp32 | 512 MiB | `bmm(aw_bm, kvo_bm)` ← **消费者 ②** |

**关键结构**：`kv_g` 有**两个消费者**（`:497` 与 `:521`）。其中 `kvo_bm` 是 view（零成本），
而 `kv_bm` 是**独立的 4096 MiB 转置副本**。前向此处已驻留 `kv_g` + `kv_bm` = **8192 MiB**。

> ⚠ 早先一处口径把两者都当成"复本"，会高估前向；也有一处把 `kvo_bm` 当独立副本。
> 以本表为准：**`kv_bm` 是副本，`kvo_bm` 是 view**。

---

## 3. 反向侧推导（VJP 语义）

`bmm(A,B)` 的 bprop 需要**两个操作数**，并产出与各自同形的梯度。逆序走：

| 步 | 事件 | 新增缓冲 | 字节 |
|---|---|---|---:|
| ① | `bmm(aw_bm, kvo_bm)` bprop | `grad_aw_bm` `[b*sq,n,topk]` | 512 MiB |
| ② | 同上 | **`grad_kvo_bm`** `[b*sq,topk,d]` | **4096 MiB** |
| ③ | `kvo_bm = reshape(kv_g)` bprop | **无**（reshape 的 bprop 亦为 reshape → `grad_kv_g` **别名** `grad_kvo_bm`） | 0 |
| ④ | softmax 链 bprop | 若干 `[b,n,sq,topk]` | 512 MiB × k |
| ⑤ | `bmm(q_bm, kv_bm)` bprop | `grad_q_bm` | 512 MiB |
| ⑥ | 同上 | **`grad_kv_bm`** `[b*sq,d,topk]` | **4096 MiB** |
| ⑦ | `kv_bm = reshape(permute(kv_g))` bprop | 逆 permute → **新缓冲**（permute 的 bprop **不能**原地） | **4096 MiB** |
| ⑧ | 累加进 `grad_kv_g` | 视实现是否原地 | 0 或 4096 |

### 共生判定（**这是本推导的关键，诚实答案比最大值少**）

- **必然共生（可辩护，8192 MiB）**：步 ⑥ 产出 `grad_kv_bm`(4096) 时，
  `grad_kv_g`(4096，持有消费者 ② 的贡献) **必然仍在世** —— 因为它还等着步 ⑦/⑧ 的第二份贡献。
  这两块**同时在世是 VJP 语义强制的**，不依赖任何实现细节。
- **无法从源码判定（+4096 MiB）**：步 ⑦ 逆 permute 的输出是否是**独立缓冲**、还是被融合进
  步 ⑧ 的累加，**取决于 MindSpore 的 permute-bprop 是否与 `AccumulateGrad` 融合**。
  快照里读不出来。**不应据"缺口正好对得上"来选它** —— 那就是拟合。

### 建议的建模口径

**声明可辩护的 8192 MiB**（= 2 × `[b,sq,topk,d]` fp32，随 `cp_kv` / tp / ep 正确缩放，
故应走 **`bwd_scratch_ref`** 而非字符串表达式 —— 后者只能 ÷cp，表达不了 tp/ep），
**把第三块 4096 记为未解决残差**并写明"需一次 MindSpore permute-bprop 的块级 tracker 观测才能定"。

---

## 4. 与既有三项的重复计账分析

| 既有项 | 是否重叠 | 理由 |
|---|---|---|
| `act_live` / `remat_saves`（= `activation_saves` 派生） | **不重叠** | 那是**前向 saved 集**（`kv_g`、`ukv_bm` 等），本项是**反向新产出的梯度缓冲**，是不同的物理内存 |
| `recomp_scratch` = `max(0, fml − ci)` | **不重叠** | `fml` 对 unfused 复本群贡献恰好 0（§1），故该项在 unfused 下不含这些字节 |
| `bwd_working_set` = `max(0, fml − bwd_scratch)` | ⚠ **需注意** | 二者都占 `bwd_scratch` 通道语义。加大 `bwd_scratch` 会**减小** `bwd_working_set`（因为是 `fml − bwd_scratch`）——**这是零和**，必须核实净效果，不能只看新增项 |

**最后一条是本项最大的实现风险**：`bwd_working_set` 与 `bwd_scratch` 在公式上互补，
天真地往 `bwd_scratch` 加字节可能被 `bwd_working_set` 的减少抵消掉。
实现时必须**实测净位移**，而不是假设新增项全额生效。

---

## 5. 预期与自检

- **unfused 格子应上升**：b/f 现 0.703–0.730；delta 比值现 0.477–0.697。
- **fused 格子必须零位移** —— fused 路径没有 `kv_g`（走 kernel scratch，已按实测 730 MiB 建模）。
  fused 动了即说明改得太钝。**这是刻意设计的自检项。**
- 报告口径用 **MAE / 最大单格误差 / 过读格数**，不用均值（均值是已证明的抵消假象：
  模型明显更差时它是 0.946）。

## 6. 未决

- 步 ⑦ 的第三块 4096 MiB（见 §3）。
- `bwd_scratch` ↔ `bwd_working_set` 的零和效应（§4 末），须实测。
- r0 滑窗分支（`dsv4_hybrid.py:406-411` 的 `r0_*`）走同一函数、同样缺反向项，量级按 `W0` 而非
  `topk` 缩放 —— 本推导未覆盖，需同法处理。
