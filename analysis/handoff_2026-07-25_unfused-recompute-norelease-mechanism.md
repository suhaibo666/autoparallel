# 交接文档 — unfused 全重算下 fp32 中间量"不释放"的机制定位（2026-07-25）

> 场景延续 `handoff_2026-07-24_caliber-switch-and-gap.md`：现场 DSv4-Flash（256卡 fused）理论 device_peak **38293** vs 真机 **58652**（理论=65%），差值 ~9G 被评估器记为「框架缺口」——**MS2.10 全重算下某些 fp32 中间量不释放**。§7.6 当时只说「dsv4_hybrid 走 save_for_backward、其 1.9GB/微批层组成**未逐张量指名（观测缺口）**」。本文把这条观测缺口的**机制**定位到源码级。
>
> 本文自包含：读完可直接接手。所有 file:line 均来自 167 真机上用户实际运行的 checkout（`/home/suhaibo/workspace/{mindformers,hyper-parallel}`，经 `docker exec shb_dsv4` 的 `/home` 挂载即运行时源码）。
> 铁律：**绝不杜撰真机数字；代码事实与"待坐实推断"分开标注**。

---

## 0. TL;DR — 最终结论（2026-07-25 真机微基准已坐实）

用户提问：真机激活偏大，是 **H1（mindformers 自定义反向存在时重算没生效）** 还是 **H2（hyper 没生效）**？

### 最终判决：**两个都不是。全重算生效了，hyper 也生效了。**

167/MS2.10 单卡微基准（§6，含**真 `UnfusedCSAIndexerLoss` + 真 `_IndexerLossAutoScaler`**）实测：

1. **H2 排除**：真 `FusedHyperConnectionModule ×4` recompute 下 `fwd_end(ON)=0`（`log_verdict/e1_dfunction.txt`）——hyper 激活正常释放。`num_residual_streams=4` 只是**放大**那份合法的重算边界 checkpoint-input（~1.9G/微批层，a9be44a 已定案为设计使然）。
2. **H1 也排除（本文原假设被自己证伪）**：`_IndexerLossAutoScaler` 的裸 `ctx.indexer_loss = kl` **只钉住它真正存的那个张量——一个标量**，不钉住其牵连的 O(S²) fp32 子图。实测 rc=ON 时 `fwd_end = 0`，**bare_ctx / saved / pyref 三种锚法完全同值**，且 NBLK=2/4/8 全为 0 → **全重算把这批 fp32 全释放了**。
3. **真正的答案 = 未建模的「单层 unfused fp32 重算工作集」**：rc=ON 下 `fwd_peak` **恒为 552 MiB（真模块 403 MiB），与 NBLK 完全无关**（2/4/8 三点同值）。即剩下的量是 **×1 层**（某层正在被重算的瞬间），**不是 ×微批深度**，也不是泄漏。
4. **附带铁证**：真模块里最大的那个 `attention_scores [b,np,sq,sk]` fp32 由 **`stop_gradient(query)/stop_gradient(compressed_kv)`**（`csa.py:794-795`）构成 → **它根本不是 saved activation**，连 rc=OFF 都只驻留 ~2 MiB/blk，纯瞬态。**所以"它没被重算释放"在机制上不可能成立**；现场 CSV 里看见它，只能是**峰值快照抓在某层正在计算的瞬间**。

> **一句话**：现场 ~9G「框架缺口」**不是"该释放没释放"（不是泄漏、不是重算失效、不是自定义反向绕钩子）**，而是 **unfused 路径单层 fp32 中间量在"该层被重算/反向的那一刻"合法占用的工作集**——评估器纯理论口径把它清零（对"驻留"是对的），但**峰值必须加一项"max 单层重算工作集"（×1）**。这也解释了为什么 fused 小得多：fused 把这批中间量塞进 kernel scratch，工作集本身就小。

**证伪记录**：本文初稿（同日早先）曾提出「裸 ctx 锚死 O(S²) KL 子图」的机制假设（保留在 §3 供追溯），并明确标注待坐实。§6 的实测**证伪了它**。结论按实测改写。

---

## 1. checkpoint wrapper 释放的到底是什么（机制基线）

调用链：模型层被 `apply_recompute` 包裹（`mindformers/pynative/distributed/activation_checkpoint.py:609` `apply_recompute` → `:545-551` `_wrap_cell_recompute` 把整个 `model.layers[layer_id]` 用 `checkpoint_wrapper(cell, context_fn=recompute_context_fn)` 包住）。

- `hyper_parallel` 侧 `checkpoint()`（`hyper-parallel/hyper_parallel/core/activation_checkpoint/activation_checkpoint.py:104-152`）最终 `return plat.checkpoint(function, *args, context_fn=composed_context_fn, use_reentrant=False)`。
- `plat.checkpoint` = **`ms.recompute`**（`hyper-parallel/hyper_parallel/platform/mindspore/platform.py:1778` `checkpoint` property `return ms.recompute`）。
- 非重入 recompute 的释放靠 **`ms.saved_tensors_hooks(pack, unpack)`**（platform 里 `activation_checkpoint/activation_swap.py:344` `class AsyncSaveOnCpu(ms.saved_tensors_hooks)`、`checkpoint_exclude_wrapper.py:62` `ms.saved_tensors_hooks(_pack_saved_tensor, _unpack_saved_tensor)`；session 侧 `platform.py` `recompute_session_ctx(session_id, retain_on_unpack=…)`、`recompute_handle_collector_ctx`）。
  - 前向：每个算子**经 hook 存下的 saved tensor** 被 pack 成 handle、**当场释放**；
  - 反向：首个 unpack 触发**整段前向重跑一次**重新生成。
- `recompute_context_fn`（`mindformers/pynative/distributed/activation_checkpoint.py:76-84`，配 `recompute_marker` `:66-73`、`is_in_recompute` `:86`）注释（约 `:52-61`）明说：*"during recompute the wrapped forward re-runs … any per-forward side effect（MoE aux-loss logging today; **reused as-is by other aux losses**）has to be skipped"* —— **indexer aux-loss 就在被重算的层内**，本应跟着 pack/重算。

> **决定性限制**：recompute **只能释放"经由 hook 存下的"张量**。任何用**裸 `ctx.属性 = tensor`** 挂住、或被区域外强引用锚住的张量，pack hook 看不见 → 不释放。

**167 实测把这条限制钉死**（`/home/suhaibo/workspace/log_verdict/v_matrix_part1.txt`、`part2.txt`，MS2.10 真机；`fwd_end`=前向结束后仍驻留，释放→~0、钉死→停在 ~4096）：

| 写法 | rc=OFF | rc=ON | 结论 |
|---|---:|---:|---|
| `ctx.save_for_backward(t)`（V2 / V3fix / DFunc+saved） | 4096 | **0** | hook 追踪 → 释放 |
| **裸 `ctx.g = t`**（V3 / DFunc+attr） | 4096 | **4096** | 绕过 hook → **钉死** |
| Combo（V2+V3+V5，detach 生产者+saved+裸attr） | 8192 | **4096** | 裸-attr 半边钉死 |
| 真 `FusedHyperConnection ×4`（`e1_dfunction.txt`） | 821 | **0** | hyper 释放 → **H2 排除** |

`save_for_backward` 由 `hyper-parallel/.../platform/mindspore/autograd_compat.py`（`enable_mindspore_backward_compat` + `_Function` 基类）接入 hook —— 故 saved 释放、裸 attr 钉死。

---

## 2. unfused 路径把这些 fp32 物化在哪（O(S²)）

`csa.py` dispatch：`construct`（`:645`）`if self.apply_dsa_kernel_fusion: return self._construct_fused(...)` `else: _construct_naive(...)`（`:656-658`）。unfused 走 `_construct_naive`（`csa.py:731`），indexer + KL loss 是**小算子拼的 Python 图**，材料全是 **O(S²) fp32**：

- `indexer.py` `CSAIndexer.construct` 非融合分支：`:234-235` `q=cast(q,fp32); k=cast(k,fp32)` → `:245` `index_scores = bmm(q,k)` = `[b,sq,n,sk]` **fp32、O(S²)**；`:247-264` relu/乘 weight/按头 sum/mask/topk，返回的是 **topk 收缩后**的 `index_scores`（`:264` `index_scores = topk_scores`）——O(S²) 在 construct 内部。
- `UnfusedCSAIndexerLoss.construct`（`indexer.py:321→409`）再造 O(S²) fp32：`:350` `attention_scores = matmul(query,key)*scale`、`:370` `[b,sq,sk]` fp32 `selection_mask`、`:379-380` `where(...) → softmax(cast(attention_scores, fp32))`、`:402` `predicted = softmax(cast(index_scores, fp32))`，最后 `:408` 得标量 `kl_div`、`:409` `return kl_div * loss_coeff`。

seq=4096 时单个 O(S²) fp32 即 GB 级；几层叠加 = handoff-2026-07-24 §2/§8.5 里 ~9G「框架缺口」（当时源码指名为 csa.py:488 / indexer.py:246,380，本 checkout 对应到上面这些行）。

---

## 3. ⛔【已证伪·保留追溯】原假设：锚点在 `_IndexerLossAutoScaler`

> **本节是同日早先提出、随后被 §6 实测证伪的假设，保留以供追溯。**
> 证伪要点：裸 ctx 只钉住**它实际存的那个张量**；`_IndexerLossAutoScaler` 存的是 **`kl_div` 标量**（`indexer.py:408` `kl_div = mean(...)`），因此钉住的量可忽略。O(S²) fp32 是由**普通算子**（relu/mul/softmax）保存的，这些**照常走 pack hook → 被释放**。V3 实验之所以钉住 4096 MiB，是因为它 `ctx.g` 存的**就是那个 4096 MiB 大张量本身**——与"存标量"是两种情形。
> 正确结论见 §0 与 §6。以下为原文。

`_construct_naive` 末尾**仅 unfused 分支**执行：

```python
# csa.py:828-829
if indexer_loss is not None and self.training:
    output = self.indexer_loss_auto_scaler.apply(output, indexer_loss)   # 把 aux-loss 挂到输出
```

`_IndexerLossAutoScaler` 是自定义 `_Function`（`indexer.py:268`）：

```python
# indexer.py:274-277
@staticmethod
def forward(ctx, output, indexer_loss):
    ctx.indexer_loss = indexer_loss     # ← 裸 ctx 属性，不走 save_for_backward
    return output
# indexer.py:280-286  backward: 注入 ones_like(indexer_loss)*main_loss_backward_scale 到 indexer_loss
```

这正是 §1 表里 **V3 / DFunc+attr 那一行的写法**。它让主 loss 反向时把 `ones*scale` 的梯度注入 `indexer_loss`，再沿 `indexer_loss.grad_fn`（softmax→matmul→fp32 cast 这条 KL 子图）回传到 query_index/key_index/weights。因为用的是**裸 `ctx.indexer_loss`**，**这个自定义 Function 的 ctx 及其牵连的整条 KL 反向子图，对 `ms.recompute` 的 saved_tensors_hooks 不可见** → pack 不到 → 不释放。

> 本质：**不是重算"重跑失败没生成"，而是这批 fp32 从一开始就不在"可丢弃集合"里**。recompute 只能丢它 pack 经手的张量；被自定义 Function 裸 ctx 锚住的子图它碰不到 —— 即 H1「自定义反向存在时，重算对这块没生效」。

注：unfused 侧对主路径做了 `ops.stop_gradient`（`csa.py:764-766` x/qr detach、`:794-795` query/compressed_kv detach 传入 `unfused_indexer_loss`），KL 子图与主注意力图**是分离的独立子图**，故这批 O(S²) fp32 **仅由 `_IndexerLossAutoScaler` 这一个锚点维系**——更坐实"锚点一旦绕过 hook，这整块就孤立地钉在显存里"。

> ⛔ **上面这段"更坐实"的推断恰恰是反的**：`stop_gradient(query)/(compressed_kv)` 意味着由它们算出的 `attention_scores`（最大的那块 O(S²) fp32）**不需要梯度 → 根本不被保存 → 纯瞬态**，refcount 一到就释放，与 recompute 无关。§6 实测证实：真模块 rc=OFF 时 `fwd_end` 仅 ~2 MiB/blk。

---

## 4. 为什么 fused 无此病（差异根 = `apply_dsa_kernel_fusion`）

`_construct_fused`（`csa.py:660-729`）**不建 Python KL 图**：走 `FusedSparseFlashMlaWithIndexerLoss.apply`（`csa.py:689`，`class` 在 `csa.py:178`）——

- 只 `save_for_backward` 一批 **bf16 输入**（`csa.py:224-236`，hook 可释放）；
- KL loss 梯度在 **backward 用融合 kernel** `npu_sparse_lightning_indexer_kl_loss_grad` 现算（`csa.py:282-296`）。

**修正后的正确表述**（§6 实测支撑）：fused 无 O(S²) fp32 **物化**，故**单层工作集本身就小**；unfused 把这批 fp32 拉成显式张量，**单层重算工作集大**。差异不在"锚点/是否释放"（两者都释放），而在**工作集大小**。test-unfused.yaml 与 test.yaml 唯一差 `apply_dsa_kernel_fusion: True→False`（`diff` 实测仅第 217 行），正是这个工作集大小开关。

---

## 5. 相关的两个既有真机事实（别回头查）

- **单卡全重算静默不 wrap（bug②）**：`log_verdict/e4_wrap_evidence.txt` —— 单卡 fused/std `L2_on` 均 **0 条 wrap 日志**，多卡 `dsv4h_pp4/pp8_recomp` = 2/1 条。对应 `log_bisect_sc_fused_L2_on/off/peak_rank0.json` 双双 **25064.9 MiB**（单卡 ON≡OFF 是伪象）。**任何单卡"层级 recompute"测量都无效，必须多卡（pp>1）**。门在 `trainer.py`（`enable_parallel=world_size>1`）。
- **裸 ctx 病的另一个已知命中点**：`dsa` 变体 `_DSAIndexerGradFunction` 前向预算 KL 梯度裸挂 ctx（handoff-2026-07-24 §4-3，一行改 `save_for_backward` 即修）。**dsv4_hybrid 的注意力主算子不中此病**（走 save_for_backward）——本文指出的 `_IndexerLossAutoScaler` 是 dsv4_hybrid **unfused** 独有的第三个命中点。

---

## 6. 微基准实测（2026-07-25，167/shb_dsv4/MS2.10 单卡）— 决定性数据

**方法**：直接调 `hp_checkpoint(blk, x, mode, context_fn=recompute_context_fn)`——**不经 trainer 的 `world_size>1` 门，故单卡也真跑重算**（绕开 bug②；`no_aux` 基线 rc=OFF/ON 有差即证明 wrap 生效）。跑 **NBLK 个独立 block** 后、**backward 之前**读 `ms.runtime.memory_allocated()`：

- `fwd_end ≈ NBLK × BIG` → 跨窗口**驻留**（泄漏）
- `fwd_end ≈ 0`，且 `fwd_peak` **与 NBLK 无关** → **已释放**，剩下的是单层瞬态工作集

脚本（已存 167）：`unfused_release_probe.py`（复刻）、`real_module_release_probe.py`（**真模块**）；结果 `log_release_probe/{out.txt,VERDICT_2026-07-25.txt}`。

### 6.1 复刻探针（BIG = 256 MiB/blk 的 O(S²) fp32）

| NBLK | rc=OFF `fwd_end` | rc=ON `fwd_end` | rc=ON `fwd_peak` | rc=ON `total` |
|---:|---:|---:|---:|---:|
| 2 | 640 | **0** | **552** | **1064** |
| 4 | 1280 | **0** | **552** | **1064** |
| 8 | 2560 | **0** | **552** | **1064** |

锚法对比（NBLK=4，`fwd_end` rc=OFF / rc=ON）：`no_aux` 32/0、**`bare_ctx`（精确复刻 `_IndexerLossAutoScaler`）1280/0**、`saved` 1280/0、`pyref` 1280/0、`dropped` 32/0。

**读数**：rc=OFF 线性增长（320 MiB/blk）；rc=ON **恒 0 驻留**、峰值 **552 恒定**（≈2×BIG，一个 block 的 scores+relu 同时在世）。**三种锚法完全同值 → 裸 ctx 不是变量。**

### 6.2 真模块探针（真 `UnfusedCSAIndexerLoss` + 真 `_IndexerLossAutoScaler`，`attention_scores` fp32 = 128 MiB/blk）

| NBLK | `real_bare` rc=OFF `fwd_end`/`peak` | rc=ON `fwd_end`/`peak` |
|---:|---:|---:|
| 2 | 4 / 277 | **0 / 403** |
| 8 | 16 / 353 | **0 / 403** |

（`pyref`/`dropped` 同量级；`no_aux` peak ≤ 2 MiB。）

**读数**：① rc=ON `fwd_end=0`、`peak` **403 恒定**（NBLK 无关）；② **rc=OFF 的 `fwd_end` 也只有 ~2 MiB/blk** —— 因为最大的 `attention_scores` 由 `stop_gradient(query)/(compressed_kv)`（`csa.py:794-795`）算出、**不需梯度→不被保存→纯瞬态**；③ rc=ON 的 peak **反而略高于** rc=OFF（403 vs 277–353），因为重算的再执行发生在反向其它缓冲还在世的时刻。

### 6.3 结论（三条，全部实测支撑）

1. **全重算对这批 O(S²) fp32 完全生效**（`fwd_end(ON)≡0`）。H1 的"重算没生效"**不成立**。
2. **裸 ctx 只钉住它实际存的张量**。`_IndexerLossAutoScaler` 存的是标量 → 钉住量可忽略。§3 假设**证伪**。（V3 钉住 4096 MiB 是因为它存的就是那个 4096 MiB 张量本身。）
3. **剩余量 = 单层重算工作集，×1 而非 ×微批深度**（三个 NBLK 点峰值同值）。→ 现场 ~9G 是**未建模的工作集**，非泄漏。

---

## 7. 下一步（评估器该改什么 + 还剩什么没验）

**评估器（这是真正的行动项，口径问题不是框架 bug）**：纯理论口径把全重算层的这批 saves 清零，对**驻留**正确，但**峰值漏了一项**。应加：

> `peak += max_over_layers( 该层前向的瞬态工作集 )`，**×1**（1F1B 下同一时刻只有一层在重算），unfused 时该项含 q/kv fp32 cast + index_scores + KL fp32 softmax；fused 时该项塌缩到 kernel scratch。

这一项**不该**按微批深度或层数放大——§6.1/§6.2 三点恒定值就是它 ×1 的证据。加完后现场 fused 38293 与真机 58652 的差应主要由"steady 驻留 + 单层工作集"解释，而 `FrameworkGapWarning` 的措辞应从"框架不释放"改为"重算工作集未建模"。

**关于四条 MS/mindformers issue（handoff-2026-07-24 §4）的订正**：
- ①「全重算释放缺口」→ **撤销**（实测释放正常，非框架缺口）；
- ②「单卡静默不 wrap」（`trainer.py` `world_size>1` 门）→ **仍然有效，可提**（本轮微基准正是靠绕开它才测到真重算）；
- ③「`dsa` 变体裸 ctx 挂**张量**」→ **仍然有效**（那里裸挂的是**大张量**，不是标量；`dsv4_hybrid` 不走）；
- ④「MS 缺 per-tensor 归属 API」→ 仍然有效。

**还没验的**：单层工作集在**真机多卡 pp>1** 下的绝对值（本轮是单卡缩比）。要拿现场量级需在 167 复制 `dsv4h_fused_pp4_recomp.yaml` → 翻 `apply_dsa_kernel_fusion:false` 跑 8 卡 A/B，比 `peak_rank0.json`；此时差值应≈**一层** unfused fp32 工作集，而非 ×层数/×深度——这可作为 §6.3 结论的多卡确认。

---

## 8. 真机资源（复验用）

- **167**（原 185，2026-07-23 换 IP；勿连 185）：容器 **`shb_dsv4`**（`docker exec`，MS2.10/CANN9.1，能跑 fused；本轮见它 `Exited 255`，`docker start shb_dsv4` 即起）。**`/home` 挂载进容器**，故 host `/home/suhaibo/workspace` 即运行时源码。SSH `ssh 192.168.9.167`（config 别名+key OK；CorpLink path-MTU 时通时断，连不上用 `fix-116-network` skill / `ssh-keygen -R`）。8×910B2，本轮 HBM 基本空闲。
- **源码**：`/home/suhaibo/workspace/mindformers/mindformers/pynative/transformers/experimental_attention_variant/{csa.py,indexer.py}`、`/home/suhaibo/workspace/hyper-parallel/hyper_parallel/{core/activation_checkpoint/activation_checkpoint.py, platform/mindspore/{platform.py, autograd_compat.py, activation_checkpoint/*}}`、`mindformers/pynative/distributed/activation_checkpoint.py`。
- **既有实测/脚本**（`/home/suhaibo/workspace/`）：**本轮新增 `unfused_release_probe.py` / `real_module_release_probe.py` + `log_release_probe/{out.txt,VERDICT_2026-07-25.txt}`**（§6 的全部数据，可 `NBLK=n MODES=... python <script>` 重跑；需 `export PYTHONPATH=/home/suhaibo/workspace/mindformers:/home/suhaibo/workspace/hyper-parallel` + `ASCEND_RT_VISIBLE_DEVICES=0`）；既有 `log_verdict/{v_matrix_part1,v_matrix_part2,e1_dfunction,e4_wrap_evidence}.txt`、`log_pattern_matrix/`、`bisect_patterns.py`/`bisect_patterns2.py`、多卡 launcher `dsv4h_fused_pp4_recomp.yaml`(pp4/dp2/ep2、8L 全重算、dsv4_hybrid、hyper on、fusion on)/`dsv4h_fused_pp8_recomp.yaml`。**尚无 unfused launcher**（要跑真机 unfused 需复制 fused recomp yaml 翻 `apply_dsa_kernel_fusion:false`）。
- **现场输入**：`C:\Users\suhaibo\Desktop\test.yaml`(fused)、`test-unfused.yaml`(仅第217行 `apply_dsa_kernel_fusion:False`)——均 256卡 pp8/dp=-1/ep32，**167(8卡)跑不了**，只作机制探针参照。

---

## 9. 关键文档索引

- **前序交接**：`analysis/handoff_2026-07-24_caliber-switch-and-gap.md`（纯理论口径切换 + 逐桶对账 + 框架缺口四条 + 可配旋钮）。
- **最终测试报告**：`analysis/dsv4_calibration_final_report_2026-07-23.md`（§7.5-7.9 驻留机制排查、§八 口径声明、§8.5 unfused CSV 逐块）。
- **本文**：把 §7.6「未逐张量指名的观测缺口」定位到源码级机制（`_IndexerLossAutoScaler` 裸 ctx 锚 O(S²) fp32 KL 子图，绕过 recompute hook）。
