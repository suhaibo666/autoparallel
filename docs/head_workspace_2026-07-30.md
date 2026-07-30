# head/embedding 侧 kernel workspace：实测 → 建模（2026-07-30）

> 承接 [`kernel_workspace_2026-07-29.md`](kernel_workspace_2026-07-29.md) §8 ④ ——
> 那一轮把 `GatherDGradV2` 那一档（**2228.003 MiB**，全跑最大单笔 workspace）列为
> 「已测、律已验证，但归属在 embedding/lm_head 段，不属本人可改范围」的**交接项**，
> 并判它是「当前最大的单笔未建模 workspace」。本轮把它建进模型。
>
> **纪律不变**：绝不编造真机数；绝不发明拟合常数；「我跑了并观察到」与「我推断」分开写。
> `REAL*` / `CSV*` 常数与 `REAL_SHA256` 指纹**一个字节未动**（§7 有过滤 diff 佐证）。

---

## 0. 一句话结论

**两件事，方向相反，都要如实说：**

| | 结论 |
|---|---|
| **建模** | 这笔 workspace 的律被**扩到了两个站点、四个点、两次独立采集，全部逐字节吻合**——`4·vocab·H + 16 MiB + 12·H·(B·S) + 3072 B`。第二个站点的数据**本来就躺在仓里**（`analysis/realmachine/pp2_norecomp/op_816362.csv`），此前无人把它与这笔 workspace 对上。它现在挂在 `embedding` op 上，`bwd@<embedding>` 事件加法计入。 |
| **锚点** | **14 个 OOM-不安全锚点里，13 个一个字节没动。**因为——见 §2 的判决——`GatherDGradV2` 是 **word-embedding 的反向**，**不是** `bwd@head`。任务书预判的这条分叉成立：**项是对的、落点是对的、但它够不着那些锚点的峰**。唯一动的是 `pp2-stage0`（1.021 → **1.032**，过读侧），因为只有它的 `bwd@<embedding>` 本来就贴着峰。 |

**没有为了让锚点动而挪过一个字节的归属。**

---

## 1. 建模对象：一条被两次独立采集逐字节验证的律

### 1.1 律

```
bwd_ws(GatherDGradV2) = 4·vocab·H  +  16 MiB  +  12·H·(B·S)  +  3072 B
                        └── ①      └── ②      └── ③           └── ④
```

- ① `4·vocab·H` —— 该 op 自己要累加进去的 `[vocab,H]` **fp32 梯度表的一份影子拷贝**
- ② `16 MiB` —— H 无关常数（两个 H 值上都恰为 **16.000** MiB）
- ③ `12·H` B/token —— 每 token 3 个 fp32 的 hidden 宽副本（H=4096 → 49152 B/token；H=1792 → 21504 B/token）
- ④ `3072 B` —— 分配器尾（在 167 的三位小数显示上正是那个恒定的 `+0.003 MiB`）

### 1.2 采集 ①（任务书交给我的那一份）：167 / MS 2.10 memory-tracker

`docs/kernel_workspace_2026-07-29.md` §4.4 / §8④。DSv4-hybrid 站点：
`vocab=129280`、**H=4096**、**B=1**、tp=1、cp=1，run `c`（fused / 无重算 / L8 / m4 / pp4·dp2·ep2）。
BWD 相位单 kernel 极大值，**stage 0**：

| S | 实测 | 本律 | 三位小数 |
|---:|---:|---:|---:|
| 1024 | 2084.003 MiB | 2185235456 B = 2084.00293 MiB | **2084.003** ✓ |
| 2048 | 2132.003 MiB | 2235567104 B = 2132.00293 MiB | **2132.003** ✓ |
| 4096 | 2228.003 MiB | 2336230400 B = 2228.00293 MiB | **2228.003** ✓ |

任务书给的 `2036.0 MiB + S·49152 B` 就是本律在 **H=4096** 上的特例：
`4·vocab·H = 2020.0 MiB`、`+16 MiB = 2036.0 MiB`、`12·H = 49152 B/token`。

### 1.3 采集 ②（本轮**新发现**，仓内既有数据）：`analysis/realmachine/pp2_norecomp/`

**完全独立的另一次采集、另一个模型、另一组维度**：DSv3 8L 无重算 pp2/dp1，
`vocab=129280`、**H=1792**、**B=2**、S=4096、tp=1、cp=1
（维度出处：`cost_eval/presets.py:32,35,36` `hidden_size=1792 / vocab_size=129280 / seq_length=4096`；
B=2 出处：`tests/test_ce_optstep.py:31` 「dp=1 pp=2 global_batch=2 → B=2」）。

`op_816362.csv`（**持有 embedding 的那个 rank**）里 `Name == GatherDGradV2` 共 24 行，按尺寸/寿命分三类：

| `Size(KB)` | 字节 | `Duration(us)` | 判 | ×n |
|---:|---:|---:|---|---:|
| **1093379.0** | **1119620096** | ≈22.9 | **瞬态 → workspace** | 3 |
| 904960.5 | 926679552 | ≈5.59e6（全程） | 长寿 → 就是 `4·vocab·H = 926679040 B` 的 fp32 **梯度表本体** + 512 B 分配器尾 | 3 |
| 16513.5 / 256.5 | — | 瞬态/长寿 | 小件，另有其主 | 9 / 9 |

本律给：`4·129280·1792 + 16 MiB + 12·1792·(2·4096) + 3072` = **1119620096 B** —— **delta = 0 B**。

对照同跑另一个 rank 文件 `op_816365.csv`（stage 1，**无** embedding）：
**没有任何 ≥1 MiB 的 `GatherDGradV2` 块**（只剩 16513.5 KB / 256.5 KB 小件）。

### 1.4 这次采集把哪些轴从「未扫」变成「已验」

| 轴 | 上一轮（任务书口径） | **本轮** |
|---|---|---|
| S | 3 点线性 ✓ | 不变 ✓ |
| **H** | **未扫** | **两点验证（1792 / 4096）**：常数项与每-token 项**各自**逐字节成立 |
| **B** | **未扫**（只测过 1） | **两点验证（1 / 2）**，经 token 数 `B·S` |
| vocab | 未扫 | **仍未扫**（两次采集都是 129280）——见 §6 |
| tp / cp | 未扫 | **仍未扫**——见 §6 |

> **推断与观测的分界**：`4·vocab·H` 这个**写法**是归因推断（vocab 未扫，我无法从数据把
> `4·vocab·H` 与 `f(vocab)·H` 分开）。支持它的证据是：该项**逐字节等于该 op 自己输出的那张
> fp32 梯度表**，而那张表在采集 ② 的同一个 CSV 里**被单独看见**（904960.5 KB 那 3 行）。
> H 与 B 的缩放**不是**推断，是两点实测。

---

## 2. ⚠ 关键判据：`GatherDGradV2` 与 `bwd@head` **不是同一个事件**

任务书要求先判这一条。**判决：不是。** 三条独立证据 + 模型侧事件线佐证。

### 2.1 证据 ①（决定性）：`4·vocab·H` 那一项**与 S 无关**

若这笔 workspace 属 **loss 侧**（CE 的 `gather(logits, target)` 的 dgrad），其主项应是
`[S·B, vocab]` fp32 = `4·vocab·B·S`，**必须 ∝S**：S=1024 时只剩 1/4。
实测三点 2084.003 / 2132.003 / 2228.003 —— S 缩到 1/4 时只掉 **6.5 %**。**证伪。**

> 这条判据非做不可，因为在 167 站点上 **S·B = 4096 == H**，两种假设在 S=4096 那一点
> **数值恰好相同**（`4·vocab·H` 与 `4·vocab·S·B` 都是 2020.0 MiB）。**正是 S 扫描把它们分开。**
> 采集 ②（H=1792 ≠ S·B=8192）再独立确认一次：实测主项 883.75 MiB = `4·vocab·H`，
> 而 `4·vocab·B·S` 在那里是 4040 MiB —— 差 4.6 倍，无法混淆。

### 2.2 证据 ②：只出现在**持有 embedding 的那个 rank/stage** 上

| 采集 | 有大块的 rank | 无大块的 rank |
|---|---|---|
| ① 167 pp4 | rank0（**stage 0**）：2228.003 MiB ×24 | rank2（stage 1）：同名 kernel 只 16.03 MiB |
| ② pp2 | `op_816362.csv`（**stage 0**）：1067.753 MiB ×3 | `op_816365.csv`（stage 1）：无任何 ≥1 MiB 块 |

模型里 `embedding` 恒在 **stage 0**、`lm_head` 恒在**末 stage**
（`cost_eval/parallel_model.py:122` `return [0] * hp + mid_map + [pp - 1] * tp_`，
注释逐字「embedding→stage0, head→末 stage」）。pp4 的末 stage 是 3、pp2 的是 1
——**两次采集里 lm_head 都不在出现大块的那个 rank 上**。

### 2.3 证据 ③：`GatherDGradV2` 是 Gather 的 dgrad，而 `[vocab,H]` 形状的 Gather 只有 word embedding

`lm_head` 前向是 MatMul（`cost_eval/layers/head.py:94/98` `OpSpec("lm_head", OpType.MATMUL, ...)`，
对应源侧 ColwiseParallel Linear），其 wgrad 由 MatMul 产出，**不走 GatherDGrad**。
且 `docs/kernel_workspace_2026-07-29.md` §4.4 逐字：`GatherDGradV2` 是**真 pyboost 算子**
（在 `task.csv` 里）→ 此处 `node_name` 可信，不受该文 §1.3 「自定义融合算子标签陈旧」的限制。

### 2.4 模型侧事件线：两者**从不相邻**，`bwd@head` 也从不是 embedding 的事件

`layer_pattern` 恒为 `[embedding] + decoder* + [lm_head]`（`cost_eval/build_llm.py:313,325`），
即 embedding = 层 0、lm_head = 末层；反向逆序 → `bwd@<head>` **最先**、`bwd@0` **最后**，
是整条反向时间线的**两个端点**。`tie_word_embeddings` 也不改变这一点：DSv3/DSv4 站点均
`tie=False`（实测打印 `tie= False`），且即便 tie，两个 op 的**事件**仍是两个。

**实测的事件距离**（`scratchpad/probe_head_ws_events.py` / `probe_emb_bwd_gap.py`，本轮跑的）：

| 锚点 | 峰值事件 | 峰值 MiB | `bwd@0`（embedding） | 差 | 本项（2228 或 1067.8）够得着？ |
|---|---|---:|---:|---:|---|
| DSv3 8L none | `bwd@9`(head) | 19591.5 | 7828.9 | 11762.7 | ✗ |
| select self_attn | `bwd@9` | 18256.0 | 7828.9 | 10427.1 | ✗ |
| select mlp | `bwd@9` | 14696.0 | 7828.9 | 6867.2 | ✗ |
| cp2-none | `bwd@9` | 19933.1 | 7828.9 | 12104.3 | ✗ |
| DSv4-fused base | `bwd@5`(head) | 13899.8 | 8441.5 | 5458.3 | ✗ |
| DSv4 mHC+MTP | `bwd@6`(head) | 18855.2 | 9522.3 | 9332.9 | ✗ |
| pp4 ON s0 | `bwd@1` | 16143.8 | 13363.4 | 2780.4 | ✗（差 552.4） |
| pp8 ON s0 | `bwd@1` | 20486.2 | 17705.8 | 2780.4 | ✗（差 552.4） |
| 185 P3-P s0 | `bwd@2` | 17381.8 | 13619.4 | 3762.4 | ✗ |
| **pp2-stage0** | `bwd@4` | 10458.1 | **9505.2** | **952.9** | **✓ → 新峰 `bwd@0`** |

> **所以：任务书预判的那条分叉成立。**本项落在正确的 op 上、值逐字节正确，但
> **14 个 OOM-不安全锚点里 13 个的峰值事件是 `bwd@head` 或某个 decoder 层的 bwd，
> 而不是 `bwd@<embedding>` → 它们一个字节不动。**
> 要抬这 13 个，需要的是 **lm_head / loss 段自己的**反向 kernel workspace（MatMul wgrad、
> log_softmax、CE 链的 kernel scratch）——那是**另一次测量**，本轮没有它，**故留 0 并记账**（§6 ①）。

---

## 3. 建模：复用既有通道，只加一个挂载点

### 3.1 改了什么（全部加法，未标注者逐字节不变）

| 文件 | 改动 |
|---|---|
| `cost_eval/layers/head.py` | 顶部新增 `_EMB_GATHER_DGRAD_BWD_WS`（含全部实测出处/边界）；`build_embedding_ops` 的 `embedding` op 挂上 `bwd_workspace` + `bwd_workspace_ref` |
| `tests/test_emb_bwd_kernel_workspace.py` | **新增 9 道门**（§3.3） |
| `tests/test_bwd_kernel_workspace.py` | 「未标注→0」那道门的**样例**由 `embedding` 移到 `lm_head`（不变量原样） |
| `tests/{test_pp4_recompute_anchor, test_probe185_recon, test_x4_p2p_pp}.py` | 三处锚点重钉（§5） |

**通道一个都没新建**（任务要求）：`OpSpec.bwd_workspace{,_ref}` → `ResolvedOp.bwd_workspace_bytes`
→ `StructureMemory.bwd_workspace`（层内取 **max**）→ `Buckets.workspace`（BWD 事件）
—— 全是 2026-07-29 那轮为融合稀疏 flash-MLA 建好的同一条路。

### 3.2 两条子通道的分工（照 r4 先例）

```python
_EMB_GATHER_DGRAD_BWD_WS = {
    "bwd_workspace":     "4*vocab*H + 16777216 + 3072",      # S 无关部分（字符串通道，不 ÷cp）
    "bwd_workspace_ref": TensorRef("emb_gather_dgrad_ws",
                                   ("B", "S", "3*H"), dtype_bytes=4),  # 每-token 部分（÷cp）
}
```

字符串通道**刻意不吃**「含 S 就 ÷cp」那条规则（`shape_eval.py` 对 `bwd_workspace` 本就不施加它）
—— 整体 ÷cp 会把常数项也除掉 → cp>1 欠读 = OOM-**不安全**。每-token 项走 TensorRef，
由 `resolve_tensor` 只对首个 S 维 ÷cp。

**它只挂在一个 op 上**（`embedding`），因为测量把它定位到了一个 kernel；
**没有把任何数字摊到层上，也没有为了让哪个锚点落位而调过一个字节。**

### 3.3 新增的 9 道门（守的是**测量**，不是模型自洽）

`tests/test_emb_bwd_kernel_workspace.py`：

1. ①的三个 seq 逐字节穿过 `build_llm_spec → ShapeEval → StructureMemory`（3 例）；
2. ②的那一笔**直接从 `op_816362.csv` 读**（不转抄常数）—— 跨 H、跨 B 的缩放被改坏必红；
3. ②里那档**长寿**块 == `4·vocab·H` 的梯度表本体（钉住「workspace 与梯度表是两块、不双计」）；
4. **只出现在持有 embedding 的 rank** 上（stage1 最大块只有 stage0 的 1/66）；
5. **归属**只在 `embedding` op 上（挡「摊到层上」与「为了让锚点动而挪到 lm_head」）；
6. **S 无关性**：`hi − lo == 12·H·ΔS`，且 S 折半后本项 > 0.9×（排除 loss 侧假设）；
7. cp>1 只切每-token 项、常数项不缩放。

以及 `tests/test_bwd_kernel_workspace.py` 里那道被**移动样例**的门：现在钉的是
**`lm_head` 段的反向 workspace 必须为 0** —— 它至今未测，留 0 是**已知欠读**，
这道门防止有人拿别处的实测值去顶（正是本轮最想守住的失败模式）。

---

## 4. 逐锚点 before → after（**全部实跑重算，非转抄**）

复现：`PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_anchor_deltas_emb_ws.py`
（在裸 HEAD 的 worktree 与本树各跑一遍逐行 diff）。

### 4.1 记分卡（`python sim_vs_real_report.py`）

| 锚点 | real MiB | before | after | 判 |
|---|---:|---:|---:|---|
| DSv3 4L full (dp2,sp) | 12473 | 0.996 | **0.996** | 不动 |
| DSv3 8L full (dp2) | 13953 | 0.992 | **0.992** | 不动 |
| DSv3 4L full ep=2 | 12474 | 0.993 | **0.993** | 不动 |
| cp2 colossal full 4L | 12433 | 0.999 | **0.999** | 不动 |
| cp2 ulysses full 4L | 12441 | 0.999 | **0.999** | 不动 |
| **pp2-stage0 (optstep)** | 10246 | 1.021 | **1.032** | **↑ 唯一变动**；峰值事件 `bwd@4` → **`bwd@0`** |
| pp2-stage1 (loss,k_ce=8) | 45655 | 0.999 | **0.999** | 不动（末 stage 无 embedding） |
| cp2-none (loss,k_ce=4) | 20119 | 0.991 | **0.991** | 不动（峰在 `bwd@9`） |
| DSv3 8L none (dp2) | 19967 | 0.981 | **0.981** | 不动（峰在 `bwd@9`） |
| select self_attn | 18828 | 0.970 | **0.970** | 不动 |
| select mlp | 15765 | 0.932 | **0.932** | 不动 |
| select both | 13953 | 1.000 | **1.000** | 不动 |
| **DSv4-fused (base)** | 15416 | 0.902 | **0.902** | **不动**（峰在 `bwd@5` = lm_head，`bwd@0` 差它 5458.3 MiB） |
| **DSv4 mHC(x4)+MTP** | 21153 | 0.891 | **0.891** | **不动**（峰在 `bwd@6` = lm_head，差 9332.9 MiB） |

### 4.2 116 std（`tests/test_std_attn_anchor.py::_peaks`）

| 锚点 | real | before | after |
|---|---:|---:|---:|
| mha pp2 s0 | 15946.1 | 0.9343 | **0.9343**（不动） |
| mha pp2 s1 | 19227.3 | 0.9889 | **0.9889** |
| mha pp1 s0 | 24191.1 | 0.9903 | **0.9903** |
| gqa pp2 s0 | 15116.0 | 0.8523 | **0.8523**（不动） |
| gqa pp2 s1 | 18459.3 | 0.9846 | **0.9846** |
| gqa pp1 s0 | 23039.1 | 0.9747 | **0.9747** |

> 116 std pp2-s0 的峰在 `bwd@4`（无重算逐层反向），`bwd@0` 够不着 → 这两个 OOM-不安全锚点
> （0.934 / 0.852）**没有被本项改善**。如实记。

### 4.3 185 / pp4 / pp8 族

| 锚点 | real | before | after | 判 |
|---|---:|---:|---:|---|
| pp4 ON s0/s1/s2/s3 | — | 0.668 / 0.746 / 0.790 / 0.894 | **同左，逐 MiB 不变** | s0 的 `bwd@0` 差峰 2780.4 > 2228.003 |
| pp4 OFF s0-s3 | — | 0.892 / 0.867 / 0.815 / 0.832 | **同左** | 同上（差 4179.9） |
| **pp4 ON +MTP s3** | 39898 | 28291.7（0.709） | **29789.8（0.747）** | **↑** MTP 层内含共享 embedding op |
| pp4 ON +MTP s0-s2 | — | 不变 | **不变** | — |
| pp8 ON s0…s7 | — | 0.827/1.045/**1.001**/0.963/1.150/1.017/1.101/0.956 | **逐位相同** | 除 s0 外各 stage 无 embedding；s0 差峰 2780.4 |
| **185 std ON MHA s0** | 11131.6 | 8042.8（0.723） | **8310.7（0.747）** | **↑** 峰值事件易主 `bwd@0` |
| **185 std ON GQA s0** | 10747.6 | 7562.8（0.704） | **8046.7（0.749）** | **↑** 同上 |
| 185 std ON MHA/GQA s1 | — | 不变 | **不变** | 末 stage 无 embedding |
| 185 P3-P m8 s0 | 25343.5 | 17381.8（0.686） | **17381.8** | 不动（差 3762.4） |
| 185 F0 / F 每层差分 / U1 | — | 0.844 / 0.754 / 0.966 | **同左** | 峰不在 `bwd@0` |

### 4.4 边缘位（任务点名要看的）

| 位 | before | after | 说明 |
|---|---:|---:|---|
| **pp8 s2** | **1.001**（高出真机 8.7 MiB） | **1.001，逐位不动** | pp8 每 stage 1 层，s2 上**没有 embedding 伪层**（`parallel_model.py:122`：embedding 恒在 stage0）→ 本项在它上面恒 0。**没有越界。** |
| pp8 s3 | 0.963（欠读） | **0.963** | 同上 |
| pp2-stage0 | 1.021 | **1.032** | 过读侧加深 1.1 个百分点。这是**真机自己量到的那一笔**（同一次采集的 profiler 文件），不是拟合。 |

**结论：本轮没有把任何一个锚点从欠读推过 1.0**（唯一越界候选 pp8 s2 根本不带 embedding）。

---

## 5. 八跑验收门 before → after

`PYTHONIOENCODING=utf-8 PYTHONPATH=. python tools/liveness_ab_validate.py --grad-mode chain2 --deltas`

| 指标 | before | **after** |
|---|---|---|
| bucket 聚合 mean / min / max（n=28） | 0.836 / 0.700 / 1.023 | **0.836 / 0.700 / 1.023（逐位不变）** |
| hand_spec·chain2 mean / min / max | 0.855 / 0.670 / 1.110 | **0.855 / 0.670 / 1.110（逐位不变）** |
| 32 格逐格 sim/real（8 跑 × 4 stage） | — | **逐位不变**（整表 diff 为空） |
| `unfused − fused` delta 全表 | — | **逐位不变** |
| `REAL_SHA256` 指纹 | PASS `41e279e591ae4ae9…` | **PASS（同值）** |
| I1/I2 真机 ×1（层数 / 微批数） | PASS | **PASS** |
| I1/I2 模型 ×1（bucket & hand_spec） | PASS | **PASS** |

**为什么八跑门一格不动**：八跑的 stage0 峰值事件都在解码层的 bwd 上，`bwd@0` 距峰
2780.4 MiB（ON）/ 4179.9 MiB（OFF），都大于本项在该 config 的 2228.003 MiB
——**差 552.4 MiB 没够到**。这个数字本身值得记：它说明只要 head/loss 段的 workspace
被测出来并且 ≥ 552 MiB，八跑 stage0 就会被抬起来；但那要另一次测量，本轮不编。

---

## 6. 诚实清单：本测量**不**支持什么

| # | 事项 | 状态 |
|---|---|---|
| ① | **`lm_head` / loss 段自己的反向 kernel workspace** | **未测 → 留 0，已知欠读。**这是 13 个 OOM-不安全锚点的峰值事件所在（`bwd@head`，由 2020 MiB 满 vocab fp32 `bwd_scratch` 主导）。要抬它们，必须测 **MatMul wgrad / log_softmax / CE 链**的 kernel scratch，而不是把本项挪过去。已用 `tests/test_bwd_kernel_workspace.py::test_default_is_zero_and_byte_neutral_for_untagged_specs` 钉住这一格为 0。<br>**→ 2026-07-30 已闭合，见 [`head_loss_bwd_workspace_2026-07-30.md`](head_loss_bwd_workspace_2026-07-30.md)。**实测 dgrad `(2·vocab+4·H)·(B·S)+20 MiB+1024`（16 点逐字节，vocab 轴扫了 11 个值）；全库 48 条锚点/stage 里 **25 条上移、15 条由欠读翻过 1.0**（= 过读 = OOM 安全侧），OOM-不安全总数 **38 → 23**；本行说的那批「峰在 `bwd@head`」的锚点**几乎全部被抬起**，仍不安全的 23 条见该文 §14。⚠ 同时暴露出 `bwd_scratch` 的 `K_CE` 多记 4 张满 vocab fp32 平面（该文 §8.2）。 |
| ② | **vocab 轴** | **未扫**（两次采集都是 129280）。写成 `4·vocab·H` 是**归因推断**（依据：它逐字节等于该 op 自己输出的那张 fp32 梯度表，而那张表在采集 ② 的同一 CSV 里被单独看见）。若真值其实与 vocab 无关，则小 vocab 模型上本项过读、大 vocab 欠读。 |
| ③ | **tp 轴** | **未测**。`4·vocab·H` 走字符串通道 → **不 ÷tp**；而源码上 `emb_w` 是 `Shard(0)`（`head.py` 的 `shard={0:"tp"}`，pynative TP>1 无条件 RowwiseParallel）→ tp>1 时真值应更小 ⇒ 本式**过读 = OOM 安全侧**。今天所有锚点 tp=1，无一依赖这条。 |
| ④ | **cp 轴** | 结构上已分开处理（常数项不缩放、每-token 项 ÷cp），但 cp>1 **未实测**。 |
| ⑤ | **sp（序列并行）轴** | 每-token 项的 TensorRef **不标 `shard={0:"sp"}`** → tp>1+SP 时不按 sp 切 = **过读**（安全侧）。与 r4 项同一处理，未实测。 |
| ⑥ | **MTP 层内那个共享 embedding 的调用点** | 两次采集的站点**都没开 MTP** → 该调用点本身**未被直接测到**。模型给它同一个值，理由是**同一个 op / 同一个 kernel**（`multi_token_prediction.py` 对 roll 后的 input_ids 走同一个 embedding cell）。这是**推断**，不是观测；它是 pp4+MTP s3 那 +1498.0 MiB 的全部来源。 |
| ⑦ | 采集 ② 的 c/k 拆分 | 采集 ② 只有**一个** S 点 → 单条方程、两个未知量（常数项 c 与每-token 系数 k）。本文取「c 与 H 无关」（来自采集 ① 的 3 点拟合）后解出 `k(1792) = 21504 = 12·1792` —— 与 `k(4096) = 49152 = 12·4096` **同一形式**。这是**自洽解**，不是独立测出的两点。 |
| ⑧ | `+3072 B` | 在采集 ② 上**逐字节**测得；在采集 ① 上只能从三位小数显示（`+0.003 MiB`）读到，二者一致。 |
| ⑨ | 上一轮遗留的 ①②③ | `docs/kernel_workspace_2026-07-29.md` §8 的**逐层驻留欠读**（indexer 链 3 块 64.0005 MiB）、**r0/r128 的反向 workspace**、**前向 304.001 MiB 的 mHC pre-sinkhorn** —— 本轮**一条都没动**，仍未闭合。<br>**→ 其中第三条（前向 mHC pre-sinkhorn）已于 2026-07-30 闭合，见 [`mhc_fwd_workspace_2026-07-30.md`](mhc_fwd_workspace_2026-07-30.md)**；结论与本文同款且更彻底：**项对、落点对，但一个锚点都够不着**（36/36 峰在 BWD）。另两条（逐层驻留欠读 / r0·r128 反向 workspace）仍未闭合。 |

---

## 7. 重钉台账（old → new，逐条理由）

**未动**：`REAL*` / `CSV*` 一切真机常数、`REAL_SHA256` 指纹表、两条真机不变量与模型 ×1
不变量、run d 不可评分规则、`nr_moe_frag_factor` / `kept_frag_factor` 两个标定 margin
（**一个字节没动**），以及 **14 个记分卡锚点里的 13 个**。

核验方法（同上一轮）：

```bash
git diff 228582d HEAD -U0 -- . ':(exclude)docs' ':(exclude)scratchpad' | grep -E '^-' | grep -Ei 'REAL|CSV|MEASURED|sha256'
# → 空。即**没有任何既有真机常数行被删改**；新增的 REAL_EMB_GATHER_DGRAD_WS_MiB
#   在新文件 tests/test_emb_bwd_kernel_workspace.py 里，是**新增测量**，不替换任何旧值。
```

| 位置 | old → new | 理由 |
|---|---|---|
| `cost_eval/layers/head.py::build_embedding_ops` | 无 → 挂 `bwd_workspace{,_ref}` | 两轮独立真机采集的实测律，四点逐字节 |
| `tests/test_emb_bwd_kernel_workspace.py` | **新增**（9 例） | 新机制的守卫门，见 §3.3 |
| `tests/test_bwd_kernel_workspace.py::test_default_is_zero_...` | 样例 `embedding` → **`lm_head`** | **不变量原样保留**（未标注 op → 0）；embedding 现已标注，样例必须移。新样例顺带把「head 段仍为 0」这条**已知欠读**钉住 |
| `tests/test_probe185_recon.py::_STD_ON_THEO` | MHA s0 8042.8 → **8310.7**；GQA s0 7562.8 → **8046.7**；两个 s1 **未动** | s0 峰值事件易主 `bwd@0`；末 stage 无 embedding。四点仍全部 `sim < real`（框架缺口不变量未动） |
| `tests/test_pp4_recompute_anchor.py::THEO_MTP` | s3 28291.7 → **29789.8**；s0-s2 **未动** | MTP 层内共享 embedding op → 层内 `max(730.0, 2228.003)`，净增 1498.0 |
| `tests/test_x4_p2p_pp.py::test_pp2_..._anchor` | s0 10458.1 → **10573.0**（峰值事件 `bwd@4` → `bwd@0`）；s1 **未动** | 该 config 自己的 profiler 量到的 1067.753 MiB。s0 仍 OOM-安全（≥10246.0 的硬断言原样保留） |
| `scorecard_anchors.py` pp2-stage0 | ratio 1.021 → **1.032**，**band `(1.00, 1.06)` 未动** | 1.032 仍在带内；**刻意不放宽上界**，让继续漂移必须触红 |
| `tests/test_pp4_recompute_anchor.py::_PP8_OVER_BAND` | **未动** `(1.00, 1.16)` | pp8 逐 stage 逐位不变（各 stage 无 embedding 伪层） |
| `tests/test_acceptance_gate.py` 全部 golden | **未动** | 八跑 32 格逐位不变 |

**没有删除任何一条不变量**；新增用例守的是本轮新引入的机制，照 O4 `norm_kind` 的先例。

---

## 8. 验收

```
python -m pytest tests -q   →  1949 passed, 268 warnings
```

基线 **1940**（`228582d`）+ **9 道新守卫门**（`tests/test_emb_bwd_kernel_workspace.py`）。
八跑门 32 格 + 两条聚合 + `REAL_SHA256` + 四条 ×1 不变量：**全部 PASS 且逐位不变**（§5）。

复现：

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=. python sim_vs_real_report.py                 # 记分卡
PYTHONIOENCODING=utf-8 PYTHONPATH=. python tools/liveness_ab_validate.py --grad-mode chain2 --deltas
PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_head_ws_events.py     # item-2 事件线判据
PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_emb_bwd_gap.py        # bwd@0 距峰
PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_anchor_deltas_emb_ws.py
python -m pytest tests -q
```

---

## 9. 仍是 OOM-不安全的锚点（本轮之后）

**本项抬起来的**：`pp2-stage0` 本就安全（1.021 → 1.032，更安全）；
185 std ON s0 两点与 pp4+MTP s3 属「框架缺口」档（`sim < real` 是该门的**不变量**，非记分卡口径），缺口收窄。

**仍然 OOM-不安全（一个都没被本项改善，原因见 §2.4）**：

| 锚点 | ratio | 峰值事件 | 本项为何够不着 |
|---|---:|---|---|
| `DSv4 mHC(x4)+MTP` | 0.891 | `bwd@6` = lm_head | `bwd@0` 差峰 9332.9 MiB |
| `DSv4-fused (base)` | 0.902 | `bwd@5` = lm_head | 差 5458.3 MiB |
| `std 116 gqa pp2 s0` | 0.852 | `bwd@4`（解码层） | 该 stage 的 `bwd@0` 不是峰 |
| `std 116 mha pp2 s0` | 0.934 | `bwd@4` | 同上 |
| `select mlp (keep-attn)` | 0.932 | `bwd@9` = lm_head | 差 6867.2 MiB |
| `select self_attn` | 0.970 | `bwd@9` | 差 10427.1 MiB |
| `DSv3 8L none (dp2)` | 0.981 | `bwd@9` | 差 11762.7 MiB |
| `cp2-none` | 0.991 | `bwd@9` | 差 12104.3 MiB |
| `pp2-stage1` | 0.999 | `bwd@9` | 末 stage 无 embedding |
| pp4 OFF s0-s3 | 0.892 / 0.867 / 0.815 / 0.832 | 解码层 bwd | `bwd@0` 差峰 4179.9 |
| pp4 ON s0-s3 | 0.668 / 0.746 / 0.790 / 0.894 | 解码层 bwd | 差 2780.4（**只差 552.4 就够到**） |
| pp8 s0 / s3 / s7 | 0.827 / 0.963 / 0.956 | 解码层 / head | s3、s7 所在 stage 无 embedding |
| 185 P3-P / F0 / F 每层 / U1 | 0.686 / 0.844 / 0.754 / 0.966 | — | 峰不在 `bwd@0` |

**下一步该测什么（按能抬起多少排序）**：
1. **`lm_head` / loss 段的反向 kernel workspace** —— 直接命中上表前 9 行的峰值事件。
   做法与本轮同：`MS_ALLOC_CONF=memory_tracker:True` 跑一次**末 stage** 有 loss 的 config，
   在 `bwd@head` 窗口里取 BWD 相位单 kernel 极大值（本轮的 `Erfinv` 标记法可直接复用）。
2. r0 / r128 的反向 workspace（上一轮 §8 ②）—— 抬 pp4/pp8 的解码层 stage。
3. ~~前向 mHC pre-sinkhorn 的 304.001 MiB（上一轮 §8 ③）。~~ **已于 2026-07-30 建模完成**（[`mhc_fwd_workspace_2026-07-30.md`](mhc_fwd_workspace_2026-07-30.md)）；实测落位逐字节，但**没有抬起任何锚点**——它的峰全在 BWD 侧。本条因此**不再是**「能抬起多少」清单上的候选，第 1 条（lm_head/loss 段反向）仍是首位。
