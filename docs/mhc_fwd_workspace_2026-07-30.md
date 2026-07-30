# 前向 mHC pre-sinkhorn kernel workspace：实测 → 建模（2026-07-30）

> 承接 [`kernel_workspace_2026-07-29.md`](kernel_workspace_2026-07-29.md) §8 ③ ——
> 那一轮把 **304.001 MiB**（融合 mHC `npu_mhc_pre_sinkhorn` 的前向 workspace，每层 2 次，
> 三个层型同值）列为「已测、已归属，但它的 op 在 `cost_eval/layers/residual.py`，
> **本轮不属本人可改范围**」的交接项，并注明「模型当前 fwd `workspace` 只有 16.0 MiB
> （`_fa_workspace`），**欠读 288 MiB/层**」。本轮把它建进模型。
> 它是 kernel-workspace 那条线里**最后一笔已测而未建模**的量
> （另两笔已分别由 [`kernel_workspace_2026-07-29.md`](kernel_workspace_2026-07-29.md)
> 与 [`head_workspace_2026-07-30.md`](head_workspace_2026-07-30.md) 建完）。
>
> **纪律不变**：绝不编造真机数；绝不发明拟合常数；「我跑了并观察到」与「我推断」分开写。
> **无 NPU 访问**，一切真机对照均用已存档实测数。`REAL_*` / `CSV_*` 常数与 `REAL_SHA256`
> 指纹**一个字节未动**（§7 有过滤 diff 佐证）。

复现：

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_mhc_fwd_ws.py         # 结构可达性 + before/after
PYTHONIOENCODING=utf-8 PYTHONPATH=. python scratchpad/probe_mhc_fwd_ws_ledger.py  # 逐锚点台账（§4）
PYTHONIOENCODING=utf-8 PYTHONPATH=. python tools/liveness_ab_validate.py --grad-mode chain2 --deltas
PYTHONIOENCODING=utf-8 python -m pytest tests -q
```

---

## 0. 一句话结论

| | 结论 |
|---|---|
| **建模** | 实测值**逐位落位**：三个层型的层内 fwd `workspace` 由 `16.000 / 64.000 / 64.000` 变成 **`304.001 / 304.001 / 304.001` MiB**，与真机逐（层，相位）窗口 `max_ws`（`kernel_workspace_2026-07-29.md` §4.2）**三个层型同值**这一点逐位吻合。挂在两个 `*_hc_pre_sinkhorn` op 上（= 实测「同层前向窗口内出现两次」），**只在融合分支**。 |
| **锚点** | **一个都没动。** 量到事件线的 **36 个**锚点/stage，**峰值事件全部是 BWD 事件，没有一个是 FWD**。前向侧的项结构上顶不到它们。融合-mHC 锚点里**前向头寸最小的是 `pp4 OFF s0`，还差 4015.1 MiB**；本项在那儿只加 240.0 MiB。 |

**没有为了让锚点动而把它挪到反向 op 上。** 任务书预判的这条分叉成立，且比
[`head_workspace_2026-07-30.md`](head_workspace_2026-07-30.md)（14 个里 13 个不动）更彻底：
这次是**全部锚点、八跑门 32 格、两条聚合，一个字节都没动**（整份输出逐行 diff 为空）。

---

## 1. 建模对象：一笔实测量，以及它的精度边界

### 1.1 测量（不是我跑的 —— 2026-07-29 那一轮在 167 上跑的，本轮只做建模）

`docs/kernel_workspace_2026-07-29.md`，MS2.10 / CANN9.1 / 910B2，
`MS_ALLOC_CONF="memory_tracker:True"`，run `c` 家族（fused / 无重算 / L8 / m4 /
pp4·dp2·ep2，b=1、tp=1、cp=1、H=4096、n=`num_residual_streams`=4）：

| 出处 | 逐字 |
|---|---|
| §4.1 逐相位表 | **FWD** 相位单 kernel workspace 极大值：rank0（stage0 = r0+r4）**304.001 MiB**、rank2（stage1 = r128+r4）**304.001 MiB** |
| §4.2 逐（层，相位）窗口表 | layer0 r0 dense 12 个窗口 `max_ws` 中位 **304.001**；layer1 r4 moe 12 个窗口 **304.001** |
| §4.3 归属原文 | `t=29631 >>> WORKSPACE 304.00 MiB`，紧跟在一批「名义 `Concat`」的输出 `0.000/32.000/0.063/0.250/0.063/0.375/0.016/2.500/10.000 MiB` 之后 |
| §4.3 | 同一模式在**该层窗口内出现两次**（t=29631 与 t=29922，后者紧接 dense FFN 的 `RmsNorm→MatMulExt→SplitWithSizeView→SiLU→Mul→MatMulExt`）→ 每层两次，每次 304.001 |

**归属证据**：tracker 的 `node_name` 对自定义融合算子是「上一条 pyboost 任务」的**陈旧标签**
（该文 §1.3 —— 融合算子走 pyboost 之外的 launch 路径，根本不进 `task.csv`），故只能按
**输出签名**反推。上面那 8 个输出逐张对上 `_fused_hc_ops` 里逐字声明的
`npu_mhc_pre_sinkhorn` 输出/ctx 集（`h_in` 32.0 / `h_post` 0.0625 / `h_res` 0.25 /
`h_pre` 0.0625 / `hc_before_norm` 0.375 / `inv_rms` 0.0156 / `sum_out` 2.5 / `norm_out` 10.0，
源 `custom_op_impl.py:390-391` + `mhc_pre_sinkhorn.cc:24-50`）。

> **这是推断，不是 tracker 直接告诉我们的**（与该文 §8 ⑦ 同一条限制）。支持它的是
> **8 张输出逐张字节吻合** + **每层恰好两次、第二次紧邻 FFN** 这两条独立特征。

### 1.2 ⚠ 测量点数：两处记载**不一致**，本文按更弱的那条用

| 出处 | 逐字 |
|---|---|
| `kernel_workspace_2026-07-29.md` §8 ③ | 「S=4096/2048 **两点**确认 S 无关」 |
| 该文 §9 的复现命令 | 三次跑：`c_ws`(S=4096) / `s2048_ws` / `s1024_ws` |
| 本轮任务交接书 | 「304.001 MiB at S=1024, 2048 and 4096 → S-independent」 |

两处**都判 S 无关**，只在点数上不一致（2 点 vs 3 点）。本项建成**常数**，
故两种读法**结果等价**——按更弱的那条（至少 2048/4096 两点）也足以支持「不含 S」。
**如实记下这条不一致**，不替任何一方补话。

### 1.3 字节值 318768128：观测与推断的分界

- **观测**：tracker 报的是**三位小数**的 `304.001 MiB`。它不是整字节数
  （`304.001 × 2²⁰` 不是整数），所以「实测值」本身带 ±512 B 量级的显示精度。
- **推断**：本站点的分配器块尺寸**全是 512 B 的整数倍**——仓内既有采集的直证：
  `analysis/realmachine/pp2_norecomp/op_816362.csv` 的 `Size(KB)` 列取值形如
  `1093379.0` / `904960.5` / `16513.5` / `256.5`（半 KB 粒度）；
  [`head_workspace_2026-07-30.md`](head_workspace_2026-07-30.md) §1.3 逐字节测到的 `+3072 B`
  尾也是 512 的倍数。
  在 `304.001` 的显示窗口 `[318767628, 318768676)` 里，形如 `304 MiB + k·512 B` 的取值**唯一**：

  ```
  318767104 (= 304 MiB) + 1024 = 318768128 → 304.0009765625 → 显示 "304.001"  ✓
  318767104 +  512 = 318767616 → 304.0004882812 → 显示 "304.000"  ✗
  318767104 + 2048 = 318769152 → 304.0019531250 → 显示 "304.002"  ✗
  ```

- **残余不确定度 ≤ 1.5 KiB = 0.0005 %**，抬不动任何一个锚点。取该值的理由是让模型
  **逐位复现实测的三位小数显示**（304 MiB 整会显示成 `304.000`，与实测不符）。

守卫门相应做成**两层精度**（§3.3）：显示级按 `f"{MiB:.3f}" == "304.001"` 比
（容忍 ±512 B = 测量自身的精度），字节级另有两道 pin 挡任何编辑。

---

## 2. ⚠ 关键判据：前向侧的项**结构上**够得着哪些锚点

任务书要求先判这一条、并且**不许为了让锚点动而把项挪到反向 op 上**。

### 2.1 通道的相位边界（源判）

`cost_eval/mem_timeline.py:583-585`（FWD 事件）：

```python
B.workspace = sm.workspace
rec(f"fwd:{lid}", ev_mb, ev_chunk)
B.workspace = 0
```

即 **只有 `fwd:<lid>` 这一枚事件带 `Buckets.workspace`**；`fwd_end`（`:625`）在清零之后，
BWD 事件（`:659`）带的是**另一条通道** `sm.bwd_workspace`。
所以「本项还差多少才能顶翻某锚点的峰」= `peak − max(fwd:<lid>)`，
**不是** `peak − max(全部前向事件)`。

### 2.2 实测的事件线：**36 个锚点/stage，36 个峰在 BWD，0 个在 FWD**

（我跑了并观察到：`scratchpad/probe_mhc_fwd_ws_ledger.py`，逐 stage 走 `record_timeline=True`。）

| 锚点族 | 峰值事件 | 融合 mHC？ | **前向头寸**（peak − max `fwd:<lid>`，MiB） |
|---|---|---|---:|
| pp4 ON s0/s1/s2/s3 | `bwd@2` / `bwd@4` / `bwd@6` / `bwd@9` | ✓ | 7282.1 / 6270.7 / 6270.7 / 10681.4 |
| **pp4 OFF s0/s1/s2**/s3 | `bwd@2` / `bwd@4` / `bwd@6` / `bwd@9` | ✓ | **4015.1** / 4143.1 / 4143.1 / 9154.0 |
| pp4 MTP s0–s3 | 同上 / `bwd@9`(MTP 层) | ✓ | 7282.1 / 6270.7 / 6270.7 / 16714.1 |
| pp8 s0–s7 | `bwd@1`…`bwd@9` | ✓ | 6217.2 – 11185.8 |
| 185 P3-P m8 s0 | `bwd@2` | ✓ | 6406.4 |
| 185 F0 4L / U1 4L / U2 8L | `bwd@5` / `bwd@5` / `bwd@9` | ✓ | 6066.8 |
| 185 std ON kv32/kv8 s0 | `bwd@0`(embedding) | ✗（`hc=1`） | 2092.0 |
| 185 std ON kv32/kv8 s1 | `bwd@9`(lm_head) | ✗ | 8624.1 / 8576.1 |
| 记分卡 DSv3 族 6 条 | `bwd@9`(lm_head) ×5、`bwd@0` ×1 | ✗ | 1245.1 – 35393.8 |
| 记分卡 DSv4 两条 | `bwd@5` / `bwd@6`(lm_head) | ✗（**非融合**，见下） | 4680.3 / 4693.9 |

上表的前 8 行（**28 个**锚点/stage：pp4×3 组 12 + pp8 8 + P3-P 1 + 185 F/U 3 + 185 std 4）
由 `probe_mhc_fwd_ws_ledger.py` 量；后 2 行（记分卡族 **8 条**：DSv3 8L none / select self_attn /
select mlp / pp2-stage0 / pp2-stage1 / cp2-none / DSv4-fused base / DSv4 mHC+MTP）由
`probe_mhc_fwd_ws.py` 的最后一节量。合计 **36 个量到事件线的锚点/stage，36 个峰在 BWD、
0 个在 FWD**。

其余 12 条（116 std 六点 + 记分卡 `DSv3 4L full`/`8L full`/`4L full ep=2`/`cp2 colossal`/
`cp2 ulysses`/`select both` 六条）本轮**没有单独量事件线** —— 不需要：
它们**无 mHC**（`hc=1` / DSv3），本项在它们身上结构上恒 0，与峰在哪无关。

**判决**：

1. **没有任何一个锚点的峰值事件是前向事件**（36/36 在 BWD）。
2. 融合-mHC 锚点里前向头寸最小的是 **`pp4 OFF s0` = 4015.1 MiB**；本项在那个 stage 的
   逐层增量是 **240.0 MiB**（层内 `max` 由 MoE staging 的 64.0 换成 304.001）。
   **差 3775.1 MiB**，量级差一个数量级 —— 不是「差一点点」。
3. 记分卡两条 DSv4 锚（`DSv4-fused base` 0.906 / `DSv4 mHC(x4)+MTP` 0.894）**根本不走本分支**：
   它们的真机跑容器 vendor OPP 里没有 `aclnnMhcPreSinkhorn`，mHC 走**非融合**
   （三处独立源，[`fused_mhc_branch_mismatch_2026-07-30.md`](fused_mhc_branch_mismatch_2026-07-30.md) §2.2），
   `scorecard_anchors._dsv4_sim` 显式传 `use_fused_mhc=False`。本项对它们**恒 0**，
   与「够不够得着」无关。
4. 116 std / 185 std / DSv3 族**没有 mHC 模块**（`hc=1` → `mhc_wrap` no-op）→ 恒 0。

> **所以本项没有改善任何一个 OOM-不安全锚点，而这是正确结果、不是建模疏忽。**
> 要抬它们，需要的是**它们各自峰值事件所在那个 kernel** 的 workspace ——
> 主要是 `lm_head`/loss 段的反向（至今未测、留 0，见 §6 ①），
> 那是**另一次测量**，不是把本项挪过去。

### 2.3 它**没有**从 `forward_max_live` 那条侧门漏进反向（量到了余量）

`cost_eval/structure_mem.py:200` 逐字 `peak = max(peak, live + op.workspace_bytes)`
—— 即 fwd workspace **确实**参与 `forward_max_live`，而后者又喂
`recomp_scratch = max(0, fml − ci)`（全重算）与 `bwd_working_set = max(0, fml − bwd_scratch)`
（无重算），两者都落在 **BWD 事件**上。这是这条通道**既有**的性质（`_fa_workspace`、
MoE staging、DSA KL workspace 一直如此），本轮**没有改动它**。

那么本项有没有经这条侧门抬高反向？**实测：没有，而且量出了余量。**
（我跑了并观察到，逐 op 打印 `live + ws`：）

| 层型 | `fml` before | `fml` after | 设定 `fml` 的 op | 两个 pre_sinkhorn 处 `live + ws` | **距 `fml` 的余量** |
|---|---:|---:|---|---:|---:|
| `dsv4hyb_r0_dense` | 544.2 | **544.2** | `q_hnorm`（live 544.2, ws 0） | 128.1 + 304.001 = 432.1 | **112.2 MiB** |
| `dsv4hyb_r4_moe` | 552.2 | **552.2** | `q_hnorm`（live 552.2, ws 0） | 128.1 + 304.001 = 432.1 | **120.2 MiB** |
| `dsv4hyb_r128_moe` | 544.2 | **544.2** | `q_hnorm`（live 544.2, ws 0） | 128.1 + 304.001 = 432.1 | **112.2 MiB** |

pre-sinkhorn 是**层的第 0 个 / 第 16-18 个 op**，那一刻在世的只有打包残差流本身
（`[S,B,n·H]` bf16 = 128.0 MiB）加零头，故 `432.1 < 544.2` —— **本项对 `fml` 逐字节无影响**，
`bwd_working_set` / `recomp_scratch` 一个字节不动。

> **这个余量值得单独记**：只要将来某笔前向 workspace 让某个 op 的 `live + ws` 超过
> 该层 `fml`（今天还差 **112.2 / 120.2 MiB**），这条侧门就会开始把前向瞬态送进反向断面。
> 那时**必须先判它对不对**（全重算下前向 kernel 确实重跑 → 进 `recomp_scratch` 是对的；
> 无重算下前向 kernel 不重跑 → 进 `bwd_working_set` 是 `fml` 当代理量的**副作用**，
> 方向上是过读 = OOM 安全侧）。本轮**没有碰**这条口径。

---

## 3. 建模：复用既有 **fwd** 通道，只加一个挂载点

### 3.1 改了什么（全部加法，未标注者逐字节不变）

| 文件 | 改动 |
|---|---|
| `cost_eval/layers/residual.py` | 新增 `_MHC_PRE_SINKHORN_FWD_WS = "318768128"`（含全部实测出处/精度/边界）；`_fused_hc_ops` 的 `{prefix}_hc_pre_sinkhorn` op 挂 `workspace=`；`_unfused_hc_ops` docstring 显式写明「本分支恒 0 = **未测**，不是已测为 0」 |
| `tests/test_mhc_fwd_kernel_workspace.py` | **新增 24 道门**（§3.3） |
| `scratchpad/probe_mhc_fwd_ws{,_ledger}.py` | 结构可达性探针 + 逐锚点台账 |

**通道一个都没新建**：`OpSpec.workspace` → `ResolvedOp.workspace_bytes` →
`StructureMemory.workspace`（层内 **max**）→ `Buckets.workspace`（**FWD** 事件）
—— 这条路从 `_fa_workspace`（flash softmax-LSE）与 MoE staging 那时就在。

**刻意不碰**的是 2026-07-29 那轮为反向新建的 `bwd_workspace{,_ref}`：
那是 **BWD 事件**的通道。两条通道同名不同相位，混用等于把一笔前向瞬态搬到反向断面上去。

### 3.2 为什么挂在这两个 op 上、为什么是常数

```python
OpSpec(f"{prefix}_hc_pre_sinkhorn", OpType.ELEMENTWISE, [streams], h_pre,
       params=[rms_w, proj_w],
       saves=[streams, h_pre, hc_before_norm, inv_rms, sum_out, norm_out],
       workspace=_MHC_PRE_SINKHORN_FWD_WS)          # ← 本轮
```

- **归属到 op**：测量把它定位到了**一个具体 kernel**（§1.1 的输出签名），模型就必须挂在对应的
  op 上。`prefix ∈ {attn, ffn}` → 每层**两个**载体，与实测「同层窗口内出现两次」对上。
  **没有把任何数字摊到层上，也没有为了让哪个锚点落位而调过一个字节。**
- **层内取 max 不是保守假设，是实测判据**：workspace 块寿命恒 **1 个 tracker tick**、
  峰值那一刻**至多一块在世**（`kernel_workspace_2026-07-29.md` §3 全部 4554 个块寿命直方图
  `[(1, 4554)]`、§4.1）。故 attn_hc 与 ffn_hc 两笔**不相加**，层内 = 304.001。
- **常数、走字符串通道**：实测 S 无关 → 表达式里不含 `S` → `shape_eval.py:255` 的
  「含 S 就 ÷cp」**结构上不触发**。这是刻意的：整体 ÷cp 会让 cp>1 欠读 = OOM-**不安全**。
- **只挂融合分支**：`_fused_hc_ops` 只在 `d.use_fused_mhc` 为真时被调用
  （`residual.py:192` `if getattr(d,"use_fused_mhc",False): _fused_hc_ops else _unfused_hc_ops`），
  故门控**是结构性的**，不需要另加 if。`use_fused_mhc` 这个旋钮本轮之前刚被补齐到扁平 query 路
  （[`fused_mhc_branch_mismatch_2026-07-30.md`](fused_mhc_branch_mismatch_2026-07-30.md)），
  两条路（yaml 导入 / 扁平 query）现在读同一个值。

### 3.3 新增的 24 道门（守的是**测量**，不是模型自洽）

`tests/test_mhc_fwd_kernel_workspace.py`：

| 组 | 门 | 守什么 |
|---|---|---|
| 值 + S 无关 | 3 层型 × 3 seq = **9 例** + 1 例 S 折 4 倍逐字节相等 | 实测值穿过整条链；挡「顺手改成 ∝S 的式子」（反向那笔**是** ∝S 的，两者不可混淆） |
| 归属 | 3 例（逐层型） | 恰好两个 `*_hc_pre_sinkhorn`、**各扛整份**（不是各一半、不摊层上）；既有载体（flash LSE 16.0 / MoE staging 64.0）逐字节未被波及 |
| 未测留 0 | 3 + 3 + 1 = **7 例** | 非融合分支无 pre_sinkhorn op、层内 fwd ws 退回 16.0/64.0、常数不出现在该分支；无 mHC 的模型里没有载体 |
| 相位不串 | 1 例 | 本项只占 `OpSpec.workspace`；`bwd_workspace{,_ref}` 必须为 `None`；同一 r4 层上 fwd 304.001 与 bwd 730.0 **各在各的桶里** |
| cp | 1 例（带**非空转自检**：cp=2 必须真把 `activation_saves` 切小） | 常数不吃 ÷cp（除掉即 cp>1 欠读 = OOM-不安全） |
| **重建口吞字段** | 2 例：`mhc_wrap` 一道 + **MTP 层单列一道** | 见下 |

**为什么 MTP 要单列一道**：`cost_eval/layers/head.py:285` 拿 `mhc_wrap` 的 **`wrapped[0]`**
做 `dataclasses.replace` 补一条 inputs 边 —— 而融合分支下 `wrapped[0]` **正是**
`attn_hc_pre_sinkhorn`，即本项的载体。这是「重建口静默吞字段」这条 bug class 的**第三个现场**
（前两次真的吞过：`norm_kind` 一次、`workspace_ref` 一次，见 `residual.py::_rebuild` 注释与
[`census_fix_residual_carrier_2026-07-29.md`](census_fix_residual_carrier_2026-07-29.md) §5b）。

**变异验证（我跑了并观察到，三次改完即还原）**：

| 变异 | 结果 |
|---|---|
| ① 常数 `318768128 → 318768640`（+512 B） | **只红 2 道字节门**；9 道显示门按设计容忍（±512 B = 测量自身精度） |
| ② 常数减半 | **红 14 道** |
| ③ `head.py:285` 回退成手写字段清单（丢 `workspace`） | **只有 MTP 那一道变红** —— 没有它这条回潮就漏网 |

---

## 4. 逐锚点 before → after

**全部实跑重算，非转抄**（`scratchpad/probe_mhc_fwd_ws.py` 在本树与 `git stash` 掉本轮
`residual.py` 后各跑一遍逐行 diff；`scratchpad/probe_mhc_fwd_ws_ledger.py` 出 ratio 表）。

### 4.1 判决：**全部锚点一个都没动**

`probe_mhc_fwd_ws.py` 的 before/after diff **只有一个 hunk、78 行**，全部落在
「逐层型 fwd workspace」与「best_fwd 那一列」上；**`peak=` 与 `@峰值事件` 两列一个字符没变**。

| 锚点 | real | before | **after** | 判 |
|---|---:|---:|---:|---|
| pp4 ON s0/s1/s2/s3 | 24153.3 / 14641.7 / 14097.7 / 23508.0 | 0.6725 / 0.8265 / 0.8402 / 0.8954 | **同左，逐 MiB 不变** | 峰在 `bwd@*` |
| pp4 OFF s0/s1/s2/s3 | 30395.0 / 21019.4 / 17799.9 / 27822.0 | 0.9089 / 0.9404 / 0.8738 / 0.8376 | **同左** | 同上（头寸 4015.1，本项 240.0） |
| pp4 MTP s0–s3 | 24153.0 / 14641.0 / 14100.0 / 39898.0 | 0.6725 / 0.8265 / 0.8401 / 0.7509 | **同左** | 同上 |
| pp8 s0–s7 | — | 0.8274 / 1.0554 / **1.0007** / **1.0698** / 1.0519 / 1.1244 / 1.1093 / 0.9556 | **逐位相同** | 同上 |
| 185 P3-P m8 s0 | 25343.5 | 0.6909 | **同左** | 同上 |
| 185 F0 4L / U1 4L / U2 8L | 26499.0 / 40194.0 / 56010.0 | 0.8536 / 1.1168 / 1.3719 | **同左** | 同上 |
| 185 std ON kv32/kv8 s0,s1 | — | 0.7466 / 0.9621 / 0.7487 / 0.9659 | **同左** | 无 mHC → 本项恒 0 |
| 116 std 六点 | — | 0.9343 / 0.9889 / 0.9903 / 0.8523 / 0.9846 / 0.9747 | **同左** | 无 mHC |
| 记分卡 14 条 | — | 0.9961 / 0.9925 / 0.9926 / 0.9993 / 0.9986 / 1.0319 / 0.9990 / 0.9907 / 0.9812 / 0.9696 / 0.9322 / 1.0005 / 0.9058 / 0.8944 | **逐位相同** | DSv3 族无 mHC；两条 DSv4 锚是**非融合** mHC（§2.2 第 3 条） |

**动的只有内部量**（不进任何锚点的峰）：逐层型 `StructureMemory.workspace`
`16.000 / 64.000 / 64.000 → 304.001 / 304.001 / 304.001`，以及各 stage 的
`fwd:<lid>` 事件断面（+288.0 于 r0 层、+240.0 于 r4/r128 层）。

### 4.2 边缘位（任务点名要看的）—— **原地不动**

| 位 | before | **after** | 说明 |
|---|---:|---:|---|
| **pp8 s2** | 1.0007（sim 11686.7 vs 真机 11678.0，高 **8.7 MiB**） | **1.0007，逐 MiB 不动** | 峰在 `bwd@3`，前向头寸 6217.2 MiB。**没有越界。** |
| **pp8 s3** | 1.0698（sim 12751.2 vs 11919.0） | **1.0698，逐 MiB 不动** | 峰在 `bwd@4`，前向头寸 7239.2 MiB |

**本轮没有把任何一个锚点从欠读推过 1.0，也没有把任何过读推得更过读。**（一个字节没动。）

---

## 5. 八跑验收门 before → after

`PYTHONIOENCODING=utf-8 PYTHONPATH=. python tools/liveness_ab_validate.py --grad-mode chain2 --deltas`
（before = `git stash push -- cost_eval/layers/residual.py` 后跑）：

| 指标 | before | **after** |
|---|---|---|
| bucket 聚合 mean / min / max（n=28） | 0.841 / 0.703 / 1.038 | **0.841 / 0.703 / 1.038** |
| hand_spec·chain2 mean / min / max（n=28） | 0.860 / 0.675 / 1.110 | **0.860 / 0.675 / 1.110** |
| 32 格逐格 sim/real（8 跑 × 4 stage） | — | **逐位不变** |
| `unfused − fused` delta 全表 | — | **逐位不变** |
| I1/I2 **真机** ×1（层数 / 微批数） | PASS / PASS | **PASS / PASS** |
| I1/I2 **模型** ×1（bucket & hand_spec，共 4 条） | PASS ×4 | **PASS ×4** |
| `REAL_SHA256` 指纹 | `41e279e591ae4ae9…` PASS | **同值 PASS** |
| **整份输出** | — | **`diff` 为空（逐行相同）** |

**为什么八跑门一格不动**：八跑各 stage 的峰值事件同样都在 BWD 上
（bucket 路与锚点路共用 `mem_timeline`）；`hand_spec` 那条 liveness 路**确实读**
`workspace_bytes`（`cost_eval/liveness/graph.py:244` → `simulate.py:597` 逐 op 采样），
但它的峰同样不落在这些采样点上 —— 故两条来源都逐位不变。

---

## 6. 诚实清单：本测量**不**支持什么

| # | 事项 | 状态 |
|---|---|---|
| ① | **`lm_head` / loss 段自己的反向 kernel workspace** | **未测 → 留 0，已知欠读。**这是当前**最要紧的待测项**：§2.2 量到事件线的 36 条里，峰值事件落在 **lm_head 反向**（8L 模型 `bwd@9` / 4L 模型 `bwd@5` / 4L+MTP `bwd@6`）的有 **15 条**，其中 **13 条 OOM-不安全**（ratio<1；另 2 条 `185 U1` 1.117 / `185 U2` 1.372 在过读侧）。本轮**没有也不该**用本项去补它（相位、kernel、量级都不同）。已由 `tests/test_bwd_kernel_workspace.py::test_default_is_zero_and_byte_neutral_for_untagged_specs` 钉住那一格为 0。<br>**→ 2026-07-30 已闭合，见 [`head_loss_bwd_workspace_2026-07-30.md`](head_loss_bwd_workspace_2026-07-30.md)。**本文这里预判的「15 条峰在 lm_head 反向、13 条 OOM-不安全」正是那一轮的目标；实测落位后 12 条被抬起。 |
| ② | **非融合 mHC 分支的前向 workspace** | **未测 → 留 0，已知欠读。**非融合走 `rms_norm/matmul/sinkhorn` 小算子链（`hyper_connection.py:246-301`），**根本不是这个 kernel**；167 那次跑的是 fused 配置。**不拿融合分支的数去顶**（照 r0/r128/unfused 反向 workspace 留 0 的先例）。影响面：记分卡两条 DSv4 锚（0.906 / 0.894）走的正是这条分支。 |
| ③ | **B / n（`num_residual_streams`）/ H / `mhc_sinkhorn_iterations` 四条轴** | **一律未扫**（只有 B=1 / n=4 / H=4096 / 站点默认迭代数**这一个点**）。本项是常数 → **不承诺**随它们变化。若真值随某维增长则更大配置上欠读，反之过读。**今天全部走融合分支的锚点（pp4 / pp8 / MTP / 185 F·U 相位）都恰在这一个点上**，故没有任何在用锚点依赖外推。 |
| ④ | **S 轴的点数** | 记载不一致（§1.2）：该文 §8③ 说 2 点、§9 与交接书说 3 点。两者都判 S 无关，本项按更弱的那条用。 |
| ⑤ | **cp 轴** | 结构上正确（常数不 ÷cp），但 cp>1 **未实测**。已有守卫门钉住「不被 cp 除」这个方向（除掉即 OOM-不安全）。 |
| ⑥ | **tp / sp 轴** | 字符串通道不切分 → tp>1 不 ÷tp = **过读 = OOM 安全侧**；真值未测。今天所有走本分支的锚点 tp=1。 |
| ⑦ | **字节值的最后 1 KiB** | 观测只到三位小数；`+1024 B` 是基于分配器 512 B 块粒度的**推断**（§1.3）。残余 ≤ 1.5 KiB。 |
| ⑧ | **全重算下这笔 workspace 会不会在反向重跑时再出现** | 模型口径上**会**（`forward_max_live` 含 op workspace → `recomp_scratch`），但今天**恰好没生效**（§2.3：还差 112.2 / 120.2 MiB 才够到 `fml`）。真机上 run `c` 是**无重算**跑，**没有采集重算相位的 FWD workspace**。 |
| ⑨ | 归属本身 | 是**输出签名反推**，不是 tracker 直接给的（§1.1 末）。要直接归属需 CANN 侧 profiler 的 kernel 名。 |
| ⑩ | 上几轮遗留 | `kernel_workspace_2026-07-29.md` §8 ①（逐层驻留欠读 / indexer 链）与 ②（r0/r128 的**反向** workspace）本轮**一条都没动**，仍未闭合。 |

---

## 7. `REAL_*` / `CSV_*` / 指纹未动的 diff 级证明

**判据（照上一轮 [`head_workspace_2026-07-30.md`](head_workspace_2026-07-30.md) §7 的同款方法）**
是「有没有既有真机常数行被**删改**」——即只看 `-` 侧：

```bash
$ git diff 51d98c3 HEAD -U0 -- . ':(exclude)docs' ':(exclude)scratchpad' \
    | grep -E '^-' | grep -v '^---' | grep -Ei 'REAL|CSV|MEASURED|sha256'
# → 空（零命中）。**没有任何既有真机常数行被删改。**
```

`+` 侧另有 **6 处命中**，逐条如实列出：全部在**新文件**
`tests/test_mhc_fwd_kernel_workspace.py` 里，全部是本轮**新增测量**
`REAL_MHC_PRE_SINKHORN_FWD_WS_DISPLAY = "304.001"`（1 处定义 + 5 处引用），
**不替换任何旧值** —— 与上一轮新增 `REAL_EMB_GATHER_DGRAD_WS_MiB` 同款。

本轮改到的文件只有两个（`git diff --stat`）：

```
 cost_eval/layers/residual.py           |  74 +++++++-
 tests/test_mhc_fwd_kernel_workspace.py | 336 +++++++++++++++++++++++++++++++++
```

（`residual.py` 那 1 行删除 = 被改写的那条 `OpSpec(...)` 语句本身，不是常数。）
八跑门的 `REAL_SHA256` 校验也在 §5 里 PASS（同值 `41e279e591ae4ae9…`）。

---

## 8. 重钉台账（old → new，逐条理由）

**本轮一条锚点都没有重钉 —— 因为一条都没动。**

| 位置 | old → new | 理由 |
|---|---|---|
| `cost_eval/layers/residual.py::_fused_hc_ops` | 无 → 挂 `workspace` | 167 真机 memory-tracker 实测，归属由输出签名逐张反推（§1.1） |
| `cost_eval/layers/residual.py::_unfused_hc_ops` | 无 → **docstring 显式记「留 0 = 未测」** | 防止后人把融合分支的实测数拿来顶（§6 ②） |
| `tests/test_mhc_fwd_kernel_workspace.py` | **新增**（24 例） | 新机制的守卫门，见 §3.3 |
| `tests/test_pp4_recompute_anchor.py`（`THEO_ON` / `BAND_OFF` / `THEO_MTP` / `THEO_PP8` / `_PP8_UNDER` / `_PP8_OVER_BAND`） | **未动** | 逐 stage 逐 MiB 不变（§4.1） |
| `tests/test_probe185_recon.py` / `tests/test_x4_p2p_pp.py` / `tests/test_std_attn_anchor.py` | **未动** | 同上 |
| `scorecard_anchors.py` 14 条 ratio 与 band | **未动** | 同上；两条 DSv4 锚走非融合分支，本项对它们恒 0 |
| `tests/test_acceptance_gate.py` 全部 golden（32 格 + 两条聚合） | **未动** | 八跑门整份输出逐行 diff 为空（§5） |
| `REAL_*` / `CSV_*` / `REAL_SHA256` / 两条真机不变量 / 模型 ×1 不变量 / `nr_moe_frag_factor` / `kept_frag_factor` | **一个字节没动** | §7 |

**没有删除任何一条不变量、没有删除任何用例**；新增用例守的是本轮新引入的机制，
照 `docs/opdag_walker_core_2026-07-25.md` §6.6 的先例。
（该先例里「只移动举例、保留不变量」的那一半本轮**没有用到** —— 没有任何举例值需要移动。）

---

## 9. 验收

```
$ PYTHONIOENCODING=utf-8 python -m pytest tests -q
2010 passed, 268 warnings in 99.99s (0:01:39)
```

基线 **1986**（`51d98c3`）+ **24 道新守卫门**。八跑门 32 格 + 两条聚合 + `REAL_SHA256`
+ 四条 ×1 不变量：**全部 PASS 且逐位不变**（§5）。

---

## 10. 本轮之后仍是 **OOM-不安全** 的锚点（模型欠读真机）

**本项一条都没改善**，原因见 §2.2。**不调参掩盖。**

| 锚点 | ratio | 峰值事件 | 本项为何够不着 |
|---|---:|---|---|
| pp4 ON s0 | 0.673 | `bwd@2`（解码层） | 前向头寸 7282.1 ≫ 本项 240.0 |
| pp4 ON s1 / s2 / s3 | 0.827 / 0.840 / 0.895 | `bwd@4` / `bwd@6` / `bwd@9`(lm_head) | 头寸 6270.7 / 6270.7 / 10681.4 |
| pp4 OFF s0 / s1 / s2 / s3 | 0.909 / 0.940 / 0.874 / 0.838 | `bwd@2` / `bwd@4` / `bwd@6` / `bwd@9` | **头寸最小的一组**：4015.1 / 4143.1 / 4143.1 / 9154.0 |
| pp4 MTP s3 | 0.751 | `bwd@9`（MTP 解码层） | 头寸 16714.1 |
| pp8 s0 / s7 | 0.827 / 0.956 | `bwd@1` / `bwd@9`(lm_head) | 头寸 9439.5 / 11185.8 |
| 185 P3-P m8 s0 | 0.691 | `bwd@2` | 头寸 6406.4 |
| 185 F0 4L | 0.854 | `bwd@5`(lm_head) | 头寸 6066.8 |
| 185 std ON kv32 s0 / s1 | 0.747 / 0.962 | `bwd@0`(embedding) / `bwd@9`(lm_head) | **无 mHC** → 本项恒 0 |
| 185 std ON kv8 s0 / s1 | 0.749 / 0.966 | 同上 | 同上 |
| 116 std mha pp2 s0 / s1 / pp1 s0 | 0.934 / 0.989 / 0.990 | — | 无 mHC |
| 116 std gqa pp2 s0 / s1 / pp1 s0 | 0.852 / 0.985 / 0.975 | — | 无 mHC |
| `DSv4-fused (base)` | 0.906 | `bwd@5`(lm_head) | 真机跑是**非融合** mHC → 本项恒 0 |
| `DSv4 mHC(x4)+MTP` | 0.894 | `bwd@6`(lm_head) | 同上 |
| `DSv3 8L none (dp2)` | 0.981 | `bwd@9`(lm_head) | 无 mHC |
| `select self_attn` / `select mlp` | 0.970 / 0.932 | `bwd@9`(lm_head) | 无 mHC |
| `cp2-none` / `pp2-stage1` | 0.991 / 0.999 | `bwd@9`(lm_head) | 无 mHC |
| `DSv3 4L full` 系列 / `cp2 colossal` / `cp2 ulysses` | 0.996 / 0.993 / 0.993 / 0.999 / 0.999 | — | 无 mHC |

**过读侧（OOM 安全）不动的**：`pp8 s1–s6`（1.001–1.124）、`185 U1 4L` 1.117、
`185 U2 8L` 1.372、`pp2-stage0` 1.032、`select both` 1.001 —— **一个字节没动**。

### 下一步该测什么（按能抬起多少排序，与上一轮的排序一致且更有据）

1. ~~**`lm_head` / loss 段的反向 kernel workspace**~~ —— **已于 2026-07-30 完成**（[`head_loss_bwd_workspace_2026-07-30.md`](head_loss_bwd_workspace_2026-07-30.md)）。上表里 **10 个 OOM-不安全锚点**的
   峰值事件就在那里（`bwd@9` / `bwd@5` / `bwd@6`）。做法与前两轮同：
   `MS_ALLOC_CONF=memory_tracker:True` 跑一次**末 stage 有 loss** 的 config，
   在 `bwd@head` 窗口取 BWD 相位单 kernel 极大值（`Erfinv` 标记法可直接复用）。
2. **r0 / r128 的反向 workspace**（`kernel_workspace_2026-07-29.md` §8 ②）—— 抬 pp4/pp8
   的解码层 stage（那是上表里头寸最小、峰在解码层 bwd 的那一批）。
3. **非融合 mHC 分支的前向 workspace**（本文 §6 ②）—— 只影响记分卡两条 DSv4 锚。
4. 逐层驻留欠读 / indexer 链（同上 §8 ①）—— 不是 workspace 口径。

---

## Related

- [`kernel_workspace_2026-07-29.md`](kernel_workspace_2026-07-29.md) —— 本项的测量出处（§4.1/§4.2/§4.3）与交接（§8 ③）
- [`head_workspace_2026-07-30.md`](head_workspace_2026-07-30.md) —— 同一条线的上一轮（word-embedding 反向 `GatherDGradV2`）；「项对、落点对、但够不着锚点」这条分叉的先例
- [`fused_mhc_branch_mismatch_2026-07-30.md`](fused_mhc_branch_mismatch_2026-07-30.md) —— `use_fused_mhc` 旋钮补齐那一轮；本项的门控就建在它上面，§2.2 也是记分卡两条 DSv4 锚走非融合的判据
- [`census_fix_residual_carrier_2026-07-29.md`](census_fix_residual_carrier_2026-07-29.md) §5b —— `mhc_wrap` 静默吞 `workspace_ref` 那次；本轮 §3.3 的两道重建口门守的是同一条 bug class
- [`opdag_walker_core_2026-07-25.md`](opdag_walker_core_2026-07-25.md) §6.6 —— 「保留不变量、只移动举例」的改测试规矩（本轮无需移动任何举例）
