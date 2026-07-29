# 符号轴结构恢复（2026-07-29）

> **性质**：实现 + 实测台账。**边做边记**（任务书要求）。
> **立项依据**：`docs/next_fix_adversarial_review_2026-07-28.md` §6 —— 该复核用**反事实实测**
> （`scratchpad/adv_probe_cf.py`）证伪了 `docs/next_fix_diagnosis_2026-07-28.md` 推荐的
> 单点修 `csa.py:485`：L1 跳过只从 42 降到 **36**（预言 ≤5），extracted L1 `activation_saves`
> 只从 1067.6 涨到 **1071.6 MiB**（预言 [3000,5000]），链条在 **11 行之后**（`csa.py:496`
> BMM `needs_axis_structure`）就断了。四个级联根里**三个**归约到同一项能力缺口。
> **基线**：`feat/unified-llm-modelspec` @ `f0d9773`，`python -m pytest tests -q` → 1864 passed。
>
> 记法：**[RAN]** = 我实跑并粘了输出；**[SRC]** = 逐字读源（带 `file:line`）；
> **[INFER]** = 推断，不是观测。铁律：不编造尺寸 / 轴 / 拟合常数；真机数字一律来自 `REAL`。
> **本文引用的每个探针都已进库**（`scratchpad/axis_probe_*.py`）—— 这是对上一轮
> 「诊断书的 `[RAN]` 证据不可复跑」（复核 §0）的直接纠正。

---

## 0. 结论先行

**做了什么**：给符号 shape 层加了一项能力 —— 让它能**携带并恢复轴结构**，而不只是元素数。
落地为**六条**互相咬合的改动（每条都是算子定义或已核验的事实，没有一条是猜）：

| # | 能力 | 落点 | 依据 |
|---|---|---|---|
| ① | **表达式规范化**（线性式抵消 / 整除式折叠） | `sym_shape.normalize_axis` | 恒等变形，值不变（测试钉住） |
| ② | **整除事实**：`ratio·(S//ratio) == S` | `sym_shape.DivFacts` + `to_resolved._Folder._div_facts` | 由 `DimTable.S % ratio == 0` **逐字核验**后才成立 |
| ③ | **permute / transpose 精确重排** | `shape_infer._permute` | `out.shape[i] = x.shape[dims[i]]`（算子定义）；轴序 walker 早就记了 |
| ④ | **squeeze 精确去轴** | `shape_infer._squeeze` | 只去**已证为 1** 的轴；证不出 → `~` |
| ⑤ | **切片带步长 / reshape 单缺维** | `shape_infer._slice` / `_reshape` | 步长整除可证才用；单缺维由**元素数守恒**唯一确定（= `-1` 的定义） |
| ⑥ | **`broadcast_to` 建真节点 + `arange(start, stop)`** | `construct_walker` + `shape_infer._constant` | 改变元素数的"视图"不能只做别名 |

**外加两条 fail-loud 守卫**（本轮实测抓到真错，见 §3）：
`_concat` 校验 `cat` 的前置条件（rank 相同 + 非拼接轴逐轴相等），
`_permute` 校验 `dims` 是 `range(rank)` 的一个置换。

**验收（对照 §1 基线）**：

| 量 | 基线 [RAN] | 现在 [RAN] | 验收线 |
|---|---:|---:|---|
| L1（unfused r0）跳过 op | **42** | **6** | ≤ 13 ✅ |
| L1（fused r0）跳过 op | 13 | **0** | — |
| r0 抽取侧 `activation_saves`（unfused） | 1067.6 MiB | **4496.8 MiB** | ≥ 5000 ❌（见 §6.3） |
| 级联根 `compressor.py:216` ×4 | 在 | **出表** ✅ | 出表 |
| 级联根 `compressor.py:233` ×3 | 在 | **出表** ✅ | 出表 |
| 级联根 `deepseek_v4_hybrid_attention.py:205` ×1 | 在 | **出表** ✅ | 出表 |
| 级联根 `csa.py:485` ×1（及其后继 `:496`） | 在 | **出表** ✅ | 出表 |
| 全局跳过 op（fused / unfused） | 502 / 1010 | **256 / 809** | — |

> **四个级联根全部出表。** 唯一未达的是 `r0 saves ≥ 5000` —— 实测 4496.8，
> 且 §6.3 逐张量证明**再往上加就是编**：抽取侧那 12 项与诊断书 §5 P2 按
> `bprop_rules.PIN` 独立推出的清单**逐项精确相等**（6 项该在的全在、6 项不该在的全不在）。

---

## 1. 基线 [RAN]

`scratchpad/axis_probe_baseline.py`（本轮新增，已进库）：

```
=== UNFUSED (run b) ===   GLOBAL n_nodes=3976 n_ops=1956 skipped=1010
  L1:dsv4hyb_r0_dense   ops=162 skipped= 42
  L2:dsv4hyb_r4_moe     ops=280 skipped=173
  L3:dsv4hyb_r128_moe   ops=212 skipped= 92
  级联根: x4 Elementwise compressor.py:216 needs_axis_structure
          x3 View        compressor.py:233 slice_bounds_unknown
          x1 IndexSelect csa.py:485        constant_shape_unknown
  逐层 saves: r0 1067.6 / r4 650.5 / r128 566.9 / lm_head 2052.0 MiB

=== FUSED (run a) ===     GLOBAL n_nodes=2568 n_ops=1564 skipped=502
  L1:dsv4hyb_r0_dense   ops=120 skipped= 13
  级联根: x4 compressor.py:216 / x3 compressor.py:233
          x1 View deepseek_v4_hybrid_attention.py:205 needs_axis_structure
```

**与复核 §3.1 的数字逐字一致**（1956/1010、42/173/92、三根、1067.6/650.5/566.9/2052.0）。

四个级联根的**逐字原貌**（`scratchpad/axis_probe_baseline.py` 的 ROOT SITES 段）：

```
compressor.py:216  Elementwise  needs_axis_structure
  in : __arg__i4:~((2·((2·B·S·v_head_dim)//(4·B·(S//4)))·B·(S//4))
                   +(2·((2·B·S·v_head_dim)//(4·B·(S//4)))·B·(S//4))):fp32
compressor.py:233  View  slice_bounds_unknown   index=':total_seq_len__i7:self.compress_ratio'
csa.py:485         IndexSelect  constant_shape_unknown
  in : kv_flat__i10:~((64+v_head_dim-64)·B·S):bf16
  in : flat_indices__i10:(128·S):int64
deepseek_v4_hybrid_attention.py:205  View  needs_axis_structure（split 要按 split_dim 换轴）
  in : core_out:~B·S·n_heads·(64+v_head_dim-64):bf16
```

这四行把「缺什么能力」写得很清楚：`64+v_head_dim-64` 没被归一、`4·(S//4)` 没被折回 `S`、
permute 一律退 `~`、切片不认步长。

---

## 2. 逐条改动与依据

### 2.1 表达式规范化（`sym_shape`，能力 ①）

新增 `normalize_axis(f, facts)` / `normalize_axes`：

* **线性式**按顶层带号项摊平归并 —— `64+v_head_dim-64` → `v_head_dim`。
  必须**递归摊平**（`_expand_linear`）：`sub`/`add` 会把线性式包成一个不可分的原子串，
  嵌套一层就藏住可抵消的项（`(v_head_dim-64)+64` 若不摊平，外层看到的是两个不同的 key）。
* **整除式**先规范化分子分母再试**精确消元**；约不干净时**原样重造**原子串
  （串形与 `floordiv` / `divide_expr` 逐字一致 ⇒ 什么都没折叠时规范化是恒等映射，
  报表不会无谓地抖）。
* **幂等**：渲染时过一遍 `_canon_sum`，与 `parse_axis` 是同一个不动点（测试钉住）。

连带修掉两个**渲染歧义**（都是本轮实测暴露的静默错）：

* `_is_atom_expr` 现在把**差式**也算作需要括号的原子。此前 `sub()` 产出的 `v_head_dim-64`
  进乘积后写成 `B·v_head_dim-64`，`consumer._sym_value` 的差式档按**最后一个** `-` 左右切，
  读成 `B·v_head_dim − 64`（真值 `B·(v_head_dim−64)`）。
* `_is_composite` 现在对"单个单元内部含顶层 `·`"也加括号。`floordiv` 造的
  `n_heads·v_head_dim//8` 是**一个**单元，裸着放进 shape 串
  `8·1024·n_heads·v_head_dim//8` 后，`parse_shape` 按 `·` 切会得到 **4 条轴**而不是 3 ——
  rank 凭空多一条。这个错此前**看不见**（没人查 rank）；本轮 `_permute` 的置换校验一上来就撞上了。

`consumer._sym_value` 相应改用同一条**括号感知的带号切分**（`_split_signed`），
删掉旧的 `rpartition("-")` 档（它对 `S-(a-b)` 会切在错的位置）。

### 2.2 整除事实（能力 ②）—— 核验，不假定

[SRC] `compressor.py:196-201`：

```python
cutoff = (sq // ratio) * ratio
if cutoff < sq:            # :197-199 —— 除不尽时**真的会截断**
    kv = kv[:cutoff]
n_compressed = cutoff // ratio
```

⇒ `ratio·(S//ratio) == S` **当且仅当** `ratio | S`。故：

* `DivFacts` 只存**已核验**的 `(n, term)`；
* `div_facts_from_values(pairs, values)` 按**实际值**核验，除不尽的进 `rejected`
  （不是事实、且**显式可见** → `Coverage.div_facts_rejected`）；
* `assert_divisible(n, term, values)` 是显式断言档，**不成立就 ValueError**
  （值查不到也 fail-loud —— 不许把"查不到"当"成立"）；
* `to_resolved._Folder._div_facts()` 用**本层自己**的 `compress_ratio` 与 `DimTable.S`
  生成，`ratio <= 1` 时**没有**任何事实。

效果 [RAN]：`compressor.py:216` 的输入串从
`~((2·((2·B·S·v_head_dim)//(4·B·(S//4)))·B·(S//4))+(…))`
折成 `~(2·B·S·v_head_dim)` —— 分母 `4·B·(S//4)` 折回 `B·S`，消元退化成精确的多重集差。

### 2.3 permute / transpose（能力 ③）

轴序**源里逐字写着**（`csa.py:472` 的 `(1,2,0,3)`、`:494` 的 `(0,2,1,3)`…），
walker 也**早就抠进 attrs**（`construct_walker.py:2295-2303` 的 `permute_dims`、
`:2254-2257` 的 `perm`）—— `shape_infer` 此前**根本没读它**，一律退 `numel_only`。
`unfused_compressed_sparse_attn` 函数体里有 **8 处** permute，整条主链因此拿不到轴结构。

守卫：`dims` 必须恰是 `range(rank)` 的一个置换，否则**退回既有的 `~` 档**并记账
（退回而不是整条放弃 —— permute 恒不改元素数，`~` 是准确表述，`?` 反而少给了已知信息）。

### 2.4 squeeze（能力 ④）—— 三档都是算子定义

已证 == 1 → 去掉（精确）；已证 **≠ 1**（纯整数且非 1）→ 源侧 squeeze 是 **no-op**，透传（精确）；
判不出（符号轴，可能是 1）→ `~`。**不许**拿"大概不是 1"把轴留着往下算。

### 2.5 切片带步长 / reshape 单缺维（能力 ⑤）

* `_slice` 支持 `":<stop>:<step>"`。长度 = `ceil(stop/step)`，只有 `step | stop` 时才等于
  `stop//step` ⇒ 整除性**必须可证**（系数整除，或已核验的整除事实），证不出就记账保 `?`
  （"除不尽时 floor 与 ceil 差 1 —— 差的那一格会静默变成一个错的张量尺寸"）。
  真源 `compressor.py:230-233`：`total_seq_len = n_compressed * self.compress_ratio` ⇒ 系数整除。
* `_reshape`：**恰好一个**目标维解不出时，它由元素数守恒**唯一确定** —— 与 `-1` 是同一条
  算子定义。真源 `csa.py:481-482` 的 `sk = kv_full.shape[0]`（下标取轴，不是元组解包，
  walker 没导出）。**两个以上**解不出仍退 `~`（那时确实无解）。
  每次用这条都往 `Coverage.declared_shapes` 记一条（带 `file:line` 与理由），**可见**。

### 2.6 `broadcast_to` 与 `arange(start, stop)`（能力 ⑥）

* [SRC] `csa.py:445/460` `mint.unsqueeze(matrix, 0).broadcast_to((batch_size, seqlen, W))`。
  `construct_walker._handle_chained_call` 此前把**任何**链式"视图"方法都只做别名（不建节点）。
  对 reshape/permute/transpose 无害（元素数恒不变），对 `broadcast_to` 是**静默丢批维**：
  实测 `flat_indices`（`csa.py:484`）因此解成 `128·S` 而非 `B·S·128` —— B=1 时数值恰好相同、
  **B>1 就整层少读一个 B 倍**。现在发真节点 + 记 `broadcast_shape`，由新的 `broadcast` 规则定形。
* [SRC] `csa.py:439` `mint.arange(1, seqlen + 1, dtype=mstype.int32)`：walker 此前只记
  `args[0]`，`argc=2` 的形态在 `shape_infer` 里无解 → `constant_shape_unknown`。
  现在逐个位置实参都记，`arange(start, stop)` → `(stop − start,)`（算子定义）。
  三实参形态（`step`）**不支持** —— 长度要向上取整，取整就是猜。

---

## 3. 本轮抓到的**真错**（守卫的价值）

规范化 + permute 精确化把大量此前藏在 `~` 后面的形状**暴露**出来，其中两处是真的错的：

**(a) `csa.py:818` 的 `cat`** [RAN]（`scratchpad/axis_probe_chain.py csa.py 760 825`）：

```
csa.py:818  View concat
      in  matrix__i3:S·128:int32              <- window_idxs，rank 2（批维丢了）
      in  compress_topk_idxs__i1:1·S·128:bf16 <- 按边桥接错位拿到的窗口分支，rank 3
      OUT topk_idxs__i1:S·(128+S)             <- topk 轴被算成 128+4096 = 4224
```

`cat` 要求各输入 rank 相同 —— 这里 2 vs 3，**证明**上游至少一处解错了。没有守卫时，
这个错的 `topk` 轴经 `csa.py:485` 的元素数守恒传导下去，r4 层 `activation_saves`
从 ~22 GiB 变成 **86127.6 MiB** [RAN]。加了 `concat_shape_mismatch` 守卫后整条拒绝，
r4 退回 617.5 MiB（**诚实的下界**，而不是一个大得离谱的假值）。

> 这正是任务书说的失败类：**一个错的轴静默产出一个错的张量尺寸**。
> 守卫的判据不是"看着不对"，而是**算子的前置条件**。

**(b) `deepseek_v4_hybrid_attention.py:287` 的 permute** [RAN]：置换校验报
`轴 [0, 2, 1] 不是 rank=4 的一个置换` —— 根因是 §2.1 末尾那个**渲染歧义**
（`n_heads·v_head_dim//8` 被 `parse_shape` 切成两条轴）。修掉渲染后该点自解。

---

## 4. 现在的覆盖度 [RAN]

`scratchpad/axis_probe_baseline.py`：

```
=== UNFUSED (run b) ===   GLOBAL n_nodes=4006 n_ops=2084 skipped=961
  L1:dsv4hyb_r0_dense   ops=236 skipped=  6      （基线 162 / 42）
  L2:dsv4hyb_r4_moe     ops=274 skipped=178      （基线 280 / 173）
  L3:dsv4hyb_r128_moe   ops=238 skipped= 81      （基线 212 / 92）
  级联根: x4 Elementwise compressor.py:216 needs_axis_structure
          x3 View        csa.py:747        concat_shape_mismatch
          x1 View        csa.py:506        no_input_shape
  逐层 saves: r0 4496.8 / r4 617.5 / r128 565.9 / lm_head 2052.0 MiB

=== FUSED (run a) ===     GLOBAL n_nodes=2568 n_ops=1832 skipped=368
  L1:dsv4hyb_r0_dense   ops=146 skipped=  0      （基线 120 / 13）
  L2:dsv4hyb_r4_moe     ops=262 skipped= 65      （基线 236 / 78）
  L3:dsv4hyb_r128_moe   ops=200 skipped= 36      （基线 154 / 59）
  级联根: x4 Elementwise compressor.py:216 needs_axis_structure
          x3 TopK        router.py:393     reduce_axis_unknown
  逐层 saves: r0 1677.2 / r4 1467.1 / r128 1173.2 / lm_head 2052.0 MiB
```

r0（unfused）逐张量 [RAN]（`scratchpad/axis_probe_saves.py 1`，前 10）：

```
  1024.0 MiB  kvo_bm__i10   (B·S)·128·v_head_dim              4b  csa.py:518
  1024.0 MiB  kv_bm__i10    (B·S)·v_head_dim·128              4b  csa.py:495
   512.0 MiB  q_bm__i10     (B·S)·n_heads·v_head_dim          4b  csa.py:494
   256.0 MiB  q             S·B·(n_heads·v_head_dim)          2b  deepseek_v4:240
   256.0 MiB  cg            8·(B·S)·(n_heads·(v_head_dim//8)) 2b  deepseek_v4:286
   128.0 MiB  scores__i10   B·S·n_heads·128                   4b  csa.py:496
   128.0 MiB  exp_scores    B·n_heads·S·128                   4b  csa.py:511
   128.0 MiB  aw_bm__i10    (B·S)·n_heads·128                 4b  csa.py:517
```

`128` = `csa_window_size`（r0 的 topk 轴），与 [SRC] `csa.py:449-461`
`get_window_topk_idxs(self.window_size, b, sq)` 逐字对上。
复核 §3.4 反事实里"一个都没出现"的 `q_bm`/`kv_bm`/`kv_g`·`kvo_bm`/`scores`/`exp_scores`/`aw_bm`
**现在全部在场**，且量级与诊断书 §5 P2 的逐张量预言（512 / 1024 / 1024 / 128 / 128 / 128）
**逐项对上**（本文这一档是 fp32，故 ×2）。

---

## 5. 第二轮：`compressor.py:216` 出表（`chunk` 的轴）

§4 的表里 `compressor.py:216 ×4` 还挡着。链路逐字追（`scratchpad/axis_probe_chain.py compressor.py 190 225`）：
`:203/:204` 的 reshape 现在**精确**解出 `S//4·4·B·(2·v_head_dim)`（整除事实生效），
`:209` 也精确 —— 断点在 `:212-213` 的 `_overlap_transform`，具体是它第一行
[SRC] `compressor.py:169` `tensor_prev, tensor_next = self.chunk(tensor, 2, dim=-1)`：

```
compressor.py:169  View chunk   in kv:S//4·4·B·(2·v_head_dim)   OUT tensor_prev:~(B·S·v_head_dim)
```

`_chunk` **只用了份数、把轴丢了**（压成"单轴 = 总积/k"）—— 而 `chunks`/`chunk_dim` walker
早就记了（`construct_walker.py:2286-2295`）。补上后逐段恢复，并与源侧 docstring 逐字对上：

```
chunk(dim=-1)  S//4·4·B·(2·v_head_dim) → S//4·4·B·v_head_dim
cat(dim=1)     → S//4·8·B·v_head_dim      （:159-160 docstring「[n_groups, 2*ratio, b, head_dim]」）
softmax(dim=1) → 同形
sum(dim=1)     → S//4·B·v_head_dim        （:184 docstring「[sq // compress_ratio, b, head_dim]」）
```

份数有**两条独立的源侧事实**（字面量实参 / 元组解包元数），不一致就 fail-loud；
轴长除不尽份数时退回"只知元素数"档（各份根本不等长，floor 值是猜）。

**四个级联根至此全部出表。**

---

## 6. 验收数字（全部 [RAN]）

### 6.1 覆盖度

| | fused（run a） | unfused（run b） |
|---|---|---|
| 基线 | 2568 节点 → 1564 op（跳过 **502**） | 3976 节点 → 1956 op（跳过 **1010**） |
| 现在 | 2568 节点 → **2056** op（跳过 **256**） | 4006 节点 → **2388** op（跳过 **809**） |

> unfused 的节点数 3976 → 4006（+30）是 `broadcast_to` 现在建真节点所致
> （每层 CSA 两处 × 相关层数），**不是**图变大了。

逐层（unfused）：L1 42→**6**、L2/L4/L6/L8(r4) 173→**140**、L3/L5/L7(r128) 92→**81**。
逐层（fused）：L1 13→**0**、r4 78→**37**、r128 59→**36**。

### 6.2 级联根表

| 基线 | 现在 |
|---|---|
| ×4 `compressor.py:216` `needs_axis_structure` | **出表** |
| ×3 `compressor.py:233` `slice_bounds_unknown` | **出表** |
| ×1 `csa.py:485` `constant_shape_unknown`（unfused）→ 其后继 `:496` | **出表** |
| ×1 `deepseek_v4_hybrid_attention.py:205` `needs_axis_structure`（fused） | **出表** |
| — | ×4 `csa.py:777`（unfused）/ ×4 `indexer.py:219`（fused）`constant_shape_unknown` |
| — | ×3 `csa.py:747` `concat_shape_mismatch` / ×3 `router.py:393` `reduce_axis_unknown` |
| — | ×1 `csa.py:506` `no_input_shape` |

### 6.3 r0 抽取侧 `activation_saves`：**1067.6 → 4496.8 MiB**（验收线 ≥5000，**未达**）

**但它是对的**，逐张量可核（`scratchpad/axis_probe_reconcile.py`）：

```
  layer total              4496.8 MiB over 31 tensors
  of which csa.py:*        3079.5 MiB over 12 tensors
  （诊断书 §5 P2 对 `unfused_compressed_sparse_attn` 函数体的净预言 = 3072 MiB）

  --- 诊断书 P2「应当出现」的 6 项：6/6 在场，且**逐项数值精确相等** ---
    kv_bm 1024→1024.0 · kvo_bm 1024→1024.0 · q_bm 512→512.0
    scores 128→128.0 · exp_scores 128→128.0 · aw_bm 128→128.0
  --- 诊断书 P2「应当**不**出现」的 6 项：6/6 不在场 ---
    uq_f32 / kv_gathered / uout_f32 / uout_pm / attn_weights / uscore1
```

函数体 **3079.5** vs 预言 **3072**，Δ = **+7.5 MiB**，且这 7.5 逐项可点名：
`flat_indices` 4.0（`csa.py:484`）+ `matrix` 2.0（`:455`）+ `sum_exp` 1.0（`:513`）+
`__arg__i8` 0.5（`:458`）+ `__arg__i6` 0.0（`:455`）。

> 这是本轮**最强的一条正确性证据**：不是"解出来了"，而是解出来的东西
> **逐张量等于源码 + `bprop_rules.PIN` 独立推出的那份清单**。
> 顺带**兑现了诊断书自己的 E3 判据**（`uout_f32`/`uq_f32`/`kv_gathered` 不在抽取侧 saves）——
> 复核 §3.4 说它"既没被证伪、也没被检验"，现在被**检验且通过**。

**为什么到不了 5000（不是能力问题，是"再往上加就是编"）**：
剩下 6 个跳过 op 全部级联自 `csa.py:506`（`attn_sink` 是 `Parameter`，
作为**自由函数实参**内联后形参没拿到声明形状），它们的字节量级是 1–2 MiB。
把 4496.8 抬到 5000 需要再加 ~503 MiB，而按 `PIN` 表**没有**任何一项该被加进来 ——
唯一能加的是"同名同尺寸的连续重绑被合成一个"那部分（§8-2），
那需要 SSA 版本感知的张量身份，是**另一项**能力，且带双算风险。

### 6.4 8 跑门（`tools/liveness_ab_validate.py --grad-mode chain2 --deltas --gate`）

```
  bucket       n=28  mean=0.946  min=0.748  max=1.394      （逐字不变）
  extracted    n=0  （该来源无任何可评分格：解不出图或全 OOM）
  hand_spec    n=28  mean=0.955  min=0.729  max=1.400      （逐字不变）
  PASS  REAL 指纹: sha256=41e279e591ae4ae9…                （未动）
  PASS  I1/I2 真机 ×1 / bucket ×1 / hand_spec ×1            （6 条全 PASS）
  SKIP  I1/I2 extracted ×1（IncompleteExtraction @indexer.py:219）
  验收门结论: PASS
```

`extracted` **仍然 8/8 拒绝出数** —— 这是**预期且必须的**：`IncompleteExtraction`
一行没放宽（`to_resolved.py:1432-1443` 逐字不变），绝不拿一个下界冒充峰值。
诊断书 §5 的 P3 纪律条款逐条满足。

### 6.5 字节中性

```
$ git worktree add … f0d9773 && python scratchpad/dump_numbers_ab.py > before.txt   # 4265 行
$ python scratchpad/dump_numbers_ab.py > after.txt                                  # 4265 行
$ diff before.txt after.txt ; echo $?
0
```

**`bucket` / `hand_spec` 的每一个数字逐字节不变，diff 为空。**

### 6.6 测试

`python -m pytest tests -q` → **1892 passed**（基线 1864 + 本轮 28 条新钉）。

改了**期望值**的既有测试共 2 个文件、3 处，全部按
`docs/opdag_walker_core_2026-07-25.md` §6.6 的先例办（不变量逐字保留、只换例子、逐条记原因）：

| 测试 | 改了什么 | 原因 |
|---|---|---|
| `test_opdag_components.py::test_all_four_dsv4_cells_still_clean_after_axis_capture` | unfused `CompressedSparseAttention` 207→**209** 节点 / 240→**242** 边；`DSv4HybridSelfAttention` 240→**242** / 281→**283** | `csa.py:445/460` 的 `.broadcast_to(...)` 现在建**真节点**（它改变元素数，别名掉 = 静默丢批维）。**恰好 +2**，与源里那两处一一对应 |
| `test_to_resolved_adapter.py::LEDGER_BLOCKERS` | 加入本轮新暴露的 5 处，旧 4 处保留在册备查 | 台账的定义就是"当前级联根逐条在册"；`got <= LEDGER_BLOCKERS` 这条不变量**逐字未动** |
| `test_to_resolved_adapter.py` 里 `assert "compressor.py:216" in got` | 换成 `got & {csa.py:747, router.py:393, csa.py:777, indexer.py:219}` | 该断言的**作用**是"台账不许是空的：必须点名一个仍在挡着的根"。原来点名的那个已被修好、出表；换成两支各自的当前第一根，作用逐字保留 |

---

## 7. 纪律核对（逐条）

| 铁律 | 本轮怎么守的 |
|---|---|
| 不编尺寸 / 轴 / 拟合常数 | 每条新规则都是**算子定义**（permute 的置换、reshape 的元素数守恒、chunk 的等分、arange 的长度、broadcast_to 的目标形）或**已核验的事实**（`ratio \| S`）。证不出就退 `~` 或保 `?` |
| 整除身份先核验后用 | `DivFacts` 只装核验过的；核验不过进 `Coverage.div_facts_rejected`（可见）；`assert_divisible` 不成立即 `ValueError` |
| 不许"声明即事实" | 本轮**没有**新增任何声明通路。`reshape` 单缺维那条虽然进 `declared_shapes` 台账，但它不是声明 —— 它是元素数守恒**推出来的**，记台账只为可见 |
| 只用既有 `OpType` | 一个新 op 类型都没加；改动全在既有 `View` 子类型与 `Constant` 分支里 |
| 不从 `__init__` 缺省值推结构 | 整除事实取的是 `DimTable.S` 与**层型串导出的** `compress_ratio`（`_ratio_of(layer_type)`），不是 `csa.py:556` 那个缺省 0 |
| 每条结构性断言带 `file:line` | 见上文各节；新原因码 `concat_shape_mismatch` 的消息里带的是**算子前置条件**，不是"看着不对" |
| 探针必须进库 | `scratchpad/axis_probe_{baseline,roots,chain,saves,reconcile}.py` 五个全部随代码提交 |
| 真机数字只来自 `REAL` | 本轮**没有**引用任何新的真机数字；`REAL_SHA256` 未动、门里 PASS |

---

## 8. 还挡着 `extracted` 出数的东西（诚实排序）

按"离一个可辩护的峰值还差多远"排，**不是**按修起来多容易。

| # | 阻塞项 | 性质 | 它挡住什么 | 我的判断 |
|---|---|---|---|---|
| **1** | **`workspace_bytes` / `bwd_scratch_bytes` 按契约恒 0** | **结构性天花板**，不是 bug（`to_resolved_adapter_2026-07-25.md` §5.1：kernel 实现细节，源码读不出） | 即使图 100% 解出，`extracted` 的**峰值**也系统性偏低，不能与真机比 | **这是第一顺位**。在它被关掉之前，"让 `extracted` 出一个可辩护的峰值"这件事**定义上做不到**。只能另立标定项（真机 workspace 探针），或把 `extracted` 的定位永久钉死在"逐层 `activation_saves` 的独立测量"上 |
| **2** | 同名**同尺寸**的连续重绑被合成一个张量（`csa.py:497/502` 的 `scores` 链） | 已知近似（`_Folder._bykey` 的 docstring 逐字承认） | 系统性**欠读**；r0 上目测 ~128 MiB/层 | 要 SSA 版本感知的张量身份。**带双算风险**（视图别名不能算两次）→ 必须与 `detach_aliases` 一起设计 |
| **3** | `csa.py:747` `cat` 前置条件不成立：压缩器的**内联返回值**少了 `compressor.py:224` 的 `unsqueeze(pooled, -2)` | walker 的自由函数/子 Cell **返回值绑定**取早了一步 | 整个 r4/r128 的 CSA 主链（unfused 侧 ×3 层，每层 ~140 个 op） | 修起来在 walker 的 return 绑定；**收益最大的一项**（r4 是欠读最硬的层型）。今天由 `concat_shape_mismatch` 守卫**显式拒绝**，不会静默算错 |
| **4** | `csa.py:506` `attn_sink` 是 `Parameter`，作为**自由函数实参**内联后形参没拿到声明形状 | 同上，walker 的**实参绑定** | r0 层最后 6 个 op（1–2 MiB） | 覆盖度上好看，字节上无关紧要 |
| **5** | `indexer.py:219` `Tensor([int(key_length) % ratio])`；`router.py:393` `topk` 的 `k` 未记 | 两条**walker attrs 缺失**，各一行 | fused 支 ×4 / ×3 | 便宜，但都在小张量上 |
| **6** | `csa.py:777` `arange(n_compressed)` | 级联自 #3 | — | 修了 #3 自解 |

**一句话**：图的覆盖度已经不是主要矛盾了（fused 跳过 502→256、unfused 1010→809，
且 r0 层的逐张量清单**已被独立核对为正确**）。真正挡在"可辩护的峰值"前面的是 **#1**——
它是**契约层面的缺口**，不是抽取器的能力缺口，任何继续在 `shape_infer` 上使劲的方案都绕不过它。
