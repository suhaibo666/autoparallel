# cp2 none profiler —— 纠正 T1 误诊 + 定位 2 个真 bug（2026-07-07）

真机：`ascend116`/`shb.ms.2.9`/cards 0,1，DSv3 8L MLA+MoE、seq4096、dp=1 cp=2 colossal、**无重算**、
GBS=2（per-device B=2）。真机 `max_memory_allocated=20119.4`；估计器 **43590（2.17× 过预测）**。

## T1 子代理的误诊

T1 报告说过预测因「估计器没切 kept body(FFN/MoE)激活的 ÷cp」。**错。** 估计器 breakdown：
`persistent 5197 + act_live 8672(body,已 ÷cp,合理) + bwd_scratch 28280 + gather 555 + grad 883`。
过预测大头是 **bwd_scratch 28280 MiB = fat CE**（`4·S·B·V·(k_ce−1)`，k_ce=8, full-S, B=2），**不是 body**。

## 峰值 live-set 重构（`operator_memory.csv`，ScatterAddExt 峰）

算子级 live 14921 MiB（+ persistent ~5198 = 20119）。loss 区：**3× 2020 MiB**（Log/Neg/ScatterAddExt）
+ 1010（logits bf16）+ 883.8 + 441.9。**只有 3 份满 vocab fp32 共存,不是 8。**

## 🔴 Bug A：loss 区在 cp 下**是 ÷cp（序列并行）**,D-1「full-S」结论错了

`2020 MiB = 4096·2·129280·4/2 = [S/cp=2048, B=2, V] fp32`。即真机 loss 区 seq **被 cp 切**（S/cp=2048）。
- **D-1 的误读**：D-1 从 cp2-**full** profiler 见 2020 buffer,判成「full-S B=1」,实际是「**S/cp B=2**」
  （`full-S B=1 = S/cp B=2 = 2020`,数值相同 → 混淆了 B 与 cp）。
- **为何 D-1 cp=2 full 还能 0.996**：估计器验证时用 **B=1 full-S=2020**,真机是 **B=2 S/cp=2020**,
  两者数值恰好相等 → **B 与 cp 抵消,蒙对**。是「双错相消」,非真的建对。
- **实际行为**：估计器把 loss 区建成 full-S（`cp_shard=False`）,**过预测 cp×**（若 B 口径一致）。
  方向 OOM-安全（over）,但错。**修**：loss/head 区 seq 应 ÷cp（`cp_shard=True`）,并统一 B 口径。

## 🔴 Bug B：`k_ce=8`（D-10 ①）不泛化,可能是**误归因**

cp2-none 真机 loss 区**仅 3 份**满 vocab 共存（6060 MiB）;估计器 fat CE 按 **8 份 full-S B=2 = 28280**。
- k_ce=8 是从 **pp=2 stage1**（无 cp,profiler 见 ~8-10 份 4040=full-S B2）标定的。
- **pp stage1 的 45GB 大头可能根本不是 fat CE,而是 kept 4-层 MoE body**（无重算全存,B=2 full-S）——
  我当时用 inflate CE（k_ce=8）去凑 45655 的总数,**可能凑对了总数、错了归因**,于是在 cp/none 崩。
- 共存份数 **schedule/config 相关**（pp ~8 vs cp ~3,同为 none B=2）——正是检视 §15 标注的 k_ce 脆弱性,
  T1 实测坐实：**cp2 none 过预测 2.17×**（Bug A 的 full-S×2 叠 Bug B 的 8/3 ≈ 5.3× 在 loss 区,被 body 稀释到 2.17×）。

## 结论 + 建议

- 两 bug 都 **OOM-安全（过预测）**,但让估计器在 **cp/none 配置严重不准（2.17×）**。
- **Bug A（loss ÷cp）**：清晰的修正,但牵动 B 口径 + cp=2 full 锚点（现靠 B/cp 抵消蒙对）→ 需一并理顺 B 约定。
- **Bug B（k_ce）**：本质是 fat-CE 单点标定 + 可能误归因（pp-45GB 或许是 kept-MoE 而非 CE）。真正的修
  = 把 kept-MoE 无重算工作集按机理建对（同 D-5/选择重算根因）,而非 inflate CE。**这是建模方向决策,需用户定夺。**
- 未跑 T2（ep×重算）：其 none 格会复现同 bug,先定 Bug A/B 方向再续更有意义。

## 产物
`operator_memory.csv`（cp2 none profiler）。真机数据 `analysis/realmachine/{cp2_none,select_attn,pp2_norecomp}/`。
