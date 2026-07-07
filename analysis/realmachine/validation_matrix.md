# 仿真器 vs 真机 —— 重算 × 并行 组合验证矩阵（规划 + 结果）

> **目标**:系统验证「选择重算 × 各种并行切分」组合下,仿真器预测 vs 真机实测的内存建模关系。
> **模型**:DSv3 8L MLA+MoE、seq4096、bf16 compute/fp32 params、3 步。**baseline** `feat/unified-llm-modelspec`。
> **方法**:每格 = 真机 `max_memory_allocated`（`run_axis.sh` MEMPROBE）vs 同配置估计器预测 → ratio。

## 本 build 栈限制（真机实测,矩阵已避开死格）

1. **SP+MoE** 不支持 → 所有 MoE 配置 `SP=0`。
2. **TP+MoE**（Detach layout bug）→ 不测 tp。
3. **PP+重算** 互斥 → pp 只与 `none` 交叉（`NORECOMP=1`）。
4. **pp>2** 崩（pynative AdamW 对 decoder-only 中间 stage param/state 配对,MLA/GQA 同崩）→ 只测 pp≤2。

## 重算轴 R
`none`(NORECOMP=1) · `full`(默认) · `sel_attn`(SELECT=self_attention:0-7) ·
`sel_ffn`(SELECT=feed_forward:0-7) · `sel_flash`(SELECT=flash:0-7,细粒度单 op,验证 D-3 per-op)

## 并行轴 P（仅可跑格）
- **P0** dp=2 pp=1（fsdp baseline）
- **P1** dp=1 cp=2 colossal pp=1
- **P2** dp=1 cp=2 ulysses pp=1
- **P3** dp=2 ep=2 pp=1
- **P4** dp=1 pp=2（仅 none）

## 矩阵（✓=已验;NEW=待跑;✗=栈限制）

| P \ R | none | full | sel_attn | sel_ffn | sel_flash |
|---|---|---|---|---|---|
| **P0** dp=2 | NEW | 13953✓ | 18828✓ | 19967✓ | NEW |
| **P1** cp2 colossal | NEW | 12433✓ | NEW | — | — |
| **P2** cp2 ulysses | NEW | 12441✓ | NEW | — | — |
| **P3** dp2 ep2 | NEW | NEW | NEW | — | — |
| **P4** pp2 | 45655✓(s1) | ✗ | ✗ | ✗ | ✗ |

## 科学问题（矩阵要回答的）

1. **选择重算误差可分离性**:sel_attn 的 ~18% 欠预测,跨 dp/cp/ep **是否恒定**?（若恒定→误差与并行正交,可独立建模;若随并行变→有交互洞）
2. **cp 是否切「保留」激活**:full 重算下 cp 对峰值≈0 收益（loss 主导）。**none/select 下** body 激活可见,cp 应减之。colossal（KV all-gather 全 S）vs ulysses（全 ÷cp）应有别 → 验证 D-1 body÷cp。
3. **ep × 重算**:专家权重 ÷ep,专家激活 ÷ep;重算下 MoE 保留激活的欠预测（选择重算根因）是否随 ep 缩放。
4. **细粒度单 op（sel_flash）**:D-3 per-op 自估在单 op 下有乐观偏差,真机验证方向。
5. **optstep FSDP 修复**（`aa30d1f`）:dp=2 各格的 optstep 已 ÷fsdp,验证不再过预测。

## 执行分组（子代理,顺序跑避免抢卡）

- **T1（cp × 重算）**:P1col×{none,sel_attn}、P2uly×{none,sel_attn}、P0×none（cp 对照基准）。→ 回答 Q2/Q1。
- **T2（ep × 重算 + 细粒度）**:P3ep2×{full,none,sel_attn}、P0×sel_flash。→ 回答 Q3/Q4。

## 结果（子代理回填）

<!-- T1/T2 填真机/估计器/ratio 表 + 结论 -->

### T1 (cp × 重算)

真机：DSv3 8L MLA+MoE、seq4096、3 步、SP off、GBS=2、pp=1、2 卡。dp=1→B=2/卡；dp=2→B=1/卡。全部 5/5 `EXIT 0`，双 rank 对称。

| id | 配置 | 重算 | 真机 MiB | 估计器 MiB | ratio(est/real) | 跑通 |
|----|------|------|----------|-----------|-----------------|------|
| c1 | dp1 cp2 colossal | none | 20119.4 | 43590.0 | 2.167 | ✓ |
| c2 | dp1 cp2 colossal | sel_attn | 18788.3 | 22445.5 | 1.195 | ✓ |
| c3 | dp1 cp2 ulysses | none | 19935.4 | 43366.0 | 2.175 | ✓ |
| c4 | dp1 cp2 ulysses | sel_attn | 18796.3 | 22445.5 | 1.194 | ✓ |
| c5 | dp2 cp1（无 cp 参照） | none | 19967.3 | 26182.0 | 1.311 | ✓ |

**Q2 — cp 是否切「保留」激活?**
- **真机：切。** c1/c3(cp2,B=2) ≈ c5(无 cp,B=1) ≈ 20 GB（20119 / 19935 / 19967，近乎相等）。把 per-card batch 从 B=1 翻到 B=2、同时加 cp=2，峰值几乎不变 → cp=2 把翻倍的 body 保留激活又 ÷2 切回去了。这是「cp 切 body kept 激活」的直接证据。
- **估计器：没(完整)切。** 干净判据 = c1/c3 的 est/real（≈2.17）是否 ≈ c5 的 est/real（1.31）→ **严重偏离**。本地 cp-sweep 诊断证实：估计器 cp1→cp2（B=2）只从 50924 降到 43590（−14%），cp2→cp4 再降到 39923，**远不到真机的 ~−50%**。即估计器只把「序列相关」那部分激活按 cp 切，对占大头的「batch 相关」body kept 激活几乎没切 → cp=2 none 下过预测 2.17x。**这是 cp 建模的洞：kept body(FFN/MoE)激活的 ÷cp 缺失/不足。**（注：c5 无 cp 的 none 本身也过预测 1.31x，是与 cp 无关的 none-重算基线偏差；cp 的洞在其上再叠 2.17/1.31≈1.65x。）

**Q3 — colossal vs ulysses?**
- 真机 none：c1(col) 20119.4 vs c3(uly) 19935.4，**colossal 高 184 MiB(~0.9%)**。方向对（colossal all-gather KV 到全 S → 略高），但幅度极小——MLA 的 KV 是低秩压缩(kv_a latent)，KV cache 相对总量很小。
- 真机 sel_attn：c2(col) 18788.3 ≈ c4(uly) 18796.3（Δ 8 MiB）——attention/KV 被重算掉，col/uly 差异消失。
- 估计器同向：c1 43590 vs c3 43366，col 高 224 MiB。小 KV delta 的方向建模正确。

**Q4 — sel_attn × cp:0.82 欠预测是否保留?**
- dp=2 参照(P0 sel_attn)：est 15361 / real 18828 = **0.816**（复现 ~0.82 self_attention 欠预测）。
- cp=2 下(c2/c4)：est 22445 / real ~18790 = **1.194** —— **翻成了过预测,没有保留**。原因：sel_attn 把 attention 重算掉后，峰值由 kept 的 FFN/MoE body 激活主导，而估计器对这部分 ÷cp 缺失（Q2 同一个洞）把估计抬高 ~1.19/0.82≈1.45x，盖过并反转了原 attention 的欠预测。

**Q1 — 误差可分离性：否。** sel_attn 误差**不与 cp 正交**——存在 cp 交互洞：dp 下 0.816(欠) → cp 下 1.19(过)。根因是 kept body 激活的 cp 切分在估计器里缺失，与重算轴叠加时误差方向被它主导。修法：让 body(FFN/MoE) kept 激活按 cp ÷cp（如真机），而非仅切序列相关项。

**崩溃**：无。cleanup 已执行（log_c1–c5 / /tmp/run_c*·prep_c* 已删、无 run_ds3_mem 残留进程）。

#### ⚠️ T1 控制器复核更正（profiler 定位，真机数据有效但**归因被推翻**）

子代理的 Q2 归因「估计器没切 kept body 激活」**错**。控制器查 c1 估计器 breakdown +
cp2-none profiler（`analysis/realmachine/cp2_none/DIAGNOSIS.md`）证：过预测大头是 **fat CE
bwd_scratch 28280 MiB**（k_ce=8, full-S, B=2）,**不是 body**（act_live 8672 已 ÷cp、合理）。真机
定位**两个真 bug**：

- **Bug A**：loss 区在 cp 下**是 ÷cp（序列并行）**,估计器建成 full-S → 过预测 cp×。**D-1「loss full-S」
  结论错**（把 cp2 profiler 的 2020 buffer 误读成 B=1 full-S,实为 **B=2 S/cp**;cp=2 full 之所以 0.996 是
  B 与 cp 数值抵消**蒙对**）。
- **Bug B**：`k_ce=8` 不泛化——cp2-none 真机仅 **3 份**满 vocab 共存,估计器按 **8 份**。k_ce=8 从
  pp=2 stage1 标定,那里的 45GB 大头**可能是 kept-MoE body 而非 fat CE**（误归因）。

两 bug 均 **OOM-安全（过预测）**,但让 cp/none 配置严重不准（2.17×）。修 Bug A 牵动 B 口径 + cp 锚点;
Bug B 本质是 kept-MoE 无重算工作集应按机理建（同 D-5/选择重算根因）,非 inflate CE → **建模方向决策,待用户定夺**。
