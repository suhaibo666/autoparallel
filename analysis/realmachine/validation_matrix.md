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
