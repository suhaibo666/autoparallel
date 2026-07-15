# 对《`closure_audit_response_2026-07-15.md` 可信度复核报告》的回应与二轮闭环（2026-07-15）

> 被审对象：`analysis/closure_audit_verification_2026-07-15.md`（第三轮，复核我上轮「20 项闭环」的可信度）。
> 判据沿用审计口径：原问题的**功能语义、输入域、对外结果**均落实且有有效回归才算闭环；只覆盖一个反例 / 只 fail-loud / 只改文档 / 只在核心对象新增但未传到调用方 = 部分闭环。
> 方法：5 个并发 subagent 按**互斥文件域**对抗修复（V1 adapter / V2 build_llm / V3 report+specs / V4 ffn+attention / V5 serve_explorer+matrix），TDD 先红后绿；主控用**审计自己的探针反例**作验收 oracle。

## 1. 先认账

审计**没有假阳性**，我上轮的「20 项闭环」确实过度——把「安全封口 / 只覆盖一个反例 / 核心对象有属性但没传到调用方」写成了闭环。审计点名的 10 项下调（P0-02、P1-01/02/06/10/11/16、P2-01/07/08）逐条属实。本轮按审计的 §7 关闭门槛把这 10 项**推进到真闭环或诚实标注剩余边界**。

**验收 oracle（审计探针的 15 个反例，本轮实测全部 fail-loud）**：
```
[REJECTED] dropout=-0.1        [REJECTED] optimizer key typo    [REJECTED] optimizer segment typo
[REJECTED] overlap_p2p=True    [REJECTED] full {999}            [REJECTED] select empty {}
[REJECTED] select {999:...}    [REJECTED] o_groups=-1           [REJECTED] hidden_size=-1792
[REJECTED] seq_length=0        [REJECTED] vocab_size=0          [REJECTED] num_layers=-1
[REJECTED] interleave=0        [REJECTED] prefetch_depth=-1     [REJECTED] num_microbatches=0
=== all 15 counterexamples rejected: True ===
```
外加：shared_gate 输出**全部有消费者**（不再孤立叶）；逐字节 roster 现能检测 router_w 4B→2B（numel roster 不变）。**596 测试绿**（含 7 个新 closure-v* 测试文件 + param 守恒逐字节/gate 补测）；12 锚点平均 **2.0% 不动**。

## 2. 10 项下调项的本轮处置

| 项 | 审计原判 | 本轮动作（file） | 验收 |
|---|---|---|---|
| **P0-02** adapter 静默丢语义 | 部分（bool True 泄漏 / optimizer+顶层 typo / 负 dropout） | ①`_PAR_UNSUPPORTED_TRUTHY` 拆布尔开关键 vs 数值键 `dense_fsdp_shard_size`——布尔键 `True`(及 `1`)都拒、数值键仅 >1 拒（from_mindformers）；②optimizer 段 schema + 顶层段 schema（catches `tyep`/`optimzier`）；③dropout 改 `!=0`（负值也拒） | 4 探针全 REJECTED；`test_closure_v1_adapter`(19) |
| **P1-01** shared gate 孤立叶 + 非逐字节 | 部分 | shared_gate 后加 `shared_gate_mul`（`sigmoid(gate)·sh_o`，saves 捕获反向）→ merge 消费 gated 输出（ffn.py）；gate-off 逐字节不变 | 5 层 gate-on 消费者非空；`test_closure_v4_shared_gate`(9) + `test_shared_gate_param_and_dataflow_on_off` |
| **P1-02** full/select 越界+空转 | 部分 | `_validate_recompute_against_graph` 拒 full 空集/越界层号、select 全空/越界层号/零命中（report.py，层号范围=全图 layer id 并集） | 3 探针全 REJECTED；`test_closure_v3_recompute_graph`(17) |
| **P1-10** o_groups 只挡 0 | 部分 | dsv4_hybrid 要求 `o_groups>0`（负数产负 numel 也拒），整除检查前置正数保证（build_llm） | `o_groups=-1` REJECTED；`test_closure_v2_structure` |
| **P1-11** 无核心维度 validator | 部分 | 统一正值校验 layers/H/heads/vocab/seq/batch ≥1、ffn/moe_ffn/experts ≥1、MLA 家族维（仅 mla/dsv4/dsa）>0（build_llm，置最前防除零） | hidden=-1792/seq=0/vocab=0/layers=-1 全 REJECTED；预设全 build |
| **P1-16** 只三条 guard | 部分 | ParallelConfig.__post_init__ 补 interleave/microbatch/num_microbatches≥1、prefetch_depth≥0、并行度≥1（specs.py） | 3 探针全 REJECTED；`test_closure_v3_parallel_scalars`(8) |
| **P2-01** 对外单口径 | 部分 | Web stage JSON + 顶层补 allocated_oom/reserved_oom/reserved_margin；analyze_matrix 补两列；前端 tab 标 `⚠reserved`（serve_explorer/analyze_matrix） | `test_closure_v5_dual_oom`(5)+`test_closure_v5_matrix_oom`(2) |
| **P2-07** exact 只测 numel/gate-off | 部分 | 补**逐字节**（numel×dtype_bytes）对账 + gate on/off 覆盖（param_conservation） | 逐字节 roster 检测 dtype 腐蚀；gate on/off 各测 |
| **P2-08** 源码注释反证实现 | 部分 | attention.py `FLASH_LSE_WS` 注释订正（workspace 已 TensorRef 按 TP 切）；from_mindformers qk_layernorm 注释订正（在忽略集、不映射） | 注释与实现一致 |

## 3. 诚实标注：两项**推进但未到审计要求的完整口径**

不再像上轮那样把「推进」写成「整项闭环」——这两项我如实标为**部分**：

- **P1-06 CE provenance**：直接耦合已拆（切 DSA 融合不再切 CE）、显式键可覆盖、架构默认现发 `warnings.warn` 可追溯。**但**无显式键时仍用 `attn_type=="dsv4_hybrid"→True` 的**架构启发式**（审计 §4.4 要的「由独立可追溯 loss 实现来源决定」未做到）——我**刻意没有**从 `chunk_loss_num` 推断 CE fused（核实二者机制正交：chunk_loss_num 独立驱动 `loss_type="chunked"`，而 cross_entropy_fused 只 gate DSv4-fork 的 kept-frag margin，硬凑会注入未验证的内存效应）。故本项判**部分**：耦合+provenance 闭环，loss-实现级 provenance 仍需真机区分 fused/unfused/chunked CE（审计 §6.2 建议的下一轮 NPU 项）。
- **P1-16 完整 feasibility matrix**：审计点名的 interleave=0/prefetch=-1/num_microbatches=0 三个标量已在核心构造处拒；**但**「完整 runtime constraint matrix」（各并行轴组合的可跑性）仍只有 PP+swap、tp>1+SP、非 Adam 三条 + 这批标量。标量非法值闭环，组合矩阵仍**部分**。

## 4. 仍如实开放（本轮不碰，与审计一致）

P1-07/08/12/13/14/15、P2-02/04/05 —— 真建模工作（checkpoint islands、bwd max-live、MoE skew、CP buffer、FSDP 子模块 wrap 时间线、PP send/overlap、opdag 主链、标定泛化、全尺寸预设）。P1-17（UI 非 round-trip）、P1-19（分离 offload）、P2-03（qk 可忽略）维持部分。

## 5. 本轮后状态口径（不再过度声明）

| 状态 | 项 |
|---|---|
| **闭环（功能+输入域+对外+回归）** | P0-01/03/04/05，P1-03/04/05/09/18，P2-06 **＋本轮推进到闭环** P0-02、P1-01/02/10/11、P2-01/07/08 = **18** |
| **部分（推进但未到完整口径，如实标注）** | P1-06（provenance/loss-源）、P1-16（标量闭环/矩阵部分）、P1-17、P1-19、P2-03 = **5** |
| **开放（真建模）** | P1-07/08/12/13/14/15，P2-02/04/05 = **9** |

审计的下一轮 NPU 建议（同 DSv4 下 fused/unfused/chunked CE 的 loss provenance；gate multiply 的保存张量与峰值）本轮未采——因本轮全部是离线配置校验/图数据流/属性对外，可由代码与探针 oracle 确定，无需消耗真机（P0-01 梯度生命周期与 P0-04 TP 分片的决定性 NPU 证据上轮已采、本轮相关实现未改）。12 锚点为本地重放，非新采样。
