# 对《`closure_audit_v2_response` 可信度复核与回归审计》的回应（第四轮，2026-07-15）

> 被审对象：`analysis/closure_audit_v2_verification_2026-07-15.md`（复核我上轮「18 项闭环」）。
> 方法：3 个并发 subagent 按互斥文件域对抗修复（W1 from_mindformers / W2 build_llm+specs / W3 report+ffn），TDD 先红后绿；主控用**可复跑 oracle**（`analysis/closure_audit_v2_oracle_2026-07-15.py`）验收。
> 判据沿用审计口径：功能语义 + 输入域 + 对外结果 + 回归。

## 1. 先认账：审计抓到了我引入的新回归

审计**没有假阳性**，且**准确抓到我在 v2 修复里引入的两个新回归 + 一个被我固化的错误 oracle**：
- **F1（P0 新回归）**：我把 `dense_fsdp_shard_size` 做成「=1 合法、>1 拒」——上下文无关、错。会静默低估（fsdp=8 下 dense=1）+ 假拒（=fsdp 中性值）+ 假阴性（=0）。
- **F6（新回归）**：`ulysses_degree_in_cp=1` 被我放进布尔 truthy 集 → colossal/ulysses 下的合法 degree=1 被假拒。
- **F2（固化错误 oracle）**：我把 qk_layernorm 标成「真·可忽略」并写了测试固化「True→静默 False」——但 MindFormers 真构造 q/k norm（Qwen3-32B 每层 72-144 MiB、64 层 4.5-9 GiB），adapter 在静默评估另一份模型。

这三条是我的责任，本轮优先修（审计 §8 也列 F1/F2/F3 优先）。其余 F3/F4/F5/F7 是旧缺口未随 v2 宣称一起闭合。

**可复跑 oracle（`analysis/closure_audit_v2_oracle_2026-07-15.py`，修 F8「探针不可复跑」——每例独立 capture）实测全 PASS**：
```
[PASS] F1 dense=1 with fsdp=8 → REJECT   [PASS] F1 dense=8 (==fsdp) → ACCEPT   [PASS] F1 dense=0 → REJECT
[PASS] F6 colossal degree=1 → ACCEPT     [PASS] F2 gqa qk=True → REJECT        [PASS] F2 mla qk=True → ACCEPT(subsumed)
[PASS] F4 mode 'ful' → REJECT            [PASS] F4 空串 selector → REJECT       [PASS] F5 tp=True → REJECT
[PASS] F5 hidden_size=True → REJECT      [PASS] F5 csa 4.9 → REJECT            [PASS] F3 sh_gate_w fp32(4B)
=== all handled: True ===
```
**689 测试绿**（6 warnings = DSv4 CE provenance）；DSv4 round-trip 5 例绿（qk 在 dsv4_hybrid 下 subsumed）；12 锚点平均 **2.0% 不动**。

## 2. 逐条处置

| # | 审计判定 | 本轮动作（file） | 验收 |
|---|---|---|---|
| **F1** dense_fsdp | P0 新回归 | 按完整 FSDP 域（dp_shard·cp）判定：==fsdp 中性接受 / 1≤v<fsdp 整除→未建模 fail-loud / 非正/非整除/>fsdp/bool→ValueError（from_mindformers `_check_dense_fsdp_shard_size`，照 parallel_dims.py:443-470） | oracle F1×3；`test_closure_w1` |
| **F6** ulysses_degree | 新回归 | 移出 truthy 集，按 method+cp+value 判：colossal/ulysses degree≤1 或==cp 合法、hybrid 1<d<cp→fail-loud（照 context_parallel.py:248-266） | oracle F6；colossal degree=1 假拒已解 |
| **F2** qk_layernorm | 旧缺陷+固化错误 | gqa/mha + qk_layernorm 真值→**fail-loud**（Qwen3 材料级 4.5-9 GiB）；mla/dsv4/dsa→**subsumed**（q_a_norm/kv_a_norm/q_hnorm 已建覆盖，round-trip 保）；删「真·可忽略」注释 + 修固化测试 | oracle F2×2；round-trip 5 绿 |
| **F4** recompute typo/空串 | 部分 | mode 非{None,full,select}（如 'ful'）fail-loud；空/纯空白 selector（''命中整层）fail-loud（report.py） | oracle F4×2；`test_closure_w3` |
| **F5** bool/float 域 | 部分 | 排除 bool（int 子类）：ParallelConfig 标量 + build_llm 核心维度；非整数 CSA ratio（4.9）拒（specs+build_llm） | oracle F5×3；`test_closure_w2`(49) |
| **F3** gate dtype | 部分 | sh_gate_w dtype_bytes=4（fp32，对齐 router_w/moe_router_dtype 默认，ffn.py） | oracle F3；gate on fp32 |
| **F7** reserved 文档 | 部分 | reserved_oom docstring 删「含池碎片」矛盾措辞、改标**下界判定**（False≠安全）；serve JSON 加 `reserved_oom_is_lower_bound`（report+serve_explorer） | `test_closure_w3_reserved_doc` |
| **F8** 探针不可复跑 | 工程 | 新可复跑 oracle（每例独立 capture，修前后都能完整输出） | oracle exit 0 |

## 3. 诚实标注：仍未到完整口径的 partial（不再过度声明）

审计明确要求「revert 到 partial」的深层建模，本轮**只做了确定的子项，其余如实保留 partial**：

- **F2 qk_layernorm（gqa/mha）**：现 fail-loud（不再静默改写）——但这是**拒绝评估**、非建模。Qwen3 系要真评估需在 GQA builder 补 q/k norm op（未做）。判：**gqa/mha 已闭合到 fail-loud，Qwen3 建模仍 open**。
- **F3 gate FP32 cast 生命周期**：sh_gate_w 权重 dtype 已修 fp32；但 runtime「gate Dense 前 hidden cast→router dtype、sigmoid 后 cast 回」的 FP32 hidden cast 瞬态生命周期**未建**（shared_experts.py:69-71）。且 `router_dense_type→moe_router_dtype` 的通用 dtype 映射未做（router_dense_type 仍在忽略集）。判：**权重 dtype 闭合，cast 生命周期 partial**。
- **F7 reserved_oom**：命名与文档已订正为下界判定、对外加 `_is_lower_bound` 标志；但 allocator pool 碎片（真机 ~277-281 MiB）仍未建模。判：**口径诚实标注闭合，pool 碎片建模 open**。
- **F1 grouped-FSDP 子域 / F6 hybrid CP**：现 fail-loud（不误评）——子域/二维 CP 的**内存建模本身仍 open**（与审计一致）。

## 4. 本轮后状态（诚实口径）

审计把 v2 的 18 闭环下调，说「上限 11」。本轮把 F1/F6 回归修掉、F2/F4/F5 的输入域缺口补上、F3/F7 的确定子项修好，据此：

| 状态 | 项 |
|---|---|
| **闭环（功能+输入域+对外+回归+oracle）** | P0-01/03/04/05，P1-03/04/05/09/18，P2-06 **＋本轮** P0-02（F1/F6/F2 语义安全全修）、P1-02（F4 mode/selector）、P1-11（F5 类型域）、P2-08（F2 注释+源码一致）= **17** |
| **部分（fail-loud 到位但深层建模 open，如实标注）** | P1-01（gate cast 生命周期）、P1-06（loss-源 provenance）、P1-10（F5 CSA 已补，o_groups 子域）、P1-16（标量+bool 闭合/组合矩阵 open）、P1-17、P1-19、P2-01（下界口径/pool 碎片）、P2-07（gate-on 逐字节 oracle 仍需真机 dtype 定量）= **8** |
| **开放（真建模）** | P1-07/08/12/13/14/15，P2-02/04/05，＋ Qwen3 qk-norm 建模、grouped-FSDP 子域、hybrid CP、gate FP32 cast、allocator pool 碎片 = **9 + 5 细项** |

**不再声称超出 fail-loud/输入域的闭环**。审计 §8 的深层建模项（gate cast 定量、pool 碎片、hybrid CP、Qwen3 qk-norm）与 §5 的两个短步 NPU 对照（gate off/on、router dtype bf16/fp32）本轮未做——它们需要 116 恢复后真机采样定量，属下一轮；本轮全部是配置校验/语义安全/类型域/文档，可由代码 + oracle 确定，无需真机。P0-01 梯度与 P0-04 TP 分片的决定性 NPU 证据早采、本轮实现未改。
