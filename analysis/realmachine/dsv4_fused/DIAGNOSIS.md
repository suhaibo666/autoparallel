# DSv4-align fused 真机验证 — 残差归因（2026-07-06）

真机：`ascend116` / `shb.ms.2.9` / cards 2,3 / 4L seq2048 heads64 v512 FSDP-2 无重算 **fused**。
`prep_dsv4align.py FUSED=1`，`run_ds3_memprobe.py`（clean）+ `run_ds3_memtimeline.py`（profiler）。

## 峰值锚点（clean，authoritative）
| rank | peak_alloc | peak_reserved | framework_reserve |
|---|---|---|---|
| 0/1 | **15415.5 MiB** | 16092/16096 | 676.5/680.5 |

**逐字节复现上一会话 15415.5 锚点（0.0% 漂移）。** 峰值算子 = `ScatterAddExt`（loss 反向，与 DSv3 同签名）。
评估器（修复后）预测 **14336.5** → 残差 **1079.0 MiB（0.930，UNDER，OOM-不安全向）**。

## 峰值时刻 live-set 重构（operator_memory.csv，alloc≤T_peak<release）
真机峰值时算子级 live = **9747.0 MiB**；`15415.5 − 9747.0 = 5668 ≈ persistent 5662.8`
（params+opt 在 init 期分配、不在 operator trace 内）→ 对上。

| 真机 size 类 | ×数 | 小计 MiB | 评估器是否建模 |
|---|---|---|---|
| 1010 fp32 vocab | 3 | 3030 | ✅ `logsm`(saved)+`probs`+`grad_log_softmax`(bwd_scratch)——**恰 3 个，子代理"漏建 CE buffer"假设被证伪** |
| 505 bf16 vocab logits | 1 | 505 | ✅ `logits_lm` |
| 883.8 fp32 grad（满参） | 1 | 883.8 | ✅ `grad_buf`（FSDP 满梯度） |
| 256 fp32 | 8 | 2048 | ✅ `q_hnorm_fp32`+`cg_fp32` ×4 层（**本次修复**） |
| 128 bf16 | 8 | 1024 | ✅ `q`+`core_out` ×4 层 |
| **96** | **4** | **384** | ❌ = ½·`wq_up`[1536,64·512] fp32（每层 1，×4） |
| **441.9** | **1** | **441.9** | ❌ = ½·emb/head[129280,1792] fp32 |
| <40 小张量尾 | 291 | 1430.4 | 部分（`o_group_out`/`g`/kv/q_a/router…）——真机尾更重 |

## 结论（source-faithful，不 fudge）
1. **修复结构正确、已真机验证**：所有大结构 buffer **逐类精确吻合**——fp32 vocab×3、bf16 logits、
   FSDP 满梯度、dsv4 的 256 类(q_hnorm+cg)×4、128 类(q+core_out)×4。修复把该建的都建对了。
2. **子代理"额外 CE fp32 buffer"假设证伪**：真机 loss 区恰 3 个 fp32 vocab（评估器已建 3 个）。
   佐证：DSv3 用**同一** loss、seq **翻倍**(4096)，残差仅 ~64 MiB——若 CE 欠建一个 buffer，
   DSv3 会欠建**翻倍**而非近零。残差是 **dsv4 专属**，非共享 CE。
3. **残差 1079 = 框架/优化器级参数形状 fp32 瞬态 + 反向逐 op scratch**：
   未建的 96×4 与 441.9 是**½ 参数形状**（wq_up、emb/head 的 fp32 分片），算子名是
   `Muls/Div/Square/Sqrt/Addcmul/RmsNorm`——**RMSNorm 反向 + AdamW** 的 elementwise 原语。
   即 §8.5② 的**反向/优化器工作集尾**在 loss-反向峰值处与激活共存；评估器 `act_live=Σsaves`
   只计"已 save 的激活"，不计这些 in-flight 瞬态。**非**漏建 saved 激活。
4. **仍 UNDER 7%（OOM-不安全向）**。达真·OOM-safe 需显式建"参数形状反向/优化器瞬态工作集"
   （§8.5② 扩到峰值事件），或设显式 OOM 安全 margin（非结构 fudge 常数）。属需用户定夺的建模/安全姿态决策。

## 产物
`operator_memory.csv`（12012 行）、`memory_record.csv`（48003 行）、`memory_summary_rank0.txt`。
