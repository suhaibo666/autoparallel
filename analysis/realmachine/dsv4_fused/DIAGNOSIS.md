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
| **96** | **4** | **384** | ❌ 融合 DSA kernel 内部 fp32 `[2048,12288]`（12288/tok=64×192，**非 config 维**），前向分配、held 过反向峰值 |
| **441.9** | **1** | **441.9** | ❌ = ½·vocab·H fp32 `[129280,1792]/2`，lm_head 反向 `View` 瞬态（~22ms，**在共享 head.py**） |
| <40 小张量尾 | 291 | 253.1 | 异构 allocator/kernel 尾 |

## 结论（source-faithful，不 fudge）
1. **修复结构正确、已真机验证**：所有大结构 buffer **逐类精确吻合**——fp32 vocab×3、bf16 logits、
   FSDP 满梯度、dsv4 的 256 类(q_hnorm+cg)×4、128 类(q+core_out)×4。修复把该建的都建对了。
2. **子代理"额外 CE fp32 buffer"假设证伪**：真机 loss 区恰 3 个 fp32 vocab（评估器已建 3 个）。
   佐证：DSv3 用**同一** loss、seq **翻倍**(4096)，残差仅 ~64 MiB——若 CE 欠建一个 buffer，
   DSv3 会欠建**翻倍**而非近零。残差是 **dsv4 专属**，非共享 CE。
3. **残差 1079 = 融合 DSA kernel 内部 fp32 保留 + 共享 head 反向瞬态 + allocator 尾**（2026-07-06 二次取证更正）：
   - **384 = 融合 DSA kernel 内部 fp32 `[2048,12288]` ×4**（前向分配、held 过峰值 = 前向 save）。
     子代理 monkeypatch `ops.rms_norm`（全模型 RMSNorm 唯一入口）实测**无 96 MiB 输出**——它不是
     Python 层 rms_norm 激活，而是 `npu_sparse_attn_shared_kv` AscendC kernel 内部保留量，
     `save_for_backward` 列表里都是 bf16/小量。12288/tok=64×192，**非任何 config 维**（v_head_dim=512、
     qk_nope=448、qk_rope=64 均不匹配）→ **无符号 shape、只能测**。
   - **441.9 = ½·vocab·H fp32** lm_head 反向 `View` 瞬态，在**共享 `head.py`**（DSv3 同 head，但 DSv3
     无等价瞬态 → 归属不确定；建在 head.py 会动 DSv3 峰值）。
   - **253 = 异构 allocator/kernel 尾**（291 个小张量）。
   **更正**：原稿 #3 把 96 归为「½·wq_up + AdamW elementwise」是**错的**——½·wq_up 与 `[2048,12288]`
   numel 巧合同为 25.16M elem，误导了参数形状猜测。`Muls/Div/Square` 的 96MiB 行是**另外的 AdamW 瞬态、不在峰值 live**。
   `qk_layernorm`(q_layernorm 6 + kv_layernorm 2 = 8 MiB bf16)**已建模**，非欠计。
4. **仍 UNDER 7%（OOM-不安全向）**，且残差**非**干净的漏建 saved 激活——是**融合 kernel 内部量 + 共享 head 瞬态**，
   **无 config 公式可算**。→ 达 OOM-safe 只能：**(A)** 加 DSv4-scoped 的**实测** working-set 项（~384 融合
   kernel fp32，标注 kernel-internal，与已接受的 DSv3 平台常数同法）；**(B)** A + 折入 441.9/尾 → ~1.0；
   **(C)** 显式 OOM margin 旋钮；**(D)** 诚实记录、维持 0.930。**均需用户定夺**（"消除经验常数"原则 vs OOM 安全冲突）。

## 产物
`operator_memory.csv`（12012 行）、`memory_record.csv`（48003 行）、`memory_summary_rank0.txt`。
