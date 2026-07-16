# FlashAttention 缓存激活 vs 真实 FA 算子 —— 真机验证（2026-07-16）

> 问题（用户）：flash attn 要缓存的激活是否和真实的 FA 一致？因为真机用的是 FlashAttention 算子计算。
> 结论：**一致，且经真机算子级探针直接证实**——评估器缓存的正是 FA `save_for_backward` 的
> Q / K,V / O / softmax(max,sum)，**不含 S×S 注意力矩阵**。无需改代码（P1-09 建模本就正确，现真机坐实）。

## 1. 评估器现有建模（P1-09，2026-07-14）

flash op 的 `saves`（前向驻留到本层反向 = act_live；重算层随 saves 丢弃、反向重物化）：

| 路径 | flash `saves` | 语义 |
|---|---|---|
| **GQA** | `[qkv, attn, fa_stats]` | 融合 Q/K/V 投影 + 输出 O + softmax 统计 |
| **MLA** | `[qb_out, kvb_out, attn, fa_stats]` | Q 上投影 + KV 上投影 + O + softmax 统计 |

- `fa_stats = _fa_stats()` = **`[2, B, n_heads, S, 8]` fp32**（softmax_max + softmax_sum 两张），head 维 ÷tp、S 维 ÷cp。
- flash 前向/重算瞬态由 `workspace_ref = _fa_workspace()`（同形，不入 act_live）承载。
- **关键**：不建 S×S 注意力矩阵——FA 反向靠 Q/K/V/O + softmax 统计**重算** attention，不物化 S×S。

## 2. 真机算子探针（`FlashAttentionScore`，116/shb.ms.2.9，MindSpore 2.10 / CANN 9.0，B=1 N=8 S=4096 D=128）

直接调用 Ascend `FlashAttentionScore` 前向，打印输出张量（脚本 `.claude/skills/.../fa_probe.py`）：

| 前向输出 | 真机 shape | dtype | 对评估器 |
|---|---|---|---|
| `softmax_max` | **(1, 8, 4096, 8)** = `[B, N, S, 8]` | **Float32** | ✅ = `_fa_stats` 上半 |
| `softmax_sum` | **(1, 8, 4096, 8)** = `[B, N, S, 8]` | **Float32** | ✅ = `_fa_stats` 下半 |
| `softmax_out` | **(1,)** —— 空占位 | BFloat16 | ✅ **不物化 S×S 矩阵** |
| `attention_out` | (1, 8, 4096, 128) = `[B, N, S, D]` | BFloat16 | ✅ = `attn`（O）|

**逐条坐实**：① softmax 统计末维**确为 8**、fp32、两张（max+sum）——`_fa_stats` 的 `[2,B,N,S,8]fp32` **精确命中**；
② `softmax_out` 为空 `(1,)`——**真 FA 不存 S×S**，与「靠 Q/K/V/O+stats 重算」建模一致；
③ `attention_out=[B,N,S,D]bf16` = `attn` 存量。`FlashAttentionScoreGrad` 的输入 = {q,k,v,attention_out,softmax_max,softmax_sum,dy}
→ 保存集 = **Q/K/V + O + (max,sum)**，正是评估器 flash `saves`（GQA `qkv`+`attn`+`fa_stats` / MLA `qb_out`+`kvb_out`+`attn`+`fa_stats`）。

## 3. 字节 & 切分（评估器口径，与算子一致）

- softmax 统计（一层，tp=1 cp=1）：`2·B·n_heads·S·8·4B`。DSv3-mini MLA = 2.0 MiB；GQA(16 头) = 4.0 MiB。
- 切分实测（本地）：`fa_stats` 随 **tp（head 维）** 与 **cp（S 维）** 双切——tp1cp1=2.0 → tp2 或 cp2=1.0 → tp2cp2=0.5 MiB。与算子 `[B,N,S,8]` 的 head/S 维物理切分一致。
- MLA flash `saves` 合计（DSv3-mini，dp=2）：`qb_out`12 + `kvb_out`20 + `attn`12 + `fa_stats`2 = **46 MiB/层**。

## 4. 聚合真机交叉校验（FA saves 驻留 act_live 的路径）

full 重算锚点**丢弃** FA saves（重算），故用**无重算**（全存激活）配置直接暴露 FA saves 在 act_live 的贡献：

| DSv3 4L seq4096 | 实测 peak_alloc | 评估器预测 | 预测/实测 |
|---|---|---|---|
| **无重算**（全存，含 FA saves 46 MiB/层驻留） | **15285.4 MiB** | 15672.1 | **1.0253** |
| full 重算（FA saves 丢弃，对照） | 12473.1（历史锚点） | 12437.9 | 0.9972 |

- 无重算 - full 重算 delta：真机 15285.4−12473.1 = 2812.3 MiB；预测 15672.1−12437.9 = 3234.2 MiB。
- **方向 OOM 安全**：无重算评估器**偏高 2.53%**——FA saves **未被低估**（低估才 OOM 危险）。
- **归因（诚实）**：FA 字节已由 §2 算子探针**逐字节坐实**；2.53% 偏高**不在 FA**，而在无重算下**全层 act_live 于 loss 峰共存**假设的整体保守（含 norm fp32 cast 长驻等），属**独立、次要、保守**项——与 FA 缓存是否忠实无关。

## 5. 结论

1. **FA 缓存激活与真实 FA 一致**——算子级探针直接证实：保存集 = Q/K/V + O + softmax(max,sum)`[B,N,S,8]fp32`，**不含 S×S**。
2. **无需改建模**：P1-09 已把 FA saves 建成 `save_for_backward` 同款，现真机坐实（含末维 8、fp32、空 softmax_out）。
3. 聚合无重算真机 2.53% 偏高为 OOM 安全、且**不源于 FA**（FA 字节精确），记为独立次要保守项。

新增回归 `tests/test_flash_attn_saves_contract.py` 钉住 FA 保存契约（Q/K/V/O + `[2,B,N,S,8]fp32` 统计、无 S×S），
`attention.py` P1-09 注释补 2026-07-16 真机算子确认。复现脚本 `real-machine-memory-sim/fa_probe.py`。
