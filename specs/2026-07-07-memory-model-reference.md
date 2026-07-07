# 显存仿真模型 —— 计算方式 + 符号化公式参考

> **baseline**: `feat/unified-llm-modelspec` @ `07a471f`（2026-07-07）。
> **用途**:一页说清评估器**怎么算每卡峰值显存**——总体思想、十个内存桶、每一项的**符号化字节公式**
> （用 batch/hidden/seq/vocab 等标准符号），以及**整体求和式**。所有标定常数带出处、真机锚点在 §14。
> 详细整改历程见 `2026-07-06-audit-remediation.md`（D-1..D-10）、`2026-07-01-...design.md`。

---

## 0. 总体思想:事件驱动的逐 stage 峰值

预测目标 = 每卡 **`max_memory_allocated`**(OOM 主判据)。方法:对**每个 PP stage** 独立跑一遍
「前向逐层 → fwd_end → 反向逐层 → 优化器 step」的**事件时间线**,每个事件算一次当下总占用,
峰值 = 逐事件取 max:

```
P_stage = max_event ( Σ_bucket B_i(event) + framework_reserve )
tightest = argmax_s P_s ;  OOM = (max_s P_s > HBM)
```

`framework_reserve` 生产默认 **0**(经验兜底常数已消除,机理项都散进下面的桶/公式)。
落地:`mem_timeline.MemTimeline.simulate`(事件循环 + `rec()` 逐事件取 max);`report.Evaluator` 门面。

## 1. 十个内存桶(每个事件的瞬时状态,`Buckets`)

| 桶 | 含义 | 何时活 |
|---|---|---|
| `persistent` | 参数 + 优化器状态(**剔梯度**) | 全程常驻 |
| `act_live` | 存活的 saved 激活 | 前向存、反向逐层释 |
| `gather_buf` | FSDP all-gather 权重 + 预取双缓冲 | 逐层前/反向 |
| `grad_buf` | 一层的满梯度(reduce-scatter 前) | 该层反向 |
| `recomp_scratch` | full 重算层重跑 forward 的峰值 | 该层反向 |
| `bwd_scratch` | 反向瞬态物化(loss probs/grad 等) | 该 op 反向 |
| `bwd_working_set` | 无重算层反向工作集(激活梯度) | 该层反向 |
| `swap_buf` | 激活 swap H2D 预取 | 被卸载层反向 |
| `workspace` | 算子 workspace(flash/MoE staging) | 该 op 前向 |
| `optstep` | ② 优化器-step 瞬态 | 反向后 step |

## 2. 符号定义

| 符号 | 含义 | 符号 | 含义 |
|---|---|---|---|
| `B` | 每卡 micro-batch | `E,k,C` | 专家数 / topk / capacity_factor |
| `S` | 序列长 | `F,Fm` | dense / MoE FFN 中间维 |
| `H` | hidden | `rq,rkv` | MLA q/kv lora rank |
| `V` | vocab | `dno,dro,dv` | MLA nope/rope/v head 维 |
| `L` | 本 stage decoder 层数 | `nh,nkv,dh` | 头数 / KV 组 / head_dim |
| `bc,bg` | compute(bf16=2)/grad(fp32=4) 字节 | `so` | opt 状态字节/参(bf16 参 14 / fp32 参 12) |

**并行分母(切分因子)**:`dp`(dp_shard)、`cp`、`tp`、`ep`、`pp`;`sp = tp`(开 SP 时,否则 1);
`fsdp = dp·cp`;`efsdp = (dp·cp·tp)/ep`。

**激活序列分母** `σS`（2026-07-07 Bug A 修正：loss/head 区**也 ÷cp**）:
```
σS = sp·cp   (SP 标注的激活:body + loss/head 的 h_final)
   = cp      (非 SP 激活:attention 内部、logits/logsm/probs 等 loss/head 区)
   = 1       (colossal CP 的 KV 维 all-gather)
```
> loss/head 区**随 cp ÷cp**（真机 cp2-none profiler:loss buffer=[S/cp,B,V]）。此前 D-1 误判 full-S
> （把 [B=2,S/cp] 误读成 [B=1,full-S]，数值都 2020）→ cp 下过预测 loss 区,已修（见 §7、§15）。

**标定常数**:`k_ce=8`、`k_opt=6`、`blk=512`、`hccl=200MB`(出处见 §12)。

---

## 3. persistent —— 逐权重参数量 + 公式

**逐权重参数量(numel)**:

| 权重 | numel |
|---|---|
| embedding | `N_emb = V·H` |
| lm_head(untie) | `N_hd = V·H`（tie 时 0） |
| GQA attn / 层 | `N_a^gqa = H·(nh+2·nkv)·dh + nh·dh·H` |
| MLA attn / 层 | `N_a^mla = H·rq + rq·nh·(dno+dro) + H·(rkv+dro) + rkv·nh·(dno+dv) + nh·dv·H` |
| dense FFN / 层 | `N_f = 3·F·H`（gated SwiGLU）/ `2·F·H`（ungated） |
| MoE 专家 / 层 | `N_e = 3·E·Fm·H` |
| MoE 共享 + router / 层 | `N_s = 3·nsh·Fm·H` ，`N_r = H·E` |

**公式**（专家权重走 efsdp、其余走 fsdp;剔梯度——grad 是反向瞬态在 `grad_buf`）:

```
persistent = so · ( N_dense / fsdp  +  N_expert / efsdp )
  N_dense  = N_emb + N_hd + Σ_层(N_a + N_f或N_s + N_r)   （本 stage 的非专家参数）
  N_expert = Σ_MoE层 N_e                                 （本 stage 的专家参数）
```

## 4. act_live —— 逐激活字节(无重算,单 microbatch)

**通式**(每个 saved 张量,符号 shape `(S,B,d)`):
```
byte(tensor) = ⌈ (S/σS) · B · (d/τ) · bc ⌉_blk       τ = tp（该维 tp 切）else 1
```

**逐层足迹**(GQA + dense 为例,略 blk 取整):
```
a_层 ≈ (S·B·bc / σS) · [ 2H            ← ln1 + ln2（H 不 tp 切、seq 按 σS 切）
                        + ((nh+2nkv)dh + nh·dh)/tp     ← qkv + attn_out
                        + 3F/tp ]                       ← g(2F) + act(F)
      + 64·B·nh·S/σS                                    ← flash softmax LSE（见 §6）
```
- **MoE 层**:把 `3F/tp` 换成 dispatched/专家中间量 ∝ `S·B·k·C·H`(专家维 ÷ep,∝dispatched token)。
- **mHC**:`a_层 ×n`(残差流打包 `[S,B,n·H]`)。

**求和**:
```
act_live = Σ_(本stage各层) a_层  +  a_loss          （全重算时各层只留 checkpoint_input）
```

## 5. 反向三桶

设 `fml`(forward_max_live)= 层内 mini-时间线上同时存活激活峰 ≈ `max_op( Σ 活张量 )`。

```
recomp_scratch  = max(0, fml − ckpt_in)          （full 重算层:重跑 forward）
bwd_working_set = max(0, fml − bwd_scratch)       （无重算层:激活梯度 dL/dact 与激活同形共存）
grad_buf        = (N_a + N_f)/tp · bg             （当前层满梯度,reduce-scatter 前）
```
- **细粒度选择性重算(D-3)**:选中/非选中逐 op 分;选中段 `recomp = fml(选中) − 已 pin 的进入边界`
  （只减**确已常驻**的输入,修单 op 低估);非选中段走 `bwd_working_set`。退化:全选==full、全不选==none。

## 6. 框架 / kernel 项(符号)

| 项 | 公式(bytes) | 落点 |
|---|---|---|
| FSDP gather | `gather_buf = N_层/tp · bc · (1 + depth)` | 当前层满参 + 预取双缓冲(depth 默认 1) |
| flash workspace | `64·B·nh·S / σS` | =2 张 LSE `[B,nh,S,8]`×4B;∝S·nh |
| MoE staging | `2·S·B·k·C·H / σS` | 置换发送/散射(bf16) |
| mHC 反向 | `4·S·B·n·H / σS` | sinkhorn grad |
| 分配器碎片 | 逐张量 `⌈·⌉_512` | MindSpore `DynamicMemPoolBestFit` |

## 7. loss 区(① unfused CE)

```
a_loss = S·B·V·bc            ← logits(bf16, saved)
       + 4·S·B·V             ← log_softmax(fp32, saved)

bwd_scratch_loss = 4·(S/σS)·B·V·(k_ce − 1)   若【无重算 且 unfused CE】
                 = 8·(S/σS)·B·V              否则（fused / 重算,精简 = probs+grad 2 份）
```
- 门控 `cross_entropy_fused`(DSv4 融合=True 恒精简);loss 区随 cp ÷σS（Bug A）。
- **k_ce 与制度相关**(真机 profiler 实测共存份数):**流水线末 stage(pp>1)→k_ce=8**（pp2-stage1 见
  8~10 份满 vocab fp32）;**单 stage(pp=1)→k_ce=4**（cp2-none/select 仅 3~4 份共存）。CE 链共存数
  受 pipeline 调度影响,故按制度分（非单点常数）。
- **loss/head 区随 cp ÷cp**（Bug A 修正,真机 cp2-none profiler:loss buffer=[S/cp,B,V]）。

## 8. optimizer-step 事件(②)

反向全结束后 AdamW 逐权重更新,峰在**最大单权重**的 fp32 瞬态(grad/Square(g²)/sqrt(v̂)/m̂/update):
```
P_optstep = persistent + k_opt · max_w( numel_w / fsdp ) · 4        (k_opt=6)
```
最大权重通常是 `N_emb/fsdp` 或 `N_hd/fsdp = V·H/fsdp`。与激活桶**互斥**(step 时激活已释);
通常被更大的反向峰盖住,**无 loss 的 stage(如 pp emb 首 stage)才露头**。

## 9. 切分因子(代入上式的分母)

```
参数     ÷ fsdp（专家 ÷efsdp）
激活 seq ÷ σS       ；  tp 维 ÷ tp  ；  专家维 ÷ ep
pp: 不切张量,切「层」→ stage 子集
```
**cp 按算法(D-1)**:`ulysses/ring/hybrid` body 全 ÷cp;`colossal` body ÷cp 但 **KV all-gather 到全 S(σS=1)**;
**loss/head 区随 cp ÷cp**（Bug A 修正,2026-07-07——非「全 S」）。由 `context_parallel_method` 门控。

## 10. HCCL(reserved 口径,D-2)

```
hccl = 200MB · N_comm
N_comm = 1(world) + 1[dp·cp>1] + 1[tp>1] + 1[ep>1] + 1[pp>1] + 1[dp_replicate>1]
```
**不进 allocated 峰值**(真机 ep=2 证);`reserved_estimate = allocated_peak + hccl`。

## 11. PP 每 stage + VPP + 1F1B

- **层分配(D-8)**:首选显式 `layers_per_stage`(忠实 mindformers offset);否则均匀切。
- **1F1B warmup**:stage 的在飞 microbatch 数 `= min(pp−1−stage, m)`,每个钉一份激活(首 stage 最深)。
- **VPP(v>1,D-4)**:按 chunk 累加 `Σ_chunk n_c·(L/V)`;warmup `= (pp−stage−1)·2 + (v−1)·g`。

---

## 12. 标定常数汇总(全部有出处、非拟合 blob)

| 常数 | 值 | 依据 |
|---|---|---|
| 块对齐 `blk` | 512B | MindSpore `kDynamicMemAlignSize` |
| flash LSE 系数 | 64 (=2×8×4) | CANN `FlashAttentionScore` softmax_max/sum `[B,nh,S,8]` fp32 |
| AdamW 持久 `so` | 14 / 12 B/参 | bf16 2 + master 4 + m 4 + v 4 ／ fp32 4+4+4 |
| `k_ce` | 8(pp>1) / 4(pp=1) | profiler:流水线末 stage vs 单 stage 的满 vocab fp32 共存份数（制度相关） |
| `k_opt` | 6 | profiler(pp=2 stage0,AdamW 更新 op 链) |
| HCCL/组 | 200MB | 真机日志 hcclBufferSize(CANN 9.0),reserved 池 |

---

## 13. 整体求和公式

单 stage 峰值 = 持久项 + 逐事件桶和的最大,再与优化器-step 事件取 max:

```
P_stage = max(  persistent + P_optstep*                       ← §8,通常非峰
              , max_event [ persistent + Σ_bucket B_i(event) ] )
```

**主导事件通常是「loss 层反向」**(有 loss 的 stage、无重算),代入各式:

```
P_stage(loss-bwd) ≈  so·(N_dense/fsdp + N_expert/efsdp)        ← persistent §3
                   + Σ_层 a_层                                  ← act_live §4（无重算全存）
                   + S·B·V·bc + 4·S·B·V                        ← a_loss §7
                   + 4·S·B·V·(k_ce−1)                          ← CE bwd_scratch ①
                   + (V·H/fsdp)·bg                             ← head 满梯度
```

**无 loss 的 stage(如 pp 首 stage)** 峰值常是 §8:
```
P_stage ≈ persistent + k_opt·(V·H/fsdp)·4
```

全 stage:`tightest = argmax_s P_s`;`OOM = (max_s P_s > HBM)`;`reserved ≈ max_s P_s + hccl`。

## 14. 代入验证 + 真机锚点

DSv3 8L 无重算 pp=2(`B=2, S=4096, V=129280, H=1792`):
- **stage1 loss-bwd** = persistent + Σ层 + a_loss + `4·S·B·V·7`(①) ≈ **43.7 GB**(真机 45.7,**0.956**)。
- **stage0** = persistent + `6·(V·H/1)·4`(=6×883.8 MiB,②) ≈ **10.3 GB**(真机 10.25,**1.006**)。

**cp × 重算 真机验证矩阵(2026-07-07,Bug A + k_ce 制度化后)**:

| 锚点 | 真机 | 估计器 | ratio |
|---|---|---|---|
| DSv3 4L 全重算 | 12473 | 12409.5 | 0.995 |
| cp2 colossal/ulysses full(B=2) | 12433/12441 | 12409.5 | **0.998** |
| pp2-stage0(优化器 step) | 10246 | 10311 | 1.006 |
| pp2-stage1(loss,k_ce=8) | 45655 | 43659 | 0.956 |
| **cp2-none(loss,k_ce=4)** | 20119 | 18326 | **0.911**（修前 2.17× 过预测） |
| select self_attention | 18828 | 15361 | 0.816 ⚠️ |
| DSv4-fused | 15415 | 14336 | 0.930 ⚠️ |

**Bug A(loss ÷cp)+ k_ce 制度化把 cp 从 2.17× 过预测拉回 0.91**（真机机理正确）。cp2 full 的 0.998 现是
**真的对齐**（此前 0.996 是 B=1·full-S 与 B=2·S/cp 数值抵消蒙对,见 §7）。**select/DSv4 仍欠预测**（见 §15）。

**选择性重算(D-3,真机 2026-07-07,DSv3 8L dp=2)**——`recompute:{mode:select,select_module:{M:[0-7]}}`:

| select_module | 真机 | 估计器 | ratio |
|---|---|---|---|
| both(≈full,退化端) | 13953 | 13833 | **0.991** ✅ |
| `self_attention` | 18828 | 15361 | **0.816** ⚠️ |
| `feed_forward` | 19967 | 14554 | **0.729** ⚠️ |

**退化端(both==full)精确**,但**部分选择系统性欠预测 ~18-27%(OOM-不安全)**。根因非 D-3 机制,
而是**保留模块(尤其 MoE)的无重算激活 + 反向工作集**在 loss 峰值处欠计——**同 §D-10 ①/D-5 一族**
(profiler:峰值 live 里 14×112 MiB + 大量 FFN/MoE 小张量尾未建全)。详见
`analysis/realmachine/select_attn/DIAGNOSIS.md`。

## 15. 诚实边界 + 已知栈限制

- `k_ce/k_opt` 是 profiler 标定计数(共存数受 allocator/microbatch 影响,同平台常数档,非 op 图纯导出)。
- 未建模的小残差:DSv4-fused 7%(融合 kernel 内部量,D-5)、DSv3 ~0.5% sub-block 尾(rope cos/sin、norm rstd、cast 临时)。
- 未真机核对:cp>1(除 pp=2 外)、VPP、pp>2 每-stage、swap。
- **保留-MoE loss-峰值欠预测（select 0.816 / DSv4 0.930,OOM-不安全,尝试修未果）**:2026-07-07 试按机理
  补「MoE grouped-GEMM 反向再物化」(ep-中间量足迹)到 act_live——**失败**:该项对所有事件生效,把
  **无-loss stage 的逐层反向事件过度抬高**（pp2-stage0 10311→12082 过预测),而只在 loss-峰值需要。
  即 kept-MoE 欠计**仅在 loss-backward 事件**（多层 saves 与 loss 区共存那一刻），单一 act_live 项无法
  只补该事件而不误伤逐层反向。**已回退**。根因确认是「保留 MoE-FFN 前向 saves 在 loss 峰被简化 op 图
  漏建（真机 grouped-GEMM 的 permute/capacity-pad/cast 中间量）」,需**按事件定向**的 MoE 模型或显式
  margin,当前机理未及 → 保留为已知残差。cp 侧的欠计经查**非** kept-MoE 而是 Bug A/k_ce(已修)。
- **此 mindformers build 多维并行栈限制**(真机实测):① SP+MoE 不支持;② TP+MoE(Detach layout bug);
  ③ PP+重算互斥;④ **pp>2 崩在 pynative pipeline+优化器对 decoder-only 中间 stage 的 param/state
  配对(1D layernorm 权重 `[H]` 与 2D 投影权重 `[H,·]` 配错)**——MLA(`[H,rq]`)与 GQA(`[H,H]`)、
  dp=1/2 均复现,与注意力类型无关。故 pp>2 只有估计器预测、无法真机核对。

## 关联
- 整改历程:`2026-07-06-audit-remediation.md`（D-1 cp / D-2 HCCL / D-3 选重 / D-4 VPP / D-6 ungated /
  D-7 转换器 / D-8 stage / D-10 CE+optstep）。
- 设计:`2026-07-01-unified-llm-modelspec-design.md`（op 图 §7-10）、`2026-06-29-...design.md`（§8 内存）。
- 真机技能:`.claude/skills/real-machine-memory-sim/`;真机锚点数据 `analysis/realmachine/`。
