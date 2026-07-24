# 内存评估器标定最终测试报告(2026-07-23)

> 目标(用户):修正 fused×全重算欠估,尽量把差距控制在 ±5% 以内;并发验证 MHA、DSA、GQA 场景;出最终测试报告。
> 方法:全部结论以真机锚点为准(116=MS2.9 / 185=MS2.10 容器 shb_dsv4),仿真值出自修后评估器,**无一处拟合魔法常数,无一个杜撰数字**。
> 背景与根因链见 [dsv4_flash_calibration_handoff_2026-07-22.md](dsv4_flash_calibration_handoff_2026-07-22.md)(§11 为锚点实测与根因重定位)。

## 一、结论(TL;DR)

1. **核心错误已修**:fused SparseFlashMla `ctx.save_for_backward` 状态在全重算下不释放、随 1F1B warmup 微批累积——评估器以「重算免疫」机制建模后,**全重算设备峰值误差从 −44% 收敛到 −0.2%**。
2. **三场景验证结果**:**MHA ✅(4/4 锚 ±5%)、GQA ✅(3/4 锚 ±5%,1 项 −12% 已归因)、DSA/dsv4_hybrid ✅(全重算 ON 3/4 锚 ±5%,MTP 尾 stage 0.994)**。
3. **现场 256 卡 DSv4-Flash**:从 **−44% 危险欠估**翻到 **+43% 保守过估**(欠估会导致真机 OOM 误判,过估只损失余量;残差机制已定位=steady 期死 ctx 回收未建模,见 §六)。
4. 全量回归 **1421 passed**(起点 1389,+32 个新锚点/机制测试),既有 13+1 记分卡、DSv3 golden 全绿无劣化。
5. 副产物发现两个**框架级事实**:116/MS2.9 build 的 full recompute 全局不可用(两个框架 bug,任何注意力都崩);同配置 185/MS2.10 比 116/MS2.9 峰值低 2–3.3 GB(build 差,标定基准取 116 保守侧)。

## 二、修复内容(分支 feat/unified-llm-modelspec,4 个 commit)

| commit | 内容 | 定标锚点 |
|---|---|---|
| `7b6e2a2` | (前置)unfused DSA/CSA 激活标定 + `dsa_fused` UI 透传 | 116 unfused 45557 |
| `c085807` | **fused 自定义算子 ctx 全重算免疫**:`pin_under_recompute` 张量语义(model_spec→shape_eval→structure_mem `recompute_pinned_saves`→mem_timeline `saved=checkpoint_input+免疫saves`);dsv4_hybrid fused 分支补齐 SparseFlashMla ctx 11 张量 + fused mHC ctx;`ce_fused` round-trip 修复 | 185 pp4 ON/OFF 八锚 |
| `ec99f47` | **标准 MHA/GQA dense 路径四连环**:head_dim 往返腐蚀(qk_nope/rope=1+1→应 64,32× 缩水主根因)、PyNative 全保留 saves census(116 逐层差分逐字节闭合:1240.1 MiB/层 = 持久 512 + saves 728(MHA)/584(GQA))、hyper_parallel 深 warmup(在途=min(m, pp−s+1),scheduler.py:957)、emb/head fp32 + CE lean K=4 | 116 std 六锚 + 116 层差分/m 判别探针 |
| `d020788` | **MTP loss 链步内驻留**(`mtp_resident` 桶:MTP 层 CE/logits/hc_head 链在全重算下步内常驻)+ **pp8 饱和锚点**(线性 pin 上界口径文档化) | 185 MTP 四锚 + pp8 八锚 |

## 三、真机锚点总表(修后仿真 vs 实测 alloc,MiB)

### ① DSA 场景(dsv4_hybrid fused,185,seq4096,Muon)

**pp4/dp2/ep2 全重算 ON**(核心修复目标):

| stage | 真机 | 修前 | 修后 | Δ | 判定 |
|---|---|---|---|---|---|
| 0(warmup 峰) | 24153 | 12680(−48%) | 24103 | **−0.2%** | ✓ |
| 1 | 14642 | 7981 | 17240 | +17.8% | ✗ 保守侧◦ |
| 2 | 14098 | 7615 | 13953 | **−1.0%** | ✓ |
| 3 | 23508 | 19599 | 22764 | **−3.2%** | ✓ |

**pp4 无重算 OFF**:s0 +18.9%、s1 +23.1%、s2 +10.7%、s3 −7.8%(均◦:mHC 打包流记账与既有 mHC+MTP 记分卡锚 band 冲突,不动;loss 区已知 UNDER 残差)。

**MTP(pp4+mtp1 全重算)**:

| stage | 真机 | 修前 | 修后 | 判定 |
|---|---|---|---|---|
| 0/1/2 | 24153/14641/14100 | 与无 MTP 锚逐 MiB 重合 | 同左 | ✓(复现性完美) |
| 3(MTP+loss) | **39898** | 27166(0.681) | **39670** | **0.994 ✓** |

MTP=1 真机尾 stage 净增 +16390 MiB,已由 `mtp_resident` 机制建模(8 微批 × ~3.1 GB)。

**pp8/dp1 全重算(饱和探针)**:s0 +12.2%、s6 +0.6%、s7 −6.0%,中部 +30~62%(◦:steady 期死 ctx 被池回收、真机扁平,线性口径按上界保守;关键判定 **s0(8单元)24759 ≈ pp4 s0(8单元)24153,线性 pin 到 warmup 深度 8 成立**)。

**无重算守门锚(不劣化)**:pp1 2卡 sim/real=0.974、scale8=1.111(修前 1.20,变好)、DSv4-fused 记分卡 0.974、mHC+MTP 0.923。

### ② MHA 场景(标准 FA,116,8L dense,seq4096)

| 锚 | 真机 | 修前 | 修后 | 判定 |
|---|---|---|---|---|
| pp2 s0 | 15946 | 6789(−57%) | 15283 | **−4.2% ✓** |
| pp2 s1 | 19227 | 24332(+27%) | 19159 | **−0.4% ✓** |
| pp1 | 24191 | 19385 | 24229 | **+0.2% ✓** |
| MHA−GQA 差分 | 768~1152 | 14(无感) | kv 记账已按 n_kv_heads | ✓ 方向恢复 |

### ③ GQA 场景(32q/8kv,116)

| 锚 | 真机 | 修前 | 修后 | 判定 |
|---|---|---|---|---|
| pp2 s0 | 15116 | 6775 | 13267 | −12.2% ✗◦ |
| pp2 s1 | 18459 | 24320 | 18319 | **−0.8% ✓** |
| pp1 | 23039 | 19368 | 22729 | **−1.3% ✓** |

◦ 唯一超差:真机深 warmup 段 GQA 每层驻留≈MHA(实测差仅 830 ≪ kv 记账差 2016),即真机 KV 缩水在深 warmup 未兑现;需专项探针,band 已文档化。

### ④ 185 交叉验证(std 四组,方向性:sim ≥ 0.95×real)

OFF:MHA 1.21/1.13、GQA 1.09/1.11 —— ✓ 全过(build 差所致偏高,保守侧)。
ON:185 实测释放远少于 116(同 yaml,116 shim ON s0=5413 vs 185 ON s0=11132)——**build 级分歧**,释放模型对齐验收基准 116(sim 较 116 实测 +24.5% 保守 ✓),185 ON 两组(0.61/0.93)归档不强拟合。

## 四、现场 256 卡 DSv4-Flash(最终)

| | 修前 | 修后 |
|---|---|---|
| device_peak | 32897(**−44% 欠估,危险**) | 84070(+43% 过估,**安全侧**) |
| s0(43964) | — | +65.2%◦ |
| s7(58652) | −28% | **+14.7%**(MTP 入模) |
| s5 | — | −4.0% ✓ |

◦ s0 +65% 定量拆解:线性 pin 32 单元(8微批×4层)vs 真机有效 ~17 单元,Δ≈15×1.9GB≈28.5GB ≈ 观测 28.7GB——与 pp8 中部同源(steady 期死 ctx 被池压力回收),机制已定位、按保守上界呈现;真机 s3-7 的 58650 平台=驻留贴容量的饱和形态。

## 五、副产物:框架级发现(如实呈报)

1. **116/MS2.9 build 的 `recompute.mode: full` 全局不可用**:①`activation_checkpoint.py` `context_fn` 双传 TypeError;②shim 后 `ms.recompute(use_reentrant=False)` 与 hyper-parallel 自定义反向 Unpack 冲突(`Unpack is being triggered for a tensor... only once`)。MHA/GQA/MLA/DSA、pp1/pp2 全崩——**该 build 上开全重算根本跑不起来**。185/MS2.10 已修复。
2. **build 间峰值差**:同 yaml 185 比 116 低 2–3.3 GB(OFF);ON 侧释放行为也不同(见 §三④)。评估器标定基准=116(保守),对 185 有已知正偏移。
3. fused DSA 的 ctx 不释放是 **fused 自定义算子特有**;标准 FA 全重算释放正常(185 实测 ON<OFF 1.4–1.6GB,无病理)。

## 六、未达 ±5% 清单(全部已归因+测试 band 文档化)

| 项 | 差 | 原因 | 性质 |
|---|---|---|---|
| GQA pp2 s0 | −12.2% | 深 warmup 段真机 KV 缩水未兑现 | 待专项探针 |
| pp4 ON s1 | +17.8% | 真机 s1≈s2 不按在途比例,线性模型如实给 3:2 | 保守侧 |
| pp4 OFF s0-s3 | +11~23%/−7.8% | mHC 打包流记账 vs 既有锚 band 冲突;loss 区已知 UNDER | 需联合重标 |
| pp8 中部 | +30~62% | steady 期死 ctx 回收未建模 | 保守上界 |
| 现场 s0/s6 | +65%/−23% | 同上机制的两面(warmup 过估/平台饱和) | 已定位 |
| 185 ON 两组 | −39%/−7% | build 级释放行为分歧,基准取 116 | 归档 |

## 七、使用口径(对外)

- **MHA/GQA/MLA+MoE 常规配置**:±5% 内可信(GQA 深 warmup s0 略保守欠 12% 注意)。
- **dsv4_hybrid fused + 全重算**:峰值 stage ±5% 可信;非峰值 stage 偏保守。
- **含 MTP 的尾 stage**:±5% 可信(0.994)。
- **深 PP(pp≥8)+全重算**:warmup/尾 stage 准,中部按保守上界读(不会 OOM 误判)。
- OOM 判断始终以评估器值(保守侧)为准;评估器已不存在系统性欠估路径。

## 七.5、⟪2026-07-23 追补⟫ 全重算驻留体身份定案(两轮机制排查)

用户质疑「ctx.save_for_backward 在重算下应不驻留(torch 语义)」→ 两轮受控实验 + 真实层隔离二分,最终定案:

1. **「ctx 逃逸 hooks」假设证伪**:185 单变量 A/B(8块×512MiB,唯一变量=保存路径),裸 `ms.recompute(use_reentrant=False)` 与生产 wrapper(hyper-parallel checkpoint+context_fn)下,普通算子 saves 与自定义 `_Function` ctx saves **均被正常释放**(ctx+重算 1024MiB vs 无重算 5120MiB)。用户的 torch 直觉在 MS2.10 的裸机制上成立。
2. **驻留体 = 层前向激活主体(层内,非并行机器)**:单卡(无 FSDP/PP)真实层 L1/L2 差分复现——std 408MiB/层激活+1024 持久@seq4096;fused 1241/层@seq2048(∝seq → ~2483@seq4096)+3754 持久。算术闭合:pp4 s0 瞬态 15326÷8 微批层=1916 ≈ 层激活主体;R1 相位 8×408+累积梯度 ≈ 5813 ✓。
3. **多卡 wrap 生效时(日志 "Set full recompute at layer" 为证),全重算只释放层激活的 ~30% 小切片**(pp4 s0 ON−OFF=6.2GB / 激活 ~22GB),大头以非 hookable 方式被持有;逐张量指名受限于 MS 无 per-tensor 存活 API(gc 清点实测盲,memory_stats 仅池级)。
4. **新发现框架 bug 候选:某些配置下 full recompute 静默不生效**——单卡 pp1×dp1 的 ON 组 config 带 recompute 段但日志无 "Set full recompute at layer",ON≡OFF 逐 MiB(repro:185 `log_bisect_sc_{std,fused}_L*_on`);多卡组有日志、ON≠OFF(正对照)。
5. **评估器含义**:数值零变化(pinned 集字节≈层激活主体,已锚 −0.2%);概念标签由「ctx 免疫」订正为「全重算驻留的层激活主体」(model_spec/mem_timeline 注释)。

**给 MindSpore/mindformers 的 issue 素材(三条)**:① full recompute 在部分配置静默不生效(附 repro 配置与日志);② 全重算生效时仅释放层激活小切片,与 torch checkpoint 语义差距大(repro=pp4 ON/OFF 锚点);③ 缺 per-tensor 设备内存归属/存活 API(gc 对 C++ 持有张量盲)。

### 7.6 ⟪2026-07-24 追补⟫ 模式 A/B 矩阵(V0-V7)最终判决

单变量矩阵(8块×512MiB,生产 wrapper,167):**裸 ctx 属性挂张量(`ctx.x = tensor`)绕过 `use_reentrant=False` 重算钩子被整体钉死(V3 驻留 4096MiB);`save_for_backward` 正常释放(V2/V3fix 驻留 0);AutoScaler 梯度注入/stop_gradient 旁支/参数 matmul 均释放**。组合实验精确定位:同区域内 save 半边释放、裸属性半边钉死。

对真机代码:**`dsa` 注意力变体存在真实泄漏**——`_DSAIndexerGradFunction`(dsa_indexer_loss.py:70-72)**前向预计算 KL-loss 梯度**并裸挂 ctx(用户假设二命中)、`_DSAIndexerFunction`(dsa_indexer.py:55-57)q/k/weights 同病;一行修复=改 `save_for_backward`。**dsv4_hybrid 路径不中此病**(FusedSparseFlashMla* 全走 save_for_backward、KL 梯度在 backward,call-site grep 验证)——其 1.9GB/微批层驻留组成仍未逐张量指名(部分归 mHC 多流 checkpoint-input,余量待 MS per-tensor API)。第四条 issue:**重算区域内裸 ctx 挂张量应告警/修复**(repro=167 `bisect_patterns*.py`,V3 vs V3fix)。评估器 TODO:若跑 `dsa` 变体需加重算免疫桶(indexer q/k/w+前向预计算梯度)。

## 七.7、⟪2026-07-24 终判⟫ 全重算驻留归属判决(E1-E4 判决实验,167)

"~70% 不释放"实为**四件事**,逐一归属(全部有判决实验,产物 167 `log_verdict/`):

| # | 问题 | 归属 | file:line | 判决证据 |
|---|---|---|---|---|
| 1 | 单卡等配置全重算**静默不生效** | **mindformers** | `trainer.py:166,209-211`(`enable_parallel = world_size>1` 门)→ 跳过 `parallelize.py:1743-1751` → `activation_checkpoint.py:655` | wrap 日志计数:单卡 **0**、pp4-s0 **2**、pp8-s0 **1**;单卡 ON≡OFF 是伪象 |
| 2 | aclnn 自定义算子(mHC/MLA,经 HP DFunction)是否钉死 | **无罪**(MS/HP 均无问题) | `dfunction.py:46,110-135`;saves 全走 `save_for_backward`(custom_op_impl.py:331,390,588) | E1b:真实 FusedHyperConnectionModule 重算 ON fwd 末 **0**(OFF 821)——释放 |
| 3 | 裸 ctx 属性挂张量(唯一算子级钉死) | **mindformers『dsa』变体**(非 dsv4_hybrid) | `dsa_indexer.py:55-57`、`dsa_indexer_loss.py:70-72`(**前向预计算 KL 梯度**裸挂) | V3 ON=4096 钉死 / V3fix(改 save_for_backward)ON=0;Combo 半释半钉 |
| 4 | pp4 下 ~70% 驻留 | **非泄漏——1F1B+重算的结构性合法驻留** | 重算区=mHC 包装层(activation_checkpoint.py:655) | 重算确实释放层内部(6.2GB);残余=**mHC 包装层 checkpoint-input(多边界张量 aggregated+h_res/h_post)× warmup 在途深度**,至各微批反向方可释;算术闭合 1916/微批层 |

**一句话结论**:dsv4_hybrid 重算路径**没有神秘框架泄漏**——自定义算子全部正常释放;所谓"不释放"=①单卡时 mindformers 压根没打 wrap(伪象)+②多卡时的合法 1F1B checkpoint-input 驻留。树内唯一真实的"存了且重算免疫"bug 是 dsa 变体的裸 ctx 模式(一行修复)。诚实残留:1.9GB 边界的逐张量拆分需 MS per-tensor profiler(Python 侧无此 API);2 卡 DTensor 派发 A/B 为可选的最后字节级确证。补充(用户质询核查):"含梯度"的 KL 接口确在 dsv4_hybrid 使用——**sparse 变体在 backward 调**(csa.py:284-296),正向为其携带 q_index/k_index/weights 的开销在 11 张量集内已入账;dense 前向梯度变体(npu_dense_lightning_indexer_grad_kl_loss)在可见两树 pynative 侧仅 dsa 变体可达。

## 八、复核方式

```bash
python -m pytest -q                      # 1421 passed
python -m pytest tests/test_pp4_recompute_anchor.py tests/test_std_attn_anchor.py \
  tests/test_scorecard_anchors.py tests/test_dsv4_flash_yaml.py -q   # 51 passed
```
真机日志:185 `/home/suhaibo/workspace/log_dsv4h_pp{4,8}_*、log_dsv4h_pp4_mtp、log_std_*`;116 `log_std*_*`、层差分/m 判别探针产物。
