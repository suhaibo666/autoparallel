# 交接文档 — 评估器纯理论口径 + 框架缺口暴露(2026-07-24)

> 场景:offline 解析式显存评估器 `pynative-cost-evaluator`(预测 MindFormers PyNative 5D 并行每卡峰值)。
> 本文自包含:读完可直接接手,不需要之前的对话上下文。分支 `feat/unified-llm-modelspec`,HEAD `bb5f826`(已 push,remote github.com/suhaibo666/autoparallel)。
> 铁律(全程):**绝不杜撰真机数字;不用拟合常数顶数字(用结构量+真机锚校准)**。commit 末尾 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。

---

## 0. TL;DR — 评估器现在是什么口径

**2026-07-24 做了一次口径级切换**:评估器从"锚定真机、经验补偿"改为**纯 mindformers/MindSpore 理论口径 + 框架缺口显式暴露**。含义:
- **结构桶(参数/优化器/梯度/gather/激活/反向工作集/optstep)按代码语义如实估计**,已全部对齐真机 CSV 逐块数据(误差 ≤±5%,optstep 可配调准)。
- **全重算下"层激活主体不释放"这件事不再用经验 pin 顶平**——纯理论只保留重算区域边界(checkpoint_input,bf16 128/微批层),真机多驻留的量作为**框架缺口**(`FrameworkGapWarning` + 报告 §八/§8.5),不吸收进数字。
- 结论:**理论峰值 = 结构下界;真机 = 结构 + 框架缺口**。OOM 判断按"理论 + 缺口栏"读,勿直采理论峰值。

现场 test.yaml(256卡 fused DSv4-Flash)当前:理论 device_peak **38293** vs 真机 **58652**(理论=真机 65%,差 = 框架缺口)。

---

## 1. 全部相关 commit(本轮,时间倒序)

| commit | 内容 |
|---|---|
| `bb5f826` | **Muon NS workspace 倍数开放可配**:`OptimizerSpec.ns_workspace_mult`(默认3=旧常量,逐字节不变)→ simulate 参数 → yaml `optimizer.ns_workspace_mult`/UI「Muon NS workspace 倍数」/round-trip 三入口 |
| `7987cca` | **三结构桶补齐(§8.5 CSV 锚)**:①FSDP re-gather gather_buf(前向2unit→全重算反向Σ全重算层 param_full);②Muon NS 反向重叠 optstep(互斥假设→反向峰叠加);③全重算反向工作集 bwd_working_set(fml−bwd_scratch)。全 gated `pp>1+is_full`(OFF/pp1/非Muon 逐字节不变) |
| `4b135ba` | §8.5 unfused 真机内存池逐块对账,框架缺口首次源码级指名(csa.py/indexer.py) |
| `7a08dd2` | **口径切换主 commit**:去全部经验 pin(dsv4 fused ctx/mHC/std ON/饱和cap/MTP拟合)、checkpoint_input 修 bf16 128、FrameworkGapWarning、报告 §八 |
| `a9be44a`~`5ed21f0` | (前序机制排查,已完结,见报告 §7.5-7.9:E5 两卡 DTensor 判决、模式矩阵、驻留体身份定案) |

全量测试基线 **1463 passed**(复核过,非转述)。

---

## 2. 逐桶对账(unfused DSv4-Flash stage7,唯一有 CSV 逐块真机数据的 stage)

真机来源:现场 `memory_block_sync_fixed.csv`(UTF-16,63860 块,带 `python_stack` 源码归属)。独立复核:存活集重建 = `actual_used_memory` 38390 MiB 逐字节吻合。**此池仅瞬态**(持久参数+优化器在独立池,`is_persistent` 块共 29.5 KiB)。此 CSV 是 **stage7**(末 stage,MTP+CE 共存)。

| 桶 | 理论 | 真机 CSV | 差 | 算法 / 真机源码 / 差因 |
|---|---|---:|---:|---|
| **持久态** | 12082 | (独立池,不在CSV) | — | 参数副本 bf16 2404 + Muon master 4807 + momentum 4807 + v(仅非矩阵)64。Muon 口径正确、+持久后整机50.5G 与58G cap 自洽。**理论应最准,CSV 无法交叉验但预算无破绽** |
| **gather_buf** | 10010 | 10491(`param.py:519`,102块) | −4.6% ✓ | Σ本stage全重算层 param_full(全重算反向 re-gather 整段)。差=块对齐残差 |
| **grad_buf+grad_accum** | 2020+4807 | 6845(`param.py:862`) | +0.3% ✓ | 梯度路径**最准** |
| **optstep(Muon NS)** | 1536 | 2264(`muon.py`,877块) | −32% ◐ | 一份最大矩阵 NS(mult=3);真机多专家 NS 并发。**可配** `muon_ns_mult` 调准(→4.4 顶到2264) |
| **激活+反向** | 3798+4040 | ~18790 | **+10952** | 见下,拆两半 |

**激活+反向 +10952 拆解(源码归属)**:
- **框架缺口 ~9.1 GiB**(纯理论正确清零,真机不释放,FrameworkGapWarning 暴露):`csa.py:488` q/kv fp32 cast **5632** + `csa.py:510/512` softmax内部 1281 + `indexer.py:246` index_scores O(S²) 1024 + `indexer.py:380` KL-loss fp32 softmax 1024 + 杂项 192。评估器 unfused 分支**结构上认得**这些(`dsv4_hybrid.py:26-35`),纯理论口径把 saves 清零 → 落缺口。
- **反向工作集 ~3.3 GiB(已修 bwd_working_set)**:`autograd_compat.py:204` 7366(dL/dact + MoE grouped-gemm 反向)+ `linear.py:135` 269。

**结论**:结构桶全部对齐真机(≤±5%);唯一大差 = 框架缺口 ~9G(MS2.10 全重算不释放 unfused fp32 物化,已源码指名,非评估器误差)。

---

## 3. 可配旋钮(用户校准入口,不改代码贴合真机)

| 旋钮 | 位置 | 默认 | 作用 |
|---|---|---|---|
| `muon_ns_mult` / `OptimizerSpec.ns_workspace_mult` | yaml `optimizer.ns_workspace_mult` / UI「Muon NS workspace 倍数」 | 3.0 | Muon NS workspace 倍数(×分片×4B)。真机 NS 实现/融合不同可调;→4.4 顶到 CSV 2264 |
| `dsa_fused` | yaml `apply_dsa_kernel_fusion` / UI | True | fused=中间量走 kernel scratch;unfused=物化 kv_gathered/index_scores/attn_weights(结构上认得,全重算下纯理论清零→框架缺口) |
| `muon_per_head` | UI/opt | False | per-head NS 按头切(注意力投影 optstep ÷n_heads) |
| `ce_fused`/`ce_lean` | UI | 架构默认 | CE 融合/lean 口径 |
| `bwd_scratch_conservative` | HardwareSpec | False | 反向 scratch 保守/估计双模式 |

---

## 4. 框架缺口清单(四条,报告 §八 定稿,给 MindSpore/mindformers 提 issue 用)

1. **全重算释放缺口**:理论边界 128M/微批层 vs 真机 ~1.9G;MS2.10 全重算只释 ~30%(pp4 ON−OFF 仅省 6.2/21.6GB)。**E5 两卡 DTensor 判决证非钩子旁路**(saved_tensors 机制正常释放)→ 1.9G 是**重算区域 checkpoint-input(mHC 多流边界)× 1F1B warmup 在途深度**,是**合法结构 + 模型设计(hc_mult=4)放大**,非泄漏。unfused 额外的 csa/indexer fp32 物化(§8.5 指名)真机不释放,归此缺口。
2. **单卡/非并行静默不 wrap**:`trainer.py:166,209-211`(`enable_parallel=world_size>1` 门跳过 apply_recompute)→ **mindformers bug**(单卡开全重算等于没开)。
3. **dsa 变体裸 ctx 挂张量**:`dsa_indexer.py:55-57` / `dsa_indexer_loss.py:70-72`(**前向预计算 KL 梯度**裸挂 ctx,绕过重算钩子被钉死)→ **mindformers bug,一行修复**=改 `save_for_backward`(V3fix 证)。**dsv4_hybrid 不中此病**(走 save_for_backward)。
4. **MS 缺 per-tensor 设备内存归属 API**:gc 对 C++ 持有张量盲、memory_stats 仅池级。(现场 memory_block_sync CSV 恰好补上了这个能力——带 python_stack,这是 §8.5 能做源码归属的关键。)

---

## 5. 真机资源(校准/复验用)

- **原 185 机器 IP 已变为 192.168.9.167**(185 现在是别的机器,勿连)。容器 `shb_dsv4`(docker exec),MS2.10/CANN9.1;能跑 fused。SSH key 对 167,但**167 网络时通时断**(CorpLink VPN path-MTU 黑洞,用 `fix-116-network` skill 诊断;IP 可能再变,连不上先 `ssh-keygen -R` + 验证 workspace 里的 log_probe/dsv4h yaml 确认是本机)。
- **116**(192.168.9.116):MS2.9,**跑不了 fused**(缺算子+ninja,且全重算全局崩=框架 bug②);web 服务部署在 `http://192.168.9.116:8850`(更代码后 `tar czf - --exclude=__pycache__ --exclude='*.pyc' cost_eval serve_explorer.py | ssh root@192.168.9.116 'tar xzf - -C /root/pynative-cost-evaluator && bash /root/pynative-cost-evaluator/run.sh 8850'`)。
- **真机锚点汇总**见报告 §三(185 pp4/pp8/MTP、116 std MHA/GQA);**现场完整八 stage** 见 §7.8(43964…58652);**unfused CSV 逐块**见 §8.5。
- 现场输入文件:`C:\Users\suhaibo\Desktop\test.yaml`(fused)、`test-unfused.yaml`、`memory_block_sync_fixed/memory_block_sync_fixed.csv`。分析脚本在 scratchpad(`probe1-7.py`/`final.py` 等)。

---

## 6. 复现片段(仿真侧,直接可跑)

```python
import warnings, yaml; warnings.simplefilter("ignore")
import serve_explorer as S
from cost_eval.configs.from_mindformers import from_mindformers_dict
mf=yaml.safe_load(open(r"C:\Users\suhaibo\Desktop\test.yaml",encoding="utf-8"))
mf2,_=S._mf_adapt(mf); w=[]; S._materialize_nested_offset(mf2,w)
fields=S._bundle_to_fields(from_mindformers_dict(mf2))
DEF={"preset":"custom","ffn":"3072","select":"attn","sel_layers":"","sel_ops":"","vpp":"1","mbs":"","grad_bytes":"4"}
q=dict(DEF); q.update({k:str(v) for k,v in fields.items() if v is not None})
r=S.eval_config(q)     # r["device_peak"], r["stages"][i]["peak"], ["timeline"] 逐事件 buckets
```
逐桶 dump:取 `r["stages"][i]["timeline"]` 里 total 最大的事件的 `buckets`;持久分解 `r["stages"][i]["persist_breakdown"]`;结构层激活 `r["stages"][i]["graph"]`(每层 act_mib/entry_mib/full_act_mib)。

---

## 7. 遗留 / 下一步(按优先级)

1. **框架缺口逐 stage 化(可选)**:§8.5 只对了 stage7(CSV 所在)。若要现场各 stage 的框架缺口量,需各 stage 的 CSV 或用 stage7 的"unfused fp32 物化/层 × 层数"外推(评估器可加一个"框架缺口估计"提示列,不进 OOM 数字)。
2. **optstep 多专家 NS 并发(−32%)**:当前一份最大矩阵。若要建准需真机 NS 并发度(几个专家权重同时 NS)——或直接让用户用 `muon_ns_mult` 调。
3. **fused 路径无 CSV 逐块**:现场只给了 unfused CSV。fused 的框架缺口(csa fused kernel 走 scratch,理论上比 unfused 小)未逐块验证;若拿到 fused CSV 可同法对账。
4. **MS/mindformers issue**:四条素材齐(§4),尤其 bug②③可执行(单卡不wrap、dsa 裸 ctx 一行修复)——用户决定是否提。
5. **167 真机复验**:口径切换后未在真机重复跑(纯代码变更);若要字节级确认三结构桶修复,需 167 上 unfused pp8 重跑取 CSV。

---

## 8. 关键文档索引

- **最终测试报告(主文档)**:`analysis/dsv4_calibration_final_report_2026-07-23.md`——§一-七 标定历程、§7.5-7.9 全重算驻留机制排查终判、§八 口径切换声明+框架缺口清单、§8.5 unfused CSV 逐块对账。
- **前序交接**:`analysis/dsv4_flash_calibration_handoff_2026-07-22.md`(标定起点、真机 env recipe)。
- **本文**:口径切换后的当前状态 + 逐桶对账 + 可配旋钮。
