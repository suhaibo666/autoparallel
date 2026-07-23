# DSv4-Flash 内存估计标定 — 交接文档（2026-07-22）

> 场景：offline 解析式内存评估器 `pynative-cost-evaluator`（预测 MindFormers PyNative 5D 并行训练每卡峰值显存）。
> 起因：用户现场 `test.yaml`（DeepSeek-V4-Flash）导入失败 → 修好后仿真 18G vs 真机 58G（3.2×）→ 一路排查标定。
> 本文自包含：读完可直接接手，不需要之前的对话上下文。

---

## 0. TL;DR（当前状态）

- **导入已修**、**dp_shard 8× 误判已修**（这两条闭掉 3.2× 里的 ~2×，仿真 18119→32897）。
- **模型经真机验证是准的**：所有**能真跑的** MoE/dsv4_hybrid 配置，仿真都在 **−6% ~ +20%** 内（偏保守方向）。
- **唯一未闭**：现场 **DSv4-Flash（fused + full recompute + seq4096 + 256 专家）仿真 32897 vs 真机 58650（1.78× 偏低）**。
- **已排除**的原因（都实测/推演否定）：dp_shard（=32 已确认）、MoE 专家切分（efsdp=1=dp//ep，比值 1.0000 精确）、reserved 口径（185 实测 reserved−alloc 仅 +1.4%）、fused 注意力路径（185 实测 ±6~20%）。
- **唯一没排除的变量 = full recompute**（所有能验锚点都是 no-recompute）。
- **最强假设**：全重算下，**DSA indexer 的 `index_scores` 因要算 `indexer_loss`（KL loss，每步都在）不被释放、逐层常驻**，seq4096 下每层 ~GB；我的全重算模型把它当普通激活释放了 → 偏低。
- **下一步（唯一）**：在 **185** 跑 **pp4 + full recompute + seq4096** 缩小锚点 → 若真机 ≫ 仿真，坐实假设，改 `dsv4_hybrid.py` 让全重算下 DSA/indexer 的 KL-loss 依赖张量常驻。
- **⟪2026-07-22 更新⟫ 锚点已跑完（结果全在 §11）：KL/index_scores 假设【证伪】；真根因【已定位】= fused SparseFlashMla 的 `ctx.save_for_backward`（11 个张量，seq4096 下 ~1–2 GB/层）在全重算下不释放、随 PP warmup 微批累积；仿真把全重算建成"只留 checkpoint_input"→ stage0 欠估 1.90×。修法与标定靶在 §11.5。**

---

## 1. 代码仓库 & 已提交的修复

- 仓库：`E:\97-codes\torch_parallel\pynative-cost-evaluator`，分支 `feat/unified-llm-modelspec`，remote `github.com/suhaibo666/autoparallel`。
- Web 服务 `serve_explorer.py` 部署在 116 的 `http://192.168.9.116:8850`（多用户/线程隔离；更代码后 `tar czf - --exclude=__pycache__ --exclude='*.pyc' cost_eval serve_explorer.py | ssh root@192.168.9.116 'tar xzf - -C /root/pynative-cost-evaluator && bash /root/pynative-cost-evaluator/run.sh 8850'`）。
- 提交末尾统一 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。并发分支开发**必须显式 git add 指定文件**再提交（有过被并发提交清掉暂存的教训）。

| commit | 内容 |
|---|---|
| `6e0a87f` | **yaml 导入 8 连环拦截点全修**（顶层段 checkpoint/lr_scheduler、optimizer Muon+旋钮键、model 异名 num_residual_streams→hc_mult/index_*→dsa_indexer_*/compress_ratios→csa_compress_ratios、parallelism moe_token_dispatcher/overlap 近似 warn、layers_per_stage list 写法、tiny-param FSDP2 ceil 补齐、Muon 优化器、`_bundle_to_fields` 补 optimizer/hc/muon_per_head、parse_yaml 捕获 Python 警告入 UI）。测试 `tests/test_dsv4_flash_yaml.py`。 |
| `2e8f6b2` | **dp_shard 解析修**：`data_parallel_shard: -1` auto 解析误用 `num_microbatches=1`（应为 pp）→ 把 256卡/pp8 的真 dp_shard=32 算成 256 → efsdp/持久欠估 8×。改用 `num_microbatches=pp`，加 `-1` 时警告。仿真 18119→32897。 |
| `4c896ab` | persistent 组成分解（参数副本/master/m/v，Muon 无矩阵 v），UI「持久态组成」卡片 + `StaticMem.persistent_breakdown`。 |
| `7b6e2a2` | **unfused dsv4_hybrid 激活修**（fork 做的）：indexer `index_scores` 补 `[B,S,n_idx=64,S]`（indexer.py:245）、CSA `kv_g_fp32`+2GB/层 fp32 副本（csa.py:490）、attn_weights fp32（csa.py:531）—— **仅 unfused 分支**（fused 15415.5 锚点不动）。**并揪出第二个 bug**：`eval_config` 丢了 `dsa_fused` → UI/yaml 路径一直用 fused（偏低）估计，已 surfaced。unfused 仿真 28172→36276（真机 45557 的 80%）。测试 `tests/test_dsv4_hybrid_unfused_activation.py`。 |

全量测试当前 **1389 passed**。

---

## 2. 两台真机（关键：谁能跑 fused）

### 116 = `192.168.9.116`（root，SSH key 配好；**跑不了 fused**）
- MindSpore **2.9.0**（`/root/miniconda3/envs/szy_py10/bin/python`）。
- **缺 fused 算子** `npu_sparse_flash_mla`（hyper-parallel 构建落后于 mindformers master）→ **fused DSv4 编译不起来**。
- mindformers：`/home/suhaibo/workspace/deepseek_v4/mindformers`（master）；memprobe/harness：`tests/st/.../test_deepseekv4/{run_ds3_memprobe.py,prep_dsv4align.py,prep_ds3_sim.py}`。
- 跑训练的 env 前缀（host，非容器）：`source /home/zdh/cann-9.0.0/set_env.sh`（host `/usr/local/Ascend` 是空的！）+ `source /home/suhaibo/vendors/custom_transformer/bin/set_env.bash` + `export TORCH_DEVICE_BACKEND_AUTOLOAD=0`（szy_py10 有坏 torch_npu 会 autoload）+ `PYTHONPATH=/home/suhaibo/workspace/pydeps:/home/suhaibo/workspace/deepseek_v4/mindformers:/home/suhaibo/workspace/deepseek_v4/hyper-parallel`（pydeps=补齐的 mindformers python 依赖，勿动共享 env）。
- 卡：910B3，共享机，跑前 `npu-smi info` 查空闲。

### 185 = `192.168.9.185`（root，SSH key 配好；**能跑 fused ✅**）
- 容器 **`shb_dsv4`**（镜像 `mindformers:dsv4-fused-ms2.10.0-cann9.1.0-beta.3-py3.11`）；所有命令 `docker exec shb_dsv4 bash -lc "…"`。
- MindSpore **2.10.0**，CANN 9.1.0-beta.3。**fused 算子在**：`from hyper_parallel.custom_ops.experimental import npu_sparse_flash_mla` OK（需 PYTHONPATH）。
- 代码：`/home/suhaibo/workspace/mindformers`（有 `run_mindformer.py` + `mindformers/pynative/trainer/trainer.py`）+ `/home/suhaibo/workspace/hyper-parallel`（有 fused ops）。
- **跑 fused 的必备**（缺一不可，都是踩坑记录）：
  1. `PYTHONPATH=/home/suhaibo/workspace/mindformers:/home/suhaibo/workspace/hyper-parallel`（用户 checkout 优先于镜像默认 `/home/work/*`，后者较旧）。
  2. `source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh`（否则 `init_process_group()` 报 `No module named 'hccl'`）。
  3. `pip install ninja`（fused ops 首次 JIT 编译 C++ 扩展 `hyper_parallel_custom_ops_ms`，缺 ninja 报错 —— **这正是 116 跑不了的另一半原因**）。
  4. **`deterministic: false`**（config 里）—— fused DSA 反向 `SparseFlashMlaGrad` 不支持确定性（`EZ1008 ... not support deterministic yet`）。
- 卡：8× 910B2，跑前查空闲（共享机；曾见别人 `test_eight_cards` pytest 抢卡）。
- **无现成 memprobe harness**，需自己写 thin runner（`Trainer(config).train()` + `ms.runtime.max_memory_allocated()/max_memory_reserved()` + 0.25s 采样线程持久化峰值到盘，防 teardown segfault）。

---

## 3. 真机锚点 vs 仿真（全部实测；这是标定的地基）

| # | 配置 | 真机 alloc / reserved (MiB) | 我的仿真 | 判定 |
|---|---|---|---|---|
| 1 | ds3 普通 MLA+MoE, pp1/dp2/ep2, no-recomp（116） | 20815 / 21116 | 19527 | **−6% ✅** |
| 2 | DSv4-align 4L **fused**（锚点，test_scorecard_anchors） | 15415.5 band | ~14972（在带内） | ✅ |
| 3 | **unfused** dsv4_hybrid 4L/seq2048/mHC4/8exp, pp1/dp2/ep2, no-recomp（116） | 45557 / 48326；静态 floor 15826 | 36276（修后） | 80%（未闭，缺逐桶归因） |
| 4 | **fused** dsv4_hybrid 4L/seq2048/mHC4/8exp, pp1/dp2/ep2, no-recomp（185） | **26499 / 29826**；静态 floor 14837 | **28172** | **+6% ✅**（夹在 alloc/reserved 间） |
| 5 | **fused** 规模探针 8L/32exp, dp8/ep8/pp1, no-recomp（185） | ~26251 / 28788；floor 11767 | 31453 | **+20% 偏高** |
| 6 | **DSv4-Flash 现场**（256卡, dp32/pp8/ep32, 43L+mtp, 256exp, seq4096, **full recompute**, fused, Muon） | stage0=43964, stage3-7≈58650（用户 `pp_peak_memory.sh 256 8`） | **32897** | **−44%（1.78×）❌ 未闭** |

**规律**：能真跑的（1/4/5，全 no-recompute）我**偏保守（偏高）**；唯独 DSv4-Flash（唯一 full-recompute）我**偏低**。同一模型不会又高又低 → 病在 full-recompute 特有的量。

---

## 4. 已排除的原因（别再回头查这些）

1. **dp_shard**：用户确认现场 256 卡，`dp_shard = 总卡/(pp·tp·cp) = 32`，**ep 不进这个除法**（ep 在 dp 内部对专家切）。我模型 dp_shard=32、efsdp=1，正确。
2. **MoE 专家过切**（用户怀疑点）：实测 `efsdp = dp_shard//ep = 32//32 = 1`，每卡专家 numel = 201,326,592 = 8 专家全份（3×4096×2048×8），**比值 1.0000 精确没多切**。persistent 12082 里专家部分正确。
3. **reserved 口径**：185 fused 实测 reserved−alloc 仅 +1.4%（301MiB/20815）；116 ds3 +1.4%；185 dsv4 +12%（碎片，2.7G）。**58650 是接近 alloc 的真值,不是碎片虚高**。
4. **fused 注意力路径**：185 实测 2卡 +6%、8卡规模 +20%，准。fused kernel 把 index_scores/kv_gathered 走 scratch 不物化（26G ≪ unfused 45G），我模型的 `dsa_fused` gating 正确。
5. **持久态整体**：185 fused 静态 floor 14837 vs 我 persistent 13468（91%，只差一点，是 grad 预留的记账口径差，非 bug）。

---

## 5. 唯一未闭 & 最强假设

**变量隔离**：#1/#4/#5（准/偏高）全 no-recompute；#6（偏低）唯一 full-recompute。**差别就是全重算这一个开关。**

**假设（待 185 pp+recompute 验证）**：DSA lightning indexer 前向产 `index_scores`，且**要算 `indexer_loss`（KL loss，真机每步日志都有 `indexer_loss≈1e-5`）** → 该张量**反向要用 → 全重算下也不能释放 → 逐层常驻**。seq4096、`index_n_heads=64` 下这是每层 ~GB 级。我的全重算模型（`RecomputeSpec.is_full` → 只留 `checkpoint_input`）把它当普通激活丢了 → 偏低。这是**「全重算释放规则」漏了 DSA loss 依赖张量**，与切分无关。

（注：`7b6e2a2` 修的是 **unfused 的 saves**，没管**全重算下 KL 依赖张量不释放**这件事；且现场是 **fused**，fused 下 index_scores 是 kernel scratch，但 indexer_loss 仍需某种常驻 —— 具体形状要在 185 实测 + 读 `indexer.py` KL loss 反向依赖确认。）

---

## 6. 下一步（唯一，能钉死）

**在 185 跑 `pp=4 + full recompute + seq4096`（8卡缩小版）**，这是所有能验锚点**唯一没覆盖**的组合。
- 若真机 ≫ 仿真 → 坐实「全重算下 DSA/indexer KL 依赖张量常驻」→ 改 `cost_eval/layers/dsv4_hybrid.py`：让 indexer 的 KL-loss 依赖张量（index_scores 或其反向所需量）在 `recompute.is_full` 下**仍计入 act_live（不可重算）**；对真值收敛，加回归测试。
- ⚠ **风险**：full recompute 在旧 build 有 `recompute() context_fn` 冲突 bug（MS 2.9 **和** 2.10 签名相同都可能中）；pp>1 + grad-accum 有 `_unsharded_param=None` bug。但 **185 是产出 58650 的同款 fused-DSv4 build，真机本来就这么跑出 58650 的，大概率能跑**。若 185 也撞 context_fn，则退而求其次：让用户从**现场那个 58650 run 的 rank 日志**直接 grep 逐 stage `Used/Actual peak memory usage` + 尝试拿 `memory_stats()` 分类，我按真值改。

---

## 7. 复现片段（仿真侧，直接可跑）

现场 yaml：`C:\Users\suhaibo\xwechat_files\suhaibo1993_9a8e\msg\file\2026-07\test.yaml`（已拷到 116 `/home/suhaibo/workspace/dsv4_flash_test.yaml`）。

**跑现场 DSv4-Flash 仿真**（得 32897）：
```python
import yaml, serve_explorer as S
from cost_eval.configs.from_mindformers import from_mindformers_dict
mf=yaml.safe_load(open(r"C:\Users\suhaibo\xwechat_files\suhaibo1993_9a8e\msg\file\2026-07\test.yaml",encoding="utf-8"))
mf2,vpp=S._mf_adapt(mf); w=[]; S._materialize_nested_offset(mf2,w)
fields=S._bundle_to_fields(from_mindformers_dict(mf2))
DEF={"preset":"custom","ffn":"3072","select":"attn","sel_layers":"","sel_ops":"","vpp":"1","mbs":"","grad_bytes":"4"}
q=dict(DEF); q.update({k:str(v) for k,v in fields.items() if v is not None})
r=S.eval_config(q); print(r["device_peak"])  # 32897；逐 stage r["stages"]
```

**缩小 dsv4_hybrid 配置**（185 跑的那个，改 `apply_dsa_kernel_fusion`/`use_fused_mhc` 切 fused/unfused）：deepseek_v4 / dsv4_hybrid / 4层 / seq2048 / hidden4096 / heads64 / q_lora1024 / kv_lora512 / qk_nope448 / qk_rope64 / v_head512 / o_groups8 / o_lora1024 / mHC num_residual_streams4 / DSA index_n_heads64,head_dim128,topk512 / compress_ratios[0,4,128,4] / 8专家 topk2 moe_inter2048 / first_k_dense1 / pp1 dp2 ep2 / no-recompute / global_batch2 / Muon / bf16。fused 仿真 28172，unfused 仿真 36276（真机 45557）。
- 直接 `build_llm_spec(b.llm)` 会因 `kv_lora_rank=0` / `csa_compress_ratios` 长度报错 —— UI 路径会把 kv_lora coerce 到 1；探针里 `dataclasses.replace(b.llm, kv_lora_rank=(b.llm.kv_lora_rank or 1), csa_compress_ratios=tuple(b.llm.csa_compress_ratios[:b.llm.num_layers]))`。

**185 跑 fused 缩小配置的 config**（用户 checkout 里已有 `/home/suhaibo/workspace/dsv4h_fused_pp1.yaml` 和 `dsv4h_fused_scale8.yaml`；memprobe runner `/home/suhaibo/workspace/run_memprobe_185.py`；数据 `dsv4h_data/`）。**config 必备字段**（否则崩）：`dsa_indexer_loss_coeff: 0.001` + `dsa_indexer_use_sparse_loss: true`（否则 indexer.py `mul(None)` 崩）、`n_group: 0`（n_group>0 需 topk_group）、`deterministic: false`（fused DSA grad 不支持确定性）。启动：
```
docker exec shb_dsv4 bash -lc "source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh; cd /home/suhaibo/workspace/mindformers && PYTHONPATH=/home/suhaibo/workspace/mindformers:/home/suhaibo/workspace/hyper-parallel ASCEND_RT_VISIBLE_DEVICES=0,1 msrun --worker_num=2 --local_worker_num=2 --master_port=9200 --log_dir=/home/suhaibo/workspace/log_X --join=True /home/suhaibo/workspace/run_memprobe_185.py --config /home/suhaibo/workspace/dsv4h_fused_pp1.yaml"
```

---

## 8. 相关代码位置（评估器内）

- `cost_eval/layers/dsv4_hybrid.py` — dsv4_hybrid op 图（indexer/CSA/sparse_attn/mHC-wrap），`dsa_fused` gating 在此。**下一步要改这里的全重算释放规则**。
- `cost_eval/layers/residual.py` — mHC（hyper_connection）残差流。
- `cost_eval/mem_timeline.py` — 时间线仿真：`simulate` 里 `recompute.is_full(lid)` → `saved = sm.checkpoint_input`（**全重算只留层入口**；漏 KL 依赖张量就在这条分支）；1F1B warmup `pinned[(mb,lid)]` 按微批累加（已验随 m 放大、pp 封顶）。
- `cost_eval/structure_mem.py` — 逐结构去重 + persistent（`(numel//divisor)*osb`，divisor=fsdp/efsdp；FSDP2 ceil 补齐）+ `estimate_select_memory`（选择性重算）+ checkpoint_input（首个 save op）。
- `cost_eval/static_mem.py` — `compute`（每 stage persistent）+ `persistent_breakdown`（分量拆解）。
- `cost_eval/parallel_model.py` — `dense_fsdp_degree()` / `efsdp_degree()`（=fsdp·tp//ep）。
- `cost_eval/configs/from_mindformers.py` — yaml→bundle 转换（`_build_parallel` 里 dp_shard/num_microbatches 解析、`_MODEL_KEY_ALIASES`、fail-loud 白名单）。
- `cost_eval/specs.py` — `OptimizerSpec`（AdamW/Muon）、`is_muon_matrix_weight`。
- `serve_explorer.py` — UI + `eval_config`（主入口）+ `_bundle_to_fields` + `parse_and_validate`。

---

## 9. 可复用的后台 agent（SendMessage 续跑，上下文还在）

- `a37ab5c74c57ed26f` — 116 上跑过 ds3 pp4/pp1、dryrun、unfused dsv4（有 116 全套 env/harness 上下文）。
- `accbe10d6cc704d16` — 185 上跑过 fused dsv4（2卡+8卡规模），**有 185 全套 env + 踩坑修复上下文**（deterministic/ninja/hccl/config 必备字段）。**跑 185 pp4+recompute 首选续这个。**
- `a202dd25127071288` — fork（改 unfused dsv4_hybrid 激活那次，7b6e2a2）。

---

## 10. 给用户的诚实口径（当前可对外说的）

- **普通 MLA/GQA + MoE + PP/FSDP/EP**：评估器 **±6% 准**，放心用。
- **DSv4-Flash 类（fused + full recompute + 256 专家 + seq4096）**：绝对值**偏低约 1.8×**，用时**乘 ~1.8 修正**、OOM 判断留足余量。
- 根因**已定位到 full-recompute 下 DSA/indexer KL 依赖张量常驻未建模**（非切分、非配置、非口径 —— 都实测排除），**待 185 pp+recompute 锚点验证后即可改对**。

> [!deprecated] 上一行的"KL 依赖张量"定位已被 §11 锚点实测**订正**：KL 假设证伪，真根因是 fused 注意力 ctx 保存集。对外口径改为 §11.6。

---

## 11. ⟪2026-07-22 追加⟫ 185 pp4+全重算+seq4096 锚点结果 — 假设证伪、根因重定位

（agent `accbe10d6cc704d16` 实测；配置/日志留在 185,见 §11.7。**pp4+full recompute 没崩**（无 context_fn / `_unsharded_param` bug），no-recompute 也没 OOM；两组各 3 步，loss ~11.94–12.0，`indexer_loss≈1.4e-5` 每步都在。rank→stage = `stage×2+dp_rank`。）

**锚点配置**（两组仅 recompute 不同）：8 层 dsv4_hybrid **fused**、pp4（2层/stage）/dp2/ep2/tp1/cp1 = 8 卡、seq4096、hidden4096、8 专家（ep2→4/卡）、Muon、bf16、global_batch 8（dp2×mbs1×**4 微批**）、deterministic false、`dsa_indexer_loss_coeff 0.001`/`dsa_indexer_use_sparse_loss true`/`n_group 0`。

### 11.1 真机逐 stage 峰值（MiB）

| stage (ranks) | **ON alloc** | ON reserved | **OFF alloc** | OFF reserved | **ON−OFF alloc** |
|---|---|---|---|---|---|
| 0 (0,1) | **24153.3** | 28472.0 | **30395.0** | 38418.0 | **−6241.7** |
| 1 (2,3) | 14641.7 | 16012.0 | 21019.4 | 22302.0 | −6377.7 |
| 2 (4,5) | 14097.7 | 15054.0 | 17799.9 | 19006.0 | −3702.2 |
| 3 (6,7) | 23508.0 | 28468.0 | 27822.0 | 30132.0 | −4314.0 |
| **device peak** | **24153.3 (s0)** | **28472.0** | **30395.0 (s0)** | **38418.0** | **−6241.7** |

真机上**全重算只省 6.2 GB alloc**（峰值 stage）。两组峰值都在 **stage0（1F1B warmup）**，stage3（head/CE）次之 —— 经典 1F1B 形状。

### 11.2 仿真对照（alloc MiB）与真/仿比

| stage | ON real | ON sim | ON 真/仿 | OFF real | OFF sim | OFF 真/仿 |
|---|---|---|---|---|---|---|
| 0 | 24153 | 12680 | **1.90×** | 30395 | 35801 | 0.85× |
| 1 | 14642 | 7981 | 1.83× | 21019 | 25839 | 0.81× |
| 2 | 14098 | 7615 | 1.85× | 17800 | 19538 | 0.91× |
| 3 | 23508 | 19599 | 1.20× | 27822 | 35661 | 0.78× |

**在受控 8 卡锚点上精确复现了现场签名**：no-recompute 仿真偏高 +18%、full-recompute 仿真偏低 −19%（256 卡现场即 32897 vs 58650）。交叉变量 = full recompute,已隔离。

### 11.3 决定性证据：ON−OFF 差值

- **真机**全重算省（stage0）：**6242 MiB**。
- **仿真**全重算"省"（stage0）：**23121 MiB**（device-peak 口径 16202）。
- → **仿真把全重算能释放的量高估了 ~3.7×**：仿真认为每微批 act_live 塌缩到 ~1.3 GB（只剩 checkpoint_input），真机 stage0 仍压着 ~15 GB。

### 11.4 假设判定

- **「index_scores 因 indexer_loss KL 不释放」= 证伪（fused 路径）**。源码证据（185）：
  - fused 的 `indexer_loss` 在注意力算子**反向里**才算，不从前向持有：`csa.py:303` `_compute_fused_indexer_loss(...)` 位于 `FusedSparseFlashMlaWithIndexerLoss.backward`（`csa.py:249`）。
  - 存的是**逐层 detach 标量**：`utils.py:41-56` `save_to_indexer_losses_tracker` 做 `loss.detach()` 进 `[num_layers]` 张量,且**重算期间 early-return**（`utils.py:42` `is_in_recompute()`）。无图、无内存量。
  - fused 下 `index_scores` 是 kernel scratch（`csa.py:220`、`:689`），Python 侧从不物化。
- **标定缺口本身 = 证实并重定位**：病灶在 **PP-warmup stage 的全重算 act_live**（`mem_timeline.py:505` `saved = sm.checkpoint_input`）。

### 11.5 真根因 + 修法

**fused SparseFlashMla 自定义算子每次调用 `ctx.save_for_backward` 存 11 个张量**（`csa.py:224-235`：query/ori_kv/cmp_kv/sparse_indices/query_index/key_index/weights/cmp_residual/sinks/output/softmax_lse），seq4096 下多个是 `B×S×heads×dim` 级（仅 `output` [1,4096,64,512]bf16 ≈ 268 MB），合计 **~1–2 GB/层**。这套 per-微批 ctx 状态**在全重算下不被释放**（MindSpore `use_reentrant=False` checkpoint,`activation_checkpoint.py:151`；wrapper `mindformers/pynative/distributed/activation_checkpoint.py:76-87,545-551,654`），随 1F1B warmup 微批累积：stage0 真机瞬态 ~15 GB ÷（4微批×2层）≈ **1.9 GB/微批层**,量级正对上 fused ctx 集。仿真把全重算建为"只剩 checkpoint_input(~1.3GB)+单层瞬态重物化"→ 完全漏掉 warmup 累积 → stage0 欠估 1.90×（stage3 无 warmup 堆积,只差 1.20×）。

**评估器修法（方向）**：`dsv4_hybrid` fused 层在 `recompute.is_full` 下保留量 = checkpoint_input **+ fused 注意力 per-微批 ctx 足迹**（KV/output/lse 按 seq×heads×dim），使 warmup `act_live×微批` 贴真机；顺带核 no-recompute 的 activation_saves（当前 +18% 偏高）。**标定靶**：ON s0=24153、OFF s0=30395、真实重算节省=6.2 GB（不是现在模型的 16–23 GB）。改 `cost_eval/mem_timeline.py:505-506,574-580` + `cost_eval/layers/dsv4_hybrid.py`（fused saves 集）+ `cost_eval/structure_mem.py:259-277,409`。

唯一没能纯从源码闭合的推断：MindSpore `use_reentrant=False` checkpoint 是否释放自定义 `_Function` 的 `save_for_backward` —— 但内存算术（真机只省 6.2 GB vs 模型 16–23 GB;峰值在 warmup stage）说明该状态**实质上没被释放**。

### 11.6 对外口径（替代 §10 第 2 条）

- DSv4-Flash 类欠估根因 = **全重算 × PP-warmup 交互**：fused 注意力算子 ctx 保存集（~1–2 GB/层@seq4096）不随全重算释放、按 warmup 微批数累积;非切分、非配置、非 reserved 口径（锚点全部固定排除）。
- 修复落地前,fused+全重算+PP 配置的仿真值按 **stage0 欠估 ~1.9×** 心算修正。

### 11.7 本次产物（185 `/home/suhaibo/workspace/`）

`dsv4h_fused_pp4_recomp.yaml`、`dsv4h_fused_pp4_norecomp.yaml`、`gen_dsv4h_data_4096.py`、`dsv4h_data_4096/`（seq4096 数据集,已验）、`log_dsv4h_pp4_recomp/`、`log_dsv4h_pp4_norecomp/`;仿真脚本在本机 scratchpad `run_sim_pp4.py`。收尾干净：0 残留进程、8 卡全空闲、共享源码未改（仅 `kernel_meta/` JIT 缓存）。
