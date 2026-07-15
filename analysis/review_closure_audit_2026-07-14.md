# 逐条答复闭环审计：不是“32 项全修”，而是 8 项闭环、15 项部分闭环、9 项仍开放

> **评估器基线**：`feat/unified-llm-modelspec` @ `96630a5`，2026-07-14  
> **被审答复**：`analysis/review_response_2026-07-14.md`  
> **真机基线**：`192.168.9.116` / `shb.ms.2.9` / MindFormers `master` @ `97466872d` / MindSpore 2.10 / 2×Ascend 910B3  
> **闭环判据**：原问题的功能语义已实现并有正向/反向证据；仅 fail-loud、warning 或文档化边界记为“部分闭环”，不等同于功能实现。

## 1. 结论

`review_response_2026-07-14.md` 做到了 **32 项全部答复**，但没有做到 32 项全部闭环。按原报告问题边界和当前公共核心 API 复核：

| 状态 | 数量 | 项目 |
|---|---:|---|
| **闭环** | **8** | P0-01/03/04/05，P1-03/04/05/18 |
| **部分闭环** | **15** | P0-02，P1-01/02/06/09/10/11/16/17/19，P2-01/03/06/07/08 |
| **仍开放** | **9** | P1-07/08/12/13/14/15，P2-02/04/05 |

没有发现一整项原审查结论属于假阳性。两处子结论需要收窄：

1. P1-08 是保守高估而非 OOM 不安全的低估；但原报告本就标为“高估为主”，所以不是假阳性。当前仍直接求和各 op scratch（`cost_eval/structure_mem.py:188`），准确性问题真实存在。
2. P1-15 的 recv 激活主体可由首层输入 `act_live` 隐含覆盖；但 send、通信期间别名/共存、overlap 双缓冲、GPipe/zero-bubble 和显式 VPP chunk 配额仍未建，因此只能说原描述的 recv 部分过宽，不能推翻整项。

## 2. 新鲜验证证据

### 2.1 本地回归

- `python -m pytest -q tests/test_review_evidence.py -vv`：**7 passed**。
- `python -m pytest -q`：**475 passed**。
- `python -m compileall -q cost_eval serve_explorer.py`：通过。

这些结果证明已编码断言没有回归，但不能单独证明回复中的语义主张。尤其 `test_reshard_policy_changes_the_gather_lifetime` 只检查 always/never“不相等”，`test_optimizer_event_retains_accumulated_gradient_buffers` 只检查 `grad_accum > 0`（`tests/test_review_evidence.py:81-96,157-167`）。

### 2.2 语义反例探针

运行：

```bash
python analysis/review_closure_probe_2026-07-14.py
```

关键实测结果：

- P0-01：optimizer 事件 `grad_accum=1276.743 MiB`，与逐层 FSDP shard 梯度求和**逐字节相等**。
- P0-02：`reshard_after_forward_policy: nevver`、`local_batch_szie`、`max_device_memry`、`swap.enablee` 均未报错，分别回落到 typo/default 语义；非零 `attention_dropout=0.1` 生成的 `LLMConfig` 与 0.0 完全相同。
- P1-02：直接核心 API 的 `full_layers=set()` 与无重算完全相同且不报错；零命中 select 也被接受并产生不同时间线。
- P1-06：仅切换 DSA fusion，`cross_entropy_fused` 跟着 `True→False`，证明耦合仍在。
- P1-09：TP=1/2 的 `fa_stats` 为 2/1 MiB（正确减半），workspace 却都是 2 MiB（未按 TP 切）。
- P1-10/11：`o_groups=0` 到 shape resolve 才抛 `ZeroDivisionError`；`topk=-1` 产生 `local_numel=-8388608`；capacity=0 产生零尺寸张量。
- P1-16/19：核心 API 接受 `TP=2 + sequence_parallel=False`；也接受 `OptimizerSpec(type="SGD")`，但仍生成 AdamW 的固定 `K_OPT` optstep。
- P2-01：构造出 `.oom == False`、但 `reserved_estimate_bytes > max_device_memory` 的报告。
- P2-06：VPP timeline 中 `(fwd_end, mb=0..3)` 各重复两次，`TimelineSample` 没有 chunk 字段。

### 2.3 真机复验

116 上先执行 `git pull --ff-only`，确认 MindFormers **Already up to date**，HEAD=`97466872d`。本次重新执行而非沿用答复文档旧日志：

**Job A：DSv4 hybrid 4L、FSDP-2、卡 6/7、1 step**

```text
before_optimizer: grad_bytes=3962450432 = 3778.9 MiB
before_optimizer: current_alloc=7566.1 MiB
after_zero_grad:  grad_bytes=0, current_alloc=5676.6 MiB
差值: 1889.5 MiB = 3778.9 / 2
peak_alloc=15415.5 MiB, peak_reserved=16074.0 MiB
```

这对 P0-01 是决定性闭环：梯度是 step-scoped cumulative，FSDP-2 下 optimizer 前驻留本地一半，zero_grad 后释放；当前 `grad_accum` 状态转移位于 `cost_eval/mem_timeline.py:666-698`。

**Job B：DSv4 hybrid 4L、TP=2、只初始化 DTensor、卡 6/7**

```text
embedding.word_embeddings.weight local_shape=(64640,1792), local_numel=115834880
output_layer.weight               local_shape=(64640,1792), local_numel=115834880
placements=(StridedShard(dim=0, split_factor=2), Shard(dim=0))
```

两 rank 一致，P0-04 的 vocab 权重切分与评估器 `231669760/2` 完全一致。TP=2 完整训练仍受 fork 的 DSA+TP kernel 问题限制，因此本证据证明分片，不声称验证了 TP=2 全局峰。

## 3. 32 项逐条裁定

### 3.1 P0

| 项 | 审计状态 | 证据与边界 |
|---|---|---|
| P0-01 累计梯度 | **闭环** | `grad_accum` 首次反向逐层增加、optimizer 后清零（`cost_eval/mem_timeline.py:666-698`）；本地逐字节公式 + 本次 NPU 1889.5 MiB 双证据。 |
| P0-02 adapter 静默丢语义 | **部分闭环** | parallel/recompute 已有 schema（`cost_eval/configs/from_mindformers.py:545-562,650-704`），但 training/context/swap 未全键校验；swap 只检查 `enable` 真值（`:758-764`）。四个 typo 和非零 dropout 仍静默回落。 |
| P0-03 reshard 死配置 | **闭环（有新校验缺口）** | always/never/default 已进入 resident 生命周期（`cost_eval/mem_timeline.py:446-466`），签名确实不同；但 `ParallelConfig.__post_init__` 只校验 CP method（`cost_eval/specs.py:38-43`），`nevver` 被当 default。原“死配置”已修，新问题是枚举未校验。 |
| P0-04 TP vocab 栈 | **闭环** | logits/vocab 权重切分在 `cost_eval/layers/head.py:47-60,95-106`；本地 logits 减半且 vocab-CE scratch 公式正确；NPU 两 rank 的 embedding/output 均为 115834880。 |
| P0-05 DSA KL | **闭环（预估计口径）** | `idx_kl_loss.workspace_ref=kl_scores`（`cost_eval/layers/dsa.py:133-138,184-188`）；本地 TP=2/CP=2 精确得到主 attention heads 口径的 128 MiB。尚无独立 DSA pynative profiler，故不能升级为真机标定口径。 |

### 3.2 P1

| 项 | 审计状态 | 证据与边界 |
|---|---|---|
| P1-01 router/norm/final/shared gate 参数 | **部分闭环** | router、各 norm、final norm 已补；NPU optimizer 表也确认。可是当前 op 图没有 `shared_experts_gate`；运行时在 `use_shared_expert_gating=True` 时确有 `[H,1]` Dense（MindFormers `mindformers/pynative/transformers/moe/shared_experts.py:56-64`），adapter 只能拒绝该键。更重要的是答复声称“参数守恒改逐模块对账”，实际测试仍是全局 ±2%（`tests/test_param_conservation.py:1-4,42,61`）。 |
| P1-02 exclude/零命中/full 空集 | **部分闭环** | adapter 对 exclude、full 缺列表、未知 cell 已 fail-loud，UI 也查零命中（`serve_explorer.py:510-516`）；但直接 `RecomputeSpec` 仍接受 full 空集/零命中。安全入口改善，不是核心语义完整实现。 |
| P1-03 同名异 shape | **闭环** | mHC 重命名 + resolve 不变量，正向反例已翻绿。 |
| P1-04 embedding dtype | **闭环** | TensorRef 已消费 dtype，adapter 按真机 compute/gather 口径映射，相关回归通过。 |
| P1-05 独立 DSA 入口 | **闭环** | adapter `experimental_attention_variant=dsa` 可构造并完成 `build_llm_spec`；探针返回 `attn_type=dsa`。 |
| P1-06 loss 与 DSA fusion 耦合 | **部分闭环，且回复与代码直接矛盾** | TP vocab-CE 和 chunked loss 已接入；但答复 `:42-44` 写“已解耦、pynative 恒 unfused”，代码仍是 `cross_entropy_fused=_dsa_fused(model)`（`cost_eval/configs/from_mindformers.py:429-436`）。探针只翻 DSA fusion 就翻 CE fusion。 |
| P1-07 checkpoint islands | **仍开放** | 回复已明确暂缓；选择集仍不是一等 checkpoint regions。 |
| P1-08 bwd scratch max-live | **仍开放（接受的保守上界）** | 当前仍 `sum(op.bwd_scratch_bytes)`（`cost_eval/structure_mem.py:188`）。方向 OOM 安全不等于准确性闭环。 |
| P1-09 FA stats 生命周期/TP 缩放 | **部分闭环** | stats 已作为 saves 且按 TP/CP 切（`cost_eval/layers/attention.py:20-36`）；但同文件明确承认 workspace 字符串不按 TP 切（`:26-29`），探针 TP=2 workspace 不变。原问题含两部分，只修了一部分。 |
| P1-10 DSv4 ratio/o_groups 校验 | **部分闭环** | ratio 集合、长度、整除已校验；`o_groups=0` 未提前拒绝（`cost_eval/build_llm.py:40-55,65-70`），最终裸 `ZeroDivisionError`。 |
| P1-11 通用结构校验 | **部分闭环** | head floor、GQA groups、freq 长度、topk 上界已补；topk≤0、capacity≤0 等仍无 guard，实测可产生负/零 numel（相关条件只见 `cost_eval/build_llm.py:56-69`）。 |
| P1-12 MoE 平均负载 | **仍开放** | 仍是 balanced/floor 口径，无 skew/capacity ceil/padding/最忙 rank；回复也只做文档化。 |
| P1-13 CP buffer | **仍开放** | fused-QKV 的 KV 分量、ring/ulysses/colossal 双缓冲与 kernel workspace 仍是 caveat。 |
| P1-14 FSDP wrap 粒度 | **仍开放** | 回复说 experts 独立组已建，但代码明确写 experts/router 的独立 wrap 时间线“并入层组”（`cost_eval/mem_timeline.py:453-454`），实际仍用 `sm.param_full_bytes` 整层 resident。 |
| P1-15 PP/VPP | **仍开放** | recv 主体子结论收窄；send/overlap/更多 schedule/显式 VPP chunk ranges 均未实现，回复也列入残留。 |
| P1-16 feasibility | **部分闭环** | `Evaluator` 拒绝 PP+swap，adapter 拒绝 TP>1 且 SP=false；但直接核心 API 仍接受后者，且没有完整 runtime matrix。两个例子修复不能等同“validator 已闭环”。 |
| P1-17 UI round-trip | **部分闭环** | warning 和措辞已收窄，但功能仍丢字段：`_bundle_to_fields` 不返回 dp_replicate/reshard/offload/prefetch（`serve_explorer.py:640-659`），真正评估仍固定 AdamW/64GiB/swap-off（`:518-523`）。`README_explorer.md:35` 的“评估仍按解析值”与代码不符。 |
| P1-18 nested offset | **闭环** | 延迟物化 + 4 条回归覆盖缺 N、兜底后物化、仍缺时 warning。 |
| P1-19 optimizer/offload | **部分闭环** | `grad_accum` 拆出后 K_OPT 6→4，adapter 会拒绝非 Adam；但核心 API 接受 SGD 并仍生成固定 Adam optstep（`cost_eval/mem_timeline.py:676-696`），param/grad/optimizer 分离 offload 仍不可表达。 |

### 3.3 P2

| 项 | 审计状态 | 证据与边界 |
|---|---|---|
| P2-01 `.oom` 口径 | **部分闭环** | 文档和 `reserved_estimate_bytes()` 已有（`cost_eval/report.py:20-35`），但仍只有 allocated `.oom` 单布尔，没有 `reserved_oom`/三态结果。探针证明 `.oom=False` 时 reserved 可已超阈值。 |
| P2-02 opdag 主链 | **仍开放** | 回复明确维持交叉验证工具定位，Evaluator 不消费。 |
| P2-03 结构覆盖 | **部分闭环** | final norm 已补；但 adapter 把 `qk_layernorm`、attention/hidden dropout 放入忽略集（`cost_eval/configs/from_mindformers.py:71-78`）。实测 qk=true 变 false，dropout=0.1 与 0.0 完全同图。 |
| P2-04 标定常数泛化 | **仍开放** | K_CE、K_OPT、kept_frag 仍是少量 regime 标定；两个锚点仍低于 0.95，回复也承认未解决。 |
| P2-05 全尺寸预设可信度 | **仍开放** | 仍是外推/循环 ratios，只有 UI “预估计”标签。 |
| P2-06 timeline 标识 | **部分闭环，回复表述不实** | `TimelineSample` 只新增 `mb`（`cost_eval/mem_timeline.py:228-234`），事件字符串仍 `fwd:<lid>/fwd_end/bwd@<lid>`（`:561,580,654`），没有 chunk 字段；VPP 同 `(event,mb)` 重复。答复所称“事件名加 microbatch/chunk 序号”未发生。 |
| P2-07 测试固化风险 | **部分闭环** | 反例测试有价值，但 reshard 只断言“不相等”、grad 只断言 `>0`；参数守恒仍是 ±2%，不是答复声称的逐模块精确对账。新增探针正好穿过现有 475 个测试。 |
| P2-08 文档漂移 | **部分闭环，仍有明确反例** | 部分核心注释已改；但 `tests/test_review_evidence.py:1-4` 仍声称 strict XFAIL，实际已无 xfail；`tests/test_ce_optstep.py:1-5` 仍写 7×，实现是 K_OPT=4+grad；README 的 UI 口径也与代码冲突。 |

## 4. 回复中的“假闭环”与原审查“假阳性”要分开

原审查没有整项假阳性；真正存在的是回复中的状态过度归并：

- **代码直接反证回复**：P1-06（仍耦合）、P1-14（仍并入整层）、P2-06（无 chunk）、P2-07（非逐模块精确）、P2-08（文档仍漂移）。
- **只完成安全封口却写成能力已修**：P0-02、P1-02、P1-16、P1-17、P2-01。fail-loud/warning 能避免静默错算，值得保留，但不等于功能语义已实现。
- **只修原问题的一半**：P1-01、P1-09、P1-10、P1-11、P1-19、P2-03。

因此，`review_response_2026-07-14.md:157-160` 的“无一条未答复”是对的；若把它理解成“32 项均闭环”则是错的。

## 5. 建议的关闭门槛

优先把下列反例转成正式回归，再更新答复状态：

1. adapter 所有输入段做 schema 校验，至少拒绝 typo reshard、training/context/swap 未知键；非零 dropout/qk norm 要么建模，要么 fail-loud。
2. 删除 DSA→CE 的经验耦合；CE 形态必须由独立、可追溯的 loss 配置决定。
3. FA workspace 使用 TensorRef 或显式 TP 分母，不只修 saved stats。
4. `ParallelConfig`/`RecomputeSpec`/`OptimizerSpec` 核心入口统一校验，不能只在 UI/adapter 封口。
5. timeline 增加 `(mb, chunk, layer)` 结构化身份；不要靠不可唯一的字符串。
6. 参数守恒改成按运行时模块清单逐项 exact 对账；把 ±2% 只保留为跨版本观察指标。
7. `.oom` 输出至少拆为 `allocated_oom` 与 `reserved_capacity_exceeded`，并保留不确定余量。

