# 内存评估器审查证据（2026-07-14）

> **归档说明：** 本文只是真机与本地反例附件；完整功能问题、代码缺陷、修正结论和修复顺序见 `analysis/final_code_review_2026-07-14.md`。

## 结论分级

| 审查项 | 状态 | 证据 |
|---|---|---|
| MindFormers 并行字段导入丢失 | 本地确认 | 输入 `ulysses / interleave=4 / reshard=never`，实际得到 `colossal / 1 / default` |
| `reshard_after_forward` 不影响时间线 | 本地确认 | `always` 与 `never` 的逐事件 total/gather signature 完全相同 |
| MoE router 参数未进入 LayerSpec | 本地确认，真机旁证 | 本地 router `params=[]`；NPU optimizer 参数列表包含 `decoder.layers.N.mlp.router.weight` |
| embedding/lm_head 未按 TP 切分 | 本地确认，NPU 未测 | TP=1 和 TP=2 的两项 local numel 都是 `231,669,760` |
| mHC/MTP 同名异 shape | 本地确认，NPU 未测 | MTP 层 `x` 同时解析为 `7,340,032` 和 `29,360,128` 元素 |
| 累计梯度在 optimizer 事件缺失 | NPU + 本地确认 | NPU optimizer 前有全部 FP32 grad；评估器 `optstep.grad_buf == 0` |
| DSA 独立 attention 路径的 KL loss 漏项 | 尚未真机验证 | 本次运行的是 `dsv4_hybrid` fused DSA，不是新增的独立 `attn_type=dsa` 路径 |
| `oom` 是否必须使用 reserved | 仅确认口径分离 | 真机确认 reserved > allocated；本次没有构造临界容量 OOM，不能判定现有布尔语义错误 |

## 环境

- 服务器：`192.168.9.116`，容器 `shb.ms.2.9`
- NPU：2 × Ascend 910B3，物理卡 `6,7`
- MindSpore：2.10
- MindFormers checkout：`/home/suhaibo/workspace/deepseek_v4/mindformers`
- MindFormers commit：`97466872d`
- 更新检查：`git pull --ff-only` 返回 `Already up to date.`
- 评估器：本地 `9f2372c` 加当前工作区中尚未提交的 DSA 相关改动
- 真机配置：DSv4 hybrid 4L，seq=2048，FSDP-2，TP/CP/EP/PP=1，fused DSA，无重计算，1 step

运行前卡 6/7 均为 AICore 0%，HBM 约 6%；未清理机器上其他用户的低占用常驻进程。

## 真机梯度生命周期

探针通过继承 PyNative Trainer，仅在原始 `_optimizer_update()` 调用前读取 Parameter；随后继续调用原实现，不修改梯度或优化器数学。

关键输出（rank 0/1 相同）：

```text
[MAIN_GRAD] rank=0 phase=before_optimizer buffers=74 main_grad_buffers=0
param_grad_buffers=74 grad_numel=990612608 grad_bytes=3962450432
grad_MiB=3778.9 current_alloc_MiB=7566.1

[MAIN_GRAD] rank=1 phase=before_optimizer buffers=74 main_grad_buffers=0
param_grad_buffers=74 grad_numel=990612608 grad_bytes=3962450432
grad_MiB=3778.9 current_alloc_MiB=7566.1

[MAIN_GRAD] phase=after_zero_grad buffers=0 grad_bytes=0
current_alloc_MiB=5676.6
```

该配置的参数为 FP32，因此梯度保存在 `param.grad`；bf16 参数路径才使用 `param.main_grad`。逻辑梯度总量是 3778.9 MiB，FSDP-2 的物理分片应占：

```text
3778.9 / 2 = 1889.45 MiB
```

真机 allocated 差值为：

```text
7566.1 - 5676.6 = 1889.5 MiB
```

二者吻合。这证明所有已规约梯度在 optimizer 前同时驻留，optimizer 后由 `zero_grad()` 释放。正确结论不是“梯度跨 step 永久驻留”，而是“梯度在本 step 的反向后半段至 optimizer 期间累计驻留”。

评估器反例：

```text
event='optstep'
grad_buf=0
optstep=2780037120
```

因此当前时间线确实没有表达累计梯度与 optimizer scratch 的共存。

## 真机峰值对比

```text
NPU max allocated : 15415.5 MiB
NPU max reserved  : 16074.0--16096.0 MiB
评估器预测         : 14915.5 MiB @ bwd@5
pred / real       : 0.9676
```

当前配置的全局峰值仍由 fused loss 区域主导，所以累计梯度缺失没有把总体误差扩大到 1889.5 MiB；它主要影响 loss 被 TP/vocab-parallel CE 压低、optimizer 或 transformer backward 成为峰值的配置。

allocated 与 reserved 的差为约 658.5--680.5 MiB。该结果证明两种口径必须分开报告，但没有在设备容量临界点触发一次 OOM，因此不能仅凭本次实验断言 `PeakMemoryReport.oom` 必须改成 reserved 判定。

## 本地可执行反例

测试文件：`tests/test_review_evidence.py`

将 expected-failure 当作普通测试运行：

```powershell
python -m pytest --runxfail -q tests/test_review_evidence.py
```

结果：

```text
6 failed in 0.35s
```

失败值分别为：

1. `('colossal', 1, 'default') != ('ulysses', 4, 'never')`
2. `always_signature == never_signature`
3. router 参数数量 `[0]`
4. TP=2 的 embedding/head 仍为 `231669760`，预期 `115834880`
5. `{(5, 'mtp'): {'x': [7340032, 29360128]}}`
6. `optstep.breakdown.grad_buf == 0`

常规模式：

```powershell
python -m pytest -q -rxX tests/test_review_evidence.py
```

结果：`6 xfailed`，退出码 0。所有标记均为 `strict=True`；对应实现修复后会变成 XPASS 并让测试失败，提醒移除 xfail。

## 尚需的真机矩阵

本次已使用三个单步作业完成探针定位，没有继续占用共享 NPU。以下结论仍需独立差分实验：

1. dense-only TP=1/2：验证 embedding、output layer、vocab-parallel loss 的每卡缩放。
2. FSDP-2 `reshard=always/default/never`：记录 forward-end、optimizer 前 current allocated 和全局峰值。
3. 独立 `attn_type=dsa`：确认 indexer KL loss 的真实中间量；不能用 `dsv4_hybrid` 的 fused 结果替代。
4. mHC+MTP：在修正张量身份前后比较 MTP layer 的 forward/backward 工作集。

## 复现资产与清理

- `analysis/realmachine/run_main_grad_probe.py`
- `analysis/realmachine/run_main_grad_probe.sh`
- `tests/test_review_evidence.py`

容器和服务器 `/tmp` 下的探针、三个专用日志目录已经删除。作业结束后卡 6 HBM 回到 6%、AICore 0%。
