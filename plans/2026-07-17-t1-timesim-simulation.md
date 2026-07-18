# T1a 仿真核心：op_cost + segment_sim + pipeline_sim 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 spec `2026-07-16-step-time-cost-model-design.md` 的 T1 仿真核心：producer per-tensor 分片状态（T0 交接要点5，attention 族段打通）、op_cost（默认 η，M8）、segment_sim（段内多流 DES + 三态归因，M9-L1）、pipeline_sim（1F1B/VPP 全局 DES + 步收尾，M9-L2）、StepTimeReport 门面，L0 不变量 + L1 轻量互证全绿。

**Architecture:** 分层离散事件仿真（spec D4）：TimedSegment（T0 产物，本计划先以 per-tensor 分片状态扩展到 MLA）→ CostModel 逐 op 定价（roofline×默认η，provenance="theory"）→ segment_sim 对一个 (stage,phase,chunk) 完整 pass 做 H/D/C_* 多流推进并按"先查发射、再查依赖"三态归因 → pipeline_sim 消费 `schedule.py` 调度序做全局 DES → report 聚合 T_step/MFU/HFU/瓶颈。时间侧与内存侧解耦契约（§2.2 五条）全程有效，import-lint 回归门在 T0 已就位。

**Tech Stack:** Python 3.13 + pytest + stdlib（timesim 绝不 import mindspore/mindformers/内存仿真模块）。mindformers 源根 `E:\97-codes\torch_parallel\mindformers\mindformers`（`MINDFORMERS_ROOT` 可覆盖，无源则 skip——沿用 conftest 既有模式）。

**范围裁决（对照 spec §9-T1）：** spec T1 还含「explorer 时间面板上线」——**拆到 T1b 单独计划**（面板消费本计划产出的 StepTimeReport JSON，是独立可交付的 UI 工程；一计划一份可独立测试的软件）。本计划吸收 T0 交接要点 5（per-tensor 分片状态=T1 第一批工作）、要点 3（FSDP bwd 重 gather 决策=注入，随 `reshard_after_forward`，spec §3.3c 已定）、要点 6 前半（`_axis_value` 公名化）。

**执行约定：**
- 仓库根：`E:\97-codes\torch_parallel\pynative-cost-evaluator`，分支 `feat/unified-llm-modelspec`（沿用本仓惯例，不建 worktree）。
- 每个任务末跑**全量** `python -m pytest tests/ -q`——12 内存锚点 + T0 timesim 测试是硬回归门。
- 所有源事实引用（`file:line`）以执行时实际源码为准；行号漂移按惯用法重定位并更新注释。
- 测试里的硬件常数一律用 SYNTH 合成硬件（整数好算、断言零公差）；`DEFAULT_910B` 只是占位默认（厂商规格代入，`calibrated=False`，spec §7.2-1），不进任何断言。

---

## 文件结构总览

```
cost_eval/opdag/consumer.py          # 改：_axis_value → 公名 axis_value（T0 交接6）
cost_eval/timesim/producer.py        # 重写：per-tensor 分片状态（交接5 端态设计）+ SPL 支持
cost_eval/timesim/shard_rules.py     # 改：weight_local 加 SequenceParallelLinear 分支 + 公名导入
cost_eval/timesim/frame_comm.py      # 改：inject_cp ring 结构化（cp 块 FA + cp-1 跳 p2p）；
                                     #     新增 fsdp_regather（bwd 权重重 gather，交接3 决策）
cost_eval/timesim/ir.py              # 改：CommSpec docstring p2p 跳数口径更新（结构化后恒 1 跳）
cost_eval/timesim/machine.py         # 新：TimeHardware（时间侧硬件常数，契约4 独立命名空间）
cost_eval/timesim/op_cost.py         # 新：OpCost + CostModel（M8，spec §4.1）
cost_eval/timesim/pass_builder.py    # 新：多层段拼接成 pass（spec §5.2 连续 pass 语义）
cost_eval/timesim/segment_sim.py     # 新：段内多流 DES + 三态归因（M9-L1，spec §5）
cost_eval/timesim/pipeline_sim.py    # 新：全局调度 DES（M9-L2，spec §6.1/6.2）
cost_eval/timesim/report.py          # 新：StepTimeReport + evaluate_step_time 门面（§6.3/6.4）
tests/conftest.py                    # 改：新增 mla_dag 夹具
tests/test_timesim_producer.py       # 改：两个 T0 守卫测试按 per-tensor 语义更新
tests/test_timesim_mla_segment.py    # 新
tests/test_timesim_frame_comm.py     # 改：cp ring 结构化断言更新
tests/test_timesim_op_cost.py        # 新
tests/test_timesim_pass_builder.py   # 新
tests/test_timesim_segment_sim.py    # 新
tests/test_timesim_pipeline_sim.py   # 新
tests/test_timesim_report.py         # 新
tests/test_timesim_l0.py             # 新（L0 阶梯 + L1 轻量互证）
README.md                            # 改：状态区一行
```

## 已核实源事实（2026-07-17 现场探针，实施时如漂移按惯用法重定位）

1. **SequenceParallelLinear**（mindformers `tensor_parallel/layers.py:819`）：继承 ColumnParallelLinear，docstring「A is not parallelized. X is parallelized with data_parallel and sequence_parallel」；shard 布局（:845-866）输入/输出均 `layout(("cp","tp"),"dp","None")`（S 轴 (cp·tp) 分片贯通）、权重 `("None","None")` 不切。→ producer 语义：权重全量、无通信、SP 驻留透传。
2. **MLA 段 DAG**（`extract_cell` + `infer_shapes(dag, {"x": "S·B·H"})`，26 节点）已基本落实符号 shape；残留 `?` 仅四类且全部被 producer 既有契约容忍：host-only `shape` 读（节点1 out）、`rotary_pos_emb`/`attention_mask` 外部输入（ins 容忍）、尾部 `ori_dtype`。**注意 conftest 的 mlp 种子名是 `hidden_states`，MLA 的是 `x`**。
3. **split 多输出无需改 walker**：`construct_walker._emit`（:898-908）把 tuple 全部目标登记进 SSA/producer（边完整）；`split_targets` attrs + shape_infer 已把第二输出（`value__i1`/`k_pos_emb__i0`/`q_pos_emb__i1`）的消费端 ref 解析出正确符号 shape（探针实证）。producer 只需用 `split_targets` 建 name→node 映射。
4. **MLA 的 SP 布局重分布是隐式的**：`shard_self_attn`（multi_latent_attention.py:713-727）声明 `split_3d` 出 S=(cp,tp) 分片、`expand_dims`/`tile_kv` 入 S=cp 分片、`pe_concat` 入 heads=tp 分片——rope 支（不过 Column）与 k_no_pe（过 Column）在 pe_concat 汇合前，框架按布局差自动重分布（S 轴 tp-gather）。construct 源码内无显式通信（grep AllGather/ReduceScatter 零命中）。→ producer 在**多输入汇合点**对 S 分歧注入 `injected:layout-redistribution` AG。
5. **carrier 事实**：MLP fc1 `out_dim="2·ffn_hidden"`（interleaved 布局切 ffn_hidden，T0 已核实）；MLA q_up `out_dim="n_heads·(qk_head_dim+qk_pos_emb_head_dim)"`、kv_up `="n_heads·(qk_head_dim+v_head_dim)"`、proj `in_dim="n_heads·v_head_dim"`（Megatron 语义按 head 切）。两者统一为「out_dim 顶层乘积的首个**符号**因子」。
6. **消费端 API**：`ParallelConfig`（specs.py）有 `reshard_after_forward: str = "default"  # always|never|default`、`dp_shard/dp_replicate/num_microbatches/interleave`；`OptimizerSpec.state_bytes_per_param=14`、`grad_dtype_bytes=4`；`schedule.py` 提供 `build_1f1b(stage,pp,m)`（Event.kind/mb）与 `interleaved_virtual_order(stage,pp,m,v,group_size)`（`[(kind,mb,chunk)]`，group_size 默认=pp）；`consumer._SYM2FIELD` 已含 MLA 全部 token（`qk_head_dim→qk_nope_head_dim` 等）。
7. **specs.HardwareSpec 是内存侧的**（max_device_memory/framework_reserve），无算力/带宽字段——时间侧常数按契约4（标定命名空间分离）独立放 `timesim/machine.py`，不动 specs.py。

---

# Phase A — producer 强化（交接债清偿 + attention 段打通）

## Task 1: 测试基线 + `_axis_value` 公名化（交接要点6）

**Files:**
- Modify: `cost_eval/opdag/consumer.py`（`_axis_value` 定义后追加公名）
- Modify: `cost_eval/timesim/shard_rules.py:15`（import 改公名）
- Test: `tests/test_opdag_consumer.py`（追加）

- [x] **Step 1: 跑全量测试，记录基线 N_baseline**

Run: `python -m pytest tests/ -q`
Expected: 全绿。记下通过数 N_baseline（后续每任务对照，不允许减少）。

- [x] **Step 2: 写失败测试**

```python
# tests/test_opdag_consumer.py 追加
def test_axis_value_public_name():
    """T0 交接要点6：跨包消费（timesim.shard_rules）用公名，私名保留兼容。"""
    from cost_eval.opdag import consumer
    assert consumer.axis_value is consumer._axis_value
```

Run: `python -m pytest tests/test_opdag_consumer.py -q` → FAIL（AttributeError: axis_value）。

- [x] **Step 3: 实现**

`consumer.py` 的 `_axis_value` 函数定义之后追加：

```python
# 公共名（T0→T1 交接要点6）：timesim.shard_rules 等跨包消费方用此名；
# 私名 _axis_value 保留（本模块内部/存量引用兼容）。
axis_value = _axis_value
```

`shard_rules.py` 的 import 行改为：

```python
from ..opdag.consumer import axis_value as _axis_value
```

（shard_rules 内部引用名不变，改动最小。）

- [x] **Step 4: 跑测试 + 全量回归** → 通过数 = N_baseline + 1。

- [x] **Step 5: Commit**

```bash
git add cost_eval/opdag/consumer.py cost_eval/timesim/shard_rules.py tests/test_opdag_consumer.py
git commit -m "refactor(opdag): _axis_value 公名化 axis_value(T1-1,交接要点6)"
```

## Task 2: producer per-tensor 分片状态（交接要点5 端态设计）

**Files:**
- Rewrite: `cost_eval/timesim/producer.py`
- Modify: `tests/test_timesim_producer.py:142-180`（两个 T0 守卫测试按新语义更新）
- Test: `tests/test_timesim_producer.py`（追加 per-tensor 性质测试）

**动机（交接要点5）**：T0 的 `sp_active`/`feat_sharded` 是两个全局 bool——MLP 单链恰好成立；MLA 多支路（rope 支不过 Column，与过 Column 的 k_no_pe 在 pe_concat 汇合）立即失效。端态设计=状态挂**生产者节点**：`state[node_id] = {carrier_sym: divisor}`。

- [x] **Step 1: 写失败测试（先写 per-tensor 性质，用纯合成 DAG，不依赖 mindformers）**

```python
# tests/test_timesim_producer.py 追加（文件头 import 区补 from types import SimpleNamespace）

def _synth_dag(nodes, edges, opaque=()):
    return SimpleNamespace(cell="synth", nodes=nodes, edges=edges,
                           opaque_calls=list(opaque))


def _node(id, op, src, ins, out, module="", attrs=None):
    return SimpleNamespace(id=id, op=op, src=src, ins=ins, out=out,
                           module=module, attrs=attrs or {})


# 合成测试用 dims：qk_head_dim token 映射 DimTable.qk_nope_head_dim（consumer._SYM2FIELD），
# 既有模块级 DIMS 未设该字段（=0 → 未解析 fail-loud），此处单独给值。
DIMS_ATTN = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                     S=4096, B=1, vocab=129280, n_layers=4,
                     qk_nope_head_dim=224, v_head_dim=224)


def test_per_tensor_state_branch_isolation():
    """per-tensor 核心性质：不过 Column 的旁支不携 feature 分片（T0 全局位在此必错）。
    合成结构 = MLA pe_concat 惯用法缩影：主支过 Column（heads 分片），旁支直连 View。"""
    nodes = [
        _node(1, "MatMul", "f.py:1", ["x:S·B·H:bf16"],
              "u:S·B·(n_heads·qk_head_dim):bf16", module="ColumnParallelLinear",
              attrs={"in_dim": "H", "out_dim": "n_heads·qk_head_dim"}),
        _node(2, "View", "f.py:2", ["u:S·B·(n_heads·qk_head_dim):bf16"],
              "u4:S·B·n_heads·qk_head_dim:bf16", attrs={"view": "reshape"}),
        _node(3, "View", "f.py:3", ["x:S·B·H:bf16"],
              "side:S·B·H:bf16", attrs={"view": "reshape"}),
    ]
    seg = build_segment("s.fwd", _synth_dag(nodes, [[1, 2]]), DIMS_ATTN, Degrees(tp=2))
    by_src = {o.src: o for o in seg.ops if o.op_type != "CommOp"}
    # 主支：flat 轴÷tp → reshape 后 n_heads 轴÷tp（carrier=n_heads，恰一轴）
    assert by_src["f.py:1"].out_shape == (4096, 1, 8 * 224 // 2)
    assert by_src["f.py:2"].out_shape == (4096, 1, 4, 224)
    # 旁支：从未过 Column → 不分片（T0 全局 feat_sharded 位在此会误除或 fail-loud）
    assert by_src["f.py:3"].out_shape == (4096, 1, 1792)


def test_per_tensor_sp_reconciliation_ag():
    """S 分歧汇合（源事实4：MLA pe_concat 惯用法）：SP 驻留支与已聚合支在多输入 op 汇合 →
    驻留支注入 layout-redistribution AG（volume=分片字节，AG 口径）。"""
    nodes = [
        _node(1, "MatMul", "f.py:1", ["x:S·B·H:bf16"],
              "u:S·B·(n_heads·qk_head_dim):bf16", module="ColumnParallelLinear",
              attrs={"in_dim": "H", "out_dim": "n_heads·qk_head_dim"}),
        _node(2, "View", "f.py:2", ["x:S·B·H:bf16"],
              "side:S·B·H:bf16", attrs={"view": "reshape"}),
        _node(3, "Elementwise", "f.py:3",
              ["u:S·B·(n_heads·qk_head_dim):bf16", "side:S·B·H:bf16"],
              "z:S·B·H:bf16"),
    ]
    deg = Degrees(tp=2, sequence_parallel=True)
    seg = build_segment("s.fwd", _synth_dag(nodes, [[1, 3], [2, 3]]), DIMS_ATTN, deg)
    comms = [o for o in seg.ops if o.op_type == "CommOp"]
    # Column 前模块语义 AG（.ag）+ side 支在节点3 汇合前的重分布 AG（.ag1）
    assert [c.module for c in comms] == ["injected:module-semantics",
                                         "injected:layout-redistribution"]
    redis = comms[1]
    assert redis.comm.ctype == "all_gather"
    assert redis.in_shapes == ((2048, 1, 1792),)          # S/(tp) 驻留分片
    assert redis.out_shape == (4096, 1, 1792)
    assert redis.comm.volume_bytes == 2048 * 1792 * 2      # AG=分片入参字节
    # 汇合节点吃聚合后的 side → 两输入 S 一致
    z = next(o for o in seg.ops if o.src == "f.py:3")
    assert z.in_shapes[1] == (4096, 1, 1792)
    assert redis.op_id in z.deps


def test_carrier_duplicate_axes_fail_loud():
    """同一 carrier 命中 ≥2 轴 = 真歧义 → fail-loud（per-tensor 后守卫收窄到此形态）。"""
    nodes = [
        _node(1, "MatMul", "f.py:1", ["x:S·B·H:bf16"],
              "u:S·B·(n_heads·qk_head_dim):bf16", module="ColumnParallelLinear",
              attrs={"in_dim": "H", "out_dim": "n_heads·qk_head_dim"}),
        _node(2, "View", "f.py:2", ["u:S·B·(n_heads·qk_head_dim):bf16"],
              "bad:S·B·n_heads·(n_heads·qk_head_dim):bf16", attrs={"view": "reshape"}),
    ]
    with pytest.raises(ValueError, match="命中 2 轴"):
        build_segment("s.fwd", _synth_dag(nodes, [[1, 2]]), DIMS_ATTN, Degrees(tp=2))


def test_row_without_sharded_input_fail_loud():
    """tp>1 的 Row 输入无 feature carrier（上游没有 Column）→ fail-loud（Megatron Row 恒
    消费分片激活；静默不切会双倍计算量）。"""
    nodes = [
        _node(1, "MatMul", "f.py:1", ["x:S·B·H:bf16"],
              "y:S·B·H:bf16", module="RowParallelLinear",
              attrs={"in_dim": "H", "out_dim": "H"}),
    ]
    with pytest.raises(ValueError, match="Row"):
        build_segment("s.fwd", _synth_dag(nodes, []), DIMS_ATTN, Degrees(tp=2))
```

同文件 **改两处 T0 守卫测试**（语义收窄，不是删守卫）：

1. `test_column_inside_feature_shard_zone_fail_loud`（:142）：Column∘Column 仍必须 raise ValueError——若该测试用 `match=` 断言了 T0 报错文案（含"per-tensor 分片跟踪（T1）"字样），把 match 放宽为 `match="Column"`（报错保留，措辞随实现更新）。
2. `test_build_segment_feature_axis_ambiguity_fail_loud`（:160）：T0 的「两 sym 拆两轴=歧义」在 carrier 规则下**已可正确解析**（carrier 只命中一轴）——把该测试改写为**正向**断言：同一合成 DAG 现在成功构段、feature 轴落在 carrier 轴上正确 ÷tp（保留原 docstring 并注明 T1 语义变更）；真歧义守卫由上面新测试 `test_carrier_duplicate_axes_fail_loud` 接棒。
3. `test_build_segment_unknown_module_fail_loud`（:94）：若它拿 `"SequenceParallelLinear"` 当未知模块（T0 注册为"已知未实现"）——SPL 在本计划 Task 2/3 转正支持，把测试里的模块名换成真假名（如 `"TotallyUnknownLinear"`），守卫语义不变。

- [x] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_timesim_producer.py -q`
Expected: 新测试 FAIL（`test_per_tensor_state_branch_isolation` 在 T0 全局位下 side 支被误判 0 轴命中 fail-loud）。

- [x] **Step 3: 重写 producer.py（完整替换文件）**

```python
# cost_eval/timesim/producer.py
"""TimedOpSeq producer（spec §3.3：b 并行代入 + c 通信注入装配）——per-tensor 分片状态版（T1）。

输入契约与 T0 版一致：只收 extract_cell 后经 opdag.shape_infer.infer_shapes 落实符号 shape 的
DAG；Col/Row/SPL MatMul 须携 in_dim/out_dim attrs；含通信样 opaque_calls 的 DAG 须显式
opaque_comm_ok=True。`?` 容忍规则不变：ins 一律容忍为空 tuple；out 仅 device 流 fail-loud。

**per-tensor 分片状态**（T0 交接要点5 端态设计）：T0 的 sp_active/feat_sharded 是两个全局
bool——对 MLP 单链成立，MLA 多支路（rope 支不过 Column、与过 Column 的 k_no_pe 在 pe_concat
汇合）立即失效。本版把状态挂**生产者节点**：node_state[node_id] = {carrier_sym: divisor}：
  - "S" 键 = SP 驻留（S 轴在 cp 之外再 ÷tp）。段入口种子：外部输入（无生产者）sym 含 S 轴且
    deg.sequence_parallel 且 tp>1 时 = {"S": tp}；input_states 参数可按名覆盖。
  - 其余键 = feature carrier（Column 输出起）。carrier = out_dim 顶层乘积**首个符号因子**
    （_carrier_sym）：MLP fc1 "2·ffn_hidden"→"ffn_hidden"（interleaved 布局切 ffn_hidden，
    T0 核实）；MLA q_up "n_heads·(qk_head_dim+qk_pos_emb_head_dim)"→"n_heads"（Megatron
    按 head 切）。切分轴唯一化：reshape 把 heads/head_dim 拆两轴后只有 n_heads 轴命中——
    T0「≥2 轴命中歧义」自然消失；真歧义（同 carrier 现身 ≥2 轴）仍 fail-loud。
状态传播规则（_merge_states / 各模块分支）：
  - 单输入 op：透传；out sym 缺 carrier 轴 → fail-loud（防静默丢跟踪）。
  - 多输出 split：共享一个节点 state（切的是 D 轴，各半分片语义相同）；消费端经
    attrs["split_targets"] 登记的 name→node 映射找到生产者。
  - 多输入汇合：feature carrier 取并集（无 carrier 侧=复制量，真机按目的 op shard
    in_strategy 本地切片、零通信——multi_latent_attention.py shard_self_attn :715-722
    pe_concat/tile_kv 布局实证；in_shapes 因此允许 heads 轴不齐，host_only 无成本）；
    "S" 分歧（部分 SP 驻留、部分已聚合）→ 给驻留侧注入 all_gather
    （module="injected:layout-redistribution"）——对应 semi-auto 布局重分布（真机声明点
    expand_dims :722，本模块注入点=首个多输入汇合 op，位置差几个小 op 的 S 局部度，
    量级 S·B·rope_dim 字节，诚实边界）。
  - ColumnParallelLinear：输入含 feature carrier → fail-loud（Column∘Column 不在支持族）；
    输入含 "S" → 矩乘前注入 all_gather（模块语义注入，comm_probe 实证 Column 源无显式通信）
    并清 "S"；输出 = {carrier(out_dim): tp}（tp>1）。
  - RowParallelLinear：tp>1 时输入必须携恰一个 feature carrier 且命中 in_dim syms、且无 "S"
    残留（Megatron Row 恒消费 feature 分片/全 seq 激活）；矩乘后注入 RS（sp，出 {"S":tp}）
    / AR（非 sp，出 {}）——源惯用法 layers.py:619/:621；下游经 redirect 指向 .rs（F3 Bug B）。
  - SequenceParallelLinear（T1 新支持，layers.py:819「A is not parallelized. X is
    parallelized with data_parallel and sequence_parallel」，shard 布局 :845-866 入/出均
    ("cp","tp") 切 S、权重不切）：权重全量、无通信、状态透传。
deps 只记跨流依赖（同流 FIFO 隐含，ir.py 字段注释）；依赖发现按输入 ref 名→生产者节点
（与 dag.edges 等价——walker 对 tuple 全目标登记 producer，_emit :898-908）。
"""
from __future__ import annotations

from .ir import (TimedOp, TimedSegment, CommSpec, tensor_bytes,
                 STREAM_DEVICE, STREAM_HOST_ONLY, COMM_STREAM)
from .shard_rules import Degrees, axis_values, localize, weight_local
from ..opdag.comm_probe import COMM_CLS
from ..opdag.sym_shape import parse_shape, _split_top

_COL = "ColumnParallelLinear"
_ROW = "RowParallelLinear"
_SPL = "SequenceParallelLinear"
_KNOWN_MATMUL_MODULES = {_COL, _ROW, _SPL, ""}

# 从 comm_probe 的通信原语类名表派生（单一事实源）——新原语进 COMM_CLS 时本守卫自动跟进。
_OPAQUE_COMM_MARKERS = tuple(COMM_CLS)


def _parse_ref(ref: str):
    """opdag 'name:符号shape:dtype' → (name, sym_shape, dtype)。"""
    parts = ref.split(":")
    if len(parts) != 3:
        raise ValueError(f"producer: 非法 TensorRef {ref!r}（fail-loud）")
    return parts[0], parts[1], parts[2]


def _wrap_dim(tok: str) -> str:
    """PART A 的 in_dim/out_dim 扁平 token 拼复合 weight sym 前补括号消歧（T0 注释）。"""
    return f"({tok})" if "·" in tok else tok


def _carrier_sym(out_dim: str) -> str:
    """out_dim 顶层乘积的首个**符号**因子 = TP 切分 carrier（模块 docstring 源事实5）。"""
    for part in _split_top(out_dim, "·"):
        p = part.strip()
        if p.startswith("(") and p.endswith(")"):
            p = p[1:-1].strip()
        if p and not p.isdigit():
            return p
    raise ValueError(f"producer: out_dim {out_dim!r} 无符号因子，carrier 不可定（fail-loud）")


def _has_s_axis(sym: str) -> bool:
    if not sym or sym == "?":
        return False
    return any("S" in f.syms for f in parse_shape(sym))


def _localize_with_state(sym: str, dims, deg: Degrees, state: dict, src: str) -> tuple[int, ...]:
    """符号 shape → local：先 localize 代入全局均匀轴（S÷cp、E÷ep），再按 per-tensor 状态
    对 carrier 命中轴（恰一轴强制）÷divisor。"""
    axes = parse_shape(sym)
    out = localize(axis_values(sym, dims), sym, deg, feat_div_last=1, sp_active=False)
    for carrier, div in state.items():
        hits = [i for i, f in enumerate(axes) if carrier in f.syms]
        if not hits:
            raise ValueError(
                f"producer: {sym!r} @ {src} 不含分片 carrier {carrier!r} 轴——跟踪断裂（fail-loud）")
        if len(hits) > 1:
            raise ValueError(
                f"producer: {sym!r} @ {src} 的 carrier {carrier!r} 命中 {len(hits)} 轴——歧义（fail-loud）")
        i = hits[0]
        if out[i] % div:
            raise ValueError(
                f"producer: {sym!r} 第{i}轴={out[i]} 不被 {div} 整除 @ {src}（fail-loud）")
        out[i] //= div
    return tuple(out)


def build_segment(seg_id: str, dag, dims, deg: Degrees, *, phase: str = "fwd",
                   opaque_comm_ok: bool = False,
                   input_states: dict | None = None) -> TimedSegment:
    if not opaque_comm_ok:
        for call in dag.opaque_calls:
            expr = call.get("expr", "")
            if any(marker in expr for marker in _OPAQUE_COMM_MARKERS):
                raise ValueError(
                    f"producer: dag.opaque_calls 命中疑似通信调用 @{call.get('src')}: "
                    f"{expr!r}——walker fallthrough 不应被静默吞掉（schema.py opaque_calls "
                    f"消费契约）。调用方现场核实语义后传 opaque_comm_ok=True 放行。")
    cell = dag.cell
    id2opid = {n.id: f"{cell}#{n.id}" for n in dag.nodes}

    ops: list[TimedOp] = []
    stream_of: dict[int, str] = {}
    redirect: dict[int, tuple[str, str]] = {}
    node_state: dict[int, dict] = {}
    name2node: dict[str, int] = {}
    seed = {"S": deg.tp} if (deg.sequence_parallel and deg.tp > 1) else {}
    input_states = input_states or {}

    def _producer_ref(p: int) -> tuple[str, str]:
        if p in redirect:
            return redirect[p]
        stream = stream_of.get(p)
        if stream is None:
            raise ValueError(
                f"producer: 节点 id={p} 的生产者尚未发射（前向依赖，walker 序理论不可达）"
                f"（fail-loud）")
        return id2opid[p], stream

    for n in dag.nodes:
        module = n.module or ""
        out_name, out_sym, out_dt = _parse_ref(n.out) if n.out else ("", "", "bf16")
        this_stream = STREAM_HOST_ONLY if n.op == "View" else STREAM_DEVICE

        if n.op == "MatMul" and deg.tp > 1 and module not in _KNOWN_MATMUL_MODULES:
            raise ValueError(
                f"producer: MatMul@{n.src} 的 module {module!r} 不在已知集"
                f"{sorted(_KNOWN_MATMUL_MODULES - {''})}（tp>1 下静默不切分会错切——fail-loud）")

        # —— 输入侧：ref 名 → [名, sym, dtype, 状态, 生产者节点id|None, 重分布AG|None] ——
        # 名字优先匹配（node.out 名 + split_targets）；未匹配名与未匹配入边"恰一对一"时按
        # 排除法配对——walker 的链式视图别名把目标名指到内层节点而**不新增 out 名**
        # （construct_walker.py:461-469：ssa/producer 有记录、DAG 节点无此名），名字查不到
        # 但边在。sym=="?" 的 ref（rotary_pos_emb/attention_mask 类外部量）不参与配对。
        # 配对歧义（候选名>1 或 剩余边>1 且有候选名）→ fail-loud 不猜。
        producers = list(dict.fromkeys(in_edges.get(n.id, ())))
        in_infos: list[list] = []
        for ref in n.ins:
            nm, sym, dt = _parse_ref(ref)
            in_infos.append([nm, sym, dt, None, name2node.get(nm), None])
        matched = {info[4] for info in in_infos if info[4] is not None}
        free_prods = [p for p in producers if p not in matched]
        candidates = [info for info in in_infos
                      if info[4] is None and info[1] and info[1] != "?"]
        if free_prods and candidates:
            if len(free_prods) == 1 and len(candidates) == 1:
                candidates[0][4] = free_prods[0]
            else:
                raise ValueError(
                    f"producer: 节点 {n.src} 有 {len(free_prods)} 条未匹配入边与 "
                    f"{len(candidates)} 个未匹配输入名，无法唯一配对（fail-loud）")
        leftover_prods = [p for p in free_prods
                          if all(info[4] != p for info in in_infos)]
        for info in in_infos:
            if info[4] is not None:
                info[3] = dict(node_state.get(info[4], {}))
            elif info[0] in input_states:
                info[3] = dict(input_states[info[0]])
            else:
                info[3] = dict(seed) if _has_s_axis(info[1]) else {}

        # —— "S" 分歧重分布（多输入汇合，模块 docstring 规则；源事实4）——
        real = [info for info in in_infos if info[1] and info[1] != "?"]
        if len(real) > 1:
            s_flags = [bool(info[3].get("S")) for info in real]
            if any(s_flags) and not all(s_flags):
                for k, info in enumerate(in_infos):
                    nm, sym, dt, st, pnode, _ = info
                    if not (sym and sym != "?" and st.get("S")):
                        continue
                    shard_shape = _localize_with_state(sym, dims, deg, st, n.src)
                    new_st = {c: d for c, d in st.items() if c != "S"}
                    full_shape = _localize_with_state(sym, dims, deg, new_st, n.src)
                    ag_deps = ()
                    if pnode is not None:
                        ag_deps = (_producer_ref(pnode)[0],)
                    ag = TimedOp(
                        op_id=f"{id2opid[n.id]}.ag{k}", op_type="CommOp", phase=phase,
                        in_shapes=(shard_shape,), out_shape=full_shape, dtype=dt,
                        stream=COMM_STREAM["tp"], src=n.src, deps=ag_deps,
                        module="injected:layout-redistribution",
                        comm=CommSpec("all_gather", tensor_bytes(shard_shape, dt),
                                      "tp", deg.tp))
                    ops.append(ag)
                    info[3] = new_st
                    info[5] = ag

        def _base_deps() -> tuple[str, ...]:
            deps, seen = [], set()
            refs = [((ag.op_id, ag.stream) if ag is not None else _producer_ref(pnode))
                    for nm, sym, dt, st, pnode, ag in in_infos if ag is not None or pnode is not None]
            refs += [_producer_ref(p) for p in leftover_prods]   # 无名可配的入边不丢依赖
            for ref in refs:
                if ref[1] != this_stream and ref[0] not in seen:
                    deps.append(ref[0])
                    seen.add(ref[0])
            return tuple(deps)

        # ================= 线性层族 =================
        if n.op == "MatMul" and module in (_COL, _ROW, _SPL):
            nm, x_sym, x_dt, x_st, x_pnode, x_ag = in_infos[0]
            in_dim, out_dim = n.attrs.get("in_dim"), n.attrs.get("out_dim")
            if not in_dim or not out_dim:
                raise ValueError(
                    f"producer: MatMul@{n.src}（{module}）缺 in_dim/out_dim attrs"
                    f"（PART A 未标注线性维度，fail-loud）")
            weight_sym = f"{_wrap_dim(in_dim)}·{_wrap_dim(out_dim)}"
            feat_carriers = {c: d for c, d in x_st.items() if c != "S"}

            if module in (_COL, _SPL) and feat_carriers:
                raise ValueError(
                    f"producer: {module}@{n.src} 输入落在 feature 分片区内"
                    f"（Column∘Column 不在支持族——fail-loud）")

            deps = _base_deps()
            if module == _COL and deg.tp > 1 and x_st.get("S"):
                gather_in = _localize_with_state(x_sym, dims, deg, x_st, n.src)
                x_st = {c: d for c, d in x_st.items() if c != "S"}
                gather_out = _localize_with_state(x_sym, dims, deg, x_st, n.src)
                # AG 挂 comm_tp 流：其 deps 取该输入的生产者（跨流规则从 AG 自己的视角，F3 Bug A）
                ag_deps = ()
                if x_ag is not None:
                    ag_deps = (x_ag.op_id,)
                elif x_pnode is not None:
                    ag_deps = (_producer_ref(x_pnode)[0],)
                gather_op = TimedOp(
                    op_id=id2opid[n.id] + ".ag", op_type="CommOp", phase=phase,
                    in_shapes=(gather_in,), out_shape=gather_out, dtype=x_dt,
                    stream=COMM_STREAM["tp"], src=n.src, deps=ag_deps,
                    module="injected:module-semantics",
                    comm=CommSpec("all_gather", tensor_bytes(gather_in, x_dt), "tp", deg.tp))
                ops.append(gather_op)
                deps = (gather_op.op_id,)

            if module == _ROW and deg.tp > 1:
                if x_st.get("S"):
                    raise ValueError(
                        f"producer: Row@{n.src} 输入残留 SP 驻留（Megatron Row 恒消费全 seq "
                        f"feature 分片激活）——fail-loud")
                in_syms = set()
                for f in parse_shape(_wrap_dim(in_dim)):
                    in_syms |= set(f.syms)
                if len(feat_carriers) != 1 or not (set(feat_carriers) & in_syms):
                    raise ValueError(
                        f"producer: Row@{n.src} 输入 feature 分片状态 {feat_carriers or '{}'} "
                        f"与 in_dim {in_dim!r} 不匹配（无 Column 上游/carrier 不命中——fail-loud）")

            x_local = _localize_with_state(x_sym, dims, deg, x_st, n.src)
            weight_shape = weight_local(weight_sym, dims, module, deg.tp)

            if module == _COL and deg.tp > 1:
                out_st = {_carrier_sym(out_dim): deg.tp}
            elif module == _SPL:
                out_st = dict(x_st)                       # SP 驻留透传（源事实1）
            else:                                          # Row（通信前全量）/ tp==1
                out_st = {}
            out_shape = _localize_with_state(out_sym, dims, deg, out_st, n.src) \
                if out_sym and out_sym != "?" else ()
            if not out_shape:
                raise ValueError(f"producer: device op {n.src} 输出 shape 未解析（fail-loud）")
            mm_op = TimedOp(op_id=id2opid[n.id], op_type="MatMul", phase=phase,
                            in_shapes=(x_local, weight_shape), out_shape=out_shape,
                            dtype=out_dt, stream=STREAM_DEVICE, src=n.src, deps=deps,
                            module=module)
            ops.append(mm_op)
            stream_of[n.id] = STREAM_DEVICE
            node_state[n.id] = out_st
            name2node[out_name] = n.id

            if module == _ROW and deg.tp > 1:
                full = out_shape
                ctype = "reduce_scatter" if deg.sequence_parallel else "all_reduce"
                post_st = {"S": deg.tp} if deg.sequence_parallel else {}
                out_after = _localize_with_state(out_sym, dims, deg, post_st, n.src)
                # src 双落点说明见 T0 注释：带 bias 分支 layers.py:619/:621（无 bias :646/:648）
                rs_op = TimedOp(
                    op_id=id2opid[n.id] + ".rs", op_type="CommOp", phase=phase,
                    in_shapes=(full,), out_shape=out_after, dtype=out_dt,
                    stream=COMM_STREAM["tp"],
                    src="layers.py:619" if ctype == "reduce_scatter" else "layers.py:621",
                    deps=(mm_op.op_id,), module=module,
                    comm=CommSpec(ctype, tensor_bytes(full, out_dt), "tp", deg.tp))
                ops.append(rs_op)
                redirect[n.id] = (rs_op.op_id, COMM_STREAM["tp"])
                node_state[n.id] = post_st
            continue

        # ================= 通用节点 =================
        merged: dict[str, int] = {}
        for nm, sym, dt, st, pnode, ag in in_infos:
            for c, d in st.items():
                if c in merged and merged[c] != d:
                    raise ValueError(
                        f"producer: 汇合节点 {n.src} 的 carrier {c!r} 度数冲突"
                        f"（{merged[c]} vs {d}）——fail-loud")
                merged[c] = d
        in_shapes = tuple(
            _localize_with_state(sym, dims, deg, st, n.src) if sym and sym != "?" else ()
            for nm, sym, dt, st, pnode, ag in in_infos
        )
        if out_sym and out_sym != "?":
            # 汇合出边状态 = 并集中 out sym 实际含有的 carrier（_localize_with_state 对缺轴
            # fail-loud，此处先过滤——多输入并集里允许某 carrier 只属于部分输入（切片口径），
            # 但**单输入透传**缺轴仍要炸：单输入时不过滤，保留 T0 防丢跟踪守卫。
            if len(real) > 1:
                axes_syms: set = set()
                for f in parse_shape(out_sym):
                    axes_syms |= set(f.syms)
                out_st = {c: d for c, d in merged.items() if c in axes_syms}
            else:
                out_st = merged
            out_shape = _localize_with_state(out_sym, dims, deg, out_st, n.src)
        else:
            if this_stream == STREAM_DEVICE:
                raise ValueError(f"producer: device op {n.src} 输出 shape 未解析（fail-loud）")
            out_st = merged
            out_shape = ()
        ops.append(TimedOp(op_id=id2opid[n.id], op_type=n.op, phase=phase,
                           in_shapes=in_shapes, out_shape=out_shape, dtype=out_dt,
                           stream=this_stream, src=n.src, deps=_base_deps(), module=module))
        stream_of[n.id] = this_stream
        node_state[n.id] = out_st
        if out_name:
            name2node[out_name] = n.id
        for t in n.attrs.get("split_targets", ()):           # 源事实3：split 多目标共享本节点
            name2node[t] = n.id

    return TimedSegment(seg_id=seg_id, ops=tuple(ops))
```

- [x] **Step 4: 跑 producer 测试 + 全量回归**

Run: `python -m pytest tests/test_timesim_producer.py tests/ -q`
Expected: 新增测试全 PASS；MLP 行为测试（tp2_sp/tp1/AG deps/RS redirect）**逐字节不变**全 PASS；两个改写的守卫测试 PASS；全量无回退。若 MLP 行为测试失败=重构改了语义，回退重查（此任务是纯行为保持 + 能力扩展）。

- [x] **Step 5: Commit**

```bash
git add cost_eval/timesim/producer.py tests/test_timesim_producer.py
git commit -m "refactor(timesim): producer per-tensor 分片状态(T1-2,交接要点5,carrier=out_dim 首符号因子,S 分歧重分布注入)"
```

## Task 3: MLA 段端到端（SequenceParallelLinear + attention 族打通）

**Files:**
- Modify: `cost_eval/timesim/shard_rules.py`（weight_local 加 SPL 分支）
- Modify: `tests/conftest.py`（mla_dag 夹具）
- Test: `tests/test_timesim_mla_segment.py`

- [x] **Step 1: conftest 加 MLA 夹具（与 test_opdag_mla.py 同一提取参数 + infer_shapes 种子 `x`）**

```python
# tests/conftest.py 追加

@pytest.fixture(scope="session")
def mla_dag():
    """真 mindformers MLASelfAttention（DSv3, mla_qkv_concat=False）+ shape 推断。
    种子名是 `x`（MLA construct 入参，≠ MLP 的 hidden_states——2026-07-17 探针核实）。"""
    if not os.path.isdir(MF_ROOT):
        pytest.skip(f"mindformers 源根不存在: {MF_ROOT}")
    from cost_eval.opdag.extractor import extract_cell
    from cost_eval.opdag.module_resolver import resolve_layer_spec
    from cost_eval.opdag.shape_infer import infer_shapes
    spec_flags = {
        "multi_latent_attention": True, "mla_qkv_concat": False, "num_experts": 256,
        "qk_layernorm": True, "sparse_attention": False, "fused_norm": True,
        "moe_grouped_gemm": True, "use_contiguous_weight_layout_attention": False,
        "use_interleaved_weight_layout_mlp": True,
    }
    mla_flags = {
        "use_dsa": False, "use_flash_attention": True,
        "use_eod_attn_mask_compression": False, "cp": 1, "cp_ds": 1,
        "input_layout": "BNSD", "q_lora_rank": 1536,
        "compute_dtype": "bf16", "layernorm_compute_dtype": "fp32",
    }
    top = resolve_layer_spec(MF_ROOT, spec_flags)
    dag = extract_cell(
        MF_ROOT, "parallel_core/training_graph/transformer/multi_latent_attention.py",
        "MLASelfAttention", top.submodules["self_attention"], mla_flags,
        present_params={"rotary_pos_emb"})
    return infer_shapes(dag, {"x": "S·B·H"})
```

- [x] **Step 2: 写失败测试**

```python
# tests/test_timesim_mla_segment.py
"""MLA（attention 族）段端到端：per-tensor 分片状态 + SequenceParallelLinear（T1 Phase A 靶点，
T0 交接要点5「attention 族 cell 会最先撞上守卫」的闭环验收）。

MLA dims：DSv3 比例缩小——q_lora_rank=1536、kv_lora_rank=512、qk_nope=128、qk_rope=64、
v_head=128、n_heads=8（q_head_dim=192）。"""
import pytest

from cost_eval.model_spec import DimTable
from cost_eval.timesim.shard_rules import Degrees
from cost_eval.timesim.producer import build_segment
from cost_eval.timesim.ir import op_flops

DIMS = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                S=4096, B=1, vocab=129280, n_layers=4,
                q_lora_rank=1536, kv_lora_rank=512,
                qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128)


def test_mla_tp1_builds_end_to_end(mla_dag):
    seg = build_segment("l0.attn.fwd", mla_dag, DIMS, Degrees())
    assert all(o.op_type != "CommOp" for o in seg.ops)
    mm = [o for o in seg.ops if o.op_type == "MatMul"]
    assert [o.module for o in mm] == [
        "SequenceParallelLinear", "SequenceParallelLinear",
        "ColumnParallelLinear", "ColumnParallelLinear", "RowParallelLinear"]
    fa = next(o for o in seg.ops if o.op_type == "FlashAttention")
    # 提取序 (S,B,N,D)：q/k D=192(nope128+rope64)，v D=128
    assert fa.in_shapes[0] == (4096, 1, 8, 192)
    assert fa.in_shapes[2] == (4096, 1, 8, 128)
    assert fa.out_shape == (4096, 1, 8, 128)


def test_mla_tp2_sp_comm_structure(mla_dag):
    """tp=2+SP 的通信结构（per-tensor 语义）：q_up/kv_up 各自 gather 自己的输入（两条
    module-semantics AG——T0 全局位只会发一条）、rope 支在 pe_concat 汇合注入一条
    layout-redistribution AG（源事实4）、proj 后一条 RS。"""
    deg = Degrees(tp=2, sequence_parallel=True)
    seg = build_segment("l0.attn.fwd", mla_dag, DIMS, deg)
    comms = [o for o in seg.ops if o.op_type == "CommOp"]
    kinds = [(o.comm.ctype, o.module) for o in comms]
    assert kinds.count(("all_gather", "injected:module-semantics")) == 2
    assert kinds.count(("all_gather", "injected:layout-redistribution")) == 1
    assert kinds.count(("reduce_scatter", "RowParallelLinear")) == 1
    assert len(comms) == 4
    assert all(o.stream == "comm_tp" for o in comms)


def test_mla_tp2_sp_shapes(mla_dag):
    deg = Degrees(tp=2, sequence_parallel=True)
    seg = build_segment("l0.attn.fwd", mla_dag, DIMS, deg)
    mm = [o for o in seg.ops if o.op_type == "MatMul"]
    # SPL（q_down :797）：SP 驻留——输入 S/2、权重全量、输出 S/2（源事实1）
    q_down = mm[0]
    assert q_down.in_shapes == ((2048, 1, 1792), (1792, 1536))
    assert q_down.out_shape == (2048, 1, 1536)
    # q_up（Column :734）：AG 后全 seq，flat 出轴 carrier=n_heads ÷2 → 8·192/2=768
    q_up = mm[2]
    assert q_up.in_shapes[0] == (4096, 1, 1536)
    assert q_up.out_shape == (4096, 1, 768)
    # FA：heads ÷2、seq 全长
    fa = next(o for o in seg.ops if o.op_type == "FlashAttention")
    assert fa.in_shapes[0] == (4096, 1, 4, 192)
    assert fa.in_shapes[2] == (4096, 1, 4, 128)
    # proj（Row :306）：入 (4096,1,512)、权重 (512,1792)、RS 后 (2048,1,1792)
    proj = mm[4]
    assert proj.in_shapes == ((4096, 1, 512), (512, 1792))
    rs = next(o for o in seg.ops if o.op_type == "CommOp"
              and o.comm.ctype == "reduce_scatter")
    assert rs.out_shape == (2048, 1, 1792)


def test_mla_tp_shard_conserves_gemm_flops(mla_dag):
    """性质（全局守恒）：tp=2+sp 下 per-rank 每个 GEMM 都减半 → 总 per-rank = 全局/2。
    **五个矩乘各自 ÷tp，机理不同但都减半**：SPL（q_down/kv_down）靠序列 S 分片（SP 把 token
    维分到各 rank，probe 实证 q_down 入 (2048,1,1792)——SPL"权重不切"是 weight shape 的性质，
    由 test_mla_tp2_sp_shapes 的 q_down 权重 (1792,1536) 全量断言捕获，**不是** flops 不变）；
    Column（q_up/kv_up）靠输出 feature 分片；Row（proj）靠输入 feature 分片。逐 src 都减半。"""
    full = {o.src: op_flops(o) for o in
            build_segment("s", mla_dag, DIMS, Degrees()).ops if o.op_type == "MatMul"}
    tp2 = {o.src: op_flops(o) for o in
           build_segment("s", mla_dag, DIMS, Degrees(tp=2, sequence_parallel=True)).ops
           if o.op_type == "MatMul"}
    assert set(full) == set(tp2)
    for src, mod_full in full.items():
        assert tp2[src] * 2 == mod_full, src          # 每个矩乘 per-rank 减半（含 SPL）
    assert sum(tp2.values()) * 2 == sum(full.values())   # 全局守恒
```

- [x] **Step 3: 跑测试确认失败**（weight_local 对 SPL 无分支时权重意外被切/或 producer 守卫报 SPL 未知——按报错逐一）

- [x] **Step 4: shard_rules.weight_local 加 SPL 分支**

```python
def weight_local(sym: str, dims, module: str, tp: int) -> tuple[int, ...]:
    """线性层**权重**的 local shape（切分轴随模块语义，不是一律末轴）：
    ColumnParallelLinear   权重 [in, out] → out(末轴) ÷tp；
    RowParallelLinear      权重 [in, out] → in(轴0)  ÷tp；
    SequenceParallelLinear 权重不切（layers.py:819「A is not parallelized」，:850 布局
    ("None","None")——T1 Task 3）。"""
    vals = axis_values(sym, dims)
    if tp > 1:
        if module == "ColumnParallelLinear":
            vals[-1] = _div_exact(vals[-1], tp, "col-weight out")
        elif module == "RowParallelLinear":
            vals[0] = _div_exact(vals[0], tp, "row-weight in")
        elif module == "SequenceParallelLinear":
            pass                                   # 权重全量（每 rank 复制）
    return tuple(vals)
```

（producer 的 `_KNOWN_MATMUL_MODULES` 已在 Task 2 含 SPL。）

- [x] **Step 5: 跑测试（4 条全过）+ 全量回归** → **Step 6: Commit**

```bash
git add cost_eval/timesim/shard_rules.py tests/conftest.py tests/test_timesim_mla_segment.py
git commit -m "feat(timesim): MLA 段端到端——SequenceParallelLinear 权重不切+SP 透传(T1-3,layers.py:819 实证)"
```

## Task 4: inject_cp ring 结构化（跳数从系数变结构，overlap 涌现）

**Files:**
- Modify: `cost_eval/timesim/frame_comm.py`（inject_cp colossal 支）
- Modify: `cost_eval/timesim/ir.py`（CommSpec docstring p2p 口径）
- Modify: `tests/test_timesim_frame_comm.py`（colossal 断言更新）

**动机**：T0 版 colossal = 每 FA 前 1 条 p2p、跳数系数 (cp−1) 留给 op_cost——但 ring attention 的真实结构是 **cp 个 FA 块 × (cp−1) 跳 kv p2p 交替**，块间 p2p 可与上一块 FA 重叠。DES 的价值就在结构性 overlap 涌现（spec §5.5），系数化会把可遮盖通信错算成串行。结构化后 p2p 恒单跳，op_cost 不再需要跳数系数（口径单一化）。同时修正 T0 的 FA 算力低估：单 FA 节点（q=S/cp, kv=S/cp）只算了 1/cp 的注意力量；cp 个块结构化后总量对上。

- [x] **Step 1: 更新测试（改 test_timesim_frame_comm.py 的 colossal 用例）**

找到现有 colossal 断言测试（`kinds == ["p2p"]` 形态），替换为：

```python
def test_cp_ring_structural_blocks():
    """colossal ring 结构化（T1-4）：cp 个 FA 块、块间 cp-1 条单跳 kv p2p；
    p2p_k deps=()（段首即可发）、FA_k deps 含 p2p_k → p2p 与上一块 FA 的重叠交给 DES 涌现。
    下游依赖不换绑：最末块保留原 op_id。"""
    from cost_eval.timesim.ir import TimedOp, TimedSegment
    from cost_eval.timesim.frame_comm import inject_cp
    fa = TimedOp(op_id="a#0", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224),) * 3 + ((),), out_shape=(2048, 1, 8, 224),
                 dtype="bf16", stream="device", src="attention.py:1", deps=("a#9",))
    seg = inject_cp(TimedSegment("l.fwd", (fa,)), cp=4, method="colossal")
    fas = [o for o in seg.ops if o.op_type == "FlashAttention"]
    p2ps = [o for o in seg.ops if o.op_type == "CommOp"]
    assert len(fas) == 4 and len(p2ps) == 3
    assert [o.comm.ctype for o in p2ps] == ["p2p"] * 3
    assert all(o.stream == "comm_cp" and o.deps == () for o in p2ps)
    # 交替序：FA0, p2p1, FA1, p2p2, FA2, p2p3, FA3
    assert [o.op_type for o in seg.ops] == ["FlashAttention", "CommOp"] * 3 + ["FlashAttention"]
    # 末块保留原 id（下游 deps 不换绑）；前块带 .cpK 后缀
    assert fas[-1].op_id == "a#0"
    assert fas[0].op_id == "a#0.cp0"
    # 每块 FA 保留原跨流 deps；k≥1 块追加对应 p2p dep
    assert fas[0].deps == ("a#9",)
    assert p2ps[0].op_id in fas[1].deps and "a#9" in fas[1].deps
    # p2p 载荷 = k+v 单块字节（单跳口径）
    kv = 2048 * 1 * 8 * 224 * 2
    assert all(o.comm.volume_bytes == 2 * kv for o in p2ps)


def test_cp_ulysses_unchanged():
    """ulysses 支行为保持 T0（FA 前后各一条 a2a，载荷只切 qkv）。"""
    from cost_eval.timesim.ir import TimedOp, TimedSegment
    from cost_eval.timesim.frame_comm import inject_cp
    fa = TimedOp(op_id="a#0", op_type="FlashAttention", phase="fwd",
                 in_shapes=((2048, 1, 8, 224),) * 3, out_shape=(2048, 1, 8, 224),
                 dtype="bf16", stream="device", src="attention.py:1")
    seg = inject_cp(TimedSegment("l.fwd", (fa,)), cp=2, method="ulysses")
    kinds = [o.comm.ctype for o in seg.ops if o.op_type == "CommOp"]
    assert kinds == ["all_to_all", "all_to_all"]
```

（若旧文件已有等价 ulysses 用例则保留原用例、只删旧 colossal 断言。）

- [x] **Step 2: 跑确认失败** → **Step 3: 实现**

`frame_comm.py` 的 `inject_cp` colossal 支替换为：

```python
    ops: list[TimedOp] = []
    for op in seg.ops:
        if op.op_type != "FlashAttention":
            ops.append(op)
            continue
        if method == "colossal":
            # ring 结构化（T1-4）：cp 个 FA 块 × 块间 cp-1 条单跳 kv p2p。每块 shape 与
            # localize 后的原 FA 相同（q=S/cp × kv 环转块=S/cp）；p2p deps=()（kv 段首可发，
            # 与上一块 FA 的重叠由 DES 涌现——spec §5.5 位置即语义）；末块保留原 op_id，
            # 下游 deps 不换绑。causal zigzag 负载均衡差异归 op_cost 的 causal 系数（v1 均匀）。
            kv_bytes = (tensor_bytes(op.in_shapes[1], op.dtype)
                        + tensor_bytes(op.in_shapes[2], op.dtype))
            for k in range(cp):
                if k > 0:
                    p2p = TimedOp(
                        op_id=f"{op.op_id}.p2p{k}", op_type="CommOp", phase=op.phase,
                        in_shapes=(op.in_shapes[1], op.in_shapes[2]),
                        out_shape=op.in_shapes[1], dtype=op.dtype,
                        stream=COMM_STREAM["cp"], deps=(),
                        src="cp[colossal]:module-semantics",
                        comm=CommSpec("p2p", kv_bytes, "cp", cp))
                    ops.append(p2p)
                    blk_deps = op.deps + (p2p.op_id,)
                else:
                    blk_deps = op.deps
                blk_id = op.op_id if k == cp - 1 else f"{op.op_id}.cp{k}"
                ops.append(TimedOp(
                    op_id=blk_id, op_type="FlashAttention", phase=op.phase,
                    in_shapes=op.in_shapes, out_shape=op.out_shape, dtype=op.dtype,
                    stream=op.stream, src=op.src, deps=blk_deps, module=op.module))
        else:   # ulysses：保持 T0 行为（FA 前后各一条 all_to_all，载荷只切 qkv）
            ...（原 ulysses 分支代码原样保留）...
    return TimedSegment(seg.seg_id, tuple(ops))
```

`ir.py` CommSpec docstring 的 p2p 行改为：

```
      - p2p=单跳载荷字节（**恒单跳**：cp ring 的 cp−1 跳已由 frame_comm.inject_cp 结构化为
        cp−1 条 p2p op（T1-4），pp 本就单跳——op_cost 不再施跳数系数）；
```

- [x] **Step 4: 跑测试 + 全量回归** → **Step 5: Commit**

```bash
git add cost_eval/timesim/frame_comm.py cost_eval/timesim/ir.py tests/test_timesim_frame_comm.py
git commit -m "feat(timesim): inject_cp ring 结构化——cp 块 FA+单跳 p2p,overlap 交 DES 涌现(T1-4,修 FA 算力 1/cp 低估)"
```

---

# Phase B — op_cost（M8）

## Task 5: machine.py——时间侧硬件常数（契约4 独立命名空间）

**Files:**
- Create: `cost_eval/timesim/machine.py`
- Test: `tests/test_timesim_op_cost.py`（先写 machine 部分）

- [x] **Step 1: 写失败测试**

```python
# tests/test_timesim_op_cost.py
"""machine.TimeHardware + op_cost.CostModel（M8，spec §4.1）。
测试恒用 SYNTH 合成硬件（整数好算，断言零公差）；DEFAULT_910B 只测元属性不测数值。"""
import pytest

from cost_eval.timesim.machine import TimeHardware, DEFAULT_910B, synth_hw


def test_synth_hw_roundtrip():
    hw = synth_hw()
    assert hw.peak("bf16") == 100e12
    assert hw.link("tp") == (1.0, 1e11)          # (alpha_us, bytes_per_s)
    assert hw.host_us("View", "fwd") == 1.0
    assert hw.host_us("MatMul", "bwd") == 2.0    # 相位缺省价
    assert hw.calibrated is False


def test_default_910b_marked_uncalibrated():
    """厂商规格代入（spec §7.2 诚实边界1）：任何消费方都能看到未标定标记。"""
    assert DEFAULT_910B.calibrated is False
    assert DEFAULT_910B.name == "910B"
    for axis in ("tp", "cp", "ep", "dp", "pp"):
        alpha, bw = DEFAULT_910B.link(axis)
        assert alpha > 0 and bw > 0


def test_unknown_axis_fail_loud():
    with pytest.raises(KeyError):
        synth_hw().link("nvlink")
```

- [x] **Step 2: 跑确认失败** → **Step 3: 实现**

```python
# cost_eval/timesim/machine.py
"""时间侧硬件常数（spec §2.1 的 HardwareSpec 输入位）。

**为什么不进 specs.HardwareSpec**：那是内存侧命名空间（max_device_memory/framework_reserve），
解耦契约4「标定命名空间分离」——时间标定（峰值/带宽/η/host 单价）绝不回流内存侧，反之亦然。

单位约定：peak_flops = FLOP/s；hbm_bw / link bw = bytes/s；alpha = us；host 单价 = us。
DEFAULT_910B 全部是**厂商规格/公开口径占位**（calibrated=False，spec §7.2-1：跨机 α-β 未标定、
report 须标记）——T2 经验库标定后换库即换数，代码零改动（结构与常数分离铁律）。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TimeHardware:
    peak_flops: dict                 # dtype → FLOP/s（cube 峰值）
    hbm_bw: float                    # bytes/s
    link_alpha_us: dict              # 通信轴 → 启动时延 us（axis: tp/cp/ep/dp/pp）
    link_bw: dict                    # 通信轴 → bytes/s（v1 按轴给定；topo 推断=T2）
    eta: dict                        # {"gemm","fa","bw","opt"} → 默认效率（三级退化第3级）
    host_unit_us: dict               # {("*",phase): us} 相位缺省 + {(op_type,phase): us} 覆盖
    name: str = ""
    calibrated: bool = False

    def peak(self, dtype: str) -> float:
        return self.peak_flops.get(dtype) or self.peak_flops["bf16"]

    def link(self, axis: str) -> tuple[float, float]:
        return (self.link_alpha_us[axis], self.link_bw[axis])   # 未知轴 KeyError=fail-loud

    def host_us(self, op_type: str, phase: str) -> float:
        v = self.host_unit_us.get((op_type, phase))
        return v if v is not None else self.host_unit_us[("*", phase)]


def synth_hw(**over) -> TimeHardware:
    """测试用合成硬件：整数好算（peak=100 TFLOPS、HBM=1 TB/s、链路=0.1 TB/s、η 全 1），
    断言可零公差。生产勿用。"""
    base = dict(
        peak_flops={"bf16": 100e12, "fp16": 100e12, "fp32": 25e12},
        hbm_bw=1e12,
        link_alpha_us={a: 1.0 for a in ("tp", "cp", "ep", "dp", "pp")},
        link_bw={a: 1e11 for a in ("tp", "cp", "ep", "dp", "pp")},
        eta={"gemm": 1.0, "fa": 1.0, "bw": 1.0, "opt": 1.0},
        host_unit_us={("*", "fwd"): 1.0, ("*", "bwd"): 2.0, ("*", "recomp"): 1.0,
                      ("View", "fwd"): 1.0, ("MatMul", "bwd"): 2.0},
        name="synth", calibrated=False,
    )
    base.update(over)
    return TimeHardware(**base)


# 910B 占位（**未标定**，厂商规格/公开口径；T2 标定覆盖）。数值仅用于「未标定相对排序」模式
# （spec D1 退化档），不进任何测试断言。
DEFAULT_910B = TimeHardware(
    peak_flops={"bf16": 376e12, "fp16": 376e12, "fp32": 94e12},
    hbm_bw=1.6e12,
    link_alpha_us={"tp": 8.0, "cp": 8.0, "ep": 10.0, "dp": 10.0, "pp": 15.0},
    link_bw={"tp": 2.8e11, "cp": 2.8e11, "ep": 2.5e10, "dp": 2.5e10, "pp": 2.5e10},
    eta={"gemm": 0.7, "fa": 0.45, "bw": 0.8, "opt": 0.8},   # spec §4.2 三级退化默认档
    host_unit_us={("*", "fwd"): 3.0, ("*", "bwd"): 1.8, ("*", "recomp"): 3.0},
    name="910B", calibrated=False,
)
```

- [x] **Step 4: 跑测试 + 全量** → **Step 5: Commit**

```bash
git add cost_eval/timesim/machine.py tests/test_timesim_op_cost.py
git commit -m "feat(timesim): TimeHardware 时间侧硬件常数(T1-5,契约4 独立命名空间,910B 占位标未标定)"
```

## Task 6: op_cost.py——OpCost + CostModel

**Files:**
- Create: `cost_eval/timesim/op_cost.py`
- Test: `tests/test_timesim_op_cost.py`（追加）

- [x] **Step 1: 追加失败测试**

```python
# tests/test_timesim_op_cost.py 追加
from cost_eval.timesim.ir import TimedOp, CommSpec
from cost_eval.timesim.op_cost import CostModel, OpCost

HW = synth_hw()
CM = CostModel(HW)


def _op(**kw):
    base = dict(op_id="n0", op_type="MatMul", phase="fwd",
                in_shapes=((4096, 1, 1792), (1792, 3072)), out_shape=(4096, 1, 3072),
                dtype="bf16", stream="device", src="mlp.py:1")
    base.update(kw)
    return TimedOp(**base)


def test_gemm_cost_exact():
    c = CM.cost(_op())
    flops = 2 * 4096 * 1 * 1792 * 3072
    assert c.flops == flops
    assert c.t_dev_us == pytest.approx(flops / 100e12 * 1e6)   # η_gemm=1
    assert c.t_comm_us == 0.0
    assert c.bound == "compute"                                 # AI 远超 ridge=100
    assert c.provenance == "theory"
    assert c.eta_key == "gemm:bf16"


def test_fa_cost_formula():
    """FA fwd = 2·B·N·Sq·Skv·(Dq+Dv)·causal系数（spec §4.1；causal 默认 0.5）；Grad=2.5×。"""
    fa = _op(op_type="FlashAttention",
             in_shapes=((4096, 1, 8, 192), (4096, 1, 8, 192), (4096, 1, 8, 128), ()),
             out_shape=(4096, 1, 8, 128))
    c = CM.cost(fa)
    want = int(2 * 1 * 8 * 4096 * 4096 * (192 + 128) * 0.5)
    assert c.flops == want
    g = CM.cost(fa := _op(op_type="FlashAttentionGrad", phase="bwd",
                          in_shapes=fa.in_shapes, out_shape=fa.out_shape))
    assert g.flops == int(want * 2.5)
    full = CostModel(HW, causal=False).cost(_op(
        op_type="FlashAttention",
        in_shapes=((4096, 1, 8, 192), (4096, 1, 8, 192), (4096, 1, 8, 128), ()),
        out_shape=(4096, 1, 8, 128)))
    assert full.flops == want * 2


def test_fa_requires_4d_fail_loud():
    fa = _op(op_type="FlashAttention", in_shapes=((4096, 1792), (4096, 1792), (4096, 1792)))
    with pytest.raises(ValueError, match="4D"):
        CM.cost(fa)


def test_bandwidth_op_cost():
    """带宽类（flops=0）：t_dev = bytes_rw/(HBM·η)；空 shape（`?` 容忍位）计 0 字节。"""
    norm = _op(op_type="Norm", in_shapes=((4096, 1, 1792), ()), out_shape=(4096, 1, 1792))
    c = CM.cost(norm)
    b = 4096 * 1792 * 2 * 2                       # in + out，bf16
    assert c.flops == 0 and c.bytes_rw == b
    assert c.t_dev_us == pytest.approx(b / 1e12 * 1e6)
    assert c.bound == "memory"


def test_view_host_only_zero_device():
    v = _op(op_type="View", stream="host_only")
    c = CM.cost(v)
    assert c.t_dev_us == 0.0 and c.bound == "host"
    assert c.t_host_us == 1.0


def test_comm_costs_per_ctype():
    """t_comm 按 ir.py volume 口径（AG=分片、RS/AR=全量、p2p 单跳）+ ring 系数（spec §4.1）。"""
    n, bw, alpha = 4, 1e11, 1.0
    mk = lambda ct, vol: _op(op_type="CommOp", stream="comm_tp", in_shapes=(), out_shape=(),
                             comm=CommSpec(ct, vol, "tp", n))
    v = 1_000_000
    assert CM.cost(mk("all_gather", v)).t_comm_us == pytest.approx(alpha + v * (n - 1) / bw * 1e6)
    assert CM.cost(mk("reduce_scatter", v)).t_comm_us == pytest.approx(alpha + v * (n - 1) / n / bw * 1e6)
    assert CM.cost(mk("all_reduce", v)).t_comm_us == pytest.approx(alpha + 2 * v * (n - 1) / n / bw * 1e6)
    assert CM.cost(mk("all_to_all", v)).t_comm_us == pytest.approx(alpha + v * (n - 1) / n / bw * 1e6)
    assert CM.cost(mk("p2p", v)).t_comm_us == pytest.approx(alpha + v / bw * 1e6)
    assert CM.cost(mk("all_gather", v)).bound == "comm"


def test_generic_grad_bytes_convention():
    """通用 <op>Grad（bwd_rules 约定 in=(dy,*fwd_ins)、out=dy）：bytes 直接按 shapes 求和。"""
    g = _op(op_type="NormGrad", phase="bwd",
            in_shapes=((4096, 1, 1792), (4096, 1, 1792)), out_shape=(4096, 1, 1792))
    c = CM.cost(g)
    assert c.bytes_rw == 3 * 4096 * 1792 * 2
    assert c.t_host_us == 2.0                     # bwd 相位单价


def test_host_dominated_flag():
    tiny = _op(op_type="Cast", in_shapes=((8, 8),), out_shape=(8, 8))
    assert CM.cost(tiny).host_dominated is True
```

- [x] **Step 2: 跑确认失败** → **Step 3: 实现**

```python
# cost_eval/timesim/op_cost.py
"""M8 op_cost（spec §4.1）：TimedOp → OpCost。

T1 = 三级退化的第 3 级（roofline × 默认 η，provenance="theory"，spec §4.2）；OpTimeLibrary
命中/内插（第 1/2 级）T2 接入——CostModel 预留 lib 参数位，传入即 NotImplementedError
（防静默假装标定过）。

FLOPs/bytes 公式按 op_type 内置（spec §4.1）：
  GEMM 族   = ir.op_flops（2·numel(A)·N_out，对 fwd/dX/dW 一致成立）；
  FA        = 2·B·N·Sq·Skv·(Dq+Dv)·causal系数（QKᵀ+PV 两个 batched GEMM；causal=0.5，
              ring zigzag 均衡差异并入该系数）——q/k/v 取 in_shapes[:3]，**约定提取序 4D
              (S,B,N,D)**（MLA 探针实证；非 4D fail-loud 不猜 layout）；Grad=2.5×（spec）；
  带宽类    = bytes_rw = Σ in/out tensor_bytes（空 shape=`?` 容忍位计 0）；
  View/host_only = 全 0（发射成本走 host 单价）；
  CommOp    = t_comm = α(axis) + moved/BW(axis)，moved 按 ir.py CommSpec volume 口径 ×
              ring 系数：AG(分片)×(n−1)、RS/AR(全量)×(n−1)/n（AR 再 ×2）、A2A×(n−1)/n
              （对分带宽口径）、p2p 恒单跳（T1-4 结构化后无跳数系数）。
bound/host_dominated 是 op 内禀静态属性；真 host-bound（device 空洞）只能由 segment_sim
涌现（spec §4.1「分类口径的关键区分」）。"""
from __future__ import annotations

from dataclasses import dataclass

from .ir import TimedOp, tensor_bytes, op_flops, STREAM_HOST_ONLY
from .machine import TimeHardware

_GEMM = ("MatMul", "GroupedMatMul")


@dataclass(frozen=True)
class OpCost:
    t_host_us: float
    t_dev_us: float
    t_comm_us: float
    flops: int
    bytes_rw: int
    arith_intensity: float
    ridge: float
    bound: str                # compute | memory | comm | host
    host_dominated: bool
    eta_key: str
    provenance: str           # T1 恒 "theory"（spec §4.2 第3级）


def _fa_flops(top: TimedOp, causal: bool) -> int:
    if len(top.in_shapes) < 3 or any(len(s) != 4 for s in top.in_shapes[:3]):
        raise ValueError(
            f"op_cost: FlashAttention 期待 4D q/k/v（提取序 (S,B,N,D)），"
            f"got {top.in_shapes[:3]} @ {top.src}——fail-loud 不猜 layout")
    q, k, v = top.in_shapes[:3]
    sq, b, n, dq = q
    skv, dv = k[0], v[3]
    coeff = 0.5 if causal else 1.0
    return int(2 * b * n * sq * skv * (dq + dv) * coeff)


class CostModel:
    def __init__(self, hw: TimeHardware, *, causal: bool = True, lib=None):
        if lib is not None:
            raise NotImplementedError("OpTimeLibrary=T2（spec §9）；T1 只有 theory 档")
        self.hw = hw
        self.causal = causal

    def _bytes_rw(self, top: TimedOp) -> int:
        total = sum(tensor_bytes(s, top.dtype) for s in top.in_shapes if s)
        if top.out_shape:
            total += tensor_bytes(top.out_shape, top.dtype)
        return total

    def comm_time_us(self, c) -> float:
        """CommSpec → us（公有：report 的步收尾 dp_replicate AR 等复用同一口径）。"""
        alpha, bw = self.hw.link(c.group_axis)
        n, v = c.group_size, c.volume_bytes
        if c.ctype == "all_gather":
            moved = v * (n - 1)
        elif c.ctype == "reduce_scatter":
            moved = v * (n - 1) / n
        elif c.ctype == "all_reduce":
            moved = 2 * v * (n - 1) / n
        elif c.ctype == "all_to_all":
            moved = v * (n - 1) / n
        elif c.ctype == "p2p":
            moved = v
        else:
            raise ValueError(f"op_cost: 未知 ctype {c.ctype!r}（fail-loud）")
        return alpha + moved / bw * 1e6

    def cost(self, top: TimedOp) -> OpCost:
        hw = self.hw
        t_host = hw.host_us(top.op_type, top.phase)
        bytes_rw = self._bytes_rw(top)
        ridge = hw.peak(top.dtype) / hw.hbm_bw

        if top.op_type == "CommOp":
            t_comm = self.comm_time_us(top.comm)
            return OpCost(t_host, 0.0, t_comm, 0, bytes_rw, 0.0, ridge, "comm",
                          t_host > t_comm, f"comm:{top.comm.ctype}", "theory")

        if top.stream == STREAM_HOST_ONLY or top.op_type == "View":
            return OpCost(t_host, 0.0, 0.0, 0, bytes_rw, 0.0, ridge, "host",
                          True, "host_only", "theory")

        if top.op_type in _GEMM:
            flops, eta_key = op_flops(top), "gemm"
        elif top.op_type == "FlashAttention":
            flops, eta_key = _fa_flops(top, self.causal), "fa"
        elif top.op_type == "FlashAttentionGrad":
            flops, eta_key = int(_fa_flops(top, self.causal) * 2.5), "fa"
        else:
            flops, eta_key = 0, "bw"

        if flops > 0:
            t_dev = flops / (hw.peak(top.dtype) * hw.eta[eta_key]) * 1e6
        else:
            t_dev = bytes_rw / (hw.hbm_bw * hw.eta["bw"]) * 1e6
        ai = flops / bytes_rw if bytes_rw else 0.0
        bound = "compute" if ai >= ridge else "memory"
        return OpCost(t_host, t_dev, 0.0, flops, bytes_rw, ai, ridge, bound,
                      t_host > t_dev, f"{eta_key}:{top.dtype}", "theory")


def price_segment(seg, cm: CostModel) -> dict:
    """整段定价：{op_id → OpCost}（segment_sim/report 的输入）。"""
    return {op.op_id: cm.cost(op) for op in seg.ops}
```

- [x] **Step 4: 跑测试 + 全量** → **Step 5: Commit**

```bash
git add cost_eval/timesim/op_cost.py tests/test_timesim_op_cost.py
git commit -m "feat(timesim): OpCost+CostModel——roofline×默认η+per-ctype 通信系数(T1-6,M8,provenance=theory)"
```

---

# Phase C — segment_sim（M9-L1）

## Task 7: pass_builder——多层段拼接成连续 pass（spec §5.2）

**Files:**
- Create: `cost_eval/timesim/pass_builder.py`
- Test: `tests/test_timesim_pass_builder.py`

**动机（spec §5.2）**：PyNative 的 host 不在层边界停——L1 仿真单元是一个 (stage, phase, microbatch[, chunk]) 的**完整连续 pass**，按层孤立仿真再求和会掐断 host run-ahead 系统性高估。故需把 per-layer TimedSegment 拼成单段：op_id 加层前缀防碰撞（同 cell 多层 id 相同）、段内 deps 同步改写、层标记保留供 SegmentTime.per_layer 聚合。

- [x] **Step 1: 写失败测试**

```python
# tests/test_timesim_pass_builder.py
"""concat_segments：op_id 层前缀 + deps 改写 + 层标记。"""
from cost_eval.timesim.ir import TimedOp, TimedSegment
from cost_eval.timesim.pass_builder import concat_segments, layer_of


def _seg(seg_id):
    a = TimedOp(op_id="c#0", op_type="MatMul", phase="fwd",
                in_shapes=((4, 4), (4, 4)), out_shape=(4, 4), dtype="bf16",
                stream="device", src="f.py:1")
    b = TimedOp(op_id="c#1", op_type="CommOp", phase="fwd", in_shapes=((4, 4),),
                out_shape=(4, 4), dtype="bf16", stream="comm_tp", src="f.py:2",
                deps=("c#0",))
    return TimedSegment(seg_id, (a, b))


def test_concat_prefixes_and_remaps():
    p = concat_segments("s0.mb0.fwd", [_seg("layer_0.fwd"), _seg("layer_1.fwd")])
    assert p.seg_id == "s0.mb0.fwd"
    ids = [o.op_id for o in p.ops]
    assert ids == ["layer_0.fwd/c#0", "layer_0.fwd/c#1",
                   "layer_1.fwd/c#0", "layer_1.fwd/c#1"]
    assert p.ops[1].deps == ("layer_0.fwd/c#0",)
    assert p.ops[3].deps == ("layer_1.fwd/c#0",)          # 层内 deps 只指向本层前缀
    assert layer_of(p.ops[2].op_id) == "layer_1.fwd"


def test_concat_drops_cross_layer_unknown_deps():
    """跨段/外部 dep（段内查无此 id）保留原样——segment_sim 视作段首已满足
    （pass 边界截断，spec §7.2-4）；本函数不静默删除信息。"""
    a = TimedOp(op_id="c#0", op_type="MatMul", phase="bwd",
                in_shapes=((4, 4), (4, 4)), out_shape=(4, 4), dtype="bf16",
                stream="device", src="f.py:1", deps=("external#9",))
    p = concat_segments("s0.mb0.bwd", [TimedSegment("layer_0.bwd", (a,))])
    assert p.ops[0].deps == ("external#9",)
```

- [x] **Step 2: 跑确认失败** → **Step 3: 实现**

```python
# cost_eval/timesim/pass_builder.py
"""把 per-layer TimedSegment 拼成一个连续 pass 段（spec §5.2：L1 仿真单元=完整 pass，
host 流一条贯到底；"层"只是报告标记）。

op_id 前缀 = f"{layer_seg_id}/"（同 cell 多层的 opdag 节点 id 相同，不前缀必碰撞）；
deps 改写规则：dep 指向本层段内某 op → 加同层前缀；查无此 id（跨段/外部）→ 原样保留，
segment_sim 对未知 dep 按段首已满足处理（pass 边界 run-ahead 截断，spec §7.2-4 有界近似）。"""
from __future__ import annotations

from dataclasses import replace

from .ir import TimedOp, TimedSegment


def layer_of(op_id: str) -> str:
    """pass 内 op_id → 层标记（无前缀=拼接前的裸段，归 ""）。"""
    return op_id.rsplit("/", 1)[0] if "/" in op_id else ""


def concat_segments(seg_id: str, segments: list) -> TimedSegment:
    ops: list[TimedOp] = []
    for seg in segments:
        local_ids = {o.op_id for o in seg.ops}
        prefix = seg.seg_id + "/"
        for o in seg.ops:
            ops.append(replace(
                o, op_id=prefix + o.op_id,
                deps=tuple((prefix + d) if d in local_ids else d for d in o.deps)))
    return TimedSegment(seg_id, tuple(ops))
```

- [x] **Step 4: 跑测试 + 全量** → **Step 5: Commit**

```bash
git add cost_eval/timesim/pass_builder.py tests/test_timesim_pass_builder.py
git commit -m "feat(timesim): pass_builder 多层段拼接——连续 pass 语义(T1-7,spec §5.2)"
```

## Task 8: segment_sim——段内多流 DES + 三态归因

**Files:**
- Create: `cost_eval/timesim/segment_sim.py`
- Test: `tests/test_timesim_segment_sim.py`

- [x] **Step 1: 写失败测试**

```python
# tests/test_timesim_segment_sim.py
"""段内多流 DES（M9-L1，spec §5）：H 串行不等 device、D/C_* FIFO+跨流 deps、三态归因守恒。
全部用手搓 TimedOp + 手写 OpCost（不依赖 CostModel——L1 逻辑与定价解耦）。"""
import pytest

from cost_eval.timesim.ir import TimedOp, CommSpec, TimedSegment
from cost_eval.timesim.op_cost import OpCost
from cost_eval.timesim.segment_sim import simulate_segment


def _cost(host=1.0, dev=0.0, comm=0.0, bound="compute"):
    return OpCost(host, dev, comm, 0, 0, 0.0, 100.0, bound, host > dev, "t", "theory")


def _dev(op_id, deps=(), src="f.py:1"):
    return TimedOp(op_id=op_id, op_type="MatMul", phase="fwd",
                   in_shapes=((4, 4), (4, 4)), out_shape=(4, 4), dtype="bf16",
                   stream="device", src=src, deps=deps)


def _comm(op_id, deps=(), axis="tp"):
    return TimedOp(op_id=op_id, op_type="CommOp", phase="fwd", in_shapes=((4, 4),),
                   out_shape=(4, 4), dtype="bf16", stream=f"comm_{axis}", src="f.py:2",
                   deps=deps, comm=CommSpec("all_reduce", 32, axis, 2))


def _view(op_id, deps=()):
    return TimedOp(op_id=op_id, op_type="View", phase="fwd", in_shapes=((4, 4),),
                   out_shape=(4, 4), dtype="bf16", stream="host_only", src="f.py:3",
                   deps=deps)


def test_device_pipelining_no_gap():
    """host 发射快于 device 执行 → device 背靠背，无空洞。"""
    seg = TimedSegment("s", (_dev("a"), _dev("b"), _dev("c")))
    costs = {i: _cost(host=1.0, dev=10.0) for i in ("a", "b", "c")}
    st = simulate_segment(seg, costs)
    assert st.duration_us == pytest.approx(1.0 + 30.0)      # 首 op 等发射，后续流水
    assert st.t_host_gap == pytest.approx(1.0)
    assert st.t_compute == pytest.approx(30.0)
    assert st.host_len_us == pytest.approx(3.0)


def test_host_bound_emerges():
    """host 单价 > device 时长 → 每 op 都等发射 → host_gap 主导（D2：涌现非拍定）。"""
    seg = TimedSegment("s", (_dev("a"), _dev("b"), _dev("c")))
    costs = {i: _cost(host=10.0, dev=1.0) for i in ("a", "b", "c")}
    st = simulate_segment(seg, costs)
    assert st.duration_us == pytest.approx(31.0)            # 3×10 发射 + 尾 op 1
    assert st.t_host_gap == pytest.approx(28.0)             # 10-1=9 ×2 + 首 10
    assert st.t_compute == pytest.approx(3.0)


def test_exposed_comm_attribution():
    """b 依赖慢通信 → device 空洞记 exposed_comm[tp]（发射早已完成——先查发射再查依赖）。"""
    seg = TimedSegment("s", (_dev("a"), _comm("c1", deps=("a",)), _dev("b", deps=("c1",))))
    costs = {"a": _cost(host=1.0, dev=5.0), "c1": _cost(host=1.0, comm=50.0),
             "b": _cost(host=1.0, dev=5.0)}
    st = simulate_segment(seg, costs)
    # 手算：a 发射@1→D[1,6]；c1 发射@2、dep a@6 → C_tp[6,56]；b 发射@3、dep c1@56 → D[56,61]
    assert st.duration_us == pytest.approx(61.0)
    assert st.t_exposed_comm["tp"] == pytest.approx(50.0)
    assert st.t_host_gap == pytest.approx(1.0)
    assert st.t_compute == pytest.approx(10.0)


def test_overlapped_comm_not_exposed():
    """通信与后续无依赖计算并行 → 不暴露（overlap 是位置+DES 的涌现结果）。"""
    seg = TimedSegment("s", (_dev("a"), _comm("c1", deps=("a",)), _dev("b")))
    costs = {"a": _cost(host=1.0, dev=5.0), "c1": _cost(host=1.0, comm=3.0),
             "b": _cost(host=1.0, dev=20.0)}
    st = simulate_segment(seg, costs)
    assert st.t_exposed_comm.get("tp", 0.0) == 0.0
    assert st.duration_us == pytest.approx(26.0)


def test_three_state_conservation():
    """L0①：Σ(t_compute+t_membound+t_host_gap+Σexposed) == makespan（逐段守恒可测）。"""
    seg = TimedSegment("s", (_view("v0"), _dev("a", deps=("v0",)),
                             _comm("c1", deps=("a",)), _dev("b", deps=("c1",)),
                             _view("v1", deps=("b",))))
    costs = {"v0": _cost(host=2.0), "a": _cost(host=1.0, dev=7.0, bound="memory"),
             "c1": _cost(host=1.0, comm=9.0), "b": _cost(host=3.0, dev=4.0),
             "v1": _cost(host=2.0)}
    st = simulate_segment(seg, costs)
    total = st.t_compute + st.t_membound + st.t_host_gap + sum(st.t_exposed_comm.values())
    assert total == pytest.approx(st.duration_us)
    assert st.t_membound == pytest.approx(7.0)


def test_per_layer_and_top_contributors():
    from cost_eval.timesim.pass_builder import concat_segments
    seg = concat_segments("p", [TimedSegment("layer_0.fwd", (_dev("a"),)),
                                TimedSegment("layer_1.fwd", (_dev("a"),))])
    costs = {o.op_id: _cost(host=1.0, dev=5.0) for o in seg.ops}
    st = simulate_segment(seg, costs)
    assert set(st.per_layer) == {"layer_0.fwd", "layer_1.fwd"}
    assert st.per_layer["layer_1.fwd"] == pytest.approx(5.0)
    assert st.top_contributors[0] in ("layer_0.fwd/a", "layer_1.fwd/a")
```

- [x] **Step 2: 跑确认失败** → **Step 3: 实现**

```python
# cost_eval/timesim/segment_sim.py
"""段内多流离散事件仿真（M9-L1，spec §5）。

流推进语义（§5.1）：
  H（host 发射）：全部 op 按段内序串行 `h_end_i = h_start_i + t_host_i`——PyNative 异步下发，
    **默认不等 device**（同步点白名单 v1 = pass 边界，段内无）。
  D（device 计算）：FIFO；`d_start = max(该 op 的 h_end, D 前 op 结束, 跨流 deps 完成)`。
  C_*（通信按 group 轴分道）：同道 FIFO + 跨流 deps；不同轴通信域并发（多流带宽争抢不建，
    §7.2-2）。
host_only op 的"完成时刻"= 其 h_end（视图只有发射成本）。
未知 dep（跨段/外部，pass_builder 契约）→ 段首已满足=0（run-ahead 跨 pass 截断，§7.2-4）。

三态归因（§5.3，涌现非拍定；判据次序先查发射、再查依赖）：对 D 流 [0, makespan] 逐段积分：
  D 忙 → 按 OpCost.bound 劈 t_compute / t_membound（comm/host 类不上 D 流）；
  D 闲且下一 op 未被 H 发射 → t_host_gap；
  D 闲且已发射但依赖未完 → t_exposed_comm[阻塞 dep 的通信轴]（取完成最晚的 comm dep；
    无 comm dep 可归（纯 host_only 依赖）→ 归 host_gap，保守恒）。
  设备之后的尾段 [d_last, makespan]：逐通信道busy区间归 exposed（重叠取完成更晚者），
    其余（host 尾巴）归 host_gap——Σ三态 == makespan 恒等（L0①，测试守恒）。"""
from __future__ import annotations

from dataclasses import dataclass

from .ir import TimedSegment, STREAM_DEVICE, STREAM_HOST_ONLY
from .pass_builder import layer_of


@dataclass(frozen=True)
class SegmentTime:
    seg_id: str
    duration_us: float
    t_compute: float
    t_membound: float
    t_host_gap: float
    t_exposed_comm: dict
    host_len_us: float
    per_layer: dict
    top_contributors: list


def simulate_segment(seg: TimedSegment, costs: dict) -> SegmentTime:
    h_clock = 0.0
    end: dict[str, float] = {}
    lane_clock: dict[str, float] = {}
    dev_rows = []                       # (start, fin, op, h_end, blocking_comm_axis|None)
    comm_rows = []                      # (start, fin, axis)
    op_by_id = {o.op_id: o for o in seg.ops}

    for op in seg.ops:
        c = costs[op.op_id]
        h_clock += c.t_host_us
        if op.stream == STREAM_HOST_ONLY:
            end[op.op_id] = h_clock
            continue
        dep_end = 0.0
        blocking = None
        for d in op.deps:
            e = end.get(d, 0.0)          # 未知 dep=段首已满足（模块 docstring）
            if e > dep_end:
                dep_end = e
                dep_op = op_by_id.get(d)
                blocking = (dep_op.comm.group_axis
                            if dep_op is not None and dep_op.comm is not None else None)
        dur = c.t_comm_us if op.op_type == "CommOp" else c.t_dev_us
        start = max(lane_clock.get(op.stream, 0.0), h_clock, dep_end)
        fin = start + dur
        lane_clock[op.stream] = fin
        end[op.op_id] = fin
        if op.stream == STREAM_DEVICE:
            dev_rows.append((start, fin, op, h_clock, blocking))
        else:
            comm_rows.append((start, fin, op.comm.group_axis, op.op_id))

    makespan = max(list(lane_clock.values()) + [h_clock] + [0.0])

    t_compute = t_membound = host_gap = 0.0
    exposed: dict[str, float] = {}
    per_layer: dict[str, float] = {}
    cursor = 0.0
    for start, fin, op, h_end, blocking in dev_rows:
        gap = start - cursor
        if gap > 0:
            hg = min(max(h_end - cursor, 0.0), gap)         # 先查发射
            host_gap += hg
            rest = gap - hg
            if rest > 0:                                     # 再查依赖
                if blocking is not None:
                    exposed[blocking] = exposed.get(blocking, 0.0) + rest
                else:
                    host_gap += rest                         # 无 comm 可归（保守恒）
        c = costs[op.op_id]
        if c.bound == "memory":
            t_membound += fin - start
        else:
            t_compute += fin - start
        per_layer[layer_of(op.op_id)] = per_layer.get(layer_of(op.op_id), 0.0) + (fin - start)
        cursor = fin

    # 设备后尾段 [cursor, makespan]（device 全空闲）：**扫描线**逐子区间归因——每个子区间取
    # 当刻仍在传输、且完成最晚的通信轴（"取完成最晚者"，与 dev-gap 的 blocking 同语义），无任何
    # 通信在传则归 host_gap。逐子区间恰归一次 → Σ==makespan 守恒。
    # （不可用"按 fin 排序 + 单一前沿"——跨轴嵌套/交叉区间下会漏计并错配轴，破坏 L0① 守恒，
    #  frame_comm 的 FSDP/EP/CP deps=() 预取 + 段内 tp 通信尾即真实触发该形态，T1-8 对抗性 review 实证。）
    tail_ivals = [(max(s, cursor), f, axis) for s, f, axis, _ in comm_rows if f > cursor]
    if tail_ivals:
        pts = sorted({cursor, makespan}
                     | {p for s, f, _ in tail_ivals for p in (s, f) if cursor <= p <= makespan})
        for lo, hi in zip(pts, pts[1:]):
            if hi <= lo:
                continue
            active = [(f, axis) for s, f, axis in tail_ivals if s <= lo < f]
            if active:
                exposed_axis = max(active)[1]                # 完成最晚的活跃轴
                exposed[exposed_axis] = exposed.get(exposed_axis, 0.0) + (hi - lo)
            else:
                host_gap += hi - lo                          # 尾段无通信在传 → host 尾巴
    elif makespan > cursor:
        host_gap += makespan - cursor                        # 纯 host 尾（无通信尾段，如末尾 View）

    dur_of = {op.op_id: fin - start for start, fin, op, _, _ in dev_rows}
    dur_of.update({op_id: fin - start for start, fin, _, op_id in comm_rows})
    top = sorted(dur_of, key=dur_of.get, reverse=True)[:10]

    return SegmentTime(seg.seg_id, makespan, t_compute, t_membound, host_gap,
                       exposed, h_clock, per_layer, top)
```

- [x] **Step 4: 跑测试（手算断言逐条核对，失败=先查测试手算再查实现——三态判据次序是 spec 规定，不许倒换）+ 全量回归**

- [x] **Step 5: Commit**

```bash
git add cost_eval/timesim/segment_sim.py tests/test_timesim_segment_sim.py
git commit -m "feat(timesim): segment_sim 段内多流 DES+三态归因守恒(T1-8,M9-L1,spec §5)"
```

---

# Phase D — pipeline_sim（M9-L2）+ StepTimeReport

## Task 9: pipeline_sim——全局调度 DES

**Files:**
- Create: `cost_eval/timesim/pipeline_sim.py`
- Test: `tests/test_timesim_pipeline_sim.py`

- [x] **Step 1: 写失败测试**

```python
# tests/test_timesim_pipeline_sim.py
"""全局调度 DES（M9-L2，spec §6.1/6.2）：调度序取自 schedule.py（时间侧不重新发明调度），
p2p=依赖边时延（不占 device 资源），bubble=仿真空闲积分（非公式）。"""
import pytest

from cost_eval.timesim.pipeline_sim import simulate_pipeline


def _uniform(pp, tf=10.0, tb=20.0, v=1):
    return {(s, k, c): (tf if k == "FWD" else tb)
            for s in range(pp) for k in ("FWD", "BWD") for c in range(v)}


def test_plain_1f1b_closed_form_exact():
    """L0③退化还原：均匀 stage + p2p=0 → T = (m+pp−1)·(tf+tb)（逐值精确，零公差）。"""
    r = simulate_pipeline(_uniform(2), pp=2, m=4)
    assert r.t_total_us == pytest.approx((4 + 2 - 1) * 30.0)
    # 均匀退化的 bubble 恒等式：每 stage busy = m·(tf+tb)
    assert all(b == pytest.approx(4 * 30.0) for b in r.per_stage_busy)
    assert r.per_stage_bubble[0] == pytest.approx(150.0 - 120.0)
    assert r.bubble_fraction == pytest.approx((2 - 1) / (4 + 2 - 1))


def test_bubble_shrinks_with_m():
    """L0④性质：m↑ → bubble%↓。"""
    b4 = simulate_pipeline(_uniform(4), pp=4, m=4).bubble_fraction
    b16 = simulate_pipeline(_uniform(4), pp=4, m=16).bubble_fraction
    assert b16 < b4


def test_p2p_latency_stretches_pipeline():
    r0 = simulate_pipeline(_uniform(2), pp=2, m=4, p2p_us=0.0)
    r5 = simulate_pipeline(_uniform(2), pp=2, m=4, p2p_us=5.0)
    assert r5.t_total_us > r0.t_total_us


def test_vpp_reduces_bubble():
    """VPP（v=2，chunk 时长=整段一半）应压 bubble → T 更短（性质，非精确式）。"""
    t1 = simulate_pipeline(_uniform(4, 10.0, 20.0, v=1), pp=4, m=8, v=1).t_total_us
    t2 = simulate_pipeline(_uniform(4, 5.0, 10.0, v=2), pp=4, m=8, v=2).t_total_us
    assert t2 < t1


def test_vpp_event_count_and_conservation():
    r = simulate_pipeline(_uniform(2, 5.0, 10.0, v=2), pp=2, m=4, v=2)
    per_stage = [len([e for e in r.events if e.stage == s]) for s in range(2)]
    assert per_stage == [2 * 4 * 2] * 2                       # 2·m·v 事件/stage
    for s in range(2):
        assert r.per_stage_busy[s] + r.per_stage_bubble[s] == pytest.approx(r.t_total_us)


def test_missing_duration_fail_loud():
    with pytest.raises(KeyError):
        simulate_pipeline({(0, "FWD", 0): 1.0}, pp=2, m=2)


def test_critical_path_ends_at_last_event():
    r = simulate_pipeline(_uniform(2), pp=2, m=4)
    last = max(r.events, key=lambda e: e.end)
    assert r.critical_path[-1] == (last.stage, last.kind, last.mb, last.chunk)
    assert len(r.critical_path) >= 2
```

- [x] **Step 2: 跑确认失败** → **Step 3: 实现**

```python
# cost_eval/timesim/pipeline_sim.py
"""全局调度仿真（M9-L2，spec §6）。

事件 = (stage, kind, mb, chunk)，时长 = durations[(stage, kind, chunk)]（稳态口径：每
(stage, phase, chunk) 一个时长——非均匀 stage 是自然输入，缺 key = KeyError fail-loud）。
每 stage 事件序取自 cost_eval.schedule（契约2 中立模块）：v<=1 → build_1f1b；v>1 →
interleaved_virtual_order（group_size 默认 pp，Megatron model_parallel_config.py:519-520）。

跨 stage 依赖（§6.2；虚拟 stage 号 k = chunk·pp + stage，Megatron round-robin 放置）：
  F(k) 需 F(k−1)：s>0 → (s−1,同c)；s==0 且 c>0 → (pp−1, c−1)；
  B(k) 需 B(k+1)：s<pp−1 → (s+1,同c)；s==pp−1 且 c<v−1 → (0, c+1)；
  B 对自身 F 的依赖由同 stage 调度序 FIFO 隐含（schedule 序保证 F 先于其 B）。
p2p：作**依赖边时延**加在跨 stage 依赖上（不占 device 资源，v1；overlap_p2p/dxdw 等 v1.5
换调度生成器）；同 stage 的 chunk 转移（pp=1 退化）不加 p2p。
推进：逐 stage 顺序取下一事件、依赖满足即调度（stage 同刻只执行一个事件）；一整轮无进展 =
调度死锁 → fail-loud（调度序或依赖规则被打破）。
bubble = 每 stage 在 [0, T] 的空闲积分（仿真结果非公式）；闭式 (pp−1)/(m·v+pp−1) 只作
report 参考值（§6.2）。critical_path：从全局最晚事件沿"决定 start 的约束"回溯。"""
from __future__ import annotations

from dataclasses import dataclass

from ..schedule import build_1f1b, interleaved_virtual_order


@dataclass(frozen=True)
class StageEvent:
    stage: int
    kind: str          # "FWD" | "BWD"
    mb: int
    chunk: int
    start: float
    end: float


@dataclass(frozen=True)
class PipelineResult:
    t_total_us: float
    events: tuple
    per_stage_busy: tuple
    per_stage_bubble: tuple
    bubble_fraction: float
    critical_path: tuple      # ((stage, kind, mb, chunk), ...) 首→尾


def _dep_of(kind: str, s: int, mb: int, c: int, pp: int, v: int):
    if kind == "FWD":
        if s > 0:
            return ("FWD", s - 1, mb, c)
        if c > 0:
            return ("FWD", pp - 1, mb, c - 1)
        return None
    if s < pp - 1:
        return ("BWD", s + 1, mb, c)
    if c < v - 1:
        return ("BWD", 0, mb, c + 1)
    return None


def simulate_pipeline(durations: dict, pp: int, m: int, *, v: int = 1,
                       group_size: int | None = None, p2p_us: float = 0.0) -> PipelineResult:
    gs = group_size or pp
    if v > 1 and gs != pp:
        # VPP v1 仅支持深度优先分组 gs==pp（Megatron 默认）；gs≠pp 广度优先变体调度序与本
        # 反向依赖模型不自洽（pp≥4 实测死锁）→ fail-loud，v1.5。（Task 9 实测发现）
        raise ValueError(
            f"pipeline_sim: VPP(v={v}) 时间仿真 v1 仅支持深度优先分组 group_size==pp；"
            f"gs={gs}≠pp={pp} 的广度优先变体不自洽（v1.5）——fail-loud")
    orders = []
    for s in range(pp):
        if v <= 1:
            evs = [(e.kind, e.mb, 0) for e in build_1f1b(s, pp, m)]
        else:
            # VPP 反向 chunk 反转（Megatron get_model_chunk_id(forward=False)=v−1−chunk）：
            # interleaved_virtual_order 为 mem_timeline 设计（反向 chunk 升序、FIFO），但跨
            # stage 反向依赖要求同 stage 反向 chunk 降序（B(c,s) 需 B(c+1,s)）——时间侧消费时
            # 反转使之自洽（契约3 各自 walk，mem 侧不反转、schedule.py 不动）。v=1 反转为恒等。
            evs = [(k, mb, (v - 1 - c) if k == "BWD" else c)
                   for (k, mb, c) in interleaved_virtual_order(s, pp, m, v, gs)]
        orders.append(evs)

    done: dict[tuple, float] = {}
    blocker: dict[tuple, tuple | None] = {}
    prev_ev: dict[int, tuple | None] = {s: None for s in range(pp)}
    ptr = [0] * pp
    clock = [0.0] * pp
    events: list[StageEvent] = []
    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for s in range(pp):
            while ptr[s] < len(orders[s]):
                kind, mb, c = orders[s][ptr[s]]
                dep = _dep_of(kind, s, mb, c, pp, v)
                if dep is not None:
                    dkey = (dep[1], dep[0], dep[2], dep[3])   # (stage, kind, mb, chunk)
                    if dkey not in done:
                        break
                    lat = p2p_us if dep[1] != s else 0.0
                    ready = max(clock[s], done[dkey] + lat)
                    blk = dkey if done[dkey] + lat >= clock[s] else prev_ev[s]
                else:
                    ready = clock[s]
                    blk = prev_ev[s]
                fin = ready + durations[(s, kind, c)]
                key = (s, kind, mb, c)
                done[key] = fin
                blocker[key] = blk
                events.append(StageEvent(s, kind, mb, c, ready, fin))
                clock[s] = fin
                prev_ev[s] = key
                ptr[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("pipeline_sim: 调度死锁（依赖环/调度序被打破）——fail-loud")

    t_total = max(clock)
    busy = [0.0] * pp
    for e in events:
        busy[e.stage] += e.end - e.start
    bubble = [t_total - b for b in busy]
    bf = sum(bubble) / (pp * t_total) if t_total else 0.0

    last = max(events, key=lambda e: e.end)
    path = []
    k = (last.stage, last.kind, last.mb, last.chunk)
    seen = set()
    while k is not None and k not in seen:
        path.append(k)
        seen.add(k)
        k = blocker.get(k)
    return PipelineResult(t_total, tuple(events), tuple(busy), tuple(bubble), bf,
                          tuple(reversed(path)))
```

- [x] **Step 4: 跑测试 + 全量回归**

Run: `python -m pytest tests/test_timesim_pipeline_sim.py tests/ -q`
`test_plain_1f1b_closed_form_exact` 是零公差精确式（1F1B DES 与闭式在均匀退化下逐值相等，
手推轨迹见 spec §7.1-L0③）；失败按 systematic-debugging 处置，不许加公差凑数。

- [x] **Step 5: Commit**

```bash
git add cost_eval/timesim/pipeline_sim.py tests/test_timesim_pipeline_sim.py
git commit -m "feat(timesim): pipeline_sim 全局 DES——1F1B/VPP 调度消费+p2p 依赖时延+bubble 涌现(T1-9,M9-L2)"
```

## Task 10: StepTimeReport + evaluate_step_time 门面（含步收尾）

**Files:**
- Create: `cost_eval/timesim/report.py`
- Modify: `cost_eval/timesim/frame_comm.py`（追加 fsdp_regather）
- Test: `tests/test_timesim_report.py`

- [x] **Step 1: 写失败测试**

```python
# tests/test_timesim_report.py
"""StepTimeReport 门面（spec §6.3/6.4）：合成小段端到端（不依赖 mindformers），
验证组装管线、步收尾、MFU/HFU、fsdp_regather。"""
import pytest

from cost_eval.specs import OptimizerSpec
from cost_eval.timesim.ir import TimedOp, TimedSegment
from cost_eval.timesim.shard_rules import Degrees
from cost_eval.timesim.machine import synth_hw
from cost_eval.timesim.report import evaluate_step_time


def _layer(i):
    """一层 = 一个带权重的 GEMM + 一个 Norm（module="" 无 TP 语义，合成层）。"""
    mm = TimedOp(op_id="c#0", op_type="MatMul", phase="fwd",
                 in_shapes=((1024, 1, 512), (512, 512)), out_shape=(1024, 1, 512),
                 dtype="bf16", stream="device", src=f"l{i}.py:1")
    nm = TimedOp(op_id="c#1", op_type="Norm", phase="fwd",
                 in_shapes=((1024, 1, 512),), out_shape=(1024, 1, 512),
                 dtype="bf16", stream="device", src=f"l{i}.py:2", deps=("c#0",))
    return TimedSegment(f"layer_{i}.fwd", (mm, nm))


HW = synth_hw()


def test_report_basic_composition():
    r = evaluate_step_time([_layer(0), _layer(1)], Degrees(), HW, pp=1, m=2)
    assert r.t_step_us > 0
    assert r.t_step_us == pytest.approx(
        r.t_pipeline_us + r.t_opt_us + r.t_grad_sync_tail_us + r.fixed_step_us)
    assert r.uncalibrated is True
    assert r.provenance_mix == {"hit": 0.0, "model": 0.0, "theory": 1.0}
    assert 0.0 < r.mfu <= 1.0 and r.hfu == pytest.approx(r.mfu)   # 无重算 HFU==MFU
    assert r.bottleneck_ranking[0][1] >= r.bottleneck_ranking[-1][1]


def test_recompute_scissors():
    """L0④剪刀差：recompute=full → t_step↑、MFU↓、HFU>MFU（spec §6.4）。"""
    base = evaluate_step_time([_layer(0), _layer(1)], Degrees(), HW, pp=1, m=2)
    rc = evaluate_step_time([_layer(0), _layer(1)], Degrees(), HW, pp=1, m=2,
                            recompute="full")
    assert rc.t_step_us > base.t_step_us
    assert rc.mfu < base.mfu
    assert rc.hfu > rc.mfu


def test_pp2_uses_pipeline_and_p2p():
    r1 = evaluate_step_time([_layer(0), _layer(1)], Degrees(pp=2), HW, pp=2, m=4,
                            p2p_bytes=1024 * 512 * 2)
    r0 = evaluate_step_time([_layer(0), _layer(1)], Degrees(pp=2), HW, pp=2, m=4)
    assert r1.t_pipeline_us > r0.t_pipeline_us          # p2p 时延拉长
    assert r1.bubble_fraction > 0
    assert r1.bubble_fraction_closed_form == pytest.approx(1 / 5)


def test_opt_step_and_ddp_tail():
    opt = OptimizerSpec()
    r = evaluate_step_time([_layer(0), _layer(1)], Degrees(), HW, pp=1, m=1,
                           opt=opt, dp_replicate=2)
    assert r.t_opt_us > 0
    assert r.t_grad_sync_tail_us > 0                    # dp_replicate AR 尾（v1 保守暴露）


def test_fsdp_regather_and_grad_rs():
    """dp_shard>1：fwd 段头 AG（inject_fsdp）→ bwd 尾对偶 RS（grad sync 自动涌现）+
    reshard!=never 时 bwd 段头重 gather（交接要点3 裁决）。"""
    from cost_eval.timesim.frame_comm import inject_fsdp, fsdp_regather
    from cost_eval.timesim.bwd_rules import expand_bwd
    fwd = inject_fsdp(_layer(0), dp_shard=4)
    bwd = expand_bwd(fwd)
    bwd2 = fsdp_regather(bwd, fwd, dp_shard=4)
    comms = [o for o in bwd2.ops if o.op_type == "CommOp"]
    assert comms[0].comm.ctype == "all_gather" and comms[0].phase == "bwd"   # 重 gather 段头
    assert comms[-1].comm.ctype == "reduce_scatter"                          # grad RS 段尾
    assert comms[-1].comm.volume_bytes == comms[0].comm.volume_bytes * 4     # AG分片→RS全量
    # reshard="never" 语义由门面控制：不调 fsdp_regather 即无重 gather
    assert all(o.comm.ctype != "all_gather" for o in bwd.ops if o.op_type == "CommOp")
```

- [x] **Step 2: 跑确认失败** → **Step 3: 实现**

`frame_comm.py` 追加（文件头 import 补 `from dataclasses import replace`）：

```python
def fsdp_regather(bwd_seg: TimedSegment, fwd_seg: TimedSegment, dp_shard: int) -> TimedSegment:
    """ZeRO-3 bwd 权重重 gather（spec §3.3c「fwd 预取 + bwd 重 gather，随
    reshard_after_forward」；T0 交接要点3 裁决=注入，门面按 reshard!="never" 调用）。
    克隆 fwd 段的 .fsdp_ag 到 bwd 段头（fwd 无 gather / dp_shard<=1 → 恒等）。
    注：fwd AG 的 bwd 对偶（grad reduce-scatter）由 expand_bwd 自动产出且落 bwd 段尾
    （fwd 段头反转），本函数只补"重新拿回权重"这一条。"""
    if dp_shard <= 1:
        return bwd_seg
    src_ag = next((o for o in fwd_seg.ops
                   if o.op_type == "CommOp" and o.op_id.endswith(".fsdp_ag")), None)
    if src_ag is None:
        return bwd_seg
    ag = replace(src_ag, op_id=f"{bwd_seg.seg_id}.fsdp_ag", phase="bwd")
    return TimedSegment(bwd_seg.seg_id, (ag,) + bwd_seg.ops)
```

```python
# cost_eval/timesim/report.py
"""StepTimeReport（spec §6.4）+ evaluate_step_time 门面：段装配 → 定价 → L1 → L2 → 步收尾。

装配口径（v1）：
  - 输入 = per-layer **fwd** TimedSegment 列表（producer 产物，未注入框架通信——本门面统一做
    cp/ep/fsdp 注入，组合序 cp→ep→fsdp，位置语义见各注入器 docstring）；层数须被 pp 整除
    （v1 均匀切层，layers_per_stage 非均匀=v1.5）；stage 内 chunk 均衡切分复用
    schedule.chunk_layer_ids（契约2 中立模块）。
  - pass = 该 (stage, chunk) 全部层段 concat（spec §5.2 连续 pass）；bwd pass = 各层
    expand_bwd 逆序 concat（+ reshard!="never" 时 fsdp_regather）。
  - 稳态口径：每 (stage, phase, chunk) 仿真一次，pipeline 内 m 个微批复用同一时长（§6.1）。
步收尾（§6.3）：
  1. grad sync：per-layer grad RS 已在 bwd 段内（expand_bwd 对偶自动涌现，可遮盖部分在段内
     DES 自动遮盖）；dp_replicate>1 的 DDP grad AR 无处可挂 → 作 barrier 尾**全暴露**串行加
     （v1 保守，诚实边界）。
  2. optimizer step：带宽类粗口径 = 本 stage 权重字节/2 · (state_bytes_per_param +
     grad_dtype_bytes) ÷ dp_shard ÷ (HBM·η_opt)（分片优化器；opt=None → 0——评估"纯前反向"）。
  3. per-step 固定开销 fixed_step_us：标定常数（T2 锚点反解），默认 0、单列不混 η。
MFU/HFU（§6.4，Megatron 惯例分开报）：分子 = m·Σ_(stage,chunk,phase) OpCost.flops（MFU 不含
recomp、HFU 含）；分母 = t_step · peak(dtype) · pp（每 stage world/pp 个 rank 算各自分片，
约分后剩 pp——推导见测试）。provenance_mix 按 (t_dev+t_comm) 加权聚合（§4.3）。"""
from __future__ import annotations

from dataclasses import dataclass

from ..schedule import chunk_layer_ids
from .ir import CommSpec, tensor_bytes
from .machine import TimeHardware
from .op_cost import CostModel, price_segment
from .pass_builder import concat_segments
from .segment_sim import simulate_segment
from .pipeline_sim import simulate_pipeline
from .bwd_rules import expand_bwd
from .frame_comm import inject_cp, inject_ep, inject_fsdp, fsdp_regather
from .shard_rules import Degrees

_GEMM = ("MatMul", "GroupedMatMul")


@dataclass(frozen=True)
class StepTimeReport:
    t_step_us: float
    t_pipeline_us: float
    t_opt_us: float
    t_grad_sync_tail_us: float
    fixed_step_us: float
    per_stage: tuple              # dict/stage：busy/bubble/host_gap/exposed_comm（×m 聚合）
    bubble_fraction: float
    bubble_fraction_closed_form: float
    critical_path: tuple
    mfu: float
    hfu: float
    provenance_mix: dict
    bottleneck_ranking: tuple     # ((名, us), ...) 降序
    uncalibrated: bool


def _weight_bytes(seg) -> int:
    return sum(tensor_bytes(o.in_shapes[1], o.dtype) for o in seg.ops
               if o.op_type in _GEMM and o.phase == "fwd" and len(o.in_shapes) >= 2)


def evaluate_step_time(layer_segments: list, deg: Degrees, hw: TimeHardware, *,
                        pp: int, m: int, v: int = 1, group_size: int | None = None,
                        recompute: str | None = None, recomp_comm: bool = False,
                        cp_method: str = "colossal", dp_replicate: int = 1,
                        reshard_after_forward: str = "default",
                        opt=None, p2p_bytes: int = 0, fixed_step_us: float = 0.0,
                        causal: bool = True) -> StepTimeReport:
    L = len(layer_segments)
    if pp <= 0 or L % pp:
        raise ValueError(f"report: 层数 {L} 不被 pp={pp} 整除（v1 均匀切层——fail-loud）")
    cm = CostModel(hw, causal=causal)
    per_stage_layers = [layer_segments[s * (L // pp):(s + 1) * (L // pp)]
                        for s in range(pp)]

    durations: dict = {}
    seg_times: dict = {}
    flops_eff = flops_recomp = 0
    prov_us: dict = {"hit": 0.0, "model": 0.0, "theory": 0.0}
    stage_weight_bytes = [0] * pp
    for s in range(pp):
        chunks = chunk_layer_ids(list(range(len(per_stage_layers[s]))), v)
        for c, idxs in enumerate(chunks):
            fwd_layers = []
            for k in idxs:
                seg = per_stage_layers[s][k]
                seg = inject_cp(seg, deg.cp, method=cp_method)
                seg = inject_ep(seg, deg.ep)
                seg = inject_fsdp(seg, deg.dp)
                fwd_layers.append(seg)
            bwd_layers = [expand_bwd(fs, recompute=recompute, recomp_comm=recomp_comm)
                          for fs in reversed(fwd_layers)]
            # recompute="full"+recomp_comm=True 时重算前缀已重放 .fsdp_ag（重 gather），再
            # fsdp_regather 会双 AG 重复计（Task 10 review）——故此组合跳过；其余仍需 regather。
            if (deg.dp > 1 and reshard_after_forward != "never"
                    and not (recompute == "full" and recomp_comm)):
                bwd_layers = [fsdp_regather(bs, fs, deg.dp)
                              for bs, fs in zip(bwd_layers, reversed(fwd_layers))]
            for kind, layers in (("FWD", fwd_layers), ("BWD", bwd_layers)):
                p = concat_segments(f"s{s}.c{c}.{kind.lower()}", layers)
                costs = price_segment(p, cm)
                st = simulate_segment(p, costs)
                durations[(s, kind, c)] = st.duration_us
                seg_times[(s, kind, c)] = st
                for op in p.ops:
                    oc = costs[op.op_id]
                    if op.phase == "recomp":
                        flops_recomp += oc.flops
                    else:
                        flops_eff += oc.flops
                    prov_us[oc.provenance] = prov_us.get(oc.provenance, 0.0) \
                        + oc.t_dev_us + oc.t_comm_us
            stage_weight_bytes[s] += sum(_weight_bytes(f) for f in fwd_layers)

    p2p_us = 0.0
    if p2p_bytes and pp > 1:
        alpha, bw = hw.link("pp")
        p2p_us = alpha + p2p_bytes / bw * 1e6
    pipe = simulate_pipeline(durations, pp, m, v=v, group_size=group_size, p2p_us=p2p_us)

    # —— 步收尾（§6.3）——
    grad_tail = 0.0
    if dp_replicate > 1:
        gb = max(stage_weight_bytes) // 2 * (opt.grad_dtype_bytes if opt else 4)
        grad_tail = cm.comm_time_us(CommSpec("all_reduce", gb, "dp", dp_replicate))
    t_opt = 0.0
    if opt is not None:
        params = max(stage_weight_bytes) // 2                    # bf16 权重 → 参数量
        traffic = params * (opt.state_bytes_per_param + opt.grad_dtype_bytes)
        t_opt = traffic / max(deg.dp, 1) / (hw.hbm_bw * hw.eta["opt"]) * 1e6 \
            + hw.host_us("MatMul", "fwd")                        # host 发射一笔（粗口径）
    t_step = pipe.t_total_us + grad_tail + t_opt + fixed_step_us

    # —— MFU/HFU（模块 docstring 推导）——
    denom = t_step * 1e-6 * hw.peak("bf16") * pp
    mfu = m * flops_eff / denom if denom else 0.0
    hfu = m * (flops_eff + flops_recomp) / denom if denom else 0.0

    total_prov = sum(prov_us.values()) or 1.0
    prov_mix = {k: prov_us.get(k, 0.0) / total_prov for k in ("hit", "model", "theory")}

    per_stage = []
    agg = {"compute": 0.0, "membound": 0.0, "host": 0.0, "bubble": sum(pipe.per_stage_bubble)}
    for s in range(pp):
        hostg = m * sum(seg_times[k].t_host_gap for k in seg_times if k[0] == s)
        exp: dict = {}
        for k, st in seg_times.items():
            if k[0] != s:
                continue
            for ax, us in st.t_exposed_comm.items():
                exp[ax] = exp.get(ax, 0.0) + m * us
            agg["compute"] += m * st.t_compute
            agg["membound"] += m * st.t_membound
        agg["host"] += hostg
        for ax, us in exp.items():
            agg[f"comm_{ax}"] = agg.get(f"comm_{ax}", 0.0) + us
        per_stage.append({"stage": s, "busy_us": pipe.per_stage_busy[s],
                          "bubble_us": pipe.per_stage_bubble[s],
                          "host_gap_us": hostg, "exposed_comm": exp})
    if grad_tail:
        agg["grad_sync_tail"] = grad_tail
    ranking = tuple(sorted(agg.items(), key=lambda kv: kv[1], reverse=True))

    v_ = max(v, 1)
    bf_closed = (pp - 1) / (m * v_ + pp - 1) if (m * v_ + pp - 1) else 0.0
    return StepTimeReport(
        t_step_us=t_step, t_pipeline_us=pipe.t_total_us, t_opt_us=t_opt,
        t_grad_sync_tail_us=grad_tail, fixed_step_us=fixed_step_us,
        per_stage=tuple(per_stage), bubble_fraction=pipe.bubble_fraction,
        bubble_fraction_closed_form=bf_closed, critical_path=pipe.critical_path,
        mfu=mfu, hfu=hfu, provenance_mix=prov_mix, bottleneck_ranking=ranking,
        uncalibrated=not hw.calibrated)
```

- [x] **Step 4: 跑测试 + 全量回归** → **Step 5: Commit**

```bash
git add cost_eval/timesim/report.py cost_eval/timesim/frame_comm.py tests/test_timesim_report.py
git commit -m "feat(timesim): StepTimeReport 门面+步收尾+fsdp_regather(T1-10,spec §6.3/6.4,交接要点3 闭环)"
```

---

# Phase E — L0/L1 验证阶梯 + 收尾

## Task 11: L0 不变量套件 + L1 轻量互证

**Files:**
- Test: `tests/test_timesim_l0.py`

- [x] **Step 1: 写测试（真源 MLP 段驱动全链路；数条应直接 PASS——前序任务已就绪，FAIL 按
  systematic-debugging 处置，不许调公差）**

```python
# tests/test_timesim_l0.py
"""验证阶梯 L0（spec §7.1）：①三态守恒（segment_sim 测试已盖，此处盖 pipeline 级）
③退化还原（单卡串行 roofline 和；1F1B 闭式在 test_timesim_pipeline_sim 已盖）
④性质（tp↑→GEMM t↓通信↑；m↑/recompute 剪刀差在既有测试已盖）
+ L1 轻量互证（MFU 数量级 sanity；Calculon 对标=T2 人工步，见计划收尾说明）。"""
import pytest

from cost_eval.model_spec import DimTable
from cost_eval.timesim.shard_rules import Degrees
from cost_eval.timesim.producer import build_segment
from cost_eval.timesim.machine import synth_hw
from cost_eval.timesim.op_cost import CostModel, price_segment
from cost_eval.timesim.report import evaluate_step_time

DIMS = DimTable(H=1792, F=3072, n_heads=8, n_kv=8, head_dim=224,
                S=4096, B=1, vocab=129280, n_layers=4)


def _layers(mlp_dag, deg, n=2):
    return [build_segment(f"layer_{i}.fwd", mlp_dag, DIMS, deg) for i in range(n)]


def test_l0_serial_roofline_sum_exact(mlp_dag):
    """L0③：全度=1 + host=0 + 无重算 + pp=1,m=1 → t_pipeline == Σ op 时长（串行 roofline 和，
    零公差）。"""
    hw = synth_hw(host_unit_us={("*", "fwd"): 0.0, ("*", "bwd"): 0.0, ("*", "recomp"): 0.0})
    layers = _layers(mlp_dag, Degrees())
    r = evaluate_step_time(layers, Degrees(), hw, pp=1, m=1)
    from cost_eval.timesim.bwd_rules import expand_bwd
    cm = CostModel(hw)
    total = 0.0
    for seg in layers:
        for costs in (price_segment(seg, cm), price_segment(expand_bwd(seg), cm)):
            total += sum(c.t_dev_us + c.t_comm_us for c in costs.values())
    assert r.t_pipeline_us == pytest.approx(total)


def test_l0_pipeline_conservation(mlp_dag):
    """L0①（pipeline 级）：每 stage busy+bubble == T_pipeline。"""
    layers = _layers(mlp_dag, Degrees(), n=4)
    r = evaluate_step_time(layers, Degrees(pp=2), synth_hw(), pp=2, m=4)
    for st in r.per_stage:
        assert st["busy_us"] + st["bubble_us"] == pytest.approx(r.t_pipeline_us)


def test_l0_tp_scissors_gemm_down_comm_up(mlp_dag):
    """L0④：tp↑ → 单卡 GEMM 时间↓、通信时间↑（spec §7.1-L0④）。"""
    hw = synth_hw()
    cm = CostModel(hw)
    seg1 = build_segment("l.fwd", mlp_dag, DIMS, Degrees())
    seg2 = build_segment("l.fwd", mlp_dag, DIMS, Degrees(tp=2, sequence_parallel=True))
    def split(seg):
        costs = price_segment(seg, cm)
        gemm = sum(costs[o.op_id].t_dev_us for o in seg.ops if o.op_type == "MatMul")
        comm = sum(costs[o.op_id].t_comm_us for o in seg.ops)
        return gemm, comm
    g1, c1 = split(seg1)
    g2, c2 = split(seg2)
    assert g2 == pytest.approx(g1 / 2)
    assert c1 == 0.0 and c2 > 0.0


def test_l1_mfu_sanity(mlp_dag):
    """L1 轻量：合成硬件（η=1、host 缺省单价）下 dense MLP 栈的 MFU 落公开数量级
    （0.1~1.0）；未标定绝对值无意义，只做量级 sanity（spec §7.1-L1）。"""
    r = evaluate_step_time(_layers(mlp_dag, Degrees(), n=4), Degrees(), synth_hw(),
                           pp=1, m=2)
    assert 0.1 < r.mfu <= 1.0


def test_l1_report_flags_uncalibrated():
    """诚实边界：DEFAULT_910B（厂商规格）驱动的报告必须带 uncalibrated 标记（§7.2-1）。"""
    from cost_eval.timesim.machine import DEFAULT_910B
    from cost_eval.timesim.ir import TimedOp, TimedSegment
    mm = TimedOp(op_id="c#0", op_type="MatMul", phase="fwd",
                 in_shapes=((64, 1, 64), (64, 64)), out_shape=(64, 1, 64),
                 dtype="bf16", stream="device", src="x.py:1")
    r = evaluate_step_time([TimedSegment("layer_0.fwd", (mm,))], Degrees(),
                           DEFAULT_910B, pp=1, m=1)
    assert r.uncalibrated is True
```

- [x] **Step 2: 跑测试 + 全量回归**（`test_l0_serial_roofline_sum_exact` 零公差——若差在
  epsilon 级查浮点求和顺序，差在数量级查 pass 组装重复/遗漏）

- [x] **Step 3: Commit**

```bash
git add tests/test_timesim_l0.py
git commit -m "test(timesim): L0 阶梯——串行 roofline 还原/pipeline 守恒/tp 剪刀差+L1 MFU sanity(T1-11,spec §7.1)"
```

## Task 12: README 状态 + 交接要点固化 + 全量收尾

**Files:**
- Modify: `README.md`（状态区）
- Modify: 本计划文档（执行后补交接要点终稿）

- [ ] **Step 1: README 状态区追加**

```markdown
- 🔶 P1（时间模型）T1a 仿真核心完成：producer per-tensor 分片状态（MLA/attention 族段打通、
  SequenceParallelLinear）、inject_cp ring 结构化、op_cost（roofline×默认η，provenance=theory）、
  segment_sim（多流 DES+三态归因守恒）、pipeline_sim（1F1B/VPP 全局 DES）、StepTimeReport
  门面（MFU/HFU/瓶颈拆解，恒标未标定）。L0 阶梯+L1 轻量互证全绿。设计见
  specs/2026-07-16-step-time-cost-model-design.md；下一步 T1b（explorer 时间面板）/
  T2（OpTimeLibrary+真机标定）。
```

- [ ] **Step 2: 全量回归最终确认**

Run: `python -m pytest tests/ -q`
Expected: 全绿，通过数 ≥ N_baseline + 本计划新增测试数；12 内存锚点一个不少。

- [ ] **Step 3: 在本计划文档末尾补「T1a → T1b/T2 交接要点」终稿**（执行过程中的实际发现为准，
  至少覆盖：per-tensor 状态的新守卫触发面、MLA 段的 `?` 容忍清单、报告 JSON 形态与 T1b 面板
  的对接字段、DEFAULT_910B 各常数的替换点）。

- [ ] **Step 4: Commit**

```bash
git add README.md plans/2026-07-17-t1-timesim-simulation.md
git commit -m "docs(timesim): T1a 收尾——README 状态+交接要点固化(T1-12)"
```

---

## 完成判据（对照 spec §9-T1）

| spec T1 项 | 任务 | 验收 |
|---|---|---|
| op_cost（默认 η） | Task 5, 6 | OpCost 全字段 + per-ctype 通信系数 + provenance=theory |
| segment_sim | Task 7, 8 | 连续 pass 语义 + 三态归因守恒（L0①） |
| pipeline_sim | Task 9 | 1F1B 闭式零公差还原 + VPP 事件数/守恒 + bubble 涌现 |
| StepTimeReport | Task 10 | T_step 组装恒等式 + MFU/HFU 剪刀差 + 瓶颈排序 + uncalibrated 标记 |
| L0/L1 全绿 | Task 11（+8/9/10 内嵌） | 守恒/退化还原/性质 + MFU sanity |
| （前置）attention 族段 | Task 2, 3, 4 | per-tensor 分片 + SPL + MLA 四断言 + cp ring 结构化 |
| explorer 时间面板 | **T1b（下一计划）** | 消费本计划 StepTimeReport；四面板见 spec §8 |

**明示不在 T1a**（spec 对应位置）：
- explorer 时间面板（§8）→ T1b 计划；
- OpTimeLibrary/microbench runner/真机标定（§4.2、L2-L4）→ T2；Calculon 对标（L1）为 T2 人工步；
- embedding/loss 段成本忠实化（T0 交接要点1 的 Gather 出形/权重 ins/int dtype、loss 种子契约）
  ——本计划 pipeline 只用 transformer 层段，首末 stage 的 embedding/head/loss 时长并入
  per-step 固定开销标定常数（§6.3-3 口径），T2 锚点标定时一并清偿；
- 非均匀 layers_per_stage、select 重算、MoE 段 producer 打通、mc2/swap/zero-bubble（§9 v1.5）；
- **VPP 非默认分组 group_size≠pp（广度优先变体）**：pipeline_sim 时间侧只建模深度优先
  gs=pp（Megatron 默认，DSv3 用之）；gs≠pp 与跨 stage 反向依赖不自洽，fail-loud 留 v1.5
  （Task 9 实测发现，非计划原列——interleaved_virtual_order 为 mem 侧 FIFO 设计，时间侧需
  反向 chunk 反转，深度优先下已验证自洽）；
- 多通信流带宽争抢、跨 pass host run-ahead（§7.2 诚实边界 2/4，v1 有界近似）；
- crosscheck uncovered 记录接入 gpt_segments（T0 交接要点6 后半，opdag 簿记）→ T2 census 对账时一并做。

## T1a → T1b/T2 交接要点（执行后补终稿）

（Task 12 Step 3 填写。）
